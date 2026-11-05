"""Abnahmetests Phase 7: Builder, I/O, Analyse, End-to-End.

Siehe docs/ROADMAP.md, Phase 7. Testsystem für den Builder: zwei DNA-Duplexe + Ionen (1558 Atome,
tests/reference/inputs/dna_duplex_ions.pdb, AMBER OL15 – ASSUMPTION(OQ-03): OL21 nach Entscheidung
steht in den OpenMM-Standardfeldern nicht zur Verfügung).
"""

import numpy as np
import pytest

from biooptomd import builder
from biooptomd import io as bio
from biooptomd.nonbonded import cutoff_from_kappa, debye_kappa, water_permittivity

pytestmark = pytest.mark.gpu

_core = pytest.importorskip("biooptomd._core")


def _rms(f):
    return np.sqrt(np.mean(np.asarray(f) ** 2))


def _duplex_spec():
    import openmm_nonbonded as onb
    from openmm import app

    system, pos, box = onb.load_system()
    pdb = app.PDBFile(str(onb.PDB))
    return system, pdb, pos, box


@pytest.mark.reference
def test_builder_roundtrip_openmm():
    """Energie eines aus ParmEd gebauten Systems = OpenMM-Energie (OpenMM-Referenz mit
    Debye-Hückel + Switching, identischen Ausschlüssen und FP32-gerundeten Parametern wie in
    Phase 4): Gesamtenergie rel. < 1e-4, Kräfte < 1e-4 der größten RMS-Kraft eines Einzelterms.
    Derselbe Vergleich für from_openmm; beide Wege liefern dieselbe Topologie."""
    parmed = pytest.importorskip("parmed")
    import openmm_nonbonded as onb

    system, pdb, pos, box = _duplex_spec()
    structure = parmed.openmm.load_topology(pdb.topology, system, xyz=pdb.positions)
    spec_pm = builder.from_parmed(structure)
    spec_mm = builder.from_openmm(system, pdb.topology, pos)
    for key in ("charges", "sigma", "epsilon"):
        np.testing.assert_allclose(getattr(spec_pm, key), getattr(spec_mm, key), atol=1e-12)
    np.testing.assert_allclose(spec_pm.masses, spec_mm.masses, atol=1e-2)  # ParmEd rundet Massen
    for key in builder.BONDED_KEYS:
        np.testing.assert_allclose(spec_pm.bonded[key], spec_mm.bonded[key], rtol=1e-12, atol=0)
    assert spec_pm.is_hydrogen.sum() == spec_mm.is_hydrogen.sum() > 0

    p32 = pos.astype(np.float32).astype(np.float64)
    n = len(pos)
    for spec in (spec_pm, spec_mm):
        ctx, info = builder.build_context(spec, 1, box, 300.0, 0.15, positions=p32)
        assert info.r_cutoff == pytest.approx(5.0 / info.kappa[0])  # ADR 0006 (> 2 nm)
        ctx.compute_forces()

        def f32(x):
            return float(np.float32(x))

        ref = onb.reference_system(
            system, (*box, onb.BOX_Z), f32(info.kappa[0]), f32(info.eps_r[0]),
            f32(info.r_switch), f32(info.r_cutoff), float32_params=True,
        )  # fmt: skip
        groups = [0, 1, 2, 3, onb.NONBONDED_GROUP]
        e_ref, f_ref = onb.evaluate(ref, p32, groups)
        assert ctx.energies()[0, 1:].sum() == pytest.approx(e_ref, rel=1e-4)
        scale = max(_rms(onb.evaluate(ref, p32, [g])[1]) for g in groups)
        f = ctx.forces()[0, :n, :3].astype(np.float64)
        assert np.abs(f - f_ref).max() < 1e-4 * scale
        assert ctx.status() == 0


@pytest.mark.reference
def test_builder_options():
    """Builder: κ_r aus I und T_r mit ε_r(T_r), gemeinsames r_c; Box unverändert; HMR (Masse
    erhalten) und Constraints; Typen, Anker und Dipol-Probe."""
    system, pdb, pos, box = _duplex_spec()
    spec = builder.from_openmm(system, pdb.topology, pos)
    temps = np.array([290.0, 310.0, 330.0])
    ctx, info = builder.build_context(
        spec, 3, box, temps, 0.05, hmr=True, constraints=True,
        types=builder.assign_types(spec, {2: spec.is_hydrogen}, anchor_atoms=[0]),
    )  # fmt: skip
    np.testing.assert_allclose(info.eps_r, water_permittivity(temps))
    np.testing.assert_allclose(info.kappa, debye_kappa(0.05, temps, water_permittivity(temps)))
    assert (info.r_switch, info.r_cutoff) == pytest.approx(cutoff_from_kappa(info.kappa))
    np.testing.assert_array_equal(ctx.box(), np.tile(np.float32(box), (3, 1)))
    assert max(box) < 2 * info.r_cutoff, "Bildsumme aktiv (L < 2 r_c), Box unverändert"
    assert info.masses.sum() == pytest.approx(spec.masses.sum())
    assert info.masses[spec.is_hydrogen].min() == pytest.approx(3.024)
    assert info.constrained and ctx.has_constraints
    t = ctx.types()[0, : spec.n_atoms, 1]
    assert t[0] == 0 and (t[1:][spec.is_hydrogen[1:]] == 2).all()
    assert (t[1:][~spec.is_hydrogen[1:]] == 1).all()
    plugins = builder.add_anchors(ctx, [0], height_nm=0.22, z_top_nm=0.118)
    assert len(plugins) == 1
    off, atoms, a, b = builder.dipole_probe([np.arange(5, 15)] * 3, [(5, 14)] * 3)
    assert off.tolist() == [0, 10, 20, 30] and a.tolist() == [5] * 3 and b.tolist() == [14] * 3
    ctx.add_dipole_tracker(off, atoms, a, b)
    ctx.compute_forces()
    ctx.observe_now()
    assert ctx.status() == 0 and np.isfinite(ctx.energies()).all()
    with pytest.raises(ValueError):
        builder.assign_types(spec, {0: [1, 2]})


def _small_system(rng, n=96):
    """Zufällige Ketten (Bindungen, Winkel, H-Atome) als SystemSpec – ohne OpenMM."""
    heavy = np.arange(0, n, 3)
    gx, gy, gz = np.meshgrid(np.arange(4), np.arange(4), np.arange(len(heavy) // 16), indexing="ij")
    grid = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()]) * [1.0, 1.0, 0.9] + [0.3, 0.3, 0.8]
    pos = np.zeros((n, 3))
    pos[heavy] = grid + rng.normal(0.0, 0.05, grid.shape)
    bonds = np.array([(h, h + 1) for h in heavy] + [(h, h + 2) for h in heavy], np.int32)
    for h in heavy:  # H-Atome nahe an ihr Schweratom setzen
        pos[h + 1] = pos[h] + [0.1, 0.0, 0.0]
        pos[h + 2] = pos[h] + [0.0, 0.1, 0.0]
    angles = np.array([(h + 1, h, h + 2) for h in heavy], np.int32)
    elements = np.array(["C", "H", "H"] * (n // 3))
    bonded = {
        "bonds": bonds, "bond_k": np.full(len(bonds), 3e5), "bond_r0": np.full(len(bonds), 0.1),
        "angles": angles, "angle_k": np.full(len(angles), 300.0),
        "angle_theta0": np.full(len(angles), np.pi / 2),
        "torsions": np.zeros((0, 4), np.int32), "torsion_k": np.zeros(0),
        "torsion_n": np.zeros(0, np.int32), "torsion_phase": np.zeros(0),
        "pairs14": np.zeros((0, 2), np.int32), "pair_qq": np.zeros(0), "pair_sigma": np.zeros(0),
        "pair_eps": np.zeros(0),
    }  # fmt: skip
    excl = np.concatenate([bonds, np.array([(h + 1, h + 2) for h in heavy], np.int32)])
    return builder.SystemSpec(
        masses=np.where(elements == "H", 1.008, 12.011), charges=rng.uniform(-0.5, 0.5, n),
        sigma=np.where(elements == "H", 0.11, 0.34), epsilon=np.where(elements == "H", 0.06, 0.36),
        bonded=bonded, excluded=excl, elements=elements, positions=pos,
    )  # fmt: skip


def _checkpoint_system(spec, constraints):
    ctx, _ = builder.build_context(
        spec, 2, (4.0, 4.0), [300.0, 340.0], 0.15, hmr=constraints, constraints=constraints,
        types=np.ones(spec.n_atoms, np.int32),
    )  # fmt: skip
    ctx.add_planar_surface(
        np.array([0.0, 2.0], np.float32), np.array([0.0, 0.3], np.float32),
        phi0=3.0, kappa=np.ones(2, np.float32),
    )  # fmt: skip
    ctx.set_integrator(0.004 if constraints else 0.001, 1.0, seed=123)
    return ctx


@pytest.mark.parametrize("constraints", [False, True])
def test_checkpoint_bitexact(tmp_path, constraints):
    """Lauf 2N Schritte == N Schritte + Checkpoint + N Schritte in einem neu aufgebauten Kontext:
    Positionen (inkl. FP64-Korrektur), Geschwindigkeiten, Imagezähler, Kräfte und Energien
    bitidentisch; ebenso mit Constraints/HMR (Δt = 4 fs)."""
    spec = _small_system(np.random.default_rng(7))
    n_steps = 300
    ref = _checkpoint_system(spec, constraints)
    ref.step(2 * n_steps)
    expected = ref.save_state()

    first = _checkpoint_system(spec, constraints)
    first.step(n_steps)
    path = tmp_path / "state.npz"
    bio.save_checkpoint(first, path)
    del first

    second = _checkpoint_system(spec, constraints)
    second.set_integrator(0.5, 0.0, seed=0)  # wird vom Checkpoint ersetzt
    assert bio.load_checkpoint(second, path) == n_steps
    assert second.current_step == n_steps
    second.step(n_steps)
    got = second.save_state()
    for key in ("pos", "pos_corr", "vel", "image"):
        assert np.array_equal(got[key].view(np.uint8), expected[key].view(np.uint8)), key
    assert got["step"] == expected["step"] == 2 * n_steps
    assert got["time"] == expected["time"]
    assert np.array_equal(second.forces_fixed_point(), ref.forces_fixed_point())
    assert np.array_equal(second.energies_fixed_point(), ref.energies_fixed_point())
    assert np.abs(expected["image"]).max() > 0 or constraints, "Test soll Boxwechsel enthalten"
    assert ref.status() == 0


def test_checkpoint_mismatch(tmp_path):
    """Checkpoint in einen Kontext anderer Größe laden → ValueError."""
    spec = _small_system(np.random.default_rng(8))
    ctx = _checkpoint_system(spec, False)
    path = tmp_path / "c.npz"
    bio.save_checkpoint(ctx, path)
    other = _core.SimulationContext(3, spec.n_atoms)
    with pytest.raises(ValueError):
        bio.load_checkpoint(other, path)


def test_trajectory_writers(tmp_path):
    """DCD (je Replica) und HDF5: Rücklesen liefert die Positionen (float32, nm), Box und Zeit;
    unwrap schreibt x + n·L."""
    spec = _small_system(np.random.default_rng(9))
    ctx = _checkpoint_system(spec, False)
    _core.testing.shift_x(ctx, 2.0)  # Hälfte der Atome über die Boxgrenze ⇒ Imagezähler ≠ 0
    ctx.wrap_xy()
    frames, frames_unwrapped = [], []
    with bio.DCDWriter(ctx, str(tmp_path / "t_{replica}.dcd"), interval=50) as dcd, \
         bio.DCDWriter(ctx, str(tmp_path / "u_{replica}.dcd"), interval=50, unwrap=True) as dcu:  # fmt: skip
        for _ in range(4):
            ctx.step(50)
            dcd.write()
            dcu.write()
            x = ctx.positions_precise()[:, : spec.n_atoms, :3]
            frames.append(x)
            img = ctx.images()[:, : spec.n_atoms].astype(np.float64)
            frames_unwrapped.append(
                x + np.concatenate([img, np.zeros_like(img[..., :1])], -1) * 4.0
            )
    frames, frames_unwrapped = np.stack(frames), np.stack(frames_unwrapped)
    for r in range(2):
        pos, box = bio.read_dcd(tmp_path / f"t_{r}.dcd")
        assert pos.shape == (4, spec.n_atoms, 3)
        np.testing.assert_allclose(pos, frames[:, r], rtol=0, atol=1e-6)
        np.testing.assert_allclose(box, 4.0)
        pos_u, _ = bio.read_dcd(tmp_path / f"u_{r}.dcd")
        np.testing.assert_allclose(pos_u, frames_unwrapped[:, r], rtol=1e-6, atol=1e-5)
    assert np.abs(frames_unwrapped - frames).max() > 1.0, "Boxwechsel im Test erwartet"

    h5py = pytest.importorskip("h5py")
    with bio.HDF5Writer(ctx, tmp_path / "t.h5") as h5:
        for _ in range(3):
            ctx.step(10)
            h5.write()
    with h5py.File(tmp_path / "t.h5") as f:
        assert f["positions"].shape == (3, 2, spec.n_atoms, 3)
        np.testing.assert_array_equal(f["step"][:], [210, 220, 230])
        np.testing.assert_allclose(f["time"][:], np.array([210, 220, 230]) * 0.001)
        np.testing.assert_allclose(
            f["positions"][-1], ctx.positions_precise()[:, : spec.n_atoms, :3], atol=1e-6
        )


@pytest.mark.slow
@pytest.mark.reference
def test_end_to_end_biosensor():
    """Oberflächenfixiertes Aptamer (examples/aptamer_on_graphene), 8 Replicas, 1 ns:
    Observer liefern endliche, normierte Zeitreihen (|μ̂| = 1, 0 ≤ κ² ≤ 4, 0 ≤ E ≤ 1, d > 0);
    n(z) → n_w für z ≫ Molekülhöhe; Anker bleibt an der Oberfläche; keine Statusbits."""
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "examples" / "aptamer_on_graphene"
    spec_ = importlib.util.spec_from_file_location(
        "aptamer_on_graphene", path / "aptamer_on_graphene.py"
    )
    ex = importlib.util.module_from_spec(spec_)
    spec_.loader.exec_module(ex)

    ctx, obs, meta = ex.build(n_replicas=8, capacity=1000, stride=250)
    ctx.step(250_000)  # 1 ns bei 4 fs
    assert ctx.status() == 0
    assert ctx.time == pytest.approx(1000.0)

    dip = obs["dipole"].result()["frames"]
    assert dip.shape == (8, 1000, 7) and np.isfinite(dip).all()
    np.testing.assert_allclose(np.linalg.norm(dip[..., 3:6], axis=-1), 1.0, atol=1e-5)
    assert (dip[..., 6] > 0).all(), "Fluorophor-Sonde unter der Oberfläche"
    fret = obs["fret"].result()
    for key in ("r_da", "kappa2", "efficiency"):
        assert fret[key].shape == (8, 1000) and np.isfinite(fret[key]).all()
    assert (fret["r_da"] > 0).all()
    assert (fret["kappa2"] >= 0).all() and (fret["kappa2"] <= 4 + 1e-5).all()
    assert (fret["efficiency"] >= 0).all() and (fret["efficiency"] <= 1).all()
    res = ex.summarize(obs, meta)
    assert np.isfinite(res["E_dynamic"]).all() and np.isfinite(res["E_static"]).all()
    assert (res["E_static"] <= res["E_dynamic"] + 1e-12).all()  # Jensen

    # n(z): oberhalb der maximalen Ausdehnung (12 nt × ≤ 0.7 nm + Anker < 9 nm) kein Atom ⇒ n = n_w.
    nz = obs["nz"].result()
    assert nz["n_frames"][0] == 1000 and np.isfinite(nz["n"]).all()
    top = nz["z"] > 9.0
    np.testing.assert_array_equal(nz["n"][:, top], np.float32(1.333))
    assert (nz["n"][:, nz["z"] < 3.0] > 1.333).any(axis=1).all(), "Molekül im Profil sichtbar"

    # Anker: z am Ende nahe der Sollhöhe (k_z = 10⁴ ⇒ σ_z ≈ √(k_BT/k) ≈ 0.016 nm; 6σ ≈ 0.1 nm).
    z_anchor = ctx.positions_precise()[:, meta["anchor"], 2]
    assert np.abs(z_anchor - meta["anchor_z"]).max() < 0.1
    e = obs["energy"].result()["energies"]
    assert np.isfinite(e).all()
