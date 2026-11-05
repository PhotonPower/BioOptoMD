"""Abnahmetests Phase 9.3: GB-neck2 (GBn2, AMBER igb = 8; ADR 0012, ROADMAP Phase 9.3).

Testsystem wie test_phase9: DNA-Duplex (Stränge A, B), AMBER OL15 + implicit/gbn2.xml (mbondi3-Radien,
Nukleinsäure-Parameter für Skalierung und α, β, γ).
"""

import gb_ref
import numpy as np
import pytest
from test_phase9 import (
    KAPPA,
    _f32,
    _gb_context,
    _openmm_nonbonded_gb,
    _pairs14,
    _rms,
)
from test_phase9 import _duplex as _duplex_any

from biooptomd import builder
from biooptomd.nonbonded import lj_type_table, nonbonded_topology, padded_atom_count

GBN2 = "implicit/gbn2.xml"


def _duplex(kappa=KAPPA):
    return _duplex_any(kappa, gb_xml=GBN2)


def _model(spec, f32=False):
    cast = _f32 if f32 else (lambda x: np.asarray(x, np.float64))
    return gb_ref.gbn2_model(
        cast(spec.gb_abg), spec.gb_neck_index, cast(spec.gb_neck_m0), cast(spec.gb_neck_d0)
    )


@pytest.mark.reference
def test_gbn2_reference_vs_openmm():
    """NumPy-Referenz (GBn2) vs. OpenMM-CustomGBForce (gbn2.xml, κ = 1/nm, ACE) + NonbondedForce:
    Energie rel. < 1e-6 (OpenMM-Konstante C = 138.935485 ⇒ 2·10⁻⁷), Kräfte < 1e-5 RMS. Der
    Builder übernimmt Parameter und Neck-Tabellen aus OpenMM; die Tabellen sind asymmetrisch, und
    die vertauschte Ausrichtung ergäbe einen deutlichen Fehler (Orientierung geprüft)."""
    system, top, pos = _duplex()
    spec = builder.from_openmm(system, top, pos)
    assert spec.gb_model == "gbn2"
    assert spec.gb_neck_m0.shape == spec.gb_neck_d0.shape == (spec.gb_neck_index.max() + 1,) * 2
    assert np.abs(spec.gb_neck_m0 - spec.gb_neck_m0.T).max() > 0.01, "Tabellen asymmetrisch"
    types, table = lj_type_table(spec.sigma, spec.epsilon)
    args = (
        pos,
        spec.charges,
        spec.gb_radius,
        spec.gb_scale,
        types,
        table,
        spec.excluded,
        KAPPA,
        78.5,
    )
    e, f, born, _ = gb_ref.gb_nonbonded(*args, model=_model(spec))
    e14, f14 = _pairs14(spec, pos)
    e_omm, f_omm = _openmm_nonbonded_gb(system, pos)
    assert e + e14 == pytest.approx(e_omm, rel=1e-6)
    assert np.abs(f + f14 - f_omm).max() < 1e-5 * _rms(f_omm)
    transposed = gb_ref.gbn2_model(
        spec.gb_abg, spec.gb_neck_index, spec.gb_neck_m0.T, spec.gb_neck_d0.T
    )
    e_t, _, _, _ = gb_ref.gb_nonbonded(*args, model=transposed)
    assert abs(e_t + e14 - e_omm) > 1e-3 * abs(e_omm)
    assert (born > 0).all()


@pytest.mark.parametrize(
    "box, n_max, r_s, r_c",
    [(None, 0, None, None), (None, 0, 1.0, 1.2), ((2.5, 2.5), 2, 1.6, 1.8)],
)
def test_gbn2_reference_forces_fd(box, n_max, r_s, r_c):
    """GBn2-Referenzkräfte (inkl. Neck-Term und dessen Ableitung) vs. finite Differenzen,
    rel. < 1e-5 der größten Kraft; asymmetrische Zufallstabellen, α, β, γ je Atom."""
    from test_phase4 import _random_system

    rng = np.random.default_rng(31)
    n = 30
    pos, sig, eps, q, excl = _random_system(rng, n, 2.5, 10)
    types, table = lj_type_table(sig, eps)
    rho = rng.uniform(0.12, 0.185, n)
    scale = rng.uniform(0.2, 1.6, n)
    u = 4
    model = gb_ref.gbn2_model(
        np.column_stack([rng.uniform(0.3, 0.9, n), rng.uniform(0.2, 0.9, n), rng.uniform(0.1, 0.5, n)]),
        rng.integers(0, u, n), rng.uniform(0.1, 0.6, (u, u)), rng.uniform(0.25, 0.4, (u, u)),
    )  # fmt: skip
    # Kein Paar am Neck-Cutoff (Stufe, dort ist E nicht differenzierbar)
    d = np.linalg.norm(pos[:, None] - pos[None], axis=-1)
    cut = rho[:, None] + rho[None] + gb_ref.NECK_CUT
    assert np.abs(d - cut)[np.triu_indices(n, 1)].min() > 1e-4

    def run(p):
        return gb_ref.gb_nonbonded(
            p, q, rho, scale, types, table, excl, 0.8, 78.5, box=box, n_max=n_max, r_s=r_s,
            r_c=r_c, model=model,
        )  # fmt: skip

    _, f, _, _ = run(pos)
    h = 1e-6
    f_fd = np.zeros_like(pos)
    for a in range(n):
        for k in range(3):
            p = pos.copy()
            p[a, k] += h
            e_plus = run(p)[0]
            p[a, k] -= 2 * h
            f_fd[a, k] = -(e_plus - run(p)[0]) / (2 * h)
    assert np.abs(f - f_fd).max() < 1e-5 * np.abs(f).max()


@pytest.fixture
def core():
    core = pytest.importorskip("biooptomd._core")
    if not core.cuda_device_count():
        pytest.skip("kein CUDA-Gerät")
    return core


def _gbn2_kwargs(spec):
    return {
        "offset": 0.0195141, "alpha_beta_gamma": [spec.gb_abg],
        "neck_index": [spec.gb_neck_index], "neck_m0": [spec.gb_neck_m0],
        "neck_d0": [spec.gb_neck_d0],
    }  # fmt: skip


@pytest.mark.gpu
@pytest.mark.reference
def test_gbn2_gpu_vs_numpy(core):
    """Isolierter Duplex, r_c = 12 nm: GPU-Born-Radien rel. < 1e-5, Nonbonded-Energie rel. < 1e-5,
    Kräfte < 1e-4 RMS gegen die NumPy-Referenz (GBn2 mit Neck)."""
    system, top, pos = _duplex()
    pos = _f32(pos - pos.mean(axis=0) + [15.0, 15.0, 10.0])
    spec = builder.from_openmm(system, top, pos)
    n = spec.n_atoms
    types, topo = nonbonded_topology(padded_atom_count(n), spec.sigma, spec.epsilon, spec.excluded)
    ctx = _gb_context(
        core, 1, pos, spec.charges, types, topo, spec.gb_radius, spec.gb_scale, (30.0, 30.0),
        KAPPA, 78.5, 11.8, 12.0, **_gbn2_kwargs(spec),
    )  # fmt: skip
    e_ref, f_ref, born, _ = gb_ref.gb_nonbonded(
        pos, _f32(spec.charges), _f32(spec.gb_radius), _f32(spec.gb_scale), types,
        topo["lj_table"].astype(np.float64), spec.excluded, float(np.float32(KAPPA)), 78.5,
        model=_model(spec, f32=True),
    )  # fmt: skip
    np.testing.assert_allclose(ctx.nonbonded_born_radii()[0, :n], born, rtol=1e-5)
    assert ctx.energies()[0, 5] == pytest.approx(e_ref, rel=1e-5)
    f = ctx.forces()[0, :n, :3].astype(np.float64)
    assert np.abs(f - f_ref).max() < 1e-4 * _rms(f_ref)
    assert ctx.status() == 0


@pytest.mark.gpu
@pytest.mark.parametrize("box, r_c", [((6.0, 6.0), 1.8), ((2.5, 3.0), 1.8)])
def test_gbn2_cutoff_and_images_vs_numpy(core, box, r_c):
    """GBn2 mit Neck im Minimum-Image-Pfad (Vorauswahl) und mit Bildsumme: GPU vs. NumPy-
    Gittersumme, Born-Radien rel. < 1e-5, Kräfte < 1e-4 RMS, Energie rel. < 1e-5."""
    from test_phase4 import _random_system

    rng = np.random.default_rng(32)
    n, u = 70, 4
    pos, sig, eps, q, excl = _random_system(rng, n, min(box), 30)
    pos[:, 0] *= box[0] / min(box)
    pos[:, 1] *= box[1] / min(box)
    rho, scale = rng.uniform(0.12, 0.185, n), rng.uniform(0.2, 1.6, n)
    abg = np.column_stack(
        [rng.uniform(0.3, 0.9, n), rng.uniform(0.2, 0.9, n), rng.uniform(0.1, 0.5, n)]
    )
    idx = rng.integers(0, u, n).astype(np.int32)
    m0, d0 = rng.uniform(0.1, 0.6, (u, u)), rng.uniform(0.25, 0.4, (u, u))
    types, topo = nonbonded_topology(padded_atom_count(n), sig, eps, excl)
    ctx = _gb_context(
        core, 1, pos, q, types, topo, rho, scale, box, 0.8, 78.5, r_c - 0.2, r_c,
        offset=0.0195141, alpha_beta_gamma=[abg], neck_index=[idx], neck_m0=[m0], neck_d0=[d0],
    )  # fmt: skip
    e_ref, f_ref, born, _ = gb_ref.gb_nonbonded(
        _f32(pos), _f32(q), _f32(rho), _f32(scale), types, topo["lj_table"].astype(np.float64),
        excl, float(np.float32(0.8)), 78.5, box=box, n_max=int(np.ceil(r_c / min(box))) + 1,
        r_s=float(np.float32(r_c - 0.2)), r_c=float(np.float32(r_c)),
        model=gb_ref.gbn2_model(_f32(abg), idx, _f32(m0), _f32(d0)),
    )  # fmt: skip
    np.testing.assert_allclose(ctx.nonbonded_born_radii()[0, :n], born, rtol=1e-5)
    assert ctx.energies()[0, 5] == pytest.approx(e_ref, rel=1e-5)
    f = ctx.forces()[0, :n, :3].astype(np.float64)
    assert np.abs(f - f_ref).max() < 1e-4 * _rms(f_ref)
    assert ctx.status() == 0


@pytest.mark.gpu
@pytest.mark.reference
def test_gbn2_vs_openmm(core):
    """Builder-Weg (solvent="gb-gbn2"), alle Terme vs. OpenMM (gbn2.xml, κ = 1/nm, ACE, NoCutoff,
    FP64): Energie rel. < 1e-4, Kräfte < 1e-4 der größten RMS-Kraft eines Terms. Falsches Modell
    im Builder wird abgelehnt."""
    import openmm
    from openmm import unit

    system, top, pos = _duplex()
    pos = _f32(pos - pos.mean(axis=0) + [15.0, 15.0, 10.0])
    spec = builder.from_openmm(system, top, pos)
    with pytest.raises(ValueError):
        builder.build_context(spec, 1, (30.0, 30.0), 300.0, 0.15, solvent="gb-obc2")
    ctx, _ = builder.build_context(
        spec, 1, (30.0, 30.0), 300.0, 0.15, solvent="gb-gbn2", eps_r=78.5, kappa=KAPPA,
        r_cutoff=12.0,
    )  # fmt: skip
    ctx.compute_forces()
    for g, force in enumerate(system.getForces()):
        force.setForceGroup(g)
    octx = openmm.Context(
        system, openmm.VerletIntegrator(0.001), openmm.Platform.getPlatformByName("Reference")
    )
    octx.setPositions(pos * unit.nanometer)
    kj_nm = unit.kilojoule_per_mole / unit.nanometer
    st = octx.getState(getEnergy=True, getForces=True)
    e_ref = st.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    f_ref = np.array(st.getForces(asNumpy=True).value_in_unit(kj_nm))
    scale = max(
        _rms(octx.getState(getForces=True, groups={g}).getForces(asNumpy=True).value_in_unit(kj_nm))
        for g in range(system.getNumForces())
    )
    assert ctx.energies()[0, 1:].sum() == pytest.approx(e_ref, rel=1e-4)
    f = ctx.forces()[0, : spec.n_atoms, :3].astype(np.float64)
    assert np.abs(f - f_ref).max() < 1e-4 * scale
    assert ctx.status() == 0


@pytest.mark.gpu
@pytest.mark.slow
def test_gbn2_nve(core):
    """NVE 10 ps, Δt = 0.5 fs ohne Constraints, GBn2 bei 1 M (r_c = 2 nm < Molekül), Box 8 × 8 nm:
    Drift < 1e-4 kJ/mol/ps/Atom, Fluktuation < 1e-3·⟨E_kin⟩ (wie test_gb_nve). Die Stufe des
    Neck-Terms bei ρ_i + ρ_j + 0.68 nm (AMBER-Modell) bleibt darunter (gemessen ≈ 5e-5)."""
    system, top, pos = _duplex()
    pos = pos - pos.mean(axis=0) + [4.0, 4.0, 4.0]
    spec = builder.from_openmm(system, top, pos)
    n_rep, n = 2, spec.n_atoms
    ctx, info = builder.build_context(spec, n_rep, (8.0, 8.0), 300.0, 1.0, solvent="gb-gbn2")
    assert info.r_cutoff == 2.0
    rng = np.random.default_rng(93)
    m = spec.masses
    v = rng.normal(0.0, 1.0, (n_rep, n, 3)) * np.sqrt(core.units.KB * 300.0 / m)[None, :, None]
    v -= (v * m[None, :, None]).sum(axis=1, keepdims=True) / m.sum()
    ctx.set_velocities(v)
    ctx.set_integrator(0.0005, 0.0, seed=0)
    tracker = ctx.add_energy_tracker(stride=20, capacity=1000)
    ctx.step(20_000)
    res = tracker.result()
    en = res["energies"]
    total = en[:, :, 0] + en[:, :, 1:].sum(axis=2)
    t_ps = res["steps"] * 0.0005
    drift = np.array([np.polyfit(t_ps, total[r], 1)[0] for r in range(n_rep)]) / n
    fluct = total.std(axis=1) / en[:, :, 0].mean(axis=1)
    assert np.all(np.abs(drift) < 1e-4), f"Drift {drift}"
    assert np.all(fluct < 1e-3), f"Fluktuation {fluct}"
    assert ctx.status() == 0
