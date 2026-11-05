"""Abnahmetests Phase 3: kovalente Kräfte (Bonds, Angles, Torsionen, 1-4-Paare).

Siehe docs/ROADMAP.md, Phase 3. Testmolekül: DNA-Trinukleotid 5'-CGC-3' (AMBER OL15), Topologie in
tests/reference/inputs/dna_trinucleotide_topology.npz (erzeugt von generate_all.py).
"""

from pathlib import Path

import bonded_ref as ref
import numpy as np
import pytest

INPUTS = Path(__file__).resolve().parents[1] / "reference" / "inputs"
TOPO_KEYS = (
    "bonds", "bond_k", "bond_r0", "angles", "angle_k", "angle_theta0", "torsions", "torsion_k",
    "torsion_n", "torsion_phase", "pairs14", "pair_qq", "pair_sigma", "pair_eps",
)  # fmt: skip
TERMS = ("bond", "angle", "torsion", "pair14")
TERM_KEYS = {
    "bond": ("bonds", "bond_k", "bond_r0"),
    "angle": ("angles", "angle_k", "angle_theta0"),
    "torsion": ("torsions", "torsion_k", "torsion_n", "torsion_phase"),
    "pair14": ("pairs14", "pair_qq", "pair_sigma", "pair_eps"),
}
KAPPA, EPS_R = 1.0, 78.5


def _load():
    data = np.load(INPUTS / "dna_trinucleotide_topology.npz")
    topo = {k: data[k] for k in TOPO_KEYS}
    return topo, data["positions"], data["masses"]


def _as_float32(topo):
    """Parameter so, wie die GPU sie speichert (float32), damit der Vergleich nur den Kernel prüft."""
    return {
        k: (v.astype(np.float32).astype(np.float64) if v.dtype.kind == "f" else v)
        for k, v in topo.items()
    }


def _only(topo, term):
    """Topologie, die nur einen Term-Typ enthält."""
    return {k: topo[k] for k in TERM_KEYS[term]}


def _rms(f):
    return np.sqrt(np.mean(np.asarray(f) ** 2))


# --- CPU: Referenz gegen finite Differenzen (AGENTS.md §2.3) ---------------------------------


def test_reference_finite_differences():
    """NumPy-Referenz jedes Terms: F = −∇E (zentrale Differenzen, FP64), auch über Boxgrenzen."""
    topo, pos, _ = _load()
    box = np.array([3.0, 3.0])
    pos = pos - pos.mean(axis=0) + [0.0, 1.5, 1.5]  # Molekül liegt über der Boxgrenze x = 0
    pos[:, :2] %= box  # gewrappt ⇒ Minimum Image entscheidend
    parts = ref.total(pos, topo, KAPPA, EPS_R, box)
    for term in TERMS:
        f_fd = ref.finite_difference_forces(
            lambda p, name=term: ref.term(name, p, topo, KAPPA, EPS_R, box)[0], pos
        )
        f = parts[term][1]
        assert np.abs(f - f_fd).max() < 1e-5 * np.abs(f).max(), term
        assert np.abs(f.sum(axis=0)).max() < 1e-9 * np.abs(f).max(), f"{term}: Σ F ≠ 0"


@pytest.mark.reference
def test_reference_matches_openmm_conventions():
    """NumPy-Referenz = OpenMM (FP64) je Term – sichert Konventionen (Torsionsvorzeichen, ½-Faktoren)."""
    import openmm_bonded as om

    system, pos = om.load_system()
    topo = om.extract_topology(system)
    out = om.evaluate(om.reference_system(system, KAPPA, EPS_R), pos)
    mine = ref.total(pos, topo, KAPPA, EPS_R)
    for term in TERMS:
        assert mine[term][0] == pytest.approx(out[term][0], rel=1e-10), term
        assert np.abs(mine[term][1] - out[term][1]).max() < 1e-10 * _rms(out[term][1]), term


# --- GPU --------------------------------------------------------------------------------------


@pytest.fixture
def core():
    if not pytest.importorskip("biooptomd._core").cuda_device_count():
        pytest.skip("kein CUDA-Gerät")
    from biooptomd import _core

    return _core


def _context(core, n_rep, pos, masses, box=10.0):
    n = pos.shape[0]
    ctx = core.SimulationContext(n_rep, n)
    ctx.set_box(np.full((n_rep, 2), box, np.float32))
    ctx.set_masses(np.broadcast_to(masses, (n_rep, n)).astype(np.float32))
    ctx.set_positions(np.broadcast_to(pos, (n_rep, n, 3)).astype(np.float32))
    return ctx


def _gpu_eval(
    core, topologies, replica_topology, positions, masses, box=10.0, kappa=None, eps_r=None
):
    n_rep = len(replica_topology)
    kappa = np.full(n_rep, KAPPA) if kappa is None else np.asarray(kappa)
    eps_r = np.full(n_rep, EPS_R) if eps_r is None else np.asarray(eps_r)
    ctx = _context(core, n_rep, positions[0], masses, box)
    ctx.set_positions(np.asarray(positions, np.float32))
    ctx.add_bonded(
        topologies,
        np.asarray(replica_topology, np.int32),
        kappa.astype(np.float32),
        eps_r.astype(np.float32),
    )
    ctx.compute_forces()
    n = positions.shape[1]
    return ctx, ctx.forces()[:, :n, :3].astype(np.float64), ctx.energies()


ENERGY_COL = {"bond": 1, "angle": 2, "torsion": 3, "pair14": 4}  # Spalten in energies()


@pytest.mark.gpu
def test_bonded_vs_numpy(core):
    """Jeder Term einzeln gegen NumPy: rel. Kraftfehler < 1e-5 (bezogen auf die RMS-Kraft);
    zwei Replicas, eine davon über die Boxgrenze gewrappt (Minimum Image)."""
    topo, pos, masses = _load()
    rng = np.random.default_rng(30)
    box = 4.0
    base = pos - pos.mean(axis=0) + [box / 2, box / 2, 2.0]
    shifted = base + [box / 2, 0.3, 0.0]
    shifted[:, :2] %= box
    positions = np.stack([base, shifted]) + rng.normal(0, 0.005, (2,) + pos.shape)  # verzerrt
    positions = positions.astype(np.float32).astype(np.float64)
    for term in TERMS:
        t = _as_float32(_only(topo, term))
        _, f_gpu, e_gpu = _gpu_eval(core, [t], [0, 0], positions, masses, box)
        for r in range(2):
            e_ref, f_ref = ref.term(term, positions[r], t, KAPPA, EPS_R, [box, box])
            err = np.abs(f_gpu[r] - f_ref).max() / _rms(f_ref)
            assert err < 1e-5, f"{term}, Replica {r}: rel. Fehler {err:.2e}"
            assert e_gpu[r, ENERGY_COL[term]] == pytest.approx(e_ref, rel=1e-5), term


@pytest.mark.gpu
@pytest.mark.reference
def test_bonded_vs_openmm(core):
    """Gesamtkraft aller Atome gegen OpenMM (Reference, FP64): max. Abweichung < 1e-4 · RMS-Kraft."""
    import openmm_bonded as om

    system, pos = om.load_system()
    topo = om.extract_topology(system)
    # Translationsinvarianz: verschobene, auf float32 gerundete Positionen in beiden Engines.
    p = (pos - pos.mean(axis=0) + [5.0, 5.0, 3.0]).astype(np.float32).astype(np.float64)
    _, f_gpu, e_gpu = _gpu_eval(core, [topo], [0], p[None], topo["masses"])
    e_omm, f_omm = om.evaluate(om.reference_system(system, KAPPA, EPS_R), p)["total"]
    assert np.abs(f_gpu[0] - f_omm).max() < 1e-4 * _rms(f_omm)
    assert e_gpu[0, 1:5].sum() == pytest.approx(e_omm, rel=1e-4)


@pytest.mark.gpu
def test_degenerate_geometries(core):
    """Kollineare Winkel, Torsion bei 0 und π, kollineare Torsionsatome, zusammenfallende Atome:
    kein NaN/Inf, keine Statusbits; endliche Fälle stimmen mit der Referenz überein."""
    pos = np.array(
        [
            [1.0, 1.0, 1.0], [1.1, 1.0, 1.0], [1.2, 1.0, 1.0],  # 0-1-2 kollinear (θ = π)
            [1.0, 2.0, 1.0], [1.1, 2.0, 1.0], [1.1, 2.1, 1.0], [1.0, 2.1, 1.0],  # 3-4-5-6 cis, φ = 0
            [1.0, 3.0, 1.0], [1.1, 3.0, 1.0], [1.1, 3.1, 1.0], [1.2, 3.1, 1.0],  # 7-8-9-10 trans, φ = π
            [2.0, 1.0, 1.0], [2.0, 1.0, 1.0],  # 11, 12 zusammenfallend
        ]
    )  # fmt: skip
    topo = {
        "bonds": np.array([[0, 1], [11, 12]]), "bond_k": np.array([300.0, 300.0]),
        "bond_r0": np.array([0.1, 0.1]),
        "angles": np.array([[0, 1, 2], [3, 4, 5]]), "angle_k": np.array([200.0, 200.0]),
        "angle_theta0": np.array([2.0, 1.9]),
        "torsions": np.array([[3, 4, 5, 6], [7, 8, 9, 10], [0, 1, 2, 3]]),
        "torsion_k": np.array([5.0, 5.0, 5.0]), "torsion_n": np.array([1, 2, 3]),
        "torsion_phase": np.array([0.5, 0.3, 0.0]),
        "pairs14": np.array([[11, 12], [0, 6]]), "pair_qq": np.array([0.2, -0.1]),
        "pair_sigma": np.array([0.3, 0.3]), "pair_eps": np.array([0.5, 0.5]),
    }  # fmt: skip
    pos32 = pos.astype(np.float32).astype(np.float64)
    ctx, f_gpu, e_gpu = _gpu_eval(core, [topo], [0], pos32[None], np.full(len(pos), 12.0))
    assert np.isfinite(f_gpu).all() and np.isfinite(e_gpu).all()
    assert ctx.status() == 0
    parts = ref.total(pos32, topo, KAPPA, EPS_R)
    f_ref = sum(parts[t][1] for t in TERMS)
    np.testing.assert_allclose(f_gpu[0], f_ref, atol=1e-3 * _rms(f_ref))
    np.testing.assert_allclose(e_gpu[0, 1:5], [parts[t][0] for t in TERMS], rtol=1e-5, atol=1e-5)
    # Grenzfall-Konvention: kollinearer Winkel und kollineare Torsion tragen keine Kraft bei.
    assert parts["angle"][1][2].tolist() == [0.0, 0.0, 0.0]


@pytest.mark.gpu
def test_nve_energy_conservation(core):
    """γ = 0, Δt = 0.5 fs, 200 000 Schritte (100 ps): Drift (Regressionssteigung) < 1e-4
    kJ/mol/ps/Atom, Fluktuation < 1e-3·⟨E_kin⟩. Molekül bei realistischen Koordinaten (~5 nm),
    wo FP32-Zustand die Grenze verfehlen würde (ADR 0009). Protokoll: Entscheidung des
    Projektinhabers 2026-11-03 (5000 Schritte lösen 1e-4 statistisch nicht auf)."""
    topo, pos, masses = _load()
    n = pos.shape[0]
    kb = core.units.KB
    rng = np.random.default_rng(31)
    ctx = _context(core, 4, pos - pos.mean(axis=0) + [5.0, 5.0, 3.0], masses)
    v = rng.normal(0.0, 1.0, (4, n, 3)) * np.sqrt(kb * 300.0 / masses)[None, :, None]
    v -= (v * masses[None, :, None]).sum(axis=1, keepdims=True) / masses.sum()
    ctx.set_velocities(v.astype(np.float32))
    ctx.add_bonded(
        [topo], np.zeros(4, np.int32), np.full(4, KAPPA, np.float32), np.full(4, EPS_R, np.float32)
    )
    ctx.set_integrator(0.0005, 0.0, seed=0)
    tracker = ctx.add_energy_tracker(stride=100, capacity=2000)
    ctx.step(200_000)
    e = tracker.result()["energies"]
    total = e[:, :, 0] + e[:, :, 1:].sum(axis=2)
    t_ps = tracker.result()["steps"] * 0.0005
    drift = np.array([np.polyfit(t_ps, total[r], 1)[0] for r in range(4)]) / n
    fluct = total.std(axis=1) / e[:, :, 0].mean(axis=1)
    assert np.all(np.abs(drift) < 1e-4), f"Drift {drift} kJ/mol/ps/Atom"
    assert np.all(fluct < 1e-3), f"Fluktuation {fluct}"
    assert ctx.status() == 0


@pytest.mark.gpu
def test_replica_topologies_and_padding(core):
    """Verschiedene Topologien je Replica (ADR 0005); gleiche Topologie ⇒ identische Kräfte;
    Padding-Atome bleiben kraftfrei."""
    topo, pos, masses = _load()
    p = pos - pos.mean(axis=0) + [5.0, 5.0, 3.0]
    bonds_only = _only(topo, "bond")
    positions = np.stack([p, p, p])
    ctx, f, e = _gpu_eval(core, [topo, bonds_only], [0, 1, 0], positions, masses)
    assert not ctx.forces()[:, pos.shape[0] :].any(), "Padding-Atome erhalten Kraft"
    np.testing.assert_array_equal(f[0], f[2])
    assert e[1, 2:5].tolist() == [0.0, 0.0, 0.0]
    np.testing.assert_allclose(e[1, 1], e[0, 1], rtol=1e-6)


@pytest.mark.gpu
def test_pair14_screening_per_replica(core):
    """κ und ε_r der 1-4-Debye-Hückel-Abschirmung gelten pro Replica (z. B. Temperaturleiter)."""
    topo, pos, masses = _load()
    t = _as_float32(_only(topo, "pair14"))
    p = (pos - pos.mean(axis=0) + [5.0, 5.0, 3.0]).astype(np.float32).astype(np.float64)
    kappa = np.array([0.0, 1.0, 3.0], np.float32)
    eps_r = np.array([1.0, 78.5, 40.0], np.float32)
    _, f, e = _gpu_eval(core, [t], [0, 0, 0], np.stack([p, p, p]), masses, kappa=kappa, eps_r=eps_r)
    for r in range(3):
        e_ref, f_ref = ref.term("pair14", p, t, float(kappa[r]), float(eps_r[r]))
        assert np.abs(f[r] - f_ref).max() < 1e-5 * _rms(f_ref), r
        assert e[r, ENERGY_COL["pair14"]] == pytest.approx(e_ref, rel=1e-5), r
    assert len({round(float(x), 6) for x in e[:, ENERGY_COL["pair14"]]}) == 3


@pytest.mark.gpu
def test_topology_validation(core):
    """add_bonded lehnt Tippfehler in Schlüsseln, falsche Shapes, verlustbehaftete Casts und
    Indizes außerhalb [0, n_atoms) ab."""
    topo, pos, masses = _load()
    p = pos - pos.mean(axis=0) + [5.0, 5.0, 3.0]
    good = _only(topo, "bond")
    bad_cases = [
        {**good, "dihedrals": topo["torsions"]},  # unbekannter Schlüssel
        {**good, "bonds": good["bonds"].T.copy()},  # (2, n) statt (n, 2)
        {**good, "bonds": good["bonds"].astype(np.float64) + 0.5},  # Float-Indizes
        {**good, "bonds": good["bonds"].astype(np.int64) + 2**32},  # int32-Überlauf
        {**good, "bonds": np.array([[0, 500]]), "bond_k": [1.0], "bond_r0": [0.1]},  # Index > n
    ]
    for bad in bad_cases:
        with pytest.raises(ValueError):
            _gpu_eval(core, [bad], [0], p[None], masses)
