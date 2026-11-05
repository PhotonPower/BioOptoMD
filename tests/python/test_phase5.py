"""Abnahmetests Phase 5: H-Bindungs-Constraints (SHAKE/RATTLE), HMR und N_dof.

Siehe docs/ROADMAP.md, Phase 5. Testsystem: zwei DNA-Duplexe (1BNA, AMBER OL15) aus
tests/reference/inputs/dna_duplex_ions_topology.npz; für den Oberflächentest ohne freie Ionen.
"""

from pathlib import Path

import constraints_ref as cref
import numpy as np
import pytest
from scipy import stats

from biooptomd.constraints import constraint_topology, repartition_hydrogen_mass
from biooptomd.nonbonded import nonbonded_topology, padded_atom_count

INPUTS = Path(__file__).resolve().parents[1] / "reference" / "inputs"
BONDED_KEYS = (
    "bonds", "bond_k", "bond_r0", "angles", "angle_k", "angle_theta0", "torsions", "torsion_k",
    "torsion_n", "torsion_phase", "pairs14", "pair_qq", "pair_sigma", "pair_eps",
)  # fmt: skip
KAPPA, EPS_R = 1.04, 78.5


def _load(drop_ions=False):
    d = np.load(INPUTS / "dna_duplex_ions_topology.npz")
    d = {k: d[k] for k in d.files if not k.startswith("meta_")}
    if drop_ions:  # Ionen sind die letzten Atome und haben keine kovalenten Terme
        bonded_atoms = np.unique(d["bonds"])
        n = int(bonded_atoms.max()) + 1
        assert (bonded_atoms == np.arange(n)).all()
        for k in ("positions", "masses", "charges", "sigma", "epsilon"):
            d[k] = d[k][:n]
        assert (d["excluded"] < n).all()
    return d


def _block_mean_sigma(x, n_blocks=20):
    n = x.shape[-1] // n_blocks * n_blocks
    blocks = x[..., :n].reshape(x.shape[:-1] + (n_blocks, -1)).mean(axis=-1)
    return blocks.mean(axis=-1), blocks.std(axis=-1, ddof=1) / np.sqrt(n_blocks)


def _bond_deviation(x, clusters, distances, box):
    """max_k |r_k − d_k| / d_k über alle Constraints, x (n_rep, n, 3)."""
    dev = 0.0
    for k in range(3):
        ok = clusters[:, 1 + k] >= 0
        dv = x[:, clusters[ok, 1 + k]] - x[:, clusters[ok, 0]]
        dv[..., :2] -= np.asarray(box) * np.rint(dv[..., :2] / np.asarray(box))
        dev = max(dev, float(np.abs(np.linalg.norm(dv, axis=-1) / distances[ok, k] - 1.0).max()))
    return dev


# --- CPU: Builder ------------------------------------------------------------------------------


def test_builder_hmr_and_clusters():
    """HMR erhält die Gesamtmasse, H = 3.024 amu; jedes H in genau einem Cluster; constrainte
    Bindungen werden aus der kovalenten Topologie entfernt."""
    d = _load()
    topo = {k: d[k] for k in BONDED_KEYS}
    is_h = d["masses"] < 1.5
    m = repartition_hydrogen_mass(d["masses"], topo["bonds"], is_h)
    assert m.sum() == pytest.approx(d["masses"].sum(), rel=1e-12)
    np.testing.assert_allclose(m[is_h], 3.024)
    assert (m[~is_h] < d["masses"][~is_h] + 1e-12).all()
    cons, reduced = constraint_topology(topo, is_h)
    cl = cons["clusters"]
    members = cl[cl >= 0]
    assert len(members) == len(np.unique(members)), "Atom in mehreren Clustern"
    assert set(cl[:, 1:][cl[:, 1:] >= 0]) == set(np.flatnonzero(is_h))
    assert not is_h[cl[:, 0]].any()
    assert len(reduced["bonds"]) == len(topo["bonds"]) - is_h.sum()
    assert not is_h[reduced["bonds"]].any()


# --- GPU ----------------------------------------------------------------------------------------


@pytest.fixture
def core():
    if not pytest.importorskip("biooptomd._core").cuda_device_count():
        pytest.skip("kein CUDA-Gerät")
    from biooptomd import _core

    return _core


def _context(
    core, d, n_rep, dt, gamma, seed=1, surface=False, temperature=300.0, anchors=2, tol=1e-6
):
    """Duplex-System mit HMR + H-Constraints, kovalent + nicht-kovalent (+ optional Oberfläche)."""
    n = len(d["masses"])
    topo = {k: d[k] for k in BONDED_KEYS}
    is_h = d["masses"] < 1.5
    m = repartition_hydrogen_mass(d["masses"], topo["bonds"], is_h)
    cons, topo_r = constraint_topology(topo, is_h)
    types, nbt = nonbonded_topology(padded_atom_count(n), d["sigma"], d["epsilon"], d["excluded"])
    ctx = core.SimulationContext(n_rep, n)
    ctx.set_box(np.tile(d["box"], (n_rep, 1)).astype(np.float32))
    ctx.set_masses(np.tile(m, (n_rep, 1)).astype(np.float32))
    ctx.set_positions(np.tile(d["positions"], (n_rep, 1, 1)).astype(np.float32))
    ctx.set_charges(np.tile(d["charges"], (n_rep, 1)).astype(np.float32))
    t2 = np.zeros((n_rep, n, 2), np.int32)
    t2[..., 0] = types
    ctx.set_types(t2)
    ctx.set_temperatures(np.full(n_rep, temperature, np.float32))
    k, e = np.full(n_rep, KAPPA, np.float32), np.full(n_rep, EPS_R, np.float32)
    ctx.add_bonded([topo_r], np.zeros(n_rep, np.int32), k, e)
    ctx.set_nonbonded_tiled([nbt], np.zeros(n_rep, np.int32), k, e)
    ctx.set_constraints([cons], np.zeros(n_rep, np.int32), tolerance=tol)
    if surface:
        # 9-3-Wand bei z = 0 (ASSUMPTION(OQ-05): graphenartige Platzhalterwerte) und je Duplex ein
        # Ankeratom (tiefstes Schweratom): z-Restraint + schwacher xy-Restraint (ASSUMPTION(OQ-07)).
        # Oberflächen-Typ 1 für alle Atome (Typ 0 ist reserviert, ADR 0007)
        t2[..., 1] = 1
        ctx.set_types(t2)
        ctx.add_planar_surface(np.array([0.0, 2.0], np.float32), np.array([0.0, 0.3], np.float32))
        z = d["positions"][:, 2]
        parts = [slice(0, n)] if anchors == 1 else [slice(0, n // 2), slice(n // 2, n)]
        anchor_atoms = [int(np.argmin(np.where(~is_h[s], z[s], np.inf))) + s.start for s in parts]
        rep = np.repeat(np.arange(n_rep), len(anchor_atoms)).astype(np.int32)
        atom = np.tile(anchor_atoms, n_rep).astype(np.int32)
        ref = d["positions"][atom].astype(np.float32)
        ctx.add_position_restraints(
            rep, atom, ref, np.full(len(atom), 1000.0, np.float32), dims=(False, False, True)
        )
        ctx.add_position_restraints(
            rep, atom, ref, np.full(len(atom), 10.0, np.float32), dims=(True, True, False)
        )
    ctx.set_integrator(dt, gamma, seed)
    return ctx, cons, m


@pytest.mark.gpu
def test_constraint_step_vs_numpy(core):
    """Ein NVE-Schritt (B, RATTLE, A+SHAKE, RATTLE, A+SHAKE) gegen die NumPy-Referenz mit denselben
    Kräften: Positionen auf 1e-9 nm gleich, Constraints erfüllt. Beide Seiten sind auf 1e-12
    auskonvergiert, sodass der Vergleich die Lösung der Constraint-Gleichungen prüft und nicht das
    Iterationsverfahren (GPU: M-SHAKE/direktes RATTLE, Referenz: Gauss-Seidel). Mit der Default-
    Toleranz 1e-6 dürfen beide Lösungen bis ~1e-7 nm abweichen."""
    d = _load()
    n = len(d["masses"])
    ctx, cons, m = _context(core, d, 1, 0.002, 0.0, tol=1e-12)
    rng = np.random.default_rng(50)
    ctx.set_velocities(
        rng.normal(0.0, 1.0, (1, n, 3)) * np.sqrt(core.units.KB * 300.0 / m)[None, :, None]
    )
    ctx.apply_constraints()
    ctx.compute_forces()
    x0 = ctx.positions_precise()[0, :n]
    v0 = ctx.velocities()[0, :n, :3]
    f0 = ctx.forces()[0, :n, :3].astype(np.float64)
    inv_m = ctx.velocities()[0, :n, 3]
    x_ref, _ = cref.constrained_positions_after_step(
        x0, v0, f0, inv_m, cons["clusters"], cons["distances"].astype(np.float64), d["box"], 0.002,
        tol=1e-12, tol_v=1e-14,
    )  # fmt: skip
    ctx.step(1)
    x1 = ctx.positions_precise()[0, :n]
    images = ctx.images()[0, :n]
    x1[:, :2] += images * d["box"]  # ungewrappt vergleichen
    assert np.abs(x1 - x_ref).max() < 1e-9
    assert _bond_deviation(x1[None], cons["clusters"], cons["distances"], d["box"]) < 1e-6
    assert ctx.status() == 0


@pytest.mark.gpu
def test_constraint_tolerance(core):
    """Langevin bei 300 K, Δt = 2 fs: nach jedem der 200 Schritte max. rel. Bindungslängenabweichung
    < 1e-6, und die Geschwindigkeiten haben keine Komponente entlang der Constraints."""
    d = _load()
    n = len(d["masses"])
    ctx, cons, _ = _context(core, d, 2, 0.002, 1.0)
    for _ in range(200):
        ctx.step(1)
        x = ctx.positions_precise()[:, :n]
        assert _bond_deviation(x, cons["clusters"], cons["distances"], d["box"]) < 1e-6
    v = ctx.velocities()[:, :n, :3]
    cl = cons["clusters"]
    for k in range(3):
        ok = cl[:, 1 + k] >= 0
        r = x[:, cl[ok, 1 + k]] - x[:, cl[ok, 0]]
        r[..., :2] -= d["box"] * np.rint(r[..., :2] / d["box"])
        rv = np.einsum("rij,rij->ri", r, v[:, cl[ok, 1 + k]] - v[:, cl[ok, 0]])
        assert np.abs(rv).max() < 1e-8 * np.einsum("rij,rij->ri", r, r).max()
    assert ctx.status() == 0


@pytest.mark.gpu
def test_equipartition_with_constraints(core):
    """Freie starre Cluster (CH, CH₂, CH₃) bei 280/380 K: ⟨E_kin⟩ = ½ N_dof k_B T mit
    N_dof = 3 N − N_constraints (4σ); degrees_of_freedom() liefert genau diesen Wert."""
    rng = np.random.default_rng(51)
    temps = np.array([280.0, 380.0])
    n_h_per = np.tile([1, 2, 3], 200)
    n = int((1 + n_h_per).sum())
    pos = np.zeros((n, 3))
    masses = np.zeros(n)
    clusters = -np.ones((len(n_h_per), 4), np.int32)
    distances = np.zeros((len(n_h_per), 3), np.float32)
    a = 0
    for c, nh in enumerate(n_h_per):
        center = rng.uniform([0, 0, 1], [10, 10, 9])
        pos[a], masses[a] = center, 12.0 - 2.016 * nh
        clusters[c, 0] = a
        for k in range(nh):
            u = rng.normal(size=3)
            pos[a + 1 + k] = center + 0.109 * u / np.linalg.norm(u)
            masses[a + 1 + k] = 3.024
            clusters[c, 1 + k] = a + 1 + k
            distances[c, k] = 0.109
        a += 1 + nh
    ctx = core.SimulationContext(2, n)
    ctx.set_box(np.full((2, 2), 10.0, np.float32))
    ctx.set_masses(np.tile(masses, (2, 1)).astype(np.float32))
    ctx.set_positions(np.tile(pos, (2, 1, 1)).astype(np.float32))
    ctx.set_temperatures(temps.astype(np.float32))
    ctx.set_constraints([{"clusters": clusters, "distances": distances}], np.zeros(2, np.int32))
    n_dof = ctx.degrees_of_freedom()
    assert n_dof.tolist() == [3 * n - int(n_h_per.sum())] * 2
    gamma, dt = 2.0, 0.001
    ctx.set_integrator(dt, gamma, seed=7)
    ctx.step(int(5.0 / gamma / dt))
    tracker = ctx.add_energy_tracker(stride=10, capacity=4000)
    ctx.step(30_000)
    mean, sigma = _block_mean_sigma(tracker.result()["energies"][:, :, 0])
    expected = 0.5 * n_dof * core.units.KB * temps
    assert np.all(np.abs(mean - expected) < 4.0 * sigma), (mean, expected, sigma)
    assert np.all(sigma < 0.01 * expected)
    assert ctx.status() == 0


def _single_duplex(d):
    """Erster Duplex des Systems (ohne Ionen und ohne zweiten Duplex) als "Aptamer"."""
    d = _load(drop_ions=True) if d is None else d
    n = len(d["masses"]) // 2
    out = dict(d)
    for k in ("positions", "masses", "charges", "sigma", "epsilon"):
        out[k] = d[k][:n]
    out["excluded"] = d["excluded"][(d["excluded"] < n).all(axis=1)]
    for idx, params in (
        ("bonds", ("bond_k", "bond_r0")), ("angles", ("angle_k", "angle_theta0")),
        ("torsions", ("torsion_k", "torsion_n", "torsion_phase")),
        ("pairs14", ("pair_qq", "pair_sigma", "pair_eps")),
    ):  # fmt: skip
        keep = (d[idx] < n).all(axis=1)
        assert ((d[idx] >= n).all(axis=1) | keep).all(), "Duplexe sind kovalent gekoppelt"
        out[idx] = d[idx][keep]
        for p in params:
            out[p] = d[p][keep]
    return out


@pytest.mark.gpu
@pytest.mark.slow
def test_stability_4fs(core):
    """DNA-Duplex an der Oberfläche (9-3-Wand, verankert), HMR + H-Constraints, Δt = 4 fs, 1 ns bei
    300 K ohne Explosion (endlich, Constraints erfüllt, Statuswort 0, Molekül über der Wand); die
    Verteilung der potentiellen Energie ist konsistent mit einem 1-ns-Lauf bei Δt = 1 fs.

    Statistik (AGENTS.md Regel 8): Auswertung erst nach 500 ps (das System relaxiert langsam), je
    Lauf 8 unabhängige Replicas. (a) ⟨E_pot⟩ mit σ aus den Replica-Mittelwerten: |Δ| < 4σ. (b) KS-Test
    auf Samples im Abstand von 50 ps (Autokorrelation von E_pot dort ≈ 0.08), p > 0.001.

    Verglichen wird E_pot (Konfigurationen): BAOAB sampelt Konfigurationen sehr genau, die kinetische
    Temperatur aus Geschwindigkeiten ist dagegen bei großem Δt um O(Δt²) zu niedrig (PHYSICS_MODEL §1).
    """
    d = _single_duplex(None)
    n = len(d["masses"])
    n_rep = 8

    def run(dt):
        ctx, cons, _ = _context(
            core, d, n_rep, dt, 1.0, seed=int(dt * 1e6), surface=True, anchors=1
        )
        tracker = ctx.add_energy_tracker(stride=round(50.0 / dt), capacity=100)
        ctx.step(round(1000.0 / dt))
        res = tracker.result()
        t_ps = res["steps"] * dt
        e_pot = res["energies"][:, :, 1:].sum(axis=2)
        return ctx, cons, e_pot[:, t_ps > 500.0]

    ctx4, cons, e4 = run(0.004)
    x = ctx4.positions_precise()[:, :n]
    assert np.isfinite(e4).all() and np.isfinite(x).all()
    assert ctx4.status() == 0
    assert _bond_deviation(x, cons["clusters"], cons["distances"], d["box"]) < 1e-6
    assert x[..., 2].min() > 0.1, "Atom durch die Wand"
    assert x[..., 2].max() < 15.0, "Molekül hat sich von der Oberfläche gelöst"
    _, _, e1 = run(0.001)
    m4, m1 = e4.mean(axis=1), e1.mean(axis=1)  # je Replica (unabhängig)
    sigma = np.sqrt(m4.var(ddof=1) / n_rep + m1.var(ddof=1) / n_rep)
    assert abs(m4.mean() - m1.mean()) < 4.0 * sigma, (m4.mean(), m1.mean(), sigma)
    p = stats.ks_2samp(e4.ravel(), e1.ravel()).pvalue
    assert p > 1e-3, f"KS p = {p:.2e}"


@pytest.mark.gpu
def test_constraints_immobile_atoms(core):
    """Cluster mit inv_mass = 0 (nur über restore_state erreichbar): festes Schweratom (n_h = 2),
    festes H (n_h = 1) – der gekoppelte Löser bzw. sein Gauss-Seidel-Fallback projiziert wie die
    NumPy-Referenz (auf 1e-12 auskonvergiert, Vergleich 1e-9 nm), unbewegliche Atome bleiben
    stehen. Cluster mit beiden Atomen fest und verletzter Länge: unverändert, Statusbit
    STATUS_CONSTRAINT_FAILURE."""
    x = np.array(
        [[1.0, 1.0, 1.0], [1.13, 1.02, 1.0], [0.97, 1.11, 1.01],  # C (fest), H, H
         [2.0, 1.0, 1.0], [2.0, 1.12, 0.98],  # C, H (fest)
         [3.0, 1.0, 1.0], [3.15, 1.0, 1.0]],  # C, H (beide fest)
        np.float64,
    )  # fmt: skip
    n = len(x)
    clusters = np.array([[0, 1, 2, -1], [3, 4, -1, -1], [5, 6, -1, -1]], np.int32)
    distances = np.array([[0.1, 0.1, 0], [0.1, 0, 0], [0.1, 0, 0]], np.float32)
    box = (5.0, 5.0)
    ctx = core.SimulationContext(1, n)
    ctx.set_box(np.array([box], np.float32))
    ctx.set_masses(np.array([[12.0, 1.0, 1.0, 12.0, 1.0, 12.0, 1.0]], np.float32))
    ctx.set_positions(x[None].astype(np.float32))
    rng = np.random.default_rng(51)
    ctx.set_velocities(rng.normal(0.0, 1.0, (1, n, 3)))
    ctx.set_constraints(
        [{"clusters": clusters, "distances": distances}], np.zeros(1, np.int32), tolerance=1e-12
    )
    st = ctx.save_state()
    st["vel"][0, [0, 4, 5, 6], 3] = 0.0  # inv_mass = 0
    st["vel"][0, [0, 4, 5, 6], :3] = 0.0
    ctx.restore_state(
        st["pos"], st["pos_corr"], st["vel"], st["image"], st["step"], st["time"],
        st["previous_time"],
    )  # fmt: skip
    x0 = ctx.positions_precise()[0, :n]
    v0 = ctx.velocities()[0, :n, :3].astype(np.float64)
    inv_m = ctx.velocities()[0, :n, 3]
    ctx.apply_constraints()
    x1 = ctx.positions_precise()[0, :n]
    v1 = ctx.velocities()[0, :n, :3]

    x_ref, v_ref = x0.copy(), v0.copy()
    cref.shake(x_ref, x0, inv_m, clusters[:2], distances[:2].astype(np.float64), box, tol=1e-12)
    cref.rattle(x_ref, v_ref, inv_m, clusters[:2], box, tol_v=1e-14)
    mobile = slice(0, 5)
    np.testing.assert_allclose(x1[mobile], x_ref[mobile], rtol=0, atol=1e-9)
    np.testing.assert_allclose(v1[mobile], v_ref[mobile], rtol=0, atol=1e-9)
    for atom in (0, 4, 5, 6):  # unbewegliche Atome stehen still
        np.testing.assert_array_equal(x1[atom], x0[atom])
        np.testing.assert_array_equal(v1[atom], v0[atom])
    for c, h in ((0, 1), (0, 2), (3, 4)):
        assert np.linalg.norm(x1[h] - x1[c]) == pytest.approx(float(np.float32(0.1)), rel=1e-10)
    assert np.linalg.norm(x1[6] - x1[5]) == pytest.approx(0.15)  # nicht projizierbar
    assert ctx.status() & core.STATUS_CONSTRAINT_FAILURE
