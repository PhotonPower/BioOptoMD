"""Abnahmetests Phase 4: nicht-kovalente Kräfte (Tiled All-Pairs, LJ + Debye-Hückel, Switching).

Siehe docs/ROADMAP.md, Phase 4. Testsystem: zwei DNA-Dodecamer-Duplexe (PDB 1BNA, AMBER OL15) + 40 Na⁺
+ 2 Mg²⁺ (1558 Atome), tests/reference/inputs/dna_duplex_ions_topology.npz (generate_all.py).
ASSUMPTION(OQ-03): Fluorophor fehlt (keine GAFF2-Parameter verfügbar), ersetzt durch zweiten Duplex.
"""

from pathlib import Path

import nonbonded_ref as nr
import numpy as np
import pytest

from biooptomd.nonbonded import (
    exclusion_tiles,
    lj_type_table,
    nonbonded_topology,
    padded_atom_count,
    tile_pairs,
)

INPUTS = Path(__file__).resolve().parents[1] / "reference" / "inputs"
BONDED_KEYS = (
    "bonds", "bond_k", "bond_r0", "angles", "angle_k", "angle_theta0", "torsions", "torsion_k",
    "torsion_n", "torsion_phase", "pairs14", "pair_qq", "pair_sigma", "pair_eps",
)  # fmt: skip
KAPPA, EPS_R, R_S, R_C = 1.04, 78.5, 1.8, 2.0
NB_COL = 5  # Spalte von "nonbonded" in energies()


def _rms(f):
    return np.sqrt(np.mean(np.asarray(f) ** 2))


def _f32(x):
    return np.asarray(x, np.float32).astype(np.float64)


def _load_system():
    d = np.load(INPUTS / "dna_duplex_ions_topology.npz")
    return {k: d[k] for k in d.files if not k.startswith("meta_")}


def _random_system(rng, n, box, n_excl, min_dist=0.32):
    """Zufallssystem mit Mindestabstand (typischer vdW-Kontakt), damit die RMS-Kraft nicht von
    einzelnen unphysikalisch engen LJ-Kontakten dominiert wird (dort misst max/RMS nur FP32-Rundung)."""
    pos = np.empty((n, 3))
    k = 0
    while k < n:
        p = rng.uniform(0.0, 1.0, 3) * [box, box, 3.0]
        d = pos[:k] - p
        d[:, :2] -= box * np.rint(d[:, :2] / box)
        if k == 0 or np.min(np.einsum("ij,ij->i", d, d)) >= min_dist**2:
            pos[k] = p
            k += 1
    sigma = rng.choice([0.0, 0.25, 0.3, 0.34], n)
    eps = np.where(sigma > 0, rng.choice([0.2, 0.4, 0.9], n), 0.0)
    q = rng.uniform(-1.0, 1.0, n)
    excl = np.unique(np.sort(rng.integers(0, n, (n_excl, 2)), axis=1), axis=0)
    excl = excl[excl[:, 0] != excl[:, 1]]
    return pos, sigma, eps, q, excl


# --- CPU: Builder und Referenz ---------------------------------------------------------------


def test_builder_exclusion_masks():
    """Bitmasken = Brute-Force-Ausschlussmatrix (beide Richtungen in Diagonal-Tiles)."""
    rng = np.random.default_rng(40)
    n = 150
    n_pad = padded_atom_count(n)
    pairs = rng.integers(0, n, (400, 2))
    mask_index, masks = exclusion_tiles(n_pad, pairs)
    want = np.zeros((n_pad, n_pad), bool)
    for i, j in pairs:
        if i != j:
            want[i, j] = want[j, i] = True
    got = np.zeros_like(want)
    for p, (ti, tj) in enumerate(tile_pairs(n_pad)):
        if mask_index[p] < 0:
            continue
        for a in range(32):
            for b in range(32):
                if (int(masks[mask_index[p], a]) >> b) & 1:
                    got[ti * 32 + a, tj * 32 + b] = True
    got |= got.T
    np.testing.assert_array_equal(got, want)
    assert (mask_index >= 0).sum() == len(masks)


def test_builder_lj_table():
    sigma = np.array([0.3, 0.3, 0.25, 0.0])
    eps = np.array([0.5, 0.5, 0.1, 0.0])
    types, table = lj_type_table(sigma, eps)
    assert types.min() >= 1 and types[0] == types[1] != types[2]
    assert not table[0, :, 1].any() and not table[:, 0, 1].any()  # Typ 0: ε = 0
    for i in range(4):
        for j in range(4):
            s, e = table[types[i], types[j]]
            assert s == pytest.approx(0.5 * (sigma[i] + sigma[j]))
            assert e == pytest.approx(np.sqrt(eps[i] * eps[j]))


def test_reference_finite_differences():
    """NumPy-Referenz: F = −∇E (FP64), inkl. Switching-Bereich und Minimum Image."""
    rng = np.random.default_rng(41)
    pos, sigma, eps, q, excl = _random_system(rng, 40, 4.2, 30)
    types, table = lj_type_table(sigma, eps)
    table = table.astype(np.float64)
    box = (4.2, 4.2)

    def energy(p):
        return nr.nonbonded(p, q, types, table, excl, box, KAPPA, EPS_R, R_S, R_C)[0]

    _, f = nr.nonbonded(pos, q, types, table, excl, box, KAPPA, EPS_R, R_S, R_C)
    h = 1e-5
    f_fd = np.zeros_like(pos)
    for a in range(len(pos)):
        for d in range(3):
            p = pos.copy()
            p[a, d] += h
            ep = energy(p)
            p[a, d] -= 2 * h
            f_fd[a, d] = -(ep - energy(p)) / (2 * h)
    assert np.abs(f - f_fd).max() < 1e-5 * np.abs(f).max()
    r = np.linspace(1.7, 2.1, 9)
    s, _ = nr.switch(r, R_S, R_C)
    assert s[0] == 1.0 and s[-1] == 0.0 and np.all(np.diff(s) <= 0)


# --- GPU ----------------------------------------------------------------------------------------


@pytest.fixture
def core():
    if not pytest.importorskip("biooptomd._core").cuda_device_count():
        pytest.skip("kein CUDA-Gerät")
    from biooptomd import _core

    return _core


def _context(core, n_rep, n, box, positions, charges, types, masses=None):
    ctx = core.SimulationContext(n_rep, n)
    ctx.set_box(np.broadcast_to(np.asarray(box, np.float32), (n_rep, 2)).copy())
    ctx.set_masses(
        np.broadcast_to(np.float32(12.0) if masses is None else masses, (n_rep, n)).astype(
            np.float32
        )
    )
    ctx.set_positions(np.asarray(positions, np.float32))
    ctx.set_charges(np.asarray(charges, np.float32))
    t2 = np.zeros((n_rep, n, 2), np.int32)
    t2[..., 0] = types
    ctx.set_types(t2)
    return ctx


@pytest.mark.gpu
def test_pair_vs_numpy(core):
    """Kleine Systeme (versch. N, über Boxgrenzen, im Switching-Bereich) gegen O(N²)-NumPy:
    rel. Kraftfehler < 1e-5 (bezogen auf die RMS-Kraft), Energie < 1e-5 relativ; 3 Replicas mit
    unterschiedlichen Positionen, Topologien und Abschirmungen."""
    rng = np.random.default_rng(42)
    for n in (5, 37, 64, 130):
        box = 4.2
        n_pad = padded_atom_count(n)
        systems = [_random_system(rng, n, box, n // 2) for _ in range(3)]
        topos, all_types = [], []
        for _, sigma, eps, _, excl in systems:
            t, topo = nonbonded_topology(n_pad, sigma, eps, excl)
            topos.append(topo)
            all_types.append(t)
        kappa = np.array([0.0, 1.04, 2.5])
        eps_r = np.array([1.0, 78.5, 40.0])
        ctx = _context(
            core, 3, n, (box, box), np.stack([s[0] for s in systems]), np.stack([s[3] for s in systems]),
            np.stack(all_types),
        )  # fmt: skip
        ctx.set_nonbonded_tiled(
            topos, np.arange(3, dtype=np.int32), kappa.astype(np.float32), eps_r.astype(np.float32)
        )
        ctx.compute_forces()
        f = ctx.forces()[:, :n, :3].astype(np.float64)
        e = ctx.energies()[:, NB_COL]
        for r, (pos, _, _, q, excl) in enumerate(systems):
            e_ref, f_ref = nr.nonbonded(
                _f32(pos), _f32(q), all_types[r], topos[r]["lj_table"].astype(np.float64), excl,
                (box, box), float(np.float32(kappa[r])), float(np.float32(eps_r[r])), R_S, R_C,
            )  # fmt: skip
            assert np.abs(f[r] - f_ref).max() < 1e-5 * _rms(f_ref), (n, r)
            assert e[r] == pytest.approx(e_ref, rel=1e-5, abs=1e-6), (n, r)
        assert ctx.status() == 0
        assert not ctx.forces()[:, n:].any(), "Padding-Atome erhalten Kraft"


@pytest.mark.gpu
@pytest.mark.reference
def test_full_system_vs_openmm(core):
    """2 Duplexe + Ionen (1558 Atome) gegen OpenMM (CustomNonbondedForce mit DH + Switching,
    identische Ausschlüsse, Reference-Plattform): Nonbonded und Gesamtkraft (inkl. kovalenter Terme)
    max. Abweichung < 1e-4 · RMS-Kraft, Energie < 1e-4 relativ."""
    import openmm_nonbonded as onb

    system, pos, box = onb.load_system()
    q, sig, eps, excl = onb.extract_nonbonded(system)
    d = _load_system()
    n = len(q)
    p = _f32(pos)  # beide Engines an denselben FP32-Positionen
    ref = onb.reference_system(system, (*box, onb.BOX_Z), KAPPA, EPS_R, R_S, R_C)
    types, nbt = nonbonded_topology(padded_atom_count(n), sig, eps, excl)
    ctx = _context(core, 1, n, box, p[None], q[None], types[None], d["masses"])
    k1, e1 = np.full(1, KAPPA, np.float32), np.full(1, EPS_R, np.float32)
    ctx.set_nonbonded_tiled([nbt], np.zeros(1, np.int32), k1, e1, R_S, R_C)
    ctx.compute_forces()
    e_nb, f_nb = onb.evaluate(ref, p, [onb.NONBONDED_GROUP])
    f = ctx.forces()[0, :n, :3].astype(np.float64)
    assert np.abs(f - f_nb).max() < 1e-4 * _rms(f_nb)
    assert ctx.energies()[0, NB_COL] == pytest.approx(e_nb, rel=1e-4)

    # Zusätzlich (über die ROADMAP hinaus): Gesamtkraft inkl. kovalenter Terme. OpenMM bekommt
    # dieselben FP32-gerundeten kovalenten Parameter wie die GPU (wie die Positionen), sonst misst
    # der Vergleich nur die Parameterrundung (k·δr0 ≈ 2e-3 kJ/mol/nm pro Bindung).
    ref32 = onb.reference_system(
        system, (*box, onb.BOX_Z), KAPPA, EPS_R, R_S, R_C, float32_params=True
    )
    ctx.add_bonded([{k: d[k] for k in BONDED_KEYS}], np.zeros(1, np.int32), k1, e1)
    ctx.compute_forces()
    e_tot, f_tot = onb.evaluate(ref32, p, [0, 1, 2, 3, onb.NONBONDED_GROUP])
    # FP32-Rundungsfehler der Kernels skalieren mit der Größe der Einzelbeiträge (RMS je Term
    # ≈ 100), nicht mit der fast ausgeglichenen Summe (RMS ≈ 17); Maßstab daher die größte RMS-Kraft
    # eines Einzelterms. (Das Roadmap-Kriterium – Nonbonded allein – ist oben unverändert geprüft.)
    scale = max(_rms(onb.evaluate(ref32, p, [g])[1]) for g in (0, 1, 2, 3, onb.NONBONDED_GROUP))
    f = ctx.forces()[0, :n, :3].astype(np.float64)
    assert np.abs(f - f_tot).max() < 1e-4 * scale
    assert ctx.energies()[0, 1:].sum() == pytest.approx(e_tot, rel=1e-4)


def _full_context(core, n_rep, replica_slot, d, types, nbt, rng):
    """n_rep Replicas; in `replica_slot` das Referenzsystem, sonst gestörte Positionen/Ladungen."""
    n = len(d["masses"])
    pos = np.repeat(d["positions"][None], n_rep, axis=0)
    q = np.repeat(d["charges"][None], n_rep, axis=0)
    for r in range(n_rep):
        if r != replica_slot:
            pos[r] += rng.normal(0.0, 0.005, pos[r].shape)
            q[r] = rng.permutation(q[r])
    ctx = _context(core, n_rep, n, d["box"], pos, q, np.repeat(types[None], n_rep, 0), d["masses"])
    kappa = np.where(np.arange(n_rep) == replica_slot, KAPPA, rng.uniform(0.5, 2.0, n_rep)).astype(
        np.float32
    )
    eps_r = np.full(n_rep, EPS_R, np.float32)
    ctx.set_nonbonded_tiled([nbt], np.zeros(n_rep, np.int32), kappa, eps_r, R_S, R_C)
    ctx.add_bonded([{k: d[k] for k in BONDED_KEYS}], np.zeros(n_rep, np.int32), kappa, eps_r)
    ctx.compute_forces()
    return ctx


@pytest.mark.gpu
def test_padding_and_batch_independence(core):
    """Dasselbe System in Replica 0 von 1 und in Replica 7 von 16 liefert bitidentische Kräfte und
    Energien; die übrigen Replicas (andere Positionen, Ladungen, κ) beeinflussen es nicht."""
    d = _load_system()
    n = len(d["masses"])
    types, nbt = nonbonded_topology(padded_atom_count(n), d["sigma"], d["epsilon"], d["excluded"])
    rng = np.random.default_rng(43)
    single = _full_context(core, 1, 0, d, types, nbt, rng)
    batch = _full_context(core, 16, 7, d, types, nbt, rng)
    np.testing.assert_array_equal(
        batch.forces_fixed_point()[:, 7], single.forces_fixed_point()[:, 0]
    )
    np.testing.assert_array_equal(batch.energies_fixed_point()[7], single.energies_fixed_point()[0])
    assert not batch.forces()[:, n:].any(), "Padding-Atome erhalten Kraft"
    assert not np.array_equal(batch.forces_fixed_point()[:, 6], batch.forces_fixed_point()[:, 7])
    assert batch.status() == 0


@pytest.mark.gpu
def test_nve_full_system(core):
    """NVE 10 ps, Δt = 1 fs, kovalent + nicht-kovalent: Drift < 1e-4 kJ/mol/ps/Atom und
    Fluktuation < 1e-3·⟨E_kin⟩ (Kriterien aus Phase 3)."""
    d = _load_system()
    n = len(d["masses"])
    m = d["masses"]
    types, nbt = nonbonded_topology(padded_atom_count(n), d["sigma"], d["epsilon"], d["excluded"])
    n_rep = 4
    ctx = _context(
        core, n_rep, n, d["box"], np.repeat(d["positions"][None], n_rep, 0),
        np.repeat(d["charges"][None], n_rep, 0), np.repeat(types[None], n_rep, 0), m,
    )  # fmt: skip
    kb = core.units.KB
    rng = np.random.default_rng(44)
    v = rng.normal(0.0, 1.0, (n_rep, n, 3)) * np.sqrt(kb * 300.0 / m)[None, :, None]
    v -= (v * m[None, :, None]).sum(axis=1, keepdims=True) / m.sum()
    ctx.set_velocities(v)
    k, e = np.full(n_rep, KAPPA, np.float32), np.full(n_rep, EPS_R, np.float32)
    ctx.set_nonbonded_tiled([nbt], np.zeros(n_rep, np.int32), k, e, R_S, R_C)
    ctx.add_bonded([{key: d[key] for key in BONDED_KEYS}], np.zeros(n_rep, np.int32), k, e)
    ctx.set_integrator(0.001, 0.0, seed=0)
    tracker = ctx.add_energy_tracker(stride=10, capacity=1000)
    ctx.step(10_000)
    res = tracker.result()
    en = res["energies"]
    total = en[:, :, 0] + en[:, :, 1:].sum(axis=2)
    t_ps = res["steps"] * 0.001
    drift = np.array([np.polyfit(t_ps, total[r], 1)[0] for r in range(n_rep)]) / n
    fluct = total.std(axis=1) / en[:, :, 0].mean(axis=1)
    assert np.all(np.abs(drift) < 1e-4), f"Drift {drift}"
    assert np.all(fluct < 1e-3), f"Fluktuation {fluct}"
    assert ctx.status() == 0


@pytest.mark.gpu
def test_nonbonded_validation(core):
    """Box < 2 r_c wird abgelehnt; unbekannter LJ-Typ setzt STATUS_UNKNOWN_TYPE; Typ 0 braucht ε = 0."""
    rng = np.random.default_rng(45)
    pos, sigma, eps, q, excl = _random_system(rng, 40, 3.9, 10)
    types, topo = nonbonded_topology(64, sigma, eps, excl)
    ctx = _context(core, 1, 40, (3.9, 3.9), pos[None], q[None], types[None])
    ctx.set_nonbonded_tiled(
        [topo], np.zeros(1, np.int32), np.ones(1, np.float32), np.full(1, 78.5, np.float32)
    )
    with pytest.raises(RuntimeError):
        ctx.compute_forces()  # 3.9 < 2 · 2.0
    ctx.set_box(np.full((1, 2), 4.2, np.float32))
    bad_types = np.zeros((1, 40, 2), np.int32)
    bad_types[0, :, 0] = types
    bad_types[0, 0, 0] = 99
    ctx.set_types(bad_types)
    ctx.compute_forces()
    assert ctx.status() & core.STATUS_UNKNOWN_TYPE
    # ε = 0 mit σ > 0 bei winzigem Abstand: kein 0·inf = NaN (falscher STATUS_NON_FINITE).
    ctx.clear_status()
    t_eps0, topo0 = nonbonded_topology(64, np.full(40, 0.3), np.zeros(40), np.zeros((0, 2), int))
    close = pos.copy()
    close[1] = close[0] + [2e-4, 0.0, 0.0]
    ctx.set_positions(close[None].astype(np.float32))
    ctx.set_types(np.stack([t_eps0, np.zeros(40, np.int32)], axis=-1)[None])
    ctx.set_nonbonded_tiled(
        [topo0], np.zeros(1, np.int32), np.ones(1, np.float32), np.full(1, 78.5, np.float32)
    )
    ctx.set_charges(np.zeros((1, 40), np.float32))
    ctx.compute_forces()
    assert ctx.status() == 0 and not ctx.forces().any()
    bad = dict(topo, lj_table=topo["lj_table"] + 0.1)  # Typ 0 mit ε ≠ 0
    with pytest.raises(ValueError):
        ctx.set_nonbonded_tiled(
            [bad], np.zeros(1, np.int32), np.ones(1, np.float32), np.ones(1, np.float32)
        )
