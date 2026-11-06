"""Abnahmetests Phase 9.6: Tile-Paar-Liste (ADR 0014, ROADMAP Phase 9.6).

Kriterium: bitidentisch zum bisherigen Kernel (ein Warp je Tile-Paar, ``tile_list="off"``), weil
jedes Tile-Paar gleich gerechnet und über Fixed-Point-Integer akkumuliert wird (ADR 0002).
"""

import numpy as np
import pytest

from biooptomd import builder
from biooptomd import nonbonded as _nonbonded

pytestmark = pytest.mark.gpu


@pytest.fixture
def core():
    core = pytest.importorskip("biooptomd._core")
    if not core.cuda_device_count():
        pytest.skip("kein CUDA-Gerät")
    return core


def _chains(seed=1, n=96):
    from test_phase7 import _small_system

    return _small_system(np.random.default_rng(seed), n)


def _first_atoms(spec, m):
    """Die ersten m Atome einer Kettenvorlage (Terme und Exclusions nur innerhalb)."""
    import dataclasses

    bonded = {}
    for key, idx_key in (("bonds", "bond"), ("angles", "angle")):
        keep = (spec.bonded[key] < m).all(axis=1)
        bonded[key] = spec.bonded[key][keep]
        for k, v in spec.bonded.items():
            if k.startswith(idx_key + "_"):
                bonded[k] = v[keep]
    for k, v in spec.bonded.items():
        bonded.setdefault(k, v)
    return dataclasses.replace(
        spec, masses=spec.masses[:m], charges=spec.charges[:m], sigma=spec.sigma[:m],
        epsilon=spec.epsilon[:m], bonded=bonded, elements=spec.elements[:m],
        positions=spec.positions[:m], excluded=spec.excluded[(spec.excluded < m).all(axis=1)],
    )  # fmt: skip


def _pair(spec, n_rep, box, ionic=1.0, positions=None, **kw):
    """Zwei gleich aufgebaute Kontexte, mit und ohne Liste, Kräfte berechnet."""
    out = []
    for tile_list in ("on", "off"):
        ctx, info = builder.build_context(spec, n_rep, box, 300.0, ionic, positions=positions,
                                          tile_list=tile_list, **kw)  # fmt: skip
        ctx.compute_forces()
        out.append(ctx)
    return out, info


def _assert_identical(a, b):
    np.testing.assert_array_equal(a.energies_fixed_point(), b.energies_fixed_point())
    np.testing.assert_array_equal(a.forces(), b.forces())


def _spread(spec, n_copies, box, rng):
    """n_copies Kopien der Ketten, zufällig in x/y verteilt (verschiedene Positionen je Replica)."""
    pos = spec.positions
    return np.array([pos - pos.mean(0) + [rng.uniform(0, box[0]), rng.uniform(0, box[1]), 3.0]
                     for _ in range(n_copies)])  # fmt: skip


def test_tile_list_bitidentical_dh(core):
    """Debye-Hückel, 4 Replicas mit verschiedenen Boxen (alle ≥ 2 r_c) und Positionen: Energien und
    Kräfte bitidentisch; danach 200 Langevin-Schritte, Positionen bitidentisch."""
    spec = _chains()
    boxes = np.array([(8.0, 8.0), (9.0, 8.5), (10.0, 10.0), (8.2, 12.0)])
    pos = _spread(spec, 4, (8.0, 8.0), np.random.default_rng(2))
    (a, b), info = _pair(spec, 4, boxes, positions=pos)
    assert info.r_cutoff == 2.0 and (boxes >= 2 * info.r_cutoff).all()
    _assert_identical(a, b)
    for ctx in (a, b):
        ctx.set_integrator(0.001, 1.0, seed=4)
        ctx.step(200)
    np.testing.assert_array_equal(a.positions_precise(), b.positions_precise())
    _assert_identical(a, b)
    assert a.status() == 0


def test_tile_list_per_replica_topology(core):
    """Zwei Topologien (ADR 0005, verschiedene Exclusions) über die Replicas verteilt: bitidentisch."""
    spec = _chains()
    out = []
    for tile_list in ("on", "off"):
        ctx, info = builder.build_context(spec, 4, (8.0, 8.0), 300.0, 1.0, tile_list=tile_list)
        _, t0 = _nonbonded.nonbonded_topology(ctx.n_pad, spec.sigma, spec.epsilon, spec.excluded)
        _, t1 = _nonbonded.nonbonded_topology(ctx.n_pad, spec.sigma, spec.epsilon,
                                              spec.excluded[::2])  # fmt: skip
        k = info.kappa.astype(np.float32)
        e = info.eps_r.astype(np.float32)
        ctx.set_nonbonded_tiled([t0, t1], np.array([0, 1, 1, 0], np.int32), k, e,
                                r_switch=info.r_switch, r_cutoff=info.r_cutoff,
                                tile_list=tile_list)  # fmt: skip
        ctx.compute_forces()
        out.append(ctx)
    _assert_identical(*out)
    e = out[0].energies()[:, 5]  # Spalte "nonbonded"
    assert e[0] == e[3] and e[1] == e[2] and e[0] != e[1]


@pytest.mark.parametrize("case", ["all_culled", "none_culled", "single_tile", "boundary"])
def test_tile_list_extremes(core, case):
    """Grenzfälle: alle Tile-Paare außer der Diagonale verworfen, keines verworfen, ein einziges
    Tile, Atome an der periodischen Grenze (Paare nur über Minimum Image): bitidentisch."""
    rng = np.random.default_rng(5)
    spec = _first_atoms(_chains(n=48), 30) if case == "single_tile" else _chains()
    pos = spec.positions.copy()
    box = (8.0, 8.0)
    if case == "all_culled":
        pos[:, 2] += (np.arange(len(pos)) // 32) * 10.0
    elif case == "none_culled":
        pos = (pos - pos.mean(0)) * 0.3 + [4.0, 4.0, 3.0]
    elif case == "boundary":
        pos[:, 0] = np.where(np.arange(len(pos)) % 2 == 0, 0.05, 7.95) + rng.normal(
            0, 0.02, len(pos)
        )
    (a, b), _ = _pair(spec, 2, box, positions=pos)
    _assert_identical(a, b)
    if case != "all_culled":
        assert np.abs(a.energies()[:, 5]).max() > 0


@pytest.mark.reference
@pytest.mark.parametrize("solvent", ["gb-obc2", "gb-gbn2"])
def test_tile_list_bitidentical_gb(core, solvent):
    """GB-OBC2 und GBn2 (drei Tile-Passes), Duplex in 8 × 8 nm, 2 Replicas: bitidentisch."""
    from test_phase9 import _duplex

    system, top, pos = _duplex(gb_xml=f"implicit/{solvent[3:]}.xml")
    pos = pos - pos.mean(axis=0) + [4.0, 4.0, 4.0]
    spec = builder.from_openmm(system, top, pos)
    (a, b), info = _pair(spec, 2, (8.0, 8.0), solvent=solvent)
    assert info.r_cutoff == 2.0
    _assert_identical(a, b)


@pytest.mark.reference
def test_tile_list_bitidentical_20k(core):
    """Benchmark-System 1 × 20 k (18 696 Atome, 585 Tiles, r_c = 3.92 nm): bitidentisch; die Liste
    enthält nur einen Teil der Tile-Paare (Vorauswahl greift)."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarks"))
    import throughput

    system, top, pos, box = throughput.make_system("20k")
    spec = builder.from_openmm(system, top, pos)
    (a, b), info = _pair(spec, 1, box, ionic=0.15)
    assert info.r_cutoff > 3.9
    _assert_identical(a, b)


def test_tile_list_determinism(core):
    """Zwei Läufe mit Liste (1000 Langevin-Schritte, 4 Replicas) sind bitidentisch."""
    spec = _chains()
    pos = _spread(spec, 4, (8.0, 8.0), np.random.default_rng(6))
    runs = []
    for _ in range(2):
        ctx, _ = builder.build_context(spec, 4, (8.0, 8.0), 300.0, 1.0, positions=pos)
        ctx.set_integrator(0.001, 1.0, seed=9)
        ctx.step(1000)
        runs.append((ctx.positions_precise(), ctx.energies_fixed_point()))
    np.testing.assert_array_equal(runs[0][0], runs[1][0])
    np.testing.assert_array_equal(runs[0][1], runs[1][1])


def test_tile_list_auto(core):
    """Auto-Auswahl: Liste nur, wenn π (r_c + 2 nm)² < 0.5 · L_x L_y für alle Replicas; Bildsumme
    (L < 2 r_c) nie; Ergebnis in jedem Fall bitidentisch zu 'off'."""
    spec = _chains()
    cases = [((8.0, 8.0), False), ((24.0, 32.0), True), ((3.0, 3.0), False)]
    for box, expected in cases:
        ctx, _ = builder.build_context(spec, 2, box, 300.0, 1.0)
        assert ctx.nonbonded_uses_tile_list == expected, box
        ctx.compute_forces()
        ref, _ = builder.build_context(spec, 2, box, 300.0, 1.0, tile_list="off")
        ref.compute_forces()
        _assert_identical(ctx, ref)
    mixed, _ = builder.build_context(spec, 2, np.array([(24.0, 32.0), (8.0, 8.0)]), 300.0, 1.0)
    assert not mixed.nonbonded_uses_tile_list
