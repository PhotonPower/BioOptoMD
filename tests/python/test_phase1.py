"""Abnahmetests Phase 1: DeviceState, Fixed-Point, wrap_xy, NumPy/DLPack-Export.

Siehe docs/ROADMAP.md, Phase 1. Alle Tests benötigen ein CUDA-Gerät.
"""

import numpy as np
import pytest

from core_ref import accumulate_fixed_point, wrap_coordinate

pytestmark = pytest.mark.gpu

_core = pytest.importorskip("biooptomd._core")

N_REPLICAS = 4
N_ATOMS = 500


def _ctx(n_replicas=N_REPLICAS, n_atoms=N_ATOMS):
    return _core.SimulationContext(n_replicas, n_atoms)


def _random_positions(rng, n_replicas, n_atoms, scale=5.0):
    return rng.uniform(-scale, scale, (n_replicas, n_atoms, 3)).astype(np.float32)


# --- Abnahmetests (Namen verbindlich laut ROADMAP.md) ----------------------------------------


def test_allocate_batch():
    """4 Replicas × 500 Atome → N_pad = 512, korrekte Shapes, Padding q = 0 und inv_mass = 0."""
    ctx = _ctx()
    assert (ctx.n_replicas, ctx.n_atoms, ctx.n_pad) == (4, 500, 512)
    assert _core.padded_atom_count(500) == 512
    assert _core.padded_atom_count(512) == 512
    assert _core.padded_atom_count(513) == 544

    r, p, nt = 4, 512, _core.N_ENERGY_TERMS
    expected = {
        "positions": ((r, p, 4), np.float32),
        "velocities": ((r, p, 4), np.float32),
        "forces": ((r, p, 4), np.float32),
        "forces_fixed_point": ((3, r, p), np.int64),
        "energies_fixed_point": ((r, nt), np.int64),
        "types": ((r, p, 2), np.int32),
        "box": ((r, 2), np.float32),
        "temperatures": ((r,), np.float32),
        "images": ((r, p, 2), np.int32),
    }
    for name, (shape, dtype) in expected.items():
        a = getattr(ctx, name)()
        assert a.shape == shape, name
        assert a.dtype == dtype, name
        assert not a.any(), f"{name} nach Allokation nicht 0"

    # Echte Atome mit Nicht-Null-Werten belegen; Padding muss unverändert bleiben.
    rng = np.random.default_rng(1)
    q = rng.uniform(-1.0, 1.0, (r, N_ATOMS)).astype(np.float32)
    m = rng.uniform(1.0, 200.0, (r, N_ATOMS)).astype(np.float32)
    types = rng.integers(1, 10, (r, N_ATOMS, 2)).astype(np.int32)
    xyz = _random_positions(rng, r, N_ATOMS)
    v = rng.normal(0.0, 1.0, (r, N_ATOMS, 3)).astype(np.float32)
    ctx.set_positions(xyz)
    ctx.set_charges(q)
    ctx.set_masses(m)
    ctx.set_velocities(v)
    ctx.set_types(types)

    pos, vel, typ = ctx.positions(), ctx.velocities(), ctx.types()
    np.testing.assert_array_equal(pos[:, :N_ATOMS, :3], xyz)
    np.testing.assert_array_equal(pos[:, :N_ATOMS, 3], q)
    np.testing.assert_array_equal(vel[:, :N_ATOMS, :3], v)
    np.testing.assert_array_equal(
        vel[:, :N_ATOMS, 3], (1.0 / m.astype(np.float64)).astype(np.float32)
    )
    np.testing.assert_array_equal(typ[:, :N_ATOMS], types)

    pad = slice(N_ATOMS, None)
    assert not pos[:, pad].any(), "Padding-Positionen/-Ladungen nicht 0"
    assert not vel[:, pad].any(), "Padding-Geschwindigkeiten/inv_mass nicht 0"
    assert not typ[:, pad].any(), "Padding-Typen nicht 0"


def test_roundtrip_numpy_and_dlpack():
    """Positionen setzen, Dummy-Kernel (+1.0 nm in x), über NumPy und DLPack (CuPy) zurücklesen."""
    cp = pytest.importorskip("cupy")
    rng = np.random.default_rng(2)
    ctx = _ctx()
    xyz = _random_positions(rng, N_REPLICAS, N_ATOMS)
    q = rng.uniform(-1.0, 1.0, (N_REPLICAS, N_ATOMS)).astype(np.float32)
    ctx.set_positions(xyz)
    ctx.set_charges(q)

    _core.testing.shift_x(ctx, 1.0)

    expected = np.zeros((N_REPLICAS, ctx.n_pad, 4), dtype=np.float32)
    expected[:, :N_ATOMS, :3] = xyz
    expected[:, :N_ATOMS, 0] += np.float32(1.0)  # float32-Addition wie auf dem Gerät
    expected[:, :N_ATOMS, 3] = q

    host = ctx.positions()
    np.testing.assert_array_equal(host, expected)

    view = ctx.positions(dlpack=True)
    assert view.__dlpack_device__() == (2, ctx.device)  # kDLCUDA
    dev = cp.from_dlpack(view)
    assert dev.shape == (N_REPLICAS, ctx.n_pad, 4) and dev.dtype == cp.float32
    np.testing.assert_array_equal(cp.asnumpy(dev), expected)

    # Zero-Copy: Schreiben über CuPy ist im Kontext sichtbar und umgekehrt.
    dev[0, 0, 1] = 42.0
    cp.cuda.Stream.null.synchronize()
    assert ctx.positions()[0, 0, 1] == np.float32(42.0)
    _core.testing.shift_x(ctx, 1.0)
    ctx.synchronize()
    assert float(dev[0, 0, 0]) == float(np.float32(expected[0, 0, 0] + np.float32(1.0)))

    # Die Sicht hält den Kontext (und damit den Speicher) am Leben.
    del ctx, view
    assert float(dev[0, 0, 1]) == 42.0


def test_fixed_point_roundtrip():
    """Zufallskräfte ±1e4 kJ/mol/nm: Fehler < 1e-6 relativ, bitidentisch über 10 Reihenfolgen."""
    rng = np.random.default_rng(3)
    ctx = _ctx()
    n_total = N_REPLICAS * ctx.n_pad
    real = (np.arange(N_REPLICAS)[:, None] * ctx.n_pad + np.arange(N_ATOMS)[None, :]).ravel()

    # 16 Beiträge pro echtem Atom + 4096 Beiträge auf wenige „heiße“ Atome (hohe Konkurrenz).
    hot = real[rng.choice(real.size, 8, replace=False)]
    atom_index = np.concatenate([np.repeat(real, 16), rng.choice(hot, 4096)]).astype(np.int32)
    forces = rng.uniform(-1.0e4, 1.0e4, (atom_index.size, 3)).astype(np.float32)

    expected_fp = accumulate_fixed_point(n_total, atom_index, forces)
    expected_f = np.zeros((3, n_total))
    for d in range(3):
        np.add.at(expected_f[d], atom_index, forces[:, d].astype(np.float64))

    results = []
    for _ in range(10):
        order = rng.permutation(atom_index.size)
        ctx.clear_forces()
        _core.testing.accumulate_forces(ctx, atom_index[order], forces[order])
        ctx.fixed_point_to_float()
        fp = ctx.forces_fixed_point().reshape(3, n_total)
        np.testing.assert_array_equal(fp, expected_fp)  # exakt = Referenzsumme
        results.append((fp, ctx.forces()))

    for fp, f in results[1:]:
        assert np.array_equal(fp, results[0][0]), "Fixed-Point nicht bitidentisch"
        assert np.array_equal(f.view(np.int32), results[0][1].view(np.int32)), "float-Kraft"

    f = results[0][1].reshape(n_total, 4)
    got = f[:, :3].T.astype(np.float64)
    nonzero = expected_f != 0.0
    rel = np.abs(got[nonzero] - expected_f[nonzero]) / np.abs(expected_f[nonzero])
    assert rel.max() < 1.0e-6, f"max. rel. Fehler {rel.max():.3e}"
    assert not got[~nonzero].any(), "Atome ohne Beitrag (inkl. Padding) haben Kraft != 0"
    assert not f[:, 3].any()


def test_wrap_xy():
    """Wrap nach [0, L), Imagezähler und ungewrappte Rekonstruktion (exakt, siehe Docstring).

    Die gewrappten Koordinaten und Imagezähler müssen bitgenau der NumPy-Referenz entsprechen.
    Die Rekonstruktion x_w + n·L ist exakt, wenn |x| ≥ L (x − n·L in float32 darstellbar);
    für x ∈ (−L, 0) ist x + L genau einmal gerundet (Fehler ≤ ½ ulp(L)).
    """
    rng = np.random.default_rng(4)
    n_atoms = 200
    box = np.array([[3.0, 4.5], [5.1, 2.7], [7.3, 7.3]], dtype=np.float32)
    n_rep = box.shape[0]
    ctx = _ctx(n_rep, n_atoms)
    ctx.set_box(box)
    big_l = box[:, None, :]  # (r, 1, 2)

    xyz = rng.uniform(-3.0, 4.0, (n_rep, n_atoms, 3)).astype(np.float32)
    xyz[..., :2] *= big_l
    # Randfälle: 0, L, −L, 2L, knapp unter 0 / L, −0.0, große Werte.
    for r in range(n_rep):
        for d in range(2):
            lx = box[r, d]
            edge = [0.0, lx, -lx, 2 * lx, -1e-7, np.nextafter(lx, np.float32(0)), -0.0, 1e3 * lx]
            xyz[r, : len(edge), d] = np.array(edge, dtype=np.float32)
    q = rng.uniform(-1.0, 1.0, (n_rep, n_atoms)).astype(np.float32)
    ctx.set_positions(xyz)
    ctx.set_charges(q)

    ctx.wrap_xy()
    pos, img = ctx.positions(), ctx.images()
    ref_x, ref_n = wrap_coordinate(xyz[..., :2], big_l)

    w = pos[:, :n_atoms, :2]
    n = img[:, :n_atoms]
    np.testing.assert_array_equal(w, ref_x)
    np.testing.assert_array_equal(n, ref_n)
    assert (w >= 0).all() and (w < big_l).all(), "nicht in [0, L)"
    np.testing.assert_array_equal(pos[:, :n_atoms, 2], xyz[..., 2])  # z nie gewrappt
    np.testing.assert_array_equal(pos[:, :n_atoms, 3], q)
    assert not pos[:, n_atoms:].any() and not img[:, n_atoms:].any(), "Padding verändert"

    recon = w.astype(np.float64) + n * big_l.astype(np.float64)
    err = np.abs(recon - xyz[..., :2].astype(np.float64))
    exact = np.abs(xyz[..., :2]) >= big_l
    assert (err[exact] == 0).all(), "Rekonstruktion für |x| >= L nicht exakt"
    half_ulp = 0.5 * np.spacing(np.broadcast_to(big_l, err.shape))
    assert (err <= half_ulp).all()

    # Mehrschrittig: wiederholt verschieben und wrappen → Imagezähler akkumulieren.
    x_ref, n_ref = ref_x[..., 0].copy(), ref_n[..., 0].copy()
    for _ in range(50):
        _core.testing.shift_x(ctx, 0.7)
        ctx.wrap_xy()
        x_ref, dn = wrap_coordinate(x_ref + np.float32(0.7), big_l[..., 0])
        n_ref = n_ref + dn
    pos, img = ctx.positions(), ctx.images()
    np.testing.assert_array_equal(pos[:, :n_atoms, 0], x_ref)
    np.testing.assert_array_equal(img[:, :n_atoms, 0], n_ref)
    np.testing.assert_array_equal(img[:, :n_atoms, 1], ref_n[..., 1])


# --- Zusätzliche Tests der Infrastruktur ------------------------------------------------------


def test_force_plugin_pipeline():
    """compute_forces: clear → ForcePlugins → float; Padding-Atome erhalten keine Kraft."""
    ctx = _ctx(2, 40)
    _core.testing.add_uniform_force(ctx, 1.5, -2.0, 0.25)
    _core.testing.add_uniform_force(ctx, 0.5, 0.0, 0.0)
    assert ctx.num_forces == 2
    for _ in range(3):  # mehrfach: clear_forces muss zwischen den Aufrufen löschen
        ctx.compute_forces()
    f = ctx.forces()
    np.testing.assert_array_equal(f[:, :40, :3], np.broadcast_to([2.0, -2.0, 0.25], (2, 40, 3)))
    assert not f[:, 40:].any()
    assert not ctx.energies_fixed_point().any()


def test_padding_variants():
    """N_pad für Vielfache von 32 und Einzelatome; Padding bleibt nach allen Settern 0."""
    for n_atoms, n_pad in [(1, 32), (31, 32), (32, 32), (33, 64), (64, 64)]:
        ctx = _ctx(3, n_atoms)
        assert ctx.n_pad == n_pad
        ctx.set_charges(np.ones((3, n_atoms), np.float32))
        ctx.set_masses(np.ones((3, n_atoms), np.float32))
        assert not ctx.positions()[:, n_atoms:].any()
        assert not ctx.velocities()[:, n_atoms:].any()


def test_input_validation():
    ctx = _ctx(2, 10)
    with pytest.raises(ValueError):
        ctx.set_positions(np.zeros((2, 11, 3), np.float32))
    with pytest.raises(ValueError):
        ctx.set_masses(np.zeros((2, 10), np.float32))  # Masse 0
    with pytest.raises(ValueError):
        ctx.set_box(np.array([[1.0, -1.0], [1.0, 1.0]], np.float32))
    with pytest.raises(ValueError):
        ctx.set_positions(np.full((2, 10, 3), np.nan, np.float32))
    with pytest.raises(ValueError):
        _core.testing.accumulate_forces(ctx, np.array([64], np.int32), np.zeros((1, 3), np.float32))
    with pytest.raises(RuntimeError):
        ctx.wrap_xy()  # Box nicht gesetzt
    with pytest.raises(ValueError):
        _core.SimulationContext(0, 10)
    with pytest.raises(ValueError):
        _core.SimulationContext(1, 2**31 - 1)  # n_pad würde int überlaufen


def test_wrap_range_and_box_reset():
    """|x/L| >= 2^20 wird abgelehnt (FP32-Wrap dort ungenau); set_box setzt Imagezähler zurück."""
    ctx = _ctx(2, 10)
    box = np.full((2, 2), 3.0, np.float32)
    ctx.set_box(box)
    xyz = np.zeros((2, 10, 3), np.float32)
    xyz[1, 3, 0] = 1.0e10
    with pytest.raises(ValueError):
        ctx.set_positions(xyz)

    xyz[1, 3, 0] = 1.0e6  # |x/L| < 2^20 mit L = 3, ok
    ctx.set_positions(xyz)
    with pytest.raises(ValueError):
        ctx.set_box(np.full((2, 2), 0.5, np.float32))  # 1e6 / 0.5 > 2^20
    np.testing.assert_array_equal(ctx.box(), box)  # unverändert nach Fehler

    ctx.wrap_xy()
    assert ctx.images()[1, 3, 0] == 333333
    ctx.set_box(box)
    assert not ctx.images().any(), "set_box muss die Imagezähler zurücksetzen"
