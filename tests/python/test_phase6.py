"""Abnahmetests Phase 6: Optik-Observer (Dipol, FRET, Brechungsindex) und VoxelGridForce3D.

Siehe docs/ROADMAP.md, Phase 6, und docs/PHYSICS_MODEL.md §4.3, §6.
"""

import numpy as np
import observers_ref as oref
import pytest
from scipy import stats

from biooptomd.voxel import values_from_potential

pytestmark = pytest.mark.gpu

_core = pytest.importorskip("biooptomd._core")


def _ctx(n_rep, n, box=10.0, masses=12.0):
    ctx = _core.SimulationContext(n_rep, n)
    ctx.set_box(np.full((n_rep, 2), box, np.float32))
    ctx.set_masses(np.broadcast_to(np.asarray(masses, np.float32), (n_rep, n)).copy())
    return ctx


def _csr(groups):
    offsets = np.concatenate([[0], np.cumsum([len(g) for g in groups])]).astype(np.int32)
    return offsets, np.concatenate(groups).astype(np.int32)


# --- Dipol / FRET ---------------------------------------------------------------------------------


def test_dipole_static_geometry():
    """Bekannte Geometrien → exakte Werte für R_F, μ̂, θ und d (auch über die Boxgrenze)."""
    box, z_s = 10.0, 0.5
    pos = np.zeros((3, 8, 3), np.float32)
    # Gruppe 0..3 symmetrisch um (2, 3, 4) mit gleichen Massen; Achse 4 → 5.
    for r in range(3):
        pos[r, :4] = [[1.5, 3, 4], [2.5, 3, 4], [2, 2.5, 4], [2, 3.5, 4]]
    pos[0, 4], pos[0, 5] = [1, 1, 1], [1, 1, 2]  # μ̂ = ẑ, θ = 0
    pos[1, 4], pos[1, 5] = [1, 1, 1], [4, 5, 1]  # μ̂ = (0.6, 0.8, 0), θ = π/2
    pos[2, 4], pos[2, 5] = [0.5, 5, 5], [9.5, 5, 5]  # über die Boxgrenze: μ̂ = (−1, 0, 0)
    ctx = _ctx(3, 8, box)
    ctx.set_positions(pos)
    off, atoms = _csr([np.arange(4)] * 3)
    tr = ctx.add_dipole_tracker(
        off, atoms, np.full(3, 4, np.int32), np.full(3, 5, np.int32), z_surface=z_s
    )
    ctx.observe_now()
    f = tr.result()["frames"][:, 0]
    np.testing.assert_array_equal(f[:, :3], np.tile([2.0, 3.0, 4.0], (3, 1)))
    np.testing.assert_array_equal(f[:, 3:6], [[0, 0, 1], [0.6, 0.8, 0], [-1, 0, 0]])
    np.testing.assert_array_equal(f[:, 6], 4.0 - z_s)
    np.testing.assert_array_equal(np.arccos(f[:, 5]), [0.0, np.pi / 2, np.pi / 2])  # θ = acos(μ̂·ẑ)

    # Gruppe über die Boxgrenze nach wrap_xy: Schwerpunkt aus ungewrappten Koordinaten.
    rng = np.random.default_rng(60)
    p = rng.uniform(9.0, 11.0, (3, 8, 3)).astype(np.float32)
    ctx.set_positions(p)
    ctx.wrap_xy()
    ctx.observe_now()
    f = tr.result()["frames"][:, 1]
    for r in range(3):
        ref = oref.dipole_frame(
            p[r].astype(np.float64), np.full(8, 12.0), np.arange(4), 4, 5, (box, box), z_s
        )
        np.testing.assert_allclose(f[r], ref, rtol=0, atol=1e-12)


def _random_unit(rng, n):
    u = rng.normal(size=(n, 3))
    return u / np.linalg.norm(u, axis=1, keepdims=True)


def test_kappa2_isotropic():
    """Isotrop zufällige Donor-/Akzeptor-Dipole und Verbindungsrichtungen: ⟨κ²⟩ = 2/3 (4σ);
    jeder Einzelwert = NumPy-Referenz; E für R0 > 0."""
    rng = np.random.default_rng(61)
    n_rep, n_frames, r0 = 1024, 8, 5.0
    ctx = _ctx(n_rep, 4, box=20.0)
    off, atoms_d = _csr([np.array([0, 1])] * n_rep)
    _, atoms_a = _csr([np.array([2, 3])] * n_rep)
    tr = ctx.add_fret_tracker(
        off, atoms_d, np.zeros(n_rep, np.int32), np.ones(n_rep, np.int32),
        off, atoms_a, np.full(n_rep, 2, np.int32), np.full(n_rep, 3, np.int32), r0=r0, capacity=n_frames,
    )  # fmt: skip
    refs = []
    for _ in range(n_frames):
        mu_d, mu_a, r_hat = (
            _random_unit(rng, n_rep),
            _random_unit(rng, n_rep),
            _random_unit(rng, n_rep),
        )
        c_d = rng.uniform(8.0, 12.0, (n_rep, 3))
        dist = rng.uniform(2.0, 6.0, n_rep)
        c_a = c_d + dist[:, None] * r_hat
        pos = np.stack(
            [c_d - 0.05 * mu_d, c_d + 0.05 * mu_d, c_a - 0.05 * mu_a, c_a + 0.05 * mu_a], 1
        )
        pos = pos.astype(np.float32)
        ctx.set_positions(pos)
        ctx.observe_now()
        p = pos.astype(np.float64)
        md, ma = p[:, 1] - p[:, 0], p[:, 3] - p[:, 2]
        md /= np.linalg.norm(md, axis=1, keepdims=True)
        ma /= np.linalg.norm(ma, axis=1, keepdims=True)
        rv = 0.5 * (p[:, 2] + p[:, 3]) - 0.5 * (p[:, 0] + p[:, 1])
        refs.append((np.linalg.norm(rv, axis=1), oref.kappa2(md, ma, rv)))
    res = tr.result()
    r_ref = np.stack([a for a, _ in refs], 1)
    k_ref = np.stack([b for _, b in refs], 1)
    np.testing.assert_allclose(res["r_da"], r_ref, rtol=1e-12)
    np.testing.assert_allclose(res["kappa2"], k_ref, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(res["efficiency"], oref.fret_efficiency(r_ref, k_ref, r0), rtol=1e-9)
    k2 = res["kappa2"].ravel()
    sigma = k2.std(ddof=1) / np.sqrt(k2.size)
    assert abs(k2.mean() - 2.0 / 3.0) < 4.0 * sigma, (k2.mean(), sigma)


def test_ring_buffer():
    """Kapazität 3, 5 Frames: die letzten 3 in zeitlicher Reihenfolge, 2 überschrieben; reset()."""
    ctx = _ctx(2, 4)
    ctx.set_positions(np.zeros((2, 4, 3), np.float32) + [1.0, 1.0, 1.0])
    off, atoms = _csr([np.array([0, 1])] * 2)
    tr = ctx.add_dipole_tracker(
        off, atoms, np.zeros(2, np.int32), np.full(2, 2, np.int32), stride=1, capacity=3
    )
    pos = np.zeros((2, 4, 3), np.float32)
    for k in range(5):
        pos[:, 0] = [1.0 + k, 1.0, 1.0]
        pos[:, 1] = [1.0 + k, 1.0, 1.0]
        pos[:, 2] = [1.0 + k, 1.0, 2.0]
        ctx.set_positions(pos)
        ctx.observe_now()
    res = tr.result()
    assert tr.overwritten == 2 and res["frames"].shape == (2, 3, 7)
    np.testing.assert_array_equal(res["frames"][0, :, 0], [3.0, 4.0, 5.0])
    tr.reset()
    assert tr.result()["frames"].shape == (2, 0, 7)


# --- Brechungsindex ------------------------------------------------------------------------------


def test_refractive_index_pure_water():
    """Keine Atome im Profilbereich → n(z) = n_w exakt."""
    ctx = _ctx(2, 20)
    pos = np.full((2, 20, 3), 5.0, np.float32)
    pos[..., 2] = 9.0  # oberhalb des Profils (0 … 5 nm)
    ctx.set_positions(pos)
    prof = ctx.add_refractive_index_profiler(
        0.1, 50, np.array([1e-3], np.float32), np.array([0.02], np.float32)
    )
    ctx.observe_now()
    ctx.observe_now()
    res = prof.result()
    np.testing.assert_array_equal(res["n"], float(np.float32(1.333)))
    assert not res["s_alpha_mean"].any() and not res["s_volume_mean"].any()
    assert res["n_frames"][0] == 2


def test_refractive_index_uniform_layer():
    """Homogene Atomschicht bekannter Dichte (Gitter, je Schicht gleich viele Atome) →
    analytisches Lorentz-Lorenz-Ergebnis, rel. Fehler < 1e-4; außerhalb n = n_w; Varianz 0."""
    box, dz, a_lat = 4.0, 0.1, 0.25
    xy = (np.arange(16) + 0.5) * a_lat
    zs = 1.0 + (np.arange(10) + 0.5) * dz  # Schichten 10 … 19
    gx, gy, gz = np.meshgrid(xy, xy, zs, indexing="ij")
    pos = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], 1).astype(np.float32)
    n = len(pos)
    alpha, vol, n_w = np.float32(1e-4), np.float32(2e-3), np.float32(1.333)
    ctx = _ctx(1, n, box)
    ctx.set_positions(pos[None])
    prof = ctx.add_refractive_index_profiler(
        dz, 40, np.array([alpha]), np.array([vol]), n_water=n_w
    )
    for _ in range(3):
        ctx.observe_now()
    res = prof.result()
    per_layer = 16 * 16
    v_k = box * box * dz
    expected = oref.lorentz_lorenz(
        per_layer * float(alpha), per_layer * float(vol), v_k, float(n_w)
    )
    layer = (np.arange(40) >= 10) & (np.arange(40) < 20)
    np.testing.assert_allclose(res["n"][0, layer], expected, rtol=1e-4)
    np.testing.assert_array_equal(res["n"][0, ~layer], float(n_w))
    np.testing.assert_allclose(res["s_alpha_mean"][0, layer], per_layer * float(alpha), rtol=1e-6)
    assert res["s_alpha_var"].max() < 1e-12 * (per_layer * float(alpha)) ** 2
    np.testing.assert_allclose(res["z"][:3], [0.05, 0.15, 0.25], rtol=1e-6)

    # Gauß-Verschmierung erhält die Gesamtsumme (Schicht weit innerhalb des Profils).
    ctx2 = _ctx(1, n, box)
    ctx2.set_positions(pos[None])
    sm = ctx2.add_refractive_index_profiler(
        dz, 40, np.array([alpha]), np.array([vol]), smear_sigma=0.05
    )
    ctx2.observe_now()
    res2 = sm.result()
    s = res2["s_alpha_mean"][0]
    assert s.sum() == pytest.approx(n * float(alpha), rel=1e-5)
    ref_a, ref_v = oref.layer_histograms_smeared(
        pos[:, 2],
        np.full(n, float(alpha)),
        np.full(n, float(vol)),
        0.0,
        dz,
        40,
        float(np.float32(0.05)),
    )
    # Fixed-Point hat eine absolute Auflösung von 2^-32 je Beitrag; in den äußersten Tail-Schichten
    # (Beiträge ~1e-7 nm³) dominiert sie den relativen Fehler ⇒ absolute Toleranz relativ zur Schicht.
    np.testing.assert_allclose(s, ref_a, rtol=1e-5, atol=1e-6 * ref_a.max())
    np.testing.assert_allclose(res2["s_volume_mean"][0], ref_v, rtol=1e-5, atol=1e-6 * ref_v.max())
    assert s[9] > 0 and s[20] > 0  # Beiträge in die Nachbarschichten


def test_observer_ignores_padding():
    """Padding-Atome (bei z = 0, Optik-Typ 0 mit α′ > 0) tragen zu keinem Observer bei;
    Gruppen mit Padding-Indizes werden abgelehnt."""
    n = 33  # n_pad = 64
    ctx = _ctx(2, n)
    pos = np.full((2, n, 3), 5.0, np.float32)
    pos[..., 2] = 6.0
    ctx.set_positions(pos)
    prof = ctx.add_refractive_index_profiler(
        0.1, 20, np.array([1e-3], np.float32), np.array([0.02], np.float32), z_min=-1.0
    )
    ctx.observe_now()
    res = prof.result()
    assert not res["s_alpha_mean"].any(), "Padding-Atome im Profil gezählt"
    np.testing.assert_array_equal(res["n"], float(np.float32(1.333)))
    off, atoms = _csr([np.array([0, 40])] * 2)  # 40 ist ein Padding-Platz
    with pytest.raises(ValueError):
        ctx.add_dipole_tracker(off, atoms, np.zeros(2, np.int32), np.ones(2, np.int32))
    with pytest.raises(ValueError):
        off, atoms = _csr([np.array([0, 1])] * 2)
        ctx.add_dipole_tracker(off, atoms, np.zeros(2, np.int32), np.full(2, 50, np.int32))


# --- VoxelGridForce3D ------------------------------------------------------------------------------


def _linear_grid(a, c, shape=(17, 17, 17), origin=(0.5, 0.5, 0.5), spacing=0.125):
    nz, ny, nx = shape
    z, y, x = np.meshgrid(
        *(origin[d] + np.arange(n) * spacing for d, n in ((2, nz), (1, ny), (0, nx))), indexing="ij"
    )
    v = a[0] * x + a[1] * y + a[2] * z + c
    return values_from_potential(v, (spacing,) * 3)


def test_voxel_linear_field():
    """Lineares Potential → konstante Kraft −s·∇V exakt (auch mit Typ-Skalierung), Energie linear;
    außerhalb des Gitters keine Kraft. Harmonisches Potential in z → Boltzmann-Verteilung (χ²)."""
    a, c = np.array([1.5, -2.0, 0.75]), 0.25
    values = _linear_grid(a, c)
    rng = np.random.default_rng(62)
    n = 200
    pos = rng.uniform(0.55, 2.45, (1, n, 3)).astype(np.float32)
    pos[0, -5:, 2] = 4.0  # außerhalb (z > 0.5 + 16·0.125 = 2.5)
    ctx = _ctx(1, n)
    ctx.set_positions(pos)
    types = np.zeros((1, n, 2), np.int32)
    types[0, ::2, 1] = 1
    ctx.set_types(types)
    scale = np.array([1.0, 2.5], np.float32)
    ctx.add_voxel_grid(values, (0.5, 0.5, 0.5), (0.125, 0.125, 0.125), scale)
    ctx.compute_forces()
    f = ctx.forces()[0, :n, :3]
    s = scale[types[0, :, 1]][:, None]
    np.testing.assert_array_equal(f[:-5], (-s * a.astype(np.float32))[:-5])
    assert not f[-5:].any()
    e_ref = (s[:-5, 0] * (pos[0, :-5].astype(np.float64) @ a + c)).sum()
    assert ctx.energies()[0, 8] == pytest.approx(e_ref, rel=1e-6)  # Spalte "external"
    assert ctx.status() == 0

    # Harmonisches Potential V = ½k(z − z0)² (Gradient linear in z ⇒ Interpolation exakt):
    # unabhängige Atome, Verteilung in z gegen Boltzmann.
    k, z0, temp = 50.0, 3.0, 300.0
    nz = 41
    zg = 1.0 + np.arange(nz) * 0.1
    pot = np.broadcast_to((0.5 * k * (zg - z0) ** 2)[:, None, None], (nz, 8, 8))
    grid = values_from_potential(pot, (1.25, 1.25, 0.1), periodic_xy=True)
    n2 = 2000
    ctx = _ctx(2, n2, box=10.0, masses=20.0)
    ctx.set_temperatures(np.full(2, temp, np.float32))
    p2 = rng.uniform(0.0, 10.0, (2, n2, 3)).astype(np.float32)
    p2[..., 2] = z0
    ctx.set_positions(p2)
    ctx.add_voxel_grid(
        grid, (0.0, 0.0, 1.0), (1.25, 1.25, 0.1), np.array([1.0], np.float32), periodic_xy=True
    )
    ctx.set_integrator(0.002, 2.0, seed=3)
    ctx.step(2500)
    zs = []
    for _ in range(30):
        ctx.step(1500)
        zs.append(ctx.positions()[:, :n2, 2].ravel().astype(np.float64))
    zs = np.concatenate(zs)
    sigma = np.sqrt(_core.units.KB * temp / k)
    edges = np.linspace(z0 - 4.5 * sigma, z0 + 4.5 * sigma, 46)
    observed, _ = np.histogram(zs, edges)
    expected = np.diff(stats.norm.cdf(edges, loc=z0, scale=sigma))
    expected = expected / expected.sum() * observed.sum()
    keep = expected >= 5
    p = stats.chisquare(
        observed[keep], expected[keep] / expected[keep].sum() * observed[keep].sum()
    ).pvalue
    assert p > 1e-3, f"χ² p = {p:.2e}"


def test_voxel_vs_numpy_interpolation():
    """Zufallsfeld (auch periodisch in x/y): Kraft und Energie = NumPy-Trilinear-Referenz."""
    rng = np.random.default_rng(63)
    n = 300
    # Anisotrop (nx ≠ ny ≠ nz, Δx ≠ Δy), damit vertauschte Achsen auffallen.
    cases = {False: ((5, 7, 9), (0.45, 0.6, 0.5)), True: ((6, 10, 8), (0.5, 0.4, 0.35))}
    for periodic, (shape, spacing) in cases.items():
        values = rng.normal(size=shape + (4,)).astype(np.float32)
        origin = (0.0, 0.0, 0.2) if periodic else (0.3, 0.1, 0.2)
        ctx = _ctx(1, n, box=4.0)
        pos = rng.uniform(0.0, 4.0, (1, n, 3)).astype(np.float32)
        pos[..., 2] = rng.uniform(0.2, 2.2, n)
        ctx.set_positions(pos)
        ctx.add_voxel_grid(
            values, origin, spacing, np.array([1.0], np.float32), periodic_xy=periodic
        )
        ctx.compute_forces()
        f = ctx.forces()[0, :n, :3].astype(np.float64)
        e = 0.0
        for i in range(n):
            ref = oref.trilinear(
                values.astype(np.float64), origin, spacing, pos[0, i].astype(np.float64), periodic
            )
            if ref is None:
                assert not f[i].any()
                continue
            np.testing.assert_allclose(f[i], -ref[1:], rtol=1e-5, atol=1e-5)
            e += ref[0]
        assert ctx.energies()[0, 8] == pytest.approx(e, rel=1e-5, abs=1e-4)


def test_fret_minimum_image():
    """Donor und Akzeptor beiderseits der Boxgrenze: R_DA mit Minimum Image (1 nm statt L − 1)."""
    ctx = _ctx(1, 4, box=10.0)
    pos = np.array([[[0.4, 5, 5], [0.6, 5, 5], [9.4, 5, 5], [9.6, 5, 5]]], np.float32)
    ctx.set_positions(pos)
    off, d_atoms = _csr([np.array([0, 1])])
    _, a_atoms = _csr([np.array([2, 3])])
    one = np.ones(1, np.int32)
    tr = ctx.add_fret_tracker(off, d_atoms, 0 * one, one, off, a_atoms, 2 * one, 3 * one)
    ctx.observe_now()
    res = tr.result()
    assert res["r_da"][0, 0] == pytest.approx(1.0, abs=1e-6)
    assert res["kappa2"][0, 0] == pytest.approx(4.0, abs=1e-9)  # kollinear: (1 − 3)² = 4


def _voxel_mismatch(h, n_rep=256, seed=64):
    """RMS-Abweichung zwischen GPU-Kraft (interpolierter Gradient) und −∇ der GPU-Energie
    (finite Differenzen) für V = 3 sin(2x) cos(1.5y) + 2 (z − 1)² auf einem Gitter mit Abstand h."""
    n_pts = round(2.0 / h) + 1
    z, y, x = np.meshgrid(*(np.arange(n_pts) * h for _ in range(3)), indexing="ij")
    pot = 3.0 * np.sin(2.0 * x) * np.cos(1.5 * y) + 2.0 * (z - 1.0) ** 2
    exact = np.stack(
        [
            pot,
            6.0 * np.cos(2 * x) * np.cos(1.5 * y),
            -4.5 * np.sin(2 * x) * np.sin(1.5 * y),
            4.0 * (z - 1.0),
        ],
        -1,
    ).astype(np.float32)  # analytischer Gradient an den Gitterpunkten
    rng = np.random.default_rng(seed)
    p0 = rng.uniform(0.2, 1.8, (n_rep, 1, 3))
    ctx = _ctx(n_rep, 1)
    ctx.add_voxel_grid(exact, (0.0, 0.0, 0.0), (h, h, h), np.array([1.0], np.float32))

    def eval_at(p):
        ctx.set_positions(p.astype(np.float32))
        ctx.compute_forces()
        return ctx.energies()[:, 8].copy(), ctx.forces()[:, 0, :3].astype(np.float64)

    _, f = eval_at(p0)
    f_fd = np.zeros_like(f)
    step = 1e-3 * h
    for d in range(3):
        dp = np.zeros(3)
        dp[d] = step
        e_plus, _ = eval_at(p0 + dp)
        e_minus, _ = eval_at(p0 - dp)
        f_fd[:, d] = -(e_plus - e_minus) / (2 * step)
    return np.sqrt(((f - f_fd) ** 2).sum() / (f_fd**2).sum())


def test_voxel_energy_force_consistency():
    """Quantifiziert die Energienicht-Erhaltung (PHYSICS_MODEL §4.3): Der interpolierte Gradient weicht
    vom Gradienten des (zellweise linearen) interpolierten Potentials in O(Δ) ab. Gemessen für ein
    glattes Feld (Wellenzahl 2/nm): ≈ 5 % RMS bei Δ = 0.1 nm, Halbierung bei Δ = 0.05 nm."""
    m1 = _voxel_mismatch(0.1)
    m2 = _voxel_mismatch(0.05)
    assert 0.02 < m1 < 0.1, m1
    assert 1.6 < m1 / m2 < 2.4, (m1, m2)  # erste Ordnung in Δ


def test_observer_and_voxel_guards():
    """Schwerpunkt ohne Massen → STATUS_NON_FINITE; Boxänderung nach dem Hinzufügen wird bei
    periodischem Voxelgitter und beim Profiler (ohne reset) erkannt."""
    ctx = _core.SimulationContext(1, 4)
    ctx.set_box(np.full((1, 2), 4.0, np.float32))
    ctx.set_positions(np.full((1, 4, 3), 1.0, np.float32) + np.arange(4)[None, :, None])
    off, atoms = _csr([np.array([0, 1])])
    ctx.add_dipole_tracker(off, atoms, np.zeros(1, np.int32), np.ones(1, np.int32))
    ctx.observe_now()
    assert ctx.status() & _core.STATUS_NON_FINITE

    ctx = _ctx(1, 4, box=4.0)
    ctx.set_positions(np.full((1, 4, 3), 1.0, np.float32))
    grid = np.zeros((4, 8, 8, 4), np.float32)
    ctx.add_voxel_grid(
        grid, (0.0, 0.0, 0.0), (0.5, 0.5, 0.5), np.array([1.0], np.float32), periodic_xy=True
    )
    prof = ctx.add_refractive_index_profiler(
        0.1, 10, np.array([1e-3], np.float32), np.array([0.0], np.float32)
    )
    ctx.compute_forces()
    ctx.observe_now()
    ctx.set_box(np.full((1, 2), 5.0, np.float32))
    with pytest.raises(ValueError):
        ctx.compute_forces()  # Gitter deckt die neue Box nicht mehr ab
    with pytest.raises(RuntimeError):
        ctx.observe_now()  # Profiler: Box seit reset() geändert
    prof.reset()
