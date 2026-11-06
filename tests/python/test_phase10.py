"""Abnahmetests Phase 10: selbstkonsistente Kopplung MD ↔ FEM (ADR 0016).

10.1 ``PolarizabilityGridObserver``, 10.2 ``VoxelGridForce3D.set_values``.
Siehe docs/ROADMAP.md, Phase 10, und docs/adr/0016-md-fem-coupling.md.
"""

import numpy as np
import observers_ref as oref
import pytest

from biooptomd.voxel import values_from_potential

pytestmark = pytest.mark.gpu

_core = pytest.importorskip("biooptomd._core")

N_W = 1.333
ALPHA = np.array([1.2e-3, 0.4e-3, 2.0e-3], np.float32)
VOLUME = np.array([0.012, 0.004, 0.0], np.float32)  # Typ 2 ohne Volumen ⇒ Punkt


def _ctx(n_rep, n, box=(4.0, 4.0), masses=12.0):
    ctx = _core.SimulationContext(n_rep, n)
    ctx.set_box(np.broadcast_to(np.asarray(box, np.float32), (n_rep, 2)).copy())
    ctx.set_masses(np.broadcast_to(np.asarray(masses, np.float32), (n_rep, n)).copy())
    return ctx


def _system(n=300, seed=3, box=4.0, z_range=(0.8, 4.2)):
    rng = np.random.default_rng(seed)
    pos = np.empty((n, 3))
    pos[:, :2] = rng.uniform(-0.5 * box, 1.5 * box, (n, 2))  # auch außerhalb der Grundzelle
    pos[:, 2] = rng.uniform(*z_range, n)
    types = rng.integers(0, len(ALPHA), n).astype(np.int32)
    return pos.astype(np.float32), types


def _set(ctx, pos, types):
    ctx.set_positions(pos[None] if pos.ndim == 2 else pos)
    t = np.zeros((ctx.n_replicas, pos.shape[-2], 2), np.int32)
    t[..., 1] = types
    ctx.set_types(t)


def _lorentz_lorenz_eps(s_a, s_v, vk, n_w):
    f_w = np.maximum(0.0, 1.0 - s_v / vk)
    lw = (n_w**2 - 1.0) / (n_w**2 + 2.0)
    big_l = 4.0 * np.pi / 3.0 * s_a / vk + f_w * lw
    return (1.0 + 2.0 * big_l) / (1.0 - big_l)


# --- 10.1 PolarizabilityGridObserver -----------------------------------------------------------


@pytest.mark.parametrize("periodic", [True, False])
@pytest.mark.parametrize("smearing", ["gauss", "point"])
def test_polarizability_grid_vs_numpy(periodic, smearing):
    """Histogramme S_α, S_V je Voxel, Mittel und Varianz über zwei Frames sowie ε gegen die
    NumPy-Referenz (rel. 1e-6; absolute Toleranz 1e-6 des Maximums für Gauß-Ausläufer an der
    Fixed-Point-Auflösung). Nicht periodisch: Gitter kleiner als die Box, Anteile außerhalb fallen
    weg."""
    box = 4.0
    pos1, types = _system(box=box)
    pos2, _ = _system(box=box, seed=4)
    if periodic:
        shape, origin, spacing = (16, 16, 20), (0.0, 0.0, 0.5), (0.25, 0.25, 0.2)
    else:
        shape, origin, spacing = (10, 12, 15), (0.7, 0.3, 1.0), (0.25, 0.2, 0.2)
    ctx = _ctx(1, len(types))
    obs = ctx.add_polarizability_grid(
        shape, origin, spacing, ALPHA, VOLUME, n_water=N_W, smearing=smearing,
        periodic_xy=periodic,
    )  # fmt: skip
    refs = []
    for pos in (pos1, pos2):
        _set(ctx, pos, types)
        ctx.observe_now()
        refs.append(
            oref.grid_histograms(
                pos,
                ALPHA[types].astype(np.float64),
                VOLUME[types].astype(np.float64),
                origin,
                spacing,
                shape,
                periodic_xy=periodic,
                point=smearing == "point",
            )
        )
    res = obs.result()
    assert res["n_frames"][0] == 2 and obs.n_frames == 2
    assert res["eps"].shape == (1, shape[2], shape[1], shape[0])
    a = np.stack([r[0] for r in refs])
    v = np.stack([r[1] for r in refs])
    for key, ref in (
        ("s_alpha_mean", a.mean(0)), ("s_volume_mean", v.mean(0)),
        ("s_alpha_var", a.var(0)), ("s_volume_var", v.var(0)),
    ):  # fmt: skip
        np.testing.assert_allclose(res[key][0], ref, rtol=1e-6, atol=1e-6 * np.abs(ref).max(),
                                   err_msg=key)  # fmt: skip
    vk = float(np.prod(spacing))
    eps_ref = _lorentz_lorenz_eps(a.mean(0), v.mean(0), vk, N_W)
    np.testing.assert_allclose(res["eps"][0], eps_ref, rtol=1e-6)
    np.testing.assert_allclose(res["x"], origin[0] + (np.arange(shape[0]) + 0.5) * spacing[0],
                               rtol=1e-6)  # fmt: skip
    if periodic:  # nichts geht verloren (Gauß-Fenster ±4σ liegt in z im Gitter)
        assert a[0].sum() == pytest.approx(ALPHA[types].astype(np.float64).sum(), rel=1e-6)
    assert res["s_alpha_mean"][0].sum() > 0


def test_polarizability_grid_matches_profiler():
    """Die Summe über x/y reproduziert S_α(z), S_V(z) des 1D-Profilers mit Gauß-Verschmierung
    (rel. 1e-5), bei gleichem Δz und z_min; zwei Replicas mit verschiedenen Konfigurationen."""
    box, dz, z_min, nz = 4.0, 0.2, 0.5, 20
    pos_a, types = _system(seed=5)
    pos_b, _ = _system(seed=6)
    pos = np.stack([pos_a, pos_b])
    ctx = _ctx(2, len(types))
    grid = ctx.add_polarizability_grid(
        (16, 16, nz), (0.0, 0.0, z_min), (box / 16, box / 16, dz), ALPHA, VOLUME, n_water=N_W,
        periodic_xy=True,
    )  # fmt: skip
    prof = ctx.add_refractive_index_profiler(
        dz, nz, ALPHA, VOLUME, z_min=z_min, n_water=N_W, smearing="gauss"
    )
    _set(ctx, pos, types)
    ctx.observe_now()
    g, p = grid.result(), prof.result()
    for key in ("s_alpha_mean", "s_volume_mean"):
        layer = g[key].sum(axis=(2, 3))
        ref = p[key]
        np.testing.assert_allclose(layer, ref, rtol=1e-5, atol=1e-6 * ref.max(), err_msg=key)
    assert not np.array_equal(g["s_alpha_mean"][0], g["s_alpha_mean"][1])


def test_polarizability_grid_deterministic():
    """Zwei identische Läufe liefern bitgleiche Ergebnisse (Fixed-Point, ADR 0002); reines
    Lösungsmittel ergibt ε = n_w² exakt; ungültige Parameter werfen."""
    pos, types = _system(n=500, seed=8)
    out = []
    for _ in range(2):
        ctx = _ctx(1, len(types))
        obs = ctx.add_polarizability_grid(
            (16, 16, 20), (0.0, 0.0, 0.5), (0.25, 0.25, 0.2), ALPHA, VOLUME, periodic_xy=True
        )
        _set(ctx, pos, types)
        for _ in range(3):
            ctx.observe_now()
        out.append(obs.result())
    for key in ("eps", "s_alpha_mean", "s_alpha_var", "s_volume_mean", "s_volume_var"):
        np.testing.assert_array_equal(out[0][key], out[1][key], err_msg=key)
    eps = out[0]["eps"][0]
    empty = (out[0]["s_alpha_mean"][0] == 0) & (out[0]["s_volume_mean"][0] == 0)
    assert empty.any()
    np.testing.assert_array_equal(eps[empty], float(np.float32(N_W)) ** 2)  # n_w als float32

    ctx = _ctx(1, 4)
    with pytest.raises(ValueError):
        ctx.add_polarizability_grid((4, 4, 4), (0, 0, 0), (0.25,) * 3, ALPHA, VOLUME,
                                    smearing="sphere")  # fmt: skip
    with pytest.raises(ValueError):  # Voxel zu klein für das Gauß-Fenster
        ctx.add_polarizability_grid((4, 4, 4), (0, 0, 0), (0.01,) * 3, ALPHA, VOLUME)
    with pytest.raises(ValueError):  # periodic_xy, Gitter deckt die Box nicht ab
        ctx.add_polarizability_grid((10, 10, 4), (0, 0, 0), (0.25,) * 3, ALPHA, VOLUME,
                                    periodic_xy=True)  # fmt: skip


# --- 10.2 VoxelGridForce3D.set_values -------------------------------------------------------------


def _field(shape, spacing, a, g):
    nz, ny, nx = shape
    zz, yy, xx = np.meshgrid(*(spacing * np.arange(m) for m in (nz, ny, nx)), indexing="ij")
    w = a + g[0] * np.sin(xx) + g[1] * yy * zz + g[2] * zz**2
    return values_from_potential(w, (spacing,) * 3)


def test_voxel_set_values():
    """``set_values`` ergibt bitgleich dieselben Kräfte, Energien und Trajektorien wie ein neu
    angelegtes Gitter mit denselben Werten; die Kräfte werden vor dem nächsten Schritt neu
    berechnet; falsche Shape wirft."""
    shape, spacing = (12, 13, 14), 0.4
    old = _field(shape, spacing, 30.0, (2.0, -1.0, 0.5))
    new = _field(shape, spacing, 45.0, (-3.0, 0.7, 1.5))
    pos, types = _system(n=200, seed=9, box=4.0, z_range=(0.5, 4.0))
    pos[:, :2] = np.mod(pos[:, :2], 4.0)
    scale = np.array([-0.02, 0.01, -0.005], np.float32)

    def make(values):
        ctx = _ctx(1, len(types), box=(5.2, 5.6))
        _set(ctx, pos, types)
        grid = ctx.add_voxel_grid(values, (0.0, 0.0, 0.0), (spacing,) * 3, scale)
        ctx.set_integrator(0.002, 1.0, 17)
        return ctx, grid

    ctx_a, grid_a = make(old)
    ctx_a.compute_forces()
    f_old = ctx_a.forces().copy()
    grid_a.set_values(new)
    ctx_b, _ = make(new)
    ctx_a.compute_forces()
    ctx_b.compute_forces()
    np.testing.assert_array_equal(ctx_a.forces(), ctx_b.forces())
    np.testing.assert_array_equal(ctx_a.energies(), ctx_b.energies())
    assert not np.array_equal(f_old, ctx_a.forces())

    # set_values zwischen Schritten: step() rechnet die Kräfte vor dem ersten B neu.
    ctx_c, grid_c = make(old)
    ctx_d, _ = make(new)
    ctx_c.step(5)
    ctx_d.step(5)  # gleicher Seed, aber anderes Feld ⇒ nur Vergleich nach dem Umschalten
    ctx_e, grid_e = make(old)
    ctx_e.step(5)
    np.testing.assert_array_equal(ctx_c.positions(), ctx_e.positions())
    grid_c.set_values(new)
    grid_e.set_values(new)
    ctx_c.step(5)
    ctx_e.step(5)
    np.testing.assert_array_equal(ctx_c.positions(), ctx_e.positions())
    assert not np.array_equal(ctx_c.positions(), ctx_d.positions())

    with pytest.raises(ValueError):
        grid_a.set_values(new[:-1])
    with pytest.raises(ValueError):
        grid_a.set_values(np.full_like(new, np.nan))


def test_polarizability_grid_padding_and_unknown_type():
    """Padding-Atome (bei (0, 0, 0), n kein Vielfaches der Blockgröße) tragen nichts bei, obwohl das
    Gitter den Ursprung abdeckt (AGENTS.md Regel 6); ein unbekannter Optik-Typ setzt
    STATUS_UNKNOWN_TYPE und trägt nichts bei."""
    pos, types = _system(n=37, seed=12, z_range=(0.3, 1.5))
    shape, origin, spacing = (16, 16, 10), (0.0, 0.0, -0.5), (0.25, 0.25, 0.25)
    ctx = _ctx(1, len(types))
    obs = ctx.add_polarizability_grid(shape, origin, spacing, ALPHA, VOLUME, periodic_xy=True)
    _set(ctx, pos, types)
    ctx.observe_now()
    ref_a, ref_v = oref.grid_histograms(
        pos, ALPHA[types].astype(np.float64), VOLUME[types].astype(np.float64), origin, spacing,
        shape, periodic_xy=True,
    )  # fmt: skip
    res = obs.result()
    np.testing.assert_allclose(res["s_alpha_mean"][0], ref_a, rtol=1e-6, atol=1e-6 * ref_a.max())
    np.testing.assert_allclose(res["s_volume_mean"][0], ref_v, rtol=1e-6, atol=1e-6 * ref_v.max())
    assert ctx.status() == 0

    bad = types.copy()
    bad[0] = len(ALPHA)  # unbekannter Typ
    ctx2 = _ctx(1, len(types))
    obs2 = ctx2.add_polarizability_grid(shape, origin, spacing, ALPHA, VOLUME, periodic_xy=True)
    _set(ctx2, pos, bad)
    ctx2.observe_now()
    assert ctx2.status() & _core.STATUS_UNKNOWN_TYPE
    keep = np.arange(len(types)) != 0
    ref_a2, _ = oref.grid_histograms(
        pos[keep], ALPHA[types[keep]].astype(np.float64), VOLUME[types[keep]].astype(np.float64),
        origin, spacing, shape, periodic_xy=True,
    )  # fmt: skip
    np.testing.assert_allclose(obs2.result()["s_alpha_mean"][0], ref_a2, rtol=1e-6,
                               atol=1e-6 * ref_a2.max())  # fmt: skip
