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


# --- 10.3 biooptomd.coupling: TMM-Referenz -------------------------------------------------------

LAM = 632.8  # nm
N_PRISM = 1.515  # BK7
EPS_W633 = 1.3317**2
# ASSUMPTION(OQ-25): ε_Au(633 nm) = −11.6 + 1.2i (gängiger Tabellenwert, Größenordnung Johnson & Christy);
# für die Tests nur als typisches Metall, nicht als Materialreferenz.
EPS_AU = -11.6 + 1.2j


def test_tmm_reference():
    """TMM-Referenz (ADR 0016): (1) Fresnel-Koeffizienten einer Grenzfläche analytisch (rel. 1e-12),
    s und p; (2) Drei-Schicht-Airy-Formel r = (r₁₂ + r₂₃ e^{2iβ})/(1 + r₁₂ r₂₃ e^{2iβ}) für
    Prisma | Gold 50 nm | Wasser (rel. 1e-12, inkl. Resonanz); (3) ohne Absorption R + T = 1
    (Vielschicht, beide Polarisationen, auch Totalreflexion); (4) Feld im Wasser klingt evaneszent
    mit exp(−2 Im k_z Δz) ab, |E|² für s stetig an Grenzflächen; (5) SPR-Minimum (p) innerhalb 1° der
    Dispersion des Oberflächenplasmons der dicken Schicht, Tiefe R < 0.05."""
    from biooptomd.coupling import tmm

    k0 = 2 * np.pi / LAM

    def kz(e, kx):
        v = np.sqrt(k0**2 * complex(e) - kx**2 + 0j)
        return -v if v.imag < 0 or (v.imag == 0 and v.real < 0) else v

    def r_if(e1, e2, kx, pol):
        k1, k2 = kz(e1, kx), kz(e2, kx)
        if pol == "s":
            return (k1 - k2) / (k1 + k2)
        return (e2 * k1 - e1 * k2) / (e2 * k1 + e1 * k2)

    for pol in ("s", "p"):
        for deg in (0.0, 20.0, 45.0, 70.0):
            th = np.deg2rad(deg)
            kx = k0 * N_PRISM * np.sin(th)
            res = tmm([N_PRISM**2, EPS_W633], [], LAM, th, pol)
            ref = r_if(N_PRISM**2, EPS_W633, kx, pol)
            assert abs(res.r - ref) <= 1e-12 * max(abs(ref), 1e-3), (pol, deg)
            assert res.R + res.T == pytest.approx(1.0, abs=1e-12) or deg == 70.0
        for deg in np.linspace(40.0, 80.0, 41):
            th = np.deg2rad(deg)
            kx = k0 * N_PRISM * np.sin(th)
            d = 50.0
            beta = kz(EPS_AU, kx) * d
            r12, r23 = r_if(N_PRISM**2, EPS_AU, kx, pol), r_if(EPS_AU, EPS_W633, kx, pol)
            airy = (r12 + r23 * np.exp(2j * beta)) / (1 + r12 * r23 * np.exp(2j * beta))
            res = tmm([N_PRISM**2, EPS_AU, EPS_W633], [d], LAM, th, pol)
            assert abs(res.r - airy) <= 1e-12 * abs(airy), (pol, deg)

    # (3) verlustfreie Vielschicht
    eps = [N_PRISM**2, 2.1, 1.9, 2.4, 1.5, EPS_W633]
    thick = [80.0, 35.0, 120.0, 10.0]
    for pol in ("s", "p"):
        for deg in (0.0, 30.0, 55.0, 75.0):
            res = tmm(eps, thick, LAM, np.deg2rad(deg), pol)
            assert res.R + res.T == pytest.approx(1.0, abs=1e-12), (pol, deg)
            if deg == 75.0:  # Totalreflexion ins Wasser
                assert res.R == pytest.approx(1.0, abs=1e-12)

    # (4) evaneszentes Feld und Stetigkeit
    th = np.deg2rad(72.0)
    res = tmm([N_PRISM**2, EPS_AU, EPS_W633], [50.0], LAM, th, "p")
    im_kz = kz(EPS_W633, k0 * N_PRISM * np.sin(th)).imag
    z = np.array([55.0, 65.0])
    i1, i2 = res.intensity(z)
    assert i2 / i1 == pytest.approx(np.exp(-2 * im_kz * 10.0), rel=1e-10)
    res_s = tmm(eps, thick, LAM, np.deg2rad(30.0), "s")
    for zb in np.cumsum(thick):
        lo, hi = res_s.intensity(np.array([zb - 1e-9, zb + 1e-9]))
        assert lo == pytest.approx(hi, rel=1e-6)

    # (4b) p-Feld in der Halbebene 2 gegen den Fresnel-Transmissionskoeffizienten des E-Feldes
    n2 = np.sqrt(EPS_W633)
    for deg in (0.0, 30.0, 50.0):
        th = np.deg2rad(deg)
        c1 = np.cos(th)
        c2 = np.sqrt(1 - (N_PRISM * np.sin(th) / n2) ** 2)
        t_e = 2 * N_PRISM * c1 / (n2 * c1 + N_PRISM * c2)
        res = tmm([N_PRISM**2, EPS_W633], [], LAM, th, "p")
        assert res.intensity(np.array([0.0 + 1e-6]))[0] == pytest.approx(abs(t_e) ** 2, rel=1e-9)
        # Einfallsmedium: stehende Welle aus einfallender und reflektierter E-Welle (s-Fall)
        res_s = tmm([N_PRISM**2, EPS_W633], [], LAM, th, "s")
        z = -37.0
        kz0 = 2 * np.pi / LAM * N_PRISM * c1
        ref = abs(np.exp(1j * kz0 * z) + res_s.r * np.exp(-1j * kz0 * z)) ** 2
        assert res_s.intensity(np.array([z]))[0] == pytest.approx(ref, rel=1e-12)

    # (4c) Stabilität bei dicken Metallschichten: |E|² direkt über dem Metall fällt wie
    # exp(−2 Im k_z d); 500 nm → 1500 nm ändern es um exp(−2 Im k_z · 1000 nm) (rel. 1e-6).
    th = np.deg2rad(72.0)
    kx = k0 * N_PRISM * np.sin(th)
    vals = {}
    for d in (500.0, 1500.0):
        res = tmm([N_PRISM**2, EPS_AU, EPS_W633], [d], LAM, th, "p")
        vals[d] = res.intensity(np.array([d + 1.0]))[0]
        assert np.isfinite(vals[d]) and res.R == pytest.approx(
            abs(r_if(N_PRISM**2, EPS_AU, kx, "p")) ** 2, rel=1e-9
        )
    assert vals[1500.0] / vals[500.0] == pytest.approx(
        np.exp(-2 * kz(EPS_AU, kx).imag * 1000.0), rel=1e-6
    )

    # (5) SPR-Minimum
    degs = np.linspace(65.0, 80.0, 3001)
    r_curve = [tmm([N_PRISM**2, EPS_AU, EPS_W633], [50.0], LAM, np.deg2rad(a), "p").R for a in degs]
    i_min = int(np.argmin(r_curve))
    n_sp = np.sqrt(EPS_AU.real * EPS_W633 / (EPS_AU.real + EPS_W633))
    theta_sp = np.rad2deg(np.arcsin(n_sp / N_PRISM))
    print(
        f"SPR-Minimum {degs[i_min]:.2f}°, R = {r_curve[i_min]:.4f}; SP-Dispersion {theta_sp:.2f}°"
    )
    assert abs(degs[i_min] - theta_sp) < 1.0
    assert r_curve[i_min] < 0.05


def test_tmm_solver_grid():
    """``TmmSolver``: reines Wasser im Beobachtergitter entspricht dem Stapel ohne Gitterschichten;
    W hängt nur von z ab, Einheiten über I = ½ n c ε₀ |E₀|² (bei 0° ohne Schichten und mit
    n_incident = n_w ist W = |E₀|²)."""
    from biooptomd.coupling import EpsGrid, ForceGrid, TmmSolver, field_amplitude_sq, tmm

    grid = ForceGrid((5, 4, 9), (0.0, 0.0, 0.0), (1.0, 1.0, 0.5))
    water = EpsGrid(np.full((6, 2, 2), EPS_W633), (0.0, 0.0, 0.5), (2.0, 2.0, 0.4), 1.3317)
    solver = TmmSolver(LAM, 72.0, "p", N_PRISM, [(EPS_AU, 50.0)], intensity_w_m2=1e10)
    w = solver.solve(water, grid)
    assert w.shape == (9, 4, 5)
    assert np.all(w == w[:, :1, :1])
    res = tmm([N_PRISM**2, EPS_AU, EPS_W633], [50.0], LAM, np.deg2rad(72.0), "p")
    ref = field_amplitude_sq(1e10, N_PRISM) * res.intensity(grid.nodes(2) + 50.0)
    np.testing.assert_allclose(w[:, 0, 0], ref, rtol=1e-12)

    plain = TmmSolver(LAM, 0.0, "s", 1.3317, [], intensity_w_m2=1e10)
    w0 = plain.solve(water, grid)
    np.testing.assert_allclose(w0, field_amplitude_sq(1e10, 1.3317), rtol=1e-12)
    # unabhängiger Wert: I = 1 W/m² in Vakuum ⇒ |E₀|² = 2 Z₀ I mit Z₀ = 376.730313668 Ω
    assert field_amplitude_sq(1.0, 1.0) == pytest.approx(2 * 376.730313668 * 1e-18, rel=1e-9)


# --- 10.3 biooptomd.coupling: Fixpunkt-Treiber -------------------------------------------------
# Testschicht: unabhängige Atome (64 je Replica, 4 Replicas), in z harmonisch bei z = 1 nm gehalten
# (k = 100 kJ/mol/nm²), lateral frei (periodisch, Box 4 nm), über Gold 50 nm im Kretschmann-Aufbau
# (p, 72.6°, nahe SPR). α′ = 1e-3 nm³, v = 0.01 nm³. Das ε-Gitter ist lateral eine Zelle (der
# TMM-Solver nutzt nur das laterale Mittel), Δz = 0.1 nm. Intensitäten sind künstlich hoch, damit
# die Kraft messbar ist (Größenordnung ADR 0016).

C_ALPHA, C_VOL = np.array([1e-3]), np.array([0.01])
C_FG = ((8, 8, 13), (0.0, 0.0, 0.0), (0.5, 0.5, 0.25))


def _layer(n_rep=4, n=64, box=4.0, seed=1, k=100.0):
    ctx = _ctx(n_rep, n, box=(box, box))
    rng = np.random.default_rng(seed)
    pos = np.empty((n_rep, n, 3), np.float32)
    pos[..., :2] = rng.uniform(0, box, (n_rep, n, 2))
    pos[..., 2] = 1.0 + 0.1 * rng.standard_normal((n_rep, n))
    ctx.set_positions(pos)
    ctx.set_types(np.zeros((n_rep, n, 2), np.int32))
    ref = np.zeros((n_rep * n, 3), np.float32)
    ref[:, 2] = 1.0
    ctx.add_position_restraints(
        np.repeat(np.arange(n_rep), n).astype(np.int32), np.tile(np.arange(n), n_rep).astype(np.int32),
        ref, np.full(n_rep * n, k, np.float32), dims=(False, False, True),
    )  # fmt: skip
    ctx.set_temperatures(np.full(n_rep, 300.0, np.float32))
    ctx.set_integrator(0.002, 5.0, 7)
    return ctx


def _driver(
    ctx,
    intensity,
    beta=0.5,
    tol=0.1,
    checkpoint_dir=None,
    eps_dz=0.1,
    eps_nz=20,
    pol="p",
    tol_w=None,
):
    from biooptomd.coupling import ForceGrid, SelfConsistentOptics, TmmSolver
    from biooptomd.voxel import optical_scale

    solver = TmmSolver(LAM, 72.6, pol, N_PRISM, [(EPS_AU, 50.0)], intensity_w_m2=intensity)
    grid = ForceGrid(*C_FG, periodic_xy=True)
    scale = optical_scale(C_ALPHA, C_VOL, 1.3317, amplitude="peak")
    return SelfConsistentOptics(
        ctx, solver, grid, scale, eps_shape=(1, 1, eps_nz), eps_origin=(0.0, 0.0, 0.0),
        eps_spacing=(4.0, 4.0, eps_dz), alpha=C_ALPHA, volume=C_VOL, n_water=1.3317, n_equil=1000,
        n_sample=4000, stride=20, beta=beta, tol=tol, tol_w=tol_w, checkpoint_dir=checkpoint_dir,
    )  # fmt: skip


@pytest.mark.slow
def test_coupling_fixed_point():
    """Fixpunkt (ADR 0016): (1) Konvergenz in <= 10 Iterationen (β = 0.5; Δε < 0.1 über dem
    ε-Rauschen von einigen Prozent, Residuum ΔW < 2e-3 über dem Rauschboden ≈ 1e-3); die Schicht senkt
    W bei z = 1 nm messbar (> 1 %). (2) Das Ergebnis hängt nicht von β ab: Nach je 15 Iterationen
    (β = 0.3: Kontraktion 0.7¹⁵ ≈ 5e-3) stimmen die W-Profile für β = 0.3, 0.5 und 1.0 innerhalb von
    0.5 % überein. (3) Feldstärke 0 ist bitgleich zur ungekoppelten MD (kein Gitter, kein Observer)."""
    intensity = 5e16
    results = {}
    for beta in (0.5, 0.3, 1.0):
        drv = _driver(_layer(), intensity, beta=beta, tol_w=2e-3)
        w0 = drv.w.copy()
        if beta == 0.5:
            hist = drv.run(max_iter=10)
            assert drv.converged(), hist
        drv.tol = drv.tol_w = 0.0  # β-Vergleich: feste Zahl von Iterationen
        hist = drv.run(max_iter=15)
        print(beta, [(h["d_eps"], round(h["d_w"], 5), round(h["eps_noise"], 4)) for h in hist])
        results[beta] = drv.w[:, 0, 0]
        base = w0[:, 0, 0]
    # Die Schicht senkt W in ihrem Inneren lokal um ≈ 8 % (p, E_z ∝ 1/ε): Selbstfeld-Warnung (OQ-24).
    assert max(h["self_field"] for h in hist) > 0.05
    i_z = 4  # Knoten z = 1 nm
    assert results[0.5][i_z] < 0.99 * w0[i_z, 0, 0]
    np.testing.assert_allclose(results[0.3], results[1.0], rtol=5e-3)
    np.testing.assert_allclose(results[0.5], results[1.0], rtol=5e-3)
    # trennschärfer: der Effekt der Schicht W − W₀ selbst stimmt überein
    effect = results[1.0] - base
    for beta in (0.3, 0.5):
        rel = np.linalg.norm((results[beta] - base) - effect) / np.linalg.norm(effect)
        print(f"β = {beta}: ‖ΔW_β − ΔW_1‖/‖ΔW_1‖ = {rel:.4f}")
        assert rel < 0.1

    ctx_a, ctx_b = _layer(), _layer()
    drv = _driver(ctx_a, 0.0)
    drv.run(max_iter=2)
    assert np.all(drv.w == 0.0)
    ctx_b.step(2 * (1000 + 4000))
    np.testing.assert_array_equal(ctx_a.positions(), ctx_b.positions())
    np.testing.assert_array_equal(ctx_a.energies(), ctx_b.energies())


def test_coupling_scaling():
    """Im schwachen Feld skaliert die Verschiebung von ⟨z⟩ der Testschicht linear mit der Intensität
    (s-Polarisation, evaneszentes Feld ohne Plasmon: die Schicht ändert W kaum, die Kraft ist der
    äußere Gradient): Δz(2 I)/Δz(I) = 2 ± 0.05, und Δz stimmt mit F/k = −s W′(z₀)/k überein
    (10 %). ⟨z⟩ aus dem ε-Beobachter (S_α-gewichtete Zellmitten) der letzten Iteration; gleiche
    Zufallszahlen in allen Läufen, daher sind die Differenzen rauscharm."""
    from biooptomd.voxel import optical_scale

    shifts, w_prime = {}, {}
    for intensity in (0.0, 1e20, 2e20):
        drv = _driver(_layer(), intensity, beta=1.0, eps_dz=0.05, eps_nz=40, pol="s")
        drv.run(max_iter=3)
        res = drv.observer.result()
        prof = res["s_alpha_mean"].sum(axis=(2, 3)).mean(axis=0)
        shifts[intensity] = float((prof * res["z"]).sum() / prof.sum())
        w_prime[intensity] = (drv.w[5, 0, 0] - drv.w[3, 0, 0]) / 0.5  # dW/dz bei z = 1 nm
    d1 = shifts[1e20] - shifts[0.0]
    d2 = shifts[2e20] - shifts[0.0]
    s_scale = float(optical_scale(C_ALPHA, C_VOL, 1.3317, amplitude="peak")[0])
    expected = -s_scale * w_prime[1e20] / 100.0
    print(f"⟨z⟩-Verschiebung {d1:.3e} / {d2:.3e} nm, Verhältnis {d2 / d1:.4f}, F/k {expected:.3e}")
    assert d2 / d1 == pytest.approx(2.0, abs=0.05)
    assert d1 == pytest.approx(expected, rel=0.1)


def test_coupling_resume(tmp_path):
    """Abbruch nach Iteration k und Fortsetzen aus dem Checkpoint ergibt bitgleich denselben Verlauf,
    dasselbe W und dieselben Positionen wie ein durchgehender Lauf (ADR 0016 Punkt 5)."""
    full = _driver(_layer(), 5e16, checkpoint_dir=tmp_path / "full")
    for _ in range(4):
        full.step()
    part = _driver(_layer(), 5e16, checkpoint_dir=tmp_path / "full")
    part.resume(2)  # nach zwei abgeschlossenen Iterationen
    assert part.iteration == 2
    part.checkpoint_dir = tmp_path / "part"
    for _ in range(2):
        part.step()
    assert part.history == full.history
    np.testing.assert_array_equal(part.w, full.w)
    np.testing.assert_array_equal(part.ctx.positions(), full.ctx.positions())


def test_coupling_self_field_warning():
    """OQ-24 (d): Verändert die Schicht W stärker als ``self_field_warn``, warnt der Treiber einmal
    (RuntimeWarning); der Verlauf enthält ``self_field``."""
    drv = _driver(_layer(), 5e16)
    drv.self_field_warn = 1e-3  # Schicht senkt W um einige Prozent
    with pytest.warns(RuntimeWarning, match="OQ-24"):
        drv.step()
    assert drv.history[-1]["self_field"] > 1e-3


# --- 10.3 hp-FEM-Adapter -----------------------------------------------------------------------


def test_hpfem_adapter_planar():
    """hp-FEM-Adapter (ADR 0016, OQ-20): Kretschmann Prisma | Au 50 nm | Wasser mit einer planaren
    Schicht im Beobachtergitter (ε-Profil über 5 Zellen, Kontrast bis 0.3). W aus hp-FEM gegen den
    TMM-Referenzsolver an allen Knoten des Kraftgitters, rel. < 1 % (ADR); Einheiten und Amplitude
    über den Fall ohne Schicht (gleiche Normierung I = ½ n c ε₀ |E₀|²). p und s. Nur mit hp-FEM."""
    pytest.importorskip("hpfem")
    from biooptomd.coupling import EpsGrid, ForceGrid, HpFemGratingSolver, TmmSolver

    grid = ForceGrid((4, 2, 13), (0.0, 0.0, 0.0), (1.0, 2.0, 0.25), periodic_xy=True)
    profile = EPS_W633 + np.array([0.05, 0.3, 0.2, 0.1, 0.0])
    layer = EpsGrid(
        np.broadcast_to(profile[:, None, None], (5, 1, 2)).copy(), (0.0, 0.0, 0.5), (2.0, 4.0, 0.4),
        1.3317,
    )  # fmt: skip
    water = EpsGrid(np.full((5, 1, 2), EPS_W633), (0.0, 0.0, 0.5), (2.0, 4.0, 0.4), 1.3317)
    for pol in ("p", "s"):
        kw = {"substrate": [(EPS_AU, 50.0)], "intensity_w_m2": 1e10}
        tmm_solver = TmmSolver(LAM, 72.6, pol, N_PRISM, **kw)
        fem_solver = HpFemGratingSolver(LAM, 72.6, pol, N_PRISM, **kw)
        for eps in (water, layer):
            w_ref = tmm_solver.solve(eps, grid)
            w_fem = fem_solver.solve(eps, grid)
            rel = np.abs(w_fem - w_ref).max() / np.abs(w_ref).max()
            print(f"{pol}: max |W_FEM − W_TMM| / max W = {rel:.2e}")
            assert w_fem.shape == w_ref.shape
            np.testing.assert_allclose(w_fem, w_ref, rtol=1e-2)
        assert fem_solver.last_result.R == pytest.approx(tmm_solver_r(tmm_solver, layer), rel=1e-2)


def tmm_solver_r(solver, eps):
    from biooptomd.coupling import tmm

    e, d, _ = solver.stack(eps)
    return tmm(e, d, solver.wavelength_nm, np.deg2rad(solver.angle_deg), solver.pol).R


def test_npz_file_solver(tmp_path):
    """Datei-Adapter (ADR 0016 Punkt 4c): ein externer Prozess (eigener Python-Interpreter) liest
    ``eps_k.npz``, rechnet mit dem TMM und schreibt ``field_k.npz``; das Ergebnis ist bitgleich zum
    TmmSolver im selben Prozess. Fehlt die Ausgabe, gibt es einen Fehler."""
    import subprocess
    import sys

    from biooptomd.coupling import EpsGrid, ForceGrid, NpzFileSolver, TmmSolver

    grid = ForceGrid((4, 2, 13), (0.0, 0.0, 0.0), (1.0, 2.0, 0.25), periodic_xy=True)
    eps = EpsGrid(EPS_W633 + np.linspace(0.0, 0.2, 10)[:, None, None] * np.ones((10, 1, 2)),
                  (0.0, 0.0, 0.5), (2.0, 4.0, 0.2), 1.3317)  # fmt: skip
    script = tmp_path / "ext_solver.py"
    script.write_text(
        "import sys\nimport numpy as np\n"
        "from biooptomd.coupling import TmmSolver, read_eps_file\n"
        "eps, grid = read_eps_file(sys.argv[1])\n"
        f"s = TmmSolver({LAM}, 72.6, 'p', {N_PRISM}, [({EPS_AU!r}, 50.0)], intensity_w_m2=1e10)\n"
        "np.savez(sys.argv[2], w=s.solve(eps, grid))\n",
        encoding="utf-8",
    )
    solver = NpzFileSolver(tmp_path / "work", [sys.executable, str(script), "{input}", "{output}"])
    w_ext = solver.solve(eps, grid)
    w_ref = TmmSolver(LAM, 72.6, "p", N_PRISM, [(EPS_AU, 50.0)], intensity_w_m2=1e10).solve(
        eps, grid
    )
    np.testing.assert_array_equal(w_ext, w_ref)
    assert (tmp_path / "work" / "eps_000.npz").exists() and solver.calls == 1

    broken = NpzFileSolver(tmp_path / "w2", lambda inp, out: None)
    with pytest.raises(RuntimeError):
        broken.solve(eps, grid)
    with pytest.raises(subprocess.CalledProcessError):
        NpzFileSolver(tmp_path / "w3", [sys.executable, "-c", "raise SystemExit(3)"]).solve(
            eps, grid
        )
