"""Abnahmetests Phase 9.5: ImageChargeForce (ADR 0013, ROADMAP Phase 9.5)."""

import numpy as np
import pytest
from image_charge_ref import green, image_charges, potential

from biooptomd import builder

R_S, R_C = 1.8, 2.0
Z_IM = 0.3
KAPPA = np.array([1.0, 1.3])
EPS = np.array([78.5, 60.0])


def _system(rng, n=70, box=5.0, z_max=2.6):
    """Zufällige Ladungen über der Bildebene (einige unter z_im + ζ_c, einige ungeladen)."""
    pos = np.column_stack([rng.uniform(0, box, n), rng.uniform(0, box, n),
                           Z_IM + rng.uniform(0.15, z_max, n)])  # fmt: skip
    pos[:3, 2] = Z_IM + np.array([0.05, -0.1, 0.12])  # zwei geclampt, eins knapp darüber
    q = rng.uniform(-1.0, 1.0, n)
    q[5] = 0.0
    return pos, q


# --- Referenz ------------------------------------------------------------------------------------
def test_image_reference_dirichlet():
    """Referenz (reines NumPy, läuft auch ohne GPU): (a) Konstruktion: Potential von Ladungen +
    Bildern verschwindet auf der Ebene z_im (f = 1, ohne Switching); (b) die eigentliche Prüfung:
    U = ½ Σ_i q_i Φ_ind(r_i) (Energie der influenzierten Ladung) aus explizit gespiegelten Ladungen,
    ohne und mit periodischen Bildern; Eigenbild einer Ladung analytisch −½ q² G(2ζ); (c) Kräfte
    vs. finite Differenzen, rel. < 1e-6 (mit Switching, Bildsumme, Clamping)."""
    rng = np.random.default_rng(1)
    pos, q = _system(rng, n=12, box=5.0, z_max=1.5)
    pos[:3, 2] = Z_IM + np.array([0.2, 0.3, 0.4])  # (a, b) ohne Clamping
    pts = np.column_stack([rng.uniform(-3, 8, 50), rng.uniform(-3, 8, 50), np.full(50, Z_IM)])
    phi = potential(pts, pos, q, Z_IM, 1.0, 1.1, 78.5)
    scale = np.abs(potential(pts + [0, 0, 0.3], pos, q, Z_IM, 0.0, 1.1, 78.5)).max()
    assert np.abs(phi).max() < 1e-12 * scale
    # (b) mit großer Box: nur n = 0 trägt bei, kein Cutoff
    e, _ = image_charges(pos, q, (1e4, 1e4), Z_IM, 0.7, 1.1, 78.5, R_S, R_C, switching=False,
                         n_images=(0, 0))  # fmt: skip
    mirror = pos * [1, 1, -1] + [0, 0, 2 * Z_IM]
    induced = -0.7 * (q[None] * green(np.linalg.norm(pos[:, None] - mirror[None], axis=-1),
                                      1.1, 78.5)[0]).sum(axis=1)  # fmt: skip
    np.testing.assert_allclose(e, 0.5 * np.sum(q * induced), rtol=1e-12)
    # (b, periodisch): alle Atome in x, y ∈ [0, L/2) ⇒ Minimum Image verschiebt nichts, die
    # Gittersumme der Referenz entspricht der expliziten Summe über gespiegelte Gitterbilder.
    box, n_img = 3.0, 4
    pos_p = np.column_stack([rng.uniform(0, box / 2, 8), rng.uniform(0, box / 2, 8),
                             Z_IM + rng.uniform(0.2, 1.0, 8)])  # fmt: skip
    q_p = rng.uniform(-1, 1, 8)
    e_p, _ = image_charges(pos_p, q_p, (box, box), Z_IM, 0.7, 1.1, 78.5, R_S, R_C,
                           switching=False, n_images=(n_img, n_img))  # fmt: skip
    shifts = np.array([(a * box, b * box, 0.0) for a in range(-n_img, n_img + 1)
                       for b in range(-n_img, n_img + 1)])  # fmt: skip
    mirror_p = pos_p * [1, 1, -1] + [0, 0, 2 * Z_IM]
    d = np.linalg.norm(
        pos_p[:, None, None] - (mirror_p[None, :, None] + shifts[None, None]), axis=-1
    )
    induced_p = -0.7 * (q_p[None, :, None] * green(d, 1.1, 78.5)[0]).sum(axis=(1, 2))
    np.testing.assert_allclose(e_p, 0.5 * np.sum(q_p * induced_p), rtol=1e-12)
    zeta = 0.37
    e1, f1 = image_charges(np.array([[1.0, 2.0, Z_IM + zeta]]), np.array([0.8]), (1e4, 1e4), Z_IM,
                           1.0, 1.1, 78.5, R_S, R_C, switching=False, n_images=(0, 0))  # fmt: skip
    g, dg = green(2 * zeta, 1.1, 78.5)
    np.testing.assert_allclose(e1, -0.5 * 0.64 * g, rtol=1e-14)
    np.testing.assert_allclose(f1[0], [0.0, 0.0, 0.64 * dg], rtol=1e-14)  # F = −dU/dz = ½q²·2G'
    # (c) finite Differenzen, kleine Box (Bildsumme), Clamping (Atome 0, 1 ohne z-Kraft)
    pos, q = _system(rng, n=12, box=3.0, z_max=1.5)
    args = ((3.0, 3.0), Z_IM, -0.5, 1.1, 70.0, R_S, R_C)
    _, f = image_charges(pos, q, *args)
    h = 1e-6
    fd = np.zeros_like(f)
    for i in range(len(q)):
        for d in range(3):
            pp, pm = pos.copy(), pos.copy()
            pp[i, d] += h
            pm[i, d] -= h
            fd[i, d] = -(image_charges(pp, q, *args)[0] - image_charges(pm, q, *args)[0]) / (2 * h)
    assert f[0, 2] == 0.0 and f[1, 2] == 0.0
    np.testing.assert_allclose(f, fd, rtol=1e-6, atol=1e-6 * np.abs(f).max())


# --- GPU ------------------------------------------------------------------------------------------
@pytest.fixture
def core():
    return pytest.importorskip("biooptomd._core")


def _ctx(core, pos, q, box, factor=1.0, image_sum=False, kappa=KAPPA, eps=EPS):
    n_rep = len(kappa)
    ctx = core.SimulationContext(n_rep, len(q))
    ctx.set_box(np.full((n_rep, 2), box, np.float32))
    p = np.broadcast_to(pos, (n_rep,) + pos.shape).astype(np.float32)
    ctx.set_positions(np.ascontiguousarray(p))
    ctx.set_charges(np.ascontiguousarray(np.broadcast_to(q, (n_rep, len(q))).astype(np.float32)))
    ctx.set_masses(np.full((n_rep, len(q)), 12.0, np.float32))
    ctx.set_temperatures(np.full(n_rep, 300.0, np.float32))
    plugin = ctx.add_image_charges(Z_IM, factor, kappa.astype(np.float32), eps.astype(np.float32),
                                   R_S, R_C, image_sum=image_sum)  # fmt: skip
    return ctx, plugin


def _term(core):
    """Spalte des Terms 'image' in ctx.energies() (Spalte 0 = E_kin)."""
    return 1 + list(core.ENERGY_TERMS).index("image")


def builder_core():
    return pytest.importorskip("biooptomd._core")


@pytest.mark.gpu
@pytest.mark.parametrize("factor", [1.0, -0.5])
@pytest.mark.parametrize(("box", "image_sum"), [(5.0, False), (5.0, True), (3.0, False)])
def test_image_gpu_vs_numpy(core, factor, box, image_sum):
    """GPU vs. NumPy: Energie rel. < 1e-5, Kräfte < 1e-4 RMS (relativ zur RMS-Kraft), zwei
    Replicas mit verschiedenem κ/ε_r; Minimum Image (L = 5 ≥ 2 r_c), erzwungene Bildsumme und
    Bildsumme wegen L = 3 < 2 r_c; f = 1 und f = −0.5. Padding-Atome (70 → 96) ohne Kraft."""
    rng = np.random.default_rng(2)
    pos, q = _system(rng, box=box)
    ctx, plugin = _ctx(core, pos, q, box, factor, image_sum)
    assert plugin.uses_image_sum == (image_sum or box < 2 * R_C)
    ctx.compute_forces()
    x = ctx.positions_precise()[:, : len(q), :3]
    e_gpu = ctx.energies()[:, _term(core)]
    f_gpu = ctx.forces()[:, : len(q), :3]
    for r in range(2):
        e_ref, f_ref = image_charges(x[r], q, (box, box), Z_IM, factor, KAPPA[r], EPS[r], R_S, R_C)
        assert e_gpu[r] == pytest.approx(e_ref, rel=1e-5)
        rms = np.sqrt(np.mean((f_gpu[r] - f_ref) ** 2)) / np.sqrt(np.mean(f_ref**2))
        assert rms < 1e-4, rms
    assert np.all(ctx.forces()[:, len(q) :] == 0.0)  # Padding (AGENTS Regel 6)
    assert ctx.status() == 0


@pytest.mark.gpu
def test_image_lateral_culling(core):
    """Tile-Vorauswahl im Minimum-Image-Pfad (L = 8 ≥ 2 r_c): vier Tiles als x-Streifen nahe der
    Wand – A x ∈ [0.2, 0.8], B [2.5, 3.0] (Lücke zu A 1.7 nm < r_c), C [5.0, 5.4] (Lücke zu B 2.0,
    zu A über die Grenze 2.8 ⇒ verworfen), D [7.3, 7.9] (24 Atome + Padding; zu A über die
    periodische Grenze nur 0.3 nm – darf nicht verworfen werden). GPU = NumPy wie
    test_image_gpu_vs_numpy, und gleich dem Bildsummenpfad ohne laterale Vorauswahl."""
    rng = np.random.default_rng(8)
    strips = [(0.2, 0.8, 32), (2.5, 3.0, 32), (5.0, 5.4, 32), (7.3, 7.9, 24)]
    pos = np.concatenate([np.column_stack([rng.uniform(lo, hi, m), rng.uniform(0, 8, m),
                                           Z_IM + rng.uniform(0.15, 0.6, m)])
                          for lo, hi, m in strips])  # fmt: skip
    pos[32:64, 0] = np.where(rng.random(32) < 0.3, 3.0 - 1e-3, pos[32:64, 0])  # B-Kante
    pos[64:96, 0] = np.where(rng.random(32) < 0.3, 5.0 + 1e-3, pos[64:96, 0])  # C-Kante
    q = rng.uniform(-1.0, 1.0, len(pos))
    ctx, plugin = _ctx(core, pos, q, 8.0)
    assert not plugin.uses_image_sum
    ctx.compute_forces()
    ref_ctx, _ = _ctx(core, pos, q, 8.0, image_sum=True)
    ref_ctx.compute_forces()
    x = ctx.positions_precise()[:, : len(q), :3]
    f_gpu = ctx.forces()[:, : len(q), :3]
    for r in range(2):
        e_ref, f_ref = image_charges(x[r], q, (8.0, 8.0), Z_IM, 1.0, KAPPA[r], EPS[r], R_S, R_C)
        assert ctx.energies()[r, _term(core)] == pytest.approx(e_ref, rel=1e-5)
        rms = np.sqrt(np.mean((f_gpu[r] - f_ref) ** 2)) / np.sqrt(np.mean(f_ref**2))
        assert rms < 1e-4, rms
        # A–D über die periodische Grenze trägt bei (sonst wäre D fast kraftfrei in x)
        assert np.abs(f_ref[96:, 0]).max() > 1e-3
    np.testing.assert_allclose(ctx.energies(), ref_ctx.energies(), rtol=1e-6)
    np.testing.assert_allclose(ctx.forces(), ref_ctx.forces(), rtol=1e-5, atol=1e-5)
    assert np.all(ctx.forces()[:, len(q) :] == 0.0)


@pytest.mark.gpu
@pytest.mark.parametrize("box", [5.0, 3.0])
def test_image_far_atoms_zero(core, box):
    """Atome mit ζ > r_c erhalten exakt null Bildkraft (s ≥ ζ_i + ζ_j > r_c, Vorauswahl), auch
    neben nahen Atomen; nahe Atome spüren ihre Bilder."""
    rng = np.random.default_rng(3)
    pos, q = _system(rng, n=40, box=box, z_max=1.0)
    far = np.zeros(len(q), bool)
    far[20:] = True
    pos[far, 2] = Z_IM + R_C + rng.uniform(0.01, 2.0, far.sum())
    ctx, _ = _ctx(core, pos, q, box)
    ctx.compute_forces()
    f = ctx.forces()[:, : len(q), :3]
    assert np.all(f[:, far] == 0.0)
    assert np.all(np.abs(f[:, ~far & (q != 0)]).sum(axis=-1) > 0)
    pos[:, 2] = Z_IM + R_C + rng.uniform(0.01, 2.0, len(q))
    ctx, _ = _ctx(core, pos, q, box)
    ctx.compute_forces()
    assert np.all(ctx.forces()[:, : len(q), :3] == 0.0)
    assert np.all(ctx.energies()[:, _term(core)] == 0.0)


@pytest.mark.gpu
def test_image_clamp(core):
    """ζ < ζ_c: Energie konstant in z und gleich der FP64-Referenz bei ζ = ζ_c (rel. 1e-5, Stetigkeit),
    z-Kraft null, Zähler zählt geladene Atome je Replica und Auswertung (ungeladene nicht);
    reset_counters setzt zurück."""
    pos = np.array([[1.0, 1.0, Z_IM + 0.05], [2.5, 1.0, Z_IM + 0.6], [1.0, 3.0, Z_IM - 0.2]])
    q = np.array([1.0, -0.7, 0.0])
    energies = []
    for z in (Z_IM + 0.05, Z_IM - 0.3):
        pos[0, 2] = z
        ctx, plugin = _ctx(core, pos, q, 5.0)
        ctx.compute_forces()
        energies.append(ctx.energies()[:, _term(core)].copy())
        assert np.all(ctx.forces()[:, 0, 2] == 0.0)
        assert plugin.clamp_count() == 2  # 2 Replicas × 1 geladenes Atom
        ctx.compute_forces()
        assert plugin.clamp_count() == 4
        plugin.reset_counters()
        assert plugin.clamp_count() == 0
    np.testing.assert_allclose(energies[0], energies[1], rtol=0, atol=0)
    at_clamp = pos.copy()
    at_clamp[0, 2] = Z_IM + 0.1
    x = ctx.positions_precise()[:, :3, :3].astype(np.float64)
    for r in range(2):
        at_clamp[1:, :] = x[r, 1:]
        e_ref, _ = image_charges(at_clamp, q, (5.0, 5.0), Z_IM, 1.0, KAPPA[r], EPS[r], R_S, R_C)
        assert energies[0][r] == pytest.approx(e_ref, rel=1e-5)


@pytest.mark.gpu
def test_image_set_screening(core):
    """Nach ctx.set_screening (Replica Exchange) gleich einem frischen Kontext mit den neuen Werten."""
    rng = np.random.default_rng(4)
    pos, q = _system(rng)
    ctx, _ = _ctx(core, pos, q, 5.0)
    new_k, new_e = np.array([1.4, 0.9]), np.array([65.0, 75.0])
    ctx.set_screening(new_k.astype(np.float32), new_e.astype(np.float32))
    ctx.compute_forces()
    fresh, _ = _ctx(core, pos, q, 5.0, kappa=new_k, eps=new_e)
    fresh.compute_forces()
    np.testing.assert_array_equal(ctx.energies(), fresh.energies())
    np.testing.assert_array_equal(ctx.forces(), fresh.forces())


@pytest.mark.gpu
def test_image_determinism(core):
    """Zwei identische Läufe mit Bildladungen (Langevin) sind bitidentisch."""
    rng = np.random.default_rng(5)
    pos, q = _system(rng)
    pos[:3, 2] = Z_IM + 0.5
    out = []
    for _ in range(2):
        ctx, _ = _ctx(core, pos, q, 5.0)
        ctx.add_planar_surface(np.array([0.0, 2.0], np.float32), np.array([0.0, 0.3], np.float32),
                               z0=Z_IM - 0.2)  # fmt: skip
        ctx.set_types(np.ones((2, len(q), 2), np.int32))
        ctx.set_integrator(0.001, 1.0, seed=3)
        ctx.step(500)
        out.append((ctx.positions_precise(), ctx.energies()))
    np.testing.assert_array_equal(out[0][0], out[1][0])
    np.testing.assert_array_equal(out[0][1], out[1][1])


def _chain_on_gold(rng, n_rep=2):
    from test_phase7 import _small_system

    spec = _small_system(rng)
    ctx, info = builder.build_context(spec, n_rep, (4.0, 4.0), 300.0, 0.15, hmr=False,
                                      types=np.ones(spec.n_atoms, np.int32))  # fmt: skip
    return spec, ctx, info


@pytest.mark.gpu
def test_image_builder():
    """builder.add_image_charges: Gold ⇒ f = 1, z_im = z0 + z_top + d_image, κ/ε_r/r_c aus BuildInfo
    (Energie gleich direktem Aufruf); Graphen ohne Faktor ⇒ Fehler (OQ-18)."""
    from biooptomd import surface

    rng = np.random.default_rng(6)
    _, ctx, info = _chain_on_gold(rng)
    builder.add_image_charges(ctx, info, "gold", z0_nm=-0.1, d_image_nm=0.05)
    ctx.compute_forces()
    _, ref, info2 = _chain_on_gold(np.random.default_rng(6))
    z_im = -0.1 + surface.MATERIALS["gold"].z_top_nm + 0.05
    ref.add_image_charges(z_im, 1.0, info2.kappa.astype(np.float32),
                          info2.eps_r.astype(np.float32), info2.r_switch, info2.r_cutoff)  # fmt: skip
    ref.compute_forces()
    np.testing.assert_array_equal(ctx.energies(), ref.energies())
    assert np.all(ctx.energies()[:, _term(builder_core())] != 0.0)
    with pytest.raises(ValueError, match="OQ-18"):
        builder.add_image_charges(ctx, info, "graphene")


@pytest.mark.gpu
@pytest.mark.slow
def test_image_nve():
    """NVE 10 ps, Δt = 0.5 fs: Ketten (Debye-Hückel, 0.15 M, L = 4 nm < 2 r_c ⇒ Bildsumme) über einer
    9-3-Wand (Gold) mit Bildladungen, Drift < 1e-4 kJ/mol/ps/Atom, Fluktuation < 1e-3·⟨E_kin⟩
    (Kriterien wie test_gb_nve)."""
    from biooptomd import surface

    rng = np.random.default_rng(7)
    spec, ctx, info = _chain_on_gold(rng)
    n = spec.n_atoms
    t = surface.wall_table("gold", info.lj_table, np.array([0.0, 0.2]))
    ctx.add_planar_surface(t["prefactor"], t["sigma"], form=t["form"], z0=0.0)
    plugin = builder.add_image_charges(ctx, info, "gold")
    m = spec.masses
    kb = 0.0083144626
    v = rng.normal(0.0, 1.0, (2, n, 3)) * np.sqrt(kb * 300.0 / m)[None, :, None]
    v -= (v * m[None, :, None]).sum(axis=1, keepdims=True) / m.sum()
    ctx.set_velocities(v)
    ctx.set_integrator(0.0005, 0.0, seed=0)
    tracker = ctx.add_energy_tracker(stride=20, capacity=1000)
    ctx.step(20_000)
    res = tracker.result()
    en = res["energies"]
    total = en[:, :, 0] + en[:, :, 1:].sum(axis=2)
    t_ps = res["steps"] * 0.0005
    drift = np.array([np.polyfit(t_ps, total[r], 1)[0] for r in range(2)]) / n
    fluct = total.std(axis=1) / en[:, :, 0].mean(axis=1)
    assert np.all(np.abs(drift) < 1e-4), f"Drift {drift}"
    assert np.all(fluct < 1e-3), f"Fluktuation {fluct}"
    assert np.all(en[:, :, _term(builder_core())] != 0.0), "Bildterm soll aktiv sein"
    # H-Atome erreichen die Goldwand bis ζ < ζ_c (σ_iw(H) ≈ 0.19 nm): Clamping tritt auf, die
    # Energie bleibt stetig (Drift oben eingeschlossen).
    assert plugin.clamp_count() > 0
    print(f"Drift {drift} kJ/mol/ps/Atom, Fluktuation {fluct}, Clamping {plugin.clamp_count()}")
    assert ctx.status() == 0
