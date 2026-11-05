"""Abnahmetests Phase 2: Langevin-BAOAB, PlanarSurfaceForce, HarmonicRestraintPullingForce.

Siehe docs/ROADMAP.md, Phase 2, und docs/TESTING.md (statistische Toleranzen: 4σ aus
Blockmittelung, Verteilungstests mit p > 0.001, feste Seeds).
"""

import numpy as np
import pytest
from philox_ref import KAT, philox4x32_10
from restraint_ref import com_pulling, com_pulling_work_increment, position_restraint
from scipy import stats
from surface_ref import (
    WALL_FORMS,
    clamp_distance,
    gouy_chapman,
    surface_energy_force,
    wall,
    wall_energy_force,
    wall_minimum,
)

pytestmark = pytest.mark.gpu

_core = pytest.importorskip("biooptomd._core")
KB = _core.units.KB
TERM = {name: 1 + j for j, name in enumerate(_core.ENERGY_TERMS)}  # Spalte in energies()


def _ctx(n_replicas, n_atoms, box=5.0, mass=20.0, temperatures=300.0):
    ctx = _core.SimulationContext(n_replicas, n_atoms)
    ctx.set_box(np.full((n_replicas, 2), box, np.float32))
    ctx.set_masses(np.broadcast_to(np.float32(mass), (n_replicas, n_atoms)).astype(np.float32))
    ctx.set_temperatures(np.broadcast_to(temperatures, (n_replicas,)).astype(np.float32))
    return ctx


def _block_mean_sigma(x, n_blocks=20):
    """Mittelwert und σ des Mittelwerts aus Blockmittelung über die letzte Achse."""
    n = x.shape[-1] // n_blocks * n_blocks
    blocks = x[..., :n].reshape(x.shape[:-1] + (n_blocks, -1)).mean(axis=-1)
    return blocks.mean(axis=-1), blocks.std(axis=-1, ddof=1) / np.sqrt(n_blocks)


def _chi2_pvalue(samples, edges, expected_prob):
    """χ²-Test eines Histogramms gegen erwartete Bin-Wahrscheinlichkeiten (Bins mit < 5 erwarteten
    Einträgen werden mit Nachbarn zusammengelegt)."""
    observed, _ = np.histogram(samples, edges)
    expected = expected_prob / expected_prob.sum() * observed.sum()
    obs_m, exp_m, acc_o, acc_e = [], [], 0.0, 0.0
    for o, e in zip(observed, expected, strict=True):
        acc_o += o
        acc_e += e
        if acc_e >= 5.0:
            obs_m.append(acc_o)
            exp_m.append(acc_e)
            acc_o = acc_e = 0.0
    obs_m[-1] += acc_o
    exp_m[-1] += acc_e
    return stats.chisquare(obs_m, exp_m).pvalue


# --- Abnahmetests (Namen verbindlich laut ROADMAP.md) ----------------------------------------


def test_equipartition_per_replica():
    """4 Replicas bei 250/300/350/400 K, 2000 freie Atome: ⟨E_kin⟩ = ½ N_dof k_B T_r (4σ)."""
    temps = np.array([250.0, 300.0, 350.0, 400.0])
    gamma, dt = 2.0, 0.002
    ctx = _ctx(4, 2000, temperatures=temps)
    rng = np.random.default_rng(10)
    ctx.set_positions(rng.uniform(0.0, 5.0, (4, 2000, 3)))
    ctx.set_integrator(dt, gamma, seed=1234)
    ctx.step(int(5.0 / gamma / dt))  # Equilibrierung 5/γ
    tracker = ctx.add_energy_tracker(stride=10, capacity=4000)
    ctx.step(30000)  # 60 ps; Korrelationszeit von E_kin: 1/(2γ) = 0.25 ps

    e_kin = tracker.result()["energies"][:, :, 0]
    mean, sigma = _block_mean_sigma(e_kin)  # 20 Blöcke à 3 ps = 12 Korrelationszeiten
    expected = 0.5 * ctx.degrees_of_freedom() * KB * temps
    assert np.all(np.abs(mean - expected) < 4.0 * sigma), (mean, expected, sigma)
    assert np.all(sigma < 0.01 * expected)  # Test ist aussagekräftig


def test_velocity_distribution():
    """Maxwell-Boltzmann pro Komponente (verschiedene Massen und T): KS-Test p > 0.001."""
    temps = np.array([280.0, 420.0])
    gamma, dt = 2.0, 0.002
    ctx = _core.SimulationContext(2, 2000)
    ctx.set_box(np.full((2, 2), 5.0, np.float32))
    masses = np.tile(np.array([1.0, 16.0, 100.0], np.float32), (2, 667))[:, :2000]
    ctx.set_masses(masses)
    ctx.set_temperatures(temps.astype(np.float32))
    ctx.set_integrator(dt, gamma, seed=99)
    ctx.step(int(5.0 / gamma / dt))

    scaled = [[] for _ in temps]
    for _ in range(6):
        ctx.step(int(2.5 / dt))  # 2.5 ps ≫ 1/γ zwischen den Schnappschüssen
        v = ctx.velocities()[:, :2000, :3].astype(np.float64)
        for r, t in enumerate(temps):
            scaled[r].append((v[r] * np.sqrt(masses[r, :, None] / (KB * t))).ravel())
    for r in range(len(temps)):
        p = stats.kstest(np.concatenate(scaled[r]), "norm").pvalue
        assert p > 1e-3, f"Replica {r}: KS p = {p:.2e}"


def test_restraint_boltzmann():
    """Unabhängige Atome im harmonischen Restraint (auch über die Boxgrenze): P(Δ) ∝ exp(−V/k_BT)."""
    temps = np.array([300.0, 400.0])
    k, gamma, dt, n = 200.0, 2.0, 0.002, 1000
    ctx = _ctx(2, n, temperatures=temps)
    rng = np.random.default_rng(11)
    ref = rng.uniform(0.0, 5.0, (2, n, 3)).astype(np.float32)
    ref[:, : n // 4, 0] = 0.01  # Referenz direkt an der Boxgrenze ⇒ Minimum Image nötig
    ref[..., 2] += 2.0
    ctx.set_positions(ref)
    replica, atom = np.divmod(np.arange(2 * n, dtype=np.int32), n)
    ctx.add_position_restraints(
        replica.astype(np.int32),
        atom.astype(np.int32),
        ref.reshape(-1, 3),
        np.full(2 * n, k, np.float32),
    )
    ctx.set_integrator(dt, gamma, seed=5)
    ctx.step(int(5.0 / gamma / dt))

    deltas = [[] for _ in temps]
    for _ in range(30):
        ctx.step(1500)  # 3 ps; Abklingrate γ/2 = 1/ps
        pos = ctx.positions()[:, :n, :3].astype(np.float64)
        d = pos - ref
        d[..., :2] -= 5.0 * np.rint(d[..., :2] / 5.0)
        for r in range(2):
            deltas[r].append(d[r].ravel())
    for r, t in enumerate(temps):
        sigma = np.sqrt(KB * t / k)
        edges = np.linspace(-4.5 * sigma, 4.5 * sigma, 46)
        prob = np.diff(stats.norm.cdf(edges, scale=sigma))
        p = _chi2_pvalue(np.concatenate(deltas[r]), edges, prob)
        assert p > 1e-3, f"Replica {r}: χ² p = {p:.2e}"


@pytest.mark.parametrize("form", WALL_FORMS)
def test_wall_boltzmann(form):
    """Atome zwischen Wand (alle Formen, ADR 0007) und harmonischem z-Restraint: P(z) vs.
    Boltzmann (χ²)."""
    temps = np.array([300.0, 350.0])
    eps_w, sigma_w, k_z, z_ref = 2.0, 0.3, 100.0, 0.7
    gamma, dt, n = 2.0, 0.001, 2000
    ctx = _ctx(2, n, mass=40.0, temperatures=temps)
    rng = np.random.default_rng(12)
    start = rng.uniform(0.0, 5.0, (2, n, 3)).astype(np.float32)
    start[..., 2] = z_ref
    ctx.set_positions(start)
    ctx.set_types(np.broadcast_to(np.int32([0, 1]), (2, n, 2)).copy())
    ctx.add_planar_surface(
        np.array([0.0, eps_w], np.float32), np.array([0.0, sigma_w], np.float32), form=form
    )
    replica, atom = np.divmod(np.arange(2 * n, dtype=np.int32), n)
    ctx.add_position_restraints(
        replica.astype(np.int32),
        atom.astype(np.int32),
        start.reshape(-1, 3),
        np.full(2 * n, k_z, np.float32),
        dims=(False, False, True),
    )
    ctx.set_integrator(dt, gamma, seed=6)
    ctx.step(int(5.0 / gamma / dt))

    zs = [[] for _ in temps]
    for _ in range(40):
        ctx.step(2000)  # 2 ps
        z = ctx.positions()[:, :n, 2].astype(np.float64)
        for r in range(2):
            zs[r].append(z[r])
    grid = np.linspace(0.05, 2.0, 200001)
    v_tot = wall_energy_force(grid, form, eps_w, sigma_w, 0.335)[0]
    v_tot += 0.5 * k_z * (grid - z_ref) ** 2
    edges = np.linspace(0.15, 1.5, 55)
    for r, t in enumerate(temps):
        w = np.exp(-(v_tot - v_tot.min()) / (KB * t))
        cdf = np.concatenate([[0.0], np.cumsum(0.5 * (w[1:] + w[:-1]) * np.diff(grid))])
        prob = np.diff(np.interp(edges, grid, cdf))
        p = _chi2_pvalue(np.concatenate(zs[r]), edges, prob)
        assert p > 1e-3, f"Replica {r}: χ² p = {p:.2e}"


def _gouy_chapman_ctx(z, q, temps, phi0, kappa, linear=False):
    """Kontext mit Atomen bei Höhe z (n,) und Ladung q (n,) in jeder Replica, nur Gouy-Chapman."""
    n_rep, n = len(temps), len(z)
    ctx = _ctx(n_rep, n, temperatures=np.asarray(temps))
    xyz = np.zeros((n_rep, n, 3), np.float32)
    xyz[..., 0] = np.linspace(0.0, 5.0, n, endpoint=False)
    xyz[..., 2] = z
    ctx.set_positions(xyz)
    ctx.set_charges(np.broadcast_to(q, (n_rep, n)).astype(np.float32))
    ctx.add_planar_surface(
        np.zeros(1, np.float32),
        np.zeros(1, np.float32),
        phi0=phi0,
        kappa=np.asarray(kappa, np.float32),
        gouy_chapman="linear" if linear else "nonlinear",
    )
    ctx.compute_forces()
    return ctx


@pytest.mark.parametrize("phi0_mv", [25.0, 100.0, 200.0, -100.0])
def test_gouy_chapman_force(phi0_mv):
    """Nichtlineares Gouy-Chapman (OQ-06): Referenzkraft = −dE/dz (finite Differenzen, FP64,
    rel. < 1e-4); GPU-Kraft und -Energie gegen die Referenz, γ und κ pro Replica (T_r)."""
    rng = np.random.default_rng(13)
    n = 300
    z = rng.uniform(-0.2, 3.0, n)
    z[:3] = [0.0, 1e-3, -0.1]
    q = rng.uniform(-1.0, 1.0, n)
    temps = np.array([280.0, 300.0, 340.0])
    kappa = np.array([1.10, 1.04, 0.95])  # κ_r je Replica (OQ-10)
    phi0, h = phi0_mv * 1e-3 * _core.units.FARADAY, 1e-4  # Φ0 < 0: Vorzeichenpfad

    for t, k in zip(temps, kappa, strict=True):
        kt = KB * t
        _, f_ref = gouy_chapman(z, q, phi0, k, kt)
        e_plus, _ = gouy_chapman(z + h, q, phi0, k, kt)
        e_minus, _ = gouy_chapman(z - h, q, phi0, k, kt)
        inside = np.abs(z) > 2 * h
        f_fd = -(e_plus - e_minus) / (2 * h)
        nz = inside & (z > 0)
        assert (np.abs(f_fd - f_ref)[nz] / np.abs(f_ref[nz])).max() < 1e-4
        assert np.abs(f_fd[inside & (z < 0)]).max() < 1e-9  # OQ-12: V konstant hinter der Wand

    ctx = _gouy_chapman_ctx(z, q, temps, phi0, kappa)
    zf = z.astype(np.float32).astype(np.float64)  # tatsächlich verwendete float32-Werte
    qf = q.astype(np.float32).astype(np.float64)
    f_gpu = ctx.forces()[:, :n].astype(np.float64)
    e_gpu = ctx.energies()[:, TERM["surface"]]
    assert not f_gpu[..., :2].any()
    for r, (t, k) in enumerate(zip(temps, kappa, strict=True)):
        kf = float(np.float32(k))
        e_ref, f_ref = gouy_chapman(zf, qf, phi0, kf, KB * float(np.float32(t)))
        np.testing.assert_allclose(f_gpu[r, :, 2], f_ref, rtol=1e-5, atol=1e-5)
        assert abs(e_gpu[r] - e_ref.sum()) < 1e-5 * np.abs(e_ref).sum()


def test_gouy_chapman_linear_limit():
    """OQ-06: nichtlineare Form → lineare Form für Φ0 → 0 (rel. < 1e-3 bei 1 mV); γ pro Replica
    aus T_r; die Linearisierung überschätzt das Fernfeld bei 100 mV; große Φ0 bleiben endlich."""
    z = np.linspace(0.0, 3.0, 64)
    q = np.ones_like(z)
    temps, kappa = np.array([300.0, 350.0]), np.array([1.0, 0.9])

    def run(phi0_mv, linear):
        ctx = _gouy_chapman_ctx(z, q, temps, phi0_mv * 1e-3 * _core.units.FARADAY, kappa, linear)
        return ctx.forces()[:, :64, 2].astype(np.float64), ctx.energies()[:, TERM["surface"]]

    f_nl, e_nl = run(1.0, False)
    f_li, e_li = run(1.0, True)
    assert np.abs(f_nl / f_li - 1.0).max() < 1e-3
    assert np.abs(e_nl / e_li - 1.0).max() < 1e-3

    f_nl, _ = run(100.0, False)
    f_li, _ = run(100.0, True)
    far = z > 2.0
    for r, t in enumerate(temps):
        # F_lin/F_nl = Φ0 (1 − u²)/(4 k_BT γ); Fernfeld (u → 0): Φ0/(4 k_BT γ) ≈ 1.3 (OQ-06)
        phi0 = 0.1 * _core.units.FARADAY
        gamma = np.tanh(phi0 / (4 * KB * t))
        u = gamma * np.exp(-kappa[r] * z[far])
        np.testing.assert_allclose(
            f_li[r, far] / f_nl[r, far], phi0 * (1 - u * u) / (4 * KB * t * gamma), rtol=2e-4
        )
        assert 1.2 < phi0 / (4 * KB * t * gamma) < 1.4

    # Φ0 = 1 V: γ = tanh(Φ0/4k_BT) rundet in float32 auf 1. V(0) = qΦ0 exakt; F(0) =
    # 2k_BTκ sinh(Φ0/2k_BT) ≈ 10⁸ bleibt im Fixed-Point-Bereich (keine Kraftschranke für GC).
    phi_big = 1.0 * _core.units.FARADAY
    ctx = _gouy_chapman_ctx(z[:1], q[:1], temps, phi_big, kappa)
    e0 = ctx.energies()[:, TERM["surface"]]
    np.testing.assert_allclose(e0, phi_big, rtol=1e-5)
    f0 = ctx.forces()[:, 0, 2].astype(np.float64)
    kt = KB * temps
    np.testing.assert_allclose(f0, 2 * kt * kappa * np.sinh(phi_big / (2 * kt)), rtol=1e-4)
    assert ctx.status() == 0


@pytest.mark.parametrize("form", WALL_FORMS)
def test_wall_forms_vs_numpy(form):
    """ADR 0007: Energie/Kraft jeder Wandform vs. NumPy und finite Differenzen (rel. < 1e-4);
    Lage und Tiefe des Minimums analytisch (9-3, 10-4) bzw. numerisch (10-4-3)."""
    c, sigma, delta = 1.7, 0.32, 0.335
    z_min, v_min = wall_minimum(form, c, sigma, delta)
    if form == "9-3":
        assert z_min == pytest.approx(0.4 ** (1 / 6) * sigma, rel=1e-12)
        assert v_min == pytest.approx(-np.sqrt(10.0) / 3.0 * c, rel=1e-12)
    elif form == "10-4":
        assert z_min == sigma
        assert v_min == pytest.approx(-0.6 * c, rel=1e-12)
    else:
        assert v_min < -0.6 * c, "tiefere Lagen vertiefen das Minimum gegenüber 10-4"

    def energy(x):
        return wall_energy_force(x, form, c, sigma, delta)[0]

    z = np.linspace(0.6 * sigma, 4.0 * sigma, 400)
    h = 1e-6
    _, f_ref = wall_energy_force(z, form, c, sigma, delta)
    f_fd = -(energy(z + h) - energy(z - h)) / (2 * h)
    big = np.abs(f_ref) > 1e-3 * np.abs(f_ref).max()
    assert (np.abs(f_fd - f_ref)[big] / np.abs(f_ref[big])).max() < 1e-4

    n = len(z)
    ctx = _ctx(1, n)
    xyz = np.zeros((1, n, 3), np.float32)
    xyz[0, :, 0] = np.linspace(0.0, 5.0, n, endpoint=False)
    xyz[0, :, 2] = z
    ctx.set_positions(xyz)
    ctx.set_types(np.broadcast_to(np.int32([0, 1]), (1, n, 2)).copy())
    surf = ctx.add_planar_surface(
        np.array([0.0, c], np.float32), np.array([0.0, sigma], np.float32), form=form, delta=delta
    )
    ctx.compute_forces()
    zf = xyz[0, :, 2].astype(np.float64)
    e_ref, f_ref, clamped = wall(zf, form, float(np.float32(c)), float(np.float32(sigma)), delta)
    assert not clamped.any() and surf.clamp_count() == 0
    f_gpu = ctx.forces()[0, :n, 2].astype(np.float64)
    big = np.abs(f_ref) > 1e-3 * np.abs(f_ref).max()
    assert (np.abs(f_gpu - f_ref)[big] / np.abs(f_ref[big])).max() < 1e-4
    np.testing.assert_allclose(f_gpu, f_ref, rtol=1e-4, atol=1e-5 * np.abs(f_ref).max())
    e_gpu = ctx.energies()[0, TERM["surface"]]
    assert abs(e_gpu - e_ref.sum()) < 1e-5 * np.abs(e_ref).sum()

    # Minimum auf dem Gerät: die Kraft wechselt bei z* das Vorzeichen und verschwindet dort.
    probe = z_min * np.array([1 - 1e-2, 1.0, 1 + 1e-2])
    ctx2 = _ctx(1, 3)
    xyz2 = np.zeros((1, 3, 3), np.float32)
    xyz2[0, :, 0] = [0.0, 1.0, 2.0]
    xyz2[0, :, 2] = probe
    ctx2.set_positions(xyz2)
    ctx2.set_types(np.broadcast_to(np.int32([0, 1]), (1, 3, 2)).copy())
    ctx2.add_planar_surface(
        np.array([0.0, c], np.float32), np.array([0.0, sigma], np.float32), form=form, delta=delta
    )
    ctx2.compute_forces()
    f_probe = ctx2.forces()[0, :3, 2]
    assert f_probe[0] > 0 > f_probe[2], "Kraft wechselt am Minimum das Vorzeichen"
    assert abs(f_probe[1]) < 1e-3 * abs(f_probe[0])
    e_probe = wall_energy_force(probe.astype(np.float32).astype(np.float64), form, c, sigma, delta)
    assert e_probe[0][1] < min(e_probe[0][0], e_probe[0][2])


def test_wall_mixing_rules():
    """ADR 0007: Builder-Tabelle (biooptomd.surface.wall_table) vs. manuell berechnete
    Lorentz-Berthelot-Werte; Oberflächen-Typ 0 und LJ-Typ 0 ohne Wand; λ skaliert ε."""
    from biooptomd import surface
    from biooptomd.nonbonded import lj_type_table

    sig_atoms = np.array([0.33, 0.25, 0.33, 0.30])
    eps_atoms = np.array([0.36, 0.06, 0.36, 0.71])
    _, lj = lj_type_table(sig_atoms, eps_atoms)
    lam = np.array([1.0, 1.0, 0.5])
    for name, mat in surface.MATERIALS.items():
        t = surface.wall_table(name, lj, lam)
        assert t["form"] == mat.form and t["prefactor"].shape == (3, lj.shape[0])
        for s_type in range(3):
            for l_type in range(lj.shape[0]):
                s_l, e_l = (float(v) for v in lj[l_type, l_type])
                s_iw = 0.5 * (s_l + mat.sigma_nm)
                e_iw = np.sqrt(e_l * mat.epsilon_kj_mol) * (lam[s_type] if s_type else 0.0)
                if mat.form == "9-3":
                    c = 2 * np.pi / 3 * mat.density * e_iw * s_iw**3
                elif mat.form == "10-4":
                    c = 2 * np.pi * mat.density * e_iw * s_iw**2
                else:
                    c = 2 * np.pi * mat.density * e_iw * s_iw**2 * mat.delta_nm
                assert t["prefactor"][s_type, l_type] == pytest.approx(c, rel=1e-6, abs=1e-30)
                assert t["sigma"][s_type, l_type] == pytest.approx(s_iw, rel=1e-6)
        assert not t["prefactor"][0].any(), "Oberflächen-Typ 0 ist reserviert (ε = 0)"
        assert not t["prefactor"][:, 0].any(), "LJ-Typ 0 (Padding) ohne Wand"

    # 10-4-Tiefe für Graphen: V(σ) = −(6/5) π ρ_s ε_iw σ_iw² (ADR 0007).
    g = surface.MATERIALS["graphene"]
    t = surface.wall_table("graphene", lj, np.ones(2))
    s_iw = 0.5 * (float(lj[1, 1, 0]) + g.sigma_nm)
    e_iw = np.sqrt(float(lj[1, 1, 1]) * g.epsilon_kj_mol)
    _, depth = wall_minimum("10-4", float(t["prefactor"][1, 1]), float(t["sigma"][1, 1]))
    assert depth == pytest.approx(-1.2 * np.pi * g.density * e_iw * s_iw**2, rel=1e-5)
    with pytest.raises(ValueError):
        surface.wall_table("gold", lj, np.array([1.0, 1.5]))

    # Tabelle pro (Oberflächen-Typ, LJ-Typ) auf dem Gerät: jedes Atom spürt seinen Eintrag.
    n = 6
    ctx = _ctx(1, n)
    xyz = np.zeros((1, n, 3), np.float32)
    xyz[0, :, 0] = np.arange(n)
    xyz[0, :, 2] = 0.45
    types = np.zeros((1, n, 2), np.int32)
    types[0, :, 0] = [1, 2, 3, 1, 2, 0]
    types[0, :, 1] = [1, 1, 2, 2, 0, 1]
    ctx.set_positions(xyz)
    ctx.set_types(types)
    tg = surface.wall_table("graphene", lj, lam)
    surf = ctx.add_planar_surface(tg["prefactor"], tg["sigma"], form=tg["form"], delta=tg["delta"])
    ctx.compute_forces()
    f = ctx.forces()[0, :n, 2].astype(np.float64)
    for i in range(n):
        s_type, l_type = types[0, i, 1], types[0, i, 0]
        c_i = float(tg["prefactor"][s_type, l_type])
        _, f_ref = wall_energy_force(0.45, "10-4", c_i, float(tg["sigma"][s_type, l_type]))
        assert f[i] == pytest.approx(f_ref, rel=1e-5, abs=1e-7)
    assert f[4] == 0.0 and f[5] == 0.0
    assert f[0] != f[3], "gleicher LJ-Typ, anderer Oberflächen-Typ (λ) ⇒ andere Kraft"
    assert surf.clamp_distances().shape == (3, lj.shape[0])
    assert surf.unknown_type_count() == 0


def test_determinism():
    """Gleicher Seed ⇒ bitidentische Trajektorie (inkl. Wand, Restraints, Pulling); anderer Seed
    ⇒ verschieden."""

    def run(seed):
        n, n_rep = 300, 3
        ctx = _ctx(n_rep, n, temperatures=np.array([280.0, 300.0, 320.0]))
        rng = np.random.default_rng(14)
        xyz = rng.uniform(0.0, 5.0, (n_rep, n, 3)).astype(np.float32)
        xyz[..., 2] += 0.5
        ctx.set_positions(xyz)
        ctx.set_charges(rng.uniform(-1, 1, (n_rep, n)).astype(np.float32))
        ctx.set_types(np.broadcast_to(np.int32([0, 1]), (n_rep, n, 2)).copy())
        ctx.add_planar_surface(
            np.array([0.0, 2.0], np.float32),
            np.array([0.0, 0.3], np.float32),
            phi0=2.0,
            kappa=np.ones(n_rep, np.float32),
        )
        replica, atom = np.divmod(np.arange(n_rep * 50, dtype=np.int32), 50)
        ctx.add_position_restraints(
            replica.astype(np.int32),
            atom.astype(np.int32),
            xyz[:, :50].reshape(-1, 3),
            np.full(n_rep * 50, 100.0, np.float32),
        )
        offsets = np.arange(n_rep + 1, dtype=np.int32) * 20
        group = np.tile(np.arange(100, 120, dtype=np.int32), n_rep)
        pull = ctx.add_com_pulling(
            offsets,
            group,
            np.full((n_rep, 3), 2.5, np.float32),
            np.tile(np.float32([1.0, 0.0, 0.0]), (n_rep, 1)),
            np.full(n_rep, 0.1, np.float32),
            np.full(n_rep, 50.0, np.float32),
        )
        ctx.set_integrator(0.002, 1.0, seed)
        ctx.step(2000)
        return (
            ctx.positions(),
            ctx.velocities(),
            ctx.forces_fixed_point(),
            ctx.images(),
            ctx.energies_fixed_point(),
            pull.work(),
        )

    a, b, c = run(7), run(7), run(8)
    for x, y in zip(a, b, strict=True):
        assert np.array_equal(x.view(np.uint8), y.view(np.uint8)), "nicht bitidentisch"
    assert not np.array_equal(a[0], c[0]), "anderer Seed liefert gleiche Trajektorie"


def test_jarzynski_harmonic():
    """Pulling in rein harmonischem Potential: ΔG = −k_BT ln⟨exp(−βW)⟩ = 0 innerhalb 4σ."""
    n_rep, temp, mass, k, v, gamma, dt = 2000, 300.0, 20.0, 100.0, 0.1, 5.0, 0.002
    kt = KB * temp
    ctx = _ctx(n_rep, 1, mass=mass, temperatures=temp, box=10.0)
    rng = np.random.default_rng(15)
    r0 = np.full((n_rep, 3), [5.0, 5.0, 3.0], np.float32)
    # Exakter Gleichgewichtsstart bei t = 0: x ~ N(R₀, k_BT/k), v ~ Maxwell-Boltzmann.
    xyz = r0[:, None, :] + np.zeros((n_rep, 1, 3), np.float32)
    xyz[:, 0, 0] += rng.normal(0.0, np.sqrt(kt / k), n_rep).astype(np.float32)
    ctx.set_positions(xyz)
    ctx.set_velocities(rng.normal(0.0, np.sqrt(kt / mass), (n_rep, 1, 3)).astype(np.float32))
    pull = ctx.add_com_pulling(
        np.arange(n_rep + 1, dtype=np.int32),
        np.zeros(n_rep, np.int32),
        r0,
        np.tile(np.float32([1.0, 0.0, 0.0]), (n_rep, 1)),
        np.full(n_rep, v, np.float32),
        np.full(n_rep, k, np.float32),
    )
    ctx.set_integrator(dt, gamma, seed=21)
    ctx.step(1000)  # 2 ps, Zugstrecke 0.2 nm

    work = pull.work()
    assert work.mean() > 0.0  # zweiter Hauptsatz: ⟨W⟩ ≥ ΔG = 0
    expw = np.exp(-(work - work.min()) / kt)
    delta_g = -kt * np.log(expw.mean()) + work.min()
    sigma = kt * expw.std(ddof=1) / (np.sqrt(n_rep) * expw.mean())  # Delta-Methode
    assert abs(delta_g) < 4.0 * sigma, (delta_g, sigma, work.mean())


# --- Referenzvergleiche und Infrastruktur ----------------------------------------------------


def test_philox_kat_on_device():
    """Philox4x32-10 auf dem Gerät: Random123-KAT und bitgenau gleich der NumPy-Referenz."""
    for counter, key, expected in KAT:
        out = _core.testing.philox(np.array([counter], np.uint32), key[0], key[1])[0]
        assert tuple(int(x) for x in out) == expected
    rng = np.random.default_rng(16)
    counters = rng.integers(0, 2**32, (4096, 4), dtype=np.uint64).astype(np.uint32)
    np.testing.assert_array_equal(
        _core.testing.philox(counters, 0xDEADBEEF, 0x12345678),
        philox4x32_10(counters, np.array([0xDEADBEEF, 0x12345678], np.uint32)),
    )


@pytest.mark.parametrize("form", WALL_FORMS)
def test_surface_vs_numpy(form):
    """Wand + nichtlineares Gouy-Chapman gegen NumPy: Tabelle pro (Oberflächen-Typ, LJ-Typ),
    C = 0, Clamp mit Kraftschranke (ADR 0008), unbekannte Typen, T_r und κ_r pro Replica.
    Alle Kräfte werden verglichen, auch an der Wand."""
    rng = np.random.default_rng(17)
    n = 500
    # (Oberfläche, LJ); Zeile 0 reserviert (ADR 0007), Zeile 3 mit C = 0 und σ > 0
    eps = np.array([[0.0, 0.0], [2.0, 1.0], [5.0, 3.0], [0.0, 0.0]], np.float32)
    sigma = np.array([[0.0, 0.0], [0.3, 0.35], [0.25, 0.28], [0.3, 0.3]], np.float32)
    z0, clamp, phi0, delta = 0.2, 0.1, 3.0, 0.335
    kappa = np.array([1.5, 1.3], np.float32)
    temps = np.array([300.0, 330.0], np.float32)
    ctx = _ctx(2, n, temperatures=temps)
    xyz = rng.uniform(0.0, 5.0, (2, n, 3)).astype(np.float32)
    xyz[..., 2] = rng.uniform(z0 + 0.01, z0 + 2.0, (2, n))  # inkl. Bereich unterhalb ζ_c
    xyz[:, :5, 2] = z0 + 0.01  # sicher unterhalb ζ_c ⇒ Clamp
    types = np.zeros((2, n, 2), np.int32)
    types[..., 0] = rng.integers(0, 2, (2, n))
    types[..., 1] = rng.integers(0, 4, (2, n))
    types[:, :5, 1] = 1
    types[:, :5, 0] = 0
    types[1, -1, 1] = 7  # unbekannter Oberflächen-Typ ⇒ keine Wand, Zähler
    types[1, -2, 0] = 5  # unbekannter LJ-Typ ⇒ keine Wand, Zähler
    q = rng.uniform(-1, 1, (2, n)).astype(np.float32)
    ctx.set_positions(xyz)
    ctx.set_types(types)
    ctx.set_charges(q)
    surf = ctx.add_planar_surface(
        eps,
        sigma,
        form=form,
        delta=delta,
        z0=z0,
        clamp_factor=clamp,
        phi0=phi0,
        kappa=kappa,
    )
    ctx.compute_forces()

    t, lt = types[..., 1], types[..., 0]
    known = (t < eps.shape[0]) & (lt < eps.shape[1])
    e_t = np.where(known, eps[np.minimum(t, 3), np.minimum(lt, 1)], 0.0)
    s_t = np.where(known, sigma[np.minimum(t, 3), np.minimum(lt, 1)], 0.0)
    e_ref = np.zeros((2, n))
    f_ref = np.zeros((2, n))
    clamped = np.zeros((2, n), bool)
    for r in range(2):
        e_ref[r], f_ref[r], clamped[r] = surface_energy_force(
            xyz[r, :, 2].astype(np.float64),
            q[r].astype(np.float64),
            e_t[r],
            s_t[r],
            form,
            delta,
            z0,
            clamp,
            phi0,
            float(kappa[r]),
            KB * float(temps[r]),
        )
    f = ctx.forces()[:, :n].astype(np.float64)
    np.testing.assert_allclose(f[..., 2], f_ref, rtol=2e-5, atol=1e-4)
    assert np.abs(f[..., 2]).max() <= 1.0e6 * (1 + 1e-5), "Kraftschranke am Clamp verletzt"
    assert not f[..., :2].any()
    e_gpu = ctx.energies()[:, TERM["surface"]]
    np.testing.assert_allclose(e_gpu, e_ref.sum(axis=1), rtol=1e-5)
    assert surf.clamp_count() == clamped.sum() >= 10
    assert surf.unknown_type_count() == 2
    zeta_c = np.vectorize(lambda e, s: clamp_distance(form, float(e), float(s), delta, clamp))(
        eps, sigma
    )
    np.testing.assert_allclose(surf.clamp_distances(), zeta_c, rtol=1e-6)
    assert zeta_c[1, 0] > clamp * sigma[1, 0], "ζ_F muss für C = 2 den 0.1·σ-Clamp dominieren"
    with pytest.raises(ValueError):  # Oberflächen-Typ 0 ist reserviert
        ctx.add_planar_surface(np.array([1.0], np.float32), np.array([0.3], np.float32))
    assert ctx.status() == 0, "keine Sättigung bei begrenzter Wandkraft"
    assert not ctx.forces()[:, n:].any(), "Padding-Atome erhalten Kraft"


def test_restraint_vs_numpy():
    """Positions-Restraints (Minimum Image, Achsenmaske) und Schwerpunkt-Pulling (ungewrappt,
    Massengewichte, Arbeit nach einem Schritt) gegen NumPy."""
    rng = np.random.default_rng(18)
    n, n_rep, box = 64, 2, 4.0
    ctx = _ctx(n_rep, n, box=box)
    masses = rng.uniform(1.0, 30.0, (n_rep, n)).astype(np.float32)
    ctx.set_masses(masses)
    xyz = rng.uniform(-6.0, 10.0, (n_rep, n, 3)).astype(np.float32)  # teils außerhalb ⇒ Images
    ctx.set_positions(xyz)
    ctx.wrap_xy()

    m = 40
    replica = rng.integers(0, n_rep, m).astype(np.int32)
    atom = rng.choice(n, m, replace=False).astype(np.int32)
    ref = rng.uniform(0.0, box, (m, 3)).astype(np.float32)
    k = rng.uniform(10.0, 500.0, m).astype(np.float32)
    ctx.add_position_restraints(replica, atom, ref, k, dims=(True, False, True))

    offsets = np.array([0, 10, 25], np.int32)
    group = np.concatenate([rng.choice(n, 10, replace=False), rng.choice(n, 15, replace=False)])
    group = group.astype(np.int32)
    r0 = rng.uniform(0.0, box, (n_rep, 3)).astype(np.float32)
    direction = np.array([[1.0, 2.0, 0.5], [0.0, -1.0, 1.0]], np.float32)
    vel = np.array([0.3, -0.2], np.float32)
    kp = np.array([80.0, 150.0], np.float32)
    pull = ctx.add_com_pulling(offsets, group, r0, direction, vel, kp)
    ctx.set_integrator(0.002, 1.0, seed=3)
    ctx.compute_forces()

    pos = ctx.positions()[:, :n, :3].astype(np.float64)
    img = ctx.images()[:, :n].astype(np.float64)
    boxes = np.full((n_rep, 2), box)
    f_ref = np.zeros((n_rep, n, 3))
    e_ref = np.zeros(n_rep)
    e, f = position_restraint(pos[replica, atom], ref, k, [1, 0, 1], boxes[replica])
    np.add.at(f_ref, (replica, atom), f)
    np.add.at(e_ref, replica, e)
    unwrapped = pos.copy()
    unwrapped[..., :2] += img * box
    for r in range(n_rep):
        g = group[offsets[r] : offsets[r + 1]]
        w = masses[r, g].astype(np.float64)
        w /= w.sum()
        e_hat = direction[r] / np.linalg.norm(direction[r].astype(np.float64))
        e, f, _ = com_pulling(unwrapped[r, g], w, r0[r], e_hat, vel[r], kp[r], 0.0)
        f_ref[r, g] += f
        e_ref[r] += e
    np.testing.assert_allclose(ctx.forces()[:, :n, :3], f_ref, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(ctx.energies()[:, TERM["restraint"]], e_ref, rtol=1e-5)
    assert not pull.work().any()

    # Ein Schritt: ΔW = V(x₁, Δt) − V(x₁, 0) bei den Positionen der Kraftauswertung.
    ctx.step(1)
    pos = ctx.positions()[:, :n, :3].astype(np.float64)
    unwrapped = pos.copy()
    unwrapped[..., :2] += ctx.images()[:, :n] * box
    for r in range(n_rep):
        g = group[offsets[r] : offsets[r + 1]]
        w = masses[r, g].astype(np.float64)
        w /= w.sum()
        e_hat = direction[r] / np.linalg.norm(direction[r].astype(np.float64))
        dw = com_pulling_work_increment(unwrapped[r, g], w, r0[r], e_hat, vel[r], kp[r], 0.0, 0.002)
        np.testing.assert_allclose(pull.work()[r], dw, rtol=1e-4, atol=1e-6)


def test_nve_harmonic():
    """γ = 0: Gesamtenergie in harmonischen Restraints erhalten (Plausibilität von B/A/O-Reihenfolge)."""
    n, k, dt = 500, 300.0, 0.001
    ctx = _ctx(2, n)
    rng = np.random.default_rng(19)
    ref = rng.uniform(1.0, 4.0, (2, n, 3)).astype(np.float32)
    ctx.set_positions(ref + rng.normal(0.0, 0.05, ref.shape).astype(np.float32))
    ctx.set_velocities(rng.normal(0.0, 0.3, (2, n, 3)).astype(np.float32))
    replica, atom = np.divmod(np.arange(2 * n, dtype=np.int32), n)
    ctx.add_position_restraints(
        replica.astype(np.int32),
        atom.astype(np.int32),
        ref.reshape(-1, 3),
        np.full(2 * n, k, np.float32),
    )
    ctx.set_integrator(dt, 0.0, seed=0)
    tracker = ctx.add_energy_tracker(stride=10, capacity=1000)
    ctx.step(5000)
    e = tracker.result()["energies"]
    total = e[:, :, 0] + e[:, :, TERM["restraint"]]
    # Gemessen ~3e-6; ein O(ωΔt)-Fehler (z. B. falsche B/A-Reihenfolge) läge bei ~4e-3.
    assert np.all(total.std(axis=1) < 5e-5 * e[:, :, 0].mean(axis=1))
    assert np.all(np.abs(total[:, -1] - total[:, 0]) < 5e-5 * e[:, :, 0].mean(axis=1))


def test_integrator_api_and_padding():
    ctx = _core.SimulationContext(2, 40)
    with pytest.raises(RuntimeError):
        ctx.step(1)  # kein Integrator
    with pytest.raises(ValueError):
        ctx.set_integrator(0.0, 1.0, 0)
    ctx.set_integrator(0.002, 1.0, 0)
    with pytest.raises(RuntimeError):
        ctx.step(1)  # keine Box
    ctx.set_box(np.full((2, 2), 3.0, np.float32))
    ctx.set_masses(np.full((2, 40), 10.0, np.float32))
    ctx.set_temperatures(np.full(2, 300.0, np.float32))
    ctx.step(100)
    assert ctx.current_step == 100 and ctx.time == pytest.approx(0.2)
    assert not ctx.positions()[:, 40:].any() and not ctx.velocities()[:, 40:].any()
    assert ctx.velocities()[:, :40, :3].any()
    tracker = ctx.add_energy_tracker(stride=3, capacity=5)
    ctx.step(30)
    assert tracker.n_frames == 5 and tracker.dropped_frames == 5
    np.testing.assert_array_equal(tracker.result()["steps"], [102, 105, 108, 111, 114])


@pytest.mark.slow
@pytest.mark.parametrize("seed", [101, 202, 303, 404, 505])
def test_equipartition_and_velocities_seeds(seed):
    """TESTING.md §Statistische Toleranzen 5: Equipartition und Maxwell-Boltzmann für 5 Seeds."""
    temps = np.array([250.0, 400.0])
    gamma, dt, n = 2.0, 0.002, 2000
    ctx = _ctx(2, n, mass=12.0, temperatures=temps)
    ctx.set_integrator(dt, gamma, seed=seed)
    ctx.step(int(5.0 / gamma / dt))
    tracker = ctx.add_energy_tracker(stride=10, capacity=4000)
    ctx.step(30000)
    mean, sigma = _block_mean_sigma(tracker.result()["energies"][:, :, 0])
    expected = 0.5 * ctx.degrees_of_freedom() * KB * temps
    assert np.all(np.abs(mean - expected) < 4.0 * sigma), (mean, expected, sigma)
    v = ctx.velocities()[:, :n, :3].astype(np.float64)
    for r, t in enumerate(temps):
        assert stats.kstest((v[r] * np.sqrt(12.0 / (KB * t))).ravel(), "norm").pvalue > 1e-3


def test_pulling_follows_mass_change_and_dt_change():
    """Gewichte folgen set_masses nach add_com_pulling; t läuft bei dt-Wechsel fortlaufend weiter."""
    ctx = _ctx(1, 2, box=6.0)
    ctx.set_masses(np.array([[1.0, 1.0]], np.float32))
    xyz = np.array([[[1.0, 3.0, 3.0], [3.0, 3.0, 3.0]]], np.float32)
    ctx.set_positions(xyz)
    pull = ctx.add_com_pulling(
        np.array([0, 2], np.int32),
        np.array([0, 1], np.int32),
        np.zeros((1, 3), np.float32),
        np.array([[1.0, 0.0, 0.0]], np.float32),
        np.zeros(1, np.float32),
        np.ones(1, np.float32),
    )
    ctx.compute_forces()
    assert pull.extension()[0] == pytest.approx(2.0)  # Schwerpunkt bei x = 2
    ctx.set_masses(np.array([[3.0, 1.0]], np.float32))
    ctx.compute_forces()
    assert pull.extension()[0] == pytest.approx(1.5)  # neuer Schwerpunkt (3·1 + 1·3)/4

    ctx.set_integrator(0.002, 1.0, seed=0)
    ctx.step(10)
    ctx.set_integrator(0.001, 1.0, seed=0)
    ctx.step(10)
    assert ctx.time == pytest.approx(0.03)


def test_fixed_point_saturation_and_status():
    """ADR 0008: Beiträge > 2^30 werden gesättigt, NaN zu 0; beides setzt Statusbits."""
    ctx = _ctx(1, 4)
    assert ctx.status() == 0
    atom = np.array([0, 0, 1, 2], np.int32)
    f = np.array(
        [[1e12, -1e12, 5.0], [1e3, 0.0, 0.0], [3.0, 0.0, 0.0], [np.nan, 1.0, 0.0]], np.float32
    )
    ctx.clear_forces()
    _core.testing.accumulate_forces(ctx, atom, f)
    ctx.fixed_point_to_float()
    fp = ctx.forces_fixed_point()[:, 0, :4]
    limit = 2**30 * 2**32
    # Ein gesättigter Beitrag + 1e3. (Zwei gesättigte gleichgerichtete Beiträge würden 2^63
    # erreichen und überlaufen – laut ADR 0008 nicht verhinderbar, aber per Flag erkannt.)
    assert fp[0, 0] == limit + 1000 * 2**32
    assert fp[1, 0] == -limit and fp[2, 0] == 5 * 2**32
    assert fp[0, 1] == 3 * 2**32
    assert fp[0, 2] == 0 and fp[1, 2] == 2**32  # NaN → 0, Rest unverändert
    status = ctx.status()
    assert status & _core.STATUS_FIXED_POINT_SATURATED
    assert status & _core.STATUS_NON_FINITE
    assert not status & _core.STATUS_WRAP_RANGE
    ctx.clear_status()
    assert ctx.status() == 0


def test_wrap_range_status():
    """ADR 0008: wrap_xy mit |x/L| >= 2^20 setzt STATUS_WRAP_RANGE (z. B. explodierende Atome)."""
    ctx = _ctx(1, 4, box=2.0)
    ctx.set_positions(np.ones((1, 4, 3), np.float32))
    ctx.wrap_xy()
    assert ctx.status() == 0
    _core.testing.shift_x(ctx, 3.0e6)  # 3e6 / 2 > 2^20
    ctx.wrap_xy()
    assert ctx.status() == _core.STATUS_WRAP_RANGE
