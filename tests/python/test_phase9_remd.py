"""Abnahmetests Phase 9.2: Replica Exchange (ADR 0011, ROADMAP Phase 9.2)."""

import numpy as np
import pytest

from biooptomd import builder
from biooptomd import io as bio
from biooptomd.nonbonded import cutoff_from_kappa, debye_kappa, water_permittivity
from biooptomd.remd import ReplicaExchange
from biooptomd.units import KB

pytestmark = pytest.mark.gpu

_core = pytest.importorskip("biooptomd._core")

TEMPS = np.array([300.0, 320.0, 345.0, 370.0])  # ε_r(T) von Wasser gilt bis 373 K
IONIC = 0.15
# r_c der Leiter: aus min κ über alle Zustände (ADR 0006), für frische Vergleichskontexte
R_CUT_LADDER = cutoff_from_kappa(debye_kappa(IONIC, TEMPS, water_permittivity(TEMPS)))[1]


def _charged_system(rng, n=96):
    """Ketten wie in test_phase7 (Bindungen, Winkel, Ladungen) als SystemSpec."""
    from test_phase7 import _small_system

    return _small_system(rng, n)


def _remd_context(spec, temps=TEMPS, constraints=False):
    ctx, info = builder.build_context(
        spec, len(temps), (4.0, 4.0), temps, IONIC, hmr=constraints, constraints=constraints,
        types=np.ones(spec.n_atoms, np.int32),
    )  # fmt: skip
    ctx.add_planar_surface(
        np.array([0.0, 2.0], np.float32), np.array([0.0, 0.3], np.float32), phi0=3.0,
        kappa=info.kappa.astype(np.float32),
    )  # fmt: skip
    ctx.set_integrator(0.002 if constraints else 0.001, 2.0, seed=11)
    return ctx, info


def _fresh_energy(spec, positions, state_temps, box=(4.0, 4.0)):
    """Potentielle Energie je Slot eines frischen Kontexts mit Positionen und Zustandsparametern."""
    ctx, info = builder.build_context(
        spec, len(state_temps), box, state_temps, IONIC, positions=positions,
        types=np.ones(spec.n_atoms, np.int32), r_cutoff=R_CUT_LADDER,
    )  # fmt: skip
    ctx.add_planar_surface(
        np.array([0.0, 2.0], np.float32), np.array([0.0, 0.3], np.float32), phi0=3.0,
        kappa=info.kappa.astype(np.float32),
    )  # fmt: skip
    ctx.compute_forces()
    return ctx.energies()[:, 1:].sum(axis=1)


def test_remd_exchange_mechanics():
    """Jede Entscheidung = unabhängig berechnete Metropolis-Entscheidung mit Kreuzenergien aus
    frischen Kontexten (Hamiltonian-Kriterium, κ(T), ε(T), γ(T)); nach dem Tausch tragen die Slots
    T, κ, ε des neuen Zustands (Energie = frischer Kontext); Geschwindigkeiten exakt skaliert.
    Der frische Kontext erhält den r_c der Leiter (min κ über alle Zustände, ADR 0006)."""
    rng = np.random.default_rng(1)
    spec = _charged_system(rng)
    ctx, _ = _remd_context(spec)
    remd = ReplicaExchange(ctx, TEMPS, ionic_strength_molar=IONIC, seed=5)
    mirror = np.random.default_rng(5)
    beta = 1.0 / (KB * TEMPS)
    n_acc = 0
    max_h_minus_t = 0.0
    for cycle in range(12):
        ctx.step(100)
        x = ctx.positions_precise()[:, : spec.n_atoms, :3]
        v_before = ctx.velocities()[:, : spec.n_atoms, :3].copy()
        state = remd.state.copy()
        u_old = ctx.energies()[:, 1:].sum(axis=1)
        # Erwartete Entscheidung aus frischen Kontexten
        parity = cycle % 2
        slot = np.empty_like(state)
        slot[state] = np.arange(len(state))
        trial = state.copy()
        for i in range(parity, len(TEMPS) - 1, 2):
            a, b = slot[i], slot[i + 1]
            trial[a], trial[b] = i + 1, i
        u_cross = np.array(
            [_fresh_energy(spec, x[r][None], TEMPS[trial[r]][None])[0] for r in range(len(state))]
        )
        np.testing.assert_allclose(u_old, [
            _fresh_energy(spec, x[r][None], TEMPS[state[r]][None])[0] for r in range(len(state))
        ], rtol=1e-9)  # fmt: skip
        expected = state.copy()
        for i in range(parity, len(TEMPS) - 1, 2):
            a, b = slot[i], slot[i + 1]
            j = i + 1
            delta = (
                beta[j] * u_cross[a]
                + beta[i] * u_cross[b]
                - beta[i] * u_old[a]
                - beta[j] * u_old[b]
            )
            # Reines Temperaturkriterium (κ, ε, γ fest) zum Vergleich
            delta_t = (beta[j] - beta[i]) * (u_old[a] - u_old[b])
            max_h_minus_t = max(max_h_minus_t, abs(delta - delta_t))
            if delta <= 0.0 or mirror.random() < np.exp(-delta):
                expected[a], expected[b] = j, i
        accepted = remd.attempt()
        assert np.array_equal(remd.state, expected), cycle
        n_acc += accepted.sum()
        # Slotparameter und Energie wie frischer Kontext im neuen Zustand
        np.testing.assert_array_equal(ctx.temperatures(), TEMPS[remd.state].astype(np.float32))
        u_now = ctx.energies()[:, 1:].sum(axis=1)
        np.testing.assert_allclose(
            u_now,
            [_fresh_energy(spec, x[r][None], TEMPS[remd.state[r]][None])[0] for r in range(4)],
            rtol=1e-9,
        )
        factor = np.sqrt(TEMPS[remd.state] / TEMPS[state])
        np.testing.assert_allclose(
            ctx.velocities()[:, : spec.n_atoms, :3], v_before * factor[:, None, None], rtol=1e-15
        )
    assert 0 < n_acc < 12 * 2, "Test soll angenommene und abgelehnte Tausche enthalten"
    # Der zustandsabhängige Hamiltonian-Anteil ist nicht vernachlässigbar (Faktor e^{0.1} ≈ 1.1 in p)
    assert max_h_minus_t > 0.1, max_h_minus_t
    assert ctx.status() == 0


def test_set_screening():
    """set_screening wirkt auf Nonbonded (DH und GB) und Gouy-Chapman; im GB-Modus bleiben die
    1-4-Terme unabgeschirmt (follow_screening = False), im DH-Modus folgen sie."""
    import test_phase9 as t9

    system, top, pos = t9._duplex()
    pos = pos - pos.mean(axis=0) + [4.0, 4.0, 4.0]
    spec = builder.from_openmm(system, top, pos)
    for solvent in ("debye-huckel", "gb-obc2"):
        ctx, info = builder.build_context(
            spec, 2, (8.0, 8.0), [300.0, 300.0], 0.15, solvent=solvent
        )
        ctx.compute_forces()
        e0 = ctx.energies().copy()
        k = info.kappa.astype(np.float32)
        e = info.eps_r.astype(np.float32)
        ctx.set_screening(
            np.array([k[0], 2 * k[1]], np.float32), np.array([e[0], 0.8 * e[1]], np.float32)
        )
        ctx.compute_forces()
        e1 = ctx.energies()
        np.testing.assert_array_equal(e1[0], e0[0])  # Replica 0 unverändert
        assert e1[1, 5] != e0[1, 5], "Nonbonded folgt κ/ε"
        if solvent == "gb-obc2":
            assert e1[1, 4] == e0[1, 4], "GB: 1-4 bleibt unabgeschirmt"
        else:
            assert e1[1, 4] != e0[1, 4], "DH: 1-4 folgt κ/ε"
        # Gleich einem frisch aufgebauten Kontext, der κ/ε direkt beim Aufbau erhält
        fresh, _ = builder.build_context(
            spec, 2, (8.0, 8.0), [300.0, 300.0], 0.15, solvent=solvent,
            eps_r=[e[0], 0.8 * e[1]], kappa=[k[0], 2 * k[1]], r_cutoff=info.r_cutoff,
        )  # fmt: skip
        fresh.compute_forces()
        np.testing.assert_array_equal(fresh.forces_fixed_point(), ctx.forces_fixed_point())

    # Gouy-Chapman: Wandenergie folgt κ
    spec_s = _charged_system(np.random.default_rng(2))
    ctx, info = _remd_context(spec_s, temps=np.array([300.0, 330.0]))
    ctx.compute_forces()
    w0 = ctx.energies()[:, 6].copy()
    ctx.set_screening(np.full(2, 2.0, np.float32), info.eps_r.astype(np.float32))
    ctx.compute_forces()
    assert np.all(ctx.energies()[:, 6] != w0)


@pytest.mark.slow
def test_remd_sampling_harmonic():
    """N = 20 unabhängige Atome im harmonischen Restraint (k = 100 kJ/mol/nm²) je Slot, 4 Zustände,
    Langevin, Austausch alle 1 ps: je Zustand ⟨U⟩ = (3/2) N k_B T_k innerhalb 4σ (σ aus
    Blockmittelung, 20 Blöcke), Akzeptanzrate je Nachbarpaar vs. Erwartung
    E[min(1, e^{(β_i − β_j)(U_i − U_j)})] mit U_k ~ Gamma(3N/2, k_B T_k), innerhalb 4σ (binomial;
    Korrelation der Energie nach 1 ps: ρ_U = C_x² ≈ 0.02 für ω² = 100/12 ps⁻², γ = 5/ps)."""
    from biooptomd.analysis import block_average

    n_atoms, k_spring, n_rep = 20, 100.0, len(TEMPS)
    ctx = _core.SimulationContext(n_rep, n_atoms)
    ctx.set_box(np.full((n_rep, 2), 10.0, np.float32))
    ctx.set_masses(np.full((n_rep, n_atoms), 12.0, np.float32))
    rng = np.random.default_rng(3)
    x0 = rng.uniform(2.0, 8.0, (n_rep, n_atoms, 3)).astype(np.float32)
    ctx.set_positions(x0)
    rep_idx, atom_idx = np.divmod(np.arange(n_rep * n_atoms, dtype=np.int32), n_atoms)
    ctx.add_position_restraints(
        rep_idx.astype(np.int32), atom_idx.astype(np.int32), x0.reshape(-1, 3),
        np.full(n_rep * n_atoms, k_spring, np.float32),
    )  # fmt: skip
    ctx.set_integrator(0.002, 5.0, seed=21)
    remd = ReplicaExchange(ctx, TEMPS, seed=8)
    ctx.step(2500)
    samples = [[] for _ in TEMPS]

    def record(r):
        u = ctx.energies()[:, 7]  # Restraint-Term
        for slot, state in enumerate(r.state):
            samples[state].append(u[slot])

    n_cycles = 1200
    remd.run(n_cycles, 500, callback=record)
    assert ctx.status() == 0
    for k, t in enumerate(TEMPS):
        mean, sem = block_average(np.array(samples[k]), n_blocks=20)
        expected = 1.5 * n_atoms * KB * t
        assert abs(mean - expected) < 4 * sem, (k, mean, expected, sem)
    mc = np.random.default_rng(4)
    beta = 1.0 / (KB * TEMPS)
    for i in range(n_rep - 1):
        ui = mc.gamma(1.5 * n_atoms, KB * TEMPS[i], 400_000)
        uj = mc.gamma(1.5 * n_atoms, KB * TEMPS[i + 1], 400_000)
        p_expected = np.minimum(1.0, np.exp((beta[i] - beta[i + 1]) * (ui - uj))).mean()
        assert 0.1 < p_expected < 0.9, "Leiter so gewählt, dass die Rate aussagekräftig ist"
        p = remd.accepted[i] / remd.attempts[i]
        sigma = np.sqrt(p_expected * (1 - p_expected) / remd.attempts[i])
        assert abs(p - p_expected) < 4 * sigma, (i, p, p_expected)


def test_remd_checkpoint_bitexact(tmp_path):
    """2M Zyklen == M Zyklen + Checkpoint (Dynamik + Austauschzustand) + M Zyklen in neu aufgebautem
    Kontext: Positionen, Geschwindigkeiten, Zustände, Zähler bitidentisch (mit Constraints)."""
    spec = _charged_system(np.random.default_rng(6))
    m = 6

    def make():
        ctx, _ = _remd_context(spec, constraints=True)
        return ctx, ReplicaExchange(ctx, TEMPS, ionic_strength_molar=IONIC, seed=9)

    ref_ctx, ref = make()
    ref.run(2 * m, 100)
    first_ctx, first = make()
    first.run(m, 100)
    bio.save_checkpoint(first_ctx, tmp_path / "c.npz", remd=first)
    del first_ctx, first
    ctx, remd = make()
    with pytest.raises(ValueError):
        bio.load_checkpoint(ctx, tmp_path / "c.npz")  # REMD-Daten ohne remd=…
    bio.load_checkpoint(ctx, tmp_path / "c.npz", remd=remd)
    remd.run(m, 100)
    a, b = ref_ctx.save_state(), ctx.save_state()
    for key in ("pos", "pos_corr", "vel", "image"):
        assert np.array_equal(a[key].view(np.uint8), b[key].view(np.uint8)), key
    assert np.array_equal(ref.state, remd.state)
    assert np.array_equal(ref.accepted, remd.accepted) and np.array_equal(
        ref.attempts, remd.attempts
    )
    assert ref.accepted.sum() > 0
    np.testing.assert_array_equal(debye_kappa(IONIC, TEMPS, water_permittivity(TEMPS)), remd.kappa)
    # Inkonsistenz Temperaturen ↔ Austauschzustand wird erkannt
    ctx.set_temperatures(np.full(len(TEMPS), 300.0, np.float32))
    with pytest.raises(RuntimeError):
        remd.attempt()


def test_remd_cutoff_warning():
    """Kontext mit nur einer Temperatur gebaut (r_c aus deren κ), Leiter mit kleinerem κ:
    Warnung, weil r_c für die Zustände zu klein ist (ADR 0006)."""
    spec = _charged_system(np.random.default_rng(7))
    ctx, _ = builder.build_context(
        spec, 4, (4.0, 4.0), 370.0, 1.0, types=np.ones(spec.n_atoms, np.int32)
    )
    ctx.set_integrator(0.001, 1.0, seed=1)
    with pytest.warns(UserWarning):
        ReplicaExchange(ctx, TEMPS, ionic_strength_molar=0.01)
