"""Abnahmetests Phase 9.7 (OQ-14): E_kin nach dem O-Schritt als zweite Temperatur (ROADMAP 9.7)."""

import sys
from pathlib import Path

import numpy as np
import pytest

from biooptomd import builder

KB = 0.0083144626


@pytest.fixture
def core():
    return pytest.importorskip("biooptomd._core")


def _system_2k():
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarks"))
    import throughput

    system, top, pos, box = throughput.make_system("2k")
    return builder.from_openmm(system, top, pos), pos, box, throughput


@pytest.mark.gpu
def test_kinetic_mid_free_particles(core):
    """Ohne Kräfte und mit γ = 0 ändert kein Teilschritt v: kinetic_energy_mid() ist bitgleich zu
    E_kin am vollen Schritt; NaN vor dem ersten Schritt; der Tracker speichert je Frame den Wert des
    beobachteten Schritts (letzter Frame = kinetic_energy_mid() nach step)."""
    n_rep, n = 3, 40
    rng = np.random.default_rng(1)
    ctx = core.SimulationContext(n_rep, n)
    ctx.set_box(np.full((n_rep, 2), 10.0, np.float32))
    ctx.set_positions(rng.uniform(1.0, 9.0, (n_rep, n, 3)).astype(np.float32))
    ctx.set_masses(np.full((n_rep, n), 12.0, np.float32))
    ctx.set_temperatures(np.full(n_rep, 300.0, np.float32))
    ctx.set_velocities(rng.normal(0.0, 0.5, (n_rep, n, 3)))
    ctx.set_integrator(0.002, 0.0, seed=1)
    assert np.all(np.isnan(ctx.kinetic_energy_mid()))
    tracker = ctx.add_energy_tracker(stride=5, capacity=10)
    ctx.step(20)
    mid = ctx.kinetic_energy_mid()
    np.testing.assert_array_equal(mid, ctx.energies()[:, 0])
    res = tracker.result()
    assert res["kinetic_mid"].shape == (n_rep, 4)
    np.testing.assert_array_equal(res["kinetic_mid"], res["energies"][:, :, 0])
    np.testing.assert_array_equal(res["kinetic_mid"][:, -1], mid)


@pytest.mark.gpu
def test_kinetic_mid_constraints(core):
    """Pfad mit Constraints (2k, HMR, 4 fs, Langevin): Tracker-Frames gleich kinetic_energy_mid() im
    jeweils beobachteten Schritt (Lauf in Stücken der Länge stride), endlich, verschieden von E_kin
    am vollen Schritt; deterministisch (zweiter Lauf bitgleich)."""
    spec, pos, box, tp = _system_2k()
    out = []
    for _ in range(2):
        ctx, _ = builder.build_context(spec, 2, box, 300.0, tp.IONIC, positions=pos, hmr=True,
                                       h_mass=tp.H_MASS, constraints=True)  # fmt: skip
        ctx.set_integrator(0.004, 1.0, seed=3)
        tracker = ctx.add_energy_tracker(stride=10, capacity=10)
        direct = []
        for _ in range(3):
            ctx.step(10)
            direct.append(ctx.kinetic_energy_mid())
        res = tracker.result()
        np.testing.assert_array_equal(res["kinetic_mid"], np.array(direct).T)
        assert np.all(np.isfinite(res["kinetic_mid"]))
        assert np.all(res["kinetic_mid"] != res["energies"][:, :, 0])
        assert ctx.status() == 0
        out.append(res["kinetic_mid"])
    np.testing.assert_array_equal(out[0], out[1])


@pytest.mark.gpu
@pytest.mark.slow
def test_temperature_mid_4fs(core):
    """2k-System (HMR, H-Constraints, γ = 1/ps), 300 K, 8 Replicas: T aus E_kin nach O bei
    Δt = 4 fs (100 ps nach 100 ps Äquilibrierung; das System relaxiert in den ersten ≈ 50 ps und gibt
    dabei Wärme ab) innerhalb 4σ (40 Blockmittel à 20 ps) und 1 % von 300 K; die
    Abweichung von E_kin am vollen Schritt skaliert mit Δt² (4 fs vs. 2 fs: Faktor 3–5), d. h. sie ist
    der bekannte Diskretisierungsfehler (PHYSICS_MODEL §1) und nicht der Thermostat."""
    spec, pos, box, tp = _system_2k()
    n_rep = 8
    bias = {}
    for dt_fs, eq_ps, run_ps in ((4.0, 100.0, 100.0), (2.0, 50.0, 40.0)):
        dt = dt_fs * 1e-3
        ctx, _ = builder.build_context(spec, n_rep, box, 300.0, tp.IONIC, positions=pos, hmr=True,
                                       h_mass=tp.H_MASS, constraints=True)  # fmt: skip
        ctx.set_integrator(dt, 1.0, seed=11)
        ctx.step(round(eq_ps / dt))
        tracker = ctx.add_energy_tracker(stride=round(0.2 / dt), capacity=1000)
        ctx.step(round(run_ps / dt))
        res = tracker.result()
        ndof = ctx.degrees_of_freedom().astype(np.float64)[:, None]
        t_mid = 2.0 * res["kinetic_mid"] / (ndof * KB)
        t_full = 2.0 * res["energies"][:, :, 0] / (ndof * KB)
        blocks = t_mid.reshape(n_rep, 5, -1).mean(axis=2).ravel()
        mean, sem = blocks.mean(), blocks.std(ddof=1) / np.sqrt(blocks.size)
        if dt_fs == 4.0:
            assert abs(mean - 300.0) < 4.0 * sem, (mean, sem)
            assert abs(mean - 300.0) < 3.0, mean
        bias[dt_fs] = 300.0 - t_full.mean()
        print(f"dt {dt_fs} fs: T_mid {mean:.2f} ± {sem:.2f} K, T_full {t_full.mean():.2f} K")
        assert ctx.status() == 0
    assert 3.0 < bias[4.0] / bias[2.0] < 5.0, bias
