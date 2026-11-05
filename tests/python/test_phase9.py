"""Abnahmetests Phase 9.1: Generalized Born OBC2 mit Salz (ADR 0010, ROADMAP Phase 9.1).

Testsystem: ein DNA-Duplex (Stränge A, B aus tests/reference/inputs/dna_duplex_ions.pdb, AMBER OL15
+ implicit/obc2.xml; ASSUMPTION(OQ-15): OBC2/mbondi2).
"""

from pathlib import Path

import gb_ref
import numpy as np
import pytest

from biooptomd import builder
from biooptomd.nonbonded import lj_type_table

ONE_4PI_EPS0 = 138.935456
KAPPA = 1.0  # 1/nm
PDB = Path(__file__).resolve().parents[1] / "reference" / "inputs" / "dna_duplex_ions.pdb"


def _rms(f):
    return np.sqrt(np.mean(np.asarray(f) ** 2))


def _duplex(kappa=KAPPA):
    """OpenMM-System mit OBC2 (ACE) und Salz, Topologie, Positionen [nm]."""
    from openmm import app, unit

    pdb = app.PDBFile(str(PDB))
    mod = app.Modeller(pdb.topology, pdb.positions)
    mod.delete([c for c in mod.topology.chains() if c.id not in ("A", "B")])
    ff = app.ForceField("amber14/DNA.OL15.xml", "implicit/obc2.xml")
    system = ff.createSystem(
        mod.topology, nonbondedMethod=app.NoCutoff, constraints=None,
        implicitSolventKappa=kappa / unit.nanometer,
    )  # fmt: skip
    return system, mod.topology, np.array(mod.getPositions().value_in_unit(unit.nanometer))


def _pairs14(spec, pos):
    """Unabgeschirmte 1-4-Terme (Coulomb mit ε_in = 1, LJ) – in OpenMM Teil der NonbondedForce."""
    b = spec.bonded
    i, j = b["pairs14"].T
    d = pos[i] - pos[j]
    r = np.linalg.norm(d, axis=1)
    sr6 = (b["pair_sigma"] / r) ** 6
    v = ONE_4PI_EPS0 * b["pair_qq"] / r + 4 * b["pair_eps"] * (sr6 * sr6 - sr6)
    dv = -ONE_4PI_EPS0 * b["pair_qq"] / r**2 - 4 * b["pair_eps"] * (12 * sr6 * sr6 - 6 * sr6) / r
    f = np.zeros_like(pos)
    fp = (-dv / r)[:, None] * d
    np.add.at(f, i, fp)
    np.add.at(f, j, -fp)
    return v.sum(), f


def _openmm_nonbonded_gb(system, pos):
    """Energie und Kräfte von NonbondedForce + CustomGBForce (Reference-Plattform, FP64)."""
    import openmm
    from openmm import unit

    for g, f in enumerate(system.getForces()):
        f.setForceGroup(g)
    names = [type(f).__name__ for f in system.getForces()]
    groups = {names.index("NonbondedForce"), names.index("CustomGBForce")}
    ctx = openmm.Context(
        system, openmm.VerletIntegrator(0.001), openmm.Platform.getPlatformByName("Reference")
    )
    ctx.setPositions(pos * unit.nanometer)
    st = ctx.getState(getEnergy=True, getForces=True, groups=groups)
    return (
        st.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole),
        np.array(
            st.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer)
        ),
    )


@pytest.mark.reference
@pytest.mark.parametrize("ace", [True, False])
def test_gb_reference_vs_openmm(ace):
    """NumPy-Referenz vs. OpenMM-CustomGBForce (obc2.xml, κ = 1/nm) + NonbondedForce, ohne Cutoff:
    Energie rel. < 1e-6, Kräfte < 1e-5 der RMS-Kraft. (OpenMM rechnet den GB-Teil mit
    C = 138.935485 statt ONE_4PI_EPS0 = 138.935456 ⇒ systematisch 2·10⁻⁷ relativ.)"""
    system, top, pos = _duplex()
    if not ace:
        import openmm

        gb = next(f for f in system.getForces() if isinstance(f, openmm.CustomGBForce))
        ace_term = next(
            k
            for k in range(gb.getNumEnergyTerms())
            if "28.3919551" in gb.getEnergyTermParameters(k)[0]
        )
        expr, kind = gb.getEnergyTermParameters(ace_term)
        gb.setEnergyTermParameters(ace_term, "0*" + expr, kind)
    spec = builder.from_openmm(system, top, pos)
    assert spec.gb_radius is not None and (spec.gb_radius > 0.1).all()
    types, table = lj_type_table(spec.sigma, spec.epsilon)
    e, f, born, terms = gb_ref.gb_nonbonded(
        pos, spec.charges, spec.gb_radius, spec.gb_scale, types, table, spec.excluded, KAPPA, 78.5,
        ace=ace,
    )  # fmt: skip
    assert (terms["ace"] > 0) == ace
    e14, f14 = _pairs14(spec, pos)
    e_omm, f_omm = _openmm_nonbonded_gb(system, pos)
    assert e + e14 == pytest.approx(e_omm, rel=1e-6)
    assert np.abs(f + f14 - f_omm).max() < 1e-5 * _rms(f_omm)
    assert (born >= spec.gb_radius - gb_ref.OFFSET).all()  # B ≥ or (Abschirmung vergrößert B)


@pytest.mark.parametrize(
    "box, n_max, r_s, r_c",
    [(None, 0, None, None), (None, 0, 1.0, 1.2), ((2.5, 2.5), 2, 1.6, 1.8)],
)
def test_gb_reference_forces_fd(box, n_max, r_s, r_c):
    """Referenzkräfte (analytisch, inkl. Kettenregel über die Born-Radien, Switching und
    Bildsumme) vs. finite Differenzen der Referenzenergie: rel. < 1e-5 der größten Kraft."""
    from test_phase4 import _random_system

    rng = np.random.default_rng(3)
    n = 30
    pos, sig, eps, q, excl = _random_system(rng, n, 2.5, 10)
    types, table = lj_type_table(sig, eps)
    rho = rng.uniform(0.12, 0.2, n)
    scale = rng.uniform(0.7, 0.9, n)

    def run(p):
        return gb_ref.gb_nonbonded(
            p, q, rho, scale, types, table, excl, 0.8, 78.5, box=box, n_max=n_max, r_s=r_s, r_c=r_c
        )

    _, f, _, _ = run(pos)
    h = 1e-6
    f_fd = np.zeros_like(pos)
    for a in range(n):
        for d in range(3):
            p = pos.copy()
            p[a, d] += h
            e_plus = run(p)[0]
            p[a, d] -= 2 * h
            f_fd[a, d] = -(e_plus - run(p)[0]) / (2 * h)
    assert np.abs(f - f_fd).max() < 1e-5 * np.abs(f).max()


# --- GPU ------------------------------------------------------------------------------------


@pytest.fixture
def core():
    core = pytest.importorskip("biooptomd._core")
    if not core.cuda_device_count():
        pytest.skip("kein CUDA-Gerät")
    return core


def _f32(x):
    return np.asarray(x, np.float32).astype(np.float64)


def _gb_context(core, n_rep, pos, q, types, topo, rho, scale, box, kappa, eps_out, r_s, r_c, **kw):
    """Kontext nur mit der GB-Nonbonded-Strategie (ohne kovalente Terme)."""
    n = pos.shape[-2]
    ctx = core.SimulationContext(n_rep, n)
    ctx.set_box(np.broadcast_to(np.float32(box), (n_rep, 2)).copy())
    ctx.set_masses(np.full((n_rep, n), 12.0, np.float32))
    ctx.set_positions(np.broadcast_to(pos, (n_rep, n, 3)).astype(np.float32))
    ctx.set_charges(np.broadcast_to(q, (n_rep, n)).astype(np.float32))
    t2 = np.zeros((n_rep, n, 2), np.int32)
    t2[..., 0] = types
    ctx.set_types(t2)
    ctx.set_nonbonded_gb(
        [topo], np.zeros(n_rep, np.int32), np.full(n_rep, kappa, np.float32),
        np.full(n_rep, eps_out, np.float32), [rho], [scale], r_switch=r_s, r_cutoff=r_c, **kw,
    )  # fmt: skip
    ctx.compute_forces()
    return ctx


def _isolated_duplex():
    """Duplex mittig in einer 30-nm-Box, r_c = 12 nm > Molekülausdehnung (kein Switching aktiv)."""
    system, top, pos = _duplex()
    pos = pos - pos.mean(axis=0) + [15.0, 15.0, 10.0]
    spec = builder.from_openmm(system, top, pos)
    assert np.ptp(pos, axis=0).max() < 11.0
    return system, spec, _f32(pos)


@pytest.mark.gpu
@pytest.mark.reference
def test_gb_born_radii_vs_numpy(core):
    """GPU-Born-Radien und Nonbonded-Energie/-Kräfte (LJ + Coulomb + GB + ACE) vs. NumPy-Referenz
    für den isolierten Duplex: Born-Radien rel. < 1e-5, Energie rel. < 1e-5, Kräfte < 1e-4 RMS."""
    from biooptomd.nonbonded import nonbonded_topology, padded_atom_count

    _, spec, pos = _isolated_duplex()
    n = spec.n_atoms
    types, topo = nonbonded_topology(padded_atom_count(n), spec.sigma, spec.epsilon, spec.excluded)
    ctx = _gb_context(
        core, 1, pos, spec.charges, types, topo, spec.gb_radius, spec.gb_scale, (30.0, 30.0),
        KAPPA, 78.5, 11.8, 12.0,
    )  # fmt: skip
    e_ref, f_ref, born, _ = gb_ref.gb_nonbonded(
        pos, _f32(spec.charges), _f32(spec.gb_radius), _f32(spec.gb_scale), types,
        topo["lj_table"].astype(np.float64), spec.excluded, float(np.float32(KAPPA)), 78.5,
    )  # fmt: skip
    b_gpu = ctx.nonbonded_born_radii()
    np.testing.assert_allclose(b_gpu[0, :n], born, rtol=1e-5)
    assert not b_gpu[0, n:].any()
    assert ctx.energies()[0, 5] == pytest.approx(e_ref, rel=1e-5)
    f = ctx.forces()[0, :n, :3].astype(np.float64)
    assert np.abs(f - f_ref).max() < 1e-4 * _rms(f_ref)
    assert ctx.status() == 0


@pytest.mark.gpu
@pytest.mark.reference
def test_gb_vs_openmm(core):
    """Isolierter Duplex, Builder-Weg (solvent="gb-obc2"), alle Terme vs. OpenMM (obc2.xml,
    κ = 1/nm, ACE, NoCutoff, FP64): Energie rel. < 1e-4, Kräfte < 1e-4 der größten RMS-Kraft eines
    Terms (FP32-Rundung skaliert mit den Einzelbeiträgen, wie Phase 4)."""
    import openmm
    from openmm import unit

    from biooptomd.nonbonded import nonbonded_topology

    system, spec, pos = _isolated_duplex()
    ctx, _ = builder.build_context(
        spec, 1, (30.0, 30.0), 300.0, 0.15, positions=pos, solvent="gb-obc2", eps_r=78.5,
        r_cutoff=12.0,
    )  # fmt: skip
    # κ wie im OpenMM-System (1/nm) statt aus der Ionenstärke
    _, nbt = nonbonded_topology(ctx.n_pad, spec.sigma, spec.epsilon, spec.excluded)
    ctx.set_nonbonded_gb(
        [nbt], np.zeros(1, np.int32), np.full(1, KAPPA, np.float32), np.full(1, 78.5, np.float32),
        [spec.gb_radius], [spec.gb_scale], r_switch=11.8, r_cutoff=12.0,
    )  # fmt: skip
    ctx.compute_forces()
    for g, force in enumerate(system.getForces()):
        force.setForceGroup(g)
    octx = openmm.Context(
        system, openmm.VerletIntegrator(0.001), openmm.Platform.getPlatformByName("Reference")
    )
    octx.setPositions(pos * unit.nanometer)
    kj_nm = unit.kilojoule_per_mole / unit.nanometer
    st = octx.getState(getEnergy=True, getForces=True)
    e_ref = st.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    f_ref = np.array(st.getForces(asNumpy=True).value_in_unit(kj_nm))
    scale = max(
        _rms(octx.getState(getForces=True, groups={g}).getForces(asNumpy=True).value_in_unit(kj_nm))
        for g in range(system.getNumForces())
    )
    n = spec.n_atoms
    assert ctx.energies()[0, 1:].sum() == pytest.approx(e_ref, rel=1e-4)
    f = ctx.forces()[0, :n, :3].astype(np.float64)
    assert np.abs(f - f_ref).max() < 1e-4 * scale
    assert ctx.status() == 0


@pytest.mark.gpu
@pytest.mark.parametrize("box, r_c", [((6.0, 6.0), 1.8), ((2.5, 3.0), 1.8)])
def test_gb_cutoff_and_images_vs_numpy(core, box, r_c):
    """Cutoff kleiner als das System (Switching aktiv) im Minimum-Image-Pfad (L ≥ 2 r_c, mit
    Vorauswahl) und mit Bildsumme (L < 2 r_c, Eigenbilder): GPU vs. NumPy-Gittersumme,
    Born-Radien rel. < 1e-5, Kräfte < 1e-4 RMS, Energie rel. < 1e-5."""
    from test_phase4 import _random_system

    from biooptomd.nonbonded import nonbonded_topology, padded_atom_count

    rng = np.random.default_rng(91)
    n = 70
    pos, sig, eps, q, excl = _random_system(rng, n, min(box), 30)
    pos[:, 0] *= box[0] / min(box)
    pos[:, 1] *= box[1] / min(box)
    rho = rng.uniform(0.12, 0.2, n)
    scale = rng.uniform(0.7, 0.9, n)
    types, topo = nonbonded_topology(padded_atom_count(n), sig, eps, excl)
    ctx = _gb_context(core, 1, pos, q, types, topo, rho, scale, box, 0.8, 78.5, r_c - 0.2, r_c)
    n_max = int(np.ceil(r_c / min(box))) + 1
    e_ref, f_ref, born, _ = gb_ref.gb_nonbonded(
        _f32(pos), _f32(q), _f32(rho), _f32(scale), types, topo["lj_table"].astype(np.float64),
        excl, float(np.float32(0.8)), 78.5, box=box, n_max=n_max,
        r_s=float(np.float32(r_c - 0.2)), r_c=float(np.float32(r_c)),
    )  # fmt: skip
    np.testing.assert_allclose(ctx.nonbonded_born_radii()[0, :n], born, rtol=1e-5)
    assert ctx.energies()[0, 5] == pytest.approx(e_ref, rel=1e-5)
    f = ctx.forces()[0, :n, :3].astype(np.float64)
    assert np.abs(f - f_ref).max() < 1e-4 * _rms(f_ref)
    assert ctx.status() == 0


@pytest.mark.gpu
def test_gb_salt_far_field(core):
    """Zwei Ionen bei r ≫ B: Paarenergie E_C + E_GB (E(beide) − E(einzeln)) → Debye-Hückel
    C q_i q_j e^{−κr}/(ε_out r), rel. < 1e-3. Mit s = 0 entschirmen sich die Ionen nicht
    (H ≡ 0, B konstant), sodass die Differenz nur den Paarterm enthält; mit s > 0 käme die
    Änderung der Selbstenergien durch die gegenseitige Entschirmung hinzu (≈ 1e-3 kJ/mol)."""
    from biooptomd.nonbonded import nonbonded_topology

    kappa, eps_out, r = 1.2, 78.5, 2.5
    q = np.array([1.0, -1.0])
    pos = np.array([[5.0, 5.0, 5.0], [5.0 + r, 5.0, 5.0]])
    rho, scale = np.array([0.18, 0.18]), np.zeros(2)
    sig, eps = np.zeros(2), np.zeros(2)

    def energy(p):
        types, topo = nonbonded_topology(32, sig, eps, np.zeros((0, 2), int))
        ctx = _gb_context(
            core, 1, p, q, types, topo, rho, scale, (20.0, 20.0), kappa, eps_out, 4.8, 5.0,
            ace=False,
        )  # fmt: skip
        return ctx.energies()[0, 5]

    far = pos.copy()
    far[1, 2] += 9.0  # außerhalb r_c ⇒ nur Selbstterme
    pair = energy(pos) - energy(far)
    dh = ONE_4PI_EPS0 * q[0] * q[1] * np.exp(-kappa * r) / (eps_out * r)
    assert pair == pytest.approx(dh, rel=1e-3)


@pytest.mark.gpu
def test_gb_padding_batch_determinism(core):
    """Gleiches System allein und als Replica 2 von 4 (andere Replicas gestört): bitidentische
    Kräfte; wiederholter Aufruf bitidentisch; Padding-Atome ohne Kraft."""
    from test_phase4 import _random_system

    from biooptomd.nonbonded import nonbonded_topology, padded_atom_count

    rng = np.random.default_rng(92)
    n = 45
    pos, sig, eps, q, excl = _random_system(rng, n, 4.5, 20)
    rho, scale = rng.uniform(0.12, 0.2, n), rng.uniform(0.7, 0.9, n)
    types, topo = nonbonded_topology(padded_atom_count(n), sig, eps, excl)
    one = _gb_context(core, 1, pos, q, types, topo, rho, scale, (4.5, 4.5), 0.8, 78.5, 2.0, 2.2)
    shake = (np.arange(4) != 2)[:, None, None]
    batch_pos = np.repeat(pos[None], 4, 0) + rng.normal(0, 0.02, (4, n, 3)) * shake
    many = _gb_context(
        core, 4, batch_pos, q, types, topo, rho, scale, (4.5, 4.5), 0.8, 78.5, 2.0, 2.2
    )
    assert np.array_equal(one.forces_fixed_point()[:, 0], many.forces_fixed_point()[:, 2])
    first = many.forces_fixed_point().copy()
    many.compute_forces()
    assert np.array_equal(first, many.forces_fixed_point())
    assert not many.forces()[:, n:].any(), "Padding-Atome erhalten Kraft"
    assert many.status() == 0


@pytest.mark.gpu
@pytest.mark.slow
def test_gb_nve(core):
    """NVE 10 ps, Δt = 1 fs, Duplex mit GB-OBC2 (Cutoff aus κ, 0.15 M, Box 8 × 8 nm): Drift
    < 1e-4 kJ/mol/ps/Atom und Fluktuation < 1e-3·⟨E_kin⟩ (Kriterien aus Phase 3/4)."""
    system, top, pos = _duplex()
    pos = pos - pos.mean(axis=0) + [4.0, 4.0, 4.0]
    spec = builder.from_openmm(system, top, pos)
    n_rep, n = 2, spec.n_atoms
    ctx, _ = builder.build_context(spec, n_rep, (8.0, 8.0), 300.0, 0.15, solvent="gb-obc2")
    rng = np.random.default_rng(93)
    m = spec.masses
    v = rng.normal(0.0, 1.0, (n_rep, n, 3)) * np.sqrt(core.units.KB * 300.0 / m)[None, :, None]
    v -= (v * m[None, :, None]).sum(axis=1, keepdims=True) / m.sum()
    ctx.set_velocities(v)
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
