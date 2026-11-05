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
