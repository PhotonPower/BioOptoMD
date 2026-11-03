"""OpenMM-Referenz für die nicht-kovalenten Terme (ROADMAP Phase 4).

Referenzsystem = kovalente Terme (wie openmm_bonded, inkl. 1-4 mit Debye-Hückel) + CustomNonbondedForce
    V = 4ε[(σ/r)^12 − (σ/r)^6] + C q1 q2 exp(−κr)/(ε_r r),  σ = (σ1+σ2)/2, ε = sqrt(ε1 ε2)
mit Switching (r_s, r_c), CutoffPeriodic und allen Exceptions der NonbondedForce als Ausschlüsse
(1-2, 1-3, 1-4). OpenMM ist in z periodisch; BOX_Z ist so groß gewählt, dass in z keine Bilder
innerhalb r_c liegen (BioOptoMD: z nicht periodisch, ADR 0003).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import openmm
from openmm import app, unit

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import openmm_bonded

PDB = HERE / "inputs" / "dna_duplex_ions.pdb"
FORCEFIELD = ("amber14/DNA.OL15.xml", "amber14/tip3p.xml")
ONE_4PI_EPS0 = 138.935456
BOX_Z = 30.0
NONBONDED_GROUP = 4


def load_system():
    pdb = app.PDBFile(str(PDB))
    ff = app.ForceField(*FORCEFIELD)
    system = ff.createSystem(pdb.topology, nonbondedMethod=app.NoCutoff, constraints=None)
    positions = np.array(pdb.getPositions(asNumpy=True).value_in_unit(unit.nanometer))
    box = pdb.topology.getPeriodicBoxVectors().value_in_unit(unit.nanometer)
    return system, positions, (box[0][0], box[1][1])


def extract_nonbonded(system):
    """Ladungen [e], σ [nm], ε [kJ/mol] je Atom und alle ausgeschlossenen Paare (Exceptions)."""
    nb = next(f for f in system.getForces() if isinstance(f, openmm.NonbondedForce))
    q, sig, eps = [], [], []
    for i in range(nb.getNumParticles()):
        c, s, e = nb.getParticleParameters(i)
        q.append(c.value_in_unit(unit.elementary_charge))
        sig.append(s.value_in_unit(unit.nanometer))
        eps.append(e.value_in_unit(unit.kilojoule_per_mole))
    excl = [nb.getExceptionParameters(k)[:2] for k in range(nb.getNumExceptions())]
    return np.array(q), np.array(sig), np.array(eps), np.array(excl, np.int32).reshape(-1, 2)


def reference_system(
    system, box, kappa, eps_r, r_s=1.8, r_c=2.0, include_bonded=True, float32_params=False
):
    """Kovalente Gruppen 0–3 (optional) + Nonbonded-Gruppe 4."""
    ref = (
        openmm_bonded.reference_system(system, kappa, eps_r, float32_params)
        if include_bonded
        else openmm.System()
    )
    if not include_bonded:
        for i in range(system.getNumParticles()):
            ref.addParticle(system.getParticleMass(i))
    ref.setDefaultPeriodicBoxVectors(
        openmm.Vec3(box[0], 0, 0), openmm.Vec3(0, box[1], 0), openmm.Vec3(0, 0, box[2])
    )
    q, sig, eps, excl = extract_nonbonded(system)
    f = openmm.CustomNonbondedForce(
        "4*eps*((sig/r)^12-(sig/r)^6) + C*q1*q2*exp(-kappa*r)/(epsr*r);"
        "sig=0.5*(sig1+sig2); eps=sqrt(eps1*eps2)"
    )
    for name, value in (("C", ONE_4PI_EPS0), ("kappa", kappa), ("epsr", eps_r)):
        f.addGlobalParameter(name, value)
    for name in ("q", "sig", "eps"):
        f.addPerParticleParameter(name)
    for a in range(len(q)):
        f.addParticle([q[a], sig[a], eps[a]])
    for i, j in excl:
        f.addExclusion(int(i), int(j))
    f.setNonbondedMethod(openmm.CustomNonbondedForce.CutoffPeriodic)
    f.setCutoffDistance(r_c)
    f.setUseSwitchingFunction(True)
    f.setSwitchingDistance(r_s)
    f.setUseLongRangeCorrection(False)
    f.setForceGroup(NONBONDED_GROUP)
    ref.addForce(f)
    return ref


def evaluate(ref_system, positions, groups):
    """Energie [kJ/mol] und Kräfte [kJ/mol/nm] der Kraftgruppen `groups` (Reference, FP64)."""
    ctx = openmm.Context(
        ref_system, openmm.VerletIntegrator(0.001), openmm.Platform.getPlatformByName("Reference")
    )
    ctx.setPositions(positions * unit.nanometer)
    st = ctx.getState(getEnergy=True, getForces=True, groups=set(groups))
    return (
        st.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole),
        np.array(
            st.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer)
        ),
    )
