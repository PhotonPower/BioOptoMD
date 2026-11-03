"""OpenMM-Referenz für die kovalenten Terme (ROADMAP Phase 3, Aufgabe 4).

Testmolekül: DNA-Trinukleotid 5'-CGC-3' (inputs/dna_trinucleotide.pdb), AMBER OL15
(ASSUMPTION(OQ-03)), Vakuum, NoCutoff. Das Referenzsystem enthält nur Bonds, Angles, Torsionen und
die 1-4-Paare; die 1-4-Elektrostatik ist wie in BioOptoMD abgeschirmt (Debye-Hückel), daher als
CustomBondForce: V = 4ε[(σ/r)^12 − (σ/r)^6] + C qq exp(−κr)/(ε_r r) mit den bereits skalierten
Exception-Parametern aus OpenMMs NonbondedForce (AMBER: s_el = 1/1.2, s_LJ = 1/2).

Benötigt OpenMM (Marker `reference`).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import openmm
from openmm import app, unit

PDB = Path(__file__).resolve().parent / "inputs" / "dna_trinucleotide.pdb"
FORCEFIELD = ("amber14/DNA.OL15.xml",)
ONE_4PI_EPS0 = 138.935456
GROUPS = {"bond": 0, "angle": 1, "torsion": 2, "pair14": 3}


def load_system():
    pdb = app.PDBFile(str(PDB))
    ff = app.ForceField(*FORCEFIELD)
    system = ff.createSystem(pdb.topology, nonbondedMethod=app.NoCutoff, constraints=None)
    positions = np.array(pdb.getPositions(asNumpy=True).value_in_unit(unit.nanometer))
    return system, positions


def extract_topology(system):
    """Parameter aus einem OpenMM-System → dict von NumPy-Arrays (Einheiten nm, rad, kJ/mol, e)."""
    forces = {type(f).__name__: f for f in system.getForces()}
    hb = forces["HarmonicBondForce"]
    bonds, bond_r0, bond_k = [], [], []
    for b in range(hb.getNumBonds()):
        i, j, r0, k = hb.getBondParameters(b)
        bonds.append((i, j))
        bond_r0.append(r0.value_in_unit(unit.nanometer))
        bond_k.append(k.value_in_unit(unit.kilojoule_per_mole / unit.nanometer**2))
    ha = forces["HarmonicAngleForce"]
    angles, angle_t0, angle_k = [], [], []
    for a in range(ha.getNumAngles()):
        i, j, k_, t0, k = ha.getAngleParameters(a)
        angles.append((i, j, k_))
        angle_t0.append(t0.value_in_unit(unit.radian))
        angle_k.append(k.value_in_unit(unit.kilojoule_per_mole / unit.radian**2))
    pt = forces["PeriodicTorsionForce"]
    tors, tor_n, tor_ph, tor_k = [], [], [], []
    for t in range(pt.getNumTorsions()):
        i, j, k_, l_, n, ph, k = pt.getTorsionParameters(t)
        tors.append((i, j, k_, l_))
        tor_n.append(n)
        tor_ph.append(ph.value_in_unit(unit.radian))
        tor_k.append(k.value_in_unit(unit.kilojoule_per_mole))
    nb = forces["NonbondedForce"]
    pairs, qq, sig, eps = [], [], [], []
    for e in range(nb.getNumExceptions()):
        i, j, q, s, ep = nb.getExceptionParameters(e)
        q = q.value_in_unit(unit.elementary_charge**2)
        ep = ep.value_in_unit(unit.kilojoule_per_mole)
        if q == 0.0 and ep == 0.0:
            continue  # 1-2/1-3-Ausschluss
        pairs.append((i, j))
        qq.append(q)
        sig.append(s.value_in_unit(unit.nanometer))
        eps.append(ep)
    charges = np.array(
        [
            nb.getParticleParameters(i)[0].value_in_unit(unit.elementary_charge)
            for i in range(nb.getNumParticles())
        ]
    )
    masses = np.array(
        [
            system.getParticleMass(i).value_in_unit(unit.dalton)
            for i in range(system.getNumParticles())
        ]
    )
    i32 = lambda x: np.array(x, dtype=np.int32).reshape(-1, len(x[0]) if x else 1)
    return {
        "bonds": i32(bonds), "bond_k": np.array(bond_k), "bond_r0": np.array(bond_r0),
        "angles": i32(angles), "angle_k": np.array(angle_k), "angle_theta0": np.array(angle_t0),
        "torsions": i32(tors), "torsion_k": np.array(tor_k), "torsion_n": np.array(tor_n, np.int32),
        "torsion_phase": np.array(tor_ph),
        "pairs14": i32(pairs), "pair_qq": np.array(qq), "pair_sigma": np.array(sig), "pair_eps": np.array(eps),
        "charges": charges, "masses": masses,
    }  # fmt: skip


def _f32(x):
    return float(np.float32(x))


def _round_parameters_f32(force):
    """Setzt alle Parameter einer Bond/Angle/Torsion-Kraft auf ihre FP32-Rundung (wie auf der GPU)."""
    if isinstance(force, openmm.HarmonicBondForce):
        for b in range(force.getNumBonds()):
            i, j, r0, k = force.getBondParameters(b)
            force.setBondParameters(b, i, j, _f32(r0.value_in_unit(unit.nanometer)), _f32(k._value))
    elif isinstance(force, openmm.HarmonicAngleForce):
        for a in range(force.getNumAngles()):
            i, j, k_, t0, k = force.getAngleParameters(a)
            force.setAngleParameters(
                a, i, j, k_, _f32(t0.value_in_unit(unit.radian)), _f32(k._value)
            )
    elif isinstance(force, openmm.PeriodicTorsionForce):
        for t in range(force.getNumTorsions()):
            i, j, k_, l_, n, ph, k = force.getTorsionParameters(t)
            force.setTorsionParameters(
                t, i, j, k_, l_, n, _f32(ph.value_in_unit(unit.radian)), _f32(k._value)
            )


def reference_system(system, kappa, eps_r, float32_params=False):
    """Neues System: Bonds/Angles/Torsionen (Gruppen 0–2) + 1-4 als CustomBondForce (Gruppe 3).

    float32_params: alle Parameter auf FP32 runden (wie BioOptoMD sie speichert), damit ein Vergleich
    nur die Arithmetik prüft und nicht die Parameterrundung.
    """
    ref = openmm.System()
    for i in range(system.getNumParticles()):
        ref.addParticle(system.getParticleMass(i))
    for f in system.getForces():
        name = type(f).__name__
        if name in ("HarmonicBondForce", "HarmonicAngleForce", "PeriodicTorsionForce"):
            g = openmm.XmlSerializer.clone(f)
            if float32_params:
                _round_parameters_f32(g)
            g.setForceGroup(
                {"HarmonicBondForce": 0, "HarmonicAngleForce": 1, "PeriodicTorsionForce": 2}[name]
            )
            ref.addForce(g)
    topo = extract_topology(system)
    pair = openmm.CustomBondForce("4*eps*((sig/r)^12-(sig/r)^6) + C*qq*exp(-kappa*r)/(epsr*r)")
    pair.addGlobalParameter("C", ONE_4PI_EPS0)
    pair.addGlobalParameter("kappa", kappa)
    pair.addGlobalParameter("epsr", eps_r)
    for name in ("qq", "sig", "eps"):
        pair.addPerBondParameter(name)
    for (i, j), q, s, e in zip(
        topo["pairs14"], topo["pair_qq"], topo["pair_sigma"], topo["pair_eps"], strict=True
    ):
        pair.addBond(int(i), int(j), [_f32(q), _f32(s), _f32(e)] if float32_params else [q, s, e])
    pair.setForceGroup(3)
    ref.addForce(pair)
    return ref


def evaluate(ref_system, positions):
    """Energien [kJ/mol] und Kräfte [kJ/mol/nm] je Term und gesamt (Reference-Plattform, FP64)."""
    ctx = openmm.Context(
        ref_system, openmm.VerletIntegrator(0.001), openmm.Platform.getPlatformByName("Reference")
    )
    ctx.setPositions(positions * unit.nanometer)
    out = {}
    for name, group in GROUPS.items():
        st = ctx.getState(getEnergy=True, getForces=True, groups={group})
        out[name] = (
            st.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole),
            np.array(
                st.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer)
            ),
        )
    st = ctx.getState(getEnergy=True, getForces=True)
    out["total"] = (
        st.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole),
        np.array(
            st.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer)
        ),
    )
    return out
