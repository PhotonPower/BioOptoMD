"""Erzeugt tests/reference/inputs/dna_duplex_ions.pdb (einmalig, Ergebnis ist committet).

Testsystem für Phase 4 (~1500 Atome): zwei B-DNA-Dodecamer-Duplexe (RCSB PDB 1BNA, beide Stränge,
AMBER OL15) nebeneinander in einer Box 8 × 8 nm (x/y periodisch), neutralisiert mit 40 Na⁺ und 2 Mg²⁺
(Ionenparameter aus amber14/tip3p.xml). ASSUMPTION(OQ-03): Die ROADMAP verlangt „Aptamer +
Fluorophor“; Fluorophor-Parameter (GAFF2 + RESP) sind ohne zusätzliche Werkzeuge nicht verfügbar,
daher als Platzhalter ein zweiter Duplex. Die Ionen werden zufällig (Seed 0) mit Mindestabstand
platziert; anschließend kurze Minimierung mit dem BioOptoMD-Modell (Debye-Hückel + Switching).

Aufruf (Netzzugang nötig):  python tests/reference/make_duplex_ions.py
"""

from __future__ import annotations

import io
import sys
import urllib.request
from pathlib import Path

import numpy as np
import openmm
from openmm import app, unit

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import openmm_nonbonded as onb

URL = "https://files.rcsb.org/download/1BNA.pdb"
BOX = (8.0, 8.0)
N_NA, N_MG = 40, 2


def main() -> None:
    text = urllib.request.urlopen(URL, timeout=30).read().decode()
    lines = [ln for ln in text.splitlines() if ln.startswith("ATOM") and ln[21] in "AB"]
    pdb = app.PDBFile(io.StringIO("\n".join(lines + ["END"])))
    ff = app.ForceField(*onb.FORCEFIELD)
    single = app.Modeller(pdb.topology, pdb.positions)
    single.addHydrogens(ff)
    xyz = np.array(single.getPositions().value_in_unit(unit.nanometer))
    xyz -= xyz.mean(axis=0)

    modeller = app.Modeller(app.Topology(), [])
    for shift in ([2.2, 4.0, 3.5], [5.8, 4.0, 3.5]):
        modeller.add(single.topology, (xyz + shift) * unit.nanometer)

    rng = np.random.default_rng(0)
    dna = np.array(modeller.getPositions().value_in_unit(unit.nanometer))
    ions: list[tuple[str, np.ndarray]] = []
    taken = [p for p in dna]
    for name, count in (("MG", N_MG), ("NA", N_NA)):
        placed = 0
        while placed < count:
            p = rng.uniform([0.0, 0.0, 0.8], [BOX[0], BOX[1], 6.2])
            d = np.array(taken) - p
            d[:, :2] -= np.array(BOX) * np.rint(d[:, :2] / np.array(BOX))
            if np.min(np.einsum("ij,ij->i", d, d)) < 0.6**2:
                continue
            ions.append((name, p))
            taken.append(p)
            placed += 1
    top = app.Topology()
    chain = top.addChain()
    element = {"NA": app.element.sodium, "MG": app.element.magnesium}
    for name, _ in ions:
        res = top.addResidue(name, chain)
        top.addAtom(name, element[name], res)
    modeller.add(top, np.array([p for _, p in ions]) * unit.nanometer)

    system = ff.createSystem(modeller.topology, nonbondedMethod=app.NoCutoff, constraints=None)
    ref = onb.reference_system(system, box=(*BOX, onb.BOX_Z), kappa=1.04, eps_r=78.5)
    ctx = openmm.Context(
        ref, openmm.VerletIntegrator(0.001), openmm.Platform.getPlatformByName("Reference")
    )
    ctx.setPositions(modeller.getPositions())
    openmm.LocalEnergyMinimizer.minimize(ctx, maxIterations=300)
    positions = ctx.getState(getPositions=True).getPositions()
    modeller.topology.setPeriodicBoxVectors(
        [openmm.Vec3(BOX[0], 0, 0), openmm.Vec3(0, BOX[1], 0), openmm.Vec3(0, 0, onb.BOX_Z)]
        * unit.nanometer
    )
    out = HERE / "inputs" / "dna_duplex_ions.pdb"
    with out.open("w") as f:
        app.PDBFile.writeFile(modeller.topology, positions, f, keepIds=False)
    print(f"{out}: {modeller.topology.getNumAtoms()} Atome")


if __name__ == "__main__":
    main()
