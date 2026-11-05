"""Erzeugt tests/reference/inputs/dna_trinucleotide.pdb (einmalig, Ergebnis ist committet).

Quelle: RCSB PDB 1BNA (Dickerson-Drew-Dodecamer, B-DNA), Strang A, Nukleotide 1–3 (5'-CGC-3').
Wasserstoffe werden mit OpenMM (amber14/DNA.OL15) ergänzt; die Struktur wird kurz in Vakuum mit dem
Kraftfeld minimiert, damit die Testgeometrie keine extremen Kräfte enthält.

Aufruf (Netzzugang nötig):  python tests/reference/make_trinucleotide.py
"""

from __future__ import annotations

import io
import urllib.request
from pathlib import Path

import openmm
from openmm import app, unit

URL = "https://files.rcsb.org/download/1BNA.pdb"
OUT = Path(__file__).resolve().parent / "inputs" / "dna_trinucleotide.pdb"
FORCEFIELD = ("amber14/DNA.OL15.xml",)


def main() -> None:
    text = urllib.request.urlopen(URL, timeout=30).read().decode()
    lines = [
        ln
        for ln in text.splitlines()
        if ln.startswith("ATOM") and ln[21] == "A" and int(ln[22:26]) <= 3
    ]
    pdb = app.PDBFile(io.StringIO("\n".join(lines + ["END"])))
    modeller = app.Modeller(pdb.topology, pdb.positions)
    # 5'-Ende von 1BNA trägt kein Phosphat. Die Residuennamen bleiben DC/DG/DC: addHydrogens
    # erkennt die Termini selbst, createSystem ordnet die Templates (DC5, DG, DC3) über den
    # Bindungsgraphen zu.
    forcefield = app.ForceField(*FORCEFIELD)
    modeller.addHydrogens(forcefield)
    system = forcefield.createSystem(modeller.topology, nonbondedMethod=app.NoCutoff)
    integrator = openmm.VerletIntegrator(1.0 * unit.femtosecond)
    sim = app.Simulation(
        modeller.topology, system, integrator, openmm.Platform.getPlatformByName("Reference")
    )
    sim.context.setPositions(modeller.positions)
    sim.minimizeEnergy(maxIterations=200)
    positions = sim.context.getState(getPositions=True).getPositions()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as f:
        app.PDBFile.writeFile(modeller.topology, positions, f, keepIds=True)
    print(f"{OUT}: {modeller.topology.getNumAtoms()} Atome")


if __name__ == "__main__":
    main()
