"""Erzeugt alle Referenzdaten (benötigt OpenMM). Aufruf: python tests/reference/generate_all.py

- inputs/dna_trinucleotide_topology.npz  (klein, committet): Topologie, Positionen, Massen des
  Testmoleküls aus OpenMM (AMBER OL15), damit GPU-Tests ohne OpenMM laufen.
- data/openmm_bonded.npz (gitignored): OpenMM-Energien und -Kräfte je Term (κ = 1/nm, ε_r = 78.5).

Jede Datei enthält die OpenMM-Version und alle Parameter als Metadaten (TESTING.md).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import openmm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import openmm_bonded

KAPPA, EPS_R = 1.0, 78.5


def main() -> None:
    system, positions = openmm_bonded.load_system()
    topo = openmm_bonded.extract_topology(system)
    meta = {"openmm_version": openmm.__version__, "forcefield": ",".join(openmm_bonded.FORCEFIELD)}
    out = HERE / "inputs" / "dna_trinucleotide_topology.npz"
    np.savez_compressed(
        out, positions=positions, **topo, **{f"meta_{k}": v for k, v in meta.items()}
    )
    print(out)

    ref = openmm_bonded.evaluate(openmm_bonded.reference_system(system, KAPPA, EPS_R), positions)
    data = HERE / "data"
    data.mkdir(exist_ok=True)
    arrays = {}
    for name, (energy, forces) in ref.items():
        arrays[f"energy_{name}"] = energy
        arrays[f"forces_{name}"] = forces
    np.savez_compressed(
        data / "openmm_bonded.npz",
        kappa=KAPPA,
        eps_r=EPS_R,
        **arrays,
        **{f"meta_{k}": v for k, v in meta.items()},
    )
    print(data / "openmm_bonded.npz")


if __name__ == "__main__":
    main()
