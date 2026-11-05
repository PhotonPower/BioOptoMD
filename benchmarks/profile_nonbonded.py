"""Laufzeit von compute_forces (nur TiledAllPairs) für das Phase-4-Testsystem; Ziel für Nsight Compute.

Aufruf: python benchmarks/profile_nonbonded.py [n_replicas]   (Standard 16)
"""

import sys
import time

import numpy as np

import biooptomd._core as core
from biooptomd.nonbonded import nonbonded_topology, padded_atom_count

d = np.load("tests/reference/inputs/dna_duplex_ions_topology.npz")
n = len(d["masses"])
R = int(sys.argv[1]) if len(sys.argv) > 1 else 16
types, nbt = nonbonded_topology(padded_atom_count(n), d["sigma"], d["epsilon"], d["excluded"])
ctx = core.SimulationContext(R, n)
ctx.set_box(np.tile(d["box"], (R, 1)).astype(np.float32))
ctx.set_positions(np.tile(d["positions"], (R, 1, 1)).astype(np.float32))
ctx.set_charges(np.tile(d["charges"], (R, 1)).astype(np.float32))
t2 = np.zeros((R, n, 2), np.int32)
t2[..., 0] = types
ctx.set_types(t2)
ctx.set_nonbonded_tiled(
    [nbt], np.zeros(R, np.int32), np.full(R, 1.04, np.float32), np.full(R, 78.5, np.float32)
)
for _ in range(3):
    ctx.compute_forces()
ctx.synchronize()
t = time.perf_counter()
N = 200
for _ in range(N):
    ctx.compute_forces()
ctx.synchronize()
print(f"R={R}: compute_forces {1e6 * (time.perf_counter() - t) / N:.1f} us")
