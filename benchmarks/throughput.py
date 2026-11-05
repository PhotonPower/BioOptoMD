"""Durchsatz BioOptoMD vs. OpenMM (ROADMAP Phase 8).

Gleiches Modell in beiden Engines: AMBER OL15 (DNA) + Ionen, Debye-Hückel (I = 0.15 M, ε_r(T) von
Wasser) mit Switching, r_c aus κ (ADR 0006), 1-4 abgeschirmt, HMR (3.024 amu) und H-Constraints,
Δt = 4 fs, Langevin γ = 1/ps, T = 300 K. OpenMM: CUDA-Plattform (mixed precision),
CustomNonbondedForce mit CutoffPeriodic, ein Context pro Replica, nacheinander in einem Prozess
(in Blöcken zu 32 Contexts).

Systeme (aus tests/reference/inputs/dna_duplex_ions.pdb, Box 8 × 8 nm je Einheit):
  "1k"  – ein Duplex + 20 Na⁺ (≈ 0.8 k Atome)
  "2k"  – zwei Duplexe + 40 Na⁺ + 2 Mg²⁺ (1558 Atome)
  "20k" – 3 × 4 Kopien von "2k" in einer 24 × 32-nm-Box (18 696 Atome)

Aufruf:  python benchmarks/throughput.py [--cases 1x20k 32x2k 128x1k] [--steps 2000]
Ausgabe: Tabelle (Markdown) mit Gesamt-ns/Tag je Engine und Verhältnis.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests" / "reference"))

import openmm
import openmm_nonbonded as onb
from openmm import app, unit

from biooptomd import builder

TEMPERATURE, IONIC, DT_PS, FRICTION, H_MASS = 300.0, 0.15, 0.004, 1.0, 3.024
UNIT_BOX = 8.0


def make_system(name: str):
    """OpenMM-System (ohne Constraints), Topologie, Positionen [nm], Box (L_x, L_y) [nm]."""
    pdb = app.PDBFile(str(onb.PDB))
    modeller = app.Modeller(pdb.topology, pdb.positions)
    box = (UNIT_BOX, UNIT_BOX)
    if name == "1k":
        chains = list(modeller.topology.chains())
        ions = [r for r in chains[4].residues()]
        drop = (
            list(chains[2:4])
            + [r for r in ions if r.name != "NA"][:]
            + [r for r in ions if r.name == "NA"][20:]
        )
        modeller.delete(drop)
    elif name == "20k":
        unit_top, unit_pos = modeller.topology, modeller.getPositions()
        xyz = np.array(unit_pos.value_in_unit(unit.nanometer))
        modeller = app.Modeller(app.Topology(), [])
        for ix in range(3):
            for iy in range(4):
                modeller.add(unit_top, (xyz + [ix * UNIT_BOX, iy * UNIT_BOX, 0.0]) * unit.nanometer)
        box = (3 * UNIT_BOX, 4 * UNIT_BOX)
    elif name != "2k":
        raise ValueError(name)
    ff = app.ForceField(*onb.FORCEFIELD)
    system = ff.createSystem(modeller.topology, nonbondedMethod=app.NoCutoff, constraints=None)
    pos = np.array(modeller.getPositions().value_in_unit(unit.nanometer))
    return system, modeller.topology, pos, box


def bench_biooptomd(spec, pos, box, n_rep, steps, warmup):
    ctx, info = builder.build_context(
        spec, n_rep, box, TEMPERATURE, IONIC, positions=pos, hmr=True, h_mass=H_MASS,
        constraints=True,
    )  # fmt: skip
    ctx.set_integrator(DT_PS, FRICTION, seed=1)
    ctx.step(warmup)
    ctx.synchronize()
    t0 = time.perf_counter()
    ctx.step(steps)
    ctx.synchronize()
    elapsed = time.perf_counter() - t0
    if ctx.status() != 0:
        raise RuntimeError(f"BioOptoMD Statuswort {ctx.status()}")
    return n_rep * steps * DT_PS / elapsed * 86400.0 / 1000.0, info, elapsed


def openmm_system(system, topology, info, box):
    """Referenzsystem wie in Phase 4 plus HMR und H-Constraints."""
    ref = onb.reference_system(
        system, (*box, onb.BOX_Z), float(info.kappa[0]), float(info.eps_r[0]), info.r_switch,
        info.r_cutoff,
    )  # fmt: skip
    for i in range(ref.getNumParticles()):
        ref.setParticleMass(i, float(info.masses[i]))
    atoms = list(topology.atoms())
    hb = next(f for f in system.getForces() if isinstance(f, openmm.HarmonicBondForce))
    for b in range(hb.getNumBonds()):
        i, j, r0, _ = hb.getBondParameters(b)
        if atoms[i].element.symbol == "H" or atoms[j].element.symbol == "H":
            ref.addConstraint(i, j, r0)
    return ref


def bench_openmm(ref, pos, n_rep, steps, warmup, batch=32):
    """Ein Context pro Replica, nacheinander. Mehr als ``batch`` gleichzeitige CUDA-Contexts legt
    OpenMM unter Windows nicht an; die Replicas laufen daher in Blöcken, die Zeiten werden summiert
    (gleichwertig, da die Contexts ohnehin sequentiell rechnen)."""
    platform = openmm.Platform.getPlatformByName("CUDA")
    elapsed = 0.0
    for start in range(0, n_rep, batch):
        contexts = []
        for r in range(start, min(n_rep, start + batch)):
            integ = openmm.LangevinMiddleIntegrator(TEMPERATURE, FRICTION, DT_PS)
            integ.setRandomNumberSeed(r + 1)
            ctx = openmm.Context(ref, integ, platform, {"Precision": "mixed"})
            ctx.setPositions(pos * unit.nanometer)
            ctx.setVelocitiesToTemperature(TEMPERATURE, r + 1)
            integ.step(warmup)
            contexts.append((ctx, integ))
        for ctx, _ in contexts:
            ctx.getState()  # synchronisiert
        t0 = time.perf_counter()
        for _, integ in contexts:
            integ.step(steps)
        for ctx, _ in contexts:
            ctx.getState(getEnergy=True)  # wartet auf alle Kernels
        elapsed += time.perf_counter() - t0
        del contexts
    return n_rep * steps * DT_PS / elapsed * 86400.0 / 1000.0, elapsed


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--cases", nargs="+", default=["1x20k", "32x2k", "128x1k"])
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--openmm-steps", type=int, default=None, help="Default: wie --steps")
    args = p.parse_args(argv)
    rows = []
    for case in args.cases:
        n_rep, name = case.split("x")
        n_rep = int(n_rep)
        system, topology, pos, box = make_system(name)
        spec = builder.from_openmm(system, topology, pos)
        bomd, info, t_b = bench_biooptomd(spec, pos, box, n_rep, args.steps, args.warmup)
        ref = openmm_system(system, topology, info, box)
        omm, t_o = bench_openmm(ref, pos, n_rep, args.openmm_steps or args.steps, args.warmup)
        rows.append((case, spec.n_atoms, info.r_cutoff, bomd, omm, bomd / omm))
        print(
            f"{case}: N = {spec.n_atoms}, r_c = {info.r_cutoff:.2f} nm, BioOptoMD {bomd:.0f} ns/Tag "
            f"({t_b:.1f} s), OpenMM {omm:.0f} ns/Tag ({t_o:.1f} s), Faktor {bomd / omm:.2f}",
            flush=True,
        )
    print("\n| Fall | Atome/Replica | r_c [nm] | BioOptoMD [ns/Tag] | OpenMM [ns/Tag] | Faktor |")
    print("|---|---|---|---|---|---|")
    for case, n, rc, b, o, f in rows:
        print(f"| {case} | {n} | {rc:.2f} | {b:.0f} | {o:.0f} | {f:.2f} |")


if __name__ == "__main__":
    main()
