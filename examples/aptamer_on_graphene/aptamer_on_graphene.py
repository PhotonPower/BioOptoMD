"""Oberflächenfixiertes Aptamer auf Graphen (ROADMAP Phase 7, Aufgabe 5).

Ablauf: AMBER-Kraftfeld (OpenMM) → biooptomd.builder → Graphen-Wand (10-4, ADR 0007) + Anker
(OQ-07) → Replicas bei verschiedenen Temperaturen → Observer (Dipol, FRET, n(z), Energie) →
Auswertung (biooptomd.analysis) und Ausgabe (DCD je Replica, n(z) als CSV, Checkpoint).

Platzhalter (klar markiert, siehe docs/OPEN_QUESTIONS.md):
- ASSUMPTION(OQ-03): Als „Aptamer“ dient Strang A des B-DNA-Dodecamers 1BNA (AMBER OL15, aus
  tests/reference/inputs/dna_duplex_ions.pdb); ein Fluorophor mit AMBER-DYES-Parametern fehlt noch.
  Donor- bzw. Dipolsonde: Base des 3'-Nukleotids (G12), Akzeptor: Base des 5'-Nukleotids (C1).
- ASSUMPTION(OQ-07): Anker = O5' des 5'-Nukleotids (Stellvertreter für einen Linker), z-Restraint auf
  ANCHOR_HEIGHT über der Graphenebene.
- ASSUMPTION(OQ-01/02/05): Optik- und Wandparameter aus den unkalibrierten Builder-Tabellen, λ = 1.

Aufruf:  python examples/aptamer_on_graphene/aptamer_on_graphene.py --steps 25000 --out run1
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from biooptomd import analysis, builder, surface
from biooptomd import io as bio

ROOT = Path(__file__).resolve().parents[2]
PDB = ROOT / "tests" / "reference" / "inputs" / "dna_duplex_ions.pdb"
FORCEFIELD = ("amber14/DNA.OL15.xml",)
BOX_NM = (6.0, 6.0)  # Bedeckung Γ = 1/L² ≈ 2.8·10¹² cm⁻² (OQ-09); L < 2 r_c ⇒ Bildsumme
IONIC_STRENGTH = 0.15  # mol/L
N_WATER = builder.OPTICS_TARGETS["dna"]["n_water"]  # Wasser bei der Kalibrierwellenlänge 632.8 nm
ANCHOR_HEIGHT = 0.35  # nm über der Graphenebene (ASSUMPTION(OQ-07))
R0_NM = 5.0  # Förster-Radius (Platzhalter)
DT_PS = 0.004  # mit HMR + H-Constraints
FRICTION = 1.0  # 1/ps
BASE_ATOMS = {"DG": ("N9", "C8", "N7", "C5", "C6", "O6", "N1", "C2", "N2", "N3", "C4"),
              "DC": ("N1", "C2", "O2", "N3", "C4", "N4", "C5", "C6")}  # fmt: skip
DIPOLE_AXIS = {"DG": ("C4", "C6"), "DC": ("N1", "C4")}  # ASSUMPTION(OQ-03): Achse in der Basenebene


def load_strand():
    """Strang A (12 nt) als SystemSpec, so ausgerichtet, dass der Anker unten liegt."""
    from openmm import app, unit

    pdb = app.PDBFile(str(PDB))
    modeller = app.Modeller(pdb.topology, pdb.positions)
    modeller.delete([c for c in modeller.topology.chains() if c.id != "A"])
    ff = app.ForceField(*FORCEFIELD)
    system = ff.createSystem(
        modeller.topology, nonbondedMethod=app.NoCutoff, constraints=None, removeCMMotion=False
    )
    pos = np.array(modeller.getPositions().value_in_unit(unit.nanometer))
    spec = builder.from_openmm(system, modeller.topology, pos)
    residues = list(modeller.topology.residues())
    atom_index = {(a.residue.index, a.name): a.index for a in modeller.topology.atoms()}
    anchor = atom_index[(residues[0].index, "O5'")]

    # Rotation: Vektor Anker → Schwerpunkt auf +z, Anker auf z = ANCHOR_HEIGHT, Mitte der Box.
    com = pos.mean(axis=0)
    u = com - pos[anchor]
    u /= np.linalg.norm(u)
    v = np.cross(u, [0.0, 0.0, 1.0])
    s, c = np.linalg.norm(v), u[2]
    if s > 1e-12:
        k = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]]) / s
        rot = np.eye(3) + s * k + (1 - c) * k @ k
    else:
        rot = np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    pos = (pos - pos[anchor]) @ rot.T
    # Anker auf ANCHOR_HEIGHT; nur falls ein anderes Atom dann unter 0.3 nm läge, alles anheben.
    lift = max(0.0, 0.3 - (pos[:, 2].min() + ANCHOR_HEIGHT))
    pos += [BOX_NM[0] / 2, BOX_NM[1] / 2, ANCHOR_HEIGHT + lift]
    spec.positions = pos

    def probe(res):
        name = res.name.rstrip("35")
        group = [atom_index[(res.index, a)] for a in BASE_ATOMS[name]]
        a, b = (atom_index[(res.index, x)] for x in DIPOLE_AXIS[name])
        return group, (a, b)

    return spec, anchor, probe(residues[-1]), probe(residues[0]), ANCHOR_HEIGHT + lift


def build(n_replicas: int = 8, t_min: float = 300.0, t_max: float = 340.0, seed: int = 2026,
          capacity: int = 1000, stride: int = 250):  # fmt: skip
    """Kontext mit Wand, Anker, Integrator und Observern. Returns (ctx, observers dict, meta)."""
    spec, anchor, donor, acceptor, anchor_z = load_strand()
    temps = np.geomspace(t_min, t_max, n_replicas)
    # α′/v klassenweise an dn/dc der DNA kalibriert (OQ-01/02, ADR 0016 Punkt 7b).
    dna_optics, _ = builder.calibrate_optics("dna")  # skaliert auch Anker-S, Ionen nicht
    types, alpha, volume, names = builder.optical_tables(
        spec, anchor_atoms=[anchor], table=dna_optics
    )
    ctx, info = builder.build_context(
        spec, n_replicas, BOX_NM, temps, IONIC_STRENGTH, hmr=True, constraints=True, types=types
    )
    # Graphen-Wand: Zeilen = Optik-/Oberflächen-Typen, Typ 0 (Anker) ohne Wand, λ = 1.
    wall = surface.wall_table("graphene", info.lj_table, np.ones(len(names)))
    ctx.add_planar_surface(wall["prefactor"], wall["sigma"], form=wall["form"], delta=wall["delta"])
    builder.add_anchors(ctx, [anchor], height_nm=anchor_z - wall["z_top"], z_top_nm=wall["z_top"])
    ctx.set_integrator(DT_PS, FRICTION, seed)

    reps = n_replicas
    d_off, d_atoms, d_a, d_b = builder.dipole_probe([donor[0]] * reps, [donor[1]] * reps)
    a_off, a_atoms, a_a, a_b = builder.dipole_probe([acceptor[0]] * reps, [acceptor[1]] * reps)
    obs = {
        "dipole": ctx.add_dipole_tracker(
            d_off, d_atoms, d_a, d_b, z_surface=wall["z_top"], stride=stride, capacity=capacity
        ),
        "fret": ctx.add_fret_tracker(
            d_off,
            d_atoms,
            d_a,
            d_b,
            a_off,
            a_atoms,
            a_a,
            a_b,
            r0=R0_NM,
            stride=stride,
            capacity=capacity,
        ),
        "nz": ctx.add_refractive_index_profiler(
            0.05, 200, alpha, volume, z_min=0.0, n_water=N_WATER, stride=stride
        ),
        "energy": ctx.add_energy_tracker(stride, capacity),
    }
    meta = {
        "spec": spec, "info": info, "temperatures": temps, "anchor": anchor, "names": names,
        "anchor_z": anchor_z,
    }  # fmt: skip
    return ctx, obs, meta


def summarize(obs, meta, n_water: float = N_WATER):
    """Kennzahlen je Replica aus den Observer-Zeitreihen."""
    fret = obs["fret"].result()
    dip = obs["dipole"].result()["frames"]
    nz = obs["nz"].result()
    out = {
        "temperature": meta["temperatures"],
        "E_dynamic": analysis.fret_efficiency(fret["r_da"], fret["kappa2"], R0_NM, "dynamic"),
        "E_static": analysis.fret_efficiency(fret["r_da"], fret["kappa2"], R0_NM, "static"),
        "R_DA_mean": fret["r_da"].mean(axis=1),
        "d_mean": dip[..., 6].mean(axis=1),
        "cos_theta_mean": dip[..., 5].mean(axis=1),
        "z": nz["z"],
        "n": nz["n"],
    }
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--steps", type=int, default=250_000, help="Zeitschritte (4 fs)")
    p.add_argument("--replicas", type=int, default=8)
    p.add_argument("--out", type=Path, default=Path("aptamer_on_graphene_out"))
    args = p.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    ctx, obs, meta = build(args.replicas)
    chunk = 2500  # 10 ps je DCD-Frame
    with bio.DCDWriter(ctx, str(args.out / "replica_{replica:02d}.dcd"), interval=chunk) as dcd:
        for _ in range(max(1, args.steps // chunk)):
            ctx.step(chunk)
            dcd.write()
    bio.save_checkpoint(ctx, args.out / "checkpoint.npz")
    res = summarize(obs, meta)
    analysis.export_refractive_index(args.out / "n_of_z.csv", res["z"], res["n"])
    for r, t in enumerate(res["temperature"]):
        print(
            f"T = {t:6.1f} K  E_dyn = {res['E_dynamic'][r]:.3f}  E_stat = {res['E_static'][r]:.3f}"
            f"  <R_DA> = {res['R_DA_mean'][r]:.2f} nm  <d> = {res['d_mean'][r]:.2f} nm"
        )
    print(f"Status {ctx.status()}, Ausgabe in {args.out}")


if __name__ == "__main__":
    main()
