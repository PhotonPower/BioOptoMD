"""OQ-15-Validierung (a): Stabilität des B-DNA-Duplex 1BNA mit OL21 in GBn2 vs. OBC2.

System: Drew-Dickerson-Dodecamer CGCGAATTCGCG (Stränge A, B aus
tests/reference/inputs/dna_duplex_ions.pdb, Wasserstoffe ergänzt und kurz minimiert), AMBER OL21
(`amber19/DNA.OL21.xml`), GB-Modell `implicit/gbn2.xml` oder `implicit/obc2.xml`, Salz 0.15 M
(κ, ε_r(T), Cutoff aus κ), 300 K, HMR + H-Constraints, Δt = 4 fs, Langevin γ = 1/ps; isoliert in einer
10 × 10-nm-Box (Duplexlänge + r_c < L, keine Wechselwirkung mit Bildern über den Cutoff).

Messgrößen alle 100 ps je Replica (Positionen ungewrappt, Kabsch-Überlagerung):
- RMSD [nm] der Schweratome zur Startstruktur, alle 12 bp und die inneren 8 bp (3–10, ohne Enden).
- Anteil intakter Watson-Crick-Paare: N1(Purin)–N3(Pyrimidin) < 0.35 nm (alle, innere 8 bp).
- Kleinster P–P-Abstand zwischen den Strängen je P, gemittelt [nm] (Maß für die kleine Furche).

Aufruf:  python examples/gb_validation/duplex_stability.py --model gbn2 --ns 100 --out run_gbn2
Ausgabe: <out>/series.npz (Zeitreihen), <out>/summary.json, <out>/checkpoint.npz (Fortsetzung).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from biooptomd import builder
from biooptomd import io as bio

ROOT = Path(__file__).resolve().parents[2]
PDB = ROOT / "tests" / "reference" / "inputs" / "dna_duplex_ions.pdb"
BOX = (10.0, 10.0)
DT, FRICTION, TEMPERATURE, IONIC = 0.004, 1.0, 300.0, 0.15
FRAME_STEPS = 25_000  # 100 ps
CHUNK_FRAMES = 100  # Checkpoint alle 10 ns
HBOND_CUT = 0.35  # nm
PURINES = ("DA", "DG")


def load(model: str):
    """SystemSpec, Topologie-Indizes für die Messgrößen, Startpositionen [nm]."""
    from openmm import app, unit

    pdb = app.PDBFile(str(PDB))
    mod = app.Modeller(pdb.topology, pdb.positions)
    mod.delete([c for c in mod.topology.chains() if c.id not in ("A", "B")])
    ff = app.ForceField("amber19/DNA.OL21.xml", f"implicit/{model}.xml")
    system = ff.createSystem(mod.topology, nonbondedMethod=app.NoCutoff, constraints=None)
    pos = np.array(mod.getPositions().value_in_unit(unit.nanometer))
    pos = pos - pos.mean(axis=0) + [BOX[0] / 2, BOX[1] / 2, 5.0]
    spec = builder.from_openmm(system, mod.topology, pos)
    chains = list(mod.topology.chains())
    res_a, res_b = list(chains[0].residues()), list(chains[1].residues())
    heavy = np.array([a.element.symbol != "H" for a in mod.topology.atoms()])
    inner = np.zeros(len(heavy), bool)
    for chain in (res_a, res_b):
        for k, res in enumerate(chain):
            if 2 <= k <= 9:  # innere 8 bp (Reste 3–10 je Strang)
                for atom in res.atoms():
                    inner[atom.index] = heavy[atom.index]
    pairs = []
    for k in range(12):
        ra, rb = res_a[k], res_b[11 - k]
        atom = {r: {x.name: x.index for x in r.atoms()} for r in (ra, rb)}
        ia = atom[ra]["N1" if ra.name.rstrip("35") in PURINES else "N3"]
        ib = atom[rb]["N1" if rb.name.rstrip("35") in PURINES else "N3"]
        pairs.append((ia, ib))
    p_a = [a.index for r in res_a for a in r.atoms() if a.name == "P"]
    p_b = [a.index for r in res_b for a in r.atoms() if a.name == "P"]
    idx = {"heavy": np.flatnonzero(heavy), "inner": np.flatnonzero(inner),
           "pairs": np.array(pairs), "p_a": np.array(p_a), "p_b": np.array(p_b)}  # fmt: skip
    return spec, idx, pos


def kabsch_rmsd(x, ref):
    """RMSD nach optimaler Überlagerung (Kabsch), x und ref (n, 3)."""
    xc, rc = x - x.mean(0), ref - ref.mean(0)
    u, _, vt = np.linalg.svd(xc.T @ rc)
    d = np.sign(np.linalg.det(u @ vt))
    rot = u @ np.diag([1.0, 1.0, d]) @ vt
    return float(np.sqrt(np.mean(np.sum((xc @ rot - rc) ** 2, axis=1))))


def metrics(x, ref, idx):
    """(RMSD alle, RMSD innen, bp intakt alle, bp intakt innen, mittlerer min P–P) für eine Replica."""
    pa, pb = idx["pairs"][:, 0], idx["pairs"][:, 1]
    intact = np.linalg.norm(x[pa] - x[pb], axis=1) < HBOND_CUT
    dpp = np.linalg.norm(x[idx["p_a"]][:, None] - x[idx["p_b"]][None], axis=-1)
    return (
        kabsch_rmsd(x[idx["heavy"]], ref[idx["heavy"]]),
        kabsch_rmsd(x[idx["inner"]], ref[idx["inner"]]),
        float(intact.mean()),
        float(intact[2:10].mean()),
        float(0.5 * (dpp.min(axis=1).mean() + dpp.min(axis=0).mean())),
    )


def unwrapped(ctx, n):
    x = ctx.positions_precise()[:, :n, :3].copy()
    img = ctx.images()[:, :n].astype(np.float64)
    box = ctx.box().astype(np.float64)
    x[..., 0] += img[..., 0] * box[:, None, 0]
    x[..., 1] += img[..., 1] * box[:, None, 1]
    return x


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", choices=("gbn2", "obc2"), required=True)
    p.add_argument("--ns", type=float, default=100.0, help="Simulationszeit je Replica [ns]")
    p.add_argument("--replicas", type=int, default=4)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    spec, idx, ref = load(args.model)
    ctx, info = builder.build_context(
        spec, args.replicas, BOX, TEMPERATURE, IONIC, solvent=f"gb-{args.model}", hmr=True,
        constraints=True,
    )  # fmt: skip
    ctx.set_integrator(DT, FRICTION, args.seed)
    n_frames = round(args.ns * 1000 / (FRAME_STEPS * DT))
    series_path, ckpt = args.out / "series.npz", args.out / "checkpoint.npz"
    series = np.zeros((0, args.replicas, 5))
    if ckpt.exists() and series_path.exists():
        bio.load_checkpoint(ctx, ckpt)
        series = np.load(series_path)["series"]
    t0 = time.perf_counter()
    while len(series) < n_frames:
        block = []
        for _ in range(min(CHUNK_FRAMES, n_frames - len(series))):
            ctx.step(FRAME_STEPS)
            x = unwrapped(ctx, spec.n_atoms)
            block.append([metrics(x[r], ref, idx) for r in range(args.replicas)])
        if ctx.status() != 0:
            raise RuntimeError(f"Statuswort {ctx.status()}")
        series = np.concatenate([series, np.array(block)])
        np.savez(series_path, series=series, time_ns=np.arange(1, len(series) + 1) * 0.1)
        bio.save_checkpoint(ctx, ckpt)
        print(
            f"{args.model}: {len(series) * 0.1:.1f} ns, {time.perf_counter() - t0:.0f} s",
            flush=True,
        )
    second_half = series[len(series) // 2 :]
    names = ["rmsd_all_nm", "rmsd_inner_nm", "bp_intact_all", "bp_intact_inner", "min_pp_nm"]
    x0 = ref
    summary = {
        "model": args.model, "force_field": "amber19/DNA.OL21.xml", "ns_per_replica": args.ns,
        "replicas": args.replicas, "r_cutoff_nm": info.r_cutoff, "kappa_per_nm": float(info.kappa[0]),
        "start": dict(zip(names, metrics(x0, ref, idx), strict=True)),
        "second_half_mean": {k: float(second_half[..., i].mean()) for i, k in enumerate(names)},
        "second_half_replica_means": {
            k: second_half[..., i].mean(axis=0).tolist() for i, k in enumerate(names)
        },
    }  # fmt: skip
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["second_half_mean"], indent=2))


if __name__ == "__main__":
    main()
