"""OQ-15-Validierung (b): Faltung des DNA-Mini-Hairpins d(GCGAAGC) (PDB 1PQT) mit GBn2 + OL21 und
Replica Exchange.

System: 1PQT, NMR-Modell 1 (Padrta et al., J. Biomol. NMR 24, 1 (2002); 7 nt, GAA-Triloop, Stamm
G1:C7, C2:G6), AMBER OL21 (`amber19/DNA.OL21.xml`), `implicit/gbn2.xml`, Salz 0.15 M (κ(T), ε_r(T),
Cutoff aus κ), HMR + H-Constraints, Δt = 4 fs, Langevin γ = 1/ps, Box 12 × 12 nm (größer als
Ausdehnung + r_c auch für den gestreckten Strang).

REMD: 4 unabhängige Leitern (``ReplicaExchange(ladders=4)``) zu 8 Temperaturen, geometrisch
300–370 K (ε_r(T) von Wasser gilt bis 373 K), Austauschversuch jede ps. Leitern 0, 1 starten aus der
NMR-Struktur, Leitern 2, 3 aus zwei verschiedenen gestreckten, ungepaarten Strukturen.

Gestreckte Startstrukturen: OpenMM (gleiches Kraftfeld und GB-Modell, NoCutoff), Flat-Bottom-Restraint
k = 1000 kJ/(mol·nm²) auf den Abstand C1'(G1)–C1'(C7) ≥ 3.0 nm, 1 ns bei 300 K, dann 1 ns bei 370 K
mit verschiedenen Seeds; geprüft wird: kein Stammpaar intakt.

Messgrößen alle 100 ps je Slot, dem Zustand des Slots zugeordnet:
- RMSD [nm] der Schweratome zu NMR-Modell 1 (Kabsch), alle 7 nt.
- Stammpaare G1:C7 und C2:G6 intakt (N1(Purin)–N3(Pyrimidin) < 0.35 nm).
- gefaltet ⇔ beide Stammpaare intakt und RMSD < 0.25 nm. ASSUMPTION: Schwelle; NMR-Ensemble
  streut ≤ 0.047 nm um Modell 1, nativ gestartete Läufe 0.05–0.15 nm (300–370 K), gestreckte
  Starts 0.67 und 0.86 nm.

Aufruf:  python examples/gb_validation/hairpin_folding.py --ns 1000 --out run_hairpin
Ausgabe: <out>/series.npz (rmsd, stem je Frame und Slot, state je Frame und Slot),
<out>/summary.json, <out>/checkpoint.npz (Fortsetzung), <out>/starts.npz (Startstrukturen).
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path

import numpy as np

from biooptomd import builder
from biooptomd import io as bio
from biooptomd.remd import ReplicaExchange

PDB_URL = "https://files.rcsb.org/download/1PQT.pdb"
BOX = (12.0, 12.0)
DT, FRICTION, IONIC = 0.004, 1.0, 0.15
LADDER = np.geomspace(300.0, 370.0, 8)
N_LADDERS = 4
EXCHANGE_STEPS = 250  # 1 ps
FRAME_CYCLES = 100  # 100 ps
CHUNK_FRAMES = 100  # Checkpoint alle 10 ns
HBOND_CUT = 0.35  # nm
FOLDED_RMSD = 0.25  # nm
STEM = ((0, 6), (1, 5))  # G1:C7, C2:G6 (Reste 0-basiert)


def load_structure(cache: Path):
    """OpenMM-Topologie, System, NMR-Positionen [nm] (Modell 1)."""
    from openmm import app, unit

    if not cache.exists():
        cache.write_text(urllib.request.urlopen(PDB_URL, timeout=60).read().decode())
    pdb = app.PDBFile(str(cache))  # liest Modell 1
    ff = app.ForceField("amber19/DNA.OL21.xml", "implicit/gbn2.xml")
    system = ff.createSystem(pdb.topology, nonbondedMethod=app.NoCutoff, constraints=None)
    pos = np.array(pdb.getPositions().value_in_unit(unit.nanometer))
    return pdb.topology, system, pos


def indices(topology):
    res = list(topology.residues())
    heavy = np.flatnonzero([a.element.symbol != "H" for a in topology.atoms()])
    name = [{a.name: a.index for a in r.atoms()} for r in res]

    def wc(k):  # Watson-Crick-Stickstoff: N1 des Purins, N3 des Pyrimidins
        return name[k]["N1" if res[k].name in ("DA", "DG") else "N3"]

    stem = np.array([(wc(i), wc(j)) for i, j in STEM])
    c1 = (name[0]["C1'"], name[6]["C1'"])
    return {"heavy": heavy, "stem": stem, "c1": c1}


def kabsch_rmsd(x, ref):
    xc, rc = x - x.mean(0), ref - ref.mean(0)
    u, _, vt = np.linalg.svd(xc.T @ rc)
    d = np.sign(np.linalg.det(u @ vt))
    rot = u @ np.diag([1.0, 1.0, d]) @ vt
    return float(np.sqrt(np.mean(np.sum((xc @ rot - rc) ** 2, axis=1))))


def metrics(x, ref, idx):
    """(RMSD [nm], Stammpaare intakt (2,)) für eine Struktur."""
    s = idx["stem"]
    intact = np.linalg.norm(x[s[:, 0]] - x[s[:, 1]], axis=1) < HBOND_CUT
    return kabsch_rmsd(x[idx["heavy"]], ref[idx["heavy"]]), intact


def unfolded_starts(topology, pos, idx, seeds=(11, 12)):
    """Zwei gestreckte, ungepaarte Strukturen (OpenMM, Restraint auf C1'(G1)–C1'(C7) ≥ 3 nm)."""
    import openmm
    from openmm import app, unit

    out = []
    for seed in seeds:
        system = app.ForceField("amber19/DNA.OL21.xml", "implicit/gbn2.xml").createSystem(
            topology, nonbondedMethod=app.NoCutoff, constraints=app.HBonds
        )  # H-Constraints für 2 fs
        bond = openmm.CustomBondForce("0.5*k*step(r0-r)*(r0-r)^2")
        bond.addGlobalParameter("k", 1000.0)
        bond.addGlobalParameter("r0", 3.0)
        bond.addBond(*idx["c1"], [])
        system.addForce(bond)
        integ = openmm.LangevinMiddleIntegrator(300 * unit.kelvin, 1 / unit.picosecond, 0.002)
        integ.setRandomNumberSeed(seed)
        sim = app.Simulation(topology, system, integ)
        sim.context.setPositions(pos * unit.nanometer)
        sim.minimizeEnergy()
        sim.context.setVelocitiesToTemperature(300 * unit.kelvin, seed)
        sim.step(500_000)
        integ.setTemperature(370 * unit.kelvin)
        sim.step(500_000)
        integ.setTemperature(300 * unit.kelvin)
        sim.step(50_000)
        x = sim.context.getState(getPositions=True).getPositions(asNumpy=True)
        x = np.asarray(x.value_in_unit(unit.nanometer))
        _, intact = metrics(x, pos, idx)
        if intact.any():
            raise RuntimeError(f"Startstruktur (Seed {seed}) noch gepaart")
        out.append(x)
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--ns", type=float, default=1000.0, help="Simulationszeit je Replica [ns]")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    topology, system, ref = load_structure(args.out / "1PQT.pdb")
    idx = indices(topology)
    starts_path = args.out / "starts.npz"
    if starts_path.exists():
        unfolded = list(np.load(starts_path)["unfolded"])
    else:
        unfolded = unfolded_starts(topology, ref, idx)
        np.savez(starts_path, unfolded=np.array(unfolded), native=ref)
    m = len(LADDER)
    centre = np.array([BOX[0] / 2, BOX[1] / 2, 5.0])
    blocks = [ref, ref, unfolded[0], unfolded[1]]
    positions = np.array([b - b.mean(0) + centre for b in blocks for _ in range(m)])
    spec = builder.from_openmm(system, topology, positions[0])
    temps_all = np.tile(LADDER, N_LADDERS)
    ctx, info = builder.build_context(
        spec, len(temps_all), BOX, temps_all, IONIC, positions=positions, solvent="gb-gbn2",
        hmr=True, constraints=True,
    )  # fmt: skip
    ctx.set_integrator(DT, FRICTION, args.seed)
    remd = ReplicaExchange(ctx, LADDER, ionic_strength_molar=IONIC, seed=args.seed,
                           ladders=N_LADDERS)  # fmt: skip
    n_frames = round(args.ns * 1000 / (FRAME_CYCLES * EXCHANGE_STEPS * DT))
    series_path, ckpt = args.out / "series.npz", args.out / "checkpoint.npz"
    n_slots = len(temps_all)
    rmsd = np.zeros((0, n_slots))
    stem = np.zeros((0, n_slots, len(STEM)), bool)
    state = np.zeros((0, n_slots), np.int64)
    if ckpt.exists() and series_path.exists():
        bio.load_checkpoint(ctx, ckpt, remd=remd)
        d = np.load(series_path)
        rmsd, stem, state = d["rmsd"], d["stem"], d["state"]
    t0 = time.perf_counter()
    while len(rmsd) < n_frames:
        b_rmsd, b_stem, b_state = [], [], []
        for _ in range(min(CHUNK_FRAMES, n_frames - len(rmsd))):
            remd.run(FRAME_CYCLES, EXCHANGE_STEPS)
            x = ctx.positions_precise()[:, : spec.n_atoms, :3]  # Kabsch: Wrapping nur in x/y,
            img = ctx.images()[:, : spec.n_atoms].astype(np.float64)  # daher ungewrappt
            box = ctx.box().astype(np.float64)
            x = x.copy()
            x[..., 0] += img[..., 0] * box[:, None, 0]
            x[..., 1] += img[..., 1] * box[:, None, 1]
            frame = [metrics(x[r], ref, idx) for r in range(n_slots)]
            b_rmsd.append([f[0] for f in frame])
            b_stem.append([f[1] for f in frame])
            b_state.append(remd.state.copy())
        if ctx.status() != 0:
            raise RuntimeError(f"Statuswort {ctx.status()}")
        rmsd = np.concatenate([rmsd, b_rmsd])
        stem = np.concatenate([stem, b_stem])
        state = np.concatenate([state, b_state])
        np.savez(series_path, rmsd=rmsd, stem=stem, state=state, ladder=LADDER,
                 time_ns=np.arange(1, len(rmsd) + 1) * 0.1)  # fmt: skip
        bio.save_checkpoint(ctx, ckpt, remd=remd)
        acc = remd.acceptance_ratio()
        print(
            f"{len(rmsd) * 0.1:.1f} ns, {time.perf_counter() - t0:.0f} s, Akzeptanz "
            f"{np.nanmin(acc):.2f}–{np.nanmax(acc):.2f}",
            flush=True,
        )
    summary = analyse(rmsd, stem, state, m)
    summary.update(r_cutoff_nm=info.r_cutoff, ladder_k=LADDER.tolist(),
                   acceptance=np.nan_to_num(remd.acceptance_ratio(), nan=-1).tolist())  # fmt: skip
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("folded_300K_by_ladder", "first_fold_ns")}, indent=2))


def analyse(rmsd, stem, state, m):
    """Gefaltet-Anteil je Leiter im 300-K-Zustand (zweite Hälfte), erste Faltung der Leitern aus
    gestreckten Strukturen (Zeit, zu der irgendein Slot der Leiter gefaltet ist)."""
    folded = stem.all(axis=-1) & (rmsd < FOLDED_RMSD)
    n_frames, n_slots = rmsd.shape
    ladders = n_slots // m
    half = n_frames // 2
    out = {"frames": int(n_frames), "folded_300K_by_ladder": [], "rmsd_300K_median_by_ladder": [],
           "folded_by_state_second_half": [], "first_fold_ns": []}  # fmt: skip
    for b in range(ladders):
        cold = state == b * m  # Slot im 300-K-Zustand der Leiter b
        f = folded[half:][cold[half:]]
        out["folded_300K_by_ladder"].append(float(f.mean()))
        out["rmsd_300K_median_by_ladder"].append(float(np.median(rmsd[half:][cold[half:]])))
        per_state = [float(folded[half:][state[half:] == b * m + k].mean()) for k in range(m)]
        out["folded_by_state_second_half"].append(per_state)
        any_fold = folded[:, b * m : (b + 1) * m].any(axis=1)
        out["first_fold_ns"].append(
            float((np.argmax(any_fold) + 1) * 0.1) if any_fold.any() else None
        )
    return out


if __name__ == "__main__":
    main()
