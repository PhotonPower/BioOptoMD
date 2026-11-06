"""CG-Antikörper (IgG, PDB 1IGT) über Gold mit Replica Exchange (ROADMAP Phase 11.4, ADR 0017).

Ablauf: 1IGT (Maus-IgG2a, 4 Ketten, 1316 Residuen; Glykane und Wasser weggelassen) →
``biooptomd.cg.enm_go_from_pdb`` mit sieben Domänen (je Arm Fv und CL/CH1, die beiden CH2, das
CH3-Paar; das Gelenk B/D 223–245 ohne Domäne, nur Kette, Winkel, Disulfide und Gō-Kontakte) →
Gold-Wand (9-3, ADR 0007) mit λ je Residuum (``cg.solvent_scale``) + Bildladungen (ADR 0013) +
Decke (halbharmonisch über ``add_voxel_grid``) → unabhängige REMD-Leitern zu 8 Temperaturen
(300–370 K, ε_r(T) und κ(T) je Zustand) → Adsorption und Orientierung je Zustand.

Platzhalter (siehe docs/OPEN_QUESTIONS.md):
- ASSUMPTION(OQ-29): λ = 1 für alle Residuen (unkalibriert) ⇒ Adsorptionsstärke nur qualitativ.
  Die Wand ist je Perle ≈ 5 kJ/mol tief; eine aufliegende Domäne mit einigen zehn Kontakten bindet
  mit Hunderten kJ/mol, die Adsorption ist dann auch bei 370 K praktisch irreversibel. REMD wandelt
  die Orientierungsklassen so nicht ineinander um: Die Anteile bei 300 K spiegeln Startorientierung
  und Kinetik des ersten Kontakts, kein Gleichgewicht. ``summary.json`` zählt deshalb die
  Desorptionsereignisse je Leiter (Ergodizitätsindikator); erst wenn es viele gibt, sind die
  Klassenanteile als Gleichgewicht lesbar.
- ASSUMPTION(OQ-30): CG-Parameter (ENM 0.9 nm / 500 kJ/mol/nm², Gō 2.5 kJ/mol, Perlen-LJ).
- ASSUMPTION(OQ-18): Bildebene in der obersten Goldlage, f = 1.
- Glykane (CH2) fehlen; die Domänengrenzen sind aus der Nummerierung von 1IGT gewählt.
- ASSUMPTION(OQ-31): Decke bei z = 30 nm (k = 100 kJ/mol/nm²): legt das Volumen des gelösten
  Zustands fest (Box 30 × 30 × 30 nm ≈ 60 µM für ein IgG); die Adsorptionsgleichgewichte hängen
  davon ab.
- ASSUMPTION(OQ-31): Langevin γ = 1/ps ohne Bezug zur Wasserreibung: Zeiten sind keine
  physikalischen Kinetiken.
- Kontakt: Perle mit ζ − z_top < 0.6 nm; eine Gruppe (Fab 1, Fab 2, Fc) liegt auf, wenn
  mindestens 3 ihrer Perlen Kontakt haben. Orientierung: desorbiert, „Fc unten“ (nur Fc), „Fab
  unten“ (nur Fab), „seitlich“ (ein Fab + Fc), „flach“ (beide Fab + Fc). Kontakte nur über
  Gelenkperlen oder mit < 3 Perlen je Gruppe zählen als desorbiert.

Eingabe: tests/reference/inputs/1igt_protein.pdb = RCSB 1IGT, nur die Zeilen HEADER, TITLE, ATOM,
TER, END (``grep -E "^(HEADER|TITLE|ATOM|TER|END)" 1IGT.pdb``; Glykane/HETATM und SSBOND entfernt).
Fehlt die Datei, wird sie aus RCSB geladen und ebenso gefiltert.

Messgrößen alle 100 ps je Slot: Schwerpunkthöhe über der obersten Goldlage, Kontakte je Gruppe,
cos(Achse, z) je Gruppe (Achse: Gelenk-Schwerpunkt → Gruppenschwerpunkt), Zustand des Slots.

Aufruf:  python examples/igg_on_gold/igg_on_gold.py --ns 2000 --out run_igg
Ausgabe: <out>/series.npz, <out>/summary.json, <out>/checkpoint.npz (Fortsetzung).
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.request
from pathlib import Path

import numpy as np

from biooptomd import builder, cg, surface
from biooptomd import io as bio
from biooptomd.remd import ReplicaExchange

ROOT = Path(__file__).resolve().parents[2]
PDB = ROOT / "tests" / "reference" / "inputs" / "1igt_protein.pdb"
PDB_URL = "https://files.rcsb.org/download/1IGT.pdb"
DOMAINS = [
    [("A", 1, 107), ("B", 1, 113)],  # Fv, Arm 1
    [("A", 108, 214), ("B", 114, 222)],  # CL/CH1, Arm 1
    [("C", 1, 107), ("D", 1, 113)],  # Fv, Arm 2
    [("C", 108, 214), ("D", 114, 222)],  # CL/CH1, Arm 2
    [("B", 246, 360)],  # CH2
    [("D", 246, 360)],  # CH2
    [("B", 361, 474), ("D", 361, 474)],  # CH3-Paar
]
GROUPS = {"fab1": (0, 1), "fab2": (2, 3), "fc": (4, 5, 6)}
BOX = (30.0, 30.0)
CEILING_NM, K_CEILING = 30.0, 100.0  # ASSUMPTION(OQ-31): Decke, s. o.
DT, FRICTION, IONIC = 0.02, 1.0, 0.15  # FRICTION: ASSUMPTION(OQ-31)
LADDER = np.geomspace(300.0, 370.0, 8)
EXCHANGE_STEPS = 50  # 1 ps
FRAME_CYCLES = 100  # 100 ps
CHUNK_FRAMES = 100  # Checkpoint alle 10 ns
START_GAP_NM = 2.0  # unterste Perle über der obersten Goldlage beim Start
CONTACT_NM, GROUP_MIN_CONTACTS = 0.6, 3
CLASSES = ("desorbiert", "Fc unten", "Fab unten", "seitlich", "flach")


def atomic_savez(path: Path, **arrays) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.savez(f, **arrays)
    os.replace(tmp, path)


def atomic_checkpoint(ctx, path: Path, **kw) -> None:
    tmp = path.with_name(path.name + ".tmp")
    bio.save_checkpoint(ctx, tmp, **kw)
    os.replace(tmp, path)


def load_model(pdb: Path = PDB):
    if not pdb.exists():
        text = urllib.request.urlopen(PDB_URL, timeout=60).read().decode()
        keep = ("HEADER", "TITLE", "ATOM", "TER", "END")
        pdb.write_text("".join(line for line in text.splitlines(keepends=True)
                               if line.startswith(keep)))  # fmt: skip
    return cg.enm_go_from_pdb(pdb, domains=DOMAINS)


def ceiling_grid(box, height=CEILING_NM, k=K_CEILING, width=5.0, dz=0.1):
    """Halbharmonische Decke V = ½ k (z − h)² für z > h als Gitter (nz, 2, 2, 4), periodisch in
    x/y; oberhalb h + width wirkt nichts (bei k = 100 kJ/mol/nm² unerreichbar). Das Plugin
    interpoliert V und ∇V linear zwischen den Knoten: die Kraft ist exakt, die Energie weicht
    um höchstens k Δz²/8 je Perle oberhalb h von der Parabel ab (0.125 kJ/mol bei Δz = 0.1 nm)."""
    z = height + np.arange(round(width / dz) + 1) * dz
    vals = np.zeros((len(z), 2, 2, 4), np.float32)
    vals[..., 0] = (0.5 * k * (z - height) ** 2)[:, None, None]
    vals[..., 3] = (k * (z - height))[:, None, None]
    return vals, (0.0, 0.0, height), (box[0] / 2, box[1] / 2, dz)


def build(n_ladders: int = 4, seed: int = 2026, lam=None, model=None, ceiling: bool = True):
    """Kontext, REMD und Metadaten. Jeder Slot startet in einer eigenen zufälligen Orientierung,
    die unterste Perle START_GAP_NM über der obersten Goldlage. ``ceiling=False`` nur für Tests."""
    m = load_model() if model is None else model
    spec = m.spec
    z_top = surface.MATERIALS["gold"].z_top_nm
    rng = np.random.default_rng(seed)
    x0 = spec.positions - spec.positions.mean(axis=0)
    n_slots = n_ladders * len(LADDER)
    pos = []
    for _ in range(n_slots):
        q, r = np.linalg.qr(rng.normal(size=(3, 3)))
        q *= np.sign(np.diag(r))  # gleichverteilt auf O(3) ...
        q[:, 0] *= np.sign(np.linalg.det(q))  # ... und als echte Drehung (keine Spiegelung)
        x = x0 @ q
        pos.append(x + [BOX[0] / 2, BOX[1] / 2, z_top + START_GAP_NM - x[:, 2].min()])
    temps = np.tile(LADDER, n_ladders)
    ctx, info = builder.build_context(spec, n_slots, BOX, temps, IONIC, positions=np.array(pos),
                                      types=m.types)  # fmt: skip
    t = surface.wall_table("gold", info.lj_table, cg.solvent_scale(m, lam))
    ctx.add_planar_surface(t["prefactor"], t["sigma"], form=t["form"], delta=t["delta"], z0=0.0)
    builder.add_image_charges(ctx, info, "gold")
    if ceiling:
        vals, origin, spacing = ceiling_grid(BOX)
        scale = np.ones(len(m.names), np.float32)
        scale[0] = 0.0  # Typ 0 (Padding/Anker) wie in wall_table ohne äußeres Potential
        ctx.add_voxel_grid(vals, origin, spacing, scale, periodic_xy=True)
    ctx.set_integrator(DT, FRICTION, seed)
    remd = ReplicaExchange(ctx, LADDER, ionic_strength_molar=IONIC, seed=seed, ladders=n_ladders)
    groups = {g: np.flatnonzero(np.isin(m.domain, d)) for g, d in GROUPS.items()}
    meta = {"model": m, "info": info, "z_top": z_top, "groups": groups,
            "hinge": np.flatnonzero(m.domain < 0)}  # fmt: skip
    return ctx, remd, meta


def unwrapped(ctx, n):
    """Positionen [nm] (R, n, 3), in x/y mit den Bildzählern entfaltet."""
    x = ctx.positions_precise()[:, :n, :3].astype(np.float64)
    img = ctx.images()[:, :n].astype(np.float64)
    box = ctx.box().astype(np.float64)
    x[..., 0] += img[..., 0] * box[:, None, 0]
    x[..., 1] += img[..., 1] * box[:, None, 1]
    return x


def metrics(x, meta):
    """Je Slot: Schwerpunkthöhe über z_top, Kontakte je Gruppe (3,), cos(Achse, z) je Gruppe (3,).
    Schwerpunkte massengewichtet; das Molekül ist nach dem Entfalten zusammenhängend."""
    masses = meta["model"].spec.masses
    zeta = x[..., 2] - meta["z_top"]
    contact = zeta < CONTACT_NM

    def com(idx):
        w = masses[idx]
        return (x[:, idx] * w[None, :, None]).sum(axis=1) / w.sum()

    hinge = com(meta["hinge"])
    height = com(np.arange(len(masses)))[:, 2] - meta["z_top"]
    contacts, cosines = [], []
    for idx in meta["groups"].values():
        contacts.append(contact[:, idx].sum(axis=1))
        axis = com(idx) - hinge
        cosines.append(axis[:, 2] / np.linalg.norm(axis, axis=1))
    return height, np.stack(contacts, axis=1), np.stack(cosines, axis=1)


def classify(contacts):
    """Orientierungsklasse je Frame/Slot (Index in CLASSES) aus den Kontakten (…, 3)."""
    on = contacts >= GROUP_MIN_CONTACTS
    n_fab = on[..., 0].astype(int) + on[..., 1]
    fc = on[..., 2]
    cls = np.zeros(on.shape[:-1], np.int64)  # desorbiert
    cls[fc & (n_fab == 0)] = 1
    cls[~fc & (n_fab > 0)] = 2
    cls[fc & (n_fab == 1)] = 3
    cls[fc & (n_fab == 2)] = 4
    return cls


def frame_ns() -> float:
    return FRAME_CYCLES * EXCHANGE_STEPS * DT / 1000.0


def analyse(height, contacts, cosines, state, m):
    """Je Leiter im 300-K-Zustand (zweite Hälfte): adsorbierter Anteil, Klassenanteile, mittlere
    Kontaktzahl und Höhe, ⟨cos⟩ je Gruppe (alle Frames und nur adsorbierte), erste Adsorption
    (irgendein Slot der Leiter in einer Klasse > 0) und Desorptionsereignisse (Wechsel adsorbiert →
    desorbiert je Slot, d. h. je Walker, über den ganzen Lauf; Ergodizitätsindikator, s. o.)."""
    n_frames, n_slots = height.shape
    half = n_frames // 2
    cls = classify(contacts)
    ads = cls > 0
    out = {"frames": int(n_frames), "frame_ns": frame_ns(), "classes": list(CLASSES),
           "by_ladder": []}  # fmt: skip
    for b in range(n_slots // m):
        sl = slice(b * m, (b + 1) * m)
        cold = state[half:] == b * m
        c = cls[half:][cold]
        cos_cold = cosines[half:][cold]
        any_ads = ads[:, sl].any(axis=1)
        desorb = (ads[:-1, sl] & ~ads[1:, sl]).sum(axis=0)
        out["by_ladder"].append({
            "adsorbed_300K": float((c > 0).mean()),
            "class_fractions_300K": [float((c == k).mean()) for k in range(len(CLASSES))],
            "contacts_300K_mean": float(contacts[half:][cold].sum(axis=-1).mean()),
            "height_300K_mean_nm": float(height[half:][cold].mean()),
            "cos_axis_300K_mean": cos_cold.mean(axis=0).tolist(),
            "cos_axis_300K_mean_adsorbed": (cos_cold[c > 0].mean(axis=0).tolist()
                                            if (c > 0).any() else None),
            "first_adsorption_ns": (float((np.argmax(any_ads) + 1) * frame_ns())
                                    if any_ads.any() else None),
            "desorption_events_by_walker": desorb.tolist(),
        })  # fmt: skip
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--ns", type=float, default=2000.0, help="Simulationszeit je Replica [ns]")
    p.add_argument("--ladders", type=int, default=4)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    ctx, remd, meta = build(args.ladders, args.seed)
    n = meta["model"].spec.n_atoms
    m = len(LADDER)
    n_slots = ctx.n_replicas
    steps_per_frame = FRAME_CYCLES * EXCHANGE_STEPS
    n_frames = round(args.ns * 1000 / (steps_per_frame * DT))
    series_path, ckpt = args.out / "series.npz", args.out / "checkpoint.npz"
    height = np.zeros((0, n_slots))
    contacts = np.zeros((0, n_slots, 3), np.int64)
    cosines = np.zeros((0, n_slots, 3))
    state = np.zeros((0, n_slots), np.int64)
    if ckpt.exists():
        with np.load(series_path) as d:  # Datei schließen, sonst scheitert os.replace (Windows)
            series = {key: d[key] for key in ("height", "contacts", "cosines", "state")}
        step = bio.load_checkpoint(ctx, ckpt, remd=remd)
        k = step // steps_per_frame
        if step % steps_per_frame or k > len(series["height"]):
            raise RuntimeError(f"Checkpoint (Schritt {step}) passt nicht zur Zeitreihe")
        height, contacts, cosines, state = (series[key][:k] for key in ("height", "contacts",
                                                                         "cosines", "state"))  # fmt: skip
    t0, f0 = time.perf_counter(), len(height)
    while len(height) < n_frames:
        blk = {"h": [], "c": [], "a": [], "s": []}
        for _ in range(min(CHUNK_FRAMES, n_frames - len(height))):
            remd.run(FRAME_CYCLES, EXCHANGE_STEPS)
            h, c, a = metrics(unwrapped(ctx, n), meta)
            blk["h"].append(h)
            blk["c"].append(c)
            blk["a"].append(a)
            blk["s"].append(remd.state.copy())
        if ctx.status() != 0:
            raise RuntimeError(f"Statuswort {ctx.status()}")
        height = np.concatenate([height, blk["h"]])
        contacts = np.concatenate([contacts, blk["c"]])
        cosines = np.concatenate([cosines, blk["a"]])
        state = np.concatenate([state, blk["s"]])
        atomic_savez(series_path, height=height, contacts=contacts, cosines=cosines, state=state,
                     ladder=LADDER, time_ns=np.arange(1, len(height) + 1) * frame_ns())  # fmt: skip
        atomic_checkpoint(ctx, ckpt, remd=remd)
        el = time.perf_counter() - t0
        rate = (len(height) - f0) * steps_per_frame * DT / 1000 / el * 86400  # ns/Tag je Replica
        cls = classify(contacts[-1])
        acc = remd.acceptance_ratio()
        print(f"{len(height) * frame_ns():.1f} ns, {el:.0f} s, {rate:.0f} ns/Tag je Replica, Akzeptanz "
              f"{np.nanmin(acc):.2f}–{np.nanmax(acc):.2f}, adsorbiert {(cls > 0).sum()}/{n_slots}",
              flush=True)  # fmt: skip
    summary = analyse(height, contacts, cosines, state, m)
    summary.update(n_beads=int(n), n_replicas=int(n_slots), dt_ps=DT, ladder_k=LADDER.tolist(),
                   r_cutoff_nm=meta["info"].r_cutoff,
                   acceptance=np.nan_to_num(remd.acceptance_ratio(), nan=-1).tolist())  # fmt: skip
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["by_ladder"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
