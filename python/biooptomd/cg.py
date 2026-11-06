"""Coarse-Grained-Modelle (ADR 0017, ROADMAP Phase 11).

Stufe 1: Proteine (Antikörper) als Cα-Netz mit Gō-Kontakten (:func:`enm_go_from_pdb`).

- Perle je Residuum am Cα; Masse = Residuenmasse (Summenformel der Residue im Peptid,
  :data:`biooptomd.builder.ELEMENT_MASS`), Ladung bei pH 7 (Lys/Arg +1, Asp/Glu −1, His 0) plus
  geladene Termini je Kette.
- Kette: harmonische Bindungen zwischen aufeinanderfolgenden Cα (r₀ aus der Struktur).
- Elastisches Netz (ENM, ElNeDyn-artig): harmonische Federn zwischen Cα-Paaren derselben Domäne mit
  Abstand < ``enm_cutoff_nm`` (nicht Kettennachbarn). Wie der MARTINI-Bindungstyp 6 erzeugen sie
  keine Ausschlüsse: Abstoßung und Debye-Hückel wirken zwischen ihnen weiter.
- Gō-Kontakte zwischen Residuen verschiedener Domänen (oder ohne Domäne, z. B. Gelenk) mit einem
  Schweratomabstand < ``go_cutoff_nm`` in der Struktur (|i − j| >= 3 in derselben Kette): Paarterm
  12-10 (``pair_form = 1``) mit σ = nativer Cα-Abstand, Tiefe ``eps_go``; Debye-Hückel des Paares
  im selben Term (das Paar ist im Nonbonded-Kernel ausgeschlossen).
- Unspezifisch: LJ mit ``sigma_bead``/``eps_bead`` (fast nur abstoßend) und Debye-Hückel über den
  Nonbonded-Kernel; ausgeschlossen sind Kettennachbarn 1-2 und 1-3 sowie die Gō-Paare.
- Optik: α′ und v je Perle = Summe der kalibrierten Atomwerte der Residue (Summenregel, Phase 10.0),
  ein Optik-Typ je Residuenname.

ASSUMPTION(ADR 0017): Parameter-Defaults nach ElNeDyn (Periole et al., JCTC 2009: Cutoff 0.9 nm,
k = 500 kJ/mol/nm²) bzw. Karanicolas & Brooks (Protein Sci. 2002: 12-10-Kontakte); k_bond, eps_go,
sigma_bead, eps_bead sind Annahmen und im Test bzw. an Referenzdaten zu prüfen. Einheiten: nm, amu,
e, kJ/mol.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import builder

RESIDUE_CHARGE = {"LYS": 1.0, "ARG": 1.0, "ASP": -1.0, "GLU": -1.0}
HIS_ALIASES = {"HID": "HIS", "HIE": "HIS", "HIP": "HIS", "HSD": "HIS", "HSE": "HIS", "HSP": "HIS"}


@dataclass
class Residue:
    chain: str
    resseq: int
    icode: str
    name: str
    ca: np.ndarray
    heavy: np.ndarray


@dataclass
class CGModel:
    """Ergebnis von :func:`enm_go_from_pdb`.

    Attributes:
        spec: :class:`biooptomd.builder.SystemSpec` der Perlen (für ``build_context``).
        residues: Residuen in Perlenreihenfolge.
        domain: Domänenindex je Perle (−1 = keiner Domäne zugeordnet).
        enm_pairs, go_pairs: Indexpaare (m, 2) der Federn bzw. Kontakte.
        types: Optik-/Oberflächen-Typ je Perle (>= 1, Typ 0 bleibt für Anker/Wandausnahmen frei).
        alpha, volume: α′ und v je Typ [nm³] (Index 0 = 0); names: Residuenname je Typ.
    """

    spec: builder.SystemSpec
    residues: list
    domain: np.ndarray
    enm_pairs: np.ndarray
    go_pairs: np.ndarray
    types: np.ndarray
    alpha: np.ndarray
    volume: np.ndarray
    names: np.ndarray


def read_pdb_residues(path, chains=None) -> list:
    """Aminosäure-Residuen (ATOM-Einträge, erstes Modell, altLoc ' ' oder 'A') mit Cα und allen
    Schweratomen [nm]. Residuen ohne Cα werden übersprungen."""
    out: dict = {}
    order = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            rec = line[:6]
            if rec.startswith("ENDMDL"):
                break
            if rec != "ATOM  ":
                continue
            alt = line[16]
            if alt not in (" ", "A"):
                continue
            chain = line[21]
            if chains is not None and chain not in chains:
                continue
            name = line[12:16].strip()
            element = line[76:78].strip() or name[0]
            if element == "H":
                continue
            res = (chain, int(line[22:26]), line[26])
            if res not in out:
                resname = HIS_ALIASES.get(line[17:20].strip(), line[17:20].strip())
                out[res] = {"name": resname, "ca": None, "heavy": []}
                order.append(res)
            xyz = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])]) * 0.1
            out[res]["heavy"].append(xyz)
            if name == "CA":
                out[res]["ca"] = xyz
    return [
        Residue(
            c, s, i, out[(c, s, i)]["name"], out[(c, s, i)]["ca"], np.array(out[(c, s, i)]["heavy"])
        )
        for c, s, i in order
        if out[(c, s, i)]["ca"] is not None
    ]


def _residue_composition(name: str) -> dict:
    if name not in builder._RESIDUES:
        raise ValueError(f"cg: unbekanntes Residuum {name!r} (nur die 20 Standardaminosäuren)")
    return builder.parse_formula(builder._RESIDUES[name][0])


def enm_go_from_pdb(path, *, chains=None, domains=None, enm_cutoff_nm: float = 0.9,
                    k_enm: float = 500.0, k_bond: float = 1.0e4, go_cutoff_nm: float = 0.45,
                    eps_go: float = 2.5, sigma_bead: float = 0.4, eps_bead: float = 0.05,
                    charged_termini: bool = True, optics=None) -> CGModel:  # fmt: skip
    """Cα-ENM/Gō-Modell eines Proteins (ADR 0017 Stufe 1).

    Args:
        path: PDB-Datei. chains: Ketten-IDs oder None (alle).
        domains: Liste von Domänen; jede Domäne ist eine Liste von (Kette, erstes, letztes Residuum)
            (inklusive). None ⇒ alle Residuen bilden eine Domäne. Residuen ohne Domäne (Gelenk) haben
            keine Federn, nur Kette und Gō-Kontakte.
        enm_cutoff_nm, k_enm: Federn zwischen Cα derselben Domäne [nm], [kJ/mol/nm²].
        k_bond: Kettenbindung [kJ/mol/nm²]. go_cutoff_nm: Schweratomabstand nativer Kontakte [nm].
        eps_go: Kontakttiefe [kJ/mol]. sigma_bead, eps_bead: unspezifisches LJ [nm], [kJ/mol].
        charged_termini: N-Terminus +1, C-Terminus −1 je Kette.
        optics: Element-Tabelle für α′/v; Default ``builder.calibrate_optics("protein")[0]``.
    """
    residues = read_pdb_residues(path, chains)
    n = len(residues)
    if n < 2:
        raise ValueError("cg: weniger als zwei Residuen mit Cα")
    pos = np.array([r.ca for r in residues])
    names = [r.name for r in residues]
    comps = [_residue_composition(m) for m in names]
    masses = np.array([sum(k * builder.ELEMENT_MASS[e] for e, k in c.items()) for c in comps])
    charges = np.array([RESIDUE_CHARGE.get(m, 0.0) for m in names])

    # Ketten und Kettenbindungen (aufeinanderfolgend, Cα-Abstand < 0.42 nm: keine Lücke)
    chain_of = np.array([r.chain for r in residues])
    bonds = []
    for i in range(n - 1):
        if chain_of[i] == chain_of[i + 1] and np.linalg.norm(pos[i + 1] - pos[i]) < 0.42:
            bonds.append((i, i + 1))
    if charged_termini:
        for c in dict.fromkeys(chain_of):
            idx = np.flatnonzero(chain_of == c)
            charges[idx[0]] += 1.0
            charges[idx[-1]] -= 1.0

    # Domänen
    domain = np.full(n, -1, np.int64)
    if domains is None:
        domain[:] = 0
    else:
        for d, spans in enumerate(domains):
            for chain, first, last in spans:
                sel = np.array([r.chain == chain and first <= r.resseq <= last for r in residues])
                if np.any(domain[sel] >= 0):
                    raise ValueError("cg: Residuum in mehreren Domänen")
                domain[sel] = d

    bonded_set = {tuple(b) for b in bonds}
    # 1-3 entlang der Kette
    excl13 = [(i, j + 1) for (i, j) in bonds if (j, j + 1) in bonded_set]
    d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    iu, ju = np.triu_indices(n, 1)
    same_dom = (domain[iu] == domain[ju]) & (domain[iu] >= 0)
    not_bonded = np.array([(a, b) not in bonded_set for a, b in zip(iu, ju, strict=True)])
    enm_mask = same_dom & (d[iu, ju] < enm_cutoff_nm) & not_bonded
    enm_pairs = np.stack([iu[enm_mask], ju[enm_mask]], 1)

    # Gō-Kontakte: nicht in derselben Domäne, Schweratome < go_cutoff, |i−j| >= 3 in einer Kette
    go = []
    for a, b in zip(iu[~same_dom], ju[~same_dom], strict=True):
        if chain_of[a] == chain_of[b] and abs(a - b) < 3:
            continue
        if d[a, b] > go_cutoff_nm + 2.0:  # grobe Vorauswahl über Cα (Seitenketten < 1 nm)
            continue
        ha, hb = residues[a].heavy, residues[b].heavy
        dmin = np.sqrt(((ha[:, None, :] - hb[None, :, :]) ** 2).sum(-1).min())
        if dmin < go_cutoff_nm:
            go.append((a, b))
    go_pairs = np.array(go, np.int64).reshape(-1, 2)

    excluded = np.array(sorted(bonded_set | set(excl13) | {tuple(p) for p in go_pairs.tolist()}),
                        np.int32).reshape(-1, 2)  # fmt: skip
    all_bonds = np.concatenate([np.array(bonds, np.int64).reshape(-1, 2), enm_pairs])
    bond_k = np.concatenate([np.full(len(bonds), k_bond), np.full(len(enm_pairs), k_enm)])
    bond_r0 = d[all_bonds[:, 0], all_bonds[:, 1]] if len(all_bonds) else np.zeros(0)
    bonded = {
        "bonds": all_bonds.astype(np.int32), "bond_k": bond_k, "bond_r0": bond_r0,
        "angles": np.zeros((0, 3), np.int32), "angle_k": np.zeros(0), "angle_theta0": np.zeros(0),
        "torsions": np.zeros((0, 4), np.int32), "torsion_k": np.zeros(0),
        "torsion_n": np.zeros(0, np.int32), "torsion_phase": np.zeros(0),
        "pairs14": go_pairs.astype(np.int32),
        "pair_qq": charges[go_pairs[:, 0]] * charges[go_pairs[:, 1]] if len(go_pairs) else np.zeros(0),
        "pair_sigma": d[go_pairs[:, 0], go_pairs[:, 1]] if len(go_pairs) else np.zeros(0),
        "pair_eps": np.full(len(go_pairs), eps_go),
        "pair_form": np.ones(len(go_pairs), np.int32),
    }  # fmt: skip
    spec = builder.SystemSpec(
        masses=masses, charges=charges, sigma=np.full(n, sigma_bead), epsilon=np.full(n, eps_bead),
        bonded=bonded, excluded=excluded, elements=np.full(n, "X"), positions=pos,
        atom_names=np.full(n, "CA"), residue_names=np.array(names),
    )  # fmt: skip

    table = builder.calibrate_optics("protein")[0] if optics is None else optics
    type_names = list(dict.fromkeys(names))
    types = np.array([type_names.index(m) + 1 for m in names], np.int32)
    alpha = np.zeros(len(type_names) + 1)
    volume = np.zeros(len(type_names) + 1)
    for t, m in enumerate(type_names, start=1):
        comp = _residue_composition(m)
        alpha[t] = sum(k * table[e][0] for e, k in comp.items())
        volume[t] = sum(k * table[e][1] for e, k in comp.items())
    return CGModel(spec, residues, domain, enm_pairs, go_pairs, types, alpha.astype(np.float32),
                   volume.astype(np.float32), np.array([""] + type_names))  # fmt: skip
