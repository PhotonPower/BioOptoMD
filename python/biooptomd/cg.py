"""Coarse-Grained-Modelle (ADR 0017, ROADMAP Phase 11).

Stufe 1: Proteine (Antikörper) als Cα-Netz mit Gō-Kontakten (:func:`enm_go_from_pdb`).

- Perle je Residuum am Cα; Masse = Residuenmasse (Summenformel der Residue im Peptid,
  :data:`biooptomd.builder.ELEMENT_MASS`, H₂O am letzten Residuum jeder Kette), Ladung bei pH 7
  (Lys/Arg +1, Asp/Glu −1, His 0; explizite Protonierungsnamen wie HIP, ASH, GLH, LYN mit ihrer
  Ladung) plus geladene Termini je Kette (am ersten bzw. letzten aufgelösten Residuum).
- Kette: harmonische Bindungen zwischen aufeinanderfolgenden Cα (r₀ aus der Struktur), Lücken
  (Cα–Cα > 0.42 nm) trennen die Kette mit Warnung.
- Biegung: kosinus-harmonische Cα-Winkel (``angle_form = 1``, θ0 aus der Struktur) für alle
  Kettentripel, damit auch das Gelenk (ohne Federn) nicht frei gelenkig ist.
- Disulfidbrücken (SG–SG < 0.25 nm zwischen Cys): harmonische Bindung zwischen den Cα (r₀ nativ,
  ``k_bond``), ausgeschlossen wie Kettenbindungen.
- Elastisches Netz (ENM, ElNeDyn-artig): harmonische Federn zwischen Cα-Paaren derselben Domäne mit
  Abstand < ``enm_cutoff_nm`` (nicht Kettennachbarn). Wie der MARTINI-Bindungstyp 6 erzeugen sie
  keine Ausschlüsse: Abstoßung und Debye-Hückel wirken zwischen ihnen weiter.
- Gō-Kontakte zwischen Residuen verschiedener Domänen (oder ohne Domäne, z. B. Gelenk) mit einem
  Schweratomabstand < ``go_cutoff_nm`` in der Struktur (|i − j| >= 3 in derselben Kette): Paarterm
  mit Minimum −``eps_go`` beim nativen Cα-Abstand d, Form ``go_form`` "12-10" (``pair_form = 1``,
  σ = d) oder "12-6" (σ = d/2^(1/6)); Debye-Hückel des Paares im selben Term (das Paar ist im
  Nonbonded-Kernel ausgeschlossen).
- Unspezifisch: LJ mit ``sigma_bead``/``eps_bead`` (fast nur abstoßend) und Debye-Hückel über den
  Nonbonded-Kernel; ausgeschlossen sind Kettennachbarn 1-2 und 1-3, Disulfide und die Gō-Paare.
- Optik: α′ und v je Perle = Summe der kalibrierten Atomwerte der Residue (Summenregel, Phase 10.0),
  ein Optik-Typ je Residuenname (bzw. „…+H2O“ für C-terminale Residuen).

ASSUMPTION(OQ-30): Defaults nach ElNeDyn (Periole et al., JCTC 2009: Cutoff 0.9 nm,
k = 500 kJ/mol/nm²) und Karanicolas & Brooks (Protein Sci. 2002: 12-10-Kontakte); k_bond, k_angle,
eps_go, sigma_bead, eps_bead und His neutral sind Annahmen. Einheiten: nm, amu, e, kJ/mol.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass

import numpy as np

from . import builder

RESIDUE_CHARGE = {"LYS": 1.0, "ARG": 1.0, "ASP": -1.0, "GLU": -1.0}
# Alias → (Standardresiduum für Formel/Optik, Ladung oder None = Standardladung)
RESIDUE_ALIASES = {
    "HID": ("HIS", None), "HIE": ("HIS", None), "HSD": ("HIS", None), "HSE": ("HIS", None),
    "HIP": ("HIS", 1.0), "HSP": ("HIS", 1.0), "CYX": ("CYS", None), "ASH": ("ASP", 0.0),
    "GLH": ("GLU", 0.0), "LYN": ("LYS", 0.0), "MSE": ("MET", None),
}  # fmt: skip
HETATM_RESIDUES = {"MSE"}  # modifizierte Residuen, die oft als HETATM gespeichert sind
BOND_GAP_NM = 0.42  # trans 0.38 nm, cis-Pro ≈ 0.29 nm, über ein fehlendes Residuum >= 0.5 nm
DISULFIDE_NM = 0.25


@dataclass
class Residue:
    chain: str
    resseq: int
    icode: str
    name: str
    charge: float
    ca: np.ndarray
    heavy: np.ndarray
    sg: np.ndarray | None


@dataclass
class CGModel:
    """Ergebnis von :func:`enm_go_from_pdb`.

    Attributes:
        spec: :class:`biooptomd.builder.SystemSpec` der Perlen (für ``build_context``).
        residues: Residuen in Perlenreihenfolge.
        domain: Domänenindex je Perle (−1 = keiner Domäne zugeordnet).
        enm_pairs, go_pairs, disulfides: Indexpaare (m, 2).
        types: Optik-/Oberflächen-Typ je Perle (>= 1, Typ 0 bleibt für Anker/Wandausnahmen frei).
        alpha, volume: α′ und v je Typ [nm³] (Index 0 = 0); names: Name je Typ.
    """

    spec: builder.SystemSpec
    residues: list
    domain: np.ndarray
    enm_pairs: np.ndarray
    go_pairs: np.ndarray
    disulfides: np.ndarray
    types: np.ndarray
    alpha: np.ndarray
    volume: np.ndarray
    names: np.ndarray


def _element(line: str, atom_name: str) -> str:
    el = line[76:78].strip()
    if el:
        return el.upper()
    return re.sub(r"^\d+", "", atom_name)[:1].upper()  # „1HB“ → H


def read_pdb_residues(path, chains=None) -> list:
    """Aminosäure-Residuen (ATOM sowie HETATM für :data:`HETATM_RESIDUES`, erstes Modell) mit Cα,
    allen Schweratomen (ohne H/D) und ggf. Cys-SG [nm]. Je Atom zählt die erste altLoc.
    Residuen ohne Cα werden übersprungen."""
    out: dict = {}
    order = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            rec = line[:6]
            if rec.startswith("ENDMDL"):
                break
            resname = line[17:20].strip()
            if rec != "ATOM  " and not (rec == "HETATM" and resname in HETATM_RESIDUES):
                continue
            chain = line[21]
            if chains is not None and chain not in chains:
                continue
            name = line[12:16].strip()
            if _element(line, name) in ("H", "D"):
                continue
            res = (chain, int(line[22:26]), line[26])
            if res not in out:
                std, charge = RESIDUE_ALIASES.get(resname, (resname, None))
                out[res] = {"name": std, "charge": charge, "atoms": {}}
                order.append(res)
            atoms = out[res]["atoms"]
            if name in atoms:  # weitere altLoc desselben Atoms
                continue
            atoms[name] = np.array([float(line[30:38]), float(line[38:46]),
                                    float(line[46:54])]) * 0.1  # fmt: skip
    residues = []
    for key in order:
        r = out[key]
        if "CA" not in r["atoms"]:
            continue
        charge = RESIDUE_CHARGE.get(r["name"], 0.0) if r["charge"] is None else r["charge"]
        heavy = np.array(list(r["atoms"].values()))
        sg = r["atoms"].get("SG") if r["name"] == "CYS" else None
        residues.append(
            Residue(key[0], key[1], key[2], r["name"], charge, r["atoms"]["CA"], heavy, sg)
        )
    return residues


def _residue_composition(name: str, c_terminal: bool = False) -> dict:
    if name not in builder._RESIDUES:
        raise ValueError(f"cg: unbekanntes Residuum {name!r} (Standardaminosäuren und Aliase)")
    comp = builder.parse_formula(builder._RESIDUES[name][0])
    if c_terminal:  # H₂O der Kette dem letzten Residuum zuschlagen
        comp = {**comp, "H": comp.get("H", 0) + 2, "O": comp.get("O", 0) + 1}
    return comp


def enm_go_from_pdb(path, *, chains=None, domains=None, enm_cutoff_nm: float = 0.9,
                    k_enm: float = 500.0, k_bond: float = 1.0e4, k_angle: float = 50.0,
                    go_cutoff_nm: float = 0.45, eps_go: float = 2.5, go_form: str = "12-10",
                    sigma_bead: float = 0.4, eps_bead: float = 0.05, charged_termini: bool = True,
                    optics=None) -> CGModel:  # fmt: skip
    """Cα-ENM/Gō-Modell eines Proteins (ADR 0017 Stufe 1).

    Args:
        path: PDB-Datei. chains: Ketten-IDs oder None (alle).
        domains: Liste von Domänen; jede Domäne ist eine Liste von (Kette, erstes, letztes Residuum)
            (resseq inklusive). None ⇒ eine Domäne je Kette. Residuen ohne Domäne (Gelenk) haben
            keine Federn, nur Kette, Winkel und Gō-Kontakte. Spans ohne Treffer sind ein Fehler.
        enm_cutoff_nm, k_enm: Federn zwischen Cα derselben Domäne [nm], [kJ/mol/nm²].
        k_bond: Ketten- und Disulfidbindung [kJ/mol/nm²]. k_angle: kosinus-harmonischer Cα-Winkel
            [kJ/mol] (0 ⇒ keine Winkel).
        go_cutoff_nm: Schweratomabstand nativer Kontakte [nm]. eps_go: Kontakttiefe [kJ/mol].
        go_form: "12-10" (Karanicolas-Brooks) oder "12-6".
        sigma_bead, eps_bead: unspezifisches LJ [nm], [kJ/mol].
        charged_termini: N-Terminus +1, C-Terminus −1 je Kette.
        optics: Element-Tabelle für α′/v; Default ``builder.calibrate_optics("protein")[0]``.
    """
    if go_form not in ("12-10", "12-6"):
        raise ValueError("go_form: '12-10' oder '12-6'")
    residues = read_pdb_residues(path, chains)
    n = len(residues)
    if n < 2:
        raise ValueError("cg: weniger als zwei Residuen mit Cα")
    pos = np.array([r.ca for r in residues])
    names = [r.name for r in residues]
    chain_of = np.array([r.chain for r in residues])
    last_of_chain = np.zeros(n, bool)
    for c in dict.fromkeys(chain_of):
        last_of_chain[np.flatnonzero(chain_of == c)[-1]] = True
    comps = [_residue_composition(m, bool(t)) for m, t in zip(names, last_of_chain, strict=True)]
    masses = np.array([sum(k * builder.ELEMENT_MASS[e] for e, k in c.items()) for c in comps])
    charges = np.array([r.charge for r in residues])

    bonds = []
    for i in range(n - 1):
        if chain_of[i] != chain_of[i + 1]:
            continue
        if np.linalg.norm(pos[i + 1] - pos[i]) < BOND_GAP_NM:
            bonds.append((i, i + 1))
        else:
            r0, r1 = residues[i], residues[i + 1]
            warnings.warn(
                f"cg: Kettenlücke in {r0.chain} zwischen {r0.resseq}{r0.icode.strip()} und "
                f"{r1.resseq}{r1.icode.strip()} (Cα–Cα > {BOND_GAP_NM} nm), keine Bindung",
                RuntimeWarning,
                stacklevel=2,
            )
    if charged_termini:
        for c in dict.fromkeys(chain_of):
            idx = np.flatnonzero(chain_of == c)
            charges[idx[0]] += 1.0
            charges[idx[-1]] -= 1.0

    # Domänen
    domain = np.full(n, -1, np.int64)
    if domains is None:
        for d, c in enumerate(dict.fromkeys(chain_of)):
            domain[chain_of == c] = d
    else:
        for d, spans in enumerate(domains):
            for chain, first, last in spans:
                sel = np.array([r.chain == chain and first <= r.resseq <= last for r in residues])
                if not np.any(sel):
                    raise ValueError(
                        f"cg: Domänen-Span ({chain}, {first}, {last}) trifft kein Residuum"
                    )
                if np.any(domain[sel] >= 0):
                    raise ValueError("cg: Residuum in mehreren Domänen")
                domain[sel] = d

    # Disulfide
    cys = [i for i, r in enumerate(residues) if r.sg is not None]
    disulfides = []
    for a_i, a in enumerate(cys):
        for b in cys[a_i + 1 :]:
            if np.linalg.norm(residues[a].sg - residues[b].sg) < DISULFIDE_NM:
                disulfides.append((a, b))
    disulfides = np.array(disulfides, np.int64).reshape(-1, 2)

    bonded_set = {tuple(b) for b in bonds}
    excl13 = [(i, j + 1) for (i, j) in bonds if (j, j + 1) in bonded_set]
    angles = np.array([(i, j, j + 1) for (i, j) in bonds if (j, j + 1) in bonded_set],
                      np.int64).reshape(-1, 3)  # fmt: skip
    ss_set = {tuple(p) for p in disulfides.tolist()}
    d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    iu, ju = np.triu_indices(n, 1)
    same_dom = (domain[iu] == domain[ju]) & (domain[iu] >= 0)
    not_bonded = np.array([(a, b) not in bonded_set and (a, b) not in ss_set
                           for a, b in zip(iu, ju, strict=True)])  # fmt: skip
    enm_mask = same_dom & (d[iu, ju] < enm_cutoff_nm) & not_bonded
    enm_pairs = np.stack([iu[enm_mask], ju[enm_mask]], 1)

    go = []
    for a, b in zip(iu[~same_dom], ju[~same_dom], strict=True):
        if chain_of[a] == chain_of[b] and abs(a - b) < 3:
            continue
        if (a, b) in ss_set or d[a, b] > go_cutoff_nm + 2.0:  # Cα-Vorauswahl (Seitenketten < 1 nm)
            continue
        ha, hb = residues[a].heavy, residues[b].heavy
        if np.sqrt(((ha[:, None, :] - hb[None, :, :]) ** 2).sum(-1).min()) < go_cutoff_nm:
            go.append((a, b))
    go_pairs = np.array(go, np.int64).reshape(-1, 2)

    excluded = np.array(sorted(bonded_set | set(excl13) | ss_set | {tuple(p) for p in go_pairs.tolist()}),
                        np.int32).reshape(-1, 2)  # fmt: skip
    chain_bonds = np.array(bonds, np.int64).reshape(-1, 2)
    all_bonds = np.concatenate([chain_bonds, disulfides, enm_pairs])
    bond_k = np.concatenate([np.full(len(chain_bonds) + len(disulfides), k_bond),
                             np.full(len(enm_pairs), k_enm)])  # fmt: skip
    bond_r0 = d[all_bonds[:, 0], all_bonds[:, 1]]
    if k_angle <= 0:
        angles = np.zeros((0, 3), np.int64)
    va = pos[angles[:, 0]] - pos[angles[:, 1]]
    vb = pos[angles[:, 2]] - pos[angles[:, 1]]
    theta0 = np.arctan2(np.linalg.norm(np.cross(va, vb), axis=1), (va * vb).sum(1))
    d_go = d[go_pairs[:, 0], go_pairs[:, 1]] if len(go_pairs) else np.zeros(0)
    bonded = {
        "bonds": all_bonds.astype(np.int32), "bond_k": bond_k, "bond_r0": bond_r0,
        "angles": angles.astype(np.int32), "angle_k": np.full(len(angles), float(k_angle)),
        "angle_theta0": theta0, "angle_form": np.ones(len(angles), np.int32),
        "torsions": np.zeros((0, 4), np.int32), "torsion_k": np.zeros(0),
        "torsion_n": np.zeros(0, np.int32), "torsion_phase": np.zeros(0),
        "pairs14": go_pairs.astype(np.int32),
        "pair_qq": charges[go_pairs[:, 0]] * charges[go_pairs[:, 1]] if len(go_pairs) else np.zeros(0),
        "pair_sigma": d_go if go_form == "12-10" else d_go / 2.0 ** (1.0 / 6.0),
        "pair_eps": np.full(len(go_pairs), eps_go),
        "pair_form": np.full(len(go_pairs), 1 if go_form == "12-10" else 0, np.int32),
    }  # fmt: skip
    spec = builder.SystemSpec(
        masses=masses, charges=charges, sigma=np.full(n, sigma_bead), epsilon=np.full(n, eps_bead),
        bonded=bonded, excluded=excluded, elements=np.full(n, "X"), positions=pos,
        atom_names=np.full(n, "CA"), residue_names=np.array(names),
    )  # fmt: skip

    table = builder.calibrate_optics("protein")[0] if optics is None else optics
    labels = [m + ("+H2O" if t else "") for m, t in zip(names, last_of_chain, strict=True)]
    type_names = list(dict.fromkeys(labels))
    types = np.array([type_names.index(lab) + 1 for lab in labels], np.int32)
    alpha = np.zeros(len(type_names) + 1)
    volume = np.zeros(len(type_names) + 1)
    for t, lab in enumerate(type_names, start=1):
        comp = _residue_composition(lab.split("+")[0], lab.endswith("+H2O"))
        alpha[t] = sum(k * table[e][0] for e, k in comp.items())
        volume[t] = sum(k * table[e][1] for e, k in comp.items())
    return CGModel(spec, residues, domain, enm_pairs, go_pairs, disulfides, types,
                   alpha.astype(np.float32), volume.astype(np.float32),
                   np.array([""] + type_names))  # fmt: skip
