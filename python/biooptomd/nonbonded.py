"""Builder-Hilfen für den Tiled-All-Pairs-Nonbonded-Kernel (ROADMAP Phase 4, docs/PHYSICS_MODEL.md §3).

- :func:`lj_type_table`: Atom-LJ-Parameter → Typindizes + Paartabelle (Lorentz-Berthelot).
  Typ 0 ist reserviert (ε = 0) für Padding-Atome (AGENTS.md Regel 6).
- :func:`exclusion_tiles`: ausgeschlossene Paare (1-2, 1-3, 1-4) → 32-bit-Bitmasken je (Atom, Tile),
  nur für Tile-Paare mit mindestens einem Ausschluss (Tile-Liste).
- :func:`tile_pairs`: Reihenfolge der Tile-Paare (I ≤ J), identisch zum Kernel.
- :func:`water_permittivity`, :func:`debye_kappa`: ε_r(T) und κ(I, T, ε_r) pro Replica (OQ-10).
- :func:`cutoff_from_kappa`: r_c = max(2 nm, n_κ/min κ_r), r_s = r_c − 0.2 nm (ADR 0006). Die Box
  bleibt unverändert; bei L < 2 r_c summiert der Kernel über periodische Bilder.

Einheiten: σ in nm, ε in kJ/mol, I in mol/L, T in K, κ in 1/nm.
"""

from __future__ import annotations

import numpy as np

from . import units

TILE = 32


def padded_atom_count(n_atoms: int) -> int:
    """ceil(n_atoms / 32) · 32."""
    return (n_atoms + TILE - 1) // TILE * TILE


def water_permittivity(temperature_k):
    """Relative Permittivität von Wasser ε_r(T) nach Malmberg & Maryott (1956), 0–100 °C:
    ε_r = 87.740 − 0.40008 t + 9.398·10⁻⁴ t² − 1.410·10⁻⁶ t³ mit t = T − 273.15 K in °C.

    ASSUMPTION(OQ-10): Koeffizienten aus der Sekundärliteratur, an der Originalarbeit nicht
    verifiziert (Checkliste in docs/OPEN_QUESTIONS.md).
    """
    t = np.asarray(temperature_k, np.float64) - 273.15
    if ((t < 0.0) | (t > 100.0)).any():
        raise ValueError("water_permittivity: gültig für 273.15 K ≤ T ≤ 373.15 K")
    return 87.740 - 0.40008 * t + 9.398e-4 * t**2 - 1.410e-6 * t**3


def debye_kappa(ionic_strength_molar, temperature_k, eps_r):
    """Inverse Debye-Länge κ [1/nm] = sqrt(8π l_B N_A I), l_B = ONE_4PI_EPS0/(ε_r k_B T) [nm].

    Args:
        ionic_strength_molar: Ionenstärke I [mol/L] aller nicht explizit modellierten Ionen
            (OQ-08: explizite Mg²⁺ zählen nicht mit), >= 0.
        temperature_k: T [K], > 0 (Skalar oder je Replica).
        eps_r: relative Permittivität (Skalar oder je Replica), > 0.
    """
    ionic = np.asarray(ionic_strength_molar, np.float64)
    t = np.asarray(temperature_k, np.float64)
    eps = np.asarray(eps_r, np.float64)
    if (ionic < 0).any() or (t <= 0).any() or (eps <= 0).any():
        raise ValueError("debye_kappa: I >= 0, T > 0, eps_r > 0 erwartet")
    bjerrum = units.ONE_4PI_EPS0 / (eps * units.KB * t)  # nm
    ion_density = units.AVOGADRO * ionic * 1.0e-24  # 1/nm³ (1 L = 10²⁴ nm³)
    return np.sqrt(8.0 * np.pi * bjerrum * ion_density)


def cutoff_from_kappa(kappa_per_nm, n_kappa: float = 5.0, r_min: float = 2.0, width: float = 0.2):
    """Cutoff aus der Debye-Länge (ADR 0006): r_c = max(r_min, n_κ / min_r κ_r), r_s = r_c − width.

    Ein gemeinsames r_c für alle Replicas (ein Kernel). Bei n_κ = 5 ist DH bei r_c auf
    exp(−5) ≈ 0.7 % der reinen Coulomb-Wechselwirkung abgefallen. Die Box wird nicht angepasst
    (L ist eine physikalische Eingabe, OQ-09).

    Args:
        kappa_per_nm: κ_r [1/nm] je Replica (> 0; ohne Salz ist kein Cutoff aus κ definiert).

    Returns:
        (r_switch, r_cutoff) [nm].
    """
    kappa = np.atleast_1d(np.asarray(kappa_per_nm, np.float64))
    if kappa.size == 0 or not (kappa > 0).all() or not np.isfinite(kappa).all():
        raise ValueError("cutoff_from_kappa: κ > 0 für alle Replicas erwartet")
    if not (n_kappa > 0 and r_min > width > 0):
        raise ValueError("cutoff_from_kappa: n_kappa > 0 und r_min > width > 0 erwartet")
    r_c = max(r_min, n_kappa / float(kappa.min()))
    return r_c - width, r_c


def tile_pairs(n_pad: int) -> np.ndarray:
    """Alle Tile-Paare (I, J) mit I ≤ J in Kernel-Reihenfolge: for I: for J ≥ I. int32 (n_tp, 2)."""
    n_tiles = n_pad // TILE
    return np.array([(i, j) for i in range(n_tiles) for j in range(i, n_tiles)], dtype=np.int32)


def lj_type_table(sigma_nm, epsilon_kj_mol, decimals: int = 6):
    """Fasst gleiche (σ, ε) zu Typen zusammen und baut die Paartabelle.

    Args:
        sigma_nm: σ_i [nm], (n_atoms,)
        epsilon_kj_mol: ε_i [kJ/mol], (n_atoms,)
        decimals: Rundung beim Zusammenfassen gleicher Parameter.

    Returns:
        types: int32 (n_atoms,), Werte ≥ 1 (0 = Padding-Typ).
        table: float32 (n_types, n_types, 2) mit (σ_ij, ε_ij), σ_ij = (σ_i + σ_j)/2,
            ε_ij = sqrt(ε_i ε_j); Zeile/Spalte 0 hat ε = 0.
    """
    sigma = np.asarray(sigma_nm, np.float64)
    eps = np.asarray(epsilon_kj_mol, np.float64)
    if sigma.shape != eps.shape or sigma.ndim != 1:
        raise ValueError("sigma und epsilon: gleiche 1D-Shape erwartet")
    if (eps < 0).any() or (sigma < 0).any():
        raise ValueError("sigma, epsilon müssen >= 0 sein")
    keys = np.round(np.stack([sigma, eps], axis=1), decimals)
    _, first, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    types = (inverse.reshape(-1) + 1).astype(np.int32)
    # Originalwerte (nicht die gerundeten Schlüssel) des ersten Atoms jedes Typs verwenden.
    s = np.concatenate([[0.0], sigma[first]])
    e = np.concatenate([[0.0], eps[first]])
    table = np.empty((len(s), len(s), 2), np.float32)
    table[..., 0] = 0.5 * (s[:, None] + s[None, :])
    table[..., 1] = np.sqrt(e[:, None] * e[None, :])
    return types, table


def nonbonded_topology(n_pad: int, sigma_nm, epsilon_kj_mol, excluded_pairs):
    """Topologie-dict für ``SimulationContext.set_nonbonded_tiled`` und LJ-Typen der Atome.

    Args:
        n_pad: gepaddete Atomzahl des Kontexts.
        sigma_nm, epsilon_kj_mol: LJ-Parameter je Atom (n_atoms,).
        excluded_pairs: ausgeschlossene Paare (1-2, 1-3, 1-4), int (n_excl, 2).

    Returns:
        (types int32 (n_atoms,) für ``types[..., 0]``, dict mit lj_table, mask_index, masks).
    """
    types, table = lj_type_table(sigma_nm, epsilon_kj_mol)
    mask_index, masks = exclusion_tiles(n_pad, excluded_pairs)
    return types, {"lj_table": table, "mask_index": mask_index, "masks": masks}


def exclusion_tiles(n_pad: int, pairs) -> tuple[np.ndarray, np.ndarray]:
    """Bitmasken der ausgeschlossenen Paare für den Tile-Kernel.

    Bit b in ``masks[m, a]`` ist gesetzt ⇔ Paar (I·32 + a, J·32 + b) ist ausgeschlossen, wobei
    (I, J) = ``tile_pairs(n_pad)[p]`` und ``mask_index[p] = m``. Für Diagonal-Tiles (I = J) werden
    beide Richtungen gesetzt. Selbstwechselwirkung und Paare mit Padding-Atomen behandelt der Kernel.

    Args:
        n_pad: gepaddete Atomzahl (Vielfaches von 32).
        pairs: ausgeschlossene Paare lokaler Atomindizes, int (n_excl, 2).

    Returns:
        mask_index: int32 (n_tile_pairs,), −1 = Tile-Paar ohne Ausschlüsse.
        masks: uint32 (n_mask_tiles, 32).
    """
    if n_pad % TILE:
        raise ValueError("n_pad muss ein Vielfaches von 32 sein")
    pairs = np.asarray(pairs, np.int64).reshape(-1, 2)
    if pairs.size and (pairs.min() < 0 or pairs.max() >= n_pad):
        raise ValueError("Atomindex außerhalb [0, n_pad)")
    tps = tile_pairs(n_pad)
    n_tiles = n_pad // TILE
    lookup = -np.ones((n_tiles, n_tiles), np.int64)
    lookup[tps[:, 0], tps[:, 1]] = np.arange(len(tps))
    mask_index = -np.ones(len(tps), np.int32)
    blocks: dict[int, np.ndarray] = {}

    def set_bit(i: int, j: int) -> None:
        ti, tj = i // TILE, j // TILE
        p = int(lookup[ti, tj])
        if p not in blocks:
            blocks[p] = np.zeros(TILE, np.uint32)
        blocks[p][i % TILE] |= np.uint32(1) << np.uint32(j % TILE)

    for i, j in pairs:
        i, j = int(i), int(j)
        if i == j:
            continue
        if i // TILE > j // TILE:  # Tile-Paare sind nur für I ≤ J gespeichert
            i, j = j, i
        if i // TILE == j // TILE:
            set_bit(i, j)
            set_bit(j, i)
        else:
            set_bit(i, j)
    masks = np.zeros((len(blocks), TILE), np.uint32)
    for m, p in enumerate(sorted(blocks)):
        mask_index[p] = m
        masks[m] = blocks[p]
    return mask_index, masks
