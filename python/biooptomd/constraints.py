"""Builder-Hilfen für H-Bindungs-Constraints und Hydrogen Mass Repartitioning (ROADMAP Phase 5,
docs/PHYSICS_MODEL.md §5).

- :func:`repartition_hydrogen_mass`: H-Masse auf ``h_mass`` (Default 3.024 amu), die Differenz wird vom
  gebundenen Schweratom abgezogen (Gesamtmasse erhalten).
- :func:`hbond_clusters`: Cluster (Schweratom + seine H-Atome) für SHAKE/RATTLE.
- :func:`constraint_topology`: Cluster-dict für ``SimulationContext.set_constraints`` und die kovalente
  Topologie ohne die constrainten Bindungen (wie OpenMM mit ``constraints=HBonds``).

Einheiten: amu, nm.
"""

from __future__ import annotations

import numpy as np

MAX_H_PER_CLUSTER = 3


def _bond_partners(bonds, is_hydrogen):
    """Für jede H-Schweratom-Bindung: (heavy, h, Bond-Index)."""
    bonds = np.asarray(bonds, np.int64).reshape(-1, 2)
    h = np.asarray(is_hydrogen, bool)
    out = []
    for k, (i, j) in enumerate(bonds):
        if h[i] and h[j]:
            raise ValueError(f"H-H-Bindung ({i}, {j}) wird nicht unterstützt")
        if h[i] != h[j]:
            out.append((j, i, k) if h[i] else (i, j, k))
    return out


def repartition_hydrogen_mass(masses, bonds, is_hydrogen, h_mass: float = 3.024):
    """Hydrogen Mass Repartitioning.

    Args:
        masses: Atommassen [amu], (n,).
        bonds: Bindungen (n_b, 2) lokaler Indizes.
        is_hydrogen: bool (n,).
        h_mass: Zielmasse der H-Atome [amu].

    Returns:
        Neue Massen [amu] (n,), Gesamtmasse unverändert.
    """
    m = np.array(masses, np.float64, copy=True)
    h = np.asarray(is_hydrogen, bool)
    partners = _bond_partners(bonds, is_hydrogen)
    bonded_h = [hh for _, hh, _ in partners]
    if len(bonded_h) != h.sum() or set(bonded_h) != set(np.flatnonzero(h)):
        raise ValueError("jedes H-Atom braucht genau eine Bindung zu einem Schweratom")
    for heavy, hh, _ in partners:
        delta = h_mass - m[hh]
        m[hh] += delta
        m[heavy] -= delta
    if (m[~h] < h_mass).any():
        raise ValueError("HMR würde ein Schweratom leichter als h_mass machen")
    return m


def hbond_clusters(bonds, bond_r0, is_hydrogen):
    """Constraint-Cluster aus den H-Schweratom-Bindungen.

    Returns:
        clusters: int32 (n_clusters, 4): (Schweratom, H1, H2, H3), fehlende H = −1.
        distances: float32 (n_clusters, 3): Sollabstände [nm] (= Bindungs-r0), fehlende = 0.
        constrained: bool (n_bonds,): welche Bindungen constraint sind.
    """
    r0 = np.asarray(bond_r0, np.float64)
    partners = _bond_partners(bonds, is_hydrogen)
    by_heavy: dict[int, list[tuple[int, float]]] = {}
    constrained = np.zeros(len(r0), bool)
    for heavy, hh, k in partners:
        by_heavy.setdefault(int(heavy), []).append((int(hh), float(r0[k])))
        constrained[k] = True
    clusters = -np.ones((len(by_heavy), 4), np.int32)
    distances = np.zeros((len(by_heavy), 3), np.float32)
    for c, heavy in enumerate(sorted(by_heavy)):
        hs = by_heavy[heavy]
        if len(hs) > MAX_H_PER_CLUSTER:
            raise ValueError(f"Schweratom {heavy} hat mehr als {MAX_H_PER_CLUSTER} H-Atome")
        clusters[c, 0] = heavy
        for k, (hh, d) in enumerate(hs):
            clusters[c, 1 + k] = hh
            distances[c, k] = d
    return clusters, distances, constrained


def constraint_topology(bonded_topology: dict, is_hydrogen):
    """Cluster-dict und kovalente Topologie ohne die constrainten H-Bindungen.

    Returns:
        (constraints dict {clusters, distances}, bonded dict ohne constrainte Bindungen).
    """
    clusters, distances, constrained = hbond_clusters(
        bonded_topology["bonds"], bonded_topology["bond_r0"], is_hydrogen
    )
    reduced = dict(bonded_topology)
    keep = ~constrained
    for key in ("bonds", "bond_k", "bond_r0"):
        reduced[key] = np.asarray(bonded_topology[key])[keep]
    return {"clusters": clusters, "distances": distances}, reduced
