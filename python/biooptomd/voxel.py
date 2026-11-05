"""Hilfen für ``VoxelGridForce3D`` (docs/PHYSICS_MODEL.md §4.3).

Layout der Gitterwerte: ``values[k, j, i] = (V, ∂V/∂x, ∂V/∂y, ∂V/∂z)`` am Punkt
``(x0 + i·Δx, y0 + j·Δy, z0 + k·Δz)``; Shape (nz, ny, nx, 4), float32. Einheiten: V in kJ/mol pro
Skalierungseinheit (z. B. pro nm³ Polarisierbarkeitsvolumen, OQ-04), Gradient in kJ/mol/nm.
"""

from __future__ import annotations

import numpy as np


def values_from_potential(potential, spacing_nm, periodic_xy: bool = False) -> np.ndarray:
    """Gitterwerte aus einem Potential auf dem Gitter; Gradient per zentraler Differenz (2. Ordnung,
    an Rändern einseitig; bei ``periodic_xy`` zyklisch in x/y).

    Args:
        potential: V [kJ/mol], Shape (nz, ny, nx).
        spacing_nm: (Δx, Δy, Δz) [nm].
        periodic_xy: Gitter ist in x/y periodisch (deckt die Box exakt ab).

    Returns:
        float32 (nz, ny, nx, 4).
    """
    v = np.asarray(potential, np.float64)
    if v.ndim != 3:
        raise ValueError("potential: Shape (nz, ny, nx) erwartet")
    dx, dy, dz = (float(s) for s in spacing_nm)
    if periodic_xy:
        gx = (np.roll(v, -1, axis=2) - np.roll(v, 1, axis=2)) / (2 * dx)
        gy = (np.roll(v, -1, axis=1) - np.roll(v, 1, axis=1)) / (2 * dy)
    else:
        gx = np.gradient(v, dx, axis=2)
        gy = np.gradient(v, dy, axis=1)
    gz = np.gradient(v, dz, axis=0)
    return np.stack([v, gx, gy, gz], axis=-1).astype(np.float32)
