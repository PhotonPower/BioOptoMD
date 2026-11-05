"""Hilfen für ``VoxelGridForce3D`` (docs/PHYSICS_MODEL.md §4.3).

Layout der Gitterwerte: ``values[k, j, i] = (V, ∂V/∂x, ∂V/∂y, ∂V/∂z)`` am Punkt
``(x0 + i·Δx, y0 + j·Δy, z0 + k·Δz)``; Shape (nz, ny, nx, 4), float32. Einheiten: V in kJ/mol pro
Skalierungseinheit, Gradient in kJ/mol/nm. Im Optikmodus ist die Gittergröße W = |E₀|² in (V/nm)²
und die Skalierung je Typ kommt aus :func:`optical_scale` (OQ-04).
"""

from __future__ import annotations

import numpy as np

from . import units


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


def optical_scale(alpha_nm3, volume_nm3, n_water: float, *, amplitude: str):
    """Typskalierung s_i [kJ/mol pro (V/nm)²] der optischen Gradientenkraft (OQ-04).

    Das Gitter speichert W = |E₀|² in (V/nm)² (z. B. aus FEM) samt Gradient; U_i = s_i W.
    Mit dem Polarisierbarkeitskontrast gegenüber dem verdrängten Wasser

        L_w = (n_w² − 1)/(n_w² + 2),   α′_eff,i = α′_i − (3/4π) v_i L_w
        s_i = −α′_eff,i · FARADAY² / (4 · ONE_4PI_EPS0)     (Spitzenamplitude, ¼ Re α |E₀|²)

    Bei RMS-Feldern ist der Vorfaktor ½ statt ¼ (s_i verdoppelt). Streu-/Absorptionskräfte (Im α)
    sind nicht abgebildet; Fluorophore sollten im Optikmodus s_i = 0 erhalten (OQ-04).

    Args:
        alpha_nm3: α′_i [nm³] je Optik-Typ (OQ-01).
        volume_nm3: v_i [nm³] je Optik-Typ, dieselben wie im RefractiveIndexProfiler (OQ-02).
        n_water: Brechungsindex des Lösungsmittels bei der Wellenlänge.
        amplitude: Pflichtargument, 'peak' (E(t) = Re[E₀ e^{−iωt}]) oder 'rms'.

    Returns:
        float32 (n_types,) – direkt als ``scale`` für ``ctx.add_voxel_grid``.
    """
    if amplitude not in ("peak", "rms"):
        raise ValueError("amplitude: 'peak' oder 'rms'")
    alpha = np.asarray(alpha_nm3, np.float64)
    vol = np.asarray(volume_nm3, np.float64)
    if alpha.shape != vol.shape or alpha.ndim != 1:
        raise ValueError("alpha und volume: gleiche 1D-Shape (n_types,)")
    l_w = (n_water**2 - 1.0) / (n_water**2 + 2.0)
    alpha_eff = alpha - 3.0 / (4.0 * np.pi) * vol * l_w
    factor = 0.25 if amplitude == "peak" else 0.5
    return (-factor * alpha_eff * units.FARADAY**2 / units.ONE_4PI_EPS0).astype(np.float32)
