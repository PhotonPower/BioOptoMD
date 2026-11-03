"""NumPy-Referenz (FP64) für PlanarSurfaceForce, docs/PHYSICS_MODEL.md §4.1.

9-3-Wand bei z = z0 (Abstand ζ = z − z0), Parameter pro Oberflächen-Typ t = types.y:

    V_i(ζ) = ε_t [ (2/15)(σ_t/ζ)^9 − (σ_t/ζ)^3 ]               [kJ/mol]
    F_z,i  = −dV/dζ = ε_t [ (6/5) σ_t^9/ζ^10 − 3 σ_t^3/ζ^4 ]    [kJ/mol/nm]

Für ζ < ζ_c wird ζ auf ζ_c gesetzt (V und F dort konstant fortgesetzt; das Ereignis wird gezählt),
mit ζ_c,t = max(clamp_factor·σ_t, ζ_F,t) und |F(ζ_F,t)| = F_max (ADR 0006). Atome mit ε_t = 0 oder
σ_t = 0 spüren keine Wand.

Linearisiertes Gouy-Chapman (optional), Φ0 [kJ/(mol·e)], κ [1/nm]:

    V_i = q_i Φ0 exp(−κ ζ),   F_z,i = q_i Φ0 κ exp(−κ ζ)     für ζ ≥ 0
    V_i = q_i Φ0,             F_z,i = 0                      für ζ < 0

(ASSUMPTION(OQ-09): hinter der Wand konstant fortgesetzt; dort sollte wegen der 9-3-Wand nie
ein Atom sein.)
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq


def wall_force(zeta, eps, sigma):
    """F_z [kJ/mol/nm] der 9-3-Wand im Abstand ζ [nm]."""
    s3 = (sigma / zeta) ** 3
    return eps * (6.0 / 5.0 * s3**3 - 3.0 * s3) / zeta


def clamp_distance(eps, sigma, clamp_factor=0.1, max_force=1.0e6):
    """ζ_c = max(clamp_factor·σ, ζ_F) mit |F(ζ_F)| = max_force auf dem repulsiven Ast (ADR 0006)."""
    if eps == 0.0 or sigma <= 0.0:
        return 0.0
    lo = clamp_factor * sigma
    z_star = 0.4 ** (1.0 / 6.0) * sigma
    if lo >= z_star or abs(wall_force(lo, eps, sigma)) <= max_force:
        return lo
    return brentq(lambda z: abs(wall_force(z, eps, sigma)) - max_force, lo, z_star, xtol=1e-15)


def wall_93(z, eps, sigma, z0=0.0, clamp_factor=0.1, max_force=1.0e6):
    """Energie [kJ/mol] und F_z [kJ/mol/nm] der 9-3-Wand pro Atom, plus Clamp-Maske."""
    z = np.asarray(z, np.float64)
    eps = np.asarray(eps, np.float64)
    sigma = np.asarray(sigma, np.float64)
    active = (eps != 0.0) & (sigma > 0.0)
    zeta_c = np.vectorize(lambda e, s: clamp_distance(e, s, clamp_factor, max_force))(eps, sigma)
    zeta = z - z0
    clamped = active & (zeta < zeta_c)
    zeta = np.where(clamped, zeta_c, zeta)
    zeta = np.where(active, zeta, 1.0)
    s3 = (sigma / zeta) ** 3
    s9 = s3**3
    e = np.where(active, eps * (2.0 / 15.0 * s9 - s3), 0.0)
    f = np.where(active, eps * (6.0 / 5.0 * s9 - 3.0 * s3) / zeta, 0.0)
    # Hinter dem Clamp ist V konstant ⇒ F = 0 wäre energiekonsistent; spezifiziert ist jedoch
    # die Kraft am Clamp-Punkt (repulsiv, schiebt das Atom zurück). Siehe PHYSICS_MODEL §4.1.
    return e, f, clamped


def gouy_chapman(z, q, phi0, kappa, z0=0.0):
    """Energie [kJ/mol] und F_z [kJ/mol/nm] des linearisierten Gouy-Chapman-Terms pro Atom."""
    z = np.asarray(z, np.float64)
    q = np.asarray(q, np.float64)
    zeta = z - z0
    ex = np.exp(-kappa * np.maximum(zeta, 0.0))
    e = q * phi0 * ex
    f = np.where(zeta >= 0.0, q * phi0 * kappa * ex, 0.0)
    return e, f


def surface_energy_force(
    z, q, eps, sigma, z0=0.0, clamp_factor=0.1, phi0=0.0, kappa=0.0, max_force=1.0e6
):
    """Summe aus 9-3-Wand und Gouy-Chapman pro Atom."""
    e1, f1, clamped = wall_93(z, eps, sigma, z0, clamp_factor, max_force)
    e2, f2 = gouy_chapman(z, q, phi0, kappa, z0)
    return e1 + e2, f1 + f2, clamped
