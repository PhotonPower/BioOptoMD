"""NumPy-Referenz (FP64) für PlanarSurfaceForce, docs/PHYSICS_MODEL.md §4.1 und ADR 0007.

Wand bei z = z0 (Abstand ζ = z − z0). Parameter pro Eintrag (Oberflächen-Typ s, LJ-Typ l):
Vorfaktor C [kJ/mol] und σ [nm]; s = σ/ζ. Wandformen (ADR 0007, C enthält ρ, ε_iw und σ_iw):

    9-3     V = C [ (2/15) s^9 − s^3 ]                      C = (2π/3) ρ ε σ³
            F = C [ (6/5) s^9 − 3 s^3 ] / ζ
    10-4    V = C [ (2/5) s^10 − s^4 ]                      C = 2π ρ_s ε σ²
            F = 4 C [ s^10 − s^4 ] / ζ
    10-4-3  V = C [ (2/5) s^10 − s^4 ] − B/(ζ + a)^3        C = 2π ρ ε σ² Δ
            F = 4 C [ s^10 − s^4 ] / ζ − 3 B/(ζ + a)^4      B = C σ⁴/(3Δ), a = 0.61 Δ

Für ζ < ζ_c wird ζ auf ζ_c gesetzt (V und F dort ausgewertet, das Ereignis gezählt), mit
ζ_c = max(clamp_factor·σ, ζ_F) und |F(ζ_F)| = F_max auf dem repulsiven Ast (ADR 0008).
Einträge mit C = 0 oder σ = 0 erzeugen keine Wandkraft.

Gouy-Chapman (OQ-06), nichtlinear (Default), Φ0 [kJ/(mol·e)], κ [1/nm], k_BT [kJ/mol]:

    γ = tanh(Φ0/(4 k_BT)),  u = γ exp(−κ ζ)
    V_i = q_i · 2 k_BT · ln[(1 + u)/(1 − u)],   F_z,i = q_i · 4 k_BT κ u/(1 − u²)

linear (Grenzfall Φ0 → 0):  V_i = q_i Φ0 exp(−κ ζ),  F_z,i = q_i Φ0 κ exp(−κ ζ).
Für ζ < 0 gilt in beiden Fällen V = V(0), F = 0 (ASSUMPTION(OQ-12)).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

WALL_FORMS = ("9-3", "10-4", "10-4-3")
GRAPHITE_SHIFT = 0.61  # a = 0.61 Δ (Steele)


def wall_energy_force(zeta, form, c, sigma, delta=0.335):
    """Energie [kJ/mol] und F_z [kJ/mol/nm] der Wandform `form` im Abstand ζ > 0 [nm]."""
    zeta = np.asarray(zeta, np.float64)
    s = sigma / zeta
    if form == "9-3":
        s3 = s**3
        return c * (2.0 / 15.0 * s3**3 - s3), c * (1.2 * s3**3 - 3.0 * s3) / zeta
    if form not in ("10-4", "10-4-3"):
        raise ValueError(f"unbekannte Wandform {form!r}")
    s4 = s**4
    s10 = s4 * s4 * s * s
    e = c * (0.4 * s10 - s4)
    f = 4.0 * c * (s10 - s4) / zeta
    if form == "10-4-3":
        b = c * sigma**4 / (3.0 * delta)
        za = zeta + GRAPHITE_SHIFT * delta
        e = e - b / za**3
        f = f - 3.0 * b / za**4
    return e, f


def wall_minimum(form, c, sigma, delta=0.335):
    """Lage z* [nm] und Tiefe V(z*) [kJ/mol]: analytisch für 9-3 und 10-4, numerisch für 10-4-3."""
    if form == "9-3":
        z = 0.4 ** (1.0 / 6.0) * sigma
    elif form == "10-4":
        z = sigma
    else:
        z = brentq(
            lambda x: wall_energy_force(x, form, c, sigma, delta)[1], 0.5 * sigma, 1.5 * sigma
        )
    return z, float(wall_energy_force(z, form, c, sigma, delta)[0])


def clamp_distance(form, c, sigma, delta=0.335, clamp_factor=0.1, max_force=1.0e6):
    """ζ_c = max(clamp_factor·σ, ζ_F) mit |F(ζ_F)| = max_force auf dem repulsiven Ast (ADR 0008)."""
    if c == 0.0 or sigma <= 0.0:
        return 0.0

    def force(z):
        return float(wall_energy_force(z, form, c, sigma, delta)[1])

    lo = clamp_factor * sigma
    sign = np.sign(c)  # repulsiver Ast: F hat das Vorzeichen von C
    z_zero = brentq(force, 0.5 * sigma, 1.5 * sigma)
    if lo >= z_zero or abs(force(lo)) <= max_force:
        return lo
    return brentq(lambda z: sign * force(z) - max_force, lo, z_zero, xtol=1e-15)


def wall(z, form, c, sigma, delta=0.335, z0=0.0, clamp_factor=0.1, max_force=1.0e6):
    """Energie, F_z pro Atom und Clamp-Maske; c, sigma pro Atom (bereits aus der Tabelle gewählt)."""
    z = np.asarray(z, np.float64)
    c = np.broadcast_to(np.asarray(c, np.float64), z.shape)
    sigma = np.broadcast_to(np.asarray(sigma, np.float64), z.shape)
    active = (c != 0.0) & (sigma > 0.0)
    zeta_c = np.zeros_like(z)
    for key in {(a, b) for a, b in zip(c[active], sigma[active], strict=True)}:
        mask = active & (c == key[0]) & (sigma == key[1])
        zeta_c[mask] = clamp_distance(form, key[0], key[1], delta, clamp_factor, max_force)
    zeta = z - z0
    clamped = active & (zeta < zeta_c)
    zeta = np.where(clamped, zeta_c, zeta)
    zeta = np.where(active, zeta, 1.0)
    e, f = wall_energy_force(
        zeta, form, np.where(active, c, 0.0), np.where(active, sigma, 1.0), delta
    )
    return np.where(active, e, 0.0), np.where(active, f, 0.0), clamped


def gouy_chapman(z, q, phi0, kappa, kt, z0=0.0, linear=False):
    """Energie [kJ/mol] und F_z [kJ/mol/nm] des Gouy-Chapman-Terms pro Atom (OQ-06, OQ-12)."""
    z = np.asarray(z, np.float64)
    q = np.asarray(q, np.float64)
    zeta = z - z0
    ex = np.exp(-kappa * np.maximum(zeta, 0.0))
    if linear:
        e = q * phi0 * ex
        f = q * phi0 * kappa * ex
    else:
        u = np.tanh(phi0 / (4.0 * kt)) * ex
        e = q * 4.0 * kt * np.arctanh(u)
        f = q * 4.0 * kt * kappa * u / (1.0 - u * u)
    return e, np.where(zeta >= 0.0, f, 0.0)


def surface_energy_force(
    z,
    q,
    c,
    sigma,
    form="9-3",
    delta=0.335,
    z0=0.0,
    clamp_factor=0.1,
    phi0=0.0,
    kappa=0.0,
    kt=1.0,
    max_force=1.0e6,
    linear=False,
):
    """Summe aus Wand und Gouy-Chapman pro Atom (Energie, F_z, Clamp-Maske)."""
    e1, f1, clamped = wall(z, form, c, sigma, delta, z0, clamp_factor, max_force)
    if phi0 == 0.0:
        return e1, f1, clamped
    e2, f2 = gouy_chapman(z, q, phi0, kappa, kt, z0, linear)
    return e1 + e2, f1 + f2, clamped
