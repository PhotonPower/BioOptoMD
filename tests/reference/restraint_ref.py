"""NumPy-Referenz (FP64) für HarmonicRestraintPullingForce, docs/PHYSICS_MODEL.md §4.2.

Positions-Restraint (Atom i, Referenz r⁰, Federkonstante k [kJ/mol/nm²], Dimensionsmaske m_d):

    Δ_d = r_i,d − r⁰_d   (für d = x, y mit Minimum Image: Δ_d −= L_d·rint(Δ_d / L_d))
    V   = ½ k Σ_d m_d Δ_d²                  [kJ/mol]
    F_d = −k m_d Δ_d                        [kJ/mol/nm]

Schwerpunkt-Pulling (Gruppe G, Gewichte w_i = m_i / M_G, ungewrappte Koordinaten
x_i + n_i·L), Zugrichtung ê (Einheitsvektor), Geschwindigkeit v [nm/ps], Zeit t [ps]:

    s   = ê·(R_G − R₀),  u = s − v t
    V   = ½ k u²
    F_i = −k u w_i ê
    Arbeit beim Protokollschritt t → t' bei festem x:  ΔW = V(x, t') − V(x, t)
    (exakte diskrete Form von W = ∫ ∂V/∂t dt = −∫ k u v dt).
"""

from __future__ import annotations

import numpy as np


def minimum_image_xy(delta, box):
    """Δ [nm] (..., 3) mit Minimum Image in x/y; box (..., 2) [nm]."""
    d = np.array(delta, dtype=np.float64, copy=True)
    big_l = np.asarray(box, np.float64)
    d[..., :2] -= big_l * np.rint(d[..., :2] / big_l)
    return d


def position_restraint(pos, ref, k, mask, box):
    """Energie [kJ/mol] (m,) und Kraft [kJ/mol/nm] (m, 3) je Restraint."""
    d = minimum_image_xy(np.asarray(pos, np.float64) - np.asarray(ref, np.float64), box)
    m = np.asarray(mask, np.float64)
    k = np.asarray(k, np.float64)[..., None]
    e = 0.5 * (k * m * d * d).sum(axis=-1)
    f = -k * m * d
    return e, f


def com_pulling(unwrapped_pos, weights, r0, direction, velocity, k, t):
    """Energie [kJ/mol], Kräfte [kJ/mol/nm] (n, 3) für eine Gruppe, plus u [nm]."""
    x = np.asarray(unwrapped_pos, np.float64)
    w = np.asarray(weights, np.float64)
    e_hat = np.asarray(direction, np.float64)
    r_g = (w[:, None] * x).sum(axis=0)
    u = e_hat @ (r_g - np.asarray(r0, np.float64)) - velocity * t
    energy = 0.5 * k * u * u
    forces = -k * u * w[:, None] * e_hat[None, :]
    return energy, forces, u


def com_pulling_work_increment(unwrapped_pos, weights, r0, direction, velocity, k, t_old, t_new):
    """ΔW = V(x, t_new) − V(x, t_old) [kJ/mol]."""
    e_new = com_pulling(unwrapped_pos, weights, r0, direction, velocity, k, t_new)[0]
    e_old = com_pulling(unwrapped_pos, weights, r0, direction, velocity, k, t_old)[0]
    return e_new - e_old
