"""NumPy-Referenz (FP64, O(N²)) für die nicht-kovalenten Terme, docs/PHYSICS_MODEL.md §3.

Für alle Paare i < j, die nicht ausgeschlossen sind (1-2, 1-3, 1-4), mit Minimum Image in x/y:

    V_LJ = 4 ε_ij [(σ_ij/r)^12 − (σ_ij/r)^6]       (σ_ij, ε_ij aus der Typ-Paar-Tabelle)
    V_DH = C q_i q_j exp(−κ r) / (ε_r r),          C = ONE_4PI_EPS0
    V    = (V_LJ + V_DH) · S(r)                    für r < r_c, sonst 0
    S(r) = 1                                       für r ≤ r_s
         = 1 − 6x⁵ + 15x⁴ − 10x³,  x = (r − r_s)/(r_c − r_s)   für r_s < r < r_c
    F_i  = −dV/dr · d/r,  dV/dr = (V_LJ' + V_DH') S + (V_LJ + V_DH) S'

(Switching wie OpenMM CustomNonbondedForce.useSwitchingFunction.) Paare mit r² < EPS_R2 tragen nicht bei.
"""

from __future__ import annotations

import numpy as np

ONE_4PI_EPS0 = 138.935456
EPS_R2 = 1.0e-8


def switch(r, r_s, r_c):
    """S(r) und dS/dr."""
    x = np.clip((r - r_s) / (r_c - r_s), 0.0, 1.0)
    s = 1.0 - 6.0 * x**5 + 15.0 * x**4 - 10.0 * x**3
    ds = (-30.0 * x**4 + 60.0 * x**3 - 30.0 * x**2) / (r_c - r_s)
    inside = (r > r_s) & (r < r_c)
    return np.where(r <= r_s, 1.0, np.where(inside, s, 0.0)), np.where(inside, ds, 0.0)


def nonbonded(pos, charges, types, table, excluded, box, kappa, eps_r, r_s=1.8, r_c=2.0):
    """Energie [kJ/mol] und Kräfte [kJ/mol/nm] (n, 3).

    Args:
        pos: (n, 3) [nm]; charges: (n,) [e]; types: (n,) LJ-Typindex; table: (T, T, 2) (σ, ε).
        excluded: Paare (n_excl, 2) lokaler Indizes; box: (Lx, Ly) [nm]; kappa [1/nm]; eps_r.
    """
    pos = np.asarray(pos, np.float64)
    q = np.asarray(charges, np.float64)
    n = len(pos)
    i, j = np.triu_indices(n, 1)
    keep = np.ones(len(i), bool)
    if len(excluded):
        ex = np.asarray(excluded, np.int64)
        a, b = np.minimum(ex[:, 0], ex[:, 1]), np.maximum(ex[:, 0], ex[:, 1])
        code = i.astype(np.int64) * n + j
        keep &= ~np.isin(code, a * n + b)
    i, j = i[keep], j[keep]
    d = pos[i] - pos[j]
    big_l = np.asarray(box, np.float64)
    d[:, :2] -= big_l * np.rint(d[:, :2] / big_l)
    r2 = np.einsum("ij,ij->i", d, d)
    ok = (r2 >= EPS_R2) & (r2 < r_c * r_c)
    i, j, d, r2 = i[ok], j[ok], d[ok], r2[ok]
    r = np.sqrt(r2)
    tab = np.asarray(table, np.float64)
    sig = tab[types[i], types[j], 0]
    eps = tab[types[i], types[j], 1]
    sr6 = (sig / r) ** 6
    v_lj = 4.0 * eps * (sr6 * sr6 - sr6)
    dv_lj = -4.0 * eps * (12.0 * sr6 * sr6 - 6.0 * sr6) / r
    ex = np.exp(-kappa * r)
    v_dh = ONE_4PI_EPS0 * q[i] * q[j] * ex / (eps_r * r)
    dv_dh = -v_dh * (kappa + 1.0 / r)
    s, ds = switch(r, r_s, r_c)
    v = (v_lj + v_dh) * s
    dv = (dv_lj + dv_dh) * s + (v_lj + v_dh) * ds
    f = (-dv / r)[:, None] * d
    forces = np.zeros_like(pos)
    np.add.at(forces, i, f)
    np.add.at(forces, j, -f)
    return v.sum(), forces
