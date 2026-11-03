"""NumPy-Referenz (FP64) für die kovalenten Terme, docs/PHYSICS_MODEL.md §2 (OpenMM-Konventionen).

Alle Abstandsvektoren mit Minimum Image in x/y (ADR 0003). Einheiten: nm, rad, kJ/mol, e.

Bond (i, j):            V = ½ k (r − r0)²,           d = r_i − r_j,  F_i = −k (r − r0) d/r,  F_j = −F_i
Angle (i, j, k):        a = r_i − r_j, b = r_k − r_j, n = a × b,
                        θ = atan2(|n|, a·b),  V = ½ k (θ − θ0)²,  V' = k (θ − θ0)
                        F_i = −V' (a × n)/(|a|² |n|),  F_k = −V' (n × b)/(|b|² |n|),  F_j = −F_i − F_k
Torsion (i, j, k, l):   b1 = r_j − r_i, b2 = r_k − r_j, b3 = r_l − r_k, m = b1 × b2, n = b2 × b3
                        φ = atan2(|b2| b1·n, m·n)  (IUPAC, trans = π),  V = k [1 + cos(nφ − φ0)]
                        V' = −k n sin(nφ − φ0)
                        F_i = −V' (−|b2|/|m|²) m,      F_l = −V' (|b2|/|n|²) n
                        F_j = −(1 + c_m) F_i + c_n F_l,  F_k = −(1 + c_n) F_l + c_m F_i
                        mit c_m = (b1·b2)/|b2|²,  c_n = (b3·b2)/|b2|²        (Blondel & Karplus 1996)
1-4-Paar (i, j):        V = 4ε[(σ/r)^12 − (σ/r)^6] + C qq exp(−κ r)/(ε_r r),  C = ONE_4PI_EPS0
                        (σ, ε, qq bereits skaliert, z. B. AMBER s_LJ = ½, s_el = 1/1.2)

Grenzfälle (AGENTS.md Regel 4): r² < EPS_R2 ⇒ Kraft 0 (Bond, Paar); |n|² < EPS_CROSS (kollinearer
Winkel) bzw. |m|² oder |n|² < EPS_CROSS (Torsion) ⇒ Kraft 0, da die Ableitung dort nicht definiert ist.
"""

from __future__ import annotations

import numpy as np

ONE_4PI_EPS0 = 138.935456  # identisch zu biooptomd.units
EPS_R2 = 1.0e-8  # nm²
EPS_CROSS = 1.0e-12  # nm⁴, Untergrenze für |a × b|² (= biooptomd.units.EPS_CROSS2)


def _min_image(d, box):
    d = np.array(d, dtype=np.float64, copy=True)
    if box is not None:
        big_l = np.asarray(box, np.float64)
        d[..., :2] -= big_l * np.rint(d[..., :2] / big_l)
    return d


def _dot(a, b):
    return np.einsum("...i,...i->...", a, b)


def bonds(pos, idx, k, r0, box=None):
    """Energie [kJ/mol] und Kräfte [kJ/mol/nm] (n_atoms, 3)."""
    pos = np.asarray(pos, np.float64)
    idx = np.asarray(idx)
    d = _min_image(pos[idx[:, 0]] - pos[idx[:, 1]], box)
    r2 = _dot(d, d)
    ok = r2 >= EPS_R2
    r = np.sqrt(np.where(ok, r2, 1.0))
    dr = np.where(ok, r - r0, 0.0)
    energy = 0.5 * np.asarray(k) * (np.sqrt(r2) - r0) ** 2
    fi = (-np.asarray(k) * dr / r)[:, None] * d
    forces = np.zeros_like(pos)
    np.add.at(forces, idx[:, 0], fi)
    np.add.at(forces, idx[:, 1], -fi)
    return energy.sum(), forces


def angle_values(pos, idx, box=None):
    pos = np.asarray(pos, np.float64)
    idx = np.asarray(idx)
    a = _min_image(pos[idx[:, 0]] - pos[idx[:, 1]], box)
    b = _min_image(pos[idx[:, 2]] - pos[idx[:, 1]], box)
    n = np.cross(a, b)
    return np.arctan2(np.sqrt(_dot(n, n)), _dot(a, b)), a, b, n


def angles(pos, idx, k, theta0, box=None):
    pos = np.asarray(pos, np.float64)
    idx = np.asarray(idx)
    theta, a, b, n = angle_values(pos, idx, box)
    dv = np.asarray(k) * (theta - theta0)
    energy = 0.5 * np.asarray(k) * (theta - theta0) ** 2
    n2 = _dot(n, n)
    ok = n2 >= EPS_CROSS
    nn = np.sqrt(np.where(ok, n2, 1.0))
    fi = np.where(ok, -dv / (_dot(a, a) * nn), 0.0)[:, None] * np.cross(a, n)
    fk = np.where(ok, -dv / (_dot(b, b) * nn), 0.0)[:, None] * np.cross(n, b)
    forces = np.zeros_like(pos)
    np.add.at(forces, idx[:, 0], fi)
    np.add.at(forces, idx[:, 2], fk)
    np.add.at(forces, idx[:, 1], -fi - fk)
    return energy.sum(), forces


def torsion_values(pos, idx, box=None):
    pos = np.asarray(pos, np.float64)
    idx = np.asarray(idx)
    b1 = _min_image(pos[idx[:, 1]] - pos[idx[:, 0]], box)
    b2 = _min_image(pos[idx[:, 2]] - pos[idx[:, 1]], box)
    b3 = _min_image(pos[idx[:, 3]] - pos[idx[:, 2]], box)
    m = np.cross(b1, b2)
    n = np.cross(b2, b3)
    b2n = np.sqrt(_dot(b2, b2))
    phi = np.arctan2(b2n * _dot(b1, n), _dot(m, n))
    return phi, b1, b2, b3, m, n


def torsions(pos, idx, k, periodicity, phase, box=None):
    pos = np.asarray(pos, np.float64)
    idx = np.asarray(idx)
    phi, b1, b2, b3, m, n = torsion_values(pos, idx, box)
    k = np.asarray(k, np.float64)
    per = np.asarray(periodicity, np.float64)
    arg = per * phi - np.asarray(phase, np.float64)
    energy = k * (1.0 + np.cos(arg))
    dv = -k * per * np.sin(arg)
    m2, n2, b22 = _dot(m, m), _dot(n, n), _dot(b2, b2)
    ok = (m2 >= EPS_CROSS) & (n2 >= EPS_CROSS) & (b22 >= EPS_R2)
    b2n = np.sqrt(np.where(ok, b22, 1.0))
    fi = np.where(ok, dv * b2n / np.where(ok, m2, 1.0), 0.0)[:, None] * m
    fl = np.where(ok, -dv * b2n / np.where(ok, n2, 1.0), 0.0)[:, None] * n
    cm = np.where(ok, _dot(b1, b2) / np.where(ok, b22, 1.0), 0.0)[:, None]
    cn = np.where(ok, _dot(b3, b2) / np.where(ok, b22, 1.0), 0.0)[:, None]
    fj = -(1.0 + cm) * fi + cn * fl
    fk = -(1.0 + cn) * fl + cm * fi
    forces = np.zeros_like(pos)
    for col, f in ((0, fi), (1, fj), (2, fk), (3, fl)):
        np.add.at(forces, idx[:, col], f)
    return energy.sum(), forces


def pairs14(pos, idx, qq, sigma, eps, kappa, eps_r, box=None):
    pos = np.asarray(pos, np.float64)
    idx = np.asarray(idx)
    d = _min_image(pos[idx[:, 0]] - pos[idx[:, 1]], box)
    r2 = _dot(d, d)
    ok = r2 >= EPS_R2
    r = np.sqrt(np.where(ok, r2, 1.0))
    sigma, eps, qq = (np.asarray(x, np.float64) for x in (sigma, eps, qq))
    sr6 = (sigma / r) ** 6
    e_lj = 4.0 * eps * (sr6 * sr6 - sr6)
    f_lj = 4.0 * eps * (12.0 * sr6 * sr6 - 6.0 * sr6) / r  # −dV/dr
    ex = np.exp(-kappa * r)
    e_dh = ONE_4PI_EPS0 * qq * ex / (eps_r * r)
    f_dh = ONE_4PI_EPS0 * qq * ex * (kappa * r + 1.0) / (eps_r * r * r)  # −dV/dr
    energy = np.where(ok, e_lj + e_dh, 0.0)
    fi = (np.where(ok, (f_lj + f_dh) / r, 0.0))[:, None] * d
    forces = np.zeros_like(pos)
    np.add.at(forces, idx[:, 0], fi)
    np.add.at(forces, idx[:, 1], -fi)
    return energy.sum(), forces


def term(name, pos, topo, kappa=1.0, eps_r=78.5, box=None):
    """Energie und Kräfte eines Term-Typs ("bond", "angle", "torsion", "pair14") einer Topologie
    (dict wie in openmm_bonded.extract_topology)."""
    t = topo
    if name == "bond":
        return bonds(pos, t["bonds"], t["bond_k"], t["bond_r0"], box)
    if name == "angle":
        return angles(pos, t["angles"], t["angle_k"], t["angle_theta0"], box)
    if name == "torsion":
        return torsions(pos, t["torsions"], t["torsion_k"], t["torsion_n"], t["torsion_phase"], box)
    if name == "pair14":
        return pairs14(
            pos, t["pairs14"], t["pair_qq"], t["pair_sigma"], t["pair_eps"], kappa, eps_r, box
        )
    raise ValueError(name)


def total(pos, topo, kappa, eps_r, box=None):
    """Energie und Kräfte aller vier Term-Typen als dict."""
    return {
        name: term(name, pos, topo, kappa, eps_r, box)
        for name in ("bond", "angle", "torsion", "pair14")
    }


def finite_difference_forces(energy_fn, pos, h=1e-4):
    """Zentrale finite Differenzen −∂E/∂x in FP64 (TESTING.md: h = 1e-4 nm)."""
    pos = np.asarray(pos, np.float64)
    f = np.zeros_like(pos)
    for a in range(pos.shape[0]):
        for d in range(3):
            p = pos.copy()
            p[a, d] += h
            e_plus = energy_fn(p)
            p[a, d] -= 2 * h
            e_minus = energy_fn(p)
            f[a, d] = -(e_plus - e_minus) / (2 * h)
    return f
