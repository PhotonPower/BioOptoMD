"""NumPy-Referenz (FP64) für Generalized Born OBC2 mit Salz (ADR 0010, AMBER igb = 5).

Mit ρ_i (intrinsischer Radius), δ = 0.009 nm, or_i = ρ_i − δ, sr_i = s_i · or_i, C = ONE_4PI_EPS0:

    I_i   = Σ_{j≠i} H(r_ij, or_i, sr_j)                       (alle Paare inkl. ausgeschlossener)
    H     = ½ [1/L − 1/U + ¼ (r − sr²/r)(1/U² − 1/L²) + ½ ln(L/U)/r],  falls r + sr > or, sonst 0
            U = r + sr, L = max(or, |r − sr|)
    B_i   = 1 / (1/or_i − tanh(ψ − 0.8 ψ² + 4.85 ψ³)/ρ_i),  ψ = I_i or_i
    E_self = −½ C (1/ε_in − e^{−κB}/ε_out) q² / B
    E_ACE  = 28.3919551 (ρ + 0.14)² (ρ/B)⁶                     (optional)
    E_GB   = −C (1/ε_in − e^{−κf}/ε_out) q_i q_j / f,  f = sqrt(r² + B_i B_j exp(−r²/(4 B_i B_j)))
    E_C    = C q_i q_j / (ε_in r)  (nur nicht ausgeschlossene Paare)
    E_LJ   = 4ε[(σ/r)¹² − (σ/r)⁶]  (nur nicht ausgeschlossene Paare)

Cutoff (optional): nicht ausgeschlossene Paare (E_LJ + E_C + E_GB)·S(r) für r < r_c, ausgeschlossene
Paare E_GB ohne Switching; Born-Integrale hart bei r_c. Periodisch (optional, ``n_max > 0``): Summe
über Bilder n = (n_x L_x, n_y L_y) der Rohabstände; Exclusions nur für das Minimum-Image-Bild;
Eigenbilder: I_i += H(|n|, or_i, sr_i), Energie ½ (E_C + E_GB + E_LJ)·S.

Kräfte analytisch: F = −∂E/∂x bei festen B (Paarterme) − Σ_k (∂E/∂B_k)(dB_k/dI_k) ∂I_k/∂x.
"""

from __future__ import annotations

import numpy as np
from nonbonded_ref import switch

ONE_4PI_EPS0 = 138.935456
OFFSET = 0.009  # nm
ACE_PREFACTOR = 28.3919551  # 4π γ, kJ/mol/nm²
ACE_PROBE = 0.14  # nm
OBC2 = (1.0, 0.8, 4.85)  # α, β, γ


def h_integral(r, o, s):
    """HCT-Integral H(r, or_i, sr_j) und dH/dr (vektorisiert)."""
    r = np.asarray(r, np.float64)
    u = r + s
    d = np.abs(r - s)
    use_d = d > o
    lo = np.where(use_d, d, o)
    dl = np.where(use_d, np.sign(r - s), 0.0)
    a = r - s * s / r
    da = 1.0 + s * s / (r * r)
    log_lu = np.log(lo / u)
    h = 0.5 * (1 / lo - 1 / u + 0.25 * a * (1 / u**2 - 1 / lo**2) + 0.5 * log_lu / r)
    dh = 0.5 * (
        -dl / lo**2
        + 1 / u**2
        + 0.25 * da * (1 / u**2 - 1 / lo**2)
        + 0.25 * a * (-2 / u**3 + 2 * dl / lo**3)
        + 0.5 * ((dl / lo - 1 / u) / r - log_lu / r**2)
    )
    active = u > o
    return np.where(active, h, 0.0), np.where(active, dh, 0.0)


def born_radii(integral, rho):
    """B_i und dB_i/dI_i (OBC2)."""
    a, b, g = OBC2
    o = rho - OFFSET
    psi = integral * o
    x = a * psi - b * psi**2 + g * psi**3
    th = np.tanh(x)
    born = 1.0 / (1.0 / o - th / rho)
    dborn = born**2 * (1 - th**2) * (a - 2 * b * psi + 3 * g * psi**2) * o / rho
    return born, dborn


def _image_pairs(pos, box, n_max, r_c):
    """Liste (i, j, d, n) aller Paare i ≤ j (i = j nur für n ≠ 0) und Bilder mit |d| < r_c."""
    n = len(pos)
    ii, jj = np.triu_indices(n, 0)
    out = []
    shifts = [(0, 0)] if box is None else [
        (nx, ny) for nx in range(-n_max, n_max + 1) for ny in range(-n_max, n_max + 1)
    ]  # fmt: skip
    for nx, ny in shifts:
        zero = nx == 0 and ny == 0
        keep = ii != jj if zero else np.ones(len(ii), bool)
        i, j = ii[keep], jj[keep]
        d = pos[i] - pos[j]
        if box is not None:
            d = d + np.array([nx * box[0], ny * box[1], 0.0])
        r2 = np.einsum("ij,ij->i", d, d)
        ok = r2 < (np.inf if r_c is None else r_c * r_c)
        out.append((i[ok], j[ok], d[ok], np.full(ok.sum(), nx), np.full(ok.sum(), ny)))
    cat = [np.concatenate(x) for x in zip(*out, strict=True)]
    return cat


def gb_nonbonded(
    pos, q, rho, scale, types, table, excluded, kappa, eps_out, eps_in=1.0, box=None, n_max=0,
    r_s=None, r_c=None, ace=True,
):  # fmt: skip
    """Energie [kJ/mol], Kräfte [kJ/mol/nm] (n, 3), Born-Radien [nm] und Energieterme.

    Args:
        pos (n, 3) [nm]; q (n,) [e]; rho, scale (n,) GB-Parameter; types/table: LJ wie
        nonbonded_ref; excluded (m, 2); kappa [1/nm]; eps_out, eps_in; box (L_x, L_y) oder None;
        n_max Bilder je Richtung; r_s, r_c [nm] oder None (kein Cutoff).
    """
    pos = np.asarray(pos, np.float64)
    q = np.asarray(q, np.float64)
    rho = np.asarray(rho, np.float64)
    o = rho - OFFSET
    sr = np.asarray(scale, np.float64) * o
    n = len(pos)
    tab = np.asarray(table, np.float64)
    excl = {(min(a, b), max(a, b)) for a, b in np.asarray(excluded, np.int64).reshape(-1, 2)}
    i, j, d, nx, ny = _image_pairs(pos, box, n_max, r_c)
    r = np.sqrt(np.einsum("ij,ij->i", d, d))
    self_img = i == j
    if box is not None:
        raw = pos[i, :2] - pos[j, :2]
        mi = -np.rint(raw / np.asarray(box))
        at_mi = (mi[:, 0] == nx) & (mi[:, 1] == ny)
    else:
        at_mi = np.ones(len(i), bool)
    is_ex = np.array([(a, b) in excl for a, b in zip(i, j, strict=True)], bool) & at_mi

    # Born-Integrale (beide Richtungen; Eigenbild: jede Verschiebung zählt einmal für i)
    h_ij, dh_ij = h_integral(r, o[i], sr[j])  # Beitrag zu I_i durch j
    h_ji, dh_ji = h_integral(r, o[j], sr[i])  # Beitrag zu I_j durch i
    integral = np.zeros(n)
    np.add.at(integral, i, h_ij)
    np.add.at(integral, j, np.where(self_img, 0.0, h_ji))
    if box is not None:  # Eigenbilder mit n und −n: _image_pairs liefert beide, i = j
        pass
    born, dborn = born_radii(integral, rho)

    c = ONE_4PI_EPS0
    ex_b = np.exp(-kappa * born)
    g_self = 1 / eps_in - ex_b / eps_out
    e_self = -0.5 * c * g_self * q**2 / born
    de_db = -0.5 * c * q**2 * (kappa * ex_b / eps_out / born - g_self / born**2)
    e_ace = np.zeros(n)
    if ace:
        e_ace = ACE_PREFACTOR * (rho + ACE_PROBE) ** 2 * (rho / born) ** 6
        de_db = de_db - 6.0 * e_ace / born

    # Paare bei festen B
    bi, bj = born[i], born[j]
    p = bi * bj
    ex = np.exp(-(r**2) / (4 * p))
    f = np.sqrt(r**2 + p * ex)
    qq = q[i] * q[j]
    ek = np.exp(-kappa * f)
    g = 1 / eps_in - ek / eps_out
    v_gb = -c * g * qq / f
    dv_gb_df = -c * qq * (kappa * ek / eps_out / f - g / f**2)
    df_dr = (2 * r - 0.5 * r * ex) / (2 * f)
    df_dbi = bj * ex * (1 + r**2 / (4 * p)) / (2 * f)
    df_dbj = bi * ex * (1 + r**2 / (4 * p)) / (2 * f)
    sig = tab[types[i], types[j], 0]
    eps = tab[types[i], types[j], 1]
    sr6 = np.where(eps > 0, (sig / r) ** 6, 0.0)
    v_lj = 4 * eps * (sr6 * sr6 - sr6)
    dv_lj = -4 * eps * (12 * sr6 * sr6 - 6 * sr6) / r
    v_c = c * qq / (eps_in * r)
    dv_c = -v_c / r
    nonex = ~is_ex
    v_ne = np.where(nonex, v_lj + v_c, 0.0) + v_gb
    dv_ne = np.where(nonex, dv_lj + dv_c, 0.0) + dv_gb_df * df_dr
    if r_c is not None:
        s, ds = switch(r, r_s, r_c)
        s = np.where(nonex, s, 1.0)
        ds = np.where(nonex, ds, 0.0)
    else:
        s, ds = np.ones_like(r), np.zeros_like(r)
    weight = np.where(self_img, 0.5, 1.0)
    v_pair = v_ne * s * weight
    dv_dr = (dv_ne * s + v_ne * ds) * weight
    de_db_pair_i = dv_gb_df * df_dbi * s * weight
    de_db_pair_j = dv_gb_df * df_dbj * s * weight
    np.add.at(de_db, i, de_db_pair_i)
    np.add.at(de_db, j, de_db_pair_j)

    forces = np.zeros_like(pos)
    unit = d / r[:, None]
    fpair = np.where(self_img, 0.0, -dv_dr)[:, None] * unit
    np.add.at(forces, i, fpair)
    np.add.at(forces, j, -fpair)
    # Kettenregel über die Born-Radien
    de_di = de_db * dborn
    fchain = -(de_di[i] * dh_ij + np.where(self_img, 0.0, de_di[j] * dh_ji))[:, None] * unit
    fchain = np.where(self_img[:, None], 0.0, fchain)
    np.add.at(forces, i, fchain)
    np.add.at(forces, j, -fchain)
    energy = e_self.sum() + e_ace.sum() + v_pair.sum()
    terms = {
        "self": e_self.sum(), "ace": e_ace.sum(), "pair": v_pair.sum(),
        "integral": integral,
    }  # fmt: skip
    return energy, forces, born, terms
