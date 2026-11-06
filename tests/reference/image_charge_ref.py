"""NumPy-Referenz (FP64, O(N² · Bilder)) für ImageChargeForce, ADR 0013.

Abgeschirmte Bildladungen an einer leitfähigen Ebene z = z_im (Dirichlet-Rand des linearisierten
Poisson-Boltzmann-Problems im Halbraum z > z_im):

    G(s)   = C exp(−κ s) / (ε_r s),   C = ONE_4PI_EPS0,   φ(s) = G(s) S(s)  (Switching wie nonbonded_ref)
    ζ_i    = max(z_i − z_im, ζ_c)                       (Clamping, ADR 0013 Punkt 6)
    s_ij,n = |(Δx_ij + n_x L_x, Δy_ij + n_y L_y, ζ_i + ζ_j)|   (Δ mit Minimum Image)
    U      = −f [ Σ_{i<j} Σ_n q_i q_j φ(s_ij,n) + ½ Σ_i Σ_n q_i² φ(s_ii,n) ]

Alle Paare zählen (keine Exclusions), das Eigenbild (i = i, n = 0) eingeschlossen. Für L >= 2 r_c
genügt n = 0 (Minimum Image). Kräfte F = −∂U/∂r analytisch; ∂ζ_i/∂z_i = 0 für geclampte Atome.
"""

from __future__ import annotations

import numpy as np
from nonbonded_ref import ONE_4PI_EPS0, switch


def green(s, kappa, eps_r):
    """G(s) und dG/ds."""
    g = ONE_4PI_EPS0 * np.exp(-kappa * s) / (eps_r * s)
    return g, -g * (kappa + 1.0 / s)


def image_charges(pos, charges, box, z_image, factor, kappa, eps_r, r_s, r_c, zeta_clamp=0.1,
                  n_images=None, switching=True):  # fmt: skip
    """Energie [kJ/mol] und Kräfte [kJ/mol/nm] (n, 3).

    Args:
        pos: (n, 3) [nm]; charges: (n,) [e]; box: (L_x, L_y) [nm]; z_image: z_im [nm];
        factor: Bildfaktor f; kappa [1/nm]; eps_r; r_s, r_c [nm]; zeta_clamp: ζ_c [nm].
        n_images: (n_x, n_y) maximale Gitterverschiebung; Default ⌈r_c/L + ½⌉, 0 für L >= 2 r_c.
        switching: False ⇒ S = 1 ohne Cutoff (für den Dirichlet-Test).
    """
    pos = np.asarray(pos, np.float64)
    q = np.asarray(charges, np.float64)
    lx, ly = float(box[0]), float(box[1])
    if n_images is None:
        n_images = tuple(0 if L >= 2 * r_c else int(np.ceil(r_c / L + 0.5)) for L in (lx, ly))
    zeta_raw = pos[:, 2] - z_image
    free = zeta_raw > zeta_clamp
    zeta = np.where(free, zeta_raw, zeta_clamp)
    n = len(q)
    energy = 0.0
    forces = np.zeros((n, 3))
    for i in range(n):
        for j in range(i, n):
            qq = q[i] * q[j]
            if qq == 0.0:
                continue
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dx -= lx * np.round(dx / lx)
            dy -= ly * np.round(dy / ly)
            sz = zeta[i] + zeta[j]
            for nx in range(-n_images[0], n_images[0] + 1):
                for ny in range(-n_images[1], n_images[1] + 1):
                    sx, sy = dx + nx * lx, dy + ny * ly
                    s = np.sqrt(sx * sx + sy * sy + sz * sz)
                    if switching and s >= r_c:
                        continue
                    g, dg = green(s, kappa, eps_r)
                    if switching:
                        sw, dsw = switch(np.array(s), r_s, r_c)
                        phi, dphi = g * float(sw), dg * float(sw) + g * float(dsw)
                    else:
                        phi, dphi = g, dg
                    weight = 0.5 if i == j else 1.0
                    energy -= factor * weight * qq * phi
                    c = -factor * qq * dphi / s  # ∂U/∂s · (1/s) für das ganze Paar
                    if i == j:  # nur z: x/y-Abstand ist konstant; ½·(c s_z + c s_z)
                        if free[i]:
                            forces[i, 2] -= c * sz
                        continue
                    forces[i, 0] -= c * sx
                    forces[i, 1] -= c * sy
                    forces[j, 0] += c * sx
                    forces[j, 1] += c * sy
                    if free[i]:
                        forces[i, 2] -= c * sz
                    if free[j]:
                        forces[j, 2] -= c * sz
    return energy, forces


def potential(points, pos, charges, z_image, factor, kappa, eps_r):
    """Potential [kJ/(mol·e)] von Ladungen + Bildern (ohne Switching, nur n = 0) an points (m, 3):
    Φ(p) = Σ_j q_j [G(|p − r_j|) − f G(|p − r̄_j|)], r̄_j = Spiegelbild an z_im."""
    pos = np.asarray(pos, np.float64)
    mirror = pos.copy()
    mirror[:, 2] = 2.0 * z_image - pos[:, 2]
    d = np.linalg.norm(points[:, None] - pos[None], axis=-1)
    dm = np.linalg.norm(points[:, None] - mirror[None], axis=-1)
    g = green(d, kappa, eps_r)[0]
    gm = green(dm, kappa, eps_r)[0]
    return (charges[None] * (g - factor * gm)).sum(axis=1)
