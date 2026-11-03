"""NumPy-Referenzen (FP64) für die Optik-Observer und VoxelGridForce3D (docs/PHYSICS_MODEL.md §4.3, §6).

Dipol-Sonde (Gruppe G mit Massen m_i, Achsenatome A, B), ungewrappte Positionen x_i (x/y + n·L):
    R_F = Σ m_i x_i / Σ m_i,   μ̂ = (r_B − r_A)/|r_B − r_A|  (Minimum Image in x/y),   d = z(R_F) − z_s
FRET: R = R_A − R_D (Minimum Image in x/y), R̂ = R/|R|,
    κ² = (μ̂_D·μ̂_A − 3(μ̂_D·R̂)(μ̂_A·R̂))²,   E = 1 / (1 + (|R|/R0)^6 · (2/3)/κ²)
Brechungsindex (Lorentz-Lorenz mit Wasser-Hintergrund), Schicht k = floor((z − z_min)/Δz), V_k = L_x L_y Δz:
    f_w = max(0, 1 − S_V/V_k),  L = (4π/3) S_α/V_k + f_w (n_w² − 1)/(n_w² + 2),  n = sqrt((1 + 2L)/(1 − L))
Voxel: Gitterpunkte x0 + i·Δ; Werte (V, ∂V/∂x, ∂V/∂y, ∂V/∂z) trilinear interpoliert;
    E_i = s_t V(r_i),  F_i = −s_t ∇V(r_i)  (interpolierter Gradient, nicht Gradient der Interpolation)
"""

from __future__ import annotations

import numpy as np


def _mi(d, box):
    d = np.array(d, np.float64, copy=True)
    d[..., :2] -= np.asarray(box, np.float64) * np.rint(d[..., :2] / np.asarray(box, np.float64))
    return d


def dipole_frame(unwrapped, masses, group, atom_a, atom_b, box, z_surface=0.0):
    """(R_F (3), μ̂ (3), d) für eine Replica."""
    x = np.asarray(unwrapped, np.float64)
    m = np.asarray(masses, np.float64)[group]
    r_f = (m[:, None] * x[group]).sum(axis=0) / m.sum()
    axis = _mi(x[atom_b] - x[atom_a], box)
    mu = axis / np.linalg.norm(axis)
    return np.concatenate([r_f, mu, [r_f[2] - z_surface]])


def kappa2(mu_d, mu_a, r_vec):
    r_hat = np.asarray(r_vec, np.float64) / np.linalg.norm(r_vec, axis=-1, keepdims=True)
    a = np.einsum("...i,...i->...", mu_d, mu_a)
    b = np.einsum("...i,...i->...", mu_d, r_hat)
    c = np.einsum("...i,...i->...", mu_a, r_hat)
    return (a - 3.0 * b * c) ** 2


def fret_efficiency(r, k2, r0):
    return 1.0 / (1.0 + (np.asarray(r) / r0) ** 6 * (2.0 / 3.0) / np.asarray(k2))


def lorentz_lorenz(s_alpha, s_volume, layer_volume, n_w):
    """n je Schicht aus S_α [nm³], S_V [nm³] und V_k [nm³]."""
    f_w = np.maximum(0.0, 1.0 - np.asarray(s_volume, np.float64) / layer_volume)
    lw = (n_w**2 - 1.0) / (n_w**2 + 2.0)
    big_l = 4.0 * np.pi / 3.0 * np.asarray(s_alpha, np.float64) / layer_volume + f_w * lw
    return np.sqrt((1.0 + 2.0 * big_l) / (1.0 - big_l))


def layer_histograms(z, alpha, volume, z_min, dz, n_bins):
    """S_α(k), S_V(k) mit Atomen als Punkte (ohne Verschmierung)."""
    k = np.floor((np.asarray(z, np.float64) - z_min) / dz).astype(np.int64)
    ok = (k >= 0) & (k < n_bins)
    s_a = np.bincount(k[ok], weights=np.asarray(alpha, np.float64)[ok], minlength=n_bins)
    s_v = np.bincount(k[ok], weights=np.asarray(volume, np.float64)[ok], minlength=n_bins)
    return s_a, s_v


def trilinear(values, origin, spacing, point, periodic_xy=False):
    """Trilineare Interpolation von values (nz, ny, nx, 4) an point (3,); None außerhalb."""
    nz, ny, nx, _ = values.shape
    u = (np.asarray(point, np.float64) - np.asarray(origin, np.float64)) / np.asarray(
        spacing, np.float64
    )
    i0 = np.floor(u).astype(int)
    f = u - i0
    dims = (nx, ny, nz)
    idx = []
    for d in range(3):
        lo, hi = i0[d], i0[d] + 1
        if periodic_xy and d < 2:
            lo, hi = lo % dims[d], hi % dims[d]
        elif lo < 0 or hi > dims[d] - 1:
            return None
        idx.append((lo, hi))
    out = np.zeros(4)
    for cz, wz in ((0, 1 - f[2]), (1, f[2])):
        for cy, wy in ((0, 1 - f[1]), (1, f[1])):
            for cx, wx in ((0, 1 - f[0]), (1, f[0])):
                out += wx * wy * wz * values[idx[2][cz], idx[1][cy], idx[0][cx]]
    return out
