"""NumPy-Referenzen für die Kern-Kernels (fixed_point, wrap_xy in FP64 nach ADR 0007).

Die Referenzen bilden die Gerätearithmetik bitgenau nach, damit Tests exakt (nicht mit
Toleranz) vergleichen können. Einheiten: nm, kJ/mol/nm.
"""

from __future__ import annotations

import numpy as np

FIXED_POINT_SCALE = 2.0**32  # identisch zu biooptomd.units.FIXED_POINT_SCALE


def to_fixed_point(values: np.ndarray) -> np.ndarray:
    """float32 → int64-Fixed-Point wie ``__float2ll_rn(v * 2^32)``.

    v·2^32 ist in float64 exakt; ``np.rint`` rundet wie ``_rn`` zur nächsten geraden Zahl.
    Gültig für |v| < 2^31.
    """
    v = np.asarray(values, dtype=np.float32).astype(np.float64)
    return np.rint(v * FIXED_POINT_SCALE).astype(np.int64)


def accumulate_fixed_point(n_total: int, atom_index: np.ndarray, forces: np.ndarray) -> np.ndarray:
    """Exakte Fixed-Point-Summe der Beiträge, Shape (3, n_total) int64 (SoA fx | fy | fz).

    Args:
        n_total: Anzahl Atomplätze n_replicas·n_pad.
        atom_index: globale Atomindizes, int (m,).
        forces: Kräfte [kJ/mol/nm], float32 (m, 3).
    """
    acc = np.zeros((3, n_total), dtype=np.int64)
    fp = to_fixed_point(forces)
    for d in range(3):
        np.add.at(acc[d], atom_index, fp[:, d])
    return acc


def wrap_coordinate_mixed(hi, lo, box_length):
    """Bitgenaue Nachbildung des FP64-Wraps (ADR 0007) in src/core/core_kernels.cu.

    Args:
        hi, lo: FP32-Position und FP32-Korrektur (x = hi + lo in FP64) [nm].
        box_length: Boxlänge [nm] > 0 (float32), broadcastbar.

    Returns:
        (hi', lo', n): neue FP32-Teile mit hi' + lo' ∈ [0, L) und x = hi' + lo' + n·L exakt.
    """
    x = np.asarray(hi, np.float32).astype(np.float64) + np.asarray(lo, np.float32).astype(
        np.float64
    )
    big_l = np.asarray(box_length, np.float32).astype(np.float64)
    n = np.floor(x / big_l)
    r = x - n * big_l  # in FP64 exakt (≤ 48 signifikante Bit), entspricht fma(−n, L, x)
    neg = r < 0
    r = np.where(neg, r + big_l, r)
    n = np.where(neg, n - 1, n)
    over = r >= big_l
    r = np.where(over, r - big_l, r)
    n = np.where(over, n + 1, n)
    new_hi = r.astype(np.float32)
    new_lo = (r - new_hi.astype(np.float64)).astype(np.float32)
    return new_hi, new_lo, n.astype(np.int32)
