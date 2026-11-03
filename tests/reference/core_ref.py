"""NumPy-Referenzen für die Kern-Kernels aus Phase 1 (fixed_point, wrap_xy).

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


def wrap_coordinate(x: np.ndarray, box_length: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Bitgenaue Nachbildung von ``wrap_coordinate`` in src/core/core_kernels.cu.

    Args:
        x: Koordinaten [nm], float32.
        box_length: Boxlänge [nm] > 0, float32, broadcastbar zu ``x``.

    Returns:
        (x_wrapped in [0, L) als float32, n als int32) mit x ≈ x_wrapped + n·L.
    """
    x = np.asarray(x, dtype=np.float32)
    big_l = np.asarray(box_length, dtype=np.float32)
    n = np.floor(x / big_l).astype(np.float32)  # IEEE-Division in float32 wie CUDA ohne fast-math
    # fmaf(-n, L, x): exaktes x − n·L (in float64 exakt darstellbar), einmal nach float32 gerundet.
    r = (x.astype(np.float64) - n.astype(np.float64) * big_l.astype(np.float64)).astype(np.float32)
    neg = r < 0
    r = np.where(neg, r + big_l, r).astype(np.float32)
    n = np.where(neg, n - np.float32(1), n)
    over = r >= big_l
    r = np.where(over, r - big_l, r).astype(np.float32)
    n = np.where(over, n + np.float32(1), n)
    return r, n.astype(np.int32)
