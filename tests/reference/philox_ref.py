"""NumPy-Referenz für Philox4x32-10 (Salmon et al., SC'11; Random123).

Bitgenaue Nachbildung von src/core/philox.cuh. Zählerbelegung im Integrator (AGENTS.md Regel 5):
counter = (i_local, replica, step_lo, step_hi | purpose << 24), key = (seed_lo, seed_hi).
"""

from __future__ import annotations

import numpy as np

M0 = np.uint64(0xD2511F53)
M1 = np.uint64(0xCD9E8D57)
W0 = np.uint64(0x9E3779B9)
W1 = np.uint64(0xBB67AE85)
MASK32 = np.uint64(0xFFFFFFFF)

# Known-Answer-Tests aus Random123 (kat_vectors): (counter, key) → Ausgabe.
KAT = [
    ((0, 0, 0, 0), (0, 0), (0x6627E8D5, 0xE169C58D, 0xBC57AC4C, 0x9B00DBD8)),
    (
        (0xFFFFFFFF,) * 4,
        (0xFFFFFFFF, 0xFFFFFFFF),
        (0x408F276D, 0x41C83B0E, 0xA20BC7C6, 0x6D5451FD),
    ),
    (
        (0x243F6A88, 0x85A308D3, 0x13198A2E, 0x03707344),
        (0xA4093822, 0x299F31D0),
        (0xD16CFE09, 0x94FDCCEB, 0x5001E420, 0x24126EA1),
    ),
]


def philox4x32_10(counter: np.ndarray, key: np.ndarray) -> np.ndarray:
    """Philox4x32-10, vektorisiert.

    Args:
        counter: uint32 (..., 4)
        key: uint32 (..., 2), broadcastbar zu counter[..., :2]

    Returns:
        uint32 (..., 4)
    """
    c = np.asarray(counter, dtype=np.uint32).astype(np.uint64)
    k = np.broadcast_to(np.asarray(key, dtype=np.uint32), c.shape[:-1] + (2,)).copy()
    c0, c1, c2, c3 = (c[..., j] for j in range(4))
    k0 = k[..., 0].astype(np.uint64)
    k1 = k[..., 1].astype(np.uint64)
    for rnd in range(10):
        if rnd > 0:
            k0 = (k0 + W0) & MASK32  # Weyl-Folge mod 2^32
            k1 = (k1 + W1) & MASK32
        p0 = M0 * c0
        p1 = M1 * c2
        hi0, lo0 = p0 >> np.uint64(32), p0 & MASK32
        hi1, lo1 = p1 >> np.uint64(32), p1 & MASK32
        c0 = hi1 ^ c1 ^ k0
        c1 = lo1
        c2 = hi0 ^ c3 ^ k1
        c3 = lo0
    return np.stack([c0, c1, c2, c3], axis=-1).astype(np.uint32)


def langevin_counter(i_local, replica, step, purpose=0):
    """Zähler (i_local, replica, step_lo, step_hi | purpose << 24) als uint32 (..., 4)."""
    i_local, replica, step = np.broadcast_arrays(
        np.asarray(i_local, np.uint64), np.asarray(replica, np.uint64), np.asarray(step, np.uint64)
    )
    hi = ((step >> np.uint64(32)) & np.uint64(0x00FFFFFF)) | (np.uint64(purpose) << np.uint64(24))
    return np.stack([i_local, replica, step & MASK32, hi], axis=-1).astype(np.uint32)


def seed_key(seed: int) -> np.ndarray:
    """64-bit-Seed → Philox-Schlüssel (seed_lo, seed_hi)."""
    return np.array([seed & 0xFFFFFFFF, (seed >> 32) & 0xFFFFFFFF], dtype=np.uint32)
