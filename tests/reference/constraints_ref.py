"""NumPy-Referenz (FP64) für SHAKE/RATTLE in BAOAB (docs/PHYSICS_MODEL.md §1, §5).

Cluster: Schweratom c und H-Atome h_k mit Sollabständen d_k, w = 1/m (inverse Massen).

SHAKE (nach einem A-Schritt x' = x + h v, h = Δt/2), Referenzvektoren s_k = x_hk − x_c VOR dem A-Schritt:
    wiederhole bis max_k |r_k² − d_k²| / d_k² < 2·tol:
        für jedes k:  r_k = x_hk − x_c,  g = (r_k² − d_k²) / (2 (s_k·r_k)(w_c + w_hk))
                      x_hk −= g w_hk s_k,   x_c += g w_c s_k
    Geschwindigkeitskorrektur: v += (x_constrained − x') / h
RATTLE (Geschwindigkeiten), r_k = x_hk − x_c der aktuellen (constrainten) Positionen:
    wiederhole bis max_k |r_k·(v_hk − v_c)| < tol_v · r_k²:
        für jedes k:  κ = r_k·(v_hk − v_c) / (r_k² (w_c + w_hk)),  v_hk −= κ w_hk r_k,  v_c += κ w_c r_k

Alle Abstandsvektoren mit Minimum Image in x/y. Ein Schritt mit γ = 0 (kein Rauschen):
    B, RATTLE, A+SHAKE, (O: v unverändert), RATTLE, A+SHAKE, [Kräfte], B, RATTLE.
"""

from __future__ import annotations

import numpy as np


def _mi(d, box):
    d = np.array(d, np.float64, copy=True)
    d[:2] -= np.asarray(box, np.float64) * np.rint(d[:2] / np.asarray(box, np.float64))
    return d


def shake(x, x_ref, inv_m, clusters, distances, box, tol=1e-6, max_iter=1000):
    """Positions-Constraints in place auf x (n, 3); Referenzrichtungen aus x_ref."""
    for cl, dist in zip(clusters, distances, strict=True):
        c = cl[0]
        hs = [(h, d) for h, d in zip(cl[1:], dist, strict=True) if h >= 0]
        s = {h: _mi(x_ref[h] - x_ref[c], box) for h, _ in hs}
        for _ in range(max_iter):
            done = True
            for h, d in hs:
                r = _mi(x[h] - x[c], box)
                diff = r @ r - d * d
                if abs(diff) > 2.0 * tol * d * d:
                    done = False
                    g = diff / (2.0 * (s[h] @ r) * (inv_m[c] + inv_m[h]))
                    x[h] -= g * inv_m[h] * s[h]
                    x[c] += g * inv_m[c] * s[h]
            if done:
                break
        else:
            raise RuntimeError("SHAKE konvergiert nicht")


def rattle(x, v, inv_m, clusters, box, tol_v=1e-10, max_iter=1000):
    """Geschwindigkeits-Constraints in place auf v."""
    for cl in clusters:
        c = cl[0]
        hs = [h for h in cl[1:] if h >= 0]
        r = {h: _mi(x[h] - x[c], box) for h in hs}
        for _ in range(max_iter):
            done = True
            for h in hs:
                rv = r[h] @ (v[h] - v[c])
                if abs(rv) > tol_v * (r[h] @ r[h]):
                    done = False
                    kappa = rv / ((r[h] @ r[h]) * (inv_m[c] + inv_m[h]))
                    v[h] -= kappa * inv_m[h] * r[h]
                    v[c] += kappa * inv_m[c] * r[h]
            if done:
                break
        else:
            raise RuntimeError("RATTLE konvergiert nicht")


def constrained_positions_after_step(
    x, v, f, inv_m, clusters, distances, box, dt, tol=1e-6, tol_v=1e-10
):
    """Positionen und Geschwindigkeiten nach B, RATTLE, A+SHAKE, RATTLE, A+SHAKE (γ = 0).

    Die Positionen am Schrittende hängen nicht von der Kraft am neuen Ort ab (letztes B fehlt).
    Mit kleinen Toleranzen ist das Ergebnis die (eindeutige) Lösung der Constraint-Gleichungen,
    unabhängig vom Iterationsverfahren.
    """
    x = np.array(x, np.float64, copy=True)
    v = np.array(v, np.float64, copy=True)
    h = 0.5 * dt
    w = np.asarray(inv_m, np.float64)[:, None]
    v += h * np.asarray(f, np.float64) * w
    rattle(x, v, inv_m, clusters, box, tol_v)
    for _ in range(2):  # A + SHAKE, dazwischen O (γ = 0 ⇒ nur RATTLE)
        x_old = x.copy()
        x_un = x + h * v
        x = x_un.copy()
        shake(x, x_old, inv_m, clusters, distances, box, tol)
        v += (x - x_un) / h
        rattle(x, v, inv_m, clusters, box, tol_v)
    return x, v
