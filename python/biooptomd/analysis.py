"""Auswertung der Observer-Zeitreihen (ROADMAP Phase 7, Aufgabe 4; PHYSICS_MODEL.md §6).

- :func:`fret_efficiency`: FRET-Effizienz aus den Rohzeitreihen R_DA(t), κ²(t) in drei
  Mittelungsregimen (hängen vom Verhältnis Rotations-/Abstandsdynamik zur Fluoreszenzlebensdauer ab).
- :func:`block_average`, :func:`block_analysis`: Mittelwert und Standardfehler korrelierter
  Zeitreihen (Blockmittelung nach Flyvbjerg & Petersen 1989).
- :func:`refractive_index_layers`, :func:`export_refractive_index`: n(z) als Schichtstapel
  (Dicke, n) für Transfer-Matrix-Solver.

Einheiten: nm, ps.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

FRET_REGIMES = ("dynamic", "static", "static_isotropic", "static_distance")


def fret_efficiency(r_da, kappa2, r0_nm: float, regime: str = "dynamic", axis: int = -1):
    """Mittlere FRET-Effizienz über die Frames (Achse ``axis``).

    Rate relativ zur Donor-Zerfallsrate: k_T/k_D = (3/2) κ² (R₀/R)⁶ (R₀ für κ² = 2/3).

    - ``"dynamic"``: Orientierung und Abstand mitteln während der Lebensdauer (schnell):
      E = ⟨k⟩/(1 + ⟨k⟩) mit k = (3/2) κ² (R₀/R)⁶.
    - ``"static"``: beide eingefroren während der Lebensdauer: E = ⟨k/(1 + k)⟩.
    - ``"static_isotropic"``: Orientierung schnell und isotrop (κ² → 2/3), Abstand eingefroren:
      E = ⟨1/(1 + (R/R₀)⁶)⟩ (κ² wird ignoriert).
    - ``"static_distance"``: Orientierung schnell, aber anisotrop (z. B. nahe der Oberfläche),
      Abstand eingefroren: E = ⟨k̄/(1 + k̄)⟩ mit k̄ = (3/2) ⟨κ²⟩ (R₀/R)⁶, ⟨κ²⟩ über alle Frames.

    Args:
        r_da: R_DA [nm] (…, n_frames); kappa2: κ² (gleiche Shape); r0_nm: Förster-Radius [nm].
    """
    if regime not in FRET_REGIMES:
        raise ValueError(f"regime: eines von {FRET_REGIMES}")
    if not r0_nm > 0:
        raise ValueError("r0_nm > 0 erwartet")
    r = np.asarray(r_da, np.float64)
    q6 = (r0_nm / r) ** 6
    if regime == "static_isotropic":
        return np.mean(q6 / (1.0 + q6), axis=axis)
    if regime == "static_distance":
        kbar = 1.5 * np.mean(np.asarray(kappa2, np.float64), axis=axis, keepdims=True) * q6
        return np.mean(kbar / (1.0 + kbar), axis=axis)
    k = 1.5 * np.asarray(kappa2, np.float64) * q6
    if regime == "static":
        return np.mean(k / (1.0 + k), axis=axis)
    km = np.mean(k, axis=axis)
    return km / (1.0 + km)


def block_average(x, n_blocks: int = 20, axis: int = -1):
    """Mittelwert und Standardfehler aus ``n_blocks`` gleich langen Blöcken (Rest verworfen).

    Returns:
        (mean, sem) über ``axis``; sem = std(Blockmittel, ddof=1)/√n_blocks.
    """
    x = np.moveaxis(np.asarray(x, np.float64), axis, -1)
    if n_blocks < 2 or x.shape[-1] < n_blocks:
        raise ValueError("block_average: mindestens 2 Blöcke und ein Wert je Block")
    m = x.shape[-1] // n_blocks * n_blocks
    blocks = x[..., :m].reshape(x.shape[:-1] + (n_blocks, -1)).mean(axis=-1)
    return blocks.mean(axis=-1), blocks.std(axis=-1, ddof=1) / np.sqrt(n_blocks)


def block_analysis(x):
    """Flyvbjerg-Petersen-Blocktransformation einer 1D-Zeitreihe.

    Halbiert die Reihe wiederholt durch Paarmittelung und schätzt je Stufe den Standardfehler
    σ_k = sqrt(Var_k/(n_k − 1)). Bei dekorrelierten Blöcken erreicht σ_k ein Plateau.

    Returns:
        dict mit ``block_size`` (Werte je Block), ``sem`` und ``sem_error`` (Fehler der Schätzung,
        σ_k/sqrt(2(n_k − 1))) je Stufe, sowie ``sem_plateau``: σ_k der ersten Stufe (mit noch
        mindestens 32 Blöcken), ab der σ_{k+1} innerhalb des Schätzfehlers nicht mehr steigt;
        ohne Plateau das größte σ_k dieser Stufen (und ``converged`` = False).
    """
    y = np.asarray(x, np.float64).ravel()
    sizes, sems, errs = [], [], []
    size = 1
    while len(y) >= 4:
        n = len(y)
        sem = np.sqrt(y.var(ddof=0) / (n - 1))
        sizes.append(size)
        sems.append(sem)
        errs.append(sem / np.sqrt(2.0 * (n - 1)))
        y = 0.5 * (y[: n // 2 * 2 : 2] + y[1 : n // 2 * 2 : 2])
        size *= 2
    sizes, sems, errs = np.array(sizes), np.array(sems), np.array(errs)
    n_blocks = np.asarray(x).size // sizes
    ok = np.flatnonzero(n_blocks >= 32)
    plateau, converged = (sems[ok].max() if len(ok) else sems.max()), False
    for k in ok[:-1]:
        if sems[k + 1] - sems[k] <= errs[k]:
            plateau, converged = sems[k], True
            break
    return {
        "block_size": sizes, "sem": sems, "sem_error": errs, "sem_plateau": float(plateau),
        "converged": converged,
    }  # fmt: skip


def refractive_index_layers(z_nm, n, dz_nm: float, n_water: float, tol: float = 1e-9):
    """n(z)-Profil → Schichtstapel für Transfer-Matrix-Solver.

    Benachbarte Schichten mit |Δn| ≤ tol (Default: nur Rundung) werden zusammengefasst (Dicke addiert, n
    dickengewichtet gemittelt); die oberste Schicht mit n = n_w ist das Halbraum-Medium und wird
    entfernt, ebenso NaN-Schichten nicht toleriert (ValueError).

    Args:
        z_nm: Schichtmitten [nm] (``result()['z']``); n: n je Schicht (eine Replica);
        dz_nm: Schichtdicke [nm]; n_water: Brechungsindex des Lösungsmittels.

    Returns:
        (thickness [nm], n) als 1D-Arrays, von der Oberfläche nach außen.
    """
    z = np.asarray(z_nm, np.float64)
    nn = np.asarray(n, np.float64)
    if z.shape != nn.shape or z.ndim != 1:
        raise ValueError("z und n: gleiche 1D-Shape")
    if not np.isfinite(nn).all():
        raise ValueError("n enthält NaN (L >= 1, unphysikalische Parameter)")
    thick, vals = [dz_nm], [nn[0]]
    for v in nn[1:]:
        if abs(v - vals[-1]) <= tol:
            vals[-1] = (vals[-1] * thick[-1] + v * dz_nm) / (thick[-1] + dz_nm)
            thick[-1] += dz_nm
        else:
            thick.append(dz_nm)
            vals.append(v)
    thick, vals = np.array(thick), np.array(vals)
    while len(vals) and abs(vals[-1] - n_water) <= max(tol, 1e-7):  # float32-Ausgabe
        thick, vals = thick[:-1], vals[:-1]
    return thick, vals


def export_refractive_index(path, z_nm, n, header: str = "") -> None:
    """Schreibt n(z) als CSV (UTF-8; z [nm], n); mehrere Replicas als weitere Spalten."""
    z = np.asarray(z_nm, np.float64)
    nn = np.atleast_2d(np.asarray(n, np.float64))
    if nn.shape[-1] != len(z):
        raise ValueError("n: (n_bins,) oder (n_replicas, n_bins)")
    cols = ["z_nm"] + [f"n_replica_{r}" for r in range(nn.shape[0])]
    head = (header + "\n" if header else "") + ",".join(cols)
    np.savetxt(
        Path(path),
        np.column_stack([z, nn.T]),
        delimiter=",",
        header=head,
        comments="# ",
        encoding="utf-8",
    )
