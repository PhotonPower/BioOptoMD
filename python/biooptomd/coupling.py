"""Selbstkonsistente Kopplung MD ↔ Optik (ADR 0016, ROADMAP Phase 10).

- :class:`EpsGrid`, :class:`ForceGrid`: ⟨ε⟩ auf dem Beobachtergitter (Zellen) bzw. die Knoten des
  Kraftgitters (``VoxelGridForce3D``).
- :class:`OpticalSolver`: Protokoll ``solve(eps, grid) -> W`` mit W = |E₀|² [(V/nm)²]
  (Spitzenamplitude, OQ-04) an den Knoten von ``grid``, Shape (nz, ny, nx).
- :func:`tmm` und :class:`TmmSolver`: 1D-Transfermatrix-Referenz (s/p, z. B. Kretschmann über Gold).
- :class:`SelfConsistentOptics`: Picard-Iteration mit Dämpfung β, Checkpoint je Iteration.

Einheiten: nm, (V/nm)², Intensität in W/m² (SI). Zeitkonvention e^{−iωt}: absorbierende Medien haben
Im ε > 0.
"""

from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np

from . import io as _io
from . import units
from .voxel import values_from_potential


def field_amplitude_sq(intensity_w_m2: float, n_medium: float) -> float:
    """|E₀|² [(V/nm)²] einer ebenen Welle der Intensität I [W/m²] im Medium n:
    I = ½ n c ε₀ |E₀|² (Spitzenamplitude)."""
    return 2.0 * intensity_w_m2 / (n_medium * units.C_LIGHT * units.EPS0_SI) * 1e-18


# --- Gitter und Protokoll ------------------------------------------------------------------------


@dataclass
class EpsGrid:
    """⟨ε⟩ je Zelle (nz, ny, nx); Zellen ab der unteren Ecke ``origin`` [nm] mit ``spacing`` [nm];
    ``n_water`` des Lösungsmittels bei der Wellenlänge."""

    eps: np.ndarray
    origin: tuple
    spacing: tuple
    n_water: float

    def centers(self, axis: int) -> np.ndarray:
        n = self.eps.shape[2 - axis]
        return self.origin[axis] + (np.arange(n) + 0.5) * self.spacing[axis]


@dataclass
class ForceGrid:
    """Knoten des Kraftgitters: shape (nx, ny, nz), origin, spacing [nm] wie ``add_voxel_grid``."""

    shape: tuple
    origin: tuple
    spacing: tuple
    periodic_xy: bool = False

    def nodes(self, axis: int) -> np.ndarray:
        return self.origin[axis] + np.arange(self.shape[axis]) * self.spacing[axis]

    @property
    def zyx_shape(self) -> tuple:
        return (self.shape[2], self.shape[1], self.shape[0])


class OpticalSolver(Protocol):
    def solve(self, eps: EpsGrid, grid: ForceGrid) -> np.ndarray:
        """W = |E₀|² [(V/nm)²] an den Knoten von ``grid``, Shape (nz, ny, nx)."""
        ...


# --- Transfermatrix-Referenz ---------------------------------------------------------------------


def _kz(eps, kx2, k0):
    kz = np.sqrt(k0**2 * np.asarray(eps, complex) - kx2 + 0j)
    # Zerfallende bzw. auslaufende Welle: Im kz > 0, bei Im kz = 0 Re kz >= 0.
    flip = (kz.imag < 0) | ((kz.imag == 0) & (kz.real < 0))
    return np.where(flip, -kz, kz)


@dataclass
class TmmResult:
    """Ergebnis von :func:`tmm`. Amplituden ``amp[j] = (A_j, B_j)`` der vor- bzw. rücklaufenden Welle
    je Schicht j, bezogen auf die Unterkante z_j der Schicht (Schicht 0: auf z = 0); s: E_y, p: H_y,
    normiert auf A_0 = 1. ``kx``, ``kz`` [1/nm], ``k0`` = 2π/λ [1/nm], ``z_bounds`` [nm] (Grenzflächen,
    die erste bei 0)."""

    r: complex
    t: complex
    R: float
    T: float
    pol: str
    kx: float
    kz: np.ndarray
    eps: np.ndarray
    z_bounds: np.ndarray
    amp: np.ndarray
    k0: float

    def _intensity_in(self, j, z):
        ref = np.where(j == 0, 0.0, self.z_bounds[np.maximum(j - 1, 0)])
        phase = 1j * self.kz[j] * (z - ref)
        a = self.amp[j, 0] * np.exp(phase)
        b = self.amp[j, 1] * np.exp(-phase)
        if self.pol == "s":
            return np.abs(a + b) ** 2
        n0sq = self.eps[0].real
        num = np.abs(self.kz[j] * (a - b)) ** 2 + np.abs(self.kx * (a + b)) ** 2
        return num / np.abs(self.eps[j]) ** 2 * n0sq / self.k0**2

    def intensity(self, z) -> np.ndarray:
        """|E(z)|²/|E_inc|² (Verstärkung gegenüber der einfallenden Welle in Schicht 0), z [nm].

        Bei p springt E_z an Grenzflächen (ε E_z stetig); für z genau auf einer Grenzfläche
        (|Δz| <= 1e-9 nm) ist der Wert das Mittel beider Seiten.
        """
        z = np.asarray(z, np.float64)
        right = np.searchsorted(self.z_bounds, z, side="right")  # 0 … N−1
        left = np.searchsorted(self.z_bounds, z, side="left")
        out = self._intensity_in(right, z)
        idx = np.clip(np.searchsorted(self.z_bounds, z), 0, len(self.z_bounds) - 1)
        on = np.abs(self.z_bounds[idx] - z) <= 1e-9
        if np.any(on):
            j_hi = np.where(on, idx + 1, right)
            j_lo = np.where(on, idx, left)
            both = 0.5 * (self._intensity_in(j_lo, z) + self._intensity_in(j_hi, z))
            out = np.where(on, both, out)
        return out


def tmm(eps, thickness_nm, wavelength_nm: float, angle_rad: float, pol: str) -> TmmResult:
    """Ebene Welle aus der Halbebene 0 (reelles ε₀, z < 0) unter dem Winkel ``angle_rad`` [rad] auf
    den Schichtstapel 1 … N−2 (Dicken ``thickness_nm`` [nm] >= 0, erste Grenzfläche bei z = 0) über
    der Halbebene N−1; Vakuumwellenlänge ``wavelength_nm`` [nm].

    Randbedingungen: Tangentiales E und H stetig, d. h. A + B und q (A − B) mit q = k_z (s) bzw.
    q = k_z/ε (p, Amplitude H_y). Numerisch stabil von oben: In der letzten Halbebene A = 1, B = 0,
    rückwärts durch die Grenzflächen; in absorbierenden oder evaneszenten Schichten wächst dabei nur
    die physikalisch dominante Lösung. Danach Normierung auf A_0 = 1: r = B_0, t = A_{N−1}.
    R = |r|², T = Re q_{N−1}/q_0 · |t|².
    """
    eps = np.asarray(eps, complex)
    d = np.asarray(thickness_nm, np.float64)
    if eps.ndim != 1 or len(eps) < 2 or len(d) != len(eps) - 2:
        raise ValueError("tmm: eps (N,), thickness (N − 2,)")
    if np.any(d < 0) or not np.all(np.isfinite(d)):
        raise ValueError("tmm: Dicken endlich und >= 0")
    if pol not in ("s", "p"):
        raise ValueError("tmm: pol 's' oder 'p'")
    if eps[0].imag != 0 or eps[0].real <= 0:
        raise ValueError("tmm: Einfallsmedium braucht reelles ε > 0")
    k0 = 2.0 * np.pi / wavelength_nm
    kx = k0 * np.sqrt(eps[0].real) * np.sin(angle_rad)
    kz = _kz(eps, kx**2, k0)
    if np.any(kz == 0):
        raise ValueError("tmm: k_z = 0 (genau streifender Einfall/Grenzwinkel) nicht definiert")
    q = kz if pol == "s" else kz / eps
    lengths = np.concatenate([[0.0], d])  # Weg in Schicht j bis zur nächsten Grenzfläche
    n = len(eps)
    amp = np.empty((n, 2), complex)
    amp[-1] = (1.0, 0.0)
    for j in range(n - 2, -1, -1):
        # Werte von Schicht j an ihrer Oberkante aus Schicht j + 1 an deren Unterkante.
        s_ab, d_ab = amp[j + 1, 0] + amp[j + 1, 1], amp[j + 1, 0] - amp[j + 1, 1]
        ratio = q[j + 1] / q[j]
        a_top, b_top = 0.5 * (s_ab + ratio * d_ab), 0.5 * (s_ab - ratio * d_ab)
        ph = np.exp(1j * kz[j] * lengths[j])
        amp[j] = (a_top / ph, b_top * ph)
    if not np.all(np.isfinite(amp)):
        raise ValueError("tmm: Überlauf (Schichten zu dick für double)")
    amp /= amp[0, 0]
    r, t = amp[0, 1], amp[-1, 0]
    z_bounds = np.concatenate([[0.0], np.cumsum(d)])
    return TmmResult(
        r=complex(r), t=complex(t), R=float(abs(r) ** 2),
        T=float((q[-1].real / q[0].real) * abs(t) ** 2), pol=pol, kx=float(kx), kz=kz, eps=eps,
        z_bounds=z_bounds, amp=amp, k0=k0,
    )  # fmt: skip


@dataclass
class TmmSolver:
    """Planarer Referenzsolver (ADR 0016 Punkt 4a): Einfallsmedium (Prisma, ``n_incident``) | feste
    Schichten ``substrate`` [(ε, Dicke nm), …] bis zur Oberfläche z = ``z_surface`` [nm] |
    Wasserspalt bis zur Gitterunterkante | Schichten des Beobachtergitters (lateral gemitteltes ⟨ε⟩
    je z-Zelle) | Wasser. Einfallswinkel ``angle_deg`` [°], Vakuumwellenlänge ``wavelength_nm``;
    die einfallende Welle hat die Intensität ``intensity_w_m2`` [W/m²] im Einfallsmedium. W = |E|²
    hängt nur von z ab und wird auf alle Knoten (x, y) übertragen.

    Näherung: Das laterale Mittel ⟨ε⟩ ist für E_x exakt (Parallelschaltung), für E_z wäre ⟨1/ε⟩⁻¹
    richtig (Wiener-Grenzen); der Unterschied ist zweiter Ordnung im lateralen Kontrast.
    """

    wavelength_nm: float
    angle_deg: float
    pol: str
    n_incident: float
    substrate: list = field(default_factory=list)
    intensity_w_m2: float = 1e10  # 1 MW/cm²
    z_surface: float = 0.0

    def stack(self, eps: EpsGrid):
        """(ε je Schicht inkl. Halbebenen, Dicken [nm], z der ersten Grenzfläche in Gitterkoordinaten)."""
        eps_w = eps.n_water**2
        layers = [complex(self.n_incident**2)]
        thick = []
        for e, dd in self.substrate:
            layers.append(complex(e))
            thick.append(float(dd))
        z0 = eps.origin[2]
        if z0 < self.z_surface - 1e-9:
            raise ValueError("TmmSolver: Beobachtergitter beginnt unterhalb der Oberfläche")
        if z0 > self.z_surface:
            layers.append(complex(eps_w))
            thick.append(z0 - self.z_surface)
        profile = eps.eps.mean(axis=(1, 2))
        for e in profile:
            layers.append(complex(e))
            thick.append(float(eps.spacing[2]))
        layers.append(complex(eps_w))
        z_first = self.z_surface - float(np.sum([dd for _, dd in self.substrate]))
        return np.array(layers), np.array(thick), z_first

    def field_profile(self, eps: EpsGrid, z) -> np.ndarray:
        """W(z) [(V/nm)²] an Positionen z [nm] (Gitterkoordinaten)."""
        e, d, z_first = self.stack(eps)
        res = tmm(e, d, self.wavelength_nm, np.deg2rad(self.angle_deg), self.pol)
        e0 = field_amplitude_sq(self.intensity_w_m2, self.n_incident)
        return e0 * res.intensity(np.asarray(z, np.float64) - z_first)

    def solve(self, eps: EpsGrid, grid: ForceGrid) -> np.ndarray:
        w_z = self.field_profile(eps, grid.nodes(2))
        return np.broadcast_to(w_z[:, None, None], grid.zyx_shape).copy()


# --- Fixpunkt-Treiber ----------------------------------------------------------------------------


def eps_from_histograms(s_alpha, s_volume, voxel_volume: float, n_water: float) -> np.ndarray:
    """ε = n² je Voxel aus mittleren Histogrammen S_α, S_V [nm³] und dem Voxelvolumen [nm³]
    (Lorentz-Lorenz mit Wasser-Hintergrund, §6.3); NaN, wo L >= 1."""
    f_w = np.maximum(0.0, 1.0 - np.asarray(s_volume, np.float64) / voxel_volume)
    lw = (n_water**2 - 1.0) / (n_water**2 + 2.0)
    big_l = 4.0 * np.pi / 3.0 * np.asarray(s_alpha, np.float64) / voxel_volume + f_w * lw
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(big_l < 1.0, (1.0 + 2.0 * big_l) / (1.0 - big_l), np.nan)


def _save_npz_atomic(path: Path, **arrays) -> None:
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        np.savez(f, **arrays)
    os.replace(tmp, path)


class SelfConsistentOptics:
    """Picard-Iteration mit Dämpfung (ADR 0016 Punkt 1, 5).

    Iteration k: MD-Segment unter W_k (``n_equil`` Schritte verworfen, dann ``n_sample`` Schritte mit
    dem ε-Observer in ``n_blocks`` Blöcken), ⟨ε⟩_k über die Replicas ``replicas`` (Mittel der
    Histogramme, dann Lorentz-Lorenz), W_new = solver.solve(⟨ε⟩_k), W_{k+1} = (1 − β) W_k + β W_new.

    Konvergenzmaße (L2 über das Gitter): Δε = ‖⟨ε⟩_k − ⟨ε⟩_{k−1}‖ / ‖⟨ε⟩_k − ε_w‖ (relativ zum
    Kontrast, da ε_w den Absolutwert dominiert) und das Residuum ΔW = ‖W_new − W_k‖ / ‖W_k‖
    (unabhängig von β, anders als die Schrittweite β ΔW). Konvergiert, wenn beide zwei Iterationen
    in Folge unter ``tol`` bzw. ``tol_w`` (Default ``tol``) liegen. ``tol`` muss über dem
    statistischen Fehler liegen: ``eps_noise`` im Verlauf ist der Standardfehler von ⟨ε⟩ aus den
    Blockmitteln (relativ zum Kontrast, mit S_α und S_V gemeinsam ausgewertet); zwei unabhängige
    Mittel unterscheiden sich um etwa √2 · ``eps_noise``.

    Gültigkeit (OQ-24): Die Atome spüren auch das Feld, das ihre eigene Schicht verändert (bei
    p-Polarisation etwa die Absenkung von E_z = D_z/ε im Inneren dichter Schichten). Das ist das
    Kontinuumsbild; mikroskopisch spürt ein Molekül sein eigenes Feld nicht. Der Verlauf enthält
    ``self_field`` = max |W_new − W_Wasser| / max W_Wasser; über ``self_field_warn`` (Default 0.05)
    warnt der Treiber einmal.

    Ohne Replica Exchange: ``replicas`` sind feste Slots, ``ctx.step`` tauscht nicht; bei REMD müsste
    der Zielzustand über die Slots verfolgt werden (offen, ROADMAP Phase 10).

    W_0 ist die Lösung für reines Wasser (ungekoppeltes Feld). Mit ``checkpoint_dir`` wird nach jeder
    Iteration ``iter_{k:03d}.npz`` (Kontext, :func:`biooptomd.io.save_checkpoint`) und
    ``coupling_{k:03d}.npz`` (W, ⟨ε⟩, Verlauf) geschrieben, k = Zahl der abgeschlossenen
    Iterationen; :meth:`resume` setzt dort fort.

    Args:
        ctx: ``SimulationContext`` mit Integrator; Optik-Typen in ``types[..., 1]``.
        solver: :class:`OpticalSolver`.
        force_grid: Knoten des Kraftgitters (:class:`ForceGrid`).
        scale: s_i je Optik-Typ [kJ/mol pro (V/nm)²], z. B. ``voxel.optical_scale(...)``.
        eps_shape, eps_origin, eps_spacing: Beobachtergitter (nx, ny, nz), untere Ecke [nm],
            Zellgröße [nm] (``add_polarizability_grid``; periodisch in x/y wie ``force_grid``).
        alpha, volume: α′ und v je Optik-Typ [nm³] (kalibriert, ``builder.calibrate_optics``).
        n_water: Brechungsindex des Lösungsmittels bei der Wellenlänge.
        n_equil, n_sample: Schritte je Iteration (verworfen bzw. gemessen); ``n_sample`` muss ein
            Vielfaches von ``n_blocks · stride`` sein.
        stride: Observer-Abstand [Schritte]. n_blocks: Blöcke für die Rauschabschätzung (>= 2).
        beta: Dämpfung in (0, 1]. tol, tol_w: Konvergenzgrenzen für Δε bzw. ΔW.
        self_field_warn: Schwelle der Selbstfeld-Warnung (OQ-24).
        replicas: gemittelte Replica-Slots (Default alle; alle mit gleicher Frame-Zahl).
        smearing: ``"gauss"`` oder ``"point"``. checkpoint_dir: Verzeichnis oder None.
    """

    def __init__(self, ctx, solver: OpticalSolver, force_grid: ForceGrid, scale, *,
                 eps_shape, eps_origin, eps_spacing, alpha, volume, n_water: float,
                 n_equil: int, n_sample: int, stride: int = 100, n_blocks: int = 4,
                 beta: float = 0.5, tol: float = 1e-2, tol_w: float | None = None,
                 self_field_warn: float = 0.05, replicas=None, smearing: str = "gauss",
                 checkpoint_dir=None):  # fmt: skip
        if not 0.0 < beta <= 1.0:
            raise ValueError("beta in (0, 1]")
        if n_blocks < 2 or n_sample % (n_blocks * stride) != 0 or n_equil < 0:
            raise ValueError("n_blocks >= 2, n_sample Vielfaches von n_blocks·stride, n_equil >= 0")
        self.ctx, self.solver, self.grid = ctx, solver, force_grid
        self.beta, self.tol = float(beta), float(tol)
        self.tol_w = self.tol if tol_w is None else float(tol_w)
        self.n_equil, self.n_sample, self.n_blocks = int(n_equil), int(n_sample), int(n_blocks)
        self.n_water = float(n_water)
        self.replicas = np.arange(ctx.n_replicas) if replicas is None else np.asarray(replicas)
        self.eps_origin = tuple(float(v) for v in eps_origin)
        self.eps_spacing = tuple(float(v) for v in eps_spacing)
        self.eps_periodic = bool(force_grid.periodic_xy)
        self.observer = ctx.add_polarizability_grid(
            tuple(eps_shape), self.eps_origin, self.eps_spacing, np.asarray(alpha, np.float32),
            np.asarray(volume, np.float32), n_water=n_water, smearing=smearing,
            periodic_xy=self.eps_periodic, stride=stride,
        )  # fmt: skip
        nx, ny, nz = eps_shape
        water = EpsGrid(np.full((nz, ny, nx), n_water**2), self.eps_origin, self.eps_spacing,
                        n_water)  # fmt: skip
        self.w = self._checked(solver.solve(water, force_grid))
        self.w_water = self.w.copy()
        self.self_field_warn = float(self_field_warn)
        self._warned = False
        self.voxel = ctx.add_voxel_grid(
            self._values(self.w), force_grid.origin, force_grid.spacing,
            np.asarray(scale, np.float32), periodic_xy=force_grid.periodic_xy,
        )  # fmt: skip
        self.eps = None
        self.history: list[dict] = []
        self.checkpoint_dir = None if checkpoint_dir is None else Path(checkpoint_dir)

    def _checked(self, w) -> np.ndarray:
        w = np.asarray(w, np.float64)
        if w.shape != self.grid.zyx_shape:
            raise ValueError(f"Solver lieferte {w.shape}, erwartet {self.grid.zyx_shape}")
        if not np.all(np.isfinite(w)) or np.any(w < 0):
            raise ValueError("Solver lieferte W mit NaN/Inf oder W < 0")
        return w

    def _values(self, w):
        return values_from_potential(w, self.grid.spacing, periodic_xy=self.grid.periodic_xy)

    @property
    def iteration(self) -> int:
        return len(self.history)

    def _sample_eps(self):
        self.ctx.step(self.n_equil)
        vk = float(np.prod(self.eps_spacing))
        blocks_a, blocks_v = [], []
        for _ in range(self.n_blocks):
            self.observer.reset()
            self.ctx.step(self.n_sample // self.n_blocks)
            res = self.observer.result()
            blocks_a.append(res["s_alpha_mean"][self.replicas].mean(axis=0))
            blocks_v.append(res["s_volume_mean"][self.replicas].mean(axis=0))
        eps = eps_from_histograms(np.mean(blocks_a, 0), np.mean(blocks_v, 0), vk, self.n_water)
        bad = ~np.isfinite(eps)
        if np.any(bad):
            k = np.argwhere(bad)[0]
            raise ValueError(
                f"⟨ε⟩ nicht endlich in {int(bad.sum())} Voxeln (z. B. Index (z, y, x) = {tuple(k)}): "
                "L >= 1, α′/v oder Voxelgröße unphysikalisch"
            )
        per_block = [
            eps_from_histograms(a, v, vk, self.n_water)
            for a, v in zip(blocks_a, blocks_v, strict=True)
        ]
        sem = np.std(per_block, axis=0, ddof=1) / np.sqrt(self.n_blocks)
        return eps, sem

    def step(self) -> dict:
        """Eine Iteration; gibt den Verlaufseintrag zurück."""
        eps, noise = self._sample_eps()
        contrast = eps - self.n_water**2
        c_norm = float(np.linalg.norm(contrast))
        if self.eps is None:
            d_eps = None  # erste Iteration: kein Vorgänger
        else:
            diff = float(np.linalg.norm(eps - self.eps))
            d_eps = 0.0 if diff == 0.0 else diff / max(c_norm, 1e-300)
        grid = EpsGrid(eps, self.eps_origin, self.eps_spacing, self.n_water)
        w_new = self._checked(self.solver.solve(grid, self.grid))
        w_next = (1.0 - self.beta) * self.w + self.beta * w_new
        w_ref = float(np.linalg.norm(self.w))
        diff_w = float(np.linalg.norm(w_new - self.w))
        d_w = 0.0 if diff_w == 0.0 else diff_w / max(w_ref, 1e-300)
        self.voxel.set_values(self._values(w_next))
        self.w, self.eps = w_next, eps
        entry = {
            "iteration": self.iteration,
            "d_eps": d_eps,
            "d_w": d_w,
            "eps_noise": float(np.linalg.norm(noise) / max(c_norm, 1e-300)),
            "contrast_norm": c_norm,
            "w_norm": float(np.linalg.norm(w_next)),
            "md_step": int(self.ctx.save_state()["step"]),
            "self_field": self._self_field(w_new),
        }
        self.history.append(entry)
        if self.checkpoint_dir is not None:
            self._checkpoint()
        return entry

    def _self_field(self, w_new) -> float:
        ref = float(np.abs(self.w_water).max())
        value = float(np.abs(w_new - self.w_water).max() / ref) if ref > 0 else 0.0
        if value > self.self_field_warn and not self._warned:
            self._warned = True
            warnings.warn(
                f"SelfConsistentOptics: die Schicht ändert W um {value:.1%} (> "
                f"{self.self_field_warn:.0%}); Selbstfeldbeitrag zur Kraft nicht vernachlässigbar "
                "(OQ-24, Mean-Field-Gültigkeit OQ-21)",
                RuntimeWarning,
                stacklevel=3,
            )
        return value

    def converged(self) -> bool:
        if len(self.history) < 2:
            return False
        return all(
            h["d_eps"] is not None and h["d_eps"] < self.tol and h["d_w"] < self.tol_w
            for h in self.history[-2:]
        )

    def run(self, max_iter: int = 10) -> list:
        while self.iteration < max_iter and not self.converged():
            self.step()
        return self.history

    def _checkpoint(self) -> None:
        k = self.iteration
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        _io.save_checkpoint(self.ctx, self.checkpoint_dir / f"iter_{k:03d}.tmp")
        os.replace(
            self.checkpoint_dir / f"iter_{k:03d}.tmp", self.checkpoint_dir / f"iter_{k:03d}.npz"
        )
        _save_npz_atomic(
            self.checkpoint_dir / f"coupling_{k:03d}.npz", w=self.w, eps=self.eps,
            history=json.dumps(self.history),
        )  # fmt: skip

    def resume(self, iteration: int) -> None:
        """Setzt nach ``iteration`` abgeschlossenen Iterationen fort (Checkpoint-Dateien): W, ⟨ε⟩,
        Verlauf und Kontextzustand. Der Kontext muss gleich aufgebaut sein (inkl. dieses Treibers);
        die beiden Dateien müssen zum selben MD-Schritt gehören."""
        if self.checkpoint_dir is None:
            raise ValueError("resume braucht checkpoint_dir")
        with np.load(self.checkpoint_dir / f"coupling_{iteration:03d}.npz") as d:
            w = np.array(d["w"])
            eps = np.array(d["eps"])
            history = json.loads(str(d["history"]))
        step = _io.load_checkpoint(self.ctx, self.checkpoint_dir / f"iter_{iteration:03d}.npz")
        if len(history) != iteration or history[-1]["md_step"] != step:
            raise ValueError("resume: coupling- und iter-Checkpoint passen nicht zusammen")
        self.w, self.eps, self.history = w, eps, history
        self.voxel.set_values(self._values(self.w))
