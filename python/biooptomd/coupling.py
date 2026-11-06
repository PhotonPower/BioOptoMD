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


# --- hp-FEM-Adapter (ADR 0016 Punkt 4b, OQ-20) ------------------------------------------------


@dataclass
class HpFemGratingSolver:
    """hp-FEM (https://github.com/PhotonPower/hp-FEM, ``hpfem.grating.solve``) im selben Prozess für
    Strukturen, die in x periodisch (Periode = Box L_x) und in y invariant sind: Einfallsmedium
    (Prisma, ``n_incident``) | ``substrate`` [(ε, Dicke nm), …] bis z = ``z_surface`` | Wasser mit
    dem Molekülbereich des Beobachtergitters. ⟨ε⟩ wird über y gemittelt und je FEM-Zelle im
    Bereich über ``MaterialMap.set_cell`` gesetzt: ``interpolation="cell"`` (Default) nimmt den Wert
    der ε-Zelle, die den Schwerpunkt enthält (⟨ε⟩ ist ein Zellmittel; konsistent mit dem TMM),
    ``"linear"`` interpoliert bilinear in (x, z) zwischen den Zellmitten; |E|² wird an
    den Knoten (x, z) des Kraftgitters über ``GratingResult.field`` ausgewertet und auf y übertragen.

    Koordinaten: hp-FEM-y = z_first − z (Licht von oben, ``stack.top`` = 0 an der ersten
    Grenzfläche), hp-FEM-x = x (Einheitszelle [x₀, x₀ + L_x) mit x₀ = Herkunft des ε-Gitters, damit
    die Netzspalten auf den ε-Zellgrenzen liegen), Einheiten m ↔ nm. Die einfallende Welle hat in
    hp-FEM die Amplitude 1 V/m, daher W = |E₀|² · Σ|E_c|² mit |E₀|² aus ``intensity_w_m2`` (W/m², im
    Einfallsmedium).

    Netz: strukturiert, Zeilen an allen Stapel-Grenzflächen, ε-Zellgrenzen und PML-Anfängen,
    Zellhöhe ``h_far`` [nm] in Prisma und Wasser, ``h_layer`` in den Substratschichten, die ε-Zellen im
    Bereich (``refine_region`` Zeilen je Zelle); in x ``nx_per_cell`` Spalten je ε-Zelle über die
    ganze Höhe (strukturiert, keine hängenden Knoten). Viele ε-Spalten erzeugen daher dünne, breite
    Zellen in Prisma und Wasser: 60 × 80 ε-Zellen à 0.1 nm brauchen bei ``order`` 4 etwa 400k
    Unbekannte und ~1.5 min je Lösung (RTX-3090-Host); lateral gröbere ε-Gitter oder
    ``hpfem.grating.estimate_memory`` vorab. PML ``pml_nm`` oben und unten (Default: 1.25 lokale
    Wellenlängen λ/min(n), Zellhöhe ``h_far``/2, über hp-FEMs Empfehlung von einer halben
    Wellenlänge plus Reserve), Wasserpuffer ``water_nm`` unter dem Bereich.
    Warnungen aus ``result.diagnostics`` und ein Leistungsbilanz-Residuum über ``balance_warn``
    werden als ``RuntimeWarning`` weitergereicht.

    Knoten des Kraftgitters auf einer Netzlinie (z und x) erhalten das Mittel der angrenzenden
    Zellen (2 bzw. an Ecken 4 Seiten), wie :meth:`TmmResult.intensity`.
    """

    wavelength_nm: float
    angle_deg: float
    pol: str
    n_incident: float
    substrate: list = field(default_factory=list)
    intensity_w_m2: float = 1e10
    z_surface: float = 0.0
    order: int = 4
    h_far: float = 40.0
    h_layer: float = 5.0
    refine_region: int = 1
    nx_per_cell: int = 1
    pml_nm: float | None = None
    water_nm: float = 400.0
    prism_nm: float = 300.0
    interpolation: str = "cell"
    balance_warn: float = 1e-3
    last_result: object = field(default=None, repr=False)

    def solve(self, eps: EpsGrid, grid: ForceGrid) -> np.ndarray:
        import hpfem
        import hpfem.grating as hg

        if self.interpolation not in ("cell", "linear"):
            raise ValueError("HpFemGratingSolver: interpolation 'cell' oder 'linear'")

        nm = 1e-9
        if not grid.periodic_xy:
            raise ValueError("HpFemGratingSolver: Kraftgitter periodisch in x/y (Periode = Box)")
        period = grid.shape[0] * grid.spacing[0]
        nx_e = eps.eps.shape[2]
        if abs(nx_e * eps.spacing[0] - period) > 1e-6 * period:
            raise ValueError(
                "HpFemGratingSolver: ε-Gitter und Kraftgitter brauchen dieselbe Periode in x"
            )
        x0 = float(eps.origin[0])
        eps_xz = eps.eps.mean(axis=1)  # (nz, nx), Mittel über y (in hp-FEM invariant)
        eps_w = eps.n_water**2
        d_sub = [float(d) for _, d in self.substrate]
        z_first = self.z_surface - sum(d_sub)
        z0, dz, nz_e = eps.origin[2], eps.spacing[2], eps.eps.shape[0]
        z_top_region = z0 + nz_e * dz
        if z0 < self.z_surface - 1e-9:
            raise ValueError(
                "HpFemGratingSolver: Beobachtergitter beginnt unterhalb der Oberfläche"
            )
        z_nodes = grid.nodes(2)
        z_prism = z_first - self.prism_nm  # Beginn der oberen PML (unser z)
        if z_nodes.min() <= z_prism:
            raise ValueError("HpFemGratingSolver: Kraftgitterknoten liegen in der oberen PML")
        pml = self.pml_nm
        if pml is None:
            lam_local = self.wavelength_nm / min(self.n_incident, eps.n_water)
            h_pml = 0.5 * self.h_far
            pml = float(np.ceil(1.25 * lam_local / h_pml) * h_pml)
        z_bottom = max(z_top_region, float(z_nodes.max())) + self.water_nm

        def lin(a, b, h):
            n = max(1, int(np.ceil((b - a) / h - 1e-9)))
            return np.linspace(a, b, n + 1)

        pieces = [lin(z_prism - pml, z_prism, 0.5 * self.h_far), lin(z_prism, z_first, self.h_far)]
        z = z_first
        for d in d_sub:
            pieces.append(lin(z, z + d, self.h_layer))
            z += d
        if z0 > self.z_surface:
            pieces.append(lin(self.z_surface, z0, min(self.h_layer, z0 - self.z_surface)))
        pieces.append(np.linspace(z0, z_top_region, nz_e * self.refine_region + 1))
        pieces.append(lin(z_top_region, z_bottom, self.h_far))
        pieces.append(lin(z_bottom, z_bottom + pml, 0.5 * self.h_far))
        zs = np.sort(np.concatenate(pieces))
        zs = zs[np.concatenate([[True], np.diff(zs) > 1e-6])]  # Duplikate an Stückgrenzen
        ys = (z_first - zs)[::-1]  # hp-FEM-y aufsteigend
        nx = nx_e * self.nx_per_cell
        ny = len(ys) - 1
        mesh = hpfem.rectangle(
            nx, ny, np.array([x0 * nm, 0.0]), np.array([(x0 + period) * nm, 1.0])
        )
        xs_mesh = x0 + np.arange(nx + 1) * (period / nx)
        for v in range(mesh.num_vertices):
            x, y = mesh.vertex(v)
            i = int(np.rint((x / nm - x0) / (period / nx)))
            mesh.set_vertex(v, np.array([xs_mesh[i] * nm, ys[int(np.rint(y * ny))] * nm]))
        stack = hpfem.LayerStack2D(
            hpfem.Material.dielectric(self.n_incident),
            [hpfem.Layer(hpfem.Material(complex(e)), float(d) * nm) for e, d in self.substrate],
            hpfem.Material(complex(eps_w)),
            0.0,
        )
        mm = hpfem.MaterialMap(stack.incidence_medium)
        cent = np.asarray(mesh.cell_centroids)
        xe = x0 + (np.arange(nx_e) + 0.5) * eps.spacing[0]
        ze = z0 + (np.arange(nz_e) + 0.5) * dz
        for c in range(mesh.num_cells):
            xc, yc = float(cent[c][0]), float(cent[c][1])
            zc = z_first - yc / nm
            if z0 < zc < z_top_region:
                if self.interpolation == "cell":
                    i = int(np.floor((xc / nm - x0) / eps.spacing[0])) % nx_e
                    k = min(int(np.floor((zc - z0) / dz)), nz_e - 1)
                    value = eps_xz[k, i]
                else:
                    value = _bilinear_periodic_x(eps_xz, xe, ze, xc / nm, zc, period)
                mm.set_cell(c, hpfem.Material(complex(value)))
            else:
                mm.set_cell(c, stack.material_at([xc, yc]))
        res = hg.solve(
            mesh, mm, stack, self.pol, np.deg2rad(self.angle_deg), 0.0,
            2.0 * np.pi * units.C_LIGHT / (self.wavelength_nm * nm), order=self.order,
            cover_line=0.5 * self.prism_nm * nm,
            substrate_line=(z_first - z_bottom + 0.5 * self.water_nm) * nm,
            pml={"top": pml * nm, "bottom": pml * nm},
        )  # fmt: skip
        self.last_result = res
        for diag in getattr(res, "diagnostics", None) or []:
            level = str(getattr(diag, "level", getattr(diag, "severity", ""))).lower()
            if "warn" in level:
                warnings.warn(f"HpFemGratingSolver (hp-FEM): {diag}", RuntimeWarning, stacklevel=2)
        resid = float(getattr(res, "power_balance_residual", 0.0) or 0.0)
        if abs(resid) > self.balance_warn:
            warnings.warn(
                f"HpFemGratingSolver: Leistungsbilanz-Residuum {resid:.1e} (> {self.balance_warn:g});"
                " PML/Netz verfeinern",
                RuntimeWarning,
                stacklevel=2,
            )
        return np.broadcast_to(
            self._sample(res, grid, x0, period, nx, z_first, z_first - ys)[:, None, :],
            grid.zyx_shape,
        ).copy()

    def _sample(self, res, grid, x0, period, nx, z_first, z_lines):
        nm = 1e-9
        delta = 1e-4  # nm
        z_nodes, x_nodes = grid.nodes(2), grid.nodes(0)
        xm = x0 + np.mod(x_nodes - x0, period)  # in die Einheitszelle [x0, x0 + L)
        h = period / nx
        frac = (xm - x0) / h
        x_on = np.abs(frac - np.rint(frac)) * h <= 1e-9
        z_on = np.any(np.abs(z_nodes[:, None] - z_lines[None, :]) <= 1e-9, axis=1)
        w = np.empty((len(z_nodes), len(x_nodes)))
        for k, zk in enumerate(z_nodes):
            zsh = (zk - delta, zk + delta) if z_on[k] else (zk,)
            for i, xi in enumerate(xm):
                if x_on[i]:
                    xl = x0 + np.rint(frac[i]) * h
                    left = xl - delta if xl - x0 > 0.5 * h else x0 + period - delta  # x0 ≡ x0 + L
                    right = xl + delta if xl - x0 < period - 0.5 * h else x0 + delta
                    xsh = (left, right)
                else:
                    xsh = (xi,)
                pts = np.array([[xx, z_first - zz] for xx in xsh for zz in zsh]) * nm
                w[k, i] = (np.abs(res.field(pts)) ** 2).sum(axis=1).mean()
        return field_amplitude_sq(self.intensity_w_m2, self.n_incident) * w


def _bilinear_periodic_x(values, xc, zc, x, z, period):
    """Bilineare Interpolation auf Zellmitten (xc periodisch, zc am Rand konstant fortgesetzt)."""
    nx, nz = len(xc), len(zc)
    dx = period / nx
    u = (x - xc[0]) / dx
    i0 = int(np.floor(u))
    fx = u - i0
    i0, i1 = i0 % nx, (i0 + 1) % nx
    if nz == 1:
        k0 = k1 = 0
        fz = 0.0
    else:
        dz = zc[1] - zc[0]
        v = np.clip((z - zc[0]) / dz, 0.0, nz - 1.0)
        k0 = min(int(np.floor(v)), nz - 2)
        k1, fz = k0 + 1, v - k0
    a = values[k0, i0] * (1 - fx) + values[k0, i1] * fx
    b = values[k1, i0] * (1 - fx) + values[k1, i1] * fx
    return a * (1 - fz) + b * fz


@dataclass
class HpFemSolver3D:
    """hp-FEM ``Scattering3D`` im selben Prozess für Strukturen, die in x und y variieren (ADR 0016
    Punkt 4b, ROADMAP 10.5): Bloch-periodische Einheitszelle [x₀, x₀ + L_x) × [y₀, y₀ + L_y) (Box;
    x₀, y₀ = Herkunft des ε-Gitters), Streufeld-Formulierung mit dem Schichtstapel Einfallsmedium
    (Prisma, ``n_incident``) | ``substrate`` [(ε, Dicke nm), …] bis z = ``z_surface`` | Wasser als
    analytischem Hintergrund (``LayerStack3D``), PML nur oben und unten.

    ⟨ε⟩ je Tetraeder im Molekülbereich: Wert der ε-Zelle, die den Schwerpunkt enthält (Netzlinien
    auf allen ε-Zellgrenzen, ``n_per_cell`` Unterteilungen je Zelle und Achse). |E|² an den Knoten
    des Kraftgitters über ``Scattering3D.sample`` (Gesamtfeld); Knoten auf Netzlinien erhalten das
    Mittel der angrenzenden Zellen (bis zu 8). Koordinaten und Amplitude wie
    :class:`HpFemGratingSolver` (hp-FEM-z = z_first − z, einfallende Welle 1 V/m, Einfallsebene x–z).

    Parameter: ``wavelength_nm`` [nm], ``angle_deg`` [°] (Einfallsebene x–z), ``pol`` 's'/'p',
    ``intensity_w_m2`` [W/m²] im Einfallsmedium, ``z_surface`` [nm]; Netz: ``h_far`` [nm] in Prisma und
    Wasser, ``h_layer`` [nm] in den Substratschichten, ``pml_nm`` [nm] (Default 1.25 λ/min(n), dort
    Zellhöhe ``h_far``/2; bei Speicherdruck der einfachste Hebel), ``water_nm`` und ``prism_nm`` [nm]
    Abstand zur PML.

    Genauigkeit (ROADMAP 10.5): planar ist Grad 2 ausreichend (gegen TMM 1e-4); bei lateral
    strukturiertem ε liegt Grad 2 an Knoten nahe ε-Kanten bei ≈ 1 %, Grad 3 bei ≈ 0.5 % – für
    Hotspots Grad 3 verwenden. Es gibt keine Leistungsbilanz-Prüfung wie im 2.5D-Adapter.

    Aufwand: strukturiertes Tetraedernetz (6 je Quader) mit allen ε-Spalten über die ganze Höhe;
    Polynomgrad ``order`` (Nédélec 3D, <= 4), im Molekülbereich optional ``order_region`` (kleine
    Zellen, dort genügt ein niedrigerer Grad). Die Unbekannten wachsen etwa mit
    n_x · n_y · n_z · order³; Messwerte in docs/ROADMAP.md 10.5. Das Windows-Wheel von hp-FEM 0.4.0
    rechnet einfädig mit SparseLU (ohne OpenMP, MUMPS, cuDSS); feinere laterale Auflösungen brauchen
    einen Build mit diesen Backends.
    """

    wavelength_nm: float
    angle_deg: float
    pol: str
    n_incident: float
    substrate: list = field(default_factory=list)
    intensity_w_m2: float = 1e10
    z_surface: float = 0.0
    order: int = 2
    order_region: int | None = None
    h_far: float = 40.0
    h_layer: float = 5.0
    n_per_cell: int = 1
    pml_nm: float | None = None
    water_nm: float = 400.0
    prism_nm: float = 300.0
    last_problem: object = field(default=None, repr=False)
    last_unknowns: int = 0

    def solve(self, eps: EpsGrid, grid: ForceGrid) -> np.ndarray:
        import hpfem

        nm = 1e-9
        if self.pol not in ("s", "p"):
            raise ValueError("HpFemSolver3D: pol 's' oder 'p'")
        for o in (self.order, self.order_region):
            if o is not None and not 1 <= int(o) <= 4:
                raise ValueError("HpFemSolver3D: order, order_region in 1 … 4")
        if not grid.periodic_xy:
            raise ValueError("HpFemSolver3D: Kraftgitter periodisch in x/y (Periode = Box)")
        nz_e, ny_e, nx_e = eps.eps.shape
        lx, ly = grid.shape[0] * grid.spacing[0], grid.shape[1] * grid.spacing[1]
        if (
            abs(nx_e * eps.spacing[0] - lx) > 1e-6 * lx
            or abs(ny_e * eps.spacing[1] - ly) > 1e-6 * ly
        ):
            raise ValueError("HpFemSolver3D: ε-Gitter und Kraftgitter brauchen dieselbe Periode")
        x0, y0 = float(eps.origin[0]), float(eps.origin[1])
        d_sub = [float(d) for _, d in self.substrate]
        z_first = self.z_surface - sum(d_sub)
        z0, dz = eps.origin[2], eps.spacing[2]
        z_top_region = z0 + nz_e * dz
        if z0 < self.z_surface - 1e-9:
            raise ValueError("HpFemSolver3D: Beobachtergitter beginnt unterhalb der Oberfläche")
        z_nodes = grid.nodes(2)
        z_prism = z_first - self.prism_nm
        if z_nodes.min() <= z_prism:
            raise ValueError("HpFemSolver3D: Kraftgitterknoten liegen in der oberen PML")
        pml = self.pml_nm
        if pml is None:
            h_pml = 0.5 * self.h_far
            lam_local = self.wavelength_nm / min(self.n_incident, eps.n_water)
            pml = float(np.ceil(1.25 * lam_local / h_pml) * h_pml)
        z_bottom = max(z_top_region, float(z_nodes.max())) + self.water_nm

        def lin(a, b, h):
            n = max(1, int(np.ceil((b - a) / h - 1e-9)))
            return np.linspace(a, b, n + 1)

        pieces = [lin(z_prism - pml, z_prism, 0.5 * self.h_far), lin(z_prism, z_first, self.h_far)]
        z = z_first
        for d in d_sub:
            pieces.append(lin(z, z + d, self.h_layer))
            z += d
        if z0 > self.z_surface:
            pieces.append(lin(self.z_surface, z0, min(self.h_layer, z0 - self.z_surface)))
        pieces.append(np.linspace(z0, z_top_region, nz_e * self.n_per_cell + 1))
        pieces.append(lin(z_top_region, z_bottom, self.h_far))
        pieces.append(lin(z_bottom, z_bottom + pml, 0.5 * self.h_far))
        zs = np.sort(np.concatenate(pieces))
        zs = zs[np.concatenate([[True], np.diff(zs) > 1e-6])]
        zhp = (z_first - zs)[::-1]  # hp-FEM-z aufsteigend
        nx, ny, nz = nx_e * self.n_per_cell, ny_e * self.n_per_cell, len(zhp) - 1
        xs = x0 + np.arange(nx + 1) * (lx / nx)
        ys = y0 + np.arange(ny + 1) * (ly / ny)
        mesh = hpfem.box(nx, ny, nz, np.zeros(3), np.ones(3))
        for v in range(mesh.num_vertices):
            u = np.asarray(mesh.vertex(v))
            i, j, k = int(np.rint(u[0] * nx)), int(np.rint(u[1] * ny)), int(np.rint(u[2] * nz))
            mesh.set_vertex(v, np.array([xs[i], ys[j], zhp[k]]) * nm)
        stack = hpfem.LayerStack3D(
            hpfem.Material.dielectric(self.n_incident),
            [hpfem.Layer(hpfem.Material(complex(e)), float(d) * nm) for e, d in self.substrate],
            hpfem.Material(complex(eps.n_water**2)),
            0.0,
        )
        mm = hpfem.MaterialMap(stack.incidence_medium)
        cent = np.asarray(mesh.cell_centroids) / nm
        orders = np.full(mesh.num_cells, self.order, np.int64)
        for c in range(mesh.num_cells):
            xc, yc, zc_hp = cent[c]
            zc = z_first - zc_hp
            if z0 < zc < z_top_region:
                if self.order_region is not None:
                    orders[c] = self.order_region
                i = int(np.floor((xc - x0) / eps.spacing[0])) % nx_e
                j = int(np.floor((yc - y0) / eps.spacing[1])) % ny_e
                k = min(int(np.floor((zc - z0) / dz)), nz_e - 1)
                mm.set_cell(c, hpfem.Material(complex(eps.eps[k, j, i])))
            else:
                mm.set_cell(c, stack.material_at([xc * nm, yc * nm, zc_hp * nm]))
        k0 = 2.0 * np.pi / (self.wavelength_nm * nm)
        polarisation = hpfem.Polarisation.P if self.pol == "p" else hpfem.Polarisation.S
        wave = stack.plane_wave(k0, np.deg2rad(self.angle_deg), polarisation, 1.0)
        setup = hpfem.ScatteringSetup3D()
        setup.omega = k0 * units.C_LIGHT
        setup.materials = mm
        setup.background = stack
        setup.incident = wave.field
        setup.formulation = hpfem.Formulation.SCATTERED_FIELD
        lower = np.array([x0, y0, zhp[0] + pml]) * nm
        upper = np.array([x0 + lx, y0 + ly, zhp[-1] - pml]) * nm
        setup.pml = hpfem.PmlBox3D(
            lower, upper, [0.0, 0.0, 0.0, 0.0, pml * nm, pml * nm], k0,
            min(self.n_incident, eps.n_water),
        )  # fmt: skip
        kx = k0 * self.n_incident * np.sin(np.deg2rad(self.angle_deg))
        bt = hpfem.box_tag
        setup.periodic = [
            hpfem.PeriodicPair3D(
                bt.X_MIN,
                bt.X_MAX,
                np.array([lx * nm, 0.0, 0.0]),
                complex(np.exp(1j * kx * lx * nm)),
            ),
            hpfem.PeriodicPair3D(bt.Y_MIN, bt.Y_MAX, np.array([0.0, ly * nm, 0.0]), 1.0 + 0j),
        ]
        dofs = hpfem.NedelecDofMap3D(mesh, [int(o) for o in orders])
        problem = hpfem.Scattering3D(dofs, setup)
        solution = problem.solve()
        self.last_problem, self.last_unknowns = problem, int(dofs.num_dofs)
        locator = hpfem.PointLocator3D(mesh)

        # Auswertepunkte: Knoten in die Einheitszelle, auf Netzlinien ±δ in jeder betroffenen Achse.
        delta = 1e-4  # nm
        z_lines = z_first - zhp
        xo = _line_offsets(grid.nodes(0), x0, lx, nx, delta)
        yo = _line_offsets(grid.nodes(1), y0, ly, ny, delta)
        zo = [
            (zk - delta, zk + delta) if np.any(np.abs(z_lines - zk) <= 1e-9) else (zk,)
            for zk in z_nodes
        ]
        pts, owner = [], []
        for k, zz in enumerate(zo):
            for j, yy in enumerate(yo):
                for i, xx in enumerate(xo):
                    node = (k * len(yo) + j) * len(xo) + i
                    for a in xx:
                        for b in yy:
                            for c in zz:
                                pts.append((a, b, z_first - c))
                                owner.append(node)
        values, _ = problem.sample(solution, locator, np.array(pts) * nm)
        intensity = (np.abs(values) ** 2).sum(axis=1)
        owner = np.array(owner)
        n_nodes = len(zo) * len(yo) * len(xo)
        w = np.bincount(owner, intensity, n_nodes) / np.bincount(owner, minlength=n_nodes)
        if not np.all(np.isfinite(w)):
            raise RuntimeError("HpFemSolver3D: Feld an Kraftgitterknoten nicht endlich")
        e0 = field_amplitude_sq(self.intensity_w_m2, self.n_incident)
        return (e0 * w).reshape(grid.zyx_shape)


def _line_offsets(values, origin, length, n, delta):
    """Je Knoten die Auswertekoordinaten in der Einheitszelle: einer, oder links/rechts einer
    Netzlinie (Linien origin + i·length/n; origin ≡ origin + length)."""
    h = length / n
    m = origin + np.mod(values - origin, length)
    frac = (m - origin) / h
    out = []
    for vi, fi in zip(m, frac, strict=True):
        if abs(fi - np.rint(fi)) * h > 1e-9:
            out.append((vi,))
            continue
        line = origin + np.rint(fi) * h
        left = line - delta if line - origin > 0.5 * h else origin + length - delta
        right = line + delta if line - origin < length - 0.5 * h else origin + delta
        out.append((left, right))
    return out


# --- Datei-Adapter (ADR 0016 Punkt 4c) -----------------------------------------------------------


@dataclass
class NpzFileSolver:
    """Rückfall für Solver außerhalb des Prozesses (anderer Interpreter, entfernter Rechner).

    Je Aufruf wird ``workdir/eps_{k:03d}.npz`` geschrieben (``eps`` (nz, ny, nx), ``eps_origin``,
    ``eps_spacing`` [nm], ``n_water``, Kraftgitter ``grid_shape`` (nx, ny, nz), ``grid_origin``,
    ``grid_spacing`` [nm], ``grid_periodic_xy``), dann ``command`` ausgeführt (Liste von Argumenten;
    ``{input}`` und ``{output}`` werden ersetzt; alternativ eine Python-Funktion ``(input, output)``).
    Der Solver schreibt ``workdir/field_{k:03d}.npz`` mit ``w`` = |E₀|² [(V/nm)²] (nz, ny, nx).
    k zählt die Aufrufe dieser Instanz ab ``calls`` (nach ``resume`` mit neuer Instanz ``calls`` auf
    die Zahl der bisherigen Solver-Aufrufe setzen, sonst werden alte Dateien überschrieben).
    """

    workdir: object
    command: object
    timeout_s: float | None = None
    calls: int = 0

    def solve(self, eps: EpsGrid, grid: ForceGrid) -> np.ndarray:
        import subprocess

        work = Path(self.workdir)
        work.mkdir(parents=True, exist_ok=True)
        k = self.calls
        self.calls += 1
        inp, out = work / f"eps_{k:03d}.npz", work / f"field_{k:03d}.npz"
        _save_npz_atomic(
            inp, eps=np.asarray(eps.eps, np.float64), eps_origin=np.asarray(eps.origin, np.float64),
            eps_spacing=np.asarray(eps.spacing, np.float64), n_water=np.float64(eps.n_water),
            grid_shape=np.asarray(grid.shape, np.int64), grid_origin=np.asarray(grid.origin, np.float64),
            grid_spacing=np.asarray(grid.spacing, np.float64),
            grid_periodic_xy=np.bool_(grid.periodic_xy),
        )  # fmt: skip
        if out.exists():
            out.unlink()
        if callable(self.command):
            self.command(str(inp), str(out))
        else:
            args = [str(a).replace("{input}", str(inp)).replace("{output}", str(out))
                    for a in self.command]  # fmt: skip
            subprocess.run(args, check=True, timeout=self.timeout_s)
        if not out.exists():
            raise RuntimeError(f"NpzFileSolver: {out} wurde nicht geschrieben")
        with np.load(out, allow_pickle=False) as d:
            return np.array(d["w"], np.float64)


def read_eps_file(path) -> tuple[EpsGrid, ForceGrid]:
    """Liest eine Eingabedatei von :class:`NpzFileSolver` (für externe Solver-Skripte in Python)."""
    with np.load(path, allow_pickle=False) as d:
        eps = EpsGrid(np.array(d["eps"]), tuple(d["eps_origin"]), tuple(d["eps_spacing"]),
                      float(d["n_water"]))  # fmt: skip
        grid = ForceGrid(tuple(int(v) for v in d["grid_shape"]), tuple(d["grid_origin"]),
                         tuple(d["grid_spacing"]), bool(d["grid_periodic_xy"]))  # fmt: skip
    return eps, grid
