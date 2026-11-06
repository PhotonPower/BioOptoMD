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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np

from . import io as _io
from .voxel import values_from_potential

C_LIGHT = 299792458.0  # m/s
EPS0 = 8.8541878128e-12  # F/m


def field_amplitude_sq(intensity_w_m2: float, n_medium: float) -> float:
    """|E₀|² [(V/nm)²] einer ebenen Welle der Intensität I [W/m²] im Medium n:
    I = ½ n c ε₀ |E₀|² (Spitzenamplitude)."""
    return 2.0 * intensity_w_m2 / (n_medium * C_LIGHT * EPS0) * 1e-18


# --- Gitter und Protokoll ------------------------------------------------------------------------


@dataclass
class EpsGrid:
    """⟨ε⟩ je Zelle (nz, ny, nx); Zellen ab der unteren Ecke ``origin`` mit ``spacing`` [nm]."""

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
    """Ergebnis von :func:`tmm`. Amplituden (A_j, B_j) der vor- bzw. rücklaufenden Welle je Schicht,
    bezogen auf die Unterkante z_j der Schicht (Schicht 0: auf z = 0); s: E_y, p: H_y."""

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

    def intensity(self, z) -> np.ndarray:
        """|E(z)|²/|E_inc|² (Verstärkung gegenüber der einfallenden Welle in Schicht 0)."""
        z = np.asarray(z, np.float64)
        j = np.searchsorted(self.z_bounds, z, side="right")  # 0 … N−1
        ref = np.where(j == 0, 0.0, self.z_bounds[np.maximum(j - 1, 0)])
        phase = 1j * self.kz[j] * (z - ref)
        a = self.amp[j, 0] * np.exp(phase)
        b = self.amp[j, 1] * np.exp(-phase)
        if self.pol == "s":
            return np.abs(a + b) ** 2
        n0sq = self.eps[0].real
        num = np.abs(self.kz[j] * (a - b)) ** 2 + np.abs(self.kx * (a + b)) ** 2
        return num / np.abs(self.eps[j]) ** 2 * n0sq / self.k0**2


def tmm(eps, thickness_nm, wavelength_nm: float, angle_rad: float, pol: str) -> TmmResult:
    """Ebene Welle aus der Halbebene 0 (reelles ε₀, z < 0) auf den Schichtstapel 1 … N−2 (Dicken
    ``thickness_nm``, erste Grenzfläche bei z = 0) über der Halbebene N−1.

    Randbedingungen: Tangentiales E und H stetig, d. h. A + B und q (A − B) mit q = k_z (s) bzw.
    q = k_z/ε (p, Amplitude H_y). r folgt aus B_{N−1} = 0. R = |r|², T = Re q_{N−1}/q_0 · |t|².
    """
    eps = np.asarray(eps, complex)
    d = np.asarray(thickness_nm, np.float64)
    if eps.ndim != 1 or len(eps) < 2 or len(d) != len(eps) - 2:
        raise ValueError("tmm: eps (N,), thickness (N − 2,)")
    if pol not in ("s", "p"):
        raise ValueError("tmm: pol 's' oder 'p'")
    if eps[0].imag != 0 or eps[0].real <= 0:
        raise ValueError("tmm: Einfallsmedium braucht reelles ε > 0")
    k0 = 2.0 * np.pi / wavelength_nm
    kx = k0 * np.sqrt(eps[0].real) * np.sin(angle_rad)
    kz = _kz(eps, kx**2, k0)
    q = kz if pol == "s" else kz / eps
    lengths = np.concatenate([[0.0], d])  # Weg in Schicht j bis zur nächsten Grenzfläche

    def sweep(a0, b0):
        out = np.empty((len(eps), 2), complex)
        out[0] = (a0, b0)
        for j in range(len(eps) - 1):
            ph = np.exp(1j * kz[j] * lengths[j])
            a, b = out[j, 0] * ph, out[j, 1] / ph
            ratio = q[j] / q[j + 1]
            out[j + 1] = (
                0.5 * ((1 + ratio) * a + (1 - ratio) * b),
                0.5 * ((1 - ratio) * a + (1 + ratio) * b),
            )
        return out

    u, v = sweep(1.0, 0.0), sweep(0.0, 1.0)
    r = -u[-1, 1] / v[-1, 1]
    amp = u + r * v
    amp[-1, 1] = 0.0  # numerisch exakt null
    t = amp[-1, 0]
    z_bounds = np.concatenate([[0.0], np.cumsum(d)])
    return TmmResult(
        r=complex(r), t=complex(t), R=float(abs(r) ** 2),
        T=float((q[-1].real / q[0].real) * abs(t) ** 2), pol=pol, kx=float(kx), kz=kz, eps=eps,
        z_bounds=z_bounds, amp=amp, k0=k0,
    )  # fmt: skip


@dataclass
class TmmSolver:
    """Planarer Referenzsolver (ADR 0016 Punkt 4a): Einfallsmedium (Prisma, ``n_incident``) | feste
    Schichten ``substrate`` [(ε, Dicke nm), …] bis zur Oberfläche z = ``z_surface`` | Wasserspalt bis
    zur Gitterunterkante | Schichten des Beobachtergitters (lateral gemitteltes ⟨ε⟩ je z-Zelle) |
    Wasser. Die einfallende Welle hat die Intensität ``intensity_w_m2`` im Einfallsmedium; W = |E|²
    hängt nur von z ab und wird auf alle Knoten (x, y) übertragen.
    """

    wavelength_nm: float
    angle_deg: float
    pol: str
    n_incident: float
    substrate: list = field(default_factory=list)
    intensity_w_m2: float = 1e10  # 1 MW/cm²
    z_surface: float = 0.0

    def stack(self, eps: EpsGrid):
        """(ε je Schicht inkl. Halbebenen, Dicken, z der ersten Grenzfläche in Gitterkoordinaten)."""
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
        """W(z) [(V/nm)²] an Positionen z (Gitterkoordinaten)."""
        e, d, z_first = self.stack(eps)
        res = tmm(e, d, self.wavelength_nm, np.deg2rad(self.angle_deg), self.pol)
        e0 = field_amplitude_sq(self.intensity_w_m2, self.n_incident)
        return e0 * res.intensity(np.asarray(z, np.float64) - z_first)

    def solve(self, eps: EpsGrid, grid: ForceGrid) -> np.ndarray:
        w_z = self.field_profile(eps, grid.nodes(2))
        return np.broadcast_to(w_z[:, None, None], grid.zyx_shape).copy()


# --- Fixpunkt-Treiber ----------------------------------------------------------------------------


def eps_from_histograms(s_alpha, s_volume, voxel_volume: float, n_water: float) -> np.ndarray:
    """ε = n² je Voxel aus mittleren Histogrammen (Lorentz-Lorenz mit Wasser-Hintergrund, §6.3)."""
    f_w = np.maximum(0.0, 1.0 - np.asarray(s_volume, np.float64) / voxel_volume)
    lw = (n_water**2 - 1.0) / (n_water**2 + 2.0)
    big_l = 4.0 * np.pi / 3.0 * np.asarray(s_alpha, np.float64) / voxel_volume + f_w * lw
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(big_l < 1.0, (1.0 + 2.0 * big_l) / (1.0 - big_l), np.nan)


class SelfConsistentOptics:
    """Picard-Iteration mit Dämpfung (ADR 0016 Punkt 1, 5).

    Iteration k: MD-Segment unter W_k (``n_equil`` Schritte verworfen, dann ``n_sample`` Schritte mit
    dem ε-Observer), ⟨ε⟩_k über die Replicas ``replicas`` (Mittel der Histogramme, dann
    Lorentz-Lorenz), W_new = solver.solve(⟨ε⟩_k), W_{k+1} = (1 − β) W_k + β W_new.

    Konvergenzmaße (L2 über das Gitter): Δε = ‖⟨ε⟩_k − ⟨ε⟩_{k−1}‖ / ‖⟨ε⟩_k − ε_w‖ (relativ zum
    Kontrast, da ε_w den Absolutwert dominiert) und das Residuum ΔW = ‖W_new − W_k‖ / ‖W_k‖ (unabhängig
    von β, anders als die Schrittweite β ΔW). Konvergiert, wenn
    beide zwei Iterationen in Folge unter ``tol`` bzw. ``tol_w`` (Default ``tol``) liegen; ``tol`` muss
    über dem statistischen Fehler
    von ⟨ε⟩ liegen (der Verlauf enthält eine Rauschabschätzung ``eps_noise`` aus der Varianz über
    Frames, ohne Korrelationskorrektur, daher eine Untergrenze).

    W_0 ist die Lösung für reines Wasser (ungekoppeltes Feld). Mit ``checkpoint_dir`` wird nach jeder
    Iteration ``iter_{k:03d}.npz`` (Kontext, :func:`biooptomd.io.save_checkpoint`) und
    ``coupling_{k:03d}.npz`` (W, ⟨ε⟩, Verlauf) geschrieben, k = Zahl der abgeschlossenen
    Iterationen; :meth:`resume` setzt dort fort.
    """

    def __init__(self, ctx, solver: OpticalSolver, force_grid: ForceGrid, scale, *,
                 eps_shape, eps_origin, eps_spacing, alpha, volume, n_water: float,
                 n_equil: int, n_sample: int, stride: int = 100, beta: float = 0.5,
                 tol: float = 1e-2, tol_w: float | None = None, replicas=None, smearing: str = "gauss",
                 checkpoint_dir=None):  # fmt: skip
        if not 0.0 < beta <= 1.0:
            raise ValueError("beta in (0, 1]")
        if n_sample < stride or n_equil < 0:
            raise ValueError("n_sample >= stride, n_equil >= 0")
        self.ctx, self.solver, self.grid = ctx, solver, force_grid
        self.beta, self.tol = float(beta), float(tol)
        self.tol_w = self.tol if tol_w is None else float(tol_w)
        self.n_equil, self.n_sample = int(n_equil), int(n_sample)
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
        self.w = np.asarray(solver.solve(water, force_grid), np.float64)
        self.voxel = ctx.add_voxel_grid(
            self._values(self.w), force_grid.origin, force_grid.spacing,
            np.asarray(scale, np.float32), periodic_xy=force_grid.periodic_xy,
        )  # fmt: skip
        self.eps = None
        self.history: list[dict] = []
        self.checkpoint_dir = None if checkpoint_dir is None else Path(checkpoint_dir)

    def _values(self, w):
        if w.shape != self.grid.zyx_shape:
            raise ValueError(f"Solver lieferte {w.shape}, erwartet {self.grid.zyx_shape}")
        return values_from_potential(w, self.grid.spacing, periodic_xy=self.grid.periodic_xy)

    @property
    def iteration(self) -> int:
        return len(self.history)

    def _sample_eps(self):
        self.ctx.step(self.n_equil)
        self.observer.reset()
        self.ctx.step(self.n_sample)
        res = self.observer.result()
        s_a = res["s_alpha_mean"][self.replicas].mean(axis=0)
        s_v = res["s_volume_mean"][self.replicas].mean(axis=0)
        vk = float(np.prod(self.eps_spacing))
        eps = eps_from_histograms(s_a, s_v, vk, self.n_water)
        # Rauschabschätzung: Standardfehler von S_α über Frames (unkorreliert angenommen) → δε.
        n_frames = max(int(res["n_frames"][0]), 1)
        sem = np.sqrt(
            res["s_alpha_var"][self.replicas].mean(axis=0) / (n_frames * len(self.replicas))
        )
        lw_eps = (self.n_water**2 + 2.0) ** 2 / 3.0  # dε/dL bei ε_w
        noise = 4.0 * np.pi / 3.0 * lw_eps * sem / vk
        return eps, noise

    def step(self) -> dict:
        """Eine Iteration; gibt den Verlaufseintrag zurück."""
        eps, noise = self._sample_eps()
        contrast = eps - self.n_water**2
        c_norm = float(np.linalg.norm(contrast))
        if self.eps is None:
            d_eps = np.inf
        else:
            diff = float(np.linalg.norm(eps - self.eps))
            d_eps = 0.0 if diff == 0.0 else diff / max(c_norm, 1e-300)
        grid = EpsGrid(eps, self.eps_origin, self.eps_spacing, self.n_water)
        w_new = np.asarray(self.solver.solve(grid, self.grid), np.float64)
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
        }
        self.history.append(entry)
        if self.checkpoint_dir is not None:
            self._checkpoint()
        return entry

    def converged(self) -> bool:
        if len(self.history) < 2:
            return False
        return all(h["d_eps"] < self.tol and h["d_w"] < self.tol_w for h in self.history[-2:])

    def run(self, max_iter: int = 10) -> list:
        while self.iteration < max_iter and not self.converged():
            self.step()
        return self.history

    def _checkpoint(self) -> None:
        k = self.iteration
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        _io.save_checkpoint(self.ctx, self.checkpoint_dir / f"iter_{k:03d}.npz")
        with open(self.checkpoint_dir / f"coupling_{k:03d}.npz", "wb") as f:
            np.savez(f, w=self.w, eps=self.eps, history=json.dumps(self.history))

    def resume(self, iteration: int) -> None:
        """Setzt nach ``iteration`` abgeschlossenen Iterationen fort (Checkpoint-Dateien): W, ⟨ε⟩, Verlauf und
        Kontextzustand. Der Kontext muss gleich aufgebaut sein (inkl. dieses Treibers)."""
        if self.checkpoint_dir is None:
            raise ValueError("resume braucht checkpoint_dir")
        with np.load(self.checkpoint_dir / f"coupling_{iteration:03d}.npz") as d:
            self.w = np.array(d["w"])
            self.eps = np.array(d["eps"])
            self.history = json.loads(str(d["history"]))
        self.voxel.set_values(self._values(self.w))
        _io.load_checkpoint(self.ctx, self.checkpoint_dir / f"iter_{iteration:03d}.npz")
