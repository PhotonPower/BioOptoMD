"""I/O: Checkpoints und Trajektorien (ROADMAP Phase 7, Aufgabe 3).

- :func:`save_checkpoint` / :func:`load_checkpoint`: bitgenaue Wiederherstellung des dynamischen
  Zustands (Positionen inkl. FP64-Korrektur, Geschwindigkeiten, Imagezähler, Schrittzähler = Philox-
  Zähler, Zeit) plus Box, Temperaturen und Integrator-Parametern (dt, γ, Seed). Kräfte, Wände,
  Restraints und Observer gehören zum Aufbau des Systems und werden nicht gespeichert; der Kontext
  muss vor dem Laden genauso aufgebaut sein. Observer-Puffer und die Pulling-Arbeit sind nicht Teil
  des Checkpoints.
- :class:`DCDWriter`: eine DCD-Datei je Replica (CHARMM/NAMD-Format wie OpenMM: Å, Box je Frame).
- :class:`HDF5Writer`: eine Datei für alle Replicas (``positions`` (n_frames, n_replicas, n_atoms, 3)
  in nm, ``step``, ``time`` [ps], ``box`` [nm]); benötigt h5py.
- :func:`read_dcd`: Rückleser (für Tests und Analyse).

Positionen werden in FP64 (``positions_precise``) gelesen und als float32 geschrieben; ``unwrap=True``
schreibt ungewrappte Koordinaten x + n_x L_x (Imagezähler, ADR 0003).
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

CHECKPOINT_VERSION = 1
_STATE_KEYS = ("pos", "pos_corr", "vel", "image")


# --- Checkpoints ------------------------------------------------------------------------------


def save_checkpoint(ctx, path) -> None:
    """Schreibt den Zustand von ``ctx`` nach ``path`` (NumPy-``.npz``, keine Pickles)."""
    st = ctx.save_state()
    meta = {
        "version": np.int64(CHECKPOINT_VERSION),
        "n_replicas": np.int64(ctx.n_replicas),
        "n_atoms": np.int64(ctx.n_atoms),
        "step": np.int64(st["step"]),
        "time": np.float64(st["time"]),
        "previous_time": np.float64(st["previous_time"]),
        "box": ctx.box(),
        "temperatures": ctx.temperatures(),
    }
    if ctx.has_integrator:
        p = ctx.integrator_params
        meta["dt"] = np.float64(p["dt"])
        meta["friction"] = np.float64(p["friction"])
        meta["seed"] = np.uint64(p["seed"])
    with open(path, "wb") as f:
        np.savez(f, **{k: st[k] for k in _STATE_KEYS}, **meta)


def load_checkpoint(ctx, path, restore_integrator: bool = True) -> int:
    """Stellt einen Checkpoint in einem gleich aufgebauten ``ctx`` wieder her.

    Reihenfolge: Box (setzt die Imagezähler zurück), Temperaturen, Integrator, dann der dynamische
    Zustand. Danach liefert ``ctx.step(n)`` bitgenau dieselbe Trajektorie wie der ursprüngliche Lauf.

    Returns:
        Schrittzähler des Checkpoints.
    """
    with np.load(path, allow_pickle=False) as d:
        if int(d["version"]) != CHECKPOINT_VERSION:
            raise ValueError(f"Checkpoint-Version {int(d['version'])} nicht unterstützt")
        if int(d["n_replicas"]) != ctx.n_replicas or int(d["n_atoms"]) != ctx.n_atoms:
            raise ValueError("Checkpoint passt nicht zum Kontext (n_replicas, n_atoms)")
        ctx.set_box(np.ascontiguousarray(d["box"], np.float32))
        ctx.set_temperatures(np.ascontiguousarray(d["temperatures"], np.float32))
        if restore_integrator and "dt" in d:
            ctx.set_integrator(float(d["dt"]), float(d["friction"]), int(d["seed"]))
        ctx.restore_state(
            np.ascontiguousarray(d["pos"]), np.ascontiguousarray(d["pos_corr"]),
            np.ascontiguousarray(d["vel"]), np.ascontiguousarray(d["image"]),
            int(d["step"]), float(d["time"]), float(d["previous_time"]),
        )  # fmt: skip
        return int(d["step"])


# --- Trajektorien -----------------------------------------------------------------------------


def _frame_positions(ctx, unwrap: bool) -> np.ndarray:
    """(n_replicas, n_atoms, 3) [nm] in FP64."""
    x = ctx.positions_precise()[:, : ctx.n_atoms, :3]
    if unwrap:
        img = ctx.images()[:, : ctx.n_atoms].astype(np.float64)
        box = ctx.box().astype(np.float64)
        x = x.copy()
        x[..., 0] += img[..., 0] * box[:, None, 0]
        x[..., 1] += img[..., 1] * box[:, None, 1]
    return x


def _record(fmt: str, *values) -> bytes:
    body = struct.pack("<" + fmt, *values)
    return struct.pack("<i", len(body)) + body + struct.pack("<i", len(body))


class DCDWriter:
    """Eine DCD-Datei je Replica: ``pattern.format(replica=r)``.

    Args:
        ctx: Kontext.
        pattern: Pfadmuster mit ``{replica}``, z. B. ``"traj_{replica:02d}.dcd"``.
        interval: Schritte zwischen zwei Frames (nur Kopfzeile, NSAVC).
        unwrap: ungewrappte Koordinaten schreiben.
    """

    def __init__(self, ctx, pattern: str, interval: int = 1, unwrap: bool = False):
        self.ctx, self.unwrap, self.n_frames = ctx, unwrap, 0
        self.paths = [Path(pattern.format(replica=r)) for r in range(ctx.n_replicas)]
        # Dateien bleiben bis close() offen (Frames werden fortlaufend angehängt).
        self._files = [open(p, "wb") for p in self.paths]  # noqa: SIM115
        dt_ps = ctx.dt if ctx.has_integrator else 0.0
        # Zeiteinheit der DCD: AKMA (1 AKMA = 0.04888821 ps), wie CHARMM/OpenMM
        delta = dt_ps / 0.04888821
        self._first_step = ctx.current_step
        for f in self._files:
            f.write(self._header(0, delta, interval))
            f.write(_record("i80s", 1, b"Created by BioOptoMD".ljust(80)))
            f.write(_record("i", ctx.n_atoms))
        self._delta, self._interval = delta, interval

    def _header(self, n_frames: int, delta: float, interval: int) -> bytes:
        icntrl = [0] * 20
        icntrl[0], icntrl[1], icntrl[2] = n_frames, self._first_step, interval
        icntrl[3] = interval * n_frames
        icntrl[10], icntrl[19] = 1, 24  # Box je Frame, CHARMM-Version 24
        ints = struct.pack("<20i", *icntrl)
        ints = ints[:36] + struct.pack("<f", delta) + ints[40:]  # ICNTRL[9] = Δt (float)
        body = b"CORD" + ints
        return struct.pack("<i", len(body)) + body + struct.pack("<i", len(body))

    def write(self) -> None:
        """Schreibt den aktuellen Frame aller Replicas (synchronisiert)."""
        x = _frame_positions(self.ctx, self.unwrap) * 10.0  # nm → Å
        box = self.ctx.box().astype(np.float64) * 10.0
        self.n_frames += 1
        for r, f in enumerate(self._files):
            # Einheitszelle (A, γ, B, β, α, C); z nicht periodisch ⇒ C = 0
            f.write(_record("6d", box[r, 0], 90.0, box[r, 1], 90.0, 90.0, 0.0))
            for d in range(3):
                f.write(_record(f"{self.ctx.n_atoms}f", *x[r, :, d].astype(np.float32)))
            pos = f.tell()
            f.seek(0)
            f.write(self._header(self.n_frames, self._delta, self._interval))
            f.seek(pos)
            f.flush()

    def close(self) -> None:
        for f in self._files:
            f.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def read_dcd(path):
    """Liest eine mit :class:`DCDWriter` geschriebene Datei.

    Returns:
        (positions float32 (n_frames, n_atoms, 3) [nm], box float64 (n_frames, 2) [nm]).
    """
    data = Path(path).read_bytes()
    off = 0

    def record():
        nonlocal off
        (n,) = struct.unpack_from("<i", data, off)
        body = data[off + 4 : off + 4 + n]
        if struct.unpack_from("<i", data, off + 4 + n)[0] != n:
            raise ValueError("DCD: Record-Länge inkonsistent")
        off += n + 8
        return body

    head = record()
    if head[:4] != b"CORD":
        raise ValueError("DCD: keine CORD-Datei")
    icntrl = struct.unpack("<20i", head[4:84])
    n_frames, has_box = icntrl[0], icntrl[10]
    record()  # Titel
    (n_atoms,) = struct.unpack("<i", record())
    pos = np.empty((n_frames, n_atoms, 3), np.float32)
    box = np.zeros((n_frames, 2))
    for k in range(n_frames):
        if has_box:
            cell = struct.unpack("<6d", record())
            box[k] = (cell[0] / 10.0, cell[2] / 10.0)
        for d in range(3):
            pos[k, :, d] = np.frombuffer(record(), "<f4") / np.float32(10.0)
    return pos, box


class HDF5Writer:
    """Alle Replicas in einer HDF5-Datei (erweiterbare Datensätze, float32 positions in nm).

    Datensätze: ``positions`` (n_frames, n_replicas, n_atoms, 3) [nm], ``step`` (n_frames,),
    ``time`` (n_frames,) [ps], ``box`` (n_frames, n_replicas, 2) [nm]; Attribut ``unwrapped``.
    """

    def __init__(self, ctx, path, unwrap: bool = False, compression: str | None = "gzip"):
        import h5py

        self.ctx, self.unwrap = ctx, unwrap
        r, n = ctx.n_replicas, ctx.n_atoms
        self._f = h5py.File(path, "w")
        self._f.attrs["unwrapped"] = unwrap
        self._f.attrs["units"] = "nm, ps"
        kw = {"compression": compression} if compression else {}
        self._pos = self._f.create_dataset(
            "positions", (0, r, n, 3), np.float32, maxshape=(None, r, n, 3),
            chunks=(1, r, n, 3), **kw,
        )  # fmt: skip
        self._step = self._f.create_dataset("step", (0,), np.int64, maxshape=(None,))
        self._time = self._f.create_dataset("time", (0,), np.float64, maxshape=(None,))
        self._box = self._f.create_dataset("box", (0, r, 2), np.float32, maxshape=(None, r, 2))

    def write(self) -> None:
        """Hängt den aktuellen Frame an (synchronisiert)."""
        k = self._pos.shape[0]
        for ds in (self._pos, self._step, self._time, self._box):
            ds.resize(k + 1, axis=0)
        self._pos[k] = _frame_positions(self.ctx, self.unwrap).astype(np.float32)
        self._step[k] = self.ctx.current_step
        self._time[k] = self.ctx.time
        self._box[k] = self.ctx.box()
        self._f.flush()

    def close(self) -> None:
        self._f.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
