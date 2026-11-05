"""Replica Exchange (Temperatur-Austausch auf dem Host, ROADMAP Phase 9.2, ADR 0011).

Jede Replica-Slot r trägt einen thermodynamischen Zustand k = state[r] (Temperatur T_k und, nach
OQ-10, die davon abhängige Abschirmung κ_k, ε_k). Ausgetauscht werden die Zustände, nicht die
Koordinaten; danach werden die Geschwindigkeiten mit sqrt(T_neu/T_alt) skaliert.

Weil die potentielle Energie über κ(T), ε_r(T) und γ(T) (Gouy-Chapman) selbst vom Zustand abhängt,
ist das Akzeptanzkriterium das des Hamiltonian-Austauschs:

    Δ = β_j U_j(x_a) + β_i U_i(x_b) − β_i U_i(x_a) − β_j U_j(x_b),   p_acc = min(1, e^{−Δ})

für Slot a im Zustand i und Slot b im Zustand j. Die Kreuzenergien U_j(x_a) liefert eine zusätzliche
Kraftberechnung mit den vertauschten Zuständen (alle Paare eines Versuchs gleichzeitig). Ohne
zustandsabhängige Parameter reduziert sich Δ auf (β_i − β_j)(U(x_b) − U(x_a)).

Paarschema: abwechselnd Nachbarpaare (0,1), (2,3), … und (1,2), (3,4), … der Temperaturleiter;
jeder einzelne Zug erfüllt die detaillierte Bilanz, die deterministische Abfolge die Balance.

Mehrere unabhängige Leitern in einem Kontext (``ladders=L``): Die Slots zerfallen in L
zusammenhängende Blöcke zu m = n_replicas/L; Block b trägt die Zustände b·m … b·m + m − 1 mit
denselben Temperaturen, getauscht wird nur innerhalb eines Blocks (z. B. Konvergenzprüfung mit
verschiedenen Startstrukturen bei vollem Durchsatz).
Zufallszahlen aus ``numpy.random.Generator`` (PCG64, Seed) ⇒ deterministisch.

Nicht zustandsabhängig: die Oberflächenladung bzw. Φ0 der Gouy-Chapman-Wand (κ folgt dem Zustand),
Voxelgitter. Observer messen je Slot, nicht je Zustand: Zeitreihen über ``state`` je Zyklus den
Zuständen zuordnen (z. B. per ``callback`` in :meth:`ReplicaExchange.run`).

Einheiten: T in K, κ in 1/nm, Energien in kJ/mol.
"""

from __future__ import annotations

import json
import warnings

import numpy as np

from . import nonbonded as _nonbonded
from . import units


class ReplicaExchange:
    """Temperatur-Replica-Exchange für einen ``SimulationContext``.

    Args:
        ctx: Kontext mit n_replicas Slots; Integrator und Kräfte bereits gesetzt. Der Cutoff r_c muss
            für alle Zustände gelten (ADR 0006: r_c = n_κ/min_k κ_k – ``build_context`` mit allen
            Temperaturen aufbauen).
        temperatures_k: T_k [K] der Zustände einer Leiter, aufsteigend, (n_replicas / ladders,).
        ionic_strength_molar: Ionenstärke [mol/L]; dann ε_k = ε_r,Wasser(T_k) und κ_k aus I, T_k, ε_k
            (OQ-10). Oder ``kappa``/``eps_r`` je Zustand explizit; ohne beides bleibt die Abschirmung
            unverändert (reiner Temperaturaustausch). ``kappa``/``eps_r`` je Zustand einer Leiter
            oder skalar.
        seed: Seed der Akzeptanzzufallszahlen.
        ladders: Anzahl unabhängiger Leitern L (s. o.); n_replicas muss durch L teilbar sein.

    Attributes:
        temperatures: T je globalem Zustand, (n_replicas,) (die Leiter L-mal wiederholt).
        attempts, accepted: je Paar (k, k+1) globaler Zustände, (n_replicas − 1,); Paare über eine
            Blockgrenze werden nie versucht.
    """

    def __init__(self, ctx, temperatures_k, ionic_strength_molar=None, kappa=None, eps_r=None,
                 seed: int = 0, ladders: int = 1):  # fmt: skip
        self.ctx = ctx
        n = ctx.n_replicas
        if ladders < 1 or n % ladders != 0:
            raise ValueError("ladders: Teiler von n_replicas")
        self.ladders = int(ladders)
        self.ladder_size = m = n // self.ladders
        ladder = np.asarray(temperatures_k, np.float64)
        if ladder.shape != (m,) or not np.all(np.diff(ladder) > 0):
            raise ValueError("temperatures_k: aufsteigend, eine je Replica einer Leiter")
        self.temperatures = np.tile(ladder, self.ladders)
        if ionic_strength_molar is not None:
            if kappa is not None or eps_r is not None:
                raise ValueError("entweder ionic_strength_molar oder kappa/eps_r")
            eps_r = _nonbonded.water_permittivity(self.temperatures)
            kappa = _nonbonded.debye_kappa(ionic_strength_molar, self.temperatures, eps_r)
        elif kappa is not None or eps_r is not None:
            kappa = None if kappa is None else self._per_state(kappa)
            eps_r = None if eps_r is None else self._per_state(eps_r)
        self.kappa = None if kappa is None else np.broadcast_to(np.asarray(kappa, np.float64), (n,))
        self.eps_r = None if eps_r is None else np.broadcast_to(np.asarray(eps_r, np.float64), (n,))
        if (self.kappa is None) != (self.eps_r is None):
            raise ValueError("kappa und eps_r nur gemeinsam")
        if self.kappa is not None and ctx.nonbonded_cutoff > 0:
            needed = _nonbonded.cutoff_from_kappa(self.kappa)[1]
            if ctx.nonbonded_cutoff < needed * (1 - 1e-6):
                warnings.warn(
                    f"r_c = {ctx.nonbonded_cutoff:.3f} nm < 5/min κ_k = {needed:.3f} nm (ADR 0006): "
                    "Kontext mit allen Temperaturen der Leiter aufbauen",
                    stacklevel=2,
                )
        self.state = np.arange(n)  # Zustand je Slot
        self.rng = np.random.default_rng(seed)
        self.n_cycles = 0
        self.attempts = np.zeros(n - 1, np.int64)  # je Nachbarpaar (k, k+1) der Leiter
        self.accepted = np.zeros(n - 1, np.int64)
        self._apply(self.state)
        ctx.compute_forces()

    # --- Zustände ----------------------------------------------------------------------------
    def _per_state(self, values):
        """Skalar oder (Leiterlänge,) → je globalem Zustand."""
        v = np.asarray(values, np.float64)
        return v if v.ndim == 0 else np.tile(np.broadcast_to(v, (self.ladder_size,)), self.ladders)

    def _apply(self, state):
        self.ctx.set_temperatures(self.temperatures[state].astype(np.float32))
        if self.kappa is not None:
            self.ctx.set_screening(
                self.kappa[state].astype(np.float32), self.eps_r[state].astype(np.float32)
            )

    def _potential(self):
        return self.ctx.energies()[:, 1:].sum(axis=1)

    @property
    def slot_of_state(self) -> np.ndarray:
        """Slot, in dem Zustand k gerade liegt, (n_replicas,)."""
        out = np.empty_like(self.state)
        out[self.state] = np.arange(len(self.state))
        return out

    def acceptance_ratio(self) -> np.ndarray:
        """Akzeptanzrate je Nachbarpaar (k, k+1); NaN ohne Versuche."""
        with np.errstate(invalid="ignore", divide="ignore"):
            return self.accepted / self.attempts

    # --- Austausch ---------------------------------------------------------------------------
    def attempt(self) -> np.ndarray:
        """Ein Austauschversuch aller Paare der aktuellen Parität. Erwartet Kräfte/Energien zum
        aktuellen Zustand (nach ``ctx.step`` gegeben). Returns: bool (n_pairs,) akzeptiert je
        Leiterpaar (False für nicht versuchte Paare)."""
        n, m = len(self.state), self.ladder_size
        parity = self.n_cycles % 2
        self.n_cycles += 1
        pairs = [(b * m + k, b * m + k + 1) for b in range(self.ladders)
                 for k in range(parity, m - 1, 2)]  # fmt: skip
        accepted = np.zeros(n - 1, bool)
        if not pairs:
            return accepted
        beta = 1.0 / (units.KB * self.temperatures)
        expected_t = self.temperatures[self.state].astype(np.float32)
        if not np.array_equal(self.ctx.temperatures(), expected_t):
            raise RuntimeError(
                "Temperaturen des Kontexts passen nicht zum Austauschzustand "
                "(nach load_checkpoint ohne remd=…?)"
            )
        if not self.ctx.forces_valid:
            self.ctx.compute_forces()
        u_old = self._potential()
        slot = self.slot_of_state
        trial = self.state.copy()
        for i, j in pairs:
            a, b = slot[i], slot[j]
            trial[a], trial[b] = j, i
        self._apply(trial)
        self.ctx.compute_forces()
        u_new = self._potential()
        final = self.state.copy()
        factor = np.ones(n)
        for i, j in pairs:
            a, b = slot[i], slot[j]
            delta = (
                beta[j] * u_new[a] + beta[i] * u_new[b] - beta[i] * u_old[a] - beta[j] * u_old[b]
            )
            if not np.isfinite(delta):
                raise RuntimeError(f"Replica Exchange: Δ nicht endlich (Paar {i}, {j})")
            self.attempts[i] += 1
            if delta <= 0.0 or self.rng.random() < np.exp(-delta):
                accepted[i] = True
                self.accepted[i] += 1
                final[a], final[b] = j, i
                factor[a] = np.sqrt(self.temperatures[j] / self.temperatures[i])
                factor[b] = np.sqrt(self.temperatures[i] / self.temperatures[j])
        if not np.array_equal(final, trial):
            self._apply(final)
            self.ctx.compute_forces()
        if np.any(factor != 1.0):
            self.ctx.scale_velocities(factor)
        self.state = final
        return accepted

    def run(self, n_cycles: int, steps_per_cycle: int, callback=None):
        """n_cycles × (steps_per_cycle Schritte, ein Austauschversuch). ``callback(self)`` optional
        nach jedem Zyklus (z. B. Zustandsprotokoll)."""
        for _ in range(n_cycles):
            self.ctx.step(steps_per_cycle)
            self.attempt()
            if callback is not None:
                callback(self)

    # --- Checkpoint ---------------------------------------------------------------------------
    def state_dict(self) -> dict:
        """Zustand für Checkpoints (zusätzlich zu ``biooptomd.io.save_checkpoint``)."""
        return {
            "state": self.state.copy(),
            "n_cycles": self.n_cycles,
            "attempts": self.attempts.copy(),
            "accepted": self.accepted.copy(),
            "rng": self.rng.bit_generator.state,
        }

    def to_arrays(self) -> dict:
        """Austauschzustand als reine NumPy-Arrays (für ``np.savez`` ohne Pickle); der RNG-Zustand
        (128-bit-Ganzzahlen) als JSON-Bytes."""
        d = self.state_dict()
        return {
            "remd_state": d["state"], "remd_n_cycles": np.int64(d["n_cycles"]),
            "remd_attempts": d["attempts"], "remd_accepted": d["accepted"],
            "remd_rng": np.frombuffer(json.dumps(d["rng"]).encode(), np.uint8),
            "remd_temperatures": self.temperatures,
        }  # fmt: skip

    def from_arrays(self, a) -> None:
        """Gegenstück zu :meth:`to_arrays`."""
        if not np.array_equal(np.asarray(a["remd_temperatures"]), self.temperatures):
            raise ValueError("Checkpoint: andere Temperaturleiter")
        self.load_state_dict({
            "state": np.asarray(a["remd_state"]), "n_cycles": int(a["remd_n_cycles"]),
            "attempts": np.asarray(a["remd_attempts"]), "accepted": np.asarray(a["remd_accepted"]),
            "rng": json.loads(bytes(np.asarray(a["remd_rng"], np.uint8)).decode()),
        })  # fmt: skip

    def load_state_dict(self, d: dict) -> None:
        """Stellt den Austauschzustand wieder her und setzt die Zustandsparameter je Slot. Danach
        ``biooptomd.io.load_checkpoint`` (Temperaturen identisch) oder direkt weiterrechnen."""
        self.state = np.asarray(d["state"]).copy()
        self.n_cycles = int(d["n_cycles"])
        self.attempts = np.asarray(d["attempts"]).copy()
        self.accepted = np.asarray(d["accepted"]).copy()
        self.rng.bit_generator.state = d["rng"]
        self._apply(self.state)
        self.ctx.compute_forces()
