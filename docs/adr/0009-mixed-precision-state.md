# ADR 0009: Mixed-Precision-Zustand (FP64-Geschwindigkeiten, Positionskorrektur)

- **Status:** Proposed (Entscheidung des Projektinhabers am 2026-11-03; formale Annahme ausstehend)
- **Datum:** 2026-11-03
- **Ändert:** ARCHITECTURE.md §2.2 (`d_vel`, neu `d_pos_corr`)

## Kontext
Positionen und Geschwindigkeiten lagen bisher als `float4` vor. Jede Integrator-Aktualisierung rundet den
Zustand auf FP32; die Rundungsfehler bilden einen Random Walk der Energie. Messung (DNA-Trinukleotid,
92 Atome, nur kovalente Terme, NVE, Δt = 0.5 fs, 100 ps, 8 Replicas, Regressionssteigung von E_tot):

| Molekül bei | ulp(x) | Drift [kJ/mol/ps/Atom] |
|---|---|---|
| (1.2, 1.2, 1.2) nm | 1.2·10⁻⁷ nm | 2·10⁻⁵ (Mittel) |
| (5, 5, 3) nm | 4.8·10⁻⁷ nm | 1.6·10⁻⁴ (Mittel), bis 5·10⁻⁴ |

Biosensor-Boxen haben 8–10 nm Kantenlänge; die Phase-3-Grenze von 10⁻⁴ wäre dort nicht einzuhalten.
Ein NumPy-Prototyp bestätigt, dass Kräfte aus FP32-gerundeten Positionen allein (bei präzisem Zustand)
keinen Random Walk erzeugen, der die Grenze verletzt.

## Entscheidung
Nach dem Vorbild von OpenMMs „mixed precision“:
1. `d_vel` wird `double4` (vx, vy, vz [nm/ps], 1/m [1/amu]).
2. Neu `d_pos_corr` (`float4`, xyz = Korrektur, w = 0): Die Position ist x = d_pos.xyz + d_pos_corr.xyz
   (≈ 48 Bit Mantisse). `d_pos` bleibt die FP32-Rundung von x und wird von allen Kraft- und
   Observer-Kernels unverändert gelesen.
3. Integrator, `wrap_xy` und der Schwerpunkt im Pulling rechnen mit x in FP64 und schreiben
   d_pos = fl32(x), d_pos_corr = fl32(x − d_pos).
4. Kraft-Kernels bleiben FP32 (AGENTS.md: Kernel-Arithmetik FP32), Kräfte Fixed-Point (ADR 0002).

## Alternativen
- Positionen komplett in FP64: alle Kraft-Kernels müssten doppelt so viel Speicher lesen oder konvertieren.
- Fixed-Point-Positionen (int32/int64): genaue Wraps, aber jede Kraftkernel-Ladung braucht Konvertierung.
- Molekül nahe am Ursprung halten: keine Lösung für ausgedehnte Oberflächen-Boxen.
- Build-Option FP32/mixed: doppelte Testmatrix ohne erkennbaren Nutzen.

## Konsequenzen
- +16 Byte (Geschwindigkeit) und +16 Byte (Korrektur) pro Atomplatz; wenige FP64-Operationen pro Atom und
  Schritt (vernachlässigbar gegenüber den Kraft-Kernels, auch bei FP64 = 1/64 FP32).
- `wrap_xy` rechnet in FP64: die ungewrappte Rekonstruktion x_w + n·L ist exakt (löst OQ-11). Die FP32-
  Komponente d_pos.x liegt in [0, L]; sie kann durch Rundung genau L sein, die FP64-Position liegt in [0, L).
- Python: `velocities()` liefert float64; neu `position_corrections()` und `positions_precise()`.
- Tests: `test_phase1.py::test_wrap_xy` (FP64-Wrap, exakte Rekonstruktion),
  `test_phase3.py::test_nve_energy_conservation` (100 ps bei realistischen Koordinaten).
