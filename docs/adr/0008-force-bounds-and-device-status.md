# ADR 0008: Kraftschranke am Wand-Clamp, Fixed-Point-Sättigung und Geräte-Statusflags

- **Status:** Proposed (Entscheidung des Projektinhabers zu OQ-13 am 2026-11-03; formale Annahme ausstehend)
- **Datum:** 2026-11-03
- **Ergänzt:** ADR 0002

## Kontext
ADR 0002 akkumuliert Kräfte und Energien als int64 mit Skala 2³² und setzt |F| < 2³¹ voraus. PHYSICS_MODEL §4.1
clampte die 9-3-Wand bei ζ_c = 0.1 σ. Dort ist |F| ≈ 1.2·10⁹ ε/ζ_c ≈ 10¹¹ kJ/mol/nm, also weit außerhalb des
darstellbaren Bereichs. `__float2ll_rn` sättigt außerhalb von int64, die anschließende Ganzzahlsumme kann aber
modulo 2⁶⁴ überlaufen und das Vorzeichen kippen; das ist in `test_phase2.py` nachgewiesen (OQ-13). NaN-Beiträge
werden zu einem undefinierten Wert. Eine explodierende Simulation bleibt dadurch unbemerkt.

## Entscheidung
1. **Kraftschranke am Clamp.** `PlanarSurfaceForce` clampt bei ζ_c,t = max(clamp_factor·σ_t, ζ_F,t), wobei ζ_F,t
   der Abstand mit |F_wall(ζ_F,t)| = F_max ist (repulsiver Ast, Bisektion auf dem Host in FP64, pro Typ).
   Default F_max = 10⁶ kJ/mol/nm. Damit ist die Wandkraft eines Atoms nie größer als F_max.
2. **Sättigung pro Beitrag.** Jede Umwandlung float → Fixed-Point begrenzt |v| auf 2³⁰ (kJ/mol/nm bzw. kJ/mol).
   NaN-Beiträge werden zu 0. Beides setzt ein Bit im Geräte-Statuswort. In der warpweiten Energiesumme
   (`warp_add_energy`) gilt pro Lane ±2²⁶, damit die Summe über 32 Lanes 2³¹ nicht überschreitet.
3. **Geräte-Statuswort.** `DeviceState::d_status` (ein `unsigned int`, `atomicOr`) mit den Bits
   `FIXED_POINT_SATURATED` (1), `NON_FINITE` (2), `WRAP_RANGE` (4, |x/L| ≥ 2²⁰ in `wrap_xy`).
   Lesen nur auf Anfrage (`status()`, `clear_status()`); kein Host-Sync im Hot-Path (AGENTS.md Regel 11/12).

## Alternativen
- Nur Kraftschranke: schützt die Wand, aber nicht künftige Kraftterme und nicht NaN.
- Nur Sättigung: Wand-Clamp erzeugt dann regulär gesättigte Werte; das Flag wäre bei jedem Wandkontakt gesetzt.
- Prüfung und Abbruch in jedem Schritt: erfordert Host-Synchronisation pro Schritt (Regel 11/12).
- Sättigung der Summe statt pro Beitrag: mit Ganzzahl-Atomics nicht atomar machbar.

## Konsequenzen
- Einzelbeiträge sind auf ±2³⁰ begrenzt; die Summe kann erst bei ≥ 2 gesättigten gleichgerichteten Beiträgen
  überlaufen – dann ist das Flag aber bereits gesetzt.
- Bitidentität bleibt erhalten (Sättigung ist deterministisch).
- Wandkraft und -energie unterhalb ζ_c sind konstant (wie bisher spezifiziert), aber jetzt endlich darstellbar.
- Tests: `test_phase2.py::test_surface_vs_numpy` (Clamp mit Kraftschranke, alle Kräfte verglichen),
  `test_phase2.py::test_fixed_point_saturation_and_status`, `test_phase2.py::test_wrap_range_status`.
