# Projektstatus

> Diese Datei wird am Ende **jeder** Arbeitssitzung aktualisiert (AGENTS.md §2.4).

**Aktive Phase:** 2 – Langevin-BAOAB & Basis-Umgebungsplugins
(Implementierung vollständig, alle Abnahmetests grün auf RTX 3090; **Abnahme durch den Projektinhaber offen**,
siehe OQ-10)
**Referenz-Hardware:** RTX 6000 Ada 48 GB (CC 8.9), Host 1 TB RAM

## Abgeschlossen
- Phase 0: Repository-Gerüst, Dokumentation, Agenten-Anweisungen (Initial-Setup).
- Phase 1 (PR #1, Branch `phase-1/build-devicestate-bindings`): Build, `DeviceState`, `SimulationContext`,
  Plugin-Interfaces, Kernels `clear_forces`/`fixed_point_to_float`/`wrap_xy`, nanobind-Modul mit NumPy-
  und DLPack-Export. **Abgenommen** am 2026-11-03 durch den Projektinhaber auf RTX 3090 (CC 8.6);
  ein Kontrolllauf auf der RTX 6000 Ada wird nachgeholt.

## Phase 2 – erledigt (Branch `phase-2/langevin-surface-restraint`, baut auf Phase 1 auf)
Commits `146346f` (core), `3dc7cb6` (forces), `bde78ab` (bindings), `6c810ad` (tests), `c8aadff` (style):
1. `LangevinIntegrator` (BAOAB, `include/biooptomd/langevin.hpp`, `src/core/langevin.cu`), Philox4x32-10
   zählerbasiert (`src/core/philox.cuh`, KAT-geprüft), T pro Replica, γ = 0 ⇒ NVE.
   `SimulationContext::step(n)` nach ARCHITECTURE.md §1; Simulationszeit fortlaufend summiert.
2. `PlanarSurfaceForce` (`src/forces/planar_surface.*`): 9-3-Wand pro Oberflächen-Typ, Clamp-Zähler,
   Zähler für unbekannte Typen, optional linearisiertes Gouy-Chapman.
3. `HarmonicRestraintPullingForce` (`src/forces/harmonic_restraint.*`): Positions-Restraints (Minimum Image,
   Achsenmaske), Schwerpunkt-Pulling über ungewrappte Koordinaten, Arbeit pro Replica in FP64 auf dem Gerät,
   Umbrella = v 0 mit R₀ pro Replica. Gewichte folgen `set_masses`.
4. Energieausgabe: `energies()` (E_kin + E_pot je Term) und Observer `EnergyTracker` (VRAM-Puffer).
- NumPy-Referenzen: `tests/reference/{philox_ref,surface_ref,restraint_ref}.py`.

## Getestet (RTX 3090, Windows 11, MSVC 17.14, CUDA 12.9, Python 3.11, nanobind 3.1)
- `pytest`: 30/30 grün (Phase 1: 10, Phase 2: 20 inkl. aller 7 Abnahmetests und 5-Seed-`slow`-Test).
- `compute-sanitizer --tool memcheck`: 0 Fehler (Phase 1 komplett, Phase 2 ohne die langen Statistiktests,
  die unter memcheck > 10 min dauern; alle Kernels sind abgedeckt).
- `compute-sanitizer --tool racecheck` (Shared-Memory-Kernels `k_energy_frame`, `k_com_pulling`, minimales
  Skript mit Pulling + Wand + EnergyTracker, 3 Schritte): 0 Hazards. Über pytest ist racecheck zu langsam (> 30 min).
- Build Release ohne Warnungen; ruff und clang-format sauber.
- Ein unabhängiger Review-Agent hat BAOAB, Philox-Zähler, Formeln, Minimum Image, Arbeit und Warp-Shuffle
  geprüft; gefundene Punkte (dt-Wechsel, veraltete Kräfte nach DLPack-Schreibzugriff/`set_masses`,
  Box-Muller-Randfall, zu lockere NVE-Toleranz) sind behoben und durch Tests abgedeckt.

## Rote Tests
– keine.

## Offene Punkte / Hinweise für den nächsten Agenten
- **OQ-10 gelöst** (Entscheidung Projektinhaber, ADR 0006, Status `Proposed` bis zur formalen Annahme):
  Wand-Clamp ζ_c = max(0.1 σ, ζ_F) mit |F(ζ_F)| = 10⁶ kJ/mol/nm; Fixed-Point-Sättigung pro Beitrag auf ±2³⁰
  (±2²⁶ pro Lane in `warp_add_energy`); Statuswort `d_status` mit Bits Sättigung / NaN / Wrap-Bereich,
  lesbar über `ctx.status()`. Tests: `test_surface_vs_numpy` (jetzt alle Kräfte), `test_fixed_point_saturation_and_status`,
  `test_wrap_range_status`.
- OQ-08 (Wrap-„exakt“) und OQ-09 (Gouy-Chapman für ζ < 0) bitte bestätigen.
- Explodierende Trajektorien setzen jetzt Statusbits; `step()` prüft sie nicht selbst (kein Sync im Hot-Path) –
  Aufrufer sollten `ctx.status()` periodisch abfragen. Zwei gesättigte gleichgerichtete Beiträge können die
  int64-Summe weiterhin überlaufen lassen (dann ist das Flag gesetzt).
- **Beobachtung (ungeklärt):** Unter `compute-sanitizer --tool memcheck` meldet nanobind beim Beenden von pytest
  „leaked 15 instances“ (SimulationContext, Plugin-Handles), wenn mehrere Phase-2-Tests in einem Prozess laufen
  (`-k "philox or surface_vs or restraint_vs or nve or integrator_api or pulling_follows or gouy or determinism"`).
  Ohne Sanitizer (gleiche Auswahl, jeder Test einzeln) und mit einem einzelnen Test unter Sanitizer: kein Leck.
  memcheck selbst meldet 0 Fehler. Nächster Schritt: `gc.get_referrers` auf die Kontexte am Testende.
- Atomzahl ist pro Batch einheitlich; Topologie-Offsets pro Replica (ADR 0005) kommen in Phase 3.
- Statistische Tests laufen mit festen Seeds; `test_jarzynski_harmonic` liegt mit Seed 21 bei z ≈ −3
  (innerhalb 4σ; über 40 Seeds unauffällig, mittleres z ≈ 0).

## Nächster konkreter Schritt
1. Projektinhaber: ADR 0006 auf `Accepted` setzen, Phase 2 abnehmen, PRs #1 und #2 mergen.
2. Danach Phase 3, Aufgabe 1: NumPy-Referenz `tests/reference/bonded_ref.py` (Bonds, Angles über atan2,
   Torsionen, 1-4) mit Finite-Differenzen-Test.
3. Nachholen: Kontrolllauf auf der RTX 6000 Ada (`pytest`, `compute-sanitizer`), CUDA-Version eintragen.

## Sitzungslog
| Datum | Agent/Person | Zusammenfassung |
|---|---|---|
| 2026-11-03 | Claude (Setup) | Phase 0 angelegt. Build-Dateien sind ungetestete Entwürfe. |
| 2026-11-03 | Claude Opus 5.5 | Phase 1 implementiert und auf RTX 3090/Windows getestet (PR #1). Projektinhaber hat Phase 1 auf RTX 3090 abgenommen. |
| 2026-11-03 | Claude Opus 5.5 | Phase 2 implementiert (PR #2); OQ-10 nach Entscheidung des Projektinhabers per ADR 0006 umgesetzt, 30/30 Tests grün. |
