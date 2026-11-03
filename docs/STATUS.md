# Projektstatus

> Diese Datei wird am Ende **jeder** Arbeitssitzung aktualisiert (AGENTS.md §2.4).

**Aktive Phase:** 6 – In-situ Observer & VoxelGridForce3D
**Phase 5 abgenommen** am 2026-11-03 durch den Projektinhaber (RTX 3090).
**Referenz-Hardware:** RTX 6000 Ada 48 GB (CC 8.9), Host 1 TB RAM

## Abgeschlossen
- Phase 0: Repository-Gerüst, Dokumentation, Agenten-Anweisungen.
- Phase 1 (PR #1): Build, `DeviceState`, `SimulationContext`, Plugin-Interfaces, Kern-Kernels, nanobind-Modul.
- Phase 2 (PR #2): Langevin-BAOAB, `PlanarSurfaceForce`, `HarmonicRestraintPullingForce`, `EnergyTracker`,
  ADR 0006.
- Phase 3 (PR #3): `BondedForce`, präzises Minimum Image, ADR 0007 (Mixed Precision), NVE-Protokoll 100 ps.
  Alle drei vom Projektinhaber am 2026-11-03 auf RTX 3090 abgenommen.

- Phase 4 (PR #4): Tiled All-Pairs (LJ + Debye-Hückel, Switching), Builder `biooptomd.nonbonded`.
  Abgenommen 2026-11-03.

## Phase 5 – erledigt (Branch `phase-5/constraints-hmr`, baut auf Phase 4 auf)
1. HMR im Builder: `biooptomd.constraints.repartition_hydrogen_mass` (H → 3.024 amu, Gesamtmasse erhalten).
2. SHAKE/RATTLE auf H-Bindungen (`src/core/constraints.*`): ein Thread pro Cluster (Schweratom + ≤ 3 H),
   FP64 auf dem Mixed-Precision-Zustand, Topologie pro Replica (ADR 0005), integriert in BAOAB
   (B, RATTLE, A+SHAKE, O, RATTLE, A+SHAKE, Kräfte, B, RATTLE); ohne Constraints unveränderter
   fusionierter Kernel. `constraint_topology` entfernt constrainte Bindungen aus dem Bindungsterm.
   Neues Statusbit `STATUS_CONSTRAINT_FAILURE`.
3. N_dof = 3·N_mobile − N_constraints in `degrees_of_freedom()`.
- Referenz `tests/reference/constraints_ref.py`; GPU-Schritt = NumPy auf 1e-9 nm.
- **Befund:** BAOAB-Kinetiktemperatur bei 4 fs um ~10 % zu niedrig (268 K bei 300 K), O(Δt²)-Bias, kein
  Fehler (Konfigurationen korrekt; ⟨E_pot⟩ unabhängig von Δt). Dokumentiert in PHYSICS_MODEL §1, OQ-11.
  Der Stabilitätstest vergleicht daher die E_pot-Verteilung.
- Durchsatz: ≈ 3 400 Schritte/s bei 4 × 1558 Atomen, 4 fs ⇒ ≈ 1.2 µs/Tag je Replica.

## Getestet (RTX 3090, Windows 11, MSVC 17.14, CUDA 12.9, Python 3.11, nanobind 3.1, OpenMM 8.6.1)
- `pytest`: 52/52 grün (46 schnell + 6 `slow`). Phase 5: `test_constraint_tolerance`,
  `test_equipartition_with_constraints`, `test_stability_4fs` (slow, 148 s) + Builder- und Schritt-Referenztest.
- `compute-sanitizer --tool memcheck` (Phase 5): 0 Fehler.

## Phase 5 – Nacharbeit nach Review (2026-11-04)
- `test_stability_4fs` statistisch neu aufgebaut (AGENTS.md Regel 8): Review maß im alten Fenster 50–250 ps
  noch Relaxation (Drift ≈ 0.65σ) und korrelierte 5-ps-Samples. Jetzt: ein Duplex, 8 unabhängige Replicas,
  beide Läufe 1 ns, Auswertung ab 500 ps; ⟨E_pot⟩-Vergleich mit σ aus Replica-Mittelwerten (4σ) und KS auf
  50-ps-Samples. Grün (330 s, `slow`).
- `inv_mass = 0` in Clustern: SHAKE bewegt solche Atome nicht mehr, RATTLE überspringt Nenner 0 (kein NaN).
- `repartition_hydrogen_mass` lehnt H-Atome mit mehreren Bindungen ab.
- Bekannte Testlücke (vom Review als physikalisch irrelevant eingestuft): Das RATTLE nach dem ersten B und
  nach O ist durch keinen Test einzeln abgesichert (die SHAKE-Geschwindigkeitskorrektur kompensiert es).

## Rote Tests
– keine.

## Offene Punkte / Hinweise für den nächsten Agenten
- Nsight Compute nachholen, sobald GPU-Counter freigeschaltet sind (Befehl in `benchmarks/README.md`).
- ADR 0006 und ADR 0007 auf `Accepted` setzen; OQ-09 bestätigen; OQ-03 (Fluorophor-Parameter) offen.
- Tiled All-Pairs prüft alle Tile-Paare (keine Cutoff-Vorauswahl); Optimierung/Cell-List später (Phase 8/9).
- nanobind-„leaked instances“ nur unter compute-sanitizer mit mehreren Tests pro Prozess (siehe frühere Notiz).
- Explodierende Trajektorien setzen Statusbits; `step()` prüft sie nicht selbst.
- clang-format: CI (ubuntu) nutzt Version 21, lokal (pip) Version 18; sie richten nachgestellte Kommentare in
  Structs unterschiedlich aus. Vor dem Push mit beiden prüfen oder Kommentare über die Member setzen.

## Nächster konkreter Schritt
1. Projektinhaber: Phase 5 abnehmen; OQ-11 (Temperaturschätzer) entscheiden; PRs #1 → … → Phase 5 mergen.
2. Phase 6, Aufgabe 1: Observer `DipoleAndDistanceTracker`, `FRETTracker`, `RefractiveIndexProfiler`
   (PHYSICS_MODEL §6), zuerst NumPy-Referenzen.
3. Nachholen: Kontrolllauf auf der RTX 6000 Ada; Nsight Compute (Counter-Freigabe).

## Sitzungslog
| Datum | Agent/Person | Zusammenfassung |
|---|---|---|
| 2026-11-03 | Claude (Setup) | Phase 0 angelegt. Build-Dateien sind ungetestete Entwürfe. |
| 2026-11-03 | Claude Opus 5.5 | Phase 1 implementiert (PR #1), vom Projektinhaber auf RTX 3090 abgenommen. |
| 2026-11-03 | Claude Opus 5.5 | Phase 2 implementiert (PR #2), ADR 0006; abgenommen. |
| 2026-11-03 | Claude Opus 5.5 | Phase 3 implementiert, ADR 0007 (Mixed Precision), NVE-Protokoll 100 ps; 39/39 Tests grün. |
| 2026-11-03 | Claude Opus 5.5 | Phase 4 implementiert (Tiled All-Pairs, Builder, OpenMM-Vergleich), 47/47 Tests grün; Nsight Compute blockiert (Counter-Rechte). |
| 2026-11-03 | Claude Opus 5.5 | Phase 5 implementiert (HMR, SHAKE/RATTLE, N_dof), 52/52 Tests grün; BAOAB-Kinetikbias dokumentiert (OQ-11). |
