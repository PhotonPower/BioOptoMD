# Projektstatus

> Diese Datei wird am Ende **jeder** Arbeitssitzung aktualisiert (AGENTS.md §2.4).

**Aktive Phase:** 7 – Python-Builder, I/O & End-to-End (noch nicht begonnen)
**Phase 6 abgenommen** am 2026-11-04 durch den Projektinhaber (RTX 3090).
**Referenz-Hardware:** RTX 6000 Ada 48 GB (CC 8.9), Host 1 TB RAM

## Abgeschlossen
- Phase 0: Repository-Gerüst, Dokumentation, Agenten-Anweisungen.
- Phase 1 (PR #1): Build, `DeviceState`, `SimulationContext`, Plugin-Interfaces, Kern-Kernels, nanobind-Modul.
- Phase 2 (PR #2): Langevin-BAOAB, `PlanarSurfaceForce`, `HarmonicRestraintPullingForce`, `EnergyTracker`,
  ADR 0008.
- Phase 3 (PR #3): `BondedForce`, präzises Minimum Image, ADR 0009 (Mixed Precision), NVE-Protokoll 100 ps.
  Alle drei vom Projektinhaber am 2026-11-03 auf RTX 3090 abgenommen.

- Phase 4 (PR #4): Tiled All-Pairs (LJ + Debye-Hückel, Switching), Builder `biooptomd.nonbonded`.
  Abgenommen 2026-11-03.

- Phase 5 (PR #5): HMR, SHAKE/RATTLE auf H-Bindungen, N_dof. Abgenommen 2026-11-03; Nacharbeit nach
  Review siehe unten.

## Phase 6 – erledigt (Branch `phase-6/observers-voxel`, baut auf Phase 5 auf)
1. `DipoleAndDistanceTracker`, `FRETTracker` (Rohzeitreihen R_DA, κ², optional E), `RefractiveIndexProfiler`
   (S_α, S_V mit Varianz, n(z) nach Lorentz-Lorenz, optionale normierte Gauß-Verschmierung) –
   `src/observers/optics.*`, PHYSICS_MODEL §6.
2. Ringpuffer im VRAM (`src/observers/ring_buffer.hpp`), Download nur auf Anfrage, `reset()`;
   `ctx.observe_now()` für statische Auswertungen.
3. `VoxelGridForce3D` (`src/forces/voxel_grid.*`): 3D-Textur, Laden aus NumPy (Ursprung, Spacing in nm),
   Skalierung je Optik-Typ, optional periodisch in x/y; `biooptomd.voxel.values_from_potential`.
- Referenzen `tests/reference/observers_ref.py`.
- Parameter α′ und v je Typ sind Eingaben (OQ-01, OQ-02 weiter offen); optische Kopplungsnormierung OQ-04.

- Review-Agent: keine Codefehler; umgesetzt: schärfere Tests (anisotropes Voxelgitter, FRET über die Boxgrenze,
  Verschmierung pro Schicht gegen Referenz), Quantifizierung der Voxel-Energienicht-Erhaltung (O(Δ), 5 % bei
  Δ = 0.1 nm, PHYSICS_MODEL §4.3), Statusbit bei Schwerpunkt ohne Massen, Box-Versionszähler (periodisches
  Voxelgitter und Profiler erkennen `set_box` nach dem Hinzufügen), n = NaN bei L ≥ 1 dokumentiert.

## Getestet (RTX 3090, Windows 11, MSVC 17.14, CUDA 12.9, Python 3.11, nanobind 3.1, OpenMM 8.6.1)
- `pytest`: 63/63 grün (57 schnell + 6 `slow`); Phase 6: alle 6 Abnahmetests + Ringpuffer-, FRET-Minimum-
  Image-, Voxel-Interpolations-, Konsistenz- und Schutztests.
- `compute-sanitizer`: memcheck 0 Fehler, racecheck 0 Hazards (Dipol-/FRET-/Profiler-/Voxel-Kernels).

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
- ADR 0008 und ADR 0009 auf `Accepted` setzen; OQ-12 bestätigen; OQ-03 (Fluorophor-Parameter) offen.
- Tiled All-Pairs prüft alle Tile-Paare (keine Cutoff-Vorauswahl); Optimierung/Cell-List später (Phase 8/9).
- nanobind-„leaked instances“ nur unter compute-sanitizer mit mehreren Tests pro Prozess (siehe frühere Notiz).
- Explodierende Trajektorien setzen Statusbits; `step()` prüft sie nicht selbst.
- clang-format: CI (ubuntu) nutzt Version 21, lokal (pip) Version 18; sie richten nachgestellte Kommentare in
  Structs unterschiedlich aus. Vor dem Push mit beiden prüfen oder Kommentare über die Member setzen.

## Nächster konkreter Schritt
1. Projektinhaber: OQ-14 (Temperaturschätzer) entscheiden; PRs #1 → #6 nacheinander mergen (gestapelt,
   nach jedem Merge Basis des nächsten PR auf `main` umstellen).
2. Phase 7, Aufgabe 1: `biooptomd.builder.from_parmed()/from_openmm()` – die bisher in den Tests verstreuten
   Schritte (Topologie-Extraktion, LJ-Typen, Ausschlüsse, HMR, Constraints) zu einem Builder bündeln.
3. Nachholen: Kontrolllauf auf der RTX 6000 Ada; Nsight Compute (Counter-Freigabe).

## Sitzungslog
| Datum | Agent/Person | Zusammenfassung |
|---|---|---|
| 2026-11-03 | Claude (Setup) | Phase 0 angelegt. Build-Dateien sind ungetestete Entwürfe. |
| 2026-11-03 | Claude Opus 5.5 | Phase 1 implementiert (PR #1), vom Projektinhaber auf RTX 3090 abgenommen. |
| 2026-11-03 | Claude Opus 5.5 | Phase 2 implementiert (PR #2), ADR 0008; abgenommen. |
| 2026-11-03 | Claude Opus 5.5 | Phase 3 implementiert, ADR 0009 (Mixed Precision), NVE-Protokoll 100 ps; 39/39 Tests grün. |
| 2026-11-03 | Claude Opus 5.5 | Phase 4 implementiert (Tiled All-Pairs, Builder, OpenMM-Vergleich), 47/47 Tests grün; Nsight Compute blockiert (Counter-Rechte). |
| 2026-11-03 | Claude Opus 5.5 | Phase 5 implementiert (HMR, SHAKE/RATTLE, N_dof), 52/52 Tests grün; BAOAB-Kinetikbias dokumentiert (OQ-14). |
| 2026-11-04 | Claude Opus 5.5 | Phase-5-Review-Nacharbeit (Stabilitätstest nach Regel 8); Phase 6 implementiert (Optik-Observer, Ringpuffer, VoxelGridForce3D), Review umgesetzt, 63/63 Tests grün. |
| 2026-11-04 | Claude Opus 5.5 | Phase 6 abgenommen (PR #6). Sitzungsende; nächster Schritt: Phase 7 auf neuem Branch `phase-7/...` ab `phase-6/observers-voxel`. |
| 2026-11-04 | Claude (Review) | Review der offenen Fragen. Vorschläge zu OQ-01…07 in `OPEN_QUESTIONS.md`, neue OQ-08 (Mg²⁺). ADR 0006 (Cutoff aus Debye-Länge) und ADR 0007 (Wandformen) als *Proposed*. Korrekturen in PHYSICS_MODEL (falsche Cutoff-Aussage, Gouy-Chapman nichtlinear, Optik-Normierung, Verschmierung), ARCHITECTURE (Shape `dip.result()`), ROADMAP (neue Abnahmetests), `test_units.py` (prüft jetzt auch `float`-Konstanten). **Offen: Entscheidung des Projektinhabers zu ADR 0006/0007 und OQ-01…08.** |
| 2026-11-04 | Claude (Review 2) | Literatur-Checkliste per Websuche geprüft (Stand in `OPEN_QUESTIONS.md`). Nachträge zu OQ-01…08 (u. a. OL24, AMBER-DYES nur Protein-Linker, NucProt, Überschusspolarisierbarkeit, microMg). Neue OQ-09 (Box/Bedeckung vs. Cutoff) und OQ-10 (κ(T)). ADR 0006 Rev. 2: Box aus Bedeckung, Bildsummation bei L < 2 r_c (ändert ADR 0003, Entscheidung nötig). ADR 0007: Lage von z = 0. **Offen: Entscheidung zu ADR 0003/0006/0007 und OQ-01…10.** |
| 2026-11-04 | Projektinhaber (eingetragen von Claude) | Vorschläge aus Review 1 und 2 angenommen: ADR 0006 Rev. 2 und ADR 0007 → Accepted, Nachtrag in ADR 0003 (Bildsummation für Non-Bonded bei L < 2 r_c). OQ-03/04/05/06/09/10 → `resolved`; OQ-01/02/07/08 → `assumption` (Vorgehen entschieden, Kalibrierung bzw. Verifikation laut Checkliste offen). |
| 2026-11-04 | Claude (Build) | Env angelegt (CUDA 12.6), Build + leeres `_core`-Modul laufen, 2 Tests grün. Keine Quelländerungen außer dieser Datei. |
| 2026-11-05 | Claude Opus 5.5 | `origin/main` (PR #7 Review/ADR-Entscheidungen, PR #8 Build-Umgebung RTX 6000 Ada) in den Phasen-Stapel gemergt. Doppelte Nummern aufgelöst: unsere ADR 0006 → **0008** (Kraftschranke/Statusflags), ADR 0007 → **0009** (Mixed Precision), unsere OQ-08…11 → **OQ-11…14**. ADR 0006/0007 auf `main` (Cutoff aus κ, Wandformen) gelten unverändert. |
