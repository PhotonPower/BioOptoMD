# Projektstatus

> Diese Datei wird am Ende **jeder** Arbeitssitzung aktualisiert (AGENTS.md §2.4).

**Aktive Phase:** 9.1 – Generalized Born OBC2 (Branch `phase-9/gb-obc2`, ADR 0010 Proposed). Reihenfolge
vom Projektinhaber am 2026-11-05 festgelegt: GB-OBC2 zuerst (mbondi2, Salz, ACE optional mit Default an,
erweiterbar für GBn2).
**Phase 8 abgenommen** am 2026-11-05 durch den Projektinhaber (RTX 3090, PR #11).
**Phase 7 abgenommen** am 2026-11-05 durch den Projektinhaber (RTX 3090).
**Nachrüstung (PR #9) abgenommen** am 2026-11-05 durch den Projektinhaber (RTX 3090).
**Phase 6 abgenommen** am 2026-11-04 durch den Projektinhaber (RTX 3090), vor den neuen Abnahmetests.
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

## Phase 8 – erledigt, abgenommen 2026-11-05 (PR #11)
- `benchmarks/throughput.py`: BioOptoMD vs. OpenMM 8.6.1 (CUDA, mixed) mit gleichem Modell (DH, r_c aus κ
  = 3.92 nm, HMR, H-Constraints 1e-6, 4 fs). Ergebnis (RTX 3090, 3 Läufe, `benchmarks/README.md`):
  1 × 20 k Faktor 0.59, **32 × 2 k Faktor 3.52 (3.46–3.68), Ziel ≥ 3 erfüllt**, 128 × 1 k Faktor 9.7.
- Optimierungen: Tile-Vorauswahl über achsenparallele Hüllen (Minimum-Image-Dreiecksungleichung,
  bitidentisch; im Bildsummen-Pfad aus), M-SHAKE und direktes RATTLE (gekoppeltes Cluster-System statt
  Gauss-Seidel; Toleranzen unverändert).
- Tests: `test_tile_culling_vs_numpy`, `test_constraints_immobile_atoms` (inv_mass = 0, Fallback,
  STATUS_CONSTRAINT_FAILURE); `test_constraint_step_vs_numpy` vergleicht auf 1e-12 auskonvergierte
  Lösungen (verfahrensunabhängig, Schranke 1e-9 nm unverändert).
- Getestet: `pytest` 94/94 grün (inkl. `slow`); compute-sanitizer memcheck 0 Fehler, racecheck 0 Hazards
  (k_tile_bounds, Tiled-Kernel, SHAKE/RATTLE).
- Review-Agent: keine Ergebnisfehler; umgesetzt: GPU-Test für unbewegliche Atome, relativer
  Sicherheitsabstand der Vorauswahl, Randpaar-Relevanz im Culling-Test belegt, gleiche
  Constraint-Toleranz in OpenMM, Endlichkeitsprüfung, Spanne über 3 Läufe, README-Begründungen
  (Multi-Process unter WDDM, abweichende Atomzahlen).
- Für den Vergleich wurde das OpenMM-CUDA-Plugin (`openmm[cuda12]`) ins venv installiert; conda-forge-
  OpenMM aus `environment.yml` enthält CUDA bereits.
- Offen: Zellliste für große Einzelsysteme (Phase 9+); Kontrolllauf auf der RTX 6000 Ada; Nsight Compute.

## Phase 7 – erledigt, abgenommen 2026-11-05 (Branch `phase-7/builder-io`, PR #10)
1. `biooptomd.builder`: `from_openmm`/`from_parmed` → `SystemSpec`; `build_context` (LJ-Typen,
   Exclusion-Tiles, HMR, H-Constraints, κ_r/ε_r(T_r), Cutoff aus κ, Box unverändert, Typ-Default 1).
   Lehnt mehrfache Kräfte gleichen Typs, Parameter-Offsets und virtuelle Sites ab.
2. Helfer: `add_anchors` (Höhe über z_top, OQ-07), `assign_types`, `dipole_probe`, `optical_tables`
   (α′, v je Element, ASSUMPTION(OQ-01/02), unkalibriert).
3. `biooptomd.io`: bitgenaue Checkpoints (`save/restore_dynamic_state`, prüft Ladungen/Massen gegen den
   Kontext; Pulling beginnt die Arbeitsakkumulation nach dem Restore neu, W selbst und Observer-Puffer
   sind nicht im Checkpoint), `DCDWriter` je Replica (C = 0, z nicht periodisch), `read_dcd`,
   `HDF5Writer`. Neue Konstante `AKMA_TIME_PS`.
4. `biooptomd.analysis`: FRET-Regime dynamic/static/static_isotropic/static_distance, Blockmittelung
   (Flyvbjerg-Petersen mit Plateau-Kriterium), n(z)-Schichtstapel, CSV-Export.
5. `examples/aptamer_on_graphene/` (Platzhalter OQ-01/02/03/05/07 markiert).
- Abnahmetests: `test_builder_roundtrip_openmm` (ParmEd und OpenMM, Energie rel. ≈ 5e-8),
  `test_checkpoint_bitexact` (mit/ohne Constraints, mit zeitabhängigem Pulling, Imagezähler ≠ 0),
  `test_end_to_end_biosensor` (`slow`, 8 Replicas, 1 ns in ≈ 80 s). Dazu `test_builder_options`,
  `test_checkpoint_mismatch`, `test_trajectory_writers`, `test_analysis.py`.
- Review-Agent: keine Fehler mit Auswirkung; umgesetzt: Typ-Default 1 statt 0, strengere
  OpenMM-Prüfungen, Ankerhöhe im Beispiel (0.35 nm wie dokumentiert, Höhe relativ zu z_top),
  datenbasierte n(z)-Prüfung, Checkpoint-Test schärfer, Massen-/Ladungsprüfung beim Laden,
  DCD-ISTART/NSTEP wie OpenMM, Plateau-Kriterium, 4σ-Toleranzen in test_analysis.
- Getestet (RTX 3090, CUDA 12.9): `pytest` 92/92 grün (inkl. `slow`), compute-sanitizer memcheck 0 Fehler
  auf Checkpoint-/Trajektorientests.
- Hinweis: ParmEd rundet Massen (bis 0.004 amu Abweichung zu OpenMM); ohne Einfluss auf Energien.

## Integration von `main` (2026-11-05)
`origin/main` (PR #7: Review, ADR 0006 Rev. 2 und ADR 0007 Accepted, OQ-01…10 entschieden; PR #8:
Build-Umgebung RTX 6000 Ada mit conda, CUDA 12.6) wurde kaskadierend in die Branches der PRs #1 → #6
gemergt und gepusht. Doppelte Nummern: unsere ADR 0006 → **0008** (Kraftschranke/Statusflags),
ADR 0007 → **0009** (Mixed Precision), unsere OQ-08…11 → **OQ-11…14**.

## Nachrüstung (Branch `phase-6/nachruestung-adr0006-0007`, PR #9 auf PR #6)
- Phase 2 (`9255cae`): Wandformen 9-3/10-4/10-4-3 (ADR 0007), Tabelle pro (Oberflächen-Typ, LJ-Typ),
  `biooptomd.surface.wall_table` (Lorentz-Berthelot, λ pro Oberflächen-Typ, Typ 0 reserviert und im
  C++-Kern erzwungen), nichtlineares Gouy-Chapman (OQ-06) mit γ aus T_r und κ_r je Replica.
- Phase 4 (`7c57edd`): Cutoff aus κ (`cutoff_from_kappa`, `debye_kappa`, `water_permittivity`, OQ-10),
  Summe über periodische Bilder bei L < 2 r_c (ADR 0006 Rev. 2); Box wird nie geändert.
- Phase 6 (siehe Commit unten): Kugelschnitt-Verschmierung (Default), Gauß σ = R/√5, Punkt (OQ-02);
  `biooptomd.voxel.optical_scale(..., amplitude=...)` mit Überschusspolarisierbarkeit (OQ-04).
- Review-Agent: keine Rechenfehler. Umgesetzt: Fehler bei nichtlinearem GC mit T_r = 0 statt stillem
  Wegfall; Oberflächen-Typ 0 im Kern erzwungen; Test mit Φ0 < 0; Φ0 = 1 V prüft V(0) = qΦ0 und
  F(0) = 2k_BTκ sinh(Φ0/2k_BT); Voxel-Normierungstest mit Kräften weit über der Fixed-Point-Auflösung;
  Energietoleranz im Wandformtest 1e-5; GC-Fernfeld über 2·atanh(u).
- Getestet (RTX 3090, CUDA 12.9): `pytest` 82/82 grün (inkl. `slow`); compute-sanitizer memcheck 0 Fehler,
  racecheck 0 Hazards (Wandformen, GC, Bildsumme, Profiler Kugel/Gauß).
- **Offen (ROADMAP Phase 6, `reference`):** `test_refractive_index_dn_dc` braucht die Kalibrierung der
  Parameter (OQ-01/02, `assumption`); `test_voxel_optical_sphere_consistency` braucht eine eigene
  Clausius-Mossotti-Vergleichsrechnung. Beide nicht umgesetzt.
- Hinweis für Phase 7 (REMD): GC liest T_r live, κ_r und ε_r (GC, Bonded, Nonbonded) bleiben auf den
  Startwerten – beim Temperaturtausch mitführen. λ derzeit nur pro Oberflächen-Typ (Atomklassen über
  eigene Oberflächen-Typen).

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
- ADR 0008 und ADR 0009 auf `Accepted` setzen; OQ-12 bestätigen. OQ-03 ist entschieden (OL21, AMBER-DYES);
  das Phase-4-Testsystem nutzt noch OL15 ohne Fluorophor.
- Build auf der RTX 6000 Ada (laut PR #8: conda-forge-Workaround ohne Anaconda-ToS, CUDA 12.6, Python 3.12)
  ist mit dem leeren Modul belegt; Kontrolllauf der Phasen-Tests dort steht aus.
- Tiled All-Pairs überspringt Tile-Paare außerhalb r_c (Phase 8), Aufwand aber weiter O(N²) Tile-Paare;
  Zellliste später (Phase 9+).
- nanobind-„leaked instances“ nur unter compute-sanitizer mit mehreren Tests pro Prozess (siehe frühere Notiz).
- Explodierende Trajektorien setzen Statusbits; `step()` prüft sie nicht selbst.
- clang-format: CI (ubuntu) nutzt Version 21, lokal (pip) Version 18; sie richten nachgestellte Kommentare in
  Structs unterschiedlich aus. Vor dem Push mit beiden prüfen oder Kommentare über die Member setzen.

## Nächster konkreter Schritt
1. Projektinhaber: Phase 8 abnehmen (Push/PR auf Anfrage); OQ-14 (Temperaturschätzer) entscheiden;
   PRs #1 → #6, #9, #10 nacheinander mergen.
2. Danach Phase 9+ nach Forschungsbedarf (ROADMAP); offen aus Phase 6: zwei `reference`-Tests
   (dn/dc-Kalibrierung, Clausius-Mossotti-Konsistenz).
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
| 2026-11-05 | Projektinhaber | Nachrüstung (PR #9) abgenommen; weiter mit Phase 7. |
| 2026-11-05 | Projektinhaber | Phase 7 (PR #10) abgenommen; weiter mit Phase 8. |
| 2026-11-05 | Projektinhaber | Phase 8 (PR #11) abgenommen; weiter mit Phase 9+. |
| 2026-11-05 | Claude Opus 5.5 | Phase 8: Benchmark vs. OpenMM (32×2k Faktor 3.5), Tile-Vorauswahl, M-SHAKE/RATTLE; Review umgesetzt, 94/94 Tests grün. |
| 2026-11-05 | Claude Opus 5.5 | Phase 7 implementiert (PR #10): Builder, Checkpoints, DCD/HDF5, Analyse, Beispiel + End-to-End-Test; Review umgesetzt. |
| 2026-11-05 | Claude Opus 5.5 | Nachrüstung Phasen 2/4/6 (PR #9): Wandformen + Mischungsregeln, nichtlineares GC, κ-Cutoff + Bildsumme, Kugelschnitt-Verschmierung, optische Normierung; Review umgesetzt. Offen: zwei `reference`-Tests aus Phase 6 (s. o.). |
