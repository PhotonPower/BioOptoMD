# Projektstatus

> Diese Datei wird am Ende **jeder** Arbeitssitzung aktualisiert (AGENTS.md §2.4).

**Aktive Phase:** 5 – Constraints & Hydrogen Mass Repartitioning
**Phase 4 abgenommen** am 2026-11-03 durch den Projektinhaber (RTX 3090).
**Referenz-Hardware:** RTX 6000 Ada 48 GB (CC 8.9), Host 1 TB RAM

## Abgeschlossen
- Phase 0: Repository-Gerüst, Dokumentation, Agenten-Anweisungen.
- Phase 1 (PR #1): Build, `DeviceState`, `SimulationContext`, Plugin-Interfaces, Kern-Kernels, nanobind-Modul.
- Phase 2 (PR #2): Langevin-BAOAB, `PlanarSurfaceForce`, `HarmonicRestraintPullingForce`, `EnergyTracker`,
  ADR 0008.
- Phase 3 (PR #3): `BondedForce`, präzises Minimum Image, ADR 0009 (Mixed Precision), NVE-Protokoll 100 ps.
  Alle drei vom Projektinhaber am 2026-11-03 auf RTX 3090 abgenommen.

## Phase 4 – erledigt (Branch `phase-4/tiled-nonbonded`, baut auf Phase 3 auf)
1. Tile-Kernel `k_tiled_all_pairs<DebyeHuckel>` (`src/nonbonded/`): ein Warp pro 32×32-Tile-Paar, j-Atome
   im Shared Memory mit rotierendem, konfliktfreiem Zugriff; LJ + Debye-Hückel, Switching wie OpenMM
   (faktorisiert, ohne FP32-Auslöschung), Minimum Image, Box-Prüfung L ≥ 2 r_c. 48 Register, keine Spills.
2. Exclusion-Bitmasken und LJ-Typtabelle im Python-Builder `biooptomd.nonbonded` (`nonbonded_topology`).
3. LJ-Typtabelle pro Replica im Shared Memory (Typ 0 = Padding, ε = 0 erzwungen); neues Statusbit
   `STATUS_UNKNOWN_TYPE`.
4. Elektrostatik als Template-Parameter (`detail::DebyeHuckel`), vorbereitet für GB-OBC.
5. Laufzeit/Ressourcen in `benchmarks/README.md` (≈ 1.3·10¹¹ Paarprüfungen/s bei 64 Replicas).
   **Nsight-Compute-Metriken fehlen**: `ncu` meldet `ERR_NVGPUCTRPERM` (Counter-Freigabe braucht Admin).
- Referenzen: `tests/reference/nonbonded_ref.py` (O(N²), FP64; = OpenMM auf 1e-14), `openmm_nonbonded.py`
  (CustomNonbondedForce, CutoffPeriodic, identische Ausschlüsse), Testsystem `make_duplex_ions.py`
  (2 × 1BNA-Duplex + 40 Na⁺ + 2 Mg²⁺, 1558 Atome; ASSUMPTION OQ-03: kein Fluorophor verfügbar).
- PHYSICS_MODEL §3: Switching-Funktion definiert (war offen).

## Getestet (RTX 3090, Windows 11, MSVC 17.14, CUDA 12.9, Python 3.11, nanobind 3.1, OpenMM 8.6.1)
- `pytest`: 47/47 grün (inkl. `reference` und `slow`). Phase 4: alle 4 Abnahmetests + Builder-, Referenz-
  und Validierungstests. NVE (1558 Atome, 10 ps, 1 fs): Drift 3–5·10⁻⁵ kJ/mol/ps/Atom, Fluktuation 6.6·10⁻⁴.
- `compute-sanitizer`: memcheck 0 Fehler, racecheck 0 Hazards, synccheck 0 Fehler.
- Review-Agent: keine kritischen Fehler; Testschärfe per Fehlerinjektion belegt (Toleranz 10²–10⁵× überschritten
  bei falschen Masken, fehlendem Minimum Image, nur einem LJ-Typ). Zwei Kleinigkeiten behoben (kein
  Handle-Rückgabewert von `set_nonbonded_tiled`; LJ nur bei ε ≠ 0 → kein falscher NaN-Status).
- Testanpassungen (begründet, Roadmap-Kriterien unverändert): Zufallssysteme mit Mindestabstand 0.32 nm;
  zusätzlicher Gesamtkraftvergleich gegen OpenMM mit FP32-Parametern und Einzelterm-RMS als Maßstab.

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
1. Projektinhaber: Phase 4 abnehmen; PRs #1 → #2 → #3 → Phase 4 nacheinander mergen.
2. Phase 5, Aufgabe 1: HMR im Builder; dann SHAKE/RATTLE auf H-Bindungen (zuerst NumPy-Referenz).
3. Nachholen: Kontrolllauf auf der RTX 6000 Ada; Nsight Compute.

## Sitzungslog
| Datum | Agent/Person | Zusammenfassung |
|---|---|---|
| 2026-11-03 | Claude (Setup) | Phase 0 angelegt. Build-Dateien sind ungetestete Entwürfe. |
| 2026-11-03 | Claude Opus 5.5 | Phase 1 implementiert (PR #1), vom Projektinhaber auf RTX 3090 abgenommen. |
| 2026-11-03 | Claude Opus 5.5 | Phase 2 implementiert (PR #2), ADR 0008; abgenommen. |
| 2026-11-03 | Claude Opus 5.5 | Phase 3 implementiert, ADR 0009 (Mixed Precision), NVE-Protokoll 100 ps; 39/39 Tests grün. |
| 2026-11-03 | Claude Opus 5.5 | Phase 4 implementiert (Tiled All-Pairs, Builder, OpenMM-Vergleich), 47/47 Tests grün; Nsight Compute blockiert (Counter-Rechte). |
| 2026-11-04 | Claude (Review) | Review der offenen Fragen. Vorschläge zu OQ-01…07 in `OPEN_QUESTIONS.md`, neue OQ-08 (Mg²⁺). ADR 0006 (Cutoff aus Debye-Länge) und ADR 0007 (Wandformen) als *Proposed*. Korrekturen in PHYSICS_MODEL (falsche Cutoff-Aussage, Gouy-Chapman nichtlinear, Optik-Normierung, Verschmierung), ARCHITECTURE (Shape `dip.result()`), ROADMAP (neue Abnahmetests), `test_units.py` (prüft jetzt auch `float`-Konstanten). **Offen: Entscheidung des Projektinhabers zu ADR 0006/0007 und OQ-01…08.** |
| 2026-11-04 | Claude (Review 2) | Literatur-Checkliste per Websuche geprüft (Stand in `OPEN_QUESTIONS.md`). Nachträge zu OQ-01…08 (u. a. OL24, AMBER-DYES nur Protein-Linker, NucProt, Überschusspolarisierbarkeit, microMg). Neue OQ-09 (Box/Bedeckung vs. Cutoff) und OQ-10 (κ(T)). ADR 0006 Rev. 2: Box aus Bedeckung, Bildsummation bei L < 2 r_c (ändert ADR 0003, Entscheidung nötig). ADR 0007: Lage von z = 0. **Offen: Entscheidung zu ADR 0003/0006/0007 und OQ-01…10.** |
| 2026-11-04 | Projektinhaber (eingetragen von Claude) | Vorschläge aus Review 1 und 2 angenommen: ADR 0006 Rev. 2 und ADR 0007 → Accepted, Nachtrag in ADR 0003 (Bildsummation für Non-Bonded bei L < 2 r_c). OQ-03/04/05/06/09/10 → `resolved`; OQ-01/02/07/08 → `assumption` (Vorgehen entschieden, Kalibrierung bzw. Verifikation laut Checkliste offen). |
| 2026-11-04 | Claude (Build) | Env angelegt (CUDA 12.6), Build + leeres `_core`-Modul laufen, 2 Tests grün. Keine Quelländerungen außer dieser Datei. |
| 2026-11-05 | Claude Opus 5.5 | `origin/main` (PR #7 Review/ADR-Entscheidungen, PR #8 Build-Umgebung RTX 6000 Ada) in den Phasen-Stapel gemergt. Doppelte Nummern aufgelöst: unsere ADR 0006 → **0008** (Kraftschranke/Statusflags), ADR 0007 → **0009** (Mixed Precision), unsere OQ-08…11 → **OQ-11…14**. ADR 0006/0007 auf `main` (Cutoff aus κ, Wandformen) gelten unverändert. |
