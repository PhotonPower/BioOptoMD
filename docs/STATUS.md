# Projektstatus

> Diese Datei wird am Ende **jeder** Arbeitssitzung aktualisiert (AGENTS.md §2.4).

**Aktive Phase:** 4 – Nicht-kovalente Kräfte (Tiled All-Pairs)
**Phase 3 abgenommen** am 2026-11-03 durch den Projektinhaber (RTX 3090).
**Referenz-Hardware:** RTX 6000 Ada 48 GB (CC 8.9), Host 1 TB RAM

## Abgeschlossen
- Phase 0: Repository-Gerüst, Dokumentation, Agenten-Anweisungen.
- Phase 1 (PR #1): Build, `DeviceState`, `SimulationContext`, Plugin-Interfaces, Kern-Kernels, nanobind-Modul
  mit NumPy/DLPack. Abgenommen 2026-11-03 (RTX 3090).
- Phase 2 (PR #2): Langevin-BAOAB (Philox), `PlanarSurfaceForce`, `HarmonicRestraintPullingForce`,
  `EnergyTracker`, ADR 0006 (Kraftschranke, Fixed-Point-Sättigung, Statusflags). Abgenommen 2026-11-03.

## Phase 3 – erledigt (Branch `phase-3/bonded-forces`, baut auf Phase 2 auf)
Commits `de3ecd7` (Referenzen), `fb85461` (BondedForce), `e01d6ca` (Bindings/Tests), `2e94e89` (ADR 0007),
`5da71b3` (Mixed Precision), `05820bf` (Tests), `a70e75e` (Docs):
1. NumPy-Referenz `tests/reference/bonded_ref.py` (FP64, Minimum Image, Grenzfälle), Finite-Differenzen-Test
   (`test_reference_finite_differences`, läuft auch in der CI ohne GPU) und Abgleich mit OpenMM auf 1e-10.
2. `BondedForce` (`src/forces/bonded.*`): Bonds, Angles (atan2), periodische Torsionen/Impropers (IUPAC),
   1-4-Paare (LJ + Debye-Hückel, κ und ε_r pro Replica).
3. Topologie pro Replica über `(begin, count)`-Offsets (ADR 0005); `add_bonded` validiert Schlüssel, Shapes,
   dtypes und Indizes.
4. OpenMM-Referenz (`openmm_bonded.py`, `generate_all.py`): DNA-Trinukleotid 5'-CGC-3' aus PDB 1BNA,
   AMBER OL15 (ASSUMPTION OQ-03), 92 Atome; Topologie als `inputs/dna_trinucleotide_topology.npz` committet.
- Präzises Minimum Image (`src/core/pbc.cuh`): betragsgrößere Koordinate zuerst verschieben (Sterbenz);
  behebt FP32-Fehler ~ulp(L) bei Bindungen über die Boxgrenze (auch in Restraints).
- **ADR 0007 Mixed Precision** (Entscheidung Projektinhaber, Status `Proposed`): Geschwindigkeiten FP64
  (`Vec4d`, 32-Byte-aligned statt des ab CUDA 13 deprecateten `double4`), Positionskorrektur `d_pos_corr`;
  Integrator, `wrap_xy`, Pulling-Schwerpunkt in FP64. Kraft-Kernels unverändert FP32.
  Messung NVE 100 ps bei 5 nm: Drift im Mittel 7·10⁻⁶ (vorher 1.6·10⁻⁴) kJ/mol/ps/Atom.
  Löst OQ-08 (Wrap-Rekonstruktion jetzt bitgenau exakt).
- ROADMAP: NVE-Protokoll auf 200 000 Schritte (100 ps) geändert (Entscheidung Projektinhaber), da 5000
  Schritte eine Drift von 1e-4 statistisch nicht auflösen (selbst exaktes FP64 lag bei 1.9·10⁻⁴).

## Getestet (RTX 3090, Windows 11, MSVC 17.14, CUDA 12.9, Python 3.11, nanobind 3.1, OpenMM 8.6.1)
- `pytest`: 39/39 grün (inkl. `reference`-Tests mit OpenMM und `slow`).
- `compute-sanitizer --tool memcheck` (Phase 1+3 ohne den 100-ps-Lauf; Phase 2 zuvor): 0 Fehler;
  `racecheck` der Shared-Memory-Kernels: 0 Hazards.
- Build ohne Warnungen; ruff und clang-format sauber.
- Review-Agent: keine Physik-/Konsistenzfehler; Hinweise (double4-Deprecation, Test der Abschirmung pro
  Replica, strengere Topologie-Validierung) umgesetzt.

## Rote Tests
– keine.

## Offene Punkte / Hinweise für den nächsten Agenten
- ADR 0006 und ADR 0007 auf `Accepted` setzen (Projektinhaber).
- OQ-09 (Gouy-Chapman für ζ < 0) bitte bestätigen. OQ-03 (Kraftfeld) bleibt Annahme OL15.
- nanobind meldet unter `compute-sanitizer` mit mehreren Tests pro Prozess „leaked instances“; ohne Sanitizer
  nicht reproduzierbar. Nächster Schritt: `gc.get_referrers` auf die Kontexte am Testende.
- Ein beschreibbarer DLPack-View auf `positions()` lässt `d_pos_corr` unverändert (Fehler ≤ ½ ulp, harmlos).
- Explodierende Trajektorien setzen Statusbits (ADR 0006); `step()` prüft sie nicht selbst.
- Atomzahl ist pro Batch einheitlich; unterschiedliche Systeme werden über Topologien + Padding abgebildet.

## Nächster konkreter Schritt
1. Projektinhaber: Phase 3 abnehmen; PRs #1 → #2 → Phase 3 nacheinander mergen.
2. Phase 4, Aufgabe 1: 32×32-Tile-Kernel (LJ + Debye-Hückel, Switching, Minimum Image), zuerst O(N²)-NumPy-
   Referenz mit Exclusions. Die 1-4-Paare sind im Nonbonded-Kernel auszuschließen (PHYSICS_MODEL §3).
3. Nachholen: Kontrolllauf auf der RTX 6000 Ada.

## Sitzungslog
| Datum | Agent/Person | Zusammenfassung |
|---|---|---|
| 2026-11-03 | Claude (Setup) | Phase 0 angelegt. Build-Dateien sind ungetestete Entwürfe. |
| 2026-11-03 | Claude Opus 5.5 | Phase 1 implementiert (PR #1), vom Projektinhaber auf RTX 3090 abgenommen. |
| 2026-11-03 | Claude Opus 5.5 | Phase 2 implementiert (PR #2), ADR 0006; abgenommen. |
| 2026-11-03 | Claude Opus 5.5 | Phase 3 implementiert, ADR 0007 (Mixed Precision), NVE-Protokoll 100 ps; 39/39 Tests grün. |
