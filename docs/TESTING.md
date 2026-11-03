# Teststrategie

## Ebenen

| Ebene | Ort | Zweck |
|---|---|---|
| Referenz-Physik | `tests/reference/*.py` | NumPy-Implementierungen jeder Energie/Kraft, langsam aber offensichtlich korrekt |
| Kraft-Konsistenz | alle Kraftterme | Kraft = −∇E per zentraler finiter Differenz (h = 1e-4 nm, in FP64 auf der Referenz) |
| Kreuzvalidierung | `tests/reference/openmm_*.py` | Identisches Modell in OpenMM (Custom*Force), Vergleich Kräfte & Energien |
| Statistische Mechanik | `test_phase2/5/6` | Equipartition, Boltzmann-Verteilungen, Jarzynski |
| Dynamik | NVE-Tests | Energieerhaltung, Drift |
| Robustheit | Degenerierte Geometrien | kein NaN/Inf, Clamp-Zähler |
| Determinismus | alle Phasen ab 1 | Bitidentität bei gleichem Seed |

## Marker

- `gpu` – benötigt CUDA-Gerät (automatischer Skip sonst).
- `slow` – > 30 s; nicht im schnellen Lauf.
- `reference` – benötigt OpenMM/ParmEd; Skip, wenn nicht installiert.

## Statistische Toleranzen

Thermodynamische Größen sind Zufallsvariablen. Vorgehen:
1. Equilibrierung verwerfen (mind. 5/γ ps).
2. Blockmittelung (≥ 20 Blöcke), Blocklänge ≥ 5 Korrelationszeiten → σ des Mittelwerts.
3. Akzeptanz: |Messung − Erwartung| < 4σ.
4. Verteilungen: χ²- oder KS-Test mit p > 0.001.
5. Feste Seeds, damit Tests reproduzierbar sind; zusätzlich ein `slow`-Test mit 5 verschiedenen Seeds.

## Referenzdaten

- Werden **nicht** als große Binärdateien committet, sondern durch `tests/reference/generate_all.py` erzeugt und unter `tests/reference/data/` (gitignored) gecacht.
- Kleine Eingabestrukturen (PDB/PRMTOP < 1 MB) dürfen committet werden, in `tests/reference/inputs/`.
- Jeder Generator schreibt die OpenMM-Version und alle Parameter in die Metadaten der `.npz`.

## Sanitizer & Profiling

- `compute-sanitizer --tool memcheck pytest -m "gpu and not slow" -x` vor jedem Phasenabschluss.
- `racecheck` für Kernels mit Shared Memory (Tile-Kernel, Observer).
- Nsight Compute für Phase 4 und 8; Kennzahlen in `benchmarks/README.md`.

## CI

GitHub Actions (`.github/workflows/ci.yml`) hat keine GPU: dort laufen Lint, Format-Check und `pytest -m "not gpu"`. GPU-Tests laufen lokal auf der RTX 6000 Ada; optional später ein Self-Hosted-Runner (Label `gpu`).
