# Projektstatus

> Diese Datei wird am Ende **jeder** Arbeitssitzung aktualisiert (AGENTS.md §2.4).

**Aktive Phase:** 1 – Build-System, DeviceState, Plugin-Gerüst, Python-Bindings
(Implementierung vollständig, **Abnahme auf der Referenz-GPU RTX 6000 Ada steht noch aus**)
**Referenz-Hardware:** RTX 6000 Ada 48 GB (CC 8.9), Host 1 TB RAM

## Abgeschlossen
- Phase 0: Repository-Gerüst, Dokumentation, Agenten-Anweisungen (Initial-Setup).
- Phase 1, Aufgaben 1–7 (Branch `phase-1/build-devicestate-bindings`, Commits `f6a14c6`, `0f243a7`,
  `6b17572`, `3773865`; PR noch nicht erstellt):
  1. CMake + scikit-build-core + nanobind bauen; Architekturen 86;89 (Bug im Phase-0-Entwurf behoben:
     `CMAKE_CUDA_ARCHITECTURES` muss **vor** `project()` gesetzt werden, sonst sm_52). Release
     `-O3 -lineinfo -Xptxas=-warn-spills`, Debug `-G` (ohne `-warn-spills`, weil `-G` systematisch spillt).
  2. `include/biooptomd/cuda_check.hpp` (`BOMD_CUDA_CHECK`, `BOMD_LAUNCH_CHECK`, optional
     `-DBOMD_SYNC_LAUNCH_CHECK` zum Debuggen).
  3. `DeviceState` (trivially copyable), `SimulationContext` (RAII: `DeviceBuffer`, `CudaStream`).
  4. `ForcePlugin`, `ObserverPlugin`, `NonbondedStrategy` in `include/biooptomd/plugin.hpp`;
     `SimulationContext::compute_forces()` als Pipeline.
  5. Kernels `clear_forces`, `fixed_point_to_float`, `wrap_xy` in `src/core/core_kernels.cu`.
  6. `biooptomd._core`: Export aller Puffer als NumPy-Kopie und DLPack-Zero-Copy (`nb::array_api`),
     Test-Hooks in `_core.testing`, kompilierte Konstanten in `_core.units`.
  7. pytest-Marker/Skips waren schon vorhanden; `conftest.py` ergänzt um den Pfad zu `tests/reference`.

## Getestet (Entwicklungsmaschine, **nicht** Referenz)
RTX 3090 (CC 8.6), Windows 11, MSVC 17.14, CUDA 12.9, Python 3.11 (venv, kein Conda), nanobind 3.1,
CuPy 14 (cupy-cuda12x):
- `pytest`: 10/10 grün, darunter alle Abnahmetests aus ROADMAP Phase 1.
- `compute-sanitizer --tool memcheck pytest -m "gpu and not slow"`: 0 Fehler.
- Release- und Debug-Build ohne Warnungen; ruff und clang-format sauber.
- CI-Pfad (`PYTHONPATH=python pytest -m "not gpu and not reference"`, ohne gebautes Modul) grün.

## In Arbeit
– nichts halb Fertiges.

## Rote Tests
– keine.

## Offene Punkte / Hinweise für den nächsten Agenten
- OQ-08 (Auslegung „exakte Rekonstruktion“ in `test_wrap_xy`) bitte vom Projektinhaber bestätigen.
- `wrap_xy` ist nur für |x/L| < 2²⁰ korrekt. `set_positions`/`set_box` prüfen Eingaben, aber eine
  während der Dynamik explodierende Trajektorie wird noch nicht erkannt. Phase 2 sollte im Integrator
  ein Geräte-Statusflag (NaN/Bereich) einführen.
- Fixed-Point-Überlauf (|F| ≥ 2³¹) wird nicht erkannt (ADR 0002, als Grenze dokumentiert).
- Echte Atomzahl ist pro Batch einheitlich (`n_atoms`). Für unterschiedliche Systeme pro Replica
  (ADR 0005) braucht es in Phase 3 eine Atomzahl/Topologie-Offsets pro Replica.
- Setter synchronisieren den Stream (bewusst, nicht im Hot-Path).

## Nächster konkreter Schritt
1. Auf der RTX 6000 Ada: `conda env create -f environment.yml`, `pip install -v -e . --no-build-isolation`,
   `pytest` und `compute-sanitizer` ausführen und die CUDA-Version hier eintragen. Danach Phase 1 als
   abgeschlossen markieren und PR nach `main` mergen.
2. Phase 2, Aufgabe 1: `LangevinIntegrator` (BAOAB, Philox), zuerst NumPy-Referenz.

## Sitzungslog
| Datum | Agent/Person | Zusammenfassung |
|---|---|---|
| 2026-11-03 | Claude (Setup) | Phase 0 angelegt. Build-Dateien sind ungetestete Entwürfe. |
| 2026-11-03 | Claude Opus 5.5 | Phase 1 implementiert und auf RTX 3090/Windows getestet; Abnahme auf Referenz-GPU offen. |
