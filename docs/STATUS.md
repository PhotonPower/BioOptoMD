# Projektstatus

> Diese Datei wird am Ende **jeder** Arbeitssitzung aktualisiert (AGENTS.md §2.4).

**Aktive Phase:** 1 – Build-System, DeviceState, Plugin-Gerüst, Python-Bindings
**Referenz-Hardware:** RTX 6000 Ada 48 GB (CC 8.9), Host 1 TB RAM

## Abgeschlossen
- Phase 0: Repository-Gerüst, Dokumentation, Agenten-Anweisungen (Initial-Setup).

## In Arbeit
– (noch nichts)

## Rote Tests
– keine (es existieren noch keine GPU-Tests)

## Nächster konkreter Schritt
1. `environment.yml` auf der Referenzmaschine installieren, CUDA-Toolkit-Version prüfen und in diese Datei eintragen.
2. `CMakeLists.txt`/`pyproject.toml` (Entwürfe aus Phase 0) zum Bauen bringen; leeres nanobind-Modul `biooptomd._core` importierbar.
3. Danach Phase-1-Aufgaben 2–7 laut `docs/ROADMAP.md`.

## Sitzungslog
| Datum | Agent/Person | Zusammenfassung |
|---|---|---|
| 2026-11-03 | Claude (Setup) | Phase 0 angelegt. Build-Dateien sind ungetestete Entwürfe. |
