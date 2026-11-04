# Projektstatus

> Diese Datei wird am Ende **jeder** Arbeitssitzung aktualisiert (AGENTS.md §2.4).

**Aktive Phase:** 1 – Build-System, DeviceState, Plugin-Gerüst, Python-Bindings
**Referenz-Hardware:** RTX 6000 Ada 48 GB (CC 8.9), Host 1 TB RAM

## Abgeschlossen
- Phase 0: Repository-Gerüst, Dokumentation, Agenten-Anweisungen (Initial-Setup).

## In Arbeit
- Phase 1, Aufgabe 1 erledigt (Build läuft, noch nicht committet): Env `biooptomd` angelegt, `pip install -v -e . --no-build-isolation` baut, `import biooptomd._core` funktioniert, `pytest` 2 passed.
  - Toolchain in der Env: CUDA 12.6 (nvcc 12.6.85, `cuda-version=12.6`), Python 3.12. Treiber/GPU: RTX 6000 Ada. Der System-nvcc (HPC SDK 12.9) ist nicht relevant; Env-nvcc steht nach `conda activate` zuerst im PATH.
  - Env-Anlage: `conda env create -f environment.yml` scheitert ohne Anaconda-ToS-Zustimmung (CondaToSNonInteractiveError für `pkgs/main`, `pkgs/r`). Workaround ohne ToS: `conda create -n biooptomd --override-channels -c conda-forge <Pakete aus environment.yml>`.
  - Hinweis: Das Modul wird nach `site-packages/biooptomd/_core*.so` installiert (nicht in `python/biooptomd/`); `python/biooptomd/` ist per Editable-Install eingebunden.

## Rote Tests
– keine (es existieren noch keine GPU-Tests)

## Nächster konkreter Schritt
1. ~~environment.yml installieren / Build zum Laufen bringen~~ (erledigt, s. o.).
2. Phase-1-Aufgaben 2–7 laut `docs/ROADMAP.md`.

## Sitzungslog
| Datum | Agent/Person | Zusammenfassung |
|---|---|---|
| 2026-11-03 | Claude (Setup) | Phase 0 angelegt. Build-Dateien sind ungetestete Entwürfe. |
| 2026-11-04 | Claude (Review) | Review der offenen Fragen. Vorschläge zu OQ-01…07 in `OPEN_QUESTIONS.md`, neue OQ-08 (Mg²⁺). ADR 0006 (Cutoff aus Debye-Länge) und ADR 0007 (Wandformen) als *Proposed*. Korrekturen in PHYSICS_MODEL (falsche Cutoff-Aussage, Gouy-Chapman nichtlinear, Optik-Normierung, Verschmierung), ARCHITECTURE (Shape `dip.result()`), ROADMAP (neue Abnahmetests), `test_units.py` (prüft jetzt auch `float`-Konstanten). **Offen: Entscheidung des Projektinhabers zu ADR 0006/0007 und OQ-01…08.** |
| 2026-11-04 | Claude (Review 2) | Literatur-Checkliste per Websuche geprüft (Stand in `OPEN_QUESTIONS.md`). Nachträge zu OQ-01…08 (u. a. OL24, AMBER-DYES nur Protein-Linker, NucProt, Überschusspolarisierbarkeit, microMg). Neue OQ-09 (Box/Bedeckung vs. Cutoff) und OQ-10 (κ(T)). ADR 0006 Rev. 2: Box aus Bedeckung, Bildsummation bei L < 2 r_c (ändert ADR 0003, Entscheidung nötig). ADR 0007: Lage von z = 0. **Offen: Entscheidung zu ADR 0003/0006/0007 und OQ-01…10.** |
| 2026-11-04 | Projektinhaber (eingetragen von Claude) | Vorschläge aus Review 1 und 2 angenommen: ADR 0006 Rev. 2 und ADR 0007 → Accepted, Nachtrag in ADR 0003 (Bildsummation für Non-Bonded bei L < 2 r_c). OQ-03/04/05/06/09/10 → `resolved`; OQ-01/02/07/08 → `assumption` (Vorgehen entschieden, Kalibrierung bzw. Verifikation laut Checkliste offen). |
| 2026-11-04 | Claude (Build) | Env angelegt (CUDA 12.6), Build + leeres `_core`-Modul laufen, 2 Tests grün. Keine Quelländerungen außer dieser Datei. |
