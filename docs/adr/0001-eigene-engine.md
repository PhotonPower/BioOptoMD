# ADR 0001: Eigene MD-Engine statt OpenMM-Plugin

- **Status:** Accepted
- **Datum:** 2026-11-03

## Kontext
OpenMM bietet GPU-MD, AMBER-Kraftfelder, implizites Lösungsmittel und eine Plugin-Schnittstelle. BioOptoMD könnte als OpenMM-Plugin (Oberfläche + Observer) realisiert werden.

## Entscheidung
Eigene C++/CUDA-Engine. Gründe: (1) volle Kontrolle über Kernels und Speicherlayout für Forschung; (2) effizientes Batching vieler kleiner Systeme auf einer GPU, was OpenMM (ein Context pro System) schlecht abdeckt; (3) In-situ-Observer ohne Host-Roundtrip.

## Alternativen
- OpenMM-Plugin: weniger Aufwand, aber kein Batch-Regime, eingeschränkte Kontrolle.
- GROMACS/LAMMPS: großer Code, schlecht für kleine Batch-Systeme, Erweiterung aufwendig.

## Konsequenzen
- Hoher Validierungsaufwand → OpenMM wird als **Referenzgenerator** in allen Kraft-Tests eingesetzt.
- Konventionen (½k, Exclusions, 1-4-Skalierung) folgen OpenMM.
- Phase 8 muss den Durchsatzvorteil im Batch-Regime nachweisen.
