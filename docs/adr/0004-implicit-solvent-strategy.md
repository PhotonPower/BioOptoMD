# ADR 0004: Implizites Lösungsmittel – Debye-Hückel (V1), Generalized Born (V2)

- **Status:** Accepted
- **Datum:** 2026-11-03

## Kontext
Explizites Wasser würde die Atomzahl um Faktor ~10 erhöhen und das Batch-Regime zerstören. Debye-Hückel bildet Salzabschirmung ab, aber keine Desolvatation – kritisch für Aptamer-Faltung und Bindung.

## Entscheidung
V1 (Phase 4): Debye-Hückel mit Switching, als Template-Parameter des Non-Bonded-Kernels. V2 (Phase 9): GB-OBC2 mit Salzkorrektur, gleiche Kernel-Schnittstelle (zusätzliche Born-Radien-Passes). Mg²⁺ explizit als Teilchen.

## Konsequenzen
- Ergebnisse aus V1 zu Konformationsgleichgewichten sind qualitativ; das muss in Publikationen/Beispielen benannt werden.
- Die Non-Bonded-Architektur muss mehrere Passes (GB) unterstützen.
