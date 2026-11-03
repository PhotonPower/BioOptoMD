# ADR 0003: Periodizität in x/y, nicht periodisch in z

- **Status:** Accepted
- **Datum:** 2026-11-03

## Kontext
Biosensor-Oberflächen tragen viele Rezeptoren; Oberflächenbelegung und das laterale Mittel für n(z) erfordern eine unendlich ausgedehnte Fläche. Nachträgliches Einbauen von PBC in Paar-Kernels ist fehleranfällig.

## Entscheidung
Von Phase 1 an: Box (L_x, L_y) pro Replica, Minimum Image in allen Paar-Berechnungen (Non-Bonded, 1-4, Restraints, FRET-Abstand). z ist nicht periodisch; die Oberfläche liegt bei z = 0. Für Einzelmolekül-Simulationen wird eine große Box gewählt (L ≥ Molekülgröße + 2 r_c).

## Konsequenzen
- Bedingung L_x, L_y ≥ 2 r_c wird beim Setup geprüft.
- Imagezähler für ungewrappte Trajektorien.
- Bindungen über Boxgrenzen hinweg müssen Minimum Image nutzen.
