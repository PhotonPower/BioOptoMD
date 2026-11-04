# ADR 0003: Periodizität in x/y, nicht periodisch in z

- **Status:** Accepted
- **Datum:** 2026-11-03

## Kontext
Biosensor-Oberflächen tragen viele Rezeptoren; Oberflächenbelegung und das laterale Mittel für n(z) erfordern eine unendlich ausgedehnte Fläche. Nachträgliches Einbauen von PBC in Paar-Kernels ist fehleranfällig.

## Entscheidung
Von Phase 1 an: Box (L_x, L_y) pro Replica, Minimum Image in allen Paar-Berechnungen (Non-Bonded, 1-4, Restraints, FRET-Abstand). z ist nicht periodisch; die Oberfläche liegt bei z = 0. Für Einzelmolekül-Simulationen wird eine große Box gewählt (L ≥ Molekülgröße + 2 r_c).

## Nachtrag (2026-11-04, ADR 0006 Rev. 2)
Für **Non-Bonded-Paare** (LJ + Debye-Hückel) gilt Minimum Image nur, solange L_x, L_y ≥ 2 r_c. Ist die Box kleiner – sie wird aus der gewünschten Bedeckung gewählt und nie automatisch vergrößert –, summiert der Non-Bonded-Kernel explizit über periodische Bilder in x/y (Details: ADR 0006). Bonded-Terme, 1-4, Restraints und Observer bleiben bei Minimum Image. Die Bedingung „L ≥ 2 r_c“ unter Konsequenzen gilt nur noch für diese Terme bzw. für den Minimum-Image-Pfad.

## Konsequenzen
- Bedingung L_x, L_y ≥ 2 r_c wird beim Setup geprüft.
- Imagezähler für ungewrappte Trajektorien.
- Bindungen über Boxgrenzen hinweg müssen Minimum Image nutzen.
