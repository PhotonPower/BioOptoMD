# ADR 0006: Non-Bonded-Cutoff aus der Debye-Länge ableiten

- **Status:** Proposed
- **Datum:** 2026-11-04

## Kontext
PHYSICS_MODEL §3 legt r_s = 1,8 nm, r_c = 2,0 nm fest und begründet das damit, dass Debye-Hückel bei 150 mM (κ⁻¹ ≈ 0,785 nm) bei r_c „auf < 1 % abgefallen“ sei. Das trifft nicht zu:

| Bezug | Wert bei r_c = 2,0 nm, 150 mM |
|---|---|
| exp(−κ r_c), d. h. relativ zur reinen Coulomb-Wechselwirkung bei r_c | ≈ 7,8 % |
| V_DH(r_c) / V_DH(0,3 nm) | ≈ 1,7 % |

Für ein DNA-Rückgrat mit vielen Phosphaten summiert sich der abgeschnittene Anteil systematisch. Bei niedriger Ionenstärke wird der feste Cutoff unbrauchbar (10 mM: κ⁻¹ ≈ 3,0 nm > r_c).

Der Tiled-All-Pairs-Kernel (Phase 4) berechnet ohnehin alle Paare; der **Rechenaufwand hängt nicht vom Cutoff ab**. Ein größerer Cutoff kostet also nur über die Box-Bedingung L_x, L_y ≥ 2 r_c (ADR 0003).

## Entscheidung
Der Builder setzt pro Simulation

```
r_c = max(2.0 nm, n_κ / κ)      mit n_κ = 5 (Default, konfigurierbar)
r_s = r_c − 0.2 nm
```

Bei n_κ = 5 beträgt der Restanteil exp(−5) ≈ 0,7 % relativ zu Coulomb bei r_c. Die Box-Bedingung L ≥ 2 r_c wird beim Setup geprüft; ist die Box zu klein, bricht der Builder mit einer Fehlermeldung ab, die das nötige L nennt (kein stilles Verkleinern von r_c).

LJ verwendet denselben Cutoff (gemeinsame Switching-Funktion, ein Kernel).

## Alternativen
- **Fester Cutoff 2,0 nm (Status quo):** einfach, aber systematischer Fehler (s. o.) und bei niedriger Salzkonzentration falsch.
- **Kein Cutoff (nur Minimum Image):** bei kleiner Box unstetig an L/2; widerspricht ADR 0003.
- **Ewald/PME mit Abschirmung:** unnötig komplex für implizites Lösungsmittel mit endlichem κ.

## Konsequenzen
- Bei niedriger Ionenstärke wachsen die Boxen (10 mM → r_c ≈ 15 nm → L ≥ 30 nm). Für Einzelmolekül-Simulationen akzeptabel, da die Atomzahl (implizites Wasser) nicht steigt; beim Cell-List-Strategy (Phase 9+) wird der Aufwand jedoch cutoffabhängig – dann neu bewerten.
- PHYSICS_MODEL §3 wird korrigiert (falsche „< 1 %“-Aussage entfernt).
- Neuer Abnahmetest Phase 4: `test_phase4.py::test_cutoff_from_kappa` – Builder setzt r_c korrekt für I = 10, 150, 1000 mM; Box zu klein → Fehler; Energie gegenüber Referenz ohne Cutoff (große Box) mit relativer Abweichung < 1 %.
