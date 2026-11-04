# ADR 0006: Non-Bonded-Cutoff aus der Debye-Länge, Box aus der Bedeckung

- **Status:** Proposed (Rev. 2)
- **Datum:** 2026-11-04
- **Bezug:** OQ-09, OQ-10; ändert ADR 0003 (Minimum Image) für Non-Bonded-Paare

## Kontext
PHYSICS_MODEL §3 legte r_s = 1,8 nm, r_c = 2,0 nm fest und begründete das damit, dass Debye-Hückel bei 150 mM (κ⁻¹ ≈ 0,785 nm) bei r_c „auf < 1 % abgefallen“ sei. Das trifft nicht zu:

| Bezug | Wert bei r_c = 2,0 nm, 150 mM |
|---|---|
| exp(−κ r_c), d. h. relativ zur reinen Coulomb-Wechselwirkung bei r_c | ≈ 7,8 % |
| V_DH(r_c) / V_DH(0,3 nm) | ≈ 1,7 % |

Für ein DNA-Rückgrat mit vielen Phosphaten summiert sich der abgeschnittene Anteil systematisch. Bei niedriger Ionenstärke wird der feste Cutoff unbrauchbar (10 mM: κ⁻¹ ≈ 3,0 nm > r_c).

Der Tiled-All-Pairs-Kernel (Phase 4) berechnet ohnehin alle Paare; der Rechenaufwand hängt nicht vom Cutoff ab.

**Rev. 2:** Rev. 1 löste das Problem über die Box-Bedingung L ≥ 2 r_c. Bei einem Rezeptor pro Box legt L aber die Oberflächenbelegung Γ = 1/L² fest (OQ-09). Bei 10 mM erzwang Rev. 1 L ≥ 30 nm (Γ ≈ 1·10¹¹ cm⁻²), während reale Sensoren oft bei 10¹²–10¹³ cm⁻² liegen. Die Box ist eine physikalische Eingabe und darf nicht aus numerischen Gründen wachsen.

## Entscheidung (Vorschlag)

**1. Cutoff aus der Debye-Länge, pro Replica-Satz einheitlich**

```
κ_r  aus I, T_r, ε_r(T_r)               (OQ-10; sonst global)
r_c  = max(2.0 nm, n_κ / min_r κ_r)      n_κ = 5 (Default, konfigurierbar)
r_s  = r_c − 0.2 nm
```

Bei n_κ = 5 beträgt der Restanteil exp(−5) ≈ 0,7 % relativ zu Coulomb bei r_c. Ein gemeinsames r_c für alle Replicas hält den Kernel einheitlich. LJ verwendet denselben Cutoff (gemeinsame Switching-Funktion).

**2. Box aus der Bedeckung**

Der Builder nimmt L_x, L_y (oder Γ) als physikalische Eingabe und verändert sie nie.

**3. Periodische Bilder, wenn L < 2 r_c**

- Ist L_x, L_y ≥ 2 r_c: Minimum Image wie bisher.
- Sonst summiert der Non-Bonded-Kernel für jedes Paar (i, j), ausgehend vom Minimum-Image-Abstand r_ij, über alle Gittervektoren **n** = (n_x L_x, n_y L_y) mit |n_x| ≤ ⌈r_c/L_x + ½⌉, |n_y| ≤ ⌈r_c/L_y + ½⌉ und |r_ij + **n**| < r_c. (Das „+ ½“ ist nötig, weil r_ij selbst bis L/2 pro Richtung betragen kann.)
- **Eigenbilder** (i = j, **n** ≠ 0) werden mit Faktor ½ mitgezählt; ihre Kraft verschwindet aus Symmetriegründen, die Energie nicht.
- **Exclusions** und 1-4-Ausschluss gelten nur für **n** = 0.
- Bonded-Terme, Restraints und Observer (FRET-Abstand) bleiben bei Minimum Image (ADR 0003 unverändert für diese).

Kosten: Faktor ≈ (2 n_max + 1)² Bildverschiebungen pro Tile, z. B. 49 bei 10 mM (r_c ≈ 15,2 nm) und L = 10 nm. Bei 150 mM und L ≥ 7,9 nm entfällt die Bildsumme ganz. Der teure Fall tritt also nur bei niedrigem Salz und hoher Bedeckung auf – genau dort, wo er physikalisch nötig ist.

## Alternativen
- **Fester Cutoff 2,0 nm (Status quo):** systematischer Fehler (s. o.), bei niedriger Salzkonzentration falsch.
- **Rev. 1 – Box vergrößern (L ≥ 2 r_c):** einfach, verfälscht aber die Bedeckung und unterdrückt Nachbarwechselwirkungen.
- **Verkleinern von r_c auf L/2:** stiller Genauigkeitsverlust; abgelehnt.
- **2D-Ewald mit Abschirmung:** unnötig, da die abgeschirmte Gittersumme exponentiell konvergiert.

## Konsequenzen
- ADR 0003 muss für Non-Bonded-Paare angepasst werden (Entscheidung des Projektinhabers, AGENTS.md: ADRs > ARCHITECTURE).
- PHYSICS_MODEL §3 wird korrigiert.
- Bei Cell-List-Strategien (Phase 9+) wird der Aufwand cutoffabhängig – dann neu bewerten.
- Abnahmetests Phase 4 (ersetzen den Test aus Rev. 1):
  - `test_phase4.py::test_cutoff_from_kappa` – r_c korrekt für I = 10/150/1000 mM und für Replicas unterschiedlicher T; Builder ändert L nie.
  - `test_phase4.py::test_image_sum_vs_lattice_reference` – elektrostatische Energie und Kräfte vs. NumPy-Gittersumme mit großem n_max; rel. RMS-Kraftfehler < 1e-3, elektrostatische Energie rel. < 1e-3 (Gesamtenergie ist als Maß ungeeignet).
  - `test_phase4.py::test_image_sum_self_images` – einzelnes geladenes Atom in kleiner Box: Energie = ½ Σ_{n≠0} V_DH(|n|), Kraft = 0.
  - `test_phase4.py::test_image_sum_reduces_to_minimum_image` – für L ≥ 2 r_c sind beide Pfade bis auf Rundung identisch.
