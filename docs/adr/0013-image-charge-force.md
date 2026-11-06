# ADR 0013: ImageChargeForce – abgeschirmte Bildladungen an leitfähigen Oberflächen

- **Status:** Accepted (Projektinhaber 2026-11-05, Kriterien ROADMAP 9.5 und Defaults OQ-18 bestätigt; Präzisierungen der Punkte 6, 7, 9 bestätigt und Phase 9.5 abgenommen 2026-11-06)
- **Datum:** 2026-11-05
- **Bezug:** OQ-16 (dielektrische Grenze), OQ-18 (Bildebene, Graphen), ADR 0003 (x/y-Periodizität),
  ADR 0006 (Cutoff aus κ, Bildsumme), ADR 0007 (Wandformen), ADR 0008 (Kraftschranken),
  ADR 0010/0012 (GB), ADR 0011 (Replica Exchange), PHYSICS_MODEL §3 „Bekannte Grenzen“, ROADMAP 9.5

## Kontext
Die Elektrostatik behandelt den Halbraum unter der Wand bisher als Lösungsmittel. An einer Metall-
elektrode (Gold) ist das falsch: Das Metall hält sein Potential fest und influenziert Ladungen, die
jede Ladung des Moleküls zur Oberfläche ziehen. Für einzelne Ladungen ist der Effekt in Wasser klein,
aber er ist kollektiv: Ein Polyelektrolyt wie DNA spürt die Bilder aller seiner Phosphate. Eine
Abschätzung für ε_r = 78.5 und 0.15 M (κ = 1.28 nm⁻¹):

| Situation | Bildenergie |
|---|---|
| eine Elementarladung, 0.2 / 0.3 / 0.5 nm über der Bildebene | −1.3 / −0.7 / −0.25 kJ/mol |
| 12 Phosphate (−1 e, Abstand 0.7 nm) in 0.6 nm Höhe | ≈ −5.6 kJ/mol ≈ −2.2 k_BT (ohne Cutoff; mit r_c = 2 nm −5.1) |

Der Gouy-Chapman-Term (PHYSICS_MODEL §4.1) beschreibt nur das Feld der Oberflächenladung, nicht die
Antwort des Metalls auf das Molekül. PHYSICS_MODEL §3 führt `ImageChargeForce` deshalb als V2 auf.

## Entscheidung

1. **Modell: abgeschirmte Bildladungen an einer leitfähigen Ebene (Dirichlet-Rand).** Für Ladungen q_i
   bei z_i > z_im in einem Elektrolyten mit (ε_r, κ) und einem idealen Leiter auf konstantem Potential
   bei z ≤ z_im ist die Green'sche Funktion des linearisierten Poisson-Boltzmann-Problems im Halbraum
   exakt `G(r, r') = G_κ(|r − r'|) − G_κ(|r − r̄'|)`, mit G_κ(r) = k_e e^{−κr}/(ε_r r),
   k_e = ONE_4PI_EPS0 und dem Spiegelbild r̄ = (x, y, 2 z_im − z). Begründung: Die Gleichung
   (∇² − κ²)φ = −ρ/ε ist translations- und spiegelinvariant, das Bild verschwindet auf der Ebene genau
   (numerisch geprüft: φ(z_im) = 0 bis auf Rundung). Das ist im Rahmen des Debye-Hückel-Modells keine
   Näherung. Zusätzliche Energie:

   ```
   U_im = −f · [ Σ_{i<j} q_i q_j G_κ(s_ij) S(s_ij) + ½ Σ_i q_i² G_κ(2ζ_i) S(2ζ_i) ]
   s_ij = |(Δx_ij, Δy_ij, ζ_i + ζ_j)|  (Δx, Δy mit Minimum Image bzw. Bildsumme),  ζ_i = z_i − z_im
   ```

   Kräfte analytisch. Die z-Kräfte eines Paars haben dasselbe Vorzeichen (beide Atome werden zur Wand
   gezogen); der Impuls in z ist nicht erhalten, das ist bei einer externen Wand korrekt.

2. **Bildfaktor f ∈ [−1, 1].** f = 1 ist der ideale Leiter (Default für Gold). f < 1 ist eine
   Näherung: Für einen dielektrischen Halbraum (ε_m) gilt ohne Salz f = (ε_m − ε_r)/(ε_m + ε_r) exakt.
   Mit κ > 0 gibt es dafür keine einfache Bildlösung, f ist dann ASSUMPTION(OQ-18). Ein f < 0
   (ε_m < ε_r, z. B. eine hydrophobe Isolatorschicht) liefert abstoßende Bilder und damit eine grobe
   dielektrische Ausschlusswirkung (Teil von OQ-16). Für Graphen und Graphit gibt es keinen Default:
   Graphen ist ein unvollständiger Leiter, f muss der Nutzer angeben (OQ-18).

3. **Bildebene z_im** als Parameter, relativ zur obersten Atomlage z_top des Materials (ADR 0007
   Punkt 5): `z_im = z_top + d_im`. ASSUMPTION(OQ-18): d_im = 0 als Default. Die physikalische Lage
   (bei Metallen typischerweise einige Zehntel Å außerhalb der Atomlage, materialabhängig) ist nicht
   verifiziert; das Bild ist für ζ ≲ 0.1 nm ohnehin nicht mehr gültig.

4. **Alle Paare, keine Exclusions.** Die Bildwechselwirkung ist eine echte Polarisation des Metalls,
   keine Korrektur der intramolekularen Coulomb-Terme; auch gebundene Atome und das Atom selbst
   (Eigenbild, Faktor ½) zählen. Atome mit q = 0 tragen nicht bei.

5. **Cutoff, Switching, Periodizität wie der Non-Bonded-Kernel.** r_s, r_c und S(r) aus ADR 0006 und
   PHYSICS_MODEL §3 (gleiche Werte wie die Nonbonded-Strategie); Minimum Image in x/y für L ≥ 2 r_c,
   sonst Bildsumme über |n| ≤ ⌈r_c/L + ½⌉ (ADR 0006 Rev. 2). Weil s_ij ≥ ζ_i + ζ_j, tragen nur Atome
   mit ζ < r_c bei; die Tile-Vorauswahl nutzt zusätzlich die untere Schranke ζ_min,i + ζ_min,j aus den
   Tile-Hüllen (`tile_common.cuh`).

6. **Schutz bei ζ → 0 (ADR 0008).** ζ wird auf ζ_c = 0.1 nm geclampt (Energie stetig, Kraft in z dann 0),
   jedes Clamping zählt ein Device-Zähler wie bei der Wand; ζ_c >= 0.02 nm wird erzwungen.
   *Präzisierung nach der Implementierung (vom Projektinhaber am 2026-11-06 bestätigt):* An Gold
   mit d_im = 0 erreichen H-Atome ζ < ζ_c regelmäßig (σ_iw(H) ≈ 0.19 nm); ζ_c ist dort ein
   Modellparameter und keine reine Schutzgrenze, ASSUMPTION(OQ-18) Punkt (d). Für q = 1 und ε_r = 78.5 ist die Eigenbild-
   kraft bei ζ_c ≈ 45 kJ/mol/nm, weit unter F_max. Ankeratome (Oberflächen-Typ 0, OQ-07) nehmen teil,
   sofern sie geladen sind; liegt ihre Höhe unter z_im + ζ_c, greift das Clamping.

7. **Implementierung als ForcePlugin** `src/forces/image_charge.{hpp,cu}`, ein eigener Tile-Pass über
   alle Paare (i ≤ j). j-Koordinaten werden beim Laden in den Shared Memory gespiegelt, sonst wie
   `TiledAllPairs` (Fixed-Point-Akkumulation, ADR 0002; Kernel deterministisch). κ_r und ε_r je
   Replica, `set_screening` für Replica Exchange (ADR 0011). Python: `ctx.add_image_charges(z_image,
   factor, kappa, eps_r, r_switch, r_cutoff)`; Builder-Funktion `builder.add_image_charges(ctx, info,
   material)` (statt der zuerst genannten Option `image_charges=True`, weil `build_context` keine
   Wand anlegt; bestätigt 2026-11-06) übernimmt z_top aus dem Material, κ, ε_r und r_c aus `BuildInfo`.

8. **Kombination mit GB (OQ-16).** Mit `solvent="gb-*"` nutzt der Bildterm ε_r = ε_out und κ des
   GB-Modells, also das Fernfeld von GB (das in Debye-Hückel übergeht). Die Desolvatation an der Wand
   fehlt weiterhin; ASSUMPTION(OQ-16).

9. **Kombination mit Gouy-Chapman.** Im linearen Modell überlagern sich die Felder: Gouy-Chapman setzt
   das Potential der diffusen Schicht Φ₀, die Bilder halten das Potential des Moleküls auf der Ebene
   bei 0. Das entspricht einer Elektrode auf konstantem Potential, streng nur, wenn beide Ebenen
   zusammenfallen: Gouy-Chapman hat seinen Ursprung bei z0, die Bildebene liegt bei
   z0 + z_top + d_im (Gold: 0.118 nm darüber). Mit nichtlinearem Gouy-Chapman ist
   die Überlagerung eine Näherung (gültig, solange die Ladungen des Moleküls das Profil nur linear
   stören).

## Alternativen
- **Constant-Potential-Methode mit fluktuierenden Gauß-Ladungen auf Wandatomen** (Siepmann-Sprik,
  Reed et al.): genauer an der Atomlage, braucht aber explizite Wandatome, ein lineares
  Gleichungssystem je Schritt und explizites Wasser. Passt nicht zum Kontinuumsmodell mit flacher Wand.
- **Polarisierbares Gold mit starren Dipolen (GolP-CHARMM, OQ-05):** ebenfalls atomistisch und für
  explizites Wasser parametrisiert; bleibt Referenz für die Kalibrierung.
- **Nur Eigenbilder (Einkörperterm im Wandkernel):** billig, aber für Polyelektrolyte falsch, weil
  die Paarterme das gleiche Gewicht haben (bei den 12 Phosphaten oben ≈ 66 % von U_im). Verworfen.
- **Bildterm im Non-Bonded-Kernel mitrechnen:** spart einen Pass, koppelt aber die Wand an alle
  Nonbonded-Strategien (DH, OBC2, GBn2, später Cell-List). Verworfen zugunsten eines Plugins.
- **Dielektrische Grenze mit exakter Green'scher Funktion (Sommerfeld-Integral):** für f ≠ 1 und κ > 0
  korrekt, aber eine tabellierte 2D-Funktion je (κ, ε_m/ε_r); später möglich, falls OQ-18 das verlangt.

## Konsequenzen
- Im Debye-Hückel-Modell ist die Leiterwand damit exakt beschrieben, im GB-Modell nur im Fernfeld.
- Kosten: ein zusätzlicher Tile-Pass, etwa so teuer wie der Debye-Hückel-Pass, für Atome weiter als r_c
  von der Wand nahezu null (Vorauswahl). Messung gehört zur Abnahme.
- Nicht modelliert: Stern-Schicht und Ionenausschluss direkt an der Wand, reduziertes ε von Wasser an
  der Grenzfläche (beides würde den Effekt verstärken), Metall-Thomas-Fermi-Abschirmung.
- Abnahmetests: ROADMAP Phase 9.5.
