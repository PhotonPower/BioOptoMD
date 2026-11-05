# ADR 0010: Generalized Born OBC2 mit Salz und Cutoff aus κ

- **Status:** Accepted (Projektinhaber 2026-11-05, mit Abnahme Phase 9.1)
- **Datum:** 2026-11-05
- **Bezug:** ADR 0004 (V2 = GB-OBC2), ADR 0006 (Cutoff aus κ, Bildsumme), OQ-15, OQ-16

## Kontext
ADR 0004 legt für V2 GB-OBC2 mit Salzkorrektur fest, aber nicht die Kopplung an das Cutoff-/Switching-
Konzept (ADR 0006), die Behandlung ausgeschlossener Paare, den unpolaren Term und die Parameterquelle.
Referenz ist OpenMM `implicit/obc2.xml` (AMBER `igb=5`, `implicitSolventKappa`).

## Entscheidung (Projektinhaber 2026-11-05: OBC2 erweiterbar, ACE optional mit Default an)

**1. Modell (wie AMBER igb = 5 / OpenMM `GBSAOBC2Force`).** Mit ρ_i (intrinsischer Radius), Offset
δ = 0.009 nm, or_i = ρ_i − δ, sr_j = s_j · or_j (HCT-Skalierung), C = ONE_4PI_EPS0:

```
I_i  = Σ_{j≠i} H(r_ij, or_i, sr_j)                              (alle Paare, auch ausgeschlossene)
H    = ½ [1/L − 1/U + ¼ (r − sr²/r)(1/U² − 1/L²) + ½ ln(L/U)/r]   falls r + sr > or, sonst 0
       U = r + sr, L = max(or, |r − sr|)
B_i  = 1 / (1/or_i − tanh(ψ − 0.8 ψ² + 4.85 ψ³)/ρ_i),  ψ = I_i or_i
E_self,i = −½ C (1/ε_in − e^{−κ B_i}/ε_out) q_i² / B_i
E_ACE,i  = 4π γ (ρ_i + 0.14 nm)² (ρ_i/B_i)⁶,  4πγ = 28.3919551 kJ/mol/nm² (optional, Default an)
E_GB,ij  = −C (1/ε_in − e^{−κ f}/ε_out) q_i q_j / f,  f = sqrt(r² + B_i B_j exp(−r²/(4 B_i B_j)))
E_C,ij   = C q_i q_j / (ε_in r)                                  (nur nicht ausgeschlossene Paare)
```

ε_in = 1, ε_out = ε_r(T_r) und κ_r je Replica (OQ-10). 1-4-Coulomb im Bonded-Kernel unabgeschirmt
(κ = 0, ε_r = ε_in, AMBER-Skalierung). Für r ≫ B gilt E_C + E_GB → C q_i q_j e^{−κr}/(ε_out r)
(Debye-Hückel), daher bleibt das Cutoff-Konzept von ADR 0006 gültig.

**2. Cutoff.** Die Switching-Funktion S(r) (r_s, r_c aus κ) wirkt auf (E_LJ + E_C + E_GB) nicht
ausgeschlossener Paare; ausgeschlossene Paare (alle kürzer als r_s) tragen E_GB ohne Switching bei.
Die Beiträge zu den Born-Integralen werden mit derselben Switching-Funktion geglättet, H·S(r)
(H ~ sr³/r⁴, Änderung gegenüber dem harten Schnitt von OpenMM < 10⁻⁴ relativ in I). Ein harter
Schnitt erzeugte bei jedem Kreuzen von r_c einen Energiesprung ~ dE/dI · H(r_c) (≈ 2·10⁻³ kJ/mol bei
r_c = 2 nm); im NVE-Test mit r_c kleiner als das Molekül ergab das eine Drift von 2·10⁻⁴
kJ/mol/ps/Atom (Kriterium 10⁻⁴), daher die Glättung. Die Engulfment-Korrektur 2(1/or − 1/L) für r < or − sr fehlt wie in OpenMMs
`CustomGBForce` (gleiche Referenz). Numerik: Für nicht ausgeschlossene Paare wird E_C + E_GB
auslöschungsfrei als C qq [(1/ε_in) δ/(r f) + e^{−κf}/(ε_out f)] mit δ = f − r gerechnet. Bei L < 2 r_c werden I_i, E_GB und E_C über periodische Bilder summiert (ADR 0006);
Eigenbilder tragen zu I_i voll (H(|n|, or_i, sr_i)) und zu E_GB mit ½ bei.

**3. Kernels** (gleiche Tile-Struktur, Exclusion-Masken und Vorauswahl wie `TiledAllPairs`):
(a) Tile-Pass I_i (Fixed-Point, beide Richtungen), (b) elementweise B_i, dB_i/dI_i, Selbst- und
ACE-Energie, ∂E/∂B_i (Selbstanteil), (c) Tile-Pass Paarenergie/-kräfte bei festen B und ∂E/∂B_i
(Fixed-Point), (d) dE/dI_i = ∂E/∂B_i · dB_i/dI_i (beim Laden in Pass e), (e) Tile-Pass Kettenregel
F_i −= Σ_j [dE/dI_i ∂H(r, or_i, sr_j)/∂r + dE/dI_j ∂H(r, or_j, sr_i)/∂r] r̂_ij. Alle Akkumulationen
über Fixed-Point (ADR 0002) ⇒ deterministisch.

**4. Erweiterbarkeit.** Das Born-Radien-Modell ist ein Template-Parameter (`ObcII`); GBn2 (Neck-
Integral, mbondi3) kann als zweite Variante folgen (OQ-15).

**5. Parameterquelle.** ρ_i und s_i aus der OpenMM-`CustomGBForce` von `implicit/obc2.xml`
(mbondi2-Radien); der Builder liest sie aus dem System.

## Alternativen
- **OpenMM-Konvention mit (1/f − 1/r_c)-Verschiebung:** nur für den GB-Term, nicht für Coulomb; passt
  nicht zum Switching des kombinierten Paars und erzeugt Kraftsprünge. Verworfen.
- **GBn2 sofort:** besser für Nukleinsäuren, aber mehr Aufwand; vom Projektinhaber zurückgestellt.
- **Gemischtes DH + GB:** doppelte Abschirmung; verworfen.

## Konsequenzen
- Neue Nonbonded-Strategie `TiledGeneralizedBorn`; DH bleibt Default.
- Im GB-Modus fehlt die dielektrische Grenze der Oberfläche (Wand als Lösungsmittel behandelt): OQ-16.
- Abnahmetests: ROADMAP Phase 9.1.
