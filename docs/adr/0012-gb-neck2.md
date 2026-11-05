# ADR 0012: GB-neck2 (GBn2) als zweites GB-Modell

- **Status:** Accepted (Projektinhaber 2026-11-05, mit Abnahme Phase 9.3)
- **Datum:** 2026-11-05
- **Bezug:** OQ-15, ADR 0010 (GB-OBC2, Kernel und Cutoff-Konzept), ROADMAP Phase 9.3

## Kontext
OQ-15: Für Nukleinsäuren gilt GB-neck2 (AMBER `igb = 8`, Nguyen, Pérez, Bermeo & Simmerling 2015,
Nukleinsäure-Parameter; Zitat aus dem Gedächtnis, nicht verifiziert) als deutlich besser als OBC2. Der Projektinhaber hat GBn2 am 2026-11-05 als
nächste Erweiterung gewählt; ADR 0010 sah den Kernel dafür als erweiterbar vor.

## Entscheidung
1. **Modell wie OpenMM `implicit/gbn2.xml` / `GBSAGBn2Force`:** Offset δ = 0.0195141 nm, mbondi3-Radien,
   HCT-Skalierung und (α, β, γ) des tanh je Atom (eigene Sätze für Nukleinsäure-Reste), zusätzlich zum
   HCT-Integral der Neck-Term
   `I_i += 0.826836 · m0_ij / (1 + 100 (r − d0_ij)² + 3·10⁵ (r − d0_ij)⁶)` für r < ρ_i + ρ_j + 0.68 nm.
   Salzkorrektur, ACE (mit ρ = or + δ), Kopplung an Cutoff/Switching (inkl. Glättung der Born-
   Integrale mit S(r)) und Bildsumme wie ADR 0010.
2. **Parameter und Tabellen aus OpenMM:** Der Builder liest ρ, s, α, β, γ, den Radiusindex und die von
   OpenMM bereits interpolierten Tabellen getm0/getd0 (Discrete2DFunction) aus der `CustomGBForce`.
   Die Tabellen sind asymmetrisch; Index (Atom i, Atom j) wie `getm0(radindex1, radindex2)`.
3. **Neck-Stufe** bei ρ_i + ρ_j + 0.68 nm bleibt wie im AMBER-Modell (keine Glättung): NVE-Drift mit
   GBn2 gemessen ≈ 5·10⁻⁵ kJ/mol/ps/Atom (0.5 fs ohne Constraints und 1 fs mit Constraints), unter dem
   Kriterium 10⁻⁴ und gleichauf mit OBC2.
4. **Ein Kernel für beide Modelle:** OBC2 ist der Spezialfall ohne Neck-Tabelle mit α, β, γ = (1, 0.8, 4.85)
   und δ = 0.009 nm; Builder-Option `solvent="gb-gbn2"`.

## Alternativen
- **Eigene Neck-Tabellen aus der Literatur übernehmen:** Fehlerquelle bei Interpolation/Einheiten;
  OpenMM-Tabellen sind gegen AMBER validiert. Verworfen.
- **Neck-Stufe glätten:** ändert das Modell gegenüber AMBER ohne messbaren Nutzen (Drift bereits klein).

## Konsequenzen
- GBn2 kostet etwa so viel wie OBC2 (ein Tabellenzugriff je Paar innerhalb ~1 nm zusätzlich).
- Gültigkeit: Radien 0.1–0.2 nm (Tabellenbereich, OpenMM prüft das beim Aufbau).
- Abnahmetests: ROADMAP Phase 9.3.
