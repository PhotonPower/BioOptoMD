# ADR 0007: Wählbare Wandform und Parameter aus Mischungsregeln

- **Status:** Proposed
- **Datum:** 2026-11-04
- **Bezug:** OQ-05, OQ-07

## Kontext
PHYSICS_MODEL §4.1 verwendet für alle Oberflächen eine 9-3-Wand. Diese entsteht durch Integration eines LJ-Potentials über einen **Halbraum** homogen verteilter Atome und ist für Gold (Volumenkristall) angemessen. Eine Graphen-Monolage ist dagegen eine einzelne Atomebene; ihr integriertes Potential ist das 10-4-Potential. Für mehrlagiges Graphit gilt Steeles 10-4-3-Potential. Die 9-3-Form für Graphen hat einen falschen Abfall (z⁻³ statt z⁻⁴) und damit eine falsche Reichweite der Adsorption.

Zusätzlich sind ε_w, σ_w bisher freie Parameter pro Oberflächen-Typ (Platzhalter).

## Entscheidung

**1. Wandform als Template-Parameter von `PlanarSurfaceForce`** (kein virtueller Device-Code, AGENTS.md Regel 9):

| Form | Potential V_i(z) | Anwendung |
|---|---|---|
| `Wall93` | (2π/3) ρ ε_ij σ_ij³ [ (2/15)(σ_ij/z)⁹ − (σ_ij/z)³ ] | Gold, Volumenmaterial (ρ: Atomdichte, nm⁻³) |
| `Wall104` | 2π ρ_s ε_ij σ_ij² [ (2/5)(σ_ij/z)¹⁰ − (σ_ij/z)⁴ ] | Graphen-Monolage (ρ_s ≈ 38,2 nm⁻²) |
| `Wall1043` | 2π ρ ε_ij σ_ij² Δ [ (2/5)(σ_ij/z)¹⁰ − (σ_ij/z)⁴ − σ_ij⁴ / (3Δ (z + 0,61Δ)³) ] | Graphit (Δ = 0,335 nm Lagenabstand) |

Kräfte (F_z = −dV/dz):
- `Wall93`: F_z = (2π/3) ρ ε σ³ [ (6/5) σ⁹/z¹⁰ − 3 σ³/z⁴ ]
- `Wall104`: F_z = 2π ρ_s ε σ² · 4 [ σ¹⁰/z¹¹ − σ⁴/z⁵ ]; Minimum bei z* = σ, V(z*) = −(6/5) π ρ_s ε σ²
- `Wall1043`: wie `Wall104` (mit ρΔ statt ρ_s) plus Term −2π ρ ε σ⁶ / (z + 0,61Δ)⁴

Der z-Clamp (Default 0,1 σ_w) und der Diagnosezähler bleiben für alle Formen gleich.

**2. Parameter aus Mischungsregeln:** Für Atomtyp i und Wandatom w: σ_iw = (σ_i + σ_w)/2, ε_iw = √(ε_i ε_w) (Lorentz-Berthelot, konsistent mit dem Kraftfeld). Der Builder berechnet die effektiven Vorfaktoren pro (LJ-Typ, Oberflächen-Typ) und lädt sie als Tabelle hoch. Wandatom-Parameter (vorläufig, zu verifizieren, siehe OQ-Checkliste): Kohlenstoff nach Steele, Gold nach INTERFACE-FF (Heinz et al.).

**3. Lösungsmittelkorrektur:** Pro Oberflächen-Typ und Atomklasse optionaler Skalierungsfaktor λ ∈ (0, 1] auf ε_iw. Kalibrierung an Adsorptions-PMFs (Nukleobasen/Nukleoside) in **explizitem** Wasser. Default λ = 1 mit `ASSUMPTION(OQ-05)`-Kommentar. Für Gold mit INTERFACE-FF-Parametern ist λ < 1 praktisch Pflicht, weil diese Parameter in explizitem Wasser validiert sind (Review 2).

**4. Ausnahmen pro Atom:** Oberflächen-Typ 0 (ε = 0) ist reserviert für Atome, die nicht mit der Wand wechselwirken (Padding-Atome, Thiol-Ankeratom nach OQ-07).

**5. Lage von z = 0 (Review 2):** Für `Wall93` liegt die Kontinuumsgrenze z = 0 einen halben Lagenabstand unter der obersten Atomlage (Au(111): Δ/2 ≈ 0,118 nm mit Δ = a/√3 ≈ 0,2355 nm). Für `Wall104` liegt die Atomebene bei z = 0, für `Wall1043` die oberste Lage. Der Builder dokumentiert z_top (Höhe der obersten Atomlage) pro Material. Alle Höhenangaben (Chemisorption OQ-07, SAM-Wand, Observer-Ursprung d) beziehen sich auf z_top, nicht auf z = 0.

## Alternativen
- **Nur 9-3, Graphen per Fit:** Ein Fit kann den falschen Abfall nicht korrigieren; Parameter wären nicht übertragbar.
- **Explizite Wandatome:** physikalisch genauer (Korrugation), kostet aber Atome und zerstört das Batch-Regime.

## Konsequenzen
- PHYSICS_MODEL §4.1 wird um die Formen erweitert; die Builder-API erhält `wall_form` und `material`.
- Korrugation der Oberfläche (laterale Struktur) bleibt vernachlässigt; bei Bedarf über `VoxelGridForce3D`.
- Neue Abnahmetests Phase 2:
  - `test_phase2.py::test_wall_forms_vs_numpy` – jede Form: Energie und Kraft vs. NumPy-Referenz, Kraft vs. finite Differenzen, rel. Fehler < 1e-4; Lage und Tiefe des Minimums analytisch.
  - `test_phase2.py::test_wall_mixing_rules` – Builder-Tabelle vs. manuell berechnete Lorentz-Berthelot-Werte.
  - `test_wall_boltzmann` wird für alle drei Formen parametrisiert.
