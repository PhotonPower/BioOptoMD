# Physikalisches Modell

Alle Formeln in den Einheiten aus `UNITS_AND_CONVENTIONS.md`. Kraftfeld-Konventionen folgen **OpenMM** (Faktor ½ bei harmonischen Termen), damit OpenMM direkt als Referenz dienen kann.

---

## 1. Integrator: Langevin BAOAB

Pro Replica r mit Temperatur T_r, Reibung γ, Zeitschritt Δt:

```
c1 = exp(-γ Δt)
c2 = sqrt((1 - c1²) · k_B T_r / m_i)
B:  v ← v + (Δt/2) F/m
A:  x ← x + (Δt/2) v
O:  v ← c1 v + c2 ξ,     ξ ~ N(0,1) aus Philox(seed, r, step, i)
A:  x ← x + (Δt/2) v
    F ← F(x)
B:  v ← v + (Δt/2) F/m
```

- γ = 0 ⇒ Velocity Verlet (NVE) – Grundlage für Energieerhaltungs-Tests.
- Mit Constraints (Phase 5): Positions-Constraints nach jedem A, Geschwindigkeits-Constraints (RATTLE) nach jedem B und nach O.
- Atome mit `inv_mass = 0` bewegen sich nicht und erhalten kein Rauschen.
- Freiheitsgrade pro Replica: `N_dof = 3·N_mobile − N_constraints` (kein Abzug für Schwerpunkt, da die Wand Translationsinvarianz in z bricht und Langevin ohnehin keinen Impuls erhält).

## 2. Kovalente Terme (Phase 3)

| Term | Energie | Bemerkung |
|---|---|---|
| Bond | ½ k_b (r − r₀)² | |
| Angle | ½ k_θ (θ − θ₀)² | θ über `atan2(|a×b|, a·b)` |
| Periodic torsion | k_φ [1 + cos(nφ − φ₀)] | Impropers in AMBER ebenfalls periodisch |
| 1-4 Paare | LJ·s_LJ + Coulomb·s_el | explizite Paarliste; AMBER: s_el = 1/1.2, s_LJ = 1/2; Elektrostatik der 1-4-Paare mit demselben Debye-Hückel-Kernel wie Non-Bonded |

## 3. Nicht-kovalente Terme (Phase 4)

**Lennard-Jones:** `V = 4ε_ij[(σ_ij/r)¹² − (σ_ij/r)⁶]`, Kombination nach Lorentz-Berthelot, vorab als Typ-Paar-Tabelle (`N_types × N_types`, float2) im Shared/Constant Memory.

**Elektrostatik V1 – Debye-Hückel (implizites Wasser mit Salz):**
`V = ONE_4PI_EPS0 · q_i q_j · exp(−κ r) / (ε_r · r)`, ε_r = 78.5 (Default), κ aus Ionenstärke. κ enthält alle nicht explizit modellierten Ionen. Temperaturabhängigkeit von κ und ε_r pro Replica: OQ-10.

**Cutoff:** Beide Terme mit Switching-Funktion zwischen r_s und r_c. Der Cutoff wird aus der Debye-Länge abgeleitet (ADR 0006): `r_c = max(2.0 nm, 5/κ)`, `r_s = r_c − 0.2 nm`. Damit ist DH bei r_c auf exp(−5) ≈ 0,7 % der reinen Coulomb-Wechselwirkung abgefallen. (Der frühere feste Wert r_c = 2,0 nm ließ bei 150 mM noch ≈ 8 % übrig.) Da der Tiled-All-Pairs-Kernel alle Paare rechnet, kostet ein größerer Cutoff keine Rechenzeit. Die Box L_x, L_y ist eine physikalische Eingabe (Bedeckung Γ = 1/(L_x L_y), OQ-09) und wird nie angepasst. Für L ≥ 2 r_c gilt Minimum Image; sonst summiert der Kernel explizit über periodische Bilder in x/y, inklusive Eigenbilder mit Faktor ½; Exclusions gelten nur für das Nullbild (ADR 0006 Rev. 2).

**Exclusions:** 1-2 und 1-3 vollständig ausgeschlossen, 1-4 im Non-Bonded-Kernel ausgeschlossen (werden in der Bonded-Stufe skaliert berechnet). Kodierung als 32-bit-Bitmaske pro (Atom, Tile).

**Explizite Ionen:** Divalente Ionen (Mg²⁺) können als echte Teilchen mit LJ-Parametern vorhanden sein; die monovalente Salzumgebung steckt in κ. **Achtung (OQ-08):** Ohne Hydrathülle kollabiert Mg²⁺ im impliziten Wasser in direkten Phosphatkontakt und bindet viel zu stark. Bis zur Entscheidung über ein effektives hydratisiertes Ion oder einen Flat-Bottom-Term nur in Tests verwenden.

**Bekannte Grenzen (bewusst akzeptiert für V1):**
- Keine Desolvatationsenergie → Faltungsgleichgewichte von Aptameren sind qualitativ. V2: Generalized Born (OBC2), siehe ADR 0004.
- Keine Bildladungen an Metalloberflächen → V2: `ImageChargeForce`.
- Wasserstruktur an der Oberfläche (Hydratationsschichten) fehlt.

## 4. Oberflächen- und Umgebungsplugins

### 4.1 `PlanarSurfaceForce` (Phase 2)
Die Wandform ist ein Template-Parameter (ADR 0007): `Wall93` (Volumenmaterial, z. B. Gold), `Wall104` (Graphen-Monolage), `Wall1043` (Graphit). Formeln und Kräfte aller Formen stehen in ADR 0007. Die Parameter pro (LJ-Typ, Oberflächen-Typ) leitet der Builder über Lorentz-Berthelot aus den Kraftfeld-LJ-Parametern und den Wandatom-Parametern ab, optional mit Lösungsmittel-Skalierung λ (OQ-05). Oberflächen-Typ 0 hat ε = 0 und ist für Padding-Atome und Thiol-Ankeratome reserviert (OQ-07).

9-3-Wand bei z = 0, Parameter pro Oberflächen-Typ (`d_types.y`):

```
V_i(z) = ε_w,i [ (2/15)(σ_w,i/z)⁹ − (σ_w,i/z)³ ]
F_z,i  = ε_w,i [ (6/5) σ_w,i⁹/z¹⁰ − 3 σ_w,i³/z⁴ ]
```
Minimum bei z* = (2/5)^(1/6) σ_w mit V(z*) ≈ −1.054 ε_w. Der Builder akzeptiert wahlweise die Topftiefe und rechnet um. Für z < z_clamp (Default 0.1 σ_w) wird z geclampt; das Ereignis wird in einem Device-Zähler registriert (Diagnose).

Optional elektrochemisches Potential (Gouy-Chapman, 1:1-Elektrolyt). Nach OQ-06: nichtlineare geschlossene Lösung als Default, linear nur als Grenzfall:
```
γ_r    = tanh(Φ₀ / (4 k_B T_r))          pro Replica (T_r aus d_temperature)
u(z)   = γ_r exp(−κ z)
V_i(z) = q_i · 2 k_B T_r · ln[(1 + u)/(1 − u)]  = q_i · 4 k_B T_r · atanh(u)   (numerisch: atanhf)
F_z,i  = q_i · 4 k_B T_r · κ · u / (1 − u²)
```
Linearisiert (Φ₀ ≪ k_BT/e ≈ 25 mV): `V_i(z) = q_i Φ₀ exp(−κ z)`, `F_z,i = q_i Φ₀ κ exp(−κ z)`. Die Linearisierung überschätzt das Fernfeld um ca. 30 % bei 100 mV und etwa Faktor 2 bei 200 mV.

Φ₀ in kJ/(mol·e) (Builder rechnet aus Volt um). **Φ₀ ist das Potential der diffusen Schicht**, nicht das angelegte Elektrodenpotential; der Abfall über die Stern-Schicht ist nicht modelliert.

**Thiol-Anker und SAM (OQ-07):** Das Ankeratom erhält Oberflächen-Typ 0 (keine Wandwechselwirkung) und einen z-Restraint auf Chemisorptionshöhe (§4.2). Eine Mercaptohexanol-Rückfüllschicht wird optional als zusätzliche weiche, rein repulsive Wand (WCA-artig) bei z ≈ 0,8–1,0 nm modelliert, die nur auf Atome oberhalb des Linkers wirkt.

### 4.2 `HarmonicRestraintPullingForce` (Phase 2)
- **Positions-Restraint:** `V = ½ k |r_i − r_i⁰|²` (optional nur ausgewählte Dimensionen, mit Minimum Image in x/y).
- **Schwerpunkt-Pulling:** Gruppe G mit Schwerpunkt R_G; `V = ½ k (ê·(R_G − R₀) − v t)²` mit Zugrichtung ê und Geschwindigkeit v. Kraft massengewichtet auf die Atome von G verteilt.
- Die geleistete Arbeit `W(t) = −∫ k (ê·(R_G − R₀) − v t) v dt` wird pro Replica auf dem Device akkumuliert (Jarzynski/Crooks).
- Umbrella-Modus = Pulling mit v = 0 und verschiedenen R₀ pro Replica.

### 4.3 `VoxelGridForce3D` (Phase 6)
3D-Textur (`cudaTextureObject_t`, float4) mit (V, ∂V/∂x, ∂V/∂y, ∂V/∂z) pro Voxel, trilinear interpoliert, skaliert pro Atomtyp (`d_types.y`). Anwendungen: Potential gekrümmter Nanokavitäten, optische Gradientenkraft `F = ¼ Re(α) ∇|E₀|²` (α = 4πε₀ α′, E₀ komplexe Feldamplitude) aus FEM-Feldern. Hinweis: interpolierter Gradient ≠ Gradient des interpolierten Potentials → leichte Energienicht-Erhaltung; Größe in Tests quantifizieren.

**Normierung optischer Kräfte (OQ-04):** Im Optik-Modus speichert das Gitter W = |E₀|² in (V/nm)² mit E₀ als komplexer **Spitzen**amplitude (`E(t) = Re[E₀ e^{−iωt}]`), samt Gradient. Der Typskalierungsfaktor ist
```
s_i = −α′_i · FARADAY² / (4 · ONE_4PI_EPS0)      ⇒  U_i = s_i W,  F_i = −s_i ∇W
```
Bei RMS-Feldern ist der Vorfaktor ½ statt ¼; die Lade-Funktion verlangt daher das Pflichtargument `amplitude="peak" | "rms"`.

**Überschusspolarisierbarkeit (OQ-04):** In s_i steht α′_eff,i = α′_i − (3/4π) v_i (n_w² − 1)/(n_w² + 2), d. h. der Kontrast gegenüber dem verdrängten Wasser, mit denselben v_i wie im `RefractiveIndexProfiler`. Ohne Korrektur wäre die Kraft um rund Faktor 2 zu groß. Streu-/Absorptionskräfte (Im α) sind nicht konservativ und nicht abgebildet; Fluorophore sind im Optikmodus per Default ausgenommen. Größenordnung: 1 MW/cm² in Wasser, α′ ≈ 10⁻³ nm³ → U ≈ 10⁻⁷ kJ/mol pro Atom; relevant erst in plasmonischen Hotspots (Feldverstärkung ≳ 10³).

## 5. Constraints & HMR (Phase 5)

- **HMR:** Wasserstoffmasse auf 3.024 amu, Differenz vom gebundenen Schweratom abgezogen (Gesamtmasse erhalten).
- **H-Bindungs-Constraints:** SHAKE/RATTLE, parallelisiert über Cluster (Schweratom + seine H-Atome), Toleranz 1e-6 relativ.
- Ziel: Δt = 4 fs stabil bei 300 K mit implizitem Lösungsmittel.

## 6. In-situ Optik-Observer (Phase 6)

### 6.1 `DipoleAndDistanceTracker`
Pro Frame und Replica: Schwerpunkt R_F der Fluorophorgruppe, Übergangsdipol-Einheitsvektor `μ̂ = (r_B − r_A)/|r_B − r_A|`, Abstand `d = z(R_F) − z_surface`, `cos θ = μ̂·ẑ`. Ausgabe: Zeitreihe (n_replicas, n_frames, 7) = (R_F, μ̂, d).

### 6.2 `FRETTracker`
Donor- und Akzeptor-Dipole wie oben, `R_DA` mit Minimum Image in x/y,
```
κ² = (μ̂_D·μ̂_A − 3(μ̂_D·R̂)(μ̂_A·R̂))²
```
Ausgabe: **Rohzeitreihen** R_DA(t), κ²(t). Die Effizienz `E = 1/(1 + (R/R₀)⁶ · (2/3)/κ²)` mit R₀ für κ² = 2/3 wird nur optional ausgegeben. Mittelung (dynamisch vs. statisch) erfolgt in `biooptomd.analysis`, da sie vom Verhältnis Rotations-/Fluoreszenzzeit abhängt.

### 6.3 `RefractiveIndexProfiler`
Schichten Δz über der Oberfläche, Fläche A = L_x L_y. Pro Schicht k (Volumen V_k = A Δz), Histogramme per `atomicAdd` (Fixed-Point):
- `S_α(k) = Σ_i α′_i` (Polarisierbarkeitsvolumen bei Wellenlänge λ, Typtabelle)
- `S_V(k) = Σ_i v_i` (van-der-Waals-Volumen des Atoms)

Effektives Medium (Lorentz-Lorenz mit Lösungsmittel-Hintergrund):
```
f_w(k)          = max(0, 1 − S_V(k)/V_k)                    (Wasser-Volumenanteil)
L(k)            = (4π/3) S_α(k)/V_k + f_w(k) · (n_w² − 1)/(n_w² + 2)
n(k)            = sqrt((1 + 2L)/(1 − L))
```
Ausgabe: n(z), sowie die Rohhistogramme S_α, S_V und ihre Varianz, damit andere Mischungsregeln (z. B. Maxwell-Garnett) nachträglich in Python möglich sind.

**Verschmierung (OQ-02):** Default ist der exakte Kugelschnitt: Atom i mit R_i = (3 v_i/4π)^{1/3} trägt zu Schicht [z_a, z_b] den Volumenanteil π∫(R_i² − (z − z_i)²) dz bei (analytisch, kompakter Träger, keine Beiträge unter z = 0 durch Ausläufer). Option: Gauß-Verschmierung mit σ_i = R_i/√5 (gleiche z-Varianz wie eine homogene Kugel). Bei Δz = 0,05 nm sind Schichten deutlich dünner als ein Atom; mit Punktatomen greift das `max(0, …)` in f_w häufig und verzerrt n(z) systematisch. Punktatome bleiben als Option für Tests.

**Parameter (OQ-01/02):** v_i aus Voronoi-Volumina (ProtOr für Proteine, NucProt für Nukleinsäuren; Summe ≈ partielles spezifisches Volumen nur bei Proteinen), α′_i aus Typtabelle mit klassenweisem Skalierungsfaktor, gemeinsam kalibriert an dn/dc (Proteine ≈ 0,185–0,19 mL/g, DNA ≈ 0,17 mL/g), Dispersion über Cauchy-Term. Fluorophore erhalten eine separate, resonanzbewusste Polarisierbarkeit oder werden per Option ausgeschlossen.
