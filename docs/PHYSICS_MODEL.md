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
- **Bekannte Eigenschaft (gemessen in Phase 5):** BAOAB sampelt Konfigurationen sehr genau, die kinetische Temperatur aus den Geschwindigkeiten am vollen Schritt ist aber um O(Δt²) zu niedrig (harmonischer Oszillator: ⟨p²⟩/m = kT·(1 − ω²Δt²/4)). Duplex-System, HMR + H-Constraints, 300 K: T_kin = 268 / 292 / 298 / 299.5 K bei Δt = 4 / 2 / 1 / 0.5 fs, ⟨E_pot⟩ dagegen unabhängig von Δt. Konfigurationsgrößen (E_pot, Geometrie, n(z)) sind daher die verlässlichen Observablen; die „Temperatur“ aus `energies()` ist bei 4 fs kein Maß für die Thermostat-Temperatur (OQ-14).
- **Zweite Temperatur (Phase 9.7, OQ-14):** E_kin der Geschwindigkeiten nach dem O-Schritt
  (+ RATTLE), `ctx.kinetic_energy_mid()` bzw. `kinetic_mid` im EnergyTracker, T = 2 E / (N_dof k_B).
  Für den harmonischen Oszillator ist sie exakt. Im 2k-System (HMR, H-Constraints, γ = 1/ps) liegt sie
  bei 300.4 ± 0.3 / 300.2 / 300.2 K für Δt = 4 / 2 / 1 fs; E_kin am vollen Schritt ergibt dort
  268.5 / 292.3 / 298.2 K (Bias ∝ Δt²). Für die Energiebilanz (NVE) bleibt E_kin am vollen Schritt
  maßgeblich.

## 2. Kovalente Terme (Phase 3)

| Term | Energie | Bemerkung |
|---|---|---|
| Bond | ½ k_b (r − r₀)² | |
| Angle | ½ k_θ (θ − θ₀)² | θ über `atan2(|a×b|, a·b)` |
| Angle, kosinus-harmonisch (`angle_form = 1`, ADR 0017) | ½ k (cos θ − cos θ₀)² | k in kJ/mol; Kraft endlich und → 0 bei θ = 0, π (MARTINI, CG) |
| Periodic torsion | k_φ [1 + cos(nφ − φ₀)] | Impropers in AMBER ebenfalls periodisch |
| 1-4 Paare | LJ·s_LJ + Coulomb·s_el | explizite Paarliste; AMBER: s_el = 1/1.2, s_LJ = 1/2; Elektrostatik der 1-4-Paare mit demselben Debye-Hückel-Kernel wie Non-Bonded |
| Paar 12-10 (`pair_form = 1`, ADR 0017) | ε[5(σ/r)¹² − 6(σ/r)¹⁰] + Debye-Hückel | native Gō-Kontakte (Karanicolas-Brooks), Minimum −ε bei r = σ |

## 3. Nicht-kovalente Terme (Phase 4)

**Lennard-Jones:** `V = 4ε_ij[(σ_ij/r)¹² − (σ_ij/r)⁶]`, Kombination nach Lorentz-Berthelot, vorab als Typ-Paar-Tabelle (`N_types × N_types`, float2) im Shared/Constant Memory.

**Elektrostatik V1 – Debye-Hückel (implizites Wasser mit Salz):**
`V = ONE_4PI_EPS0 · q_i q_j · exp(−κ r) / (ε_r · r)`, ε_r = 78.5 (Default), κ aus Ionenstärke. κ enthält alle nicht explizit modellierten Ionen. Temperaturabhängigkeit von κ und ε_r pro Replica: OQ-10.

**Cutoff:** Beide Terme mit Switching-Funktion zwischen r_s und r_c. Der Cutoff wird aus der Debye-Länge abgeleitet (ADR 0006): `r_c = max(2.0 nm, 5/κ)`, `r_s = r_c − 0.2 nm`. Damit ist DH bei r_c auf exp(−5) ≈ 0,7 % der reinen Coulomb-Wechselwirkung abgefallen. (Der frühere feste Wert r_c = 2,0 nm ließ bei 150 mM noch ≈ 8 % übrig.) Da der Tiled-All-Pairs-Kernel alle Paare rechnet, kostet ein größerer Cutoff keine Rechenzeit. Die Box L_x, L_y ist eine physikalische Eingabe (Bedeckung Γ = 1/(L_x L_y), OQ-09) und wird nie angepasst. Für L ≥ 2 r_c gilt Minimum Image; sonst summiert der Kernel explizit über periodische Bilder in x/y, inklusive Eigenbilder mit Faktor ½; Exclusions gelten nur für das Nullbild (ADR 0006 Rev. 2). Implementierung: Das Nullbild ist das Minimum-Image-Bild (der gebundene Nachbar in seiner nächsten Kopie). Pro Richtung werden |n| ≤ ⌈r_c/L + ½⌉ Verschiebungen geprüft, für L ≥ 2 r_c in dieser Richtung keine. Die Bildsumme läuft nur, wenn eine Box einer Replica kleiner als 2 r_c ist; sonst bleibt es beim Minimum-Image-Kernel. Builder: `biooptomd.nonbonded.cutoff_from_kappa`, κ_r aus `debye_kappa(I, T_r, water_permittivity(T_r))` (OQ-10).
Switching wie OpenMM (`useSwitchingFunction`): `V = (V_LJ + V_el)·S(r)` mit `S = 1 − 10x³ + 15x⁴ − 6x⁵ = (1−x)³(1 + 3x + 6x²)`, `x = (r − r_s)/(r_c − r_s)` für r_s < r < r_c (S = 1 darunter, 0 darüber); die Kraft enthält den Term −V·dS/dr. Im Kernel faktorisiert ausgewertet (keine Auslöschung für x → 1).

**Elektrostatik V2 – Generalized Born OBC2 mit Salz (Phase 9.1, ADR 0010):** Option
`solvent="gb-obc2"` im Builder (Nonbonded-Strategie `TiledGeneralizedBorn`). Coulomb mit ε_in = 1
(auch 1-4), Abschirmung und Desolvatation im GB-Term (AMBER igb = 5, mbondi2-Radien aus
`implicit/obc2.xml`, Salzkorrektur nach Srinivasan, ACE optional/Default an); Formeln in ADR 0010 und
`tests/reference/gb_ref.py`. Für r ≫ B geht E_C + E_GB in die Debye-Hückel-Form über, daher gelten
Cutoff aus κ, Switching des kombinierten Paars und Bildsumme unverändert. Ausgeschlossene Paare tragen
nur E_GB bei (ohne Switching), Born-Integrale werden bei r_c hart abgeschnitten. Kosten ≈ 3× DH
(drei Tile-Passes). GB-neck2 (Phase 9.3, ADR 0012, `solvent="gb-gbn2"`): mbondi3, α/β/γ je Atom, Neck-Term aus den OpenMM-Tabellen (Stufe bei ρ_i + ρ_j + 0.68 nm wie AMBER igb = 8). Grenzen: OBC2 statt GB-neck2 für Nukleinsäuren (OQ-15), keine dielektrische
Grenze an der Oberfläche (OQ-16).

**Exclusions:** 1-2 und 1-3 vollständig ausgeschlossen, 1-4 im Non-Bonded-Kernel ausgeschlossen (werden in der Bonded-Stufe skaliert berechnet). Kodierung als 32-bit-Bitmaske pro (Atom, Tile).

**Explizite Ionen:** Divalente Ionen (Mg²⁺) können als echte Teilchen mit LJ-Parametern vorhanden sein; die monovalente Salzumgebung steckt in κ. **Achtung (OQ-08):** Ohne Hydrathülle kollabiert Mg²⁺ im impliziten Wasser in direkten Phosphatkontakt und bindet viel zu stark. Bis zur Entscheidung über ein effektives hydratisiertes Ion oder einen Flat-Bottom-Term nur in Tests verwenden.

**Bekannte Grenzen (bewusst akzeptiert für V1):**
- Mit Debye-Hückel (V1) keine Desolvatationsenergie → Faltungsgleichgewichte von Aptameren sind qualitativ. V2: Generalized Born (OBC2), ADR 0004/0010, ab Phase 9.1 verfügbar.
- Bildladungen an Metalloberflächen: seit Phase 9.5 optional (`ImageChargeForce`, §4.4, ADR 0013).
- Wasserstruktur an der Oberfläche (Hydratationsschichten) fehlt.

## 4. Oberflächen- und Umgebungsplugins

### 4.1 `PlanarSurfaceForce` (Phase 2)
Die Wandform ist ein Template-Parameter (ADR 0007): `Wall93` (Volumenmaterial, z. B. Gold), `Wall104` (Graphen-Monolage), `Wall1043` (Graphit). Formeln und Kräfte aller Formen stehen in ADR 0007. Die Parameter pro (LJ-Typ, Oberflächen-Typ) leitet der Builder über Lorentz-Berthelot aus den Kraftfeld-LJ-Parametern und den Wandatom-Parametern ab, optional mit Lösungsmittel-Skalierung λ (OQ-05). Oberflächen-Typ 0 hat ε = 0 und ist für Padding-Atome und Thiol-Ankeratome reserviert (OQ-07); `PlanarSurfaceForce` lehnt Tabellen mit C ≠ 0 in Zeile 0 ab. λ gilt derzeit pro Oberflächen-Typ; Atomklassen werden über eigene Oberflächen-Typen abgebildet.

Implementierung: Tabelle (C, σ_iw) pro (Oberflächen-Typ s = `d_types.y`, LJ-Typ l = `d_types.x`), erstellt von `biooptomd.surface.wall_table` (Materialtabelle `MATERIALS`, Lorentz-Berthelot, λ pro Oberflächen-Typ). Eine Tabelle mit nur einem LJ-Typ gilt für alle LJ-Typen. Der Kernel ist ein Template über die Form; die Form wird pro Launch auf dem Host gewählt. Mit s = σ_iw/ζ und dem Vorfaktor C aus ADR 0007:

```
9-3     V = C [ (2/15) s⁹ − s³ ]                        F_z = C [ (6/5) s⁹ − 3 s³ ] / ζ
10-4    V = C [ (2/5) s¹⁰ − s⁴ ]                        F_z = 4C [ s¹⁰ − s⁴ ] / ζ
10-4-3  V = C [ (2/5) s¹⁰ − s⁴ ] − B/(ζ + 0.61Δ)³       F_z = 4C [ s¹⁰ − s⁴ ] / ζ − 3B/(ζ + 0.61Δ)⁴,  B = C σ⁴/(3Δ)
```
Minima: 9-3 bei z* = (2/5)^(1/6) σ mit V(z*) = −(√10/3) C ≈ −1.054 C; 10-4 bei z* = σ mit V = −(3/5) C; 10-4-3 numerisch (tiefer als 10-4). Für ζ < ζ_c wird ζ geclampt; das Ereignis wird in einem Device-Zähler registriert (Diagnose). ζ_c = max(0.1 σ_iw, ζ_F) mit |F(ζ_F)| = F_max (Default 10⁶ kJ/mol/nm), damit die Wandkraft im Fixed-Point-Bereich bleibt (ADR 0008). Atome mit Typen außerhalb der Tabelle spüren keine Wand (eigener Zähler).

Optional elektrochemisches Potential (Gouy-Chapman, 1:1-Elektrolyt). Nach OQ-06: nichtlineare geschlossene Lösung als Default, linear nur als Grenzfall:
```
γ_r    = tanh(Φ₀ / (4 k_B T_r))          pro Replica (T_r aus d_temperature)
u(z)   = γ_r exp(−κ z)
V_i(z) = q_i · 2 k_B T_r · ln[(1 + u)/(1 − u)]  = q_i · 4 k_B T_r · atanh(u)   (numerisch: s. u.)
F_z,i  = q_i · 4 k_B T_r · κ · u / (1 − u²)
```
Numerik: Gerechnet wird mit |Φ₀| (u > 0) und 1 − u = (1 − e^{−κζ}) + (1 − γ) e^{−κζ} ohne Auslöschung, damit V für γ → 1 (große Φ₀) endlich bleibt. κ_r kommt pro Replica (OQ-10). Das nichtlineare Gouy-Chapman verlangt T_r > 0 für alle Replicas (sonst Fehler in `compute`). Der Term hat keine Kraftschranke nach ADR 0008: Bei großem Φ₀ ist F(0) = 2k_BTκ·sinh(Φ₀/2k_BT) und kann den Fixed-Point-Bereich sprengen (dann Statusbit gesetzt). Für ζ < 0 gilt V = V(0), F = 0 (ASSUMPTION(OQ-12)).

Linearisiert (Φ₀ ≪ k_BT/e ≈ 25 mV, Option `gouy_chapman="linear"`): `V_i(z) = q_i Φ₀ exp(−κ z)`, `F_z,i = q_i Φ₀ κ exp(−κ z)`. Die Linearisierung überschätzt das Fernfeld um ca. 30 % bei 100 mV und etwa Faktor 2 bei 200 mV.

Φ₀ in kJ/(mol·e) (Builder rechnet aus Volt um). **Φ₀ ist das Potential der diffusen Schicht**, nicht das angelegte Elektrodenpotential; der Abfall über die Stern-Schicht ist nicht modelliert.

**Thiol-Anker und SAM (OQ-07):** Das Ankeratom erhält Oberflächen-Typ 0 (keine Wandwechselwirkung) und einen z-Restraint auf Chemisorptionshöhe (§4.2). Eine Mercaptohexanol-Rückfüllschicht wird optional als zusätzliche weiche, rein repulsive Wand (WCA-artig) bei z ≈ 0,8–1,0 nm modelliert, die nur auf Atome oberhalb des Linkers wirkt.

### 4.2 `HarmonicRestraintPullingForce` (Phase 2)
- **Positions-Restraint:** `V = ½ k |r_i − r_i⁰|²` (optional nur ausgewählte Dimensionen, mit Minimum Image in x/y).
- **Schwerpunkt-Pulling:** Gruppe G mit Schwerpunkt R_G; `V = ½ k (ê·(R_G − R₀) − v t)²` mit Zugrichtung ê und Geschwindigkeit v. Kraft massengewichtet auf die Atome von G verteilt.
- Die geleistete Arbeit `W(t) = −∫ k (ê·(R_G − R₀) − v t) v dt` wird pro Replica auf dem Device akkumuliert (Jarzynski/Crooks).
- Umbrella-Modus = Pulling mit v = 0 und verschiedenen R₀ pro Replica.

### 4.3 `VoxelGridForce3D` (Phase 6)
3D-Textur (`cudaTextureObject_t`, float4) mit (V, ∂V/∂x, ∂V/∂y, ∂V/∂z) pro Voxel, trilinear interpoliert, skaliert pro Atomtyp (`d_types.y`). Anwendungen: Potential gekrümmter Nanokavitäten, optische Gradientenkraft `F = ¼ Re(α) ∇|E₀|²` (α = 4πε₀ α′, E₀ komplexe Feldamplitude) aus FEM-Feldern. Hinweis: interpolierter Gradient ≠ Gradient des interpolierten Potentials → leichte Energienicht-Erhaltung. Gemessen (`test_voxel_energy_force_consistency`, glattes Feld mit Wellenzahl 2/nm): RMS-Abweichung 5.3 % bei Δ = 0.1 nm, 2.5 % bei Δ = 0.05 nm – erste Ordnung in Δ (die Ableitung des trilinearen Interpolanten ist zellweise konstant). Für NVE-Läufe mit Voxelfeldern daher feine Gitter wählen. Die Kopplungsnormierung für optische Kräfte ist OQ-04.

**Normierung optischer Kräfte (OQ-04):** Im Optik-Modus speichert das Gitter W = |E₀|² in (V/nm)² mit E₀ als komplexer **Spitzen**amplitude (`E(t) = Re[E₀ e^{−iωt}]`), samt Gradient. Der Typskalierungsfaktor ist
```
s_i = −α′_i · FARADAY² / (4 · ONE_4PI_EPS0)      ⇒  U_i = s_i W,  F_i = −s_i ∇W
```
Bei RMS-Feldern ist der Vorfaktor ½ statt ¼; die Lade-Funktion verlangt daher das Pflichtargument `amplitude="peak" | "rms"`. Implementierung: `biooptomd.voxel.optical_scale(alpha, volume, n_water, amplitude=..., local_field=True)` liefert s_i je Optik-Typ für `add_voxel_grid`; das Gitter selbst baut `values_from_potential(W, spacing)`.

**Überschusspolarisierbarkeit (OQ-04):** In s_i steht α′_eff,i = α′_i − (3/4π) v_i (n_w² − 1)/(n_w² + 2), d. h. der Kontrast gegenüber dem verdrängten Wasser, mit denselben v_i wie im `RefractiveIndexProfiler`. Ohne Korrektur wäre die Kraft um rund Faktor 2 zu groß. Streu-/Absorptionskräfte (Im α) sind nicht konservativ und nicht abgebildet; Fluorophore sind im Optikmodus per Default ausgenommen.

**Lokalfeldfaktor (Phase 10.0, `test_voxel_optical_sphere_consistency`, OQ-22):** Für eine dichte Kugel mit Lorentz-Lorenz-Mischung ergibt das additive Modell α′ = a³ (L_p − L_w), Clausius-Mossotti im Medium dagegen a³ ε_w (ε_p − ε_w)/(ε_p + 2 ε_w). Das Verhältnis CM/additiv ist ε_w (ε_w + 2)(ε_p + 2)/(3(ε_p + 2 ε_w)). In Vakuum ist es exakt 1; für kleinen Kontrast geht es gegen den Lorentz-Faktor ((n_w² + 2)/3)² = 1.585 (n_w = 1.333). Bei n_p = 1.5 beträgt es 1.638, bei n_p = 1.6 1.668. Im Wasser unterschätzt das additive Modell die optische Kraft also um rund 40 % gegenüber dem Kontinuumsbild, das auch der FEM-Solver verwendet. **Entscheidung (OQ-22, Projektinhaber 2026-11-06):** `optical_scale` multipliziert per Default mit dem Lorentz-Faktor ((n_w² + 2)/3)²; damit liegt die Kraft bis n_p = 1.6 etwa 5 % unter Clausius-Mossotti (n_p = 1.5: 3 %). Der Faktor ist zugleich konsistent mit dem kalibrierten Profiler: Δε = (n_w² + 2)²/3 · ΔL ergibt Δε V/(4π) = λ_L Σ α′_eff (`test_optical_force_dilute_solution`); ein CM-Faktor würde die Depolarisation doppelt zählen, da das FEM-Feld ⟨ε⟩ bereits enthält. `local_field=False` ergibt das rein additive Modell. Größenordnung: 1 MW/cm² in Wasser, α′ ≈ 10⁻³ nm³ → U ≈ 10⁻⁷ kJ/mol pro Atom; relevant erst in plasmonischen Hotspots (Feldverstärkung ≳ 10³).

### 4.4 `ImageChargeForce` (Phase 9.5, ADR 0013)
Abgeschirmte Bildladungen an einer leitfähigen Ebene z = z_im (Dirichlet-Rand des linearisierten
Poisson-Boltzmann-Problems; im Debye-Hückel-Modell für f = 1 exakt):
```
U = −f [ Σ_{i<j} q_i q_j φ(s_ij) + ½ Σ_i q_i² φ(2ζ_i) ],   φ(s) = k_e e^{−κs}/(ε_r s) · S(s)
s_ij = |(Δx_ij, Δy_ij, ζ_i + ζ_j)|,   ζ_i = smax(z_i − z_im)  (C¹-glattes max(·, ζ_c))
smax(u) = u (u ≥ ζ_c + h),  ζ_c + (u − ζ_c + h)²/(4h) (|u − ζ_c| < h),  ζ_c (u ≤ ζ_c − h),  h = ζ_c/2
```
Alle Paare ohne Exclusions, Eigenbild mit ½; Δx, Δy mit Minimum Image bzw. Bildsumme (L < 2 r_c,
auch für das Eigenbild, dessen Gitterbilder hier eine z-Kraft liefern); r_s, r_c, S(r), κ_r, ε_r wie
die Nonbonded-Strategie (bei GB ε_out, Näherung nach OQ-16). Die z-Kräfte eines Paars zeigen in dieselbe
Richtung (externe Wand, kein Impulserhalt in z). Unter ζ_c − h wird geclampt (ζ = ζ_c, z-Kraft 0),
zwischen ζ_c ± h geht ζ C¹-glatt in z − z_im über (Energie und Kraft stetig, OQ-18 d, Phase 9.7);
ζ_c = 0.1 nm, h = 0.05 nm; der Zähler `clamp_count` zählt Atome unter ζ_c + h. H-Atome an Gold
erreichen das regelmäßig. Bildfaktor f ∈ [−1, 1]:
1 = idealer Leiter (Gold), f < 1 Näherung für dielektrische Wände, für Graphen ohne Default (OQ-18).
Bildebene z_im = z0 + z_top + d_im, ASSUMPTION(OQ-18): d_im = 0 und ζ_c = 0.1 nm (an Gold ein
Modellparameter, weil H-Atome ζ < ζ_c erreichen). Gouy-Chapman hat seinen Ursprung bei z0, nicht bei
z_im. Energie-Term `image`.
Builder: `biooptomd.builder.add_image_charges(ctx, info, "gold")`. NumPy-Referenz:
`tests/reference/image_charge_ref.py`. Größenordnung (ε_r = 78.5, 0.15 M): −0.7 kJ/mol für eine
Elementarladung 0.3 nm über der Ebene; kollektiv für Polyelektrolyte (12 Phosphate in 0.6 nm Höhe
≈ −5.6 kJ/mol ohne Cutoff, davon ≈ 66 % Paarterme). Kosten gemessen: +4 % (DH) bzw. +7 % (GBn2) je
Schritt für einen aufrecht stehenden Duplex (758 Atome, 8 Replicas; GPU dabei geteilt, Verhältnis
indikativ).

### Replica Exchange (Phase 9.2, ADR 0011)
Temperatur-Austausch auf dem Host (`biooptomd.remd.ReplicaExchange`): Zustände (T_k, κ_k, ε_k) wandern
zwischen den Slots, Akzeptanz nach dem Hamiltonian-Kriterium mit Kreuzenergien, weil κ, ε_r und γ (Gouy-
Chapman) von T abhängen (OQ-10); Geschwindigkeiten werden mit sqrt(T_neu/T_alt) skaliert.

## 5. Constraints & HMR (Phase 5)

- **HMR:** Wasserstoffmasse auf 3.024 amu, Differenz vom gebundenen Schweratom abgezogen (Gesamtmasse erhalten).
- **H-Bindungs-Constraints:** SHAKE/RATTLE, parallelisiert über Cluster (Schweratom + seine H-Atome, ≤ 3 H), Toleranz 1e-6 relativ. Ein Thread pro Cluster, FP64 auf dem Mixed-Precision-Zustand (ADR 0009). Löser je Cluster (Phase 8): SHAKE als Newton-Iteration auf dem gekoppelten m×m-System (M-SHAKE, m ≤ 3 Constraints, quadratische Konvergenz), RATTLE als direkte Lösung des linearen Systems A λ = b mit A_kl = w_0 r_k·r_l + δ_kl w_k r_k² (ein Schritt, danach Residuumsprüfung); bei singulärem System (unbewegliche Atome) der entkoppelte Gauss-Seidel-Schritt. Ablauf: B, RATTLE, A+SHAKE (Geschwindigkeitskorrektur v += Δx/(Δt/2)), O, RATTLE, A+SHAKE, Kräfte, B, RATTLE. Constrainte Bindungen werden aus dem Bindungsterm entfernt (wie OpenMM `constraints=HBonds`). Nicht-Konvergenz setzt `STATUS_CONSTRAINT_FAILURE`.
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

### 6.3 `RefractiveIndexProfiler` (3D: `PolarizabilityGridObserver`)
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

**Verschmierung (OQ-02):** Default ist der exakte Kugelschnitt: Atom i mit R_i = (3 v_i/4π)^{1/3} trägt zu Schicht [z_a, z_b] den Volumenanteil π∫(R_i² − (z − z_i)²) dz bei (analytisch, kompakter Träger, keine Beiträge unter z = 0 durch Ausläufer). Option: Gauß-Verschmierung mit σ_i = R_i/√5 (gleiche z-Varianz wie eine homogene Kugel). Bei Δz = 0,05 nm sind Schichten deutlich dünner als ein Atom; mit Punktatomen greift das `max(0, …)` in f_w häufig und verzerrt n(z) systematisch. Punktatome bleiben als Option für Tests. API: `add_refractive_index_profiler(..., smearing="sphere" | "gauss" | "point")`; Atomtypen mit v_i = 0 werden immer als Punkt gezählt.

**Parameter (OQ-01/02):** v_i aus Voronoi-Volumina (ProtOr für Proteine, NucProt für Nukleinsäuren; Summe ≈ partielles spezifisches Volumen nur bei Proteinen), α′_i aus Typtabelle mit klassenweisem Skalierungsfaktor, gemeinsam kalibriert an dn/dc (Proteine ≈ 0,185–0,19 mL/g, DNA ≈ 0,17 mL/g), Dispersion über Cauchy-Term. Fluorophore erhalten eine separate, resonanzbewusste Polarisierbarkeit oder werden per Option ausgeschlossen.

**3D-Karte (Phase 10.1, ADR 0016):** `PolarizabilityGridObserver` wendet dieselbe Mischung je Voxel (V_k = Δx Δy Δz, Zellen ab der unteren Ecke) an und liefert ε = n². Verschmierung separabel w = w_x w_y w_z mit dem Gauß-Gewicht des 1D-Profilers je Achse (σ = R/√5, Fenster ±4σ, normiert auf das Fenster) oder Punktatome; die Summe über x/y ergibt exakt das Gauß-Profil des 1D-Profilers, wenn das Gitter periodisch in x/y ist oder alle Atome samt ±4σ-Fenster abdeckt. Voxel ≥ 0.2 nm empfohlen (Bias durch max(0, ·) in f_w).

**Stand Phase 10.0:** Klassenweise Kalibrierung (`builder.calibrate_optics`, Projektinhaber 2026-11-06): Volumina kovalenter Elemente mit s_v auf v̄ skaliert (Protein 0.73 mL/g, DNA 0.58 mL/g nach NucProt, Zahlenwert nicht verifiziert, OQ-23), α′ mit s_α an dn/dc (Protein 0.190 ± 0.003 mL/g bei 589 nm, DNA 0.170 ± 0.005 mL/g bei 633 nm bezogen auf das Na-Salz, OQ-23); Ionen (Na⁺, Mg²⁺, Shannon-Radien) bleiben unkalibriert. Ergebnis: s_α = 0.983 bzw. 1.002, s_v = 0.666 bzw. 0.654 (Bondi-Überlappung); unkalibriert lag dn/dc bei 0.064 bzw. 0.065. Dass s_α nahe 1 liegt, stützt die Miller-Werte nur für die gewählten v̄ (s_α hängt an v̄: DNA v̄ = 0.50 / 0.55 / 0.58 → s_α = 0.93 / 0.97 / 1.00). `test_refractive_index_dn_dc` und `test_optical_force_dilute_solution` grün. Grenzen: Elementtypen erfassen Aromatizität nicht; eine Tabelle je System (keine gemischten Protein-DNA-Systeme). ProtOr/NucProt je Atomtyp und Cauchy-Dispersion folgen, wenn eigene Messwerte vorliegen.
