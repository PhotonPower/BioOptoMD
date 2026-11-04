# Offene Fragen

Format: `OQ-NN` · Status (`open` / `assumption` / `resolved`) · Frage · aktuelle Annahme · Vorschlag · Entscheider.
Im Code referenziert als `// ASSUMPTION(OQ-NN): …`.

Vorschläge aus dem Review vom 2026-11-04 sind unten ausführlich begründet. Sie ändern den Status **nicht**: Erst der Projektinhaber setzt eine Frage auf `assumption` oder `resolved` (AGENTS.md §4).

| ID | Status | Frage | Aktuelle Annahme | Vorschlag (Review 2026-11-04) | Entscheider |
|---|---|---|---|---|---|
| OQ-01 | open | Quelle für atomare Polarisierbarkeiten α′(λ) für Nukleinsäuren/Proteine/Fluorophore? (Miller 1990, Thole-Modell, AMOEBA-Parameter, QM-Fit?) Statische vs. optische Frequenz? | Platzhalter: elementbasierte Tabelle (Miller), λ-unabhängig | Typtabelle (Miller) + globaler Skalierungsfaktor, kalibriert gemeinsam mit OQ-02 an dn/dc; Cauchy-Dispersion; Fluorophore separat (§OQ-01) | Projektinhaber |
| OQ-02 | open | Atomvolumina für Volumenausschluss im n(z)-Profil: Bondi-vdW-Volumen oder Voronoi-basierte Volumina aus Proteindichte? | Bondi-Radien, Überlappung ignoriert | Voronoi-Volumina (Tsai/Gerstein), Summe = partielles spezifisches Volumen; Gauß-Verschmierung als Default (§OQ-02) | Projektinhaber |
| OQ-03 | open | Kraftfeld für Aptamere: AMBER OL15 / bsc1 / OL21? Parameterquelle für Fluorophore (GAFF2 + RESP)? | OL15 + GAFF2 | DNA: OL21, sonst OL15 (je nach tleap-Verfügbarkeit); RNA: OL3; Farbstoffe: AMBER-DYES, GAFF2+RESP nur als Fallback (§OQ-03) | Projektinhaber |
| OQ-04 | open | Normierung der optischen Gradientenkraft im `VoxelGridForce3D` (Feldstärke in V/nm, Leistungsdichte, Mittelung über Periode)? | Eingabe bereits als Potential in kJ/mol pro α′-Einheit | Gitter speichert \|E₀\|² (Spitzenamplitude, V/nm); U_i = −α′_i \|E₀\|² FARADAY² / (4·ONE_4PI_EPS0) (§OQ-04) | Projektinhaber |
| OQ-05 | open | Wie wird die 9-3-Wand für Graphen vs. Gold parametrisiert (ε_w, σ_w pro Atomtyp)? Literatur (z. B. Steele-Potential) oder eigener Fit an All-Atom-Graphen? | Steele-artige Werte für Graphen, Platzhalter für Gold | Wandform wählbar (9-3 Gold, 10-4 Graphen, 10-4-3 Graphit), Parameter aus Mischungsregeln, Lösungsmittelkorrektur per Skalierung an Explizit-Wasser-PMFs; siehe ADR 0007 | Projektinhaber |
| OQ-06 | open | Ist linearisiertes Gouy-Chapman bei relevanten Elektrodenpotentialen (> 25 mV) ausreichend, oder nichtlineare Lösung als Tabelle? | linearisiert | Nichtlineare geschlossene Lösung (keine Tabelle); linear nur als Testgrenzfall (§OQ-06) | Projektinhaber |
| OQ-07 | open | Thiol-Gold-Anker: harter Positions-Restraint oder lateral beweglicher Anker (nur z-Restraint)? | nur z-Restraint + schwacher xy-Restraint | z-Restraint auf Chemisorptionshöhe, Ankeratom von Wand ausgenommen; xy nur bei mehreren Rezeptoren pro Box; optionale SAM/MCH-Wand (§OQ-07) | Projektinhaber |
| OQ-08 | open | Wie werden explizite Mg²⁺-Ionen im impliziten Wasser behandelt, ohne dass sie mangels Hydrathülle in direkten Phosphatkontakt kollabieren? | (neu) keine | Effektives „hydratisiertes Mg“ (vergrößertes σ) oder Flat-Bottom-Abstandsterm Mg–O(P); Kalibrierung an Explizit-Wasser-Daten (§OQ-08) | Projektinhaber |

> **Hinweis zu Literaturangaben:** Die unten genannten Arbeiten und Zahlenwerte wurden im Review aus dem Gedächtnis angegeben und **nicht** gegen eine Datenbank geprüft. Vor Übernahme in Code oder ADR-Status `Accepted` verifizieren (Checkliste am Ende).

---

## Ausführliche Vorschläge (Review 2026-11-04)

### OQ-01 · Polarisierbarkeiten α′(λ)

OQ-01 und OQ-02 hängen über die Lorentz-Lorenz-Mischung im `RefractiveIndexProfiler` (PHYSICS_MODEL §6.3) direkt zusammen und sollten **gemeinsam** kalibriert werden.

- **Biomoleküle (nicht absorbierend im Sichtbaren):** Additive Typtabelle (Miller 1990, ahp-Werte) ist brauchbar, weil die Dispersion zwischen statischer und optischer Polarisierbarkeit im Sichtbaren nur wenige Prozent beträgt. λ-Abhängigkeit über einen Cauchy-Term `α′(λ) = α′_∞ (1 + B/λ²)`.
- **Kalibrierung:** Ein globaler Skalierungsfaktor pro Molekülklasse (Protein, Nukleinsäure) wird so gewählt, dass das Modell inklusive Volumenausschluss (OQ-02) das gemessene Brechungsindexinkrement reproduziert: dn/dc ≈ 0,185–0,19 mL/g (Proteine), ≈ 0,17 mL/g (DNA), jeweils bei ~589 nm.
- **Fluorophore:** In der Nähe der Absorption ist Re α′ resonant und nicht additiv. Fluorophore erhalten im Profiler eine eigene Gesamtpolarisierbarkeit (QM, z. B. TD-DFT, oder Messwert), verteilt auf ihre Atome, oder werden per Option ausgeschlossen.
- **Abnahmetest (Phase 6, neu):** `test_refractive_index_dn_dc` – homogene Lösung bekannter Konzentration c ergibt n = n_w + c · dn/dc innerhalb der Kalibriertoleranz.

### OQ-02 · Atomvolumina

- Bondi-Radien **ohne** Überlappungskorrektur überschätzen das Molekülvolumen stark, da kovalent gebundene Atome sich erheblich überlappen. Damit wird f_w systematisch zu klein.
- **Vorschlag:** Voronoi-basierte Atom-/Gruppenvolumina aus Kristallstrukturen (Tsai, Taylor, Chothia & Gerstein 1999 bzw. Folgearbeiten). Kontrolle: Summe der Atomvolumina ≈ partielles spezifisches Volumen (≈ 0,73 mL/g Proteine; grob 0,5–0,55 mL/g Nukleinsäuren, salzabhängig).
- **Gauß-Verschmierung als Default:** Bei Δz = 0,05 nm sind die Schichten viel dünner als ein Atom. Mit Punktatomen greift das `max(0, …)` in f_w häufig und erzeugt einen systematischen Bias. Breite σ_i = Atomradius/√5 (gleiche Varianz wie eine homogene Kugel; Vorschlag, im Test prüfen).

### OQ-03 · Kraftfeld

- **DNA-Aptamere:** OL21 (Verfeinerung der OL15-Linie) bevorzugt, falls in der verwendeten AmberTools-Version über tleap verfügbar; sonst OL15. bsc1 ist gleichwertig. **RNA-Aptamere:** OL3.
- Begründung für Pragmatismus: Der Unterschied zwischen diesen Varianten ist klein gegenüber dem Modellfehler durch Debye-Hückel ohne Desolvatation (ADR 0004). Entscheidend ist, dass Builder (ParmEd) und OpenMM-Referenz **dasselbe** prmtop verwenden.
- **Farbstoffe:** Vorhandene AMBER-DYES-Parameter (Graen et al. 2014; Schepers & Gohlke 2020) für Cy3/Cy5, Alexa, ATTO verwenden. GAFF2 + RESP nur für Farbstoffe ohne Parametersatz.
- **Übergangsdipol:** Atompaar (A, B) pro Farbstoff einmalig aus QM bestimmen, im Builder als Tabelle hinterlegen (Phase 7).

### OQ-04 · Normierung der optischen Gradientenkraft

Konvention: Feld `E(t) = Re[E₀ e^{−iωt}]`, E₀ = komplexe **Spitzen**amplitude. Dann (zeitgemittelt, α = 4πε₀ α′):

```
U_i = −¼ Re(α_i) |E₀|²  =  −α′_i |E₀|² / (4 · ONE_4PI_EPS0)
```

mit |E₀| in kJ/(mol·e·nm) = |E₀|[V/nm] · FARADAY. Im `VoxelGridForce3D` wird daher **|E₀|²** in (V/nm)² samt Gradient gespeichert; der Typskalierungsfaktor ist `s_i = −α′_i · FARADAY² / (4 · ONE_4PI_EPS0)`. Bei RMS-Feldern ist der Vorfaktor ½ statt ¼ – der Lader erhält ein Pflichtargument `amplitude="peak" | "rms"`.

**Größenordnung:** 1 MW/cm² in Wasser → |E₀| ≈ 2,4·10⁻³ V/nm; mit α′ ≈ 10⁻³ nm³ pro Atom folgt U ≈ 10⁻⁷ kJ/mol (k_BT ≈ 2,5 kJ/mol). Erst bei plasmonischer Feldverstärkung ≳ 10³ (Intensität ≳ 10⁶) werden ~0,1 kJ/mol pro Atom erreicht. Optische Kräfte sind daher nur in Hotspots relevant; Priorität von `VoxelGridForce3D` liegt zunächst bei Kavitätenpotentialen.

### OQ-05 · Wandpotential Graphen vs. Gold

Siehe **ADR 0007** (Proposed). Kurzfassung:

- Die 9-3-Wand entsteht durch Integration über einen **Halbraum** – korrekt für Gold, **nicht** für eine Graphen-Monolage. Dort gilt das 10-4-Potential (eine Atomebene), für Graphit Steeles 10-4-3.
- ε_w, σ_w pro Atomtyp werden über Lorentz-Berthelot aus den Kraftfeld-LJ-Parametern und den Wandatom-Parametern abgeleitet (kein freier Fit).
- Vakuumabgeleitete Parameter überschätzen im impliziten Wasser die Adsorption (fehlende Konkurrenz mit Wasser). Korrektur: typweiser Skalierungsfaktor λ_i auf ε_w, kalibriert an Adsorptions-PMFs von Nukleobasen/Nukleosiden in **explizitem** Wasser (Literatur oder eigene OpenMM-Läufe).

### OQ-06 · Gouy-Chapman

Die nichtlineare Lösung für einen 1:1-Elektrolyten ist geschlossen; eine Tabelle ist unnötig:

```
γ    = tanh(Φ₀ / (4 k_B T))                    (Φ₀ in kJ/(mol·e), k_B T in kJ/mol)
u(z) = γ · exp(−κ z)
Φ(z) = 2 k_B T · ln[(1 + u)/(1 − u)]
V_i  = q_i Φ(z)
F_z,i = q_i · 4 k_B T · κ · u / (1 − u²)
```

Grenzfall Φ₀ → 0 ergibt exakt die lineare Form. Fehler der Linearisierung im Fernfeld (effektiv 4 k_BT γ statt Φ₀): ca. +30 % bei 100 mV, ca. Faktor 2 bei 200 mV. Kosten: ein `log`, ein `exp` pro Atom.

- γ hängt von T ab → pro Replica aus `d_temperature` berechnen (Replica Exchange!).
- **Φ₀ ist das Potential der diffusen Schicht**, nicht das angelegte Elektrodenpotential (Abfall über die Stern-Schicht). Muss im Builder-Docstring stehen.

### OQ-07 · Thiol-Gold-Anker

1. **Ankeratom von der Wand ausnehmen** (Oberflächen-Typ mit ε_w = 0). Sonst stößt die Wand das S-Atom genau dort ab, wo der Restraint es festhält.
2. **z-Restraint** auf Chemisorptionshöhe (≈ 0,2–0,25 nm über der obersten Au-Lage; genaue Lage hängt von der Definition z = 0 der Wand ab – im Builder dokumentieren).
3. **xy-Restraint** nur, wenn mehrere Rezeptoren pro Box eine definierte Belegungsdichte haben sollen. Bei einem Rezeptor pro Box ist er unnötig, da n(z) lateral gemittelt wird.
4. **SAM/MCH-Backfill:** Reale Sensoren tragen fast immer eine Mercaptohexanol-Schicht, die das Flachliegen des Aptamers verhindert und die Konformation stärker prägt als die Ankerdetails. Günstiges Modell: zusätzliche weiche, rein repulsive Wand (WCA-artig) bei z ≈ 0,8–1,0 nm, die nur auf Atome oberhalb des Linkers wirkt (eigener Oberflächen-Typ).

### OQ-08 · Explizites Mg²⁺ im impliziten Wasser (neu)

Ohne Hydrathülle kollabiert Mg²⁺ in direkten Kontakt mit Phosphat-Sauerstoffen und bindet um Größenordnungen zu stark; real dominiert die Bindung über die äußere Sphäre (Mg(H₂O)₆²⁺). Optionen:

- (a) Effektives hydratisiertes Ion: σ_Mg so vergrößert, dass der Kontaktabstand Mg–O(P) dem äußeren Sphärenabstand entspricht.
- (b) Flat-Bottom-Abstandsterm zwischen Mg und O1P/O2P (Phase-2-Plugin wiederverwendbar).

Beide Varianten brauchen eine Kalibrierung (z. B. Mg–Phosphat-PMF in explizitem Wasser mit OpenMM). Bis zur Entscheidung: Mg²⁺ nur in Tests, nicht in Produktionsläufen.

---

## Checkliste: zu verifizierende Literatur/Zahlen

- [ ] Miller 1990 (J. Am. Chem. Soc.), ahp-Atomhybrid-Polarisierbarkeiten
- [ ] dn/dc Proteine (~0,185–0,19 mL/g) und DNA (~0,17 mL/g), Wellenlänge und Puffer
- [ ] Tsai/Gerstein et al. 1999 (J. Mol. Biol.), Standard-Atomvolumina
- [ ] Partielles spezifisches Volumen Nukleinsäuren (salzabhängig)
- [ ] Verfügbarkeit OL21 in der verwendeten AmberTools-/tleap-Version
- [ ] AMBER-DYES: Graen et al. 2014 (JCTC), Schepers & Gohlke 2020 (J. Chem. Phys.)
- [ ] Steele 1973 (Surf. Sci.): σ_C ≈ 0,340 nm, ε_C/k_B ≈ 28 K, Δ = 0,335 nm, ρ_C(Graphit) = 114 nm⁻³
- [ ] Heinz et al. 2008 (J. Phys. Chem. C), INTERFACE-FF-LJ-Parameter für Au
- [ ] Au–S-Chemisorptionshöhe relativ zur obersten Au-Lage
