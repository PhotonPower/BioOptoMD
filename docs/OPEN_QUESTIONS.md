# Offene Fragen

Format: `OQ-NN` · Status (`open` / `assumption` / `resolved`) · Frage · frühere Annahme · Entscheidung · Entscheider.

Status-Bedeutung: `resolved` = Vorgehen entschieden und vollständig spezifiziert. `assumption` = Vorgehen entschieden, aber Parameterwerte müssen noch kalibriert oder verifiziert werden. Code dazu wird mit `// ASSUMPTION(OQ-NN): …` markiert.
Im Code referenziert als `// ASSUMPTION(OQ-NN): …`.

Die Vorschläge aus den Reviews vom 2026-11-04 (inklusive der Nachträge aus Review 2) wurden am 2026-11-04 vom Projektinhaber **angenommen**. Die Begründungen stehen unten. Was bei `assumption`-Fragen noch zu kalibrieren ist, steht in der Checkliste am Ende und in den Abnahmetests der ROADMAP.

| ID | Status | Frage | Frühere Annahme | Entscheidung (Review 2026-11-04, inkl. Nachträge) | Entscheider |
|---|---|---|---|---|---|
| OQ-01 | assumption | Quelle für atomare Polarisierbarkeiten α′(λ) für Nukleinsäuren/Proteine/Fluorophore? (Miller 1990, Thole-Modell, AMOEBA-Parameter, QM-Fit?) Statische vs. optische Frequenz? | Platzhalter: elementbasierte Tabelle (Miller), λ-unabhängig | Typtabelle (Miller) + globaler Skalierungsfaktor, kalibriert gemeinsam mit OQ-02 an dn/dc; Cauchy-Dispersion; Fluorophore separat (§OQ-01) **2026-11-06:** Noch keine eigenen dn/dc-Messwerte. Kalibriert wird an Literaturwerten (Proteine 0.190 ± 0.003 mL/g bei 589 nm, DNA ≈ 0.17 mL/g bei 633 nm), deren Unsicherheit als Toleranz dient (ADR 0016 Punkt 7). **Erster Schritt umgesetzt:** Klassenweise Kalibrierung (`builder.calibrate_optics`, Projektinhaber 2026-11-06): Volumina kovalenter Elemente mit s_v auf v̄ skaliert (Protein 0.73 mL/g, DNA 0.58 mL/g nach NucProt, Zahlenwert nicht verifiziert, OQ-23), α′ mit s_α an dn/dc (Protein 0.190 ± 0.003 mL/g bei 589 nm, DNA 0.170 ± 0.005 mL/g bei 633 nm bezogen auf das Na-Salz, OQ-23); Ionen (Na⁺, Mg²⁺, Shannon-Radien) bleiben unkalibriert. Ergebnis: s_α = 0.983 bzw. 1.002, s_v = 0.666 bzw. 0.654 (Bondi-Überlappung); unkalibriert lag dn/dc bei 0.064 bzw. 0.065. Dass s_α nahe 1 liegt, stützt die Miller-Werte nur für die gewählten v̄ (s_α hängt an v̄: DNA v̄ = 0.50 / 0.55 / 0.58 → s_α = 0.93 / 0.97 / 1.00). `test_refractive_index_dn_dc` und `test_optical_force_dilute_solution` grün. Grenzen: Elementtypen erfassen Aromatizität nicht; eine Tabelle je System (keine gemischten Protein-DNA-Systeme). ProtOr/NucProt je Atomtyp und Cauchy-Dispersion folgen, wenn eigene Messwerte vorliegen. | Projektinhaber (2026-11-04) |
| OQ-02 | assumption | Atomvolumina für Volumenausschluss im n(z)-Profil: Bondi-vdW-Volumen oder Voronoi-basierte Volumina aus Proteindichte? | Bondi-Radien, Überlappung ignoriert | Voronoi-Volumina (ProtOr/NucProt), Summe = v̄ nur bei Proteinen; Kugelschnitt-Verschmierung als Default (§OQ-02, Nachtrag) **2026-11-06:** Noch keine eigenen dn/dc-Messwerte. Kalibriert wird an Literaturwerten (Proteine 0.190 ± 0.003 mL/g bei 589 nm, DNA ≈ 0.17 mL/g bei 633 nm), deren Unsicherheit als Toleranz dient (ADR 0016 Punkt 7). **Erster Schritt umgesetzt:** Klassenweise Kalibrierung (`builder.calibrate_optics`, Projektinhaber 2026-11-06): Volumina kovalenter Elemente mit s_v auf v̄ skaliert (Protein 0.73 mL/g, DNA 0.58 mL/g nach NucProt, Zahlenwert nicht verifiziert, OQ-23), α′ mit s_α an dn/dc (Protein 0.190 ± 0.003 mL/g bei 589 nm, DNA 0.170 ± 0.005 mL/g bei 633 nm bezogen auf das Na-Salz, OQ-23); Ionen (Na⁺, Mg²⁺, Shannon-Radien) bleiben unkalibriert. Ergebnis: s_α = 0.983 bzw. 1.002, s_v = 0.666 bzw. 0.654 (Bondi-Überlappung); unkalibriert lag dn/dc bei 0.064 bzw. 0.065. Dass s_α nahe 1 liegt, stützt die Miller-Werte nur für die gewählten v̄ (s_α hängt an v̄: DNA v̄ = 0.50 / 0.55 / 0.58 → s_α = 0.93 / 0.97 / 1.00). `test_refractive_index_dn_dc` und `test_optical_force_dilute_solution` grün. Grenzen: Elementtypen erfassen Aromatizität nicht; eine Tabelle je System (keine gemischten Protein-DNA-Systeme). ProtOr/NucProt je Atomtyp und Cauchy-Dispersion folgen, wenn eigene Messwerte vorliegen. | Projektinhaber (2026-11-04) |
| OQ-03 | resolved | Kraftfeld für Aptamere: AMBER OL15 / bsc1 / OL21? Parameterquelle für Fluorophore (GAFF2 + RESP)? | OL15 + GAFF2 | DNA: OL21 (tleap ab AmberTools23), OL24 als Option; RNA: OL3; Farbstoffe: AMBER-DYES, Linker zur Nukleinsäure selbst parametrisieren (§OQ-03, Nachtrag) | Projektinhaber (2026-11-04) |
| OQ-04 | resolved | Normierung der optischen Gradientenkraft im `VoxelGridForce3D` (Feldstärke in V/nm, Leistungsdichte, Mittelung über Periode)? | Eingabe bereits als Potential in kJ/mol pro α′-Einheit | Gitter speichert \|E₀\|² (Spitzenamplitude, V/nm); U_i = −α′_eff,i \|E₀\|² FARADAY² / (4·ONE_4PI_EPS0) mit α′_eff = Überschuss gegenüber verdrängtem Wasser; Im α nicht abgebildet (§OQ-04, Nachtrag) | Projektinhaber (2026-11-04) |
| OQ-05 | resolved | Wie wird die 9-3-Wand für Graphen vs. Gold parametrisiert (ε_w, σ_w pro Atomtyp)? Literatur (z. B. Steele-Potential) oder eigener Fit an All-Atom-Graphen? | Steele-artige Werte für Graphen, Platzhalter für Gold | Wandform wählbar (9-3 Gold, 10-4 Graphen, 10-4-3 Graphit), Parameter aus Mischungsregeln, Lösungsmittelkorrektur per Skalierung an Explizit-Wasser-PMFs; siehe ADR 0007 | Projektinhaber (2026-11-04) |
| OQ-06 | resolved | Ist linearisiertes Gouy-Chapman bei relevanten Elektrodenpotentialen (> 25 mV) ausreichend, oder nichtlineare Lösung als Tabelle? | linearisiert | Nichtlineare geschlossene Lösung (keine Tabelle); linear nur als Testgrenzfall (§OQ-06) | Projektinhaber (2026-11-04) |
| OQ-07 | assumption | Thiol-Gold-Anker: harter Positions-Restraint oder lateral beweglicher Anker (nur z-Restraint)? | nur z-Restraint + schwacher xy-Restraint | z-Restraint auf Chemisorptionshöhe, Ankeratom von Wand ausgenommen; xy nur bei mehreren Rezeptoren pro Box; optionale SAM/MCH-Wand (§OQ-07) | Projektinhaber (2026-11-04) |
| OQ-08 | assumption | Wie werden explizite Mg²⁺-Ionen im impliziten Wasser behandelt, ohne dass sie mangels Hydrathülle in direkten Phosphatkontakt kollabieren? | (neu) keine | Effektives „hydratisiertes Mg“ (vergrößertes σ) oder Flat-Bottom-Abstandsterm Mg–O(P); Kalibrierung an Explizit-Wasser-Daten mit microMg/nanoMg (§OQ-08, Nachtrag) | Projektinhaber (2026-11-04) |
| OQ-09 | resolved | Boxgröße L legt bei einem Rezeptor pro Box die Bedeckung 1/L² fest. Wie wird das mit r_c aus der Debye-Länge (ADR 0006) vereinbart, wenn L < 2 r_c? | (neu) L ≥ 2 r_c, Bedeckung folgt daraus | L aus gewünschter Bedeckung; bei L < 2 r_c explizite Summation über periodische Bilder in x/y (ADR 0006 Rev. 2, ändert ADR 0003) (§OQ-09) | Projektinhaber (2026-11-04) |
| OQ-10 | resolved | Temperaturabhängigkeit von κ und ε_r im Replica Exchange? | (neu) κ, ε_r für alle Replicas gleich | κ_r und ε_r(T_r) pro Replica, r_c = Maximum über Replicas (§OQ-10) | Projektinhaber (2026-11-04) |
| OQ-11 | resolved | ROADMAP Phase 1 fordert für `test_wrap_xy` eine „exakte“ ungewrappte Rekonstruktion; in FP32 nicht für alle x möglich. | – | Durch ADR 0009 gelöst: Wrap in FP64 auf der Mixed-Precision-Position, Rekonstruktion x_w + n·L = x bitgenau. | Projektinhaber (2026-11-03) |
| OQ-12 | assumption | Gouy-Chapman-Term für Atome hinter der Wand (ζ < 0): wie fortsetzen? | – | V = q Φ0 konstant, F = 0 (energiekonsistent; die 9-3-Wand verhindert ζ < 0 praktisch). | – (Bestätigung offen) |
| OQ-13 | resolved | Widerspruch PHYSICS_MODEL §4.1 ↔ ADR 0002: Clamp bei 0.1 σ erzeugte Kräfte ≫ 2³¹, Fixed-Point-Summen konnten überlaufen. | – | Entschieden 2026-11-03 (Projektinhaber): Kraftschranke für ζ_c + Sättigung pro Beitrag + Geräte-Statusflags, siehe ADR 0008. | Projektinhaber (2026-11-03) |
| OQ-14 | resolved | Welcher Temperaturschätzer soll gemeldet werden? Die kinetische Temperatur am vollen BAOAB-Schritt ist bei Δt = 4 fs um ~10 % zu niedrig (PHYSICS_MODEL §1), obwohl die Konfigurationen korrekt bei 300 K gesampelt werden. Alternativen: Geschwindigkeiten nach dem O-Schritt (Halbschritt), konfigurationelle Temperatur, oder nur dokumentieren. | – | Unverändert: E_kin am vollen Schritt, Bias dokumentiert. **Projektinhaber 2026-11-06:** E_kin am vollen Schritt bleibt (Energie, NVE-Prüfung); zusätzlich wird eine zweite Temperatur gemeldet – Geschwindigkeiten nach dem O-Schritt oder konfigurationelle Temperatur, je nachdem, welche den Bias bei 4 fs nachweislich beseitigt (Messung Phase 9.7). **Umgesetzt (Phase 9.7):** E_kin nach dem O-Schritt (+ RATTLE), `ctx.kinetic_energy_mid()` und `kinetic_mid` im EnergyTracker; 2k-System bei 4 fs 300.4 ± 0.3 K statt 268.5 K (PHYSICS_MODEL §1). | Projektinhaber (2026-11-06) |
| OQ-15 | assumption | GB-Modell und Radiensatz für Nukleinsäuren: OBC2 (mbondi2, ADR 0004) oder GB-neck2 (GBn2, mbondi3, Nguyen et al. 2014), das für DNA/RNA-Strukturen als deutlich besser gilt? | – | Projektinhaber 2026-11-05: zuerst OBC2 mit mbondi2 und Salzkorrektur (ADR 0010); GBn2 mit mbondi3 und Nukleinsäure-Parametern als zweites Modell ergänzt (ADR 0012, Phase 9.3). **Entschieden 2026-11-05 (§OQ-15):** GBn2 als Default für Nukleinsäure-Systeme; Validierung mit OL21: (a) Duplex-Stabilität nach Auswertung auf der 100-ns-Skala bestanden (Claude, 2026-11-05; bestätigt vom Projektinhaber 2026-11-06), (b) Hairpin 1PQT läuft. | Projektinhaber (2026-11-05) |
| OQ-16 | resolved | GB nahe der Oberfläche: Die Wand (Graphen/Gold) ist im GB-Modell Lösungsmittel; die dielektrische Grenze (Desolvatation an der Oberfläche, Bildladungen) fehlt. Wie modellieren (Wand als Niedrig-ε-Halbraum in den Born-Integralen, ImageChargeForce)? Folgen: Desolvatation an der Wand fehlt, ACE überschätzt die zugängliche Fläche an der Wand, der Gouy-Chapman-Wandterm arbeitet weiter mit κ (konsistent, aber nicht aus GB abgeleitet). | – | Vorerst nicht modelliert, in ADR 0010 dokumentiert. **Projektinhaber 2026-11-06:** vorerst nicht modelliert (ADR 0010 dokumentiert die Folgen); erst bei auffälligen Adsorptionsergebnissen angehen, dann mit eigenem ADR (Wand als Niedrig-ε-Halbraum in den Born-Integralen). | Projektinhaber (2026-11-06) |
| OQ-17 | resolved | Replica Exchange mit Gouy-Chapman: Ist Φ0 der diffusen Schicht über die Temperaturleiter konstant (fest vorgegebenes Potential) oder folgt es bei fester Oberflächenladung der Grahame-Gleichung Φ0(T, κ)? | – | Φ0 fest über alle Zustände; κ folgt dem Zustand (ADR 0011). **Projektinhaber 2026-11-06:** Φ0 fest über alle Zustände (potentialkontrollierte Elektrode); κ folgt dem Zustand (ADR 0011). Modus „feste Oberflächenladung“ (Grahame) erst bei Bedarf. | Projektinhaber (2026-11-06) |
| OQ-18 | assumption | ImageChargeForce (ADR 0013): (a) Lage der Bildebene d_im relativ zur obersten Atomlage für Au(111) (Literaturwert nicht verifiziert; Websuche 2026-11-05 ohne Au-Wert, nur Al(111)/Jellium); (b) Bildfaktor f für Graphen/Graphit (unvollständiger Leiter, kein Default); (c) für f < 1 mit Salz ist das Bild nur eine Näherung – genügt das oder braucht es die exakte Green'sche Funktion? (d) ζ_c: An Gold mit d_im = 0 erreichen H-Atome ζ < ζ_c = 0.1 nm regelmäßig (Implementierung 9.5), ζ_c ist dort ein Modellparameter – Wert, ggf. C¹-glattes Clamping? | – | d_im = 0 (Bildebene in der obersten Atomlage); f = 1 für Gold, für Graphen vom Nutzer anzugeben; ζ_c = 0.1 nm. **Projektinhaber 2026-11-06:** (a) d_im = 0 bleibt als ASSUMPTION (kein verifizierter Au(111)-Wert); (b) kein Default für Graphen, f vom Nutzer anzugeben; (c) Bildnäherung genügt vorerst; (d) ζ_c = 0.1 nm bleibt, Clamping wird C¹-glatt (Phase 9.7). | Projektinhaber (2026-11-06) |
| OQ-19 | decided | Phase 9.6: Das bestätigte Messkriterium „1 × 20 k ≥ 1.25× schneller“ (ROADMAP) widerspricht der Messung nach ADR 0014 Rev. 1 (≈ 1.14–1.18× bei geteilter GPU; i-Tile-Akkumulation ohne Gewinn). Kriterium nach der Endmessung bei freier GPU auf den gemessenen Gewinn senken, oder Phase 9.6 erst mit einer weiteren Optimierung (z. B. Constraint-Löser, 27 % bei 1 × 20 k) abnehmen? | – | Projektinhaber 2026-11-06: Phase 9.6 erst später abnehmen (Kriterium 1.25× bleibt); nächster Hebel Constraint-Löser, Endmessung bei freier GPU. | Projektinhaber (2026-11-06) |
| OQ-20 | resolved | MD ↔ FEM (ADR 0016): Welcher FEM-Solver wird genutzt (COMSOL, JCMsuite, FEniCS, MEEP, …), und welches Austauschformat (ε-Gitter hinaus, |E₀|² herein: NumPy-npz, HDF5, VTK-ImageData, CSV)? Wer interpoliert vom FEM-Netz auf das reguläre Gitter (Solver-Export oder BioOptoMD)? | – | **Projektinhaber 2026-11-06:** hp-FEM (github.com/PhotonPower/hp-FEM). Kopplung im selben Python-Prozess (`HpFemSolver`): ⟨ε⟩ je Zelle am Schwerpunkt über `MaterialMap.set_cell`, |E|² an den Kraftgitterpunkten über `problem.sample`; npz-Datei-Adapter als Rückfall (getrennte Interpreter unter Windows: MSYS2 vs. MSVC). | Projektinhaber (2026-11-06) |
| OQ-21 | open | MD ↔ FEM (ADR 0016): Ist die Mean-Field-Näherung (Feld aus ⟨ε⟩ statt aus Momentankonfigurationen) für die Zielsysteme zulässig? | – | ASSUMPTION: zulässig bei verdünnten Schichten und schwacher Kopplung; Prüfung über die Streuung von W aus Einzelframes (ADR 0016 Punkt 1). | – (offen) |
| OQ-22 | resolved | Lokalfeldfaktor der optischen Kraft (OQ-04, Phase 10.0): Das additive Überschussmodell unterschätzt die Kraft im Wasser gegenüber Clausius-Mossotti (Kontinuumsbild des FEM) um den Faktor ε_w (ε_w + 2)(ε_p + 2)/(3(ε_p + 2 ε_w)) ≈ 1.59–1.67 (n_p = n_w … 1.6). Korrektur (a) keine, (b) konstanter Lorentz-Faktor ((n_w² + 2)/3)² in `optical_scale` (exakt für kleinen Kontrast, ≈ 5 % Abweichung bis n_p = 1.6), (c) lokal aus ⟨ε⟩ (Phase 10, abhängig vom Feld)? | additiv ohne Faktor | Vorschlag Claude: (b) als Default, Option `local_field=False` für das bisherige Verhalten. **Projektinhaber 2026-11-06: (b)**, umgesetzt in `optical_scale(..., local_field=True)`. | Projektinhaber (2026-11-06) |
| OQ-23 | resolved | dn/dc-Kalibrierung DNA (Phase 10.0): (a) Auf welche Masse bezieht sich der Handbuchwert 0.17 mL/g (Na-Salz oder freie Säure/Anion; Puffer)? Unterschied 7.5 %, größer als die Toleranz. (b) Stimmt v̄ = 0.58 mL/g als NucProt-Packungsvolumen (Voss & Gerstein 2005; Zahlenwert in der Recherche nicht gefunden)? (c) Toleranz ± 0.005 als Spanne der Handbuchwerte. | ASSUMPTION: Bezug Na-Salz (ein Na⁺ je Phosphat in der Masse, optisch implizit), v̄ = 0.58, ± 0.005 | Vorschlag Claude: beibehalten bis zur ersten eigenen Messung im eigenen Puffer; dann (a) und (c) durch die Messung ersetzen, (b) bei Einführung von NucProt an der Quelle prüfen. **Projektinhaber 2026-11-06:** Vorschlag angenommen (Annahmen beibehalten bis zur ersten eigenen Messung). | Projektinhaber (2026-11-06) |
| OQ-24 | decided | MD ↔ FEM, Selbstfeld der Schicht (Phase 10.3): Bei p-Polarisation senkt eine dichte Adsorbatschicht das Normalfeld im Inneren (E_z = D_z/ε), das Mean-Field-W hat dort eine Delle. Die Atome der Schicht spüren deren Gradienten, im Prototyp (α′ = 5e-3 nm³, 4 Atome/nm², Kretschmann nahe SPR) um ein Vielfaches stärker als den äußeren Gradienten (Schicht kollabiert bzw. spreizt sich). Kontinuumsphysikalisch entspricht das der Kelvin-/Helmholtz-Kraftdichte (Netto-Selbstkraft der Schicht null, innere Spannung nicht), mikroskopisch spürt ein Molekül sein eigenes Feld aber nicht. Wie behandeln: (a) so lassen (Kontinuumsbild), (b) Kraft aus dem Feld ohne die Schicht des Moleküls selbst (Selbstausschluss je Molekül, teuer), (c) W für die Kraft in z über die Schichtdicke glätten, (d) Gültigkeitsgrenze dokumentieren (Δε der Schicht klein) und im Treiber warnen? | – | Vorschlag Claude: (d) jetzt (Warnung, wenn das Feld innerhalb der Schicht um mehr als einige Prozent variiert), (b) als Option prüfen, sobald ein realistisches System (Aptamerschicht) läuft. **Projektinhaber 2026-11-06:** Vorschlag angenommen: (d) umgesetzt (`self_field` im Verlauf, Warnung über `self_field_warn` = 5 %), (b) Selbstausschluss als Option prüfen, sobald ein realistisches System läuft. | Projektinhaber (2026-11-06) |
| OQ-25 | resolved | Abnahmekriterium SPR-Winkel (`test_tmm_reference`, ADR 0016): Die Recherche liefert für Prisma BK7 / Au 50 nm / Wasser bei 633 nm keinen konsistenten Literaturwinkel (≈ 68° in zwei Quellen, passt nicht zu ε_Au ≈ −11.6 + 1.2i, das ≈ 72° ergibt). Ersatz: TMM gegen die exakte Airy-Formel (rel. 1e-12) und SPR-Minimum innerhalb 1° der Plasmonen-Dispersion (72.24° gegen 72.75°). | – | Vorschlag Claude: Ersatzkriterium annehmen; Literaturwinkel ergänzen, sobald ε_Au aus derselben Quelle wie ein gemessener Winkel vorliegt (hp-FEM-Bibliothek: Johnson & Christy). **Projektinhaber 2026-11-06:** Ersatzkriterium angenommen. | Projektinhaber (2026-11-06) |
| OQ-26 | resolved | hp-FEM-Build für den 3D-Adapter (ROADMAP 10.5): Das Windows-Wheel hp-FEM 0.4.0 rechnet einfädig (`set_num_threads` bleibt 1) mit SparseLU; MUMPS und cuDSS sind nicht enthalten. Eine Lösung mit lateralen ε-Zellen von 1 nm dauert 5.5 min (Grad 2), feinere sind praktisch nicht machbar. Soll hp-FEM (Projekt des Projektinhabers) mit OpenMP und MUMPS/cuDSS gebaut und in der venv installiert werden? **Nachtrag (Review 10.5):** Grenze ist vor allem der Speicher der SparseLU-Faktoren (`hpfem.estimate_memory`: 1 nm lateral, Grad 2 ≈ 23 GB; Grad 3 ≈ 179 GB; 0.5 nm ≈ 207 GB). cuDSS auf der RTX 3090 (24 GB) allein reicht dafür nicht; entscheidend ist eine Sortierung mit wenig Auffüllung (MUMPS/METIS, cuDSS-Reordering) und weniger PML-Zeilen. | – | Vorschlag Claude: ja, cuDSS-Build (RTX 3090) als zusätzliches Wheel; bis dahin laterale ε-Zellen ≥ 2 nm im 3D-Adapter. **Projektinhaber 2026-11-06:** eigenen Build erstellen (Claude). **Umgesetzt 2026-11-06:** eigener Build hp-FEM 0.4.0 (MSYS2 UCRT64, OpenMP, MUMPS 5.9, cuDSS-Lader; `hpfem_gpu.dll` mit MSVC/CUDA 12.9/cuDSS 0.8; delvewheel) in der venv. 1 nm lateral: 328 s → 9.2 s (cuDSS), 0.5 nm: 78 s; Anleitung README, `tools/build_hpfem_gpu.bat`. | Projektinhaber (2026-11-06) |
| OQ-27 | decided | CG-Modell für Aptamere (ADR 0017 Stufe 2): 3SPN.2C (drei Stellen je Nukleotid, Debye-Hückel, winkelabhängige Basenpaarung/Stapelung, neuer Mehrkörper-Kernel), oxDNA2 (starre Nukleotide, Rotationsfreiheitsgrade, tiefer Integrator-Umbau) oder Dry MARTINI (implizit, aber nur MARTINI 2 und ohne verifizierte DNA-Parameter)? Literaturstand (Parameter, Gültigkeit für ssDNA-Haarnadeln, Salzabhängigkeit) vor der Umsetzung verifizieren. | Aptamere bleiben atomistisch | Vorschlag Claude: 3SPN.2C prüfen und als Stufe 2 planen; bis dahin atomistische Aptamere, CG nur für Antikörper. **Projektinhaber 2026-11-06:** Vorschlag angenommen. | Projektinhaber (2026-11-06) |
| OQ-28 | decided | Gemischte Auflösung (ADR 0017 Punkt 5): atomistisches Aptamer und CG-Antikörper in einem Kontext. LJ zwischen Atom und Perle über die volle Paartabelle – mit welchen Parametern (Lorentz-Berthelot aus Atom-σ und Perlen-σ, oder kalibriert an atomistischen Kontakt-PMFs)? Gemeinsamer Zeitschritt nach der atomistischen Komponente. | – | Vorschlag Claude: zunächst Lorentz-Berthelot als ASSUMPTION, Kalibrierung erst bei Bedarf (Bindungsstudien Aptamer–Antikörper). **Projektinhaber 2026-11-06:** Vorschlag angenommen. | Projektinhaber (2026-11-06) |
| OQ-29 | decided | Wandparameter für CG-Perlen (ADR 0017): Gold/Graphen-Wand über Perlen-σ/ε nach ADR 0007 – λ je Perlentyp aus Adsorptionsenergien von Aminosäuren auf Gold (Literatur, z. B. aus Simulationen mit GolP/INTERFACE-FF) oder aus atomistischen Läufen der eigenen Engine kalibrieren? | λ = 1 (unkalibriert) | Vorschlag Claude: aus atomistischen Adsorptionsläufen der eigenen Engine je Residuentyp (konsistent mit dem Rest), Literatur als Gegenprobe. **Projektinhaber 2026-11-06:** Vorschlag angenommen. | Projektinhaber (2026-11-06) |
| OQ-30 | assumption | Parameter des Cα-ENM/Gō-Modells (ADR 0017, `biooptomd.cg`): k_bond = 1e4 kJ/mol/nm², k_angle = 50 kJ/mol (kosinus-harmonisch), eps_go = 2.5 kJ/mol (≈ 1 k_BT), unspezifisches LJ σ = 0.4 nm / ε = 0.05 kJ/mol, His neutral; ENM nach ElNeDyn (0.9 nm, 500 kJ/mol/nm²). | Defaults wie genannt | Vorschlag Claude: an IgG-Referenzdaten prüfen (Fluktuationen gegen atomistische MD bzw. Experiment, Gelenkwinkelverteilung), bis dahin als Annahme kennzeichnen. | – (offen) |

Implementierungsnotizen aus den Phasen 1–6 (Stand vor der Entscheidung):
- **OQ-03:** OL15 + GAFF2. Phase-4-Testsystem enthält mangels verfügbarer GAFF2/RESP-Parameter keinen Fluorophor; Platzhalter: zweiter DNA-Duplex (1558 Atome, `make_duplex_ions.py`).

> **Hinweis zu Literaturangaben:** Die Angaben des ersten Reviews stammten aus dem Gedächtnis. Im zweiten Review (2026-11-04) wurden Bibliographie und, wo möglich, Zahlenwerte per Websuche gegen Verlags-, PubMed-/PMC- und Projektseiten geprüft. Der Prüfstand jedes Punkts steht in der Checkliste am Ende. Punkte ohne Haken sind weiterhin unverifiziert. Primärtabellen (Parameterwerte) wurden nur dort eingesehen, wo es ausdrücklich vermerkt ist.

---

## Begründungen (Review 2026-11-04, angenommen)

### OQ-01 · Polarisierbarkeiten α′(λ)

OQ-01 und OQ-02 hängen über die Lorentz-Lorenz-Mischung im `RefractiveIndexProfiler` (PHYSICS_MODEL §6.3) direkt zusammen und sollten **gemeinsam** kalibriert werden.

- **Biomoleküle (nicht absorbierend im Sichtbaren):** Additive Typtabelle (Miller 1990, ahp-Werte) ist brauchbar, weil die Dispersion zwischen statischer und optischer Polarisierbarkeit im Sichtbaren nur wenige Prozent beträgt. λ-Abhängigkeit über einen Cauchy-Term `α′(λ) = α′_∞ (1 + B/λ²)`.
- **Kalibrierung:** Ein globaler Skalierungsfaktor pro Molekülklasse (Protein, Nukleinsäure) wird so gewählt, dass das Modell inklusive Volumenausschluss (OQ-02) das gemessene Brechungsindexinkrement reproduziert: dn/dc ≈ 0,185–0,19 mL/g (Proteine), ≈ 0,17 mL/g (DNA), jeweils bei ~589 nm.
- **Fluorophore:** In der Nähe der Absorption ist Re α′ resonant und nicht additiv. Fluorophore erhalten im Profiler eine eigene Gesamtpolarisierbarkeit (QM, z. B. TD-DFT, oder Messwert), verteilt auf ihre Atome, oder werden per Option ausgeschlossen.
- **Abnahmetest (Phase 6, neu):** `test_refractive_index_dn_dc` – homogene Lösung bekannter Konzentration c ergibt n = n_w + c · dn/dc innerhalb der Kalibriertoleranz.

**Nachtrag (Review 2):**
- Miller, J. Am. Chem. Soc. **112**, 8533–8542 (1990), bestätigt. Der Begleitartikel ebd. 8543–8551 liefert anisotrope Atompolarisierbarkeiten (Tensor) – relevant, falls später eine orientierungsabhängige Polarisierbarkeit (Doppelbrechung dünner Schichten) gebraucht wird.
- dn/dc Proteine: Sequenzbasierte Auswertung (Zhao et al., Biophys. J. 2011, PMC3149238) ergibt für unmodifizierte Proteine im Mittel ≈ 0,190 mL/g bei 589 nm (Standardabweichung ≈ 0,003 mL/g). Der verbreitete Konsenswert 0,185 mL/g bezieht sich auf ~633 nm. Die Kalibrierwellenlänge muss daher im Test explizit stehen.
- dn/dc Nukleinsäuren: DNA ≈ 0,17 mL/g, RNA 0,17–0,19 mL/g in wässrigem Puffer bei 632,8 nm (Tabellenwerte nach Theisen et al., *Refractive Increment Data-Book*). Nur Handbuchwerte, Salzabhängigkeit offen → für den eigenen Puffer messen oder Unsicherheit als Toleranz in `test_refractive_index_dn_dc` übernehmen.
- Zusätzliche Kalibrierroute: Die McMeekin-Methode (Residuenrefraktion + v̄ pro Aminosäure, Grundlage des Zhao-Rechners) liefert dn/dc **pro Residuum**. Damit lässt sich die Typtabelle auf Residuenebene prüfen statt nur über einen globalen Faktor.

### OQ-02 · Atomvolumina

- Bondi-Radien **ohne** Überlappungskorrektur überschätzen das Molekülvolumen stark, da kovalent gebundene Atome sich erheblich überlappen. Damit wird f_w systematisch zu klein.
- **Vorschlag:** Voronoi-basierte Atom-/Gruppenvolumina aus Kristallstrukturen (Tsai, Taylor, Chothia & Gerstein 1999 bzw. Folgearbeiten). Kontrolle: Summe der Atomvolumina ≈ partielles spezifisches Volumen (≈ 0,73 mL/g Proteine; grob 0,5–0,55 mL/g Nukleinsäuren, salzabhängig).
- **Gauß-Verschmierung als Default:** Bei Δz = 0,05 nm sind die Schichten viel dünner als ein Atom. Mit Punktatomen greift das `max(0, …)` in f_w häufig und erzeugt einen systematischen Bias. Breite σ_i = Atomradius/√5 (gleiche Varianz wie eine homogene Kugel; Vorschlag, im Test prüfen).

**Nachtrag (Review 2):**
- Tsai, Taylor, Chothia & Gerstein, J. Mol. Biol. **290**, 253–266 (1999), bestätigt (ProtOr-Satz, 173 Proteinatomtypen).
- Nukleinsäuren: Voss & Gerstein, J. Mol. Biol. **346**, 477–492 (2005), NucProt-Satz, ein konsistenter Volumensatz für RNA und Protein. DNA-Basen haben laut den Autoren ähnliche Packungsvolumina wie RNA-Basen. → NucProt als Quelle für Aptamere.
- **Achtung Kalibrierbedingung:** Das Voronoi-Packungsvolumen von RNA liegt nach Voss/Gerstein bei ≈ 0,58 mL/g. Experimentelle partielle spezifische Volumina von Nukleinsäuresalzen sind wegen Elektrostriktion und Gegenionen kleiner (Größenordnung aus dem ersten Review, nicht verifiziert). Die Bedingung „Σ v_i = v̄“ gilt daher nur für Proteine. Für Nukleinsäuren wird gegen NucProt kalibriert und die Abweichung zu v̄ dokumentiert.
- **Verschmierung, Alternative zur Gauß-Form:** Exakter Kugelschnitt: Der Anteil des Atomvolumens in Schicht [z_a, z_b] folgt analytisch aus π∫(R² − (z − z_i)²) dz. Vorteile: kompakter Träger, keine Gauß-Ausläufer unter die Oberfläche (z < 0), exakte Volumenerhaltung, gleicher Aufwand. Vorschlag: Kugelschnitt als Default, Gauß als Option, Punktatome für Tests. R_i = (3 v_i / 4π)^{1/3} aus dem Voronoi-Volumen.

### OQ-03 · Kraftfeld

- **DNA-Aptamere:** OL21 (Verfeinerung der OL15-Linie) bevorzugt, falls in der verwendeten AmberTools-Version über tleap verfügbar; sonst OL15. bsc1 ist gleichwertig. **RNA-Aptamere:** OL3.
- Begründung für Pragmatismus: Der Unterschied zwischen diesen Varianten ist klein gegenüber dem Modellfehler durch Debye-Hückel ohne Desolvatation (ADR 0004). Entscheidend ist, dass Builder (ParmEd) und OpenMM-Referenz **dasselbe** prmtop verwenden.
- **Farbstoffe:** Vorhandene AMBER-DYES-Parameter (Graen et al. 2014; Schepers & Gohlke 2020) für Cy3/Cy5, Alexa, ATTO verwenden. GAFF2 + RESP nur für Farbstoffe ohne Parametersatz.
- **Übergangsdipol:** Atompaar (A, B) pro Farbstoff einmalig aus QM bestimmen, im Builder als Tabelle hinterlegen (Phase 7).

**Nachtrag (Review 2):**
- OL21 ist in tleap über `leaprc.DNA.OL21` ab AmberTools23 verfügbar. Die Amber-Seite empfiehlt (Stand AmberTools24) OL21 für DNA und OL3 für RNA.
- Neuer: **OL24** (Zgarbová, Šponer & Jurečka, J. Chem. Theory Comput. 2025, doi:10.1021/acs.jctc.4c01100). OL24 stabilisiert die A-Form über eine neue Zuckerpucker-Torsion und übernimmt χOL3 aus RNA. Die Olomouc-Gruppe empfiehlt es inzwischen als aktuelles DNA-Kraftfeld, besonders für Protein-DNA-Komplexe. Für B-DNA-Haarnadeln und G-Quadruplexe teilen OL21 und OL24 die relevanten Korrekturen (βOL1, εζOL1, α/γOL21). Vorschlag: OL21 als Default, OL24 als Builder-Option. Offen: ab welcher AmberTools-Version OL24 in tleap enthalten ist.
- AMBER-DYES: Graen, Hoefling & Grubmüller, J. Chem. Theory Comput. **10**, 5505–5512 (2014), bestätigt (22 Farbstoffe aus Alexa, Atto, Cy). „AMBER-DYES in AMBER“: Schepers & Gohlke, J. Chem. Phys. 2020 (PMID 32534525), bestätigt. **Einschränkung:** Diese Implementierung enthält Linker zur Kopplung an **Proteine**. Linker zur Nukleinsäure (z. B. Aminohexyl- oder C6-Thiol-Modifikationen) sind dort nicht enthalten und müssen selbst parametrisiert werden (GAFF2 + RESP), mit dem Farbstoffkern aus AMBER-DYES.

### OQ-04 · Normierung der optischen Gradientenkraft

Konvention: Feld `E(t) = Re[E₀ e^{−iωt}]`, E₀ = komplexe **Spitzen**amplitude. Dann (zeitgemittelt, α = 4πε₀ α′):

```
U_i = −¼ Re(α_i) |E₀|²  =  −α′_i |E₀|² / (4 · ONE_4PI_EPS0)
```

mit |E₀| in kJ/(mol·e·nm) = |E₀|[V/nm] · FARADAY. Im `VoxelGridForce3D` wird daher **|E₀|²** in (V/nm)² samt Gradient gespeichert; der Typskalierungsfaktor ist `s_i = −α′_i · FARADAY² / (4 · ONE_4PI_EPS0)`. Bei RMS-Feldern ist der Vorfaktor ½ statt ¼ – der Lader erhält ein Pflichtargument `amplitude="peak" | "rms"`.

**Größenordnung:** 1 MW/cm² in Wasser → |E₀| ≈ 2,4·10⁻³ V/nm; mit α′ ≈ 10⁻³ nm³ pro Atom folgt U ≈ 10⁻⁷ kJ/mol (k_BT ≈ 2,5 kJ/mol). Erst bei plasmonischer Feldverstärkung ≳ 10³ (Intensität ≳ 10⁶) werden ~0,1 kJ/mol pro Atom erreicht. Optische Kräfte sind daher nur in Hotspots relevant; Priorität von `VoxelGridForce3D` liegt zunächst bei Kavitätenpotentialen.

**Nachtrag (Review 2): Überschusspolarisierbarkeit.** Ein Atom in Wasser verdrängt Lösungsmittel. Die Gradientenkraft hängt am Polarisierbarkeits-**Kontrast**, nicht an α′ allein. Konsistent mit dem Lorentz-Lorenz-Ansatz in PHYSICS_MODEL §6.3:

```
L_w       = (n_w² − 1)/(n_w² + 2)
α′_eff,i  = α′_i − (3/4π) · v_i · L_w          (v_i aus OQ-02)
s_i       = −α′_eff,i · FARADAY² / (4 · ONE_4PI_EPS0)
```

Größenordnung: Für v_i ≈ 0,01 nm³ ist (3/4π) v_i L_w ≈ 0,5·10⁻³ nm³, also etwa die Hälfte eines typischen α′_i ≈ 10⁻³ nm³. Ohne diese Korrektur ist die optische Kraft um rund Faktor 2 überschätzt. Wasserstoffreiche Gruppen können sogar negativen Kontrast haben.

Offen bleibt der Lokalfeldfaktor. Ob E₀ das makroskopische Feld in Wasser ist (FEM-Ausgabe) und wie das additive Bild mit dem Clausius-Mossotti-Ergebnis einer dielektrischen Kugel zusammenpasst, soll ein Konsistenztest klären (`test_voxel_optical_sphere_consistency`, ROADMAP Phase 6).

**Ergebnis (Phase 10.0, 2026-11-06):** Der Test ist umgesetzt. In Vakuum stimmt das additive Bild exakt mit Clausius-Mossotti überein. Im Wasser unterschätzt es die Kraft um den Faktor ≈ 1.59–1.67, für kleinen Kontrast ist das der Lorentz-Faktor ((n_w² + 2)/3)². Entscheidung über eine Korrektur: OQ-22.

**Grenzen:** Streu- und Absorptionskraft (∝ Im α, Phasengradient) sind nicht konservativ und lassen sich nicht als Potential im Voxelgitter darstellen. Für Fluorophore nahe der Resonanz ist das Modell daher unvollständig. Vorschlag: Fluorophore im Optikmodus per Default ausschließen. Erwärmung in plasmonischen Hotspots ist nicht modelliert.

### OQ-05 · Wandpotential Graphen vs. Gold

Siehe **ADR 0007** (Accepted). Kurzfassung:

- Die 9-3-Wand entsteht durch Integration über einen **Halbraum** – korrekt für Gold, **nicht** für eine Graphen-Monolage. Dort gilt das 10-4-Potential (eine Atomebene), für Graphit Steeles 10-4-3.
- ε_w, σ_w pro Atomtyp werden über Lorentz-Berthelot aus den Kraftfeld-LJ-Parametern und den Wandatom-Parametern abgeleitet (kein freier Fit).
- Vakuumabgeleitete Parameter überschätzen im impliziten Wasser die Adsorption (fehlende Konkurrenz mit Wasser). Korrektur: typweiser Skalierungsfaktor λ_i auf ε_w, kalibriert an Adsorptions-PMFs von Nukleobasen/Nukleosiden in **explizitem** Wasser (Literatur oder eigene OpenMM-Läufe).

**Nachtrag (Review 2):**
- Steele, Surf. Sci. **36**, 317–352 (1973), bibliographisch bestätigt. Die Zahlen σ_C = 0,340 nm, Δ = 0,335 nm, ρΔ = 38,2 nm⁻² (Graphen-Flächendichte) sind in der Sekundärliteratur konsistent. ε_C/k_B = 28 K wurde **nicht** an der Primärtabelle geprüft. Hinweis aus der Literatur (Do et al.): Experimente (LEED, RHEED) zeigen kleinere Adsorptionsabstände als das Steele-Modell. Für physisorbierte Gase ist das relevant, für Nukleobasen in Wasser ist es durch λ ohnehin zu kalibrieren.
- Heinz, Vaia, Farmer & Naik, J. Phys. Chem. C **112**, 17281–17290 (2008), bestätigt: 12-6- und 9-6-LJ-Parameter für Au u. a., angepasst an Dichte und Oberflächenenergie, kompatibel mit AMBER/CHARMM. Werte selbst nicht abgeglichen. Die Grenzflächeneigenschaften wurden in **explizitem** Wasser validiert. Im impliziten Wasser fehlt die Wasserverdrängung, deshalb ist die λ-Skalierung für Gold praktisch Pflicht und nicht optional.
- Polarisierbares Gold als Referenz für die Kalibrierung: GolP-CHARMM (Wright, Rodger, Corni & Walsh, J. Chem. Theory Comput. **9**, 1616–1630 (2013)), Au(111) und Au(100), Bildladungen über starre Dipole.
- Option für die Wand: Ein abgeschirmter Bildladungsterm ist für das Eigenbild einatomar (Wandkernel). Kreuzterme zwischen Atom i und dem Bild von Atom j sind jedoch paarweise und gehörten in den Non-Bonded-Kernel. Nicht für V1. Falls gewünscht, eigene OQ.
- **Lage von z = 0** (Ergänzung zu ADR 0007): Bei `Wall93` beginnt das Kontinuum nicht an der obersten Atomlage, sondern etwa Δ/2 darunter (Au(111): Δ = a/√3 ≈ 0,2355 nm). Bei `Wall104` liegt die Atomebene bei z = 0. Die Konvention bestimmt direkt die Chemisorptionshöhe in OQ-07 und gehört deshalb in ADR 0007.

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

**Nachtrag (Review 2):** Numerik: `ln((1+u)/(1−u))` als `2·atanhf(u)` auswerten. Das ist stabiler, wenn γ → 1 (große Φ₀). Der z-Clamp garantiert u ≤ γ < 1.

### OQ-07 · Thiol-Gold-Anker

1. **Ankeratom von der Wand ausnehmen** (Oberflächen-Typ mit ε_w = 0). Sonst stößt die Wand das S-Atom genau dort ab, wo der Restraint es festhält.
2. **z-Restraint** auf Chemisorptionshöhe (≈ 0,2–0,25 nm über der obersten Au-Lage; genaue Lage hängt von der Definition z = 0 der Wand ab – im Builder dokumentieren).
3. **xy-Restraint** nur, wenn mehrere Rezeptoren pro Box eine definierte Belegungsdichte haben sollen. Bei einem Rezeptor pro Box ist er unnötig, da n(z) lateral gemittelt wird.
4. **SAM/MCH-Backfill:** Reale Sensoren tragen fast immer eine Mercaptohexanol-Schicht, die das Flachliegen des Aptamers verhindert und die Konformation stärker prägt als die Ankerdetails. Günstiges Modell: zusätzliche weiche, rein repulsive Wand (WCA-artig) bei z ≈ 0,8–1,0 nm, die nur auf Atome oberhalb des Linkers wirkt (eigener Oberflächen-Typ).

**Nachtrag (Review 2):** Der Adsorptionsplatz von Alkanthiolaten auf Au(111) ist in der Literatur nicht einheitlich: Bridge, Hollow oder zwischen beiden (z. B. Yourdshahyan & Rappe, J. Chem. Phys. 2002), sowie Au-Adatom-Komplexe mit S nahezu über einem Oberflächenatom (NIXSW/NEXAFS). Die S-Höhe ist damit kein fester Literaturwert. Vorschlag: Höhe als Builder-Parameter mit dokumentiertem Default, Sensitivität von n(z) und d(t) gegen ±0,05 nm im Test prüfen. GolP-CHARMM-kompatible Au–S-Parameter gibt es von Futera & Blumberger (Zeitschrift nicht verifiziert).

### OQ-08 · Explizites Mg²⁺ im impliziten Wasser (neu)

Ohne Hydrathülle kollabiert Mg²⁺ in direkten Kontakt mit Phosphat-Sauerstoffen und bindet um Größenordnungen zu stark; real dominiert die Bindung über die äußere Sphäre (Mg(H₂O)₆²⁺). Optionen:

- (a) Effektives hydratisiertes Ion: σ_Mg so vergrößert, dass der Kontaktabstand Mg–O(P) dem äußeren Sphärenabstand entspricht.
- (b) Flat-Bottom-Abstandsterm zwischen Mg und O1P/O2P (Phase-2-Plugin wiederverwendbar).

Beide Varianten brauchen eine Kalibrierung (z. B. Mg–Phosphat-PMF in explizitem Wasser mit OpenMM). Bis zur Entscheidung: Mg²⁺ nur in Tests, nicht in Produktionsläufen.

**Nachtrag (Review 2):**
- Für die Referenz-PMFs in explizitem Wasser **keine** Standard-Mg²⁺-Parameter verwenden. Übliche 12-6-Parametersätze verfehlen Solvatationsenergie und Hydratationsabstand gleichzeitig und unterschätzen den Wasseraustausch um Größenordnungen. Stattdessen: microMg/nanoMg (Grotz, Cruz-León & Schwierz, J. Chem. Theory Comput. **17**, 2530–2540 (2021), TIP3P, reproduzieren u. a. die Bindungsaffinität an Phosphat-O der RNA) bzw. die Erweiterung für Ion-Nukleinsäure-Wechselwirkungen (Cruz-León, Grotz & Schwierz, J. Chem. Phys. **154**, 171102 (2021)). nanoMg tauscht Wasser im ns-Bereich aus und eignet sich für konvergierte PMFs in vertretbarer Zeit.
- Ionenstärke-Buchhaltung festlegen: κ enthält alle **nicht explizit** modellierten Ionen (monovalentes Salz und Cl⁻ aus MgCl₂), explizite Mg²⁺ nicht. Sonst wird doppelt gezählt.

### OQ-09 · Boxgröße, Bedeckung und Cutoff (neu)

Bei einem Rezeptor pro Box ist die Oberflächenbelegung Γ = 1/(L_x L_y). Die Box ist also eine **physikalische** Größe. ADR 0006 (Rev. 1) koppelte L ≥ 2 r_c an die Debye-Länge. Bei 10 mM (r_c ≈ 15 nm) folgte daraus L ≥ 30 nm, also Γ ≈ 1·10¹¹ cm⁻². Reale Aptamersensoren liegen oft bei 10¹²–10¹³ cm⁻² (L ≈ 3–10 nm). Gerade bei niedrigem Salz sind die Nachbarwechselwirkungen dann physikalisch relevant und dürfen nicht durch Vergrößern der Box verschwinden.

Entscheidung (ADR 0006 Rev. 2, Accepted): L kommt aus der gewünschten Bedeckung. Ist L < 2 r_c, summiert der Non-Bonded-Kernel explizit über periodische Bilder in x/y. Die Summe konvergiert wegen exp(−κr) schnell. Das ändert ADR 0003 („Minimum Image in allen Paar-Berechnungen“) und braucht daher eine ausdrückliche Entscheidung.

### OQ-10 · Temperaturabhängigkeit von κ und ε_r (neu)

Gouy-Chapman berechnet γ bereits pro Replica aus T_r (OQ-06). κ hängt aber ebenfalls von T ab (κ ∝ 1/√(ε_r T)), und ε_r von Wasser fällt mit T merklich (Größenordnung −0,4 %/K, nicht verifiziert). Entscheidung: ε_r(T_r) über eine empirische Fitformel (z. B. Malmberg & Maryott 1956, nicht verifiziert), κ_r pro Replica, r_c als Maximum über alle Replicas (ein Kernel, eine Box-Prüfung). Bis die Fitformel verifiziert ist (Checkliste), wird sie im Code mit `ASSUMPTION(OQ-10)` markiert.

---

### OQ-15 · GB-Modell für Aptamer-Produktionsläufe – angenommen vom Projektinhaber (2026-11-05)

**Vorschlag:** GB-neck2 (`solvent="gb-gbn2"`, ADR 0012) mit den Nukleinsäure-Parametern aus
`implicit/gbn2.xml` als **Default für alle Systeme mit Nukleinsäuren** (Aptamere, Duplexe, Komplexe);
OBC2 bleibt als Vergleichs- und Altmodell. Salz wie bisher über κ_r (Srinivasan-Korrektur, OQ-10),
ACE an (Default von OpenMM `gbn2.xml`). **Status nach Annahme:** `assumption`. Das Vorgehen ist damit
entschieden, offen bleibt die Validierung mit OL21 (unten).

**Begründung**
- Die Nukleinsäure-Parametrisierung von GB-neck2 (Nguyen, Pérez, Bermeo & Simmerling, J. Chem. Theory
  Comput. 11(8), 3714–3728 (2015); PMC4805114) senkt den Fehler der Solvatationsenergie gegenüber
  Poisson-Boltzmann deutlich. Sie stabilisiert DNA/RNA-Duplexe, Quadruplexe und Protein-Nukleinsäure-
  Komplexe und faltet kleine DNA/RNA-Hairpins in nahezu native Strukturen. Für Aptamere, deren Funktion
  an Sekundär- und Tertiärstruktur hängt, ist das der entscheidende Punkt.
- OBC2 gibt DNA-Strukturgrößen (z. B. Furchenbreiten) laut älteren Vergleichen schlechter wieder als
  HCT/GBMV2 und erzeugt bei Proteinen einen Helix-Bias. Das war die Motivation für GB-neck2.
- Kosten: GBn2 ≈ OBC2 (Phase 9.3), der Wechsel ist also ohne Durchsatzverlust möglich.

**Einschränkungen (offen zu halten)**
1. Die GBn2-Nukleinsäure-Parameter wurden mit einem älteren AMBER-Nukleinsäure-Kraftfeld trainiert. Mit
   welchem genau, ist nicht verifiziert; vermutlich ff99bsc0-Familie. Die Kombination mit **OL21** (OQ-03)
   ist nicht publiziert validiert.
2. Neuere Arbeiten (GB-Neck3-Preprint 2024) berichten bei Proteinen eine Überstabilisierung von α-Helices
   mit GB-neck2. Für Aptamer-Protein-Komplexe ist das zu beachten.
3. Ob der ACE-Term in den Validierungsläufen des Papers aktiv war, ist nicht verifiziert. Eventuell
   gegenprüfen, ob `ace=False` die bessere Wahl ist.
4. Keine dielektrische Grenze an der Oberfläche (OQ-16). Die Wahl des GB-Modells löst das nicht.

**Validierung vor Produktionsläufen (Vorschlag, eigene Aufgabe)**
- (a) B-DNA-Duplex 1BNA (OL21, GBn2 vs. OBC2, 0.15 M, 300 K, 4 × 100 ns mit HMR/4 fs): RMSD zur
  Kristallstruktur, Basenpaar-Erhalt, Furchenbreiten.
- (b) Kleiner DNA-Hairpin (Literaturbeispiel aus Nguyen et al. 2015): Faltung aus gestreckter Struktur
  mit Replica Exchange (Phase 9.2).
- (c) Optional Vergleich mit expliziter Wasser-Referenz (OpenMM, TIP3P/OPC) für ein Aptamer-Fragment.
Akzeptanz: GBn2 mindestens so stabil wie OBC2 und strukturell näher am Experiment. Sonst OQ-15 neu
bewerten (z. B. GB-Neck3 oder OL21-angepasste Radien).

**Ergebnis (a), 2026-11-05** (`examples/gb_validation/duplex_stability.py`, Seed 2026, RTX 3090;
Mittel über 50–100 ns, ± Standardfehler über die 4 Replicas. Alle Replicas starten aus derselben
Struktur, das ± ist daher ein Rauschmaß, keine Präzision; „0.84 ± 0.00“ heißt: alle 4 Replicas sitzen
bei 10/12 intakten Paaren fest. Referenz ist die Startstruktur: 1BNA
mit Wasserstoffen, 300 Schritte minimiert (OL15, Debye-Hückel), 0.068 nm Schweratom-RMSD zum Kristall.

| Größe | GBn2 | OBC2 |
|---|---|---|
| RMSD Schweratome, alle 12 bp [nm] | 0.215 ± 0.005 | 0.361 ± 0.004 |
| RMSD Schweratome, innere 8 bp [nm] | 0.150 ± 0.001 | 0.197 ± 0.001 |
| Watson-Crick-Paare intakt, alle | 0.94 ± 0.02 | 0.84 ± 0.00 |
| Watson-Crick-Paare intakt, innere 8 | 0.999 | 0.994 ± 0.001 |
| kleinster P–P-Abstand zwischen den Strängen [nm] (Start 1.269) | 1.246 ± 0.003 | 1.238 ± 0.005 |

- OBC2: Der RMSD (alle) steigt über 0–60 ns von 0.28 auf 0.36 nm und bleibt dann konstant. Gleichzeitig
  fällt der Anteil intakter Paare von 0.99 auf 0.83, d. h. im Mittel sind die beiden Endpaare dauerhaft
  geöffnet (Fraying); der Kern bleibt gepaart, ist aber stärker verzerrt (0.197 nm).
- GBn2: Der RMSD bleibt über 100 ns bei 0.20–0.22 nm, der Kern bei 0.15 nm. Die Endpaare öffnen
  zeitweise. Ihr Anteil intakter Paare sinkt in den letzten 30 ns leicht (0.97 → 0.91 je 10-ns-Block);
  ob das ein beginnender Trend oder Schwankung ist, klärt erst ein längerer Lauf.
- Keine Statusfehler, keine Konstraint-Probleme; Durchsatz ≈ 1000 ns/Tag je Replica (4 Replicas).

**Bewertung (a), Einschätzung Claude (vom Projektinhaber bestätigt 2026-11-06):** Akzeptanzkriterium auf der 100-ns-Skala erfüllt; ob GBn2 in
längeren Läufen ebenfalls ausfranst (OBC2 brauchte ≈ 60 ns), ist offen. GBn2 hält den Duplex stabiler als OBC2 und bleibt
deutlich näher an der Kristallstruktur (Kern 0.15 statt 0.20 nm). Einschränkungen: Furchenbreiten
wurden nicht direkt ausgewertet, der minimale P–P-Abstand ist nur ein grobes Ersatzmaß und
unterscheidet die Modelle nicht. Gemessen wurde ein einziges Sequenzsystem bei einem Seed. (b)
Hairpin-Faltung und (c) explizites Wasser stehen aus.

**Quellen (per Websuche geprüft, 2026-11-05):** Stony Brook Research Connect / PMC4805114 (Nguyen et
al. 2015, Bibliographie und Abstract); chemRxiv 666325fbe7ccf7753a3a337c (GB-Neck3-Preprint, 2024).
Volltext von PMC4805114 nicht eingesehen (Browser-Prüfung), Trainingskraftfeld daher unverifiziert.

## Checkliste: Literatur/Zahlen (Stand Review 2, 2026-11-04)

Legende: [x] bibliographisch bestätigt (Websuche), Zahlenwerte wie angegeben · [~] Quelle bestätigt, Zahlenwert nur aus Sekundärquelle oder nicht geprüft · [ ] offen

- [x] Miller, J. Am. Chem. Soc. 112, 8533–8542 (1990), ahp; Tensor-Begleitartikel ebd. 8543–8551
- [x] dn/dc Proteine: ≈ 0,190 mL/g bei 589 nm (Zhao et al., Biophys. J. 2011, PMC3149238); 0,185 mL/g bei ~633 nm
- [~] dn/dc DNA 0,17 mL/g, RNA 0,17–0,19 mL/g (632,8 nm, Handbuchwerte; Puffer-/Salzabhängigkeit offen)
- [x] Tsai, Taylor, Chothia & Gerstein, J. Mol. Biol. 290, 253–266 (1999), ProtOr
- [x] Voss & Gerstein, J. Mol. Biol. 346, 477–492 (2005), NucProt; RNA-Packungsvolumen ≈ 0,58 mL/g
- [ ] Experimentelles partielles spezifisches Volumen von DNA/RNA-Salzen (salzabhängig)
- [x] OL21 in tleap (`leaprc.DNA.OL21`) ab AmberTools23; Amber-Empfehlung DNA OL21 / RNA OL3
- [x] OL24: Zgarbová, Šponer & Jurečka, J. Chem. Theory Comput. 2025, doi:10.1021/acs.jctc.4c01100
- [ ] OL24 in welcher AmberTools-Version über tleap verfügbar
- [x] AMBER-DYES: Graen, Hoefling & Grubmüller, J. Chem. Theory Comput. 10, 5505–5512 (2014)
- [x] AMBER-DYES in AMBER: Schepers & Gohlke, J. Chem. Phys. 2020 (PMID 32534525) – nur Protein-Linker
- [x] Steele, Surf. Sci. 36, 317–352 (1973)
- [~] Steele-Parameter σ_C 0,340 nm, Δ 0,335 nm, ρ 114 nm⁻³ (konsistent); ε_C/k_B 28 K nicht an Primärtabelle geprüft
- [x] Heinz, Vaia, Farmer & Naik, J. Phys. Chem. C 112, 17281–17290 (2008); Zahlenwerte Au nicht abgeglichen
- [x] GolP-CHARMM: Wright, Rodger, Corni & Walsh, J. Chem. Theory Comput. 9, 1616–1630 (2013)
- [x] Grotz, Cruz-León & Schwierz, J. Chem. Theory Comput. 17, 2530–2540 (2021), microMg/nanoMg
- [x] Cruz-León, Grotz & Schwierz, J. Chem. Phys. 154, 171102 (2021)
- [~] Au–S-Adsorptionsplatz/-höhe: Literatur uneinheitlich (Bridge/Hollow/Adatom), kein fester Wert
- [ ] Futera & Blumberger, Au–S-Parameter kompatibel mit GolP-CHARMM (Zeitschrift)
- [ ] ε_r(T) von Wasser (Malmberg & Maryott 1956)
