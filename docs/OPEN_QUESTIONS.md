# Offene Fragen

Format: `OQ-NN` · Status (`open` / `assumption` / `resolved`) · Frage · aktuelle Annahme · Entscheider.
Im Code referenziert als `// ASSUMPTION(OQ-NN): …`.

| ID | Status | Frage | Aktuelle Annahme |
|---|---|---|---|
| OQ-01 | open | Quelle für atomare Polarisierbarkeiten α′(λ) für Nukleinsäuren/Proteine/Fluorophore? (Miller 1990, Thole-Modell, AMOEBA-Parameter, QM-Fit?) Statische vs. optische Frequenz? | Platzhalter: elementbasierte Tabelle (Miller), λ-unabhängig |
| OQ-02 | open | Atomvolumina für Volumenausschluss im n(z)-Profil: Bondi-vdW-Volumen oder Voronoi-basierte Volumina aus Proteindichte? | Bondi-Radien, Überlappung ignoriert |
| OQ-03 | open | Kraftfeld für Aptamere: AMBER OL15 / bsc1 / OL21? Parameterquelle für Fluorophore (GAFF2 + RESP)? | OL15 + GAFF2 |
| OQ-04 | open | Normierung der optischen Gradientenkraft im `VoxelGridForce3D` (Feldstärke in V/nm, Leistungsdichte, Mittelung über Periode)? | Eingabe bereits als Potential in kJ/mol pro α′-Einheit |
| OQ-05 | open | Wie wird die 9-3-Wand für Graphen vs. Gold parametrisiert (ε_w, σ_w pro Atomtyp)? Literatur (z. B. Steele-Potential) oder eigener Fit an All-Atom-Graphen? | Steele-artige Werte für Graphen, Platzhalter für Gold |
| OQ-06 | open | Ist linearisiertes Gouy-Chapman bei relevanten Elektrodenpotentialen (> 25 mV) ausreichend, oder nichtlineare Lösung als Tabelle? | linearisiert |
| OQ-07 | open | Thiol-Gold-Anker: harter Positions-Restraint oder lateral beweglicher Anker (nur z-Restraint)? | nur z-Restraint + schwacher xy-Restraint |
| OQ-08 | assumption | ROADMAP Phase 1 fordert für `test_wrap_xy` eine „exakte“ ungewrappte Rekonstruktion. In FP32 ist x − n·L für x ∈ (−L, 0) nicht immer darstellbar. Was ist gemeint? | Exakt (bitgenau) für \|x\| ≥ L, sonst Fehler ≤ ½ ulp(L) (eine Rundung von x + L); gewrappte Werte und Imagezähler bitgenau gleich der NumPy-Referenz. Projektinhaber bitte bestätigen. |
| OQ-09 | assumption | Gouy-Chapman-Term für Atome hinter der Wand (ζ < 0): wie fortsetzen? | V = q Φ0 konstant, F = 0 (energiekonsistent; die 9-3-Wand verhindert ζ < 0 praktisch). |
| OQ-10 | open | Widerspruch PHYSICS_MODEL §4.1 ↔ ADR 0002: Am Clamp-Punkt ζ_c = 0.1 σ ist \|F\| ≈ 1.2·10⁹ ε/ζ_c ≈ 10¹¹ kJ/mol/nm ≫ 2³¹, und schon Atome knapp oberhalb erzeugen Energien > 2³¹ kJ/mol. Folge: Fixed-Point-Beiträge sättigen, die int64-Summe kann überlaufen und das Vorzeichen kippen (im Test nachgewiesen). Soll (a) ζ_c so gewählt werden, dass \|F(ζ_c)\| unter einer Schranke bleibt (z. B. 10⁶ kJ/mol/nm), (b) die Fixed-Point-Konvertierung pro Beitrag auf ±2³⁰ sättigen und ein Geräte-Fehlerflag setzen, oder beides? | Unverändert: Clamp 0.1 σ laut Spezifikation, keine Sättigung. Für reale Trajektorien irrelevant (V(ζ_c) ~ 10⁸ kJ/mol), aber eine explodierende Simulation wird nicht erkannt. Vorschlag: (a) + (b) per ADR. |

