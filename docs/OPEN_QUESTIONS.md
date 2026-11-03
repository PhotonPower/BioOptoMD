# Offene Fragen

Format: `OQ-NN` · Status (`open` / `assumption` / `resolved`) · Frage · aktuelle Annahme · Entscheider.
Im Code referenziert als `// ASSUMPTION(OQ-NN): …`.

| ID | Status | Frage | Aktuelle Annahme |
|---|---|---|---|
| OQ-01 | open | Quelle für atomare Polarisierbarkeiten α′(λ) für Nukleinsäuren/Proteine/Fluorophore? (Miller 1990, Thole-Modell, AMOEBA-Parameter, QM-Fit?) Statische vs. optische Frequenz? | Platzhalter: elementbasierte Tabelle (Miller), λ-unabhängig |
| OQ-02 | open | Atomvolumina für Volumenausschluss im n(z)-Profil: Bondi-vdW-Volumen oder Voronoi-basierte Volumina aus Proteindichte? | Bondi-Radien, Überlappung ignoriert |
| OQ-03 | open | Kraftfeld für Aptamere: AMBER OL15 / bsc1 / OL21? Parameterquelle für Fluorophore (GAFF2 + RESP)? | OL15 + GAFF2. Phase-4-Testsystem enthält mangels verfügbarer GAFF2/RESP-Parameter keinen Fluorophor; Platzhalter: zweiter DNA-Duplex (1558 Atome, `make_duplex_ions.py`). |
| OQ-04 | open | Normierung der optischen Gradientenkraft im `VoxelGridForce3D` (Feldstärke in V/nm, Leistungsdichte, Mittelung über Periode)? | Eingabe bereits als Potential in kJ/mol pro α′-Einheit |
| OQ-05 | open | Wie wird die 9-3-Wand für Graphen vs. Gold parametrisiert (ε_w, σ_w pro Atomtyp)? Literatur (z. B. Steele-Potential) oder eigener Fit an All-Atom-Graphen? | Steele-artige Werte für Graphen, Platzhalter für Gold |
| OQ-06 | open | Ist linearisiertes Gouy-Chapman bei relevanten Elektrodenpotentialen (> 25 mV) ausreichend, oder nichtlineare Lösung als Tabelle? | linearisiert |
| OQ-07 | open | Thiol-Gold-Anker: harter Positions-Restraint oder lateral beweglicher Anker (nur z-Restraint)? | nur z-Restraint + schwacher xy-Restraint |
| OQ-08 | resolved | ROADMAP Phase 1 fordert für `test_wrap_xy` eine „exakte“ ungewrappte Rekonstruktion; in FP32 nicht für alle x möglich. | Durch ADR 0007 gelöst: Wrap in FP64 auf der Mixed-Precision-Position, Rekonstruktion x_w + n·L = x bitgenau. |
| OQ-09 | assumption | Gouy-Chapman-Term für Atome hinter der Wand (ζ < 0): wie fortsetzen? | V = q Φ0 konstant, F = 0 (energiekonsistent; die 9-3-Wand verhindert ζ < 0 praktisch). |
| OQ-10 | resolved | Widerspruch PHYSICS_MODEL §4.1 ↔ ADR 0002: Clamp bei 0.1 σ erzeugte Kräfte ≫ 2³¹, Fixed-Point-Summen konnten überlaufen. | Entschieden 2026-11-03 (Projektinhaber): Kraftschranke für ζ_c + Sättigung pro Beitrag + Geräte-Statusflags, siehe ADR 0006. |

