# Beispiele

## `aptamer_on_graphene/`
Oberflächenfixiertes Aptamer auf Graphen (ROADMAP Phase 7): Builder aus einem AMBER-Kraftfeld
(OpenMM), Graphen-Wand (10-4, ADR 0007), Anker (OQ-07), 8 Replicas bei 300–340 K mit HMR und
H-Constraints (Δt = 4 fs), Observer (Dipol, FRET, n(z), Energie), Auswertung mit `biooptomd.analysis`.

```bash
python examples/aptamer_on_graphene/aptamer_on_graphene.py --steps 250000 --out run1
```

Ausgabe in `run1/`: DCD je Replica (10 ps je Frame), `n_of_z.csv`, `checkpoint.npz`; auf der Konsole
FRET-Effizienz (dynamisch/statisch), ⟨R_DA⟩ und ⟨d⟩ je Replica.

Platzhalter (siehe Kopf des Skripts und `docs/OPEN_QUESTIONS.md`): DNA-Strang aus 1BNA statt eines
Aptamers mit Fluorophor (OQ-03), O5'-Anker statt Linker (OQ-07), unkalibrierte Optik- und
Wandparameter (OQ-01/02/05). Die Zahlen sind daher nur qualitativ. Die Optikparameter sind
inzwischen an dn/dc kalibriert (Phase 10.0).

## `aptamer_spr_gold/`
Aptamerschicht über Gold im Kretschmann-Aufbau mit selbstkonsistenter Optik (ROADMAP Phase 10.4,
ADR 0016): derselbe Strang über einer Gold-Wand mit Bildladungen und Anker, 4 Replicas bei 300 K,
`biooptomd.coupling.SelfConsistentOptics` mit dem TMM-Referenzsolver (BK7 | Au 50 nm | ⟨ε⟩(z) |
Wasser, p, 72.6°, 633 nm).

```bash
python examples/aptamer_spr_gold/aptamer_spr_gold.py --iterations 10 --out spr_run
```

Ausgabe: Verlauf je Iteration (Δε, ΔW, Rauschen, Selbstfeld), `profiles.npz` (ε(z), W(z), W₀(z)),
`history.json`, Checkpoints je Iteration; auf der Konsole R mit und ohne Schicht (Sensorsignal δR).
Referenzlauf (2026-11-06, RTX 3090, 1 m 33 s, ε-Gitter damals bis 8 nm): δR ≈ −4.0e-3; ΔW sinkt in 10 Iterationen von 1.1e-2
auf 1.7e-3. Δε ist anfangs vor allem die Relaxation des Strangs zum Gold (bei 1 MW/cm² ist die
optische Kraft klein), dabei wächst der Selbstfeldanteil auf ≈ 17 % (Warnung OQ-24). Für Produktion
länger äquilibrieren, bevor die Iteration beginnt.

## `igg_on_gold/`
CG-Antikörper (IgG, PDB 1IGT) über Gold mit Replica Exchange (ROADMAP Phase 11.4, ADR 0017):
`biooptomd.cg.enm_go_from_pdb` mit sieben Domänen (je Arm Fv und CL/CH1, zwei CH2, CH3-Paar; Gelenk
ohne Domäne), Gold-Wand mit λ je Residuum (`cg.solvent_scale`, unkalibriert, OQ-29), Bildladungen,
Decke bei 30 nm (Gitterpotential), 4 Leitern × 8 Temperaturen (300–370 K), Δt = 20 fs.

```bash
python examples/igg_on_gold/igg_on_gold.py --ns 2000 --out run_igg
```

Ausgabe: `series.npz` (Höhe, Kontakte und Achsen-Kosinus je Gruppe Fab 1/Fab 2/Fc, Zustand je
Slot, alle 100 ps), `summary.json` (je Leiter bei 300 K: adsorbierter Anteil, Orientierungsklassen
desorbiert/„Fc unten“/„Fab unten“/seitlich/flach), `checkpoint.npz` (bitgleich fortsetzbar).
Durchsatz (RTX 3090, 1316 Perlen, Wand + Bilder + Decke, Paarliste): 9.7 µs/Tag je Replica bei
8 Replicas, 4.6 µs/Tag je Replica (148 µs/Tag gesamt) bei 32 Replicas. Die Adsorptionsstärke ist
mit λ = 1 nur qualitativ (OQ-29).
