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
Wandparameter (OQ-01/02/05). Die Zahlen sind daher nur qualitativ.
