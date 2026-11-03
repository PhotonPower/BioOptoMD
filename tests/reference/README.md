# Referenzimplementierungen

- `*_ref.py`: NumPy-Referenzen (FP64) jeder Energie-/Kraftfunktion, inkl. Finite-Differenzen-Checks.
- `openmm_*.py`: Generatoren, die das identische Modell in OpenMM aufsetzen und Kräfte/Energien nach `data/` (gitignored) schreiben.
- `generate_all.py`: erzeugt alle Referenzdaten.
- `inputs/`: kleine Eingabestrukturen (< 1 MB).

Siehe `docs/TESTING.md`.
