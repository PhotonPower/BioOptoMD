# ADR 0005: Topologie und Parameter pro Replica

- **Status:** Accepted
- **Datum:** 2026-11-03

## Kontext
Replicas sollen auch unterschiedliche Systeme sein können (Aptamer-Mutationen, verschiedene Fluorophore). Nur Zustandsarrays zu batchen reicht dafür nicht.

## Entscheidung
Alle Topologie-/Parameter-Arrays (Bindungslisten, LJ-Tabellen, Exclusion-Tiles, Ladungen, Massen, Observer-Atomgruppen) werden über ein Offset-Array `d_topo_offset[r]` pro Replica indiziert. Gleiche Topologie = alle Offsets zeigen auf denselben Block (kein Speicher-Mehrbedarf). Alle Replicas teilen `N_pad` (Maximum + Padding).

## Konsequenzen
- Kernels lesen Listenlängen pro Replica; Load-Imbalance bei sehr unterschiedlichen Systemen möglich (akzeptiert).
- Test `test_padding_and_batch_independence` sichert Unabhängigkeit der Replicas.
