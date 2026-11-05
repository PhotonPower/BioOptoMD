# ADR 0014: Kompakte Tile-Paarliste statt Cell-List für große Einzelsysteme

- **Status:** Proposed
- **Datum:** 2026-11-06
- **Bezug:** ARCHITECTURE §6 („CellList für > 10 k Atome“), ADR 0002 (Fixed-Point), ADR 0006
  (Cutoff aus κ, Bildsumme), Phase 8 (Benchmark 1 × 20 k: 0.59× OpenMM), ROADMAP 9.6

## Kontext
ROADMAP und ARCHITECTURE sehen für Systeme über 10 k Atome eine Cell-List-Strategie vor. Vor dem Entwurf
wurde gemessen, wo der Aufwand im Fall 1 × 20 k (18 696 Atome, 585 Tiles, r_c = 3.92 nm, Box 24 × 32 nm)
tatsächlich liegt. Die GPU war dabei mit einem Produktionslauf geteilt, daher sind nur Verhältnisse
aussagekräftig:

1. **Räumliche Sortierung bringt nichts.** Tiles in Atomreihenfolge sind bei Biomolekülen bereits
   kompakt (Ketten zusammenhängend nummeriert). Nach der Vorauswahl bleiben 22 767 von 171 405
   Tile-Paaren (13.3 %, 23.3 M Paarprüfungen für 11.4 M echte Paare < r_c). Morton-sortiert in
   1-nm-Zellen sind es 25 128 (14.7 %) – mehr, weil Zellgrenzen die Ketten zerschneiden.
2. **Der Leerlauf dominiert.** Der Kernel startet einen Warp je Tile-Paar (171 k Warps, 42.8 k Blöcke),
   jeder Block lädt zuerst die LJ-Typtabelle in den Shared Memory, dann verwerfen 87 % der Warps ihr
   Paar. `compute_forces` kostet 922 µs; liegen alle Tiles so weit auseinander, dass jedes Paar außer der
   Diagonale verworfen wird, immer noch 371 µs. Etwa 300 µs je Auswertung sind also reiner Leerlauf.
3. Nsight Systems je Schritt: Nonbonded 70 %, SHAKE 16 %, RATTLE 8 % (Constraint-Löser in FP64, ein
   Thread je Cluster – eigener Befund, nicht Teil dieses ADR).

Eine Cell-List im klassischen Sinn (Atome nach Zellen sortieren) behebt Punkt 2 nicht und
verschlechtert Punkt 1.

## Entscheidung

1. **Tile-Paarliste je Auswertung** im Minimum-Image-Pfad von `TiledAllPairs` (L ≥ 2 r_c): Ein Kernel
   prüft mit einem Thread je (Replica, Tile-Paar) die Hüllen wie bisher (`tiles_out_of_range`, gleiche
   Margins) und schreibt die verbleibenden Tile-Paar-Indizes in eine Liste je Replica (Kapazität =
   alle Tile-Paare, Zähler je Replica auf dem Gerät, warpweise aggregierte Atomics).
2. **Interaktionskernel mit fester Gittergröße:** Gitter (B, n_replicas) mit B aus SM-Zahl und
   Replica-Zahl (persistente Blöcke), jeder Block lädt die LJ-Tabelle einmal und arbeitet die Liste
   seiner Replica warpweise in Gitterschritten ab. Der Host kennt die Listenlänge nicht und muss sie
   nicht kennen (keine Synchronisation).
3. **Rechnung je Tile-Paar unverändert** (gleiche Lane-Zuordnung, gleiche Rotation der j-Atome, gleiche
   Exclusion-Masken). Weil Kräfte und Energien über Fixed-Point-Integer akkumuliert werden (ADR 0002),
   ist das Ergebnis **bitidentisch** zum bisherigen Kernel, unabhängig von der Reihenfolge der Liste.
4. **Keine Verlet-Haut, Neuaufbau bei jeder Auswertung.** Die Liste folgt den aktuellen Hüllen; es gibt
   keinen veralteten Zustand und keine Rebuild-Logik. Kosten: ein Thread je Tile-Paar (171 k bei 20 k
   Atomen) statt eines Warps mit Shared-Memory-Prolog.
5. **Atomreihenfolge bleibt.** Keine räumliche Sortierung (Messung oben); Exclusion-Masken, Observer und
   Topologie bleiben unberührt.
6. **Gleiche Technik für `TiledGeneralizedBorn`** (drei Tile-Passes mit derselben Vorauswahl), sofern der
   Aufwand gering bleibt; sonst als eigener Schritt. `ImageChargeForce` (ein billiger Pass, eigene
   Vorauswahl) bleibt unverändert.
7. **Bildsummenpfad (L < 2 r_c) unverändert**, dort gibt es keine Vorauswahl.
8. Die Strategie ersetzt den bisherigen Minimum-Image-Pfad vollständig (kein Nutzerschalter);
   ein interner Schalter `use_tile_list = false` bleibt für Tests und Vergleichsmessungen.

## Alternativen
- **Cell-List mit räumlicher Sortierung (OpenMM-artig, Hilbert-/Morton-Ordnung):** verschlechtert hier
  die Tile-Kompaktheit; braucht Permutationen, Exclusion-Prüfung über Atompaare statt Tile-Masken und
  bricht die Bitidentität. Für Systeme mit vielen freien Kleinmolekülen später möglich.
- **Verlet-Liste mit Haut und seltenem Neuaufbau:** spart den billigen Listenaufbau, braucht aber eine
  Verschiebungsprüfung und Rebuild-Steuerung ohne Host-Synchronisation. Erst sinnvoll, wenn der
  Listenaufbau messbar ins Gewicht fällt.
- **Warp verwirft vor dem Shared-Memory-Prolog:** halbiert nur einen Teil des Leerlaufs; 171 k Warps
  bleiben.

## Konsequenzen
- Erwartung (vor Messung): Nonbonded bei 1 × 20 k um etwa ein Drittel schneller; für 32 × 2 k kleiner
  Effekt (dort überleben mehr Tile-Paare). Ob das für ≥ 1× OpenMM reicht, hängt auch an den
  Constraint-Lösern (Punkt 3 im Kontext).
- Speicher: n_replicas × n_tile_pairs × 4 Byte (1 × 20 k: 0.7 MB; 128 × 1 k: 0.15 MB).
- Abnahmetests: ROADMAP Phase 9.6.
