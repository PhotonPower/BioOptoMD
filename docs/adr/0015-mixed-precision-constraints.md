# ADR 0015: Constraint-Löser mit gemischter Präzision (iterative Verfeinerung)

- **Status:** Rejected (Projektinhaber nahm den Vorschlag 2026-11-06 an („ok, weiter“); der Prototyp
  verfehlte das Kriterium von Punkt 4, daher zurückgezogen 2026-11-06)
- **Datum:** 2026-11-06
- **Bezug:** ADR 0009 (gemischte Präzision des Zustands), Phase 5 (SHAKE/RATTLE, Toleranzen), Phase 8
  (M-SHAKE, direktes RATTLE), ADR 0014 / OQ-19 (Phase 9.6, Kriterium 1 × 20 k ≥ 1.25×)

## Kontext
Bei 1 × 20 k Atomen (HMR, H-Constraints, 4 fs) verbrauchen SHAKE und RATTLE etwa 27 % der GPU-Zeit je
Schritt (Nsight Systems, GPU geteilt). Beide rechnen vollständig in FP64 mit einem Thread je Cluster
(Schweratom + bis zu 3 H). Die RTX 3090 hat 1/64 der FP32-Rate in FP64; RTX 6000 Ada ebenso.
Zwei bitidentische Maßnahmen (Blockgröße nach Auslastung, Bildverschiebung einmal je SHAKE-Aufruf)
brachten zusammen nur ≈ +5 % (1 × 20 k) – die Löser sind durch FP64-Arithmetik begrenzt, nicht
durch Parallelität.

Die Toleranzen bleiben unverändert (AGENTS.md): SHAKE |r_k|² − d_k² ≤ 2·10⁻⁶ d_k², RATTLE
|r_k·Δv_k| < 10⁻¹⁰ r_k² [1/ps]. Letzteres liegt relativ bei ≈ 10⁻¹¹ und ist in FP32 nicht erreichbar.

## Entscheidung (Vorschlag)
1. **Iterative Verfeinerung mit gemischter Präzision:** Residuen (f_k bzw. r_k·Δv_k), Konvergenzprüfung
   und Aktualisierung von Positionen und Geschwindigkeiten bleiben FP64. Jacobi-Matrix bzw.
   Gram-Matrix, deren Faktorisierung und der Löseschritt (3 × 3) laufen in FP32. Die
   Fixpunktiteration x ← x − J₃₂⁻¹ f₆₄(x) konvergiert linear mit Rate ≈ κ(J)·ε₃₂ ≈ 10⁻⁶ je
   Schritt bis zur FP64-Genauigkeit des Residuums; erwartet 1–2 Iterationen mehr als heute.
2. **Gleiche Toleranzen, gleiche Statusbits;** max_iterations bleibt 200.
3. **Vergleichspfad:** `precision = "double"` (heutiger Kernel) bleibt wählbar, Default nach Messung.
4. **Nur übernehmen, wenn messbar:** Ein Prototyp misst zuerst die Kernelzeiten. Bringt er bei
   1 × 20 k weniger als 30 % kürzere Constraint-Zeit, wird der ADR zurückgezogen (Rejected).

## Alternativen
- **Alles in FP32:** braucht eine gelockerte RATTLE-Toleranz (10⁻¹⁰ ist in FP32 unerreichbar) –
  verworfen (AGENTS.md: Toleranzen nicht lockern).
- **Analytisches SETTLE-artiges Verfahren für Cluster:** gibt es geschlossen nur für starre
  Dreiecke (Wasser); für CH₃/NH₃ mit flexiblem Schweratom nicht direkt anwendbar.
- **Mehrere Threads je Cluster:** die Cluster sind klein (≤ 3 Constraints); der Aufwand liegt in
  der FP64-Kette, nicht in fehlender Parallelität (Blockgrößen-Messung).

## Konsequenzen
- Ergebnisse sind nicht mehr bitidentisch zum FP64-Pfad, aber innerhalb der unveränderten
  Toleranzen; Determinismus bleibt (gleiche Operationen je Lauf).
- Abnahmetests (Vorschlag, ROADMAP 9.6 bzw. 9.7):
  - alle bestehenden Constraint-Tests aus Phase 5 und 8 unverändert grün (Toleranzen, NVE-Drift mit
    Constraints, Verteilungstests);
  - neu: Abweichung mixed vs. double nach einem SHAKE/RATTLE-Aufruf ≤ Toleranz (Positionen
    relativ ≤ 10⁻⁶ d, Geschwindigkeiten im Rahmen von 10⁻¹⁰ r²); Iterationszahl ≤ 2 mehr als
    double;
  - Messung: Constraint-Kernelzeit 1 × 20 k mindestens 30 % kürzer, sonst Rückzug (Punkt 4).

## Ergebnis des Prototyps (2026-11-06)
Prototyp nach Punkt 1–3 (Template `kMixed` in `k_shake`/`k_rattle`, FP32-Löser mit Pivotschwelle
1e-6 und FP64-Rückfall; `set_constraints(..., precision=...)`), lokaler Branch
`proto/adr-0015-mixed-constraints` (Commit 24d4a81, nicht übernommen). Nsight Systems, 1 × 20 k,
1000 Schritte, GPU geteilt (Hairpin-Lauf parallel), Kernelzeiten Median / Minimum [µs]:

| Kernel | double | mixed |
|---|---|---|
| SHAKE | 28.8 / 14.3 | 22.9 / 15.6 |
| RATTLE | 15.9 / 9.6 | 21.0 / 11.0 |
| Summe (Median) | 44.7 | 43.9 |

Durchsatz (abwechselnd gemessen, Median aus 5): 1 × 20 k 0.99×, 32 × 2 k 0.98×.

- Die Constraint-Zeit sinkt höchstens um ≈ 2 % (Median) und steigt im Minimum sogar. Das liegt weit
  unter den geforderten 30 %, daher wird der ADR nach Punkt 4 zurückgezogen.
- RATTLE wird langsamer. Die zusätzliche Verfeinerungsiteration (FP64-Residuum und -Aktualisierung)
  kostet mehr, als die FP32-Lösung spart.
- Bei SHAKE spart die FP32-Jacobi-Matrix im Median etwa 20 %, im Minimum nichts. Auch „nur SHAKE
  mixed“ bliebe mit ≈ 13 % unter dem Kriterium.
- Folgerung: Die Kernel sind nicht durch die FP64-Arithmetik der 3×3-Lösungen begrenzt, sondern
  durch Laden/Speichern (Vec4d-Geschwindigkeiten, Mixed-Precision-Positionen), Latenz bei wenigen
  Threads (≈ 7 k Cluster) und den FP64-Anteil, der ohnehin FP64 bleiben muss (Residuen,
  Aktualisierung). Die Kontext-Annahme „durch FP64-Arithmetik begrenzt“ war in dieser Form falsch.
