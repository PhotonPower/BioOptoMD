# ADR 0015: Constraint-Löser mit gemischter Präzision (iterative Verfeinerung)

- **Status:** Proposed
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
