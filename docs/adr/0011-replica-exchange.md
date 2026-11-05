# ADR 0011: Temperatur-Replica-Exchange mit zustandsabhängiger Abschirmung

- **Status:** Proposed
- **Datum:** 2026-11-05
- **Bezug:** ROADMAP Phase 9.2, OQ-10 (κ_r, ε_r(T_r)), ADR 0006 (r_c aus κ), ADR 0010 (GB)

## Kontext
Phase 9.2 (Reihenfolge vom Projektinhaber am 2026-11-05 festgelegt) verlangt Temperatur-Austausch
auf dem Host. Nach OQ-10 hängen κ_r und ε_r von T_r ab, Gouy-Chapman zusätzlich über γ(T_r). Die
potentielle Energie ist damit zustandsabhängig; das übliche Kriterium exp[(β_i − β_j)(U_i − U_j)]
verletzt dann die detaillierte Bilanz.

## Entscheidung
1. **Zustände statt Koordinaten tauschen.** Slot r trägt Zustand k = state[r] mit (T_k, κ_k, ε_k).
   Ein Tausch setzt T, κ, ε der Slots neu (`set_temperatures`, neues `set_screening` an alle Plugins)
   und skaliert die Geschwindigkeiten mit sqrt(T_neu/T_alt) (`scale_velocities`). Positionen,
   Constraints und Philox-Zähler (Slot-Index) bleiben.
2. **Hamiltonian-Kriterium.** Δ = β_j U_j(x_a) + β_i U_i(x_b) − β_i U_i(x_a) − β_j U_j(x_b),
   p = min(1, e^{−Δ}). Die Kreuzenergien liefert eine Kraftberechnung mit allen Versuchspaaren
   gleichzeitig vertauscht; abgelehnte Paare werden zurückgesetzt (zweite Kraftberechnung nur dann).
3. **Paarschema** Nachbarn der Leiter mit abwechselnder Parität; Zufallszahlen auf dem Host
   (numpy PCG64, Seed) ⇒ deterministisch, Zustand im Checkpoint (`state_dict`).
4. **Cutoff** bleibt fest: r_c = n_κ/min_k κ_k über alle Zustände (ADR 0006) beim Aufbau.
5. **GB:** 1-4-Coulomb bleibt unabgeschirmt (`follow_screening = False`), Nonbonded erhält κ_k und ε_out,k.

## Alternativen
- **Koordinaten tauschen:** Host↔Device-Transfer aller Positionen und Geschwindigkeiten je Versuch,
  verletzt AGENTS.md Regel 12 im Geist; verworfen.
- **Reines Temperaturkriterium (κ, ε fest auf einem Wert):** widerspricht OQ-10; verworfen.

## Konsequenzen
- Neuer Plugin-Hook `set_screening`; Plugins mit κ/ε-Abhängigkeit müssen ihn implementieren.
- Je Austauschversuch 1–2 zusätzliche Kraftberechnungen (bei Austausch alle ~0.5–1 ps vernachlässigbar).
- Abnahmetests: ROADMAP Phase 9.2.
