# ADR 0016: Selbstkonsistente Kopplung MD ↔ FEM (optische Kräfte rückgekoppelt)

- **Status:** Accepted (Projektinhaber 2026-11-06, Kriterien bestätigt; Solver hp-FEM, OQ-20)
- **Datum:** 2026-11-06
- **Bezug:** OQ-01/02 (α′, Atomvolumina, Kalibrierung offen), OQ-04 (Normierung, Überschusspolarisierbarkeit,
  Lokalfeldfaktor offen), PHYSICS_MODEL §4.3 (`VoxelGridForce3D`) und §6.3 (`RefractiveIndexProfiler`),
  ADR 0005 (Topologie je Replica), ADR 0011 (Replica Exchange). Reihenfolge: Projektinhaber 2026-11-06
  (MD ↔ FEM vor Coarse-Grained).

## Kontext
Heute läuft die Kopplung nur in eine Richtung:
- MD → Optik: Der `RefractiveIndexProfiler` liefert n(z) in Schichten. `analysis.refractive_index_layers`
  macht daraus einen Schichtstapel für Transfer-Matrix-Solver.
- Optik → MD: `VoxelGridForce3D` liest ein festes Gitter W = |E₀|² (Spitzenamplitude, (V/nm)²) samt
  Gradient. `voxel.optical_scale` setzt U_i = −¼ α′_eff,i |E₀|² mit dem Polarisierbarkeitskontrast
  gegenüber Wasser (OQ-04).

Was fehlt, ist die Rückkopplung. Die adsorbierten Moleküle ändern die lokale Permittivität und damit
das Nahfeld; genau das ist das Sensorsignal. Das geänderte Feld ändert wiederum die Kräfte auf die
Moleküle. Ein selbstkonsistenter Zustand ist ein Fixpunkt: Die Verteilung ⟨ε(r)⟩ aus der MD erzeugt
ein Feld W(r), und unter diesem Feld reproduziert die MD dieselbe Verteilung.

**Größenordnung (entscheidet über den Nutzen, OQ-04):**
- In Wasser bei 1 MW/cm² gilt |E₀| ≈ 2.4·10⁻³ V/nm. Mit α′ ≈ 10⁻³ nm³ ergibt das U ≈ 10⁻⁷ kJ/mol je
  Atom.
- Für ein Aptamer mit ≈ 10³ Atomen und Intensitätsverstärkung G = |E/E_in|² ist U_mol ≈ 10⁻⁴ · G
  kJ/mol (bei 1 MW/cm²; linear in der Eingangsintensität).
- Das ist eine obere Abschätzung: alle Atome im selben Feld, gleiches Vorzeichen von α′_eff.
- Bei 1 MW/cm² gilt damit:

  | Verstärkung G | U_mol |
  |---|---|
  | 10³ | ≈ 0.04 k_BT |
  | 10⁴ | ≈ 0.4 k_BT |
  | 10⁵–10⁶ (plasmonische Hotspots) | ≈ 4–40 k_BT |

  Bei höherer Eingangsintensität wird die Kopplung entsprechend früher relevant.
- Folgerung: Die Rückkopplung lohnt sich nur bei starken Feldern bzw. Hotspots. Für schwache Felder
  genügt die einseitige Kopplung.
- Die Rückwirkung der Moleküle auf das Feld (δW/W) ist dagegen auch bei schwachen Feldern messbar,
  denn sie ist das Sensorsignal. Die Kraft ist allerdings dann bedeutungslos.

Randbedingungen:
- Der FEM-Solver ist extern: **hp-FEM** (https://github.com/PhotonPower/hp-FEM, Projektinhaber
  2026-11-06, OQ-20). Eigenschaften von hp-FEM, die die Schnittstelle bestimmen:
  - Python-Paket mit SI-Einheiten und Zeitkonvention e^{−iωt}; Felder sind komplexe
    Spitzenamplituden, genau wie OQ-04.
  - Material je Zelle über `MaterialMap.set_cell`, Zellschwerpunkte über `mesh.cell_centroids()`.
  - Feldwerte an Punkten über `problem.sample(solution, locator, points)`; Schichtstapel als
    Hintergrund (`LayerStack3D`).
- Das optische Feld folgt der Konfiguration instantan. Ein FEM-Lauf je MD-Schritt ist aber unmöglich
  (Sekunden bis Minuten je Lösung gegenüber ~1 ms je MD-Schritt).

## Entscheidung (Vorschlag)
1. **Mean-Field-Fixpunktiteration (Picard mit Dämpfung).** Iteration k:
   1. MD-Segment der Länge τ unter dem Feld W_k (Voxelgitter); die ersten τ_eq werden verworfen.
   2. ⟨ε⟩_k aus dem Rest des Segments, gemittelt über Frames und über die Replicas im Zielzustand.
   3. Externer Solver: W_new = solve(⟨ε⟩_k).
   4. Mischen: W_{k+1} = (1 − β) W_k + β W_new mit β ∈ (0, 1], Default 0.5.

   Abbruch, wenn sowohl die relative Änderung von ⟨ε⟩ als auch die von W (L2 über das Gitter) zwei
   Iterationen in Folge unter einer Toleranz liegen. Die Toleranz liegt über dem statistischen Fehler
   der Mittel (Blockmittel). Eine Toleranz unter dem Rauschen wäre nicht erreichbar.

   *Annahme (OQ-21):* Mean-Field ist zulässig, wenn die Feldänderung durch die Fluktuation einzelner
   Moleküle klein gegen die durch ihr Mittel ist (verdünnte Schicht, schwache Kopplung). Das wird
   geprüft, indem man den Solver einmal mit Einzelframes statt mit dem Mittel aufruft und die Streuung
   von W berichtet.
2. **3D-Permittivitätskarte als neuer Observer `PolarizabilityGridObserver`.**
   - Er ist das 3D-Gegenstück zum `RefractiveIndexProfiler`: Fixed-Point-Histogramme S_α und S_V je
     Voxel, deterministisch, mit Summe und Quadratsumme über Frames.
   - ε = n² je Voxel nach derselben Lorentz-Lorenz-Mischung mit Wasser-Hintergrund (§6.3).
   - Atome werden gaußförmig verschmiert (σ = R_i/√5, separabel je Achse, Fenster ±4σ wie im
     1D-Profiler, damit die x/y-Summe dessen Gauß-Profil exakt reproduziert; umgesetzt 2026-11-06
     statt der ursprünglich vorgeschlagenen 3σ). Ein exakter
     Kugel-Voxel-Schnitt in 3D wäre teuer.
   - Das Gitter ist unabhängig vom Kraftgitter. Voxelgröße ≥ 0.2 nm wird empfohlen (kleinere Voxel
     verstärken den Bias durch max(0, ·), siehe OQ-02).
   - Der 1D-Profiler bleibt für Schichtgeometrien.
3. **`VoxelGridForce3D::set_values` zur Laufzeit.**
   - Gleiche Shape, Herkunft und Abstände; neue Werte werden in das bestehende Array hochgeladen,
     stream-geordnet und ohne Neuinitialisierung.
   - `forces_valid` wird zurückgesetzt. Die Energie ist nur innerhalb eines Segments erhalten.
4. **Solver als Python-Protokoll** `solve(eps_grid: EpsGrid) -> FieldGrid` (`biooptomd.coupling`;
   umgesetzt als `solve(eps: EpsGrid, grid: ForceGrid) -> W` (nz, ny, nx): das Kraftgitter wird
   übergeben statt im Ergebnis zurückgegeben, damit der Treiber die Geometrie festlegt):
   - Eingang: ε auf dem Beobachtergitter mit Herkunft, Abständen und Wellenlänge.
   - Ausgang: W = |E₀|² (Spitzenamplitude) auf dem Kraftgitter; der Gradient wird in BioOptoMD
     berechnet (`values_from_potential`).
   - Mitgeliefert werden zwei Adapter:
     - (a) **TMM-Referenzsolver** (NumPy, 1D, geschichtetes Medium, p/s-Polarisation, z. B. Kretschmann/SPR
       über Gold). Er ist exakt für planare Schichten, die Referenz der Tests und nützlich für
       planare Sensoren. Er dient auch als Gegenprobe zu hp-FEM: Im planaren Fall muss hp-FEM mit
       `LayerStack3D` dasselbe W liefern.
     - (b) **hp-FEM-Adapter `HpFemSolver`** (OQ-20), läuft im selben Python-Prozess. Der Nutzer liefert
       eine Fabrik für Netz und Setup seiner Geometrie (Materialien, Anregung, PML bzw. Schichtstapel)
       sowie die Zellen des Molekülbereichs.
       - Je Zelle im Molekülbereich wird ε = ⟨ε⟩ am Schwerpunkt (trilinear aus dem Beobachtergitter)
         über `set_cell` gesetzt. Die Zellen dort sollen nicht gröber als die Merkmale von ⟨ε⟩ sein.
       - Dann lösen und |E|² an den Punkten des Kraftgitters über `problem.sample` auswerten.
       - Umrechnung: nm ↔ m (Faktor 10⁻⁹), W[(V/nm)²] = Σ_c |E_c[V/m]|² · 10⁻¹⁸.
       - Die Eingangsamplitude folgt aus der Intensität: I = ½ n c ε₀ |E₀|².
     - (c) **Datei-Adapter** (npz auf regulärem Gitter) als Rückfall, falls hp-FEM nicht im selben
       Interpreter läuft. *Geprüft 2026-11-06:* Das Release-Wheel hp-FEM 0.4.0 (cp311-win_amd64, für
       python.org-CPython) importiert zusammen mit BioOptoMD im selben Interpreter, mit
       `Scattering3D`, `MaterialMap.set_cell` und `LayerStack3D`. Der Rückfall ist damit unter
       Windows nicht nötig, bleibt aber für entfernte Rechner.
5. **Treiber `SelfConsistentOptics`.**
   - Hält Iterationszähler, Mischparameter und Verlauf (ε- und W-Änderung, Observablen je Iteration).
   - Ein Checkpoint je Iteration (Kontext über `io.save_checkpoint` plus W, ⟨ε⟩, Verlauf), damit
     teure FEM-Iterationen fortsetzbar sind.
   - Bei Replica Exchange werden nur die Replicas im Zielzustand (T, κ) gemittelt, nicht die ganze
     Leiter. Das Feld gilt für alle Replicas gleich (ein Gitter je Kontext).
6. **Grenzen, unverändert aus OQ-04:**
   - Nur die konservative Gradientenkraft (Re α); keine Streu- oder Absorptionskraft (Im α).
   - Keine Erwärmung in Hotspots; Fluorophore im Optikmodus ausgeschlossen.
7. **Voraussetzungen vor der Kopplung (Phase 10.0):**
   - (a) `test_voxel_optical_sphere_consistency` (Lokalfeldfaktor, offen seit Phase 6). *Umgesetzt
     2026-11-06*; Korrektur nach OQ-22 (b): Lorentz-Faktor ((n_w² + 2)/3)² in `optical_scale`.
   - (b) Kalibrierung von α′ und v an dn/dc (`test_refractive_index_dn_dc`, OQ-01/02). Eigene
     Messwerte gibt es noch nicht (Projektinhaber 2026-11-06). Kalibriert wird an Literaturwerten,
     deren Unsicherheit als Toleranz dient: Proteine 0.190 ± 0.003 mL/g bei 589 nm (Zhao et al.
     2011), DNA ≈ 0.17 mL/g bei 633 nm (Handbuchwert, Salzabhängigkeit offen). Eigene Messungen
     ersetzen die Werte später. *Umgesetzt 2026-11-06* als klassenweise Skalierung
     (`builder.calibrate_optics`). Annahmen für DNA (OQ-23): Bezugsmasse Na-Salz, Toleranz ± 0.005,
     v̄ = 0.58 mL/g (NucProt, nicht verifiziert).

   Ohne beides ist die Rückkopplung quantitativ nicht belastbar. Kräfte und ε hängen direkt an α′_eff.

## Alternativen
- **Feld je N MD-Schritte aus der Momentankonfiguration neu lösen („on the fly“).** Physikalisch
  näher an der instantanen Antwort, aber ein FEM-Lauf je wenige ps ist zu teuer, und Einzelframes
  sind verrauscht. Verworfen; bleibt als Prüfung der Mean-Field-Annahme (Punkt 1, OQ-21).
- **Eigener FEM-/FDTD-Solver in BioOptoMD.** Großer Aufwand und außerhalb des Kerns. Die Nutzer haben
  validierte Solver für ihre Strukturen. Verworfen; nur der 1D-TMM als Referenz.
- **Rückkopplung analytisch über die Clausius-Mossotti-Polarisation der Schicht (ohne Solver).** Nur
  für einfache Geometrien. Deckt der TMM-Adapter für planare Fälle ab.
- **Anderson-/Broyden-Beschleunigung statt linearer Mischung.** Bei schwacher Kopplung konvergiert
  Picard in wenigen Iterationen, und das MD-Rauschen stört Quasi-Newton-Verfahren. Zunächst
  verworfen, nachrüstbar am Treiber.
- **ε pro Replica statt Mittel.** Verletzt die Annahme eines gemeinsamen Feldes (ein Gitter je
  Kontext). Bei Bedarf über mehrere Kontexte lösbar.

## Konsequenzen
- Neue Bausteine:
  - C++/CUDA: `PolarizabilityGridObserver`, `VoxelGridForce3D::set_values`.
  - Python: `biooptomd.coupling` (Protokoll, TMM-Referenz, Datei-Adapter, Treiber).
  - Keine Änderung an bestehenden Kraft- oder Integratorpfaden.
- Die Ergebnisse sind nur so gut wie α′ und v (OQ-01/02) und der Lokalfeldfaktor (OQ-04). Daher
  kommen die Voraussetzungen zuerst (Punkt 7).
- Über ein ganzes Segment ist die Energie nicht erhalten (Feldsprung zwischen Segmenten); NVE-Tests
  gelten nur innerhalb eines Segments.
- Abnahmetests (Vorschlag, ROADMAP Phase 10):
  - **Observer:**
    - `test_polarizability_grid_vs_numpy`: Histogramme und ε gegen eine NumPy-Referenz (rel. 1e-6).
    - Die Summe über x/y reproduziert S_α(z) und S_V(z) des 1D-Profilers mit Gauß-Verschmierung
      (rel. 1e-5).
    - Deterministisch (zwei Läufe bitgleich).
  - **Gitter:** `test_voxel_set_values`. `set_values` ergibt bitgleich dieselben Kräfte wie ein neu
    angelegtes Gitter mit denselben Werten; eine falsche Shape gibt einen Fehler.
  - **TMM-Referenz:** `test_tmm_reference`.
    - Fresnel-Koeffizienten einer Grenzfläche analytisch (rel. 1e-12).
    - SPR-Minimum für Gold/Wasser bei 633 nm am Literaturwinkel (Toleranz 0.5°, Literaturwert per
      Websuche verifizieren).
    - Energieerhaltung |r|² + |t|²·(Faktor) = 1 ohne Absorption.
  - **Fixpunkt:** `test_coupling_fixed_point`. Planare Schicht über Gold mit dem TMM-Adapter.
    - Konvergenz in ≤ 10 Iterationen.
    - Das Ergebnis hängt nicht von β ab (β = 0.3 und 1.0 innerhalb des statistischen Fehlers).
    - Bei Feldstärke 0 ist es identisch mit der ungekoppelten MD (bitgleich).
  - **Größenordnung:** `test_coupling_scaling`. Die Verschiebung von ⟨z⟩ einer Testschicht skaliert
    im schwachen Feld linear mit der Intensität.
  - **hp-FEM-Adapter:** `test_hpfem_adapter_planar` (`reference`, nur wenn hp-FEM installiert ist).
    Planare Schicht über Gold: W aus hp-FEM gegen den TMM-Referenzsolver, rel. < 1 % nach
    hp-Verfeinerung; Einheiten und Amplitudenkonvention über den Fall ohne Schicht (Fresnel).
  - **Fortsetzen:** `test_coupling_resume`. Abbruch und Fortsetzen nach Iteration k liefern dasselbe
    Ergebnis wie ein durchgehender Lauf (bitgleich bei gleichem Seed).
