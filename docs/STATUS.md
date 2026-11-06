# Projektstatus

> Diese Datei wird am Ende **jeder** Arbeitssitzung aktualisiert (AGENTS.md §2.4).

**Aktive Phase:** 10 – MD ↔ FEM-Kopplung (Branch `phase-10/md-fem`, lokal), 10.0 umgesetzt; offen 9.4 (Hairpin-Lauf (b)) und 9.6 (Schlussmessung). Phasen 9.5 und 9.7 abgenommen 2026-11-06.
Alle PRs #1–#6, #9–#14 am 2026-11-05 nach `main` gemergt (Merge-Commits, Reihenfolge des Stapels).
**Phase 9.3 (GB-neck2) abgenommen** am 2026-11-05, ADR 0012 und Abnahmekriterien bestätigt.
**Phase 9.2 (Replica Exchange) abgenommen** am 2026-11-05, ADR 0011 und Abnahmekriterien bestätigt.
**Phase 8 abgenommen** am 2026-11-05 durch den Projektinhaber (RTX 3090, PR #11).
**Phase 7 abgenommen** am 2026-11-05 durch den Projektinhaber (RTX 3090).
**Nachrüstung (PR #9) abgenommen** am 2026-11-05 durch den Projektinhaber (RTX 3090).
**Phase 6 abgenommen** am 2026-11-04 durch den Projektinhaber (RTX 3090), vor den neuen Abnahmetests.
**Referenz-Hardware:** RTX 6000 Ada 48 GB (CC 8.9), Host 1 TB RAM

## Abgeschlossen
- Phase 0: Repository-Gerüst, Dokumentation, Agenten-Anweisungen.
- Phase 1 (PR #1): Build, `DeviceState`, `SimulationContext`, Plugin-Interfaces, Kern-Kernels, nanobind-Modul.
- Phase 2 (PR #2): Langevin-BAOAB, `PlanarSurfaceForce`, `HarmonicRestraintPullingForce`, `EnergyTracker`,
  ADR 0008.
- Phase 3 (PR #3): `BondedForce`, präzises Minimum Image, ADR 0009 (Mixed Precision), NVE-Protokoll 100 ps.
  Alle drei vom Projektinhaber am 2026-11-03 auf RTX 3090 abgenommen.

- Phase 4 (PR #4): Tiled All-Pairs (LJ + Debye-Hückel, Switching), Builder `biooptomd.nonbonded`.
  Abgenommen 2026-11-03.

- Phase 5 (PR #5): HMR, SHAKE/RATTLE auf H-Bindungen, N_dof. Abgenommen 2026-11-03; Nacharbeit nach
  Review siehe unten.

## Phase 10 – MD ↔ FEM-Kopplung: ADR 0016 (Accepted 2026-11-06), Branch `phase-10/md-fem`
- ADR 0016: Mean-Field-Fixpunktiteration mit Dämpfung, 3D-ε-Observer, `VoxelGridForce3D::set_values`,
  Solver als Python-Protokoll (TMM-Referenz, hp-FEM-Adapter im selben Prozess, npz-Rückfall);
  Voraussetzungen Lokalfeldtest und dn/dc-Kalibrierung an Literaturwerten (Phase 10.0).
- Phase 10.0 (a): `test_phase6.py::test_voxel_optical_sphere_consistency` grün. Der Profiler bildet
  α′, v auf n_p ab; die GPU-Kraft ist die additive Summe (rel. 1e-5); das Verhältnis CM/additiv ist
  analytisch 1.638 bei n_p = 1.5 in Wasser und exakt 1 in Vakuum. OQ-22 (b): `optical_scale`
  multipliziert per Default mit ((n_w² + 2)/3)² (`local_field=False` für das additive Modell).
- Phase 10.3 (Teil): `biooptomd.coupling` mit TMM-Referenz und Treiber `SelfConsistentOptics`. Tests:
  TMM gegen Fresnel und Airy (rel. 1e-12), R + T = 1, evaneszenter Abfall, SPR-Minimum 72.24° gegen
  Plasmonen-Dispersion 72.75° (OQ-25: Literaturwinkel nicht konsistent belegt); Fixpunkt konvergiert
  in 6 Iterationen (β = 0.5), W-Profile für β = 0.3/0.5/1.0 innerhalb 0.5 %, Feld 0 bitgleich zur
  ungekoppelten MD; ⟨z⟩-Verschiebung linear (Verhältnis 2.0000) und = F/k; Fortsetzen bitgleich.
  Befund OQ-24 (Selbstfeld dichter Schichten bei p-Polarisation).
  Review 10.3 umgesetzt: stabile TMM-Rückwärtsrekursion (dicke Metallschichten), p-Feld gegen
  Fresnel-t_E, Rauschen aus Blockmitteln (ADR Punkt 1), NaN-/W-Prüfung, atomare Checkpoints,
  Konstanten C_LIGHT/EPS0_SI in units, `test_coupling_fixed_point` als slow; Treiber noch ohne REMD.
- Phase 10.3 abgeschlossen: `HpFemGratingSolver` (hp-FEM 0.4.0 im selben Interpreter; Kretschmann mit
  Schicht: max. Abweichung zu TMM p 1.0e-3, s 1.7e-7; ohne Schicht 1e-6) und `NpzFileSolver`
  (externer Prozess bitgleich). hp-FEM ist nicht in der venv installiert; der Test läuft mit
  PYTHONPATH auf das Wheel und wird sonst übersprungen. Offen: `HpFemSolver` (hp-FEM ohne
  gmsh: Einheitszellennetz selbst bauen), npz-Adapter, Beispiel 10.4.
- Phase 10.1/10.2: `PolarizabilityGridObserver` (`ctx.add_polarizability_grid`) und
  `VoxelGridForce3D.set_values`. `test_phase10.py` (7 Tests) grün: Histogramme/Varianz/ε gegen NumPy
  (rel. 1e-6, periodisch und nicht, Gauß und Punkt), x/y-Summe = 1D-Profiler (rel. 1e-5),
  deterministisch, `set_values` bitgleich zu neu angelegtem Gitter (Kräfte, Energien, Trajektorie).
  Review umgesetzt (int-Grenze, Fensterreserve, Box-Prüfung nur bei Änderung, Padding-/Typ-Test).
  Gauß-Fenster ±4σ statt 3σ vom Projektinhaber bestätigt (2026-11-06).
- Phase 10.0 (b): `test_phase6.py::test_refractive_index_dn_dc[protein|dna]` grün (GPU-Profiler,
  c = 10 mg/mL: dn/dc 0.1901 bzw. 0.1701 mL/g). `builder.calibrate_optics`: s_α = 0.983 / 1.002,
  s_v = 0.666 / 0.654; unkalibriert 0.064 / 0.065. DNA bezogen auf das Na-Salz (OQ-23); Ionen mit
  Shannon-Radien, nicht kalibriert. `test_optical_force_dilute_solution`: GPU-Kraft einer kalibrierten
  Lösungsschicht = Kontinuumskraft ¼ ε₀ Δε V ∇W (rel. 1.3e-3) nur mit Lorentz-Faktor. Review-Befunde
  zu 10.0 umgesetzt. Das Beispiel `aptamer_on_graphene` nutzt die
  DNA-Kalibrierung. Grenze: Elementtypen erfassen Aromatizität nicht (Residuen-dn/dc nach McMeekin
  folgt mit ProtOr/NucProt).
- OQ-20 entschieden (hp-FEM). hp-FEM 0.4.0 (Release-Wheel cp311-win_amd64) und BioOptoMD laufen im
  selben Interpreter (geprüft in einem getrennten Verzeichnis, venv unverändert). Offen: OQ-21
  (Mean-Field).

## Phase 9.7 – Folgearbeiten OQ-14 / OQ-18 (d): abgenommen 2026-11-06 (Branch `phase-9/followups`, PR #18)
- C¹-glattes ζ-Clamping in `ImageChargeForce` (h = ζ_c/2): Referenz `smooth_zeta`, neue Tests
  `test_image_smooth_clamp_reference` (Stetigkeit, Kraft = −dU/dz) und `test_image_smooth_clamp_gpu`;
  alle 17 Tests aus 9.5 + neu grün; NVE-Drift 0.7–1.1·10⁻⁵ kJ/mol/ps/Atom (vorher 1–2·10⁻⁵),
  Testgrenze von 1e-4 auf 3e-5 verschärft. NaN-Positionen werden nicht mehr auf ζ_c maskiert,
  sondern pflanzen sich fort (vorher `fmaxf(NaN, ζ_c)` = ζ_c).
- Volle Suite 161/161 grün (vor den Review-Fixes), danach Phase 2, 9.2 (REMD), 9.5 (Image), 9.7: 59/59.
- Zweite Temperatur (OQ-14): E_kin nach dem O-Schritt (+ RATTLE), `ctx.kinetic_energy_mid()` und
  `kinetic_mid` im EnergyTracker. Die Reduktion läuft nur in beobachteten Schritten und im letzten
  Schritt von `step(n)`.
  - Messung 2k-System (HMR, H-Constraints, γ = 1/ps): 300.4 ± 0.3 / 300.2 / 300.2 K bei 4 / 2 / 1 fs;
    am vollen Schritt 268.5 / 292.3 / 298.2 K (Bias ∝ Δt²).
  - Das Kriterium „bei 1 fs stimmen beide überein“ war falsch und ist revidiert (ROADMAP 9.7, vom
    Projektinhaber bestätigt 2026-11-06).
  - `tests/python/test_phase9_temperature.py`: 3 Tests grün.
  - Review (unabhängiger Agent): kein kritischer Befund. Umgesetzt: NaN nach `set_velocities` und
    `restore_state`, Test „am Stück vs. in Stücken“ (Stride-Auswahl), NVE-Grenze 3e-5, Zähler über
    ζ_raw statt dζ < 1, Doku. Nicht umgesetzt: Reduktion im letzten Schritt von `step(n)` lazy machen
    (eine kleine Reduktion je `step()`-Aufruf, vernachlässigbar).

## Phase 9.6 – Tile-Paarliste (ADR 0014 inkl. Rev. 1 Accepted), implementiert; Abnahme später (OQ-19)
- Vorab-Messung 1 × 20 k: räumliche Sortierung bringt nichts (Atomreihenfolge 13.3 % lebende
  Tile-Paare, Morton 14.7 %) ⇒ Tile-Paarliste statt Cell-List (Projektinhaber 2026-11-06).
- Implementiert: Listenaufbau (Thread je Tile-Paar, warpweise Atomics), persistenter Kernel je
  Replica für DH und die drei GB-Passes; Rechnung je Tile-Paar als gemeinsame Gerätefunktion
  `tile_pair`; Auswahl `tile_list = "auto" | "on" | "off"`, Property `ctx.nonbonded_uses_tile_list`.
- `tests/python/test_phase9_tile_list.py`: 15 Tests, alle bitidentisch zu `"off"` (DH, GB-OBC2/GBn2,
  2k × 32 und 20k, per-Replica-Topologien, Grenzfälle, Auto inkl. `set_box`, viele Tile-Paare je Warp);
  Listenlänge geprüft (`ctx.nonbonded_tile_list_counts()`); memcheck/racecheck ohne Befund.
- Review (unabhängiger Agent): kein kritischer Befund; umgesetzt: Tests prüfen, dass die Liste aktiv ist
  und verwirft, Gitterschleife mit vielen Tile-Paaren je Warp (DH, GB), SM-Zahl je Kontext-Gerät,
  Listenpuffer nur bei `tile_list != "off"`, `std::numbers::pi`, Doku (Rauschen, OQ-19).
- Messung bei geteilter GPU (Verhältnisse, vorläufig, Rauschen ≈ ±4 %): 1 × 20 k ≈ 1.14–1.18×, 32 × 2 k 1.01×,
  128 × 1 k 1.00×. Kriterium 1.25× nicht erreicht; i-Tile-Akkumulation gemessen und verworfen
  (0.77× bzw. neutral). Details ADR 0014 Rev. 1. Endmessung bei freier GPU.
- Constraint-Löser: zwei bitidentische Verbesserungen (Blockgröße nach Auslastung 128/64/32,
  Bildverschiebung je Bindung einmal je SHAKE-Aufruf), zusammen ≈ +5 % bei 1 × 20 k und ≈ +3 % bei
  32 × 2 k. ADR 0015 (3×3-Lösungen in FP32) angenommen, prototypisch gemessen und zurückgezogen:
  Constraint-Zeit nur ≈ −2 % statt der geforderten −30 % (ADR 0015, Ergebnis).

## Phase 9.5 – ImageChargeForce: abgenommen 2026-11-06 (Branch `phase-9/image-charge`, PR #16)
- ADR 0013 Accepted (Projektinhaber 2026-11-05, Kriterien und Defaults OQ-18 bestätigt).
- `src/forces/image_charge.{hpp,cu}` (Tile-Pass, Vorauswahl in z und lateral, Bildsumme, Clamping-
  Zähler, `set_screening`), neuer Energie-Term `image`, Binding `add_image_charges`, Builder
  `add_image_charges(ctx, info, "gold")`, Referenz `tests/reference/image_charge_ref.py`.
- `tests/python/test_phase9_image.py`: 15 Tests grün (inkl. NVE: Drift 1–2·10⁻⁵ kJ/mol/ps/Atom, laterale
  Vorauswahl über die periodische Grenze); volle Suite 140/140 (vor Review-Fixes), danach Phase 7 + 9.5 grün;
  memcheck und racecheck ohne Befund.
- Kosten: +4 % (DH) / +7 % (GBn2) je Schritt, Duplex aufrecht auf Gold, 8 Replicas (GPU dabei mit
  dem Hairpin-Lauf geteilt, Verhältnis indikativ).
- Review (unabhängiger Agent): kein kritischer Befund; umgesetzt: Test der lateralen Vorauswahl,
  Padding-Prüfung, periodische Energieprüfung der Referenz, schärferer Stetigkeitstest, ζ_c als
  ASSUMPTION(OQ-18) mit Untergrenze 0.02 nm, `add_anchors(z0_nm=…)`, Doku (ADR 0013 Punkte 6/7/9 –
  Präzisierungen, vom Projektinhaber am 2026-11-06 bestätigt).
- Beobachtung: H-Atome erreichen an Gold ζ < ζ_c = 0.1 nm (σ_iw(H) ≈ 0.19 nm), Clamping ist dort
  der Normalfall – passt zu OQ-18 (Lage der Bildebene).

## Phase 9.4 – Validierung GBn2 + OL21 (OQ-15), in Arbeit
- OL21 ist in OpenMM 8.6 vorhanden (`amber19/DNA.OL21.xml`), kompatibel mit `implicit/gbn2.xml` und
  `obc2.xml` (frühere Notiz „OL21 nicht verfügbar“ war falsch).
- (a) `examples/gb_validation/duplex_stability.py`: 1BNA, 4 × 100 ns je Modell, HMR/4 fs, 0.15 M.
  **Bestanden:** GBn2 RMSD 0.215 nm (alle) / 0.150 nm (innen), 94 % Paare intakt; OBC2 0.361 / 0.197 nm,
  84 % (Endpaare dauerhaft offen). Tabelle und Einschränkungen in OPEN_QUESTIONS §OQ-15.
- (b) Hairpin-Faltung, Projektinhaber 2026-11-05: d(GCGAAGC), PDB 1PQT (NMR, sequenzidentisch;
  per PDB geprüft). `examples/gb_validation/hairpin_folding.py`: 4 unabhängige REMD-Leitern × 8 T
  (300–370 K), 2 aus NMR-, 2 aus gestreckter Struktur, 1 µs je Replica (≈ 22 h), gestartet 2026-11-05.
  Dafür neu: `ReplicaExchange(ladders=L)` (Test `test_remd_multiple_ladders`).

## Phase 9.3 – GB-neck2: abgenommen 2026-11-05 (Branch `phase-9/gbn2`)
- ADR 0012 (Accepted 2026-11-05): Modell wie OpenMM `implicit/gbn2.xml`, Parameter und interpolierte Neck-Tabellen
  aus der CustomGBForce (asymmetrisch, Ausrichtung geprüft), Neck-Stufe wie AMBER.
- `gb_ref`: Modell-dict (`obc2_model`, `gbn2_model`), Neck-Term mit Ableitung; vs. OpenMM 2·10⁻⁷.
- Kernel: α/β/γ je Atom, Offset als Parameter, Neck-Term in Born-Integralen und Kettenregel; OBC2 als
  Spezialfall (alle OBC2-Tests unverändert grün). Bindings `set_nonbonded_gb(..., offset,
  alpha_beta_gamma, neck_index, neck_m0, neck_d0)`; Builder `solvent="gb-gbn2"`.
- Tests `test_phase9_gbn2.py` 10/10 grün (GPU vs. Referenz: Born-Radien 1.7e-6, Energie 5e-8, Kräfte
  3e-6 RMS); NVE-Drift ≈ 5e-5 (wie OBC2). pytest gesamt 124/124; memcheck/racecheck sauber.
- Review-Agent: Physik korrekt; umgesetzt: OBC2 wieder mit double-Konstanten (bitgleich zu Phase 9.1),
  Formprüfungen in `set_nonbonded_gb` (abg (n, 3), quadratische Neck-Tabellen gleicher Shape, Integer-
  Index, endliche Werte), Neck nur mit α/β/γ, Offset global dokumentiert, Builder akzeptiert
  handgebaute OBC2-Specs ohne `gb_model`, FD-Schutz am Neck-Cutoff über alle Bilder, Test mit mehreren
  Topologien (verschiedene U, ACE aus) und Negativtests.

## Phase 9.2 – Replica Exchange: abgenommen 2026-11-05 (Branch `phase-9/remd`)
- ADR 0011 (Accepted 2026-11-05): Zustände (T, κ, ε) statt Koordinaten tauschen, Hamiltonian-Kriterium mit
  Kreuzenergien (κ(T), ε(T), γ(T) nach OQ-10), Geschwindigkeits-Skalierung, fester r_c der Leiter.
- C++: Plugin-Hook `set_screening` (Bonded mit `follow_screening`, PlanarSurface, TiledAllPairs/GB),
  `SimulationContext::set_screening`, `scale_velocities`; Bindings.
- `biooptomd.remd.ReplicaExchange` (attempt/run, Akzeptanzstatistik, state_dict für Checkpoints).
- Tests `test_phase9_remd.py` 5/5 grün: Entscheidungen = unabhängige Metropolis-Rechnung mit frischen
  Kontexten, Parameter/Energie/Geschwindigkeiten nach Tausch, set_screening (DH/GB/Gouy-Chapman),
  Sampling harmonischer Atome (⟨U⟩ und Akzeptanzraten vs. Gamma-Erwartung, 4σ), Checkpoint bitgenau.
  pytest gesamt 114/114; memcheck 0 Fehler.
- Hinweis: ε_r(T) von Wasser gilt bis 373 K – Leitern darüber brauchen eine andere Fitformel (OQ-10).
- Review-Agent: Kriterium korrekt; umgesetzt: REMD-Zustand im npz-Checkpoint (RNG als JSON-Bytes,
  `save/load_checkpoint(..., remd=…)`), Prüfung Temperaturen ↔ Austauschzustand und Frische der
  Energien in `attempt()`, Warnung bei zu kleinem r_c, Sampling-Test mit 1 ps Abstand und
  Blockmittelung, Nachweis des Hamiltonian-Anteils (|Δ_H − Δ_T| > 0.1), nicht zirkulärer
  set_screening-Vergleich (`build_context(kappa=…)`), `set_temperatures` invalidiert Kräfte,
  OQ-17 (Φ0(T) bei fester Oberflächenladung).

## Phase 9.1 – GB-OBC2: abgenommen 2026-11-05 (PR #12)
- ADR 0010 (Accepted 2026-11-05): Modell, Kopplung an Cutoff/Switching/Bildsumme, Passes; OQ-15 (OBC2 vs. GBn2),
  OQ-16 (dielektrische Grenze an der Oberfläche, offen). Abnahmetests als Vorschlag in ROADMAP 9.1.
- NumPy-Referenz `tests/reference/gb_ref.py` (vs. OpenMM-CustomGBForce 2·10⁻⁷, FD 2·10⁻⁸).
- `TiledGeneralizedBorn` (`src/nonbonded/generalized_born.*`, abgeleitet von `TiledAllPairs`, gemeinsame
  Bausteine in `tile_common.cuh`): Born-Integrale, Born-Radien/Selbst-/ACE-Energie, Paarterme,
  Kettenregel; Fixed-Point, Vorauswahl, Bildsumme. Bindings `set_nonbonded_gb`, `nonbonded_born_radii`.
- Builder: `from_openmm` liest ρ, s aus `implicit/obc2.xml`; `build_context(solvent="gb-obc2", ace=...)`
  (1-4 unabgeschirmt mit ε_in = 1).
- Tests `test_phase9.py`: 15/15 grün (Referenz vs. OpenMM inkl. Born-Radien einzelner Atome, FD,
  GPU-Born-Radien, GPU vs. OpenMM gesamt, Cutoff/Bildsumme, Salz-Fernfeld nahe r_c, Replicas mit
  verschiedenen Topologien/κ/ε_out in beiden Pfaden, Kraft-Energie-Konsistenz, NVE); compute-sanitizer
  memcheck 0 Fehler, racecheck 0 Hazards. Kosten ≈ 3× DH (`benchmarks/README.md`).
- Review-Agent: keine Fehler; umgesetzt: auslöschungsfreie Form von E_C + E_GB im Fernfeld (vorher bis
  2e-3 relativer FP32-Fehler nahe r_c), Born-Radien direkt gegen OpenMM, Tests für mehrere Topologien/κ
  und Bildpfad, NVE mit r_c < Molekülgröße. Dabei gefunden: harter Born-Cutoff erzeugt Energiesprünge →
  Born-Integrale jetzt mit S(r) geglättet (ADR 0010). NVE ohne Constraints: bei 1 fs Drift ≈ 1.5e-4
  (Integrationsfehler, ∝ Δt, unabhängig von r_c/ACE, Kraft-Energie-Konsistenz geprüft); Test mit 0.5 fs.
  Bitte bestätigen (geänderter Vorschlag in ROADMAP 9.1).

## Phase 8 – erledigt, abgenommen 2026-11-05 (PR #11)
- `benchmarks/throughput.py`: BioOptoMD vs. OpenMM 8.6.1 (CUDA, mixed) mit gleichem Modell (DH, r_c aus κ
  = 3.92 nm, HMR, H-Constraints 1e-6, 4 fs). Ergebnis (RTX 3090, 3 Läufe, `benchmarks/README.md`):
  1 × 20 k Faktor 0.59, **32 × 2 k Faktor 3.52 (3.46–3.68), Ziel ≥ 3 erfüllt**, 128 × 1 k Faktor 9.7.
- Optimierungen: Tile-Vorauswahl über achsenparallele Hüllen (Minimum-Image-Dreiecksungleichung,
  bitidentisch; im Bildsummen-Pfad aus), M-SHAKE und direktes RATTLE (gekoppeltes Cluster-System statt
  Gauss-Seidel; Toleranzen unverändert).
- Tests: `test_tile_culling_vs_numpy`, `test_constraints_immobile_atoms` (inv_mass = 0, Fallback,
  STATUS_CONSTRAINT_FAILURE); `test_constraint_step_vs_numpy` vergleicht auf 1e-12 auskonvergierte
  Lösungen (verfahrensunabhängig, Schranke 1e-9 nm unverändert).
- Getestet: `pytest` 94/94 grün (inkl. `slow`); compute-sanitizer memcheck 0 Fehler, racecheck 0 Hazards
  (k_tile_bounds, Tiled-Kernel, SHAKE/RATTLE).
- Review-Agent: keine Ergebnisfehler; umgesetzt: GPU-Test für unbewegliche Atome, relativer
  Sicherheitsabstand der Vorauswahl, Randpaar-Relevanz im Culling-Test belegt, gleiche
  Constraint-Toleranz in OpenMM, Endlichkeitsprüfung, Spanne über 3 Läufe, README-Begründungen
  (Multi-Process unter WDDM, abweichende Atomzahlen).
- Für den Vergleich wurde das OpenMM-CUDA-Plugin (`openmm[cuda12]`) ins venv installiert; conda-forge-
  OpenMM aus `environment.yml` enthält CUDA bereits.
- Offen: Zellliste für große Einzelsysteme (Phase 9+); Kontrolllauf auf der RTX 6000 Ada; Nsight Compute.

## Phase 7 – erledigt, abgenommen 2026-11-05 (Branch `phase-7/builder-io`, PR #10)
1. `biooptomd.builder`: `from_openmm`/`from_parmed` → `SystemSpec`; `build_context` (LJ-Typen,
   Exclusion-Tiles, HMR, H-Constraints, κ_r/ε_r(T_r), Cutoff aus κ, Box unverändert, Typ-Default 1).
   Lehnt mehrfache Kräfte gleichen Typs, Parameter-Offsets und virtuelle Sites ab.
2. Helfer: `add_anchors` (Höhe über z_top, OQ-07), `assign_types`, `dipole_probe`, `optical_tables`
   (α′, v je Element, ASSUMPTION(OQ-01/02), unkalibriert).
3. `biooptomd.io`: bitgenaue Checkpoints (`save/restore_dynamic_state`, prüft Ladungen/Massen gegen den
   Kontext; Pulling beginnt die Arbeitsakkumulation nach dem Restore neu, W selbst und Observer-Puffer
   sind nicht im Checkpoint), `DCDWriter` je Replica (C = 0, z nicht periodisch), `read_dcd`,
   `HDF5Writer`. Neue Konstante `AKMA_TIME_PS`.
4. `biooptomd.analysis`: FRET-Regime dynamic/static/static_isotropic/static_distance, Blockmittelung
   (Flyvbjerg-Petersen mit Plateau-Kriterium), n(z)-Schichtstapel, CSV-Export.
5. `examples/aptamer_on_graphene/` (Platzhalter OQ-01/02/03/05/07 markiert).
- Abnahmetests: `test_builder_roundtrip_openmm` (ParmEd und OpenMM, Energie rel. ≈ 5e-8),
  `test_checkpoint_bitexact` (mit/ohne Constraints, mit zeitabhängigem Pulling, Imagezähler ≠ 0),
  `test_end_to_end_biosensor` (`slow`, 8 Replicas, 1 ns in ≈ 80 s). Dazu `test_builder_options`,
  `test_checkpoint_mismatch`, `test_trajectory_writers`, `test_analysis.py`.
- Review-Agent: keine Fehler mit Auswirkung; umgesetzt: Typ-Default 1 statt 0, strengere
  OpenMM-Prüfungen, Ankerhöhe im Beispiel (0.35 nm wie dokumentiert, Höhe relativ zu z_top),
  datenbasierte n(z)-Prüfung, Checkpoint-Test schärfer, Massen-/Ladungsprüfung beim Laden,
  DCD-ISTART/NSTEP wie OpenMM, Plateau-Kriterium, 4σ-Toleranzen in test_analysis.
- Getestet (RTX 3090, CUDA 12.9): `pytest` 92/92 grün (inkl. `slow`), compute-sanitizer memcheck 0 Fehler
  auf Checkpoint-/Trajektorientests.
- Hinweis: ParmEd rundet Massen (bis 0.004 amu Abweichung zu OpenMM); ohne Einfluss auf Energien.

## Integration von `main` (2026-11-05)
`origin/main` (PR #7: Review, ADR 0006 Rev. 2 und ADR 0007 Accepted, OQ-01…10 entschieden; PR #8:
Build-Umgebung RTX 6000 Ada mit conda, CUDA 12.6) wurde kaskadierend in die Branches der PRs #1 → #6
gemergt und gepusht. Doppelte Nummern: unsere ADR 0006 → **0008** (Kraftschranke/Statusflags),
ADR 0007 → **0009** (Mixed Precision), unsere OQ-08…11 → **OQ-11…14**.

## Nachrüstung (Branch `phase-6/nachruestung-adr0006-0007`, PR #9 auf PR #6)
- Phase 2 (`9255cae`): Wandformen 9-3/10-4/10-4-3 (ADR 0007), Tabelle pro (Oberflächen-Typ, LJ-Typ),
  `biooptomd.surface.wall_table` (Lorentz-Berthelot, λ pro Oberflächen-Typ, Typ 0 reserviert und im
  C++-Kern erzwungen), nichtlineares Gouy-Chapman (OQ-06) mit γ aus T_r und κ_r je Replica.
- Phase 4 (`7c57edd`): Cutoff aus κ (`cutoff_from_kappa`, `debye_kappa`, `water_permittivity`, OQ-10),
  Summe über periodische Bilder bei L < 2 r_c (ADR 0006 Rev. 2); Box wird nie geändert.
- Phase 6 (siehe Commit unten): Kugelschnitt-Verschmierung (Default), Gauß σ = R/√5, Punkt (OQ-02);
  `biooptomd.voxel.optical_scale(..., amplitude=...)` mit Überschusspolarisierbarkeit (OQ-04).
- Review-Agent: keine Rechenfehler. Umgesetzt: Fehler bei nichtlinearem GC mit T_r = 0 statt stillem
  Wegfall; Oberflächen-Typ 0 im Kern erzwungen; Test mit Φ0 < 0; Φ0 = 1 V prüft V(0) = qΦ0 und
  F(0) = 2k_BTκ sinh(Φ0/2k_BT); Voxel-Normierungstest mit Kräften weit über der Fixed-Point-Auflösung;
  Energietoleranz im Wandformtest 1e-5; GC-Fernfeld über 2·atanh(u).
- Getestet (RTX 3090, CUDA 12.9): `pytest` 82/82 grün (inkl. `slow`); compute-sanitizer memcheck 0 Fehler,
  racecheck 0 Hazards (Wandformen, GC, Bildsumme, Profiler Kugel/Gauß).
- **Offen (ROADMAP Phase 6, `reference`):** `test_refractive_index_dn_dc` braucht die Kalibrierung der
  Parameter (OQ-01/02, `assumption`); `test_voxel_optical_sphere_consistency` braucht eine eigene
  Clausius-Mossotti-Vergleichsrechnung. Beide nicht umgesetzt.
- Hinweis für Phase 7 (REMD): GC liest T_r live, κ_r und ε_r (GC, Bonded, Nonbonded) bleiben auf den
  Startwerten – beim Temperaturtausch mitführen. λ derzeit nur pro Oberflächen-Typ (Atomklassen über
  eigene Oberflächen-Typen).

## Phase 6 – erledigt (Branch `phase-6/observers-voxel`, baut auf Phase 5 auf)
1. `DipoleAndDistanceTracker`, `FRETTracker` (Rohzeitreihen R_DA, κ², optional E), `RefractiveIndexProfiler`
   (S_α, S_V mit Varianz, n(z) nach Lorentz-Lorenz, optionale normierte Gauß-Verschmierung) –
   `src/observers/optics.*`, PHYSICS_MODEL §6.
2. Ringpuffer im VRAM (`src/observers/ring_buffer.hpp`), Download nur auf Anfrage, `reset()`;
   `ctx.observe_now()` für statische Auswertungen.
3. `VoxelGridForce3D` (`src/forces/voxel_grid.*`): 3D-Textur, Laden aus NumPy (Ursprung, Spacing in nm),
   Skalierung je Optik-Typ, optional periodisch in x/y; `biooptomd.voxel.values_from_potential`.
- Referenzen `tests/reference/observers_ref.py`.
- Parameter α′ und v je Typ sind Eingaben (OQ-01, OQ-02 weiter offen); optische Kopplungsnormierung OQ-04.

- Review-Agent: keine Codefehler; umgesetzt: schärfere Tests (anisotropes Voxelgitter, FRET über die Boxgrenze,
  Verschmierung pro Schicht gegen Referenz), Quantifizierung der Voxel-Energienicht-Erhaltung (O(Δ), 5 % bei
  Δ = 0.1 nm, PHYSICS_MODEL §4.3), Statusbit bei Schwerpunkt ohne Massen, Box-Versionszähler (periodisches
  Voxelgitter und Profiler erkennen `set_box` nach dem Hinzufügen), n = NaN bei L ≥ 1 dokumentiert.

## Getestet (RTX 3090, Windows 11, MSVC 17.14, CUDA 12.9, Python 3.11, nanobind 3.1, OpenMM 8.6.1)
- `pytest`: 63/63 grün (57 schnell + 6 `slow`); Phase 6: alle 6 Abnahmetests + Ringpuffer-, FRET-Minimum-
  Image-, Voxel-Interpolations-, Konsistenz- und Schutztests.
- `compute-sanitizer`: memcheck 0 Fehler, racecheck 0 Hazards (Dipol-/FRET-/Profiler-/Voxel-Kernels).

## Phase 5 – Nacharbeit nach Review (2026-11-04)
- `test_stability_4fs` statistisch neu aufgebaut (AGENTS.md Regel 8): Review maß im alten Fenster 50–250 ps
  noch Relaxation (Drift ≈ 0.65σ) und korrelierte 5-ps-Samples. Jetzt: ein Duplex, 8 unabhängige Replicas,
  beide Läufe 1 ns, Auswertung ab 500 ps; ⟨E_pot⟩-Vergleich mit σ aus Replica-Mittelwerten (4σ) und KS auf
  50-ps-Samples. Grün (330 s, `slow`).
- `inv_mass = 0` in Clustern: SHAKE bewegt solche Atome nicht mehr, RATTLE überspringt Nenner 0 (kein NaN).
- `repartition_hydrogen_mass` lehnt H-Atome mit mehreren Bindungen ab.
- Bekannte Testlücke (vom Review als physikalisch irrelevant eingestuft): Das RATTLE nach dem ersten B und
  nach O ist durch keinen Test einzeln abgesichert (die SHAKE-Geschwindigkeitskorrektur kompensiert es).

## Rote Tests
– keine.

## Offene Punkte / Hinweise für den nächsten Agenten
- Nsight Compute nachholen, sobald GPU-Counter freigeschaltet sind (Befehl in `benchmarks/README.md`).
- ADR 0008 und ADR 0009 auf `Accepted` setzen; OQ-12 bestätigen. OQ-03 ist entschieden (OL21, AMBER-DYES);
  das Phase-4-Testsystem nutzt noch OL15 ohne Fluorophor.
- Build auf der RTX 6000 Ada (laut PR #8: conda-forge-Workaround ohne Anaconda-ToS, CUDA 12.6, Python 3.12)
  ist mit dem leeren Modul belegt; Kontrolllauf der Phasen-Tests dort steht aus.
- Tiled All-Pairs überspringt Tile-Paare außerhalb r_c (Phase 8), Aufwand aber weiter O(N²) Tile-Paare;
  Zellliste später (Phase 9+).
- nanobind-„leaked instances“ nur unter compute-sanitizer mit mehreren Tests pro Prozess (siehe frühere Notiz).
- Explodierende Trajektorien setzen Statusbits; `step()` prüft sie nicht selbst.
- clang-format: CI (ubuntu) nutzt Version 21, lokal (pip) Version 18; sie richten nachgestellte Kommentare in
  Structs unterschiedlich aus. Vor dem Push mit beiden prüfen oder Kommentare über die Member setzen.

## Nächster konkreter Schritt
1. Hairpin-Lauf (OQ-15 b) auswerten, PR #15 aktualisieren, Phase 9.4 zur Abnahme; danach
   Schlussmessung 9.6 auf freier GPU (OQ-19).
2. Phase 10.4: Beispiel Aptamerschicht über Gold im Kretschmann-Aufbau; danach Abnahme Phase 10.
3. Nachholen: Kontrolllauf auf der RTX 6000 Ada; Nsight Compute (Counter-Freigabe).

## Sitzungslog
| Datum | Agent/Person | Zusammenfassung |
|---|---|---|
| 2026-11-03 | Claude (Setup) | Phase 0 angelegt. Build-Dateien sind ungetestete Entwürfe. |
| 2026-11-03 | Claude Opus 5.5 | Phase 1 implementiert (PR #1), vom Projektinhaber auf RTX 3090 abgenommen. |
| 2026-11-03 | Claude Opus 5.5 | Phase 2 implementiert (PR #2), ADR 0008; abgenommen. |
| 2026-11-03 | Claude Opus 5.5 | Phase 3 implementiert, ADR 0009 (Mixed Precision), NVE-Protokoll 100 ps; 39/39 Tests grün. |
| 2026-11-03 | Claude Opus 5.5 | Phase 4 implementiert (Tiled All-Pairs, Builder, OpenMM-Vergleich), 47/47 Tests grün; Nsight Compute blockiert (Counter-Rechte). |
| 2026-11-03 | Claude Opus 5.5 | Phase 5 implementiert (HMR, SHAKE/RATTLE, N_dof), 52/52 Tests grün; BAOAB-Kinetikbias dokumentiert (OQ-14). |
| 2026-11-04 | Claude Opus 5.5 | Phase-5-Review-Nacharbeit (Stabilitätstest nach Regel 8); Phase 6 implementiert (Optik-Observer, Ringpuffer, VoxelGridForce3D), Review umgesetzt, 63/63 Tests grün. |
| 2026-11-04 | Claude Opus 5.5 | Phase 6 abgenommen (PR #6). Sitzungsende; nächster Schritt: Phase 7 auf neuem Branch `phase-7/...` ab `phase-6/observers-voxel`. |
| 2026-11-04 | Claude (Review) | Review der offenen Fragen. Vorschläge zu OQ-01…07 in `OPEN_QUESTIONS.md`, neue OQ-08 (Mg²⁺). ADR 0006 (Cutoff aus Debye-Länge) und ADR 0007 (Wandformen) als *Proposed*. Korrekturen in PHYSICS_MODEL (falsche Cutoff-Aussage, Gouy-Chapman nichtlinear, Optik-Normierung, Verschmierung), ARCHITECTURE (Shape `dip.result()`), ROADMAP (neue Abnahmetests), `test_units.py` (prüft jetzt auch `float`-Konstanten). **Offen: Entscheidung des Projektinhabers zu ADR 0006/0007 und OQ-01…08.** |
| 2026-11-04 | Claude (Review 2) | Literatur-Checkliste per Websuche geprüft (Stand in `OPEN_QUESTIONS.md`). Nachträge zu OQ-01…08 (u. a. OL24, AMBER-DYES nur Protein-Linker, NucProt, Überschusspolarisierbarkeit, microMg). Neue OQ-09 (Box/Bedeckung vs. Cutoff) und OQ-10 (κ(T)). ADR 0006 Rev. 2: Box aus Bedeckung, Bildsummation bei L < 2 r_c (ändert ADR 0003, Entscheidung nötig). ADR 0007: Lage von z = 0. **Offen: Entscheidung zu ADR 0003/0006/0007 und OQ-01…10.** |
| 2026-11-04 | Projektinhaber (eingetragen von Claude) | Vorschläge aus Review 1 und 2 angenommen: ADR 0006 Rev. 2 und ADR 0007 → Accepted, Nachtrag in ADR 0003 (Bildsummation für Non-Bonded bei L < 2 r_c). OQ-03/04/05/06/09/10 → `resolved`; OQ-01/02/07/08 → `assumption` (Vorgehen entschieden, Kalibrierung bzw. Verifikation laut Checkliste offen). |
| 2026-11-04 | Claude (Build) | Env angelegt (CUDA 12.6), Build + leeres `_core`-Modul laufen, 2 Tests grün. Keine Quelländerungen außer dieser Datei. |
| 2026-11-05 | Claude Opus 5.5 | `origin/main` (PR #7 Review/ADR-Entscheidungen, PR #8 Build-Umgebung RTX 6000 Ada) in den Phasen-Stapel gemergt. Doppelte Nummern aufgelöst: unsere ADR 0006 → **0008** (Kraftschranke/Statusflags), ADR 0007 → **0009** (Mixed Precision), unsere OQ-08…11 → **OQ-11…14**. ADR 0006/0007 auf `main` (Cutoff aus κ, Wandformen) gelten unverändert. |
| 2026-11-05 | Projektinhaber | Nachrüstung (PR #9) abgenommen; weiter mit Phase 7. |
| 2026-11-05 | Projektinhaber | Phase 7 (PR #10) abgenommen; weiter mit Phase 8. |
| 2026-11-05 | Projektinhaber | Phase 8 (PR #11) abgenommen; weiter mit Phase 9+. |
| 2026-11-05 | Projektinhaber | Phase 9: GB-OBC2 zuerst (mbondi2, Salz, ACE optional/Default an, erweiterbar für GBn2). |
| 2026-11-05 | Projektinhaber | Phase 9.1 (PR #12) abgenommen, ADR 0010 Accepted; nächste Erweiterung: Replica Exchange. |
| 2026-11-05 | Projektinhaber | Phase 9.2 abgenommen, ADR 0011 und Kriterien bestätigt; nächste Erweiterung: GB-neck2. |
| 2026-11-05 | Claude Opus 5.5 | Phase 9.3 GB-neck2 implementiert (ADR 0012). |
| 2026-11-05 | Projektinhaber | Phase 9.3 abgenommen, ADR 0012 und Kriterien bestätigt; Vorschlag zu OQ-15 angenommen (GBn2 Default für Nukleinsäuren, Validierung offen). |
| 2026-11-05 | Claude Opus 5.5 | PRs #1–#6, #9–#14 nach main gemergt; Validierung (a) Duplex 1BNA gestartet. |
| 2026-11-05 | Claude Opus 5.5 | Validierung (a) abgeschlossen: GBn2 stabiler und näher am Kristall als OBC2 (PR #15). |
| 2026-11-05 | Projektinhaber | ADR 0013 (ImageChargeForce) angenommen, Kriterien 9.5 und Defaults OQ-18 bestätigt. |
| 2026-11-06 | Projektinhaber | Präzisierungen ADR 0013 (Punkte 6, 7, 9) bestätigt, Phase 9.5 abgenommen (PR #16). |
| 2026-11-06 | Projektinhaber | ADR 0014 (Tile-Paarliste statt Cell-List) angenommen, Rev. 1 bestätigt; Phase 9.6 erst später abnehmen, Kriterium 1.25× bleibt (OQ-19). |
| 2026-11-06 | Projektinhaber / Claude | ADR 0015 (Constraint-Löser mit gemischter Präzision) angenommen; Prototyp verfehlt das Kriterium (−2 % statt −30 %) ⇒ nach Punkt 4 zurückgezogen (Rejected). |
| 2026-11-06 | Projektinhaber | Vorschläge angenommen: OQ-15 (a) bestätigt; OQ-14 zweite Temperatur (Schätzer nach Messung); OQ-16 vorerst nicht modelliert; OQ-17 Φ0 fest; OQ-18 (a)–(c) wie Annahme, (d) C¹-glattes Clamping; Phase 9.6 als Entwurf-PR pushen; #15/#16 gemeinsam mergen nach Abnahme 9.4; nächste große Phase MD ↔ FEM-Kopplung (vor Coarse-Grained); Prototyp-Branch ADR 0015 lokal behalten. |
| 2026-11-06 | Projektinhaber | Revidiertes Kriterium ROADMAP 9.7 (Bias-Verhältnis 4 fs / 2 fs zwischen 3 und 5) bestätigt; Phase 9.7 abgenommen, PR #18 gestapelt auf #17. |
| 2026-11-06 | Projektinhaber | ADR 0016 (MD ↔ FEM) angenommen, Kriterien bestätigt; Solver hp-FEM (OQ-20); keine eigenen dn/dc-Messwerte, Kalibrierung an Literaturwerten (OQ-01/02). |
| 2026-11-06 | Projektinhaber | OQ-22 (b): Lorentz-Lokalfeldfaktor als Default in `optical_scale`; Kalibrierung zunächst klassenweise (v auf v̄, α′ an dn/dc) statt ProtOr/NucProt. |
| 2026-11-06 | Claude Opus 5.5 | Review Phase 10.0 umgesetzt: DNA-Bezugsmasse Na-Salz und v̄ als OQ-23, Ionenvolumina nach Shannon, Kalibrierung aller kovalenten Elemente, Elementprüfung, Test Profiler ↔ Kraft. |
| 2026-11-06 | Projektinhaber | Gauß-Fenster ±4σ (ADR 0016 Punkt 2) bestätigt; OQ-24 (d) Warnung, (b) später; OQ-25 Ersatzkriterium SPR angenommen. |
| 2026-11-05 | Claude Opus 5.5 | Phase 9.2 Replica Exchange implementiert (ADR 0011 Proposed), 113/113 Tests grün. |
| 2026-11-05 | Claude Opus 5.5 | Phase 9.1 GB-OBC2 implementiert (ADR 0010 Proposed, gb_ref, TiledGeneralizedBorn, Builder), 12/12 Phase-9-Tests grün. |
| 2026-11-05 | Claude Opus 5.5 | Phase 8: Benchmark vs. OpenMM (32×2k Faktor 3.5), Tile-Vorauswahl, M-SHAKE/RATTLE; Review umgesetzt, 94/94 Tests grün. |
| 2026-11-05 | Claude Opus 5.5 | Phase 7 implementiert (PR #10): Builder, Checkpoints, DCD/HDF5, Analyse, Beispiel + End-to-End-Test; Review umgesetzt. |
| 2026-11-05 | Claude Opus 5.5 | Nachrüstung Phasen 2/4/6 (PR #9): Wandformen + Mischungsregeln, nichtlineares GC, κ-Cutoff + Bildsumme, Kugelschnitt-Verschmierung, optische Normierung; Review umgesetzt. Offen: zwei `reference`-Tests aus Phase 6 (s. o.). |
