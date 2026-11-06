# Roadmap

Phasen werden **strikt sequenziell** abgearbeitet (AGENTS.md §2.1). Jede Phase endet mit grünen Abnahmetests auf der Referenz-GPU (RTX 6000 Ada). Die Testnamen sind verbindlich, damit `docs/STATUS.md` eindeutig darauf verweisen kann.

---

## Phase 0 – Repository-Gerüst ✅
Dokumentation, Verzeichnisstruktur, Agenten-Anweisungen, Build-Entwürfe.

---

## Phase 1 – Build-System, DeviceState, Plugin-Gerüst, Python-Bindings

**Aufgaben**
1. `CMakeLists.txt` + `pyproject.toml` (scikit-build-core, nanobind) lauffähig machen; Architekturen `86;89`; Release-Flags `-O3`, Debug mit `-G -lineinfo`.
2. `include/biooptomd/cuda_check.hpp`: `BOMD_CUDA_CHECK`, `BOMD_LAUNCH_CHECK`.
3. `DeviceState` (trivially copyable, nur Pointer + Dimensionen) und `SimulationContext` (RAII-Besitzer aller Puffer, Stream). Layout exakt nach ARCHITECTURE.md §2, inklusive Fixed-Point-Kraftpuffer, Box pro Replica, Imagezähler.
4. Abstrakte Klassen `ForcePlugin`, `ObserverPlugin`, `NonbondedStrategy` (Host).
5. Kernels: `clear_forces`, `fixed_point_to_float`, `wrap_xy`.
6. nanobind-Modul `biooptomd._core`; Arrays als NumPy (Kopie) **und** DLPack (Zero-Copy) exportieren.
7. pytest-Infrastruktur: Marker `gpu`, `slow`, `reference`; automatischer Skip von `gpu`-Tests ohne CUDA.

**Abnahmetests**
- `test_units.py::test_cpp_python_constants_identical`
- `test_phase1.py::test_allocate_batch` – 4 Replicas × 500 Atome → `N_pad = 512`, alle Arrays korrekt dimensioniert, Padding-Atome korrekt initialisiert (q = 0, inv_mass = 0).
- `test_phase1.py::test_roundtrip_numpy_and_dlpack` – Positionen setzen, Dummy-Kernel (+1.0 nm in x) ausführen, über NumPy und über DLPack (CuPy) zurücklesen.
- `test_phase1.py::test_fixed_point_roundtrip` – Zufallskräfte im Bereich ±10⁴ kJ/mol/nm akkumulieren, Fehler < 1e-6 relativ; Ergebnis unabhängig von der Atomic-Reihenfolge (bitidentisch über 10 Läufe).
- `test_phase1.py::test_wrap_xy` – Wrap in [0, L) und Imagezähler, ungewrappte Rekonstruktion exakt.

---

## Phase 2 – Langevin-BAOAB & Basis-Umgebungsplugins

**Aufgaben**
1. `LangevinIntegrator` (BAOAB, Philox mit Zähler (seed, replica, step, atom), T pro Replica, γ = 0 ⇒ NVE).
2. Plugin `PlanarSurfaceForce` (Wandform als Template-Parameter `Wall93`/`Wall104`/`Wall1043` nach ADR 0007 + optional Gouy-Chapman, nichtlinear nach OQ-06), Parameter pro (LJ-Typ, Oberflächen-Typ).
3. Plugin `HarmonicRestraintPullingForce` (Positions-Restraint, COM-Pulling, Arbeit auf Device akkumuliert, Umbrella-Modus).
4. Energieausgabe pro Replica und Term.

**Abnahmetests**
- `test_phase2.py::test_equipartition_per_replica` – 4 Replicas bei 250/300/350/400 K, 2000 freie Atome, keine Wechselwirkung. ⟨E_kin⟩ = ½ N_dof k_B T_r innerhalb 4σ (σ aus Blockmittelung).
- `test_phase2.py::test_velocity_distribution` – Maxwell-Boltzmann-Verteilung der Geschwindigkeitskomponenten, KS-Test p > 0.001.
- `test_phase2.py::test_restraint_boltzmann` – unabhängige Atome im harmonischen Restraint: Histogramm P(x) vs. exp(−V/k_BT), χ²-Test p > 0.001.
- `test_phase2.py::test_wall_boltzmann` – Atome zwischen 9-3-Wand und harmonischem z-Restraint: P(z) vs. Boltzmann, χ²-Test.
- `test_phase2.py::test_gouy_chapman_force` – Kraft vs. finite Differenzen der Energie, rel. Fehler < 1e-4 (nichtlineare Form, Φ₀ ∈ {25, 100, 200} mV).
- `test_phase2.py::test_gouy_chapman_linear_limit` – nichtlineare Form → lineare Form für Φ₀ → 0 (rel. Abweichung < 1e-3 bei 1 mV); γ wird pro Replica aus T_r berechnet.
- `test_phase2.py::test_wall_forms_vs_numpy` – alle Wandformen: Energie/Kraft vs. NumPy-Referenz und finite Differenzen, rel. Fehler < 1e-4; Lage und Tiefe des Minimums analytisch (ADR 0007).
- `test_phase2.py::test_wall_mixing_rules` – Builder-Parametertabelle vs. manuell berechnete Lorentz-Berthelot-Werte.
- `test_phase2.py::test_determinism` – zwei Läufe mit gleichem Seed bitidentisch, anderer Seed verschieden.
- `test_phase2.py::test_jarzynski_harmonic` – Pulling in rein harmonischem Potential: ⟨exp(−βW)⟩ liefert ΔG = 0 innerhalb Fehlerbalken.

---

## Phase 3 – Kovalente Kräfte

**Aufgaben**
1. NumPy-Referenz für alle Terme in `tests/reference/bonded_ref.py` mit Finite-Differenzen-Test.
2. CUDA-Kernels für Bonds, Angles (atan2), Periodic Torsions/Impropers, 1-4-Paare (LJ + Debye-Hückel skaliert).
3. Topologie-Upload pro Replica (Offsets nach ADR 0005).
4. OpenMM-Referenzgenerator: Testmolekül (DNA-Trinukleotid, AMBER OL15 oder bsc1) in Vakuum/NoCutoff mit CustomBondForce für 1-4-DH.

**Abnahmetests**
- `test_phase3.py::test_bonded_vs_numpy` – jeder Term einzeln, rel. Kraftfehler < 1e-5.
- `test_phase3.py::test_bonded_vs_openmm` – Gesamtkraft aller Atome, max. Abweichung < 1e-4 relativ zur RMS-Kraft.
- `test_phase3.py::test_degenerate_geometries` – kollineare Winkel, Torsion bei 0/π, kein NaN/Inf.
- `test_phase3.py::test_nve_energy_conservation` – γ = 0, Δt = 0.5 fs, 200 000 Schritte (100 ps; geändert am 2026-11-03 durch den Projektinhaber, da 5000 Schritte eine Drift von 1e-4 statistisch nicht auflösen): Drift (Regressionssteigung) < 1e-4 kJ/mol/ps/Atom, Fluktuation < 1e-3 der mittleren E_kin.

---

## Phase 4 – Nicht-kovalente Kräfte (Tiled All-Pairs)

**Aufgaben**
1. 32×32-Tile-Kernel, Shared-Memory, LJ + Debye-Hückel, Switching zwischen r_s und r_c, Minimum Image in x/y. Cutoff aus der Debye-Länge (ADR 0006); bei L < 2 r_c Summation über periodische Bilder statt Minimum Image.
2. Exclusion-Bitmasken (1-2, 1-3, 1-4), Generierung im Builder (Python), Upload als Tile-Liste.
3. LJ-Typ-Paar-Tabelle pro Replica.
4. Elektrostatik als Template-Parameter (`DebyeHuckel`), vorbereitet für `GeneralizedBornOBC`.
5. Kernel-Profiling mit Nsight Compute; Ergebnis in `benchmarks/README.md`.

**Abnahmetests**
- `test_phase4.py::test_pair_vs_numpy` – kleine Systeme gegen O(N²)-NumPy mit Minimum Image, rel. Fehler < 1e-5.
- `test_phase4.py::test_full_system_vs_openmm` – Aptamer + Fluorophor (~1500 Atome) inkl. Ionen, OpenMM mit CustomNonbondedForce (DH + Switching) und identischen Exclusions; max. Kraftabweichung < 1e-4 relativ zur RMS-Kraft, Energie < 1e-4 relativ.
- `test_phase4.py::test_padding_and_batch_independence` – dasselbe System in Replica 0 von 1 und in Replica 7 von 16 liefert identische Kräfte; Replicas beeinflussen sich nicht.
- `test_phase4.py::test_nve_full_system` – NVE 10 ps, Δt = 1 fs, Drift im Rahmen von Phase 3.
- `test_phase4.py::test_cutoff_from_kappa` – r_c korrekt für I = 10/150/1000 mM und Replicas unterschiedlicher T; Builder ändert L nie (ADR 0006 Rev. 2).
- `test_phase4.py::test_image_sum_vs_lattice_reference` – L < 2 r_c: elektrostatische Energie und Kräfte vs. NumPy-Gittersumme, rel. RMS-Kraftfehler < 1e-3, elektrostatische Energie rel. < 1e-3.
- `test_phase4.py::test_image_sum_self_images` – einzelnes geladenes Atom: Energie = ½ Σ_{n≠0} V_DH(|n|), Kraft = 0.
- `test_phase4.py::test_image_sum_reduces_to_minimum_image` – L ≥ 2 r_c: beide Pfade bis auf Rundung identisch.

---

## Phase 5 – Constraints & Hydrogen Mass Repartitioning

**Aufgaben**
1. HMR im Builder.
2. SHAKE/RATTLE auf H-Bindungen, parallel über Cluster, integriert in BAOAB.
3. Korrekte N_dof in Temperaturberechnung.

**Abnahmetests**
- `test_phase5.py::test_constraint_tolerance` – max. relative Bindungslängenabweichung < 1e-6 nach jedem Schritt.
- `test_phase5.py::test_equipartition_with_constraints` – Temperatur stimmt mit korrigierten N_dof (4σ).
- `test_phase5.py::test_stability_4fs` – Aptamer an Oberfläche, Δt = 4 fs, 1 ns bei 300 K ohne Explosion; Energieverteilung konsistent mit Δt = 1 fs-Lauf (KS-Test).

---

## Phase 6 – In-situ Observer & VoxelGridForce3D

**Aufgaben**
1. `DipoleAndDistanceTracker`, `FRETTracker` (Rohzeitreihen), `RefractiveIndexProfiler` (S_α, S_V, n(z)) nach PHYSICS_MODEL.md §6.
2. Ringpuffer auf dem Device, Download nur auf Anfrage, `reset()`.
3. `VoxelGridForce3D` mit 3D-Textur, Lade-Funktion aus NumPy-Array (Gitterursprung, Spacing in nm).

**Abnahmetests**
- `test_phase6.py::test_dipole_static_geometry` – bekannte Geometrie → exakte Werte für μ̂, θ, d.
- `test_phase6.py::test_kappa2_isotropic` – zufällig isotrop orientierte Dipole: ⟨κ²⟩ = 2/3 innerhalb 4σ.
- `test_phase6.py::test_refractive_index_pure_water` – keine Atome → n(z) = n_w exakt.
- `test_phase6.py::test_refractive_index_uniform_layer` – synthetische homogene Atomschicht bekannter Dichte → analytisches Lorentz-Lorenz-Ergebnis, rel. Fehler < 1e-4.
- `test_phase6.py::test_voxel_linear_field` – lineares Potential im Gitter → konstante Kraft exakt; harmonisches Potential → Boltzmann-Verteilung (χ²).
- `test_phase6.py::test_observer_ignores_padding`.
- `test_phase6.py::test_refractive_index_smearing` – einzelnes Atom: Summe über Schichten von S_α und S_V exakt erhalten; Kugelschnitt (Default) entspricht der analytischen Schichtvolumenformel, keine Beiträge unter z = 0; Gauß-Option mit σ = R/√5.
- `test_phase6.py::test_refractive_index_dn_dc` (`reference`) – homogene Protein- bzw. DNA-Lösung der Konzentration c: n = n_w + c·dn/dc innerhalb der Kalibriertoleranz (OQ-01/02).
- `test_phase6.py::test_voxel_optical_normalization` – konstantes |E₀|²-Gradientenfeld: Kraft = −s_i ∇W mit s_i aus α′_eff nach OQ-04; `amplitude="rms"` ergibt Faktor 2; Atom mit α′_i = (3/4π) v_i L_w erfährt keine Kraft.
- `test_phase6.py::test_voxel_optical_sphere_consistency` (`reference`) – dichte Atomkugel mit Brechungsindex n_p (aus dem Profiler) im Feldgradienten: Gesamtkraft vs. Dipolkraft einer dielektrischen Kugel (Clausius-Mossotti mit n_p/n_w); Abweichung dokumentieren (klärt den Lokalfeldfaktor, OQ-04).

---

## Phase 7 – Python-Builder, I/O & End-to-End

**Aufgaben**
1. `biooptomd.builder.from_parmed()` / `from_openmm()`: Topologie → gepaddete SoA, Exclusions, 1-4-Liste, HMR, Debye-Länge aus I, T, ε_r.
2. Helfer: Ankeratome auswählen (Oberflächen-Typ 0, z-Restraint auf Chemisorptionshöhe, OQ-07), Fluorophor-Gruppe/Dipolachse definieren (Tabelle pro Farbstoff, OQ-03), Oberflächen-Typen zuweisen, Atom-Polarisierbarkeiten und -Volumina zuweisen (OQ-01/02), automatischer Cutoff (ADR 0006), optionale SAM/MCH-Wand.
3. I/O: Trajektorien als DCD oder HDF5 (pro Replica), Checkpoints (Positionen, Geschwindigkeiten, RNG-Zähler, Schrittzähler) bitgenau wiederherstellbar.
4. `biooptomd.analysis`: FRET-Mittelungsregime, Blockmittelung, n(z)-Export für Transfer-Matrix-Solver.
5. Beispiel `examples/aptamer_on_graphene/`.

**Abnahmetests**
- `test_phase7.py::test_builder_roundtrip_openmm` – Energie eines aus ParmEd gebauten Systems = OpenMM-Energie.
- `test_phase7.py::test_checkpoint_bitexact` – Lauf 2N Schritte == N Schritte + Checkpoint + N Schritte.
- `test_phase7.py::test_end_to_end_biosensor` (`slow`) – oberflächenfixiertes Aptamer, 8 Replicas, 1 ns: Observer liefern endliche, normierte Zeitreihen; n(z) → n_w für z ≫ Molekülhöhe.

---

## Phase 8 – Performance & Benchmark

- Durchsatz (Gesamt-ns/Tag über alle Replicas) für 1×20 k, 32×2 k, 128×1 k Atome vs. OpenMM (gleiches Modell, CUDA-Plattform, ein Context pro Replica bzw. Multi-Process).
- Ziel: ≥ 3× OpenMM-Gesamtdurchsatz im Batch-Regime 32×2 k. Ergebnisse in `benchmarks/README.md`.

---

## Phase 9+ – Erweiterungen (Reihenfolge nach Forschungsbedarf)

### Phase 9.1 – Generalized Born OBC2 (ADR 0010, Proposed; Reihenfolge vom Projektinhaber am 2026-11-05 festgelegt)

**Aufgaben**
1. NumPy-Referenz `tests/reference/gb_ref.py` (Born-Radien, Energie, analytische Kräfte).
2. Nonbonded-Strategie `TiledGeneralizedBorn` (LJ + Coulomb + GB-OBC2 mit Salz, ACE optional, Switching
   und Bildsumme wie ADR 0006, Tile-Vorauswahl); Bindings.
3. Builder: GB-Parameter (ρ, s) aus OpenMM (`implicit/obc2.xml`), Auswahl `solvent="debye-huckel" | "gb-obc2"`.

**Abnahmetests (Vorschlag, mit ADR 0010 vom Projektinhaber zu bestätigen)**
- `test_phase9.py::test_gb_reference_vs_openmm` (`reference`) – NumPy-Referenz vs. OpenMM-`CustomGBForce`
  (obc2.xml, κ > 0, ACE an/aus): Energie rel. < 1e-6; Born-Radien einzelner Atome (aus dem Selbstterm
  mit Einzelladung rekonstruiert, OpenMM stellt sie nicht bereit) rel. < 1e-6.
- `test_phase9.py::test_gb_reference_forces_fd` – Referenzkräfte vs. finite Differenzen, rel. < 1e-5.
- `test_phase9.py::test_gb_born_radii_vs_numpy` – GPU-Born-Radien vs. Referenz, rel. < 1e-5.
- `test_phase9.py::test_gb_vs_openmm` (`reference`) – isolierter Duplex (r_c größer als das Molekül):
  Gesamtenergie und Kräfte vs. OpenMM, Energie rel. < 1e-4, Kräfte < 1e-4 der größten RMS-Kraft eines Terms.
- `test_phase9.py::test_gb_cutoff_and_images_vs_numpy` – r_c < Molekülgröße und L < 2 r_c: GPU vs. Referenz
  (gleiches Switching, Bildsumme), Kräfte < 1e-4 RMS.
- `test_phase9.py::test_gb_salt_far_field` – zwei Ionen bei r ≫ B: E_C + E_GB → Debye-Hückel, rel. < 1e-3.
- `test_phase9.py::test_gb_padding_batch_determinism` – Padding ohne Beitrag, Replicas mit verschiedenen
  Topologien, κ und ε_out unabhängig und bitidentisch (Minimum-Image- und Bildsummenpfad).
- `test_phase9.py::test_gb_force_energy_consistency` – GPU-Kraft = −∇(GPU-Energie) (Richtungsableitung), rel. < 1e-3.
- `test_phase9.py::test_gb_nve` (`slow`) – NVE 10 ps, Δt = 0.5 fs ohne Constraints, r_c kleiner als das Molekül:
  Drift im Rahmen von Phase 3. (Bei 1 fs ohne Constraints driftet GB mit ≈ 1.5e-4 – Integrationsfehler, skaliert
  stark mit Δt, unabhängig von r_c; mit H-Constraints bei 1 fs gleichauf mit DH.)

### Phase 9.2 – Replica Exchange (ADR 0011, Accepted; abgenommen 2026-11-05)

**Aufgaben**
1. Plugin-Hook `set_screening` (κ_r, ε_r je Replica zur Laufzeit), `scale_velocities`.
2. `biooptomd.remd.ReplicaExchange`: Zustandstausch mit Hamiltonian-Kriterium (Kreuzenergien), Statistik,
   Checkpoint.

**Abnahmetests (vom Projektinhaber am 2026-11-05 bestätigt)**
- `test_phase9_remd.py::test_remd_exchange_mechanics` – jede Entscheidung gleich der unabhängig (frische Kontexte)
  berechneten Metropolis-Entscheidung; nach Tausch T, κ, ε je Slot getauscht (Energien gleich frischem Kontext),
  Geschwindigkeiten exakt mit sqrt(T_neu/T_alt) skaliert.
- `test_phase9_remd.py::test_remd_sampling_harmonic` (`slow`) – unabhängige harmonische Atome, 4 Zustände:
  ⟨U⟩_k = (3/2) N k_B T_k innerhalb 4σ; Akzeptanzraten vs. analytische Erwartung (Gamma-Verteilungen) innerhalb 4σ.
- `test_phase9_remd.py::test_remd_checkpoint_bitexact` – 2M Zyklen == M + Checkpoint (inkl. Austauschzustand) + M.
- `test_phase9_remd.py::test_set_screening` – set_screening wirkt auf Nonbonded (DH/GB) und Gouy-Chapman, im GB-Modus
  nicht auf die 1-4-Terme.

### Phase 9.3 – GB-neck2 (ADR 0012, Accepted; abgenommen 2026-11-05)

**Aufgaben**
1. NumPy-Referenz um GBn2 erweitern (Offset, α/β/γ je Atom, Neck-Term).
2. Kernel: Neck-Term und α/β/γ je Atom im bestehenden GB-Kernel; Bindings.
3. Builder: Parameter und Neck-Tabellen aus `implicit/gbn2.xml`, `solvent="gb-gbn2"`.

**Abnahmetests (vom Projektinhaber am 2026-11-05 bestätigt)**
- `test_phase9_gbn2.py::test_gbn2_reference_vs_openmm` (`reference`) – Energie rel. < 1e-6, Kräfte < 1e-5 RMS;
  Tabellenausrichtung geprüft.
- `test_phase9_gbn2.py::test_gbn2_reference_forces_fd` – Kräfte inkl. Neck vs. finite Differenzen, rel. < 1e-5.
- `test_phase9_gbn2.py::test_gbn2_gpu_vs_numpy` (`reference`), `::test_gbn2_cutoff_and_images_vs_numpy` – Born-Radien
  rel. < 1e-5, Energie rel. < 1e-5, Kräfte < 1e-4 RMS (Minimum-Image- und Bildsummenpfad).
- `test_phase9_gbn2.py::test_gbn2_vs_openmm` (`reference`) – Builder-Weg gesamt vs. OpenMM, Energie/Kräfte < 1e-4.
- `test_phase9_gbn2.py::test_gbn2_nve` (`slow`) – NVE wie test_gb_nve (0.5 fs), Drift < 1e-4.

### Phase 9.4 – Validierung GBn2 + OL21 (OQ-15; vom Projektinhaber am 2026-11-05 gewählt; abgenommen 2026-11-06)

**Aufgaben**
1. (a) Duplex 1BNA, GBn2 vs. OBC2, 4 × 100 ns, 0.15 M, 300 K (`examples/gb_validation/duplex_stability.py`).
2. (b) Hairpin d(GCGAAGC), PDB 1PQT (Wahl des Projektinhabers): REMD-Faltung aus gestreckter Struktur,
   4 unabhängige Leitern × 8 T (300–370 K), 1 µs je Replica (`examples/gb_validation/hairpin_folding.py`).
3. `ReplicaExchange(ladders=L)`: mehrere unabhängige Leitern je Kontext.

**Abnahmekriterien (Vorschlag, Bestätigung durch den Projektinhaber offen)**
- (a) GBn2 mindestens so stabil wie OBC2 (intakte Paare) und näher an der Kristallstruktur (RMSD), OQ-15.
- (b) Leitern aus gestreckter Struktur erreichen den nativen Zustand; Anteil gefaltet bei 300 K in der
  zweiten Hälfte in allen 4 Leitern übereinstimmend (Abweichung im Rahmen der Leiter-zu-Leiter-Streuung).
- `test_phase9_remd.py::test_remd_multiple_ladders`, `::test_remd_explicit_screening_ladders` –
  Entscheidungen, Zähler und Zustandsparameter je Leiter exakt wie unabhängig berechnet.

### Phase 9.5 – ImageChargeForce (ADR 0013, Accepted; abgenommen 2026-11-06)

**Aufgaben**
1. NumPy-Referenz `tests/reference/image_charge_ref.py` (Energie, Kräfte, Minimum Image und Gittersumme).
2. Plugin `src/forces/image_charge.{hpp,cu}`: Tile-Pass mit gespiegelten j-Koordinaten, Vorauswahl in z,
   Clamping-Zähler, `set_screening`; Bindings `add_image_charges`.
3. Builder: `builder.add_image_charges(ctx, info, material)` (z_top, κ, ε_r, r_c aus BuildInfo und Material).

**Abnahmetests (vom Projektinhaber am 2026-11-05 bestätigt)**
- `test_phase9_image.py::test_image_reference_dirichlet` (reines NumPy) – Potential von Ladungen + Bildern
  auf der Ebene z_im verschwindet (< 1e-12 relativ, ohne Switching); Eigenbildenergie einer Ladung
  analytisch; Kräfte der Referenz vs. finite Differenzen rel. < 1e-6.
- `::test_image_gpu_vs_numpy`, `::test_image_lateral_culling` – Energie rel. < 1e-5, Kräfte < 1e-4 RMS, mehrere Replicas mit
  verschiedenem κ/ε_r, Minimum-Image- und Bildsummenpfad, f = 1 und f = −0.5.
- `::test_image_far_atoms_zero` – Atome mit ζ > r_c erhalten exakt null Bildkraft (Vorauswahl).
- `::test_image_clamp` – ζ < ζ_c: Energie stetig, Kraft in z null, Zähler zählt (seit Phase 9.7:
  ζ_raw ≤ ζ_c − h, C¹-Übergang darüber, siehe 9.7).
- `::test_image_set_screening` – nach `set_screening` gleich einem frischen Kontext (REMD).
- `::test_image_nve` (`slow`) – DH + Wand + Bilder, NVE-Drift < 1e-4 kJ/mol/ps/Atom wie test_gb_nve
  (seit Phase 9.7 < 3e-5).
- `::test_image_determinism` – zwei Läufe bitidentisch.
- Messung (kein Kriterium): Kosten des Passes für ein Aptamer auf Gold relativ zu Debye-Hückel.

### Phase 9.6 – Tile-Paarliste für große Einzelsysteme (ADR 0014, Accepted 2026-11-06)

**Aufgaben**
1. Listenaufbau je Auswertung (Thread je Tile-Paar, Hüllentest wie bisher), persistenter
   Interaktionskernel über die Liste je Replica; `TiledAllPairs` (DH) und `TiledGeneralizedBorn`.
2. Auswahl `tile_list = "auto" | "on" | "off"` (ADR 0014 Rev. 1), Default `"auto"`.

**Abnahmetests (vom Projektinhaber am 2026-11-06 bestätigt)**
- `test_phase9_tile_list.py::test_tile_list_bitidentical` – Energien und Kräfte bitidentisch zum bisherigen
  Kernel (`tile_list="off"`): 2k-System mit mehreren Replicas, 20k-System, verschiedene Boxen je Replica,
  per-Replica-Topologien (ADR 0005); DH und GB-OBC2/GBn2.
- `::test_tile_list_extremes` – alle Tile-Paare verworfen (nur Diagonale), keines verworfen, ein einzelnes Tile,
  Atome nahe der periodischen Grenze: bitidentisch.
- `::test_tile_list_determinism` – zwei Läufe (1000 Schritte, Langevin) bitidentisch.
- `::test_tile_list_auto`, `::test_tile_list_set_box` – Auto-Auswahl nach Boxgröße (auch nach `set_box`),
  Bildsumme nie, bitidentisch (ADR 0014 Rev. 1).
- `::test_tile_list_many_per_warp_2k`, `::test_tile_list_many_per_warp_gb` – Liste erzwungen, 1 Block je
  Replica (viele Tile-Paare je Warp), 2k mit 32 Replicas und GB: bitidentisch.
- Messkriterium: vom Projektinhaber am 2026-11-06 nach der Endmessung von 1.25× auf ≥ 1.10× gesenkt
  (gemessen 1.12×, OQ-19).
- Bildsummenpfad (L < 2 r_c) unverändert (bestehende Tests aus Phase 4 und 9.1).
- memcheck und racecheck ohne Befund.
- Messung (Kriterium): 1 × 20 k mindestens ~~1.25×~~ 1.10× schneller als vorher (OQ-19); 32 × 2 k und 128 × 1 k nicht
  langsamer (Toleranz 2 %); Vergleich mit OpenMM berichtet, kein Muss. Messung erst bei freier GPU.

### Phase 9.7 – Folgearbeiten aus OQ-14 und OQ-18 (d) (Projektinhaber 2026-11-06; abgenommen 2026-11-06)
1. Zweite Temperatur (OQ-14): Kandidaten Geschwindigkeiten nach dem O-Schritt und konfigurationelle
   Temperatur messen; der Schätzer, der den Bias bei Δt = 4 fs (HMR, H-Constraints) beseitigt, wird
   zusätzlich gemeldet. E_kin am vollen Schritt bleibt unverändert.
   *Umgesetzt:* E_kin nach O (+ RATTLE), `ctx.kinetic_energy_mid()` und `kinetic_mid` im
   EnergyTracker. Die konfigurationelle Temperatur bräuchte die Spur der Hesse-Matrix aller
   Kraftterme; sie war nicht nötig, weil schon der erste Kandidat den Bias beseitigt.
2. C¹-glattes Clamping von ζ in `ImageChargeForce` (OQ-18 d), ζ_c = 0.1 nm bleibt.

Abnahmetests (revidierte Fassung vom Projektinhaber bestätigt 2026-11-06):
- Zweite Temperatur bei 300 K, Δt = 4 fs, 1 × 2 k mit Constraints: |⟨T⟩ − 300 K| < 1 % (4σ).
  ~~Bei 1 fs stimmen beide Schätzer überein (4σ).~~ Revidiert nach der Messung: Auch bei 1 fs liegt
  E_kin am vollen Schritt noch 1.8 K unter 300 K (Bias ∝ Δt²: 31.5 / 7.7 / 1.8 K bei 4 / 2 / 1 fs).
  Stattdessen gilt: Das Verhältnis der Bias von 4 fs zu 2 fs liegt zwischen 3 und 5.
  `tests/python/test_phase9_temperature.py::test_temperature_mid_4fs`.
- Image: Kraft = −∇E (finite Differenzen) auch im Übergangsbereich um ζ_c; Energie und Kraft stetig
  über ζ_c; NVE-Drift mit H-Atomen nahe Gold nicht schlechter als in Phase 9.5; alle Tests aus 9.5 grün.

### Phase 10 – Selbstkonsistente Kopplung MD ↔ FEM (ADR 0016, Accepted 2026-11-06; Solver hp-FEM) – abgenommen 2026-11-06
0. Voraussetzungen: `test_voxel_optical_sphere_consistency` (Lokalfeldfaktor, OQ-04; *umgesetzt
   2026-11-06*, Ergebnis Faktor ≈ 1.6, OQ-22 (b): Lorentz-Faktor in `optical_scale`) und Kalibrierung
   α′/v an dn/dc (`test_refractive_index_dn_dc`, OQ-01/02; *umgesetzt 2026-11-06*, klassenweise
   Skalierung `builder.calibrate_optics`; Kopplung Profiler ↔ Kraft: `test_optical_force_dilute_solution`).
   Ohne eigene Messwerte: Literaturwerte, deren Unsicherheit
   als Toleranz dient (ADR 0016 Punkt 7).
1. `PolarizabilityGridObserver` (3D-ε-Karte, Gauß-Verschmierung, Lorentz-Lorenz). *Umgesetzt
   2026-11-06* (`ctx.add_polarizability_grid`, `test_phase10.py`).
2. `VoxelGridForce3D::set_values` (Laufzeit-Update). *Umgesetzt 2026-11-06* (`test_voxel_set_values`).
3. `biooptomd.coupling`: Solver-Protokoll, TMM-Referenzsolver (1D), hp-FEM-Adapter `HpFemSolver`
   (im selben Prozess; npz-Datei-Adapter als Rückfall), Treiber `SelfConsistentOptics` (Picard mit
   Dämpfung β, Checkpoint je Iteration). *Umgesetzt 2026-11-06:* Protokoll, `tmm`/`TmmSolver`,
   `SelfConsistentOptics` (`test_tmm_reference`, `test_tmm_solver_grid`, `test_coupling_fixed_point`,
   `test_coupling_scaling`, `test_coupling_resume`). hp-FEM-Adapter `HpFemGratingSolver` (in x
   periodisch, in y invariant; `test_hpfem_adapter_planar`, rel. 3e-5 gegen TMM;
   `test_hpfem_adapter_lateral`) und
   `NpzFileSolver` (`test_npz_file_solver`). Offen: allgemeiner 3D-Adapter (`Scattering3D` mit
   `LayerStack3D`), Treiber mit Replica Exchange (Zielzustand über die Slots, ADR 0016 Punkt 5).
4. Beispiel: Aptamerschicht über Gold im Kretschmann-Aufbau mit dem TMM-Adapter. *Umgesetzt
   2026-11-06* (`examples/aptamer_spr_gold`, `test_example_aptamer_spr_gold`).
5. *Nächster Schritt (Projektinhaber 2026-11-06):* 3D-hp-FEM-Adapter (`Scattering3D` mit
   `LayerStack3D`, Bloch-Einheitszelle in x und y, strukturiertes Tetraedernetz `hpfem.box`) für
   Strukturen, die in x und y variieren (Nanopartikel, Hotspots). Kriterien (Vorschlag): an einer in y
   invarianten Struktur gleiches W wie `HpFemGratingSolver` (rel. < 1 %); planar gegen TMM
   (rel. < 1 %); Speicher und Laufzeit je Lösung über der ε-Auflösung im Molekülbereich dokumentiert.
   Offen daneben: Replica Exchange im Treiber.

Abnahmetests (Vorschlag): siehe ADR 0016, Konsequenzen (`test_polarizability_grid_vs_numpy`,
`test_voxel_set_values`, `test_tmm_reference`, `test_coupling_fixed_point`, `test_coupling_scaling`,
`test_hpfem_adapter_planar`, `test_coupling_resume`); vom Projektinhaber am 2026-11-06 bestätigt.

- Generalized Born OBC2 (ADR 0004)
- Replica-Exchange-Moves (Temperatur-Austausch auf Host-Seite)
- `ImageChargeForce` für Metalloberflächen
- ~~Cell-List-Strategy für > 10 k Atome~~ – ersetzt durch die Tile-Paarliste (Phase 9.6, ADR 0014)
- Coarse-Grained-Kraftfelder (MARTINI, ENM für Antikörper)
- Gekoppelte MD ↔ FEM-Iteration (optische Kräfte rückgekoppelt)
