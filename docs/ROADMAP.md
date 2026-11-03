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
2. Plugin `PlanarSurfaceForce` (9-3 + optional Gouy-Chapman), Parameter pro Oberflächen-Typ.
3. Plugin `HarmonicRestraintPullingForce` (Positions-Restraint, COM-Pulling, Arbeit auf Device akkumuliert, Umbrella-Modus).
4. Energieausgabe pro Replica und Term.

**Abnahmetests**
- `test_phase2.py::test_equipartition_per_replica` – 4 Replicas bei 250/300/350/400 K, 2000 freie Atome, keine Wechselwirkung. ⟨E_kin⟩ = ½ N_dof k_B T_r innerhalb 4σ (σ aus Blockmittelung).
- `test_phase2.py::test_velocity_distribution` – Maxwell-Boltzmann-Verteilung der Geschwindigkeitskomponenten, KS-Test p > 0.001.
- `test_phase2.py::test_restraint_boltzmann` – unabhängige Atome im harmonischen Restraint: Histogramm P(x) vs. exp(−V/k_BT), χ²-Test p > 0.001.
- `test_phase2.py::test_wall_boltzmann` – Atome zwischen 9-3-Wand und harmonischem z-Restraint: P(z) vs. Boltzmann, χ²-Test.
- `test_phase2.py::test_gouy_chapman_force` – Kraft vs. finite Differenzen der Energie, rel. Fehler < 1e-4.
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
- `test_phase3.py::test_nve_energy_conservation` – γ = 0, Δt = 0.5 fs, 5000 Schritte: Drift < 1e-4 kJ/mol/ps/Atom, Fluktuation < 1e-3 der mittleren E_kin.

---

## Phase 4 – Nicht-kovalente Kräfte (Tiled All-Pairs)

**Aufgaben**
1. 32×32-Tile-Kernel, Shared-Memory, LJ + Debye-Hückel, Switching zwischen r_s und r_c, Minimum Image in x/y.
2. Exclusion-Bitmasken (1-2, 1-3, 1-4), Generierung im Builder (Python), Upload als Tile-Liste.
3. LJ-Typ-Paar-Tabelle pro Replica.
4. Elektrostatik als Template-Parameter (`DebyeHuckel`), vorbereitet für `GeneralizedBornOBC`.
5. Kernel-Profiling mit Nsight Compute; Ergebnis in `benchmarks/README.md`.

**Abnahmetests**
- `test_phase4.py::test_pair_vs_numpy` – kleine Systeme gegen O(N²)-NumPy mit Minimum Image, rel. Fehler < 1e-5.
- `test_phase4.py::test_full_system_vs_openmm` – Aptamer + Fluorophor (~1500 Atome) inkl. Ionen, OpenMM mit CustomNonbondedForce (DH + Switching) und identischen Exclusions; max. Kraftabweichung < 1e-4 relativ zur RMS-Kraft, Energie < 1e-4 relativ.
- `test_phase4.py::test_padding_and_batch_independence` – dasselbe System in Replica 0 von 1 und in Replica 7 von 16 liefert identische Kräfte; Replicas beeinflussen sich nicht.
- `test_phase4.py::test_nve_full_system` – NVE 10 ps, Δt = 1 fs, Drift im Rahmen von Phase 3.

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

---

## Phase 7 – Python-Builder, I/O & End-to-End

**Aufgaben**
1. `biooptomd.builder.from_parmed()` / `from_openmm()`: Topologie → gepaddete SoA, Exclusions, 1-4-Liste, HMR, Debye-Länge aus I, T, ε_r.
2. Helfer: Ankeratome auswählen, Fluorophor-Gruppe/Dipolachse definieren, Oberflächen-Typen zuweisen, Atom-Polarisierbarkeiten zuweisen (Tabelle aus OQ-01).
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

- Generalized Born OBC2 (ADR 0004)
- Replica-Exchange-Moves (Temperatur-Austausch auf Host-Seite)
- `ImageChargeForce` für Metalloberflächen
- Cell-List-Strategy für > 10 k Atome
- Coarse-Grained-Kraftfelder (MARTINI, ENM für Antikörper)
- Gekoppelte MD ↔ FEM-Iteration (optische Kräfte rückgekoppelt)
