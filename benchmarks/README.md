# Benchmarks

Wird in Phase 4 (Kernel-Profiling) und Phase 8 (Durchsatz vs. OpenMM) gefüllt. Jede Messung mit Datum, Commit, GPU, Treiber-/CUDA-Version.

## Phase 4 – `k_tiled_all_pairs<DebyeHuckel>` (2026-11-03)

- **Umgebung:** RTX 3090 (CC 8.6, max. SM-Takt 2100 MHz), Treiber 610.88, CUDA 12.9, Windows 11 (WDDM),
  Branch `phase-4/tiled-nonbonded` (Arbeitsstand auf `c5ea3b2`), Release-Build (`-O3 -lineinfo`).
- **System:** 2 DNA-Dodecamer-Duplexe + 42 Ionen, 1558 Atome (N_pad = 1568, 49 Tiles, 1225 Tile-Paare je
  Replica), 18 LJ-Typen, r_c = 2.0 nm, Box 8 × 8 nm. Nur Nonbonded (`compute_forces` inkl. clear/convert).
- **Ressourcen** (`cuobjdump --dump-resource-usage`): 48 Register/Thread, **0 Byte Local Memory (keine
  Spills)**, 4096 B statisches Shared Memory + T²·8 B dynamisch (18 Typen: 2592 B), 128 Threads/Block.

| Replicas | `compute_forces` [µs] | Paarprüfungen/Aufruf | Paarprüfungen/s |
|---|---|---|---|
| 1 | 29 | 1.2·10⁶ | 4.1·10¹⁰ |
| 4 | 65 | 4.9·10⁶ | 7.5·10¹⁰ |
| 16 | 168 | 1.97·10⁷ | 1.17·10¹¹ |
| 64 | 598 | 7.9·10⁷ | 1.32·10¹¹ |

(Paarprüfungen = alle 32×32-Paare der Tile-Paare, also ≈ N_pad²/2 je Replica; davon liegt bei r_c = 2 nm
nur ein Bruchteil innerhalb des Cutoffs. Bei 1 Replica dominiert der Launch-Overhead.)

**Gesamtsimulation** (kovalent + nicht-kovalent, NVE, 4 Replicas × 1558 Atome): ≈ 9 600 Schritte/s,
also ≈ 830 ns/Tag je Replica bei Δt = 1 fs.

**Nsight Compute: ausstehend.** `ncu` meldet `ERR_NVGPUCTRPERM` – der Zugriff auf die GPU-Performance-
Counter muss unter Windows freigeschaltet werden (NVIDIA Control Panel → Desktop → Developer Settings →
„Allow access to the GPU performance counters to all users“ bzw. Ausführung als Administrator). Danach:

```bat
"C:\Program Files\NVIDIA Corporation\Nsight Compute 2026.3.0\target\windows-desktop-win7-x64\ncu.exe" ^
  --kernel-name regex:k_tiled_all_pairs --launch-skip 3 --launch-count 1 --set full ^
  .venv\Scripts\python benchmarks\profile_nonbonded.py 16
```

**Erwartete Optimierungen (nicht Teil von Phase 4):** Tile-Paare außerhalb des Cutoffs überspringen
(Bounding-Box-Test je Tile), Cell-List-Strategie für > 10 k Atome (ARCHITECTURE §6).

## Phase 8 – Durchsatz vs. OpenMM (2026-11-05)

- **Umgebung:** RTX 3090 (CC 8.6), Treiber 610.88, CUDA 12.9, Windows 11 (WDDM), Release-Build,
  Branch `phase-8/benchmark`. OpenMM 8.6.1 (pip, CUDA-Plugin `openmm[cuda12]`), Platform CUDA,
  Precision `mixed`.
- **Modell (in beiden Engines gleich):** AMBER OL15 + Ionen, Debye-Hückel (I = 0.15 M, ε_r(300 K) = 77.6),
  Switching r_s = r_c − 0.2 nm, **r_c = 3.92 nm aus κ (ADR 0006)**, 1-4 abgeschirmt, HMR (3.024 amu),
  H-Constraints, Δt = 4 fs, Langevin γ = 1/ps, 300 K. OpenMM: Referenzsystem aus
  `tests/reference/openmm_nonbonded.py` (CustomNonbondedForce, CutoffPeriodic) + `LangevinMiddleIntegrator`,
  ein Context pro Replica, nacheinander in einem Prozess (in Blöcken zu 32 Contexts; mehr legt OpenMM
  unter Windows nicht gleichzeitig an).
- **Systeme:** Box 8 × 8 nm je Einheit (Bedeckung fest, OQ-09): „1k“ = ein Duplex + 20 Na⁺ (778 Atome),
  „2k“ = zwei Duplexe + 42 Ionen (1558), „20k“ = 3 × 4 Kopien von „2k“ (18 696 Atome, 24 × 32 nm).
- **Messung:** `python benchmarks/throughput.py --steps 2000 --warmup 200`; Gesamt-ns/Tag über alle Replicas.

| Fall | Atome/Replica | BioOptoMD [ns/Tag] | OpenMM [ns/Tag] | Faktor |
|---|---|---|---|---|
| 1 × 20 k | 18 696 | 653 | 1 082 | 0.60 |
| **32 × 2 k** | 1 558 | **13 804** | 3 801 | **3.63** (Ziel ≥ 3) |
| 128 × 1 k | 778 | 41 760 | 4 404 | 9.48 |

Die OpenMM-Werte schwanken zwischen Läufen um etwa ±7 % (3 534–4 019 ns/Tag für 32 × 2 k).

**Optimierungen in Phase 8** (alle ohne Änderung der Ergebnisse über die Rundung hinaus):
1. *Tile-Vorauswahl:* je Tile die achsenparallele Hülle der Atome (`k_tile_bounds`); Tile-Paare, deren
   Hüllen mit Minimum Image weiter als r_c + 1e-3 nm entfernt sind, überspringt der Warp (übersprungene
   Paare tragen exakt 0 bei, bitidentisch). 1 × 20 k: 305 → 607 ns/Tag. Im Bildsummen-Pfad
   (L < 2 r_c) nicht aktiv.
2. *Constraint-Löser:* SHAKE als Newton-Iteration auf dem gekoppelten Cluster-System (M-SHAKE), RATTLE als
   direkte Lösung des linearen Systems statt Gauss-Seidel. Vorher (Nsight Systems, 32 × 2 k) 5 × 63 µs
   pro Schritt für SHAKE/RATTLE; Schritt 932 → 785 µs. 32 × 2 k: 12 534 → 13 804 ns/Tag.

**Profil pro Schritt (32 × 2 k, vor Optimierung 2):** Nonbonded 428 µs, RATTLE 3 × 63 µs, SHAKE 2 × 63 µs,
Torsionen 40 µs, übrige Kernels < 20 µs.

**Grenzen:** Ohne Zellliste wächst der Aufwand je Replica mit N²; für ein einzelnes großes System
(1 × 20 k) ist OpenMM mit Nachbarliste schneller. Zellliste/Nachbarliste: Phase 9+ (ARCHITECTURE §6).
Nsight Compute (Counter-Rechte) weiterhin ausstehend.
