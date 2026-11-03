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
