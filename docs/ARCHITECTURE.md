# Architektur

## 1. Überblick

```
Python (biooptomd)                       C++ Host                        CUDA Device
──────────────────                       ────────                        ───────────
builder  ── Topologie (ParmEd/OpenMM)    SimulationContext               DeviceState (SoA, gepaddet)
   │        → gepaddete SoA-Arrays  ───▶   ├─ Topology (pro Replica)       d_pos, d_vel, d_force_fp
Simulation ── step(n), get_*() ─────▶     ├─ Integrator                    Bonded-/Nonbonded-Kernels
observers  ◀─ DLPack / NumPy  ◀────────   ├─ NonbondedStrategy             ForcePlugin-Kernels
                                           ├─ vector<ForcePlugin>          Observer-Kernels
                                           └─ vector<ObserverPlugin>       Akkumulatoren im VRAM
```

Ein Zeitschritt (BAOAB, siehe `PHYSICS_MODEL.md`):

```
B  v += (dt/2) F/m
A  x += (dt/2) v          → Constraints (ab Phase 5)
O  v = c1 v + c2 ξ        (Philox, T pro Replica)
A  x += (dt/2) v          → Constraints, Wrap in x/y
   clear_forces()
   bonded()  nonbonded()  for p in force_plugins: p.compute()
   convert_fixed_point_to_float()
B  v += (dt/2) F/m        → Geschwindigkeits-Constraints
   if step % stride == 0: for o in observers: o.observe()
```

Alles läuft auf **einem** `cudaStream_t`; die Reihenfolge der Kernel-Launches ist die Synchronisation.

## 2. Speicherlayout

### 2.1 Indizierung
- `N_replicas` Replicas, jede mit maximal `N_atoms` echten Atomen.
- `N_pad = ceil(N_atoms / 32) * 32`. Globaler Index: `i_global = r * N_pad + i_local`.
- Alle Replicas haben dieselbe `N_pad`. Unterschiedliche echte Atomzahlen (z. B. Mutationen) werden mit Dummy-Atomen aufgefüllt (AGENTS.md Regel 6).

### 2.2 Zustands-Arrays (`DeviceState`)

| Array | Typ | Inhalt | Größe |
|---|---|---|---|
| `d_pos` | `float4` | x, y, z [nm], q [e] | `N_replicas·N_pad` |
| `d_pos_corr` | `float4` | Positionskorrektur: x = d_pos.xyz + d_pos_corr.xyz (ADR 0009) | `N_replicas·N_pad` |
| `d_vel` | `double4` | vx, vy, vz [nm/ps], 1/m [1/amu] (FP64, ADR 0009) | `N_replicas·N_pad` |
| `d_force_fp` | `long long` ×3 (SoA: fx, fy, fz) | Fixed-Point-Kraft, Skala 2³² | `3·N_replicas·N_pad` |
| `d_force` | `float4` | Fx, Fy, Fz [kJ/mol/nm], frei | `N_replicas·N_pad` |
| `d_energy_fp` | `long long` | Potentielle Energie pro Replica und Term | `N_replicas·N_TERMS` |
| `d_types` | `int2` | (LJ-Typ, Optik/Oberflächen-Typ) | `N_replicas·N_pad` |
| `d_box` | `float2` | (Lx, Ly) [nm] pro Replica | `N_replicas` |
| `d_temperature` | `float` | T_r [K] | `N_replicas` |
| `d_image` | `int2` | Box-Imagezähler x/y (für ungewrapte Koordinaten) | `N_replicas·N_pad` |
| `d_status` | `unsigned int` | Statusbits (Sättigung, NaN, Wrap-Bereich), ADR 0008 | 1 |

**Entscheidung:** Parameter- und Topologie-Arrays (Bindungslisten, LJ-Tabellen, Exclusions) sind **pro Replica** gespeichert (ADR 0005). Gemeinsame Topologie ist der Spezialfall, in dem alle Replicas auf denselben Puffer zeigen (Offset-Array `d_topo_offset[r]`).

### 2.3 Fixed-Point-Kräfte
`F_fp = (long long)(F * 2^32)`, Akkumulation mit `atomicAdd((unsigned long long*)…)`. Nach allen Kraft-Kernels konvertiert ein Kernel nach `d_force` (float). Begründung: Determinismus und Präzision bei FP32-Kernels, ADR 0002.

## 3. Host-Schnittstellen (C++)

```cpp
namespace bomd {

struct DeviceState;          // nur Pointer + Dimensionen, trivially copyable → als Kernel-Argument nutzbar
class  SimulationContext;    // besitzt Speicher, Stream, Plugins

class ForcePlugin {
public:
  virtual ~ForcePlugin() = default;
  virtual std::string_view name() const = 0;
  virtual void initialize(SimulationContext&) {}          // Speicher allokieren, Parameter hochladen
  virtual void compute(DeviceState& s, cudaStream_t st) = 0;  // MUSS in d_force_fp akkumulieren
  virtual bool contributes_energy() const { return true; }
};

class ObserverPlugin {
public:
  virtual ~ObserverPlugin() = default;
  virtual std::string_view name() const = 0;
  virtual int  stride() const = 0;                        // alle n Schritte
  virtual void initialize(SimulationContext&) {}
  virtual void observe(const DeviceState& s, std::int64_t step, cudaStream_t st) = 0;  // long ist unter Windows 32 bit
  virtual void reset() = 0;                                // Akkumulatoren leeren
  virtual ObserverResult download() const = 0;             // Device → Host, nur auf Anfrage
};

class NonbondedStrategy {   // Strategy-Pattern: TiledAllPairs (V1), CellList (V2)
public:
  virtual ~NonbondedStrategy() = default;
  virtual void initialize(SimulationContext&) {}
  virtual void compute(DeviceState&, cudaStream_t) = 0;
};

class ElectrostaticsModel;  // DebyeHuckel (V1), GeneralizedBornOBC (V2) – als Template-Parameter
                            // des Nonbonded-Kernels, NICHT virtuell im Device-Code.
}
```

Virtuelle Funktionen existieren **nur** auf dem Host. Jedes Plugin startet eigene Kernels.

Implementierung (Phase 1): `include/biooptomd/plugin.hpp`, `simulation_context.hpp`, `device_state.hpp`.
`SimulationContext::compute_forces()` führt `clear_forces → Nonbonded → ForcePlugins → fixed_point_to_float` aus.
Die Spalten von `d_energy_fp` sind das Enum `bomd::EnergyTerm` (`kNumEnergyTerms` = N_TERMS).

### 3.1 Python-Export der Zustandsarrays (Phase 1)

`biooptomd._core.SimulationContext` exportiert jeden Puffer im **rohen, gepaddeten Layout** aus §2.2
(z. B. `positions()` → `float32 (n_replicas, n_pad, 4)`, `forces_fixed_point()` → `int64 (3, n_replicas, n_pad)`):

- `x()` bzw. `x(dlpack=False)`: NumPy-Kopie (synchronisiert den Stream).
- `x(dlpack=True)`: Zero-Copy-Sicht auf den Gerätespeicher (`__dlpack__`, z. B. `cupy.from_dlpack`). Der Stream wird
  vorher synchronisiert; die Sicht hält den Kontext am Leben. Spätere Kernels des Kontexts verändern die Daten, und
  fremde Streams sind nicht automatisch geordnet (der Kontext-Stream ist blockierend, also mit dem Legacy-Default-Stream
  geordnet). Bei Bedarf `ctx.synchronize()`.

Ein DLPack-Export markiert die Kräfte als veraltet (`invalidate_forces`), weil die Sicht beschreibbar ist.

Setter (`set_positions`, `set_charges`, `set_masses`, …) nehmen nur echte Atome (`(n_replicas, n_atoms, …)`) und
lassen Padding-Atome unverändert. Test-Hooks ohne Produktionsphysik liegen in `biooptomd._core.testing`.

## 4. Module

| Verzeichnis | Inhalt | Phase |
|---|---|---|
| `include/biooptomd/` | Öffentliche Header: `units.hpp`, `device_state.hpp`, `plugin.hpp`, `cuda_check.hpp` | 1 |
| `src/core/` | `SimulationContext`, Speicherverwaltung, Fixed-Point-Konvertierung, Integrator, Constraints | 1, 2, 5 |
| `src/forces/` | Bonded-Kernels; Plugins `PlanarSurfaceForce`, `HarmonicRestraintPullingForce`, `VoxelGridForce3D`, `ImageChargeForce` (ADR 0013) | 2, 3, 6, 9 |
| `src/nonbonded/` | `TiledAllPairs` (LJ + Debye-Hückel, Exclusion-Bitmasken, Tile-Vorauswahl), `TiledGeneralizedBorn` (LJ + Coulomb + GB-OBC2, ADR 0010), gemeinsame Bausteine `tile_common.cuh`; später `CellList` | 4, 8, 9 |
| `src/observers/` | `DipoleAndDistanceTracker`, `FRETTracker`, `RefractiveIndexProfiler` | 6 |
| `src/bindings/` | nanobind-Modul `_core`, DLPack-Export | 1 ff. |
| `python/biooptomd/` | `units`, `builder`, `simulation`, `observers`, `io`, `analysis` | 1 ff. |
| `tests/python/` | pytest, Marker `gpu`, `slow`, `reference` | alle |
| `tests/cpp/` | GoogleTest für Host-Logik, die ohne Python testbar sein soll | optional |
| `tests/reference/` | NumPy-Referenzimplementierungen, OpenMM-Generatoren | alle |
| `benchmarks/` | Durchsatz vs. OpenMM | 8 |

## 5. Python-API (Zielbild)

```python
import biooptomd as bomd

topo = bomd.builder.from_parmed(structure, hmr=True)          # → gepaddete SoA-Arrays
sim  = bomd.Simulation(topo, n_replicas=32, box_xy=(8.0, 8.0),
                       temperatures=bomd.geometric_ladder(290, 400, 32),
                       dt=0.004, friction=1.0, seed=42)
sim.add_force(bomd.forces.PlanarSurface(material="graphene", solvent_scale=..., phi0=0.05, kappa=...))
sim.add_force(bomd.forces.HarmonicRestraint(atoms=anchor, k=5000.0))
dip = sim.add_observer(bomd.observers.DipoleAndDistance(atom_a=..., atom_b=..., stride=100))
nz  = sim.add_observer(bomd.observers.RefractiveIndexProfile(dz=0.05, z_max=10.0,
                       wavelength_nm=633.0, stride=100))
sim.step(1_000_000)
mu_t   = dip.result()        # (n_replicas, n_frames, 7) = (R_F [nm], μ̂, d [nm]), s. PHYSICS_MODEL §6.1
n_of_z = nz.result()         # (n_replicas, n_bins)
x = sim.positions(dlpack=True)   # Zero-Copy nach CuPy/PyTorch
```

## 6. Erweiterungspunkte (geplant, nicht jetzt implementieren)

- `NonbondedStrategy::CellList` für > 10 k Atome.
- `ElectrostaticsModel::GeneralizedBornOBC`.
- ~~`ImageChargeForce` für leitfähige Oberflächen~~ – umgesetzt in Phase 9.5 (ADR 0013).
- `ReplicaExchange` (Host-gesteuerter Austausch von Temperaturen, nicht von Koordinaten).
- Coarse-Grained-Kraftfelder (MARTINI, ENM) – nutzen dieselben Bonded-Kernels.

### 3.2 Integration, Energien und Plugins (Phase 2)

- `set_integrator(dt, friction, seed)` + `step(n)`; Schrittfolge exakt wie in §1. Observer feuern nach dem
  abschließenden B, wenn `step % stride == 0`. Die Simulationszeit wird fortlaufend aufsummiert
  (`time`, `previous_time` für Protokoll-Arbeit).
- Rauschen: Philox4x32-10 (`src/core/philox.cuh`), Zähler `(i_local, replica, step_lo, step_hi[23:0] | purpose<<24)`,
  Schlüssel `(seed_lo, seed_hi)`; Box-Muller mit 23-Bit-Uniforms.
- `energies()` → `(n_replicas, 1 + N_ENERGY_TERMS)`: Spalte 0 = E_kin (FP64-Reduktion), dann E_pot je Term
  (`ENERGY_TERMS`). `EnergyTracker` (`add_energy_tracker`) speichert dieselben Frames im VRAM.
- Plugins über Fabrikmethoden, die eine an den Kontext gebundene Referenz liefern:
  `add_planar_surface`, `add_position_restraints`, `add_com_pulling`.
- Energiebeiträge pro Atom werden warpweise in Fixed-Point summiert (`warp_add_energy`), dann ein Atomic pro Warp.


### 3.3 Nicht-kovalente Terme (Phase 4)

- `NonbondedStrategy`-Implementierung `TiledAllPairs` (`src/nonbonded/`), gesetzt über
  `ctx.set_nonbonded_tiled(topologies, replica_topology, kappa, eps_r, r_switch, r_cutoff)`.
- Ein Warp pro 32×32-Tile-Paar (I ≤ J), `blockIdx.y` = Replica; j-Atome im Shared Memory, rotierender
  Zugriff (konfliktfreie j-Kraftakkumulation), Kräfte je Tile in FP32 summiert, dann Fixed-Point-Atomics.
- Elektrostatik als Template-Parameter (`detail::DebyeHuckel`), vorbereitet für `GeneralizedBornOBC`.
- Topologie je Replica (ADR 0005): LJ-Typtabelle (T × T, (σ, ε), Typ 0 = Padding mit ε = 0, im Shared
  Memory) und Exclusion-Bitmasken je Tile-Paar (`mask_index` −1 = keine). Erzeugung im Python-Builder
  `biooptomd.nonbonded.nonbonded_topology()`; LJ-Typen der Atome in `types[..., 0]`.

### 3.4 Constraints und HMR (Phase 5)

- `ConstraintSet` (`src/core/constraints.*`) gehört zum `SimulationContext` (kein ForcePlugin, weil es in
  die Integrator-Teilschritte eingreift): `ctx.set_constraints(topologies, replica_topology, tolerance)`.
- Mit Constraints läuft BAOAB in Teilschritten (Kernels B, A, O + Cluster-Kernels SHAKE/RATTLE); ohne
  Constraints bleibt der fusionierte Kernel (bitidentisch zu Phase 2).
- Python-Builder `biooptomd.constraints`: `repartition_hydrogen_mass` (HMR), `constraint_topology`
  (Cluster + kovalente Topologie ohne constrainte Bindungen).

### 3.5 Optik-Observer und Voxel-Potential (Phase 6)

- Observer (`src/observers/optics.*`) mit Ringpuffer im VRAM (`DeviceRingBuffer`: älteste Frames werden
  überschrieben, `overwritten` zählt sie; Download chronologisch, nur auf Anfrage):
  `add_dipole_tracker` (R_F, μ̂, d), `add_fret_tracker` (R_DA, κ², optional E), `add_refractive_index_profiler`
  (S_α, S_V je Schicht, Mittel und Varianz über Frames; n(z) nach Lorentz-Lorenz beim Download).
- `ctx.observe_now()` führt alle Observer am aktuellen Zustand aus (statische Auswertungen, Tests).
- Gruppen/Achsen der Sonden pro Replica als CSR (ADR 0005); Schwerpunkte in FP64 aus der Mixed-Precision-
  Position und den Imagezählern (ungewrappt).
- `VoxelGridForce3D` (`src/forces/voxel_grid.*`, `ctx.add_voxel_grid`): 3D-Textur (float4: V, ∇V) mit
  Point-Sampling und FP32-Trilinearinterpolation im Kernel (ASSUMPTION: Hardware-Filterung hat nur 9-Bit-
  Gewichte), Skalierung je Optik-Typ, optional periodisch in x/y. Builder-Hilfe `biooptomd.voxel`.
