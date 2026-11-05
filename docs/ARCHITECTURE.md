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
| `d_vel` | `float4` | vx, vy, vz [nm/ps], 1/m [1/amu] | `N_replicas·N_pad` |
| `d_force_fp` | `long long` ×3 (SoA: fx, fy, fz) | Fixed-Point-Kraft, Skala 2³² | `3·N_replicas·N_pad` |
| `d_force` | `float4` | Fx, Fy, Fz [kJ/mol/nm], frei | `N_replicas·N_pad` |
| `d_energy_fp` | `long long` | Potentielle Energie pro Replica und Term | `N_replicas·N_TERMS` |
| `d_types` | `int2` | (LJ-Typ, Optik/Oberflächen-Typ) | `N_replicas·N_pad` |
| `d_box` | `float2` | (Lx, Ly) [nm] pro Replica | `N_replicas` |
| `d_temperature` | `float` | T_r [K] | `N_replicas` |
| `d_image` | `int2` | Box-Imagezähler x/y (für ungewrapte Koordinaten) | `N_replicas·N_pad` |

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

Setter (`set_positions`, `set_charges`, `set_masses`, …) nehmen nur echte Atome (`(n_replicas, n_atoms, …)`) und
lassen Padding-Atome unverändert. Test-Hooks ohne Produktionsphysik liegen in `biooptomd._core.testing`.

## 4. Module

| Verzeichnis | Inhalt | Phase |
|---|---|---|
| `include/biooptomd/` | Öffentliche Header: `units.hpp`, `device_state.hpp`, `plugin.hpp`, `cuda_check.hpp` | 1 |
| `src/core/` | `SimulationContext`, Speicherverwaltung, Fixed-Point-Konvertierung, Integrator, Constraints | 1, 2, 5 |
| `src/forces/` | Bonded-Kernels; Plugins `PlanarSurfaceForce`, `HarmonicRestraintPullingForce`, `VoxelGridForce3D`, später `ImageChargeForce` | 2, 3, 6, 9 |
| `src/nonbonded/` | `TiledAllPairs` (LJ + Debye-Hückel, Exclusion-Bitmasken), später `CellList`, GB-OBC | 4, 9 |
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
sim.add_force(bomd.forces.PlanarSurface(z0=0.0, eps_by_type=..., phi0=0.05, kappa=1.04))
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
- `ImageChargeForce` für leitfähige Oberflächen.
- `ReplicaExchange` (Host-gesteuerter Austausch von Temperaturen, nicht von Koordinaten).
- Coarse-Grained-Kraftfelder (MARTINI, ENM) – nutzen dieselben Bonded-Kernels.
