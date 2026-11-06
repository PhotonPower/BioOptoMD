// BioOptoMD – DeviceState: nicht-besitzende Sicht auf alle Zustands-Arrays einer Batch-Simulation.
// Layout exakt nach docs/ARCHITECTURE.md §2. Wird per Wert an Kernels übergeben.
#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <type_traits>

namespace bomd {

/// Atome pro Kachel / Warp. N_pad ist ein Vielfaches davon (ARCHITECTURE.md §2.1).
inline constexpr int kTileSize = 32;

/// Rundet eine Atomzahl auf das nächste Vielfache von kTileSize auf.
constexpr int padded_atom_count(int n_atoms) {
  return ((n_atoms + kTileSize - 1) / kTileSize) * kTileSize;
}

/// Terme der potentiellen Energie; Spalten von d_energy_fp (pro Replica N_TERMS Einträge).
/// Interface-Stub: die zugehörigen Kraftterme entstehen in Phase 2–6.
/// Vier FP64-Werte mit 32-Byte-Alignment (eine Ladeoperation pro Zugriff). Ersetzt CUDAs
/// `double4`, das ab CUDA 13 als deprecated markiert ist (ADR 0009).
struct alignas(32) Vec4d {
  double x, y, z, w;
};
static_assert(sizeof(Vec4d) == 32, "Vec4d muss 32 Byte groß sein");

/// Bits des Geräte-Statusworts DeviceState::d_status (ADR 0008), per atomicOr gesetzt.
enum StatusBit : unsigned int {
  kStatusFixedPointSaturated = 1u,  ///< ein Kraft-/Energiebeitrag wurde auf ±2^30 gesättigt
  kStatusNonFinite = 2u,            ///< ein Beitrag war NaN (als 0 gezählt)
  kStatusWrapRange = 4u,            ///< wrap_xy mit |x/L| >= 2^20 (FP32-Wrap ungenau)
  kStatusUnknownType =
      8u,  ///< Typindex außerhalb einer Parametertabelle (LJ: als Typ 0; Optik/Voxel: kein Beitrag)
  kStatusConstraintFailure = 16u,  ///< SHAKE/RATTLE nicht konvergiert (max_iterations erreicht)
};

enum EnergyTerm : int {
  kEnergyBond = 0,
  kEnergyAngle,
  kEnergyTorsion,
  kEnergyPair14,
  kEnergyNonbonded,
  kEnergySurface,
  kEnergyRestraint,
  kEnergyExternal,  // z. B. VoxelGridForce3D
  kEnergyImage,     // ImageChargeForce (ADR 0013)
  kNumEnergyTerms
};

/// Nur Pointer + Dimensionen, besitzt nichts (Besitzer: SimulationContext).
///
/// Indizierung: i_global = r * n_pad + i_local, r ∈ [0, n_replicas), i_local ∈ [0, n_pad).
/// Atome mit i_local >= n_atoms sind Padding-Atome (q = 0, inv_mass = 0, LJ-Typ 0;
/// AGENTS.md Regel 6).
struct DeviceState {
  int n_replicas = 0;  ///< Anzahl Replicas
  int n_atoms = 0;     ///< echte Atome pro Replica
  int n_pad = 0;       ///< gepaddete Atome pro Replica (Vielfaches von kTileSize)

  float4* d_pos = nullptr;  ///< (x, y, z) [nm], q [e];                      n_replicas·n_pad
  /// Positionskorrektur (ADR 0009): x = d_pos.xyz + d_pos_corr.xyz [nm], w = 0; n_replicas·n_pad
  float4* d_pos_corr = nullptr;
  Vec4d* d_vel = nullptr;  ///< (vx, vy, vz) [nm/ps], 1/m [1/amu] in FP64 (ADR 0009)
  /// Fixed-Point-Kraft (Skala 2^32, kJ/mol/nm), SoA: [fx(alle) | fy(alle) | fz(alle)],
  /// je n_replicas·n_pad Einträge (ADR 0002).
  long long* d_force_fp = nullptr;
  float4* d_force = nullptr;         ///< (Fx, Fy, Fz) [kJ/mol/nm], w frei;  n_replicas·n_pad
  long long* d_energy_fp = nullptr;  ///< Fixed-Point-Energie [kJ/mol], n_replicas·kNumEnergyTerms
  int2* d_types = nullptr;           ///< (LJ-Typ, Optik-/Oberflächen-Typ); n_replicas·n_pad
  float2* d_box = nullptr;           ///< (Lx, Ly) [nm];                     n_replicas
  float* d_temperature = nullptr;    ///< T_r [K];                           n_replicas
  int2* d_image = nullptr;           ///< Box-Imagezähler (nx, ny);          n_replicas·n_pad
  unsigned int* d_status = nullptr;  ///< Statuswort (StatusBit, ADR 0008);   1
  /// E_kin [kJ/mol] der Geschwindigkeiten nach O (+ RATTLE) im letzten Schritt, der sie erfasst hat
  /// (OQ-14, PHYSICS_MODEL §1); NaN vor dem ersten Schritt.                     n_replicas
  double* d_ekin_mid = nullptr;

  /// Gesamtzahl Atomplätze n_replicas·n_pad.
  __host__ __device__ int n_total() const { return n_replicas * n_pad; }

  __host__ __device__ long long* d_force_fp_x() const { return d_force_fp; }
  __host__ __device__ long long* d_force_fp_y() const { return d_force_fp + n_total(); }
  __host__ __device__ long long* d_force_fp_z() const { return d_force_fp + 2 * n_total(); }
};

static_assert(std::is_trivially_copyable_v<DeviceState>,
              "DeviceState muss trivially copyable sein (Kernel-Argument)");

}  // namespace bomd
