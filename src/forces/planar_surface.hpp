// BioOptoMD – PlanarSurfaceForce: 9-3-Wand + optional linearisiertes Gouy-Chapman bei z = z0
// (docs/PHYSICS_MODEL.md §4.1). NumPy-Referenz: tests/reference/surface_ref.py.
#pragma once

#include <cstdint>
#include <string_view>
#include <vector>

#include "biooptomd/device_buffer.hpp"
#include "biooptomd/plugin.hpp"

namespace bomd {

struct PlanarSurfaceParams {
  /// ε_w pro Oberflächen-Typ (Index = DeviceState::d_types.y) [kJ/mol]; ε = 0 ⇒ keine Wand.
  std::vector<float> epsilon_kj_mol;
  /// σ_w pro Oberflächen-Typ [nm], >= 0; σ = 0 ⇒ keine Wand.
  std::vector<float> sigma_nm;
  float z0_nm = 0.0f;         ///< Lage der Wand [nm]
  float clamp_factor = 0.1f;  ///< ζ_c >= clamp_factor·σ (> 0)
  /// Kraftschranke am Clamp (ADR 0008): ζ_c >= ζ_F mit |F_wall(ζ_F)| = max_force [kJ/mol/nm].
  float max_force_kj_mol_nm = 1.0e6f;
  float phi0_kj_mol_e = 0.0f;  ///< Φ0 [kJ/(mol·e)], 0 ⇒ kein Gouy-Chapman-Term
  float kappa_per_nm = 0.0f;   ///< inverse Debye-Länge κ [1/nm], >= 0
};

class PlanarSurfaceForce final : public ForcePlugin {
 public:
  explicit PlanarSurfaceForce(PlanarSurfaceParams params);

  std::string_view name() const override { return "PlanarSurfaceForce"; }
  void initialize(SimulationContext& ctx) override;
  /// Kräfte in d_force_fp, Energie in Term kEnergySurface.
  void compute(DeviceState& s, cudaStream_t stream) override;

  const PlanarSurfaceParams& params() const { return params_; }
  /// ζ_c,t = max(clamp_factor·σ_t, ζ_F,t) pro Typ [nm] (0 für Typen ohne Wand).
  const std::vector<float>& clamp_distances() const { return zeta_c_; }
  /// Anzahl Kraftauswertungen mit geclamptem ζ seit dem letzten reset_counters() (synchronisiert).
  std::uint64_t clamp_count() const;
  /// Anzahl Auswertungen mit Oberflächen-Typ >= Anzahl Typen (keine Wand angewendet).
  std::uint64_t unknown_type_count() const;
  void reset_counters();

 private:
  std::uint64_t counter(int index) const;

  PlanarSurfaceParams params_;
  SimulationContext* ctx_ = nullptr;
  std::vector<float> zeta_c_;
  DeviceBuffer<float4> type_params_;           // (ε, σ, ζ_c, 0) pro Typ
  DeviceBuffer<unsigned long long> counters_;  // [clamp, unknown_type]
};

}  // namespace bomd
