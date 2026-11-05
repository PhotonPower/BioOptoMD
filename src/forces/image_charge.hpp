// BioOptoMD – ImageChargeForce: abgeschirmte Bildladungen an einer leitfähigen Ebene z = z_im
// (ADR 0013, PHYSICS_MODEL.md §4.4). NumPy-Referenz: tests/reference/image_charge_ref.py.
#pragma once

#include <cstdint>
#include <string_view>
#include <vector>

#include "biooptomd/device_buffer.hpp"
#include "biooptomd/plugin.hpp"

namespace bomd {

/// Untergrenze für ζ_c [nm]: s >= 2 ζ_c hält s² weit über EPS_R2 (AGENTS Regel 4) und die
/// Eigenbildkraft unter k_e q²/(4 ε_r ζ_c²) ≈ 8.7·10⁴ q²/ε_r kJ/mol/nm < F_max (ADR 0008).
inline constexpr float kMinZetaClamp = 0.02f;

struct ImageChargeParams {
  float z_image_nm = 0.0f;  ///< Bildebene z_im [nm] (ADR 0013 Punkt 3)
  /// Bildfaktor f ∈ [−1, 1]: 1 = idealer Leiter (exakt im DH-Modell), sonst Näherung (OQ-18).
  float factor = 1.0f;
  std::vector<float> kappa_per_nm;  ///< κ_r je Replica, >= 0
  std::vector<float> eps_r;         ///< ε_r (bei GB ε_out) je Replica, > 0
  float r_switch_nm = 1.8f;         ///< r_s wie die Nonbonded-Strategie, 0 < r_s < r_c
  float r_cutoff_nm = 2.0f;         ///< r_c wie die Nonbonded-Strategie (ADR 0006)
  /// ζ_c >= kMinZetaClamp: ζ = max(z − z_im, ζ_c) (ADR 0013 Punkt 6). ASSUMPTION(OQ-18): 0.1 nm;
  /// an Gold mit d_im = 0 erreichen H-Atome ζ < ζ_c regelmäßig, ζ_c ist dort ein Modellparameter.
  float zeta_clamp_nm = 0.1f;
  bool always_image_sum = false;  ///< true ⇒ Bildsummenpfad auch für L >= 2 r_c (nur Tests)
};

/// Ein Tile-Pass über alle Paare I ≤ J je Replica (Eigenbilder in Diagonal-Tiles), ohne
/// Exclusions. Energie in Term kEnergyImage.
class ImageChargeForce final : public ForcePlugin {
 public:
  explicit ImageChargeForce(ImageChargeParams params);

  std::string_view name() const override { return "ImageChargeForce"; }
  void initialize(SimulationContext& ctx) override;
  void compute(DeviceState& s, cudaStream_t stream) override;
  void set_screening(const std::vector<float>& kappa, const std::vector<float>& eps_r,
                     cudaStream_t stream) override;

  const ImageChargeParams& params() const { return params_; }
  /// true, wenn compute() über periodische Bilder summiert (eine Box < 2 r_c oder erzwungen).
  bool uses_image_sum() const;
  /// Anzahl geladener Atome mit z − z_im <= ζ_c, summiert über Replicas und alle compute() seit
  /// reset_counters() (synchronisiert).
  std::uint64_t clamp_count() const;
  void reset_counters();

 private:
  ImageChargeParams params_;
  SimulationContext* ctx_ = nullptr;
  int n_tile_pairs_ = 0;
  DeviceBuffer<int2> d_tile_pairs_;             // (I, J), I ≤ J
  DeviceBuffer<float2> d_screening_;            // (κ, ε_r) je Replica
  DeviceBuffer<float4> d_tile_bounds_;          // (Mittelpunkt, Halbbreite) je (Replica, Tile)
  DeviceBuffer<unsigned long long> d_counter_;  // geclampte Atome
};

}  // namespace bomd
