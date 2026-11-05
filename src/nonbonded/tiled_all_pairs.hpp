// BioOptoMD – TiledAllPairs: nicht-kovalente Wechselwirkungen (LJ + Debye-Hückel, Switching,
// Minimum Image in x/y bzw. Summe über periodische Bilder bei L < 2 r_c, ADR 0006) über alle
// 32×32-Tile-Paare je Replica (ROADMAP Phase 4, PHYSICS_MODEL.md §3). NumPy-Referenz:
// tests/reference/nonbonded_ref.py. Builder: python/biooptomd/nonbonded.py.
#pragma once

#include <cstdint>
#include <vector>

#include "biooptomd/device_buffer.hpp"
#include "biooptomd/plugin.hpp"

namespace bomd {

/// Eine Nonbonded-Topologie. Erzeugung: biooptomd.nonbonded.nonbonded_topology().
struct NonbondedTopology {
  /// Anzahl LJ-Typen T (Typ 0 = Padding, ε = 0), 1 ≤ T ≤ 64
  int n_types = 0;
  /// (T, T, 2): (σ_ij [nm], ε_ij [kJ/mol])
  std::vector<float> lj_table;
  /// (n_tile_pairs,): Index in masks oder −1
  std::vector<int> mask_index;
  /// (n_mask_tiles, 32): Ausschluss-Bitmasken
  std::vector<std::uint32_t> masks;
};

struct TiledAllPairsParams {
  std::vector<NonbondedTopology> topologies;
  std::vector<int> replica_topology;  ///< (n_replicas,)
  std::vector<float> kappa_per_nm;    ///< Debye-Hückel κ je Replica, >= 0
  std::vector<float> eps_r;           ///< relative Permittivität je Replica, > 0
  float r_switch_nm = 1.8f;           ///< r_s, 0 < r_s < r_c
  /// r_c [nm]; Builder: biooptomd.nonbonded.cutoff_from_kappa (ADR 0006). Bei L_x oder L_y < 2 r_c
  /// summiert der Kernel über periodische Bilder.
  float r_cutoff_nm = 2.0f;
  /// true ⇒ Bildpfad auch für L >= 2 r_c (nur für Tests; Ergebnis gleich Minimum Image).
  bool always_image_sum = false;
};

/// Nicht final: TiledGeneralizedBorn (ADR 0010) nutzt Topologie-, Masken- und Hüllenpuffer.
class TiledAllPairs : public NonbondedStrategy {
 public:
  explicit TiledAllPairs(TiledAllPairsParams params);

  void initialize(SimulationContext& ctx) override;
  /// Kräfte in d_force_fp, Energie in kEnergyNonbonded. Wirft std::logic_error ohne Box.
  void compute(DeviceState& s, cudaStream_t stream) override;
  /// κ_r und ε_r (bei GB: ε_out) je Replica neu setzen; r_c bleibt (ADR 0006: Maximum über alle
  /// Zustände bereits beim Aufbau, ADR 0011).
  void set_screening(const std::vector<float>& kappa, const std::vector<float>& eps_r,
                     cudaStream_t stream) override;
  /// true, wenn compute() über periodische Bilder summiert (eine Box < 2 r_c oder erzwungen).
  bool uses_image_sum() const;

  int n_tile_pairs() const { return n_tile_pairs_; }
  const TiledAllPairsParams& params() const { return params_; }

 protected:
  TiledAllPairsParams params_;
  SimulationContext* ctx_ = nullptr;
  int n_tile_pairs_ = 0;
  int max_types_ = 0;
  DeviceBuffer<int2> d_tile_pairs_;      // (I, J), I ≤ J
  DeviceBuffer<float2> d_lj_;            // alle Tabellen konkateniert
  DeviceBuffer<int> d_mask_index_;       // n_tile_pairs je Topologie
  DeviceBuffer<std::uint32_t> d_masks_;  // alle Masken konkateniert
  DeviceBuffer<int4> d_replica_;         // (lj_offset, n_types, mask_index_offset, mask_base)
  DeviceBuffer<float2> d_screening_;     // (κ, ε_r) je Replica
  DeviceBuffer<float4> d_tile_bounds_;   // (Mittelpunkt, Halbbreite) je (Replica, Tile)
};

}  // namespace bomd
