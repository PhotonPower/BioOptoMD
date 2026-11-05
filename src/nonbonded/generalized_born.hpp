// BioOptoMD – TiledGeneralizedBorn: LJ + Coulomb + Generalized Born OBC2 (ADR 0010, AMBER igb = 5)
// oder GB-neck2 (ADR 0012, igb = 8) mit Salz und optionalem ACE-Term, Switching/Cutoff aus κ und
// Bildsumme wie ADR 0006.
// NumPy-Referenz: tests/reference/gb_ref.py.
#pragma once

#include <vector>

#include "biooptomd/device_buffer.hpp"
#include "nonbonded/tiled_all_pairs.hpp"

namespace bomd {

struct GeneralizedBornParams {
  /// Topologien, Replica-Zuordnung, κ_r [1/nm], ε_out = eps_r je Replica, r_s, r_c wie
  /// TiledAllPairs.
  TiledAllPairsParams base;
  /// Intrinsischer Radius ρ [nm] je Topologie und Atom (n_atoms), > OFFSET (0.009 nm).
  std::vector<std::vector<float>> radius_nm;
  /// HCT-Skalierungsfaktor s je Topologie und Atom (n_atoms), >= 0.
  std::vector<std::vector<float>> scale;
  /// Radius-Offset δ [nm]: or = ρ − δ (OBC2 0.009, GBn2 0.0195141); gilt für alle Topologien
  /// (ein Modell je Strategie).
  float offset_nm = 0.009f;
  /// (α, β, γ) des Born-Radius-tanh je Topologie und Atom (n_atoms·3); leer ⇒ OBC2 (1, 0.8, 4.85).
  std::vector<std::vector<float>> alpha_beta_gamma;
  /// GBn2-Neck (ADR 0012): Radiusindex je Topologie und Atom (n_atoms), Tabellen m0 [1/nm] und
  /// d0 [nm] (U·U, Index idx_i·U + idx_j); neck_size U je Topologie, 0 ⇒ kein Neck-Term.
  std::vector<std::vector<int>> neck_index;
  std::vector<std::vector<float>> neck_m0, neck_d0;
  std::vector<int> neck_size;
  float eps_in = 1.0f;  ///< Dielektrizitätskonstante des gelösten Moleküls
  bool ace = true;      ///< unpolarer ACE-Term (4πγ = 28.3919551 kJ/mol/nm², Sonde 0.14 nm)
};

/// Fünf Passes je Kraftberechnung (ADR 0010 Punkt 3): Born-Integrale (Tile), Born-Radien + Selbst-
/// und ACE-Energie (elementweise), Paarterme bei festen B (Tile), Kettenregel über die Born-Radien
/// (Tile). Alle Akkumulationen über Fixed-Point ⇒ deterministisch. Energie in kEnergyNonbonded.
class TiledGeneralizedBorn final : public TiledAllPairs {
 public:
  explicit TiledGeneralizedBorn(GeneralizedBornParams params);

  void initialize(SimulationContext& ctx) override;
  void compute(DeviceState& s, cudaStream_t stream) override;

  const GeneralizedBornParams& gb_params() const { return gb_; }
  /// Born-Radien B [nm] der letzten Kraftberechnung, (n_replicas·n_pad), Padding 0
  /// (synchronisiert).
  std::vector<float> born_radii() const;

 private:
  GeneralizedBornParams gb_;
  DeviceBuffer<float4> d_gb_atom_;      // (or, sr, ρ, Neck-Index) je Topologie-Atom (n_pad)
  DeviceBuffer<float4> d_gb_abg_;       // (α, β, γ, 0) je Topologie-Atom (n_pad)
  bool use_abg_ = false;                // false ⇒ OBC2-Konstanten (double) im Kernel
  DeviceBuffer<float2> d_neck_;         // (m0, d0) aller Topologien konkateniert
  DeviceBuffer<int2> d_neck_info_;      // (Offset in d_neck_, U) je Replica
  DeviceBuffer<int> d_gb_offset_;       // Offset in d_gb_atom_ je Replica
  DeviceBuffer<long long> d_integral_;  // I_i (Fixed-Point) je Atomplatz
  DeviceBuffer<long long> d_de_db_;     // ∂E/∂B_i (Fixed-Point) je Atomplatz
  DeviceBuffer<float> d_born_;          // B_i [nm]
  DeviceBuffer<float> d_db_di_;         // dB_i/dI_i [nm²]
};

}  // namespace bomd
