// BioOptoMD – TiledGeneralizedBorn: LJ + Coulomb + Generalized Born OBC2 mit Salz und optionalem
// ACE-Term (ADR 0010, AMBER igb = 5), Switching/Cutoff aus κ und Bildsumme wie ADR 0006.
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
  DeviceBuffer<float4> d_gb_atom_;      // (or, sr, ρ, 0) je Topologie-Atom, konkateniert (n_pad)
  DeviceBuffer<int> d_gb_offset_;       // Offset in d_gb_atom_ je Replica
  DeviceBuffer<long long> d_integral_;  // I_i (Fixed-Point) je Atomplatz
  DeviceBuffer<long long> d_de_db_;     // ∂E/∂B_i (Fixed-Point) je Atomplatz
  DeviceBuffer<float> d_born_;          // B_i [nm]
  DeviceBuffer<float> d_db_di_;         // dB_i/dI_i [nm²]
};

}  // namespace bomd
