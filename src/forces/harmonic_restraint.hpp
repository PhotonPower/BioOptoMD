// BioOptoMD – HarmonicRestraintPullingForce (docs/PHYSICS_MODEL.md §4.2):
// Positions-Restraints und/oder Schwerpunkt-Pulling mit Arbeitsakkumulation auf dem Gerät.
// NumPy-Referenz: tests/reference/restraint_ref.py.
#pragma once

#include <array>
#include <cstdint>
#include <string_view>
#include <vector>

#include "biooptomd/device_buffer.hpp"
#include "biooptomd/plugin.hpp"

namespace bomd {

/// m Positions-Restraints V = ½ k Σ_d mask_d Δ_d² (Minimum Image in x/y).
struct PositionRestraintParams {
  std::vector<int> replica;                    ///< Replica je Restraint, (m,)
  std::vector<int> atom;                       ///< lokaler Atomindex (< n_atoms) je Restraint, (m,)
  std::vector<float> reference_nm;             ///< Referenzposition r⁰ [nm], (m, 3)
  std::vector<float> k_kj_mol_nm2;             ///< Federkonstante k [kJ/mol/nm²] >= 0, (m,)
  std::array<bool, 3> dims{true, true, true};  ///< wirksame Dimensionen (x, y, z)
};

/// Schwerpunkt-Pulling pro Replica r: V = ½ k_r (ê_r·(R_G − R₀,r) − v_r t)².
/// Umbrella-Modus: v_r = 0 und verschiedene R₀,r.
struct ComPullingParams {
  std::vector<int> group_offsets;     ///< CSR-Offsets (n_replicas + 1)
  std::vector<int> group_atoms;       ///< lokale Atomindizes der Gruppen, je Gruppe verschieden
  std::vector<float> r0_nm;           ///< R₀ [nm], (n_replicas, 3)
  std::vector<float> direction;       ///< Zugrichtung (wird normiert), (n_replicas, 3)
  std::vector<float> velocity_nm_ps;  ///< v [nm/ps], (n_replicas,)
  std::vector<float> k_kj_mol_nm2;    ///< k [kJ/mol/nm²] >= 0, (n_replicas,)
};

class HarmonicRestraintPullingForce final : public ForcePlugin {
 public:
  explicit HarmonicRestraintPullingForce(PositionRestraintParams restraints);
  explicit HarmonicRestraintPullingForce(ComPullingParams pulling);

  std::string_view name() const override { return "HarmonicRestraintPullingForce"; }
  /// Prüft Indizes; beim Pulling werden die Gewichte w_i = m_i/M_G aus den aktuellen Massen
  /// berechnet (Massen müssen vorher gesetzt sein und > 0).
  void initialize(SimulationContext& ctx) override;
  /// Kräfte in d_force_fp, Energie in Term kEnergyRestraint. Pulling: Ist seit dem letzten
  /// Aufruf genau ein Schritt vergangen, wird ΔW = V(x, t) − V(x, t − Δt) pro Replica addiert.
  void compute(DeviceState& s, cudaStream_t stream) override;

  bool is_pulling() const { return pulling_; }
  /// Akkumulierte Arbeit W [kJ/mol] pro Replica (nur Pulling; synchronisiert).
  std::vector<double> work() const;
  /// u = ê·(R_G − R₀) − v t [nm] pro Replica aus dem letzten compute (nur Pulling).
  std::vector<double> extension() const;
  void reset_work();

 private:
  /// w_i = m_i / M_G aus den aktuellen Massen auf dem Gerät (synchronisiert).
  void update_weights();

  bool pulling_;
  PositionRestraintParams restraints_;
  ComPullingParams pull_;
  SimulationContext* ctx_ = nullptr;
  std::int64_t last_step_ = -1;
  bool has_last_step_ = false;
  std::uint64_t masses_version_ = 0;

  // Positions-Restraints
  DeviceBuffer<int2> d_restraint_atom_;   // (i_global, replica)
  DeviceBuffer<float4> d_restraint_ref_;  // (x0, y0, z0, k)
  // Pulling
  DeviceBuffer<int> d_group_offsets_;
  DeviceBuffer<int> d_group_atoms_;
  DeviceBuffer<float> d_group_weights_;
  DeviceBuffer<float4> d_r0_k_;   // (R₀, k)
  DeviceBuffer<float4> d_dir_v_;  // (ê, v)
  DeviceBuffer<double> d_work_;   // W pro Replica
  DeviceBuffer<double> d_extension_;
};

}  // namespace bomd
