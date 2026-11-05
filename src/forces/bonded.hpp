// BioOptoMD – BondedForce: Bonds, Angles, periodische Torsionen/Impropers und 1-4-Paare
// (LJ + Debye-Hückel, skaliert) nach docs/PHYSICS_MODEL.md §2. NumPy-Referenz:
// tests/reference/bonded_ref.py. Topologie pro Replica über Offsets (ADR 0005).
#pragma once

#include <string_view>
#include <vector>

#include "biooptomd/device_buffer.hpp"
#include "biooptomd/plugin.hpp"

namespace bomd {

/// Eine Topologie (ein Molekülsystem). Atomindizes sind lokal (< n_atoms), row-major.
struct BondedTopology {
  std::vector<int> bonds;            ///< (n_b, 2)
  std::vector<float> bond_k;         ///< k [kJ/mol/nm²]
  std::vector<float> bond_r0;        ///< r0 [nm]
  std::vector<int> angles;           ///< (n_a, 3), Scheitel in der Mitte
  std::vector<float> angle_k;        ///< k [kJ/mol/rad²]
  std::vector<float> angle_theta0;   ///< θ0 [rad]
  std::vector<int> torsions;         ///< (n_t, 4), Impropers ebenfalls periodisch
  std::vector<float> torsion_k;      ///< k [kJ/mol]
  std::vector<int> torsion_n;        ///< Periodizität n >= 1
  std::vector<float> torsion_phase;  ///< φ0 [rad]
  std::vector<int> pairs14;          ///< (n_p, 2)
  std::vector<float> pair_qq;        ///< skaliertes Ladungsprodukt [e²]
  std::vector<float> pair_sigma;     ///< σ [nm] (kombiniert)
  std::vector<float> pair_eps;       ///< ε [kJ/mol] (kombiniert, skaliert)
};

struct BondedParams {
  std::vector<BondedTopology> topologies;
  std::vector<int> replica_topology;  ///< Topologie-Index je Replica, (n_replicas,)
  std::vector<float> kappa_per_nm;    ///< κ der 1-4-Debye-Hückel-Abschirmung je Replica, >= 0
  std::vector<float> eps_r;           ///< relative Permittivität je Replica, > 0
  /// true: 1-4-Abschirmung folgt set_screening (Debye-Hückel); false: fest (z. B. GB, κ = 0, ε =
  /// 1).
  bool follow_screening = true;
};

/// Alle Paarvektoren mit Minimum Image in x/y (ADR 0003). Energien in kEnergyBond, kEnergyAngle,
/// kEnergyTorsion, kEnergyPair14.
class BondedForce final : public ForcePlugin {
 public:
  explicit BondedForce(BondedParams params);

  std::string_view name() const override { return "BondedForce"; }
  /// Prüft Indizes und lädt die Topologien hoch.
  void initialize(SimulationContext& ctx) override;
  void compute(DeviceState& s, cudaStream_t stream) override;

  void set_screening(const std::vector<float>& kappa, const std::vector<float>& eps_r,
                     cudaStream_t stream) override;

  /// Anzahl der Terme (Bonds, Angles, Torsionen, 1-4) einer Replica.
  std::vector<int> term_counts(int replica) const;

 private:
  /// Ein Term-Typ: konkatenierte Einträge aller Topologien + (begin, count) je Replica.
  struct TermBuffers {
    DeviceBuffer<int4> index;
    DeviceBuffer<float4> param;
    DeviceBuffer<int2> range;
    std::vector<int2> h_range;
    int max_count = 0;
  };

  BondedParams params_;
  TermBuffers bonds_, angles_, torsions_, pairs_;
  DeviceBuffer<float2> d_screening_;  // (κ, ε_r) je Replica
  std::vector<float2> h_screening_;   // Host-Kopie
};

}  // namespace bomd
