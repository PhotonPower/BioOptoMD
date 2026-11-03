// BioOptoMD – H-Bindungs-Constraints (SHAKE/RATTLE), parallel über Cluster (Schweratom + H-Atome),
// docs/PHYSICS_MODEL.md §1, §5. NumPy-Referenz: tests/reference/constraints_ref.py.
#pragma once

#include <cuda_runtime.h>

#include <vector>

#include "biooptomd/device_buffer.hpp"
#include "biooptomd/device_state.hpp"

namespace bomd {

class SimulationContext;

/// Constraint-Cluster einer Topologie. Erzeugung: biooptomd.constraints.constraint_topology().
struct ConstraintTopology {
  /// (n_clusters, 4): (Schweratom, H1, H2, H3) lokale Indizes, fehlende H = −1
  std::vector<int> clusters;
  /// (n_clusters, 3): Sollabstände [nm], fehlende = 0
  std::vector<float> distances;
};

struct ConstraintParams {
  std::vector<ConstraintTopology> topologies;
  std::vector<int> replica_topology;  ///< (n_replicas,)
  /// Relative Längentoleranz |r − d|/d (PHYSICS_MODEL §5: 1e-6)
  double tolerance = 1e-6;
  /// RATTLE: |r·Δv| < velocity_tolerance · r² [1/ps]
  double velocity_tolerance = 1e-10;
  int max_iterations = 200;
};

/// Besitzt die Cluster-Listen pro Replica (ADR 0005) und startet die SHAKE/RATTLE-Kernels.
/// Ein Thread pro Cluster rechnet iterativ in FP64 auf der Mixed-Precision-Position (ADR 0007).
/// Nicht-Konvergenz setzt kStatusConstraintFailure.
class ConstraintSet {
 public:
  ConstraintSet(ConstraintParams params, const SimulationContext& ctx);

  /// A-Schritt (x += h·v) für Cluster-Atome, dann SHAKE mit den Bindungsrichtungen vor dem
  /// Schritt und Geschwindigkeitskorrektur v += Δx/h.
  void drift_and_shake(const DeviceState& s, double half_dt, cudaStream_t stream) const;
  /// Projiziert die aktuellen Positionen auf die Constraints (Referenz = aktuelle Positionen),
  /// ohne Geschwindigkeitskorrektur. Für Startstrukturen.
  void project_positions(const DeviceState& s, cudaStream_t stream) const;
  /// RATTLE: entfernt Relativgeschwindigkeiten entlang der Constraints.
  void rattle(const DeviceState& s, cudaStream_t stream) const;

  /// 1 für Atomplätze in einem Cluster (n_replicas·n_pad); der freie A-Schritt überspringt sie.
  const unsigned char* d_in_cluster() const { return d_in_cluster_.data(); }
  /// Anzahl Constraints je Replica (für N_dof).
  const std::vector<int>& constraint_counts() const { return counts_; }

 private:
  void launch_shake(const DeviceState& s, double half_dt, bool drift, cudaStream_t stream) const;

  ConstraintParams params_;
  int max_clusters_ = 0;
  std::vector<int> counts_;
  DeviceBuffer<int4> d_clusters_;
  DeviceBuffer<float4> d_distances_;
  DeviceBuffer<int2> d_range_;  // (begin, count) je Replica
  DeviceBuffer<unsigned char> d_in_cluster_;
};

}  // namespace bomd
