// BioOptoMD – Langevin-Integrator BAOAB (docs/PHYSICS_MODEL.md §1).
#pragma once

#include <cuda_runtime.h>

#include <cstdint>

#include "biooptomd/device_state.hpp"

namespace bomd {

/// Parameter des BAOAB-Integrators. T pro Replica steht in DeviceState::d_temperature.
struct LangevinParams {
  double dt_ps = 0.002;          ///< Zeitschritt Δt [ps], > 0
  double friction_per_ps = 1.0;  ///< Reibung γ [1/ps], >= 0; γ = 0 ⇒ NVE (Velocity Verlet)
  std::uint64_t seed = 0;        ///< Philox-Schlüssel
};

/// BAOAB, aufgeteilt in zwei Kernels um die Kraftberechnung herum:
///   first_half:  B (v += Δt/2·F/m), A (x += Δt/2·v), O (v = c1 v + c2 ξ), A
///   second_half: B
/// mit c1 = exp(−γΔt), c2 = sqrt((1 − c1²) k_B T_r / m_i), ξ ~ N(0,1) aus
/// Philox(seed, replica, step, atom). Atome mit inv_mass = 0 (inkl. Padding) bleiben unverändert.
/// Zustand (Position, Geschwindigkeit) in FP64 (ADR 0007); Kräfte FP32.
class LangevinIntegrator {
 public:
  explicit LangevinIntegrator(const LangevinParams& params);

  const LangevinParams& params() const { return params_; }

  /// B A O A für Schritt `step` (Rauschen aus Philox mit diesem Zähler). Setzt F(x) in d_force
  /// voraus.
  void first_half(const DeviceState& s, std::int64_t step, cudaStream_t stream) const;
  /// Abschließendes B mit F(x_neu) in d_force.
  void second_half(const DeviceState& s, cudaStream_t stream) const;

 private:
  LangevinParams params_;
  double half_dt_;      // Δt/2 [ps]
  double c1_;           // exp(−γΔt)
  double noise_coeff_;  // (1 − c1²)·k_B [kJ/(mol K)]
};

}  // namespace bomd
