// BioOptoMD – Elektrostatik-Modelle als Template-Parameter des Nonbonded-Kernels
// (ARCHITECTURE.md §3: nicht virtuell im Device-Code). V1: Debye-Hückel; vorbereitet für
// GeneralizedBornOBC (ADR 0004), das zusätzlich Born-Radien pro Atom benötigt.
#pragma once

#include <cuda_runtime.h>

#include "biooptomd/units.hpp"

namespace bomd::detail {

/// Abgeschirmte Coulomb-Wechselwirkung (PHYSICS_MODEL.md §3):
///   V = C q_i q_j exp(−κ r) / (ε_r r),   −dV/dr = V (κ + 1/r),   C = ONE_4PI_EPS0.
/// Parameter pro Replica: (κ [1/nm], ε_r).
struct DebyeHuckel {
  using Params = float2;  // (κ, ε_r)

  __device__ __forceinline__ static void pair(const Params& p, float qq, float r, float inv_r,
                                              float& v, float& minus_dv_dr) {
    v = units::ONE_4PI_EPS0_F * qq * expf(-p.x * r) * inv_r / p.y;
    minus_dv_dr = v * (p.x + inv_r);
  }
};

}  // namespace bomd::detail
