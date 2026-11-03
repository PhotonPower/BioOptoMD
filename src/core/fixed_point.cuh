// BioOptoMD – Fixed-Point-Akkumulation von Kräften und Energien (ADR 0002).
//
// Darstellung: v_fp = round_to_nearest_even(v · 2^32) als int64 (Zweierkomplement).
// Akkumulation über atomicAdd auf unsigned long long: Ganzzahladdition modulo 2^64 ist
// assoziativ und kommutativ ⇒ das Ergebnis ist unabhängig von der Reihenfolge der Atomics.
//
// Gültigkeitsbereich: |v| < 2^31 ≈ 2.1·10^9 (kJ/mol/nm bzw. kJ/mol), auch für die Summe.
// Außerhalb (und für NaN) liefert __float2ll_rn 0x8000000000000000; ein Überlauf wird in
// Phase 1 nicht erkannt. Auflösung 2^-32 ≈ 2.3·10^-10.
#pragma once

#include <cuda_runtime.h>

#include "biooptomd/device_state.hpp"
#include "biooptomd/units.hpp"

namespace bomd::detail {

inline constexpr float kInvFixedPointScaleF = 1.0f / units::FIXED_POINT_SCALE_F;  // 2^-32, exakt

/// float → Fixed-Point. Multiplikation mit 2^32 ist exakt; einzige Rundung ist die nach int64.
__device__ __forceinline__ long long to_fixed_point(float v) {
  return __float2ll_rn(v * units::FIXED_POINT_SCALE_F);
}

/// Fixed-Point → float, korrekt gerundet: __ll2float_rn rundet einmal, die Skalierung mit
/// 2^-32 ist exakt (das Ergebnis liegt weit oberhalb des Subnormal-Bereichs).
__device__ __forceinline__ float from_fixed_point(long long v) {
  return __ll2float_rn(v) * kInvFixedPointScaleF;
}

/// Atomare Addition von v (float) auf einen Fixed-Point-Akkumulator.
__device__ __forceinline__ void atomic_add_fixed_point(long long* acc, float v) {
  atomicAdd(reinterpret_cast<unsigned long long*>(acc),
            static_cast<unsigned long long>(to_fixed_point(v)));
}

/// Addiert die Kraft (fx, fy, fz) [kJ/mol/nm] auf Atom i_global.
__device__ __forceinline__ void add_force(const DeviceState& s, int i_global, float fx, float fy,
                                          float fz) {
  atomic_add_fixed_point(s.d_force_fp_x() + i_global, fx);
  atomic_add_fixed_point(s.d_force_fp_y() + i_global, fy);
  atomic_add_fixed_point(s.d_force_fp_z() + i_global, fz);
}

/// Addiert die Energie e [kJ/mol] auf Term `term` der Replica r.
__device__ __forceinline__ void add_energy(const DeviceState& s, int replica, int term, float e) {
  atomic_add_fixed_point(s.d_energy_fp + replica * kNumEnergyTerms + term, e);
}

}  // namespace bomd::detail
