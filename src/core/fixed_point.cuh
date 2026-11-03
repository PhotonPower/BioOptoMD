// BioOptoMD – Fixed-Point-Akkumulation von Kräften und Energien (ADR 0002, ADR 0006).
//
// Darstellung: v_fp = round_to_nearest_even(v · 2^32) als int64 (Zweierkomplement).
// Akkumulation über atomicAdd auf unsigned long long: Ganzzahladdition modulo 2^64 ist
// assoziativ und kommutativ ⇒ das Ergebnis ist unabhängig von der Reihenfolge der Atomics.
//
// ADR 0006: Jeder Beitrag wird auf |v| <= kFixedPointMaxAbs = 2^30 (kJ/mol/nm bzw. kJ/mol)
// gesättigt, NaN wird zu 0; beides setzt ein Bit in DeviceState::d_status. Die Summe kann erst
// bei mehreren gesättigten, gleichgerichteten Beiträgen überlaufen – dann ist das Flag gesetzt.
// Auflösung 2^-32 ≈ 2.3·10^-10.
#pragma once

#include <cuda_runtime.h>

#include "biooptomd/device_state.hpp"
#include "biooptomd/units.hpp"

namespace bomd::detail {

inline constexpr float kInvFixedPointScaleF = 1.0f / units::FIXED_POINT_SCALE_F;  // 2^-32, exakt
/// Sättigungsgrenze pro Beitrag (ADR 0006).
inline constexpr float kFixedPointMaxAbs = 1073741824.0f;  // 2^30
/// Sättigungsgrenze pro Lane in warp_add_energy: 32·2^26 = 2^31 ⇒ Warpsumme läuft nicht über.
inline constexpr float kWarpEnergyMaxAbs = 67108864.0f;  // 2^26

/// float → Fixed-Point mit Sättigung auf ±limit und Statusbits (ADR 0006).
/// Multiplikation mit 2^32 ist exakt; einzige Rundung ist die nach int64.
__device__ __forceinline__ long long to_fixed_point(float v, unsigned int* status,
                                                    float limit = kFixedPointMaxAbs) {
  if (!(fabsf(v) <= limit)) {  // auch für NaN wahr
    if (isnan(v)) {
      atomicOr(status, kStatusNonFinite);
      v = 0.0f;
    } else {
      atomicOr(status, kStatusFixedPointSaturated);
      v = copysignf(limit, v);
    }
  }
  return __float2ll_rn(v * units::FIXED_POINT_SCALE_F);
}

/// Fixed-Point → float, korrekt gerundet: __ll2float_rn rundet einmal, die Skalierung mit
/// 2^-32 ist exakt (das Ergebnis liegt weit oberhalb des Subnormal-Bereichs).
__device__ __forceinline__ float from_fixed_point(long long v) {
  return __ll2float_rn(v) * kInvFixedPointScaleF;
}

/// Atomare Addition von v (float) auf einen Fixed-Point-Akkumulator.
__device__ __forceinline__ void atomic_add_fixed_point(long long* acc, float v,
                                                       unsigned int* status) {
  atomicAdd(reinterpret_cast<unsigned long long*>(acc),
            static_cast<unsigned long long>(to_fixed_point(v, status)));
}

/// Addiert die Kraft (fx, fy, fz) [kJ/mol/nm] auf Atom i_global.
__device__ __forceinline__ void add_force(const DeviceState& s, int i_global, float fx, float fy,
                                          float fz) {
  atomic_add_fixed_point(s.d_force_fp_x() + i_global, fx, s.d_status);
  atomic_add_fixed_point(s.d_force_fp_y() + i_global, fy, s.d_status);
  atomic_add_fixed_point(s.d_force_fp_z() + i_global, fz, s.d_status);
}

/// Addiert die Energie e [kJ/mol] auf Term `term` der Replica r.
__device__ __forceinline__ void add_energy(const DeviceState& s, int replica, int term, float e) {
  atomic_add_fixed_point(s.d_energy_fp + replica * kNumEnergyTerms + term, e, s.d_status);
}

/// Warpweite Summe einer Energie e [kJ/mol] (Fixed-Point, daher deterministisch) und eine
/// atomare Addition pro Warp. Voraussetzungen: alle 32 Lanes des Warps rufen die Funktion auf
/// (inaktive Atome mit e = 0) und gehören zur selben Replica (blockIdx.y = Replica).
/// Pro Lane wird auf ±2^26 kJ/mol gesättigt, damit die Warpsumme nicht überläuft.
__device__ __forceinline__ void warp_add_energy(const DeviceState& s, int replica, int term,
                                                float e) {
  long long v = to_fixed_point(e, s.d_status, kWarpEnergyMaxAbs);
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) v += __shfl_down_sync(0xffffffffu, v, offset);
  if ((threadIdx.x & 31) == 0 && v != 0)
    atomicAdd(
        reinterpret_cast<unsigned long long*>(s.d_energy_fp + replica * kNumEnergyTerms + term),
        static_cast<unsigned long long>(v));
}

}  // namespace bomd::detail
