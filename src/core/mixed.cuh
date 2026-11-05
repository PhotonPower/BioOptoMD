// BioOptoMD – Mixed-Precision-Zustand (ADR 0009): Position x = d_pos.xyz + d_pos_corr.xyz in FP64.
#pragma once

#include <cuda_runtime.h>

#include "biooptomd/device_state.hpp"

namespace bomd::detail {

/// Position von Atomplatz i in FP64 (≈ 48 Bit aus FP32-Wert + FP32-Korrektur).
__device__ __forceinline__ double3 load_position(const DeviceState& s, int i) {
  const float4 p = s.d_pos[i];
  const float4 c = s.d_pos_corr[i];
  return make_double3(static_cast<double>(p.x) + c.x, static_cast<double>(p.y) + c.y,
                      static_cast<double>(p.z) + c.z);
}

/// Speichert x: d_pos.xyz = fl32(x), d_pos_corr.xyz = fl32(x − d_pos.xyz); q (w) bleibt erhalten.
__device__ __forceinline__ void store_position(const DeviceState& s, int i, double3 x) {
  float4 p = s.d_pos[i];
  p.x = __double2float_rn(x.x);
  p.y = __double2float_rn(x.y);
  p.z = __double2float_rn(x.z);
  s.d_pos[i] = p;
  s.d_pos_corr[i] = make_float4(__double2float_rn(x.x - p.x), __double2float_rn(x.y - p.y),
                                __double2float_rn(x.z - p.z), 0.0f);
}

}  // namespace bomd::detail
