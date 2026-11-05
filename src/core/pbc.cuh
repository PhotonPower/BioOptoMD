// BioOptoMD – Minimum Image in x/y (ADR 0003), genauigkeitsoptimiert für FP32.
#pragma once

#include <cuda_runtime.h>

namespace bomd::detail {

/// a − b mit Minimum Image für eine Koordinate (Box L > 0).
///
/// Naiv (a − b) − n·L rundet a − b auf der Skala der größeren Koordinate (~L): liegt eine Bindung
/// über der Boxgrenze (a ≈ 0, b ≈ L), geht dabei der Großteil der Mantisse des kleinen Abstands
/// verloren (Fehler ~ ulp(L) ≈ 5·10^-7 nm bei L = 8 nm). Stattdessen wird die betragsgrößere
/// Koordinate zuerst um n·L verschoben – für Koordinaten in [0, L) ist das nach dem
/// Sterbenz-Lemma exakt – und erst dann subtrahiert, sodass nur eine Rundung auf der Skala des
/// Abstands bleibt.
__device__ __forceinline__ float minimum_image_1d(float a, float b, float box_length) {
  const float n = rintf((a - b) / box_length);
  if (n == 0.0f) return a - b;
  return fabsf(a) >= fabsf(b) ? (a - n * box_length) - b : a - (b + n * box_length);
}

/// p_a − p_b (xyz) mit Minimum Image in x und y; z nicht periodisch.
__device__ __forceinline__ float3 minimum_image_delta(float4 pa, float4 pb, float2 box) {
  return make_float3(minimum_image_1d(pa.x, pb.x, box.x), minimum_image_1d(pa.y, pb.y, box.y),
                     pa.z - pb.z);
}

}  // namespace bomd::detail
