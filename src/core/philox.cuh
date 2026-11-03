// BioOptoMD – Philox4x32-10 (Salmon et al., SC'11), zählerbasiert und zustandslos
// (AGENTS.md Regel 5). NumPy-Referenz: tests/reference/philox_ref.py (bitgenau, KAT-geprüft).
//
// Zählerbelegung: counter = (i_local, replica, step_lo, step_hi[23:0] | purpose << 24),
// key = (seed_lo, seed_hi). Verschiedene Zwecke (Langevin-O-Schritt, Initialgeschwindigkeiten, …)
// erhalten disjunkte Zahlenströme über `purpose`.
#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace bomd::detail {

enum PhiloxPurpose : unsigned int {
  kPhiloxLangevin = 0,
};

__host__ __device__ __forceinline__ void mulhilo32(unsigned int a, unsigned int b, unsigned int& hi,
                                                   unsigned int& lo) {
  const unsigned long long p = static_cast<unsigned long long>(a) * b;
  hi = static_cast<unsigned int>(p >> 32);
  lo = static_cast<unsigned int>(p);
}

/// Philox4x32 mit 10 Runden.
__host__ __device__ __forceinline__ uint4 philox4x32_10(uint4 c, uint2 k) {
  constexpr unsigned int kM0 = 0xD2511F53u, kM1 = 0xCD9E8D57u;
  constexpr unsigned int kW0 = 0x9E3779B9u, kW1 = 0xBB67AE85u;
#pragma unroll
  for (int round = 0; round < 10; ++round) {
    if (round > 0) {
      k.x += kW0;
      k.y += kW1;
    }
    unsigned int hi0, lo0, hi1, lo1;
    mulhilo32(kM0, c.x, hi0, lo0);
    mulhilo32(kM1, c.z, hi1, lo1);
    c = make_uint4(hi1 ^ c.y ^ k.x, lo1, hi0 ^ c.w ^ k.y, lo0);
  }
  return c;
}

__host__ __device__ __forceinline__ uint4 philox_counter(unsigned int i_local, unsigned int replica,
                                                         std::int64_t step, unsigned int purpose) {
  const auto s = static_cast<unsigned long long>(step);
  return make_uint4(i_local, replica, static_cast<unsigned int>(s),
                    (static_cast<unsigned int>(s >> 32) & 0x00FFFFFFu) | (purpose << 24));
}

__host__ __device__ __forceinline__ uint2 philox_key(std::uint64_t seed) {
  return make_uint2(static_cast<unsigned int>(seed), static_cast<unsigned int>(seed >> 32));
}

/// uint32 → float in (0, 1): obere 23 Bit + ½, mal 2^-23. k + ½ mit k < 2^23 ist in float exakt,
/// das Ergebnis liegt also in [2^-24, 1 − 2^-24], nie 0 oder 1.
__device__ __forceinline__ float uint_to_open_unit(unsigned int x) {
  return (static_cast<float>(x >> 9) + 0.5f) * 1.1920928955078125e-7f;
}

/// Box-Muller: zwei unabhängige N(0, 1)-Werte aus zwei Zufallsworten.
/// Größter Betrag sqrt(−2 ln 2^-24) ≈ 5.8 (Schwanz jenseits davon abgeschnitten, P ≈ 7·10^-9).
__device__ __forceinline__ float2 box_muller(unsigned int a, unsigned int b) {
  const float radius = sqrtf(-2.0f * logf(uint_to_open_unit(a)));
  float s, c;
  sincospif(2.0f * uint_to_open_unit(b), &s, &c);
  return make_float2(radius * c, radius * s);
}

}  // namespace bomd::detail
