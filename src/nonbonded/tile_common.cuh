// BioOptoMD – gemeinsame Bausteine der Tile-Kernels (TiledAllPairs, TiledGeneralizedBorn):
// Switching (PHYSICS_MODEL.md §3), Bildanzahl (ADR 0006 Rev. 2), Tile-Hüllen und Vorauswahl.
#pragma once

#include <cuda_runtime.h>

#include "biooptomd/device_state.hpp"
#include "core/pbc.cuh"

namespace bomd::detail {

/// Warps pro Block; jeder Warp bearbeitet ein Tile-Paar. 4 Warps = 128 Threads: genug Warps
/// pro SM für Latenzverdeckung bei kleinem Shared-Memory-Bedarf (Tabelle + 4 × 32 j-Atome).
inline constexpr int kWarpsPerBlock = 4;

struct SwitchParams {
  float r_s, r_c, r_c2, inv_width;  // r_s, r_c [nm], r_c² [nm²], 1/(r_c − r_s) [1/nm]
};

inline SwitchParams make_switch(float r_s, float r_c) {
  return SwitchParams{r_s, r_c, r_c * r_c, 1.0f / (r_c - r_s)};
}

/// S(r) und dS/dr für r_s < r < r_c: S = 1 − 10x³ + 15x⁴ − 6x⁵ = y³(1 + 3x + 6x²),
/// dS/dx = −30 x² y², y = 1 − x; faktorisiert, weil die Monomform für x → 1 in FP32 auslöscht
/// (y direkt aus r_c − r). Für r ≤ r_s: S = 1, dS/dr = 0.
__device__ __forceinline__ void switching(float dist, const SwitchParams& sw, float& s, float& ds) {
  if (dist > sw.r_s) {
    const float x = (dist - sw.r_s) * sw.inv_width;
    const float y = (sw.r_c - dist) * sw.inv_width;
    s = y * y * y * (1.0f + x * (3.0f + 6.0f * x));
    ds = -30.0f * x * x * y * y * sw.inv_width;
  } else {
    s = 1.0f;
    ds = 0.0f;
  }
}

/// Sicherheitsabstand der Tile-Vorauswahl: absolut [nm] plus relativ zur Koordinatengröße
/// (≈ 8 ulp in FP32), deckt die Rundung von Mittelpunkt und Halbbreite (|x − c| ≤ h + ulp) und
/// des Kernel-Abstands auch für große Koordinaten ab.
inline constexpr float kCullMargin = 1.0e-3f;
inline constexpr float kCullMarginRel = 1.0e-6f;

/// Achsenparallele Hülle der echten Atome je (Replica, Tile): (Mittelpunkt, Halbbreite) als zwei
/// float4 in bounds[2·(r·n_tiles + tile) + {0, 1}]; Halbbreite.w = −1 für Tiles ohne echte Atome.
void launch_tile_bounds(const DeviceState& s, float4* bounds, cudaStream_t stream);

/// Untere Schranke des Abstands zweier Hüllen (Mittelpunkte c, Halbbreiten h) mit Minimum Image
/// in x/y: Auf dem Kreis gilt d(x_i, x_j) ≥ d(c_i, c_j) − h_i − h_j (Dreiecksungleichung), in z
/// dasselbe auf der Geraden. true ⇔ kein Paar der beiden Tiles kann innerhalb von r_c liegen.
__device__ __forceinline__ bool tiles_out_of_range(const float4* bounds, int r, int n_tiles,
                                                   int2 tp, float2 box, float r_c) {
  const float4* bi = bounds + 2 * (static_cast<long long>(r) * n_tiles + tp.x);
  const float4* bj = bounds + 2 * (static_cast<long long>(r) * n_tiles + tp.y);
  const float4 ci = bi[0], hi = bi[1], cj = bj[0], hj = bj[1];
  if (hi.w < 0.0f || hj.w < 0.0f) return true;  // Tile ohne echte Atome
  const float gx = fmaxf(0.0f, fabsf(minimum_image_1d(ci.x, cj.x, box.x)) - hi.x - hj.x);
  const float gy = fmaxf(0.0f, fabsf(minimum_image_1d(ci.y, cj.y, box.y)) - hi.y - hj.y);
  const float gz = fmaxf(0.0f, fabsf(ci.z - cj.z) - hi.z - hj.z);
  const float extent = fmaxf(fmaxf(fabsf(ci.x) + hi.x, fabsf(ci.y) + hi.y), fabsf(ci.z) + hi.z) +
                       fmaxf(fmaxf(fabsf(cj.x) + hj.x, fabsf(cj.y) + hj.y), fabsf(cj.z) + hj.z);
  const float reach = r_c + kCullMargin + kCullMarginRel * extent;
  return gx * gx + gy * gy + gz * gz >= reach * reach;
}

/// Anzahl Bildverschiebungen je Richtung (ADR 0006 Rev. 2): 0 für L >= 2 r_c (jedes verschobene
/// Bild ist dann mindestens L/2 >= r_c entfernt), sonst ⌈r_c/L + ½⌉ (der Minimum-Image-Abstand
/// selbst kann bis L/2 betragen).
__device__ __forceinline__ int image_count(float box_length, float r_c) {
  return box_length >= 2.0f * r_c ? 0 : static_cast<int>(ceilf(r_c / box_length + 0.5f));
}

}  // namespace bomd::detail
