// BioOptoMD – ImageChargeForce, siehe image_charge.hpp. Formeln: ADR 0013 und
// tests/reference/image_charge_ref.py.
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>

#include "biooptomd/cuda_check.hpp"
#include "biooptomd/simulation_context.hpp"
#include "biooptomd/units.hpp"
#include "core/fixed_point.cuh"
#include "core/pbc.cuh"
#include "forces/image_charge.hpp"
#include "nonbonded/tile_common.cuh"

namespace bomd {
namespace {

using detail::image_count;
using detail::kWarpsPerBlock;
using detail::SwitchParams;

struct ImagePlane {
  float z_image, zeta_clamp, factor;
};

/// C¹-glattes Clamping (OQ-18 d): ζ = raw für raw ≥ ζ_c + h, ζ = ζ_c für raw ≤ ζ_c − h,
/// dazwischen ζ = ζ_c + (raw − ζ_c + h)²/(4h); h = ζ_c/2. ζ ≥ max(raw, ζ_c), dζ/draw ∈ [0, 1].
__device__ __forceinline__ float smooth_zeta(float raw, float zeta_clamp, float& dzeta) {
  const float h = kZetaSmoothFraction * zeta_clamp;
  const float u = raw - zeta_clamp + h;  // ∈ [0, 2h] im Übergang
  if (u >= 2.0f * h) {
    dzeta = 1.0f;
    return raw;
  }
  if (u <= 0.0f) {
    dzeta = 0.0f;
    return zeta_clamp;
  }
  dzeta = u / (2.0f * h);
  return zeta_clamp + u * u / (4.0f * h);
}

/// Untere Schranke von s = |(Δx, Δy, ζ_i + ζ_j)| für zwei Tiles: ζ ≥ max(z_min − z_im, ζ_c) je
/// Tile (gilt auch für das geglättete ζ), im Minimum-Image-Pfad zusätzlich die lateralen Lücken der
/// Hüllen (wie detail::tiles_out_of_range). true ⇔ kein Paar (und kein Bild) liegt innerhalb von
/// r_c.
template <bool kLateral>
__device__ __forceinline__ bool image_tiles_out_of_range(const float4* bounds, int r, int n_tiles,
                                                         int2 tp, float2 box, float r_c,
                                                         ImagePlane plane) {
  const float4* bi = bounds + 2 * (static_cast<long long>(r) * n_tiles + tp.x);
  const float4* bj = bounds + 2 * (static_cast<long long>(r) * n_tiles + tp.y);
  const float4 ci = bi[0], hi = bi[1], cj = bj[0], hj = bj[1];
  if (hi.w < 0.0f || hj.w < 0.0f) return true;  // Tile ohne echte Atome
  const float zi = fmaxf(ci.z - hi.z - plane.z_image, plane.zeta_clamp);
  const float zj = fmaxf(cj.z - hj.z - plane.z_image, plane.zeta_clamp);
  float gap2 = (zi + zj) * (zi + zj);
  if constexpr (kLateral) {
    const float gx = fmaxf(0.0f, fabsf(detail::minimum_image_1d(ci.x, cj.x, box.x)) - hi.x - hj.x);
    const float gy = fmaxf(0.0f, fabsf(detail::minimum_image_1d(ci.y, cj.y, box.y)) - hi.y - hj.y);
    gap2 += gx * gx + gy * gy;
  }
  const float extent = fmaxf(fmaxf(fabsf(ci.x) + hi.x, fabsf(ci.y) + hi.y), fabsf(ci.z) + hi.z) +
                       fmaxf(fmaxf(fabsf(cj.x) + hj.x, fabsf(cj.y) + hj.y), fabsf(cj.z) + hj.z) +
                       fabsf(plane.z_image);
  const float reach = r_c + detail::kCullMargin + detail::kCullMarginRel * extent;
  return gap2 >= reach * reach;
}

/// Ein Warp pro Tile-Paar (I ≤ J), blockIdx.y = Replica; Aufbau wie k_tiled_all_pairs. j-Atome
/// liegen gespiegelt im Shared Memory: (x, y, ζ = smooth_zeta(z − z_im), q) und dζ/dz. Keine
/// Exclusions; in Diagonal-Tiles wird b < a übersprungen, b = a ist das Eigenbild (Energie ½, nur
/// z-Kraft, weil der laterale Abstand konstant ist). Die z-Kräfte eines Paars haben dasselbe
/// Vorzeichen und tragen den Faktor dζ/dz (0 für geclampte Atome).
///
/// kImages (L < 2 r_c, ADR 0006 Rev. 2): Summe über n = (n_x L_x, n_y L_y), auch für das
/// Eigenbild (dessen Gitterbilder tragen hier eine z-Kraft bei, anders als im Nonbonded-Kernel).
template <bool kImages>
__global__ void __launch_bounds__(kWarpsPerBlock * 32)
    k_image_charges(DeviceState s, const int2* tile_pairs, int n_tile_pairs,
                    const float2* screening, SwitchParams sw, ImagePlane plane,
                    const float4* tile_bounds, unsigned long long* clamp_counter) {
  __shared__ float4 sh_pos[kWarpsPerBlock][32];
  __shared__ float sh_dz[kWarpsPerBlock][32];
  __shared__ float3 sh_fj[kWarpsPerBlock][32];

  const int r = blockIdx.y;
  const int warp = threadIdx.x / 32;
  const int a = threadIdx.x % 32;
  const int p = blockIdx.x * kWarpsPerBlock + warp;
  if (p >= n_tile_pairs) return;  // ganzer Warp

  const int2 tp = tile_pairs[p];
  const bool diagonal = tp.x == tp.y;
  const int base = r * s.n_pad;
  const int i_local = tp.x * 32 + a;
  const int j_local = tp.y * 32 + a;
  const bool valid_i = i_local < s.n_atoms;
  const bool valid_j = j_local < s.n_atoms;
  const float4 pi = s.d_pos[base + i_local];
  float dz_i;
  const float zeta_i = smooth_zeta(pi.z - plane.z_image, plane.zeta_clamp, dz_i);
  if (diagonal && valid_i && pi.w != 0.0f && dz_i < 1.0f) atomicAdd(clamp_counter, 1ull);
  if (image_tiles_out_of_range<!kImages>(tile_bounds, r, s.n_pad / 32, tp, s.d_box[r], sw.r_c,
                                         plane))
    return;

  const float4 pj = s.d_pos[base + j_local];
  float dz_j;
  const float zeta_j = smooth_zeta(pj.z - plane.z_image, plane.zeta_clamp, dz_j);
  sh_dz[warp][a] = dz_j;
  sh_pos[warp][a] = make_float4(pj.x, pj.y, zeta_j, pj.w);
  sh_fj[warp][a] = make_float3(0.0f, 0.0f, 0.0f);

  unsigned int skip = ~__ballot_sync(0xffffffffu, valid_j && pj.w != 0.0f);
  if (diagonal) skip |= (1u << a) - 1u;  // b < a: Paar genau einmal
  if (!valid_i || pi.w == 0.0f) skip = 0xffffffffu;

  const float2 scr = screening[r];  // (κ, ε_r)
  const float g_scale = units::ONE_4PI_EPS0_F / scr.y;
  const float2 box = s.d_box[r];
  float3 fi = make_float3(0.0f, 0.0f, 0.0f);
  float energy = 0.0f;
  int n_img_x = 0, n_img_y = 0;
  if constexpr (kImages) {
    n_img_x = image_count(box.x, sw.r_c);
    n_img_y = image_count(box.y, sw.r_c);
  }
  __syncwarp();

  for (int k = 0; k < 32; ++k) {
    const int b = (a + k) & 31;
    if (!((skip >> b) & 1u)) {
      const float4 q_j = sh_pos[warp][b];
      const bool self_image = diagonal && b == a;
      const float dz_j = sh_dz[warp][b];
      const float qq = pi.w * q_j.w;
      const float dx0 = detail::minimum_image_1d(pi.x, q_j.x, box.x);
      const float dy0 = detail::minimum_image_1d(pi.y, q_j.y, box.y);
      const float sz = zeta_i + q_j.z;
      for (int nx = -n_img_x; nx <= n_img_x; ++nx) {
        for (int ny = -n_img_y; ny <= n_img_y; ++ny) {
          float sx = dx0, sy = dy0;
          if constexpr (kImages) {
            sx += static_cast<float>(nx) * box.x;
            sy += static_cast<float>(ny) * box.y;
          }
          const float s2 = sx * sx + sy * sy + sz * sz;
          if (s2 >= sw.r_c2) continue;
          const float dist = sqrtf(s2);
          const float inv_s = 1.0f / dist;
          const float g = g_scale * expf(-scr.x * dist) * inv_s;
          float phi = g, dphi = -g * (scr.x + inv_s);  // G(s), dG/ds
          float sv, ds;
          detail::switching(dist, sw, sv, ds);
          dphi = dphi * sv + phi * ds;
          phi *= sv;
          const float c = -plane.factor * qq * dphi * inv_s;  // (∂U/∂s)/s
          if (self_image) {
            energy -= 0.5f * plane.factor * qq * phi;
            fi.z -= c * sz * dz_i;
          } else {
            energy -= plane.factor * qq * phi;
            fi.x -= c * sx;
            fi.y -= c * sy;
            fi.z -= c * sz * dz_i;
            float3& fj = sh_fj[warp][b];
            fj.x += c * sx;
            fj.y += c * sy;
            fj.z -= c * sz * dz_j;
          }
        }
      }
    }
    __syncwarp();
  }

  if (valid_i && (fi.x != 0.0f || fi.y != 0.0f || fi.z != 0.0f))
    detail::add_force(s, base + i_local, fi.x, fi.y, fi.z);
  const float3 fj = sh_fj[warp][a];
  if (valid_j && (fj.x != 0.0f || fj.y != 0.0f || fj.z != 0.0f))
    detail::add_force(s, base + j_local, fj.x, fj.y, fj.z);
  detail::warp_add_energy(s, r, kEnergyImage, energy);
}

void require(bool condition, const std::string& message) {
  if (!condition) throw std::invalid_argument("ImageChargeForce: " + message);
}

void check_screening(const std::vector<float>& kappa, const std::vector<float>& eps_r) {
  require(kappa.size() == eps_r.size(), "kappa und eps_r gleich lang");
  for (std::size_t r = 0; r < kappa.size(); ++r)
    require(
        std::isfinite(kappa[r]) && kappa[r] >= 0.0f && std::isfinite(eps_r[r]) && eps_r[r] > 0.0f,
        "kappa >= 0 und eps_r > 0 erwartet");
}

}  // namespace

ImageChargeForce::ImageChargeForce(ImageChargeParams params) : params_(std::move(params)) {
  require(std::isfinite(params_.r_switch_nm) && std::isfinite(params_.r_cutoff_nm) &&
              params_.r_switch_nm > 0.0f && params_.r_switch_nm < params_.r_cutoff_nm,
          "0 < r_switch < r_cutoff erwartet");
  require(std::isfinite(params_.factor) && params_.factor >= -1.0f && params_.factor <= 1.0f,
          "factor in [-1, 1] erwartet");
  require(std::isfinite(params_.zeta_clamp_nm) && params_.zeta_clamp_nm >= kMinZetaClamp,
          "zeta_clamp >= 0.02 nm erwartet");
  require(std::isfinite(params_.z_image_nm), "z_image endlich");
  check_screening(params_.kappa_per_nm, params_.eps_r);
}

void ImageChargeForce::initialize(SimulationContext& ctx) {
  ctx_ = &ctx;
  const int n_rep = ctx.n_replicas();
  require(params_.kappa_per_nm.size() == static_cast<std::size_t>(n_rep),
          "kappa, eps_r brauchen n_replicas Einträge");
  const int n_tiles = ctx.n_pad() / 32;
  std::vector<int2> tps;
  for (int i = 0; i < n_tiles; ++i)
    for (int j = i; j < n_tiles; ++j) tps.push_back(make_int2(i, j));
  n_tile_pairs_ = static_cast<int>(tps.size());
  const cudaStream_t st = ctx.stream();
  d_tile_pairs_ = DeviceBuffer<int2>(tps.size());
  d_tile_pairs_.upload_async(tps.data(), st);
  d_screening_ = DeviceBuffer<float2>(n_rep);
  d_tile_bounds_ = DeviceBuffer<float4>(static_cast<std::size_t>(n_rep) * n_tiles * 2);
  d_counter_ = DeviceBuffer<unsigned long long>(1);
  d_counter_.zero_async(st);
  set_screening(params_.kappa_per_nm, params_.eps_r, st);  // synchronisiert; tps lebt bis hier
}

void ImageChargeForce::set_screening(const std::vector<float>& kappa,
                                     const std::vector<float>& eps_r, cudaStream_t stream) {
  check_screening(kappa, eps_r);
  require(kappa.size() == params_.kappa_per_nm.size(), "kappa braucht n_replicas Einträge");
  params_.kappa_per_nm = kappa;
  params_.eps_r = eps_r;
  std::vector<float2> screening(kappa.size());
  for (std::size_t r = 0; r < kappa.size(); ++r) screening[r] = make_float2(kappa[r], eps_r[r]);
  d_screening_.upload_async(screening.data(), stream);
  BOMD_CUDA_CHECK(cudaStreamSynchronize(stream));  // screening lebt bis hier
}

bool ImageChargeForce::uses_image_sum() const {
  if (params_.always_image_sum) return true;
  const float min_box = 2.0f * params_.r_cutoff_nm;
  for (const float2& b : ctx_->host_box())
    if (b.x < min_box || b.y < min_box) return true;
  return false;
}

void ImageChargeForce::compute(DeviceState& s, cudaStream_t stream) {
  if (ctx_->host_box().empty()) throw std::logic_error("ImageChargeForce: Box nicht gesetzt");
  const SwitchParams sw = detail::make_switch(params_.r_switch_nm, params_.r_cutoff_nm);
  const ImagePlane plane{params_.z_image_nm, params_.zeta_clamp_nm, params_.factor};
  const dim3 grid((n_tile_pairs_ + kWarpsPerBlock - 1) / kWarpsPerBlock, s.n_replicas);
  detail::launch_tile_bounds(s, d_tile_bounds_.data(), stream);
  const auto launch = [&](auto kernel) {
    kernel<<<grid, kWarpsPerBlock * 32, 0, stream>>>(s, d_tile_pairs_.data(), n_tile_pairs_,
                                                     d_screening_.data(), sw, plane,
                                                     d_tile_bounds_.data(), d_counter_.data());
    BOMD_LAUNCH_CHECK();
  };
  if (uses_image_sum()) {
    launch(k_image_charges<true>);
  } else {
    launch(k_image_charges<false>);
  }
}

std::uint64_t ImageChargeForce::clamp_count() const {
  unsigned long long value = 0;
  ctx_->activate();
  BOMD_CUDA_CHECK(cudaMemcpyAsync(&value, d_counter_.data(), sizeof(value), cudaMemcpyDeviceToHost,
                                  ctx_->stream()));
  BOMD_CUDA_CHECK(cudaStreamSynchronize(ctx_->stream()));
  return value;
}

void ImageChargeForce::reset_counters() {
  ctx_->activate();
  d_counter_.zero_async(ctx_->stream());
}

}  // namespace bomd
