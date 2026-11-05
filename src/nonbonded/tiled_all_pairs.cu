// BioOptoMD – TiledAllPairs, siehe tiled_all_pairs.hpp. Formeln: PHYSICS_MODEL.md §3 und
// tests/reference/nonbonded_ref.py (V = (V_LJ + V_el)·S(r), Switching wie OpenMM).
#include <math_constants.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>

#include "biooptomd/cuda_check.hpp"
#include "biooptomd/simulation_context.hpp"
#include "biooptomd/units.hpp"
#include "core/fixed_point.cuh"
#include "core/pbc.cuh"
#include "nonbonded/electrostatics.cuh"
#include "nonbonded/tile_common.cuh"
#include "nonbonded/tiled_all_pairs.hpp"

namespace bomd {
namespace {

constexpr int kMaxTypes = 64;
using detail::image_count;
using detail::kWarpsPerBlock;
using detail::SwitchParams;
using detail::tiles_out_of_range;

/// Hülle je (Replica, Tile), siehe detail::launch_tile_bounds (ein Warp pro Tile).
__global__ void k_tile_bounds(DeviceState s, float4* bounds) {
  const int tile = blockIdx.x;
  const int r = blockIdx.y;
  const int i_local = tile * 32 + threadIdx.x;
  const bool valid = i_local < s.n_atoms;
  const float4 p = s.d_pos[r * s.n_pad + i_local];
  float lo_x = valid ? p.x : CUDART_INF_F, lo_y = valid ? p.y : CUDART_INF_F,
        lo_z = valid ? p.z : CUDART_INF_F;
  float hi_x = valid ? p.x : -CUDART_INF_F, hi_y = valid ? p.y : -CUDART_INF_F;
  float hi_z = valid ? p.z : -CUDART_INF_F;
  for (int off = 16; off > 0; off >>= 1) {
    lo_x = fminf(lo_x, __shfl_xor_sync(0xffffffffu, lo_x, off));
    lo_y = fminf(lo_y, __shfl_xor_sync(0xffffffffu, lo_y, off));
    lo_z = fminf(lo_z, __shfl_xor_sync(0xffffffffu, lo_z, off));
    hi_x = fmaxf(hi_x, __shfl_xor_sync(0xffffffffu, hi_x, off));
    hi_y = fmaxf(hi_y, __shfl_xor_sync(0xffffffffu, hi_y, off));
    hi_z = fmaxf(hi_z, __shfl_xor_sync(0xffffffffu, hi_z, off));
  }
  if (threadIdx.x == 0) {
    const int n_tiles = s.n_pad / 32;
    float4* out = bounds + 2 * (static_cast<long long>(r) * n_tiles + tile);
    if (lo_x > hi_x) {  // keine echten Atome
      out[0] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
      out[1] = make_float4(0.0f, 0.0f, 0.0f, -1.0f);
    } else {
      out[0] = make_float4(0.5f * (lo_x + hi_x), 0.5f * (lo_y + hi_y), 0.5f * (lo_z + hi_z), 0.0f);
      out[1] = make_float4(0.5f * (hi_x - lo_x), 0.5f * (hi_y - lo_y), 0.5f * (hi_z - lo_z), 0.0f);
    }
  }
}

/// Ein Warp pro Tile-Paar (I ≤ J), blockIdx.y = Replica. Lane a hält Atom i = I·32 + a; die
/// 32 j-Atome liegen im Shared Memory und werden rotierend (b = (a + k) mod 32) besucht, sodass
/// in jedem Schritt jede Lane ein anderes j schreibt (konfliktfreie j-Kraftakkumulation).
/// Ausgeschlossen: Bit b der Exclusion-Maske, in Diagonal-Tiles b ≤ a (jedes Paar genau einmal),
/// Padding-Atome (i_local >= n_atoms) und r² < EPS_R2. Im Minimum-Image-Pfad überspringt der
/// Warp Tile-Paare, deren Hüllen weiter als r_c auseinander liegen (alle Beiträge wären exakt 0,
/// das Ergebnis bleibt bitidentisch).
///
/// kImages (ADR 0006 Rev. 2, L < 2 r_c): Für jedes Paar wird über die Gittervektoren
/// n = (n_x L_x, n_y L_y), |n_x| ≤ image_count(L_x), |n_y| ≤ image_count(L_y), summiert, ausgehend
/// vom Minimum-Image-Abstand. Exclusions gelten nur für n = 0. Eigenbilder (i = j, n ≠ 0) tragen
/// ½ V(|n|) zur Energie bei; ihre Kräfte heben sich paarweise (n, −n) auf und werden nicht
/// berechnet. Für L >= 2 r_c ist image_count = 0 und das Ergebnis gleich dem Minimum-Image-Pfad.
template <typename Elec, bool kImages>
__global__ void __launch_bounds__(kWarpsPerBlock * 32)
    k_tiled_all_pairs(DeviceState s, const int2* tile_pairs, int n_tile_pairs, const float2* lj,
                      const int* mask_index, const std::uint32_t* masks, const int4* replica,
                      const typename Elec::Params* elec, SwitchParams sw,
                      const float4* tile_bounds) {
  extern __shared__ float2 sh_lj[];  // T × T (σ, ε)
  __shared__ float4 sh_pos[kWarpsPerBlock][32];
  __shared__ int sh_type[kWarpsPerBlock][32];
  __shared__ float3 sh_fj[kWarpsPerBlock][32];

  const int r = blockIdx.y;
  const int4 rep = replica[r];  // (lj_offset, n_types, mask_index_offset, mask_base)
  const int n_types = rep.y;
  for (int k = threadIdx.x; k < n_types * n_types; k += blockDim.x) sh_lj[k] = lj[rep.x + k];
  __syncthreads();

  const int warp = threadIdx.x / 32;
  const int a = threadIdx.x % 32;
  const int p = blockIdx.x * kWarpsPerBlock + warp;
  if (p >= n_tile_pairs) return;  // ganzer Warp; danach nur noch __syncwarp

  const int2 tp = tile_pairs[p];
  if constexpr (!kImages) {
    if (tiles_out_of_range(tile_bounds, r, s.n_pad / 32, tp, s.d_box[r], sw.r_c)) return;
  }
  const int base = r * s.n_pad;
  const int i_local = tp.x * 32 + a;
  const int j_local = tp.y * 32 + a;
  const bool valid_i = i_local < s.n_atoms;
  const bool valid_j = j_local < s.n_atoms;

  const float4 pi = s.d_pos[base + i_local];
  int ti = s.d_types[base + i_local].x;
  int tj = s.d_types[base + j_local].x;
  if (ti >= n_types || tj >= n_types) atomicOr(s.d_status, kStatusUnknownType);
  ti = ti < n_types ? ti : 0;
  sh_pos[warp][a] = s.d_pos[base + j_local];
  sh_type[warp][a] = tj < n_types ? tj : 0;
  sh_fj[warp][a] = make_float3(0.0f, 0.0f, 0.0f);

  const unsigned int valid_j_bits = __ballot_sync(0xffffffffu, valid_j);
  const int mi = mask_index[rep.z + p];
  unsigned int skip = mi >= 0 ? masks[rep.w + mi * 32 + a] : 0u;
  if (tp.x == tp.y) skip |= (2u << a) - 1u;  // Bits 0..a (für a = 31 alle Bits)
  skip |= ~valid_j_bits;
  if (!valid_i) skip = 0xffffffffu;
  // Bildpfad, n ≠ 0: keine Exclusions; in Diagonal-Tiles wird b < a übersprungen (Paar genau
  // einmal), b = a ist das Eigenbild.
  unsigned int skip_image = ~valid_j_bits;
  if (tp.x == tp.y) skip_image |= (1u << a) - 1u;
  if (!valid_i) skip_image = 0xffffffffu;

  const typename Elec::Params ep = elec[r];
  const float2 box = s.d_box[r];
  float3 fi = make_float3(0.0f, 0.0f, 0.0f);
  float energy = 0.0f;
  __syncwarp();

  int n_img_x = 0, n_img_y = 0;
  if constexpr (kImages) {
    n_img_x = image_count(box.x, sw.r_c);
    n_img_y = image_count(box.y, sw.r_c);
  }

  for (int k = 0; k < 32; ++k) {
    const int b = (a + k) & 31;
    const float3 d0 = detail::minimum_image_delta(pi, sh_pos[warp][b], box);
    const float qq = pi.w * sh_pos[warp][b].w;
    const float2 se = sh_lj[ti * n_types + sh_type[warp][b]];  // (σ, ε)
    for (int nx = -n_img_x; nx <= n_img_x; ++nx) {
      for (int ny = -n_img_y; ny <= n_img_y; ++ny) {
        const bool zero_image = nx == 0 && ny == 0;
        if (((zero_image ? skip : skip_image) >> b) & 1u) continue;
        const bool self_image = kImages && !zero_image && tp.x == tp.y && b == a;
        float3 d = d0;
        if constexpr (kImages) {
          d.x += static_cast<float>(nx) * box.x;
          d.y += static_cast<float>(ny) * box.y;
        }
        const float r2 = d.x * d.x + d.y * d.y + d.z * d.z;
        if (r2 < sw.r_c2 && r2 >= units::EPS_R2) {
          const float dist = sqrtf(r2);
          const float inv_r = 1.0f / dist;
          float v = 0.0f, f = 0.0f;  // f = −dV/dr
          if (se.y != 0.0f) {        // ε = 0: kein LJ (vermeidet 0·inf = NaN bei r ≪ σ)
            const float sr2 = se.x * se.x * inv_r * inv_r;
            const float sr6 = sr2 * sr2 * sr2;
            v = 4.0f * se.y * (sr6 * sr6 - sr6);
            f = 4.0f * se.y * (12.0f * sr6 * sr6 - 6.0f * sr6) * inv_r;
          }
          float v_el, f_el;
          Elec::pair(ep, qq, dist, inv_r, v_el, f_el);
          v += v_el;
          f += f_el;
          if (dist > sw.r_s) {
            // S = 1 − 10x³ + 15x⁴ − 6x⁵ = y³(1 + 3x + 6x²), dS/dx = −30 x² y², y = 1 − x;
            // faktorisiert, weil die Monomform für x → 1 in FP32 auslöscht. y direkt aus r_c − r.
            const float x = (dist - sw.r_s) * sw.inv_width;
            const float y = (sw.r_c - dist) * sw.inv_width;
            const float sv = y * y * y * (1.0f + x * (3.0f + 6.0f * x));
            const float ds = -30.0f * x * x * y * y * sw.inv_width;
            f = f * sv - v * ds;
            v *= sv;
          }
          if (self_image) {
            energy += 0.5f * v;
          } else {
            const float c = f * inv_r;
            const float3 fv = make_float3(c * d.x, c * d.y, c * d.z);
            fi.x += fv.x;
            fi.y += fv.y;
            fi.z += fv.z;
            float3& fj = sh_fj[warp][b];
            fj.x -= fv.x;
            fj.y -= fv.y;
            fj.z -= fv.z;
            energy += v;
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
  detail::warp_add_energy(s, r, kEnergyNonbonded, energy);
}

void require(bool condition, const std::string& message) {
  if (!condition) throw std::invalid_argument("TiledAllPairs: " + message);
}

}  // namespace

namespace detail {
void launch_tile_bounds(const DeviceState& s, float4* bounds, cudaStream_t stream) {
  k_tile_bounds<<<dim3(s.n_pad / 32, s.n_replicas), 32, 0, stream>>>(s, bounds);
  BOMD_LAUNCH_CHECK();
}
}  // namespace detail

TiledAllPairs::TiledAllPairs(TiledAllPairsParams params) : params_(std::move(params)) {
  require(!params_.topologies.empty(), "mindestens eine Topologie nötig");
  require(std::isfinite(params_.r_switch_nm) && std::isfinite(params_.r_cutoff_nm) &&
              params_.r_switch_nm > 0.0f && params_.r_switch_nm < params_.r_cutoff_nm,
          "0 < r_switch < r_cutoff erwartet");
  for (const auto& t : params_.topologies) {
    require(t.n_types >= 1 && t.n_types <= kMaxTypes, "n_types muss in [1, 64] liegen");
    require(t.lj_table.size() == 2u * t.n_types * t.n_types, "lj_table muss (T, T, 2) sein");
    require(t.masks.size() % 32 == 0, "masks muss (n, 32) sein");
    for (std::size_t k = 0; k < t.lj_table.size(); ++k)
      require(std::isfinite(t.lj_table[k]) && t.lj_table[k] >= 0.0f, "LJ-Parameter endlich, >= 0");
    for (int k = 0; k < t.n_types; ++k)  // Typ 0 = Padding (AGENTS.md Regel 6)
      require(t.lj_table[2 * k + 1] == 0.0f, "Typ 0 muss ε = 0 haben");
  }
}

void TiledAllPairs::initialize(SimulationContext& ctx) {
  ctx_ = &ctx;
  const int n_rep = ctx.n_replicas();
  require(params_.replica_topology.size() == static_cast<std::size_t>(n_rep) &&
              params_.kappa_per_nm.size() == static_cast<std::size_t>(n_rep) &&
              params_.eps_r.size() == static_cast<std::size_t>(n_rep),
          "replica_topology, kappa, eps_r brauchen n_replicas Einträge");

  const int n_tiles = ctx.n_pad() / 32;
  std::vector<int2> tps;
  for (int i = 0; i < n_tiles; ++i)
    for (int j = i; j < n_tiles; ++j) tps.push_back(make_int2(i, j));
  n_tile_pairs_ = static_cast<int>(tps.size());

  std::vector<float2> lj;
  std::vector<int> mask_index;
  std::vector<std::uint32_t> masks;
  std::vector<int4> topo_offsets;
  for (const auto& t : params_.topologies) {
    require(t.mask_index.size() == static_cast<std::size_t>(n_tile_pairs_),
            "mask_index passt nicht zu n_pad (" + std::to_string(n_tile_pairs_) + " Tile-Paare)");
    const int n_mask_tiles = static_cast<int>(t.masks.size() / 32);
    for (int m : t.mask_index) require(m >= -1 && m < n_mask_tiles, "mask_index außerhalb");
    topo_offsets.push_back(make_int4(static_cast<int>(lj.size()), t.n_types,
                                     static_cast<int>(mask_index.size()),
                                     static_cast<int>(masks.size())));
    for (int k = 0; k < t.n_types * t.n_types; ++k)
      lj.push_back(make_float2(t.lj_table[2 * k], t.lj_table[2 * k + 1]));
    mask_index.insert(mask_index.end(), t.mask_index.begin(), t.mask_index.end());
    masks.insert(masks.end(), t.masks.begin(), t.masks.end());
    max_types_ = std::max(max_types_, t.n_types);
  }
  std::vector<int4> replica(n_rep);
  std::vector<float2> screening(n_rep);
  for (int r = 0; r < n_rep; ++r) {
    const int t = params_.replica_topology[r];
    require(t >= 0 && t < static_cast<int>(params_.topologies.size()), "Topologie-Index ungültig");
    require(std::isfinite(params_.kappa_per_nm[r]) && params_.kappa_per_nm[r] >= 0.0f &&
                std::isfinite(params_.eps_r[r]) && params_.eps_r[r] > 0.0f,
            "kappa >= 0 und eps_r > 0 erwartet");
    replica[r] = topo_offsets[t];
    screening[r] = make_float2(params_.kappa_per_nm[r], params_.eps_r[r]);
  }

  const cudaStream_t st = ctx.stream();
  d_tile_pairs_ = DeviceBuffer<int2>(tps.size());
  d_lj_ = DeviceBuffer<float2>(lj.size());
  d_mask_index_ = DeviceBuffer<int>(mask_index.size());
  d_masks_ = DeviceBuffer<std::uint32_t>(std::max<std::size_t>(masks.size(), 1));
  d_replica_ = DeviceBuffer<int4>(n_rep);
  d_screening_ = DeviceBuffer<float2>(n_rep);
  d_tile_bounds_ = DeviceBuffer<float4>(static_cast<std::size_t>(n_rep) * n_tiles * 2);
  d_tile_pairs_.upload_async(tps.data(), st);
  d_lj_.upload_async(lj.data(), st);
  d_mask_index_.upload_async(mask_index.data(), st);
  if (!masks.empty())
    BOMD_CUDA_CHECK(cudaMemcpyAsync(d_masks_.data(), masks.data(),
                                    masks.size() * sizeof(std::uint32_t), cudaMemcpyHostToDevice,
                                    st));
  d_replica_.upload_async(replica.data(), st);
  d_screening_.upload_async(screening.data(), st);
  ctx.synchronize();  // Host-Vektoren leben bis hier
}

void TiledAllPairs::set_screening(const std::vector<float>& kappa, const std::vector<float>& eps_r,
                                  cudaStream_t stream) {
  std::vector<float2> screening(kappa.size());
  for (std::size_t r = 0; r < kappa.size(); ++r) {
    params_.kappa_per_nm[r] = kappa[r];
    params_.eps_r[r] = eps_r[r];
    screening[r] = make_float2(kappa[r], eps_r[r]);
  }
  d_screening_.upload_async(screening.data(), stream);
  BOMD_CUDA_CHECK(cudaStreamSynchronize(stream));  // screening lebt bis hier
}

bool TiledAllPairs::uses_image_sum() const {
  if (params_.always_image_sum) return true;
  const float min_box = 2.0f * params_.r_cutoff_nm;
  for (const float2& b : ctx_->host_box())
    if (b.x < min_box || b.y < min_box) return true;
  return false;
}

void TiledAllPairs::compute(DeviceState& s, cudaStream_t stream) {
  if (ctx_->host_box().empty()) throw std::logic_error("TiledAllPairs: Box nicht gesetzt");
  const float r_s = params_.r_switch_nm, r_c = params_.r_cutoff_nm;
  const SwitchParams sw = detail::make_switch(r_s, r_c);
  const dim3 grid((n_tile_pairs_ + kWarpsPerBlock - 1) / kWarpsPerBlock, s.n_replicas);
  const std::size_t shmem = static_cast<std::size_t>(max_types_) * max_types_ * sizeof(float2);
  const auto launch = [&](auto kernel) {
    kernel<<<grid, kWarpsPerBlock * 32, shmem, stream>>>(
        s, d_tile_pairs_.data(), n_tile_pairs_, d_lj_.data(), d_mask_index_.data(), d_masks_.data(),
        d_replica_.data(), d_screening_.data(), sw, d_tile_bounds_.data());
    BOMD_LAUNCH_CHECK();
  };
  // Bildsumme nur, wenn eine Box kleiner als 2 r_c ist (ADR 0006 Rev. 2); sonst Minimum Image
  // mit Tile-Vorauswahl.
  if (uses_image_sum()) {
    launch(k_tiled_all_pairs<detail::DebyeHuckel, true>);
  } else {
    detail::launch_tile_bounds(s, d_tile_bounds_.data(), stream);
    launch(k_tiled_all_pairs<detail::DebyeHuckel, false>);
  }
}

}  // namespace bomd
