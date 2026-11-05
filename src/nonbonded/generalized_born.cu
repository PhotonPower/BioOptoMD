// BioOptoMD – TiledGeneralizedBorn, siehe generalized_born.hpp. Formeln: ADR 0010 und
// tests/reference/gb_ref.py. Tile-Schleife wie k_tiled_all_pairs (rotierende j-Lanes, Shared-
// Memory-Akkumulation für j, Bildsumme bei L < 2 r_c, Vorauswahl im Minimum-Image-Pfad).
#include <cmath>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "biooptomd/cuda_check.hpp"
#include "biooptomd/simulation_context.hpp"
#include "biooptomd/units.hpp"
#include "core/core_kernels.cuh"
#include "core/fixed_point.cuh"
#include "core/pbc.cuh"
#include "nonbonded/generalized_born.hpp"
#include "nonbonded/tile_common.cuh"

namespace bomd {
namespace {

using detail::image_count;
using detail::kWarpsPerBlock;
using detail::SwitchParams;
using detail::tiles_out_of_range;

constexpr float kNeckScale = 0.826836f;       // GBn2 (ADR 0012)
constexpr float kNeckCut = 0.68f;             // nm, Neck-Term für r < ρ_i + ρ_j + 0.68 nm
constexpr double kAcePrefactor = 28.3919551;  // 4πγ [kJ/mol/nm²]
constexpr double kAceProbe = 0.14;            // nm

/// HCT-Integral H(r, or_i, sr_j) und dH/dr (ADR 0010; gb_ref.h_integral). 0 für r + sr ≤ or.
__device__ __forceinline__ float h_integral(float r, float o, float s, float& dh) {
  const float u = r + s;
  if (!(u > o) || !(o > 0.0f)) {
    dh = 0.0f;
    return 0.0f;
  }
  const float diff = r - s;
  const float d = fabsf(diff);
  const bool use_d = d > o;
  const float lo = use_d ? d : o;
  const float dl = use_d ? copysignf(1.0f, diff) : 0.0f;
  const float inv_r = 1.0f / r, inv_u = 1.0f / u, inv_l = 1.0f / lo;
  const float a = r - s * s * inv_r;
  const float da = 1.0f + s * s * inv_r * inv_r;
  const float log_lu = logf(lo * inv_u);
  const float iu2 = inv_u * inv_u, il2 = inv_l * inv_l;
  dh = 0.5f * (-dl * il2 + iu2 + 0.25f * da * (iu2 - il2) +
               0.25f * a * (-2.0f * iu2 * inv_u + 2.0f * dl * il2 * inv_l) +
               0.5f * ((dl * inv_l - inv_u) * inv_r - log_lu * inv_r * inv_r));
  return 0.5f * (inv_l - inv_u + 0.25f * a * (iu2 - il2) + 0.5f * log_lu * inv_r);
}

/// GBn2-Neck-Term s·m0/(1 + 100 x² + 3·10⁵ x⁶), x = r − d0 [nm], für r < ρ_i + ρ_j + 0.68 nm,
/// und seine Ableitung dn. (m0, d0) = table[info.x + idx_i·U + idx_j]; U = info.y = 0 ⇒ 0.
/// Die Stufe am Neck-Cutoff ist Teil des AMBER-Modells (igb = 8).
__device__ __forceinline__ float neck_term(float r, float rho_i, float rho_j, int idx_i, int idx_j,
                                           const float2* table, int2 info, float& dn) {
  dn = 0.0f;
  if (info.y == 0 || !(r < rho_i + rho_j + kNeckCut)) return 0.0f;
  const float2 md = table[info.x + idx_i * info.y + idx_j];
  const float x = r - md.y;
  const float x2 = x * x;
  const float inv = 1.0f / (1.0f + 100.0f * x2 + 3.0e5f * x2 * x2 * x2);
  dn = -kNeckScale * md.x * (200.0f * x + 1.8e6f * x2 * x2 * x) * inv * inv;
  return kNeckScale * md.x * inv;
}

/// Gemeinsamer Kopf der Tile-Kernels. Liefert false, wenn der Warp nichts zu tun hat.
struct TileSetup {
  int2 tp;
  int base, i_local, j_local;
  bool valid_i, valid_j;
  unsigned int skip0, skip_image;  // n = 0 (ohne Exclusions) bzw. n ≠ 0
  float2 box;
  int n_img_x, n_img_y;
};

template <bool kImages>
__device__ __forceinline__ bool tile_setup(const DeviceState& s, const int2* tile_pairs,
                                           int n_tile_pairs, const float4* bounds, float r_c, int p,
                                           int a, TileSetup& t) {
  if (p >= n_tile_pairs) return false;  // ganzer Warp
  const int r = blockIdx.y;
  t.tp = tile_pairs[p];
  t.box = s.d_box[r];
  if constexpr (!kImages) {
    if (tiles_out_of_range(bounds, r, s.n_pad / 32, t.tp, t.box, r_c)) return false;
  }
  t.base = r * s.n_pad;
  t.i_local = t.tp.x * 32 + a;
  t.j_local = t.tp.y * 32 + a;
  t.valid_i = t.i_local < s.n_atoms;
  t.valid_j = t.j_local < s.n_atoms;
  const unsigned int valid_j_bits = __ballot_sync(0xffffffffu, t.valid_j);
  const bool diag = t.tp.x == t.tp.y;
  t.skip0 = ~valid_j_bits | (diag ? (2u << a) - 1u : 0u);
  t.skip_image = ~valid_j_bits | (diag ? (1u << a) - 1u : 0u);
  if (!t.valid_i) t.skip0 = t.skip_image = 0xffffffffu;
  t.n_img_x = t.n_img_y = 0;
  if constexpr (kImages) {
    t.n_img_x = image_count(t.box.x, r_c);
    t.n_img_y = image_count(t.box.y, r_c);
  }
  return true;
}

/// Pass (a): I_i += H(r, or_i, sr_j)·S(r), I_j += H(r, or_j, sr_i)·S(r) für alle Paare innerhalb
/// r_c (auch ausgeschlossene); Eigenbild: I_i += H(|n|, or_i, sr_i)·S je n ≠ 0. S glättet den
/// Cutoff (ein harter Schnitt erzeugt bei jedem Kreuzen von r_c einen Energiesprung, ADR 0010).
template <bool kImages>
__global__ void __launch_bounds__(kWarpsPerBlock * 32)
    k_gb_integrals(DeviceState s, const int2* tile_pairs, int n_tile_pairs, const float4* gb_atom,
                   const int* gb_offset, const float2* neck, const int2* neck_info, SwitchParams sw,
                   const float4* bounds, long long* integral) {
  __shared__ float4 sh_pos[kWarpsPerBlock][32];
  __shared__ float4 sh_gb[kWarpsPerBlock][32];  // (or, sr, ρ, Neck-Index)
  __shared__ float sh_acc[kWarpsPerBlock][32];
  const int warp = threadIdx.x / 32, a = threadIdx.x % 32;
  TileSetup t;
  if (!tile_setup<kImages>(s, tile_pairs, n_tile_pairs, bounds, sw.r_c,
                           blockIdx.x * kWarpsPerBlock + warp, a, t))
    return;
  const int off = gb_offset[blockIdx.y];
  const float4 pi = s.d_pos[t.base + t.i_local];
  const float4 gi = gb_atom[off + t.i_local];
  sh_pos[warp][a] = s.d_pos[t.base + t.j_local];
  sh_gb[warp][a] = gb_atom[off + t.j_local];
  sh_acc[warp][a] = 0.0f;
  const int2 ni = neck_info[blockIdx.y];
  float acc_i = 0.0f;
  __syncwarp();
  for (int k = 0; k < 32; ++k) {
    const int b = (a + k) & 31;
    const float3 d0 = detail::minimum_image_delta(pi, sh_pos[warp][b], t.box);
    const float4 gb_j = sh_gb[warp][b];
    for (int nx = -t.n_img_x; nx <= t.n_img_x; ++nx) {
      for (int ny = -t.n_img_y; ny <= t.n_img_y; ++ny) {
        const bool zero_image = nx == 0 && ny == 0;
        if (((zero_image ? t.skip0 : t.skip_image) >> b) & 1u) continue;
        const bool self_image = kImages && !zero_image && t.tp.x == t.tp.y && b == a;
        float3 d = d0;
        if constexpr (kImages) {
          d.x += static_cast<float>(nx) * t.box.x;
          d.y += static_cast<float>(ny) * t.box.y;
        }
        const float r2 = d.x * d.x + d.y * d.y + d.z * d.z;
        if (r2 < sw.r_c2 && r2 >= units::EPS_R2) {
          const float dist = sqrtf(r2);
          float dh, dn, sv, ds;
          detail::switching(dist, sw, sv, ds);
          const int idx_i = static_cast<int>(gi.w), idx_j = static_cast<int>(gb_j.w);
          acc_i += sv * (h_integral(dist, gi.x, gb_j.y, dh) +
                         neck_term(dist, gi.z, gb_j.z, idx_i, idx_j, neck, ni, dn));
          if (!self_image)
            sh_acc[warp][b] += sv * (h_integral(dist, gb_j.x, gi.y, dh) +
                                     neck_term(dist, gb_j.z, gi.z, idx_j, idx_i, neck, ni, dn));
        }
      }
    }
    __syncwarp();
  }
  if (t.valid_i && acc_i != 0.0f)
    detail::atomic_add_fixed_point(integral + t.base + t.i_local, acc_i, s.d_status);
  const float acc_j = sh_acc[warp][a];
  if (t.valid_j && acc_j != 0.0f)
    detail::atomic_add_fixed_point(integral + t.base + t.j_local, acc_j, s.d_status);
}

/// Pass (b): B_i, dB_i/dI_i, Selbst- und ACE-Energie, ∂E/∂B_i (Selbstanteil) in FP64.
__global__ void k_gb_radii(DeviceState s, const float4* gb_atom, const float4* gb_abg,
                           const int* gb_offset, const float2* screening, float eps_in, int ace,
                           const long long* integral, float* born, float* db_di, long long* de_db) {
  const int i_local = blockIdx.x * blockDim.x + threadIdx.x;
  const int r = blockIdx.y;
  float energy = 0.0f;
  if (i_local < s.n_pad) {
    const int i = r * s.n_pad + i_local;
    if (i_local < s.n_atoms) {
      const float4 g = gb_atom[gb_offset[r] + i_local];
      const float4 abg = gb_abg[gb_offset[r] + i_local];  // (α, β, γ)
      const double o = g.x, rho = g.z;
      const double psi = static_cast<double>(detail::from_fixed_point(integral[i])) * o;
      const double x = abg.x * psi - abg.y * psi * psi + abg.z * psi * psi * psi;
      const double th = tanh(x);
      const double b = 1.0 / (1.0 / o - th / rho);
      const double dbdi =
          b * b * (1.0 - th * th) * (abg.x - 2.0 * abg.y * psi + 3.0 * abg.z * psi * psi) * o / rho;
      const float2 sc = screening[r];  // (κ, ε_out)
      const double q = s.d_pos[i].w;
      const double ex = exp(-static_cast<double>(sc.x) * b);
      const double gs = 1.0 / eps_in - ex / sc.y;
      const double c = units::ONE_4PI_EPS0;
      double e = -0.5 * c * gs * q * q / b;
      double dedb = -0.5 * c * q * q * (sc.x * ex / sc.y / b - gs / (b * b));
      if (ace) {
        const double rb = rho / b;
        const double e_ace =
            kAcePrefactor * (rho + kAceProbe) * (rho + kAceProbe) * rb * rb * rb * rb * rb * rb;
        e += e_ace;
        dedb -= 6.0 * e_ace / b;
      }
      energy = static_cast<float>(e);
      born[i] = static_cast<float>(b);
      db_di[i] = static_cast<float>(dbdi);
      de_db[i] = detail::to_fixed_point(static_cast<float>(dedb), s.d_status);
    } else {
      born[i] = 0.0f;
      db_di[i] = 0.0f;
      de_db[i] = 0;
    }
  }
  detail::warp_add_energy(s, r, kEnergyNonbonded, energy);
}

/// Pass (c): Paarterme bei festen B. Nicht ausgeschlossen: (E_LJ + E_C + E_GB)·S(r); ausgeschlossen
/// (Nullbild, Maske): E_GB ohne Switching; Eigenbild ½. Kräfte, Energie und ∂E/∂B (Fixed-Point).
template <bool kImages>
__global__ void __launch_bounds__(kWarpsPerBlock * 32)
    k_gb_pairs(DeviceState s, const int2* tile_pairs, int n_tile_pairs, const float2* lj,
               const int* mask_index, const std::uint32_t* masks, const int4* replica,
               const float2* screening, float eps_in, SwitchParams sw, const float4* bounds,
               const float* born, long long* de_db) {
  extern __shared__ float2 sh_lj[];  // T × T (σ, ε)
  __shared__ float4 sh_pos[kWarpsPerBlock][32];
  __shared__ int sh_type[kWarpsPerBlock][32];
  __shared__ float sh_born[kWarpsPerBlock][32];
  __shared__ float4 sh_acc[kWarpsPerBlock][32];  // (Fx, Fy, Fz, ∂E/∂B_j)
  const int r = blockIdx.y;
  const int4 rep = replica[r];  // (lj_offset, n_types, mask_index_offset, mask_base)
  const int n_types = rep.y;
  for (int k = threadIdx.x; k < n_types * n_types; k += blockDim.x) sh_lj[k] = lj[rep.x + k];
  __syncthreads();

  const int warp = threadIdx.x / 32, a = threadIdx.x % 32;
  const int p = blockIdx.x * kWarpsPerBlock + warp;
  TileSetup t;
  if (!tile_setup<kImages>(s, tile_pairs, n_tile_pairs, bounds, sw.r_c, p, a, t)) return;
  const float4 pi = s.d_pos[t.base + t.i_local];
  int ti = s.d_types[t.base + t.i_local].x;
  int tj = s.d_types[t.base + t.j_local].x;
  if (ti >= n_types || tj >= n_types) atomicOr(s.d_status, kStatusUnknownType);
  ti = ti < n_types ? ti : 0;
  sh_pos[warp][a] = s.d_pos[t.base + t.j_local];
  sh_type[warp][a] = tj < n_types ? tj : 0;
  sh_born[warp][a] = born[t.base + t.j_local];
  sh_acc[warp][a] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  const float b_i = born[t.base + t.i_local];
  const int mi = mask_index[rep.z + p];
  const unsigned int excl = (mi >= 0 && t.valid_i) ? masks[rep.w + mi * 32 + a] : 0u;
  const float2 sc = screening[r];  // (κ, ε_out)
  const float inv_eps_in = 1.0f / eps_in, inv_eps_out = 1.0f / sc.y;
  const float c = units::ONE_4PI_EPS0_F;
  float3 fi = make_float3(0.0f, 0.0f, 0.0f);
  float de_db_i = 0.0f, energy = 0.0f;
  __syncwarp();

  for (int k = 0; k < 32; ++k) {
    const int b = (a + k) & 31;
    const float4 pj = sh_pos[warp][b];
    const float3 d0 = detail::minimum_image_delta(pi, pj, t.box);
    const float qq = pi.w * pj.w;
    const float b_j = sh_born[warp][b];
    const float2 se = sh_lj[ti * n_types + sh_type[warp][b]];  // (σ, ε)
    for (int nx = -t.n_img_x; nx <= t.n_img_x; ++nx) {
      for (int ny = -t.n_img_y; ny <= t.n_img_y; ++ny) {
        const bool zero_image = nx == 0 && ny == 0;
        if (((zero_image ? t.skip0 : t.skip_image) >> b) & 1u) continue;
        const bool self_image = kImages && !zero_image && t.tp.x == t.tp.y && b == a;
        const bool excluded = zero_image && ((excl >> b) & 1u);
        float3 d = d0;
        if constexpr (kImages) {
          d.x += static_cast<float>(nx) * t.box.x;
          d.y += static_cast<float>(ny) * t.box.y;
        }
        const float r2 = d.x * d.x + d.y * d.y + d.z * d.z;
        if (!(r2 < sw.r_c2 && r2 >= units::EPS_R2)) continue;
        const float dist = sqrtf(r2);
        const float inv_r = 1.0f / dist;
        // GB: E = −C (1/ε_in − e^{−κf}/ε_out) q_i q_j / f, f = sqrt(r² + P e^{−r²/(4P)}), P = B_i
        // B_j
        const float pp = b_i * b_j;
        const float ex = expf(-0.25f * r2 / pp);
        const float f = sqrtf(r2 + pp * ex);
        const float inv_f = 1.0f / f;
        const float ek = expf(-sc.x * f);
        const float g = inv_eps_in - ek * inv_eps_out;
        const float dv_df = -c * qq * (sc.x * ek * inv_eps_out * inv_f - g * inv_f * inv_f);
        const float df_dr = dist * (1.0f - 0.25f * ex) * inv_f;
        const float tb = 0.5f * ex * (1.0f + 0.25f * r2 / pp) * inv_f;  // ∂f/∂B_i = B_j·tb
        float v, dv_dr;                                                 // V und dV/dr
        float sv = 1.0f;
        if (excluded) {
          v = -c * g * qq * inv_f;
          dv_dr = dv_df * df_dr;
        } else {
          // E_C + E_GB = C qq [(1/ε_in)(1/r − 1/f) + e^{−κf}/(ε_out f)] ohne Auslöschung:
          // 1/r − 1/f = δ/(r f), δ = f − r = P e^{−r²/4P}/(f + r), dδ/dr = −(δ + r e/4)/f.
          // (Die direkte Summe C qq/(ε_in r) − C qq g/f verliert im Fernfeld ~e^{κr} an Stellen.)
          const float delta = pp * ex / (f + dist);
          const float ddelta = -(delta + 0.25f * dist * ex) * inv_f;
          const float t1 = delta * inv_r * inv_f;
          const float dt1 = ddelta * inv_r * inv_f - t1 * (inv_r + df_dr * inv_f);
          const float t2 = ek * inv_f;
          const float dt2 = -t2 * (sc.x + inv_f) * df_dr;
          v = c * qq * (inv_eps_in * t1 + inv_eps_out * t2);
          dv_dr = c * qq * (inv_eps_in * dt1 + inv_eps_out * dt2);
          if (se.y != 0.0f) {  // ε = 0: kein LJ (vermeidet 0·inf = NaN bei r ≪ σ)
            const float sr2 = se.x * se.x * inv_r * inv_r;
            const float sr6 = sr2 * sr2 * sr2;
            v += 4.0f * se.y * (sr6 * sr6 - sr6);
            dv_dr -= 4.0f * se.y * (12.0f * sr6 * sr6 - 6.0f * sr6) * inv_r;
          }
          float ds;
          detail::switching(dist, sw, sv, ds);
          dv_dr = dv_dr * sv + v * ds;
          v *= sv;
        }
        const float w = self_image ? 0.5f : 1.0f;
        energy += w * v;
        const float scale_b = w * sv * dv_df;
        if (self_image) {
          de_db_i += scale_b * (b_j + b_i) * tb;  // beide Radien gehören zu Atom i
        } else {
          de_db_i += scale_b * b_j * tb;
          const float cf = -dv_dr * inv_r;
          const float3 fv = make_float3(cf * d.x, cf * d.y, cf * d.z);
          fi.x += fv.x;
          fi.y += fv.y;
          fi.z += fv.z;
          float4& acc = sh_acc[warp][b];
          acc.x -= fv.x;
          acc.y -= fv.y;
          acc.z -= fv.z;
          acc.w += scale_b * b_i * tb;
        }
      }
    }
    __syncwarp();
  }
  const int gi = t.base + t.i_local, gj = t.base + t.j_local;
  if (t.valid_i) {
    if (fi.x != 0.0f || fi.y != 0.0f || fi.z != 0.0f) detail::add_force(s, gi, fi.x, fi.y, fi.z);
    if (de_db_i != 0.0f) detail::atomic_add_fixed_point(de_db + gi, de_db_i, s.d_status);
  }
  const float4 acc = sh_acc[warp][a];
  if (t.valid_j) {
    if (acc.x != 0.0f || acc.y != 0.0f || acc.z != 0.0f)
      detail::add_force(s, gj, acc.x, acc.y, acc.z);
    if (acc.w != 0.0f) detail::atomic_add_fixed_point(de_db + gj, acc.w, s.d_status);
  }
  detail::warp_add_energy(s, r, kEnergyNonbonded, energy);
}

/// Pass (e): F_i −= [dE/dI_i ∂(H_ij S)/∂r + dE/dI_j ∂(H_ji S)/∂r] r̂_ij, F_j = −F_i;
/// dE/dI = ∂E/∂B · dB/dI. Eigenbilder ohne Kraft.
template <bool kImages>
__global__ void __launch_bounds__(kWarpsPerBlock * 32)
    k_gb_chain(DeviceState s, const int2* tile_pairs, int n_tile_pairs, const float4* gb_atom,
               const int* gb_offset, const float2* neck, const int2* neck_info, SwitchParams sw,
               const float4* bounds, const long long* de_db, const float* db_di) {
  __shared__ float4 sh_pos[kWarpsPerBlock][32];
  __shared__ float4 sh_gb[kWarpsPerBlock][32];  // (or, sr, ρ, Neck-Index)
  __shared__ float sh_dedi[kWarpsPerBlock][32];
  __shared__ float3 sh_fj[kWarpsPerBlock][32];
  const int warp = threadIdx.x / 32, a = threadIdx.x % 32;
  TileSetup t;
  if (!tile_setup<kImages>(s, tile_pairs, n_tile_pairs, bounds, sw.r_c,
                           blockIdx.x * kWarpsPerBlock + warp, a, t))
    return;
  const int off = gb_offset[blockIdx.y];
  const int gi = t.base + t.i_local, gj = t.base + t.j_local;
  const float4 pi = s.d_pos[gi];
  const float4 gbi = gb_atom[off + t.i_local];
  const float dedi_i = detail::from_fixed_point(de_db[gi]) * db_di[gi];
  sh_pos[warp][a] = s.d_pos[gj];
  sh_gb[warp][a] = gb_atom[off + t.j_local];
  sh_dedi[warp][a] = detail::from_fixed_point(de_db[gj]) * db_di[gj];
  sh_fj[warp][a] = make_float3(0.0f, 0.0f, 0.0f);
  const int2 ni = neck_info[blockIdx.y];
  float3 fi = make_float3(0.0f, 0.0f, 0.0f);
  __syncwarp();
  for (int k = 0; k < 32; ++k) {
    const int b = (a + k) & 31;
    const float3 d0 = detail::minimum_image_delta(pi, sh_pos[warp][b], t.box);
    const float4 g_j = sh_gb[warp][b];
    const float dedi_j = sh_dedi[warp][b];
    const int idx_i = static_cast<int>(gbi.w), idx_j = static_cast<int>(g_j.w);
    for (int nx = -t.n_img_x; nx <= t.n_img_x; ++nx) {
      for (int ny = -t.n_img_y; ny <= t.n_img_y; ++ny) {
        const bool zero_image = nx == 0 && ny == 0;
        if (((zero_image ? t.skip0 : t.skip_image) >> b) & 1u) continue;
        if (kImages && !zero_image && t.tp.x == t.tp.y && b == a) continue;  // Eigenbild
        float3 d = d0;
        if constexpr (kImages) {
          d.x += static_cast<float>(nx) * t.box.x;
          d.y += static_cast<float>(ny) * t.box.y;
        }
        const float r2 = d.x * d.x + d.y * d.y + d.z * d.z;
        if (!(r2 < sw.r_c2 && r2 >= units::EPS_R2)) continue;
        const float dist = sqrtf(r2);
        float dh_ij, dh_ji, dn_ij, dn_ji, sv, ds;
        const float h_ij = h_integral(dist, gbi.x, g_j.y, dh_ij) +
                           neck_term(dist, gbi.z, g_j.z, idx_i, idx_j, neck, ni, dn_ij);
        const float h_ji = h_integral(dist, g_j.x, gbi.y, dh_ji) +
                           neck_term(dist, g_j.z, gbi.z, idx_j, idx_i, neck, ni, dn_ji);
        dh_ij += dn_ij;
        dh_ji += dn_ji;
        detail::switching(dist, sw, sv, ds);
        const float cf =
            -(dedi_i * (dh_ij * sv + h_ij * ds) + dedi_j * (dh_ji * sv + h_ji * ds)) / dist;
        const float3 fv = make_float3(cf * d.x, cf * d.y, cf * d.z);
        fi.x += fv.x;
        fi.y += fv.y;
        fi.z += fv.z;
        float3& fj = sh_fj[warp][b];
        fj.x -= fv.x;
        fj.y -= fv.y;
        fj.z -= fv.z;
      }
    }
    __syncwarp();
  }
  if (t.valid_i && (fi.x != 0.0f || fi.y != 0.0f || fi.z != 0.0f))
    detail::add_force(s, gi, fi.x, fi.y, fi.z);
  const float3 fj = sh_fj[warp][a];
  if (t.valid_j && (fj.x != 0.0f || fj.y != 0.0f || fj.z != 0.0f))
    detail::add_force(s, gj, fj.x, fj.y, fj.z);
}

void require(bool condition, const std::string& message) {
  if (!condition) throw std::invalid_argument("TiledGeneralizedBorn: " + message);
}

}  // namespace

TiledGeneralizedBorn::TiledGeneralizedBorn(GeneralizedBornParams params)
    : TiledAllPairs(params.base), gb_(std::move(params)) {
  const std::size_t n_topo = gb_.base.topologies.size();
  require(gb_.radius_nm.size() == n_topo && gb_.scale.size() == n_topo,
          "radius und scale je Topologie nötig");
  for (std::size_t t = 0; t < n_topo; ++t) {
    require(gb_.radius_nm[t].size() == gb_.scale[t].size(), "radius und scale gleich lang");
    for (std::size_t i = 0; i < gb_.radius_nm[t].size(); ++i)
      require(std::isfinite(gb_.radius_nm[t][i]) && gb_.radius_nm[t][i] > gb_.offset_nm &&
                  std::isfinite(gb_.scale[t][i]) && gb_.scale[t][i] >= 0.0f,
              "radius > offset und scale >= 0 erwartet");
    if (t < gb_.alpha_beta_gamma.size() && !gb_.alpha_beta_gamma[t].empty())
      require(gb_.alpha_beta_gamma[t].size() == 3 * gb_.radius_nm[t].size(),
              "alpha_beta_gamma: (n_atoms, 3) je Topologie");
    const int u = t < gb_.neck_size.size() ? gb_.neck_size[t] : 0;
    if (u > 0) {
      require(gb_.neck_m0[t].size() == static_cast<std::size_t>(u) * u &&
                  gb_.neck_d0[t].size() == static_cast<std::size_t>(u) * u &&
                  gb_.neck_index[t].size() == gb_.radius_nm[t].size(),
              "Neck: m0, d0 (U, U) und Index (n_atoms) je Topologie");
      for (int idx : gb_.neck_index[t]) require(idx >= 0 && idx < u, "Neck-Index außerhalb [0, U)");
    }
  }
  require(std::isfinite(gb_.offset_nm) && gb_.offset_nm >= 0.0f, "offset >= 0 erwartet");
  require(std::isfinite(gb_.eps_in) && gb_.eps_in > 0.0f, "eps_in > 0 erwartet");
}

void TiledGeneralizedBorn::initialize(SimulationContext& ctx) {
  TiledAllPairs::initialize(ctx);
  const int n_rep = ctx.n_replicas(), n_atoms = ctx.n_atoms(), n_pad = ctx.n_pad();
  std::vector<float4> atoms, abg;
  std::vector<float2> neck;
  std::vector<int> topo_offset;
  std::vector<int2> topo_neck;
  for (std::size_t t = 0; t < gb_.radius_nm.size(); ++t) {
    require(gb_.radius_nm[t].size() == static_cast<std::size_t>(n_atoms),
            "radius/scale brauchen n_atoms Einträge");
    const bool has_abg = t < gb_.alpha_beta_gamma.size() && !gb_.alpha_beta_gamma[t].empty();
    const int u = t < gb_.neck_size.size() ? gb_.neck_size[t] : 0;
    topo_offset.push_back(static_cast<int>(atoms.size()));
    topo_neck.push_back(make_int2(static_cast<int>(neck.size()), u));
    for (int k = 0; k < u * u; ++k)
      neck.push_back(make_float2(gb_.neck_m0[t][k], gb_.neck_d0[t][k]));
    for (int i = 0; i < n_pad; ++i) {
      if (i < n_atoms) {
        const float o = gb_.radius_nm[t][i] - gb_.offset_nm;
        const float idx = u > 0 ? static_cast<float>(gb_.neck_index[t][i]) : 0.0f;
        atoms.push_back(make_float4(o, gb_.scale[t][i] * o, gb_.radius_nm[t][i], idx));
        abg.push_back(has_abg ? make_float4(gb_.alpha_beta_gamma[t][3 * i],
                                            gb_.alpha_beta_gamma[t][3 * i + 1],
                                            gb_.alpha_beta_gamma[t][3 * i + 2], 0.0f)
                              : make_float4(1.0f, 0.8f, 4.85f, 0.0f));  // OBC2
      } else {
        atoms.push_back(make_float4(0.0f, 0.0f, 0.0f, 0.0f));  // Padding: kein Beitrag
        abg.push_back(make_float4(0.0f, 0.0f, 0.0f, 0.0f));
      }
    }
  }
  std::vector<int> offset(n_rep);
  std::vector<int2> neck_info(n_rep);
  for (int r = 0; r < n_rep; ++r) {
    offset[r] = topo_offset[params_.replica_topology[r]];
    neck_info[r] = topo_neck[params_.replica_topology[r]];
  }
  if (neck.empty()) neck.push_back(make_float2(0.0f, 0.0f));  // Puffer nie leer
  const cudaStream_t st = ctx.stream();
  const std::size_t n_total = static_cast<std::size_t>(n_rep) * n_pad;
  d_gb_atom_ = DeviceBuffer<float4>(atoms.size());
  d_gb_abg_ = DeviceBuffer<float4>(abg.size());
  d_neck_ = DeviceBuffer<float2>(neck.size());
  d_neck_info_ = DeviceBuffer<int2>(n_rep);
  d_gb_offset_ = DeviceBuffer<int>(n_rep);
  d_integral_ = DeviceBuffer<long long>(n_total);
  d_de_db_ = DeviceBuffer<long long>(n_total);
  d_born_ = DeviceBuffer<float>(n_total);
  d_db_di_ = DeviceBuffer<float>(n_total);
  d_gb_atom_.upload_async(atoms.data(), st);
  d_gb_abg_.upload_async(abg.data(), st);
  d_neck_.upload_async(neck.data(), st);
  d_neck_info_.upload_async(neck_info.data(), st);
  d_gb_offset_.upload_async(offset.data(), st);
  d_born_.zero_async(st);
  ctx.synchronize();  // Host-Vektoren leben bis hier
}

void TiledGeneralizedBorn::compute(DeviceState& s, cudaStream_t stream) {
  if (ctx_->host_box().empty()) throw std::logic_error("TiledGeneralizedBorn: Box nicht gesetzt");
  const SwitchParams sw = detail::make_switch(params_.r_switch_nm, params_.r_cutoff_nm);
  const dim3 grid((n_tile_pairs_ + kWarpsPerBlock - 1) / kWarpsPerBlock, s.n_replicas);
  const dim3 block(kWarpsPerBlock * 32);
  const std::size_t shmem = static_cast<std::size_t>(max_types_) * max_types_ * sizeof(float2);
  const bool images = uses_image_sum();
  if (!images) detail::launch_tile_bounds(s, d_tile_bounds_.data(), stream);
  d_integral_.zero_async(stream);

  const auto run = [&](auto tag) {
    constexpr bool kImg = decltype(tag)::value;
    k_gb_integrals<kImg><<<grid, block, 0, stream>>>(
        s, d_tile_pairs_.data(), n_tile_pairs_, d_gb_atom_.data(), d_gb_offset_.data(),
        d_neck_.data(), d_neck_info_.data(), sw, d_tile_bounds_.data(), d_integral_.data());
    BOMD_LAUNCH_CHECK();
    k_gb_radii<<<dim3(detail::elementwise_grid(s.n_pad), s.n_replicas), detail::kElementwiseBlock,
                 0, stream>>>(s, d_gb_atom_.data(), d_gb_abg_.data(), d_gb_offset_.data(),
                              d_screening_.data(), gb_.eps_in, gb_.ace ? 1 : 0, d_integral_.data(),
                              d_born_.data(), d_db_di_.data(), d_de_db_.data());
    BOMD_LAUNCH_CHECK();
    k_gb_pairs<kImg><<<grid, block, shmem, stream>>>(
        s, d_tile_pairs_.data(), n_tile_pairs_, d_lj_.data(), d_mask_index_.data(), d_masks_.data(),
        d_replica_.data(), d_screening_.data(), gb_.eps_in, sw, d_tile_bounds_.data(),
        d_born_.data(), d_de_db_.data());
    BOMD_LAUNCH_CHECK();
    k_gb_chain<kImg>
        <<<grid, block, 0, stream>>>(s, d_tile_pairs_.data(), n_tile_pairs_, d_gb_atom_.data(),
                                     d_gb_offset_.data(), d_neck_.data(), d_neck_info_.data(), sw,
                                     d_tile_bounds_.data(), d_de_db_.data(), d_db_di_.data());
    BOMD_LAUNCH_CHECK();
  };
  if (images)
    run(std::true_type{});
  else
    run(std::false_type{});
}

std::vector<float> TiledGeneralizedBorn::born_radii() const {
  std::vector<float> out(d_born_.size());
  ctx_->activate();
  d_born_.download_async(out.data(), ctx_->stream());
  BOMD_CUDA_CHECK(cudaStreamSynchronize(ctx_->stream()));
  return out;
}

}  // namespace bomd
