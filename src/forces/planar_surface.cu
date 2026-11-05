// BioOptoMD – PlanarSurfaceForce, docs/PHYSICS_MODEL.md §4.1, ADR 0007.
#include <cfloat>
#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

#include "biooptomd/cuda_check.hpp"
#include "biooptomd/simulation_context.hpp"
#include "biooptomd/units.hpp"
#include "core/core_kernels.cuh"
#include "core/fixed_point.cuh"
#include "forces/planar_surface.hpp"

namespace bomd {
namespace {

using detail::elementwise_grid;
using detail::kElementwiseBlock;

constexpr double kGraphiteShift = 0.61;  // a = 0.61 Δ im 10-4-3-Potential (Steele)

/// Wandenergie und F_z = −dV/dζ für ζ > 0 (ADR 0007), p = (C, σ, ζ_c, B), a = 0.61 Δ.
template <WallForm Form>
__device__ __forceinline__ void wall_energy_force(float4 p, float a, float zeta, float& e,
                                                  float& f) {
  const float s = p.y / zeta;
  if constexpr (Form == WallForm::kWall93) {
    // V = C[(2/15)s^9 − s^3], F = C[(6/5)s^9 − 3s^3]/ζ
    const float s3 = s * s * s;
    const float s9 = s3 * s3 * s3;
    e = p.x * ((2.0f / 15.0f) * s9 - s3);
    f = p.x * (1.2f * s9 - 3.0f * s3) / zeta;
  } else {
    // V = C[(2/5)s^10 − s^4], F = 4C[s^10 − s^4]/ζ
    const float s2 = s * s;
    const float s4 = s2 * s2;
    const float s10 = s4 * s4 * s2;
    e = p.x * (0.4f * s10 - s4);
    f = 4.0f * p.x * (s10 - s4) / zeta;
    if constexpr (Form == WallForm::kWall1043) {
      // Zusatzterm der tieferen Lagen: V = −B/(ζ + a)^3, F = −3B/(ζ + a)^4
      const float inv = 1.0f / (zeta + a);
      const float inv3 = inv * inv * inv;
      e -= p.w * inv3;
      f -= 3.0f * p.w * inv3 * inv;
    }
  }
}

/// Ein Thread pro Atomplatz, blockIdx.y = Replica. Kein frühes return vor warp_add_energy:
/// alle Lanes müssen teilnehmen (Threads ohne echtes Atom tragen e = 0 bei).
template <WallForm Form>
__global__ void k_planar_surface(DeviceState s, const float4* type_params, int n_surface_types,
                                 int n_lj_types, float a, float z0, float phi0, bool gc_linear,
                                 const float* kappa_r, unsigned long long* counters) {
  const int i_local = blockIdx.x * blockDim.x + threadIdx.x;
  const int r = blockIdx.y;
  float energy = 0.0f;
  if (i_local < s.n_atoms) {
    const int i = r * s.n_pad + i_local;
    const float4 pos = s.d_pos[i];
    const float zeta = pos.z - z0;
    float fz = 0.0f;

    const int2 t = s.d_types[i];
    const int lj = n_lj_types == 1 ? 0 : t.x;
    if (t.y < 0 || t.y >= n_surface_types || lj < 0 || lj >= n_lj_types) {
      atomicAdd(&counters[1], 1ull);
    } else {
      const float4 p = type_params[t.y * n_lj_types + lj];  // (C, σ, ζ_c, B)
      if (p.x != 0.0f && p.y > 0.0f) {
        // ζ_c = max(clamp_factor·σ, ζ_F) > 0 (ADR 0008) schützt die Division (AGENTS.md Regel 4)
        // und begrenzt |F| auf max_force. Unterhalb von ζ_c werden V und F am Punkt ζ_c
        // ausgewertet und das Ereignis gezählt.
        float z_eff = zeta;
        if (zeta < p.z) {
          z_eff = p.z;
          atomicAdd(&counters[0], 1ull);
        }
        float e = 0.0f, f = 0.0f;
        wall_energy_force<Form>(p, a, z_eff, e, f);
        energy += e;
        fz += f;
      }
    }

    // Gouy-Chapman (OQ-06), PHYSICS_MODEL §4.1. ASSUMPTION(OQ-12): für ζ < 0 V = V(0), F = 0.
    if (phi0 != 0.0f && pos.w != 0.0f) {
      const float kappa = kappa_r[r];
      const float zp = fmaxf(zeta, 0.0f);
      const float ex = expf(-kappa * zp);
      if (gc_linear) {
        // V = q Φ0 exp(−κζ), F_z = q Φ0 κ exp(−κζ)
        energy += pos.w * phi0 * ex;
        if (zeta >= 0.0f) fz += pos.w * phi0 * kappa * ex;
      } else {
        const float kt = units::KB_F * s.d_temperature[r];
        // T_r = 0: Grenzfall vollständiger Abschirmung, kein Beitrag (kt im Nenner).
        if (kt > 0.0f) {
          // V = q·2k_BT·ln[(1+u)/(1−u)], F_z = q·4k_BT·κ·u/(1−u²), u = γ e^{−κζ}, γ = tanh(x).
          // Mit |Φ0| gerechnet (u > 0), Vorzeichen von Φ0 am Ende. 1 − u ohne Auslöschung:
          // 1 − u = (1 − e^{−κζ}) + (1 − γ) e^{−κζ}, 1 − γ = 2e^{−2x}/(1 + e^{−2x}); damit bleibt
          // V auch für γ → 1 (große Φ0) endlich.
          const float x = fabsf(phi0) / (4.0f * kt);
          const float em = expf(-2.0f * x);
          const float u = tanhf(x) * ex;
          const float one_minus_u =
              fmaxf(-expm1f(-kappa * zp) + 2.0f * em / (1.0f + em) * ex, FLT_MIN);
          const float qs = phi0 > 0.0f ? pos.w : -pos.w;
          energy += qs * 2.0f * kt * (log1pf(u) - logf(one_minus_u));
          if (zeta >= 0.0f) fz += qs * 4.0f * kt * kappa * u / ((1.0f + u) * one_minus_u);
        }
      }
    }

    if (fz != 0.0f) detail::atomic_add_fixed_point(s.d_force_fp_z() + i, fz, s.d_status);
  }
  detail::warp_add_energy(s, r, kEnergySurface, energy);
}

/// F_z(ζ) der Wandform in FP64 (wie wall_energy_force, PHYSICS_MODEL.md §4.1).
double wall_force(WallForm form, double c, double sigma, double delta, double zeta) {
  const double s = sigma / zeta;
  if (form == WallForm::kWall93) {
    const double s3 = s * s * s;
    return c * (1.2 * s3 * s3 * s3 - 3.0 * s3) / zeta;
  }
  const double s4 = std::pow(s, 4.0);
  double f = 4.0 * c * (s4 * s4 * s * s - s4) / zeta;
  if (form == WallForm::kWall1043)
    f -= c * std::pow(sigma, 4.0) / delta / std::pow(zeta + kGraphiteShift * delta, 4.0);
  return f;
}

/// ζ_c = max(clamp_factor·σ, ζ_F) mit |F(ζ_F)| = f_max (ADR 0008). Der Nulldurchgang z_0 der Kraft
/// liegt für alle Formen in (σ/2, 3σ/2) (F(σ/2) > 0 > F(3σ/2)); auf (0, z_0) fällt F von +∞
/// monoton (für 10-4-3 dominiert dort der repulsive s^10-Term). Beides per Bisektion.
float clamp_distance(WallForm form, double c, double sigma, double delta, double clamp_factor,
                     double f_max) {
  if (c == 0.0 || sigma <= 0.0) return 0.0f;
  double lo = 0.5 * sigma, hi = 1.5 * sigma;
  for (int it = 0; it < 200; ++it) {
    const double mid = 0.5 * (lo + hi);
    (wall_force(form, c, sigma, delta, mid) > 0.0 ? lo : hi) = mid;
  }
  const double z_zero = hi;
  lo = clamp_factor * sigma;
  if (lo >= z_zero || std::fabs(wall_force(form, c, sigma, delta, lo)) <= f_max)
    return static_cast<float>(lo);
  hi = z_zero;  // |F(hi)| ≈ 0 < f_max < |F(lo)|
  for (int it = 0; it < 200; ++it) {
    const double mid = 0.5 * (lo + hi);
    (std::fabs(wall_force(form, c, sigma, delta, mid)) > f_max ? lo : hi) = mid;
  }
  return static_cast<float>(hi);  // |F(hi)| <= f_max
}

}  // namespace

PlanarSurfaceForce::PlanarSurfaceForce(PlanarSurfaceParams params) : params_(std::move(params)) {
  const int form = static_cast<int>(params_.form);
  if (form < 0 || form > 2) throw std::invalid_argument("PlanarSurfaceForce: unbekannte Wandform");
  if (params_.n_lj_types < 1) throw std::invalid_argument("PlanarSurfaceForce: n_lj_types >= 1");
  const std::size_t n = params_.prefactor_kj_mol.size();
  if (params_.sigma_nm.size() != n || n % params_.n_lj_types != 0)
    throw std::invalid_argument(
        "PlanarSurfaceForce: prefactor und sigma brauchen gleiche Länge n_surface·n_lj_types");
  n_surface_types_ = static_cast<int>(n / params_.n_lj_types);
  for (std::size_t k = 0; k < n; ++k) {
    if (!(std::isfinite(params_.prefactor_kj_mol[k]) && params_.prefactor_kj_mol[k] >= 0.0f) ||
        !(std::isfinite(params_.sigma_nm[k]) && params_.sigma_nm[k] >= 0.0f))
      throw std::invalid_argument("PlanarSurfaceForce: C, σ endlich und >= 0");
  }
  if (!(std::isfinite(params_.delta_nm) && params_.delta_nm > 0.0f))
    throw std::invalid_argument("PlanarSurfaceForce: delta muss > 0 sein");
  if (!(std::isfinite(params_.clamp_factor) && params_.clamp_factor > 0.0f))
    throw std::invalid_argument("PlanarSurfaceForce: clamp_factor muss > 0 sein");
  if (!(std::isfinite(params_.max_force_kj_mol_nm) && params_.max_force_kj_mol_nm > 0.0f))
    throw std::invalid_argument("PlanarSurfaceForce: max_force muss > 0 sein");
  if (!std::isfinite(params_.z0_nm) || !std::isfinite(params_.phi0_kj_mol_e))
    throw std::invalid_argument("PlanarSurfaceForce: z0, phi0 endlich");
  for (float k : params_.kappa_per_nm)
    if (!(std::isfinite(k) && k >= 0.0f))
      throw std::invalid_argument("PlanarSurfaceForce: kappa endlich und >= 0");
  zeta_c_.resize(n);
  for (std::size_t k = 0; k < n; ++k)
    zeta_c_[k] =
        clamp_distance(params_.form, params_.prefactor_kj_mol[k], params_.sigma_nm[k],
                       params_.delta_nm, params_.clamp_factor, params_.max_force_kj_mol_nm);
}

void PlanarSurfaceForce::initialize(SimulationContext& ctx) {
  ctx_ = &ctx;
  const int n_rep = ctx.n_replicas();
  if (params_.phi0_kj_mol_e != 0.0f && static_cast<int>(params_.kappa_per_nm.size()) != n_rep)
    throw std::invalid_argument("PlanarSurfaceForce: kappa braucht einen Wert je Replica");
  std::vector<float> kappa(params_.kappa_per_nm);
  kappa.resize(n_rep, 0.0f);
  const std::size_t n = params_.sigma_nm.size();
  std::vector<float4> tp(n);
  for (std::size_t k = 0; k < n; ++k) {
    const double c = params_.prefactor_kj_mol[k], sigma = params_.sigma_nm[k];
    const double b = params_.form == WallForm::kWall1043
                         ? c * std::pow(sigma, 4.0) / (3.0 * params_.delta_nm)
                         : 0.0;
    tp[k] = make_float4(params_.prefactor_kj_mol[k], params_.sigma_nm[k], zeta_c_[k],
                        static_cast<float>(b));
  }
  type_params_ = DeviceBuffer<float4>(n);
  type_params_.upload_async(tp.data(), ctx.stream());
  kappa_ = DeviceBuffer<float>(n_rep);
  kappa_.upload_async(kappa.data(), ctx.stream());
  counters_ = DeviceBuffer<unsigned long long>(2);
  counters_.zero_async(ctx.stream());
  ctx.synchronize();  // tp, kappa leben bis hier
}

void PlanarSurfaceForce::compute(DeviceState& s, cudaStream_t stream) {
  const dim3 grid(elementwise_grid(s.n_atoms), s.n_replicas);
  const float a = static_cast<float>(kGraphiteShift * params_.delta_nm);
  const auto launch = [&](auto kernel) {
    kernel<<<grid, kElementwiseBlock, 0, stream>>>(
        s, type_params_.data(), n_surface_types_, params_.n_lj_types, a, params_.z0_nm,
        params_.phi0_kj_mol_e, params_.gouy_chapman_linear, kappa_.data(), counters_.data());
    BOMD_LAUNCH_CHECK();
  };
  switch (params_.form) {
    case WallForm::kWall93:
      launch(k_planar_surface<WallForm::kWall93>);
      break;
    case WallForm::kWall104:
      launch(k_planar_surface<WallForm::kWall104>);
      break;
    case WallForm::kWall1043:
      launch(k_planar_surface<WallForm::kWall1043>);
      break;
  }
}

std::uint64_t PlanarSurfaceForce::counter(int index) const {
  unsigned long long value = 0;
  ctx_->activate();
  BOMD_CUDA_CHECK(cudaMemcpyAsync(&value, counters_.data() + index, sizeof(value),
                                  cudaMemcpyDeviceToHost, ctx_->stream()));
  BOMD_CUDA_CHECK(cudaStreamSynchronize(ctx_->stream()));
  return value;
}

std::uint64_t PlanarSurfaceForce::clamp_count() const {
  return counter(0);
}
std::uint64_t PlanarSurfaceForce::unknown_type_count() const {
  return counter(1);
}

void PlanarSurfaceForce::reset_counters() {
  ctx_->activate();
  counters_.zero_async(ctx_->stream());
}

}  // namespace bomd
