// BioOptoMD – PlanarSurfaceForce, docs/PHYSICS_MODEL.md §4.1.
#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

#include "biooptomd/cuda_check.hpp"
#include "biooptomd/simulation_context.hpp"
#include "core/core_kernels.cuh"
#include "core/fixed_point.cuh"
#include "forces/planar_surface.hpp"

namespace bomd {
namespace {

using detail::elementwise_grid;
using detail::kElementwiseBlock;

/// Ein Thread pro Atomplatz, blockIdx.y = Replica. Kein frühes return vor warp_add_energy:
/// alle Lanes müssen teilnehmen (Threads ohne echtes Atom tragen e = 0 bei).
__global__ void k_planar_surface(DeviceState s, const float4* type_params, int n_types, float z0,
                                 float phi0, float kappa, unsigned long long* counters) {
  const int i_local = blockIdx.x * blockDim.x + threadIdx.x;
  const int r = blockIdx.y;
  float energy = 0.0f;
  if (i_local < s.n_atoms) {
    const int i = r * s.n_pad + i_local;
    const float4 p = s.d_pos[i];
    const float zeta = p.z - z0;
    float fz = 0.0f;

    // 9-3-Wand (§4.1): V = ε[(2/15)(σ/ζ)^9 − (σ/ζ)^3], F_z = ε[(6/5)σ^9/ζ^10 − 3σ^3/ζ^4]
    const int t = s.d_types[i].y;
    if (t >= n_types) {
      atomicAdd(&counters[1], 1ull);
    } else {
      const float4 es = type_params[t];  // (ε, σ, ζ_c, 0)
      if (es.x != 0.0f && es.y > 0.0f) {
        // ζ_c = max(clamp_factor·σ, ζ_F) > 0 (ADR 0006) schützt die Division (AGENTS.md Regel 4)
        // und begrenzt |F| auf max_force. Unterhalb von ζ_c werden V und F am Punkt ζ_c
        // ausgewertet und das Ereignis gezählt.
        const float zeta_c = es.z;
        float z_eff = zeta;
        if (zeta < zeta_c) {
          z_eff = zeta_c;
          atomicAdd(&counters[0], 1ull);
        }
        const float sr = es.y / z_eff;
        const float s3 = sr * sr * sr;
        const float s9 = s3 * s3 * s3;
        energy += es.x * ((2.0f / 15.0f) * s9 - s3);
        fz += es.x * (1.2f * s9 - 3.0f * s3) / z_eff;
      }
    }

    // Linearisiertes Gouy-Chapman (§4.1): V = q Φ0 exp(−κζ), F_z = q Φ0 κ exp(−κζ) für ζ >= 0.
    // ASSUMPTION(OQ-09): für ζ < 0 konstant V = q Φ0, F = 0.
    if (phi0 != 0.0f && p.w != 0.0f) {
      const float ex = expf(-kappa * fmaxf(zeta, 0.0f));
      energy += p.w * phi0 * ex;
      if (zeta >= 0.0f) fz += p.w * phi0 * kappa * ex;
    }

    if (fz != 0.0f) detail::atomic_add_fixed_point(s.d_force_fp_z() + i, fz, s.d_status);
  }
  detail::warp_add_energy(s, r, kEnergySurface, energy);
}

/// F_wall(ζ) = ε[(6/5)σ^9/ζ^10 − 3σ^3/ζ^4] in FP64 (PHYSICS_MODEL.md §4.1).
double wall_force(double eps, double sigma, double zeta) {
  const double s3 = std::pow(sigma / zeta, 3.0);
  return eps * (1.2 * s3 * s3 * s3 - 3.0 * s3) / zeta;
}

/// ζ_c = max(clamp_factor·σ, ζ_F) mit |F(ζ_F)| = f_max (ADR 0006). Auf (0, z*) mit
/// z* = (2/5)^(1/6) σ < σ fällt |F| streng monoton (dF/dζ ∝ σ^6 − ζ^6), daher Bisektion.
float clamp_distance(double eps, double sigma, double clamp_factor, double f_max) {
  if (eps == 0.0 || sigma <= 0.0) return 0.0f;
  double lo = clamp_factor * sigma;
  const double z_star = std::pow(0.4, 1.0 / 6.0) * sigma;
  if (lo >= z_star || std::fabs(wall_force(eps, sigma, lo)) <= f_max) return static_cast<float>(lo);
  double hi = z_star;  // |F(hi)| = 0 < f_max < |F(lo)|
  for (int it = 0; it < 200; ++it) {
    const double mid = 0.5 * (lo + hi);
    (std::fabs(wall_force(eps, sigma, mid)) > f_max ? lo : hi) = mid;
  }
  return static_cast<float>(hi);  // |F(hi)| <= f_max
}

}  // namespace

PlanarSurfaceForce::PlanarSurfaceForce(PlanarSurfaceParams params) : params_(std::move(params)) {
  if (params_.epsilon_kj_mol.size() != params_.sigma_nm.size())
    throw std::invalid_argument("PlanarSurfaceForce: epsilon und sigma brauchen gleiche Länge");
  for (std::size_t t = 0; t < params_.sigma_nm.size(); ++t) {
    if (!std::isfinite(params_.epsilon_kj_mol[t]) || !std::isfinite(params_.sigma_nm[t]) ||
        params_.sigma_nm[t] < 0.0f)
      throw std::invalid_argument("PlanarSurfaceForce: ε endlich, σ endlich und >= 0");
  }
  if (!(std::isfinite(params_.clamp_factor) && params_.clamp_factor > 0.0f))
    throw std::invalid_argument("PlanarSurfaceForce: clamp_factor muss > 0 sein");
  if (!(std::isfinite(params_.max_force_kj_mol_nm) && params_.max_force_kj_mol_nm > 0.0f))
    throw std::invalid_argument("PlanarSurfaceForce: max_force muss > 0 sein");
  zeta_c_.resize(params_.sigma_nm.size());
  for (std::size_t t = 0; t < zeta_c_.size(); ++t)
    zeta_c_[t] = clamp_distance(params_.epsilon_kj_mol[t], params_.sigma_nm[t],
                                params_.clamp_factor, params_.max_force_kj_mol_nm);
  if (!std::isfinite(params_.z0_nm) || !std::isfinite(params_.phi0_kj_mol_e) ||
      !(std::isfinite(params_.kappa_per_nm) && params_.kappa_per_nm >= 0.0f))
    throw std::invalid_argument("PlanarSurfaceForce: z0, phi0 endlich; kappa endlich und >= 0");
}

void PlanarSurfaceForce::initialize(SimulationContext& ctx) {
  ctx_ = &ctx;
  const std::size_t n_types = params_.sigma_nm.size();
  std::vector<float4> tp(n_types);
  for (std::size_t t = 0; t < n_types; ++t)
    tp[t] = make_float4(params_.epsilon_kj_mol[t], params_.sigma_nm[t], zeta_c_[t], 0.0f);
  type_params_ = DeviceBuffer<float4>(n_types);
  type_params_.upload_async(tp.data(), ctx.stream());
  counters_ = DeviceBuffer<unsigned long long>(2);
  counters_.zero_async(ctx.stream());
  ctx.synchronize();  // tp lebt bis hier
}

void PlanarSurfaceForce::compute(DeviceState& s, cudaStream_t stream) {
  const dim3 grid(elementwise_grid(s.n_atoms), s.n_replicas);
  k_planar_surface<<<grid, kElementwiseBlock, 0, stream>>>(
      s, type_params_.data(), static_cast<int>(params_.sigma_nm.size()), params_.z0_nm,
      params_.phi0_kj_mol_e, params_.kappa_per_nm, counters_.data());
  BOMD_LAUNCH_CHECK();
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
