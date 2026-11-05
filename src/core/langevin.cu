// BioOptoMD – BAOAB-Kernels, docs/PHYSICS_MODEL.md §1.
#include <cmath>
#include <stdexcept>

#include "biooptomd/cuda_check.hpp"
#include "biooptomd/langevin.hpp"
#include "biooptomd/units.hpp"
#include "core/core_kernels.cuh"
#include "core/mixed.cuh"
#include "core/philox.cuh"

namespace bomd {
namespace {

using detail::elementwise_grid;
using detail::kElementwiseBlock;

/// B A O A (PHYSICS_MODEL.md §1) in FP64 (ADR 0009). Ein Thread pro Atom, blockIdx.y = Replica.
__global__ void k_baoa(DeviceState s, double half_dt, double c1, double noise_coeff, uint2 key,
                       std::int64_t step) {
  const int i_local = blockIdx.x * blockDim.x + threadIdx.x;
  const int r = blockIdx.y;
  if (i_local >= s.n_atoms) return;
  const int i = r * s.n_pad + i_local;
  Vec4d v = s.d_vel[i];
  const double inv_m = v.w;
  if (inv_m == 0.0) return;  // unbeweglich: kein Rauschen, keine Bewegung (§1)
  double3 x = detail::load_position(s, i);
  const float4 f = s.d_force[i];

  // B
  v.x += half_dt * f.x * inv_m;
  v.y += half_dt * f.y * inv_m;
  v.z += half_dt * f.z * inv_m;
  // A
  x.x += half_dt * v.x;
  x.y += half_dt * v.y;
  x.z += half_dt * v.z;
  // O: c2 = sqrt((1 − c1²) k_B T_r / m)
  const double c2 = sqrt(noise_coeff * s.d_temperature[r] * inv_m);
  const uint4 bits = detail::philox4x32_10(
      detail::philox_counter(static_cast<unsigned int>(i_local), static_cast<unsigned int>(r), step,
                             detail::kPhiloxLangevin),
      key);
  const float2 n01 = detail::box_muller(bits.x, bits.y);
  const float2 n23 = detail::box_muller(bits.z, bits.w);
  v.x = c1 * v.x + c2 * n01.x;
  v.y = c1 * v.y + c2 * n01.y;
  v.z = c1 * v.z + c2 * n23.x;
  // A
  x.x += half_dt * v.x;
  x.y += half_dt * v.y;
  x.z += half_dt * v.z;

  detail::store_position(s, i, x);  // q unverändert
  s.d_vel[i] = v;                   // inv_mass (w) unverändert
}

/// Abschließendes B: v += Δt/2·F/m (FP64).
__global__ void k_b(DeviceState s, double half_dt) {
  const int i_local = blockIdx.x * blockDim.x + threadIdx.x;
  const int r = blockIdx.y;
  if (i_local >= s.n_atoms) return;
  const int i = r * s.n_pad + i_local;
  Vec4d v = s.d_vel[i];
  if (v.w == 0.0) return;
  const float4 f = s.d_force[i];
  v.x += half_dt * f.x * v.w;
  v.y += half_dt * f.y * v.w;
  v.z += half_dt * f.z * v.w;
  s.d_vel[i] = v;
}

}  // namespace

LangevinIntegrator::LangevinIntegrator(const LangevinParams& params) : params_(params) {
  if (!(std::isfinite(params.dt_ps) && params.dt_ps > 0.0))
    throw std::invalid_argument("LangevinIntegrator: dt_ps muss endlich und > 0 sein");
  if (!(std::isfinite(params.friction_per_ps) && params.friction_per_ps >= 0.0))
    throw std::invalid_argument("LangevinIntegrator: friction_per_ps muss endlich und >= 0 sein");
  const double gdt = params.friction_per_ps * params.dt_ps;
  half_dt_ = 0.5 * params.dt_ps;
  c1_ = std::exp(-gdt);
  noise_coeff_ = -std::expm1(-2.0 * gdt) * units::KB;  // (1 − c1²)·k_B
}

void LangevinIntegrator::first_half(const DeviceState& s, std::int64_t step,
                                    cudaStream_t stream) const {
  const dim3 grid(elementwise_grid(s.n_atoms), s.n_replicas);
  k_baoa<<<grid, kElementwiseBlock, 0, stream>>>(s, half_dt_, c1_, noise_coeff_,
                                                 detail::philox_key(params_.seed), step);
  BOMD_LAUNCH_CHECK();
}

void LangevinIntegrator::second_half(const DeviceState& s, cudaStream_t stream) const {
  const dim3 grid(elementwise_grid(s.n_atoms), s.n_replicas);
  k_b<<<grid, kElementwiseBlock, 0, stream>>>(s, half_dt_);
  BOMD_LAUNCH_CHECK();
}

}  // namespace bomd
