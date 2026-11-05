// BioOptoMD – Test-Hooks, siehe testing.hpp.
#include <stdexcept>
#include <string_view>

#include "biooptomd/cuda_check.hpp"
#include "core/core_kernels.cuh"
#include "core/fixed_point.cuh"
#include "core/mixed.cuh"
#include "core/philox.cuh"
#include "core/testing.hpp"

namespace bomd::testing {
namespace {

using detail::elementwise_grid;
using detail::kElementwiseBlock;

__global__ void k_shift_x(DeviceState s, float dx) {
  const int i_local = blockIdx.x * blockDim.x + threadIdx.x;
  const int r = blockIdx.y;
  if (i_local >= s.n_atoms) return;
  const int i = r * s.n_pad + i_local;
  double3 x = detail::load_position(s, i);  // FP64-Zustand (ADR 0009)
  x.x += dx;
  detail::store_position(s, i, x);
}

__global__ void k_accumulate_forces(DeviceState s, const int* atom, const float* f, int m) {
  const int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= m) return;
  detail::add_force(s, atom[k], f[3 * k], f[3 * k + 1], f[3 * k + 2]);
}

__global__ void k_uniform_force(DeviceState s, float fx, float fy, float fz) {
  const int i_local = blockIdx.x * blockDim.x + threadIdx.x;
  const int r = blockIdx.y;
  if (i_local >= s.n_atoms) return;
  detail::add_force(s, r * s.n_pad + i_local, fx, fy, fz);
}

__global__ void k_philox(const uint4* c, int n, uint2 key, uint4* out) {
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j < n) out[j] = detail::philox4x32_10(c[j], key);
}

class UniformForce final : public ForcePlugin {
 public:
  UniformForce(float fx, float fy, float fz) : fx_(fx), fy_(fy), fz_(fz) {}
  std::string_view name() const override { return "testing.UniformForce"; }
  bool contributes_energy() const override { return false; }
  void compute(DeviceState& s, cudaStream_t stream) override {
    const dim3 grid(elementwise_grid(s.n_atoms), s.n_replicas);
    k_uniform_force<<<grid, kElementwiseBlock, 0, stream>>>(s, fx_, fy_, fz_);
    BOMD_LAUNCH_CHECK();
  }

 private:
  float fx_, fy_, fz_;
};

}  // namespace

void shift_x(SimulationContext& ctx, float dx_nm) {
  ctx.activate();
  const DeviceState& s = ctx.state();
  const dim3 grid(elementwise_grid(s.n_atoms), s.n_replicas);
  k_shift_x<<<grid, kElementwiseBlock, 0, ctx.stream()>>>(s, dx_nm);
  BOMD_LAUNCH_CHECK();
}

void accumulate_forces(SimulationContext& ctx, const int* atom_global, const float* f_kj_mol_nm,
                       int m) {
  if (m <= 0) return;
  const int n_total = ctx.state().n_total();
  for (int k = 0; k < m; ++k)
    if (atom_global[k] < 0 || atom_global[k] >= n_total)
      throw std::invalid_argument("accumulate_forces: Atomindex außerhalb [0, n_replicas·n_pad)");
  ctx.activate();
  DeviceBuffer<int> d_atom(m);
  DeviceBuffer<float> d_f(3 * static_cast<std::size_t>(m));
  d_atom.upload_async(atom_global, ctx.stream());
  d_f.upload_async(f_kj_mol_nm, ctx.stream());
  k_accumulate_forces<<<elementwise_grid(m), kElementwiseBlock, 0, ctx.stream()>>>(
      ctx.state(), d_atom.data(), d_f.data(), m);
  BOMD_LAUNCH_CHECK();
  ctx.synchronize();  // Staging-Puffer leben bis hier.
}

void philox(const unsigned int* counters, int n, unsigned int key_lo, unsigned int key_hi,
            unsigned int* out) {
  if (n <= 0) return;
  DeviceBuffer<uint4> d_c(n), d_out(n);
  BOMD_CUDA_CHECK(cudaMemcpy(d_c.data(), counters, n * sizeof(uint4), cudaMemcpyHostToDevice));
  k_philox<<<elementwise_grid(n), kElementwiseBlock>>>(d_c.data(), n, make_uint2(key_lo, key_hi),
                                                       d_out.data());
  BOMD_LAUNCH_CHECK();
  BOMD_CUDA_CHECK(cudaMemcpy(out, d_out.data(), n * sizeof(uint4), cudaMemcpyDeviceToHost));
}

std::unique_ptr<ForcePlugin> make_uniform_force(float fx, float fy, float fz) {
  return std::make_unique<UniformForce>(fx, fy, fz);
}

}  // namespace bomd::testing
