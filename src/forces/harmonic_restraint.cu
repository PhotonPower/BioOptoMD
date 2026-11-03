// BioOptoMD – HarmonicRestraintPullingForce, docs/PHYSICS_MODEL.md §4.2.
#include <cmath>
#include <stdexcept>
#include <utility>

#include "biooptomd/cuda_check.hpp"
#include "biooptomd/simulation_context.hpp"
#include "core/core_kernels.cuh"
#include "core/fixed_point.cuh"
#include "core/pbc.cuh"
#include "forces/harmonic_restraint.hpp"

namespace bomd {
namespace {

using detail::elementwise_grid;
using detail::kElementwiseBlock;

void require(bool condition, const char* message) {
  if (!condition) throw std::invalid_argument(message);
}

/// Positions-Restraint: Δ = r − r⁰ (Minimum Image in x/y), V = ½ k Σ mask Δ², F = −k mask Δ.
__global__ void k_position_restraints(DeviceState s, const int2* atom, const float4* ref, int m,
                                      float mx, float my, float mz) {
  const int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= m) return;
  const int i = atom[k].x;
  const int r = atom[k].y;
  const float4 p = s.d_pos[i];
  const float4 q = ref[k];
  const float2 box = s.d_box[r];
  const float3 d = detail::minimum_image_delta(p, q, box);  // ADR 0003; Box > 0 garantiert
  const float dx = d.x, dy = d.y, dz = d.z;
  const float kk = q.w;
  const float energy = 0.5f * kk * (mx * dx * dx + my * dy * dy + mz * dz * dz);
  detail::add_force(s, i, -kk * mx * dx, -kk * my * dy, -kk * mz * dz);
  detail::add_energy(s, r, kEnergyRestraint, energy);
}

constexpr int kPullBlock = 256;

/// Schwerpunkt-Pulling, ein Block pro Replica. R_G aus ungewrappten Koordinaten x + n·L in FP64
/// (feste Reduktionsreihenfolge ⇒ deterministisch). u = ê·(R_G − R₀) − v t,
/// V = ½ k u², F_i = −k u w_i ê; bei do_work: W += ½k[(s − v t)² − (s − v t_prev)²].
__global__ void k_com_pulling(DeviceState s, const int* offsets, const int* atoms,
                              const float* weights, const float4* r0_k, const float4* dir_v,
                              double t, double t_prev, int do_work, double* work,
                              double* extension) {
  __shared__ double sx[kPullBlock], sy[kPullBlock], sz[kPullBlock];
  __shared__ float s_u;
  const int r = blockIdx.x;
  const int begin = offsets[r], end = offsets[r + 1];
  const float2 box = s.d_box[r];

  double ax = 0.0, ay = 0.0, az = 0.0;
  for (int g = begin + threadIdx.x; g < end; g += blockDim.x) {
    const int i = r * s.n_pad + atoms[g];
    const float4 p = s.d_pos[i];
    const int2 img = s.d_image[i];
    const double w = weights[g];
    ax += w * (static_cast<double>(p.x) + static_cast<double>(img.x) * box.x);
    ay += w * (static_cast<double>(p.y) + static_cast<double>(img.y) * box.y);
    az += w * static_cast<double>(p.z);
  }
  sx[threadIdx.x] = ax;
  sy[threadIdx.x] = ay;
  sz[threadIdx.x] = az;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      sx[threadIdx.x] += sx[threadIdx.x + stride];
      sy[threadIdx.x] += sy[threadIdx.x + stride];
      sz[threadIdx.x] += sz[threadIdx.x + stride];
    }
    __syncthreads();
  }

  const float4 rk = r0_k[r];
  const float4 ev = dir_v[r];
  if (threadIdx.x == 0) {
    const double proj = ev.x * (sx[0] - rk.x) + ev.y * (sy[0] - rk.y) + ev.z * (sz[0] - rk.z);
    const double u = proj - static_cast<double>(ev.w) * t;
    detail::add_energy(s, r, kEnergyRestraint, static_cast<float>(0.5 * rk.w * u * u));
    if (do_work) {
      const double u_prev = proj - static_cast<double>(ev.w) * t_prev;
      work[r] += 0.5 * rk.w * (u * u - u_prev * u_prev);
    }
    extension[r] = u;
    s_u = static_cast<float>(u);
  }
  __syncthreads();

  const float coeff = -rk.w * s_u;  // F_i = −k u w_i ê
  for (int g = begin + threadIdx.x; g < end; g += blockDim.x) {
    const float f = coeff * weights[g];
    detail::add_force(s, r * s.n_pad + atoms[g], f * ev.x, f * ev.y, f * ev.z);
  }
}

}  // namespace

HarmonicRestraintPullingForce::HarmonicRestraintPullingForce(PositionRestraintParams restraints)
    : pulling_(false), restraints_(std::move(restraints)) {
  const std::size_t m = restraints_.atom.size();
  require(restraints_.replica.size() == m && restraints_.k_kj_mol_nm2.size() == m &&
              restraints_.reference_nm.size() == 3 * m,
          "PositionRestraints: replica, atom, reference (m,3), k müssen zusammenpassen");
  for (float k : restraints_.k_kj_mol_nm2)
    require(std::isfinite(k) && k >= 0.0f, "PositionRestraints: k endlich und >= 0");
  for (float x : restraints_.reference_nm)
    require(std::isfinite(x), "PositionRestraints: Referenz nicht endlich");
}

HarmonicRestraintPullingForce::HarmonicRestraintPullingForce(ComPullingParams pulling)
    : pulling_(true), pull_(std::move(pulling)) {}

void HarmonicRestraintPullingForce::initialize(SimulationContext& ctx) {
  ctx_ = &ctx;
  const int n_rep = ctx.n_replicas();
  const cudaStream_t st = ctx.stream();

  if (!pulling_) {
    const std::size_t m = restraints_.atom.size();
    std::vector<int2> atom(m);
    std::vector<float4> ref(m);
    for (std::size_t k = 0; k < m; ++k) {
      const int r = restraints_.replica[k], a = restraints_.atom[k];
      require(r >= 0 && r < n_rep && a >= 0 && a < ctx.n_atoms(),
              "PositionRestraints: Replica- oder Atomindex außerhalb des Bereichs");
      atom[k] = make_int2(r * ctx.n_pad() + a, r);
      ref[k] = make_float4(restraints_.reference_nm[3 * k], restraints_.reference_nm[3 * k + 1],
                           restraints_.reference_nm[3 * k + 2], restraints_.k_kj_mol_nm2[k]);
    }
    d_restraint_atom_ = DeviceBuffer<int2>(m);
    d_restraint_ref_ = DeviceBuffer<float4>(m);
    d_restraint_atom_.upload_async(atom.data(), st);
    d_restraint_ref_.upload_async(ref.data(), st);
    ctx.synchronize();
    return;
  }

  // --- Pulling: Validierung und Gewichte w_i = m_i / M_G -------------------------------------
  const auto& off = pull_.group_offsets;
  require(off.size() == static_cast<std::size_t>(n_rep) + 1 && off.front() == 0 &&
              off.back() == static_cast<int>(pull_.group_atoms.size()),
          "ComPulling: group_offsets braucht n_replicas + 1 Einträge, 0 … len(group_atoms)");
  require(pull_.r0_nm.size() == 3u * n_rep && pull_.direction.size() == 3u * n_rep &&
              pull_.velocity_nm_ps.size() == static_cast<std::size_t>(n_rep) &&
              pull_.k_kj_mol_nm2.size() == static_cast<std::size_t>(n_rep),
          "ComPulling: r0/direction (n_replicas,3), velocity/k (n_replicas,) erwartet");

  std::vector<float4> r0_k(n_rep), dir_v(n_rep);
  for (int r = 0; r < n_rep; ++r) {
    require(off[r] < off[r + 1], "ComPulling: jede Replica braucht eine nichtleere Gruppe");
    std::vector<bool> seen(ctx.n_atoms(), false);
    for (int g = off[r]; g < off[r + 1]; ++g) {
      const int a = pull_.group_atoms[g];
      require(a >= 0 && a < ctx.n_atoms(), "ComPulling: Atomindex außerhalb [0, n_atoms)");
      require(!seen[a], "ComPulling: Atom doppelt in der Gruppe");
      seen[a] = true;
    }
    const double ex = pull_.direction[3 * r], ey = pull_.direction[3 * r + 1],
                 ez = pull_.direction[3 * r + 2];
    const double norm = std::sqrt(ex * ex + ey * ey + ez * ez);
    require(std::isfinite(norm) && norm > 0.0, "ComPulling: Zugrichtung muss != 0 sein");
    const float k = pull_.k_kj_mol_nm2[r], v = pull_.velocity_nm_ps[r];
    require(std::isfinite(k) && k >= 0.0f && std::isfinite(v), "ComPulling: k >= 0, v endlich");
    r0_k[r] = make_float4(pull_.r0_nm[3 * r], pull_.r0_nm[3 * r + 1], pull_.r0_nm[3 * r + 2], k);
    dir_v[r] = make_float4(static_cast<float>(ex / norm), static_cast<float>(ey / norm),
                           static_cast<float>(ez / norm), v);
  }

  d_group_offsets_ = DeviceBuffer<int>(off.size());
  d_group_atoms_ = DeviceBuffer<int>(pull_.group_atoms.size());
  d_group_weights_ = DeviceBuffer<float>(pull_.group_atoms.size());
  d_r0_k_ = DeviceBuffer<float4>(n_rep);
  d_dir_v_ = DeviceBuffer<float4>(n_rep);
  d_work_ = DeviceBuffer<double>(n_rep);
  d_extension_ = DeviceBuffer<double>(n_rep);
  d_group_offsets_.upload_async(off.data(), st);
  d_group_atoms_.upload_async(pull_.group_atoms.data(), st);
  d_r0_k_.upload_async(r0_k.data(), st);
  d_dir_v_.upload_async(dir_v.data(), st);
  d_work_.zero_async(st);
  d_extension_.zero_async(st);
  ctx.synchronize();
  update_weights();
}

void HarmonicRestraintPullingForce::update_weights() {
  const SimulationContext& ctx = *ctx_;
  const auto& off = pull_.group_offsets;
  std::vector<float4> vel(static_cast<std::size_t>(ctx.n_replicas()) * ctx.n_pad());
  ctx.activate();
  BOMD_CUDA_CHECK(cudaMemcpyAsync(vel.data(), ctx.state().d_vel, vel.size() * sizeof(float4),
                                  cudaMemcpyDeviceToHost, ctx.stream()));
  BOMD_CUDA_CHECK(cudaStreamSynchronize(ctx.stream()));
  std::vector<float> weights(pull_.group_atoms.size());
  for (int r = 0; r < ctx.n_replicas(); ++r) {
    double total = 0.0;
    for (int g = off[r]; g < off[r + 1]; ++g) {
      const float inv_m = vel[static_cast<std::size_t>(r) * ctx.n_pad() + pull_.group_atoms[g]].w;
      require(inv_m > 0.0f, "ComPulling: Gruppenatome brauchen gesetzte Massen > 0");
      total += 1.0 / inv_m;
    }
    for (int g = off[r]; g < off[r + 1]; ++g) {
      const float inv_m = vel[static_cast<std::size_t>(r) * ctx.n_pad() + pull_.group_atoms[g]].w;
      weights[g] = static_cast<float>((1.0 / inv_m) / total);
    }
  }
  d_group_weights_.upload_async(weights.data(), ctx.stream());
  BOMD_CUDA_CHECK(cudaStreamSynchronize(ctx.stream()));  // weights lebt bis hier
  masses_version_ = ctx.masses_version();
}

void HarmonicRestraintPullingForce::compute(DeviceState& s, cudaStream_t stream) {
  if (!pulling_) {
    const int m = static_cast<int>(restraints_.atom.size());
    if (m == 0) return;
    const auto& d = restraints_.dims;
    k_position_restraints<<<elementwise_grid(m), kElementwiseBlock, 0, stream>>>(
        s, d_restraint_atom_.data(), d_restraint_ref_.data(), m, d[0] ? 1.0f : 0.0f,
        d[1] ? 1.0f : 0.0f, d[2] ? 1.0f : 0.0f);
    BOMD_LAUNCH_CHECK();
    return;
  }
  if (masses_version_ != ctx_->masses_version()) update_weights();  // nur nach set_masses
  const std::int64_t step = ctx_->current_step();
  const bool do_work = has_last_step_ && step == last_step_ + 1;
  k_com_pulling<<<s.n_replicas, kPullBlock, 0, stream>>>(
      s, d_group_offsets_.data(), d_group_atoms_.data(), d_group_weights_.data(), d_r0_k_.data(),
      d_dir_v_.data(), ctx_->time_ps(), ctx_->previous_time_ps(), do_work ? 1 : 0, d_work_.data(),
      d_extension_.data());
  BOMD_LAUNCH_CHECK();
  last_step_ = step;
  has_last_step_ = true;
}

std::vector<double> HarmonicRestraintPullingForce::work() const {
  if (!pulling_) return {};
  std::vector<double> out(d_work_.size());
  ctx_->activate();
  d_work_.download_async(out.data(), ctx_->stream());
  BOMD_CUDA_CHECK(cudaStreamSynchronize(ctx_->stream()));
  return out;
}

std::vector<double> HarmonicRestraintPullingForce::extension() const {
  if (!pulling_) return {};
  std::vector<double> out(d_extension_.size());
  ctx_->activate();
  d_extension_.download_async(out.data(), ctx_->stream());
  BOMD_CUDA_CHECK(cudaStreamSynchronize(ctx_->stream()));
  return out;
}

void HarmonicRestraintPullingForce::reset_work() {
  if (!pulling_) return;
  ctx_->activate();
  d_work_.zero_async(ctx_->stream());
}

}  // namespace bomd
