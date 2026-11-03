// BioOptoMD – SHAKE/RATTLE-Kernels, siehe constraints.hpp und tests/reference/constraints_ref.py.
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>

#include "biooptomd/cuda_check.hpp"
#include "biooptomd/simulation_context.hpp"
#include "core/constraints.hpp"
#include "core/core_kernels.cuh"
#include "core/mixed.cuh"

namespace bomd {
namespace {

/// Ein Thread pro Cluster; wenig Arbeit pro Thread, daher kleine Blöcke für viele Replicas.
constexpr int kClusterBlock = 128;
constexpr int kMaxH = 3;

struct Cluster {
  int n_h;
  int atom[1 + kMaxH];   // globale Indizes; [0] = Schweratom
  double w[1 + kMaxH];   // inverse Massen
  double d2[kMaxH];      // Sollabstand²
  double3 x[1 + kMaxH];  // Positionen (FP64)
  double3 v[1 + kMaxH];  // Geschwindigkeiten; w (1/m) separat in w[]
};

__device__ __forceinline__ double3 sub(double3 a, double3 b) {
  return make_double3(a.x - b.x, a.y - b.y, a.z - b.z);
}
__device__ __forceinline__ double dot3(double3 a, double3 b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}
/// a − b mit Minimum Image in x/y (FP64; Cluster-Atome liegen nahe beieinander).
__device__ __forceinline__ double3 delta(double3 a, double3 b, float2 box) {
  double3 d = sub(a, b);
  d.x -= box.x * rint(d.x / box.x);
  d.y -= box.y * rint(d.y / box.y);
  return d;
}

__device__ __forceinline__ bool load_cluster(const DeviceState& s, const int4* clusters,
                                             const float4* distances, const int2* range, int r,
                                             int t, Cluster& c) {
  const int2 rg = range[r];
  if (t >= rg.y) return false;
  const int4 id = clusters[rg.x + t];
  const float4 dist = distances[rg.x + t];
  const int ids[4] = {id.x, id.y, id.z, id.w};
  const float ds[3] = {dist.x, dist.y, dist.z};
  const int base = r * s.n_pad;
  c.n_h = 0;
  for (int k = 0; k < 1 + kMaxH; ++k) {
    if (k > 0 && ids[k] < 0) break;
    c.atom[k] = base + ids[k];
    const Vec4d vel = s.d_vel[c.atom[k]];
    c.v[k] = make_double3(vel.x, vel.y, vel.z);
    c.w[k] = vel.w;
    c.x[k] = detail::load_position(s, c.atom[k]);
    if (k > 0) {
      c.d2[k - 1] = static_cast<double>(ds[k - 1]) * ds[k - 1];
      ++c.n_h;
    }
  }
  return true;
}

__device__ __forceinline__ void store_cluster(const DeviceState& s, const Cluster& c) {
  for (int k = 0; k <= c.n_h; ++k) {
    detail::store_position(s, c.atom[k], c.x[k]);
    s.d_vel[c.atom[k]] = Vec4d{c.v[k].x, c.v[k].y, c.v[k].z, c.w[k]};
  }
}

/// SHAKE (optional nach einem A-Schritt mit h = Δt/2): Referenzrichtungen s_k vor dem Schritt.
__global__ void k_shake(DeviceState s, const int4* clusters, const float4* distances,
                        const int2* range, double h, int drift, double tol, int max_iter) {
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  const int r = blockIdx.y;
  Cluster c;
  if (!load_cluster(s, clusters, distances, range, r, t, c)) return;
  const float2 box = s.d_box[r];
  double3 ref[kMaxH];
  for (int k = 1; k <= c.n_h; ++k) ref[k - 1] = delta(c.x[k], c.x[0], box);
  double3 x_un[1 + kMaxH];
  for (int k = 0; k <= c.n_h; ++k) {
    if (drift) {
      c.x[k].x += h * c.v[k].x;
      c.x[k].y += h * c.v[k].y;
      c.x[k].z += h * c.v[k].z;
    }
    x_un[k] = c.x[k];
  }
  bool converged = false;
  for (int it = 0; it < max_iter && !converged; ++it) {
    converged = true;
    for (int k = 1; k <= c.n_h; ++k) {
      const double3 rk = delta(c.x[k], c.x[0], box);
      const double diff = dot3(rk, rk) - c.d2[k - 1];
      if (fabs(diff) > 2.0 * tol * c.d2[k - 1]) {
        converged = false;
        const double denom = 2.0 * dot3(ref[k - 1], rk) * (c.w[0] + c.w[k]);
        if (denom == 0.0) continue;  // entartet (Bindung ⟂ Referenz); Nicht-Konvergenz meldet es
        const double g = diff / denom;
        c.x[k].x -= g * c.w[k] * ref[k - 1].x;
        c.x[k].y -= g * c.w[k] * ref[k - 1].y;
        c.x[k].z -= g * c.w[k] * ref[k - 1].z;
        c.x[0].x += g * c.w[0] * ref[k - 1].x;
        c.x[0].y += g * c.w[0] * ref[k - 1].y;
        c.x[0].z += g * c.w[0] * ref[k - 1].z;
      }
    }
  }
  if (!converged) atomicOr(s.d_status, kStatusConstraintFailure);
  if (drift) {
    const double inv_h = 1.0 / h;
    for (int k = 0; k <= c.n_h; ++k) {
      c.v[k].x += (c.x[k].x - x_un[k].x) * inv_h;
      c.v[k].y += (c.x[k].y - x_un[k].y) * inv_h;
      c.v[k].z += (c.x[k].z - x_un[k].z) * inv_h;
    }
  }
  store_cluster(s, c);
}

/// RATTLE: |r_k·(v_k − v_c)| < tol_v · r_k² für alle k.
__global__ void k_rattle(DeviceState s, const int4* clusters, const float4* distances,
                         const int2* range, double tol_v, int max_iter) {
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  const int r = blockIdx.y;
  Cluster c;
  if (!load_cluster(s, clusters, distances, range, r, t, c)) return;
  const float2 box = s.d_box[r];
  double3 rk[kMaxH];
  double r2[kMaxH];
  for (int k = 1; k <= c.n_h; ++k) {
    rk[k - 1] = delta(c.x[k], c.x[0], box);
    r2[k - 1] = dot3(rk[k - 1], rk[k - 1]);
  }
  bool converged = false;
  for (int it = 0; it < max_iter && !converged; ++it) {
    converged = true;
    for (int k = 1; k <= c.n_h; ++k) {
      const double3 dv =
          make_double3(c.v[k].x - c.v[0].x, c.v[k].y - c.v[0].y, c.v[k].z - c.v[0].z);
      const double rv = dot3(rk[k - 1], dv);
      if (fabs(rv) > tol_v * r2[k - 1]) {
        converged = false;
        const double kappa = rv / (r2[k - 1] * (c.w[0] + c.w[k]));
        c.v[k].x -= kappa * c.w[k] * rk[k - 1].x;
        c.v[k].y -= kappa * c.w[k] * rk[k - 1].y;
        c.v[k].z -= kappa * c.w[k] * rk[k - 1].z;
        c.v[0].x += kappa * c.w[0] * rk[k - 1].x;
        c.v[0].y += kappa * c.w[0] * rk[k - 1].y;
        c.v[0].z += kappa * c.w[0] * rk[k - 1].z;
      }
    }
  }
  if (!converged) atomicOr(s.d_status, kStatusConstraintFailure);
  for (int k = 0; k <= c.n_h; ++k)
    s.d_vel[c.atom[k]] = Vec4d{c.v[k].x, c.v[k].y, c.v[k].z, c.w[k]};  // Positionen unverändert
}

void require(bool condition, const std::string& message) {
  if (!condition) throw std::invalid_argument("Constraints: " + message);
}

}  // namespace

ConstraintSet::ConstraintSet(ConstraintParams params, const SimulationContext& ctx)
    : params_(std::move(params)) {
  const int n_rep = ctx.n_replicas(), n_atoms = ctx.n_atoms(), n_pad = ctx.n_pad();
  require(!params_.topologies.empty(), "mindestens eine Topologie nötig");
  require(params_.replica_topology.size() == static_cast<std::size_t>(n_rep),
          "replica_topology braucht n_replicas Einträge");
  require(params_.tolerance > 0.0 && params_.tolerance < 1e-2, "tolerance in (0, 1e-2)");
  require(params_.velocity_tolerance > 0.0, "velocity_tolerance > 0");
  require(params_.max_iterations > 0, "max_iterations > 0");

  std::vector<int4> clusters;
  std::vector<float4> distances;
  std::vector<int2> topo_range;
  std::vector<int> topo_count;
  std::vector<std::vector<unsigned char>> topo_mask;
  for (const auto& t : params_.topologies) {
    require(t.clusters.size() % 4 == 0 && t.distances.size() * 4 == t.clusters.size() * 3,
            "clusters (n, 4) und distances (n, 3) passen nicht zusammen");
    const int n_cl = static_cast<int>(t.clusters.size() / 4);
    topo_range.push_back(make_int2(static_cast<int>(clusters.size()), n_cl));
    std::vector<unsigned char> mask(n_pad, 0);
    int count = 0;
    for (int c = 0; c < n_cl; ++c) {
      int ids[4];
      float d[3];
      for (int k = 0; k < 4; ++k) ids[k] = t.clusters[4 * c + k];
      for (int k = 0; k < 3; ++k) d[k] = t.distances[3 * c + k];
      require(ids[0] >= 0 && ids[0] < n_atoms, "Schweratom-Index außerhalb [0, n_atoms)");
      require(ids[1] >= 0, "Cluster braucht mindestens ein H-Atom");
      bool ended = false;
      for (int k = 1; k < 4; ++k) {
        if (ids[k] < 0) {
          ended = true;
          continue;
        }
        require(!ended, "fehlende H-Atome nur am Ende eines Clusters (-1)");
        require(ids[k] < n_atoms, "H-Index außerhalb [0, n_atoms)");
        require(std::isfinite(d[k - 1]) && d[k - 1] > 0.0f, "Sollabstände endlich und > 0");
        ++count;
      }
      for (int k = 0; k < 4; ++k) {
        if (ids[k] < 0) continue;
        require(!mask[ids[k]], "Atom " + std::to_string(ids[k]) + " in mehreren Clustern");
        mask[ids[k]] = 1;
      }
      clusters.push_back(make_int4(ids[0], ids[1], ids[2], ids[3]));
      distances.push_back(make_float4(d[0], d[1], d[2], 0.0f));
    }
    topo_count.push_back(count);
    topo_mask.push_back(std::move(mask));
  }
  std::vector<int2> range(n_rep);
  std::vector<unsigned char> in_cluster(static_cast<std::size_t>(n_rep) * n_pad, 0);
  counts_.resize(n_rep);
  for (int r = 0; r < n_rep; ++r) {
    const int t = params_.replica_topology[r];
    require(t >= 0 && t < static_cast<int>(params_.topologies.size()), "Topologie-Index ungültig");
    range[r] = topo_range[t];
    counts_[r] = topo_count[t];
    max_clusters_ = std::max(max_clusters_, range[r].y);
    std::copy(topo_mask[t].begin(), topo_mask[t].end(),
              in_cluster.begin() + static_cast<std::ptrdiff_t>(r) * n_pad);
  }
  const cudaStream_t st = ctx.stream();
  d_clusters_ = DeviceBuffer<int4>(std::max<std::size_t>(clusters.size(), 1));
  d_distances_ = DeviceBuffer<float4>(std::max<std::size_t>(distances.size(), 1));
  d_range_ = DeviceBuffer<int2>(n_rep);
  d_in_cluster_ = DeviceBuffer<unsigned char>(in_cluster.size());
  if (!clusters.empty()) {
    BOMD_CUDA_CHECK(cudaMemcpyAsync(d_clusters_.data(), clusters.data(),
                                    clusters.size() * sizeof(int4), cudaMemcpyHostToDevice, st));
    BOMD_CUDA_CHECK(cudaMemcpyAsync(d_distances_.data(), distances.data(),
                                    distances.size() * sizeof(float4), cudaMemcpyHostToDevice, st));
  }
  d_range_.upload_async(range.data(), st);
  d_in_cluster_.upload_async(in_cluster.data(), st);
  BOMD_CUDA_CHECK(cudaStreamSynchronize(st));  // Host-Vektoren leben bis hier
}

void ConstraintSet::launch_shake(const DeviceState& s, double half_dt, bool drift,
                                 cudaStream_t stream) const {
  if (max_clusters_ == 0) return;
  const dim3 grid((max_clusters_ + kClusterBlock - 1) / kClusterBlock, s.n_replicas);
  k_shake<<<grid, kClusterBlock, 0, stream>>>(s, d_clusters_.data(), d_distances_.data(),
                                              d_range_.data(), half_dt, drift ? 1 : 0,
                                              params_.tolerance, params_.max_iterations);
  BOMD_LAUNCH_CHECK();
}

void ConstraintSet::drift_and_shake(const DeviceState& s, double half_dt,
                                    cudaStream_t stream) const {
  launch_shake(s, half_dt, true, stream);
}

void ConstraintSet::project_positions(const DeviceState& s, cudaStream_t stream) const {
  launch_shake(s, 0.0, false, stream);
}

void ConstraintSet::rattle(const DeviceState& s, cudaStream_t stream) const {
  if (max_clusters_ == 0) return;
  const dim3 grid((max_clusters_ + kClusterBlock - 1) / kClusterBlock, s.n_replicas);
  k_rattle<<<grid, kClusterBlock, 0, stream>>>(s, d_clusters_.data(), d_distances_.data(),
                                               d_range_.data(), params_.velocity_tolerance,
                                               params_.max_iterations);
  BOMD_LAUNCH_CHECK();
}

}  // namespace bomd
