// BioOptoMD – BondedForce, docs/PHYSICS_MODEL.md §2. Formeln (Energie, Kraft, Vorzeichen) und
// Grenzfälle sind in tests/reference/bonded_ref.py dokumentiert; die Kernels bilden sie in FP32
// nach.
#include <algorithm>
#include <cmath>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <utility>

#include "biooptomd/cuda_check.hpp"
#include "biooptomd/simulation_context.hpp"
#include "biooptomd/units.hpp"
#include "core/core_kernels.cuh"
#include "core/fixed_point.cuh"
#include "core/pbc.cuh"
#include "forces/bonded.hpp"

namespace bomd {
namespace {

using detail::elementwise_grid;
using detail::kElementwiseBlock;

// --- Vektorhilfen ----------------------------------------------------------------------------

__device__ __forceinline__ float3 operator+(float3 a, float3 b) {
  return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
__device__ __forceinline__ float3 operator-(float3 a) {
  return make_float3(-a.x, -a.y, -a.z);
}
__device__ __forceinline__ float3 operator*(float c, float3 a) {
  return make_float3(c * a.x, c * a.y, c * a.z);
}
__device__ __forceinline__ float dot(float3 a, float3 b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}
__device__ __forceinline__ float3 cross(float3 a, float3 b) {
  return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

/// Differenzvektor p_a − p_b mit Minimum Image in x/y (ADR 0003, src/core/pbc.cuh).
__device__ __forceinline__ float3 delta(const DeviceState& s, int a, int b, float2 box) {
  return detail::minimum_image_delta(s.d_pos[a], s.d_pos[b], box);
}

__device__ __forceinline__ void add_force3(const DeviceState& s, int i, float3 f) {
  detail::add_force(s, i, f.x, f.y, f.z);
}

// --- Kernels (ein Thread pro Term, blockIdx.y = Replica; kein frühes return vor warp_add_energy)

/// Bond: V = ½k(r − r0)², F_i = −k(r − r0) d/r, F_j = −F_i. r² < EPS_R2 ⇒ F = 0.
__global__ void k_bonds(DeviceState s, const int4* index, const float4* param, const int2* range) {
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  const int r = blockIdx.y;
  const int2 rg = range[r];
  float energy = 0.0f;
  if (t < rg.y) {
    const int4 id = index[rg.x + t];
    const float4 p = param[rg.x + t];  // (k, r0)
    const int base = r * s.n_pad;
    const float3 d = delta(s, base + id.x, base + id.y, s.d_box[r]);
    const float r2 = dot(d, d);
    const float dist = sqrtf(r2);
    const float dr = dist - p.y;
    energy = 0.5f * p.x * dr * dr;
    if (r2 >= units::EPS_R2) {
      const float3 fi = (-p.x * dr / dist) * d;
      add_force3(s, base + id.x, fi);
      add_force3(s, base + id.y, -fi);
    }
  }
  detail::warp_add_energy(s, r, kEnergyBond, energy);
}

/// Angle: θ = atan2(|a×b|, a·b) (AGENTS.md Regel 3), V = ½k(θ − θ0)².
/// F_i = −V'(a×n)/(|a|²|n|), F_k = −V'(n×b)/(|b|²|n|), F_j = −F_i − F_k; |n|² < EPS_CROSS2 ⇒ F = 0.
__global__ void k_angles(DeviceState s, const int4* index, const float4* param, const int2* range) {
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  const int r = blockIdx.y;
  const int2 rg = range[r];
  float energy = 0.0f;
  if (t < rg.y) {
    const int4 id = index[rg.x + t];
    const float4 p = param[rg.x + t];  // (k, θ0)
    const int base = r * s.n_pad;
    const float2 box = s.d_box[r];
    const float3 a = delta(s, base + id.x, base + id.y, box);
    const float3 b = delta(s, base + id.z, base + id.y, box);
    const float3 n = cross(a, b);
    const float n2 = dot(n, n);
    const float theta = atan2f(sqrtf(n2), dot(a, b));
    const float dth = theta - p.y;
    energy = 0.5f * p.x * dth * dth;
    if (n2 >= units::EPS_CROSS2) {
      const float dv = p.x * dth;
      const float nn = sqrtf(n2);
      const float3 fi = (-dv / (dot(a, a) * nn)) * cross(a, n);
      const float3 fk = (-dv / (dot(b, b) * nn)) * cross(n, b);
      add_force3(s, base + id.x, fi);
      add_force3(s, base + id.z, fk);
      add_force3(s, base + id.y, -(fi + fk));
    }
  }
  detail::warp_add_energy(s, r, kEnergyAngle, energy);
}

/// Periodische Torsion (IUPAC): φ = atan2(|b2| b1·n, m·n), V = k[1 + cos(nφ − φ0)].
/// Kräfte nach Blondel & Karplus (1996), siehe bonded_ref.py; degeneriert ⇒ F = 0.
__global__ void k_torsions(DeviceState s, const int4* index, const float4* param,
                           const int2* range) {
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  const int r = blockIdx.y;
  const int2 rg = range[r];
  float energy = 0.0f;
  if (t < rg.y) {
    const int4 id = index[rg.x + t];
    const float4 p = param[rg.x + t];  // (k, n, φ0, 0)
    const int base = r * s.n_pad;
    const float2 box = s.d_box[r];
    const float3 b1 = delta(s, base + id.y, base + id.x, box);
    const float3 b2 = delta(s, base + id.z, base + id.y, box);
    const float3 b3 = delta(s, base + id.w, base + id.z, box);
    const float3 m = cross(b1, b2);
    const float3 n = cross(b2, b3);
    const float b22 = dot(b2, b2);
    const float b2n = sqrtf(b22);
    const float phi = atan2f(b2n * dot(b1, n), dot(m, n));
    const float arg = p.y * phi - p.z;
    energy = p.x * (1.0f + cosf(arg));
    const float m2 = dot(m, m), n2 = dot(n, n);
    if (m2 >= units::EPS_CROSS2 && n2 >= units::EPS_CROSS2 && b22 >= units::EPS_R2) {
      const float dv = -p.x * p.y * sinf(arg);
      const float3 fi = (dv * b2n / m2) * m;
      const float3 fl = (-dv * b2n / n2) * n;
      const float cm = dot(b1, b2) / b22;
      const float cn = dot(b3, b2) / b22;
      add_force3(s, base + id.x, fi);
      add_force3(s, base + id.w, fl);
      add_force3(s, base + id.y, (-(1.0f + cm)) * fi + cn * fl);
      add_force3(s, base + id.z, (-(1.0f + cn)) * fl + cm * fi);
    }
  }
  detail::warp_add_energy(s, r, kEnergyTorsion, energy);
}

/// 1-4-Paar: V = 4ε[(σ/r)^12 − (σ/r)^6] + C qq exp(−κr)/(ε_r r) (PHYSICS_MODEL.md §2, §3).
/// r² < EPS_R2 ⇒ V = 0, F = 0.
__global__ void k_pairs14(DeviceState s, const int4* index, const float4* param, const int2* range,
                          const float2* screening) {
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  const int r = blockIdx.y;
  const int2 rg = range[r];
  float energy = 0.0f;
  if (t < rg.y) {
    const int4 id = index[rg.x + t];
    const float4 p = param[rg.x + t];  // (qq, σ, ε, 0)
    const int base = r * s.n_pad;
    const float3 d = delta(s, base + id.x, base + id.y, s.d_box[r]);
    const float r2 = dot(d, d);
    if (r2 >= units::EPS_R2) {
      const float2 sc = screening[r];  // (κ, ε_r)
      const float dist = sqrtf(r2);
      const float sr2 = p.y * p.y / r2;
      const float sr6 = sr2 * sr2 * sr2;
      const float ex = expf(-sc.x * dist);
      const float c = units::ONE_4PI_EPS0_F * p.x * ex / (sc.y * dist);
      energy = 4.0f * p.z * (sr6 * sr6 - sr6) + c;
      // −dV/dr / r: LJ 4ε(12 sr6² − 6 sr6)/r², DH c (κr + 1)/r²
      const float f_over_r =
          (4.0f * p.z * (12.0f * sr6 * sr6 - 6.0f * sr6) + c * (sc.x * dist + 1.0f)) / r2;
      const float3 fi = f_over_r * d;
      add_force3(s, base + id.x, fi);
      add_force3(s, base + id.y, -fi);
    }
  }
  detail::warp_add_energy(s, r, kEnergyPair14, energy);
}

// --- Host-Hilfen ---------------------------------------------------------------------------

void require(bool condition, const std::string& message) {
  if (!condition) throw std::invalid_argument("BondedForce: " + message);
}

void check_sizes(std::size_t n_terms, std::size_t n_index, int arity,
                 std::initializer_list<std::size_t> params, const char* name) {
  require(n_index == n_terms * arity, std::string(name) + ": Indexarray hat falsche Länge");
  for (std::size_t p : params)
    require(p == n_terms, std::string(name) + ": Parameterarrays passen nicht zu den Indizes");
}

}  // namespace

BondedForce::BondedForce(BondedParams params) : params_(std::move(params)) {
  require(!params_.topologies.empty(), "mindestens eine Topologie nötig");
  for (const auto& t : params_.topologies) {
    check_sizes(t.bond_k.size(), t.bonds.size(), 2, {t.bond_r0.size()}, "bonds");
    check_sizes(t.angle_k.size(), t.angles.size(), 3, {t.angle_theta0.size()}, "angles");
    check_sizes(t.torsion_k.size(), t.torsions.size(), 4,
                {t.torsion_n.size(), t.torsion_phase.size()}, "torsions");
    check_sizes(t.pair_qq.size(), t.pairs14.size(), 2, {t.pair_sigma.size(), t.pair_eps.size()},
                "pairs14");
    for (int n : t.torsion_n) require(n >= 1, "torsion_n muss >= 1 sein");
  }
}

void BondedForce::initialize(SimulationContext& ctx) {
  const int n_rep = ctx.n_replicas();
  const int n_atoms = ctx.n_atoms();
  require(params_.replica_topology.size() == static_cast<std::size_t>(n_rep) &&
              params_.kappa_per_nm.size() == static_cast<std::size_t>(n_rep) &&
              params_.eps_r.size() == static_cast<std::size_t>(n_rep),
          "replica_topology, kappa, eps_r brauchen n_replicas Einträge");
  for (int r = 0; r < n_rep; ++r) {
    const int t = params_.replica_topology[r];
    require(t >= 0 && t < static_cast<int>(params_.topologies.size()), "Topologie-Index ungültig");
    require(std::isfinite(params_.kappa_per_nm[r]) && params_.kappa_per_nm[r] >= 0.0f,
            "kappa endlich und >= 0");
    require(std::isfinite(params_.eps_r[r]) && params_.eps_r[r] > 0.0f, "eps_r endlich und > 0");
  }

  // Konkatenieren: Topologie t belegt [begin_t, begin_t + count_t) je Term-Typ.
  auto build = [&](TermBuffers& tb, int arity, auto&& index_of, auto&& param_of, auto&& count_of,
                   const char* name) {
    std::vector<int4> index;
    std::vector<float4> param;
    std::vector<int2> topo_range(params_.topologies.size());
    for (std::size_t ti = 0; ti < params_.topologies.size(); ++ti) {
      const auto& topo = params_.topologies[ti];
      const int count = static_cast<int>(count_of(topo));
      topo_range[ti] = make_int2(static_cast<int>(index.size()), count);
      for (int k = 0; k < count; ++k) {
        int ids[4] = {0, 0, 0, 0};
        for (int a = 0; a < arity; ++a) {
          ids[a] = index_of(topo)[static_cast<std::size_t>(k) * arity + a];
          require(ids[a] >= 0 && ids[a] < n_atoms,
                  std::string(name) + ": Atomindex außerhalb [0, n_atoms)");
          for (int b = 0; b < a; ++b)
            require(ids[a] != ids[b], std::string(name) + ": Atom doppelt in einem Term");
        }
        index.push_back(make_int4(ids[0], ids[1], ids[2], ids[3]));
        param.push_back(param_of(topo, k));
      }
    }
    tb.h_range.resize(n_rep);
    tb.max_count = 0;
    for (int r = 0; r < n_rep; ++r) {
      tb.h_range[r] = topo_range[params_.replica_topology[r]];
      tb.max_count = std::max(tb.max_count, tb.h_range[r].y);
    }
    tb.index = DeviceBuffer<int4>(index.size());
    tb.param = DeviceBuffer<float4>(param.size());
    tb.range = DeviceBuffer<int2>(n_rep);
    tb.index.upload_async(index.data(), ctx.stream());
    tb.param.upload_async(param.data(), ctx.stream());
    tb.range.upload_async(tb.h_range.data(), ctx.stream());
    ctx.synchronize();  // Host-Vektoren leben bis hier
  };

  build(
      bonds_, 2, [](const BondedTopology& t) -> const std::vector<int>& { return t.bonds; },
      [](const BondedTopology& t, int k) {
        return make_float4(t.bond_k[k], t.bond_r0[k], 0.f, 0.f);
      },
      [](const BondedTopology& t) { return t.bond_k.size(); }, "bonds");
  build(
      angles_, 3, [](const BondedTopology& t) -> const std::vector<int>& { return t.angles; },
      [](const BondedTopology& t, int k) {
        return make_float4(t.angle_k[k], t.angle_theta0[k], 0.f, 0.f);
      },
      [](const BondedTopology& t) { return t.angle_k.size(); }, "angles");
  build(
      torsions_, 4, [](const BondedTopology& t) -> const std::vector<int>& { return t.torsions; },
      [](const BondedTopology& t, int k) {
        return make_float4(t.torsion_k[k], static_cast<float>(t.torsion_n[k]), t.torsion_phase[k],
                           0.f);
      },
      [](const BondedTopology& t) { return t.torsion_k.size(); }, "torsions");
  build(
      pairs_, 2, [](const BondedTopology& t) -> const std::vector<int>& { return t.pairs14; },
      [](const BondedTopology& t, int k) {
        return make_float4(t.pair_qq[k], t.pair_sigma[k], t.pair_eps[k], 0.f);
      },
      [](const BondedTopology& t) { return t.pair_qq.size(); }, "pairs14");

  std::vector<float2> screening(n_rep);
  for (int r = 0; r < n_rep; ++r)
    screening[r] = make_float2(params_.kappa_per_nm[r], params_.eps_r[r]);
  d_screening_ = DeviceBuffer<float2>(n_rep);
  d_screening_.upload_async(screening.data(), ctx.stream());
  ctx.synchronize();
}

void BondedForce::compute(DeviceState& s, cudaStream_t stream) {
  auto grid = [&](const TermBuffers& tb) {
    return dim3(elementwise_grid(tb.max_count), s.n_replicas);
  };
  if (bonds_.max_count > 0) {
    k_bonds<<<grid(bonds_), kElementwiseBlock, 0, stream>>>(
        s, bonds_.index.data(), bonds_.param.data(), bonds_.range.data());
    BOMD_LAUNCH_CHECK();
  }
  if (angles_.max_count > 0) {
    k_angles<<<grid(angles_), kElementwiseBlock, 0, stream>>>(
        s, angles_.index.data(), angles_.param.data(), angles_.range.data());
    BOMD_LAUNCH_CHECK();
  }
  if (torsions_.max_count > 0) {
    k_torsions<<<grid(torsions_), kElementwiseBlock, 0, stream>>>(
        s, torsions_.index.data(), torsions_.param.data(), torsions_.range.data());
    BOMD_LAUNCH_CHECK();
  }
  if (pairs_.max_count > 0) {
    k_pairs14<<<grid(pairs_), kElementwiseBlock, 0, stream>>>(
        s, pairs_.index.data(), pairs_.param.data(), pairs_.range.data(), d_screening_.data());
    BOMD_LAUNCH_CHECK();
  }
}

std::vector<int> BondedForce::term_counts(int replica) const {
  return {bonds_.h_range.at(replica).y, angles_.h_range.at(replica).y,
          torsions_.h_range.at(replica).y, pairs_.h_range.at(replica).y};
}

}  // namespace bomd
