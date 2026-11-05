// BioOptoMD – elementweise Kern-Kernels: clear_forces, fixed_point_to_float, wrap_xy, Scatter.
#include "biooptomd/cuda_check.hpp"
#include "core/core_kernels.cuh"
#include "core/fixed_point.cuh"

namespace bomd::detail {
namespace {

__global__ void k_clear_forces(DeviceState s) {
  const long long n_force = 3LL * s.n_total();
  const long long n_energy = static_cast<long long>(s.n_replicas) * kNumEnergyTerms;
  const long long stride = static_cast<long long>(gridDim.x) * blockDim.x;
  for (long long i = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x; i < n_force;
       i += stride) {
    s.d_force_fp[i] = 0;
    if (i < n_energy) s.d_energy_fp[i] = 0;
  }
}

__global__ void k_fixed_point_to_float(DeviceState s) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= s.n_total()) return;
  s.d_force[i] =
      make_float4(from_fixed_point(s.d_force_fp_x()[i]), from_fixed_point(s.d_force_fp_y()[i]),
                  from_fixed_point(s.d_force_fp_z()[i]), 0.0f);
}

/// Wrappt eine Koordinate x nach [0, L) und gibt die Anzahl der abgezogenen Boxlängen zurück,
/// sodass x_alt ≈ x_neu + n·L (UNITS_AND_CONVENTIONS.md, ADR 0003).
///
/// r = fmaf(-n, L, x) rundet nur einmal; ist |x| >= L (gleicher oder größerer Exponent), ist
/// x − n·L in float exakt darstellbar und damit die Rekonstruktion exakt. Für x ∈ (−L, 0) wird
/// x + L einmal gerundet (Fehler ≤ ½ ulp(L)). Die beiden Korrekturen fangen die Rundung von
/// x/L und von x + L ab, damit r garantiert in [0, L) liegt – gültig für |x/L| < 2^20
/// (SimulationContext::kMaxBoxImages); weit darüber ist floorf(x/L) um mehr als 1 falsch.
__device__ __forceinline__ int wrap_coordinate(float& x, float box_length) {
  float n = floorf(x / box_length);
  float r = fmaf(-n, box_length, x);
  if (r < 0.0f) {
    r += box_length;
    n -= 1.0f;
  }
  if (r >= box_length) {
    r -= box_length;
    n += 1.0f;
  }
  x = r;
  return __float2int_rz(n);  // sättigend statt UB bei |n| > INT_MAX
}

__global__ void k_wrap_xy(DeviceState s) {
  const int i_local = blockIdx.x * blockDim.x + threadIdx.x;
  const int r = blockIdx.y;
  if (i_local >= s.n_atoms) return;  // Padding-Atome bleiben unverändert.
  const int i = r * s.n_pad + i_local;
  const float2 box = s.d_box[r];
  float4 p = s.d_pos[i];
  int2 img = s.d_image[i];
  img.x += wrap_coordinate(p.x, box.x);
  img.y += wrap_coordinate(p.y, box.y);
  s.d_pos[i] = p;  // z und q unverändert.
  s.d_image[i] = img;
}

__global__ void k_scatter_xyz(DeviceState s, float4* dst, const float* src) {
  const int i_local = blockIdx.x * blockDim.x + threadIdx.x;
  const int r = blockIdx.y;
  if (i_local >= s.n_atoms) return;
  const long long j = (static_cast<long long>(r) * s.n_atoms + i_local) * 3;
  float4& d = dst[r * s.n_pad + i_local];
  d.x = src[j];
  d.y = src[j + 1];
  d.z = src[j + 2];
}

__global__ void k_scatter_w(DeviceState s, float4* dst, const float* src) {
  const int i_local = blockIdx.x * blockDim.x + threadIdx.x;
  const int r = blockIdx.y;
  if (i_local >= s.n_atoms) return;
  dst[r * s.n_pad + i_local].w = src[static_cast<long long>(r) * s.n_atoms + i_local];
}

/// Grid für Kernels mit blockIdx.y = Replica: so muss kein Thread r = i / n_atoms berechnen,
/// und Padding-Atome werden per Index ausgeschlossen.
dim3 per_replica_grid(const DeviceState& s) {
  return dim3(elementwise_grid(s.n_atoms), s.n_replicas);
}

}  // namespace

void launch_clear_forces(const DeviceState& s, cudaStream_t stream) {
  if (s.n_total() == 0) return;
  k_clear_forces<<<elementwise_grid(3LL * s.n_total()), kElementwiseBlock, 0, stream>>>(s);
  BOMD_LAUNCH_CHECK();
}

void launch_fixed_point_to_float(const DeviceState& s, cudaStream_t stream) {
  if (s.n_total() == 0) return;
  k_fixed_point_to_float<<<elementwise_grid(s.n_total()), kElementwiseBlock, 0, stream>>>(s);
  BOMD_LAUNCH_CHECK();
}

void launch_wrap_xy(const DeviceState& s, cudaStream_t stream) {
  if (s.n_total() == 0) return;
  k_wrap_xy<<<per_replica_grid(s), kElementwiseBlock, 0, stream>>>(s);
  BOMD_LAUNCH_CHECK();
}

void launch_scatter_xyz(const DeviceState& s, float4* dst, const float* src, cudaStream_t stream) {
  if (s.n_total() == 0) return;
  k_scatter_xyz<<<per_replica_grid(s), kElementwiseBlock, 0, stream>>>(s, dst, src);
  BOMD_LAUNCH_CHECK();
}

void launch_scatter_w(const DeviceState& s, float4* dst, const float* src, cudaStream_t stream) {
  if (s.n_total() == 0) return;
  k_scatter_w<<<per_replica_grid(s), kElementwiseBlock, 0, stream>>>(s, dst, src);
  BOMD_LAUNCH_CHECK();
}

}  // namespace bomd::detail
