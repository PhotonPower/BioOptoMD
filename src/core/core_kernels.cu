// BioOptoMD – elementweise Kern-Kernels: clear_forces, fixed_point_to_float, wrap_xy, Scatter.
#include "biooptomd/cuda_check.hpp"
#include "biooptomd/units.hpp"
#include "core/core_kernels.cuh"
#include "core/fixed_point.cuh"
#include "core/mixed.cuh"

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

/// Wrappt eine Koordinate x (FP64, ADR 0009) nach [0, L) und gibt die Anzahl der abgezogenen
/// Boxlängen n zurück, sodass x_alt = x_neu + n·L exakt gilt (UNITS_AND_CONVENTIONS.md, ADR 0003).
///
/// x hat ≈ 48 signifikante Bit, n·L höchstens 24 + 21 Bit (|x/L| < 2^20): x − n·L ist in FP64
/// exakt darstellbar, fma(−n, L, x) also exakt. Die beiden Korrekturen fangen die Rundung von
/// x/L ab, damit das Ergebnis in [0, L) liegt.
__device__ __forceinline__ int wrap_coordinate(double& x, double box_length) {
  double n = floor(x / box_length);
  double r = fma(-n, box_length, x);
  if (r < 0.0) {
    r += box_length;
    n -= 1.0;
  }
  if (r >= box_length) {
    r -= box_length;
    n += 1.0;
  }
  x = r;
  return __double2int_rz(n);  // sättigend statt UB bei |n| > INT_MAX
}

__global__ void k_wrap_xy(DeviceState s) {
  const int i_local = blockIdx.x * blockDim.x + threadIdx.x;
  const int r = blockIdx.y;
  if (i_local >= s.n_atoms) return;  // Padding-Atome bleiben unverändert.
  const int i = r * s.n_pad + i_local;
  const float2 box = s.d_box[r];
  double3 x = load_position(s, i);
  int2 img = s.d_image[i];
  // ADR 0008: außerhalb |x/L| < 2^20 ist der Wrap nicht mehr exakt.
  if (!(fabs(x.x) < 1048576.0 * box.x && fabs(x.y) < 1048576.0 * box.y))
    atomicOr(s.d_status, kStatusWrapRange);
  img.x += wrap_coordinate(x.x, box.x);
  img.y += wrap_coordinate(x.y, box.y);
  store_position(s, i, x);  // z und q unverändert
  s.d_image[i] = img;
}

template <typename V, typename T>
__global__ void k_scatter_xyz(DeviceState s, V* dst, const T* src) {
  const int i_local = blockIdx.x * blockDim.x + threadIdx.x;
  const int r = blockIdx.y;
  if (i_local >= s.n_atoms) return;
  const long long j = (static_cast<long long>(r) * s.n_atoms + i_local) * 3;
  V& d = dst[r * s.n_pad + i_local];
  d.x = src[j];
  d.y = src[j + 1];
  d.z = src[j + 2];
}

template <typename V, typename T>
__global__ void k_scatter_w(DeviceState s, V* dst, const T* src) {
  const int i_local = blockIdx.x * blockDim.x + threadIdx.x;
  const int r = blockIdx.y;
  if (i_local >= s.n_atoms) return;
  dst[r * s.n_pad + i_local].w = src[static_cast<long long>(r) * s.n_atoms + i_local];
}

/// Blockgröße der Energiereduktion (ein Block pro Replica).
constexpr int kEnergyBlock = 256;

__global__ void k_energy_frame(DeviceState s, double* frame) {
  __shared__ double partial[kEnergyBlock];
  const int r = blockIdx.x;
  double sum = 0.0;
  for (int i_local = threadIdx.x; i_local < s.n_atoms; i_local += blockDim.x) {
    const Vec4d v = s.d_vel[r * s.n_pad + i_local];
    if (v.w > 0.0) sum += (v.x * v.x + v.y * v.y + v.z * v.z) / v.w;
  }
  partial[threadIdx.x] = sum;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0;
       stride >>= 1) {  // feste Reihenfolge ⇒ deterministisch
    if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
    __syncthreads();
  }
  double* out = frame + static_cast<long long>(r) * kEnergyFrameColumns;
  if (threadIdx.x == 0) out[0] = 0.5 * partial[0];
  if (threadIdx.x < kNumEnergyTerms)
    out[1 + threadIdx.x] = static_cast<double>(s.d_energy_fp[r * kNumEnergyTerms + threadIdx.x]) *
                           (1.0 / units::FIXED_POINT_SCALE);
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

void launch_energy_frame(const DeviceState& s, double* frame, cudaStream_t stream) {
  if (s.n_total() == 0) return;
  k_energy_frame<<<s.n_replicas, kEnergyBlock, 0, stream>>>(s, frame);
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

void launch_scatter_xyz(const DeviceState& s, Vec4d* dst, const double* src, cudaStream_t stream) {
  if (s.n_total() == 0) return;
  k_scatter_xyz<<<per_replica_grid(s), kElementwiseBlock, 0, stream>>>(s, dst, src);
  BOMD_LAUNCH_CHECK();
}

void launch_scatter_w(const DeviceState& s, Vec4d* dst, const double* src, cudaStream_t stream) {
  if (s.n_total() == 0) return;
  k_scatter_w<<<per_replica_grid(s), kElementwiseBlock, 0, stream>>>(s, dst, src);
  BOMD_LAUNCH_CHECK();
}

}  // namespace bomd::detail
