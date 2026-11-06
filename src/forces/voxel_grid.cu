// BioOptoMD – VoxelGridForce3D, siehe voxel_grid.hpp.
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>

#include "biooptomd/cuda_check.hpp"
#include "biooptomd/simulation_context.hpp"
#include "core/core_kernels.cuh"
#include "core/fixed_point.cuh"
#include "core/mixed.cuh"
#include "forces/voxel_grid.hpp"

namespace bomd {
namespace {

struct GridGeometry {
  int nx, ny, nz;
  float x0, y0, z0;
  float inv_dx, inv_dy, inv_dz;
  int periodic_xy;
};

/// Gitterindex und Bruchteil einer Koordinate; false, wenn außerhalb (nicht periodisch).
__device__ __forceinline__ bool locate(float x, float x0, float inv_d, int n, bool periodic,
                                       int& i0, int& i1, float& f) {
  const float u = (x - x0) * inv_d;
  const float fl = floorf(u);
  f = u - fl;
  int i = static_cast<int>(fl);
  if (periodic) {
    i %= n;
    if (i < 0) i += n;
    i0 = i;
    i1 = (i + 1 == n) ? 0 : i + 1;
    return true;
  }
  if (i < 0 || i + 1 > n - 1) return false;
  i0 = i;
  i1 = i + 1;
  return true;
}

/// Lineare Interpolation a + f·(b − a): exakt für a = b (konstante Felder).
__device__ __forceinline__ float4 lerp4(float4 a, float4 b, float f) {
  return make_float4(a.x + f * (b.x - a.x), a.y + f * (b.y - a.y), a.z + f * (b.z - a.z),
                     a.w + f * (b.w - a.w));
}

/// Ein Thread pro Atomplatz, blockIdx.y = Replica; kein frühes return vor warp_add_energy.
__global__ void k_voxel(DeviceState s, cudaTextureObject_t tex, GridGeometry g, const float* scale,
                        int n_types) {
  const int i_local = blockIdx.x * blockDim.x + threadIdx.x;
  const int r = blockIdx.y;
  float energy = 0.0f;
  if (i_local < s.n_atoms) {
    const int i = r * s.n_pad + i_local;
    const int t = s.d_types[i].y;
    const float st = t < n_types ? scale[t] : 0.0f;
    if (t >= n_types) atomicOr(s.d_status, kStatusUnknownType);
    if (st != 0.0f) {
      const double3 x = detail::load_position(s, i);
      int x0, x1, y0, y1, z0, z1;
      float fx, fy, fz;
      if (locate(static_cast<float>(x.x), g.x0, g.inv_dx, g.nx, g.periodic_xy, x0, x1, fx) &&
          locate(static_cast<float>(x.y), g.y0, g.inv_dy, g.ny, g.periodic_xy, y0, y1, fy) &&
          locate(static_cast<float>(x.z), g.z0, g.inv_dz, g.nz, false, z0, z1, fz)) {
        // Point-Sampling an Texelmitten (i + 0.5), Interpolation in FP32.
        auto at = [&](int a, int b, int c) {
          return tex3D<float4>(tex, a + 0.5f, b + 0.5f, c + 0.5f);
        };
        const float4 c00 = lerp4(at(x0, y0, z0), at(x1, y0, z0), fx);
        const float4 c10 = lerp4(at(x0, y1, z0), at(x1, y1, z0), fx);
        const float4 c01 = lerp4(at(x0, y0, z1), at(x1, y0, z1), fx);
        const float4 c11 = lerp4(at(x0, y1, z1), at(x1, y1, z1), fx);
        const float4 v = lerp4(lerp4(c00, c10, fy), lerp4(c01, c11, fy), fz);
        energy = st * v.x;
        detail::add_force(s, i, -st * v.y, -st * v.z, -st * v.w);
      }
    }
  }
  detail::warp_add_energy(s, r, kEnergyExternal, energy);
}

void require(bool condition, const std::string& message) {
  if (!condition) throw std::invalid_argument("VoxelGridForce3D: " + message);
}

}  // namespace

VoxelGridForce3D::VoxelGridForce3D(VoxelGridParams params) : params_(std::move(params)) {
  const auto& p = params_;
  for (int d = 0; d < 3; ++d) {
    require(p.shape[d] >= 2, "jede Gitterdimension braucht >= 2 Punkte");
    require(std::isfinite(p.spacing_nm[d]) && p.spacing_nm[d] > 0.0f, "spacing > 0");
    require(std::isfinite(p.origin_nm[d]), "origin endlich");
  }
  const std::size_t n = static_cast<std::size_t>(p.shape[0]) * p.shape[1] * p.shape[2];
  require(p.values.size() == 4 * n, "values muss (nz, ny, nx, 4) sein");
  for (float v : p.values) require(std::isfinite(v), "values endlich");
  require(!p.scale_by_type.empty(), "scale_by_type braucht mindestens einen Typ");
  for (float v : p.scale_by_type) require(std::isfinite(v), "scale_by_type endlich");
}

VoxelGridForce3D::~VoxelGridForce3D() {
  if (texture_) cudaDestroyTextureObject(texture_);
  if (array_) cudaFreeArray(array_);
}

void VoxelGridForce3D::check_box() const {
  const auto& p = params_;
  if (!p.periodic_xy) return;
  require(!ctx_->host_box().empty(), "periodic_xy verlangt eine gesetzte Box");
  for (const float2& b : ctx_->host_box())
    require(std::fabs(p.shape[0] * p.spacing_nm[0] - b.x) <= 1e-5f * b.x &&
                std::fabs(p.shape[1] * p.spacing_nm[1] - b.y) <= 1e-5f * b.y,
            "periodic_xy verlangt nx·Δx = L_x und ny·Δy = L_y für jede Replica");
}

void VoxelGridForce3D::initialize(SimulationContext& ctx) {
  ctx_ = &ctx;
  box_version_ = ctx.box_version();
  const auto& p = params_;
  if (p.periodic_xy) {
    for (const float2& b : ctx.host_box())
      require(std::fabs(p.shape[0] * p.spacing_nm[0] - b.x) <= 1e-5f * b.x &&
                  std::fabs(p.shape[1] * p.spacing_nm[1] - b.y) <= 1e-5f * b.y,
              "periodic_xy verlangt nx·Δx = L_x und ny·Δy = L_y für jede Replica");
    require(!ctx.host_box().empty(), "periodic_xy verlangt eine gesetzte Box");
  }
  const cudaChannelFormatDesc desc = cudaCreateChannelDesc<float4>();
  const cudaExtent extent = make_cudaExtent(p.shape[0], p.shape[1], p.shape[2]);
  BOMD_CUDA_CHECK(cudaMalloc3DArray(&array_, &desc, extent));
  upload_values(p.values);

  cudaResourceDesc res = {};
  res.resType = cudaResourceTypeArray;
  res.res.array.array = array_;
  cudaTextureDesc td = {};
  td.addressMode[0] = td.addressMode[1] = td.addressMode[2] = cudaAddressModeClamp;
  td.filterMode = cudaFilterModePoint;
  td.readMode = cudaReadModeElementType;
  td.normalizedCoords = 0;
  BOMD_CUDA_CHECK(cudaCreateTextureObject(&texture_, &res, &td, nullptr));

  d_scale_ = DeviceBuffer<float>(p.scale_by_type.size());
  d_scale_.upload_async(p.scale_by_type.data(), ctx.stream());
  ctx.synchronize();
}

void VoxelGridForce3D::upload_values(const std::vector<float>& values) {
  const auto& p = params_;
  cudaMemcpy3DParms copy = {};
  copy.srcPtr = make_cudaPitchedPtr(const_cast<float*>(values.data()), p.shape[0] * sizeof(float4),
                                    p.shape[0], p.shape[1]);
  copy.dstArray = array_;
  copy.extent = make_cudaExtent(p.shape[0], p.shape[1], p.shape[2]);
  copy.kind = cudaMemcpyHostToDevice;
  // Stream-geordnet: Kernel davor lesen noch die alten Werte. Die Host-Quelle bleibt bis zur
  // Synchronisation gültig.
  BOMD_CUDA_CHECK(cudaMemcpy3DAsync(&copy, ctx_->stream()));
  ctx_->synchronize();
}

void VoxelGridForce3D::set_values(std::vector<float> values) {
  require(ctx_ != nullptr, "set_values vor initialize()");
  require(values.size() == params_.values.size(),
          "set_values: Shape muss (nz, ny, nx, 4) des bestehenden Gitters sein");
  for (float v : values) require(std::isfinite(v), "values endlich");
  ctx_->activate();
  ctx_->synchronize();    // laufende Kernel lesen die Textur noch
  upload_values(values);  // erst nach erfolgreichem Upload die Host-Kopie ersetzen
  params_.values = std::move(values);
  ctx_->invalidate_forces();
}

void VoxelGridForce3D::compute(DeviceState& s, cudaStream_t stream) {
  if (box_version_ != ctx_->box_version()) {  // Box geändert: Periodizität erneut prüfen
    check_box();
    box_version_ = ctx_->box_version();
  }
  const auto& p = params_;
  const GridGeometry g{p.shape[0],
                       p.shape[1],
                       p.shape[2],
                       p.origin_nm[0],
                       p.origin_nm[1],
                       p.origin_nm[2],
                       1.0f / p.spacing_nm[0],
                       1.0f / p.spacing_nm[1],
                       1.0f / p.spacing_nm[2],
                       p.periodic_xy ? 1 : 0};
  const dim3 grid(detail::elementwise_grid(s.n_atoms), s.n_replicas);
  k_voxel<<<grid, detail::kElementwiseBlock, 0, stream>>>(s, texture_, g, d_scale_.data(),
                                                          static_cast<int>(p.scale_by_type.size()));
  BOMD_LAUNCH_CHECK();
}

}  // namespace bomd
