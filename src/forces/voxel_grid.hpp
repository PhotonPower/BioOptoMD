// BioOptoMD – VoxelGridForce3D: externes Potential aus einem 3D-Gitter (V, ∇V) als 3D-Textur
// (docs/PHYSICS_MODEL.md §4.3). NumPy-Referenz: tests/reference/observers_ref.py (trilinear).
#pragma once

#include <cuda_runtime.h>

#include <array>
#include <string_view>
#include <vector>

#include "biooptomd/device_buffer.hpp"
#include "biooptomd/plugin.hpp"

namespace bomd {

struct VoxelGridParams {
  /// (nz, ny, nx, 4) row-major: values[k][j][i] = (V [kJ/mol], ∂V/∂x, ∂V/∂y, ∂V/∂z [kJ/mol/nm])
  /// am Punkt (x0 + i·Δx, y0 + j·Δy, z0 + k·Δz), jeweils pro Skalierungseinheit.
  std::vector<float> values;
  std::array<int, 3> shape{0, 0, 0};         ///< (nx, ny, nz), je >= 2
  std::array<float, 3> origin_nm{0, 0, 0};   ///< (x0, y0, z0)
  std::array<float, 3> spacing_nm{0, 0, 0};  ///< (Δx, Δy, Δz) > 0
  std::vector<float> scale_by_type;          ///< s_t je Optik-/Oberflächen-Typ (d_types.y)
  /// Gitter periodisch in x/y (nx·Δx = L_x, ny·Δy = L_y); sonst trägt ein Atom außerhalb nichts
  /// bei.
  bool periodic_xy = false;
};

/// E_i = s_t · V(r_i), F_i = −s_t · ∇V(r_i) mit trilinear interpoliertem V und ∇V (der
/// interpolierte Gradient ist nicht der Gradient des interpolierten Potentials, §4.3). Gelesen wird
/// per Point-Sampling aus einer 3D-Textur, interpoliert in FP32 im Kernel.
/// ASSUMPTION: Die Hardware-Interpolation (cudaFilterModeLinear) nutzt nur 9-Bit-Gewichte und wäre
/// für Kraftgenauigkeit ~1e-5 zu grob; daher manuelle Interpolation (8 Texturzugriffe).
/// Energie in kEnergyExternal. Atome außerhalb des Gitters (nicht-periodische Richtungen): 0.
class VoxelGridForce3D final : public ForcePlugin {
 public:
  explicit VoxelGridForce3D(VoxelGridParams params);
  ~VoxelGridForce3D() override;
  VoxelGridForce3D(const VoxelGridForce3D&) = delete;
  VoxelGridForce3D& operator=(const VoxelGridForce3D&) = delete;

  std::string_view name() const override { return "VoxelGridForce3D"; }
  void initialize(SimulationContext& ctx) override;
  void compute(DeviceState& s, cudaStream_t stream) override;

 private:
  VoxelGridParams params_;
  cudaArray_t array_ = nullptr;
  cudaTextureObject_t texture_ = 0;
  DeviceBuffer<float> d_scale_;
};

}  // namespace bomd
