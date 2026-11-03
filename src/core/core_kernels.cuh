// BioOptoMD – elementweise Kern-Kernels (Phase 1). Launch-Wrapper in core_kernels.cu.
#pragma once

#include <cuda_runtime.h>

#include "biooptomd/device_state.hpp"

namespace bomd::detail {

/// Blockgröße der elementweisen Kernels. Grid linearisiert über alle n_replicas·n_pad
/// Atomplätze: keine Replica-spezifische Arbeit, daher kein blockIdx.y = replica nötig.
inline constexpr int kElementwiseBlock = 256;

inline int elementwise_grid(long long n) {
  return static_cast<int>((n + kElementwiseBlock - 1) / kElementwiseBlock);
}

/// d_force_fp (3·n_total) und d_energy_fp (n_replicas·kNumEnergyTerms) auf 0.
void launch_clear_forces(const DeviceState& s, cudaStream_t stream);

/// d_force[i] = (from_fp(fx[i]), from_fp(fy[i]), from_fp(fz[i]), 0) für alle Atomplätze.
void launch_fixed_point_to_float(const DeviceState& s, cudaStream_t stream);

/// x/y echter Atome nach [0, L) wrappen; d_image fortschreiben. Box muss > 0 sein.
void launch_wrap_xy(const DeviceState& s, cudaStream_t stream);

/// dst[r·n_pad + i].xyz = src[(r·n_atoms + i)·3 + 0..2] für i < n_atoms; w unverändert.
void launch_scatter_xyz(const DeviceState& s, float4* dst, const float* src, cudaStream_t stream);

/// dst[r·n_pad + i].w = src[r·n_atoms + i] für i < n_atoms; xyz unverändert.
void launch_scatter_w(const DeviceState& s, float4* dst, const float* src, cudaStream_t stream);

/// Spalten eines Energie-Frames: [E_kin, E_pot(Term 0), …, E_pot(Term N_TERMS−1)] in kJ/mol.
inline constexpr int kEnergyFrameColumns = 1 + kNumEnergyTerms;

/// Schreibt pro Replica r einen Frame nach frame[r·kEnergyFrameColumns + c] (double):
/// E_kin = ½ Σ m v² über echte Atome mit inv_mass > 0 (FP64-Reduktion) und die Terme aus
/// d_energy_fp (Fixed-Point → double, exakt bis 2^53).
void launch_energy_frame(const DeviceState& s, double* frame, cudaStream_t stream);

}  // namespace bomd::detail
