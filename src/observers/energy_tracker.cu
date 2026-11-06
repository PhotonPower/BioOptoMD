// BioOptoMD – EnergyTracker, siehe energy_tracker.hpp.
#include <stdexcept>
#include <vector>

#include "biooptomd/cuda_check.hpp"
#include "biooptomd/simulation_context.hpp"
#include "core/core_kernels.cuh"
#include "observers/energy_tracker.hpp"

namespace bomd {

EnergyTracker::EnergyTracker(int stride, int capacity) : stride_(stride), capacity_(capacity) {
  if (stride <= 0) throw std::invalid_argument("EnergyTracker: stride muss > 0 sein");
  if (capacity <= 0) throw std::invalid_argument("EnergyTracker: capacity muss > 0 sein");
}

void EnergyTracker::initialize(SimulationContext& ctx) {
  ctx_ = &ctx;
  n_replicas_ = ctx.n_replicas();
  frames_ = DeviceBuffer<double>(static_cast<std::size_t>(capacity_) * n_replicas_ *
                                 detail::kEnergyFrameColumns);
  mid_ = DeviceBuffer<double>(static_cast<std::size_t>(capacity_) * n_replicas_);
  steps_.reserve(static_cast<std::size_t>(capacity_));
}

void EnergyTracker::observe(const DeviceState& s, std::int64_t step, cudaStream_t stream) {
  if (n_frames_ >= capacity_) {
    ++dropped_;
    return;
  }
  double* frame = frames_.data() +
                  static_cast<std::size_t>(n_frames_) * n_replicas_ * detail::kEnergyFrameColumns;
  detail::launch_energy_frame(s, frame, stream);
  BOMD_CUDA_CHECK(cudaMemcpyAsync(mid_.data() + static_cast<std::size_t>(n_frames_) * n_replicas_,
                                  s.d_ekin_mid, n_replicas_ * sizeof(double),
                                  cudaMemcpyDeviceToDevice, stream));
  steps_.push_back(step);
  ++n_frames_;
}

void EnergyTracker::reset() {
  n_frames_ = 0;
  dropped_ = 0;
  steps_.clear();
}

ObserverResult EnergyTracker::download() const {
  const int cols = detail::kEnergyFrameColumns;
  std::vector<double> raw(static_cast<std::size_t>(n_frames_) * n_replicas_ * cols);
  ctx_->activate();
  if (!raw.empty())
    BOMD_CUDA_CHECK(cudaMemcpyAsync(raw.data(), frames_.data(), raw.size() * sizeof(double),
                                    cudaMemcpyDeviceToHost, ctx_->stream()));
  std::vector<double> raw_mid(static_cast<std::size_t>(n_frames_) * n_replicas_);
  if (!raw_mid.empty())
    BOMD_CUDA_CHECK(cudaMemcpyAsync(raw_mid.data(), mid_.data(), raw_mid.size() * sizeof(double),
                                    cudaMemcpyDeviceToHost, ctx_->stream()));
  BOMD_CUDA_CHECK(cudaStreamSynchronize(ctx_->stream()));

  HostArray energies{"energies", "kJ/mol", {n_replicas_, n_frames_, cols}, {}};
  energies.data.resize(raw.size());
  for (int f = 0; f < n_frames_; ++f)  // (frame, replica, col) → (replica, frame, col)
    for (int r = 0; r < n_replicas_; ++r)
      for (int c = 0; c < cols; ++c)
        energies.data[(static_cast<std::size_t>(r) * n_frames_ + f) * cols + c] =
            raw[(static_cast<std::size_t>(f) * n_replicas_ + r) * cols + c];

  HostArray steps{"steps", "1", {n_frames_}, {}};
  steps.data.reserve(steps_.size());
  for (std::int64_t st : steps_) steps.data.push_back(static_cast<double>(st));  // exakt < 2^53

  HostArray mid{"kinetic_mid", "kJ/mol", {n_replicas_, n_frames_}, {}};
  mid.data.resize(raw_mid.size());
  for (int f = 0; f < n_frames_; ++f)
    for (int r = 0; r < n_replicas_; ++r)
      mid.data[static_cast<std::size_t>(r) * n_frames_ + f] =
          raw_mid[static_cast<std::size_t>(f) * n_replicas_ + r];

  ObserverResult result;
  result.arrays.push_back(std::move(energies));
  result.arrays.push_back(std::move(steps));
  result.arrays.push_back(std::move(mid));
  return result;
}

}  // namespace bomd
