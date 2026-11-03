// BioOptoMD – Ringpuffer für Observer-Zeitreihen im VRAM (ROADMAP Phase 6, Aufgabe 2).
// Frames der Größe (n_replicas, n_cols) double; ist der Puffer voll, werden die ältesten Frames
// überschrieben. Download nur auf Anfrage, in zeitlicher Reihenfolge.
#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <vector>

#include "biooptomd/cuda_check.hpp"
#include "biooptomd/device_buffer.hpp"

namespace bomd {

class DeviceRingBuffer {
 public:
  DeviceRingBuffer() = default;
  DeviceRingBuffer(int capacity, int n_replicas, int n_cols)
      : capacity_(capacity),
        n_replicas_(n_replicas),
        n_cols_(n_cols),
        data_(static_cast<std::size_t>(capacity) * n_replicas * n_cols),
        steps_(capacity) {}

  /// Zeiger auf den nächsten Frame-Slot (n_replicas × n_cols double); danach commit(step).
  double* next_frame() {
    return data_.data() + static_cast<std::size_t>(total_ % capacity_) * n_replicas_ * n_cols_;
  }
  void commit(std::int64_t step) {
    steps_[static_cast<std::size_t>(total_ % capacity_)] = step;
    ++total_;
  }
  void reset() { total_ = 0; }

  int n_frames() const { return static_cast<int>(total_ < capacity_ ? total_ : capacity_); }
  /// Anzahl überschriebener (verlorener) Frames.
  std::int64_t overwritten() const { return total_ > capacity_ ? total_ - capacity_ : 0; }
  int n_replicas() const { return n_replicas_; }
  int n_cols() const { return n_cols_; }

  /// Host-Kopie, Layout (n_replicas, n_frames, n_cols), älteste zuerst; synchronisiert `stream`.
  void download(cudaStream_t stream, std::vector<double>& out,
                std::vector<std::int64_t>& steps) const {
    const int nf = n_frames();
    const std::size_t frame = static_cast<std::size_t>(n_replicas_) * n_cols_;
    std::vector<double> raw(static_cast<std::size_t>(capacity_) * frame);
    data_.download_async(raw.data(), stream);
    BOMD_CUDA_CHECK(cudaStreamSynchronize(stream));
    const std::int64_t first = total_ - nf;
    out.assign(static_cast<std::size_t>(nf) * frame, 0.0);
    steps.resize(nf);
    for (int f = 0; f < nf; ++f) {
      const std::size_t slot = static_cast<std::size_t>((first + f) % capacity_);
      steps[f] = steps_[slot];
      for (int r = 0; r < n_replicas_; ++r)
        for (int c = 0; c < n_cols_; ++c)
          out[(static_cast<std::size_t>(r) * nf + f) * n_cols_ + c] =
              raw[slot * frame + static_cast<std::size_t>(r) * n_cols_ + c];
    }
  }

 private:
  int capacity_ = 0, n_replicas_ = 0, n_cols_ = 0;
  std::int64_t total_ = 0;
  DeviceBuffer<double> data_;
  std::vector<std::int64_t> steps_;
};

}  // namespace bomd
