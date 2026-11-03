// BioOptoMD – RAII-Besitzer eines CUDA-Streams.
#pragma once

#include <cuda_runtime.h>

#include <utility>

#include "biooptomd/cuda_check.hpp"

namespace bomd {

/// Nicht kopierbarer, verschiebbarer Besitzer eines blockierenden CUDA-Streams
/// (cudaStreamCreate: implizit geordnet mit dem Legacy-Default-Stream).
class CudaStream {
 public:
  CudaStream() = default;

  /// Erzeugt den Stream auf dem aktuell aktiven Gerät.
  static CudaStream create() {
    CudaStream s;
    BOMD_CUDA_CHECK(cudaStreamCreate(&s.stream_));
    return s;
  }

  ~CudaStream() {
    if (stream_ != nullptr) cudaStreamDestroy(stream_);  // Destruktor: Fehler nicht werfen.
  }

  CudaStream(const CudaStream&) = delete;
  CudaStream& operator=(const CudaStream&) = delete;
  CudaStream(CudaStream&& other) noexcept : stream_(std::exchange(other.stream_, nullptr)) {}
  CudaStream& operator=(CudaStream&& other) noexcept {
    if (this != &other) {
      if (stream_ != nullptr) cudaStreamDestroy(stream_);
      stream_ = std::exchange(other.stream_, nullptr);
    }
    return *this;
  }

  cudaStream_t get() const noexcept { return stream_; }

 private:
  cudaStream_t stream_ = nullptr;
};

}  // namespace bomd
