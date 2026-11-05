// BioOptoMD – RAII-Besitzer für Gerätespeicher (CODING_GUIDELINES: keine rohen cudaMalloc
// außerhalb dieses Wrappers).
#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <utility>

#include "biooptomd/cuda_check.hpp"

namespace bomd {

/// Nicht kopierbarer, verschiebbarer Puffer von `size()` Elementen vom Typ T im Gerätespeicher.
/// Der Inhalt ist nach der Konstruktion undefiniert; zum Initialisieren `zero_async` verwenden.
template <typename T>
class DeviceBuffer {
 public:
  DeviceBuffer() = default;

  explicit DeviceBuffer(std::size_t count) : size_(count) {
    if (count > 0) BOMD_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ptr_), bytes()));
  }

  ~DeviceBuffer() { release(); }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  DeviceBuffer(DeviceBuffer&& other) noexcept
      : ptr_(std::exchange(other.ptr_, nullptr)), size_(std::exchange(other.size_, 0)) {}

  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
    if (this != &other) {
      release();
      ptr_ = std::exchange(other.ptr_, nullptr);
      size_ = std::exchange(other.size_, 0);
    }
    return *this;
  }

  T* data() noexcept { return ptr_; }
  const T* data() const noexcept { return ptr_; }
  std::size_t size() const noexcept { return size_; }
  std::size_t bytes() const noexcept { return size_ * sizeof(T); }

  /// Setzt alle Bytes auf 0 (asynchron auf `stream`).
  void zero_async(cudaStream_t stream) {
    if (size_ > 0) BOMD_CUDA_CHECK(cudaMemsetAsync(ptr_, 0, bytes(), stream));
  }

  /// Kopiert `size()` Elemente von Host nach Gerät (asynchron; `src` muss bis zur
  /// Stream-Synchronisation gültig bleiben).
  void upload_async(const T* src, cudaStream_t stream) {
    if (size_ > 0)
      BOMD_CUDA_CHECK(cudaMemcpyAsync(ptr_, src, bytes(), cudaMemcpyHostToDevice, stream));
  }

  /// Kopiert `size()` Elemente von Gerät nach Host (asynchron).
  void download_async(T* dst, cudaStream_t stream) const {
    if (size_ > 0)
      BOMD_CUDA_CHECK(cudaMemcpyAsync(dst, ptr_, bytes(), cudaMemcpyDeviceToHost, stream));
  }

 private:
  void release() noexcept {
    if (ptr_ != nullptr) {
      cudaFree(ptr_);  // Destruktor: Fehler nicht werfen.
      ptr_ = nullptr;
    }
    size_ = 0;
  }

  T* ptr_ = nullptr;
  std::size_t size_ = 0;
};

}  // namespace bomd
