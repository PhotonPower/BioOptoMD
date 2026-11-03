// BioOptoMD – Fehlerprüfung für CUDA-API-Aufrufe und Kernel-Launches (AGENTS.md Regel 10).
//
//   BOMD_CUDA_CHECK(cudaMemcpyAsync(...));   // jede CUDA-API-Rückgabe
//   k_foo<<<grid, block, 0, stream>>>(...);
//   BOMD_LAUNCH_CHECK();                      // direkt nach jedem Kernel-Launch
//
// Beide Makros werfen bomd::CudaError (Host-Code). BOMD_LAUNCH_CHECK synchronisiert nicht
// (Regel 11); asynchrone Kernel-Fehler werden beim nächsten synchronisierenden Aufruf gemeldet.
// Mit -DBOMD_SYNC_LAUNCH_CHECK (nur zum Debuggen!) synchronisiert es zusätzlich das Gerät, damit
// ein Fehler dem richtigen Kernel zugeordnet wird.
#pragma once

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace bomd {

/// Ausnahme für CUDA-Laufzeitfehler. what() enthält Ausdruck, Datei, Zeile und CUDA-Meldung.
class CudaError : public std::runtime_error {
 public:
  CudaError(cudaError_t code, const char* expr, const char* file, int line)
      : std::runtime_error(std::string("CUDA-Fehler ") + cudaGetErrorName(code) + " (" +
                           cudaGetErrorString(code) + ") in " + expr + " bei " + file + ":" +
                           std::to_string(line)),
        code_(code) {}

  cudaError_t code() const noexcept { return code_; }

 private:
  cudaError_t code_;
};

namespace detail {

inline void cuda_check(cudaError_t code, const char* expr, const char* file, int line) {
  if (code != cudaSuccess) {
    (void)cudaGetLastError();  // Nicht-persistente Fehler zurücksetzen (Folgeaufrufe sauber).
    throw CudaError(code, expr, file, line);
  }
}

}  // namespace detail
}  // namespace bomd

#define BOMD_CUDA_CHECK(expr) ::bomd::detail::cuda_check((expr), #expr, __FILE__, __LINE__)

#ifdef BOMD_SYNC_LAUNCH_CHECK
#define BOMD_LAUNCH_CHECK()                                                                      \
  do {                                                                                           \
    ::bomd::detail::cuda_check(cudaGetLastError(), "kernel launch", __FILE__, __LINE__);         \
    ::bomd::detail::cuda_check(cudaDeviceSynchronize(), "kernel execution", __FILE__, __LINE__); \
  } while (0)
#else
#define BOMD_LAUNCH_CHECK() \
  ::bomd::detail::cuda_check(cudaGetLastError(), "kernel launch", __FILE__, __LINE__)
#endif
