// nanobind-Modul biooptomd._core – Phase-0-Platzhalter.
#include <cuda_runtime.h>
#include <nanobind/nanobind.h>

namespace nb = nanobind;

NB_MODULE(_core, m) {
  m.doc() = "BioOptoMD C++/CUDA-Kern";
  m.def(
      "cuda_device_count",
      []() {
        int n = 0;
        if (cudaGetDeviceCount(&n) != cudaSuccess) return 0;
        return n;
      },
      "Anzahl sichtbarer CUDA-Geräte.");
}
