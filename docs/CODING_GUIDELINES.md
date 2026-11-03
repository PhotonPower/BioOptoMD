# Coding Guidelines

## C++ / CUDA
- C++20 (Host), CUDA 12.x. Formatierung per `.clang-format` (Google-basiert, 100 Spalten).
- Header in `include/biooptomd/` sind öffentlich; interne Header in `src/**` mit Endung `.cuh`.
- Kernels: `__global__ void k_<name>(...)`, Device-Helfer `__device__ __forceinline__`.
- `DeviceState` per Wert an Kernels übergeben (trivially copyable, kein Besitz).
- Kein `using namespace` in Headern. Namespace `bomd`, intern `bomd::detail`.
- Speicher nur über RAII-Wrapper (`DeviceBuffer<T>`), keine rohen `cudaMalloc` außerhalb davon.
- Launch-Konfiguration: Blockgröße als `constexpr` je Kernel dokumentieren; Grid über alle Replicas (`blockIdx.y = replica` oder linearisiert – pro Kernel begründen).
- Jede Formel im Kernel mit Kommentar-Verweis auf `PHYSICS_MODEL.md §x.y`.
- Register-Spills mit `-Xptxas -warn-spills` sichtbar machen; Spills im Non-Bonded-Kernel sind ein Bug.

## Python
- Python ≥ 3.10, Formatierung/Lint mit `ruff`, Typannotationen für öffentliche API.
- NumPy-Arrays an der API-Grenze: dtype und Shape im Docstring, z. B. `positions: float32[n_replicas, n_atoms, 3] in nm`.
- Keine stillen Einheitenkonvertierungen; Konvertierungsfunktionen heißen explizit (`volt_to_kj_mol_e`).

## Git
- Conventional Commits, Branches `phase-N/<thema>`, PR-Vorlage ausfüllen.
- Keine Force-Pushes auf `main`.
