# BioOptoMD

BioOptoMD ist ein modulares C++/CUDA-Molekulardynamik-Framework mit Python-API, das die Dynamik von Biorezeptoren (Aptamere, Peptide, später Antikörper) an nanostrukturierten Sensoroberflächen simuliert und **in-situ** optische Kopplungsparameter (Übergangsdipole, FRET-Geometrie, Brechungsindexprofile n(z)) als direkten Input für elektromagnetische Optik-Solver (Transfer-Matrix, FEM, BEM) berechnet.

> **Status:** Phase 1 (Build-System, DeviceState, Python-Bindings) in Arbeit. Siehe [`docs/STATUS.md`](docs/STATUS.md).

## Warum eine eigene Engine?

BioOptoMD ist ein Forschungsprojekt mit dem Ziel voller Kontrolle über alle GPU-Kernels. Der technische Schwerpunkt, der es von OpenMM/GROMACS unterscheidet:

1. **Batch-Simulation vieler kleiner Systeme** (z. B. 32–256 Replicas à 1–5 k Atome) auf einer GPU – für Replica Exchange, Mutations-Screening und Statistik.
2. **Sensoroberfläche als First-Class-Citizen** (analytische Wände, elektrochemische Potentiale, 3D-Voxel-Felder aus FEM, laterale Periodizität).
3. **In-situ Optik-Observer**, die direkt im VRAM akkumulieren und Solver-fertige Größen liefern.

Begründung und Abgrenzung: [`docs/adr/0001-eigene-engine.md`](docs/adr/0001-eigene-engine.md).

## Dokumentation

| Dokument | Inhalt |
|---|---|
| [`AGENTS.md`](AGENTS.md) | **Verbindliche Arbeitsanweisungen für KI-Agenten** (und Menschen) |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Systemarchitektur, Speicherlayout, Plugin-Schnittstellen |
| [`docs/UNITS_AND_CONVENTIONS.md`](docs/UNITS_AND_CONVENTIONS.md) | Einheiten, Konstanten, Numerik-Konventionen |
| [`docs/PHYSICS_MODEL.md`](docs/PHYSICS_MODEL.md) | Physikalisches Modell, Potentiale, Gültigkeitsgrenzen |
| [`docs/ROADMAP.md`](docs/ROADMAP.md) | Phasenplan mit Abnahmekriterien |
| [`docs/TESTING.md`](docs/TESTING.md) | Teststrategie, Toleranzen, Referenzdaten |
| [`docs/CODING_GUIDELINES.md`](docs/CODING_GUIDELINES.md) | C++/CUDA/Python-Stil |
| [`docs/OPEN_QUESTIONS.md`](docs/OPEN_QUESTIONS.md) | Offene wissenschaftliche/technische Fragen |
| [`docs/adr/`](docs/adr/) | Architecture Decision Records |

## Zielhardware

| GPU | Compute Capability | Rolle |
|---|---|---|
| RTX 6000 Ada (48 GB) | 8.9 | Primäre Entwicklungs- und Testmaschine (Host: 1 TB RAM) |
| RTX 3090 (24 GB) | 8.6 | Sekundär |

Beide Karten haben schwache FP64-Leistung → Kernels rechnen in FP32, Kräfte werden in 64-bit-Fixed-Point akkumuliert (siehe ADR 0002).

## Build

Linux (Referenz, Conda):

```bash
conda env create -f environment.yml
conda activate biooptomd
pip install -v -e . --no-build-isolation
pytest -m "not slow"
```

Windows (ohne Conda; VS Build Tools 2022, CUDA Toolkit 12.x):

```bat
python -m venv .venv
.venv\Scripts\pip install numpy scipy pytest ruff nanobind scikit-build-core ninja cupy-cuda12x
tools\build_windows.bat
.venv\Scripts\python -m pytest -m "not slow"
```

Optional für die optische Kopplung mit hp-FEM (Phase 10, `biooptomd.coupling.HpFemGratingSolver`,
`HpFemSolver3D`): hp-FEM v0.4.0 (https://github.com/PhotonPower/hp-FEM) mit OpenMP, MUMPS und dem
cuDSS-Lader selbst bauen (MSYS2 UCRT64 mit `gcc`, `cmake`, `ninja`, `mumps`, `metis`; kurzes
Build-Verzeichnis wegen der Windows-Pfadgrenze), mit `delvewheel` die MSYS2-DLLs einpacken und
installieren; die GPU-Bibliothek separat mit `tools\build_hpfem_gpu.bat` (Git Bash):

```bash
git clone --depth 1 --branch v0.4.0 https://github.com/PhotonPower/hp-FEM hpfem_src
PATH="/c/msys64/ucrt64/bin:$PATH" CC=gcc CXX=g++ CMAKE_GENERATOR=Ninja \
  .venv/Scripts/python -m pip wheel hpfem_src -w dist --no-deps \
  --config-settings=cmake.define.HPFEM_ENABLE_MUMPS=ON --config-settings=cmake.define.HPFEM_ENABLE_CUDA=ON \
  --config-settings=build-dir=C:/Users/<name>/hpb
.venv/Scripts/python -m pip install delvewheel
.venv/Scripts/python -m delvewheel repair --add-path C:/msys64/ucrt64/bin -w dist/repaired dist/hpfem-*.whl
.venv/Scripts/python -m pip install --no-deps dist/repaired/hpfem-*.whl
cmd //c "tools\build_hpfem_gpu.bat hpfem_src C:\Users\<name>\hpb\gpu"
export HPFEM_GPU_DLL='C:\Users\<name>\hpb\gpu\bin\hpfem_gpu.dll'   # und cuDSS bin\12 auf PATH
```

## Lizenz

MIT, siehe [`LICENSE`](LICENSE).
