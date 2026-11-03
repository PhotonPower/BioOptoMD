# BioOptoMD

BioOptoMD ist ein modulares C++/CUDA-Molekulardynamik-Framework mit Python-API, das die Dynamik von Biorezeptoren (Aptamere, Peptide, später Antikörper) an nanostrukturierten Sensoroberflächen simuliert und **in-situ** optische Kopplungsparameter (Übergangsdipole, FRET-Geometrie, Brechungsindexprofile n(z)) als direkten Input für elektromagnetische Optik-Solver (Transfer-Matrix, FEM, BEM) berechnet.

> **Status:** Planungsphase / Phase 0 (Repository-Gerüst). Siehe [`docs/STATUS.md`](docs/STATUS.md).

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

## Build (Zielzustand ab Phase 1)

```bash
conda env create -f environment.yml
conda activate biooptomd
pip install -v -e . --no-build-isolation
pytest -m "not slow"
```

## Lizenz

MIT, siehe [`LICENSE`](LICENSE).
