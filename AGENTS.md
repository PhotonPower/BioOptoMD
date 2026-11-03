# AGENTS.md – Arbeitsanweisungen für KI-Agenten

Diese Datei ist verbindlich für jeden Agenten (Claude Code, Codex, Copilot, …) und für menschliche Beitragende. Bei Widerspruch zwischen dieser Datei und anderen Dokumenten gilt: **AGENTS.md > ADRs > ARCHITECTURE.md > ROADMAP.md > sonstige Docs**. Widersprüche nicht stillschweigend auflösen, sondern in `docs/OPEN_QUESTIONS.md` eintragen.

---

## 1. Pflichtlektüre vor jeder Arbeitssitzung

In dieser Reihenfolge:

1. `docs/STATUS.md` – Wo steht das Projekt? Welche Phase ist aktiv? Was hat der letzte Agent hinterlassen?
2. `docs/ROADMAP.md` – nur die **aktive Phase** und ihre Abnahmekriterien.
3. `docs/UNITS_AND_CONVENTIONS.md` – immer, vollständig.
4. `docs/ARCHITECTURE.md` – Abschnitte, die deine Aufgabe berühren.
5. Relevante ADRs in `docs/adr/`.

## 2. Arbeitsmodus

### 2.1 Strikt phasenweise
- Arbeite **nur an der aktiven Phase** laut `docs/STATUS.md`.
- Eine Phase ist erst abgeschlossen, wenn **alle** Abnahmekriterien in `docs/ROADMAP.md` erfüllt sind und die zugehörigen Tests auf der Referenz-GPU (RTX 6000 Ada, CC 8.9) grün sind.
- Vorgriffe auf spätere Phasen sind nur als **Interface-Stubs** erlaubt, die in der Architektur bereits vorgesehen sind. Keine „schnell mal mitgemachten“ Features.

### 2.2 Kleine, überprüfbare Schritte
- Ein Commit = eine logische Änderung. Commit-Messages nach [Conventional Commits](https://www.conventionalcommits.org/): `feat(core): …`, `fix(nonbonded): …`, `test(observers): …`, `docs: …`.
- Arbeite auf Branches `phase-N/<kurzbeschreibung>`, Merge nach `main` per Pull Request.
- Jede neue Physik-Funktion kommt **im selben PR** mit Test. Ohne Test kein Merge.

### 2.3 Test-First bei Physik
Bevor ein Kraft- oder Integrator-Kernel geschrieben wird:
1. Analytische Formel für Energie **und** Kraft in einem Kommentar / im Docstring notieren (inkl. Vorzeichen, Einheiten).
2. CPU-Referenzimplementierung in NumPy (`tests/reference/`), finite-Differenzen-Check der Kraft gegen die Energie.
3. Erst dann den CUDA-Kernel, getestet gegen die NumPy-Referenz **und** – sobald verfügbar – gegen OpenMM.

### 2.4 Statusübergabe
Am Ende **jeder** Sitzung `docs/STATUS.md` aktualisieren:
- Was wurde erledigt (mit Commit-Hashes / PR-Nummern)?
- Was ist halb fertig, wo genau?
- Welche Tests sind rot und warum?
- Nächster konkreter Schritt.

Der nächste Agent hat keinen anderen Kontext als das Repository.

## 3. Harte Regeln (Guardrails)

### Physik & Numerik
1. **Einheiten:** ausschließlich nm, ps, amu, kJ/mol, e, K. Konstanten nur aus `include/biooptomd/units.hpp` bzw. `biooptomd.units`. Keine Magic Numbers.
2. **Kraftakkumulation:** ausschließlich über die Fixed-Point-Puffer (`int64`, Skala `2^32`), siehe ADR 0002. Kein `atomicAdd` auf `float` für Kräfte.
3. **Winkel:** Bindungs- und Torsionswinkel über `atan2f` berechnen. Falls doch `acosf` nötig: Eingabe zwingend auf `[-1, 1]` clampen.
4. **Division durch Abstände:** immer gegen `r² < EPS_R2` absichern; dokumentieren, was im Grenzfall passiert.
5. **Determinismus:** Gleicher Seed + gleiche Eingabe ⇒ bitidentische Trajektorie. RNG ist Philox, Zähler = (seed, replica, step, atom). Kein `curand_init` pro Schritt mit sequentiellem State.
6. **Padding-Atome** (Dummies zum Auffüllen auf Vielfache von 32): Ladung 0, LJ-Typ 0 (ε = 0), `inv_mass = 0`, in keiner Bindungsliste, von allen Observern ignoriert. Tests müssen das prüfen.
7. **Periodizität:** x/y periodisch (Minimum Image, Box pro Replica), z nicht periodisch. Jeder Paar-Kernel muss Minimum Image anwenden (ADR 0003).
8. **Thermodynamische Tests** prüfen gegen statistisch begründete Toleranzen (z. B. 4σ der Schätzung über dekorrelierte Samples), nie gegen „exakt“.

### CUDA
9. Keine virtuellen Funktionen, kein `new`/`malloc` in Device-Code, keine Exceptions in Device-Code.
10. Jeder Kernel-Launch über das Makro `BOMD_LAUNCH_CHECK()`; jede CUDA-API-Rückgabe über `BOMD_CUDA_CHECK(...)`.
11. Alle Kernels laufen auf dem Stream des `SimulationContext`; kein implizites Synchronisieren (`cudaDeviceSynchronize`) im Hot-Path.
12. Kein Host↔Device-Transfer pro Zeitschritt. Observer akkumulieren auf dem Device, Download nur auf Python-Anfrage.
13. Ziel-Architekturen: `86;89`. Keine Features jenseits CC 8.6 ohne ADR.

### Code & Repository
14. **Niemals Secrets committen** (Tokens, Passwörter, `.env`). Bei Fund: sofort melden, nicht selbst „verstecken“.
15. Keine neuen Abhängigkeiten ohne Eintrag in `environment.yml` **und** Begründung im PR.
16. Öffentliche C++- und Python-APIs haben Docstrings mit Einheiten jeder Größe.
17. Keine generierten Dateien, Trajektorien oder Binärdaten > 1 MB committen. Referenzdaten werden per Skript in `tests/reference/` erzeugt.

## 4. Entscheidungen & Unsicherheiten

- Architektur- oder Physikentscheidung mit Tragweite → neuer ADR in `docs/adr/NNNN-titel.md` (Vorlage: `docs/adr/0000-template.md`), Status `Proposed`. Der Projektinhaber setzt ihn auf `Accepted`.
- Wissenschaftlich unklare Punkte (Parameterquellen, Modellgültigkeit) → `docs/OPEN_QUESTIONS.md`, **nicht raten**. Eine plausible, klar markierte Platzhalter-Annahme ist erlaubt, wenn sie im Code mit `// ASSUMPTION(OQ-NN):` gekennzeichnet ist.
- Wenn ein Test nur durch Lockern der Toleranz grün wird: Toleranz **nicht** ändern, sondern Ursache untersuchen und in `STATUS.md` dokumentieren.

## 5. Standardbefehle

```bash
# Umgebung
conda env create -f environment.yml && conda activate biooptomd

# Build (editierbar)
pip install -v -e . --no-build-isolation

# Schnelle Tests (Ziel: < 2 min)
pytest -m "not slow"

# Volle Validierung (Phasenabnahme)
pytest

# Nur GPU-unabhängige Tests (CI ohne GPU)
pytest -m "not gpu"

# Referenzdaten neu erzeugen (benötigt OpenMM)
python tests/reference/generate_all.py

# Formatierung
clang-format -i $(git ls-files '*.hpp' '*.cpp' '*.cu' '*.cuh')
ruff format python tests && ruff check python tests
```

## 6. Definition of Done (pro PR)

- [ ] Code kompiliert ohne Warnungen (`-Wall -Wextra`, nvcc `-Xptxas -warn-spills`).
- [ ] Neue/geänderte Physik hat Tests; alle Tests der aktiven und früheren Phasen grün.
- [ ] `compute-sanitizer --tool memcheck` auf den neuen Tests ohne Fehler.
- [ ] Docstrings mit Einheiten; ggf. `docs/` aktualisiert.
- [ ] `docs/STATUS.md` aktualisiert.
