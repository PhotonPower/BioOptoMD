# Einheiten & Konventionen

## Einheitensystem (verbindlich)

| Größe | Einheit | Bemerkung |
|---|---|---|
| Länge | nm | |
| Zeit | ps | |
| Masse | amu (g/mol) | |
| Energie | kJ/mol | |
| Kraft | kJ/(mol·nm) | |
| Ladung | e | |
| Temperatur | K | |
| Geschwindigkeit | nm/ps | |
| Reibung γ | 1/ps | |
| Dipolmoment | e·nm | Ausgabe optional in Debye: 1 e·nm = 48.0320 D |
| Polarisierbarkeit | nm³ (Polarisierbarkeitsvolumen α′ = α/(4πε₀)) | |
| Elektrisches Potential | kJ/(mol·e) | 1 V = 96.4853 kJ/(mol·e) |
| Ionenstärke | mol/L | nur im Python-Builder, Kernel bekommt κ in 1/nm |

## Konstanten

Nur aus `include/biooptomd/units.hpp` (C++/CUDA) bzw. `biooptomd.units` (Python). Beide Dateien müssen identische Werte enthalten; ein Test prüft das.

| Symbol | Wert | Einheit |
|---|---|---|
| `ONE_4PI_EPS0` | 138.935456 | kJ·nm/(mol·e²) |
| `KB` (k_B) | 0.0083144626 | kJ/(mol·K) |
| `FARADAY` | 96.4853321 | kJ/(mol·V) |
| `DEBYE_PER_ENM` | 48.0320471 | D/(e·nm) |
| `AVOGADRO` | 6.02214076e23 | 1/mol |
| `FIXED_POINT_SCALE` | 2³² = 4294967296 | – |

Debye-Länge (Wasser, 298 K): κ⁻¹ ≈ 0.304 nm / √(I / mol·L⁻¹). Der Builder berechnet κ exakt aus T, ε_r und I.

## Koordinatenkonventionen

- Sensoroberfläche liegt bei **z = 0**, Lösung bei z > 0. Oberflächennormale = +ẑ.
- x/y periodisch mit Box (Lx, Ly) pro Replica, Ursprung bei 0; gewrappte Koordinaten in `[0, L)`. Imagezähler in `d_image` erlauben ungewrapte Rekonstruktion.
- z wird nie gewrappt.
- Übergangsdipol: Einheitsvektor von Ankeratom A nach Ankeratom B, sofern nicht anders definiert. Polarwinkel θ zur Oberflächennormale ∈ [0, π].

## Numerik

- Kernel-Arithmetik in FP32. Ausnahme (ADR 0009): Integrator, `wrap_xy` und Schwerpunkt rechnen mit dem Mixed-Precision-Zustand (Position = `d_pos` + `d_pos_corr`, Geschwindigkeit FP64). `--use_fast_math` ist **verboten** in Kraft-Kernels (Denormals/Genauigkeit), erlaubt nur in Observer-Kernels mit Begründung.
- Kraftakkumulation Fixed-Point int64 (ADR 0002). Energieakkumulation ebenfalls.
- Reduktionen über viele Atome (kinetische Energie, Schwerpunkte) in FP64 oder Kahan-Summe, da nicht im Hot-Path.
- `EPS_R2 = 1e-8 nm²` als Untergrenze für Abstandsquadrate.

## Benennung

- C++: `snake_case` für Funktionen/Variablen, `PascalCase` für Typen, Device-Pointer mit Präfix `d_`, Host-Pointer `h_`.
- Größen mit Einheit im Namen, wo Verwechslung droht: `k_bond_kj_nm2`, `dt_ps`.
- Python: PEP 8; öffentliche Funktionen dokumentieren Einheiten im Docstring.
