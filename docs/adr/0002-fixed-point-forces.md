# ADR 0002: Fixed-Point-Akkumulation von Kräften und Energien

- **Status:** Accepted
- **Datum:** 2026-11-03

## Kontext
Kraftbeiträge aus Bonded-, Non-Bonded- und Plugin-Kernels werden parallel per Atomics summiert. `atomicAdd` auf `float` ist reihenfolgeabhängig (nicht deterministisch) und verliert Präzision. FP64-Atomics sind auf RTX 6000 Ada/RTX 3090 langsam (FP64 = 1/64 FP32).

## Entscheidung
Kräfte und Energien werden als `long long` mit Skala 2³² akkumuliert (`atomicAdd` auf `unsigned long long`, Zweierkomplement). Ein Konvertierungskernel erzeugt danach `float4 d_force`.

## Alternativen
- `float`-Atomics: schnell, aber nicht deterministisch.
- `double`-Atomics: deterministisch nur bei fester Reihenfolge, langsam.
- Warp-Reduktion ohne Atomics: nicht für alle Kernels möglich.

## Konsequenzen
- Wertebereich: |F| < 2³¹ ≈ 2.1·10⁹ kJ/mol/nm – überlauf praktisch ausgeschlossen; Auflösung 2.3·10⁻¹⁰.
- Bitidentische Trajektorien bei gleichem Seed möglich (Test `test_determinism`).
- Leichter Mehraufwand durch Konvertierungskernel.
