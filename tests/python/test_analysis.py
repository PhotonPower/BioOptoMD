"""biooptomd.analysis (ROADMAP Phase 7, Aufgabe 4) – ohne GPU."""

import numpy as np
import pytest

from biooptomd.analysis import (
    block_analysis,
    block_average,
    export_refractive_index,
    fret_efficiency,
    refractive_index_layers,
)


def test_fret_regimes():
    """κ² = 2/3, fester Abstand: alle Regime = 1/(1 + (R/R₀)⁶). Zwei Abstände: statisch = Mittel der
    Effizienzen, dynamisch = Effizienz der mittleren Rate (≥ statisch, Jensen)."""
    r0 = 5.0
    r = np.full(100, 4.0)
    k2 = np.full(100, 2.0 / 3.0)
    e0 = 1.0 / (1.0 + (4.0 / 5.0) ** 6)
    for regime in ("dynamic", "static", "static_isotropic"):
        assert fret_efficiency(r, k2, r0, regime) == pytest.approx(e0, rel=1e-12)

    r = np.array([3.0, 7.0])
    k2 = np.full(2, 2.0 / 3.0)
    q = (r0 / r) ** 6
    assert fret_efficiency(r, k2, r0, "static") == pytest.approx(np.mean(q / (1 + q)))
    assert fret_efficiency(r, k2, r0, "dynamic") == pytest.approx(q.mean() / (1 + q.mean()))
    assert fret_efficiency(r, k2, r0, "dynamic") > fret_efficiency(r, k2, r0, "static")

    # κ² ignoriert im isotropen Regime; Shape (n_replicas, n_frames) → (n_replicas,)
    rng = np.random.default_rng(1)
    r = rng.uniform(3, 7, (3, 50))
    k2 = rng.uniform(0, 4, (3, 50))
    out = fret_efficiency(r, k2, r0, "static_isotropic")
    assert out.shape == (3,)
    np.testing.assert_allclose(out, fret_efficiency(r, np.full_like(k2, 2 / 3), r0, "static"))
    with pytest.raises(ValueError):
        fret_efficiency(r, k2, r0, "mean")


def test_block_average_ar1():
    """AR(1)-Reihe x_t = φ x_{t−1} + ξ: Standardfehler des Mittels
    σ_x/√N · √((1+φ)/(1−φ)); Blockanalyse trifft ihn innerhalb von 4 Schätzfehlern, naive
    Formel unterschätzt um den Faktor ≈ √((1+φ)/(1−φ))."""
    rng = np.random.default_rng(2)
    phi, n = 0.9, 2**17
    x = np.empty(n)
    x[0] = 0.0
    xi = rng.normal(0.0, 1.0, n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + xi[t]
    sigma_x = 1.0 / np.sqrt(1 - phi**2)
    exact = sigma_x / np.sqrt(n) * np.sqrt((1 + phi) / (1 - phi))
    res = block_analysis(x)
    k = int(np.argmin(np.abs(res["sem"] - res["sem_plateau"])))
    assert abs(res["sem_plateau"] - exact) < 4 * res["sem_error"][k] + 0.05 * exact
    assert res["sem"][0] < 0.35 * exact  # naive Schätzung (Blockgröße 1)
    mean, sem = block_average(x, n_blocks=32)
    assert abs(mean) < 4 * exact
    assert sem == pytest.approx(exact, rel=0.35)  # 32 Blöcke: rel. Fehler ≈ 1/√62 ≈ 13 %
    m2, s2 = block_average(np.stack([x, 2 * x]), n_blocks=32)
    np.testing.assert_allclose(m2, [mean, 2 * mean])
    np.testing.assert_allclose(s2, [sem, 2 * sem])


def test_refractive_index_layers(tmp_path):
    """Schichtstapel: gleiche Nachbarschichten zusammengefasst, Halbraum n_w entfernt; CSV-Export."""
    dz, n_w = 0.1, 1.333
    z = (np.arange(10) + 0.5) * dz
    n = np.array([1.45, 1.45, 1.45, 1.40, 1.40, n_w, n_w, n_w, n_w, n_w])
    thick, vals = refractive_index_layers(z, n, dz, n_w)
    np.testing.assert_allclose(thick, [0.3, 0.2])
    np.testing.assert_allclose(vals, [1.45, 1.40])
    thick, vals = refractive_index_layers(z, n, dz, n_w, tol=0.06)
    np.testing.assert_allclose(thick, [0.5])
    np.testing.assert_allclose(vals, [(1.45 * 0.3 + 1.40 * 0.2) / 0.5])
    with pytest.raises(ValueError):
        refractive_index_layers(z, np.where(z > 0.5, np.nan, n), dz, n_w)

    path = tmp_path / "nz.csv"
    export_refractive_index(path, z, np.stack([n, n]), header="λ = 633 nm")
    data = np.loadtxt(path, delimiter=",", encoding="utf-8")
    assert data.shape == (10, 3)
    np.testing.assert_allclose(data[:, 0], z)
    np.testing.assert_allclose(data[:, 2], n)
