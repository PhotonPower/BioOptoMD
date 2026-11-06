"""Abnahmetests Phase 11: Coarse-Grained-Modelle (ADR 0017).

11.1 kosinus-harmonische Winkel und Paarform 12-10 (Gō) im ``BondedForce``.
Siehe docs/ROADMAP.md, Phase 11, und docs/adr/0017-coarse-grained-models.md.
"""

import bonded_ref as ref
import numpy as np
import pytest

pytestmark = pytest.mark.gpu

_core = pytest.importorskip("biooptomd._core")

KAPPA, EPS_R = 1.0, 78.5


def _eval(topo, pos, box=10.0, kappa=KAPPA, eps_r=EPS_R):
    n = pos.shape[0]
    ctx = _core.SimulationContext(1, n)
    ctx.set_box(np.full((1, 2), box, np.float32))
    ctx.set_masses(np.full((1, n), 72.0, np.float32))
    ctx.set_positions(pos[None].astype(np.float32))
    ctx.add_bonded([topo], np.zeros(1, np.int32), np.full(1, kappa, np.float32),
                   np.full(1, eps_r, np.float32))  # fmt: skip
    ctx.compute_forces()
    return ctx.forces()[0, :n, :3].astype(np.float64), ctx.energies()[0]


def _rms(f):
    return float(np.sqrt((f**2).sum(axis=1).mean()))


def _triplets(rng, thetas, r=(0.3, 0.5)):
    """Je Winkel θ ein Atomtripel (i, j, k) mit Scheitel j bei Abständen in r [nm]."""
    pos = []
    for t in thetas:
        center = rng.uniform(1.0, 9.0, 3)
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        perp = np.cross(axis, rng.normal(size=3))
        perp /= np.linalg.norm(perp)
        la, lb = rng.uniform(*r, 2)
        pos += [center + la * axis, center, center + lb * (np.cos(t) * axis + np.sin(t) * perp)]
    return np.array(pos)


def test_cosine_angle():
    """Kosinus-harmonischer Winkel V = ½k(cos θ − cos θ0)² (ADR 0017, Phase 11.1): Kräfte und
    Energie gegen die FP64-Referenz (rel. 1e-5 zur RMS-Kraft bzw. 1e-5), Referenz selbst gegen
    finite Differenzen (rel. 1e-6); nahe θ = 0 und π bleibt die Kraft endlich und geht gegen 0; über
    die Box hinweg (Minimum Image). Die harmonische θ-Form ist unverändert (gemischte Liste)."""
    rng = np.random.default_rng(11)
    thetas = np.array([0.3, 1.0, 1.7, 2.4, 3.0, 1e-4, np.pi - 1e-4, 2.0])
    pos = _triplets(rng, thetas)
    pos[-3:] += [9.7, 0.0, 0.0]  # letztes Tripel über die Box gefaltet
    pos[:, :2] %= 10.0
    pos = pos.astype(np.float32).astype(np.float64)  # wie auf der GPU gespeichert
    m = len(thetas)
    idx = np.arange(3 * m).reshape(m, 3)
    k = rng.uniform(20.0, 80.0, m)
    theta0 = rng.uniform(1.5, 2.6, m)
    form = np.ones(m, np.int32)
    form[2] = 0  # ein harmonischer θ-Winkel in derselben Liste
    topo = {"angles": idx, "angle_k": k, "angle_theta0": theta0, "angle_form": form}
    f, e = _eval(topo, pos)

    cos_sel = form == 1
    e_c, f_c = ref.angles_cosine(pos, idx[cos_sel], k[cos_sel], theta0[cos_sel], box=(10.0, 10.0))
    e_h, f_h = ref.angles(pos, idx[~cos_sel], k[~cos_sel], theta0[~cos_sel], box=(10.0, 10.0))
    f_ref = f_c + f_h
    assert np.abs(f - f_ref).max() < 1e-5 * _rms(f_ref)
    assert e[2] == pytest.approx(e_c + e_h, rel=1e-5)

    # Referenz gegen finite Differenzen (ein Tripel im Inneren)
    p0 = pos[3:6].copy()
    h = 1e-6
    for a in range(3):
        for c in range(3):
            pp, pm = p0.copy(), p0.copy()
            pp[a, c] += h
            pm[a, c] -= h
            num = -(ref.angles_cosine(pp, [[0, 1, 2]], k[1:2], theta0[1:2])[0]
                    - ref.angles_cosine(pm, [[0, 1, 2]], k[1:2], theta0[1:2])[0]) / (2 * h)  # fmt: skip
            ana = ref.angles_cosine(p0, [[0, 1, 2]], k[1:2], theta0[1:2])[1][a, c]
            assert num == pytest.approx(ana, rel=1e-6, abs=1e-6)

    # θ → 0 und π: endliche, kleine Kräfte (∝ sin θ)
    for t in (5, 6):
        ft = f[3 * t : 3 * t + 3]
        assert np.isfinite(ft).all()
        assert np.abs(ft).max() < 1e-2 * k[t]

    with pytest.raises(ValueError):
        _eval({**topo, "angle_form": np.full(m, 2, np.int32)}, pos)
    with pytest.raises(ValueError):
        _eval({**topo, "angle_form": form[:-1]}, pos)


def test_pair_go1210():
    """Paarterm 12-10 (Gō nach Karanicolas-Brooks, ADR 0017): V = ε[5(σ/r)¹² − 6(σ/r)¹⁰] plus
    Debye-Hückel; Minimum −ε bei r = σ; gemischt mit 12-6-Paaren in einer Liste; Kräfte und Energie
    gegen die FP64-Referenz (rel. 1e-5) bei Abständen 0.85 σ … 1.6 σ."""
    rng = np.random.default_rng(12)
    m = 10
    sigma = rng.uniform(0.5, 0.8, m)
    pos = []
    for k in range(m):  # Abstände im thermisch zugänglichen Bereich 0.85 σ … 1.6 σ
        c = rng.uniform(1.0, 9.0, 3)
        d = rng.normal(size=3)
        d /= np.linalg.norm(d)
        pos += [c, c + rng.uniform(0.85, 1.6) * sigma[k] * d]
    pos = np.array(pos).astype(np.float32).astype(np.float64)  # wie auf der GPU gespeichert
    idx = np.arange(2 * m).reshape(m, 2)
    eps = rng.uniform(1.0, 5.0, m)
    qq = rng.choice([0.0, 1.0, -1.0], m)
    form = np.array([1] * 7 + [0] * 3, np.int32)
    topo = {"pairs14": idx, "pair_qq": qq, "pair_sigma": sigma, "pair_eps": eps, "pair_form": form}
    f, e = _eval(topo, pos)
    g = form == 1
    e_g, f_g = ref.pairs_go1210(pos, idx[g], qq[g], sigma[g], eps[g], KAPPA, EPS_R)
    e_l, f_l = ref.pairs14(pos, idx[~g], qq[~g], sigma[~g], eps[~g], KAPPA, EPS_R)
    f_ref = f_g + f_l
    assert np.abs(f - f_ref).max() < 1e-5 * _rms(f_ref)
    assert e[4] == pytest.approx(e_g + e_l, rel=1e-5)

    # Minimum −ε bei r = σ (ohne Ladung)
    pos2 = np.array([[2.0, 2.0, 2.0], [2.0 + 0.6, 2.0, 2.0]])
    topo2 = {"pairs14": [[0, 1]], "pair_qq": [0.0], "pair_sigma": [0.6], "pair_eps": [3.0],
             "pair_form": [1]}  # fmt: skip
    f2, e2 = _eval(topo2, pos2)
    assert e2[4] == pytest.approx(-3.0, rel=1e-5)
    # Kraft im Minimum: nur die float32-Rundung von r (Krümmung 120 ε/σ² ≈ 1e3 kJ/mol/nm²)
    p32 = pos2.astype(np.float32).astype(np.float64)
    _, f2_ref = ref.pairs_go1210(p32, [[0, 1]], [0.0], [np.float32(0.6)], [3.0], KAPPA, EPS_R)
    # Absolutgenauigkeit des FP32-Kernels im Minimum: s¹² − s¹⁰ ≈ 0 mit ulp ≈ 1e-7 ⇒
    # |ΔF| ≈ 60 ε · 1e-7 / σ ≈ 3e-5 kJ/mol/nm.
    np.testing.assert_allclose(f2, f2_ref, rtol=0, atol=5e-5)
