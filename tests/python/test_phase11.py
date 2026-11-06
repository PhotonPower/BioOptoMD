"""Abnahmetests Phase 11: Coarse-Grained-Modelle (ADR 0017).

11.1 kosinus-harmonische Winkel und Paarform 12-10 (Gō) im ``BondedForce``.
Siehe docs/ROADMAP.md, Phase 11, und docs/adr/0017-coarse-grained-models.md.
"""

from pathlib import Path

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
    # Absolutgenauigkeit des FP32-Kernels im Minimum (relativ schlecht konditioniert, F ≈ 0):
    # Rückwärtsfehler von etwa einem ulp in r ⇒ |ΔF| ≈ V''(σ) · σ · ε_mach mit V'' = 120 ε/σ².
    eps_mach = float(np.finfo(np.float32).eps)
    atol = 1.5 * 120.0 * 3.0 / 0.6**2 * 0.6 * eps_mach  # ≈ 5e-5 kJ/mol/nm
    np.testing.assert_allclose(f2, f2_ref, rtol=0, atol=atol)
    with pytest.raises(ValueError):
        _eval({**topo, "pair_form": np.full(m, 3, np.int32)}, pos)
    with pytest.raises(ValueError):
        _eval({**topo, "pair_form": form[:-1]}, pos)


# --- 11.2 Builder Cα-ENM/Gō ---------------------------------------------------------------------

UBQ = Path(__file__).resolve().parents[1] / "reference" / "inputs" / "1ubq.pdb"


def _pdb_line(rec, serial, name, resname, chain, resseq, xyz_nm, element):
    x, y, z = (10.0 * np.asarray(xyz_nm)).tolist()
    return (f"{rec:<6}{serial:>5} {name:<4} {resname:>3} {chain}{resseq:>4}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {element:>2}\n")  # fmt: skip


def _two_chain_pdb(path, gap=False):
    """Zwei Ketten à 5 Residuen (Cα-Abstand 0.38 nm entlang x, Ketten 0.6 nm in y versetzt), Cys 3 in
    beiden Ketten mit SG–SG 0.20 nm (Disulfid), HIP in A, MSE als HETATM in B; optional eine Lücke
    (Cα-Abstand 0.8 nm) in Kette B zwischen Residuum 4 und 5."""
    lines, serial = [], 1
    for chain, y0, names in (("A", 0.0, ["ALA", "HIP", "CYS", "LYS", "GLY"]),
                             ("B", 0.6, ["GLU", "MSE", "CYS", "ALA", "SER"])):  # fmt: skip
        for k, res in enumerate(names):
            x = 0.38 * k + (0.42 if gap and chain == "B" and k == 4 else 0.0)
            ca = np.array([x, y0, 0.0])
            rec = "HETATM" if res == "MSE" else "ATOM"
            lines.append(_pdb_line(rec, serial, "CA", res, chain, k + 1, ca, "C"))
            serial += 1
            if res == "CYS":
                sg = ca + np.array([0.0, 0.2 if chain == "A" else -0.2, 0.0])
                lines.append(_pdb_line(rec, serial, "SG", res, chain, k + 1, sg, "S"))
                serial += 1
    path.write_text("".join(lines) + "END\n", encoding="utf-8")
    return path


def test_cg_builder_ubiquitin():
    """Builder (ADR 0017 Stufe 1) an Ubiquitin (1UBQ, 76 Residuen, eine Kette ⇒ eine Domäne):
    76 Perlen am Cα; Gesamtmasse = Summenformel C378H629N105O118S (8564.8 Da, rel. 2e-4, inkl.
    H₂O der Kette); Nettoladung 0 (7 Lys + 4 Arg, 5 Asp + 6 Glu, Termini heben sich auf);
    75 Kettenbindungen, 74 kosinus-harmonische Cα-Winkel mit θ0 aus der Struktur; Federn nur unter
    0.9 nm; keine Gō-Kontakte; Ausschlüsse = Kette 1-2 und 1-3; Optik-Summenregel: Σ α′, v der Perlen
    = Summe der kalibrierten Atomwerte der exakten Summenformel (rel. 1e-6); dn/dc ≈ 0.19 mL/g."""
    from biooptomd import builder, cg

    m = cg.enm_go_from_pdb(UBQ)
    spec = m.spec
    assert spec.n_atoms == 76
    formula = {"C": 378, "H": 629, "N": 105, "O": 118, "S": 1}
    mass = sum(k * builder.ELEMENT_MASS[e] for e, k in formula.items())
    assert spec.masses.sum() == pytest.approx(mass, rel=1e-12)
    assert mass == pytest.approx(8564.8, rel=2e-4)
    assert spec.charges.sum() == pytest.approx(0.0, abs=1e-12)
    b = spec.bonded
    n_chain = int((b["bond_k"] == 1.0e4).sum())
    assert n_chain == 75 and len(m.disulfides) == 0
    assert len(b["angles"]) == 74 and np.all(b["angle_form"] == 1)
    pos = spec.positions
    a, c = (
        pos[b["angles"][:, 0]] - pos[b["angles"][:, 1]],
        pos[b["angles"][:, 2]] - pos[b["angles"][:, 1]],
    )
    cos0 = (a * c).sum(1) / np.linalg.norm(a, axis=1) / np.linalg.norm(c, axis=1)
    np.testing.assert_allclose(np.cos(b["angle_theta0"]), cos0, rtol=1e-12, atol=1e-12)
    enm = m.enm_pairs
    assert len(enm) > 200 and len(m.go_pairs) == 0
    d = np.linalg.norm(pos[enm[:, 0]] - pos[enm[:, 1]], axis=1)
    assert d.max() < 0.9 and np.all(np.abs(enm[:, 1] - enm[:, 0]) >= 2)
    np.testing.assert_allclose(b["bond_r0"][n_chain:], d, rtol=1e-12)
    excl = {tuple(p) for p in spec.excluded.tolist()}
    assert excl == {(i, i + 1) for i in range(75)} | {(i, i + 2) for i in range(74)}

    table = builder.calibrate_optics("protein")[0]
    alpha_atoms = sum(k * table[e][0] for e, k in formula.items())
    vol_atoms = sum(k * table[e][1] for e, k in formula.items())
    assert m.alpha[m.types].astype(np.float64).sum() == pytest.approx(alpha_atoms, rel=1e-6)
    assert m.volume[m.types].astype(np.float64).sum() == pytest.approx(vol_atoms, rel=1e-6)
    assert builder.dn_dc_model(formula, table, 1.3330) == pytest.approx(0.19, abs=0.01)


def test_cg_builder_domains():
    """Domänen: Federn nur innerhalb einer Domäne, Gō-Kontakte (12-10, σ = nativer Abstand) nur
    zwischen Domänen bzw. mit Gelenkresiduen; Gelenkresiduen ohne Federn, aber mit Winkeln; Gō-Paare
    im Nonbonded-Kernel ausgeschlossen, Federn nicht (MARTINI-Konvention); ``go_form="12-6"``: Minimum
    beim nativen Abstand (σ = d/2^(1/6)); fehlerhafte Domänen werfen."""
    from biooptomd import cg

    m = cg.enm_go_from_pdb(UBQ, domains=[[("A", 1, 35)], [("A", 41, 76)]])
    dom = m.domain
    assert np.all(dom[m.enm_pairs[:, 0]] == dom[m.enm_pairs[:, 1]])
    assert np.all(dom[m.enm_pairs[:, 0]] >= 0)
    assert len(m.go_pairs) > 5
    assert np.all((dom[m.go_pairs[:, 0]] != dom[m.go_pairs[:, 1]]) | (dom[m.go_pairs[:, 0]] < 0))
    hinge = np.flatnonzero(dom < 0)
    assert len(hinge) == 5 and not np.isin(m.enm_pairs, hinge).any()
    assert np.isin(hinge, m.spec.bonded["angles"][:, 1]).all()  # Biegung auch im Gelenk
    b = m.spec.bonded
    np.testing.assert_array_equal(b["pair_form"], 1)
    pos = m.spec.positions
    d_go = np.linalg.norm(pos[m.go_pairs[:, 0]] - pos[m.go_pairs[:, 1]], axis=1)
    np.testing.assert_allclose(b["pair_sigma"], d_go)
    excl = {tuple(p) for p in m.spec.excluded.tolist()}
    assert all(tuple(p) in excl for p in m.go_pairs.tolist())
    assert not any(tuple(p) in excl for p in m.enm_pairs.tolist() if p[1] - p[0] > 2)

    m6 = cg.enm_go_from_pdb(UBQ, domains=[[("A", 1, 35)], [("A", 41, 76)]], go_form="12-6")
    np.testing.assert_array_equal(m6.spec.bonded["pair_form"], 0)
    np.testing.assert_allclose(m6.spec.bonded["pair_sigma"] * 2 ** (1 / 6), d_go)
    with pytest.raises(ValueError):
        cg.enm_go_from_pdb(UBQ, domains=[[("A", 1, 40)], [("A", 30, 76)]])
    with pytest.raises(ValueError):  # Span ohne Treffer (falsche Kette)
        cg.enm_go_from_pdb(UBQ, domains=[[("B", 1, 40)]])
    with pytest.raises(ValueError):
        cg.enm_go_from_pdb(UBQ, go_form="10-4")


def test_cg_builder_chains_disulfide(tmp_path):
    """Zwei Ketten: ohne ``domains`` je Kette eine Domäne (keine Federn zwischen Ketten);
    Disulfid Cys A3–Cys B3 (SG–SG 0.20 nm) als Bindung k_bond mit nativem r0 und Ausschluss, kein
    Gō-Kontakt dafür; HIP +1; MSE (HETATM) als MET gelesen; Termini je Kette; Lücke in der Kette
    ⇒ Warnung und keine Bindung."""
    from biooptomd import cg

    m = cg.enm_go_from_pdb(_two_chain_pdb(tmp_path / "two.pdb"))
    assert m.spec.n_atoms == 10
    assert [r.name for r in m.residues][5:7] == ["GLU", "MET"]
    np.testing.assert_array_equal(m.disulfides, [[2, 7]])
    b = m.spec.bonded
    k_ss = b["bond_k"][(b["bonds"][:, 0] == 2) & (b["bonds"][:, 1] == 7)]
    assert k_ss.tolist() == [1.0e4]
    assert (2, 7) in {tuple(p) for p in m.spec.excluded.tolist()}
    assert not any(tuple(p) == (2, 7) for p in m.go_pairs.tolist())
    assert np.all(m.domain[:5] == 0) and np.all(m.domain[5:] == 1)
    assert not np.any((m.enm_pairs[:, 0] < 5) & (m.enm_pairs[:, 1] >= 5))
    # Ladungen: A: N+1, HIP +1, LYS +1, C −1; B: N+1, GLU −1, C −1
    np.testing.assert_allclose(m.spec.charges, [1, 1, 0, 1, -1, 0, 0, 0, 0, -1])
    with pytest.warns(RuntimeWarning, match="Kettenlücke"):
        mg = cg.enm_go_from_pdb(_two_chain_pdb(tmp_path / "gap.pdb", gap=True))
    gb = mg.spec.bonded
    chain_bonds = gb["bonds"][gb["bond_k"] == 1.0e4].tolist()
    assert [
        8,
        9,
    ] not in chain_bonds  # keine Kettenbindung über die Lücke (Feder der Domäne erlaubt)


def _openmm_cg(spec, info, groups):
    """OpenMM-System des CG-Modells mit Kraftgruppen 0 Bindungen, 1 Winkel (kosinus-harmonisch),
    2 Gō-Paare (12-10 bzw. 12-6 + DH), 3 Nonbonded (LJ + DH, Switching)."""
    import openmm

    from biooptomd import units

    c_coul = units.ONE_4PI_EPS0
    kappa, eps_r = float(info.kappa[0]), float(info.eps_r[0])
    sys_ = openmm.System()
    for mass in spec.masses:
        sys_.addParticle(mass)
    sys_.setDefaultPeriodicBoxVectors(
        openmm.Vec3(10, 0, 0), openmm.Vec3(0, 10, 0), openmm.Vec3(0, 0, 30)
    )
    b = spec.bonded
    f32 = lambda x: float(np.float32(x))
    hb = openmm.HarmonicBondForce()
    hb.setUsesPeriodicBoundaryConditions(True)
    for (i, j), k, r0 in zip(b["bonds"], b["bond_k"], b["bond_r0"], strict=True):
        hb.addBond(int(i), int(j), f32(r0), f32(k))
    an = openmm.CustomAngleForce("0.5*k*(cos(theta)-c0)^2")
    an.setUsesPeriodicBoundaryConditions(True)
    an.addPerAngleParameter("k")
    an.addPerAngleParameter("c0")
    for (i, j, k_), kk, th in zip(b["angles"], b["angle_k"], b["angle_theta0"], strict=True):
        an.addAngle(int(i), int(j), int(k_), [f32(kk), f32(np.cos(np.float32(th)))])
    go = openmm.CustomBondForce(
        "select(form, eps*(5*(sig/r)^12-6*(sig/r)^10), 4*eps*((sig/r)^12-(sig/r)^6))"
        " + C*qq*exp(-kappa*r)/(epsr*r)"
    )
    go.setUsesPeriodicBoundaryConditions(True)
    for name, value in (("C", c_coul), ("kappa", kappa), ("epsr", eps_r)):
        go.addGlobalParameter(name, value)
    for name in ("qq", "sig", "eps", "form"):
        go.addPerBondParameter(name)
    for (i, j), qq, sg, ep, fm in zip(b["pairs14"], b["pair_qq"], b["pair_sigma"], b["pair_eps"],
                                      b["pair_form"], strict=True):  # fmt: skip
        go.addBond(int(i), int(j), [f32(qq), f32(sg), f32(ep), float(fm)])
    nb = openmm.CustomNonbondedForce(
        "4*eps*((sig/r)^12-(sig/r)^6) + C*q1*q2*exp(-kappa*r)/(epsr*r);"
        "sig=0.5*(sig1+sig2); eps=sqrt(eps1*eps2)"
    )
    for name, value in (("C", c_coul), ("kappa", kappa), ("epsr", eps_r)):
        nb.addGlobalParameter(name, value)
    for name in ("q", "sig", "eps"):
        nb.addPerParticleParameter(name)
    for q, s, e in zip(spec.charges, spec.sigma, spec.epsilon, strict=True):
        nb.addParticle([float(q), float(s), float(e)])
    for i, j in spec.excluded:
        nb.addExclusion(int(i), int(j))
    nb.setNonbondedMethod(openmm.CustomNonbondedForce.CutoffPeriodic)
    nb.setCutoffDistance(info.r_cutoff)
    nb.setUseSwitchingFunction(True)
    nb.setSwitchingDistance(info.r_switch)
    for g, f in enumerate((hb, an, go, nb)):
        f.setForceGroup(g)
        sys_.addForce(f)
    return sys_


@pytest.mark.reference
def test_cg_forces_vs_openmm():
    """Abnahmetest ADR 0017: CG-Ubiquitin mit zwei Domänen gegen OpenMM (Reference, FP64), je
    Kraftterm getrennt: Bindungen (Kette + Federn), kosinus-harmonische Winkel, Gō-Paare (12-10 + DH),
    Nonbonded (LJ + DH mit Switching). Konfiguration: native Struktur mit Verschiebungen σ = 0.015 nm
    (Größenordnung der Gleichverteilung der Bindungsenergie bei 300 K). Je Term max |ΔF| < 1e-4 der
    RMS-Kraft dieses Terms und Energie rel. 1e-4; so wird kein Term von einem steiferen überdeckt."""
    import dataclasses

    import openmm
    from openmm import unit

    from biooptomd import builder, cg

    m = cg.enm_go_from_pdb(UBQ, domains=[[("A", 1, 35)], [("A", 41, 76)]])
    spec = m.spec
    n = spec.n_atoms
    rng = np.random.default_rng(3)
    pos = spec.positions - spec.positions.mean(axis=0) + [5.0, 5.0, 4.0]
    pos = pos + rng.normal(0.0, 0.015, pos.shape)
    b = spec.bonded
    empty = {k: np.asarray(v)[:0] for k, v in b.items()}
    terms = {  # Name: (bonded-Teil, Nonbonded aktiv, OpenMM-Gruppen, Energiespalten)
        "bonds+angles": (
            {
                **empty,
                **{
                    k: b[k]
                    for k in (
                        "bonds",
                        "bond_k",
                        "bond_r0",
                        "angles",
                        "angle_k",
                        "angle_theta0",
                        "angle_form",
                    )
                },
            },
            False,
            {0, 1},
            (1, 2),
        ),
        "go": (
            {
                **empty,
                **{k: b[k] for k in ("pairs14", "pair_qq", "pair_sigma", "pair_eps", "pair_form")},
            },
            False,
            {2},
            (4,),
        ),
        "nonbonded": (empty, True, {3}, (5,)),
    }
    _, info = builder.build_context(spec, 1, (10.0, 10.0), 300.0, 0.15, positions=pos)
    ref_sys = _openmm_cg(spec, info, None)
    octx = openmm.Context(ref_sys, openmm.VerletIntegrator(0.001),
                          openmm.Platform.getPlatformByName("Reference"))  # fmt: skip
    for name, (bonded, nonbonded, groups, cols) in terms.items():
        sub = dataclasses.replace(
            spec, bonded=bonded, charges=spec.charges if nonbonded else np.zeros(n),
            epsilon=spec.epsilon if nonbonded else np.zeros(n),
        )  # fmt: skip
        ctx, _ = builder.build_context(sub, 1, (10.0, 10.0), 300.0, 0.15, positions=pos)
        ctx.compute_forces()
        f_gpu = ctx.forces()[0, :n, :3].astype(np.float64)
        e_gpu = float(sum(ctx.energies()[0, c] for c in cols))
        octx.setPositions(ctx.positions()[0, :n, :3].astype(np.float64) * unit.nanometer)
        st = octx.getState(getEnergy=True, getForces=True, groups=groups)
        f_ref = st.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer)
        e_ref = st.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        rms = float(np.sqrt((f_ref**2).sum(axis=1).mean()))
        err = np.abs(f_gpu - f_ref).max() / rms
        print(f"{name}: max |ΔF|/RMS = {err:.2e} (RMS {rms:.3g}), E {e_gpu:.5g} / {e_ref:.5g}")
        assert err < 1e-4, name
        assert e_gpu == pytest.approx(e_ref, rel=1e-4), name


def _kabsch_align(x, ref):
    """x (m, n, 3) je Frame optimal (Kabsch, gleiche Gewichte) auf ref (n, 3) gedreht/verschoben."""
    xc = x - x.mean(axis=1, keepdims=True)
    rc = ref - ref.mean(axis=0)
    h = np.einsum("fni,nj->fij", xc, rc)
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(np.einsum("fij,fjk->fik", u, vt)))
    u[:, :, 2] *= d[:, None]
    rot = np.einsum("fij,fjk->fik", u, vt)
    return np.einsum("fni,fij->fnj", xc, rot) + ref.mean(axis=0)


def _unwrap(x, box):
    """x (R, n, 3) in x/y periodisch gefaltet ⇒ zusammenhängend relativ zu Perle 0."""
    d = x - x[:, :1]
    d[..., :2] -= np.round(d[..., :2] / box) * box
    return x[:, :1] + d


def test_cg_wall_image():
    """Wand und Bildladungen für Perlen (ADR 0017 Stufe 1, Phase 11.3): CG-Ubiquitin über Gold
    (9-3, Perlen-LJ σ = 0.4 nm, ε = 0.05 kJ/mol nach Lorentz-Berthelot), λ je Residuenname über
    ``cg.solvent_scale`` (Lys 0.5, Glu 0.8, Gly 0.7 – auch für den C-terminalen Typ „GLY+H2O“ –,
    sonst 1; OQ-29 unkalibriert), Bildladungen (f = 1) mit den Perlenladungen. Zwei Replicas in
    verschiedener Höhe (unterste Perle 0.25 bzw. 0.6 nm über der obersten Goldlage). Energien und
    Kräfte jedes Terms getrennt gegen die NumPy-Referenzen (surface_ref, image_charge_ref) wie
    atomistisch: Energie rel. 1e-5, Kräfte rel. 1e-4 zur RMS-Kraft des Terms (Differenz eines
    Kontexts mit nur diesem Term zu einem Kontext ohne Wand und Bilder)."""
    import surface_ref
    from image_charge_ref import image_charges

    from biooptomd import _core as core
    from biooptomd import builder, cg, surface

    m = cg.enm_go_from_pdb(UBQ)
    spec = m.spec
    n = spec.n_atoms
    box = 8.0
    lam = cg.solvent_scale(m, {"LYS": 0.5, "GLU": 0.8, "GLY": 0.7})
    names = m.names.astype(str)
    assert lam[0] == 0.0 and lam.shape == (len(names),)
    assert np.all(lam[names == "LYS"] == 0.5) and np.all(lam[names == "GLU"] == 0.8)
    assert sorted(names[np.char.startswith(names, "GLY")]) == ["GLY", "GLY+H2O"]
    assert np.all(lam[np.char.startswith(names, "GLY")] == 0.7)
    assert np.all(lam[names == "ALA"] == 1.0)
    for bad in ({"TRP": 0.5}, {"GLY+H2O": 0.5}, {"LYS": 0.0}, {"LYS": 1.5}, {"LYS": np.nan}):
        with pytest.raises(ValueError):  # Trp fehlt in Ubiquitin; nur Namen ohne „+H2O“; (0, 1]
            cg.solvent_scale(m, bad)

    pos0 = spec.positions - spec.positions.mean(axis=0) + [box / 2, box / 2, 0.0]
    z_top = surface.MATERIALS["gold"].z_top_nm
    pos = np.stack([pos0 + [0.0, 0.0, z_top + h - pos0[:, 2].min()] for h in (0.25, 0.6)])

    def make(wall, image):
        ctx, info = builder.build_context(spec, 2, (box, box), 300.0, 0.15, positions=pos,
                                          types=m.types)  # fmt: skip
        if wall:
            t = surface.wall_table("gold", info.lj_table, lam)
            ctx.add_planar_surface(t["prefactor"], t["sigma"], form=t["form"], delta=t["delta"],
                                   z0=0.0)  # fmt: skip
        if image:
            builder.add_image_charges(ctx, info, "gold")
        ctx.compute_forces()
        return ctx, info

    ctx, info = make(True, True)
    base, _ = make(False, False)
    only_wall, _ = make(True, False)
    only_image, _ = make(False, True)
    t = surface.wall_table("gold", info.lj_table, lam)
    lj = info.lj_types
    c = t["prefactor"][m.types, lj].astype(np.float64)
    sig = t["sigma"][m.types, lj].astype(np.float64)
    terms = list(core.ENERGY_TERMS)
    col_wall, col_img = 1 + terms.index("surface"), 1 + terms.index("image")
    x = ctx.positions_precise()[:, :n, :3]
    f_base = base.forces()[:, :n, :3].astype(np.float64)
    df_wall = only_wall.forces()[:, :n, :3] - f_base
    df_img = only_image.forces()[:, :n, :3] - f_base
    df = ctx.forces()[:, :n, :3] - f_base
    for r in range(2):
        e_w, f_w, clamped = surface_ref.wall(x[r, :, 2], t["form"], c, sig, t["delta"], 0.0)
        assert not clamped.any()
        kappa = np.broadcast_to(info.kappa, (2,))[r]
        eps_r = np.broadcast_to(info.eps_r, (2,))[r]
        e_i, f_i = image_charges(x[r], spec.charges, (box, box), z_top, 1.0, kappa, eps_r,
                                 info.r_switch, info.r_cutoff)  # fmt: skip
        assert ctx.energies()[r, col_wall] == pytest.approx(e_w.sum(), rel=1e-5)
        assert ctx.energies()[r, col_img] == pytest.approx(e_i, rel=1e-5)
        f_wall = np.zeros_like(f_i)
        f_wall[:, 2] = f_w
        errs = []
        for got, want in ((df_wall[r], f_wall), (df_img[r], f_i), (df[r], f_wall + f_i)):
            errs.append(np.sqrt(((got - want) ** 2).sum(axis=1).mean()) / _rms(want))
        print(f"Replica {r}: E_Wand {e_w.sum():.2f}, E_Bild {e_i:.2f} kJ/mol, rel. RMS Wand/Bild/"
              f"beide {errs[0]:.1e} / {errs[1]:.1e} / {errs[2]:.1e}")  # fmt: skip
        assert max(errs) < 1e-4
        assert e_w.sum() < 0.0  # Perlen im Anziehungsbereich
    # λ wirkt je Residuum: Vorfaktor der Lys-Perlen halb so groß wie mit λ = 1
    t1 = surface.wall_table("gold", info.lj_table, np.ones(len(names)))
    c1 = t1["prefactor"][m.types, lj].astype(np.float64)
    lys = names[m.types] == "LYS"
    assert lys.sum() == 7
    np.testing.assert_allclose(c[lys], 0.5 * c1[lys], rtol=1e-6)
    assert ctx.status() == 0


def _pure_enm(path):
    """Reines elastisches Netz: Kette + Federn, ohne Winkel, Gō, LJ und Ladungen."""
    import dataclasses

    from biooptomd import cg

    m = cg.enm_go_from_pdb(path, k_angle=0.0, charged_termini=False)
    n = m.spec.n_atoms
    spec = dataclasses.replace(m.spec, charges=np.zeros(n), epsilon=np.zeros(n))
    return m, spec


def test_enm_fluctuations_vs_anm(tmp_path):
    """Abnahmetest ADR 0017: reines elastisches Netz (Ubiquitin-Kern, Residuen 1–72, Kette k = 1e4
    + Federn 500 kJ/mol/nm², ohne Winkel, Gō, LJ, Ladungen) – Fluktuationen ⟨Δr_i²⟩ aus
    Langevin-MD (BAOAB, Δt = 10 fs, γ = 1/ps, 64 Replicas × 400 ps nach 50 ps, Kabsch-Ausrichtung
    auf die native Struktur) gegen die Normalmodenanalyse k_B T · Spur der 3×3-Blöcke der
    Pseudoinversen der Hesse-Matrix (genau 6 Nullmoden). Kriterien: Korrelation > 0.95, Mittel
    rel. 5 %.

    Das C-terminale Ende 73–76 (in Lösung flexibel) ist im 0.9-nm-Netz nicht starr: Perle 76 hängt
    nur an drei Federn ⇒ siebte Nullmode (Drehung um die Achse 74–75), dafür gibt es kein ANM.
    Temperatur 30 K: Die ANM ist der harmonische Grenzfall T → 0 (3 K: MD/ANM 0.998, 30 K: 1.010 /
    Korrelation 0.99994 – die Abweichung wächst linear in T wie eine Störungskorrektur). Bei 300 K
    misst das reine Netz MD/ANM 1.63, Korrelation 0.85: Der Median je Perle liegt bei 1.16 (Störungs-
    korrektur), das Mittel treiben wenige schwach eingebundene Perlen mit Verhältnissen 1.7–4.8
    (β1/β2-Schleife um Residuen 8–12, Schleife um 47, Kernende 72) – nicht störungstheoretische
    Ausflüge; auch die mittlere Struktur verschiebt sich (gegen sie 1.42). Befund zu OQ-30."""
    from biooptomd import builder, units

    core = tmp_path / "ubq_1_72.pdb"
    core.write_text("".join(line for line in UBQ.read_text().splitlines(keepends=True)
                            if not (line.startswith("ATOM") and int(line[22:26]) > 72)))  # fmt: skip
    _, spec = _pure_enm(core)
    assert spec.n_atoms == 72
    n = spec.n_atoms
    pos = spec.positions
    b = spec.bonded
    hess = np.zeros((3 * n, 3 * n))
    for (i, j), k in zip(b["bonds"], b["bond_k"], strict=True):
        e = pos[j] - pos[i]
        blk = k * np.outer(e, e) / (e @ e)
        for p, q, s in ((i, i, 1.0), (j, j, 1.0), (i, j, -1.0), (j, i, -1.0)):
            hess[3 * p : 3 * p + 3, 3 * q : 3 * q + 3] += s * blk
    w, v = np.linalg.eigh(hess)
    assert np.all(np.abs(w[:6]) < 1e-8 * w[-1]) and w[6] > 1e-6 * w[-1]  # starr: 6 Nullmoden
    temp = 30.0
    cov = units.KB * temp * (v[:, 6:] / w[6:]) @ v[:, 6:].T
    msf_anm = np.array([np.trace(cov[3 * i : 3 * i + 3, 3 * i : 3 * i + 3]) for i in range(n)])

    reps, box = 64, 20.0
    start = pos - pos.mean(axis=0) + [box / 2, box / 2, 0.0]
    # ε_r(T) von Wasser gilt erst ab 273 K; ohne Ladungen sind ε_r und κ ohne Wirkung.
    ctx, _ = builder.build_context(spec, reps, (box, box), temp, 0.15, positions=start,
                                   eps_r=78.5, kappa=1.0)  # fmt: skip
    ctx.set_integrator(0.01, 1.0, seed=11)
    ctx.step(5_000)  # 50 ps Äquilibrierung
    frames = []
    for _ in range(400):  # 400 ps, Abstand 1 ps
        ctx.step(100)
        frames.append(_unwrap(ctx.positions_precise()[:, :n, :3].astype(np.float64), box))
    al = _kabsch_align(np.concatenate(frames), start)
    msf_md = ((al - start) ** 2).sum(axis=2).mean(axis=0)
    corr = np.corrcoef(msf_md, msf_anm)[0, 1]
    ratio = msf_md.mean() / msf_anm.mean()
    print(f"Korrelation {corr:.4f}, Mittel MD/ANM {ratio:.4f}, ⟨Δr²⟩ ANM {msf_anm.mean():.2e} nm²")
    assert corr > 0.95
    assert abs(ratio - 1.0) < 0.05
    assert ctx.status() == 0


@pytest.mark.parametrize("dt_fs", [10, 20])
def test_cg_nve(dt_fs):
    """Abnahmetest ADR 0017: NVE des vollständigen CG-Modells (Ubiquitin mit zwei Domänen: Kette,
    Federn, kosinus-harmonische Winkel, Gō 12-10, LJ + Debye-Hückel) bei Δt = 10 bzw. 20 fs, zwei
    Replicas mit Maxwell-Geschwindigkeiten bei 300 K aus der nativen Struktur. Wie die atomistischen
    NVE-Tests (test_phase3, test_gb_nve): Drift (Regressionssteigung) < 1e-4 kJ/mol/ps je Perle und
    Fluktuation < 1e-3 · ⟨E_kin⟩; dazu nach ADR 0017 Drift über 100 ps < 1e-4 · ⟨E_kin⟩.

    Messfenster 1 ns, Steigung auf 100 ps umgerechnet (Abnahmeentscheidung offen, vgl. Phase 3,
    2026-11-03): Bei 20 fs schwankt die Gesamtenergie langsam (10-ps-Blockmittel bis ≈ 1e-3 ·
    E_kin); die Steigung aus nur 100 ps hat dort einen statistischen Fehler von ≈ 1e-4 und löst das
    Kriterium nicht auf. Für lineare Drift schätzt das längere Fenster dieselbe Steigung mit etwa
    30-mal kleinerem Fehler (Vorabmessung mit 8 Replicas über 1 ns: ≤ 0.4e-4 je 100 ps bei 20 fs;
    über 100 ps einzelne Replicas bis 8.9e-4). Die Fluktuation skaliert mit Δt² (≈ 0.2e-3 bei
    10 fs, ≈ 0.8e-3 bei 20 fs), Verhalten des Schatten-Hamiltonians."""
    from biooptomd import builder, cg, units

    m = cg.enm_go_from_pdb(UBQ, domains=[[("A", 1, 35)], [("A", 41, 76)]])
    spec = m.spec
    n = spec.n_atoms
    assert len(m.go_pairs) > 0
    box = 10.0
    start = spec.positions - spec.positions.mean(axis=0) + [box / 2, box / 2, 0.0]
    ctx, _ = builder.build_context(spec, 2, (box, box), 300.0, 0.15, positions=start)
    rng = np.random.default_rng(5)
    v = rng.normal(0.0, 1.0, (2, n, 3)) * np.sqrt(units.KB * 300.0 / spec.masses)[None, :, None]
    v -= (v * spec.masses[None, :, None]).sum(axis=1, keepdims=True) / spec.masses.sum()
    ctx.set_velocities(v)
    dt = dt_fs * 1e-3
    ctx.set_integrator(dt, 0.0, seed=0)
    steps = round(1000.0 / dt)
    tracker = ctx.add_energy_tracker(stride=steps // 1000, capacity=1000)
    ctx.step(steps)
    res = tracker.result()
    en = res["energies"]
    total = en[:, :, 0] + en[:, :, 1:].sum(axis=2)
    t_ps = res["steps"] * dt
    slope = np.array([np.polyfit(t_ps, total[r], 1)[0] for r in range(2)])
    ekin = en[:, :, 0].mean(axis=1)
    rel = np.abs(slope) * 100.0 / ekin
    fluct = total.std(axis=1) / ekin
    print(f"Δt {dt_fs} fs: Drift je 100 ps rel. {rel}, je Perle {slope / n} kJ/mol/ps, "
          f"Fluktuation rel. {fluct}")  # fmt: skip
    assert np.all(rel < 1e-4)
    assert np.all(np.abs(slope) / n < 1e-4)
    assert np.all(fluct < 1e-3)
    assert ctx.status() == 0


@pytest.mark.reference
def test_cg_optics_sum_rule():
    """Abnahmetest ADR 0017, Optik: (1) Summenregel – Σ α′ und Σ v der Perlen = Summe der
    kalibrierten Atomwerte des atomistischen Ubiquitins (1UBQ + H nach OpenMM/amber14 bei pH 7,
    1231 Atome, dieselbe Summenformel), rel. 1e-6, also dasselbe dn/dc. (2) n(z) einer CG-Schicht
    (Perlen am Cα) gegen die atomistische Schicht gleicher Konfiguration: Lorentz-Lorenz mit
    Wasser-Hintergrund wie im ε-Observer, Querschnitt 6 × 6 nm, Gauß-Verschmierung w. Die
    Profilabweichung relativ zum Kontrast fällt mit w (Seitenkettenatome liegen bis ≈ 0.6 nm vom
    Cα) und liegt bei w = 0.5 nm unter 5 %; ∫ (n − n_w) dz stimmt bis auf die Nichtlinearität von
    Lorentz-Lorenz überein: Beitrag 2. Ordnung ≈ (max Δn / n_w) · Profilabweichung ≈ 0.03 · 0.05,
    also rel. 2e-3. (3) Engine-Pfad: CG-Modell → ``build_context(types=model.types)`` →
    ``add_refractive_index_profiler`` (Punktverteilung) mit ``model.alpha``/``model.volume``: S_α, S_V
    je Schicht = NumPy-Histogramm der Perlen (rel. 1e-5) und n(z) = Lorentz-Lorenz wie
    ``eps_from_histograms`` (rel. 1e-6)."""
    from openmm import app, unit

    from biooptomd import builder, cg
    from biooptomd.coupling import eps_from_histograms

    m = cg.enm_go_from_pdb(UBQ)
    table = builder.calibrate_optics("protein")[0]
    pdb = app.PDBFile(str(UBQ))
    mod = app.Modeller(pdb.topology, pdb.positions)
    mod.delete([r for r in mod.topology.residues() if r.name == "HOH"])
    mod.addHydrogens(app.ForceField("amber14-all.xml"), pH=7.0)
    el = [a.element.symbol for a in mod.topology.atoms()]
    xyz = np.array(mod.getPositions().value_in_unit(unit.nanometer))
    a_at = np.array([table[e][0] for e in el])
    v_at = np.array([table[e][1] for e in el])
    a_cg = m.alpha[m.types].astype(np.float64)
    v_cg = m.volume[m.types].astype(np.float64)
    assert len(el) == 1231
    assert a_cg.sum() == pytest.approx(a_at.sum(), rel=1e-6)
    assert v_cg.sum() == pytest.approx(v_at.sum(), rel=1e-6)

    area, n_w = 36.0, builder.OPTICS_TARGETS["protein"]["n_water"]
    z = np.linspace(-4.0, 4.0, 801)
    dz = z[1] - z[0]
    z_cg = m.spec.positions[:, 2] - xyz[:, 2].mean()
    z_at = xyz[:, 2] - xyz[:, 2].mean()

    def n_profile(zz, a, v, w):
        g = np.exp(-0.5 * ((z[:, None] - zz[None]) / w) ** 2) / (np.sqrt(2.0 * np.pi) * w)
        s_a, s_v = g @ a * dz, g @ v * dz  # je Schicht der Dicke dz
        return np.sqrt(eps_from_histograms(s_a, s_v, area * dz, n_w))

    errs = []
    for w in (0.2, 0.5, 1.0):
        n_at, n_cg = n_profile(z_at, a_at, v_at, w), n_profile(z_cg, a_cg, v_cg, w)
        err = np.sqrt(((n_cg - n_at) ** 2).sum() / ((n_at - n_w) ** 2).sum())
        i_at, i_cg = (n_at - n_w).sum() * dz, (n_cg - n_w).sum() * dz
        print(f"w = {w} nm: Profilabweichung {err:.3f}, ∫(n − n_w) dz {i_at:.5f} / {i_cg:.5f} nm, "
              f"max n − n_w {np.max(n_at - n_w):.4f}")  # fmt: skip
        errs.append(err)
        assert i_cg == pytest.approx(i_at, rel=2e-3)
    assert errs[0] > errs[1] > errs[2]
    assert errs[1] < 0.05

    # (3) Engine-Pfad mit eingefrorener Konfiguration (eine Replica, keine Dynamik)
    box, dz_e, n_bins = 6.0, 0.05, 140
    pos = m.spec.positions - m.spec.positions.mean(axis=0) + [box / 2, box / 2, 0.0]
    pos[:, 2] += 0.5123 - pos[:, 2].min()  # unterste Perle nicht auf einer Schichtgrenze
    ctx, _ = builder.build_context(m.spec, 1, (box, box), 300.0, 0.15, positions=pos,
                                   types=m.types)  # fmt: skip
    prof = ctx.add_refractive_index_profiler(dz_e, n_bins, m.alpha, m.volume, z_min=0.0,
                                             n_water=n_w, smearing="point")  # fmt: skip
    ctx.observe_now()
    res = prof.result()
    zb = ctx.positions_precise()[0, : m.spec.n_atoms, 2].astype(np.float32)
    k = np.floor(zb / np.float32(dz_e)).astype(np.int64)  # Schichtindex in FP32 wie im Kernel
    s_a = np.bincount(k, weights=a_cg, minlength=n_bins)
    s_v = np.bincount(k, weights=v_cg, minlength=n_bins)
    np.testing.assert_allclose(res["s_alpha_mean"][0], s_a, rtol=1e-5, atol=1e-6 * s_a.max())
    np.testing.assert_allclose(res["s_volume_mean"][0], s_v, rtol=1e-5, atol=1e-6 * s_v.max())
    assert res["s_alpha_mean"][0].sum() == pytest.approx(a_at.sum(), rel=1e-5)  # Summenregel
    n_ref = np.sqrt(eps_from_histograms(res["s_alpha_mean"][0], res["s_volume_mean"][0],
                                        box * box * dz_e, n_w))  # fmt: skip
    np.testing.assert_allclose(res["n"][0], n_ref, rtol=1e-6)


def _igg_example():
    import importlib.util

    path = Path(__file__).resolve().parents[2] / "examples" / "igg_on_gold" / "igg_on_gold.py"
    spec_ = importlib.util.spec_from_file_location("igg_on_gold", path)
    ex = importlib.util.module_from_spec(spec_)
    spec_.loader.exec_module(ex)
    return ex


def test_igg_example():
    """Beispiel 11.4 (examples/igg_on_gold): CG-IgG aus 1IGT – 1316 Perlen, 4 Ketten, Nettoladung 0
    (His neutral), 17 Disulfide wie SSBOND in RCSB 1IGT (darunter 3 zwischen den schweren Ketten im Gelenk und 2
    zwischen leichter und schwerer Kette); 7 Domänen, das Gelenk ohne Domäne (2 × 19 Perlen) ist
    in Fab 1, Fab 2 und Fc nicht enthalten, die Gruppen sind disjunkt und decken alle übrigen
    Perlen ab. Kontext: Startorientierungen als echte Drehungen, unterste Perle 2 nm über Gold;
    Decke ½ k (z − h)² als Gitterpotential: Energie = lineare Interpolation der Knotenwerte
    (rel. 1e-5; gegen die analytische Parabel Fehler ≤ k Δz²/8 je Perle), Kraft = −k (z − h)₊ als
    Differenz zu einem Kontext ohne Decke (das Plugin interpoliert ∇V linear, für die lineare
    Kraft exakt; Toleranz 1e-6 · max |F| aus der FP32-Interpolation); REMD-Lauf über 10 ps
    ohne Statusbits; Messgrößen endlich; Orientierungsklassen nach der Definition im Skript."""
    from biooptomd import _core as core

    ex = _igg_example()
    m = ex.load_model()
    spec = m.spec
    n = spec.n_atoms
    assert n == 1316 and sorted({r.chain for r in m.residues}) == ["A", "B", "C", "D"]
    assert spec.charges.sum() == pytest.approx(0.0, abs=1e-9)
    chain = np.array([r.chain for r in m.residues])
    resseq = np.array([r.resseq for r in m.residues])
    ss = {tuple(sorted((chain[i] + str(resseq[i]), chain[j] + str(resseq[j]))))
          for i, j in m.disulfides}  # fmt: skip
    assert len(ss) == 17
    assert {("B237", "D237"), ("B240", "D240"), ("B242", "D242")} <= ss
    assert {("A214", "B128"), ("C214", "D128")} <= ss
    assert set(np.unique(m.domain)) == set(range(-1, 7))
    hinge = np.flatnonzero(m.domain < 0)
    assert len(hinge) == 38 and set(chain[hinge]) == {"B", "D"}
    assert np.all((resseq[hinge] >= 223) & (resseq[hinge] <= 245))

    ctx, remd, meta = ex.build(n_ladders=1, seed=3, model=m)
    groups = meta["groups"]
    allg = np.concatenate(list(groups.values()))
    assert len(allg) == len(set(allg.tolist())) == n - len(hinge)
    assert not set(allg.tolist()) & set(hinge.tolist())
    x = ex.unwrapped(ctx, n)
    ref = spec.positions - spec.positions.mean(axis=0)
    for r in range(ctx.n_replicas):  # Startstruktur bis auf eine Drehung (keine Spiegelung)
        xc = x[r] - x[r].mean(axis=0)
        u, _, vt = np.linalg.svd(ref.T @ xc)
        assert np.linalg.det(u @ vt) > 0
        assert np.abs(ref @ (u @ vt) - xc).max() < 1e-4
        assert x[r, :, 2].min() - meta["z_top"] == pytest.approx(ex.START_GAP_NM, abs=1e-4)

    # Decke: Slot 0 so verschieben, dass die obersten Perlen über h liegen
    pos = ctx.positions_precise()[:, :n, :3].astype(np.float64)
    pos[0, :, 2] += ex.CEILING_NM + 0.5 - pos[0, :, 2].max()
    ctx.set_positions(pos.astype(np.float32))
    ctx.compute_forces()
    bare, _, _ = ex.build(n_ladders=1, seed=3, model=m, ceiling=False)
    bare.set_positions(pos.astype(np.float32))
    bare.compute_forces()
    df = ctx.forces()[:, :n, :3].astype(np.float64) - bare.forces()[:, :n, :3]
    z = ctx.positions_precise()[0, :n, 2].astype(np.float64)
    above = np.clip(z - ex.CEILING_NM, 0.0, None)
    col = 1 + list(core.ENERGY_TERMS).index("external")
    vals, origin, spacing = ex.ceiling_grid(ex.BOX)
    z_nodes = origin[2] + spacing[2] * np.arange(vals.shape[0])
    e_ref = np.interp(z, z_nodes, vals[:, 0, 0, 0].astype(np.float64), left=0.0).sum()
    e_parab = 0.5 * ex.K_CEILING * (above**2).sum()
    n_above = int((above > 0).sum())
    assert n_above > 3
    assert ctx.energies()[0, col] == pytest.approx(e_ref, rel=1e-5)
    assert 0.0 <= e_ref - e_parab <= n_above * ex.K_CEILING * spacing[2] ** 2 / 8 + 1e-9
    f_ref = np.zeros((n, 3))
    f_ref[:, 2] = -ex.K_CEILING * above
    tol = 1e-6 * np.abs(ctx.forces()[0, :n, :3]).max()
    np.testing.assert_allclose(df[0], f_ref, rtol=0, atol=tol)
    assert np.all(df[1:] == 0.0)
    assert np.all(ctx.energies()[1:, col] == 0.0)

    remd.run(10, ex.EXCHANGE_STEPS)  # 10 ps
    assert ctx.status() == 0
    h, c, a = ex.metrics(ex.unwrapped(ctx, n), meta)
    assert h.shape == (8,) and c.shape == (8, 3) and a.shape == (8, 3)
    assert np.isfinite(h).all() and np.isfinite(a).all() and (np.abs(a) <= 1.0 + 1e-12).all()
    on = ex.GROUP_MIN_CONTACTS
    cases = np.array([[0, 0, 0], [0, 0, on], [on, 0, 0], [on, on, 0], [0, on, on], [on, on, on],
                      [on - 1, 0, on]])  # fmt: skip
    assert ex.classify(cases).tolist() == [0, 1, 2, 2, 3, 4, 1]


def test_igg_example_analyse():
    """``analyse`` des Beispiels 11.4 an synthetischen Zeitreihen (1 Leiter × 8 Zustände, 6 Frames):
    300-K-Zustand je Frame über ``state == 0`` (wandert zwischen den Slots), zweite Hälfte;
    Klassenanteile, ⟨cos⟩ nur über adsorbierte Frames, erste Adsorption (Klasse > 0, irgendein Slot)
    und Desorptionsereignisse je Walker (Slot)."""
    ex = _igg_example()
    on = ex.GROUP_MIN_CONTACTS
    n_frames, m = 6, 8
    contacts = np.zeros((n_frames, m, 3), np.int64)
    contacts[2:, 1] = [on, on, on]  # Slot 1 ab Frame 2 flach
    contacts[3, 4, 2] = on  # Slot 4 nur in Frame 3 „Fc unten“ ⇒ eine Desorption
    state = np.tile(np.arange(m), (n_frames, 1))
    state[3:, [0, 1]] = state[3:, [1, 0]]  # ab Frame 3 liegt der 300-K-Zustand in Slot 1
    height = np.arange(n_frames * m, dtype=float).reshape(n_frames, m)
    cosines = np.full((n_frames, m, 3), 0.5)
    cosines[:, 1] = -0.25
    out = ex.analyse(height, contacts, cosines, state, m)["by_ladder"][0]
    # zweite Hälfte = Frames 3–5, 300 K jeweils in Slot 1 (flach)
    assert out["adsorbed_300K"] == 1.0
    assert out["class_fractions_300K"] == [0.0, 0.0, 0.0, 0.0, 1.0]
    assert out["height_300K_mean_nm"] == pytest.approx(height[3:, 1].mean())
    assert out["cos_axis_300K_mean_adsorbed"] == [-0.25] * 3
    assert out["first_adsorption_ns"] == pytest.approx(3 * ex.frame_ns())
    assert out["desorption_events_by_walker"] == [0, 0, 0, 0, 1, 0, 0, 0]


@pytest.mark.slow
def test_igg_example_resume(tmp_path, monkeypatch):
    """Fortsetzen des Beispiels 11.4 ist bitgleich: ``main`` mit kurzen Frames (2 Austauschzyklen
    zu 50 Schritten, Checkpoint je Frame, 1 Leiter): 4 Frames am Stück gegen 2 Frames + Fortsetzung
    auf 4 Frames; series.npz in allen Größen identisch, summary.json geschrieben."""
    ex = _igg_example()
    monkeypatch.setattr(ex, "FRAME_CYCLES", 2)
    monkeypatch.setattr(ex, "CHUNK_FRAMES", 1)
    ns = lambda k: k * ex.frame_ns()
    args = ["--ladders", "1", "--seed", "5", "--out"]
    ex.main(["--ns", str(ns(4)), *args, str(tmp_path / "full")])
    ex.main(["--ns", str(ns(2)), *args, str(tmp_path / "part")])
    ex.main(["--ns", str(ns(4)), *args, str(tmp_path / "part")])
    with (
        np.load(tmp_path / "full" / "series.npz") as a,
        np.load(tmp_path / "part" / "series.npz") as b,
    ):
        assert a["height"].shape == (4, 8)
        for key in ("height", "contacts", "cosines", "state", "time_ns"):
            np.testing.assert_array_equal(a[key], b[key], err_msg=key)
    assert (tmp_path / "part" / "summary.json").exists()
