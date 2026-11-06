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
