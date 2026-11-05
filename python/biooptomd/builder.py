"""System-Builder (ROADMAP Phase 7, Aufgaben 1–2; docs/ARCHITECTURE.md).

Bündelt die Schritte vom Kraftfeld-System zum gefüllten ``SimulationContext``:

- :func:`from_openmm` / :func:`from_parmed`: Topologie und Parameter → :class:`SystemSpec`
  (Massen, Ladungen, LJ-Parameter, kovalente Terme inkl. 1-4, Ausschlüsse, Elemente).
- :func:`build_context`: :class:`SystemSpec` → ``SimulationContext`` für n Replicas: LJ-Typen und
  Exclusion-Tiles, optional HMR und H-Bindungs-Constraints, κ_r aus Ionenstärke und T_r (ε_r(T_r)
  von Wasser, OQ-10), Cutoff aus κ (ADR 0006), Box als physikalische Eingabe (OQ-09).
- Helfer: :func:`add_anchors` (Thiol-Anker, OQ-07), :func:`assign_types` (Oberflächen-/Optik-Typen),
  :func:`dipole_probe` (Fluorophor-Gruppe und Dipolachse).

Konventionen: Elektrostatik der 1-4-Paare mit den bereits skalierten Exception-Parametern der
OpenMM-``NonbondedForce`` (AMBER: s_el = 1/1.2, s_LJ = 1/2); alle Exceptions (1-2, 1-3, 1-4) sind
im Non-Bonded-Kernel ausgeschlossen. Einheiten: nm, ps, amu, kJ/mol, e, K, mol/L.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import constraints as _constraints
from . import nonbonded as _nonbonded

BONDED_KEYS = (
    "bonds", "bond_k", "bond_r0", "angles", "angle_k", "angle_theta0", "torsions", "torsion_k",
    "torsion_n", "torsion_phase", "pairs14", "pair_qq", "pair_sigma", "pair_eps",
)  # fmt: skip


@dataclass
class SystemSpec:
    """Ein Molekülsystem (Vorlage für alle Replicas).

    Attributes:
        masses: [amu] (n,); charges: [e] (n,); sigma: [nm] (n,); epsilon: [kJ/mol] (n,).
        bonded: kovalente Terme (Schlüssel :data:`BONDED_KEYS`, lokale Atomindizes).
        excluded: im Non-Bonded-Kernel ausgeschlossene Paare (1-2, 1-3, 1-4), int32 (m, 2).
        elements: Elementsymbole (n,), z. B. für H-Erkennung und Typzuweisung.
        positions: Startpositionen [nm] (n, 3) oder None.
        atom_names, residue_names: optionale Namen (n,) für Atomauswahl.
        gb_radius: intrinsischer GB-Radius ρ [nm] (n,) oder None (ADR 0010/0012, aus
            ``implicit/obc2.xml`` bzw. ``implicit/gbn2.xml``).
        gb_scale: HCT-Skalierungsfaktor s (n,) oder None.
        gb_model: ``"obc2"`` oder ``"gbn2"`` (None ohne GB).
        gb_abg: (α, β, γ) je Atom (n, 3) (GBn2; OBC2: 1, 0.8, 4.85).
        gb_neck_index: Radiusindex je Atom (n,) für die Neck-Tabellen (nur GBn2).
        gb_neck_m0, gb_neck_d0: Neck-Tabellen (U, U) [1/nm] bzw. [nm], Index (Atom i, Atom j).
    """

    masses: np.ndarray
    charges: np.ndarray
    sigma: np.ndarray
    epsilon: np.ndarray
    bonded: dict
    excluded: np.ndarray
    elements: np.ndarray
    positions: np.ndarray | None = None
    atom_names: np.ndarray | None = None
    residue_names: np.ndarray | None = None
    gb_radius: np.ndarray | None = None
    gb_scale: np.ndarray | None = None
    gb_model: str | None = None
    gb_abg: np.ndarray | None = None
    gb_neck_index: np.ndarray | None = None
    gb_neck_m0: np.ndarray | None = None
    gb_neck_d0: np.ndarray | None = None

    @property
    def n_atoms(self) -> int:
        return len(self.masses)

    @property
    def is_hydrogen(self) -> np.ndarray:
        return self.elements == "H"


@dataclass
class BuildInfo:
    """Abgeleitete Größen eines :func:`build_context`-Aufrufs (alle je Replica, falls Array).

    Attributes:
        kappa: κ_r [1/nm]; eps_r: ε_r; r_switch, r_cutoff: [nm] (ADR 0006);
        lj_types: LJ-Typ je Atom (n,); lj_table: (T, T, 2) (σ, ε); masses: verwendete Massen [amu]
        (nach HMR); constrained: True, wenn H-Bindungen constraint sind.
    """

    kappa: np.ndarray
    eps_r: np.ndarray
    r_switch: float
    r_cutoff: float
    lj_types: np.ndarray
    lj_table: np.ndarray
    masses: np.ndarray
    constrained: bool
    extra: dict = field(default_factory=dict)


# --- Extraktion ------------------------------------------------------------------------------


def from_openmm(system, topology=None, positions=None) -> SystemSpec:
    """Parameter eines OpenMM-``System`` (HarmonicBond/Angle, PeriodicTorsion, NonbondedForce).

    Das System muss ohne Constraints erzeugt sein (``constraints=None``, ``rigidWater=False``);
    Constraints legt :func:`build_context` selbst an.

    Args:
        system: ``openmm.System``.
        topology: ``openmm.app.Topology`` für Elemente/Namen (sonst Elemente aus den Massen).
        positions: Positionen (Quantity oder [nm]-Array) oder None.
    """
    import openmm
    from openmm import unit

    if system.getNumConstraints():
        raise ValueError("from_openmm: System ohne Constraints erzeugen (constraints=None)")
    n = system.getNumParticles()
    if any(system.isVirtualSite(i) for i in range(n)):
        raise ValueError("from_openmm: virtuelle Sites werden nicht unterstützt")
    known = {"HarmonicBondForce", "HarmonicAngleForce", "PeriodicTorsionForce", "NonbondedForce",
             "CustomGBForce"}  # fmt: skip
    forces = {}
    for f in system.getForces():
        name = type(f).__name__
        if name == "CMMotionRemover":
            continue  # Schwerpunktbewegung: Langevin-Thermostat, keine Kraft
        if name not in known:
            raise ValueError(f"from_openmm: nicht unterstützte Kraft {name}")
        if name in forces:
            raise ValueError(f"from_openmm: mehrere {name} werden nicht unterstützt")
        forces[name] = f
    nm, kj = unit.nanometer, unit.kilojoule_per_mole

    bonds, bond_r0, bond_k = [], [], []
    if "HarmonicBondForce" in forces:
        hb = forces["HarmonicBondForce"]
        for b in range(hb.getNumBonds()):
            i, j, r0, k = hb.getBondParameters(b)
            bonds.append((i, j))
            bond_r0.append(r0.value_in_unit(nm))
            bond_k.append(k.value_in_unit(kj / nm**2))
    angles, angle_t0, angle_k = [], [], []
    if "HarmonicAngleForce" in forces:
        ha = forces["HarmonicAngleForce"]
        for a in range(ha.getNumAngles()):
            i, j, k_, t0, k = ha.getAngleParameters(a)
            angles.append((i, j, k_))
            angle_t0.append(t0.value_in_unit(unit.radian))
            angle_k.append(k.value_in_unit(kj / unit.radian**2))
    tors, tor_n, tor_ph, tor_k = [], [], [], []
    if "PeriodicTorsionForce" in forces:
        pt = forces["PeriodicTorsionForce"]
        for t in range(pt.getNumTorsions()):
            i, j, k_, l_, per, ph, k = pt.getTorsionParameters(t)
            tors.append((i, j, k_, l_))
            tor_n.append(per)
            tor_ph.append(ph.value_in_unit(unit.radian))
            tor_k.append(k.value_in_unit(kj))

    nb = forces.get("NonbondedForce")
    if nb is None:
        raise ValueError("from_openmm: NonbondedForce fehlt")
    if nb.getNumParticleParameterOffsets() or nb.getNumExceptionParameterOffsets():
        raise ValueError("from_openmm: Parameter-Offsets der NonbondedForce nicht unterstützt")
    q, sig, eps = np.zeros(n), np.zeros(n), np.zeros(n)
    for i in range(n):
        c, s, e = nb.getParticleParameters(i)
        q[i] = c.value_in_unit(unit.elementary_charge)
        sig[i] = s.value_in_unit(nm)
        eps[i] = e.value_in_unit(kj)
    excluded, pairs, qq, psig, peps = [], [], [], [], []
    for e in range(nb.getNumExceptions()):
        i, j, c, s, ep = nb.getExceptionParameters(e)
        excluded.append((i, j))
        c = c.value_in_unit(unit.elementary_charge**2)
        ep = ep.value_in_unit(kj)
        if c == 0.0 and ep == 0.0:
            continue  # reiner Ausschluss (1-2, 1-3)
        pairs.append((i, j))
        qq.append(c)
        psig.append(s.value_in_unit(nm))
        peps.append(ep)
    masses = np.array([system.getParticleMass(i).value_in_unit(unit.dalton) for i in range(n)])
    gb = _gb_parameters(forces["CustomGBForce"], q) if "CustomGBForce" in forces else {}

    def idx(x, arity):
        return np.asarray(x, np.int32).reshape(-1, arity)

    bonded = {
        "bonds": idx(bonds, 2), "bond_k": np.array(bond_k), "bond_r0": np.array(bond_r0),
        "angles": idx(angles, 3), "angle_k": np.array(angle_k), "angle_theta0": np.array(angle_t0),
        "torsions": idx(tors, 4), "torsion_k": np.array(tor_k),
        "torsion_n": np.array(tor_n, np.int32), "torsion_phase": np.array(tor_ph),
        "pairs14": idx(pairs, 2), "pair_qq": np.array(qq), "pair_sigma": np.array(psig),
        "pair_eps": np.array(peps),
    }  # fmt: skip

    if topology is not None:
        atoms = list(topology.atoms())
        if len(atoms) != n:
            raise ValueError("from_openmm: Topologie und System haben unterschiedliche Atomzahlen")
        elements = np.array([a.element.symbol if a.element is not None else "" for a in atoms])
        atom_names = np.array([a.name for a in atoms])
        residue_names = np.array([a.residue.name for a in atoms])
    else:
        # Ohne Topologie: H über die Masse (HMR-Systeme vorher nicht übergeben); alle anderen
        # Atome „X“ – optical_tables braucht dann eine eigene Tabelle.
        elements = np.where(masses < 1.5, "H", "X")
        atom_names = residue_names = None
    if positions is not None:
        if isinstance(positions, unit.Quantity):
            positions = positions.value_in_unit(nm)
        positions = np.asarray(
            [[p[0], p[1], p[2]] if isinstance(p, openmm.Vec3) else p for p in positions],
            np.float64,
        ).reshape(n, 3)
    return SystemSpec(
        masses=masses, charges=q, sigma=sig, epsilon=eps, bonded=bonded,
        excluded=idx(excluded, 2), elements=elements, positions=positions,
        atom_names=atom_names, residue_names=residue_names, **gb,
    )  # fmt: skip


GB_OFFSET = {"obc2": 0.009, "gbn2": 0.0195141}  # nm, Radius-Offset (ADR 0010, ADR 0012)


def _gb_parameters(gb, charges):
    """GB-Parameter aus einer OpenMM-``CustomGBForce`` von ``implicit/obc2.xml`` (charge, or, sr)
    oder ``implicit/gbn2.xml`` (charge, or, sr, alpha, beta, gamma, radindex; Tabellen getm0/getd0).

    Prüft das Modell am Born-Radius-Ausdruck und die Ladungen gegen die NonbondedForce;
    ρ = or + δ, s = sr/or. ε_in, ε_out, κ und der SA-Term des OpenMM-Systems werden nicht
    übernommen – sie kommen aus :func:`build_context`. Die Neck-Tabellen werden so übernommen, wie
    OpenMM sie interpoliert hat (Discrete2DFunction, Wert(x, y) = values[x + U·y]).
    """
    names = [gb.getPerParticleParameterName(k) for k in range(gb.getNumPerParticleParameters())]
    exprs = [gb.getComputedValueParameters(k)[1] for k in range(gb.getNumComputedValues())]
    par = np.array([gb.getParticleParameters(i) for i in range(gb.getNumParticles())], np.float64)
    if par.size and not np.allclose(par[:, 0], charges, rtol=0, atol=1e-12):
        raise ValueError("from_openmm: Ladungen in CustomGBForce und NonbondedForce verschieden")
    if names == ["charge", "or", "sr"] and any("0.8*psi^2+4.85*psi^3" in e for e in exprs):
        n = len(par)
        return {
            "gb_model": "obc2", "gb_radius": par[:, 1] + GB_OFFSET["obc2"],
            "gb_scale": par[:, 2] / par[:, 1], "gb_abg": np.tile([1.0, 0.8, 4.85], (n, 1)),
        }  # fmt: skip
    gbn2_names = ["charge", "or", "sr", "alpha", "beta", "gamma", "radindex"]
    if names == gbn2_names and any("neckScale=0.826836" in e for e in exprs):
        tables = {}
        for k in range(gb.getNumTabulatedFunctions()):
            xs, ys, values = gb.getTabulatedFunction(k).getFunctionParameters()
            tables[gb.getTabulatedFunctionName(k)] = np.asarray(values).reshape(ys, xs).T
        return {
            "gb_model": "gbn2", "gb_radius": par[:, 1] + GB_OFFSET["gbn2"],
            "gb_scale": par[:, 2] / par[:, 1], "gb_abg": par[:, 3:6].copy(),
            "gb_neck_index": par[:, 6].astype(np.int32), "gb_neck_m0": tables["getm0"],
            "gb_neck_d0": tables["getd0"],
        }  # fmt: skip
    raise ValueError(
        f"from_openmm: CustomGBForce {names} ist weder OBC2 noch GBn2 (ADR 0010, ADR 0012)"
    )


def from_parmed(structure) -> SystemSpec:
    """ParmEd-``Structure`` (z. B. aus AMBER prmtop/inpcrd) → :class:`SystemSpec`.

    Erzeugt das OpenMM-System über ``structure.createSystem(nonbondedMethod=NoCutoff,
    constraints=None, rigidWater=False)``; die Skalierung der 1-4-Paare stammt aus der Struktur.
    """
    from openmm import app

    system = structure.createSystem(
        nonbondedMethod=app.NoCutoff, constraints=None, rigidWater=False, removeCMMotion=False
    )
    positions = None
    if structure.coordinates is not None:
        positions = np.asarray(structure.coordinates, np.float64) * 0.1  # Å → nm
    return from_openmm(system, structure.topology, positions)


# --- Kontext ----------------------------------------------------------------------------------


def build_context(
    spec: SystemSpec,
    n_replicas: int,
    box_nm,
    temperatures_k,
    ionic_strength_molar: float,
    *,
    positions=None,
    eps_r=None,
    kappa=None,
    hmr: bool = False,
    h_mass: float = 3.024,
    constraints: bool = False,
    n_kappa: float = 5.0,
    r_cutoff: float | None = None,
    switch_width: float = 0.2,
    types=None,
    solvent: str = "debye-huckel",
    ace: bool = True,
    device: int = 0,
):
    """Erzeugt und füllt einen ``SimulationContext`` (ohne Integrator, Wand, Restraints, Observer).

    Args:
        spec: Systemvorlage, für alle Replicas gleich (ADR 0005: eine Topologie).
        n_replicas: Anzahl Replicas.
        box_nm: (L_x, L_y) [nm] oder (n_replicas, 2); wird nie angepasst (OQ-09).
        temperatures_k: T_r [K], Skalar oder (n_replicas,).
        ionic_strength_molar: Ionenstärke [mol/L] aller nicht explizit modellierten Ionen.
        positions: [nm] (n, 3) oder (n_replicas, n, 3); Default ``spec.positions``.
        eps_r: ε_r je Replica; Default ε_r von Wasser bei T_r (ASSUMPTION(OQ-10)).
        kappa: κ_r [1/nm] je Replica explizit; Default aus Ionenstärke, T_r und ε_r.
        hmr: Hydrogen Mass Repartitioning auf ``h_mass`` [amu].
        constraints: H-Bindungen per SHAKE/RATTLE festhalten (für Δt = 4 fs mit HMR).
        n_kappa: r_c = max(2 nm, n_κ/min κ_r) (ADR 0006); ``r_cutoff`` überschreibt r_c;
            r_s = r_c − ``switch_width`` [nm].
        types: (n,) Oberflächen-/Optik-Typ je Atom (``types[..., 1]``). Default 1 für alle Atome
            (Typ 0 = keine Wand, nur für Anker, ADR 0007); siehe :func:`assign_types`.
        solvent: ``"debye-huckel"`` (V1), ``"gb-obc2"`` oder ``"gb-gbn2"`` (ADR 0010/0012; das Modell
            muss zu ``spec.gb_model`` passen; braucht ``spec.gb_radius``
            aus einem System mit ``implicit/obc2.xml``). Bei GB ist ε_r das Lösungsmittel-ε_out,
            Coulomb (auch 1-4) wird mit ε_in = 1 unabgeschirmt gerechnet; die Abschirmung steckt im
            GB-Term. ``ace``: unpolarer ACE-Term (nur GB).

    Returns:
        (ctx, :class:`BuildInfo`).
    """
    from . import _core

    n = spec.n_atoms
    temps = np.broadcast_to(np.asarray(temperatures_k, np.float64), (n_replicas,)).copy()
    eps = (
        _nonbonded.water_permittivity(temps)
        if eps_r is None
        else np.broadcast_to(np.asarray(eps_r, np.float64), (n_replicas,)).copy()
    )
    if kappa is None:
        kappa = np.atleast_1d(_nonbonded.debye_kappa(ionic_strength_molar, temps, eps))
    kappa = np.broadcast_to(np.asarray(kappa, np.float64), (n_replicas,)).copy()
    if r_cutoff is None:
        r_s, r_c = _nonbonded.cutoff_from_kappa(kappa, n_kappa=n_kappa, width=switch_width)
    else:
        r_s, r_c = r_cutoff - switch_width, float(r_cutoff)

    masses = np.asarray(spec.masses, np.float64)
    if hmr:
        masses = _constraints.repartition_hydrogen_mass(
            masses, spec.bonded["bonds"], spec.is_hydrogen, h_mass
        )
    bonded = {k: np.asarray(spec.bonded[k]) for k in BONDED_KEYS}
    cons = None
    if constraints:
        cons, bonded = _constraints.constraint_topology(bonded, spec.is_hydrogen)

    pos = spec.positions if positions is None else positions
    if pos is None:
        raise ValueError("build_context: keine Positionen")
    pos = np.broadcast_to(np.asarray(pos, np.float64), (n_replicas, n, 3))
    box = np.broadcast_to(np.asarray(box_nm, np.float64), (n_replicas, 2))

    ctx = _core.SimulationContext(n_replicas, n, device)
    ctx.set_box(box.astype(np.float32))
    ctx.set_masses(np.broadcast_to(masses, (n_replicas, n)).astype(np.float32))
    ctx.set_positions(pos.astype(np.float32))
    ctx.set_charges(np.broadcast_to(spec.charges, (n_replicas, n)).astype(np.float32))
    lj_types, nbt = _nonbonded.nonbonded_topology(
        ctx.n_pad, spec.sigma, spec.epsilon, spec.excluded
    )
    t2 = np.zeros((n_replicas, n, 2), np.int32)
    t2[..., 0] = lj_types
    t2[..., 1] = 1 if types is None else np.asarray(types, np.int32)
    ctx.set_types(t2)
    ctx.set_temperatures(temps.astype(np.float32))
    k32, e32 = kappa.astype(np.float32), eps.astype(np.float32)
    zeros = np.zeros(n_replicas, np.int32)
    if solvent == "debye-huckel":
        ctx.add_bonded([bonded], zeros, k32, e32)
        ctx.set_nonbonded_tiled([nbt], zeros, k32, e32, r_switch=r_s, r_cutoff=r_c)
    elif solvent in ("gb-obc2", "gb-gbn2"):
        model = solvent[3:]
        spec_model = spec.gb_model or ("obc2" if spec.gb_radius is not None else None)
        if spec_model != model:
            raise ValueError(
                f"build_context: solvent={solvent!r}, aber GB-Parameter für {spec_model!r} "
                f"(System mit implicit/{model}.xml erzeugen)"
            )
        ones = np.ones(n_replicas, np.float32)
        ctx.add_bonded(  # 1-4: ε_in = 1, unabgeschirmt, auch bei Replica Exchange
            [bonded], zeros, np.zeros(n_replicas, np.float32), ones, follow_screening=False
        )
        ctx.set_nonbonded_gb(
            [nbt], zeros, k32, e32, [spec.gb_radius], [spec.gb_scale], r_switch=r_s,
            r_cutoff=r_c, ace=ace, **_gb_model_kwargs(spec),
        )  # fmt: skip
    else:
        raise ValueError("solvent: 'debye-huckel', 'gb-obc2' oder 'gb-gbn2'")
    if cons is not None:
        ctx.set_constraints([cons], zeros)
    info = BuildInfo(
        kappa=kappa, eps_r=eps, r_switch=float(r_s), r_cutoff=float(r_c), lj_types=lj_types,
        lj_table=nbt["lj_table"], masses=masses, constrained=cons is not None,
    )  # fmt: skip
    return ctx, info


def _gb_model_kwargs(spec: SystemSpec) -> dict:
    """Zusatzargumente für ``set_nonbonded_gb`` je Modell."""
    if spec.gb_model in (None, "obc2"):  # handgebaute Specs ohne gb_model: OBC2
        return {}
    return {
        "offset": GB_OFFSET["gbn2"], "alpha_beta_gamma": [spec.gb_abg],
        "neck_index": [spec.gb_neck_index], "neck_m0": [spec.gb_neck_m0],
        "neck_d0": [spec.gb_neck_d0],
    }  # fmt: skip


# --- Helfer -------------------------------------------------------------------------------------


def add_anchors(ctx, anchor_atoms, height_nm: float, z_top_nm: float = 0.0, k_z: float = 1.0e4,
                xy=None, k_xy: float = 10.0):  # fmt: skip
    """Thiol-Anker (OQ-07): z-Restraint des Ankeratoms auf z_top + height (Chemisorptionshöhe über
    der obersten Atomlage, ADR 0007 Punkt 5). Das Ankeratom muss Oberflächen-Typ 0 haben (keine
    Wandwechselwirkung, siehe :func:`assign_types`).

    Args:
        anchor_atoms: lokale Atomindizes (m,), in jeder Replica gleich.
        height_nm: Höhe über z_top [nm] (nicht über z = 0); ASSUMPTION(OQ-07): Literatur
            uneinheitlich, typ. 0.2–0.25 für Thiol auf Au(111).
        z_top_nm: Höhe der obersten Atomlage über z = 0 der Wand [nm] (``wall_table(...)['z_top']``).
        k_z: Federkonstante in z [kJ/mol/nm²].
        xy: optional (m, 2) Ankerpositionen in x/y [nm] (nur bei mehreren Rezeptoren pro Box);
            dann zusätzlich ein xy-Restraint mit ``k_xy`` [kJ/mol/nm²].

    Returns:
        Liste der erzeugten Restraint-Plugins.
    """
    atoms = np.asarray(anchor_atoms, np.int32).ravel()
    n_rep = ctx.n_replicas
    rep = np.repeat(np.arange(n_rep, dtype=np.int32), len(atoms))
    atom = np.tile(atoms, n_rep)
    ref = np.zeros((len(atom), 3), np.float32)
    ref[:, 2] = z_top_nm + height_nm
    plugins = [
        ctx.add_position_restraints(
            rep, atom, ref, np.full(len(atom), k_z, np.float32), dims=(False, False, True)
        )
    ]
    if xy is not None:
        ref_xy = np.zeros_like(ref)
        ref_xy[:, :2] = np.tile(np.asarray(xy, np.float32).reshape(-1, 2), (n_rep, 1))
        plugins.append(
            ctx.add_position_restraints(
                rep, atom, ref_xy, np.full(len(atom), k_xy, np.float32), dims=(True, True, False)
            )
        )
    return plugins


def assign_types(spec: SystemSpec, classes, anchor_atoms=()):
    """Oberflächen-/Optik-Typ je Atom (``types[..., 1]``) aus einer Klassenzuordnung.

    ``d_types.y`` ist gleichzeitig Oberflächen-Typ (Wandtabelle, λ pro Typ) und Optik-Typ (α′, v,
    Voxel-Skalierung). Typ 0 ist reserviert: keine Wand (ADR 0007) – für Ankeratome (OQ-07).

    Args:
        classes: dict {Typ ≥ 1: bool-Maske (n,) oder Indexliste}; spätere Einträge überschreiben
            frühere. Nicht zugeordnete Atome erhalten Typ 1.
        anchor_atoms: Atome mit Typ 0.

    Returns:
        int32 (n,).
    """
    t = np.ones(spec.n_atoms, np.int32)
    for typ, sel in classes.items():
        if typ < 1:
            raise ValueError("assign_types: Typ 0 ist für Ankeratome reserviert")
        sel = np.asarray(sel)
        mask = sel if sel.dtype == bool else np.isin(np.arange(spec.n_atoms), sel)
        t[mask] = typ
    t[np.asarray(anchor_atoms, np.int64)] = 0
    return t


def dipole_probe(groups, axis):
    """CSR-Eingabe für ``ctx.add_dipole_tracker`` / ``add_fret_tracker``.

    Args:
        groups: je Replica die Atomindizes der Fluorophor-Gruppe (Schwerpunkt R_F), Liste von (k,).
        axis: je Replica (atom_a, atom_b) der Übergangsdipolachse μ̂ = (r_b − r_a)/|r_b − r_a|.
            ASSUMPTION(OQ-03): Achsenatome pro Farbstoff (AMBER-DYES) noch nicht tabelliert – vom
            Aufrufer anzugeben.

    Returns:
        (offsets int32 (n_rep + 1,), atoms int32, atom_a int32 (n_rep,), atom_b int32 (n_rep,)).
    """
    offsets = np.concatenate([[0], np.cumsum([len(g) for g in groups])]).astype(np.int32)
    atoms = np.concatenate([np.asarray(g, np.int32) for g in groups])
    ax = np.asarray(axis, np.int32).reshape(-1, 2)
    return offsets, atoms, ax[:, 0].copy(), ax[:, 1].copy()


# ASSUMPTION(OQ-01/02): Platzhalter je Element, nicht kalibriert (Entscheidung: Typtabelle nach
# Miller mit Skalierungsfaktor, kalibriert an dn/dc; Volumina aus ProtOr/NucProt). Hier: α′ grob nach
# Miller (1990, J. Am. Chem. Soc. 112, 8533), v aus Bondi-Radien (Überlappung ignoriert), beide
# unverifiziert. Einheiten nm³.
ELEMENT_OPTICS = {
    "H": (0.387e-3, 4.0 / 3.0 * np.pi * 0.120**3),
    "C": (1.20e-3, 4.0 / 3.0 * np.pi * 0.170**3),
    "N": (1.09e-3, 4.0 / 3.0 * np.pi * 0.155**3),
    "O": (0.64e-3, 4.0 / 3.0 * np.pi * 0.152**3),
    "P": (3.0e-3, 4.0 / 3.0 * np.pi * 0.180**3),
    "S": (3.0e-3, 4.0 / 3.0 * np.pi * 0.180**3),
    "Na": (0.15e-3, 4.0 / 3.0 * np.pi * 0.227**3),
    "Mg": (0.08e-3, 4.0 / 3.0 * np.pi * 0.173**3),
}


def optical_tables(spec: SystemSpec, anchor_atoms=(), table=None):
    """Optik-/Oberflächen-Typ je Atom nach Element und die Tabellen α′, v je Typ.

    Typ 0 ist das Ankeratom-Element (keine Wand, ADR 0007); alle Anker müssen dasselbe Element
    haben. Typen ≥ 1 entsprechen den übrigen Elementen in der Reihenfolge des ersten Auftretens.

    Returns:
        (types int32 (n,), alpha float32 (T,) [nm³], volume float32 (T,) [nm³], names (T,)).
    """
    table = ELEMENT_OPTICS if table is None else table
    anchors = np.asarray(anchor_atoms, np.int64)
    missing = sorted(set(spec.elements) - set(table))
    if missing:
        raise ValueError(f"optical_tables: keine Parameter für {missing}")
    names = [""]
    if len(anchors):
        el = set(spec.elements[anchors])
        if len(el) != 1:
            raise ValueError("optical_tables: alle Ankeratome brauchen dasselbe Element")
        names[0] = el.pop()
    types = np.zeros(spec.n_atoms, np.int32)
    for e in dict.fromkeys(spec.elements):
        names.append(str(e))
        types[spec.elements == e] = len(names) - 1
    types[anchors] = 0
    alpha = np.array([table[e][0] if e else 0.0 for e in names], np.float32)
    volume = np.array([table[e][1] if e else 0.0 for e in names], np.float32)
    return types, alpha, volume, np.array(names)
