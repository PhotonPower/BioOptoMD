"""Builder-Hilfen für PlanarSurfaceForce (ROADMAP Phase 2, docs/PHYSICS_MODEL.md §4.1, ADR 0007).

- :data:`MATERIALS`: Wandatom-Parameter und Wandform pro Material.
- :func:`wall_table`: Vorfaktoren C und σ_iw pro (Oberflächen-Typ, LJ-Typ) aus Lorentz-Berthelot,
  mit Lösungsmittel-Skalierung λ pro Oberflächen-Typ. Oberflächen-Typ 0 ist reserviert (ε = 0) für
  Padding-Atome und Thiol-Ankeratome (OQ-07).
- :func:`volts_to_kj_mol_e`: Umrechnung von Φ0 für den Gouy-Chapman-Term.

Wandformen (C enthält Dichte, ε_iw und σ_iw; Formeln in ADR 0007):

    9-3     C = (2π/3) ρ ε_iw σ_iw³        ρ   Atomdichte [nm⁻³]
    10-4    C = 2π ρ_s ε_iw σ_iw²          ρ_s Flächendichte [nm⁻²]
    10-4-3  C = 2π ρ ε_iw σ_iw² Δ          ρ   Atomdichte [nm⁻³], Δ Lagenabstand [nm]

Einheiten: σ in nm, ε in kJ/mol, Dichten in nm⁻³ bzw. nm⁻², Potentiale in kJ/(mol·e).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import units

WALL_FORMS = ("9-3", "10-4", "10-4-3")


@dataclass(frozen=True)
class WallMaterial:
    """Parameter eines Wandmaterials.

    Attributes:
        form: Wandform ('9-3', '10-4', '10-4-3').
        sigma_nm: σ_w des Wandatoms [nm].
        epsilon_kj_mol: ε_w des Wandatoms [kJ/mol].
        density: ρ [nm⁻³] für '9-3' und '10-4-3', ρ_s [nm⁻²] für '10-4'.
        delta_nm: Lagenabstand Δ [nm] ('10-4-3'; für '9-3' der Lagenabstand, der z_top festlegt).
        z_top_nm: Höhe der obersten Atomlage über der Kontinuumsgrenze z = 0 der Wand [nm]
            (ADR 0007 Punkt 5). Höhenangaben (Chemisorption, SAM, Observer-Ursprung) beziehen sich
            auf z_top, nicht auf z = 0.
        source: Literaturquelle der Zahlenwerte.
    """

    form: str
    sigma_nm: float
    epsilon_kj_mol: float
    density: float
    delta_nm: float
    z_top_nm: float
    source: str


# ASSUMPTION(OQ-05): Wandatom-Parameter vorläufig, an den Primärtabellen nicht verifiziert
# (Checkliste in docs/OPEN_QUESTIONS.md). Steele: σ_C = 0.340 nm, ε_C/k_B = 28 K, Δ = 0.335 nm,
# ρ_s = 38.2 nm⁻² (ρ = ρ_s/Δ). Gold: INTERFACE-FF 12-6 (σ = 0.2629 nm, ε = 5.29 kcal/mol),
# fcc a = 0.40782 nm ⇒ ρ = 4/a³, Au(111)-Lagenabstand a/√3.
_AU_A = 0.40782
_AU_D111 = _AU_A / np.sqrt(3.0)
MATERIALS: dict[str, WallMaterial] = {
    "graphene": WallMaterial(
        "10-4", 0.340, 28.0 * units.KB, 38.2, 0.335, 0.0, "Steele, Surf. Sci. 36, 317 (1973)"
    ),
    "graphite": WallMaterial(
        "10-4-3", 0.340, 28.0 * units.KB, 38.2 / 0.335, 0.335, 0.0, "Steele (1973)"
    ),
    "gold": WallMaterial(
        "9-3",
        0.2629,
        5.29 * 4.184,
        4.0 / _AU_A**3,
        _AU_D111,
        0.5 * _AU_D111,
        "Heinz et al., J. Phys. Chem. C 112, 17281 (2008)",
    ),
}


def wall_prefactor(form: str, epsilon_iw, sigma_iw, density: float, delta_nm: float = 0.335):
    """Vorfaktor C [kJ/mol] der Wandform aus ε_iw [kJ/mol], σ_iw [nm] und der Dichte (s. o.)."""
    eps = np.asarray(epsilon_iw, np.float64)
    sig = np.asarray(sigma_iw, np.float64)
    if form == "9-3":
        return 2.0 * np.pi / 3.0 * density * eps * sig**3
    if form == "10-4":
        return 2.0 * np.pi * density * eps * sig**2
    if form == "10-4-3":
        return 2.0 * np.pi * density * eps * sig**2 * delta_nm
    raise ValueError(f"form: eines von {WALL_FORMS}")


def wall_table(material, lj_table, solvent_scale):
    """Parametertabelle der Wand pro (Oberflächen-Typ, LJ-Typ) nach Lorentz-Berthelot (ADR 0007).

    Args:
        material: Name aus :data:`MATERIALS` oder ein :class:`WallMaterial`.
        lj_table: LJ-Paartabelle (n_lj, n_lj, 2) aus :func:`biooptomd.nonbonded.lj_type_table`
            (Diagonale = (σ_l, ε_l)), oder ein Paar (σ_l [nm], ε_l [kJ/mol]) mit Shape (n_lj,).
        solvent_scale: λ pro Oberflächen-Typ, (n_surface,), λ ∈ (0, 1] für Typen ≥ 1. Typ 0 ist
            reserviert und erhält unabhängig vom Wert ε = 0. ASSUMPTION(OQ-05): ohne Kalibrierung
            λ = 1; für Gold mit INTERFACE-FF-Parametern ist λ < 1 praktisch Pflicht.

    Returns:
        dict mit ``prefactor`` [kJ/mol] und ``sigma`` [nm] (float32, (n_surface, n_lj)), ``form``,
        ``delta`` [nm] und ``z_top`` [nm] – direkt verwendbar als
        ``ctx.add_planar_surface(t["prefactor"], t["sigma"], form=t["form"], delta=t["delta"])``.
    """
    mat = MATERIALS[material] if isinstance(material, str) else material
    if isinstance(lj_table, np.ndarray) and lj_table.ndim == 3:
        sigma_l = np.diagonal(lj_table[..., 0]).astype(np.float64)
        eps_l = np.diagonal(lj_table[..., 1]).astype(np.float64)
    else:
        sigma_l, eps_l = (np.asarray(a, np.float64) for a in lj_table)
    if sigma_l.ndim != 1 or sigma_l.shape != eps_l.shape:
        raise ValueError("lj_table: (n_lj, n_lj, 2) oder (σ, ε) mit Shape (n_lj,)")
    if (sigma_l < 0).any() or (eps_l < 0).any():
        raise ValueError("LJ-Parameter müssen >= 0 sein")
    lam = np.asarray(solvent_scale, np.float64)
    if lam.ndim != 1 or lam.size < 1:
        raise ValueError("solvent_scale: 1D-Array (n_surface,)")
    if ((lam[1:] <= 0.0) | (lam[1:] > 1.0)).any():
        raise ValueError("solvent_scale: λ ∈ (0, 1] für Oberflächen-Typen >= 1")
    sigma_iw = 0.5 * (sigma_l + mat.sigma_nm)
    eps_iw = np.sqrt(eps_l * mat.epsilon_kj_mol)
    c = wall_prefactor(mat.form, eps_iw, sigma_iw, mat.density, mat.delta_nm)
    lam = lam.copy()
    lam[0] = 0.0  # Oberflächen-Typ 0: keine Wandwechselwirkung (Padding, Ankeratom)
    prefactor = (lam[:, None] * c[None, :]).astype(np.float32)
    sigma = np.broadcast_to(sigma_iw, prefactor.shape).astype(np.float32)
    return {
        "prefactor": prefactor,
        "sigma": sigma,
        "form": mat.form,
        "delta": float(mat.delta_nm),
        "z_top": float(mat.z_top_nm),
    }


def volts_to_kj_mol_e(phi_volts):
    """Potential [V] → [kJ/(mol·e)] (× FARADAY). Für Gouy-Chapman ist Φ0 das Potential der
    diffusen Schicht, nicht das angelegte Elektrodenpotential (Stern-Schicht nicht modelliert)."""
    return np.asarray(phi_volts, np.float64) * units.FARADAY
