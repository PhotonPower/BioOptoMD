"""Aptamerschicht über Gold im Kretschmann-Aufbau mit selbstkonsistenter Optik (ROADMAP Phase 10.4).

Ablauf: DNA-Strang (wie examples/aptamer_on_graphene, AMBER OL15) → biooptomd.builder → Gold-Wand
(9-3, ADR 0007) + Bildladungen (ADR 0013) + Anker (OQ-07) → Replicas bei derselben Temperatur
(unabhängige Proben für ⟨ε⟩; kein Replica Exchange, ADR 0016 Punkt 5) → SelfConsistentOptics mit
dem TMM-Referenzsolver: BK7-Prisma | Gold 50 nm | Schicht ⟨ε⟩(z) | Wasser, p-polarisiert nahe der
Plasmonresonanz (ADR 0016) → Verlauf der Iteration, W(z), ε(z) und die Verschiebung der
Reflektivität durch die Schicht (Sensorsignal).

Platzhalter (klar markiert, siehe docs/OPEN_QUESTIONS.md):
- ASSUMPTION(OQ-03, OQ-07): Strang und Anker wie im Graphen-Beispiel; Bedeckung 1 Strang je Box.
- ASSUMPTION(OQ-25): ε_Au(633 nm) = −11.6 + 1.2i; Prisma n = 1.515 (BK7).
- ASSUMPTION(OQ-05): Gold-Wand mit λ = 1.
- Bei 1 MW/cm² sind die optischen Kräfte klein (ADR 0016, Größenordnung); die Rückwirkung der
  Schicht auf das Feld (δW, δR) ist das Ergebnis. ``--intensity`` erhöht die Eingangsintensität.

Aufruf:  python examples/aptamer_spr_gold/aptamer_spr_gold.py --iterations 6 --out spr_run
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

from biooptomd import builder, coupling, surface
from biooptomd.voxel import optical_scale

HERE = Path(__file__).resolve().parent
LAMBDA_NM = 632.8
N_PRISM = 1.515
EPS_GOLD = -11.6 + 1.2j  # ASSUMPTION(OQ-25)
D_GOLD_NM = 50.0
ANGLE_DEG = 72.6  # knapp über dem SPR-Minimum (72.24° ohne Schicht)
DT_PS = 0.004
FRICTION = 1.0


def _graphene_example():
    path = HERE.parent / "aptamer_on_graphene" / "aptamer_on_graphene.py"
    spec_ = importlib.util.spec_from_file_location("aptamer_on_graphene", path)
    mod = importlib.util.module_from_spec(spec_)
    spec_.loader.exec_module(mod)
    return mod


def build(n_replicas: int = 4, temperature: float = 300.0, seed: int = 2026, *,
          intensity_w_m2: float = 1e10, n_equil: int = 2500, n_sample: int = 10000,
          stride: int = 50, beta: float = 0.5, checkpoint_dir=None):  # fmt: skip
    """Kontext und Treiber. Returns (ctx, driver, meta)."""
    ex = _graphene_example()
    spec, anchor, _, _, anchor_z = ex.load_strand()
    box = ex.BOX_NM
    n_water = builder.OPTICS_TARGETS["dna"]["n_water"]
    dna_optics, _ = builder.calibrate_optics("dna")
    types, alpha, volume, names = builder.optical_tables(
        spec, anchor_atoms=[anchor], table=dna_optics
    )
    temps = np.full(n_replicas, temperature)
    ctx, info = builder.build_context(
        spec, n_replicas, box, temps, ex.IONIC_STRENGTH, hmr=True, constraints=True, types=types
    )
    wall = surface.wall_table("gold", info.lj_table, np.ones(len(names)))
    ctx.add_planar_surface(wall["prefactor"], wall["sigma"], form=wall["form"], delta=wall["delta"])
    builder.add_image_charges(ctx, info, "gold")
    builder.add_anchors(ctx, [anchor], height_nm=anchor_z - wall["z_top"], z_top_nm=wall["z_top"])
    ctx.set_integrator(DT_PS, FRICTION, seed)

    # Gitter: lateral periodisch über die Box; ε lateral gemittelt (TMM), Δz = 0.1 nm bis 8 nm.
    z_surface = float(wall["z_top"])
    nz_eps, dz_eps = 80, 0.1
    force_grid = coupling.ForceGrid(
        (12, 12, 37), (0.0, 0.0, z_surface), (box[0] / 12, box[1] / 12, 0.25), periodic_xy=True
    )
    solver = coupling.TmmSolver(
        LAMBDA_NM, ANGLE_DEG, "p", N_PRISM, [(EPS_GOLD, D_GOLD_NM)],
        intensity_w_m2=intensity_w_m2, z_surface=z_surface,
    )  # fmt: skip
    scale = optical_scale(alpha, volume, n_water, amplitude="peak")
    driver = coupling.SelfConsistentOptics(
        ctx, solver, force_grid, scale, eps_shape=(1, 1, nz_eps), eps_origin=(0.0, 0.0, z_surface),
        eps_spacing=(box[0], box[1], dz_eps), alpha=alpha, volume=volume, n_water=n_water,
        n_equil=n_equil, n_sample=n_sample, stride=stride, beta=beta, tol=0.1, tol_w=2e-3,
        checkpoint_dir=checkpoint_dir,
    )  # fmt: skip
    meta = {"solver": solver, "z_surface": z_surface, "n_water": n_water, "names": names}
    return ctx, driver, meta


def reflectivity(solver: coupling.TmmSolver, eps: coupling.EpsGrid) -> float:
    e, d, _ = solver.stack(eps)
    return coupling.tmm(e, d, solver.wavelength_nm, np.deg2rad(solver.angle_deg), solver.pol).R


def summarize(driver, meta) -> dict:
    """Verlauf, Profile und Sensorsignal δR = R(Schicht) − R(Wasser) beim Arbeitswinkel."""
    eps = coupling.EpsGrid(driver.eps, driver.eps_origin, driver.eps_spacing, meta["n_water"])
    water = coupling.EpsGrid(np.full_like(driver.eps, meta["n_water"] ** 2), driver.eps_origin,
                             driver.eps_spacing, meta["n_water"])  # fmt: skip
    z_eps = eps.centers(2)
    return {
        "history": driver.history,
        "z_eps": z_eps,
        "eps_z": driver.eps.mean(axis=(1, 2)),
        "z_w": driver.grid.nodes(2),
        "w_z": driver.w[:, 0, 0],
        "w0_z": driver.w_water[:, 0, 0],
        "R_layer": reflectivity(meta["solver"], eps),
        "R_water": reflectivity(meta["solver"], water),
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--iterations", type=int, default=6)
    p.add_argument("--replicas", type=int, default=4)
    p.add_argument("--intensity", type=float, default=1e10, help="W/m² im Prisma (1e10 = 1 MW/cm²)")
    p.add_argument("--sample", type=int, default=10000, help="Messschritte je Iteration (4 fs)")
    p.add_argument("--out", type=Path, default=Path("aptamer_spr_gold_out"))
    args = p.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    ctx, driver, meta = build(args.replicas, intensity_w_m2=args.intensity, n_sample=args.sample,
                              checkpoint_dir=args.out / "checkpoints")  # fmt: skip
    driver.run(max_iter=args.iterations)
    res = summarize(driver, meta)
    for h in res["history"]:
        print(f"Iteration {h['iteration']}: Δε = {h['d_eps']}, ΔW = {h['d_w']:.2e}, "
              f"Rauschen ε {h['eps_noise']:.3f}, Selbstfeld {h['self_field']:.3f}")  # fmt: skip
    print(
        f"R(Wasser) = {res['R_water']:.5f}, R(Schicht) = {res['R_layer']:.5f}, "
        f"δR = {res['R_layer'] - res['R_water']:+.2e}; konvergiert: {driver.converged()}"
    )
    np.savez(args.out / "profiles.npz", **{k: v for k, v in res.items() if k != "history"})
    (args.out / "history.json").write_text(json.dumps(res["history"], indent=1), encoding="utf-8")
    print(f"Status {ctx.status()}, Ausgabe in {args.out}")


if __name__ == "__main__":
    main()
