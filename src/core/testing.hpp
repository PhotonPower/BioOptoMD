// BioOptoMD – Test-Hooks (nur für pytest, als biooptomd._core.testing exportiert).
// Keine Produktionsphysik: Diese Funktionen existieren, damit Kern-Infrastruktur ohne
// echte Kraftterme geprüft werden kann.
#pragma once

#include <memory>

#include "biooptomd/plugin.hpp"
#include "biooptomd/simulation_context.hpp"

namespace bomd::testing {

/// Dummy-Kernel: x += dx_nm für alle echten Atome (Test test_roundtrip_numpy_and_dlpack).
void shift_x(SimulationContext& ctx, float dx_nm);

/// Addiert m Kraftbeiträge per Fixed-Point-Atomics auf d_force_fp, ohne vorher zu löschen.
/// @param atom_global  globale Atomindizes r·n_pad + i, Shape (m,)
/// @param f_kj_mol_nm  Kräfte [kJ/mol/nm], Shape (m, 3)
/// Ein Thread pro Beitrag; die Reihenfolge der Atomics hängt von der Reihenfolge der
/// Beiträge und vom Scheduling ab.
void accumulate_forces(SimulationContext& ctx, const int* atom_global, const float* f_kj_mol_nm,
                       int m);

/// Test-ForcePlugin: konstante Kraft f [kJ/mol/nm] auf jedes echte Atom, keine Energie.
std::unique_ptr<ForcePlugin> make_uniform_force(float fx, float fy, float fz);

}  // namespace bomd::testing
