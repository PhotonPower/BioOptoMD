// BioOptoMD – PlanarSurfaceForce: Wand (9-3 / 10-4 / 10-4-3, ADR 0007) + optional Gouy-Chapman
// bei z = z0 (docs/PHYSICS_MODEL.md §4.1). NumPy-Referenz: tests/reference/surface_ref.py.
#pragma once

#include <cstdint>
#include <string_view>
#include <vector>

#include "biooptomd/device_buffer.hpp"
#include "biooptomd/plugin.hpp"

namespace bomd {

/// Wandform (ADR 0007). Der Kernel ist ein Template über die Form; die Auswahl erfolgt einmal pro
/// Launch auf dem Host (kein virtueller Device-Code, AGENTS.md Regel 9).
enum class WallForm : int {
  kWall93 = 0,    ///< Halbraum (Volumenmaterial, z. B. Gold)
  kWall104 = 1,   ///< eine Atomebene (Graphen-Monolage)
  kWall1043 = 2,  ///< Steele, Lagenstapel (Graphit)
};

struct PlanarSurfaceParams {
  WallForm form = WallForm::kWall93;
  /// Anzahl LJ-Typen der Tabelle (>= 1). Bei 1 hängt die Wand nur vom Oberflächen-Typ ab.
  int n_lj_types = 1;
  /// Vorfaktor C pro (Oberflächen-Typ s = d_types.y, LJ-Typ l = d_types.x), Index
  /// s·n_lj_types + l [kJ/mol], >= 0; C = 0 ⇒ keine Wand. Zeile s = 0 muss 0 sein (reserviert für
  /// Padding- und Ankeratome, ADR 0007). Formeln: PHYSICS_MODEL.md §4.1.
  std::vector<float> prefactor_kj_mol;
  /// σ_iw pro Eintrag wie prefactor_kj_mol [nm], >= 0; σ = 0 ⇒ keine Wand.
  std::vector<float> sigma_nm;
  float delta_nm = 0.335f;    ///< Lagenabstand Δ (nur kWall1043) [nm], > 0
  float z0_nm = 0.0f;         ///< Lage der Wand (Kontinuumsgrenze, ADR 0007 Punkt 5) [nm]
  float clamp_factor = 0.1f;  ///< ζ_c >= clamp_factor·σ (> 0)
  /// Kraftschranke am Clamp (ADR 0008): ζ_c >= ζ_F mit |F_wall(ζ_F)| = max_force [kJ/mol/nm].
  float max_force_kj_mol_nm = 1.0e6f;
  /// Φ0 der diffusen Schicht [kJ/(mol·e)], 0 ⇒ kein Gouy-Chapman-Term (OQ-06).
  float phi0_kj_mol_e = 0.0f;
  /// Inverse Debye-Länge κ_r je Replica [1/nm], >= 0 (OQ-10); leer nur ohne Gouy-Chapman.
  std::vector<float> kappa_per_nm;
  /// true ⇒ linearisiertes Gouy-Chapman (nur als Testgrenzfall, OQ-06).
  bool gouy_chapman_linear = false;
};

class PlanarSurfaceForce final : public ForcePlugin {
 public:
  explicit PlanarSurfaceForce(PlanarSurfaceParams params);

  std::string_view name() const override { return "PlanarSurfaceForce"; }
  void initialize(SimulationContext& ctx) override;
  /// Kräfte in d_force_fp, Energie in Term kEnergySurface. Wirft std::logic_error, wenn das
  /// nichtlineare Gouy-Chapman aktiv ist und eine Replica T_r <= 0 hat.
  void compute(DeviceState& s, cudaStream_t stream) override;

  const PlanarSurfaceParams& params() const { return params_; }
  /// ζ_c = max(clamp_factor·σ, ζ_F) pro Tabelleneintrag [nm] (0 für Einträge ohne Wand).
  const std::vector<float>& clamp_distances() const { return zeta_c_; }
  /// Anzahl Kraftauswertungen mit geclamptem ζ seit dem letzten reset_counters() (synchronisiert).
  std::uint64_t clamp_count() const;
  /// Anzahl Auswertungen mit Oberflächen- oder LJ-Typ außerhalb der Tabelle (keine Wand).
  std::uint64_t unknown_type_count() const;
  void reset_counters();

 private:
  std::uint64_t counter(int index) const;

  PlanarSurfaceParams params_;
  int n_surface_types_ = 0;
  SimulationContext* ctx_ = nullptr;
  std::vector<float> zeta_c_;
  DeviceBuffer<float4> type_params_;  // (C, σ, ζ_c, B) pro Eintrag, B = Cσ⁴/(3Δ) für 10-4-3
  DeviceBuffer<float> kappa_;         // κ_r je Replica
  DeviceBuffer<unsigned long long> counters_;  // [clamp, unknown_type]
};

}  // namespace bomd
