// BioOptoMD – SimulationContext: besitzt alle Gerätepuffer, den CUDA-Stream und die Plugins
// einer Batch-Simulation (docs/ARCHITECTURE.md §2–3).
#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "biooptomd/cuda_stream.hpp"
#include "biooptomd/device_buffer.hpp"
#include "biooptomd/device_state.hpp"
#include "biooptomd/langevin.hpp"
#include "biooptomd/plugin.hpp"
#include "core/constraints.hpp"

namespace bomd {

/// RAII-Besitzer des Simulationszustands. Nicht kopier- und nicht verschiebbar, weil Plugins
/// Referenzen auf den Kontext halten dürfen.
///
/// Nach der Konstruktion sind alle Arrays mit 0 initialisiert: Positionen 0 nm, q = 0 e,
/// Geschwindigkeiten 0 nm/ps, inv_mass = 0 1/amu (unbeweglich), Typen 0, Box (0, 0) = nicht
/// gesetzt, T = 0 K. Padding-Atome (i_local >= n_atoms) behalten diese Werte dauerhaft;
/// alle Setter schreiben nur echte Atome (AGENTS.md Regel 6).
///
/// Wertebereich: |x|, |y| < kMaxBoxImages·L (sonst ist wrap_xy in FP32 nicht mehr exakt
/// genug); set_positions/set_box prüfen das für Eingaben.
///
/// Alle Kernels laufen auf stream(). Die Host-Setter synchronisieren den Stream am Ende
/// (nicht im Hot-Path); die Kernel-Methoden (clear_forces, wrap_xy, …) synchronisieren nicht.
/// Dynamischer Zustand für bitgenaue Checkpoints (ROADMAP Phase 7): alle Atomplätze inkl.
/// Padding (n_replicas·n_pad), Schrittzähler (= Philox-Zähler mit dem Integrator-Seed) und Zeit.
/// Nicht enthalten: Box, Temperaturen, Integrator, Plugins (inkl. Observer und Pulling-Arbeit).
struct DynamicState {
  std::vector<float4> pos;       ///< (x, y, z) [nm], q [e]
  std::vector<float4> pos_corr;  ///< Positionskorrektur [nm] (ADR 0009)
  std::vector<Vec4d> vel;        ///< (vx, vy, vz) [nm/ps], 1/m [1/amu]
  std::vector<int2> image;       ///< Box-Imagezähler
  std::int64_t step = 0;
  double time_ps = 0.0;
  double previous_time_ps = 0.0;
};

class SimulationContext {
 public:
  /// Maximaler Betrag |x/L| bzw. |y/L| für Eingabepositionen (2^20).
  static constexpr float kMaxBoxImages = 1048576.0f;

  /// @param n_replicas  Anzahl Replicas (> 0)
  /// @param n_atoms     echte Atome pro Replica (> 0); n_pad = ceil(n_atoms/32)·32
  /// @param device      CUDA-Geräteindex
  SimulationContext(int n_replicas, int n_atoms, int device = 0);
  ~SimulationContext();

  SimulationContext(const SimulationContext&) = delete;
  SimulationContext& operator=(const SimulationContext&) = delete;
  SimulationContext(SimulationContext&&) = delete;
  SimulationContext& operator=(SimulationContext&&) = delete;

  int n_replicas() const { return state_.n_replicas; }
  int n_atoms() const { return state_.n_atoms; }
  int n_pad() const { return state_.n_pad; }
  int device() const { return device_; }
  cudaStream_t stream() const { return stream_.get(); }
  DeviceState& state() { return state_; }
  const DeviceState& state() const { return state_; }

  // --- Host-Setter (zusammenhängende, row-major Host-Arrays nur für echte Atome) ----------

  /// Positionen [nm], Shape (n_replicas, n_atoms, 3). Setzt die Imagezähler auf 0 zurück.
  void set_positions(const float* xyz_nm);
  /// Geschwindigkeiten [nm/ps], Shape (n_replicas, n_atoms, 3).
  void set_velocities(const double* v_nm_ps);
  /// Partialladungen [e], Shape (n_replicas, n_atoms).
  void set_charges(const float* q_e);
  /// Massen [amu], Shape (n_replicas, n_atoms), jede Masse endlich und > 0. Gespeichert als 1/m.
  void set_masses(const float* m_amu);
  /// (LJ-Typ, Optik-/Oberflächen-Typ), Shape (n_replicas, n_atoms, 2), Typen >= 0.
  void set_types(const int* types);
  /// Box (Lx, Ly) [nm] pro Replica, Shape (n_replicas, 2), endlich und > 0. Setzt die
  /// Imagezähler auf 0 zurück (neuer ungewrappter Ursprung = aktuelle Positionen).
  void set_box(const float* box_nm);
  /// Temperaturen [K] pro Replica, Shape (n_replicas,), endlich und >= 0.
  void set_temperatures(const float* t_k);
  /// Lösungsmittel-Abschirmung je Replica (κ_r [1/nm] >= 0, ε_r > 0, Shape (n_replicas,)) an alle
  /// Plugins und die Nonbonded-Strategie (Replica Exchange, ADR 0011). Kräfte gelten danach als
  /// veraltet.
  void set_screening(const float* kappa_per_nm, const float* eps_r);
  /// v_i *= factor_r für alle echten Atome der Replica r (Shape (n_replicas,), endlich, > 0).
  /// Für den Temperaturtausch: factor = sqrt(T_neu/T_alt). Synchronisiert.
  void scale_velocities(const double* factor);
  bool box_is_set() const { return box_set_; }
  /// Host-Kopie der Box (Lx, Ly) [nm] je Replica; leer, solange keine Box gesetzt ist.
  const std::vector<float2>& host_box() const { return h_box_; }
  /// Host-Kopie der Temperaturen T_r [K] je Replica (anfangs 0).
  const std::vector<float>& host_temperatures() const { return h_temperature_; }

  /// Geräte-Statuswort (Bits bomd::StatusBit, ADR 0008). Synchronisiert.
  unsigned int status();
  /// Setzt das Statuswort auf 0 (asynchron).
  void clear_status();

  // --- Kernels (asynchron auf stream()) ----------------------------------------------------

  /// d_force_fp und d_energy_fp auf 0 setzen.
  void clear_forces();
  /// d_force_fp (Fixed-Point) → d_force (float, kJ/mol/nm).
  void fixed_point_to_float();
  /// Koordinaten x/y echter Atome nach [0, L) wrappen, Imagezähler fortschreiben (ADR 0003).
  /// Wirft std::logic_error, wenn die Box nicht gesetzt ist.
  void wrap_xy();
  /// clear_forces → Nonbonded → alle ForcePlugins → fixed_point_to_float.
  void compute_forces();

  // --- Integration ---------------------------------------------------------------------------

  /// Setzt (oder ersetzt) den BAOAB-Integrator.
  void set_integrator(const LangevinParams& params);
  bool has_integrator() const { return integrator_ != nullptr; }
  /// Zeitschritt [ps]; 0, wenn kein Integrator gesetzt ist.
  double dt_ps() const { return integrator_ ? integrator_->params().dt_ps : 0.0; }
  /// Anzahl bisher ausgeführter Zeitschritte.
  std::int64_t current_step() const { return step_; }
  /// Simulationszeit t [ps], fortlaufend Σ Δt (bleibt bei einem Wechsel von Δt korrekt).
  double time_ps() const { return time_ps_; }
  /// Simulationszeit vor dem letzten Schritt [ps] (für Protokoll-Arbeit, z. B. Pulling).
  double previous_time_ps() const { return previous_time_ps_; }
  /// Integrator-Parameter (dt, γ, Seed); wirft std::logic_error ohne Integrator.
  const LangevinParams& integrator_params() const;
  /// Kopie des dynamischen Zustands (synchronisiert).
  DynamicState save_dynamic_state();
  /// Stellt einen mit save_dynamic_state gesicherten Zustand bitgenau wieder her. Der Kontext muss
  /// dieselbe Größe haben; Box, Temperaturen, Integrator und Plugins setzt der Aufrufer vorher
  /// (set_box setzt die Imagezähler zurück). Kräfte werden vor dem nächsten Schritt aus den
  /// wiederhergestellten Positionen neu berechnet (deterministisch, ADR 0002); die Positionen
  /// gelten als bereits projiziert (keine erneute SHAKE-Projektion).
  void restore_dynamic_state(const DynamicState& state);

  /// Markiert d_force als veraltet (z. B. nach Schreibzugriff über eine DLPack-Sicht);
  /// step() berechnet die Kräfte dann vor dem ersten B neu.
  void invalidate_forces() { forces_valid_ = false; }
  /// true, wenn d_force und die Energien dem aktuellen Zustand entsprechen.
  bool forces_valid() const { return forces_valid_; }
  /// Wird bei jedem set_masses erhöht; Plugins mit massenabhängigen Daten prüfen ihn.
  std::uint64_t masses_version() const { return masses_version_; }
  /// Wird bei jedem set_box erhöht; Plugins mit boxabhängigen Daten prüfen ihn.
  std::uint64_t box_version() const { return box_version_; }

  /// n BAOAB-Schritte (PHYSICS_MODEL.md §1, ARCHITECTURE.md §1). Pro Schritt:
  /// B A O A → wrap_xy → step += 1 → compute_forces → B → Observer mit step % stride == 0.
  /// Sind die Kräfte nach einer Zustandsänderung veraltet, werden sie vorher neu berechnet.
  /// Wirft std::logic_error ohne Integrator oder Box. Synchronisiert nicht.
  void step(std::int64_t n);

  /// Setzt (oder ersetzt) H-Bindungs-Constraints (SHAKE/RATTLE). Positionen und Geschwindigkeiten
  /// werden vor dem nächsten Schritt auf die Constraints projiziert.
  void set_constraints(ConstraintParams params);
  bool has_constraints() const { return constraints_ != nullptr; }
  /// Projiziert Positionen (SHAKE) und Geschwindigkeiten (RATTLE) sofort auf die Constraints.
  void apply_constraints();

  /// Energie-Frame pro Replica, Shape (n_replicas, 1 + kNumEnergyTerms), kJ/mol:
  /// [E_kin, E_pot je Term] zum aktuellen Zustand (E_pot aus dem letzten compute_forces).
  /// Synchronisiert.
  std::vector<double> energy_frame();
  /// Positionen x = d_pos + d_pos_corr in FP64 [nm], Shape (n_replicas, n_pad, 3). Synchronisiert.
  std::vector<double> positions_precise();
  /// N_dof = 3·N_mobile pro Replica (Atome mit inv_mass > 0; PHYSICS_MODEL.md §1). Synchronisiert.
  std::vector<int> degrees_of_freedom();
  /// Wartet, bis alle Arbeit auf stream() abgeschlossen ist.
  void synchronize();

  // --- Plugins ---------------------------------------------------------------------------

  /// Übernimmt den Besitz, ruft initialize() auf und gibt eine Referenz zurück.
  ForcePlugin& add_force(std::unique_ptr<ForcePlugin> plugin);
  ObserverPlugin& add_observer(std::unique_ptr<ObserverPlugin> plugin);
  /// Führt alle Observer einmal am aktuellen Zustand aus (unabhängig von stride; Schritt =
  /// current_step). Asynchron.
  void observe_now();
  void set_nonbonded(std::unique_ptr<NonbondedStrategy> strategy);
  std::size_t num_forces() const { return forces_.size(); }
  std::size_t num_observers() const { return observers_.size(); }
  bool has_nonbonded() const { return nonbonded_ != nullptr; }
  NonbondedStrategy* nonbonded() { return nonbonded_.get(); }

  /// Aktiviert das Gerät dieses Kontexts für den aufrufenden Host-Thread.
  void activate() const;

 private:
  void scatter_xyz(float4* dst, const float* src);
  void scatter_w(float4* dst, const float* src);
  void scatter_vel_xyz(const double* src);
  void scatter_vel_w(const double* src);
  /// Wirft, wenn ein echtes Atom |x| >= kMaxBoxImages·Lx oder |y| >= kMaxBoxImages·Ly hat.
  void check_wrap_range(const float* xyz, int atom_stride, int components) const;

  int device_ = 0;
  std::int64_t step_ = 0;
  double time_ps_ = 0.0;
  double previous_time_ps_ = 0.0;
  std::uint64_t masses_version_ = 0;
  std::uint64_t box_version_ = 0;
  bool forces_valid_ = false;       // d_force entspricht F(x) des aktuellen Zustands
  bool constraints_dirty_ = false;  // Zustand seit set_positions/velocities nicht projiziert
  CudaStream stream_;               // vor den Puffern deklariert ⇒ wird nach ihnen zerstört
  bool box_set_ = false;
  std::vector<float2> h_box_;         // Host-Kopie der Box für Eingabeprüfungen
  std::vector<float> h_temperature_;  // Host-Kopie der Temperaturen für Eingabeprüfungen

  DeviceBuffer<float4> pos_;
  DeviceBuffer<float4> pos_corr_;
  DeviceBuffer<Vec4d> vel_;
  DeviceBuffer<long long> force_fp_;
  DeviceBuffer<float4> force_;
  DeviceBuffer<long long> energy_fp_;
  DeviceBuffer<int2> types_;
  DeviceBuffer<float2> box_;
  DeviceBuffer<float> temperature_;
  DeviceBuffer<int2> image_;
  DeviceBuffer<unsigned int> status_;

  DeviceBuffer<double> energy_frame_;  // Arbeitspuffer für energy_frame()

  DeviceState state_;

  std::unique_ptr<LangevinIntegrator> integrator_;
  std::unique_ptr<ConstraintSet> constraints_;

  std::unique_ptr<NonbondedStrategy> nonbonded_;
  std::vector<std::unique_ptr<ForcePlugin>> forces_;
  std::vector<std::unique_ptr<ObserverPlugin>> observers_;
};

}  // namespace bomd
