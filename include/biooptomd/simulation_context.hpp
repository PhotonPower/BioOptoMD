// BioOptoMD – SimulationContext: besitzt alle Gerätepuffer, den CUDA-Stream und die Plugins
// einer Batch-Simulation (docs/ARCHITECTURE.md §2–3).
#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <memory>
#include <vector>

#include "biooptomd/cuda_stream.hpp"
#include "biooptomd/device_buffer.hpp"
#include "biooptomd/device_state.hpp"
#include "biooptomd/plugin.hpp"

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
  void set_velocities(const float* v_nm_ps);
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
  bool box_is_set() const { return box_set_; }

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
  /// Wartet, bis alle Arbeit auf stream() abgeschlossen ist.
  void synchronize();

  // --- Plugins ---------------------------------------------------------------------------

  /// Übernimmt den Besitz, ruft initialize() auf und gibt eine Referenz zurück.
  ForcePlugin& add_force(std::unique_ptr<ForcePlugin> plugin);
  ObserverPlugin& add_observer(std::unique_ptr<ObserverPlugin> plugin);
  void set_nonbonded(std::unique_ptr<NonbondedStrategy> strategy);
  std::size_t num_forces() const { return forces_.size(); }
  std::size_t num_observers() const { return observers_.size(); }
  bool has_nonbonded() const { return nonbonded_ != nullptr; }

  /// Aktiviert das Gerät dieses Kontexts für den aufrufenden Host-Thread.
  void activate() const;

 private:
  void scatter_xyz(float4* dst, const float* src);
  void scatter_w(float4* dst, const float* src);
  /// Wirft, wenn ein echtes Atom |x| >= kMaxBoxImages·Lx oder |y| >= kMaxBoxImages·Ly hat.
  void check_wrap_range(const float* xyz, int atom_stride, int components) const;

  int device_ = 0;
  CudaStream stream_;  // vor den Puffern deklariert ⇒ wird nach ihnen zerstört
  bool box_set_ = false;
  std::vector<float2> h_box_;  // Host-Kopie der Box für Eingabeprüfungen

  DeviceBuffer<float4> pos_;
  DeviceBuffer<float4> vel_;
  DeviceBuffer<long long> force_fp_;
  DeviceBuffer<float4> force_;
  DeviceBuffer<long long> energy_fp_;
  DeviceBuffer<int2> types_;
  DeviceBuffer<float2> box_;
  DeviceBuffer<float> temperature_;
  DeviceBuffer<int2> image_;

  DeviceState state_;

  std::unique_ptr<NonbondedStrategy> nonbonded_;
  std::vector<std::unique_ptr<ForcePlugin>> forces_;
  std::vector<std::unique_ptr<ObserverPlugin>> observers_;
};

}  // namespace bomd
