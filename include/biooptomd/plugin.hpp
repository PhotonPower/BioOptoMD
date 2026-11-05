// BioOptoMD – Host-Schnittstellen für Kraft-, Observer- und Nonbonded-Plugins
// (docs/ARCHITECTURE.md §3). Virtuelle Funktionen existieren nur auf dem Host; jedes Plugin
// startet eigene Kernels auf dem übergebenen Stream (AGENTS.md Regeln 9, 11).
#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "biooptomd/device_state.hpp"

namespace bomd {

class SimulationContext;

/// Ein heruntergeladenes Ergebnis-Array eines Observers (Host-Speicher, row-major).
struct HostArray {
  std::string name;                 ///< z. B. "dipole"
  std::string unit;                 ///< Einheit der Einträge, z. B. "e nm"
  std::vector<std::int64_t> shape;  ///< z. B. {n_replicas, n_frames, 3}
  std::vector<double> data;         ///< Produkt(shape) Einträge
};

/// Ergebnis von ObserverPlugin::download().
struct ObserverResult {
  std::vector<HostArray> arrays;
};

/// Kraftbeitrag (z. B. Oberfläche, Restraint). compute() MUSS ausschließlich in
/// DeviceState::d_force_fp (und ggf. d_energy_fp) akkumulieren (ADR 0002).
class ForcePlugin {
 public:
  virtual ~ForcePlugin() = default;
  virtual std::string_view name() const = 0;
  /// Speicher allokieren, Parameter hochladen. Wird einmal beim Hinzufügen aufgerufen.
  virtual void initialize(SimulationContext&) {}
  /// Kräfte [kJ/mol/nm] und Energien [kJ/mol] in Fixed-Point addieren.
  virtual void compute(DeviceState& s, cudaStream_t stream) = 0;
  virtual bool contributes_energy() const { return true; }
  /// Host-seitig nach SimulationContext::restore_dynamic_state: schrittabhängigen Zustand
  /// verwerfen (z. B. „letzter Schritt“ für Arbeitsinkremente).
  virtual void on_state_restored() {}
  /// Neue Lösungsmittel-Abschirmung je Replica (Replica Exchange, ADR 0011): κ_r [1/nm], ε_r
  /// (n_replicas). Plugins mit κ/ε-abhängigen Termen laden sie hoch; Default: unabhängig.
  virtual void set_screening(const std::vector<float>& /*kappa*/,
                             const std::vector<float>& /*eps_r*/, cudaStream_t /*stream*/) {}
};

/// In-situ-Messung; akkumuliert auf dem Gerät, Download nur auf Anfrage (AGENTS.md Regel 12).
class ObserverPlugin {
 public:
  virtual ~ObserverPlugin() = default;
  virtual std::string_view name() const = 0;
  /// Beobachtungsintervall in Zeitschritten (> 0).
  virtual int stride() const = 0;
  virtual void initialize(SimulationContext&) {}
  virtual void observe(const DeviceState& s, std::int64_t step, cudaStream_t stream) = 0;
  /// Akkumulatoren leeren.
  virtual void reset() = 0;
  /// Gerät → Host; darf synchronisieren.
  virtual ObserverResult download() const = 0;
};

/// Strategy für nicht-kovalente Paarwechselwirkungen: TiledAllPairs (Phase 4), CellList (V2).
class NonbondedStrategy {
 public:
  virtual ~NonbondedStrategy() = default;
  virtual void initialize(SimulationContext&) {}
  virtual void compute(DeviceState& s, cudaStream_t stream) = 0;
  /// Wie ForcePlugin::set_screening (κ_r, ε_r bzw. ε_out je Replica).
  virtual void set_screening(const std::vector<float>& /*kappa*/,
                             const std::vector<float>& /*eps_r*/, cudaStream_t /*stream*/) {}
};

}  // namespace bomd
