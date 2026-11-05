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
};

}  // namespace bomd
