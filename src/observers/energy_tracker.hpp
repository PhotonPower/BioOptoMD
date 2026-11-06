// BioOptoMD – EnergyTracker: Zeitreihe von E_kin und E_pot je Term pro Replica im VRAM
// (ROADMAP Phase 2, Aufgabe 4). Download nur auf Anfrage (AGENTS.md Regel 12).
#pragma once

#include <cstdint>
#include <string_view>
#include <vector>

#include "biooptomd/device_buffer.hpp"
#include "biooptomd/plugin.hpp"

namespace bomd {

class EnergyTracker final : public ObserverPlugin {
 public:
  /// @param stride    Beobachtungsintervall in Schritten (> 0)
  /// @param capacity  maximale Anzahl Frames (> 0); weitere Frames werden verworfen und gezählt
  EnergyTracker(int stride, int capacity);

  std::string_view name() const override { return "EnergyTracker"; }
  int stride() const override { return stride_; }
  void initialize(SimulationContext& ctx) override;
  void observe(const DeviceState& s, std::int64_t step, cudaStream_t stream) override;
  void reset() override;
  /// Arrays "energies" (n_replicas, n_frames, 1 + N_TERMS) [kJ/mol], "steps" (n_frames,) und
  /// "kinetic_mid" (n_replicas, n_frames) [kJ/mol]: E_kin nach O im beobachteten Schritt (OQ-14).
  ObserverResult download() const override;

  int n_frames() const { return n_frames_; }
  std::int64_t dropped_frames() const { return dropped_; }

 private:
  int stride_;
  int capacity_;
  int n_frames_ = 0;
  std::int64_t dropped_ = 0;
  int n_replicas_ = 0;
  std::vector<std::int64_t> steps_;
  SimulationContext* ctx_ = nullptr;
  DeviceBuffer<double> frames_;  // (capacity, n_replicas, 1 + N_TERMS)
  DeviceBuffer<double> mid_;     // (capacity, n_replicas): E_kin nach O
};

}  // namespace bomd
