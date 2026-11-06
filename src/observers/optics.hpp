// BioOptoMD – In-situ-Optik-Observer (docs/PHYSICS_MODEL.md §6): DipoleAndDistanceTracker,
// FRETTracker, RefractiveIndexProfiler, PolarizabilityGridObserver (ADR 0016).
// NumPy-Referenz: tests/reference/observers_ref.py.
// Alle Observer ignorieren Padding-Atome (AGENTS.md Regel 6) und akkumulieren im VRAM (Regel 12).
#pragma once

#include <array>
#include <cstdint>
#include <string_view>
#include <vector>

#include "biooptomd/device_buffer.hpp"
#include "biooptomd/plugin.hpp"
#include "observers/ring_buffer.hpp"

namespace bomd {

/// Eine Dipol-Sonde pro Replica: Gruppe (Schwerpunkt) + Achsenatome A → B (lokale Indizes).
struct DipoleProbeParams {
  std::vector<int> group_offsets;  ///< CSR (n_replicas + 1)
  std::vector<int> group_atoms;    ///< lokale Indizes, je Gruppe verschieden
  std::vector<int> atom_a;         ///< (n_replicas,)
  std::vector<int> atom_b;         ///< (n_replicas,)
};

namespace detail {
/// Gerätedaten einer Sonde; Validierung gegen den Kontext in upload().
struct DeviceProbe {
  DeviceBuffer<int> offsets, atoms;
  DeviceBuffer<int2> axis;  // (A, B)
  void upload(const DipoleProbeParams& p, const SimulationContext& ctx, const char* name);
};
}  // namespace detail

/// Pro Frame und Replica (R_F [nm] (3), μ̂ (3), d = z(R_F) − z_surface [nm]); R_F ungewrappt.
class DipoleAndDistanceTracker final : public ObserverPlugin {
 public:
  DipoleAndDistanceTracker(DipoleProbeParams probe, float z_surface_nm, int stride, int capacity);
  std::string_view name() const override { return "DipoleAndDistanceTracker"; }
  int stride() const override { return stride_; }
  void initialize(SimulationContext& ctx) override;
  void observe(const DeviceState& s, std::int64_t step, cudaStream_t stream) override;
  void reset() override { ring_.reset(); }
  /// "frames" (n_replicas, n_frames, 7), "steps" (n_frames,).
  ObserverResult download() const override;
  std::int64_t overwritten() const { return ring_.overwritten(); }

 private:
  DipoleProbeParams params_;
  float z_surface_;
  int stride_, capacity_;
  SimulationContext* ctx_ = nullptr;
  detail::DeviceProbe probe_;
  DeviceRingBuffer ring_;
};

/// Pro Frame und Replica (R_DA [nm], κ², E): Rohzeitreihen; E nur für r0 > 0 (sonst 0).
class FRETTracker final : public ObserverPlugin {
 public:
  FRETTracker(DipoleProbeParams donor, DipoleProbeParams acceptor, float r0_nm, int stride,
              int capacity);
  std::string_view name() const override { return "FRETTracker"; }
  int stride() const override { return stride_; }
  void initialize(SimulationContext& ctx) override;
  void observe(const DeviceState& s, std::int64_t step, cudaStream_t stream) override;
  void reset() override { ring_.reset(); }
  /// "frames" (n_replicas, n_frames, 3) = (R_DA, κ², E), "steps" (n_frames,).
  ObserverResult download() const override;
  std::int64_t overwritten() const { return ring_.overwritten(); }

 private:
  DipoleProbeParams donor_, acceptor_;
  float r0_;
  int stride_, capacity_;
  SimulationContext* ctx_ = nullptr;
  detail::DeviceProbe d_donor_, d_acceptor_;
  DeviceRingBuffer ring_;
};

/// Verteilung eines Atoms auf die Schichten (PHYSICS_MODEL.md §6.3, OQ-02). R_i = (3 v_i/4π)^(1/3).
enum class LayerSmearing : int {
  kSphere = 0,  ///< exakter Kugelschnitt mit Radius R_i (Default, kompakter Träger)
  kGauss = 1,   ///< Gauß mit σ_i = R_i/√5 (gleiche z-Varianz wie die Kugel), Fenster ±4σ_i
  kPoint = 2,   ///< Punktatom in der Schicht seines Mittelpunkts (nur für Tests)
};

struct RefractiveIndexParams {
  float dz_nm = 0.05f;            ///< Schichtdicke Δz [nm] > 0
  int n_bins = 200;               ///< Anzahl Schichten
  float z_min_nm = 0.0f;          ///< Unterkante der ersten Schicht (Oberfläche)
  std::vector<float> alpha_nm3;   ///< α′ je Optik-Typ (d_types.y) [nm³] (OQ-01)
  std::vector<float> volume_nm3;  ///< v je Optik-Typ [nm³] (OQ-02)
  float n_water = 1.333f;         ///< Brechungsindex des Lösungsmittels bei λ
  /// Verteilung auf die Schichten; Atome mit v = 0 immer als Punkt.
  LayerSmearing smearing = LayerSmearing::kSphere;
};

/// Histogramme S_α(k), S_V(k) je Frame (Fixed-Point, deterministisch), über Frames als Summe und
/// Quadratsumme akkumuliert; n(z) nach Lorentz-Lorenz beim Download (PHYSICS_MODEL.md §6.3).
class RefractiveIndexProfiler final : public ObserverPlugin {
 public:
  RefractiveIndexProfiler(RefractiveIndexParams params, int stride);
  std::string_view name() const override { return "RefractiveIndexProfiler"; }
  int stride() const override { return stride_; }
  void initialize(SimulationContext& ctx) override;
  void observe(const DeviceState& s, std::int64_t step, cudaStream_t stream) override;
  void reset() override;
  /// "z" (n_bins,) Schichtmitten [nm], "n" (n_replicas, n_bins), "s_alpha_mean"/"s_alpha_var",
  /// "s_volume_mean"/"s_volume_var" (n_replicas, n_bins) [nm³ bzw. nm⁶], "n_frames" (1,).
  /// n = NaN, wo L >= 1 (unphysikalische Parameter). Die Box darf sich seit reset() nicht ändern.
  ObserverResult download() const override;
  int n_frames() const { return n_frames_; }

 private:
  RefractiveIndexParams params_;
  int stride_;
  int n_frames_ = 0;
  std::uint64_t box_version_ = 0;
  SimulationContext* ctx_ = nullptr;
  DeviceBuffer<float4> d_type_params_;    // (α′, v, R, 0), R = (3v/4π)^(1/3)
  DeviceBuffer<long long> d_frame_hist_;  // (n_replicas, 2, n_bins) Fixed-Point
  DeviceBuffer<double> d_sums_;           // (n_replicas, 4, n_bins): Σα, Σα², ΣV, ΣV²
};

/// 3D-Gegenstück zum RefractiveIndexProfiler (ADR 0016 Punkt 2). Zellen (Voxel) [x0 + i·Δx,
/// x0 + (i+1)·Δx) usw.; anders als VoxelGridForce3D (Knoten) ist das Gitter zellzentriert.
struct PolarizabilityGridParams {
  std::array<int, 3> shape{0, 0, 0};         ///< (nx, ny, nz), je >= 1
  std::array<float, 3> origin_nm{0, 0, 0};   ///< untere Ecke (x0, y0, z0)
  std::array<float, 3> spacing_nm{0, 0, 0};  ///< (Δx, Δy, Δz) > 0
  std::vector<float> alpha_nm3;              ///< α′ je Optik-Typ (d_types.y) [nm³]
  std::vector<float> volume_nm3;             ///< v je Optik-Typ [nm³]
  float n_water = 1.333f;
  /// kGauss (Default): separables Gauß-Gewicht je Achse wie im 1D-Profiler (σ = R/√5, Fenster
  /// ±4σ, normiert auf das Fenster); kPoint: Zelle des Mittelpunkts. kSphere ist nicht erlaubt.
  LayerSmearing smearing = LayerSmearing::kGauss;
  /// Periodisch in x/y (nx·Δx = L_x, ny·Δy = L_y); sonst fallen Anteile außerhalb des Gitters weg.
  bool periodic_xy = false;
};

/// Histogramme S_α, S_V je Voxel und Frame (Fixed-Point, deterministisch), über Frames als Summe
/// und Quadratsumme akkumuliert; ε = n² nach Lorentz-Lorenz mit Wasser-Hintergrund beim Download.
/// Speicher: 48 B je Voxel und Replica auf dem Gerät (Histogramm 16 B, Summen 32 B), beim Download
/// zusätzlich 5 Felder à 8 B auf dem Host. Aufwand je Atom und Frame bis zu m_x·m_y·m_z Atomics
/// (Fensterzellen je Achse); bei kleinen Voxeln ``stride`` entsprechend wählen.
class PolarizabilityGridObserver final : public ObserverPlugin {
 public:
  PolarizabilityGridObserver(PolarizabilityGridParams params, int stride);
  std::string_view name() const override { return "PolarizabilityGridObserver"; }
  int stride() const override { return stride_; }
  void initialize(SimulationContext& ctx) override;
  void observe(const DeviceState& s, std::int64_t step, cudaStream_t stream) override;
  void reset() override;
  /// "x", "y", "z" Zellmitten [nm]; "eps", "s_alpha_mean"/"s_alpha_var", "s_volume_mean"/
  /// "s_volume_var" (n_replicas, nz, ny, nx) [1, nm³ bzw. nm⁶]; "n_frames" (1,).
  /// eps = NaN, wo L >= 1.
  ObserverResult download() const override;
  int n_frames() const { return n_frames_; }

 private:
  void check_box() const;

  PolarizabilityGridParams params_;
  int stride_;
  int n_frames_ = 0;
  std::uint64_t box_version_ = 0;
  SimulationContext* ctx_ = nullptr;
  DeviceBuffer<float4> d_type_params_;    // (α′, v, R, 0)
  DeviceBuffer<long long> d_frame_hist_;  // (n_replicas, 2, n_vox) Fixed-Point
  DeviceBuffer<double> d_sums_;           // (n_replicas, 4, n_vox): Σα, Σα², ΣV, ΣV²
};

}  // namespace bomd
