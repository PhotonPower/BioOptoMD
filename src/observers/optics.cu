// BioOptoMD – Optik-Observer, siehe optics.hpp. Formeln: PHYSICS_MODEL.md §6 und
// tests/reference/observers_ref.py.
#include <climits>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>

#include "biooptomd/cuda_check.hpp"
#include "biooptomd/simulation_context.hpp"
#include "core/core_kernels.cuh"
#include "core/fixed_point.cuh"
#include "core/mixed.cuh"
#include "observers/optics.hpp"

namespace bomd {
namespace {

constexpr int kProbeBlock = 128;  // ein Block pro Replica; Gruppen sind klein (Fluorophor)
constexpr double kPi = 3.14159265358979323846;

void require(bool condition, const std::string& message) {
  if (!condition) throw std::invalid_argument(message);
}

__device__ __forceinline__ double3 min_image(double3 d, float2 box) {
  d.x -= box.x * rint(d.x / box.x);
  d.y -= box.y * rint(d.y / box.y);
  return d;
}

/// Schwerpunkt der Gruppe (ungewrappt, massengewichtet, FP64, feste Reduktionsreihenfolge) und
/// Achsen-Einheitsvektor μ̂ = (r_B − r_A)/|r_B − r_A| (Minimum Image). Alle Threads des Blocks
/// rufen die Funktion auf; Ergebnis gültig in allen Threads.
__device__ void probe(const DeviceState& s, int r, const int* offsets, const int* atoms, int2 axis,
                      double3& com, double3& mu) {
  __shared__ double sh[4][kProbeBlock];
  const float2 box = s.d_box[r];
  const int base = r * s.n_pad;
  double ax = 0.0, ay = 0.0, az = 0.0, am = 0.0;
  for (int g = offsets[r] + threadIdx.x; g < offsets[r + 1]; g += blockDim.x) {
    const int i = base + atoms[g];
    const double3 x = detail::load_position(s, i);
    const int2 img = s.d_image[i];
    const double w = s.d_vel[i].w;
    const double m = w > 0.0 ? 1.0 / w : 0.0;
    ax += m * (x.x + static_cast<double>(img.x) * box.x);
    ay += m * (x.y + static_cast<double>(img.y) * box.y);
    az += m * x.z;
    am += m;
  }
  sh[0][threadIdx.x] = ax;
  sh[1][threadIdx.x] = ay;
  sh[2][threadIdx.x] = az;
  sh[3][threadIdx.x] = am;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride)
      for (int c = 0; c < 4; ++c) sh[c][threadIdx.x] += sh[c][threadIdx.x + stride];
    __syncthreads();
  }
  if (sh[3][0] <= 0.0 && threadIdx.x == 0)  // keine Massen gesetzt (inv_mass = 0): R_F = NaN
    atomicOr(s.d_status, kStatusNonFinite);
  com = make_double3(sh[0][0] / sh[3][0], sh[1][0] / sh[3][0], sh[2][0] / sh[3][0]);
  const double3 xa = detail::load_position(s, base + axis.x);
  const double3 xb = detail::load_position(s, base + axis.y);
  const double3 d = min_image(make_double3(xb.x - xa.x, xb.y - xa.y, xb.z - xa.z), box);
  const double len = sqrt(d.x * d.x + d.y * d.y + d.z * d.z);
  mu = make_double3(d.x / len, d.y / len, d.z / len);  // Division: korrekt gerundet (exakte Fälle)
  __syncthreads();  // sh darf vor dem nächsten Aufruf wiederverwendet werden
}

__global__ void k_dipole(DeviceState s, const int* offsets, const int* atoms, const int2* axis,
                         double z_surface, double* frame) {
  const int r = blockIdx.x;
  double3 com, mu;
  probe(s, r, offsets, atoms, axis[r], com, mu);
  if (threadIdx.x == 0) {
    double* out = frame + 7LL * r;
    out[0] = com.x;
    out[1] = com.y;
    out[2] = com.z;
    out[3] = mu.x;
    out[4] = mu.y;
    out[5] = mu.z;
    out[6] = com.z - z_surface;
  }
}

/// κ² = (μ̂_D·μ̂_A − 3(μ̂_D·R̂)(μ̂_A·R̂))², R = R_A − R_D (Minimum Image), E = 1/(1 + (R/R0)^6 (2/3)/κ²).
__global__ void k_fret(DeviceState s, const int* d_off, const int* d_atoms, const int2* d_axis,
                       const int* a_off, const int* a_atoms, const int2* a_axis, double r0,
                       double* frame) {
  const int r = blockIdx.x;
  double3 com_d, mu_d, com_a, mu_a;
  probe(s, r, d_off, d_atoms, d_axis[r], com_d, mu_d);
  probe(s, r, a_off, a_atoms, a_axis[r], com_a, mu_a);
  if (threadIdx.x == 0) {
    const double3 rv = min_image(
        make_double3(com_a.x - com_d.x, com_a.y - com_d.y, com_a.z - com_d.z), s.d_box[r]);
    const double dist = sqrt(rv.x * rv.x + rv.y * rv.y + rv.z * rv.z);
    const double3 rh = make_double3(rv.x / dist, rv.y / dist, rv.z / dist);
    const double cda = mu_d.x * mu_a.x + mu_d.y * mu_a.y + mu_d.z * mu_a.z;
    const double cdr = mu_d.x * rh.x + mu_d.y * rh.y + mu_d.z * rh.z;
    const double car = mu_a.x * rh.x + mu_a.y * rh.y + mu_a.z * rh.z;
    const double k2 = (cda - 3.0 * cdr * car) * (cda - 3.0 * cdr * car);
    double* out = frame + 3LL * r;
    out[0] = dist;
    out[1] = k2;
    if (r0 > 0.0) {
      const double q = dist / r0, q6 = q * q * q * q * q * q;
      out[2] = k2 > 0.0 ? 1.0 / (1.0 + q6 * (2.0 / 3.0) / k2) : 0.0;
    } else {
      out[2] = 0.0;
    }
  }
}

/// Pro echtem Atom: Beitrag (α′, v) seines Optik-Typs in die Schicht(en) des aktuellen Frames
/// (PHYSICS_MODEL.md §6.3). Anteile außerhalb des Profils entfallen.
///  - kSphere: Anteil der Kugel (Radius R) in [lo, hi]: mit a = max(lo, z − R) − z,
///    b = min(hi, z + R) − z ist w = 3/(4R³) [R²(b − a) − (b³ − a³)/3] (Σ_k w = 1).
///  - kGauss: w ∝ Φ((hi − z)/σ) − Φ((lo − z)/σ), σ = R/√5, über alle Schichten mit |z − z_k| ≤ 4σ,
///    normiert auf die Summe über dieses (nicht abgeschnittene) Fenster.
///  - kPoint bzw. R = 0: alles in die Schicht des Mittelpunkts.
__global__ void k_layer_histogram(DeviceState s, const float4* type_params, int n_types,
                                  double z_min, double dz, int n_bins, LayerSmearing mode,
                                  long long* hist) {
  const int i_local = blockIdx.x * blockDim.x + threadIdx.x;
  const int r = blockIdx.y;
  if (i_local >= s.n_atoms) return;  // Padding-Atome ignorieren (AGENTS.md Regel 6)
  const int i = r * s.n_pad + i_local;
  const int t = s.d_types[i].y;
  if (t >= n_types) {
    atomicOr(s.d_status, kStatusUnknownType);
    return;
  }
  const float4 av = type_params[t];  // (α′, v, R, 0)
  const double z = detail::load_position(s, i).z;
  long long* h_alpha = hist + static_cast<long long>(r) * 2 * n_bins;
  long long* h_volume = h_alpha + n_bins;
  const double radius = av.z;
  if (mode == LayerSmearing::kPoint || radius <= 0.0) {
    const double u = floor((z - z_min) / dz);
    if (u < 0.0 || u >= n_bins) return;
    const int k = static_cast<int>(u);
    detail::atomic_add_fixed_point(h_alpha + k, av.x, s.d_status);
    detail::atomic_add_fixed_point(h_volume + k, av.y, s.d_status);
    return;
  }
  const bool sphere = mode == LayerSmearing::kSphere;
  const double reach = sphere ? radius : 4.0 * radius / sqrt(5.0);
  const int w_lo = static_cast<int>(floor((z - reach - z_min) / dz));
  const int w_hi = static_cast<int>(floor((z + reach - z_min) / dz));
  const double inv = sqrt(5.0) / (radius * sqrt(2.0));  // 1/(σ√2)
  const double norm =
      sphere
          ? 1.0
          : 0.5 * (erf((z_min + (w_hi + 1) * dz - z) * inv) - erf((z_min + w_lo * dz - z) * inv));
  const double inv_r3 = 0.75 / (radius * radius * radius);
  for (int k = max(0, w_lo); k <= min(n_bins - 1, w_hi); ++k) {
    const double lo = z_min + k * dz, hi = lo + dz;
    double w;
    if (sphere) {
      const double a = fmax(lo, z - radius) - z, b = fmin(hi, z + radius) - z;
      w = b > a ? inv_r3 * (radius * radius * (b - a) - (b * b * b - a * a * a) / 3.0) : 0.0;
    } else {
      w = 0.5 * (erf((hi - z) * inv) - erf((lo - z) * inv)) / norm;
    }
    if (w <= 0.0) continue;
    detail::atomic_add_fixed_point(h_alpha + k, static_cast<float>(w * av.x), s.d_status);
    detail::atomic_add_fixed_point(h_volume + k, static_cast<float>(w * av.y), s.d_status);
  }
}

/// Frame-Histogramm → Summe und Quadratsumme (FP64), Histogramm zurücksetzen.
__global__ void k_layer_accumulate(int n_replicas, int n_bins, long long* hist, double* sums) {
  const int k = blockIdx.x * blockDim.x + threadIdx.x;
  const int r = blockIdx.y;
  if (k >= n_bins || r >= n_replicas) return;
  long long* h = hist + static_cast<long long>(r) * 2 * n_bins;
  double* out = sums + static_cast<long long>(r) * 4 * n_bins;
  const double a = static_cast<double>(h[k]) * (1.0 / units::FIXED_POINT_SCALE);
  const double v = static_cast<double>(h[n_bins + k]) * (1.0 / units::FIXED_POINT_SCALE);
  out[k] += a;
  out[n_bins + k] += a * a;
  out[2 * n_bins + k] += v;
  out[3 * n_bins + k] += v * v;
  h[k] = 0;
  h[n_bins + k] = 0;
}

// --- PolarizabilityGridObserver ----------------------------------------------------------------

constexpr int kGridWindow = 16;  // max. Zellen je Achse im Gauß-Fenster (in initialize() geprüft)

struct GridAxes {
  int n[3];
  double x0[3], d[3];
  int periodic_xy;
};

/// Zellen und Gewichte eines Atoms entlang einer Achse; Rückgabe: Anzahl m <= kGridWindow.
/// Gauß wie k_layer_histogram (Fenster ±4σ, normiert auf das Fenster, nicht auf das Gitter);
/// periodisch: Position in [x0, x0 + nΔ) gefaltet, Zellindizes modulo n.
__device__ int axis_weights(double x, double x0, double d, int n, bool periodic, bool point,
                            double radius, int* idx, double* w) {
  if (periodic) {
    const double len = n * d;
    x -= len * floor((x - x0) / len);
  }
  if (point) {
    int k = static_cast<int>(floor((x - x0) / d));
    if (periodic) {
      k %= n;
      if (k < 0) k += n;
    } else if (k < 0 || k >= n) {
      return 0;
    }
    idx[0] = k;
    w[0] = 1.0;
    return 1;
  }
  const double reach = 4.0 * radius / sqrt(5.0);
  const double inv = sqrt(5.0) / (radius * sqrt(2.0));  // 1/(σ√2)
  const int w_lo = static_cast<int>(floor((x - reach - x0) / d));
  const int w_hi = static_cast<int>(floor((x + reach - x0) / d));
  const double norm = 0.5 * (erf((x0 + (w_hi + 1) * d - x) * inv) - erf((x0 + w_lo * d - x) * inv));
  int m = 0;
  for (int k = w_lo; k <= w_hi && m < kGridWindow; ++k) {
    int kk = k;
    if (periodic) {
      kk %= n;
      if (kk < 0) kk += n;
    } else if (k < 0 || k >= n) {
      continue;
    }
    const double lo = x0 + k * d;
    const double wk = 0.5 * (erf((lo + d - x) * inv) - erf((lo - x) * inv)) / norm;
    if (wk <= 0.0) continue;
    idx[m] = kk;
    w[m] = wk;
    ++m;
  }
  return m;
}

/// Pro echtem Atom: separables Gewicht w_x w_y w_z auf die Voxel des aktuellen Frames.
__global__ void k_grid_histogram(DeviceState s, const float4* type_params, int n_types, GridAxes g,
                                 bool point_mode, long long* hist) {
  const int i_local = blockIdx.x * blockDim.x + threadIdx.x;
  const int r = blockIdx.y;
  if (i_local >= s.n_atoms) return;  // Padding-Atome ignorieren (AGENTS.md Regel 6)
  const int i = r * s.n_pad + i_local;
  const int t = s.d_types[i].y;
  if (t >= n_types) {
    atomicOr(s.d_status, kStatusUnknownType);
    return;
  }
  const float4 av = type_params[t];  // (α′, v, R, 0)
  if (av.x == 0.0f && av.y == 0.0f) return;
  const double3 x = detail::load_position(s, i);
  const bool point = point_mode || av.z <= 0.0f;
  int ix[kGridWindow], iy[kGridWindow], iz[kGridWindow];
  double wx[kGridWindow], wy[kGridWindow], wz[kGridWindow];
  const bool per = g.periodic_xy != 0;
  const int mx = axis_weights(x.x, g.x0[0], g.d[0], g.n[0], per, point, av.z, ix, wx);
  const int my = axis_weights(x.y, g.x0[1], g.d[1], g.n[1], per, point, av.z, iy, wy);
  const int mz = axis_weights(x.z, g.x0[2], g.d[2], g.n[2], false, point, av.z, iz, wz);
  const long long n_vox = static_cast<long long>(g.n[0]) * g.n[1] * g.n[2];
  long long* h_alpha = hist + static_cast<long long>(r) * 2 * n_vox;
  long long* h_volume = h_alpha + n_vox;
  for (int c = 0; c < mz; ++c) {
    for (int b = 0; b < my; ++b) {
      const double wzy = wz[c] * wy[b];
      const long long row = (static_cast<long long>(iz[c]) * g.n[1] + iy[b]) * g.n[0];
      for (int a = 0; a < mx; ++a) {
        const double w = wzy * wx[a];
        detail::atomic_add_fixed_point(h_alpha + row + ix[a], static_cast<float>(w * av.x),
                                       s.d_status);
        detail::atomic_add_fixed_point(h_volume + row + ix[a], static_cast<float>(w * av.y),
                                       s.d_status);
      }
    }
  }
}

ObserverResult frames_result(const DeviceRingBuffer& ring, cudaStream_t stream, const char* unit) {
  std::vector<double> data;
  std::vector<std::int64_t> steps;
  ring.download(stream, data, steps);
  HostArray frames{
      "frames", unit, {ring.n_replicas(), ring.n_frames(), ring.n_cols()}, std::move(data)};
  HostArray st{"steps", "1", {static_cast<std::int64_t>(steps.size())}, {}};
  for (std::int64_t v : steps) st.data.push_back(static_cast<double>(v));
  ObserverResult res;
  res.arrays.push_back(std::move(frames));
  res.arrays.push_back(std::move(st));
  return res;
}

}  // namespace

void detail::DeviceProbe::upload(const DipoleProbeParams& p, const SimulationContext& ctx,
                                 const char* name) {
  const int n_rep = ctx.n_replicas();
  const std::string who(name);
  require(p.group_offsets.size() == static_cast<std::size_t>(n_rep) + 1 &&
              p.group_offsets.front() == 0 &&
              p.group_offsets.back() == static_cast<int>(p.group_atoms.size()),
          who + ": group_offsets braucht n_replicas + 1 Einträge (0 … len(group_atoms))");
  require(p.atom_a.size() == static_cast<std::size_t>(n_rep) &&
              p.atom_b.size() == static_cast<std::size_t>(n_rep),
          who + ": atom_a/atom_b brauchen n_replicas Einträge");
  for (int r = 0; r < n_rep; ++r) {
    require(p.group_offsets[r] < p.group_offsets[r + 1], who + ": leere Gruppe");
    std::vector<bool> seen(ctx.n_atoms(), false);
    for (int g = p.group_offsets[r]; g < p.group_offsets[r + 1]; ++g) {
      const int a = p.group_atoms[g];
      require(a >= 0 && a < ctx.n_atoms(), who + ": Gruppenatom außerhalb [0, n_atoms)");
      require(!seen[a], who + ": Atom doppelt in der Gruppe");
      seen[a] = true;
    }
    require(p.atom_a[r] >= 0 && p.atom_a[r] < ctx.n_atoms() && p.atom_b[r] >= 0 &&
                p.atom_b[r] < ctx.n_atoms() && p.atom_a[r] != p.atom_b[r],
            who + ": Achsenatome A ≠ B in [0, n_atoms) erwartet");
  }
  std::vector<int2> ax(n_rep);
  for (int r = 0; r < n_rep; ++r) ax[r] = make_int2(p.atom_a[r], p.atom_b[r]);
  offsets = DeviceBuffer<int>(p.group_offsets.size());
  atoms = DeviceBuffer<int>(p.group_atoms.size());
  axis = DeviceBuffer<int2>(n_rep);
  offsets.upload_async(p.group_offsets.data(), ctx.stream());
  atoms.upload_async(p.group_atoms.data(), ctx.stream());
  axis.upload_async(ax.data(), ctx.stream());
  BOMD_CUDA_CHECK(cudaStreamSynchronize(ctx.stream()));
}

// --- DipoleAndDistanceTracker ---------------------------------------------------------------

DipoleAndDistanceTracker::DipoleAndDistanceTracker(DipoleProbeParams probe, float z_surface_nm,
                                                   int stride, int capacity)
    : params_(std::move(probe)), z_surface_(z_surface_nm), stride_(stride), capacity_(capacity) {
  require(stride > 0 && capacity > 0, "DipoleAndDistanceTracker: stride, capacity > 0");
  require(std::isfinite(z_surface_nm), "DipoleAndDistanceTracker: z_surface endlich");
}

void DipoleAndDistanceTracker::initialize(SimulationContext& ctx) {
  ctx_ = &ctx;
  probe_.upload(params_, ctx, "DipoleAndDistanceTracker");
  ring_ = DeviceRingBuffer(capacity_, ctx.n_replicas(), 7);
}

void DipoleAndDistanceTracker::observe(const DeviceState& s, std::int64_t step,
                                       cudaStream_t stream) {
  k_dipole<<<s.n_replicas, kProbeBlock, 0, stream>>>(s, probe_.offsets.data(), probe_.atoms.data(),
                                                     probe_.axis.data(), z_surface_,
                                                     ring_.next_frame());
  BOMD_LAUNCH_CHECK();
  ring_.commit(step);
}

ObserverResult DipoleAndDistanceTracker::download() const {
  ctx_->activate();
  return frames_result(ring_, ctx_->stream(), "nm | 1 | nm");
}

// --- FRETTracker ------------------------------------------------------------------------------

FRETTracker::FRETTracker(DipoleProbeParams donor, DipoleProbeParams acceptor, float r0_nm,
                         int stride, int capacity)
    : donor_(std::move(donor)),
      acceptor_(std::move(acceptor)),
      r0_(r0_nm),
      stride_(stride),
      capacity_(capacity) {
  require(stride > 0 && capacity > 0, "FRETTracker: stride, capacity > 0");
  require(std::isfinite(r0_nm) && r0_nm >= 0.0f, "FRETTracker: r0 >= 0 (0 = keine Effizienz)");
}

void FRETTracker::initialize(SimulationContext& ctx) {
  ctx_ = &ctx;
  d_donor_.upload(donor_, ctx, "FRETTracker(Donor)");
  d_acceptor_.upload(acceptor_, ctx, "FRETTracker(Akzeptor)");
  ring_ = DeviceRingBuffer(capacity_, ctx.n_replicas(), 3);
}

void FRETTracker::observe(const DeviceState& s, std::int64_t step, cudaStream_t stream) {
  k_fret<<<s.n_replicas, kProbeBlock, 0, stream>>>(
      s, d_donor_.offsets.data(), d_donor_.atoms.data(), d_donor_.axis.data(),
      d_acceptor_.offsets.data(), d_acceptor_.atoms.data(), d_acceptor_.axis.data(), r0_,
      ring_.next_frame());
  BOMD_LAUNCH_CHECK();
  ring_.commit(step);
}

ObserverResult FRETTracker::download() const {
  ctx_->activate();
  return frames_result(ring_, ctx_->stream(), "nm | 1 | 1");
}

// --- RefractiveIndexProfiler --------------------------------------------------------------------

RefractiveIndexProfiler::RefractiveIndexProfiler(RefractiveIndexParams params, int stride)
    : params_(std::move(params)), stride_(stride) {
  const auto& p = params_;
  require(stride > 0, "RefractiveIndexProfiler: stride > 0");
  require(std::isfinite(p.dz_nm) && p.dz_nm > 0.0f && p.n_bins > 0 && std::isfinite(p.z_min_nm),
          "RefractiveIndexProfiler: dz > 0, n_bins > 0, z_min endlich");
  require(!p.alpha_nm3.empty() && p.alpha_nm3.size() == p.volume_nm3.size(),
          "RefractiveIndexProfiler: alpha und volume je Optik-Typ (gleiche Länge)");
  for (std::size_t t = 0; t < p.alpha_nm3.size(); ++t)
    require(std::isfinite(p.alpha_nm3[t]) && p.alpha_nm3[t] >= 0.0f &&
                std::isfinite(p.volume_nm3[t]) && p.volume_nm3[t] >= 0.0f,
            "RefractiveIndexProfiler: alpha, volume endlich und >= 0");
  require(std::isfinite(p.n_water) && p.n_water >= 1.0f, "RefractiveIndexProfiler: n_water >= 1");
  const int mode = static_cast<int>(p.smearing);
  require(mode >= 0 && mode <= 2, "RefractiveIndexProfiler: unbekannte Verschmierung");
}

void RefractiveIndexProfiler::initialize(SimulationContext& ctx) {
  ctx_ = &ctx;
  const std::size_t n_types = params_.alpha_nm3.size();
  std::vector<float4> tp(n_types);
  for (std::size_t t = 0; t < n_types; ++t) {
    const double radius = std::cbrt(3.0 * params_.volume_nm3[t] / (4.0 * kPi));
    tp[t] =
        make_float4(params_.alpha_nm3[t], params_.volume_nm3[t], static_cast<float>(radius), 0.0f);
  }
  d_type_params_ = DeviceBuffer<float4>(n_types);
  d_frame_hist_ =
      DeviceBuffer<long long>(static_cast<std::size_t>(ctx.n_replicas()) * 2 * params_.n_bins);
  d_sums_ = DeviceBuffer<double>(static_cast<std::size_t>(ctx.n_replicas()) * 4 * params_.n_bins);
  d_type_params_.upload_async(tp.data(), ctx.stream());
  d_frame_hist_.zero_async(ctx.stream());
  d_sums_.zero_async(ctx.stream());
  ctx.synchronize();
}

void RefractiveIndexProfiler::observe(const DeviceState& s, std::int64_t, cudaStream_t stream) {
  // V_k = L_x L_y Δz wird beim Download aus der aktuellen Box berechnet ⇒ Box muss seit reset()
  // gleich sein.
  if (n_frames_ == 0) box_version_ = ctx_->box_version();
  if (box_version_ != ctx_->box_version())
    throw std::logic_error(
        "RefractiveIndexProfiler: Box seit reset() geändert – erst reset() aufrufen");
  const dim3 grid_atoms(detail::elementwise_grid(s.n_atoms), s.n_replicas);
  k_layer_histogram<<<grid_atoms, detail::kElementwiseBlock, 0, stream>>>(
      s, d_type_params_.data(), static_cast<int>(params_.alpha_nm3.size()), params_.z_min_nm,
      params_.dz_nm, params_.n_bins, params_.smearing, d_frame_hist_.data());
  BOMD_LAUNCH_CHECK();
  const dim3 grid_bins(detail::elementwise_grid(params_.n_bins), s.n_replicas);
  k_layer_accumulate<<<grid_bins, detail::kElementwiseBlock, 0, stream>>>(
      s.n_replicas, params_.n_bins, d_frame_hist_.data(), d_sums_.data());
  BOMD_LAUNCH_CHECK();
  ++n_frames_;
}

void RefractiveIndexProfiler::reset() {
  ctx_->activate();
  d_sums_.zero_async(ctx_->stream());
  n_frames_ = 0;
}

ObserverResult RefractiveIndexProfiler::download() const {
  ctx_->activate();
  const int n_rep = ctx_->n_replicas(), nb = params_.n_bins;
  std::vector<double> sums(d_sums_.size());
  d_sums_.download_async(sums.data(), ctx_->stream());
  BOMD_CUDA_CHECK(cudaStreamSynchronize(ctx_->stream()));
  const auto& box = ctx_->host_box();
  if (box.empty()) throw std::logic_error("RefractiveIndexProfiler: Box nicht gesetzt");
  const std::int64_t shape[2] = {n_rep, nb};
  auto arr = [&](const char* name, const char* unit) {
    return HostArray{name,
                     unit,
                     {shape[0], shape[1]},
                     std::vector<double>(static_cast<std::size_t>(n_rep) * nb, 0.0)};
  };
  HostArray n = arr("n", "1"), am = arr("s_alpha_mean", "nm^3"), av = arr("s_alpha_var", "nm^6"),
            vm = arr("s_volume_mean", "nm^3"), vv = arr("s_volume_var", "nm^6");
  const double nw = params_.n_water;
  const double lw = (nw * nw - 1.0) / (nw * nw + 2.0);
  const double f = n_frames_ > 0 ? 1.0 / n_frames_ : 0.0;
  for (int r = 0; r < n_rep; ++r) {
    const double vk = static_cast<double>(box[r].x) * box[r].y * params_.dz_nm;
    for (int k = 0; k < nb; ++k) {
      const double* s = sums.data() + static_cast<std::size_t>(r) * 4 * nb;
      const std::size_t o = static_cast<std::size_t>(r) * nb + k;
      am.data[o] = s[k] * f;
      av.data[o] = std::max(0.0, s[nb + k] * f - am.data[o] * am.data[o]);
      vm.data[o] = s[2 * nb + k] * f;
      vv.data[o] = std::max(0.0, s[3 * nb + k] * f - vm.data[o] * vm.data[o]);
      if (am.data[o] == 0.0 && vm.data[o] == 0.0) {
        n.data[o] = nw;  // reines Lösungsmittel: Lorentz-Lorenz liefert exakt n_w
        continue;
      }
      const double fw = std::max(0.0, 1.0 - vm.data[o] / vk);
      const double big_l = 4.0 * kPi / 3.0 * am.data[o] / vk + fw * lw;
      // L >= 1 (Lorentz-Katastrophe, unphysikalische α′/Dichte) ⇒ NaN statt einer Scheinzahl.
      n.data[o] = big_l < 1.0 ? std::sqrt((1.0 + 2.0 * big_l) / (1.0 - big_l)) : std::nan("");
    }
  }
  HostArray z{"z", "nm", {nb}, {}};
  for (int k = 0; k < nb; ++k) z.data.push_back(params_.z_min_nm + (k + 0.5) * params_.dz_nm);
  HostArray frames{"n_frames", "1", {1}, {static_cast<double>(n_frames_)}};
  ObserverResult res;
  for (HostArray* a : {&z, &n, &am, &av, &vm, &vv, &frames}) res.arrays.push_back(std::move(*a));
  return res;
}

// --- PolarizabilityGridObserver ----------------------------------------------------------------

PolarizabilityGridObserver::PolarizabilityGridObserver(PolarizabilityGridParams params, int stride)
    : params_(std::move(params)), stride_(stride) {
  const auto& p = params_;
  const std::string who = "PolarizabilityGridObserver: ";
  require(stride > 0, who + "stride > 0");
  for (int d = 0; d < 3; ++d) {
    require(p.shape[d] >= 1, who + "shape je >= 1");
    require(std::isfinite(p.spacing_nm[d]) && p.spacing_nm[d] > 0.0f, who + "spacing > 0");
    require(std::isfinite(p.origin_nm[d]), who + "origin endlich");
  }
  require(!p.alpha_nm3.empty() && p.alpha_nm3.size() == p.volume_nm3.size(),
          who + "alpha und volume je Optik-Typ (gleiche Länge)");
  for (std::size_t t = 0; t < p.alpha_nm3.size(); ++t)
    require(std::isfinite(p.alpha_nm3[t]) && p.alpha_nm3[t] >= 0.0f &&
                std::isfinite(p.volume_nm3[t]) && p.volume_nm3[t] >= 0.0f,
            who + "alpha, volume endlich und >= 0");
  require(std::isfinite(p.n_water) && p.n_water >= 1.0f, who + "n_water >= 1");
  require(p.smearing == LayerSmearing::kGauss || p.smearing == LayerSmearing::kPoint,
          who + "Verschmierung 'gauss' oder 'point'");
  const long long n_vox = static_cast<long long>(p.shape[0]) * p.shape[1] * p.shape[2];
  require(4 * n_vox <= INT_MAX, who + "zu viele Voxel (4·n_vox muss in int passen)");
  if (p.smearing == LayerSmearing::kGauss) {
    for (float v : p.volume_nm3) {
      // Wie im Kernel aus dem float-gerundeten R; eine Zelle Reserve, damit axis_weights nie kürzt
      // (Fensterbreite floor(b) − floor(a) + 1 <= floor(b − a) + 2).
      const float radius = static_cast<float>(std::cbrt(3.0 * v / (4.0 * kPi)));
      const double reach = 4.0 * radius / std::sqrt(5.0);
      for (int d = 0; d < 3; ++d)
        require(std::floor(2.0 * reach / p.spacing_nm[d]) + 3.0 <= kGridWindow,
                who +
                    "Voxel zu klein für das Gauß-Fenster (Fenster 4σ je Seite, höchstens 16 Zellen "
                    "je Achse)");
    }
  }
}

void PolarizabilityGridObserver::check_box() const {
  const auto& p = params_;
  if (!p.periodic_xy) return;
  require(!ctx_->host_box().empty(), "PolarizabilityGridObserver: periodic_xy verlangt eine Box");
  for (const float2& b : ctx_->host_box())
    require(std::fabs(p.shape[0] * p.spacing_nm[0] - b.x) <= 1e-5f * b.x &&
                std::fabs(p.shape[1] * p.spacing_nm[1] - b.y) <= 1e-5f * b.y,
            "PolarizabilityGridObserver: periodic_xy verlangt nx·Δx = L_x und ny·Δy = L_y");
}

void PolarizabilityGridObserver::initialize(SimulationContext& ctx) {
  ctx_ = &ctx;
  check_box();
  box_version_ = ctx.box_version();
  const std::size_t n_types = params_.alpha_nm3.size();
  std::vector<float4> tp(n_types);
  for (std::size_t t = 0; t < n_types; ++t) {
    const double radius = std::cbrt(3.0 * params_.volume_nm3[t] / (4.0 * kPi));
    tp[t] =
        make_float4(params_.alpha_nm3[t], params_.volume_nm3[t], static_cast<float>(radius), 0.0f);
  }
  const std::size_t n_vox =
      static_cast<std::size_t>(params_.shape[0]) * params_.shape[1] * params_.shape[2];
  d_type_params_ = DeviceBuffer<float4>(n_types);
  d_frame_hist_ = DeviceBuffer<long long>(static_cast<std::size_t>(ctx.n_replicas()) * 2 * n_vox);
  d_sums_ = DeviceBuffer<double>(static_cast<std::size_t>(ctx.n_replicas()) * 4 * n_vox);
  d_type_params_.upload_async(tp.data(), ctx.stream());
  d_frame_hist_.zero_async(ctx.stream());
  d_sums_.zero_async(ctx.stream());
  ctx.synchronize();
}

void PolarizabilityGridObserver::observe(const DeviceState& s, std::int64_t, cudaStream_t stream) {
  if (box_version_ != ctx_->box_version()) {  // Periodizität hängt an der Box, V_k nicht
    check_box();
    box_version_ = ctx_->box_version();
  }
  const auto& p = params_;
  GridAxes g{};
  for (int d = 0; d < 3; ++d) {
    g.n[d] = p.shape[d];
    g.x0[d] = p.origin_nm[d];
    g.d[d] = p.spacing_nm[d];
  }
  g.periodic_xy = p.periodic_xy ? 1 : 0;
  const dim3 grid_atoms(detail::elementwise_grid(s.n_atoms), s.n_replicas);
  k_grid_histogram<<<grid_atoms, detail::kElementwiseBlock, 0, stream>>>(
      s, d_type_params_.data(), static_cast<int>(p.alpha_nm3.size()), g,
      p.smearing == LayerSmearing::kPoint, d_frame_hist_.data());
  BOMD_LAUNCH_CHECK();
  const int n_vox = p.shape[0] * p.shape[1] * p.shape[2];
  const dim3 grid_vox(detail::elementwise_grid(n_vox), s.n_replicas);
  k_layer_accumulate<<<grid_vox, detail::kElementwiseBlock, 0, stream>>>(
      s.n_replicas, n_vox, d_frame_hist_.data(), d_sums_.data());
  BOMD_LAUNCH_CHECK();
  ++n_frames_;
}

void PolarizabilityGridObserver::reset() {
  ctx_->activate();
  d_sums_.zero_async(ctx_->stream());
  n_frames_ = 0;
}

ObserverResult PolarizabilityGridObserver::download() const {
  ctx_->activate();
  const auto& p = params_;
  const int n_rep = ctx_->n_replicas();
  const int nx = p.shape[0], ny = p.shape[1], nz = p.shape[2];
  const std::size_t nv = static_cast<std::size_t>(nx) * ny * nz;
  std::vector<double> sums(d_sums_.size());
  d_sums_.download_async(sums.data(), ctx_->stream());
  BOMD_CUDA_CHECK(cudaStreamSynchronize(ctx_->stream()));
  auto arr = [&](const char* name, const char* unit) {
    return HostArray{name, unit, {n_rep, nz, ny, nx}, std::vector<double>(n_rep * nv, 0.0)};
  };
  HostArray eps = arr("eps", "1"), am = arr("s_alpha_mean", "nm^3"),
            av = arr("s_alpha_var", "nm^6"), vm = arr("s_volume_mean", "nm^3"),
            vv = arr("s_volume_var", "nm^6");
  const double nw = p.n_water;
  const double lw = (nw * nw - 1.0) / (nw * nw + 2.0);
  const double vk = static_cast<double>(p.spacing_nm[0]) * p.spacing_nm[1] * p.spacing_nm[2];
  const double f = n_frames_ > 0 ? 1.0 / n_frames_ : 0.0;
  for (int r = 0; r < n_rep; ++r) {
    const double* s = sums.data() + static_cast<std::size_t>(r) * 4 * nv;
    for (std::size_t k = 0; k < nv; ++k) {
      const std::size_t o = static_cast<std::size_t>(r) * nv + k;
      am.data[o] = s[k] * f;
      av.data[o] = std::max(0.0, s[nv + k] * f - am.data[o] * am.data[o]);
      vm.data[o] = s[2 * nv + k] * f;
      vv.data[o] = std::max(0.0, s[3 * nv + k] * f - vm.data[o] * vm.data[o]);
      if (am.data[o] == 0.0 && vm.data[o] == 0.0) {
        eps.data[o] = nw * nw;  // reines Lösungsmittel exakt
        continue;
      }
      const double fw = std::max(0.0, 1.0 - vm.data[o] / vk);
      const double big_l = 4.0 * kPi / 3.0 * am.data[o] / vk + fw * lw;
      eps.data[o] = big_l < 1.0 ? (1.0 + 2.0 * big_l) / (1.0 - big_l) : std::nan("");
    }
  }
  ObserverResult res;
  const char* axes[3] = {"x", "y", "z"};
  for (int d = 0; d < 3; ++d) {
    HostArray c{axes[d], "nm", {p.shape[d]}, {}};
    for (int k = 0; k < p.shape[d]; ++k)
      c.data.push_back(p.origin_nm[d] + (k + 0.5) * static_cast<double>(p.spacing_nm[d]));
    res.arrays.push_back(std::move(c));
  }
  HostArray frames{"n_frames", "1", {1}, {static_cast<double>(n_frames_)}};
  for (HostArray* a : {&eps, &am, &av, &vm, &vv, &frames}) res.arrays.push_back(std::move(*a));
  return res;
}

}  // namespace bomd
