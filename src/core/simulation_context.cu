// BioOptoMD – SimulationContext: Allokation, Host-Setter, Kernel-Aufrufe, Plugin-Pipeline.
#include <climits>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "biooptomd/cuda_check.hpp"
#include "biooptomd/simulation_context.hpp"
#include "core/core_kernels.cuh"

namespace bomd {
namespace {

/// Obergrenze aus gridDim.y (Kernels mit blockIdx.y = Replica).
constexpr int kMaxReplicas = 65535;

void require(bool condition, const std::string& message) {
  if (!condition) throw std::invalid_argument(message);
}

}  // namespace

SimulationContext::SimulationContext(int n_replicas, int n_atoms, int device) : device_(device) {
  require(n_replicas > 0 && n_replicas <= kMaxReplicas,
          "n_replicas muss in [1, " + std::to_string(kMaxReplicas) + "] liegen");
  require(n_atoms > 0 && n_atoms <= INT_MAX - kTileSize, "n_atoms muss in [1, 2^31 - 33] liegen");
  // n_replicas·n_pad·3 muss in int-Indizes der Kernels passen.
  require(static_cast<long long>(n_replicas) * padded_atom_count(n_atoms) * 3 < (1LL << 31),
          "System zu groß: n_replicas * n_pad * 3 >= 2^31");

  activate();
  const int n_pad = padded_atom_count(n_atoms);
  const std::size_t n_total = static_cast<std::size_t>(n_replicas) * n_pad;

  pos_ = DeviceBuffer<float4>(n_total);
  pos_corr_ = DeviceBuffer<float4>(n_total);
  vel_ = DeviceBuffer<Vec4d>(n_total);
  force_fp_ = DeviceBuffer<long long>(3 * n_total);
  force_ = DeviceBuffer<float4>(n_total);
  energy_fp_ = DeviceBuffer<long long>(static_cast<std::size_t>(n_replicas) * kNumEnergyTerms);
  types_ = DeviceBuffer<int2>(n_total);
  box_ = DeviceBuffer<float2>(n_replicas);
  temperature_ = DeviceBuffer<float>(n_replicas);
  image_ = DeviceBuffer<int2>(n_total);
  status_ = DeviceBuffer<unsigned int>(1);
  energy_frame_ =
      DeviceBuffer<double>(static_cast<std::size_t>(n_replicas) * detail::kEnergyFrameColumns);
  // Blockierender Stream ⇒ implizit geordnet mit dem Legacy-Default-Stream (z. B. von CuPy).
  stream_ = CudaStream::create();

  // Alles 0: insbesondere Padding-Atome mit q = 0, inv_mass = 0, LJ-Typ 0 (AGENTS.md Regel 6).
  pos_.zero_async(stream_.get());
  pos_corr_.zero_async(stream_.get());
  vel_.zero_async(stream_.get());
  force_fp_.zero_async(stream_.get());
  force_.zero_async(stream_.get());
  energy_fp_.zero_async(stream_.get());
  types_.zero_async(stream_.get());
  box_.zero_async(stream_.get());
  temperature_.zero_async(stream_.get());
  image_.zero_async(stream_.get());
  status_.zero_async(stream_.get());

  state_.n_replicas = n_replicas;
  state_.n_atoms = n_atoms;
  state_.n_pad = n_pad;
  state_.d_pos = pos_.data();
  state_.d_pos_corr = pos_corr_.data();
  state_.d_vel = vel_.data();
  state_.d_force_fp = force_fp_.data();
  state_.d_force = force_.data();
  state_.d_energy_fp = energy_fp_.data();
  state_.d_types = types_.data();
  state_.d_box = box_.data();
  state_.d_temperature = temperature_.data();
  state_.d_image = image_.data();
  state_.d_status = status_.data();

  synchronize();
}

SimulationContext::~SimulationContext() {
  // Plugins zuerst freigeben (sie dürfen eigene Puffer besitzen); danach geben die
  // Member-Destruktoren die Puffer und zuletzt den Stream frei.
  cudaSetDevice(device_);
  if (stream_.get() != nullptr) cudaStreamSynchronize(stream_.get());
  observers_.clear();
  forces_.clear();
  nonbonded_.reset();
}

void SimulationContext::activate() const {
  BOMD_CUDA_CHECK(cudaSetDevice(device_));
}

unsigned int SimulationContext::status() {
  activate();
  unsigned int value = 0;
  status_.download_async(&value, stream_.get());
  synchronize();
  return value;
}

void SimulationContext::clear_status() {
  activate();
  status_.zero_async(stream_.get());
}

void SimulationContext::synchronize() {
  activate();
  BOMD_CUDA_CHECK(cudaStreamSynchronize(stream_.get()));
}

// --- Host-Setter -------------------------------------------------------------------------

void SimulationContext::scatter_xyz(float4* dst, const float* src) {
  activate();
  const std::size_t n = static_cast<std::size_t>(n_replicas()) * n_atoms() * 3;
  for (std::size_t k = 0; k < n; ++k) require(std::isfinite(src[k]), "Wert nicht endlich");
  DeviceBuffer<float> staging(n);
  staging.upload_async(src, stream_.get());
  detail::launch_scatter_xyz(state_, dst, staging.data(), stream_.get());
  synchronize();  // staging lebt bis hier.
}

void SimulationContext::scatter_w(float4* dst, const float* src) {
  activate();
  const std::size_t n = static_cast<std::size_t>(n_replicas()) * n_atoms();
  DeviceBuffer<float> staging(n);
  staging.upload_async(src, stream_.get());
  detail::launch_scatter_w(state_, dst, staging.data(), stream_.get());
  synchronize();
}

void SimulationContext::check_wrap_range(const float* xyz, int atom_stride, int components) const {
  if (!box_set_) return;
  for (int r = 0; r < n_replicas(); ++r) {
    for (int i = 0; i < n_atoms(); ++i) {
      const float* p = xyz + (static_cast<std::size_t>(r) * atom_stride + i) * components;
      require(std::fabs(p[0]) < kMaxBoxImages * h_box_[r].x &&
                  std::fabs(p[1]) < kMaxBoxImages * h_box_[r].y,
              "Position ausserhalb des wrap-faehigen Bereichs |x/L| < 2^20");
    }
  }
}

void SimulationContext::set_positions(const float* xyz_nm) {
  const std::size_t n = static_cast<std::size_t>(n_replicas()) * n_atoms() * 3;
  for (std::size_t k = 0; k < n; ++k) require(std::isfinite(xyz_nm[k]), "Wert nicht endlich");
  check_wrap_range(xyz_nm, n_atoms(), 3);
  scatter_xyz(pos_.data(), xyz_nm);
  pos_corr_.zero_async(stream_.get());  // Eingabe ist FP32 ⇒ Korrektur 0
  forces_valid_ = false;
  constraints_dirty_ = true;
  image_.zero_async(stream_.get());  // neue Positionen = neuer ungewrappter Ursprung
  synchronize();
}

void SimulationContext::scatter_vel_xyz(const double* src) {
  activate();
  const std::size_t n = static_cast<std::size_t>(n_replicas()) * n_atoms() * 3;
  for (std::size_t k = 0; k < n; ++k) require(std::isfinite(src[k]), "Wert nicht endlich");
  DeviceBuffer<double> staging(n);
  staging.upload_async(src, stream_.get());
  detail::launch_scatter_xyz(state_, vel_.data(), staging.data(), stream_.get());
  synchronize();
}

void SimulationContext::scatter_vel_w(const double* src) {
  activate();
  DeviceBuffer<double> staging(static_cast<std::size_t>(n_replicas()) * n_atoms());
  staging.upload_async(src, stream_.get());
  detail::launch_scatter_w(state_, vel_.data(), staging.data(), stream_.get());
  synchronize();
}

void SimulationContext::set_velocities(const double* v_nm_ps) {
  scatter_vel_xyz(v_nm_ps);
  constraints_dirty_ = true;
}

void SimulationContext::set_constraints(ConstraintParams params) {
  activate();
  constraints_ = std::make_unique<ConstraintSet>(std::move(params), *this);
  constraints_dirty_ = true;
}

void SimulationContext::apply_constraints() {
  if (!constraints_) return;
  activate();
  constraints_->project_positions(state_, stream_.get());
  constraints_->rattle(state_, stream_.get());
  constraints_dirty_ = false;
  forces_valid_ = false;
}

void SimulationContext::set_charges(const float* q_e) {
  const std::size_t n = static_cast<std::size_t>(n_replicas()) * n_atoms();
  for (std::size_t k = 0; k < n; ++k) require(std::isfinite(q_e[k]), "Ladung nicht endlich");
  forces_valid_ = false;
  scatter_w(pos_.data(), q_e);
}

void SimulationContext::set_masses(const float* m_amu) {
  const std::size_t n = static_cast<std::size_t>(n_replicas()) * n_atoms();
  std::vector<double> inv_mass(n);
  for (std::size_t k = 0; k < n; ++k) {
    require(std::isfinite(m_amu[k]) && m_amu[k] > 0.0f, "Massen müssen endlich und > 0 sein");
    inv_mass[k] = 1.0 / static_cast<double>(m_amu[k]);
  }
  forces_valid_ = false;  // massengewichtete Kräfte (z. B. Pulling) hängen von m ab
  ++masses_version_;
  scatter_vel_w(inv_mass.data());
}

void SimulationContext::set_types(const int* types) {
  activate();
  std::vector<int2> padded(static_cast<std::size_t>(n_replicas()) * n_pad(), int2{0, 0});
  for (int r = 0; r < n_replicas(); ++r) {
    for (int i = 0; i < n_atoms(); ++i) {
      const std::size_t k = (static_cast<std::size_t>(r) * n_atoms() + i) * 2;
      require(types[k] >= 0 && types[k + 1] >= 0, "Typen müssen >= 0 sein");
      padded[static_cast<std::size_t>(r) * n_pad() + i] = int2{types[k], types[k + 1]};
    }
  }
  forces_valid_ = false;
  types_.upload_async(padded.data(), stream_.get());
  synchronize();
}

void SimulationContext::set_box(const float* box_nm) {
  activate();
  std::vector<float2> box(n_replicas());
  for (int r = 0; r < n_replicas(); ++r) {
    const float lx = box_nm[2 * r];
    const float ly = box_nm[2 * r + 1];
    require(std::isfinite(lx) && std::isfinite(ly) && lx > 0.0f && ly > 0.0f,
            "Boxlängen müssen endlich und > 0 sein");
    box[r] = float2{lx, ly};
  }
  // Aktuelle Positionen gegen die neue Box prüfen, bevor etwas geändert wird.
  std::vector<float4> pos(pos_.size());
  pos_.download_async(pos.data(), stream_.get());
  synchronize();
  std::swap(h_box_, box);
  const bool was_set = std::exchange(box_set_, true);
  try {
    check_wrap_range(&pos[0].x, n_pad(), 4);
  } catch (...) {
    std::swap(h_box_, box);
    box_set_ = was_set;
    throw;
  }
  forces_valid_ = false;
  box_.upload_async(h_box_.data(), stream_.get());
  image_.zero_async(stream_.get());  // neue Box = neuer ungewrappter Ursprung
  ++box_version_;
  synchronize();
}

void SimulationContext::set_screening(const float* kappa_per_nm, const float* eps_r) {
  activate();
  std::vector<float> kappa(kappa_per_nm, kappa_per_nm + n_replicas());
  std::vector<float> eps(eps_r, eps_r + n_replicas());
  for (int r = 0; r < n_replicas(); ++r)
    require(std::isfinite(kappa[r]) && kappa[r] >= 0.0f && std::isfinite(eps[r]) && eps[r] > 0.0f,
            "set_screening: kappa >= 0 und eps_r > 0 erwartet");
  for (auto& force : forces_) force->set_screening(kappa, eps, stream_.get());
  if (nonbonded_) nonbonded_->set_screening(kappa, eps, stream_.get());
  forces_valid_ = false;
}

void SimulationContext::scale_velocities(const double* factor) {
  activate();
  for (int r = 0; r < n_replicas(); ++r)
    require(std::isfinite(factor[r]) && factor[r] > 0.0, "scale_velocities: Faktor > 0 erwartet");
  DeviceBuffer<double> d_factor(n_replicas());
  d_factor.upload_async(factor, stream_.get());
  detail::launch_scale_velocities(state_, d_factor.data(), stream_.get());
  synchronize();  // d_factor lebt bis hier
}

void SimulationContext::set_temperatures(const float* t_k) {
  activate();
  for (int r = 0; r < n_replicas(); ++r)
    require(std::isfinite(t_k[r]) && t_k[r] >= 0.0f, "Temperaturen müssen endlich und >= 0 sein");
  h_temperature_.assign(t_k, t_k + n_replicas());
  temperature_.upload_async(t_k, stream_.get());
  synchronize();
}

// --- Kernels ----------------------------------------------------------------------------

void SimulationContext::clear_forces() {
  activate();
  detail::launch_clear_forces(state_, stream_.get());
}

void SimulationContext::fixed_point_to_float() {
  activate();
  detail::launch_fixed_point_to_float(state_, stream_.get());
}

void SimulationContext::wrap_xy() {
  if (!box_set_) throw std::logic_error("wrap_xy: Box nicht gesetzt (set_box aufrufen)");
  activate();
  detail::launch_wrap_xy(state_, stream_.get());
}

void SimulationContext::compute_forces() {
  clear_forces();
  if (nonbonded_) nonbonded_->compute(state_, stream_.get());
  for (auto& force : forces_) force->compute(state_, stream_.get());
  fixed_point_to_float();
  forces_valid_ = true;
}

// --- Integration ------------------------------------------------------------------------

void SimulationContext::set_integrator(const LangevinParams& params) {
  integrator_ = std::make_unique<LangevinIntegrator>(params);
}

const LangevinParams& SimulationContext::integrator_params() const {
  if (!integrator_) throw std::logic_error("integrator_params: kein Integrator gesetzt");
  return integrator_->params();
}

DynamicState SimulationContext::save_dynamic_state() {
  activate();
  const std::size_t n = static_cast<std::size_t>(state_.n_total());
  DynamicState out;
  out.pos.resize(n);
  out.pos_corr.resize(n);
  out.vel.resize(n);
  out.image.resize(n);
  pos_.download_async(out.pos.data(), stream_.get());
  pos_corr_.download_async(out.pos_corr.data(), stream_.get());
  vel_.download_async(out.vel.data(), stream_.get());
  image_.download_async(out.image.data(), stream_.get());
  synchronize();
  out.step = step_;
  out.time_ps = time_ps_;
  out.previous_time_ps = previous_time_ps_;
  return out;
}

void SimulationContext::restore_dynamic_state(const DynamicState& state) {
  const std::size_t n = static_cast<std::size_t>(state_.n_total());
  require(state.pos.size() == n && state.pos_corr.size() == n && state.vel.size() == n &&
              state.image.size() == n,
          "restore_dynamic_state: Größe passt nicht zum Kontext (n_replicas·n_pad)");
  require(state.step >= 0 && std::isfinite(state.time_ps) && std::isfinite(state.previous_time_ps),
          "restore_dynamic_state: step >= 0, Zeiten endlich");
  activate();
  pos_.upload_async(state.pos.data(), stream_.get());
  pos_corr_.upload_async(state.pos_corr.data(), stream_.get());
  vel_.upload_async(state.vel.data(), stream_.get());
  image_.upload_async(state.image.data(), stream_.get());
  synchronize();
  step_ = state.step;
  time_ps_ = state.time_ps;
  previous_time_ps_ = state.previous_time_ps;
  forces_valid_ = false;
  constraints_dirty_ = false;
  for (auto& force : forces_) force->on_state_restored();
}

void SimulationContext::step(std::int64_t n) {
  require(n >= 0, "step: n muss >= 0 sein");
  if (!integrator_) throw std::logic_error("step: kein Integrator gesetzt (set_integrator)");
  if (!box_set_) throw std::logic_error("step: Box nicht gesetzt (set_box)");
  activate();
  if (constraints_ && constraints_dirty_) apply_constraints();
  if (!forces_valid_) compute_forces();
  const cudaStream_t st = stream_.get();
  const LangevinIntegrator& li = *integrator_;
  for (std::int64_t k = 0; k < n; ++k) {
    if (constraints_) {
      // B, RATTLE, A+SHAKE, O, RATTLE, A+SHAKE (PHYSICS_MODEL.md §1)
      const ConstraintSet& cs = *constraints_;
      li.kick(state_, st);
      cs.rattle(state_, st);
      li.drift(state_, cs.d_in_cluster(), st);
      cs.drift_and_shake(state_, li.half_dt(), st);
      li.thermostat(state_, step_, st);
      cs.rattle(state_, st);
      li.drift(state_, cs.d_in_cluster(), st);
      cs.drift_and_shake(state_, li.half_dt(), st);
    } else {
      li.first_half(state_, step_, st);
    }
    detail::launch_wrap_xy(state_, st);
    ++step_;
    previous_time_ps_ = time_ps_;
    time_ps_ += li.params().dt_ps;
    compute_forces();
    li.second_half(state_, st);
    if (constraints_) constraints_->rattle(state_, st);
    for (auto& observer : observers_)
      if (step_ % observer->stride() == 0) observer->observe(state_, step_, st);
  }
}

std::vector<double> SimulationContext::energy_frame() {
  activate();
  detail::launch_energy_frame(state_, energy_frame_.data(), stream_.get());
  std::vector<double> out(energy_frame_.size());
  energy_frame_.download_async(out.data(), stream_.get());
  synchronize();
  return out;
}

std::vector<int> SimulationContext::degrees_of_freedom() {
  activate();
  std::vector<Vec4d> vel(vel_.size());
  vel_.download_async(vel.data(), stream_.get());
  synchronize();
  std::vector<int> n_dof(n_replicas(), 0);
  for (int r = 0; r < n_replicas(); ++r)
    for (int i = 0; i < n_atoms(); ++i)
      if (vel[static_cast<std::size_t>(r) * n_pad() + i].w > 0.0) n_dof[r] += 3;
  if (constraints_)  // N_dof = 3·N_mobile − N_constraints (PHYSICS_MODEL.md §1)
    for (int r = 0; r < n_replicas(); ++r) n_dof[r] -= constraints_->constraint_counts()[r];
  return n_dof;
}

std::vector<double> SimulationContext::positions_precise() {
  activate();
  std::vector<float4> pos(pos_.size()), corr(pos_corr_.size());
  pos_.download_async(pos.data(), stream_.get());
  pos_corr_.download_async(corr.data(), stream_.get());
  synchronize();
  std::vector<double> out(3 * pos.size());
  for (std::size_t i = 0; i < pos.size(); ++i) {
    out[3 * i] = static_cast<double>(pos[i].x) + corr[i].x;
    out[3 * i + 1] = static_cast<double>(pos[i].y) + corr[i].y;
    out[3 * i + 2] = static_cast<double>(pos[i].z) + corr[i].z;
  }
  return out;
}

// --- Plugins ----------------------------------------------------------------------------

ForcePlugin& SimulationContext::add_force(std::unique_ptr<ForcePlugin> plugin) {
  require(plugin != nullptr, "add_force: Plugin ist null");
  activate();
  plugin->initialize(*this);
  forces_.push_back(std::move(plugin));
  forces_valid_ = false;
  return *forces_.back();
}

ObserverPlugin& SimulationContext::add_observer(std::unique_ptr<ObserverPlugin> plugin) {
  require(plugin != nullptr, "add_observer: Plugin ist null");
  require(plugin->stride() > 0, "add_observer: stride muss > 0 sein");
  activate();
  plugin->initialize(*this);
  observers_.push_back(std::move(plugin));
  return *observers_.back();
}

void SimulationContext::observe_now() {
  activate();
  for (auto& observer : observers_) observer->observe(state_, step_, stream_.get());
}

void SimulationContext::set_nonbonded(std::unique_ptr<NonbondedStrategy> strategy) {
  activate();
  if (strategy) strategy->initialize(*this);
  nonbonded_ = std::move(strategy);
  forces_valid_ = false;
}

}  // namespace bomd
