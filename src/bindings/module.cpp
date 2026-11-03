// nanobind-Modul biooptomd._core: SimulationContext, Array-Export (NumPy-Kopie und DLPack
// Zero-Copy) und Test-Hooks (_core.testing).
#include <cuda_runtime.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/array.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "biooptomd/cuda_check.hpp"
#include "biooptomd/device_state.hpp"
#include "biooptomd/simulation_context.hpp"
#include "biooptomd/units.hpp"
#include "core/testing.hpp"
#include "forces/harmonic_restraint.hpp"
#include "forces/planar_surface.hpp"
#include "observers/energy_tracker.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace {

using bomd::SimulationContext;

template <typename T>
using HostInput = nb::ndarray<const T, nb::c_contig, nb::device::cpu>;

using Shape = std::vector<std::size_t>;

/// Prüft die Shape eines Eingabe-Arrays, sonst ValueError.
template <typename T>
void check_shape(const HostInput<T>& a, const Shape& expected, const char* name) {
  bool ok = a.ndim() == expected.size();
  for (std::size_t d = 0; ok && d < expected.size(); ++d) ok = a.shape(d) == expected[d];
  if (!ok) {
    std::string msg = std::string(name) + ": erwartete Shape (";
    for (std::size_t d = 0; d < expected.size(); ++d)
      msg += std::to_string(expected[d]) + (d + 1 < expected.size() ? ", " : "");
    throw nb::value_error((msg + ")").c_str());
  }
}

std::size_t product(const Shape& shape) {
  std::size_t n = 1;
  for (std::size_t s : shape) n *= s;
  return n;
}

/// Exportiert einen Gerätepuffer mit der Shape `shape`.
///  - dlpack = false: NumPy-Kopie im Host-Speicher (synchronisiert den Stream).
///  - dlpack = true:  Zero-Copy-Sicht auf den Gerätespeicher mit __dlpack__ (z. B. für
///    cupy.from_dlpack, torch.from_dlpack). Der Stream wird vorher synchronisiert. Die Sicht
///    hält den Kontext am Leben; spätere Kernels des Kontexts verändern die Daten (keine
///    Kopie!) und sind nicht mit fremden Streams synchronisiert.
template <typename T>
nb::object export_array(SimulationContext& ctx, nb::handle owner, const T* d_ptr, Shape shape,
                        bool dlpack) {
  ctx.synchronize();
  if (dlpack) {
    ctx.invalidate_forces();  // Sicht ist beschreibbar ⇒ Kräfte vor dem nächsten step neu
    return nb::cast(nb::ndarray<nb::array_api, T>(const_cast<T*>(d_ptr), shape.size(), shape.data(),
                                                  owner, nullptr, nb::dtype<T>(),
                                                  nb::device::cuda::value, ctx.device()));
  }
  const std::size_t n = product(shape);
  T* host = new T[n];
  nb::capsule deleter(host, [](void* p) noexcept { delete[] static_cast<T*>(p); });
  BOMD_CUDA_CHECK(cudaMemcpy(host, d_ptr, n * sizeof(T), cudaMemcpyDeviceToHost));
  return nb::cast(nb::ndarray<nb::numpy, T>(host, shape.size(), shape.data(), deleter));
}

/// Registriert eine Export-Methode `name(dlpack=False)` für einen Zustandspuffer.
template <typename T, typename Getter, typename ShapeFn>
void def_export(nb::class_<SimulationContext>& cls, const char* name, Getter get_ptr,
                ShapeFn shape_fn, const char* doc) {
  cls.def(
      name,
      [get_ptr, shape_fn](SimulationContext& ctx, bool dlpack) {
        return export_array<T>(ctx, nb::find(ctx), get_ptr(ctx.state()), shape_fn(ctx), dlpack);
      },
      "dlpack"_a = false, doc);
}

/// Host-Vektor → NumPy-Array (Kopie) mit Shape `shape`.
template <typename T>
nb::object vector_to_numpy(const std::vector<T>& v, Shape shape) {
  if (product(shape) != v.size()) throw std::logic_error("vector_to_numpy: Shape passt nicht");
  T* host = new T[v.size() > 0 ? v.size() : 1];
  nb::capsule deleter(host, [](void* p) noexcept { delete[] static_cast<T*>(p); });
  std::copy(v.begin(), v.end(), host);
  return nb::cast(nb::ndarray<nb::numpy, T>(host, shape.size(), shape.data(), deleter));
}

template <typename T>
std::vector<T> to_vector(const HostInput<T>& a) {
  return std::vector<T>(a.data(), a.data() + a.size());
}

Shape atoms_shape(const SimulationContext& c, std::size_t last) {
  return {static_cast<std::size_t>(c.n_replicas()), static_cast<std::size_t>(c.n_pad()), last};
}

Shape real_atoms_shape(const SimulationContext& c, std::size_t last) {
  Shape s{static_cast<std::size_t>(c.n_replicas()), static_cast<std::size_t>(c.n_atoms())};
  if (last > 0) s.push_back(last);
  return s;
}

}  // namespace

NB_MODULE(_core, m) {
  m.doc() = "BioOptoMD C++/CUDA-Kern. Einheiten: nm, ps, amu, kJ/mol, e, K.";

  nb::register_exception_translator([](const std::exception_ptr& p, void*) {
    try {
      std::rethrow_exception(p);
    } catch (const bomd::CudaError& e) {
      PyErr_SetString(PyExc_RuntimeError, e.what());
    }
  });

  m.def(
      "cuda_device_count",
      []() {
        int n = 0;
        if (cudaGetDeviceCount(&n) != cudaSuccess) return 0;
        return n;
      },
      "Anzahl sichtbarer CUDA-Geräte (0, wenn kein Treiber/Gerät).");

  m.attr("TILE_SIZE") = bomd::kTileSize;
  m.attr("N_ENERGY_TERMS") = static_cast<int>(bomd::kNumEnergyTerms);

  // Kompilierte Konstanten aus units.hpp (Test: test_units.py).
  auto units = m.def_submodule("units", "Konstanten aus include/biooptomd/units.hpp.");
  units.attr("ONE_4PI_EPS0") = bomd::units::ONE_4PI_EPS0;
  units.attr("KB") = bomd::units::KB;
  units.attr("FARADAY") = bomd::units::FARADAY;
  units.attr("DEBYE_PER_ENM") = bomd::units::DEBYE_PER_ENM;
  units.attr("FIXED_POINT_SCALE") = bomd::units::FIXED_POINT_SCALE;
  units.attr("EPS_R2") = static_cast<double>(bomd::units::EPS_R2);
  m.def("padded_atom_count", &bomd::padded_atom_count, "n_atoms"_a,
        "Gepaddete Atomzahl ceil(n_atoms/32)·32.");

  auto cls = nb::class_<SimulationContext>(
      m, "SimulationContext",
      "Besitzt alle Gerätepuffer einer Batch-Simulation (n_replicas × n_pad Atomplätze).\n\n"
      "Export-Methoden liefern das rohe, gepaddete Layout (ARCHITECTURE.md §2.2), wahlweise\n"
      "als NumPy-Kopie (dlpack=False) oder als Zero-Copy-DLPack-Sicht (dlpack=True).");

  cls.def(nb::init<int, int, int>(), "n_replicas"_a, "n_atoms"_a, "device"_a = 0,
          "Allokiert und nullt alle Puffer. n_pad = ceil(n_atoms/32)·32.")
      .def_prop_ro("n_replicas", &SimulationContext::n_replicas)
      .def_prop_ro("n_atoms", &SimulationContext::n_atoms, "Echte Atome pro Replica.")
      .def_prop_ro("n_pad", &SimulationContext::n_pad, "Gepaddete Atome pro Replica.")
      .def_prop_ro("device", &SimulationContext::device)
      .def_prop_ro("box_is_set", &SimulationContext::box_is_set)
      .def_prop_ro("num_forces", &SimulationContext::num_forces)
      .def_prop_ro("num_observers", &SimulationContext::num_observers)
      .def_prop_ro("has_nonbonded", &SimulationContext::has_nonbonded);

  // --- Setter (nur echte Atome) -----------------------------------------------------------
  cls.def(
         "set_positions",
         [](SimulationContext& c, HostInput<float> a) {
           check_shape(a, real_atoms_shape(c, 3), "positions");
           c.set_positions(a.data());
         },
         "positions"_a,
         "Positionen [nm], float32 (n_replicas, n_atoms, 3). Setzt die Imagezähler auf 0.")
      .def(
          "set_velocities",
          [](SimulationContext& c, HostInput<float> a) {
            check_shape(a, real_atoms_shape(c, 3), "velocities");
            c.set_velocities(a.data());
          },
          "velocities"_a, "Geschwindigkeiten [nm/ps], float32 (n_replicas, n_atoms, 3).")
      .def(
          "set_charges",
          [](SimulationContext& c, HostInput<float> a) {
            check_shape(a, real_atoms_shape(c, 0), "charges");
            c.set_charges(a.data());
          },
          "charges"_a, "Partialladungen [e], float32 (n_replicas, n_atoms).")
      .def(
          "set_masses",
          [](SimulationContext& c, HostInput<float> a) {
            check_shape(a, real_atoms_shape(c, 0), "masses");
            c.set_masses(a.data());
          },
          "masses"_a, "Massen [amu] > 0, float32 (n_replicas, n_atoms); gespeichert als 1/m.")
      .def(
          "set_types",
          [](SimulationContext& c, HostInput<int32_t> a) {
            check_shape(a, real_atoms_shape(c, 2), "types");
            c.set_types(a.data());
          },
          "types"_a, "(LJ-Typ, Optik-/Oberflächen-Typ) >= 0, int32 (n_replicas, n_atoms, 2).")
      .def(
          "set_box",
          [](SimulationContext& c, HostInput<float> a) {
            check_shape(a, {static_cast<std::size_t>(c.n_replicas()), 2}, "box");
            c.set_box(a.data());
          },
          "box"_a, "Box (Lx, Ly) [nm] > 0 pro Replica, float32 (n_replicas, 2).")
      .def(
          "set_temperatures",
          [](SimulationContext& c, HostInput<float> a) {
            check_shape(a, {static_cast<std::size_t>(c.n_replicas())}, "temperatures");
            c.set_temperatures(a.data());
          },
          "temperatures"_a, "Temperaturen [K] >= 0 pro Replica, float32 (n_replicas,).");

  // --- Kernels ------------------------------------------------------------------------------
  cls.def("clear_forces", &SimulationContext::clear_forces,
          "Fixed-Point-Kräfte und -Energien auf 0 setzen (asynchron).")
      .def("fixed_point_to_float", &SimulationContext::fixed_point_to_float,
           "Fixed-Point-Kräfte nach float [kJ/mol/nm] konvertieren (asynchron).")
      .def("wrap_xy", &SimulationContext::wrap_xy,
           "x/y echter Atome nach [0, L) wrappen, Imagezähler fortschreiben (asynchron).")
      .def("compute_forces", &SimulationContext::compute_forces,
           "clear_forces → Nonbonded → ForcePlugins → fixed_point_to_float (asynchron).")
      .def("synchronize", &SimulationContext::synchronize,
           "Wartet auf alle Arbeit im Stream des Kontexts.");

  // --- Exporte (gepaddetes Rohlayout) -----------------------------------------------------
  using bomd::DeviceState;
  const auto atoms4 = [](const SimulationContext& c) { return atoms_shape(c, 4); };
  const auto atoms2 = [](const SimulationContext& c) { return atoms_shape(c, 2); };
  def_export<float>(
      cls, "positions", [](const DeviceState& s) { return &s.d_pos->x; }, atoms4,
      "float32 (n_replicas, n_pad, 4): x, y, z [nm], q [e].");
  def_export<float>(
      cls, "velocities", [](const DeviceState& s) { return &s.d_vel->x; }, atoms4,
      "float32 (n_replicas, n_pad, 4): vx, vy, vz [nm/ps], 1/m [1/amu].");
  def_export<float>(
      cls, "forces", [](const DeviceState& s) { return &s.d_force->x; }, atoms4,
      "float32 (n_replicas, n_pad, 4): Fx, Fy, Fz [kJ/mol/nm], w = 0.");
  def_export<int64_t>(
      cls, "forces_fixed_point",
      [](const DeviceState& s) { return reinterpret_cast<const int64_t*>(s.d_force_fp); },
      [](const SimulationContext& c) {
        return Shape{3, static_cast<std::size_t>(c.n_replicas()),
                     static_cast<std::size_t>(c.n_pad())};
      },
      "int64 (3, n_replicas, n_pad): Kraft·2^32 [kJ/mol/nm], SoA fx | fy | fz.");
  def_export<int64_t>(
      cls, "energies_fixed_point",
      [](const DeviceState& s) { return reinterpret_cast<const int64_t*>(s.d_energy_fp); },
      [](const SimulationContext& c) {
        return Shape{static_cast<std::size_t>(c.n_replicas()),
                     static_cast<std::size_t>(bomd::kNumEnergyTerms)};
      },
      "int64 (n_replicas, N_ENERGY_TERMS): Energie·2^32 [kJ/mol].");
  def_export<int32_t>(
      cls, "types", [](const DeviceState& s) { return &s.d_types->x; }, atoms2,
      "int32 (n_replicas, n_pad, 2): (LJ-Typ, Optik-/Oberflächen-Typ).");
  def_export<float>(
      cls, "box", [](const DeviceState& s) { return &s.d_box->x; },
      [](const SimulationContext& c) { return Shape{static_cast<std::size_t>(c.n_replicas()), 2}; },
      "float32 (n_replicas, 2): (Lx, Ly) [nm].");
  def_export<float>(
      cls, "temperatures", [](const DeviceState& s) { return s.d_temperature; },
      [](const SimulationContext& c) { return Shape{static_cast<std::size_t>(c.n_replicas())}; },
      "float32 (n_replicas,): T [K].");
  def_export<int32_t>(
      cls, "images", [](const DeviceState& s) { return &s.d_image->x; }, atoms2,
      "int32 (n_replicas, n_pad, 2): Box-Imagezähler (nx, ny).");

  // --- Integration und Energien ------------------------------------------------------------
  nb::list term_names;
  for (const char* name :
       {"bond", "angle", "torsion", "pair14", "nonbonded", "surface", "restraint", "external"})
    term_names.append(name);
  m.attr("ENERGY_TERMS") = nb::tuple(term_names);

  cls.def(
         "set_integrator",
         [](SimulationContext& c, double dt, double friction, std::uint64_t seed) {
           c.set_integrator(bomd::LangevinParams{dt, friction, seed});
         },
         "dt"_a, "friction"_a, "seed"_a,
         "Langevin-BAOAB: dt [ps] > 0, friction gamma [1/ps] >= 0 (0 = NVE), seed (uint64).")
      .def_prop_ro("has_integrator", &SimulationContext::has_integrator)
      .def_prop_ro("dt", &SimulationContext::dt_ps, "Zeitschritt [ps] (0 ohne Integrator).")
      .def_prop_ro("current_step", &SimulationContext::current_step)
      .def_prop_ro("time", &SimulationContext::time_ps, "Simulationszeit [ps].")
      .def("step", &SimulationContext::step, "n"_a,
           "n BAOAB-Schritte (asynchron). Benötigt Integrator und Box.")
      .def(
          "energies",
          [](SimulationContext& c) {
            return vector_to_numpy(c.energy_frame(),
                                   {static_cast<std::size_t>(c.n_replicas()),
                                    static_cast<std::size_t>(1 + bomd::kNumEnergyTerms)});
          },
          "float64 (n_replicas, 1 + N_ENERGY_TERMS) [kJ/mol]: Spalte 0 = E_kin, dann E_pot je\n"
          "Term in der Reihenfolge ENERGY_TERMS (aus dem letzten compute_forces/step).")
      .def(
          "degrees_of_freedom",
          [](SimulationContext& c) {
            return vector_to_numpy(c.degrees_of_freedom(),
                                   {static_cast<std::size_t>(c.n_replicas())});
          },
          "int32 (n_replicas,): N_dof = 3 * (Anzahl Atome mit inv_mass > 0).");

  // --- Observer ------------------------------------------------------------------------------
  nb::class_<bomd::EnergyTracker>(m, "EnergyTracker", "Energie-Zeitreihe im VRAM.")
      .def_prop_ro("stride", &bomd::EnergyTracker::stride)
      .def_prop_ro("n_frames", &bomd::EnergyTracker::n_frames)
      .def_prop_ro("dropped_frames", &bomd::EnergyTracker::dropped_frames)
      .def("reset", &bomd::EnergyTracker::reset)
      .def(
          "result",
          [](const bomd::EnergyTracker& t) {
            const bomd::ObserverResult res = t.download();
            const auto& e = res.arrays[0];
            std::vector<std::int64_t> steps;
            for (double st : res.arrays[1].data) steps.push_back(static_cast<std::int64_t>(st));
            nb::dict d;
            d["energies"] = vector_to_numpy(
                e.data, {static_cast<std::size_t>(e.shape[0]), static_cast<std::size_t>(e.shape[1]),
                         static_cast<std::size_t>(e.shape[2])});
            d["steps"] = vector_to_numpy(steps, {steps.size()});
            return d;
          },
          "dict: energies float64 (n_replicas, n_frames, 1 + N_ENERGY_TERMS) [kJ/mol],\n"
          "steps int64 (n_frames,).");
  cls.def(
      "add_energy_tracker",
      [](SimulationContext& c, int stride, int capacity) -> bomd::EnergyTracker& {
        return static_cast<bomd::EnergyTracker&>(
            c.add_observer(std::make_unique<bomd::EnergyTracker>(stride, capacity)));
      },
      "stride"_a, "capacity"_a, nb::rv_policy::reference_internal,
      "Registriert einen EnergyTracker (alle `stride` Schritte, höchstens `capacity` Frames).");

  // --- Kraft-Plugins -------------------------------------------------------------------------
  nb::class_<bomd::PlanarSurfaceForce>(m, "PlanarSurfaceForce",
                                       "9-3-Wand + Gouy-Chapman, PHYSICS_MODEL.md §4.1.")
      .def("clamp_count", &bomd::PlanarSurfaceForce::clamp_count,
           "Anzahl Auswertungen mit geclamptem Wandabstand.")
      .def("unknown_type_count", &bomd::PlanarSurfaceForce::unknown_type_count,
           "Anzahl Auswertungen mit Oberflächen-Typ außerhalb der Parametertabelle.")
      .def("reset_counters", &bomd::PlanarSurfaceForce::reset_counters);
  cls.def(
      "add_planar_surface",
      [](SimulationContext& c, HostInput<float> epsilon, HostInput<float> sigma, float z0,
         float clamp_factor, float phi0, float kappa) -> bomd::PlanarSurfaceForce& {
        if (epsilon.ndim() != 1 || sigma.ndim() != 1)
          throw nb::value_error("epsilon und sigma: 1D-Arrays (n_surface_types,)");
        bomd::PlanarSurfaceParams p{
            to_vector(epsilon), to_vector(sigma), z0, clamp_factor, phi0, kappa};
        return static_cast<bomd::PlanarSurfaceForce&>(
            c.add_force(std::make_unique<bomd::PlanarSurfaceForce>(std::move(p))));
      },
      "epsilon"_a, "sigma"_a, "z0"_a = 0.0f, "clamp_factor"_a = 0.1f, "phi0"_a = 0.0f,
      "kappa"_a = 0.0f, nb::rv_policy::reference_internal,
      "9-3-Wand bei z0 [nm]: epsilon [kJ/mol], sigma [nm] pro Oberflächen-Typ (types[..., 1]);\n"
      "optional Gouy-Chapman mit phi0 [kJ/(mol e)] und kappa [1/nm].");

  nb::class_<bomd::HarmonicRestraintPullingForce>(
      m, "HarmonicRestraintPullingForce", "Positions-Restraints / Schwerpunkt-Pulling, §4.2.")
      .def_prop_ro("is_pulling", &bomd::HarmonicRestraintPullingForce::is_pulling)
      .def(
          "work",
          [](const bomd::HarmonicRestraintPullingForce& f) {
            auto w = f.work();
            return vector_to_numpy(w, {w.size()});
          },
          "float64 (n_replicas,): akkumulierte Arbeit W [kJ/mol] (Pulling).")
      .def(
          "extension",
          [](const bomd::HarmonicRestraintPullingForce& f) {
            auto u = f.extension();
            return vector_to_numpy(u, {u.size()});
          },
          "float64 (n_replicas,): u = e.(R_G - R0) - v t [nm] aus dem letzten compute.")
      .def("reset_work", &bomd::HarmonicRestraintPullingForce::reset_work);
  cls.def(
      "add_position_restraints",
      [](SimulationContext& c, HostInput<int32_t> replica, HostInput<int32_t> atom,
         HostInput<float> reference, HostInput<float> k,
         std::array<bool, 3> dims) -> bomd::HarmonicRestraintPullingForce& {
        bomd::PositionRestraintParams p{to_vector(replica), to_vector(atom), to_vector(reference),
                                        to_vector(k), dims};
        return static_cast<bomd::HarmonicRestraintPullingForce&>(
            c.add_force(std::make_unique<bomd::HarmonicRestraintPullingForce>(std::move(p))));
      },
      "replica"_a, "atom"_a, "reference"_a, "k"_a, "dims"_a = std::array<bool, 3>{true, true, true},
      nb::rv_policy::reference_internal,
      "Harmonische Positions-Restraints: replica, atom (lokal) int32 (m,), reference [nm]\n"
      "float32 (m, 3), k [kJ/mol/nm^2] float32 (m,), dims = wirksame Achsen (x, y, z).");
  cls.def(
      "add_com_pulling",
      [](SimulationContext& c, HostInput<int32_t> offsets, HostInput<int32_t> atoms,
         HostInput<float> r0, HostInput<float> direction, HostInput<float> velocity,
         HostInput<float> k) -> bomd::HarmonicRestraintPullingForce& {
        bomd::ComPullingParams p{to_vector(offsets),   to_vector(atoms),    to_vector(r0),
                                 to_vector(direction), to_vector(velocity), to_vector(k)};
        return static_cast<bomd::HarmonicRestraintPullingForce&>(
            c.add_force(std::make_unique<bomd::HarmonicRestraintPullingForce>(std::move(p))));
      },
      "group_offsets"_a, "group_atoms"_a, "r0"_a, "direction"_a, "velocity"_a, "k"_a,
      nb::rv_policy::reference_internal,
      "Schwerpunkt-Pulling pro Replica: Gruppen als CSR (group_offsets (n_replicas+1,),\n"
      "group_atoms lokal), r0 [nm] und direction (n_replicas, 3), velocity [nm/ps] und\n"
      "k [kJ/mol/nm^2] (n_replicas,). Massen müssen vorher gesetzt sein.");

  // --- Test-Hooks ---------------------------------------------------------------------------
  auto testing = m.def_submodule("testing", "Test-Hooks für pytest; keine Produktions-API.");
  testing.def("shift_x", &bomd::testing::shift_x, "ctx"_a, "dx"_a,
              "Dummy-Kernel: x += dx [nm] für alle echten Atome (asynchron).");
  testing.def(
      "accumulate_forces",
      [](SimulationContext& c, HostInput<int32_t> atom, HostInput<float> f) {
        if (atom.ndim() != 1) throw nb::value_error("atom_index: erwartete Shape (m,)");
        check_shape(f, {atom.shape(0), 3}, "forces");
        bomd::testing::accumulate_forces(c, atom.data(), f.data(), static_cast<int>(atom.shape(0)));
      },
      "ctx"_a, "atom_index"_a, "forces"_a,
      "Addiert Kräfte [kJ/mol/nm], float32 (m, 3), per Fixed-Point-Atomics auf die globalen\n"
      "Atomindizes atom_index = r·n_pad + i (int32, (m,)). Löscht vorher nicht.");
  testing.def(
      "philox",
      [](HostInput<uint32_t> counters, uint32_t key_lo, uint32_t key_hi) {
        if (counters.ndim() != 2 || counters.shape(1) != 4)
          throw nb::value_error("counters: erwartete Shape (n, 4)");
        const std::size_t n = counters.shape(0);
        std::vector<uint32_t> out(4 * n);
        bomd::testing::philox(counters.data(), static_cast<int>(n), key_lo, key_hi, out.data());
        return vector_to_numpy(out, {n, 4});
      },
      "counters"_a, "key_lo"_a, "key_hi"_a,
      "Philox4x32-10 auf dem Gerät: uint32 (n, 4) -> uint32 (n, 4).");
  testing.def(
      "add_uniform_force",
      [](SimulationContext& c, float fx, float fy, float fz) {
        c.add_force(bomd::testing::make_uniform_force(fx, fy, fz));
      },
      "ctx"_a, "fx"_a, "fy"_a, "fz"_a,
      "Registriert ein Test-ForcePlugin mit konstanter Kraft [kJ/mol/nm] auf echte Atome.");
}
