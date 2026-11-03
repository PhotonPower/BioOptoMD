// nanobind-Modul biooptomd._core: SimulationContext, Array-Export (NumPy-Kopie und DLPack
// Zero-Copy) und Test-Hooks (_core.testing).
#include <cuda_runtime.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "biooptomd/cuda_check.hpp"
#include "biooptomd/device_state.hpp"
#include "biooptomd/simulation_context.hpp"
#include "biooptomd/units.hpp"
#include "core/testing.hpp"

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
      "add_uniform_force",
      [](SimulationContext& c, float fx, float fy, float fz) {
        c.add_force(bomd::testing::make_uniform_force(fx, fy, fz));
      },
      "ctx"_a, "fx"_a, "fy"_a, "fz"_a,
      "Registriert ein Test-ForcePlugin mit konstanter Kraft [kJ/mol/nm] auf echte Atome.");
}
