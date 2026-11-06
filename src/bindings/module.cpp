// nanobind-Modul biooptomd._core: SimulationContext, Array-Export (NumPy-Kopie und DLPack
// Zero-Copy) und Test-Hooks (_core.testing).
#include <cuda_runtime.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>

#include <algorithm>
#include <array>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "biooptomd/cuda_check.hpp"
#include "biooptomd/device_state.hpp"
#include "biooptomd/simulation_context.hpp"
#include "biooptomd/units.hpp"
#include "core/testing.hpp"
#include "forces/bonded.hpp"
#include "forces/harmonic_restraint.hpp"
#include "forces/image_charge.hpp"
#include "forces/planar_surface.hpp"
#include "forces/voxel_grid.hpp"
#include "nonbonded/generalized_born.hpp"
#include "nonbonded/tiled_all_pairs.hpp"
#include "observers/energy_tracker.hpp"
#include "observers/optics.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace {

using bomd::SimulationContext;

template <typename T>
using HostInput = nb::ndarray<const T, nb::c_contig, nb::device::cpu>;

using Shape = std::vector<std::size_t>;

/// tile_list-Option der Nonbonded-Strategien (ADR 0014).
bomd::TileListMode tile_list_mode(const std::string& mode) {
  if (mode == "auto") return bomd::TileListMode::kAuto;
  if (mode == "on") return bomd::TileListMode::kOn;
  if (mode == "off") return bomd::TileListMode::kOff;
  throw nb::value_error("tile_list: 'auto', 'on' oder 'off'");
}

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

/// dict[key] → std::vector<T>; fehlender Schlüssel ⇒ leer. Ganzzahlfelder müssen einen
/// Integer-dtype haben und in int32 passen (kein stilles Abschneiden/Überlaufen); Indexfelder
/// (arity > 0) müssen die Shape (n, arity) haben, Parameterfelder (n,).
template <typename T>
std::vector<T> dict_vector(const nb::dict& d, const char* key, int arity = 0) {
  if (!d.contains(key)) return {};
  nb::module_ np = nb::module_::import_("numpy");
  nb::object arr = np.attr("asarray")(d[key]);
  const std::string kind = nb::cast<std::string>(arr.attr("dtype").attr("kind"));
  const std::size_t size = nb::cast<std::size_t>(arr.attr("size"));
  const int ndim = nb::cast<int>(arr.attr("ndim"));
  const std::string where = std::string("Topologie '") + key + "': ";
  if (size > 0) {
    if (arity > 0) {
      if (ndim != 2 || nb::cast<int>(arr.attr("shape")[1]) != arity)
        throw nb::value_error(
            (where + "Shape (n, " + std::to_string(arity) + ") erwartet").c_str());
    } else if (ndim != 1) {
      throw nb::value_error((where + "1D-Array erwartet").c_str());
    }
  }
  if constexpr (std::is_floating_point_v<T>) {
    if (size > 0 && kind != "f" && kind != "i" && kind != "u")
      throw nb::value_error((where + "numerischer dtype erwartet").c_str());
    arr = np.attr("ascontiguousarray")(arr, "dtype"_a = "float32");
  } else {
    if (size > 0 && kind != "i" && kind != "u")
      throw nb::value_error((where + "Integer-dtype erwartet").c_str());
    if (size > 0 && (nb::cast<long long>(nb::int_(arr.attr("min")())) < INT32_MIN ||
                     nb::cast<long long>(nb::int_(arr.attr("max")())) > INT32_MAX))
      throw nb::value_error((where + "Werte außerhalb des int32-Bereichs").c_str());
    arr = np.attr("ascontiguousarray")(arr, "dtype"_a = "int32");
  }
  return to_vector(nb::cast<HostInput<T>>(arr));
}

/// Erlaubte Schlüssel eines Topologie-dicts; "charges" und "masses" (aus extract_topology)
/// werden akzeptiert und ignoriert. Unbekannte Schlüssel (Tippfehler) ⇒ ValueError.
bomd::BondedTopology topology_from_dict(const nb::dict& d) {
  static const std::set<std::string> known = {
      "bonds",      "bond_k",    "bond_r0",   "angles",        "angle_k", "angle_theta0",
      "torsions",   "torsion_k", "torsion_n", "torsion_phase", "pairs14", "pair_qq",
      "pair_sigma", "pair_eps",  "charges",   "masses"};
  for (auto [key, value] : d) {
    const std::string k = nb::cast<std::string>(key);
    if (!known.count(k))
      throw nb::value_error(("Topologie: unbekannter Schlüssel '" + k + "'").c_str());
  }
  bomd::BondedTopology t;
  t.bonds = dict_vector<int32_t>(d, "bonds", 2);
  t.bond_k = dict_vector<float>(d, "bond_k");
  t.bond_r0 = dict_vector<float>(d, "bond_r0");
  t.angles = dict_vector<int32_t>(d, "angles", 3);
  t.angle_k = dict_vector<float>(d, "angle_k");
  t.angle_theta0 = dict_vector<float>(d, "angle_theta0");
  t.torsions = dict_vector<int32_t>(d, "torsions", 4);
  t.torsion_k = dict_vector<float>(d, "torsion_k");
  t.torsion_n = dict_vector<int32_t>(d, "torsion_n");
  t.torsion_phase = dict_vector<float>(d, "torsion_phase");
  t.pairs14 = dict_vector<int32_t>(d, "pairs14", 2);
  t.pair_qq = dict_vector<float>(d, "pair_qq");
  t.pair_sigma = dict_vector<float>(d, "pair_sigma");
  t.pair_eps = dict_vector<float>(d, "pair_eps");
  return t;
}

/// Nonbonded-Topologie aus dict {lj_table (T,T,2), mask_index (n_tp,), masks (m,32)}.
bomd::NonbondedTopology nonbonded_topology_from_dict(const nb::dict& d) {
  static const std::set<std::string> known = {"lj_table", "mask_index", "masks"};
  for (auto [key, value] : d) {
    const std::string k = nb::cast<std::string>(key);
    if (!known.count(k))
      throw nb::value_error(("Nonbonded-Topologie: unbekannter Schlüssel '" + k + "'").c_str());
    (void)value;
  }
  for (const char* k : {"lj_table", "mask_index", "masks"})
    if (!d.contains(k))
      throw nb::value_error((std::string("Nonbonded-Topologie: Schlüssel fehlt: ") + k).c_str());
  nb::module_ np = nb::module_::import_("numpy");
  nb::object table = np.attr("asarray")(d["lj_table"]);
  const nb::tuple shape = nb::cast<nb::tuple>(table.attr("shape"));
  if (nb::len(shape) != 3 || nb::cast<int>(shape[0]) != nb::cast<int>(shape[1]) ||
      nb::cast<int>(shape[2]) != 2)
    throw nb::value_error("lj_table: Shape (T, T, 2) erwartet");
  bomd::NonbondedTopology t;
  t.n_types = nb::cast<int>(shape[0]);
  t.lj_table = to_vector(
      nb::cast<HostInput<float>>(np.attr("ascontiguousarray")(table, "dtype"_a = "float32")));
  t.mask_index = to_vector(nb::cast<HostInput<int32_t>>(
      np.attr("ascontiguousarray")(d["mask_index"], "dtype"_a = "int32")));
  nb::object masks = np.attr("ascontiguousarray")(d["masks"], "dtype"_a = "uint32");
  t.masks = to_vector(nb::cast<HostInput<uint32_t>>(masks));
  return t;
}

/// ObserverResult → dict {name: numpy-Array mit der gespeicherten Shape}.
nb::dict result_to_dict(const bomd::ObserverResult& res) {
  nb::dict d;
  for (const auto& a : res.arrays) {
    Shape shape;
    for (std::int64_t s : a.shape) shape.push_back(static_cast<std::size_t>(s));
    d[a.name.c_str()] = vector_to_numpy(a.data, shape);
  }
  return d;
}

bomd::DipoleProbeParams probe_params(const HostInput<int32_t>& offsets,
                                     const HostInput<int32_t>& atoms, const HostInput<int32_t>& a,
                                     const HostInput<int32_t>& b) {
  return bomd::DipoleProbeParams{to_vector(offsets), to_vector(atoms), to_vector(a), to_vector(b)};
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
  m.attr("STATUS_FIXED_POINT_SATURATED") =
      static_cast<unsigned int>(bomd::kStatusFixedPointSaturated);
  m.attr("STATUS_NON_FINITE") = static_cast<unsigned int>(bomd::kStatusNonFinite);
  m.attr("STATUS_WRAP_RANGE") = static_cast<unsigned int>(bomd::kStatusWrapRange);
  m.attr("STATUS_UNKNOWN_TYPE") = static_cast<unsigned int>(bomd::kStatusUnknownType);
  m.attr("STATUS_CONSTRAINT_FAILURE") = static_cast<unsigned int>(bomd::kStatusConstraintFailure);

  // Kompilierte Konstanten aus units.hpp (Test: test_units.py).
  auto units = m.def_submodule("units", "Konstanten aus include/biooptomd/units.hpp.");
  units.attr("ONE_4PI_EPS0") = bomd::units::ONE_4PI_EPS0;
  units.attr("KB") = bomd::units::KB;
  units.attr("FARADAY") = bomd::units::FARADAY;
  units.attr("DEBYE_PER_ENM") = bomd::units::DEBYE_PER_ENM;
  units.attr("FIXED_POINT_SCALE") = bomd::units::FIXED_POINT_SCALE;
  units.attr("EPS_R2") = static_cast<double>(bomd::units::EPS_R2);
  units.attr("EPS_CROSS2") = static_cast<double>(bomd::units::EPS_CROSS2);
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
          [](SimulationContext& c, HostInput<double> a) {
            check_shape(a, real_atoms_shape(c, 3), "velocities");
            c.set_velocities(a.data());
          },
          "velocities"_a, "Geschwindigkeiten [nm/ps], float64 (n_replicas, n_atoms, 3).")
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
      .def("status", &SimulationContext::status,
           "Geräte-Statuswort (Bits STATUS_*, ADR 0008); 0 = in Ordnung. Synchronisiert.")
      .def("clear_status", &SimulationContext::clear_status, "Statuswort auf 0 setzen.")
      .def("synchronize", &SimulationContext::synchronize,
           "Wartet auf alle Arbeit im Stream des Kontexts.");

  // --- Exporte (gepaddetes Rohlayout) -----------------------------------------------------
  using bomd::DeviceState;
  const auto atoms4 = [](const SimulationContext& c) { return atoms_shape(c, 4); };
  const auto atoms2 = [](const SimulationContext& c) { return atoms_shape(c, 2); };
  def_export<float>(
      cls, "positions", [](const DeviceState& s) { return &s.d_pos->x; }, atoms4,
      "float32 (n_replicas, n_pad, 4): x, y, z [nm], q [e].");
  def_export<double>(
      cls, "velocities", [](const DeviceState& s) { return &s.d_vel->x; }, atoms4,
      "float64 (n_replicas, n_pad, 4): vx, vy, vz [nm/ps], 1/m [1/amu] (ADR 0009).");
  def_export<float>(
      cls, "position_corrections", [](const DeviceState& s) { return &s.d_pos_corr->x; }, atoms4,
      "float32 (n_replicas, n_pad, 4): Korrektur zu positions() (ADR 0009), w = 0.");
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
  for (const char* name : {"bond", "angle", "torsion", "pair14", "nonbonded", "surface",
                           "restraint", "external", "image"})
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
      .def_prop_ro(
          "integrator_params",
          [](const SimulationContext& c) {
            const auto& p = c.integrator_params();
            nb::dict d;
            d["dt"] = p.dt_ps;
            d["friction"] = p.friction_per_ps;
            d["seed"] = p.seed;
            return d;
          },
          "dict dt [ps], friction [1/ps], seed; RuntimeError ohne Integrator.")
      .def(
          "save_state",
          [](SimulationContext& c) {
            const bomd::DynamicState st = c.save_dynamic_state();
            const std::size_t r = static_cast<std::size_t>(c.n_replicas());
            const std::size_t n = static_cast<std::size_t>(c.n_pad());
            std::vector<float> pos(4 * st.pos.size()), corr(4 * st.pos.size());
            std::vector<double> vel(4 * st.vel.size());
            std::vector<int32_t> image(2 * st.image.size());
            std::memcpy(pos.data(), st.pos.data(), pos.size() * sizeof(float));
            std::memcpy(corr.data(), st.pos_corr.data(), corr.size() * sizeof(float));
            std::memcpy(vel.data(), st.vel.data(), vel.size() * sizeof(double));
            std::memcpy(image.data(), st.image.data(), image.size() * sizeof(int32_t));
            nb::dict d;
            d["pos"] = vector_to_numpy(pos, {r, n, 4});
            d["pos_corr"] = vector_to_numpy(corr, {r, n, 4});
            d["vel"] = vector_to_numpy(vel, {r, n, 4});
            d["image"] = vector_to_numpy(image, {r, n, 2});
            d["step"] = st.step;
            d["time"] = st.time_ps;
            d["previous_time"] = st.previous_time_ps;
            return d;
          },
          "Dynamischer Zustand (bitgenau, inkl. Padding): pos float32 (n_replicas, n_pad, 4) =\n"
          "(x, y, z [nm], q [e]), pos_corr float32 (…, 4) [nm], vel float64 (…, 4) = (v [nm/ps],\n"
          "1/m [1/amu]), image int32 (…, 2), step, time [ps], previous_time [ps]. Für Checkpoints\n"
          "siehe biooptomd.io.")
      .def(
          "restore_state",
          [](SimulationContext& c, HostInput<float> pos, HostInput<float> pos_corr,
             HostInput<double> vel, HostInput<int32_t> image, std::int64_t step, double time,
             double previous_time) {
            const std::size_t r = static_cast<std::size_t>(c.n_replicas());
            const std::size_t n = static_cast<std::size_t>(c.n_pad());
            check_shape(pos, {r, n, 4}, "pos");
            check_shape(pos_corr, {r, n, 4}, "pos_corr");
            check_shape(vel, {r, n, 4}, "vel");
            check_shape(image, {r, n, 2}, "image");
            bomd::DynamicState st;
            st.pos.resize(r * n);
            st.pos_corr.resize(r * n);
            st.vel.resize(r * n);
            st.image.resize(r * n);
            std::memcpy(st.pos.data(), pos.data(), pos.size() * sizeof(float));
            std::memcpy(st.pos_corr.data(), pos_corr.data(), pos_corr.size() * sizeof(float));
            std::memcpy(st.vel.data(), vel.data(), vel.size() * sizeof(double));
            std::memcpy(st.image.data(), image.data(), image.size() * sizeof(int32_t));
            st.step = step;
            st.time_ps = time;
            st.previous_time_ps = previous_time;
            c.restore_dynamic_state(st);
          },
          "pos"_a, "pos_corr"_a, "vel"_a, "image"_a, "step"_a, "time"_a, "previous_time"_a,
          "Stellt einen mit save_state gesicherten Zustand bitgenau wieder her (Box, "
          "Temperaturen,\n"
          "Integrator und Plugins vorher setzen; set_box setzt die Imagezähler zurück).")
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
          "kinetic_energy_mid",
          [](SimulationContext& c) {
            return vector_to_numpy(c.kinetic_energy_mid(),
                                   {static_cast<std::size_t>(c.n_replicas())});
          },
          "float64 (n_replicas,) [kJ/mol]: E_kin der Geschwindigkeiten nach dem O-Schritt "
          "(+RATTLE)\n"
          "des letzten step(); NaN vor dem ersten Schritt und nach set_velocities/restore_state;\n"
          "observe_now() ändert ihn nicht. Bei großem dt unverzerrt, anders als\n"
          "E_kin am vollen Schritt (OQ-14, PHYSICS_MODEL §1). T = 2 E / (N_dof k_B).")
      .def(
          "positions_precise",
          [](SimulationContext& c) {
            return vector_to_numpy(c.positions_precise(), {static_cast<std::size_t>(c.n_replicas()),
                                                           static_cast<std::size_t>(c.n_pad()), 3});
          },
          "float64 (n_replicas, n_pad, 3) [nm]: positions + position_corrections (ADR 0009).")
      .def(
          "degrees_of_freedom",
          [](SimulationContext& c) {
            return vector_to_numpy(c.degrees_of_freedom(),
                                   {static_cast<std::size_t>(c.n_replicas())});
          },
          "int32 (n_replicas,): N_dof = 3 * (Anzahl Atome mit inv_mass > 0) - Anzahl Constraints.");

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
            const auto& mid = res.arrays[2];
            d["kinetic_mid"] = vector_to_numpy(mid.data, {static_cast<std::size_t>(mid.shape[0]),
                                                          static_cast<std::size_t>(mid.shape[1])});
            return d;
          },
          "dict: energies float64 (n_replicas, n_frames, 1 + N_ENERGY_TERMS) [kJ/mol],\n"
          "steps int64 (n_frames,), kinetic_mid float64 (n_replicas, n_frames) [kJ/mol]: E_kin\n"
          "nach dem O-Schritt im beobachteten Schritt (OQ-14).");
  cls.def(
      "add_energy_tracker",
      [](SimulationContext& c, int stride, int capacity) -> bomd::EnergyTracker& {
        return static_cast<bomd::EnergyTracker&>(
            c.add_observer(std::make_unique<bomd::EnergyTracker>(stride, capacity)));
      },
      "stride"_a, "capacity"_a, nb::rv_policy::reference_internal,
      "Registriert einen EnergyTracker (alle `stride` Schritte, höchstens `capacity` Frames).");

  // --- Kraft-Plugins -------------------------------------------------------------------------
  nb::class_<bomd::PlanarSurfaceForce>(
      m, "PlanarSurfaceForce", "Wand (9-3/10-4/10-4-3) + Gouy-Chapman, PHYSICS_MODEL.md §4.1.")
      .def("clamp_count", &bomd::PlanarSurfaceForce::clamp_count,
           "Anzahl Auswertungen mit geclamptem Wandabstand.")
      .def("unknown_type_count", &bomd::PlanarSurfaceForce::unknown_type_count,
           "Anzahl Auswertungen mit Oberflächen-Typ außerhalb der Parametertabelle.")
      .def("reset_counters", &bomd::PlanarSurfaceForce::reset_counters)
      .def(
          "clamp_distances",
          [](const bomd::PlanarSurfaceForce& f) {
            const auto& z = f.clamp_distances();
            const std::size_t n_lj = static_cast<std::size_t>(f.params().n_lj_types);
            return vector_to_numpy(z, {z.size() / n_lj, n_lj});
          },
          "float32 (n_surface_types, n_lj_types): zeta_c = max(clamp_factor*sigma, zeta_F) [nm],\n"
          "ADR 0008.");
  cls.def(
      "add_planar_surface",
      [](SimulationContext& c, HostInput<float> prefactor, HostInput<float> sigma,
         const std::string& form, float delta, float z0, float clamp_factor, float phi0,
         std::optional<HostInput<float>> kappa, float max_force,
         const std::string& gouy_chapman) -> bomd::PlanarSurfaceForce& {
        if (prefactor.ndim() < 1 || prefactor.ndim() > 2)
          throw nb::value_error("prefactor: (n_surface_types,) oder (n_surface_types, n_lj_types)");
        if (sigma.ndim() != prefactor.ndim() || sigma.shape(0) != prefactor.shape(0) ||
            (sigma.ndim() == 2 && sigma.shape(1) != prefactor.shape(1)))
          throw nb::value_error("sigma braucht dieselbe Shape wie prefactor");
        bomd::PlanarSurfaceParams p;
        if (form == "9-3") {
          p.form = bomd::WallForm::kWall93;
        } else if (form == "10-4") {
          p.form = bomd::WallForm::kWall104;
        } else if (form == "10-4-3") {
          p.form = bomd::WallForm::kWall1043;
        } else {
          throw nb::value_error("form: '9-3', '10-4' oder '10-4-3'");
        }
        if (gouy_chapman != "nonlinear" && gouy_chapman != "linear")
          throw nb::value_error("gouy_chapman: 'nonlinear' oder 'linear'");
        p.n_lj_types = prefactor.ndim() == 2 ? static_cast<int>(prefactor.shape(1)) : 1;
        p.prefactor_kj_mol = to_vector(prefactor);
        p.sigma_nm = to_vector(sigma);
        p.delta_nm = delta;
        p.z0_nm = z0;
        p.clamp_factor = clamp_factor;
        p.max_force_kj_mol_nm = max_force;
        p.phi0_kj_mol_e = phi0;
        if (kappa) {
          check_shape(*kappa, {static_cast<std::size_t>(c.n_replicas())}, "kappa");
          p.kappa_per_nm = to_vector(*kappa);
        }
        p.gouy_chapman_linear = gouy_chapman == "linear";
        return static_cast<bomd::PlanarSurfaceForce&>(
            c.add_force(std::make_unique<bomd::PlanarSurfaceForce>(std::move(p))));
      },
      "prefactor"_a, "sigma"_a, "form"_a = "9-3", "delta"_a = 0.335f, "z0"_a = 0.0f,
      "clamp_factor"_a = 0.1f, "phi0"_a = 0.0f, "kappa"_a = nb::none(), "max_force"_a = 1.0e6f,
      "gouy_chapman"_a = "nonlinear", nb::rv_policy::reference_internal,
      "Wand bei z0 [nm] (ADR 0007): form '9-3' | '10-4' | '10-4-3', Vorfaktor C [kJ/mol] und\n"
      "sigma [nm] je (Oberflächen-Typ types[..., 1], LJ-Typ types[..., 0]); 1D-Arrays\n"
      "(n_surface_types,) gelten für alle LJ-Typen. Tabellen baut biooptomd.surface.wall_table.\n"
      "delta [nm]: Lagenabstand (nur 10-4-3). Clamp bei zeta_c = max(clamp_factor*sigma,\n"
      "zeta_F) mit |F(zeta_F)| = max_force [kJ/mol/nm]. Optional Gouy-Chapman (OQ-06): phi0 =\n"
      "Potential der diffusen Schicht [kJ/(mol e)] (nicht das Elektrodenpotential), kappa [1/nm]\n"
      "je Replica (n_replicas,), T_r aus set_temperatures; gouy_chapman='linear' nur als Test.");

  nb::class_<bomd::ImageChargeForce>(m, "ImageChargeForce",
                                     "Abgeschirmte Bildladungen an z = z_image (ADR 0013).")
      .def_prop_ro("uses_image_sum", &bomd::ImageChargeForce::uses_image_sum,
                   "True, wenn über periodische Bilder summiert wird (Box < 2 r_c).")
      .def("clamp_count", &bomd::ImageChargeForce::clamp_count,
           "Anzahl geladener Atome mit z - z_image < 1.5*zeta_clamp (C1-glattes Clamping,\n"
           "Übergang zeta_clamp +- zeta_clamp/2), summiert über Replicas und Kraftauswertungen.")
      .def("reset_counters", &bomd::ImageChargeForce::reset_counters);
  cls.def(
      "add_image_charges",
      [](SimulationContext& c, float z_image, float factor, HostInput<float> kappa,
         HostInput<float> eps_r, float r_switch, float r_cutoff, float zeta_clamp,
         bool image_sum) -> bomd::ImageChargeForce& {
        const std::size_t n_rep = static_cast<std::size_t>(c.n_replicas());
        check_shape(kappa, {n_rep}, "kappa");
        check_shape(eps_r, {n_rep}, "eps_r");
        bomd::ImageChargeParams p;
        p.z_image_nm = z_image;
        p.factor = factor;
        p.kappa_per_nm = to_vector(kappa);
        p.eps_r = to_vector(eps_r);
        p.r_switch_nm = r_switch;
        p.r_cutoff_nm = r_cutoff;
        p.zeta_clamp_nm = zeta_clamp;
        p.always_image_sum = image_sum;
        return static_cast<bomd::ImageChargeForce&>(
            c.add_force(std::make_unique<bomd::ImageChargeForce>(std::move(p))));
      },
      "z_image"_a, "factor"_a, "kappa"_a, "eps_r"_a, "r_switch"_a, "r_cutoff"_a,
      "zeta_clamp"_a = 0.1f, "image_sum"_a = false, nb::rv_policy::reference_internal,
      "Bildladungen an der Ebene z_image [nm] (ADR 0013): U = -f [sum_{i<j} q_i q_j phi(s_ij)\n"
      "+ 1/2 sum_i q_i^2 phi(2 zeta_i)], phi = Debye-Hückel mit kappa [1/nm], eps_r je Replica\n"
      "(n_replicas,) und Switching r_switch..r_cutoff [nm] (Werte der Nonbonded-Strategie).\n"
      "factor f: 1 = idealer Leiter, sonst Näherung (OQ-18). zeta = C1-glattes max(z - z_image,\n"
      "zeta_clamp). image_sum=True erzwingt den Bildsummenpfad (Tests). Energie-Term 'image'.");

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

  nb::class_<bomd::BondedForce>(m, "BondedForce",
                                "Bonds, Angles, Torsionen, 1-4-Paare, PHYSICS_MODEL.md §2.")
      .def("term_counts", &bomd::BondedForce::term_counts, "replica"_a,
           "[n_bonds, n_angles, n_torsions, n_pairs14] der Replica.");
  cls.def(
      "add_bonded",
      [](SimulationContext& c, nb::list topologies, HostInput<int32_t> replica_topology,
         HostInput<float> kappa, HostInput<float> eps_r,
         bool follow_screening) -> bomd::BondedForce& {
        bomd::BondedParams p;
        p.follow_screening = follow_screening;
        for (nb::handle h : topologies)
          p.topologies.push_back(topology_from_dict(nb::cast<nb::dict>(h)));
        p.replica_topology = to_vector(replica_topology);
        p.kappa_per_nm = to_vector(kappa);
        p.eps_r = to_vector(eps_r);
        return static_cast<bomd::BondedForce&>(
            c.add_force(std::make_unique<bomd::BondedForce>(std::move(p))));
      },
      "topologies"_a, "replica_topology"_a, "kappa"_a, "eps_r"_a, "follow_screening"_a = true,
      nb::rv_policy::reference_internal,
      "Kovalente Terme. topologies: Liste von dicts mit bonds (n,2), bond_k [kJ/mol/nm^2],\n"
      "bond_r0 [nm], angles (n,3), angle_k [kJ/mol/rad^2], angle_theta0 [rad], torsions (n,4),\n"
      "torsion_k [kJ/mol], torsion_n, torsion_phase [rad], pairs14 (n,2), pair_qq [e^2],\n"
      "pair_sigma [nm], pair_eps [kJ/mol] (lokale Atomindizes). replica_topology int32\n"
      "(n_replicas,), kappa [1/nm] und eps_r (n_replicas,) für die 1-4-Debye-Hückel-Abschirmung;\n"
      "follow_screening=False hält sie bei set_screening fest (GB: kappa = 0, eps_r = 1).");

  nb::class_<bomd::TiledAllPairs>(m, "TiledAllPairs",
                                  "LJ + Debye-Hückel über alle Tile-Paare, PHYSICS_MODEL.md §3.")
      .def_prop_ro("n_tile_pairs", &bomd::TiledAllPairs::n_tile_pairs);
  cls.def(
      "set_nonbonded_tiled",
      [](SimulationContext& c, nb::list topologies, HostInput<int32_t> replica_topology,
         HostInput<float> kappa, HostInput<float> eps_r, float r_switch, float r_cutoff,
         const std::string& image_sum, const std::string& tile_list, int max_list_blocks) {
        if (image_sum != "auto" && image_sum != "always")
          throw nb::value_error("image_sum: 'auto' oder 'always'");
        bomd::TiledAllPairsParams p;
        for (nb::handle h : topologies)
          p.topologies.push_back(nonbonded_topology_from_dict(nb::cast<nb::dict>(h)));
        p.replica_topology = to_vector(replica_topology);
        p.kappa_per_nm = to_vector(kappa);
        p.eps_r = to_vector(eps_r);
        p.r_switch_nm = r_switch;
        p.r_cutoff_nm = r_cutoff;
        p.always_image_sum = image_sum == "always";
        p.tile_list = tile_list_mode(tile_list);
        p.max_list_blocks = max_list_blocks;
        c.set_nonbonded(std::make_unique<bomd::TiledAllPairs>(std::move(p)));
      },
      "topologies"_a, "replica_topology"_a, "kappa"_a, "eps_r"_a, "r_switch"_a = 1.8f,
      "r_cutoff"_a = 2.0f, "image_sum"_a = "auto", "tile_list"_a = "auto", "max_list_blocks"_a = 0,
      "Nonbonded (Tiled All-Pairs). topologies: Liste von dicts aus\n"
      "biooptomd.nonbonded.nonbonded_topology(); replica_topology int32 (n_replicas,),\n"
      "kappa [1/nm], eps_r (n_replicas,), r_switch/r_cutoff [nm] (biooptomd.nonbonded.\n"
      "cutoff_from_kappa, ADR 0006). LJ-Typen aus types[..., 0]. Bei L_x oder L_y < 2 r_c\n"
      "Summe über periodische Bilder; image_sum='always' erzwingt sie (Test).\n"
      "tile_list: 'auto' (Liste nur bei großer Box), 'on' oder 'off' (ADR 0014; Ergebnis\n"
      "bitidentisch).\n"
      "Ersetzt eine zuvor gesetzte Nonbonded-Strategie (daher kein zurückgegebenes Handle).");

  cls.def(
      "set_screening",
      [](SimulationContext& c, HostInput<float> kappa, HostInput<float> eps_r) {
        const std::size_t n = static_cast<std::size_t>(c.n_replicas());
        check_shape(kappa, {n}, "kappa");
        check_shape(eps_r, {n}, "eps_r");
        c.set_screening(kappa.data(), eps_r.data());
      },
      "kappa"_a, "eps_r"_a,
      "Lösungsmittel-Abschirmung je Replica neu setzen (Replica Exchange, ADR 0011): kappa "
      "[1/nm],\n"
      "eps_r (bei GB eps_out), float32 (n_replicas,). Wirkt auf Nonbonded, 1-4 (falls\n"
      "follow_screening) und Gouy-Chapman; r_c bleibt.");
  cls.def(
      "scale_velocities",
      [](SimulationContext& c, HostInput<double> factor) {
        check_shape(factor, {static_cast<std::size_t>(c.n_replicas())}, "factor");
        c.scale_velocities(factor.data());
      },
      "factor"_a, "v *= factor[r] für alle echten Atome der Replica r, float64 (n_replicas,) > 0.");
  cls.def(
      "set_nonbonded_gb",
      [](SimulationContext& c, nb::list topologies, HostInput<int32_t> replica_topology,
         HostInput<float> kappa, HostInput<float> eps_out, nb::list radius, nb::list scale,
         float r_switch, float r_cutoff, float eps_in, bool ace, const std::string& image_sum,
         float offset, nb::object alpha_beta_gamma, nb::object neck_index, nb::object neck_m0,
         nb::object neck_d0, const std::string& tile_list, int max_list_blocks) {
        if (image_sum != "auto" && image_sum != "always")
          throw nb::value_error("image_sum: 'auto' oder 'always'");
        if (nb::len(radius) != nb::len(topologies) || nb::len(scale) != nb::len(topologies))
          throw nb::value_error("radius und scale: je Topologie ein Array (n_atoms,)");
        bomd::GeneralizedBornParams p;
        for (nb::handle h : topologies)
          p.base.topologies.push_back(nonbonded_topology_from_dict(nb::cast<nb::dict>(h)));
        p.base.replica_topology = to_vector(replica_topology);
        p.base.kappa_per_nm = to_vector(kappa);
        p.base.eps_r = to_vector(eps_out);
        p.base.r_switch_nm = r_switch;
        p.base.r_cutoff_nm = r_cutoff;
        p.base.always_image_sum = image_sum == "always";
        p.base.tile_list = tile_list_mode(tile_list);
        p.base.max_list_blocks = max_list_blocks;
        nb::module_ np = nb::module_::import_("numpy");
        for (std::size_t t = 0; t < nb::len(radius); ++t) {
          p.radius_nm.push_back(to_vector(nb::cast<HostInput<float>>(
              np.attr("ascontiguousarray")(radius[t], "dtype"_a = "float32"))));
          p.scale.push_back(to_vector(nb::cast<HostInput<float>>(
              np.attr("ascontiguousarray")(scale[t], "dtype"_a = "float32"))));
        }
        p.eps_in = eps_in;
        p.ace = ace;
        p.offset_nm = offset;
        const std::size_t n_topo = nb::len(topologies);
        const auto per_topo = [&](nb::object list, const char* name) {
          if (list.is_none()) return nb::list();
          nb::list l = nb::cast<nb::list>(list);
          if (nb::len(l) != n_topo)
            throw nb::value_error((std::string(name) + ": je Topologie ein Array").c_str());
          return l;
        };
        nb::list abg = per_topo(alpha_beta_gamma, "alpha_beta_gamma");
        for (nb::handle a : abg) {
          nb::object arr = np.attr("ascontiguousarray")(a, "dtype"_a = "float32");
          if (nb::cast<int>(arr.attr("ndim")) != 2 || nb::cast<int>(arr.attr("shape")[1]) != 3)
            throw nb::value_error("alpha_beta_gamma: Shape (n_atoms, 3) je Topologie");
          p.alpha_beta_gamma.push_back(to_vector(nb::cast<HostInput<float>>(arr)));
        }
        nb::list ni = per_topo(neck_index, "neck_index");
        nb::list m0 = per_topo(neck_m0, "neck_m0");
        nb::list d0 = per_topo(neck_d0, "neck_d0");
        if (nb::len(ni) != nb::len(m0) || nb::len(ni) != nb::len(d0))
          throw nb::value_error("neck_index, neck_m0, neck_d0 nur gemeinsam");
        for (std::size_t t = 0; t < nb::len(ni); ++t) {
          nb::object table = np.attr("ascontiguousarray")(m0[t], "dtype"_a = "float32");
          nb::object table_d = np.attr("ascontiguousarray")(d0[t], "dtype"_a = "float32");
          const int nd = nb::cast<int>(table.attr("ndim"));
          if (nd != 2 || nb::cast<int>(table_d.attr("ndim")) != 2 ||
              nb::cast<int>(table.attr("shape")[0]) != nb::cast<int>(table.attr("shape")[1]) ||
              nb::cast<int>(table_d.attr("shape")[0]) != nb::cast<int>(table.attr("shape")[0]) ||
              nb::cast<int>(table_d.attr("shape")[1]) != nb::cast<int>(table.attr("shape")[1]))
            throw nb::value_error("neck_m0, neck_d0: quadratische Tabellen (U, U) gleicher Shape");
          nb::object idx = np.attr("asarray")(ni[t]);
          const std::string kind = nb::cast<std::string>(idx.attr("dtype").attr("kind"));
          if (kind != "i" && kind != "u")
            throw nb::value_error("neck_index: Integer-dtype erwartet");
          const int u = nb::cast<int>(table.attr("shape")[0]);
          p.neck_size.push_back(u);
          p.neck_m0.push_back(to_vector(nb::cast<HostInput<float>>(table)));
          p.neck_d0.push_back(to_vector(nb::cast<HostInput<float>>(table_d)));
          p.neck_index.push_back(to_vector(nb::cast<HostInput<int32_t>>(
              np.attr("ascontiguousarray")(idx, "dtype"_a = "int32"))));
        }
        c.set_nonbonded(std::make_unique<bomd::TiledGeneralizedBorn>(std::move(p)));
      },
      "topologies"_a, "replica_topology"_a, "kappa"_a, "eps_out"_a, "radius"_a, "scale"_a,
      "r_switch"_a = 1.8f, "r_cutoff"_a = 2.0f, "eps_in"_a = 1.0f, "ace"_a = true,
      "image_sum"_a = "auto", "offset"_a = 0.009f, "alpha_beta_gamma"_a = nb::none(),
      "neck_index"_a = nb::none(), "neck_m0"_a = nb::none(), "neck_d0"_a = nb::none(),
      "tile_list"_a = "auto", "max_list_blocks"_a = 0,
      "Nonbonded mit Generalized Born OBC2 (ADR 0010) oder GBn2 (ADR 0012): LJ + Coulomb\n"
      "(eps_in) + GB mit Salz.\n"
      "topologies wie set_nonbonded_tiled; kappa [1/nm] und eps_out (n_replicas,); radius rho "
      "[nm]\n"
      "und scale s je Topologie (n_atoms,), aus biooptomd.builder (implicit/obc2.xml); r_switch,\n"
      "r_cutoff [nm] (cutoff_from_kappa); ace = unpolarer ACE-Term. 1-4-Coulomb im Bonded-Kernel\n"
      "unabgeschirmt setzen (kappa = 0, eps_r = eps_in). GB-neck2 (ADR 0012): offset [nm],\n"
      "alpha_beta_gamma (n_atoms, 3), neck_index (n_atoms,), neck_m0 [1/nm] und neck_d0 [nm]\n"
      "(U, U) je Topologie (biooptomd.builder aus implicit/gbn2.xml).");
  cls.def(
      "nonbonded_born_radii",
      [](SimulationContext& c) {
        auto* gb = dynamic_cast<bomd::TiledGeneralizedBorn*>(c.nonbonded());
        if (gb == nullptr) throw nb::value_error("keine GB-Nonbonded-Strategie gesetzt");
        return vector_to_numpy(gb->born_radii(), {static_cast<std::size_t>(c.n_replicas()),
                                                  static_cast<std::size_t>(c.n_pad())});
      },
      "Born-Radien B [nm] der letzten Kraftberechnung, float32 (n_replicas, n_pad), Padding 0.");

  cls.def(
         "set_constraints",
         [](SimulationContext& c, nb::list topologies, HostInput<int32_t> replica_topology,
            double tolerance, int max_iterations) {
           bomd::ConstraintParams p;
           nb::module_ np = nb::module_::import_("numpy");
           for (nb::handle h : topologies) {
             const nb::dict d = nb::cast<nb::dict>(h);
             for (auto [key, value] : d) {
               const std::string k = nb::cast<std::string>(key);
               if (k != "clusters" && k != "distances")
                 throw nb::value_error(
                     ("Constraint-Topologie: unbekannter Schlüssel '" + k + "'").c_str());
               (void)value;
             }
             nb::object cl = np.attr("asarray")(d["clusters"]);
             nb::object di = np.attr("asarray")(d["distances"]);
             if (nb::cast<std::string>(cl.attr("dtype").attr("kind")) != "i")
               throw nb::value_error("clusters: Integer-dtype erwartet");
             bomd::ConstraintTopology t;
             t.clusters = to_vector(nb::cast<HostInput<int32_t>>(
                 np.attr("ascontiguousarray")(cl, "dtype"_a = "int32")));
             t.distances = to_vector(nb::cast<HostInput<float>>(
                 np.attr("ascontiguousarray")(di, "dtype"_a = "float32")));
             p.topologies.push_back(std::move(t));
           }
           p.replica_topology = to_vector(replica_topology);
           p.tolerance = tolerance;
           p.max_iterations = max_iterations;
           c.set_constraints(std::move(p));
         },
         "topologies"_a, "replica_topology"_a, "tolerance"_a = 1e-6, "max_iterations"_a = 200,
         "H-Bindungs-Constraints (SHAKE/RATTLE). topologies: Liste von dicts {clusters int32\n"
         "(n,4) (Schweratom, H1..H3, -1 = fehlt), distances float32 (n,3) [nm]} aus\n"
         "biooptomd.constraints.constraint_topology(); tolerance = relative Längentoleranz.")
      .def("apply_constraints", &SimulationContext::apply_constraints,
           "Projiziert Positionen und Geschwindigkeiten sofort auf die Constraints.")
      .def_prop_ro("has_constraints", &SimulationContext::has_constraints)
      .def_prop_ro("forces_valid", &SimulationContext::forces_valid,
                   "true, wenn Kräfte und Energien zum aktuellen Zustand gehören.")
      .def_prop_ro(
          "nonbonded_cutoff",
          [](SimulationContext& c) {
            auto* t = dynamic_cast<bomd::TiledAllPairs*>(c.nonbonded());
            return t == nullptr ? 0.0f : t->params().r_cutoff_nm;
          },
          "r_c [nm] der Tile-Nonbonded-Strategie (0 ohne).");
  cls.def_prop_ro(
      "nonbonded_uses_tile_list",
      [](SimulationContext& c) {
        auto* t = dynamic_cast<bomd::TiledAllPairs*>(c.nonbonded());
        return t != nullptr && !t->uses_image_sum() && t->uses_tile_list();
      },
      "True, wenn die Tile-Nonbonded-Strategie die Tile-Paar-Liste nutzt (ADR 0014).");
  cls.def(
      "nonbonded_tile_list_counts",
      [](SimulationContext& c) {
        auto* t = dynamic_cast<bomd::TiledAllPairs*>(c.nonbonded());
        std::vector<int> v = t == nullptr ? std::vector<int>{} : t->tile_list_counts();
        return vector_to_numpy(v, {v.size()});
      },
      "int32 (n_replicas,): Länge der Tile-Paar-Liste aus der letzten Auswertung mit Liste\n"
      "(Diagnose, synchronisiert); leer bei tile_list='off'.");

  // --- Optik-Observer (Phase 6) ---------------------------------------------------------------
  cls.def("observe_now", &SimulationContext::observe_now,
          "Führt alle Observer einmal am aktuellen Zustand aus (asynchron).");
  nb::class_<bomd::DipoleAndDistanceTracker>(m, "DipoleAndDistanceTracker",
                                             "R_F, μ̂, d je Frame, PHYSICS_MODEL.md §6.1.")
      .def_prop_ro("overwritten", &bomd::DipoleAndDistanceTracker::overwritten)
      .def("reset", &bomd::DipoleAndDistanceTracker::reset)
      .def(
          "result",
          [](const bomd::DipoleAndDistanceTracker& t) { return result_to_dict(t.download()); },
          "dict: frames float64 (n_replicas, n_frames, 7) = (R_F [nm] ungewrappt, μ̂, d [nm]),\n"
          "steps (n_frames,). Ringpuffer: älteste Frames werden überschrieben.");
  cls.def(
      "add_dipole_tracker",
      [](SimulationContext& c, HostInput<int32_t> offsets, HostInput<int32_t> atoms,
         HostInput<int32_t> atom_a, HostInput<int32_t> atom_b, float z_surface, int stride,
         int capacity) -> bomd::DipoleAndDistanceTracker& {
        return static_cast<bomd::DipoleAndDistanceTracker&>(
            c.add_observer(std::make_unique<bomd::DipoleAndDistanceTracker>(
                probe_params(offsets, atoms, atom_a, atom_b), z_surface, stride, capacity)));
      },
      "group_offsets"_a, "group_atoms"_a, "atom_a"_a, "atom_b"_a, "z_surface"_a = 0.0f,
      "stride"_a = 1, "capacity"_a = 1000, nb::rv_policy::reference_internal,
      "Dipol-/Abstands-Tracker: Fluorophorgruppe je Replica als CSR (lokale Indizes),\n"
      "Dipolachse atom_a → atom_b (n_replicas,), z_surface [nm].");

  nb::class_<bomd::FRETTracker>(m, "FRETTracker", "R_DA, κ², E je Frame, PHYSICS_MODEL.md §6.2.")
      .def_prop_ro("overwritten", &bomd::FRETTracker::overwritten)
      .def("reset", &bomd::FRETTracker::reset)
      .def(
          "result",
          [](const bomd::FRETTracker& t) {
            nb::dict d = result_to_dict(t.download());
            nb::object f = d["frames"];
            d["r_da"] = f[nb::make_tuple(nb::slice(nb::none(), nb::none(), nb::none()),
                                         nb::slice(nb::none(), nb::none(), nb::none()), 0)];
            d["kappa2"] = f[nb::make_tuple(nb::slice(nb::none(), nb::none(), nb::none()),
                                           nb::slice(nb::none(), nb::none(), nb::none()), 1)];
            d["efficiency"] = f[nb::make_tuple(nb::slice(nb::none(), nb::none(), nb::none()),
                                               nb::slice(nb::none(), nb::none(), nb::none()), 2)];
            return d;
          },
          "dict: r_da [nm], kappa2, efficiency (0 ohne r0) je (n_replicas, n_frames), frames,\n"
          "steps.");
  cls.def(
      "add_fret_tracker",
      [](SimulationContext& c, HostInput<int32_t> d_off, HostInput<int32_t> d_atoms,
         HostInput<int32_t> d_a, HostInput<int32_t> d_b, HostInput<int32_t> a_off,
         HostInput<int32_t> a_atoms, HostInput<int32_t> a_a, HostInput<int32_t> a_b, float r0,
         int stride, int capacity) -> bomd::FRETTracker& {
        return static_cast<bomd::FRETTracker&>(c.add_observer(std::make_unique<bomd::FRETTracker>(
            probe_params(d_off, d_atoms, d_a, d_b), probe_params(a_off, a_atoms, a_a, a_b), r0,
            stride, capacity)));
      },
      "donor_offsets"_a, "donor_atoms"_a, "donor_a"_a, "donor_b"_a, "acceptor_offsets"_a,
      "acceptor_atoms"_a, "acceptor_a"_a, "acceptor_b"_a, "r0"_a = 0.0f, "stride"_a = 1,
      "capacity"_a = 1000, nb::rv_policy::reference_internal,
      "FRET-Tracker: Donor- und Akzeptor-Sonde wie add_dipole_tracker; r0 [nm] > 0 liefert\n"
      "zusätzlich E (Förster-Radius für κ² = 2/3).");

  nb::class_<bomd::RefractiveIndexProfiler>(m, "RefractiveIndexProfiler",
                                            "n(z) nach Lorentz-Lorenz, PHYSICS_MODEL.md §6.3.")
      .def_prop_ro("n_frames", &bomd::RefractiveIndexProfiler::n_frames)
      .def("reset", &bomd::RefractiveIndexProfiler::reset)
      .def(
          "result",
          [](const bomd::RefractiveIndexProfiler& p) {
            nb::dict d = result_to_dict(p.download());
            return d;
          },
          "dict: z (n_bins,) Schichtmitten [nm], n, s_alpha_mean/var, s_volume_mean/var\n"
          "(n_replicas, n_bins) [nm^3 bzw. nm^6], n_frames.");
  cls.def(
      "add_refractive_index_profiler",
      [](SimulationContext& c, float dz, int n_bins, HostInput<float> alpha,
         HostInput<float> volume, float z_min, float n_water, const std::string& smearing,
         int stride) -> bomd::RefractiveIndexProfiler& {
        bomd::LayerSmearing mode;
        if (smearing == "sphere") {
          mode = bomd::LayerSmearing::kSphere;
        } else if (smearing == "gauss") {
          mode = bomd::LayerSmearing::kGauss;
        } else if (smearing == "point") {
          mode = bomd::LayerSmearing::kPoint;
        } else {
          throw nb::value_error("smearing: 'sphere', 'gauss' oder 'point'");
        }
        bomd::RefractiveIndexParams p{dz,      n_bins, z_min, to_vector(alpha), to_vector(volume),
                                      n_water, mode};
        return static_cast<bomd::RefractiveIndexProfiler&>(
            c.add_observer(std::make_unique<bomd::RefractiveIndexProfiler>(std::move(p), stride)));
      },
      "dz"_a, "n_bins"_a, "alpha"_a, "volume"_a, "z_min"_a = 0.0f, "n_water"_a = 1.333f,
      "smearing"_a = "sphere", "stride"_a = 1, nb::rv_policy::reference_internal,
      "Brechungsindexprofil: Schichten der Dicke dz [nm] ab z_min; alpha (α′ [nm^3]) und\n"
      "volume (v [nm^3]) je Optik-Typ types[..., 1]; n_water. smearing (OQ-02): 'sphere'\n"
      "(Default, Kugelschnitt mit R = (3v/4π)^(1/3)), 'gauss' (σ = R/√5) oder 'point'.");

  nb::class_<bomd::PolarizabilityGridObserver>(
      m, "PolarizabilityGridObserver", "3D-Permittivitätskarte ε = n² (Lorentz-Lorenz), ADR 0016.")
      .def_prop_ro("n_frames", &bomd::PolarizabilityGridObserver::n_frames)
      .def("reset", &bomd::PolarizabilityGridObserver::reset)
      .def(
          "result",
          [](const bomd::PolarizabilityGridObserver& p) { return result_to_dict(p.download()); },
          "dict: x (nx,), y (ny,), z (nz,) Zellmitten [nm]; eps, s_alpha_mean/var,\n"
          "s_volume_mean/var (n_replicas, nz, ny, nx) [1, nm^3 bzw. nm^6]; n_frames.");
  cls.def(
      "add_polarizability_grid",
      [](SimulationContext& c, std::array<int, 3> shape, std::array<float, 3> origin,
         std::array<float, 3> spacing, HostInput<float> alpha, HostInput<float> volume,
         float n_water, const std::string& smearing, bool periodic_xy,
         int stride) -> bomd::PolarizabilityGridObserver& {
        bomd::PolarizabilityGridParams p;
        p.shape = shape;
        p.origin_nm = origin;
        p.spacing_nm = spacing;
        p.alpha_nm3 = to_vector(alpha);
        p.volume_nm3 = to_vector(volume);
        p.n_water = n_water;
        if (smearing == "gauss") {
          p.smearing = bomd::LayerSmearing::kGauss;
        } else if (smearing == "point") {
          p.smearing = bomd::LayerSmearing::kPoint;
        } else {
          throw nb::value_error("smearing: 'gauss' oder 'point'");
        }
        p.periodic_xy = periodic_xy;
        return static_cast<bomd::PolarizabilityGridObserver&>(c.add_observer(
            std::make_unique<bomd::PolarizabilityGridObserver>(std::move(p), stride)));
      },
      "shape"_a, "origin"_a, "spacing"_a, "alpha"_a, "volume"_a, "n_water"_a = 1.333f,
      "smearing"_a = "gauss", "periodic_xy"_a = false, "stride"_a = 1,
      nb::rv_policy::reference_internal,
      "3D-Permittivitätskarte (ADR 0016): Zellen (nx, ny, nz) ab der unteren Ecke origin [nm] mit\n"
      "spacing [nm]; alpha (α′ [nm^3]) und volume (v [nm^3]) je Optik-Typ types[..., 1]; n_water.\n"
      "smearing: 'gauss' (Default, σ = R/√5 je Achse, Fenster ±4σ) oder 'point'. periodic_xy\n"
      "verlangt nx·Δx = L_x und ny·Δy = L_y.");

  // --- VoxelGridForce3D -------------------------------------------------------------------------
  nb::class_<bomd::VoxelGridForce3D>(m, "VoxelGridForce3D", "Externes Gitterpotential, §4.3.")
      .def(
          "set_values",
          [](bomd::VoxelGridForce3D& g,
             nb::ndarray<const float, nb::shape<-1, -1, -1, 4>, nb::c_contig, nb::device::cpu>
                 values) {
            const auto& shape = g.params().shape;
            if (values.shape(0) != static_cast<std::size_t>(shape[2]) ||
                values.shape(1) != static_cast<std::size_t>(shape[1]) ||
                values.shape(2) != static_cast<std::size_t>(shape[0]))
              throw nb::value_error("set_values: Shape (nz, ny, nx, 4) des bestehenden Gitters");
            g.set_values(std::vector<float>(values.data(), values.data() + values.size()));
          },
          "values"_a,
          "Neue Werte float32 (nz, ny, nx, 4) gleicher Shape (ADR 0016 Punkt 3); Kräfte werden\n"
          "vor dem nächsten Schritt neu berechnet.");
  cls.def(
      "add_voxel_grid",
      [](SimulationContext& c,
         nb::ndarray<const float, nb::shape<-1, -1, -1, 4>, nb::c_contig, nb::device::cpu> values,
         std::array<float, 3> origin, std::array<float, 3> spacing, HostInput<float> scale,
         bool periodic_xy) -> bomd::VoxelGridForce3D& {
        bomd::VoxelGridParams p;
        p.values.assign(values.data(), values.data() + values.size());
        p.shape = {static_cast<int>(values.shape(2)), static_cast<int>(values.shape(1)),
                   static_cast<int>(values.shape(0))};
        p.origin_nm = origin;
        p.spacing_nm = spacing;
        p.scale_by_type = to_vector(scale);
        p.periodic_xy = periodic_xy;
        return static_cast<bomd::VoxelGridForce3D&>(
            c.add_force(std::make_unique<bomd::VoxelGridForce3D>(std::move(p))));
      },
      "values"_a, "origin"_a, "spacing"_a, "scale"_a, "periodic_xy"_a = false,
      nb::rv_policy::reference_internal,
      "Gitterpotential: values float32 (nz, ny, nx, 4) = (V [kJ/mol], dV/dx, dV/dy, dV/dz\n"
      "[kJ/mol/nm]) pro Skalierungseinheit (biooptomd.voxel.values_from_potential), origin und\n"
      "spacing [nm], scale je Optik-/Oberflächen-Typ types[..., 1].");

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
