
// C.f. https://numpy.org/doc/1.21/reference/c-api/array.html#importing-the-api
#define PY_ARRAY_UNIQUE_SYMBOL _cpp2py_ARRAY_API
#ifndef CLAIR_C2PY_WRAP_GEN
#ifdef __clang__
// #pragma clang diagnostic ignored "-W#warnings"
#endif
#ifdef __GNUC__
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#pragma GCC diagnostic ignored "-Wcast-function-type"
#pragma GCC diagnostic ignored "-Wcpp"
#endif

#define C2PY_VERSION_MAJOR 0
#define C2PY_VERSION_MINOR 1

#include <c2py/c2py.hpp>

using c2py::operator""_a;

// ==================== Wrapped classes =====================

// ==================== enums =====================

// ==================== module classes =====================

// ==================== module functions ====================

// calculate_kpq_map
static auto const fun_0 = c2py::dispatcher_f_kw_t{c2py::cfun(
    [](const nda::array<double, 2> &kpts_crys,
       const nda::array<double, 1> &q_vec) {
      return coqui_py::calculate_kpq_map(kpts_crys, q_vec);
    },
    "kpts_crys", "q_vec")};

// mbpt
static auto const fun_1 = c2py::dispatcher_f_kw_t{
    c2py::cfun(
        [](const std::string &solver_type, const std::string &mbpt_params,
           coqui_py::ThcCoulomb &h_int,
           const nda::array<ComplexType, 5> &C_ksIai,
           const nda::array<long, 3> &band_window,
           const nda::array<double, 2> &kpts_crys,
           std::optional<std::map<std::string, nda::array<ComplexType, 5>>>
               local_polarizabilities) {
          return coqui_py::mbpt(solver_type, mbpt_params, h_int, C_ksIai,
                                band_window, kpts_crys, local_polarizabilities);
        },
        "solver_type", "mbpt_params", "h_int", "C_ksIai", "band_window",
        "kpts_crys", "local_polarizabilities"),
    c2py::cfun(
        [](const std::string &solver_type, const std::string &mbpt_params,
           coqui_py::CholCoulomb &h_int,
           const nda::array<ComplexType, 5> &C_ksIai,
           const nda::array<long, 3> &band_window,
           const nda::array<double, 2> &kpts_crys,
           std::optional<std::map<std::string, nda::array<ComplexType, 5>>>
               local_polarizabilities) {
          return coqui_py::mbpt(solver_type, mbpt_params, h_int, C_ksIai,
                                band_window, kpts_crys, local_polarizabilities);
        },
        "solver_type", "mbpt_params", "h_int", "C_ksIai", "band_window",
        "kpts_crys", "local_polarizabilities"),
    c2py::cfun(
        [](const std::string &solver_type, const std::string &mbpt_params,
           coqui_py::ThcCoulomb &h_int) {
          return coqui_py::mbpt(solver_type, mbpt_params, h_int);
        },
        "solver_type", "mbpt_params", "h_int"),
    c2py::cfun(
        [](const std::string &solver_type, const std::string &mbpt_params,
           coqui_py::CholCoulomb &h_int) {
          return coqui_py::mbpt(solver_type, mbpt_params, h_int);
        },
        "solver_type", "mbpt_params", "h_int"),
    c2py::cfun(
        [](const std::string &solver_type, const std::string &mbpt_params,
           coqui_py::ThcCoulomb &h_int, coqui_py::ThcCoulomb &h_int_hf) {
          return coqui_py::mbpt(solver_type, mbpt_params, h_int, h_int_hf);
        },
        "solver_type", "mbpt_params", "h_int", "h_int_hf"),
    c2py::cfun(
        [](const std::string &solver_type, const std::string &mbpt_params,
           coqui_py::ThcCoulomb &h_int, coqui_py::CholCoulomb &h_int_hf) {
          return coqui_py::mbpt(solver_type, mbpt_params, h_int, h_int_hf);
        },
        "solver_type", "mbpt_params", "h_int", "h_int_hf"),
    c2py::cfun(
        [](const std::string &solver_type, const std::string &mbpt_params,
           coqui_py::CholCoulomb &h_int, coqui_py::ThcCoulomb &h_int_hf) {
          return coqui_py::mbpt(solver_type, mbpt_params, h_int, h_int_hf);
        },
        "solver_type", "mbpt_params", "h_int", "h_int_hf"),
    c2py::cfun(
        [](const std::string &solver_type, const std::string &mbpt_params,
           coqui_py::CholCoulomb &h_int, coqui_py::CholCoulomb &h_int_hf) {
          return coqui_py::mbpt(solver_type, mbpt_params, h_int, h_int_hf);
        },
        "solver_type", "mbpt_params", "h_int", "h_int_hf"),
    c2py::cfun(
        [](const std::string &solver_type, const std::string &mbpt_params,
           coqui_py::ThcCoulomb &h_int, coqui_py::ThcCoulomb &h_int_hartree,
           coqui_py::ThcCoulomb &h_int_exchange) {
          return coqui_py::mbpt(solver_type, mbpt_params, h_int, h_int_hartree,
                                h_int_exchange);
        },
        "solver_type", "mbpt_params", "h_int", "h_int_hartree",
        "h_int_exchange"),
    c2py::cfun(
        [](const std::string &solver_type, const std::string &mbpt_params,
           coqui_py::ThcCoulomb &h_int, coqui_py::ThcCoulomb &h_int_hartree,
           coqui_py::CholCoulomb &h_int_exchange) {
          return coqui_py::mbpt(solver_type, mbpt_params, h_int, h_int_hartree,
                                h_int_exchange);
        },
        "solver_type", "mbpt_params", "h_int", "h_int_hartree",
        "h_int_exchange"),
    c2py::cfun(
        [](const std::string &solver_type, const std::string &mbpt_params,
           coqui_py::ThcCoulomb &h_int, coqui_py::CholCoulomb &h_int_hartree,
           coqui_py::ThcCoulomb &h_int_exchange) {
          return coqui_py::mbpt(solver_type, mbpt_params, h_int, h_int_hartree,
                                h_int_exchange);
        },
        "solver_type", "mbpt_params", "h_int", "h_int_hartree",
        "h_int_exchange"),
    c2py::cfun(
        [](const std::string &solver_type, const std::string &mbpt_params,
           coqui_py::ThcCoulomb &h_int, coqui_py::CholCoulomb &h_int_hartree,
           coqui_py::CholCoulomb &h_int_exchange) {
          return coqui_py::mbpt(solver_type, mbpt_params, h_int, h_int_hartree,
                                h_int_exchange);
        },
        "solver_type", "mbpt_params", "h_int", "h_int_hartree",
        "h_int_exchange"),
    c2py::cfun(
        [](const std::string &solver_type, const std::string &mbpt_params,
           coqui_py::CholCoulomb &h_int, coqui_py::ThcCoulomb &h_int_hartree,
           coqui_py::ThcCoulomb &h_int_exchange) {
          return coqui_py::mbpt(solver_type, mbpt_params, h_int, h_int_hartree,
                                h_int_exchange);
        },
        "solver_type", "mbpt_params", "h_int", "h_int_hartree",
        "h_int_exchange"),
    c2py::cfun(
        [](const std::string &solver_type, const std::string &mbpt_params,
           coqui_py::CholCoulomb &h_int, coqui_py::ThcCoulomb &h_int_hartree,
           coqui_py::CholCoulomb &h_int_exchange) {
          return coqui_py::mbpt(solver_type, mbpt_params, h_int, h_int_hartree,
                                h_int_exchange);
        },
        "solver_type", "mbpt_params", "h_int", "h_int_hartree",
        "h_int_exchange"),
    c2py::cfun(
        [](const std::string &solver_type, const std::string &mbpt_params,
           coqui_py::CholCoulomb &h_int, coqui_py::CholCoulomb &h_int_hartree,
           coqui_py::ThcCoulomb &h_int_exchange) {
          return coqui_py::mbpt(solver_type, mbpt_params, h_int, h_int_hartree,
                                h_int_exchange);
        },
        "solver_type", "mbpt_params", "h_int", "h_int_hartree",
        "h_int_exchange"),
    c2py::cfun(
        [](const std::string &solver_type, const std::string &mbpt_params,
           coqui_py::CholCoulomb &h_int, coqui_py::CholCoulomb &h_int_hartree,
           coqui_py::CholCoulomb &h_int_exchange) {
          return coqui_py::mbpt(solver_type, mbpt_params, h_int, h_int_hartree,
                                h_int_exchange);
        },
        "solver_type", "mbpt_params", "h_int", "h_int_hartree",
        "h_int_exchange")};

// run_lr
static auto const fun_2 = c2py::dispatcher_f_kw_t{c2py::cfun(
    [](const std::string &lr_params, coqui_py::ThcCoulomb &h_int,
       const nda::array<double, 1> &q_vec,
       std::optional<nda::array<ComplexType, 4>> DeltaH0_skij,
       bool include_hartree, bool include_exchange, std::string gw_mode_str,
       int max_iter, double tol, bool fix_density, std::string iter_alg,
       double mixing, int max_subsp_size, int diis_warmup,
       std::optional<nda::array<ComplexType, 4>> DeltaX_left,
       std::optional<nda::array<ComplexType, 4>> DeltaX_right,
       std::optional<nda::array<ComplexType, 3>> DeltaV_qPQ) {
      return coqui_py::run_lr(lr_params, h_int, q_vec, DeltaH0_skij,
                              include_hartree, include_exchange, gw_mode_str,
                              max_iter, tol, fix_density, iter_alg, mixing,
                              max_subsp_size, diis_warmup, DeltaX_left,
                              DeltaX_right, DeltaV_qPQ);
    },
    "lr_params", "h_int", "q_vec", "DeltaH0_skij", "include_hartree",
    "include_exchange", "gw_mode_str", "max_iter", "tol", "fix_density",
    "iter_alg", "mixing", "max_subsp_size", "diis_warmup", "DeltaX_left",
    "DeltaX_right", "DeltaV_qPQ")};

static const auto doc_d_0 =
    fun_0.doc(R"DOC(
Compute k+q mapping for linear response calculations

Parameters
----------
kpts_crys : {par_0}
   - [INPUT] k-points in crystal coordinates (nkpts, 3)
q_vec : {par_1}
   - [INPUT] perturbation wavevector in crystal coordinates (3,)

Returns
-------
{ret_0}
   - [OUTPUT] k → k+q index mapping (nkpts,)
)DOC",
              {{c2py::python_typename<const nda::array<double, 2> &>()},
               {c2py::python_typename<const nda::array<double, 1> &>()}},
              {c2py::python_typename<nda::array<long, 1>>()});
static const auto doc_d_1 = fun_1.doc(R"DOC()DOC");
static const auto doc_d_2 = fun_2.doc(
    R"DOC(
Unified linear response calculation

Reads the unperturbed state from the checkpoint, runs the LR SCF loop
  ΔH0 → ΔG → ΔDm → [ΔF] → [ΔΣ] → ΔG → ... until convergence,
and writes results to the "linear_response" group of the output checkpoint.

Parameters
----------
lr_params : {par_0}
   - [INPUT] JSON string with params (prefix, output, input_type, input_iter, h0_source, div_corr)
h_int : {par_1}
   - [INPUT] THC ERI handler
q_vec : {par_2}
   - [INPUT] Perturbation wavevector in crystal coords (3,)
DeltaH0_skij : {par_3}
   - [INPUT] Perturbation matrix (ns, nk, nb, nb);
                              required on the MPI global root, None elsewhere.
include_hartree : {par_4}
   - [INPUT] Include ΔJ in SCF loop
include_exchange : {par_5}
   - [INPUT] Include ΔK in SCF loop
gw_mode : {par_6}
   - [INPUT] GW mode: "none", "fixed_W", or "full"
max_iter : {par_7}
   - [INPUT] Maximum SCF iterations (1 = one-shot)
tol : {par_8}
   - [INPUT] Convergence tolerance
fix_density : {par_9}
   - [INPUT] If true, compute Δμ to enforce ΔN=0
iter_alg : {par_10}
   - [INPUT] Iteration algorithm: "damping" or "DIIS"
mixing : {par_11}
   - [INPUT] Damping/mixing parameter
max_subsp_size : {par_12}
   - [INPUT] DIIS subspace size
diis_warmup : {par_13}
   - [INPUT] DIIS warmup iterations
DeltaX_left : {par_14}
   - [INPUT] Optional δ^q X collocation perturbation (root only)
DeltaX_right : {par_15}
   - [INPUT] Optional δ^{-q} X collocation perturbation (root only)
DeltaV_qPQ : {par_16}
   - [INPUT] Optional THC Coulomb perturbation δV (root only)

Returns
-------
{ret_0}
   - [OUTPUT] Tuple of (niter, Delta_mu)
)DOC",
    {{c2py::python_typename<const std::string &>()},
     {c2py::python_typename<coqui_py::ThcCoulomb &>()},
     {c2py::python_typename<const nda::array<double, 1> &>()},
     {c2py::python_typename<std::optional<nda::array<ComplexType, 4>>>()},
     {c2py::python_typename<bool>()},
     {c2py::python_typename<bool>()},
     {},
     {c2py::python_typename<int>()},
     {c2py::python_typename<double>()},
     {c2py::python_typename<bool>()},
     {c2py::python_typename<std::string>()},
     {c2py::python_typename<double>()},
     {c2py::python_typename<int>()},
     {c2py::python_typename<int>()},
     {c2py::python_typename<std::optional<nda::array<ComplexType, 4>>>()},
     {c2py::python_typename<std::optional<nda::array<ComplexType, 4>>>()},
     {c2py::python_typename<std::optional<nda::array<ComplexType, 3>>>()}},
    {c2py::python_typename<std::tuple<int, double>>()});
//--------------------- module function table  -----------------------------

static PyMethodDef module_methods[] = {
    {"calculate_kpq_map", (PyCFunction)c2py::pyfkw<fun_0>,
     METH_VARARGS | METH_KEYWORDS, doc_d_0.c_str()},
    {"mbpt", (PyCFunction)c2py::pyfkw<fun_1>, METH_VARARGS | METH_KEYWORDS,
     doc_d_1.c_str()},
    {"run_lr", (PyCFunction)c2py::pyfkw<fun_2>, METH_VARARGS | METH_KEYWORDS,
     doc_d_2.c_str()},
    {nullptr, nullptr, 0, nullptr} // Sentinel
};

//--------------------- module struct & init error definition ------------

//// module doc directly in the code or "" if not present...
/// Or mandatory ?
static struct PyModuleDef module_def = {
    PyModuleDef_HEAD_INIT,
    "mbpt_module",                          /* name of module */
    R"RAWDOC(MBPT module for CoQui)RAWDOC", /* module documentation, may be NULL
                                             */
    -1, /* size of per-interpreter state of the module, or -1 if the module
           keeps state in global variables. */
    module_methods,
    NULL,
    NULL,
    NULL,
    NULL};

//--------------------- module init function -----------------------------

extern "C" __attribute__((visibility("default"))) PyObject *
PyInit_mbpt_module() {

  if (not c2py::check_python_version("mbpt_module"))
    return NULL;

  // import numpy iff 'numpy/arrayobject.h' included
#ifdef Py_ARRAYOBJECT_H
  import_array();
#endif

  PyObject *m;

  if (PyType_Ready(&c2py::wrap_pytype<c2py::py_range>) < 0)
    return NULL;

  m = PyModule_Create(&module_def);
  if (m == NULL)
    return NULL;

  auto &conv_table = *c2py::conv_table_sptr.get();

  conv_table[std::type_index(typeid(c2py::py_range)).name()] =
      &c2py::wrap_pytype<c2py::py_range>;

  return m;
}
#endif
// CLAIR_WRAP_GEN
