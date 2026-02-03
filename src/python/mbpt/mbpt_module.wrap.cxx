/**
 * ==========================================================================
 * CoQuí: Correlated Quantum ínterface
 *
 * Copyright (c) 2022-2025 Simons Foundation & The CoQuí developer team
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ==========================================================================
 */



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

// mbpt
static auto const fun_0 = c2py::dispatcher_f_kw_t{
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

static const auto doc_d_0 = fun_0.doc(R"DOC()DOC");

// lr_dyson - run LR Dyson calculation
static auto const fun_lr_dyson = c2py::dispatcher_f_kw_t{
    c2py::cfun(
        [](const std::string &lr_params,
           coqui_py::ThcCoulomb &h_int,
           nda::array<double, 1> const& q_vec,
           nda::array<ComplexType, 4> const& DeltaH0_skij,
           bool fix_density) {
          return coqui_py::lr_dyson(lr_params, h_int, q_vec, DeltaH0_skij, fix_density);
        },
        "lr_params", "h_int", "q_vec", "DeltaH0_skij", "fix_density"),
    c2py::cfun(
        [](const std::string &lr_params,
           coqui_py::CholCoulomb &h_int,
           nda::array<double, 1> const& q_vec,
           nda::array<ComplexType, 4> const& DeltaH0_skij,
           bool fix_density) {
          return coqui_py::lr_dyson(lr_params, h_int, q_vec, DeltaH0_skij, fix_density);
        },
        "lr_params", "h_int", "q_vec", "DeltaH0_skij", "fix_density")};

static const auto doc_lr_dyson = fun_lr_dyson.doc(R"DOC(
Run linear response Dyson equation calculation.

Solves: ΔG(k,iω) = G(k+q,iω) · [ΔH0(k) - Δμ·S(k)] · G(k,iω)

This function reads the unperturbed Green's function from a previous HF/GW
checkpoint, solves the LR Dyson equation, and writes the results (ΔG, ΔDm)
back to the checkpoint file.

Two modes are available:
- fix_density=False: Use Δμ=0 (chemical potential fixed)
- fix_density=True: Compute Δμ to enforce ΔN=0 (particle number fixed)

Parameters
----------
lr_params : str
    JSON string with parameters:
    - prefix: Input checkpoint prefix (reads {prefix}.mbpt.h5)
    - output: Output checkpoint prefix (default: same as prefix)
h_int : ThcCoulomb or CholCoulomb
    ERI handler from the original calculation
q_vec : np.ndarray
    Perturbation wavevector in crystal coordinates, shape (3,)
DeltaH0_skij : np.ndarray
    Perturbation matrix, shape (ns, nk, nb, nb)
fix_density : bool
    If True, compute Δμ to enforce ΔN=0 (default False)

Returns
-------
float
    The Δμ value used (computed if fix_density=True, otherwise 0.0)
)DOC");

// calculate_kpq_map
static auto const fun_kpq_map = c2py::dispatcher_f_kw_t{
    c2py::cfun(
        [](nda::array<double, 2> const& kpts_crys,
           nda::array<double, 1> const& q_vec) {
          return coqui_py::calculate_kpq_map(kpts_crys, q_vec);
        },
        "kpts_crys", "q_vec")};

static const auto doc_kpq_map = fun_kpq_map.doc(R"DOC(
Compute k+q mapping for linear response calculations.

Given a k-point grid and a perturbation wavevector q, compute the mapping
kpq_map[ik] = ik' where k[ik] + q = k[ik'] (mod G).

Parameters
----------
kpts_crys : np.ndarray
    k-points in crystal coordinates, shape (nkpts, 3)
q_vec : np.ndarray
    Perturbation wavevector in crystal coordinates, shape (3,)

Returns
-------
np.ndarray
    k → k+q index mapping, shape (nkpts,)
)DOC");

// lr_hf - compute LR Fock matrix from LR density matrix
static auto const fun_lr_hf = c2py::dispatcher_f_kw_t{
    c2py::cfun(
        [](coqui_py::ThcCoulomb &h_int,
           nda::array<double, 1> const& q_vec,
           nda::array<ComplexType, 4> const& DeltaDm_skij,
           nda::array<ComplexType, 4> const& S_skij,
           bool compute_hartree,
           bool compute_exchange) {
          return coqui_py::lr_hf(h_int, q_vec, DeltaDm_skij, S_skij,
                                 compute_hartree, compute_exchange);
        },
        "h_int", "q_vec", "DeltaDm_skij", "S_skij", "compute_hartree", "compute_exchange")};

static const auto doc_lr_hf = fun_lr_hf.doc(R"DOC(
Compute linear response Fock matrix from LR density matrix.

Computes ΔF = ΔJ + ΔK from the LR density matrix ΔDm using THC-ERI.

Parameters
----------
h_int : ThcCoulomb
    THC ERI handler
q_vec : np.ndarray
    Perturbation wavevector in crystal coordinates, shape (3,)
DeltaDm_skij : np.ndarray
    LR density matrix, shape (ns, nk, nb, nb)
S_skij : np.ndarray
    Overlap matrix, shape (ns, nk, nb, nb)
compute_hartree : bool
    Whether to compute the Hartree (Coulomb) term
compute_exchange : bool
    Whether to compute the Exchange term

Returns
-------
np.ndarray
    LR Fock matrix, shape (ns, nk, nb, nb)
)DOC");

// hf_evaluate - compute HF self-energy (Fock matrix) from density matrix
static auto const fun_hf_evaluate = c2py::dispatcher_f_kw_t{
    c2py::cfun(
        [](coqui_py::ThcCoulomb &h_int,
           nda::array<ComplexType, 4> const& Dm_skij,
           nda::array<ComplexType, 4> const& S_skij,
           bool compute_hartree,
           bool compute_exchange) {
          return coqui_py::hf_evaluate(h_int, Dm_skij, S_skij,
                                       compute_hartree, compute_exchange);
        },
        "h_int", "Dm_skij", "S_skij", "compute_hartree", "compute_exchange"),
    c2py::cfun(
        [](coqui_py::CholCoulomb &h_int,
           nda::array<ComplexType, 4> const& Dm_skij,
           nda::array<ComplexType, 4> const& S_skij,
           bool compute_hartree,
           bool compute_exchange) {
          return coqui_py::hf_evaluate(h_int, Dm_skij, S_skij,
                                       compute_hartree, compute_exchange);
        },
        "h_int", "Dm_skij", "S_skij", "compute_hartree", "compute_exchange")};

static const auto doc_hf_evaluate = fun_hf_evaluate.doc(R"DOC(
Compute HF self-energy (Fock matrix) from density matrix.

Computes F = J + K from the density matrix Dm.

Parameters
----------
h_int : ThcCoulomb or CholCoulomb
    ERI handler (THC or Cholesky)
Dm_skij : np.ndarray
    Density matrix, shape (ns, nk, nb, nb)
S_skij : np.ndarray
    Overlap matrix, shape (ns, nk, nb, nb)
compute_hartree : bool
    Whether to compute the Hartree (Coulomb) term
compute_exchange : bool
    Whether to compute the Exchange term

Returns
-------
np.ndarray
    Fock matrix F, shape (ns, nk, nb, nb)
)DOC");

// lr_hf_scf - run LR-HF SCF calculation
static auto const fun_lr_hf_scf = c2py::dispatcher_f_kw_t{
    c2py::cfun(
        [](const std::string &lr_params,
           coqui_py::ThcCoulomb &h_int,
           nda::array<double, 1> const& q_vec,
           nda::array<ComplexType, 4> const& DeltaH0_skij,
           int max_iter,
           double tol,
           bool fix_density) {
          return coqui_py::lr_hf_scf(lr_params, h_int, q_vec, DeltaH0_skij,
                                      max_iter, tol, fix_density);
        },
        "lr_params", "h_int", "q_vec", "DeltaH0_skij", "max_iter", "tol", "fix_density")};

static const auto doc_lr_hf_scf = fun_lr_hf_scf.doc(R"DOC(
Run linear response Hartree-Fock SCF calculation.

Runs the full LR-HF SCF loop:
    ΔH0 → ΔG → ΔDm → ΔF → ΔG → ... (iterate until convergence)

This function reads the unperturbed Green's function from a previous HF/GW
checkpoint, runs the LR-HF SCF loop, and writes the results (ΔG, ΔDm, ΔF, Δμ)
back to the checkpoint file.

Parameters
----------
lr_params : str
    JSON string with parameters:
    - prefix: Input checkpoint prefix (reads {prefix}.mbpt.h5)
    - output: Output checkpoint prefix (default: same as prefix)
h_int : ThcCoulomb
    THC ERI handler from the original calculation
q_vec : np.ndarray
    Perturbation wavevector in crystal coordinates, shape (3,)
DeltaH0_skij : np.ndarray
    Perturbation matrix, shape (ns, nk, nb, nb)
max_iter : int
    Maximum SCF iterations (default 50)
tol : float
    Convergence tolerance for ||ΔDm_new - ΔDm_old|| (default 1e-8)
fix_density : bool
    If True, compute Δμ to enforce ΔN=0 (default True)

Returns
-------
tuple
    (niter, Delta_mu) where niter is the number of iterations and
    Delta_mu is the computed chemical potential shift.
)DOC");

//--------------------- module function table  -----------------------------

static PyMethodDef module_methods[] = {
    {"mbpt", (PyCFunction)c2py::pyfkw<fun_0>, METH_VARARGS | METH_KEYWORDS,
     doc_d_0.c_str()},
    {"lr_dyson", (PyCFunction)c2py::pyfkw<fun_lr_dyson>,
     METH_VARARGS | METH_KEYWORDS, doc_lr_dyson.c_str()},
    {"calculate_kpq_map_cpp", (PyCFunction)c2py::pyfkw<fun_kpq_map>,
     METH_VARARGS | METH_KEYWORDS, doc_kpq_map.c_str()},
    {"lr_hf_cpp", (PyCFunction)c2py::pyfkw<fun_lr_hf>,
     METH_VARARGS | METH_KEYWORDS, doc_lr_hf.c_str()},
    {"hf_evaluate_cpp", (PyCFunction)c2py::pyfkw<fun_hf_evaluate>,
     METH_VARARGS | METH_KEYWORDS, doc_hf_evaluate.c_str()},
    {"lr_hf_scf_cpp", (PyCFunction)c2py::pyfkw<fun_lr_hf_scf>,
     METH_VARARGS | METH_KEYWORDS, doc_lr_hf_scf.c_str()},
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
