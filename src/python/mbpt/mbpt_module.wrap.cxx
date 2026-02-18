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

// run_lr - unified linear response calculation
static auto const fun_run_lr = c2py::dispatcher_f_kw_t{
    c2py::cfun(
        [](const std::string &lr_params,
           coqui_py::ThcCoulomb &h_int,
           nda::array<double, 1> const& q_vec,
           nda::array<ComplexType, 4> const& DeltaH0_skij,
           bool include_hartree,
           bool include_exchange,
           bool include_gw_sigma,
           int max_iter,
           double tol,
           bool fix_density,
           std::string iter_alg,
           double mixing,
           int max_subsp_size,
           int diis_warmup) {
          return coqui_py::run_lr(lr_params, h_int, q_vec, DeltaH0_skij,
                                  include_hartree, include_exchange, include_gw_sigma,
                                  max_iter, tol, fix_density,
                                  iter_alg, mixing, max_subsp_size, diis_warmup);
        },
        "lr_params", "h_int", "q_vec", "DeltaH0_skij",
        "include_hartree", "include_exchange", "include_gw_sigma",
        "max_iter", "tol", "fix_density",
        "iter_alg", "mixing", "max_subsp_size", "diis_warmup")};

static const auto doc_run_lr = fun_run_lr.doc(R"DOC(
Run unified linear response calculation.

Runs the LR SCF loop with configurable Hartree, Exchange, and GW self-energy:
    ΔH0 → ΔG → ΔDm → [ΔF] → [ΔΣ] → ΔG → ... (iterate until convergence)

Subsumes both lr_dyson (one-shot, all flags false) and lr_hf_scf (HF only).

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
include_hartree : bool
    Include Hartree (Coulomb) term in SCF loop
include_exchange : bool
    Include Exchange term in SCF loop
include_gw_sigma : bool
    Include GW self-energy in SCF loop
max_iter : int
    Maximum SCF iterations (1 = one-shot)
tol : float
    Convergence tolerance for ||ΔDm_new - ΔDm_old||
fix_density : bool
    If True, compute Δμ to enforce ΔN=0
iter_alg : str
    Iteration algorithm: "damping" or "DIIS"
mixing : float
    Damping/mixing parameter
max_subsp_size : int
    Maximum DIIS subspace size
diis_warmup : int
    Number of warmup iterations before DIIS

Returns
-------
tuple
    (niter, Delta_mu) where niter is the number of iterations and
    Delta_mu is the computed chemical potential shift.
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


// lr_gw_sigma - compute LR GW self-energy ΔΣ = -ΔG ⊙ W_c + div_corr
static auto const fun_lr_gw_sigma = c2py::dispatcher_f_kw_t{
    c2py::cfun(
        [](const std::string &lr_params,
           coqui_py::ThcCoulomb &h_int,
           nda::array<double, 1> const& q_pert,
           nda::array<ComplexType, 5> const& DeltaG_tskij) {
          return coqui_py::lr_gw_sigma(lr_params, h_int, q_pert, DeltaG_tskij);
        },
        "lr_params", "h_int", "q_pert", "DeltaG_tskij")};

static const auto doc_lr_gw_sigma = fun_lr_gw_sigma.doc(R"DOC(
Compute linear response GW self-energy with fixed W.

Computes ΔΣ = -ΔG ⊙ W_c (R-space Hadamard product) + q→0 divergence correction.
Reads W from thc_screened_interaction.h5 and eps_inv_head from checkpoint.

Parameters
----------
lr_params : str
    JSON string with parameters:
    - prefix: Input checkpoint prefix (reads {prefix}.mbpt.h5)
h_int : ThcCoulomb
    THC ERI handler
q_pert : np.ndarray
    LR perturbation wavevector in crystal coordinates, shape (3,)
DeltaG_tskij : np.ndarray
    LR Green's function, shape (nt, ns, nk, nb, nb)

Returns
-------
np.ndarray
    LR self-energy ΔΣ, shape (nt, ns, nk, nb, nb)
)DOC");

// lr_gw_Pi - compute LR polarization ΔP = -ΔG·G - G·ΔG
static auto const fun_lr_gw_Pi = c2py::dispatcher_f_kw_t{
    c2py::cfun(
        [](coqui_py::ThcCoulomb &h_int,
           nda::array<double, 1> const& q_pert,
           nda::array<ComplexType, 5> const& G_tskij,
           nda::array<ComplexType, 5> const& DeltaG_tskij) {
          return coqui_py::lr_gw_Pi(h_int, q_pert, G_tskij, DeltaG_tskij);
        },
        "h_int", "q_pert", "G_tskij", "DeltaG_tskij")};

static const auto doc_lr_gw_Pi = fun_lr_gw_Pi.doc(R"DOC(
Compute linear response polarization ΔP = -ΔG·G - G·ΔG (R-space Hadamard product).

Uses lr_thc_comm for ΔG factors (asymmetric X(k+q)/X(k)),
thc_solver_comm for G factors (symmetric).

Parameters
----------
h_int : ThcCoulomb
    THC ERI handler
q_pert : np.ndarray
    LR perturbation wavevector in crystal coordinates, shape (3,)
G_tskij : np.ndarray
    Unperturbed Green's function, shape (nt, ns, nk, nb, nb)
DeltaG_tskij : np.ndarray
    LR Green's function, shape (nt, ns, nk, nb, nb)

Returns
-------
np.ndarray
    LR polarization ΔP, shape (nt_half, nkpts, NP, NP)
)DOC");

// gw_evaluate_Pi - evaluate standard RPA polarization P[G]
static auto const fun_gw_evaluate_Pi = c2py::dispatcher_f_kw_t{
    c2py::cfun(
        [](const std::string &lr_params,
           coqui_py::ThcCoulomb &h_int,
           nda::array<ComplexType, 5> const& G_tskij) {
          return coqui_py::gw_evaluate_Pi(lr_params, h_int, G_tskij);
        },
        "lr_params", "h_int", "G_tskij")};

static const auto doc_gw_evaluate_Pi = fun_gw_evaluate_Pi.doc(R"DOC(
Evaluate standard RPA polarization P[G].

Computes the RPA polarization from a given Green's function.
Used for finite-difference testing of LR polarization.

Parameters
----------
lr_params : str
    JSON string with parameters:
    - prefix: Input checkpoint prefix (reads {prefix}.mbpt.h5)
h_int : ThcCoulomb
    THC ERI handler
G_tskij : np.ndarray
    Green's function, shape (nt, ns, nk, nb, nb)

Returns
-------
np.ndarray
    Polarization P, shape (nt_half, nkpts, NP, NP)
)DOC");

// gw_evaluate_sigma - evaluate GW self-energy Σ = -G ⊙ W_c + div_corr
static auto const fun_gw_evaluate_sigma = c2py::dispatcher_f_kw_t{
    c2py::cfun(
        [](const std::string &lr_params,
           coqui_py::ThcCoulomb &h_int,
           nda::array<ComplexType, 5> const& G_tskij) {
          return coqui_py::gw_evaluate_sigma(lr_params, h_int, G_tskij);
        },
        "lr_params", "h_int", "G_tskij")};

static const auto doc_gw_evaluate_sigma = fun_gw_evaluate_sigma.doc(R"DOC(
Evaluate GW self-energy Σ = -G ⊙ W_c + divergence correction using W from file.

Same R-space computation as lr_gw_sigma but for a full (non-LR) Green's function.
Used for finite-difference testing: Σ[G+εΔG] computed by passing G+εΔG as input.

Parameters
----------
lr_params : str
    JSON string with parameters:
    - prefix: Input checkpoint prefix (reads {prefix}.mbpt.h5)
h_int : ThcCoulomb
    THC ERI handler
G_tskij : np.ndarray
    Green's function, shape (nt, ns, nk, nb, nb)

Returns
-------
np.ndarray
    Self-energy Σ, shape (nt, ns, nk, nb, nb)
)DOC");

//--------------------- module function table  -----------------------------

static PyMethodDef module_methods[] = {
    {"mbpt", (PyCFunction)c2py::pyfkw<fun_0>, METH_VARARGS | METH_KEYWORDS,
     doc_d_0.c_str()},
    {"run_lr_cpp", (PyCFunction)c2py::pyfkw<fun_run_lr>,
     METH_VARARGS | METH_KEYWORDS, doc_run_lr.c_str()},
    {"calculate_kpq_map_cpp", (PyCFunction)c2py::pyfkw<fun_kpq_map>,
     METH_VARARGS | METH_KEYWORDS, doc_kpq_map.c_str()},
    {"lr_hf_cpp", (PyCFunction)c2py::pyfkw<fun_lr_hf>,
     METH_VARARGS | METH_KEYWORDS, doc_lr_hf.c_str()},
    {"hf_evaluate_cpp", (PyCFunction)c2py::pyfkw<fun_hf_evaluate>,
     METH_VARARGS | METH_KEYWORDS, doc_hf_evaluate.c_str()},
    {"lr_gw_sigma_cpp", (PyCFunction)c2py::pyfkw<fun_lr_gw_sigma>,
     METH_VARARGS | METH_KEYWORDS, doc_lr_gw_sigma.c_str()},
    {"gw_evaluate_sigma_cpp", (PyCFunction)c2py::pyfkw<fun_gw_evaluate_sigma>,
     METH_VARARGS | METH_KEYWORDS, doc_gw_evaluate_sigma.c_str()},
    {"lr_gw_Pi_cpp", (PyCFunction)c2py::pyfkw<fun_lr_gw_Pi>,
     METH_VARARGS | METH_KEYWORDS, doc_lr_gw_Pi.c_str()},
    {"gw_evaluate_Pi_cpp", (PyCFunction)c2py::pyfkw<fun_gw_evaluate_Pi>,
     METH_VARARGS | METH_KEYWORDS, doc_gw_evaluate_Pi.c_str()},
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
