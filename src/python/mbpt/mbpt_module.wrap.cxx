
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

// compute_eps_inv_head
static auto const fun_1 = c2py::dispatcher_f_kw_t{c2py::cfun(
    [](const std::string &lr_params, coqui_py::ThcCoulomb &h_int,
       std::optional<nda::array<ComplexType, 4>> W_c_tqPQ) {
      return coqui_py::compute_eps_inv_head(lr_params, h_int, W_c_tqPQ);
    },
    "lr_params", "h_int", "W_c_tqPQ")};

// gw_evaluate_Pi
static auto const fun_2 = c2py::dispatcher_f_kw_t{c2py::cfun(
    [](const std::string &lr_params, coqui_py::ThcCoulomb &h_int,
       std::optional<nda::array<ComplexType, 5>> G_tskij) {
      return coqui_py::gw_evaluate_Pi(lr_params, h_int, G_tskij);
    },
    "lr_params", "h_int", "G_tskij")};

// gw_evaluate_W_from_Pi
static auto const fun_3 = c2py::dispatcher_f_kw_t{c2py::cfun(
    [](const std::string &lr_params, coqui_py::ThcCoulomb &h_int,
       std::optional<nda::array<ComplexType, 4>> Pi_tqPQ) {
      return coqui_py::gw_evaluate_W_from_Pi(lr_params, h_int, Pi_tqPQ);
    },
    "lr_params", "h_int", "Pi_tqPQ")};

// gw_evaluate_sigma
static auto const fun_4 = c2py::dispatcher_f_kw_t{c2py::cfun(
    [](const std::string &lr_params, coqui_py::ThcCoulomb &h_int,
       std::optional<nda::array<ComplexType, 5>> G_tskij, bool div_corr) {
      return coqui_py::gw_evaluate_sigma(lr_params, h_int, G_tskij, div_corr);
    },
    "lr_params", "h_int", "G_tskij", "div_corr")};

// gw_evaluate_sigma_with_W
static auto const fun_5 = c2py::dispatcher_f_kw_t{c2py::cfun(
    [](const std::string &lr_params, coqui_py::ThcCoulomb &h_int,
       std::optional<nda::array<ComplexType, 5>> G_tskij,
       std::optional<nda::array<ComplexType, 4>> W_c_qtPQ,
       const nda::array<ComplexType, 1> &eps_inv_head, bool div_corr) {
      return coqui_py::gw_evaluate_sigma_with_W(
          lr_params, h_int, G_tskij, W_c_qtPQ, eps_inv_head, div_corr);
    },
    "lr_params", "h_int", "G_tskij", "W_c_qtPQ", "eps_inv_head", "div_corr")};

// hf_evaluate
static auto const fun_6 = c2py::dispatcher_f_kw_t{
    c2py::cfun(
        [](coqui_py::ThcCoulomb &h_int,
           const nda::array<ComplexType, 4> &Dm_skij,
           const nda::array<ComplexType, 4> &S_skij, bool compute_hartree,
           bool compute_exchange) {
          return coqui_py::hf_evaluate(h_int, Dm_skij, S_skij, compute_hartree,
                                       compute_exchange);
        },
        "h_int", "Dm_skij", "S_skij", "compute_hartree", "compute_exchange"),
    c2py::cfun(
        [](coqui_py::CholCoulomb &h_int,
           const nda::array<ComplexType, 4> &Dm_skij,
           const nda::array<ComplexType, 4> &S_skij, bool compute_hartree,
           bool compute_exchange) {
          return coqui_py::hf_evaluate(h_int, Dm_skij, S_skij, compute_hartree,
                                       compute_exchange);
        },
        "h_int", "Dm_skij", "S_skij", "compute_hartree", "compute_exchange")};

// lr_gw_Pi
static auto const fun_7 = c2py::dispatcher_f_kw_t{c2py::cfun(
    [](coqui_py::ThcCoulomb &h_int, const nda::array<double, 1> &q_pert,
       std::optional<nda::array<ComplexType, 5>> G_tskij,
       std::optional<nda::array<ComplexType, 5>> DeltaG_tskij,
       std::optional<nda::array<ComplexType, 4>> DeltaX_left,
       std::optional<nda::array<ComplexType, 4>> DeltaX_right) {
      return coqui_py::lr_gw_Pi(h_int, q_pert, G_tskij, DeltaG_tskij,
                                DeltaX_left, DeltaX_right);
    },
    "h_int", "q_pert", "G_tskij", "DeltaG_tskij", "DeltaX_left",
    "DeltaX_right")};

// lr_gw_W
static auto const fun_8 = c2py::dispatcher_f_kw_t{c2py::cfun(
    [](const std::string &lr_params, coqui_py::ThcCoulomb &h_int,
       const nda::array<double, 1> &q_pert,
       std::optional<nda::array<ComplexType, 4>> DeltaPi_tqPQ) {
      return coqui_py::lr_gw_W(lr_params, h_int, q_pert, DeltaPi_tqPQ);
    },
    "lr_params", "h_int", "q_pert", "DeltaPi_tqPQ")};

// lr_gw_sigma_DeltaG
static auto const fun_9 = c2py::dispatcher_f_kw_t{c2py::cfun(
    [](const std::string &lr_params, coqui_py::ThcCoulomb &h_int,
       const nda::array<double, 1> &q_pert,
       std::optional<nda::array<ComplexType, 5>> DeltaG_tskij) {
      return coqui_py::lr_gw_sigma_DeltaG(lr_params, h_int, q_pert,
                                          DeltaG_tskij);
    },
    "lr_params", "h_int", "q_pert", "DeltaG_tskij")};

// lr_gw_sigma_DeltaW
static auto const fun_10 = c2py::dispatcher_f_kw_t{c2py::cfun(
    [](const std::string &lr_params, coqui_py::ThcCoulomb &h_int,
       const nda::array<double, 1> &q_pert,
       std::optional<nda::array<ComplexType, 5>> G_tskij,
       std::optional<nda::array<ComplexType, 4>> DeltaW_qtPQ) {
      return coqui_py::lr_gw_sigma_DeltaW(lr_params, h_int, q_pert, G_tskij,
                                          DeltaW_qtPQ);
    },
    "lr_params", "h_int", "q_pert", "G_tskij", "DeltaW_qtPQ")};

// lr_hf
static auto const fun_11 = c2py::dispatcher_f_kw_t{c2py::cfun(
    [](coqui_py::ThcCoulomb &h_int, const nda::array<double, 1> &q_vec,
       const nda::array<ComplexType, 4> &DeltaDm_skij,
       const nda::array<ComplexType, 4> &S_skij, bool compute_hartree,
       bool compute_exchange) {
      return coqui_py::lr_hf(h_int, q_vec, DeltaDm_skij, S_skij,
                             compute_hartree, compute_exchange);
    },
    "h_int", "q_vec", "DeltaDm_skij", "S_skij", "compute_hartree",
    "compute_exchange")};

// lr_qp_approx
static auto const fun_12 = c2py::dispatcher_f_kw_t{c2py::cfun(
    [](coqui_py::ThcCoulomb &h_int, const std::string &prefix,
       const nda::array<ComplexType, 5> &DeltaSigma_tskij,
       const nda::array<ComplexType, 4> &MO_skia,
       const nda::array<ComplexType, 3> &E_ska, double mu,
       const nda::array<long, 1> &kpq_map, bool q_is_gamma,
       std::string off_diag_mode, std::string ac_alg, int Nfit, double eta) {
      return coqui_py::lr_qp_approx(h_int, prefix, DeltaSigma_tskij, MO_skia,
                                    E_ska, mu, kpq_map, q_is_gamma,
                                    off_diag_mode, ac_alg, Nfit, eta);
    },
    "h_int", "prefix", "DeltaSigma_tskij", "MO_skia", "E_ska", "mu", "kpq_map",
    "q_is_gamma", "off_diag_mode", "ac_alg", "Nfit", "eta")};

// mbpt
static auto const fun_13 = c2py::dispatcher_f_kw_t{
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
static auto const fun_14 = c2py::dispatcher_f_kw_t{c2py::cfun(
    [](const std::string &lr_params, coqui_py::ThcCoulomb &h_int,
       const nda::array<double, 1> &q_vec,
       std::optional<nda::array<ComplexType, 5>> DeltaH0_mskij,
       bool include_hartree, bool include_exchange, std::string gw_mode_str,
       int max_iter, double tol, bool fix_density, std::string iter_alg,
       double mixing, int max_subsp_size, int diis_warmup,
       std::optional<nda::array<ComplexType, 4>> DeltaX_left,
       std::optional<nda::array<ComplexType, 4>> DeltaX_right,
       std::optional<nda::array<ComplexType, 3>> DeltaV_qPQ) {
      return coqui_py::run_lr(lr_params, h_int, q_vec, DeltaH0_mskij,
                              include_hartree, include_exchange, gw_mode_str,
                              max_iter, tol, fix_density, iter_alg, mixing,
                              max_subsp_size, diis_warmup, DeltaX_left,
                              DeltaX_right, DeltaV_qPQ);
    },
    "lr_params", "h_int", "q_vec", "DeltaH0_mskij", "include_hartree",
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
static const auto doc_d_1 = fun_1.doc(
    R"DOC(
Compute eps_inv_head from W_c in THC product basis

Parameters
----------
lr_params : {par_0}
   - [INPUT] JSON string with params (prefix for IAFT)
h_int : {par_1}
   - [INPUT] THC ERI handler
W_c_tqPQ : {par_2}
   - [INPUT] Correlation screened interaction W_c (nt_half, nkpts, NP, NP)

Returns
-------
{ret_0}
   - [OUTPUT] eps_inv_head (nt_half,)
)DOC",
    {{c2py::python_typename<const std::string &>()},
     {c2py::python_typename<coqui_py::ThcCoulomb &>()},
     {c2py::python_typename<std::optional<nda::array<ComplexType, 4>>>()}},
    {c2py::python_typename<nda::array<ComplexType, 1>>()});
static const auto doc_d_2 = fun_2.doc(
    R"DOC(
Evaluate standard RPA polarization P[G] (FD helper)

Parameters
----------
lr_params : {par_0}
   - [INPUT] JSON string with params (prefix)
h_int : {par_1}
   - [INPUT] THC ERI handler
G_tskij : {par_2}
   - [INPUT] Green's function (nt, ns, nk, nb, nb)

Returns
-------
{ret_0}
   - [OUTPUT] P (nt_half, nkpts, NP, NP)
)DOC",
    {{c2py::python_typename<const std::string &>()},
     {c2py::python_typename<coqui_py::ThcCoulomb &>()},
     {c2py::python_typename<std::optional<nda::array<ComplexType, 5>>>()}},
    {c2py::python_typename<nda::array<ComplexType, 4>>()});
static const auto doc_d_3 = fun_3.doc(
    R"DOC(
Evaluate W_c from Π via W Dyson equation (FD helper)

Parameters
----------
lr_params : {par_0}
   - [INPUT] JSON string with params (prefix)
h_int : {par_1}
   - [INPUT] THC ERI handler
Pi_tqPQ : {par_2}
   - [INPUT] Polarization (nt_half, nkpts, NP, NP)

Returns
-------
{ret_0}
   - [OUTPUT] W_c (nt_half, nkpts, NP, NP)
)DOC",
    {{c2py::python_typename<const std::string &>()},
     {c2py::python_typename<coqui_py::ThcCoulomb &>()},
     {c2py::python_typename<std::optional<nda::array<ComplexType, 4>>>()}},
    {c2py::python_typename<nda::array<ComplexType, 4>>()});
static const auto doc_d_4 = fun_4.doc(
    R"DOC(
Evaluate GW self-energy Σ = -G ⊙ W_c [+ div_corr] using W from file

Parameters
----------
lr_params : {par_0}
   - [INPUT] JSON string with params (prefix)
h_int : {par_1}
   - [INPUT] THC ERI handler
G_tskij : {par_2}
   - [INPUT] Green's function (nt, ns, nk, nb, nb)
div_corr : {par_3}
   - [INPUT] Whether to apply divergence correction

Returns
-------
{ret_0}
   - [OUTPUT] Σ (nt, ns, nk, nb, nb)
)DOC",
    {{c2py::python_typename<const std::string &>()},
     {c2py::python_typename<coqui_py::ThcCoulomb &>()},
     {c2py::python_typename<std::optional<nda::array<ComplexType, 5>>>()},
     {c2py::python_typename<bool>()}},
    {c2py::python_typename<nda::array<ComplexType, 5>>()});
static const auto doc_d_5 = fun_5.doc(
    R"DOC(
Evaluate GW self-energy with provided W and G (FD helper)

Parameters
----------
lr_params : {par_0}
   - [INPUT] JSON string with params (prefix)
h_int : {par_1}
   - [INPUT] THC ERI handler
G_tskij : {par_2}
   - [INPUT] Green's function (nt, ns, nk, nb, nb)
W_c_qtPQ : {par_3}
   - [INPUT] Screened interaction (nkpts, nt_half, NP, NP)
eps_inv_head : {par_4}
   - [INPUT] Inverse dielectric head (nt_half,)
div_corr : {par_5}
   - [INPUT] Whether to apply divergence correction

Returns
-------
{ret_0}
   - [OUTPUT] Σ (nt, ns, nk, nb, nb)
)DOC",
    {{c2py::python_typename<const std::string &>()},
     {c2py::python_typename<coqui_py::ThcCoulomb &>()},
     {c2py::python_typename<std::optional<nda::array<ComplexType, 5>>>()},
     {c2py::python_typename<std::optional<nda::array<ComplexType, 4>>>()},
     {c2py::python_typename<const nda::array<ComplexType, 1> &>()},
     {c2py::python_typename<bool>()}},
    {c2py::python_typename<nda::array<ComplexType, 5>>()});
static const auto doc_d_6 =
    fun_6.doc(R"DOC(
Compute HF self-energy (Fock matrix) from a density matrix

Parameters
----------
h_int : {par_0}
   - [INPUT] THC or Cholesky ERI handler
Dm_skij : {par_1}
   - [INPUT] Density matrix (ns, nk, nb, nb)
S_skij : {par_2}
   - [INPUT] Overlap matrix (ns, nk, nb, nb)
compute_hartree : {par_3}
   - [INPUT] Whether to compute Hartree term
compute_exchange : {par_4}
   - [INPUT] Whether to compute Exchange term

Returns
-------
{ret_0}
   - [OUTPUT] Fock matrix (ns, nk, nb, nb)
)DOC",
              {{c2py::python_typename<coqui_py::ThcCoulomb &>(),
                c2py::python_typename<coqui_py::CholCoulomb &>()},
               {c2py::python_typename<const nda::array<ComplexType, 4> &>()},
               {c2py::python_typename<const nda::array<ComplexType, 4> &>()},
               {c2py::python_typename<bool>()},
               {c2py::python_typename<bool>()}},
              {c2py::python_typename<nda::array<ComplexType, 4>>()});
static const auto doc_d_7 = fun_7.doc(
    R"DOC(
Compute LR polarization ΔP = -ΔG·G - G·ΔG (R-space)

Parameters
----------
h_int : {par_0}
   - [INPUT] THC ERI handler
q_pert : {par_1}
   - [INPUT] LR perturbation wavevector (3,)
G_tskij : {par_2}
   - [INPUT] Unperturbed Green's function (nt, ns, nk, nb, nb)
DeltaG_tskij : {par_3}
   - [INPUT] LR Green's function (nt, ns, nk, nb, nb)
DeltaX_left : {par_4}
   - [INPUT, optional] δ^q X(k) (ns, nkpts, NP, nb)
DeltaX_right : {par_5}
   - [INPUT, optional] δ^{-q} X(k+q) at storage k
                           When both are provided, primary→aux IBC is applied.

Returns
-------
{ret_0}
   - [OUTPUT] ΔP (nt_half, nkpts, NP, NP)
)DOC",
    {{c2py::python_typename<coqui_py::ThcCoulomb &>()},
     {c2py::python_typename<const nda::array<double, 1> &>()},
     {c2py::python_typename<std::optional<nda::array<ComplexType, 5>>>()},
     {c2py::python_typename<std::optional<nda::array<ComplexType, 5>>>()},
     {c2py::python_typename<std::optional<nda::array<ComplexType, 4>>>()},
     {c2py::python_typename<std::optional<nda::array<ComplexType, 4>>>()}},
    {c2py::python_typename<nda::array<ComplexType, 4>>()});
static const auto doc_d_8 = fun_8.doc(
    R"DOC(
Compute LR screened interaction ΔW = (Z+W_c) · ΔΠ · (Z+W_c)

Parameters
----------
lr_params : {par_0}
   - [INPUT] JSON string with params (prefix)
h_int : {par_1}
   - [INPUT] THC ERI handler
q_pert : {par_2}
   - [INPUT] LR perturbation wavevector (3,)
DeltaPi_tqPQ : {par_3}
   - [INPUT] LR polarization (nt_half, nkpts, NP, NP)

Returns
-------
{ret_0}
   - [OUTPUT] ΔW_c (nt_half, nkpts, NP, NP)
)DOC",
    {{c2py::python_typename<const std::string &>()},
     {c2py::python_typename<coqui_py::ThcCoulomb &>()},
     {c2py::python_typename<const nda::array<double, 1> &>()},
     {c2py::python_typename<std::optional<nda::array<ComplexType, 4>>>()}},
    {c2py::python_typename<nda::array<ComplexType, 4>>()});
static const auto doc_d_9 = fun_9.doc(
    R"DOC(
Compute LR GW self-energy term 1: ΔΣ = -ΔG ⊙ W_c + div_corr (fixed W, R-space)

Parameters
----------
lr_params : {par_0}
   - [INPUT] JSON string with params (prefix)
h_int : {par_1}
   - [INPUT] THC ERI handler
q_pert : {par_2}
   - [INPUT] LR perturbation wavevector (3,)
DeltaG_tskij : {par_3}
   - [INPUT] LR Green's function (nt, ns, nk, nb, nb)

Returns
-------
{ret_0}
   - [OUTPUT] ΔΣ (nt, ns, nk, nb, nb)
)DOC",
    {{c2py::python_typename<const std::string &>()},
     {c2py::python_typename<coqui_py::ThcCoulomb &>()},
     {c2py::python_typename<const nda::array<double, 1> &>()},
     {c2py::python_typename<std::optional<nda::array<ComplexType, 5>>>()}},
    {c2py::python_typename<nda::array<ComplexType, 5>>()});
static const auto doc_d_10 = fun_10.doc(
    R"DOC(
Compute LR GW self-energy term 2: -G ⊙ ΔW (no div correction)

Computes ΔΣ = -G ⊙ ΔW from a pre-computed DeltaW.

Parameters
----------
lr_params : {par_0}
   - [INPUT] JSON string with params (prefix)
h_int : {par_1}
   - [INPUT] THC ERI handler
q_pert : {par_2}
   - [INPUT] LR perturbation wavevector (3,)
G_tskij : {par_3}
   - [INPUT] Unperturbed Green's function (nt, ns, nk, nb, nb)
DeltaW_qtPQ : {par_4}
   - [INPUT] LR screened interaction (nkpts, nt_half, NP, NP)

Returns
-------
{ret_0}
   - [OUTPUT] ΔΣ (nt, ns, nk, nb, nb)
)DOC",
    {{c2py::python_typename<const std::string &>()},
     {c2py::python_typename<coqui_py::ThcCoulomb &>()},
     {c2py::python_typename<const nda::array<double, 1> &>()},
     {c2py::python_typename<std::optional<nda::array<ComplexType, 5>>>()},
     {c2py::python_typename<std::optional<nda::array<ComplexType, 4>>>()}},
    {c2py::python_typename<nda::array<ComplexType, 5>>()});
static const auto doc_d_11 =
    fun_11.doc(R"DOC(
Compute LR Fock matrix from LR density matrix

Computes ΔF = ΔJ + ΔK from the LR density matrix ΔDm.

Parameters
----------
h_int : {par_0}
   - [INPUT] THC ERI handler
q_vec : {par_1}
   - [INPUT] Perturbation wavevector (3,)
DeltaDm_skij : {par_2}
   - [INPUT] LR density matrix (ns, nk, nb, nb)
S_skij : {par_3}
   - [INPUT] Overlap matrix (ns, nk, nb, nb)
compute_hartree : {par_4}
   - [INPUT] Whether to compute Hartree term
compute_exchange : {par_5}
   - [INPUT] Whether to compute Exchange term

Returns
-------
{ret_0}
   - [OUTPUT] LR Fock matrix (ns, nk, nb, nb)
)DOC",
               {{c2py::python_typename<coqui_py::ThcCoulomb &>()},
                {c2py::python_typename<const nda::array<double, 1> &>()},
                {c2py::python_typename<const nda::array<ComplexType, 4> &>()},
                {c2py::python_typename<const nda::array<ComplexType, 4> &>()},
                {c2py::python_typename<bool>()},
                {c2py::python_typename<bool>()}},
               {c2py::python_typename<nda::array<ComplexType, 4>>()});
static const auto doc_d_12 =
    fun_12.doc(R"DOC(
Statify a dynamic LR self-energy ΔΣ into a static ΔV_QPGW (test API)

Python entry point for methods::lr_qp_approx (the q-aware LR-qpGW static
map). Wraps the numpy inputs into node-shared arrays, reads the IAFT from
the checkpoint, builds a qp_params_t from the AC parameters, and returns the
resulting static ΔV_QPGW(k) in the primary basis.

Parameters
----------
h_int : {par_0}
   - [INPUT] THC ERI handler (source of MPI + MF)
prefix : {par_1}
   - [INPUT] checkpoint prefix; IAFT read from prefix.mbpt.h5
DeltaSigma_tskij : {par_2}
   - [INPUT] dynamic ΔΣ_k(τ), (nt, ns, nk, nb, nb)
MO_skia : {par_3}
   - [INPUT] frozen QP MO coefficients C, (ns, nk, nb, nb)
E_ska : {par_4}
   - [INPUT] frozen QP energies ε, (ns, nk, nb)
mu : {par_5}
   - [INPUT] frozen chemical potential
kpq_map : {par_6}
   - [INPUT] k → k+q index map, (nk,)
q_is_gamma : {par_7}
   - [INPUT] whether q ≈ 0 (enables Hermitization)
off_diag_mode : {par_8}
   - [INPUT] "qp_energy" or "fermi"
ac_alg : {par_9}
   - [INPUT] analytic-continuation algorithm (e.g. "pade")
Nfit : {par_10}
   - [INPUT] # of AC fit parameters
eta : {par_11}
   - [INPUT] AC broadening

Returns
-------
{ret_0}
   - [OUTPUT] static ΔV_QPGW(k), (ns, nk, nb, nb)
)DOC",
               {{c2py::python_typename<coqui_py::ThcCoulomb &>()},
                {c2py::python_typename<const std::string &>()},
                {c2py::python_typename<const nda::array<ComplexType, 5> &>()},
                {c2py::python_typename<const nda::array<ComplexType, 4> &>()},
                {c2py::python_typename<const nda::array<ComplexType, 3> &>()},
                {c2py::python_typename<double>()},
                {c2py::python_typename<const nda::array<long, 1> &>()},
                {c2py::python_typename<bool>()},
                {c2py::python_typename<std::string>()},
                {c2py::python_typename<std::string>()},
                {c2py::python_typename<int>()},
                {c2py::python_typename<double>()}},
               {c2py::python_typename<nda::array<ComplexType, 4>>()});
static const auto doc_d_13 = fun_13.doc(R"DOC()DOC");
static const auto doc_d_14 = fun_14.doc(
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
DeltaH0_mskij : {par_3}
   - [INPUT] Perturbations (nmodes, ns, nk, nb, nb);
                              required on the MPI global root, None elsewhere.
                              All share the one q_vec; each is written to its
                              own "linear_response/mode{m}" group when nmodes > 1.
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
   - [OUTPUT] Tuple of per-mode (niter, Delta_mu) arrays
)DOC",
    {{c2py::python_typename<const std::string &>()},
     {c2py::python_typename<coqui_py::ThcCoulomb &>()},
     {c2py::python_typename<const nda::array<double, 1> &>()},
     {c2py::python_typename<std::optional<nda::array<ComplexType, 5>>>()},
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
    {c2py::python_typename<
        std::tuple<nda::array<long, 1>, nda::array<double, 1>>>()});
//--------------------- module function table  -----------------------------

static PyMethodDef module_methods[] = {
    {"calculate_kpq_map", (PyCFunction)c2py::pyfkw<fun_0>,
     METH_VARARGS | METH_KEYWORDS, doc_d_0.c_str()},
    {"compute_eps_inv_head", (PyCFunction)c2py::pyfkw<fun_1>,
     METH_VARARGS | METH_KEYWORDS, doc_d_1.c_str()},
    {"gw_evaluate_Pi", (PyCFunction)c2py::pyfkw<fun_2>,
     METH_VARARGS | METH_KEYWORDS, doc_d_2.c_str()},
    {"gw_evaluate_W_from_Pi", (PyCFunction)c2py::pyfkw<fun_3>,
     METH_VARARGS | METH_KEYWORDS, doc_d_3.c_str()},
    {"gw_evaluate_sigma", (PyCFunction)c2py::pyfkw<fun_4>,
     METH_VARARGS | METH_KEYWORDS, doc_d_4.c_str()},
    {"gw_evaluate_sigma_with_W", (PyCFunction)c2py::pyfkw<fun_5>,
     METH_VARARGS | METH_KEYWORDS, doc_d_5.c_str()},
    {"hf_evaluate", (PyCFunction)c2py::pyfkw<fun_6>,
     METH_VARARGS | METH_KEYWORDS, doc_d_6.c_str()},
    {"lr_gw_Pi", (PyCFunction)c2py::pyfkw<fun_7>, METH_VARARGS | METH_KEYWORDS,
     doc_d_7.c_str()},
    {"lr_gw_W", (PyCFunction)c2py::pyfkw<fun_8>, METH_VARARGS | METH_KEYWORDS,
     doc_d_8.c_str()},
    {"lr_gw_sigma_DeltaG", (PyCFunction)c2py::pyfkw<fun_9>,
     METH_VARARGS | METH_KEYWORDS, doc_d_9.c_str()},
    {"lr_gw_sigma_DeltaW", (PyCFunction)c2py::pyfkw<fun_10>,
     METH_VARARGS | METH_KEYWORDS, doc_d_10.c_str()},
    {"lr_hf", (PyCFunction)c2py::pyfkw<fun_11>, METH_VARARGS | METH_KEYWORDS,
     doc_d_11.c_str()},
    {"lr_qp_approx", (PyCFunction)c2py::pyfkw<fun_12>,
     METH_VARARGS | METH_KEYWORDS, doc_d_12.c_str()},
    {"mbpt", (PyCFunction)c2py::pyfkw<fun_13>, METH_VARARGS | METH_KEYWORDS,
     doc_d_13.c_str()},
    {"run_lr", (PyCFunction)c2py::pyfkw<fun_14>, METH_VARARGS | METH_KEYWORDS,
     doc_d_14.c_str()},
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
