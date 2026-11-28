/**
 * ==========================================================================
 * CoQuí: Correlated Quantum ínterface
 *
 * Copyright (c) 2022-2026 Simons Foundation & The CoQuí developer team
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

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <limits>

#include "itertools/itertools.hpp"
#include "IO/AppAbort.hpp"
#include "IO/app_loggers.h"
#include "IO/ptree/ptree_utilities.hpp"
#include "mean_field/MF.hpp"
#include "methods/ERI/mb_eri_context.h"
#include "methods/SCF/lr_driver.hpp"
#include "methods/SCF/lr_hessian.hpp"
#include "methods/SCF/lr_setup.hpp"
#include "methods/SCF/lr_dyson.hpp"
#include "methods/SCF/simple_dyson.h"
#include "methods/MBPT_drivers.h"
#include "nda/blas.hpp"
#include "utilities/check.hpp"
#include "numerics/imag_axes_ft/iaft_utils.hpp"
#include "methods/SCF/lr_efield.hpp"

// Band-basis building blocks (declared in lr_efield.hpp) and the driver run_lr_efield_calc (declared in
// MBPT_drivers.h, bound to Python as run_lr_efield). The LR setup it shares with run_lr_calc is in lr_setup.hpp.

namespace methods {

namespace {
using matrix_t = nda::matrix<ComplexType>;
/// Σ_ij A_ij B_ji = tr[A B]
inline ComplexType trace_of_product(matrix_t const& A, matrix_t const& B) {
  ComplexType t = 0.0;
  const long n = A.extent(0);
  for (long i = 0; i < n; ++i)
    for (long j = 0; j < n; ++j) t += A(i, j) * B(j, i);
  return t;
}
} // namespace

void build_efield_lr_source(utils::mpi_context_t<mpi3::communicator>& mpi,
                            sArray_t<Array_view_5D_t> const& sG_wskij, sArray_t<Array_view_5D_t> const& sB_wskij,
                            std::array<sArray_t<Array_view_5D_t> const*, 3> sv_wskij,
                            sArray_t<Array_view_6D_t>& sVE_xwskij) {
  decltype(nda::range::all) all;
  auto [nw, ns, nk, nb, nb2] = sG_wskij.shape();
  utils::check(sVE_xwskij.shape() == std::array<long, 6>{3, nw, ns, nk, nb, nb2},
               "build_efield_lr_source: V^E must have the shape (3, nw_f, ns, nk, nb, nb) of G(iw).");
  const ComplexType minus_i_half(0.0, -0.5);
  sVE_xwskij.node_sync();
  matrix_t G(nb, nb), B(nb, nb), v(nb, nb), GB(nb, nb), BG(nb, nb), VE(nb, nb);
  for (long nsk = mpi.node_comm.rank(); nsk < nw * ns * nk; nsk += mpi.node_comm.size()) {
    long n = nsk / (ns * nk), is = (nsk / nk) % ns, ik = nsk % nk;
    G = sG_wskij.local()(n, is, ik, all, all);
    B = sB_wskij.local()(n, is, ik, all, all);
    // 1. GB = G·B, BG = B·G
    GB = G * B;
    BG = B * G;
    // 2. V^E_α = -(i/2)(v_α·GB - BG·v_α)
    for (long a = 0; a < 3; ++a) {
      v = sv_wskij[a]->local()(n, is, ik, all, all);
      VE = minus_i_half * (v * GB - BG * v);
      sVE_xwskij.local()(a, n, is, ik, all, all) = VE;
    }
  }
  sVE_xwskij.node_sync();
}

void build_sigma2_lr_source(utils::mpi_context_t<mpi3::communicator>& mpi,
                            sArray_t<Array_view_5D_t> const& sG_wskij, sArray_t<Array_view_5D_t> const& sB_wskij,
                            sArray_t<Array_view_5D_t>& sDeltaH_wskij) {
  decltype(nda::range::all) all;
  auto [nw, ns, nk, nb, nb2] = sG_wskij.shape();
  utils::check(sB_wskij.shape() == sG_wskij.shape() and sDeltaH_wskij.shape() == sG_wskij.shape(),
               "build_sigma2_lr_source: B(iw) and the source 2BGB must have the shape (nw_f, ns, nk, nb, nb) "
               "of G(iw).");
  sDeltaH_wskij.node_sync();
  matrix_t G(nb, nb), B(nb, nb), BG(nb, nb), BGB(nb, nb);
  for (long nsk = mpi.node_comm.rank(); nsk < nw * ns * nk; nsk += mpi.node_comm.size()) {
    long n = nsk / (ns * nk), is = (nsk / nk) % ns, ik = nsk % nk;
    G = sG_wskij.local()(n, is, ik, all, all);
    B = sB_wskij.local()(n, is, ik, all, all);
    // 1. 2 B G B
    BG = B * G;
    BGB = 2.0 * (BG * B);
    sDeltaH_wskij.local()(n, is, ik, all, all) = BGB;
  }
  sDeltaH_wskij.node_sync();
}

efield_trace_terms contract_efield_terms(utils::mpi_context_t<mpi3::communicator>& mpi,
                                         imag_axes_ft::IAFT const& FT,
                                         sArray_t<Array_view_5D_t> const& sG_wskij,
                                         sArray_t<Array_view_5D_t> const& sB_wskij,
                                         std::array<sArray_t<Array_view_5D_t> const*, 3> sv_wskij,
                                         sArray_t<Array_view_5D_t> const& sSigma2_wskij,
                                         nda::array<double, 1> const& k_weight, double spin_factor,
                                         double volume) {
  decltype(nda::range::all) all;
  auto [nw, ns, nk, nb, nb2] = sG_wskij.shape();
  // trace_vGpvGp_wab(n, a, b), trace_Sigma2_wab(n, a, b): the (iω_n) trace of each term, summed over this rank's (s,k)
  nda::array<ComplexType, 3> trace_vGpvGp_wab(nw, 3, 3), trace_Sigma2_wab(nw, 3, 3);
  trace_vGpvGp_wab() = 0.0;
  trace_Sigma2_wab() = 0.0;
  matrix_t G(nb, nb), B(nb, nb), Sigma2(nb, nb), GB(nb, nb), Gp(nb, nb), GSigma2(nb, nb), GSigma2G(nb, nb);
  std::array<matrix_t, 3> v, vGp, GSigma2Gv, Gv;
  for (long a = 0; a < 3; ++a) {
    v[a] = matrix_t(nb, nb);
    vGp[a] = matrix_t(nb, nb);
    GSigma2Gv[a] = matrix_t(nb, nb);
    Gv[a] = matrix_t(nb, nb);
  }
  auto [nsk0, nsk1] = itertools::chunk_range(0, nw * ns * nk, long(mpi.comm.size()), long(mpi.comm.rank()));
  for (long nsk = nsk0; nsk < nsk1; ++nsk) {
    long n = nsk / (ns * nk), is = (nsk / nk) % ns, ik = nsk % nk;
    const double w = spin_factor * k_weight(ik);
    G = sG_wskij.local()(n, is, ik, all, all);
    B = sB_wskij.local()(n, is, ik, all, all);
    Sigma2 = sSigma2_wskij.local()(n, is, ik, all, all);
    // 1. G' = -G B G
    GB = G * B;
    Gp = -1.0 * (GB * G);
    // 2. term 2: vGp_α = v_α G',  trace_vGpvGp_wab += w tr[vGp_a vGp_b] = w tr[v_a G' v_b G']
    // 3. term 3: GSigma2Gv_α = (G Σ'' G) v_α, Gv_β = G v_β,
    //    trace_Sigma2_wab += w tr[GSigma2Gv_a Gv_b] = w tr[Σ'' G v_a G v_b G]
    GSigma2 = G * Sigma2;
    GSigma2G = GSigma2 * G;
    for (long a = 0; a < 3; ++a) {
      v[a] = sv_wskij[a]->local()(n, is, ik, all, all);
      vGp[a] = v[a] * Gp;
      GSigma2Gv[a] = GSigma2G * v[a];
      Gv[a] = G * v[a];
    }
    for (long a = 0; a < 3; ++a)
      for (long b = 0; b < 3; ++b) {
        trace_vGpvGp_wab(n, a, b) += w * trace_of_product(vGp[a], vGp[b]);
        trace_Sigma2_wab(n, a, b) += w * trace_of_product(GSigma2Gv[a], Gv[b]);
      }
  }
  // 4. Σ_{s,k} over the global communicator
  mpi.comm.all_reduce_in_place_n(trace_vGpvGp_wab.data(), trace_vGpvGp_wab.size(), std::plus<>{});
  mpi.comm.all_reduce_in_place_n(trace_Sigma2_wab.data(), trace_Sigma2_wab.size(), std::plus<>{});
  // 5. term = (1/(4Ω)) (1/β) Σ_n trace(iω_n)
  efield_trace_terms terms{nda::array<ComplexType, 2>(3, 3), nda::array<ComplexType, 2>(3, 3)};
  nda::array<ComplexType, 2> trace_w(nw, 1);
  for (long a = 0; a < 3; ++a)
    for (long b = 0; b < 3; ++b) {
      trace_w(all, 0) = trace_vGpvGp_wab(all, a, b);
      terms.chi2_vGpvGp(a, b) = imag_axes_ft::matsubara_sum(FT, trace_w, imag_axes_ft::fermion, /*w_weighted=*/true) / (4.0 * volume);
      trace_w(all, 0) = trace_Sigma2_wab(all, a, b);
      terms.chi2_Sigma2(a, b) = imag_axes_ft::matsubara_sum(FT, trace_w, imag_axes_ft::fermion, /*w_weighted=*/true) / (4.0 * volume);
    }
  return terms;
}

double max_trace_G_half_beta(imag_axes_ft::IAFT const& FT, sArray_t<Array_view_5D_t> const& sG_tskij) {
  decltype(nda::range::all) all;
  auto [nt, ns, nk, nb, nb2] = sG_tskij.shape();
  // 1. τ* = the node of smallest |x|, x = 2τ/β - 1 (τ = β/2)
  auto const& tau = FT.tau_mesh_f();
  long it = 0;
  for (long t = 1; t < nt; ++t)
    if (std::abs(tau(t)) < std::abs(tau(it))) it = t;
  // 2. max_{s,k} |tr G(τ*)|
  double max_tr = 0.0;
  for (long is = 0; is < ns; ++is)
    for (long ik = 0; ik < nk; ++ik) {
      ComplexType tr = 0.0;
      for (long i = 0; i < nb; ++i) tr += sG_tskij.local()(it, is, ik, i, i);
      max_tr = std::max(max_tr, std::abs(tr));
    }
  return max_tr;
}

// utils::striped_norm (element_partition.hpp) is not reused: it takes a pre-sliced pair (a, b) and returns √‖a‖², while
// this needs ‖A‖² of a whole node-shared window.
template<typename sArray_type>
double striped_norm2(utils::mpi_context_t<mpi3::communicator>& mpi, sArray_type const& sA) {
  // 1. This rank's contiguous chunk of the flattened array
  ComplexType const* data = sA.local().data();
  auto [i0, i1] = itertools::chunk_range(0, long(sA.local().size()), long(mpi.comm.size()), long(mpi.comm.rank()));
  double norm2 = 0.0;
  for (long i = i0; i < i1; ++i) norm2 += std::norm(data[i]);
  // 2. Sum over the ranks
  return mpi.comm.all_reduce_value(norm2);
}

template double striped_norm2(utils::mpi_context_t<mpi3::communicator>&, sArray_t<Array_view_4D_t> const&);
template double striped_norm2(utils::mpi_context_t<mpi3::communicator>&, sArray_t<Array_view_5D_t> const&);

double B_minus_S_relative_norm(utils::mpi_context_t<mpi3::communicator>& mpi, sArray_t<Array_view_5D_t> const& sB_wskij,
                               sArray_t<Array_view_4D_t> const& sS_skij, double S_norm2) {
  auto [nw, ns, nk, nb, nb2] = sB_wskij.shape();
  // 1. ‖B - S‖² over (n, s, k, ij), the flattened (n, s, k) striped over the global communicator
  double B_minus_S_norm2 = 0.0;
  auto [nsk0, nsk1] = itertools::chunk_range(0, nw * ns * nk, long(mpi.comm.size()), long(mpi.comm.rank()));
  for (long nsk = nsk0; nsk < nsk1; ++nsk) {
    long n = nsk / (ns * nk), is = (nsk / nk) % ns, ik = nsk % nk;
    for (long i = 0; i < nb; ++i)
      for (long j = 0; j < nb2; ++j)
        B_minus_S_norm2 += std::norm(sB_wskij.local()(n, is, ik, i, j) - sS_skij.local()(is, ik, i, j));
  }
  B_minus_S_norm2 = mpi.comm.all_reduce_value(B_minus_S_norm2);
  // 2. √(‖B - S‖² / (nw ‖S‖²))
  return std::sqrt(B_minus_S_norm2 / (double(nw) * S_norm2));
}

nda::array<double, 2> norm_S_sk(sArray_t<Array_view_4D_t> const& sS_skij) {
  auto [ns, nk, nb, nb2] = sS_skij.shape();
  nda::array<double, 2> S_norm_sk(ns, nk);
  for (long is = 0; is < ns; ++is)
    for (long ik = 0; ik < nk; ++ik) {
      double norm2 = 0.0;
      for (long i = 0; i < nb; ++i)
        for (long j = 0; j < nb2; ++j) norm2 += std::norm(sS_skij.local()(is, ik, i, j));
      S_norm_sk(is, ik) = std::sqrt(norm2);
    }
  return S_norm_sk;
}

double trace_GE_relative(ComplexType dN, sArray_t<Array_view_4D_t> const& sDeltaDm_skij,
                         nda::array<double, 2> const& S_norm_sk, nda::array<double, 1> const& k_weight,
                         double spin_factor) {
  auto [ns, nk, nb, nb2] = sDeltaDm_skij.shape();
  // 1.-2. dm_scale = Σ_{s,k} w_k ‖ΔDm_sk‖ ‖S_sk‖ (the spin factor is applied in step 3)
  double dm_scale = 0.0;
  for (long is = 0; is < ns; ++is)
    for (long ik = 0; ik < nk; ++ik) {
      double norm2 = 0.0;
      for (long i = 0; i < nb; ++i)
        for (long j = 0; j < nb2; ++j) norm2 += std::norm(sDeltaDm_skij.local()(is, ik, i, j));
      dm_scale += k_weight(ik) * std::sqrt(norm2) * S_norm_sk(is, ik);
    }
  // 3. |ΔN| / (spin scale); ΔDm = 0 has no charge response
  if (dm_scale == 0.0) return 0.0;
  return std::abs(dN) / (spin_factor * dm_scale);
}

namespace {

/// Warning level of the run_lr_efield_calc diagnostics (trace_G_half_beta, |κ̄| 4π/|ε∞ - 1|, trace_GE_rel, herm_M): a
/// relative size at which the neglected or violated property would show in ε∞; chosen for the warnings, not derived.
/// trace_G_half_beta is an amplitude, the square root of the thermal occupation, so its 1e-6 is an occupation of ~1e-12.
constexpr double efield_diagnostic_tol = 1e-6;

/// The results of run_lr_efield_calc, filled phase by phase; the field names are the names of the h5 datasets.
struct efield_results {
  std::string method, precond_method;  ///< attributes of linear_response_efield/
  double tol = 0.0;
  nda::array<double, 2> eps_inf = nda::zeros<double>(3, 3);
  nda::array<ComplexType, 2> chi2 = nda::zeros<ComplexType>(3, 3), chi2_GE_sym = nda::zeros<ComplexType>(3, 3),
                             chi2_GE_plain = nda::zeros<ComplexType>(3, 3);
  efield_trace_terms trace_terms;  ///< chi2_vGpvGp, chi2_Sigma2
  lr_hessian_result_t hess;        ///< herm_plain, herm_sym, herm_M
  double eps_inf_imag_max = 0.0, trace_G_half_beta = 0.0, kappa_bar = 0.0, DeltaDm_Gp_rel = 0.0, B_minus_S_rel = 0.0,
         dkappa_dmu = 0.0;
  nda::array<double, 1> trace_GE_rel = nda::zeros<double>(3);
  nda::array<long, 1> niter = nda::zeros<long>(8);  ///< [A1, v_x, v_y, v_z, Σ'', G^E_x, G^E_y, G^E_z]
};

/**
 * Writes the results of run_lr_efield_calc to the `linear_response_efield/` group of `out_file` (replaced if present).
 *
 * Called by run_lr_efield_calc once, at its end, on the global root only. The arrays and scalars are the ones
 * named in the Python `run_lr_efield` docstring, written under the same names, plus the attributes method,
 * precond_method and tol.
 */
void write_efield_results(std::string const& out_file, efield_results const& res) {
  // 1. Open the output file, replacing a previous linear_response_efield/
  h5::file file(out_file, std::filesystem::exists(out_file) ? 'a' : 'w');
  h5::group root_grp(file);
  if (root_grp.has_subgroup("linear_response_efield"))
    H5Ldelete(h5::hid_t(root_grp), "linear_response_efield", H5P_DEFAULT);
  auto grp = root_grp.create_group("linear_response_efield");
  // 2. The tensors, the diagnostics and the iteration counts
  nda::h5_write(grp, "eps_inf", res.eps_inf, false);
  nda::h5_write(grp, "chi2", res.chi2, false);
  nda::h5_write(grp, "chi2_GE_sym", res.chi2_GE_sym, false);
  nda::h5_write(grp, "chi2_GE_plain", res.chi2_GE_plain, false);
  nda::h5_write(grp, "chi2_vGpvGp", res.trace_terms.chi2_vGpvGp, false);
  nda::h5_write(grp, "chi2_Sigma2", res.trace_terms.chi2_Sigma2, false);
  h5::h5_write(grp, "herm_plain", res.hess.herm_plain);
  h5::h5_write(grp, "herm_sym", res.hess.herm_sym);
  h5::h5_write(grp, "herm_M", res.hess.herm_M);
  h5::h5_write(grp, "eps_inf_imag_max", res.eps_inf_imag_max);
  h5::h5_write(grp, "trace_G_half_beta", res.trace_G_half_beta);
  h5::h5_write(grp, "kappa_bar", res.kappa_bar);
  h5::h5_write(grp, "DeltaDm_Gp_rel", res.DeltaDm_Gp_rel);
  h5::h5_write(grp, "B_minus_S_rel", res.B_minus_S_rel);
  h5::h5_write(grp, "dkappa_dmu", res.dkappa_dmu);
  nda::h5_write(grp, "trace_GE_rel", res.trace_GE_rel, false);
  nda::h5_write(grp, "niter", res.niter, false);
  // 3. The run's settings
  h5::h5_write_attribute(grp, "method", res.method);
  h5::h5_write_attribute(grp, "precond_method", res.precond_method);
  h5::h5_write_attribute(grp, "tol", res.tol);
  app_log(2, "  - linear_response_efield/ written to {}", out_file);
}

} // namespace

/**
 * Electronic dielectric tensor ε∞ from eight Q = 0 LR solves on one ground state (note efield_lr.tex, eq:chi2work):
 *
 *   ε∞_ab = δ_ab - 4π Re χ̄⁽²⁾_ab,
 *   χ̄⁽²⁾_ab = <G^E_a, V^E_b> + 1/4 Tr[v_a G' v_b G'] + 1/4 Tr[Σ'' G v_a G v_b G]   (symmetrized in ab),
 *   V^E_a = -(i/2)(v_a G B - B G v_a),  B = S - Σ' = S + ΔF + ΔΣ(iω),  G' = -G B G.
 *
 * The three phases (kernel K_a of `method`; K^X = K_a without the Hartree and ΔW rungs, i.e. at fixed W;
 * fix_density = false throughout):
 *   A1  K_a, ΔH0 = S                   -> G', B
 *   X   K^X, v modes (ΔH0 = v0_a) and the Σ'' mode (source 2 B G B)   -> v_a, Σ'', terms 2 and 3
 *   A2  K_a, source V^E_a, hessian     -> G^E_a, term 1 = H_sym[b, a]/Ω
 * For method "GW", K_a is the full kernel K = K^H + K^X + K^W, so the A1 and A2 solves contain K^W; GW0 has K^W = 0.
 * A phase whose kernel has a Σ (GW0, GW) is preconditioned by HSEX (the static kernel {X = V + W_c(iν=0), and H only
 * if the phase kernel has it}: run_lr's precond_method = "HSEX" always carries H; default precond_tol and
 * precond_max_iter) unless iter_alg is DIIS, which cannot take a preconditioner; the preconditioner changes the rate,
 * not the answer.
 * Each phase builds its own lr_driver and W and destroys them at its end, so the peak memory is one phase's.
 *
 * Called by Python `run_lr_efield`. Writes <output>.mbpt.h5:linear_response_efield/ (replaced if present).
 *
 * Keys read from `pt`:
 *   prefix, output (default prefix), input_type ("scf"), input_iter (-1), h0_source ("compute")
 *   method          "none", "Hartree", "HF", "GW0" or "GW" (required)
 *   max_iter, tol   LR SCF iterations and tolerance of every solve (50, 1e-8)
 *   iter_alg.*      as run_lr
 *   reuse_W, recompute_W, screened_interaction_file   as run_lr. A W kept by an earlier call is replaced when
 *                   recompute_W is set without reuse_W, and that W is released at the end of this call unless
 *                   reuse_W is set. Without recompute_W and reuse_W the call neither reads nor changes the kept W.
 * The kernel, hessian, fix_density and preconditioner keys of run_lr are driven here and rejected if given.
 *
 * Requires a full-BZ, npol = 1 mean field with the velocity matrix v0 (pw2coqui add_velocity), and a ground state
 * whose self-energy is that of `method` (Σ = 0 for none/Hartree/HF, the GW Σ for GW0 and GW).
 */
template<typename eri_t>
void run_lr_efield_calc(eri_t &eri, ptree const& pt) {
  auto& thc = eri.corr_eri->get();
  auto mf = thc.MF();
  auto& mpi = thc.mpi();
  using sArray_5D_t = math::shm::shared_array<Array_view_5D_t>;
  using sArray_6D_t = math::shm::shared_array<Array_view_6D_t>;

  // 1. Parse the input and check it before anything is built
  std::string err = std::string("run_lr_efield_calc - Incorrect input - ");
  auto prefix = io::get_value<std::string>(pt, "prefix", err + "prefix");
  auto output = io::get_value_with_default<std::string>(pt, "output", prefix);
  auto input_grp = io::get_value_with_default<std::string>(pt, "input_type", "scf");
  auto input_iter = io::get_value_with_default<long>(pt, "input_iter", -1);
  auto h0_source = io::get_value_with_default<std::string>(pt, "h0_source", "compute");
  auto method = io::get_value<std::string>(pt, "method", err + "method");
  auto max_iter = io::get_value_with_default<int>(pt, "max_iter", 50);
  auto tol = io::get_value_with_default<double>(pt, "tol", 1e-8);
  auto reuse_W = io::get_value_with_default<bool>(pt, "reuse_W", false);
  auto recompute_W = io::get_value_with_default<bool>(pt, "recompute_W", false);
  auto screened_interaction_file = io::get_value_with_default<std::string>(pt, "screened_interaction_file", "");
  utils::check(tol >= 0.0, "run_lr_efield_calc: tol must be >= 0, got {}.", tol);
  utils::check(max_iter >= 1, "run_lr_efield_calc: max_iter must be >= 1, got {}.", max_iter);
  utils::check(method == "none" || method == "Hartree" || method == "HF" || method == "GW0" || method == "GW",
               "run_lr_efield_calc: method must be 'none', 'Hartree', 'HF', 'GW0' or 'GW', got '{}'.", method);
  for (auto key : {"fix_density", "lr_hessian", "split_sigma_terms", "precond_method", "precond_tol",
                   "precond_max_iter", "include_xc", "include_hartree", "include_exchange", "gw_mode",
                   "exclude_hartree_head", "div_treatment"})
    utils::check(!io::check_child_exists(pt, key),
                 "run_lr_efield_calc: '{}' is set by the driver (method fixes the kernel, fix_density is off, "
                 "the hessian is always evaluated); remove it.", key);
  utils::check(io::get_value_with_default<std::string>(pt, "unperturbed", "checkpoint") == "checkpoint",
               "run_lr_efield_calc: unperturbed must be 'checkpoint'.");
  utils::check(mf->nkpts() == mf->nkpts_ibz(),
               "run_lr_efield_calc: the mean field is symmetry-reduced (nkpts_ibz = {} < nkpts = {}). Unfold it to "
               "the full BZ first (mean_field.unfold_mf and post_proc.unfold_checkpoint).", mf->nkpts_ibz(), mf->nkpts());
  utils::check(mf->npol() == 1, "run_lr_efield_calc: npol = {} is not supported, only npol = 1.", mf->npol());
  utils::check(mf->has_v0(),
               "run_lr_efield_calc: the mean field has no velocity matrix Orbitals/v0_skxij. Regenerate it with "
               "pw2coqui add_velocity = .true. (augmented mean fields carry none).");
  std::string input_file = prefix + ".mbpt.h5";
  utils::check(std::filesystem::exists(input_file), "run_lr_efield_calc: Input checkpoint {} does not exist!", input_file);
  utils::check(!read_hf_div_treatment(*mpi, input_file, input_grp).ends_with("_metal"),
               "run_lr_efield_calc: a '*_metal' hf_div_treatment is not supported, ε∞ needs the gapped treatment.");

  const lr_kernel_spec kernel_a = kernel_spec_from_method(method);
  lr_kernel_spec kernel_x = kernel_a;  // K^X: the Q → 0 limit of χ̄ takes no Hartree rung and no ΔW rung (fixed W)
  kernel_x.hartree = false;
  kernel_x.sigma_G_dW = false;
  const bool has_sigma = kernel_a.has_sigma();
  // W is cached across phases only if the caller keeps it or it is recomputed (one G(τ) fit, not three)
  const bool cache_W = reuse_W || recompute_W;
  if (recompute_W && !reuse_W) W_cache.reset();

  utils::TimerManager init_timer;
  for (auto& timer : {"LR_INIT_DYSON", "LR_INIT_READ_SCF", "LR_INIT_UPDATE_G", "LR_INIT_LOAD_W"}) init_timer.add(timer);
  imag_axes_ft::IAFT ft(imag_axes_ft::read_iaft(input_file, false));
  const long nw = ft.nw_f(), nt = ft.nt_f(), ns = mf->nspin(), nk = mf->nkpts_ibz(), nb = mf->nbnd();
  const double volume = mf->volume();
  const double spin_factor = (ns == 1 && mf->npol() == 1) ? 2.0 : 1.0;
  nda::array<double, 1> k_weight(mf->k_weight());
  nda::array<double, 1> q_vec = {0.0, 0.0, 0.0};
  const double U_MB = double(nw) * ns * nk * nb * nb * 16.0 / 1e6;

  // 2. Unperturbed state: Dyson solver, G(iω), G(τ), Σ(τ), μ; the ground-state diagnostics
  auto state = load_lr_unperturbed_state(*mpi, *mf, ft, h0_source, prefix, input_grp, input_iter, "checkpoint",
                                         init_timer);
  auto& gs_dyson = state.dyson;
  auto& sG_wskij = state.sG_wskij;
  auto& sG_tskij = *state.opt_sG_tskij;  // alive until the end: recompute_W reads it in the first phase that builds W
  auto const& sS_skij = gs_dyson.sS_skij();
  efield_results res;
  res.method = method;
  res.tol = tol;
  res.trace_G_half_beta = max_trace_G_half_beta(ft, sG_tskij);
  const bool has_sigma_gs = striped_norm2(*mpi, state.sSigma_tskij) > 0.0;  // the checkpoint's ground state carries a GW self-energy
  { auto released = std::move(state.sSigma_tskij); }  // free Σ(τ) of the ground state: nothing below reads it
  utils::check(has_sigma_gs == has_sigma,
               "run_lr_efield_calc: method '{}' needs a ground state {} a self-energy, but the checkpoint {} {}.",
               method, has_sigma ? "with" : "without", input_file,
               has_sigma_gs ? "carries one (use method = 'GW0' or 'GW')" : "has Σ = 0 (use method = 'HF' or lower)");
  app_log(2, "run_lr_efield_calc: method {}, K_a = {{{}}}, K^X = {{{}}}; U = {:.2f} MB, efield arrays ≈ 7U per node "
             "(lr_setup prints the memory estimate of the solver incl. W)",
          method, kernel_a.to_string(), kernel_x.to_string(), U_MB);

  // 3. v0_α into a node-shared (ns, nk, 3, nb, nb) array, read once per node
  auto sv0_skxij = math::shm::make_shared_array<nda::array_view<ComplexType, 5>>(*mpi, {ns, nk, 3, nb, nb});
  if (mpi->node_comm.root()) mf->read_v0(sv0_skxij.local());
  sv0_skxij.node_sync();
  mpi->comm.barrier();

  // Arrays shared by every phase
  auto sDeltaG_tskij = math::shm::make_shared_array<Array_view_5D_t>(*mpi, {nt, ns, nk, nb, nb});
  auto sDeltaDm_skij = math::shm::make_shared_array<Array_view_4D_t>(*mpi, {ns, nk, nb, nb});
  auto sDeltaF_skij = math::shm::make_shared_array<Array_view_4D_t>(*mpi, {ns, nk, nb, nb});
  // Dynamic source of phase X: zero for the v solves, then 2 B G B for the Σ'' solve
  auto sDeltaH_wskij = math::shm::make_shared_array<Array_view_5D_t>(*mpi, {nw, ns, nk, nb, nb});
  std::optional<sArray_5D_t> opt_sDeltaSigma;
  if (has_sigma) opt_sDeltaSigma.emplace(math::shm::make_shared_array<Array_view_5D_t>(*mpi, {nt, ns, nk, nb, nb}));
  auto* pDeltaSigma = opt_sDeltaSigma ? &(*opt_sDeltaSigma) : nullptr;

  const lr_iter_params iter_params = read_lr_iter_params(pt, /*qp_static_sigma=*/false);
  const bool use_precond = iter_params.alg != "DIIS";  // false under DIIS, which takes no preconditioner
  if (has_sigma && !use_precond)
    app_log(2, "  - iter_alg DIIS takes no preconditioner: the HSEX preconditioner is dropped");

  // The lr_params of one phase; `DeltaH_wskij_source` is a view of its dynamic source (std::nullopt: none), which must
  // stay valid while p is used.
  // W must outlive the lr_driver that p is given to.
  // exclude_hartree_head stays false: at Q = 0 the THC Coulomb has no G = 0 term, so V is already V̄ (lr_hf::build_Uq_PQ).
  auto make_params = [&](lr_kernel_spec const& kernel, lr_W_inputs const& W,
                         opt_const_view_5D_t DeltaH_wskij_source) {
    lr_params p;
    fill_lr_kernel_params(p, kernel, W, iter_params, max_iter, tol, /*fix_density=*/false, /*save_DeltaG=*/false);
    if (DeltaH_wskij_source) p.DeltaH_wskij.emplace(*DeltaH_wskij_source);
    if (kernel.has_sigma() && use_precond) {  // HSEX preconditioner, whose K_pc has the Hartree rung only if the kernel has it
      p.precond_kernel = lr_kernel_spec{.hartree = kernel.hartree, .exchange = true};
      p.precond_hsex = true;
    }
    return p;
  };
  auto load_W_of_phase = [&](lr_kernel_spec const& kernel) {
    auto W = load_lr_W(*mpi, thc, ft, input_file, input_grp, pt, kernel.has_sigma(), recompute_W, cache_W,
                       screened_interaction_file, "checkpoint", sG_tskij, nullptr, init_timer);
    utils::check(!W.div_treatment.ends_with("_metal"),
                 "run_lr_efield_calc: a '*_metal' div_treatment ('{}') is not supported, ε∞ needs the gapped "
                 "treatment (note sec:largeDN).", W.div_treatment);
    return W;
  };

  std::optional<sArray_5D_t> opt_sB_wskij, opt_sSigma2_wskij;
  std::array<std::optional<sArray_5D_t>, 3> opt_sv_wskij;

  // 4. Phase A1: G' and B = S + ΔF + ΔΣ(iω), from the response to ΔH0 = S
  {
    auto W = load_W_of_phase(kernel_a);
    lr_driver driver(gs_dyson, q_vec);
    auto p = make_params(kernel_a, W, /*DeltaH_wskij_source=*/std::nullopt);
    driver.lr_setup(sG_wskij, thc, W.opt_dW ? &(*W.opt_dW) : nullptr, p);
    // 4.1 ΔH0 = S solve, ΔF and ΔΣ(τ) of its final Dyson pass
    auto [it, dmu] = driver.lr_solve_one(sDeltaG_tskij, sDeltaDm_skij, sDeltaF_skij, pDeltaSigma, sS_skij.local(),
                                         thc, p);
    res.niter(0) = it;
    // 4.2 B(iω) = S + ΔF + τ->ω ΔΣ(τ)
    opt_sB_wskij.emplace(math::shm::make_shared_array<Array_view_5D_t>(*mpi, {nw, ns, nk, nb, nb}));
    lr_total_one_body_perturbation_w(*mpi, ft, sS_skij.local(), sDeltaF_skij, pDeltaSigma, *opt_sB_wskij);
    // 4.3 κ̄ = ΔN[G']/Ω = -ΔN[ΔG]/Ω (G' = -ΔG);  ‖ΔDm‖/‖S‖;  ‖B - S‖/‖S‖ over (iω, s, k).
    //     κ̄ is real: S = S† and every source obeys ΔH(-iω) = ΔH(iω)†, so ΔDm = ΔDm† and Tr[S ΔDm] is real
    res.kappa_bar = -driver.dyson().compute_lr_Nelec(sDeltaDm_skij).real() / volume;
    const double S_norm2 = striped_norm2(*mpi, sS_skij);
    res.DeltaDm_Gp_rel = std::sqrt(striped_norm2(*mpi, sDeltaDm_skij) / S_norm2);
    res.B_minus_S_rel = B_minus_S_relative_norm(*mpi, *opt_sB_wskij, sS_skij, S_norm2);
    mpi->comm.barrier();
  }

  // 5. Phase X: v_α (ΔH0 = v0_α) and Σ'' (source 2 B G B) with K^X, then terms 2 and 3
  {
    auto W = load_W_of_phase(kernel_x);
    lr_driver driver(gs_dyson, q_vec);
    auto p = make_params(kernel_x, W, sDeltaH_wskij.local());
    driver.lr_setup(sG_wskij, thc, W.opt_dW ? &(*W.opt_dW) : nullptr, p);
    decltype(nda::range::all) all;
    // 5.1 v_α = ΔH0 + ΔF + ΔΣ(iω) of the solve with ΔH0 = v0_α and a zero source
    sDeltaH_wskij.set_zero();
    for (long a = 0; a < 3; ++a) {
      auto v0_a = sv0_skxij.local()(all, all, a, all, all);
      auto [it, dmu] = driver.lr_solve_one(sDeltaG_tskij, sDeltaDm_skij, sDeltaF_skij, pDeltaSigma, v0_a,
                                           thc, p);
      res.niter(1 + a) = it;
      opt_sv_wskij[a].emplace(math::shm::make_shared_array<Array_view_5D_t>(*mpi, {nw, ns, nk, nb, nb}));
      lr_total_one_body_perturbation_w(*mpi, ft, v0_a, sDeltaF_skij, pDeltaSigma, *opt_sv_wskij[a]);
    }
    // 5.2 Σ'' = ΔF + ΔΣ(iω) of the solve with ΔH0 = 0 and the source 2 B G B
    build_sigma2_lr_source(*mpi, sG_wskij, *opt_sB_wskij, sDeltaH_wskij);
    auto [it, dmu] = driver.lr_solve_one(sDeltaG_tskij, sDeltaDm_skij, sDeltaF_skij, pDeltaSigma,
                                         /*DeltaH0_skij=*/std::nullopt, thc, p);
    res.niter(4) = it;
    opt_sSigma2_wskij.emplace(math::shm::make_shared_array<Array_view_5D_t>(*mpi, {nw, ns, nk, nb, nb}));
    lr_total_one_body_perturbation_w(*mpi, ft, /*DeltaH0_skij=*/std::nullopt, sDeltaF_skij, pDeltaSigma,
                                     *opt_sSigma2_wskij);
    res.dkappa_dmu = driver.dyson().compute_lr_Nelec(sDeltaDm_skij).real() / volume;
    // 5.3 Terms 2 and 3, then Σ'' is not needed any more
    res.trace_terms = contract_efield_terms(*mpi, ft, sG_wskij, *opt_sB_wskij,
                                        {&*opt_sv_wskij[0], &*opt_sv_wskij[1], &*opt_sv_wskij[2]},
                                        *opt_sSigma2_wskij, k_weight, spin_factor, volume);
    opt_sSigma2_wskij.reset();
    mpi->comm.barrier();
  }

  // 6. V^E_α = -(i/2)(v_α G B - B G v_α), node-shared and kept through A2; then B and v are not needed any more
  sArray_6D_t sVE_xwskij = math::shm::make_shared_array<Array_view_6D_t>(*mpi, {3, nw, ns, nk, nb, nb});
  build_efield_lr_source(*mpi, sG_wskij, *opt_sB_wskij, {&*opt_sv_wskij[0], &*opt_sv_wskij[1], &*opt_sv_wskij[2]},
                         sVE_xwskij);
  opt_sB_wskij.reset();
  for (auto& opt_sv : opt_sv_wskij) opt_sv.reset();

  // 7. Phase A2: G^E_α with K_a and the source V^E_α, with the stationary hessian over the three modes
  {
    auto W = load_W_of_phase(kernel_a);
    lr_driver driver(gs_dyson, q_vec);
    // The source of mode a is V^E_a, a view of the node-shared stack; mode 0's for lr_setup, which reads only its presence
    decltype(nda::range::all) all;
    auto p = make_params(kernel_a, W, sVE_xwskij.local()(0, all, all, all, all, all));
    p.hessian_nmodes = 3;
    p.hessian_stationary = true;
    driver.lr_setup(sG_wskij, thc, W.opt_dW ? &(*W.opt_dW) : nullptr, p);
    lr_hessian_t hessian(mpi, ft, k_weight, spin_factor, 3, has_sigma, ns, nk, nb, /*plain_only=*/false,
                         /*has_dynamic_source=*/true);
    const auto S_norm_sk = norm_S_sk(sS_skij);
    for (long a = 0; a < 3; ++a) {
      // 7.1 The source is V^E_a, ΔH0 = 0
      p.DeltaH_wskij.emplace(sVE_xwskij.local()(a, all, all, all, all, all));
      auto [it, dmu] = driver.lr_solve_one(sDeltaG_tskij, sDeltaDm_skij, sDeltaF_skij, pDeltaSigma,
                                           /*DeltaH0_skij=*/std::nullopt, thc, p);
      res.niter(5 + a) = it;
      // 7.2 trace_GE_rel = |ΔN_a| / (spin Σ_{s,k} w_k ‖S_k‖ ‖ΔDm_a,k‖), the Cauchy-Schwarz scale of ΔN = Tr[S ΔDm]
      res.trace_GE_rel(a) = trace_GE_relative(driver.dyson().compute_lr_Nelec(sDeltaDm_skij), sDeltaDm_skij, S_norm_sk,
                                          k_weight, spin_factor);
      // 7.3 Store the mode for the hessian
      hessian.store_mode(a, sDeltaDm_skij, sDeltaF_skij, pDeltaSigma, &sDeltaG_tskij);
      mpi->comm.barrier();
    }
    // 7.4 H_plain, H_sym (term 1) from the stored modes; ΔH0 = 0 for every mode (no ΔH0 stack)
    res.hess = hessian.evaluate(driver.dyson(), sDeltaDm_skij, sDeltaF_skij, pDeltaSigma, &sDeltaG_tskij,
                            /*sDeltaH0_skij_mskij=*/nullptr, /*fix_density=*/false, &sVE_xwskij);
    hessian.print_timers();
    mpi->comm.barrier();
  }

  thc.release_dZ_cache();
  if (recompute_W && !reuse_W) W_cache.reset();  // drop only the W this call built; an earlier reuse_W call's W stays

  // 8. χ̄⁽²⁾ = ½(T + Tᵀ), T = term 1 + term 2 + term 3; ε∞ = δ - 4π Re χ̄⁽²⁾
  nda::array<ComplexType, 2> T(3, 3);
  auto& eps_inf = res.eps_inf;
  auto& chi2 = res.chi2;
  auto& chi2_GE_sym = res.chi2_GE_sym;
  auto& chi2_GE_plain = res.chi2_GE_plain;
  for (long a = 0; a < 3; ++a)
    for (long b = 0; b < 3; ++b) {
      chi2_GE_sym(a, b) = res.hess.hessian_sym(b, a) / volume;      // <G^E_a, V^E_b> = H[b, a] / Ω
      chi2_GE_plain(a, b) = res.hess.hessian_plain(b, a) / volume;
      T(a, b) = chi2_GE_sym(a, b) + res.trace_terms.chi2_vGpvGp(a, b) + res.trace_terms.chi2_Sigma2(a, b);
    }
  for (long a = 0; a < 3; ++a)
    for (long b = 0; b < 3; ++b) {
      chi2(a, b) = 0.5 * (T(a, b) + T(b, a));
      eps_inf(a, b) = (a == b ? 1.0 : 0.0) - 4.0 * M_PI * chi2(a, b).real();
      res.eps_inf_imag_max = std::max(res.eps_inf_imag_max, 4.0 * M_PI * std::abs(chi2(a, b).imag()));
    }

  // 9. Diagnostics (note sec:diagnostics): one root-rank warning per failed check
  // |κ̄| 4π / |ε∞_xx - 1|; infinite for a vanishing ε∞_xx - 1 unless κ̄ = 0 too
  const double eps_xx_minus_1 = std::abs(eps_inf(0, 0) - 1.0);
  const double kappa_rel = eps_xx_minus_1 > 0.0 ? std::abs(res.kappa_bar) * 4.0 * M_PI / eps_xx_minus_1
                                                : (res.kappa_bar != 0.0 ? std::numeric_limits<double>::infinity() : 0.0);
  if (mpi->comm.root()) {  // every rank holds the same values; one warning is enough
    if (res.trace_G_half_beta > efield_diagnostic_tol)
      app_warning("run_lr_efield_calc: max|tr G(β/2)| = {:.2e} > {:.0e}, the ground state is not gapped at its μ "
                  "(or β is too small); the thermal terms of ε∞ are not included.", res.trace_G_half_beta,
                  efield_diagnostic_tol);
    if (kappa_rel > efield_diagnostic_tol)
      app_warning("run_lr_efield_calc: |κ̄| 4π/|ε∞ - 1| = {:.2e} > {:.0e}, the compressibility ∂N/∂μ is not negligible.",
                  kappa_rel, efield_diagnostic_tol);
    const double trace_GE_rel_max = std::max({res.trace_GE_rel(0), res.trace_GE_rel(1), res.trace_GE_rel(2)});
    if (trace_GE_rel_max > efield_diagnostic_tol)
      app_warning("run_lr_efield_calc: trace_GE_rel = {:.2e} > {:.0e}, G^E carries a charge response.",
                  trace_GE_rel_max, efield_diagnostic_tol);
    if (res.hess.herm_M > efield_diagnostic_tol) {
      if (kernel_a.sigma_G_dW)
        app_warning("run_lr_efield_calc: herm_M = {:.2e} > {:.0e}, the q = 0 ΔW part of the GW kernel is not self-adjoint "
                    "on this grid.", res.hess.herm_M, efield_diagnostic_tol);
      else
        app_warning("run_lr_efield_calc: herm_M = {:.2e} > {:.0e}, the kernel is not self-adjoint on this grid.",
                    res.hess.herm_M, efield_diagnostic_tol);
    }
  }
  app_log(1, "\n  ε∞ (xx, yy, zz) = {:.6f}  {:.6f}  {:.6f}", eps_inf(0, 0), eps_inf(1, 1), eps_inf(2, 2));

  // 10. The root writes linear_response_efield/
  res.precond_method = has_sigma && use_precond ? "HSEX" : "none";
  if (mpi->comm.root()) write_efield_results(output + ".mbpt.h5", res);
  mpi->comm.barrier();
}

// run_lr_efield_calc instantiation (THC only, as run_lr_calc)
template void run_lr_efield_calc(mb_eri_t<thc_reader_t, thc_reader_t, thc_reader_t, thc_reader_t>&, ptree const&);

} // namespace methods
