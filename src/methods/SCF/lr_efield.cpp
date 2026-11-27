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

#include "itertools/itertools.hpp"
#include "nda/blas.hpp"
#include "utilities/check.hpp"
#include "numerics/imag_axes_ft/iaft_utils.hpp"
#include "methods/SCF/lr_efield.hpp"

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

void lr_total_one_body_w(utils::mpi_context_t<mpi3::communicator>& mpi, imag_axes_ft::IAFT const& FT,
                         Array_view_4D_t const* DeltaH0_skij,
                         sArray_t<Array_view_4D_t> const& sDeltaF_skij,
                         sArray_t<Array_view_5D_t> const* sDeltaSigma_tskij,
                         sArray_t<Array_view_5D_t>& sX_wskij) {
  decltype(nda::range::all) all;
  auto [nw, ns, nk, nb, nb2] = sX_wskij.shape();
  if (sDeltaSigma_tskij) {
    // 1.1 Make ΔΣ(τ) written by other ranks visible, and X(iω) free to be overwritten (earlier readers are done)
    utils::check(sDeltaSigma_tskij->shape() == std::array<long, 5>{FT.nt_f(), ns, nk, nb, nb2} and
                     sX_wskij.shape() == std::array<long, 5>{FT.nw_f(), ns, nk, nb, nb2},
                 "lr_total_one_body_w: ΔΣ(tau) must be (nt_f, ns, nk, nb, nb) and X(iw) (nw_f, ns, nk, nb, nb) of the same IAFT.");
    sDeltaSigma_tskij->node_comm()->barrier();
    sX_wskij.node_sync();
    // 1.2 X(iω)_sk = FT.tau_to_w ΔΣ(τ)_sk, for the (s,k) strided over this node's ranks
    nda::array<ComplexType, 3> X_t(FT.nt_f(), nb, nb2), X_w(FT.nw_f(), nb, nb2);
    for (long sk = mpi.node_comm.rank(); sk < ns * nk; sk += mpi.node_comm.size()) {
      long is = sk / nk, ik = sk % nk;
      X_t = sDeltaSigma_tskij->local()(all, is, ik, all, all);
      FT.tau_to_w(X_t, X_w, imag_axes_ft::fermion);
      sX_wskij.local()(all, is, ik, all, all) = X_w;
    }
    sX_wskij.node_sync();
  } else {
    // 1.1 Σ-free kernel: X(iω) = 0
    sX_wskij.set_zero();
  }
  // 2. X(iω) += ΔH0 + ΔF at every iω, for the (s,k) strided over this node's ranks
  nda::matrix<ComplexType> one_body(nb, nb2);
  for (long sk = mpi.node_comm.rank(); sk < ns * nk; sk += mpi.node_comm.size()) {
    long is = sk / nk, ik = sk % nk;
    one_body = sDeltaF_skij.local()(is, ik, all, all);
    if (DeltaH0_skij) one_body += (*DeltaH0_skij)(is, ik, all, all);
    for (long n = 0; n < nw; ++n) sX_wskij.local()(n, is, ik, all, all) += one_body;
  }
  // 3. Publish X(iω) to the node
  sX_wskij.node_sync();
}

void build_field_sources(utils::mpi_context_t<mpi3::communicator>& mpi,
                         sArray_t<Array_view_5D_t> const& sG_wskij, sArray_t<Array_view_5D_t> const& sB_wskij,
                         std::array<sArray_t<Array_view_5D_t> const*, 3> sv_wskij,
                         sArray_t<Array_view_6D_t>& sVE_xwskij) {
  decltype(nda::range::all) all;
  auto [nw, ns, nk, nb, nb2] = sG_wskij.shape();
  utils::check(sVE_xwskij.shape() == std::array<long, 6>{3, nw, ns, nk, nb, nb2},
               "build_field_sources: V^E must have the shape (3, nw_f, ns, nk, nb, nb) of G(iw).");
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

void build_sigma2_source(utils::mpi_context_t<mpi3::communicator>& mpi,
                         sArray_t<Array_view_5D_t> const& sG_wskij, sArray_t<Array_view_5D_t> const& sB_wskij,
                         sArray_t<Array_view_5D_t>& sDeltaH_wskij) {
  decltype(nda::range::all) all;
  auto [nw, ns, nk, nb, nb2] = sG_wskij.shape();
  utils::check(sB_wskij.shape() == sG_wskij.shape() and sDeltaH_wskij.shape() == sG_wskij.shape(),
               "build_sigma2_source: B(iw) and the source 2BGB must have the shape (nw_f, ns, nk, nb, nb) of G(iw).");
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

} // namespace methods
