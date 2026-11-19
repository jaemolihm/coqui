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
                         sArray_t<Array_view_4D_t> const* sDeltaH0_skij,
                         sArray_t<Array_view_4D_t> const& sDeltaF_skij,
                         sArray_t<Array_view_5D_t> const* sDeltaSigma_tskij,
                         sArray_t<Array_view_5D_t>& sX_wskij) {
  decltype(nda::range::all) all;
  auto [nw, ns, nk, nb, nb2] = sX_wskij.shape();
  // 1. X(iω) = τ->ω ΔΣ(τ), or zero for a Σ-free kernel
  if (sDeltaSigma_tskij) shared_tau_to_w(FT, *sDeltaSigma_tskij, sX_wskij);
  else sX_wskij.set_zero();
  // 2. X(iω) += ΔH0 + ΔF at every iω, for the (s,k) strided over this node's ranks
  nda::matrix<ComplexType> one_body(nb, nb2);
  for (long sk = mpi.node_comm.rank(); sk < ns * nk; sk += mpi.node_comm.size()) {
    long is = sk / nk, ik = sk % nk;
    one_body = sDeltaF_skij.local()(is, ik, all, all);
    if (sDeltaH0_skij) one_body = one_body + matrix_t(sDeltaH0_skij->local()(is, ik, all, all));
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
  for (long b = mpi.node_comm.rank(); b < nw * ns * nk; b += mpi.node_comm.size()) {
    long n = b / (ns * nk), is = (b / nk) % ns, ik = b % nk;
    G = sG_wskij.local()(n, is, ik, all, all);
    B = sB_wskij.local()(n, is, ik, all, all);
    // 1. GB = G·B, BG = B·G
    GB = G * B;
    BG = B * G;
    // 2. V^E_α = -(i/2)(v_α·GB - BG·v_α)
    for (long x = 0; x < 3; ++x) {
      v = sv_wskij[x]->local()(n, is, ik, all, all);
      VE = minus_i_half * (v * GB - BG * v);
      sVE_xwskij.local()(x, n, is, ik, all, all) = VE;
    }
  }
  sVE_xwskij.node_sync();
}

void build_mu2_source(utils::mpi_context_t<mpi3::communicator>& mpi,
                      sArray_t<Array_view_5D_t> const& sG_wskij, sArray_t<Array_view_5D_t> const& sB_wskij,
                      sArray_t<Array_view_5D_t>& sDeltaH_wskij) {
  decltype(nda::range::all) all;
  auto [nw, ns, nk, nb, nb2] = sG_wskij.shape();
  sDeltaH_wskij.node_sync();
  matrix_t G(nb, nb), B(nb, nb), BG(nb, nb), BGB(nb, nb);
  for (long b = mpi.node_comm.rank(); b < nw * ns * nk; b += mpi.node_comm.size()) {
    long n = b / (ns * nk), is = (b / nk) % ns, ik = b % nk;
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
  // c(n, 3a+b): the (iω_n) trace of each term, summed over this rank's (s,k)
  nda::array<ComplexType, 3> c2(nw, 3, 3), c3(nw, 3, 3);
  c2() = 0.0;
  c3() = 0.0;
  matrix_t G(nb, nb), B(nb, nb), S2(nb, nb), GB(nb, nb), Gp(nb, nb), Y(nb, nb), tmp(nb, nb);
  std::array<matrix_t, 3> v, P, A, C;
  for (long a = 0; a < 3; ++a) {
    v[a] = matrix_t(nb, nb);
    P[a] = matrix_t(nb, nb);
    A[a] = matrix_t(nb, nb);
    C[a] = matrix_t(nb, nb);
  }
  auto [b0, b1] = itertools::chunk_range(0, nw * ns * nk, long(mpi.comm.size()), long(mpi.comm.rank()));
  for (long b = b0; b < b1; ++b) {
    long n = b / (ns * nk), is = (b / nk) % ns, ik = b % nk;
    const double w = spin_factor * k_weight(ik);
    G = sG_wskij.local()(n, is, ik, all, all);
    B = sB_wskij.local()(n, is, ik, all, all);
    S2 = sSigma2_wskij.local()(n, is, ik, all, all);
    // 1. G' = -G B G
    GB = G * B;
    Gp = -1.0 * (GB * G);
    // 2. term 2: P_α = v_α G',  c2_αβ += w tr[P_α P_β] = w tr[v_α G' v_β G']
    // 3. term 3: A_α = (G Σ'' G) v_α, C_β = G v_β,  c3_αβ += w tr[A_α C_β] = w tr[Σ'' G v_α G v_β G]
    tmp = G * S2;
    Y = tmp * G;
    for (long a = 0; a < 3; ++a) {
      v[a] = sv_wskij[a]->local()(n, is, ik, all, all);
      P[a] = v[a] * Gp;
      A[a] = Y * v[a];
      C[a] = G * v[a];
    }
    for (long a = 0; a < 3; ++a)
      for (long c = 0; c < 3; ++c) {
        c2(n, a, c) += w * trace_of_product(P[a], P[c]);
        c3(n, a, c) += w * trace_of_product(A[a], C[c]);
      }
  }
  // 4. Σ_{s,k} over the global communicator
  mpi.comm.all_reduce_in_place_n(c2.data(), c2.size(), std::plus<>{});
  mpi.comm.all_reduce_in_place_n(c3.data(), c3.size(), std::plus<>{});
  // 5. term = (1/(4Ω)) (1/β) Σ_n c(iω_n)
  efield_trace_terms terms{nda::array<ComplexType, 2>(3, 3), nda::array<ComplexType, 2>(3, 3)};
  nda::array<ComplexType, 2> c_w(nw, 1);
  for (long a = 0; a < 3; ++a)
    for (long c = 0; c < 3; ++c) {
      c_w(all, 0) = c2(all, a, c);
      terms.vGpvGp(a, c) = imag_axes_ft::fermion_matsubara_sum(FT, c_w) / (4.0 * volume);
      c_w(all, 0) = c3(all, a, c);
      terms.Sigma2_GvGvG(a, c) = imag_axes_ft::fermion_matsubara_sum(FT, c_w) / (4.0 * volume);
    }
  return terms;
}

ComplexType matsubara_pairing(utils::mpi_context_t<mpi3::communicator>& mpi, imag_axes_ft::IAFT const& FT,
                              sArray_t<Array_view_5D_t> const& sA_wskij, sArray_t<Array_view_5D_t> const& sB_wskij,
                              nda::array<double, 1> const& k_weight, double spin_factor, double volume) {
  decltype(nda::range::all) all;
  auto [nw, ns, nk, nb, nb2] = sA_wskij.shape();
  utils::check(sB_wskij.shape() == sA_wskij.shape(), "matsubara_pairing: the two arrays must have one shape.");
  nda::array<ComplexType, 2> c_w(nw, 1);
  c_w() = 0.0;
  matrix_t A(nb, nb), B(nb, nb);
  auto [b0, b1] = itertools::chunk_range(0, nw * ns * nk, long(mpi.comm.size()), long(mpi.comm.rank()));
  for (long b = b0; b < b1; ++b) {
    long n = b / (ns * nk), is = (b / nk) % ns, ik = b % nk;
    A = sA_wskij.local()(n, is, ik, all, all);
    B = sB_wskij.local()(n, is, ik, all, all);
    // 1. c(iω_n) += spin w_k tr[A B]
    c_w(n, 0) += spin_factor * k_weight(ik) * trace_of_product(A, B);
  }
  // 2. Σ_{s,k} over the global communicator, then (1/β) Σ_n c / Ω
  mpi.comm.all_reduce_in_place_n(c_w.data(), c_w.size(), std::plus<>{});
  return imag_axes_ft::fermion_matsubara_sum(FT, c_w) / volume;
}

double thermal_occupation(imag_axes_ft::IAFT const& FT, sArray_t<Array_view_5D_t> const& sG_tskij) {
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

} // namespace methods
