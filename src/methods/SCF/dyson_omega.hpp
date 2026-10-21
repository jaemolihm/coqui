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


#ifndef COQUI_DYSON_OMEGA_HPP
#define COQUI_DYSON_OMEGA_HPP

#include "configuration.hpp"
#include "nda/nda.hpp"
#include "nda/linalg.hpp"
#include "numerics/imag_axes_ft/IAFT.hpp"
#include "numerics/shared_array/nda.hpp"
#include "utilities/check.hpp"
#include "utilities/mpi_context.h"

/**
 * The Dyson equation of simple_dyson::solve_dyson kept in iω: the pointwise Green's function of a ground state,
 * which the solver itself stores only as T·G(iω) in τ (T = w_to_tau).
 *
 * Not a call into simple_dyson: its G(iω) lives only in a distributed array inside solve_dyson and is dropped
 * after the transform to τ.
 */
namespace methods {

  /**
   * G(iω_n) = [(iω_n + μ) S - H0 - F - (B Σ)(iω_n)]⁻¹ for one (s, k) block and every fermionic n, as
   * simple_dyson::solve_dyson (B = tau_to_w).
   * Sigma_tij: (nt_f, nb, nb), G_wij: (nw_f, nb, nb).
   */
  inline void dyson_G_omega_block(const imag_axes_ft::IAFT& ft, double mu,
                                  const nda::MemoryArrayOfRank<2> auto& H0, const nda::MemoryArrayOfRank<2> auto& S,
                                  const nda::MemoryArrayOfRank<2> auto& F,
                                  const nda::MemoryArrayOfRank<3> auto& Sigma_tij,
                                  nda::MemoryArrayOfRank<3> auto&& G_wij) {
    long nw = ft.nw_f();
    long nb = H0.shape(0);
    utils::check(Sigma_tij.shape(0) == ft.nt_f() and G_wij.shape(0) == nw and Sigma_tij.shape(1) == nb and
                     Sigma_tij.shape(2) == nb and G_wij.shape(1) == nb and G_wij.shape(2) == nb,
                 "dyson_G_omega_block: shape mismatch (Sigma ({}, {}, {}), G ({}, {}, {}), nb = {}).",
                 Sigma_tij.shape(0), Sigma_tij.shape(1), Sigma_tij.shape(2), G_wij.shape(0), G_wij.shape(1),
                 G_wij.shape(2), nb);
    nda::array<ComplexType, 3> Sigma_t(Sigma_tij);
    nda::array<ComplexType, 3> Sigma_w(nw, nb, nb);
    ft.tau_to_w(Sigma_t, Sigma_w, imag_axes_ft::fermion);
    nda::matrix<ComplexType> X(nb, nb);
    for (long n = 0; n < nw; ++n) {
      ComplexType omega_mu = ft.omega(ft.wn_mesh_f()(n)) + mu;
      X = omega_mu * S - H0 - F - Sigma_w(n, nda::ellipsis{});
      nda::matrix<ComplexType> Xinv = nda::inverse(X);
      G_wij(n, nda::ellipsis{}) = Xinv;
    }
  }

  /**
   * G(iω) of the whole IBZ in node-shared memory, (nw_f, ns, nk_ibz, nb, nb), from dyson_G_omega_block.
   * The (s, k) blocks are dealt round-robin over mpi.comm and combined with shared_array::all_reduce(), as
   * lr_precompute_G_omega. Collective on mpi.comm.
   */
  template<math::shm::SharedArray sArr4_t, math::shm::SharedArray sArr5_t>
  auto pointwise_G_omega(utils::mpi_context_t<mpi3::communicator>& mpi, double mu, const sArr4_t& sH0_skij,
                         const sArr4_t& sS_skij, const sArr4_t& sF_skij, const sArr5_t& sSigma_tskij,
                         const imag_axes_ft::IAFT& ft) {
    using Array_view_5D_t = nda::array_view<ComplexType, 5>;
    auto [nt, ns, nk, nb, nb2] = sSigma_tskij.shape();
    long nw = ft.nw_f();
    utils::check(nt == ft.nt_f(), "pointwise_G_omega: Sigma has {} τ points, expected {}.", nt, ft.nt_f());
    auto sG_wskij = math::shm::make_shared_array<Array_view_5D_t>(mpi, {nw, ns, nk, nb, nb});
    sG_wskij.set_zero(); // ends with fence + node_sync

    auto H0 = sH0_skij.local();
    auto S = sS_skij.local();
    auto F = sF_skij.local();
    auto Sigma = sSigma_tskij.local();
    auto G = sG_wskij.local();
    nda::array<ComplexType, 3> G_w(nw, nb, nb);
    for (long i = mpi.comm.rank(); i < ns * nk; i += mpi.comm.size()) {
      long is = i / nk;
      long ik = i % nk;
      dyson_G_omega_block(ft, mu, H0(is, ik, nda::ellipsis{}), S(is, ik, nda::ellipsis{}), F(is, ik, nda::ellipsis{}),
                          Sigma(nda::range::all, is, ik, nda::range::all, nda::range::all), G_w);
      G(nda::range::all, is, ik, nda::range::all, nda::range::all) = G_w;
    }
    sG_wskij.win().fence();
    sG_wskij.all_reduce();
    return sG_wskij;
  }

} // namespace methods

#endif // COQUI_DYSON_OMEGA_HPP
