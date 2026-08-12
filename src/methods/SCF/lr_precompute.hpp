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


#ifndef COQUI_LR_PRECOMPUTE_HPP
#define COQUI_LR_PRECOMPUTE_HPP

#include "IO/app_loggers.h"
#include "nda/blas.hpp"
#include "numerics/distributed_array/nda.hpp"
#include "numerics/imag_axes_ft/IAFT.hpp"
#include "numerics/shared_array/nda.hpp"
#include "utilities/kpoint_utils.hpp"
#include "utilities/lr_utils.hpp"
#include "utilities/mpi_context.h"
#include "methods/ERI/detail/concepts.hpp"
#include "methods/HF/thc_solver_comm.hpp"
#include "mean_field/MF.hpp"

namespace methods {

/**
 * @brief Precompute G(iω) in shared memory for LR Dyson equation.
 *
 * Transforms sG_tskij from tau to frequency space and stores the result in
 * a shared memory array. Node root performs the FT; all ranks on the same
 * node share the result via shared memory.
 *
 * @param mpi        - [INPUT] MPI context (comm, node_comm, internode_comm)
 * @param sG_tskij   - [INPUT] Unperturbed Green's function in tau (nt, ns, nk, nb, nb)
 * @param FT         - [INPUT] IAFT object for Fourier transform
 * @return Shared memory G(iω) array (nw, ns, nk, nb, nb)
 */
template<math::shm::SharedArray sArr_t>
auto lr_precompute_G_omega(utils::mpi_context_t<mpi3::communicator>& mpi,
                           const sArr_t& sG_tskij,
                           const imag_axes_ft::IAFT& FT) {
  decltype(nda::range::all) all;
  auto [nts, ns, nkpts_ibz, nbnd, nbnd2] = sG_tskij.shape();
  auto nw = FT.nw_f();

  app_log(2, "lr_precompute_G_omega: FT G(tau) -> G(iw) [shared memory]");
  app_log(2, "  shape: ({}, {}, {}, {}, {}) -> ({}, {}, {}, {}, {})",
          nts, ns, nkpts_ibz, nbnd, nbnd, nw, ns, nkpts_ibz, nbnd, nbnd);

  using Array_view_5D_t = nda::array_view<ComplexType, 5>;
  auto sG_wskij = math::shm::make_shared_array<Array_view_5D_t>(
      mpi, {nw, ns, nkpts_ibz, nbnd, nbnd});

  // Fan out (s, k) work across mpi.comm; each rank writes its slab and
  // all_reduce makes the result visible to every node.
  sG_wskij.set_zero();  // ends with fence + node_sync

  int rank = mpi.comm.rank();
  int size = mpi.comm.size();
  nda::array<ComplexType, 3> G_t_buf(nts, nbnd, nbnd);
  nda::array<ComplexType, 3> G_w_buf(nw, nbnd, nbnd);
  auto G_t_loc = sG_tskij.local();
  auto G_w_loc = sG_wskij.local();
  for (int i = rank; i < ns * nkpts_ibz; i += size) {
    int is = i / nkpts_ibz;
    int ik = i % nkpts_ibz;
    G_t_buf = G_t_loc(all, is, ik, all, all);
    FT.tau_to_w(G_t_buf, G_w_buf, imag_axes_ft::fermion);
    G_w_loc(all, is, ik, all, all) = G_w_buf;
  }
  sG_wskij.win().fence();
  sG_wskij.all_reduce();

  return sG_wskij;
}

/**
 * @brief In-place q→R Fourier transform of W, (t,q,P,Q) → (t,R,P,Q).
 *
 * Input/output: dW_tqPQ with (t,q,P,Q) layout, pgrid (tpools,1,np_P,np_Q).
 * The transform is done in place and the same array is returned, so the caller's
 * (t,q) copy is consumed. For nkpts == 1 (Gamma-only), q-space == R-space and
 * there is nothing to do.
 *
 * @param dW_tqPQ  - [INPUT/OUTPUT] W_c in THC basis, shape (nt_half, nkpts, NP, NQ).
 * @param thc      - [INPUT] THC-ERI handler (provides MF for Qpts, lattv, kp_grid)
 * @return dW_tRPQ with (t,R,P,Q) layout
 */
template<nda::MemoryArray Array_4D_t, typename communicator_t>
auto lr_precompute_W_tRPQ(memory::darray_t<Array_4D_t, communicator_t>& dW_tqPQ_in,
                          THC_ERI auto& thc) {
  auto MF = thc.MF();
  auto mpi = thc.mpi();

  auto dW_tqPQ = std::move(dW_tqPQ_in);

  auto [nt_half, nkpts, NP, NQ] = dW_tqPQ.global_shape();
  auto [nt_loc, nk_loc, NP_loc, NQ_loc] = dW_tqPQ.local_shape();

  if (nkpts == 1) {
    app_log(2, "lr_precompute_W_tRPQ: nkpts=1 (Gamma-only), no transform needed");
    return dW_tqPQ;
  }

  app_log(2, "lr_precompute_W_tRPQ: in-place q->R transform on W");
  app_log(2, "  global: ({}, {}, {}, {}), local: ({}, {}, {}, {})",
          nt_half, nkpts, NP, NQ, nt_loc, nk_loc, NP_loc, NQ_loc);

  // Compute f_qR Fourier coefficients
  using sArray_2D_t = math::shm::shared_array<nda::array_view<ComplexType, 2>>;
  sArray_2D_t sf_qR(*mpi, {nkpts, nkpts});
  utils::k_to_R_coefficients(mpi->comm, nda::range(nkpts),
                             MF->Qpts(), MF->lattv(), MF->kp_grid(), sf_qR);
  auto f_qR = sf_qR.local();

  // Work buffers
  nda::array<ComplexType, 3> W_buf_qPQ(nkpts, NP_loc, NQ_loc);
  nda::matrix<ComplexType> W_buf_RPQ(nkpts, NP_loc * NQ_loc);

  // Transform each local tau slice in-place
  for (long it_local = 0; it_local < nt_loc; ++it_local) {
    // Copy W(it_local, q, :, :) to contiguous buffer
    for (long iq = 0; iq < nkpts; ++iq)
      W_buf_qPQ(iq, nda::ellipsis{}) = dW_tqPQ.local()(it_local, iq, nda::ellipsis{});

    // gemm: W_RPQ = f_qR @ W_qPQ
    auto W_2D_view = nda::reshape(W_buf_qPQ, std::array<long, 2>{nkpts, NP_loc * NQ_loc});
    nda::blas::gemm(f_qR, W_2D_view, W_buf_RPQ);

    // Copy back to dW from gemm output (now stores W(t,R,P,Q))
    auto W_RPQ_3D = nda::reshape(W_buf_RPQ, std::array<long, 3>{nkpts, NP_loc, NQ_loc});
    for (long iR = 0; iR < nkpts; ++iR)
      dW_tqPQ.local()(it_local, iR, nda::ellipsis{}) = W_RPQ_3D(iR, nda::ellipsis{});
  }

  mpi->comm.barrier();
  app_log(2, "lr_precompute_W_tRPQ: done");
  return dW_tqPQ;
}

/**
 * @brief Precompute the unperturbed G in the THC aux basis, in R-space, for
 *        the (τ, β−τ) pair at every local first-half τ point.
 *
 * G is constant during the LR SCF loop, and evaluate_lr_Pi needs it. This caches
 *   G^R_{PQ}(τ_it)        → dG_tsRPQ  (it, s, R, P, Q)
 *   G^R_{PQ}(β − τ_it)    → dG_mtau_tsRPQ (it, s, R, P, Q)
 * for it in the local τ range of the LR τ-dist (lr_W_q_local_dist), which is
 * the distribution shared by Π, dW_tRPQ and ΔW in the gw_full pipeline.
 * The pair covers all G accesses: Π term 1 uses conj(G^R(β−τ)), Π term 2 uses
 * G^R(τ), Σ term 2 uses G^R(τ) (forward pass) and G^R(β−τ) (backward pass).
 * (β−τ_it lives at grid index nt−1−it: the IAFT τ-grid is symmetric about β/2,
 * so the index reflection x→−x is exactly τ→β−τ.)
 *
 * For nkpts == 1 no k→R FT is applied (k-space == R-space), matching the
 * consumers' Γ-only branches.
 *
 * Memory: 2 · nt_loc · ns · nkpts · NP_loc · NQ_loc complex per rank
 * (≈ 2× the local footprint of ΔΠ).
 *
 * @param G_tskij - [INPUT] Unperturbed G(τ), (nt_f, ns, nkpts_ibz, nbnd, nbnd)
 * @param thc     - [INPUT] THC-ERI handler
 * @return pair (dG_tsRPQ, dG_mtau_tsRPQ), global shape (nt_half, ns, nkpts, NP, NP),
 *         pgrid {tpools, 1, 1, np_P, np_Q}
 */
template<THC_ERI THC_t>
auto lr_precompute_G_R_pair(const nda::array_view<ComplexType, 5>& G_tskij,
                            THC_t& thc) {
  using local_Array_4D_t = nda::array<ComplexType, 4>;
  using local_Array_5D_t = nda::array<ComplexType, 5>;
  using math::nda::make_distributed_array;
  decltype(nda::range::all) all;

  auto mpi = thc.mpi();
  auto MF = thc.MF();
  long nt_f = G_tskij.shape(0);
  long ns = G_tskij.shape(1);
  long nkpts = MF->nkpts();
  long Np = thc.Np();
  long nt_half = (nt_f % 2 == 0) ? nt_f / 2 : nt_f / 2 + 1;

  auto [tau_pgrid, tau_bsize] = utils::lr_W_q_local_dist(mpi->comm.size(), nt_half, Np);
  auto [tpools, q_dummy, np_P, np_Q] = tau_pgrid;

  app_log(2, "lr_precompute_G_R_pair: caching G^R(τ) / G^R(β−τ) in aux basis");
  app_log(2, "  global: ({}, {}, {}, {}, {}) x2, pgrid: ({}, 1, 1, {}, {})",
          nt_half, ns, nkpts, Np, Np, tpools, np_P, np_Q);

  auto dG_tsRPQ = make_distributed_array<local_Array_5D_t>(
      mpi->comm, {tpools, 1, 1, np_P, np_Q}, {nt_half, ns, nkpts, Np, Np},
      {1, 1, 1, tau_bsize[2], tau_bsize[3]});
  auto dG_mtau_tsRPQ = make_distributed_array<local_Array_5D_t>(
      mpi->comm, {tpools, 1, 1, np_P, np_Q}, {nt_half, ns, nkpts, Np, Np},
      {1, 1, 1, tau_bsize[2], tau_bsize[3]});

  long t_origin = dG_tsRPQ.origin()[0];
  auto [nt_loc, ns_loc, nk_loc, NP_loc, NQ_loc] = dG_tsRPQ.local_shape();

  // Work buffer in the same PQ tiling, built on the t-intra subgroup
  mpi3::communicator t_intra_comm = mpi->comm.split(t_origin, mpi->comm.rank());
  utils::check(t_intra_comm.size() == np_P * np_Q,
               "lr_precompute_G_R_pair: t_intra_comm.size() = {} != np_P*np_Q",
               t_intra_comm.size());
  auto dG_skPQ = make_distributed_array<local_Array_4D_t>(
      t_intra_comm, {1, 1, np_P, np_Q}, {ns, nkpts, Np, Np},
      {1, 1, tau_bsize[2], tau_bsize[3]});

  // f_Rk (same transform as Π's term factors and Σ term 2)
  math::shm::shared_array<nda::array_view<ComplexType, 2>> sf_Rk(*mpi, {nkpts, nkpts});
  if (nkpts != 1)
    utils::k_to_R_coefficients(mpi->comm, nda::range(nkpts), MF->kpts(), MF->lattv(), MF->kp_grid(), sf_Rk);
  auto f_Rk = sf_Rk.local();

  nda::matrix<ComplexType> ft_out(nkpts, NP_loc * NQ_loc);

  for (long it = 0; it < nt_loc; ++it) {
    long it_g = t_origin + it;
    for (int pass = 0; pass < 2; ++pass) {
      long idx = (pass == 0) ? it_g : nt_f - 1 - it_g;
      auto& target = (pass == 0) ? dG_tsRPQ : dG_mtau_tsRPQ;

      solvers::thc_solver_comm::primary_to_aux(0, 0, G_tskij(idx, nda::ellipsis{}),
                                               dG_skPQ, thc, MF->kp_to_ibz(), MF->kp_trev());

      auto G_skPQ_loc = dG_skPQ.local();
      if (nkpts != 1) {
        for (long is = 0; is < ns; ++is) {
          auto G_2D = nda::reshape(G_skPQ_loc(is, nda::ellipsis{}),
                                   std::array<long, 2>{nkpts, NP_loc * NQ_loc});
          nda::blas::gemm(f_Rk, G_2D, ft_out);
          target.local()(it, is, all, all, all) =
              nda::reshape(ft_out, std::array<long, 3>{nkpts, NP_loc, NQ_loc});
        }
      } else {
        target.local()(it, nda::ellipsis{}) = G_skPQ_loc;
      }
    }
  }
  mpi->comm.barrier();
  app_log(2, "lr_precompute_G_R_pair: done");

  return std::make_pair(std::move(dG_tsRPQ), std::move(dG_mtau_tsRPQ));
}

} // namespace methods

#endif // COQUI_LR_PRECOMPUTE_HPP
