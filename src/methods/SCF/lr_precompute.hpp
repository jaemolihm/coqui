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


#ifndef COQUI_LR_PRECOMPUTE_HPP
#define COQUI_LR_PRECOMPUTE_HPP

#include "IO/app_loggers.h"
#include "nda/blas.hpp"
#include "numerics/distributed_array/nda.hpp"
#include "numerics/imag_axes_ft/IAFT.hpp"
#include "numerics/shared_array/nda.hpp"
#include "utilities/kpoint_utils.hpp"
#include "utilities/mpi_context.h"
#include "methods/ERI/detail/concepts.hpp"
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
    FT.tau_to_w(G_t_buf, G_w_buf, imag_axes_ft::fermi);
    G_w_loc(all, is, ik, all, all) = G_w_buf;
  }
  sG_wskij.win().fence();
  sG_wskij.all_reduce();

  return sG_wskij;
}

/**
 * @brief Transpose W from (q,t,P,Q) to (t,R,P,Q) with in-place q→R Fourier transform.
 *
 * Input: dW_qtPQ with (q,t,P,Q) layout, pgrid (1,tpools,np_P,np_Q).
 * Returns: dW_tRPQ with (t,R,P,Q) layout, pgrid (tpools,1,np_P,np_Q).
 *
 * Performs (q,t)→(t,q) transpose via transpose_axes_01, then in-place q→R FT.
 * For nkpts == 1 (Gamma-only), no FT is needed (q-space == R-space).
 *
 * @param dW_qtPQ  - [INPUT] W_c in THC basis, shape (nkpts, nt_half, NP, NQ).
 * @param thc      - [INPUT] THC-ERI handler (provides MF for Qpts, lattv, kp_grid)
 * @return dW_tRPQ with (t,R,P,Q) layout
 */
template<nda::MemoryArray Array_4D_t, typename communicator_t>
auto lr_precompute_W_tRPQ(memory::darray_t<Array_4D_t, communicator_t>& dW_qtPQ,
                          THC_ERI auto& thc) {
  auto MF = thc.MF();
  auto mpi = thc.mpi();

  // Transpose (q,t) → (t,q)
  auto dW_tqPQ = utils::transpose_axes_01(dW_qtPQ, mpi->comm);

  auto [nt_half, nkpts, NP, NQ] = dW_tqPQ.global_shape();
  auto [nt_loc, nk_loc, NP_loc, NQ_loc] = dW_tqPQ.local_shape();

  if (nkpts == 1) {
    app_log(2, "lr_precompute_W_tRPQ: nkpts=1 (Gamma-only), no transform needed");
    return dW_tqPQ;
  }

  app_log(2, "lr_precompute_W_tRPQ: transpose (q,t)->(t,q) + in-place q->R transform on W");
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

} // namespace methods

#endif // COQUI_LR_PRECOMPUTE_HPP
