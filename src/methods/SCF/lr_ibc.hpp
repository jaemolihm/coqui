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


#ifndef COQUI_LR_IBC_HPP
#define COQUI_LR_IBC_HPP

#include "nda/nda.hpp"
#include "nda/blas.hpp"
#include "numerics/shared_array/nda.hpp"
#include "numerics/distributed_array/nda.hpp"

#include "IO/app_loggers.h"
#include "utilities/mpi_context.h"
#include "utilities/check.hpp"
#include "utilities/kpoint_utils.hpp"
#include "mean_field/MF.hpp"
#include "methods/ERI/detail/concepts.hpp"
#include "methods/HF/thc_solver_comm.hpp"

namespace methods {

  /**
   * @brief Incomplete Basis Correction (IBC) data for LR THC transforms.
   *
   * When the THC collocation matrix X is perturbed (δX), the primary↔aux transforms
   * of LR quantities get additional correction terms.
   *
   * The primary→aux correction requires O(N_μ²) memory, so it is computed on-the-fly
   * using DeltaX and the unperturbed quantity (Dm or G).
   *
   * The aux→primary correction only requires O(N_band²) memory, so it is precomputed
   * once and stored here:
   *   DeltaF_ibc(k,m,n)   = δX†·F_PQ·X + X†·F_PQ·δX     (from HF self-energy)
   *   DeltaSigma_ibc(τ,k,m,n) = δX†·Σ_PQ·X + X†·Σ_PQ·δX (from GW self-energy)
   *
   * These are added at the callsite after each aux→primary transform, rather than
   * inside the low-level transform code.
   */
  struct lr_ibc_DeltaX {
    using sArray_5D_t = math::shm::shared_array<nda::array_view<ComplexType, 5>>;

    // Lifetime contract:
    //   `DeltaX` and `DeltaX_minusq` are non-owning views into a
    //   `shared_array<array_view<ComplexType,4>>` allocated by the caller
    //   (currently `run_lr_calc` and `lr_gw_Pi_calc` in MBPT_drivers.cpp).
    //   The owning shared_array MUST outlive this struct. Do NOT store an
    //   `lr_ibc_DeltaX` past the scope of the function that allocated the
    //   underlying shared_array; do NOT copy it across MPI rank boundaries.
    //
    // Core DeltaX arrays (full BZ indexed, shape: ns, nkpts, Np, nb).
    nda::array_view<ComplexType, 4> DeltaX;         // δ^q X
    nda::array_view<ComplexType, 4> DeltaX_minusq;  // δ^{-q} X

    // Perturbation wavevector and k→k+q mapping
    nda::array<double, 1> q_vec;       // crystal coordinates (3,)
    nda::array<int, 1> kpq_map;        // k → k+q mapping, full BZ (nkpts,)

    // Unperturbed quantities for primary→aux on-the-fly correction (pointers, not copies)
    const nda::array<ComplexType, 4>* Dm_ab = nullptr;     // (ns, nk_ibz, nb, nb) — for HF
    const sArray_5D_t* sG_tskij = nullptr;                 // shared array ptr — for GW/Pi

    // Precomputed IBC corrections for aux→primary (added at callsite)
    nda::array<ComplexType, 4> DeltaF_ibc_skij;            // (ns, nk_ibz, nb, nb)

    // Held in a node-local shared-memory window (one copy per node) when
    // present. std::nullopt when the GW-Sigma path is inactive.
    std::optional<sArray_5D_t> sDeltaSigma_ibc_tskij;      // (nt, ns, nk_ibz, nb, nb)
  };


  // =========================================================================
  // IBC precomputation functions (template implementations)
  // =========================================================================

  /**
   * @brief Compute V_HF in THC auxiliary (PQ) basis directly, without band-basis round-trip.
   *
   * The IBC correction needs V_HF_PQ (equilibrium Hartree-Fock potential in aux basis).
   * Computing it as X*(F-H0)*X† introduces large errors because X†X ≠ I.
   * This function computes V_HF_PQ directly by:
   *   1. Forward-transform Dm_ab → Dm_PQ
   *   2. Compute Hartree J_PQ and Exchange K_PQ using the Coulomb kernel
   *   3. Return V_HF_PQ = J_PQ + K_PQ (distributed array)
   *
   * Mirrors the forward half of hf_t::thc_hf_Xqindep (thc_hf.icc), stopping before
   * aux_to_primary. Does NOT include the Madelung finite-size correction (applied
   * separately in band space by lr_hf::LR_HF_K_correction).
   *
   * @param mpi  - [INPUT] MPI context
   * @param MF   - [INPUT] Mean-field handler
   * @param thc  - [INPUT] THC-ERI handler (provides X, Z, MF)
   * @param Dm_skij  - [INPUT] Equilibrium density matrix (ns, nk_ibz, nb, nb)
   * @param compute_hartree  - [INPUT] Include Hartree (J) contribution
   * @param compute_exchange - [INPUT] Include Exchange (K) contribution
   * @return Distributed array V_HF_PQ with pgrid (1, 1, np_P, np_Q),
   *         shape (ns, nk_ibz, NP, NP)
   */
  template<typename mpi_context_t>
  auto lr_precompute_V_HF_PQ(mpi_context_t& mpi,
                              const mf::MF* MF,
                              THC_ERI auto& thc,
                              const nda::array<ComplexType, 4>& Dm_skij,
                              bool compute_hartree,
                              bool compute_exchange) {
    using local_Array_4D_t = nda::array<ComplexType, 4>;
    using math::nda::make_distributed_array;

    long NP = thc.Np();
    long ns = Dm_skij.shape(0);
    long npol = MF->npol();
    long nkpts = MF->nkpts();
    long nkpts_ibz = MF->nkpts_ibz();

    int np = mpi.comm.size();
    int np_P = utils::find_proc_grid_min_diff(np, 1, 1);
    int np_Q = np / np_P;
    nda::array<long, 1> R_grid = MF->kp_grid();

    app_log(2, "lr_precompute_V_HF_PQ: computing V_HF directly in PQ space");
    app_log(2, "  Hartree={}, Exchange={}, NP={}, nkpts={}", compute_hartree, compute_exchange, NP, nkpts);

    // Allocate distributed arrays
    auto dDm_skPQ = make_distributed_array<local_Array_4D_t>(
        mpi.comm, {1, 1, np_P, np_Q}, {ns, nkpts, NP, NP});
    auto dF_skPQ = make_distributed_array<local_Array_4D_t>(
        mpi.comm, {1, 1, np_P, np_Q}, {ns, nkpts_ibz, NP, NP});
    dF_skPQ.local() = ComplexType(0.0);

    auto NP_loc = dDm_skPQ.local_shape()[2];
    auto NQ_loc = dDm_skPQ.local_shape()[3];
    auto P_origin = dDm_skPQ.origin()[2];

    // Coulomb kernel
    auto dU_qPQ = thc.dZ({1, np_P, np_Q});
    auto dU_qPQ_loc = dU_qPQ.local();

    // U(q=0) for Hartree
    nda::array<ComplexType, 2> Uq0_PQ(NP_loc, NQ_loc);
    Uq0_PQ() = dU_qPQ_loc(0, nda::ellipsis{});

    // Diagonal indices for Hartree
    auto Q_origin = dDm_skPQ.origin()[3];
    std::vector<std::pair<long, long>> diag_idx;
    for (long iP = 0; iP < NP_loc; ++iP) {
      long P = iP + P_origin;
      for (long iQ = 0; iQ < NQ_loc; ++iQ) {
        long Q = iQ + Q_origin;
        if (P == Q) diag_idx.push_back({iP, iQ});
      }
    }

    // FT shared array for k<->R transforms
    math::shm::shared_array<nda::array_view<ComplexType, 2>> sf_Rk(mpi, {nkpts, nkpts});
    nda::matrix<ComplexType> buffer;

    // Accumulate Dm_QQ for Hartree
    nda::array<ComplexType, 1> Dm_QQ(NP, ComplexType(0.0));

    if (compute_hartree && !compute_exchange) {
      // === HARTREE-ONLY PATH ===
      for (auto ip : nda::range(npol)) {
        solvers::thc_solver_comm::primary_to_aux(ip, ip, Dm_skij, dDm_skPQ, thc,
                                                  MF->kp_to_ibz(), MF->kp_trev());

        // FT k→R
        if (nkpts != 1) {
          auto f_Rk = sf_Rk.local();
          utils::k_to_R_coefficients(mpi.comm, nda::range(nkpts), MF->kpts(), MF->lattv(), R_grid, sf_Rk);
          auto Dm_3D = nda::reshape(dDm_skPQ.local(), std::array<long, 3>{ns, nkpts, NP_loc * NQ_loc});
          if (buffer.shape() != std::array<long, 2>{nkpts, NP_loc * NQ_loc})
            buffer.resize(std::array<long, 2>{nkpts, NP_loc * NQ_loc});
          for (int s = 0; s < ns; ++s) {
            nda::blas::gemm(f_Rk, Dm_3D(s, nda::ellipsis{}), buffer);
            Dm_3D(s, nda::ellipsis{}) = buffer;
          }
        }

        // Extract diagonal of Dm_PQ at R=0
        double factor = (ns == 1 && npol == 1) ? 2.0 : 1.0;
        auto Dm_sRPQ = dDm_skPQ.local();
        for (long is = 0; is < ns; ++is) {
          for (auto idx : diag_idx)
            Dm_QQ(idx.first + P_origin) += factor * Dm_sRPQ(is, 0, idx.first, idx.second);
        }
      } // ip

      // Hartree: J_PP = U_PQ * Dm_QQ
      dDm_skPQ.communicator()->all_reduce_in_place_n(Dm_QQ.data(), Dm_QQ.size(), std::plus<>{});
      nda::array<ComplexType, 1> J_PP(NP, ComplexType(0.0));
      nda::blas::gemv(Uq0_PQ, Dm_QQ(dU_qPQ.local_range(2)), J_PP(dU_qPQ.local_range(1)));
      dDm_skPQ.communicator()->all_reduce_in_place_n(J_PP.data(), J_PP.size(), std::plus<>{});

      // Set F_PQ = diag(J_PP) — same at all k-points
      auto F_skPQ = dF_skPQ.local();
      for (long is = 0; is < ns; ++is) {
        for (long ik = 0; ik < nkpts_ibz; ++ik) {
          for (auto idx : diag_idx) {
            F_skPQ(is, ik, idx.first, idx.second) = J_PP(idx.first + P_origin);
          }
        }
      }

    } else if (compute_exchange) {
      // === EXCHANGE (+ optional HARTREE) PATH ===

      // FT U(q) → U(R)
      if (nkpts != 1) {
        buffer.resize(std::array<long, 2>{nkpts, NP_loc * NQ_loc});
        auto f_Rk = sf_Rk.local();
        utils::k_to_R_coefficients(mpi.comm, nda::range(nkpts), MF->Qpts(), MF->lattv(), R_grid, sf_Rk);
        auto U_2D = nda::reshape(dU_qPQ_loc, std::array<long, 2>{nkpts, NP_loc * NQ_loc});
        nda::blas::gemm(f_Rk, U_2D, buffer);
        U_2D = buffer;
      }
      auto& U_RPQ_loc = dU_qPQ_loc;

      for (auto ip : nda::range(npol)) {
        for (auto iq : nda::range(ip, npol)) {
          // Forward transform
          solvers::thc_solver_comm::primary_to_aux(ip, iq, Dm_skij, dDm_skPQ, thc,
                                                    MF->kp_to_ibz(), MF->kp_trev());

          // FT k→R
          if (nkpts != 1) {
            auto f_Rk = sf_Rk.local();
            utils::k_to_R_coefficients(mpi.comm, nda::range(nkpts), MF->kpts(), MF->lattv(), R_grid, sf_Rk);
            auto Dm_3D = nda::reshape(dDm_skPQ.local(), std::array<long, 3>{ns, nkpts, NP_loc * NQ_loc});
            if (buffer.shape() != std::array<long, 2>{nkpts, NP_loc * NQ_loc})
              buffer.resize(std::array<long, 2>{nkpts, NP_loc * NQ_loc});
            for (int s = 0; s < ns; ++s) {
              nda::blas::gemm(f_Rk, Dm_3D(s, nda::ellipsis{}), buffer);
              Dm_3D(s, nda::ellipsis{}) = buffer;
            }
          }

          // Hartree contribution (diagonal blocks only)
          if (compute_hartree && ip == iq) {
            double factor = (ns == 1 && npol == 1) ? 2.0 : 1.0;
            auto Dm_sRPQ = dDm_skPQ.local();
            for (long is = 0; is < ns; ++is) {
              for (auto idx : diag_idx)
                Dm_QQ(idx.first + P_origin) += factor * Dm_sRPQ(is, 0, idx.first, idx.second);
            }

            if (npol == 1) {
              dDm_skPQ.communicator()->all_reduce_in_place_n(Dm_QQ.data(), Dm_QQ.size(), std::plus<>{});
              nda::array<ComplexType, 1> J_PP(NP, ComplexType(0.0));
              nda::blas::gemv(Uq0_PQ, Dm_QQ(dU_qPQ.local_range(2)), J_PP(dU_qPQ.local_range(1)));
              dDm_skPQ.communicator()->all_reduce_in_place_n(J_PP.data(), J_PP.size(), std::plus<>{});

              auto F_skPQ = dF_skPQ.local();
              for (long is = 0; is < ns; ++is) {
                for (long ik = 0; ik < nkpts_ibz; ++ik) {
                  for (auto idx : diag_idx)
                    F_skPQ(is, ik, idx.first, idx.second) += J_PP(idx.first + P_origin);
                }
              }
            }
          }

          // Exchange: K(R) = -Dm(R) ⊙ U(R)
          auto had_prod2 = nda::map([](ComplexType x, ComplexType y) { return -1.0 * (x * y); });
          for (long s = 0; s < ns; ++s) {
            auto K_RPQ = dDm_skPQ.local()(s, nda::ellipsis{});
            K_RPQ = had_prod2(K_RPQ, U_RPQ_loc);
          }
          auto& dK_sRPQ = dDm_skPQ;

          // FT K R→k, accumulate onto F_PQ
          if (nkpts != 1) {
            auto f_kR = sf_Rk.local();
            utils::R_to_k_coefficients(mpi.comm, nda::range(nkpts), MF->kpts(), MF->lattv(), R_grid, sf_Rk);
            auto K_R_3D = nda::reshape(dK_sRPQ.local(), std::array<long, 3>{ns, nkpts, NP_loc * NQ_loc});
            auto F_k_3D = nda::reshape(dF_skPQ.local(), std::array<long, 3>{ns, nkpts_ibz, NP_loc * NQ_loc});
            ComplexType scl = (ip == iq) ? ComplexType(1.0) : ComplexType(2.0);
            for (int s = 0; s < ns; ++s) {
              nda::blas::gemm(scl, f_kR(nda::range(nkpts_ibz), nda::range::all),
                              K_R_3D(s, nda::ellipsis{}), ComplexType(1.0), F_k_3D(s, nda::ellipsis{}));
            }
          } else {
            ComplexType scl = (ip == iq) ? ComplexType(1.0) : ComplexType(2.0);
            auto F_loc = dF_skPQ.local();
            auto K_loc = dK_sRPQ.local();
            for (long is = 0; is < ns; ++is)
              for (long ik = 0; ik < nkpts_ibz; ++ik)
                F_loc(is, ik, nda::ellipsis{}) += scl * K_loc(is, 0, nda::ellipsis{});
          }
        } // iq
      } // ip
    }

    dU_qPQ.reset();
    dDm_skPQ.reset();
    mpi.comm.barrier();

    app_log(2, "lr_precompute_V_HF_PQ: done");
    return dF_skPQ;
  }


  /**
   * @brief Compute the DeltaX aux→primary IBC correction for a rank-4 distributed PQ array.
   *
   * For each (s, k_ibz):
   *   result(s,k,m,n) = δX_mq†(k+q) · A_PQ(s,k) · X(k)
   *                    + X†(k+q) · A_PQ(s,k+q) · δX(k)
   *
   * where A_PQ is the unperturbed quantity in aux basis, and X is the THC collocation matrix.
   *
   * Postcondition: `result_skij` is **byte-identical on every MPI rank** (final
   * step of this routine broadcasts from rank 0). Callers like
   * `compute_DeltaSigma_ibc` rely on this to write into a node-local
   * shared_array with only a `node_comm.root()` guard — every node's root sees
   * the same value, so no internode communication is needed afterwards.
   *
   * @param result_skij  - [OUTPUT] (ns, nk_ibz, nb, nb) — zeroed and filled
   * @param A_skPQ       - [INPUT] distributed (ns, nk, NP, NP), k-space indexed
   * @param ibc          - [INPUT] lr_ibc_DeltaX with DeltaX, DeltaX_minusq, kpq_map
   * @param thc          - [INPUT] THC-ERI handler
   * @param kp_map       - [INPUT] IBZ k → full BZ k mapping, i.e. ks_to_k(0) (nkpts_ibz,)
   */
  template<typename dArray_4D_t>
  void compute_ibc_aux_to_primary(
      nda::array<ComplexType, 4>& result_skij,
      const dArray_4D_t& A_skPQ,
      const lr_ibc_DeltaX& ibc,
      THC_ERI auto& thc,
      nda::ArrayOfRank<1> auto const& kp_map) {

    auto mpi = thc.mpi();
    auto MF  = thc.MF();

    long ns = A_skPQ.global_shape()[0];
    long nkpts = A_skPQ.global_shape()[1];
    long NP_loc = A_skPQ.local_shape()[2];
    long NQ_loc = A_skPQ.local_shape()[3];
    long P_offset = A_skPQ.origin()[2];
    long Q_offset = A_skPQ.origin()[3];
    long nkpts_ibz = result_skij.shape(1);
    long nbnd = result_skij.shape(2);

    utils::check(nkpts == A_skPQ.local_shape()[1],
                 "compute_ibc_aux_to_primary: Does not support k-distributed PQ arrays.");

    nda::range P_rng(P_offset, P_offset + NP_loc);
    nda::range Q_rng(Q_offset, Q_offset + NQ_loc);
    decltype(nda::range::all) all;

    result_skij = ComplexType(0.0);

    nda::array<ComplexType, 2> Ask_aQ(nbnd, NQ_loc);
    nda::matrix<ComplexType> Oab_buffer(nbnd, nbnd);

    auto A_loc = A_skPQ.local();  // (ns, nk, NP_loc, NQ_loc)

    for (long is = 0; is < ns; ++is) {
      for (long ik_ibz = 0; ik_ibz < nkpts_ibz; ++ik_ibz) {
        long k = kp_map(ik_ibz);
        int kpq = ibc.kpq_map(k);

        auto X_L = thc.X(is, 0, kpq);   // X(k+q), (NP, nb)
        auto X_R = thc.X(is, 0, k);      // X(k),   (NP, nb)
        auto DX   = ibc.DeltaX(is, k, all, all);          // δ^q X(k), stored at index k
        // DeltaX_minusq[ik] stores δ^{-q} X(k_ik + q). We want δ^{-q} X(k+q) for outer
        // loop index k, so read at index k — NOT kpq (the BZ index of k+q).
        auto DX_mq = ibc.DeltaX_minusq(is, k, all, all);  // δ^{-q} X(k+q)

        Oab_buffer = ComplexType(0.0);

        // From δ^p A_mn^k = X^{k'*} δ^p A^k X^k + [δ^{-p} X^{k'*}] A^k X^k + X^{k'*} A^{k'} [δ^p X^k]
        // where k' = k+q. Term 1 uses A at k, Term 2 uses A at k' (NOT k).
        // This is because the left X is at k' and the right X is at k in the LR transform.

        // Term 1: [δ^{-q} X(k+q)]† · A_PQ(k) · X(k)
        nda::blas::gemm(nda::dagger(DX_mq(P_rng, all)), A_loc(is, k, all, all), Ask_aQ);
        nda::blas::gemm(ComplexType(1.0), Ask_aQ, X_R(Q_rng, all),
                        ComplexType(1.0), Oab_buffer);

        // Term 2: X(k+q)† · A_PQ(k+q) · δ^q X(k)   [note: A at k+q, not k]
        nda::blas::gemm(nda::dagger(X_L(P_rng, all)), A_loc(is, kpq, all, all), Ask_aQ);
        nda::blas::gemm(ComplexType(1.0), Ask_aQ, DX(Q_rng, all),
                        ComplexType(1.0), Oab_buffer);

        // Reduce across PQ ranks
        mpi->comm.reduce_in_place_n(Oab_buffer.data(), Oab_buffer.size(), std::plus<>{}, 0);
        if (mpi->comm.rank() == 0) {
          result_skij(is, ik_ibz, all, all) = Oab_buffer;
        }
      }
    }

    // Broadcast from rank 0
    mpi->comm.broadcast_n(result_skij.data(), result_skij.size(), 0);
  }


  /**
   * @brief Compute DeltaSigma_ibc: the DeltaX aux→primary correction on the
   *        unperturbed GW self-energy Σ_PQ.
   *
   * For each τ-point, computes Σ_PQ(τ) = -G_PQ(τ) ⊙ W_PQ(τ) in R-space,
   * then applies the DeltaX aux→primary correction to get band-space IBC.
   *
   * Uses the same algorithm as gw_t::eval_Sigma_all_Rspace but per-τ and
   * without the final aux→primary step (replaced by DeltaX correction).
   *
   * @param dW_tRPQ  - [INPUT] W_c in R-space (nt_half, nR, NP, NP).
   *                   Distribution: pgrid = (tpools, 1, np_P, np_Q).
   *                   Already FFT'd to R-space by lr_precompute_W_tRPQ.
   */
  template<typename mpi_context_t, typename dArray_4D_t>
  void compute_DeltaSigma_ibc(
      typename lr_ibc_DeltaX::sArray_5D_t& sResult_tskij,
      mpi_context_t& mpi,
      const mf::MF* MF,
      THC_ERI auto& thc,
      const lr_ibc_DeltaX& ibc,
      dArray_4D_t& dW_tRPQ) {

    using local_Array_4D_t = nda::array<ComplexType, 4>;
    using math::nda::make_distributed_array;
    using shape_t_3 = std::array<long, 3>;

    long ns = MF->nspin();
    long nkpts = MF->nkpts();
    long nkpts_ibz = MF->nkpts_ibz();
    long nbnd = MF->nbnd();
    long NP = thc.Np();

    auto [nt_half_W, nR_W, NP_W, NQ_W] = dW_tRPQ.global_shape();
    long nt_half = nt_half_W;
    long nt = sResult_tskij.shape()[0];  // full tau grid

    auto [tpools, qpools_W, np_P_W, np_Q_W] = dW_tRPQ.grid();

    // Determine PQ grid for rank-4 buffers: all processes, no tau distribution.
    int np = mpi.comm.size();
    int np_P = utils::find_proc_grid_min_diff(np, 1, 1);
    int np_Q = np / np_P;

    // Gather W into a local array with pgrid (1,1,np_P,np_Q) so all ranks
    // own all τ-points. This is a one-time cost for IBC precomputation.
    auto dW_full = make_distributed_array<local_Array_4D_t>(
        mpi.comm, {1, 1, np_P, np_Q}, {nt_half, nkpts, NP, NP});
    math::nda::redistribute(dW_tRPQ, dW_full);

    auto NP_loc = dW_full.local_shape()[2];
    auto NQ_loc = dW_full.local_shape()[3];

    // Allocate rank-4 buffers for per-τ computation
    auto dG_skPQ = make_distributed_array<local_Array_4D_t>(
        mpi.comm, {1, 1, np_P, np_Q}, {ns, nkpts, NP, NP});
    auto dSigma_skPQ = make_distributed_array<local_Array_4D_t>(
        mpi.comm, {1, 1, np_P, np_Q}, {ns, nkpts, NP, NP});

    // FFT coefficients
    math::shm::shared_array<nda::array_view<ComplexType, 2>> sf_Rk(mpi, {nkpts, nkpts});
    math::shm::shared_array<nda::array_view<ComplexType, 2>> sf_kR(mpi, {nkpts, nkpts});
    if (nkpts != 1) {
      utils::k_to_R_coefficients(mpi.comm, nda::range(nkpts), MF->kpts(), MF->lattv(), MF->kp_grid(), sf_Rk);
      utils::R_to_k_coefficients(mpi.comm, nda::range(nkpts), MF->kpts(), MF->lattv(), MF->kp_grid(), sf_kR);
    }
    nda::matrix<ComplexType> fft_buffer;
    if (nkpts != 1) fft_buffer.resize(nkpts, NP_loc * NQ_loc);

    auto had_neg = nda::map([](ComplexType x, ComplexType y) { return -x * y; });

    // Temp for DeltaX correction per τ-point
    nda::array<ComplexType, 4> Sigma_ibc_tau(std::array<long,4>{ns, nkpts_ibz, nbnd, nbnd});

    app_log(2, "    IBC: computing DeltaSigma_ibc ({} τ-points)...", nt);

    // We iterate over the half-grid. For each it_w, compute both forward (τ) and backward (β-τ).
    for (long it_w = 0; it_w < nt_half; ++it_w) {
      for (int pass = 0; pass < 2; ++pass) {
        bool backward = (pass == 1);
        if (backward && it_w >= nt / 2) continue;  // skip if past midpoint

        long it_g = backward ? (nt - 1 - it_w) : it_w;
        long it_out = backward ? (nt - 1 - it_w) : it_w;

        // Step 1: G(τ) → G_PQ via standard primary_to_aux (q=0)
        auto G_tau = ibc.sG_tskij->local()(it_g, nda::ellipsis{});
        solvers::thc_solver_comm::primary_to_aux(0, 0, G_tau, dG_skPQ, thc,
                                                  MF->kp_to_ibz(), MF->kp_trev());

        // Step 2: k→R FFT on G_PQ
        if (nkpts != 1) {
          auto f_Rk = sf_Rk.local();
          auto G_3D = nda::reshape(dG_skPQ.local(), shape_t_3{ns, nkpts, NP_loc * NQ_loc});
          for (long s = 0; s < ns; ++s) {
            nda::blas::gemm(f_Rk, G_3D(s, nda::ellipsis{}), fft_buffer);
            G_3D(s, nda::ellipsis{}) = fft_buffer;
          }
        }

        // Step 3: Σ_RPQ = -G_RPQ ⊙ W_RPQ (Hadamard product)
        // All ranks own all τ-points (tpools=1 after redistribution).
        // W(τ) is used for both forward and backward passes because W_c(β-τ) = W_c(τ).
        auto W_loc = dW_full.local();  // (nt_half, nR, NP_loc, NQ_loc)
        for (long s = 0; s < ns; ++s) {
          auto G_RPQ = dG_skPQ.local()(s, nda::ellipsis{});  // (nR, NP_loc, NQ_loc)
          auto W_RPQ = W_loc(it_w, nda::ellipsis{});           // (nR, NP_loc, NQ_loc)
          dSigma_skPQ.local()(s, nda::ellipsis{}) = had_neg(G_RPQ, W_RPQ);
        }

        // Step 4: R→k FFT on Σ_PQ
        if (nkpts != 1) {
          auto f_kR = sf_kR.local();
          auto Sigma_3D = nda::reshape(dSigma_skPQ.local(), shape_t_3{ns, nkpts, NP_loc * NQ_loc});
          for (long s = 0; s < ns; ++s) {
            nda::blas::gemm(f_kR, Sigma_3D(s, nda::ellipsis{}), fft_buffer);
            Sigma_3D(s, nda::ellipsis{}) = fft_buffer;
          }
        }

        // Step 5: Compute DeltaX aux→primary correction on Σ_PQ.
        // Sigma_ibc_tau is identical on every rank after compute_ibc_aux_to_primary
        // (internal allreduce), so only the node root needs to write into the
        // shared-memory window for sResult_tskij.
        compute_ibc_aux_to_primary(Sigma_ibc_tau, dSigma_skPQ, ibc, thc, MF->ks_to_k(0));
        if (mpi.node_comm.root()) {
          sResult_tskij.local()(it_out, nda::ellipsis{}) = Sigma_ibc_tau;
        }
      } // pass
    } // it_w

    mpi.comm.barrier();
  }


  /**
   * @brief Build a fully initialized lr_ibc_DeltaX with precomputed IBC corrections.
   *
   * Computes:
   *   1. V_HF_PQ directly in aux basis (via lr_precompute_V_HF_PQ)
   *   2. DeltaF_ibc = DeltaX correction on V_HF_PQ
   *   3. DeltaSigma_ibc = DeltaX correction on Σ_GW_PQ
   *
   * @param dW_tRPQ  - [INPUT] W_c in R-space, nullable (required if include_gw_sigma)
   */
  template<typename mpi_context_t, typename dW_t>
  lr_ibc_DeltaX build_lr_ibc(
      mpi_context_t& mpi,
      const mf::MF* MF,
      THC_ERI auto& thc,
      nda::array_view<ComplexType, 4> DeltaX_arr,
      nda::array_view<ComplexType, 4> DeltaX_minusq_arr,
      const nda::array<double, 1>& q_vec,
      const nda::array<int, 1>& kpq_map,
      const nda::array<ComplexType, 4>* Dm_ab,
      const lr_ibc_DeltaX::sArray_5D_t* sG_tskij,
      dW_t* dW_tRPQ,
      bool include_hartree, bool include_exchange, bool include_gw_sigma) {

    long ns = MF->nspin();
    long nkpts_ibz = MF->nkpts_ibz();
    long nbnd = MF->nbnd();

    // Initialize empty IBC correction arrays
    nda::array<ComplexType, 4> DeltaF_ibc(std::array<long,4>{ns, nkpts_ibz, nbnd, nbnd});
    DeltaF_ibc = ComplexType(0.0);

    // Partially construct ibc so we can pass it to compute functions.
    // sDeltaSigma_ibc_tskij is left as std::nullopt and emplaced below if the
    // GW-Sigma path is active.
    lr_ibc_DeltaX ibc{
        DeltaX_arr, DeltaX_minusq_arr,
        q_vec, kpq_map,
        Dm_ab, sG_tskij,
        std::move(DeltaF_ibc),
        std::nullopt
    };

    // --- Compute DeltaF_ibc from V_HF_PQ ---
    if (Dm_ab && (include_hartree || include_exchange)) {
      app_log(2, "    IBC: computing V_HF_PQ directly in aux basis...");

      // Use lr_precompute_V_HF_PQ (from lr_precompute.hpp)
      auto dF_PQ = lr_precompute_V_HF_PQ(mpi, MF, thc, *Dm_ab,
                                           include_hartree, include_exchange);

      // Compute DeltaX correction on V_HF_PQ → DeltaF_ibc
      ibc.DeltaF_ibc_skij.resize({ns, nkpts_ibz, nbnd, nbnd});
      compute_ibc_aux_to_primary(ibc.DeltaF_ibc_skij, dF_PQ, ibc, thc, MF->ks_to_k(0));
      {
        double norm2 = 0.0;
        for (long is = 0; is < ns; ++is)
          for (long ik = 0; ik < nkpts_ibz; ++ik)
            norm2 += std::pow(nda::frobenius_norm(ibc.DeltaF_ibc_skij(is, ik, nda::range::all, nda::range::all)), 2);
        app_log(2, "    IBC: DeltaF_ibc computed, norm = {:.6e}", std::sqrt(norm2));
      }
    }

    // --- Compute DeltaSigma_ibc from Σ_GW_PQ ---
    if (include_gw_sigma && dW_tRPQ && sG_tskij) {
      long nt = sG_tskij->shape()[0];
      ibc.sDeltaSigma_ibc_tskij.emplace(
          math::shm::make_shared_array<nda::array_view<ComplexType, 5>>(
              mpi, std::array<long, 5>{nt, ns, nkpts_ibz, nbnd, nbnd}));
      if (mpi.node_comm.root()) {
        ibc.sDeltaSigma_ibc_tskij->local() = ComplexType(0.0);
      }
      mpi.comm.barrier();
      compute_DeltaSigma_ibc(*ibc.sDeltaSigma_ibc_tskij, mpi, MF, thc, ibc, *dW_tRPQ);
      {
        // All ranks see the same node-shared data; compute the norm locally
        // and let app_log gate by global rank.
        auto sigma_view = ibc.sDeltaSigma_ibc_tskij->local();
        double norm2 = 0.0;
        for (long it = 0; it < nt; ++it)
          for (long is = 0; is < ns; ++is)
            for (long ik = 0; ik < nkpts_ibz; ++ik)
              norm2 += std::pow(nda::frobenius_norm(sigma_view(it, is, ik, nda::range::all, nda::range::all)), 2);
        app_log(2, "    IBC: DeltaSigma_ibc computed, norm = {:.6e}", std::sqrt(norm2));
      }
    }

    return ibc;
  }

} // namespace methods

#endif // COQUI_LR_IBC_HPP
