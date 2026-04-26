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


#include <optional>

#include "mpi3/communicator.hpp"
#include "nda/nda.hpp"
#include "nda/blas.hpp"
#include "numerics/shared_array/nda.hpp"
#include "numerics/distributed_array/nda.hpp"
#include "numerics/nda_functions.hpp"

#include "IO/app_loggers.h"
#include "utilities/mpi_context.h"
#include "utilities/Timer.hpp"
#include "utilities/kpoint_utils.hpp"
#include "utilities/lr_utils.hpp"

#include "mean_field/MF.hpp"
#include "numerics/imag_axes_ft/IAFT.hpp"
#include "methods/ERI/detail/concepts.hpp"
#include "methods/ERI/div_treatment_e.hpp"
#include "methods/ERI/thc_reader_t.hpp"
#include "methods/HF/lr_thc_comm.hpp"
#include "methods/GW/gw_t.h"
// Include thc_gw.icc for Sigma_div_correction definition (template in gw_t)
#include "methods/GW/thc_gw.icc"
#include "methods/GW/lr_gw.hpp"

namespace methods {
  namespace solvers {

    lr_gw::lr_gw(const imag_axes_ft::IAFT *ft,
                 nda::array<double, 1> const& q_pert,
                 div_treatment_e div)
      : _q_pert(q_pert),
        _gw(ft, div),
        _Timer() {

      for (auto& v: {"EVALUATE_SIGMA_R",
                      "SIGMA_ALLOC",
                      "SIGMA_PRIM_TO_AUX", "SIGMA_FT_R",
                      "SIGMA_HADPROD_R", "SIGMA_AUX_TO_PRIM"}) {
        _Timer.add(v);
      }
    }

    void lr_gw::_init_kpq_map(thc_reader_t& thc) {
      if (_kpq_map_initialized) return;
      auto MF = thc.MF();
      long nkpts = MF->nkpts();
      auto kpts_crys = MF->kpts_crystal();
      _kpq_map.resize(nkpts);
      utils::calculate_kpq_map(kpts_crys, _q_pert, _kpq_map);
      _kpq_map_initialized = true;
    }

    // =========================================================================
    // evaluate_sigma: Both terms fused, ΔΣ = -ΔG ⊙ W_c - G ⊙ ΔW
    // =========================================================================
    void lr_gw::evaluate_sigma(
        sArrv_5D_t& sDeltaSigma_tskij,
        const nda::array_view<ComplexType, 5>& DeltaG_tskij,
        dArr_4D_t& dW_tRPQ,
        const nda::array_view<ComplexType, 5>& G_tskij,
        dArr_4D_t& dDeltaW_qtPQ,
        thc_reader_t& thc,
        const lr_ibc_DeltaX* ibc) {

      _init_kpq_map(thc);

      app_log(2, "\n  LR-GW self-energy (fused): ΔΣ = -ΔG ⊙ W_c - G ⊙ ΔW (R-space)");
      app_log(2, "    q_pert = ({:.6f}, {:.6f}, {:.6f})", _q_pert(0), _q_pert(1), _q_pert(2));

      sDeltaSigma_tskij.set_zero();
      _eval_sigma_Rspace(sDeltaSigma_tskij, thc, &DeltaG_tskij, &dW_tRPQ, &G_tskij, &dDeltaW_qtPQ,
                         ibc);

      app_log(2, "  LR-GW self-energy (fused) done.\n");
      print_timers();
    }


    // =========================================================================
    // evaluate_sigma_DeltaG: Term 1, ΔΣ = -ΔG ⊙ W_c
    // =========================================================================
    void lr_gw::evaluate_sigma_DeltaG(
        sArrv_5D_t& sDeltaSigma_tskij,
        const nda::array_view<ComplexType, 5>& DeltaG_tskij,
        dArr_4D_t& dW_tRPQ,
        thc_reader_t& thc,
        const lr_ibc_DeltaX* ibc) {

      _init_kpq_map(thc);

      app_log(2, "\n  LR-GW self-energy (term 1): ΔΣ = -ΔG ⊙ W_c (R-space)");
      app_log(2, "    q_pert = ({:.6f}, {:.6f}, {:.6f})", _q_pert(0), _q_pert(1), _q_pert(2));

      sDeltaSigma_tskij.set_zero();
      _eval_sigma_Rspace(sDeltaSigma_tskij, thc, &DeltaG_tskij, &dW_tRPQ, nullptr, nullptr,
                         ibc);

      app_log(2, "  LR-GW self-energy (term 1) done.\n");
      print_timers();
    }


    // =========================================================================
    // evaluate_sigma_DeltaW: Term 2, ΔΣ = -G ⊙ ΔW
    // =========================================================================
    void lr_gw::evaluate_sigma_DeltaW(
        sArrv_5D_t& sDeltaSigma_tskij,
        const nda::array_view<ComplexType, 5>& G_tskij,
        dArr_4D_t& dDeltaW_qtPQ,
        thc_reader_t& thc) {

      _init_kpq_map(thc);

      app_log(2, "\n  LR-GW self-energy (term 2): ΔΣ = -G ⊙ ΔW (R-space, no div_corr)");
      app_log(2, "    q_pert = ({:.6f}, {:.6f}, {:.6f})", _q_pert(0), _q_pert(1), _q_pert(2));

      sDeltaSigma_tskij.set_zero();
      _eval_sigma_Rspace(sDeltaSigma_tskij, thc, nullptr, nullptr, &G_tskij, &dDeltaW_qtPQ);

      app_log(2, "  LR-GW self-energy (term 2) done.\n");
      print_timers();
    }


    // =========================================================================
    // apply_div_correction_DeltaG: Term 1 div corr (all q_pert)
    //   ΔΣ += -madelung * eps_inv_head * S(k+q) · ΔG · S(k)
    // =========================================================================
    void lr_gw::apply_div_correction_DeltaG(
        sArrv_5D_t& sDeltaSigma_tskij,
        const nda::array_view<ComplexType, 5>& DeltaG_tskij,
        const nda::array_view<ComplexType, 4>& S_skij,
        thc_reader_t& thc,
        const nda::array<ComplexType, 1>& eps_inv_head) {
      _init_kpq_map(thc);
      app_log(2, "  LR div correction (term 1, ΔG): S(k+q) · ΔG · S(k)");
      _sigma_div_correction(sDeltaSigma_tskij, DeltaG_tskij, S_skij, thc, eps_inv_head);
    }


    // =========================================================================
    // apply_div_correction_G: Term 2 div corr (q_pert=0 only)
    //   ΔΣ += -madelung * Δeps_inv_head * S(k) · G · S(k)
    // =========================================================================
    void lr_gw::apply_div_correction_G(
        sArrv_5D_t& sDeltaSigma_tskij,
        const nda::array_view<ComplexType, 5>& G_tskij,
        const nda::array_view<ComplexType, 4>& S_skij,
        thc_reader_t& thc,
        const nda::array<ComplexType, 1>& delta_eps_inv_head) {
      utils::check(utils::is_q_gamma(_q_pert),
                   "apply_div_correction_G: term 2 is only valid at q_pert=0, "
                   "but q_pert = ({}, {}, {})", _q_pert(0), _q_pert(1), _q_pert(2));
      app_log(2, "  LR div correction (term 2, G): S(k) · G · S(k) [q_pert=0]");
      _gw.Sigma_div_correction(sDeltaSigma_tskij, G_tskij, S_skij, thc, delta_eps_inv_head);
    }


    // =========================================================================
    // _eval_sigma_Rspace: unified per-tau R-space convolution workhorse
    // =========================================================================
    void lr_gw::_eval_sigma_Rspace(
        sArrv_5D_t& sDeltaSigma_tskij,
        thc_reader_t& thc,
        const nda::array_view<ComplexType, 5>* DeltaG_tskij,
        dArr_4D_t* dW_tRPQ,
        const nda::array_view<ComplexType, 5>* G_tskij,
        dArr_4D_t* dDeltaW_qtPQ,
        const lr_ibc_DeltaX* ibc) {
      _Timer.start("EVALUATE_SIGMA_R");

      bool do_term1 = (DeltaG_tskij != nullptr);
      bool do_term2 = (G_tskij != nullptr);
      utils::check(do_term1 || do_term2,
                   "lr_gw::_eval_sigma_Rspace: at least one term must be active");
      utils::check(!do_term1 || dW_tRPQ != nullptr,
                   "lr_gw::_eval_sigma_Rspace: DeltaG requires dW_tRPQ");
      utils::check(!do_term2 || dDeltaW_qtPQ != nullptr,
                   "lr_gw::_eval_sigma_Rspace: G requires dDeltaW_qtPQ");

      auto MF = thc.MF();
      auto mpi = thc.mpi();
      utils::check(MF->nqpts() == MF->nqpts_ibz() and MF->nqpts() == MF->nkpts(),
                   "lr_gw::_eval_sigma_Rspace: Symmetry not allowed. nqpts={}, nqpts_ibz={}, nkpts={}",
                   MF->nqpts(), MF->nqpts_ibz(), MF->nkpts());

      // Get dimensions from whichever G is available
      auto& G_ref = do_term1 ? *DeltaG_tskij : *G_tskij;
      auto ns = G_ref.shape(1);
      auto nt = G_ref.shape(0);
      auto nk_ibz = G_ref.shape(2);
      auto nbnd = G_ref.shape(3);

      // Extract tau distribution and array dimensions from whichever W is available.
      // Term 1: dW_tRPQ has (t,R,P,Q) layout, pgrid (tpools,1,np_P,np_Q).
      // Term 2: dDeltaW_qtPQ has (q,t,P,Q) layout, pgrid (1,tpools,np_P,np_Q).
      dArr_4D_t& dW_ref = do_term1 ? *dW_tRPQ : *dDeltaW_qtPQ;
      auto dW_lshape = dW_ref.local_shape();
      auto dW_gshape = dW_ref.global_shape();
      auto dW_grid = dW_ref.grid();
      auto dW_bsize = dW_ref.block_size();

      auto NP_loc = dW_lshape[2], NQ_loc = dW_lshape[3];
      auto NP = dW_gshape[2], NQ = dW_gshape[3];
      auto np_P = dW_grid[2], np_Q = dW_grid[3];
      auto P_bs = dW_bsize[2], Q_bs = dW_bsize[3];

      // Term 1's W: axis 0 = t (distributed), axis 1 = R (undivided)
      long nt_loc, nkpts, t_origin;
      if (do_term1) {
        nt_loc = dW_tRPQ->local_shape()[0];
        nkpts = dW_tRPQ->global_shape()[1];
        t_origin = dW_tRPQ->origin()[0];
        utils::check(dW_tRPQ->local_shape()[1] == nkpts,
                     "lr_gw: R-axis must be undivided, nR_loc={} != nkpts={}", dW_tRPQ->local_shape()[1], nkpts);
        utils::check(dW_tRPQ->grid()[1] == 1,
                     "lr_gw: R-axis must be undivided, pgrid[1]={}", dW_tRPQ->grid()[1]);
      } else {
        // Term 2 only: axis 0 = q (undivided), axis 1 = t (distributed)
        nkpts = dDeltaW_qtPQ->local_shape()[0];
        nt_loc = dDeltaW_qtPQ->local_shape()[1];
        t_origin = dDeltaW_qtPQ->origin()[1];
        utils::check(dDeltaW_qtPQ->local_shape()[0] == nkpts,
                     "lr_gw: q-axis must be undivided, nq_loc={} != nkpts={}", dDeltaW_qtPQ->local_shape()[0], nkpts);
        utils::check(dDeltaW_qtPQ->grid()[0] == 1,
                     "lr_gw: q-axis must be undivided, pgrid[0]={}", dDeltaW_qtPQ->grid()[0]);
      }

      // Cross-check term 2 tau distribution if both terms active
      if (do_term1 && do_term2) {
        utils::check(dDeltaW_qtPQ->local_shape()[1] == nt_loc,
                     "lr_gw: term1 nt_loc={} != term2 nt_loc={}", nt_loc, dDeltaW_qtPQ->local_shape()[1]);
        utils::check(dDeltaW_qtPQ->origin()[1] == t_origin,
                     "lr_gw: term1 t_origin={} != term2 t_origin={}", t_origin, dDeltaW_qtPQ->origin()[1]);
      }

      app_log(2, "    nt_loc={}, nkpts={}, t_origin={}, NP_loc={}, NQ_loc={}",
              nt_loc, nkpts, t_origin, NP_loc, NQ_loc);

      // === Step 0: Create tau subcommunicator and allocate rank-4 work arrays ===
      _Timer.start("SIGMA_ALLOC");

      mpi3::communicator tau_comm = mpi->comm.split(t_origin, mpi->comm.rank());
      auto tau_mpi = utils::make_mpi_context(tau_comm);

      using local_Array_4D_t = memory::array<HOST_MEMORY, ComplexType, 4>;
      using math::nda::make_distributed_array;

      auto dG_skPQ = make_distributed_array<local_Array_4D_t>(
          tau_comm, {1, 1, np_P, np_Q}, {ns, nkpts, NP, NQ},
          {1, 1, P_bs, Q_bs});

      auto dSigma_skPQ = make_distributed_array<local_Array_4D_t>(
          tau_comm, {1, 1, np_P, np_Q}, {ns, nkpts, NP, NQ},
          {1, 1, P_bs, Q_bs});

      using Arrv_4D_t = nda::array_view<ComplexType, 4>;
      auto sSigma_skij = math::shm::make_shared_array<Arrv_4D_t>(
          tau_mpi, {ns, nk_ibz, nbnd, nbnd});

      nda::matrix<ComplexType> ft_buffer(nkpts, NP_loc * NQ_loc);

      // W tau-slice buffers: one per active term
      nda::array<ComplexType, 3> W1_tau_RPQ;
      if (do_term1) W1_tau_RPQ.resize(nkpts, NP_loc, NQ_loc);

      nda::array<ComplexType, 3> W2_tau_RPQ;
      if (do_term2) W2_tau_RPQ.resize(nkpts, NP_loc, NQ_loc);

      _Timer.stop("SIGMA_ALLOC");

      // === Step 1: Compute FT coefficients (once) ===
      using sArray_2D_t = math::shm::shared_array<nda::array_view<ComplexType, 2>>;
      sArray_2D_t sf_Rk(tau_mpi, {nkpts, nkpts});
      sArray_2D_t sf_kR(tau_mpi, {nkpts, nkpts});

      if (nkpts != 1) {
        utils::k_to_R_coefficients(tau_comm, nda::range(nkpts), MF->kpts(), MF->lattv(), MF->kp_grid(), sf_Rk);
        utils::R_to_k_coefficients(tau_comm, nda::range(nkpts), MF->kpts(), MF->lattv(), MF->kp_grid(), sf_kR);
      }

      std::optional<sArray_2D_t> opt_sf_qR;
      if (nkpts != 1 && do_term2) {
        opt_sf_qR.emplace(tau_mpi, std::array<long, 2>{nkpts, nkpts});
        utils::k_to_R_coefficients(tau_comm, nda::range(nkpts), MF->Qpts(), MF->lattv(), MF->kp_grid(), *opt_sf_qR);
      }

      // Hadamard product lambda
      auto neg_prod = nda::map([](ComplexType x, ComplexType y) { return -1.0 * (x * y); });

      // === Step 2: Per-tau loop with inner minus_t loop ===
      sDeltaSigma_tskij.win().fence();

      for (long it_local = 0; it_local < nt_loc; ++it_local) {
        long it_w = t_origin + it_local;  // global W tau index (first half: 0..nt_half-1)

        // --- Copy W slices once per tau (reused for both minus_t passes) ---
        if (do_term1) {
          // dW_tRPQ layout: (t, R, P, Q)
          for (long iR = 0; iR < nkpts; ++iR)
            W1_tau_RPQ(iR, nda::ellipsis{}) = dW_tRPQ->local()(it_local, iR, nda::ellipsis{});
        }
        if (do_term2) {
          // dDeltaW_qtPQ layout: (q, t, P, Q) — copy then q→R FT
          for (long iq = 0; iq < nkpts; ++iq)
            W2_tau_RPQ(iq, nda::ellipsis{}) = dDeltaW_qtPQ->local()(iq, it_local, nda::ellipsis{});

          if (nkpts != 1) {
            _Timer.start("SIGMA_FT_R");
            auto f_qR = opt_sf_qR->local();
            auto W2_2D = nda::reshape(W2_tau_RPQ, shape_t<2>{nkpts, NP_loc * NQ_loc});
            nda::blas::gemm(f_qR, W2_2D, ft_buffer);
            W2_2D = ft_buffer;
            _Timer.stop("SIGMA_FT_R");
          }
        }

        // --- Inner loop over forward (τ) and backward (β-τ) passes ---
        for (int pass = 0; pass < 2; ++pass) {
          bool minus_t = (pass == 1);

          // For backward pass, skip if at or past midpoint (already written by forward)
          if (minus_t && it_w >= static_cast<long>(nt / 2)) continue;

          long it_g = minus_t ? (nt - 1 - it_w) : it_w;
          long it_out = minus_t ? (nt - 1 - it_w) : it_w;

          // --- Accumulate Sigma in R-space from active terms ---
          dSigma_skPQ.local() = 0;

          // Term 1: ΔG ⊙ W_c
          if (do_term1) {
            _Timer.start("SIGMA_PRIM_TO_AUX");
            dG_skPQ.local() = 0;
            auto DeltaG_slice = (*DeltaG_tskij)(it_g, nda::ellipsis{});
            if (ibc && ibc->sG_tskij) {
              // DeltaX correction: pass unperturbed G(τ) view as O_unpert (no copy)
              auto G_tau_view = ibc->sG_tskij->local()(it_g, nda::ellipsis{});
              lr_thc_comm::primary_to_aux(0, 0, DeltaG_slice, dG_skPQ, thc,
                                          MF->kp_to_ibz(), MF->kp_trev(), _kpq_map,
                                          ibc, &G_tau_view);
            } else {
              lr_thc_comm::primary_to_aux(0, 0, DeltaG_slice, dG_skPQ, thc,
                                          MF->kp_to_ibz(), MF->kp_trev(), _kpq_map);
            }
            _Timer.stop("SIGMA_PRIM_TO_AUX");

            if (nkpts != 1) {
              _Timer.start("SIGMA_FT_R");
              auto f_Rk = sf_Rk.local();
              auto G_3D = nda::reshape(dG_skPQ.local(), shape_t<3>{ns, nkpts, NP_loc * NQ_loc});
              for (long s = 0; s < ns; ++s) {
                nda::blas::gemm(f_Rk, G_3D(s, nda::ellipsis{}), ft_buffer);
                G_3D(s, nda::ellipsis{}) = ft_buffer;
              }
              _Timer.stop("SIGMA_FT_R");
            }

            _Timer.start("SIGMA_HADPROD_R");
            for (long s = 0; s < ns; ++s) {
              auto G_RPQ = dG_skPQ.local()(s, nda::ellipsis{});
              auto Sigma_RPQ = dSigma_skPQ.local()(s, nda::ellipsis{});
              Sigma_RPQ += neg_prod(G_RPQ, W1_tau_RPQ);
            }
            _Timer.stop("SIGMA_HADPROD_R");
          }

          // Term 2: G ⊙ ΔW (reuses dG_skPQ buffer)
          if (do_term2) {
            _Timer.start("SIGMA_PRIM_TO_AUX");
            dG_skPQ.local() = 0;
            auto G_slice = (*G_tskij)(it_g, nda::ellipsis{});
            thc_solver_comm::primary_to_aux(0, 0, G_slice, dG_skPQ, thc,
                                           MF->kp_to_ibz(), MF->kp_trev());
            _Timer.stop("SIGMA_PRIM_TO_AUX");

            if (nkpts != 1) {
              _Timer.start("SIGMA_FT_R");
              auto f_Rk = sf_Rk.local();
              auto G_3D = nda::reshape(dG_skPQ.local(), shape_t<3>{ns, nkpts, NP_loc * NQ_loc});
              for (long s = 0; s < ns; ++s) {
                nda::blas::gemm(f_Rk, G_3D(s, nda::ellipsis{}), ft_buffer);
                G_3D(s, nda::ellipsis{}) = ft_buffer;
              }
              _Timer.stop("SIGMA_FT_R");
            }

            _Timer.start("SIGMA_HADPROD_R");
            for (long s = 0; s < ns; ++s) {
              auto G_RPQ = dG_skPQ.local()(s, nda::ellipsis{});
              auto Sigma_RPQ = dSigma_skPQ.local()(s, nda::ellipsis{});
              Sigma_RPQ += neg_prod(G_RPQ, W2_tau_RPQ);
            }
            _Timer.stop("SIGMA_HADPROD_R");
          }

          // --- R→k FFT on accumulated Sigma (once) ---
          if (nkpts != 1) {
            _Timer.start("SIGMA_FT_R");
            auto f_kR = sf_kR.local();
            auto Sigma_3D = nda::reshape(dSigma_skPQ.local(), shape_t<3>{ns, nkpts, NP_loc * NQ_loc});
            for (long s = 0; s < ns; ++s) {
              nda::blas::gemm(f_kR, Sigma_3D(s, nda::ellipsis{}), ft_buffer);
              Sigma_3D(s, nda::ellipsis{}) = ft_buffer;
            }
            _Timer.stop("SIGMA_FT_R");
          }

          // --- aux_to_primary on accumulated Sigma (once) ---
          _Timer.start("SIGMA_AUX_TO_PRIM");
          sSigma_skij.set_zero();
          lr_thc_comm::aux_to_primary(0, 0, ComplexType(1.0), dSigma_skPQ,
                                      sSigma_skij, thc, MF->ks_to_k(0), _kpq_map);

          // Add precomputed IBC correction for this τ-point
          if (ibc && ibc->DeltaSigma_ibc_tskij.size() > 0) {
            if (sSigma_skij.node_comm()->root()) {
              sSigma_skij.local() += ibc->DeltaSigma_ibc_tskij(it_out, nda::ellipsis{});
            }
            sSigma_skij.win().fence();
          }

          if (tau_comm.rank() == 0) {
            sDeltaSigma_tskij.local()(it_out, nda::ellipsis{}) += sSigma_skij.local();
          }
          _Timer.stop("SIGMA_AUX_TO_PRIM");
        } // pass
      } // it_local

      // Synchronize and distribute across all nodes
      sDeltaSigma_tskij.win().fence();
      sDeltaSigma_tskij.all_reduce();

      _Timer.stop("EVALUATE_SIGMA_R");
    }


    // =========================================================================
    // _sigma_div_correction: LR-aware divergence correction
    //   ΔΣ^div_{ij} = factor · S(k+q) · ΔG · S(k)
    // =========================================================================
    void lr_gw::_sigma_div_correction(
        sArrv_5D_t& sDeltaSigma_tskij,
        const nda::array_view<ComplexType, 5>& DeltaG_tskij,
        const nda::array_view<ComplexType, 4>& S_skij,
        thc_reader_t& thc,
        const nda::array<ComplexType, 1>& eps_inv_head) {
      auto div_treatment = _gw.div_treatmemnt();
      app_log(2, "  LR divergence correction: {}", div_enum_to_string(div_treatment));

      if (thc.MF()->nqpts_ibz() == 1 and div_treatment != ignore_g0) {
        app_log(2, "    nqpts_ibz == 1, taking div_treatment = ignore_g0");
        div_treatment = ignore_g0;
      }

      if (div_treatment == ignore_g0) {
        return;
      } else if (div_treatment == gygi or div_treatment == gygi_extrplt or div_treatment == gygi_extrplt_2d) {
        auto MF = thc.MF();
        auto mpi = thc.mpi();
        app_log(2, "    - madelung = {}", MF->madelung());
        using math::shm::make_shared_array;
        auto [nts, ns, nkpts, nbnd, nbnd2] = sDeltaSigma_tskij.shape();
        auto nt_half = (nts % 2 == 0) ? nts / 2 : nts / 2 + 1;

        int rank = mpi->comm.rank();
        int size = mpi->comm.size();

        using Array_view_5D_t = nda::array_view<ComplexType, 5>;
        decltype(nda::range::all) all;

        // ΔΣ^div_{ij} = factor · S(k+q) · ΔG · S(k)
        nda::array<ComplexType, 2> buffer_ib(nbnd, nbnd);
        auto sDelta_tskij = make_shared_array<Array_view_5D_t>(*mpi, sDeltaSigma_tskij.shape());
        auto Delta_loc = sDelta_tskij.local();
        sDelta_tskij.win().fence();
        for (size_t tsk = rank; tsk < nts * ns * nkpts; tsk += size) {
          size_t it = tsk / (ns * nkpts);
          size_t is = (tsk / nkpts) % ns;
          size_t ik = tsk % nkpts;

          size_t it_pos = (it < nt_half) ? it : nts - it - 1;
          RealType factor = -1.0 * MF->madelung() * eps_inv_head(it_pos).real();

          auto DeltaG_ab = DeltaG_tskij(it, is, ik, nda::ellipsis{});
          auto Delta_ij = Delta_loc(it, is, ik, nda::ellipsis{});
          auto S_kq = S_skij(is, _kpq_map(ik), all, all);  // S at k+q
          auto S_k = S_skij(is, ik, all, all);              // S at k
          nda::blas::gemm(S_kq, DeltaG_ab, buffer_ib);
          nda::blas::gemm(ComplexType(factor), buffer_ib, S_k, ComplexType(0.0), Delta_ij);
        }
        sDelta_tskij.win().fence();
        sDelta_tskij.all_reduce();

        if (sDeltaSigma_tskij.node_comm()->root())
          sDeltaSigma_tskij.local() += sDelta_tskij.local();
        sDeltaSigma_tskij.communicator()->barrier();
      } else {
        utils::check(false, "Unsupported divergence treatment: {}", div_enum_to_string(div_treatment));
      }
      app_log(2, "");
    }

  } // solvers
} // methods
