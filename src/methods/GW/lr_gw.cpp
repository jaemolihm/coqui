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
#include "methods/ERI/thc_reader_t.hpp"
#include "methods/HF/lr_thc_comm.hpp"
#include "methods/GW/gw_t.h"
#include "methods/GW/lr_gw.hpp"

namespace methods {
  namespace solvers {

    lr_gw::lr_gw(const imag_axes_ft::IAFT *ft,
                 nda::array<double, 1> const& q_pert,
                 std::string div)
      : _q_pert(q_pert),
        _gw(ft, div),
        _Timer() {

      for (auto& v: {"EVALUATE_SIGMA_R",
                      "SIGMA_ALLOC",
                      "SIGMA_PRIM_TO_AUX", "SIGMA_FT_R",
                      "SIGMA_HADPROD_R", "SIGMA_AUX_TO_PRIM",
                      "SIGMA_FT_COEFF", "SIGMA_W_COPY",
                      "SIGMA_PRE_FENCE", "SIGMA_FINAL_REDUCE",
                      "SIGMA_DIV_CORR",
                      "SIGMA_A2P_SETZERO", "SIGMA_A2P_PREDIV",
                      "SIGMA_A2P_ALLOC", "SIGMA_A2P_GEMM",
                      "SIGMA_A2P_SKEW", "SIGMA_A2P_REDUCE",
                      "SIGMA_A2P_AXPY", "SIGMA_A2P_SHMREDUCE",
                      "SIGMA_A2P_ACCUM"}) {
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
        const dArr_5D_t& dG_tsRPQ,
        const dArr_5D_t& dG_mtau_tsRPQ,
        const lr_ibc_DeltaX* ibc) {

      _init_kpq_map(thc);

      app_log(3, "\n  LR-GW self-energy (fused): ΔΣ = -ΔG ⊙ W_c - G ⊙ ΔW (R-space)");
      app_log(3, "    q_pert = ({:.6f}, {:.6f}, {:.6f})", _q_pert(0), _q_pert(1), _q_pert(2));

      sDeltaSigma_tskij.set_zero();
      _eval_sigma_Rspace(sDeltaSigma_tskij, thc, &DeltaG_tskij, &dW_tRPQ, &G_tskij, &dDeltaW_qtPQ,
                         ibc, &dG_tsRPQ, &dG_mtau_tsRPQ);

      app_log(3, "  LR-GW self-energy (fused) done.\n");
      print_timers(3);  // per-step diagnostics only at verbosity >= 3
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

      app_log(3, "\n  LR-GW self-energy (term 1): ΔΣ = -ΔG ⊙ W_c (R-space)");
      app_log(3, "    q_pert = ({:.6f}, {:.6f}, {:.6f})", _q_pert(0), _q_pert(1), _q_pert(2));

      sDeltaSigma_tskij.set_zero();
      _eval_sigma_Rspace(sDeltaSigma_tskij, thc, &DeltaG_tskij, &dW_tRPQ, nullptr, nullptr,
                         ibc);

      app_log(3, "  LR-GW self-energy (term 1) done.\n");
      print_timers(3);  // per-step diagnostics only at verbosity >= 3
    }


    // =========================================================================
    // evaluate_sigma_DeltaW: Term 2, ΔΣ = -G ⊙ ΔW
    // =========================================================================
    void lr_gw::evaluate_sigma_DeltaW(
        sArrv_5D_t& sDeltaSigma_tskij,
        const nda::array_view<ComplexType, 5>& G_tskij,
        dArr_4D_t& dDeltaW_qtPQ,
        thc_reader_t& thc,
        const dArr_5D_t& dG_tsRPQ,
        const dArr_5D_t& dG_mtau_tsRPQ) {

      _init_kpq_map(thc);

      app_log(3, "\n  LR-GW self-energy (term 2): ΔΣ = -G ⊙ ΔW (R-space, no div_corr)");
      app_log(3, "    q_pert = ({:.6f}, {:.6f}, {:.6f})", _q_pert(0), _q_pert(1), _q_pert(2));

      sDeltaSigma_tskij.set_zero();
      _eval_sigma_Rspace(sDeltaSigma_tskij, thc, nullptr, nullptr, &G_tskij, &dDeltaW_qtPQ,
                         nullptr, &dG_tsRPQ, &dG_mtau_tsRPQ);

      app_log(3, "  LR-GW self-energy (term 2) done.\n");
      print_timers(3);  // per-step diagnostics only at verbosity >= 3
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
      app_log(3, "  LR div correction (term 1, ΔG): S(k+q) · ΔG · S(k)");
      _Timer.start("SIGMA_DIV_CORR");
      _sigma_div_correction(sDeltaSigma_tskij, DeltaG_tskij, S_skij, thc, eps_inv_head);
      _Timer.stop("SIGMA_DIV_CORR");
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
      _init_kpq_map(thc);  // identity at q_pert=0
      app_log(3, "  LR div correction (term 2, G): S(k) · G · S(k) [q_pert=0]");
      _Timer.start("SIGMA_DIV_CORR");
      // At q_pert=0, _kpq_map is the identity, so the LR kernel
      // S(k+q)·G·S(k) reduces exactly to gw_t::Sigma_div_correction's
      // S(k)·G·S(k). Route through the node-local LR kernel to avoid the
      // ground-state version's per-call shm window + global all_reduce.
      _sigma_div_correction(sDeltaSigma_tskij, G_tskij, S_skij, thc, delta_eps_inv_head);
      _Timer.stop("SIGMA_DIV_CORR");
    }


    // =========================================================================
    // _setup_workspace: build the cached per-run workspace on the first call
    // (tau subcommunicator, work arrays, shm windows, FT coefficients / FFT
    // objects). The τ-dist is fixed for the solver's lifetime, so a plain
    // _setup_done guard suffices. PQ grid/block parameters are read from dW_ref.
    // =========================================================================
    void lr_gw::_setup_workspace(thc_reader_t& thc, dArr_4D_t const& dW_ref,
                                  bool do_term1, bool do_term2,
                                  long ns, long nk_ibz, long nbnd) {
      if (_setup_done) {
        // The cached workspace is specific to the term combination it was built
        // for (term2-only buffers are conditionally allocated); reusing it under
        // different flags would deref an empty optional. Fail loudly instead.
        utils::check(do_term1 == _setup_term1 && do_term2 == _setup_term2,
                     "lr_gw::_setup_workspace: workspace built for (term1={}, term2={}) "
                     "but reused with (term1={}, term2={})",
                     _setup_term1, _setup_term2, do_term1, do_term2);
        return;
      }
      using local_Array_4D_t = memory::array<HOST_MEMORY, ComplexType, 4>;
      using math::nda::make_distributed_array;
      using Arrv_4D_t = nda::array_view<ComplexType, 4>;

      auto mpi = thc.mpi();
      auto MF = thc.MF();

      // PQ grid/block from dW_ref (axes 2,3, layout-independent); t_origin and
      // nkpts depend on the layout (term1: t,R,P,Q; term2: q,t,P,Q).
      auto gshape = dW_ref.global_shape();
      auto grid   = dW_ref.grid();
      auto bsize  = dW_ref.block_size();
      auto lshape = dW_ref.local_shape();
      long NP = gshape[2], NQ = gshape[3];
      long NP_loc = lshape[2], NQ_loc = lshape[3];
      long np_P = grid[2], np_Q = grid[3];
      long P_bs = bsize[2], Q_bs = bsize[3];
      long nkpts   = do_term1 ? gshape[1]      : gshape[0];
      long t_origin = do_term1 ? dW_ref.origin()[0] : dW_ref.origin()[1];

      _tau_comm.emplace(mpi->comm.split(t_origin, mpi->comm.rank()));
      _tau_mpi.emplace(utils::make_mpi_context(*_tau_comm));

      _dG_skPQ.emplace(make_distributed_array<local_Array_4D_t>(
          *_tau_comm, {1, 1, np_P, np_Q}, {ns, nkpts, NP, NQ},
          {1, 1, P_bs, Q_bs}));
      _dSigma_skPQ.emplace(make_distributed_array<local_Array_4D_t>(
          *_tau_comm, {1, 1, np_P, np_Q}, {ns, nkpts, NP, NQ},
          {1, 1, P_bs, Q_bs}));
      _sSigma_skij.emplace(math::shm::make_shared_array<Arrv_4D_t>(
          *_tau_mpi, std::array<long, 4>{ns, nk_ibz, nbnd, nbnd}));

      // W2 tau-slice buffer (term 2 only; term 1 uses a contiguous view)
      if (do_term2) _W2_tau_RPQ.resize(nkpts, NP_loc, NQ_loc);

      // k<->R transforms: blocked FFT by default; COQUI_LR_DEBUG_GEMM_FT=1
      // selects the gemm path with explicit FT coefficients (kept for testing).
      _Timer.start("SIGMA_FT_COEFF");
      const bool use_gemm_ft = utils::lr_debug_gemm_ft();
      if (nkpts != 1 && use_gemm_ft) {
        _ft_buffer.resize(nkpts, NP_loc * NQ_loc);
        _sf_Rk.emplace(*_tau_mpi, std::array<long, 2>{nkpts, nkpts});
        _sf_kR.emplace(*_tau_mpi, std::array<long, 2>{nkpts, nkpts});
        utils::k_to_R_coefficients(*_tau_comm, nda::range(nkpts), MF->kpts(), MF->lattv(), MF->kp_grid(), *_sf_Rk);
        utils::R_to_k_coefficients(*_tau_comm, nda::range(nkpts), MF->kpts(), MF->lattv(), MF->kp_grid(), *_sf_kR);
        if (do_term2) {
          _sf_qR.emplace(*_tau_mpi, std::array<long, 2>{nkpts, nkpts});
          utils::k_to_R_coefficients(*_tau_comm, nda::range(nkpts), MF->Qpts(), MF->lattv(), MF->kp_grid(), *_sf_qR);
        }
      }
      // _fft_q is built only for term 2.
      _fft_k.reset();
      _fft_q.reset();
      if (nkpts != 1 && !use_gemm_ft) {
        _fft_k.emplace(MF->kpts(), MF->lattv(), MF->kp_grid());
        if (do_term2)
          _fft_q.emplace(MF->Qpts(), MF->lattv(), MF->kp_grid());
      }
      if (nkpts != 1)
        app_log(3, "    Sigma k<->R transform: {}",
                use_gemm_ft ? "gemm (COQUI_LR_DEBUG_GEMM_FT)" : "FFT");
      _Timer.stop("SIGMA_FT_COEFF");

      _setup_term1 = do_term1;
      _setup_term2 = do_term2;
      _setup_done = true;
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
        const lr_ibc_DeltaX* ibc,
        const dArr_5D_t* dG_tsRPQ,
        const dArr_5D_t* dG_mtau_tsRPQ) {
      _Timer.start("EVALUATE_SIGMA_R");

      bool do_term1 = (DeltaG_tskij != nullptr);
      bool do_term2 = (G_tskij != nullptr);
      utils::check(do_term1 || do_term2,
                   "lr_gw::_eval_sigma_Rspace: at least one term must be active");
      utils::check(!do_term1 || dW_tRPQ != nullptr,
                   "lr_gw::_eval_sigma_Rspace: DeltaG requires dW_tRPQ");
      utils::check(!do_term2 || dDeltaW_qtPQ != nullptr,
                   "lr_gw::_eval_sigma_Rspace: G requires dDeltaW_qtPQ");
      // Term 2 always reads the unperturbed G^R from the precomputed cache.
      utils::check(!do_term2 || (dG_tsRPQ != nullptr && dG_mtau_tsRPQ != nullptr),
                   "lr_gw::_eval_sigma_Rspace: term 2 (G ⊙ ΔW) requires the G^R cache");

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
      // The PQ grid/block parameters are re-derived from dW_ref inside
      // _setup_workspace; here we only need the local PQ shape for the loop.
      dArr_4D_t& dW_ref = do_term1 ? *dW_tRPQ : *dDeltaW_qtPQ;
      auto dW_lshape = dW_ref.local_shape();
      auto NP_loc = dW_lshape[2], NQ_loc = dW_lshape[3];

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

      // Cross-check term 2 tau distribution and PQ tiling if both terms active
      if (do_term1 && do_term2) {
        utils::check(dDeltaW_qtPQ->local_shape()[1] == nt_loc,
                     "lr_gw: term1 nt_loc={} != term2 nt_loc={}", nt_loc, dDeltaW_qtPQ->local_shape()[1]);
        utils::check(dDeltaW_qtPQ->origin()[1] == t_origin,
                     "lr_gw: term1 t_origin={} != term2 t_origin={}", t_origin, dDeltaW_qtPQ->origin()[1]);
        utils::check(dDeltaW_qtPQ->local_shape()[2] == NP_loc &&
                     dDeltaW_qtPQ->local_shape()[3] == NQ_loc &&
                     dDeltaW_qtPQ->origin()[2] == dW_tRPQ->origin()[2] &&
                     dDeltaW_qtPQ->origin()[3] == dW_tRPQ->origin()[3],
                     "lr_gw: term2 PQ tiling mismatch vs term1 "
                     "(local ({},{}) origin ({},{}) vs local ({},{}) origin ({},{}))",
                     dDeltaW_qtPQ->local_shape()[2], dDeltaW_qtPQ->local_shape()[3],
                     dDeltaW_qtPQ->origin()[2], dDeltaW_qtPQ->origin()[3],
                     NP_loc, NQ_loc, dW_tRPQ->origin()[2], dW_tRPQ->origin()[3]);
      }

      app_log(3, "    nt_loc={}, nkpts={}, t_origin={}, NP_loc={}, NQ_loc={}",
              nt_loc, nkpts, t_origin, NP_loc, NQ_loc);

      // The G^R cache (it,s,R,P,Q) must share this call's τ-dist (hard check).
      if (do_term2) {
        long P_org_W = dW_ref.origin()[2], Q_org_W = dW_ref.origin()[3];
        std::array<long, 5> want_lshape = {nt_loc, ns, nkpts, NP_loc, NQ_loc};
        for (auto const* c : {dG_tsRPQ, dG_mtau_tsRPQ}) {
          utils::check(c->local_shape() == want_lshape && c->origin()[0] == t_origin &&
                       c->origin()[3] == P_org_W && c->origin()[4] == Q_org_W,
                       "lr_gw::_eval_sigma_Rspace: G^R cache layout mismatch vs τ-dist "
                       "(want local_shape (t,s,R,P,Q)=({},{},{},{},{}), origin t/P/Q={}/{}/{}).",
                       nt_loc, ns, nkpts, NP_loc, NQ_loc, t_origin, P_org_W, Q_org_W);
        }
      }

      // === Steps 0+1: tau subcommunicator, work arrays, FT coefficients ===
      // Built once and cached across SCF iterations (comm/node splits are
      // collective; shm windows and distributed buffers are expensive to
      // recreate). Kept out of the kernel for readability.
      _Timer.start("SIGMA_ALLOC");
      _setup_workspace(thc, dW_ref, do_term1, do_term2, ns, nk_ibz, nbnd);

      // Local aliases so the loop body below reads as before.
      // The Σ buffer is accumulated in R-space (dSigma_sRPQ) and FT'd in place
      // to k-space (aliased dSigma_skPQ) before aux_to_primary.
      auto& tau_comm = *_tau_comm;
      auto& dG_skPQ = *_dG_skPQ;
      auto& dSigma_sRPQ = *_dSigma_skPQ;
      auto& sSigma_skij = *_sSigma_skij;
      auto& ft_buffer = _ft_buffer;
      auto& W2_tau_RPQ = _W2_tau_RPQ;
      // gemm-path coefficients; engaged only when the FFT optionals are empty
      auto& opt_sf_Rk = _sf_Rk;
      auto& opt_sf_kR = _sf_kR;
      auto& opt_sf_qR = _sf_qR;
      _Timer.stop("SIGMA_ALLOC");

      // Hadamard product lambda
      auto neg_prod = nda::map([](ComplexType x, ComplexType y) { return -1.0 * (x * y); });

      // === Step 2: Per-tau loop with inner minus_t loop ===
      // Timed: this fence synchronizes all ranks sharing the ΔΣ window, so it
      // also absorbs any load imbalance from the preceding driver steps.
      _Timer.start("SIGMA_PRE_FENCE");
      sDeltaSigma_tskij.win().fence();
      _Timer.stop("SIGMA_PRE_FENCE");

      for (long it_local = 0; it_local < nt_loc; ++it_local) {
        long it_w = t_origin + it_local;  // global W tau index (first half: 0..nt_half-1)

        // --- W slices once per tau (reused for both minus_t passes) ---
        // Term 1's W (dW_tRPQ, (t,R,P,Q)) is a contiguous leading-index slice,
        // taken as a view inside the pass loop. Term 2's ΔW (dDeltaW_qtPQ,
        // (q,t,P,Q)) has strided t, so copy this τ-slice then q→R FT it.
        if (do_term2) {
          _Timer.start("SIGMA_W_COPY");
          for (long iq = 0; iq < nkpts; ++iq)
            W2_tau_RPQ(iq, nda::ellipsis{}) = dDeltaW_qtPQ->local()(iq, it_local, nda::ellipsis{});
          _Timer.stop("SIGMA_W_COPY");

          if (nkpts != 1) {
            _Timer.start("SIGMA_FT_R");
            auto W2_2D = nda::reshape(W2_tau_RPQ, shape_t<2>{nkpts, NP_loc * NQ_loc});
            if (_fft_q) {
              // Blocked FFT is safe in-place (block-sequential), no copy-back.
              _fft_q->k_to_R(W2_2D, W2_2D);
            } else {
              auto f_qR = opt_sf_qR->local();
              nda::blas::gemm(f_qR, W2_2D, ft_buffer);
              W2_2D = ft_buffer;
            }
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
          // The first active term assigns, the rest accumulate, so the buffer
          // needs no separate zeroing pass.
          bool first_term = true;

          // Term 1: ΔG ⊙ W_c
          if (do_term1) {
            _Timer.start("SIGMA_PRIM_TO_AUX");
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
              auto G_3D = nda::reshape(dG_skPQ.local(), shape_t<3>{ns, nkpts, NP_loc * NQ_loc});
              for (long s = 0; s < ns; ++s) {
                if (_fft_k) {
                  _fft_k->k_to_R(G_3D(s, nda::ellipsis{}), G_3D(s, nda::ellipsis{}));
                } else {
                  auto f_Rk = opt_sf_Rk->local();
                  nda::blas::gemm(f_Rk, G_3D(s, nda::ellipsis{}), ft_buffer);
                  G_3D(s, nda::ellipsis{}) = ft_buffer;
                }
              }
              _Timer.stop("SIGMA_FT_R");
            }

            _Timer.start("SIGMA_HADPROD_R");
            auto W1_tau_RPQ = dW_tRPQ->local()(it_local, nda::ellipsis{});
            for (long s = 0; s < ns; ++s) {
              auto G_RPQ = dG_skPQ.local()(s, nda::ellipsis{});
              auto Sigma_RPQ = dSigma_sRPQ.local()(s, nda::ellipsis{});
              if (first_term)
                Sigma_RPQ = neg_prod(G_RPQ, W1_tau_RPQ);
              else
                Sigma_RPQ += neg_prod(G_RPQ, W1_tau_RPQ);
            }
            first_term = false;
            _Timer.stop("SIGMA_HADPROD_R");
          }

          // Term 2: G ⊙ ΔW. The unperturbed G^R is read from the precomputed
          // cache (τ for the forward pass, β−τ for the backward pass), so the
          // per-τ Primary→Aux + k→R FT are skipped.
          if (do_term2) {
            auto& cache = minus_t ? *dG_mtau_tsRPQ : *dG_tsRPQ;
            _Timer.start("SIGMA_HADPROD_R");
            for (long s = 0; s < ns; ++s) {
              auto G_RPQ = cache.local()(it_local, s, nda::ellipsis{});
              auto Sigma_RPQ = dSigma_sRPQ.local()(s, nda::ellipsis{});
              if (first_term)
                Sigma_RPQ = neg_prod(G_RPQ, W2_tau_RPQ);
              else
                Sigma_RPQ += neg_prod(G_RPQ, W2_tau_RPQ);
            }
            first_term = false;
            _Timer.stop("SIGMA_HADPROD_R");
          }

          // --- R→k FFT on accumulated Sigma (once); R-space → k-space alias ---
          if (nkpts != 1) {
            _Timer.start("SIGMA_FT_R");
            auto Sigma_3D = nda::reshape(dSigma_sRPQ.local(), shape_t<3>{ns, nkpts, NP_loc * NQ_loc});
            for (long s = 0; s < ns; ++s) {
              if (_fft_k) {
                _fft_k->R_to_k(Sigma_3D(s, nda::ellipsis{}), Sigma_3D(s, nda::ellipsis{}));
              } else {
                auto f_kR = opt_sf_kR->local();
                nda::blas::gemm(f_kR, Sigma_3D(s, nda::ellipsis{}), ft_buffer);
                Sigma_3D(s, nda::ellipsis{}) = ft_buffer;
              }
            }
            _Timer.stop("SIGMA_FT_R");
          }
          auto& dSigma_skPQ = dSigma_sRPQ;  // now in k-space after the R→k FT

          // --- aux_to_primary on accumulated Sigma (once) ---
          _Timer.start("SIGMA_AUX_TO_PRIM");
          _Timer.start("SIGMA_A2P_SETZERO");
          sSigma_skij.set_zero();
          _Timer.stop("SIGMA_A2P_SETZERO");
          lr_thc_comm::aux_to_primary(0, 0, ComplexType(1.0), dSigma_skPQ,
                                      sSigma_skij, thc, MF->ks_to_k(0), _kpq_map,
                                      &_Timer);

          // Add precomputed IBC correction for this τ-point
          if (ibc && ibc->sDeltaSigma_ibc_tskij.has_value()) {
            if (sSigma_skij.node_comm()->root()) {
              sSigma_skij.local() += ibc->sDeltaSigma_ibc_tskij->local()(it_out, nda::ellipsis{});
            }
            sSigma_skij.win().fence();
          }

          _Timer.start("SIGMA_A2P_ACCUM");
          if (tau_comm.rank() == 0) {
            sDeltaSigma_tskij.local()(it_out, nda::ellipsis{}) += sSigma_skij.local();
          }
          _Timer.stop("SIGMA_A2P_ACCUM");
          _Timer.stop("SIGMA_AUX_TO_PRIM");
        } // pass
      } // it_local

      // Synchronize and distribute across all nodes
      _Timer.start("SIGMA_FINAL_REDUCE");
      sDeltaSigma_tskij.win().fence();
      // Every τ block is written by exactly one rank globally (tau_comm root,
      // disjoint τ ranges across pools), so the node-parallel reduction is
      // bit-identical to all_reduce() here.
      sDeltaSigma_tskij.all_reduce_parallel();
      _Timer.stop("SIGMA_FINAL_REDUCE");

      _Timer.stop("EVALUATE_SIGMA_R");
    }


    // =========================================================================
    // _sigma_div_correction: LR-aware divergence correction
    //   ΔΣ^div_{ij} = factor · S(k+q) · ΔG · S(k)
    //
    // sDeltaSigma_tskij is shared memory, so the full ΔΣ is already replicated
    // on every node. gw_t::Sigma_div_correction splits the work across all
    // ranks into a temp and recombines with an internode all_reduce; profiling
    // showed that all_reduce dominated here. Instead, each node recomputes the
    // full (t,s,k) set (cheap gemms) and accumulates in place — trading
    // redundant flops for zero internode comm.
    // =========================================================================
    void lr_gw::_sigma_div_correction(
        sArrv_5D_t& sDeltaSigma_tskij,
        const nda::array_view<ComplexType, 5>& DeltaG_tskij,
        const nda::array_view<ComplexType, 4>& S_skij,
        thc_reader_t& thc,
        const nda::array<ComplexType, 1>& eps_inv_head) {
      auto div_treatment = _gw.div_treatment();
      app_log(3, "  LR divergence correction: {}", div_treatment);

      if (thc.MF()->nqpts_ibz() == 1 and div_treatment != "ignore_g0") {
        app_log(3, "    nqpts_ibz == 1, taking div_treatment = ignore_g0");
        div_treatment = "ignore_g0";
      }

      if (div_treatment == "ignore_g0") {
        return;
      } else if (div_treatment.find("gygi") != std::string::npos) {
        auto MF = thc.MF();
        app_log(3, "    - madelung = {}", MF->madelung());
        auto [nts, ns, nkpts, nbnd, nbnd2] = sDeltaSigma_tskij.shape();
        auto nt_half = (nts % 2 == 0) ? nts / 2 : nts / 2 + 1;

        decltype(nda::range::all) all;

        // ΔΣ^div_{ij} += factor · S(k+q) · ΔG · S(k), accumulated in place.
        // Each node computes the full (t,s,k) set. The calculation is distributed
        // only over node_comm (single node). Avoid inter-node communication.
        auto node_comm = sDeltaSigma_tskij.node_comm();
        size_t node_rank = node_comm->rank();
        size_t node_size = node_comm->size();

        nda::array<ComplexType, 2> buffer_ib(nbnd, nbnd);
        auto Sigma_loc = sDeltaSigma_tskij.local();
        sDeltaSigma_tskij.win().fence();
        for (size_t tsk = node_rank; tsk < nts * ns * nkpts; tsk += node_size) {
          size_t it = tsk / (ns * nkpts);
          size_t is = (tsk / nkpts) % ns;
          size_t ik = tsk % nkpts;

          size_t it_pos = (it < nt_half) ? it : nts - it - 1;
          RealType factor = -1.0 * MF->madelung() * eps_inv_head(it_pos).real();

          auto DeltaG_ab = DeltaG_tskij(it, is, ik, nda::ellipsis{});
          auto Sigma_ij = Sigma_loc(it, is, ik, nda::ellipsis{});
          auto S_kq = S_skij(is, _kpq_map(ik), all, all);  // S at k+q
          auto S_k = S_skij(is, ik, all, all);              // S at k
          nda::blas::gemm(S_kq, DeltaG_ab, buffer_ib);
          nda::blas::gemm(ComplexType(factor), buffer_ib, S_k, ComplexType(1.0), Sigma_ij);
        }
        sDeltaSigma_tskij.win().fence();
        sDeltaSigma_tskij.communicator()->barrier();
      } else {
        utils::check(false, "Unsupported divergence treatment: {}", div_treatment);
      }
      app_log(3, "");
    }

  } // solvers
} // methods
