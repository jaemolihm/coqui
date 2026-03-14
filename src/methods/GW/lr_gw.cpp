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
      nda::array<int, 1> kpq_map_int(nkpts);
      auto kpts_crys = MF->kpts_crystal();
      utils::calculate_kpq_map(kpts_crys, _q_pert, kpq_map_int);
      _kpq_map.resize(nkpts);
      for (long ik = 0; ik < nkpts; ++ik) _kpq_map(ik) = kpq_map_int(ik);
      _kpq_map_initialized = true;
    }

    // =========================================================================
    // evaluate_sigma_DeltaG: Term 1, ΔΣ = -ΔG ⊙ W_c
    // =========================================================================
    void lr_gw::evaluate_sigma_DeltaG(
        sArrv_5D_t& sDeltaSigma_tskij,
        const nda::array_view<ComplexType, 5>& DeltaG_tskij,
        dArr_4D_t& dW_tRPQ,
        thc_reader_t& thc) {

      _init_kpq_map(thc);

      // TODO: div_corr not yet implemented for LR GW (see lr_gw.hpp class docstring)
      app_log(1, "\n  LR-GW self-energy (term 1): ΔΣ = -ΔG ⊙ W_c (R-space, no div_corr)");
      app_log(1, "    q_pert = ({:.6f}, {:.6f}, {:.6f})", _q_pert(0), _q_pert(1), _q_pert(2));

      auto [tpools, Rpools, np_P, np_Q] = dW_tRPQ.grid();
      app_log(2, "    processor grid for W: (t, R, P, Q) = ({}, {}, {}, {})", tpools, Rpools, np_P, np_Q);

      sDeltaSigma_tskij.set_zero();

      // τ ∈ [0, β/2] — W is pre-transformed to R-space
      eval_DeltaSigma_Rspace(DeltaG_tskij, dW_tRPQ, sDeltaSigma_tskij, thc, false, _kpq_map, _kpq_map, true);
      // τ ∈ [β/2, β]
      eval_DeltaSigma_Rspace(DeltaG_tskij, dW_tRPQ, sDeltaSigma_tskij, thc, true, _kpq_map, _kpq_map, true);

      app_log(1, "  LR-GW self-energy (term 1) done.\n");
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

      // TODO: div_corr not yet implemented for LR GW (see lr_gw.hpp class docstring)
      app_log(1, "\n  LR-GW self-energy (term 2): ΔΣ = -G ⊙ ΔW (R-space, no div_corr)");
      app_log(1, "    q_pert = ({:.6f}, {:.6f}, {:.6f})", _q_pert(0), _q_pert(1), _q_pert(2));

      auto [qpools, tpools, np_P, np_Q] = dDeltaW_qtPQ.grid();
      app_log(2, "    processor grid for ΔW: (t, q, P, Q) = ({}, {}, {}, {})", tpools, qpools, np_P, np_Q);

      // G is unperturbed → identity kpq_map for primary_to_aux
      // ΔΣ has k+q structure → _kpq_map for aux_to_primary
      long nkpts = thc.MF()->nkpts();
      nda::array<long, 1> identity_map(nkpts);
      for (long ik = 0; ik < nkpts; ++ik) identity_map(ik) = ik;

      sDeltaSigma_tskij.set_zero();

      // τ ∈ [0, β/2] — ΔW is in q-space, needs q→R FT
      eval_DeltaSigma_Rspace(G_tskij, dDeltaW_qtPQ, sDeltaSigma_tskij, thc, false, identity_map, _kpq_map, false);
      // τ ∈ [β/2, β]
      eval_DeltaSigma_Rspace(G_tskij, dDeltaW_qtPQ, sDeltaSigma_tskij, thc, true, identity_map, _kpq_map, false);

      app_log(1, "  LR-GW self-energy (term 2) done.\n");
      print_timers();
    }


    // =========================================================================
    // eval_DeltaSigma_Rspace: per-tau R-space convolution workhorse
    // =========================================================================
    void lr_gw::eval_DeltaSigma_Rspace(
        const nda::array_view<ComplexType, 5>& G_tskij,
        dArr_4D_t& dW_tPQ,
        sArrv_5D_t& sDeltaSigma_tskij,
        thc_reader_t& thc,
        bool minus_t,
        const nda::array<long, 1>& kpq_map_G,
        const nda::array<long, 1>& kpq_map_Sigma,
        bool W_is_Rspace) {
      _Timer.start("EVALUATE_SIGMA_R");
      auto MF = thc.MF();
      auto mpi = thc.mpi();
      utils::check(MF->nqpts() == MF->nqpts_ibz() and MF->nqpts() == MF->nkpts(),
                   "lr_gw::eval_DeltaSigma_Rspace: Symmetry not allowed. nqpts={}, nqpts_ibz={}, nkpts={}",
                   MF->nqpts(), MF->nqpts_ibz(), MF->nkpts());
      auto ns = G_tskij.shape(1);
      auto nt = G_tskij.shape(0);
      auto nk_ibz = G_tskij.shape(2);
      auto nbnd = G_tskij.shape(3);

      // Extract layout-dependent quantities from dW.
      // W_is_Rspace=true:  dW has (t,R,P,Q) layout, pgrid (tpools,1,np_P,np_Q)
      // W_is_Rspace=false: dW has (q,t,P,Q) layout, pgrid (1,tpools,np_P,np_Q)
      // Axes 2,3 (P,Q) are layout-independent.
      auto dW_lshape = dW_tPQ.local_shape();
      auto dW_gshape = dW_tPQ.global_shape();
      auto dW_grid = dW_tPQ.grid();
      auto dW_bsize = dW_tPQ.block_size();

      auto NP_loc = dW_lshape[2], NQ_loc = dW_lshape[3];
      auto NP = dW_gshape[2], NQ = dW_gshape[3];
      auto np_P = dW_grid[2], np_Q = dW_grid[3];
      auto P_bs = dW_bsize[2], Q_bs = dW_bsize[3];

      // Axes 0,1 swap meaning depending on layout
      long nt_loc, nkpts, t_origin;
      if (W_is_Rspace) {
        // (t,R,P,Q): axis 0 = t (distributed), axis 1 = R (undivided)
        nt_loc = dW_lshape[0]; nkpts = dW_gshape[1];
        t_origin = dW_tPQ.origin()[0];
        utils::check(dW_lshape[1] == nkpts,
                     "lr_gw::eval_DeltaSigma_Rspace: nR_loc = {} != nkpts", dW_lshape[1]);
        utils::check(dW_grid[1] == 1,
                     "lr_gw::eval_DeltaSigma_Rspace: R-axis must be undivided, pgrid[1] = {}", dW_grid[1]);
      } else {
        // (q,t,P,Q): axis 0 = q (undivided), axis 1 = t (distributed)
        nkpts = dW_lshape[0]; nt_loc = dW_lshape[1];
        t_origin = dW_tPQ.origin()[1];
        utils::check(dW_lshape[0] == nkpts,
                     "lr_gw::eval_DeltaSigma_Rspace: nq_loc = {} != nkpts", dW_lshape[0]);
        utils::check(dW_grid[0] == 1,
                     "lr_gw::eval_DeltaSigma_Rspace: q-axis must be undivided, pgrid[0] = {}", dW_grid[0]);
      }

      // === Step 0: Create tau subcommunicator and allocate rank-4 work arrays ===
      _Timer.start("SIGMA_ALLOC");

      // Tau subcommunicator: procs sharing the same W tau slices form a group
      mpi3::communicator tau_comm = mpi->comm.split(t_origin, mpi->comm.rank());
      auto tau_mpi = utils::make_mpi_context(tau_comm);

      // Rank-4 distributed arrays on tau subcommunicator: (ns, nkpts, NP, NQ)
      using local_Array_4D_t = memory::array<HOST_MEMORY, ComplexType, 4>;
      using math::nda::make_distributed_array;

      auto dG_skPQ = make_distributed_array<local_Array_4D_t>(
          tau_comm, {1, 1, np_P, np_Q}, {ns, nkpts, NP, NQ},
          {1, 1, P_bs, Q_bs});

      auto dSigma_skPQ = make_distributed_array<local_Array_4D_t>(
          tau_comm, {1, 1, np_P, np_Q}, {ns, nkpts, NP, NQ},
          {1, 1, P_bs, Q_bs});

      // Shared array for aux_to_primary output (rank-4)
      using Arrv_4D_t = nda::array_view<ComplexType, 4>;
      auto sSigma_skij = math::shm::make_shared_array<Arrv_4D_t>(
          tau_mpi, {ns, nk_ibz, nbnd, nbnd});

      // FT buffer and W tau-slice buffer
      nda::matrix<ComplexType> ft_buffer(nkpts, NP_loc * NQ_loc);
      nda::array<ComplexType, 3> W_tau_RPQ(nkpts, NP_loc, NQ_loc);

      _Timer.stop("SIGMA_ALLOC");

      // === Step 1: Compute FT coefficients (once, outside tau loop) ===
      using sArray_2D_t = math::shm::shared_array<nda::array_view<ComplexType, 2>>;
      sArray_2D_t sf_Rk(tau_mpi, {nkpts, nkpts});
      sArray_2D_t sf_kR(tau_mpi, {nkpts, nkpts});

      if (nkpts != 1) {
        // f_Rk: G k→R (using kpts)
        utils::k_to_R_coefficients(tau_comm, nda::range(nkpts), MF->kpts(), MF->lattv(), MF->kp_grid(), sf_Rk);
        // f_kR: Sigma R→k (using kpts)
        utils::R_to_k_coefficients(tau_comm, nda::range(nkpts), MF->kpts(), MF->lattv(), MF->kp_grid(), sf_kR);
      }
      // f_qR for W q→R: only needed when W is still in q-space
      std::optional<sArray_2D_t> opt_sf_qR;
      if (nkpts != 1 && !W_is_Rspace) {
        opt_sf_qR.emplace(tau_mpi, std::array<long, 2>{nkpts, nkpts});
        utils::k_to_R_coefficients(tau_comm, nda::range(nkpts), MF->Qpts(), MF->lattv(), MF->kp_grid(), *opt_sf_qR);
      }

      // === Step 2: Per-tau loop ===
      sDeltaSigma_tskij.win().fence();

      for (long it_local = 0; it_local < nt_loc; ++it_local) {
        long it_w = t_origin + it_local;  // global tau index in W (first half: 0..nt_half-1)
        long it_g = minus_t ? (nt - 1 - it_w) : it_w;     // G input tau index
        long it_out = minus_t ? (nt - 1 - it_w) : it_w;    // output tau index

        // For minus_t, skip midpoint (already written by minus_t=false call)
        if (minus_t && it_w >= static_cast<long>(nt / 2)) continue;

        // --- 2a. primary_to_aux: G_skij(it_g) → dG_skPQ ---
        _Timer.start("SIGMA_PRIM_TO_AUX");
        dG_skPQ.local() = 0;
        auto G_skij_slice = G_tskij(it_g, nda::ellipsis{});
        lr_thc_comm::primary_to_aux(0, 0, G_skij_slice, dG_skPQ, thc,
                                    MF->kp_to_ibz(), MF->kp_trev(), kpq_map_G);
        _Timer.stop("SIGMA_PRIM_TO_AUX");

        // --- 2b. Copy W tau slice to contiguous buffer ---
        if (W_is_Rspace) {
          // dW layout: (t,R,P,Q) — W is already in R-space
          for (long iR = 0; iR < nkpts; ++iR)
            W_tau_RPQ(iR, nda::ellipsis{}) = dW_tPQ.local()(it_local, iR, nda::ellipsis{});
        } else {
          // dW layout: (q,t,P,Q) — W still in q-space
          for (long iq = 0; iq < nkpts; ++iq)
            W_tau_RPQ(iq, nda::ellipsis{}) = dW_tPQ.local()(iq, it_local, nda::ellipsis{});
        }

        // --- 2c. k→R FFT on G ---
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

        // --- 2d. W q→R FT (only when W is still in q-space) ---
        if (nkpts != 1 && !W_is_Rspace) {
          _Timer.start("SIGMA_FT_R");
          auto f_qR = opt_sf_qR->local();
          auto W_tau_2D = nda::reshape(W_tau_RPQ, shape_t<2>{nkpts, NP_loc * NQ_loc});
          nda::blas::gemm(f_qR, W_tau_2D, ft_buffer);
          W_tau_2D = ft_buffer;
          _Timer.stop("SIGMA_FT_R");
        }
        // After FT: dG_skPQ and W_tau_RPQ are both in R-space
        auto& dG_tau_sRPQ = dG_skPQ;  // alias: k-axis is now R-space

        // --- 2e. Hadamard: dSigma(s,R,P,Q) = -G(s,R,P,Q) * W_tau_RPQ(R,P,Q) ---
        _Timer.start("SIGMA_HADPROD_R");
        auto had_prod2 = nda::map([](ComplexType x, ComplexType y) { return -1.0 * (x * y); });
        for (long s = 0; s < ns; ++s) {
          auto G_RPQ = dG_tau_sRPQ.local()(s, nda::ellipsis{});
          auto Sigma_RPQ = dSigma_skPQ.local()(s, nda::ellipsis{});
          Sigma_RPQ = had_prod2(G_RPQ, W_tau_RPQ);
        }
        _Timer.stop("SIGMA_HADPROD_R");

        // --- 2f. R→k FFT on Sigma ---
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

        // --- 2g. aux_to_primary: dSigma_skPQ → sSigma_skij ---
        _Timer.start("SIGMA_AUX_TO_PRIM");
        sSigma_skij.set_zero();
        lr_thc_comm::aux_to_primary(0, 0, ComplexType(1.0), dSigma_skPQ,
                                    sSigma_skij, thc, MF->ks_to_k(0), kpq_map_Sigma);

        // --- 2h. Copy to output at the correct tau index ---
        if (tau_comm.rank() == 0) {
          sDeltaSigma_tskij.local()(it_out, nda::ellipsis{}) += sSigma_skij.local();
        }
        _Timer.stop("SIGMA_AUX_TO_PRIM");
      }

      // Synchronize and distribute across all nodes
      sDeltaSigma_tskij.win().fence();
      sDeltaSigma_tskij.all_reduce();

      _Timer.stop("EVALUATE_SIGMA_R");
    }


    // =========================================================================
    // _sigma_div_correction: LR-aware divergence correction
    // =========================================================================
    void lr_gw::_sigma_div_correction(
        sArrv_5D_t& sDeltaSigma_tskij,
        const nda::array_view<ComplexType, 5>& DeltaG_tskij,
        thc_reader_t& thc,
        const nda::array<ComplexType, 1>& eps_inv_head) {
      auto div_treatment = _gw.div_treatmemnt();
      app_log(1, "  LR divergence correction: {}", div_enum_to_string(div_treatment));

      if (thc.MF()->nqpts_ibz() == 1 and div_treatment != ignore_g0) {
        app_log(2, "    nqpts_ibz == 1, taking div_treatment = ignore_g0");
        div_treatment = ignore_g0;
      }

      if (div_treatment == ignore_g0) {
        return;
      } else if (div_treatment == gygi or div_treatment == gygi_extrplt or div_treatment == gygi_extrplt_2d) {
        auto MF = thc.MF();
        auto mpi = thc.mpi();
        app_log(1, "    - madelung = {}", MF->madelung());
        using math::shm::make_shared_array;
        auto [nts, ns, nkpts, nbnd, nbnd2] = sDeltaSigma_tskij.shape();
        auto nt_half = (nts % 2 == 0) ? nts / 2 : nts / 2 + 1;
        auto NP = thc.Np();

        int rank = mpi->comm.rank();
        int size = mpi->comm.size();

        using Array_view_4D_t = nda::array_view<ComplexType, 4>;
        using Array_view_5D_t = nda::array_view<ComplexType, 5>;
        auto chi_head = thc.basis_head()(0, nda::range::all);

        // T^L_{ia}(k) = Σ_P conj(X^{k+q}_{Pi}) · X^{k+q}_{Pa} · conj(χ_P)
        auto TL_skia = make_shared_array<Array_view_4D_t>(*mpi, {ns, nkpts, nbnd, nbnd});
        auto TL_loc = TL_skia.local();
        TL_skia.win().fence();
        for (size_t skia = rank; skia < ns * nkpts * nbnd * nbnd; skia += size) {
          size_t is = skia / (nkpts * nbnd * nbnd);
          size_t ik = (skia / (nbnd * nbnd)) % nkpts;
          size_t i = (skia / nbnd) % nbnd;
          size_t a = skia % nbnd;
          auto Xsk = thc.X(is, 0, _kpq_map(ik));  // X at k+q_pert
          for (size_t P = 0; P < NP; ++P) {
            TL_loc(is, ik, i, a) += nda::conj(Xsk(P, i)) * Xsk(P, a) * nda::conj(chi_head(P));
          }
        }
        TL_skia.win().fence();
        TL_skia.all_reduce();

        // T^R_{jb}(k) = Σ_P conj(X^k_{Pj}) · X^k_{Pb} · conj(χ_P)
        auto TR_skia = make_shared_array<Array_view_4D_t>(*mpi, {ns, nkpts, nbnd, nbnd});
        auto TR_loc = TR_skia.local();
        TR_skia.win().fence();
        for (size_t skia = rank; skia < ns * nkpts * nbnd * nbnd; skia += size) {
          size_t is = skia / (nkpts * nbnd * nbnd);
          size_t ik = (skia / (nbnd * nbnd)) % nkpts;
          size_t i = (skia / nbnd) % nbnd;
          size_t a = skia % nbnd;
          auto Xsk = thc.X(is, 0, ik);  // X at k
          for (size_t P = 0; P < NP; ++P) {
            TR_loc(is, ik, i, a) += nda::conj(Xsk(P, i)) * Xsk(P, a) * nda::conj(chi_head(P));
          }
        }
        TR_skia.win().fence();
        TR_skia.all_reduce();

        // ΔΣ^div_{ij} = factor · T^L · ΔG · (T^R)†
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

          auto G_ab = DeltaG_tskij(it, is, ik, nda::ellipsis{});
          auto Delta_ij = Delta_loc(it, is, ik, nda::ellipsis{});
          auto TL_ia = TL_loc(is, ik, nda::ellipsis{});
          nda::blas::gemm(TL_ia, G_ab, buffer_ib);
          nda::blas::gemm(ComplexType(factor), buffer_ib, nda::dagger(TR_loc(is, ik, nda::ellipsis{})),
                     ComplexType(0.0), Delta_ij);
        }
        sDelta_tskij.win().fence();
        sDelta_tskij.all_reduce();

        if (sDeltaSigma_tskij.node_comm()->root())
          sDeltaSigma_tskij.local() += sDelta_tskij.local();
        sDeltaSigma_tskij.communicator()->barrier();
      } else {
        utils::check(false, "Unsupported divergence treatment: {}", div_enum_to_string(div_treatment));
      }
      app_log(1, "");
    }

  } // solvers
} // methods
