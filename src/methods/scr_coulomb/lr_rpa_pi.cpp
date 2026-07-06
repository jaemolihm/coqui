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


#include <array>
#include <string>

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
#include "methods/ERI/detail/concepts.hpp"
#include "methods/ERI/thc_reader_t.hpp"
#include "methods/HF/lr_thc_comm.hpp"
#include "methods/scr_coulomb/lr_rpa_pi.hpp"

namespace methods {
  namespace solvers {

    lr_rpa_pi::lr_rpa_pi(nda::array<double, 1> const& q_pert)
      : _q_pert(q_pert),
        _Timer() {

      for (auto& v: {"EVALUATE_LR_PI", "PI_LR_ALLOC", "PI_LR_PRIM_TO_AUX",
                      "PI_LR_TREV_PHASE", "PI_LR_FT_R", "PI_LR_HADPROD",
                      "PI_LR_SCALE"}) {
        _Timer.add(v);
      }
    }

    void lr_rpa_pi::_init_kpq_map(thc_reader_t& thc) {
      if (_kpq_map_initialized) return;
      auto MF = thc.MF();
      long nkpts = MF->nkpts();
      auto kpts_crys = MF->kpts_crystal();
      _kpq_map.resize(nkpts);
      utils::calculate_kpq_map(kpts_crys, _q_pert, _kpq_map);
      _kpq_map_initialized = true;
    }

    void lr_rpa_pi::_setup_workspace(thc_reader_t& thc,
                                      dArr_4D_t const& dDeltaPi_tqPQ, long ns) {
      if (_setup_done) return;
      using local_Array_4D_t = nda::array<ComplexType, 4>;
      using math::nda::make_distributed_array;

      auto mpi = thc.mpi();
      auto MF = thc.MF();
      long nkpts = MF->nkpts();

      // τ-dist parameters read from the output array (single source of truth).
      auto [t_origin, q_origin, P_origin, Q_origin] = dDeltaPi_tqPQ.origin();
      auto [nt_loc, nq_loc, NP_loc, NQ_loc] = dDeltaPi_tqPQ.local_shape();
      long Np = dDeltaPi_tqPQ.global_shape()[2];
      auto tau_pgrid = dDeltaPi_tqPQ.grid();
      auto tau_bsize = dDeltaPi_tqPQ.block_size();
      long np_P = tau_pgrid[2], np_Q = tau_pgrid[3];

      // FT matrices
      _sf_Rk.emplace(*mpi, std::array<long, 2>{nkpts, nkpts});
      utils::k_to_R_coefficients(mpi->comm, nda::range(nkpts), MF->kpts(), MF->lattv(), MF->kp_grid(), *_sf_Rk);
      _sf_qR.emplace(*mpi, std::array<long, 2>{nkpts, nkpts});
      utils::R_to_k_coefficients(mpi->comm, nda::range(nkpts), MF->Qpts_ibz(), MF->lattv(), MF->kp_grid(), *_sf_qR);

      // f_minus_Rk for k→(-R) transform in Term 2: conj(e^{-ikR}/Nk) = e^{+ikR}/Nk
      _f_minus_Rk = nda::conj(_sf_Rk->local());

      // Blocked-FFT k<->R transforms (2-4x faster than the gemms above).
      if (nkpts != 1) {
        _fft_k.emplace(MF->kpts(), MF->lattv(), MF->kp_grid());
        _fft_q.emplace(MF->Qpts_ibz(), MF->lattv(), MF->kp_grid());
        app_log(3, "    k<->R transform: FFT");
      }

      // phase_ipR(iR) = e^{ip·R} where p = _q_pert (crystal coords), R = integer lattice indices
      // p·R = 2π(p[0]*a + p[1]*b + p[2]*c) by biorthogonality of direct/reciprocal lattice
      _phase_ipR.resize(nkpts);
      {
        auto kp_grid = MF->kp_grid();
        long nx = kp_grid(0), ny = kp_grid(1), nz = kp_grid(2);
        for (long iR = 0; iR < nkpts; ++iR) {
          long a = iR / (ny * nz), b = (iR / nz) % ny, c = iR % nz;
          if (a > nx / 2) a -= nx;
          if (b > ny / 2) b -= ny;
          if (c > nz / 2) c -= nz;
          double pR = 2.0 * M_PI * (_q_pert(0) * a + _q_pert(1) * b + _q_pert(2) * c);
          _phase_ipR(iR) = std::exp(ComplexType(0.0, pR));
        }
      }

      // t_intra communicator for rank-4 distributed buffers (collective split).
      _t_intra_comm.emplace(mpi->comm.split(t_origin, mpi->comm.rank()));
      utils::check(_t_intra_comm->size()==np_P*np_Q, "t_intra_comm.size() = {} != np_P*np_Q", _t_intra_comm->size());

      // Aux-basis buffer for the ΔG factor (lr_thc_comm). The unperturbed G
      // factor is supplied precomputed via the G^R cache, so no G buffer here.
      _dDeltaG_skPQ.emplace(make_distributed_array<local_Array_4D_t>(
          *_t_intra_comm, {1, 1, np_P, np_Q}, {ns, nkpts, Np, Np},
          {1, 1, tau_bsize[2], tau_bsize[3]}));
      // Out-of-place FT buffer (term 1 uses it for ΔG^R, term 2 for ΔG^{-R};
      // the two terms run sequentially, so one buffer suffices).
      _fft_out.resize(nkpts, NP_loc*NQ_loc);
      _DeltaPi_RPQ.resize(nkpts, NP_loc, NQ_loc);

      _setup_done = true;
    }

    /**
     * LR Polarization: ΔΠ^q(τ) from ΔG and G (R-space Hadamard product).
     *
     * Adapted from scr_coulomb_t::eval_Pi_rpa_Rspace (rpa_pi.icc:22-180).
     *
     * Π(τ) = G(τ)G(-τ), so ΔΠ = ΔG·G(-) + G·ΔG(-).
     * We use G(β-τ) = -G(-τ), with the sign absorbed into sp_factor.
     * The unperturbed G^R(τ)/G^R(β-τ) factors are supplied precomputed (in the
     * aux basis, from lr_precompute_G_R_pair); only ΔG is transformed per τ.
     * Two terms per τ point (p = q_pert):
     *
     * Term 1: ΔΠ^{(1),R}_{PQ}(τ) = ΔG^R_{PQ}(τ) · conj(G^R_{PQ}(β-τ))
     *   1. ΔG^k_{mn}(τ)   → ΔG^k_{PQ}     lr_thc_comm::primary_to_aux
     *   3. ΔG^k           → ΔG^R          k→R Fourier transform
     *   4. ΔG^R · conj(G^R(β-τ)) → ΔΠ^{(1),R}  Hadamard (G^R from cache, with conj)
     *
     * Term 2: ΔΠ^{(2),R}_{PQ}(τ) = G^R_{PQ}(τ) · e^{ip·R} [ΔG^{-R}(β-τ)]^T_{PQ}
     *   6. ΔG^k_{mn}(β-τ)  → [ΔG^k_{PQ}]^T  lr_thc_comm::primary_to_aux_transposed (dagger + conj)
     *   7. ΔG^k → ΔG^{-R} (conj(f_Rk))                       k→(-R) FT
     *   8. ΔG^{-R} → e^{ip·R} [ΔG^{-R}]^T                    transpose + phase
     *   9. G^R(τ) · phase·[ΔG^{-R}]^T → ΔΠ^{(2),R}           Hadamard (G^R from cache, no conj)
     *
     * Combine:
     *  10. sp_factor · (ΔΠ^{(1)} + ΔΠ^{(2)}) → ΔΠ^q          R→q Fourier transform
     *
     * conj(G^R) in term 1 uses G^R Hermiticity: conj(G^R_{PQ}) = G_{QP}(-R).
     * Term 2 cannot use conj(ΔG) since ΔG is NOT Hermitian at q≠0;
     * instead explicitly computes ΔG(-R) and transposes PQ.
     *
     * lr_thc_comm handles ΔG (asymmetric X(k+q)/X(k)).
     */
    dArr_4D_t lr_rpa_pi::evaluate_lr_Pi(
        const nda::array_view<ComplexType, 5>& G_tskij,
        const nda::array_view<ComplexType, 5>& DeltaG_tskij,
        thc_reader_t& thc,
        const dArr_5D_t& dG_tsRPQ,
        const dArr_5D_t& dG_mtau_tsRPQ,
        const lr_ibc_DeltaX* ibc)
    {
      using local_Array_4D_t = nda::array<ComplexType, 4>;
      using math::nda::make_distributed_array;
      _Timer.start("EVALUATE_LR_PI");
      _init_kpq_map(thc);

      auto mpi = thc.mpi();
      auto MF = thc.MF();

      long nkpts = MF->nkpts();

      long nt_f  = G_tskij.shape(0);
      long ns    = G_tskij.shape(1);
      long Np    = thc.Np();
      long nt_half = (nt_f % 2 == 0)? nt_f / 2 : nt_f / 2 + 1;
      double sp_factor = (ns == 2)? -1.0 : -2.0;

      utils::check(MF->nqpts() == MF->nqpts_ibz() and MF->nqpts() == MF->nkpts(),
                   "lr_rpa_pi::evaluate_lr_Pi: Symmetry not allowed. nqpts={}, nqpts_ibz={}, nkpts={}",
                   MF->nqpts(), MF->nqpts_ibz(), MF->nkpts());

      utils::check(G_tskij.shape() == DeltaG_tskij.shape(),
                   "lr_rpa_pi::evaluate_lr_Pi: G and DeltaG shape mismatch. G=({},{},{},{},{}), DeltaG=({},{},{},{},{})",
                   G_tskij.shape(0), G_tskij.shape(1), G_tskij.shape(2), G_tskij.shape(3), G_tskij.shape(4),
                   DeltaG_tskij.shape(0), DeltaG_tskij.shape(1), DeltaG_tskij.shape(2), DeltaG_tskij.shape(3), DeltaG_tskij.shape(4));

      // τ-dist: distribute over τ and PQ
      auto [tau_pgrid, tau_bsize] = utils::lr_W_q_local_dist(mpi->comm.size(), nt_half, Np);

      app_log(3, "\n  LR polarization: ΔΠ = ΔG·G + G·ΔG (R-space)");
      app_log(3, "    q_pert = ({:.6f}, {:.6f}, {:.6f})", _q_pert(0), _q_pert(1), _q_pert(2));
      app_log(3, "    processor grid for Pi: (t, q, P, Q) = ({}, {}, {}, {})",
                 tau_pgrid[0], tau_pgrid[1], tau_pgrid[2], tau_pgrid[3]);

      _Timer.start("PI_LR_ALLOC");
      auto dDeltaPi_tqPQ = make_distributed_array<local_Array_4D_t>(
          mpi->comm, tau_pgrid, {nt_half, nkpts, Np, Np}, tau_bsize);
      auto [t_origin, q_origin, P_origin, Q_origin] = dDeltaPi_tqPQ.origin();
      auto [nt_loc, nq_loc, NP_loc, NQ_loc] = dDeltaPi_tqPQ.local_shape();

      // One-time setup (comm split, shm FT-coefficient windows, phase array,
      // work buffers), built lazily on the first call and reused across SCF
      // iterations. Kept out of the kernel for readability.
      _setup_workspace(thc, dDeltaPi_tqPQ, ns);

      // Local aliases so the loop body below reads as before
      auto f_Rk = _sf_Rk->local();
      auto f_qR = _sf_qR->local();
      auto& f_minus_Rk = _f_minus_Rk;
      auto& phase_ipR = _phase_ipR;
      auto& dDeltaG_skPQ = *_dDeltaG_skPQ;
      auto& fft_out = _fft_out;
      auto& DeltaPi_RPQ = _DeltaPi_RPQ;
      _Timer.stop("PI_LR_ALLOC");

      auto DeltaG_skPQ_loc = dDeltaG_skPQ.local();
      auto DeltaPi_tqPQ_loc  = dDeltaPi_tqPQ.local();

      // Unperturbed G^R factors are supplied precomputed (constant across SCF
      // iterations; IBC-independent — corrections enter via ΔG): the per-τ G
      // Primary→Aux and k→R FT are skipped and the Hadamard products read the
      // cache slices directly. The cache must share Π's (it,s,R,P,Q) τ-dist.
      std::array<long, 5> want_lshape = {nt_loc, ns, nkpts, NP_loc, NQ_loc};
      for (auto const* c : {&dG_tsRPQ, &dG_mtau_tsRPQ}) {
        utils::check(c->local_shape() == want_lshape && c->origin()[0] == t_origin &&
                     c->origin()[3] == P_origin && c->origin()[4] == Q_origin,
                     "evaluate_lr_Pi: G^R cache layout mismatch vs Π τ-dist "
                     "(want local_shape (t,s,R,P,Q)=({},{},{},{},{}), origin t/P/Q={}/{}/{}).",
                     nt_loc, ns, nkpts, NP_loc, NQ_loc, t_origin, P_origin, Q_origin);
      }

      auto had_prod2 = nda::map([](ComplexType x, ComplexType y) { return (x * y); });

      // ============================================
      // Main loop over τ
      // ============================================
      for (long it = 0; it < nt_loc; ++it) {
        long it_global = it + t_origin;
        long it_beta_minus = nt_f - it_global - 1;

        // ============================================
        // Term 1: ΔΠ^{(1),R} = ΔG^R(τ) · conj(G^R(β-τ))
        // ============================================

        // === Step 1: ΔG^k_{mn}(τ) → ΔG^k_{PQ}(τ)  [lr_thc_comm::primary_to_aux] ===
        //     Input: DeltaG_tskij(τ), Output: dDeltaG_skPQ
        _Timer.start("PI_LR_PRIM_TO_AUX");
        if (ibc && ibc->sG_tskij) {
          auto G_tau_view = ibc->sG_tskij->local()(it_global, nda::ellipsis{});
          lr_thc_comm::primary_to_aux(0, 0, DeltaG_tskij(it_global, nda::ellipsis{}),
                                       dDeltaG_skPQ, thc,
                                       MF->kp_to_ibz(), MF->kp_trev(), _kpq_map,
                                       ibc, &G_tau_view);
        } else {
          lr_thc_comm::primary_to_aux(0, 0, DeltaG_tskij(it_global, nda::ellipsis{}),
                                       dDeltaG_skPQ, thc,
                                       MF->kp_to_ibz(), MF->kp_trev(), _kpq_map);
        }
        _Timer.stop("PI_LR_PRIM_TO_AUX");

        // === Steps 3+4 fused: k→R FT of ΔG + Hadamard with cached G^R(β-τ) ===
        //     ΔΠ^{(1),R}_{PQ}(τ) = Σ_s ΔG^R(s,τ) · conj(G^R(s,β-τ))
        if (nkpts != 1) {
          for (long is = 0; is < ns; ++is) {
            auto DeltaG_2D = nda::reshape(DeltaG_skPQ_loc(is, nda::ellipsis{}),
                                           shape_t<2>{nkpts, NP_loc * NQ_loc});
            _Timer.start("PI_LR_FT_R");
            if (_fft_k) _fft_k->k_to_R(DeltaG_2D, fft_out);
            else        nda::blas::gemm(f_Rk, DeltaG_2D, fft_out);
            _Timer.stop("PI_LR_FT_R");

            _Timer.start("PI_LR_HADPROD");
            auto A_3D = nda::reshape(fft_out, shape_t<3>{nkpts, NP_loc, NQ_loc});
            auto B_3D = dG_mtau_tsRPQ.local()(it, is, nda::ellipsis{});
            if (is == 0) DeltaPi_RPQ = had_prod2(A_3D, nda::conj(B_3D));
            else         DeltaPi_RPQ += had_prod2(A_3D, nda::conj(B_3D));
            _Timer.stop("PI_LR_HADPROD");
          }
        } else {
          _Timer.start("PI_LR_HADPROD");
          auto Gm_skPQ = dG_mtau_tsRPQ.local()(it, nda::ellipsis{});
          DeltaPi_RPQ = had_prod2(DeltaG_skPQ_loc(0, nda::ellipsis{}), nda::conj(Gm_skPQ(0, nda::ellipsis{})));
          for (long s = 1; s < ns; ++s) {
            DeltaPi_RPQ += had_prod2(DeltaG_skPQ_loc(s, nda::ellipsis{}), nda::conj(Gm_skPQ(s, nda::ellipsis{})));
          }
          _Timer.stop("PI_LR_HADPROD");
        }

        // ============================================
        // Term 2: ΔΠ^{(2),R} = G^R(τ) · e^{ip·R} [ΔG^{-R}(β-τ)]^T
        // [ΔG_PQ]^T computed directly via primary_to_aux_transposed.
        // ============================================

        // === Step 6: ΔG^k_{mn}(β-τ) → [ΔG^k_{PQ}]^T directly ===
        //     Uses identity: [O_PQ]^T = conj(X_R @ O^H @ X_L^H)
        //     Dagger and output conjugation are handled inside primary_to_aux_transposed.
        _Timer.start("PI_LR_PRIM_TO_AUX");
        if (ibc && ibc->sG_tskij) {
          auto G_beta_view = ibc->sG_tskij->local()(it_beta_minus, nda::ellipsis{});
          lr_thc_comm::primary_to_aux_transposed(0, 0, DeltaG_tskij(it_beta_minus, nda::ellipsis{}),
                                                  dDeltaG_skPQ, thc,
                                                  MF->kp_to_ibz(), MF->kp_trev(), _kpq_map,
                                                  ibc, &G_beta_view);
        } else {
          lr_thc_comm::primary_to_aux_transposed(0, 0, DeltaG_tskij(it_beta_minus, nda::ellipsis{}),
                                                  dDeltaG_skPQ, thc,
                                                  MF->kp_to_ibz(), MF->kp_trev(), _kpq_map);
        }
        _Timer.stop("PI_LR_PRIM_TO_AUX");

        // === Steps 7+8+9 fused: k→(-R) FT of ΔG + phase + Hadamard with cached G^R(τ) ===
        //     ΔΠ^{(2),R}(τ) += Σ_s G^R(s,τ) · e^{ip·R} [ΔG^{-R}(s,β-τ)]^T
        if (nkpts != 1) {
          for (long is = 0; is < ns; ++is) {
            auto DeltaG_2D = nda::reshape(DeltaG_skPQ_loc(is, nda::ellipsis{}),
                                           shape_t<2>{nkpts, NP_loc * NQ_loc});
            _Timer.start("PI_LR_FT_R");
            if (_fft_k) _fft_k->k_to_mR(DeltaG_2D, fft_out);
            else        nda::blas::gemm(f_minus_Rk, DeltaG_2D, fft_out);
            _Timer.stop("PI_LR_FT_R");

            // Phase multiply on fft_out: e^{ip·R} for each R
            _Timer.start("PI_LR_TREV_PHASE");
            auto B_3D = nda::reshape(fft_out, shape_t<3>{nkpts, NP_loc, NQ_loc});
            for (long iR = 0; iR < nkpts; ++iR) {
              B_3D(iR, nda::ellipsis{}) *= phase_ipR(iR);
            }
            _Timer.stop("PI_LR_TREV_PHASE");

            _Timer.start("PI_LR_HADPROD");
            auto A_3D = dG_tsRPQ.local()(it, is, nda::ellipsis{});
            DeltaPi_RPQ += had_prod2(A_3D, B_3D);
            _Timer.stop("PI_LR_HADPROD");
          }
        } else {
          // nkpts == 1: phase multiply in-place, then Hadamard
          _Timer.start("PI_LR_TREV_PHASE");
          for (long is = 0; is < ns; ++is) {
            for (long iR = 0; iR < nkpts; ++iR) {
              DeltaG_skPQ_loc(is, iR, nda::ellipsis{}) *= phase_ipR(iR);
            }
          }
          _Timer.stop("PI_LR_TREV_PHASE");

          _Timer.start("PI_LR_HADPROD");
          auto G_skPQ_cached = dG_tsRPQ.local()(it, nda::ellipsis{});
          for (long s = 0; s < ns; ++s) {
            DeltaPi_RPQ += had_prod2(G_skPQ_cached(s, nda::ellipsis{}), DeltaG_skPQ_loc(s, nda::ellipsis{}));
          }
          _Timer.stop("PI_LR_HADPROD");
        }

        // === Step 10: sp_factor · (ΔΠ^{(1)} + ΔΠ^{(2)}) → ΔΠ^q  [R→q FT] ===
        //     Input: DeltaPi_RPQ. Output: DeltaPi_tqPQ_loc
        _Timer.start("PI_LR_SCALE");
        DeltaPi_RPQ *= sp_factor;
        _Timer.stop("PI_LR_SCALE");

        if (nkpts != 1) {
          _Timer.start("PI_LR_FT_R");
          auto DeltaPi_tq_2D = nda::reshape(DeltaPi_tqPQ_loc(it, nda::ellipsis{}),
                                              shape_t<2>{nkpts, NP_loc * NQ_loc});
          auto DeltaPi_R_2D = nda::reshape(DeltaPi_RPQ, shape_t<2>{nkpts, NP_loc * NQ_loc});
          if (_fft_q) _fft_q->R_to_k(DeltaPi_R_2D, DeltaPi_tq_2D);
          else        nda::blas::gemm(f_qR, DeltaPi_R_2D, DeltaPi_tq_2D);
          _Timer.stop("PI_LR_FT_R");
        } else {
          DeltaPi_tqPQ_loc(it, nda::ellipsis{}) = DeltaPi_RPQ;
        }
      } // it

      _Timer.stop("EVALUATE_LR_PI");

      app_log(3, "  LR polarization done.\n");
      print_timers(3);

      return dDeltaPi_tqPQ;
    }

  } // solvers
} // methods
