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
#include "methods/HF/thc_solver_comm.hpp"
#include "methods/scr_coulomb/lr_rpa_pi.hpp"

namespace methods {
  namespace solvers {

    lr_rpa_pi::lr_rpa_pi(nda::array<double, 1> const& q_pert)
      : _q_pert(q_pert),
        _Timer() {

      for (auto& v: {"EVALUATE_LR_PI", "PI_LR_ALLOC", "PI_LR_PRIM_TO_AUX",
                      "PI_LR_TREV_PHASE", "PI_LR_FT_R", "PI_LR_HADPROD"}) {
        _Timer.add(v);
      }
    }

    void lr_rpa_pi::_init_kpq_map(thc_reader_t& thc) {
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

    /**
     * LR Polarization: ΔΠ^q(τ) from ΔG and G (R-space Hadamard product).
     *
     * Adapted from scr_coulomb_t::eval_Pi_rpa_Rspace (rpa_pi.icc:22-180).
     *
     * Π(τ) = G(τ)G(-τ), so ΔΠ = ΔG·G(-) + G·ΔG(-).
     * We use G(β-τ) = -G(-τ), with the sign absorbed into sp_factor.
     * Two terms per τ point (p = q_pert):
     *
     * Term 1: ΔΠ^{(1),R}_{PQ}(τ) = ΔG^R_{PQ}(τ) · conj(G^R_{PQ}(β-τ))
     *   1. ΔG^k_{mn}(τ)   → ΔG^k_{PQ}     lr_thc_comm::primary_to_aux
     *   2. G^k_{mn}(β-τ)  → G^k_{PQ}      thc_solver_comm::primary_to_aux
     *   3. ΔG^k, G^k      → ΔG^R, G^R     k→R Fourier transform
     *   4. ΔG^R · conj(G^R) → ΔΠ^{(1),R}  Hadamard product (with conj)
     *
     * Term 2: ΔΠ^{(2),R}_{PQ}(τ) = G^R_{PQ}(τ) · e^{ip·R} [ΔG^{-R}(β-τ)]^T_{PQ}
     *   5. G^k_{mn}(τ)     → G^k_{PQ}     thc_solver_comm::primary_to_aux
     *   6. ΔG^k_{mn}(β-τ)  → [ΔG^k_{PQ}]^T  lr_thc_comm::primary_to_aux_transposed (dagger + conj)
     *   7. G^k → G^R (f_Rk),  ΔG^k → ΔG^{-R} (conj(f_Rk))   k→R / k→(-R) FT
     *   8. ΔG^{-R} → e^{ip·R} [ΔG^{-R}]^T                    transpose + phase
     *   9. G^R · phase·[ΔG^{-R}]^T → ΔΠ^{(2),R}              Hadamard (no conj)
     *
     * Combine:
     *  10. sp_factor · (ΔΠ^{(1)} + ΔΠ^{(2)}) → ΔΠ^q          R→q Fourier transform
     *
     * conj(G^R) in term 1 uses G^R Hermiticity: conj(G^R_{PQ}) = G_{QP}(-R).
     * Term 2 cannot use conj(ΔG) since ΔG is NOT Hermitian at q≠0;
     * instead explicitly computes ΔG(-R) and transposes PQ.
     *
     * lr_thc_comm handles ΔG (asymmetric X(k+q)/X(k)),
     * thc_solver_comm handles G (symmetric X(k)/X(k)).
     */
    dArr_4D_t lr_rpa_pi::evaluate_lr_Pi(
        const nda::array_view<ComplexType, 5>& G_tskij,
        const nda::array_view<ComplexType, 5>& DeltaG_tskij,
        thc_reader_t& thc)
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
      auto [tpools_v, _q, np_P, np_Q] = tau_pgrid;

      app_log(1, "\n  LR polarization: ΔΠ = ΔG·G + G·ΔG (R-space)");
      app_log(1, "    q_pert = ({:.6f}, {:.6f}, {:.6f})", _q_pert(0), _q_pert(1), _q_pert(2));
      app_log(2, "    processor grid for Pi: (t, q, P, Q) = ({}, {}, {}, {})",
                 tau_pgrid[0], tau_pgrid[1], tau_pgrid[2], tau_pgrid[3]);

      _Timer.start("PI_LR_ALLOC");
      auto dDeltaPi_tqPQ = make_distributed_array<local_Array_4D_t>(
          mpi->comm, tau_pgrid, {nt_half, nkpts, Np, Np}, tau_bsize);
      auto [t_origin, q_origin, P_origin, Q_origin] = dDeltaPi_tqPQ.origin();
      auto [nt_loc, nq_loc, NP_loc, NQ_loc] = dDeltaPi_tqPQ.local_shape();

      // FT matrices
      math::shm::shared_array<nda::array_view<ComplexType,2>> sf_Rk(*mpi, {nkpts,nkpts});
      utils::k_to_R_coefficients(mpi->comm, nda::range(nkpts), MF->kpts(), MF->lattv(), MF->kp_grid(), sf_Rk);
      auto f_Rk = sf_Rk.local();
      math::shm::shared_array<nda::array_view<ComplexType,2>> sf_qR(*mpi, {nkpts,nkpts});
      utils::R_to_k_coefficients(mpi->comm, nda::range(nkpts), MF->Qpts_ibz(), MF->lattv(), MF->kp_grid(), sf_qR);
      auto f_qR = sf_qR.local();

      // f_minus_Rk for k→(-R) transform in Term 2: conj(e^{-ikR}/Nk) = e^{+ikR}/Nk
      nda::matrix<ComplexType> f_minus_Rk = nda::conj(f_Rk);

      // phase_ipR(iR) = e^{ip·R} where p = _q_pert (crystal coords), R = integer lattice indices
      // p·R = 2π(p[0]*a + p[1]*b + p[2]*c) by biorthogonality of direct/reciprocal lattice
      nda::array<ComplexType, 1> phase_ipR(nkpts);
      {
        auto kp_grid = MF->kp_grid();
        long nx = kp_grid(0), ny = kp_grid(1), nz = kp_grid(2);
        for (long iR = 0; iR < nkpts; ++iR) {
          long a = iR / (ny * nz), b = (iR / nz) % ny, c = iR % nz;
          if (a > nx / 2) a -= nx;
          if (b > ny / 2) b -= ny;
          if (c > nz / 2) c -= nz;
          double pR = 2.0 * M_PI * (_q_pert(0) * a + _q_pert(1) * b + _q_pert(2) * c);
          phase_ipR(iR) = std::exp(ComplexType(0.0, pR));
        }
      }

      // t_intra communicator for rank-4 distributed buffers
      mpi3::communicator t_intra_comm = mpi->comm.split(t_origin, mpi->comm.rank());
      utils::check(t_intra_comm.size()==np_P*np_Q, "t_intra_comm.size() = {} != np_P*np_Q", t_intra_comm.size());

      // Buffers: DeltaG for lr_thc_comm (ΔG factor), G for thc_solver_comm (G factor)
      auto dDeltaG_skPQ = make_distributed_array<local_Array_4D_t>(
          t_intra_comm, {1, 1, np_P, np_Q}, {ns, nkpts, Np, Np},
          {1, 1, tau_bsize[2], tau_bsize[3]});
      auto dG_skPQ = make_distributed_array<local_Array_4D_t>(
          t_intra_comm, {1, 1, np_P, np_Q}, {ns, nkpts, Np, Np},
          {1, 1, tau_bsize[2], tau_bsize[3]});
      // Out-of-place FT buffers (replace single fft_work_PQ)
      nda::matrix<ComplexType> fft_out_A(nkpts, NP_loc*NQ_loc);
      nda::matrix<ComplexType> fft_out_B(nkpts, NP_loc*NQ_loc);
      _Timer.stop("PI_LR_ALLOC");

      auto DeltaG_skPQ_loc = dDeltaG_skPQ.local();
      auto G_skPQ_loc   = dG_skPQ.local();
      auto DeltaPi_tqPQ_loc  = dDeltaPi_tqPQ.local();

      auto had_prod2 = nda::map([](ComplexType x, ComplexType y) { return (x * y); });
      nda::array<ComplexType, 3> DeltaPi_RPQ(nkpts, NP_loc, NQ_loc);

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
        lr_thc_comm::primary_to_aux(0, 0, DeltaG_tskij(it_global, nda::ellipsis{}),
                                     dDeltaG_skPQ, thc,
                                     MF->kp_to_ibz(), MF->kp_trev(), _kpq_map);
        _Timer.stop("PI_LR_PRIM_TO_AUX");

        // === Step 2: G^k_{mn}(β-τ) → G^k_{PQ}(β-τ)  [thc_solver_comm::primary_to_aux] ===
        //     Input: G_tskij(β-τ), Output: dG_skPQ
        _Timer.start("PI_LR_PRIM_TO_AUX");
        thc_solver_comm::primary_to_aux(0, 0, G_tskij(it_beta_minus, nda::ellipsis{}),
                                         dG_skPQ, thc, MF->kp_to_ibz(), MF->kp_trev());
        _Timer.stop("PI_LR_PRIM_TO_AUX");

        // === Steps 3+4 fused: k→R FT (out-of-place) + Hadamard per spin ===
        //     ΔΠ^{(1),R}_{PQ}(τ) = Σ_s ΔG^R(s,τ) · conj(G^R(s,β-τ))
        if (nkpts != 1) {
          for (long is = 0; is < ns; ++is) {
            auto DeltaG_2D = nda::reshape(DeltaG_skPQ_loc(is, nda::ellipsis{}),
                                           shape_t<2>{nkpts, NP_loc * NQ_loc});
            auto G_2D = nda::reshape(G_skPQ_loc(is, nda::ellipsis{}),
                                         shape_t<2>{nkpts, NP_loc * NQ_loc});
            _Timer.start("PI_LR_FT_R");
            nda::blas::gemm(f_Rk, DeltaG_2D, fft_out_A);
            nda::blas::gemm(f_Rk, G_2D, fft_out_B);
            _Timer.stop("PI_LR_FT_R");

            _Timer.start("PI_LR_HADPROD");
            auto A_3D = nda::reshape(fft_out_A, shape_t<3>{nkpts, NP_loc, NQ_loc});
            auto B_3D = nda::reshape(fft_out_B, shape_t<3>{nkpts, NP_loc, NQ_loc});
            if (is == 0) DeltaPi_RPQ = had_prod2(A_3D, nda::conj(B_3D));
            else         DeltaPi_RPQ += had_prod2(A_3D, nda::conj(B_3D));
            _Timer.stop("PI_LR_HADPROD");
          }
        } else {
          _Timer.start("PI_LR_HADPROD");
          DeltaPi_RPQ = had_prod2(DeltaG_skPQ_loc(0, nda::ellipsis{}), nda::conj(G_skPQ_loc(0, nda::ellipsis{})));
          for (long s = 1; s < ns; ++s) {
            DeltaPi_RPQ += had_prod2(DeltaG_skPQ_loc(s, nda::ellipsis{}), nda::conj(G_skPQ_loc(s, nda::ellipsis{})));
          }
          _Timer.stop("PI_LR_HADPROD");
        }

        // ============================================
        // Term 2: ΔΠ^{(2),R} = G^R(τ) · e^{ip·R} [ΔG^{-R}(β-τ)]^T
        // [ΔG_PQ]^T computed directly via primary_to_aux_transposed.
        // ============================================

        // === Step 5: G^k_{mn}(τ) → G^k_{PQ}(τ)  [thc_solver_comm::primary_to_aux] ===
        //     Input: G_tskij(τ), Output: dG_skPQ
        _Timer.start("PI_LR_PRIM_TO_AUX");
        thc_solver_comm::primary_to_aux(0, 0, G_tskij(it_global, nda::ellipsis{}),
                                         dG_skPQ, thc, MF->kp_to_ibz(), MF->kp_trev());
        _Timer.stop("PI_LR_PRIM_TO_AUX");

        // === Step 6: ΔG^k_{mn}(β-τ) → [ΔG^k_{PQ}]^T directly ===
        //     Uses identity: [O_PQ]^T = conj(X_R @ O^H @ X_L^H)
        //     Dagger and output conjugation are handled inside primary_to_aux_transposed.
        _Timer.start("PI_LR_PRIM_TO_AUX");
        lr_thc_comm::primary_to_aux_transposed(0, 0, DeltaG_tskij(it_beta_minus, nda::ellipsis{}),
                                                dDeltaG_skPQ, thc,
                                                MF->kp_to_ibz(), MF->kp_trev(), _kpq_map);
        _Timer.stop("PI_LR_PRIM_TO_AUX");

        // === Steps 7+8+9 fused: k→R/(-R) FT (out-of-place) + phase + Hadamard per spin ===
        //     ΔΠ^{(2),R}(τ) += Σ_s G^R(s,τ) · e^{ip·R} [ΔG^{-R}(s,β-τ)]^T
        if (nkpts != 1) {
          for (long is = 0; is < ns; ++is) {
            auto G_2D = nda::reshape(G_skPQ_loc(is, nda::ellipsis{}),
                                         shape_t<2>{nkpts, NP_loc * NQ_loc});
            auto DeltaG_2D = nda::reshape(DeltaG_skPQ_loc(is, nda::ellipsis{}),
                                           shape_t<2>{nkpts, NP_loc * NQ_loc});
            _Timer.start("PI_LR_FT_R");
            nda::blas::gemm(f_Rk, G_2D, fft_out_A);
            nda::blas::gemm(f_minus_Rk, DeltaG_2D, fft_out_B);
            _Timer.stop("PI_LR_FT_R");

            // Phase multiply on fft_out_B: e^{ip·R} for each R
            _Timer.start("PI_LR_TREV_PHASE");
            auto B_3D = nda::reshape(fft_out_B, shape_t<3>{nkpts, NP_loc, NQ_loc});
            for (long iR = 0; iR < nkpts; ++iR) {
              B_3D(iR, nda::ellipsis{}) *= phase_ipR(iR);
            }
            _Timer.stop("PI_LR_TREV_PHASE");

            _Timer.start("PI_LR_HADPROD");
            auto A_3D = nda::reshape(fft_out_A, shape_t<3>{nkpts, NP_loc, NQ_loc});
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
          for (long s = 0; s < ns; ++s) {
            DeltaPi_RPQ += had_prod2(G_skPQ_loc(s, nda::ellipsis{}), DeltaG_skPQ_loc(s, nda::ellipsis{}));
          }
          _Timer.stop("PI_LR_HADPROD");
        }

        // === Step 10: sp_factor · (ΔΠ^{(1)} + ΔΠ^{(2)}) → ΔΠ^q  [R→q FT] ===
        //     Input: DeltaPi_RPQ. Output: DeltaPi_tqPQ_loc
        DeltaPi_RPQ *= sp_factor;

        if (nkpts != 1) {
          _Timer.start("PI_LR_FT_R");
          auto DeltaPi_tq_2D = nda::reshape(DeltaPi_tqPQ_loc(it, nda::ellipsis{}),
                                              shape_t<2>{nkpts, NP_loc * NQ_loc});
          auto DeltaPi_R_2D = nda::reshape(DeltaPi_RPQ, shape_t<2>{nkpts, NP_loc * NQ_loc});
          nda::blas::gemm(f_qR, DeltaPi_R_2D, DeltaPi_tq_2D);
          _Timer.stop("PI_LR_FT_R");
        } else {
          DeltaPi_tqPQ_loc(it, nda::ellipsis{}) = DeltaPi_RPQ;
        }
      } // it
      dDeltaPi_tqPQ.communicator()->barrier();

      _Timer.stop("EVALUATE_LR_PI");

      app_log(1, "  LR polarization done.\n");
      print_timers();

      return dDeltaPi_tqPQ;
    }

  } // solvers
} // methods
