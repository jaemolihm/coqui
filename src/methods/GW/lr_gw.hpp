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


#ifndef COQUI_LR_GW_HPP
#define COQUI_LR_GW_HPP

#include "nda/nda.hpp"
#include "numerics/distributed_array/nda.hpp"
#include "numerics/shared_array/nda.hpp"

#include "utilities/mpi_context.h"
#include "utilities/Timer.hpp"

#include "mean_field/MF.hpp"
#include "numerics/imag_axes_ft/IAFT.hpp"
#include "methods/ERI/detail/concepts.hpp"
#include "methods/ERI/div_treatment_e.hpp"
#include "methods/SCF/lr_ibc.hpp"
#include "methods/GW/gw_t.h"

namespace methods {

  class thc_reader_t;  // forward declaration

  namespace solvers {
    using namespace memory;

    using sArrv_5D_t = math::shm::shared_array<nda::array_view<ComplexType, 5>>;
    using dArr_4D_t = memory::darray_t<nda::array<ComplexType, 4>, mpi3::communicator>;

    /**
     * Linear response GW self-energy: ΔΣ = -ΔG ⊙ W_c - G ⊙ ΔW.
     *
     * W_c = W - v is the correlation part of the screened interaction
     * (bare Coulomb v subtracted).
     *
     * Two terms:
     *   Term 1 (evaluate_sigma_DeltaG): -ΔG ⊙ W_c using lr_thc_comm
     *     with kpq_map for both primary_to_aux (ΔG) and aux_to_primary (ΔΣ).
     *   Term 2 (evaluate_sigma_DeltaW): -G ⊙ ΔW using lr_thc_comm
     *     with identity kpq_map for primary_to_aux (G is unperturbed)
     *     and _kpq_map for aux_to_primary (ΔΣ has k+q structure).
     *
     * Divergence correction is not yet implemented for LR GW. The q→0
     * divergence is 1/q² (same as standard GW), but the correction requires
     * Δε⁻¹_head which is not yet computed. Both terms are currently computed
     * without div_corr; for FD testing, the reference Σ must also be computed
     * without div_corr for consistency.
     *
     * Naming convention:
     *   q_pert = LR perturbation wavevector (constructor param).
     *   The GW convolution variable (summed over internally) is separate.
     *
     * Requires nkpts == nkpts_ibz (no symmetry).
     */
    class lr_gw {
    public:
      template<int N>
      using shape_t = std::array<long, N>;

    public:
      lr_gw(const imag_axes_ft::IAFT *ft,
            nda::array<double, 1> const& q_pert,
            div_treatment_e div = gygi);

      ~lr_gw() {}

      /**
       * Both terms fused: ΔΣ = -ΔG ⊙ W_c - G ⊙ ΔW
       *
       * Computes both terms in a single pass over tau, fusing the R→k FFT
       * and aux_to_primary on Sigma (done once per tau instead of twice).
       *
       * @param sDeltaSigma_tskij - [OUTPUT] LR self-energy (nts, ns, nkpts_ibz, nbnd, nbnd)
       * @param DeltaG_tskij      - [INPUT] LR Green's function (nts, ns, nkpts_ibz, nbnd, nbnd)
       * @param dW_tRPQ           - [INPUT] W_c in THC+R-space (nt_half, nR, NP, NP)
       *   Pre-transformed to R-space by lr_precompute_W_tRPQ.
       *   Distribution: pgrid = (tpools, 1, np_P, np_Q).
       * @param G_tskij           - [INPUT] Unperturbed Green's function (nts, ns, nkpts_ibz, nbnd, nbnd)
       * @param dDeltaW_qtPQ      - [INPUT] ΔW in THC+q-space (nkpts, nt_half, NP, NP)
       *   Axis order: (q, t, P, Q).
       *   Distribution: pgrid = (1, tpools, np_P, np_Q).
       * @param thc               - [INPUT] THC-ERI handler
       */
      void evaluate_sigma(
          sArrv_5D_t& sDeltaSigma_tskij,
          const nda::array_view<ComplexType, 5>& DeltaG_tskij,
          dArr_4D_t& dW_tRPQ,
          const nda::array_view<ComplexType, 5>& G_tskij,
          dArr_4D_t& dDeltaW_qtPQ,
          thc_reader_t& thc,
          const lr_ibc_DeltaX* ibc = nullptr);

      /**
       * Term 1 only: ΔΣ = -ΔG ⊙ W_c (R-space)
       *
       * @param sDeltaSigma_tskij - [OUTPUT] LR self-energy (nts, ns, nkpts_ibz, nbnd, nbnd)
       * @param DeltaG_tskij      - [INPUT] LR Green's function (nts, ns, nkpts_ibz, nbnd, nbnd)
       * @param dW_tRPQ           - [INPUT] W_c in THC+R-space (nt_half, nR, NP, NP)
       *   Pre-transformed to R-space by lr_precompute_W_tRPQ.
       *   Distribution: pgrid = (tpools, 1, np_P, np_Q).
       * @param thc               - [INPUT] THC-ERI handler
       */
      void evaluate_sigma_DeltaG(
          sArrv_5D_t& sDeltaSigma_tskij,
          const nda::array_view<ComplexType, 5>& DeltaG_tskij,
          dArr_4D_t& dW_tRPQ,
          thc_reader_t& thc,
          const lr_ibc_DeltaX* ibc = nullptr);

      /**
       * Term 2 only: ΔΣ = -G ⊙ ΔW
       *
       * @param sDeltaSigma_tskij - [OUTPUT] LR self-energy (nts, ns, nkpts_ibz, nbnd, nbnd)
       * @param G_tskij           - [INPUT] Unperturbed Green's function (nts, ns, nkpts_ibz, nbnd, nbnd)
       * @param dDeltaW_qtPQ      - [INPUT] ΔW in THC+q-space (nkpts, nt_half, NP, NP)
       *   Axis order: (q, t, P, Q).
       *   Distribution: pgrid = (1, tpools, np_P, np_Q).
       * @param thc               - [INPUT] THC-ERI handler
       */
      void evaluate_sigma_DeltaW(
          sArrv_5D_t& sDeltaSigma_tskij,
          const nda::array_view<ComplexType, 5>& G_tskij,
          dArr_4D_t& dDeltaW_qtPQ,
          thc_reader_t& thc);

    inline void print_timers() {
      app_log(2, "\n  LR_GW timers");
      app_log(2, "  -------------");
      if (_Timer.has("EVALUATE_SIGMA_R")) {
        app_log(2, "    LR Sigma (R-space):             {0:8.3f} sec  {1:4d} calls", _Timer.elapsed("EVALUATE_SIGMA_R"), _Timer.number_of_calls("EVALUATE_SIGMA_R"));
        app_log(2, "      - Alloc:                      {0:8.3f} sec  {1:4d} calls", _Timer.elapsed("SIGMA_ALLOC"), _Timer.number_of_calls("SIGMA_ALLOC"));
        app_log(2, "      - Primary->Aux:               {0:8.3f} sec  {1:4d} calls", _Timer.elapsed("SIGMA_PRIM_TO_AUX"), _Timer.number_of_calls("SIGMA_PRIM_TO_AUX"));
        app_log(2, "      - FT (k<->R):                 {0:8.3f} sec  {1:4d} calls", _Timer.elapsed("SIGMA_FT_R"), _Timer.number_of_calls("SIGMA_FT_R"));
        app_log(2, "      - Hadamard product:           {0:8.3f} sec  {1:4d} calls", _Timer.elapsed("SIGMA_HADPROD_R"), _Timer.number_of_calls("SIGMA_HADPROD_R"));
        app_log(2, "      - Aux->Primary:               {0:8.3f} sec  {1:4d} calls", _Timer.elapsed("SIGMA_AUX_TO_PRIM"), _Timer.number_of_calls("SIGMA_AUX_TO_PRIM"));
      }
      app_log(2, "");
    }

    private:
      nda::array<double, 1> _q_pert;
      gw_t _gw;                       // stores div_treatment settings (friend access)
      bool _kpq_map_initialized = false;
      nda::array<long, 1> _kpq_map;  // k → k+q_pert mapping (full BZ); identity for q_pert=0
      utils::TimerManager _Timer;

      /// Initialize _kpq_map from THC's mean-field k-points (lazy, once)
      void _init_kpq_map(thc_reader_t& thc);

      /**
       * R-space convolution workhorse: ΔΣ = -ΔG⊙W - G⊙ΔW.
       *
       * Handles all three modes via null/non-null pointers:
       *   Term 1 only: DeltaG + dW non-null, G + dDeltaW null
       *   Term 2 only: G + dDeltaW non-null, DeltaG + dW null
       *   Both fused:  all non-null
       *
       * Merges both τ halves (forward + backward) in a single call.
       * Term 1's W is always R-space (t,R,P,Q); term 2's ΔW is always
       * q-space (q,t,P,Q) with per-tau q→R FT.
       *
       * @param DeltaG_tskij   - [INPUT] LR Green's function (term 1), or nullptr
       * @param dW_tRPQ        - [INPUT] W_c in R-space (t,R,P,Q), or nullptr
       * @param G_tskij        - [INPUT] Unperturbed Green's function (term 2), or nullptr
       * @param dDeltaW_qtPQ   - [INPUT] ΔW in q-space (q,t,P,Q), or nullptr
       */
      void _eval_sigma_Rspace(
          sArrv_5D_t& sDeltaSigma_tskij,
          thc_reader_t& thc,
          const nda::array_view<ComplexType, 5>* DeltaG_tskij,
          dArr_4D_t* dW_tRPQ,
          const nda::array_view<ComplexType, 5>* G_tskij,
          dArr_4D_t* dDeltaW_qtPQ,
          const lr_ibc_DeltaX* ibc = nullptr);

      /**
       * LR-aware divergence correction: ΔΣ^div = factor · T^L · ΔG · (T^R)†
       *
       * Adapted from gw_t::Sigma_div_correction (thc_gw.icc:445-526).
       * For q_pert≠0, the left and right T matrices use different X:
       *   T^L_{ia}(k) = Σ_P conj(X^{k+q}_{Pi}) · X^{k+q}_{Pa} · conj(χ_P)
       *   T^R_{jb}(k) = Σ_P conj(X^k_{Pj})     · X^k_{Pb}     · conj(χ_P)
       * Standard Sigma_div_correction uses T^L = T^R = T(k), which is only
       * correct at q_pert=0.
       */
      void _sigma_div_correction(
          sArrv_5D_t& sDeltaSigma_tskij,
          const nda::array_view<ComplexType, 5>& DeltaG_tskij,
          thc_reader_t& thc,
          const nda::array<ComplexType, 1>& eps_inv_head);

    }; // lr_gw
  } // solvers
} // methods

#endif //COQUI_LR_GW_HPP
