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
#include "methods/GW/gw_t.h"

namespace methods {
  namespace solvers {
    using namespace memory;

    /**
     * Linear response GW self-energy: ΔΣ = -ΔG ⊙ W_c - G ⊙ ΔW.
     *
     * W_c = W - v is the correlation part of the screened interaction
     * (bare Coulomb v subtracted).
     *
     * Two terms:
     *   Term 1 (evaluate_sigma_DeltaG): -ΔG ⊙ W_c using lr_thc_comm
     *     for asymmetric X(k+q_pert)/X(k) transforms.
     *   Term 2 (evaluate_sigma_DeltaW): -G ⊙ ΔW using standard thc_solver_comm
     *     (G is unperturbed, so symmetric transforms are correct).
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
      template<nda::MemoryArray Array_base_t>
      using sArray_t = math::shm::shared_array<Array_base_t>;
      template<int N>
      using shape_t = std::array<long, N>;

    public:
      lr_gw(const imag_axes_ft::IAFT *ft,
            nda::array<double, 1> const& q_pert,
            div_treatment_e div = gygi);

      ~lr_gw() {}

      /**
       * Term 1: ΔΣ = -ΔG ⊙ W_c (R-space)
       *
       * TODO: Divergence correction not yet implemented (see class docstring).
       *
       * @param sDeltaSigma_tskij - [OUTPUT] LR self-energy (nts, ns, nkpts_ibz, nbnd, nbnd)
       * @param DeltaG_tskij      - [INPUT] LR Green's function (nts, ns, nkpts_ibz, nbnd, nbnd)
       * @param dW_qtPQ           - [INPUT] Screened interaction in THC basis (nkpts, nt_half, NP, NP)
       * @param thc               - [INPUT] THC-ERI handler
       */
      template<nda::MemoryArray Array_5D_t, nda::MemoryArray Array_4D_t, typename communicator_t>
      void evaluate_sigma_DeltaG(
          sArray_t<Array_5D_t>& sDeltaSigma_tskij,
          const nda::MemoryArrayOfRank<5> auto& DeltaG_tskij,
          memory::darray_t<Array_4D_t, communicator_t>& dW_qtPQ,
          THC_ERI auto& thc);

      /**
       * Term 2: ΔΣ = -G ⊙ ΔW using standard thc_solver_comm transforms
       *
       * G is unperturbed, so standard (symmetric) THC transforms are correct.
       * Reuses _gw.eval_Sigma_all() by passing ΔW in place of W.
       * TODO: Divergence correction not yet implemented (see class docstring).
       *
       * @param sDeltaSigma_tskij - [OUTPUT] LR self-energy (nts, ns, nkpts_ibz, nbnd, nbnd)
       * @param G_tskij           - [INPUT] Unperturbed Green's function (nts, ns, nkpts_ibz, nbnd, nbnd)
       * @param dDeltaW_qtPQ      - [INPUT] LR screened interaction in THC basis (nkpts, nt_half, NP, NP)
       * @param thc               - [INPUT] THC-ERI handler
       */
      template<nda::MemoryArray Array_5D_t, nda::MemoryArray Array_4D_t, typename communicator_t>
      void evaluate_sigma_DeltaW(
          sArray_t<Array_5D_t>& sDeltaSigma_tskij,
          const nda::MemoryArrayOfRank<5> auto& G_tskij,
          memory::darray_t<Array_4D_t, communicator_t>& dDeltaW_qtPQ,
          THC_ERI auto& thc);

    inline void print_timers() {
      app_log(2, "\n  LR_GW timers");
      app_log(2, "  -------------");
      if (_Timer.has("EVALUATE_SIGMA_R")) {
        app_log(2, "    LR Sigma -DG*W (R-space):       {0:.3f} sec", _Timer.elapsed("EVALUATE_SIGMA_R"));
        app_log(2, "      - Alloc/transpose:            {0:.3f} sec", _Timer.elapsed("SIGMA_ALLOC_R"));
        app_log(2, "      - Primary->Aux:               {0:.3f} sec", _Timer.elapsed("SIGMA_PRIM_TO_AUX"));
        app_log(2, "      - FT (k<->R):                 {0:.3f} sec", _Timer.elapsed("SIGMA_FT_R"));
        app_log(2, "      - Hadamard product:            {0:.3f} sec", _Timer.elapsed("SIGMA_HADPROD_R"));
        app_log(2, "      - Aux->Primary:               {0:.3f} sec", _Timer.elapsed("SIGMA_AUX_TO_PRIM"));
      }
      if (_Timer.has("EVALUATE_SIGMA_DELTAW")) {
        app_log(2, "    LR Sigma -G*DW (via gw_t):      {0:.3f} sec", _Timer.elapsed("EVALUATE_SIGMA_DELTAW"));
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
      void _init_kpq_map(const auto& thc) {
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
       * R-space convolution — single code path for all q_pert.
       * Adapted from gw_t::eval_Sigma_all_Rspace (thc_gw.icc:67-213).
       *
       * Changes from standard eval_Sigma_all_Rspace:
       * 1. Step 1: lr_thc_comm::primary_to_aux (rank-5) with _kpq_map
       * 2. Steps 2-4: Identical (k→R FFT, Hadamard, R→k FFT)
       * 3. Step 5: lr_thc_comm::aux_to_primary (rank-5) with _kpq_map
       */
      template<bool Winp_in_Rspace, bool Wout_in_Rspace,
          nda::MemoryArray Array_5D_t, nda::MemoryArray Array_4D_t,
          typename communicator_t>
      void eval_DeltaSigma_Rspace(const nda::MemoryArrayOfRank<5> auto& DeltaG_tskij,
                                  memory::darray_t<Array_4D_t, communicator_t>& dW_qtPQ,
                                  sArray_t<Array_5D_t>& sDeltaSigma_tskij,
                                  THC_ERI auto& thc,
                                  bool minus_t);

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
      template<nda::MemoryArray Array_view_5D_t>
      void _sigma_div_correction(sArray_t<Array_view_5D_t>& sDeltaSigma_tskij,
                                 const nda::MemoryArrayOfRank<5> auto& DeltaG_tskij,
                                 THC_ERI auto& thc,
                                 const nda::array<ComplexType, 1>& eps_inv_head);

    }; // lr_gw
  } // solvers
} // methods

#endif //COQUI_LR_GW_HPP
