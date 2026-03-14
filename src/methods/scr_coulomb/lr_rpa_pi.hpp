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


#ifndef COQUI_LR_RPA_PI_HPP
#define COQUI_LR_RPA_PI_HPP

#include "nda/nda.hpp"
#include "numerics/distributed_array/nda.hpp"
#include "numerics/shared_array/nda.hpp"

#include "utilities/mpi_context.h"
#include "utilities/Timer.hpp"

#include "mean_field/MF.hpp"
#include "numerics/imag_axes_ft/IAFT.hpp"
#include "methods/ERI/detail/concepts.hpp"

namespace methods {

  class thc_reader_t;  // forward declaration

  namespace solvers {
    using namespace memory;

    using dArr_4D_t = memory::darray_t<nda::array<ComplexType, 4>, mpi3::communicator>;

    /**
     * Linear response RPA polarization: ΔP = -ΔG·G - G·ΔG.
     *
     * Computes the LR polarization via product-rule differentiation of P = -G·G
     * using R-space Hadamard product (convolution in k-space).
     *
     * Uses lr_thc_comm for ΔG factors (asymmetric X(k+q_pert)/X(k)),
     * thc_solver_comm for G factors (symmetric X(k)/X(k)).
     * Output is ΔP(τ,q,P,Q) for τ ∈ [0, β/2], same format as standard Pi.
     *
     * Separated from lr_gw because the polarization bubble is logically
     * part of the screened Coulomb machinery (scr_coulomb), not the
     * GW self-energy solver. lr_gw depends on this class for its W update.
     *
     * Requires nqpts == nqpts_ibz == nkpts (no symmetry).
     */
    class lr_rpa_pi {
    public:
      template<int N>
      using shape_t = std::array<long, N>;

    public:
      lr_rpa_pi(nda::array<double, 1> const& q_pert);

      ~lr_rpa_pi() {}

      /**
       * Compute LR polarization: ΔΠ = ΔG·G + G·ΔG (R-space Hadamard product).
       *
       * Two terms per τ point (p = q_pert):
       *   Term 1: ΔG_{PQ}(R,τ)       · conj(G_{PQ}(R,β-τ))
       *   Term 2: e^{ip·R} · G_{PQ}(R,τ) · ΔG_{QP}(-R,β-τ)
       *
       * @param G_tskij      - [IN] Unperturbed G(τ), (nts, ns, nkpts_ibz, nbnd, nbnd)
       * @param DeltaG_tskij - [IN] LR Green's function ΔG(τ), same shape
       * @param thc          - [IN] THC-ERI handler
       * @return ΔP(τ,q,P,Q) distributed array (nt_half, nkpts, NP, NP)
       *         Distribution: τ-dist {tpools, 1, np_P, np_Q}.
       */
      dArr_4D_t evaluate_lr_Pi(
          const nda::array_view<ComplexType, 5>& G_tskij,
          const nda::array_view<ComplexType, 5>& DeltaG_tskij,
          thc_reader_t& thc);

    inline void print_timers() {
      app_log(2, "\n  LR_RPA_PI timers");
      app_log(2, "  -----------------");
      app_log(2, "    LR Pi eval:                     {0:8.3f} sec  {1:4d} calls", _Timer.elapsed("EVALUATE_LR_PI"), _Timer.number_of_calls("EVALUATE_LR_PI"));
      app_log(2, "      - Alloc:                      {0:8.3f} sec  {1:4d} calls", _Timer.elapsed("PI_LR_ALLOC"), _Timer.number_of_calls("PI_LR_ALLOC"));
      app_log(2, "      - Primary->Aux:               {0:8.3f} sec  {1:4d} calls", _Timer.elapsed("PI_LR_PRIM_TO_AUX"), _Timer.number_of_calls("PI_LR_PRIM_TO_AUX"));
      app_log(2, "      - Trev unfold + phase:        {0:8.3f} sec  {1:4d} calls", _Timer.elapsed("PI_LR_TREV_PHASE"), _Timer.number_of_calls("PI_LR_TREV_PHASE"));
      app_log(2, "      - FT (k<->R):                 {0:8.3f} sec  {1:4d} calls", _Timer.elapsed("PI_LR_FT_R"), _Timer.number_of_calls("PI_LR_FT_R"));
      app_log(2, "      - Hadamard product:           {0:8.3f} sec  {1:4d} calls\n", _Timer.elapsed("PI_LR_HADPROD"), _Timer.number_of_calls("PI_LR_HADPROD"));
    }

    private:
      nda::array<double, 1> _q_pert;
      bool _kpq_map_initialized = false;
      nda::array<long, 1> _kpq_map;  // k → k+q_pert mapping (full BZ); identity for q_pert=0
      utils::TimerManager _Timer;

      /// Initialize _kpq_map from THC's mean-field k-points (lazy, once)
      void _init_kpq_map(thc_reader_t& thc);

    }; // lr_rpa_pi
  } // solvers
} // methods

#endif //COQUI_LR_RPA_PI_HPP
