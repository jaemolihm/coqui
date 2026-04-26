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


#ifndef COQUI_LR_SCR_COULOMB_T_HPP
#define COQUI_LR_SCR_COULOMB_T_HPP

#include "configuration.hpp"
#include "nda/nda.hpp"
#include "nda/blas.hpp"
#include "numerics/distributed_array/nda.hpp"

#include "utilities/mpi_context.h"
#include "utilities/Timer.hpp"
#include "utilities/lr_utils.hpp"
#include "IO/app_loggers.h"

#include "numerics/imag_axes_ft/IAFT.hpp"
#include "methods/ERI/detail/concepts.hpp"
#include "methods/scr_coulomb/scr_coulomb_t.h"

namespace methods {
namespace solvers {

  /**
   * @brief Linear response screened Coulomb interaction solver.
   *
   * Computes the linearized W Dyson equation:
   *   ΔW_c(q, iω) = (Z(q) + W_c(q, iω)) · ΔΠ(q, iω) · (Z(q) + W_c(q, iω))
   *
   * Main entry point: solve_lr_dyson_W() takes ΔΠ(τ) and W_c(τ), handles
   * the τ↔ω Fourier transforms internally, and returns ΔW_c(τ). Mirrors the
   * interface of scr_coulomb_t::dyson_W_from_Pi_tau().
   *
   * Separated from scr_coulomb_t because the standard class computes W from Π
   * via matrix inversion: W = (I - Z·Π)^{-1}·Z - Z, while the LR version
   * linearizes this to two matrix multiplications.
   */
  class lr_scr_coulomb_t {
  public:
    lr_scr_coulomb_t(const imag_axes_ft::IAFT* ft,
                     nda::array<double, 1> q_pert);

    ~lr_scr_coulomb_t() {}

    /**
     * Compute W_full(iω) = W_c(iω) + V from W_c(τ).
     *
     * Input: dW_c_tqPQ in q-local (τ-dist) distribution, consumed (reset after FT).
     * Output: dW_full_wqPQ in PQ-local distribution (ready for lr_dyson_W_in_place).
     *
     * @param dW_c_tqPQ - [IN] W_c(τ) in τ-dist, consumed (reset after use)
     * @param thc       - [IN] THC-ERI handler
     * @return W_full(iω) = W_c(iω) + V in PQ-local distribution
     */
    template<nda::MemoryArrayOfRank<4> local_Array_t, typename communicator_t>
    auto compute_W_full_omega(
        memory::darray_t<local_Array_t, communicator_t>& dW_c_tqPQ,
        THC_ERI auto& thc)
    -> memory::darray_t<local_Array_t, mpi3::communicator>;

    /**
     * LR W Dyson: ΔΠ(τ) → ΔW_c(τ), in-place.
     *
     * Input: dDeltaPi_tqPQ in q-local (τ-dist) distribution.
     * On output: dDeltaPi_tqPQ is overwritten with ΔW_c(τ) in q-local distribution.
     *
     * @param dDeltaPi_tqPQ   - [IN/OUT] ΔΠ(τ) on input, ΔW_c(τ) on output (τ-dist)
     * @param dW_full_wqPQ    - [IN] W_c(iω) + V in PQ-local distribution (not consumed)
     * @param thc             - [IN] THC-ERI handler
     */
    template<nda::MemoryArrayOfRank<4> local_Array_t, typename communicator_t,
             typename dW_full_t>
    void solve_lr_dyson_W(
        memory::darray_t<local_Array_t, communicator_t>& dDeltaPi_tqPQ,
        const dW_full_t& dW_full_wqPQ,
        THC_ERI auto& thc);

    /**
     * LR W Dyson in Matsubara frequency space (in-place).
     * ΔΠ at storage iq holds the (q+Q, iq) Bloch sector; the Dyson formula is
     *   Δ^Q W^{q}(iω) = W_full^{q+Q}(iω) · Δ^Q Π^{q}(iω) · W_full^{q}(iω)
     * where q is the bosonic wavevector of Coulomb interaction and Q is the perturbation wavevector.
     *
     * Both arrays must be in PQ-local distribution (pgrid[2]==1, pgrid[3]==1).
     * On output, dDeltaPi_wqPQ contains ΔW_c(iω).
     *
     * @param dDeltaPi_wqPQ - [IN/OUT] ΔΠ on input, ΔW_c on output
     * @param dW_full_wqPQ  - [IN] W_full = W_c(iω) + V in PQ-local distribution
     */
    template<nda::MemoryArray Array_4D_t, typename communicator_t,
             typename dW_full_t>
    void lr_dyson_W_in_place(
        memory::darray_t<Array_4D_t, communicator_t>& dDeltaPi_wqPQ,
        const dW_full_t& dW_full_wqPQ);

  inline void print_timers() {
    app_log(2, "\n  LR_SCR_COULOMB timers");
    app_log(2, "  ----------------------");
    if (_Timer.has("LR_W_FT_TAU_TO_W") && _Timer.elapsed("LR_W_FT_TAU_TO_W") > 0) {
      app_log(2, "    LR W FT (τ→ω):                  {0:8.3f} sec  {1:4d} calls", _Timer.elapsed("LR_W_FT_TAU_TO_W"), _Timer.number_of_calls("LR_W_FT_TAU_TO_W"));
    }
    app_log(2, "    LR W Dyson (iω):                {0:8.3f} sec  {1:4d} calls", _Timer.elapsed("EVALUATE_LR_W"), _Timer.number_of_calls("EVALUATE_LR_W"));
    if (_Timer.has("LR_W_FT_W_TO_TAU") && _Timer.elapsed("LR_W_FT_W_TO_TAU") > 0) {
      app_log(2, "    LR W FT (ω→τ):                  {0:8.3f} sec  {1:4d} calls", _Timer.elapsed("LR_W_FT_W_TO_TAU"), _Timer.number_of_calls("LR_W_FT_W_TO_TAU"));
    }
    app_log(2, "");
  }

  private:
    const imag_axes_ft::IAFT* _ft;
    utils::TimerManager _Timer;
    nda::array<double, 1> _q_pert;
    bool _kpq_map_initialized = false;
    nda::array<int, 1> _kpq_map;  // iq → iq+q_pert mapping (full BZ); empty for q_pert=Γ

    void _init_kpq_map(THC_ERI auto& thc);

  }; // lr_scr_coulomb_t

} // solvers
} // methods

#include "methods/scr_coulomb/lr_scr_coulomb_t.icc"

#endif //COQUI_LR_SCR_COULOMB_T_HPP
