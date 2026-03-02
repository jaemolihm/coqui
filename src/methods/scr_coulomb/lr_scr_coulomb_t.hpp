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
#include "numerics/distributed_array/nda.hpp"

#include "utilities/mpi_context.h"
#include "utilities/Timer.hpp"
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
   * Main entry point: lr_dyson_W_from_Pi_tau() takes ΔΠ(τ) and W_c(τ), handles
   * the τ↔ω Fourier transforms internally, and returns ΔW_c(τ). Mirrors the
   * interface of scr_coulomb_t::dyson_W_from_Pi_tau().
   *
   * Separated from scr_coulomb_t because the standard class computes W from Π
   * via matrix inversion: W = (I - Z·Π)^{-1}·Z - Z, while the LR version
   * linearizes this to two matrix multiplications.
   */
  class lr_scr_coulomb_t {
  public:
    lr_scr_coulomb_t(const imag_axes_ft::IAFT* ft);

    ~lr_scr_coulomb_t() {}

    /**
     * LR W Dyson: ΔΠ(τ) → ΔW_c(τ).
     *
     * Mirrors scr_coulomb_t::dyson_W_from_Pi_tau but for the linearized equation.
     * Internally: FT ΔΠ(τ)→ΔΠ(iω), FT W_c(τ)→W_c(iω), lr_dyson_W_in_place, FT ΔW_c(iω)→ΔW_c(τ).
     *
     * @param dDeltaPi_tqPQ - [IN] ΔΠ(τ) in τ-space, reset after use
     * @param dW_c_tqPQ     - [IN] W_c(τ) in τ-space, reset after use
     * @param thc           - [IN] THC-ERI handler
     * @return ΔW_c(τ) distributed array with same pgrid/bsize as input
     */
    template<nda::MemoryArrayOfRank<4> local_Array_t, typename communicator_t>
    auto lr_dyson_W_from_Pi_tau(
        memory::darray_t<local_Array_t, communicator_t>& dDeltaPi_tqPQ,
        memory::darray_t<local_Array_t, communicator_t>& dW_c_tqPQ,
        THC_ERI auto& thc)
    -> memory::darray_t<local_Array_t, mpi3::communicator>;

    /**
     * LR W Dyson in Matsubara frequency space (in-place).
     * ΔΠ(iω) → ΔW_c(iω) = (Z + W_c) · ΔΠ · (Z + W_c)
     *
     * Both arrays must be in ω-space with shape (nw_half, nqpts, NP, NP).
     * On output, dDeltaPi_wqPQ contains ΔW_c(iω).
     *
     * @param dDeltaPi_wqPQ - [IN/OUT] ΔΠ on input, ΔW_c on output
     * @param dW_c_wqPQ     - [IN] Correlated screened interaction W_c(iω)
     * @param thc           - [IN] THC-ERI handler
     */
    template<nda::MemoryArray Array_4D_t, typename communicator_t>
    void lr_dyson_W_in_place(
        memory::darray_t<Array_4D_t, communicator_t>& dDeltaPi_wqPQ,
        memory::darray_t<Array_4D_t, communicator_t>& dW_c_wqPQ,
        THC_ERI auto& thc);

  inline void print_timers() {
    app_log(2, "\n  LR_SCR_COULOMB timers");
    app_log(2, "  ----------------------");
    app_log(2, "    LR W Dyson (iω):                {0:.3f} sec\n", _Timer.elapsed("EVALUATE_LR_W"));
  }

  private:
    const imag_axes_ft::IAFT* _ft;
    utils::TimerManager _Timer;

  }; // lr_scr_coulomb_t

} // solvers
} // methods

#include "methods/scr_coulomb/lr_scr_coulomb_t.icc"

#endif //COQUI_LR_SCR_COULOMB_T_HPP
