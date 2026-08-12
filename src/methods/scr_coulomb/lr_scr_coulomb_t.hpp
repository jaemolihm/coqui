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


#ifndef COQUI_LR_SCR_COULOMB_T_HPP
#define COQUI_LR_SCR_COULOMB_T_HPP

#include <optional>
#include <type_traits>

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
#include "methods/scr_coulomb/scr_coulomb_fourier_t.h"

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
    // Concrete LR distributed-array type (host, mpi3); used by the W_full(q+Q)
    // member cache below.
    using dArr4D_concrete_t = memory::darray_t<nda::array<ComplexType, 4>, mpi3::communicator>;

    lr_scr_coulomb_t(const imag_axes_ft::IAFT* ft,
                     nda::array<double, 1> q_pert);

    ~lr_scr_coulomb_t() {}

    /**
     * Compute W_full(iω) = W_c(iω) + V from W_c(τ).
     *
     * Input: dW_c_tqPQ in q-local (τ-dist) distribution.
     * Output: dW_full_wqPQ in PQ-local distribution (ready for lr_dyson_W_in_place).
     *
     * @param dW_c_tqPQ   - [IN] W_c(τ) in τ-dist
     * @param thc         - [IN] THC-ERI handler
     * @param reset_input - [IN] release dW_c_tqPQ once the FT has consumed it.
     *                      True whenever this is W_c's only consumer: the release
     *                      happens inside the FT, before the ω-side output is
     *                      allocated, so it lowers the peak in a way a caller-side
     *                      reset after the call cannot. Pass false only to keep the
     *                      array for a second consumer (the driver hands the same
     *                      (t,q) copy on to lr_precompute_W_tRPQ).
     *                      Deliberately has no default — releasing a caller's array
     *                      is not something a call site should inherit silently.
     * @return W_full(iω) = W_c(iω) + V in PQ-local distribution
     */
    template<nda::MemoryArrayOfRank<4> local_Array_t, typename communicator_t>
    auto compute_W_full_omega(
        memory::darray_t<local_Array_t, communicator_t>& dW_c_tqPQ,
        THC_ERI auto& thc,
        bool reset_input)
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
     * Stateless: both W operands are passed in (not read from a member), so the
     * function is a pure function of its arguments. solve_lr_dyson_W supplies the
     * (q+Q) operand from _dW_full_qpQ_wqPQ (built once in compute_W_full_omega);
     * for Q=Γ it passes dW_full_wqPQ for both, since W_full(q+Q) = W_full(q).
     *
     * @param dDeltaPi_wqPQ    - [IN/OUT] ΔΠ on input, ΔW_c on output
     * @param dW_full_wqPQ     - [IN] W_full(q) = W_c(iω) + V in PQ-local distribution
     * @param dW_full_qpQ_wqPQ - [IN] W_full(q+Q) left operand (== dW_full_wqPQ for Q=Γ)
     */
    template<nda::MemoryArray Array_4D_t, typename communicator_t,
             typename dW_full_t>
    void lr_dyson_W_in_place(
        memory::darray_t<Array_4D_t, communicator_t>& dDeltaPi_wqPQ,
        const dW_full_t& dW_full_wqPQ,
        const dW_full_t& dW_full_qpQ_wqPQ);

    bool is_q_gamma() const { return _is_q_gamma; }

  /**
   * Print the LR W Dyson timer block (header + total + subclocks) at log
   * level `level` (2: standard, 3: per-step diagnostics).
   */
  inline void print_timers(int level = 2) {
    app_log(level, "\n  LR_SCR_COULOMB timers");
    app_log(level, "  ----------------------");
    app_log(level, "    Solve LR W Dyson:               {0:8.3f} sec  {1:4d} calls", _Timer.elapsed("SOLVE_LR_DYSON_W"), _Timer.number_of_calls("SOLVE_LR_DYSON_W"));
    print_subclocks(level, "    ");
    app_log(level, "");
  }

  /**
   * Print only the component clocks, each line prefixed by `indent`.
   * Embedded (with deeper indent) in lr_driver's final hierarchical report.
   * The flat FT component clocks (redistribute / gemm / alloc / leak check)
   * come from the member scr_coulomb_fourier_t's internal timers; they only cover
   * solve_lr_dyson_W calls (compute_W_full_omega uses a separate local
   * scr_coulomb_fourier_t so the one-time setup FT does not pollute them).
   */
  inline void print_subclocks(int level, const std::string& indent) {
    auto& ftT = _scr_fourier.timer();
    app_log(level, "{0}  - LR W FT (τ→ω):              {1:8.3f} sec  {2:4d} calls", indent, _Timer.elapsed("LR_W_FT_TAU_TO_W"), _Timer.number_of_calls("LR_W_FT_TAU_TO_W"));
    app_log(level, "{0}  - LR W Dyson (iω):            {1:8.3f} sec  {2:4d} calls", indent, _Timer.elapsed("EVALUATE_LR_W"), _Timer.number_of_calls("EVALUATE_LR_W"));
    app_log(level, "{0}  - LR W (q+Q)-operand comm:    {1:8.3f} sec  {2:4d} calls", indent, _Timer.elapsed("LR_W_COMM_QPQ"), _Timer.number_of_calls("LR_W_COMM_QPQ"));
    app_log(level, "{0}  - LR W FT (ω→τ):              {1:8.3f} sec  {2:4d} calls", indent, _Timer.elapsed("LR_W_FT_W_TO_TAU"), _Timer.number_of_calls("LR_W_FT_W_TO_TAU"));
    app_log(level, "{0}    - redistribute (τ→ω, ω→τ): {1:8.3f} sec  {2:4d} calls", indent, ftT.elapsed("FT_REDISTRIBUTE"), ftT.number_of_calls("FT_REDISTRIBUTE"));
  }

  private:
    const imag_axes_ft::IAFT* _ft;
    utils::TimerManager _Timer;
    // Persistent FT engine: keeps the internal redistribute/gemm/leak-check
    // timers of tau_to_w / w_to_tau accumulating across solve_lr_dyson_W calls
    // (a per-call local instance would lose them before printing).
    scr_coulomb_fourier_t _scr_fourier;
    nda::array<double, 1> _q_pert;     // perturbation wavevector (size 3, crystal coords)
    bool _is_q_gamma;                  // set in constructor; true iff _q_pert ≈ Γ
    bool _kpq_map_initialized = false; // lazy init for _kpq_map (needs THC handle)
    nda::array<int, 1> _kpq_map;       // iq → iq+q_pert (full BZ); only built for Q≠Γ

    // q-shifted W_full(q+Q) for the Q≠Γ Dyson. Built once during setup and reused during
    // LR iterations. Empty for Q=Γ (the operand is then W_full itself).
    std::optional<dArr4D_concrete_t> _dW_full_qpQ_wqPQ;

    void _init_kpq_map(THC_ERI auto& thc);

    /**
     * Build the Q≠Γ Dyson (q+Q) operand W_full(q+Q) into _dW_full_qpQ_wqPQ:
     *   dW_full_qpQ[iw, iq, P, Q] = dW_full_wqPQ[iw, kpq_map(iq), P, Q]
     * via an Alltoallv within the q-pool subgroup. No-op for Q=Γ (the Dyson then
     * reuses W_full(q) for both operands). Precondition: _kpq_map populated.
     */
    void _build_W_full_qpQ(const dArr4D_concrete_t& dW_full_wqPQ,
                           THC_ERI auto& thc);

  }; // lr_scr_coulomb_t

} // solvers
} // methods

#include "methods/scr_coulomb/lr_scr_coulomb_t.icc"

#endif //COQUI_LR_SCR_COULOMB_T_HPP
