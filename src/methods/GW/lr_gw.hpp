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


#ifndef COQUI_LR_GW_HPP
#define COQUI_LR_GW_HPP

#include <optional>

#include "nda/nda.hpp"
#include "numerics/distributed_array/nda.hpp"
#include "numerics/fft/fft_kR.hpp"
#include "numerics/shared_array/nda.hpp"

#include "utilities/mpi_context.h"
#include "utilities/Timer.hpp"

#include "mean_field/MF.hpp"
#include "numerics/imag_axes_ft/IAFT.hpp"
#include "methods/ERI/detail/concepts.hpp"
#include "methods/SCF/lr_ibc.hpp"
#include "methods/GW/gw_t.h"

namespace methods {

  class thc_reader_t;  // forward declaration

  namespace solvers {
    using namespace memory;

    using sArrv_5D_t = math::shm::shared_array<nda::array_view<ComplexType, 5>>;
    using dArr_4D_t = memory::darray_t<nda::array<ComplexType, 4>, mpi3::communicator>;
    using dArr_5D_t = memory::darray_t<nda::array<ComplexType, 5>, mpi3::communicator>;

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
     * Divergence correction (q→0, 1/q² singularity) via apply_div_correction_DeltaG
     * and apply_div_correction_G:
     *   Term 1 (all q_pert): ΔΣ^div += -madelung * eps_inv_head * S(k+q) · ΔG · S(k)
     *   Term 2 (q_pert=0):   ΔΣ^div += -madelung * Δeps_inv_head * S(k) · G · S(k)
     * Called by lr_driver after evaluate_sigma / evaluate_sigma_DeltaG.
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
            std::string div = "gygi");

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
       * @param dG_tsRPQ          - [INPUT] cached G^R(τ) in aux basis from
       *   lr_precompute_G_R_pair (it, s, R, P, Q). Term 2 reads G^R from the
       *   cache and skips the per-τ G Primary→Aux + k→R FT.
       * @param dG_mtau_tsRPQ     - [INPUT] cached G^R(β−τ), same layout
       * @param ibc               - [INPUT] optional DeltaX IBC correction data
       */
      void evaluate_sigma(
          sArrv_5D_t& sDeltaSigma_tskij,
          const nda::array_view<ComplexType, 5>& DeltaG_tskij,
          dArr_4D_t& dW_tRPQ,
          const nda::array_view<ComplexType, 5>& G_tskij,
          dArr_4D_t& dDeltaW_qtPQ,
          thc_reader_t& thc,
          const dArr_5D_t& dG_tsRPQ,
          const dArr_5D_t& dG_mtau_tsRPQ,
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
       * @param dG_tsRPQ          - [INPUT] cached G^R(τ) in aux basis (it,s,R,P,Q)
       * @param dG_mtau_tsRPQ     - [INPUT] cached G^R(β−τ), same layout
       */
      void evaluate_sigma_DeltaW(
          sArrv_5D_t& sDeltaSigma_tskij,
          const nda::array_view<ComplexType, 5>& G_tskij,
          dArr_4D_t& dDeltaW_qtPQ,
          thc_reader_t& thc,
          const dArr_5D_t& dG_tsRPQ,
          const dArr_5D_t& dG_mtau_tsRPQ);

      /**
       * Divergence correction term 1 (all q_pert):
       *   ΔΣ += -madelung * eps_inv_head(τ) * S(k+q) · ΔG · S(k)
       * Uses _sigma_div_correction with S(k+q)/S(k).
       */
      void apply_div_correction_DeltaG(
          sArrv_5D_t& sDeltaSigma_tskij,
          const nda::array_view<ComplexType, 5>& DeltaG_tskij,
          const nda::array_view<ComplexType, 4>& S_skij,
          thc_reader_t& thc,
          const nda::array<ComplexType, 1>& eps_inv_head);

      /**
       * Divergence correction term 2 (q_pert=0 only):
       *   ΔΣ += -madelung * Δeps_inv_head(τ) * S(k) · G · S(k)
       * Uses _sigma_div_correction, called only for q_pert = 0 so S(k+q) = S(k)
       * on the left and right.
       */
      void apply_div_correction_G(
          sArrv_5D_t& sDeltaSigma_tskij,
          const nda::array_view<ComplexType, 5>& G_tskij,
          const nda::array_view<ComplexType, 4>& S_skij,
          thc_reader_t& thc,
          const nda::array<ComplexType, 1>& delta_eps_inv_head);

    /// Print the LR Sigma timer block (header + subclocks) at log level `level`.
    inline void print_timers(int level = 2) {
      app_log(level, "\n  LR_GW timers");
      app_log(level, "  -------------");
      print_subclocks(level, "  ");
      app_log(level, "");
    }

    /// Print only the component clocks, each line prefixed by `indent`.
    /// Embedded (with deeper indent) in lr_driver's final hierarchical report.
    /// The R-space total leads; transpose/div-correction siblings live outside
    /// EVALUATE_SIGMA_R, so the R-space components below sum to that line, not
    /// to the driver's "LR GW Sigma (total)". The deeper-indented block under
    /// Aux->Primary breaks that one line down and is already counted in it —
    /// never add those lines to the sum.
    inline void print_subclocks(int level, const std::string& indent) {
      app_log(level, "{0}  - Sigma R-space (total):      {1:8.3f} sec  {2:4d} calls", indent, _Timer.elapsed("EVALUATE_SIGMA_R"), _Timer.number_of_calls("EVALUATE_SIGMA_R"));
      app_log(level, "{0}  - Alloc:                      {1:8.3f} sec  {2:4d} calls", indent, _Timer.elapsed("SIGMA_ALLOC"), _Timer.number_of_calls("SIGMA_ALLOC"));
      app_log(level, "{0}  - FT coefficients:            {1:8.3f} sec  {2:4d} calls", indent, _Timer.elapsed("SIGMA_FT_COEFF"), _Timer.number_of_calls("SIGMA_FT_COEFF"));
      app_log(level, "{0}  - W slice copy:               {1:8.3f} sec  {2:4d} calls", indent, _Timer.elapsed("SIGMA_W_COPY"), _Timer.number_of_calls("SIGMA_W_COPY"));
      app_log(level, "{0}  - Pre-loop fence (sync):      {1:8.3f} sec  {2:4d} calls", indent, _Timer.elapsed("SIGMA_PRE_FENCE"), _Timer.number_of_calls("SIGMA_PRE_FENCE"));
      app_log(level, "{0}  - Primary->Aux:               {1:8.3f} sec  {2:4d} calls", indent, _Timer.elapsed("SIGMA_PRIM_TO_AUX"), _Timer.number_of_calls("SIGMA_PRIM_TO_AUX"));
      app_log(level, "{0}  - FT (k<->R):                 {1:8.3f} sec  {2:4d} calls", indent, _Timer.elapsed("SIGMA_FT_R"), _Timer.number_of_calls("SIGMA_FT_R"));
      app_log(level, "{0}  - Hadamard product:           {1:8.3f} sec  {2:4d} calls", indent, _Timer.elapsed("SIGMA_HADPROD_R"), _Timer.number_of_calls("SIGMA_HADPROD_R"));
      app_log(level, "{0}  - Aux->Primary:               {1:8.3f} sec  {2:4d} calls", indent, _Timer.elapsed("SIGMA_AUX_TO_PRIM"), _Timer.number_of_calls("SIGMA_AUX_TO_PRIM"));
      app_log(level, "{0}    - Buffer alloc:             {1:8.3f} sec  {2:4d} calls", indent, _Timer.elapsed("SIGMA_A2P_ALLOC"), _Timer.number_of_calls("SIGMA_A2P_ALLOC"));
      app_log(level, "{0}    - GEMM:                     {1:8.3f} sec  {2:4d} calls", indent, _Timer.elapsed("SIGMA_A2P_GEMM"), _Timer.number_of_calls("SIGMA_A2P_GEMM"));
      app_log(level, "{0}    - MPI reduce:               {1:8.3f} sec  {2:4d} calls", indent, _Timer.elapsed("SIGMA_A2P_REDUCE"), _Timer.number_of_calls("SIGMA_A2P_REDUCE"));
      app_log(level, "{0}    - AXPY (root):              {1:8.3f} sec  {2:4d} calls", indent, _Timer.elapsed("SIGMA_A2P_AXPY"), _Timer.number_of_calls("SIGMA_A2P_AXPY"));
      app_log(level, "{0}  - Final reduce (ΔΣ):          {1:8.3f} sec  {2:4d} calls", indent, _Timer.elapsed("SIGMA_FINAL_REDUCE"), _Timer.number_of_calls("SIGMA_FINAL_REDUCE"));
      app_log(level, "{0}  - Div correction:             {1:8.3f} sec  {2:4d} calls", indent, _Timer.elapsed("SIGMA_DIV_CORR"), _Timer.number_of_calls("SIGMA_DIV_CORR"));
    }

    private:
      nda::array<double, 1> _q_pert;
      gw_t _gw;                       // stores div_treatment settings (friend access)
      bool _kpq_map_initialized = false;
      nda::array<int, 1> _kpq_map;   // k → k+q_pert mapping (full BZ); identity for q_pert=0
      utils::TimerManager _Timer;

      // Cached per-run setup for _eval_sigma_Rspace, built lazily on the first
      // call by _setup_workspace and reused across SCF iterations (comm split +
      // node split inside make_mpi_context are collective; shm windows and
      // distributed buffers are expensive to recreate). The τ-dist is fixed for
      // the solver's lifetime. The guard is keyed on the (do_term1, do_term2)
      // combination, not a bare bool: the term2-only buffers (_fft_q, _W2_tau_RPQ,
      // _sf_qR) are built only when do_term2 is set, so reusing a workspace built
      // for a different term combination would dereference an empty optional.
      // _setup_workspace aborts loudly if asked to reuse under different flags.
      // Member order matters: the darray buffers store a pointer to _tau_comm,
      // and the shm arrays reference _tau_mpi — both must outlive (= be declared
      // before) them.
      using sArr_2D_t = math::shm::shared_array<nda::array_view<ComplexType, 2>>;
      using dArr_loc_4D_t = memory::darray_t<memory::array<HOST_MEMORY, ComplexType, 4>, mpi3::communicator>;
      bool _setup_done = false;
      bool _setup_term1 = false, _setup_term2 = false;
      std::optional<mpi3::communicator> _tau_comm;
      std::optional<utils::mpi_context_t<mpi3::communicator>> _tau_mpi;
      std::optional<dArr_loc_4D_t> _dG_skPQ, _dSigma_skPQ;
      std::optional<sArr_2D_t> _sf_Rk, _sf_kR, _sf_qR;
      // Blocked-FFT k<->R transforms; COQUI_LR_DEBUG_GEMM_FT=1 leaves these
      // empty and uses the gemm path (coefficient matrices above) instead.
      std::optional<math::fft::fft_kR_t> _fft_k, _fft_q;
      nda::matrix<ComplexType> _ft_buffer;
      nda::array<ComplexType, 3> _W2_tau_RPQ;
      // Reduction buffer handed to aux_to_primary_accumulate. nda::array
      // zero-initializes, so leaving it to the kernel costs a fresh
      // (ns·nk, nbnd, nbnd) mapping and its first-touch page faults on every one
      // of the 2·nt_loc calls per Σ evaluation. A function-static buffer would
      // hide the lifetime, never free, and not show in print_memory_estimate;
      // making lr_thc_comm an instantiated object would touch all of its call
      // sites for one buffer.
      nda::array<ComplexType, 3> _a2p_buf;

      /// Initialize _kpq_map from THC's mean-field k-points (lazy, once)
      void _init_kpq_map(thc_reader_t& thc);

      /**
       * Build the cached per-run workspace (tau subcommunicator, work arrays,
       * shm windows, FT coefficients / FFT objects) on the first call; no-op
       * once built. PQ grid/block and t_origin are read from dW_ref (term1:
       * t,R,P,Q layout; term2: q,t,P,Q layout).
       */
      void _setup_workspace(thc_reader_t& thc, dArr_4D_t const& dW_ref,
                            bool do_term1, bool do_term2,
                            long ns, long nk_ibz, long nbnd);

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
          const lr_ibc_DeltaX* ibc = nullptr,
          const dArr_5D_t* dG_tsRPQ = nullptr,
          const dArr_5D_t* dG_mtau_tsRPQ = nullptr);

      /**
       * LR-aware divergence correction: ΔΣ^div = factor · S(k+q) · ΔG · S(k)
       *
       * For q_pert≠0, the left and right overlap matrices use different k:
       *   S_L(k) = S(k+q_pert)
       *   S_R(k) = S(k)
       * At q_pert=0, S_L = S_R = S(k).
       */
      void _sigma_div_correction(
          sArrv_5D_t& sDeltaSigma_tskij,
          const nda::array_view<ComplexType, 5>& DeltaG_tskij,
          const nda::array_view<ComplexType, 4>& S_skij,
          thc_reader_t& thc,
          const nda::array<ComplexType, 1>& eps_inv_head);

    }; // lr_gw
  } // solvers
} // methods

#endif //COQUI_LR_GW_HPP
