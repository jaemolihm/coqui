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


#ifndef COQUI_LR_RPA_PI_HPP
#define COQUI_LR_RPA_PI_HPP

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

namespace methods {

  class thc_reader_t;  // forward declaration

  namespace solvers {
    using namespace memory;

    using dArr_4D_t = memory::darray_t<nda::array<ComplexType, 4>, mpi3::communicator>;
    using dArr_5D_t = memory::darray_t<nda::array<ComplexType, 5>, mpi3::communicator>;

    /**
     * Linear response RPA polarization: ΔP = -ΔG·G - G·ΔG.
     *
     * Computes the LR polarization via product-rule differentiation of P = -G·G
     * using R-space Hadamard product (convolution in k-space).
     *
     * Uses lr_thc_comm for the ΔG factors (asymmetric X(k+q_pert)/X(k)); the
     * unperturbed G^R factors are supplied precomputed via lr_precompute_G_R_pair.
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
       * @param G_tskij      - [IN] Unperturbed G(τ), (nts, ns, nkpts_ibz, nbnd, nbnd).
       *                       Used only for dimensions / shape validation; the G
       *                       factor is read from the dG_tsRPQ / dG_mtau_tsRPQ cache.
       * @param DeltaG_tskij - [IN] LR Green's function ΔG(τ), same shape
       * @param thc          - [IN] THC-ERI handler
       * @param dG_tsRPQ     - [IN] cached G^R(τ) in aux basis from
       *                       lr_precompute_G_R_pair (it, s, R, P, Q); must share
       *                       Π's τ-dist. The G factor is always read from here.
       * @param dG_mtau_tsRPQ    - [IN] cached G^R(β−τ), same layout
       * @param ibc          - [IN] optional DeltaX IBC correction data
       * @return ΔP(τ,q,P,Q) distributed array (nt_half, nkpts, NP, NP)
       *         Distribution: τ-dist {tpools, 1, np_P, np_Q}.
       */
      dArr_4D_t evaluate_lr_Pi(
          const nda::array_view<ComplexType, 5>& G_tskij,
          const nda::array_view<ComplexType, 5>& DeltaG_tskij,
          thc_reader_t& thc,
          const dArr_5D_t& dG_tsRPQ,
          const dArr_5D_t& dG_mtau_tsRPQ,
          const lr_ibc_DeltaX* ibc = nullptr);

    /// Print the LR Pi timer block (header + total + subclocks) at log level `level`.
    inline void print_timers(int level = 2) {
      app_log(level, "\n  LR_RPA_PI timers");
      app_log(level, "  -----------------");
      app_log(level, "    LR Pi eval:                     {0:8.3f} sec  {1:4d} calls", _Timer.elapsed("EVALUATE_LR_PI"), _Timer.number_of_calls("EVALUATE_LR_PI"));
      print_subclocks(level, "    ");
      app_log(level, "");
    }

    /// Print only the component clocks, each line prefixed by `indent`.
    /// Embedded (with deeper indent) in lr_driver's final hierarchical report.
    inline void print_subclocks(int level, const std::string& indent) {
      app_log(level, "{0}  - Alloc:                      {1:8.3f} sec  {2:4d} calls", indent, _Timer.elapsed("PI_LR_ALLOC"), _Timer.number_of_calls("PI_LR_ALLOC"));
      app_log(level, "{0}  - Primary->Aux:               {1:8.3f} sec  {2:4d} calls", indent, _Timer.elapsed("PI_LR_PRIM_TO_AUX"), _Timer.number_of_calls("PI_LR_PRIM_TO_AUX"));
      app_log(level, "{0}  - Trev unfold + phase:        {1:8.3f} sec  {2:4d} calls", indent, _Timer.elapsed("PI_LR_TREV_PHASE"), _Timer.number_of_calls("PI_LR_TREV_PHASE"));
      app_log(level, "{0}  - FT (k<->R):                 {1:8.3f} sec  {2:4d} calls", indent, _Timer.elapsed("PI_LR_FT_R"), _Timer.number_of_calls("PI_LR_FT_R"));
      app_log(level, "{0}  - Hadamard product:           {1:8.3f} sec  {2:4d} calls", indent, _Timer.elapsed("PI_LR_HADPROD"), _Timer.number_of_calls("PI_LR_HADPROD"));
      app_log(level, "{0}  - Scale (sp_factor):          {1:8.3f} sec  {2:4d} calls", indent, _Timer.elapsed("PI_LR_SCALE"), _Timer.number_of_calls("PI_LR_SCALE"));
    }

    private:
      nda::array<double, 1> _q_pert;
      bool _kpq_map_initialized = false;
      nda::array<int, 1> _kpq_map;   // k → k+q_pert mapping (full BZ); identity for q_pert=0
      utils::TimerManager _Timer;

      // Cached per-run setup, built lazily on the first evaluate_lr_Pi call by
      // _setup_workspace and reused across SCF iterations (comm split is
      // collective; shm windows and distributed buffers are expensive to
      // recreate). The τ-dist is fixed for the solver's lifetime, so a plain
      // _setup_done guard suffices. Member order matters: the darray buffers
      // store a pointer to _t_intra_comm, so the communicator must outlive
      // (= be declared before) them.
      using sArr_2D_t = math::shm::shared_array<nda::array_view<ComplexType, 2>>;
      bool _setup_done = false;
      std::optional<mpi3::communicator> _t_intra_comm;
      std::optional<sArr_2D_t> _sf_Rk, _sf_qR;
      nda::matrix<ComplexType> _f_minus_Rk;
      // Blocked-FFT k<->R transforms; empty (→ gemm fallback with the
      // coefficient matrices above) when the grid is not an unshifted full mesh.
      std::optional<math::fft::fft_kR_t> _fft_k, _fft_q;
      nda::array<ComplexType, 1> _phase_ipR;
      std::optional<dArr_4D_t> _dDeltaG_skPQ;
      nda::matrix<ComplexType> _fft_out;
      nda::array<ComplexType, 3> _DeltaPi_RPQ;

      /// Initialize _kpq_map from THC's mean-field k-points (lazy, once)
      void _init_kpq_map(thc_reader_t& thc);

      /// Build the cached per-run workspace (comm split, FT-coefficient windows,
      /// FFT objects, phase array, distributed buffers) on the first call.
      /// τ-dist parameters are read from dDeltaPi_tqPQ; no-op once built.
      void _setup_workspace(thc_reader_t& thc,
                            dArr_4D_t const& dDeltaPi_tqPQ, long ns);

    }; // lr_rpa_pi
  } // solvers
} // methods

#endif //COQUI_LR_RPA_PI_HPP
