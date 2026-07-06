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


#ifndef COQUI_LR_DRIVER_HPP
#define COQUI_LR_DRIVER_HPP

#include "utilities/Timer.hpp"
#include "IO/app_loggers.h"

#include "utilities/mpi_context.h"
#include "mean_field/MF.hpp"
#include "numerics/imag_axes_ft/IAFT.hpp"
#include "numerics/shared_array/nda.hpp"
#include "methods/SCF/simple_dyson.h"
#include "methods/SCF/lr_dyson.hpp"
#include "methods/SCF/lr_diis.hpp"
#include "methods/HF/lr_hf.hpp"
#include "methods/ERI/detail/concepts.hpp"

namespace methods {

/**
 * GW self-energy update mode for the LR SCF loop.
 *
 * none:    ΔΣ_GW = 0
 * fixed_W: ΔΣ = -ΔG ⊙ W,  ΔW = 0
 * full:    ΔΣ = -ΔG ⊙ W - G ⊙ ΔW,  ΔW = (Z+W_c) · ΔΠ · (Z+W_c)
 */
enum class lr_gw_update_mode { none, fixed_W, full };

/**
 * @class lr_driver
 * @brief Driver for self-consistent Linear Response calculations
 *
 * Coordinates the unified LR SCF loop with configurable components:
 *
 *   ΔH0 → ΔG → ΔDm → [ΔF] → [ΔΣ] → ΔG → ... (iterate until convergence)
 *
 * Subsumes LR-Dyson (one-shot, no HF/GW), LR-HF SCF (Hartree+Exchange),
 * and LR-GW+HF SCF (Hartree+Exchange+GW self-energy).
 */
class lr_driver {
public:
  using mpi_context_t = utils::mpi_context_t<mpi3::communicator>;
  template<nda::Array Array_base_t>
  using sArray_t = math::shm::shared_array<Array_base_t>;
  using Array_view_4D_t = nda::array_view<ComplexType, 4>;
  using Array_view_5D_t = nda::array_view<ComplexType, 5>;

  /**
   * @brief Constructor
   *
   * @param dyson     - [INPUT] Simple Dyson solver (provides MF, FT, H0, S)
   * @param q_vec     - [INPUT] Perturbation wavevector in crystal coordinates (3,)
   */
  lr_driver(simple_dyson& dyson, nda::array<double, 1> const& q_vec);

  lr_driver(lr_driver const&) = delete;
  lr_driver(lr_driver &&) = default;
  lr_driver & operator=(const lr_driver &) = delete;
  lr_driver & operator=(lr_driver &&) = delete;

  ~lr_driver() = default;

  /**
   * @brief Run unified LR SCF loop
   *
   * @param sDeltaG_tskij      - [OUTPUT] Converged LR Green's function (nt, ns, nk, nb, nb)
   * @param sDeltaDm_skij      - [OUTPUT] Converged LR density matrix (ns, nk, nb, nb)
   * @param sDeltaF_skij       - [OUTPUT] Converged LR Fock matrix (ns, nk, nb, nb)
   * @param sDeltaSigma_tskij  - [OUTPUT] Converged LR self-energy (nt, ns, nk, nb, nb), nullptr if not used
   * @param sG_tskij           - [INPUT] Unperturbed Green's function (nt, ns, nk, nb, nb)
   * @param sDeltaH0_skij      - [INPUT] Perturbation (ns, nk, nb, nb)
   * @param thc                - [INPUT] THC ERI handler
   * @param include_hartree    - [INPUT] Include ΔJ in SCF loop
   * @param include_exchange   - [INPUT] Include ΔK in SCF loop
   * @param gw_mode            - [INPUT] GW self-energy mode (none/fixed_W/full)
   * @param dW_qtPQ            - [INPUT] Screened interaction (nullable, required if gw_mode != none)
   * @param eps_inv_head       - [INPUT] Inverse dielectric head (nullable, required if gw_mode != none)
   * @param max_iter           - [INPUT] Maximum iterations (1 = one-shot)
   * @param tol                - [INPUT] Convergence tolerance for ||ΔDm_new - ΔDm_old||
   * @param fix_density        - [INPUT] If true, compute Δμ to enforce ΔN=0
   * @param iter_params        - [INPUT] Iteration algorithm parameters (damping/DIIS)
   * @return Tuple of (number of iterations, final Δμ)
   */
  template<THC_ERI THC_t, typename dW_t>
  std::tuple<int, double> run_lr(
      sArray_t<Array_view_5D_t>& sDeltaG_tskij,
      sArray_t<Array_view_4D_t>& sDeltaDm_skij,
      sArray_t<Array_view_4D_t>& sDeltaF_skij,
      sArray_t<Array_view_5D_t>* sDeltaSigma_tskij,
      const sArray_t<Array_view_5D_t>& sG_tskij,
      const sArray_t<Array_view_4D_t>& sDeltaH0_skij,
      THC_t& thc,
      bool include_hartree, bool include_exchange, lr_gw_update_mode gw_mode,
      dW_t* dW_qtPQ, const nda::array<ComplexType, 1>* eps_inv_head,
      int max_iter, double tol, bool fix_density,
      const lr_iter_params& iter_params,
      const sArray_t<Array_view_4D_t>* sDeltaX_left = nullptr,
      const sArray_t<Array_view_4D_t>* sDeltaX_right = nullptr,
      const nda::array<ComplexType, 4>* Dm_ab = nullptr,
      bool div_corr = true,
      const nda::array_view<ComplexType, 3>* DeltaV_qPQ = nullptr);

  void print_setup_timers();

  /// Final hierarchical timer report, printed once after the LR SCF loop.
  void print_timers();

  // Accessors
  const nda::array<int, 1>& kpq_map() const { return _lr_dyson.kpq_map(); }
  const nda::array<double, 1>& q_vec() const { return _lr_dyson.q_vec(); }
  bool is_q_gamma() const { return _lr_dyson.is_q_gamma(); }

private:
  simple_dyson& _dyson;
  std::shared_ptr<mpi_context_t> _mpi;
  const mf::MF* _MF;

  lr_dyson _lr_dyson;
  std::unique_ptr<solvers::lr_hf> _lr_hf;
  std::unique_ptr<lr_diis> _lr_diis;

  int _nts;
  int _ns;
  int _nkpts;
  int _nkpts_ibz;
  int _nbnd;

  utils::TimerManager _Timer;
};

} // namespace methods

#endif // COQUI_LR_DRIVER_HPP
