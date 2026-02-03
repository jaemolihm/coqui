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
#include "methods/HF/lr_hf.hpp"
#include "methods/ERI/detail/concepts.hpp"

namespace methods {

/**
 * @class lr_driver
 * @brief Driver for self-consistent Linear Response Hartree-Fock (LR-HF) calculations
 *
 * This class coordinates the LR-HF SCF loop:
 *
 *   ΔH0 → ΔG → ΔDm → ΔF → ΔG → ... (iterate until convergence)
 *
 * The algorithm is:
 * 1. Initialize: ΔF = 0, Δμ = 0
 * 2. For iter = 1, ..., max_iter:
 *    a. Solve LR Dyson: ΔG = G_{k+q} · [ΔH0 + ΔF - Δμ·S] · G_k
 *    b. Compute LR density: ΔDm = -ΔG(τ=β⁻)
 *    c. If q=0: Update Δμ to enforce ΔN=0
 *    d. Compute LR Fock: ΔF = lr_hf(ΔDm)
 *    e. Check convergence: ||ΔDm_new - ΔDm_old|| < tol
 * 3. Return converged ΔG, ΔDm, ΔF, Δμ
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
   * @brief Run LR-HF SCF loop
   *
   * @param sDeltaG_tskij   - [OUTPUT] Converged LR Green's function (nt, ns, nk, nb, nb)
   * @param sDeltaDm_skij   - [OUTPUT] Converged LR density matrix (ns, nk, nb, nb)
   * @param sDeltaF_skij    - [OUTPUT] Converged LR Fock matrix (ns, nk, nb, nb)
   * @param sG_tskij        - [INPUT] Unperturbed Green's function (nt, ns, nk, nb, nb)
   * @param sDeltaH0_skij   - [INPUT] Perturbation (ns, nk, nb, nb)
   * @param thc             - [INPUT] THC ERI handler
   * @param max_iter        - [INPUT] Maximum iterations (default: 50)
   * @param tol             - [INPUT] Convergence tolerance for ||ΔDm_new - ΔDm_old|| (default: 1e-8)
   * @param fix_density     - [INPUT] If true, compute Δμ to enforce ΔN=0 (default: true)
   * @return Tuple of (number of iterations, final Δμ)
   */
  template<THC_ERI THC_t>
  std::tuple<int, double> run_lr_hf(
      sArray_t<Array_view_5D_t>& sDeltaG_tskij,
      sArray_t<Array_view_4D_t>& sDeltaDm_skij,
      sArray_t<Array_view_4D_t>& sDeltaF_skij,
      const sArray_t<Array_view_5D_t>& sG_tskij,
      const sArray_t<Array_view_4D_t>& sDeltaH0_skij,
      THC_t& thc,
      int max_iter = 50,
      double tol = 1e-8,
      bool fix_density = true);

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

  int _nts;
  int _ns;
  int _nkpts;
  int _nkpts_ibz;
  int _nbnd;

  utils::TimerManager _Timer;
};

} // namespace methods

#endif // COQUI_LR_DRIVER_HPP
