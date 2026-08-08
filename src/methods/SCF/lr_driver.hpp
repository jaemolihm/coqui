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
#include "methods/SCF/lr_ibc.hpp"
#include "methods/HF/lr_hf.hpp"
#include "methods/GW/lr_gw.hpp"
#include "methods/scr_coulomb/lr_rpa_pi.hpp"
#include "methods/scr_coulomb/lr_scr_coulomb_t.hpp"
#include "methods/ERI/detail/concepts.hpp"
#include "methods/SCF/qp_params_t.h"

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
 * The kernel components K = δ(ΔF + ΔΣ)/δΔG applied by the LR SCF loop.
 *
 * The named methods form a nested ladder
 *
 *   none ⊂ Hartree {H} ⊂ HF {H,X} ⊂ GW0 {H,X,Σ1} ⊂ GW {H,X,Σ1,Σ2}
 *
 * with Σ1 = -ΔG⊙W_c and Σ2 = -G⊙ΔW. A two-step run splits K into a part
 * resummed to all orders (K_sc) and a remainder expanded to finite order
 * (K_pert = kernel(total) \ kernel(sc)), and the remainder has no method name
 * of its own — hence the component mask rather than a method enum.
 */
struct lr_kernel_spec {
  bool hartree = false;
  bool exchange = false;
  bool sigma_dG_W = false;   // Σ1 = -ΔG ⊙ W_c
  bool sigma_G_dW = false;   // Σ2 = -G  ⊙ ΔW

  bool has_sigma() const { return sigma_dG_W || sigma_G_dW; }
  bool empty()     const { return !hartree && !exchange && !has_sigma(); }

  bool operator==(lr_kernel_spec const&) const = default;

  /// True if every component of `sub` is also in *this.
  bool contains(lr_kernel_spec const& sub) const {
    return (hartree    || !sub.hartree)    && (exchange   || !sub.exchange) &&
           (sigma_dG_W || !sub.sigma_dG_W) && (sigma_G_dW || !sub.sigma_G_dW);
  }

  /// True if any component appears in both masks.
  bool overlaps(lr_kernel_spec const& o) const {
    return (hartree && o.hartree) || (exchange && o.exchange) ||
           (sigma_dG_W && o.sigma_dG_W) || (sigma_G_dW && o.sigma_G_dW);
  }

  /// Component list for logging, e.g. "H, X, Σ1" ("-" when empty).
  std::string to_string() const;
};

/**
 * Expand a method name on the ladder ("none", "Hartree", "HF", "GW0", "GW")
 * into its component mask. Single source of truth for the `method`,
 * `two_step_sc_method` and `two_step_pert_method` inputs.
 */
lr_kernel_spec kernel_spec_from_method(std::string const& name);

/// K_pert = total \ sc, requiring sc ⊆ total.
lr_kernel_spec kernel_diff(lr_kernel_spec const& total, lr_kernel_spec const& sc);

/**
 * LR-qpGW static-map inputs. When passed to run_lr (non-null), the driver runs
 * in "qp_static_sigma" mode: after the dynamic ΔΣ(iω) is assembled it is
 * statified via lr_qp_approx into a static ΔV_QPGW(k) (frozen orbitals C/ε/μ),
 * which enters the frequency-independent one-body term of the Dyson RHS in place
 * of the dynamic ΔΣ. DIIS/damping and convergence then track ΔV_QPGW.
 */
struct lr_qp_static_params {
  qp_params_t qp_params;
  const math::shm::shared_array<nda::array_view<ComplexType, 4>>* sMO_skia = nullptr;
  const math::shm::shared_array<nda::array_view<ComplexType, 3>>* sE_ska = nullptr;
  double mu = 0.0;
};

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
   * @param sc_kernel          - [INPUT] Kernel components resummed self-consistently
   * @param pert_kernel        - [INPUT] Kernel components applied perturbatively
   *                              (K_pert = kernel(total) \ kernel(sc)). Empty for
   *                              a plain single-kernel run.
   * @param pert_order         - [INPUT] Truncation order n of the K_pert expansion.
   *                              0 (or an empty pert_kernel) runs the sc kernel
   *                              alone, i.e. exactly the single-kernel path.
   *                              Costs n K_pert evaluations, each on a converged
   *                              inner K_sc solve; max_iter counts total inner
   *                              iterations across all n+1 stages.
   * @param dW_tqPQ_in         - [INPUT] Screened interaction W_c(τ) in (t,q,P,Q)
   *                              (nullable, required if a Σ is active). Consumed:
   *                              it becomes dW_tRPQ in place, so the caller must not
   *                              use it after the call.
   * @param eps_inv_head       - [INPUT] Inverse dielectric head (nullable, required if a Σ is active)
   * @param max_iter           - [INPUT] Maximum iterations (1 = one-shot)
   * @param tol                - [INPUT] Convergence tolerance for ||ΔDm_new - ΔDm_old||
   * @param fix_density        - [INPUT] If true, compute Δμ to enforce ΔN=0
   * @param iter_params        - [INPUT] Iteration algorithm parameters (damping/DIIS)
   * @param DeltaF_ibc_out     - [OUTPUT] If non-null AND DeltaX was provided, the
   *                              precomputed IBC aux→primary correction
   *                              δX†·F_PQ·X + X†·F_PQ·δX (ns, nk_ibz, nb, nb)
   *                              is copied here before return. Otherwise left
   *                              untouched (size 0 → caller can skip persisting).
   * @param F_PQ_out           - [OUTPUT] If non-null AND DeltaX was provided, the
   *                              gathered unperturbed V_HF in aux basis
   *                              (ns, nk_ibz, NP, NP) is copied here.
   * @param DeltaF_PQ_out      - [OUTPUT] If non-null AND HF is active, the LR Fock
   *                              in aux basis at convergence (ns, nk_ibz, NP, NP)
   *                              is gathered here via one extra lr_hf::evaluate
   *                              call. Consumed, with F_PQ_out, by the Python
   *                              phonon post-processors (ΔΔF_ibc T1/T3 terms).
   * @param include_xc         - [INPUT] LR-DFT: add the semilocal xc kernel to the
   *                              direct channel, i.e. use (V + Vxc)(q) in ΔJ.
   *                              Requires include_hartree and a THC carrying Vxc;
   *                              rejected together with include_exchange.
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
      lr_kernel_spec sc_kernel, lr_kernel_spec pert_kernel, int pert_order,
      dW_t* dW_tqPQ_in, const nda::array<ComplexType, 1>* eps_inv_head,
      int max_iter, double tol, bool fix_density,
      const lr_iter_params& iter_params,
      const sArray_t<Array_view_4D_t>* sDeltaX_left = nullptr,
      const sArray_t<Array_view_4D_t>* sDeltaX_right = nullptr,
      const nda::array<ComplexType, 4>* Dm_ab = nullptr,
      bool div_corr = true,
      std::string div_treatment = "gygi",
      const nda::array_view<ComplexType, 3>* DeltaV_qPQ = nullptr,
      sArray_t<Array_view_5D_t>* sDeltaSigma_term2_tskij = nullptr,
      const lr_qp_static_params* qp_static = nullptr,
      sArray_t<Array_view_4D_t>* sDeltaVcorr_skij = nullptr,
      nda::array<ComplexType, 4>* DeltaF_ibc_out = nullptr,
      nda::array<ComplexType, 4>* F_PQ_out = nullptr,
      nda::array<ComplexType, 4>* DeltaF_PQ_out = nullptr,
      bool include_xc = false);

  /**
   * Estimate and report (verbosity 1 summary, verbosity 2 breakdown) the
   * per-node memory footprint of the large LR arrays: the node-replicated
   * shared band-basis arrays (~ nk·nt·nb²) and the comm-distributed aux-basis
   * arrays (~ nk·nt·NP²), the striped previous-iterate/DIIS history, and the
   * per-iteration transients. Called once at the top of run_lr so the layout
   * can be inspected before the arrays allocate.
   *
   * `n_extra_sigma` counts the additional ΔΣ-sized shared arrays a split-kernel
   * run allocates on top of the total ΔΣ.
   */
  void print_memory_estimate(long NP, bool include_gw_sigma, bool gw_full,
                             long n_extra_sigma, bool use_diis,
                             size_t max_subsp_size);

  /**
   * Report (verbosity 2) the MPI distribution (proc-grid) each family of large
   * LR arrays uses, so the estimate above can be cross-checked against the
   * actual layouts. Printed right after print_memory_estimate.
   */
  void print_distribution_summary(long NP, bool include_gw_sigma, bool gw_full);

  void print_setup_timers();

  /**
   * Final hierarchical timer report, printed once after the LR SCF loop.
   * Each "LR_* (total)" line is followed by the corresponding solver's
   * subclocks (indented). The Pi/W/Sigma solvers live in run_lr's scope, so
   * they are passed in as pointers (null = solver not used, subclocks skipped).
   *
   * A split-kernel run has two Σ channels with their own solver and their own
   * "_PERT" clocks; `gw_solver_pert` non-null adds that second block, so the
   * cost of the perturbative kernel can be read off separately.
   */
  void print_timers(solvers::lr_rpa_pi* pi_solver = nullptr,
                    solvers::lr_scr_coulomb_t* scr_solver = nullptr,
                    solvers::lr_gw* gw_solver = nullptr,
                    solvers::lr_gw* gw_solver_pert = nullptr);

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
