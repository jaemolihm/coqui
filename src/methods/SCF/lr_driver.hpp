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
 * LR-qpGW static-map inputs. When passed to lr_setup (non-null), the driver runs
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
 * Acceleration of the OUTER (perturbative-source) iteration of a split-kernel
 * run. The outer sequence is a Picard iteration on an affine map,
 *
 *   S_0 = 0,   S_{m+1} = K_pert[ (1 - χ₀K_sc)⁻¹ χ₀ (ΔH0 + S_m) ],
 *
 * so the same accelerator the inner SCF uses applies to it verbatim. The outer
 * accelerator is a separate object with its own subspace, history and warmup —
 * nothing is shared with the inner `lr_diis`.
 *
 * With the defaults (alg "damping", tol = 0) nothing is allocated and no new
 * code runs: the outer loop is a plain Neumann series and `two_step_order`
 * counts truncation orders.
 *
 * With acceleration on, ORDER COUNTING IS ABANDONED: the source is a
 * combination of previous sources, so the result is an accelerated iterate
 * toward the FULL two_step_pert_method fixed point and `two_step_order` is an
 * iteration cap, not a truncation order.
 */
struct lr_outer_accel_params {
  /// alg, subspace depth and warmup of the OUTER accelerator. "damping" means
  /// no acceleration; `mixing` only damps the accelerator's own warmup steps.
  lr_iter_params iter;
  /// > 0 switches the outer loop from "run exactly pert_order stages" to
  /// "iterate until the stage-to-stage ‖ΔDm‖ change is below tol, capped at
  /// pert_order stages".
  double tol = 0.0;
  /// History vectors required before the outer DIIS extrapolates (see lr_diis).
  /// 2 puts the first extrapolation at the second outer step.
  size_t min_subsp = 2;

  /// True when the outer loop is accelerated or tolerance-driven, i.e. when the
  /// run is no longer an order-`pert_order` truncation. Single source of truth
  /// for the buffers, the logging and the checkpoint provenance fields.
  bool active() const { return iter.alg == "DIIS" || tol > 0.0; }
};

/// Per-node footprint of one lr_diis history: 2 vectors (trial + residual) per
/// subspace entry, of the quantities the accelerator actually stores.
struct lr_diis_hist_t {
  long depth = 0;     ///< max_subsp_size; 0 = no history (accelerator off)
  long n_F = 0;       ///< # of ΔF-sized (ns,nk,nb,nb) quantities stored per entry
  long n_Sigma = 0;   ///< # of ΔΣ-sized (nt,ns,nk,nb,nb) quantities per entry
};

/**
 * Everything an LR solve needs that does not depend on the perturbation ΔH0:
 * which terms are active, how the SCF iterates, and the optional inputs shared
 * by every perturbation at this q.
 *
 * Caller-owned: lr_setup and lr_solve_one take it by reference and keep no copy,
 * so the struct and every pointer in it must outlive the last lr_solve_one call.
 * The same instance must be handed to both — lr_setup allocates against it.
 *
 * The derived predicates below are deliberately functions rather than stored
 * flags — one source of truth, and no way for a cached copy to drift from the
 * field it was computed from.
 */
struct lr_params {
  using sArray_4D_t = math::shm::shared_array<nda::array_view<ComplexType, 4>>;

  // --- Active terms ---
  bool include_hartree = false;   ///< ΔJ in the SCF loop
  bool include_exchange = false;  ///< ΔK in the SCF loop
  lr_gw_update_mode gw_mode = lr_gw_update_mode::none;
  /// LR-DFT: add the semilocal xc kernel to the direct channel, i.e. use
  /// (V + Vxc)(q) in ΔJ. Requires include_hartree and a THC carrying Vxc;
  /// rejected together with include_exchange.
  bool include_xc = false;

  // --- SCF control ---
  int max_iter = 1;               ///< 1 = one-shot
  double tol = 1e-8;              ///< on ||ΔDm_new - ΔDm_old||
  bool fix_density = false;       ///< compute Δμ to enforce ΔN = 0
  lr_iter_params iter_params{};   ///< damping / DIIS

  // --- Split-kernel (two-step) schedule ---
  /// Kernel components resummed self-consistently by the inner SCF loop. Left
  /// empty it is filled from the active terms above, i.e. the single-kernel path.
  lr_kernel_spec sc_kernel{};
  /// Kernel components applied perturbatively, K_pert = kernel(total) \ sc_kernel.
  /// Empty for a plain single-kernel run.
  lr_kernel_spec pert_kernel{};
  /// Truncation order n of the K_pert expansion. 0 (or an empty pert_kernel)
  /// runs the sc kernel alone. Costs n K_pert evaluations, each on a converged
  /// inner K_sc solve; max_iter counts total inner iterations across all n+1
  /// stages. With outer_accel active this is an iteration cap, not an order.
  int pert_order = 0;
  /// Acceleration of the outer (perturbative-source) iteration. Default-valued
  /// leaves the plain Neumann series untouched. Requires a split-kernel run.
  lr_outer_accel_params outer_accel{};

  /// Split-kernel run: the perturbative channel is a distinct kernel evaluated
  /// outside the SCF resummation. Single source of truth for the extra buffers,
  /// the outer loop and the checkpoint provenance fields.
  bool two_step() const { return !pert_kernel.empty() && pert_order > 0; }

  // --- Screened interaction ---
  /// Inverse dielectric head on the τ axis; required when gw_mode != none.
  const nda::array<ComplexType, 1>* eps_inv_head = nullptr;
  bool div_corr = true;                     ///< Madelung / head divergence corrections
  std::string div_treatment = "gygi";

  // --- Optional inputs ---
  const sArray_4D_t* sDeltaX_left = nullptr;   ///< δ^q X, with sDeltaX_right: IBC
  const sArray_4D_t* sDeltaX_right = nullptr;  ///< δ^{-q} X
  const nda::array<ComplexType, 4>* Dm_ab = nullptr;      ///< unperturbed Dm, for δX/δV
  const nda::array_view<ComplexType, 3>* DeltaV_qPQ = nullptr;  ///< δV^q in the aux basis
  /// Non-null puts the driver in LR-qpGW static-map mode.
  const lr_qp_static_params* qp_static = nullptr;

  // --- Output selection ---
  /// Store the two ΔΣ terms separately instead of fusing them (one-shot G0W0).
  bool split_sigma_terms = false;
  /// Gather the unperturbed V_HF in the aux basis during the IBC build.
  bool keep_F_PQ = false;

  bool need_hf() const { return include_hartree || include_exchange; }
  bool include_gw_sigma() const { return gw_mode != lr_gw_update_mode::none; }
  bool gw_full() const { return gw_mode == lr_gw_update_mode::full; }
  /// LR-qpGW: the dynamic ΔΣ(iω) is statified into a ΔV_QPGW(k) that enters the
  /// Dyson RHS in its place, and is the quantity mixed and tracked.
  bool qp_mode() const { return qp_static != nullptr; }
  /// Which quantity is mixed/tracked alongside ΔF: the static ΔV_QPGW in qp
  /// mode, the dynamic ΔΣ(iω) otherwise.
  bool has_Vcorr() const { return qp_mode(); }
  bool has_Sigma() const { return include_gw_sigma() && !qp_mode(); }
  bool has_deltax() const { return sDeltaX_left && sDeltaX_right; }
  bool use_diis() const { return iter_params.alg == "DIIS"; }
  double mixing() const { return iter_params.mixing; }
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
   * @brief Build everything an LR SCF solve needs that does not depend on ΔH0.
   *
   * The solvers, the cached W_full(iω) / dW_tRPQ / G^R / G(iω) operands, the
   * DIIS history and the previous-iterate buffers are all functions of the
   * unperturbed state and the q-vector alone, so several perturbations at the
   * same q share one setup. Nothing here is seeded from ΔH0 — the perturbation
   * enters only through the Dyson RHS — which is what makes repeated lr_solve_one
   * calls independent of each other.
   *
   * The solvers and buffers built here are shaped by `p` — which solvers exist
   * (gw_mode, split_sigma_terms, qp_static), the previous-iterate sizes, the DIIS
   * capacity — so every lr_solve_one must be handed the same `p`. `dW_wqPQ_in` is
   * consumed: it becomes dW_full(iω) and dW_c(t,R,P,Q) in place, so the caller
   * must not use it afterwards.
   *
   * **q_vec is fixed for the lifetime of the driver.** _lr_hf, the lr_gw /
   * lr_rpa_pi / lr_scr_coulomb_t solvers and their kpq maps are all latched to
   * the constructor's q, and none of them re-checks it; a batch spanning several
   * q-points would silently reuse the wrong maps.
   */
  template<THC_ERI THC_t, typename dW_t>
  void lr_setup(
      const sArray_t<Array_view_5D_t>& sG_tskij,
      THC_t& thc,
      dW_t* dW_wqPQ_in,
      const lr_params& p);

  /**
   * @brief Run the LR SCF loop for one perturbation. Requires a prior lr_setup.
   *
   * `p` must be the one lr_setup was given: it selects which solvers and buffers
   * this call reads, and those were allocated by lr_setup.
   *
   * Zeros the per-perturbation state (ΔF, ΔΣ, ΔV_QPGW, the previous iterates and
   * the DIIS subspace) on entry, so the result depends only on ΔH0 and not on
   * whichever perturbation ran before it. No clock is reset: every timer
   * accumulates over the whole call, so the table printed after the last
   * perturbation is the cost of the entire batch and each intermediate one is a
   * running total. The LR_DRIVER_SETUP_* clocks likewise stay as measured, the
   * setup being paid once for all perturbations.
   *
   * The output arrays may be the same ones on every call — they are fully
   * overwritten.
   *
   * @return Tuple of (number of iterations, final Δμ)
   */
  template<THC_ERI THC_t>
  std::tuple<int, double> lr_solve_one(
      sArray_t<Array_view_5D_t>& sDeltaG_tskij,
      sArray_t<Array_view_4D_t>& sDeltaDm_skij,
      sArray_t<Array_view_4D_t>& sDeltaF_skij,
      sArray_t<Array_view_5D_t>* sDeltaSigma_tskij,
      const sArray_t<Array_view_5D_t>& sG_tskij,
      const sArray_t<Array_view_4D_t>& sDeltaH0_skij,
      THC_t& thc,
      const lr_params& p,
      sArray_t<Array_view_5D_t>* sDeltaSigma_term2_tskij = nullptr,
      sArray_t<Array_view_4D_t>* sDeltaVcorr_out_skij = nullptr,
      nda::array<ComplexType, 4>* DeltaF_ibc_out = nullptr,
      nda::array<ComplexType, 4>* F_PQ_out = nullptr,
      nda::array<ComplexType, 4>* DeltaF_PQ_out = nullptr,
      int* n_pert_applied_out = nullptr);

  /**
   * Estimate and report (verbosity 1 summary, verbosity 2 breakdown) the
   * per-node memory footprint of the large LR arrays: the node-replicated
   * shared band-basis arrays (~ nk·nt·nb²) and the comm-distributed aux-basis
   * arrays (~ nk·nt·NP²), the striped previous-iterate/DIIS history, and the
   * per-iteration transients. Called once at the top of lr_setup so the layout
   * can be inspected before the arrays allocate.
   *
   * `extra_sigma` names the additional ΔΣ-sized shared arrays a split-kernel
   * run allocates on top of the total ΔΣ: the per-channel buffers.
   *
   * `n_sigma_prev` counts the ΔΣ-sized striped previous iterates — the inner
   * loop's tracked ΔΣ and the outer accelerator's previous source.
   *
   * `inner_hist` / `outer_hist` are the DIIS histories of the inner SCF and of
   * the outer (split-kernel source) loop. Each subspace entry holds a trial AND
   * a residual vector, so a depth-d history is 2d arrays striped over the
   * global comm — at the default inner depth this dominates every other LR
   * array at production sizes, which is why it is listed rather than left
   * implicit.
   */
  void print_memory_estimate(long NP, bool include_gw_sigma, bool gw_full,
                             std::vector<std::string> const& extra_sigma = {},
                             long n_sigma_prev = 0,
                             lr_diis_hist_t inner_hist = {},
                             lr_diis_hist_t outer_hist = {});

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
   * subclocks (indented). The Pi/W/Sigma solvers are passed in as pointers
   * (null = solver not used, subclocks skipped).
   *
   * The report has the same lines for every kernel. A split kernel evaluates ΔΣ
   * on several lr_gw instances (`gw_solver_pert` for the perturbative channel,
   * `gw_solver_term2` for a split G·ΔW term) against their own clock keys; all
   * of them are summed into the single Σ line and Σ table, so a split run's
   * report stays comparable line by line with a one-step run's.
   */
  void print_timers(solvers::lr_rpa_pi* pi_solver = nullptr,
                    solvers::lr_scr_coulomb_t* scr_solver = nullptr,
                    solvers::lr_gw* gw_solver = nullptr,
                    solvers::lr_gw* gw_solver_pert = nullptr,
                    solvers::lr_gw* gw_solver_term2 = nullptr);

  // Accessors
  const nda::array<int, 1>& kpq_map() const { return _lr_dyson.kpq_map(); }
  const nda::array<double, 1>& q_vec() const { return _lr_dyson.q_vec(); }
  bool is_q_gamma() const { return _lr_dyson.is_q_gamma(); }

private:
  // Concrete distributed-array types of the cached aux-basis operands. The LR
  // path is instantiated for THC + host mpi3 only (see lr_gw / lr_scr_coulomb_t),
  // so naming them concretely here costs no generality.
  using dArr_4D_t = memory::darray_t<nda::array<ComplexType, 4>, mpi3::communicator>;
  using dArr_5D_t = memory::darray_t<nda::array<ComplexType, 5>, mpi3::communicator>;

  /**
   * Turn the caller's W_c(iω) into the two operands the SCF loop reads every
   * iteration: W_full(iω) for the ΔW Dyson (gw_full only) and W_c(t,R,P,Q) for
   * ΔΣ = −ΔG⊙W_c. Consumes dW_wqPQ_in. See the definition for the ordering
   * constraint between them.
   */
  template<THC_ERI THC_t, typename dW_t>
  void lr_setup_W(dW_t* dW_wqPQ_in, THC_t& thc, bool gw_full,
                  solvers::lr_scr_coulomb_t* lr_scr,
                  std::optional<dW_t>& opt_dW_full_wqPQ,
                  std::optional<dW_t>& opt_dW_tRPQ);

  simple_dyson& _dyson;
  std::shared_ptr<mpi_context_t> _mpi;
  const mf::MF* _MF;

  lr_dyson _lr_dyson;

  // --- Solvers, built once by lr_setup and reused by every lr_solve_one. Each
  //     latches the perturbation q and (for lr_gw) a workspace keyed on its
  //     (term1, term2) usage, which is why the split-kernel path needs two.
  //     Declared before the arrays below: darray_t stores a raw communicator_t*,
  //     so communicator-owning objects must outlive the arrays built on them.
  std::unique_ptr<solvers::lr_hf> _lr_hf;
  std::unique_ptr<solvers::lr_gw> _lr_gw;        // ΔΣ of K_sc (fused, or term 1 when split)
  std::unique_ptr<solvers::lr_gw> _lr_gw_pert;   // ΔΣ of K_pert, split-kernel path only
  std::unique_ptr<solvers::lr_gw> _lr_gw2;       // ΔΣ term 2, split-output path only
  std::unique_ptr<solvers::lr_rpa_pi> _lr_pi;
  std::unique_ptr<solvers::lr_scr_coulomb_t> _lr_scr;
  std::unique_ptr<lr_diis> _lr_diis;
  /// Accelerator of the outer (perturbative-source) iteration. Its own subspace,
  /// history and warmup: the inner one restarts at every stage boundary while
  /// this one is keyed on the outer step index, so the two share nothing.
  std::unique_ptr<lr_diis> _outer_diis;

  // --- Cached operands (constant across perturbations and SCF iterations) ---
  // sG_wskij must be a member, not a local: lr_dyson caches dN/dμ against the
  // *address* of this array, so a per-call local would silently invalidate it.
  std::optional<sArray_t<Array_view_5D_t>> _sG_wskij;
  std::optional<dArr_4D_t> _opt_dW_full_wqPQ;    // W_full(iω), ω-side
  std::optional<dArr_4D_t> _opt_dW_tRPQ;         // W_c(t,R,P,Q), τ-dist
  std::optional<dArr_5D_t> _opt_dG_tsRPQ, _opt_dG_mtau_tsRPQ;  // G^R(τ)/G^R(β−τ)
  std::optional<lr_ibc_DeltaX> _opt_ibc;

  /**
   * One elementwise-mixed quantity (ΔF, ΔΣ or ΔV_QPGW), striped over the global
   * comm: this rank owns [i0, i1) of the flattened array and keeps only that
   * slice of the previous iterate. The whole job then stores each previous
   * iterate once instead of once per node.
   */
  struct lr_iterate_history {
    long n_flat = 0;                  ///< elements of the flattened array (0 = inactive)
    long i0 = 0, i1 = 0;              ///< this rank's slice
    nda::array<ComplexType, 1> prev;  ///< previous iterate over [i0, i1)

    void alloc(utils::part_map const& pmap, long n) {
      n_flat = n;
      std::tie(i0, i1) = pmap.my_slice(n);
      prev = nda::array<ComplexType, 1>(i1 - i0);
    }
    void zero() { prev() = ComplexType{0}; }
    /// This rank's slice of `A`, flattened — what the save, mixing and norms use.
    auto slice(auto&& A) const {
      return nda::reshape(A, std::array<long, 1>{n_flat})(nda::range(i0, i1));
    }
  };

  // --- Per-perturbation buffers: allocated once, zeroed at the top of each
  //     lr_solve_one so mode m's solve cannot see mode m−1's state.
  utils::part_map _pmap;
  std::optional<sArray_t<Array_view_4D_t>> _sDeltaDm_prev_skij;
  std::optional<sArray_t<Array_view_4D_t>> _sDeltaVcorr_skij;
  lr_iterate_history _DeltaF, _DeltaSigma, _DeltaVcorr;   // ΔF, ΔΣ(iω), ΔV_QPGW

  // --- Split-kernel (two-step) buffers. sDeltaF_skij / sDeltaSigma_tskij always
  //     hold the TOTAL (sc + pert) quantities; a per-channel buffer exists only
  //     for a quantity BOTH channels contribute to, otherwise the sole
  //     contributing channel writes the caller's array directly.
  std::optional<sArray_t<Array_view_4D_t>> _sDeltaF_sc, _sDeltaF_pert;
  std::optional<sArray_t<Array_view_5D_t>> _sDeltaSigma_sc, _sDeltaSigma_pert;
  /// Previous perturbative source, for the outer accelerator's extrapolation
  /// and its residual. Striped exactly like the inner loop's iterates.
  lr_iterate_history _DeltaF_pert, _DeltaSigma_pert;
  /// ΔDm at the previous stage boundary, whose change is the outer termination
  /// criterion. Kept whole: its norm is taken on the node_comm path.
  std::optional<sArray_t<Array_view_4D_t>> _sDeltaDm_stage_prev;

  bool _setup_done = false;

  int _nts;
  int _ns;
  int _nkpts;
  int _nkpts_ibz;
  int _nbnd;

  utils::TimerManager _Timer;
};

} // namespace methods

#endif // COQUI_LR_DRIVER_HPP
