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
 * with Σ1 = -ΔG⊙W_c (label "dGW") and Σ2 = -G⊙ΔW (label "GdW"). A two-step run
 * splits K into a part resummed to all orders (K_sc) and a remainder expanded
 * to finite order (K_pert = kernel(total) \ kernel(sc)), and the remainder has
 * no method name of its own — hence the component mask rather than a method
 * enum.
 */
struct lr_kernel_spec {
  bool hartree = false;
  bool exchange = false;
  bool sigma_dG_W = false;   // Σ1 = -ΔG ⊙ W_c, label "dGW"
  bool sigma_G_dW = false;   // Σ2 = -G  ⊙ ΔW,  label "GdW"

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

  /// Component list for logging, e.g. "H, X, dGW" ("-" when empty).
  std::string to_string() const;
};

/**
 * Expand a method name on the ladder ("none", "Hartree", "HF", "GW0", "GW")
 * into its component mask. Single source of truth for the `method` and
 * `two_step_inner_method` inputs.
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
 * toward the FULL `method` fixed point and `two_step_order` is an
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
  /// HSEX: evaluate the exchange channel with the statically screened kernel
  /// V + W_c(iν=0) instead of the bare V, i.e.
  ///   ΔF_x + ΔΣ_SEX = -ΔDm ⊙ W(iν=0).
  /// A kernel substitution inside the existing exchange path, not a new
  /// self-energy channel: no τ axis, no ΔΣ buffer, one aux<->primary transform
  /// pair per iteration exactly as bare-exchange LR-HF has. Requires
  /// include_exchange and a W_c on the ω axis handed to lr_setup.
  ///
  /// Deliberately a flag on the exchange channel rather than a component of
  /// lr_kernel_spec: X_sex is not on the none ⊂ H ⊂ HF ⊂ GW0 ⊂ GW ladder, so
  /// kernel_diff() cannot express the remainder K_GW \ K_HSEX (it would give
  /// {Σ1, Σ2} and silently drop the static counter-term ΔDm ⊙ W_c(0)). Keeping
  /// it off lr_kernel_spec makes that composition unrepresentable rather than
  /// wrong.
  /// The aux-basis arrays carry no G=0 component, so the head of the contracted
  /// kernel enters as a Madelung term; W(iν=0)'s is -madelung·ε⁻¹(iν=0) against
  /// bare exchange's -madelung·1. It is taken from eps_inv_head_w, the same
  /// head the GW Σ uses, so exchange and correlation are treated consistently.
  bool exchange_static_W = false;

  // --- SCF control ---
  int max_iter = 1;               ///< 1 = one-shot
  double tol = 1e-8;              ///< on ||ΔDm_new - ΔDm_old||
  bool fix_density = false;       ///< compute Δμ to enforce ΔN = 0
  /// Whether the caller will read sDeltaG_tskij after the solve (it is the
  /// checkpoint's save_DeltaG). Replicating ΔG(τ) is the most expensive step of
  /// the Dyson phase, so a Σ-free run that is not going to write it never pays:
  /// see the sDeltaG_tskij note on lr_solve_one for what that means for callers.
  bool save_DeltaG = true;
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
  /// The same head (ε⁻¹ - 1) on the ω axis, q→0 extrapolated. Required when
  /// exchange_static_W, which reads its ν = 0 component. Supplied on the ω axis
  /// rather than transformed back from eps_inv_head: for a "*_metal"
  /// div_treatment, extrapolate_eps_inv_q0 hand-sets the ν = 0 entry, and a τ
  /// round trip smears exactly that value.
  const nda::array<ComplexType, 1>* eps_inv_head_w = nullptr;
  /// Divergence treatments the unperturbed run used, read from the checkpoint.
  std::string div_treatment = "gygi";       ///< correlation / GW head
  std::string hf_div_treatment = "gygi";    ///< HF exchange Madelung term

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
  /// Evaluate the phonon C_term1 through the variationally-stationary
  /// (quadratic-error) functional as well as the plain contraction. Opt-in: it
  /// costs one extra Dyson solve per perturbation plus two striped ω stores, and
  /// it needs the RAW (pre-mixing) ΔF/ΔΣ of the final iteration, which is why
  /// lr_solve_one keeps track of whether the returned arrays are mixed.
  /// See lr_energy_curvature.hpp and docs/plan_lr_c1_quadratic_functional.md.
  bool energy_curvature = false;

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
  /// Whether the inner SCF loop mixes at all, i.e. whether the returned ΔF/ΔΣ
  /// can differ from the last kernel evaluation. Both algorithms go through
  /// lr_diis, so this is also the predicate for building its ring: damping with
  /// mixing >= 1 is the identity and needs neither.
  bool inner_mixes() const { return use_diis() || mixing() < 1.0; }
};

/**
 * The kernel split a run executes: the components the inner SCF loop resums
 * (K_sc), those applied perturbatively (K_pert), and the unions on which every
 * solver, buffer and W-operand decision is taken.
 *
 * Derived from `p` alone and recomputed identically by lr_setup, lr_solve_one and
 * the post-solve hooks, so what one allocates is exactly what the others read.
 */
struct lr_kernel_split {
  lr_kernel_spec sc{};
  lr_kernel_spec pert{};
  bool do_pert = false;            ///< a perturbative pass actually runs

  bool sc_hf = false,    pert_hf = false;
  bool sc_sigma = false, pert_sigma = false;
  bool need_hf = false;            ///< ΔF is evaluated by some channel
  bool include_gw_sigma = false;   ///< ΔΣ is evaluated by some channel
  bool gw_full = false;            ///< Σ2 = -G⊙ΔW anywhere, hence the ΔW Dyson

  /// A quantity BOTH channels contribute to. It is the only case needing
  /// per-channel buffers and a total rebuilt every inner iteration; a quantity
  /// carried by one channel is written straight into the caller's array.
  bool split_F = false, split_Sigma = false;

  bool qp_mode = false;
  bool has_Vcorr = false;     ///< the static ΔV_QPGW is the mixed/tracked quantity
  bool has_Sigma = false;     ///< the dynamic ΔΣ enters the Dyson RHS
  bool has_Sigma_sc = false;  ///< ... and the sc channel's copy is the tracked one

  /// The perturbative channel owes the static counter-term +ΔDm ⊙ W_c(0) on
  /// top of its mask, because K_sc is HSEX. See lr_params::exchange_static_W.
  bool pert_sex_counterterm = false;
};

lr_kernel_split make_kernel_split(lr_params const& p);

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
   * sDeltaG_tskij is the one exception: it is written only when something asks
   * for it, i.e. when the kernel carries a Σ or p.save_DeltaG is set. Otherwise
   * ΔG(τ) is left distributed inside lr_dyson and this array keeps whatever it
   * held before. A caller that reads it must set p.save_DeltaG.
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
   * @brief Re-apply K_pert on the ΔG the last lr_solve_one returned, overwriting
   *        the perturbative source with a fresh one.
   *
   * Split-kernel runs only; a no-op otherwise. Inside a stage the perturbative
   * source is frozen at K_pert(ΔG at stage start), so the returned ΔΣ'/ΔF' is not
   * K(ΔG') and the stationary C_term1 functional loses its second-order property
   * — the same violation as handing it the mixed iterate. One extra K_pert
   * evaluation restores ΔV' = ΔH0 + K_sc(ΔG') + K_pert(ΔG') exactly.
   *
   * Must run immediately before materialize_raw_kernel(), which reads the arrays
   * this overwrites; the ordering is checked there.
   */
  template<THC_ERI THC_t>
  void refresh_pert_on_final_G(
      const sArray_t<Array_view_4D_t>& sDeltaDm_skij,
      const sArray_t<Array_view_5D_t>& sDeltaG_tskij,
      const sArray_t<Array_view_5D_t>& sG_tskij,
      sArray_t<Array_view_4D_t>& sDeltaF_skij,
      sArray_t<Array_view_5D_t>* sDeltaSigma_tskij,
      THC_t& thc,
      const lr_params& p);

  /**
   * @brief Replace the returned (mixed) ΔF / ΔΣ with the RAW kernel output of the
   *        final iteration.
   *
   * The stationary C_term1 functional needs ΔV' = ΔH0 + K(ΔG'), which the mixing
   * block destroys: it overwrites the shared arrays in place, and the "previous
   * iterate" buffers hold the previous *already mixed* value, not the raw one.
   * The raw slice is still available in the inner accelerator's ring, which
   * stores it ahead of both the extrapolation and the damping write.
   *
   * Returns false when the arrays are already raw (no mixing ran in the final
   * iteration — a one-iteration stage, or undamped Picard), and true when they
   * were rebuilt. On the split-kernel path the raw total is raw-sc + pert, the
   * pert channel being read from the buffers refresh_pert_on_final_G() has just
   * overwritten; the outer accelerator's ring is deliberately NOT consulted,
   * since its newest slot is an extrapolate.
   *
   * Collective on comm. Call after the checkpoint dump: it changes what the
   * shared arrays hold.
   */
  bool materialize_raw_kernel(sArray_t<Array_view_4D_t>& sDeltaF_skij,
                              sArray_t<Array_view_5D_t>* sDeltaSigma_tskij,
                              const lr_params& p);

  /**
   * @brief One extra LR Dyson solve on a caller-supplied ΔV, for the stationary
   *        C_term1 functional: ΔG'' / ΔDm'' from ΔH0 + ΔF'_raw + ΔΣ'_raw.
   *
   * A thin forward to lr_dyson::solve_lr_dyson with the caller's output arrays,
   * plus the ΔG(τ) replication when the functional's Matsubara term needs it.
   *
   * `fix_density` must be the flag pass 1 used: with it the solve recomputes Δμ
   * self-consistently from this ΔV, which keeps the constrained bubble
   * P_fd = P − u u†/⟨S,u⟩ — still self-adjoint — the same operator pass 1
   * applied. lr_dyson takes no Δμ input (it always solves at Δμ = 0 and shifts
   * afterwards), so forwarding the flag is the whole of it.
   *
   * @return the Δμ of the extra solve
   */
  double c1_extra_dyson(sArray_t<Array_view_5D_t>& sDeltaG_out,
                        sArray_t<Array_view_4D_t>& sDeltaDm_out,
                        const sArray_t<Array_view_4D_t>& sDeltaH0_skij,
                        const sArray_t<Array_view_4D_t>& sDeltaF_raw_skij,
                        const sArray_t<Array_view_5D_t>* sDeltaSigma_raw_tskij,
                        bool fix_density, bool need_DeltaG);

  /**
   * Report (verbosity 2) the two energy-curvature clocks lr_driver owns: the
   * extra Dyson solve and the K_pert refresh. Separate from print_timers()
   * because that one runs at the end of every lr_solve_one, i.e. inside the mode
   * loop, where the extra Dyson has not happened yet — the feature's dominant
   * cost would read 0.000 sec in every run. Call it after the pass-2 loop.
   */
  void print_c1_timers();

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
   *
   * `need_Delta_mu` marks the fix_density, q=Γ path — the only one that allocates
   * lr_dyson's Δμ response dG/dμ(τ), a second ΔG(τ)-sized distributed array.
   *
   * `exchange_static_W` marks the HSEX kernel, which keeps W_c(iν=0) resident.
   */
  void print_memory_estimate(long NP, bool include_gw_sigma, bool gw_full,
                             std::vector<std::string> const& extra_sigma = {},
                             long n_sigma_prev = 0,
                             lr_diis_hist_t inner_hist = {},
                             lr_diis_hist_t outer_hist = {},
                             bool need_Delta_mu = false,
                             bool exchange_static_W = false);

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
  /// ε⁻¹(iν=0), the HSEX exchange Madelung scale. Only meaningful when
  /// exchange_static_W is set.
  double hsex_head_factor() const { return _hsex_head_factor; }

private:
  // Concrete distributed-array types of the cached aux-basis operands. The LR
  // path is instantiated for THC + host mpi3 only (see lr_gw / lr_scr_coulomb_t),
  // so naming them concretely here costs no generality.
  using dArr_3D_t = memory::darray_t<nda::array<ComplexType, 3>, mpi3::communicator>;
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

  /**
   * Evaluate the Σ components of `lr_kernel` into `sSigma_tskij_out`, overwriting
   * it. Each divergence correction is applied by the evaluator that owns the term
   * it corrects, so passing the overlap and the heads is all this has to do.
   *
   * The static ΔF is NOT here: it is one lr_hf::evaluate call with no ΔP/ΔW
   * pipeline behind it, so K_sc inlines it in the SCF loop and K_pert takes it in
   * apply_pert_kernel. This routine is the Σ pipeline (ΔP → ΔW → ΔΣ) shared by
   * the two channels.
   *
   * A method rather than a lambda inside lr_solve_one because the post-solve
   * K_pert refresh has to run the same evaluation from outside the SCF loop.
   * `clk_*` name the timer regions of the channel being evaluated: the two
   * channels run the same evaluators, and the cost argument for a split kernel is
   * exactly the sc/pert breakdown.
   */
  template<THC_ERI THC_t>
  void eval_sigma_channel(lr_kernel_spec const& lr_kernel,
                          solvers::lr_gw& gw_solver,
                          sArray_t<Array_view_5D_t>& sSigma_tskij_out,
                          const lr_ibc_DeltaX* ibc_ptr,
                          const char* clk_pi, const char* clk_w, const char* clk_sigma,
                          const sArray_t<Array_view_5D_t>& sDeltaG_tskij,
                          const sArray_t<Array_view_5D_t>& sG_tskij,
                          THC_t& thc,
                          const lr_params& p,
                          sArray_t<Array_view_5D_t>* sDeltaSigma_term2_tskij);

  /**
   * One whole K_pert evaluation on the supplied ΔDm / ΔG, writing the perturbative
   * source in place: the static ΔF branch (including the HSEX counter-term) plus
   * eval_sigma_channel for the Σ branch. Shared by the in-loop stage boundary and
   * the post-solve refresh so the kernel call exists exactly once.
   */
  template<THC_ERI THC_t>
  void apply_pert_kernel(lr_kernel_split const& k,
                         sArray_t<Array_view_4D_t>& sDeltaF_pert_skij,
                         sArray_t<Array_view_5D_t>* pDeltaSigma_pert,
                         const sArray_t<Array_view_4D_t>& sDeltaDm_skij,
                         const sArray_t<Array_view_5D_t>& sDeltaG_tskij,
                         const sArray_t<Array_view_5D_t>& sG_tskij,
                         THC_t& thc,
                         const lr_params& p);

  /**
   * A split kernel evaluates ΔF / ΔΣ in two channels (K_sc every inner iteration,
   * K_pert only at a stage boundary) into separate buffers, so the total the Dyson
   * solve and the checkpoint consume has to be rebuilt from them: total <- sc +
   * pert. Striped over node_comm because a ΔΣ total is GBs at production sizes and
   * is rebuilt every inner iteration.
   */
  template<typename Arr_t>
  void refresh_total(Arr_t& total, Arr_t const& sc_part, Arr_t const& pert_part);

  /// Rebuild whichever of ΔF / ΔΣ both channels contribute to. A quantity carried
  /// by one channel needs nothing — that channel wrote the caller's array.
  void refresh_split_totals(lr_kernel_split const& k,
                            sArray_t<Array_view_4D_t>& sDeltaF_skij,
                            sArray_t<Array_view_5D_t>* sDeltaSigma_tskij);

  /**
   * The HSEX kernel lr_hf contracts: V + W_c(iν=0) for the self-consistent
   * channel, −W_c(iν=0) for the counter-term a split run's remainder owes. Their
   * heads ε⁻¹(0) and 1 − ε⁻¹(0) sum back to bare exchange's 1. Empty unless the
   * run is HSEX.
   */
  std::optional<solvers::lr_hf::hsex_kernel_t> hsex_kernel(bool counter_term);

  simple_dyson& _dyson;
  std::shared_ptr<mpi_context_t> _mpi;
  const mf::MF* _MF;

  lr_dyson _lr_dyson;

  // --- Solvers, built once by lr_setup and reused by every lr_solve_one. Each
  //     latches the perturbation q and a cache keyed on its usage — the exchange
  //     kernel for lr_hf, the (term1, term2) combination for lr_gw — which is why
  //     the split-kernel path needs two of each.
  //     Declared before the arrays below: darray_t stores a raw communicator_t*,
  //     so communicator-owning objects must outlive the arrays built on them.
  std::unique_ptr<solvers::lr_hf> _lr_hf;        // ΔF of K_sc, and the total ΔF_PQ output
  std::unique_ptr<solvers::lr_hf> _lr_hf_pert;   // ΔF of K_pert, split-kernel path only
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
  std::optional<dArr_3D_t> _opt_dWc0_qPQ;        // W_c(iν=0)(q,P,Q), HSEX kernel
  double _hsex_head_factor = 1.0;                // ... and its q→0 Madelung scale
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

  // --- State of the last lr_solve_one, read by the energy-curvature hooks.
  /// The returned ΔF/ΔΣ went through the mixing block in the final iteration,
  /// hence are the mixed iterate rather than the raw kernel output.
  bool _mixed_last_iter = false;
  /// This rank's `_pmap` slice of the raw (pre-mixing) ΔF_sc / ΔΣ_sc of the most
  /// recent mixed iteration, copied out of the accelerator's ring the moment it
  /// is stored. Reading the ring later cannot work: it is reset at every
  /// split-kernel stage boundary, so a solve whose budget ran out on one would
  /// have nothing to read. Allocated only when lr_params::energy_curvature is
  /// set, so the ordinary path neither copies nor allocates.
  nda::array<ComplexType, 1> _raw_F_slice, _raw_S_slice;
  bool _raw_slice_captured = false;
  /// refresh_pert_on_final_G() has run since the last lr_solve_one, so the pert
  /// buffers hold fresh K_pert(ΔG'). Cleared by materialize_raw_kernel().
  bool _pert_refreshed_post_solve = false;

  int _nts;
  int _ns;
  int _nkpts;
  int _nkpts_ibz;
  int _nbnd;

  utils::TimerManager _Timer;
};

} // namespace methods

#endif // COQUI_LR_DRIVER_HPP
