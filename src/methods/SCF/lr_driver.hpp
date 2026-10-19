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

#include <optional>
#include <vector>

#include "utilities/Timer.hpp"
#include "IO/app_loggers.h"

#include "utilities/mpi_context.h"
#include "mean_field/MF.hpp"
#include "numerics/imag_axes_ft/IAFT.hpp"
#include "numerics/shared_array/nda.hpp"
#include "methods/SCF/simple_dyson.h"
#include "methods/SCF/lr_dyson.hpp"
#include "methods/SCF/lr_diis.hpp"
#include "methods/SCF/lr_krylov.hpp"
#include "methods/SCF/lr_ibc.hpp"
#include "methods/HF/lr_hf.hpp"
#include "methods/GW/lr_gw.hpp"
#include "methods/scr_coulomb/lr_rpa_pi.hpp"
#include "methods/scr_coulomb/lr_scr_coulomb_t.hpp"
#include "methods/SCF/qp_params_t.h"
#include "methods/embedding/dc_response.hpp"

namespace methods {

class thc_reader_t;

/**
 * The kernel components K = δ(ΔF + ΔΣ)/δΔG applied by the LR SCF loop.
 *
 * The named methods form a nested ladder
 *
 *   none ⊂ Hartree {H} ⊂ HF {H,X} ⊂ GW0 {H,X,Σ1} ⊂ GW {H,X,Σ1,Σ2}
 *
 * with Σ1 = -ΔG⊙W_c (label "dGW") and Σ2 = -G⊙ΔW (label "GdW"). Both the
 * operator's kernel and the GCR preconditioner's are masks on this ladder.
 */
struct lr_kernel_spec {
  bool hartree = false;
  bool exchange = false;
  bool sigma_dG_W = false;   // Σ1 = -ΔG ⊙ W_c, label "dGW"
  bool sigma_G_dW = false;   // Σ2 = -G  ⊙ ΔW,  label "GdW"

  bool has_F() const { return hartree || exchange; }
  bool has_sigma() const { return sigma_dG_W || sigma_G_dW; }
  bool empty() const { return !has_F() && !has_sigma(); }

  /// Component list for logging, e.g. "H, X, dGW" ("-" when empty).
  std::string to_string() const;
};

/**
 * Expand a method name on the ladder ("none", "Hartree", "HF", "GW0", "GW")
 * into its component mask. Single source of truth for the `method` and
 * `precond_method` inputs.
 */
lr_kernel_spec kernel_spec_from_method(std::string const& name);

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

/// Per-node footprint of the accelerator's history (lr_diis ring or lr_gcr basis):
/// n_vec vectors of the quantities the accelerator actually stores.
struct lr_diis_hist_t {
  long depth = 0;     ///< subspace depth / basis capacity; 0 = accelerator off
  long n_vec = 0;     ///< stored vectors, each of n_F ΔF + n_Sigma ΔΣ
  long n_F = 0;       ///< # of ΔF-sized (ns,nk,nb,nb) quantities per vector
  long n_Sigma = 0;   ///< # of ΔΣ-sized (nt,ns,nk,nb,nb) quantities per vector
};

/**
 * Embedding data of an LR run: the projector C and the aux-basis projector B,
 * from which the driver forms δG^loc (by the caller, from the returned ΔG) and
 * δW^loc (after every ΔW solve of the kernel). With dc, the kernel also subtracts
 * the linearized double counting. Plain data; the projection math is
 * methods::local_projection and the double counting methods::lr_dc_*. The k+Q map
 * is lr_dyson::kpq_map().
 */
struct embedding_info {
  nda::array<ComplexType, 5> C_skIai;       ///< projector, MF k order (full BZ)
  std::vector<nda::range> W_rng;            ///< band window of the projector
  nda::array_view<ComplexType, 5> B_qIPab;  ///< bosonic projector B^q (node-shared view)

  /// Ground-state factors of the double counting of a GW+EDMFT ground state.
  struct dc_t {
    nda::array<ComplexType, 5> Gloc_tsIab;  ///< G^loc(τ) (nt, ns, 1, n, n)
    nda::array<ComplexType, 5> Wloc_pb;     ///< W^loc(τ) (nt_half, n, n, n, n), product basis
    dc_cells_t cells;
  };
  /// Set for a GW+EDMFT ground state: the kernel then includes the linearized double counting.
  std::optional<dc_t> dc;
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
  /// The operator's kernel K_total, applied every iteration.
  lr_kernel_spec kernel{};
  /// LR-DFT: add the semilocal xc kernel to the direct channel, i.e. use
  /// (V + Vxc)(q) in ΔJ. Requires kernel.hartree and a THC carrying Vxc;
  /// rejected together with kernel.exchange.
  bool include_xc = false;
  /// HSEX: evaluate the exchange channel with the statically screened kernel
  /// V + W_c(iν=0) instead of the bare V, i.e.
  ///   ΔF_x + ΔΣ_SEX = -ΔDm ⊙ W(iν=0).
  /// A kernel substitution inside the existing exchange path, not a new
  /// self-energy channel: no τ axis, no ΔΣ buffer, one aux<->primary transform
  /// pair per iteration exactly as bare-exchange LR-HF has. Requires
  /// kernel.exchange and a W_c on the ω axis handed to lr_setup.
  ///
  /// A flag on the exchange channel rather than a component of lr_kernel_spec:
  /// X_sex is not on the none ⊂ H ⊂ HF ⊂ GW0 ⊂ GW ladder.
  /// The aux-basis arrays carry no G=0 component, so the head of the contracted
  /// kernel enters as a Madelung term; W(iν=0)'s is -madelung·ε⁻¹(iν=0) against
  /// bare exchange's -madelung·1. It is taken from eps_inv_head_w, the same
  /// head the GW Σ uses, so exchange and correlation are treated consistently.
  bool exchange_static_W = false;

  // --- SCF control ---
  int max_iter = 1;               ///< 1 = one-shot
  /// DIIS/damping: on the successive-iterate differences of ΔDm, ΔF, ΔΣ.
  /// GCR: on the residual ||g(x) - x|| of the iterate, the concatenation
  /// (ΔF, ΔΣ) (ΔF alone for a Σ-free kernel).
  double tol = 1e-8;
  bool fix_density = false;       ///< compute Δμ to enforce ΔN = 0
  /// Whether the checkpoint reads sDeltaG_tskij after the solve. The hessian path
  /// reads it regardless when the kernel has a Σ (see lr_solve_one).
  bool save_DeltaG = true;
  lr_iter_params iter_params{};   ///< damping / DIIS

  // --- GCR preconditioner ---
  /// Σ-free kernel K_pc whose resolvent z = (I - K_pc Pχ)⁻¹ r preconditions GCR
  /// on the full equation; empty = no preconditioner. Changes the convergence
  /// rate, not the answer. Requires iter_params.alg == "GCR" and a Σ kernel.
  /// K_pc never takes the static inputs (ΔV_qPQ, Dm_ab, include_xc) or the IBC
  /// correction: it must be linear in the residual.
  lr_kernel_spec precond_kernel{};
  /// K_pc's exchange contracts V + W_c(iν=0) (HSEX), as exchange_static_W does
  /// for the operator.
  bool precond_hsex = false;

  // --- Screened interaction ---
  /// Inverse dielectric head on the τ axis; required when the kernel has a Σ.
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
  /// Embedding data (nullptr: none). The driver then also forms δW^loc; see embedding_info.
  const embedding_info* embedding = nullptr;

  // --- Output selection ---
  /// Store ΔΣ term 2 = -G⊙ΔW in its own array as well as in the total (one-shot
  /// G0W0).
  bool split_sigma_terms = false;
  /// Gather the unperturbed V_HF in the aux basis during the IBC build.
  bool keep_F_PQ = false;
  /// Evaluate the free-energy hessian through the variationally-stationary
  /// (quadratic-error) functional as well as the plain contraction, for a batch of
  /// this many perturbations; 0 leaves it off. Opt-in: it costs one extra Dyson
  /// solve and one kernel evaluation per perturbation, plus two striped ω stores.
  ///
  /// A count rather than a flag because print_memory_estimate sizes the ω stores
  /// from it. See lr_hessian.hpp for the functional and its conventions.
  long hessian_nmodes = 0;

  bool has_deltax() const { return sDeltaX_left && sDeltaX_right; }
  bool hessian() const { return hessian_nmodes > 0; }
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
  // Concrete distributed-array types of W and the cached aux-basis operands. The
  // LR path is instantiated for THC + host mpi3 only (see lr_gw /
  // lr_scr_coulomb_t), so naming them concretely here costs no generality.
  using dArr_3D_t = memory::darray_t<nda::array<ComplexType, 3>, mpi3::communicator>;
  using dArr_4D_t = memory::darray_t<nda::array<ComplexType, 4>, mpi3::communicator>;
  using dArr_5D_t = memory::darray_t<nda::array<ComplexType, 5>, mpi3::communicator>;

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
   * (kernel, split_sigma_terms, qp_static), the previous-iterate sizes, the DIIS
   * capacity — so every lr_solve_one must be handed the same `p`. `dW_wqPQ_in` is
   * consumed: it becomes dW_full(iω) and dW_c(t,R,P,Q) in place, so the caller
   * must not use it afterwards.
   *
   * **q_vec is fixed for the lifetime of the driver.** _lr_hf, the lr_gw /
   * lr_rpa_pi / lr_scr_coulomb_t solvers and their kpq maps are all latched to
   * the constructor's q, and none of them re-checks it; a batch spanning several
   * q-points would silently reuse the wrong maps.
   */
  void lr_setup(
      const sArray_t<Array_view_5D_t>& sG_tskij,
      thc_reader_t& thc,
      dArr_4D_t* dW_wqPQ_in,
      const lr_params& p);

  /**
   * @brief Run the LR SCF loop for one perturbation. Requires a prior lr_setup.
   *
   * `p` must be the one lr_setup was given: it selects which solvers and buffers
   * this call reads, and those were allocated by lr_setup.
   *
   * Zeros the per-perturbation state (ΔF, ΔΣ, ΔV_QPGW and the accelerator
   * subspace) on entry, so the result depends only on ΔH0 and not on
   * whichever perturbation ran before it. No clock is reset: every timer
   * accumulates over the whole call, so the table printed after the last
   * perturbation is the cost of the entire batch and each intermediate one is a
   * running total. The LR_DRIVER_SETUP_* clocks likewise stay as measured, the
   * setup being paid once for all perturbations.
   *
   * The output arrays may be the same ones on every call — they are fully
   * overwritten. qp mode requires sDeltaVcorr_skij: ΔV_QPGW is iterated in it.
   *
   * sDeltaG_tskij is the one exception: it holds the ΔG(τ) of the returned ΔDm
   * only if p.save_DeltaG is set or p.hessian() and the kernel has a Σ (the
   * hessian's Σ refresh reads it).
   *
   * @return Tuple of (number of iterations, final Δμ)
   */
  std::tuple<int, double> lr_solve_one(
      sArray_t<Array_view_5D_t>& sDeltaG_tskij,
      sArray_t<Array_view_4D_t>& sDeltaDm_skij,
      sArray_t<Array_view_4D_t>& sDeltaF_skij,
      sArray_t<Array_view_5D_t>* sDeltaSigma_tskij,
      const sArray_t<Array_view_5D_t>& sG_tskij,
      const sArray_t<Array_view_4D_t>& sDeltaH0_skij,
      thc_reader_t& thc,
      const lr_params& p,
      sArray_t<Array_view_5D_t>* sDeltaSigma_term2_tskij = nullptr,
      sArray_t<Array_view_4D_t>* sDeltaVcorr_skij = nullptr,
      nda::array<ComplexType, 4>* DeltaF_ibc_out = nullptr,
      nda::array<ComplexType, 4>* F_PQ_out = nullptr,
      nda::array<ComplexType, 4>* DeltaF_PQ_out = nullptr);

  /**
   * @brief The run's LR Dyson solver.
   *
   * Exposed for the stationary hessian functional, which needs ONE more solve on
   * a caller-supplied ΔV = ΔH0 + ΔF_raw + ΔΣ_raw, and must use *this* object: the
   * same operator with the same cached setup the original solves applied.
   *
   * `lr_hessian_t::evaluate` drives it — `solve_lr_dyson(...)` and, only when
   * the functional's Matsubara term needs ΔG(τ), `materialize_DeltaG_tau(...)`. Note
   * that `fix_density` must be the flag the original solves used: with it the solve
   * recomputes Δμ self-consistently from this ΔV, which keeps the constrained bubble
   * P_fd = P − u u†/⟨S,u⟩ — still self-adjoint — that they applied. lr_dyson
   * takes no Δμ input (it always solves at Δμ = 0 and shifts afterwards), so
   * passing the flag is the whole of it.
   */
  lr_dyson& dyson() {
    utils::check(_setup_done, "lr_driver::dyson: call lr_setup first.");
    return _lr_dyson;
  }

  /**
   * Estimate and report (verbosity 1 summary, verbosity 2 breakdown) the
   * per-node memory footprint of the large LR arrays: the node-replicated
   * shared band-basis arrays (~ nk·nt·nb²) and the comm-distributed aux-basis
   * arrays (~ nk·nt·NP²), the striped previous-iterate/DIIS history, and the
   * per-iteration transients. Called once at the top of lr_setup so the layout
   * can be inspected before the arrays allocate.
   *
   * `n_sigma_prev` counts the ΔΣ-sized striped previous iterates.
   *
   * `inner_hist` is the DIIS history (or GCR basis) of the SCF loop. Each
   * subspace entry holds a trial AND a residual vector, so a depth-d history is
   * 2d arrays striped over the global comm — at the default depth this dominates
   * every other LR array at production sizes, which is why it is listed rather
   * than left implicit.
   *
   * `precond` marks a preconditioned GCR run, which keeps one more ΔΣ-sized
   * shared array (the gathered residual's ΔΣ part).
   *
   * `need_Delta_mu` marks the fix_density, q=Γ path — the only one that allocates
   * lr_dyson's Δμ response dG/dμ(τ), a second ΔG(τ)-sized distributed array.
   *
   * `exchange_static_W` marks the HSEX kernel, which keeps W_c(iν=0) resident.
   *
   * `hessian_nmodes` is the perturbation batch of the stationary hessian estimator
   * (0 = off). Its two striped ω stores scale with it and are the largest thing the
   * feature allocates, so they are listed here with everything else rather than
   * reported separately by lr_hessian_t.
   */
  void print_memory_estimate(long NP, bool any_Sigma, bool needs_dW,
                             long n_sigma_prev = 0,
                             lr_diis_hist_t inner_hist = {},
                             bool precond = false,
                             bool need_Delta_mu = false,
                             bool exchange_static_W = false,
                             long hessian_nmodes = 0);

  /**
   * Report (verbosity 2) the MPI distribution (proc-grid) each family of large
   * LR arrays uses, so the estimate above can be cross-checked against the
   * actual layouts. Printed right after print_memory_estimate.
   */
  void print_distribution_summary(long NP, bool any_Sigma, bool needs_dW);

  void print_setup_timers();

  /**
   * Final hierarchical timer report, printed once after the LR SCF loop.
   * Each "LR_* (total)" line is followed by the corresponding solver's
   * subclocks (indented). The Pi/W/Sigma solvers are passed in as pointers
   * (null = solver not used, subclocks skipped).
   *
   * The report has the same lines for every kernel. Split-output mode evaluates
   * the G·ΔW term on its own lr_gw (`gw_solver_term2`) against its own clock keys;
   * both are summed into the single Σ line and Σ table, so the report stays
   * comparable line by line with a fused run's.
   */
  void print_timers(solvers::lr_rpa_pi* pi_solver = nullptr,
                    solvers::lr_scr_coulomb_t* scr_solver = nullptr,
                    solvers::lr_gw* gw_solver = nullptr,
                    solvers::lr_gw* gw_solver_term2 = nullptr);

  // Accessors
  bool is_q_gamma() const { return _lr_dyson.is_q_gamma(); }
  /// ε⁻¹(iν=0), the HSEX exchange Madelung scale. Only meaningful when
  /// exchange_static_W is set.
  double hsex_head_factor() const { return _hsex_head_factor; }

  /// δW^loc(τ) (nt_half, n, n, n, n) of the last kernel evaluation of the current lr_solve_one
  /// (lr_params::embedding); empty if it ran none.
  const std::optional<nda::array<ComplexType, 5>>& DeltaWloc() const { return _DeltaWloc_tabcd; }

private:
  /**
   * Turn the caller's W_c(iω) into the two operands the SCF loop reads every
   * iteration: W_full(iω) for the ΔW Dyson (gw_full only) and W_c(t,R,P,Q) for
   * ΔΣ = −ΔG⊙W_c. Consumes dW_wqPQ_in. See the definition for the ordering
   * constraint between them.
   */
  void lr_setup_W(dArr_4D_t* dW_wqPQ_in, thc_reader_t& thc, bool gw_full,
                  solvers::lr_scr_coulomb_t* lr_scr,
                  std::optional<dArr_4D_t>& opt_dW_full_wqPQ,
                  std::optional<dArr_4D_t>& opt_dW_tRPQ);

  /**
   * Apply one whole kernel — the operator's p.kernel or the preconditioner's
   * p.precond_kernel — to the supplied ΔDm / ΔG. Both branches are optional — a
   * kernel carrying only ΔF does no Σ work, and vice versa.
   *
   * @param precond         - [INPUT]  apply K_pc (on _lr_hf_pc, without the
   *                                   static inputs) instead of the operator
   * @param sDeltaF_out     - [OUTPUT] this kernel's ΔF, OVERWRITTEN (not
   *                                   accumulated). Untouched when the kernel
   *                                   carries no ΔF.
   * @param pDeltaSigma_out - [OUTPUT] this kernel's ΔΣ, overwritten; null
   *                                   evaluates the ΔF branch alone.
   * @param sDeltaDm_skij   - [INPUT]  ΔDm the ΔF branch contracts
   * @param sDeltaG_tskij   - [INPUT]  ΔG the Σ branch contracts
   * @param sG_tskij        - [INPUT]  unperturbed G, for ΔΠ and term 2
   * @param sDeltaSigma_term2_tskij - [OUTPUT] ΔΣ term 2 alone; non-null iff
   *                                   p.split_sigma_terms
   * @param DeltaF_PQ_out   - [OUTPUT] optional root-only aux-basis ΔF_PQ
   *                                   (see lr_hf::evaluate)
   */
  void apply_kernel(bool precond,
                    sArray_t<Array_view_4D_t>& sDeltaF_out,
                    sArray_t<Array_view_5D_t>* pDeltaSigma_out,
                    const sArray_t<Array_view_4D_t>& sDeltaDm_skij,
                    const sArray_t<Array_view_5D_t>& sDeltaG_tskij,
                    const sArray_t<Array_view_5D_t>& sG_tskij,
                    thc_reader_t& thc,
                    const lr_params& p,
                    sArray_t<Array_view_5D_t>* sDeltaSigma_term2_tskij = nullptr,
                    nda::array<ComplexType, 4>* DeltaF_PQ_out = nullptr);

  /**
   * The GW self-energy branch of apply_kernel, operator only: ΔΠ -> ΔW -> ΔΣ,
   * into `sSigma_tskij_out`, overwriting it. Each divergence correction is applied
   * by the evaluator that owns the term it corrects, so passing the overlap and
   * the heads is all this has to do.
   *
   * Split out from apply_kernel because it is long, not because it is a separate
   * concept: the static ΔF branch is one lr_hf::evaluate call with none of this
   * pipeline behind it.
   */
  void apply_kernel_gw(sArray_t<Array_view_5D_t>& sSigma_tskij_out,
                       const sArray_t<Array_view_5D_t>& sDeltaG_tskij,
                       const sArray_t<Array_view_5D_t>& sG_tskij,
                       thc_reader_t& thc,
                       const lr_params& p,
                       sArray_t<Array_view_5D_t>* sDeltaSigma_term2_tskij);

  /**
   * The GCR preconditioner z = P_c⁻¹ r. `r` and `z` are this rank's slices of
   * the concatenated (ΔF, ΔΣ) residual and direction, as lr_gcr holds them.
   *
   * Solves z_F = r_F + K_pc[Pχ(z_F, r_Σ)] for the ΔF part and passes ΔΣ through,
   * z_Σ = r_Σ: one full Dyson pass gives the ΔDm of r_Σ, then each sweep is a
   * ΔDm-only Dyson pass on r_F + y_F, one K_pc evaluation and one step of the
   * ΔF-only _pc_gcr on y_F = z_F - r_F. The sweeps stop at
   * ‖r_in‖ ≤ pc_rel_tol·‖r_in,0‖ or after pc_max_sweeps; the outer GCR is
   * flexible, so the inner accuracy changes the rate only. The _pc_gcr basis is
   * recycled across applies (reset_rhs): its operator I - K_pc Pχ is the same
   * on every call. K_pc runs without the static inputs, so the inner problem has
   * no constant term.
   *
   * Collective on comm.
   */
  void precond_apply(nda::array<ComplexType, 1> const& r, nda::array<ComplexType, 1>& z,
                     const sArray_t<Array_view_5D_t>& sG_tskij, thc_reader_t& thc,
                     const lr_params& p);
  static constexpr double pc_rel_tol = 0.01;
  static constexpr int pc_max_sweeps = 10;

  /// The HSEX kernel V + W_c(iν=0) lr_hf contracts. Empty unless the operator or
  /// the preconditioner is HSEX.
  std::optional<solvers::lr_hf::hsex_kernel_t> hsex_kernel();

  simple_dyson& _dyson;
  std::shared_ptr<mpi_context_t> _mpi;
  const mf::MF* _MF;

  lr_dyson _lr_dyson;

  // --- Solvers, built once by lr_setup and reused by every lr_solve_one. Each
  //     latches the perturbation q and a cache keyed on its usage — the exchange
  //     kernel for lr_hf, the (term1, term2) combination for lr_gw. The
  //     preconditioner has its own lr_hf, since its exchange kernel may differ
  //     from the operator's.
  //     Declared before the arrays below: darray_t stores a raw communicator_t*,
  //     so communicator-owning objects must outlive the arrays built on them.
  std::unique_ptr<solvers::lr_hf> _lr_hf;        // ΔF of the operator, and the ΔF_PQ output
  std::unique_ptr<solvers::lr_hf> _lr_hf_pc;     // ΔF of the preconditioner kernel
  std::unique_ptr<solvers::lr_gw> _lr_gw;        // ΔΣ (term 1 in split-output mode)
  std::unique_ptr<solvers::lr_gw> _lr_gw2;       // ΔΣ term 2, split-output path only
  std::unique_ptr<solvers::lr_rpa_pi> _lr_pi;
  std::unique_ptr<solvers::lr_scr_coulomb_t> _lr_scr;
  /// DIIS, when iter_params.alg == "DIIS" and the kernel is not empty.
  std::unique_ptr<lr_diis> _lr_diis;
  /// GCR in place of _lr_diis when iter_params.alg == "GCR" and the kernel is
  /// not empty.
  std::unique_ptr<lr_gcr> _lr_gcr;
  /// The preconditioner's ΔF-only inner GCR, preconditioned runs only.
  std::unique_ptr<lr_gcr> _pc_gcr;

  // --- Cached operands (constant across perturbations and SCF iterations) ---
  // sG_wskij must be a member, not a local: lr_dyson caches dN/dμ against the
  // *address* of this array, so a per-call local would silently invalidate it.
  std::optional<sArray_t<Array_view_5D_t>> _sG_wskij;
  std::optional<dArr_4D_t> _opt_dW_full_wqPQ;    // W_full(iω), q-dist
  std::optional<dArr_4D_t> _opt_dW_tRPQ;         // W_c(t,R,P,Q), τ-dist
  std::optional<dArr_3D_t> _opt_dWc0_qPQ;        // W_c(iν=0)(q,P,Q), HSEX kernel
  double _hsex_head_factor = 1.0;                // ... and its q→0 Madelung scale
  std::optional<dArr_5D_t> _opt_dG_tsRPQ, _opt_dG_mtau_tsRPQ;  // G^R(τ)/G^R(β−τ)
  std::optional<lr_ibc_DeltaX> _opt_ibc;

  /**
   * One iterated quantity (ΔDm, ΔF, ΔΣ or ΔV_QPGW), striped over the global
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
    /// This rank's slice of `A`, flattened — what the save, mixing and norms use.
    auto slice(auto&& A) const {
      return nda::reshape(A, std::array<long, 1>{n_flat})(nda::range(i0, i1));
    }
  };

  // --- Per-perturbation buffers: allocated once. Each is either zeroed at the top
  //     of lr_solve_one or written before it is read, so mode m's solve cannot
  //     see mode m−1's state.
  utils::part_map _pmap;
  lr_iterate_history _DeltaDm, _DeltaF, _DeltaSigma, _DeltaVcorr;   // ΔDm, ΔF, ΔΣ(iω), ΔV_QPGW

  // --- ΔDm-only Dyson pass (Σ-free operator kernel).
  /// The ΔF the last ΔDm-only pass was fed. The ΔG(τ) of that ΔDm is formed from
  /// it, since the caller's ΔF has been mixed by then.
  std::optional<sArray_t<Array_view_4D_t>> _sDeltaF_dyson_in;
  /// δW^loc(τ) of the last kernel evaluation of the current lr_solve_one.
  std::optional<nda::array<ComplexType, 5>> _DeltaWloc_tabcd;

  // --- Preconditioner buffers (preconditioned runs only): the gathered residual
  //     (r_F, r_Σ), the ΔDm of r_Σ, the inner iterate y_F and its ΔDm.
  std::optional<sArray_t<Array_view_4D_t>> _sDeltaF_pc_r, _sDeltaF_pc_y;
  std::optional<sArray_t<Array_view_5D_t>> _sDeltaSigma_r;
  std::optional<sArray_t<Array_view_4D_t>> _sDeltaDm_pc, _sDeltaDm_pc_y;
  /// Sweeps made by the last precond_apply, for the iteration log.
  int _pc_sweeps = 0;

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
