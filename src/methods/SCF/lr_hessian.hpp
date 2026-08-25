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


#ifndef COQUI_LR_HESSIAN_HPP
#define COQUI_LR_HESSIAN_HPP

#include <memory>
#include <optional>
#include <vector>

#include "nda/nda.hpp"
#include "utilities/mpi_context.h"
#include "utilities/element_partition.hpp"
#include "utilities/Timer.hpp"
#include "numerics/imag_axes_ft/IAFT.hpp"
#include "numerics/shared_array/nda.hpp"
#include "methods/SCF/lr_dyson.hpp"

namespace methods {

/**
 * Mode-pair matrices of one hessian evaluation. Every entry carries the k-weight
 * and spin factor already, so each is directly comparable with the Python
 * reference implementation.
 */
struct lr_hessian_result_t {
  nda::array<ComplexType, 2> hessian_plain;  ///< Tr(ΔH0_λ, ΔDm_p), the plain estimator
  nda::array<ComplexType, 2> hessian_sym;    ///< static' + M' − M, the stationary one
  nda::array<ComplexType, 2> M;              ///< ⟨K ΔX_λ, ΔX_p⟩, computed in full
  nda::array<ComplexType, 2> M_prime;        ///< ⟨K ΔX_λ, ΔX'_p⟩
  nda::array<ComplexType, 2> static_prime;   ///< Tr(ΔH0_λ, ΔDm'_p)
  double herm_plain = 0.0;   ///< ‖hessian_plain − hessian_plain†‖ / ‖hessian_plain‖
  double herm_sym = 0.0;     ///< ‖hessian_sym − hessian_sym†‖ / ‖hessian_sym‖
  double herm_M = 0.0;       ///< ‖M − M†‖ / ‖M‖  (self-adjointness of K, measured)
  /// Δμ of each mode's extra Dyson solve, as that solve recomputed it. Zero
  /// throughout unless the run is fix_density at q = Γ.
  nda::array<double, 1> Delta_mu_improved;
};

/**
 * @class lr_hessian_t
 * @brief Variationally-stationary (quadratic-error) estimator of the free-energy
 *        hessian.
 *
 * ## 1. The quantity
 *
 * The hessian of the free energy with respect to two perturbations λ, p is
 *
 *   H[λ,p] = Tr(ΔH0_λ, ΔDm_p)
 *
 * and `hessian_plain` is exactly that. It is LINEAR in the error of ΔDm, so a
 * truncated LR solve pollutes it at first order.
 *
 * ## 2. What the LR solve returns, and what it does not satisfy
 *
 * One lr_solve_one gives `(ΔG, ΔDm, ΔF, ΔΣ)`, where ΔF/ΔΣ are the RAW (pre-mixing)
 * kernel output of the final iteration. That set satisfies the kernel relation
 * exactly,
 *
 *   (ΔF, ΔΣ) = K(ΔDm, ΔG)                                        [holds]
 *
 * because the loop applies K after the Dyson solve and get_full_kernel_result
 * hands back the value before mixing touched it. What it does NOT satisfy is the
 * Dyson equation,
 *
 *   (ΔDm, ΔG) = D[ΔH0 + ΔF + ΔΣ]                                 [fails by the
 *                                                                 truncation error]
 *
 * since the loop stopped on a tolerance rather than at the fixed point. The whole
 * construction below exists to be insensitive, to first order, to exactly that
 * failure.
 *
 * ## 3. The stationary form
 *
 * Write ΔX = (ΔDm, ΔG) and let `ΔX' = (ΔDm', ΔG')` be the Dyson-postprocessed
 * result: ONE more Dyson solve on the potential the returned solution implies,
 *
 *   ΔV = ΔH0 + ΔF + ΔΣ,        ΔX' = D[ΔV].
 *
 * Then
 *
 *   hessian_sym[λ,p] = Tr(ΔH0_λ, ΔDm'_p) + M'[λ,p] − M[λ,p]
 *   M [λ,p] = Tr(ΔF_λ, ΔDm_p ) + Tr_ω(ΔΣ_λ, ΔG_p )    = ⟨K ΔX_λ, ΔX_p⟩
 *   M'[λ,p] = Tr(ΔF_λ, ΔDm'_p) + Tr_ω(ΔΣ_λ, ΔG'_p)    = ⟨K ΔX_λ, ΔX'_p⟩
 *
 * is stationary about the exact solution, so its error is quadratic. The price is
 * the one extra Dyson solve per mode. At convergence ΔX' = ΔX and the correction
 * `M' − M + Tr(ΔH0, ΔDm') − Tr(ΔH0, ΔDm)` cancels identically, which is what
 * `evaluate` reports as ‖hessian_sym − hessian_plain‖.
 *
 * ## 4. The two traces, and why the dynamic one carries a conj and a −iω_n
 *
 * Start from the plain frequency trace, which has neither:
 *
 *   Tr[A B] = (1/β) Σ_n Σ_{ij} A_ij(iω_n) B_ji(iω_n).
 *
 * The operands here are RESPONSES at the perturbation wavevector q, and the
 * hessian pairs mode λ against mode p through the (−q) partner of λ. For a real
 * perturbation that partner is the dagger,
 *
 *   A^{−q}(τ) = [A^{q}(τ)]†,      hence   A^{−q}(iω_n) = [A^{q}(−iω_n)]†,
 *
 * the second form being the τ-dagger identity FT[A(τ)†](iω_n) = A(−iω_n)†.
 * Substituting that for the left operand turns the plain trace into
 *
 *   Tr[A^{−q} B^{q}] = (1/β) Σ_n Σ_{ij} conj(A_ji(−iω_n)) B_ji(iω_n),
 *
 * which is what the code evaluates. Two consequences worth naming, because both
 * look like bugs otherwise: the conj and the reflected frequency index come from
 * that substitution, not from an inner-product convention imposed by hand; and NO
 * orbital transpose survives, since summing conj(A_ji) B_ji over all (i,j) is the
 * elementwise Frobenius pairing. So, with `refl(n)` the iω_n → −iω_n permutation
 * of the fermionic grid:
 *
 *   Tr  (A,B) = spin · Σ_{s,k} w_k Σ_{ij} conj(A[s,k,i,j]) B[s,k,i,j]
 *   c(iω_n)   = Σ_{s,k,i,j} w_k conj(A[refl(n),s,k,i,j]) B[n,s,k,i,j]
 *   Tr_ω(A,B) = spin · (1/β) Σ_n c(iω_n) = −spin · c(τ = β⁻)
 *
 * The single minus sign belongs to the τ = β⁻ form, not to the (1/β) Σ_n one —
 * they are the same number, via (1/β) Σ_n A(iω_n) = −A(τ=β⁻), and `trace_matsubara`
 * evaluates the second by the first.
 *
 * No k → k+q map appears, and should not. At finite q every operand is stored at
 * index k with the SAME leg convention — lr_dyson builds ΔG = G_{k+q} · X · G_k at
 * index k, so ΔH0, ΔF, ΔΣ and ΔG all carry legs (k+q, k). Matching inner legs then
 * forces the pairing to be A^{−q}(k+q) against B^{q}(k), and A^{−q}(k+q) =
 * [A^{q}(k)]† — so both operands are read at the SAME stored index k and there is
 * no shift to apply.
 *
 * The k-sum itself is the driver's IBZ convention, not something this contraction
 * establishes: everything is carried on nkpts_ibz with the unperturbed k_weight,
 * and w_k is the weight of k. At finite q the little group of q is smaller than
 * the full point group, so that reduction is the driver's assumption to justify
 * (compute_lr_Nelec makes the same one). Here it only fixes where the weights go:
 * into the right operand, once, at store time.
 *
 * This pairing is the one under which both the bubble P (ω-local) and the kernel
 * K (τ-local) are self-adjoint, which is the same property that makes the
 * estimator second order — hence `herm_M` below is a diagnostic of the estimator
 * and not merely of `M`.
 *
 * A static left operand is n-independent under `refl`, so its share of every
 * ⟨ΔΣ,·⟩ collapses onto `Tr` against ΔDm = −ΔG(β⁻). ΔF is therefore never fed
 * through the fermionic FT — a constant is not antiperiodic and has no fermionic
 * representation.
 *
 * ## 5. Distribution
 *
 * Everything is striped over the *global* communicator by a contiguous element
 * slice of the flattened (s,k,i,j) band array, the `utils::part_map` partition the
 * LR driver already uses. Reads are local (both ΔG and ΔΣ are node-replicated shm)
 * and the ω axis of each stored slab is whole, so the IAFT transforms stay
 * rank-local.
 *
 * The collectives are: one all_reduce per mode PAIR inside trace_matsubara (the
 * striped c(iω_n) has to be summed before the FT that closes the ω sum), the
 * reduction of the four static accumulators at the end of evaluate(), and the shm
 * completions
 * in rebuild_raw_kernel (see its COST NOTE — one per τ point, not one per mode).
 * The per-pair reduction is the deliberate cost of `Tr_ω` being a function that
 * returns a number; it is 2·nmodes² small collectives, so a large mode batch pays
 * for it in latency rather than in bytes.
 *
 * ## 6. Hermiticity is measured, never assumed
 *
 * `M` and `hessian_sym` are computed in full, both triangles, and their
 * antihermitian parts are reported. `A^{−q} = [A^{q}]†` in §4 holds for a
 * perturbation with the real-potential TRS structure; a perturbation without it
 * (a random Hermitian test ΔH0, say) breaks the τ-evenness the ΔΠ→ΔW path relies
 * on, and `herm_M` is what detects that. Physical phonon perturbations have the
 * structure, so on those `herm_M` sits at the roundoff floor.
 *
 * ## 7. Storing, then evaluating
 *
 * No trace is taken while the caller's mode loop runs. `store_mode` only packs one
 * mode's five operands into the striped stores; every contraction happens after the
 * loop. It has to be that way round, because a mode's data does not survive the
 * next one: the raw pre-mixing ΔF/ΔΣ live in the inner accelerator's ring, which
 * lr_solve_one resets per solve, and ΔDm/ΔG live in shm windows the next solve
 * overwrites in place. Storing is a copy-out, not a first half of the arithmetic.
 *
 * Evaluation is then all of `evaluate`: one loop over p that solves ΔX'_p and
 * takes column p of all four terms against every stored λ, followed by the
 * reduction and the combination. The equation appears in exactly one place.
 *
 * Why the traces are inside that loop rather than in one double loop after it: only
 * the PRIMED right operands force it. ΔDm'_p / ΔG'_p are dropped as soon as mode p
 * is done, so their traces have to be taken while they exist. Keeping them for every
 * mode instead would add a THIRD ω store beside `_Sw` and `_Gw` — +50% on the
 * largest allocation the feature makes. The unprimed operands are in the stores and
 * could have been contracted at any later point; they ride along in the same loop
 * because splitting them out bought nothing but a second copy of the equation.
 *
 * ## 8. Entry points
 *
 * Four, and no more: the constructor sizes the stores, `store_mode` copies one
 * mode out of the caller's mode loop, `evaluate` computes everything, and
 * `print_timers` reports. Everything else is private vocabulary — packing, the two
 * traces, the shm rebuild.
 */
class lr_hessian_t {
public:
  using mpi_context_t = utils::mpi_context_t<mpi3::communicator>;
  template<nda::Array Array_base_t>
  using sArray_t = math::shm::shared_array<Array_base_t>;
  using Array_view_4D_t = nda::array_view<ComplexType, 4>;
  using Array_view_5D_t = nda::array_view<ComplexType, 5>;

  /**
   * @param mpi         - [INPUT] MPI context; the stores are striped over mpi->comm
   * @param FT          - [INPUT] IAFT of the run (kept by reference, must outlive this)
   * @param k_weight    - [INPUT] w_k over the IBZ, summing to 1
   * @param spin_factor - [INPUT] 2 for a spin-unpolarized scalar run, 1 otherwise
   * @param nmodes      - [INPUT] perturbations solved in this batch (D7: npert)
   * @param has_sigma   - [INPUT] the kernel carries a dynamic ΔΣ, so `Tr_ω` is live
   * @param ns, nk_ibz, nbnd - [INPUT] band-array dimensions
   */
  lr_hessian_t(std::shared_ptr<mpi_context_t> mpi,
                        imag_axes_ft::IAFT const& FT,
                        nda::array<double, 1> const& k_weight,
                        double spin_factor, long nmodes, bool has_sigma,
                        long ns, long nk_ibz, long nbnd);

  lr_hessian_t(lr_hessian_t const&) = delete;
  lr_hessian_t& operator=(lr_hessian_t const&) = delete;

  /**
   * @brief Store mode `p`: transform the dynamic operands to ω once and
   *        stripe-store them, together with the statics.
   *
   * A copy-out, not a stage of the arithmetic: no trace is taken here. It runs in
   * the caller's mode loop because that is the only place mode p's operands exist —
   * the raw (pre-mixing) ΔF/ΔΣ live in the per-solve DIIS ring and ΔDm/ΔG in shm
   * windows, and the next mode's solve destroys both.
   *
   * @param p          - [INPUT] mode index in [0, nmodes)
   * @param sDeltaDm   - [INPUT] ΔDm_p
   * @param sDeltaH0   - [INPUT] ΔH0_p
   * @param sDeltaF    - [INPUT] raw ΔF_p (get_full_kernel_result must have run)
   * @param sDeltaSigma- [INPUT] raw ΔΣ_p, null for a Σ-free kernel
   * @param sDeltaG    - [INPUT] ΔG_p, null for a Σ-free kernel (never read then)
   */
  void store_mode(long p,
                  sArray_t<Array_view_4D_t> const& sDeltaDm,
                  sArray_t<Array_view_4D_t> const& sDeltaH0,
                  sArray_t<Array_view_4D_t> const& sDeltaF,
                  sArray_t<Array_view_5D_t> const* sDeltaSigma,
                  sArray_t<Array_view_5D_t> const* sDeltaG);

  /**
   * @brief Evaluate the whole functional: for every stored mode, solve the extra
   *        Dyson equation and contract the result against every other mode.
   *
   * One pass over the perturbations, and the only entry point that computes
   * anything. Per mode p it refills ΔH0 from the caller's stack, puts mode p's
   * stored raw ΔF/ΔΣ back into shm, solves ΔX'_p = D[ΔH0_p + ΔF_p + ΔΣ_p], and
   * takes column p of all four terms of the functional against every stored λ.
   * Then it reduces, applies the prefactors, forms hessian_sym and measures the
   * diagnostics.
   *
   * Every mode must already be stored: λ runs over all of them, so nothing can be
   * contracted until store_mode has seen the last one.
   *
   * `dyson` must be the SAME solver the original solves used — the same operator
   * with the same cached setup — and `fix_density` the flag they were given, so
   * each extra solve recomputes Δμ from its own ΔV and applies the same constrained
   * (still self-adjoint) bubble.
   *
   * The shm arrays are the caller's own mode-loop buffers, free scratch once the
   * last checkpoint dump has run: ΔF/ΔΣ are overwritten with the rebuilt raw
   * values, ΔDm/ΔG with the extra solve's output, and none of it is persisted.
   *
   * Collective on mpi->comm; single-shot (the accumulators are reduced in place),
   * and every rank returns the same matrices.
   *
   * @param dyson      - [IN/OUT] the run's LR Dyson solver
   * @param sDeltaH0   - [IN/OUT] ΔH0 window, refilled per mode from the stack
   * @param sDeltaDm   - [OUT] scratch for ΔDm'_p
   * @param sDeltaF    - [OUT] scratch for the rebuilt raw ΔF_p
   * @param sDeltaSigma- [OUT] scratch for the rebuilt raw ΔΣ_p; null if Σ-free
   * @param sDeltaG    - [OUT] scratch for ΔG'_p; null if Σ-free
   * @param DeltaH0_mskij_root - [INPUT] (nmodes,ns,nk,nb,nb) ΔH0 stack, engaged on
   *                                     the global root only
   * @param fix_density- [INPUT] the flag the original solves used
   */
  lr_hessian_result_t evaluate(
      lr_dyson& dyson,
      sArray_t<Array_view_4D_t>& sDeltaH0,
      sArray_t<Array_view_4D_t>& sDeltaDm,
      sArray_t<Array_view_4D_t>& sDeltaF,
      sArray_t<Array_view_5D_t>* sDeltaSigma,
      sArray_t<Array_view_5D_t>* sDeltaG,
      std::optional<nda::array<ComplexType, 5>> const& DeltaH0_mskij_root,
      bool fix_density);

  /// Report (verbosity 2) the whole hessian timing table: the stores, the shm
  /// rebuild, the extra Dyson, the pair contractions and the Matsubara tail. The
  /// K_pert refresh is the one clock this object does not own — it is billed inside
  /// lr_driver::get_full_kernel_result — so it is passed in from
  /// lr_driver::hessian_refresh_sec() rather than reported separately.
  void print_timers(double pert_refresh_sec, int pert_refresh_calls);

private:
  /**
   * @brief Put the stored raw ΔF_p / ΔΣ_p back into shm, so the extra Dyson solve
   *        can be handed ΔV = ΔH0 + ΔF + ΔΣ.
   *
   * ΔΣ(τ) is rebuilt by a rank-local `w_to_tau` of this rank's ω slab followed by
   * one completion collective per τ point; the τ axis is never split, which is why
   * the partition runs over the (s,k,i,j) elements only.
   *
   * COST NOTE. That is `nt` internode allgathervs per mode (~100 at production),
   * serialized on the node root, where a single ΔΣ-wide completion was budgeted.
   * The total bytes are the same and it is a no-op on one node
   * (`complete_node_slices` returns immediately for n_nodes <= 1), so nothing below
   * np ~ 100 on a single node measures it at all: read `LR_HESS_REBUILD` from a
   * genuine multi-node run before quoting an overhead for this feature. A single
   * collective would need the store transposed to (element, τ), which would in turn
   * split the ω axis the IAFT transforms need whole.
   */
  void rebuild_raw_kernel(long p,
                          sArray_t<Array_view_4D_t>& sDeltaF_out,
                          sArray_t<Array_view_5D_t>* sDeltaSigma_out);

  /// Σ_e w_e conj(a_e) b_e over this rank's elements, `b` already weighted.
  ComplexType trace_static_local(nda::array<ComplexType, 1> const& a,
                         nda::array<ComplexType, 1> const& b) const;

  /**
   * @brief Tr_ω(A,B), whole: the ω sum included, so this returns the number the
   *        equation asks for rather than a per-ω accumulator to be finished later.
   *
   *   c(iω_n) = Σ_e conj(A[refl(n), e]) B[n, e]        (w_k already folded into B)
   *   Tr_ω(A,B) = (1/β) Σ_n c(iω_n) = −c(τ = β⁻)
   *
   * COLLECTIVE on mpi->comm, unlike trace_static_local — c is striped, so it has to
   * be reduced before the FT, and doing the FT here means the reduction happens per
   * mode PAIR. That is 2·nmodes² all_reduces of nw elements rather than one of
   * nw·nmodes²: far fewer bytes, but latency in proportion to the batch, so a large
   * mode batch pays for the clean signature in collective count. The FT itself is
   * nothing either way.
   */
  ComplexType trace_matsubara(nda::array<ComplexType, 2> const& Aw,
                              nda::array<ComplexType, 2> const& Bw) const;

  /// Reduce a striped c(iω_n) and return (1/β) Σ_n c(iω_n) = −c(τ = β⁻).
  /// Collective; shared by trace_matsubara and self_pairing, which build their
  /// c(iω_n) differently but finish it identically. `c` is consumed.
  ComplexType matsubara_sum(nda::array<ComplexType, 2>& c) const;

  /// Pack this rank's element slice of a node-replicated (nt,ns,nk,nb,nb) array
  /// into `(nt, nloc)` and transform it to `(nw, nloc)`. `weighted` folds in w_k,
  /// which every RIGHT operand carries and no left operand does.
  void pack_to_omega(sArray_t<Array_view_5D_t> const& src,
                     nda::array<ComplexType, 2>& dst_w, bool weighted) const;

  /// Pack this rank's element slice of a node-replicated (ns,nk,nb,nb) array.
  nda::array<ComplexType, 1> pack_static(sArray_t<Array_view_4D_t> const& src,
                                         bool weighted) const;

  /// ⟨A,A⟩ on the stored ΔΣ of mode 0: real and positive by construction, so its
  /// sign is what a wrong pairing breaks. Collective; a pure diagnostic, kept out
  /// of the assembly so that reads as the equation.
  ComplexType self_pairing() const;

  std::shared_ptr<mpi_context_t> _mpi;
  imag_axes_ft::IAFT const* _FT;
  double _spin_factor;
  long _nmodes;
  bool _has_sigma;
  long _nt, _nw, _nk, _nbnd, _nF;
  long _i0 = 0, _i1 = 0, _nloc = 0;

  utils::part_map _pmap;
  /// iω_n → −iω_n permutation of the fermionic grid.
  nda::array<long, 1> _refl;
  /// w_k of each of this rank's elements, and the same as a complex multiplier.
  nda::array<ComplexType, 1> _wloc;

  /// Per-mode stores, striped: ω slabs of the *left* (unweighted) and *right*
  /// (w_k-weighted) dynamic operands, and the statics. Folding w_k into the right
  /// operand at store time turns every pair contraction into one flat dotc.
  std::vector<nda::array<ComplexType, 2>> _Sw;    ///< ΔΣ_λ(iω), unweighted
  std::vector<nda::array<ComplexType, 2>> _Gw;    ///< ΔG_p(iω), weighted
  std::vector<nda::array<ComplexType, 1>> _dH0;   ///< ΔH0_λ, unweighted
  std::vector<nda::array<ComplexType, 1>> _dF;    ///< ΔF_λ, unweighted
  std::vector<nda::array<ComplexType, 1>> _dDm;  ///< ΔDm_p, weighted

  // Mode-pair accumulators. Each term of the functional is Tr(static) +
  // Tr_ω(dynamic), and the two halves are held apart for one reason: they differ in
  // whether they are already reduced.
  //
  //   *_stat  are RANK-LOCAL partials (trace_static_local), summed over comm once
  //           at the end of evaluate().
  //   *_dyn   are COMPLETE (trace_matsubara reduces internally) and must NOT be
  //           reduced again — doing so would multiply them by the rank count.
  //
  //   H_plain = plain
  //   H_sym   = static' + M' − M
  //
  nda::array<ComplexType, 2> _plain_stat;         ///< Tr(ΔH0_λ, ΔDm_p)
  nda::array<ComplexType, 2> _static_prime_stat;  ///< Tr(ΔH0_λ, ΔDm'_p)
  nda::array<ComplexType, 2> _M_stat;             ///< Tr(ΔF_λ, ΔDm_p)
  nda::array<ComplexType, 2> _M_dyn;              ///< Tr_ω(ΔΣ_λ, ΔG_p)
  nda::array<ComplexType, 2> _Mp_stat;            ///< Tr(ΔF_λ, ΔDm'_p)
  nda::array<ComplexType, 2> _Mp_dyn;             ///< Tr_ω(ΔΣ_λ, ΔG'_p)

  std::vector<bool> _stored;
  /// evaluate() reduces the accumulators in place, so it is single-shot.
  bool _evaluated = false;
  utils::TimerManager _Timer;
};

} // namespace methods

#endif // COQUI_LR_HESSIAN_HPP
