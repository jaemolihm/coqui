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
#include <vector>

#include "nda/nda.hpp"
#include "utilities/mpi_context.h"
#include "utilities/element_partition.hpp"
#include "utilities/Timer.hpp"
#include "numerics/imag_axes_ft/IAFT.hpp"
#include "numerics/shared_array/nda.hpp"

namespace methods {

/**
 * Mode-pair matrices of one hessian evaluation. Every entry carries the k-weight
 * and spin factor already, so each is directly comparable with the Python
 * reference implementation.
 */
struct lr_hessian_result_t {
  nda::array<ComplexType, 2> hessian_plain;  ///< Tr(ΔH0_λ, ΔDm_p), the plain estimator
  nda::array<ComplexType, 2> hessian_sym;    ///< static2 + M2 − N, the stationary one
  nda::array<ComplexType, 2> N;              ///< ⟨K ΔX_λ, ΔX_p⟩, computed in full
  nda::array<ComplexType, 2> M2;             ///< ⟨K ΔX_λ, ΔX'_p⟩
  nda::array<ComplexType, 2> static2;        ///< Tr(ΔH0_λ, ΔDm'_p)
  double herm_plain = 0.0;   ///< ‖hessian_plain − hessian_plain†‖ / ‖hessian_plain‖
  double herm_sym = 0.0;     ///< ‖hessian_sym − hessian_sym†‖ / ‖hessian_sym‖
  double herm_N = 0.0;       ///< ‖N − N†‖ / ‖N‖  (self-adjointness of K, measured)
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
 * because the loop applies K after the Dyson solve and get_kernel_before_mixing
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
 *   hessian_sym[λ,p] = Tr(ΔH0_λ, ΔDm'_p) + M2[λ,p] − N[λ,p]
 *   N [λ,p] = Tr(ΔF_λ, ΔDm_p ) + Tr_ω(ΔΣ_λ, ΔG_p )    = ⟨K ΔX_λ, ΔX_p⟩
 *   M2[λ,p] = Tr(ΔF_λ, ΔDm'_p) + Tr_ω(ΔΣ_λ, ΔG'_p)    = ⟨K ΔX_λ, ΔX'_p⟩
 *
 * is stationary about the exact solution, so its error is quadratic. The price is
 * the one extra Dyson solve per mode. At convergence ΔX' = ΔX and the correction
 * `M2 − N + Tr(ΔH0, ΔDm') − Tr(ΔH0, ΔDm)` cancels identically, which is what
 * `assemble` reports as ‖hessian_sym − hessian_plain‖.
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
 * they are the same number, via (1/β) Σ_n A(iω_n) = −A(τ=β⁻), and `assemble`
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
 * estimator second order — hence `herm_N` below is a diagnostic of the estimator
 * and not merely of `N`.
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
 * LR driver already uses. Reads are local (both ΔG and ΔΣ are node-replicated
 * shm), the ω axis of each stored slab is whole so the IAFT transforms stay
 * rank-local, and the collectives are confined to assemble()'s accumulator
 * reductions plus the shm completions in rebuild_raw_kernel (see its COST NOTE —
 * one per τ point, not one per mode).
 *
 * ## 6. Hermiticity is measured, never assumed
 *
 * `N` and `hessian_sym` are computed in full, both triangles, and their
 * antihermitian parts are reported. `A^{−q} = [A^{q}]†` in §4 holds for a
 * perturbation with the real-potential TRS structure; a perturbation without it
 * (a random Hermitian test ΔH0, say) breaks the τ-evenness the ΔΠ→ΔW path relies
 * on, and `herm_N` is what detects that. Physical phonon perturbations have the
 * structure, so on those `herm_N` sits at the roundoff floor.
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
   * @brief Pass-1 capture of mode `p`: transform the dynamic operands to ω once
   *        and stripe-store them, together with the statics.
   *
   * No contraction happens here — the mode loop only stores, because the raw
   * (pre-mixing) ΔF/ΔΣ live in the per-solve DIIS ring and are invalidated by
   * the next mode's solve.
   *
   * @param p          - [INPUT] mode index in [0, nmodes)
   * @param sDeltaDm   - [INPUT] ΔDm_p
   * @param sDeltaH0   - [INPUT] ΔH0_p
   * @param sDeltaF    - [INPUT] raw ΔF_p (get_kernel_before_mixing must have run)
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
   * @brief Pass-2 step: put the stored raw ΔF_p / ΔΣ_p back into shm, so the
   *        extra Dyson solve can be handed ΔV = ΔH0 + ΔF + ΔΣ.
   *
   * ΔΣ(τ) is rebuilt by a rank-local `w_to_tau` of this rank's ω slab followed
   * by one completion collective per τ point; the τ axis is never split, which
   * is why the partition runs over the (s,k,i,j) elements only.
   *
   * COST NOTE. That is `nt` internode allgathervs per mode (~100 at production),
   * serialized on the node root, where a single ΔΣ-wide completion was budgeted.
   * The total bytes are the same and it is a no-op on one node
   * (`complete_node_slices` returns immediately for n_nodes <= 1), so nothing
   * below np ~ 100 on a single node measures it at all: read `LR_HESS_REBUILD` from
   * a genuine multi-node run before quoting an overhead for this feature. A
   * single collective would need the store transposed to (element, τ), which
   * would in turn split the ω axis the IAFT transforms need whole.
   *
   * Collective on mpi->comm. The destinations are the mode loop's shm arrays,
   * free scratch after the last checkpoint dump.
   */
  void rebuild_raw_kernel(long p,
                          sArray_t<Array_view_4D_t>& sDeltaF_out,
                          sArray_t<Array_view_5D_t>* sDeltaSigma_out);

  /**
   * @brief Pass-2 step: accumulate everything the improved solution contributes.
   *
   * `M2(:,λ,p)` and the `Tr(ΔF_λ, ΔDm'_p)` / `Tr(ΔH0_λ, ΔDm'_p)` parts are
   * accumulated for every λ against this one p, so ΔG'_p / ΔDm'_p never persist.
   *
   * @param p        - [INPUT] mode index
   * @param sDeltaDm - [INPUT] ΔDm'_p
   * @param sDeltaG  - [INPUT] ΔG'_p, null for a Σ-free kernel
   */
  void set_improved(long p,
                    sArray_t<Array_view_4D_t> const& sDeltaDm,
                    sArray_t<Array_view_5D_t> const* sDeltaG);

  /**
   * @brief Finish: `N` over all mode pairs, reduce, apply the Matsubara tail and
   *        the prefactors, assemble `hessian_sym` and measure the diagnostics.
   *
   * Collective on mpi->comm; every rank returns the same matrices (the tail is
   * replicated, not root-only).
   */
  lr_hessian_result_t assemble();

  /// Report (verbosity 2) the pass-1/pass-2 clocks owned by this object: the
  /// stores, the shm rebuild, the pair contractions and the Matsubara tail. The
  /// K_pert refresh is lr_driver's clock, and the extra Dyson is the caller's,
  /// since the caller drives lr_driver::dyson() itself; both appear in
  /// lr_driver::print_hessian_timers().
  void print_timers();

private:
  /// Σ_e w_e conj(a_e) b_e over this rank's elements, `b` already weighted.
  ComplexType trace_static_local(nda::array<ComplexType, 1> const& a,
                         nda::array<ComplexType, 1> const& b) const;

  /// acc(n) += Σ_e conj(A[refl(n), e]) B[n, e], with the w_k folded into B.
  void trace_matsubara_local(nda::array<ComplexType, 2> const& Aw,
                  nda::array<ComplexType, 2> const& Bw,
                  nda::array_view<ComplexType, 1> acc) const;

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
  /// of assemble() so that reads as the equation.
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
  std::vector<nda::array<ComplexType, 1>> _dDm1;  ///< ΔDm_p, weighted

  // Mode-pair accumulators, one pair of arrays per TERM of the functional. Each
  // term is Tr(static) + Tr_ω(dynamic); the two halves are accumulated apart
  // because the ω half needs the Matsubara tail before it can be summed. All are
  // reduced once, in assemble().
  //
  //   H_plain = plain
  //   H_sym   = static2 + M2 − N
  //
  nda::array<ComplexType, 2> _plain_stat;              ///< Tr(ΔH0_λ, ΔDm_p)
  nda::array<ComplexType, 2> _static2_stat;            ///< Tr(ΔH0_λ, ΔDm'_p)
  nda::array<ComplexType, 2> _N_stat;                  ///< Tr(ΔF_λ, ΔDm_p)
  nda::array<ComplexType, 3> _N_mats;                  ///< Tr_ω(ΔΣ_λ, ΔG_p)
  nda::array<ComplexType, 2> _M2_stat;                 ///< Tr(ΔF_λ, ΔDm'_p)
  nda::array<ComplexType, 3> _M2_mats;                 ///< Tr_ω(ΔΣ_λ, ΔG'_p)

  std::vector<bool> _stored, _improved;
  /// assemble() reduces the accumulators in place, so it is single-shot.
  bool _assembled = false;
  utils::TimerManager _Timer;
};

} // namespace methods

#endif // COQUI_LR_HESSIAN_HPP
