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


#ifndef COQUI_LR_ENERGY_CURVATURE_HPP
#define COQUI_LR_ENERGY_CURVATURE_HPP

#include <memory>
#include <optional>
#include <vector>

#include "nda/nda.hpp"
#include "utilities/mpi_context.h"
#include "utilities/element_partition.hpp"
#include "utilities/Timer.hpp"
#include "numerics/imag_axes_ft/IAFT.hpp"
#include "numerics/shared_array/nda.hpp"

namespace methods {

/**
 * Mode-pair matrices of one energy-curvature evaluation. Every entry carries the
 * k-weight and spin factor already, so each is directly comparable with the
 * Python `compute_C_term1` output.
 */
struct lr_c1_result_t {
  nda::array<ComplexType, 2> C1_plain;   ///< Stat(ΔH0_λ, ΔDm'_p), the plain estimator
  nda::array<ComplexType, 2> C1_sym;     ///< static2 + M2 − N, the stationary estimator
  nda::array<ComplexType, 2> N;          ///< ⟨ΔΣ'_λ, ΔG'_p⟩, computed in full
  nda::array<ComplexType, 2> M2;         ///< ⟨ΔΣ'_λ, ΔG''_p⟩
  nda::array<ComplexType, 2> static2;    ///< Stat(ΔH0_λ, ΔDm''_p)
  double herm_plain = 0.0;               ///< ‖C1_plain − C1_plain†‖ / ‖C1_plain‖
  double herm_sym = 0.0;                 ///< ‖C1_sym − C1_sym†‖ / ‖C1_sym‖  (H1 detector)
  double herm_N = 0.0;                   ///< ‖N − N†‖ / ‖N‖  (direct H1 detector)
};

/**
 * @class lr_energy_curvature_t
 * @brief Variationally-stationary (quadratic-error) estimator of the phonon
 *        `C_term1`.
 *
 * The plain estimator `C1[λ,p] = Stat(ΔH0_λ, ΔDm'_p)` is linear in the error of
 * the truncated LR solution. Evaluating the same quantity through the stationary
 * functional
 *
 *   C1_sym[λ,p] = Stat(ΔH0_λ, ΔDm''_p) + M2[λ,p] − N[λ,p]
 *   N [λ,p] = Stat(ΔF'_λ, ΔDm'_p ) + Mats(ΔΣ'_λ, ΔG'_p )
 *   M2[λ,p] = Stat(ΔF'_λ, ΔDm''_p) + Mats(ΔΣ'_λ, ΔG''_p)
 *
 * makes the error quadratic, at the cost of one extra Dyson solve per mode.
 * `(ΔG', ΔDm', ΔF', ΔΣ')` is the truncated solve's output with ΔF'/ΔΣ' the *raw*
 * (pre-mixing) kernel output, and `(ΔG'', ΔDm'')` the extra Dyson solve on
 * `ΔV' = ΔH0 + ΔF' + ΔΣ'`.
 *
 * Two contractions, with independent normalizations:
 *   Stat(A,B) = spin · Σ_{s,k} w_k Σ_{ij} conj(A[s,k,i,j]) B[s,k,i,j]
 *   c(iω_n)   = Σ_{s,k,i,j} w_k conj(A[refl(n),s,k,i,j]) B[n,s,k,i,j]
 *   Mats(A,B) = spin · (1/β) Σ_n c(iω_n) = −spin · c(τ = β⁻)
 * with `refl(n)` the iω_n → −iω_n permutation of the fermionic grid. The single
 * minus sign belongs to the τ = β⁻ form, not to the (1/β) Σ_n one — they are the
 * same number, and `assemble` evaluates the second by the first. The `Mats`
 * pairing is the τ-domain conjugate transpose, FT[A(τ)†](iω_n) = A(−iω_n)†, which
 * is the unique form under which both the bubble P (ω-local) and the kernel K
 * (τ-local) are self-adjoint — see docs/plan_lr_c1_quadratic_functional.md, D1.
 * It needs no orbital transpose: the trace of a conjugate transpose against a
 * matrix is the elementwise Frobenius pairing. The leading −1 and the implicit
 * 1/β both come from the identity (1/β) Σ_n A(iω_n) = −A(τ=β⁻).
 *
 * A static left operand is n-independent under `refl`, so its share of every
 * ⟨ΔΣ,·⟩ collapses onto `Stat` against ΔDm = −ΔG(β⁻). ΔF is therefore never fed
 * through the fermionic FT — a constant is not antiperiodic and has no fermionic
 * representation.
 *
 * Distribution. Everything is striped over the *global* communicator by a
 * contiguous element slice of the flattened (s,k,i,j) band array, the
 * `utils::part_map` partition the LR driver already uses. Reads are local (both
 * ΔG and ΔΣ are node-replicated shm), the ω axis of each stored slab is whole so
 * the IAFT transforms stay rank-local, and the only collectives are the two
 * accumulator reductions in assemble() plus one completion per shm rebuild.
 *
 * Hermiticity of `N` and of `C1_sym` is *measured*, never assumed: both matrices
 * are computed in full, both triangles, and the residuals are correctness
 * diagnostics only.
 */
class lr_energy_curvature_t {
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
   * @param has_sigma   - [INPUT] the kernel carries a dynamic ΔΣ, so `Mats` is live
   * @param ns, nk_ibz, nbnd - [INPUT] band-array dimensions
   */
  lr_energy_curvature_t(std::shared_ptr<mpi_context_t> mpi,
                        imag_axes_ft::IAFT const& FT,
                        nda::array<double, 1> const& k_weight,
                        double spin_factor, long nmodes, bool has_sigma,
                        long ns, long nk_ibz, long nbnd);

  lr_energy_curvature_t(lr_energy_curvature_t const&) = delete;
  lr_energy_curvature_t& operator=(lr_energy_curvature_t const&) = delete;

  /**
   * @brief Pass-1 capture of mode `p`: transform the dynamic operands to ω once
   *        and stripe-store them, together with the statics.
   *
   * No contraction happens here — the mode loop only stores, because the raw
   * (pre-mixing) ΔF'/ΔΣ' live in the per-solve DIIS ring and are invalidated by
   * the next mode's solve.
   *
   * @param p          - [INPUT] mode index in [0, nmodes)
   * @param sDeltaDm   - [INPUT] ΔDm'_p
   * @param sDeltaH0   - [INPUT] ΔH0_p
   * @param sDeltaF    - [INPUT] raw ΔF'_p (materialize_raw_kernel must have run)
   * @param sDeltaSigma- [INPUT] raw ΔΣ'_p, null for a Σ-free kernel
   * @param sDeltaG    - [INPUT] ΔG'_p, null for a Σ-free kernel (never read then)
   */
  void store_mode(long p,
                  sArray_t<Array_view_4D_t> const& sDeltaDm,
                  sArray_t<Array_view_4D_t> const& sDeltaH0,
                  sArray_t<Array_view_4D_t> const& sDeltaF,
                  sArray_t<Array_view_5D_t> const* sDeltaSigma,
                  sArray_t<Array_view_5D_t> const* sDeltaG);

  /**
   * @brief Pass-2 step: put the stored raw ΔF'_p / ΔΣ'_p back into shm, so the
   *        extra Dyson solve can be handed ΔV' = ΔH0 + ΔF' + ΔΣ'.
   *
   * ΔΣ'(τ) is rebuilt by a rank-local `w_to_tau` of this rank's ω slab followed
   * by one completion collective per τ point; the τ axis is never split, which
   * is why the partition runs over the (s,k,i,j) elements only.
   *
   * COST NOTE. That is `nt` internode allgathervs per mode (~100 at production),
   * serialized on the node root, where a single ΔΣ-wide completion was budgeted.
   * The total bytes are the same and it is a no-op on one node
   * (`complete_node_slices` returns immediately for n_nodes <= 1), so nothing
   * below np ~ 100 on a single node measures it at all: read `LR_C1_REBUILD` from
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
   * `M2(:,λ,p)` and the `Stat(ΔF'_λ, ΔDm''_p)` / `Stat(ΔH0_λ, ΔDm''_p)` parts are
   * accumulated for every λ against this one p, so ΔG''_p / ΔDm''_p never persist.
   *
   * @param p        - [INPUT] mode index
   * @param sDeltaDm - [INPUT] ΔDm''_p
   * @param sDeltaG  - [INPUT] ΔG''_p, null for a Σ-free kernel
   */
  void set_improved(long p,
                    sArray_t<Array_view_4D_t> const& sDeltaDm,
                    sArray_t<Array_view_5D_t> const* sDeltaG);

  /**
   * @brief Finish: `N` over all mode pairs, reduce, apply the Matsubara tail and
   *        the prefactors, assemble `C1_sym` and measure the diagnostics.
   *
   * Collective on mpi->comm; every rank returns the same matrices (the tail is
   * replicated, not root-only).
   */
  lr_c1_result_t assemble();

  /// The D1 convention this binary implements, for the self-describing checkpoint.
  static std::string convention();

  /// Report (verbosity 2) the pass-1/pass-2 clocks owned by this object. The
  /// extra Dyson and the K_pert refresh are billed to lr_driver's own LR_C1_*
  /// clocks instead, since they run inside it.
  void print_timers();

private:
  /// Σ_e w_e conj(a_e) b_e over this rank's elements, `b` already weighted.
  ComplexType stat_local(nda::array<ComplexType, 1> const& a,
                         nda::array<ComplexType, 1> const& b) const;

  /// acc(n) += Σ_e conj(A[refl(n), e]) B[n, e], with the w_k folded into B.
  void mats_local(nda::array<ComplexType, 2> const& Aw,
                  nda::array<ComplexType, 2> const& Bw,
                  nda::array_view<ComplexType, 1> acc) const;

  /// Pack this rank's element slice of a node-replicated (nt,ns,nk,nb,nb) array
  /// into `(nt, nloc)` and transform it to `(nw, nloc)`.
  void pack_to_omega(sArray_t<Array_view_5D_t> const& src,
                     nda::array<ComplexType, 2>& dst_w) const;

  /// Pack this rank's element slice of a node-replicated (ns,nk,nb,nb) array.
  nda::array<ComplexType, 1> pack_static(sArray_t<Array_view_4D_t> const& src) const;

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
  std::vector<nda::array<ComplexType, 2>> _Sw;    ///< ΔΣ'_λ(iω), unweighted
  std::vector<nda::array<ComplexType, 2>> _Gw;    ///< ΔG'_p(iω), weighted
  std::vector<nda::array<ComplexType, 1>> _dH0;   ///< ΔH0_λ, unweighted
  std::vector<nda::array<ComplexType, 1>> _dF;    ///< ΔF'_λ, unweighted
  std::vector<nda::array<ComplexType, 1>> _dDm1;  ///< ΔDm'_p, weighted

  /// Per-ω accumulators of the two dynamic mode-pair matrices, and the static
  /// parts. Reduced once, in assemble().
  nda::array<ComplexType, 3> _accN, _accM2;
  nda::array<ComplexType, 2> _statN, _statM2, _statPlain, _statH0_2;

  std::vector<bool> _stored, _improved;
  /// assemble() reduces the accumulators in place, so it is single-shot.
  bool _assembled = false;
  utils::TimerManager _Timer;
};

} // namespace methods

#endif // COQUI_LR_ENERGY_CURVATURE_HPP
