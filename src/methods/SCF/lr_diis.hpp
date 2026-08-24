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

#ifndef COQUI_LR_DIIS_HPP
#define COQUI_LR_DIIS_HPP

#include <algorithm>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
#include <numeric>
#include <cmath>

#include "nda/nda.hpp"
#include "nda/blas.hpp"
#include "nda/linalg/eigenelements.hpp"
#include "IO/app_loggers.h"
#include "utilities/element_partition.hpp"

namespace methods {

/**
 * @brief Parameters for LR SCF iteration algorithm
 *
 * Mirrors the ground-state iter_alg pattern (damping vs DIIS).
 */
struct lr_iter_params {
  std::string alg = "damping";
  double mixing = 1.0;
  size_t max_subsp_size = 5;
  size_t diis_warmup = 3;
};


/**
 * @brief Standalone DIIS accelerator for linear-response SCF
 *
 * Why not reuse diis_alg<Vector> / VSpace<Vector>?
 *   The existing DIIS infrastructure (numerics/iter_scf/diis/) has a hard
 *   dependency on HDF5 file I/O: VSpace stores and retrieves every vector
 *   via write_to_file/read_from_file, and overlap(i,j) reads two vectors
 *   from disk. This design is appropriate for ground-state SCF where
 *   checkpointing is essential, but for LR the arrays are tiny (e.g. 512
 *   complex numbers for Si) and the SCF loop is tight, so disk I/O would
 *   be pure overhead. The Vector interface also requires read_from_file/
 *   write_to_file methods, and the ground-state uses a composite FockSigma
 *   vector type with commutator residuals — quite different from LR where
 *   we have a single 4D ΔF with a simple difference residual.
 *
 *   This class reuses the same B-matrix solve algorithm (eigendecomposition
 *   pseudoinverse from diis_alg.hpp::compute_coefs_c1) with in-memory storage
 *   instead of HDF5-backed VSpace.
 *
 *   SPMD / distributed: next_step_combined is called on every rank of the
 *   supplied communicator. Each rank stores and operates on only its own
 *   contiguous element-slice of every trial/residual vector (std::vector<Vec1D>),
 *   the slice being the one named by the part_map it is handed. Striped over
 *   the global communicator, the whole history is stored exactly once across the
 *   job rather than once per node. The B-matrix overlaps are formed from
 *   per-rank partial dot products combined with a single all_reduce, so every
 *   rank computes identical coefficients.
 *
 * Uses difference residual: res = ΔF_new - ΔF_prev
 */
class lr_diis {
public:
  using Array_4D = nda::array<ComplexType, 4>;
  using Array_5D = nda::array<ComplexType, 5>;
  using Vec1D    = nda::array<ComplexType, 1>;

  /**
   * @param max_subsp_size - maximum number of history vectors kept
   * @param warmup_iter    - damping-only steps before extrapolation is allowed
   * @param mixing         - damping factor used while in warmup (>= 1 is a no-op)
   * @param min_subsp      - history vectors required before extrapolating,
   *                         counted AFTER the current step is stored. 3 (the
   *                         default) puts the first extrapolation on the third
   *                         call; 2 enables a two-term (secant / Aitken) step
   *                         at the second call.
   */
  lr_diis(size_t max_subsp_size, size_t warmup_iter, double mixing,
          size_t min_subsp = 3)
      : _max_subsp_size(max_subsp_size),
        _warmup_iter(warmup_iter),
        _mixing(mixing),
        _min_subsp(min_subsp),
        _B(0, 0) {}

  /**
   * @brief Empty the subspace, keeping every history buffer allocated.
   *
   * For restarting the accelerator between independent solves — successive
   * perturbations of a batched run — without touching the storage. The history
   * is `2·S·(|ΔF| + |ΔΣ|)` complex striped over the job (hundreds of GB at
   * production sizes), so freeing and regrowing it per solve is exactly what
   * must not happen; the ring below keeps the slots and only forgets them.
   * `_B` is (S+1)² and is cheap to rebuild.
   */
  void reset() {
    _n = 0;
    _head = 0;
    _B = nda::matrix<ComplexType>(0, 0);
  }
  /**
   * @brief True when this accelerator only ever damps, never extrapolates.
   *
   * The subspace is capped at _max_subsp_size, so a _min_subsp above that makes
   * the warmup gate `_n < _min_subsp` fire on every step for the lifetime of the
   * object: every call takes the damping write. It is how a damping run is
   * expressed as an lr_diis, and it is what lets the B-matrix work be skipped.
   */
  bool is_simple_mixing() const { return _min_subsp > _max_subsp_size; }

  /**
   * @brief One combined DIIS step on (ΔF, ΔΣ), striped over `comm`.
   *
   * SPMD: called on every rank of `comm`. Each rank owns the contiguous element
   * slice of the flattened ΔF/ΔΣ named by `pmap` and stores only that slice of
   * every trial / residual vector, so the DIIS history is partitioned across
   * `comm` (no per-rank and, with the global comm, no per-node replication).
   * The B-matrix overlaps are formed from per-rank partial dot products combined
   * with a single all_reduce, so every rank holds the identical B and computes
   * identical coefficients. The mixed result is written in place into each
   * rank's slice of DeltaF / DeltaSigma; completing the (node-replicated)
   * arrays afterwards is the caller's job.
   *
   * Combined residual: B(i,j) = <res_F_i, res_F_j> + <res_Sigma_i, res_Sigma_j>.
   *
   * @param comm             - [INPUT] communicator the array work is striped over
   * @param pmap             - [INPUT] element partition over `comm`
   * @param DeltaF           - [IN/OUT] full ΔF (new on entry, this rank's slice
   *                                    mixed on exit); pass .local()
   * @param DeltaF_prev      - [INPUT]  this rank's slice of the previous ΔF
   * @param DeltaSigma       - [IN/OUT] full ΔΣ (as DeltaF); empty if no GW
   * @param DeltaSigma_prev  - [INPUT]  this rank's slice of the previous ΔΣ
   * @param iter             - [INPUT]  current iteration number (1-based)
   */
  template<typename Comm, typename FView, typename FPrev, typename SView, typename SPrev>
  void next_step_combined(Comm& comm, utils::part_map const& pmap,
                          FView DeltaF, FPrev const& DeltaF_prev,
                          SView DeltaSigma, SPrev const& DeltaSigma_prev,
                          int iter) {
    const bool has_sigma = (DeltaSigma.size() > 0);

    // This rank's contiguous slice of the flattened ΔF. The "prev" arrays are
    // already stored as exactly this slice.
    const long nF = DeltaF.size();
    auto [fF0, fF1] = pmap.my_slice(nF);
    auto F_loc      = nda::reshape(DeltaF, std::array<long, 1>{nF})(nda::range(fF0, fF1));
    auto const& F_prev_loc = DeltaF_prev;
    utils::check(static_cast<long>(F_prev_loc.size()) == fF1 - fF0,
                 "lr_diis: ΔF_prev slice size {} != partition slice size {}",
                 F_prev_loc.size(), fF1 - fF0);

    long sS0 = 0, sS1 = 0;
    const long nS = has_sigma ? static_cast<long>(DeltaSigma.size()) : 0;
    if (has_sigma) {
      std::tie(sS0, sS1) = pmap.my_slice(nS);
      utils::check(static_cast<long>(DeltaSigma_prev.size()) == sS1 - sS0,
                   "lr_diis: ΔΣ_prev slice size {} != partition slice size {}",
                   DeltaSigma_prev.size(), sS1 - sS0);
    }

    // Damping write (warmup / ill-conditioned fallback): out_slice <- mixing*new +
    // (1-mixing)*prev. mixing >= 1 leaves the arrays unchanged (out == new).
    auto write_damping = [&]() {
      if (_mixing < 1.0) {
        F_loc = _mixing * F_loc + (1.0 - _mixing) * F_prev_loc;
        if (has_sigma) {
          auto S_loc = nda::reshape(DeltaSigma, std::array<long, 1>{nS})(nda::range(sS0, sS1));
          S_loc = _mixing * S_loc + (1.0 - _mixing) * DeltaSigma_prev;
        }
      }
    };

    // Store this rank's slices into the newest ring slot, then update B (one
    // all_reduce over comm). Assigning into the slot keeps the buffers the
    // previous solve/iteration allocated: past the first max_subsp_size+1
    // iterations this loop performs no allocation at all.
    //
    // The residuals feed local_overlap, which only update_B reaches, so a
    // configuration that never extrapolates must not pay for them either: they
    // cost a full-slice subtraction and a slice of storage per iteration for a
    // matrix nothing builds.
    const bool build_res = !is_simple_mixing();
    const size_t s = push_slot();
    _xF[s] = F_loc;
    if (build_res) _resF[s] = F_loc - F_prev_loc;
    if (has_sigma) {
      auto S_loc = nda::reshape(DeltaSigma, std::array<long, 1>{nS})(nda::range(sS0, sS1));
      _xS[s] = S_loc;
      if (build_res) _resS[s] = S_loc - DeltaSigma_prev;
    }

    // Warmup gate, evaluated on the subspace INCLUDING the step just stored.
    const bool warmup =
        (iter <= static_cast<int>(_warmup_iter) + 1 || _n < _min_subsp);

    // A subspace that can never reach _min_subsp never extrapolates, so B is
    // dead work: two full-slice dotc's per history entry plus an all_reduce
    // every iteration, feeding a matrix nothing reads. That is exactly the
    // depth-1 ring a damping run builds. Any configuration that CAN extrapolate
    // takes the branch unchanged, so DIIS is untouched.
    if (!is_simple_mixing()) update_B(comm, has_sigma);

    if (_n > _max_subsp_size) purge_oldest();

    if (warmup) {
      app_log(3, "    DIIS: warmup iter {} -> damping (mixing={:.2f}, subspace={})",
              iter, _mixing, _n);
      write_damping();
      return;
    }

    auto C = compute_coefs();

    double c_norm = 0.0;
    for (long i = 0; i < C.size(); ++i) c_norm += std::abs(C(i));

    // Σ|c_i| bounds how much the extrapolation amplifies the error of the
    // vectors it combines; it is the diagnostic for an accelerator that is
    // fed noisy (e.g. only loosely converged) iterates.
    app_log(3, "    DIIS: sum|c_i| = {:.3e}", c_norm);

    if (c_norm < 1e-14) {
      app_log(2, "    DIIS: extrapolation failed (ill-conditioned B) -> fallback to damping");
      write_damping();
      return;
    }

    // Extrapolate: out_slice <- Σ_i C(i) * trial_i  (in place).
    make_linear_comb_into(F_loc, _xF, C);
    if (has_sigma) {
      auto S_loc = nda::reshape(DeltaSigma, std::array<long, 1>{nS})(nda::range(sS0, sS1));
      make_linear_comb_into(S_loc, _xS, C);
    }

    app_log(3, "    DIIS: extrapolation with subspace size {}", _n);
  }

  /**
   * @brief This rank's slice of the newest step's ΔF / ΔΣ, as it was BEFORE any
   *        mixing.
   *
   * next_step_combined stores the incoming iterate into the ring ahead of both
   * the extrapolation and the damping write, so this is the raw kernel output of
   * the step just taken — the one quantity the caller's shared arrays no longer
   * hold once mixing has run. Invalidated by reset(), and by the next
   * next_step_combined call.
   *
   * newest_xS() is meaningful only when the second slot actually carried a
   * quantity; a ΔF-only call leaves it empty.
   */
  Vec1D const& newest_xF() const {
    utils::check(_n > 0, "lr_diis::newest_xF: the subspace is empty.");
    return _xF[slot(_n - 1)];
  }
  Vec1D const& newest_xS() const {
    utils::check(_n > 0, "lr_diis::newest_xS: the subspace is empty.");
    return _xS[slot(_n - 1)];
  }

private:
  size_t _max_subsp_size;
  size_t _warmup_iter;
  double _mixing;
  size_t _min_subsp;
  // History stores only THIS rank's contiguous element-slice of each flattened
  // trial / residual vector (striped across comm), not the full arrays. The four
  // vectors are a fixed-capacity ring: `_n` entries live, chronological entry i
  // in slot `(_head + i) % capacity`. Eviction moves `_head`, so no buffer is
  // ever destroyed and no slice is ever reallocated once the ring has filled.
  std::vector<Vec1D> _xF, _resF;   // ΔF trial / residual slices
  std::vector<Vec1D> _xS, _resS;   // ΔΣ trial / residual slices
  size_t _n = 0;                   // live entries
  size_t _head = 0;                // ring origin
  nda::matrix<ComplexType> _B;

  /// Storage slot holding chronological entry `i` (0 = oldest).
  size_t slot(size_t i) const { return (_head + i) % _xF.size(); }

  /**
   * @brief Claim the slot for the newest entry, growing the ring if needed.
   *
   * The caller assigns into the four vectors at the returned slot; nda's
   * assignment resizes only when the shape differs, so the first solve sizes
   * each slice and every later one reuses the same allocation.
   */
  size_t push_slot() {
    if (_n == _xF.size()) {
      // Growth happens only before the first eviction, i.e. while the ring
      // origin is still 0 — so appending keeps the entries in order. Once the
      // subspace has overflowed once, capacity is max_subsp_size+1 forever.
      utils::check(_head == 0, "lr_diis: ring growth with _head = {} != 0", _head);
      _xF.emplace_back();
      _resF.emplace_back();
      _xS.emplace_back();
      _resS.emplace_back();
    }
    size_t s = slot(_n);
    ++_n;
    return s;
  }

  /// Local (this rank's slice) contribution to the combined overlap
  /// <res_F_i, res_F_j> + <res_Sigma_i, res_Sigma_j>, for chronological i, j.
  ComplexType local_overlap(size_t i, size_t j, bool has_sigma) const {
    const size_t si = slot(i), sj = slot(j);
    ComplexType result = 0.0;
    if (_resF[si].size() > 0) result += nda::blas::dotc(_resF[si], _resF[sj]);
    if (has_sigma && _resS[si].size() > 0) {
      result += nda::blas::dotc(_resS[si], _resS[sj]);
    }
    return result;
  }

  /**
   * @brief Rebuild last row/column of B using partial overlaps + one all_reduce.
   *
   * Each rank contributes the dot products over its slice; the new row is
   * all_reduced so every rank ends with the identical full B.
   */
  template<typename Comm>
  void update_B(Comm& comm, bool has_sigma) {
    const size_t n = _n;
    nda::matrix<ComplexType> Bnew(n, n);
    Bnew() = 0;

    const size_t n_old = n - 1;
    for (size_t i = 0; i < n_old; ++i)
      for (size_t j = 0; j < n_old; ++j)
        Bnew(i, j) = _B(i, j);

    // New row/col: per-rank partial overlaps reduced in a single all_reduce.
    nda::array<ComplexType, 1> col(n);
    col() = 0;
    for (size_t i = 0; i < n; ++i) col(i) = local_overlap(i, n - 1, has_sigma);
    comm.all_reduce_in_place_n(col.data(), static_cast<long>(n), std::plus<>{});

    for (size_t i = 0; i < n_old; ++i) {
      Bnew(i, n - 1) = col(i);
      Bnew(n - 1, i) = std::conj(col(i));
    }
    Bnew(n - 1, n - 1) = col(n - 1);

    _B = Bnew;
  }

  /**
   * @brief Forget the oldest vector and shrink the B-matrix.
   *
   * The history buffers are untouched: advancing the ring origin frees the slot
   * for the next push to overwrite. B stays indexed chronologically, so it does
   * shift — it is (S+1)² and costs nothing.
   */
  void purge_oldest() {
    // B is empty when it is never built (is_simple_mixing()); there is then
    // nothing to shift, and forming a (n-1) x (n-1) matrix from n = 0 would be
    // a negative extent.
    size_t n = _B.shape()[0];
    if (n > 0) {
      nda::matrix<ComplexType> Bnew(n - 1, n - 1);
      for (size_t i = 0; i < n - 1; ++i) {
        for (size_t j = 0; j < n - 1; ++j) {
          Bnew(i, j) = _B(i + 1, j + 1);
        }
      }
      _B = Bnew;
    }

    _head = (_head + 1) % _xF.size();
    --_n;
  }

  /**
   * @brief Compute DIIS coefficients via eigendecomposition pseudoinverse
   *
   * Algorithm from diis_alg.hpp::compute_coefs_c1():
   * 1. Jacobi-scale B (real part) to unit diagonal
   * 2. Eigendecompose and pseudoinvert with threshold
   * 3. Solve for B^{-1} * 1, undo the scaling, normalize to sum=1
   */
  nda::array<double, 1> compute_coefs() const {
    auto B_real = nda::make_regular(nda::real(_B));
    const long n = B_real.shape()[0];

    // Solve with the Jacobi-scaled Bt = D^(-1/2) B D^(-1/2), D = diag(B), so that
    // Bt_ii = 1 and Bt_ij = cos<(r_i, r_j). Exact change of variables, but it is
    // what makes the cut below meaningful: B_ii = ||r_i||^2 and DIIS residual
    // norms decay geometrically, so cond(B) is dominated by that decay while
    // cond(Bt) sees only genuine linear dependence. The scaling is undone on x.
    nda::array<double, 1> d(n), bb(n);
    for (long i = 0; i < n; ++i) {
      d(i)  = (B_real(i, i) > 0.0) ? std::sqrt(B_real(i, i)) : 1.0;
      bb(i) = 1.0 / d(i);  // rhs D^(-1/2) 1
    }
    for (long i = 0; i < n; ++i)
      for (long j = 0; j < n; ++j) B_real(i, j) /= d(i) * d(j);

    auto [eig, evecs] = nda::linalg::eigenelements(B_real);
    auto evecs_tr = nda::make_regular(nda::transpose(evecs));

    nda::matrix<double> Binv(n, n);
    nda::matrix<double> eig_inv(n, n);
    nda::matrix<double> I_tmp(n, n);
    Binv() = 0;
    eig_inv() = 0;

    auto eig_abs = nda::map([](double x) { return std::abs(x); })(eig);
    double eig_max = nda::max_element(eig);
    double eig_min_abs = nda::min_element(eig_abs);

    const double eig_thresh = 1E-14;

    // Diagnostic only: a zero smallest |eigenvalue| would print inf/nan.
    if (eig_min_abs > 0.0)
      app_log(3, "    DIIS: Jacobi-scaled B condition number = {:.2e}",
              nda::max_element(eig_abs) / eig_min_abs);
    else
      app_log(3, "    DIIS: Jacobi-scaled B is singular (smallest |eigenvalue| = 0)");

    // Pseudoinverse: only keep positive eigenvalues above eig_thresh*eig_max.
    for (long i = 0; i < eig.size(); ++i) {
      if (eig(i) > eig_thresh * eig_max) {
        eig_inv(i, i) = 1.0 / eig(i);
      }
    }

    nda::blas::gemm(evecs, eig_inv, I_tmp);
    nda::blas::gemm(I_tmp, evecs_tr, Binv);

    nda::array<double, 1> x(n);
    nda::blas::gemv(1.0, Binv, bb, 0.0, x);
    for (long i = 0; i < n; ++i) x(i) /= d(i);

    double sum = std::accumulate(x.begin(), x.end(), 0.0);
    if (std::abs(sum) < 1e-14) {
      // Ill-conditioned: return zeros to signal failure
      nda::array<double, 1> zeros(x.size());
      zeros() = 0.0;
      return zeros;
    }

    nda::array<double, 1> C(x.size());
    for (long i = 0; i < x.size(); ++i) {
      C(i) = x(i) / sum;
    }
    return C;
  }

  /**
   * @brief Write Σ_i C(i) * hist[i] into `out` (this rank's slice).
   *
   * `out` is a 1D slice view of the output array; `hist` are the matching
   * trial-vector slices, walked in chronological order to match C.
   */
  template<typename OutView>
  void make_linear_comb_into(OutView&& out, const std::vector<Vec1D>& hist,
                             const nda::array<double, 1>& C) const {
    out() = ComplexType{0};
    for (size_t i = 0; i < _n; ++i) {
      out += C(i) * hist[slot(i)];
    }
  }
};

} // namespace methods

#endif // COQUI_LR_DIIS_HPP
