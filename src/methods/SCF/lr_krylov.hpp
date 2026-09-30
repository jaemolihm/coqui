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

#ifndef COQUI_LR_KRYLOV_HPP
#define COQUI_LR_KRYLOV_HPP

#include <cmath>
#include <vector>

#include "nda/nda.hpp"
#include "nda/blas.hpp"
#include "IO/app_loggers.h"
#include "utilities/element_partition.hpp"

namespace methods {

/**
 * @brief GCR solver for the affine LR fixed point x = g(x), in probe form, with a
 *        basis kept across right-hand sides.
 *
 * The LR SCF map at a fixed source is affine, g(x) = M x + b, so the fixed point
 * solves A x = b with A = I − M. Each call is handed one probe y and g(y), and
 * forms the residual r(y) = g(y) − y. With the iterate x and its residual r held
 * here, the probe direction z = y − x has
 *
 *   A z = r − r(y) = w,
 *
 * exactly, with no homogeneous kernel: every constant in g (the IBC/δV terms, a
 * frozen perturbative source) cancels in the difference. (z, w) is
 * orthonormalized against the stored pairs (CGS2, real inner product) and
 * stored, and x, r take the minimum-residual update along it. The next probe is
 * x + r (no preconditioner), or x itself once ‖r‖ ≤ tol.
 *
 * Only b changes between the stages of a split-kernel solve, so the stored
 * (z_i, w_i = A z_i) stay valid: reset_rhs() keeps them, and the first probe of
 * the new right-hand side is projected onto them (a Galerkin minimum-residual
 * start over the recycled space). clear() drops them, for a new perturbation.
 * At capacity the oldest pair is evicted.
 *
 * The inner product is Re Σ conj(a) b: with fix_density at q = Γ the map is
 * ℝ-linear only, and the real inner product is what keeps GCR exact there.
 *
 * SPMD: every rank of `comm` holds its `pmap` slice of every vector, and the
 * scalars are reduced over `comm`, one all_reduce per orthogonalization pass.
 * A breakdown (A z inside the span of the stored W) restarts with an empty basis.
 *
 * Named for the flexible variant of docs/plan_lr_krylov_solver.md (Phase 1): the
 * preconditioner hook z = P⁻¹ r is reserved for it; here z = r.
 *
 * Why not src/numerics/iter_scf/: its solvers keep their vectors in HDF5-backed
 * VSpace storage behind a file-I/O Vector interface, and have no notion of a
 * striped slice or of recycling a basis across right-hand sides. lr_diis
 * already stands apart from it for the same reasons.
 */
class lr_fgcr {
public:
  using Vec1D = nda::array<ComplexType, 1>;

  explicit lr_fgcr(size_t max_basis) : _cap(max_basis) {
    utils::check(_cap >= 1, "lr_fgcr: the basis capacity must be >= 1.");
  }

  /// Forget the basis (buffers kept) and the right-hand side: a new solve.
  void clear() {
    _n = 0;
    _head = 0;
    _have_rhs = false;
    _warned_full = false;
  }
  /// Keep the basis; the next step starts a new right-hand side.
  void reset_rhs() { _have_rhs = false; }

  size_t capacity() const { return _cap; }
  size_t basis_size() const { return _n; }
  /// ‖r(y₀)‖ and the residual left after projecting it onto the recycled
  /// basis, for the first probe of the current right-hand side.
  double rhs_residual_raw() const { return _r0_raw; }
  double rhs_residual_projected() const { return _r0_proj; }

  /**
   * @brief One GCR step.
   *
   * @param F_inout   - [IN/OUT] the full array (pass .local()); g(y) on entry, and
   *                            this rank's slice of the next probe (x + r), or of x
   *                            once converged, on exit
   * @param F_probe   - [INPUT]  this rank's slice of the probe y g was applied to
   * @param tol       - [INPUT]  convergence threshold on ‖r‖
   * @param converged - [OUTPUT] ‖r‖ ≤ tol
   * @return ‖r‖ after the step
   */
  template<typename Comm, typename FView, typename FProbe>
  double step(Comm& comm, utils::part_map const& pmap, FView F_inout,
              FProbe const& F_probe, double tol, bool& converged) {
    const long nF = F_inout.size();
    auto [f0, f1] = pmap.my_slice(nF);
    auto F_loc = nda::reshape(F_inout, std::array<long, 1>{nF})(nda::range(f0, f1));
    utils::check(static_cast<long>(F_probe.size()) == f1 - f0,
                 "lr_fgcr: probe slice size {} != partition slice size {}",
                 F_probe.size(), f1 - f0);

    Vec1D r_y = F_loc - F_probe;
    if (!_have_rhs) {
      _x = F_probe;
      _r = r_y;
      _r0_raw = norm(comm, _r);
      // Minimum-residual start over the recycled space: x += Z h, r −= W h with
      // h = Wᵀ r, twice for orthogonality.
      for (int pass = 0; pass < 2 && _n > 0; ++pass) {
        auto h = dots(comm, _r);
        for (size_t i = 0; i < _n; ++i) {
          _x += h[i] * _Z[slot(i)];
          _r -= h[i] * _W[slot(i)];
        }
      }
      _r0_proj = (_n > 0) ? norm(comm, _r) : _r0_raw;
      _have_rhs = true;
    } else {
      Vec1D z = F_probe - _x;
      Vec1D w = _r - r_y;                 // = A z
      // CGS2. The first pass also carries ‖w‖² before orthogonalization, the
      // scale the breakdown test below is relative to.
      double w0n = 0.0;
      for (int pass = 0; pass < 2; ++pass) {
        std::vector<double> h(_n + (pass == 0 ? 1 : 0), 0.0);
        if (w.size() > 0) {
          for (size_t i = 0; i < _n; ++i) h[i] = std::real(nda::blas::dotc(_W[slot(i)], w));
          if (pass == 0) h[_n] = std::real(nda::blas::dotc(w, w));
        }
        if (h.empty()) break;
        comm.all_reduce_in_place_n(h.data(), static_cast<long>(h.size()), std::plus<>{});
        if (pass == 0) w0n = std::sqrt(h[_n]);
        for (size_t i = 0; i < _n; ++i) {
          w -= h[i] * _W[slot(i)];
          z -= h[i] * _Z[slot(i)];
        }
        if (_n == 0) break;
      }
      // ‖w‖² and ⟨w, r⟩ in one reduction.
      double acc[2] = {0.0, 0.0};
      if (w.size() > 0) {
        acc[0] = std::real(nda::blas::dotc(w, w));
        acc[1] = std::real(nda::blas::dotc(w, _r));
      }
      comm.all_reduce_in_place_n(acc, 2, std::plus<>{});
      const double wn = std::sqrt(acc[0]);
      if (wn <= 1e-14 * w0n || wn == 0.0) {
        // A z lies in the span of the stored W to working precision: this probe
        // adds no direction, and repeating it would add none either. Restart
        // from the next probe x + r with an empty basis; its first step then
        // re-forms r from the true residual.
        app_log(1, "    [WARNING] lr_fgcr: breakdown (||w|| = {:.3e} after "
                   "orthogonalization, {:.3e} before); restarting with an empty "
                   "basis.", wn, w0n);
        _n = 0;
        _head = 0;
        _have_rhs = false;
      } else {
        w *= 1.0 / wn;
        z *= 1.0 / wn;
        const size_t s = push_slot();
        _Z[s] = z;
        _W[s] = w;
        const double alpha = acc[1] / wn;
        _x += alpha * z;
        _r -= alpha * w;
      }
    }

    const double rn = norm(comm, _r);
    converged = (rn <= tol);
    if (converged) F_loc = _x;
    else           F_loc = _x + _r;
    return rn;
  }

private:
  size_t _cap;
  size_t _n = 0;       // live basis pairs
  size_t _head = 0;    // ring origin
  bool _have_rhs = false;
  bool _warned_full = false;
  double _r0_raw = 0.0, _r0_proj = 0.0;
  std::vector<Vec1D> _Z, _W;   // this rank's slices of z_i and w_i = A z_i
  Vec1D _x, _r;

  size_t slot(size_t i) const { return (_head + i) % _Z.size(); }

  size_t push_slot() {
    if (_n == _cap) {
      if (!_warned_full) {
        app_log(1, "    [WARNING] lr_fgcr: the basis reached its capacity ({}); "
                   "evicting the oldest pair from now on.", _cap);
        _warned_full = true;
      }
      _head = (_head + 1) % _Z.size();
      --_n;
    } else if (_n == _Z.size()) {
      // Growth happens only before the first eviction, so appending keeps order.
      _Z.emplace_back();
      _W.emplace_back();
    }
    size_t s = slot(_n);
    ++_n;
    return s;
  }

  template<typename Comm>
  std::vector<double> dots(Comm& comm, Vec1D const& v) const {
    std::vector<Vec1D const*> W(_n);
    for (size_t i = 0; i < _n; ++i) W[i] = &_W[slot(i)];
    return utils::striped_dotc_batch(comm, W, v);
  }

  template<typename Comm>
  static double norm(Comm& comm, Vec1D const& v) {
    double n2 = (v.size() > 0) ? std::real(nda::blas::dotc(v, v)) : 0.0;
    comm.all_reduce_in_place_n(&n2, 1, std::plus<>{});
    return std::sqrt(n2);
  }
};

} // namespace methods

#endif // COQUI_LR_KRYLOV_HPP
