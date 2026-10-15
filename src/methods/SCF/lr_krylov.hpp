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
#include <functional>
#include <tuple>
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
 * exactly, with no homogeneous kernel: every constant in g (the IBC/δV terms)
 * cancels in the difference. (z, w) is orthonormalized against the stored pairs
 * (CGS2, real inner product) and stored, and x, r take the minimum-residual
 * update along it. The next probe is x + M⁻¹r with an optional preconditioner
 * M⁻¹ (x + r without one), or x itself once ‖r‖ ≤ tol. Since w is formed from
 * the probe actually taken, M⁻¹ may change from step to step (an inexact inner
 * solve): this is flexible GCR, and M⁻¹ affects the rate only.
 *
 * Stagnation. With z = r the step reduces ‖r‖ by the factor √(1 − η²), with
 * η = |⟨w⊥, r⟩| / (‖w⊥‖ ‖r‖) and w⊥ the orthogonalized image. Since r ⟂ W,
 * ⟨w⊥, r⟩ = ⟨A r, r⟩, so where A is not positive-real (⟨A r, r⟩ ≈ 0) truncated
 * GCR stalls: each step stores a pair that reduces nothing and evicts a useful
 * one. When η < eta_kick the pair is stored as usual, but the next probe is
 * x + ‖r‖ w_new along the newest image instead of x + r: its image A w_new
 * extends the Krylov space (as GMRES does) where A r would not. The kick costs no
 * extra evaluation and leaves every step with η ≥ eta_kick unchanged.
 * An image nearly inside span(W) (‖w⊥‖ ≤ dep_tol ‖w‖) is not stored and takes no
 * step, since normalizing it would mostly amplify round-off; the next probe is
 * then a kick, and if that probe was itself a kick the basis restarts, as at a
 * breakdown. No progress for stag_steps steps (‖r‖ above stag_ratio of its
 * last marked value) raises stagnated() for one step, for the caller to warn.
 *
 * For a sequence of right-hand sides of one operator A, the stored
 * (z_i, w_i = A z_i) stay valid: reset_rhs() keeps them, and the first probe of
 * the new right-hand side is projected onto them (a Galerkin minimum-residual
 * start over the recycled space). reset() drops them, for a new operator or
 * perturbation. At capacity the oldest pair is evicted.
 *
 * The iterate is (ΔF, ΔΣ(τ)), ΔΣ empty for a Σ-free kernel; every vector here
 * is this rank's slice of the concatenation [ΔF; ΔΣ]. The inner product is
 * Re Σ conj(a) b over it, unweighted, the same combined product lr_diis's
 * B-matrix uses: with fix_density at q = Γ the map is
 * ℝ-linear only, and the real inner product is what keeps GCR exact there.
 * The map must be affine for w = A z and the recycled pairs to hold, which rules
 * out the qpGW statification (its Padé continuation is not additive in ΔΣ).
 *
 * Memory: 2·capacity stored pairs plus x, r and two scratch vectors, each a
 * (ΔF + ΔΣ)-sized slice. With a Σ in the kernel one pair costs a ΔΣ, as one
 * DIIS depth unit does; the pairs are allocated as the basis grows.
 *
 * SPMD: every rank of `comm` holds its `pmap` slice of every vector, and the
 * scalars are reduced over `comm`, one all_reduce per orthogonalization pass.
 * A breakdown (A z inside the span of the stored W) restarts with an empty basis.
 *
 * Why not src/numerics/iter_scf/: its solvers keep their vectors in HDF5-backed
 * VSpace storage behind a file-I/O Vector interface, and have no notion of a
 * striped slice or of recycling a basis across right-hand sides. lr_diis
 * already stands apart from it for the same reasons.
 */
class lr_gcr {
public:
  using Vec1D = nda::array<ComplexType, 1>;

  explicit lr_gcr(size_t max_basis) : _cap(max_basis) {
    utils::check(_cap >= 1, "lr_gcr: the basis capacity must be >= 1.");
  }

  /// A step with η below this redirects the next probe along the newest image.
  static constexpr double eta_kick = 0.01;
  /// ‖w⊥‖ / ‖w‖ at or below this marks the image as dependent on the stored W.
  static constexpr double dep_tol = 1e-8;
  /// stagnated() is raised after this many steps without ‖r‖ dropping below
  /// stag_ratio times its last marked value.
  static constexpr int stag_steps = 20;
  static constexpr double stag_ratio = 0.99;

  /// Forget the basis (buffers kept) and the right-hand side: a new solve.
  void reset() {
    _n = 0;
    _head = 0;
    _have_rhs = false;
    _warned_full = false;
    _kick = false;
    _kicks = 0;
  }
  /// Keep the basis; the next step starts a new right-hand side.
  void reset_rhs() { _have_rhs = false; }

  size_t capacity() const { return _cap; }
  size_t basis_size() const { return _n; }
  /// ‖r(y₀)‖ and the residual left after projecting it onto the recycled
  /// basis, for the first probe of the current right-hand side.
  double rhs_residual_raw() const { return _r0_raw; }
  double rhs_residual_projected() const { return _r0_proj; }
  /// The probe handed out by the last step is a kick along the newest image.
  bool kicked() const { return _kick; }
  /// Kicks since the last reset().
  size_t kicks() const { return _kicks; }
  /// The last step completed stag_steps steps without progress.
  bool stagnated() const { return _stagnated; }

  /**
   * @brief One GCR step on the iterate (ΔF, ΔΣ).
   *
   * The same argument layout as lr_diis::next_step_combined: two slots, each a
   * full array plus this rank's slice of the probe. The second slot may be empty
   * (a ΔF-only iterate). Internally the two slices are concatenated into one
   * vector [F; S], and every inner product is Re⟨F, F'⟩ + Re⟨S, S'⟩, unweighted.
   *
   * @param F_inout   - [IN/OUT] the full ΔF (pass .local()); g(y) on entry, and
   *                            this rank's slice of the next probe (x + r,
   *                            x + M⁻¹r, or a kick), or of x once converged, on exit
   * @param F_probe   - [INPUT]  this rank's slice of the ΔF probe y g was applied to
   * @param S_inout   - [IN/OUT] the full ΔΣ, as F_inout; empty for a ΔF-only iterate
   * @param S_probe   - [INPUT]  this rank's slice of the ΔΣ probe
   * @param tol       - [INPUT]  convergence threshold on ‖r‖ of the concatenation
   * @param converged - [OUTPUT] ‖r‖ ≤ tol
   * @param precond   - [INPUT]  optional z = M⁻¹ r on this rank's slices of the
   *                            concatenation; collective. Null: z = r
   * @return ‖r‖ after the step
   */
  template<typename Comm, typename FView, typename FProbe, typename SView, typename SProbe>
  double step(Comm& comm, utils::part_map const& pmap,
              FView F_inout, FProbe const& F_probe,
              SView S_inout, SProbe const& S_probe,
              double tol, bool& converged,
              std::function<void(Vec1D const&, Vec1D&)> const* precond = nullptr) {
    const long nF = F_inout.size();
    auto [f0, f1] = pmap.my_slice(nF);
    auto F_loc = nda::reshape(F_inout, std::array<long, 1>{nF})(nda::range(f0, f1));
    utils::check(static_cast<long>(F_probe.size()) == f1 - f0,
                 "lr_gcr: ΔF probe slice size {} != partition slice size {}",
                 F_probe.size(), f1 - f0);
    const long nS = S_inout.size();
    long s0 = 0, s1 = 0;
    if (nS > 0) {
      std::tie(s0, s1) = pmap.my_slice(nS);
      utils::check(static_cast<long>(S_probe.size()) == s1 - s0,
                   "lr_gcr: ΔΣ probe slice size {} != partition slice size {}",
                   S_probe.size(), s1 - s0);
    }
    const long mF = f1 - f0, n = mF + (s1 - s0);
    auto rF = nda::range(0, mF);
    auto rS = nda::range(mF, n);

    // _g <- g(y), _y <- y, both as [F; S]. Sized once per (layout of) solve.
    if (_g.size() != n) { _g = Vec1D(n); _y = Vec1D(n); }
    _g(rF) = F_loc;
    _y(rF) = F_probe;
    if (nS > 0) {
      auto S_loc = nda::reshape(S_inout, std::array<long, 1>{nS})(nda::range(s0, s1));
      _g(rS) = S_loc;
      _y(rS) = S_probe;
    }

    _g -= _y;                             // r(y) = g(y) − y
    const bool probe_was_kick = _kick;
    bool kick = false, extended = false;
    _stagnated = false;
    if (!_have_rhs) {
      // First probe of a right-hand side
      if (_x.size() != n) { _x = Vec1D(n); _r = Vec1D(n); }
      _x() = _y;
      _r() = _g;
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
      _r_mark = _r0_proj;
      _nflat = 0;
    } else {
      // Later probe: extend the basis
      extended = true;
      auto& z = _y;
      auto& w = _g;
      z -= _x;                            // z = y − x
      w() = _r - w;                       // w = r − r(y) = A z
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
      const bool dependent = (wn <= dep_tol * w0n);
      if (wn <= 1e-14 * w0n || wn == 0.0 || (dependent && probe_was_kick)) {
        // A z lies in the span of the stored W to working precision, or nearly
        // so after a kick: this probe adds no direction, and repeating it would
        // add none either. Restart from the next probe x + r with an empty
        // basis; its first step then re-forms r from the true residual.
        app_log(1, "    [WARNING] lr_gcr: breakdown (||w|| = {:.3e} after "
                   "orthogonalization, {:.3e} before); restarting with an empty "
                   "basis.", wn, w0n);
        _n = 0;
        _head = 0;
        _have_rhs = false;
      } else if (dependent) {
        // A z nearly inside span(W): not stored, no step; kick next
        kick = true;
      } else {
        w *= 1.0 / wn;
        z *= 1.0 / wn;
        const size_t s = push_slot();
        _Z[s] = z;
        _W[s] = w;
        const double alpha = acc[1] / wn;
        _x += alpha * z;
        _r -= alpha * w;
        // η of this step; ‖r‖ before it is the previous step's return value.
        kick = (std::abs(alpha) < eta_kick * _rn);
      }
    }

    const double rn = norm(comm, _r);
    converged = (rn <= tol);
    if (extended && _have_rhs) {
      // Progress check
      if (rn < stag_ratio * _r_mark) {
        _r_mark = rn;
        _nflat = 0;
      } else if (++_nflat == stag_steps) {
        _stagnated = !converged;
      }
    }
    _kick = kick && !converged;
    _rn = rn;
    if (converged) {
      // Converged: hand back x
      _y() = _x;
    } else if (_kick) {
      // Next probe x + ‖r‖ w_new, along the newest stored image
      ++_kicks;
      _y() = _x + rn * _W[slot(_n - 1)];
    } else if (precond) {
      // Next probe x + M⁻¹ r
      if (_z.size() != n) _z = Vec1D(n);
      (*precond)(_r, _z);
      _y() = _x + _z;
    } else {
      // Next probe x + r
      _y() = _x + _r;
    }
    F_loc = _y(rF);
    if (nS > 0) {
      auto S_loc = nda::reshape(S_inout, std::array<long, 1>{nS})(nda::range(s0, s1));
      S_loc = _y(rS);
    }
    return rn;
  }

private:
  size_t _cap;
  size_t _n = 0;       // live basis pairs
  size_t _head = 0;    // ring origin
  bool _have_rhs = false;
  bool _warned_full = false;
  bool _kick = false;         // the last probe handed out is a kick
  bool _stagnated = false;
  size_t _kicks = 0;
  int _nflat = 0;             // steps since ‖r‖ last dropped below stag_ratio·_r_mark
  double _rn = 0.0, _r_mark = 0.0;  // ‖r‖ after the last step; last marked ‖r‖
  double _r0_raw = 0.0, _r0_proj = 0.0;
  std::vector<Vec1D> _Z, _W;   // this rank's slices of z_i and w_i = A z_i
  Vec1D _x, _r;
  // Per-step scratch, kept across calls: g(y), then r(y), then w; y, then z,
  // then the next probe; the preconditioned residual M⁻¹ r.
  Vec1D _g, _y, _z;

  size_t slot(size_t i) const { return (_head + i) % _Z.size(); }

  size_t push_slot() {
    if (_n == _cap) {
      if (!_warned_full) {
        app_log(1, "    [WARNING] lr_gcr: the basis reached its capacity ({}); "
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

  // Re⟨W_i, v⟩ for every stored pair, reduced over comm in one all_reduce: each
  // rank holds its pmap slice of every vector, so the summed partial dots are
  // the full ones.
  template<typename Comm>
  std::vector<double> dots(Comm& comm, Vec1D const& v) const {
    std::vector<double> h(_n, 0.0);
    if (v.size() > 0)
      for (size_t i = 0; i < _n; ++i) h[i] = std::real(nda::blas::dotc(_W[slot(i)], v));
    if (!h.empty()) comm.all_reduce_in_place_n(h.data(), static_cast<long>(h.size()), std::plus<>{});
    return h;
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
