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

#ifndef COQUI_SPMD_FOCK_SIGMA_DIIS_HPP
#define COQUI_SPMD_FOCK_SIGMA_DIIS_HPP

#include <algorithm>
#include <complex>
#include <iomanip>
#include <iostream>
#include <utility>
#include <vector>

#include "configuration.hpp"
#include "utilities/check.hpp"
#include "IO/app_loggers.h"

#include "nda/nda.hpp"
#include "nda/blas.hpp"

#include "numerics/iter_scf/diis/diis_coefs.hpp"
#include "numerics/iter_scf/diis/diis_timers.hpp"

namespace iter_scf {

/**
 * @brief SPMD (element-sliced) in-memory DIIS engine for the Dyson-SCF
 * FockSigma extrapolation, following the lr_diis.hpp pattern.
 *
 * Why not reuse VSpace<FockSigma>/diis_alg<FockSigma>? That stack operates on
 * whole FockSigma vectors held by a single rank (with optional HDF5 backing)
 * and its residual kernel/opt_state plumbing assumes full-array access; here
 * every rank owns only a contiguous element-slice of the flattened (F, Sigma)
 * pair, the subspace history is slice-local, and the residual arrives
 * pre-computed (the A10 distributed commutator) through a node-shared window.
 * The small dense B solve is shared with diis_alg via compute_diis_coefs_c1
 * (diis_coefs.hpp), so the coefficient algorithm exists in one place only.
 *
 * Layout / responsibilities:
 *  - Slices: the flat F (size nF) and Sigma (size nS) index ranges are split
 *    into contiguous chunks over node_comm (F and Sigma sliced independently).
 *    The node-shared source arrays are replicated per node, so each node's
 *    ranks collectively cover the full vector.
 *  - History: per-rank slice copies of every trial vector (_xF/_xS) and every
 *    residual (_resS; the Fock block of the commutator residual is identically
 *    zero, so it is not stored and contributes exactly 0 to every overlap,
 *    matching the serial B bit-for-bit in structure). Growth/purge bookkeeping
 *    mirrors diis_alg::next_step exactly.
 *  - B matrix: per-slice zdotc partials. Only the node that hosts the global
 *    root reduces its partials (its ranks cover the full vector exactly once,
 *    so the sum is complete and independent of other nodes' slice boundaries);
 *    the tiny row is then broadcast from the global root, so every rank on
 *    every node assembles the identical B and computes identical coefficients.
 *  - Extrapolation / damping: elementwise on slices, written in place into the
 *    node-shared F/Sigma windows. Given identical coefficients, each element
 *    depends only on same-index inputs, so the result is bitwise identical
 *    across nodes regardless of node sizes -- no internode broadcast needed.
 *  - Warmup damping: out = mixing*new + (1-mixing)*prev with prev = the
 *    previous accepted (post-mix) state, kept as an in-memory slice from the
 *    previous solve (byte-identical to the scf/iter{N-1} checkpoint the serial
 *    damp_t re-reads). On the first solve after a restart there is no
 *    in-memory prev; the caller stages it from the checkpoint (needs_prev_state).
 *
 * Convergence maxima are returned as per-rank partials; the caller reduces
 * them with a max (exactly order-independent).
 */
class spmd_fs_diis {
public:
  using Vec1D = nda::array<ComplexType, 1>;

  bool initialized() const { return _inited; }

  void configure(double mixing, size_t max_subsp_size, size_t warmup_iter) {
    _mixing = mixing;
    _max_subsp = max_subsp_size;
    _warmup = warmup_iter;
  }

  /**
   * @brief Capture this rank's slice of the starting state x0 (the SPMD
   * analogue of diis_init's x_start).
   *
   * @param capture_prev - at iteration 1 the current state IS the accepted
   *   iter-1 state (no mixing is applied at iteration 1), so it doubles as the
   *   previous accepted state for the first warmup damping. On a lazy
   *   (restart) init the current state is the new trial, NOT the accepted
   *   previous state -> pass false; the first damping then stages the
   *   checkpoint state via needs_prev_state().
   */
  template<typename NodeComm, nda::MemoryArrayOfRank<4> AF, nda::MemoryArrayOfRank<5> AS>
  void init_x0(NodeComm& node_comm, const AF& F, const AS& Sigma, bool capture_prev) {
    _nF = static_cast<long>(F.size());
    _nS = static_cast<long>(Sigma.size());
    std::tie(_f0, _f1) = rank_slice(_nF, node_comm.size(), node_comm.rank());
    std::tie(_s0, _s1) = rank_slice(_nS, node_comm.size(), node_comm.rank());
    push_x(F, Sigma);
    if (capture_prev) {
      _prevF = _xF.back();
      _prevS = _xS.back();
      _has_prev = true;
    }
    _warmup_count = 0;
    _inited = true;
    app_log(2, "DIIS: SPMD element-sliced subspace active ({} ranks/node, "
               "slice sizes F {} / Sigma {})",
            node_comm.size(), _f1 - _f0, _s1 - _s0);
  }

  /// Whether the next solve will consume the commutator residual C_t (i.e.
  /// grow the residual subspace). False on the grow-x-only first DIIS call,
  /// so the caller can skip computing a residual that is never consumed.
  bool needs_residual_next() const { return _inited && _xF.size() >= 2; }

  /// Whether the next solve will apply damping without an in-memory previous
  /// accepted state, so the caller must stage scf/iter{N-1} from the
  /// checkpoint (restart edge).
  bool needs_prev_state() const {
    if (_has_prev) return false;
    const size_t nx = _inited ? _xF.size() : 1; // lazy init pushes x0 first
    const bool warmup_next = (nx == 1) || (_warmup_count + 1 <= _warmup);
    return warmup_next || _resS.empty();
  }

  /**
   * @brief One SPMD DIIS step on (F, Sigma). Called on every rank of `comm`.
   *
   * Control flow mirrors diis_t::solve + diis_alg::next_step: warmup while
   * the trial space has 1 vector or warmup_count <= warmup_iter (damping),
   * then grow trial+residual subspaces (purging the oldest at max size) and
   * extrapolate when more than one residual is available.
   *
   * @param comm      - global communicator (B row broadcast, root logging)
   * @param node_comm - node communicator (B partial reduce on node 0)
   * @param on_node0  - true on the node hosting the global root
   * @param F, Sigma  - node-shared local() views; trial on entry, accepted
   *                    (extrapolated or damped) state on exit (each rank
   *                    writes its own slice; caller fences the windows)
   * @param C_t       - completed distributed commutator residual (node-shared
   *                    local() view), or nullptr when needs_residual_next()
   *                    was false
   * @param F_prev, S_prev - staged previous accepted state (restart edge;
   *                    nullptr unless needs_prev_state() was true)
   * @return per-rank partial {max|dF|, max|dSigma|}; caller max-reduces.
   */
  template<typename Comm, typename NodeComm,
           nda::MemoryArrayOfRank<4> AF, nda::MemoryArrayOfRank<5> AS,
           typename AC, typename AFP, typename ASP>
  std::array<double, 2> solve(Comm& comm, NodeComm& node_comm, bool on_node0,
                              AF&& F, AS&& Sigma, const AC* C_t,
                              const AFP* F_prev, const ASP* S_prev, long iter) {
    (void)iter;
    diis_timers::ScopeLog _d2log;
    diis_timers::spmd_total.start();
    utils::check(_inited, "spmd_fs_diis: init_x0 must be called before solve");
    utils::check(static_cast<long>(F.size()) == _nF and
                 static_cast<long>(Sigma.size()) == _nS,
                 "spmd_fs_diis: array sizes changed between init_x0 and solve");

    _warmup_count += 1;
    const size_t nx_pre = _xF.size();
    const bool warmup = (nx_pre == 1) || (_warmup_count <= _warmup);

    auto F_loc = slice_of(F, _nF, _f0, _f1);
    auto S_loc = slice_of(Sigma, _nS, _s0, _s1);

    std::array<double, 2> conv{0.0, 0.0};

    if (warmup) {
      app_log(2, "DIIS: Warmup iteration {}/{}. Simple damping will be executed instead.\n",
              _warmup_count, _warmup);
      if (nx_pre <= 1) {
        app_log(2, "DIIS: Growing vector subspace only. No extrapolation.\n");
        push_x(F, Sigma);
      } else {
        grow_with_residual(comm, node_comm, on_node0, F, Sigma, C_t);
      }
      app_log(4, "DIIS: DIIS vector space size: {}", _xF.size());
      app_log(4, "DIIS: DIIS residual space size: {}\n", _resS.size());
      conv = apply_damping(F_loc, S_loc, F_prev, S_prev);
    } else {
      grow_with_residual(comm, node_comm, on_node0, F, Sigma, C_t);
      if (_resS.size() > 1) {
        app_log(2, "DIIS: Performing the DIIS extrapolation. \n");
        diis_timers::spmd_coefs.start();
        auto C = compute_diis_coefs_c1(_B); // identical on every rank (B replicated)
        diis_timers::spmd_coefs.stop();
        if (comm.rank() == 0) {
          print_B();
          print_C(C);
        }
        log_predicted_error(C);

        // Extrapolate this rank's slices (the trial copy just pushed is
        // _x*.back(), so the windows can be overwritten afterwards).
        diis_timers::spmd_lincomb.start();
        Vec1D rF = lincomb(_xF, C);
        Vec1D rS = lincomb(_xS, C);
        diis_timers::spmd_lincomb.stop();

        diis_timers::spmd_conv.start();
        conv[0] = max_absdiff(_xF.back(), rF);
        conv[1] = max_absdiff(_xS.back(), rS);
        diis_timers::spmd_conv.stop();

        diis_timers::spmd_writeback.start();
        F_loc = rF;
        S_loc = rS;
        diis_timers::spmd_writeback.stop();

        diis_timers::spmd_prev_store.start();
        _prevF = std::move(rF);
        _prevS = std::move(rS);
        _has_prev = true;
        diis_timers::spmd_prev_store.stop();
      } else {
        app_log(2, "DIIS: DIIS extrapolation condition is not satisfied -> Skipping the extrapolation step\n");
        app_log(2, "DIIS: Performing simple damping instead.\n");
        conv = apply_damping(F_loc, S_loc, F_prev, S_prev);
      }
    }
    diis_timers::spmd_total.stop();
    return conv;
  }

private:
  double _mixing = 0.2;
  size_t _max_subsp = 5;
  size_t _warmup = 5;
  size_t _warmup_count = 0;
  bool _inited = false;
  bool _has_prev = false;
  long _nF = 0, _nS = 0;
  long _f0 = 0, _f1 = 0, _s0 = 0, _s1 = 0;
  // This rank's contiguous slices of every trial vector / residual. The trial
  // history mirrors x_vsp (it may exceed max_subsp by the initial grow-only
  // vectors, exactly as the serial x_vsp does); residuals mirror res_vsp.
  std::vector<Vec1D> _xF, _xS;
  std::vector<Vec1D> _resS;
  nda::matrix<ComplexType> _B{0, 0};
  Vec1D _prevF, _prevS; // previous accepted (post-mix) state slice

  /// Contiguous [i0, i1) slice of [0, n) owned by rank r of nr (as lr_diis).
  static std::pair<long, long> rank_slice(long n, long nr, long r) {
    const long chunk = (n + nr - 1) / nr;
    const long i0 = std::min(r * chunk, n);
    const long i1 = std::min(i0 + chunk, n);
    return {i0, i1};
  }

  template<typename A>
  static auto slice_of(A&& a, long n, long i0, long i1) {
    return nda::reshape(a, std::array<long, 1>{n})(nda::range(i0, i1));
  }

  template<nda::MemoryArrayOfRank<4> AF, nda::MemoryArrayOfRank<5> AS>
  void push_x(const AF& F, const AS& Sigma) {
    diis_timers::spmd_hist_store.start();
    _xF.emplace_back(slice_of(F, _nF, _f0, _f1));
    _xS.emplace_back(slice_of(Sigma, _nS, _s0, _s1));
    diis_timers::spmd_hist_store.stop();
  }

  /// Mirror of diis_alg::next_step's grow path: purge the oldest trial /
  /// residual / B row+col at max subspace size, add the new residual's B
  /// row (before storing it, as update_overlaps does), store residual+trial.
  template<typename Comm, typename NodeComm,
           nda::MemoryArrayOfRank<4> AF, nda::MemoryArrayOfRank<5> AS, typename AC>
  void grow_with_residual(Comm& comm, NodeComm& node_comm, bool on_node0,
                          const AF& F, const AS& Sigma, const AC* C_t) {
    utils::check(C_t != nullptr, "spmd_fs_diis: commutator residual required but not provided");
    if (_resS.size() >= _max_subsp) {
      app_log(2, "DIIS: Reached maximum subspace -> the first vector will be kicked out of the subspace.\n");
      purge_oldest();
    } else {
      app_log(2, "DIIS: Growing vector and residual subspaces for DIIS\n");
    }
    // Residual slice: only the Sigma block. The Fock block of the commutator
    // residual is identically zero (com_diis_residual pairs C_t with a zero
    // Fock block), so it contributes exactly 0 to every overlap -- identical
    // to the serial B where the <F_i|F_j> part vanishes exactly.
    diis_timers::spmd_hist_store.start();
    Vec1D u{slice_of(*C_t, _nS, _s0, _s1)};
    diis_timers::spmd_hist_store.stop();
    update_B(comm, node_comm, on_node0, u);
    _resS.push_back(std::move(u));
    push_x(F, Sigma);
  }

  /**
   * @brief Extend B by the new residual's row/column.
   *
   * Per-slice zdotc partials are summed over node 0's node_comm only (its
   * ranks cover the full vector exactly once, so the sum is complete and
   * unique regardless of other nodes' slice boundaries), then the tiny row is
   * broadcast from the global root: every rank ends with the identical B.
   */
  template<typename Comm, typename NodeComm>
  void update_B(Comm& comm, NodeComm& node_comm, bool on_node0, const Vec1D& u) {
    const size_t n = _resS.size() + 1;
    nda::array<ComplexType, 1> col(n);
    col() = 0;
    diis_timers::spmd_B_dots.start();
    if (on_node0 and u.size() > 0) {
      for (size_t i = 0; i + 1 < n; ++i) col(i) = nda::blas::dotc(_resS[i], u); // <res_i|u>
      col(n - 1) = nda::blas::dotc(u, u);
    }
    diis_timers::spmd_B_dots.stop();
    diis_timers::spmd_B_comm.start();
    if (on_node0)
      node_comm.all_reduce_in_place_n(col.data(), static_cast<long>(n), std::plus<>{});
    comm.broadcast_n(col.data(), static_cast<long>(n), 0);
    diis_timers::spmd_B_comm.stop();

    nda::matrix<ComplexType> Bnew(n, n);
    Bnew() = 0;
    for (size_t i = 0; i + 1 < n; ++i)
      for (size_t j = 0; j + 1 < n; ++j) Bnew(i, j) = _B(i, j);
    for (size_t i = 0; i + 1 < n; ++i) {
      Bnew(i, n - 1) = col(i);
      Bnew(n - 1, i) = std::conj(col(i));
    }
    Bnew(n - 1, n - 1) = col(n - 1);
    _B = Bnew;
  }

  void purge_oldest() {
    const size_t n = _B.shape()[0];
    nda::matrix<ComplexType> Bnew(n - 1, n - 1);
    for (size_t i = 0; i + 1 < n; ++i)
      for (size_t j = 0; j + 1 < n; ++j) Bnew(i, j) = _B(i + 1, j + 1);
    _B = Bnew;
    _resS.erase(_resS.begin());
    _xF.erase(_xF.begin());
    _xS.erase(_xS.begin());
  }

  /// Serial VSpace::make_linear_comb pairing and accumulation order: step i
  /// adds C(nC-1-i) * hist[n-1-i] (newest first). Elementwise, so the result
  /// is independent of the slice boundaries.
  static Vec1D lincomb(const std::vector<Vec1D>& hist, const nda::array<double, 1>& C) {
    const size_t n = hist.size();
    Vec1D r(hist.back().size());
    r() = 0;
    for (size_t i = 0; i < n and i < static_cast<size_t>(C.size()); ++i) {
      const ComplexType coeff = C(C.size() - 1 - i);
      r += coeff * hist[n - 1 - i];
    }
    return r;
  }

  static double max_absdiff(const Vec1D& a, const Vec1D& b) {
    double m = 0.0;
    for (long i = 0; i < a.size(); ++i) m = std::max(m, std::abs(a(i) - b(i)));
    return m;
  }

  /// Exact damp_t::solve arithmetic on this rank's slice:
  /// out = mixing*new + (1-mixing)*prev; partial conv = max|prev - out|.
  template<typename View>
  double damp_mix_slice(View&& H, const Vec1D& Hprev) {
    H = _mixing * H + (1.0 - _mixing) * Hprev;
    double m = 0.0;
    for (long i = 0; i < H.size(); ++i) m = std::max(m, std::abs(Hprev(i) - H(i)));
    return m;
  }

  template<typename FView, typename SView, typename AFP, typename ASP>
  std::array<double, 2> apply_damping(FView&& F_loc, SView&& S_loc,
                                      const AFP* F_prev, const ASP* S_prev) {
    // Previous accepted state: in-memory slice from the previous solve, or the
    // staged checkpoint state on the restart edge.
    Vec1D stageF, stageS;
    const Vec1D *pF = &_prevF, *pS = &_prevS;
    if (not _has_prev) {
      utils::check(F_prev != nullptr and S_prev != nullptr,
                   "spmd_fs_diis: no previous accepted state (in memory or staged)");
      diis_timers::spmd_damp_stage.start();
      stageF = Vec1D{slice_of(*F_prev, _nF, _f0, _f1)};
      stageS = Vec1D{slice_of(*S_prev, _nS, _s0, _s1)};
      diis_timers::spmd_damp_stage.stop();
      pF = &stageF;
      pS = &stageS;
    }
    std::array<double, 2> conv{0.0, 0.0};
    diis_timers::spmd_damp_mix.start();
    conv[0] = damp_mix_slice(F_loc, *pF);
    conv[1] = damp_mix_slice(S_loc, *pS);
    diis_timers::spmd_damp_mix.stop();
    diis_timers::spmd_prev_store.start();
    _prevF = Vec1D{F_loc};
    _prevS = Vec1D{S_loc};
    _has_prev = true;
    diis_timers::spmd_prev_store.stop();
    return conv;
  }

  // Log parity with diis_alg::next_step's print_B/print_C/predicted error.
  void print_B() const {
    std::cout << "DIIS: error overlaps B:" << std::endl;
    std::cout << std::setprecision(10);
    for (auto i : nda::range(0, _B.shape()[0])) {
      for (auto j : nda::range(0, _B.shape()[1])) std::cout << _B(i, j) << " ";
      std::cout << std::endl;
    }
  }

  void print_C(const nda::array<double, 1>& C) const {
    std::cout << "DIIS: Extrapolation coefficients:" << std::endl;
    std::cout << std::setprecision(10);
    for (auto i : nda::range(0, C.size())) std::cout << ComplexType{C(i)} << " ";
    std::cout << std::endl;
  }

  void log_predicted_error(const nda::array<double, 1>& C) const {
    if (_B.shape()[0] != _B.shape()[1] or _B.shape()[0] != C.size()) return;
    nda::array<ComplexType, 1> Cc(C.size());
    for (long i = 0; i < C.size(); ++i) Cc(i) = C(i);
    nda::array<ComplexType, 1> vec_error(_B.shape()[0]);
    nda::blas::gemv(_B, Cc, vec_error);
    const ComplexType exp_error = nda::sum(vec_error);
    app_log(2, "DIIS: Squared predicted error of extrapolated vector (e,e) = {}",
            std::real(exp_error));
  }
};

} // namespace iter_scf

#endif // COQUI_SPMD_FOCK_SIGMA_DIIS_HPP
