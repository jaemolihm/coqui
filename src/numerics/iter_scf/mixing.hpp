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

#ifndef COQUI_MIXING_HPP
#define COQUI_MIXING_HPP

#include <algorithm>
#include <array>
#include <cmath>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "configuration.hpp"
#include "utilities/check.hpp"
#include "utilities/element_partition.hpp"
#include "IO/app_loggers.h"

#include "nda/nda.hpp"
#include "nda/blas.hpp"
#include "numerics/nda_functions.hpp"  // nda::blas 3-argument gemm/gemv

#include "numerics/iter_scf/diis/diis_coefs.hpp"
#include "numerics/iter_scf/diis/diis_timers.hpp"

/**
 * Element-sliced mixing (damping or DIIS) of the two node-shared arrays (F, Sigma) that
 * an SCF iteration updates together, for the ground-state Dyson SCF (F, Sigma(tau)), the qp
 * SCF (the effective Hamiltonian, no Sigma), dmft_embed, and the linear response (dF,
 * dSigma(tau) or dV_QPGW; dSigma may be absent).
 *
 * Every rank owns one contiguous slice of the flattened F and of the flattened
 * Sigma (utils::part_map, node-major parts over the global comm), keeps only its
 * slices of the mixing history, and writes the mixed result in place into its slices
 * of the node-shared windows. The windows are completed afterwards
 * (utils::complete_shared_slices). The whole job then stores each history vector once.
 *
 * Every rank sees only its slices, so the residual of each trial arrives precomputed
 * from the caller. The small dense solve for the DIIS coefficients is shared
 * (compute_diis_coefs_c1).
 *
 * The LR has a third accelerator, GCR (methods::lr_gcr, methods/SCF/lr_krylov.hpp). It needs
 * an affine fixed-point map, so only the LR offers it; it takes the same hist_vec_layout.
 *
 * DIIS with the frequency-dependent commutator residual of the Dyson SCF:
 * P. Pokhilko, C.-N. Yeh, D. Zgid, J. Chem. Phys. 156, 094101 (2022),
 * https://doi.org/10.1063/5.0082586
 */
namespace iter_scf {

/// One flattened vector of mixing data: this rank's slices of the flattened F and Sigma
/// (S empty if absent). Holds the previous accepted state, a DIIS trial or residual, or a
/// GCR probe. Which elements the slices are is given by a hist_vec_layout, shared by all
/// the vectors of one run.
struct hist_vec {
  nda::array<ComplexType, 1> F, S;
};

/// Which elements of the flattened F (F_size_total elements) and Sigma (S_size_total; 0: F
/// only) this rank owns: one contiguous range of each. One layout per run, shared by every
/// hist_vec of that run. F_view_slice / S_view_slice view this rank's slice of a full array;
/// store_slice copies the slices into a hist_vec.
class hist_vec_layout {
public:
  hist_vec_layout() = default;

  /// Not collective.
  hist_vec_layout(utils::part_map pmap, long F_size_total, long S_size_total)
      : _pmap(std::move(pmap)), _F_size_total(F_size_total), _S_size_total(S_size_total) {
    std::tie(_F_start, _F_end) = _pmap.my_slice(_F_size_total);
    std::tie(_S_start, _S_end) = _pmap.my_slice(_S_size_total);
  }

  /// The partition of the global comm into parts (one per rank, node-major) and this rank's
  /// part; utils::complete_shared_slices reads it to fill the node-shared windows.
  utils::part_map const& pmap() const { return _pmap; }
  /// Number of elements of the whole flattened F and Sigma.
  long F_size_total() const { return _F_size_total; }
  long S_size_total() const { return _S_size_total; }
  /// [start, end) of this rank's slice in the flattened F and Sigma.
  std::pair<long, long> F_startend() const { return {_F_start, _F_end}; }
  std::pair<long, long> S_startend() const { return {_S_start, _S_end}; }
  /// Number of elements of this rank's slices.
  long F_len() const { return _F_end - _F_start; }
  long S_len() const { return _S_end - _S_start; }
  /// Ranges of the F and Sigma parts in the concatenation [F slice; Sigma slice] of length
  /// F_len() + S_len() (the GCR vectors of the LR).
  nda::range F_range() const { return nda::range(0, F_len()); }
  nda::range S_range() const { return nda::range(F_len(), F_len() + S_len()); }

  /// 1D view of this rank's slice of the full F_skij (or the full Sigma_tskij).
  template<typename A> auto F_view_slice(A&& F_skij) const {
    return nda::reshape(F_skij, std::array<long, 1>{_F_size_total})(nda::range(_F_start, _F_end));
  }
  template<typename A> auto S_view_slice(A&& S_tskij) const {
    return nda::reshape(S_tskij, std::array<long, 1>{_S_size_total})(nda::range(_S_start, _S_end));
  }

  /// Copy this rank's slices of the full arrays into v_vec.
  template<typename AF, typename AS>
  void store_slice(hist_vec& v_vec, AF const& F_skij, AS const& S_tskij) const {
    v_vec.F = F_view_slice(F_skij);
    v_vec.S = S_view_slice(S_tskij);
  }

private:
  utils::part_map _pmap;
  long _F_size_total = 0, _S_size_total = 0;
  long _F_start = 0, _F_end = 0, _S_start = 0, _S_end = 0;
};

/// Update in-place F and S by mixing them with a previous iterate, prev_vec.
/// (F, S) = mixing * (F, S) + (1 - mixing) * prev_vec.
/// Each rank only updates the range it stores in layout.
/// Do nothing for mixing >= 1.
template<typename AF, typename AS>
void damp_slices(hist_vec_layout const& layout, double mixing, AF&& F_skij, AS&& S_tskij,
                 hist_vec const& prev_vec) {
  if (mixing >= 1.0) return;
  auto F_slice = layout.F_view_slice(F_skij);
  F_slice = mixing * F_slice + (1.0 - mixing) * prev_vec.F;
  if (layout.S_len() > 0) {
    auto S_slice = layout.S_view_slice(S_tskij);
    S_slice = mixing * S_slice + (1.0 - mixing) * prev_vec.S;
  }
}

/**
 * @brief DIIS over sliced vectors: the stored (trial, residual) entries, at most
 * max_subsp of them (the oldest is dropped first), and the residual overlap matrix B.
 *
 * The caller supplies the residual of each trial (the Dyson-SCF commutator, or the
 * difference of successive iterates in the LR). B_ij = <r_i|r_j> summed over the two
 * blocks, each rank forming its slice of the dot products and one all_reduce
 * completing them, so that every rank holds the same B and the same coefficients.
 * Extrapolation needs more than `warmup` steps and at least `min_subsp` stored entries;
 * otherwise, or when the coefficients do not exist, the step damps.
 */
class diis_mixer {
public:
  struct config {
    double mixing;       ///< damping factor of the warmup and fallback steps
    size_t max_subsp;    ///< stored (trial, residual) entries
    size_t warmup;       ///< steps that only damp
    size_t min_subsp;    ///< entries needed to extrapolate
    int log_level;
  };

  /// A max_subsp below min_subsp can never extrapolate: the engine then only damps.
  explicit diis_mixer(config c) : _cfg(c), _B(0, 0) {
    if (not active())
      app_log(1, "[WARNING] DIIS: a subspace of depth {} can never reach the {} entries needed "
                 "to extrapolate; the iteration will only damp (mixing = {}). Raise the "
                 "subspace depth to use DIIS.", _cfg.max_subsp, _cfg.min_subsp, _cfg.mixing);
  }

  config const& cfg() const { return _cfg; }
  /// Whether the subspace can ever reach min_subsp.
  bool active() const { return _cfg.max_subsp >= _cfg.min_subsp; }
  /// mix() calls since reset().
  size_t steps() const { return _steps; }
  /// Stored (trial, residual) entries.
  size_t size() const { return _n; }

  /// Forget the subspace, keeping every history buffer.
  void reset() {
    _n = 0;
    _head = 0;
    _steps = 0;
    _B = nda::matrix<ComplexType>(0, 0);
  }

  hist_vec const& newest_trial() const { return _entries[entry_index(_n - 1)].x_vec; }

  /**
   * @brief Store the trial (F, S) with its residual res_vec as the newest entry,
   * evicting the oldest entry first when max_subsp are stored, and extend B. Collective on comm.
   *
   * @param F_skij, S_tskij - full arrays of the trial (this rank's slices are stored)
   * @param res_vec         - this rank's slices of the residual; res_vec.F may be empty
   */
  template<typename Comm, typename AF, typename AS>
  void push(Comm& comm, hist_vec_layout const& layout, AF const& F_skij, AS const& S_tskij,
            hist_vec const& res_vec) {
    if (not active()) return;
    diis_timers::mix_hist_store.start();
    if (_n == _cfg.max_subsp) evict_oldest();
    if (_n == _entries.size()) {
      // Growth happens only before the first eviction, so appending keeps the order.
      utils::check(_head == 0, "diis_mixer: growth of the history with head = {} != 0", _head);
      _entries.emplace_back();
    }
    auto& s = _entries[entry_index(_n)];
    layout.store_slice(s.x_vec, F_skij, S_tskij);
    s.res_vec = res_vec;
    ++_n;
    diis_timers::mix_hist_store.stop();
    update_B(comm);
  }

  /**
   * @brief Mix this rank's slices of (F, S) in place: the DIIS extrapolation of the
   * stored trials, or damping against `prev_vec` when there is none. Returns true if it
   * extrapolated. Every rank takes the same branch.
   */
  template<typename AF, typename AS>
  bool mix(hist_vec_layout const& layout, AF&& F_skij, AS&& S_tskij, hist_vec const& prev_vec) {
    ++_steps;
    const int L = _cfg.log_level;
    auto damp = [&]() {
      diis_timers::mix_damp.start();
      damp_slices(layout, _cfg.mixing, F_skij, S_tskij, prev_vec);
      diis_timers::mix_damp.stop();
    };
    if (_steps <= _cfg.warmup) {
      app_log(L, "DIIS: warmup iteration {}/{}: damping (mixing = {}, subspace size {})",
              _steps, _cfg.warmup, _cfg.mixing, _n);
      damp();
      return false;
    }
    if (_n < _cfg.min_subsp) {
      app_log(L, "DIIS: subspace size {} < {}: damping (mixing = {})",
              _n, _cfg.min_subsp, _cfg.mixing);
      damp();
      return false;
    }

    diis_timers::mix_coefs.start();
    auto C = compute_diis_coefs_c1(_B, L); // identical on every rank (B replicated)
    diis_timers::mix_coefs.stop();
    if (not C) {
      app_log(1, "[WARNING] DIIS: the residual-overlap matrix is singular (the residuals vanish "
                 "or are linearly dependent to machine precision); skipping the extrapolation "
                 "and damping instead (mixing = {}).", _cfg.mixing);
      damp();
      return false;
    }
    log_extrapolation(*C);

    diis_timers::mix_lincomb.start();
    auto F_slice = layout.F_view_slice(F_skij);
    auto S_slice = layout.S_view_slice(S_tskij);
    F_slice() = ComplexType{0};
    S_slice() = ComplexType{0};
    // Newest entry first.
    for (size_t k = _n; k-- > 0;) {
      const ComplexType c = (*C)(k);
      auto const& x_vec = _entries[entry_index(k)].x_vec;
      F_slice += c * x_vec.F;
      S_slice += c * x_vec.S;
    }
    diis_timers::mix_lincomb.stop();
    return true;
  }

private:
  struct entry_t {
    hist_vec x_vec, res_vec;  ///< trial and its residual
  };

  config _cfg;
  // Circular buffer of the stored entries: _n of them are live, and the i-th oldest is
  // _entries[(_head + i) % _entries.size()]. Evicting the oldest only advances _head, so no
  // vector is destroyed or reallocated once max_subsp entries exist.
  std::vector<entry_t> _entries;
  size_t _n = 0;
  size_t _head = 0;
  size_t _steps = 0;
  nda::matrix<ComplexType> _B;

  /// Position in _entries of the i-th oldest live entry.
  size_t entry_index(size_t i) const { return (_head + i) % _entries.size(); }

  /// Forget the oldest entry and the first row and column of B.
  void evict_oldest() {
    const size_t n = _B.shape()[0];
    nda::matrix<ComplexType> Bnew(n - 1, n - 1);
    for (size_t i = 0; i + 1 < n; ++i)
      for (size_t j = 0; j + 1 < n; ++j) Bnew(i, j) = _B(i + 1, j + 1);
    _B = Bnew;
    _head = (_head + 1) % _entries.size();
    --_n;
  }

  /// Extend B by the row and column of the newest residual: this rank's dot products over
  /// its slices of both blocks (an empty block adds nothing), one all_reduce.
  template<typename Comm>
  void update_B(Comm& comm) {
    const size_t n = _n;
    auto const& rn_vec = _entries[entry_index(n - 1)].res_vec;
    nda::array<ComplexType, 1> col(n);
    col() = 0;
    diis_timers::mix_B_dots.start();
    for (size_t i = 0; i < n; ++i) {
      auto const& ri_vec = _entries[entry_index(i)].res_vec;
      ComplexType acc = 0.0;
      if (ri_vec.F.size() > 0) acc += nda::blas::dotc(ri_vec.F, rn_vec.F);
      if (ri_vec.S.size() > 0) acc += nda::blas::dotc(ri_vec.S, rn_vec.S);
      col(i) = acc;
    }
    diis_timers::mix_B_dots.stop();
    diis_timers::mix_B_comm.start();
    comm.all_reduce_in_place_n(col.data(), static_cast<long>(n), std::plus<>{});
    diis_timers::mix_B_comm.stop();

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

  void log_extrapolation(nda::array<double, 1> const& C) const {
    const int L = _cfg.log_level;
    double c_norm = 0.0;
    for (long i = 0; i < C.size(); ++i) c_norm += std::abs(C(i));
    app_log(L, "DIIS: extrapolation with subspace size {}, sum|c_i| = {:.3e}", _n, c_norm);
    app_log(L, "DIIS: error overlaps B:");
    for (long i = 0; i < _B.shape()[0]; ++i) {
      std::string row;
      for (long j = 0; j < _B.shape()[1]; ++j) row += fmt::format("({:.10e},{:.10e}) ", _B(i, j).real(), _B(i, j).imag());
      app_log(L, "  {}", row);
    }
    std::string crow;
    for (long i = 0; i < C.size(); ++i) crow += fmt::format("{:.10e} ", C(i));
    app_log(L, "DIIS: extrapolation coefficients: {}", crow);
    nda::array<ComplexType, 1> Cc(C.size()), vec_error(_B.shape()[0]);
    for (long i = 0; i < C.size(); ++i) Cc(i) = C(i);
    nda::blas::gemv(_B, Cc, vec_error);
    app_log(L, "DIIS: squared predicted error of the extrapolated vector (e,e) = {}",
            std::real(nda::sum(vec_error)));
  }
};

/**
 * Everything one SCF loop keeps from one mixing step to the next, for the Dyson SCF, the qp
 * SCF and dmft_embed: the slicing (layout), this rank's slices of the previous accepted state
 * (prev_vec), and, for DIIS, the history (diis). The SCF arrays are overwritten in place by
 * each Dyson pass and each mixing, so prev_vec is the only in-memory copy of the previous
 * state. iter_scf_t owns one (iter_scf_t::state()), so it lives as long as the solver, across
 * the iterations of scf_loop / qp_scf_loop; dmft_embed resets it at every call. The LR does not
 * use it: lr_driver keeps its own layout and previous iterate and calls diis_mixer directly.
 */
struct mixing_state {
  hist_vec_layout layout;
  /// This rank's slices of the previous accepted (post-mix) state: the damping target of the
  /// damping, DIIS warmup and DIIS fallback steps, and the reference of their convergence
  /// measure.
  hist_vec prev_vec;
  bool has_prev = false;  ///< whether prev_vec is filled (false until the first step or a stage from the checkpoint)
  double mixing = 1.0;       ///< damping factor of the damping algorithm
  std::optional<diis_mixer> diis;  ///< DIIS only

  bool initialized() const { return layout.F_size_total() > 0; }

  /// Cut the slices of F (F_size_total elements) and Sigma (S_size_total; 0: F only);
  /// `diis_cfg` selects DIIS over damping. Collective on mpi.comm.
  template<typename MPI_Context_t>
  void init(MPI_Context_t& mpi, long F_size_total, long S_size_total, double mixing_,
            std::optional<diis_mixer::config> diis_cfg) {
    layout = hist_vec_layout(utils::make_part_map(mpi), F_size_total, S_size_total);
    mixing = mixing_;
    has_prev = false;
    diis.reset();
    if (diis_cfg) diis.emplace(*diis_cfg);
  }

  /// Whether the next step consumes a residual.
  bool needs_residual() const { return diis and diis->active() and diis->steps() > 0; }

  /**
   * @brief One mixing step on this rank's slices of the node-shared F and Sigma
   * (trial on entry, accepted state on exit), then prev_vec <- the accepted state.
   *
   * @param residual_vec - this rank's slices of the residual of the trial (F or Sigma part
   *                       may be empty), or nullptr when needs_residual() was false
   * @return per-rank {max|ref - out|_F, max|ref - out|_Sigma}, ref the previous accepted
   *         state (damped) or the newest trial (extrapolated); the caller max-reduces.
   */
  template<typename Comm, typename AF, typename AS>
  std::array<double, 2> step(Comm& comm, AF&& F_skij, AS&& S_tskij, hist_vec const* residual_vec) {
    diis_timers::mix_total.start();
    // In-place update of this rank's slices of F_skij and S_tskij by mixing (DIIS, or damping
    // against prev_vec)
    bool extrapolated = false;
    if (diis) {
      // Store the trial (F_skij, S_tskij) and its residual as the newest DIIS entry (none at the first step)
      if (residual_vec) diis->push(comm, layout, F_skij, S_tskij, *residual_vec);
      // Extrapolate from the stored trials, or damp against prev_vec (warmup, too few entries,
      // singular B). The entries are pre-mix trials, so the previous accepted state is only in prev_vec.
      extrapolated = diis->mix(layout, F_skij, S_tskij, prev_vec);
    } else {
      diis_timers::mix_damp.start();
      damp_slices(layout, mixing, F_skij, S_tskij, prev_vec);
      diis_timers::mix_damp.stop();
    }

    // Check convergence
    diis_timers::mix_conv.start();
    hist_vec const& ref_vec = extrapolated ? diis->newest_trial() : prev_vec;
    auto max_absdiff = [](auto const& a, auto const& b) {
      double m = 0.0;
      for (long i = 0; i < a.size(); ++i) m = std::max(m, std::abs(a(i) - b(i)));
      return m;
    };
    std::array<double, 2> conv{max_absdiff(ref_vec.F, layout.F_view_slice(F_skij)),
                               max_absdiff(ref_vec.S, layout.S_view_slice(S_tskij))};
    diis_timers::mix_conv.stop();

    // Store the mixed result to prev_vec to use in the next mixing step
    diis_timers::mix_prev_store.start();
    layout.store_slice(prev_vec, F_skij, S_tskij);
    has_prev = true;
    diis_timers::mix_prev_store.stop();
    diis_timers::mix_total.stop();
    return conv;
  }
};

} // namespace iter_scf

#endif // COQUI_MIXING_HPP
