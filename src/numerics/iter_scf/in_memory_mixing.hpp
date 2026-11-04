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

#ifndef COQUI_IN_MEMORY_MIXING_HPP
#define COQUI_IN_MEMORY_MIXING_HPP

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
 * In-memory, element-sliced mixing of the pair (F, Sigma) of node-shared arrays, for
 * the ground-state Dyson SCF (F, Sigma(tau)) and the linear response (dF, dSigma(tau)
 * or dV_QPGW; dSigma may be absent).
 *
 * Every rank owns one contiguous slice of the flattened F and of the flattened
 * Sigma (utils::part_map, node-major parts over the global comm), keeps only its
 * slices of the mixing history, and writes the mixed result in place into its slices
 * of the node-shared windows. The windows are completed afterwards
 * (self_energy_unflatten). The whole job then stores each history vector once.
 *
 * Why not VSpace/diis_alg (disk path of diis_t)? That stack holds whole FockSigma
 * vectors on one rank behind a file-I/O Vector interface (HDF5-backed VSpace), and its
 * residual kernel assumes full-array access. Here every rank sees only a slice, the
 * residual arrives precomputed, and the arrays are too large to round-trip through
 * disk. The small dense solve for the DIIS coefficients is shared
 * (compute_diis_coefs_c1).
 */
namespace iter_scf {

/// One vector of the sliced pair: this rank's slices of F and Sigma (S empty if absent).
struct flat_self_energy {
  nda::array<ComplexType, 1> F, S;
};

/// This rank's cut of the pair (F, Sigma) of flattened sizes nF and nS (nS = 0: F only).
class self_energy_flatten {
public:
  self_energy_flatten() = default;

  /// Not collective.
  self_energy_flatten(utils::part_map pmap, long nF, long nS)
      : _pmap(std::move(pmap)), _nF(nF), _nS(nS) {
    std::tie(_f0, _f1) = _pmap.my_slice(_nF);
    std::tie(_s0, _s1) = _pmap.my_slice(_nS);
  }

  utils::part_map const& pmap() const { return _pmap; }
  long nF() const { return _nF; }
  long nS() const { return _nS; }
  /// Flat element ranges [i0, i1) of F and Sigma owned by this rank.
  std::pair<long, long> F_range() const { return {_f0, _f1}; }
  std::pair<long, long> S_range() const { return {_s0, _s1}; }
  /// Slice lengths.
  long mF() const { return _f1 - _f0; }
  long mS() const { return _s1 - _s0; }
  /// Ranges of the F and Sigma parts of the concatenation [F; Sigma] of the slices.
  nda::range rF() const { return nda::range(0, mF()); }
  nda::range rS() const { return nda::range(mF(), mF() + mS()); }

  /// 1D view of this rank's slice of the full array `full` (F or Sigma).
  template<typename A> auto F(A&& full) const {
    return nda::reshape(full, std::array<long, 1>{_nF})(nda::range(_f0, _f1));
  }
  template<typename A> auto S(A&& full) const {
    return nda::reshape(full, std::array<long, 1>{_nS})(nda::range(_s0, _s1));
  }

  /// v <- this rank's slices of the full arrays.
  template<typename AF, typename AS>
  void store(flat_self_energy& v, AF const& F_full, AS const& S_full) const {
    v.F = F(F_full);
    v.S = S(S_full);
  }

private:
  utils::part_map _pmap;
  long _nF = 0, _nS = 0;
  long _f0 = 0, _f1 = 0, _s0 = 0, _s1 = 0;
};

/**
 * @brief Complete the node-shared arrays after every rank wrote only its slices of
 * them (utils::complete_shared_slices). Collective on mpi.comm.
 *
 * Each node then holds valid data only for the slices of its own ranks; the
 * completed array is exact when the mixing is elementwise with coefficients identical
 * on every rank, and every producer of the arrays ends in an internode all_reduce, so
 * that no node holds a private variant of the input.
 */
template<typename MPI_Context_t, typename... sArr_t>
void self_energy_unflatten(MPI_Context_t& mpi, self_energy_flatten const& flat, sArr_t&... arrs) {
  utils::complete_shared_slices(mpi, flat.pmap(), arrs...);
}

/// x <- mixing * x + (1 - mixing) * prev on this rank's slices; nothing for mixing >= 1.
template<typename AF, typename AS>
void damp_slices(self_energy_flatten const& lay, double mixing, AF&& F, AS&& S,
                 flat_self_energy const& prev) {
  if (mixing >= 1.0) return;
  auto F_loc = lay.F(F);
  F_loc = mixing * F_loc + (1.0 - mixing) * prev.F;
  if (lay.mS() > 0) {
    auto S_loc = lay.S(S);
    S_loc = mixing * S_loc + (1.0 - mixing) * prev.S;
  }
}

/**
 * @brief DIIS over sliced vectors: a ring of (trial, residual) pairs and the residual
 * overlap matrix B.
 *
 * The caller supplies the residual of each trial (the Dyson-SCF commutator, or the
 * difference of successive iterates in the LR). B_ij = <r_i|r_j> summed over the two
 * blocks, each rank forming its slice of the dot products and one all_reduce
 * completing them, so that every rank holds the same B and the same coefficients.
 * Extrapolation needs more than `warmup` steps and at least `min_subsp` stored pairs;
 * otherwise, or when the coefficients do not exist, the step damps.
 */
class in_memory_diis {
public:
  struct config {
    double mixing;       ///< damping factor of the warmup and fallback steps
    size_t max_subsp;    ///< stored (trial, residual) pairs
    size_t warmup;       ///< steps that only damp
    size_t min_subsp;    ///< pairs needed to extrapolate
    int log_level;
  };

  /// A max_subsp below min_subsp can never extrapolate: the engine then only damps.
  explicit in_memory_diis(config c) : _cfg(c), _B(0, 0) {
    if (not active())
      app_log(1, "[WARNING] DIIS: a subspace of depth {} can never reach the {} pairs needed "
                 "to extrapolate; the iteration will only damp (mixing = {}). Raise the "
                 "subspace depth to use DIIS.", _cfg.max_subsp, _cfg.min_subsp, _cfg.mixing);
  }

  config const& cfg() const { return _cfg; }
  /// Whether the subspace can ever reach min_subsp.
  bool active() const { return _cfg.max_subsp >= _cfg.min_subsp; }
  /// mix() calls since reset().
  size_t steps() const { return _steps; }
  /// Live (trial, residual) pairs.
  size_t size() const { return _n; }

  /// Forget the subspace, keeping every history buffer.
  void reset() {
    _n = 0;
    _head = 0;
    _steps = 0;
    _B = nda::matrix<ComplexType>(0, 0);
  }

  flat_self_energy const& newest_trial() const { return _slots[slot(_n - 1)].x; }

  /**
   * @brief Store the trial (F, S) with its residual (rF, rS) in the newest slot,
   * evicting the oldest pair at max_subsp first, and extend B. Collective on comm.
   *
   * @param F, S     - full arrays of the trial (this rank's slices are stored)
   * @param rF, rS   - this rank's slices of the residual; rF may be empty
   */
  template<typename Comm, typename AF, typename AS, typename RF, typename RS>
  void push(Comm& comm, self_energy_flatten const& lay, AF const& F, AS const& S,
            RF const& rF, RS const& rS) {
    if (not active()) return;
    diis_timers::inmem_hist_store.start();
    if (_n == _cfg.max_subsp) evict_oldest();
    if (_n == _slots.size()) {
      // Growth happens only before the first eviction, so appending keeps the order.
      utils::check(_head == 0, "in_memory_diis: ring growth with head = {} != 0", _head);
      _slots.emplace_back();
    }
    auto& s = _slots[slot(_n)];
    lay.store(s.x, F, S);
    s.res.F = rF;
    s.res.S = rS;
    ++_n;
    diis_timers::inmem_hist_store.stop();
    update_B(comm);
  }

  /**
   * @brief Mix this rank's slices of (F, S) in place: the DIIS extrapolation of the
   * stored trials, or damping against `prev` when there is none. Returns true if it
   * extrapolated. Every rank takes the same branch.
   */
  template<typename AF, typename AS>
  bool mix(self_energy_flatten const& lay, AF&& F, AS&& S, flat_self_energy const& prev) {
    ++_steps;
    const int L = _cfg.log_level;
    auto damp = [&]() {
      diis_timers::inmem_damp_mix.start();
      damp_slices(lay, _cfg.mixing, F, S, prev);
      diis_timers::inmem_damp_mix.stop();
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

    diis_timers::inmem_coefs.start();
    auto C = compute_diis_coefs_c1(_B, L); // identical on every rank (B replicated)
    diis_timers::inmem_coefs.stop();
    if (not C) {
      app_log(1, "[WARNING] DIIS: the residual-overlap matrix is singular (the residuals vanish "
                 "or are linearly dependent to machine precision); skipping the extrapolation "
                 "and damping instead (mixing = {}).", _cfg.mixing);
      damp();
      return false;
    }
    log_extrapolation(*C);

    diis_timers::inmem_lincomb.start();
    auto F_loc = lay.F(F);
    auto S_loc = lay.S(S);
    F_loc() = ComplexType{0};
    S_loc() = ComplexType{0};
    // Newest pair first.
    for (size_t k = _n; k-- > 0;) {
      const ComplexType c = (*C)(k);
      auto const& x = _slots[slot(k)].x;
      F_loc += c * x.F;
      S_loc += c * x.S;
    }
    diis_timers::inmem_lincomb.stop();
    return true;
  }

private:
  struct slot_t {
    flat_self_energy x, res;
  };

  config _cfg;
  // A fixed-capacity ring: _n live pairs, chronological pair i in slot (_head + i) % size.
  // Evicting moves _head, so no buffer is destroyed or reallocated once the ring is full.
  std::vector<slot_t> _slots;
  size_t _n = 0;
  size_t _head = 0;
  size_t _steps = 0;
  nda::matrix<ComplexType> _B;

  size_t slot(size_t i) const { return (_head + i) % _slots.size(); }

  /// Forget the oldest pair and the first row and column of B.
  void evict_oldest() {
    const size_t n = _B.shape()[0];
    nda::matrix<ComplexType> Bnew(n - 1, n - 1);
    for (size_t i = 0; i + 1 < n; ++i)
      for (size_t j = 0; j + 1 < n; ++j) Bnew(i, j) = _B(i + 1, j + 1);
    _B = Bnew;
    _head = (_head + 1) % _slots.size();
    --_n;
  }

  /// Extend B by the row and column of the newest residual: this rank's dot products over
  /// its slices of both blocks (an empty block adds nothing), one all_reduce.
  template<typename Comm>
  void update_B(Comm& comm) {
    const size_t n = _n;
    auto const& rn = _slots[slot(n - 1)].res;
    nda::array<ComplexType, 1> col(n);
    col() = 0;
    diis_timers::inmem_B_dots.start();
    for (size_t i = 0; i < n; ++i) {
      auto const& ri = _slots[slot(i)].res;
      ComplexType acc = 0.0;
      if (ri.F.size() > 0) acc += nda::blas::dotc(ri.F, rn.F);
      if (ri.S.size() > 0) acc += nda::blas::dotc(ri.S, rn.S);
      col(i) = acc;
    }
    diis_timers::inmem_B_dots.stop();
    diis_timers::inmem_B_comm.start();
    comm.all_reduce_in_place_n(col.data(), static_cast<long>(n), std::plus<>{});
    diis_timers::inmem_B_comm.stop();

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

/// Ground-state Dyson-SCF state of the in-memory mixers (damping, or DIIS with a damping
/// warmup): the slicing, the previous accepted state, and the DIIS engine for DIIS.
struct in_memory_mixing {
  self_energy_flatten lay;
  flat_self_energy prev;     ///< this rank's slices of the previous accepted (post-mix) state
  bool has_prev = false;
  double mixing = 1.0;       ///< damping factor of the damping algorithm
  std::optional<in_memory_diis> diis;  ///< DIIS only

  bool initialized() const { return lay.nF() > 0; }

  /// Cut the slices of F (nF elements) and Sigma (nS); `diis_cfg` selects DIIS over
  /// damping. Collective on mpi.comm.
  template<typename MPI_Context_t>
  void init(MPI_Context_t& mpi, long nF, long nS, double mixing_,
            std::optional<in_memory_diis::config> diis_cfg) {
    lay = self_energy_flatten(utils::make_part_map(mpi), nF, nS);
    mixing = mixing_;
    has_prev = false;
    diis.reset();
    if (diis_cfg) diis.emplace(*diis_cfg);
  }

  /// Whether the next step consumes a commutator residual.
  bool needs_residual() const { return diis and diis->active() and diis->steps() > 0; }

  /**
   * @brief One mixing step on this rank's slices of the node-shared F and Sigma
   * (trial on entry, accepted state on exit), then prev <- the accepted state.
   *
   * @param C_t - commutator residual (node-shared view), or nullptr when
   *              needs_residual() was false
   * @return per-rank {max|ref - out|_F, max|ref - out|_Sigma}, ref the previous accepted
   *         state (damped) or the newest trial (extrapolated); the caller max-reduces.
   */
  template<typename Comm, typename AF, typename AS, typename AC>
  std::array<double, 2> step(Comm& comm, AF&& F, AS&& S, const AC* C_t) {
    diis_timers::inmem_total.start();
    bool extrapolated = false;
    if (diis) {
      // Only the Sigma block of the commutator is non-zero, so no F block is stored.
      if (C_t) diis->push(comm, lay, F, S, nda::array<ComplexType, 1>{}, lay.S(*C_t));
      extrapolated = diis->mix(lay, F, S, prev);
    } else {
      diis_timers::inmem_damp_mix.start();
      damp_slices(lay, mixing, F, S, prev);
      diis_timers::inmem_damp_mix.stop();
    }

    diis_timers::inmem_conv.start();
    flat_self_energy const& ref = extrapolated ? diis->newest_trial() : prev;
    auto max_absdiff = [](auto const& a, auto const& b) {
      double m = 0.0;
      for (long i = 0; i < a.size(); ++i) m = std::max(m, std::abs(a(i) - b(i)));
      return m;
    };
    std::array<double, 2> conv{max_absdiff(ref.F, lay.F(F)), max_absdiff(ref.S, lay.S(S))};
    diis_timers::inmem_conv.stop();

    diis_timers::inmem_prev_store.start();
    lay.store(prev, F, S);
    has_prev = true;
    diis_timers::inmem_prev_store.stop();
    diis_timers::inmem_total.stop();
    return conv;
  }
};

} // namespace iter_scf

#endif // COQUI_IN_MEMORY_MIXING_HPP
