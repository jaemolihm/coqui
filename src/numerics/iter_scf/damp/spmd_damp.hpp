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

#ifndef COQUI_SPMD_DAMP_HPP
#define COQUI_SPMD_DAMP_HPP

#include <algorithm>
#include <array>
#include <utility>

#include "configuration.hpp"
#include "utilities/check.hpp"
#include "utilities/element_partition.hpp"

#include "nda/nda.hpp"

#include "numerics/iter_scf/diis/diis_timers.hpp"

namespace iter_scf {

/**
 * @brief Element-sliced damping of the Dyson-SCF (F, Sigma) pair with the
 * previous accepted state kept in memory.
 *
 *   out = mixing * new + (1 - mixing) * prev,   conv = max|prev - out|,
 *
 * with prev the previous accepted (post-mix) state, the same arithmetic as
 * damp_t::solve. The flattened F and Sigma are cut into contiguous slices over
 * the global comm (utils::part_map, node-major), each rank keeps only its own
 * slice of prev, so the run holds one copy of (F, Sigma) in total. The caller
 * completes the node-shared windows afterwards (utils::complete_node_slices).
 *
 * damp_t::solve re-reads prev from scf/iter{N-1}, which chkpt_slim omits for
 * Sigma; this class needs the checkpoint only when there is no in-memory prev
 * (the first mixing after a restart, staged by the caller).
 *
 * Used by the Dyson-SCF damping (damp_t::spmd) and by the warmup damping of the
 * in-memory DIIS (spmd_fs_diis).
 */
class spmd_damp {
public:
  using Vec1D = nda::array<ComplexType, 1>;

  /// Cut the slices of F (nF elements) and Sigma (nS). Collective on mpi.comm.
  template<typename MPI_Context_t>
  void init(MPI_Context_t& mpi, long nF, long nS) {
    _nF = nF;
    _nS = nS;
    _pmap = utils::make_part_map(mpi);
    std::tie(_f0, _f1) = _pmap.my_slice(_nF);
    std::tie(_s0, _s1) = _pmap.my_slice(_nS);
    _has_prev = false;
    _inited = true;
  }

  bool initialized() const { return _inited; }
  bool has_prev() const { return _has_prev; }
  long nF() const { return _nF; }
  long nS() const { return _nS; }
  utils::part_map const& pmap() const { return _pmap; }
  std::pair<long, long> F_range() const { return {_f0, _f1}; }
  std::pair<long, long> S_range() const { return {_s0, _s1}; }

  /// This rank's slice of the full (node-shared) F or Sigma.
  template<typename A> auto F_slice(A&& F) const { return slice_of(F, _nF, _f0, _f1); }
  template<typename A> auto S_slice(A&& S) const { return slice_of(S, _nS, _s0, _s1); }

  /// Record this rank's slices of the accepted state.
  void set_prev(Vec1D F_loc, Vec1D S_loc) {
    _prevF = std::move(F_loc);
    _prevS = std::move(S_loc);
    _has_prev = true;
  }

  /**
   * @brief Damp this rank's slices in place and record the result as prev.
   *
   * @param F_loc, S_loc   - this rank's slices (F_slice / S_slice of the windows)
   * @param F_prev, S_prev - staged previous state slices, used only when there
   *                         is no in-memory prev (restart edge); else nullptr
   * @return per-rank partial {max|dF|, max|dSigma|}; caller max-reduces.
   */
  template<typename FView, typename SView>
  std::array<double, 2> apply(double mixing, FView&& F_loc, SView&& S_loc,
                              const Vec1D* F_prev, const Vec1D* S_prev) {
    const Vec1D *pF = &_prevF, *pS = &_prevS;
    if (not _has_prev) {
      utils::check(F_prev != nullptr and S_prev != nullptr,
                   "spmd_damp: no previous accepted state (in memory or staged)");
      utils::check(F_prev->size() == _f1 - _f0 and S_prev->size() == _s1 - _s0,
                   "spmd_damp: staged previous state has the wrong slice size "
                   "(F {} vs {}, Sigma {} vs {})",
                   F_prev->size(), _f1 - _f0, S_prev->size(), _s1 - _s0);
      pF = F_prev;
      pS = S_prev;
    }
    std::array<double, 2> conv{0.0, 0.0};
    diis_timers::spmd_damp_mix.start();
    conv[0] = mix_slice(mixing, F_loc, *pF);
    conv[1] = mix_slice(mixing, S_loc, *pS);
    diis_timers::spmd_damp_mix.stop();
    diis_timers::spmd_prev_store.start();
    set_prev(Vec1D{F_loc}, Vec1D{S_loc});
    diis_timers::spmd_prev_store.stop();
    return conv;
  }

private:
  bool _inited = false;
  bool _has_prev = false;
  long _nF = 0, _nS = 0;
  long _f0 = 0, _f1 = 0, _s0 = 0, _s1 = 0;
  utils::part_map _pmap;
  Vec1D _prevF, _prevS;

  template<typename A>
  static auto slice_of(A&& a, long n, long i0, long i1) {
    return nda::reshape(a, std::array<long, 1>{n})(nda::range(i0, i1));
  }

  template<typename View>
  static double mix_slice(double mixing, View&& H, const Vec1D& Hprev) {
    H = mixing * H + (1.0 - mixing) * Hprev;
    double m = 0.0;
    for (long i = 0; i < H.size(); ++i) m = std::max(m, std::abs(Hprev(i) - H(i)));
    return m;
  }
};

} // iter_scf

#endif // COQUI_SPMD_DAMP_HPP
