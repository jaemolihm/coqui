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


#ifndef COQUI_LR_W_QPOOL_EXCHANGE_HPP
#define COQUI_LR_W_QPOOL_EXCHANGE_HPP

#include <algorithm>
#include <array>
#include <limits>
#include <utility>
#include <vector>

#include "configuration.hpp"
#include "nda/nda.hpp"
#include "numerics/distributed_array/nda.hpp"

#include "utilities/check.hpp"
#include "utilities/lr_utils.hpp"
#include "utilities/mpi_context.h"
#include "methods/scr_coulomb/scr_coulomb_fourier_t.h"

namespace methods {
namespace solvers {

/**
 * q-pool-aligned exchange between the FT-buffer distribution and the LR ω-side
 * distribution of the ΔW pipeline.
 *
 * The two distributions of the W Dyson path,
 *
 *   buffer  (1, nq, m_P, m_Q)   (ω local, q and PQ split)  -- what the FT produces
 *   ω-side  (m, nq, 1,   1  )   (ω and q split, PQ local)  -- what the Dyson needs
 *
 * with m = m_P·m_Q, hold exactly the same data and differ only in *which* of the
 * m ranks of a q-pool owns which piece. If the ω-side array is built on a
 * communicator permuted by
 *
 *   key(r) = (r % m)·nq + r / m
 *
 * then a rank keeps its q-tile across the two layouts, and the whole
 * redistribution collapses to an all-to-all *inside* the m-rank q-pool
 * {q·m … q·m+m−1} instead of a global one. That is what this plan performs.
 *
 * Why not math::nda::redistribute:
 *  (i) it asserts communicator congruence
 *      (redistribute_alltoallv, nda_utils.hpp: `*A.communicator() == *B.communicator()`,
 *      and mpi3's operator== accepts only MPI_CONGRUENT). A permuted split is
 *      MPI_SIMILAR, never congruent, so the generic path refuses these operands
 *      outright -- there is no per-call fallback to it, which is why the strategy
 *      is picked once, up front, by lr_W_omega_layout_for().
 *  (ii) even if it accepted them it would issue an MPI_Alltoallv over the whole
 *      communicator, where an m-rank (single-node) exchange suffices.
 *
 * How the bytes move: m−1 blocking Sendrecv rounds around the q-pool ring plus a
 * local copy for the self block, staging through **one** temp the size of the
 * largest peer block (local/m). A derived-datatype MPI_Alltoallw with m cached
 * MPI_Type_create_subarray descriptors on the sub-block side is the documented
 * escape hatch if the serialized rounds ever measure badly; it removes the temp
 * at the price of datatype lifetime management and a raw-MPI dependency (the
 * mpi3 wrapper exposes no Alltoallw).
 */

/// Rank permutation aligning the ω-side q-tiles with the FT-buffer ones:
/// position of global rank `r` inside the permuted communicator.
/// A bijection on [0, m·nqpools) -- the transpose of the (m × nqpools) index
/// matrix -- with inverse key⁻¹(s) = (s % nqpools)·m + s / nqpools.
inline long lr_W_qpool_key(long r, long m, long nqpools) {
  return (r % m) * nqpools + r / m;
}

/**
 * Layout of the ω side of the ΔW pipeline, and which of the three strategies
 * runs. Single source of truth for both the runtime (compute_W_full_omega /
 * solve_lr_dyson_W) and the reporting (lr_driver::print_memory_estimate,
 * print_distribution_summary) -- re-deriving it in the report is how the two
 * drift apart.
 *
 *   C : m == 1. The FT buffer already has (P, Q) local, so the ω side simply
 *       adopts it. Both fused FT branches fire; 2 redistribute hops per iteration.
 *   A : the q-pool exchange (this header). 2 hops + 2 intra-pool exchanges.
 *   B : anything else. 4 hops, and the ω-side FT staging buffer is really used.
 */
struct lr_W_omega_layout {
  bool use_qpool_exchange = false;  // strategy A
  bool need_ft_buffer_w   = false;  // strategy B: the ω-side FT buffer is acquired
  long m       = 1;                 // ranks per q-pool on the buffer side
  long nqpools = 1;
  std::array<long, 4> b_pgrid{}, b_bsize{};   // FT-buffer distribution
  std::array<long, 4> w_pgrid{}, w_bsize{};   // ω-side distribution
};

/**
 * Pick the ω-side layout and strategy from pure arithmetic (no communication).
 * Assumes the two THC aux axes have equal length (NP == NQ), as lr_W_proc_grid does.
 */
inline lr_W_omega_layout
lr_W_omega_layout_for(long nproc, long nq, long nw_half, long NP) {
  lr_W_omega_layout L;
  std::tie(L.b_pgrid, L.b_bsize) =
      scr_coulomb_fourier_t::ft_buffer_dist(nproc, {nw_half, nq, NP, NP});
  L.nqpools = L.b_pgrid[1];
  L.m = L.b_pgrid[2] * L.b_pgrid[3];

  if (L.b_pgrid[2] == 1 && L.b_pgrid[3] == 1) {
    // Strategy C: the ω side reuses the FT-buffer layout, so both fused FT
    // branches fire and no ω-side staging buffer is ever acquired.
    L.w_pgrid = L.b_pgrid;
    L.w_bsize = L.b_bsize;
    return L;
  }

  std::tie(L.w_pgrid, L.w_bsize) = utils::lr_W_proc_grid(nproc, nq, nw_half, NP);

  // Strategy-A guard. G4 (a {m, nqpools, 1, 1} ω grid) is what keeps the Dyson at
  // P = Q = 1, i.e. a local gemm per (iω, q) tile; if lr_W_proc_grid ever returns
  // a PQ-split ω grid the guard simply fails and strategy B runs.
  const bool g1 = nproc > 1;                       // nproc == 1 takes the FT's single-rank path
  const bool g2 = L.nqpools > 1;
  const bool g3 = L.m > 1 && L.nqpools * L.m == nproc;
  const bool g4 = L.w_pgrid == std::array<long, 4>{L.m, L.nqpools, 1, 1};
  const bool g5 = L.b_bsize[0] == 1 && L.b_bsize[1] == 1 &&
                  L.w_bsize[0] == 1 && L.w_bsize[1] == 1;   // ω/q block quantum 1
  L.use_qpool_exchange = g1 && g2 && g3 && g4 && g5;
  L.need_ft_buffer_w   = not L.use_qpool_exchange;
  return L;
}

/**
 * Cached exchange plan between the two layouts above, for one q-pool.
 *
 * Stateful because the peer ranges and counts are constant across every solve;
 * build() reads them off the two real distributed arrays (never re-deriving
 * chunk_range boundaries by hand) and all-gathers the m peer descriptors once.
 */
class lr_W_qpool_plan {
 public:
  /**
   * @param comm_world  - [IN] the unpermuted communicator both splits came from
   * @param qpool_comm  - [IN] the m ranks sharing a q-tile; must outlive the plan
   * @param buf         - [IN] an array in the FT-buffer distribution
   * @param omg         - [IN] an array in the (permuted) ω-side distribution
   */
  template<class dA, class dB>
  void build(mpi3::communicator& comm_world, mpi3::communicator& qpool_comm,
             const dA& buf, const dB& omg) {
    _qpool_comm = std::addressof(qpool_comm);
    _m = qpool_comm.size();
    _p = qpool_comm.rank();
    // Communicator identity of the two operands, re-checked on every call. Size
    // and rank rather than mpi3's operator==, which demands MPI_CONGRUENT and so
    // reports false for two distinct wrappers around one handle (MPI_IDENT).
    // (size, rank) is exactly what separates the permuted ω-side communicator
    // from the unpermuted buffer-side one, which is the confusion to catch.
    _buf_np = buf.communicator()->size(); _buf_rank = buf.communicator()->rank();
    _omg_np = omg.communicator()->size(); _omg_rank = omg.communicator()->rank();

    auto bg = buf.grid(),  bs = buf.global_shape();
    auto wg = omg.grid(),  ws = omg.global_shape();
    utils::check(bg[0] == 1, "lr_W_qpool_plan: buffer ω axis must be undivided, got {}", bg[0]);
    utils::check(wg[2] == 1 && wg[3] == 1,
                 "lr_W_qpool_plan: ω-side (P,Q) must be local, got ({},{})", wg[2], wg[3]);
    utils::check(bs == ws, "lr_W_qpool_plan: global shape mismatch between the two layouts");
    utils::check(_m == bg[2] * bg[3] && _m == wg[0],
                 "lr_W_qpool_plan: q-pool size {} != m_P·m_Q ({}) or ω pools ({})",
                 _m, bg[2] * bg[3], wg[0]);
    utils::check(_p == (long)comm_world.rank() % _m,
                 "lr_W_qpool_plan: q-pool rank {} != world rank {} mod {}",
                 _p, (long)comm_world.rank(), _m);
    // The alignment claim: both layouts must give this rank the same q-tile.
    utils::check(buf.origin()[1] == omg.origin()[1] &&
                 buf.local_shape()[1] == omg.local_shape()[1],
                 "lr_W_qpool_plan: q ranges differ between the two layouts "
                 "(buffer [{},{}), ω [{},{}))",
                 buf.origin()[1], buf.origin()[1] + buf.local_shape()[1],
                 omg.origin()[1], omg.origin()[1] + omg.local_shape()[1]);

    _nw = bs[0]; _nq_loc = buf.local_shape()[1]; _NP = bs[2]; _NQ = bs[3];
    _m_Q = bg[3];
    utils::check(buf.local_shape()[0] == _nw,
                 "lr_W_qpool_plan: buffer ω extent {} != {}", buf.local_shape()[0], _nw);
    utils::check(omg.local_shape()[2] == _NP && omg.local_shape()[3] == _NQ,
                 "lr_W_qpool_plan: ω-side (P,Q) block is not the full {}x{}", _NP, _NQ);

    // One gather of the per-slot descriptors: the ω tile from the ω side, the
    // (P, Q) panel from the buffer side.
    std::array<long, 6> me = {omg.origin()[0],    omg.local_shape()[0],
                              buf.origin()[2],    buf.local_shape()[2],
                              buf.origin()[3],    buf.local_shape()[3]};
    std::vector<long> gathered(6 * _m);
    qpool_comm.all_gather_n(me.data(), 6, gathered.data());

    _w_org.resize(_m); _w_len.resize(_m);
    _P_org.resize(_m); _P_len.resize(_m);
    _Q_org.resize(_m); _Q_len.resize(_m);
    for (long p = 0; p < _m; ++p) {
      _w_org[p] = gathered[6*p + 0]; _w_len[p] = gathered[6*p + 1];
      _P_org[p] = gathered[6*p + 2]; _P_len[p] = gathered[6*p + 3];
      _Q_org[p] = gathered[6*p + 4]; _Q_len[p] = gathered[6*p + 5];
    }

    // The ω tiles must tile [0, nw) contiguously in slot order, and the (P, Q)
    // panels must tile [0,NP)x[0,NQ) with slot p == pP·m_Q + pQ (row-major grid
    // decomposition). Both are exactly the alignment the exchange assumes.
    long w_acc = 0;
    for (long p = 0; p < _m; ++p) {
      utils::check(_w_org[p] == w_acc && _w_len[p] >= 0,
                   "lr_W_qpool_plan: ω tiles are not a contiguous partition at slot {}", p);
      w_acc += _w_len[p];
    }
    utils::check(w_acc == _nw, "lr_W_qpool_plan: ω tiles sum to {} != {}", w_acc, _nw);

    long panel_acc = 0, max_panel = 0;
    for (long p = 0; p < _m; ++p) {
      const long pP = p / _m_Q, pQ = p % _m_Q;
      // Same P panel for every slot in a P row, same Q panel down a Q column.
      utils::check(_P_org[p] == _P_org[pP * _m_Q] && _P_len[p] == _P_len[pP * _m_Q] &&
                   _Q_org[p] == _Q_org[pQ]        && _Q_len[p] == _Q_len[pQ],
                   "lr_W_qpool_plan: (P,Q) panels do not follow the row-major grid at slot {}", p);
      panel_acc += _P_len[p] * _Q_len[p];
      max_panel = std::max(max_panel, _P_len[p] * _Q_len[p]);
    }
    utils::check(panel_acc == _NP * _NQ,
                 "lr_W_qpool_plan: (P,Q) panels cover {} elements != {}", panel_acc, _NP * _NQ);

    // One peer-sized staging temp, reused by both directions and every round:
    // forward receives a (nw_loc, nq_loc, NP_loc(p'), NQ_loc(p')) block, backward
    // packs one. The complementary side of each round is contiguous in the other
    // array and needs no staging.
    _tmp_n = _w_len[_p] * _nq_loc * max_panel;

    // mpi3 truncates *every* count it passes to MPI to int, and a Sendrecv round
    // has four: the two staged ones are bounded by _tmp_n, but the two that index
    // the buffer-side array are _w_len[peer]·buf_stride0, which exceeds _tmp_n
    // whenever a peer owns a longer ω tile than this rank. Bound the real maximum.
    const long buf_stride0 = _nq_loc * _P_len[_p] * _Q_len[_p];
    const long max_w_len = _m > 0 ? *std::max_element(_w_len.begin(), _w_len.end()) : 0;
    const long max_count = std::max(_tmp_n, max_w_len * buf_stride0);
    utils::check(max_count <= (long)std::numeric_limits<int>::max(),
                 "lr_W_qpool_plan: exchange message of {} elements exceeds the MPI int "
                 "count limit (peer block {}, buffer slab {})",
                 max_count, _tmp_n, max_w_len * buf_stride0);
    _tmp = nda::array<ComplexType, 1>(std::max(temp_size(), 1L));
  }

  bool is_built() const { return _qpool_comm != nullptr; }
  long m() const { return _m; }
  /// Elements of the peer-sized staging temp: the exchange's whole memory price,
  /// persistent for the plan's lifetime. build() allocates exactly this, and
  /// compute_W_full_omega reduces and logs it against the analytic entry in
  /// lr_driver::print_memory_estimate, so the report cannot drift unnoticed.
  long temp_size() const { return _tmp_n; }

  /// FT-buffer layout → ω-side layout.
  template<class dA, class dB>
  void forward(const dA& buf, dB& omg) {
    static_assert(_c_ordered<dA>() and _c_ordered<dB>(),
                  "lr_W_qpool_plan: operands must be C-stride-ordered");
    _validate(buf, omg);
    auto buf_loc = buf.local();
    auto omg_loc = omg.local();
    const long nwl = _w_len[_p];
    const long buf_stride0 = _nq_loc * _P_len[_p] * _Q_len[_p];

    // Self block: no message, just the local slab.
    _omg_panel(omg_loc, _p) = buf_loc(nda::range(_w_org[_p], _w_org[_p] + nwl),
                                      nda::range::all, nda::range::all, nda::range::all);

    for (long s = 1; s < _m; ++s) {
      const long dst = (_p + s) % _m;
      const long src = (_p - s + _m) % _m;
      const long send_n = _w_len[dst] * buf_stride0;
      const long recv_n = nwl * _nq_loc * _P_len[src] * _Q_len[src];
      _qpool_comm->send_receive_n(buf_loc.data() + _w_org[dst] * buf_stride0, send_n, (int)dst,
                                  _tmp.data(), recv_n, (int)src);
      _omg_panel(omg_loc, src) = _tmp_as_block(nwl, _P_len[src], _Q_len[src]);
    }
  }

  /// ω-side layout → FT-buffer layout (the exact transpose of forward).
  template<class dA, class dB>
  void backward(const dA& omg, dB& buf) {
    static_assert(_c_ordered<dA>() and _c_ordered<dB>(),
                  "lr_W_qpool_plan: operands must be C-stride-ordered");
    _validate(buf, omg);
    auto omg_loc = omg.local();
    auto buf_loc = buf.local();
    const long nwl = _w_len[_p];
    const long buf_stride0 = _nq_loc * _P_len[_p] * _Q_len[_p];

    buf_loc(nda::range(_w_org[_p], _w_org[_p] + nwl),
            nda::range::all, nda::range::all, nda::range::all) = _omg_panel(omg_loc, _p);

    for (long s = 1; s < _m; ++s) {
      const long dst = (_p + s) % _m;
      const long src = (_p - s + _m) % _m;
      const long send_n = nwl * _nq_loc * _P_len[dst] * _Q_len[dst];
      const long recv_n = _w_len[src] * buf_stride0;
      _tmp_as_block(nwl, _P_len[dst], _Q_len[dst]) = _omg_panel(omg_loc, dst);
      _qpool_comm->send_receive_n(_tmp.data(), send_n, (int)dst,
                                  buf_loc.data() + _w_org[src] * buf_stride0, recv_n, (int)src);
    }
  }

 private:
  // Both send/receive offsets below are raw pointer arithmetic into the local
  // block (`data() + iw·nq_loc·NP_loc·NQ_loc`), which is the element order only
  // for a C-ordered contiguous block. Same requirement, and same guard, as
  // lr_scr_coulomb_t::lr_dyson_W_in_place.
  template<class D>
  static constexpr bool _c_ordered() { return std::decay_t<D>::is_stride_order_C(); }

  // The staging temp seen as one peer's (nw_loc, nq_loc, NP_loc, NQ_loc) block.
  // A hand-built view rather than nda::reshape: reshaping a sliced rank-1 view up
  // to rank 4 carries its stride-order tag along and fails to compile.
  nda::array_view<ComplexType, 4> _tmp_as_block(long nwl, long np, long nqq) {
    return nda::array_view<ComplexType, 4>(
        std::array<long, 4>{nwl, _nq_loc, np, nqq}, _tmp.data());
  }

  // The (P, Q) panel of pool slot p inside an ω-side local block.
  template<class Loc>
  static auto _panel(Loc&& loc, long P_org, long P_len, long Q_org, long Q_len) {
    return loc(nda::range::all, nda::range::all,
               nda::range(P_org, P_org + P_len), nda::range(Q_org, Q_org + Q_len));
  }
  template<class Loc>
  auto _omg_panel(Loc&& loc, long p) const {
    return _panel(loc, _P_org[p], _P_len[p], _Q_org[p], _Q_len[p]);
  }

  template<class dA, class dB>
  void _validate(const dA& buf, const dB& omg) const {
    utils::check(is_built(), "lr_W_qpool_plan: used before build()");
    // Shapes alone do not identify the side: the buffer-side and ω-side arrays
    // have the same global shape and differ only in the communicator they were
    // built on, so a swapped pair would pass every check below.
    utils::check(buf.communicator()->size() == _buf_np &&
                 buf.communicator()->rank() == _buf_rank &&
                 omg.communicator()->size() == _omg_np &&
                 omg.communicator()->rank() == _omg_rank,
                 "lr_W_qpool_plan: operands are not on the communicators the plan was "
                 "built for (buffer {}/{} expected {}/{}, ω {}/{} expected {}/{})",
                 buf.communicator()->rank(), buf.communicator()->size(), _buf_rank, _buf_np,
                 omg.communicator()->rank(), omg.communicator()->size(), _omg_rank, _omg_np);
    utils::check(buf.global_shape()[0] == _nw && buf.global_shape()[2] == _NP &&
                 buf.global_shape()[3] == _NQ &&
                 buf.local_shape()[0] == _nw && buf.local_shape()[1] == _nq_loc &&
                 buf.local_shape()[2] == _P_len[_p] && buf.local_shape()[3] == _Q_len[_p],
                 "lr_W_qpool_plan: buffer-side array does not match the cached plan");
    utils::check(omg.local_shape()[0] == _w_len[_p] && omg.local_shape()[1] == _nq_loc &&
                 omg.local_shape()[2] == _NP && omg.local_shape()[3] == _NQ &&
                 omg.origin()[0] == _w_org[_p],
                 "lr_W_qpool_plan: ω-side array does not match the cached plan");
  }

  // Non-owning: points at the caller's q-pool communicator, which must outlive
  // the plan (in lr_scr_coulomb_t it is a member declared before this one).
  mpi3::communicator* _qpool_comm = nullptr;
  long _buf_np = 0, _buf_rank = -1, _omg_np = 0, _omg_rank = -1;
  long _m = 0, _p = 0, _m_Q = 1;
  long _nw = 0, _nq_loc = 0, _NP = 0, _NQ = 0;
  std::vector<long> _w_org, _w_len, _P_org, _P_len, _Q_org, _Q_len;
  nda::array<ComplexType, 1> _tmp;
  long _tmp_n = 0;
};

} // solvers
} // methods

#endif // COQUI_LR_W_QPOOL_EXCHANGE_HPP
