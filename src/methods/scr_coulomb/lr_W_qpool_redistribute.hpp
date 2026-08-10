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


#ifndef COQUI_LR_W_QPOOL_REDISTRIBUTE_HPP
#define COQUI_LR_W_QPOOL_REDISTRIBUTE_HPP

#include <algorithm>
#include <array>
#include <limits>
#include <tuple>
#include <utility>

#include "configuration.hpp"
#include "nda/nda.hpp"
#include "numerics/distributed_array/nda.hpp"
#include "numerics/distributed_array/nda_utils.hpp"

#include "utilities/check.hpp"
#include "utilities/lr_utils.hpp"
#include "utilities/mpi_context.h"
#include "methods/scr_coulomb/scr_coulomb_fourier_t.h"

namespace methods {
namespace solvers {

/**
 * Crossing between the two distributions of the ΔW Dyson path,
 *
 *   buffer  (1, nq, m_P, m_Q)   (ω local, q and PQ split)  -- what the FT produces
 *   ω-side  (m, nq, 1,   1  )   (ω and q split, PQ local)  -- what the Dyson needs
 *
 * with m = m_P·m_Q. Both hold the same data and differ only in which of the m ranks
 * of a q-pool owns which piece, so with the ω side built on the communicator
 * permuted by lr_W_qpool_key every rank keeps its q-tile and the crossing is a
 * math::nda::redistribute *inside* the pool: each side's local block is wrapped as a
 * distributed_array_view over the pool's own box (nw, nq_loc, NP, NQ).
 *
 * The same call on the world communicator is not an option — redistribute asserts
 * communicator congruence (nda_utils.hpp:723) and a permuted split is MPI_SIMILAR,
 * which is why the strategy is picked up front by lr_W_omega_layout_for instead of
 * fallen back to per call — and it would alltoall over all nproc ranks where m
 * on-node ranks suffice.
 */

/// Rank permutation aligning the ω-side q-tiles with the FT-buffer ones: position of
/// global rank `r` in the permuted communicator. The transpose of the (m × nqpools)
/// index matrix, hence a bijection on [0, m·nqpools).
inline long lr_W_qpool_key(long r, long m, long nqpools) {
  return (r % m) * nqpools + r / m;
}

/**
 * Layout of the ω side of the ΔW pipeline, and which of the three strategies runs.
 * Single source of truth for both the runtime (compute_W_full_omega /
 * solve_lr_dyson_W) and the reporting (lr_driver::print_memory_estimate,
 * print_distribution_summary) -- re-deriving it in the report is how the two drift.
 *
 *   C : m == 1. The FT buffer already has (P, Q) local, so the ω side adopts it.
 *       Both fused FT branches fire; 2 redistribute hops per iteration.
 *   A : the q-pool redistribute (this header). 2 hops + 2 intra-pool redistributes.
 *   B : anything else. 4 hops, and the ω-side FT staging buffer is really used.
 */
struct lr_W_omega_layout {
  bool use_qpool_redistribute = false;  // strategy A
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
  L.use_qpool_redistribute = g1 && g2 && g3 && g4 && g5;
  L.need_ft_buffer_w   = not L.use_qpool_redistribute;
  return L;
}

namespace detail {

/**
 * Wrap the two layouts' local blocks as distributed arrays over the q-pool
 * communicator. Both then describe the same global box on the same communicator,
 * which is all redistribute needs -- it reads only origin() and local_shape().
 *
 * The pairing rests on pool rank p owning buffer grid position (p/m_Q, p%m_Q) and ω
 * tile p, which is make_distributed_array's row-major placement. That is asserted
 * only through the grid shapes below, not re-derived: a wrong placement cannot pass
 * silently, since the pool's blocks would stop tiling the box and
 * redistribute_alltoallv's coverage asserts would fire (nda_utils.hpp:838).
 */
template<class dBuf, class dOmg>
auto lr_W_qpool_views(mpi3::communicator& comm_world, mpi3::communicator& qpool_comm,
                      dBuf& buf, dOmg& omg) {
  const long m = qpool_comm.size();
  auto bg = buf.grid(), bs = buf.global_shape(), bo = buf.origin(), bl = buf.local_shape();
  auto wg = omg.grid(), ws = omg.global_shape(), wo = omg.origin(), wl = omg.local_shape();
  const long m_P = bg[2], m_Q = bg[3];

  // The views below hardcode origin 0 on the axes each side keeps whole, and take
  // this rank's q-tile as the pool's q extent, so those must really be whole and
  // must agree between the two layouts.
  utils::check(bs == ws and bl[0] == bs[0] and wl[2] == ws[2] and wl[3] == ws[3] and
               bo[1] == wo[1] and bl[1] == wl[1],
               "lr_W_qpool_views: layouts not paired -- buffer ω {} of {}, q [{},{}); "
               "ω-side (P,Q) {}x{} of {}x{}, q [{},{})",
               bl[0], bs[0], bo[1], bo[1] + bl[1],
               wl[2], wl[3], ws[2], ws[3], wo[1], wo[1] + wl[1]);
  utils::check(m == m_P * m_Q and m == wg[0] and
               qpool_comm.rank() == (long)comm_world.rank() % m,
               "lr_W_qpool_views: pool of {} ranks does not match m_P·m_Q ({}), ω pools ({}) "
               "or world rank {}", m, m_P * m_Q, wg[0], (long)comm_world.rank());
  // redistribute_alltoallv passes per-peer element counts to MPI_Alltoallv as int,
  // and each is bounded by the sender's local block.
  const long max_block = std::max((long)buf.local().size(), (long)omg.local().size());
  utils::check(max_block <= (long)std::numeric_limits<int>::max(),
               "lr_W_qpool_views: local block of {} elements exceeds the MPI int count limit",
               max_block);

  std::array<long, 4> gshape = {bs[0], bl[1], bs[2], bs[3]};
  std::array<long, 4> unit = {1, 1, 1, 1};
  auto bloc = buf.local();
  auto oloc = omg.local();
  return std::make_pair(
      math::nda::distributed_array_view<decltype(bloc), mpi3::communicator>(
          std::addressof(qpool_comm), std::array<long, 4>{1, 1, m_P, m_Q}, gshape,
          std::array<long, 4>{0, 0, bo[2], bo[3]}, unit, bloc),
      math::nda::distributed_array_view<decltype(oloc), mpi3::communicator>(
          std::addressof(qpool_comm), std::array<long, 4>{m, 1, 1, 1}, gshape,
          std::array<long, 4>{wo[0], 0, 0, 0}, unit, oloc));
}

} // detail

/// FT-buffer layout → ω-side layout, within one q-pool. `comm_world` is the
/// unpermuted communicator both splits came from, `qpool_comm` the m ranks sharing
/// a q-tile.
template<class dBuf, class dOmg>
void lr_W_qpool_redistribute_forward(mpi3::communicator& comm_world, mpi3::communicator& qpool_comm,
                                     dBuf& buf, dOmg& omg) {
  auto [bv, ov] = detail::lr_W_qpool_views(comm_world, qpool_comm, buf, omg);
  math::nda::redistribute(bv, ov);
}

/// ω-side layout → FT-buffer layout (the exact transpose of the above).
template<class dOmg, class dBuf>
void lr_W_qpool_redistribute_backward(mpi3::communicator& comm_world, mpi3::communicator& qpool_comm,
                                      dOmg& omg, dBuf& buf) {
  auto [bv, ov] = detail::lr_W_qpool_views(comm_world, qpool_comm, buf, omg);
  math::nda::redistribute(ov, bv);
}

} // solvers
} // methods

#endif // COQUI_LR_W_QPOOL_REDISTRIBUTE_HPP
