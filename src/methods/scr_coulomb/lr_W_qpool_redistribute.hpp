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
#include <utility>

#include "configuration.hpp"
#include "itertools/itertools.hpp"
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
 * q-pool-aligned redistribute between the FT-buffer distribution and the LR ω-side
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
 * redistribution collapses to a redistribute *inside* the m-rank q-pool
 * {q·m … q·m+m−1} instead of a global one.
 *
 * The move is math::nda::redistribute, but issued on the q-pool sub-communicator
 * rather than on the world one: each side's local block is wrapped as a
 * distributed_array_view over the m ranks of the pool, whose common global shape
 * is the pool's own box (nw, nq_loc, NP, NQ). Both views then live on the *same*
 * communicator and the generic path applies unchanged.
 *
 * Why the world-communicator redistribute is unusable here:
 *  (i) it asserts communicator congruence
 *      (redistribute_alltoallv, nda_utils.hpp: `*A.communicator() == *B.communicator()`,
 *      and mpi3's operator== accepts only MPI_CONGRUENT). A permuted split is
 *      MPI_SIMILAR, never congruent, so the generic path refuses the two full
 *      arrays outright -- there is no per-call fallback to it, which is why the
 *      strategy is picked once, up front, by lr_W_omega_layout_for().
 *  (ii) even if it accepted them it would issue an MPI_Alltoallv over the whole
 *      communicator, where an m-rank (single-node) redistribute suffices.
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

/// Block of `extent` owned by tile `itile` of `ntiles`, for the block quantum
/// `b` a distributed_array stores. Mirrors make_distributed_array's rule
/// (nda_utils.hpp): chunk_range over whole blocks, remainder to the last tile.
inline std::pair<long, long> lr_W_qpool_tile(long extent, long ntiles, long itile, long b) {
  auto [org, end] = itertools::chunk_range(0L, extent / std::max(b, 1L), ntiles, itile);
  org *= b;
  long len = (itile == ntiles - 1) ? extent - org : end * b - org;
  return {org, len};
}

/**
 * Wrap the two layouts' local blocks as distributed arrays over the q-pool
 * communicator, and assert everything the pairing rests on.
 *
 * The pool's box is (nw, nq_loc, NP, NQ): the buffer side splits its (P, Q) axes
 * over the m ranks, the ω side splits ω. Both views therefore describe the same
 * global data on the same communicator, which is all redistribute needs -- it
 * reads only origin() and local_shape() off the descriptors.
 */
template<class dBuf, class dOmg>
auto lr_W_qpool_views(mpi3::communicator& comm_world, mpi3::communicator& qpool_comm,
                      dBuf& buf, dOmg& omg) {
  const long m = qpool_comm.size();
  const long p = qpool_comm.rank();
  auto bg = buf.grid(), bs = buf.global_shape(), bo = buf.origin(), bl = buf.local_shape();
  auto wg = omg.grid(), ws = omg.global_shape(), wo = omg.origin(), wl = omg.local_shape();
  auto bb = buf.block_size(), wb = omg.block_size();
  const long m_P = bg[2], m_Q = bg[3];

  utils::check(bs == ws, "lr_W_qpool_views: global shape mismatch between the two layouts");
  utils::check(bg[0] == 1, "lr_W_qpool_views: buffer ω axis must be undivided, got {}", bg[0]);
  utils::check(wg[2] == 1 && wg[3] == 1,
               "lr_W_qpool_views: ω-side (P,Q) must be local, got ({},{})", wg[2], wg[3]);
  utils::check(m == m_P * m_Q && m == wg[0],
               "lr_W_qpool_views: q-pool size {} != m_P·m_Q ({}) or ω pools ({})",
               m, m_P * m_Q, wg[0]);
  utils::check(bl[0] == bs[0] && wl[2] == ws[2] && wl[3] == ws[3],
               "lr_W_qpool_views: local block is not full along the undivided axes "
               "(ω {} of {}, (P,Q) {}x{} of {}x{})",
               bl[0], bs[0], wl[2], wl[3], ws[2], ws[3]);
  // Both layouts must hand this rank the same q-tile: the q axis is a bystander
  // of the redistribute and becomes the views' second (undivided) axis.
  utils::check(bo[1] == wo[1] && bl[1] == wl[1],
               "lr_W_qpool_views: q ranges differ between the two layouts "
               "(buffer [{},{}), ω [{},{}))", bo[1], bo[1] + bl[1], wo[1], wo[1] + wl[1]);

  // The pool-rank → grid-position map, asserted rather than assumed. Both arrays
  // come from make_distributed_array, which places world rank r row-major over
  // the proc grid: on the buffer grid (1, nq, m_P, m_Q) that is
  // (pP, pQ) = ((r/m_Q) % m_P, r % m_Q), and with p = r % m and m = m_P·m_Q that
  // is (p/m_Q, p%m_Q). The ω array is built on the communicator permuted by
  // lr_W_qpool_key, where r sits at (r%m)·nq + r/m, so on the grid (m, nq, 1, 1)
  // it owns ω tile r % m = p. Recompute the blocks those positions own and demand
  // the arrays agree; a wrong map would otherwise silently misroute everything.
  utils::check(p == (long)comm_world.rank() % m,
               "lr_W_qpool_views: q-pool rank {} != world rank {} mod {}",
               p, (long)comm_world.rank(), m);
  auto [P_org, P_len] = lr_W_qpool_tile(bs[2], m_P, p / m_Q, bb[2]);
  auto [Q_org, Q_len] = lr_W_qpool_tile(bs[3], m_Q, p % m_Q, bb[3]);
  utils::check(bo[2] == P_org && bl[2] == P_len && bo[3] == Q_org && bl[3] == Q_len,
               "lr_W_qpool_views: buffer (P,Q) block [{},{})x[{},{}) is not the one grid "
               "position ({},{}) owns, [{},{})x[{},{})",
               bo[2], bo[2] + bl[2], bo[3], bo[3] + bl[3], p / m_Q, p % m_Q,
               P_org, P_org + P_len, Q_org, Q_org + Q_len);
  auto [w_org, w_len] = lr_W_qpool_tile(ws[0], m, p, wb[0]);
  utils::check(wo[0] == w_org && wl[0] == w_len,
               "lr_W_qpool_views: ω-side block [{},{}) is not ω tile {}, [{},{})",
               wo[0], wo[0] + wl[0], p, w_org, w_org + w_len);

  // redistribute_alltoallv passes per-peer element counts to MPI_Alltoallv as
  // int, and each is bounded by the sender's local block.
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

/**
 * FT-buffer layout → ω-side layout, within one q-pool.
 *
 * @param comm_world  - [IN] the unpermuted communicator both splits came from
 * @param qpool_comm  - [IN] the m ranks sharing a q-tile
 * @param buf         - [IN] an array in the FT-buffer distribution
 * @param omg         - [OUT] an array in the (permuted) ω-side distribution
 */
template<class dBuf, class dOmg>
void lr_W_qpool_redistribute_forward(mpi3::communicator& comm_world, mpi3::communicator& qpool_comm,
                        dBuf& buf, dOmg& omg) {
  auto [bv, ov] = detail::lr_W_qpool_views(comm_world, qpool_comm, buf, omg);
  math::nda::redistribute(bv, ov);
}

/// ω-side layout → FT-buffer layout (the exact transpose of lr_W_qpool_redistribute_forward).
template<class dOmg, class dBuf>
void lr_W_qpool_redistribute_backward(mpi3::communicator& comm_world, mpi3::communicator& qpool_comm,
                         dOmg& omg, dBuf& buf) {
  auto [bv, ov] = detail::lr_W_qpool_views(comm_world, qpool_comm, buf, omg);
  math::nda::redistribute(ov, bv);
}

} // solvers
} // methods

#endif // COQUI_LR_W_QPOOL_REDISTRIBUTE_HPP
