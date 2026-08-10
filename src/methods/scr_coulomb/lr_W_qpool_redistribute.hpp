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
 *   buffer  (1, nqpools, m_P,     m_Q  )  (ω local, q and PQ split) -- the FT's own
 *   ω-side  (nwpools, nqpools, np_P, np_Q)  (ω and q split)         -- the Dyson's
 *
 * with m := m_P·m_Q == nwpools·np_P·np_Q ranks per q-pool. Both hold the same data
 * and differ only in which of the m ranks of a q-pool owns which piece, so with the ω
 * side built on the communicator permuted by lr_W_qpool_key every rank keeps its
 * q-tile and the crossing is a math::nda::redistribute *inside* the pool: each side's
 * local block is wrapped as a distributed_array_view over the pool's own box
 * (nw, nq_loc, NP, NQ).
 *
 * The same call on the world communicator is not an option: redistribute asserts
 * communicator congruence (nda_utils.hpp:723) and a permuted split is MPI_SIMILAR.
 * Hence the permutation is fixed once, up front, by lr_W_omega_layout_for.
 */

/**
 * Rank permutation aligning the ω-side q-tiles with the FT-buffer ones: position of
 * global rank `r` in the permuted communicator.
 *
 * make_distributed_array walks the grid axes from the fastest (C order), so on the
 * buffer grid (1, nqpools, m_P, m_Q) rank r sits at q-tile qB = r/m and in-pool slot
 * p = r % m, while on the ω grid (nwpools, nqpools, np_P, np_Q) position s sits at
 * q-tile (s/u) % nqpools with u := np_P·np_Q. Writing p = w·u + c, demanding the two
 * q-tiles agree gives
 *
 *   key(r) = w·(nqpools·u) + qB·u + c ,  qB = r/m, w = (r%m)/u, c = (r%m)%u
 *
 * i.e. r = qB·m + w·u + c and key(r) are the mixed-radix representations of the same
 * digits (qB, w, c) in radices (nqpools, nwpools, u) and (nwpools, nqpools, u): the
 * key is a transposition of the first two digits, hence a bijection on [0, nproc),
 * with key(0) == 0 and inverse w = s/(nqpools·u), qB = (s/u) % nqpools, c = s % u.
 * At u == 1 it reduces to (r % m)·nqpools + r/m.
 */
inline long lr_W_qpool_key(long r, long m, long nqpools, long u) {
  const long qB = r / m, p = r % m;
  return (p / u) * (nqpools * u) + qB * u + (p % u);
}

/**
 * The two distributions of the ω side of the ΔW pipeline. Single source of truth for
 * both the runtime (compute_W_full_omega / solve_lr_dyson_W) and the reporting
 * (lr_driver::print_memory_estimate, print_distribution_summary) -- re-deriving it in
 * the report is how the two drift.
 *
 * The FT always lands in the buffer distribution (so both of its fused branches fire)
 * and the q-pool redistribute always carries it to the ω one; m == 1 is the degenerate
 * case where the two coincide and the ω array simply aliases the buffer array.
 */
struct lr_W_omega_layout {
  long m       = 1;   // ranks per q-pool, m_P·m_Q == nwpools·u
  long nqpools = 1;   // q-pools, identical on both sides
  long nwpools = 1;   // ω-side ω-pools
  long u       = 1;   // ω-side np_P·np_Q
  std::array<long, 4> b_pgrid{}, b_bsize{};   // FT-buffer distribution
  std::array<long, 4> w_pgrid{}, w_bsize{};   // ω-side distribution
};

/**
 * Derive the ω-side layout from pure arithmetic (no communication).
 * Assumes the two THC aux axes have equal length (NP == NQ), as lr_W_proc_grid does.
 */
inline lr_W_omega_layout
lr_W_omega_layout_for(long nproc, long nq, long nw_half, long NP) {
  lr_W_omega_layout L;
  std::tie(L.b_pgrid, L.b_bsize) =
      scr_coulomb_fourier_t::ft_buffer_dist(nproc, {nw_half, nq, NP, NP});
  L.nqpools = L.b_pgrid[1];
  L.m = L.b_pgrid[2] * L.b_pgrid[3];

  std::tie(L.w_pgrid, L.w_bsize) = utils::lr_W_proc_grid(nproc, nq, nw_half, NP);
  L.nwpools = L.w_pgrid[0];
  L.u = L.w_pgrid[2] * L.w_pgrid[3];

  if (L.m == 1) {
    // One rank per q-pool: nqpools == nproc leaves lr_W_proc_grid nothing to split
    // over ω or (P, Q), so it returns the buffer *pgrid* -- but not its *bsize*: it
    // caps at 1024 while ft_buffer_dist does not, so the two differ at NP > 1024.
    // The fused FT branches compare bsize exactly
    // (scr_coulomb_fourier_t.cpp:111, :187), so taking lr_W_proc_grid's bsize here
    // would silently reintroduce an ω staging buffer and a redistribute hop. The
    // ω-side bsize at m == 1 is therefore the buffer's, not lr_W_proc_grid's.
    utils::check(L.w_pgrid == L.b_pgrid,
                 "lr_W_omega_layout_for: at m == 1 the ω-side pgrid ({},{},{},{}) must "
                 "equal the FT-buffer one ({},{},{},{})",
                 L.w_pgrid[0], L.w_pgrid[1], L.w_pgrid[2], L.w_pgrid[3],
                 L.b_pgrid[0], L.b_pgrid[1], L.b_pgrid[2], L.b_pgrid[3]);
    L.w_bsize = L.b_bsize;
  }

  utils::check(L.nqpools * L.m == nproc and L.nwpools * L.u == L.m,
               "lr_W_omega_layout_for: pool arithmetic broken -- nqpools {} · m {} != "
               "nproc {}, or nwpools {} · u {} != m",
               L.nqpools, L.m, nproc, L.nwpools, L.u);
  utils::check(L.b_bsize[1] == 1 and L.w_bsize[1] == 1,
               "lr_W_omega_layout_for: the two sides must share the q tiling, got q "
               "block quanta {} (buffer) and {} (ω side)", L.b_bsize[1], L.w_bsize[1]);
  return L;
}

namespace detail {

/**
 * Wrap the two layouts' local blocks as distributed arrays over the q-pool
 * communicator. Both then describe the same global box on the same communicator,
 * which is all redistribute needs -- it reads only origin() and local_shape().
 *
 * The pairing rests on pool rank p owning buffer grid position (p/m_Q, p%m_Q) and ω
 * grid position (p/u, (p%u)/np_Q, (p%u)%np_Q), which is make_distributed_array's
 * row-major placement under the lr_W_qpool_key permutation. That is asserted only
 * through the grid shapes below, not re-derived: a wrong placement cannot pass
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

  // The views below hardcode origin 0 on the ω axis of the buffer side (it keeps ω
  // whole) and take this rank's q-tile as the pool's q extent, so that must really be
  // whole and the q tiles must agree between the two layouts.
  utils::check(bs == ws and bl[0] == bs[0] and bo[1] == wo[1] and bl[1] == wl[1],
               "lr_W_qpool_views: layouts not paired -- buffer ω {} of {}, q [{},{}); "
               "ω-side (P,Q) {}x{} of {}x{}, q [{},{})",
               bl[0], bs[0], bo[1], bo[1] + bl[1],
               wl[2], wl[3], ws[2], ws[3], wo[1], wo[1] + wl[1]);
  utils::check(m == m_P * m_Q and m == wg[0] * wg[2] * wg[3] and
               qpool_comm.rank() == (long)comm_world.rank() % m,
               "lr_W_qpool_views: pool of {} ranks does not match m_P·m_Q ({}), "
               "nwpools·np_P·np_Q ({}) or world rank {}",
               m, m_P * m_Q, wg[0] * wg[2] * wg[3], (long)comm_world.rank());
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
          std::addressof(qpool_comm), std::array<long, 4>{wg[0], 1, wg[2], wg[3]}, gshape,
          std::array<long, 4>{wo[0], 0, wo[2], wo[3]}, unit, oloc));
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
