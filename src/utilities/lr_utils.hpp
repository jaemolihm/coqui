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


#ifndef UTILITIES_LR_UTILS_HPP
#define UTILITIES_LR_UTILS_HPP

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <string_view>
#include <utility>
#include <vector>

#include "IO/app_loggers.h"
#include "utilities/check.hpp"
#include "utilities/proc_grid_partition.hpp"
#include "numerics/distributed_array/nda.hpp"
#include "mpi3/communicator.hpp"
#include "nda/nda.hpp"
#include "nda/blas.hpp"

namespace utils {

/**
 * @brief Compute k+q mapping for linear response calculations
 *
 * Given a k-point grid and a perturbation wavevector q, compute the mapping
 * kpq_map[ik] = ik' where k[ik] + q = k[ik'] (mod G).
 *
 * @param kpts_crys   - [INPUT] k-points in crystal coordinates (nkpts, 3)
 * @param q_vec       - [INPUT] perturbation wavevector in crystal coordinates (3,)
 * @param kpq_map     - [OUTPUT] k → k+q index mapping (nkpts,)
 * @param threshold   - [INPUT] tolerance for k-point matching
 */
inline void calculate_kpq_map(nda::ArrayOfRank<2> auto const& kpts_crys,
                              nda::ArrayOfRank<1> auto const& q_vec,
                              nda::ArrayOfRank<1> auto&& kpq_map,
                              double threshold = 1e-6) {
  long nkpts = kpts_crys.shape(0);
  utils::check(kpts_crys.shape(1) == 3, "calculate_kpq_map: kpts_crys.shape(1) != 3");
  utils::check(q_vec.shape(0) == 3, "calculate_kpq_map: q_vec.shape(0) != 3");
  utils::check(kpq_map.shape(0) == nkpts, "calculate_kpq_map: kpq_map size mismatch");

  kpq_map() = -1;

  for (long ik = 0; ik < nkpts; ++ik) {
    // k + q in crystal coordinates
    double kpq0 = kpts_crys(ik, 0) + q_vec(0);
    double kpq1 = kpts_crys(ik, 1) + q_vec(1);
    double kpq2 = kpts_crys(ik, 2) + q_vec(2);

    // Find k' such that k' = k + q (mod G)
    for (long ikp = 0; ikp < nkpts; ++ikp) {
      double d0 = std::abs(kpts_crys(ikp, 0) - kpq0);
      double d1 = std::abs(kpts_crys(ikp, 1) - kpq1);
      double d2 = std::abs(kpts_crys(ikp, 2) - kpq2);

      // Apply periodic boundary conditions
      d0 -= std::floor(d0);
      d1 -= std::floor(d1);
      d2 -= std::floor(d2);
      d0 -= std::round(d0);
      d1 -= std::round(d1);
      d2 -= std::round(d2);

      if (d0 * d0 + d1 * d1 + d2 * d2 < threshold * threshold) {
        kpq_map(ik) = ikp;
        break;
      }
    }

    utils::check(kpq_map(ik) >= 0,
                 "calculate_kpq_map: Could not find k+q for ik={}, k=({}, {}, {}), q=({}, {}, {})",
                 ik, kpts_crys(ik, 0), kpts_crys(ik, 1), kpts_crys(ik, 2),
                 q_vec(0), q_vec(1), q_vec(2));
  }
}

/**
 * @brief Check if q is commensurate with the k-point grid
 *
 * @param kpts_crys   - [INPUT] k-points in crystal coordinates (nkpts, 3)
 * @param q_vec       - [INPUT] perturbation wavevector in crystal coordinates (3,)
 * @param threshold   - [INPUT] tolerance for k-point matching
 * @return true if q is commensurate, false otherwise
 */
inline bool is_q_commensurate(nda::ArrayOfRank<2> auto const& kpts_crys,
                              nda::ArrayOfRank<1> auto const& q_vec,
                              double threshold = 1e-6) {
  long nkpts = kpts_crys.shape(0);

  for (long ik = 0; ik < nkpts; ++ik) {
    double kpq0 = kpts_crys(ik, 0) + q_vec(0);
    double kpq1 = kpts_crys(ik, 1) + q_vec(1);
    double kpq2 = kpts_crys(ik, 2) + q_vec(2);

    bool found = false;
    for (long ikp = 0; ikp < nkpts; ++ikp) {
      double d0 = std::abs(kpts_crys(ikp, 0) - kpq0);
      double d1 = std::abs(kpts_crys(ikp, 1) - kpq1);
      double d2 = std::abs(kpts_crys(ikp, 2) - kpq2);

      d0 -= std::floor(d0);
      d1 -= std::floor(d1);
      d2 -= std::floor(d2);
      d0 -= std::round(d0);
      d1 -= std::round(d1);
      d2 -= std::round(d2);

      if (d0 * d0 + d1 * d1 + d2 * d2 < threshold * threshold) {
        found = true;
        break;
      }
    }
    if (!found) return false;
  }
  return true;
}

/**
 * @brief Check if q is approximately zero (Gamma point)
 *
 * @param q_vec       - [INPUT] perturbation wavevector in crystal coordinates (3,)
 * @param threshold   - [INPUT] tolerance
 * @return true if q is approximately zero, false otherwise
 */
inline bool is_q_gamma(nda::ArrayOfRank<1> auto const& q_vec, double threshold = 1e-6) {
  double d0 = std::abs(q_vec(0));
  double d1 = std::abs(q_vec(1));
  double d2 = std::abs(q_vec(2));

  // Apply periodic boundary conditions
  d0 -= std::floor(d0);
  d1 -= std::floor(d1);
  d2 -= std::floor(d2);
  d0 -= std::round(d0);
  d1 -= std::round(d1);
  d2 -= std::round(d2);

  return d0 * d0 + d1 * d1 + d2 * d2 < threshold * threshold;
}

/**
 * @brief Find IBZ index of a given q-vector
 *
 * Searches k-points (k-grid == q-grid) for the one matching q_vec (mod G),
 * then maps to IBZ via qp_to_ibz.
 *
 * @param kpts_crys   - [INPUT] k-points in crystal coordinates (nkpts, 3)
 * @param q_vec       - [INPUT] perturbation wavevector in crystal coordinates (3,)
 * @param qp_to_ibz   - [INPUT] full BZ q-point → IBZ q-point mapping (nkpts,)
 * @param threshold   - [INPUT] tolerance for k-point matching
 * @return IBZ index of q_vec
 */
inline long find_q_ibz_index(nda::ArrayOfRank<2> auto const& kpts_crys,
                             nda::ArrayOfRank<1> auto const& q_vec,
                             nda::ArrayOfRank<1> auto const& qp_to_ibz,
                             double threshold = 1e-6) {
  long nkpts = kpts_crys.shape(0);
  utils::check(kpts_crys.shape(1) == 3, "find_q_ibz_index: kpts_crys.shape(1) != 3");
  utils::check(q_vec.shape(0) == 3, "find_q_ibz_index: q_vec.shape(0) != 3");
  utils::check(qp_to_ibz.shape(0) == nkpts, "find_q_ibz_index: qp_to_ibz size mismatch");

  for (long ik = 0; ik < nkpts; ++ik) {
    double d0 = std::abs(kpts_crys(ik, 0) - q_vec(0));
    double d1 = std::abs(kpts_crys(ik, 1) - q_vec(1));
    double d2 = std::abs(kpts_crys(ik, 2) - q_vec(2));

    // Apply periodic boundary conditions
    d0 -= std::floor(d0);
    d1 -= std::floor(d1);
    d2 -= std::floor(d2);
    d0 -= std::round(d0);
    d1 -= std::round(d1);
    d2 -= std::round(d2);

    if (d0 * d0 + d1 * d1 + d2 * d2 < threshold * threshold) {
      return qp_to_ibz(ik);
    }
  }

  utils::check(false, "find_q_ibz_index: Could not find q=({}, {}, {}) in k-point grid",
               q_vec(0), q_vec(1), q_vec(2));
  return -1; // unreachable
}

/**
 * @brief Find full-BZ index of a given q-vector
 *
 * Searches a full-BZ q-grid for the entry matching q_vec (mod G).
 * Mirrors find_q_ibz_index but returns the raw full-BZ index (no IBZ folding).
 *
 * @param qpts_crys   - [INPUT] q-points in crystal coordinates (nkpts, 3)
 * @param q_vec       - [INPUT] target q-vector in crystal coordinates (3,)
 * @param threshold   - [INPUT] tolerance for q-point matching
 * @return full-BZ index of q_vec
 */
inline long find_q_full_index(nda::ArrayOfRank<2> auto const& qpts_crys,
                              nda::ArrayOfRank<1> auto const& q_vec,
                              double threshold = 1e-6) {
  long nkpts = qpts_crys.shape(0);
  utils::check(qpts_crys.shape(1) == 3, "find_q_full_index: qpts_crys.shape(1) != 3");
  utils::check(q_vec.shape(0) == 3, "find_q_full_index: q_vec.shape(0) != 3");

  for (long ik = 0; ik < nkpts; ++ik) {
    double d0 = std::abs(qpts_crys(ik, 0) - q_vec(0));
    double d1 = std::abs(qpts_crys(ik, 1) - q_vec(1));
    double d2 = std::abs(qpts_crys(ik, 2) - q_vec(2));

    // Apply periodic boundary conditions
    d0 -= std::floor(d0);
    d1 -= std::floor(d1);
    d2 -= std::floor(d2);
    d0 -= std::round(d0);
    d1 -= std::round(d1);
    d2 -= std::round(d2);

    if (d0 * d0 + d1 * d1 + d2 * d2 < threshold * threshold) {
      return ik;
    }
  }

  utils::check(false, "find_q_full_index: Could not find q=({}, {}, {}) in q-point grid",
               q_vec(0), q_vec(1), q_vec(2));
  return -1; // unreachable
}

/// q-local distribution: pgrid = {tpools, 1, np_P, np_Q}
/// q-axis is local (undivided). Distributes over τ and PQ.
inline auto lr_W_q_local_dist(long nproc, long nt, long NP)
    -> std::tuple<std::array<long,4>, std::array<long,4>>
{
  long tpools = find_proc_grid_max_npools(nproc, nt, 0.2);
  long np_PQ = nproc / tpools;
  long np_P = find_proc_grid_min_diff(np_PQ, 1, 1);
  long np_Q = np_PQ / np_P;

  std::array<long, 4> pgrid = {tpools, 1, np_P, np_Q};
  // Per-dimension block sizes: each rank gets 1 SLATE tile per PQ dimension.
  // This ensures the block distribution is recognized as 2D cyclic by SLATE.
  long P_bs = std::max(NP / std::max(np_P, 1L), 1L);
  long Q_bs = std::max(NP / std::max(np_Q, 1L), 1L);
  std::array<long, 4> bsize = {1, 1, P_bs, Q_bs};

  return {pgrid, bsize};
}

/// τ-local distribution: pgrid = {1, qpools, np_P, np_Q}
/// First axis (τ or ω) is local (undivided). Distributes over q and PQ.
/// Used as intermediate distribution for tau_to_w / lr_dyson_W_in_place / w_to_tau.
inline auto lr_W_tau_local_dist(long nproc, long nq, long NP)
    -> std::tuple<std::array<long,4>, std::array<long,4>>
{
  long np = nproc;

  long nqpools = find_proc_grid_max_npools(np, nq, 0.2);
  np /= nqpools;
  long np_P = find_proc_grid_min_diff(np, 1, 1);
  long np_Q = np / np_P;

  std::array<long, 4> pgrid = {1, nqpools, np_P, np_Q};
  // Per-dimension block sizes: each rank gets 1 SLATE tile per PQ dimension.
  // This ensures the block distribution is recognized as 2D cyclic by SLATE.
  long P_bs = std::max(NP / std::max(np_P, 1L), 1L);
  long Q_bs = std::max(NP / std::max(np_Q, 1L), 1L);
  std::array<long, 4> bsize = {1, 1, P_bs, Q_bs};

  return {pgrid, bsize};
}

/// LR ω-side distribution: pgrid = {nwpools, nqpools, np_P, np_Q}.
/// Mirrors scr_coulomb_t::W_omega_proc_grid: maximize nqpools, then nwpools,
/// then split the remainder between np_P and np_Q. Used by lr_dyson_W_in_place
/// and the W_full(iω) buffer it reads.
///
/// Assumes the (P, Q) global axes have the same length (THC: NP==NQ); the
/// returned PQ block sizes are derived from NP for both axes.
inline auto lr_W_proc_grid(long nproc, long nq, long nw_half, long NP)
    -> std::tuple<std::array<long,4>, std::array<long,4>>
{
  long np = nproc;
  long nqpools = find_proc_grid_max_npools(np, nq, 0.2);
  np /= nqpools;
  long nwpools = find_proc_grid_max_npools(np, nw_half, 0.2);
  np /= nwpools;
  long np_P = find_proc_grid_min_diff(np, 1, 1);
  long np_Q = np / np_P;

  check(nqpools > 0 && nqpools <= nq,
        "lr_W_proc_grid: nqpools <= 0 or nqpools > nq. nqpools={}", nqpools);
  check(nwpools > 0 && nwpools <= nw_half,
        "lr_W_proc_grid: nwpools <= 0 or nwpools > nw_half. nwpools={}", nwpools);
  check(nwpools * nqpools * np_P * np_Q == nproc,
        "lr_W_proc_grid: pgrid product != nproc ({} * {} * {} * {} != {})",
        nwpools, nqpools, np_P, np_Q, nproc);

  std::array<long, 4> pgrid = {nwpools, nqpools, np_P, np_Q};
  std::array<long, 4> bsize = {1, 1, 1, 1};
  bsize[2] = std::min({1024L, NP / std::max(np_P, 1L), NP / std::max(np_Q, 1L)});
  if (bsize[2] < 1) bsize[2] = 1;
  bsize[3] = bsize[2];
  return {pgrid, bsize};
}

/// Validate that a distributed 4D array follows the lr_W_q_local_dist pattern:
/// pgrid[1] == 1 (q undivided).
template<typename darray_t>
void check_W_q_local_dist(const darray_t& d, const std::string& caller) {
  auto pgrid = d.grid();
  check(pgrid[1] == 1,
        "{}: expected q-local dist (pgrid[1]==1), got pgrid=({},{},{},{})",
        caller, pgrid[0], pgrid[1], pgrid[2], pgrid[3]);
}

/// Debug switch: force the gemm k<->R path (instead of the blocked FFT) in the
/// LR solvers when the env var COQUI_LR_DEBUG_GEMM_FT is set to a non-zero value.
inline bool lr_debug_gemm_ft() {
  static const bool flag = [] {
    const char* env = std::getenv("COQUI_LR_DEBUG_GEMM_FT");
    return env != nullptr && std::string_view(env) != "0";
  }();
  return flag;
}

/**
 * Transpose first two axes of a distributed 4D array: (A, B, P, Q) → (B, A, P, Q).
 *
 * Requires one of the first two axes to be undivided (pgrid=1), which allows
 * a purely local reorder with no MPI communication.
 * All LR call sites satisfy this precondition (τ-dist has pgrid[1]==1).
 */
template<typename Array_4D_t, typename communicator_t>
auto transpose_axes_01(
    memory::darray_t<Array_4D_t, communicator_t>& d_in,
    communicator_t& comm) {
  auto [g0, g1, g2, g3] = d_in.grid();
  auto [s0, s1, s2, s3] = d_in.global_shape();
  auto [b0, b1, b2, b3] = d_in.block_size();

  check(g0 == 1 || g1 == 1,
      "transpose_axes_01: one of the first two axes must be undivided "
      "(pgrid[0]={}, pgrid[1]={})", g0, g1);

  auto d_out = math::nda::make_distributed_array<Array_4D_t>(
      comm, {g1, g0, g2, g3}, {s1, s0, s2, s3}, {b1, b0, b2, b3});

  auto in_loc = d_in.local();
  auto out_loc = d_out.local();
  long n0 = d_in.local_shape()[0];
  long n1 = d_in.local_shape()[1];
  check(d_out.local_shape()[0] == n1 && d_out.local_shape()[1] == n0,
      "transpose_axes_01: output local shape ({},{}) != expected ({},{})",
      d_out.local_shape()[0], d_out.local_shape()[1], n1, n0);
  for (long i01 = 0; i01 < n0 * n1; ++i01) {
    long i0 = i01 / n1;
    long i1 = i01 % n1;
    out_loc(i1, i0, nda::ellipsis{}) = in_loc(i0, i1, nda::ellipsis{});
  }
  comm.barrier();
  return d_out;
}

/**
 * @brief Partition identity of this rank in a global element-striping of a
 *        node-replicated array.
 *
 * The elements of a flattened node-replicated array are cut into `nparts` =
 * comm.size() contiguous parts, one per rank, so that quantities derived from
 * the array (a DIIS history, a "previous iterate" copy, a norm) are stored and
 * computed once globally instead of once per node.
 *
 * Parts are numbered by (node index, rank-within-node) rather than by global
 * rank: global ranks are not guaranteed to be numbered contiguously by node,
 * and each node owning one *contiguous* run of parts is what lets the
 * completion step be a single allgatherv over the node roots.
 */
struct lr_part_map {
  long part_idx   = 0;  ///< this rank's part, in [0, nparts)
  long nparts     = 1;  ///< == comm.size()
  long node_first = 0;  ///< first part owned by this rank's node
  long node_count = 1;  ///< number of parts owned by this rank's node
  /// Exclusive scan of the per-node part counts (size n_nodes + 1, last entry
  /// == nparts). Filled on node roots only — it is what the completion
  /// allgatherv needs to build its counts/displacements.
  std::vector<long> node_part_offsets;

  /// Contiguous [i0, i1) slice of [0, n) owned by part `idx` of `np`.
  /// Deterministic and identical on every rank.
  static std::pair<long, long> slice(long n, long np, long idx) {
    const long chunk = (n + np - 1) / np;
    const long i0 = std::min(idx * chunk, n);
    const long i1 = std::min(i0 + chunk, n);
    return {i0, i1};
  }

  /// This rank's slice of [0, n).
  std::pair<long, long> my_slice(long n) const { return slice(n, nparts, part_idx); }
};

/**
 * @brief Build the lr_part_map for `mpi`. Collective on mpi.comm.
 *
 * Node roots all-gather their node_comm size over internode_comm and
 * exclusive-scan it; the result is broadcast within each node.
 */
template<typename MPI_Context_t>
lr_part_map make_lr_part_map(MPI_Context_t& mpi) {
  lr_part_map pmap;
  pmap.nparts     = mpi.comm.size();
  pmap.node_count = mpi.node_comm.size();

  long node_first = 0;
  if (mpi.node_comm.root()) {
    const long n_nodes = mpi.internode_comm.size();
    std::vector<long> sizes(n_nodes, 0);
    long my_size = pmap.node_count;
    mpi.internode_comm.all_gather_n(&my_size, 1, sizes.data());
    pmap.node_part_offsets.assign(n_nodes + 1, 0);
    for (long j = 0; j < n_nodes; ++j)
      pmap.node_part_offsets[j + 1] = pmap.node_part_offsets[j] + sizes[j];
    check(pmap.node_part_offsets[n_nodes] == pmap.nparts,
          "make_lr_part_map: per-node sizes sum to {} != comm.size() {}",
          pmap.node_part_offsets[n_nodes], pmap.nparts);
    node_first = pmap.node_part_offsets[mpi.internode_comm.rank()];
  }
  mpi.node_comm.broadcast_n(&node_first, 1, 0);

  pmap.node_first = node_first;
  pmap.part_idx   = node_first + mpi.node_comm.rank();
  check(pmap.part_idx >= 0 && pmap.part_idx < pmap.nparts,
        "make_lr_part_map: part_idx {} outside [0, {})", pmap.part_idx, pmap.nparts);
  return pmap;
}

/**
 * @brief Complete a node-replicated array after every rank wrote only its own
 *        lr_part_map slice.
 *
 * Called on node roots only, with `data`/`n` the flattened shared array. Each
 * node holds one contiguous element run, so one allgatherv over the node roots
 * fills in the rest. Chunked because MPI counts are `int`; without it the LR
 * ΔΣ (10 GB in one call) silently truncates.
 *
 * MPI_IN_PLACE (rather than mpi3's all_gatherv_n) because the send region is
 * this node's slice *of the receive buffer*, which a non-in-place allgatherv
 * may not alias.
 */
template<typename Comm>
void lr_complete_node_slices(Comm& internode_comm, lr_part_map const& pmap,
                             ComplexType* data, long n) {
  const long n_nodes = internode_comm.size();
  if (n_nodes <= 1 || n <= 0) return;
  check((long)pmap.node_part_offsets.size() == n_nodes + 1,
        "lr_complete_node_slices: node_part_offsets not built (call on node roots only)");

  // First element owned by each node; node j covers [beg[j], beg[j+1]).
  std::vector<long> beg(n_nodes + 1);
  for (long j = 0; j < n_nodes; ++j)
    beg[j] = lr_part_map::slice(n, pmap.nparts, pmap.node_part_offsets[j]).first;
  beg[n_nodes] = n;

  const long max_chunk = 1000000000L;  // < 2^31, so counts/displs fit in int
  std::vector<int> counts(n_nodes), displs(n_nodes);
  for (long c0 = 0; c0 < n; c0 += max_chunk) {
    const long c1 = std::min(c0 + max_chunk, n);
    for (long j = 0; j < n_nodes; ++j) {
      const long a = std::max(beg[j], c0);
      const long b = std::min(beg[j + 1], c1);
      counts[j] = (int)std::max(0L, b - a);
      displs[j] = (counts[j] > 0) ? (int)(a - c0) : 0;
    }
    MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_CXX_DOUBLE_COMPLEX,
                   data + c0, counts.data(), displs.data(),
                   MPI_CXX_DOUBLE_COMPLEX, internode_comm.handle());
  }
}

/**
 * @brief Frobenius norm and successive-iterate difference norm from pre-sliced
 *        1D views, reduced over `comm`.
 *
 * The striped counterpart of lr_distributed_norm: the caller has already cut
 * both arrays down to this rank's lr_part_map slice (the "previous iterate" is
 * only stored as that slice), so there is nothing left to re-slice here. The
 * union of the slices covers the array exactly once, so the all_reduce gives
 * the full norm on every rank.
 *
 * @param comm         - [INPUT] communicator the slices are striped over
 * @param a            - [INPUT] this rank's slice of the current array
 * @param b            - [INPUT] this rank's slice of the previous array
 * @param compute_diff - [INPUT] whether to also compute ||a - b||_F
 * @return {||A||_F, ||A - A_prev||_F}
 */
template<typename Comm, typename ArrA, typename ArrB>
std::pair<double, double> lr_striped_norm(Comm& comm, ArrA const& a, ArrB const& b,
                                          bool compute_diff) {
  if (compute_diff) {
    check(a.size() == b.size(),
          "lr_striped_norm: slice sizes differ ({} vs {})", a.size(), b.size());
  }
  double n2 = 0.0, d2 = 0.0;
  if (a.size() > 0) {
    n2 = std::real(nda::blas::dotc(a, a));
    if (compute_diff) {
      d2 = nda::sum(nda::map([](auto x, auto y) {
        double d = std::abs(x - y);
        return d * d;
      })(a, b));
    }
  }
  double acc[2] = {n2, d2};
  comm.all_reduce_in_place_n(acc, 2, std::plus<>{});
  return {std::sqrt(acc[0]), std::sqrt(acc[1])};
}

/**
 * @brief Distributed Frobenius norm and successive-iterate difference norm.
 *
 * Computes ||A||_F and ||A - A_prev||_F with the work distributed over the
 * ranks of `comm`. Since the Frobenius norm is independent of how the elements
 * are grouped, the array is viewed as 1D and each rank takes a contiguous slice
 * of the element range; the two partial sums are then all-reduced so every rank
 * returns identical totals.
 *
 * `A` and `A_prev` must be contiguous (e.g. a shared-array `.local()` view) and
 * share the same shape. The difference norm reads `A - A_prev` lazily (no
 * temporary is allocated). When `compute_diff` is false the second component is
 * returned as 0.
 *
 * Typical use: pass `mpi.node_comm` with node-replicated shared arrays; every
 * rank on the node then holds the full norm. (A trailing global broadcast is
 * still appropriate if exact cross-node bit-agreement is required.)
 *
 * @param comm         - [INPUT] communicator the block work is striped over
 * @param A            - [INPUT] current array/view, rank >= 2
 * @param A_prev       - [INPUT] previous array/view (same shape as A)
 * @param compute_diff - [INPUT] whether to also compute ||A - A_prev||_F
 * @return {||A||_F, ||A - A_prev||_F}
 */
template<typename Comm, typename ArrA, typename ArrB>
std::pair<double, double> lr_distributed_norm(Comm& comm,
                                              ArrA const& A,
                                              ArrB const& A_prev,
                                              bool compute_diff) {
  // The Frobenius norm is partition-invariant, so view the (contiguous) array
  // as 1D and give each rank a contiguous slice of the element range.
  const long n = A.size();
  if (compute_diff) {
    check(static_cast<long>(A_prev.size()) == n,
          "lr_distributed_norm: A and A_prev sizes differ ({} vs {})", A_prev.size(), n);
  }
  const long nr = comm.size();
  const long r = comm.rank();
  const long chunk = (n + nr - 1) / nr;
  const long i0 = std::min(r * chunk, n);
  const long i1 = std::min(i0 + chunk, n);

  double n2 = 0.0, d2 = 0.0;
  if (i1 > i0) {
    auto a_sub = nda::reshape(A, std::array<long, 1>{n})(nda::range(i0, i1));
    n2 = std::real(nda::blas::dotc(a_sub, a_sub));  // ||slice||^2 via BLAS
    if (compute_diff) {
      auto b_sub = nda::reshape(A_prev, std::array<long, 1>{A_prev.size()})(nda::range(i0, i1));
      // ||a - b||^2 as a no-alloc lazy reduction (BLAS can't take the lazy a-b).
      d2 = nda::sum(nda::map([](auto x, auto y) {
        double d = std::abs(x - y);
        return d * d;
      })(a_sub, b_sub));
    }
  }

  double acc[2] = {n2, d2};
  comm.all_reduce_in_place_n(acc, 2, std::plus<>{});
  return {std::sqrt(acc[0]), std::sqrt(acc[1])};
}

} // namespace utils

#endif // UTILITIES_LR_UTILS_HPP
