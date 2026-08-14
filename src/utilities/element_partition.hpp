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


#ifndef UTILITIES_ELEMENT_PARTITION_HPP
#define UTILITIES_ELEMENT_PARTITION_HPP

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

#include "configuration.hpp"
#include "utilities/check.hpp"

#include "mpi3/communicator.hpp"
#include "nda/nda.hpp"
#include "nda/blas.hpp"

/**
 * Element-wise partitioning of a node-replicated array across the ranks of a
 * global communicator, with node-major part numbering.
 *
 * `proc_grid_partition.hpp` partitions the process grid; this partitions the
 * *elements* of an array over the ranks of an existing communicator, so that a
 * quantity derived from a node-replicated array (an iterative-solver history, a
 * "previous iterate" copy, a norm) is stored and computed once globally instead
 * of once per node. Users: the LR driver / LR DIIS and the ground-state SPMD
 * Fock-Sigma DIIS.
 */
namespace utils {

/**
 * @brief Partition identity of this rank in a global element-striping of a
 *        node-replicated array.
 *
 * The elements of a flattened node-replicated array are cut into `nparts` =
 * comm.size() contiguous parts, one per rank.
 *
 * Parts are numbered by (node index, rank-within-node) rather than by global
 * rank: global ranks are not guaranteed to be numbered contiguously by node,
 * and each node owning one *contiguous* run of parts is what lets the
 * completion step be a single allgatherv over the node roots.
 */
struct part_map {
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
 * @brief Build the part_map for `mpi`. Collective on mpi.comm.
 *
 * Node roots all-gather their node_comm size over internode_comm and
 * exclusive-scan it; the result is broadcast within each node.
 */
template<typename MPI_Context_t>
part_map make_part_map(MPI_Context_t& mpi) {
  part_map pmap;
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
          "make_part_map: per-node sizes sum to {} != comm.size() {}",
          pmap.node_part_offsets[n_nodes], pmap.nparts);
    node_first = pmap.node_part_offsets[mpi.internode_comm.rank()];
  }
  mpi.node_comm.broadcast_n(&node_first, 1, 0);

  pmap.node_first = node_first;
  pmap.part_idx   = node_first + mpi.node_comm.rank();
  check(pmap.part_idx >= 0 && pmap.part_idx < pmap.nparts,
        "make_part_map: part_idx {} outside [0, {})", pmap.part_idx, pmap.nparts);
  return pmap;
}

/**
 * @brief Complete a node-replicated array after every rank wrote only its own
 *        part_map slice.
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
void complete_node_slices(Comm& internode_comm, part_map const& pmap,
                          ComplexType* data, long n) {
  const long n_nodes = internode_comm.size();
  if (n_nodes <= 1 || n <= 0) return;
  check((long)pmap.node_part_offsets.size() == n_nodes + 1,
        "complete_node_slices: node_part_offsets not built (call on node roots only)");

  // First element owned by each node; node j covers [beg[j], beg[j+1]).
  std::vector<long> beg(n_nodes + 1);
  for (long j = 0; j < n_nodes; ++j)
    beg[j] = part_map::slice(n, pmap.nparts, pmap.node_part_offsets[j]).first;
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
 * The striped counterpart of a whole-array distributed norm: the caller has
 * already cut both arrays down to this rank's part_map slice (the "previous
 * iterate" is only stored as that slice), so there is nothing left to re-slice
 * here. The union of the slices covers the array exactly once, so the
 * all_reduce gives the full norm on every rank.
 *
 * @param comm         - [INPUT] communicator the slices are striped over
 * @param a            - [INPUT] this rank's slice of the current array
 * @param b            - [INPUT] this rank's slice of the previous array
 * @param compute_diff - [INPUT] whether to also compute ||a - b||_F
 * @return {||A||_F, ||A - A_prev||_F}
 */
template<typename Comm, typename ArrA, typename ArrB>
std::pair<double, double> striped_norm(Comm& comm, ArrA const& a, ArrB const& b,
                                       bool compute_diff) {
  if (compute_diff) {
    check(a.size() == b.size(),
          "striped_norm: slice sizes differ ({} vs {})", a.size(), b.size());
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

} // namespace utils

#endif // UTILITIES_ELEMENT_PARTITION_HPP
