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

#ifndef NUMERICS_SHARED_ARRAY_IO_HPP
#define NUMERICS_SHARED_ARRAY_IO_HPP

#include <optional>

#include "configuration.hpp"
#include "mpi3/communicator.hpp"
#include "nda/nda.hpp"
#include "utilities/mpi_context.h"
#include "utilities/check.hpp"

#include "numerics/shared_array/nda.hpp"

namespace math {
  namespace shm {

    namespace mpi3 = boost::mpi3;

    /**
     * Build a node-local shared_array from an input array held only on the
     * MPI global root.
     *
     * Steps:
     *   1. Resolve the global shape (broadcast a tiny `Rank`-element array
     *      from rank 0 to all ranks).
     *   2. Allocate a shared_array (one window per node, sized only on
     *      node roots).
     *   3. Global rank 0 (also a node root, by convention node 0) copies
     *      the input into its node-local window.
     *   4. `broadcast_to_nodes(0)` propagates the data across
     *      `internode_comm` to every other node's window. This call is
     *      already chunked internally (1e9 elements at a time) so it is
     *      safe for arrays whose element count exceeds INT_MAX.
     *   5. The intra-node sync at the end of `broadcast_to_nodes` makes
     *      the data visible to every rank on every node.
     *
     * Contract:
     *   - On `mpi.comm.root()`, `src_on_root` MUST hold a value.
     *   - On every other rank, `src_on_root` is ignored (typically
     *     `std::nullopt`).
     *   - This routine is collective on `mpi.comm`.
     *
     * @tparam T      element type
     * @tparam Rank   array rank
     * @param mpi     MPI context (provides comm / internode_comm / node_comm)
     * @param src_on_root  source array on the global root; nullopt elsewhere
     * @return shared_array<nda::array_view<T, Rank>> populated identically
     *         on every node.
     */
    template<typename T, int Rank>
    auto make_shared_from_root_input(
        utils::mpi_context_t<mpi3::communicator, mpi3::shared_communicator>& mpi,
        const std::optional<::nda::array<T, Rank>>& src_on_root)
    {
      using Array_view_t = ::nda::array_view<T, Rank>;

      std::array<long, Rank> shape{};
      if (mpi.comm.root()) {
        utils::check(src_on_root.has_value(),
                     "make_shared_from_root_input: rank 0 must provide source array");
        shape = src_on_root->shape();
      }
      mpi.comm.broadcast_n(shape.data(), Rank, 0);

      auto sArr = make_shared_array<Array_view_t>(mpi, shape);

      if (mpi.comm.root()) {
        sArr.local() = *src_on_root;
      }
      // Internode bcast (one rank per node) + intra-node sync; chunked internally.
      sArr.broadcast_to_nodes(0);

      return sArr;
    }

    /**
     * Variant that takes an explicit shape rather than deducing it from the
     * source array. Use this when callers already know the expected shape
     * (e.g. from a checkpoint or another array) and want to validate it
     * collectively or when `src_on_root` may be a view into a larger buffer.
     */
    template<typename T, int Rank>
    auto make_shared_from_root_input(
        utils::mpi_context_t<mpi3::communicator, mpi3::shared_communicator>& mpi,
        std::array<long, Rank> shape,
        const std::optional<::nda::array<T, Rank>>& src_on_root)
    {
      using Array_view_t = ::nda::array_view<T, Rank>;

      if (mpi.comm.root()) {
        utils::check(src_on_root.has_value(),
                     "make_shared_from_root_input: rank 0 must provide source array");
        utils::check(src_on_root->shape() == shape,
                     "make_shared_from_root_input: shape mismatch on rank 0");
      }

      auto sArr = make_shared_array<Array_view_t>(mpi, shape);

      if (mpi.comm.root()) {
        sArr.local() = *src_on_root;
      }
      sArr.broadcast_to_nodes(0);

      return sArr;
    }

  } // namespace shm
} // namespace math

#endif // NUMERICS_SHARED_ARRAY_IO_HPP
