/**
 * ==========================================================================
 * CoQuí: Correlated Quantum ínterface
 *
 * Copyright (c) 2022-2025 Simons Foundation & The CoQuí developer team
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


#ifndef COQUI_DARRAY_UTILS_HPP
#define COQUI_DARRAY_UTILS_HPP

#include "nda/nda.hpp"
#include "numerics/distributed_array/nda.hpp"
#include "utilities/check.hpp"
#include "utilities/proc_grid_partition.hpp"

namespace utils {

/**
 * Transpose first two axes of a distributed 4D array: (A, B, P, Q) → (B, A, P, Q).
 * Works with any processor grid. Gathers to full array, transposes, then redistributes.
 *
 * TODO: This all_reduce gathers the FULL array on every rank — O(N) memory per rank.
 * Fine for small arrays but will not scale for large systems. Replace with
 * MPI_Alltoallv-based distributed transpose when this becomes a bottleneck.
 */
template<typename Array_4D_t, typename communicator_t>
auto transpose_axes_01(
    memory::darray_t<Array_4D_t, communicator_t>& d_in,
    communicator_t& comm) {
  auto [dim0, dim1, dim2, dim3] = d_in.global_shape();

  // Gather full array on all ranks
  Array_4D_t full(dim0, dim1, dim2, dim3);
  full() = 0.0;
  {
    auto loc = d_in.local();
    auto [o0, o1, o2, o3] = d_in.origin();
    auto [n0, n1, n2, n3] = d_in.local_shape();
    for (long i0 = 0; i0 < n0; ++i0)
      for (long i1 = 0; i1 < n1; ++i1)
        for (long i2 = 0; i2 < n2; ++i2)
          for (long i3 = 0; i3 < n3; ++i3)
            full(i0 + o0, i1 + o1, i2 + o2, i3 + o3) = loc(i0, i1, i2, i3);
  }
  comm.all_reduce_in_place_n(full.data(), full.size(), std::plus<>{});

  // Transpose axes 0 and 1
  Array_4D_t transposed(dim1, dim0, dim2, dim3);
  for (long i0 = 0; i0 < dim0; ++i0)
    for (long i1 = 0; i1 < dim1; ++i1)
      transposed(i1, i0, nda::ellipsis{}) = full(i0, i1, nda::ellipsis{});

  // Create output distributed array with swapped grid axes
  auto [g0, g1, g2, g3] = d_in.grid();
  auto d_out = math::nda::make_distributed_array<Array_4D_t>(
      comm, {g1, g0, g2, g3}, {dim1, dim0, dim2, dim3});

  // Fill local portion from transposed array
  {
    auto out_loc = d_out.local();
    auto [o0, o1, o2, o3] = d_out.origin();
    auto [n0, n1, n2, n3] = d_out.local_shape();
    for (long i0 = 0; i0 < n0; ++i0)
      for (long i1 = 0; i1 < n1; ++i1)
        for (long i2 = 0; i2 < n2; ++i2)
          for (long i3 = 0; i3 < n3; ++i3)
            out_loc(i0, i1, i2, i3) = transposed(i0 + o0, i1 + o1, i2 + o2, i3 + o3);
  }
  comm.barrier();
  return d_out;
}

} // namespace utils

#endif // COQUI_DARRAY_UTILS_HPP
