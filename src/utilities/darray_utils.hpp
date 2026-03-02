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

namespace utils {

/**
 * Transpose first two axes of a distributed 4D array: (A, B, P, Q) → (B, A, P, Q).
 * Requires {1,1,1,1} processor grid (all data on root). Allocates new distributed array.
 */
template<typename Array_4D_t, typename communicator_t>
auto transpose_axes_01(
    memory::darray_t<Array_4D_t, communicator_t>& d_in,
    communicator_t& comm) {
  auto [g0, g1, g2, g3] = d_in.grid();
  utils::check(g0 == 1 && g1 == 1 && g2 == 1 && g3 == 1,
               "transpose_axes_01: requires {{1,1,1,1}} proc grid, got ({},{},{},{})",
               g0, g1, g2, g3);
  auto [dim0, dim1, dim2, dim3] = d_in.global_shape();
  auto d_out = math::nda::make_distributed_array<Array_4D_t>(
      comm, {1, 1, 1, 1}, {dim1, dim0, dim2, dim3});
  if (comm.root()) {
    auto in_loc = d_in.local();
    auto out_loc = d_out.local();
    for (long i0 = 0; i0 < dim0; ++i0)
      for (long i1 = 0; i1 < dim1; ++i1)
        out_loc(i1, i0, nda::ellipsis{}) = in_loc(i0, i1, nda::ellipsis{});
  }
  comm.barrier();
  return d_out;
}

} // namespace utils

#endif // COQUI_DARRAY_UTILS_HPP
