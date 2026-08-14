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


#ifndef UTILITIES_H5_FLAT_SLICE_HPP
#define UTILITIES_H5_FLAT_SLICE_HPP

#include <array>
#include <string>
#include <tuple>
#include <utility>

#include "configuration.hpp"
#include "utilities/check.hpp"

#include "h5/h5.hpp"
#include "nda/nda.hpp"
#include "nda/h5.hpp"

namespace utils {

namespace detail {

template<int Rank, size_t... I>
auto ranges_as_tuple(std::array<::nda::range, Rank> const& r, std::index_sequence<I...>) {
  return std::make_tuple(r[I]...);
}

} // namespace detail

/**
 * @brief Read the flat (C-order) element range [i0, i1) of a rank-`Rank`
 *        dataset into the contiguous buffer `out`.
 *
 * Lets a rank read only the element slice it owns, instead of every rank (or
 * every node) materializing the whole dataset. A contiguous run of flat indices
 * is not one rectangular hyperslab, but it is the union of at most 2*Rank-1 of
 * them: a partial block at each end of every axis plus one full-width block in
 * between. Each emitted block is itself a contiguous run of flat indices, so it
 * is read straight into its own segment of `out` — no staging copy anywhere.
 *
 * @param grp   - [INPUT] group holding the dataset
 * @param name  - [INPUT] dataset name
 * @param dims  - [INPUT] dataset shape
 * @param i0,i1 - [INPUT] flat element range, 0 <= i0 <= i1 <= prod(dims)
 * @param out   - [OUTPUT] buffer of at least (i1 - i0) elements
 */
template<int Rank>
void h5_read_flat_range(h5::group& grp, std::string const& name,
                        std::array<long, Rank> const& dims,
                        long i0, long i1, ComplexType* out) {
  static_assert(Rank >= 1, "h5_read_flat_range: Rank must be >= 1");
  if (i1 <= i0) return;

  // stride[k] = number of elements per unit step of axis k (C order).
  std::array<long, Rank> stride;
  stride[Rank - 1] = 1;
  for (int k = Rank - 2; k >= 0; --k) stride[k] = stride[k + 1] * dims[k + 1];
  check(i0 >= 0 and i1 <= stride[0] * dims[0],
        "h5_read_flat_range: range [{}, {}) outside the dataset ({} elements)",
        i0, i1, stride[0] * dims[0]);

  std::array<::nda::range, Rank> rng;
  long pos = i0;  // next flat index to read, so out + (pos - i0) is where it goes

  // One rectangular block: [beg, end) on `level`, the full extent below it, and
  // whatever the callers above fixed on the axes above it.
  auto block = [&](int level, long beg, long end) {
    for (int k = level + 1; k < Rank; ++k) rng[k] = ::nda::range(0, dims[k]);
    rng[level] = ::nda::range(beg, end);
    std::array<long, Rank> shape;
    long n = 1;
    for (int k = 0; k < Rank; ++k) { shape[k] = rng[k].size(); n *= shape[k]; }
    ::nda::array_view<ComplexType, Rank> v(shape, out + (pos - i0));
    ::nda::h5_read(grp, name, v,
                   detail::ranges_as_tuple<Rank>(rng, std::make_index_sequence<Rank>{}));
    pos += n;
  };

  // [a, b) are flat offsets relative to the start of the sub-block selected by
  // the axes above `level` (the whole dataset at level 0).
  auto rec = [&](auto&& self, int level, long a, long b) -> void {
    if (a >= b) return;
    const long L = stride[level];
    if (level == Rank - 1) { block(level, a, b); return; }  // L == 1 here
    const long ia = a / L, ib = (b - 1) / L;
    if (ia == ib) {  // wholly inside one index of this axis
      rng[level] = ::nda::range(ia, ia + 1);
      self(self, level + 1, a - ia * L, b - ia * L);
      return;
    }
    long lo = ia;
    if (a % L != 0) {  // partial head
      rng[level] = ::nda::range(ia, ia + 1);
      self(self, level + 1, a - ia * L, L);
      lo = ia + 1;
    }
    const long hi = (b % L == 0) ? ib + 1 : ib;  // full-width middle [lo, hi)
    if (hi > lo) block(level, lo, hi);
    if (b % L != 0 and ib >= lo) {  // partial tail
      rng[level] = ::nda::range(ib, ib + 1);
      self(self, level + 1, 0, b - ib * L);
    }
  };

  rec(rec, 0, i0, i1);
  check(pos == i1, "h5_read_flat_range: covered [{}, {}) instead of [{}, {})",
        i0, pos, i0, i1);
}

} // namespace utils

#endif // UTILITIES_H5_FLAT_SLICE_HPP
