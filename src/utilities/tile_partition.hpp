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


#ifndef UTILITIES_TILE_PARTITION_HPP
#define UTILITIES_TILE_PARTITION_HPP

#include <algorithm>
#include <array>
#include <utility>
#include <string_view>

#include "itertools/itertools.hpp"
#include "utilities/check.hpp"

namespace utils
{

/**
 * Balanced tile partition of an axis.
 *
 * An axis of extent `N` is cut into `t` tiles by itertools::chunk_range, so with
 * `a = N/t` and `r = N%t` the first `r` tiles hold `a+1` elements and the rest hold
 * `a`; tile `i` starts at `i*a + min(i,r)`. Tiles are then handed to the `np` ranks
 * of that axis by the same rule, applied to the tile index. Two consequences are
 * the reason distributed_array stores the tile *count* rather than a tile size:
 *
 *  - the partition is a function of `(N, t)` alone, so two axes with equal extent
 *    and equal tile count have identical tile boundaries whatever their process
 *    grids -- which is what slate's gemm/getri conformability requires;
 *  - no tile and no rank is empty as long as `np <= t <= N`, and per-rank loads
 *    differ by at most one element per tile.
 *
 * `t == N` reproduces a tile size of one, i.e. the plain balanced element
 * distribution of chunk_range.
 */

/**
 * Per-axis tile count meaning "one element per tile" on every axis, i.e. the plain
 * balanced element distribution. It is the `0` sentinel resolved by
 * resolve_tile_counts, spelled so that a tile-count argument is not mistaken for an
 * origin or a shape at the call site, and so that every elementwise-tiled array is
 * greppable.
 */
template<size_t R>
inline constexpr std::array<long,R> one_per_tile = {};

/// [first,last) of tile `i` of the partition of [0,N) into `t` tiles.
inline std::pair<long,long> tile_range(long N, long t, long i)
{
  utils::check(t > 0 and t <= N, "tile_range: invalid tile count t:{} for N:{}", t, N);
  utils::check(i >= 0 and i < t, "tile_range: tile index out of range i:{} t:{}", i, t);
  auto [f,l] = itertools::chunk_range(0, N, t, i);
  return {long(f), long(l)};
}

/// First element of tile `i`.
inline long tile_offset(long N, long t, long i) { return tile_range(N,t,i).first; }

/// Number of elements in tile `i`.
inline long tile_extent(long N, long t, long i)
{
  auto [f,l] = tile_range(N,t,i);
  return l-f;
}

/// Tile owning element `idx`: the inverse of tile_offset.
inline long tile_of(long N, long t, long idx)
{
  utils::check(t > 0 and t <= N, "tile_of: invalid tile count t:{} for N:{}", t, N);
  utils::check(idx >= 0 and idx < N, "tile_of: index out of range idx:{} N:{}", idx, N);
  long a = N/t, r = N%t;
  // first r tiles have a+1 elements, the remaining t-r have a (a >= 1 since t <= N)
  if (idx < r*(a+1)) return idx/(a+1);
  return r + (idx - r*(a+1))/a;
}

/**
 * Tile count for an axis of extent `N` distributed over `p_max` ranks with tiles of
 * at most `cap` elements: the smallest multiple of `p_max` that keeps every tile at
 * or below `cap`, clamped to `N`.
 *
 * A multiple of `p_max` gives every rank the same number of tiles, and `p_max` is
 * the larger of the two grid extents of a 2-D array so that the row and column
 * partitions of a square operand agree (the getrf/getri precondition mt == nt).
 */
inline long balanced_tile_count(long N, long p_max, long cap)
{
  utils::check(N >= 1, "balanced_tile_count: N must be >= 1, got {}", N);
  utils::check(p_max >= 1, "balanced_tile_count: p_max must be >= 1, got {}", p_max);
  utils::check(cap >= 1, "balanced_tile_count: cap must be >= 1, got {}", cap);
  utils::check(p_max <= N, "balanced_tile_count: p_max:{} exceeds N:{}", p_max, N);
  long per_rank = (N + p_max*cap - 1)/(p_max*cap);   // ceil(N/(p_max*cap)) >= 1
  return std::min(N, p_max*per_rank);
}

/**
 * Per-axis tile counts from per-axis tile-size caps: axis n gets
 * balanced_tile_count(shape[n], p[n], cap[n]), or shape[n] (one element per tile)
 * when cap[n] <= 0.
 *
 * Pass the SAME p on two axes that have to share a partition -- the square
 * operand of getrf/getri, or the contracted axis of a gemm -- normally the
 * larger of their two grid extents. p[n] = grid[n] is right for an axis with no
 * such partner.
 */
template<size_t R>
inline std::array<long,R> balanced_tile_counts(std::array<long,R> const& shape,
                                               std::array<long,R> const& p,
                                               std::array<long,R> const& cap)
{
  std::array<long,R> t;
  for(size_t n=0; n<R; ++n)
    t[n] = (cap[n] <= 0 ? shape[n] : balanced_tile_count(shape[n],p[n],cap[n]));
  return t;
}

/**
 * A per-axis tile-size cap: a request for the balanced tile count that keeps every
 * tile at or below `cap[n]`, left for the factory that knows the array's shape and
 * process grid to resolve. Its own type, so that a cap can never be passed where a
 * count is expected.
 *
 * The factory resolves it against the array's OWN grid extent on each axis, which is
 * right only when no two axes have to share a partition. Two axes that do -- the
 * square operand of getrf/getri, or the contracted axis of a gemm -- must go through
 * balanced_tile_counts with the paired `p` on both, and say so.
 *
 * `cap[n] <= 0` means one element per tile, as in balanced_tile_counts.
 */
template<size_t R>
struct tile_caps {
  std::array<long,R> cap;
  // Not an aggregate, and explicit: brace elision would otherwise let a plain
  // `{...}` tile-count argument match this overload too, and every factory call
  // passing a count literal becomes ambiguous.
  explicit constexpr tile_caps(std::array<long,R> c) : cap(c) {}
};

/// [first,last) of the elements of [0,N) owned by rank `ip` of `np`, when the axis
/// is cut into `t` tiles and the tiles are dealt out by chunk_range.
inline std::pair<long,long> local_range_of_rank(long N, long t, long np, long ip)
{
  utils::check(np >= 1 and np <= t, "local_range_of_rank: invalid np:{} for t:{}", np, t);
  utils::check(ip >= 0 and ip < np, "local_range_of_rank: rank out of range ip:{} np:{}", ip, np);
  auto [t0,t1] = itertools::chunk_range(0, t, np, ip);
  return {tile_offset(N,t,long(t0)), (t1 == t ? N : tile_offset(N,t,long(t1)))};
}

/**
 * Resolve and validate a per-axis tile-count array in place.
 *
 * `t[n] == 0` is the sentinel for "one element per tile", i.e. `t[n] = extents[n]`,
 * which reproduces a tile size of one. Anything else must satisfy
 * `grid[n] <= t[n] <= extents[n]`; out-of-range counts are rejected rather than
 * clamped, because a silently clamped count gives two axes different tile
 * boundaries and slate's gemm does not check that.
 *
 * Axes with a non-positive extent or grid are left alone: those are reset or
 * dummy-constructed arrays, which carry no partition.
 */
template<size_t R>
inline void resolve_tile_counts(std::array<long,R>& t,
                                std::array<long,R> const& extents,
                                std::array<long,R> const& grid,
                                std::string_view who)
{
  for(size_t n=0; n<R; ++n) {
    if(extents[n] <= 0 or grid[n] <= 0) continue;
    if(t[n] == 0) t[n] = extents[n];
    utils::check(t[n] >= grid[n] and t[n] <= extents[n],
      "{}: tile count out of range - dim:{}, tile count:{}, extent:{}, grid:{}. "
      "It must satisfy grid <= tile count <= extent (0 means one element per tile).",
      who, n, t[n], extents[n], grid[n]);
  }
}

/**
 * Resolve the `0` sentinels of a *requested* tile count against the extents of the
 * array it will describe, without validating. Distribution helpers return the
 * sentinel (they are deliberately extent-agnostic on the pooled axes) while a
 * constructed array stores the resolved count, so the two are only comparable
 * after this. Use it wherever a stored tile_count() is tested against a helper's
 * output -- an unresolved comparison silently reports "different distribution".
 */
template<size_t R>
inline std::array<long,R> resolved_tile_counts(std::array<long,R> t,
                                               std::array<long,R> const& extents)
{
  for(size_t n=0; n<R; ++n) if(t[n] == 0) t[n] = extents[n];
  return t;
}

} // namespace utils

#endif
