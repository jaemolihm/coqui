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
 * Three numbers describe one axis of a distributed array, and they are independent:
 *
 *   N   the global extent -- how many elements the axis has;
 *   t   the TILE COUNT   -- how many contiguous pieces those N elements are cut into;
 *   p   the process grid extent -- how many ranks the axis is spread over.
 *
 * A tile is the unit of distribution and the unit slate blocks on. Elements are cut
 * into `t` tiles, then the `t` tiles are dealt out to the `p` ranks, so a rank owns
 * one or more WHOLE tiles and `p <= t <= N` is required: `t < p` would leave a rank
 * empty, `t > N` an empty tile.
 *
 * Both cuts are itertools::chunk_range, i.e. both are balanced: cutting `M` things
 * into `n` groups gives the first `M%n` groups one extra thing, and group `i` starts
 * at `i*(M/n) + min(i,M%n)`.
 *
 * Worked example -- N = 10 elements, t = 4 tiles, p = 2 ranks:
 *
 *   elements -> tiles:  10 = 4*2 + 2, so the first 2 tiles hold 3 elements and the
 *                       other 2 hold 2.
 *   tiles -> ranks:     4 = 2*2 + 0, so each rank holds 2 tiles.
 *
 *      element   0  1  2 | 3  4  5 | 6  7 | 8  9
 *      tile          0   |    1    |   2  |  3
 *      rank      \____ 0 ______/ \____ 1 _____/
 *
 *   so tile_offset(10,4,i) = 0,3,6,8, tile_extent(10,4,i) = 3,3,2,2,
 *   tile_of(10,4,5) = 1, local_range_of_rank(10,4,2,0) = [0,6).
 *
 * On the same axis t = 2 would give tiles [0,5), [5,10) and one tile per rank, and
 * t = 10 would give one element per tile and 5 tiles per rank. All three are legal
 * partitions of the same array over the same ranks; the tile count is what picks
 * between them.
 *
 * Two consequences are the reason distributed_array stores the tile COUNT rather
 * than a tile size:
 *
 *  - the partition is a function of `(N,t)` alone, so two axes with equal extent and
 *    equal tile count have identical tile boundaries whatever their process grids --
 *    which is what slate's gemm and getri require of their operands and do not check;
 *  - no tile and no rank is empty as long as `p <= t <= N`, and per-rank loads differ
 *    by at most one element per tile. A tile SIZE cannot express the
 *    `(a+1)*r + a*(t-r)` split at all, so it has to dump the remainder somewhere.
 */

/**
 * The tile count meaning "one element per tile on every axis": no blocking, the axis
 * is spread over its ranks element by element.
 *
 * It is a value, not a function -- `utils::one_per_tile<2>` is
 * `std::array<long,2>{0,0}`. Those zeros are the "not specified" spelling that
 * resolve_tile_counts fills in, and an omitted tile-count argument gives exactly the
 * same thing, since std::array value-initializes to zeros. So spell it out only where
 * the argument cannot simply be left off: a tile count that is not the last argument,
 * one that has to be a named variable, or one worth seeing at a call site where it
 * sits next to a grid and a shape that look just like it.
 *
 * Right for an array that is a distribution and nothing more -- a pooled buffer, a
 * redistribute target, an axis that is never split. Wrong for a slate operand, where
 * it asks for `N*M` tiles of a single element each: legal, and ruinous. Nothing checks
 * that, and it is not new -- an omitted argument and the `{1,1,...}` it replaced both
 * meant a tile size of one before this change too.
 */
template<size_t R>
inline constexpr std::array<long,R> one_per_tile = {};

/// First element of tile `i` of the partition of [0,N) into `t` tiles, i.e.
/// `i*a + min(i,r)` with `a = N/t`, `r = N%t`.
inline long tile_offset(long N, long t, long i)
{
  utils::check(t > 0 and t <= N, "tile_offset: invalid tile count t:{} for N:{}", t, N);
  utils::check(i >= 0 and i < t, "tile_offset: tile index out of range i:{} t:{}", i, t);
  return long(itertools::chunk_range(0, N, t, i).first);
}

/// Number of elements in tile `i`: `a+1` for the first `r` tiles, `a` for the rest.
inline long tile_extent(long N, long t, long i)
{
  return (i+1 == t ? N : tile_offset(N,t,i+1)) - tile_offset(N,t,i);
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
 * Pick the tile count for an axis: the smallest multiple of `p_max` that keeps every
 * tile at or below `max_tile_size` elements, clamped to `N`.
 *
 *     t = min(N, p_max * ceil(N/(p_max*max_tile_size)))
 *
 * The three arguments are easy to mix up, so, in the vocabulary of the file header:
 *
 *   N              the extent of the axis, in ELEMENTS;
 *   p_max          how many RANKS the axis is spread over -- `grid[n]`, except that
 *                  two axes forced to share a partition both pass the LARGER of their
 *                  two grid extents (see balanced_tile_counts);
 *   max_tile_size  the largest a tile may get, in ELEMENTS.
 *
 * What comes back is a tile COUNT, not a length: tile `i` is `tile_extent(N,t,i)` long,
 * so with `a = N/t` and `r = N%t` the first `r` tiles hold `a+1` elements and the rest
 * hold `a`.
 *
 * A multiple of `p_max` so that every rank of that axis gets the same number of tiles.
 * The `ceil` is at least 1, so `t >= p_max` ALWAYS: however large `max_tile_size` is,
 * the count can never fall below the rank count and leave a rank empty. `t == 1`
 * therefore happens only at `p_max == 1`.
 *
 * Worked -- an N x M matrix on an np x mp process grid, `S` for max_tile_size:
 *
 * | N, M | np, mp | S | p_row, p_col | t_row, t_col | tile sizes | rank (i,j) holds |
 * |---|---|---|---|---|---|---|
 * | 3000, 3000 | 2, 4 | 1024 | 4, 4 | 4, 4 | 750, 750 | 1500 x 750 |
 * | 3000, 500 | 2, 4 | 1024 | 2, 4 | 4, 4 | 750, 125 | 1500 x 125 |
 * | 3000, 3000 | 4, 4 | 256 | 4, 4 | 12, 12 | 250, 250 | 750 x 750 |
 * | 100, 100 | 10, 10 | 100 | 10, 10 | 10, 10 | 10, 10 | 10 x 10 |
 * | 100, 100 | 10, 10 | 3 | 10, 10 | 40, 40 | 3 and 2 | 12 x 12, or 8 x 8 |
 * | 100, 100 | 1, 1 | 100 | 1, 1 | 1, 1 | 100, 100 | 100 x 100 |
 *
 * Row 1 is why `p_max` exists: the operand is square, so the row and column partitions
 * must agree and both axes take `max(2,4) = 4`. Letting each axis take its own grid
 * extent would give `t_row = 2` against `t_col = 4`, i.e. 1500-element row tiles
 * against 750-element column ones, `mt() != nt()`, and getri aborting inside slate.
 *
 * Row 4 is the case where `max_tile_size` is bigger than any tile could be anyway:
 * `ceil(100/(10*100)) = 1`, so the bound is inactive and the count lands on `p_max`,
 * one tile per rank. It does NOT collapse to `t = 1` -- that needs `p_max = 1`, row 6.
 *
 * Row 5 is the opposite end: a bound small enough to force 4 tiles per rank, and the
 * only row where the tiles are ragged -- 20 tiles of 3 elements, then 20 of 2. It is
 * also the only row where the per-rank loads differ, and the reason the last column
 * carries two entries: each rank owns 4 whole tiles, so the five ranks holding the
 * 3-element tiles get 12 elements per axis and the five holding the 2-element ones
 * get 8. Tiles are dealt out whole, so a ragged tile count is a ragged rank load.
 */
inline long balanced_tile_count(long N, long p_max, long max_tile_size)
{
  utils::check(N >= 1, "balanced_tile_count: N must be >= 1, got {}", N);
  utils::check(p_max >= 1, "balanced_tile_count: p_max must be >= 1, got {}", p_max);
  utils::check(max_tile_size >= 1, "balanced_tile_count: max_tile_size must be >= 1, got {}", max_tile_size);
  utils::check(p_max <= N, "balanced_tile_count: p_max:{} exceeds N:{}", p_max, N);
  long per_rank = (N + p_max*max_tile_size - 1)/(p_max*max_tile_size);   // >= 1
  return std::min(N, p_max*per_rank);
}

/**
 * Per-axis tile counts from per-axis maximum tile sizes: axis n gets
 * balanced_tile_count(shape[n], p[n], max_tile_size[n]), or shape[n] (one element per
 * tile) when max_tile_size[n] <= 0.
 *
 * Pass the SAME p on two axes that have to share a partition -- the square operand of
 * getrf/getri, or the contracted axis of a gemm -- normally the larger of their two
 * grid extents. p[n] = grid[n] is right for an axis with no such partner.
 */
template<size_t R>
inline std::array<long,R> balanced_tile_counts(std::array<long,R> const& shape,
                                               std::array<long,R> const& p,
                                               std::array<long,R> const& max_tile_size)
{
  std::array<long,R> t;
  for(size_t n=0; n<R; ++n)
    t[n] = (max_tile_size[n] <= 0 ? shape[n] : balanced_tile_count(shape[n],p[n],max_tile_size[n]));
  return t;
}

/**
 * Per-axis maximum tile sizes: a request for the balanced tile count that keeps every
 * tile at or below `max_size[n]` elements, left for the factory -- which knows the
 * array's shape and process grid -- to turn into counts. Its own type, so that a
 * maximum size can never be passed where a count is expected.
 *
 * The factory resolves it against the array's OWN grid extent on each axis, which is
 * right only when no two axes have to share a partition. Two axes that do -- the
 * square operand of getrf/getri, or the contracted axis of a gemm -- must go through
 * balanced_tile_counts with the paired `p` on both, and say so.
 *
 * `max_size[n] <= 0` means one element per tile, as in balanced_tile_counts.
 */
template<size_t R>
struct max_tile_sizes {
  std::array<long,R> max_size;
  // Not an aggregate, and explicit: brace elision would otherwise let a plain
  // `{...}` tile-count argument match this overload too, and every factory call
  // passing a count literal becomes ambiguous.
  explicit constexpr max_tile_sizes(std::array<long,R> s) : max_size(s) {}
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
 * A tile count of `0` on an axis means "not specified: one element per tile". Fill
 * those in from the extents of the array being described: `t[n] == 0` becomes
 * `t[n] = extents[n]`, which is the same partition as a tile size of one.
 *
 * `0` rather than `N` because a distribution helper has to be able to say "one
 * element per tile" without knowing how long the axis is -- ft_buffer_dist, for
 * instance, returns a tile count for an axis whose extent one of its callers has not
 * fixed yet -- and because it is what an omitted tile-count argument gives you,
 * std::array being value-initialized to zeros.
 *
 * Hence the asymmetry this function exists for. An array that has been CONSTRUCTED
 * stores filled-in counts, so tile_count() never returns a 0; a helper's return value
 * still carries them. Comparing the two directly reports "different distribution" for
 * two identical distributions:
 *
 *     W.tile_count()                                  == {48, 8, 1, 1}   // filled in
 *     scr_coulomb_fourier_t::ft_buffer_dist(...)       == {48, 8, 0, 0}   // as returned
 *     resolved_tile_counts(ft_buffer_dist(...), shape) == {48, 8, 1, 1}   // comparable
 *
 * So put a helper's output through this before testing it against a stored count.
 *
 * Axes with a non-positive extent carry no partition (reset or dummy-constructed
 * arrays) and are left alone.
 */
template<size_t R>
inline std::array<long,R> resolved_tile_counts(std::array<long,R> t,
                                               std::array<long,R> const& extents)
{
  for(size_t n=0; n<R; ++n) if(t[n] == 0 and extents[n] > 0) t[n] = extents[n];
  return t;
}

/**
 * Fill in the `0`s in place, as resolved_tile_counts, and then validate: every
 * resulting count must satisfy `grid[n] <= t[n] <= extents[n]`, the condition for a
 * partition with no empty rank and no empty tile. Out-of-range counts are rejected
 * rather than clamped, because a silently clamped count gives two axes different tile
 * boundaries and slate's gemm does not check that.
 *
 * Axes with a non-positive extent or grid are left alone, as in resolved_tile_counts.
 */
template<size_t R>
inline void resolve_tile_counts(std::array<long,R>& t,
                                std::array<long,R> const& extents,
                                std::array<long,R> const& grid,
                                std::string_view who)
{
  t = resolved_tile_counts(t, extents);
  for(size_t n=0; n<R; ++n) {
    if(extents[n] <= 0 or grid[n] <= 0) continue;
    utils::check(t[n] >= grid[n] and t[n] <= extents[n],
      "{}: tile count out of range - dim:{}, tile count:{}, extent:{}, grid:{}. "
      "It must satisfy grid <= tile count <= extent (0 means one element per tile).",
      who, n, t[n], extents[n], grid[n]);
  }
}

} // namespace utils

#endif
