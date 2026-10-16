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


#undef NDEBUG

#include <algorithm>
#include <array>
#include <vector>

#include "catch2/catch.hpp"

#include "configuration.hpp"
#include "IO/AppAbort.hpp"
#include "IO/app_loggers.h"

#include "utilities/tile_partition.hpp"

namespace bdft_tests
{

namespace
{

long ceil_div(long a, long b) { return (a + b - 1)/b; }

// every factorization p*q == np
std::vector<std::pair<long,long>> factorizations(long np)
{
  std::vector<std::pair<long,long>> f;
  for (long p = 1; p <= np; ++p)
    if (np%p == 0) f.push_back({p, np/p});
  return f;
}

} // anonymous

TEST_CASE("tile_partition_invariants", "[utilities]")
{
  using utils::tile_offset;
  using utils::tile_extent;
  using utils::tile_of;
  using utils::local_range_of_rank;

  std::vector<long> extents;
  for (long N = 1; N <= 64; ++N) extents.push_back(N);
  for (long N : {130l, 181l, 283l, 333l, 403l, 511l, 1687l, 2229l}) extents.push_back(N);

  const std::vector<long> nprocs = {1,2,4,8,12,16,32,64,128,192,768,1024};

  for (long N : extents) {

    // -- properties of the element -> tile map, for every tile count ------------
    std::vector<long> tcounts;
    for (long t = 1; t <= std::min(N, 70l); ++t) tcounts.push_back(t);
    if (N > 70) { tcounts.push_back(N/3); tcounts.push_back(N/2); tcounts.push_back(N-1); tcounts.push_back(N); }

    for (long t : tcounts) {
      if (t < 1 or t > N) continue;
      const long a = N/t, r = N%t;

      long covered = 0, prev_last = 0, prev_extent = 0;
      long emin = N+1, emax = -1;
      for (long i = 0; i < t; ++i) {
        const long f = tile_offset(N,t,i), l = f + tile_extent(N,t,i);
        // (1) no gap, no overlap
        REQUIRE(f == prev_last);
        REQUIRE(l > f);
        prev_last = l;
        covered += l-f;
        emin = std::min(emin, l-f);
        emax = std::max(emax, l-f);
        // (3) closed form of the offset: the t-r small tiles first, the r large last
        REQUIRE(f == i*a + std::max(0l, i-(t-r)));
        REQUIRE(tile_extent(N,t,i) == (i < t-r ? a : a+1));
        // (9) the extents are non-decreasing, which is what slate's QR needs: the
        // panel of tile column j has k = tile_extent(N,t,j) columns and accumulates
        // its k x k triangular factor in some tile i >= j, whose height tpqrt hands
        // lapack as lda. Every such tile must be at least k tall.
        REQUIRE(tile_extent(N,t,i) >= prev_extent);
        prev_extent = tile_extent(N,t,i);
      }
      // (1) tiles partition [0,N)
      REQUIRE(prev_last == N);
      REQUIRE(covered == N);
      // (2) tile extents differ by at most one
      REQUIRE(emax - emin <= 1);

      // (8) index -> tile -> offset round trip
      for (long i = 0; i < t; ++i) REQUIRE(tile_of(N,t,tile_offset(N,t,i)) == i);
      for (long idx = 0; idx < N; ++idx) {
        long i = tile_of(N,t,idx);
        const long f = tile_offset(N,t,i), l = f + tile_extent(N,t,i);
        REQUIRE(idx >= f);
        REQUIRE(idx < l);
      }
    }

    // -- properties of the tile -> rank map -------------------------------------
    for (long np : nprocs) {
      if (np > N) continue;
      for (auto [p_row,p_col] : factorizations(np)) {
        if (p_row > N or p_col > N) continue;
        const long p_max = std::max(p_row,p_col);

        for (long max_tile_size : {1l, 16l, 1024l}) {
          const long t = utils::balanced_tile_count(N, p_max, max_tile_size);
          // the helper never returns an unusable count
          REQUIRE(t >= p_max);
          REQUIRE(t <= N);
          // (6) every rank of the p_max axis owns at least one tile
          REQUIRE(t/p_max >= 1);
          // tiles honour max_tile_size whenever it is reachable, i.e. t was not
          // clamped down to N by the min
          if (t == p_max*((N + p_max*max_tile_size - 1)/(p_max*max_tile_size)))
            REQUIRE(tile_extent(N,t,0) <= max_tile_size);

          // (7) equal tile counts on two axes of equal extent => identical
          // boundaries, whatever their process grids. This is the getri mt == nt
          // precondition and the gemm contracted-axis precondition, and it is the
          // reason the stored quantity is a count and not a size: the partition
          // must be a function of (N, t) alone.
          for (long i = 0; i < t; ++i) {
            REQUIRE(tile_offset(N,t,i) == i*(N/t) + std::max(0l, i-(t-N%t)));
            REQUIRE(tile_offset(N,t,i) + tile_extent(N,t,i) ==
                    (i+1)*(N/t) + std::max(0l, i+1-(t-N%t)));
            REQUIRE(local_range_of_rank(N,t,1,0) == std::pair<long,long>{0,N});
          }

          for (long axis : {0l, 1l}) {
            const long np_a = (axis == 0 ? p_row : p_col);
            long prev = 0, lmin = N+1, lmax = -1;
            for (long ip = 0; ip < np_a; ++ip) {
              auto [f,l] = local_range_of_rank(N,t,np_a,ip);
              REQUIRE(f == prev);
              prev = l;
              // (5) no empty rank
              REQUIRE(l > f);
              lmin = std::min(lmin, l-f);
              lmax = std::max(lmax, l-f);
            }
            REQUIRE(prev == N);
            // (4) load bounds. Equality with the ideal ceil(N/np) holds exactly when
            // every rank owns a single tile; with k = t/np tiles per rank the worst
            // rank can be up to k-1 elements above the ideal.
            REQUIRE(lmax >= ceil_div(N,np_a));
            REQUIRE(lmax <= ceil_div(t,np_a)*ceil_div(N,t));
            REQUIRE(lmin >= (t/np_a)*(N/t));
            if (t == np_a) REQUIRE(lmax == ceil_div(N,np_a));
          }
        }
      }
    }
  }
}

TEST_CASE("tile_partition_reference_values", "[utilities]")
{
  // t == N is the plain balanced element partition, i.e. tile size one
  REQUIRE(utils::balanced_tile_count(403, 1, 1) == 403);
  for (long i = 0; i < 403; ++i) REQUIRE(utils::tile_extent(403,403,i) == 1);

  // N = 403 over the min_diff grids of a few rank counts, max_tile_size 1024 inactive
  REQUIRE(utils::balanced_tile_count(403, 8, 1024) == 8);
  REQUIRE(utils::balanced_tile_count(403, 32, 1024) == 32);
  REQUIRE(utils::balanced_tile_count(403, 128, 1024) == 128);
  REQUIRE(utils::balanced_tile_count(403, 403, 1024) == 403);
  // 403 = 8*50 + 3: five tiles of 50 then three of 51; max rank load 51 == ceil(403/8)
  REQUIRE(utils::tile_extent(403, 8, 0) == 50);
  REQUIRE(utils::tile_extent(403, 8, 4) == 50);
  REQUIRE(utils::tile_extent(403, 8, 5) == 51);
  REQUIRE(utils::tile_extent(403, 8, 7) == 51);
  REQUIRE(utils::local_range_of_rank(403, 8, 8, 0) == std::pair<long,long>{0,50});
  REQUIRE(utils::local_range_of_rank(403, 8, 8, 7) == std::pair<long,long>{352,403});

  // max_tile_size does bite for a large axis: 2229 over 2 ranks needs two tiles per rank
  REQUIRE(utils::balanced_tile_count(2229, 2, 1024) == 4);
  REQUIRE(utils::tile_extent(2229, 4, 0) == 557);
  REQUIRE(utils::tile_extent(2229, 4, 3) == 558);
  REQUIRE(utils::local_range_of_rank(2229, 4, 2, 0) == std::pair<long,long>{0,1114});
  REQUIRE(utils::local_range_of_rank(2229, 4, 2, 1) == std::pair<long,long>{1114,2229});

  // max_tile_size can never pull the count below p_max, so a bound larger than the
  // axis leaves one tile per rank rather than collapsing to a single tile
  REQUIRE(utils::balanced_tile_count(100, 10, 100) == 10);
  REQUIRE(utils::balanced_tile_count(100, 10, 1024) == 10);
  REQUIRE(utils::balanced_tile_count(100, 10, 10) == 10);
  REQUIRE(utils::balanced_tile_count(100, 10, 3) == 40);    // 4 tiles per rank
  REQUIRE(utils::balanced_tile_count(100, 1, 100) == 1);    // t == 1 needs p_max == 1
  for (long i = 0; i < 10; ++i) REQUIRE(utils::tile_extent(100, 10, i) == 10);

  // 1687 over a 3-rank axis: 562/562/563, not the floor-division 562 with a 563 remainder
  REQUIRE(utils::balanced_tile_count(1687, 3, 1024) == 3);
  REQUIRE(utils::tile_extent(1687, 3, 0) == 562);
  REQUIRE(utils::tile_extent(1687, 3, 1) == 562);
  REQUIRE(utils::tile_extent(1687, 3, 2) == 563);

  // inverse map on a ragged partition
  REQUIRE(utils::tile_of(1687, 3, 561) == 0);
  REQUIRE(utils::tile_of(1687, 3, 562) == 1);
  REQUIRE(utils::tile_of(1687, 3, 1123) == 1);
  REQUIRE(utils::tile_of(1687, 3, 1124) == 2);
}

TEST_CASE("tile_partition_edge_cases", "[utilities]")
{
  using utils::tile_offset;
  using utils::tile_extent;
  using utils::tile_of;
  using utils::local_range_of_rank;
  using utils::balanced_tile_count;

  // N = 1: one element, one tile, one rank
  REQUIRE(tile_offset(1,1,0) == 0);
  REQUIRE(tile_extent(1,1,0) == 1);
  REQUIRE(tile_of(1,1,0) == 0);
  REQUIRE(local_range_of_rank(1,1,1,0) == std::pair<long,long>{0,1});

  // t = 1: the whole axis in one tile, so every element maps to tile 0
  REQUIRE(tile_offset(7,1,0) == 0);
  REQUIRE(tile_extent(7,1,0) == 7);
  for (long idx = 0; idx < 7; ++idx) REQUIRE(tile_of(7,1,idx) == 0);

  // t = N: one element per tile, and the index maps are the identity
  for (long i = 0; i < 7; ++i) {
    REQUIRE(tile_offset(7,7,i) == i);
    REQUIRE(tile_extent(7,7,i) == 1);
    REQUIRE(tile_of(7,7,i) == i);
  }

  // a ragged partition, 7 = 3*2 + 1: tiles [0,2), [2,4), [4,7)
  REQUIRE(tile_offset(7,3,0) == 0);
  REQUIRE(tile_offset(7,3,2) == 4);          // first element of the last tile
  REQUIRE(tile_extent(7,3,0) == 2);
  REQUIRE(tile_extent(7,3,2) == 3);          // the one large tile, and it is last
  REQUIRE(tile_of(7,3,0) == 0);              // first element of the axis
  REQUIRE(tile_of(7,3,3) == 1);              // last element before the large tile
  REQUIRE(tile_of(7,3,4) == 2);              // first element of it
  REQUIRE(tile_of(7,3,6) == 2);              // last element of the axis
  REQUIRE(local_range_of_rank(7,3,3,2) == std::pair<long,long>{4,7});   // one tile per rank
  REQUIRE(local_range_of_rank(7,3,1,0) == std::pair<long,long>{0,7});   // all tiles on one rank

  // r = 0: every tile the same size, no ragged tail
  for (long i = 0; i < 4; ++i) REQUIRE(tile_extent(8,4,i) == 2);

  // max_tile_size edges: exact fit, one element over, and a size so small that the
  // count would exceed N and is clamped to it
  REQUIRE(balanced_tile_count(1024, 1, 1024) == 1);
  REQUIRE(balanced_tile_count(1025, 1, 1024) == 2);
  REQUIRE(balanced_tile_count(10, 4, 1) == 10);    // 4*ceil(10/4) = 12, clamped to 10
  REQUIRE(balanced_tile_count(10, 1, 1) == 10);
  REQUIRE(balanced_tile_count(10, 10, 1) == 10);
}

TEST_CASE("resolved_tile_counts", "[utilities]")
{
  using utils::resolved_tile_counts;
  const std::array<long,4> shape{36, 512, 1687, 1687};

  // a 0 is filled in with the EXTENT of that axis, not with 1: one element per tile
  // means as many tiles as there are elements. This is scr_coulomb_fourier_t's
  // ft_buffer_dist output, which leaves (w,q) unspecified and fixes 3 tiles on (P,Q).
  REQUIRE(resolved_tile_counts<4>({0, 0, 3, 3}, shape) == std::array<long,4>{36, 512, 3, 3});
  // already-resolved counts are left alone, so the function is idempotent
  REQUIRE(resolved_tile_counts<4>({36, 512, 3, 3}, shape) == std::array<long,4>{36, 512, 3, 3});
  REQUIRE(resolved_tile_counts<4>(resolved_tile_counts<4>({0, 0, 3, 3}, shape), shape)
          == resolved_tile_counts<4>({0, 0, 3, 3}, shape));
  // an axis with a non-positive extent carries no partition and keeps its 0
  REQUIRE(resolved_tile_counts<4>({0, 0, 3, 3}, {0, 512, 1687, 1687})
          == std::array<long,4>{0, 512, 3, 3});
}

} // bdft_tests
