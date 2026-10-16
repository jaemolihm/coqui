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

#include <array>
#include <optional>

#include "catch2/catch.hpp"

#include "configuration.hpp"
#include "utilities/proc_grid_partition.hpp"

namespace bdft_tests
{

template<size_t N>
long grid_size(std::array<long,N> const& g)
{
  long p = 1;
  for(auto n : g) p *= n;
  return p;
}

// whether some grid of np ranks fits the extents (brute force over the first two dimensions)
bool exists_fit_3(long np, std::array<long,3> const& ext)
{
  for(long a=1; a<=ext[0]; ++a)
    for(long b=1; b<=ext[1]; ++b)
      if(np%(a*b) == 0 and np/(a*b) <= ext[2]) return true;
  return false;
}

TEST_CASE("proc_grid_fit", "[utilities]")
{
  using g3 = std::array<long,3>;

  // a fitting preferred grid is returned unchanged, even when it is not the default choice
  REQUIRE( utils::find_proc_grid_fit<3>(8, {4,10,10}, g3{1,2,4}).value() == g3{1,2,4} );
  REQUIRE( utils::find_proc_grid_fit<3>(8, {4,10,10}, g3{2,2,2}).value() == g3{2,2,2} );
  // ... and ignored when it does not fit or has the wrong size
  REQUIRE( utils::find_proc_grid_fit<3>(8, {2,100,100}, g3{8,1,1}).value() == g3{2,2,2} );
  REQUIRE( utils::find_proc_grid_fit<3>(8, {4,10,10}, g3{2,2,1}).value() == g3{4,2,1} );
  // default: maximize n_0, then split the rest evenly between the last two (larger first)
  REQUIRE( utils::find_proc_grid_fit<3>(12, {2,100,100}).value() == g3{2,3,2} );
  REQUIRE( utils::find_proc_grid_fit<2>(6, {100,100}).value() == std::array<long,2>{3,2} );
  REQUIRE( utils::find_proc_grid_fit<1>(5, {7}).value() == std::array<long,1>{5} );
  // no grid fits
  REQUIRE( not utils::find_proc_grid_fit<3>(7, {2,3,5}).has_value() );
  REQUIRE( not utils::find_proc_grid_fit<1>(8, {7}).has_value() );
  REQUIRE( utils::find_proc_grid_fit_or_abort<3>(8, {4,10,10}, g3{2,2,2}, "test") == g3{2,2,2} );

  // every returned grid has np ranks and fits; a grid is found exactly when one exists
  for(long np=1; np<=36; ++np)
    for(long e0 : {1l,2l,3l,5l})
      for(long e1 : {1l,4l,6l})
        for(long e2 : {1l,2l,7l,40l}) {
          g3 ext{e0,e1,e2};
          auto g = utils::find_proc_grid_fit<3>(np, ext);
          REQUIRE( g.has_value() == exists_fit_3(np, ext) );
          if(g) {
            REQUIRE( grid_size(*g) == np );
            REQUIRE( utils::proc_grid_fits(*g, ext) );
          }
        }
}

TEST_CASE("proc_grid_fit_groups_subset", "[utilities]")
{
  using g3 = std::array<long,3>;

  // the whole communicator, with the preferred grid, whenever a grid of np ranks fits
  {
    auto [g, grid] = utils::find_proc_grid_fit_groups<3>(8, {4,10,10}, g3{1,2,4});
    REQUIRE( g == 8 );
    REQUIRE( grid == g3{1,2,4} );
    auto [gs, grids] = utils::find_proc_grid_fit_subset<3>(8, {4,10,10}, g3{1,2,4});
    REQUIRE( gs == 8 );
    REQUIRE( grids == g3{1,2,4} );
  }
  // groups: the largest divisor of np that fits; subset: the largest g <= np that fits
  {
    auto [g, grid] = utils::find_proc_grid_fit_groups<3>(8, {1,1,3});
    REQUIRE( g == 2 );
    REQUIRE( grid == g3{1,1,2} );
    auto [gs, grids] = utils::find_proc_grid_fit_subset<3>(8, {1,1,3});
    REQUIRE( gs == 3 );
    REQUIRE( grids == g3{1,1,3} );
  }
  // nothing to distribute: one rank
  {
    auto [g, grid] = utils::find_proc_grid_fit_groups<3>(6, {1,1,1});
    REQUIRE( g == 1 );
    REQUIRE( grid == g3{1,1,1} );
    auto [gs, grids] = utils::find_proc_grid_fit_subset<3>(6, {1,1,1});
    REQUIRE( gs == 1 );
    REQUIRE( grids == g3{1,1,1} );
  }
  // the pref of a smaller g is not used (it would not have g ranks)
  {
    auto [g, grid] = utils::find_proc_grid_fit_subset<3>(8, {7,1,1}, g3{8,1,1});
    REQUIRE( g == 7 );
    REQUIRE( grid == g3{7,1,1} );
  }
  // properties: g divides np (groups), g <= np (subset), the grid has g ranks and fits
  for(long np=1; np<=24; ++np)
    for(g3 ext : {g3{1,1,1}, g3{1,3,2}, g3{5,1,1}, g3{2,2,7}, g3{4,6,10}}) {
      auto [g, grid] = utils::find_proc_grid_fit_groups<3>(np, ext);
      REQUIRE( np%g == 0 );
      REQUIRE( grid_size(grid) == g );
      REQUIRE( utils::proc_grid_fits(grid, ext) );
      auto [gs, grids] = utils::find_proc_grid_fit_subset<3>(np, ext);
      REQUIRE( gs <= np );
      REQUIRE( gs >= g );
      REQUIRE( grid_size(grids) == gs );
      REQUIRE( utils::proc_grid_fits(grids, ext) );
      if(utils::find_proc_grid_fit<3>(np, ext).has_value()) {
        REQUIRE( g == np );
        REQUIRE( gs == np );
      }
    }
}

} // bdft_tests
