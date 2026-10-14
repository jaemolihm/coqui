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

#include <algorithm>
#include <array>
#include <numeric>
#include <random>

#include "catch2/catch.hpp"

#include "configuration.hpp"
#include "mpi3/environment.hpp"
#include "utilities/kpoint_utils.hpp"

namespace {

// Gamma-centred n0 x n1 x n2 grid in crystal coordinates, i-major, optionally folded to [-0.5, 0.5) and shifted.
nda::array<double, 2> make_grid(std::array<long, 3> n, bool fold = false, std::array<double, 3> shift = {0, 0, 0}) {
  nda::array<double, 2> k(n[0] * n[1] * n[2], 3);
  long p = 0;
  for (long a = 0; a < n[0]; ++a)
    for (long b = 0; b < n[1]; ++b)
      for (long c = 0; c < n[2]; ++c, ++p) {
        std::array<long, 3> i = {a, b, c};
        for (int d = 0; d < 3; ++d) {
          double x = (double(i[d]) + shift[d]) / double(n[d]);
          if (fold and x >= 0.5) x -= 1.0;
          k(p, d) = x;
        }
      }
  return k;
}

nda::array<double, 2> permute(nda::array<double, 2> const& k, unsigned seed) {
  std::vector<long> order(k.shape()[0]);
  std::iota(order.begin(), order.end(), 0);
  std::shuffle(order.begin(), order.end(), std::mt19937(seed));
  nda::array<double, 2> out(k.shape());
  for (long p = 0; p < long(order.size()); ++p) out(p, nda::range::all) = k(order[p], nda::range::all);
  return out;
}

// fast (if it applies) and brute-force tables, compared exactly; returns whether the fast path applied
bool compare(std::array<long, 3> N, nda::array<double, 2> const& kpts, nda::array<double, 2> const& Qpts) {
  auto world = boost::mpi3::environment::get_world_instance();
  long nk = kpts.shape()[0];
  nda::array<int, 2> qk_fast(nk, nk), qk_ref(nk, nk), qk_main(nk, nk);
  nda::array<int, 1> qm_fast(nk), qm_ref(nk), qm_main(nk);
  nda::array<double, 2> bg = nda::eye<double>(3);
  nda::array<long, 1> grid = {N[0], N[1], N[2]};
  utils::calculate_Qpt_maps_bruteforce(world, kpts, Qpts, qk_ref, qm_ref);
  utils::calculate_Qpt_maps(world, bg, grid, kpts, Qpts, qk_main, qm_main);
  REQUIRE(qk_main == qk_ref);
  REQUIRE(qm_main == qm_ref);
  bool fast = utils::calculate_Qpt_maps_regular_grid(grid, kpts, Qpts, qk_fast, qm_fast);
  if (fast) {
    REQUIRE(qk_fast == qk_ref);
    REQUIRE(qm_fast == qm_ref);
  }
  return fast;
}

} // namespace

TEST_CASE("Qpt_maps_regular_grid", "[utilities][kpoint]") {
  SECTION("Gamma-centred odd, even, non-cubic") {
    for (auto n : std::vector<std::array<long, 3>>{{3, 3, 3}, {4, 4, 4}, {4, 3, 2}, {1, 1, 5}, {2, 1, 1}, {1, 1, 1}}) {
      auto k = make_grid(n);
      REQUIRE(compare(n, k, k));
      auto kf = make_grid(n, true);
      REQUIRE(compare(n, kf, kf));
    }
  }
  SECTION("32x32x1") {
    auto k = make_grid({32, 32, 1});
    REQUIRE(compare({32, 32, 1}, k, k));
  }
  SECTION("2x2x2 with coordinates -0.5 - 6e-16") {
    const double h = -0.5000000000000006;
    nda::array<double, 2> k = {{0, 0, 0}, {0, 0, h}, {0, h, 0}, {0, h, h},
                               {h, 0, 0}, {h, 0, h}, {h, h, 0}, {h, h, h}};
    REQUIRE(compare({2, 2, 2}, k, k));
  }
  SECTION("permuted k and Q orders") {
    auto k = make_grid({4, 3, 2}, true);
    REQUIRE(compare({4, 3, 2}, permute(k, 1), permute(k, 2)));
  }
  SECTION("fallback: grid dimensions do not match the points") {
    auto k = make_grid({4, 2, 2});
    for (auto N : std::vector<std::array<long, 3>>{{2, 4, 2}, {4, 4, 1}, {8, 2, 1}, {4, 2, 1}, {0, 0, 0}, {-4, -2, 2}})
      REQUIRE_FALSE(compare(N, k, k));
  }
  SECTION("fallback: shifted grid, Q Gamma-centred") {
    REQUIRE_FALSE(compare({4, 4, 2}, make_grid({4, 4, 2}, false, {0.5, 0.5, 0.5}), make_grid({4, 4, 2})));
  }
  SECTION("fallback: grid point off by more than the tolerance") {
    auto k = make_grid({3, 3, 1});
    k(4, 0) += 1e-6;   // still within the brute-force match tolerance
    REQUIRE_FALSE(compare({3, 3, 1}, k, make_grid({3, 3, 1})));
  }
  SECTION("fast path rejects incomplete or duplicated lists") {
    nda::array<int, 2> qk(6, 6);
    nda::array<int, 1> qm(6);
    auto k = make_grid({3, 2, 1});
    auto kdup = nda::array<double, 2>(k);
    kdup(5, nda::range::all) = k(4, nda::range::all);
    REQUIRE_FALSE(utils::calculate_Qpt_maps_regular_grid(nda::array<long, 1>{3, 2, 1}, kdup, k, qk, qm));
    auto k4 = make_grid({4, 1, 1});
    nda::array<double, 2> kinc = k4(nda::range(0, 3), nda::range::all);   // 3 of the 4 points
    nda::array<int, 2> qk3(3, 3);
    nda::array<int, 1> qm3(3);
    REQUIRE_FALSE(utils::calculate_Qpt_maps_regular_grid(nda::array<long, 1>{4, 1, 1}, kinc, kinc, qk3, qm3));
    REQUIRE_FALSE(utils::calculate_Qpt_maps_regular_grid(nda::array<long, 1>{3, 1, 1}, kinc, kinc, qk3, qm3));
  }
}
