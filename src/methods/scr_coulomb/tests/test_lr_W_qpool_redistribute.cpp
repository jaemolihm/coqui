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

/**
 * Tests methods/scr_coulomb/lr_W_qpool_redistribute.hpp, the ω-side transfer of the
 * ΔW pipeline: the layout helper lr_W_omega_layout_for and the intra-q-pool
 * redistribute that crosses between the FT-buffer distribution (ω local, q and (P,Q)
 * split) and the ω-side one (ω and q split, (P,Q) split u ways).
 *
 * This is the only automated coverage of the permutation key at u > 1: no production
 * configuration reaches it (the 192-rank benchmark has u == 1), so the {6,2,5} case
 * below is what exercises the general digit swap.
 *
 * The redistribute is a pure permutation of data, so each element is filled with an
 * exact integer encoding of its own global (w, q, P, Q) index. A round trip must
 * then reproduce the encoding element-for-element, which makes the check bitwise
 * rather than a tolerance — a misdirected block shows up as a wrong index, not as
 * a small residual. Covered: even and ragged ω tiles, ragged (P,Q) panels, and
 * m_Q > 1; plus the layout predicate rejecting the cases that must not select it.
 */

#include "catch2/catch.hpp"

#include "mpi3/environment.hpp"
#include "mpi3/communicator.hpp"

#include "configuration.hpp"
#include "IO/AppAbort.hpp"
#include "IO/app_loggers.h"

#include "nda/nda.hpp"
#include "numerics/distributed_array/nda.hpp"
#include "methods/scr_coulomb/lr_W_qpool_redistribute.hpp"

namespace bdft_tests {

using namespace methods::solvers;
template<int Rank> using shape_t = std::array<long, Rank>;

/**
 * Analytic fill: an exact integer encoding of the *global* index tuple. Any
 * misrouted element differs from the expected value by at least one, so the
 * comparisons below can demand exact equality rather than a tolerance.
 */
static ComplexType qpool_ref(long iw, long iq, long iP, long iQ,
                             long nq, long NP, long NQ) {
  double v = double(((iw * nq + iq) * NP + iP) * NQ + iQ);
  return ComplexType(v, -v - 0.5);
}

/**
 * Round trip through the q-pool redistribute for one shape whose q-pools hold more
 * than one rank:
 *   1. fill the FT-buffer-layout array from the global-index function,
 *   2. lr_W_qpool_redistribute_forward and check every ω-side element against the function
 *      evaluated at *its own* global indices (the ω-side origin/local_range come
 *      from make_distributed_array on the permuted communicator, so this validates
 *      the global map, not just self-consistency),
 *   3. lr_W_qpool_redistribute_backward into a fresh buffer-layout array and demand
 *      bit-identity with 1.
 * Returns true if the case has m > 1 and therefore really moved data.
 */
static bool run_roundtrip(boost::mpi3::communicator& world,
                          long nw, long nq, long NP) {
  const long nproc = world.size();
  auto layout = lr_W_omega_layout_for(nproc, nq, nw, NP);
  if (layout.m == 1) return false;   // the two layouts coincide; nothing crosses

  const long m = layout.m, nqpools = layout.nqpools, u = layout.u;
  const long r = world.rank();
  auto comm_perm  = world.split(0, (int)lr_W_qpool_key(r, m, nqpools, u));
  auto qpool_comm = world.split((int)(r / m), (int)(r % m));

  shape_t<4> gshape = {nw, nq, NP, NP};
  auto buf = math::nda::make_distributed_array<nda::array<ComplexType, 4>>(
      world, layout.b_pgrid, gshape, layout.b_bsize);
  auto omg = math::nda::make_distributed_array<nda::array<ComplexType, 4>>(
      comm_perm, layout.w_pgrid, gshape, layout.w_bsize);
  auto buf2 = math::nda::make_distributed_array<nda::array<ComplexType, 4>>(
      world, layout.b_pgrid, gshape, layout.b_bsize);

  auto fill = [&](auto& d, bool poison) {
    auto o = d.origin();
    auto s = d.local_shape();
    auto loc = d.local();
    for (long i0 = 0; i0 < s[0]; ++i0)
      for (long i1 = 0; i1 < s[1]; ++i1)
        for (long i2 = 0; i2 < s[2]; ++i2)
          for (long i3 = 0; i3 < s[3]; ++i3)
            loc(i0, i1, i2, i3) =
                poison ? ComplexType(-7.0, 13.0)
                       : qpool_ref(o[0]+i0, o[1]+i1, o[2]+i2, o[3]+i3, nq, NP, NP);
  };
  auto count_mismatch = [&](auto const& d) {
    auto o = d.origin();
    auto s = d.local_shape();
    auto loc = d.local();
    long bad = 0;
    for (long i0 = 0; i0 < s[0]; ++i0)
      for (long i1 = 0; i1 < s[1]; ++i1)
        for (long i2 = 0; i2 < s[2]; ++i2)
          for (long i3 = 0; i3 < s[3]; ++i3)
            if (loc(i0, i1, i2, i3) !=
                qpool_ref(o[0]+i0, o[1]+i1, o[2]+i2, o[3]+i3, nq, NP, NP)) ++bad;
    return bad;
  };

  fill(buf, false);
  fill(omg, true);
  fill(buf2, true);

  lr_W_qpool_redistribute_forward(world, qpool_comm, buf, omg);
  long bad_fwd = count_mismatch(omg);

  lr_W_qpool_redistribute_backward(world, qpool_comm, omg, buf2);
  long bad_bwd = count_mismatch(buf2);

  long bad[2] = {bad_fwd, bad_bwd};
  world.all_reduce_in_place_n(bad, 2, std::plus<>{});
  INFO("nw=" << nw << " nq=" << nq << " NP=" << NP << " m=" << m
       << " nwpools=" << layout.nwpools << " u=" << u
       << " b_pgrid=(" << layout.b_pgrid[0] << "," << layout.b_pgrid[1] << ","
       << layout.b_pgrid[2] << "," << layout.b_pgrid[3] << ")"
       << " w_pgrid=(" << layout.w_pgrid[0] << "," << layout.w_pgrid[1] << ","
       << layout.w_pgrid[2] << "," << layout.w_pgrid[3] << ")");
  REQUIRE(bad[0] == 0);
  REQUIRE(bad[1] == 0);
  return true;
}

TEST_CASE("lr_W_qpool_redistribute", "[methods][lr]")
{
  auto world = boost::mpi3::environment::get_world_instance();

  // Shapes chosen so that, at the configured CTEST_NPROC (8), m > 1 and data really
  // crosses:
  //   {6,4,5}   : m=2, m_Q=1, ω tiles 3/3, P panels 2/3 (ragged P)
  //   {11,4,5}  : m=2, ragged ω tiles 6/5 on top of ragged P
  //   {12,2,5}  : m=4 with both P and Q split (2x2) on the buffer side, ragged in both
  //   {6,12,5}  : m=2 with 3 q-points per pool (nq_loc > 1)
  //   {12,13,5} : m=4 with *ragged* q pools (7 and 6) — the q axis is a bystander
  //               of the redistribute, so a wrong nq_loc silently misroutes everything
  //   {22,13,5} : ragged ω tiles (6,6,5,5) *and* m_Q > 1 (2x2) at the same time —
  //               the only case where a peer's ω extent and its Q panel both
  //               differ from ours, so every pairwise overlap has a different shape
  //   {6,2,5}   : u > 1 (m=4, nwpools=2, u=2; W=(2,2,2,1)), i.e. the ω side keeps
  //               only 2 of its 4 pool ranks on the ω axis and splits P over the
  //               other 2. The key is then the general digit swap
  //               [0,1,4,5,2,3,6,7] rather than its u == 1 reduction, and a peer
  //               differs from us on the ω axis *and* on P.
  // The layout is pure arithmetic in (nproc, nq, nw, NP), so at other rank counts
  // some of these come out with m == 1 and nothing to cross.
  long fired = 0;
  fired += run_roundtrip(world,  6,  4, 5) ? 1 : 0;
  fired += run_roundtrip(world, 11,  4, 5) ? 1 : 0;
  fired += run_roundtrip(world, 12,  2, 5) ? 1 : 0;
  fired += run_roundtrip(world,  6, 12, 5) ? 1 : 0;
  fired += run_roundtrip(world, 12, 13, 5) ? 1 : 0;
  fired += run_roundtrip(world, 22, 13, 5) ? 1 : 0;
  fired += run_roundtrip(world,  6,  2, 5) ? 1 : 0;

  // A test whose every case degenerates to the alias is vacuous: pin the exact count
  // at the rank count the suite is configured for, and demand at least one round trip
  // at every other multi-rank count. At one rank every layout has m == 1, so there is
  // nothing to exercise — say so rather than reporting a pass.
  if (world.size() == 8) {
    INFO("cases with m > 1: " << fired);
    REQUIRE(fired == 7);
  } else if (world.size() > 1) {
    INFO("cases with m > 1: " << fired);
    REQUIRE(fired > 0);
  } else {
    WARN("lr_W_qpool_redistribute: single rank — every layout has m == 1, so only the "
         "key() bijection is covered. Run with 8 ranks for the round trips.");
  }

  // The degenerate case: at nq == nproc every q-pool holds one rank, so the ω side is
  // the FT buffer itself and its block size is the buffer's (not lr_W_proc_grid's).
  {
    auto lay = lr_W_omega_layout_for(world.size(), world.size(), 6, 5);
    REQUIRE(lay.m == 1);
    REQUIRE(lay.nwpools == 1);
    REQUIRE(lay.u == 1);
    REQUIRE(lay.w_pgrid == lay.b_pgrid);
    REQUIRE(lay.w_bsize == lay.b_bsize);
  }
  // The u > 1 case really is one, and the key is the exact permutation derived for it.
  if (world.size() == 8) {
    auto lay = lr_W_omega_layout_for(8, 2, 6, 5);
    REQUIRE(lay.m == 4);
    REQUIRE(lay.nwpools == 2);
    REQUIRE(lay.u == 2);
    REQUIRE((lay.w_pgrid == shape_t<4>{2, 2, 2, 1}));
    std::vector<long> expect = {0, 1, 4, 5, 2, 3, 6, 7};
    for (long r = 0; r < 8; ++r)
      REQUIRE(lr_W_qpool_key(r, lay.m, lay.nqpools, lay.u) == expect[r]);
  }

  // key() is a bijection with the documented inverse, for every (m, nqpools, u) with
  // u | m, and it reduces to the 3-argument form at u == 1.
  for (long m : {1L, 2L, 3L, 4L, 6L, 8L}) {
    for (long nqp : {1L, 2L, 5L, 7L}) {
      for (long u = 1; u <= m; ++u) {
        if (m % u != 0) continue;
        const long nwp = m / u;
        std::vector<int> seen(m * nqp, 0);
        REQUIRE(lr_W_qpool_key(0, m, nqp, u) == 0);
        for (long r = 0; r < m * nqp; ++r) {
          long s = lr_W_qpool_key(r, m, nqp, u);
          REQUIRE(s >= 0);
          REQUIRE(s < m * nqp);
          REQUIRE(seen[s] == 0);
          seen[s] = 1;
          // key^-1: read the digits (w, qB, c) back off s in radices (nwpools, nqpools, u).
          long w = s / (nqp * u), qB = (s / u) % nqp, c = s % u;
          REQUIRE(w < nwp);
          REQUIRE(qB * m + w * u + c == r);
          if (u == 1) REQUIRE(s == (r % m) * nqp + r / m);
        }
      }
    }
  }
}

} // bdft_tests
