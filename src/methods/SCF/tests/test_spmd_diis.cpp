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

#include "catch2/catch.hpp"

#include "configuration.hpp"
#include "mpi3/environment.hpp"
#include "mpi3/communicator.hpp"
#include "mpi3/shared_communicator.hpp"

#include "nda/nda.hpp"
#include "nda/h5.hpp"
#include "h5/h5.hpp"

#include "IO/app_loggers.h"
#include "utilities/mpi_context.h"
#include "utilities/element_partition.hpp"
#include "utilities/h5_flat_slice.hpp"
#include "numerics/shared_array/nda.hpp"
#include "numerics/iter_scf/diis/spmd_fock_sigma_diis.hpp"

/**
 * The SPMD Fock-Sigma DIIS engine slices the flattened (F, Sigma) pair over the
 * *global* comm, so a node holds valid data only for its own contiguous element
 * run and the node-shared windows have to be completed by an allgatherv over
 * the node roots. Neither half of that is exercised by any end-to-end test (the
 * scf tests build diis_t with storage = "disk"), and neither can be exercised at
 * all on one node.
 *
 * A test machine is one node, so carve it into synthetic "nodes" exactly as
 * gather_to_shm_synthetic_nodes does (shared memory still works within each
 * subset). The ragged splits are the interesting ones: the part_map node-offset
 * scan is what handles nodes of different sizes, and a wrong scan shows up as a
 * node writing (or reading back) the wrong element run.
 */
namespace bdft_tests {

namespace mpi3 = boost::mpi3;
using Array_view_4D_t = nda::array_view<ComplexType, 4>;
using Array_view_5D_t = nda::array_view<ComplexType, 5>;
template<int N> using shape_t = std::array<long, N>;

namespace {

// Deterministic, element-unique values, so a misplaced element cannot cancel.
ComplexType val(long flat, long iter, double phase) {
  return ComplexType(1.0 + 0.5 * double(flat) + 3.0 * double(iter),
                     phase - 0.25 * double(flat));
}

template<typename SArr>
void fill_flat(SArr& sA, long iter, double phase) {
  auto loc = sA.local();
  const long n = loc.size();
  auto flat = nda::reshape(loc, shape_t<1>{n});
  sA.win().fence();
  if (sA.node_comm()->root())
    for (long i = 0; i < n; ++i) flat(i) = val(i, iter, phase);
  sA.win().fence();
}

} // namespace

TEST_CASE("spmd_diis_synthetic_nodes", "[methods_scf]") {
  auto world = mpi3::environment::get_world_instance();
  auto phys_node = world.split_shared();
  // The fake-node split needs every rank on one machine, and at least 4 of them
  // for an even and a ragged layout to differ. CTEST_NPROC defaults to 1, so say
  // out loud when nothing was checked -- a silent return reads as a pass.
  if (phys_node.size() != world.size()) {
    WARN("spmd_diis_synthetic_nodes skipped: ranks span "
         << world.size() / phys_node.size() << " machines; needs one");
    return;
  }
  if (world.size() < 4) {
    WARN("spmd_diis_synthetic_nodes skipped: " << world.size()
         << " rank(s), needs >= 4 (build with -DCTEST_NPROC=8)");
    return;
  }

  const shape_t<4> Fshape = {1, 3, 5, 5};      // |F|     = 75
  const shape_t<5> Sshape = {7, 1, 3, 5, 5};   // |Sigma| = 525
  const double mixing = 0.4;

  auto run = [&](int nfake, int split_at, const std::string& tag) {
    const int color = (split_at < 0) ? int((world.rank() * nfake) / world.size())
                                     : ((world.rank() < split_at) ? 0 : 1);
    auto fake_node = phys_node.split(color, world.rank());
    auto fake_internode = world.split(fake_node.rank(), world.rank());
    auto comm_copy = world;
    utils::mpi_context_t<mpi3::communicator> ctx(
        std::move(comm_copy), std::move(fake_node), std::move(fake_internode));

    math::shm::shared_array<Array_view_4D_t> sF(
        std::addressof(ctx.comm), std::addressof(ctx.internode_comm),
        std::addressof(ctx.node_comm), Fshape);
    math::shm::shared_array<Array_view_5D_t> sS(
        std::addressof(ctx.comm), std::addressof(ctx.internode_comm),
        std::addressof(ctx.node_comm), Sshape);
    math::shm::shared_array<Array_view_5D_t> sC(
        std::addressof(ctx.comm), std::addressof(ctx.internode_comm),
        std::addressof(ctx.node_comm), Sshape);

    const long nF = sF.local().size();
    const long nS = sS.local().size();

    iter_scf::spmd_fs_diis diis;
    diis.configure(mixing, /*max_subsp=*/3, /*warmup=*/2);
    fill_flat(sF, 0, 0.0);
    fill_flat(sS, 0, 1.0);
    diis.init_x0(ctx, sF.local(), sS.local(), /*capture_prev=*/true);

    // Previous accepted state, tracked independently for the damping check.
    nda::array<ComplexType, 1> prevF(nF), prevS(nS);
    for (long i = 0; i < nF; ++i) prevF(i) = val(i, 0, 0.0);
    for (long i = 0; i < nS; ++i) prevS(i) = val(i, 0, 1.0);

    for (long iter = 1; iter <= 5; ++iter) {
      fill_flat(sF, iter, 0.0);
      fill_flat(sS, iter, 1.0);
      fill_flat(sC, iter, 2.0);
      const bool warmup = (iter <= 2);  // configure(warmup = 2)

      nda::array<ComplexType, 1> newF(nF), newS(nS);
      for (long i = 0; i < nF; ++i) newF(i) = val(i, iter, 0.0);
      for (long i = 0; i < nS; ++i) newS(i) = val(i, iter, 1.0);

      auto C_loc = sC.local();
      const Array_view_5D_t* Cp = diis.needs_residual_next() ? &C_loc : nullptr;

      sF.win().fence();
      sS.win().fence();
      diis.solve(ctx.comm, sF.local(), sS.local(), Cp, nullptr, nullptr, iter);
      sF.win().fence();
      sS.win().fence();
      // Uniform on every rank, unlike internode_comm.size() on a ragged layout.
      if (ctx.comm.size() != ctx.node_comm.size()) {
        if (ctx.node_comm.root()) {
          utils::complete_node_slices(ctx.internode_comm, diis.pmap(),
                                      sF.local().data(), nF);
          utils::complete_node_slices(ctx.internode_comm, diis.pmap(),
                                      sS.local().data(), nS);
        }
        sF.node_sync();
        sS.node_sync();
      }

      // (a) Every rank on every node must see exactly the same completed array.
      auto max_spread = [&](auto& sA, long n) {
        nda::array<ComplexType, 1> ref(n);
        auto flat = nda::reshape(sA.local(), shape_t<1>{n});
        for (long i = 0; i < n; ++i) ref(i) = flat(i);
        ctx.comm.broadcast_n(ref.data(), n, 0);
        double e = 0.0;
        for (long i = 0; i < n; ++i) e = std::max(e, std::abs(flat(i) - ref(i)));
        return ctx.comm.all_reduce_value(e, mpi3::max<>{});
      };
      const double spread_F = max_spread(sF, nF);
      const double spread_S = max_spread(sS, nS);
      app_log(2, "spmd_diis [{}] iter {}: nodes = {}, cross-rank spread F {:.1e} / Sigma {:.1e}",
              tag, iter, ctx.internode_comm.size(), spread_F, spread_S);
      REQUIRE(spread_F == 0.0);
      REQUIRE(spread_S == 0.0);

      // (b) While damping, the arithmetic is elementwise and therefore
      // partition-independent: check it bit-for-bit.
      nda::array<ComplexType, 1> expF(nF), expS(nS);
      for (long i = 0; i < nF; ++i) expF(i) = mixing * newF(i) + (1.0 - mixing) * prevF(i);
      for (long i = 0; i < nS; ++i) expS(i) = mixing * newS(i) + (1.0 - mixing) * prevS(i);
      if (warmup) {
        auto fF = nda::reshape(sF.local(), shape_t<1>{nF});
        auto fS = nda::reshape(sS.local(), shape_t<1>{nS});
        double eF = 0.0, eS = 0.0;
        for (long i = 0; i < nF; ++i) eF = std::max(eF, std::abs(fF(i) - expF(i)));
        for (long i = 0; i < nS; ++i) eS = std::max(eS, std::abs(fS(i) - expS(i)));
        REQUIRE(ctx.comm.all_reduce_value(eF, mpi3::max<>{}) == 0.0);
        REQUIRE(ctx.comm.all_reduce_value(eS, mpi3::max<>{}) == 0.0);
      }
      // The accepted state is whatever the engine wrote back, damped or not.
      {
        auto fF = nda::reshape(sF.local(), shape_t<1>{nF});
        auto fS = nda::reshape(sS.local(), shape_t<1>{nS});
        for (long i = 0; i < nF; ++i) prevF(i) = fF(i);
        for (long i = 0; i < nS; ++i) prevS(i) = fS(i);
      }
    }
  };

  run(2, -1, "even, 2 nodes");
  if (world.size() >= 8) run(4, -1, "even, 4 nodes");
  run(2, world.size() / 2 + 1, "ragged");
  run(2, world.size() - 1, "ragged, 1-rank node");
  run(1, -1, "single node");
}

/**
 * The DIIS restart edge stages the previous accepted (F, Sigma) from the
 * checkpoint. Each rank reads only the flat element range it owns, which is not
 * a rectangular hyperslab -- h5_read_flat_range decomposes it into up to
 * 2*rank-1 of them. This path is only reached on a restart, so nothing else in
 * the suite covers it; the decomposition is where an off-by-one would hide.
 */
TEST_CASE("h5_read_flat_range", "[methods_scf]") {
  auto world = mpi3::environment::get_world_instance();
  const std::string fname = "h5_flat_slice_test.h5";

  auto check = [&](auto shp, const std::string& tag) {
    constexpr int R = std::tuple_size_v<decltype(shp)>;
    nda::array<ComplexType, R> ref(shp);
    long n = ref.size();
    {
      auto flat = nda::reshape(ref, shape_t<1>{n});
      for (long i = 0; i < n; ++i) flat(i) = ComplexType(1.0 + double(i), -0.5 * double(i));
    }
    if (world.root()) {
      h5::file file(fname, 'w');
      h5::group grp(file);
      nda::h5_write(grp, "A", ref, false);
    }
    world.barrier();

    auto flatref = nda::reshape(ref, shape_t<1>{n});
    h5::file file(fname, 'r');
    h5::group grp(file);
    // Whole array, single elements at both ends, ranges straddling every axis
    // boundary, an empty range, and the ceil-div slices of an 8-part partition.
    std::vector<std::pair<long, long>> ranges = {
      {0, n}, {0, 1}, {n - 1, n}, {1, n - 1}, {0, 0}, {n / 3, n / 3},
      {3, 5}, {1, n / 2 + 3}, {n / 2 - 1, n - 2}, {n - 7, n}
    };
    for (long part = 0; part < 8; ++part)
      ranges.push_back(utils::part_map::slice(n, 8, part));

    for (auto [i0, i1] : ranges) {
      nda::array<ComplexType, 1> out(std::max(1l, i1 - i0));
      out() = ComplexType(-7.0, -7.0);  // poison, so a skipped block shows up
      utils::h5_read_flat_range<R>(grp, "A", shp, i0, i1, out.data());
      double err = 0.0;
      for (long i = 0; i < i1 - i0; ++i) err = std::max(err, std::abs(out(i) - flatref(i0 + i)));
      if (err != 0.0)
        app_log(1, "h5_read_flat_range [{}]: range [{}, {}) err = {:.3e}", tag, i0, i1, err);
      REQUIRE(err == 0.0);
    }
    world.barrier();
    if (world.root()) std::remove(fname.c_str());
    world.barrier();
  };

  // Sigma-like (5D) and F-like (4D) shapes, no extent dividing the others.
  check(shape_t<5>{3, 2, 5, 4, 7}, "5D");
  check(shape_t<4>{2, 3, 5, 5}, "4D");
  check(shape_t<1>{13}, "1D");
}

} // bdft_tests
