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
#include "numerics/imag_axes_ft/IAFT.hpp"
#include "numerics/iter_scf/iter_scf_utils.hpp"

/**
 * The in-memory mixing slices the flattened (F, Sigma) pair over the *global*
 * comm, so a node holds valid data only for its own contiguous element run and the
 * node-shared windows have to be completed by an allgatherv over the node roots.
 * Neither half of that is exercised by any end-to-end test (the scf tests build
 * diis_t with storage = "disk"), and neither can be exercised at all on one node.
 *
 * A test machine is one node, so carve it into synthetic "nodes" exactly as
 * gather_to_shm_synthetic_nodes does (shared memory still works within each
 * subset). The ragged splits are the interesting ones: the part_map node-offset
 * scan is what handles nodes of different sizes, and a wrong scan shows up as a
 * node writing (or reading back) the wrong element run.
 */
namespace bdft_tests {

namespace mpi3 = boost::mpi3;
using Array_view_1D_t = nda::array_view<ComplexType, 1>;
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

// Calls f(ctx, tag) for the node layouts: even 2 / 4 nodes, ragged, a 1-rank node,
// one node. A test machine is one node, so each is carved out of it as fake nodes;
// returns false (after a WARN) when the ranks span machines or are fewer than 4,
// where nothing can be checked.
template<typename F>
bool for_each_synthetic_layout(std::string const& test, F&& f) {
  auto world = mpi3::environment::get_world_instance();
  auto phys_node = world.split_shared();
  // The fake-node split needs every rank on one machine, and at least 4 of them
  // for an even and a ragged layout to differ. CTEST_NPROC defaults to 1, so say
  // out loud when nothing was checked -- a silent return reads as a pass.
  if (phys_node.size() != world.size()) {
    WARN(test << " skipped: ranks span " << world.size() / phys_node.size()
              << " machines; needs one");
    return false;
  }
  if (world.size() < 4) {
    WARN(test << " skipped: " << world.size() << " rank(s), needs >= 4 (build with -DCTEST_NPROC=8)");
    return false;
  }
  auto run = [&](int nfake, int split_at, const std::string& tag) {
    const int color = (split_at < 0) ? int((world.rank() * nfake) / world.size())
                                     : ((world.rank() < split_at) ? 0 : 1);
    auto fake_node = phys_node.split(color, world.rank());
    auto fake_internode = world.split(fake_node.rank(), world.rank());
    auto comm_copy = world;
    utils::mpi_context_t<mpi3::communicator> ctx(
        std::move(comm_copy), std::move(fake_node), std::move(fake_internode));
    f(ctx, tag);
  };
  run(2, -1, "even, 2 nodes");
  if (world.size() >= 8) run(4, -1, "even, 4 nodes");
  run(2, world.size() / 2 + 1, "ragged");
  run(2, world.size() - 1, "ragged, 1-rank node");
  run(1, -1, "single node");
  return true;
}

template<typename Ctx, typename Shape>
auto make_shared(Ctx& ctx, Shape const& shape) {
  constexpr int R = std::tuple_size_v<Shape>;
  return math::shm::shared_array<nda::array_view<ComplexType, R>>(
      std::addressof(ctx.comm), std::addressof(ctx.internode_comm),
      std::addressof(ctx.node_comm), shape);
}

} // namespace

// iter_scf::in_memory_mixing: the damping steps against out = mixing*new + (1-mixing)*prev
// bit for bit (every layout), the extrapolation steps against agreement of every rank on
// the completed window (zero spread), with a DIIS of 3 pairs and 2 warmup steps.
TEST_CASE("in_memory_mixing_synthetic_nodes", "[methods_scf]") {
  const shape_t<4> Fshape = {1, 3, 5, 5};      // |F|     = 75
  const shape_t<5> Sshape = {7, 1, 3, 5, 5};   // |Sigma| = 525
  const double mixing = 0.4;

  for_each_synthetic_layout("in_memory_mixing_synthetic_nodes", [&](auto& ctx, std::string const& tag) {
    auto sF = make_shared(ctx, Fshape);
    auto sS = make_shared(ctx, Sshape);
    auto sC = make_shared(ctx, Sshape);
    const long nF = sF.local().size();
    const long nS = sS.local().size();

    auto complete_and_check_spread = [&](long iter) {
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
      app_log(2, "in_memory_mixing [{}] iter {}: nodes = {}, cross-rank spread F {:.1e} / Sigma {:.1e}",
              tag, iter, ctx.internode_comm.size(), spread_F, spread_S);
      REQUIRE(spread_F == 0.0);
      REQUIRE(spread_S == 0.0);
    };

    // DIIS (max_subsp 3, warmup 2, min_subsp 2) with the previous accepted state of iteration 0.
    iter_scf::in_memory_mixing mix;
    mix.init(ctx, nF, nS, mixing,
             iter_scf::in_memory_diis::config{mixing, 3, 2, 2, 2});
    fill_flat(sF, 0, 0.0);
    fill_flat(sS, 0, 1.0);
    mix.lay.store(mix.prev, sF.local(), sS.local());
    mix.has_prev = true;

    // Previous accepted state, tracked independently for the damping check.
    nda::array<ComplexType, 1> prevF(nF), prevS(nS);
    for (long i = 0; i < nF; ++i) prevF(i) = val(i, 0, 0.0);
    for (long i = 0; i < nS; ++i) prevS(i) = val(i, 0, 1.0);

    for (long iter = 1; iter <= 5; ++iter) {
      fill_flat(sF, iter, 0.0);
      fill_flat(sS, iter, 1.0);
      fill_flat(sC, iter, 2.0);
      const bool warmup = (iter <= 2);

      auto C_loc = sC.local();
      const Array_view_5D_t* Cp = mix.needs_residual() ? &C_loc : nullptr;

      sF.win().fence();
      sS.win().fence();
      mix.step(ctx.comm, sF.local(), sS.local(), Cp);
      sF.win().fence();
      sS.win().fence();
      iter_scf::self_energy_unflatten(ctx, mix.lay, sF, sS);
      complete_and_check_spread(iter);

      // (b) While damping, the arithmetic is elementwise and therefore
      // partition-independent: check it bit-for-bit.
      auto fF = nda::reshape(sF.local(), shape_t<1>{nF});
      auto fS = nda::reshape(sS.local(), shape_t<1>{nS});
      if (warmup) {
        double eF = 0.0, eS = 0.0;
        for (long i = 0; i < nF; ++i)
          eF = std::max(eF, std::abs(fF(i) - (mixing * val(i, iter, 0.0) + (1.0 - mixing) * prevF(i))));
        for (long i = 0; i < nS; ++i)
          eS = std::max(eS, std::abs(fS(i) - (mixing * val(i, iter, 1.0) + (1.0 - mixing) * prevS(i))));
        REQUIRE(ctx.comm.all_reduce_value(eF, mpi3::max<>{}) == 0.0);
        REQUIRE(ctx.comm.all_reduce_value(eS, mpi3::max<>{}) == 0.0);
      }
      // The accepted state is whatever the engine wrote back, damped or not.
      for (long i = 0; i < nF; ++i) prevF(i) = fF(i);
      for (long i = 0; i < nS; ++i) prevS(i) = fS(i);
    }

    // Damping alone (the Dyson-SCF damping) vs out = mixing*new + (1-mixing)*prev,
    // bit for bit, with prev from memory after iteration 0 was captured.
    iter_scf::in_memory_mixing damp;
    damp.init(ctx, nF, nS, mixing, std::nullopt);
    fill_flat(sF, 0, 0.0);
    fill_flat(sS, 0, 1.0);
    damp.lay.store(damp.prev, sF.local(), sS.local());
    damp.has_prev = true;
    for (long i = 0; i < nF; ++i) prevF(i) = val(i, 0, 0.0);
    for (long i = 0; i < nS; ++i) prevS(i) = val(i, 0, 1.0);
    for (long iter = 1; iter <= 3; ++iter) {
      fill_flat(sF, iter, 0.0);
      fill_flat(sS, iter, 1.0);
      sF.win().fence();
      sS.win().fence();
      damp.step(ctx.comm, sF.local(), sS.local(), static_cast<const Array_view_5D_t*>(nullptr));
      sF.win().fence();
      sS.win().fence();
      iter_scf::self_energy_unflatten(ctx, damp.lay, sF, sS);
      auto fF = nda::reshape(sF.local(), shape_t<1>{nF});
      auto fS = nda::reshape(sS.local(), shape_t<1>{nS});
      double eF = 0.0, eS = 0.0;
      for (long i = 0; i < nF; ++i) {
        prevF(i) = mixing * val(i, iter, 0.0) + (1.0 - mixing) * prevF(i);
        eF = std::max(eF, std::abs(fF(i) - prevF(i)));
      }
      for (long i = 0; i < nS; ++i) {
        prevS(i) = mixing * val(i, iter, 1.0) + (1.0 - mixing) * prevS(i);
        eS = std::max(eS, std::abs(fS(i) - prevS(i)));
      }
      app_log(2, "damping [{}] iter {}: max err F {:.1e} / Sigma {:.1e}", tag, iter, eF, eS);
      REQUIRE(ctx.comm.all_reduce_value(eF, mpi3::max<>{}) == 0.0);
      REQUIRE(ctx.comm.all_reduce_value(eS, mpi3::max<>{}) == 0.0);
    }
  });
}

// iter_scf::self_energy_flatten on the synthetic layouts: the slices tile [0, n) exactly
// once, store() then writing the slices into a poisoned window and completing it
// reproduces the reference on every rank bit for bit, and rF()/rS() address the
// concatenation [F; S] of the slices. Cases: F and Sigma of ordinary sizes, F alone
// (nS = 0, with an empty array standing in for Sigma), and a Sigma of 3 elements over
// >= 4 ranks (empty trailing slices).
TEST_CASE("self_energy_flatten_round_trip", "[methods_scf]") {
  struct sizes_t { shape_t<4> F; shape_t<5> S; bool has_S; std::string name; };
  const std::vector<sizes_t> cases = {
    {{1, 3, 5, 5}, {7, 1, 3, 5, 5}, true, "F 75 / S 525"},
    {{1, 3, 5, 5}, {0, 0, 0, 0, 0}, false, "F only"},
    {{1, 3, 5, 5}, {3, 1, 1, 1, 1}, true, "S of 3 elements"},
  };

  for_each_synthetic_layout("self_energy_flatten_round_trip", [&](auto& ctx, std::string const& tag) {
    for (auto const& c : cases) {
      INFO(tag << ", " << c.name);
      auto sF = make_shared(ctx, c.F);
      nda::array<ComplexType, 4> refF(c.F);
      nda::array<ComplexType, 5> refS(c.has_S ? c.S : shape_t<5>{0, 0, 0, 0, 0});
      const long nF = refF.size(), nS = refS.size();
      for (long i = 0; i < nF; ++i) nda::reshape(refF, shape_t<1>{nF})(i) = val(i, 1, 0.0);
      for (long i = 0; i < nS; ++i) nda::reshape(refS, shape_t<1>{nS})(i) = val(i, 1, 1.0);

      const iter_scf::self_energy_flatten lay(utils::make_part_map(ctx), nF, nS);

      // The slices tile [0, n): gather every part's range, indexed by part.
      const long np = lay.pmap().nparts;
      nda::array<long, 1> ranges(4 * np);
      ranges() = 0;
      ranges(4 * lay.pmap().part_idx + 0) = lay.F_range().first;
      ranges(4 * lay.pmap().part_idx + 1) = lay.F_range().second;
      ranges(4 * lay.pmap().part_idx + 2) = lay.S_range().first;
      ranges(4 * lay.pmap().part_idx + 3) = lay.S_range().second;
      ctx.comm.all_reduce_in_place_n(ranges.data(), 4 * np, std::plus<>{});
      REQUIRE(ranges(0) == 0);
      REQUIRE(ranges(2) == 0);
      REQUIRE(ranges(4 * (np - 1) + 1) == nF);
      REQUIRE(ranges(4 * (np - 1) + 3) == nS);
      for (long p = 0; p + 1 < np; ++p) {
        REQUIRE(ranges(4 * (p + 1) + 0) == ranges(4 * p + 1));
        REQUIRE(ranges(4 * (p + 1) + 2) == ranges(4 * p + 3));
      }
      REQUIRE(lay.mF() == lay.F_range().second - lay.F_range().first);
      REQUIRE(lay.mS() == lay.S_range().second - lay.S_range().first);

      // store(), then write the slices into poisoned windows and complete them.
      iter_scf::flat_self_energy v;
      lay.store(v, refF, refS);
      REQUIRE(v.F.size() == lay.mF());
      REQUIRE(v.S.size() == lay.mS());
      auto check_window = [&](auto& sA, auto const& ref, auto slice_of, auto const& vslice) {
        sA.win().fence();
        if (sA.node_comm()->root()) sA.local() = ComplexType(-7.0, -7.0);
        sA.win().fence();
        slice_of(sA.local()) = vslice;
        iter_scf::self_energy_unflatten(ctx, lay, sA);
        auto got = nda::reshape(sA.local(), shape_t<1>{ref.size()});
        auto exp = nda::reshape(ref, shape_t<1>{ref.size()});
        double e = 0.0;
        for (long i = 0; i < ref.size(); ++i) e = std::max(e, std::abs(got(i) - exp(i)));
        REQUIRE(ctx.comm.all_reduce_value(e, mpi3::max<>{}) == 0.0);
      };
      check_window(sF, refF, [&](auto&& A) { return lay.F(A); }, v.F);
      if (c.has_S) {
        auto sS = make_shared(ctx, c.S);
        check_window(sS, refS, [&](auto&& A) { return lay.S(A); }, v.S);
      }

      // rF() / rS() address the concatenation [F; S] of this rank's slices.
      nda::array<ComplexType, 1> cat(lay.mF() + lay.mS());
      cat(lay.rF()) = v.F;
      cat(lay.rS()) = v.S;
      REQUIRE(nda::sum(nda::abs(cat(lay.rF()) - v.F)) == 0.0);
      REQUIRE(nda::sum(nda::abs(cat(lay.rS()) - v.S)) == 0.0);
      REQUIRE(lay.rF().size() == lay.mF());
      REQUIRE(lay.rS().size() == lay.mS());
    }
  });
}

namespace {

double l2(nda::array<ComplexType, 1> const& d) { return std::sqrt(std::real(nda::blas::dotc(d, d))); }

// g(x) = M x + b on x = (F, S), F of 4 and S of 8 elements, M a complex contraction.
struct affine_map {
  static constexpr long nF = 4, nS = 8;
  nda::array<ComplexType, 2> M;
  nda::array<ComplexType, 1> b;
  affine_map() : M(nF + nS, nF + nS), b(nF + nS) {
    const long n = nF + nS;
    for (long i = 0; i < n; ++i) {
      b(i) = val(i, 0, 0.3);
      for (long j = 0; j < n; ++j)
        M(i, j) = (0.6 / double(n)) * ComplexType(std::sin(1.3 * i + 0.7 * j), std::cos(0.9 * i - 1.1 * j));
    }
  }
  nda::array<ComplexType, 1> apply(nda::array<ComplexType, 1> const& x) const {
    nda::array<ComplexType, 1> g = b;
    for (long i = 0; i < nF + nS; ++i)
      for (long j = 0; j < nF + nS; ++j) g(i) += M(i, j) * x(j);
    return g;
  }
  // The fixed point, from (I - M) x = b by Gaussian elimination with partial pivoting.
  nda::array<ComplexType, 1> solution() const {
    const long n = nF + nS;
    nda::array<ComplexType, 2> A(n, n);
    nda::array<ComplexType, 1> r = b;
    for (long i = 0; i < n; ++i)
      for (long j = 0; j < n; ++j) A(i, j) = (i == j ? 1.0 : 0.0) - M(i, j);
    for (long c = 0; c < n; ++c) {
      long piv = c;
      for (long i = c + 1; i < n; ++i)
        if (std::abs(A(i, c)) > std::abs(A(piv, c))) piv = i;
      for (long j = 0; j < n; ++j) std::swap(A(c, j), A(piv, j));
      std::swap(r(c), r(piv));
      for (long i = c + 1; i < n; ++i) {
        const ComplexType f = A(i, c) / A(c, c);
        for (long j = c; j < n; ++j) A(i, j) -= f * A(c, j);
        r(i) -= f * r(c);
      }
    }
    nda::array<ComplexType, 1> x(n);
    for (long i = n - 1; i >= 0; --i) {
      ComplexType t = r(i);
      for (long j = i + 1; j < n; ++j) t -= A(i, j) * x(j);
      x(i) = t / A(i, i);
    }
    return x;
  }
};

// Runs `nsteps` DIIS steps on g from x = 0 with `diis`, on the full arrays F, S (views the
// slicing cuts): load(g_F, g_S) sets them to g(x) on every rank, complete() completes them
// after each step; returns the final x. gs_residual: the residual has an empty F block and (g(x) - x) on S (the Dyson-SCF
// commutator layout), else g(x) - x on both blocks (the LR layout).
template<typename Comm, typename FV, typename SV, typename Load, typename Complete>
nda::array<ComplexType, 1> run_affine(Comm& comm, iter_scf::self_energy_flatten const& lay,
                                      iter_scf::in_memory_diis& diis, affine_map const& P,
                                      FV F, SV S, Load&& load, Complete&& complete,
                                      bool gs_residual, int nsteps) {
  const long n = P.nF + P.nS;
  nda::array<ComplexType, 1> x(n);
  x() = 0;
  iter_scf::flat_self_energy prev;
  for (int it = 0; it < nsteps; ++it) {
    auto g = P.apply(x);
    load(g(nda::range(0, P.nF)), g(nda::range(P.nF, n)));
    nda::array<ComplexType, 4> xF(shape_t<4>{1, 1, 2, 2});
    nda::array<ComplexType, 5> xS(shape_t<5>{2, 1, 2, 2, 1});
    nda::reshape(xF, shape_t<1>{P.nF}) = x(nda::range(0, P.nF));
    nda::reshape(xS, shape_t<1>{P.nS}) = x(nda::range(P.nF, n));
    lay.store(prev, xF, xS);
    if (gs_residual)
      diis.push(comm, lay, F, S, nda::array<ComplexType, 1>{}, lay.S(S) - prev.S);
    else
      diis.push(comm, lay, F, S, lay.F(F) - prev.F, lay.S(S) - prev.S);
    diis.mix(lay, F, S, prev);
    complete();
    for (long i = 0; i < P.nF; ++i) x(i) = nda::reshape(F, shape_t<1>{P.nF})(i);
    for (long i = 0; i < P.nS; ++i) x(P.nF + i) = nda::reshape(S, shape_t<1>{P.nS})(i);
  }
  return x;
}

} // namespace

// in_memory_diis on the affine map g(x) = M x + b, with the layout-independent solve as
// reference: (a) the LR residual g(x) - x on both blocks converges to the solution of
// (I - M) x = b in <= 20 steps on every synthetic layout; (b) the Dyson-SCF residual (empty F
// block) on every layout agrees with a single-rank run of the same engine, so the
// all_reduce of B and the completion are layout independent; reset() followed by a rerun
// reproduces a fresh engine bit for bit.
TEST_CASE("in_memory_diis_affine_fixed_point", "[methods_scf]") {
  const affine_map P;
  const auto x_star = P.solution();
  auto world = mpi3::environment::get_world_instance();

  // Single-rank reference of the Dyson-SCF layout.
  auto self = world.split(world.rank(), 0);
  utils::part_map pmap1;
  pmap1.nparts = 1;
  const iter_scf::self_energy_flatten lay1(pmap1, P.nF, P.nS);
  nda::array<ComplexType, 4> F1(shape_t<4>{1, 1, 2, 2});
  nda::array<ComplexType, 5> S1(shape_t<5>{2, 1, 2, 2, 1});
  iter_scf::in_memory_diis::config gs_cfg{1.0, 20, 0, 2, 2};
  iter_scf::in_memory_diis::config lr_cfg{1.0, 20, 0, 3, 2};
  iter_scf::in_memory_diis ref_diis(gs_cfg);
  auto load1 = [&](auto const& gF, auto const& gS) {
    nda::reshape(F1, shape_t<1>{P.nF}) = gF;
    nda::reshape(S1, shape_t<1>{P.nS}) = gS;
  };
  const auto x_ref = run_affine(self, lay1, ref_diis, P, F1(), S1(), load1, [] {}, true, 20);

  // reset() then a rerun equals a fresh engine, bit for bit.
  {
    iter_scf::in_memory_diis diis(lr_cfg);
    auto x1 = run_affine(self, lay1, diis, P, F1(), S1(), load1, [] {}, false, 20);
    REQUIRE(diis.size() == 20);
    diis.reset();
    REQUIRE(diis.size() == 0);
    REQUIRE(diis.steps() == 0);
    auto x2 = run_affine(self, lay1, diis, P, F1(), S1(), load1, [] {}, false, 20);
    REQUIRE(nda::sum(nda::abs(x1 - x2)) == 0.0);
    REQUIRE(l2(x1 - x_star) < 1e-10);
  }

  for_each_synthetic_layout("in_memory_diis_affine_fixed_point", [&](auto& ctx, std::string const& tag) {
    INFO(tag);
    auto sF = make_shared(ctx, shape_t<4>{1, 1, 2, 2});
    auto sS = make_shared(ctx, shape_t<5>{2, 1, 2, 2, 1});
    const iter_scf::self_energy_flatten lay(utils::make_part_map(ctx), P.nF, P.nS);
    auto complete = [&] { iter_scf::self_energy_unflatten(ctx, lay, sF, sS); };
    // The node root loads g(x); the fences order the load before every rank's in-place mixing.
    auto load = [&](auto const& gF, auto const& gS) {
      sF.win().fence();
      sS.win().fence();
      if (sF.node_comm()->root()) {
        nda::reshape(sF.local(), shape_t<1>{P.nF}) = gF;
        nda::reshape(sS.local(), shape_t<1>{P.nS}) = gS;
      }
      sF.win().fence();
      sS.win().fence();
    };

    for (bool gs : {false, true}) {
      iter_scf::in_memory_diis diis(gs ? gs_cfg : lr_cfg);
      auto x = run_affine(ctx.comm, lay, diis, P, sF.local(), sS.local(), load, complete, gs, 20);
      const double err_star = l2(x - x_star);
      const double err_ref = l2(x - x_ref);
      app_log(2, "in_memory_diis [{}] {}: |x - x*| = {:.2e}, |x - x_single_rank| = {:.2e}",
              tag, gs ? "Dyson-SCF residual" : "LR residual", err_star, err_ref);
      if (not gs) REQUIRE(err_star < 1e-10);
      else REQUIRE(err_ref < 1e-13);
    }
  });
}

// compute_diis_coefs_c1 returns no coefficients when they do not exist, and in_memory_diis then
// damps; a subspace that can never reach min_subsp never stores a pair and always damps.
TEST_CASE("in_memory_diis_fallback_to_damping", "[methods_scf]") {
  auto world = mpi3::environment::get_world_instance();
  auto self = world.split(world.rank(), 0);

  // Coefficients: orthonormal residuals give 1/2 each; B = 0 (vanishing residuals) and a
  // non-finite B give none.
  nda::matrix<ComplexType> B(2, 2);
  B() = 0;
  B(0, 0) = 1.0;
  B(1, 1) = 1.0;
  auto C = iter_scf::compute_diis_coefs_c1(B);
  REQUIRE(C.has_value());
  REQUIRE(std::abs((*C)(0) - 0.5) < 1e-14);
  REQUIRE(std::abs((*C)(1) - 0.5) < 1e-14);
  B() = 0;
  REQUIRE(not iter_scf::compute_diis_coefs_c1(B).has_value());
  B(0, 0) = B(1, 1) = std::nan("");
  REQUIRE(not iter_scf::compute_diis_coefs_c1(B).has_value());

  utils::part_map pmap1;
  pmap1.nparts = 1;
  const iter_scf::self_energy_flatten lay(pmap1, 5, 7);
  nda::array<ComplexType, 1> F(5), S(7), Fn(5), Sn(7);
  for (long i = 0; i < 5; ++i) { F(i) = val(i, 0, 0.0); Fn(i) = val(i, 1, 0.0); }
  for (long i = 0; i < 7; ++i) { S(i) = val(i, 0, 1.0); Sn(i) = val(i, 1, 1.0); }
  iter_scf::flat_self_energy prev;
  lay.store(prev, F, S);
  const double m = 0.3;
  auto check_damped = [&](nda::array<ComplexType, 1>& Fx, nda::array<ComplexType, 1>& Sx) {
    double e = 0.0;
    for (long i = 0; i < 5; ++i) e = std::max(e, std::abs(Fx(i) - (m * Fn(i) + (1.0 - m) * F(i))));
    for (long i = 0; i < 7; ++i) e = std::max(e, std::abs(Sx(i) - (m * Sn(i) + (1.0 - m) * S(i))));
    REQUIRE(e == 0.0);
  };

  // Zero residuals: B = 0, so the extrapolation is skipped and the step damps.
  {
    iter_scf::in_memory_diis diis({m, 5, 0, 2, 2});
    nda::array<ComplexType, 1> zeroF(5), zeroS(7);
    zeroF() = 0;
    zeroS() = 0;
    diis.push(self, lay, Fn, Sn, zeroF, zeroS);
    diis.push(self, lay, Fn, Sn, zeroF, zeroS);
    REQUIRE(diis.size() == 2);
    auto Fx = Fn, Sx = Sn;
    REQUIRE(not diis.mix(lay, Fx, Sx, prev));
    check_damped(Fx, Sx);
  }
  // max_subsp 1 < min_subsp 2: no pair is stored and every step damps.
  {
    iter_scf::in_memory_diis diis({m, 1, 0, 2, 2});
    REQUIRE(not diis.active());
    nda::array<ComplexType, 1> rF = Fn - F, rS = Sn - S;
    diis.push(self, lay, Fn, Sn, rF, rS);
    diis.push(self, lay, Fn, Sn, rF, rS);
    REQUIRE(diis.size() == 0);
    auto Fx = Fn, Sx = Sn;
    REQUIRE(not diis.mix(lay, Fx, Sx, prev));
    check_damped(Fx, Sx);
  }

  // The mixing engine of an inactive DIIS (max_subsp 1 < min_subsp 2) never asks for a
  // residual and damps without one.
  for_each_synthetic_layout("in_memory_mixing_inactive_diis", [&](auto& ctx, std::string const& tag) {
    INFO(tag);
    auto sF = make_shared(ctx, shape_t<4>{1, 1, 2, 2});
    auto sS = make_shared(ctx, shape_t<5>{2, 1, 2, 2, 1});
    const long nF = sF.local().size(), nS = sS.local().size();
    iter_scf::in_memory_mixing mix;
    mix.init(ctx, nF, nS, m, iter_scf::in_memory_diis::config{m, 1, 0, 2, 2});
    auto fill = [&](long iter) {
      sF.win().fence();
      sS.win().fence();
      if (sF.node_comm()->root()) {
        for (long i = 0; i < nF; ++i) nda::reshape(sF.local(), shape_t<1>{nF})(i) = val(i, iter, 0.0);
        for (long i = 0; i < nS; ++i) nda::reshape(sS.local(), shape_t<1>{nS})(i) = val(i, iter, 1.0);
      }
      sF.win().fence();
      sS.win().fence();
    };
    fill(0);
    mix.lay.store(mix.prev, sF.local(), sS.local());
    mix.has_prev = true;
    for (long iter = 1; iter <= 3; ++iter) {
      REQUIRE(not mix.needs_residual());
      fill(iter);
      mix.step(ctx.comm, sF.local(), sS.local(), static_cast<const Array_view_5D_t*>(nullptr));
      sF.win().fence();
      sS.win().fence();
      iter_scf::self_energy_unflatten(ctx, mix.lay, sF, sS);
      auto fF = nda::reshape(sF.local(), shape_t<1>{nF});
      // Step 1 is the damped state m*new + (1-m)*prev.
      double e = 0.0;
      if (iter == 1)
        for (long i = 0; i < nF; ++i)
          e = std::max(e, std::abs(fF(i) - (m * val(i, 1, 0.0) + (1.0 - m) * val(i, 0, 0.0))));
      REQUIRE(ctx.comm.all_reduce_value(e, mpi3::max<>{}) == 0.0);
    }
  });
}

// diis_alg (the disk/VSpace DIIS, here with in-memory VSpace<Heff> and the vector-difference
// residual) skips the extrapolation and returns 0 when the residual overlaps are singular, so
// diis_t damps; independent residuals extrapolate.
TEST_CASE("diis_alg_singular_residuals_skip", "[methods_scf]") {
  using nda::array;
  auto make_H = [](double a) { array<ComplexType, 4> H(1, 1, 2, 2); H() = a; return H; };
  auto run = [&](std::vector<array<ComplexType, 4>> const& seq) {
    iter_scf::VSpace<iter_scf::Heff> x_vsp, res_vsp;
    x_vsp.initialize("diis_alg_test_x.h5", true);
    res_vsp.initialize("diis_alg_test_r.h5", true);
    iter_scf::opt_state<iter_scf::Heff> state;
    state.initialize(iter_scf::Heff(seq[0]));
    iter_scf::diis_residual<iter_scf::Heff> residual(&x_vsp, &state);
    iter_scf::diis_alg<iter_scf::Heff> alg;
    alg.init(&state, &residual, &x_vsp, &res_vsp, 4, true, iter_scf::Heff(seq[0]));
    alg.grow_xvsp_only = true;
    REQUIRE(alg.next_step(iter_scf::Heff(seq[1])) == 0);
    alg.grow_xvsp_only = false;
    int last = 0;
    for (size_t i = 2; i < seq.size(); ++i) last = alg.next_step(iter_scf::Heff(seq[i]));
    return last;
  };

  // Identical vectors: every residual is zero, B = 0.
  REQUIRE(run({make_H(1.0), make_H(1.0), make_H(1.0), make_H(1.0), make_H(1.0)}) == 0);
  // Residuals e2 = (1,0,0,0) and e3 = (0,1,0,0) are orthogonal: B = 1, extrapolation happens.
  array<ComplexType, 4> v0 = make_H(0.0), v1 = make_H(0.0), v2 = make_H(0.0), v3 = make_H(0.0);
  v2(0, 0, 0, 0) = 1.0;
  v3 = v2;
  v3(0, 0, 0, 1) = 1.0;
  REQUIRE(run({v0, v1, v2, v3}) == 1);
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
