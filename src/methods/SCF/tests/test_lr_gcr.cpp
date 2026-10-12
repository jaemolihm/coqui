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

#include <cmath>

#include "catch2/catch.hpp"

#include "configuration.hpp"
#include "mpi3/environment.hpp"
#include "mpi3/communicator.hpp"

#include "nda/nda.hpp"

#include "utilities/element_partition.hpp"
#include "methods/SCF/lr_krylov.hpp"

/**
 * lr_gcr on a small dense affine map g(x) = M x + b over the concatenation
 * [F; S] of a 4-D and a 5-D array, driven exactly as lr_driver drives it: every
 * rank holds the full arrays, steps its own part_map slice, and the next probe
 * is reassembled from the slices. Checked against the affine residual
 * ‖M x + b − x‖ formed here, independently of the solver.
 */
namespace bdft_tests {

namespace mpi3 = boost::mpi3;
template<int N> using shape_t = std::array<long, N>;

namespace {

ComplexType pseudo(long i, long j, double s) {
  return ComplexType(std::sin(1.3 * double(i) + 0.7 * double(j) + s),
                     std::cos(0.9 * double(i) - 1.1 * double(j) + 2.0 * s));
}

struct affine_problem {
  long nF, nS;
  nda::array<ComplexType, 2> M;
  nda::array<ComplexType, 1> b;

  affine_problem(long nF_, long nS_, double seed) : nF(nF_), nS(nS_), M(nF_ + nS_, nF_ + nS_), b(nF_ + nS_) {
    const long n = nF + nS;
    for (long i = 0; i < n; ++i) {
      b(i) = pseudo(i, 0, seed);
      for (long j = 0; j < n; ++j) M(i, j) = (0.6 / double(n)) * pseudo(i, j, 0.3);
    }
  }
  nda::array<ComplexType, 1> apply(nda::array<ComplexType, 1> const& x) const {
    nda::array<ComplexType, 1> g = b;
    for (long i = 0; i < nF + nS; ++i)
      for (long j = 0; j < nF + nS; ++j) g(i) += M(i, j) * x(j);
    return g;
  }
};

// Solve g(x) = x from y = 0 with `gcr`; returns the number of steps taken and
// leaves the solution in x.
template<typename FArr, typename SArr>
int solve(mpi3::communicator& comm, utils::part_map const& pmap, methods::lr_gcr& gcr,
          affine_problem const& P, FArr& F, SArr& S, nda::array<ComplexType, 1>& x,
          double tol, int max_steps) {
  const long nF = P.nF, nS = P.nS;
  auto [f0, f1] = pmap.my_slice(nF);
  long s0 = 0, s1 = 0;
  if (nS > 0) std::tie(s0, s1) = pmap.my_slice(nS);
  nda::array<ComplexType, 1> y(nF + nS);
  y() = 0;
  for (int it = 1; it <= max_steps; ++it) {
    auto g = P.apply(y);
    auto Ff = nda::reshape(F, shape_t<1>{nF});
    Ff = g(nda::range(0, nF));
    nda::array<ComplexType, 1> Fp = y(nda::range(f0, f1));
    nda::array<ComplexType, 1> Sp;
    if (nS > 0) {
      auto Sf = nda::reshape(S, shape_t<1>{nS});
      Sf = g(nda::range(nF, nF + nS));
      Sp = y(nda::range(nF + s0, nF + s1));
    }
    bool conv = false;
    gcr.step(comm, pmap, F(), Fp, S(), Sp, tol, conv);
    // Reassemble the full probe (or x) from every rank's slice.
    nda::array<ComplexType, 1> ynew(nF + nS);
    ynew() = 0;
    ynew(nda::range(f0, f1)) = nda::reshape(F, shape_t<1>{nF})(nda::range(f0, f1));
    if (nS > 0)
      ynew(nda::range(nF + s0, nF + s1)) = nda::reshape(S, shape_t<1>{nS})(nda::range(s0, s1));
    comm.all_reduce_in_place_n(ynew.data(), ynew.size(), std::plus<>{});
    y = ynew;
    if (conv) { x = y; return it; }
  }
  x = y;
  return max_steps + 1;
}

double affine_residual(affine_problem const& P, nda::array<ComplexType, 1> const& x) {
  auto g = P.apply(x);
  g -= x;
  return std::sqrt(std::real(nda::blas::dotc(g, g)));
}

} // namespace

TEST_CASE("lr_gcr_two_slot_recycling", "[methods_scf]") {
  auto world = mpi3::environment::get_world_instance();
  utils::part_map pmap;
  pmap.nparts = world.size();
  pmap.part_idx = world.rank();

  const shape_t<4> Fshape = {1, 1, 2, 6};      // |F| = 12
  const shape_t<5> Sshape = {5, 1, 1, 2, 3};   // |S| = 30
  nda::array<ComplexType, 4> F(Fshape);
  nda::array<ComplexType, 5> S(Sshape);
  const double tol = 1e-11;

  methods::lr_gcr gcr(100);
  affine_problem P1(12, 30, 0.0);
  nda::array<ComplexType, 1> x1;
  const int n1 = solve(world, pmap, gcr, P1, F, S, x1, tol, 200);
  REQUIRE(n1 <= 200);
  REQUIRE(affine_residual(P1, x1) < 10 * tol);

  // A new right-hand side on the same operator, mostly inside the Krylov space
  // of the first (as a later stage's is): the recycled pairs stay valid, so the
  // projection removes most of r0 and fewer steps are needed.
  REQUIRE(gcr.basis_size() > 0);
  gcr.reset_rhs();
  affine_problem P2 = P1;
  for (long i = 0; i < 42; ++i) P2.b(i) = 2.0 * P1.b(i) + 1e-3 * pseudo(i, 0, 1.7);
  nda::array<ComplexType, 1> x2;
  const int n2 = solve(world, pmap, gcr, P2, F, S, x2, tol, 200);
  REQUIRE(affine_residual(P2, x2) < 10 * tol);
  REQUIRE(gcr.rhs_residual_projected() < 1e-2 * gcr.rhs_residual_raw());
  REQUIRE(n2 < n1);

  // ΔF-only iterate: an empty second slot.
  methods::lr_gcr gcr_f(100);
  affine_problem PF(12, 0, 0.4);
  nda::array<ComplexType, 5> S_empty;
  nda::array<ComplexType, 1> xf;
  const int nf = solve(world, pmap, gcr_f, PF, F, S_empty, xf, tol, 200);
  REQUIRE(nf <= 200);
  REQUIRE(affine_residual(PF, xf) < 10 * tol);
}

} // namespace bdft_tests
