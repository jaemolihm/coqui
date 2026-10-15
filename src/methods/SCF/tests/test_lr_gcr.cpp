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
#include <functional>

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

using precond_t = std::function<void(methods::lr_gcr::Vec1D const&, methods::lr_gcr::Vec1D&)>;

// Solve g(x) = x from y = 0 with `gcr`; returns the number of steps taken and
// leaves the solution in x.
template<typename FArr, typename SArr>
int solve(mpi3::communicator& comm, utils::part_map const& pmap, methods::lr_gcr& gcr,
          affine_problem const& P, FArr& F, SArr& S, nda::array<ComplexType, 1>& x,
          double tol, int max_steps, precond_t const* precond = nullptr) {
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
    gcr.step(comm, pmap, F(), Fp, S(), Sp, tol, conv, precond);
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

// Dense inverse by Gauss-Jordan with partial pivoting.
nda::array<ComplexType, 2> inverse(nda::array<ComplexType, 2> A) {
  const long n = A.extent(0);
  nda::array<ComplexType, 2> X(n, n);
  X() = 0;
  for (long i = 0; i < n; ++i) X(i, i) = 1.0;
  for (long c = 0; c < n; ++c) {
    long piv = c;
    for (long i = c + 1; i < n; ++i)
      if (std::abs(A(i, c)) > std::abs(A(piv, c))) piv = i;
    for (long j = 0; j < n; ++j) { std::swap(A(c, j), A(piv, j)); std::swap(X(c, j), X(piv, j)); }
    const ComplexType d = A(c, c);
    for (long j = 0; j < n; ++j) { A(c, j) /= d; X(c, j) /= d; }
    for (long i = 0; i < n; ++i) {
      if (i == c) continue;
      const ComplexType f = A(i, c);
      for (long j = 0; j < n; ++j) { A(i, j) -= f * A(c, j); X(i, j) -= f * X(c, j); }
    }
  }
  return X;
}

// z = Minv r on this rank's slices of the concatenation [F; S], as lr_gcr hands
// them over: gather the full r, apply the dense Minv, keep this rank's slices.
precond_t dense_precond(mpi3::communicator& comm, utils::part_map const& pmap,
                        long nF, long nS, nda::array<ComplexType, 2> const& Minv) {
  return [&comm, &pmap, nF, nS, Minv](methods::lr_gcr::Vec1D const& r,
                                      methods::lr_gcr::Vec1D& z) {
    auto [f0, f1] = pmap.my_slice(nF);
    long s0 = 0, s1 = 0;
    if (nS > 0) std::tie(s0, s1) = pmap.my_slice(nS);
    const long mF = f1 - f0;
    nda::array<ComplexType, 1> rf(nF + nS);
    rf() = 0;
    rf(nda::range(f0, f1)) = r(nda::range(0, mF));
    if (nS > 0) rf(nda::range(nF + s0, nF + s1)) = r(nda::range(mF, mF + s1 - s0));
    comm.all_reduce_in_place_n(rf.data(), rf.size(), std::plus<>{});
    nda::array<ComplexType, 1> zf(nF + nS);
    zf() = 0;
    for (long i = 0; i < nF + nS; ++i)
      for (long j = 0; j < nF + nS; ++j) zf(i) += Minv(i, j) * rf(j);
    z(nda::range(0, mF)) = zf(nda::range(f0, f1));
    if (nS > 0) z(nda::range(mF, mF + s1 - s0)) = zf(nda::range(nF + s0, nF + s1));
  };
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
  // of the first (as a later apply of the preconditioner's is): the recycled pairs
  // stay valid, so the projection removes most of r0 and fewer steps are needed.
  REQUIRE(gcr.basis_size() > 0);
  gcr.reset_rhs();
  affine_problem P2 = P1;
  for (long i = 0; i < 42; ++i) P2.b(i) = 2.0 * P1.b(i) + 1e-3 * pseudo(i, 0, 1.7);
  nda::array<ComplexType, 1> x2;
  const int n2 = solve(world, pmap, gcr, P2, F, S, x2, tol, 200);
  REQUIRE(affine_residual(P2, x2) < 10 * tol);
  REQUIRE(gcr.rhs_residual_projected() < 1e-2 * gcr.rhs_residual_raw());
  REQUIRE(n2 < n1);
  // ‖M‖ < 1 keeps A positive-real, so no step stagnates.
  REQUIRE(gcr.kicks() == 0);

  // ΔF-only iterate: an empty second slot.
  methods::lr_gcr gcr_f(100);
  affine_problem PF(12, 0, 0.4);
  nda::array<ComplexType, 5> S_empty;
  nda::array<ComplexType, 1> xf;
  const int nf = solve(world, pmap, gcr_f, PF, F, S_empty, xf, tol, 200);
  REQUIRE(nf <= 200);
  REQUIRE(affine_residual(PF, xf) < 10 * tol);
}

// Preconditioned (flexible) GCR on a nonsymmetric A = I − M: the next probe is
// x + Minv r. With Minv = A⁻¹ the first direction is the exact solution; with
// the inexact diagonal Minv the iteration converges to the same x.
TEST_CASE("lr_gcr_preconditioned", "[methods_scf]") {
  auto world = mpi3::environment::get_world_instance();
  utils::part_map pmap;
  pmap.nparts = world.size();
  pmap.part_idx = world.rank();
  const long nF = 12, nS = 30, n = nF + nS;
  const double tol = 1e-11;

  // A strongly varying diagonal, so the diagonal preconditioner does real work.
  affine_problem P(nF, nS, 0.2);
  for (long i = 0; i < n; ++i) P.M(i, i) += 0.8 * std::cos(0.37 * double(i));
  nda::array<ComplexType, 2> A(n, n);
  for (long i = 0; i < n; ++i)
    for (long j = 0; j < n; ++j) A(i, j) = (i == j ? 1.0 : 0.0) - P.M(i, j);

  nda::array<ComplexType, 4> F(shape_t<4>{1, 1, 2, 6});
  nda::array<ComplexType, 5> S(shape_t<5>{5, 1, 1, 2, 3});

  methods::lr_gcr gcr0(100);
  nda::array<ComplexType, 1> x0;
  const int n0 = solve(world, pmap, gcr0, P, F, S, x0, tol, 300);
  REQUIRE(n0 <= 300);
  REQUIRE(affine_residual(P, x0) < 10 * tol);
  const double x0n = std::sqrt(std::real(nda::blas::dotc(x0, x0)));

  // Exact inverse: the probe after the first step is x itself.
  {
    auto pc = dense_precond(world, pmap, nF, nS, inverse(A));
    methods::lr_gcr gcr(100);
    nda::array<ComplexType, 1> x;
    const int ns = solve(world, pmap, gcr, P, F, S, x, tol, 300, &pc);
    REQUIRE(ns == 2);
    REQUIRE(affine_residual(P, x) < 10 * tol);
    nda::array<ComplexType, 1> d = x - x0;
    REQUIRE(std::sqrt(std::real(nda::blas::dotc(d, d))) < 1e-9 * x0n);
  }

  // Inexact: Jacobi, Minv = diag(A)⁻¹.
  {
    nda::array<ComplexType, 2> D(n, n);
    D() = 0;
    for (long i = 0; i < n; ++i) D(i, i) = 1.0 / A(i, i);
    auto pc = dense_precond(world, pmap, nF, nS, D);
    methods::lr_gcr gcr(100);
    nda::array<ComplexType, 1> x;
    const int ns = solve(world, pmap, gcr, P, F, S, x, tol, 300, &pc);
    REQUIRE(ns <= 300);
    REQUIRE(affine_residual(P, x) < 10 * tol);
    nda::array<ComplexType, 1> d = x - x0;
    REQUIRE(std::sqrt(std::real(nda::blas::dotc(d, d))) < 1e-9 * x0n);
    if (world.root())
      app_log(1, "lr_gcr_preconditioned: plain {} steps, Jacobi {} steps", n0, ns);
  }
}

// g(x) = (I − A) x + b for a real A with xᵀ A x = 0 for every real x: each step
// along r has η = 0, so plain GCR never reduces ‖r‖, while the kick reaches A r.
TEST_CASE("lr_gcr_kick_skew", "[methods_scf]") {
  auto world = mpi3::environment::get_world_instance();
  utils::part_map pmap;
  pmap.nparts = world.size();
  pmap.part_idx = world.rank();
  const double tol = 1e-11;
  auto skew_problem = [](long nF, long nS, nda::array<double, 2> const& A,
                         nda::array<ComplexType, 1> const& b) {
    affine_problem P(nF, nS, 0.0);
    const long n = nF + nS;
    for (long i = 0; i < n; ++i)
      for (long j = 0; j < n; ++j) P.M(i, j) = (i == j ? 1.0 : 0.0) - A(i, j);
    P.b = b;
    return P;
  };

  // The 2×2 rotation A = [[0, −1], [1, 0]].
  {
    nda::array<double, 2> A(2, 2);
    A() = 0;
    A(0, 1) = -1.0;
    A(1, 0) = 1.0;
    nda::array<ComplexType, 1> b(2);
    b(0) = 1.0;
    b(1) = 0.0;
    auto P = skew_problem(2, 0, A, b);
    nda::array<ComplexType, 4> F(shape_t<4>{1, 1, 1, 2});
    nda::array<ComplexType, 5> S_empty;
    nda::array<ComplexType, 1> x;
    methods::lr_gcr gcr(10);
    const int n = solve(world, pmap, gcr, P, F, S_empty, x, tol, 50);
    REQUIRE(n <= 3);
    REQUIRE(gcr.kicks() == 1);
    REQUIRE(affine_residual(P, x) < 10 * tol);
  }

  // Two rotation blocks of different scale over [F; S], complex b, capacity 2.
  {
    nda::array<double, 2> A(4, 4);
    A() = 0;
    A(0, 1) = -1.0;
    A(1, 0) = 1.0;
    A(2, 3) = -3.0;
    A(3, 2) = 3.0;
    nda::array<ComplexType, 1> b(4);
    b(0) = ComplexType(0.3, 0.2);
    b(1) = -1.2;
    b(2) = ComplexType(0.0, 0.7);
    b(3) = 0.5;
    auto P = skew_problem(2, 2, A, b);
    nda::array<ComplexType, 4> F(shape_t<4>{1, 1, 1, 2});
    nda::array<ComplexType, 5> S(shape_t<5>{1, 1, 1, 1, 2});
    nda::array<ComplexType, 1> x;
    methods::lr_gcr gcr(2);
    const int n = solve(world, pmap, gcr, P, F, S, x, tol, 50);
    REQUIRE(n <= 6);
    REQUIRE(gcr.kicks() >= 1);
    REQUIRE(affine_residual(P, x) < 10 * tol);
  }
}

} // namespace bdft_tests
