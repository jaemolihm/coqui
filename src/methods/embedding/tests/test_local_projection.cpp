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

#include "utilities/test_common.hpp"
#include "utilities/mpi_context.h"
#include "nda/nda.hpp"
#include "numerics/shared_array/nda.hpp"
#include "numerics/distributed_array/nda.hpp"
#include "numerics/distributed_array/nda_utils.hpp"
#include "methods/embedding/local_projection.hpp"

namespace local_projection_tests {

  using namespace methods;
  namespace lp = methods::local_projection;
  using Array_view_5D_t = nda::array_view<ComplexType, 5>;

  // Deterministic pseudo-random complex number of a global index, identical on every rank
  ComplexType val(int seed, long i0, long i1 = 0, long i2 = 0, long i3 = 0, long i4 = 0) {
    double x = 0.3711*i0 + 1.7132*i1 + 0.9173*i2 + 2.1013*i3 + 0.5531*i4 + 0.7919*seed;
    return ComplexType(std::sin(x*x + 0.13*seed), std::cos(3.1*x + 0.29*seed));
  }

  // Frobenius norm
  template<typename A>
  double fnorm(const A &a) {
    double s = 0.0;
    nda::for_each(a.shape(), [&](auto... i) { s += std::norm(a(i...)); });
    return std::sqrt(s);
  }

  // Sizes: n = 3 local orbitals, nW = 4 band orbitals in the local window, a window offset of 1
  constexpr long ns = 2, nk = 4, nImps = 1, n = 3, nW = 4, nbnd = 6, nts = 3;

  struct fixture_t {
    nda::array<ComplexType, 5> C{ns, nk, nImps, n, nW};
    std::vector<nda::range> W_rng{nda::range(1, 1 + nW)};
    // k + Q on a 4-point mesh with Q = 1 (a cyclic shift: Q != 0)
    nda::array<long, 1> kpq{1, 2, 3, 0};
    fixture_t() {
      nda::for_each(C.shape(), [&](auto... i) { C(i...) = val(1, i...); });
    }
  };

  TEST_CASE("local_projection_band_adjoint", "[methods_embed]") {
    // sum_C_O_Cdag vs upfold_accumulate: the adjoint identity <S(X), y> = <X, U(y)> / N_k for random X, y at
    // Q != 0, n = 3 (tolerance: round-off). Also checks that two accumulations with alpha = 1 equal one with
    // alpha = 2 (adding to a pre-filled array, exact up to round-off).
    auto& context = utils::make_unit_test_mpi_context();
    fixture_t fx;

    auto sX = math::shm::make_shared_array<Array_view_5D_t>(*context, {nts, ns, nk, nbnd, nbnd});
    auto sU1 = math::shm::make_shared_array<Array_view_5D_t>(*context, {nts, ns, nk, nbnd, nbnd});
    auto sU2 = math::shm::make_shared_array<Array_view_5D_t>(*context, {nts, ns, nk, nbnd, nbnd});
    sX.set_zero(); sU1.set_zero(); sU2.set_zero();
    sX.win().fence();
    if (sX.node_comm()->root()) {
      auto X = sX.local();
      nda::for_each(X.shape(), [&](auto... i) { X(i...) = val(2, i...); });
    }
    sX.win().fence();

    nda::array<ComplexType, 5> y(nts, ns, nImps, n, n);
    nda::for_each(y.shape(), [&](auto... i) { y(i...) = val(3, i...); });

    auto SX = lp::sum_C_O_Cdag(sX.local(), fx.C, fx.W_rng, nullptr, nullptr, &fx.kpq, context->comm);

    lp::upfold_accumulate(sU1, y, fx.C, fx.W_rng, &fx.kpq, ComplexType(1.0));
    lp::upfold_accumulate(sU1, y, fx.C, fx.W_rng, &fx.kpq, ComplexType(1.0));
    lp::upfold_accumulate(sU2, y, fx.C, fx.W_rng, &fx.kpq, ComplexType(2.0));

    ComplexType lhs = 0.0, rhs = 0.0;
    nda::for_each(SX.shape(), [&](auto... i) { lhs += nda::conj(SX(i...)) * y(i...); });
    auto X = sX.local();
    auto U = sU1.local();
    nda::for_each(X.shape(), [&](auto... i) { rhs += nda::conj(X(i...)) * U(i...); });
    // U holds 2 U(y)
    rhs /= (2.0 * nk);
    CHECK(std::abs(lhs - rhs) < 1e-12 * std::abs(lhs));
    CHECK(std::abs(lhs) > 1e-3);

    double dU = 0.0;
    auto U2 = sU2.local();
    nda::for_each(U.shape(), [&](auto... i) { dU += std::norm(U(i...) - U2(i...)); });
    CHECK(std::sqrt(dU) < 1e-12 * fnorm(U));
  }

  TEST_CASE("local_projection_aux_adjoint", "[methods_embed]") {
    // downfold_W_block vs upfold_pi_block: the adjoint identity <D(W), y> = <W, U(y^T)> / N_q at Q != 0, n = 3,
    // NP > 1, where y^T swaps the two indices of the first orbital pair (conj(B_{P,dc}) in D vs B_{P,ab} in U).
    // Also checks that a replicated W and the same W block-distributed over the ranks give the same D(W)
    // (tolerance: round-off), and that accumulate = true on top of accumulate = false doubles U.
    auto& context = utils::make_unit_test_mpi_context();
    auto& comm = context->comm;
    fixture_t fx;
    constexpr long NP = 8, nx = 3, nq = nk;

    nda::array<ComplexType, 5> B(nq, nImps, NP, n, n);
    nda::for_each(B.shape(), [&](auto... i) { B(i...) = val(4, i...); });

    // Global (x, q, P, Q) -> W
    nda::array<ComplexType, 4> W_rep(nx, nq, NP, NP);
    nda::for_each(W_rep.shape(), [&](auto... i) { W_rep(i...) = val(5, i...); });
    auto D_rep = lp::downfold_W_block(nda::array_const_view<ComplexType, 4>(W_rep), {0, 0, 0, 0}, nx, B, fx.kpq, true, comm);

    // The process grid {1, 1, np/g, g} over NP = 8 has at most 16 ranks.
    int np = comm.size();
    long g = (np % 2 == 0) ? 2 : 1;
    std::array<long, 4> pgrid{1, 1, np / g, g};
    std::array<long, 4> bsize{nx, nq, (NP + pgrid[2] - 1) / pgrid[2], (NP + pgrid[3] - 1) / pgrid[3]};
    using local_t = nda::array<ComplexType, 4>;
    auto dW = math::nda::make_distributed_array<local_t>(comm, pgrid, {nx, nq, NP, NP}, bsize);
    auto [x0, q0, P0, Q0] = dW.origin();
    auto Wl = dW.local();
    nda::for_each(Wl.shape(), [&](long ix, long iq, long iP, long iQ) {
      Wl(ix, iq, iP, iQ) = val(5, x0 + ix, q0 + iq, P0 + iP, Q0 + iQ);
    });
    auto D_blk = lp::downfold_W_block(nda::array_const_view<ComplexType, 4>(Wl), {x0, q0, P0, Q0}, nx, B, fx.kpq, false, comm);
    CHECK(fnorm(nda::array<ComplexType, 5>(D_blk - D_rep)) < 1e-12 * fnorm(D_rep));

    // y[x, c, d, a, b], and y^T[x, d, c, a, b]
    nda::array<ComplexType, 5> y(nx, n, n, n, n), yT(nx, n, n, n, n);
    nda::for_each(y.shape(), [&](auto... i) { y(i...) = val(6, i...); });
    nda::for_each(y.shape(), [&](long x, long c, long d, long a, long b) { yT(x, d, c, a, b) = y(x, c, d, a, b); });

    auto dPi = math::nda::make_distributed_array<local_t>(comm, pgrid, {nx, nq, NP, NP}, bsize);
    dPi.local() = ComplexType(7.0, -3.0); // accumulate = false must overwrite this
    comm.barrier();
    lp::upfold_pi_block(yT, B, dPi, &fx.kpq, false);
    lp::upfold_pi_block(yT, B, dPi, &fx.kpq, true);

    ComplexType lhs = 0.0, rhs = 0.0;
    nda::for_each(D_rep.shape(), [&](auto... i) { lhs += nda::conj(D_rep(i...)) * y(i...); });
    auto Pl = dPi.local();
    nda::for_each(Pl.shape(), [&](long ix, long iq, long iP, long iQ) {
      rhs += nda::conj(val(5, x0 + ix, q0 + iq, P0 + iP, Q0 + iQ)) * Pl(ix, iq, iP, iQ);
    });
    comm.all_reduce_in_place_n(&rhs, 1, std::plus<>{});
    // Pl holds 2 U(y^T)
    rhs /= (2.0 * nq);
    CHECK(std::abs(lhs - rhs) < 1e-12 * std::abs(lhs));
    CHECK(std::abs(lhs) > 1e-3);
  }

  TEST_CASE("local_projection_upfold_pi_nullptr", "[methods_embed]") {
    // upfold_pi_block without a q+Q map (the scr_coulomb_t::upfold_pi_local path) vs an explicit loop
    // Pi[t,q,P,Q] = sum_{ab,cd} B[q,P,ab] Pi_loc[t,ab,cd] conj(B[q,Q,cd]) (tolerance: round-off).
    auto& context = utils::make_unit_test_mpi_context();
    auto& comm = context->comm;
    constexpr long NP = 8, nx = 3, nq = nk;

    nda::array<ComplexType, 5> B(nq, nImps, NP, n, n);
    nda::for_each(B.shape(), [&](auto... i) { B(i...) = val(4, i...); });
    nda::array<ComplexType, 5> Pi(nx, n, n, n, n);
    nda::for_each(Pi.shape(), [&](auto... i) { Pi(i...) = val(6, i...); });

    int np = comm.size();
    long g = (np % 2 == 0) ? 2 : 1;
    std::array<long, 4> pgrid{1, 1, np / g, g};
    std::array<long, 4> bsize{nx, nq, (NP + pgrid[2] - 1) / pgrid[2], (NP + pgrid[3] - 1) / pgrid[3]};
    using local_t = nda::array<ComplexType, 4>;
    auto dPi = math::nda::make_distributed_array<local_t>(comm, pgrid, {nx, nq, NP, NP}, bsize);
    dPi.local() = ComplexType(7.0, -3.0); // accumulate = false must overwrite this
    comm.barrier();
    lp::upfold_pi_block(Pi, B, dPi, nullptr, false);

    auto [x0, q0, P0, Q0] = dPi.origin();
    auto Pl = dPi.local();
    double err = 0.0, ref = 0.0;
    nda::for_each(Pl.shape(), [&](long ix, long iq, long iP, long iQ) {
      ComplexType r = 0.0;
      for (long a = 0; a < n; ++a) for (long b = 0; b < n; ++b)
        for (long c = 0; c < n; ++c) for (long d = 0; d < n; ++d)
          r += B(q0 + iq, 0, P0 + iP, a, b) * Pi(x0 + ix, a, b, c, d) * std::conj(B(q0 + iq, 0, Q0 + iQ, c, d));
      err += std::norm(Pl(ix, iq, iP, iQ) - r);
      ref += std::norm(r);
    });
    comm.all_reduce_in_place_n(&err, 1, std::plus<>{});
    comm.all_reduce_in_place_n(&ref, 1, std::plus<>{});
    CHECK(std::sqrt(err) < 1e-12 * std::sqrt(ref));
    CHECK(std::sqrt(ref) > 1e-3);
  }

} // local_projection_tests
