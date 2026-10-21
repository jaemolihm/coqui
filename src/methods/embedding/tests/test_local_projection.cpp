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
#include "methods/embedding/hybridization_response.hpp"
#include "methods/SCF/dyson_omega.hpp"
#include "numerics/imag_axes_ft/IAFT.hpp"

namespace local_projection_tests {

  using namespace methods;
  namespace lp = methods::local_projection;
  using Array_view_5D_t = nda::array_view<ComplexType, 5>;
  using Array_view_4D_t = nda::array_view<ComplexType, 4>;

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

  // Pointwise Dyson fixture of the iω tests: random (H0, S, F) of shape (ns, nk, nbnd, nbnd) and Sigma(tau) with small
  // entries, so that X = (iω + μ) S - H0 - F - Σ(iω) is well conditioned at every Matsubara frequency.
  struct dyson_fixture_t {
    imag_axes_ft::IAFT ft;
    double mu = 0.17;
    math::shm::shared_array<Array_view_4D_t> sH0, sS, sF;
    math::shm::shared_array<Array_view_5D_t> sSigma;
    dyson_fixture_t(utils::mpi_context_t<boost::mpi3::communicator>& context, imag_axes_ft::basis_e basis,
                    bool hermitian = false)
        : ft(100.0, 10.0, basis, "medium"),
          sH0(math::shm::make_shared_array<Array_view_4D_t>(context, {ns, nk, nbnd, nbnd})),
          sS(math::shm::make_shared_array<Array_view_4D_t>(context, {ns, nk, nbnd, nbnd})),
          sF(math::shm::make_shared_array<Array_view_4D_t>(context, {ns, nk, nbnd, nbnd})),
          sSigma(math::shm::make_shared_array<Array_view_5D_t>(context, {ft.nt_f(), ns, nk, nbnd, nbnd})) {
      for (auto* a : {&sH0, &sS, &sF}) { a->set_zero(); a->win().fence(); }
      sSigma.set_zero(); sSigma.win().fence();
      if (sH0.node_comm()->root()) {
        nda::for_each(sH0.local().shape(), [&](long s, long k, long i, long j) {
          sH0.local()(s, k, i, j) = 0.3 * val(11, s, k, i, j);
          sS.local()(s, k, i, j) = (i == j ? 1.0 : 0.0) + 0.05 * val(12, s, k, i, j);
          sF.local()(s, k, i, j) = 0.1 * val(13, s, k, i, j);
        });
        nda::for_each(sSigma.local().shape(), [&](auto... i) { sSigma.local()(i...) = 0.02 * val(14, i...); });
        // Hermitian H0, S, F and Σ(τ): [G_k(iω_n)]† = G_k(-iω_n)
        if (hermitian) {
          for (auto* a : {&sH0, &sS, &sF})
            for (long s = 0; s < ns; ++s) for (long k = 0; k < nk; ++k) for (long i = 0; i < nbnd; ++i) for (long j = i; j < nbnd; ++j) {
              auto x = a->local();
              x(s, k, i, j) = 0.5 * (x(s, k, i, j) + std::conj(x(s, k, j, i)));
              x(s, k, j, i) = std::conj(x(s, k, i, j));
            }
          auto x = sSigma.local();
          for (long t = 0; t < x.shape(0); ++t) for (long s = 0; s < ns; ++s) for (long k = 0; k < nk; ++k)
            for (long i = 0; i < nbnd; ++i) for (long j = i; j < nbnd; ++j) {
              x(t, s, k, i, j) = 0.5 * (x(t, s, k, i, j) + std::conj(x(t, s, k, j, i)));
              x(t, s, k, j, i) = std::conj(x(t, s, k, i, j));
            }
        }
      }
      for (auto* a : {&sH0, &sS, &sF}) a->win().fence();
      sSigma.win().fence();
    }
    // G_k(iω_n) = [(iω_n + μ) S - H0 - F - (BΣ)(iω_n)]⁻¹ by an explicit loop over ω_n
    nda::array<ComplexType, 3> G_ref(long s, long k) const {
      long nw = ft.nw_f();
      nda::array<ComplexType, 3> Sig_t(ft.nt_f(), nbnd, nbnd), Sig_w(nw, nbnd, nbnd), G(nw, nbnd, nbnd);
      Sig_t = sSigma.local()(nda::range::all, s, k, nda::range::all, nda::range::all);
      ft.tau_to_w(Sig_t, Sig_w, imag_axes_ft::fermion);
      for (long n_ = 0; n_ < nw; ++n_) {
        nda::matrix<ComplexType> X(nbnd, nbnd);
        for (long i = 0; i < nbnd; ++i) for (long j = 0; j < nbnd; ++j)
          X(i, j) = ComplexType(0.0, ft.wn_mesh_f()(n_) * M_PI / ft.beta() ) * sS.local()(s, k, i, j) + mu * sS.local()(s, k, i, j)
                    - sH0.local()(s, k, i, j) - sF.local()(s, k, i, j) - Sig_w(n_, i, j);
        nda::matrix<ComplexType> Xi = nda::inverse(X);
        G(n_, nda::ellipsis{}) = Xi;
      }
      return G;
    }
  };

  TEST_CASE("pointwise_G_omega", "[methods_embed]") {
    // dyson_G_omega_block and pointwise_G_omega (the (s, k) blocks dealt over the ranks, combined with all_reduce) vs
    // an explicit loop over ω_n inverting X = (iω_n + μ) S - H0 - F - (BΣ)(iω_n), and the residual X G - 1 of the
    // kernel (tolerance: round-off), for the IR and the DLR grid.
    auto& context = utils::make_unit_test_mpi_context();
    for (auto basis : {imag_axes_ft::ir_basis, imag_axes_ft::dlr_basis}) {
      dyson_fixture_t fx(*context, basis);
      long nw = fx.ft.nw_f();
      auto sG = methods::pointwise_G_omega(*context, fx.mu, fx.sH0, fx.sS, fx.sF, fx.sSigma, fx.ft);
      REQUIRE(sG.local().shape() == std::array<long, 5>{nw, ns, nk, nbnd, nbnd});
      double err = 0.0, ref = 0.0;
      for (long s = 0; s < ns; ++s) for (long k = 0; k < nk; ++k) {
        auto G = fx.G_ref(s, k);
        for (long w = 0; w < nw; ++w) {
          for (long i = 0; i < nbnd; ++i) for (long j = 0; j < nbnd; ++j) {
            err += std::norm(sG.local()(w, s, k, i, j) - G(w, i, j));
            ref += std::norm(G(w, i, j));
          }
        }
      }
      CHECK(std::sqrt(err) < 1e-12 * std::sqrt(ref));
      CHECK(std::sqrt(ref) > 1e-3);

      // kernel residual at one block: X G = 1
      nda::array<ComplexType, 3> Gb(nw, nbnd, nbnd), Sig_t(fx.ft.nt_f(), nbnd, nbnd), Sig_w(nw, nbnd, nbnd);
      methods::dyson_G_omega_block(fx.ft, fx.mu, fx.sH0.local()(1, 2, nda::ellipsis{}), fx.sS.local()(1, 2, nda::ellipsis{}),
                                   fx.sF.local()(1, 2, nda::ellipsis{}),
                                   fx.sSigma.local()(nda::range::all, 1, 2, nda::range::all, nda::range::all), Gb);
      Sig_t = fx.sSigma.local()(nda::range::all, 1, 2, nda::range::all, nda::range::all);
      fx.ft.tau_to_w(Sig_t, Sig_w, imag_axes_ft::fermion);
      double res = 0.0;
      for (long w = 0; w < nw; ++w) {
        nda::matrix<ComplexType> X(nbnd, nbnd), Gm(nbnd, nbnd);
        for (long i = 0; i < nbnd; ++i) for (long j = 0; j < nbnd; ++j) {
          X(i, j) = (ComplexType(0.0, fx.ft.wn_mesh_f()(w) * M_PI / fx.ft.beta()) + fx.mu) * fx.sS.local()(1, 2, i, j)
                    - fx.sH0.local()(1, 2, i, j) - fx.sF.local()(1, 2, i, j) - Sig_w(w, i, j);
          Gm(i, j) = Gb(w, i, j);
        }
        nda::matrix<ComplexType> R = X * Gm;
        for (long i = 0; i < nbnd; ++i) R(i, i) -= 1.0;
        res = std::max(res, fnorm(R));
      }
      CHECK(res < 1e-12);
    }
  }

  TEST_CASE("local_G_omega", "[methods_embed]") {
    // local_G_omega on the IBZ with kp_to_ibz / kp_trev (k = 3 is the time-reversal partner of the IBZ point 1,
    // G_{-k}(iω_n) = conj G_k(iω_{-n}), so its H0, S, F, Σ(τ) are the elementwise conjugates) vs the full-mesh arrays
    // without any map, and vs an explicit loop of C^k G_k C^{k†} with the force_real projection
    // G <- (G + Gᵀ)/2, G(iω_n) <- [G(iω_n) + conj G(iω_{-n})]/2 (tolerance: round-off), for the IR and DLR grid.
    auto& context = utils::make_unit_test_mpi_context();
    auto& comm = context->comm;
    constexpr long nk_ibz = 3;
    nda::array<int, 1> kp_to_ibz{0, 1, 2, 1};
    nda::array<bool, 1> kp_trev{false, false, false, true};
    fixture_t fx;
    for (auto basis : {imag_axes_ft::ir_basis, imag_axes_ft::dlr_basis}) {
      dyson_fixture_t full(*context, basis);   // full mesh; make k = 3 the conjugate of k = 1
      if (full.sH0.node_comm()->root()) {
        for (long s = 0; s < ns; ++s) {
          full.sH0.local()(s, 3, nda::ellipsis{}) = nda::conj(full.sH0.local()(s, 1, nda::ellipsis{}));
          full.sS.local()(s, 3, nda::ellipsis{}) = nda::conj(full.sS.local()(s, 1, nda::ellipsis{}));
          full.sF.local()(s, 3, nda::ellipsis{}) = nda::conj(full.sF.local()(s, 1, nda::ellipsis{}));
          full.sSigma.local()(nda::range::all, s, 3, nda::range::all, nda::range::all) =
              nda::conj(full.sSigma.local()(nda::range::all, s, 1, nda::range::all, nda::range::all));
        }
      }
      for (auto* a : {&full.sH0, &full.sS, &full.sF}) a->win().fence();
      full.sSigma.win().fence();
      // IBZ arrays: the first nk_ibz k points
      auto sH0 = math::shm::make_shared_array<Array_view_4D_t>(*context, {ns, nk_ibz, nbnd, nbnd});
      auto sS = math::shm::make_shared_array<Array_view_4D_t>(*context, {ns, nk_ibz, nbnd, nbnd});
      auto sF = math::shm::make_shared_array<Array_view_4D_t>(*context, {ns, nk_ibz, nbnd, nbnd});
      auto sSigma = math::shm::make_shared_array<Array_view_5D_t>(*context, {full.ft.nt_f(), ns, nk_ibz, nbnd, nbnd});
      sH0.win().fence(); sS.win().fence(); sF.win().fence(); sSigma.win().fence();
      if (sH0.node_comm()->root()) {
        sH0.local() = full.sH0.local()(nda::range::all, nda::range(nk_ibz), nda::range::all, nda::range::all);
        sS.local() = full.sS.local()(nda::range::all, nda::range(nk_ibz), nda::range::all, nda::range::all);
        sF.local() = full.sF.local()(nda::range::all, nda::range(nk_ibz), nda::range::all, nda::range::all);
        sSigma.local() = full.sSigma.local()(nda::range::all, nda::range::all, nda::range(nk_ibz), nda::range::all, nda::range::all);
      }
      sH0.win().fence(); sS.win().fence(); sF.win().fence(); sSigma.win().fence();
      long nw = full.ft.nw_f();

      // explicit loop on the full mesh
      nda::array<ComplexType, 5> ref(nw, ns, nImps, n, n);
      ref() = 0.0;
      for (long s = 0; s < ns; ++s) for (long k = 0; k < nk; ++k) {
        auto G = full.G_ref(s, k);
        for (long w = 0; w < nw; ++w)
          for (long a = 0; a < n; ++a) for (long b = 0; b < n; ++b) {
            ComplexType sum = 0.0;
            for (long i = 0; i < nW; ++i) for (long j = 0; j < nW; ++j)
              sum += fx.C(s, k, 0, a, i) * G(w, 1 + i, 1 + j) * std::conj(fx.C(s, k, 0, b, j));
            ref(w, s, 0, a, b) += sum / double(nk);
          }
      }
      for (bool force_real : {false, true}) {
        nda::array<ComplexType, 5> expect = ref;
        if (force_real) {
          nda::array<ComplexType, 5> sym(ref.shape());
          for (long w = 0; w < nw; ++w) for (long s = 0; s < ns; ++s) for (long a = 0; a < n; ++a) for (long b = 0; b < n; ++b)
            sym(w, s, 0, a, b) = 0.5 * (ref(w, s, 0, a, b) + ref(w, s, 0, b, a));
          for (long w = 0; w < nw; ++w) for (long s = 0; s < ns; ++s) for (long a = 0; a < n; ++a) for (long b = 0; b < n; ++b)
            expect(w, s, 0, a, b) = 0.5 * (sym(w, s, 0, a, b) + std::conj(sym(nw - 1 - w, s, 0, a, b)));
        }
        auto got_ibz = methods::local_G_omega(comm, full.ft, full.mu, sH0, sS, sF, sSigma, fx.C, fx.W_rng,
                                              &kp_to_ibz, &kp_trev, force_real);
        auto got_full = methods::local_G_omega(comm, full.ft, full.mu, full.sH0, full.sS, full.sF, full.sSigma, fx.C,
                                               fx.W_rng, nullptr, nullptr, force_real);
        nda::array<ComplexType, 5> d1 = got_ibz - expect, d2 = got_full - expect;
        CHECK(fnorm(d1) < 1e-12 * fnorm(expect));
        CHECK(fnorm(d2) < 1e-12 * fnorm(expect));
        CHECK(fnorm(expect) > 1e-3);
        if (force_real) {
          // the force_real result is symmetric in the orbitals and satisfies G(iω_n) = conj G(iω_{-n})
          double asym = 0.0;
          for (long w = 0; w < nw; ++w) for (long a = 0; a < n; ++a) for (long b = 0; b < n; ++b) {
            asym += std::norm(got_ibz(w, 0, 0, a, b) - got_ibz(w, 0, 0, b, a));
            asym += std::norm(got_ibz(w, 0, 0, a, b) - std::conj(got_ibz(nw - 1 - w, 0, 0, a, b)));
          }
          CHECK(std::sqrt(asym) < 1e-12 * fnorm(got_ibz));
        }
      }
    }
  }

  TEST_CASE("lr_local_DeltaG_omega", "[methods_embed]") {
    // lr_local_DeltaG_omega vs an explicit loop (1/N_k) Σ_k C^{k+Q} G_{k+Q}[W,:] δX_k G_k[:,W] C^{k†}, with the source
    // δX_k(iω) = ΔH0_k + ΔF_k - Δμ S_k + (BΔΣ_k)(iω) (random, with and without ΔΣ), Q != 0 (cyclic k+Q map) and
    // n = 3 (tolerance: round-off), for the IR and DLR grid.
    auto& context = utils::make_unit_test_mpi_context();
    fixture_t fx;
    for (auto basis : {imag_axes_ft::ir_basis, imag_axes_ft::dlr_basis}) {
      dyson_fixture_t d(*context, basis);
      long nw = d.ft.nw_f();
      auto sG = methods::pointwise_G_omega(*context, d.mu, d.sH0, d.sS, d.sF, d.sSigma, d.ft);
      auto sdH0 = math::shm::make_shared_array<Array_view_4D_t>(*context, {ns, nk, nbnd, nbnd});
      auto sdF = math::shm::make_shared_array<Array_view_4D_t>(*context, {ns, nk, nbnd, nbnd});
      auto sdSigma = math::shm::make_shared_array<Array_view_5D_t>(*context, {d.ft.nt_f(), ns, nk, nbnd, nbnd});
      for (auto* a : {&sdH0, &sdF}) { a->set_zero(); a->win().fence(); }
      sdSigma.set_zero(); sdSigma.win().fence();
      if (sdH0.node_comm()->root()) {
        nda::for_each(sdH0.local().shape(), [&](auto... i) { sdH0.local()(i...) = val(21, i...); sdF.local()(i...) = val(22, i...); });
        nda::for_each(sdSigma.local().shape(), [&](auto... i) { sdSigma.local()(i...) = 0.5 * val(23, i...); });
      }
      for (auto* a : {&sdH0, &sdF}) a->win().fence();
      sdSigma.win().fence();
      const double dmu = 0.37;
      nda::array<long, 1> kpq = fx.kpq;

      for (bool with_sigma : {true, false}) {
        nda::array<ComplexType, 5> ref(nw, ns, 1, n, n);
        ref() = 0.0;
        for (long s = 0; s < ns; ++s) for (long k = 0; k < nk; ++k) {
          long kq = kpq(k);
          nda::array<ComplexType, 3> dS_t(d.ft.nt_f(), nbnd, nbnd), dS_w(nw, nbnd, nbnd);
          dS_t = sdSigma.local()(nda::range::all, s, k, nda::range::all, nda::range::all);
          d.ft.tau_to_w(dS_t, dS_w, imag_axes_ft::fermion);
          for (long w = 0; w < nw; ++w) {
            nda::matrix<ComplexType> X(nbnd, nbnd), Gk(nbnd, nbnd), Gkq(nbnd, nbnd);
            for (long i = 0; i < nbnd; ++i) for (long j = 0; j < nbnd; ++j) {
              X(i, j) = sdH0.local()(s, k, i, j) + sdF.local()(s, k, i, j) - dmu * d.sS.local()(s, k, i, j)
                        + (with_sigma ? dS_w(w, i, j) : ComplexType(0.0));
              Gk(i, j) = sG.local()(w, s, k, i, j);
              Gkq(i, j) = sG.local()(w, s, kq, i, j);
            }
            nda::matrix<ComplexType> M = Gkq * X * Gk;
            for (long a = 0; a < n; ++a) for (long b = 0; b < n; ++b) {
              ComplexType sum = 0.0;
              for (long i = 0; i < nW; ++i) for (long j = 0; j < nW; ++j)
                sum += fx.C(s, kq, 0, a, i) * M(1 + i, 1 + j) * std::conj(fx.C(s, k, 0, b, j));
              ref(w, s, 0, a, b) += sum / double(nk);
            }
          }
        }
        auto got = methods::lr_local_DeltaG_omega(context->comm, d.ft, sG, sdH0, sdF, with_sigma ? &sdSigma : nullptr,
                                                  dmu, d.sS, fx.C, fx.W_rng, kpq);
        nda::array<ComplexType, 5> diff = got - ref;
        CHECK(fnorm(diff) < 1e-12 * fnorm(ref));
        CHECK(fnorm(ref) > 1e-3);
      }
    }
  }

  TEST_CASE("lr_hybridization_w", "[methods_embed]") {
    // lr_hybridization_w vs an explicit per-block loop Ginv δG Ginv - dHloc - dSigma, Ginv the inverse of the α block
    // of G (random, n = 4, orbital 3 in no block, blocks {0, 1} and {2}), zero off-block (tolerance: round-off).
    fixture_t fx;
    constexpr long nw = 7, n4 = 4;
    nda::array<ComplexType, 5> G(nw, ns, 1, n4, n4), dG(nw, ns, 1, n4, n4), dS(nw, ns, 1, n4, n4);
    nda::array<ComplexType, 4> dH(ns, 1, n4, n4);
    nda::for_each(G.shape(), [&](auto... i) { G(i...) = val(31, i...); dG(i...) = val(32, i...); dS(i...) = val(33, i...); });
    nda::for_each(dH.shape(), [&](auto... i) { dH(i...) = val(34, i...); });
    for (long w = 0; w < nw; ++w) for (long s = 0; s < ns; ++s) for (long a = 0; a < n4; ++a) G(w, s, 0, a, a) += 3.0;
    nda::array<long, 1> off{0, 2}, size{2, 1}, blk{0, 0, 1, -1};
    methods::dc_cells_t cells{off, size, blk};
    auto got = methods::lr_hybridization_w(G, dG, dH, dS, cells);

    nda::array<ComplexType, 5> ref(nw, ns, 1, n4, n4);
    ref() = 0.0;
    std::vector<std::vector<long>> blocks{{0, 1}, {2}};
    for (auto const& idx : blocks) {
      long m = idx.size();
      for (long w = 0; w < nw; ++w) for (long s = 0; s < ns; ++s) {
        nda::matrix<ComplexType> Gb(m, m), Xb(m, m);
        for (long a = 0; a < m; ++a) for (long b = 0; b < m; ++b) { Gb(a, b) = G(w, s, 0, idx[a], idx[b]); Xb(a, b) = dG(w, s, 0, idx[a], idx[b]); }
        nda::matrix<ComplexType> Gi = nda::inverse(Gb);
        nda::matrix<ComplexType> R = Gi * Xb * Gi;
        for (long a = 0; a < m; ++a) for (long b = 0; b < m; ++b)
          ref(w, s, 0, idx[a], idx[b]) = R(a, b) - dH(s, 0, idx[a], idx[b]) - dS(w, s, 0, idx[a], idx[b]);
      }
    }
    nda::array<ComplexType, 5> diff = got - ref;
    CHECK(fnorm(diff) < 1e-13 * fnorm(ref));
    CHECK(fnorm(ref) > 1e-3);
    // entries outside a block (including the orbital with orb_block = -1) are exactly zero
    for (long w = 0; w < nw; ++w) for (long s = 0; s < ns; ++s) for (long a = 0; a < n4; ++a) for (long b = 0; b < n4; ++b)
      if (blk(a) < 0 or blk(a) != blk(b)) CHECK(got(w, s, 0, a, b) == ComplexType(0.0));
  }

  TEST_CASE("local_G_omega_from_G", "[methods_embed]") {
    // local_G_omega_from_G(pointwise_G_omega(...)) vs local_G_omega of the same (H0, S, F, Σ) on the full mesh, with and
    // without force_real, IR and DLR grid: the same sum in the same order, so the difference is exactly zero.
    auto& context = utils::make_unit_test_mpi_context();
    fixture_t fx;
    for (auto basis : {imag_axes_ft::ir_basis, imag_axes_ft::dlr_basis}) {
      dyson_fixture_t d(*context, basis);
      auto sG = methods::pointwise_G_omega(*context, d.mu, d.sH0, d.sS, d.sF, d.sSigma, d.ft);
      for (bool force_real : {false, true}) {
        auto ref = methods::local_G_omega(context->comm, d.ft, d.mu, d.sH0, d.sS, d.sF, d.sSigma, fx.C, fx.W_rng,
                                          nullptr, nullptr, force_real);
        auto got = methods::local_G_omega_from_G(context->comm, d.ft, sG, fx.C, fx.W_rng, force_real);
        nda::array<ComplexType, 5> diff = got - ref;
        CHECK(fnorm(diff) == 0.0);
        CHECK(fnorm(ref) > 1e-3);
      }
    }
  }

  TEST_CASE("project_lr_local_response", "[methods_embed]") {
    // P(dG_loc^pw) (= Sym for a Hermitian dH0) of lr_local_DeltaG_omega at Q = 0 for a Hermitian, complex (time-reversal-breaking) ground state and
    // a generic complex Hermitian dH0 (nonzero time-reversal-odd part) vs the 4-point Richardson derivative (h, h/2) of the force_real ground-state map
    // local_G_omega(H0 + h dH0) (tolerance: the O(h^4) truncation at h = 2.5e-4); the unprojected dG_loc^pw is the negative
    // control (differs at O(1)). P(dH_loc) vs (X + Xᵀ + X* + X†)/4 (static P). A
    // non-Hermitian dH0 = A + iB (A, B Hermitian): P(dG[A + iB]) vs the same derivative along A + iB (P is real-linear). The tolerance 1e-8 is the FD error on IR. Q != 0: Sym of
    // random data vs the explicit formula, a symmetric input is unchanged, and dDelta of the projected response vs
    // the formula. IR and DLR grid.
    auto& context = utils::make_unit_test_mpi_context();
    auto& comm = context->comm;
    fixture_t fx;
    for (auto basis : {imag_axes_ft::ir_basis, imag_axes_ft::dlr_basis}) {
      dyson_fixture_t d(*context, basis, true);
      long nw = d.ft.nw_f();
      auto sG = methods::pointwise_G_omega(*context, d.mu, d.sH0, d.sS, d.sF, d.sSigma, d.ft);
      nda::array<long, 1> kpq_id(nk);
      for (long k = 0; k < nk; ++k) kpq_id(k) = k;

      // Hermitian dH0 of the given seed, and the non-Hermitian A + iB
      auto make_dH0 = [&](int seed_a, int seed_b, bool imag_part) {
        auto sdH0 = math::shm::make_shared_array<Array_view_4D_t>(*context, {ns, nk, nbnd, nbnd});
        sdH0.set_zero(); sdH0.win().fence();
        if (sdH0.node_comm()->root()) {
          nda::for_each(sdH0.local().shape(), [&](long s, long k, long i, long j) {
            auto herm = [&](int seed) { return 0.15 * (val(seed, s, k, i, j) + std::conj(val(seed, s, k, j, i))); };
            sdH0.local()(s, k, i, j) = herm(seed_a) + (imag_part ? ComplexType(0.0, 1.0) * herm(seed_b) : ComplexType(0.0));
          });
        }
        sdH0.win().fence();
        return sdH0;
      };
      auto sdF = math::shm::make_shared_array<Array_view_4D_t>(*context, {ns, nk, nbnd, nbnd});
      sdF.set_zero(); sdF.win().fence();
      auto lr_dG = [&](auto& sdH0) {
        return methods::lr_local_DeltaG_omega(comm, d.ft, sG, sdH0, sdF, static_cast<decltype(&d.sSigma)>(nullptr), 0.0, d.sS,
                                              fx.C, fx.W_rng, kpq_id);
      };
      auto static_dH = [&](auto& sdH0) {
        nda::array<ComplexType, 5> dH5(1, ns, 1, n, n);
        dH5() = 0.0;
        nda::array<ComplexType, 2> O(nW, nW), buf(n, nW);
        for (long s = 0; s < ns; ++s) for (long k = 0; k < nk; ++k) {
          O = sdH0.local()(s, k, fx.W_rng[0], fx.W_rng[0]);
          lp::add_C_O_Cdag(fx.C, s, k, k, 0, O, buf, dH5(0, s, 0, nda::ellipsis{}));
        }
        dH5() /= double(nk);
        nda::array<ComplexType, 4> dH(dH5(0, nda::ellipsis{}));
        return dH;
      };
      // Richardson derivative of the force_real ground-state map along dH0
      auto fd_map = [&](auto& sdH0) {
        auto gs_map = [&](double h) {
          if (d.sH0.node_comm()->root())
            nda::for_each(d.sH0.local().shape(), [&](auto... i) { d.sH0.local()(i...) += h * sdH0.local()(i...); });
          d.sH0.win().fence();
          auto g = methods::local_G_omega(comm, d.ft, d.mu, d.sH0, d.sS, d.sF, d.sSigma, fx.C, fx.W_rng, nullptr, nullptr, true);
          if (d.sH0.node_comm()->root())
            nda::for_each(d.sH0.local().shape(), [&](auto... i) { d.sH0.local()(i...) -= h * sdH0.local()(i...); });
          d.sH0.win().fence();
          return g;
        };
        auto deriv = [&](double h) {
          nda::array<ComplexType, 5> gp = gs_map(h), gm = gs_map(-h);
          nda::array<ComplexType, 5> r = (gp - gm) / (2.0 * h);
          return r;
        };
        nda::array<ComplexType, 5> d1 = deriv(2.5e-4), d2 = deriv(1.25e-4);
        nda::array<ComplexType, 5> fd = (4.0 * d2 - d1) / 3.0;
        return fd;
      };

      // Hermitian dH0 at Q = 0
      auto sdA = make_dH0(41, 0, false);
      nda::array<ComplexType, 5> dG = lr_dG(sdA);
      nda::array<ComplexType, 5> dG_proj = dG;
      nda::array<ComplexType, 4> dH = static_dH(sdA);
      nda::array<ComplexType, 4> dH_proj = dH;
      auto change = methods::project_lr_local_response(dG_proj, dH_proj, true);
      CHECK(change[0] > 1e-2);   // the time-reversal-odd part of the response
      CHECK(change[1] > 1e-2);
      nda::array<ComplexType, 4> dH_ref(dH.shape());
      for (long s = 0; s < ns; ++s) for (long a = 0; a < n; ++a) for (long b = 0; b < n; ++b)
        dH_ref(s, 0, a, b) = 0.25 * (dH(s, 0, a, b) + dH(s, 0, b, a) + std::conj(dH(s, 0, a, b)) + std::conj(dH(s, 0, b, a)));
      nda::array<ComplexType, 4> ddH = dH_proj - dH_ref;
      CHECK(fnorm(ddH) < 1e-14 * fnorm(dH));

      nda::array<ComplexType, 5> fdA = fd_map(sdA);
      nda::array<ComplexType, 5> e_proj = dG_proj - fdA, e_raw = dG - fdA;
      double rel_proj = fnorm(e_proj) / fnorm(fdA), rel_raw = fnorm(e_raw) / fnorm(fdA);
      INFO("basis " << int(basis) << ": |P dG - FD|/|FD| = " << rel_proj << ", unprojected " << rel_raw);
      CHECK(rel_proj < 1e-8);
      CHECK(rel_raw > 1e-2);
      CHECK(fnorm(fdA) > 1e-4);

      // Non-Hermitian dH0 = A + iB: P(dG[A + iB]) vs the Richardson derivative of the force_real map along A + iB
      auto sdAB = make_dH0(41, 42, true);
      nda::array<ComplexType, 5> dG_ab = lr_dG(sdAB);
      nda::array<ComplexType, 4> dH_ab = static_dH(sdAB);
      methods::project_lr_local_response(dG_ab, dH_ab, true);
      nda::array<ComplexType, 5> fd_ab = fd_map(sdAB);
      nda::array<ComplexType, 5> e_ab = dG_ab - fd_ab;
      double rel_ab = fnorm(e_ab) / fnorm(fd_ab);
      INFO("basis " << int(basis) << ": |P dG[A+iB] - FD along A+iB|/|FD| = " << rel_ab);
      CHECK(rel_ab < 1e-8);
      CHECK(fnorm(fd_ab) > 1e-4);

      // Q != 0: Sym on random data, nw rows of the iω mesh
      nda::array<ComplexType, 5> X(nw, ns, 1, n, n), Xs(nw, ns, 1, n, n);
      nda::array<ComplexType, 4> H(ns, 1, n, n);
      nda::for_each(X.shape(), [&](auto... i) { X(i...) = val(51, i...); });
      nda::for_each(H.shape(), [&](auto... i) { H(i...) = val(52, i...); });
      nda::array<ComplexType, 5> Xp = X;
      nda::array<ComplexType, 4> Hp = H;
      methods::project_lr_local_response(Xp, Hp, false);
      for (long w = 0; w < nw; ++w) for (long s = 0; s < ns; ++s) for (long a = 0; a < n; ++a) for (long b = 0; b < n; ++b)
        Xs(w, s, 0, a, b) = 0.5 * (X(w, s, 0, a, b) + X(w, s, 0, b, a));
      nda::array<ComplexType, 5> dsym = Xp - Xs;
      CHECK(fnorm(dsym) == 0.0);
      // dDelta of the Q != 0 projected response vs the formula Ginv Sym(dG) Ginv - Sym(dH) - dSigma, one block of 3 orbitals
      {
        nda::array<ComplexType, 5> Gm(nw, ns, 1, n, n), dS(nw, ns, 1, n, n);
        nda::for_each(Gm.shape(), [&](auto... i) { Gm(i...) = 0.2 * val(53, i...); dS(i...) = val(54, i...); });
        for (long w = 0; w < nw; ++w) for (long s = 0; s < ns; ++s) for (long a = 0; a < n; ++a) Gm(w, s, 0, a, a) += 2.0;
        methods::dc_cells_t cells{nda::array<long, 1>{0}, nda::array<long, 1>{n}, nda::array<long, 1>{0, 0, 0}};
        auto got = methods::lr_hybridization_w(Gm, Xp, Hp, dS, cells);
        nda::array<ComplexType, 5> ref(nw, ns, 1, n, n);
        for (long w = 0; w < nw; ++w) for (long s = 0; s < ns; ++s) {
          nda::matrix<ComplexType> Gi(n, n), Xm(n, n);
          for (long a = 0; a < n; ++a) for (long b = 0; b < n; ++b) { Gi(a, b) = Gm(w, s, 0, a, b); Xm(a, b) = Xs(w, s, 0, a, b); }
          nda::matrix<ComplexType> Gi_inv = nda::inverse(Gi);
          nda::matrix<ComplexType> R = Gi_inv * Xm * Gi_inv;
          for (long a = 0; a < n; ++a) for (long b = 0; b < n; ++b)
            ref(w, s, 0, a, b) = R(a, b) - 0.5 * (H(s, 0, a, b) + H(s, 0, b, a)) - dS(w, s, 0, a, b);
        }
        nda::array<ComplexType, 5> ddiff = got - ref;
        CHECK(fnorm(ddiff) < 1e-13 * fnorm(ref));
      }
      // a symmetric input is unchanged at Q != 0, with an exactly zero reported change
      nda::array<ComplexType, 5> Xq = Xs;
      nda::array<ComplexType, 4> Hq = H;
      auto ch = methods::project_lr_local_response(Xq, Hq, false);
      CHECK(ch[0] == 0.0);
    }
  }

} // local_projection_tests
