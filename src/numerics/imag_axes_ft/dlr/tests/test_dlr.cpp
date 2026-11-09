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
#include <map>
#include <random>
#include "nda/nda.hpp"

#include "utilities/test_common.hpp"
#include "numerics/imag_axes_ft/dlr/dlr_driver.hpp"
#include "numerics/imag_axes_ft/IAFT.hpp"
#include "numerics/imag_axes_ft/iaft_utils.hpp"

namespace bdft_tests {

  using utils::ARRAY_EQUAL;
  template<int N>
  using shape_t = std::array<long, N>;

  TEST_CASE("cppdlr", "[iaft][dlr]") {

    auto test_gfun_roundtrip = [&](double tol, cppdlr::statistic_t stat, bool ph_sym = false) {
      double beta = 100.0;
      double wmax = 1.0;
      double lambda = wmax * beta;
      double eps = 1e-12;
      int norb = 2;

      auto dlr_rf = cppdlr::build_dlr_rf(lambda, eps, cppdlr::SYM);

      // Get DLR imaginary time object
      auto it_ops = cppdlr::imtime_ops(lambda, dlr_rf, cppdlr::SYM);
      auto nts = it_ops.rank();
      nda::array<double, 1> tau_mesh(nts);
      auto dlr_it = cppdlr::rel2abs(it_ops.get_itnodes());
      for (int it = 0; it < nts; ++it) tau_mesh(it) = 2.0 * dlr_it(it) - 1.0;

      // DLR Matsubara frequency object
      auto if_ops = cppdlr::imfreq_ops(lambda, dlr_rf, stat, cppdlr::SYM);
      auto nw = if_ops.get_ifnodes().size();
      nda::array<long, 1> wn_mesh(nw);
      long stat_offset = (stat == cppdlr::Fermion) ? 1 : 0;
      for (int iw = 0; iw < nw; ++iw) wn_mesh(iw) = long(2 * if_ops.get_ifnodes(iw) + stat_offset);

      auto const iaft_stat = (stat == cppdlr::Fermion) ? imag_axes_ft::fermion : imag_axes_ft::boson;
      auto G_t_ref = imag_axes_ft::test_utils::build_g_tau(tau_mesh, beta, norb, ph_sym);
      auto G_w_ref = imag_axes_ft::test_utils::build_g_iw(wn_mesh, beta, iaft_stat, norb, ph_sym);
      app_log(2, "G_t_ref(0, 0, 0) = {}", G_t_ref(0, 0, 0));
      app_log(2, "G_w_ref(0, 0, 0) = {}", G_w_ref(0, 0, 0));

      // Verify tau -> iw recovers G(iw)
      {
        auto G_c = it_ops.vals2coefs(G_t_ref);
        auto G_w = if_ops.coefs2vals(beta, G_c);

        app_log(2, "G_w(0, 0, 0) = {}", G_w(0, 0, 0));
        ARRAY_EQUAL(G_w, G_w_ref, tol);

        auto it2if = cppdlr::build_it2if(it_ops, if_ops);
        auto G_t_ref_2D = nda::reshape(G_t_ref, std::array<long, 2>{nts, norb * norb});
        auto G_w_2D = nda::reshape(G_w, std::array<long, 2>{nw, norb * norb});
        G_w_2D = it2if * nda::matrix_const_view<ComplexType>(G_t_ref_2D) * beta;
        app_log(2, "G_w_2(0, 0, 0) = {}", G_w(0, 0, 0));
        ARRAY_EQUAL(G_w, G_w_ref, tol);
      }

      // Verify iw -> tau recovers G(t)
      {
        auto G_c = if_ops.vals2coefs(beta, G_w_ref);
        auto G_t = it_ops.coefs2vals(G_c);
        app_log(2, "G_t(0, 0, 0) = {}", G_t(0, 0, 0));
        ARRAY_EQUAL(G_t, G_t_ref, tol);

        auto if2it = cppdlr::build_if2it(if_ops, it_ops);
        auto G_w_ref_2D = nda::reshape(G_w_ref, std::array<long, 2>{nw, norb * norb});
        auto G_t_2D = nda::reshape(G_t, std::array<long, 2>{nts, norb * norb});
        G_t_2D = if2it * nda::matrix_const_view<ComplexType>(G_w_ref_2D) / beta;
        app_log(2, "G_t_2(0, 0, 0) = {}", G_t(0, 0, 0));
        ARRAY_EQUAL(G_t, G_t_ref, tol);

        auto it2if = cppdlr::build_it2if(it_ops, if_ops);
        nda::array<ComplexType, 3> G_w(nw, norb, norb);
        auto G_w_2D = nda::reshape(G_w, std::array<long, 2>{nw, norb * norb});
        G_w_2D = it2if * nda::matrix_const_view<ComplexType>(G_t_2D) * beta;
        app_log(2, "G_w_2(0, 0, 0) = {}\n", G_w(0, 0, 0));
        ARRAY_EQUAL(G_w, G_w_ref, tol);
      }
    };

    SECTION("fermion") {
      test_gfun_roundtrip(1e-10, cppdlr::Fermion);
    }

    SECTION("boson") {
      test_gfun_roundtrip(1e-10, cppdlr::Boson); 
    }

  }

  TEST_CASE("dlr_dimensions", "[iaft][dlr]") {
    auto check_dimensions = [&](double beta, double wmax, std::string prec) { 
      
      imag_axes_ft::dlr::DLR mydlr(beta, wmax, prec, true);

      REQUIRE(mydlr.nt_f == mydlr.nt_b);
      REQUIRE(mydlr.nw_f > 0);
      REQUIRE(mydlr.nw_b > 0);

      REQUIRE(mydlr.Ttw_ff.shape() == shape_t<2>{mydlr.nt_f, mydlr.nw_f});
      REQUIRE(mydlr.Twt_ff.shape() == shape_t<2>{mydlr.nw_f, mydlr.nt_f});
      REQUIRE(mydlr.Ttt_bf.shape() == shape_t<2>{mydlr.nt_b, mydlr.nt_f});
      REQUIRE(mydlr.Ttw_bb.shape() == shape_t<2>{mydlr.nt_b, mydlr.nw_b});
      REQUIRE(mydlr.Twt_bb.shape() == shape_t<2>{mydlr.nw_b, mydlr.nt_b});
      REQUIRE(mydlr.Ttt_fb.shape() == shape_t<2>{mydlr.nt_f, mydlr.nt_b});
      REQUIRE(mydlr.T_beta_t_ff.shape() == shape_t<1>{mydlr.nt_f});

    };

    check_dimensions(100.0, 1.2, "high");
    check_dimensions(100.0, 1.2, "medium");
    check_dimensions(100.0, 1.2, "low");
  }

  TEST_CASE("dlr_w_to_tau_left_inverse", "[iaft][dlr]") {
    using nda::range;
    auto mirror = [](nda::array<long, 1> const &wn_mesh) {
      std::map<long, long> index_of;
      for (long i = 0; i < wn_mesh.size(); ++i) index_of[wn_mesh(i)] = i;
      nda::array<long, 1> P(wn_mesh.size());
      for (long i = 0; i < wn_mesh.size(); ++i) P(i) = index_of.at(-wn_mesh(i));
      return P;
    };
    // random X with X(-wn) = conj(X(wn))
    auto paired_random = [&](nda::array<long, 1> const &wn_mesh, int seed, double scale) {
      std::mt19937 gen(seed);
      std::normal_distribution<double> dist(0.0, scale);
      auto P = mirror(wn_mesh);
      nda::vector<ComplexType> X(wn_mesh.size());
      X() = 0.0;
      for (long i = 0; i < X.size(); ++i) {
        if (P(i) < i) continue;
        ComplexType x(dist(gen), P(i) == i ? 0.0 : dist(gen));
        X(i) = x;
        X(P(i)) = std::conj(x);
      }
      return X;
    };

    for (auto [beta, wmax] : {std::pair{200.0, 3.0}, std::pair{100.0, 1.2}}) {
      imag_axes_ft::dlr::DLR dlr(beta, wmax, "high", false);

      for (auto stats : {imag_axes_ft::fermion, imag_axes_ft::boson}) {
        bool ferm = (stats == imag_axes_ft::fermion);
        auto const &Ttw = ferm ? dlr.Ttw_ff : dlr.Ttw_bb;
        auto const &Twt = ferm ? dlr.Twt_ff : dlr.Twt_bb;
        auto const &wn_mesh = ferm ? dlr.wn_mesh_f : dlr.wn_mesh_b;
        long nt = Ttw.extent(0);
        long nw = Ttw.extent(1);
        auto P = mirror(wn_mesh);

        // Ttw * Twt vs identity on the tau nodes
        {
          nda::matrix<ComplexType> TB = Ttw * Twt;
          double err = 0.0;
          for (long i = 0; i < nt; ++i)
            for (long j = 0; j < nt; ++j) err = std::max(err, std::abs(TB(i, j) - (i == j ? 1.0 : 0.0)));
          app_log(2, "dlr beta={} stats={}: max|Ttw Twt - I| = {}", beta, ferm, err);
          REQUIRE(err < 1e-11);
        }

        // Ttw(:, -wn) vs conj(Ttw(:, wn)), compared exactly; Im Ttw(:, wn = 0) vs 0
        {
          bool exact = true;
          for (long i = 0; i < nw; ++i) {
            for (long t = 0; t < nt; ++t) {
              exact = exact and (Ttw(t, P(i)) == std::conj(Ttw(t, i)));
              if (wn_mesh(i) == 0) exact = exact and (Ttw(t, i).imag() == 0.0);
            }
          }
          REQUIRE(exact);
        }

        // construct_w_interpolate_matrix on the DLR mesh vs Twt * Ttw, and applied to a smooth X(iw) vs X(iw)
        {
          auto M = dlr.construct_w_interpolate_matrix(wn_mesh, stats);
          nda::matrix<ComplexType> ref = Twt * Ttw;
          double err = max_element(nda::abs(M - ref));
          nda::vector<ComplexType> X(nw);
          for (long i = 0; i < nw; ++i) {
            ComplexType iw(0.0, M_PI * wn_mesh(i) / beta);
            X(i) = 1.0 / (iw - 0.7 * wmax / 3.0) + 0.3 / (iw + 0.2 * wmax / 3.0);
          }
          nda::vector<ComplexType> Y = nda::matrix<ComplexType>(M) * X;
          double err_x = nda::norm(Y - X) / nda::norm(X);
          app_log(2, "dlr beta={} stats={}: max|M - Twt Ttw| = {}, |M X - X|/|X| = {}", beta, ferm, err, err_x);
          REQUIRE(err < 1e-12);
          REQUIRE(err_x < 1e-11);
        }

        // M(-wn_out) X vs conj(M(wn_out) X) for paired X, on a positive-only mesh and a symmetric mesh
        {
          long nout = 200;
          nda::array<long, 1> pos(nout), neg(nout), sym(2 * nout);
          for (long j = 0; j < nout; ++j) {
            long n = ferm ? 2 * j + 1 : 2 * j;
            pos(j) = n;
            neg(j) = -n;
            sym(2 * j) = n;
            sym(2 * j + 1) = -n;
          }
          auto X = paired_random(wn_mesh, 7, 1.0);
          auto apply = [&](nda::array<long, 1> const &mesh) {
            auto M = dlr.construct_w_interpolate_matrix(mesh, stats);
            nda::vector<ComplexType> Y(mesh.size());
            Y = nda::matrix<ComplexType>(M) * X;
            return Y;
          };
          auto Yp = apply(pos);
          auto Yn = apply(neg);
          auto Ys = apply(sym);
          double den = nda::norm(Yp);
          double err_pn = 0.0, err_sym = 0.0;
          for (long j = 0; j < nout; ++j) {
            err_pn = std::max(err_pn, std::abs(Yn(j) - std::conj(Yp(j))));
            err_sym = std::max(err_sym, std::abs(Ys(2 * j + 1) - std::conj(Ys(2 * j))));
          }
          app_log(2, "dlr beta={} stats={}: pairing of M X: pos/neg {}, sym {}", beta, ferm, err_pn / den, err_sym / den);
          REQUIRE(err_pn / den < 1e-14);
          REQUIRE(err_sym / den < 1e-14);
        }

        // ph_sym=true: boson M with the positive half of the input, vs the folded columns of M
        if (not ferm) {
          nda::array<long, 1> wn_out = {0, 2, 4, 6, 20};
          auto M = dlr.construct_w_interpolate_matrix(wn_out, stats, false);
          auto Mph = dlr.construct_w_interpolate_matrix(wn_out, stats, true);
          long nw_half = nw / 2 + nw % 2;
          REQUIRE(Mph.shape() == std::array<long, 2>{5, nw_half});
          double err = 0.0;
          for (long iw = 0; iw < nw_half; ++iw) {
            long ipw = nw / 2 + iw, imw = nw / 2 - iw;
            for (long o = 0; o < 5; ++o)
              err = std::max(err, std::abs(Mph(o, iw) - (ipw == imw ? M(o, ipw) : M(o, ipw) + M(o, imw))));
          }
          REQUIRE(err < 1e-14);
        }
      }
    }
  }

  // Ttw_ff_w_weighted vs its definition as the |omega|-weighted least-squares left inverse of Twt_ff, and Ttw_ff
  // vs its definition as the unweighted one (random complex x, nw_f = nt_f + 1):
  //  (1) Ttw_w Twt vs 1 (round-off), so tau -> omega -> tau is exact;
  //  (2) weighted normal equations Twt^H W^2 (Twt Ttw_w x - x) vs 0, relative to |Twt^H W^2 x|
  //      (round-off scaled by cond(Twt), hence the 1e-8 tolerance);
  //  (3) pairing Ttw_w(:, -wn) vs conj(Ttw_w(:, wn));
  //  (4) orthogonal normal equations Twt^H (Twt Ttw x - x) vs 0, relative to |Twt^H x|, same tolerance.
  TEST_CASE("dlr_w_to_tau_weighted_least_squares", "[iaft][dlr]") {
    double beta = 1000.0, wmax = 30.0;
    imag_axes_ft::dlr::DLR dlr(beta, wmax, "high", false);
    long nt = dlr.nt_f, nw = dlr.nw_f;
    REQUIRE(nw == nt + 1);

    // Ttw_w Twt vs 1
    nda::matrix<ComplexType> TB = dlr.Ttw_ff_w_weighted * dlr.Twt_ff;
    double err_id = 0.0;
    for (long i = 0; i < nt; ++i)
      for (long j = 0; j < nt; ++j) err_id = std::max(err_id, std::abs(TB(i, j) - (i == j ? 1.0 : 0.0)));
    app_log(2, "dlr weighted: max|Ttw_w Twt - I| = {}", err_id);
    REQUIRE(err_id < 1e-11);

    // Normal equations: weighted for Ttw_w (weight w^2), unweighted for Ttw (weight 1)
    std::mt19937 gen(5);
    std::normal_distribution<double> dist(0.0, 1.0);
    nda::vector<ComplexType> x(nw);
    for (long i = 0; i < nw; ++i) x(i) = ComplexType(dist(gen), dist(gen));
    auto normal_eq_err = [&](nda::matrix<ComplexType> const &T, bool weighted) {
      nda::vector<ComplexType> y(nt), r(nw), Wr(nw), Wx(nw);
      y = T * x;
      r = dlr.Twt_ff * y - x;
      for (long i = 0; i < nw; ++i) {
        double w2 = weighted ? std::pow(std::abs(double(dlr.wn_mesh_f(i))) * M_PI / beta, 2) : 1.0;
        Wr(i) = w2 * r(i);
        Wx(i) = w2 * x(i);
      }
      nda::vector<ComplexType> ne_r(nt), ne_x(nt);
      ne_r = nda::transpose(nda::conj(dlr.Twt_ff)) * Wr;
      ne_x = nda::transpose(nda::conj(dlr.Twt_ff)) * Wx;
      return nda::norm(ne_r) / nda::norm(ne_x);
    };
    double err_ne_w = normal_eq_err(dlr.Ttw_ff_w_weighted, true);
    double err_ne_o = normal_eq_err(dlr.Ttw_ff, false);
    app_log(2, "dlr weighted normal equations: {}, orthogonal normal equations: {}", err_ne_w, err_ne_o);
    REQUIRE(err_ne_w < 1e-8);
    REQUIRE(err_ne_o < 1e-8);

    // Pairing
    std::map<long, long> index_of;
    for (long i = 0; i < nw; ++i) index_of[dlr.wn_mesh_f(i)] = i;
    double err_pair = 0.0;
    for (long i = 0; i < nw; ++i)
      for (long t = 0; t < nt; ++t)
        err_pair = std::max(err_pair, std::abs(dlr.Ttw_ff_w_weighted(t, index_of.at(-dlr.wn_mesh_f(i))) -
                                               std::conj(dlr.Ttw_ff_w_weighted(t, i))));
    REQUIRE(err_pair == 0.0);
  }

  // IAFT::w_to_tau_w_weighted vs its definition: on DLR it is the gemm with dlr.Ttw_ff_w_weighted (bitwise); on IR it
  // is w_to_tau(X, fermion) (bitwise), since IR has no weighted variant. Random complex X(iw, 3).
  TEST_CASE("iaft_w_to_tau_w_weighted_dispatch", "[iaft][dlr]") {
    double beta = 1000.0, wmax = 30.0;
    std::mt19937 gen(11);
    std::normal_distribution<double> dist(0.0, 1.0);
    auto random_X = [&](long nw) {
      nda::array<ComplexType, 2> X(nw, 3);
      for (long i = 0; i < nw; ++i)
        for (long j = 0; j < 3; ++j) X(i, j) = ComplexType(dist(gen), dist(gen));
      return X;
    };

    // DLR: w_to_tau_w_weighted vs gemm with Ttw_ff_w_weighted
    {
      imag_axes_ft::IAFT iaft(beta, wmax, imag_axes_ft::dlr_basis, "high");
      imag_axes_ft::dlr::DLR dlr(beta, wmax, "high", false);
      auto X = random_X(iaft.nw_f());
      nda::array<ComplexType, 2> Y(iaft.nt_f(), 3), Y_ref(iaft.nt_f(), 3);
      iaft.w_to_tau_w_weighted(X, Y);
      nda::blas::gemm(dlr.Ttw_ff_w_weighted, X, Y_ref);
      REQUIRE(nda::max_element(nda::abs(Y - Y_ref)) == 0.0);
      // and it differs from the orthogonal w_to_tau
      nda::array<ComplexType, 2> Y_o(iaft.nt_f(), 3);
      iaft.w_to_tau(X, Y_o, imag_axes_ft::fermion);
      REQUIRE(nda::max_element(nda::abs(Y - Y_o)) > 0.0);
    }
    // IR: w_to_tau_w_weighted vs w_to_tau
    {
      imag_axes_ft::IAFT iaft(beta, wmax, imag_axes_ft::ir_basis, "high");
      auto X = random_X(iaft.nw_f());
      nda::array<ComplexType, 2> Y(iaft.nt_f(), 3), Y_ref(iaft.nt_f(), 3);
      iaft.w_to_tau_w_weighted(X, Y);
      iaft.w_to_tau(X, Y_ref, imag_axes_ft::fermion);
      REQUIRE(nda::max_element(nda::abs(Y - Y_ref)) == 0.0);
    }
  }

  // Node Dyson G = w_to_tau_w_weighted[(iw - h - tau_to_w Sigma)^{-1}] on the DLR grid vs the exact G(tau) of a gapped
  // 2-orbital model whose Sigma(iw) = sum_p V_p V_p^H / (iw - xi_p) comes from bath poles inside wmax. The exact G
  // is the system block of the resolvent of the embedding Hamiltonian [[h, V], [V^H, diag(xi)]] and Sigma(tau) is
  // exact on the nodes. Relative L2 error of the |omega|-weighted fit < 2e-11 (measured 2.6e-12); the orthogonal fit
  // (w_to_tau) is > 1e-10 on this grid (measured 3.5e-10), which is why the Dyson path uses the weighted one.
  TEST_CASE("dlr_node_dyson_vs_exact", "[iaft][dlr]") {
    double beta = 1000.0, wmax = 30.0;
    imag_axes_ft::dlr::DLR dlr(beta, wmax, "high", false);
    long nt = dlr.nt_f, nw = dlr.nw_f;
    constexpr int no = 2, nb = 4, ne = no + nb;
    double h[no][no] = {{-0.15, 0.05}, {0.05, 0.25}};
    double xi[nb] = {-1.2, 0.9, -0.6 * wmax, 0.75 * wmax};
    double V[no][nb] = {{0.30, 0.10, 0.25, 0.20}, {0.05, 0.30, 0.15, 0.25}};

    // g(tau) = -exp(-tau e) / (1 + exp(-beta e)), written with tau (e >= 0) or beta - tau (e < 0) to avoid overflow
    auto g_tau = [&](double tau, double bmt, double e) {
      return e >= 0 ? -std::exp(-tau * e) / (1.0 + std::exp(-beta * e)) : -std::exp(bmt * e) / (1.0 + std::exp(beta * e));
    };

    // Eigen-decomposition of the embedding Hamiltonian (real symmetric)
    std::vector<double> A(ne * ne, 0.0), E(ne);
    for (int a = 0; a < no; ++a) {
      for (int b = 0; b < no; ++b) A[a + ne * b] = h[a][b];
      for (int p = 0; p < nb; ++p) A[a + ne * (no + p)] = A[(no + p) + ne * a] = V[a][p];
    }
    for (int p = 0; p < nb; ++p) A[(no + p) + ne * (no + p)] = xi[p];
    {
      int lwork = 64 * ne, info = 0;
      std::vector<double> work(lwork);
      nda::lapack::f77::syev('V', 'U', ne, A.data(), ne, E.data(), work.data(), lwork, info);
      REQUIRE(info == 0);
    }

    // Exact G(tau) and Sigma(tau) on the DLR tau nodes, flattened as (t, a, b)
    nda::array<ComplexType, 3> G_ex(nt, no, no), Sig_t(nt, no, no);
    for (long t = 0; t < nt; ++t) {
      double t_rel = dlr._it_ops.get_itnodes(t);
      double tau = t_rel >= 0 ? t_rel * beta : beta + t_rel * beta;
      double bmt = t_rel >= 0 ? beta - t_rel * beta : -t_rel * beta;
      for (int a = 0; a < no; ++a)
        for (int b = 0; b < no; ++b) {
          double ge = 0.0, sg = 0.0;
          for (int k = 0; k < ne; ++k) ge += A[a + ne * k] * g_tau(tau, bmt, E[k]) * A[b + ne * k];
          for (int p = 0; p < nb; ++p) sg += V[a][p] * g_tau(tau, bmt, xi[p]) * V[b][p];
          G_ex(t, a, b) = ge;
          Sig_t(t, a, b) = sg;
        }
    }

    // Node Dyson: G(iw_n) = (iw_n - h - Sigma(iw_n))^{-1}, then w_to_tau
    nda::array<ComplexType, 3> G_w(nw, no, no);
    nda::matrix<ComplexType> Sig_flat(nt, no * no);
    for (long t = 0; t < nt; ++t)
      for (int a = 0; a < no; ++a)
        for (int b = 0; b < no; ++b) Sig_flat(t, a * no + b) = Sig_t(t, a, b);
    nda::matrix<ComplexType> Sw = dlr.Twt_ff * Sig_flat;
    for (long n = 0; n < nw; ++n) {
      ComplexType iw(0.0, M_PI * dlr.wn_mesh_f(n) / beta);
      ComplexType m00 = iw - h[0][0] - Sw(n, 0), m01 = -h[0][1] - Sw(n, 1);
      ComplexType m10 = -h[1][0] - Sw(n, 2), m11 = iw - h[1][1] - Sw(n, 3);
      ComplexType det = m00 * m11 - m01 * m10;
      G_w(n, 0, 0) = m11 / det;
      G_w(n, 0, 1) = -m01 / det;
      G_w(n, 1, 0) = -m10 / det;
      G_w(n, 1, 1) = m00 / det;
    }
    nda::matrix<ComplexType> G_flat(nw, no * no);
    for (long n = 0; n < nw; ++n)
      for (int a = 0; a < no; ++a)
        for (int b = 0; b < no; ++b) G_flat(n, a * no + b) = G_w(n, a, b);
    auto rel_err = [&](nda::matrix<ComplexType> const &Gt) {
      double num = 0.0, den = 0.0;
      for (long t = 0; t < nt; ++t)
        for (int a = 0; a < no; ++a)
          for (int b = 0; b < no; ++b) {
            num += std::norm(Gt(t, a * no + b) - G_ex(t, a, b));
            den += std::norm(G_ex(t, a, b));
          }
      return std::sqrt(num / den);
    };
    nda::matrix<ComplexType> Gt_w = dlr.Ttw_ff_w_weighted * G_flat;
    nda::matrix<ComplexType> Gt_o = dlr.Ttw_ff * G_flat;
    double err_w = rel_err(Gt_w), err_o = rel_err(Gt_o);
    app_log(2, "dlr node Dyson vs exact: weighted {}, orthogonal {}", err_w, err_o);
    REQUIRE(err_w < 2e-11);
    REQUIRE(err_o > 1e-10);
  }

  // DLR Ttw * X(iw) for a noisy but paired X(iw) (Dyson-like G with a non-DLR-representable Sigma noise of
  // size 1e-6, as from CT-SEG) vs the requirement that the tau function and its density are real
  TEST_CASE("dlr_w_to_tau_noisy_real", "[iaft][dlr]") {
    imag_axes_ft::dlr::DLR dlr(200.0, 3.0, "high", false);
    auto const &wn_mesh = dlr.wn_mesh_f;
    long nw = wn_mesh.size();
    std::map<long, long> index_of;
    for (long i = 0; i < nw; ++i) index_of[wn_mesh(i)] = i;

    std::mt19937 gen(11);
    std::normal_distribution<double> dist(0.0, 1e-6);
    nda::array<ComplexType, 1> eta(nw);
    for (long i = 0; i < nw; ++i) {
      long m = index_of.at(-wn_mesh(i));
      if (m < i) continue;
      eta(i) = ComplexType(dist(gen), dist(gen));
      eta(m) = std::conj(eta(i));
    }
    std::array<double, 3> w = {0.3, 0.5, 0.2}, eps = {-0.4, 0.1, 0.9};
    double mu = 0.2;
    nda::vector<ComplexType> X(nw);
    for (long i = 0; i < nw; ++i) {
      ComplexType iw(0.0, M_PI * wn_mesh(i) / 200.0);
      ComplexType sigma = 0.1 / (iw - 0.7) + eta(i);
      X(i) = 0.0;
      for (int k = 0; k < 3; ++k) X(i) += w[k] / (iw + mu - eps[k] - sigma);
    }
    nda::vector<ComplexType> Xt(dlr.nt_f);
    Xt = dlr.Ttw_ff * X;
    double im_max = 0.0, re_max = 0.0;
    for (long t = 0; t < Xt.size(); ++t) {
      im_max = std::max(im_max, std::abs(Xt(t).imag()));
      re_max = std::max(re_max, std::abs(Xt(t).real()));
    }
    ComplexType dens = nda::blas::dot(dlr.T_beta_t_ff, Xt);
    app_log(2, "dlr noisy: max|Im X(tau)| = {}, max|Re X(tau)| = {}, |Im dens|/|dens| = {}", im_max, re_max,
            std::abs(dens.imag()) / std::abs(dens));
    REQUIRE(im_max < 1e-13);
    REQUIRE(std::abs(dens.imag()) / std::abs(dens) < 1e-13);
  }

  // Tlt (DLR tau-node values -> kernel-SVD mode coefficients):
  //  (1) orthonormality, (Tlt A_tau)^H (Tlt A_tau) vs beta * cppdlr's analytic Gram matrix get_ipmat() of the DLR
  //      basis functions (A_tau = cf2it). Compared relative to the largest entry: the Gram matrix has condition
  //      number ~1e27, so only the leading block is meaningful at double precision.
  //  (2) leakage ratio max|a_{r-2..r-1}| / max|a_0| of a two-semicircle model G (Gauss-Chebyshev quadrature of the
  //      Lehmann kernel at the node times, independent of the fine-grid construction): ~1e-12 for clean node values
  //      (DLR error ~ eps, 1e-13), growing with seeded white noise on the node values.
  TEST_CASE("dlr_leakage_modes", "[iaft][dlr]") {
    struct Grid { double beta, wmax; std::string prec; };
    for (auto [beta, wmax, prec] : {Grid{200.0, 3.0, "high"}, Grid{100.0, 1.2, "high"}, Grid{200.0, 3.0, "medium"}}) {
      imag_axes_ft::dlr::DLR dlr(beta, wmax, prec, false);
      long r = dlr.nt_f;
      REQUIRE(dlr.Tlt.shape() == shape_t<2>{r, r});

      // (1)
      {
        auto A_tau = nda::matrix<ComplexType>(dlr._it_ops.get_cf2it());
        nda::matrix<ComplexType> TA = dlr.Tlt * A_tau;
        nda::matrix<ComplexType> gram = nda::dagger(TA) * TA;
        auto ipmat = dlr._it_ops.get_ipmat();
        double max_ref = 0.0, max_err = 0.0;
        for (long i = 0; i < r; ++i)
          for (long j = 0; j < r; ++j) {
            max_ref = std::max(max_ref, std::abs(beta * ipmat(i, j)));
            max_err = std::max(max_err, std::abs(gram(i, j) - beta * ipmat(i, j)));
          }
        app_log(2, "dlr_leakage_modes beta={} wmax={} {}: max|Gram - beta*ipmat|/max|beta*ipmat| = {}", beta, wmax, prec, max_err / max_ref);
        REQUIRE(max_err < 1e-12 * max_ref);
      }

      // (2)
      {
        double lambda = beta * wmax;
        long n = 400;
        nda::vector<double> om(2 * n), wq(2 * n);
        for (long i = 1; i <= n; ++i) {
          double th = M_PI * i / (n + 1);
          double w = M_PI / (n + 1) * std::sin(th) * std::sin(th);
          om(i - 1) = 0.5 * lambda + 0.25 * lambda * std::cos(th);
          om(n + i - 1) = -0.5 * lambda + 0.25 * lambda * std::cos(th);
          wq(i - 1) = wq(n + i - 1) = w;
        }
        auto t_rel = nda::vector<double>(dlr._it_ops.get_itnodes());
        auto K = cppdlr::build_k_it(nda::vector_const_view<double>(t_rel), nda::vector_const_view<double>(om));
        nda::matrix<ComplexType> G(r, 1);
        for (long j = 0; j < r; ++j) {
          double g = 0.0;
          for (long i = 0; i < 2 * n; ++i) g += wq(i) * K(j, i);
          G(j, 0) = g;
        }
        double gmax = 0.0;
        for (long j = 0; j < r; ++j) gmax = std::max(gmax, std::abs(G(j, 0)));

        auto leakage = [&](nda::matrix<ComplexType> const &g) {
          nda::vector<ComplexType> a = dlr.Tlt * g(nda::range::all, 0);
          return std::max(std::abs(a(r - 2)), std::abs(a(r - 1))) / std::abs(a(0));
        };
        double clean = leakage(G);
        app_log(2, "dlr_leakage_modes beta={} wmax={} {}: clean = {}", beta, wmax, prec, clean);
        REQUIRE(clean < (prec == "high" ? 1e-11 : 1e-8));

        std::mt19937 gen(7);
        std::normal_distribution<double> dist(0.0, 1.0);
        double prev = clean;
        for (double noise : {1e-9, 1e-6}) {
          auto Gn = G;
          for (long j = 0; j < r; ++j) Gn(j, 0) += noise * gmax * dist(gen);
          double lk = leakage(Gn);
          app_log(2, "dlr_leakage_modes beta={} wmax={} {}: noise {} -> {}", beta, wmax, prec, noise, lk);
          REQUIRE(lk > prev);
          REQUIRE(lk > noise);
          REQUIRE(lk < 30 * noise);
          prev = lk;
        }
      }
    }
  }

  // IAFT::check_leakage on the DLR basis (serial, comm == nullptr) vs the same ratio evaluated directly from Tlt.
  TEST_CASE("dlr_check_leakage", "[iaft][dlr]") {
    imag_axes_ft::IAFT iaft(200.0, 3.0, imag_axes_ft::dlr_basis, "high");
    long r = iaft.nt_f();
    std::mt19937 gen(3);
    std::normal_distribution<double> dist(0.0, 1.0);
    nda::array<ComplexType, 2> X(r, 3);
    for (auto &x : X) x = ComplexType(dist(gen), dist(gen));
    auto T = nda::matrix<ComplexType>(iaft.Tlt_ff());
    nda::matrix<ComplexType> a = T * nda::matrix<ComplexType>(X);
    double first = 0.0, last = 0.0;
    for (long i = 0; i < 3; ++i) {
      first = std::max(first, std::abs(a(0, i)));
      last = std::max({last, std::abs(a(r - 2, i)), std::abs(a(r - 1, i))});
    }
    boost::mpi3::communicator *no_comm = nullptr;
    double lk = iaft.check_leakage(X, imag_axes_ft::fermion, no_comm, "random");
    REQUIRE(lk == Approx(last / first).epsilon(1e-12));
    REQUIRE(lk > 1e-5);
  }

} // bdft_tests
