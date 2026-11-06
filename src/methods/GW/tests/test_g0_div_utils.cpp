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

#include "utilities/test_common.hpp"
#include "utilities/mpi_context.h"
#include "mean_field/default_MF.hpp"
#include "methods/GW/g0_div_utils.hpp"

namespace g0_div_utils_tests {

  using methods::solvers::div_utils;

  constexpr double two_pi = 2.0 * 3.14159265358979;

  // Cubic lattice (lattv = identity) with a 4x4x4 q-grid, crystal coordinates in [-0.5, 0.25]^3.
  // Q_bz: the full-BZ q-points (cartesian), with the IBZ representatives first as in the mean-field objects.
  // The IBZ is that of the cubic point group + time reversal: q ~ q' if the sorted |coordinates| agree.
  // Q_ibz: the representatives, the first member of each class in the loop order below (so that the IBZ
  // misses some of the +-b axes). bz_to_ibz: full -> IBZ index.
  struct cubic_q4_t {
    nda::array<double, 2> Q_bz, Q_ibz;
    nda::array<int, 1> bz_to_ibz;
    nda::array<double, 2> lattv = {{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};

    cubic_q4_t() {
      std::vector<std::array<double, 3>> pts, keys, reps;
      std::vector<int> cls;
      for (int k = 1; k >= -2; --k) for (int j = 1; j >= -2; --j) for (int i = 1; i >= -2; --i) {
        std::array<double, 3> c = {i / 4.0, j / 4.0, k / 4.0};
        std::array<double, 3> key = {std::abs(c[0]), std::abs(c[1]), std::abs(c[2])};
        std::sort(key.begin(), key.end());
        auto it = std::find(keys.begin(), keys.end(), key);
        if (it == keys.end()) { keys.push_back(key); reps.push_back(c); it = keys.end() - 1; }
        pts.push_back(c);
        cls.push_back(int(it - keys.begin()));
      }
      long nbz = pts.size(), nibz = reps.size();
      // order: representatives first, then the other points
      std::vector<int> order;
      for (long n = 0; n < nibz; ++n)
        order.push_back(int(std::find(pts.begin(), pts.end(), reps[n]) - pts.begin()));
      for (long n = 0; n < nbz; ++n)
        if (std::find(order.begin(), order.end(), int(n)) == order.end()) order.push_back(int(n));
      Q_bz = nda::array<double, 2>(nbz, 3);
      bz_to_ibz = nda::array<int, 1>(nbz);
      Q_ibz = nda::array<double, 2>(nibz, 3);
      for (long n = 0; n < nbz; ++n) {
        for (int d = 0; d < 3; ++d) Q_bz(n, d) = two_pi * pts[order[n]][d];
        bz_to_ibz(n) = cls[order[n]];
      }
      for (long n = 0; n < nibz; ++n)
        for (int d = 0; d < 3; ++d) Q_ibz(n, d) = Q_bz(n, d);
    }
  };

  // head(q) = a_w + b_w |q|^2 + c_w |q|^4 per frequency w, evaluated at the cartesian q-points Q
  nda::array<ComplexType, 2> make_head(nda::array<double, 2> const& Q, double quad) {
    long niw = 3;
    nda::array<ComplexType, 2> eps(niw, Q.shape(0));
    for (long w = 0; w < niw; ++w)
      for (long iq = 0; iq < Q.shape(0); ++iq) {
        double q2 = Q(iq,0)*Q(iq,0) + Q(iq,1)*Q(iq,1) + Q(iq,2)*Q(iq,2);
        eps(w, iq) = ComplexType(-0.8 + 0.1*w, 0.05*w) + ComplexType(0.3 - 0.07*w, 0.02) * q2 + quad * q2 * q2;
      }
    return eps;
  }

  TEST_CASE("gygi_head_cubic_ibz_vs_full_bz", "[methods_gw][div_utils]") {
    // div_utils::extrapolate_gygi_q0 on the IBZ q-set (direction search on the full-BZ list, values from
    // qp_to_ibz columns) vs on the full-BZ q-set, for a head that is linear in |q|^2 (both must return the
    // q=0 value, exactly up to round-off) and quadratic in |q|^2 (both must agree).
    // Negative control: searching the axis points only among the IBZ points (only +b3 {0.25, 0.5}) gives
    // a different value.
    cubic_q4_t g;
    auto eps_bz  = make_head(g.Q_bz, 0.0);
    auto eps_ibz = make_head(g.Q_ibz, 0.0);
    nda::array<int, 1> identity(g.Q_bz.shape(0));
    for (int i = 0; i < identity.shape(0); ++i) identity(i) = i;
    nda::array<int, 1> ibz_identity(g.Q_ibz.shape(0));
    for (int i = 0; i < ibz_identity.shape(0); ++i) ibz_identity(i) = i;

    auto head_bz  = div_utils::extrapolate_gygi_q0(eps_bz, g.Q_bz, g.lattv, identity, 10, false);
    auto head_ibz = div_utils::extrapolate_gygi_q0(eps_ibz, g.Q_bz, g.lattv, g.bz_to_ibz, 10, false);
    auto head_ibz_only = div_utils::extrapolate_gygi_q0(eps_ibz, g.Q_ibz, g.lattv, ibz_identity, 10, false);

    nda::array<ComplexType, 1> exact(3);
    for (int w = 0; w < 3; ++w) exact(w) = ComplexType(-0.8 + 0.1*w, 0.05*w);
    double err_bz = nda::norm(head_bz - exact), err_ibz = nda::norm(head_ibz - exact);
    double diff = nda::norm(head_bz - head_ibz);
    double err_ibz_only = nda::norm(head_ibz_only - exact);
    INFO("linear: |full-BZ - exact| = " << err_bz << ", |IBZ - exact| = " << err_ibz << ", |IBZ - full-BZ| = " << diff
         << ", negative control |IBZ-only search - exact| = " << err_ibz_only);
    CHECK(err_bz < 1e-11);
    CHECK(err_ibz < 1e-11);
    CHECK(diff < 1e-11);
    CHECK(err_ibz_only > 1e-3);

    auto eps2_bz  = make_head(g.Q_bz, 0.2);
    auto eps2_ibz = make_head(g.Q_ibz, 0.2);
    auto head2_bz  = div_utils::extrapolate_gygi_q0(eps2_bz, g.Q_bz, g.lattv, identity, 10, false);
    auto head2_ibz = div_utils::extrapolate_gygi_q0(eps2_ibz, g.Q_bz, g.lattv, g.bz_to_ibz, 10, false);
    double diff2 = nda::norm(head2_bz - head2_ibz);
    INFO("quadratic: |IBZ - full-BZ| = " << diff2);
    CHECK(diff2 < 1e-11);
  }

  TEST_CASE("gygi_head_boundary_point_both_sides", "[methods_gw][div_utils]") {
    // div_utils::find_n_closest_per_direction on the full 4x4x4 grid vs the expected axis lists:
    // +bi and -bi both hold {0.25, 0.5} (the zone-boundary point is in both lists: eps_inv_00(-q) = eps_inv_00(q), |-q|^2 = |q|^2).
    cubic_q4_t g;
    auto lists = div_utils::find_n_closest_per_direction(g.Q_bz, g.lattv, 11);
    for (int dir = 0; dir < 6; ++dir) {
      std::vector<double> coords;
      for (int iq : lists[dir]) {
        double q2 = g.Q_bz(iq,0)*g.Q_bz(iq,0) + g.Q_bz(iq,1)*g.Q_bz(iq,1) + g.Q_bz(iq,2)*g.Q_bz(iq,2);
        coords.push_back(std::sqrt(q2) / two_pi);
      }
      INFO("direction " << dir << " has " << coords.size() << " points");
      REQUIRE(coords.size() == 2);
      CHECK(coords[0] == Approx(0.25));
      CHECK(coords[1] == Approx(0.5));
    }
  }

  TEST_CASE("gygi_head_two_dim_skips_b3", "[methods_gw][div_utils]") {
    // div_utils::extrapolate_gygi_q0 with two_dim=true vs the q=0 value of a head that is linear in |q|^2
    // in the xy directions and carries a constant offset 5 on the b3 axis: only the b1, b2 directions enter.
    cubic_q4_t g;
    auto eps = make_head(g.Q_bz, 0.0);
    for (int iq = 0; iq < g.Q_bz.shape(0); ++iq) {
      bool on_b3 = std::abs(g.Q_bz(iq,0)) < 1e-10 and std::abs(g.Q_bz(iq,1)) < 1e-10;
      if (on_b3) for (int w = 0; w < 3; ++w) eps(w, iq) += 5.0;
    }
    nda::array<int, 1> identity(g.Q_bz.shape(0));
    for (int i = 0; i < identity.shape(0); ++i) identity(i) = i;
    auto head2d = div_utils::extrapolate_gygi_q0(eps, g.Q_bz, g.lattv, identity, 10, true);
    auto head3d = div_utils::extrapolate_gygi_q0(eps, g.Q_bz, g.lattv, identity, 10, false);
    nda::array<ComplexType, 1> exact(3);
    for (int w = 0; w < 3; ++w) exact(w) = ComplexType(-0.8 + 0.1*w, 0.05*w);
    INFO("|2D - exact| = " << nda::norm(head2d - exact) << ", |3D - exact| = " << nda::norm(head3d - exact));
    CHECK(nda::norm(head2d - exact) < 1e-11);
    CHECK(std::abs(head3d(0) - exact(0) - ComplexType(5.0 * 2.0 / 6.0)) < 1e-11);
  }

  TEST_CASE("gygi_head_lih223_ibz_vs_full_bz", "[methods_gw][div_utils]") {
    // div_utils::extrapolate_eps_inv_q0("gygi") on the IBZ q-set of qe_lih223_sym vs the full-BZ q-set of
    // qe_lih223, vs the average over the six directions +-b1, +-b2, +-b3 of the head at the axis point of each
    // direction (0.5 on b1 and b2 on both sides, 1/3 on b3), computed from the lattice vectors. Both fixtures
    // have one axis point per direction, so the extrapolation returns that value.
    auto& mpi = utils::make_unit_test_mpi_context();
    auto mf_sym = mf::default_MF(mpi, "qe_lih223_sym", mf::h5_input_type);
    auto mf_full = mf::default_MF(mpi, "qe_lih223", mf::h5_input_type);

    auto head_of = [](auto const& Qpts_ibz) {
      nda::array<double, 2> Q(Qpts_ibz);
      return make_head(Q, 0.0);
    };
    auto head_sym = div_utils::extrapolate_eps_inv_q0(head_of(mf_sym.Qpts_ibz()), mf_sym, "gygi");
    auto head_full = div_utils::extrapolate_eps_inv_q0(head_of(mf_full.Qpts_ibz()), mf_full, "gygi");

    // expected: q = 2 pi lattv^{-1} c (crystal c = lattv q / 2 pi)
    nda::matrix<double> L(mf_full.lattv());
    auto Linv = nda::inverse(L);
    auto q2_of = [&](double c0, double c1, double c2) {
      nda::array<double, 1> c = {c0, c1, c2};
      nda::array<double, 1> q(3);
      nda::blas::gemv(two_pi, Linv, c, 0.0, q);
      return q(0)*q(0) + q(1)*q(1) + q(2)*q(2);
    };
    double q2_b1 = q2_of(0.5, 0, 0), q2_b2 = q2_of(0, 0.5, 0), q2_b3 = q2_of(0, 0, 1.0/3.0);
    nda::array<ComplexType, 1> exact(3);
    for (int w = 0; w < 3; ++w) {
      auto f = [&](double q2) { return ComplexType(-0.8 + 0.1*w, 0.05*w) + ComplexType(0.3 - 0.07*w, 0.02) * q2; };
      exact(w) = (2.0 * f(q2_b1) + 2.0 * f(q2_b2) + 2.0 * f(q2_b3)) / 6.0;
    }
    INFO("|IBZ - exact| = " << nda::norm(head_sym - exact) << ", |full-BZ - exact| = " << nda::norm(head_full - exact)
         << ", |IBZ - full-BZ| = " << nda::norm(head_sym - head_full));
    CHECK(nda::norm(head_sym - exact) < 1e-11);
    CHECK(nda::norm(head_full - exact) < 1e-11);
    CHECK(nda::norm(head_sym - head_full) < 1e-11);
  }

  TEST_CASE("gygi_average_counts_full_bz_multiplicity", "[methods_gw][div_utils]") {
    // div_utils::extrapolate_eps_inv_q0("gygi_average") on the IBZ q-set of the _sym fixtures with a head equal
    // to the IBZ column index (constant in each star), vs a brute-force mean over the smallest-|q| points of the
    // full-BZ list Qpts() of the IBZ column of each point (each star weighted by its multiplicity).
    // All three fixtures have one star at the smallest |q|, so this checks the column mapping, not the weighting.
    auto& mpi = utils::make_unit_test_mpi_context();
    for (std::string name : {"qe_lih222_sym", "qe_lih223_sym", "qe_si333_sym"}) {
    auto mf = mf::default_MF(mpi, name, mf::h5_input_type);
    long nibz = mf.nqpts_ibz();
    nda::array<ComplexType, 2> eps(3, nibz);
    for (int w = 0; w < 3; ++w) for (long i = 0; i < nibz; ++i) eps(w, i) = ComplexType(double(i) + w, 0.5 * w);
    auto head = div_utils::extrapolate_eps_inv_q0(eps, mf, "gygi_average");

    auto Q = mf.Qpts();
    auto norm_of = [&](long iq) { return std::sqrt(Q(iq,0)*Q(iq,0) + Q(iq,1)*Q(iq,1) + Q(iq,2)*Q(iq,2)); };
    double qmin = 1e300;
    for (long iq = 0; iq < mf.nqpts(); ++iq) if (norm_of(iq) > 1e-10) qmin = std::min(qmin, norm_of(iq));
    double sum = 0; long n = 0, nstar_min = 0;
    nda::array<int, 1> seen(nibz); seen() = 0;
    for (long iq = 0; iq < mf.nqpts(); ++iq) {
      if (norm_of(iq) > 1e-10 and std::abs(norm_of(iq) - qmin) < 1e-10) {
        int i = mf.qp_to_ibz(iq);
        sum += i; ++n;
        if (!seen(i)) { seen(i) = 1; ++nstar_min; }
      }
    }
    INFO("smallest-|q|: " << n << " full-BZ points in " << nstar_min << " stars; head(0) = " << head(0));
    CHECK(std::abs(head(0) - ComplexType(sum / n)) < 1e-12);
    CHECK(std::abs(head(1) - head(0) - ComplexType(1.0, 0.5)) < 1e-12);
    }
  }

} // g0_div_utils_tests
