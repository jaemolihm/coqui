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

#include <complex>

#include "catch2/catch.hpp"

#include "configuration.hpp"
#include "IO/AppAbort.hpp"
#include "IO/app_loggers.h"

#include "nda/nda.hpp"
#include "utilities/test_common.hpp"
#include "utilities/Timer.hpp"
#include "utilities/harmonics.h"
#include "utilities/ylm_qe.h"
#if defined(ENABLE_SPHERICART)
#include "sphericart.hpp" 
#endif

namespace bdft_tests
{

using utils::VALUE_EQUAL;

template<typename T>
void test_harmonics([[maybe_unused]] T m, [[maybe_unused]] T eps)
{

  utils::harmonics<T> ylm;
#if defined(ENABLE_SPHERICART)
  int Lmax = 3;
  int Lmax2 = (Lmax+1)*(Lmax+1);
  auto ylm_sp = sphericart::SphericalHarmonics<T>(Lmax);

  nda::array<T,1> y1(Lmax2), y2(Lmax2);

  for(int i=0; i<10; i++) {
    // random point
    auto r = nda::rand<T>(3);
    r() -= T(0.5);
 
    ylm.spherical_harmonics(Lmax,r,y1);
    ylm_sp.compute_sample(r.data(),3,y2.data(),Lmax2);
    
    utils::ARRAY_EQUAL(y1,y2,m,eps);
  }

#else
  app_log(0,"Compiled without sphericart support, skipping test of harmonincs.");
#endif

}


TEST_CASE("harmonics", "[utilities]")
{
  test_harmonics<double>(1e-8,1e-8);
  test_harmonics<float>(float(1e-6),float(1e-6));
}


// utils::ylmr2_qe against the closed forms of QE's ylmr2 at the axes (values and the signs of l=1 (z, -x, -y) and
// (l=2, m=2, cos) = 1/4 sqrt(15/pi) at x), and utils::ylm_rotation_qe: D^l orthogonal and Y_lm(R r) = sum_m' D(m,m')
// Y_lm'(r) on random directions that are not those used to build D, for the 48 operations of O_h (signed permutation
// matrices, proper and improper), a 60-degree rotation about z and a generic rotation. Negative control: the transpose
// of D (the inverse rotation) fails the relation for the operations that are not their own inverse. The tolerance is
// round-off of the 7x7 solve.
TEST_CASE("ylm_rotation_qe", "[utilities]")
{
  const double pi = std::acos(-1.0);
  // closed forms at the axes
  {
    nda::array<double,2> g = {{0.0,0.0,1.0}, {1.0,0.0,0.0}, {0.0,2.0,0.0}};
    nda::array<double,2> y(3, 16);
    utils::ylmr2_qe(3, g(), y());
    VALUE_EQUAL(y(0,0), std::sqrt(1.0/(4*pi)), 1e-14);
    VALUE_EQUAL(y(0,1), std::sqrt(3.0/(4*pi)), 1e-14);     // z
    VALUE_EQUAL(y(1,2), -std::sqrt(3.0/(4*pi)), 1e-14);    // -x
    VALUE_EQUAL(y(2,3), -std::sqrt(3.0/(4*pi)), 1e-14);    // -y
    VALUE_EQUAL(y(0,4), std::sqrt(5.0/(4*pi)), 1e-14);     // l=2, m=0 at z
    VALUE_EQUAL(y(1,7), 0.25*std::sqrt(15.0/pi), 1e-14);    // l=2, m=2 cos at x
    VALUE_EQUAL(y(0,9), std::sqrt(7.0/(4*pi)), 1e-14);     // l=3, m=0 at z
  }
  // operations
  std::vector<nda::matrix<double>> ops;
  {
    int perm[6][3] = {{0,1,2},{0,2,1},{1,0,2},{1,2,0},{2,0,1},{2,1,0}};
    for (auto& p : perm)
      for (int sgn=0; sgn<8; ++sgn) {
        nda::matrix<double> R = nda::zeros<double>(3,3);
        for (int i=0; i<3; ++i) R(i,p[i]) = ((sgn>>i)&1) ? -1.0 : 1.0;
        ops.push_back(R);
      }
    nda::matrix<double> Rz = nda::zeros<double>(3,3);
    Rz(0,0) = 0.5; Rz(0,1) = -std::sqrt(3.0)/2; Rz(1,0) = std::sqrt(3.0)/2; Rz(1,1) = 0.5; Rz(2,2) = 1.0;
    ops.push_back(Rz);
    // rotation by 0.7 about (1,2,3)/sqrt(14) (Rodrigues)
    double ax[3] = {1/std::sqrt(14.0), 2/std::sqrt(14.0), 3/std::sqrt(14.0)}, c = std::cos(0.7), sn = std::sin(0.7);
    nda::matrix<double> Rg(3,3);
    for (int i=0; i<3; ++i) for (int j=0; j<3; ++j) {
      Rg(i,j) = c*(i==j) + (1-c)*ax[i]*ax[j];
    }
    Rg(0,1) -= sn*ax[2]; Rg(0,2) += sn*ax[1]; Rg(1,0) += sn*ax[2]; Rg(1,2) -= sn*ax[0]; Rg(2,0) -= sn*ax[1]; Rg(2,1) += sn*ax[0];
    ops.push_back(Rg);
  }
  REQUIRE(ops.size() == 50);
  auto r = nda::rand<double>(5,3);
  r() -= 0.5;
  double max_orth = 0.0, max_rel = 0.0, max_ctrl = 0.0;
  for (auto const& R : ops) {
    nda::array<double,2> rr(5,3);
    for (long i=0; i<5; ++i) for (int d=0; d<3; ++d) { rr(i,d) = 0.0; for (int e=0; e<3; ++e) rr(i,d) += R(d,e)*r(i,e); }
    nda::array<double,2> y(5,16), yr(5,16);
    utils::ylmr2_qe(3, r(), y());
    utils::ylmr2_qe(3, rr(), yr());
    for (int l=0; l<4; ++l) {
      auto D = utils::ylm_rotation_qe(l, R);
      int n = 2*l+1;
      double ctrl = 0.0;
      for (int a=0; a<n; ++a) for (int b=0; b<n; ++b) {
        double dot = 0.0;
        for (int m=0; m<n; ++m) dot += D(m,a)*D(m,b);
        max_orth = std::max(max_orth, std::abs(dot - (a==b ? 1.0 : 0.0)));
      }
      for (long i=0; i<5; ++i)
        for (int m=0; m<n; ++m) {
          double v = 0.0, vt = 0.0;
          for (int m2=0; m2<n; ++m2) { v += D(m,m2)*y(i,l*l+m2); vt += D(m2,m)*y(i,l*l+m2); }
          max_rel = std::max(max_rel, std::abs(v - yr(i,l*l+m)));
          ctrl = std::max(ctrl, std::abs(vt - yr(i,l*l+m)));
        }
      max_ctrl = std::max(max_ctrl, ctrl);
    }
  }
  app_log(2, "ylm_rotation_qe: max |D^T D - 1| = {}, max |Y(Rr) - D Y(r)| = {}, control (D^T) = {}", max_orth, max_rel, max_ctrl);
  REQUIRE(max_orth < 1e-12);
  REQUIRE(max_rel < 1e-12);
  REQUIRE(max_ctrl > 1e-2);
}
}
