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

#include <cmath>
#include "utilities/check.hpp"
#include "utilities/ylm_qe.h"

namespace utils
{

void ylmr2_qe(int lmax, nda::array_const_view<double,2> g, nda::array_view<double,2> ylm)
{
  constexpr double eps = 1.0e-9;
  const double pi = std::acos(-1.0);
  const double fpi = 4.0*pi;
  utils::check(lmax >= 0 and lmax <= 3, "ylmr2_qe: lmax must be in [0,3], lmax:{}", lmax);
  utils::check(g.extent(1) == 3 and ylm.extent(0) == g.extent(0) and ylm.extent(1) == (lmax+1)*(lmax+1),
               "ylmr2_qe: size mismatch.");
  long ng = g.extent(0);
  if (lmax == 0) {
    ylm() = std::sqrt(1.0/fpi);
    return;
  }
  // y[lm] with the 1-based lm of ylmr2
  std::vector<double> y((lmax+1)*(lmax+1) + 1);
  for (long ig=0; ig<ng; ++ig) {
    double gmod = std::sqrt(g(ig,0)*g(ig,0) + g(ig,1)*g(ig,1) + g(ig,2)*g(ig,2));
    double cost = (gmod < eps) ? 0.0 : g(ig,2)/gmod;
    double sent = std::sqrt(std::max(0.0, 1.0 - cost*cost));
    y[1] = 1.0;
    y[2] = cost;
    y[4] = -sent/std::sqrt(2.0);
    for (int l=2; l<=lmax; ++l) {
      for (int m=0; m<=l-2; ++m) {
        int lm  = l*l + 1 + 2*m;
        int lm1 = (l-1)*(l-1) + 1 + 2*m;
        int lm2 = (l-2)*(l-2) + 1 + 2*m;
        y[lm] = cost*(2*l-1)/std::sqrt(double(l*l-m*m))*y[lm1]
              - std::sqrt(double((l-1)*(l-1)-m*m))/std::sqrt(double(l*l-m*m))*y[lm2];
      }
      int lm  = l*l + 1 + 2*l;
      int lm1 = l*l + 1 + 2*(l-1);
      int lm2 = (l-1)*(l-1) + 1 + 2*(l-1);
      y[lm1] = cost*std::sqrt(double(2*l-1))*y[lm2];
      y[lm]  = -std::sqrt(double(2*l-1))/std::sqrt(double(2*l))*sent*y[lm2];
    }
    double phi;
    if (g(ig,0) > eps)       phi = std::atan(g(ig,1)/g(ig,0));
    else if (g(ig,0) < -eps) phi = std::atan(g(ig,1)/g(ig,0)) + pi;
    else                     phi = std::copysign(pi/2.0, g(ig,1));
    int lm = 1;
    y[1] /= std::sqrt(fpi);
    for (int l=1; l<=lmax; ++l) {
      double c = std::sqrt(double(2*l+1)/fpi);
      lm += 1;
      y[lm] *= c;
      for (int m=1; m<=l; ++m) {
        lm += 2;
        double base = y[lm];
        y[lm-1] = c*std::sqrt(2.0)*base*std::cos(m*phi);
        y[lm]   = c*std::sqrt(2.0)*base*std::sin(m*phi);
      }
    }
    for (int i=0; i<(lmax+1)*(lmax+1); ++i) ylm(ig,i) = y[i+1];
  }
}

nda::array<double,2> ylm_rotation_qe(int l, nda::matrix_const_view<double> R)
{
  utils::check(l >= 0 and l <= 3, "ylm_rotation_qe: l must be in [0,3], l:{}", l);
  utils::check(R.extent(0) == 3 and R.extent(1) == 3, "ylm_rotation_qe: R must be 3x3.");
  long n = 2*l + 1;
  // fixed generic directions (QE draws random ones; D does not depend on the choice)
  static constexpr double dirs[7][3] = {{ 0.31, -0.27,  0.42}, {-0.18,  0.44,  0.12}, { 0.37,  0.09, -0.33},
                                        {-0.41, -0.23, -0.29}, { 0.05,  0.38,  0.47}, { 0.26, -0.46,  0.19},
                                        {-0.34,  0.15, -0.44}};
  nda::array<double,2> r(n,3), rr(n,3);
  for (long i=0; i<n; ++i) for (int d=0; d<3; ++d) r(i,d) = dirs[i][d];
  for (long i=0; i<n; ++i)
    for (int d=0; d<3; ++d) {
      rr(i,d) = 0.0;
      for (int e=0; e<3; ++e) rr(i,d) += R(d,e)*r(i,e);
    }
  nda::array<double,2> y(n, (l+1)*(l+1)), yr(n, (l+1)*(l+1));
  ylmr2_qe(l, r(), y());
  ylmr2_qe(l, rr(), yr());
  // Ym(m,i) = Y_{l,m}(r_i); Yr(m,i) = Y_{l,m}(R r_i);  Yr = D Ym
  nda::matrix<double> Ym(n,n), Yr(n,n);
  for (long m=0; m<n; ++m)
    for (long i=0; i<n; ++i) {
      Ym(m,i) = y(i, l*l + m);
      Yr(m,i) = yr(i, l*l + m);
    }
  nda::matrix<double> Yinv = nda::inverse(Ym);
  nda::matrix<double> D = Yr * Yinv;
  nda::array<double,2> out(n,n);
  out() = D;
  return out;
}

} // utils
