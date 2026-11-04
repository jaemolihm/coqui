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

#ifndef COQUI_DIIS_COEFS_HPP
#define COQUI_DIIS_COEFS_HPP

#include <algorithm>
#include <cmath>
#include <numeric>
#include <optional>

#include "configuration.hpp"
#include "nda/nda.hpp"
#include "nda/blas.hpp"
#include "numerics/nda_functions.hpp"  // nda::blas 3-argument gemm/gemv
#include "nda/linalg/eigenelements.hpp"
#include "IO/app_loggers.h"
#include "utilities/check.hpp"

namespace iter_scf {

/**
 * @brief Solve the DIIS constraint system for the extrapolation coefficients.
 *
 * Given the residual-overlap matrix B, return the real coefficient vector
 * c = B^{-1} 1 / (1^T B^{-1} 1). B is eigendecomposed (real part; the coefs
 * are constrained real) and inverted via a thresholded pseudoinverse, exactly
 * as in the original Pokhilko/Yeh/Zgid scheme. The disk (VSpace/diis_alg) and the
 * in-memory (in_memory_diis) paths compute their coefficients from this one source.
 *
 * @param m_B       residual-overlap matrix (n x n).
 * @param log_level verbosity level of the condition-number message.
 * @return          real extrapolation coefficients (length n), summing to 1; nullopt if
 *                  |sum_i x_i| * max_i B_ii < 1e-14 (x = B^+ 1, unnormalized), or the sum is
 *                  not finite. The product is scale-free; it is 0 when no eigenvalue of the
 *                  scaled B survives the pseudoinverse cut (B = 0 for vanishing residuals).
 */
inline std::optional<nda::array<double, 1>> compute_diis_coefs_c1(const nda::matrix<ComplexType>& m_B,
                                                                  int log_level = 2) {
  auto B = nda::make_regular(nda::real(m_B)); // only real part is needed due to constraint to real coefs
  const long n = B.shape()[0];

  // Solve with the Jacobi-scaled Bt = D^(-1/2) B D^(-1/2), D = diag(B), so that
  // Bt_ii = 1 and Bt_ij = cos<(r_i, r_j). Exact change of variables, but it is
  // what makes the cut below meaningful: B_ii = ||r_i||^2 and DIIS residual
  // norms decay geometrically, so cond(B) is dominated by that decay while
  // cond(Bt) sees only genuine linear dependence. The scaling is undone on x.
  nda::array<double, 1> d(n), bb(n);
  for (long i = 0; i < n; ++i) {
    d(i)  = (B(i, i) > 0.0) ? std::sqrt(B(i, i)) : 1.0;
    bb(i) = 1.0 / d(i);  // rhs D^(-1/2) 1
  }
  for (long i = 0; i < n; ++i)
    for (long j = 0; j < n; ++j) B(i, j) /= d(i) * d(j);

  auto [eig, evecs] = nda::linalg::eigenelements(B);
  auto evecs_tr = nda::make_regular(nda::transpose(evecs));

  nda::matrix<double> Binv(n, n); // Inverse or pseudoinverse
  nda::matrix<double> eig_inv(n, n);
  nda::matrix<double> I(n, n);
  Binv() = 0;
  eig_inv() = 0;

  auto eig_abs = nda::map([](double x) { return std::abs(x); })(eig);
  double eig_max = nda::max_element(eig);
  const double eig_min_abs = nda::min_element(eig_abs);

  const double eig_thresh = 1E-14;

  if (eig_min_abs > 0.0)
    app_log(log_level, "DIIS: Condition number of Jacobi-scaled B: {}",
            nda::max_element(eig_abs) / eig_min_abs);
  else
    app_log(log_level, "DIIS: Jacobi-scaled B is singular (smallest |eigenvalue| = 0)");

  // Pseudoinverse: only keep positive eigenvalues above eig_thresh*eig_max.
  for (auto i : nda::range(0, eig.size())) {
    if (eig(i) > eig_thresh * eig_max) {
      eig_inv(i, i) = 1.0 / (eig(i));
    }
  }

  nda::blas::gemm(evecs, eig_inv, I);
  nda::blas::gemm(I, evecs_tr, Binv);

  nda::array<double, 1> x(n);
  nda::blas::gemv(1.0, Binv, bb, 0.0, x);
  for (long i = 0; i < n; ++i) x(i) /= d(i);

  // x scales as 1/||r||^2 and max_i B_ii as ||r||^2, so the product does not depend on the
  // residual scale: it is n for n orthogonal residuals of equal norm, and 0 for B = 0.
  double sum = std::accumulate(x.begin(), x.end(), 0.0);
  double Bmax = 0.0;
  for (long i = 0; i < n; ++i) Bmax = std::max(Bmax, d(i) * d(i));
  if (not std::isfinite(sum) or std::abs(sum) * Bmax < 1e-14) return std::nullopt;
  return nda::make_regular(x / sum);
}

} // namespace iter_scf

#endif // COQUI_DIIS_COEFS_HPP
