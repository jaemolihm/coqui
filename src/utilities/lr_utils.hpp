/**
 * ==========================================================================
 * CoQuí: Correlated Quantum ínterface
 *
 * Copyright (c) 2022-2025 Simons Foundation & The CoQuí developer team
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


#ifndef UTILITIES_LR_UTILS_HPP
#define UTILITIES_LR_UTILS_HPP

#include "IO/app_loggers.h"
#include "utilities/check.hpp"
#include "nda/nda.hpp"

namespace utils {

/**
 * @brief Compute k+q mapping for linear response calculations
 *
 * Given a k-point grid and a perturbation wavevector q, compute the mapping
 * kpq_map[ik] = ik' where k[ik] + q = k[ik'] (mod G).
 *
 * @param kpts_crys   - [INPUT] k-points in crystal coordinates (nkpts, 3)
 * @param q_vec       - [INPUT] perturbation wavevector in crystal coordinates (3,)
 * @param kpq_map     - [OUTPUT] k → k+q index mapping (nkpts,)
 * @param threshold   - [INPUT] tolerance for k-point matching
 */
inline void calculate_kpq_map(nda::ArrayOfRank<2> auto const& kpts_crys,
                              nda::ArrayOfRank<1> auto const& q_vec,
                              nda::ArrayOfRank<1> auto&& kpq_map,
                              double threshold = 1e-6) {
  long nkpts = kpts_crys.shape(0);
  utils::check(kpts_crys.shape(1) == 3, "calculate_kpq_map: kpts_crys.shape(1) != 3");
  utils::check(q_vec.shape(0) == 3, "calculate_kpq_map: q_vec.shape(0) != 3");
  utils::check(kpq_map.shape(0) == nkpts, "calculate_kpq_map: kpq_map size mismatch");

  kpq_map() = -1;

  for (long ik = 0; ik < nkpts; ++ik) {
    // k + q in crystal coordinates
    double kpq0 = kpts_crys(ik, 0) + q_vec(0);
    double kpq1 = kpts_crys(ik, 1) + q_vec(1);
    double kpq2 = kpts_crys(ik, 2) + q_vec(2);

    // Find k' such that k' = k + q (mod G)
    for (long ikp = 0; ikp < nkpts; ++ikp) {
      double d0 = std::abs(kpts_crys(ikp, 0) - kpq0);
      double d1 = std::abs(kpts_crys(ikp, 1) - kpq1);
      double d2 = std::abs(kpts_crys(ikp, 2) - kpq2);

      // Apply periodic boundary conditions
      d0 -= std::floor(d0);
      d1 -= std::floor(d1);
      d2 -= std::floor(d2);
      d0 -= std::round(d0);
      d1 -= std::round(d1);
      d2 -= std::round(d2);

      if (d0 * d0 + d1 * d1 + d2 * d2 < threshold * threshold) {
        utils::check(kpq_map(ik) == -1,
                     "calculate_kpq_map: Found duplicate k+q mapping for ik={}.", ik);
        kpq_map(ik) = ikp;
        break;
      }
    }

    utils::check(kpq_map(ik) >= 0,
                 "calculate_kpq_map: Could not find k+q for ik={}, k=({}, {}, {}), q=({}, {}, {})",
                 ik, kpts_crys(ik, 0), kpts_crys(ik, 1), kpts_crys(ik, 2),
                 q_vec(0), q_vec(1), q_vec(2));
  }
}

/**
 * @brief Check if q is commensurate with the k-point grid
 *
 * @param kpts_crys   - [INPUT] k-points in crystal coordinates (nkpts, 3)
 * @param q_vec       - [INPUT] perturbation wavevector in crystal coordinates (3,)
 * @param threshold   - [INPUT] tolerance for k-point matching
 * @return true if q is commensurate, false otherwise
 */
inline bool is_q_commensurate(nda::ArrayOfRank<2> auto const& kpts_crys,
                              nda::ArrayOfRank<1> auto const& q_vec,
                              double threshold = 1e-6) {
  long nkpts = kpts_crys.shape(0);

  for (long ik = 0; ik < nkpts; ++ik) {
    double kpq0 = kpts_crys(ik, 0) + q_vec(0);
    double kpq1 = kpts_crys(ik, 1) + q_vec(1);
    double kpq2 = kpts_crys(ik, 2) + q_vec(2);

    bool found = false;
    for (long ikp = 0; ikp < nkpts; ++ikp) {
      double d0 = std::abs(kpts_crys(ikp, 0) - kpq0);
      double d1 = std::abs(kpts_crys(ikp, 1) - kpq1);
      double d2 = std::abs(kpts_crys(ikp, 2) - kpq2);

      d0 -= std::floor(d0);
      d1 -= std::floor(d1);
      d2 -= std::floor(d2);
      d0 -= std::round(d0);
      d1 -= std::round(d1);
      d2 -= std::round(d2);

      if (d0 * d0 + d1 * d1 + d2 * d2 < threshold * threshold) {
        found = true;
        break;
      }
    }
    if (!found) return false;
  }
  return true;
}

/**
 * @brief Check if q is approximately zero (Gamma point)
 *
 * @param q_vec       - [INPUT] perturbation wavevector in crystal coordinates (3,)
 * @param threshold   - [INPUT] tolerance
 * @return true if q is approximately zero, false otherwise
 */
inline bool is_q_gamma(nda::ArrayOfRank<1> auto const& q_vec, double threshold = 1e-6) {
  double d0 = std::abs(q_vec(0));
  double d1 = std::abs(q_vec(1));
  double d2 = std::abs(q_vec(2));

  // Apply periodic boundary conditions
  d0 -= std::floor(d0);
  d1 -= std::floor(d1);
  d2 -= std::floor(d2);
  d0 -= std::round(d0);
  d1 -= std::round(d1);
  d2 -= std::round(d2);

  return d0 * d0 + d1 * d1 + d2 * d2 < threshold * threshold;
}

} // namespace utils

#endif // UTILITIES_LR_UTILS_HPP
