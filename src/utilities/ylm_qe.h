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

#ifndef UTILITIES_YLM_QE_H
#define UTILITIES_YLM_QE_H

#include "nda/nda.hpp"

namespace utils
{

/**
 * Real spherical harmonics in the convention of Quantum ESPRESSO's ylmr2 (upflib/ylmr2.f90), same ordering and signs:
 * for each l, m = 0 first, then (cos, sin) pairs for m = 1..l, so index l^2 + 0 is (l,0), l^2 + 2m-1 is (l,m,cos) and
 * l^2 + 2m is (l,m,sin). Port of the recursion of ylmr2 (same eps = 1e-9 for the polar axis and the origin).
 * @param lmax - [INPUT] maximum l
 * @param g    - [INPUT] vectors, shape (ng,3), cartesian
 * @param ylm  - [OUTPUT] shape (ng,(lmax+1)^2)
 */
void ylmr2_qe(int lmax, nda::array_const_view<double,2> g, nda::array_view<double,2> ylm);

/**
 * Rotation matrix of the real spherical harmonics of angular momentum l in the QE convention, as computed by
 * PW/src/d_matrix.f90: Y_{lm}(R r) = sum_{m'} D(m,m') Y_{lm'}(r), with the (2l+1) x (2l+1) real matrix D, for the
 * cartesian orthogonal matrix R (proper or improper). D is obtained by solving this relation on 2l+1 fixed
 * directions, so it follows the ylmr2_qe convention by construction.
 */
nda::array<double,2> ylm_rotation_qe(int l, nda::matrix_const_view<double> R);

} // utils

#endif
