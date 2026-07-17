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

#ifndef ORBITALS_EPH_VERTEX_H
#define ORBITALS_EPH_VERTEX_H

#include "configuration.hpp"
#include "nda/nda.hpp"
#include "mean_field/MF.hpp"

namespace orbitals
{

/**
 * Local (bare) electron-phonon vertex evaluated directly on the mean-field
 * single-particle basis, in real space (no ISDF/ζ):
 *
 *   g_loc(s, mode, k, m, n) = ⟨φ_{m,k+q} | dV_mode | φ_{n,k}⟩
 *       = (1/nnr) Σ_r e^{i G0·r} conj(u_{m,k+q}(r)) dV_mode(r) u_{n,k}(r),
 *
 * where k+q = k' + G0 (k' = kpq_map[k] in the stored k-grid, G0 a reciprocal
 * lattice vector), and dV_mode(r) is the cell-periodic bare perturbation of
 * mode `mode` sampled on the mean-field FFT grid (C order (ix,iy,iz),
 * index = (ix*NY+iy)*NZ+iz). Works for any MF; the augmented orbitals are
 * treated as ordinary bands.
 *
 * The k-loop is split across ranks and the blocks are gathered to the root, so
 * g_loc is returned full on the MPI root and empty on every other rank (only the
 * root consumes it downstream).
 *
 * @param mf      mean field providing the orbitals
 * @param dV      (nmodes, nnr) cell-periodic perturbation on the FFT grid
 * @param q_cryst (3) phonon wavevector in crystal (fractional) coordinates
 * @return g_loc  (nspin, nmodes, nkpts, nbnd, nbnd) on root; empty elsewhere
 */
template<MEMORY_SPACE MEM = HOST_MEMORY>
auto eph_vertex_local(mf::MF& mf,
                      nda::array_const_view<ComplexType,2> dV,
                      nda::array_const_view<double,1> q_cryst)
  -> nda::array<ComplexType,5>;

} // namespace orbitals

#endif
