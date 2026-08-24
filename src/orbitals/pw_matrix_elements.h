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

#ifndef ORBITALS_PW_MATRIX_ELEMENTS_H
#define ORBITALS_PW_MATRIX_ELEMENTS_H

#include <tuple>

#include "configuration.hpp"
#include "nda/nda.hpp"
#include "mean_field/MF.hpp"

namespace orbitals
{

/**
 * Plane-wave matrix elements in the mean-field single-particle basis,
 *
 *   M(G, s, k, i, j) = <psi_{k+q,i}| e^{i(q+G).r} |psi_{k,j}>
 *                    = (1/nnr) sum_r e^{i(G+G0).r}
 *                        conj(u_{s,k',i}(r)) u_{s,k,j}(r),
 *
 * where k' = kpq_map[k] is the stored k-point with k + q = k' + G0 (G0 a
 * reciprocal lattice vector) and u are the cell-periodic orbitals on the
 * mean-field FFT grid.
 *
 * Evaluated directly: the plane wave is built on the real-space grid and the
 * r-sum is a gemm, one per (spin, k, G). This is the right algorithm for the
 * small-nG regime this routine is for — cost is nG * nbnd^2 * nnr against the
 * nbnd^2 * nnr * log(nnr) of an r -> G FFT read-out, so an FFT only pays off
 * beyond nG ~ log2(nnr) (~18 on a 64^3 grid), and it would additionally need
 * the G index to be a position in a truncated grid rather than a Miller triple.
 *
 * Normalization: none beyond the 1/nnr that makes the discrete sum a unit-cell
 * average. The orbitals are normalized to (1/nnr) sum_r |u|^2 = 1 in the cell,
 * so M is exactly the dimensionless Bloch overlap written above; in particular
 * M(G=0, s, k, i, j) = delta_ij at q = 0. No Coulomb weight sqrt(v(q+G)), no
 * 1/sqrt(Omega * nkpts) and no 1/nkpts are applied — every volume, cell-count
 * and Coulomb factor is the caller's to supply. In particular v(q+G) is singular
 * at q+G = 0, the column this routine exists for, so no Coulomb weight can be
 * applied and divided out here.
 *
 * Index order: i is the k+q band, j is the k band, and the conjugate sits on i.
 *
 * Relation to the pair densities the ERI builder computes
 * (methods::cholesky::evaluate_pair_densities, src/methods/ERI/cholesky.icc:125,
 * 128,153,162,166-167,201-202, with v from pots::potential_t, "coulomb"):
 *
 *   P^chol_G(a,b) = sqrt( v(q-G) / (Omega*nkpts) ) * M^{-G}_{ab}(k, q),
 *   v(x) = 4*pi/|x|^2 with no 1/Omega, Gab = -G0, conjugate on a (the k+q band).
 *
 * That relation is **derived by code reading and has not been verified
 * numerically** — neither evaluate_pair_densities nor cholesky::rho_g is
 * reachable from a test (both are private members of cholesky). The arm that
 * closes it without changing that access control, and the natural home for it, is
 * B4 of docs/plan_bse_susceptibility.md: build a chol_reader_t through
 * methods::make_cholesky and check one V_abcd element assembled from M, which
 * goes through the public path.
 *
 * The k-loop is split across ranks and the blocks are gathered to the root, so
 * M is returned full on the MPI root and empty on every other rank. The q+G
 * table is small and is returned on every rank.
 *
 * Memory: three nbnd x nnr complex real-space buffers per rank, plus the nG x nnr
 * table of plane waves and the root-only (nG, nspin, nkpts, nbnd, nbnd) result.
 * At nbnd = 200 and nnr = 64^3 the band buffers are ~3.3 GB/rank, and nG = 170
 * with nkpts = 64 puts ~7 GB on the root. A caller sweeping many G should
 * therefore loop over G columns, a few per call, rather than request a long G
 * list at once.
 *
 * Requires npol == 1 and a full-BZ k-grid (nkpts_ibz == nkpts): a general q
 * breaks the crystal symmetry, so a symmetry-reduced mean field would silently
 * describe a different perturbation.
 *
 * @param mf       mean field providing the orbitals
 * @param q_cryst  (3) wavevector in crystal (fractional) coordinates
 * @param G_mill   (nG, 3) integer Miller indices (integer crystal coordinates)
 *                 of the plane waves, i.e. G = sum_d G_mill(p,d) * recv(d,:).
 *                 Each component must satisfy |G_mill(p,d)| <= mesh(d)/2, the
 *                 Nyquist limit of the FFT grid the orbitals live on.
 * @return (M, qpG_cart): M is (nG, nspin, nkpts, nbnd, nbnd) on the root and
 *         empty elsewhere; qpG_cart is (nG, 3) with q + G in Cartesian
 *         coordinates, ready for v(q+G) = 4*pi/|q+G|^2.
 */
auto pw_matrix_elements(mf::MF& mf,
                        nda::array_const_view<double,1> q_cryst,
                        nda::array_const_view<long,2> G_mill)
  -> std::tuple<nda::array<ComplexType,5>, nda::array<double,2>>;

} // namespace orbitals

#endif
