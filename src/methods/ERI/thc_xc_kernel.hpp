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

#ifndef COQUI_THC_XC_KERNEL_HPP
#define COQUI_THC_XC_KERNEL_HPP

// Semilocal xc-kernel matrix Vxc^q_{uv} in the THC pivot basis.
//
// CoQuí does not evaluate any xc functional: the kernel coefficients are read
// from an external file (QE's elph.x with write_xc_kernel = .true.) and only
// contracted against the ISDF interpolating vectors zeta^q_u.
//
// Kernel, in QE's own coefficient set (LR_Modules/dgradcorr.f90, nspin = 1):
//
//   A(r)     = dmuxc + dvxc_rr                            (scalar)
//   B_i(r)   = dvxc_sr * grho_i                           (vector)
//   C_ij(r)  = dvxc_s delta_ij + dvxc_ss grho_i grho_j    (symmetric tensor)
//
//   dV_xc = A drho + B.(D drho) - div( B drho + C.(D drho) )
//
// Integrating the divergence by parts (periodic cell) and substituting the
// interpolating vectors for drho gives the matrix built here:
//
//   Vxc^q_{uv} = Int [ zeta_u A conj(zeta_v)
//                    + zeta_u B.conj(D zeta_v)
//                    + (D zeta_u).B conj(zeta_v)
//                    + (D zeta_u).C.conj(D zeta_v) ] dr
//
// Index order and conjugation match the Coulomb matrix built in
// thc::intvec_impl (C_uv = sum_G zeta_u(G) v(G) conj(zeta_v(G))), so Vxc and V
// are directly additive. Both are hermitian.
//
// D is the covariant gradient of the Bloch-factored interpolating vector. The
// THC pair-density convention is psi^{k*} psi^{k-q}, i.e. zeta^q carries
// e^{-iq.r}, and its stored G-space form satisfies
//   zeta^q_u(r) = sum_G zeta_u(q,G) e^{i(G-q).r},
// so D -> i(G - q). This is QE's fft_qgradient (i(q+G) on an e^{+iq.r} array)
// with CoQuí's opposite q convention; see accumulate_Vxc_quv.

#include <string>

#include "configuration.hpp"
#include "grids/g_grids.hpp"
#include "mpi3/communicator.hpp"
#include "nda/nda.hpp"

namespace methods {

namespace mpi3 = boost::mpi3;

/**
 * Semilocal xc-kernel coefficient fields on the external (QE dfftp) FFT mesh,
 * converted to Hartree atomic units on read.
 *
 * All fields are linearized in C order on `mesh` — index (i,j,k) at
 * (i*mesh(1) + j)*mesh(2) + k — i.e. the same convention as CoQuí's FFT grids.
 * The Fortran x-fastest layout of the HDF5 datasets is undone at read time.
 */
struct xc_kernel_fields_t {
  nda::stack_array<long, 3> mesh = {0, 0, 0};
  bool is_gradient = false;         // false for LDA: B and C are empty
  nda::array<double, 1> A;          // (nnr)
  nda::array<double, 2> B;          // (3, nnr), empty for LDA
  nda::array<double, 3> C;          // (3, 3, nnr), empty for LDA

  long nnr() const { return mesh(0) * mesh(1) * mesh(2); }
};

/**
 * Read the xc-kernel coefficient fields written by QE's elph.x
 * (write_xc_kernel = .true.) and assemble A, B, C.
 *
 * Expects file-level attributes `ngrid_dense`, `nr{1,2,3}x_dense` and
 * `xc_is_gradient`, plus rank-1 REAL(DP) datasets `dmuxc` and, for a gradient
 * functional, `dvxc_rr`, `dvxc_sr`, `dvxc_ss`, `dvxc_s`, `grho_{1,2,3}`.
 * Read on the communicator root and broadcast.
 *
 * The dumped fields are Rydberg (each coefficient is one second derivative of
 * E_xc), so every one of A, B, C carries a single Ry -> Ha factor of 1/2.
 */
xc_kernel_fields_t read_xc_kernel_fields(std::string const& fname,
                                        mpi3::communicator& comm);

/**
 * Vxc^q_{uv} for one q-point, from the interpolating vectors in G-space.
 *
 * @param xc         - [INPUT] kernel fields on their own (finer or equal) mesh
 * @param rho_g      - [INPUT] truncated G-grid the interpolating vectors live on
 * @param zeta_uG    - [INPUT] (Np, rho_g.size()) zeta^q_u(G), replicated
 * @param q_cart     - [INPUT] q in Cartesian coordinates (2*pi included)
 * @param volume     - [INPUT] cell volume
 * @param block_size - [INPUT] number of pivots per real-space block
 * @param comm       - [INPUT] communicator the (row-block) work is spread over
 * @return           (Np, Np) hermitian Vxc^q, replicated on every rank
 *
 * zeta_uG must be the *unweighted* interpolating vectors, i.e. taken before the
 * sqrt(v(G)) scaling that turns them into the Coulomb matrix.
 */
nda::array<ComplexType, 2> compute_Vxc_for_q(
    xc_kernel_fields_t const& xc,
    grids::truncated_g_grid const& rho_g,
    nda::array<ComplexType, 2> const& zeta_uG,
    nda::stack_array<double, 3> const& q_cart,
    double volume,
    long block_size,
    mpi3::communicator& comm);

} // namespace methods

#endif // COQUI_THC_XC_KERNEL_HPP
