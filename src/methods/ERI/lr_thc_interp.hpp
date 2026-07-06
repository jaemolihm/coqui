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

#ifndef COQUI_LR_THC_INTERP_HPP
#define COQUI_LR_THC_INTERP_HPP

#include "configuration.hpp"
#include "nda/nda.hpp"
#include "mean_field/MF.hpp"

namespace methods {

/**
 * Compute DeltaX(k, P, m) = Deltapsi_mk(r_P) + grad psi_mk(r_P) . Delta_r_P
 *
 * Term 1 (Deltapsi): Read perturbed wavefunctions from per-mode per-k HDF5 files
 *   and evaluate at interpolating points with Bloch phase e^{i(k+q).r_P}.
 *   File naming: {Deltapsi_prefix}_ik{ik+1}.hdf5, each containing evc (nbnd, 2*npw).
 *
 * Term 2 (gradient): Compute grad psi_mk in plane-wave basis via
 *   i(k+G) * c_G, IFFT, sample at r_P with phase e^{ik.r_P}, dot with delta_r_P.
 *
 * @param mf           Mean-field object (for reading unperturbed wfc and system info)
 * @param Deltapsi_prefix  File path prefix including mode; files are {Deltapsi_prefix}_ik{ik+1}.hdf5
 *                     Example: "temp/Deltapsi_ibc_iq1_mode1" -> temp/Deltapsi_ibc_iq1_mode1_ik1.hdf5, ...
 * @param r_P          (nP,) FFT grid indices of interpolating points (on the fft_grid)
 * @param delta_r_P    (nP, 3) grid point displacements in Cartesian Bohr
 * @param q_vec_cryst  (3,) q-vector in crystal (fractional reciprocal) coordinates
 * @param fft_grid     (3,) FFT grid dimensions. Must match the grid used for r_P
 *                     (e.g. from the THC checkpoint fft_grid dataset).
 * @return             (nspin, nk_ibz, nP, nbnd) complex array on the global root,
 *                     empty (0,0,0,0) on non-root ranks.
 */
nda::array<ComplexType, 4> compute_delta_X(
    mf::MF const& mf,
    std::string const& Deltapsi_prefix,
    nda::array<long, 1> const& r_P,
    nda::array<double, 2> const& delta_r_P,
    nda::array<double, 1> const& q_vec_cryst,
    nda::array<long, 1> const& fft_grid);

/**
 * Compute adjoint/minus-q perturbation of X:
 *   δ^{-q} X(k+q, P, m) = [δ^q]† ψ_{m,k+q}(r_P) + grad ψ_{m,k+q}(r_P) . delta_r_P
 *
 * The entry at index ik stores the quantity evaluated at k_{ik}+q (wave-function
 * side), mirroring the consumer convention in lr_thc_comm (DeltaX_right accessed
 * via kpq_map(k)).
 *
 * Term 1 ([δ^q]†ψ): Read adjoint Sternheimer solutions from per-mode per-k HDF5
 *   files and evaluate at interpolating points with Bloch phase e^{ik.r_P}. The
 *   response [δ^q]†ψ_{k+q} has k (not k+q) Bloch character.
 *   File naming: {Deltapsi_adj_prefix}_ik{ik+1}.hdf5 (same as compute_delta_X),
 *   where ik matches the storage index of the returned array; each file holds
 *   [δ^q]†ψ_{m, k_{ik}+q}.
 *
 * Term 2 (gradient): Compute grad ψ_{m,k+q} in plane-wave basis via
 *   i(k+q+G) * c_G, IFFT, sample at r_P with phase e^{i(k+q).r_P}, dot with
 *   delta_r_P. The assumption δ^{-q}_λ = conj(δ^q_λ) combined with real
 *   delta_r_P means no extra conjugation is needed.
 *
 * Assumes nkpts_ibz == nkpts (full-BZ grid, matching the IBC consumer).
 *
 * @param mf                  Mean-field object
 * @param Deltapsi_adj_prefix Prefix of adjoint-Sternheimer HDF5 files
 * @param r_P                 (nP,) FFT grid indices of interpolating points
 * @param delta_r_P           (nP, 3) grid point displacements in Cartesian Bohr
 *                            (the +q displacement; -q case uses the same real array)
 * @param q_vec_cryst         (3,) +q vector in crystal coordinates
 * @param fft_grid            (3,) FFT grid dimensions
 * @return                    (nspin, nk_ibz, nP, nbnd) complex array on the global
 *                            root, empty (0,0,0,0) on non-root ranks.
 */
nda::array<ComplexType, 4> compute_delta_X_adj(
    mf::MF const& mf,
    std::string const& Deltapsi_adj_prefix,
    nda::array<long, 1> const& r_P,
    nda::array<double, 2> const& delta_r_P,
    nda::array<double, 1> const& q_vec_cryst,
    nda::array<long, 1> const& fft_grid);

} // namespace methods

#endif // COQUI_LR_THC_INTERP_HPP
