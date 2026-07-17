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

#ifndef ORBITALS_ORBITAL_AUGMENTER_H
#define ORBITALS_ORBITAL_AUGMENTER_H

#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "configuration.hpp"
#include "nda/nda.hpp"
#include "mpi3/communicator.hpp"
#include "numerics/distributed_array/nda.hpp"
#include "mean_field/MF.hpp"
#include "IO/ptree/ptree_utilities.hpp"

namespace orbitals
{

/**
 * Strategy that turns a block of base orbitals into raw (non-orthonormal)
 * augmentation states, expressed on the SAME 'w' truncated-G grid as the base
 * orbitals. This is the only augmentation-specific step: everything downstream
 * (projection against the originals, SVD truncation, uniformization) is generic
 * and lives in add_augmentation / orthonormalize_augmentation.
 */
struct orbital_augmenter_t
{
  virtual ~orbital_augmenter_t() = default;

  // Number of raw states produced per augmented band (3 for momentum, nmodes for dpsi).
  virtual int n_raw_per_band() const = 0;

  // Provenance tag stored in the resulting mean-field file.
  virtual std::string type() const = 0;

  // For a given (ispin, ik): given the local G-slice psi_base (nbnd_aug, ng_loc)
  // starting at global G index g0, fill raw_out (n_raw_per_band()*nbnd_aug,
  // ng_loc). Raw states of band b for channel c are placed at row c*nbnd_aug+b.
  // The G distribution gives each rank a contiguous slice [g0, g0+ng_loc).
  virtual void generate_raw(int ispin, int ik, long g0,
                            nda::array_const_view<ComplexType,2> psi_base,
                            nda::array_view<ComplexType,2> raw_out) const = 0;
};

/**
 * Momentum augmentation: raw states are p̂_α ψ_b = (k+G)_α ψ_b(G) for the three
 * Cartesian directions α, evaluated on the wavefunction G-grid (same support as
 * ψ_b, so no regridding). k and G must share the Cartesian convention taken
 * from the mean-field object. The raw states carry units of 1/bohr and are
 * scaled by dtau_step (bohr) in add_augmentation, so their overall scale
 * matters: it sets the singular values used for truncation and the THC fit
 * weights.
 */
struct momentum_augmenter : orbital_augmenter_t
{
  momentum_augmenter(mf::MF& mf);

  int n_raw_per_band() const override { return 3; }
  std::string type() const override { return "momentum"; }

  void generate_raw(int ispin, int ik, long g0,
                    nda::array_const_view<ComplexType,2> psi_base,
                    nda::array_view<ComplexType,2> raw_out) const override;

private:
  nda::array<int,2> _miller;               // (ngm, 3) Miller indices of the wfc grid
  nda::stack_array<double,3,3> _recv;      // reciprocal vectors, recv(i,:) = i-th vector (Cartesian)
  nda::array<double,2> _kcart;             // (nkpts_ibz, 3) IBZ k-points (Cartesian)
};

// Build an augmenter from an input block. `type` selects the transform.
std::shared_ptr<orbital_augmenter_t> make_augmenter(mf::MF& mf, ptree const& pt);

// Augment the mean-field basis: keep the original nbnd bands and append the
// orthonormalized augmentation states produced by `augmenter` from the first
// nbnd_aug bands, truncated with singular-value cutoff epstol (uniform band
// count across k). The raw states are derivatives dψ/dτ with respect to an
// atomic displacement (units 1/bohr) and are scaled by dtau_step (bohr) before
// truncation, so epstol thresholds the dimensionless residual amplitude
// s = ||dtau_step·(dψ)⊥||. Returns a new bdft mean-field flagged as augmented,
// carrying per-band THC fit weights min(s,1) (1 for the originals).
// nbnd_aug = -1 transforms all bands; nbnd_aug = 0 adds no states and returns
// the original orbitals in the augmented bdft format (baseline).
template<MEMORY_SPACE MEM = HOST_MEMORY>
mf::MF add_augmentation(mf::MF& mf, std::string fn,
                        std::shared_ptr<orbital_augmenter_t> augmenter,
                        long nbnd_aug, double epstol, double dtau_step = 0.1);

// Augment the mean-field basis with DFPT response wavefunctions (δψ) read from
// per-(iq,mode,ik) HDF5 files. The base mf carries N = mf.nbnd() h5/nscf bands;
// nbnd_mf (=M, <=0 or >N means keep all N) selects how many are kept as the
// originals of the augmented system. For each requested phonon iq (q read from
// the matching elph_bare.iq{iq}.h5), each mode, and each source k, the response
// δψ(n,k) — which carries crystal momentum k+q — is placed on the 'w' grid at
// the deposit index j = kpq_map(ik) (with the umklapp G0 shift), for the first
// nbnd_aug (=R) bands. QE orthogonalizes δψ against all N nscf bands, so the
// contribution of the bands m ∈ [M, N) above the kept originals is added back:
//   δψ(n,k) += Σ_{m=M}^{N-1} ψ(m,k+q) g_scr(mode,m,n) reg(e(n,k) - e(m,k+q)),
// making the state the response orthogonal to only the M kept bands. reg is a
// sharp, continuous 1/x cutoff (1/x for |x|>smearing, x/smearing² otherwise);
// with it the augmentation is independent of N (any base dataset with enough
// bands gives the same result). All energies/vertices are converted Ry→Ha at
// read. mode_list selects which modes contribute (1-based indices into the
// nmodes modes of each iq; empty = all modes 1..nmodes). The
// R·|mode_list|·nq raw states per k are scaled by dtau_step (bohr, they
// are δψ/δτ responses with units 1/bohr), orthonormalized against the M
// originals, SVD-truncated at the dimensionless singular-value cutoff epstol,
// and uniformized (shared tail with add_augmentation). Requires npol==1 and a
// full-BZ k-grid (nkpts==nkpts_ibz).
template<MEMORY_SPACE MEM = HOST_MEMORY>
mf::MF add_augmentation_dpsi(mf::MF& mf, std::string fn,
                             std::string deltapsi_dir, std::string elph_dir,
                             std::vector<long> const& iq_list, long nmodes_in,
                             std::vector<long> const& mode_list,
                             long nbnd_aug, long nbnd_mf,
                             double smearing, double epstol,
                             double dtau_step = 0.1);

/**
 * Orthonormalize an augmentation block against a fixed set of originals.
 * `psi_orig` (nspin, nkpts_ibz, nbnd, ngm) are the originals, assumed already
 * orthonormal and kept unchanged. `raw_aug` (nspin, nkpts_ibz, n_raw, ngm) are
 * the raw (non-orthonormal) augmentation states, already scaled by dtau_step.
 * For every k independently we
 *   1. project raw_aug onto the orthogonal complement of psi_orig,
 *   2. diagonalize the residual overlap (eigenvalues λ = s², with s the
 *      singular value of the residual block) and keep the vectors with
 *      s = √λ >= `epstol`.
 * To keep a uniform band count, n_aug_max = max_k (#kept) is computed and the
 * top n_aug_max vectors are retained at every k. Singular values are logged at
 * verbosity >= 2 (k=0) and >= 3 (all k). Returns
 *   - the assembled orthonormal basis [originals | augmentation],
 *     (nspin, nkpts_ibz, nbnd + n_aug_max, ngm), distributed as psi_orig;
 *   - per-band THC fit weights (nspin, nkpts_ibz, nbnd + n_aug_max), replicated
 *     on every rank: 1 for the originals, min(s, 1) for the augmentation states.
 * `raw_aug` is overwritten (holds the projected residual on return).
 */
template<MEMORY_SPACE MEM = HOST_MEMORY, utils::Communicator comm_t>
auto orthonormalize_augmentation(
                    memory::darray_t<memory::array<MEM, ComplexType, 4>, comm_t>& psi_orig,
                    memory::darray_t<memory::array<MEM, ComplexType, 4>, comm_t>& raw_aug,
                    double epstol)
  -> std::tuple<memory::darray_t<memory::array<MEM, ComplexType, 4>, comm_t>,
                nda::array<double, 3>>;

/**
 * Kinetic-energy diagonal ⟨φ_i|T|φ_i⟩ = Σ_G ½|k+G|²|φ_i(G)|² on the 'w' grid,
 * (nspin, nkpts_ibz, nbnd), stored as the augmented (non-eigen) basis's
 * eigenvalues. Provides a finite estimate of the single-particle energy scale
 * (e.g. to bound the imaginary-axis frequency grid / wmax). The full ⟨φ|H0|φ⟩
 * cannot be formed here (the base MF's pseudopotential is sized for the original
 * band count); with h0_source="compute" the SCF still rebuilds the full H0.
 */
template<MEMORY_SPACE MEM = HOST_MEMORY, utils::Communicator comm_t>
auto rayleigh_eigvals(mf::MF& mf,
                    memory::darray_t<memory::array<MEM, ComplexType, 4>, comm_t>& psi)
  -> nda::array<double, 3>;

} // orbitals

#endif // ORBITALS_ORBITAL_AUGMENTER_H
