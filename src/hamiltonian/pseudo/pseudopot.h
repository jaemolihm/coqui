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


#ifndef HAMILTONIAN_PSEUDO_NCPP_H
#define HAMILTONIAN_PSEUDO_NCPP_H

#include <iostream>
#include <memory>
#include <string>

#include "configuration.hpp"
#include "hamiltonian/pseudo/pseudopot_type.hpp"
#include "IO/app_loggers.h"
#include "utilities/check.hpp"
#include "utilities/mpi_context.h"
#include "mpi3/environment.hpp"
#include "mpi3/communicator.hpp"
#include "h5/h5.hpp"
#include "nda/nda.hpp"
#include "nda/tensor.hpp"
#include "numerics/shared_array/nda.hpp"
#include "numerics/distributed_array/nda.hpp"
#include "mean_field/mf_source.hpp"
#include "utilities/symmetry.hpp"

namespace hamilt 
{

/**
 * @class pseudopot
 * @brief Handler for the pseudopotential of a given mean-field object
 *
 * This class is responsible for computing and storing the pseudopotential for a
 * specified physical system. The system, including `nbnd`, `nkpt`, Brillouin zone info,
 * and "pseudopotential", are defined upon construction through a mean-field handler.
 *
 * Contributions of the pseudopotential can be evaluated by calling
 *
 *     pseudopot.add_Vpp(..., hpsi, Hij);
 *
 * where
 *   1. Contributions from the local potentials are added to wavefunctions "hpsi"
 *   2. Contributions from the non-local potentials are added to the Hamiltonian
 *      in second quantization "Hij"
 *
 * In addition, this class also handle the Hartree potential for a given density matrix `nij`
 *
 *     pseudopot.add_Hartree(..., nij, ..., hpsi);
 *
 * in which the Hartree potential will be added to `hpsi`.
 *
 * @tparam MF_t - Type parameter for the mean-field object
 */
class pseudopot
{
  template<typename Arr>
  using sarray_t = typename math::shm::shared_array<Arr>;

  public:

  using mpi_t = utils::mpi_context_t<mpi3::communicator,mpi3::shared_communicator>;

  template<typename MF_t>
  pseudopot(MF_t &mf, std::string const filename = "");

  ~pseudopot() {}

  pseudopot(pseudopot const&) = default;
  pseudopot(pseudopot &&) = default;
  pseudopot& operator=(pseudopot const&) = default;
  pseudopot& operator=(pseudopot &&) = default;

  pp_type_e pp_type() const { return ptype; }

  void save(std::string fname, bool append = true);
  void save(h5::group& grp);

  std::shared_ptr<mpi_t> get_mpi_context() { return mpi; }

  /**
   * Add the contributions of norm-conserving pseudopotentials:
   *
   *   1. Contributions from the local potentials are added to wavefunctions "hpsi"
   *   2. Contributions from the non-local potentials are added to the Hamiltonian
   *      in second quantization "Hij"
   *
   * @param k_range - [input] Range of k-point indices
   * @param b_range - [input] Range of orbital indices "b"
   * @param psi     - [input] Single-particle basis
   * @param hpsi    - [input] \hat{h} * psi where h is an arbitrary local operator
   *                  [output] (\hat{h} + vloc) * psi
   * @param Hij     - [input] Matrix elements of an arbitrary non-local operator H_nl
   *                  [output] Matrix elements of H_nl + Vpp_nl,
   *                           where Vpp_nl is the non-local part of the pseudopotential
   */
  void add_Vpp(boost::mpi3::communicator& comm, nda::range k_range, nda::range b_range,
               math::nda::DistributedArrayOfRank<4> auto const& psi,
               math::nda::DistributedArrayOfRank<4> auto & hpsi,
               math::nda::DistributedArrayOfRank<4> auto & Hij);

  void add_Vpp(boost::mpi3::communicator& comm, nda::range k_range, nda::range b_range,
               nda::ArrayOfRank<3> auto const& nii,
               math::nda::DistributedArrayOfRank<4> auto const& psi,
               math::nda::DistributedArrayOfRank<4> auto & hpsi,
               math::nda::DistributedArrayOfRank<4> auto & Hij);

  void add_Vpp(boost::mpi3::communicator& comm, nda::range k_range, nda::range b_range,
               nda::ArrayOfRank<4> auto const& nij,
               math::nda::DistributedArrayOfRank<4> auto const& psi,
               math::nda::DistributedArrayOfRank<4> auto & hpsi,
               math::nda::DistributedArrayOfRank<4> auto & Hij);

  /**
   * Add the contributions of the Hartree potential to the wavefunctions "hpsi"
   *
   * @param k_range - [input] Range of k-point indices
   * @param nii     - [input] Diagonal density matrix (s, k, a)
   * @param psi     - [input] Single-particle basis (s, k, a, g), where g lives in the "wavefunction" grid
   * @param hpsi    - [input] \hat{H_loc} * psi, where H_loc is an arbitrary local operator
   *                  [output] (\hat{H_loc} + V_H) * psi,
   *                           where Vpp_loc is the local part of the pseudopotential
   */
  void add_Hartree(nda::range k_range,
                   nda::ArrayOfRank<3> auto const& nii,
                   math::nda::DistributedArrayOfRank<4> auto const& psi,
                   math::nda::DistributedArrayOfRank<4> auto & hpsi,
                   bool symmetrize=false);

  /**
   * Add the contributions of the Hartree potential to the wavefunctions "hpsi"
   *
   * @param k_range - [input] Range of k-point indices
   * @param nij     - [input] Density matrix (s, k, a, b)
   * @param psi     - [input] Single-particle basis (s, k, a, g), where g lives in the "wavefunction" grid
   * @param hpsi    - [input] \hat{H_loc} * psi, where H_loc is an arbitrary local operator
   *                  [output] (\hat{H_loc} + V_H) * psi,
   *                           where Vpp_loc is the local part of the pseudopotential
   */
  void add_Hartree(nda::range k_range,
                   nda::ArrayOfRank<4> auto const& nij,
                   math::nda::DistributedArrayOfRank<4> auto const& psi,
                   math::nda::DistributedArrayOfRank<4> auto & hpsi,
                   bool symmetrize=false);

  /**
   * Nonlocal projector overlaps for the electron-phonon vertex, evaluated on
   * the mean-field single-particle basis. Uses the same projectors β(G)
   * ("vkb") and D-matrix that build the nonlocal pseudopotential for H0.
   *
   *   P(0,s,k,μ,a) = ⟨β_μ,k | φ_a,k⟩
   *   P(d,s,k,μ,a) = ⟨β_μ,k | (k+G)_d φ_a,k⟩,   d = 1,2,3 → Cartesian x,y,z
   *
   * (k+G) is in Cartesian 1/bohr, so P(1..3) are the momentum-operator
   * projector overlaps ⟨β|p̂_d φ⟩. Together with Dion() and the projector→atom
   * maps they factorize the bare nonlocal e-ph vertex. Host only; h5 input for MF.
   *
   * When P2 != nullptr, the six independent second-derivative overlaps are also
   * returned, needed for the nonlocal part of the q=0 second-order vertex:
   *
   *   P2(p,s,k,μ,a) = ⟨β_μ,k | (k+G)_i (k+G)_j φ_a,k⟩,
   *
   * with pair index p = 0..5 → (i,j) = (x,x),(y,y),(z,z),(x,y),(x,z),(y,z)
   * (symmetric, Cartesian 1/bohr²).
   */
  template<typename MF_t>
  void eph_projector_overlaps(MF_t& mf, nda::array<ComplexType,5>& P,
                              nda::array<ComplexType,5>* P2 = nullptr);

  /**
   * Bare nonlocal electron-phonon vertex in the mean-field band basis:
   *
   *   g_nl(s, mode, k, m, n) = ⟨φ_{m,k+q}| dV^nl_mode |φ_{n,k}⟩,
   *   mode = 3·κ + d   (atom κ, Cartesian direction d = x,y,z), nmodes = 3·nat.
   *
   * dV^nl is the derivative of the separable Kleinman-Bylander potential
   *   V^nl = Σ_μ |β_μ⟩ D_μ ⟨β_μ|   (β_μ: projector of atom κ; D_μ: strength, Dion)
   * w.r.t. the displacement of atom κ along d. The projector rides rigidly on the
   * atom, β_μ(r) = β(r − τ_κ), so ∂_{τ_κd} β_μ = −∂_{r_d} β_μ = i p̂_d β_μ with the
   * momentum operator p̂_d = −i∂_{r_d}; hence ⟨∂β_μ|φ⟩ = i⟨β_μ|p̂_d|φ⟩, the
   * momentum-weighted projector overlap (for the Bloch basis p̂_d φ = (k+G)_d φ).
   * The derivative hits the bra projector and the ket:
   *   g_nl ∝ ⟨φ_{k+q}|β_μ⟩ D_μ ⟨∂β_μ|φ_k⟩ + ⟨φ_{k+q}|∂β_μ⟩ D_μ ⟨β_μ|φ_k⟩
   *        = i D_μ [ ⟨φ_{k+q}|β_μ⟩⟨β_μ|p̂_d|φ_k⟩ − ⟨φ_{k+q}|p̂_d|β_μ⟩⟨β_μ|φ_k⟩ ].
   *
   * We fold the two terms into one bilinear conj(B_{k+q})·D_thc·B_k (a single gemm
   * per mode) via a ± trick:
   *   B(k,m,μ') = (⟨β_ip|φ_{m,k}⟩ ± i⟨β_ip|p̂_d|φ_{m,k}⟩)/√2,
   * the '+' block sits at μ'=ip+nproj·d, the '−' block at μ'=ip+nproj·(d+3).
   * D_thc is block-diagonal in
   * μ', carrying the diagonal KB strength +D_ii on the '+' block and −D_ii on the
   * '−' block; that sign flip is what makes conj(B_{k+q})·D_thc·B_k reproduce the
   * antisymmetric (bra − ket) derivative combination above. (Momentum convention
   * alphap_QE = i·CoQuí; only diagonal Dion, e.g. ONCV, is supported.)
   *
   * Spin-independent (replicated over spin). The k-loop is split across ranks
   * and gathered to the root: returned full on the root and empty on every other
   * rank. Requires npol=1 and a full-BZ k-grid (nkpts == nkpts_ibz).
   */
  template<typename MF_t>
  auto eph_vertex_nonlocal(MF_t& mf, nda::array_const_view<double,1> q_cryst)
    -> nda::array<ComplexType,5>;

  /**
   * Bare nonlocal part of the q=0 second-order electron-phonon vertex, stored
   * compactly as (nspin, nat, 3, 3, nk, nb, nb) with dims (atom, cart_i, cart_j):
   *
   *   g2_nl(s, κ, α, β, k, m, n) = ⟨φ_{m,k}| ∂²V^nl/∂τ_{κα}∂τ_{κβ} |φ_{n,k}⟩,
   *   i.e. mode1 = 3·κ+α, mode2 = 3·κ+β. Diagonal in the atom κ, so the off-atom
   *   mode1/mode2 blocks are exactly zero and are not stored.
   *
   * Second derivative of the separable KB potential V^nl = Σ_μ |β_μ⟩ D_μ ⟨β_μ|
   * w.r.t. two displacements of the same atom. With the projector→(atom,ih) maps
   * and diagonal D_μ (Dion, ONCV) it factorizes (per projector μ of atom κ) as
   *   Σ_μ D_μ [ −⟨P2_αβ|_m ⟨P0|_n − ⟨P0|_m ⟨P2_αβ|_n
   *             + ⟨P1_α|_m ⟨P1_β|_n + ⟨P1_β|_m ⟨P1_α|_n ]
   * where P0 = ⟨β|φ⟩, P1_d = ⟨β|(k+G)_d φ⟩, P2_ij = ⟨β|(k+G)_i(k+G)_j φ⟩ come from
   * eph_projector_overlaps. Hartree, replicated over spin. The k-loop is split
   * across ranks and gathered to the root: returned full on the root and empty on
   * every other rank. npol=1, full-BZ k-grid.
   */
  template<typename MF_t>
  auto eph_vertex_nonlocal_d2(MF_t& mf)
    -> nda::array<ComplexType,7>;

  /**
   * Ionic local perturbation dV^loc_mode(r) on the dense FFT grid,
   * for the bare electron-phonon vertex at phonon wavevector q_cryst:
   *
   *   dV(G) = -i (q+G)_d · vloc_sp(|q+G|) · e^{-i(q+G)·τ_κ},  mode = 3·κ + d,
   *
   * with vloc_sp(|q+G|) reconstructed from the per-species radial local
   * pseudopotential (h5 group "vloc_radial", written by pw2coqui) via the
   * erf-compensated Simpson radial FT (QE vloc_of_g/setlocq). Returns
   * (nmodes, nnr) in **Hartree** (the UPF vloc is Rydberg; converted here),
   * C order (ix*NY+iy)*NZ+iz — the layout eph_vertex_local expects.
   */
  template<typename MF_t>
  auto build_dvloc_ion(MF_t& mf, nda::array_const_view<double,1> q_cryst)
    -> nda::array<ComplexType,2>;

  /**
   * Local part of the q=0 second-order ionic perturbation on the dense FFT grid,
   * for the bare second-order electron-phonon vertex:
   *
   *   d²V(G) = −G_i · G_j · vloc_sp(|G|) · e^{−iG·τ_κ},
   *
   * one field per (atom κ, symmetric Cartesian pair p), packed as
   * mode = 6·κ + p with p = 0..5 → (i,j) = (x,x),(y,y),(z,z),(x,y),(x,z),(y,z).
   * Uses the same per-species radial local pseudopotential (h5 "vloc_radial")
   * and radial FT (vloc_of_g) as build_dvloc_ion. Returns (6·nat, nnr) in
   * Hartree, C order (ix*NY+iy)*NZ+iz — the layout eph_vertex_local expects.
   */
  template<typename MF_t>
  auto build_d2vloc_ion(MF_t& mf)
    -> nda::array<ComplexType,2>;

  // --- accessors for assembling the nonlocal e-ph vertex ---
  // number of projectors (per polarization)
  long n_proj() const { return Pskna.shape()[2]/npol; }
  // projectors per species, species index of each atom, first-projector offset
  nda::array<int,1> const& proj_per_species() const { return nh; }
  nda::array<int,1> const& species_of_atom() const { return ityp; }
  nda::array<int,1> const& proj_offset() const { return ofs; }
  int num_polarizations() const { return npol; }
  // D-matrix (dion, in Hartree) as a host copy: (nsp, nhm*npol, nhm*npol)
  nda::array<ComplexType,3> Dion() const {
    nda::array<ComplexType,3> D(Dnn.local().shape());
    D() = Dnn.local();
    return D;
  }

  private:

  // mpi communicators
  std::shared_ptr<mpi_t> mpi;

  // pseudo type, default to ncpp and update in constructor
  pp_type_e ptype = pp_ncpp_t;

  // input type, needed for save
  mf::mf_input_file_type_e input_file_type = mf::xml_input_type;

  // input file, needed for save
  std::string input_file_name = "";

  // basic system info
  nda::stack_array<int,3> fft_mesh;
  long nnr = 0;
 
  // reciprocal lattice vectors
  nda::stack_array<double,3,3> recv;

  // reciprocal lattice vectors
  nda::stack_array<double,3,3> lattv;

  // spin-orbit
  bool spinorbit_loc = false;
  bool spinorbit_nl = false;

  // number of spins 
  int nspin = 1;

  // number of polarizations
  int npol = 1;

  /* kpoints and symmetry properties */
  long nkpts = 0;
  long nkpts_ibz = 0;
  nda::array<double, 2> kpts;      // in cartesian coordinates
  nda::array<double, 2> kpts_crys; // in crystal coordinates
  nda::array<int, 1> kp_to_ibz;
  nda::array<bool, 1> kp_trev; // symmetry operations
  std::vector<utils::symm_op> symm_list; // symmetry operations
  nda::array<int, 1> kp_symm;   // index of symmetry operation that connects kpts/kpts_crys to IRBZ

  // type of pseudo for each atom
  nda::array<int,1> ityp;

  // number of projectors for each pseudo typle
  nda::array<int,1> nh;

  // index of first projector for each atom 
  nda::array<int,1> ofs;

  // qq
  memory::unified_array<ComplexType,1> qq;

  // Matrix elements between projectors and basis orbitals (in mf)
  sarray_t<nda::array_view<ComplexType,4>> Pskna;

  // D matrix for local projectors
  //memory::unified_array<ComplexType,3> Dnn;
  sarray_t<nda::array_view<ComplexType,3>> Dnn;

  // mapping from wfc_g grid to rho grid. 
  // hard coding ecut in mf now, allow for a custom cutoff later on
  sarray_t<nda::array_view<long,1>> swfc_to_rho;

  // local pseudopotential
  sarray_t<nda::array_view<ComplexType,3>> svloc;

  // scf local potential
  sarray_t<nda::array_view<ComplexType,3>> svsc;

  // qgm
  sarray_t<nda::array_view<ComplexType,3>> qgm;

  template<typename MF_t>
  void read_vnl_pw2bgw(MF_t &mf, std::string outdir); 

  template<typename MF_t>
  void read_vnl_h5(MF_t &mf, h5::group& grp);

  // Build the projector-miller -> 'w' truncated-grid index map k2g(nk, npwx)
  // (reads "miller_k{ik}" from grp). Shared by read_vnl_h5 and
  // eph_projector_overlaps so both use identical G ordering.
  template<typename MF_t>
  void build_projector_k2g(MF_t& mf, h5::group& grp,
                           nda::array_const_view<int,1> npw,
                           nda::array_view<long,2> k2g);

  // Read projector `ib` at IBZ k-point `k` ("projector_k{k}"), scatter its
  // conjugate onto the truncated 'w' grid via k2g_k, returning vkb(ngm). `buff`
  // is a (1, npwx) scratch. Shared by read_vnl_h5 and eph_projector_overlaps.
  void read_projector_vkb(h5::group& grp, long k, int ib, int npw_k,
                          nda::array_const_view<long,1> k2g_k,
                          nda::array_view<ComplexType,2> buff,
                          nda::array_view<ComplexType,1> vkb);

  void add_vnl_impl(nda::range k_range, nda::range b_range,
               nda::ArrayOfRank<3> auto const& Dion, 
               math::nda::DistributedArrayOfRank<4> auto & Hij);

  /**
   * Add the contributions of a generic pseudopotentials:
   *
   *   1. Contributions from the local potentials are added to wavefunctions "hpsi"
   *   2. Contributions from the non-local potentials are added to the Hamiltonian
   *      in second quantization "Hij"
   *
   * @tparam Arr3   - Array type of nii
   * @tparam Arr4   - Array type of nij
   * @param k_range - [input] Range of k-point indices
   * @param b_range - [input] Range of orbital indices "b"
   * @param psi     - [input] Single-particle basis
   * @param hpsi    - [input] \hat{H_loc} * psi, where H_loc is an arbitrary local operator
   *                  [output] (\hat{H_loc} + Vpp_loc) * psi,
   *                           where Vpp_loc is the local part of the pseudopotential
   * @param Hij     - [input] Matrix elements of an arbitrary non-local operator H_nl
   *                  [output] Matrix elements of H_nl + Vpp_nl,
   *                           where Vpp_nl is the non-local part of the pseudopotential
   * @param nii     - [input] Diagonal density matrix (s, k, a)
   * @param nij     - [input] Density matrix (s, k, a, b)
   */
  template< nda::ArrayOfRank<3> Arr3, nda::ArrayOfRank<4> Arr4>
  void add_vpp_impl(boost::mpi3::communicator& comm,
               nda::range k_range, nda::range b_range, 
               math::nda::DistributedArrayOfRank<4> auto const& psi,
               math::nda::DistributedArrayOfRank<4> auto & hpsi,
               math::nda::DistributedArrayOfRank<4> auto & Hij,
               const Arr3 * nii, const Arr4 * nij);

  /**
   * Add the contributions of the Hartree potential to the wavefunctions "hpsi"
   *
   * @tparam Arr3   - Array type of "nii" array
   * @tparam Arr4   - Array type of "nij" array
   * @param k_range - [input] Range of k-point indices
   * @param psi     - [input] Single-particle basis (s, k, a, g), where g lives in the "wavefunction" grid
   * @param hpsi    - [input] \hat{H_loc} * psi, where H_loc is an arbitrary local operator
   *                  [output] (\hat{H_loc} + V_H) * psi,
   *                           where Vpp_loc is the local part of the pseudopotential
   * @param nii     - [input] Diagonal density matrix (s, k, a)
   * @param nij     - [input] Density matrix (s, k, a, b). Note that either "nii" or "nij"
   *                          should be provided, not both.
   */
  template<nda::ArrayOfRank<3> Arr3, nda::ArrayOfRank<4> Arr4>
  void add_Hartree_impl(nda::range k_range,
                        math::nda::DistributedArrayOfRank<4> auto const& psi,
                        math::nda::DistributedArrayOfRank<4> auto & hpsi,
                        const Arr3 *nii, const Arr4 *nij, bool symmetrize=false);


};

// if mf.get_pseudopot() returns a valid shared pointer, return it.
// otherwise, construct a new object managed by a shared pointer, 
// store the pointer in mf and return it.
template<typename MF_t>
std::shared_ptr<pseudopot> make_pseudopot(MF_t &mf)
{
  // sync for safety for now, this routine is blocking
  auto mpi = mf.mpi();
  mpi->comm.barrier();
  if(mf.get_pseudopot()) { return mf.get_pseudopot(); }
  else { 
    //
    auto psp = std::make_shared<pseudopot>(mf);
    mf.set_pseudopot(psp);
    if( not mf.get_pseudopot() )
      APP_ABORT("Error in make_pseudopot. Logic problem.");
    return psp;
  }
}

}

#endif
