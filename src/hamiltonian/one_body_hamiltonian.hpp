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


#ifndef HAMILTONIAN_ONE_BODY_HAMILTONIAN_HPP
#define HAMILTONIAN_ONE_BODY_HAMILTONIAN_HPP

#include "configuration.hpp"
#include "IO/app_loggers.h"
#include "utilities/check.hpp"

#include "mpi3/environment.hpp"
#include "mpi3/communicator.hpp"
#include "h5/h5.hpp"
#include "nda/nda.hpp"
#include "nda/h5.hpp"

#include "utilities/proc_grid_partition.hpp"
#include "numerics/distributed_array/nda.hpp"
#include "numerics/shared_array/nda.hpp"
#include "mean_field/MF.hpp"

#include "methods/tools/chkpt_utils.h"

#include "hamiltonian/matrix_elements.h"
#include "hamiltonian/pseudo/pseudopot.h"
#include "hamiltonian/gen_one_body_hamiltonian.icc"
#include "hamiltonian/pyscf_one_body_hamiltonian.icc"

namespace hamilt
{

namespace detail
{
/**
 * Build a distributed one-body (rank-4) quantity and scatter it into the shared
 * array `sX`. The distributed array is built over the first `n_active` ranks of
 * sX's internode communicator; any surplus ranks (when the processor grid cannot
 * tile all ranks, see utils::find_proc_grid_capped) stay idle. Correctness relies on the
 * active grid blocks covering the full global shape while every other element of
 * each node's window is zero, so the internode all_reduce assembles the complete
 * array -- hence sX is zeroed first.
 *
 * @param build - callable invoked only on active ranks with their sub-communicator;
 *                must return the distributed array (global shape == sX.shape()).
 */
template<class Array_4D_t, class Builder>
void assemble_one_body(math::shm::shared_array<Array_4D_t>& sX, long n_active, Builder&& build)
{
  sX.set_zero();
  if (sX.node_comm()->root()) {
    auto& internode = *sX.internode_comm();
    int color = (long(internode.rank()) < n_active) ? 0 : 1;
    auto active = internode.split(color, internode.rank());
    if (color == 0) {
      auto dX = build(active);
      auto Xl = sX.local();
      Xl(dX.local_range(0), dX.local_range(1), dX.local_range(2), dX.local_range(3)) = dX.local();
    }
  }
  sX.communicator()->barrier();
  sX.all_reduce();
}
} // namespace detail


/**
 * Non-interacting one-body hamiltonian associated with MF object in a distributed array
 * Only includes kinetic and pseudo-potential/external potential contributions.
 * @param mf    [input] - mean-field object
 * @param comm  [input] - communicator 
 * @param psp   [input] - pseudopotential object 
 * @param pgrid [input] - processor grid for the distributed array
 * @param bz    [input] - block size for the distributed array
 * @return - A distributed array of non-interacting one-body Hamiltonian
 *           with global shape = (nspin, nkpts, nbnd, nbnd)
 */ 
template<MEMORY_SPACE MEM = HOST_MEMORY>
auto H0(mf::MF &mf, boost::mpi3::communicator &comm, pseudopot *psp,  
        nda::range k_range = {-1,-1}, nda::range b_range = {-1,-1}, 
        std::array<long,4> pgrid = {0}, std::array<long,4> bz = {1,1,2048,2048})
{
  if(k_range == nda::range{-1,-1}) k_range = nda::range(mf.nkpts_ibz());
  if(b_range == nda::range{-1,-1}) b_range = nda::range(mf.nbnd());
  // this is, unfortunately, code dependent, so fork here!
  if (mf.mf_type() == mf::qe_source or mf.mf_type() == mf::bdft_source) {
    using larray = memory::array<MEM,ComplexType,4>;
    utils::check(psp != nullptr, "Error in H0: Missing pseudopot object.");
    auto psi = mf::read_distributed_orbital_set_ibz<larray>(mf,comm,'w',pgrid,
                               nda::range(-1,-1), k_range, b_range, bz); 
    memory::array_view<MEM,ComplexType,3> *p3=nullptr;
    memory::array_view<MEM,ComplexType,4> *p4=nullptr;
    return detail::gen_H0<MEM>(mf,comm,psp,k_range,b_range,psi,p3,p4);
  } else {
    utils::check(mf.mf_type() == mf::pyscf_source, "Source mismacth");
    utils::check(k_range.size() == mf.nkpts_ibz(), "No k_range with pyscf backend yet.");
    utils::check(b_range.size() == mf.nbnd(), "No b_range with pyscf backend yet.");
    return detail::pyscf_read_1B_from_file<MEM>(mf,"H0",comm,pgrid,bz);
  }
}

/**
 * Non-interacting Hamiltonian associated with a MF object in a shared memory array
 * Includes kinetic, pseudo-potential/external potential contributions
 * @return - A shared memory array of non-interacting Hamiltonian with global shape = (nspin, nkpts, nbnd, nbnd)
 */
template<nda::MemoryArrayOfRank<4> Array_4D_t>
void set_H0(mf::MF &mf, pseudopot *psp, math::shm::shared_array<Array_4D_t> &sH0_skij) {
  long np = sH0_skij.internode_comm()->size();
  auto [pgrid, n_active] = utils::find_proc_grid_capped<4>(
      np, {(long)mf.nspin(), (long)mf.nkpts_ibz(), (long)mf.nbnd(), 1l});
  std::array<long, 4> bsize = {1, 1, std::max(1l, std::min(1024l, mf.nbnd()/pgrid[2])),
                                     2048l};
  app_log(4, "One-body Hamiltonian in distributed array: ");
  app_log(4, "  - pgrid = ({}, {}, {}, {}), active ranks = {}/{}", pgrid[0], pgrid[1], pgrid[2], pgrid[3], n_active, np);
  app_log(4, "  - bsize = ({}, {}, {}, {})\n", bsize[0], bsize[1], bsize[2], bsize[3]);

  detail::assemble_one_body(sH0_skij, n_active, [&](boost::mpi3::communicator& c) {
    return hamilt::H0<HOST_MEMORY>(mf, c, psp,
                                   nda::range(mf.nkpts_ibz()), nda::range(mf.nbnd()),
                                   pgrid, bsize);
  });
}

/**
 * One-body hamiltonian associated with MF object in a distributed array
 * Includes kinetic, hartree and pseudo-potential/external potential contributions.
 * @param mf    [input] - mean-field object
 * @param comm  [input] - communicator 
 * @param psp   [input] - pseudopotential object 
 * @param rhoij [input] - density matrix 
 * @param pgrid [input] - processor grid for the distributed array
 * @param bz    [input] - block size for the distributed array
 * @return - A distributed array of non-interacting one-body Hamiltonian
 *           with global shape = (nspin, nkpts, nbnd, nbnd)
 */
template<MEMORY_SPACE MEM = HOST_MEMORY>
auto H(mf::MF &mf, boost::mpi3::communicator &comm, pseudopot *psp,
        nda::ArrayOfRank<4> auto const& nij,
        std::array<long,4> pgrid = {0}, std::array<long,4> bz = {1,1,2048,2048})
{ 
  using nij_type = decltype(nij);
  static_assert(memory::get_memory_space<nij_type>() == MEM, "Memory Space mismatch.");
  // this is, unfortunately, code dependent, so fork here!
  if (mf.mf_type() == mf::qe_source or mf.mf_type() == mf::bdft_source) {
    using larray = memory::array<MEM,ComplexType,4>;
    utils::check(psp != nullptr, "Error in H0: Missing pseudopot object.");
    nda::range b_rng(mf.nbnd());
    nda::range k_rng(mf.nkpts_ibz());
    auto psi = mf::read_distributed_orbital_set_ibz<larray>(mf,comm,'w',pgrid,
                               nda::range(mf.nspin()),k_rng,b_rng,bz);
    memory::array_view<MEM,ComplexType,3> *p3=nullptr;
    return detail::gen_H0<MEM>(mf,comm,psp,k_rng,b_rng,psi,p3,std::addressof(nij));
  } else {
    utils::check(mf.mf_type() == mf::pyscf_source, "Source mismatch");
    return detail::pyscf_read_1B_from_file<MEM>(mf,"H0",comm,pgrid,bz);
  }
}

/**
 * One-body hamiltonian associated with MF object in a distributed array
 * Includes kinetic, hartree and pseudo-potential/external potential contributions.
 * @param mf    [input] - mean-field object
 * @param comm  [input] - communicator 
 * @param psp   [input] - pseudopotential object 
 * @param rhoij [input] - density matrix 
 * @param pgrid [input] - processor grid for the distributed array
 * @param bz    [input] - block size for the distributed array
 * @return - A distributed array of non-interacting one-body Hamiltonian
 *           with global shape = (nspin, nkpts, nbnd, nbnd)
 */
template<MEMORY_SPACE MEM = HOST_MEMORY>
auto H(mf::MF &mf, boost::mpi3::communicator &comm, pseudopot *psp,
        nda::ArrayOfRank<3> auto const& nii,
        std::array<long,4> pgrid = {0}, std::array<long,4> bz = {1,1,2048,2048})
{
  // this is, unfortunately, code dependent, so fork here!
  if (mf.mf_type() == mf::qe_source or mf.mf_type() == mf::bdft_source) {
    using larray = memory::array<MEM,ComplexType,4>;
    utils::check(psp != nullptr, "Error in H0: Missing pseudopot object.");
    nda::range b_rng(mf.nbnd());
    nda::range k_rng(mf.nkpts_ibz());
    auto psi = mf::read_distributed_orbital_set_ibz<larray>(mf,comm,'w',pgrid,
                               nda::range(mf.nspin()),k_rng,b_rng,bz);
    memory::array_view<MEM,ComplexType,4> *p4=nullptr;
    return detail::gen_H0<MEM>(mf,comm,psp,k_rng,b_rng,psi,std::addressof(nii),p4);
  } else {
    utils::check(mf.mf_type() == mf::pyscf_source, "Source mismatch");
    return detail::pyscf_read_1B_from_file<MEM>(mf,"H0",comm,pgrid,bz);
  }
}

/**
 * Fock matrix associated with a MF object.
 * Includes Kinetic, pseudo-potential/external potential and HF/Vxc potential
 *
 * This is the pure diag(mf.eigval()) primitive. For an augmented (non-eigenstate)
 * basis the one-body Hamiltonian is not diagonal, and its stored full matrix is
 * consumed by set_fock, not here -- call set_fock rather than F to obtain the
 * one-body seed of an arbitrary mean field.
 * @return - A distributed array of one-body Hamiltonian with global shape = (nspin, nkpts, nbnd, nbnd)
 */
template<MEMORY_SPACE MEM = HOST_MEMORY>
auto F(mf::MF& mf, boost::mpi3::communicator& comm, 
       nda::range k_range = {-1,-1}, nda::range b_range = {-1,-1}, 
       std::array<long,4> pgrid = {0}, std::array<long,4> bz = {1,1,2048,2048}, 
       bool evaluate = false)
{
  if(k_range == nda::range{-1,-1}) k_range = nda::range(mf.nkpts_ibz());
  if(b_range == nda::range{-1,-1}) b_range = nda::range(mf.nbnd());
  // this is, unfortunately, code dependent, so fork here!
  if (mf.mf_type() == mf::qe_source or mf.mf_type() == mf::bdft_source) {
    auto eig = mf.eigval();
    if(evaluate or std::all_of( eig.begin(),  eig.end(), [] (auto&& v){ return v==0; })) {
      APP_ABORT("Error in hamilt::F(evaluate=true), disabled evaluation of Fock matrix without eigenvalues.");
    }
    {
      long nspin = mf.nspin();
      long nbnd = mf.nbnd();
      long nkpts = k_range.size();
      long M = b_range.size();
      utils::check(b_range.first() >= 0 and b_range.last() <= nbnd, "Band range out of bounds.");
      utils::check(k_range.first() >= 0 and k_range.last() <= mf.nkpts(), "K-point range out of bounds.");
      long np0 = std::accumulate(pgrid.cbegin(), pgrid.cend(), long(1), std::multiplies<>{});
      if( np0 == 0 ) {
        // cap every axis at its extent and spill leftover ranks across both band
        // axes so no dimension is over-subscribed (see utils::find_proc_grid_capped).
        auto [g, n_active] = utils::find_proc_grid_capped<4>(comm.size(), {nspin, nkpts, M, M});
        utils::check(n_active == comm.size(),
          "hamilt::F: cannot tile {} ranks onto (nspin={}, nkpts={}, nbnd={}); pass an "
          "explicit processor grid or use fewer ranks.", comm.size(), nspin, nkpts, M);
        pgrid = g;
      }
      using larray = memory::array<MEM,ComplexType,4>;
      auto Fij = math::nda::make_distributed_array<larray>(comm,pgrid,{nspin,nkpts,M,M},
                  {bz[0],bz[1],bz[2],bz[2]});
      auto Floc = Fij.local();
      Floc = ComplexType(0.0);
      for( auto [is, s] : itertools::enumerate(Fij.local_range(0)))
        for( auto [ik, k] : itertools::enumerate(Fij.local_range(1)))
          for( auto [ia, a] : itertools::enumerate(Fij.local_range(2)))
            for( auto [ib, b] : itertools::enumerate(Fij.local_range(3)))
              if(a==b) Floc(is,ik,ia,ib) = eig(s,k+k_range.first(),a+b_range.first());
      comm.barrier();
      return Fij;
    }
  } else {
    utils::check(mf.mf_type() == mf::pyscf_source, "Source mismatch");
    utils::check(k_range.size() == mf.nkpts_ibz(), "No k_range with pyscf backend yet.");
    utils::check(b_range.size() == mf.nbnd(), "No b_range with pyscf backend yet.");
    return detail::pyscf_read_1B_from_file<MEM>(mf,"Fock",comm,pgrid,bz);
  }
}

/**
 * DFT Hartree + exchange-correlation potential V_Hxc = V_H + V_xc in a distributed
 * array, evaluated from the converged QE local potentials (svsc - svloc) stored in the
 * pseudopot object. Used to seed the iter-0 Fock/effective Hamiltonian of an augmented
 * (non-eigenstate) basis, where the mf.eigval() slot holds kinetic Rayleigh seeds
 * rather than KS eigenvalues. Only supported for QE/bdft backends with a local/semilocal
 * functional (see gen_V_Hxc_aug guards).
 * @return - A distributed array with global shape = (nspin, nkpts_ibz, nbnd, nbnd)
 */
template<MEMORY_SPACE MEM = HOST_MEMORY>
auto V_Hxc_aug(mf::MF &mf, boost::mpi3::communicator &comm, pseudopot *psp,
               nda::range k_range = {-1,-1}, nda::range b_range = {-1,-1},
               std::array<long,4> pgrid = {0}, std::array<long,4> bz = {1,1,2048,2048})
{
  if(k_range == nda::range{-1,-1}) k_range = nda::range(mf.nkpts_ibz());
  if(b_range == nda::range{-1,-1}) b_range = nda::range(mf.nbnd());
  utils::check(mf.mf_type() == mf::qe_source or mf.mf_type() == mf::bdft_source,
               "V_Hxc_aug: only supported for QE/bdft mean-field backends.");
  using larray = memory::array<MEM,ComplexType,4>;
  utils::check(psp != nullptr, "V_Hxc_aug: Missing pseudopot object.");
  auto psi = mf::read_distributed_orbital_set_ibz<larray>(mf,comm,'w',pgrid,
                                                          nda::range(-1,-1), k_range, b_range, bz);
  return detail::gen_V_Hxc_aug<MEM>(mf,psp,k_range,b_range,psi);
}

/**
 * Read the full Kohn-Sham matrix of an augmented basis, Orbitals/H_KS_skij, into a
 * shared memory array. The dataset is stored over the IBZ only, (nspin, nkpts_ibz,
 * nbnd_stored, nbnd_stored); a run that opens the basis with fewer bands gets the
 * leading block as an h5 hyperslab, which drops the coupling to the discarded states.
 * The h5 schema belongs to the mean-field backend (MF::read_hks_matrix); this owns
 * only the node-root gating and the window synchronization.
 * Aborts unless mf.is_augmented() and mf.has_hks_matrix(): there is no fallback seed
 * for an augmented basis, so a basis without the matrix is unusable.
 */
template<nda::MemoryArrayOfRank<4> Array_4D_t>
void read_H_KS_aug(mf::MF &mf, math::shm::shared_array<Array_4D_t> &sH_KS) {
  long nspin = mf.nspin();
  long nkpts_ibz = mf.nkpts_ibz();
  long nbnd = mf.nbnd();
  utils::check(mf.is_augmented() and mf.has_hks_matrix(),
               "read_H_KS_aug: the augmented basis {} carries no Orbitals/H_KS_skij "
               "dataset. An augmented basis is not an eigenbasis, so its one-body "
               "Hamiltonian cannot be reconstructed from the eigenvalues alone: the full "
               "Kohn-Sham matrix is required. Regenerate the basis with the current code "
               "(augment_mf / augment_mf_dpsi), which always stores it.", mf.filename());
  utils::check(sH_KS.shape() == std::array<long,4>{nspin, nkpts_ibz, nbnd, nbnd},
               "read_H_KS_aug: shape ({},{},{},{}) != ({},{},{},{}).",
               sH_KS.shape()[0], sH_KS.shape()[1], sH_KS.shape()[2], sH_KS.shape()[3],
               nspin, nkpts_ibz, nbnd, nbnd);
  long nbnd_stored = nbnd;
  if (sH_KS.node_comm()->root())
    nbnd_stored = mf.read_hks_matrix(sH_KS.local());
  sH_KS.node_sync();
  // broadcast so the notice is emitted by every rank and the logger's own root
  // gating decides who prints it, rather than by whichever node root read the file
  sH_KS.node_comm()->broadcast_n(&nbnd_stored, 1, 0);
  if (nbnd_stored > nbnd)
    app_log(1, "  [WARNING] Augmented H_KS: using the leading {} of {} stored bands; the "
               "coupling to the discarded states is dropped.", nbnd, nbnd_stored);
}

/**
 * One-body Hamiltonian associated with a MF object in a shared memory array
 * Includes Kinetic, pseudo-potential/external potential and HF/Vxc potential
 * @param exclude_H0 [INPUT] - exclude kinetic + pseudo/external potential or not
 * @param sH0_skij   [INPUT, optional] - when exclude_H0 is set, subtract this
 *        precomputed H0 (global shape (nspin, nkpts_ibz, nbnd, nbnd)) instead of
 *        recomputing it. Lets callers that already hold H0 (e.g. the dyson solver)
 *        avoid the redundant O(nbnd²·npw) PW/FFT recompute. nullptr → recompute,
 *        except for an augmented basis, where it is required.
 * @return - A shared memory array of one-body Hamiltonian with global shape = (nspin, nkpts, nbnd, nbnd)
 */
template<nda::MemoryArrayOfRank<4> Array_4D_t>
void set_fock(mf::MF &mf, pseudopot *psp, math::shm::shared_array<Array_4D_t> &sF_skij,
              bool exclude_H0=false,
              const math::shm::shared_array<Array_4D_t> *sH0_skij=nullptr) {
  // An augmented basis is not an eigenbasis, so hamilt::F's diag(eigval) is only the
  // diagonal of its one-body Hamiltonian: the stored H_KS matrix is its only valid
  // seed, and a basis that does not carry one is an error.
  if (mf.is_augmented()) {
    read_H_KS_aug(mf, sF_skij);
    if (exclude_H0) {
      utils::check(sH0_skij != nullptr,
                   "set_fock: the augmented H_KS seed with exclude_H0 requires the caller "
                   "to provide sH0_skij.");
      // whole-array subtraction: unlike the distributed non-augmented path below,
      // there is no local_range to slice by, so the shapes must agree exactly
      utils::check(sH0_skij->shape() == sF_skij.shape(),
                   "set_fock: sH0_skij shape ({},{},{},{}) != sF_skij shape ({},{},{},{}).",
                   sH0_skij->shape()[0], sH0_skij->shape()[1], sH0_skij->shape()[2],
                   sH0_skij->shape()[3], sF_skij.shape()[0], sF_skij.shape()[1],
                   sF_skij.shape()[2], sF_skij.shape()[3]);
      if (sF_skij.node_comm()->root()) sF_skij.local() -= sH0_skij->local();
      sF_skij.node_sync();
    }
    return;
  }

  long np = sF_skij.internode_comm()->size();
  auto [pgrid, n_active] = utils::find_proc_grid_capped<4>(
      np, {(long)mf.nspin(), (long)mf.nkpts_ibz(), (long)mf.nbnd(), 1l});
  std::array<long, 4> bsize = {1, 1, std::max(1l, std::min(1024l, mf.nbnd()/pgrid[2])),
                                     2048l};
  app_log(4, "One-body Hamiltonian in distributed array: ");
  app_log(4, "  - pgrid = ({}, {}, {}, {}), active ranks = {}/{}", pgrid[0], pgrid[1], pgrid[2], pgrid[3], n_active, np);
  app_log(4, "  - bsize = ({}, {}, {}, {})\n", bsize[0], bsize[1], bsize[2], bsize[3]);

  detail::assemble_one_body(sF_skij, n_active, [&](boost::mpi3::communicator& c) {
    auto dF = hamilt::F<HOST_MEMORY>(mf, c, nda::range(mf.nkpts_ibz()),
                                     nda::range(mf.nbnd()), pgrid, bsize);
    if (exclude_H0) {
      if (sH0_skij != nullptr) {
        // subtract the caller-provided H0 restricted to dF's local block
        auto H0_loc = sH0_skij->local();
        dF.local() -= H0_loc(dF.local_range(0), dF.local_range(1),
                             dF.local_range(2), dF.local_range(3));
      } else {
        auto dH0 = hamilt::H0<HOST_MEMORY>(mf, c, psp, nda::range(mf.nkpts_ibz()),
                                           nda::range(mf.nbnd()), pgrid, bsize);
        dF.local() -= dH0.local();
      }
    }
    return dF;
  });
}

/**
 * Compute the matrix elements of the Hartree potential in a distributed array
 * using PWs and FFT
 *
 * @param mf      [input] - mean-field object
 * @param comm    [input] - communicator
 * @param psp     [input] - pseudopotential object
 * @param nij     [input] - density matrix (nspin, nkpts, nbnd, nbnd)
 * @param k_range [input] - index range for k-points
 * @param b_range [input] - indenx range for orbitals
 * @param pgrid   [input] - processor grid for the distributed array
 * @param bz      [input] - block size for the distributed array
 * @return - A distributed array of the Hartree Hamiltonian
 *           with global shape = (nspin, k_range.size(), b_range.size(), b_range.size())
 */
template<MEMORY_SPACE MEM = HOST_MEMORY, nda::ArrayOfRank<4> Arr4_t>
auto Vhartree(mf::MF &mf, boost::mpi3::communicator &comm, pseudopot *psp,
        Arr4_t const& nij,
        nda::range k_range = {-1,-1}, nda::range b_range = {-1,-1},
        std::array<long,4> pgrid = {0}, std::array<long,4> bz = {1,1,2048,2048})
{
  if(k_range == nda::range{-1,-1}) k_range = nda::range(mf.nkpts_ibz());
  if(b_range == nda::range{-1,-1}) b_range = nda::range(mf.nbnd());
  // this is, unfortunately, code dependent, so fork here!
  if (mf.mf_type() == mf::qe_source or mf.mf_type() == mf::bdft_source) {
    using larray = memory::array<MEM,ComplexType,4>;
    utils::check(psp != nullptr, "Error in Vhartree: Missing pseudopot object.");
    auto psi = mf::read_distributed_orbital_set_ibz<larray>(mf,comm,'w',pgrid,
                                                            nda::range(-1,-1), k_range, b_range, bz);
    memory::array_view<MEM,ComplexType,3> *p3=nullptr;
    return detail::gen_Vhartree<MEM>(mf,comm,psp,k_range,b_range,psi,p3,std::addressof(nij),false);
  } else {
    utils::check(mf.mf_type() == mf::pyscf_source, "Source mismatch");
    utils::check(mf.orb_on_fft_grid(),
                 "Vhartree: The Hartree potential cannot be evaluated using FFT if mf.orb_on_fft_grid() == false");
    utils::check(false, "Vhartree: Hartree potential using FFT is not implemented for non-orthogonal basis yet!");
    utils::check(k_range.size() == mf.nkpts_ibz(), "No k_range with pyscf backend yet.");
    utils::check(b_range.size() == mf.nbnd(), "No b_range with pyscf backend yet.");
    return detail::pyscf_read_1B_from_file<MEM>(mf,"H0",comm,pgrid,bz);
  }
}

template<MEMORY_SPACE MEM = HOST_MEMORY, nda::ArrayOfRank<3> Arr3_t>
auto Vhartree(mf::MF &mf, boost::mpi3::communicator &comm, pseudopot *psp,
              Arr3_t const& nii,
              nda::range k_range = {-1,-1}, nda::range b_range = {-1,-1},
              std::array<long,4> pgrid = {0}, std::array<long,4> bz = {1,1,2048,2048})
{
  if(k_range == nda::range{-1,-1}) k_range = nda::range(mf.nkpts_ibz());
  if(b_range == nda::range{-1,-1}) b_range = nda::range(mf.nbnd());
  // this is, unfortunately, code dependent, so fork here!
  if (mf.mf_type() == mf::qe_source or mf.mf_type() == mf::bdft_source) {
    using larray = memory::array<MEM,ComplexType,4>;
    utils::check(psp != nullptr, "Error in Vhartree: Missing pseudopot object.");
    auto psi = mf::read_distributed_orbital_set_ibz<larray>(mf,comm,'w',pgrid,
                                                            nda::range(-1,-1), k_range, b_range, bz);
    memory::array_view<MEM,ComplexType,4> *p4=nullptr;
    return detail::gen_Vhartree<MEM>(mf,comm,psp,k_range,b_range,psi,std::addressof(nii),p4,false);
  } else {
    utils::check(mf.mf_type() == mf::pyscf_source, "Source mismatch");
    utils::check(mf.orb_on_fft_grid(),
                 "Vhartree: The Hartree potential cannot be evaluated using FFT if mf.orb_on_fft_grid() == false");
    utils::check(false, "Vhartree: Hartree potential using FFT is not implemented for non-orthogonal basis yet!");
    utils::check(k_range.size() == mf.nkpts_ibz(), "No k_range with pyscf backend yet.");
    utils::check(b_range.size() == mf.nbnd(), "No b_range with pyscf backend yet.");
    return detail::pyscf_read_1B_from_file<MEM>(mf,"H0",comm,pgrid,bz);
  }
}

template<typename MPI_t>
void dump_hartree(MPI_t &mpi, mf::MF &mf, pseudopot *psp, std::string coqui_output, long scf_iter) {
  long np = mpi.comm.size();
  auto [pgrid, n_active] = utils::find_proc_grid_capped<4>(
      np, {(long)mf.nspin(), (long)mf.nkpts_ibz(), (long)mf.nbnd(), 1l});
  std::array<long, 4> bsize = {1, 1, std::max(1l, std::min(1024l, mf.nbnd()/pgrid[2])),
                                     2048l};
  app_log(4, "  - pgrid = ({}, {}, {}, {}), active ranks = {}/{}", pgrid[0], pgrid[1], pgrid[2], pgrid[3], n_active, np);
  app_log(4, "  - bsize = ({}, {}, {}, {})\n", bsize[0], bsize[1], bsize[2], bsize[3]);

  // logic of iteration
  if (scf_iter == -1) {
    std::string filename = coqui_output + ".mbpt.h5";
    h5::file file(filename, 'r');
    auto scf_grp = h5::group(file).open_group("scf");
    h5::h5_read(scf_grp, "final_iter", scf_iter);
  }
  long dm_iter = (scf_iter == 0)? scf_iter : scf_iter-1;

  using larray_view = memory::array_view<HOST_MEMORY,ComplexType,4>;
  using math::shm::make_shared_array;
  auto sDm_skij = make_shared_array<larray_view>(mpi, {mf.nspin(), mf.nkpts_ibz(), mf.nbnd(), mf.nbnd()});
  methods::chkpt::read_dm(mpi.node_comm, coqui_output, dm_iter, sDm_skij);

  std::string filename = coqui_output + ".mbpt.h5";
  app_log(2, "Dump the matrix elements of the Hartree potential: ");
  app_log(2, "  - h5 file: {}", filename);
  app_log(2, "  - h5 dataset = scf/iter{}/VH_skij\n", scf_iter);

  // Build + write the distributed potential on the active sub-communicator; when
  // the grid cannot tile all ranks (see utils::find_proc_grid_capped) the surplus stay
  // idle. Rank 0 is always active, so it owns the h5 file handle.
  int color = (long(mpi.comm.rank()) < n_active) ? 0 : 1;
  auto active = mpi.comm.split(color, mpi.comm.rank());
  if (color == 0) {
    auto dVH = hamilt::Vhartree<HOST_MEMORY>(mf, active, psp,
                                             sDm_skij.local(), nda::range(mf.nkpts_ibz()), nda::range(mf.nbnd()),
                                             pgrid, bsize);
    h5::group iter_grp;
    if (active.root()) {
      h5::file file(filename, 'a');
      auto scf_grp = h5::group(file).open_group("scf");
      utils::check(scf_grp.has_subgroup("iter"+std::to_string(scf_iter)),
                   "dump_hartree: \"scf/iter{}\" does not exist!");
      iter_grp = scf_grp.open_group("iter"+std::to_string(scf_iter));
      math::nda::h5_write(iter_grp, "VH_skij", dVH);
    } else {
      math::nda::h5_write(iter_grp, "VH_skij", dVH);
    }
  }
  mpi.comm.barrier();
}

/**
 * Overlap matrix associated with a MF object in a distributed array
 * @return - A distributed array of overlap matrix with global shape = (nspin, nkpts, nbnd, nbnd)
 */
template<MEMORY_SPACE MEM = HOST_MEMORY>
auto ovlp(mf::MF& mf, boost::mpi3::communicator& comm, 
          nda::range k_range = {-1,-1}, nda::range b_range = {-1,-1},
          std::array<long,4> pgrid = {0}, std::array<long,4> bz = {1,1,2048,2048})
{
  if(k_range == nda::range{-1,-1}) k_range = nda::range(mf.nkpts_ibz());
  if(b_range == nda::range{-1,-1}) b_range = nda::range(mf.nbnd());
  // this is, unfortunately, code dependent, so fork here!
  if (mf.mf_type() == mf::qe_source or mf.mf_type() == mf::bdft_source) {
    using larray = memory::array<MEM,ComplexType,4>;
    auto psi = mf::read_distributed_orbital_set_ibz<larray>(mf,comm,'w',pgrid,
                               nda::range(-1,-1), k_range, b_range, bz); 
    return detail::gen_ovlp<MEM,false>(comm,psi);
  } else {
    utils::check(mf.mf_type() == mf::pyscf_source, "Source mismatch");
    utils::check(k_range.size() == mf.nkpts_ibz(), "No k_range with pyscf backend yet.");
    utils::check(b_range.size() == mf.nbnd(), "No b_range with pyscf backend yet.");
    return detail::pyscf_read_1B_from_file<MEM>(mf,"ovlp",comm,pgrid,bz);
  }
}
/**
 * Diagonal elements of the overlap matrix associated with a MF object in a distributed array
 * @return - A distributed array of overlap matrix with global shape = (nspin, nkpts, nbnd, nbnd)
 */
template<MEMORY_SPACE MEM = HOST_MEMORY>
auto ovlp_diagonal(mf::MF& mf, boost::mpi3::communicator& comm, 
          nda::range k_range = {-1,-1}, nda::range b_range = {-1,-1},
          std::array<long,3> pgrid = {0}, std::array<long,3> bz = {1,1,2048})
{
  if(k_range == nda::range{-1,-1}) k_range = nda::range(mf.nkpts_ibz());
  if(b_range == nda::range{-1,-1}) b_range = nda::range(mf.nbnd());
  // this is, unfortunately, code dependent, so fork here!
  if (mf.mf_type() == mf::qe_source or mf.mf_type() == mf::bdft_source) {
    using larray = memory::array<MEM,ComplexType,4>;
    auto psi = mf::read_distributed_orbital_set_ibz<larray>(mf,comm,'w',
             {pgrid[0],pgrid[1],pgrid[2],1},nda::range(-1,-1), k_range, b_range,
             {bz[0],bz[1],bz[2],2048});
    return detail::gen_ovlp<MEM,true>(comm,psi);
  } else {
    utils::check(mf.mf_type() == mf::pyscf_source, "Source mismatch");
    utils::check(k_range.size() == mf.nkpts_ibz(), "No k_range with pyscf backend yet.");
    utils::check(b_range.size() == mf.nbnd(), "No b_range with pyscf backend yet.");
    return detail::pyscf_read_diag_1B_from_file<MEM>(mf,"ovlp",comm,pgrid,bz);
  }
}

/**
 * Overlap matrix associated with a MF object in a shared memory array
 * @return - A shared memory array of overlap matrix with global shape = (nspin, nkpts, nbnd, nbnd)
 */
template<nda::MemoryArrayOfRank<4> Array_4D_t>
void set_ovlp(mf::MF &mf, math::shm::shared_array<Array_4D_t> &sS_skij) {
  long np = sS_skij.internode_comm()->size();
  auto [pgrid, n_active] = utils::find_proc_grid_capped<4>(
      np, {(long)mf.nspin(), (long)mf.nkpts_ibz(), (long)mf.nbnd(), 1l});
  std::array<long, 4> bsize = {1, 1, std::max(1l, std::min(1024l, mf.nbnd()/pgrid[2])),
                                     2048l};
  app_log(4, "One-body Hamiltonian in distributed array: ");
  app_log(4, "  - pgrid = ({}, {}, {}, {}), active ranks = {}/{}", pgrid[0], pgrid[1], pgrid[2], pgrid[3], n_active, np);
  app_log(4, "  - bsize = ({}, {}, {}, {})\n", bsize[0], bsize[1], bsize[2], bsize[3]);

  detail::assemble_one_body(sS_skij, n_active, [&](boost::mpi3::communicator& c) {
    return ovlp<HOST_MEMORY>(mf, c, nda::range(mf.nkpts_ibz()),
                             nda::range(mf.nbnd()), pgrid, bsize);
  });
}

/**
 * Exchange-correlation hamiltonian associated with MF object in a distributed array
 * @param mf    [input] - mean-field object
 * @param comm  [input] - communicator
 * @param pgrid [input] - processor grid for the distributed array
 * @param bz    [input] - block size for the distributed array
 * @return - A distributed array of non-interacting one-body Hamiltonian
 *           with global shape = (nspin, nkpts, nbnd, nbnd)
 */
template<MEMORY_SPACE MEM = HOST_MEMORY>
auto Vxc(mf::MF &mf, boost::mpi3::communicator &comm,
         nda::range k_range = {-1,-1}, nda::range b_range = {-1,-1},
         std::array<long,4> pgrid = {0}, std::array<long,4> bz = {1,1,2048,2048})
{
  if(k_range == nda::range{-1,-1}) k_range = nda::range(mf.nkpts_ibz());
  if(b_range == nda::range{-1,-1}) b_range = nda::range(mf.nbnd());
  // this is, unfortunately, code dependent, so fork here!
  if (mf.mf_type() == mf::qe_source or mf.mf_type() == mf::bdft_source) {
    using larray = memory::array<MEM,ComplexType,4>;
    auto psi = mf::read_distributed_orbital_set_ibz<larray>(mf,comm,'w',pgrid,
                                                            nda::range(-1,-1), k_range, b_range, bz);
    return detail::gen_Vxc<MEM>(mf,k_range,b_range,psi);
  } else {
    utils::check(mf.mf_type() == mf::pyscf_source, "Source mismatch");
    utils::check(k_range.size() == mf.nkpts_ibz(), "No k_range with pyscf backend yet.");
    utils::check(b_range.size() == mf.nbnd(), "No b_range with pyscf backend yet.");
    return detail::pyscf_read_1B_from_file<MEM>(mf,"Vxc",comm,pgrid,bz);
  }
}


/**
 * Exchange-correlation Hamiltonian associated with a MF object in a shared memory array
 * @return - A shared memory array of exchange-correlation Hamiltonian
 *           with global shape = (nspin, nkpts, nbnd, nbnd)
 */
template<nda::MemoryArrayOfRank<4> Array_4D_t>
void set_Vxc(mf::MF &mf, math::shm::shared_array<Array_4D_t> &sVxc_skij) {
  long np = sVxc_skij.internode_comm()->size();
  auto [pgrid, n_active] = utils::find_proc_grid_capped<4>(
      np, {(long)mf.nspin(), (long)mf.nkpts_ibz(), (long)mf.nbnd(), 1l});
  std::array<long, 4> bsize = {1, 1, std::max(1l, std::min(1024l, mf.nbnd()/pgrid[2])),
                                     2048l};
  app_log(4, "One-body Hamiltonian in distributed array: ");
  app_log(4, "  - pgrid = ({}, {}, {}, {}), active ranks = {}/{}", pgrid[0], pgrid[1], pgrid[2], pgrid[3], n_active, np);
  app_log(4, "  - bsize = ({}, {}, {}, {})\n", bsize[0], bsize[1], bsize[2], bsize[3]);

  detail::assemble_one_body(sVxc_skij, n_active, [&](boost::mpi3::communicator& c) {
    return hamilt::Vxc<HOST_MEMORY>(mf, c,
                                    nda::range(mf.nkpts_ibz()), nda::range(mf.nbnd()),
                                    pgrid, bsize);
  });
}

template<typename MPI_t>
void dump_vxc(MPI_t &mpi, mf::MF &mf, std::string coqui_output) {
  long np = mpi.comm.size();
  auto [pgrid, n_active] = utils::find_proc_grid_capped<4>(
      np, {(long)mf.nspin(), (long)mf.nkpts_ibz(), (long)mf.nbnd(), 1l});
  std::array<long, 4> bsize = {1, 1, std::max(1l, std::min(1024l, mf.nbnd()/pgrid[2])),
                                     2048l};
  app_log(2, "Evaluate the matrix elements of the exchange-correlation potential: ");
  app_log(2, "  - mean-field backend = {}", mf::mf_source_enum_to_string(mf.mf_type()));
  app_log(4, "  - pgrid = ({}, {}, {}, {}), active ranks = {}/{}", pgrid[0], pgrid[1], pgrid[2], pgrid[3], n_active, np);
  app_log(4, "  - bsize = ({}, {}, {}, {})", bsize[0], bsize[1], bsize[2], bsize[3]);
  app_log(2, "");

  std::string filename = coqui_output + ".mbpt.h5";
  app_log(2, "Dump the matrix elements of the exchange-correlation potential: ");
  app_log(2, "  - h5 file: {}", filename);
  app_log(2, "  - h5 dataset = system/Vxc_skij\n");

  // Build + write on the active sub-communicator; surplus ranks stay idle when
  // the grid cannot tile all ranks (see utils::find_proc_grid_capped). Rank 0 is always
  // active, so it owns the h5 file handle.
  int color = (long(mpi.comm.rank()) < n_active) ? 0 : 1;
  auto active = mpi.comm.split(color, mpi.comm.rank());
  if (color == 0) {
    auto dVxc = hamilt::Vxc<HOST_MEMORY>(mf, active,
                                         nda::range(mf.nkpts_ibz()), nda::range(mf.nbnd()),
                                         pgrid, bsize);
    h5::group sys_grp;
    if (active.root()) {
      h5::file file(filename, 'a');
      h5::group grp(file);

      if (grp.has_subgroup("system")) {
        sys_grp = grp.open_group("system");
        // Read into the exact types write_metadata used, so h5 does not warn
        // about long/int view mismatches.
        decltype(mf.nspin())     ns;
        decltype(mf.nkpts())     nkpts;
        decltype(mf.nkpts_ibz()) nkpts_ibz;
        decltype(mf.nbnd())      nbnd;
        decltype(mf.npol())      npol;
        h5::h5_read(sys_grp, "number_of_spins", ns);
        h5::h5_read(sys_grp, "number_of_kpoints", nkpts);
        h5::h5_read(sys_grp, "number_of_kpoints_ibz", nkpts_ibz);
        h5::h5_read(sys_grp, "number_of_orbitals", nbnd);
        h5::h5_read(sys_grp, "number_of_polarizations", npol);
        utils::check(ns == mf.nspin(), "dump_vxc: inconsistent \"nspin\" in coqui mean-field and {}", filename);
        utils::check(nkpts == mf.nkpts(), "dump_vxc: inconsistent \"nkpts\" in coqui mean-field and {}", filename);
        utils::check(nkpts_ibz == mf.nkpts_ibz(), "dump_vxc: inconsistent \"nkpts_ibz\" in coqui mean-field and {}",
                     filename);
        utils::check(nbnd == mf.nbnd(), "dump_vxc: inconsistent \"nbnd\" in coqui mean-field and {}", filename);
        utils::check(npol == mf.npol(), "dump_vxc: inconsistent \"npol\" in coqui mean-field and {}", filename);
      } else {
        sys_grp = grp.create_group("system");
      }

      math::nda::h5_write(sys_grp, "Vxc_skij", dVxc);
    } else {
      math::nda::h5_write(sys_grp, "Vxc_skij", dVxc);
    }
  }
  mpi.comm.barrier();
}

}

#endif
