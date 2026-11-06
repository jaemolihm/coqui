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


#undef NDEBUG

#include "catch2/catch.hpp"
#include "stdio.h"

#include "mpi3/environment.hpp"
#include "mpi3/communicator.hpp"
#include "mpi3/shared_communicator.hpp"
#include "utilities/mpi_context.h"

#include "configuration.hpp"
#include "IO/AppAbort.hpp"
#include "IO/app_loggers.h"
#include "utilities/proc_grid_partition.hpp"

#include "nda/nda.hpp"
#include "numerics/distributed_array/nda.hpp"
#include "numerics/distributed_array/h5.hpp"
#include "numerics/shared_array/nda.hpp"
#include "utilities/test_common.hpp"

#include "mean_field/default_MF.hpp"
#include "mean_field/distributed_orbital_readers.hpp"
#include "hamiltonian/pseudo/pseudopot.h"
#include "hamiltonian/matrix_elements.h"
#include "hamiltonian/one_body_hamiltonian.hpp"
#include "utilities/fortran_utilities.h"
#include "utilities/qe_utilities.hpp"

#include "methods/ERI/eri_utils.hpp"
#include "methods/SCF/mb_solver_t.h"
#include "methods/SCF/scf_common.hpp"
#include "orbitals/orbital_augmenter.h"
#include "mean_field/symmetry/unfold_bz.h"
#include "utilities/kpoint_utils.hpp"

namespace bdft_tests
{

using namespace math::nda;
template <int Rank> using shape_t = std::array<long, Rank>;
using mpi_context_t = utils::mpi_context_t<mpi3::communicator,mpi3::shared_communicator>;
using math::shm::make_shared_array;
using math::shm::shared_array;
using array_view_4d_t = nda::array_view<ComplexType, 4>;
using array_view_3d_t = nda::array_view<ComplexType, 3>;

template<MEMORY_SPACE MEM>
void qe_one_body_components(mpi_context_t& mpi_context, mf::MF& mfobj)
{
  auto wfc_g = mfobj.wfc_truncated_grid();
  auto& world = mpi_context.comm;

  using larray = memory::array<MEM, ComplexType,4>;
  // psi(is,ik,ia,g)
  auto psi = mf::read_distributed_orbital_set_ibz<larray>(mfobj,world,'w');
  // (is, ik, ia, g)
  auto hpsi = make_distributed_array<larray>(world,psi.grid(),psi.global_shape(),psi.block_size());  
  hpsi.local() = ComplexType{0.0};

  long npol = mfobj.npol();
  long nbnd = mfobj.nbnd();

  long nspin = psi.global_shape()[0];
  long nkpts = psi.global_shape()[1];
  auto k_range = nda::range(nkpts);
  auto b_range = nda::range(nbnd);

  auto Hij = make_distributed_array<larray>(world,psi.grid(),{nspin,nkpts,nbnd,nbnd},
                 {psi.block_size()[0],psi.block_size()[1],psi.block_size()[2],psi.block_size()[2]});  
  Hij.local() = ComplexType(0.0);

  // kinetic part
  hamilt::add_kinetic(npol,mfobj.kpts_ibz(),*wfc_g,psi,hpsi);

  // non-local PP
  hamilt::pseudopot Vnl(mfobj);
  Vnl.add_Vpp(world,k_range,b_range,psi,hpsi,Hij);

  // Hij = conj(psi(i,:)) * hpsi(j,:) 
  math::nda::slate_ops::multiply(ComplexType(1.0),psi,math::nda::dagger(hpsi),
                                 ComplexType(1.0),Hij);
  nda::tensor::scale(ComplexType(1.0),Hij.local(),nda::tensor::op::CONJ); 

  RealType E1 = 0.0;
  auto b_rng = Hij.local_range(3);
  auto Hloc = nda::to_host(Hij.local());
  for( auto [is,s] : itertools::enumerate(Hij.local_range(0)) )  
    for( auto [ik,k] : itertools::enumerate(Hij.local_range(1)) ) { 
      auto xk = mfobj.k_weight(k); 
      for( auto [ia,a] : itertools::enumerate(Hij.local_range(2)) )  {
        if(a >= b_rng.first() and a < b_rng.last()) {
          auto w = mfobj.occ(s,k,a);  
          E1 += xk*w*std::real(Hloc(is,ik,ia,a-b_rng.first()));
        }
      }
    }
  if(nspin==1 and npol==1) E1 *= 2.0; 
  E1 = world.all_reduce_value(E1);
  app_log(2,"One body energy: {} Ha",E1);
}

template<MEMORY_SPACE MEM>
void test_H0(mpi_context_t& mpi, mf::MF& mfobj, std::optional<double> Eref)
{
  hamilt::pseudopot V(mfobj);
  auto Hij = hamilt::H0<MEM>(mfobj,mpi.comm,&V,nda::range(mfobj.nkpts_ibz()), nda::range(mfobj.nbnd()));
  RealType E1 = 0.0;
  auto b_rng = Hij.local_range(3);
  auto Hloc = nda::to_host(Hij.local());
  for( auto [is,s] : itertools::enumerate(Hij.local_range(0)) )
    for( auto [ik,k] : itertools::enumerate(Hij.local_range(1)) ) {
      auto xk = mfobj.k_weight(k); 
      for( auto [ia,a] : itertools::enumerate(Hij.local_range(2)) )  {
        if(a >= b_rng.first() and a < b_rng.last()) {
          auto w = mfobj.occ(s,k,a);
          E1 += xk*w*std::real(Hloc(is,ik,ia,a-b_rng.first()));
        }
      }
    }
  if(mfobj.nspin()==1 and mfobj.npol()==1) E1 *= 2.0;
  E1 = mpi.comm.all_reduce_value(E1);
  if( Eref.has_value() ) {
    utils::VALUE_EQUAL(E1, *Eref, 1e-5); 
  } else {
    app_log(2,"One body energy: {} Ha",E1);
  }
}

template<MEMORY_SPACE MEM>
void test_H(mpi_context_t& mpi, mf::MF& mfobj, std::optional<double> Eref)
{ 
  decltype(nda::range::all) all;
  
  {
    memory::array<MEM,ComplexType,3> nii(mfobj.nspin(),mfobj.nkpts_ibz(),mfobj.nbnd());
    if constexpr(MEM==HOST_MEMORY)  {
      nii() = mfobj.occ()(all,nda::range(mfobj.nkpts_ibz()),all);
    } else {
      nii() = nda::array<ComplexType,3>(mfobj.occ()(all,nda::range(mfobj.nkpts_ibz()),all));
    }
    hamilt::pseudopot V(mfobj);
    auto Hij = hamilt::H<MEM>(mfobj,mpi.comm,&V,nii);
    RealType E1 = 0.0;
    auto b_rng = Hij.local_range(3);
    auto Hloc = nda::to_host(Hij.local());
    for( auto [is,s] : itertools::enumerate(Hij.local_range(0)) ) 
      for( auto [ik,k] : itertools::enumerate(Hij.local_range(1)) ) {
        auto xk = mfobj.k_weight(k); 
        for( auto [ia,a] : itertools::enumerate(Hij.local_range(2)) )  {
          if(a >= b_rng.first() and a < b_rng.last()) {
            auto w = mfobj.occ(s,k,a);
            E1 += xk*w*std::real(Hloc(is,ik,ia,a-b_rng.first()));
          }
        }
      }
    if(mfobj.nspin()==1 and mfobj.npol()==1) E1 *= 2.0;
    E1 = mpi.comm.all_reduce_value(E1);
    if( Eref.has_value() ) {
      utils::VALUE_EQUAL(E1, *Eref, 1e-5); 
    } else {
      app_log(2,"One body + hartree: {} Ha",E1);
    }
  }
  {
    auto occ = mfobj.occ()(all,nda::range(mfobj.nkpts_ibz()),all);
    memory::array<MEM,ComplexType,4> nij(mfobj.nspin(),mfobj.nkpts_ibz(),mfobj.nbnd(),mfobj.nbnd());
    nij()=ComplexType(0.0);
    if constexpr(MEM==HOST_MEMORY)  {
      for(int is=0; is<mfobj.nspin(); ++is)
        for(int ik=0; ik<mfobj.nkpts_ibz(); ++ik) {
          auto nii = nij(is,ik,all,all);
          nda::diagonal(nii) = occ(is,ik,all);
        }
    } else {
      nda::array<ComplexType,4> nij_h(mfobj.nspin(),mfobj.nkpts_ibz(),mfobj.nbnd(),mfobj.nbnd());
      nij_h()=ComplexType(0.0);
      for(int is=0; is<mfobj.nspin(); ++is)
        for(int ik=0; ik<mfobj.nkpts_ibz(); ++ik) {
          auto nii = nij_h(is,ik,all,all);
          nda::diagonal(nii) = occ(is,ik,all);
        }
      nij = nij_h;
    }
    hamilt::pseudopot V(mfobj);
    auto Hij = hamilt::H<MEM>(mfobj,mpi.comm,&V,nij);
    RealType E1 = 0.0;
    auto b_rng = Hij.local_range(3);
    auto Hloc = nda::to_host(Hij.local());
    for( auto [is,s] : itertools::enumerate(Hij.local_range(0)) )
      for( auto [ik,k] : itertools::enumerate(Hij.local_range(1)) ) {
        auto xk = mfobj.k_weight(k);
        for( auto [ia,a] : itertools::enumerate(Hij.local_range(2)) )  {
          if(a >= b_rng.first() and a < b_rng.last()) {
            auto w = mfobj.occ(s,k,a);
            E1 += xk*w*std::real(Hloc(is,ik,ia,a-b_rng.first()));
          }
        }
      }
    if(mfobj.nspin()==1 and mfobj.npol()==1) E1 *= 2.0;
    E1 = mpi.comm.all_reduce_value(E1);
    if( Eref.has_value() ) {
      utils::VALUE_EQUAL(E1, *Eref, 1e-5); 
    } else {
      app_log(2,"One body + hartree: {} Ha",E1);
    }
  }
}

void get_density_ovlp(mpi_context_t& mpi, mf::MF &mf,
                      hamilt::pseudopot& psp, double beta,
                      shared_array<array_view_4d_t> &sDm_skij,
                      shared_array<array_view_4d_t> &sS_skij) {
  double mu = 0.0;
  {
    auto sHeff_skij = make_shared_array<array_view_4d_t>(
        mpi.comm, mpi.internode_comm, mpi.node_comm, {mf.nspin(), mf.nkpts_ibz(), mf.nbnd(), mf.nbnd()});
    auto sMO_skia = make_shared_array<array_view_4d_t>(
        mpi.comm, mpi.internode_comm, mpi.node_comm, {mf.nspin(), mf.nkpts_ibz(), mf.nbnd(), mf.nbnd()});
    auto sE_ska = make_shared_array<array_view_3d_t>(
        mpi.comm, mpi.internode_comm, mpi.node_comm, {mf.nspin(), mf.nkpts_ibz(), mf.nbnd()});
    hamilt::set_ovlp(mf, sS_skij);
    hamilt::set_fock(mf, &psp, sHeff_skij, false);

    // Obtains MO coefficients and energies from the given mean-field object
    methods::update_MOs(sMO_skia, sE_ska, sHeff_skij, sS_skij);
    mu = methods::update_mu(mu, mf, sE_ska, beta);
    methods::update_Dm(sDm_skij, sMO_skia, sE_ska, mu, beta);
  }
  mpi.comm.barrier();
}

template<MEMORY_SPACE MEM>
void check_F(mpi_context_t& mpi, mf::MF& mfobj) {

  // checks that F = H0 + J + Vxc
  hamilt::pseudopot V(mfobj);
  auto all = nda::range::all;
  long nspin = mfobj.nspin();
  long nkpts_ibz = mfobj.nkpts_ibz();
  long nbnd = mfobj.nbnd();
  auto Vxc = hamilt::Vxc<MEM>(mfobj, mpi.comm);
  auto H0 = hamilt::H0<MEM>(mfobj, mpi.comm, &V); 

  memory::array<MEM,ComplexType,3> occ(nspin,nkpts_ibz,nbnd);
  occ() = mfobj.occ()(all,nda::range(nkpts_ibz),all); 
  auto dJ = hamilt::Vhartree<MEM>(mfobj, mpi.comm, &V, occ);

  auto sF0 = make_shared_array<array_view_4d_t>(
      mpi.comm, mpi.internode_comm, mpi.node_comm, {nspin, nkpts_ibz, nbnd, nbnd});
  math::nda::gather_to_shm(H0, sF0);

  auto sT = make_shared_array<array_view_4d_t>(
      mpi.comm, mpi.internode_comm, mpi.node_comm, {nspin, nkpts_ibz, nbnd, nbnd});
  math::nda::gather_to_shm(dJ, sT);
  if(mpi.node_comm.root()) sF0.local() += ( sT.local() ); 
  mpi.node_comm.barrier();

  math::nda::gather_to_shm(Vxc, sT);
  if(mpi.node_comm.root()) sF0.local() += ( sT.local() ); 
  mpi.node_comm.barrier();

  auto F = hamilt::F<MEM>(mfobj, mpi.comm, nda::range(nkpts_ibz),nda::range(nbnd));
  auto sF = make_shared_array<array_view_4d_t>(
      mpi.comm, mpi.internode_comm, mpi.node_comm, {nspin, nkpts_ibz, nbnd, nbnd});
  math::nda::gather_to_shm(F, sF);
  mpi.node_comm.barrier();

  auto Abs = nda::map([](ComplexType _x_) { return std::abs(_x_); });
  double norm = -1;
  if (sF.node_comm()->root()) {
    nda::array<RealType,4> res_abs(nspin, nkpts_ibz, nbnd, nbnd);
    res_abs = Abs(sF.local() - sF0.local());
    norm = nda::max_element(res_abs);
  }
  sF.node_comm()->broadcast_n(&norm, 1, 0);
  app_log(2, "Norm of J - J2 = {}", norm);

  utils::ARRAY_EQUAL(sF.local(), sF0.local(), 1e-4);
}

// Correctness gate for the augmented DFT V_Hxc seed. On a NON-augmented (KS
// eigenstate) basis the seed
//   H_KS = H0 + <phi|svsc - svloc|phi> = H0 + V_H + V_xc
// must reproduce diag(KS eigenvalues) = hamilt::F(). This validates the
// svsc - svloc potential subtraction and the add_vloc projection independently of
// augmentation, using the eigenvalues as an external ground truth.
template<MEMORY_SPACE MEM>
void check_KS_seed(mpi_context_t& mpi, mf::MF& mfobj) {

  hamilt::pseudopot V(mfobj);
  long nspin = mfobj.nspin();
  long nkpts_ibz = mfobj.nkpts_ibz();
  long nbnd = mfobj.nbnd();

  // H0 + V_Hxc_aug, gathered into a shared array
  auto H0 = hamilt::H0<MEM>(mfobj, mpi.comm, &V);
  auto dHxc = hamilt::V_Hxc_aug<MEM>(mfobj, mpi.comm, &V);

  auto sSeed = make_shared_array<array_view_4d_t>(mpi, {nspin, nkpts_ibz, nbnd, nbnd});
  math::nda::gather_to_shm(H0, sSeed);
  auto sT = make_shared_array<array_view_4d_t>(mpi, {nspin, nkpts_ibz, nbnd, nbnd});
  math::nda::gather_to_shm(dHxc, sT);
  if(mpi.node_comm.root()) sSeed.local() += sT.local();
  mpi.node_comm.barrier();

  // reference: diag(KS eigenvalues)
  auto F = hamilt::F<MEM>(mfobj, mpi.comm, nda::range(nkpts_ibz), nda::range(nbnd));
  auto sF = make_shared_array<array_view_4d_t>(mpi, {nspin, nkpts_ibz, nbnd, nbnd});
  math::nda::gather_to_shm(F, sF);
  mpi.node_comm.barrier();

  auto Abs = nda::map([](ComplexType _x_) { return std::abs(_x_); });
  double norm = -1;
  if (sF.node_comm()->root()) {
    nda::array<RealType,4> res_abs(nspin, nkpts_ibz, nbnd, nbnd);
    res_abs = Abs(sSeed.local() - sF.local());
    norm = nda::max_element(res_abs);
  }
  sF.node_comm()->broadcast_n(&norm, 1, 0);
  app_log(2, "Norm of (H0 + V_Hxc_aug) - diag(eigval) = {}", norm);

  utils::ARRAY_EQUAL(sSeed.local(), sF.local(), 1e-4);
}

// should only be called with HF calculations or hybrid DFT
template<MEMORY_SPACE MEM>
void check_K(mpi_context_t& mpi, std::shared_ptr<mf::MF> &mfobj, double x) {

  // checks that F = H0 + J + K + Vxc
  hamilt::pseudopot V(*mfobj);
  auto all = nda::range::all;
  long nspin = mfobj->nspin();
  long nkpts_ibz = mfobj->nkpts_ibz();
  long nbnd = mfobj->nbnd();
  auto mfocc = mfobj->occ();

  auto sK = make_shared_array<array_view_4d_t>(mpi, {nspin, nkpts_ibz, nbnd, nbnd});
  {
    auto F = hamilt::F<MEM>(*mfobj, mpi.comm);
    auto H0 = hamilt::H0<MEM>(*mfobj, mpi.comm, &V);
    auto Vxc = hamilt::Vxc<MEM>(*mfobj, mpi.comm);

    nda::array<ComplexType,3> occ(nspin,nkpts_ibz,nbnd);
    occ() = mfocc(all,nda::range(nkpts_ibz),all);
    auto J = hamilt::Vhartree<MEM>(*mfobj, mpi.comm, &V, occ);

    //F.local() -= ( H0.local() + J.local() + Vxc.local());
    nda::tensor::add(ComplexType(1.0),H0.local(),ComplexType(1.0),J.local());
    nda::tensor::add(ComplexType(1.0),Vxc.local(),ComplexType(1.0),J.local());
    nda::tensor::add(ComplexType(-1.0),J.local(),ComplexType(1.0),F.local());
    math::nda::gather_to_shm(F, sK);
  }
  mpi.node_comm.barrier();

  auto sK2 = make_shared_array<array_view_4d_t>(
      mpi.comm, mpi.internode_comm, mpi.node_comm, {nspin, nkpts_ibz, nbnd, nbnd});
  {
    methods::thc_reader_t thc(mfobj,
                              methods::make_thc_reader_ptree(0.0, "", "incore", "test.h5", "bdft",
                                                             1e-8, mfobj->ecutrho(), 1, 1024));
    methods::solvers::hf_t hf("ignore_g0");

    auto sS_skij = make_shared_array<array_view_4d_t>(
        mpi.comm, mpi.internode_comm, mpi.node_comm, {nspin, nkpts_ibz, nbnd, nbnd});
    hamilt::set_ovlp(*mfobj, sS_skij);

    nda::array<ComplexType,4> occ4d(nspin,nkpts_ibz,nbnd,nbnd);
    occ4d() = ComplexType(0.0);
    for( int s=0; s<nspin; s++ )
      for( int k=0; k<nkpts_ibz; k++ ) {
        for( int a=0; a<nbnd; a++ )
          occ4d(s,k,a,a) = mfocc(s,k,a);
      }
    hf.evaluate(sK2, occ4d, thc, sS_skij.local(), false, true);
    mpi.node_comm.barrier();
    if(mpi.node_comm.root()) 
      sK2.local() *= ComplexType(x); 
  }
  mpi.node_comm.barrier();

  auto Abs = nda::map([](ComplexType _x_) { return std::abs(_x_); });
  double norm = -1;
  if (sK.node_comm()->root()) {
    nda::array<RealType,4> res_abs(nspin, nkpts_ibz, nbnd, nbnd);
    res_abs = Abs(sK.local() - sK2.local());
    norm = nda::max_element(res_abs);
  }
  sK.node_comm()->broadcast_n(&norm, 1, 0);
  app_log(2, "Norm of K - K2 = {}", norm);

  utils::ARRAY_EQUAL(sK.local(), sK2.local(), 1e-4);
}

template<MEMORY_SPACE MEM>
void check_Vxc(mpi_context_t& mpi, mf::MF& mfobj) {

  hamilt::pseudopot V(mfobj);
  auto all = nda::range::all;
  long nspin = mfobj.nspin();
  long nkpts_ibz = mfobj.nkpts_ibz();
  long nbnd = mfobj.nbnd();

  // Vxc = F - H0 - J
  auto sVxc_skij = make_shared_array<array_view_4d_t>(mpi, {nspin, nkpts_ibz, nbnd, nbnd});
  // Hartree from PW using mf.occ() 
  {
    // MAM: If mf.occ is not meaningful, need to reconstruct occupations consistent with F
    //      This might be hard in practice for metals if the MF code uses smearing
    memory::array<MEM, ComplexType, 3> occ(nspin, nkpts_ibz, nbnd);
    occ() = mfobj.occ()(all,nda::range(nkpts_ibz),all);
    auto J = hamilt::Vhartree<MEM>(mfobj, mpi.comm, &V, occ);
    auto F = hamilt::F<MEM>(mfobj, mpi.comm);
    auto H0 = hamilt::H0<MEM>(mfobj, mpi.comm, &V);

    //F.local() -= ( H0.local() + J.local() );
    nda::tensor::add(ComplexType(1.0),H0.local(),ComplexType(1.0),J.local());
    nda::tensor::add(ComplexType(-1.0),J.local(),ComplexType(1.0),F.local());
    math::nda::gather_to_shm(F, sVxc_skij);
  }

  // Vxc
  auto sVxc2_skij = make_shared_array<array_view_4d_t>(mpi, {nspin, nkpts_ibz, nbnd, nbnd});
  {
    auto Vxc = hamilt::Vxc<MEM>(mfobj, mpi.comm);
    math::nda::gather_to_shm(Vxc, sVxc2_skij);
  }

  auto Abs = nda::map([](ComplexType _x_) { return std::abs(_x_); });
  double norm = -1;
  if (sVxc_skij.node_comm()->root()) {
    nda::array<RealType,4> res_abs(nspin, nkpts_ibz, nbnd, nbnd);
    res_abs = Abs(sVxc_skij.local() - sVxc2_skij.local());
    norm = nda::max_element(res_abs);
  }
  sVxc_skij.node_comm()->broadcast_n(&norm, 1, 0);
  app_log(2, "Norm of Vxc - Vxc2 = {}", norm);

  utils::ARRAY_EQUAL(sVxc_skij.local(), sVxc2_skij.local(), 2e-5);
}

template<MEMORY_SPACE MEM>
void check_Hartree(mpi_context_t& mpi, std::shared_ptr<mf::MF> &mfobj, bool diag_dm=false) {
  hamilt::pseudopot V(*mfobj);
  long nspin = mfobj->nspin();
  long nkpts_ibz = mfobj->nkpts_ibz();
  long nbnd = mfobj->nbnd();
  std::array<long,4> shape = {nspin,nkpts_ibz,nbnd,nbnd};
  auto sDm_skij = make_shared_array<array_view_4d_t>(
      mpi.comm, mpi.internode_comm, mpi.node_comm, shape);
  auto sS_skij = make_shared_array<array_view_4d_t>(
      mpi.comm, mpi.internode_comm, mpi.node_comm, shape);
  get_density_ovlp(mpi, *mfobj, V, 1000, sDm_skij, sS_skij);

  // Hartree from ERI
  auto sJ_skij = make_shared_array<array_view_4d_t>(
      mpi.comm, mpi.internode_comm, mpi.node_comm, shape); 
  methods::thc_reader_t thc(mfobj,
                            methods::make_thc_reader_ptree(0, "", "incore", "", "coqui", 1e-9,
                                                           mfobj->ecutrho(), 1, 1024));
  methods::solvers::hf_t hf;
  hf.evaluate(sJ_skij, sDm_skij.local(), thc, sS_skij.local(), true, false);
  mpi.comm.barrier();

  // Hartree from PW
  auto sJ2_skij = make_shared_array<array_view_4d_t>(
      mpi.comm, mpi.internode_comm, mpi.node_comm, shape); 
  if (diag_dm) {
    memory::array<MEM, ComplexType, 3> Dm_ski(nspin, nkpts_ibz, nbnd);
    for (long s=0; s<mfobj->nspin(); ++s) {
      for (long k=0; k<mfobj->nkpts_ibz(); ++k) {
        auto Dm_sk = sDm_skij.local()(s,k,nda::range::all,nda::range::all);
        Dm_ski(s,k,nda::range::all) = nda::diagonal(Dm_sk);
      }
    }
    auto dJ2 = hamilt::Vhartree<MEM>(*mfobj, mpi.comm, &V, Dm_ski, nda::range(nkpts_ibz),nda::range(nbnd));
    math::nda::gather_to_shm(dJ2, sJ2_skij);
  } else {
    auto dJ2 = hamilt::Vhartree<MEM>(*mfobj, mpi.comm, &V, sDm_skij.local(), nda::range(nkpts_ibz),nda::range(nbnd));
    math::nda::gather_to_shm(dJ2, sJ2_skij);
  }
  sJ2_skij.communicator()->barrier();

  auto Abs = nda::map([](ComplexType _x_) { return std::abs(_x_); });
  double norm = -1;
  if (sJ_skij.node_comm()->root()) {
    nda::array<RealType,4> res_abs(nspin, nkpts_ibz, nbnd, nbnd);
    res_abs = Abs(sJ_skij.local() - sJ2_skij.local());
    norm = nda::max_element(res_abs);
  }
  sJ_skij.node_comm()->broadcast_n(&norm, 1, 0);
  app_log(2, "Norm of J - J2 = {}", norm);

  utils::ARRAY_EQUAL(sJ_skij.local(), sJ2_skij.local(), 1e-5);
}

// MAM: reenable and add more tests!!!
template<MEMORY_SPACE MEM>
void qe_ovlp(mpi_context_t& mpi, mf::MF& mfobj) {
  {
    auto S = hamilt::ovlp<MEM>(mfobj, mpi.comm, nda::range(mfobj.nkpts_ibz()),nda::range(mfobj.nbnd()));
    auto Sloc = memory::to_memory_space<HOST_MEMORY>(S.local());
    for (auto [is, s] : itertools::enumerate(S.local_range(0)))
      for (auto [ik, k] : itertools::enumerate(S.local_range(1))) {
        auto S_ij = Sloc(is, ik, nda::ellipsis{});
        for (auto [ia, a] : itertools::enumerate(S.local_range(2)))
          for (auto [ib, b] : itertools::enumerate(S.local_range(3))) {
            utils::VALUE_EQUAL(S_ij(ia,ib), (a==b?ComplexType(1.0):ComplexType(0.0)));
          }
      }
  }

  {
    auto S = hamilt::ovlp_diagonal<MEM>(mfobj, mpi.comm, nda::range(mfobj.nkpts_ibz()),nda::range(mfobj.nbnd()));
    auto Sloc = memory::to_memory_space<HOST_MEMORY>(S.local());
    for (auto [is, s] : itertools::enumerate(S.local_range(0)))
      for (auto [ik, k] : itertools::enumerate(S.local_range(1))) {
        auto S_ii = Sloc(is, ik, nda::ellipsis{});
        for (auto [ia, a] : itertools::enumerate(S.local_range(2)))
          utils::VALUE_EQUAL(S_ii(ia), ComplexType(1.0));
      }
  } 
}

/*
TEST_CASE("pyscf", "[hamilt]") {
  auto& mpi = utils::make_unit_test_mpi_context();
  auto mfobj = mf::default_MF(mpi_context,mf::pyscf_source);

  auto Fij = hamilt::F<HOST_MEMORY>(mfobj, world, nda::range(mfobj.nkpts_ibz()),nda::range(mfobj.nbnd()));
  auto Hij = hamilt::H0<HOST_MEMORY>(mfobj, world, nda::range(mfobj.nkpts_ibz()),nda::range(mfobj.nbnd()));
  auto Sij = hamilt::ovlp<HOST_MEMORY>(mfobj, world, nda::range(mfobj.nkpts_ibz()),nda::range(mfobj.nbnd()));

  REQUIRE( Sij.global_shape() == shape_t<4>{mfobj.nspin(), mfobj.nkpts_ibz(), mfobj.nbnd(), mfobj.nbnd()} );
  REQUIRE( Hij.global_shape() == shape_t<4>{mfobj.nspin(), mfobj.nkpts_ibz(), mfobj.nbnd(), mfobj.nbnd()} );
  REQUIRE( Fij.global_shape() == shape_t<4>{mfobj.nspin(), mfobj.nkpts_ibz(), mfobj.nbnd(), mfobj.nbnd()} );
}

TEST_CASE("pyscf_ovlp", "[hamilt]") {
  auto& mpi = utils::make_unit_test_mpi_context();

  auto [outdir,prefix] = utils::utest_filename(mf::pyscf_source);
  auto mfobj = mf::make_MF(mpi_context, mf::pyscf_source, outdir, prefix);
  auto Sij = hamilt::ovlp(mfobj, world, nda::range(mfobj.nkpts_ibz()), nda::range(mfobj.nbnd()));

  // read AO
  { // read a single orbital
    nda::array<ComplexType, 2> Orb(2,mfobj.max_npw());
    long is = Sij.local_range(0).first();
    long ik = Sij.local_range(1).first();
    long ia = Sij.local_range(2).first();
    long ib = Sij.local_range(3).first();
    mfobj.get_orbital('k',is,ik,ia, Orb(0,nda::range::all));
    mfobj.get_orbital('k',is,ik,ib, Orb(1,nda::range::all));
    ComplexType overlap = 0.0;
    for( auto [va,vb] : itertools::zip(Orb(0,nda::range::all), Orb(1,nda::range::all)) )
      overlap += va * std::conj(vb);

    auto S = Sij.local();
    utils::VALUE_EQUAL(overlap, S(0,0,0,0));
    mfobj.close();
  }

  if(Sij.local_range(2).size() >= 2 and Sij.local_range(3).size() >= 2)
  { // read a set of orbitals
    nda::array<ComplexType, 3> psi_a(1,2,mfobj.fft_grid_size());
    nda::array<ComplexType, 3> psi_b(1,2,mfobj.fft_grid_size());
    long is = Sij.local_range(0).first();
    long ik = Sij.local_range(1).first();
    long ia = Sij.local_range(2).first();
    long ib = Sij.local_range(3).first();
    // orbitals for k=[0,1), i=[0,2)
    mfobj.get_orbital_set('g', is, {ik, ik+1}, {ia, ia+2}, psi_a);
    mfobj.get_orbital_set('g', is, {ik, ik+1}, {ib, ib+2}, psi_b);

    nda::array<ComplexType, 2> Orbs_C(2,mfobj.max_npw());
    nda::array<ComplexType, 2> ov(2,2);
    nda::blas::gemm(1.0, psi_a(0,nda::ellipsis{}), nda::dagger(psi_b(0,nda::ellipsis{})), 0.0, ov);

    auto S = Sij.local();
    for(int i=0; i<2; ++i)
      for(int j=0; j<2; ++j)
        utils::VALUE_EQUAL(ov(i,j), S(0,0,i,j), 1e-12);
    mfobj.close();
  } else {
    app_warning("Too many processors in pyscf_ovlp unit test. Skipping.");
  }

  // Read orbitals into a distributed array
  {
    using local_Array_t = memory::array<HOST_MEMORY, ComplexType, 2>;
    auto dPsia = mf::read_distributed_orbital_set<local_Array_t>(mfobj, world, 'k', {1, world.size()});

    REQUIRE(dPsia.global_shape()[0] == mfobj.nspin()*mfobj.nkpts()*mfobj.nbnd());
    REQUIRE(dPsia.global_shape()[1] == mfobj.max_npw());
  }

}
*/

// Unfolded mean field written by mf::unfold_mf, whose nonlocal projectors are copied from the full-BZ projectors of
// pw2coqui (Hamiltonian/<pp_type>/fullbz, Miller indices shifted by -G0).
// (1) H0 of the unfolded mean field (set_H0 on the unfolded mean field, with its own pseudopotential) against H0 of the IBZ system, copied from
//     kp_to_ibz(k) and conjugated at time-reversal k. Without the conjugation at time-reversal k the comparison
//     fails at the size of Im H0 (qe_lih223_sym and qe_si333_sym have time-reversal k, qe_lih222_sym does not).
// (2) The per-atom projector weight N_b(k) = sum_{n in occ} sum_j |<beta_{b,j}|phi_{n,k}>|^2 of the unfolded mean field against that
//     of a no-symmetry calculation of the same system (separate QE run), k points matched by crystal coordinates. N_b
//     is invariant under unitary rotations of the occupied manifold.
// (3) An augmented IBZ mean field (momentum augmentation of qe_lih223_sym), whose h5 has no full-BZ projectors, is
//     unfolded with projector_file = the pw2coqui h5 of its parent: the view is augmented, carries the IBZ band
//     weights (expanded by kp_to_ibz), its stored Orbitals/H_KS_skij is the IBZ one copied at kp_to_ibz(k) and
//     conjugated at time-reversal k, and H0 + V_Hxc rebuilt on the view reproduces that stored matrix.
TEST_CASE("unfold_mf_projectors", "[hamilt]") {
  auto& mpi = utils::make_unit_test_mpi_context();
  auto& comm = mpi->comm;

  auto check = [&](mf::MF& mf_sym, std::string prefix, bool has_trev, std::string nosym_name, double tol_H0,
                   std::string projector_file = "") {
    long nspin = mf_sym.nspin(), nkpts = mf_sym.nkpts(), nkpts_ibz = mf_sym.nkpts_ibz(), nbnd = mf_sym.nbnd();
    long nat = mf_sym.number_of_atoms();

    // Unfolded mean field; H0 of the IBZ system and of the unfolded mean field, each with its own pseudopotential
    mf::unfold_mf(mf_sym, prefix + ".h5", projector_file);
    comm.barrier();
    auto mf_unfold = mf::make_MF(mpi, mf::bdft_source, "./", prefix, mf::h5_input_type);

    // (3) augmented mean field: flag, band weights, stored H_KS, H0 + V_Hxc
    if (mf_sym.is_augmented()) {
      REQUIRE(mf_unfold.is_augmented());
      REQUIRE(mf_unfold.has_hks_matrix());
      REQUIRE(mf_unfold.augment_type() == mf_sym.augment_type());
      auto w_sym = mf_sym.augmented_band_weights();
      auto w_unfold = mf_unfold.augmented_band_weights();
      double err_w = 0.0;
      for (long is = 0; is < nspin; ++is)
        for (long ik = 0; ik < nkpts; ++ik)
          for (long i = 0; i < nbnd; ++i)
            err_w = std::max(err_w, std::abs(w_unfold(is,ik,i) - w_sym(is,mf_sym.kp_to_ibz(ik),i)));
      app_log(2, "unfold_mf_projectors {}: max|w_unfold - w_ibz(kp_to_ibz)| = {}", prefix, err_w);
      REQUIRE(err_w == 0.0);

      nda::array<ComplexType,4> Hks_sym(nspin, nkpts_ibz, nbnd, nbnd), Hks_unfold(nspin, nkpts, nbnd, nbnd);
      if (comm.root()) {
        mf_sym.read_hks_matrix(Hks_sym());
        mf_unfold.read_hks_matrix(Hks_unfold());
      }
      comm.broadcast_n(Hks_sym.data(), Hks_sym.size(), 0);
      comm.broadcast_n(Hks_unfold.data(), Hks_unfold.size(), 0);
      double err_hks = 0.0, err_hks_noconj = 0.0, nrm_hks = 0.0;
      for (long is = 0; is < nspin; ++is)
        for (long ik = 0; ik < nkpts; ++ik) {
          long iks = mf_sym.kp_to_ibz(ik);
          bool tr = mf_sym.kp_trev(ik);
          for (long i = 0; i < nbnd; ++i)
            for (long j = 0; j < nbnd; ++j) {
              ComplexType ref = tr ? std::conj(Hks_sym(is,iks,i,j)) : Hks_sym(is,iks,i,j);
              err_hks = std::max(err_hks, std::abs(Hks_unfold(is,ik,i,j) - ref));
              nrm_hks = std::max(nrm_hks, std::abs(ref));
              if (tr) err_hks_noconj = std::max(err_hks_noconj, std::abs(Hks_unfold(is,ik,i,j) - Hks_sym(is,iks,i,j)));
            }
        }
      app_log(2, "unfold_mf_projectors {}: max|H_KS_unfold - H_KS_ibz(copy/conj)| = {}, without conj at TR k = {}, "
              "max|H_KS| = {}", prefix, err_hks, err_hks_noconj, nrm_hks);
      REQUIRE(err_hks < 1e-12);
      if (has_trev) REQUIRE(err_hks_noconj > 1e-3);

      // H0 + V_Hxc rebuilt on the view, hermitized as in the augmenter (build_ks_matrix_ibz), against its stored H_KS
      auto psp_u = hamilt::make_pseudopot(mf_unfold);
      auto sHKS = make_shared_array<array_view_4d_t>(*mpi, shape_t<4>{nspin, nkpts, nbnd, nbnd});
      hamilt::set_H0(mf_unfold, psp_u.get(), sHKS);
      auto dV = hamilt::V_Hxc_aug<HOST_MEMORY>(mf_unfold, comm, psp_u.get());
      auto sV = make_shared_array<array_view_4d_t>(*mpi, shape_t<4>{nspin, nkpts, nbnd, nbnd});
      math::nda::gather_to_shm(dV, sV);
      comm.barrier();
      double err_rebuild = 0.0;
      if (mpi->node_comm.root()) {
        auto H = sHKS.local();
        auto V = sV.local();
        for (long is = 0; is < nspin; ++is)
          for (long ik = 0; ik < nkpts; ++ik)
            for (long i = 0; i < nbnd; ++i)
              for (long j = 0; j < nbnd; ++j) {
                ComplexType h = H(is,ik,i,j) + V(is,ik,i,j);
                ComplexType hh = H(is,ik,j,i) + V(is,ik,j,i);
                err_rebuild = std::max(err_rebuild, std::abs(0.5*(h + std::conj(hh)) - Hks_unfold(is,ik,i,j)));
              }
      }
      comm.all_reduce_in_place_n(&err_rebuild, 1, boost::mpi3::max<>{});
      app_log(2, "unfold_mf_projectors {}: max|(H0+V_Hxc)_unfold - H_KS_unfold(stored)| = {}", prefix, err_rebuild);
      REQUIRE(err_rebuild < tol_H0);
    }

    auto psp_sym = hamilt::make_pseudopot(mf_sym);
    auto psp_unfold = hamilt::make_pseudopot(mf_unfold);
    auto sH0_sym = make_shared_array<array_view_4d_t>(*mpi, shape_t<4>{nspin, nkpts_ibz, nbnd, nbnd});
    auto sH0_unfold = make_shared_array<array_view_4d_t>(*mpi, shape_t<4>{nspin, nkpts, nbnd, nbnd});
    hamilt::set_H0(mf_sym, psp_sym.get(), sH0_sym);
    hamilt::set_H0(mf_unfold, psp_unfold.get(), sH0_unfold);

    // k_weight of the unfolded mean field: 1/nkpts for every k
    REQUIRE(mf_unfold.k_weight().size() == nkpts);
    for (long ik = 0; ik < nkpts; ++ik) REQUIRE(std::abs(mf_unfold.k_weight(ik) - 1.0/double(nkpts)) < 1e-14);
    REQUIRE(mf_unfold.occ().shape(1) == nkpts);

    // (1) H0 against the IBZ H0 copied from kp_to_ibz(k), conjugated at time-reversal k
    double err = 0.0, err_noconj = 0.0, nrm = 0.0;
    if (mpi->node_comm.root()) {
      auto H_sym = sH0_sym.local();
      auto H_unfold = sH0_unfold.local();
      for (long is = 0; is < nspin; ++is)
        for (long ik = 0; ik < nkpts; ++ik) {
          long iks = mf_sym.kp_to_ibz(ik);
          bool tr = mf_sym.kp_trev(ik);
          for (long i = 0; i < nbnd; ++i)
            for (long j = 0; j < nbnd; ++j) {
              ComplexType ref = tr ? std::conj(H_sym(is, iks, i, j)) : H_sym(is, iks, i, j);
              err = std::max(err, std::abs(H_unfold(is, ik, i, j) - ref));
              nrm = std::max(nrm, std::abs(ref));
              // what copying without the conjugation at time-reversal k would give
              if (tr) err_noconj = std::max(err_noconj, std::abs(H_unfold(is, ik, i, j) - H_sym(is, iks, i, j)));
            }
        }
    }
    comm.all_reduce_in_place_n(&err, 1, boost::mpi3::max<>{});
    comm.all_reduce_in_place_n(&err_noconj, 1, boost::mpi3::max<>{});
    comm.all_reduce_in_place_n(&nrm, 1, boost::mpi3::max<>{});

    // (2) per-atom projector weights against the no-symmetry system (eph_projector_overlaps needs h5 input)
    double err_N = 0.0, nrm_N = 0.0;
    if (not nosym_name.empty()) {
      auto mf_nosym = mf::default_MF(mpi, nosym_name, mf::h5_input_type);
      REQUIRE(mf_nosym.nkpts() == nkpts);
      REQUIRE(mf_nosym.number_of_atoms() == nat);
      auto psp_nosym = hamilt::make_pseudopot(mf_nosym);
      nda::array<ComplexType,5> P_nosym, P_unfold;
      psp_unfold->eph_projector_overlaps(mf_unfold, P_unfold);
      psp_nosym->eph_projector_overlaps(mf_nosym, P_nosym);
      auto ofs = psp_nosym->proj_offset();
      auto nh = psp_nosym->proj_per_species();
      auto ityp = psp_nosym->species_of_atom();
      long nocc = 0;
      for (long n = 0; n < nbnd; ++n) if (mf_nosym.occ()(0,0,n) > 0.5) ++nocc;
      REQUIRE(nocc > 0);
      auto weight = [&](nda::array<ComplexType,5> const& P, long ik, long b) {
        double N = 0.0;
        for (long n = 0; n < nocc; ++n)
          for (long j = 0; j < nh(ityp(b)); ++j) N += std::norm(P(0, 0, ik, ofs(b)+j, n));
        return N;
      };
      for (long ik = 0; ik < nkpts; ++ik) {
        // k point of the no-symmetry system equal to the unfolded mean field's k modulo a reciprocal lattice vector
        long ikn = utils::find_kpoint(mf_nosym.kpts_crystal(), mf_unfold.kpts_crystal(ik, 0),
                                      mf_unfold.kpts_crystal(ik, 1), mf_unfold.kpts_crystal(ik, 2));
        REQUIRE(ikn >= 0);
        for (long b = 0; b < nat; ++b) {
          double N_unfold = weight(P_unfold, ik, b), Nn = weight(P_nosym, ikn, b);
          err_N = std::max(err_N, std::abs(N_unfold - Nn));
          nrm_N = std::max(nrm_N, Nn);
        }
      }
    }

    // Report and compare
    app_log(2, "unfold_mf_projectors {}: max|H0_unfold - H0_ibz(copy/conj)| = {}, without conj at TR k = {}, max|H0| = {}; "
            "per-atom weight vs no-symmetry: err = {}, max N = {}",
            prefix, err, err_noconj, nrm, err_N, nrm_N);
    REQUIRE(err < tol_H0);
    if (has_trev) REQUIRE(err_noconj > 1e-3);
    if (not nosym_name.empty()) {
      REQUIRE(nrm_N > 1e-3);
      REQUIRE(err_N < 1e-7); // two QE runs; measured 4e-9 (cause not verified)
    }
    comm.barrier();
    if (comm.root()) remove((prefix + ".h5").c_str());
    comm.barrier();
  };

  // The IBZ projectors of the LiH fixtures were written by an older pw2coqui build than their fullbz group: the two
  // differ by 4.1e-8 in relative L2 norm (the fullbz group is identical to the new build's IBZ projectors to 6e-17),
  // so H0 agrees at 4.8e-8 instead of round-off. The Si fixture agrees to 8e-16.
  const double tol_H0_lih = 1e-7;
  SECTION("qe_h5_lih222") {
    auto mf_sym = mf::default_MF(mpi, "qe_lih222_sym", mf::h5_input_type);
    check(mf_sym, "unfold_mf_H0_h5_222", false, "qe_lih222", tol_H0_lih);
  }
  SECTION("qe_h5") {
    auto mf_sym = mf::default_MF(mpi, "qe_lih223_sym", mf::h5_input_type);
    check(mf_sym, "unfold_mf_H0_h5", true, "qe_lih223", tol_H0_lih);
  }
  SECTION("qe_h5_aug") {
    auto mf_parent = mf::default_MF(mpi, "qe_lih223_sym", mf::h5_input_type);
    auto augmenter = std::make_shared<orbitals::momentum_augmenter>(mf_parent);
    // 3 bands = complete multiplets at every IBZ k of this fixture (see aug_ibz_symmetry in test_orbitals.cpp)
    auto mf_sym = orbitals::add_augmentation<HOST_MEMORY>(mf_parent, "unfold_mf_aug_ibz.h5", augmenter, 3, 1e-3, 1e-1);
    REQUIRE(mf_sym.is_augmented());
    auto [outdir, pf] = utils::utest_filename("qe_lih223_sym");
    check(mf_sym, "unfold_mf_H0_aug", true, "", tol_H0_lih, outdir + pf + ".coqui.h5");
    comm.barrier();
    if (comm.root()) remove("unfold_mf_aug_ibz.h5");
    comm.barrier();
  }
  SECTION("qe_h5_si333") {
    auto mf_sym = mf::default_MF(mpi, "qe_si333_sym", mf::h5_input_type);
    check(mf_sym, "unfold_mf_H0_si333", true, "", 1e-12);
  }
}

template<MEMORY_SPACE MEM>
void test_F_impl(std::shared_ptr<mpi_context_t> &mpi)
{
  SECTION("lih223") 
  {
    auto qe_h5 = mf::default_MF(mpi, "qe_lih223", mf::h5_input_type);
    check_F<MEM>(*mpi, qe_h5);
  }

  SECTION("lih223_inv") 
  {
    auto qe_h5 = mf::default_MF(mpi, "qe_lih223_inv", mf::h5_input_type);
    check_F<MEM>(*mpi, qe_h5);
  }

  SECTION("lih223_sym") 
  {
    auto qe_h5 = mf::default_MF(mpi, "qe_lih223_sym", mf::h5_input_type);
    check_F<MEM>(*mpi, qe_h5);
  }

  SECTION("GaAs222_so") 
  {
    auto qe_h5 = mf::default_MF(mpi, "qe_GaAs222_so", mf::h5_input_type);
    check_F<MEM>(*mpi, qe_h5);
  }
}

TEST_CASE("mf_F", "[hamilt]") {
  auto& mpi = utils::make_unit_test_mpi_context();

  test_F_impl<HOST_MEMORY>(mpi);
#if defined(ENABLE_DEVICE)
  test_F_impl<DEVICE_MEMORY>(mpi);
  test_F_impl<UNIFIED_MEMORY>(mpi);
#endif
}

template<MEMORY_SPACE MEM>
void test_ks_seed_impl(std::shared_ptr<mpi_context_t> &mpi)
{
  SECTION("lih223")
  {
    auto qe_h5 = mf::default_MF(mpi, "qe_lih223", mf::h5_input_type);
    check_KS_seed<MEM>(*mpi, qe_h5);
  }

  SECTION("lih223_inv")
  {
    auto qe_h5 = mf::default_MF(mpi, "qe_lih223_inv", mf::h5_input_type);
    check_KS_seed<MEM>(*mpi, qe_h5);
  }

  SECTION("lih223_sym")
  {
    auto qe_h5 = mf::default_MF(mpi, "qe_lih223_sym", mf::h5_input_type);
    check_KS_seed<MEM>(*mpi, qe_h5);
  }

  SECTION("GaAs222_so")
  {
    auto qe_h5 = mf::default_MF(mpi, "qe_GaAs222_so", mf::h5_input_type);
    check_KS_seed<MEM>(*mpi, qe_h5);
  }
}

TEST_CASE("ks_seed", "[hamilt]") {
  auto& mpi = utils::make_unit_test_mpi_context();

  test_ks_seed_impl<HOST_MEMORY>(mpi);
#if defined(ENABLE_DEVICE)
  test_ks_seed_impl<DEVICE_MEMORY>(mpi);
  test_ks_seed_impl<UNIFIED_MEMORY>(mpi);
#endif
}

// Read path of the augmented Kohn-Sham matrix, through set_fock and its two
// exclude_H0 branches. The fixture is a zero-augmentation basis, which IS the
// parent eigenbasis, so the stored matrix has an external ground truth:
// diag(eps_QE). This also covers the LR "mf_dft" G0 seed, which is get_mf_MOs on
// top of the same set_fock call.
TEST_CASE("aug_hks_set_fock", "[hamilt]") {
  auto& mpi = utils::make_unit_test_mpi_context();
  auto qe_mf = mf::default_MF(mpi, "qe_lih223", mf::h5_input_type);

  auto augmenter = std::make_shared<orbitals::momentum_augmenter>(qe_mf);
  auto aug_mf = orbitals::add_augmentation<HOST_MEMORY>(qe_mf, "dummy_hks.h5", augmenter,
                                                        0, 1e-6, 1e-2);
  REQUIRE(aug_mf.is_augmented());
  REQUIRE(aug_mf.has_hks_matrix());

  long nspin = qe_mf.nspin();
  long nkpts_ibz = qe_mf.nkpts_ibz();
  long nbnd = qe_mf.nbnd();
  auto all = nda::range::all;
  auto ref = qe_mf.eigval()(all, nda::range(nkpts_ibz), all);
  auto psp = hamilt::make_pseudopot(aug_mf);
  shape_t<4> sh = {nspin, nkpts_ibz, nbnd, nbnd};

  auto sF = make_shared_array<array_view_4d_t>(*mpi, sh);
  hamilt::set_fock(aug_mf, psp.get(), sF, false);
  // exclude_H0 subtracts the caller-supplied H0 from the stored matrix, and the
  // result is visible on every rank of the node (node_sync)
  auto sH0 = make_shared_array<array_view_4d_t>(*mpi, sh);
  hamilt::set_H0(aug_mf, psp.get(), sH0);
  auto sFx = make_shared_array<array_view_4d_t>(*mpi, sh);
  hamilt::set_fock(aug_mf, psp.get(), sFx, true, &sH0);

  double e_diag = 0.0, e_x = 0.0, den = 0.0;
  {
    auto F = sF.local(); auto Fx = sFx.local(); auto H0 = sH0.local();
    for (long s = 0; s < nspin; ++s)
      for (long k = 0; k < nkpts_ibz; ++k)
        for (long i = 0; i < nbnd; ++i) {
          for (long j = 0; j < nbnd; ++j) {
            if (mpi->node_comm.root())
              e_diag += std::norm(F(s,k,i,j) - ((i==j) ? ComplexType(ref(s,k,i)) : ComplexType(0.0)));
            e_x += std::norm(Fx(s,k,i,j) - (F(s,k,i,j) - H0(s,k,i,j)));
          }
          if (mpi->node_comm.root()) den += ref(s,k,i)*ref(s,k,i);
        }
  }
  mpi->comm.broadcast_n(&e_diag, 1, 0);
  mpi->comm.broadcast_n(&den, 1, 0);
  mpi->comm.all_reduce_in_place_n(&e_x, 1, boost::mpi3::max<>{});
  double nrm = std::sqrt(den);
  app_log(2, "aug_hks_set_fock: ||set_fock - diag(eps_QE)||/||eps_QE||      = {}",
          std::sqrt(e_diag)/nrm);
  app_log(2, "aug_hks_set_fock: ||F_noH0 - (F - H0)||/||eps_QE||            = {}",
          std::sqrt(e_x)/nrm);
  utils::VALUE_EQUAL(std::sqrt(e_diag)/nrm, 0.0, 1e-4);
  utils::VALUE_EQUAL(std::sqrt(e_x)/nrm, 0.0, 1e-8);

  // LR "mf_dft" seed: the MO energies of the stored matrix are the parent eigenvalues
  {
    auto [sMO_skia, sE_ska] = methods::get_mf_MOs(*mpi, aug_mf);
    double e_mo = 0.0;
    if (mpi->node_comm.root()) {
      auto E = sE_ska.local();
      for (long s = 0; s < nspin; ++s)
        for (long k = 0; k < nkpts_ibz; ++k)
          for (long i = 0; i < nbnd; ++i)
            e_mo += std::norm(E(s,k,i) - ref(s,k,i));
    }
    mpi->comm.broadcast_n(&e_mo, 1, 0);
    app_log(2, "aug_hks_set_fock: ||get_mf_MOs energies - eps_QE||/||eps_QE|| = {}",
            std::sqrt(e_mo)/nrm);
    utils::VALUE_EQUAL(std::sqrt(e_mo)/nrm, 0.0, 1e-4);
  }

  mpi->comm.barrier();
  if(mpi->comm.root()) remove("dummy_hks.h5");
  mpi->comm.barrier();
}

template<MEMORY_SPACE MEM>
void test_exx_impl(std::shared_ptr<mpi_context_t> &mpi)
{
  SECTION("lih222_hf") 
  {
    auto qe_h5 = std::make_shared<mf::MF>(mf::default_MF(mpi, "qe_lih222_hf", mf::h5_input_type));
    check_K<MEM>(*mpi, qe_h5, 1.00);
  }

  SECTION("GaAs222_hf") 
  {
    auto qe_h5 = std::make_shared<mf::MF>(mf::default_MF(mpi, "qe_GaAs222_hf", mf::h5_input_type));
    check_K<MEM>(*mpi, qe_h5, 1.00);
  }

  SECTION("GaAs222_so_hf") 
  {
    auto qe_h5 = std::make_shared<mf::MF>(mf::default_MF(mpi, "qe_GaAs222_so_hf", mf::h5_input_type));
    check_K<MEM>(*mpi, qe_h5, 0.25);
  }
}

TEST_CASE("exx", "[hamilt]") {
  auto& mpi = utils::make_unit_test_mpi_context();

  test_exx_impl<HOST_MEMORY>(mpi);
#if defined(ENABLE_DEVICE)
  test_exx_impl<DEVICE_MEMORY>(mpi);
  test_exx_impl<UNIFIED_MEMORY>(mpi);
#endif
}

template<MEMORY_SPACE MEM>
void test_vxc_impl(std::shared_ptr<mpi_context_t> &mpi)
{
  SECTION("lih223") 
  {
    // CNY: mf::xml_input_type is not implemented for vxc!
    auto qe_h5 = mf::default_MF(mpi, "qe_lih223", mf::h5_input_type);
    check_Vxc<MEM>(*mpi, qe_h5);
  }

  SECTION("lih223_inv") 
  {
    auto qe_h5 = mf::default_MF(mpi, "qe_lih223_inv", mf::h5_input_type);
    check_Vxc<MEM>(*mpi, qe_h5);
  }

  SECTION("lih223_sym") 
  {
    auto qe_h5 = mf::default_MF(mpi, "qe_lih223_sym", mf::h5_input_type);
    check_Vxc<MEM>(*mpi, qe_h5);
  }

  SECTION("GaAs222_so") 
  {
    auto qe_h5 = mf::default_MF(mpi, "qe_GaAs222_so", mf::h5_input_type);
    check_Vxc<MEM>(*mpi, qe_h5);
  }
} 

TEST_CASE("vxc", "[hamilt]") {
  auto& mpi = utils::make_unit_test_mpi_context();

  test_vxc_impl<HOST_MEMORY>(mpi);
#if defined(ENABLE_DEVICE)
  test_vxc_impl<DEVICE_MEMORY>(mpi);
  test_vxc_impl<UNIFIED_MEMORY>(mpi);
#endif
}

template<MEMORY_SPACE MEM>
void test_hartree_impl(std::shared_ptr<mpi_context_t> &mpi)
{

  SECTION("lih223") 
  {
    auto qe_h5 = std::make_shared<mf::MF>(mf::default_MF(mpi, "qe_lih223", mf::h5_input_type));
    check_Hartree<MEM>(*mpi, qe_h5);
  }

  SECTION("lih223_inv") 
  {
    auto qe_h5 = std::make_shared<mf::MF>(mf::default_MF(mpi, "qe_lih223_inv", mf::h5_input_type));
    check_Hartree<MEM>(*mpi, qe_h5);
  }

  SECTION("lih223_sym") 
  {
    auto qe_h5 = std::make_shared<mf::MF>(mf::default_MF(mpi, "qe_lih223_sym", mf::h5_input_type));
    check_Hartree<MEM>(*mpi, qe_h5);
  }

  SECTION("lih223_sym_diag") 
  {
    auto qe_h5 = std::make_shared<mf::MF>(mf::default_MF(mpi, "qe_lih223_sym", mf::h5_input_type));
    check_Hartree<MEM>(*mpi, qe_h5, true); // diagonal density as the input
  }

  SECTION("GaAs222_so")
  {
    auto qe_h5 = std::make_shared<mf::MF>(mf::default_MF(mpi, "qe_GaAs222_so", mf::h5_input_type));
    check_Hartree<MEM>(*mpi, qe_h5, true); // diagonal density as the input
  }

  // Diamond Si with force_symmorphic: the symmetry list has no inversion, so half
  // the BZ is reached by time reversal, and the inversion center is off the FFT
  // grid origin. This exercises the time-reversal branch of v_h (both the full-
  // density and diagonal-density paths), which the LiH/GaAs fixtures do not.
  SECTION("si333_sym")
  {
    auto qe_h5 = std::make_shared<mf::MF>(mf::default_MF(mpi, "qe_si333_sym", mf::h5_input_type));
    check_Hartree<MEM>(*mpi, qe_h5);
  }

  SECTION("si333_sym_diag")
  {
    auto qe_h5 = std::make_shared<mf::MF>(mf::default_MF(mpi, "qe_si333_sym", mf::h5_input_type));
    check_Hartree<MEM>(*mpi, qe_h5, true); // diagonal density as the input
  }
}

TEST_CASE("hartree", "[hamilt]") {
  auto& mpi = utils::make_unit_test_mpi_context();

  test_hartree_impl<HOST_MEMORY>(mpi);
#if defined(ENABLE_DEVICE)
  test_hartree_impl<DEVICE_MEMORY>(mpi);
  test_hartree_impl<UNIFIED_MEMORY>(mpi);
#endif
}

// MAM: this will be problematic when testing custom outdir/prefix, forbid this! 
TEST_CASE("one_body_components", "[hamilt]")
{
  auto& mpi = utils::make_unit_test_mpi_context();
  auto qe_xml = mf::default_MF(mpi,mf::qe_source);
  auto qe_h5 = mf::default_MF(mpi,mf::qe_source,mf::h5_input_type);
  auto bdft_mf = mf::default_MF(mpi,mf::bdft_source);
  auto py_mf = mf::default_MF(mpi,mf::pyscf_source);

  std::optional<double> qe_Href, qe_H0ref, qe_Ovref;
  std::optional<double> qeh5_Href, qeh5_H0ref, qeh5_Ovref;

  // MAM: these should be read from h5 file and test should be done on more reference tests
  //      including wsym, winv, etc... 
  if(not std::filesystem::exists(qe_outdir+"/"+qe_prefix+".xml"))
  {
    qe_Href = -0.7275620144914019;
    qe_H0ref = -3.8613653985679552;
    qe_Ovref = 1.0;
  }
  if(not std::filesystem::exists(qe_outdir+"/"+qe_prefix+".coqui.h5"))
  {
    qeh5_Href = -0.7275620144914019;
    qeh5_H0ref = -3.8613653985679552;
    qeh5_Ovref = 1.0;
  }

// qe_one_body_components
  qe_one_body_components<HOST_MEMORY>(*mpi,qe_h5);
#if defined(ENABLE_DEVICE)
  qe_one_body_components<DEVICE_MEMORY>(*mpi,qe_h5);
  qe_one_body_components<UNIFIED_MEMORY>(*mpi,qe_h5);
#endif

//test_H0
  test_H0<HOST_MEMORY>(*mpi,qe_xml,qe_H0ref);
  test_H0<HOST_MEMORY>(*mpi,qe_h5,qeh5_H0ref);
#if defined(ENABLE_DEVICE)
  test_H0<DEVICE_MEMORY>(*mpi,qe_xml,qe_H0ref);
  test_H0<UNIFIED_MEMORY>(*mpi,qe_xml,qe_H0ref);
  test_H0<DEVICE_MEMORY>(*mpi,qe_h5,qeh5_H0ref);
  test_H0<UNIFIED_MEMORY>(*mpi,qe_h5,qeh5_H0ref);
#endif

//test_H
//  test_H<HOST_MEMORY>(*mpi,qe_xml,qe_Href);
  test_H<HOST_MEMORY>(*mpi,qe_h5,qeh5_Href);
#if defined(ENABLE_DEVICE)
//  test_H<DEVICE_MEMORY>(*mpi,qe_xml,qe_Href);
//  test_H<UNIFIED_MEMORY>(*mpi,qe_xml,qe_Href);
  test_H<DEVICE_MEMORY>(*mpi,qe_h5,qeh5_Href);
  test_H<UNIFIED_MEMORY>(*mpi,qe_h5,qeh5_Href);
#endif

//qe_ovlp
  qe_ovlp<HOST_MEMORY>(*mpi,qe_h5);
#if defined(ENABLE_DEVICE)
  qe_ovlp<DEVICE_MEMORY>(*mpi,qe_h5);
  qe_ovlp<UNIFIED_MEMORY>(*mpi,qe_h5);
#endif

}

TEST_CASE("one_body_components_so", "[hamilt]")
{
  auto& mpi = utils::make_unit_test_mpi_context();

  //std::optional<double> Eref;
  auto qe = mf::default_MF(mpi,"qe_GaAs222_so");

// qe_one_body_components
  qe_one_body_components<HOST_MEMORY>(*mpi,qe);
#if defined(ENABLE_DEVICE)
  qe_one_body_components<DEVICE_MEMORY>(*mpi,qe);
  qe_one_body_components<UNIFIED_MEMORY>(*mpi,qe);
#endif

/*
//test_H0
  test_H0<HOST_MEMORY>(*mpi,qe,Eref);
#if defined(ENABLE_DEVICE)
  test_H0<DEVICE_MEMORY>(*mpi,qe,Erref);
  test_H0<UNIFIED_MEMORY>(*mpi,qe,Eref);
#endif

//test_H
  test_H<HOST_MEMORY>(*mpi,qe_xml,qe_Href);
  test_H<HOST_MEMORY>(*mpi,qe_h5,qeh5_Href);
#if defined(ENABLE_DEVICE)
  test_H<DEVICE_MEMORY>(*mpi,qe_xml,qe_Href);
  test_H<UNIFIED_MEMORY>(*mpi,qe_xml,qe_Href);
  test_H<DEVICE_MEMORY>(*mpi,qe_h5,qeh5_Href);
  test_H<UNIFIED_MEMORY>(*mpi,qe_h5,qeh5_Href);
#endif
*/
}

}

