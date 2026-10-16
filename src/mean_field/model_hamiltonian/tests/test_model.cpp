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

#include "mpi3/environment.hpp"
#include "mpi3/communicator.hpp"
#include "mpi3/shared_communicator.hpp"

#include "configuration.hpp"
#include "nda/nda.hpp"

#include "utilities/test_common.hpp"
#include "numerics/distributed_array/nda.hpp"
#include "mean_field/default_MF.hpp"
#include "mean_field/distributed_orbital_readers.hpp"
#include "mean_field/qe/qe_readonly.hpp"
#include "mean_field/model_hamiltonian/model_readonly.hpp"
#include "mean_field/model_hamiltonian/tests/model_test_utils.hpp"
#include "mean_field/MF.hpp"
#include "hamiltonian/one_body_hamiltonian.hpp"
#include "hamiltonian/pseudo/pseudopot.h"
#include "mean_field/symmetry/bz_symmetry.hpp"

namespace model_tests {

  using utils::ARRAY_EQUAL;
  template<int N>
  using shape_t = std::array<long,N>;

  TEST_CASE("model_system", "[mean_field_model]") {
    auto& mpi = utils::make_unit_test_mpi_context();

    std::string prefix = "dummy.model";
    int ns = 1;
    int nk = 1;
    int nb = 4;
    double nel = 2.0;
    if(mpi->comm.root()) {
      h5::file file("dummy_symm.h5", 'w');
      h5::group grp(file);
      h5::group sgrp = grp.create_group("System");
      mf::bz_symm::gamma_point_h5(sgrp);
      mf::bz_symm symm("dummy_symm.h5");

      auto h = nda::array<ComplexType,4>::zeros({ns,nk,nb,nb});
      auto d = nda::array<ComplexType,4>::zeros({ns,nk,nb,nb});
      auto f = nda::array<ComplexType,4>::zeros({ns,nk,nb,nb});
    
      mf::model::model_system m(mpi,"./",prefix,symm,ns,1,nel,h,d,f); 
      m.save(prefix+".h5");
      remove("dummy_symm.h5");
    }
    mpi->comm.barrier();

    mf::model::model_system m(mpi,"./",prefix,-1); 
    
    REQUIRE(m.nspin == ns);
    REQUIRE(m.nbnd == nb);
    REQUIRE(m.nelec == nel);
    mpi->comm.barrier();

    if(mpi->comm.root())
      remove("dummy.model.h5");
    mpi->comm.barrier();
  }

  // q in crystal coordinates from the cartesian Qpts: q_crys = latt . Q / (2 pi)
  inline auto qpts_crystal(mf::MF const& mf) {
    auto Q = mf.Qpts();
    auto latt = mf.lattv();
    nda::array<double,2> qc(Q.extent(0),3);
    for(long iq=0; iq<Q.extent(0); ++iq)
      for(int i=0; i<3; ++i)
        qc(iq,i) = (latt(i,0)*Q(iq,0) + latt(i,1)*Q(iq,1) + latt(i,2)*Q(iq,2)) / (2.0*M_PI);
    return qc;
  }

  TEST_CASE("model_minimal_h5", "[mean_field_model]") {
    auto& mpi = utils::make_unit_test_mpi_context();
    long N = 4;
    long nk = N*N;
    double nelec = 1.0;
    std::string prefix = "hubbard_model_minimal";
    auto [kpts_ref, H0_ref] = square_lattice_H0(N, 1.0, 0.0);

    auto [eigval_ref, occ_ref] = model_spectrum(nk, 1);

    auto read_model = [&](std::optional<double> ef, bool with_spectrum = false) {
      if(mpi->comm.root())
        write_minimal_model_h5(prefix+".h5", N, 1.0, 0.0, nelec, 1, ef, with_spectrum);
      mpi->comm.barrier();
      mf::model::model_readonly m(mpi, "./", prefix, -1);
      mpi->comm.barrier();
      return m;
    };

    {
      auto m = read_model(std::nullopt);
      mf::MF mf(m);
      auto const& sys = m.get_sys();

      REQUIRE(mf.nkpts() == nk);
      REQUIRE(mf.nkpts_ibz() == nk);
      REQUIRE(mf.nqpts_ibz() == nk);
      REQUIRE(nda::sum(mf.k_weight()) == Approx(1.0).margin(1e-14));
      for(long ik=0; ik<nk; ++ik) REQUIRE(mf.k_weight(ik) == Approx(1.0/nk).margin(1e-15));

      // a_m . b_n = 2 pi delta_mn
      auto latt = mf.lattv();
      auto recv = mf.recv();
      for(int m_=0; m_<3; ++m_)
        for(int n=0; n<3; ++n) {
          double d = latt(m_,0)*recv(n,0) + latt(m_,1)*recv(n,1) + latt(m_,2)*recv(n,2);
          REQUIRE(d == Approx(m_==n ? 2.0*M_PI : 0.0).margin(1e-12));
        }

      // k order is the file order; Qpts = kpts; qk_to_k2 = k - q (mod G)
      auto kc = mf.kpts_crystal();
      for(long ik=0; ik<nk; ++ik)
        for(int d=0; d<3; ++d) REQUIRE(kc(ik,d) == Approx(kpts_ref(ik,d)).margin(1e-12));
      auto qc = qpts_crystal(mf);
      for(long iq=0; iq<nk; ++iq)
        for(int d=0; d<3; ++d) REQUIRE(qc(iq,d) == Approx(kpts_ref(iq,d)).margin(1e-12));
      auto qk = mf.qk_to_k2();
      auto index_of = [&](double k1, double k2) {
        long i = (long(std::lround(k1*N)) % N + N) % N;
        long j = (long(std::lround(k2*N)) % N + N) % N;
        return i*N+j;
      };
      for(long iq=0; iq<nk; ++iq)
        for(long ik=0; ik<nk; ++ik)
          REQUIRE(qk(iq,ik) == index_of(kc(ik,0)-qc(iq,0), kc(ik,1)-qc(iq,1)));

      // the optional density_matrix defaults to zero
      utils::ARRAY_EQUAL(sys.H0, H0_ref, 1e-14);
      utils::ARRAY_EQUAL(sys.FockM, H0_ref, 1e-14);
      for(long ik=0; ik<nk; ++ik)
        REQUIRE(std::abs(sys.DM(0,ik,0,0)) < 1e-15);
      REQUIRE(mf.madelung() == 0.0);

      // no spectrum in the file: eigval/occ stay empty, efermi unset
      REQUIRE(mf.eigval().size() == 0);
      REQUIRE(mf.occ().size() == 0);
      REQUIRE(mf.efermi() == 0.0);
    }

    // eigval, occ and fermi_energy from the file are kept
    {
      auto m = read_model(0.123, true);
      mf::MF mf(m);
      REQUIRE(mf.efermi() == 0.123);
      utils::ARRAY_EQUAL(mf.eigval(), eigval_ref, 0.0);
      utils::ARRAY_EQUAL(mf.occ(), occ_ref, 0.0);
    }

    // a file written by model_system::save (full /System/BZ) reads back identically
    for(bool with_spectrum : {false, true}) {
      auto m = read_model(std::nullopt, with_spectrum);
      mf::MF mf(m);
      std::string prefix2 = prefix + "_saved";
      if(mpi->comm.root()) m.get_sys().save(prefix2+".h5");
      mpi->comm.barrier();
      mf::model::model_readonly m2(mpi, "./", prefix2, -1);
      mf::MF mf2(m2);
      utils::ARRAY_EQUAL(mf2.lattv(), mf.lattv(), 1e-14);
      utils::ARRAY_EQUAL(mf2.recv(), mf.recv(), 1e-14);
      utils::ARRAY_EQUAL(mf2.kpts(), mf.kpts(), 1e-14);
      utils::ARRAY_EQUAL(mf2.kpts_crystal(), mf.kpts_crystal(), 1e-14);
      utils::ARRAY_EQUAL(mf2.Qpts(), mf.Qpts(), 1e-14);
      REQUIRE(mf2.eigval().size() == (with_spectrum ? nk : 0));
      REQUIRE(mf2.occ().size() == (with_spectrum ? nk : 0));
      if(with_spectrum) {
        utils::ARRAY_EQUAL(mf2.eigval(), eigval_ref, 0.0);
        utils::ARRAY_EQUAL(mf2.occ(), occ_ref, 0.0);
      }
      utils::ARRAY_EQUAL(mf2.k_weight(), mf.k_weight(), 1e-14);
      utils::ARRAY_EQUAL(m2.get_sys().H0, m.get_sys().H0, 1e-14);
      for(long iq=0; iq<nk; ++iq) {
        REQUIRE(mf2.qminus()(iq) == mf.qminus()(iq));
        for(long ik=0; ik<nk; ++ik) REQUIRE(mf2.qk_to_k2(iq,ik) == mf.qk_to_k2(iq,ik));
      }
      REQUIRE(mf2.efermi() == Approx(mf.efermi()).margin(1e-15));
      mpi->comm.barrier();
      if(mpi->comm.root()) remove((prefix2+".h5").c_str());
    }

    mpi->comm.barrier();
    if(mpi->comm.root()) remove((prefix+".h5").c_str());
    mpi->comm.barrier();
  }

  // the model reader aborts on a non-identity overlap; the abort itself (MPI_Abort) is not testable here
  TEST_CASE("model_overlap", "[mean_field_model]") {
    auto& mpi = utils::make_unit_test_mpi_context();
    using mf::model::model_system;
    long nb = 3;
    auto S = nda::array<ComplexType,4>::zeros({1, 2, nb, nb});
    for(long ik=0; ik<2; ++ik)
      for(long a=0; a<nb; ++a) S(0,ik,a,a) = ComplexType(1.0);
    REQUIRE(model_system::unsupported_overlap(S).empty());
    S(0,1,0,2) = ComplexType(0.0, 1e-3);
    REQUIRE(model_system::unsupported_overlap(S).find("non-identity overlap") != std::string::npos);

    // a file with an identity overlap dataset (as model_system::save and downfold_1e write) still loads
    std::string prefix = "hubbard_model_overlap";
    if(mpi->comm.root()) {
      write_minimal_model_h5(prefix+".h5", 2, 1.0, 0.0, 1.0);
      h5::file file(prefix+".h5", 'a');
      h5::group sgrp = h5::group(file).open_group("System");
      auto I = nda::array<ComplexType,4>::zeros({1, 4, 1, 1});
      I() = ComplexType(1.0);
      nda::h5_write(sgrp, "overlap", I, false);
    }
    mpi->comm.barrier();
    mf::model::model_readonly m(mpi, "./", prefix, -1);
    REQUIRE(m.get_sys().nbnd == 1);
    mpi->comm.barrier();
    if(mpi->comm.root()) remove((prefix+".h5").c_str());
    mpi->comm.barrier();
  }

  // the model reader aborts on these BZs; the abort itself (MPI_Abort) is not testable here
  TEST_CASE("model_unsupported_symmetry", "[mean_field_model]") {
    auto& mpi = utils::make_unit_test_mpi_context();
    using mf::model::model_system;
    auto s0 = mf::bz_symm::gamma_point_instance();
    REQUIRE(model_system::unsupported_symmetry(s0).empty());

    {
      auto s = s0;
      s.nqpts = 2;
      REQUIRE(model_system::unsupported_symmetry(s).find("q-point symmetry") != std::string::npos);
    }
    {
      auto s = s0;
      s.kp_trev = {true};
      REQUIRE(model_system::unsupported_symmetry(s).find("time-reversal") != std::string::npos);
    }
    {
      auto s = s0;
      s.nkpts_trev_pairs = 1;
      REQUIRE(model_system::unsupported_symmetry(s).find("time-reversal") != std::string::npos);
    }
    // a non-identity operation survives a /System/BZ round trip
    {
      auto s = s0;
      nda::array<double,2> R = {{-1.0,0,0},{0,-1.0,0},{0,0,-1.0}};
      nda::array<double,1> ft = {0,0,0};
      s.symm_list.emplace_back(utils::symm_op{R,R,ft});
      std::string fname = "model_bz_inversion.h5";
      if(mpi->comm.root()) {
        h5::file file(fname, 'w');
        h5::group grp(file);
        h5::group sgrp = grp.create_group("System");
        s.save(sgrp);
      }
      mpi->comm.barrier();
      mf::bz_symm s2;
      {
        h5::file file(fname, 'r');
        h5::group grp(file);
        h5::group sgrp = grp.open_group("System");
        REQUIRE(mf::bz_symm::can_init_from_h5(sgrp));
        s2.initialize_from_h5(sgrp);
      }
      REQUIRE(s2.symm_list.size() == 2);
      REQUIRE(model_system::unsupported_symmetry(s2).find("non-identity symmetry") != std::string::npos);
      mpi->comm.barrier();
      if(mpi->comm.root()) remove(fname.c_str());
      mpi->comm.barrier();
    }
  }

  TEST_CASE("model_hamilt_dispatch", "[mean_field_model]") {
    auto& mpi = utils::make_unit_test_mpi_context();
    long N = 2;
    long np = mpi->comm.size();
    long nb = 2*np;
    long nk = N*N;
    std::string prefix = "hubbard_model_dispatch";
    if(mpi->comm.root()) write_minimal_model_h5(prefix+".h5", N, 1.0, -0.3, 1.0, nb);
    mpi->comm.barrier();
    mf::model::model_readonly m(mpi, "./", prefix, -1);
    mf::MF mf(m);
    // the file holds H0 = fock_matrix = H0_ref, and the basis is orthonormal
    auto [kpts_ref, H0_ref] = square_lattice_H0(N, 1.0, -0.3, nb);
    nda::array<ComplexType,4> S_ref = nda::zeros<ComplexType>(std::array<long,4>{1, nk, nb, nb});
    for(long ik=0; ik<nk; ++ik)
      for(long a=0; a<nb; ++a) S_ref(0,ik,a,a) = ComplexType(1.0);

    auto psp = hamilt::make_pseudopot(mf);
    REQUIRE(psp != nullptr);

    auto check_local = [&](auto const& dA, nda::array<ComplexType,4> const& A) {
      auto loc = dA.local();
      auto rk = dA.local_range(1);
      auto ra = dA.local_range(2);
      auto rb = dA.local_range(3);
      for(long ik=0; ik<loc.extent(1); ++ik)
        for(long a=0; a<loc.extent(2); ++a)
          for(long b=0; b<loc.extent(3); ++b)
            REQUIRE(std::abs(loc(0,ik,a,b) - A(0,rk.first()+ik,ra.first()+a,rb.first()+b)) < 1e-14);
    };
    auto dH0 = hamilt::H0<HOST_MEMORY>(mf, mpi->comm, psp.get());
    check_local(dH0, H0_ref);
    auto dF = hamilt::F<HOST_MEMORY>(mf, mpi->comm);
    check_local(dF, H0_ref);
    auto dS = hamilt::ovlp<HOST_MEMORY>(mf, mpi->comm);
    check_local(dS, S_ref);

    // diagonal of the overlap with all ranks on the band axis: local rows start at a nonzero offset
    std::array<long,3> pgrid = {1,1,np};
    std::array<long,3> bsz = {1,1,2};
    auto dSd = hamilt::ovlp_diagonal<HOST_MEMORY>(mf, mpi->comm, nda::range(nk), nda::range(nb), pgrid, bsz);
    REQUIRE(dSd.local_shape()[2] == 2);
    auto loc = dSd.local();
    for(long ik=0; ik<loc.extent(1); ++ik)
      for(long a=0; a<loc.extent(2); ++a)
        REQUIRE(std::abs(loc(0,ik,a) - ComplexType(1.0)) < 1e-15);

    mpi->comm.barrier();
    if(mpi->comm.root()) remove((prefix+".h5").c_str());
    mpi->comm.barrier();
  }

/*
  TEST_CASE("model_h5", "[mean_field_model]") {
    auto& mpi_context = utils::make_unit_test_mpi_context();
    auto [outdir,prefix] = utils::utest_filename(mf::model_source);
    mf::model::model_readonly mf(mpi_context, outdir, prefix);

    // unit cell info
    REQUIRE(mf.get_sys().nelec > 0);

    // BZ info
    auto nkpts = mf.nkpts();
    REQUIRE(nkpts > 0);
    REQUIRE( mf.get_sys().bz().kp_grid.shape() == shape_t<1>{3} );
    REQUIRE( mf.kpts().shape() == shape_t<2>{nkpts, 3} );
    REQUIRE( mf.get_sys().bz().Qpts.shape() == shape_t<2>{nkpts, 3} );
    REQUIRE( mf.get_sys().bz().qk_to_k2.shape() == shape_t<2>{nkpts, nkpts} );
    REQUIRE(mf.get_sys().bz().qminus.shape()[0] == nkpts);

    // basis info
    auto nbnd = mf.get_sys().nbnd;
    REQUIRE(nbnd > 0);

  }
*/
} // model_tests
