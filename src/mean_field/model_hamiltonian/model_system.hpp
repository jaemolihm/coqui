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


#ifndef COQUI_MODEL_SYSTEM_HPP
#define COQUI_MODEL_SYSTEM_HPP

#include <string>
#include <filesystem>
#include <map>
#include <vector>
#include <algorithm>
#include <cmath>
#include "configuration.hpp"

#include <nda/nda.hpp>
#include <h5/h5.hpp>
#include "nda/h5.hpp"
#include <hdf5.h>
#include <hdf5_hl.h>

#include "utilities/check.hpp"
#include "utilities/concepts.hpp"
#include "utilities/mpi_context.h"

#include "mean_field/mf_source.hpp"
#include "mean_field/symmetry/bz_symmetry.hpp"

namespace mf {
  namespace model {
    /**
     * Struct that contains details of a Model Hamiltonian, read from hdf5 files
     */
    struct model_system {
      public:

      model_system(std::shared_ptr<utils::mpi_context_t<> > mpi_, std::string outdir_, std::string prefix_,
                  int _nbnd) : 
      mpi(std::move(mpi_)), outdir(outdir_), prefix(prefix_), filename(outdir_ + "/" + prefix_ + ".h5")
      {
        utils::check(_nbnd!=0, "Error in model_system: nbnd==0.");
        utils::check(std::filesystem::exists(filename),
		     " Error in model_system: filename: {} does not exist.",filename);
        initialize_from_h5(_nbnd);
      }

      model_system([[maybe_unused]] std::shared_ptr<utils::mpi_context_t<> > mpi_, std::string outdir_, std::string prefix_,
                   bz_symm symm, int nspin_, int npol_, double _nel,  
                   nda::MemoryArrayOfRank<4> auto const& h,
                   nda::MemoryArrayOfRank<4> auto const& d,
                   nda::MemoryArrayOfRank<4> auto const& f,
                   double mad = 0,
                   double e0 = 0):
        mpi(std::move(mpi_)), outdir(outdir_), prefix(prefix_), filename(outdir_ + "/" + prefix_ + ".h5"),
        _symm(symm),
        nelec(_nel),
        madelung(mad),
        enuc(e0),
        efermi(0.0),
        // this constructor takes no Fermi energy; 0.0 is the unset default
        has_efermi(false),
        k_weight(_symm.nkpts_ibz,1.0/double(_symm.nkpts)),
        ndims(3),
        nspin(nspin_),
        nspin_in_basis(nspin),
        npol(npol_),
        npol_in_basis(std::max(1,npol)),
        nbnd(h.extent(2)/npol_in_basis),
        H0(h),
        DM(d),
        FockM(f)
      {
        utils::check(nspin_in_basis == nspin, "nspin_in_basis != nspin not yet allowed",nspin_in_basis,nspin);
        utils::check(npol_in_basis == npol, "npol_in_basis != npol not yet allowed",npol_in_basis,npol);
        int nk = _symm.nkpts_ibz;
        utils::check(_symm.nkpts == nk, "model_system: K-point symmetry not yet implemented (nkpts {} != nkpts_ibz {}).",
                     _symm.nkpts, nk);
        auto symm_err = unsupported_symmetry(_symm);
        utils::check(symm_err.empty(), "{}", symm_err);
        utils::check(nelec > 0, "nelec=0 in model_system constructor.");
        utils::check( nspin == 1 or nspin == 2, "Invalid nspin:{}",nspin);
        utils::check(npol == 1, "model_system constructor: number_of_polarizations must be 1 (got {}).",npol);
        utils::check(nspin_in_basis <= nspin, "nspin_in_basis ({}) > nspin ({}) in model_system constructor.",nspin_in_basis,nspin);
        utils::check(npol_in_basis <= npol, "npol_in_basis ({}) > npol ({}) in model_system constructor.",npol_in_basis,npol);
        auto shp = std::array<long,4>{nspin_in_basis,nk,npol_in_basis*nbnd,npol_in_basis*nbnd};
        utils::check(H0.shape() == shp, "Dimension mismatch.");
        utils::check(DM.shape() == shp, "Dimension mismatch.");
        utils::check(FockM.shape() == shp, "Dimension mismatch.");
      } 

      model_system(const model_system& ) = default;
      model_system(model_system&& ) = default;

      model_system& operator=(const model_system& ) = default;
      model_system& operator=(model_system&& ) = default;

      ~model_system() = default;

      auto const& bz() const { return _symm; }

      void save(std::string fn) const {
        h5::file file = h5::file(fn, 'w');
        h5::group grp(file);
        save(grp);
      }

      // MPI handler for communicators
      std::shared_ptr<utils::mpi_context_t<mpi3::communicator>> mpi;

      // model output directory
      std::string outdir = "";
      // model prefix
      std::string prefix = "";
      // name of file with input data
      std::string filename = "";
      // type of input file, hardwired to h5
      mf_input_file_type_e input_file_type = h5_input_type;    

      // kpt and Qpt information
      bz_symm _symm;

      // # of electrons
      double nelec = 0.0;
      // madelung constant for finite-size corrections
      double madelung = 0.0;
      // nuclear energy
      double enuc = 0.0;
      // fermi energy
      double efermi = 0.0;
      // true when efermi carries a value assigned by the source; false means the
      // 0.0 default.
      bool has_efermi = false;

      // weight of kpoint in averages
      nda::array<double, 1> k_weight;

      // # of dimensions
      int ndims = 3; // hardwired to 3d for now
      // # of spin
      int nspin = 0;
      // # of spin in basis set
      int nspin_in_basis = 0;
      // # of spin
      int npol = 0;
      // # of spin in basis set
      int npol_in_basis = 0;

      // basis info
      // # of primary single-particle basis
      int nbnd;


      // 1-body Hamiltonian
      nda::array<ComplexType, 4> H0;
      // dm
      nda::array<ComplexType, 4> DM;
      // fock matrix 
      nda::array<ComplexType, 4> FockM;

      /************************************************************************************/
      // Variables below exist for consistenty with other backends, but have no real meaning 
      // in a model hamiltonian
      // unit cell info
      int natoms = 0;
      // # of unique species
      int nspecies = 0;
      // name/ids of species
      std::vector<std::string> species;
      // array with the id(Z!?) of atoms in unit cell
      nda::array<int, 1> at_ids = nda::array<int, 1>(0);
      // array of atom positions
      nda::array<double, 2> at_pos = nda::array<double, 2>(0,0);
      // fft grid
      nda::stack_array<int, 3> fft_mesh = {0,0,0};
      // lattice vectors: a_n[i] = recv(n,i), for the ith (x,y,z) cartesian component of the nth (1,2,3) vector.
      nda::stack_array<double, 3, 3> latt = {{0,0,0},{0,0,0},{0,0,0}};
      // reciprocal vectors: b_n[i] = recv(n,i), for the ith (x,y,z) cartesian component of the nth (1,2,3) vector.
      nda::stack_array<double, 3, 3> recv = {{0,0,0},{0,0,0},{0,0,0}};
      // plane-wave cutoff for charge density
      double ecutrho = 0.0;
      // number of points in fft grid
      int nnr = 0;
      // # of auxiliary single-particle basis
      int nbnd_aux = 0;
      // eigenvalues: [nspin_in_basis, kpoint, npol_in_basis*band]
      nda::array<double, 3> eigval_aux = nda::array<double, 3>(0,0,0);
      // eigenvalues: [nspin_in_basis, kpoint, npol_in_basis*band]
      nda::array<double, 3> aux_weight = nda::array<double, 3>(0,0,0);
      // orbitals are stored on fft grid or not
      bool orb_on_fft_grid = false;
      // eigenvalues: [nspin_in_basis, kpoint, npol_in_basis*band]
      nda::array<double, 3> eigval;
      // occupation numbers: [nspin_in_basis, kpoint, npol_in_basis*band]
      nda::array<double, 3> occ;
      bool spinorbit = false;
      /************************************************************************************/

      public:

      /*
       * The symmetry a model BZ must not carry, as an error message; empty when the BZ is the
       * plain k-point list (identity only, no q-point symmetry, no time reversal).
       */
      static std::string unsupported_symmetry(bz_symm const& s)
      {
        if(s.nqpts != s.nqpts_ibz)
          return fmt::format("model_system: q-point symmetry not supported (nqpts {} != nqpts_ibz {}).",
                             s.nqpts, s.nqpts_ibz);
        if(s.nkpts_trev_pairs != 0 or std::any_of(s.kp_trev.begin(), s.kp_trev.end(), [](bool b) { return b; }))
          return fmt::format("model_system: time-reversal symmetry not supported (nkpts_trev_pairs {}).",
                             s.nkpts_trev_pairs);
        if(s.symm_list.size() != 1)
          return fmt::format("model_system: non-identity symmetry not supported ({} symmetry operations).",
                             s.symm_list.size());
        return "";
      }

      /*
       * The reason an overlap [nspin,nk,nb,nb] is not accepted, as an error message; empty when it is
       * the identity (the model basis is orthonormal).
       */
      static std::string unsupported_overlap(nda::array<ComplexType, 4> const& S)
      {
        double dev = 0.0;
        for(long is=0; is<S.extent(0); ++is)
          for(long ik=0; ik<S.extent(1); ++ik)
            for(long a=0; a<S.extent(2); ++a)
              for(long b=0; b<S.extent(3); ++b)
                dev = std::max(dev, std::abs(S(is,ik,a,b) - ComplexType(a==b ? 1.0 : 0.0)));
        if(dev > 1e-10)
          return fmt::format("model_system: a non-identity overlap is not supported (max|S-1| = {}).", dev);
        return "";
      }

      void save(h5::group& grp) const
      {
        h5::group sgrp = (grp.has_subgroup("System") ?
                          grp.open_group("System")    :
                          grp.create_group("System", true));

        //BZ
        _symm.save(sgrp);

        // unit cell
        h5::h5_write_attribute(sgrp, "number_of_spins", nspin);
        h5::h5_write_attribute(sgrp, "number_of_polarizations", npol);
        h5::h5_write_attribute(sgrp, "number_of_dimensions", ndims);
        h5::h5_write_attribute(sgrp, "number_of_spins_in_basis", nspin_in_basis);
        h5::h5_write_attribute(sgrp, "number_of_polarizations_in_basis", npol_in_basis);
        h5::h5_write_attribute(sgrp, "number_of_bands", nbnd);
        h5::h5_write_attribute(sgrp, "number_of_elec", nelec);
        h5::h5_write_attribute(sgrp, "madelung_constant", madelung);
        h5::h5_write_attribute(sgrp, "nuclear_energy", enuc);
        // absence is what tells a reader has_efermi = false
        if(has_efermi) h5::h5_write_attribute(sgrp, "fermi_energy", efermi);

        nda::h5_write(sgrp, "kpoint_weights", k_weight, false);
        nda::h5_write(sgrp, "lattice_vectors", latt, false);
        nda::h5_write(sgrp, "reciprocal_vectors", recv, false);
        if(eigval.size() > 0) nda::h5_write(sgrp, "eigval", eigval, false);
        if(occ.size() > 0) nda::h5_write(sgrp, "occ", occ, false);

        nda::h5_write(sgrp, "H0", H0, false);
        // the basis is orthonormal; written for readers that require the dataset
        nda::array<ComplexType, 4> S(H0.shape());
        S() = ComplexType(0.0);
        for(long is=0; is<S.extent(0); ++is)
          for(long ik=0; ik<S.extent(1); ++ik)
            for(long a=0; a<S.extent(2); ++a) S(is,ik,a,a) = ComplexType(1.0);
        nda::h5_write(sgrp, "overlap", S, false);
        nda::h5_write(sgrp, "density_matrix", DM, false);
        nda::h5_write(sgrp, "fock_matrix", FockM, false);
      }

      private:
      /*
       * Reads the model from filename, group /System ("attr" = HDF5 attribute of /System; "BZ" = the
       * group /System/BZ). Units: Hartree, Bohr. Complex arrays use the nda layout (float64 with a
       * trailing (re, im) dimension and a __complex__ attribute).
       *
       *   entry                             kind        shape          status
       *   number_of_spins                   attr int                   required (1 or 2)
       *   number_of_polarizations           attr int                   required, must be 1
       *   number_of_dimensions              attr int                   optional (3)
       *   number_of_spins_in_basis          attr int                   optional (number_of_spins)
       *   number_of_polarizations_in_basis  attr int                   optional (number_of_polarizations)
       *   number_of_bands                   attr int                   required
       *   number_of_elec                    attr float                 required (electrons per cell)
       *   madelung_constant                 attr float                 required (0 for a model: no q->0 correction)
       *   nuclear_energy                    attr float                 required
       *   fermi_energy                      attr float                 optional
       *   lattice_vectors                   float64     [3,3]          required without BZ (rows); else optional (0)
       *   kpoints_crystal                   float64     [nk,3]         required without BZ; ignored with BZ
       *   kpoint_weights                    float64     [nk]           optional (1/nk)
       *   H0                                complex     [ns,nk,nb,nb]  required
       *   fock_matrix                       complex     [ns,nk,nb,nb]  required (full mean-field Hamiltonian incl. H0)
       *   density_matrix                    complex     [ns,nk,nb,nb]  optional (0)
       *   eigval, occ                       float64     [ns,nk,nb]     optional; the MBPT drivers need eigval (IAFT wmax)
       *   overlap                           complex     [ns,nk,nb,nb]  optional; must be the identity (orthonormal basis)
       *   BZ                                group                      optional (written by save() and downfold_1e)
       *
       * Without BZ, the BZ is built from lattice_vectors and kpoints_crystal: a full regular grid in crystal
       * coordinates with Gamma first, identity symmetry only; the k index of every array follows the list.
       * Aborts on: k-point, q-point, time-reversal or non-identity symmetry in BZ; a k list that is not a
       * regular Gamma-first grid; a non-identity overlap; number_of_polarizations != 1;
       * number_of_spins_in_basis != number_of_spins.
       */
      void initialize_from_h5(int _nbnd = -1)
      {
        using nda::range;
        decltype(nda::range::all) all;
        h5::file file = h5::file(filename, 'r');
        h5::group grp(file);

        h5::group sgrp = grp.open_group("System"); 

        if(sgrp.has_key("number_of_dimensions")) 
          h5::h5_read_attribute(sgrp, "number_of_dimensions", ndims);
        h5::h5_read_attribute(sgrp, "number_of_spins", nspin);
        utils::check( nspin == 1 or nspin == 2, "Invalid nspin:{}",nspin);
        if(sgrp.has_key("number_of_spins_in_basis")) 
          h5::h5_read_attribute(sgrp, "number_of_spins_in_basis", nspin_in_basis);
        else
          nspin_in_basis = nspin;
        utils::check( nspin_in_basis <= nspin and nspin_in_basis > 0, "Error: nspin_in_basis: {}",nspin_in_basis);
        h5::h5_read_attribute(sgrp, "number_of_polarizations", npol);
        if(sgrp.has_key("number_of_polarizations_in_basis"))
          h5::h5_read_attribute(sgrp, "number_of_polarizations_in_basis", npol_in_basis);
        else
          npol_in_basis = npol;
        utils::check( npol == 1, "Error in model_system: number_of_polarizations must be 1 (got {}).",npol);
        utils::check( npol_in_basis <= npol and npol_in_basis > 0, "Error: npol_in_basis: {}",npol_in_basis);
        h5::h5_read_attribute(sgrp, "number_of_elec", nelec);
        h5::h5_read_attribute(sgrp, "madelung_constant", madelung);
        h5::h5_read_attribute(sgrp, "nuclear_energy", enuc);
        if( H5Aexists(h5::hid_t(sgrp),"fermi_energy") ) {
          h5::h5_read_attribute(sgrp, "fermi_energy", efermi);
          has_efermi = true;
        }

        // temporary!
        utils::check(nspin_in_basis == nspin, "nspin_in_basis != nspin not yet allowed",nspin_in_basis,nspin);
        utils::check(npol_in_basis == npol, "npol_in_basis != npol not yet allowed",npol_in_basis,npol);

        // BZ
        if(bz_symm::can_init_from_h5(sgrp)) {
          _symm.initialize_from_h5(sgrp);
          if(sgrp.has_dataset("lattice_vectors")) {
            nda::array<double, 2> lat;
            nda::h5_read(sgrp, "lattice_vectors", lat);
            set_lattice(lat);
          }
        } else {
          utils::check(sgrp.has_dataset("lattice_vectors"),
                       "Error in model_system: lattice_vectors is required when /System/BZ is absent.");
          nda::array<double, 2> lat;
          nda::h5_read(sgrp, "lattice_vectors", lat);
          utils::check(lat.shape() == std::array<long,2>{3,3}, "Error in model_system: lattice_vectors must be [3,3].");
          set_lattice(lat);

          utils::check(sgrp.has_dataset("kpoints_crystal"),
                       "Error in model_system: kpoints_crystal is required when /System/BZ is absent.");
          nda::array<double, 2> kcrys;
          nda::h5_read(sgrp, "kpoints_crystal", kcrys);
          utils::check(kcrys.extent(1) == 3, "Error in model_system: kpoints_crystal must be [nkpts,3].");
          // cartesian k-points: k_cart = recv^T k_crys
          long nk = kcrys.extent(0);
          nda::array<double, 2> kcart = nda::array<double, 2>::zeros({nk,3});
          for(long ik=0; ik<nk; ++ik)
            for(int i=0; i<3; ++i)
              for(int n=0; n<3; ++n) kcart(ik,i) += recv(n,i)*kcrys(ik,n);
          nda::array<double, 2> rv(3,3);
          rv() = recv;
          std::vector<utils::symm_op> symm_list;
          nda::array<double, 2> E = { {1.0,0,0}, {0,1.0,0}, {0,0,1.0} };
          nda::array<double, 1> ft = {0,0,0};
          symm_list.emplace_back(utils::symm_op{E,E,ft});
          // kp_grid = 0: bz_symm derives the grid from the k list and checks that it is regular
          _symm = bz_symm(mpi->comm, true, lat, rv, nda::array<double, 1>::zeros({3}), kcart, symm_list, false);
          utils::check(_symm.nkpts == nk, "Error in model_system: inconsistent k-point list.");
        }
        utils::check(std::abs(_symm.Qpts(0,0)) + std::abs(_symm.Qpts(0,1)) + std::abs(_symm.Qpts(0,2)) < 1e-12 and
                     std::abs(_symm.kpts_crys(0,0)) + std::abs(_symm.kpts_crys(0,1)) + std::abs(_symm.kpts_crys(0,2)) < 1e-12,
                     "Error in model_system: the first k-point must be Gamma.");
        auto symm_err = unsupported_symmetry(_symm);
        utils::check(symm_err.empty(), "Error in {}", symm_err);

        h5::h5_read_attribute(sgrp, "number_of_bands", nbnd);
        utils::check(_nbnd!=0, "Error in model_system: nbnd==0.");
        if(_nbnd > 0) {
          utils::check(_nbnd <= nbnd, 
                       "Error in initialize_from_h5: nbnd:{} nbnd in h5:{}",_nbnd,nbnd);
          nbnd = _nbnd;
        }
          
        int nkpts = _symm.nkpts;
        int nkpts_ibz = _symm.nkpts_ibz;
        utils::check(nkpts == nkpts_ibz, 
          "Error in model_hamiltonian::initialize_from_h5: K-point symmetry not yet implemented.");

        if(sgrp.has_dataset("kpoint_weights")) {
          nda::h5_read(grp, "System/kpoint_weights", k_weight);
          utils::check(k_weight.size() == nkpts_ibz, "Error in model_system: kpoint_weights must be [nkpts].");
        } else
          k_weight = nda::array<double, 1>(nkpts_ibz, 1.0/double(nkpts));
        if( std::abs( nda::sum(k_weight) - 1.0 ) > 1e-6 and mpi->comm.root() )
          app_warning(" k_weight is not normalized. sum(k_weight): {}",nda::sum(k_weight)); 

        H0    = nda::array<ComplexType, 4>(nspin_in_basis, nkpts, npol_in_basis*nbnd, npol_in_basis*nbnd);
        H0()  = ComplexType(0.0); 
        DM    = H0;
        FockM = H0; 
        {
          nda::array<ComplexType, 4> e_;
          auto read_ = [&] (std::string name) {
              nda::h5_read(sgrp, name, e_);
              utils::check( e_.extent(0) == nspin_in_basis and
                            e_.extent(1) >= nkpts_ibz and 
                            e_.extent(2) >= npol_in_basis*nbnd and 
                            e_.extent(3) >= npol_in_basis*nbnd,
                            "Error with " + name +": Incorrect dimensions in h5.");
              return  e_(range(nspin_in_basis),range(nkpts_ibz),range(npol_in_basis*nbnd),range(npol_in_basis*nbnd)); 
            };
          H0(all,range(nkpts_ibz),all,all) = read_("H0"); 
          if(sgrp.has_dataset("density_matrix"))
            DM(all,range(nkpts_ibz),all,all) = read_("density_matrix"); 
          FockM(all,range(nkpts_ibz),all,all) = read_("fock_matrix");
          for(int ik=nkpts_ibz; ik<nkpts; ik++) {
            int ik_ibz = _symm.kp_to_ibz(ik);
            H0(all,ik,all,all) = H0(all,ik_ibz,all,all);
            DM(all,ik,all,all) = DM(all,ik_ibz,all,all);
            FockM(all,ik,all,all) = FockM(all,ik_ibz,all,all);
          }
        }

        // the model basis is orthonormal: an overlap dataset is accepted only if it is the identity
        if(sgrp.has_dataset("overlap")) {
          nda::array<ComplexType, 4> S_;
          nda::h5_read(sgrp, "overlap", S_);
          auto overlap_err = unsupported_overlap(S_);
          utils::check(overlap_err.empty(), "Error in {}", overlap_err);
        }

        // mean-field spectrum [nspin_in_basis, nkpts, npol_in_basis*nbnd], when the file carries it
        {
          long nb = npol_in_basis*nbnd;
          auto read_spectrum_ = [&] (std::string name, nda::array<double, 3>& v) {
              if(not sgrp.has_dataset(name)) return;
              nda::array<double, 3> e_;
              nda::h5_read(sgrp, name, e_);
              utils::check( e_.extent(0) == nspin_in_basis and
                            e_.extent(1) >= nkpts_ibz and
                            e_.extent(2) >= nb,
                            "Error with " + name +": Incorrect dimensions in h5.");
              v = e_(range(nspin_in_basis),range(nkpts_ibz),range(nb));
            };
          read_spectrum_("eigval", eigval);
          read_spectrum_("occ", occ);
        }
      }

      /*
       * Reciprocal vectors 2*pi*inv(latt)^T (rows are b_n). A zero latt gives zero.
       */
      static nda::stack_array<double, 3, 3> reciprocal_vectors(nda::array<double, 2> const& lat)
      {
        nda::stack_array<double, 3, 3> b = {{0,0,0},{0,0,0},{0,0,0}};
        bool zero = true;
        for(int i=0; i<3; ++i)
          for(int j=0; j<3; ++j)
            if(lat(i,j) != 0.0) zero = false;
        if(zero) return b;
        nda::matrix<double> B(3,3);
        for(int i=0; i<3; ++i)
          for(int j=0; j<3; ++j) B(i,j) = lat(j,i);
        nda::inverse3_in_place(B);
        for(int i=0; i<3; ++i)
          for(int j=0; j<3; ++j) b(i,j) = 2.0 * M_PI * B(i,j);
        return b;
      }

      /*
       * Sets latt and recv = 2*pi*inv(latt)^T.
       */
      void set_lattice(nda::array<double, 2> const& lat)
      {
        for(int i=0; i<3; ++i)
          for(int j=0; j<3; ++j) latt(i,j) = lat(i,j);
        recv = reciprocal_vectors(lat);
      }

    }; // model_system

  } // model
} // mf

#endif //COQUI_MODEL_SYSTEM_HPP
