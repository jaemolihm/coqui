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
#include <cctype>
#include "configuration.hpp"

#include <nda/nda.hpp>
#include <h5/h5.hpp>
#include "nda/h5.hpp"
#include "nda/linalg.hpp"
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
                   nda::MemoryArrayOfRank<4> auto const& ov,
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
        S(ov),
        DM(d),
        FockM(f)
      {
        utils::check(nspin_in_basis == nspin, "nspin_in_basis != nspin not yet allowed",nspin_in_basis,nspin);
        utils::check(npol_in_basis == npol, "npol_in_basis != npol not yet allowed",npol_in_basis,npol);
        int nk = _symm.nkpts_ibz;
        utils::check(_symm.nkpts == nk, "model_system: K-point symmetry not yet implemented (nkpts {} != nkpts_ibz {}).",
                     _symm.nkpts, nk);
        utils::check(nelec > 0, "nelec=0 in model_system constructor.");
        utils::check( nspin == 1 or nspin == 2, "Invalid nspin:{}",nspin);
        utils::check(npol > 0, "npol=0 in model_system constructor.");
        utils::check(nspin_in_basis <= nspin, "nspin_in_basis ({}) > nspin ({}) in model_system constructor.",nspin_in_basis,nspin);
        utils::check(npol_in_basis <= npol, "npol_in_basis ({}) > npol ({}) in model_system constructor.",npol_in_basis,npol);
        auto shp = std::array<long,4>{nspin_in_basis,nk,npol_in_basis*nbnd,npol_in_basis*nbnd};
        utils::check(H0.shape() == shp, "Dimension mismatch.");
        utils::check(S.shape() == shp, "Dimension mismatch.");
        utils::check(DM.shape() == shp, "Dimension mismatch.");
        utils::check(FockM.shape() == shp, "Dimension mismatch.");
        set_mf_spectrum();
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
      // q->0 treatment of the Coulomb kernel, imposed on every MBPT driver (attr div_treatment)
      std::string div_treatment = "ignore_g0";

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
      // overlap
      nda::array<ComplexType, 4> S;
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
        h5::h5_write_attribute(sgrp, "div_treatment", div_treatment);

        nda::h5_write(sgrp, "kpoint_weights", k_weight, false);
        nda::h5_write(sgrp, "lattice_vectors", latt, false);
        nda::h5_write(sgrp, "reciprocal_vectors", recv, false);
        nda::h5_write(sgrp, "eigval", eigval, false);
        nda::h5_write(sgrp, "occ", occ, false);

        nda::h5_write(sgrp, "H0", H0, false);
        nda::h5_write(sgrp, "overlap", S, false);
        nda::h5_write(sgrp, "density_matrix", DM, false);
        nda::h5_write(sgrp, "fock_matrix", FockM, false);
      }

      private:
      /*
       * Reads the model from filename (group /System). Required: the attrs number_of_spins,
       * number_of_polarizations, number_of_bands, number_of_elec, madelung_constant,
       * nuclear_energy, and the dataset H0. Without a full /System/BZ group, the BZ is built here
       * (identity symmetry only) from lattice_vectors and kpoints_crystal and/or kp_grid.
       * Optional, with defaults: kpoint_weights (1/nkpts), overlap (identity), density_matrix (0),
       * fock_matrix (H0), fermi_energy (computed from H0), div_treatment ("ignore_g0").
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
        utils::check( npol > 0, "Error: npol: {}",npol);
        utils::check( npol_in_basis <= npol and npol_in_basis > 0, "Error: npol_in_basis: {}",npol_in_basis);
        h5::h5_read_attribute(sgrp, "number_of_elec", nelec);
        h5::h5_read_attribute(sgrp, "madelung_constant", madelung);
        h5::h5_read_attribute(sgrp, "nuclear_energy", enuc);
        bool has_fermi_energy = H5Aexists(h5::hid_t(sgrp),"fermi_energy") > 0;
        if( has_fermi_energy ) {
          h5::h5_read_attribute(sgrp, "fermi_energy", efermi);
          has_efermi = true;
        }
        if( H5Aexists(h5::hid_t(sgrp),"div_treatment") > 0 ) {
          h5::h5_read_attribute(sgrp, "div_treatment", div_treatment);
          std::transform(div_treatment.begin(), div_treatment.end(), div_treatment.begin(),
                         [](unsigned char c) { return std::tolower(c); });
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

          bool has_klist = sgrp.has_dataset("kpoints_crystal");
          bool has_grid = sgrp.has_dataset("kp_grid");
          utils::check(has_klist or has_grid,
                       "Error in model_system: kpoints_crystal and/or kp_grid is required when /System/BZ is absent.");
          nda::array<double, 1> grid = nda::array<double, 1>::zeros({3});
          if(has_grid) {
            nda::array<long, 1> g;
            nda::h5_read(sgrp, "kp_grid", g);
            utils::check(g.size() == 3 and g(0) > 0 and g(1) > 0 and g(2) > 0,
                         "Error in model_system: kp_grid must hold 3 positive integers.");
            for(int i=0; i<3; ++i) grid(i) = double(g(i));
          }
          nda::array<double, 2> kcrys;
          if(has_klist) {
            nda::h5_read(sgrp, "kpoints_crystal", kcrys);
            utils::check(kcrys.extent(1) == 3, "Error in model_system: kpoints_crystal must be [nkpts,3].");
          } else {
            // i-major list: k = (i/N1, j/N2, l/N3), index (i*N2 + j)*N3 + l
            long n1 = long(grid(0)), n2 = long(grid(1)), n3 = long(grid(2));
            kcrys = nda::array<double, 2>(n1*n2*n3, 3);
            for(long i=0; i<n1; ++i)
              for(long j=0; j<n2; ++j)
                for(long l=0; l<n3; ++l) {
                  long ik = (i*n2 + j)*n3 + l;
                  kcrys(ik,0) = double(i)/double(n1);
                  kcrys(ik,1) = double(j)/double(n2);
                  kcrys(ik,2) = double(l)/double(n3);
                }
          }
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
          _symm = bz_symm(mpi->comm, true, lat, rv, grid, kcart, symm_list, false);
          utils::check(_symm.nkpts == nk, "Error in model_system: inconsistent k-point list.");
        }
        utils::check(std::abs(_symm.Qpts(0,0)) + std::abs(_symm.Qpts(0,1)) + std::abs(_symm.Qpts(0,2)) < 1e-12 and
                     std::abs(_symm.kpts_crys(0,0)) + std::abs(_symm.kpts_crys(0,1)) + std::abs(_symm.kpts_crys(0,2)) < 1e-12,
                     "Error in model_system: the first k-point must be Gamma.");

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
        S     = H0;
        DM    = H0;
        FockM = H0; 
        bool has_overlap = sgrp.has_dataset("overlap");
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
          if(has_overlap)
            S(all,range(nkpts_ibz),all,all) = read_("overlap"); 
          else
            for(int is=0; is<nspin_in_basis; ++is)
              for(int ik=0; ik<nkpts_ibz; ++ik)
                for(int a=0; a<npol_in_basis*nbnd; ++a) S(is,ik,a,a) = ComplexType(1.0);
          if(sgrp.has_dataset("density_matrix"))
            DM(all,range(nkpts_ibz),all,all) = read_("density_matrix"); 
          if(sgrp.has_dataset("fock_matrix"))
            FockM(all,range(nkpts_ibz),all,all) = read_("fock_matrix"); 
          else
            FockM(all,range(nkpts_ibz),all,all) = H0(all,range(nkpts_ibz),all,all);
          for(int ik=nkpts_ibz; ik<nkpts; ik++) {
            int ik_ibz = _symm.kp_to_ibz(ik);
            H0(all,ik,all,all) = H0(all,ik_ibz,all,all);
            S(all,ik,all,all) = S(all,ik_ibz,all,all);
            DM(all,ik,all,all) = DM(all,ik_ibz,all,all);
            FockM(all,ik,all,all) = FockM(all,ik_ibz,all,all);
          }
        }

        // eigval/occ are computed from H0 in an orthonormal basis
        if(has_overlap) {
          double dev = 0.0;
          for(int is=0; is<nspin_in_basis; ++is)
            for(int ik=0; ik<nkpts; ++ik)
              for(int a=0; a<npol_in_basis*nbnd; ++a)
                for(int b=0; b<npol_in_basis*nbnd; ++b)
                  dev = std::max(dev, std::abs(S(is,ik,a,b) - ComplexType(a==b ? 1.0 : 0.0)));
          utils::check(dev <= 1e-10,
                       "Error in model_system: non-identity overlap is not supported (max|S-1| = {}).", dev);
        }
        set_mf_spectrum(not has_fermi_energy);
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
          for(int j=0; j<3; ++j) b(i,j) = 2.0 * 3.14159265358979323846 * B(i,j);
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

      /*
       * Sets eigval and occ from H0, assuming an orthonormal basis, and efermi if compute_efermi.
       * occ is a T=0 filling with nelec electrons per cell. Levels within 1e-10*max(1,|e|) form a
       * shell; the shell at the Fermi level is filled with the same fractional occupation on every
       * member. efermi is the midpoint between the highest occupied and the lowest not fully
       * occupied shell (the shell energy if a shell is partially occupied).
       */
      void set_mf_spectrum(bool compute_efermi = true)
      {
        long nk = _symm.nkpts;
        long nb = npol_in_basis*nbnd;
        eigval = nda::array<double, 3>::zeros({nspin_in_basis, nk, nb});
        occ = nda::array<double, 3>::zeros({nspin_in_basis, nk, nb});
        for(long is=0; is<nspin_in_basis; ++is)
          for(long ik=0; ik<nk; ++ik) {
            nda::matrix<ComplexType> H_ = H0(is,ik,nda::range::all,nda::range::all);
            eigval(is,ik,nda::range::all) = nda::linalg::eigenvalues(H_);
          }

        // states sorted by energy, each holds k_weight(k) * spin_factor electrons when full
        double spin_factor = (nspin == 1 and npol == 1) ? 2.0 : 1.0;
        std::vector<std::array<long,3>> states;
        states.reserve(nspin_in_basis*nk*nb);
        for(long is=0; is<nspin_in_basis; ++is)
          for(long ik=0; ik<nk; ++ik)
            for(long n=0; n<nb; ++n) states.push_back({is,ik,n});
        auto e = [&](std::array<long,3> const& a) { return eigval(a[0],a[1],a[2]); };
        std::stable_sort(states.begin(), states.end(), [&](auto const& a, auto const& b) { return e(a) < e(b); });
        double ne = nelec;
        double e_homo = e(states.front());
        double e_lumo = e(states.back());
        bool found_lumo = false;
        size_t i = 0;
        while(i < states.size()) {
          double e0 = e(states[i]);
          size_t j = i;
          double cap = 0.0;
          while(j < states.size() and std::abs(e(states[j]) - e0) <= 1e-10*std::max(1.0,std::abs(e0))) {
            cap += k_weight(_symm.kp_to_ibz(states[j][1])) * spin_factor;
            ++j;
          }
          double f = (ne > 1e-12) ? std::min(1.0, ne/cap) : 0.0;
          for(size_t m = i; m < j; ++m) occ(states[m][0],states[m][1],states[m][2]) = f;
          ne -= f*cap;
          if(f > 0.0) e_homo = e0;
          if(f < 1.0 and not found_lumo) { e_lumo = e0; found_lumo = true; }
          i = j;
        }
        if(compute_efermi)
          efermi = found_lumo ? 0.5*(e_homo+e_lumo) : e_homo;
      }

    }; // model_system

  } // model
} // mf

#endif //COQUI_MODEL_SYSTEM_HPP
