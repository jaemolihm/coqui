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


#ifndef MF_MODULE_HPP
#define MF_MODULE_HPP

#include <tuple>
#include <vector>
#include <string>

#include "python/utils/mpi_handler.hpp"
#include "python/utils/mpi_handler.wrap.hxx"

#include "IO/ptree/InputParser.hpp"
#include "mean_field/mf_utils.hpp"
#include "orbitals/orbital_augmenter.h"
#include "orbitals/orbital_generator.h"
#include "orbitals/eph_vertex.h"
#include "hamiltonian/pseudo/pseudopot.h"

namespace coqui_py {

  C2PY_IGNORE
  inline decltype(auto) make_MF(MpiHandler &mpi, std::string mf_params, const std::string &mf_type) {
    auto parser = InputParser(mf_params);
    return std::make_shared<mf::MF>(mf::make_MF(mpi.get_mpi(), parser.get_root(), mf_type));
  }

  /**
   * @brief Mf class
   *
   * The Mf class encapsulates the state of a mean-field inputs to CoQuí.
   * This class is read-only and is responsible for accessing the input data, including
   *
   *   1. Metadata of the simulated system, such as the number of bands, spins, and k-points.
   *   2. Single-particle basis functions whom CoQuí uses to construct a many-body Hamiltonian in CoQuí.
   */
  class Mf {
  public:
    Mf(MpiHandler &mpi, std::string mf_params, const std::string &mf_type):
    _mf(make_MF(mpi, mf_params, mf_type)) {}
    C2PY_IGNORE
    Mf(std::shared_ptr<mf::MF> mf): _mf(std::move(mf)) {}

    ~Mf() = default;
    Mf(Mf const&) = default;
    Mf(Mf &&) = default;
    Mf& operator=(Mf const&) = default;
    Mf& operator=(Mf &&) = default;

    bool operator==(const Mf& other) const {
      return _mf == other._mf;
    }

    auto mf_type() const { return mf::mf_source_enum_to_string(_mf->mf_type()); }
    auto outdir() const  { return _mf->outdir(); }
    auto prefix() const { return _mf->prefix(); }
    auto filename() const { return _mf->filename(); }

    /// Unit-cell volume in atomic units. CoQuí's Coulomb kernel carries no 1/Omega,
    /// so a caller assembling v(q+G)-weighted quantities needs this explicitly.
    auto volume() const { return _mf->volume(); }

    auto nelec() const { return _mf->nelec(); }
    auto nbnd() const { return _mf->nbnd(); }
    auto nspin() const { return _mf->nspin(); }
    auto npol() const { return _mf->npol(); }

    /* FFT grid */
    auto ecutrho () const { return _mf->ecutrho(); }
    /// FFT mesh (3,). Owning, because c2py cannot return a view on anything that
    /// is not an nda::array.
    nda::array<long, 1> fft_grid() const {
      auto n = _mf->fft_grid_dim();
      nda::array<long, 1> mesh(3);
      for(int d=0; d<3; ++d) mesh(d) = n(d);
      return mesh;
    }
    auto ecutwfc () const { return _mf->wfc_truncated_grid()->ecut(); }
    auto fft_grid_wfc() const { return _mf->wfc_truncated_grid()->mesh(); }

    /* k-points info*/
    auto kp_grid() const { return _mf->kp_grid(); }
    auto nkpts() const { return _mf->nkpts(); }
    auto kpts() const { return _mf->kpts(); }
    auto kpts_crystal() const { return _mf->kpts_crystal(); }
    auto k_weights() const { return _mf->k_weight(); }
    auto nkpts_ibz() const { return _mf->nkpts_ibz(); }
    auto kpts_ibz() const { return _mf->kpts_ibz(); }
    auto nqpts() const { return _mf->nqpts(); }
    auto qpts() const { return _mf->Qpts(); }
    auto nqpts_ibz() const { return _mf->nqpts_ibz(); }
    auto qpts_ibz() const { return _mf->Qpts_ibz(); }

    auto nuclear_energy() const { return _mf->nuclear_energy(); }

    // create a new MpiHandler from mf's mpi context
    auto mpi() const { return MpiHandler(_mf->mpi()); }

    /**
     * @brief Create an augmented mean-field system from this one.
     *
     * Keeps the original nbnd bands and appends orthonormalized augmentation
     * states generated from the first nbnd_aug bands (e.g. type="momentum" adds
     * p̂_α ψ_b for α=x,y,z), truncated with singular-value cutoff epstol and
     * padded to a uniform band count across k-points. The result is written to
     * {outdir}/{prefix}.h5 as a bdft system and returned. Because the basis is
     * not an eigenbasis, many-body runs on it must use h0_source="compute".
     *
     * @param prefix     prefix of the new mean-field file
     * @param outdir     directory for the new mean-field file
     * @param type       augmentation transform ("momentum")
     * @param nbnd_aug   number of bands to transform (<= nbnd; -1 = all,
     *                   0 = none: original orbitals in augmented bdft format)
     * @param epstol     dimensionless singular-value cutoff (thresholds s, the
     *                   residual amplitude of the dtau_step-scaled raw state)
     * @param dirs       for type="momentum", the Cartesian directions (subset of
     *                   {0,1,2} = {x,y,z}) whose p̂_α ψ states are appended; an
     *                   empty list means all three directions
     * @param dtau_step  displacement step (bohr) scaling the raw dψ/dτ states
     */
    Mf augment_basis(const std::string& prefix, const std::string& outdir,
                     const std::string& type, long nbnd_aug, double epstol,
                     const std::vector<long>& dirs, double dtau_step) const {
      std::string fn = outdir + "/" + prefix + ".h5";
      auto parser = InputParser(std::string("{\"type\": \"") + type + "\"}");
      std::vector<int> dirs_i(dirs.begin(), dirs.end());
      auto augmenter = orbitals::make_augmenter(*_mf, parser.get_root(), dirs_i);
      auto new_mf = orbitals::add_augmentation<HOST_MEMORY>(*_mf, fn, augmenter,
                                                            nbnd_aug, epstol, dtau_step);
      return Mf(std::make_shared<mf::MF>(std::move(new_mf)));
    }

    /**
     * @brief Create a δψ-augmented mean-field system from this one.
     *
     * Appends orthonormalized DFPT response wavefunctions (δψ) read from
     * {deltapsi_dir}/deltapsi_iq{iq}_mode{m}_ik{k}.hdf5, using the first R
     * (=nbnd_aug) bands of each mode for every phonon iq in iq_list. δψ(n,k)
     * carries momentum k+q and is deposited on the 'w' grid at k+q. The buffer
     * bands m ∈ [R, nbnd) are added back using the screened vertex g_scr and
     * eigenvalues read (and converted Ry→Ha) from
     * {elph_dir}/elph_bare.iq{iq}.h5 (also the source of q). Requires npol==1
     * and a full-BZ k-grid. See orbitals::add_augmentation_dpsi.
     *
     * @param prefix            prefix of the new mean-field file
     * @param outdir            directory for the new mean-field file
     * @param deltapsi_dir      directory holding the deltapsi_*.hdf5 files
     * @param elph_dir          directory holding the elph_bare.iq*.h5 files
     * @param iq_list           phonon q indices to include
     * @param nmodes            modes per q (<=0 → 3*natom)
     * @param mode_list         1-based mode indices contributing raw states
     *                          (empty → all modes 1..nmodes)
     * @param nbnd_aug          δψ bands used per mode (R; -1 → all in file)
     * @param nbnd_mf           original bands kept M (<=0 or >nbnd → keep all
     *                          nbnd; buffer fills [M, nbnd))
     * @param smearing_deltapsi buffer denominator smearing σ (Ha)
     * @param epstol        dimensionless singular-value cutoff (thresholds s,
     *                      the residual amplitude of the dtau_step-scaled state)
     * @param dtau_step     displacement step (bohr) scaling the raw δψ states
     */
    Mf augment_basis_deltapsi(const std::string& prefix, const std::string& outdir,
                          const std::string& deltapsi_dir, const std::string& elph_dir,
                          const std::vector<long>& iq_list, long nmodes,
                          const std::vector<long>& mode_list,
                          long nbnd_aug, long nbnd_mf, double smearing_deltapsi,
                          double epstol, double dtau_step) const {
      std::string fn = outdir + "/" + prefix + ".h5";
      auto new_mf = orbitals::add_augmentation_dpsi<HOST_MEMORY>(
          *_mf, fn, deltapsi_dir, elph_dir, iq_list, nmodes, mode_list,
          nbnd_aug, nbnd_mf, smearing_deltapsi, epstol, dtau_step);
      return Mf(std::make_shared<mf::MF>(std::move(new_mf)));
    }

    /// Electron-phonon nonlocal projector overlaps, returned in memory (no file).
    /// Returns a tuple (P, dion, proj_per_species, species_of_atom, proj_offset):
    ///   P    : (4, nspin, nkpts_ibz, nproj*npol, nbnd) complex, replicated;
    ///          P[0] = <beta|phi>, P[1..3] = <beta|(k+G)_{x,y,z} phi>.
    ///   dion : (nsp, nhm*npol, nhm*npol) D-matrix in Hartree.
    ///   proj_per_species / species_of_atom / proj_offset : projector->atom maps.
    /// These, with npol() and the k-points, factorize the bare nonlocal e-ph
    /// vertex (assembled by coqui.compute_bare_eph_vertex).
    std::tuple<nda::array<ComplexType, 5>, nda::array<ComplexType, 3>,
               nda::array<int, 1>, nda::array<int, 1>, nda::array<int, 1>>
    eph_projector_overlaps() const {
      hamilt::pseudopot pp(*_mf);
      nda::array<ComplexType, 5> P;
      pp.eph_projector_overlaps(*_mf, P);
      return {std::move(P), pp.Dion(), pp.proj_per_species(),
              pp.species_of_atom(), pp.proj_offset()};
    }

    /// Bare nonlocal electron-phonon vertex g_nl(s,mode,k,m,n) =
    /// <phi_{m,k+q}| dV^nl_mode |phi_{n,k}>, factorized from the projector
    /// overlaps and D-matrix (Hartree). Returned full on the MPI root and empty
    /// on every other rank. npol=1, full-BZ k-grid.
    nda::array<ComplexType, 5> eph_vertex_nonlocal(
        nda::array_const_view<double, 1> q_cryst) const {
      hamilt::pseudopot pp(*_mf);
      return pp.eph_vertex_nonlocal(*_mf, q_cryst);
    }

    /// Full bare electron-phonon vertex g(s,mode,k,m,n) =
    /// <phi_{m,k+q}| dV_mode |phi_{n,k}> in the band basis (Hartree): local part
    /// (ionic dvloc rebuilt from the h5 radial vloc) + nonlocal part (projector
    /// factorization). mode = 3*kappa + d, nmodes = 3*natom. Returned full on the
    /// MPI root and empty on every other rank (the vertex is large and only the
    /// root consumes it downstream). Requires npol=1 and a full-BZ k-grid.
    nda::array<ComplexType, 5> compute_bare_eph_vertex(
        nda::array_const_view<double, 1> q_cryst) const {
      hamilt::pseudopot pp(*_mf);
      auto dv = pp.build_dvloc_ion(*_mf, q_cryst);                        // (nmodes,nnr) Ha
      auto g  = orbitals::eph_vertex_local<HOST_MEMORY>(*_mf, dv(), q_cryst);
      g += pp.eph_vertex_nonlocal(*_mf, q_cryst);
      return g;
    }

    /// Bare nonlocal part of the q=0 second-order electron-phonon vertex, stored
    /// compactly as (nspin, nat, 3, 3, nk, nb, nb), dims (atom, cart_i, cart_j):
    ///   g2_nl(s,kappa,i,j,k,m,n) = <phi_{m,k}| d^2 V^nl/dtau_i dtau_j |phi_{n,k}>.
    /// Factorized from the projector overlaps and D-matrix (Hartree); diagonal in
    /// atom, so off-atom blocks are not stored. Returned full on the MPI root and
    /// empty on every other rank. npol=1, full-BZ k-grid. Use together with a local
    /// part (e.g. eph_vertex_local on a d^2V_loc field at q=0) to assemble g2.
    nda::array<ComplexType, 7> eph_vertex_nonlocal_d2() const {
      hamilt::pseudopot pp(*_mf);
      return pp.eph_vertex_nonlocal_d2(*_mf);
    }

    /// Full bare second-order electron-phonon vertex at q=0 (Hartree), stored
    /// compactly as (nspin, nat, 3, 3, nk, nb, nb) with dims (atom, cart_i,
    /// cart_j):
    ///   g2(s,kappa,i,j,k,m,n) = <phi_{m,k}| d^2 V_bare/dtau_i dtau_j |phi_{n,k}>,
    /// i.e. mode1 = 3*kappa+i, mode2 = 3*kappa+j. d^2 V_bare is diagonal in the
    /// atom index (each ionic term depends on a single atom), so the off-atom
    /// mode1/mode2 blocks are exactly zero and are not stored. Local part: d^2V_loc
    /// (build_d2vloc_ion) applied via eph_vertex_local at q=0; nonlocal part:
    /// projector factorization (eph_vertex_nonlocal_d2). This is the
    /// "g2_bare"/"d2H0_bare" quantity QE stores at q=Gamma. Returned full on the
    /// MPI root and empty on every other rank. npol=1, full-BZ k-grid.
    nda::array<ComplexType, 7> compute_bare_eph_vertex_d2() const {
      hamilt::pseudopot pp(*_mf);
      // nonlocal part, compact atom-diagonal (nspin,nat,3,3,nk,nb,nb); root only
      auto g2 = pp.eph_vertex_nonlocal_d2(*_mf);
      // local part: 6*nat symmetric-pair d2V fields, evaluated at q=0
      auto d2v = pp.build_d2vloc_ion(*_mf);                          // (6*nat, nnr) Ha
      nda::array<double,1> q0(3); q0() = 0.0;                        // q=0
      auto gp  = orbitals::eph_vertex_local<HOST_MEMORY>(*_mf, d2v(), q0());
      //                                     gp: (nspin, 6*nat, nk, nb, nb) at q=0
      // pair p = 0..5 -> (i,j) = (x,x),(y,y),(z,z),(x,y),(x,z),(y,z)
      static constexpr int pi[6] = {0, 1, 2, 0, 0, 1};
      static constexpr int pj[6] = {0, 1, 2, 1, 2, 2};
      if(g2.size() > 0) {   // root only; non-root g2/gp are empty
        long nspin = g2.shape(0), nat = g2.shape(1), nk = g2.shape(4);
        long nb = g2.shape(5);
        for(long s=0; s<nspin; ++s)
          for(long ka=0; ka<nat; ++ka)
            for(int p=0; p<6; ++p) {
              int i = pi[p], j = pj[p];
              for(long k=0; k<nk; ++k)
                for(long m=0; m<nb; ++m)
                  for(long n=0; n<nb; ++n) {
                    ComplexType v = gp(s, 6*ka+p, k, m, n);
                    g2(s, ka, i, j, k, m, n) += v;
                    if(i != j) g2(s, ka, j, i, k, m, n) += v;
                  }
            }
      }
      return g2;
    }

    /// Local (bare) electron-phonon vertex g_loc(s,mode,k,m,n) =
    /// <phi_{m,k+q}| dV_mode |phi_{n,k}>, computed directly from the real-space
    /// orbitals. `dV` is (nmodes, nnr) cell-periodic on the FFT grid; `q_cryst`
    /// is the phonon wavevector in crystal coordinates.
    nda::array<ComplexType, 5> eph_vertex_local(
        nda::array_const_view<ComplexType, 2> dV,
        nda::array_const_view<double, 1> q_cryst) const {
      return orbitals::eph_vertex_local<HOST_MEMORY>(*_mf, dV, q_cryst);
    }

    C2PY_IGNORE
    auto get_mf() const { return _mf; }

    friend std::ostream& operator<<(std::ostream& out, const Mf& handler) {
      out << "CoQuí mean-field state\n"
          << "----------------------\n"
          << "  Type                : " << handler.mf_type() << '\n'
          << "  Prefix              : " << handler.prefix() << '\n'
          << "  Output dir          : " << handler.outdir() << '\n'
          << "  Number of electrons (nelec): " << handler.nelec() << '\n'
          << "  Bands (nbnd)        : " << handler.nbnd() << '\n'
          << "  Spins (nspin)       : " << handler.nspin() << '\n'
          << "  Polarization (npol) : " << handler.npol() << '\n'
          << "  Monkhorst-Pack grid : (" << handler.kp_grid()(0) << ", "
          << handler.kp_grid()(1) << ", " << handler.kp_grid()(2) << ")\n"
          << "  K-points            : " << handler.nkpts() << " total, "
          << handler.nkpts_ibz() << " in IBZ\n"
          << "  Q-points            : " << handler.nqpts() << " total, "
          << handler.nqpts_ibz() << " in IBZ\n"
          << "  Kinetic energy cutoff (ecutrho): " << handler.ecutrho() << " a.u.\n"
          << "  FFT grid            : (" << handler.fft_grid()(0) << ", "
          << handler.fft_grid()(1) << ", " << handler.fft_grid()(2) << ")";
      return out;
    }

  private:
    std::shared_ptr<mf::MF> _mf;

  }; // Mf

} // coqui_py

#endif
