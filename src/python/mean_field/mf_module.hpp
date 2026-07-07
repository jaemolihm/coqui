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

    auto nelec() const { return _mf->nelec(); }
    auto nbnd() const { return _mf->nbnd(); }
    auto nspin() const { return _mf->nspin(); }
    auto npol() const { return _mf->npol(); }

    /* FFT grid */
    auto ecutrho () const { return _mf->ecutrho(); }
    auto fft_grid() const { return _mf->fft_grid_dim(); }
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
     * @param epstol singular-value cutoff selecting the number kept per k
     */
    Mf augment_basis(const std::string& prefix, const std::string& outdir,
                     const std::string& type, long nbnd_aug, double epstol) const {
      std::string fn = outdir + "/" + prefix + ".h5";
      auto parser = InputParser(std::string("{\"type\": \"") + type + "\"}");
      auto augmenter = orbitals::make_augmenter(*_mf, parser.get_root());
      auto new_mf = orbitals::add_augmentation<HOST_MEMORY>(*_mf, fn, augmenter,
                                                            nbnd_aug, epstol);
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
     * @param nbnd_aug          δψ bands used per mode (R; -1 → all in file)
     * @param nbnd_mf           original bands kept M (<=0 or >nbnd → keep all
     *                          nbnd; buffer fills [M, nbnd))
     * @param smearing_deltapsi buffer denominator smearing σ (Ha)
     * @param epstol        singular-value cutoff selecting states kept per k
     */
    Mf augment_basis_deltapsi(const std::string& prefix, const std::string& outdir,
                          const std::string& deltapsi_dir, const std::string& elph_dir,
                          const std::vector<long>& iq_list, long nmodes,
                          long nbnd_aug, long nbnd_mf, double smearing_deltapsi,
                          double epstol) const {
      std::string fn = outdir + "/" + prefix + ".h5";
      auto new_mf = orbitals::add_augmentation_dpsi<HOST_MEMORY>(
          *_mf, fn, deltapsi_dir, elph_dir, iq_list, nmodes, nbnd_aug, nbnd_mf,
          smearing_deltapsi, epstol);
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
    /// overlaps and D-matrix (Hartree, replicated). npol=1, full-BZ k-grid.
    nda::array<ComplexType, 5> eph_vertex_nonlocal(
        nda::array_const_view<double, 1> q_cryst) const {
      hamilt::pseudopot pp(*_mf);
      return pp.eph_vertex_nonlocal(*_mf, q_cryst);
    }

    /// Full bare electron-phonon vertex g(s,mode,k,m,n) =
    /// <phi_{m,k+q}| dV_mode |phi_{n,k}> in the band basis (Hartree, replicated):
    /// local part (ionic dvloc rebuilt from the h5 radial vloc) + nonlocal part
    /// (projector factorization). mode = 3*kappa + d, nmodes = 3*natom.
    /// Requires npol=1 and a full-BZ k-grid (nkpts == nkpts_ibz).
    nda::array<ComplexType, 5> compute_bare_eph_vertex(
        nda::array_const_view<double, 1> q_cryst) const {
      hamilt::pseudopot pp(*_mf);
      auto dv = pp.build_dvloc_ion(*_mf, q_cryst);                        // (nmodes,nnr) Ha
      auto g  = orbitals::eph_vertex_local<HOST_MEMORY>(*_mf, dv(), q_cryst);
      g += pp.eph_vertex_nonlocal(*_mf, q_cryst);
      return g;
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
