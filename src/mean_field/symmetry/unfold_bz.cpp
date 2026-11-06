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


#include <filesystem>
#include <algorithm>
#include <array>
#include <map>
#include <nda/h5.hpp>
#include "numerics/sparse/csr_blas.hpp"
#include "numerics/nda_functions.hpp"
#include "numerics/imag_axes_ft/iaft_utils.hpp"
#include "utilities/kpoint_utils.hpp"
#include "unfold_bz.h"

namespace methods {
  void unfold_bz(utils::mpi_context_t<mpi3::communicator> &context, mf::MF &mf, std::string scf_prefix) {
    std::string filename = scf_prefix + ".mbpt.h5";
    int iter;
    std::string scf_type;
    {
      h5::file file(filename, 'r');
      auto scf_grp = h5::group(file).open_group("scf");
      h5::h5_read(scf_grp, "final_iter", iter);
      auto iter_grp = scf_grp.open_group("iter" + std::to_string(iter));
      // F_skij marks a dyson scf; Sigma_tskij may be absent (== exactly zero, e.g. HF).
      if ( iter_grp.has_dataset("F_skij") )
        scf_type = "dyson";
      else if ( iter_grp.has_dataset("E_ska") )
        scf_type = "quasiparticle";
      else
        utils::check(false, "unfold_bz: incorrect dataset from {}", filename);
    }

    app_log(2, "Brillouin zone unfolding");
    app_log(2, "------------------------");
    app_log(2, "  - scf type: {}", scf_type);
    app_log(2, "  - Monkhorst-Pack mesh:  ({},{},{})", mf.kp_grid()(0), mf.kp_grid()(1), mf.kp_grid()(2));
    app_log(2, "  - number of k-points:   {}", mf.nkpts());
    app_log(2, "  - number of irreducible k-points: {}", mf.nkpts_ibz());
    app_log(2, "  - input/output scf h5 file: {}", filename);

    if (scf_type == "dyson")
      unfold_dyson_solution(context, mf, filename, iter);
    else if (scf_type == "quasiparticle")
      unfold_qp_solution(context, mf, filename, iter);
    else
      utils::check(false, "unfold_bz: incorrect scf type {}", scf_type);
  }

  void unfold_dyson_solution(utils::mpi_context_t<mpi3::communicator> &context,
                             mf::MF &mf, std::string filename, long iter) {
    auto sF_skij_full = unfold_1e_hamiltonian(context, mf, filename,
                                              "scf/iter"+std::to_string(iter)+"/F_skij");
    auto sSigma_skij_tskij_full = unfold_dynamic_hamiltonian(context, mf, filename,
                                                             "scf/iter"+std::to_string(iter)+"/Sigma_tskij");

    if (context.comm.root()) {
      h5::file file(filename, 'a');
      auto iter_grp = h5::group(file).open_group("scf/iter"+std::to_string(iter));
      nda::h5_write(iter_grp, "F_skij_unfold", sF_skij_full.local(), false);
      nda::h5_write(iter_grp, "Sigma_tskij_unfold", sSigma_skij_tskij_full.local(), false);
    }
  }

  void unfold_qp_solution(utils::mpi_context_t<mpi3::communicator> &context,
                          mf::MF &mf, std::string filename, long iter) {

    auto sE_ska_full = unfold_qp_energy(context, mf, filename, iter);
    auto sHeff_skij_full = unfold_1e_hamiltonian(context, mf,
                                                 filename, "scf/iter"+std::to_string(iter)+"/Heff_skij");

    if (context.comm.root()) {
      h5::file file(filename, 'a');
      auto iter_grp = h5::group(file).open_group("scf/iter"+std::to_string(iter));
      nda::h5_write(iter_grp, "E_ska_unfold", sE_ska_full.local(), false);
      nda::h5_write(iter_grp, "Heff_skij_unfold", sHeff_skij_full.local(), false);
    }
  }

  auto unfold_dynamic_hamiltonian(utils::mpi_context_t<mpi3::communicator> &context,
                                  mf::MF &mf, std::string filename, std::string dataset)
  -> math::shm::shared_array<nda::array_view<ComplexType,5>> {
    using math::shm::make_shared_array;
    using local_Array_view_5D_t = nda::array_view<ComplexType, 5>;

    int nts;
    {
      nda::array<double, 1> tau_mesh;
      h5::file file(filename, 'r');
      auto iaft_grp = h5::group(file).open_group("/imaginary_fourier_transform");
      auto tau_grp = iaft_grp.open_group("tau_mesh");
      nda::h5_read(tau_grp, "fermion", tau_mesh);
      nts = tau_mesh.shape(0);
    }
    auto ns = mf.nspin();
    auto nkpts = mf.nkpts();
    auto nkpts_ibz = mf.nkpts_ibz();
    auto nbnd  = mf.nbnd();

    auto sSigma_tskij = make_shared_array<local_Array_view_5D_t>(
        context.comm, context.internode_comm, context.node_comm,
        {nts, ns, nkpts_ibz, nbnd, nbnd});

    if (context.node_comm.root()) {
      h5::file file(filename, 'r');
      auto grp = h5::group(file);
      auto Sigma_loc = sSigma_tskij.local();
      // A missing dataset means Sigma is exactly zero (e.g. HF checkpoint).
      if (grp.has_dataset(dataset))
        nda::h5_read(grp, dataset, Sigma_loc);
      else
        Sigma_loc() = 0.0;
    }
    context.node_comm.barrier();

    if (nkpts == nkpts_ibz) return sSigma_tskij;

    auto sSigma_tskij_full = make_shared_array<local_Array_view_5D_t>(
        context.comm, context.internode_comm, context.node_comm,
        {nts, ns, nkpts, nbnd, nbnd});
    auto kp_to_ibz = mf.kp_to_ibz();
    auto kp_trev = mf.kp_trev();
    auto S_tsk_full = sSigma_tskij_full.local();
    auto S_tsk = sSigma_tskij.local();
    sSigma_tskij_full.win().fence();
    for (size_t itsk=context.node_comm.rank(); itsk<nts*ns*nkpts; itsk+=context.node_comm.size()) {
      size_t it = itsk/(ns*nkpts); // itsk = it*ns*nkpts + is*nkpts + ik
      size_t is = (itsk/nkpts) % ns;
      size_t ik = itsk%nkpts;
      size_t ik_ibz = kp_to_ibz(ik);

      if (kp_trev(ik))
        S_tsk_full(it, is, ik, nda::ellipsis{}) = nda::conj(S_tsk(it, is, ik_ibz, nda::ellipsis{}));
      else
        S_tsk_full(it, is, ik, nda::ellipsis{}) = S_tsk(it, is, ik_ibz, nda::ellipsis{});
    }
    sSigma_tskij_full.win().fence();
    return sSigma_tskij_full;
  }

  auto unfold_1e_hamiltonian(utils::mpi_context_t<mpi3::communicator> &context,
                             mf::MF &mf, std::string filename, std::string dataset,
                             bool include_H0)
  -> math::shm::shared_array<nda::array_view<ComplexType,4>> {
    using math::shm::make_shared_array;
    using local_Array_view_4D_t = nda::array_view<ComplexType, 4>;

    auto ns = mf.nspin();
    auto nkpts = mf.nkpts();
    auto nkpts_ibz = mf.nkpts_ibz();
    auto nbnd  = mf.nbnd();

    auto sHeff_skij = make_shared_array<local_Array_view_4D_t>(
        context.comm, context.internode_comm, context.node_comm,
        {ns, nkpts_ibz, nbnd, nbnd});

    if (context.node_comm.root()) {
      h5::file file(filename, 'r');
      auto grp = h5::group(file);
      auto Heff_loc = sHeff_skij.local();
      nda::h5_read(grp, dataset, Heff_loc);
      if (include_H0) {
        auto sys_grp = h5::group(file).open_group("system");
        nda::array<ComplexType, 4> H0_skij;
        nda::h5_read(sys_grp, "H0_skij", H0_skij);
        sHeff_skij.local() += H0_skij;
      }
    }
    context.node_comm.barrier();
    if (nkpts == nkpts_ibz)
      return sHeff_skij;

    auto sHeff_skij_full = make_shared_array<local_Array_view_4D_t>(
        context.comm, context.internode_comm, context.node_comm,
        {ns, nkpts, nbnd, nbnd});
    auto kp_to_ibz = mf.kp_to_ibz();
    auto kp_trev = mf.kp_trev();
    auto Heff = sHeff_skij.local();
    auto Heff_full = sHeff_skij_full.local();
    sHeff_skij_full.win().fence();
    for (size_t isk=context.node_comm.rank(); isk<ns*nkpts; isk+=context.node_comm.size()) {
      size_t is = isk/nkpts;
      size_t ik = isk%nkpts;
      size_t ik_ibz = kp_to_ibz(ik);

      if (kp_trev(ik))
        Heff_full(is, ik, nda::ellipsis{}) = nda::conj(Heff(is, ik_ibz, nda::ellipsis{}));
      else
        Heff_full(is, ik, nda::ellipsis{}) = Heff(is, ik_ibz, nda::ellipsis{});
    }
    sHeff_skij_full.win().fence();

    return sHeff_skij_full;
  }

  auto unfold_qp_energy(utils::mpi_context_t<mpi3::communicator> &context, mf::MF &mf, std::string filename, long iter)
  -> math::shm::shared_array<nda::array_view<ComplexType,3>> {
    using math::shm::make_shared_array;
    using local_Array_view_3D_t = nda::array_view<ComplexType, 3>;

    auto ns = mf.nspin();
    auto nkpts = mf.nkpts();
    auto nkpts_ibz = mf.nkpts_ibz();
    auto nbnd  = mf.nbnd();

    auto sE_ska = make_shared_array<local_Array_view_3D_t>(context.comm, context.internode_comm, context.node_comm,
                                                           {ns, nkpts_ibz, nbnd});
    if (context.node_comm.root()) {
      h5::file file(filename, 'r');
      auto iter_grp = h5::group(file).open_group("scf/iter"+std::to_string(iter));
      auto E_loc = sE_ska.local();
      if (iter_grp.has_subgroup("qp_approx"))
        nda::h5_read(iter_grp, "qp_approx/E_ska", E_loc);
      else
        nda::h5_read(iter_grp, "E_ska", E_loc);
    }
    context.node_comm.barrier();
    if (nkpts == nkpts_ibz)
      return sE_ska;

    auto sE_ska_full = make_shared_array<local_Array_view_3D_t>(context.comm, context.internode_comm, context.node_comm,
                                                                {ns, nkpts, nbnd});
    auto kp_to_ibz = mf.kp_to_ibz();
    auto kp_trev = mf.kp_trev();

    sE_ska_full.win().fence();
    for (size_t isk=context.comm.rank(); isk<ns*nkpts; isk+=context.comm.size()) {
      size_t is = isk/nkpts;
      size_t ik = isk%nkpts;
      size_t ik_ibz = kp_to_ibz(ik);
      if (kp_trev(ik))
        sE_ska_full.local()(is, ik, nda::range::all) = nda::conj(sE_ska.local()(is, ik_ibz, nda::range::all));
      else
        sE_ska_full.local()(is, ik, nda::range::all) = sE_ska.local()(is, ik_ibz, nda::range::all);
    }
    sE_ska_full.win().fence();
    sE_ska_full.all_reduce();

    return sE_ska_full;
  }

  void unfold_wfc(mf::MF &mf_sym, mf::MF &mf_nosym) {
    decltype(nda::range::all) all;
    utils::check(mf_sym.mf_type()==mf::mf_source_e::qe_source and mf_nosym.mf_type()==mf::mf_source_e::qe_source,
                 "unfold_wfc: mf_source != qe_source");
    utils::check(mf_sym.nkpts()==mf_nosym.nkpts(), "unfold_wfc: inconsistent nkpts between mf_sym and mf_nosym");
    utils::check(mf_sym.nspin()==mf_nosym.nspin(), "unfold_wfc: inconsistent nspin between mf_sym and mf_nosym");
    utils::check(mf_sym.nbnd()>=mf_nosym.nbnd(), "unfold_wfc: mf_sym.nbnd()<mf_nosym.nbnd()");
    utils::check(mf_sym.nbnd()>=mf_nosym.nbnd(), "unfold_wfc: mf_sym.nbnd()<mf_nosym.nbnd()");
    utils::check(mf_sym.mpi() == mf_nosym.mpi(),
                 "unfold_wfc: mf_sym and mf_nosym must have the same mpi context");

    auto mpi = mf_sym.mpi();

    size_t ns    = mf_sym.nspin();
    size_t nkpts = mf_sym.nkpts();
    size_t nbnd  = mf_sym.nbnd();
    size_t nbnd_nosym = mf_nosym.nbnd();
    auto wfc_g = mf_sym.wfc_truncated_grid();
    long ngm = wfc_g->size();
    auto fft2gv = wfc_g->fft_to_gv();  // maps index in fft grid to position in truncated grid

    // kpts_map between mf_sym and mf_nosym:
    // mf_nosym.kpts_crystal(ik) = mf_sym.kpts_crystal( kp_maps(ik) )
    nda::array<int, 1> kp_maps(nkpts);
    utils::calculate_kp_map(kp_maps, mf_nosym.kpts_crystal(), mf_sym.kpts_crystal());

    nda::array<ComplexType,1> *Xft = nullptr;
    nda::array<ComplexType,2> psi(nbnd,ngm);
    using view = nda::basic_array_view<RealType, 2, nda::C_layout, 'A', nda::default_accessor, nda::borrowed<>>;

    for (size_t isk2=mpi->comm.rank(); isk2<ns*nkpts; isk2+=mpi->comm.size()) {
      size_t is  = isk2 / nkpts;
      size_t ik2 = isk2 % nkpts;
      size_t ik2_sym = kp_maps(ik2);

      h5::file wfc_file(mf_nosym.outdir()+"/"+mf_nosym.prefix()+".save/wfc"+std::to_string(isk2+1)+".hdf5", 'a');
      auto grp = h5::group(wfc_file);

      // read miller indices for mf_nosym from file
      auto dset_info = h5::array_interface::get_dataset_info(grp,"/MillerIndices");
      long npw = dset_info.lengths[0]; 
      nda::array<int,2> mill(npw,3);
      nda::h5_read(grp, "/MillerIndices", mill);

      // index of miller indices in wfc_g fft grid
      nda::array<long,1> k2g(npw,0);
      utils::generate_k2g(mill,k2g,wfc_g->mesh());
      // shift k2g based on the difference between ik2 and ik2_sym
      utils::transform_k2g(false, mf_sym.symm_list(0),
                           mf_sym.kpts_crystal(ik2_sym)-mf_nosym.kpts_crystal(ik2),
                           wfc_g->mesh(), mf_sym.kpts(ik2_sym), k2g, Xft);

      // orbital at ik2_sym
      mf_sym.get_orbital_set('w', is, ik2_sym, nda::range(nbnd), psi(all, all));

      // move to the correct order provided by MillerIndices
      nda::array<ComplexType, 2> psi_k2(nbnd_nosym, npw);
      for ( auto [in,n] : itertools::enumerate(k2g) ) {
        long nn = fft2gv(n);
        utils::check(nn >= 0 and nn < ngm, 
                     "unfold_wfc: fail to map miller index to wfc_g truncated grid "); 
        psi_k2(all, in) = psi(nda::range(nbnd_nosym), nn);
      }

      // write orbitals to hdf5
      std::string attribute;
      view Ov({nbnd_nosym, 2*npw}, reinterpret_cast<RealType*>(psi_k2.data()) );
      {
        auto evc_ds = grp.open_dataset("/evc");
        h5::h5_read_attribute(evc_ds, "doc:", attribute);
      }
      nda::h5_write(grp, "/evc", Ov, false);
      {
        auto evc_ds = grp.open_dataset("/evc");
        h5::h5_write_attribute(evc_ds, "doc:", attribute);
      }
    }
    mpi->comm.barrier();
  }

  void check_is_G(double ni_d, double nj_d, double nk_d) {
    long ni_i = long(std::round(ni_d));
    long nj_i = long(std::round(nj_d));
    long nk_i = long(std::round(nk_d));

    utils::check(std::abs( ni_d - double(ni_i) ) < 1e-6, "check_is_G: not a G vector - ni: {}",ni_d);
    utils::check(std::abs( nj_d - double(nj_i) ) < 1e-6, "check_is_G: not a G vector - nj: {}",nj_d);
    utils::check(std::abs( nk_d - double(nk_i) ) < 1e-6, "check_is_G: not a G vector - nk: {}",nk_d);
  }


} // methods

namespace mf {
namespace detail {
  long match_fullbz_k(nda::array<double,2> const& kfull, nda::array<double,1> const& k, std::array<int,3> &G0) {
    for (long j=0; j<kfull.shape(0); ++j) {
      bool found = true;
      for (int d=0; d<3; ++d) {
        double x = k(d) - kfull(j,d);
        G0[d] = int(std::lround(x));
        if (std::abs(x - G0[d]) > 1e-6) found = false;
      }
      if (found) return j;
    }
    return -1;
  }
} // detail

namespace {
  // Returns the pp_type of the pseudopotential of the pw2coqui h5 file of mf, whose group Hamiltonian/<pp_type>/fullbz
  // holds the projectors at every full-BZ k (pw2coqui add_pp_fullbz). Called by unfold_mf on every rank before it
  // writes anything, so that a missing group (or a mean field that is not a pw2coqui h5) aborts without a partial file.
  std::string check_full_bz_projectors(MF const& mf) {
    bool from_h5 = mf.input_file_type() == mf::h5_input_type and mf.mf_type() != mf::pyscf_source;
    std::string fn = from_h5 ? mf.filename() : mf.outdir();
    std::string type;
    bool has_fullbz = false;
    if (from_h5) {
      h5::file f(fn, 'r');
      h5::group g(f);
      if (g.has_subgroup("Hamiltonian")) {
        h5::group pp = g.open_group("Hamiltonian");
        h5::h5_read_attribute(pp, "pp_type", type);
        has_fullbz = not type.empty() and pp.has_subgroup(type) and pp.open_group(type).has_subgroup("fullbz");
      }
    }
    utils::check(has_fullbz, "unfold_mf: {} has no full-BZ projectors (Hamiltonian/<pp_type>/fullbz); "
                 "rerun pw2coqui.x with add_pp_fullbz = .true.", fn);
    return type;
  }

  // Writes the nonlocal projectors of the unfolded mean field (number_of_kpoints, max_npw, npw, miller_k*, projector_k*
  // in tgrp) by copying, for every k of mf, those of the full-BZ k_j of fb with k = k_j + G0. The stored projector at
  // k_j with Miller index n is beta(k_j + n), so at k it has Miller index m = n - G0 (k + m = k_j + n).
  // Called by unfold_mf on the root rank; there is no other source of the projectors at the unfolded k, because the
  // projectors of a rotated k are not obtained from the IBZ ones by the symmetry operations of the orbitals alone.
  void copy_full_bz_projectors(MF const& mf, h5::group& fb, h5::group& tgrp) {
    long nkpts = mf.nkpts();
    int max_npw;
    nda::array<double,2> kfull;
    nda::array<int,1> npw_full;
    h5::h5_read_attribute(fb, "max_npw", max_npw);
    nda::h5_read(fb, "kpoints", kfull);
    nda::h5_read(fb, "npw", npw_full);
    h5::h5_write_attribute(tgrp, "number_of_kpoints", int(nkpts));
    h5::h5_write_attribute(tgrp, "max_npw", max_npw);
    nda::array<int,1> npw(nkpts);
    nda::array<int,2> mill;
    nda::array<ComplexType,2> proj;
    for (long ik=0; ik<nkpts; ++ik) {
      // Full-BZ k_j equal to k modulo a reciprocal lattice vector G0
      std::array<int,3> G0;
      nda::array<double,1> k(mf.kpts_crystal(ik));
      long j = detail::match_fullbz_k(kfull, k, G0);
      utils::check(j >= 0, "unfold_mf: k-point {} ({}, {}, {}) is not in the full-BZ k list of the projectors.",
                   ik, k(0), k(1), k(2));
      // Shift the Miller indices, m = n - G0
      nda::h5_read(fb, "miller_k"+std::to_string(j), mill);
      for (long i=0; i<mill.shape(0); ++i)
        for (int d=0; d<3; ++d) mill(i,d) -= G0[d];
      nda::h5_read(fb, "projector_k"+std::to_string(j), proj);
      npw(ik) = npw_full(j);
      nda::h5_write(tgrp, "miller_k"+std::to_string(ik), mill);
      nda::h5_write(tgrp, "projector_k"+std::to_string(ik), proj);
    }
    nda::h5_write(tgrp, "npw", npw);
  }
} // anonymous

  void unfold_mf(MF &mf, std::string filename) {
    auto mpi = mf.mpi();
    utils::check(mf.nkpts() != mf.nkpts_ibz(),
                 "unfold_mf: the mean-field system has no symmetry reduction (nkpts = nkpts_ibz = {}).", mf.nkpts());
    utils::check(not mf.is_augmented(), "unfold_mf: augmented orbital bases are not supported.");
    utils::check(mf.nbnd_aux() == 0, "unfold_mf: nbnd_aux > 0 is not supported.");
    utils::check(mf.npol() == 1, "unfold_mf: npol > 1 is not supported.");
    std::string pp_type = check_full_bz_projectors(mf);

    bdft::bdft_system sys(mf, filename, false);
    {
      // full-BZ k list in the order of mf, identity as the only symmetry
      nda::array<double,2> latt(3,3), recv(3,3);
      latt() = mf.lattv();
      recv() = mf.recv();
      nda::array<double,1> kp_grid(3);
      for (int i=0; i<3; ++i) kp_grid(i) = mf.kp_grid()(i);
      nda::array<double,2> kpts(mf.kpts());
      nda::stack_array<double,3,3> E;
      E() = 0.0;
      for (int i=0; i<3; ++i) E(i,i) = 1.0;
      nda::stack_array<double,3> ft;
      ft() = 0.0;
      std::vector<utils::symm_op> symm_list;
      symm_list.emplace_back(utils::symm_op{E, E, ft});
      sys._symm = bz_symm(mpi->comm, true, latt, recv, kp_grid, kpts, symm_list, false);
    }
    // k_weight of mf is per IBZ k; the unfolded mean field has nkpts equivalent k
    sys.k_weight = nda::array<double,1>(mf.nkpts(), 1.0/double(mf.nkpts()));
    long nkpts = mf.nkpts();
    utils::check(sys._symm.nkpts == nkpts and sys._symm.nkpts_ibz == nkpts,
                 "unfold_mf: failed to build the full-BZ k list.");

    auto const* wfc_g = mf.wfc_truncated_grid();
    long ngm = wfc_g->size();
    long nbnd = mf.nbnd();
    long nspin = mf.nspin();
    auto& comm = mpi->comm;

    std::optional<h5::file> file;
    std::optional<h5::group> ogrp;
    if (comm.root()) {
      file.emplace(filename, 'w');
      h5::group grp(*file);
      sys.save(grp);
      ogrp.emplace(grp.open_group("Orbitals"));
      bdft::detail::write_wfc_metadata(*ogrp, *wfc_g);
    }
    // (s,k) pairs are read round-robin over ranks and written by the root
    nda::array<ComplexType,2> psik(nbnd, ngm);
    for (long is=0; is<nspin; ++is) {
      for (long ik=0; ik<nkpts; ++ik) {
        int owner = int((is*nkpts + ik) % comm.size());
        if (comm.rank() == owner)
          mf.get_orbital_set('w', is, ik, nda::range(nbnd), psik);
        if (owner != 0) {
          if (comm.rank() == owner)
            comm.send_n(psik.data(), psik.size(), 0, 0);
          else if (comm.root())
            comm.receive_n(psik.data(), psik.size(), owner, 0);
        }
        if (comm.root())
          nda::h5_write(*ogrp, "psi_s"+std::to_string(is)+"_k"+std::to_string(ik), psik, false);
      }
    }
    if (comm.root()) {
      // Local pseudopotential from the source file; nonlocal projectors from its full-BZ group
      h5::group grp(*file);
      grp.create_group("Hamiltonian");
      hamilt::pseudopot_to_h5(mf.fft_grid_dim(), grp, mf.filename(), mf::h5_input_type, false);
      h5::file f_src(mf.filename(), 'r');
      h5::group fb = h5::group(f_src).open_group("Hamiltonian").open_group(pp_type).open_group("fullbz");
      h5::group tgrp = grp.open_group("Hamiltonian").open_group(pp_type);
      copy_full_bz_projectors(mf, fb, tgrp);
    }
    ogrp.reset();
    file.reset();
    comm.barrier();
  }
} // mf
