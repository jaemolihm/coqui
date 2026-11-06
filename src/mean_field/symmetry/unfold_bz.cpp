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
#include "utilities/ylm_qe.h"
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
namespace {
  // atomic positions in crystal coordinates, tau_cryst(a,i) = sum_j tau_cart(a,j) latt_inv(j,i)
  nda::array<double,2> atomic_positions_crystal(MF const& mf) {
    long nat = mf.number_of_atoms();
    nda::matrix<double> latt_inv(3,3);
    for (int i=0; i<3; ++i) for (int j=0; j<3; ++j) latt_inv(i,j) = mf.lattv(i,j);
    nda::inverse3_in_place(latt_inv);
    nda::array<double,2> tau(nat,3);
    for (long a=0; a<nat; ++a)
      for (int i=0; i<3; ++i) {
        tau(a,i) = 0.0;
        for (int j=0; j<3; ++j) tau(a,i) += mf.atomic_positions(a)(j) * latt_inv(j,i);
      }
    return tau;
  }

  // perm(a) = b with R tau_a = tau_b + T(a,:), T a lattice vector, in crystal coordinates (R acts on column vectors,
  // as in the real-space map r' = R r that goes with the G' = G.Rinv map of the orbitals).
  struct atom_map_t {
    nda::array<int,1> perm;
    nda::array<long,2> T;
  };
  atom_map_t atom_map(MF const& mf, nda::array<double,2> const& tau, utils::symm_op const& S) {
    long nat = mf.number_of_atoms();
    atom_map_t m{nda::array<int,1>(nat), nda::array<long,2>(nat,3)};
    for (long a=0; a<nat; ++a) {
      nda::stack_array<double,3> x;
      for (int i=0; i<3; ++i) {
        x(i) = 0.0;
        for (int j=0; j<3; ++j) x(i) += S.R(i,j) * tau(a,j);
      }
      long found = -1;
      for (long b=0; b<nat; ++b) {
        double d2 = 0.0;
        for (int i=0; i<3; ++i) {
          double d = x(i) - tau(b,i);
          d -= std::round(d);
          d2 += d*d;
        }
        if (d2 < 1e-8 and mf.atomic_id(b) == mf.atomic_id(a)) { found = b; break; }
      }
      utils::check(found >= 0, "unfold_mf: the symmetry operation does not map atom {} onto an atom of the same species.", a);
      m.perm(a) = found;
      for (int i=0; i<3; ++i) m.T(a,i) = std::lround(x(i) - tau(found,i));
    }
    return m;
  }

  // Angular momentum of every projector of one atom of each species, from the stored projectors. The stored projector
  // of atom a, radial function i, (l,m) is  beta(k+G) = (-i)^l F_i(|q|) Y_lm(q^) exp(-i q.tau_a), q = k+G, Y_lm the
  // ylmr2 harmonics (QE vkb). The projectors j..j+2l of an atom form an (l) shell if, at every q of every IBZ k, the
  // vector g_m(q) = beta_{j+m}(q) exp(+i q.tau_a) is parallel to (Y_l0(q^), ..., Y_l,2l(q^)), i.e. g_m = c(q) Y_lm with
  // one c(q) for all m, and c is the same for all q of equal |q| (of the same k; F depends on |q| only). The
  // projectors of an atom are consumed in QE order (radial function by radial function, m = 0, (cos,sin) of m = 1, ...
  // of each), trying l from high to low since the parallel test alone accepts any projector as l = 0. Aborts if
  // the l of a projector cannot be decided.
  // @return l_of[species][projector within the atom]
  std::vector<std::vector<int>> projector_l(MF const& mf, h5::group& tgrp, nda::array<double,2> const& tau,
                                            nda::array<int,1> const& ofs, nda::array<int,1> const& nh,
                                            nda::array<int,1> const& ityp, long nkb, long npwx) {
    constexpr int lmax = 3;
    long nsp = nh.size(), nat = ofs.size();
    if (nsp == 0) return {};
    long nhmax = *std::max_element(nh.begin(), nh.end());
    const double pi = std::acos(-1.0);
    // first atom of each species
    std::vector<long> atom_of(nsp, -1);
    for (long a=nat-1; a>=0; --a) atom_of[ityp(a)] = a;
    // is_shell[(s*nhmax + j)*(lmax+1) + l]: projectors j..j+2l of species s are an l shell at all q seen so far
    auto idx = [&](long s, long j, int l) { return (s*nhmax + j)*(lmax+1) + l; };
    std::vector<char> is_shell(nsp*nhmax*(lmax+1), 1);
    nda::array<int,1> npw_all(mf.nkpts_ibz());
    nda::h5_read(tgrp, "npw", npw_all);
    for (long ik=0; ik<mf.nkpts_ibz(); ++ik) {
      int npw = npw_all(ik);
      nda::array<int,2> mill(npw,3);
      nda::h5_read(tgrp, "miller_k"+std::to_string(ik), mill);
      nda::array<ComplexType,2> proj(nkb, npwx);
      nda::h5_read(tgrp, "projector_k"+std::to_string(ik), proj);
      // q in crystal coordinates (phases) and cartesian (harmonics)
      nda::array<double,2> qc(npw,3), qcart(npw,3);
      nda::array<double,1> q2(npw);
      for (int i=0; i<npw; ++i)
        for (int d=0; d<3; ++d) {
          qc(i,d) = mf.kpts_crystal(ik, d) + mill(i,d);
          qcart(i,d) = 0.0;
          for (int e=0; e<3; ++e) qcart(i,d) += (mf.kpts_crystal(ik, e) + mill(i,e)) * mf.recv(e,d);
        }
      for (int i=0; i<npw; ++i) q2(i) = qcart(i,0)*qcart(i,0) + qcart(i,1)*qcart(i,1) + qcart(i,2)*qcart(i,2);
      // shells of equal |q|
      std::vector<int> order(npw);
      for (int i=0; i<npw; ++i) order[i] = i;
      std::sort(order.begin(), order.end(), [&](int x, int y) { return q2(x) < q2(y); });
      std::vector<int> shell(npw);
      int nshell = 0;
      for (int ii=0; ii<npw; ++ii) {
        if (ii > 0 and std::abs(q2(order[ii]) - q2(order[ii-1])) > 1e-9*std::max(1.0, q2(order[ii]))) ++nshell;
        shell[order[ii]] = nshell;
      }
      ++nshell;
      nda::array<double,2> ylm(npw, (lmax+1)*(lmax+1));
      utils::ylmr2_qe(lmax, qcart(), ylm());
      for (long s=0; s<nsp; ++s) {
        long a = atom_of[s];
        if (a < 0) continue;
        // g(j,q) = beta_j(q) exp(+i q.tau_a)
        nda::array<ComplexType,2> g(nh(s), npw);
        for (long j=0; j<nh(s); ++j)
          for (int i=0; i<npw; ++i) {
            double arg = 0.0;
            for (int d=0; d<3; ++d) arg += qc(i,d)*tau(a,d);
            g(j,i) = proj(ofs(a)+j, i) * std::exp(ComplexType(0.0, 2.0*pi*arg));
          }
        for (long j=0; j<nh(s); ++j)
          for (int l=0; l<=lmax; ++l) {
            if (not is_shell[idx(s,j,l)]) continue;
            if (j + 2*l + 1 > nh(s)) { is_shell[idx(s,j,l)] = 0; continue; }
            double gmax = 0.0;
            for (int m=0; m<2*l+1; ++m)
              for (int i=0; i<npw; ++i) gmax = std::max(gmax, std::abs(g(j+m,i)));
            if (gmax <= 0.0) { is_shell[idx(s,j,l)] = 0; continue; }
            std::vector<ComplexType> c_shell(nshell, ComplexType(0.0));
            std::vector<char> has_shell(nshell, 0);
            for (int i=0; i<npw; ++i) {
              // c = <Y,g>/<Y,Y> and the distance of g from c Y
              double yy = 0.0;
              ComplexType yg = 0.0;
              for (int m=0; m<2*l+1; ++m) { double y = ylm(i, l*l+m); yy += y*y; yg += y*g(j+m,i); }
              ComplexType c = yg/yy;
              double res2 = 0.0;
              for (int m=0; m<2*l+1; ++m) res2 += std::norm(g(j+m,i) - c*ylm(i, l*l+m));
              bool bad = std::sqrt(res2) > 1e-5*gmax;
              // c is the same on the q of a shell
              if (not has_shell[shell[i]]) { c_shell[shell[i]] = c; has_shell[shell[i]] = 1; }
              else if (std::abs(c - c_shell[shell[i]])*std::sqrt(yy) > 1e-5*gmax) bad = true;
              if (bad) { is_shell[idx(s,j,l)] = 0; break; }
            }
          }
      }
    }
    std::vector<std::vector<int>> l_of(nsp);
    for (long s=0; s<nsp; ++s) {
      long j = 0;
      while (j < nh(s)) {
        int found = -1;
        for (int l=lmax; l>=0 and found < 0; --l)
          if (is_shell[idx(s,j,l)]) found = l;
        utils::check(found >= 0, "unfold_mf: cannot assign an angular momentum to projector {} of species {} from the "
                     "stored projectors (not of the form F(|q|) Y_lm(q) exp(-i q.tau) in the QE convention).", j, s);
        for (int m=0; m<2*found+1; ++m) l_of[s].push_back(found);
        j += 2*found + 1;
      }
    }
    return l_of;
  }
} // anonymous

  nda::matrix<double> cartesian_rotation(MF const& mf, utils::symm_op const& S) {
    // x_cart = L^T x_cryst with L(i,j) = lattv(i,j), so R_cart = L^T R L^-T
    nda::matrix<double> Lt(3,3), Ltinv(3,3), R(3,3);
    for (int i=0; i<3; ++i) for (int j=0; j<3; ++j) { Lt(i,j) = mf.lattv(j,i); R(i,j) = S.R(i,j); }
    Ltinv = nda::inverse(Lt);
    nda::matrix<double> Rc = Lt * R * Ltinv;
    for (int i=0; i<3; ++i)
      for (int j=0; j<3; ++j) {
        double rtr = 0.0;
        for (int k=0; k<3; ++k) rtr += Rc(k,i) * Rc(k,j);
        utils::check(std::abs(rtr - (i == j ? 1.0 : 0.0)) < 1e-8,
                     "cartesian_rotation: the rotation is not orthogonal in cartesian coordinates.");
      }
    return Rc;
  }

namespace {
  // Nonlocal projectors of the IBZ mf, as written by write_pseudopot
  struct projector_layout_t {
    int nk_ibz, npwx, nkb, nat;
    nda::array<int,1> npw_ibz, nh, ofs, ityp;
    std::vector<std::vector<int>> l_of;  // [species][projector within the atom]
  };

  // Writes the pseudopotential of mf in bdft format to filename; false if mf has none.
  bool write_ibz_pseudopot(MF &mf, std::string const& filename) {
    {
      h5::file f_ibz(filename, 'w');
      h5::group g_ibz(f_ibz);
      bdft::detail::write_pseudopot(mf, g_ibz);
    }
    h5::file f_ibz(filename, 'r');
    return h5::group(f_ibz).has_subgroup("Hamiltonian");
  }

  projector_layout_t read_projector_layout(MF const& mf, h5::group& tgrp_ibz, nda::array<double,2> const& tau) {
    projector_layout_t L;
    h5::h5_read_attribute(tgrp_ibz, "number_of_kpoints", L.nk_ibz);
    h5::h5_read_attribute(tgrp_ibz, "max_npw", L.npwx);
    h5::h5_read_attribute(tgrp_ibz, "total_num_of_proj", L.nkb);
    h5::h5_read_attribute(tgrp_ibz, "number_of_atoms", L.nat);
    utils::check(L.nk_ibz == mf.nkpts_ibz() and L.nat == mf.number_of_atoms(),
                 "unfold_mf: pseudopotential is inconsistent with the mean field.");
    L.npw_ibz = nda::array<int,1>(L.nk_ibz);
    L.nh = nda::array<int,1>(mf.number_of_species());
    L.ofs = nda::array<int,1>(L.nat);
    L.ityp = nda::array<int,1>(L.nat);
    nda::h5_read(tgrp_ibz, "npw", L.npw_ibz);
    nda::h5_read(tgrp_ibz, "projector_offset", L.ofs);
    nda::h5_read(tgrp_ibz, "atomic_id", L.ityp);
    nda::h5_read(tgrp_ibz, "proj_per_atom", L.nh);
    int id_min = *std::min_element(L.ityp.begin(), L.ityp.end());
    utils::check(id_min == 0 or id_min == 1, "unfold_mf: invalid atomic_id array: min id {}.", id_min);
    L.ityp() -= id_min;
    L.l_of = projector_l(mf, tgrp_ibz, tau, L.ofs, L.nh, L.ityp, L.nkb, L.npwx);
    return L;
  }

  // Action of one symmetry operation on the projectors
  struct op_projector_t {
    nda::array<int,1> inv_perm;  // block b of the unfolded mean field <- block inv_perm(b) of mf
    nda::array<long,2> T;        // T(a,:) = R tau_a - tau_perm(a)
    std::array<nda::array<double,2>,4> D;  // D^l(R)
  };
  op_projector_t make_op_projector(MF const& mf, nda::array<double,2> const& tau, utils::symm_op const& S) {
    long nat = mf.number_of_atoms();
    auto m = atom_map(mf, tau, S);
    op_projector_t op{nda::array<int,1>(nat), m.T, {}};
    for (int a=0; a<nat; ++a) op.inv_perm(m.perm(a)) = a;
    auto Rc = cartesian_rotation(mf, S);
    for (int l=0; l<4; ++l) op.D[l] = utils::ylm_rotation_qe(l, Rc);
    return op;
  }

  // Projectors at the unfolded k' (crystal coordinates kunfold) from the projectors of the IBZ k of the operation; the
  // projector (atom a, radial function, l, m) at the IBZ k is
  //   beta_a,lm(q) = (-i)^l F(|q|) Y_lm(q^) exp(-i q.tau_a),  q = k+G   (QE vkb)
  // and the true projector of the symmetric system at the unfolded k' = R k (R the cartesian rotation of the operation;
  // k' = -R k with q' = -R q at time-reversal k, where beta(-q) = conj(beta(q))) on the atom b = perm(a),
  // R tau_a = tau_b + T, is, with q' = R q (G' = G.Rinv):
  //   beta_b,lm(q') = exp(+i k'.T) sum_m' D^l(R)(m,m') beta_a,lm'(q),   Y_lm(R q^) = sum_m' D^l(m,m') Y_lm'(q^),
  // (conj of beta_a at time-reversal k; D^l real; (-i)^l commutes with D^l). D^l is in the QE convention
  // (utils::ylm_rotation_qe), and the atom blocks are permuted, so block b of the unfolded mean field holds the projectors
  // centred on atom b. Symmorphic operations only.
  void unfold_projectors_k(projector_layout_t const& L, op_projector_t const& op, bool trev,
                           nda::array<double,1> const& kunfold, long npw,
                           nda::array<ComplexType,2> const& proj_ibz, nda::array<ComplexType,2> &proj_unfold) {
    proj_unfold() = 0.0;
    for (int b=0; b<L.nat; ++b) {
      int a = op.inv_perm(b);
      long n_proj = L.nh(L.ityp(b));
      utils::check(L.nh(L.ityp(a)) == n_proj and L.ofs(a) + n_proj <= L.nkb and L.ofs(b) + n_proj <= L.nkb,
                   "unfold_mf: inconsistent projector layout.");
      // exp(+i k'.T), T = R tau_a - tau_b
      double kT = 0.0;
      for (int d=0; d<3; ++d) kT += kunfold(d) * double(op.T(a,d));
      ComplexType phase = std::exp(ComplexType(0.0, 2.0*std::acos(-1.0)*kT));
      auto const& lv = L.l_of[L.ityp(b)];
      for (long j0=0; j0<n_proj; j0 += 2*lv[j0]+1) {
        int l = lv[j0];
        auto const& D = op.D[l];
        for (int m1=0; m1<2*l+1; ++m1)
          for (long i=0; i<npw; ++i) {
            ComplexType v = 0.0;
            for (int m2=0; m2<2*l+1; ++m2) {
              ComplexType x = proj_ibz(L.ofs(a)+j0+m2, i);
              v += D(m1,m2) * (trev ? std::conj(x) : x);
            }
            proj_unfold(L.ofs(b)+j0+m1, i) = phase * v;
          }
      }
    }
  }
} // anonymous

  void unfold_mf(MF &mf, std::string filename) {
    auto mpi = mf.mpi();
    utils::check(mf.nkpts() != mf.nkpts_ibz(),
                 "unfold_mf: the mean-field system has no symmetry reduction (nkpts = nkpts_ibz = {}).", mf.nkpts());
    utils::check(not mf.is_augmented(), "unfold_mf: augmented orbital bases are not supported.");
    utils::check(mf.nbnd_aux() == 0, "unfold_mf: nbnd_aux > 0 is not supported.");
    utils::check(mf.npol() == 1, "unfold_mf: npol > 1 is not supported.");

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
      h5::group grp(*file);
      grp.create_group("Hamiltonian");
      // The IBZ pseudopotential is written to a scratch file, then copied to the unfolded mean field with its nonlocal
      // projectors unfolded (unfold_projectors_k).
      std::string fn_ibz = filename + ".pp_ibz.tmp";
      bool has_pp = write_ibz_pseudopot(mf, fn_ibz);
      if (has_pp) {
        hamilt::pseudopot_to_h5(mf.fft_grid_dim(), grp, fn_ibz, mf::h5_input_type, false);
        h5::file f_ibz(fn_ibz, 'r');
        h5::group pp_ibz = h5::group(f_ibz).open_group("Hamiltonian");
        std::string type;
        h5::h5_read_attribute(pp_ibz, "pp_type", type);
        h5::group tgrp_ibz = pp_ibz.open_group(type);
        h5::group tgrp = grp.open_group("Hamiltonian").open_group(type);

        auto tau = atomic_positions_crystal(mf);
        auto L = read_projector_layout(mf, tgrp_ibz, tau);
        nda::array<int,1> npw_unfold(nkpts);
        for (long ik=0; ik<nkpts; ++ik) npw_unfold(ik) = L.npw_ibz(mf.kp_to_ibz(ik));
        h5::h5_write_attribute(tgrp, "number_of_kpoints", int(nkpts));
        h5::h5_write_attribute(tgrp, "max_npw", L.npwx);
        nda::h5_write(tgrp, "npw", npw_unfold);

        auto symm_list = mf.symm_list();
        std::vector<op_projector_t> ops;
        for (auto const& S : symm_list) ops.push_back(make_op_projector(mf, tau, S));
        nda::stack_array<double,3> Gs;
        Gs() = 0.0;
        nda::array<ComplexType,2> proj_ibz(L.nkb, L.npwx), proj_unfold(L.nkb, L.npwx);
        for (long ik=0; ik<nkpts; ++ik) {
          long ik_ibz = mf.kp_to_ibz(ik);
          long isym = mf.kp_symm(ik);
          nda::array<int,2> mill(npw_unfold(ik), 3);
          nda::h5_read(tgrp_ibz, "miller_k"+std::to_string(ik_ibz), mill);
          utils::transform_miller_indices(mf.kp_trev(ik), symm_list[isym], Gs, mill);
          nda::h5_write(tgrp, "miller_k"+std::to_string(ik), mill);

          nda::h5_read(tgrp_ibz, "projector_k"+std::to_string(ik_ibz), proj_ibz);
          nda::array<double,1> kunfold(mf.kpts_crystal(ik));
          unfold_projectors_k(L, ops[isym], mf.kp_trev(ik), kunfold, npw_unfold(ik), proj_ibz, proj_unfold);
          nda::h5_write(tgrp, "projector_k"+std::to_string(ik), proj_unfold);
        }
      }
      std::filesystem::remove(fn_ibz);
    }
    ogrp.reset();
    file.reset();
    comm.barrier();
  }

  namespace detail {
  projector_residual_t projector_little_group_residual(MF &mf, std::string fn_scratch) {
    auto& comm = mf.mpi()->comm;
    double rel_err = 0.0;
    long n_checks = 0;
    int l_max = 0;
    if (comm.root()) {
      bool has_pp = write_ibz_pseudopot(mf, fn_scratch);
      utils::check(has_pp, "projector_little_group_residual: no pseudopotential.");
      h5::file f_ibz(fn_scratch, 'r');
      h5::group pp_ibz = h5::group(f_ibz).open_group("Hamiltonian");
      std::string type;
      h5::h5_read_attribute(pp_ibz, "pp_type", type);
      h5::group tgrp = pp_ibz.open_group(type);
      auto tau = atomic_positions_crystal(mf);
      auto L = read_projector_layout(mf, tgrp, tau);
      for (auto const& lv : L.l_of) for (int l : lv) l_max = std::max(l_max, l);
      auto symm_list = mf.symm_list();
      std::vector<op_projector_t> ops;
      for (auto const& S : symm_list) ops.push_back(make_op_projector(mf, tau, S));
      nda::stack_array<double,3> Gs;
      Gs() = 0.0;
      nda::array<ComplexType,2> proj_ibz(L.nkb, L.npwx), proj_unfold(L.nkb, L.npwx);
      for (long ik=0; ik<mf.nkpts_ibz(); ++ik) {
        utils::check(mf.kp_to_ibz(ik) == ik, "projector_little_group_residual: IBZ k-points are not the first nkpts_ibz.");
        long npw = L.npw_ibz(ik);
        nda::array<int,2> mill(npw, 3);
        nda::h5_read(tgrp, "miller_k"+std::to_string(ik), mill);
        nda::h5_read(tgrp, "projector_k"+std::to_string(ik), proj_ibz);
        std::map<std::array<int,3>, long> index_of;
        for (long i=0; i<npw; ++i) index_of[{mill(i,0), mill(i,1), mill(i,2)}] = i;
        for (long isym=0; isym<long(symm_list.size()); ++isym)
          for (int trev=0; trev<2; ++trev) {
            // unfolded k' = (+-) k.Rinv (G' = G.Rinv); only the operations with k' = k + G0 map the k to itself
            nda::array<double,1> kunfold(3);
            std::array<int,3> G0;
            bool little = true;
            for (int d=0; d<3; ++d) {
              kunfold(d) = 0.0;
              for (int e=0; e<3; ++e) kunfold(d) += mf.kpts_crystal(ik, e) * symm_list[isym].Rinv(e,d);
              if (trev) kunfold(d) = -kunfold(d);
              double dk = kunfold(d) - mf.kpts_crystal(ik, d);
              G0[d] = int(std::lround(dk));
              if (std::abs(dk - G0[d]) > 1e-8) little = false;
            }
            if (not little) continue;
            nda::array<int,2> mill_unfold(mill);
            utils::transform_miller_indices(bool(trev), symm_list[isym], Gs, mill_unfold);
            unfold_projectors_k(L, ops[isym], bool(trev), kunfold, npw, proj_ibz, proj_unfold);
            // the unfolded projector at q' = k' + G' is the stored one at k + G, G = G' + G0
            double err2 = 0.0, ref2 = 0.0;
            for (long i=0; i<npw; ++i) {
              auto it = index_of.find({mill_unfold(i,0)+G0[0], mill_unfold(i,1)+G0[1], mill_unfold(i,2)+G0[2]});
              utils::check(it != index_of.end(), "projector_little_group_residual: rotated G is outside the sphere.");
              for (long j=0; j<L.nkb; ++j) {
                err2 += std::norm(proj_unfold(j,i) - proj_ibz(j,it->second));
                ref2 += std::norm(proj_ibz(j,it->second));
              }
            }
            rel_err = std::max(rel_err, std::sqrt(err2/ref2));
            ++n_checks;
          }
      }
    }
    comm.broadcast_n(&rel_err, 1, 0);
    comm.broadcast_n(&n_checks, 1, 0);
    comm.broadcast_n(&l_max, 1, 0);
    if (comm.root()) std::filesystem::remove(fn_scratch);
    comm.barrier();
    return {rel_err, n_checks, l_max};
  }
  } // detail
} // mf
