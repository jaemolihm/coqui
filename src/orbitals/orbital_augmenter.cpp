/**
 * ==========================================================================
 * CoQuí: Correlated Quantum ínterface
 *
 * Copyright (c) 2022-2025 Simons Foundation & The CoQuí developer team
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

#include <cmath>
#include <string>
#include <vector>
#include <algorithm>
#include <memory>
#include <optional>
#include <unordered_map>

#include "configuration.hpp"
#include "IO/AppAbort.hpp"
#include "IO/app_loggers.h"
#include "utilities/check.hpp"
#include "utilities/concepts.hpp"
#include "utilities/kpoint_utils.hpp"
#include "utilities/lr_utils.hpp"

#include "mpi3/environment.hpp"
#include "mpi3/communicator.hpp"

#include "h5/h5.hpp"
#include "nda/nda.hpp"
#include "nda/h5.hpp"
#include "nda/blas.hpp"
#include "nda/linalg.hpp"
#include "numerics/nda_functions.hpp"
#include "numerics/distributed_array/nda.hpp"
#include "numerics/distributed_array/nda_utils.hpp"
#include "numerics/shared_array/nda.hpp"
#include "grids/g_grids.hpp"
#include "mean_field/MF.hpp"
#include "mean_field/distributed_orbital_readers.hpp"
#include "hamiltonian/pseudo/pseudopot.h"
#include "hamiltonian/one_body_hamiltonian.hpp"
#include "itertools/itertools.hpp"

#include "orbitals/orbital_augmenter.h"

// δψ HDF5 reader (header-only, methods::detail): read_Deltapsi_k is a
// self-contained inline helper, and the deltapsi file format (MillerIndices +
// interleaved evc) is identical to the LR δψ files.
#include "methods/ERI/lr_psi_io.hpp"

namespace orbitals
{

using boost::mpi3::communicator;
using memory::darray_t;
using memory::host_array;

momentum_augmenter::momentum_augmenter(mf::MF& mf, std::vector<int> dirs)
  : _dirs(std::move(dirs))
{
  if(_dirs.empty()) _dirs = {0,1,2};
  for(int d : _dirs)
    utils::check(d >= 0 and d <= 2,
                 "momentum_augmenter: direction {} out of range (expected 0,1,2).", d);
  auto const& wfc_g = *mf.wfc_truncated_grid();
  long ngm = wfc_g.size();
  _miller = nda::array<int,2>(ngm,3);
  utils::generate_miller_index(wfc_g.gv_to_fft(), _miller, wfc_g.mesh());
  _recv = mf.recv();
  _kcart = mf.kpts_ibz();
  utils::check(_kcart.extent(0) == mf.nkpts_ibz() and _kcart.extent(1) == 3,
               "momentum_augmenter: kpts_ibz shape mismatch.");
}

void momentum_augmenter::generate_raw(int ispin, int ik, long g0,
                                      nda::array_const_view<ComplexType,2> psi_base,
                                      nda::array_view<ComplexType,2> raw_out) const
{
  (void) ispin;
  long nbnd_aug = psi_base.shape(0);
  long ng_loc = psi_base.shape(1);
  utils::check(g0 >= 0 and g0 + ng_loc <= _miller.shape(0),
               "momentum_augmenter: G slice [{},{}) out of range (ngm {}).",
               g0, g0+ng_loc, _miller.shape(0));
  long n_dir = long(_dirs.size());
  utils::check(raw_out.shape(0) == n_dir*nbnd_aug and raw_out.shape(1) == ng_loc,
               "momentum_augmenter: raw_out shape mismatch.");
  // Channel c indexes the selected directions (compact); alpha is the actual
  // Cartesian direction _dirs[c]. Raw states of band b for channel c are placed
  // at row c*nbnd_aug + b (matching add_augmentation's layout).
  for(long c=0; c<n_dir; ++c) {
    int alpha = _dirs[c];
    for(long ig=0; ig<ng_loc; ++ig) {
      long g = g0 + ig;
      double kpG = _kcart(ik,alpha)
                 + double(_miller(g,0)) * _recv(0,alpha)
                 + double(_miller(g,1)) * _recv(1,alpha)
                 + double(_miller(g,2)) * _recv(2,alpha);
      for(long b=0; b<nbnd_aug; ++b)
        raw_out(c*nbnd_aug + b, ig) = kpG * psi_base(b, ig);
    }
  }
}

std::shared_ptr<orbital_augmenter_t> make_augmenter(mf::MF& mf, ptree const& pt,
                                                    std::vector<int> const& dirs)
{
  auto type = io::get_value_with_default<std::string>(pt, "type", "momentum");
  io::tolower(type);
  if(type == "momentum") {
    return std::make_shared<momentum_augmenter>(mf, dirs);
  } else if(type == "dpsi") {
    APP_ABORT("make_augmenter: augment type 'dpsi' not implemented yet.");
  } else {
    APP_ABORT("make_augmenter: unknown augment type '{}'. Valid: 'momentum'.", type);
  }
  return nullptr;
}

template<MEMORY_SPACE MEM, utils::Communicator comm_t>
auto orthonormalize_augmentation(
                    darray_t<memory::array<MEM, ComplexType, 4>, comm_t>& psi_orig,
                    darray_t<memory::array<MEM, ComplexType, 4>, comm_t>& raw_aug,
                    double epstol)
  -> std::tuple<darray_t<memory::array<MEM, ComplexType, 4>, comm_t>,
                nda::array<double, 3>>
{
  using larray = memory::array<MEM, ComplexType, 4>;
  decltype(nda::range::all) all;
  auto comm = psi_orig.communicator();

  long nspin_tot = psi_orig.global_shape()[0];
  long nkpts_tot = psi_orig.global_shape()[1];
  long nbnd      = psi_orig.global_shape()[2];
  long n_raw     = raw_aug.global_shape()[2];
  long ngm       = psi_orig.global_shape()[3];

  utils::check(psi_orig.grid()[2]==1 and raw_aug.grid()[2]==1,
               "orthonormalize_augmentation: band distribution not supported.");
  utils::check(raw_aug.global_shape()[0]==nspin_tot and raw_aug.global_shape()[1]==nkpts_tot
               and raw_aug.global_shape()[3]==ngm, "orthonormalize_augmentation: shape mismatch.");
  utils::check(psi_orig.grid()==raw_aug.grid(), "orthonormalize_augmentation: grid mismatch.");
  utils::check(psi_orig.local_range(3)==raw_aug.local_range(3),
               "orthonormalize_augmentation: G-range mismatch.");
  utils::check(n_raw>0 and nbnd>0, "orthonormalize_augmentation: empty input.");

  long color = psi_orig.origin()[0]*nkpts_tot + psi_orig.origin()[1];
  auto k_comm = comm->split(color, comm->rank());

  auto s_range = psi_orig.local_range(0);
  auto k_range = psi_orig.local_range(1);
  long ns = s_range.size();
  long nk = k_range.size();

  auto orig_all = psi_orig.local();   // (ns, nk, nbnd, ng_loc)
  auto aug_all  = raw_aug.local();    // (ns, nk, n_raw, ng_loc)

  // pass 1: project against originals, form residual overlap, diagonalize
  std::vector<nda::array<ComplexType,2>> Vs;  // eigenvectors as rows, per local (is,ik)
  std::vector<nda::array<double,1>>      Es;   // eigenvalues (ascending), per local (is,ik)
  std::vector<long>                      counts;
  Vs.reserve(ns*nk); Es.reserve(ns*nk); counts.reserve(ns*nk);

  nda::array<ComplexType,2> P0(n_raw, nbnd);
  nda::array<ComplexType,2> S1(n_raw, n_raw);
  long n_aug_max = 0;

  for(long is=0; is<ns; ++is) {
    for(long ik=0; ik<nk; ++ik) {
      auto orig_loc = orig_all(is,ik,all,all);   // (nbnd, ng_loc)
      auto aug_loc  = aug_all(is,ik,all,all);      // (n_raw, ng_loc)

      // projection coefficients P0[j,b] = Σ_G aug(j,G) conj(orig(b,G))
      nda::blas::gemm(aug_loc, nda::dagger(orig_loc), P0);
      k_comm.all_reduce_in_place_n(P0.data(), P0.size(), std::plus<>{});
      // A⊥ = aug - P0·orig   (overwrite aug_loc, local slice)
      nda::blas::gemm(ComplexType(-1.0), P0, orig_loc, ComplexType(1.0), aug_loc);

      // residual overlap S1[i,j] = Σ_G A⊥(i) conj(A⊥(j))
      nda::blas::gemm(aug_loc, nda::dagger(aug_loc), S1);
      k_comm.all_reduce_in_place_n(S1.data(), S1.size(), std::plus<>{});

      // diagonalize the true overlap transpose(S1) on one rank, broadcast result
      nda::array<double,1> v(n_raw);
      v() = 0.0;
      if(k_comm.rank()==0) {
        auto v_d = nda::linalg::detail::_eigen_element_impl(nda::transpose(S1),'V');
        v = nda::to_host(v_d);
      }
      k_comm.broadcast_n(S1.data(), S1.size(), 0);
      k_comm.broadcast_n(v.data(), v.size(), 0);

      // s = sqrt(lambda) is the singular value of the residual block; epstol
      // thresholds s (dimensionless once raw states are dtau_step-scaled)
      long cnt = 0;
      for(long i=0;i<n_raw;++i) if(v(i) >= epstol*epstol) ++cnt;
      n_aug_max = std::max(n_aug_max, cnt);
      counts.push_back(cnt);
      Vs.push_back(S1);
      Es.push_back(v);
    }
  }

  n_aug_max = comm->all_reduce_value(n_aug_max, boost::mpi3::max<>{});
  utils::check(n_aug_max<=n_raw, "orthonormalize_augmentation: n_aug_max>n_raw (logic error).");
  if(n_aug_max == 0 and comm->root())
    app_warning("Basis augmentation: no states above epstol={:.3e}; returning the "
                "un-augmented basis ({} bands).", epstol, nbnd);
  {
    long cmin = counts.empty()? n_aug_max : *std::min_element(counts.begin(),counts.end());
    cmin = comm->all_reduce_value(cmin, boost::mpi3::min<>{});
    app_log(2, "  Basis augmentation: raw states/k = {}, kept per k (n_aug_max) = {}, "
               "min above epstol = {}, epstol (singular value) = {:.3e}", n_raw, n_aug_max, cmin, epstol);
  }

  // full singular-value spectrum, replicated on every rank: each (s,k) block is
  // filled by the root of its k-pool and summed. Es holds eigenvalues lambda in
  // ascending order; store s = sqrt(lambda) in descending order.
  auto sv_all = nda::array<double,3>::zeros({nspin_tot, nkpts_tot, n_raw});
  {
    long isk = 0;
    for(long is=0; is<ns; ++is) {
      for(long ik=0; ik<nk; ++ik, ++isk) {
        if(k_comm.rank() != 0) continue;
        auto const& v = Es[isk];
        for(long i=0; i<n_raw; ++i)
          sv_all(s_range.first()+is, k_range.first()+ik, i) = std::sqrt(std::max(v(n_raw-1-i), 0.0));
      }
    }
    comm->all_reduce_in_place_n(sv_all.data(), sv_all.size(), std::plus<>{});
  }
  {
    auto sv_line = [&](long is, long ik) {
      std::string line;
      for(long i=0; i<n_raw; ++i) {
        if(i == n_aug_max) line += " |";
        line += fmt::format(" {:.3e}", sv_all(is,ik,i));
      }
      return line;
    };
    app_log(2, "  Singular values (descending, '|' marks the kept/discarded cut):");
    app_log(2, "    is=0 ik=0:{}", sv_line(0,0));
    for(long is=0; is<nspin_tot; ++is)
      for(long ik=0; ik<nkpts_tot; ++ik) {
        if(is==0 and ik==0) continue;
        app_log(3, "    is={} ik={}:{}", is, ik, sv_line(is,ik));
      }
  }

  // per-band THC fit weights: 1 for the originals, min(s,1) for the kept
  // augmentation states (band nbnd+r pairs with the r-th largest s at each k)
  auto weights = nda::array<double,3>::zeros({nspin_tot, nkpts_tot, nbnd + n_aug_max});
  weights(nda::range::all, nda::range::all, nda::range(0,nbnd)) = 1.0;
  for(long is=0; is<nspin_tot; ++is)
    for(long ik=0; ik<nkpts_tot; ++ik)
      for(long r=0; r<n_aug_max; ++r)
        weights(is, ik, nbnd + r) = std::min(sv_all(is, ik, n_aug_max-1-r), 1.0);

  // pass 2: assemble [originals | augmentation]
  long nbnd_tot = nbnd + n_aug_max;
  auto psi_out = math::nda::make_distributed_array<larray>(*comm, psi_orig.grid(),
      {nspin_tot, nkpts_tot, nbnd_tot, ngm},
      {psi_orig.block_size()[0], psi_orig.block_size()[1], nbnd_tot, psi_orig.block_size()[3]});
  auto out_all = psi_out.local();

  nda::array<ComplexType,2> R(n_aug_max, n_raw);
  long isk = 0;
  for(long is=0; is<ns; ++is) {
    for(long ik=0; ik<nk; ++ik, ++isk) {
      auto orig_loc = orig_all(is,ik,all,all);
      auto aug_loc  = aug_all(is,ik,all,all);   // residual A⊥ (local slice)
      out_all(is,ik,nda::range(0,nbnd),all) = orig_loc;

      if(n_aug_max == 0) continue;   // no augmentation block: originals only
      auto const& V = Vs[isk];
      auto const& v = Es[isk];
      for(long r=0;r<n_aug_max;++r) {
        long src = n_raw - n_aug_max + r;   // largest eigenvalues are last
        double lam = v(src);
        if(lam < 1e-10 and comm->root())
          app_warning("Basis augmentation: retaining near-null vector (overlap eigenvalue "
                      "{:.3e} < 1e-10). Consider lowering nbnd_aug or raising epstol.", lam);
        double inv = 1.0/std::sqrt(std::max(lam, 1e-300));
        for(long j=0;j<n_raw;++j) R(r,j) = V(src,j)*inv;
      }
      nda::blas::gemm(R, aug_loc, out_all(is,ik,nda::range(nbnd,nbnd_tot),all));
    }
  }
  comm->barrier();
  return std::make_tuple(std::move(psi_out), std::move(weights));
}

template<MEMORY_SPACE MEM, utils::Communicator comm_t>
auto rayleigh_eigvals(mf::MF& mf,
                    darray_t<memory::array<MEM, ComplexType, 4>, comm_t>& psi)
  -> nda::array<double, 3>
{
  // Kinetic-energy diagonal <phi_i|T|phi_i> = sum_G 0.5*|k+G|^2 |phi_i(G)|^2 on
  // the 'w' truncated-G grid, stored as the augmented (non-eigen) basis's
  // eigenvalues. Provides a finite estimate of the single-particle energy scale
  // (e.g. to bound the imaginary-axis frequency grid / wmax). The full
  // <phi|H0|phi> cannot be formed here because the base MF's pseudopotential is
  // sized for the original band count; with h0_source="compute" the SCF still
  // rebuilds the full H0.
  auto comm = psi.communicator();
  long nspin_tot = psi.global_shape()[0];
  long nkpts_tot = psi.global_shape()[1];
  long nbnd_tot  = psi.global_shape()[2];

  auto const& wfc_g = *mf.wfc_truncated_grid();
  long ngm = wfc_g.size();
  nda::array<int,2> miller(ngm,3);
  utils::generate_miller_index(wfc_g.gv_to_fft(), miller, wfc_g.mesh());
  auto recv = mf.recv();          // (3,3) reciprocal vectors, recv(i,:) = i-th (Cartesian)
  auto kpts = mf.kpts_ibz();      // (nkpts_ibz,3) Cartesian; psi is over the IBZ
  utils::check(kpts.extent(0) >= nkpts_tot,
               "rayleigh_eigvals: kpts_ibz has {} < {} k-points.", kpts.extent(0), nkpts_tot);

  long g0 = psi.local_range(3).first();
  long ngloc = psi.local_shape()[3];
  auto s_range = psi.local_range(0);
  auto k_range = psi.local_range(1);
  auto ploc = psi.local();

  auto eig = nda::array<double,3>::zeros({nspin_tot, nkpts_tot, nbnd_tot});
  for(auto [is,s] : itertools::enumerate(s_range)) {
    for(auto [ik,k] : itertools::enumerate(k_range)) {
      for(long ig=0; ig<ngloc; ++ig) {
        long g = g0 + ig;
        double kx = kpts(k,0) + miller(g,0)*recv(0,0) + miller(g,1)*recv(1,0) + miller(g,2)*recv(2,0);
        double ky = kpts(k,1) + miller(g,0)*recv(0,1) + miller(g,1)*recv(1,1) + miller(g,2)*recv(2,1);
        double kz = kpts(k,2) + miller(g,0)*recv(0,2) + miller(g,1)*recv(1,2) + miller(g,2)*recv(2,2);
        double t = 0.5*(kx*kx + ky*ky + kz*kz);
        for(long b=0; b<nbnd_tot; ++b) {
          auto c = ploc(is,ik,b,ig);
          eig(s,k,b) += t*(std::real(c)*std::real(c) + std::imag(c)*std::imag(c));
        }
      }
    }
  }
  comm->all_reduce_in_place_n(eig.data(), eig.size(), std::plus<>{});
  return eig;
}

// Upgrade the augmented-basis eigval seed from the kinetic Rayleigh diagonal to the
// full DFT Kohn-Sham matrix
//   H_KS(i,j) = <phi_i|H0|phi_j> + <phi_i|V_H + V_xc|phi_j>,
// evaluated in the augmented basis. On success `sH_KS` (allocated by the caller with
// shape (nspin, nkpts_ibz, nbnd_aug, nbnd_aug)) holds the hermitized matrix on every
// node and the KS diagonal eps_i = Re[H_KS(i,i)] is returned over the IBZ on all ranks.
// Returns std::nullopt (with a warning, leaving sH_KS undefined) when the DFT V_Hxc is
// unavailable -- pyscf-parented augmentation (no exported DFT local potential) or a
// meta-GGA/hybrid functional (no multiplicative V_xc) -- in which case the caller keeps
// the kinetic seed. Requires a provisional write of the augmented h5 (`fn`) so a
// pseudopot for the augmented band count and the augmented orbitals can be read back;
// V_Hxc = svsc - svloc then follows from that pseudopot. (Hybrid functionals have no
// reliable exported signal yet and are not detected -- a known gap.)
template<class DArr>
std::optional<nda::array<double,3>>
try_ks_eigval_ibz(mf::MF& mf, std::string const& fn, DArr const& psi,
                  nda::array<double,3> const& band_weights,
                  nda::array<double,3> const& kinetic_eig_ibz,
                  std::string const& augment_type,
                  math::shm::shared_array<nda::array_view<ComplexType,4>>& sH_KS)
{
  // Only QE-derived parents export the DFT local potential (svsc/svloc) into the
  // augmented h5 (mirrors the pseudopot write condition in bdft_readonly). Pyscf
  // parents write no Hamiltonian group, so make_pseudopot would fail -- skip here.
  bool pp_available =
      (mf.input_file_type() == mf::xml_input_type and mf.mf_type() == mf::qe_source) or
      (mf.input_file_type() == mf::h5_input_type   and mf.mf_type() != mf::pyscf_source);
  if (not pp_available) {
    app_warning("Basis augmentation: DFT V_Hxc not available for this mean-field parent "
                "(missing scf_local_potential / vxc_with_nlcc); augmented eigval seeded "
                "with the kinetic (Rayleigh) diagonal only. This is expected for pyscf "
                "parents, QE-xml parents, and meta-GGA/hybrid functionals.");
    return std::nullopt;
  }

  // Provisional write: build the augmented h5 with the kinetic seed so its pseudopot
  // (rebuilt for the augmented band count) and orbitals can be read back. The caller
  // then writes the h5 a second time with the KS seed -- this double write is
  // intentional: the pseudopot can only be rebuilt from the augmented h5, and
  // augmentation is a one-time setup step, so the extra write is acceptable.
  // no H_KS yet -- that is what this function is about to build
  auto aug_mf = mf::MF(mf::bdft::bdft_readonly(mf, fn, psi, kinetic_eig_ibz,
                                               augment_type, band_weights, nullptr));
  auto psp = hamilt::make_pseudopot(aug_mf);
  if (not psp->has_local_vhxc()) {
    app_warning("Basis augmentation: DFT V_Hxc not available for this mean-field parent "
                "(missing scf_local_potential / vxc_with_nlcc); augmented eigval seeded "
                "with the kinetic (Rayleigh) diagonal only. This is expected for pyscf "
                "parents, QE-xml parents, and meta-GGA/hybrid functionals.");
    return std::nullopt;
  }

  auto& mpi = *aug_mf.mpi();
  long nspin = aug_mf.nspin();
  long nkpts_ibz = aug_mf.nkpts_ibz();
  long norb = aug_mf.nbnd();
  using array_view_4d_t = nda::array_view<ComplexType,4>;
  utils::check(sH_KS.shape() == std::array<long,4>{nspin, nkpts_ibz, norb, norb},
               "try_ks_eigval_ibz: sH_KS shape ({},{},{},{}) != ({},{},{},{}).",
               sH_KS.shape()[0], sH_KS.shape()[1], sH_KS.shape()[2], sH_KS.shape()[3],
               nspin, nkpts_ibz, norb, norb);

  // H_KS = H0 + V_Hxc in the augmented basis, assembled in place in sH_KS.
  hamilt::set_H0(aug_mf, psp.get(), sH_KS);

  auto dVHxc = hamilt::V_Hxc_aug<HOST_MEMORY>(aug_mf, mpi.comm, psp.get());
  auto sVHxc = math::shm::make_shared_array<array_view_4d_t>(mpi, {nspin, nkpts_ibz, norb, norb});
  math::nda::gather_to_shm(dVHxc, sVHxc);
  mpi.comm.barrier();

  // H <- (H + H^dag)/2 on every node's window: set_H0 is not hermitized and
  // gen_V_Hxc_aug is hermitian only to round-off, while every consumer of the
  // stored matrix (update_MOs' hermitian generalized eigensolver, update_G)
  // reads a single triangle. The diagonal is unchanged in exact arithmetic and
  // in floating point, so the eigval seed below is untouched by this.
  if (mpi.node_comm.root()) {
    auto H = sH_KS.local();
    H += sVHxc.local();
    for (long s = 0; s < nspin; ++s)
      for (long k = 0; k < nkpts_ibz; ++k) {
        for (long i = 0; i < norb; ++i) {
          for (long j = 0; j < i; ++j) {
            auto h = 0.5*(H(s,k,i,j) + std::conj(H(s,k,j,i)));
            H(s,k,i,j) = h;
            H(s,k,j,i) = std::conj(h);
          }
          H(s,k,i,i) = ComplexType(std::real(H(s,k,i,i)), 0.0);
        }
      }
  }
  sH_KS.node_sync();

  auto ks_eig_ibz = nda::array<double,3>::zeros({nspin, nkpts_ibz, norb});
  if (mpi.comm.root()) {
    auto H = sH_KS.local();
    for (long s = 0; s < nspin; ++s)
      for (long k = 0; k < nkpts_ibz; ++k)
        for (long i = 0; i < norb; ++i)
          ks_eig_ibz(s,k,i) = std::real(H(s,k,i,i));
  }
  mpi.comm.broadcast_n(ks_eig_ibz.data(), ks_eig_ibz.size(), 0);
  return ks_eig_ibz;
}

template<MEMORY_SPACE MEM>
mf::MF add_augmentation(mf::MF& mf, std::string fn,
                        std::shared_ptr<orbital_augmenter_t> augmenter,
                        long nbnd_aug, double epstol, double dtau_step)
{
  using larray = typename memory::array<MEM,ComplexType,4>;
  decltype(nda::range::all) all;

  auto& mpi = *mf.mpi();
  long nspin = mf.nspin();
  long nspin_in_basis = mf.nspin_in_basis();
  long nkpts_ibz = mf.nkpts_ibz();
  long nbnd = mf.nbnd();
  long ngm = mf.wfc_truncated_grid()->size();

  utils::check(nspin == nspin_in_basis,
               "add_augmentation: nspin_in_basis != nspin not supported yet.");
  utils::check(dtau_step > 0.0, "add_augmentation: dtau_step must be > 0, got {}.", dtau_step);
  if(nbnd_aug < 0 or nbnd_aug > nbnd) nbnd_aug = nbnd;
  int n_raw_per = augmenter->n_raw_per_band();
  long n_raw = n_raw_per * nbnd_aug;

  app_log(2,"*****************************************************");
  app_log(2,"*               Basis augmentation:                 *");
  app_log(2,"*****************************************************");
  app_log(2,"  - file name of new BDFT system: {}", fn);
  app_log(2,"  - augmentation type           : {}", augmenter->type());
  app_log(2,"  - original bands (kept)       : {}", nbnd);
  app_log(2,"  - bands transformed (nbnd_aug): {}", nbnd_aug);
  app_log(2,"  - raw states per k            : {} ({} channels x {})", n_raw, n_raw_per, nbnd_aug);
  app_log(2,"  - dtau_step (bohr)            : {:.4e}", dtau_step);
  app_log(2,"  - singular-value cutoff       : {:.3e} (dimensionless, thresholds s)", epstol);

  /* 4d processor grid: {s, k, bnd, g}. Bands are not distributed. */
  std::array<long,4> pgrid;
  {
    long sz = mpi.comm.size();
    long pk = utils::find_proc_grid_max_npools(sz,nkpts_ibz,1.0);
    pgrid = {1,pk,1,sz/pk};
  }
  long gblk = std::max(1L, (ngm + pgrid[3] - 1)/pgrid[3]);
  std::array<long,4> bs_orig = {1,1,nbnd,gblk};
  std::array<long,4> bs_base = {1,1,nbnd_aug,gblk};
  std::array<long,4> bs_raw  = {1,1,n_raw,gblk};

  // originals (kept, orthonormal)
  auto psi_orig = mf::read_distributed_orbital_set<larray>(mf,mpi.comm,'w',pgrid,
                    {0,nspin},{0,nkpts_ibz},{0,nbnd},bs_orig);

  if(nbnd_aug == 0) {
    // No augmentation states: emit the original orbitals in the augmented bdft
    // format (DFT KS eigval seed + pseudopot, h0_source="compute") so a
    // zero-augmentation baseline runs through the same downstream pipeline. The
    // originals are true DFT eigenstates, so their KS diagonal reproduces the parent
    // eigenvalues; when V_Hxc is unavailable the kinetic diagonal is used instead.
    auto eig_ibz = rayleigh_eigvals<MEM>(mf, psi_orig);
    auto sH_KS = math::shm::make_shared_array<nda::array_view<ComplexType,4>>(
        mpi, {nspin, nkpts_ibz, nbnd, nbnd});
    bool have_hks = false;
    if(auto ks = try_ks_eigval_ibz(mf, fn, psi_orig, nda::array<double,3>{}, eig_ibz,
                                   augmenter->type(), sH_KS)) {
      eig_ibz = std::move(*ks);
      have_hks = true;
    }
    app_log(2,"  - KS seed                     : {}",
            have_hks ? "ks_matrix (full H_KS stored)" : "kinetic (Rayleigh diagonal)");
    auto H_KS = sH_KS.local();
    return mf::MF(mf::bdft::bdft_readonly(mf, fn, psi_orig, eig_ibz, augmenter->type(),
                                          nda::array<double,3>{},
                                          have_hks ? &H_KS : nullptr));
  }

  // base block used to generate the raw augmentation states
  auto psi_base = mf::read_distributed_orbital_set<larray>(mf,mpi.comm,'w',pgrid,
                    {0,nspin},{0,nkpts_ibz},{0,nbnd_aug},bs_base);
  utils::check(psi_orig.local_range(3) == psi_base.local_range(3), "add_augmentation: G-range mismatch.");

  // raw (non-orthonormal) augmentation states, same grid/G-distribution
  auto raw_aug = math::nda::make_distributed_array<larray>(mpi.comm,pgrid,
                    {nspin,nkpts_ibz,n_raw,ngm},bs_raw);
  {
    long g0 = psi_base.local_range(3).first();
    auto base_loc = psi_base.local();
    auto raw_loc  = raw_aug.local();
    auto s_range  = psi_base.local_range(0);
    auto k_range  = psi_base.local_range(1);
    for(long is=0; is<s_range.size(); ++is)
      for(long ik=0; ik<k_range.size(); ++ik)
        augmenter->generate_raw(int(s_range.first()+is), int(k_range.first()+ik), g0,
                                base_loc(is,ik,all,all), raw_loc(is,ik,all,all));
    // raw states are dpsi/dtau (1/bohr); dtau_step (bohr) makes them
    // dimensionless displacement responses, setting the scale of the
    // singular values used for truncation and THC fit weights
    raw_aug.local() *= ComplexType(dtau_step);
  }

  // project against originals, SVD-truncate, uniformize -> [orig | aug]
  auto [psi_full, band_weights] = orthonormalize_augmentation<MEM>(psi_orig, raw_aug, epstol);

  // DFT Kohn-Sham eigval seed for the (non-eigen) augmented basis, falling back to the
  // kinetic Rayleigh diagonal when the DFT V_Hxc is unavailable.
  auto eig_ibz = rayleigh_eigvals<MEM>(mf, psi_full);
  long norb = psi_full.global_shape()[2];
  auto sH_KS = math::shm::make_shared_array<nda::array_view<ComplexType,4>>(
      mpi, {nspin, nkpts_ibz, norb, norb});
  bool have_hks = false;
  if(auto ks = try_ks_eigval_ibz(mf, fn, psi_full, band_weights, eig_ibz,
                                 augmenter->type(), sH_KS)) {
    eig_ibz = std::move(*ks);
    have_hks = true;
  }
  app_log(2,"  - KS seed                     : {}",
          have_hks ? "ks_matrix (full H_KS stored)" : "kinetic (Rayleigh diagonal)");
  auto H_KS = sH_KS.local();

  return mf::MF(mf::bdft::bdft_readonly(mf, fn, psi_full, eig_ibz, augmenter->type(),
                                        band_weights, have_hks ? &H_KS : nullptr));
}

namespace {

// Screened e-ph vertex + eigenvalues + q-vector for one phonon iq, read from
// {elph_dir}/elph_bare.iq{iq}.h5. Everything is converted Ry→Ha on read. g_scr
// (the large array) is held in a node-shared window — the global root reads and
// decodes it, then it is broadcast once per node — so only one copy exists per
// node rather than one per rank (the vertex is identical across ranks). The
// small et_k/et_kq/xq are read on the root and broadcast to every rank.
struct elph_bare_data_t {
  math::shm::shared_array<nda::array_view<ComplexType,4>> g_scr;  // (nk,nmode,m,n) Ha
  nda::array<double,2>       et_k;   // (nk, N) Ha, e(n,k)
  nda::array<double,2>       et_kq;  // (nk, N) Ha, e(m,k+q)
  nda::stack_array<double,3> xq;     // phonon wavevector, crystal coords
};

elph_bare_data_t read_elph_bare(
    utils::mpi_context_t<boost::mpi3::communicator,boost::mpi3::shared_communicator>& mpi_ctx,
    std::string const& elph_dir, long iq)
{
  constexpr double Ry2Ha = 0.5;
  std::string path = elph_dir + "/elph_bare.iq" + std::to_string(iq) + ".h5";

  // Dimensions (nk, nmode, N) from attributes: read on the global root and
  // broadcast, so only one rank opens the file (avoids an N-rank open on
  // shared filesystems), consistent with the et/xq broadcasts below.
  int nksqtot = 0, nmodes = 0, nbnd = 0;
  if(mpi_ctx.comm.root()) {
    h5::file f(path, 'r');
    h5::h5_read_attribute(f, "nksqtot", nksqtot);
    h5::h5_read_attribute(f, "nmodes",  nmodes);
    h5::h5_read_attribute(f, "nbnd",    nbnd);
  }
  mpi_ctx.comm.broadcast_n(&nksqtot, 1, 0);
  mpi_ctx.comm.broadcast_n(&nmodes,  1, 0);
  mpi_ctx.comm.broadcast_n(&nbnd,    1, 0);
  long nk = nksqtot, nm = nmodes, Nb = nbnd;

  auto g_scr = math::shm::make_shared_array<nda::array_view<ComplexType,4>>(
      mpi_ctx, {nk, nm, Nb, Nb});
  nda::array<double,2> et_k, et_kq;
  nda::stack_array<double,3> xq;

  // Global root reads + decodes g_scr into its node's window and the small
  // et/xq arrays; node roots then broadcast g_scr across nodes.
  if(mpi_ctx.comm.root()) {
    h5::file f(path, 'r');
    // QE stores g_scr as (nk, nmode, n, 2*m): 3rd axis = KET band n (at k),
    // interleaved last axis = BRA band m (at k+q). Swap the band axes on read
    // (as qe_data.py does) so g_scr(ik,mode,m,n) = ⟨ψ_{m,k+q}|dV_scr|ψ_{n,k}⟩.
    nda::array<double,4> g_raw;
    nda::h5_read(f, "g_scr", g_raw);
    utils::check(g_raw.shape(0)==nk and g_raw.shape(1)==nm and g_raw.shape(2)==Nb
                 and g_raw.shape(3)==2*Nb,
                 "read_elph_bare: g_scr shape mismatch with attributes.");
    auto g = g_scr.local();
    for(long ik=0; ik<nk; ++ik)
      for(long im=0; im<nm; ++im)
        for(long m=0; m<Nb; ++m)      // bra band (k+q)
          for(long n=0; n<Nb; ++n)    // ket band (k)
            g(ik,im,m,n) = Ry2Ha * ComplexType{g_raw(ik,im,n,2*m),
                                                g_raw(ik,im,n,2*m+1)};

    nda::h5_read(f, "et_k",  et_k);   et_k()  *= Ry2Ha;
    nda::h5_read(f, "et_kq", et_kq);  et_kq() *= Ry2Ha;
    nda::array<double,1> xqr;
    nda::h5_read(f, "xq_cryst", xqr);
    utils::check(xqr.shape(0) == 3, "read_elph_bare: xq_cryst size {} != 3.", xqr.shape(0));
    xq(0)=xqr(0); xq(1)=xqr(1); xq(2)=xqr(2);
  } else {
    et_k  = nda::array<double,2>(nk, Nb);
    et_kq = nda::array<double,2>(nk, Nb);
  }

  // g_scr: one broadcast per node (node roots), then intra-node ranks share it.
  if(g_scr.node_comm()->root())
    g_scr.internode_comm()->broadcast_n(g_scr.local().data(), g_scr.size(), 0);
  // et/xq are small: broadcast to every rank.
  mpi_ctx.comm.broadcast_n(et_k.data(),  et_k.size(),  0);
  mpi_ctx.comm.broadcast_n(et_kq.data(), et_kq.size(), 0);
  mpi_ctx.comm.broadcast_n(xq.data(),    3,            0);
  mpi_ctx.comm.barrier();

  return elph_bare_data_t{std::move(g_scr), std::move(et_k), std::move(et_kq), xq};
}

} // anonymous namespace

template<MEMORY_SPACE MEM>
mf::MF add_augmentation_dpsi(mf::MF& mf, std::string fn,
                             std::string deltapsi_dir, std::string elph_dir,
                             std::vector<long> const& iq_list, long nmodes_in,
                             std::vector<long> const& mode_list,
                             long nbnd_aug, long nbnd_mf,
                             double smearing, double epstol, double dtau_step)
{
  using larray = typename memory::array<MEM,ComplexType,4>;
  decltype(nda::range::all) all;

  auto& mpi = *mf.mpi();
  long nspin = mf.nspin();
  long nspin_in_basis = mf.nspin_in_basis();
  long nkpts_ibz = mf.nkpts_ibz();
  long nkpts = mf.nkpts();
  long N = mf.nbnd();          // nscf/h5 band count (buffer upper limit)
  // M = number of original bands kept in the augmented MF. Default (<=0 or >N)
  // keeps all N -> buffer range [M,N) is empty (dedicated-bundle behavior).
  long M = (nbnd_mf <= 0 or nbnd_mf > N) ? N : nbnd_mf;
  long ngm = mf.wfc_truncated_grid()->size();
  long natom = mf.number_of_atoms();
  long nmodes = (nmodes_in <= 0) ? 3*natom : nmodes_in;
  long nq = long(iq_list.size());

  // modes contributing raw states (1-based); empty selection = all nmodes
  std::vector<long> modes(mode_list);
  if(modes.empty()) {
    modes.resize(size_t(nmodes));
    for(long m=0; m<nmodes; ++m) modes[size_t(m)] = m+1;
  }
  for(auto m : modes)
    utils::check(m >= 1 and m <= nmodes,
                 "add_augmentation_dpsi: mode index {} outside [1, {}].", m, nmodes);
  long nmodes_sel = long(modes.size());

  utils::check(nspin == nspin_in_basis,
               "add_augmentation_dpsi: nspin_in_basis != nspin not supported yet.");
  // The elph eigenvalues/vertex read from elph_bare.iq*.h5 have no spin axis, so
  // the buffer-band correction would use the wrong dE/g_scr for a minority spin.
  utils::check(nspin == 1,
               "add_augmentation_dpsi: spin-polarized base MF (nspin==2) not "
               "supported; the elph data is spin-agnostic.");
  utils::check(mf.npol() == 1,
               "add_augmentation_dpsi: only npol==1 is supported.");
  utils::check(nkpts == nkpts_ibz,
               "add_augmentation_dpsi: requires a full-BZ k-grid "
               "(nkpts == nkpts_ibz); disable k-point symmetry.");
  utils::check(nq > 0, "add_augmentation_dpsi: empty iq_list.");
  utils::check(dtau_step > 0.0, "add_augmentation_dpsi: dtau_step must be > 0, got {}.", dtau_step);

  // R = number of δψ bands used. -1 → all bands present in the first file;
  // 0 → no δψ states (baseline: originals in augmented bdft format, no file read).
  long R = nbnd_aug;
  if(nbnd_aug != 0) {
    std::string dprefix = deltapsi_dir + "/deltapsi_iq" + std::to_string(iq_list[0]) + "_mode1";
    auto dk0 = methods::detail::read_Deltapsi_k(dprefix, 0, mf.wfc_truncated_grid()->mesh());
    if(R < 0 or R > dk0.nbnd) R = dk0.nbnd;
    utils::check(R <= dk0.nbnd,
                 "add_augmentation_dpsi: nbnd_aug (R={}) exceeds bands in δψ file ({}).",
                 R, dk0.nbnd);
  }
  long n_raw_per = nmodes_sel * nq;
  long n_raw = R * n_raw_per;

  app_log(2,"*****************************************************");
  app_log(2,"*            Basis augmentation (dpsi):             *");
  app_log(2,"*****************************************************");
  app_log(2,"  - file name of new BDFT system: {}", fn);
  app_log(2,"  - δψ directory                : {}", deltapsi_dir);
  app_log(2,"  - elph directory              : {}", elph_dir);
  app_log(2,"  - phonon iq list (size {})     : {}", nq, [&]{
            std::string s; for(auto q : iq_list) s += std::to_string(q)+" "; return s; }());
  app_log(2,"  - h5/nscf bands (N)           : {}", N);
  app_log(2,"  - original bands kept (M)     : {}", M);
  app_log(2,"  - δψ bands used (R)           : {}", R);
  app_log(2,"  - modes per q                 : {} of {} ({})", nmodes_sel, nmodes, [&]{
            std::string s; for(auto m : modes) s += std::to_string(m)+" "; return s; }());
  app_log(2,"  - buffer bands m ∈ [{}, {})   : {} band(s)", M, N, std::max(0L, N-M));
  app_log(2,"  - buffer smearing (Ha)        : {:.4e}", smearing);
  app_log(2,"  - raw states per k            : {} ({} q·modes x {})", n_raw, n_raw_per, R);
  app_log(2,"  - dtau_step (bohr)            : {:.4e}", dtau_step);
  app_log(2,"  - singular-value cutoff       : {:.3e} (dimensionless, thresholds s)", epstol);

  /* 4d processor grid: {s, k, bnd, g}. Bands are not distributed. */
  std::array<long,4> pgrid;
  {
    long sz = mpi.comm.size();
    long pk = utils::find_proc_grid_max_npools(sz,nkpts_ibz,1.0);
    pgrid = {1,pk,1,sz/pk};
  }
  long gblk = std::max(1L, (ngm + pgrid[3] - 1)/pgrid[3]);
  std::array<long,4> bs_all  = {1,1,N,gblk};
  std::array<long,4> bs_orig = {1,1,M,gblk};
  std::array<long,4> bs_raw  = {1,1,n_raw,gblk};

  // Read all N h5/nscf bands: [0,M) are the kept originals, [M,N) supply the
  // buffer ψ(m,k+q). (Bands are not distributed, so band-axis slicing is local.)
  auto psi_all = mf::read_distributed_orbital_set<larray>(mf,mpi.comm,'w',pgrid,
                    {0,nspin},{0,nkpts_ibz},{0,N},bs_all);

  // psi_orig = first M bands (the kept originals) as its own distributed array.
  auto psi_orig = math::nda::make_distributed_array<larray>(mpi.comm,pgrid,
                    {nspin,nkpts_ibz,M,ngm},bs_orig);
  psi_orig.local() = psi_all.local()(all,all,nda::range(0,M),all);

  if(R == 0) {
    // No δψ states: emit the M kept originals in the augmented bdft format
    // (DFT KS eigval seed + pseudopot, h0_source="compute") so a zero-augmentation
    // baseline runs through the same downstream pipeline. The originals are true DFT
    // eigenstates, so their KS diagonal reproduces the parent eigenvalues; when V_Hxc
    // is unavailable the kinetic diagonal is used instead. Mirrors the momentum
    // add_augmentation(nbnd_aug==0) baseline.
    auto eig_ibz = rayleigh_eigvals<MEM>(mf, psi_orig);
    auto sH_KS = math::shm::make_shared_array<nda::array_view<ComplexType,4>>(
        mpi, {nspin, nkpts_ibz, M, M});
    bool have_hks = false;
    if(auto ks = try_ks_eigval_ibz(mf, fn, psi_orig, nda::array<double,3>{}, eig_ibz,
                                   std::string("dpsi"), sH_KS)) {
      eig_ibz = std::move(*ks);
      have_hks = true;
    }
    app_log(2,"  - KS seed                     : {}",
            have_hks ? "ks_matrix (full H_KS stored)" : "kinetic (Rayleigh diagonal)");
    auto H_KS = sH_KS.local();
    return mf::MF(mf::bdft::bdft_readonly(mf, fn, psi_orig, eig_ibz, std::string("dpsi"),
                                          nda::array<double,3>{},
                                          have_hks ? &H_KS : nullptr));
  }

  // 'w' grid Miller indices + FFT-linear → row lookup (for regridding δψ)
  auto const& wfc_g = *mf.wfc_truncated_grid();
  auto mesh = wfc_g.mesh();                       // stack_array<long,3>
  std::unordered_map<long,long> fft2row;
  fft2row.reserve(size_t(ngm)*2);
  for(long r=0; r<ngm; ++r) fft2row[wfc_g.gv_to_fft()(r)] = r;

  auto kpts_cr = mf.kpts_crystal();              // (nkpts, 3) crystal

  // raw (non-orthonormal) augmentation states on the 'w' grid, zeroed
  auto raw_aug = math::nda::make_distributed_array<larray>(mpi.comm,pgrid,
                    {nspin,nkpts_ibz,n_raw,ngm},bs_raw);
  raw_aug.local() = ComplexType(0.0);

  utils::check(psi_all.local_range(1) == raw_aug.local_range(1),
               "add_augmentation_dpsi: k-range mismatch.");
  utils::check(psi_all.local_range(3) == raw_aug.local_range(3),
               "add_augmentation_dpsi: G-range mismatch.");

  auto raw_loc  = raw_aug.local();
  auto all_loc  = psi_all.local();               // N bands: originals [0,M) + buffer [M,N)
  auto s_range  = raw_aug.local_range(0);
  auto k_range  = raw_aug.local_range(1);         // owned deposit indices j
  long g0       = raw_aug.local_range(3).first();
  long ng_loc   = raw_aug.local_range(3).size();

  for(long iq_idx=0; iq_idx<nq; ++iq_idx) {
    long iq = iq_list[iq_idx];
    auto ed = read_elph_bare(mpi, elph_dir, iq);
    auto g_loc = ed.g_scr.local();   // node-shared g_scr(ik,mode,m,n), Ha
    utils::check(g_loc.shape()[0] == nkpts,
                 "add_augmentation_dpsi: elph nk {} != nkpts {}.", g_loc.shape()[0], nkpts);
    utils::check(g_loc.shape()[1] >= nmodes,
                 "add_augmentation_dpsi: elph nmode {} < nmodes {}.", g_loc.shape()[1], nmodes);
    utils::check(g_loc.shape()[2] >= N and ed.et_k.shape(1) >= N and ed.et_kq.shape(1) >= N,
                 "add_augmentation_dpsi: elph band count {} < N {}.", g_loc.shape()[2], N);

    nda::array<double,1> qv(3); qv(0)=ed.xq(0); qv(1)=ed.xq(1); qv(2)=ed.xq(2);
    nda::array<int,1> kpq_map(nkpts);
    utils::calculate_kpq_map(kpts_cr, qv, kpq_map);
    // invert: deposit index j → source k (kpq_map(ik)=j)
    nda::array<int,1> inv_map(nkpts); inv_map() = -1;
    for(long ik=0; ik<nkpts; ++ik) inv_map(kpq_map(ik)) = int(ik);

    for(long jl=0; jl<k_range.size(); ++jl) {
      long j  = k_range.first()+jl;               // deposit index (global)
      long ik = inv_map(j);                        // source k
      utils::check(ik >= 0, "add_augmentation_dpsi: no source k for deposit index {}.", j);
      // umklapp G0 = round(k(ik)+q - k(j)); δψ Miller h → h+G0 in the k(j) frame
      long G0[3];
      for(int d=0; d<3; ++d)
        G0[d] = std::lround(kpts_cr(ik,d)+qv(d)-kpts_cr(j,d));

      for(long im=0; im<nmodes_sel; ++im) {
        long mode = modes[size_t(im)] - 1;           // 0-based mode index
        std::string dprefix = deltapsi_dir + "/deltapsi_iq" + std::to_string(iq)
                            + "_mode" + std::to_string(mode+1);
        auto dk = methods::detail::read_Deltapsi_k(dprefix, ik, mesh);
        utils::check(dk.evc_raw.shape(0) >= R,
                     "add_augmentation_dpsi: δψ file (iq{},mode{},ik{}) has {} bands < R {}.",
                     iq, mode+1, ik+1, dk.evc_raw.shape(0), R);
        long npw = dk.npw;

        // target local row on the 'w' grid for each δψ G (or -1 if off-slice)
        std::vector<long> localrow(size_t(npw), -1);
        for(long ig=0; ig<npw; ++ig) {
          long n1=dk.miller(ig,0)+G0[0], n2=dk.miller(ig,1)+G0[1], n3=dk.miller(ig,2)+G0[2];
          long m1=n1%mesh(0); if(m1<0)m1+=mesh(0);
          long m2=n2%mesh(1); if(m2<0)m2+=mesh(1);
          long m3=n3%mesh(2); if(m3<0)m3+=mesh(2);
          long lin=(m1*mesh(1)+m2)*mesh(2)+m3;
          auto it=fft2row.find(lin);
          if(it!=fft2row.end()) {
            long r=it->second;
            if(r>=g0 and r<g0+ng_loc) localrow[size_t(ig)] = r-g0;
          }
        }

        for(long is_l=0; is_l<s_range.size(); ++is_l) {
          long is = s_range.first()+is_l;
          (void) is;
          for(long n=0; n<R; ++n) {
            long row = (iq_idx*nmodes_sel + im)*R + n; // band index in raw_aug
            auto out = raw_loc(is_l, jl, row, all);    // (ng_loc)
            // (1) scatter δψ(n,k) coefficients onto the local 'w' grid slice
            for(long ig=0; ig<npw; ++ig) {
              long lr = localrow[size_t(ig)];
              if(lr >= 0)
                out(lr) += ComplexType{dk.evc_raw(n, 2*ig), dk.evc_raw(n, 2*ig+1)};
            }
            // (2) buffer bands m ∈ [M,N): += ψ(m,k+q) g_scr(mode,m,n) reg(e(n,k)-e(m,k+q)).
            // These are the bands above the M kept originals (and up to the nscf
            // count N), which QE's δψ (⊥ all N bands) excluded — adding them back
            // makes the state the response orthogonal to only the M kept bands.
            // reg is a sharp, continuous 1/x cutoff: 1/x for |x|>σ, x/σ² for |x|≤σ.
            for(long m=M; m<N; ++m) {
              double dE  = ed.et_k(ik,n) - ed.et_kq(ik,m);
              double reg = (std::abs(dE) > smearing) ? (1.0/dE) : (dE/(smearing*smearing));
              ComplexType coef = g_loc(ik,mode,m,n) * reg;
              out += coef * all_loc(is_l, jl, m, all);
            }
          }
        }
      }
    }
  }

  // raw states are deltapsi/dtau (1/bohr); dtau_step (bohr) makes them
  // dimensionless displacement responses, setting the scale of the singular
  // values used for truncation and THC fit weights
  raw_aug.local() *= ComplexType(dtau_step);

  // project against originals, SVD-truncate, uniformize -> [orig | aug]
  auto [psi_full, band_weights] = orthonormalize_augmentation<MEM>(psi_orig, raw_aug, epstol);

  // DFT Kohn-Sham eigval seed for the (non-eigen) augmented basis, falling back to the
  // kinetic Rayleigh diagonal when the DFT V_Hxc is unavailable.
  auto eig_ibz  = rayleigh_eigvals<MEM>(mf, psi_full);
  long norb = psi_full.global_shape()[2];
  auto sH_KS = math::shm::make_shared_array<nda::array_view<ComplexType,4>>(
      mpi, {nspin, nkpts_ibz, norb, norb});
  bool have_hks = false;
  if(auto ks = try_ks_eigval_ibz(mf, fn, psi_full, band_weights, eig_ibz,
                                 std::string("dpsi"), sH_KS)) {
    eig_ibz = std::move(*ks);
    have_hks = true;
  }
  app_log(2,"  - KS seed                     : {}",
          have_hks ? "ks_matrix (full H_KS stored)" : "kinetic (Rayleigh diagonal)");
  auto H_KS = sH_KS.local();

  return mf::MF(mf::bdft::bdft_readonly(mf, fn, psi_full, eig_ibz, std::string("dpsi"),
                                        band_weights, have_hks ? &H_KS : nullptr));
}

// explicit template instantiations
template mf::MF add_augmentation<HOST_MEMORY>(mf::MF&,std::string,std::shared_ptr<orbital_augmenter_t>,long,double,double);
template mf::MF add_augmentation_dpsi<HOST_MEMORY>(mf::MF&,std::string,std::string,std::string,std::vector<long> const&,long,std::vector<long> const&,long,long,double,double,double);
template nda::array<double, 3> rayleigh_eigvals<HOST_MEMORY,communicator>(mf::MF&, darray_t<host_array<ComplexType, 4>, communicator>&);
template std::tuple<darray_t<host_array<ComplexType, 4>, communicator>, nda::array<double, 3>>
orthonormalize_augmentation(darray_t<host_array<ComplexType, 4>, communicator>&,
                            darray_t<host_array<ComplexType, 4>, communicator>&, double);

} // orbitals
