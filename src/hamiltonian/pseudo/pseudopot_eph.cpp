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

#include <fstream>
#include <filesystem>
#include <cmath>
#include <vector>

#include "configuration.hpp"
#include "IO/app_loggers.h"
#include "arch/arch.h"
#include "utilities/check.hpp"
#include "utilities/fortran_utilities.h"
#include "utilities/kpoint_utils.hpp"
#include "utilities/lr_utils.hpp"
#include "mpi3/environment.hpp"
#include "mpi3/communicator.hpp"
#include "h5/h5.hpp"
#include "nda/nda.hpp"
#include "nda/h5.hpp"
#include "nda/tensor.hpp"
#include "numerics/distributed_array/nda.hpp"
#include "numerics/shared_array/nda.hpp"
#include "utilities/qe_utilities.hpp"
#include "utilities/mpi_context.h"
#include "mean_field/MF.hpp"
#include "mean_field/distributed_orbital_readers.hpp"
#include "hamiltonian/add_vloc.hpp"
#include "hamiltonian/v_h.hpp"
#include "hamiltonian/pseudo/pseudopot.h"
#include "numerics/nda_functions.hpp"
#include "numerics/fft/nda.hpp"
#include "nda/blas.hpp"
#include "itertools/itertools.hpp"

namespace hamilt
{

// The six symmetric Cartesian pairs for the second-derivative overlaps P2,
// packed as p = 0..5 -> (i,j) = (x,x),(y,y),(z,z),(x,y),(x,z),(y,z).
namespace {
  constexpr int d2_pair_i[6] = {0, 1, 2, 0, 0, 1};
  constexpr int d2_pair_j[6] = {0, 1, 2, 1, 2, 2};
}

template<typename MF_t>
void pseudopot::eph_projector_overlaps(MF_t &mf, nda::array<ComplexType,5>& P,
                                       nda::array<ComplexType,5>* P2)
{
  using nda::range;
  decltype(range::all) all;

  utils::check(ptype == pp_ncpp_t,
               "pseudopot::eph_projector_overlaps: only NCPP is supported.");
  utils::check(input_file_type == mf::h5_input_type,
               "pseudopot::eph_projector_overlaps: only h5 input is supported "
               "(input_file_type must be h5).");
  utils::check(std::filesystem::exists(input_file_name),
               "pseudopot::eph_projector_overlaps: missing file: {}", input_file_name);

  long nk    = nkpts_ibz;
  long nbnd  = mf.nbnd();
  long nkb   = n_proj();

  // Output: P(d, s, k, mu*npol, a);  d=0 -> <beta|phi>, d=1..3 -> <beta|(k+G)_{x,y,z} phi>.
  P = nda::array<ComplexType,5>(4, nspin, nk, nkb*npol, nbnd);
  P() = ComplexType(0.0);
  if( P2 ) {
    // P2(p, s, k, mu*npol, a) -> <beta|(k+G)_i(k+G)_j phi>, p = 0..5 symmetric pair.
    *P2 = nda::array<ComplexType,5>(6, nspin, nk, nkb*npol, nbnd);
    (*P2)() = ComplexType(0.0);
  }
  if( nkb == 0 ) return;

  // 'w' truncated grid + (k+G) factors (Cartesian, 1/bohr)
  auto wfc_g = mf.wfc_truncated_grid();
  long ngm = wfc_g->size();
  nda::array<int,2> mill_w(ngm,3);
  utils::generate_miller_index(wfc_g->gv_to_fft(), mill_w, wfc_g->mesh());
  auto recv_ = mf.recv();
  auto kcart = mf.kpts_ibz();

  // open the pseudopotential group written next to the mean field
  h5::file file(input_file_name, 'r');
  h5::group grp0(file);
  h5::group grp1 = grp0.open_group("Hamiltonian");
  std::string type("");
  h5::h5_read_attribute(grp1, "pp_type", type);
  h5::group grp = grp1.open_group(type);

  int npwx;
  h5::h5_read_attribute(grp,"max_npw",npwx);
  nda::array<int,1> npw(nk);
  nda::h5_read(grp,"npw",npw);

  // k2g: projector miller (per k) -> 'w' truncated grid index
  sarray_t<nda::array_view<long,2>> sk2g(*mpi,{nk,npwx});
  auto k2g = sk2g.local();
  build_projector_k2g(mf, grp, npw(), k2g);

  // orbitals (band/kpt distributed, full G locally -- as in read_vnl_h5)
  using local_4Array_t = typename nda::array<ComplexType,4>;
  auto dPsia = mf::read_distributed_orbital_set_ibz<local_4Array_t>(mf,mpi->comm,'w');
  auto psi = dPsia.local();
  utils::check(psi.extent(3) == npol*ngm, "eph_projector_overlaps: shape mismatch.");
  auto bloc = dPsia.local_range(2);   // local band range
  long nb_loc = bloc.size();

  nda::array<ComplexType,2> buff(1,npwx);
  nda::array<ComplexType,1> vkb(ngm);
  // (k+G)_d weighted copies of the local orbitals, d = x,y,z
  nda::array<ComplexType,3> psi_w(3, nb_loc, npol*ngm);
  // (k+G)_i(k+G)_j weighted copies (six symmetric pairs), only if P2 requested
  nda::array<ComplexType,3> psi_w2;
  if( P2 ) psi_w2 = nda::array<ComplexType,3>(6, nb_loc, npol*ngm);

  for( auto [is,s] : itertools::enumerate(dPsia.local_range(0)) ) {
    for( auto [ik,k] : itertools::enumerate(dPsia.local_range(1)) ) {
      // build (k+G)_d * psi (and (k+G)_i(k+G)_j * psi if P2) for this k
      for( int ip=0; ip<npol; ++ip )
        for( long g=0; g<ngm; ++g ) {
          double kpg[3];
          for( int d=0; d<3; ++d )
            kpg[d] = kcart(k,d)
                   + double(mill_w(g,0))*recv_(0,d)
                   + double(mill_w(g,1))*recv_(1,d)
                   + double(mill_w(g,2))*recv_(2,d);
          for( long ia=0; ia<nb_loc; ++ia ) {
            ComplexType v = psi(is,ik,ia,ip*ngm+g);
            for( int d=0; d<3; ++d ) psi_w(d,ia,ip*ngm+g) = kpg[d]*v;
            if( P2 )
              for( int p=0; p<6; ++p )
                psi_w2(p,ia,ip*ngm+g) = kpg[d2_pair_i[p]]*kpg[d2_pair_j[p]]*v;
          }
        }
      for( int ib=0; ib<nkb; ++ib ) {
        read_projector_vkb(grp, k, ib, npw(k), k2g(k,all), buff, vkb);
        for( int ip=0; ip<npol; ++ip ) {
          // NOTE: P(0) = <beta|phi> duplicates the overlap read_vnl_h5 already
          // computes into the member Pskna. It is recomputed here because for
          // some mf implementations the overlaps may not be available.
          nda::blas::gemv(ComplexType(1.0), psi(is,ik,all,range(ip*ngm,(ip+1)*ngm)), vkb,
                          ComplexType(0.0), P(0,s,k,ib*npol+ip,bloc));
          for( int d=0; d<3; ++d )
            nda::blas::gemv(ComplexType(1.0), psi_w(d,all,range(ip*ngm,(ip+1)*ngm)), vkb,
                            ComplexType(0.0), P(d+1,s,k,ib*npol+ip,bloc));
          if( P2 )
            for( int p=0; p<6; ++p )
              nda::blas::gemv(ComplexType(1.0), psi_w2(p,all,range(ip*ngm,(ip+1)*ngm)), vkb,
                              ComplexType(0.0), (*P2)(p,s,k,ib*npol+ip,bloc));
        }
      }
    }
  }
  mpi->comm.all_reduce_in_place_n(P.data(), P.size(), std::plus<>{});
  if( P2 ) mpi->comm.all_reduce_in_place_n(P2->data(), P2->size(), std::plus<>{});
}

template<typename MF_t>
auto pseudopot::eph_vertex_nonlocal(MF_t &mf, nda::array_const_view<double,1> q_cryst)
  -> nda::array<ComplexType,5>
{
  using nda::range;
  decltype(range::all) all;

  utils::check(npol == 1, "pseudopot::eph_vertex_nonlocal: only npol=1 supported.");
  // The Bmat/g_nl assembly below uses only spin-0 overlaps and replicates over
  // spin; that is valid only for nspin==1. For collinear spin (nspin==2) the
  // orbitals differ per spin, so the nonlocal vertex would be wrong for the
  // minority spin. Guard until per-spin assembly is implemented.
  utils::check(nspin == 1, "pseudopot::eph_vertex_nonlocal: only nspin=1 supported.");
  utils::check(q_cryst.shape(0) == 3, "eph_vertex_nonlocal: q_cryst must have length 3.");
  utils::check(mf.nkpts() == nkpts_ibz,
               "eph_vertex_nonlocal: requires a full-BZ k-grid (nkpts == nkpts_ibz).");

  long nk     = nkpts_ibz;
  long nb     = mf.nbnd();
  long nproj  = n_proj();
  long nat    = ityp.size();
  long nmodes = 3*nat;

  // projector overlaps: P(d, s, k, mu, a); P(0)=<beta|phi>, P(1..3)=<beta|(k+G)_d phi>
  nda::array<ComplexType,5> P;
  eph_projector_overlaps(mf, P);

  // k+q map (crystal coordinates)
  auto kpts_crys = mf.kpts_crystal();
  nda::array<double,1> q(3); q(0)=q_cryst(0); q(1)=q_cryst(1); q(2)=q_cryst(2);
  nda::array<int,1> kpq_map(nk);
  utils::calculate_kpq_map(kpts_crys, q, kpq_map);

  // projector -> (atom, projector-within-atom) maps
  nda::array<long,1> proj_atom(nproj), proj_ih(nproj);
  for(long ia=0; ia<nat; ++ia)
    for(int j=0; j<nh(ityp(ia)); ++j) { proj_atom(ofs(ia)+j)=ia; proj_ih(ofs(ia)+j)=j; }

  auto D = Dion();   // (nsp, nhm, nhm), Hartree
  // The block-diagonal factorization below uses only the diagonal D_ii. This is
  // exact for ONCV pseudos (diagonal dion) but drops off-diagonal D_ij (same-l
  // multi-projector NCPP). Abort if dion has significant off-diagonal weight.
  {
    long nsp = D.shape(0), nhm = D.shape(1);
    double maxdiag = 0.0, maxoff = 0.0;
    for(long it=0; it<nsp; ++it)
      for(long i=0; i<nhm; ++i)
        for(long j=0; j<nhm; ++j) {
          double a = std::abs(D(it,i,j));
          if(i==j) maxdiag = std::max(maxdiag, a);
          else     maxoff  = std::max(maxoff, a);
        }
    utils::check(maxoff <= 1e-6*(maxdiag+1e-30),
                 "pseudopot::eph_vertex_nonlocal: dion has significant off-diagonal "
                 "terms (max_offdiag={:.3e}, max_diag={:.3e}); only diagonal D "
                 "(e.g. ONCV) is supported.", maxoff, maxdiag);
  }
  long M = 6*nproj;
  double s2 = std::sqrt(2.0);

  // Bmat(k, m, mu') = (<beta_ip|phi_m,k> +/- i<beta_ip|(k+G)_d phi_m,k>)/sqrt(2),
  //   mu' = ip + nproj*idir (+ nproj*3 for the '-' block); QE alphap_QE = i*CoQui.
  nda::array<ComplexType,3> Bmat(nk, nb, M);
  for(long k=0; k<nk; ++k)
    for(long m=0; m<nb; ++m)
      for(long ip=0; ip<nproj; ++ip) {
        ComplexType b = P(0,0,k,ip,m);
        for(int idir=0; idir<3; ++idir) {
          ComplexType db = ComplexType(0.0,1.0)*P(1+idir,0,k,ip,m);
          Bmat(k,m, ip+nproj*idir)     = (b + db)/s2;
          Bmat(k,m, ip+nproj*(idir+3)) = (b - db)/s2;
        }
      }

  // D_thc(mu', mode): block-diagonal D-matrix, +Dii for the '+' block, -Dii for '-'
  nda::array<ComplexType,2> Dthc(M, nmodes); Dthc() = ComplexType(0.0);
  for(long ip=0; ip<nproj; ++ip) {
    ComplexType Dii = D(ityp(proj_atom(ip)), proj_ih(ip), proj_ih(ip));
    for(int idir=0; idir<3; ++idir) {
      long imode = proj_atom(ip)*3 + idir;
      Dthc(ip+nproj*idir,     imode) =  Dii;
      Dthc(ip+nproj*(idir+3), imode) = -Dii;
    }
  }

  // g(s,mode,k)[m,n] = sum_mu' conj(Bmat(kq,m,mu')) D_thc(mu',mode) Bmat(k,n,mu')
  // Memory: the full (nspin,nmodes,nk,nb,nb) vertex is large, so each rank builds
  // only its contiguous k-block into a small local buffer; the blocks are then
  // gathered to the root (Bmat, built from the replicated projector overlaps, is
  // small and kept whole so kq outside the local block is still available).
  // Returned full on the root and empty on every other rank.
  int rank = mpi->comm.rank(), np = mpi->comm.size();
  auto [k0, k1] = itertools::chunk_range(0L, nk, long(np), long(rank));
  long nk_loc = k1 - k0;
  nda::array<ComplexType,5> gloc(nspin, nmodes, nk_loc, nb, nb); gloc() = ComplexType(0.0);
  nda::array<ComplexType,2> W(nb, M);
  for(long k=k0; k<k1; ++k) {
    long kl = k - k0;
    long kq = kpq_map(k);
    for(long imode=0; imode<nmodes; ++imode) {
      for(long m=0; m<nb; ++m)
        for(long mu=0; mu<M; ++mu)
          W(m,mu) = std::conj(Bmat(kq,m,mu)) * Dthc(mu,imode);
      nda::blas::gemm(ComplexType(1.0), W, nda::transpose(Bmat(k,all,all)),
                      ComplexType(0.0), gloc(0,imode,kl,all,all));
    }
  }
  for(int is=1; is<nspin; ++is) gloc(is,all,all,all,all) = gloc(0,all,all,all,all);

  nda::array<ComplexType,5> g;
  if(mpi->comm.root()) {
    g = nda::array<ComplexType,5>(nspin, nmodes, nk, nb, nb); g() = ComplexType(0.0);
    if(nk_loc > 0) g(all,all,range(k0,k1),all,all) = gloc;
    for(int p=1; p<np; ++p) {
      auto [pk0, pk1] = itertools::chunk_range(0L, nk, long(np), long(p));
      long pnk = pk1 - pk0;
      if(pnk == 0) continue;
      nda::array<ComplexType,5> tmp(nspin, nmodes, pnk, nb, nb);
      mpi->comm.receive_n(tmp.data(), tmp.size(), p, 0);
      g(all,all,range(pk0,pk1),all,all) = tmp;
    }
  } else if(nk_loc > 0) {
    mpi->comm.send_n(gloc.data(), gloc.size(), 0, 0);
  }
  return g;
}

template<typename MF_t>
auto pseudopot::eph_vertex_nonlocal_d2(MF_t &mf)
  -> nda::array<ComplexType,7>
{
  using nda::range;
  decltype(range::all) all;

  utils::check(npol == 1, "pseudopot::eph_vertex_nonlocal_d2: only npol=1 supported.");
  utils::check(nspin == 1, "pseudopot::eph_vertex_nonlocal_d2: only nspin=1 supported.");
  utils::check(mf.nkpts() == nkpts_ibz,
               "eph_vertex_nonlocal_d2: requires a full-BZ k-grid (nkpts == nkpts_ibz).");

  long nk     = nkpts_ibz;
  long nb     = mf.nbnd();
  long nproj  = n_proj();
  long nat    = ityp.size();
  long nmodes = 3*nat;

  // projector overlaps at q=0: P(0)=<beta|phi>, P(1..3)=<beta|(k+G)_d phi>,
  // P2(p)=<beta|(k+G)_i(k+G)_j phi> for the 6 symmetric Cartesian pairs.
  nda::array<ComplexType,5> P, P2;
  eph_projector_overlaps(mf, P, &P2);

  // per-projector atom index and diagonal KB strength D_mu (Hartree, ONCV)
  auto D = Dion();   // (nsp, nhm, nhm)
  {
    long nsp = D.shape(0), nhm = D.shape(1);
    double maxdiag = 0.0, maxoff = 0.0;
    for(long it=0; it<nsp; ++it)
      for(long i=0; i<nhm; ++i)
        for(long j=0; j<nhm; ++j) {
          double a = std::abs(D(it,i,j));
          if(i==j) maxdiag = std::max(maxdiag, a);
          else     maxoff  = std::max(maxoff, a);
        }
    utils::check(maxoff <= 1e-6*(maxdiag+1e-30),
                 "pseudopot::eph_vertex_nonlocal_d2: dion has significant off-diagonal "
                 "terms (max_offdiag={:.3e}, max_diag={:.3e}); only diagonal D "
                 "(e.g. ONCV) is supported.", maxoff, maxdiag);
  }

  // maps 3x3 direction pair (i,j) -> symmetric pair index p (inverse of d2_pair_*)
  auto pair_of = [](int i, int j) {
    for(int p=0; p<6; ++p)
      if((d2_pair_i[p]==i && d2_pair_j[p]==j) || (d2_pair_i[p]==j && d2_pair_j[p]==i))
        return p;
    return 0;  // unreachable
  };

  // d^2 V_bare / dtau_kappa dtau_kappa' vanishes for kappa != kappa' (each ionic
  // term depends on a single atom), so g2 is block-diagonal in the atom index.
  // Store it compactly as (nspin, nat, 3, 3, nk, nb, nb) — dims (atom, cart_i,
  // cart_j) — with mode1 = 3*ka+i, mode2 = 3*ka+j, instead of the dense
  // (nspin, nmodes, nmodes, ...) which is nat times larger and mostly zero.
  // Memory: each rank builds only its contiguous k-block, gathered to the root;
  // returned full on the root and empty on every other rank.
  int rank = mpi->comm.rank(), np = mpi->comm.size();
  auto [k0, k1] = itertools::chunk_range(0L, nk, long(np), long(rank));
  long nk_loc = k1 - k0;
  nda::array<ComplexType,7> g2loc(nspin, nat, 3, 3, nk_loc, nb, nb);
  g2loc() = ComplexType(0.0);

  // D-scaled ket copies for atom kappa (rows mu in [ofs, ofs+nh)): D_mu * P
  nda::array<ComplexType,2> DP0, DP1[3], DP2[6];
  for(int d=0; d<3; ++d) DP1[d] = nda::array<ComplexType,2>();
  for(int p=0; p<6; ++p) DP2[p] = nda::array<ComplexType,2>();

  for(long k=k0; k<k1; ++k) {
    long kl = k - k0;
    for(long ka=0; ka<nat; ++ka) {
      long off = ofs(ka), nmu = nh(ityp(ka));
      if(nmu == 0) continue;
      auto mu_rng = range(off, off+nmu);

      // scale kets by D_mu (row-wise)
      DP0 = nda::array<ComplexType,2>(nmu, nb);
      for(int d=0; d<3; ++d) DP1[d] = nda::array<ComplexType,2>(nmu, nb);
      for(int p=0; p<6; ++p) DP2[p] = nda::array<ComplexType,2>(nmu, nb);
      for(long j=0; j<nmu; ++j) {
        ComplexType Dj = D(ityp(ka), j, j);
        for(long m=0; m<nb; ++m) {
          DP0(j,m) = Dj * P(0,0,k,off+j,m);
          for(int d=0; d<3; ++d) DP1[d](j,m) = Dj * P(d+1,0,k,off+j,m);
          for(int p=0; p<6; ++p) DP2[p](j,m) = Dj * P2(p,0,k,off+j,m);
        }
      }

      // bra blocks (unscaled), (nmu, nb) views
      auto P0b = P(0,0,k,mu_rng,all);

      for(int i=0; i<3; ++i) {
        for(int j=0; j<3; ++j) {
          int  pij   = pair_of(i,j);
          auto gout  = g2loc(0,ka,i,j,kl,all,all);
          auto P2b   = P2(pij,0,k,mu_rng,all);
          auto P1i   = P(i+1,0,k,mu_rng,all);
          auto P1j   = P(j+1,0,k,mu_rng,all);
          // g2(m,n) = sum_mu D_mu [ -conj(P2_ij)_m P0_n - conj(P0)_m P2_ij_n
          //                         + conj(P1_i)_m P1_j_n + conj(P1_j)_m P1_i_n ]
          nda::blas::gemm(ComplexType(-1.0), nda::dagger(P2b), DP0,      ComplexType(0.0), gout);
          nda::blas::gemm(ComplexType(-1.0), nda::dagger(P0b), DP2[pij], ComplexType(1.0), gout);
          nda::blas::gemm(ComplexType( 1.0), nda::dagger(P1i), DP1[j],   ComplexType(1.0), gout);
          nda::blas::gemm(ComplexType( 1.0), nda::dagger(P1j), DP1[i],   ComplexType(1.0), gout);
        }
      }
    }
  }
  for(int is=1; is<nspin; ++is)
    g2loc(is,all,all,all,all,all,all) = g2loc(0,all,all,all,all,all,all);

  nda::array<ComplexType,7> g2;
  if(mpi->comm.root()) {
    g2 = nda::array<ComplexType,7>(nspin, nat, 3, 3, nk, nb, nb);
    g2() = ComplexType(0.0);
    if(nk_loc > 0) g2(all,all,all,all,range(k0,k1),all,all) = g2loc;
    for(int p=1; p<np; ++p) {
      auto [pk0, pk1] = itertools::chunk_range(0L, nk, long(np), long(p));
      long pnk = pk1 - pk0;
      if(pnk == 0) continue;
      nda::array<ComplexType,7> tmp(nspin, nat, 3, 3, pnk, nb, nb);
      mpi->comm.receive_n(tmp.data(), tmp.size(), p, 0);
      g2(all,all,all,all,range(pk0,pk1),all,all) = tmp;
    }
  } else if(nk_loc > 0) {
    mpi->comm.send_n(g2loc.data(), g2loc.size(), 0, 0);
  }
  return g2;
}

namespace {
  // Per-species radial data for the erf-compensated Simpson FT of vloc.
  struct vloc_radial_t {
    nda::array<double,1> r, wr, aux1;  // r; simpson_weight*rab; r*vloc + zp*erf(r)
    double zp = 0.0;
  };

  // Build the FT integrand pieces from the radial arrays (r, rab, vloc in Ha, zp).
  vloc_radial_t make_vloc_radial(nda::array_const_view<double,1> r,
                                 nda::array_const_view<double,1> rab,
                                 nda::array_const_view<double,1> vloc, double zp) {
    long len = r.size();
    long n = len - (1 - (len % 2));              // Simpson needs an odd point count
    vloc_radial_t v; v.zp = zp;
    v.r    = nda::array<double,1>(n);
    v.aux1 = nda::array<double,1>(n);
    v.wr   = nda::array<double,1>(n);
    for(long i=0;i<n;++i) {
      v.r(i)    = r(i);
      v.aux1(i) = r(i)*vloc(i) + zp*std::erf(r(i));   // cancels the -zp/r tail (Ha)
    }
    for(long i=0;i<n;++i) {
      double w = (i==0 or i==n-1) ? (1.0/3.0) : ((i%2==1) ? (4.0/3.0) : (2.0/3.0));
      v.wr(i) = w*rab(i);
    }
    return v;
  }

  // vloc(|G|) in Ha (QE vloc_of_g / setlocq); G in 1/bohr; G=0 -> 0 (the derivative
  // vertex has no G=0 term). Omega is the cell volume (bohr^3).
  double vloc_of_g(vloc_radial_t const& v, double G, double Omega) {
    if(G < 1e-10) return 0.0;
    double s = 0.0;
    long n = v.r.size();
    for(long i=0;i<n;++i) s += v.wr(i)*v.aux1(i)*std::sin(G*v.r(i));
    s /= G;
    s -= v.zp*std::exp(-G*G/4.0)/(G*G);
    return (4.0*M_PI/Omega)*s;
  }

  // System geometry + per-species radial local pseudopotential read from the MF
  // h5, shared by build_dvloc_ion (first derivative) and build_d2vloc_ion
  // (second derivative) so the radial FT input is single-sourced.
  struct vloc_geom_t {
    nda::array<double,2> recvv, tau;  // rows b_i (1/bohr); (nat,3) cart (bohr)
    nda::array<int,1>    aid;         // species per atom (0-based)
    double Omega = 0.0;              // cell volume (bohr^3)
    long   nat = 0, nsp = 0;
    std::vector<vloc_radial_t> vsp;   // per-species radial FT integrand
  };

  vloc_geom_t load_vloc_geom(std::string const& input_file_name) {
    vloc_geom_t G;
    h5::file file(input_file_name, 'r');
    h5::group grp0(file);
    h5::group sys = grp0.open_group("System");
    nda::array<double,2> latt;
    nda::h5_read(sys, "lattice_vectors",    latt);   // rows a_i (bohr)
    nda::h5_read(sys, "reciprocal_vectors", G.recvv);
    nda::h5_read(sys, "atomic_positions",   G.tau);
    nda::h5_read(sys, "atomic_id",          G.aid);
    G.nat = G.tau.shape(0);

    // normalize atomic_id to 0-based (mirror read_vnl_h5; robust to 1-based files)
    int id_min = *std::min_element(G.aid.begin(), G.aid.end());
    utils::check(id_min==0 or id_min==1,
                 "load_vloc_geom: invalid atomic_id array (min id:{}).", id_min);
    G.aid() -= id_min;

    G.Omega = std::abs(
        latt(0,0)*(latt(1,1)*latt(2,2)-latt(1,2)*latt(2,1))
      - latt(0,1)*(latt(1,0)*latt(2,2)-latt(1,2)*latt(2,0))
      + latt(0,2)*(latt(1,0)*latt(2,1)-latt(1,1)*latt(2,0)));

    h5::group ham = grp0.open_group("Hamiltonian");
    std::string type(""); h5::h5_read_attribute(ham, "pp_type", type);
    h5::group ncpp = ham.open_group(type);
    utils::check(ncpp.has_subgroup("vloc_radial") and ncpp.has_dataset("z_valence"),
                 "load_vloc_geom: the mean-field h5 '{}' is missing the "
                 "'vloc_radial' group / 'z_valence' dataset needed for the e-ph "
                 "local vertex. Regenerate the mean field with the updated pw2coqui "
                 "(which writes the per-species radial local pseudopotential).",
                 input_file_name);
    h5::group vrg = ncpp.open_group("vloc_radial");
    nda::array<double,1> zval;
    nda::h5_read(ncpp, "z_valence", zval);
    for(long a=0;a<G.nat;++a) G.nsp = std::max(G.nsp, long(G.aid(a))+1);
    G.vsp.resize(G.nsp);
    for(long it=0; it<G.nsp; ++it) {
      utils::check(vrg.has_subgroup("sp"+std::to_string(it)),
                   "load_vloc_geom: missing vloc_radial/sp{} in {}.", it, input_file_name);
      h5::group sp = vrg.open_group("sp"+std::to_string(it));
      nda::array<double,1> r, rab, vloc;
      nda::h5_read(sp, "r",    r);
      nda::h5_read(sp, "rab",  rab);
      nda::h5_read(sp, "vloc", vloc);
      vloc *= 0.5;                                   // Ry -> Ha
      G.vsp[it] = make_vloc_radial(r, rab, vloc, zval(it));
    }
    return G;
  }
} // anonymous namespace

template<typename MF_t>
auto pseudopot::build_dvloc_ion(MF_t &mf, nda::array_const_view<double,1> q_cryst)
  -> nda::array<ComplexType,2>
{
  using nda::range;
  decltype(range::all) all;

  utils::check(ptype == pp_ncpp_t, "pseudopot::build_dvloc_ion: only NCPP is supported.");
  utils::check(q_cryst.shape(0) == 3, "build_dvloc_ion: q_cryst must have length 3.");

  // dense FFT grid eph_vertex_local works on
  long NX = mf.fft_grid_dim(0), NY = mf.fft_grid_dim(1), NZ = mf.fft_grid_dim(2);
  long nnr = NX*NY*NZ;

  // System geometry + per-species radial vloc, read from the MF h5
  auto geom = load_vloc_geom(input_file_name);
  auto const& recvv = geom.recvv;
  auto const& tau   = geom.tau;
  auto const& aid   = geom.aid;
  auto const& vsp   = geom.vsp;
  double Omega = geom.Omega;
  long nat = geom.nat, nsp = geom.nsp;
  long nmodes = 3*nat;

  // q in Cartesian (1/bohr): q_cart = sum_i q_cryst(i) b_i
  double qc[3];
  for(int d=0; d<3; ++d)
    qc[d] = q_cryst(0)*recvv(0,d) + q_cryst(1)*recvv(1,d) + q_cryst(2)*recvv(2,d);

  // dV(G) = -i (q+G)_d vloc_sp(|q+G|) e^{-i(q+G).tau_ka}  on the dense grid.
  // Create the FFT plan BEFORE filling dvG: the default FFT_MEASURE plan runs
  // trial transforms that clobber the buffer during planning (cf. read_vnl_h5).
  nda::array<ComplexType,4> dvG(nmodes, NX, NY, NZ);
  math::nda::fft<true> F(dvG);
  dvG() = ComplexType(0.0);
  std::vector<double> vGsp(nsp);
  for(long ix=0; ix<NX; ++ix) {
    long m0 = (ix < (NX+1)/2) ? ix : ix-NX;
    for(long iy=0; iy<NY; ++iy) {
      long m1 = (iy < (NY+1)/2) ? iy : iy-NY;
      for(long iz=0; iz<NZ; ++iz) {
        long m2 = (iz < (NZ+1)/2) ? iz : iz-NZ;
        double qG[3];
        for(int d=0; d<3; ++d)
          qG[d] = double(m0)*recvv(0,d) + double(m1)*recvv(1,d)
                + double(m2)*recvv(2,d) + qc[d];
        double qGm = std::sqrt(qG[0]*qG[0]+qG[1]*qG[1]+qG[2]*qG[2]);
        for(long it=0; it<nsp; ++it) vGsp[it] = vloc_of_g(vsp[it], qGm, Omega);
        for(long ka=0; ka<nat; ++ka) {
          double ph = -(qG[0]*tau(ka,0)+qG[1]*tau(ka,1)+qG[2]*tau(ka,2));
          ComplexType sf(std::cos(ph), std::sin(ph));
          ComplexType vsf = vGsp[aid(ka)] * sf;
          for(int d=0; d<3; ++d)
            dvG(3*ka+d, ix, iy, iz) = ComplexType(0.0,-1.0)*qG[d]*vsf;
        }
      }
    }
  }

  // G -> r (unnormalized inverse FFT == numpy ifftn * nnr), batched over modes
  F.backward(dvG);

  // flatten to (nmodes, nnr)
  nda::array<ComplexType,2> dv(nmodes, nnr);
  auto dvG2 = nda::reshape(dvG, std::array<long,2>{nmodes, nnr});
  for(long m=0;m<nmodes;++m)
    for(long r=0;r<nnr;++r) dv(m,r) = dvG2(m,r);
  return dv;
}

template<typename MF_t>
auto pseudopot::build_d2vloc_ion(MF_t &mf)
  -> nda::array<ComplexType,2>
{
  utils::check(ptype == pp_ncpp_t, "pseudopot::build_d2vloc_ion: only NCPP is supported.");

  // dense FFT grid eph_vertex_local works on
  long NX = mf.fft_grid_dim(0), NY = mf.fft_grid_dim(1), NZ = mf.fft_grid_dim(2);
  long nnr = NX*NY*NZ;

  // System geometry + per-species radial vloc, read from the MF h5
  auto geom = load_vloc_geom(input_file_name);
  auto const& recvv = geom.recvv;
  auto const& tau   = geom.tau;
  auto const& aid   = geom.aid;
  auto const& vsp   = geom.vsp;
  double Omega = geom.Omega;
  long nat = geom.nat, nsp = geom.nsp;
  long npair = 6*nat;   // 6 symmetric Cartesian pairs per atom

  // d2V(G) = -G_i G_j vloc_sp(|G|) e^{-iG.tau_ka}  on the dense grid (q=0).
  // Create the FFT plan BEFORE filling d2vG (cf. build_dvloc_ion).
  nda::array<ComplexType,4> d2vG(npair, NX, NY, NZ);
  math::nda::fft<true> F(d2vG);
  d2vG() = ComplexType(0.0);
  std::vector<double> vGsp(nsp);
  for(long ix=0; ix<NX; ++ix) {
    long m0 = (ix < (NX+1)/2) ? ix : ix-NX;
    for(long iy=0; iy<NY; ++iy) {
      long m1 = (iy < (NY+1)/2) ? iy : iy-NY;
      for(long iz=0; iz<NZ; ++iz) {
        long m2 = (iz < (NZ+1)/2) ? iz : iz-NZ;
        double Gc[3];
        for(int d=0; d<3; ++d)
          Gc[d] = double(m0)*recvv(0,d) + double(m1)*recvv(1,d) + double(m2)*recvv(2,d);
        double Gm = std::sqrt(Gc[0]*Gc[0]+Gc[1]*Gc[1]+Gc[2]*Gc[2]);
        for(long it=0; it<nsp; ++it) vGsp[it] = vloc_of_g(vsp[it], Gm, Omega);
        for(long ka=0; ka<nat; ++ka) {
          double ph = -(Gc[0]*tau(ka,0)+Gc[1]*tau(ka,1)+Gc[2]*tau(ka,2));
          ComplexType sf(std::cos(ph), std::sin(ph));
          ComplexType vsf = vGsp[aid(ka)] * sf;
          for(int p=0; p<6; ++p)
            d2vG(6*ka+p, ix, iy, iz) = -Gc[d2_pair_i[p]]*Gc[d2_pair_j[p]]*vsf;
        }
      }
    }
  }

  // G -> r (unnormalized inverse FFT == numpy ifftn * nnr), batched over fields
  F.backward(d2vG);

  // flatten to (6*nat, nnr)
  nda::array<ComplexType,2> d2v(npair, nnr);
  auto d2vG2 = nda::reshape(d2vG, std::array<long,2>{npair, nnr});
  for(long m=0;m<npair;++m)
    for(long r=0;r<nnr;++r) d2v(m,r) = d2vG2(m,r);
  return d2v;
}

// explicit template instantiations
template void pseudopot::eph_projector_overlaps(mf::MF &, nda::array<ComplexType,5>&,
                                                nda::array<ComplexType,5>*);
template auto pseudopot::eph_vertex_nonlocal(mf::MF &, nda::array_const_view<double,1>)
    -> nda::array<ComplexType,5>;
template auto pseudopot::eph_vertex_nonlocal_d2(mf::MF &)
    -> nda::array<ComplexType,7>;
template auto pseudopot::build_dvloc_ion(mf::MF &, nda::array_const_view<double,1>)
    -> nda::array<ComplexType,2>;
template auto pseudopot::build_d2vloc_ion(mf::MF &)
    -> nda::array<ComplexType,2>;

} // namespace hamilt
