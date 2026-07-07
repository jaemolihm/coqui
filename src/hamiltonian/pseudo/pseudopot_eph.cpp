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

template<typename MF_t>
void pseudopot::eph_projector_overlaps(MF_t &mf, nda::array<ComplexType,5>& P)
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

  for( auto [is,s] : itertools::enumerate(dPsia.local_range(0)) ) {
    for( auto [ik,k] : itertools::enumerate(dPsia.local_range(1)) ) {
      // build (k+G)_d * psi for this k
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
        }
      }
    }
  }
  mpi->comm.all_reduce_in_place_n(P.data(), P.size(), std::plus<>{});
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
  nda::array<ComplexType,5> g(nspin, nmodes, nk, nb, nb); g() = ComplexType(0.0);
  nda::array<ComplexType,2> W(nb, M);
  for(long k=0; k<nk; ++k) {
    long kq = kpq_map(k);
    for(long imode=0; imode<nmodes; ++imode) {
      for(long m=0; m<nb; ++m)
        for(long mu=0; mu<M; ++mu)
          W(m,mu) = std::conj(Bmat(kq,m,mu)) * Dthc(mu,imode);
      nda::blas::gemm(ComplexType(1.0), W, nda::transpose(Bmat(k,all,all)),
                      ComplexType(0.0), g(0,imode,k,all,all));
    }
  }
  for(int is=1; is<nspin; ++is) g(is,all,all,all,all) = g(0,all,all,all,all);
  return g;
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

  // System geometry + per-species radial vloc, read from the MF h5 (as numpy did)
  h5::file file(input_file_name, 'r');
  h5::group grp0(file);
  h5::group sys = grp0.open_group("System");
  nda::array<double,2> latt, recvv, tau;
  nda::array<int,1> aid;
  nda::h5_read(sys, "lattice_vectors",    latt);   // rows a_i (bohr)
  nda::h5_read(sys, "reciprocal_vectors", recvv);  // rows b_i (1/bohr)
  nda::h5_read(sys, "atomic_positions",   tau);    // (nat,3) cart (bohr)
  nda::h5_read(sys, "atomic_id",          aid);    // species per atom
  long nat = tau.shape(0);
  long nmodes = 3*nat;

  // normalize atomic_id to 0-based (mirror read_vnl_h5; robust to 1-based files)
  {
    int id_min = *std::min_element(aid.begin(), aid.end());
    utils::check(id_min==0 or id_min==1,
                 "build_dvloc_ion: invalid atomic_id array (min id:{}).", id_min);
    aid() -= id_min;
  }

  double Omega = std::abs(
      latt(0,0)*(latt(1,1)*latt(2,2)-latt(1,2)*latt(2,1))
    - latt(0,1)*(latt(1,0)*latt(2,2)-latt(1,2)*latt(2,0))
    + latt(0,2)*(latt(1,0)*latt(2,1)-latt(1,1)*latt(2,0)));

  h5::group ham = grp0.open_group("Hamiltonian");
  std::string type(""); h5::h5_read_attribute(ham, "pp_type", type);
  h5::group ncpp = ham.open_group(type);
  utils::check(ncpp.has_subgroup("vloc_radial") and ncpp.has_dataset("z_valence"),
               "build_dvloc_ion: the mean-field h5 '{}' is missing the "
               "'vloc_radial' group / 'z_valence' dataset needed for the e-ph "
               "local vertex. Regenerate the mean field with the updated pw2coqui "
               "(which writes the per-species radial local pseudopotential).",
               input_file_name);
  h5::group vrg = ncpp.open_group("vloc_radial");
  nda::array<double,1> zval;
  nda::h5_read(ncpp, "z_valence", zval);           // Hamiltonian/{type}/z_valence
  long nsp = 0; for(long a=0;a<nat;++a) nsp = std::max(nsp, long(aid(a))+1);
  std::vector<vloc_radial_t> vsp(nsp);
  for(long it=0; it<nsp; ++it) {
    utils::check(vrg.has_subgroup("sp"+std::to_string(it)),
                 "build_dvloc_ion: missing vloc_radial/sp{} in {}.", it, input_file_name);
    h5::group sp = vrg.open_group("sp"+std::to_string(it));
    nda::array<double,1> r, rab, vloc;
    nda::h5_read(sp, "r",    r);
    nda::h5_read(sp, "rab",  rab);
    nda::h5_read(sp, "vloc", vloc);
    vloc *= 0.5;                                   // Ry -> Ha
    vsp[it] = make_vloc_radial(r, rab, vloc, zval(it));
  }

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

// explicit template instantiations
template void pseudopot::eph_projector_overlaps(mf::MF &, nda::array<ComplexType,5>&);
template auto pseudopot::eph_vertex_nonlocal(mf::MF &, nda::array_const_view<double,1>)
    -> nda::array<ComplexType,5>;
template auto pseudopot::build_dvloc_ion(mf::MF &, nda::array_const_view<double,1>)
    -> nda::array<ComplexType,2>;

} // namespace hamilt
