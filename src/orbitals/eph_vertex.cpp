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

#include "configuration.hpp"
#include "utilities/check.hpp"
#include "utilities/lr_utils.hpp"
#include "itertools/itertools.hpp"
#include "nda/nda.hpp"
#include "nda/blas.hpp"

#include "orbitals/eph_vertex.h"

namespace orbitals
{

// Design note: hamilt::add_vloc (src/hamiltonian/add_vloc.hpp) also multiplies
// real-space orbitals by a real-space potential, but it cannot be reused here:
// it produces the operator image (V·psi) back in G-space and is diagonal in k
// (psi and hpsi share the same k-point). The local e-ph vertex instead needs the
// band-band matrix element <phi_{m,k+q}|dV|phi_{n,k}>, which couples k+q to k with
// the umklapp e^{iG0·r} phase — neither of which add_vloc expresses.
template<MEMORY_SPACE MEM>
auto eph_vertex_local(mf::MF& mf,
                      nda::array_const_view<ComplexType,2> dV,
                      nda::array_const_view<double,1> q_cryst)
  -> nda::array<ComplexType,5>
{
  using nda::range;
  decltype(range::all) all;

  auto& comm = mf.mpi()->comm;
  int nspin = mf.nspin();
  int nk    = mf.nkpts();
  int nbnd  = mf.nbnd();
  int NX = mf.fft_grid_dim(0), NY = mf.fft_grid_dim(1), NZ = mf.fft_grid_dim(2);
  long nnr = long(NX)*NY*NZ;
  int nmodes = dV.shape(0);
  utils::check(dV.shape(1) == nnr,
               "eph_vertex_local: dV grid mismatch (nnr={}, dV.shape(1)={}).",
               nnr, dV.shape(1));
  utils::check(q_cryst.shape(0) == 3, "eph_vertex_local: q_cryst must have length 3.");

  auto kpts_crys = mf.kpts_crystal();
  nda::array<int,1> kpq_map(nk);
  utils::calculate_kpq_map(kpts_crys, q_cryst, kpq_map);

  // Memory: the full vertex (nspin,nmodes,nk,nbnd,nbnd) is large. Rather than
  // building it on every rank (all_reduce) — which replicates it np times and
  // OOMs at high ranks-per-node — each rank computes only its contiguous k-block
  // into a small local buffer, then blocks are gathered to the root. The result
  // is returned full on the root and empty on every other rank (only the root
  // consumes it downstream).
  int rank = comm.rank(), np = comm.size();
  auto [k0, k1] = itertools::chunk_range(0, nk, np, rank);
  int nk_loc = k1 - k0;

  auto gloc = nda::array<ComplexType,5>::zeros(
      std::array<long,5>{nspin, nmodes, nk_loc, nbnd, nbnd});

  nda::array<ComplexType,2> psik(nbnd,nnr), psikq(nbnd,nnr), psikq_conj(nbnd,nnr), B(nbnd,nnr);
  nda::array<ComplexType,1> phase(nnr);

  for(int is=0; is<nspin; ++is) {
    for(int ik=k0; ik<k1; ++ik) {
      int ikl = ik - k0;
      int ikpq = kpq_map(ik);
      // umklapp reciprocal-lattice vector: k+q = kpq + G0 (integer crystal coords)
      int G0[3];
      for(int d=0; d<3; ++d)
        G0[d] = int(std::lround(kpts_crys(ik,d) + q_cryst(d) - kpts_crys(ikpq,d)));
      // phase(r) = exp(2πi G0·frac),  frac = (ix/NX, iy/NY, iz/NZ)
      for(int ix=0; ix<NX; ++ix)
        for(int iy=0; iy<NY; ++iy)
          for(int iz=0; iz<NZ; ++iz) {
            double ph = 2.0*M_PI*( double(G0[0])*ix/NX
                                 + double(G0[1])*iy/NY
                                 + double(G0[2])*iz/NZ );
            phase((long(ix)*NY+iy)*NZ+iz) = ComplexType(std::cos(ph), std::sin(ph));
          }
      mf.get_orbital_set('r', is, ik,   range(0,nbnd), psik);
      mf.get_orbital_set('r', is, ikpq, range(0,nbnd), psikq);
      psikq_conj = nda::conj(psikq);
      for(int mode=0; mode<nmodes; ++mode) {
        for(long r=0; r<nnr; ++r) {
          ComplexType w = phase(r)*dV(mode,r);
          for(int n=0; n<nbnd; ++n) B(n,r) = w*psik(n,r);
        }
        // g(is,mode,ik,m,n) = (1/nnr) Σ_r conj(u_{m,kpq}(r)) e^{iG0r} dV(r) u_{n,k}(r)
        nda::blas::gemm(ComplexType(1.0/double(nnr)), psikq_conj, nda::transpose(B),
                        ComplexType(0.0), gloc(is,mode,ikl,all,all));
      }
    }
  }

  // Gather the per-rank k-blocks onto the root. Returned full on root, empty
  // elsewhere.
  nda::array<ComplexType,5> g;
  if(comm.root()) {
    g = nda::array<ComplexType,5>::zeros(
        std::array<long,5>{nspin, nmodes, nk, nbnd, nbnd});
    if(nk_loc > 0) g(all,all,range(k0,k1),all,all) = gloc;
    for(int p=1; p<np; ++p) {
      auto [pk0, pk1] = itertools::chunk_range(0, nk, np, p);
      int pnk = pk1 - pk0;
      if(pnk == 0) continue;
      nda::array<ComplexType,5> tmp(nspin, nmodes, pnk, nbnd, nbnd);
      comm.receive_n(tmp.data(), tmp.size(), p, 0);
      g(all,all,range(pk0,pk1),all,all) = tmp;
    }
  } else if(nk_loc > 0) {
    comm.send_n(gloc.data(), gloc.size(), 0, 0);
  }
  return g;
}

template auto eph_vertex_local<HOST_MEMORY>(
    mf::MF&, nda::array_const_view<ComplexType,2>, nda::array_const_view<double,1>)
  -> nda::array<ComplexType,5>;

} // namespace orbitals
