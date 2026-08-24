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

#include <cmath>

#include "configuration.hpp"
#include "utilities/check.hpp"
#include "utilities/kpoint_utils.hpp"
#include "utilities/lr_utils.hpp"
#include "itertools/itertools.hpp"
#include "nda/nda.hpp"
#include "nda/blas.hpp"

#include "orbitals/pw_matrix_elements.h"

namespace orbitals
{

// Design note: two routines in-tree already do parts of this, and neither can be
// reused. methods::cholesky::evaluate_pair_densities builds the same orbital
// product, but it and cholesky::rho_g are private members of cholesky; it writes
// into a distributed dArray_t rather than returning a local array; it pairs
// (ka, kb) through qk_to_k2 on the Qpts list rather than through kpq_map at an
// arbitrary q; and it folds in sqrt(v(q+G)) — singular at q+G = 0, the column
// this routine exists for — together with a 1/sqrt(Omega*nkpts) that is not
// wanted here. eph_vertex_local does the same k+q/umklapp bookkeeping and the
// same k-block gather (both reproduced below), but it contracts against a
// real-space local potential rather than a plane wave.

auto pw_matrix_elements(mf::MF& mf,
                        nda::array_const_view<double,1> q_cryst,
                        nda::array_const_view<long,2> G_mill)
  -> std::tuple<nda::array<ComplexType,5>, nda::array<double,2>>
{
  using nda::range;
  decltype(range::all) all;

  auto& comm = mf.mpi()->comm;
  long nspin = mf.nspin();
  long nk    = mf.nkpts();
  long nbnd  = mf.nbnd();
  long mesh[3] = {mf.fft_grid_dim(0), mf.fft_grid_dim(1), mf.fft_grid_dim(2)};
  long nnr = mesh[0]*mesh[1]*mesh[2];
  long nG = G_mill.extent(0);

  utils::check(q_cryst.shape(0) == 3, "pw_matrix_elements: q_cryst must have length 3.");
  utils::check(mf.npol_in_basis() == 1, "pw_matrix_elements: requires npol == 1.");
  utils::check(mf.nkpts_ibz() == nk,
               "pw_matrix_elements: requires a full-BZ k-grid, got nkpts_ibz={}, nkpts={}.",
               mf.nkpts_ibz(), nk);
  utils::check(nG > 0, "pw_matrix_elements: G_mill is empty.");
  utils::check(G_mill.extent(1) == 3,
               "pw_matrix_elements: G_mill must be (nG, 3), got second extent {}.",
               G_mill.extent(1));

  auto recv = mf.recv();
  nda::stack_array<double,3> q_cart = {0.0,0.0,0.0};
  for(int d=0; d<3; ++d)
    q_cart(d) = q_cryst(0)*recv(0,d) + q_cryst(1)*recv(1,d) + q_cryst(2)*recv(2,d);

  // Cartesian G of every requested Miller triple, and the q+G table the caller
  // needs for v(q+G). A component beyond the grid's Nyquist limit is not
  // representable on the orbital FFT mesh: e^{iG.r} would alias to a different G.
  auto G_cart = nda::array<double,2>::zeros({nG,3});
  auto qpG_cart = nda::array<double,2>::zeros({nG,3});
  for(long p=0; p<nG; ++p) {
    for(int d=0; d<3; ++d)
      utils::check(std::abs(G_mill(p,d)) <= mesh[d]/2,
                   "pw_matrix_elements: G_mill({},{}) = {} exceeds the FFT grid "
                   "Nyquist limit mesh({})/2 = {}.",
                   p, d, G_mill(p,d), d, mesh[d]/2);
    for(int d=0; d<3; ++d) {
      G_cart(p,d) = double(G_mill(p,0))*recv(0,d) + double(G_mill(p,1))*recv(1,d)
                  + double(G_mill(p,2))*recv(2,d);
      qpG_cart(p,d) = q_cart(d) + G_cart(p,d);
    }
  }

  auto kpts_crys = mf.kpts_crystal();
  nda::array<long,1> kpq_map(nk);
  utils::calculate_kpq_map(kpts_crys, q_cryst, kpq_map);

  // M is (nG, nspin, nkpts, nbnd, nbnd) and only the root consumes it, so each
  // rank fills its own contiguous k-block and the blocks are gathered.
  int rank = comm.rank(), np = comm.size();
  auto [k0, k1] = itertools::chunk_range(0, nk, np, rank);
  long nk_loc = k1 - k0;

  auto Mloc = nda::array<ComplexType,5>::zeros(
      std::array<long,5>{nG, nspin, nk_loc, nbnd, nbnd});

  if(nk_loc > 0) {
    // e^{iG.r} is k-independent, so the plane waves are built once here rather
    // than once per (spin, k, G); only the umklapp factor e^{iG0.r} below is
    // per-k, and it is folded into the k+q orbitals as the ERI builder does.
    auto phase = memory::unified_array<ComplexType,3>::zeros({mesh[0],mesh[1],mesh[2]});
    auto phase_1D = nda::reshape(phase, std::array<long,1>{nnr});
    nda::array<ComplexType,2> W(nG, nnr);
    nda::stack_array<double,3> vec = {0.0,0.0,0.0};
    for(long p=0; p<nG; ++p) {
      for(int d=0; d<3; ++d) vec(d) = G_cart(p,d);
      utils::rspace_phase_factor(mf.lattv(), vec, phase);
      W(p,all) = phase_1D;
    }

    nda::array<ComplexType,2> psik(nbnd,nnr), psikq_conj(nbnd,nnr), Apw(nbnd,nnr);

    for(long is=0; is<nspin; ++is) {
      for(long ik=k0; ik<k1; ++ik) {
        long ikl = ik - k0;
        long ikpq = kpq_map(ik);
        // umklapp reciprocal-lattice vector: k + q = kpts(ikpq) + G0
        for(int d=0; d<3; ++d)
          vec(d) = mf.kpts(ik,d) + q_cart(d) - mf.kpts(ikpq,d);
        utils::rspace_phase_factor(mf.lattv(), vec, phase);

        mf.get_orbital_set('r', is, ik, range(0,nbnd), psik);
        mf.get_orbital_set('r', is, ikpq, range(0,nbnd), psikq_conj);
        for(long ib=0; ib<nbnd; ++ib)
          for(long r=0; r<nnr; ++r)
            psikq_conj(ib,r) = std::conj(psikq_conj(ib,r))*phase_1D(r);

        for(long p=0; p<nG; ++p) {
          auto w = W(p,all);
          for(long ib=0; ib<nbnd; ++ib)
            for(long r=0; r<nnr; ++r)
              Apw(ib,r) = psikq_conj(ib,r)*w(r);
          // M(i,j) = (1/nnr) sum_r [conj(psi_{k+q,i}) e^{i(G+G0).r}] psi_{k,j}.
          // Plain transpose, not dagger: the conjugate is already on the left.
          nda::blas::gemm(ComplexType(1.0/double(nnr)), Apw, nda::transpose(psik),
                          ComplexType(0.0), Mloc(p,is,ikl,all,all));
        }
      }
    }
  }

  nda::array<ComplexType,5> M;
  if(comm.root()) {
    M = nda::array<ComplexType,5>::zeros(
        std::array<long,5>{nG, nspin, nk, nbnd, nbnd});
    if(nk_loc > 0) M(all,all,range(k0,k1),all,all) = Mloc;
    for(int p=1; p<np; ++p) {
      auto [pk0, pk1] = itertools::chunk_range(0, nk, np, p);
      long pnk = pk1 - pk0;
      if(pnk == 0) continue;
      nda::array<ComplexType,5> tmp(nG, nspin, pnk, nbnd, nbnd);
      comm.receive_n(tmp.data(), tmp.size(), p, 0);
      M(all,all,range(pk0,pk1),all,all) = tmp;
    }
  } else if(nk_loc > 0) {
    comm.send_n(Mloc.data(), Mloc.size(), 0, 0);
  }

  return std::make_tuple(std::move(M), std::move(qpG_cart));
}

} // namespace orbitals
