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


#undef NDEBUG

#include <cmath>
#include <algorithm>

#include "catch2/catch.hpp"

#include "configuration.hpp"

#include "mpi3/environment.hpp"
#include "mpi3/communicator.hpp"
#include "mpi3/shared_communicator.hpp"

#include "IO/AppAbort.hpp"
#include "IO/app_loggers.h"

#include "nda/nda.hpp"
#include "nda/h5.hpp"
#include "numerics/distributed_array/nda.hpp"
#include "numerics/distributed_array/h5.hpp"
#include "utilities/test_common.hpp"

#include "mean_field/default_MF.hpp"
#include "hamiltonian/one_body_hamiltonian.hpp"
#include "hamiltonian/pseudo/pseudopot.h"
#include "orbitals/orbital_generator.h"
#include "orbitals/orbital_augmenter.h"
#include "utilities/mpi_context.h"

namespace bdft_tests
{

using namespace math::nda;
template <int Rank> using shape_t = std::array<long, Rank>;

/*
TEST_CASE("add_pgto", "[orbit]")
{
  auto& mpi = utils::make_unit_test_mpi_context();
  auto [outdir,prefix] = utils::utest_filename(mf::qe_source);
  auto qe_mf = mf::default_MF(mpi,mf::qe_source);

  // basic (no ortho or n0)
  {
    auto bdft_mf = orbitals::add_pgto(qe_mf,"dummy.h5",
                                    outdir+"basis.txt","nwchem",-1,false,false,0.0,false);
    auto ov = hamilt::ovlp(bdft_mf, mpi->comm, nda::range(bdft_mf.nkpts_ibz()),nda::range(bdft_mf.nbnd()));
  }
  mpi->comm.barrier();
  if(mpi->comm.root()) remove("dummy.h5");
  mpi->comm.barrier();

  // orthogonal with n0
  {
    auto bdft_mf = orbitals::add_pgto(qe_mf,"dummy.h5",
                                    outdir+"basis.txt","nwchem",-1,false,true,0.0,false);
    auto ov = hamilt::ovlp(bdft_mf, mpi->comm, nda::range(bdft_mf.nkpts_ibz()), nda::range(bdft_mf.nbnd()));
    auto ov_loc = ov.local();
    double e=0.0;
    for( auto [is,s] : itertools::enumerate(ov.local_range(0)) )   
      for( auto [ik,k] : itertools::enumerate(ov.local_range(1)) )   
        for( auto [ia,a] : itertools::enumerate(ov.local_range(2)) )   
          for( auto [ib,b] : itertools::enumerate(ov.local_range(3)) )  { 
            e += std::abs(ov_loc(is,ik,ia,ib)-(a==b?1.0:0.0));
          }
    auto e_sum = mpi->comm.all_reduce_value(e);
    utils::VALUE_EQUAL(e_sum,0.0); 
  }
  mpi->comm.barrier();
  if(mpi->comm.root()) remove("dummy.h5");
}
*/

// Frobenius diagnostics of an augmented-basis Kohn-Sham matrix, all relative and
// returned on every rank. H and S need only be valid on the global root, so a
// shared-memory window or a root-only h5 read both work. n0 = number of protected
// (parent-eigenstate) bands, p = [0,n0), a = [n0,norb).
//   [0] ||H[p,p] - diag(eps_QE)|| / ||eps_QE||   protected block
//   [1] ||H[p,a]||                / ||H||        protected <-> augmented
//   [2] ||H - H^dag||             / ||H||        hermiticity residual
//   [3] ||H - diag(H)||           / ||H||        size of the off-diagonal
//   [4] ||H[p,a] - eps_p S[p,a]|| / ||H||        the sharp gate
// The protected orbitals are KS eigenstates, so <psi_n|H_KS|phi_a> = eps_n S[n,a]
// exactly. [4] is that identity; it is machine-precision small however non-orthogonal
// the bases are, and unlike [1] it also catches a global conjugation (S is complex).
// [1] is the same statement in the orthogonal limit, so it additionally reports how
// far S[p,a] is from zero. [3] is the magnitude of the defect that storing the matrix
// repairs -- logged, never asserted.
std::array<double,5> hks_metrics(mf::MF& aug_mf, mf::MF& parent, long n0,
                                 nda::array_view<ComplexType,4> H,
                                 nda::array_view<ComplexType,4> S)
{
  auto all = nda::range::all;
  long nspin = parent.nspin();
  long nkpts_ibz = parent.nkpts_ibz();
  long norb = aug_mf.nbnd();
  auto eref = parent.eigval()(all, nda::range(nkpts_ibz), all);

  auto& mpi = *aug_mf.mpi();
  std::array<double,5> m = {0.0, 0.0, 0.0, 0.0, 0.0};
  if (mpi.comm.root()) {
    utils::check(H.extent(0)==nspin and H.extent(1)==nkpts_ibz and
                 H.extent(2)==norb and H.extent(3)==norb,
                 "hks_metrics: H shape ({},{},{},{}) != ({},{},{},{}).",
                 H.extent(0),H.extent(1),H.extent(2),H.extent(3),
                 nspin,nkpts_ibz,norb,norb);
    utils::check(S.shape() == H.shape(), "hks_metrics: S and H shapes differ.");
    double n_pro=0.0, d_pro=0.0, n_cpl=0.0, n_her=0.0, n_off=0.0, n_tot=0.0, n_res=0.0;
    for (long s = 0; s < nspin; ++s)
      for (long k = 0; k < nkpts_ibz; ++k) {
        for (long i = 0; i < norb; ++i) {
          for (long j = 0; j < norb; ++j) {
            auto h = H(s,k,i,j);
            double a2 = std::norm(h);
            n_tot += a2;
            if (i != j) n_off += a2;
            n_her += std::norm(h - std::conj(H(s,k,j,i)));
            if (i < n0 and j < n0) {
              auto r = h - ((i==j) ? ComplexType(eref(s,k,i)) : ComplexType(0.0));
              n_pro += std::norm(r);
            }
            if ((i < n0) != (j < n0)) {
              n_cpl += a2;
              // the protected index carries the eigenvalue on either side
              double eps = (i < n0) ? eref(s,k,i) : eref(s,k,j);
              n_res += std::norm(h - eps*S(s,k,i,j));
            }
          }
          if (i < n0) d_pro += eref(s,k,i)*eref(s,k,i);
        }
      }
    m[0] = std::sqrt(n_pro) / std::sqrt(d_pro);
    m[1] = std::sqrt(n_cpl) / std::sqrt(n_tot);
    m[2] = std::sqrt(n_her) / std::sqrt(n_tot);
    m[3] = std::sqrt(n_off) / std::sqrt(n_tot);
    m[4] = std::sqrt(n_res) / std::sqrt(n_tot);
  }
  mpi.comm.broadcast_n(m.data(), m.size(), 0);
  app_log(2, "  H_KS protected block   ||H[p,p] - diag(eps_QE)||/||eps_QE||   = {}", m[0]);
  app_log(2, "  H_KS protected<->aug   ||H[p,a]||/||H||                       = {}", m[1]);
  app_log(2, "  H_KS hermiticity       ||H - H^dag||/||H||                    = {}", m[2]);
  app_log(2, "  H_KS off-diagonal      ||H - diag(H)||/||H||                  = {}", m[3]);
  app_log(2, "  H_KS eigenstate ident. ||H[p,a] - eps_p S[p,a]||/||H||        = {}", m[4]);
  return m;
}

// Overlap of the augmented basis, in a shared array (the [4] gate needs it).
auto aug_ovlp(mf::MF& aug_mf)
{
  auto& mpi = *aug_mf.mpi();
  auto sS = math::shm::make_shared_array<nda::array_view<ComplexType,4>>(
      mpi, {aug_mf.nspin(), aug_mf.nkpts_ibz(), aug_mf.nbnd(), aug_mf.nbnd()});
  hamilt::set_ovlp(aug_mf, sS);
  return sS;
}

// The stored matrix, materialized on the global root (zero elsewhere).
nda::array<ComplexType,4> read_hks_h5(mf::MF& aug_mf, mf::MF& parent, std::string const& fn)
{
  long norb = aug_mf.nbnd();
  nda::array<ComplexType,4> H(parent.nspin(), parent.nkpts_ibz(), norb, norb);
  H() = ComplexType(0.0);
  if (aug_mf.mpi()->comm.root()) {
    h5::file file(fn, 'r');
    h5::group grp(file);
    auto ogrp = grp.open_group("Orbitals");
    nda::h5_read(ogrp, "H_KS_skij", H);
  }
  return H;
}

// The matrix as set_fock hands it back -- the read path every seed site uses.
auto hks_via_set_fock(mf::MF& aug_mf)
{
  auto& mpi = *aug_mf.mpi();
  auto psp = hamilt::make_pseudopot(aug_mf);
  auto sF = math::shm::make_shared_array<nda::array_view<ComplexType,4>>(
      mpi, {aug_mf.nspin(), aug_mf.nkpts_ibz(), aug_mf.nbnd(), aug_mf.nbnd()});
  hamilt::set_fock(aug_mf, psp.get(), sF, false);
  return sF;
}

// Write-time DFT KS eigval seed for an augmented basis. With nbnd_aug=0 the augmented
// basis is exactly the parent's DFT eigenstates, so the KS diagonal seed
//   eps_i = Re[<phi_i|H0 + V_H + V_xc|phi_i>]
// must reproduce the parent QE eigenvalues over the IBZ. This exercises the full
// write-time path (provisional h5 write, augmented-band pseudopot rebuild, V_Hxc from
// svsc - svloc) end-to-end. Uses the h5 QE fixture so the DFT local potential is present.
TEST_CASE("aug_ks_seed", "[orbit]")
{
  auto& mpi = utils::make_unit_test_mpi_context();
  auto qe_mf = mf::default_MF(mpi, "qe_lih223", mf::h5_input_type);

  auto augmenter = std::make_shared<orbitals::momentum_augmenter>(qe_mf);
  // nbnd_aug=0: zero-augmentation baseline; the KS diagonal should equal parent eigvals.
  auto aug_mf = orbitals::add_augmentation<HOST_MEMORY>(qe_mf, "dummy_aug.h5", augmenter,
                                                        0, 1e-6, 1e-2);

  long nspin = qe_mf.nspin();
  long nkpts_ibz = qe_mf.nkpts_ibz();
  long nbnd = qe_mf.nbnd();
  auto all = nda::range::all;
  auto ref = qe_mf.eigval()(all, nda::range(nkpts_ibz), all);
  auto got = aug_mf.eigval()(all, nda::range(nkpts_ibz), all);

  double num = 0.0, den = 0.0;
  for (long s = 0; s < nspin; ++s)
    for (long k = 0; k < nkpts_ibz; ++k)
      for (long i = 0; i < nbnd; ++i) {
        double d = got(s,k,i) - ref(s,k,i);
        num += d*d;
        den += ref(s,k,i)*ref(s,k,i);
      }
  double rel = std::sqrt(num) / std::sqrt(den);
  app_log(2, "aug_ks_seed: ||KS_diag - QE_eigval|| / ||QE_eigval|| (IBZ) = {}", rel);
  utils::VALUE_EQUAL(rel, 0.0, 1e-4);

  // The basis IS the parent eigenbasis, so the stored matrix must be diag(eps_QE):
  // the whole off-diagonal, not only the protected<->augmented block, vanishes.
  REQUIRE(aug_mf.is_augmented());
  REQUIRE(aug_mf.has_hks_matrix());
  app_log(2, "aug_ks_seed: stored H_KS matrix");
  auto sS = aug_ovlp(aug_mf);
  auto H_h5 = read_hks_h5(aug_mf, qe_mf, "dummy_aug.h5");
  auto m = hks_metrics(aug_mf, qe_mf, nbnd, H_h5(), sS.local());
  utils::VALUE_EQUAL(m[0], 0.0, 1e-4);
  utils::VALUE_EQUAL(m[2], 0.0, 1e-12);
  utils::VALUE_EQUAL(m[3], 0.0, 1e-4);

  // Read path end to end: set_fock must hand back the same matrix, i.e. the
  // parent eigenvalues on the diagonal.
  {
    auto psp = hamilt::make_pseudopot(aug_mf);
    auto sF = math::shm::make_shared_array<nda::array_view<ComplexType,4>>(
        *mpi, {nspin, nkpts_ibz, nbnd, nbnd});
    hamilt::set_fock(aug_mf, psp.get(), sF, false);
    double n2 = 0.0, d2 = 0.0;
    if (mpi->node_comm.root()) {
      auto F = sF.local();
      for (long s = 0; s < nspin; ++s)
        for (long k = 0; k < nkpts_ibz; ++k)
          for (long i = 0; i < nbnd; ++i) {
            for (long j = 0; j < nbnd; ++j)
              n2 += std::norm(F(s,k,i,j) - ((i==j) ? ComplexType(ref(s,k,i)) : ComplexType(0.0)));
            d2 += ref(s,k,i)*ref(s,k,i);
          }
    }
    mpi->comm.broadcast_n(&n2, 1, 0);
    mpi->comm.broadcast_n(&d2, 1, 0);
    double rel_F = std::sqrt(n2) / std::sqrt(d2);
    app_log(2, "aug_ks_seed: ||set_fock - diag(QE_eigval)|| / ||QE_eigval|| = {}", rel_F);
    utils::VALUE_EQUAL(rel_F, 0.0, 1e-4);
  }

  mpi->comm.barrier();
  if(mpi->comm.root()) remove("dummy_aug.h5");
  mpi->comm.barrier();
}

// Same KS seed with a genuinely augmented basis (nbnd_aug>0). The original bands survive
// orthonormalize_augmentation unchanged, so the KS diagonal over the ORIGINAL band block
// (the first nbnd of the nbnd+n_aug seeded eigval) must still reproduce the parent
// eigenvalues, even with augmentation states appended. The augmented-band entries have no
// ground truth, so we only require them to be finite/real.
//
// This is also the only fixture where H_KS is genuinely non-diagonal, so it carries the
// gates on the matrix and on the set_fock read path.
//
// What the ~2e-8 protected<->augmented block is, settled by the two sweeps below:
//   - [4] tracks [1] to nine digits, so S[p,a] is zero to roundoff and the block is NOT
//     the eigenstate identity picking up non-orthogonality;
//   - it is flat to nine digits while dtau_step moves by 1000x, and the retained
//     singular values -- hence the 1/sqrt(s) roundoff amplification -- move with it, so
//     it is NOT orthogonalization roundoff.
// What is left is the parent orbitals not being exact eigenstates of the
// H0 + (svsc - svloc) operator rebuilt here: 2e-8 is the first-order residual
// ||(H_KS - eps_n) psi_n|| seen through the augmentation space. Consistently, the
// protected block [0] sits at ~4e-15, the second-order (Rayleigh-quotient) shadow of
// that same residual. It is an accuracy floor of the reconstruction, not an indexing
// error -- a transposition or conjugation would move ~6% of the norm into this block,
// seven orders larger.
TEST_CASE("aug_ks_seed_partial", "[orbit]")
{
  auto& mpi = utils::make_unit_test_mpi_context();
  auto qe_mf = mf::default_MF(mpi, "qe_lih223", mf::h5_input_type);

  long nspin = qe_mf.nspin();
  long nkpts_ibz = qe_mf.nkpts_ibz();
  long nbnd = qe_mf.nbnd();
  long nbnd_aug = std::min(2L, nbnd);
  auto all = nda::range::all;
  auto ref = qe_mf.eigval()(all, nda::range(nkpts_ibz), all);

  // (epstol, dtau_step). dtau_step scales the raw states before orthonormalization,
  // so the retained singular values s scale with it and the 1/sqrt(s) uniformization
  // amplifies roundoff proportionally: a residual driven by that grows as dtau_step
  // shrinks. epstol is swept alongside to move the truncation threshold.
  for (auto [epstol, dtau_step] : {std::pair{1e-3, 1e-1}, std::pair{1e-6, 1e-2},
                                   std::pair{1e-9, 1e-4}}) {
    auto augmenter = std::make_shared<orbitals::momentum_augmenter>(qe_mf);
    auto aug_mf = orbitals::add_augmentation<HOST_MEMORY>(qe_mf, "dummy_aug_p.h5", augmenter,
                                                          nbnd_aug, epstol, dtau_step);
    long norb = aug_mf.nbnd();
    REQUIRE(norb > nbnd); // augmentation states were actually appended
    REQUIRE(aug_mf.has_hks_matrix());
    auto got = aug_mf.eigval()(all, nda::range(nkpts_ibz), all);

    // original block (first nbnd): KS diagonal must equal parent eigenvalues
    double num = 0.0, den = 0.0;
    for (long s = 0; s < nspin; ++s)
      for (long k = 0; k < nkpts_ibz; ++k)
        for (long i = 0; i < nbnd; ++i) {
          double d = got(s,k,i) - ref(s,k,i);
          num += d*d;
          den += ref(s,k,i)*ref(s,k,i);
        }
    double rel = std::sqrt(num) / std::sqrt(den);
    app_log(2, "aug_ks_seed_partial[epstol={:.0e}, dtau={:.0e}, norb={}]: original-block "
               "||KS_diag - QE_eigval|| / ||QE_eigval|| = {}", epstol, dtau_step, norb, rel);
    utils::VALUE_EQUAL(rel, 0.0, 1e-4);

    // augmented block (nbnd..norb): no ground truth, just require finite/real seeds
    for (long s = 0; s < nspin; ++s)
      for (long k = 0; k < nkpts_ibz; ++k)
        for (long i = nbnd; i < norb; ++i)
          REQUIRE(std::isfinite(got(s,k,i)));

    auto sS = aug_ovlp(aug_mf);
    app_log(2, "aug_ks_seed_partial[epstol={:.0e}, dtau={:.0e}]: stored H_KS matrix", epstol, dtau_step);
    auto H_h5 = read_hks_h5(aug_mf, qe_mf, "dummy_aug_p.h5");
    auto m = hks_metrics(aug_mf, qe_mf, nbnd, H_h5(), sS.local());
    utils::VALUE_EQUAL(m[0], 0.0, 1e-4);
    utils::VALUE_EQUAL(m[2], 0.0, 1e-12);
    utils::VALUE_EQUAL(m[1], 0.0, 1e-4);
    // Measured: [4] tracks [1] to nine digits, i.e. S[p,a] is zero to roundoff and
    // does NOT account for the residual. The protected states are orthogonal to the
    // augmentation states as constructed; what is left is the parent orbitals not
    // being exact eigenstates of the H0 + (svsc - svloc) operator rebuilt here. It
    // is an error floor of that reconstruction, not a basis-ordering or conjugation
    // error -- either of those would move ~6% of the norm into this block.
    utils::VALUE_EQUAL(m[4], 0.0, 1e-4);

    // Read path on a genuinely non-diagonal matrix: what set_fock hands the seed
    // sites must be exactly what is on disk.
    auto sF = hks_via_set_fock(aug_mf);
    app_log(2, "aug_ks_seed_partial[epstol={:.0e}, dtau={:.0e}]: same metrics via set_fock", epstol, dtau_step);
    auto mf_ = hks_metrics(aug_mf, qe_mf, nbnd, sF.local(), sS.local());
    double dmax = 0.0;
    for (size_t i = 0; i < m.size(); ++i) dmax = std::max(dmax, std::abs(m[i] - mf_[i]));
    app_log(2, "aug_ks_seed_partial[epstol={:.0e}, dtau={:.0e}]: "
               "max|metric(set_fock) - metric(h5)| = {}", epstol, dtau_step, dmax);
    utils::VALUE_EQUAL(dmax, 0.0, 1e-14);

    mpi->comm.barrier();
    if(mpi->comm.root()) remove("dummy_aug_p.h5");
    mpi->comm.barrier();
  }
}

TEST_CASE("eig_select", "[orbit]")
{
  auto& mpi = utils::make_unit_test_mpi_context();
  auto qe_mf = mf::default_MF(mpi,mf::qe_source);

  int n0 = int(qe_mf.nelec()/2.0);
  if(const char* env_p = std::getenv("N0")) n0 = std::atoi(env_p);
  int nblk = (qe_mf.nbnd()-n0)/3;
  if(const char* env_p = std::getenv("NBLK")) nblk = std::atoi(env_p);
  
  auto bdft_mf = orbitals::eigenstate_selection(qe_mf,"dummy2.h5","linear",n0,nblk);
  auto ov = hamilt::ovlp(bdft_mf, mpi->comm, nda::range(bdft_mf.nkpts_ibz()), nda::range(bdft_mf.nbnd()));

  if(mpi->comm.root()) remove("dummy2.h5");
}

}

