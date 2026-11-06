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
#include "methods/ERI/eri_utils.hpp"
#include "methods/HF/hf_t.h"
#include "mean_field/symmetry/unfold_bz.h"
#include "utilities/symmetry.hpp"
#include <filesystem>
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

// Directory ./prefix_dir/ holding the QE fixture `fixture` with a midgap Fermi energy added to its pw2coqui h5
// (System/fermi_energy; thc nbnd_protected needs one and the LiH fixtures lack it). The other fixture files are
// symlinks. Returns the h5 path; the directory is written on the root. The copy keeps the Hamiltonian/<pp_type>/fullbz
// group (projector source of unfold_mf).
std::string make_fixture_dir_with_efermi(utils::mpi_context_t<boost::mpi3::communicator,boost::mpi3::shared_communicator>& mpi,
                                         std::string const& fixture, std::string const& dir, double efermi)
{
  auto [outdir, pf] = utils::utest_filename(fixture);
  std::string fn = dir + "/" + pf + ".coqui.h5";
  if (mpi.comm.root()) {
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    for (auto const& e : std::filesystem::directory_iterator(outdir))
      if (e.path().filename() != pf + ".coqui.h5")
        std::filesystem::create_symlink(std::filesystem::absolute(e.path()), dir + "/" + e.path().filename().string());
    std::filesystem::copy_file(outdir + pf + ".coqui.h5", fn);
    h5::file f(fn, 'a');
    h5::group sgrp = h5::group(f).open_group("System");
    h5::h5_write_attribute(sgrp, "fermi_energy", efermi);
  }
  mpi.comm.barrier();
  return fn;
}

// Smallest R >= R_min such that the gap eigval(R) - eigval(R-1) of the parent exceeds gap_min at every IBZ k, i.e. the
// first R bands form complete degenerate multiplets everywhere.
long closed_band_count(mf::MF& mf, long R_min, double gap_min)
{
  auto ev = mf.eigval();
  for (long R = R_min; R < mf.nbnd(); ++R) {
    bool closed = true;
    for (long k = 0; k < mf.nkpts_ibz(); ++k) closed = closed and (ev(0,k,R) - ev(0,k,R-1) > gap_min);
    if (closed) return R;
  }
  APP_ABORT("closed_band_count: no closed band count");
  return -1;
}

// Dense copy of a sparse nb x nb rotation matrix D
template<class Sparse_t>
nda::array<ComplexType,2> dense_of(Sparse_t const& D, long nb)
{
  nda::array<ComplexType,2> Dd(nb, nb);
  Dd() = ComplexType(0.0);
  for (long r = 0; r < nb; ++r) {
    auto vals = D[r].values();
    auto cols = D[r].columns();
    for (long i = 0; i < vals.size(); ++i) Dd(r, cols(i)) = vals(i);
  }
  return Dd;
}

// ||D D^dagger - 1||_max of a sparse nb x nb rotation matrix D
template<class Sparse_t>
double unitarity_defect(Sparse_t const& D, long nb)
{
  auto Dd = dense_of(D, nb);
  double err = 0.0;
  for (long a = 0; a < nb; ++a)
    for (long b = 0; b < nb; ++b) {
      ComplexType dd = 0.0;
      for (long c = 0; c < nb; ++c) dd += Dd(a,c) * std::conj(Dd(b,c));
      err = std::max(err, std::abs(dd - (a==b ? ComplexType(1.0) : ComplexType(0.0))));
    }
  return err;
}

// max_(s,k) ||D D^dagger - 1||_max of the stored symmetry rotations D(s,k) of mf, over all s > 0 and all k of the full BZ
double stored_D_unitarity_defect(mf::MF& mf)
{
  double err = 0.0;
  for (long s = 1; s < mf.qsymms().extent(0); ++s)
    for (long k = 0; k < mf.nkpts(); ++k) {
      auto [cjg, Dp] = mf.symmetry_rotation(s, k);
      err = std::max(err, unitarity_defect(*Dp, mf.nbnd()));
    }
  return err;
}

// IBZ augmented basis: symmetry rotations and the IBZ -> full-BZ chain.
//  (a) The IBZ augmented mean field builds (add_augmentation on a symmetry-reduced MF).
//  (b) Its stored rotations D(s,k), built by the dense branch of generate_dmatrix (assume_irreducible = not augmented),
//      are unitary: ||D D^dagger - 1||_max against the floor measured on this fixture, and to < 1e-10 when built dense
//      (no 1e-8 sparsifier) and without the row normalization (control: dirs {0}). Negative control: the
//      eigenvalue-based branch (assume_irreducible = true), which needs sorted eigval; the eigval of an augmented basis
//      is diag(H_KS), which is not sorted. Control 0 is momentum_augmenter with dirs {0} (not a vector set, so the aug
//      span is not invariant and D is non unitary); control 1 is the unsortedness of diag(H_KS) (the eigenvalue-based
//      branch aborts); control 2 is the eigenvalue-based branch on sorted eigval, which differs from the dense D.
//  (c) THC with band weights, nbnd_protected and exclude_unprotected_pairs on the IBZ augmented MF, then the HF Fock
//      matrix of the diagonal initial density, against the same THC/HF on its mf::unfold_mf view (full BZ, pivot_file
//      = the IBZ pivots). Protected block: within 5e-4. Whole matrix (plain parent, three fit settings): the unfolded
//      IBZ Fock is as close to a tight reference fit as the view Fock (err_IBZ < 1.5 err_view). Negative controls: the
//      IBZ Fock unfolded without the conjugation at time-reversal k, and the view's THC refit without the band weights.
TEST_CASE("aug_ibz_symmetry", "[orbit]")
{
  auto& mpi = utils::make_unit_test_mpi_context();
  auto& comm = mpi->comm;
  const double efermi = 0.07;  // midgap of qe_lih223_sym: max VB 0.0163, min CB 0.1268 (Ha)
  const std::string parent_dir = "./aug_ibz_parent";
  std::string parent_fn = make_fixture_dir_with_efermi(*mpi, "qe_lih223_sym", parent_dir, efermi);
  auto qe_mf = mf::make_MF(mpi, mf::qe_source, parent_dir + "/", "pwscf", mf::h5_input_type);
  REQUIRE(qe_mf.nkpts_ibz() < qe_mf.nkpts());
  long nbnd = qe_mf.nbnd();

  auto make_aug = [&](long R, std::string fn) {
    auto augmenter = std::make_shared<orbitals::momentum_augmenter>(qe_mf);
    return orbitals::add_augmentation<HOST_MEMORY>(qe_mf, fn, augmenter, R, 1e-3, 1e-1);
  };
  auto cleanup = [&](std::string fn) { comm.barrier(); if (comm.root()) remove(fn.c_str()); comm.barrier(); };

  long R = closed_band_count(qe_mf, 3, 1e-3);
  auto aug_mf = make_aug(R, "aug_ibz.h5");
  REQUIRE(aug_mf.is_augmented());
  REQUIRE(aug_mf.nkpts_ibz() < aug_mf.nkpts());
  REQUIRE(aug_mf.nbnd() > nbnd);

  // (b) unitarity of the stored D
  double eps_D = stored_D_unitarity_defect(aug_mf);
  app_log(2, "aug_ibz_symmetry: R = {}, nbnd {} -> {}; max ||D D^dagger - 1||_max (stored D) = {:.3e}",
          R, nbnd, aug_mf.nbnd(), eps_D);
  REQUIRE(eps_D < 3e-8);   // measured 2.99e-9 (floor of the 1e-8 sparsifier of to_csr), bound 10x the floor
  // max_(s,k) ||D D^dagger - 1||_max of the rotations of the bdft file ./prefix.h5, built as for an augmented basis
  // (assume_irreducible = false), stored dense (no 1e-8 sparsifier) and without the row normalization: the unitarity
  // defect that generate_dmatrix logs
  auto unnormalized_D_defect = [&](std::string const& prefix) {
    mf::bdft::bdft_readonly m(mpi, "./", prefix);
    auto slist = utils::find_inverse_symmetry(m.bz().qsymms, m.bz().symm_list);
    auto [sk_to_n, dmat] = utils::generate_dmatrix<false>(m, m.bz().symm_list, slist, false, false);
    auto D_all = dmat.local();
    long nb = D_all.extent(1);
    double err = 0.0;
    for (long isk = 0; isk < D_all.extent(0); ++isk)
      for (long a = 0; a < nb; ++a)
        for (long b = 0; b < nb; ++b) {
          ComplexType dd = 0.0;
          for (long c = 0; c < nb; ++c) dd += D_all(isk,a,c) * std::conj(D_all(isk,b,c));
          err = std::max(err, std::abs(dd - (a==b ? ComplexType(1.0) : ComplexType(0.0))));
        }
    return err;
  };
  double eps_D_raw = unnormalized_D_defect("aug_ibz");
  app_log(2, "aug_ibz_symmetry: max ||D D^dagger - 1||_max, dense and before the row normalization = {:.3e}", eps_D_raw);
  REQUIRE(eps_D_raw < 1e-10);   // measured 2.9e-13 (np 1), 3.8e-13 (np 8); dirs {0} 1.0

  // (b) control 0: momentum states along x only are not a vector set, so the aug span is not invariant under the
  // symmetry operations
  {
    auto augmenter_x = std::make_shared<orbitals::momentum_augmenter>(qe_mf, std::vector<int>{0});
    auto aug_x = orbitals::add_augmentation<HOST_MEMORY>(qe_mf, "aug_ibz_x.h5", augmenter_x, R, 1e-3, 1e-1);
    double eps_x = stored_D_unitarity_defect(aug_x);
    app_log(2, "aug_ibz_symmetry: dirs {{0}} (not invariant): max ||D D^dagger - 1||_max = {:.3e} (dirs {{0,1,2}}: {:.3e})",
            eps_x, eps_D);
    REQUIRE(eps_x > 1e-3);
    double eps_x_raw = unnormalized_D_defect("aug_ibz_x");
    app_log(2, "aug_ibz_symmetry: dirs {{0}}: max ||D D^dagger - 1||_max, dense and before the row normalization = {:.3e}",
            eps_x_raw);
    REQUIRE(eps_x_raw > 1e-3);
    cleanup("aug_ibz_x.h5");
  }
  // (b) control 1: assume_irreducible = true needs sorted eigval at every k, which diag(H_KS) is not
  {
    auto eigv = aug_mf.eigval();
    bool sorted = true;
    for (long k = 0; k < aug_mf.nkpts(); ++k)
      sorted = sorted and std::is_sorted(eigv(0,k,nda::range::all).begin(), eigv(0,k,nda::range::all).end(),
                                         [](double a, double b) { return a < b - 1e-6; });
    app_log(2, "aug_ibz_symmetry: diag(H_KS) sorted at every k: {} (generate_dmatrix(assume_irreducible=true) "
               "aborts if not)", sorted);
    REQUIRE(not sorted);
  }

  // (b) control 2: the eigenvalue-based branch (assume_irreducible = true) on the same basis, with the eigval
  // sorted so that it does not abort. It builds D block-diagonally over groups of diag(H_KS) closer than 1e-4, which
  // cuts the couplings between the states of different groups; after the row normalization such a D is still unitary,
  // so it is compared entrywise with the stored (dense) D.
  {
    comm.barrier();
    if (comm.root()) {
      std::filesystem::copy_file("aug_ibz.h5", "aug_ibz_sorted.h5", std::filesystem::copy_options::overwrite_existing);
      h5::file f("aug_ibz_sorted.h5", 'a');
      h5::group og = h5::group(f).open_group("Orbitals");
      nda::array<double,3> ev;
      nda::h5_read(og, "eigval", ev);
      for (long s = 0; s < ev.extent(0); ++s)
        for (long k = 0; k < ev.extent(1); ++k) std::sort(ev(s,k,nda::range::all).begin(), ev(s,k,nda::range::all).end());
      nda::h5_write(og, "eigval", ev, false);
    }
    comm.barrier();
    mf::bdft::bdft_readonly aug_sorted(mpi, "./", "aug_ibz_sorted");
    auto slist = utils::find_inverse_symmetry(aug_sorted.bz().qsymms, aug_sorted.bz().symm_list);
    auto [sk_to_n, dmat] = utils::generate_dmatrix<true>(aug_sorted, aug_sorted.bz().symm_list, slist, true);
    // same (s,k) -> index map in both builds, since both use the same symmetries
    double dD_irred = 0.0;
    for (long s = 1; s < aug_mf.qsymms().extent(0); ++s)
      for (long k = 0; k < aug_mf.nkpts(); ++k) {
        auto [cjg, Dp] = aug_mf.symmetry_rotation(s, k);
        long n = aug_sorted.bz().kp_trev(k) ? sk_to_n(s-1, aug_sorted.bz().kp_trev_pair(k)) : sk_to_n(s-1, k);
        auto D_dense = dense_of(*Dp, aug_mf.nbnd());
        auto D_irred = dense_of(dmat.at(n), aug_mf.nbnd());
        for (long a = 0; a < aug_mf.nbnd(); ++a)
          for (long b = 0; b < aug_mf.nbnd(); ++b) dD_irred = std::max(dD_irred, std::abs(D_irred(a,b) - D_dense(a,b)));
      }
    app_log(2, "aug_ibz_symmetry: assume_irreducible = true (sorted eigval) vs dense D: max|D_irred - D_dense| = {:.3e}",
            dD_irred);
    REQUIRE(dD_irred > 100.0 * eps_D);
    cleanup("aug_ibz_sorted.h5");
  }

  // (c) THC (weights + protected bands + pair exclusion) and the HF Fock matrix, IBZ vs unfolded view
  const long N_P = 4;
  // THC reader ptree; nbnd_orig is meaningful for an augmented mean field only
  auto make_pt = [&](std::shared_ptr<mf::MF> m, std::string pivot_file, bool band_weights, bool protect = true,
                     bool exclude = true, double thresh = 1e-4) {
    auto pt = methods::make_thc_reader_ptree(0, "", "incore", "", "bdft", thresh, 1.4*m->wfc_truncated_grid()->ecut());
    if (protect) {   // energy weights of the band tail [N_P, nbnd_orig) (+ augmentation weights if augmented)
      pt.put("nbnd_protected", N_P);
      if (m->is_augmented()) pt.put("nbnd_orig", nbnd);
      pt.put("exclude_unprotected_pairs", exclude);
    }
    pt.put("band_weights", band_weights);
    pt.put("pivot_file", pivot_file);
    pt.put("save", "");
    return pt;
  };
  // pivots of m, fitted on the IBZ k (weights + exclusion on the IBZ path)
  auto make_pivots = [&](std::shared_ptr<mf::MF> m, std::string pivot_file, bool protect = true, bool exclude = true) {
    auto pt = make_pt(m, "", true, protect, exclude);
    pt.put("save", pivot_file);
    methods::make_thc_pivots(m, pt);
  };
  // HF Fock matrix of the diagonal initial density (occupations of the parent bands, 0 on the augmentation states;
  // the occupations of an augmented mean field are zero), refit on the pivots of pt
  auto fock = [&](std::shared_ptr<mf::MF> m, auto const& pt) {
    methods::thc_reader_t thc(m, pt);
    methods::solvers::hf_t hf("ignore_g0");
    long ns = m->nspin(), nk = m->nkpts_ibz(), nb = m->nbnd();
    auto sS = math::shm::make_shared_array<nda::array_view<ComplexType,4>>(*mpi, {ns, nk, nb, nb});
    hamilt::set_ovlp(*m, sS);
    auto sF = math::shm::make_shared_array<nda::array_view<ComplexType,4>>(*mpi, {ns, nk, nb, nb});
    nda::array<ComplexType,4> Dm(ns, nk, nb, nb);
    Dm() = ComplexType(0.0);
    auto occ = qe_mf.occ();
    for (long s = 0; s < ns; ++s)
      for (long k = 0; k < nk; ++k) {
        long kp = (nk == qe_mf.nkpts_ibz()) ? k : qe_mf.kp_to_ibz(k);   // IBZ list or full-BZ list
        for (long a = 0; a < nbnd; ++a) Dm(s,k,a,a) = occ(s,kp,a);
      }
    hf.evaluate(sF, Dm, thc, sS.local(), true, true);
    mpi->node_comm.barrier();
    nda::array<ComplexType,4> F(sF.local());
    comm.broadcast_n(F.data(), F.size(), 0);
    return F;
  };
  auto fnorm = [](nda::array<ComplexType,4> const& A) {
    double n2 = 0.0;
    for (long i = 0; i < A.size(); ++i) n2 += std::norm(A.data()[i]);
    return std::sqrt(n2);
  };
  // ||A - B|| / ||B|| of two Fock matrices on the same k list
  auto rel_full = [&](nda::array<ComplexType,4> const& A, nda::array<ComplexType,4> const& B) {
    nda::array<ComplexType,4> d(A - B);
    return fnorm(d) / fnorm(B);
  };
  // ||F_ibz(copy/conj to k) - F_view(k)|| / ||F_view|| on the leading nb_blk x nb_blk block; conj_trev = false copies
  // without the conjugation at time-reversal k (negative control)
  auto rel_diff = [&](mf::MF& m, nda::array<ComplexType,4> const& Fi, nda::array<ComplexType,4> const& Fv, long nb_blk,
                      bool conj_trev = true) {
    double num = 0.0, den = 0.0;
    for (long s = 0; s < m.nspin(); ++s)
      for (long k = 0; k < m.nkpts(); ++k) {
        long kib = m.kp_to_ibz(k);
        bool tr = conj_trev and m.kp_trev(k);
        for (long i = 0; i < nb_blk; ++i)
          for (long j = 0; j < nb_blk; ++j) {
            ComplexType ref = tr ? std::conj(Fi(s,kib,i,j)) : Fi(s,kib,i,j);
            num += std::norm(ref - Fv(s,k,i,j));
            den += std::norm(Fv(s,k,i,j));
          }
      }
    return std::sqrt(num / den);
  };

  // (c) plain parent, no weights / energy weights / energy weights + pair exclusion. On the whole matrix the IBZ Fock
  // (unfolded) and the view Fock differ by the THC asymmetry of the fit (V(q) of q outside the IBZ fitted directly on the
  // view, rotated on the IBZ path); both are fits of the same quality, so the IBZ Fock is as close to a tight reference
  // F_ref (view, fresh pivots, unweighted, no protected bands, thresh 1e-6) as the view Fock: ||F_ibz,unf - F_ref|| <
  // 1.5 ||F_view - F_ref||. The weights and the exclusion make the fit of the unprotected pairs, hence both, larger.
  {
    auto qe_ptr = std::shared_ptr<mf::MF>(std::shared_ptr<void>{}, &qe_mf);   // non-owning
    mf::unfold_mf(qe_mf, "plain_view.h5");
    comm.barrier();
    auto view = std::make_shared<mf::MF>(mf::make_MF(mpi, mf::bdft_source, "./", "plain_view", mf::h5_input_type));
    REQUIRE(view->nkpts_ibz() == view->nkpts());
    auto F_ref = fock(view, make_pt(view, "", false, false, false, 1e-6));   // reference: unweighted, thresh 1e-6
    for (auto [protect, exclude] : {std::pair{false, false}, std::pair{true, false}, std::pair{true, true}}) {
      make_pivots(qe_ptr, "plain_ibz_pivots.h5", protect, exclude);
      auto F_ibz = fock(qe_ptr, make_pt(qe_ptr, "plain_ibz_pivots.h5", true, protect, exclude));
      auto F_view = fock(view, make_pt(view, "plain_ibz_pivots.h5", true, protect, exclude));
      double gap = rel_diff(qe_mf, F_ibz, F_view, nbnd), gap_P = rel_diff(qe_mf, F_ibz, F_view, N_P);
      double err_view = rel_full(F_view, F_ref), err_ibz = rel_diff(qe_mf, F_ibz, F_ref, nbnd);
      double gap_noconj = rel_diff(qe_mf, F_ibz, F_view, nbnd, false);
      app_log(2, "aug_ibz_symmetry: plain parent, energy weights {}, pair exclusion {}: ||F_ibz,unfolded - F_view||/||F_view|| "
                 "whole matrix {:.3e} (without conj at TR k {:.3e}), protected block {:.3e}; ||F - F_ref||/||F_ref|| "
                 "IBZ {:.3e}, view {:.3e}", protect, exclude, gap, gap_noconj, gap_P, err_ibz, err_view);
      // measured err_ibz / err_view 0.96-1.06 at np 1, 4 and 8; at np 8 gap 1.1e-3, 4.8e-3, 2.2e-2 against err_view
      // 1.4e-3, 6.8e-3, 3.7e-2 (no weights / weights / weights + exclusion); protected block 1.1-2.5e-4; without conj at
      // TR k 0.63
      REQUIRE(gap_P < 5e-4);
      REQUIRE(err_ibz < 1.5 * err_view);
      REQUIRE(gap_noconj > err_view);
      cleanup("plain_ibz_pivots.h5");
    }
    cleanup("plain_view.h5");
  }

  {
    auto aug_ptr = std::make_shared<mf::MF>(std::move(aug_mf));
    make_pivots(aug_ptr, "aug_ibz_pivots.h5");
    auto F_ibz = fock(aug_ptr, make_pt(aug_ptr, "aug_ibz_pivots.h5", true));

    // unfolded view of the IBZ augmented MF, projectors from the pw2coqui h5 of the parent
    mf::unfold_mf(*aug_ptr, "aug_ibz_view.h5", parent_fn);
    comm.barrier();
    auto view = std::make_shared<mf::MF>(mf::make_MF(mpi, mf::bdft_source, "./", "aug_ibz_view", mf::h5_input_type));
    REQUIRE(view->is_augmented());
    REQUIRE(view->nkpts_ibz() == view->nkpts());
    auto F_view = fock(view, make_pt(view, "aug_ibz_pivots.h5", true));
    auto F_view_nw = fock(view, make_pt(view, "aug_ibz_pivots.h5", false));

    app_log(2, "aug_ibz_symmetry: ||F_ibz|| = {}, ||F_view|| = {}, ||F_view_nw|| = {}", fnorm(F_ibz), fnorm(F_view), fnorm(F_view_nw));
    double d_w = rel_diff(*aug_ptr, F_ibz, F_view, aug_ptr->nbnd()), d_nw = rel_diff(*aug_ptr, F_ibz, F_view_nw, aug_ptr->nbnd());
    double d_w_P = rel_diff(*aug_ptr, F_ibz, F_view, N_P), d_nw_P = rel_diff(*aug_ptr, F_ibz, F_view_nw, N_P);
    app_log(2, "aug_ibz_symmetry: ||F_ibz,unfolded - F_view||/||F_view||, whole matrix: {:.3e} (view refit with band "
               "weights), {:.3e} (without); protected block [0,{}): {:.3e}, {:.3e}", d_w, d_nw, N_P, d_w_P, d_nw_P);
    double d_noconj_P = rel_diff(*aug_ptr, F_ibz, F_view, N_P, false);
    app_log(2, "aug_ibz_symmetry: protected block without conj at TR k {:.3e}", d_noconj_P);
    // measured: protected block 1.2e-4 (2.5e-1 without conj at TR k, 1.3e-3 without the weights carried); the
    // whole-matrix gap (logged) is the fit error of the unprotected pairs, checked on the plain parent above
    REQUIRE(d_w_P < 5e-4);
    REQUIRE(d_noconj_P > 5e-4);
    REQUIRE(d_nw_P > 3.0 * d_w_P);      // the band weights carried to the view matter

    cleanup("aug_ibz_view.h5");
    cleanup("aug_ibz_pivots.h5");
  }
  cleanup("aug_ibz.h5");
  comm.barrier(); if (comm.root()) std::filesystem::remove_all(parent_dir); comm.barrier();
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

