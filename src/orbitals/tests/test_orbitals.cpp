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

  mpi->comm.barrier();
  if(mpi->comm.root()) remove("dummy_aug.h5");
  mpi->comm.barrier();
}

// Same KS seed with a genuinely augmented basis (nbnd_aug>0). The original bands survive
// orthonormalize_augmentation unchanged, so the KS diagonal over the ORIGINAL band block
// (the first nbnd of the nbnd+n_aug seeded eigval) must still reproduce the parent
// eigenvalues, even with augmentation states appended. The augmented-band entries have no
// ground truth, so we only require them to be finite/real.
TEST_CASE("aug_ks_seed_partial", "[orbit]")
{
  auto& mpi = utils::make_unit_test_mpi_context();
  auto qe_mf = mf::default_MF(mpi, "qe_lih223", mf::h5_input_type);

  long nspin = qe_mf.nspin();
  long nkpts_ibz = qe_mf.nkpts_ibz();
  long nbnd = qe_mf.nbnd();
  long nbnd_aug = std::min(2L, nbnd);

  auto augmenter = std::make_shared<orbitals::momentum_augmenter>(qe_mf);
  auto aug_mf = orbitals::add_augmentation<HOST_MEMORY>(qe_mf, "dummy_aug_p.h5", augmenter,
                                                        nbnd_aug, 1e-6, 1e-2);

  long norb = aug_mf.nbnd();
  REQUIRE(norb > nbnd); // augmentation states were actually appended
  auto all = nda::range::all;
  auto ref = qe_mf.eigval()(all, nda::range(nkpts_ibz), all);
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
  app_log(2, "aug_ks_seed_partial: original-block ||KS_diag - QE_eigval|| / ||QE_eigval|| = {}", rel);
  utils::VALUE_EQUAL(rel, 0.0, 1e-4);

  // augmented block (nbnd..norb): no ground truth, just require finite/real seeds
  for (long s = 0; s < nspin; ++s)
    for (long k = 0; k < nkpts_ibz; ++k)
      for (long i = nbnd; i < norb; ++i)
        REQUIRE(std::isfinite(got(s,k,i)));

  mpi->comm.barrier();
  if(mpi->comm.root()) remove("dummy_aug_p.h5");
  mpi->comm.barrier();
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

