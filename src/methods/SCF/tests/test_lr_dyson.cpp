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

#include "catch2/catch.hpp"

#include "configuration.hpp"
#include "mpi3/environment.hpp"
#include "mpi3/communicator.hpp"
#include "mpi3/shared_communicator.hpp"

#include "utilities/test_common.hpp"

#include "nda/nda.hpp"
#include "nda/h5.hpp"

#include "utilities/mpi_context.h"
#include "mean_field/MF.hpp"
#include "mean_field/mf_utils.hpp"
#include "methods/SCF/scf_common.hpp"
#include "methods/SCF/simple_dyson.h"
#include "methods/SCF/lr_dyson.hpp"
#include "methods/SCF/lr_precompute.hpp"
#include "hamiltonian/pseudo/pseudopot.h"

namespace lr_dyson_tests {

  using utils::mpi_context_t;
  using utils::VALUE_EQUAL;
  using utils::ARRAY_EQUAL;
  namespace mpi3 = boost::mpi3;
  using namespace methods;

  TEST_CASE("lr_dyson_init", "[methods_scf]") {
    auto& context = utils::make_unit_test_mpi_context();
    std::string source_path = PROJECT_SOURCE_DIR;
    std::string filepath = source_path + "/tests/unit_test_files/pyscf/si_kp222_krhf/";

    double beta = 1000;
    double wmax = 1.2;
    auto mf = mf::make_MF(context, mf::pyscf_source, filepath, "pyscf");
    imag_axes_ft::IAFT ft(beta, wmax, imag_axes_ft::ir_basis);
    simple_dyson dyson(std::addressof(mf), std::addressof(ft));

    // Test with q = 0 (Gamma point)
    nda::array<double, 1> q_vec{0.0, 0.0, 0.0};
    lr_dyson lr_dys(dyson, q_vec);

    // Check that k+q mapping is identity for q=0
    CHECK(lr_dys.is_q_gamma() == true);
    auto kpq_map = lr_dys.kpq_map();
    for (int ik = 0; ik < mf.nkpts(); ++ik) {
      CHECK(kpq_map(ik) == ik);
    }
    context->comm.barrier();
  }

  TEST_CASE("lr_dyson_fixed_sigma_q0", "[methods_scf]") {
    /**
     * Test LR Dyson equation: ΔG = G @ ΔH0 @ G for q=0
     *
     * Validation: Compare with finite difference:
     *   ΔG_fd = (G[H0+ε·ΔH0] - G[H0-ε·ΔH0]) / (2ε)
     */
    auto& context = utils::make_unit_test_mpi_context();
    std::string source_path = PROJECT_SOURCE_DIR;
    std::string filepath = source_path + "/tests/unit_test_files/pyscf/si_kp222_krhf/";

    double beta = 100;  // Lower beta for faster test
    double wmax = 4.0;
    auto mf = mf::make_MF(context, mf::pyscf_source, filepath, "pyscf");
    imag_axes_ft::IAFT ft(beta, wmax, imag_axes_ft::ir_basis);

    int ns = mf.nspin();
    int nk = mf.nkpts_ibz();
    int nb = mf.nbnd();
    int nt = ft.nt_f();

    // Create simple_dyson and lr_dyson
    simple_dyson dyson(std::addressof(mf), std::addressof(ft));
    nda::array<double, 1> q_vec{0.0, 0.0, 0.0};
    lr_dyson lr_dys(dyson, q_vec);

    // Create shared arrays
    sArray_t<Array_view_4D_t> F(math::shm::make_shared_array<Array_view_4D_t>(
        *context, {ns, nk, nb, nb}));
    sArray_t<Array_view_5D_t> G(math::shm::make_shared_array<Array_view_5D_t>(
        *context, {nt, ns, nk, nb, nb}));
    sArray_t<Array_view_5D_t> Sigma(math::shm::make_shared_array<Array_view_5D_t>(
        *context, {nt, ns, nk, nb, nb}));
    sArray_t<Array_view_4D_t> Dm(math::shm::make_shared_array<Array_view_4D_t>(
        *context, {ns, nk, nb, nb}));
    sArray_t<Array_view_4D_t> DeltaH0(math::shm::make_shared_array<Array_view_4D_t>(
        *context, {ns, nk, nb, nb}));
    sArray_t<Array_view_5D_t> DeltaG(math::shm::make_shared_array<Array_view_5D_t>(
        *context, {nt, ns, nk, nb, nb}));
    sArray_t<Array_view_4D_t> DeltaDm(math::shm::make_shared_array<Array_view_4D_t>(
        *context, {ns, nk, nb, nb}));

    // Initialize F (Fock matrix from non-interacting system)
    hamilt::pseudopot psp(mf);
    hamilt::set_fock(mf, std::addressof(psp), F, true);

    // Get converged G at mu=0 with zero self-energy
    if (context->node_comm.root()) {
      Sigma.local()() = 0.0;
    }
    context->comm.barrier();

    double mu = update_mu(0.0, dyson, mf, ft, F, Sigma);
    update_G(dyson, mf, ft, Dm, G, F, Sigma, mu, true);
    context->comm.barrier();

    // Create a random Hermitian perturbation
    if (context->node_comm.root()) {
      std::srand(42);
      auto DH0_loc = DeltaH0.local();
      for (int is = 0; is < ns; ++is) {
        for (int ik = 0; ik < nk; ++ik) {
          for (int i = 0; i < nb; ++i) {
            for (int j = 0; j <= i; ++j) {
              double re = 0.01 * (std::rand() / double(RAND_MAX) - 0.5);
              double im = (i == j) ? 0.0 : 0.01 * (std::rand() / double(RAND_MAX) - 0.5);
              DH0_loc(is, ik, i, j) = ComplexType(re, im);
              DH0_loc(is, ik, j, i) = ComplexType(re, -im);
            }
          }
        }
      }
    }
    context->comm.barrier();

    // Create zero ΔF for LR Dyson call (no ΔΣ)
    sArray_t<Array_view_4D_t> DeltaF(math::shm::make_shared_array<Array_view_4D_t>(
        *context, {ns, nk, nb, nb}));
    if (context->node_comm.root()) {
      DeltaF.local() = ComplexType(0.0);
    }
    context->comm.barrier();

    // Precompute G(iω) in shared memory and pass to lr_dyson before solving
    auto sG_wskij = lr_precompute_G_omega(*context, G, ft);
    lr_dys.set_cached_G_omega(&sG_wskij);

    // Compute ΔG using LR Dyson (no ΔΣ, fix_density=false)
    lr_dys.solve_lr_dyson(DeltaDm, DeltaH0, DeltaF,
                          static_cast<const sArray_t<Array_view_5D_t>*>(nullptr), false);
    lr_dys.materialize_DeltaG_tau(DeltaG);
    context->comm.barrier();

    // Compute ΔG using finite difference
    double eps = 1e-5;
    sArray_t<Array_view_5D_t> G_plus(math::shm::make_shared_array<Array_view_5D_t>(
        *context, {nt, ns, nk, nb, nb}));
    sArray_t<Array_view_5D_t> G_minus(math::shm::make_shared_array<Array_view_5D_t>(
        *context, {nt, ns, nk, nb, nb}));
    sArray_t<Array_view_4D_t> Dm_tmp(math::shm::make_shared_array<Array_view_4D_t>(
        *context, {ns, nk, nb, nb}));

    // G[H0 + eps*DeltaH0]
    {
      // Temporarily modify H0
      auto H0_orig = dyson.sH0_skij().local();
      auto DH0_loc = DeltaH0.local();
      if (context->node_comm.root()) {
        for (int is = 0; is < ns; ++is) {
          for (int ik = 0; ik < nk; ++ik) {
            for (int i = 0; i < nb; ++i) {
              for (int j = 0; j < nb; ++j) {
                H0_orig(is, ik, i, j) += eps * DH0_loc(is, ik, i, j);
              }
            }
          }
        }
      }
      context->comm.barrier();

      update_G(dyson, mf, ft, Dm_tmp, G_plus, F, Sigma, mu, true);

      // Restore H0
      if (context->node_comm.root()) {
        for (int is = 0; is < ns; ++is) {
          for (int ik = 0; ik < nk; ++ik) {
            for (int i = 0; i < nb; ++i) {
              for (int j = 0; j < nb; ++j) {
                H0_orig(is, ik, i, j) -= eps * DH0_loc(is, ik, i, j);
              }
            }
          }
        }
      }
      context->comm.barrier();
    }

    // G[H0 - eps*DeltaH0]
    {
      auto H0_orig = dyson.sH0_skij().local();
      auto DH0_loc = DeltaH0.local();
      if (context->node_comm.root()) {
        for (int is = 0; is < ns; ++is) {
          for (int ik = 0; ik < nk; ++ik) {
            for (int i = 0; i < nb; ++i) {
              for (int j = 0; j < nb; ++j) {
                H0_orig(is, ik, i, j) -= eps * DH0_loc(is, ik, i, j);
              }
            }
          }
        }
      }
      context->comm.barrier();

      update_G(dyson, mf, ft, Dm_tmp, G_minus, F, Sigma, mu, true);

      // Restore H0
      if (context->node_comm.root()) {
        for (int is = 0; is < ns; ++is) {
          for (int ik = 0; ik < nk; ++ik) {
            for (int i = 0; i < nb; ++i) {
              for (int j = 0; j < nb; ++j) {
                H0_orig(is, ik, i, j) += eps * DH0_loc(is, ik, i, j);
              }
            }
          }
        }
      }
      context->comm.barrier();
    }

    // Compute finite difference: (G_plus - G_minus) / (2*eps)
    double max_error = 0.0;
    double max_DeltaG = 0.0;
    if (context->node_comm.root()) {
      auto DG_lr = DeltaG.local();
      auto Gp = G_plus.local();
      auto Gm = G_minus.local();

      for (int it = 0; it < nt; ++it) {
        for (int is = 0; is < ns; ++is) {
          for (int ik = 0; ik < nk; ++ik) {
            for (int i = 0; i < nb; ++i) {
              for (int j = 0; j < nb; ++j) {
                ComplexType DG_fd = (Gp(it, is, ik, i, j) - Gm(it, is, ik, i, j)) / (2.0 * eps);
                ComplexType diff = DG_lr(it, is, ik, i, j) - DG_fd;
                max_error = std::max(max_error, std::abs(diff));
                max_DeltaG = std::max(max_DeltaG, std::abs(DG_lr(it, is, ik, i, j)));
              }
            }
          }
        }
      }
    }
    context->comm.barrier();

    // Broadcast max_error to all processes
    context->comm.broadcast_value(max_error, 0);
    context->comm.broadcast_value(max_DeltaG, 0);

    // Check that error is small (should be O(eps^2) for central difference)
    double rel_error = max_error / (max_DeltaG + 1e-15);
    INFO("Max |ΔG_lr - ΔG_fd| = " << max_error);
    INFO("Max |ΔG_lr| = " << max_DeltaG);
    INFO("Relative error = " << rel_error);
    CHECK(max_error < 1e-4);  // Should be much smaller due to O(eps^2)
    CHECK(rel_error < 1e-3);
    context->comm.barrier();
  }

  TEST_CASE("lr_dyson_dm_hermiticity_q0", "[methods_scf]") {
    /**
     * ΔDm must be Hermitian for a Hermitian ΔH0, whatever processor grid the LR
     * Dyson equation picks.
     *
     * The grid splits the band axes of ΔG (np_i > 1) whenever the (ω, k) pools
     * cannot absorb the rank count; each rank then owns only a block of ΔG, and
     * storing the full nbnd x nbnd product into that block instead of the rank's
     * own slice leaves ΔDm ~100% non-Hermitian. That path is NOT reachable here:
     * this system has nbnd = 8, which caps np_i, and nk_ibz = 8 absorbs every
     * leftover rank at unit-test rank counts (the grid logged below is always
     * (np,1,1,1,1)). Band-distributed coverage therefore needs a larger system
     * and a rank count the pools cannot absorb.
     */
    auto& context = utils::make_unit_test_mpi_context();
    std::string source_path = PROJECT_SOURCE_DIR;
    std::string filepath = source_path + "/tests/unit_test_files/pyscf/si_kp222_krhf/";

    double beta = 100;
    double wmax = 4.0;
    auto mf = mf::make_MF(context, mf::pyscf_source, filepath, "pyscf");
    imag_axes_ft::IAFT ft(beta, wmax, imag_axes_ft::ir_basis);

    int ns = mf.nspin();
    int nk = mf.nkpts_ibz();
    int nb = mf.nbnd();
    int nt = ft.nt_f();

    // Report the grid the LR Dyson equation will use, so it is visible whether
    // this rank count actually exercises the band-distributed path. Queried from
    // the helper the solver itself allocates with, not re-derived here.
    {
      auto [pg, bs] = lr_dyson_omega_pgrid(context->comm.size(), ft.nw_f(), nk, nb);
      app_log(2, "lr_dyson_dm_hermiticity_q0: nw = {}, nk_ibz = {}, nbnd = {}, "
                 "LR pgrid (w,s,k,i,j) = ({},{},{},{},{}), band bsize = {}",
              ft.nw_f(), nk, nb, pg[0], pg[1], pg[2], pg[3], pg[4], bs[3]);
    }

    simple_dyson dyson(std::addressof(mf), std::addressof(ft));
    nda::array<double, 1> q_vec{0.0, 0.0, 0.0};
    lr_dyson lr_dys(dyson, q_vec);

    sArray_t<Array_view_4D_t> F(math::shm::make_shared_array<Array_view_4D_t>(
        *context, {ns, nk, nb, nb}));
    sArray_t<Array_view_5D_t> G(math::shm::make_shared_array<Array_view_5D_t>(
        *context, {nt, ns, nk, nb, nb}));
    sArray_t<Array_view_5D_t> Sigma(math::shm::make_shared_array<Array_view_5D_t>(
        *context, {nt, ns, nk, nb, nb}));
    sArray_t<Array_view_4D_t> Dm(math::shm::make_shared_array<Array_view_4D_t>(
        *context, {ns, nk, nb, nb}));
    sArray_t<Array_view_4D_t> DeltaH0(math::shm::make_shared_array<Array_view_4D_t>(
        *context, {ns, nk, nb, nb}));
    sArray_t<Array_view_4D_t> DeltaF(math::shm::make_shared_array<Array_view_4D_t>(
        *context, {ns, nk, nb, nb}));
    sArray_t<Array_view_5D_t> DeltaG(math::shm::make_shared_array<Array_view_5D_t>(
        *context, {nt, ns, nk, nb, nb}));
    sArray_t<Array_view_4D_t> DeltaDm(math::shm::make_shared_array<Array_view_4D_t>(
        *context, {ns, nk, nb, nb}));

    hamilt::pseudopot psp(mf);
    hamilt::set_fock(mf, std::addressof(psp), F, true);

    if (context->node_comm.root()) Sigma.local()() = 0.0;
    context->comm.barrier();

    double mu = update_mu(0.0, dyson, mf, ft, F, Sigma);
    update_G(dyson, mf, ft, Dm, G, F, Sigma, mu, true);
    context->comm.barrier();

    // Hermitian ΔH0, zero ΔF
    if (context->node_comm.root()) {
      std::srand(42);
      auto DH0_loc = DeltaH0.local();
      DeltaF.local() = ComplexType(0.0);
      for (int is = 0; is < ns; ++is) {
        for (int ik = 0; ik < nk; ++ik) {
          for (int i = 0; i < nb; ++i) {
            for (int j = 0; j <= i; ++j) {
              double re = 0.01 * (std::rand() / double(RAND_MAX) - 0.5);
              double im = (i == j) ? 0.0 : 0.01 * (std::rand() / double(RAND_MAX) - 0.5);
              DH0_loc(is, ik, i, j) = ComplexType(re, im);
              DH0_loc(is, ik, j, i) = ComplexType(re, -im);
            }
          }
        }
      }
    }
    context->comm.barrier();

    auto sG_wskij = lr_precompute_G_omega(*context, G, ft);
    lr_dys.set_cached_G_omega(&sG_wskij);
    lr_dys.solve_lr_dyson(DeltaDm, DeltaH0, DeltaF,
                          static_cast<const sArray_t<Array_view_5D_t>*>(nullptr), false);
    lr_dys.materialize_DeltaG_tau(DeltaG);
    context->comm.barrier();

    double nonherm = 0.0, max_dDm = 0.0, dm_err = 0.0;
    if (context->node_comm.root()) {
      auto dDm = DeltaDm.local();
      for (int is = 0; is < ns; ++is) {
        for (int ik = 0; ik < nk; ++ik) {
          for (int i = 0; i < nb; ++i) {
            for (int j = 0; j < nb; ++j) {
              nonherm = std::max(nonherm,
                                 std::abs(dDm(is, ik, i, j) - std::conj(dDm(is, ik, j, i))));
              max_dDm = std::max(max_dDm, std::abs(dDm(is, ik, i, j)));
            }
          }
        }
      }

      // ΔDm is built from the *distributed* ΔG(τ), on a 4-D processor grid asserted
      // to reproduce the 5-D grid's rank->(k, i, j) block map. Recompute it from the
      // gathered ΔG instead, where no such correspondence is involved: a mismatch in
      // that map would land whole blocks on the wrong rank, which the reduction in
      // gather_to_shm would silently accept.
      nda::array<ComplexType, 4> DeltaDm_ref(ns, nk, nb, nb);
      ft.tau_to_beta(DeltaG.local(), DeltaDm_ref);
      DeltaDm_ref *= -1.0;
      auto Abs = nda::map([](ComplexType _x_) { return std::abs(_x_); });
      nda::array<RealType, 4> res_abs(ns, nk, nb, nb);
      res_abs = Abs(DeltaDm_ref - dDm);
      dm_err = nda::max_element(res_abs);
    }
    context->comm.broadcast_value(nonherm, 0);
    context->comm.broadcast_value(max_dDm, 0);
    context->comm.broadcast_value(dm_err, 0);

    INFO("max |ΔDm| = " << max_dDm);
    INFO("max |ΔDm - ΔDm^H| = " << nonherm);
    INFO("max |ΔDm + ΔG(β⁻)| = " << dm_err);
    CHECK(max_dDm > 1e-12);  // a zero ΔDm would make the check vacuous
    CHECK(nonherm / (max_dDm + 1e-15) < 1e-8);
    CHECK(dm_err / (max_dDm + 1e-15) < 1e-12);
    context->comm.barrier();
  }

  TEST_CASE("lr_dyson_omega_pgrid_no_band_split", "[methods_scf]") {
    /**
     * lr_dyson_omega_pgrid is a pure function, so the rank counts and system sizes
     * that matter in production can be checked without running on them — which is
     * the only way to cover them, since the band-split path is unreachable at
     * unit-test rank counts (see lr_dyson_dm_hermiticity_q0 above).
     *
     * A split band axis makes ΔΣ unusable and LR-GW aborts, so whenever some
     * nwpools*nkpools == nproc exists with nwpools <= nw and nkpools <= nk, the
     * grid must use it. nw = 40 at 960 ranks is the case that regressed: ω
     * saturates at 40, k takes 12 of the remaining 24, and 2 ranks hit the bands.
     */
    auto has_factorisation = [](long nproc, long nw, long nk) {
      for (long a = 1; a <= std::min(nproc, nw); ++a)
        if (nproc % a == 0 and nproc / a <= nk) return true;
      return false;
    };

    for (long nw : {40L, 72L}) {
      for (long nk : {8L, 64L}) {
        for (long nproc : {1L, 2L, 8L, 12L, 64L, 192L, 384L, 768L, 960L, 1920L}) {
          auto [pgrid, bsize] = methods::lr_dyson_omega_pgrid(nproc, nw, nk, 130);
          INFO("nproc=" << nproc << " nw=" << nw << " nk=" << nk << " -> ("
               << pgrid[0] << "," << pgrid[1] << "," << pgrid[2] << ","
               << pgrid[3] << "," << pgrid[4] << ")");

          // The grid must always cover exactly the rank count.
          CHECK(pgrid[0] * pgrid[1] * pgrid[2] * pgrid[3] * pgrid[4] == nproc);
          // Pools can never exceed the axis they divide.
          CHECK(pgrid[0] <= nw);
          CHECK(pgrid[2] <= nk);
          CHECK(bsize[3] >= 1);
          CHECK(bsize[4] >= 1);

          if (has_factorisation(nproc, nw, nk)) {
            CHECK(pgrid[3] * pgrid[4] == 1);   // bands undivided => LR-GW usable
          }
        }
      }
    }
  }

  TEST_CASE("lr_dyson_omega_pgrid_tie_break", "[methods_scf]") {
    /**
     * The cost model ceil(nw/nwpools) * ceil(nk/nkpools) cannot separate every
     * factorisation: nw = 40, nk = 64 at 960 ranks admits both (15, 64) and (40, 24)
     * at max per-rank work 3. The tie is broken towards more k pools so that the
     * ΔG(iω) redistribute, whose target grid maximises k pools first, keeps its
     * all-to-all inside one k column. Pinned here because nothing downstream fails
     * if it silently flips — it only gets slower.
     */
    auto [pgrid, bsize] = methods::lr_dyson_omega_pgrid(960, 40, 64, 130);
    CHECK(pgrid[0] == 15);
    CHECK(pgrid[2] == 64);
    CHECK(pgrid[3] * pgrid[4] == 1);

    auto [pgrid2, bsize2] = methods::lr_dyson_omega_pgrid(1920, 40, 64, 130);
    CHECK(pgrid2[0] == 30);
    CHECK(pgrid2[2] == 64);
    CHECK(pgrid2[3] * pgrid2[4] == 1);
  }

  TEST_CASE("lr_dyson_dN_dmu_q0", "[methods_scf]") {
    /**
     * dN/dmu is taken from the Delta_mu response of the LR solution:
     *   dN/dmu = spin * sum_k w_k Tr[S(k) . (dDeltaDm/dDelta_mu)(k)],
     * with dDeltaDm/dDelta_mu produced by one Dyson pass at X = -S.
     *
     * Validation: the independent closed form in frequency space,
     *   dN/dmu = spin * sum_k w_k Tr[S . (G.S.G)(beta-)],
     * evaluated here per k with a serial FT. The two share no code path: the
     * solver route goes through the distributed Dyson kernel, the distributed
     * w_to_tau/tau_to_beta and a gather; this one is a local loop.
     */
    auto& context = utils::make_unit_test_mpi_context();
    std::string source_path = PROJECT_SOURCE_DIR;
    std::string filepath = source_path + "/tests/unit_test_files/pyscf/si_kp222_krhf/";

    double beta = 100;
    double wmax = 4.0;
    auto mf = mf::make_MF(context, mf::pyscf_source, filepath, "pyscf");
    imag_axes_ft::IAFT ft(beta, wmax, imag_axes_ft::ir_basis);

    int ns = mf.nspin();
    int nk = mf.nkpts_ibz();
    int nb = mf.nbnd();
    int nt = ft.nt_f();
    int nw = ft.nw_f();

    simple_dyson dyson(std::addressof(mf), std::addressof(ft));
    nda::array<double, 1> q_vec{0.0, 0.0, 0.0};
    lr_dyson lr_dys(dyson, q_vec);

    sArray_t<Array_view_4D_t> F(math::shm::make_shared_array<Array_view_4D_t>(
        *context, {ns, nk, nb, nb}));
    sArray_t<Array_view_5D_t> G(math::shm::make_shared_array<Array_view_5D_t>(
        *context, {nt, ns, nk, nb, nb}));
    sArray_t<Array_view_5D_t> Sigma(math::shm::make_shared_array<Array_view_5D_t>(
        *context, {nt, ns, nk, nb, nb}));
    sArray_t<Array_view_4D_t> Dm(math::shm::make_shared_array<Array_view_4D_t>(
        *context, {ns, nk, nb, nb}));

    hamilt::pseudopot psp(mf);
    hamilt::set_fock(mf, std::addressof(psp), F, true);
    if (context->node_comm.root()) Sigma.local()() = 0.0;
    context->comm.barrier();

    double mu = update_mu(0.0, dyson, mf, ft, F, Sigma);
    update_G(dyson, mf, ft, Dm, G, F, Sigma, mu, true);
    context->comm.barrier();

    auto sG_wskij = lr_precompute_G_omega(*context, G, ft);
    lr_dys.set_cached_G_omega(&sG_wskij);

    // --- under test: dN/dmu from the Delta_mu response ---
    lr_dys.build_dmu_response();
    double dN_dmu = lr_dys.dN_dmu();

    // --- reference: Tr[S . (G.S.G)(beta-)], serial per (s, k) ---
    auto k_weight = mf.k_weight();
    auto S_loc = dyson.sS_skij().local();
    auto G_w = sG_wskij.local();
    double spin_factor = (ns == 1 && mf.npol() == 1) ? 2.0 : 1.0;
    decltype(nda::range::all) all;

    ComplexType dN_dmu_ref(0.0);
    nda::array<ComplexType, 3> GSG_w(nw, nb, nb);
    nda::array<ComplexType, 3> GSG_t(nt, nb, nb);
    nda::matrix<ComplexType> GSG_beta(nb, nb), tmp(nb, nb), buf(nb, nb);

    int rank = context->comm.rank();
    int size = context->comm.size();
    for (int i = rank; i < ns * nk; i += size) {
      int is = i / nk;
      int ik = i % nk;
      auto S_k = S_loc(is, ik, all, all);
      for (int n = 0; n < nw; ++n) {
        auto G_w_k = G_w(n, is, ik, all, all);
        auto GSG_w_k = GSG_w(n, all, all);
        nda::blas::gemm(ComplexType(1.0), S_k, G_w_k, ComplexType(0.0), tmp);
        nda::blas::gemm(ComplexType(1.0), G_w_k, tmp, ComplexType(0.0), GSG_w_k);
      }
      ft.w_to_tau(GSG_w, GSG_t, imag_axes_ft::fermion);
      ft.tau_to_beta(GSG_t, GSG_beta);
      nda::blas::gemm(ComplexType(1.0), S_k, GSG_beta, ComplexType(0.0), buf);
      dN_dmu_ref += k_weight(ik) * nda::trace(buf);
    }
    dN_dmu_ref *= spin_factor;
    dN_dmu_ref = context->comm.all_reduce_value(dN_dmu_ref);

    // Si at beta = 100 is gapped, so dN/dmu is small; guard against the test
    // passing by comparing two zeros, then compare relatively.
    REQUIRE(std::abs(dN_dmu_ref.real()) > 1e-8);
    double rel_err = std::abs(dN_dmu - dN_dmu_ref.real()) / std::abs(dN_dmu_ref.real());
    app_log(2, "[TEST] dN/dmu: response = {:.12e}, reference = {:.12e}, rel. err = {:.3e}",
            dN_dmu, dN_dmu_ref.real(), rel_err);
    REQUIRE(rel_err < 1e-10);

    // build_dmu_response() is cached: a second call must not rebuild or drift.
    lr_dys.build_dmu_response();
    VALUE_EQUAL(lr_dys.dN_dmu(), dN_dmu, 1e-14);
  }

} // lr_dyson_tests