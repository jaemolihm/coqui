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

  TEST_CASE("lr_dyson_dm_only_vs_full", "[methods_scf]") {
    /**
     * solve_lr_dyson_dm_only against the ΔDm of solve_lr_dyson on the same RHS:
     * (a) X = ΔH0 + ΔF, (b) X = ΔH0 + ΔF + ΔΣ, with the ΔΣ part handed in as the
     * source ΔDm of a solve_lr_dyson with ΔΣ alone, (c) both with fix_density at
     * q = Γ. The two share G(iω) but not the ΔG(iω) → ΔG(τ) → ΔDm route, so they
     * agree to round-off.
     *
     * With more ranks than (s, k) blocks (nk = 8 here, e.g. 16 ranks) some ranks
     * own no block.
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
    {
      auto [pg, bs] = lr_dyson_omega_pgrid(context->comm.size(), ft.nw_f(), nk, nb);
      app_log(2, "lr_dyson_dm_only_vs_full: nw = {}, nk_ibz = {}, nbnd = {}, "
                 "LR pgrid (w,s,k,i,j) = ({},{},{},{},{})",
              ft.nw_f(), nk, nb, pg[0], pg[1], pg[2], pg[3], pg[4]);
    }

    simple_dyson dyson(std::addressof(mf), std::addressof(ft));
    auto make4 = [&]() {
      return math::shm::make_shared_array<Array_view_4D_t>(*context, {ns, nk, nb, nb});
    };
    auto make5 = [&]() {
      return math::shm::make_shared_array<Array_view_5D_t>(*context, {nt, ns, nk, nb, nb});
    };
    sArray_t<Array_view_4D_t> F(make4()), Dm(make4()), DeltaH0(make4()), DeltaF(make4());
    sArray_t<Array_view_4D_t> Dm_full(make4()), Dm_only(make4()), Dm_src(make4());
    sArray_t<Array_view_4D_t> Zero(make4());
    Zero.set_zero();
    sArray_t<Array_view_5D_t> G(make5()), Sigma(make5()), DeltaSigma(make5());

    hamilt::pseudopot psp(mf);
    hamilt::set_fock(mf, std::addressof(psp), F, true);
    if (context->node_comm.root()) Sigma.local()() = 0.0;
    context->comm.barrier();
    double mu = update_mu(0.0, dyson, mf, ft, F, Sigma);
    update_G(dyson, mf, ft, Dm, G, F, Sigma, mu, true);
    context->comm.barrier();

    if (context->node_comm.root()) {
      std::srand(7);
      auto rnd = []() { return 0.01 * (std::rand() / double(RAND_MAX) - 0.5); };
      // Hermitian in the band indices, ΔΣ at every τ.
      auto fill = [&](auto&& A) {
        for (int is = 0; is < ns; ++is)
          for (int ik = 0; ik < nk; ++ik)
            for (int i = 0; i < nb; ++i)
              for (int j = 0; j <= i; ++j) {
                double re = rnd(), im = (i == j) ? 0.0 : rnd();
                A(is, ik, i, j) = ComplexType(re, im);
                A(is, ik, j, i) = ComplexType(re, -im);
              }
      };
      fill(DeltaH0.local());
      fill(DeltaF.local());
      for (int it = 0; it < nt; ++it) fill(DeltaSigma.local()(it, nda::ellipsis{}));
    }
    context->comm.barrier();

    auto sG_wskij = lr_precompute_G_omega(*context, G, ft);
    auto fro = [](auto const& a, auto const& b, double sb) {
      double d = 0.0;
      for (long n = 0; n < a.size(); ++n) d += std::norm(a.data()[n] - sb * b.data()[n]);
      return std::sqrt(d);
    };
    auto norm = [&](auto const& a) { return fro(a, a, 0.0); };
    auto rel = [&](auto const& a, auto const& b) { return fro(a, b, 1.0) / norm(b); };

    for (auto q : {std::array<double, 3>{0.0, 0.0, 0.0}, std::array<double, 3>{0.5, 0.0, 0.0}}) {
      nda::array<double, 1> q_vec{q[0], q[1], q[2]};
      lr_dyson lr_dys(dyson, q_vec);
      lr_dys.set_cached_G_omega(&sG_wskij);
      if (lr_dys.is_q_gamma()) lr_dys.build_dmu_response();

      for (bool fd : {false, true}) {
        if (fd and not lr_dys.is_q_gamma()) continue;
        for (bool with_sigma : {false, true}) {
          const sArray_t<Array_view_5D_t>* pS = with_sigma ? &DeltaSigma : nullptr;
          double mu_full = lr_dys.solve_lr_dyson(Dm_full, DeltaH0, DeltaF, pS, fd);
          if (with_sigma) lr_dys.solve_lr_dyson(Dm_src, Zero, Zero, &DeltaSigma, false);
          double mu_only = lr_dys.solve_lr_dyson_dm_only(Dm_only, DeltaH0, DeltaF,
                                                         with_sigma ? &Dm_src : nullptr, fd);
          context->comm.barrier();
          double err = rel(Dm_only.local(), Dm_full.local());
          app_log(2, "[TEST] dm_only vs full: q = ({},{},{}) fix_density = {} ΔΣ = {}: "
                     "rel ΔDm = {:.3e}, Δμ = {:.6e}, |ΔΔμ| = {:.3e}",
                  q[0], q[1], q[2], fd, with_sigma, err, mu_full, std::abs(mu_only - mu_full));
          INFO("q = (" << q[0] << "," << q[1] << "," << q[2] << ") fd = " << fd
               << " sigma = " << with_sigma);
          CHECK(norm(Dm_full.local()) > 1e-8);
          CHECK(err < 1e-12);
          // Δμ = −ΔN(0)/(dN/dμ), and ΔN(0) is a small trace of ΔDm: compare in ΔN,
          // against the size of ΔDm.
          if (fd)
            CHECK(std::abs(mu_only - mu_full) * std::abs(lr_dys.dN_dmu()) <
                  1e-12 * norm(Dm_full.local()));
        }
      }
    }
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

  // ---- Imaginary-time LR Dyson kernel (dyson_no_inversion) ----

  namespace {

    // sqrt(Σ |a - sb·b|²) over two arrays of equal shape
    template<typename A_t, typename B_t>
    double fro_diff(A_t const& a_in, B_t const& b_in, double sb) {
      auto a = nda::make_regular(a_in);
      auto b = nda::make_regular(b_in);
      double d = 0.0;
      for (long n = 0; n < a.size(); ++n) d += std::norm(a.data()[n] - sb * b.data()[n]);
      return std::sqrt(d);
    }
    template<typename A_t, typename B_t>
    double rel_diff(A_t const& a, B_t const& b) { return fro_diff(a, b, 1.0) / fro_diff(b, b, 0.0); }

    // Random Hermitian (nb x nb) block per (s, k) of a rank-4 shared array, written by the node root
    template<typename Context_t, typename Array_t>
    void fill_hermitian_skij(Context_t& context, Array_t& A, int seed, double amp) {
      if (context->node_comm.root()) {
        std::srand(seed);
        auto rnd = [&]() { return amp * (std::rand() / double(RAND_MAX) - 0.5); };
        auto [ns, nk, nb, nb2] = A.shape();
        for (long is = 0; is < ns; ++is)
          for (long ik = 0; ik < nk; ++ik)
            for (long i = 0; i < nb; ++i)
              for (long j = 0; j <= i; ++j) {
                double re = rnd(), im = (i == j) ? 0.0 : rnd();
                A.local()(is, ik, i, j) = ComplexType(re, im);
                A.local()(is, ik, j, i) = ComplexType(re, -im);
              }
      }
      context->comm.barrier();
    }

    // Weak-Σ ground-state fixture of dyson_no_inversion_picard: F the Fock matrix of the mean field, Σ(τ) = 5e-4 G_HF(τ)
    // with G_HF the Σ = 0 Green's function at its own μ; G_hf holds G_HF on return. Returns μ of the Σ ≠ 0 problem.
    template<typename Context_t, typename MF_t, typename FT_t>
    double make_weak_sigma_gs(Context_t& context, MF_t& mf, FT_t& ft, simple_dyson& dyson_node,
                              sArray_t<Array_view_4D_t>& F, sArray_t<Array_view_5D_t>& G_hf,
                              sArray_t<Array_view_5D_t>& Sigma) {
      hamilt::pseudopot psp(mf);
      hamilt::set_fock(mf, std::addressof(psp), F, true);
      if (context->node_comm.root()) Sigma.local()() = 0.0;
      context->comm.barrier();
      double mu = update_mu(0.0, dyson_node, mf, ft, F, Sigma);
      dyson_node.solve_dyson(G_hf, F, Sigma, mu);
      Sigma.node_sync();
      if (context->node_comm.root()) Sigma.local() = 0.0005 * G_hf.local();
      Sigma.node_sync();
      return update_mu(mu, dyson_node, mf, ft, F, Sigma);
    }

    // Sweeps of a flagged simple_dyson at fixed (F, Σ, μ), from the G in `G`, until max|G_new - G_prev| < tol (at most
    // max_sweeps); returns the last change. The first call of a flagged solver is a node solve, so call it once before.
    template<typename Context_t>
    double sweep_to_fixed_point(Context_t& context, simple_dyson& dyson_tau, sArray_t<Array_view_5D_t>& G,
                                sArray_t<Array_view_4D_t> const& F, sArray_t<Array_view_5D_t> const& Sigma,
                                double mu, double tol = 1e-14, int max_sweeps = 100) {
      double change = 0.0;
      for (int it = 0; it < max_sweeps; ++it) {
        nda::array<ComplexType, 5> G_prev = G.local();
        context->comm.barrier();
        dyson_tau.solve_dyson(G, F, Sigma, mu);
        context->comm.barrier();
        change = 0.0;
        for (long n = 0; n < G_prev.size(); ++n)
          change = std::max(change, std::abs(G.local().data()[n] - G_prev.data()[n]));
        if (change < tol) break;
      }
      return change;
    }

    // Matsubara-node G(iω) = [(iω + μ) S - H0 - F - Σ(iω)]⁻¹ of the ground state, formed pointwise (not through
    // G(τ) → G(iω)), in a shared array (nw, ns, nk, nb, nb)
    template<typename Context_t, typename FT_t>
    auto pointwise_G_omega(Context_t& context, FT_t& ft, simple_dyson& dyson, sArray_t<Array_view_4D_t> const& F,
                           sArray_t<Array_view_5D_t> const& Sigma, double mu) {
      decltype(nda::range::all) all;
      auto [nt, ns, nk, nb, nb2] = Sigma.shape();
      auto G_w = math::shm::make_shared_array<Array_view_5D_t>(*context, {ft.nw_f(), ns, nk, nb, nb});
      if (context->node_comm.root()) {
        nda::array<ComplexType, 3> Sigma_t(nt, nb, nb);
        nda::matrix<ComplexType> Sigma_w(nb, nb), M(nb, nb);
        for (long is = 0; is < ns; ++is)
          for (long ik = 0; ik < nk; ++ik) {
            Sigma_t = Sigma.local()(all, is, ik, all, all);
            for (long n = 0; n < ft.nw_f(); ++n) {
              ft.tau_to_w(Sigma_t, Sigma_w, imag_axes_ft::fermion, n);
              ComplexType omega_mu = ft.omega(ft.wn_mesh()(n)) + mu;
              M = omega_mu * dyson.sS_skij().local()(is, ik, all, all) - dyson.H0()(is, ik, all, all)
                  - F.local()(is, ik, all, all) - Sigma_w;
              G_w.local()(n, is, ik, all, all) = nda::inverse(M);
            }
          }
      }
      context->comm.barrier();
      return G_w;
    }

  } // anonymous namespace

  TEST_CASE("lr_dyson_tau_fd", "[methods_scf]") {
    // C++ lr_dyson imaginary-time kernel vs central-difference FD (Richardson-extrapolated, eps and eps/2) of the converged
    // imaginary-time Picard map at frozen (F, Sigma, mu) (QE LiH, DLR beta = 1000, wmax = 30, high), for (a) a static
    // Hermitian dH0 and (b) a smooth dSigma(tau); the Matsubara-node kernel on the same G(tau) must miss the
    // derivative of the tau map by more than 10x what the tau kernel misses (the two Jacobians differ at the DLR-grid
    // floor, ~5e-9 on this grid). Smooth perturbations only: along the rough Twt directions the sweep does not contract.
    auto& context = utils::make_unit_test_mpi_context();
    std::string source_path = PROJECT_SOURCE_DIR;
    std::string filepath = source_path + "/tests/unit_test_files/qe/lih_kp222_nbnd16/";
    double beta = 1000;
    double wmax = 30.0;
    auto mf = mf::make_MF(context, mf::qe_source, filepath, "pwscf");
    imag_axes_ft::IAFT ft(beta, wmax, imag_axes_ft::dlr_basis, "high");
    long nt = ft.nt_f(), ns = mf.nspin(), nk = mf.nkpts_ibz(), nb = mf.nbnd();
    auto make4 = [&]() { return math::shm::make_shared_array<Array_view_4D_t>(*context, {ns, nk, nb, nb}); };
    auto make5 = [&]() { return math::shm::make_shared_array<Array_view_5D_t>(*context, {nt, ns, nk, nb, nb}); };
    sArray_t<Array_view_4D_t> F(make4()), Zero(make4()), DeltaH0(make4()), Dm_lr(make4());
    sArray_t<Array_view_5D_t> G_hf(make5()), G_tau(make5()), Sigma(make5()), DeltaSigma(make5()), DeltaG_lr(make5());
    Zero.set_zero();

    simple_dyson dyson_node(std::addressof(mf), std::addressof(ft), 1e-9, "bisection", false);
    simple_dyson dyson_tau(std::addressof(mf), std::addressof(ft), 1e-9, "bisection", true);
    double mu = make_weak_sigma_gs(context, mf, ft, dyson_node, F, G_hf, Sigma);

    // G(tau): the first call of the flagged solver is the node solve, then sweeps to the fixed point
    dyson_tau.solve_dyson(G_tau, F, Sigma, mu);
    double change = sweep_to_fixed_point(context, dyson_tau, G_tau, F, Sigma, mu);
    app_log(1, "[TEST] lr_dyson_tau_fd: GS sweeps converged to max|dG| = {:.3e}", change);

    // Perturbations: dH0 random Hermitian (O(1)), dSigma(tau) = 5e-4 (H G_HF + G_HF H) with H random Hermitian
    fill_hermitian_skij(context, DeltaH0, 11, 1.0);
    {
      auto H = make4();
      fill_hermitian_skij(context, H, 12, 1.0);
      if (context->node_comm.root()) {
        decltype(nda::range::all) all;
        for (long t = 0; t < nt; ++t)
          for (long is = 0; is < ns; ++is)
            for (long ik = 0; ik < nk; ++ik) {
              nda::matrix<ComplexType> Hk = H.local()(is, ik, all, all), Gk = G_hf.local()(t, is, ik, all, all);
              DeltaSigma.local()(t, is, ik, all, all) = 0.0005 * (Hk * Gk + Gk * Hk);
            }
      }
      context->comm.barrier();
    }

    // FD of the tau map: G_tau[F + eps dH0, Sigma + eps dSigma], by sweeps from G_tau
    auto G_pm = make5(), Sigma_pm = make5();
    auto F_pm = make4();
    auto fd_derivative = [&](bool perturb_H0, double eps) {
      nda::array<ComplexType, 5> G_plus, G_minus;
      for (double sign : {+1.0, -1.0}) {
        F_pm.node_sync();
        if (context->node_comm.root()) {
          F_pm.local() = F.local() + (perturb_H0 ? sign * eps : 0.0) * DeltaH0.local();
          Sigma_pm.local() = Sigma.local() + (perturb_H0 ? 0.0 : sign * eps) * DeltaSigma.local();
          G_pm.local() = G_tau.local();
        }
        F_pm.node_sync();
        double chg = sweep_to_fixed_point(context, dyson_tau, G_pm, F_pm, Sigma_pm, mu);
        INFO("FD sweeps: perturb_H0 = " << perturb_H0 << " eps = " << eps << " sign = " << sign);
        CHECK(chg < 1e-13);
        (sign > 0 ? G_plus : G_minus) = G_pm.local();
      }
      return nda::array<ComplexType, 5>((G_plus - G_minus) / (2.0 * eps));
    };

    // The three LR arms: tau kernel (flagged dyson, G(tau)), node kernel on the same G(tau) (G(iw) = tau_to_w G(tau))
    lr_dyson lr_tau(dyson_tau, nda::array<double, 1>{0.0, 0.0, 0.0});
    lr_tau.set_G_tau(&G_tau);
    lr_dyson lr_node(dyson_node, nda::array<double, 1>{0.0, 0.0, 0.0});
    auto sG_wskij = lr_precompute_G_omega(*context, G_tau, ft);
    lr_node.set_cached_G_omega(&sG_wskij);
    REQUIRE(lr_tau.tau_kernel());
    REQUIRE(!lr_node.tau_kernel());

    for (bool perturb_H0 : {true, false}) {
      // Richardson D = (4 D(eps/2) - D(eps)) / 3 removes the eps^2 term of the central difference; eps is small enough
      // that the eps^4 term is below the round-off floor of the sweeps (dH0: eps ||dH0|| ~ 1e-3; dSigma: ~1% of Sigma)
      double eps = perturb_H0 ? 3e-4 : 0.01;
      auto D1 = fd_derivative(perturb_H0, eps), D2 = fd_derivative(perturb_H0, 0.5 * eps);
      nda::array<ComplexType, 5> fd_G = (4.0 * D2 - D1) / 3.0;
      nda::array<ComplexType, 4> fd_Dm(ns, nk, nb, nb);
      {
        nda::array<ComplexType, 4> beta_G(ns, nk, nb, nb);
        ft.tau_to_beta(fd_G, beta_G);
        fd_Dm = -beta_G;
      }

      double err_G[2], err_Dm[2];
      int arm = 0;
      for (lr_dyson* lr : {&lr_tau, &lr_node}) {
        lr->solve_lr_dyson(Dm_lr, perturb_H0 ? DeltaH0 : Zero, Zero, perturb_H0 ? nullptr : &DeltaSigma, false);
        lr->materialize_DeltaG_tau(DeltaG_lr);
        context->comm.barrier();
        err_G[arm] = rel_diff(DeltaG_lr.local(), fd_G);
        err_Dm[arm] = rel_diff(Dm_lr.local(), fd_Dm);
        ++arm;
      }
      app_log(1, "[TEST] lr_dyson_tau_fd: {} FD vs tau kernel: dG {:.3e}, dDm {:.3e}; vs node kernel: dG {:.3e}, dDm {:.3e}",
              perturb_H0 ? "dH0" : "dSigma", err_G[0], err_Dm[0], err_G[1], err_Dm[1]);
      INFO("perturbation: " << (perturb_H0 ? "dH0" : "dSigma"));
      // tol: FD floor of the converged sweeps (1.6e-11 to 2e-11 measured); separation: 24x (dH0 dG) to 500x (dSigma dDm) measured
      CHECK(err_G[0] < 1e-10);
      CHECK(err_Dm[0] < 1e-10);
      CHECK(err_G[1] > 10 * err_G[0]);
      CHECK(err_Dm[1] > 10 * err_Dm[0]);
    }
    context->comm.barrier();
  }

  TEST_CASE("simple_dyson_picard_sweep_change", "[methods_scf]") {
    // simple_dyson::picard_sweep_change (the ground-state sweep residual logged by run_lr) vs the change of an actual
    // solve_dyson sweep from the same G (QE LiH, DLR beta = 100, wmax = 4, high): bitwise equal for a perturbed G,
    // G left unchanged, and below 1e-13 at the converged fixed point.
    auto& context = utils::make_unit_test_mpi_context();
    std::string source_path = PROJECT_SOURCE_DIR;
    std::string filepath = source_path + "/tests/unit_test_files/qe/lih_kp222_nbnd16/";
    auto mf = mf::make_MF(context, mf::qe_source, filepath, "pwscf");
    imag_axes_ft::IAFT ft(100.0, 4.0, imag_axes_ft::dlr_basis, "high");
    long nt = ft.nt_f(), ns = mf.nspin(), nk = mf.nkpts_ibz(), nb = mf.nbnd();
    auto make4 = [&]() { return math::shm::make_shared_array<Array_view_4D_t>(*context, {ns, nk, nb, nb}); };
    auto make5 = [&]() { return math::shm::make_shared_array<Array_view_5D_t>(*context, {nt, ns, nk, nb, nb}); };
    sArray_t<Array_view_4D_t> F(make4());
    sArray_t<Array_view_5D_t> G_hf(make5()), G_tau(make5()), G_check(make5()), Sigma(make5());

    simple_dyson dyson_node(std::addressof(mf), std::addressof(ft), 1e-9, "bisection", false);
    simple_dyson dyson_tau(std::addressof(mf), std::addressof(ft), 1e-9, "bisection", true);
    double mu = make_weak_sigma_gs(context, mf, ft, dyson_node, F, G_hf, Sigma);

    // G: node solve (first call of the flagged solver), scaled by 1.01 so that it is off the fixed point
    dyson_tau.solve_dyson(G_tau, F, Sigma, mu);
    if (context->node_comm.root()) G_tau.local() *= 1.01;
    G_tau.node_sync();
    context->comm.barrier();
    nda::array<ComplexType, 5> G_before = G_tau.local();

    // Reference: the change of a real sweep from the same G (on a copy)
    if (context->node_comm.root()) G_check.local() = G_tau.local();
    G_check.node_sync();
    context->comm.barrier();
    dyson_tau.solve_dyson(G_check, F, Sigma, mu);
    double change_ref = 0.0;
    for (long n = 0; n < G_before.size(); ++n)
      change_ref = std::max(change_ref, std::abs(G_check.local().data()[n] - G_before.data()[n]));

    double change = dyson_tau.picard_sweep_change(G_tau, F, Sigma, mu);
    app_log(1, "[TEST] picard_sweep_change: {:.6e} (real sweep {:.6e})", change, change_ref);
    CHECK(change > 1e-6);
    CHECK(change == change_ref);
    double G_moved = 0.0;
    for (long n = 0; n < G_before.size(); ++n)
      G_moved = std::max(G_moved, std::abs(G_tau.local().data()[n] - G_before.data()[n]));
    CHECK(G_moved == 0.0);

    // At the fixed point the change is below the sweep tolerance
    sweep_to_fixed_point(context, dyson_tau, G_tau, F, Sigma, mu);
    CHECK(dyson_tau.picard_sweep_change(G_tau, F, Sigma, mu) < 1e-13);
    context->comm.barrier();
  }

  // The main ctest entry runs this case at CTEST_NPROC ranks; test_methods_scf_lr_dyson_tau_np16 (registered when
  // MPIEXEC_MAX_NUMPROCS >= 16) runs it at 16 ranks, where the 8 k-points leave 2 ranks per k and the band axes of the
  // lr_dyson_tau_pgrid are split (np_i * np_j > 1).
  TEST_CASE("lr_dyson_tau_vs_node", "[methods_scf]") {
    // C++ lr_dyson imaginary-time kernel vs the Matsubara-node kernel on the node ground state (QE LiH, DLR beta = 100,
    // wmax = 4, high, where the two Jacobians coincide to ~1e-13): (a) dG and dDm for q = Gamma and q != Gamma, with and
    // without dSigma; (b) the tau dm_only pass vs the tau full pass; (c) the density functional of fix_density at Gamma
    // (dN/dmu and the Delta_mu of a dSigma-carrying perturbation) vs the node kernel on the pointwise G(iw) of the
    // ground state.
    auto& context = utils::make_unit_test_mpi_context();
    std::string source_path = PROJECT_SOURCE_DIR;
    std::string filepath = source_path + "/tests/unit_test_files/qe/lih_kp222_nbnd16/";
    double beta = 100;
    double wmax = 4.0;
    auto mf = mf::make_MF(context, mf::qe_source, filepath, "pwscf");
    imag_axes_ft::IAFT ft(beta, wmax, imag_axes_ft::dlr_basis, "high");
    long nt = ft.nt_f(), ns = mf.nspin(), nk = mf.nkpts_ibz(), nb = mf.nbnd();
    auto make4 = [&]() { return math::shm::make_shared_array<Array_view_4D_t>(*context, {ns, nk, nb, nb}); };
    auto make5 = [&]() { return math::shm::make_shared_array<Array_view_5D_t>(*context, {nt, ns, nk, nb, nb}); };
    sArray_t<Array_view_4D_t> F(make4()), Zero(make4()), DeltaH0(make4()), DeltaF(make4());
    sArray_t<Array_view_4D_t> Dm_tau(make4()), Dm_node(make4()), Dm_only(make4()), Dm_src(make4());
    sArray_t<Array_view_5D_t> G_hf(make5()), G_node(make5()), Sigma(make5()), DeltaSigma(make5());
    sArray_t<Array_view_5D_t> DeltaG_tau(make5()), DeltaG_node(make5());
    Zero.set_zero();

    simple_dyson dyson_node(std::addressof(mf), std::addressof(ft), 1e-9, "bisection", false);
    simple_dyson dyson_tau(std::addressof(mf), std::addressof(ft), 1e-9, "bisection", true);
    double mu = make_weak_sigma_gs(context, mf, ft, dyson_node, F, G_hf, Sigma);
    dyson_node.solve_dyson(G_node, F, Sigma, mu);
    auto G_w = pointwise_G_omega(context, ft, dyson_node, F, Sigma, mu);

    fill_hermitian_skij(context, DeltaH0, 21, 0.2);
    fill_hermitian_skij(context, DeltaF, 22, 0.2);
    {
      auto H = make4();
      fill_hermitian_skij(context, H, 23, 1.0);
      if (context->node_comm.root()) {
        decltype(nda::range::all) all;
        for (long t = 0; t < nt; ++t)
          for (long is = 0; is < ns; ++is)
            for (long ik = 0; ik < nk; ++ik) {
              nda::matrix<ComplexType> Hk = H.local()(is, ik, all, all), Gk = G_hf.local()(t, is, ik, all, all);
              DeltaSigma.local()(t, is, ik, all, all) = 0.5 * (Hk * Gk + Gk * Hk);  // O(1) so that its share of dN is not small
            }
      }
      context->comm.barrier();
    }

    for (auto q : {std::array<double, 3>{0.0, 0.0, 0.0}, std::array<double, 3>{0.5, 0.0, 0.0}}) {
      nda::array<double, 1> q_vec{q[0], q[1], q[2]};
      lr_dyson lr_tau(dyson_tau, q_vec);
      lr_tau.set_G_tau(&G_node);
      lr_dyson lr_node(dyson_node, q_vec);
      lr_node.set_cached_G_omega(&G_w);
      if (lr_tau.is_q_gamma()) {
        lr_tau.build_dmu_response(&F, &Sigma, mu);
        lr_node.build_dmu_response();
      }

      for (bool with_sigma : {false, true}) {
        const sArray_t<Array_view_5D_t>* pS = with_sigma ? &DeltaSigma : nullptr;
        INFO("q = (" << q[0] << "," << q[1] << "," << q[2] << ") sigma = " << with_sigma);

        // (a) tau kernel vs node kernel: dG(tau), dDm
        lr_tau.solve_lr_dyson(Dm_tau, DeltaH0, DeltaF, pS, false);
        lr_tau.materialize_DeltaG_tau(DeltaG_tau);
        lr_node.solve_lr_dyson(Dm_node, DeltaH0, DeltaF, pS, false);
        lr_node.materialize_DeltaG_tau(DeltaG_node);
        context->comm.barrier();
        double err_G = rel_diff(DeltaG_tau.local(), DeltaG_node.local());
        double err_Dm = rel_diff(Dm_tau.local(), Dm_node.local());
        app_log(2, "[TEST] tau vs node kernel: q = ({},{},{}) dSigma = {}: rel dG = {:.3e}, rel dDm = {:.3e}",
                q[0], q[1], q[2], with_sigma, err_G, err_Dm);
        CHECK(fro_diff(Dm_node.local(), Dm_node.local(), 0.0) > 1e-8);
        // tol: the two kernels differ at the DLR floor of the tau <-> iw transforms (6e-12 to 4.3e-11 measured, the larger
        // with the O(1) dSigma, whose tau_to_w feeds the node kernel)
        CHECK(err_G < 1e-10);
        CHECK(err_Dm < 1e-10);

        // (b) tau dm_only vs tau full, the dSigma part handed in as the source dDm of a dSigma-only pass
        for (bool fd : {false, true}) {
          if (fd and not lr_tau.is_q_gamma()) continue;
          double mu_full = lr_tau.solve_lr_dyson(Dm_tau, DeltaH0, DeltaF, pS, fd);
          if (with_sigma) lr_tau.solve_lr_dyson(Dm_src, Zero, Zero, &DeltaSigma, false);
          double mu_only = lr_tau.solve_lr_dyson_dm_only(Dm_only, DeltaH0, DeltaF, with_sigma ? &Dm_src : nullptr, fd, pS);
          context->comm.barrier();
          double err_only = rel_diff(Dm_only.local(), Dm_tau.local());
          app_log(2, "[TEST] tau dm_only vs full: q = ({},{},{}) dSigma = {} fix_density = {}: rel dDm = {:.3e}, |dmu diff| = {:.3e}",
                  q[0], q[1], q[2], with_sigma, fd, err_only, std::abs(mu_only - mu_full));
          CHECK(err_only < 1e-13);  // summation order only (5.6e-14 measured with dSigma, 0 without)
          if (fd) CHECK(std::abs(mu_only - mu_full) * std::abs(lr_tau.dN_dmu()) < 1e-13 * fro_diff(Dm_tau.local(), Dm_tau.local(), 0.0));
        }

        // (c) fix_density at Gamma: node functional of the tau kernel vs the node kernel on the pointwise G(iw)
        if (lr_tau.is_q_gamma()) {
          double dN_rel = std::abs(lr_tau.dN_dmu() - lr_node.dN_dmu()) / std::abs(lr_node.dN_dmu());
          double mu_tau = lr_tau.solve_lr_dyson(Dm_tau, DeltaH0, DeltaF, pS, true);
          double mu_node = lr_node.solve_lr_dyson(Dm_node, DeltaH0, DeltaF, pS, true);
          context->comm.barrier();
          double dmu_dN = std::abs(mu_tau - mu_node) * std::abs(lr_node.dN_dmu());
          app_log(2, "[TEST] fix_density: dN/dmu tau {:.12e} node {:.12e} (rel {:.3e}); dmu tau {:.6e} node {:.6e}, "
                     "|ddmu| dN/dmu = {:.3e} (|dDm| = {:.3e})",
                  lr_tau.dN_dmu(), lr_node.dN_dmu(), dN_rel, mu_tau, mu_node, dmu_dN, fro_diff(Dm_node.local(), Dm_node.local(), 0.0));
          CHECK(std::abs(lr_node.dN_dmu()) > 1e-8);
          CHECK(dN_rel < 1e-12);
          CHECK(dmu_dN < 1e-12 * fro_diff(Dm_node.local(), Dm_node.local(), 0.0));
          CHECK(rel_diff(Dm_tau.local(), Dm_node.local()) < 1e-10);
        }
      }
    }
    context->comm.barrier();
  }

} // lr_dyson_tests