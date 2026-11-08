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
#include "hamiltonian/pseudo/pseudopot.h"

namespace bdft_tests {

  using utils::mpi_context_t;
  using utils::VALUE_EQUAL;
  using utils::ARRAY_EQUAL;
  namespace mpi3 = boost::mpi3;
  using namespace methods;

  // L2 norm of all elements of a node-shared array's local view
  auto l2_norm = [](auto const &a) {
    auto a_reg = nda::make_regular(a);
    double sum = 0.0;
    for (long i = 0; i < a_reg.size(); ++i) sum += std::norm(a_reg.data()[i]);
    return std::sqrt(sum);
  };

  TEST_CASE("dyson_init", "[methods_scf]") {
    auto& context = utils::make_unit_test_mpi_context();
    std::string source_path = PROJECT_SOURCE_DIR;
    std::string filepath = source_path + "/tests/unit_test_files/pyscf/si_kp222_krhf/";

    double beta = 1000;
    double wmax = 12.0;
    auto mf = mf::make_MF(context, mf::pyscf_source, filepath, "pyscf");
    imag_axes_ft::IAFT ft(beta, wmax, imag_axes_ft::ir_basis, "high");
    simple_dyson dyson(std::addressof(mf), std::addressof(ft), 1e-9, "bisection");
  }

  TEST_CASE("dyson", "[methods_scf]") {
    auto& context = utils::make_unit_test_mpi_context();
    std::string source_path = PROJECT_SOURCE_DIR;
    std::string filepath = source_path + "/tests/unit_test_files/pyscf/si_kp222_krhf/";
    double beta = 1000;
    double wmax = 1.2;
    auto mf = mf::make_MF(context, mf::pyscf_source, filepath, "pyscf");
    imag_axes_ft::IAFT ft(beta, wmax, imag_axes_ft::ir_basis, "high");
    hamilt::pseudopot psp(mf);
    sArray_t<Array_view_4D_t> F(math::shm::make_shared_array<Array_view_4D_t>(
        *context, {mf.nspin(), mf.nkpts(), mf.nbnd(), mf.nbnd()}));
    sArray_t<Array_view_4D_t> Dm(math::shm::make_shared_array<Array_view_4D_t>(
        *context, {mf.nspin(), mf.nkpts(), mf.nbnd(), mf.nbnd()}));
    sArray_t<Array_view_5D_t> G(math::shm::make_shared_array<Array_view_5D_t>(
        *context, {ft.nt_f(), mf.nspin(), mf.nkpts(), mf.nbnd(), mf.nbnd()}));
    sArray_t<Array_view_5D_t> Sigma(math::shm::make_shared_array<Array_view_5D_t>(
        *context, {ft.nt_f(), mf.nspin(), mf.nkpts(), mf.nbnd(), mf.nbnd()}));
    hamilt::set_fock(mf, std::addressof(psp), F, true);

    simple_dyson dyson(std::addressof(mf), std::addressof(ft), 1e-9, "bisection");
    context->comm.barrier();

    double mu = update_mu(0.0, dyson, mf, ft, F, Sigma);
    CHECK(mu == Approx(0.175));
    context->comm.barrier();

    update_G(dyson, mf, ft, Dm, G, F, Sigma, mu, true);// keep mu constant
    context->comm.barrier();

    /**
     * References are obtained from PySCF in the zero-temperature formalism.
     * The accuracy is roughly 1e-8 when comparing with Dyson equation at beta = 1000
     **/
    {
      std::string source_path_ = PROJECT_SOURCE_DIR;
      std::string filename = source_path_ + "/tests/unit_test_files/pyscf/si_kp222_krhf/hf_Gw_Gt_beta1000_wmax1.2_high.h5";
      h5::file file(filename, 'r');
      h5::group grp(file);

      nda::array<std::complex<double>, 5> G_tskij_ref;
      nda::array<std::complex<double>, 4> Dm_skij_ref;
      nda::h5_read(grp, "Gt", G_tskij_ref);
      nda::h5_read(grp, "Dm", Dm_skij_ref);

      ARRAY_EQUAL(G.local(), G_tskij_ref, 1e-8);
      ARRAY_EQUAL(Dm.local(), Dm_skij_ref, 1e-8);
    }
    auto k_weight = mf.k_weight();
    auto [e_1e, e_hf] = eval_hf_energy(Dm, F, dyson.sH0_skij(), k_weight, false);
    VALUE_EQUAL(e_1e+e_hf, 0.8730537612681228, 1e-8);

    double e_corr = eval_corr_energy(context->comm, ft, G, Sigma, k_weight);
    VALUE_EQUAL(e_corr, 0.0, 1e-12);
    context->comm.barrier();
  }


  // dyson_no_inversion = true (imaginary-time Picard sweeps, DLR basis) vs the Matsubara-node Dyson solve, for
  // Sigma(tau) = 5e-4 G_HF(tau) at fixed (F, Sigma, mu); the sweeps start from the Sigma = 0 Green's function
  // (QE LiH, S = 1) and must converge to the node-Dyson G (relative L2 over all of G). Sigma is weak because
  // on this small-gap fixture the sweeps converge geometrically only for a weak Sigma.
  // The sweep tasks are (s, k, column block of G). This fixture has 8 (s,k) pairs, so the column-block split only
  // runs with more than 8 ranks. The main ctest entry runs this case at CTEST_NPROC ranks (tasks split over (s,k)
  // only); the test_methods_scf_dyson_picard_np9 entry (registered when CTEST_NPROC >= 9) runs it at 9 ranks, with
  // 2 column blocks per (s,k).
  TEST_CASE("dyson_no_inversion_picard", "[methods_scf]") {
    auto& context = utils::make_unit_test_mpi_context();
    std::string source_path = PROJECT_SOURCE_DIR;
    std::string filepath = source_path + "/tests/unit_test_files/qe/lih_kp222_nbnd16/";
    double beta = 100;
    double wmax = 4.0;
    auto mf = mf::make_MF(context, mf::qe_source, filepath, "pwscf");
    imag_axes_ft::IAFT ft(beta, wmax, imag_axes_ft::dlr_basis, "high");
    hamilt::pseudopot psp(mf);
    long nt = ft.nt_f(), ns = mf.nspin(), nk = mf.nkpts_ibz(), nb = mf.nbnd();
    sArray_t<Array_view_4D_t> F(math::shm::make_shared_array<Array_view_4D_t>(*context, {ns, nk, nb, nb}));
    sArray_t<Array_view_5D_t> G_hf(math::shm::make_shared_array<Array_view_5D_t>(*context, {nt, ns, nk, nb, nb}));
    sArray_t<Array_view_5D_t> G_node(math::shm::make_shared_array<Array_view_5D_t>(*context, {nt, ns, nk, nb, nb}));
    sArray_t<Array_view_5D_t> G_tau(math::shm::make_shared_array<Array_view_5D_t>(*context, {nt, ns, nk, nb, nb}));
    sArray_t<Array_view_5D_t> Sigma(math::shm::make_shared_array<Array_view_5D_t>(*context, {nt, ns, nk, nb, nb}));
    hamilt::set_fock(mf, std::addressof(psp), F, true);

    simple_dyson dyson_node(std::addressof(mf), std::addressof(ft), 1e-9, "bisection", false);
    simple_dyson dyson_tau(std::addressof(mf), std::addressof(ft), 1e-9, "bisection", true);

    // Sigma(tau) = 0.0005 G_HF(tau) with G_HF the Hartree-Fock Green's function at the mu of the Sigma = 0 problem
    double mu = update_mu(0.0, dyson_node, mf, ft, F, Sigma);
    dyson_node.solve_dyson(G_hf, F, Sigma, mu);
    Sigma.node_sync();  // all ranks are done reading Sigma before the node root overwrites it
    if (context->node_comm.root()) Sigma.local() = 0.0005 * G_hf.local();
    Sigma.node_sync();
    mu = update_mu(mu, dyson_node, mf, ft, F, Sigma);

    // Reference: Matsubara-node Dyson solve
    dyson_node.solve_dyson(G_node, F, Sigma, mu);
    double G_norm = l2_norm(G_node.local());

    // First call of the flagged solver is the node solve itself
    dyson_tau.solve_dyson(G_tau, F, Sigma, mu);
    CHECK(l2_norm(G_tau.local() - G_node.local()) == 0.0);

    // The node-Dyson G is a fixed point of the sweep
    dyson_tau.solve_dyson(G_tau, F, Sigma, mu);
    CHECK(l2_norm(G_tau.local() - G_node.local()) / G_norm < 1e-10);

    // Start the sweeps from the Sigma = 0 Green's function
    G_tau.node_sync();  // all ranks are done reading G_tau before the node root overwrites it
    if (context->node_comm.root()) G_tau.local() = G_hf.local();
    G_tau.node_sync();
    double err_start = l2_norm(G_tau.local() - G_node.local()) / G_norm;
    for (int it = 0; it < 12; ++it) dyson_tau.solve_dyson(G_tau, F, Sigma, mu);
    double err_converged = l2_norm(G_tau.local() - G_node.local()) / G_norm;
    app_log(1, "dyson_no_inversion: |G_tau - G_node|/|G_node| = {:.3e} (start from G_HF: {:.3e})",
            err_converged, err_start);
    CHECK(err_start > 1e-3);
    CHECK(err_converged < 1e-10);
  }

  // dyson_no_inversion = false (default): repeated solve_dyson calls at a nonzero Sigma return the same G bit for bit,
  // independent of the G in the array (the Matsubara-node solve is the only path).
  TEST_CASE("dyson_no_inversion_off", "[methods_scf]") {
    auto& context = utils::make_unit_test_mpi_context();
    std::string source_path = PROJECT_SOURCE_DIR;
    std::string filepath = source_path + "/tests/unit_test_files/qe/lih_kp222_nbnd16/";
    double beta = 100;
    double wmax = 4.0;
    auto mf = mf::make_MF(context, mf::qe_source, filepath, "pwscf");
    imag_axes_ft::IAFT ft(beta, wmax, imag_axes_ft::dlr_basis, "high");
    hamilt::pseudopot psp(mf);
    long nt = ft.nt_f(), ns = mf.nspin(), nk = mf.nkpts_ibz(), nb = mf.nbnd();
    sArray_t<Array_view_4D_t> F(math::shm::make_shared_array<Array_view_4D_t>(*context, {ns, nk, nb, nb}));
    sArray_t<Array_view_5D_t> G_first(math::shm::make_shared_array<Array_view_5D_t>(*context, {nt, ns, nk, nb, nb}));
    sArray_t<Array_view_5D_t> G_second(math::shm::make_shared_array<Array_view_5D_t>(*context, {nt, ns, nk, nb, nb}));
    sArray_t<Array_view_5D_t> Sigma(math::shm::make_shared_array<Array_view_5D_t>(*context, {nt, ns, nk, nb, nb}));
    hamilt::set_fock(mf, std::addressof(psp), F, true);

    simple_dyson dyson(std::addressof(mf), std::addressof(ft), 1e-9, "bisection");
    double mu = update_mu(0.0, dyson, mf, ft, F, Sigma);
    dyson.solve_dyson(G_first, F, Sigma, mu);
    Sigma.node_sync();  // all ranks are done reading Sigma before the node root overwrites it
    if (context->node_comm.root()) Sigma.local() = 0.0005 * G_first.local();
    Sigma.node_sync();
    dyson.solve_dyson(G_first, F, Sigma, mu);
    dyson.solve_dyson(G_second, F, Sigma, mu);
    CHECK(l2_norm(G_first.local() - G_second.local()) == 0.0);
  }


} // bdft_tests
