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

#include <cstdio>

#include "utilities/test_common.hpp"
#include "mean_field/default_MF.hpp"
#include "mean_field/mf_utils.hpp"
#include "methods/MBPT_drivers.h"
#include "numerics/imag_axes_ft/iaft_utils.hpp"
#include "methods/tools/chkpt_utils.h"
#include "mean_field/symmetry/unfold_bz.h"
#include "numerics/shared_array/nda.hpp"
#include <random>

namespace bdft_tests {

  TEST_CASE("iaft_wmax_from_mf", "[methods][mbpt][iaft]") {
    auto& mpi_context = utils::make_unit_test_mpi_context();
    
    auto run_wmax_check = [&](std::string const& mf_name) {
      
      auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi_context, mf_name));

      ptree pt;
      pt.put("beta", 1000.0);
#ifdef ENABLE_DLR
      pt.put("iaft.basis", "dlr");
#else
      pt.put("iaft.basis", "ir"); 
      pt.put("iaft.prec", "high");
#endif

      auto ft_ref = imag_axes_ft::IAFT(pt, true, mf::wmax_from_mf(*mf));
      auto wmax_ref = ft_ref.wmax();

      auto in_pt2 = pt;
      in_pt2.put("iaft.wmax", wmax_ref);
      auto ft = imag_axes_ft::IAFT(in_pt2, true, mf::wmax_from_mf(*mf) * 100.0);

      REQUIRE(ft.wmax() == Approx(ft_ref.wmax()));
      REQUIRE(ft.nt_f() == ft_ref.nt_f());
      REQUIRE(ft.nw_f() == ft_ref.nw_f());
    };

    SECTION("qe_solid") {
      run_wmax_check("qe_lih222_sym");
    }

    SECTION("pyscf_solid") {
      run_wmax_check("pyscf_si222");
    }

    SECTION("pyscf_molecule") {
      run_wmax_check("pyscf_h2o_mol");
    }

  }

  TEST_CASE("wmax_from_mf_formula", "[methods][mbpt][iaft]") {
    auto& mpi_context = utils::make_unit_test_mpi_context();
    auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi_context, "qe_lih222"));

    auto ev = mf->eigval();
    auto [it_min, it_max] = std::minmax_element(ev.data(), ev.data() + ev.size());
    double emin = *it_min, emax = *it_max, ef = mf->efermi();
    double span  = emax - emin;
    double halfw = std::max(emax - ef, ef - emin);

    // Sized for Sigma = G*W: the whole scale (span + halfw) times a fixed padding.
    // Asserts arithmetic, not accuracy.
    REQUIRE(mf::wmax_from_mf(*mf, 1.5) == Approx(1.5 * (span + halfw)));
    REQUIRE(mf::wmax_from_mf(*mf, 2.0) == Approx(2.0 * (span + halfw)));
  }

  // The "fixed_mu" mark of a checkpoint group (read by run_lr to refuse fix_density) is written on
  // the group it is given, only when mu is given, and removed again without one.
  TEST_CASE("dump_fixed_mu", "[methods][mbpt][chkpt]") {
    auto& mpi_context = utils::make_unit_test_mpi_context();
    auto& comm = mpi_context->comm;
    std::string fn = "dump_fixed_mu_test.mbpt.h5";
    if (comm.root()) {
      h5::file file(fn, 'w');
      h5::group grp(file);
      grp.create_group("scf");
    }
    comm.barrier();

    auto has = [&](std::string g) {
      int r = 0;
      if (comm.root()) {
        h5::file file(fn, 'r');
        h5::group grp(file);
        r = grp.has_subgroup(g) and grp.open_group(g).has_dataset("fixed_mu");
      }
      comm.broadcast_n(&r, 1, 0);
      return r == 1;
    };

    methods::chkpt::dump_fixed_mu(comm, fn, std::nullopt);
    methods::chkpt::dump_fixed_mu(comm, fn, std::nullopt, "embed");
    REQUIRE(not has("scf"));
    REQUIRE(not has("embed"));

    // "embed" does not exist yet: it is created
    methods::chkpt::dump_fixed_mu(comm, fn, 0.123, "embed");
    REQUIRE(has("embed"));
    REQUIRE(not has("scf"));
    if (comm.root()) {
      h5::file file(fn, 'r');
      double mu = 0.0;
      h5::h5_read(h5::group(file).open_group("embed"), "fixed_mu", mu);
      REQUIRE(mu == 0.123);
    }

    methods::chkpt::dump_fixed_mu(comm, fn, 0.5);
    REQUIRE(has("scf"));
    methods::chkpt::dump_fixed_mu(comm, fn, std::nullopt, "embed");
    REQUIRE(not has("embed"));
    REQUIRE(has("scf"));
    methods::chkpt::dump_fixed_mu(comm, fn, std::nullopt);
    REQUIRE(not has("scf"));

    comm.barrier();
    if (comm.root()) std::remove(fn.c_str());
    comm.barrier();
  }


  // chkpt::unfold_scf on a synthetic dyson checkpoint (random Dm, G, F, Sigma, H0, S on the IBZ of qe_lih223_sym,
  // which has time-reversal k) against the copy (conjugate at time-reversal k) from kp_to_ibz(k) done by hand.
  // Also checks final_iter, mu, the IAFT grid, the divergence treatments and the k list of the output.
  // The reference applies the same kp_to_ibz / kp_trev copy-conj rule as unfold_1e/dynamic_hamiltonian, so this tests
  // the plumbing (datasets, mu, final_iter, div strings), not the gauge rule; the gauge is tested by the H0 and HF
  // restart comparisons on the unfolded mean field. err_noconj only shows that the data can tell conj from copy.
  TEST_CASE("unfold_scf", "[methods][mbpt][chkpt]") {
    auto& mpi = utils::make_unit_test_mpi_context();
    auto& comm = mpi->comm;
    auto mf_ibz = mf::default_MF(mpi, "qe_lih223_sym", mf::h5_input_type);
    long ns = mf_ibz.nspin(), nk = mf_ibz.nkpts(), nki = mf_ibz.nkpts_ibz(), nb = mf_ibz.nbnd();

    // IAFT grid of the synthetic checkpoint
    ptree pt;
    pt.put("beta", 10.0);
#ifdef ENABLE_DLR
    pt.put("iaft.basis", "dlr");
#else
    pt.put("iaft.basis", "ir");
    pt.put("iaft.prec", "high");
#endif
    imag_axes_ft::IAFT ft(pt, true, mf::wmax_from_mf(mf_ibz));
    long nt = ft.nt_f();

    // Random IBZ matrices (seeded)
    using methods::chkpt::Array_view_4D_t;
    using methods::chkpt::Array_view_5D_t;
    std::mt19937 gen(7);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    auto fill = [&](auto& arr) {
      if (mpi->node_comm.root()) {
        auto a = arr.local();
        for (long i = 0; i < a.size(); ++i) a.data()[i] = ComplexType(dist(gen), dist(gen));
      }
      arr.node_sync();
    };
    auto sH0 = math::shm::make_shared_array<Array_view_4D_t>(*mpi, std::array<long,4>{ns, nki, nb, nb});
    auto sS = math::shm::make_shared_array<Array_view_4D_t>(*mpi, std::array<long,4>{ns, nki, nb, nb});
    auto sDm = math::shm::make_shared_array<Array_view_4D_t>(*mpi, std::array<long,4>{ns, nki, nb, nb});
    auto sF = math::shm::make_shared_array<Array_view_4D_t>(*mpi, std::array<long,4>{ns, nki, nb, nb});
    auto sG = math::shm::make_shared_array<Array_view_5D_t>(*mpi, std::array<long,5>{nt, ns, nki, nb, nb});
    auto sSigma = math::shm::make_shared_array<Array_view_5D_t>(*mpi, std::array<long,5>{nt, ns, nki, nb, nb});
    fill(sH0); fill(sS); fill(sDm); fill(sF); fill(sG); fill(sSigma);

    // Synthetic IBZ checkpoint, iteration 3, with both divergence treatments
    std::string in = "unfold_scf_in", out = "unfold_scf_out", unfold_prefix = "unfold_scf_unfolded";
    double mu = 0.37;
    methods::chkpt::write_metadata(comm, mf_ibz, ft, sH0, sS, in);
    methods::chkpt::dump_scf(comm, 3, sDm, sG, sF, sSigma, mu, in);
    if (comm.root()) {
      h5::file file(in + ".mbpt.h5", 'a');
      auto scf_grp = h5::group(file).open_group("scf");
      h5::h5_write(scf_grp, "div_treatment", std::string("gygi"));
      h5::h5_write(scf_grp, "hf_div_treatment", std::string("ignore_g0"));
    }
    comm.barrier();

    // Unfolded mean field and full-BZ checkpoint
    mf::unfold_mf(mf_ibz, unfold_prefix + ".h5");
    comm.barrier();
    auto mf_unfold = mf::make_MF(mpi, mf::bdft_source, "./", unfold_prefix, mf::h5_input_type);
    methods::chkpt::unfold_scf(*mpi, mf_ibz, mf_unfold, in, out);

    // k_weight of the output: 1/nkpts at every k, summing to 1
    int kw_ok = 0;
    if (comm.root()) {
      h5::file file(out + ".mbpt.h5", 'r');
      nda::array<double,1> kw;
      nda::h5_read(h5::group(file).open_group("system"), "k_weight", kw);
      kw_ok = (kw.size() == nk);
      for (long ik = 0; kw_ok and ik < nk; ++ik) kw_ok = (std::abs(kw(ik) - 1.0/double(nk)) < 1e-14);
    }
    comm.broadcast_n(&kw_ok, 1, 0);
    REQUIRE(kw_ok == 1);

    // Output read on root: scalars, k list, IAFT beta, and the error of each matrix against the hand-made copy/conj
    double err_H0 = 0, err_S = 0, err_Dm = 0, err_F = 0, err_G = 0, err_Sigma = 0, err_noconj = 0, err_k = 0;
    // Scalars read on root; compared on all ranks after a broadcast so that a failure cannot hang the other ranks.
    long iter_out = 0, nk_out = 0, nki_out = 0;
    double mu_out = 0, beta_out = 0;
    int div_ok = 0, hf_div_ok = 0;
    if (comm.root()) {
      h5::file file(out + ".mbpt.h5", 'r');
      h5::group grp(file);
      auto scf_grp = grp.open_group("scf");
      h5::h5_read(scf_grp, "final_iter", iter_out);
      auto iter_grp = scf_grp.open_group("iter3");
      h5::h5_read(iter_grp, "mu", mu_out);
      std::string div, hf_div;
      h5::h5_read(scf_grp, "div_treatment", div);
      h5::h5_read(scf_grp, "hf_div_treatment", hf_div);
      div_ok = (div == "gygi");
      hf_div_ok = (hf_div == "ignore_g0");
      auto sys_grp = grp.open_group("system");
      int nk_i = 0, nki_i = 0;
      h5::h5_read(sys_grp, "number_of_kpoints", nk_i);
      h5::h5_read(sys_grp, "number_of_kpoints_ibz", nki_i);
      nk_out = nk_i; nki_out = nki_i;
      nda::array<double,2> kc;
      nda::h5_read(sys_grp, "kpoints_crys", kc);
      for (long ik = 0; ik < nk; ++ik)
        for (int i = 0; i < 3; ++i) err_k = std::max(err_k, std::abs(kc(ik, i) - mf_ibz.kpts_crystal(ik, i)));
      {
        auto iaft_grp = grp.open_group("imaginary_fourier_transform");
        h5::h5_read(iaft_grp, "beta", beta_out);
      }

      // L2 error of the copy (conjugate at time-reversal k) from kp_to_ibz(k)
      auto check = [&](auto const& ref, std::string const& name, auto const& loc_grp, double& err) {
        constexpr bool dyn = (std::decay_t<decltype(ref)>::rank == 5);
        nda::array<ComplexType, (std::decay_t<decltype(ref)>::rank)> A;
        nda::h5_read(loc_grp, name, A);
        double e2 = 0, n2 = 0, e2_noconj = 0;
        long nt_ = dyn ? A.extent(0) : 1;
        for (long it = 0; it < nt_; ++it)
          for (long is = 0; is < ns; ++is)
            for (long ik = 0; ik < nk; ++ik) {
              long iks = mf_ibz.kp_to_ibz(ik);
              bool tr = mf_ibz.kp_trev(ik);
              for (long i = 0; i < nb; ++i)
                for (long j = 0; j < nb; ++j) {
                  ComplexType r, a;
                  if constexpr (dyn) {
                    r = ref(it, is, iks, i, j);
                    a = A(it, is, ik, i, j);
                  } else {
                    r = ref(is, iks, i, j);
                    a = A(is, ik, i, j);
                  }
                  e2 += std::norm(a - (tr ? std::conj(r) : r));
                  n2 += std::norm(r);
                  if (tr) e2_noconj += std::norm(a - r);
                }
            }
        err = std::sqrt(e2 / n2);
        err_noconj = std::max(err_noconj, std::sqrt(e2_noconj / n2));
      };
      check(sH0.local(), "H0_skij", sys_grp, err_H0);
      check(sS.local(), "S_skij", sys_grp, err_S);
      check(sDm.local(), "Dm_skij", iter_grp, err_Dm);
      check(sF.local(), "F_skij", iter_grp, err_F);
      check(sG.local(), "G_tskij", iter_grp, err_G);
      check(sSigma.local(), "Sigma_tskij", iter_grp, err_Sigma);
    }

    // Broadcast and compare on all ranks
    comm.broadcast_n(&err_H0, 1, 0); comm.broadcast_n(&err_S, 1, 0); comm.broadcast_n(&err_Dm, 1, 0);
    comm.broadcast_n(&err_F, 1, 0); comm.broadcast_n(&err_G, 1, 0); comm.broadcast_n(&err_Sigma, 1, 0);
    comm.broadcast_n(&err_noconj, 1, 0); comm.broadcast_n(&err_k, 1, 0);
    comm.broadcast_n(&iter_out, 1, 0); comm.broadcast_n(&nk_out, 1, 0); comm.broadcast_n(&nki_out, 1, 0);
    comm.broadcast_n(&mu_out, 1, 0); comm.broadcast_n(&beta_out, 1, 0);
    comm.broadcast_n(&div_ok, 1, 0); comm.broadcast_n(&hf_div_ok, 1, 0);
    app_log(2, "unfold_scf: relative errors H0 {} S {} Dm {} F {} G {} Sigma {}; without conj at TR k {}; k list {}",
            err_H0, err_S, err_Dm, err_F, err_G, err_Sigma, err_noconj, err_k);
    REQUIRE(iter_out == 3);
    REQUIRE(mu_out == mu);
    REQUIRE(div_ok == 1);
    REQUIRE(hf_div_ok == 1);
    REQUIRE(nk_out == nk);
    REQUIRE(nki_out == nk);
    REQUIRE(beta_out == ft.beta());
    REQUIRE(err_H0 == 0.0);
    REQUIRE(err_S == 0.0);
    REQUIRE(err_Dm == 0.0);
    REQUIRE(err_F == 0.0);
    REQUIRE(err_G == 0.0);
    REQUIRE(err_Sigma == 0.0);
    REQUIRE(err_k < 1e-10);
    // sensitivity: copying without the conjugation at time-reversal k is O(1) off
    REQUIRE(err_noconj > 0.1);

    comm.barrier();
    if (comm.root()) {
      std::remove((in + ".mbpt.h5").c_str());
      std::remove((out + ".mbpt.h5").c_str());
      std::remove((unfold_prefix + ".h5").c_str());
    }
    comm.barrier();
  }

} // bdft_tests
