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

} // bdft_tests
