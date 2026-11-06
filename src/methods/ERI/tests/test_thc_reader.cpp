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

#include <map>

#include "catch2/catch.hpp"

#include "mpi3/environment.hpp"
#include "mpi3/communicator.hpp"
#include "mpi3/shared_communicator.hpp"
#include "utilities/mpi_context.h"

#include "utilities/test_common.hpp"
#include "utilities/symmetry.hpp"
#include "numerics/sparse/sparse.hpp"
#include "methods/tests/test_common.hpp"

#include "mean_field/MF.hpp"
#include "mean_field/mf_utils.hpp"
#include "mean_field/default_MF.hpp"
#include "mean_field/symmetry/unfold_bz.h"
#include "methods/ERI/eri_utils.hpp"
#include "methods/ERI/thc_reader_t.hpp"

namespace bdft_tests
{
  using namespace methods;
  using methods::make_thc_reader_ptree;

  TEST_CASE("thc_incore", "[methods]") {
    auto& mpi = utils::make_unit_test_mpi_context();

    auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi, mf::pyscf_source));
    thc_reader_t thc(mf, make_thc_reader_ptree(mf->nbnd()*15, "", "incore", "", "bdft", 1e-10, mf->ecutrho(),
                   1, 1024));

    int is = 0;
    int ik = 0;
    int iq = 0;
    REQUIRE(thc.X(is, 0, ik).shape() == shape_t<2>{thc.Np(), thc.nbnd()});
    REQUIRE(thc.Z(iq).shape() == shape_t<2>{thc.Np(), thc.Np()});
  }

#if defined(ENABLE_DEVICE)
  TEST_CASE("thc_incore_device", "[methods]") {
    auto& mpi = utils::make_unit_test_mpi_context();
    auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi, mf::qe_source));

    std::map<std::string,std::string> opt = {{"nIpts",std::to_string(mf->nbnd()*15)},
              {"ecut",std::to_string(0.3*mf->ecutrho())},
              {"compute","gpu"}, 
              {"thresh",std::to_string(1e-6)}};

    thc_reader_t thc(mf, io::make_ptree(opt));

    int is = 0;
    int ik = 0;
    int iq = 0;
    REQUIRE(thc.X(is, 0, ik).shape() == shape_t<2>{thc.Np(), thc.nbnd()});
    REQUIRE(thc.Z(iq).shape() == shape_t<2>{thc.Np(), thc.Np()});
  }
#endif

  TEST_CASE("make_thc", "[methods]") {
    auto& mpi = utils::make_unit_test_mpi_context();

    auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi, mf::qe_source));
    ptree pt; 
    pt.put("nIpts","16");     
    pt.put("metric","ov");     
    pt.put("alg","cholesky");     
    pt.put("storage","incore");     
    pt.put("save","./thc_eri.h5");     
    pt.put("format","bdft");     
    auto eri = make_thc(mf, pt);

    int is = 0;
    int ik = 0;
    int iq = 0;
    REQUIRE(eri.X(is, 0, ik).shape() == shape_t<2>{eri.Np(), eri.nbnd()});
    REQUIRE(eri.Z(iq).shape() == shape_t<2>{eri.Np(), eri.Np()});
    mpi->comm.barrier();
    if(mpi->comm.root()) remove("./thc_eri.h5");
  }  

  TEST_CASE("thc_outcore", "[methods]") {
    auto& mpi = utils::make_unit_test_mpi_context();

    auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi, mf::pyscf_source));
    thc_reader_t thc(mf, make_thc_reader_ptree(mf->nbnd()*15, "", "outcore", "./thc_eri.h5", "bdft", 1e-10,
                     mf->ecutrho(), 1, 1024));

    int is = 0;
    int ik = 0;
    int iq = 0;
    REQUIRE(thc.X(is, 0, ik).shape() == shape_t<2>{thc.Np(), thc.nbnd()});
    REQUIRE(thc.Z(iq).shape() == shape_t<2>{thc.Np(), thc.Np()});
    mpi->comm.barrier();
    if(mpi->comm.root()) remove("./thc_eri.h5");
  }

  TEST_CASE("thc_ls", "[methods]") {
    auto& mpi = utils::make_unit_test_mpi_context();

    auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi, "pyscf_h2o_mol"));
    auto [outd, pref] = utils::utest_filename("pyscf_h2o_mol");
    std::string gdf_dir = outd + "/gdf_eri/";
    thc_reader_t thc(mf, make_thc_reader_ptree(mf->nbnd()*15, gdf_dir, "incore", "",
                                               "bdft", 1e-10, mf->ecutrho(),
                                               1, 1024, 1, 0.2, 0.75, false));

    int is = 0;
    int ik = 0;
    int iq = 0;
    REQUIRE(thc.X(is, 0, ik).shape() == shape_t<2>{thc.Np(), thc.nbnd()});
    REQUIRE(thc.Z(iq).shape() == shape_t<2>{thc.Np(), thc.Np()});
    mpi->comm.barrier();
    if(mpi->comm.root()) remove("./thc_eri.h5");
    if(mpi->comm.root()) {
      remove("chol_info.h5");
      remove("Vq0.h5");
    }
  }
  // THC collocation matrix X(is,k) of the unfolded mean field (mf::unfold_mf, no symmetry) against that of its symmetry-reduced
  // parent, both built on the same interpolating points (pivot_file), for every k of qe_lih223_sym (incl. time-reversal k).
  // The parent builds X(k) from its IBZ orbitals by rotation (conjugation at time-reversal k); the unfolded mean field reads rotated
  // orbitals. A missing or doubled conjugation at a time-reversal k of the unfolded mean field gives an O(1) difference.
  TEST_CASE("thc_unfold_mf_collocation", "[methods]") {
    auto& mpi = utils::make_unit_test_mpi_context();
    auto& comm = mpi->comm;
    auto mf_ibz = std::make_shared<mf::MF>(mf::default_MF(mpi, "qe_lih223_sym", mf::h5_input_type));
    long nk = mf_ibz->nkpts(), ns = mf_ibz->nspin();
    REQUIRE(mf_ibz->nkpts_ibz() < nk);
    auto trev = mf_ibz->kp_trev();
    REQUIRE(std::any_of(trev.begin(), trev.end(), [](bool b) { return b; }));

    std::string unfold_prefix = "thc_unfold_mf", pivots = "thc_unfold_pivots.h5";
    mf::unfold_mf(*mf_ibz, unfold_prefix + ".h5");
    comm.barrier();
    auto mf_unfold = std::make_shared<mf::MF>(mf::make_MF(mpi, mf::bdft_source, "./", unfold_prefix, mf::h5_input_type));

    int nIpts = mf_ibz->nbnd()*15;
    double ecut = mf_ibz->ecutrho();
    auto pt_piv = make_thc_ptree(ecut, 8, 1024, 1e-10);
    pt_piv.put("nIpts", nIpts);
    pt_piv.put("save", pivots);
    make_thc_pivots(mf_ibz, pt_piv);
    comm.barrier();

    auto make_pt = [&]() {
      auto pt = make_thc_reader_ptree(nIpts, "", "incore", "", "bdft", 1e-10, ecut, 8, 1024);
      pt.put("pivot_file", pivots);
      return pt;
    };
    thc_reader_t thc_ibz(mf_ibz, make_pt());
    thc_reader_t thc_unfold(mf_unfold, make_pt());
    REQUIRE(thc_ibz.Np() == thc_unfold.Np());

    double err_max = 0.0, err_noconj_max = 0.0;
    for (long is = 0; is < ns; ++is)
      for (long ik = 0; ik < nk; ++ik) {
        nda::array<ComplexType,2> Xi = thc_ibz.X(is, 0, ik);
        nda::array<ComplexType,2> X_unfold = thc_unfold.X(is, 0, ik);
        double nrm = nda::frobenius_norm(Xi);
        err_max = std::max(err_max, nda::frobenius_norm(X_unfold - Xi) / nrm);
        if (mf_ibz->kp_trev(ik))
          err_noconj_max = std::max(err_noconj_max, nda::frobenius_norm(nda::conj(X_unfold) - Xi) / nrm);
      }
    app_log(2, "thc_unfold_mf_collocation: max ||X_unfold - X_ibz||/||X_ibz|| = {}, with conj at TR k = {}",
            err_max, err_noconj_max);
    REQUIRE(err_max < 1e-12);
    // sensitivity: conjugating X of the unfolded mean field at time-reversal k would be O(1) off
    REQUIRE(err_noconj_max > 0.1);

    comm.barrier();
    if (comm.root()) {
      remove((unfold_prefix + ".h5").c_str());
      remove(pivots.c_str());
    }
    comm.barrier();
  }

} // bdft_tests

