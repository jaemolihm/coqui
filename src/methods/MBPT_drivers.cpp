/**
 * ==========================================================================
 * CoQuí: Correlated Quantum ínterface
 *
 * Copyright (c) 2022-2025 Simons Foundation & The CoQuí developer team
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



#include <string>
#include <string_view>

#include "configuration.hpp"
#include "mpi3/environment.hpp"
#include "mpi3/communicator.hpp"
#include "cxxopts.hpp"

#include "IO/AppAbort.hpp"
#include "IO/app_loggers.h"
#include "IO/ptree/ptree_utilities.hpp"

#include "hamiltonian/pseudo/pseudopot.h"
#include "utilities/mpi_context.h"
#include "mean_field/MF.hpp"
#include "mean_field/mf_utils.hpp"
#include "methods/ERI/mb_eri_context.h"
#include "methods/mb_state/mb_state.hpp"
#include "methods/ERI/div_treatment_e.hpp"
#include "methods/SCF/dca_dyson.h"
#include "methods/SCF/simple_dyson.h"
#include "methods/SCF/lr_dyson.hpp"
#include "methods/SCF/lr_driver.hpp"
#include "methods/SCF/lr_precompute.hpp"
#include "methods/SCF/scf_common.hpp"
#include "methods/embedding/embed_t.h"
#include "methods/embedding/embed_eri_t.h"
#include "numerics/imag_axes_ft/IAFT.hpp"
#include "numerics/iter_scf/iter_scf_utils.hpp"

#include "SCF/scf_driver.hpp"
#include "methods/GW/lr_gw.hpp"
#include "methods/scr_coulomb/lr_rpa_pi.hpp"
#include "methods/scr_coulomb/lr_scr_coulomb_t.hpp"
#include "utilities/lr_utils.hpp"
#include "MBPT_drivers.h"

namespace mpi3 = boost::mpi3;
namespace methods
{

// Helper function to prepare checkpoint file for downfold_coulomb
inline void ensure_checkpoint(std::shared_ptr<mf::MF> mf, std::string const& output,
                                 std::string const& greens_func_source, ptree const& pt) {

  if (greens_func_source == "mf" and std::filesystem::exists(output+".mbpt.h5")) {

    app_log(1, "");
    app_log(1, "╔═════════════════════════════════════════════════════════════╗");
    app_log(1, "║ [ NOTE ]                                                    ║");
    app_log(1, "║ greens_func_source is set to \"mf\", while a CoQuí checkpoint ║");
    app_log(1, "║ HDF5 with the same prefix has been detected. CoQuí will     ║");
    app_log(1, "║ read \"scf/iter0\" h5 group as the input, which should be     ║");
    app_log(1, "║ equivalent to the mean-field solution.                      ║");
    app_log(1, "╚═════════════════════════════════════════════════════════════╝\n");

  } else if (greens_func_source == "mf" and not std::filesystem::exists(output+".mbpt.h5")) {

    auto beta = io::get_value_with_default<double>(pt,"beta", 1000.0);
    auto wmax = io::get_value_with_default<double>(pt,"wmax", 12.0);
    auto iaft_prec = io::get_value_with_default<std::string>(pt, "iaft_prec", "high");
    imag_axes_ft::IAFT ft(beta, wmax, imag_axes_ft::ir_source, iaft_prec, false);
    hamilt::pseudopot psp(*mf);
    write_mf_data(*mf, ft, psp, output);

  } else if (greens_func_source == "scf" or greens_func_source == "embed") {

    utils::check(std::filesystem::exists(output+".mbpt.h5"),
                 "MBPT_drivers::ensure_checkpoint: greens_func_source == \"{}\" while the coqui h5, {}.mbpt.h5, does not exist!",
                 greens_func_source, output);

  } else {

    utils::check(false, "MBPT_drivers::ensure_checkpoint: invalid greens_func_source = {}. Valid options are \"mf\", \"scf\", and \"embed\".",
                 greens_func_source);

  }
}

/**
 * Factory function for creating simple_dyson with conditional H0 source.
 *
 * @param h0_source     - [INPUT] Source of H0: "compute" or "checkpoint"
 * @param mf_ptr        - [INPUT] Mean-field object pointer
 * @param ft_ptr        - [INPUT] IAFT pointer
 * @param chkpt_prefix  - [INPUT] Checkpoint file prefix (used if h0_source="checkpoint")
 * @param mu_tol        - [INPUT] Chemical potential tolerance (default 1e-9)
 * @return simple_dyson object
 */
inline simple_dyson make_dyson_with_h0_source(
    std::string_view h0_source,
    mf::MF* mf_ptr,
    imag_axes_ft::IAFT* ft_ptr,
    const std::string& chkpt_prefix,
    double mu_tol = 1e-9) {
  if (h0_source == "checkpoint") {
    app_log(1, "  H0 source                = checkpoint ({}.mbpt.h5)", chkpt_prefix);
    return simple_dyson(mf_ptr, ft_ptr, chkpt_prefix, mu_tol);
  } else {
    app_log(1, "  H0 source                = computed from orbitals");
    return simple_dyson(mf_ptr, ft_ptr, mu_tol);
  }
}

/**
 * Many-body perturbation calculations from a given mean-field and ERI objects with arguments in property tree.
 * Optional arguments (with default values):
 *  - beta: "1000" Inverse temperature (a.u.)
 *  - wmax: "12.0" Frequency cutoff for the IAFT grids (a.u.)
 *  - iaft_prec: "high" Precision of IAFT grids. {choices: "high", "medium", "low"}
 *  - div_treatment: "gygi" Divergent treatment for Coulomb kernel. {choices: "ignore_g0", "gygi"}
 *  - hf_div_treatment: "gygi" Divergent treatment for Coulomb kernel in HF. {choices: "ignore_g0", "gygi"}
 *  - niter: "1" Number of iterations in the self-consistent loop.
 *  - conv_thr: "1e-9" Convergence threshold for the self-consistent loop.
 *  - const_mu: "false" Fix the chemical potential during the self-consistent loop.
 *  - compute_exchange: "true" Whether to compute exchange (K) in the Fock matrix. Set to false for Hartree-only.
 *  - output: "bdft.mbpt" Prefix of the output h5 file.
 *  - restart: "false" Restart from a previous bdft.scf calculation.
 *  - t_prescreen_thresh: "0.0" Threshold for prescreening in time (GF2 only for now)
 *  - h0_source: "compute" Source of H0 matrix. {choices: "compute", "checkpoint"}
 *               "compute" (default) - Calculate H0 from plane-wave orbitals
 *               "checkpoint" - Read H0 from {output}.mbpt.h5 checkpoint
 */
template<typename eri_t>
void mbpt(std::string solver_type, eri_t &eri, ptree const& pt)
{
  auto mf = eri.corr_eri->get().MF();
  auto& mpi = eri.corr_eri->get().mpi();
  if (mpi->comm.size()%mpi->node_comm.size()!=0) {
    APP_ABORT("MBPT: number of processors on each node should be the same.");
  }
  std::string err = std::string("mbpt - Incorrect input - ");
  auto div_treatment = io::get_value_with_default<std::string>(pt, "div_treatment", "gygi");
  auto hf_div_treatment = io::get_value_with_default<std::string>(pt, "hf_div_treatment", "gygi");
  io::tolower(div_treatment);
  io::tolower(hf_div_treatment);

  auto niter = io::get_value_with_default<int>(pt,"niter",1);
  auto conv_thr = io::get_value_with_default<double>(pt,"conv_thr",1e-8);
  auto const_mu = io::get_value_with_default<bool>(pt,"const_mu",false);
  auto compute_exchange = io::get_value_with_default<bool>(pt,"compute_exchange",true);
  auto mu_tol = io::get_value_with_default<double>(pt,"mu_tolerance", 1e-9);
  auto output = io::get_value_with_default<std::string>(pt,"output","bdft.mbpt");

  auto restart = io::get_value_with_default<bool>(pt,"restart",false);
  auto greens_func_source = io::get_value_with_default<std::string>(pt,"greens_func_source", "scf");
  auto greens_func_iteration = io::get_value_with_default<long>(pt, "greens_func_iteration", -1);

  auto beta = io::get_value_with_default<double>(pt,"beta",1000.0);
  auto wmax = io::get_value_with_default<double>(pt,"wmax",12.0);
  auto iaft_prec = io::get_value_with_default<std::string>(pt, "iaft_prec", "high");

  // Parse h0_source parameter
  auto h0_source = io::get_value_with_default<std::string>(pt, "h0_source", "compute");
  io::tolower(h0_source);
  utils::check(h0_source == "compute" || h0_source == "checkpoint",
      "mbpt: unknown h0_source=\"{}\". Valid options: \"compute\", \"checkpoint\"", h0_source);

  bool chkpt_exist = std::filesystem::exists(output + ".mbpt.h5");
  if (restart and !chkpt_exist) {
    restart = false;
    app_log(1, "");
    app_log(1, "╔══════════════════════════════════════════════════════════╗");
    app_log(1, "║ [ WARNING ]                                              ║");
    app_log(1, "║ Running in restart mode while the checkpoint HDF5 does   ║");
    app_log(1, "║ not exist. Switching to the start-from-scratch mode.     ║");
    app_log(1, "╚══════════════════════════════════════════════════════════╝\n");
  } else if (not restart and chkpt_exist) {
    app_log(1, "");
    app_log(1, "╔══════════════════════════════════════════════════════════╗");
    app_log(1, "║ [ WARNING ]                                              ║");
    app_log(1, "║ An existing CoQuí checkpoint HDF5 with the same prefix   ║");
    app_log(1, "║ has been detected even though CoQuí is running in the    ║");
    app_log(1, "║ start-from-scratch mode. --> The old checkpoint will be  ║");
    app_log(1, "║ overwritten. Considering move the old HDF5 or change the ║");
    app_log(1, "║ prefix next time.                                        ║");
    app_log(1, "╚══════════════════════════════════════════════════════════╝\n");
  }

  imag_axes_ft::IAFT ft = (!restart)?
      imag_axes_ft::IAFT(beta, wmax, imag_axes_ft::ir_source, iaft_prec) :
      imag_axes_ft::read_iaft(output+".mbpt.h5");

  std::unique_ptr<iter_scf::iter_scf_t> iter_solver;

  using namespace solvers;
  hf_t hf(string_to_div_enum(hf_div_treatment));

  if(solver_type == "rpa") {
    simple_dyson dyson = make_dyson_with_h0_source(h0_source, mf.get(), &ft, output, mu_tol);
    gw_t gw(&ft, string_to_div_enum(div_treatment), output);
    MBState mb_state(mpi, ft, output);
    rpa_loop(mb_state, dyson, eri, ft, mb_solver_t(&hf,&gw));
  } else if(solver_type == "hf") {

    simple_dyson dyson = make_dyson_with_h0_source(h0_source, mf.get(), &ft, output, mu_tol);
    if (io::get_value_with_default<bool>(pt,"iter_alg.enable", true)) {
      iter_solver = std::make_unique<iter_scf::iter_scf_t>(iter_scf::make_iter_scf(pt));
    } else {
      iter_solver = nullptr;
    }
    MBState mb_state(mpi, ft, output);
    scf_loop(mb_state, dyson, eri, ft, mb_solver_t(&hf),
             iter_solver.get(), niter, restart, conv_thr, const_mu,
             greens_func_source, greens_func_iteration,
             compute_exchange);

  } else if(solver_type == "gw") {
    auto screen_type = io::get_value_with_default<std::string>(pt,"screen_type", "rpa");

    simple_dyson dyson = make_dyson_with_h0_source(h0_source, mf.get(), &ft, output, mu_tol);
    if (io::get_value_with_default<bool>(pt,"iter_alg.enable", true)) {
      iter_solver = std::make_unique<iter_scf::iter_scf_t>(iter_scf::make_iter_scf(pt));
    } else {
      iter_solver = nullptr;
    }
    solvers::scr_coulomb_t scr_eri(&ft, screen_type, string_to_div_enum(div_treatment));
    solvers::gw_t gw(&ft, string_to_div_enum(div_treatment), output);
    if (screen_type.substr(0,8)=="gw_edmft") {

      auto wannier_file = io::get_value<std::string>(pt,"wannier_file",err+"wannier_file");
      auto trans_home_cell = io::get_value_with_default<bool>(pt,"translate_home_cell",false);

      MBState mb_state(ft, output, mf, wannier_file, trans_home_cell);
      scf_loop(mb_state, dyson, eri, ft, mb_solver_t(&hf, &gw, &scr_eri),
               iter_solver.get(), niter, restart, conv_thr, const_mu,
               greens_func_source, greens_func_iteration,
               compute_exchange);

      auto dump_w_to_h5 = io::get_value_with_default<bool>(pt,"dump_w_to_h5", false);
      if (dump_w_to_h5) {
        auto& W_qtPQ = mb_state.dW_qtPQ.value();
        auto w_path = (std::filesystem::path(output).parent_path() / "thc_screened_interaction.h5").string();
        if (mb_state.mpi->comm.root()) {
          h5::file file(w_path, 'w');
          h5::group grp(file);
          math::nda::h5_write(grp, "W_qtPQ", W_qtPQ);
        } else {
          h5::group grp;
          math::nda::h5_write(grp, "W_qtPQ", W_qtPQ);
        }
      }
    } else {

      MBState mb_state(mpi, ft, output);
      scf_loop(mb_state, dyson, eri, ft, mb_solver_t(&hf, &gw, &scr_eri),
               iter_solver.get(), niter, restart, conv_thr, const_mu,
               greens_func_source, greens_func_iteration,
               compute_exchange);

      auto dump_w_to_h5 = io::get_value_with_default<bool>(pt,"dump_w_to_h5", false);
      if (dump_w_to_h5) {
        auto& W_qtPQ = mb_state.dW_qtPQ.value();
        auto w_path = (std::filesystem::path(output).parent_path() / "thc_screened_interaction.h5").string();
        if (mb_state.mpi->comm.root()) {
          h5::file file(w_path, 'w');
          h5::group grp(file);
          math::nda::h5_write(grp, "W_qtPQ", W_qtPQ);
        } else {
          h5::group grp;
          math::nda::h5_write(grp, "W_qtPQ", W_qtPQ);
        }
      }
    }

  } else if(solver_type == "gf2") {

    auto gf2_direct_type = io::get_value_with_default<std::string>(pt,"gf2_direct_type","gf2");
    auto gf2_exchange_alg = io::get_value_with_default<std::string>(pt,"gf2_exchange_alg","orb");
    auto gf2_exchange_type = io::get_value_with_default<std::string>(pt,"gf2_exchange_type","gf2");
    auto gf2_save_C = io::get_value_with_default<bool>(pt,"gf2_save_C",true);
    auto gf2_sosex_save_memory = io::get_value_with_default<bool>(pt,"gf2_sosex_save_memory",true);
    auto t_prescreen_thresh = io::get_value_with_default<double>(pt,"t_prescreen_thresh",0.0);

    simple_dyson dyson = make_dyson_with_h0_source(h0_source, mf.get(), &ft, output, mu_tol);
    if (io::get_value_with_default<bool>(pt,"iter_alg.enable", true)) {
      iter_solver = std::make_unique<iter_scf::iter_scf_t>(iter_scf::make_iter_scf(pt));
    } else {
      iter_solver = nullptr;
    }
    solvers::gf2_t gf2(mf.get(), &ft, string_to_div_enum(div_treatment),
                       gf2_direct_type, gf2_exchange_alg, gf2_exchange_type, output,
                       gf2_save_C, gf2_sosex_save_memory);
    gf2.t_thresh() = t_prescreen_thresh;

    MBState mb_state(mpi, ft, output);

    if (gf2_direct_type == "gf2") {
      scf_loop(mb_state, dyson, eri, ft, mb_solver_t(&hf, &gf2),
               iter_solver.get(), niter, restart, conv_thr, const_mu,
               greens_func_source, greens_func_iteration,
               compute_exchange);
    } else {
      solvers::scr_coulomb_t scr_eri(&ft, "rpa", string_to_div_enum(div_treatment));
      scf_loop(mb_state, dyson, eri, ft, mb_solver_t(&hf, &gf2, &scr_eri),
               iter_solver.get(), niter, restart, conv_thr, const_mu,
               greens_func_source, greens_func_iteration,
               compute_exchange);
    }

  } else if(solver_type == "gw_dca") {

    utils::check(false, "mbpt: gw_dca is not implemented!");
    /*ptree dca_pt = io::find_child(pt, "gw_mean_field");
    std::string mf_type = (mf.mf_type()==mf::qe_source)?
        "qe" : (mf.mf_type()==mf::pyscf_source)? "pyscf" : "bdft";
    mf::MF dca_mf(mf::make_MF(mpi, dca_pt, mf_type));
    dca_dyson dyson(mpi, &mf, &ft, dca_mf);
    solvers::gw_t gw(&ft, string_to_div_enum(div_treatment), output);
    scf_loop(dyson, eri, ft, mb_solver_t(&hf,&gw), nullptr,
             output, niter, restart, conv_thr, const_mu);*/

  } else if (solver_type == "qphf") {

    if (io::get_value_with_default<bool>(pt,"iter_alg.enable", true)) {
      iter_solver = std::make_unique<iter_scf::iter_scf_t>(iter_scf::make_iter_scf(pt));
    } else {
      iter_solver = nullptr;
    }
    MBState mb_state(mpi, ft, output);
    qp_context_t qp_context;
    qp_scf_loop<false>(mb_state, eri, ft, qp_context, mb_solver_t(&hf), iter_solver.get(),
                       niter, restart, conv_thr, compute_exchange);

  } else if (solver_type == "evgw0") {

    auto qp_type = io::get_value_with_default<std::string>(pt,"qp_type","sc");
    auto ac_alg  = io::get_value_with_default<std::string>(pt,"ac_alg","pade");
    auto eta     = io::get_value_with_default<double>(pt,"eta",0.0001);
    auto Nfit    = io::get_value_with_default<int>(pt,"Nfit",18);
    io::tolower(ac_alg);
    io::tolower(qp_type);
    qp_context_t qp_context(qp_type, ac_alg, Nfit, eta, conv_thr);
    if (io::get_value_with_default<bool>(pt,"iter_alg.enable", true)) {
      iter_solver = std::make_unique<iter_scf::iter_scf_t>(iter_scf::make_iter_scf(pt));
    } else {
      iter_solver = nullptr;
    }
    solvers::scr_coulomb_t scr_eri(&ft, "rpa", string_to_div_enum(div_treatment));
    solvers::gw_t gw(&ft, string_to_div_enum(div_treatment), output);
    MBState mb_state(mpi, ft, output);
    qp_scf_loop<true>(mb_state, eri, ft, qp_context, mb_solver_t(&hf,&gw,&scr_eri), iter_solver.get(),
                      niter, restart, conv_thr, compute_exchange);

  } else if (solver_type == "qpgw") {

    auto ac_alg  = io::get_value_with_default<std::string>(pt,"ac_alg","pade");
    auto eta     = io::get_value_with_default<double>(pt,"eta",0.0001);
    auto Nfit    = io::get_value_with_default<int>(pt,"Nfit",18);
    auto off_diag_mode = io::get_value_with_default<std::string>(pt,"off_diag_mode","fermi");
    io::tolower(ac_alg);
    io::tolower(off_diag_mode);
    utils::check(off_diag_mode=="fermi" or off_diag_mode=="qp_energy",
                 "unknown off_diag_mode: {}. Valid options are \"fermi\" and \"qp_energy\"");
    qp_context_t qp_context("sc", ac_alg, Nfit, eta, 1e-8, off_diag_mode);
    if (io::get_value_with_default<bool>(pt,"iter_alg.enable", true)) {
      iter_solver = std::make_unique<iter_scf::iter_scf_t>(iter_scf::make_iter_scf(pt));
    } else {
      iter_solver = nullptr;
    }
    solvers::scr_coulomb_t scr_eri(&ft, "rpa", string_to_div_enum(div_treatment));
    solvers::gw_t gw(&ft, string_to_div_enum(div_treatment), output);
    MBState mb_state(mpi, ft, output);
    qp_scf_loop<false>(mb_state, eri, ft, qp_context, mb_solver_t(&hf,&gw,&scr_eri), iter_solver.get(),
                       niter, restart, conv_thr, compute_exchange);

  } else
    APP_ABORT("mbpt: Unknown solver type: {}",solver_type);
}


template<typename eri_t>
void mbpt(std::string solver_type, eri_t &eri, ptree const& pt,
          nda::array<ComplexType, 5> const& projector_ksIai,
          nda::array<long, 3> const& band_window,
          nda::array<RealType, 2> const& kpts_crys,
          std::optional<std::map<std::string, nda::array<ComplexType, 5> > > local_polarizabilities)
{
  auto mf = eri.corr_eri->get().MF();
  auto& mpi = eri.corr_eri->get().mpi();
  if (mpi->comm.size()%mpi->node_comm.size()!=0) {
    APP_ABORT("MBPT: number of processors on each node should be the same.");
  }
  std::string err = std::string("mbpt - Incorrect input - ");
  auto div_treatment = io::get_value_with_default<std::string>(pt, "div_treatment", "gygi");
  auto hf_div_treatment = io::get_value_with_default<std::string>(pt, "hf_div_treatment", "gygi");
  io::tolower(div_treatment);
  io::tolower(hf_div_treatment);

  auto niter = io::get_value_with_default<int>(pt,"niter",1);
  auto conv_thr = io::get_value_with_default<double>(pt,"conv_thr",1e-8);
  auto const_mu = io::get_value_with_default<bool>(pt,"const_mu",false);
  auto compute_exchange = io::get_value_with_default<bool>(pt,"compute_exchange",true);
  auto mu_tol = io::get_value_with_default<double>(pt,"mu_tolerance", 1e-9);
  auto output = io::get_value_with_default<std::string>(pt,"output","bdft.mbpt");

  auto restart = io::get_value_with_default<bool>(pt,"restart",false);
  auto greens_func_source = io::get_value_with_default<std::string>(pt,"greens_func_source", "scf");
  auto greens_func_iteration = io::get_value_with_default<long>(pt, "greens_func_iteration", -1);
  bool chkpt_exist = std::filesystem::exists(output + ".mbpt.h5");
  if (restart and !chkpt_exist) {
    restart = false;
    app_log(1, "");
    app_log(1, "╔══════════════════════════════════════════════════════════╗");
    app_log(1, "║ [ WARNING ]                                              ║");
    app_log(1, "║ Running in restart mode while the checkpoint HDF5 does   ║");
    app_log(1, "║ not exist. Switching to the start-from-scratch mode.     ║");
    app_log(1, "╚══════════════════════════════════════════════════════════╝\n");
  } else if (not restart and chkpt_exist) {
    app_log(1, "");
    app_log(1, "╔══════════════════════════════════════════════════════════╗");
    app_log(1, "║ [ WARNING ]                                              ║");
    app_log(1, "║ An existing CoQuí checkpoint HDF5 with the same prefix   ║");
    app_log(1, "║ has been detected even though CoQuí is running in the    ║");
    app_log(1, "║ start-from-scratch mode. --> The old checkpoint will be  ║");
    app_log(1, "║ overwritten. Considering move the old HDF5 or change the ║");
    app_log(1, "║ prefix next time.                                        ║");
    app_log(1, "╚══════════════════════════════════════════════════════════╝\n");
  }

  auto trans_home_cell = io::get_value_with_default<bool>(pt,"translate_home_cell",false);

  auto beta = io::get_value_with_default<double>(pt,"beta",1000.0);
  auto wmax = io::get_value_with_default<double>(pt,"wmax",12.0);
  auto iaft_prec = io::get_value_with_default<std::string>(pt, "iaft_prec", "high");

  // Parse h0_source parameter
  auto h0_source = io::get_value_with_default<std::string>(pt, "h0_source", "compute");
  io::tolower(h0_source);
  utils::check(h0_source == "compute" || h0_source == "checkpoint",
      "mbpt: unknown h0_source=\"{}\". Valid options: \"compute\", \"checkpoint\"", h0_source);

  imag_axes_ft::IAFT ft = (!restart)?
                          imag_axes_ft::IAFT(beta, wmax, imag_axes_ft::ir_source, iaft_prec) :
                          imag_axes_ft::read_iaft(output+".mbpt.h5");

  std::unique_ptr<iter_scf::iter_scf_t> iter_solver;

  using namespace solvers;
  hf_t hf(string_to_div_enum(hf_div_treatment));

  if (solver_type == "gw") {

    auto screen_type = io::get_value_with_default<std::string>(pt,"screen_type", "rpa");

    simple_dyson dyson = make_dyson_with_h0_source(h0_source, mf.get(), &ft, output, mu_tol);
    if (io::get_value_with_default<bool>(pt,"iter_alg.enable", true)) {
      iter_solver = std::make_unique<iter_scf::iter_scf_t>(iter_scf::make_iter_scf(pt));
    } else {
      iter_solver = nullptr;
    }
    solvers::scr_coulomb_t scr_eri(&ft, screen_type, string_to_div_enum(div_treatment));
    solvers::gw_t gw(&ft, string_to_div_enum(div_treatment), output);
    MBState mb_state(ft, output, mf, projector_ksIai, band_window, kpts_crys, trans_home_cell, false);
    if (local_polarizabilities) {
      mb_state.set_local_polarizabilities(std::move(local_polarizabilities.value()));
      local_polarizabilities.reset();
    }

    scf_loop(mb_state, dyson, eri, ft, mb_solver_t(&hf, &gw, &scr_eri),
             iter_solver.get(), niter, restart, conv_thr, const_mu,
             greens_func_source, greens_func_iteration,
             compute_exchange);

  } else
    APP_ABORT("mbpt: Unknown solver type: {}",solver_type);
}

// FIXME this function requires HDF5_USE_FILE_LOCKING=FALSE.
void downfolding_1e(std::shared_ptr<mf::MF> mf, ptree const& pt) {
  std::string err = std::string("downfolding_1e - Incorrect input - ");
  auto prefix = io::get_value<std::string>(pt,"prefix",err+"prefix");
  auto outdir = io::get_value_with_default<std::string>(pt,"outdir","./");
  auto wannier_file = io::get_value<std::string>(pt,"wannier_file",err+"wannier_file");
  auto trans_home_cell = io::get_value_with_default<bool>(pt,"translate_home_cell",false);
  auto qp_selfenergy = io::get_value_with_default<bool>(pt,"qp_selfenergy",false);

  embed_t embed(*mf, wannier_file, trans_home_cell);

  imag_axes_ft::IAFT ft(imag_axes_ft::read_iaft(outdir+"/"+prefix+".mbpt.h5", false));

  MBState mb_state(ft, outdir+"/"+prefix, mf, wannier_file, trans_home_cell, false);

  if (qp_selfenergy) {
    auto ac_alg  = io::get_value_with_default<std::string>(pt,"ac_alg","pade");
    auto off_diag_mode = io::get_value_with_default<std::string>(pt,"off_diag_mode","qp_energy");
    io::tolower(ac_alg);
    io::tolower(off_diag_mode);
    utils::check(off_diag_mode=="fermi" or off_diag_mode=="qp_energy",
                 "unknown off_diag_mode: {}. Valid options are \"fermi\" and \"qp_energy\"");
    qp_context_t qp_context("sc", ac_alg,
                            io::get_value_with_default<int>(pt,"Nfit",30),
                            io::get_value_with_default<double>(pt,"eta",1e-6),
                            1e-8, off_diag_mode);
    embed.downfolding(mb_state, pt, &qp_context);
  } else {
    embed.downfolding(mb_state, pt);
  }
}

auto downfold_gloc_impl(std::shared_ptr<mf::MF> mf,
                        MBState&& mb_state,
                        ptree const& pt)
-> nda::array<ComplexType, 5> {
  std::string err = std::string("downfold_gloc_impl - Incorrect input - ");
  auto greens_func_source = io::get_value<std::string>(pt, "greens_func_source", err+"greens_func_source");
  auto greens_func_iteration = io::get_value_with_default<long>(pt, "greens_func_iteration", -1);
  auto force_real = io::get_value_with_default<bool>(pt, "force_real", true);
  embed_t embed(*mf);
  return embed.downfold_gloc(mb_state, force_real, greens_func_source, greens_func_iteration);
}

auto downfold_gloc(std::shared_ptr<mf::MF> mf, ptree const& pt,
                  nda::array<ComplexType, 5> const& projector_ksIai,
                  nda::array<long, 3> const& band_window,
                  nda::array<RealType, 2> const& kpts_crys)
  -> nda::array<ComplexType, 5> {
  std::string err = std::string("downfold_gloc - Incorrect input - ");
  auto prefix = io::get_value<std::string>(pt, "prefix", err+"prefix");
  auto outdir = io::get_value_with_default<std::string>(pt, "outdir", "./");
  auto trans_home_cell = io::get_value_with_default<bool>(pt, "translate_home_cell", false);
  imag_axes_ft::IAFT ft(imag_axes_ft::read_iaft(outdir+"/"+prefix+".mbpt.h5", false));
  return downfold_gloc_impl(
      mf, MBState(ft, outdir+"/"+prefix, mf, projector_ksIai, band_window, kpts_crys, trans_home_cell, false), pt);
}

auto downfold_gloc_with_projector_from_h5(std::shared_ptr<mf::MF> mf, ptree const& pt)
-> nda::array<ComplexType, 5> {
  std::string err = std::string("downfold_gloc - Incorrect input - ");
  auto prefix = io::get_value<std::string>(pt, "prefix", err+"prefix");
  auto outdir = io::get_value_with_default<std::string>(pt, "outdir", "./");
  auto wannier_file = io::get_value<std::string>(pt,"wannier_file",err+"wannier_file");
  auto trans_home_cell = io::get_value_with_default<bool>(pt, "translate_home_cell", false);
  imag_axes_ft::IAFT ft(imag_axes_ft::read_iaft(outdir+"/"+prefix+".mbpt.h5", false));
  return downfold_gloc_impl(
      mf, MBState(ft, outdir+"/"+prefix, mf, wannier_file, trans_home_cell, false), pt);
}

template<typename eri_t>
std::tuple<nda::array<ComplexType, 4>, nda::array<ComplexType, 5> >
downfold_coulomb_impl(eri_t &eri, MBState&& mb_state, ptree const& pt, 
                   std::optional<std::map<std::string, nda::array<ComplexType, 5> > > local_polarizabilities) {
  std::string err = std::string("downfold_coulomb_impl - Incorrect input - ");
  auto greens_func_source = io::tolower_copy(io::get_value<std::string>(pt, "greens_func_source"));
  greens_func_source = (greens_func_source == "mf") ? "scf" : greens_func_source;
  auto greens_func_iteration = io::get_value_with_default<long>(pt, "greens_func_iteration", -1);
  auto screen_type = io::get_value<std::string>(
      pt, "screen_type", err+"screen_type. This parameter determines the type of screened interactions for the downfolded Hamiltonian. "
                             "Valid types are \"crpa\", \"crpa_ks\", \"crpa_vasp\", "
                             "\"gw_edmft\", \"gw_edmft_rpa\", and \"gw_edmft_density\"");
  io::tolower(screen_type);
  auto permut_symm = io::get_value_with_default<bool>(pt, "permut_symm", true);
  auto force_real = io::get_value_with_default<bool>(pt, "force_real", true);
  auto div_treatment = io::tolower_copy(io::get_value_with_default<std::string>(pt, "div_treatment", "gygi"));
  auto bare_div_treatment = io::tolower_copy(io::get_value_with_default<std::string>(pt, "bare_div_treatment", "gygi"));
  auto output_in_tau = io::get_value_with_default<bool>(pt, "output_in_tau", false);
  bool write_to_hdf5 = io::get_value_with_default<bool>(pt, "write_to_hdf5", true);
  bool q_dependent_output = io::get_value_with_default<bool>(pt, "q_dependent_output", false);

  if (q_dependent_output) write_to_hdf5 = true;

  auto mf = eri.MF();

  // set local polarizabilities if provided
  if (local_polarizabilities) {
    mb_state.set_local_polarizabilities(std::move(local_polarizabilities.value()));
    local_polarizabilities.reset();
  }
  embed_eri_t embed_eri(*mf, string_to_div_enum(div_treatment),
              string_to_div_enum(bare_div_treatment), "default");
  return (output_in_tau)?
    embed_eri.compute_downfolded_coulomb_tensors<true>(
      eri, mb_state, screen_type, permut_symm, force_real, mb_state.ft, 
      greens_func_source, greens_func_iteration, write_to_hdf5, q_dependent_output) :
    embed_eri.compute_downfolded_coulomb_tensors<false>(
      eri, mb_state, screen_type, permut_symm, force_real, mb_state.ft, 
      greens_func_source, greens_func_iteration, write_to_hdf5, q_dependent_output);
}

template<typename eri_t>
std::tuple<nda::array<ComplexType, 4>, nda::array<ComplexType, 5> >
downfold_coulomb_with_projector_from_h5(eri_t &eri, ptree const& pt,
              std::optional<std::map<std::string, nda::array<ComplexType, 5> > > local_polarizabilities) {
  std::string err = std::string("downfold_coulomb - Incorrect input - ");
  auto outdir = io::get_value_with_default<std::string>(pt,"outdir","./");
  auto prefix = io::get_value<std::string>(pt,"prefix",err+"prefix");
  auto wannier_file = io::get_value<std::string>(pt,"wannier_file",err+"wannier_file");
  auto trans_home_cell = io::get_value_with_default<bool>(pt,"translate_home_cell",false);
  auto greens_func_source = io::tolower_copy(io::get_value<std::string>(pt, "greens_func_source",
      err+"greens_func_source. This parameter defines the source of input Green's function. Valid types are \"mf\", \"scf\", and \"embed\"."));

  auto mf = eri.MF();
  std::string output = outdir + "/" + prefix;
  
  ensure_checkpoint(mf, output, greens_func_source, pt);

  imag_axes_ft::IAFT ft(imag_axes_ft::read_iaft(output+".mbpt.h5", false));
  return downfold_coulomb_impl(
    eri, MBState(ft, output, mf, wannier_file, trans_home_cell, false),
    pt, local_polarizabilities);
}

template<typename eri_t>
std::tuple<nda::array<ComplexType, 4>, nda::array<ComplexType, 5> >
downfold_coulomb(eri_t &eri, ptree const& pt,
              nda::array<ComplexType, 5> const& projector_ksIai,
              nda::array<long, 3> const& band_window,
              nda::array<RealType, 2> const& kpts_crys,
              std::optional<std::map<std::string, nda::array<ComplexType, 5> > > local_polarizabilities) {
  std::string err = std::string("downfold_coulomb - Incorrect input - ");
  auto outdir = io::get_value_with_default<std::string>(pt,"outdir","./");
  auto prefix = io::get_value<std::string>(pt,"prefix",err+"prefix");
  auto trans_home_cell = io::get_value_with_default<bool>(pt,"translate_home_cell",false);
  auto greens_func_source = io::tolower_copy(io::get_value<std::string>(pt,"greens_func_source",
      err+"greens_func_source. This parameter defines the source of input Green's function. Valid types are \"mf\", \"scf\", and \"embed\". "));

  auto mf = eri.MF();
  std::string output = outdir + "/" + prefix;
  
  ensure_checkpoint(mf, output, greens_func_source, pt);

  imag_axes_ft::IAFT ft(imag_axes_ft::read_iaft(output+".mbpt.h5", false));
  return downfold_coulomb_impl(
    eri, MBState(ft, output, mf, projector_ksIai, band_window, kpts_crys, trans_home_cell, false),
    pt, local_polarizabilities);
}

/**
 * Downfolds the two-electron Hamiltonian with arguments in property tree.
 * Required arguments:
 *  - prefix: Prefix of the output and input files.
 *  - wannier_file: h5 file in which the Wannier transformation matrices are stored.
 *  - screen_type: Screening types for the partially screened interaction u(iw). {choices: "bare", "crpa", "edmft"}
 * Optional arguments (with default values):
 *  - outdir: "./" Directory where the source and output files are.
 *  - div_treatment: "gygi" Divergent treatment for Coulomb kernel. {choices: "ignore_g0", "gygi"}
 *  - bare_div_treatment: "gygi" Divergent treatment for the bare Coulomb kernel. {choices: "ignore_g0", "gygi"}
 * Optional arguments used only when outdir/prefix.mbpt.h5 does not exist:
 *  - beta: "1000" Inverse temperature (a.u.)
 *  - wmax: "12.0" Frequency cutoff for the IAFT grid (a.u.)
 *  - iaft_prec: "high" Precision of IAFT grids. {choices: "high", "medium", "low"}
 */
template<typename eri_t>
void downfolding_2e(eri_t &eri, ptree const& pt,
               std::optional<std::map<std::string, nda::array<ComplexType, 5> > > local_polarizabilities) {
  auto outdir = io::get_value_with_default<std::string>(pt, "outdir", "./");
  auto prefix = io::get_value<std::string>(pt, "prefix", "downfolding_2e - Incorrect input - prefix");
  auto wannier_file = io::get_value<std::string>(pt, "wannier_file", "downfolding_2e - Incorrect input - wannier_file");
  auto trans_home_cell = io::get_value_with_default<bool>(pt, "translate_home_cell", false);
  auto screen_type = io::tolower_copy(io::get_value<std::string>(pt, "screen_type", "downfolding_2e - Incorrect input - screen_type"));
  auto greens_func_source = io::tolower_copy(io::get_value<std::string>(pt, "greens_func_source", "downfolding_2e - Incorrect input - greens_func_source"));
  auto div_treatment = io::tolower_copy(io::get_value_with_default<std::string>(pt, "div_treatment", "gygi"));
  auto bare_div_treatment = io::tolower_copy(io::get_value_with_default<std::string>(pt, "bare_div_treatment", "gygi"));

  auto mf = eri.MF();
  std::string output = outdir + "/" + prefix;
  
  ensure_checkpoint(mf, output, greens_func_source, pt);

  imag_axes_ft::IAFT ft(imag_axes_ft::read_iaft(output+".mbpt.h5", false));
  MBState mb_state(ft, output, mf, wannier_file, trans_home_cell, false);

  if (local_polarizabilities) {
    mb_state.set_local_polarizabilities(std::move(local_polarizabilities.value()));
    local_polarizabilities.reset();
  }

  embed_eri_t embed_eri(*mf, string_to_div_enum(div_treatment),
                        string_to_div_enum(bare_div_treatment), "default");

  if (screen_type.substr(0, 8) == "gw_edmft") {
    embed_eri.downfolding_edmft(eri, mb_state, pt, screen_type);
  } else {
    embed_eri.downfolding_crpa(eri, mb_state, pt, screen_type);
  }
}

/**
 * Generates a downfolded Hamiltonian at the Hartree-Fock (HF) level. Bare 2-electron integrals
 * are calculated in the local basis, as defined by the provided projection matrix. 
 * HF frozen core contributions are added to the bare 1-body Hamiltonian in the local basis. 
 * The results are consistent with screen_type=bare and dc_type=hf in downfold_2e/downfold_1e routines.
 * Output is written in a format suitable to be read back by the mbpt modules, e.g. can be used in the 
 * mean_field and interaction sections.
 * Required arguments:
 *  - prefix: Prefix of the generated output mbpt and model files.
 *  - wannier_file: h5 file in which the Wannier transformation matrices are stored.
 * Optional arguments (with default values):
 *  - outdir: "./" Directory where the resulting prefix.mbpt.h5 and prefix.model.h5 files will be placed.
 *  - hf_div_treatment: "gygi" Divergent treatment for the bare Coulomb kernel. {choices: "ignore_g0", "gygi"}
 *  - permut_symm: false. If true, applies 4-/8-fold permutation symmetry to 2-electron interaction. Only
 applies if factorization="none".
 *  - force_real: false. If true, forces the 2-electron interaction tensor to be real.
 *  - factorization_type: "cholesky", Type of factorization. {choices: "none", "cholesky", "cholesky_high_memory", "choleksy_from_4index", "thc"}
 *  - thresh: 1e-6. Threshold used if factorization is requested.
 * Optional arguments used only when outdir/prefix.mbpt.h5 does not exist:
 *  - beta: "1000" Inverse temperature (a.u.)
 *  - wmax: "12.0" Frequency cutoff for the IAFT grids (a.u.)
 *  - iaft_prec: "high" Precision of IAFT grids. {choices: "high", "medium", "low"}
 */
template<typename eri_t>
void hf_downfold(eri_t &eri, ptree const& pt) {
  std::string err = std::string("hf_downfold - Incorrect input - ");
  auto prefix = io::get_value<std::string>(pt,"prefix",err+"prefix");
  auto outdir = io::get_value_with_default<std::string>(pt,"outdir","./");
  auto wannier_file = io::get_value<std::string>(pt,"wannier_file",err+"wannier_file");
  auto trans_home_cell = io::get_value_with_default<bool>(pt,"translate_home_cell",false);

  // two-body downfolding options
  auto hf_div_treatment = io::get_value_with_default<std::string>(pt, "hf_div_treatment", "gygi");
  auto factorization_type = io::get_value_with_default<std::string>(pt, "factorization_type", "cholesky");
  io::tolower(hf_div_treatment);
  io::tolower(factorization_type);

  utils::check( factorization_type=="none"                  or
                factorization_type=="cholesky"              or
                factorization_type=="cholesky_high_memory"  or
                factorization_type=="cholesky_from_4index"  or
                factorization_type=="thc",
                " downfold_2e: Invalid factorization_type: {}", factorization_type);

  auto mf = eri.MF();

  // mbpt and model outputs
  std::string output = outdir + "/" + prefix;

  // initialize
  imag_axes_ft::IAFT ft(
      io::get_value_with_default<double>(pt,"beta",1000.0),
      io::get_value_with_default<double>(pt,"wmax",12.0),
      imag_axes_ft::ir_source,
      io::get_value_with_default<std::string>(pt, "iaft_prec", "high"), true);
  hamilt::pseudopot psp(*mf);
  write_mf_data(*mf, ft, psp, output);
  MBState mb_state(ft, output, mf, wannier_file, trans_home_cell, false);

  // Two-body Hamiltonian
  embed_eri_t embed_eri(*mf, ignore_g0,
                        string_to_div_enum(hf_div_treatment), "model_static");
  embed_eri.downfolding_crpa(eri, mb_state, pt, "bare", factorization_type,
                             io::get_value_with_default<double>(pt, "thresh", 1e-6));

  // One-body Hamiltonian
  embed_t embed(*mf, wannier_file, trans_home_cell);
  embed.hf_downfolding(outdir, prefix, eri, ft,
                       io::get_value_with_default<bool>(pt, "force_real", true),
                       string_to_div_enum(hf_div_treatment));

}

/**
 * Generates a downfolded Hamiltonian at the GW level. cRPA Screened 2-electron integrals
 * are calculated in the local basis, as defined by the provided projection matrix. 
 * A quasi-particle approximation to the GW self-energy is applied to generate a downfolded
 * 1-body Hamiltonian in the local basis. 
 * The results are consistent with screen_type=crpa and dc_type=gw in downfold_2e/downfold_1e routines.
 * Output is written in a format suitable to be read back by the mbpt modules,
 * e.g. model Hamiltonian type mean-field chkpt file and ERI-compatible h5 chkpt file,
 * which can be used in the mean_field and interaction sections.
 * Required arguments:
 *  - prefix: Prefix of the generated output mbpt and model files.
 *  - wannier_file: h5 file in which the Wannier transformation matrices are stored.
 * Optional arguments (with default values):
 *  - outdir: "./" Directory where the resulting prefix.model.h5 files will be placed.
 *  - div_treatment: "gygi" Divergent treatment for Coulomb kernel. {choices: "ignore_g0", "gygi"}
 *  - hf_div_treatment: "gygi" Divergent treatment for the bare Coulomb kernel. {choices: "ignore_g0", "gygi"}
 *  - permut_symm: false. If true, applies 4-/8-fold permutation symmetry to 2-electron interaction. Only applies if factorization="none".
 *  - force_real: false. If true, forces the 2-electron interaction tensor to be real. 
 *  - factorization_type: "cholesky", Type of factorization. {choices: "none", "cholesky", "cholesky_high_memory", "choleksy_from_4index", "thc"}
 *  - thresh: 1e-6. Threshold used if factorization is requested.
 *  Parameters used by quasiparticle algorithm:
 *  - ac_alg: Algorithm for analytic continuation, default:pade {choices: pade}
 *  - eta: Smearing parameter: default:1e-6
 *  - Nfit: Number of terms in AC fit, default: 30
 *  - off_diag_mode: Off diagonal treatment, default: qp_energy. {choices: fermi, qp_energy} 
 */
template<typename eri_t>
void gw_downfold(eri_t &eri, ptree &pt) {
  std::string err = std::string("gw_downfold - Incorrect input - ");
  auto prefix = io::get_value<std::string>(pt,"prefix",err+"prefix");
  auto outdir = io::get_value_with_default<std::string>(pt,"outdir","./");
  auto wannier_file = io::get_value<std::string>(pt,"wannier_file",err+"wannier_file");
  auto trans_home_cell = io::get_value_with_default<bool>(pt,"translate_home_cell",false);

  // two-body downfolding options
  auto div_treatment = io::get_value_with_default<std::string>(pt, "div_treatment", "gygi");
  auto hf_div_treatment = io::get_value_with_default<std::string>(pt, "hf_div_treatment", "gygi");
  auto factorization_type = io::get_value_with_default<std::string>(pt, "factorization_type", "cholesky");
  io::tolower(div_treatment);
  io::tolower(hf_div_treatment);
  io::tolower(factorization_type);

  utils::check( factorization_type=="none"                  or
                factorization_type=="cholesky"              or
                factorization_type=="cholesky_high_memory"  or
                factorization_type=="cholesky_from_4index"  or
                factorization_type=="thc",
                " downfold_2e: Invalid factorization_type: {}", factorization_type);

  auto mf = eri.MF();

  // mbpt and model output
  std::string output = outdir + "/" + prefix;

  utils::check(std::filesystem::exists(output+".mbpt.h5"),
               "gw_downfolding: {}.mbpt.h5, does not exist!", output);

  imag_axes_ft::IAFT ft(imag_axes_ft::read_iaft(output+".mbpt.h5", false));
  // create MBstate object to store the state of the downfolding
  MBState mb_state(ft, output, mf, wannier_file, trans_home_cell, false);

  // Two-body Hamiltonian
  embed_eri_t embed_eri(*mf, string_to_div_enum(div_treatment),
                        string_to_div_enum(hf_div_treatment), "model_static");
  embed_eri.downfolding_crpa(eri, mb_state, pt, "crpa", factorization_type,
                             io::get_value_with_default<double>(pt, "thresh", 1e-6));

  // one body hamiltonian
  auto ac_alg  = io::get_value_with_default<std::string>(pt,"ac_alg","pade");
  auto off_diag_mode = io::get_value_with_default<std::string>(pt,"off_diag_mode","qp_energy");
  io::tolower(ac_alg);
  io::tolower(off_diag_mode);
  utils::check(off_diag_mode=="fermi" or off_diag_mode=="qp_energy",
               "unknown off_diag_mode: {}. Valid options are \"fermi\" and \"qp_energy\"");
  qp_context_t qp_context(
      "sc", ac_alg,
      io::get_value_with_default<int>(pt,"Nfit",30),
      io::get_value_with_default<double>(pt,"eta",1e-6),
      1e-8, off_diag_mode);
  embed_t embed(*mf, wannier_file, trans_home_cell);
  pt.put("update_dc", true);
  pt.put("dc_type", "gw");
  embed.downfolding(mb_state, pt, &qp_context, "model_static");
}

void dmft_embed_with_projector_from_h5(std::shared_ptr<mf::MF> mf, ptree const& pt,
                std::optional<std::map<std::string, nda::array<ComplexType, 4> > > local_hf_potentials,
                std::optional<std::map<std::string, nda::array<ComplexType, 5> > > local_selfenergies) {
  std::string err = std::string("dmft_embed - Incorrect input - ");
  auto prefix = io::get_value<std::string>(pt,"prefix",err+"prefix");
  auto outdir = io::get_value_with_default<std::string>(pt,"outdir","./");

  std::unique_ptr<iter_scf::iter_scf_t> iter_solver;
  if (io::get_value_with_default<bool>(pt,"iter_alg.enable", true)) {
    iter_solver = std::make_unique<iter_scf::iter_scf_t>(iter_scf::make_iter_scf(pt, 1.0));
  } else {
    iter_solver = nullptr;
  }
  imag_axes_ft::IAFT ft(imag_axes_ft::read_iaft(outdir+"/"+prefix+".mbpt.h5", false));
  MBState mb_state(ft, outdir+"/"+prefix, mf,
                   io::get_value<std::string>(pt,"wannier_file",err+"wannier_file"),
                   io::get_value_with_default<bool>(pt,"translate_home_cell",false), false);
  if (local_hf_potentials and local_selfenergies) {
    mb_state.set_local_hf_potentials(std::move(local_hf_potentials.value()));
    mb_state.set_local_selfenergies(std::move(local_selfenergies.value()));
    local_hf_potentials.reset();
    local_selfenergies.reset();
  }

  auto dyson = simple_dyson(mf.get(), &ft, mb_state.coqui_prefix,
                            io::get_value_with_default<double>(pt,"mu_tolerance", 1e-9));

  embed_t embed(*mf);
  embed.dmft_embed(mb_state, dyson, iter_solver.get(),
                   io::get_value_with_default<bool>(pt,"qp_approx_mbpt",false),
                   io::get_value_with_default<bool>(pt,"corr_only",false));
}

void dmft_embed(std::shared_ptr<mf::MF> mf, ptree const& pt,
                nda::array<ComplexType, 5> const& projector_ksIai,
                nda::array<long, 3> const& band_window,
                nda::array<RealType, 2> const& kpts_crys,
                std::optional<std::map<std::string, nda::array<ComplexType, 4> > > local_hf_potentials,
                std::optional<std::map<std::string, nda::array<ComplexType, 5> > > local_selfenergies) {
  std::string err = std::string("dmft_embed - Incorrect input - ");
  auto prefix = io::get_value<std::string>(pt,"prefix",err+"prefix");
  auto outdir = io::get_value_with_default<std::string>(pt,"outdir","./");

  std::unique_ptr<iter_scf::iter_scf_t> iter_solver;
  if (io::get_value_with_default<bool>(pt,"iter_alg.enable", true)) {
    iter_solver = std::make_unique<iter_scf::iter_scf_t>(iter_scf::make_iter_scf(pt, 1.0));
  } else {
    iter_solver = nullptr;
  }
  imag_axes_ft::IAFT ft(imag_axes_ft::read_iaft(outdir+"/"+prefix+".mbpt.h5", false));
  MBState mb_state(ft, outdir+"/"+prefix, mf,
                   projector_ksIai, band_window, kpts_crys,
                   io::get_value_with_default<bool>(pt,"translate_home_cell",false), false);
  if (local_hf_potentials and local_selfenergies) {
    mb_state.set_local_hf_potentials(std::move(local_hf_potentials.value()));
    mb_state.set_local_selfenergies(std::move(local_selfenergies.value()));
    local_hf_potentials.reset();
    local_selfenergies.reset();
  }

  auto dyson = simple_dyson(mf.get(), &ft, mb_state.coqui_prefix,
                            io::get_value_with_default<double>(pt,"mu_tolerance", 1e-9));

  embed_t embed(*mf);
  embed.dmft_embed(mb_state, dyson, iter_solver.get(),
                   io::get_value_with_default<bool>(pt,"qp_approx_mbpt",false),
                   io::get_value_with_default<bool>(pt,"corr_only",false));
}



/**
 * Compute a 4D processor grid that factors nproc across dimensions.
 *
 * Distributes processors across axes 0 and 1 first (using find_proc_grid_max_npools),
 * then splits remaining processors across axes 2 and 3 (using find_proc_grid_min_diff).
 * Axes in skip_axes are kept undivided (grid=1 on those axes).
 *
 * @return processor grid {n0, n1, n2, n3} with n0*n1*n2*n3 == nproc
 */
inline std::array<long, 4> compute_proc_grid_4D(long nproc, std::array<long, 4> shape,
                                                 std::array<bool, 4> skip_axes = {false, false, false, false}) {
  long np = nproc;
  std::array<long, 4> grid = {1, 1, 1, 1};
  // Distribute over axes 0 and 1 (unless skipped)
  for (int ax = 0; ax < 2; ++ax) {
    if (!skip_axes[ax]) {
      grid[ax] = utils::find_proc_grid_max_npools(np, shape[ax], 0.2);
      np /= grid[ax];
    }
  }
  // Distribute remaining over axes 2 and 3 (unless skipped)
  if (!skip_axes[2] && !skip_axes[3]) {
    grid[2] = utils::find_proc_grid_min_diff(np, shape[2], shape[3]);
    grid[3] = np / grid[2];
  } else if (!skip_axes[2]) {
    grid[2] = np;
  } else if (!skip_axes[3]) {
    grid[3] = np;
  }
  return grid;
}

/**
 * Wrap a regular 4D array into a distributed array with a proper MPI grid.
 *
 * The source array must be fully available on rank 0.
 * All ranks receive their local portion via broadcast + local copy.
 *
 * @param comm   - [INPUT] MPI communicator
 * @param shape  - [INPUT] Global array shape
 * @param src    - [INPUT] Source array (only meaningful on rank 0)
 * @return Distributed array with data from src
 */
template<typename communicator_t>
auto distribute_array_4D(communicator_t& comm,
                         std::array<long, 4> shape,
                         nda::array<ComplexType, 4> const& src_in,
                         std::array<bool, 4> skip_axes = {false, false, false, false})
{
  using local_Array_4D_t = nda::array<ComplexType, 4>;
  auto pgrid = compute_proc_grid_4D(comm.size(), shape, skip_axes);
  auto darray = math::nda::make_distributed_array<local_Array_4D_t>(
      comm, pgrid, shape);

  // Broadcast full array from root, then each rank extracts its local portion
  nda::array<ComplexType, 4> src(src_in);
  comm.broadcast_n(src.data(), src.size(), 0);
  auto loc = darray.local();
  auto [o0, o1, o2, o3] = darray.origin();
  auto [n0, n1, n2, n3] = darray.local_shape();
  for (long i0 = 0; i0 < n0; ++i0)
    for (long i1 = 0; i1 < n1; ++i1)
      for (long i2 = 0; i2 < n2; ++i2)
        for (long i3 = 0; i3 < n3; ++i3)
          loc(i0, i1, i2, i3) = src(i0 + o0, i1 + o1, i2 + o2, i3 + o3);
  comm.barrier();
  return darray;
}

/**
 * Helper: load W and eps_inv_head from checkpoint/thc_screened_interaction.h5
 *
 * Factored out from lr_gw_sigma_DeltaG_calc and gw_evaluate_sigma_calc to avoid duplication.
 * Also used by run_lr_calc when include_gw_sigma=true.
 *
 * @param mpi         - MPI context
 * @param thc         - THC ERI (provides Np())
 * @param input_file  - Path to checkpoint HDF5 file (for eps_inv_head)
 * @param input_grp   - SCF group name in checkpoint (e.g., "scf")
 * @param input_iter  - Iteration to read (-1 = final_iter)
 * @param nt          - Number of tau points (from IAFT)
 * @return (dW_qtPQ, eps_inv_head) tuple
 */
template<typename mpi_context_t, typename THC_t>
auto load_W_and_eps_inv_head(
    mpi_context_t& mpi,
    THC_t& thc,
    const std::string& input_file,
    const std::string& input_grp,
    long input_iter,
    long nt)
{
  auto mf = thc.MF();
  long nkpts = mf->nkpts();
  long NP = thc.Np();
  long nt_half = (nt % 2 == 0) ? nt / 2 : nt / 2 + 1;

  // Read W from thc_screened_interaction.h5 (same directory as checkpoint)
  auto w_file = (std::filesystem::path(input_file).parent_path() / "thc_screened_interaction.h5").string();
  app_log(2, "Reading W from {}...", w_file);

  // Read full array on root, then distribute across MPI ranks
  nda::array<ComplexType, 4> W_qtPQ(nkpts, nt_half, NP, NP);
  W_qtPQ() = 0.0;
  if (mpi.comm.root()) {
    h5::file file(w_file, 'r');
    h5::group grp(file);
    nda::h5_read(grp, "W_qtPQ", W_qtPQ);
  }
  // τ-dist for (q,t,P,Q): swap axes 0,1 of the (t,q) τ-dist grid
  auto [tq_pgrid, tq_bsize] = utils::lr_W_q_local_dist(mpi.comm.size(), nt_half, NP);
  std::array<long,4> qt_pgrid = {tq_pgrid[1], tq_pgrid[0], tq_pgrid[2], tq_pgrid[3]};
  std::array<long,4> qt_bsize = {tq_bsize[1], tq_bsize[0], tq_bsize[2], tq_bsize[3]};

  using local_Array_4D_t = nda::array<ComplexType, 4>;
  auto dW_qtPQ = math::nda::make_distributed_array<local_Array_4D_t>(
      mpi.comm, qt_pgrid, {nkpts, nt_half, NP, NP}, qt_bsize);

  // Broadcast and copy
  nda::array<ComplexType, 4> src(W_qtPQ);
  mpi.comm.broadcast_n(src.data(), src.size(), 0);
  auto loc = dW_qtPQ.local();
  auto [o0, o1, o2, o3] = dW_qtPQ.origin();
  auto [n0, n1, n2, n3] = dW_qtPQ.local_shape();
  for (long i0 = 0; i0 < n0; ++i0)
    for (long i1 = 0; i1 < n1; ++i1)
      for (long i2 = 0; i2 < n2; ++i2)
        for (long i3 = 0; i3 < n3; ++i3)
          loc(i0, i1, i2, i3) = src(i0+o0, i1+o1, i2+o2, i3+o3);
  mpi.comm.barrier();

  // Read eps_inv_head from checkpoint
  nda::array<ComplexType, 1> eps_inv_head(nt_half);
  if (mpi.comm.root()) {
    h5::file file(input_file, 'r');
    auto root = h5::group(file);
    auto scf_grp = root.open_group(input_grp);
    long iter = input_iter;
    if (iter == -1) {
      h5::h5_read(scf_grp, "final_iter", iter);
    }
    auto iter_grp = scf_grp.open_group("iter" + std::to_string(iter));
    nda::h5_read(iter_grp, "eps_inv_head_t", eps_inv_head);
  }
  mpi.comm.broadcast_n(eps_inv_head.data(), eps_inv_head.size(), 0);
  mpi.comm.barrier();

  return std::make_pair(std::move(dW_qtPQ), std::move(eps_inv_head));
}


// transpose_axes_01 is in utilities/lr_utils.hpp


/**
 * Unified linear response calculation.
 *
 * Runs the LR SCF loop with configurable Hartree, Exchange, and GW self-energy components:
 *   ΔH0 → ΔG → ΔDm → [ΔF] → [ΔΣ] → ΔG → ... (iterate until convergence)
 *
 * Subsumes both lr_dyson_calc (all flags false, max_iter=1) and
 * lr_hf_scf_calc (include_hartree+exchange, no GW).
 *
 * @param eri              - [INPUT] ERI handler (must be THC)
 * @param pt               - [INPUT] Parameters as property tree
 * @param q_vec            - [INPUT] Perturbation wavevector in crystal coords (3,)
 * @param DeltaH0_skij     - [INPUT] Perturbation matrix (ns, nk, nb, nb)
 * @param include_hartree  - [INPUT] Include ΔJ in SCF loop
 * @param include_exchange - [INPUT] Include ΔK in SCF loop
 * @param gw_mode          - [INPUT] GW self-energy mode (none/fixed_W/full)
 * @param max_iter         - [INPUT] Maximum SCF iterations (1 = one-shot)
 * @param tol              - [INPUT] Convergence tolerance for ||ΔDm_new - ΔDm_old||
 * @param fix_density      - [INPUT] If true, compute Δμ to enforce ΔN=0
 * @param iter_params      - [INPUT] Iteration algorithm parameters (damping/DIIS)
 * @return Tuple of (number of iterations, final Δμ)
 */
template<typename eri_t>
std::tuple<int, double> run_lr_calc(eri_t &eri, ptree const& pt,
                                     nda::array<double, 1> const& q_vec,
                                     nda::array<ComplexType, 4> const& DeltaH0_skij,
                                     bool include_hartree,
                                     bool include_exchange,
                                     lr_gw_update_mode gw_mode,
                                     int max_iter,
                                     double tol,
                                     bool fix_density,
                                     const lr_iter_params& iter_params,
                                     const nda::array<ComplexType, 4>* DeltaX_left,
                                     const nda::array<ComplexType, 4>* DeltaX_right) {
  auto mf = eri.corr_eri->get().MF();
  auto& mpi = eri.corr_eri->get().mpi();

  std::string err = std::string("run_lr_calc - Incorrect input - ");
  auto prefix = io::get_value<std::string>(pt, "prefix", err + "prefix");
  auto output = io::get_value_with_default<std::string>(pt, "output", prefix);
  auto input_grp = io::get_value_with_default<std::string>(pt, "input_type", "scf");
  auto input_iter = io::get_value_with_default<long>(pt, "input_iter", -1);
  auto h0_source = io::get_value_with_default<std::string>(pt, "h0_source", "compute");

  std::string input_file = prefix + ".mbpt.h5";
  utils::check(std::filesystem::exists(input_file),
               "run_lr_calc: Input checkpoint {} does not exist!", input_file);

  // Validate checkpoint contains required data
  {
    h5::file file(input_file, 'r');
    auto root = h5::group(file);
    utils::check(root.has_subgroup(input_grp),
                 "run_lr_calc: Checkpoint {} does not contain '{}/' group. "
                 "Run HF or GW calculation first.", input_file, input_grp);
    auto grp = root.open_group(input_grp);
    if (input_iter == -1) {
      utils::check(grp.has_dataset("final_iter"),
                   "run_lr_calc: Checkpoint {}/{}/ missing 'final_iter'. "
                   "SCF calculation may not have completed.", input_file, input_grp);
    } else {
      utils::check(grp.has_subgroup("iter" + std::to_string(input_iter)),
                   "run_lr_calc: Checkpoint {}/{}/ does not contain 'iter{}'. "
                   "Check input_iter parameter.", input_file, input_grp, input_iter);
    }
  }

  // Read IAFT from checkpoint
  imag_axes_ft::IAFT ft(imag_axes_ft::read_iaft(input_file, false));

  // Create simple_dyson
  utils::check(h0_source == "compute" || h0_source == "checkpoint",
               "run_lr_calc: h0_source must be 'compute' or 'checkpoint', got '{}'", h0_source);
  simple_dyson dyson = (h0_source == "checkpoint")
      ? simple_dyson(mf.get(), &ft, prefix)
      : simple_dyson(mf.get(), &ft);

  // Allocate shared arrays for unperturbed state
  auto sF_skij = math::shm::make_shared_array<Array_view_4D_t>(
      *mpi, {mf->nspin(), mf->nkpts_ibz(), mf->nbnd(), mf->nbnd()});
  auto sDm_skij = math::shm::make_shared_array<Array_view_4D_t>(
      *mpi, {mf->nspin(), mf->nkpts_ibz(), mf->nbnd(), mf->nbnd()});
  auto sG_tskij = math::shm::make_shared_array<Array_view_5D_t>(
      *mpi, {ft.nt_f(), mf->nspin(), mf->nkpts_ibz(), mf->nbnd(), mf->nbnd()});
  auto sSigma_tskij = math::shm::make_shared_array<Array_view_5D_t>(
      *mpi, {ft.nt_f(), mf->nspin(), mf->nkpts_ibz(), mf->nbnd(), mf->nbnd()});

  // Read F and Sigma from checkpoint
  double mu = 0.0;
  chkpt::read_scf(mpi->node_comm, sF_skij, sSigma_tskij, mu,
                  prefix, input_grp, input_iter);
  mpi->comm.barrier();

  // Compute G from F and Sigma via Dyson equation
  app_log(2, "Computing unperturbed Green's function from checkpoint...");
  app_log(2, "  h0_source = {}", h0_source);
  app_log(2, "  mu = {:.8f}", mu);
  update_G(dyson, *mf, ft, sDm_skij, sG_tskij, sF_skij, sSigma_tskij, mu, true);
  mpi->comm.barrier();

  // Allocate LR arrays
  auto sDeltaH0_skij = math::shm::make_shared_array<Array_view_4D_t>(
      *mpi, {mf->nspin(), mf->nkpts_ibz(), mf->nbnd(), mf->nbnd()});
  auto sDeltaG_tskij = math::shm::make_shared_array<Array_view_5D_t>(
      *mpi, {ft.nt_f(), mf->nspin(), mf->nkpts_ibz(), mf->nbnd(), mf->nbnd()});
  auto sDeltaDm_skij = math::shm::make_shared_array<Array_view_4D_t>(
      *mpi, {mf->nspin(), mf->nkpts_ibz(), mf->nbnd(), mf->nbnd()});
  auto sDeltaF_skij = math::shm::make_shared_array<Array_view_4D_t>(
      *mpi, {mf->nspin(), mf->nkpts_ibz(), mf->nbnd(), mf->nbnd()});
  // Only allocate ΔΣ array when GW is active
  bool include_gw_sigma = (gw_mode != lr_gw_update_mode::none);
  using sArray_5D_t = math::shm::shared_array<Array_view_5D_t>;
  std::optional<sArray_5D_t> opt_sDeltaSigma_tskij;
  if (include_gw_sigma) {
    opt_sDeltaSigma_tskij.emplace(math::shm::make_shared_array<Array_view_5D_t>(
        *mpi, {ft.nt_f(), mf->nspin(), mf->nkpts_ibz(), mf->nbnd(), mf->nbnd()}));
  }

  // Validate DeltaH0 dimensions
  utils::check(DeltaH0_skij.shape(0) == mf->nspin() &&
               DeltaH0_skij.shape(1) == mf->nkpts_ibz() &&
               DeltaH0_skij.shape(2) == mf->nbnd() &&
               DeltaH0_skij.shape(3) == mf->nbnd(),
               "run_lr_calc: DeltaH0_skij shape mismatch: expected ({},{},{},{}), got ({},{},{},{})",
               mf->nspin(), mf->nkpts_ibz(), mf->nbnd(), mf->nbnd(),
               DeltaH0_skij.shape(0), DeltaH0_skij.shape(1),
               DeltaH0_skij.shape(2), DeltaH0_skij.shape(3));

  // Copy DeltaH0 to shared array
  if (mpi->node_comm.root()) {
    sDeltaH0_skij.local() = DeltaH0_skij;
  }
  mpi->comm.barrier();

  // Load W and eps_inv_head if GW self-energy is requested
  using dW_type = decltype(load_W_and_eps_inv_head(*mpi, eri.corr_eri->get(), input_file, input_grp, input_iter, ft.nt_f()).first);
  std::optional<dW_type> opt_dW;
  std::optional<nda::array<ComplexType, 1>> opt_eps_inv;
  if (include_gw_sigma) {
    auto [dW, eps_inv] = load_W_and_eps_inv_head(*mpi, eri.corr_eri->get(), input_file, input_grp, input_iter, ft.nt_f());
    opt_dW.emplace(std::move(dW));
    opt_eps_inv.emplace(std::move(eps_inv));
  }

  // Create lr_driver and run unified SCF loop
  lr_driver driver(dyson, q_vec);
  auto& thc = eri.corr_eri->get();
  auto* pDeltaSigma = opt_sDeltaSigma_tskij ? &(*opt_sDeltaSigma_tskij) : nullptr;
  // Pass DeltaX and unperturbed Dm for DeltaX/IBC correction
  const nda::array<ComplexType, 4>* Dm_ab_ptr = nullptr;
  nda::array<ComplexType, 4> Dm_ab_local;
  if (DeltaX_left) {
    Dm_ab_local = sDm_skij.local();
    Dm_ab_ptr = &Dm_ab_local;
  }
  auto [niter, Delta_mu] = driver.run_lr(
      sDeltaG_tskij, sDeltaDm_skij, sDeltaF_skij, pDeltaSigma,
      sG_tskij, sDeltaH0_skij, thc,
      include_hartree, include_exchange, gw_mode,
      opt_dW ? &(*opt_dW) : nullptr,
      opt_eps_inv ? &(*opt_eps_inv) : nullptr,
      max_iter, tol, fix_density, iter_params,
      DeltaX_left, DeltaX_right, Dm_ab_ptr);
  mpi->comm.barrier();

  // Write results
  chkpt::dump_lr(mpi->comm, output + ".mbpt.h5", q_vec,
                 sDeltaG_tskij, sDeltaDm_skij, sDeltaF_skij, pDeltaSigma,
                 Delta_mu, niter,
                 include_hartree, include_exchange, include_gw_sigma);

  return std::make_tuple(niter, Delta_mu);
}


/**
 * Linear response GW self-energy with fixed W (term 1).
 *
 * Computes ΔΣ = -ΔG ⊙ W_c (R-space).
 * Reads W from thc_screened_interaction.h5 and eps_inv_head from checkpoint.
 * TODO: Divergence correction not yet implemented (see lr_gw.hpp class docstring).
 */
template<typename eri_t>
nda::array<ComplexType, 5> lr_gw_sigma_DeltaG_calc(
    eri_t &eri, ptree const& pt,
    nda::array<double, 1> const& q_pert,
    nda::array<ComplexType, 5> const& DeltaG_tskij) {

  auto& thc = eri.corr_eri->get();
  auto mf = thc.MF();
  auto& mpi = thc.mpi();

  std::string err = std::string("lr_gw_sigma_DeltaG_calc - Incorrect input - ");
  auto prefix = io::get_value<std::string>(pt, "prefix", err + "prefix");
  auto input_grp = io::get_value_with_default<std::string>(pt, "input_type", "scf");
  auto input_iter = io::get_value_with_default<long>(pt, "input_iter", -1);

  std::string input_file = prefix + ".mbpt.h5";
  utils::check(std::filesystem::exists(input_file),
               "lr_gw_sigma_DeltaG_calc: Input checkpoint {} does not exist!", input_file);

  // Read IAFT from checkpoint
  imag_axes_ft::IAFT ft(imag_axes_ft::read_iaft(input_file, false));

  long ns = mf->nspin();
  long nkpts_ibz = mf->nkpts_ibz();
  long nbnd = mf->nbnd();
  long nt = ft.nt_f();

  // Load W and eps_inv_head using helper
  auto [dW_qtPQ, eps_inv_head] = load_W_and_eps_inv_head(*mpi, thc, input_file, input_grp, input_iter, nt);

  // Pre-transform W to R-space: transpose (q,t)→(t,R), with q→R FT
  auto dW_tRPQ = lr_precompute_W_tRPQ(dW_qtPQ, thc);
  dW_qtPQ.reset();  // free (q,t,P,Q) memory; no longer needed

  // Create lr_gw solver and compute ΔΣ
  solvers::lr_gw lr(& ft, q_pert);

  auto sDeltaSigma_tskij = math::shm::make_shared_array<Array_view_5D_t>(
      *mpi, {nt, ns, nkpts_ibz, nbnd, nbnd});

  lr.evaluate_sigma_DeltaG(sDeltaSigma_tskij, nda::array_view<ComplexType, 5>(DeltaG_tskij), dW_tRPQ, thc);
  mpi->comm.barrier();

  // Copy result
  nda::array<ComplexType, 5> DeltaSigma_tskij(nt, ns, nkpts_ibz, nbnd, nbnd);
  if (mpi->node_comm.root()) {
    DeltaSigma_tskij = sDeltaSigma_tskij.local();
  }
  mpi->comm.barrier();
  return DeltaSigma_tskij;
}


/**
 * Evaluate GW self-energy Σ = -G ⊙ W_c [+ div_corr] using W from file.
 *
 * Uses the standard gw_t code path (thc_solver_comm, not lr_thc_comm).
 * Used for finite-difference testing of LR-GW.
 */
template<typename eri_t>
nda::array<ComplexType, 5> gw_evaluate_sigma_calc(
    eri_t &eri, ptree const& pt,
    nda::array<ComplexType, 5> const& G_tskij,
    bool div_corr) {

  auto& thc = eri.corr_eri->get();
  auto mf = thc.MF();
  auto& mpi = thc.mpi();

  std::string err = std::string("gw_evaluate_sigma_calc - Incorrect input - ");
  auto prefix = io::get_value<std::string>(pt, "prefix", err + "prefix");
  auto input_grp = io::get_value_with_default<std::string>(pt, "input_type", "scf");
  auto input_iter = io::get_value_with_default<long>(pt, "input_iter", -1);

  std::string input_file = prefix + ".mbpt.h5";
  utils::check(std::filesystem::exists(input_file),
               "gw_evaluate_sigma_calc: Input checkpoint {} does not exist!", input_file);

  // Read IAFT from checkpoint
  imag_axes_ft::IAFT ft(imag_axes_ft::read_iaft(input_file, false));

  long ns = mf->nspin();
  long nkpts_ibz = mf->nkpts_ibz();
  long nbnd = mf->nbnd();
  long nt = ft.nt_f();

  // Load W and eps_inv_head using helper
  auto [dW_qtPQ, eps_inv_head] = load_W_and_eps_inv_head(*mpi, thc, input_file, input_grp, input_iter, nt);

  // Read overlap matrix from checkpoint for divergence correction
  auto sS_skij = math::shm::make_shared_array<Array_view_4D_t>(
      *mpi, {ns, nkpts_ibz, nbnd, nbnd});
  chkpt::read_ovlp(mpi->node_comm, prefix, sS_skij);

  // Use standard gw_t code path
  solvers::gw_t gw(&ft);

  auto sSigma_tskij = math::shm::make_shared_array<Array_view_5D_t>(
      *mpi, {nt, ns, nkpts_ibz, nbnd, nbnd});

  gw.eval_Sigma_all(G_tskij, dW_qtPQ, sSigma_tskij, thc, "R");
  if (div_corr) {
    gw.Sigma_div_correction(sSigma_tskij, G_tskij, sS_skij.local(), thc, eps_inv_head);
  }
  mpi->comm.barrier();

  // Copy result
  nda::array<ComplexType, 5> Sigma_tskij(nt, ns, nkpts_ibz, nbnd, nbnd);
  if (mpi->node_comm.root()) {
    Sigma_tskij = sSigma_tskij.local();
  }
  mpi->comm.barrier();
  return Sigma_tskij;
}


/**
 * Linear response polarization ΔP = -ΔG·G - G·ΔG (R-space).
 *
 * Creates lr_rpa_pi solver, calls evaluate_lr_Pi,
 * copies distributed result to regular array.
 */
template<typename eri_t>
nda::array<ComplexType, 4> lr_gw_Pi_calc(
    eri_t &eri,
    nda::array<double, 1> const& q_pert,
    nda::array<ComplexType, 5> const& G_tskij,
    nda::array<ComplexType, 5> const& DeltaG_tskij) {

  auto& thc = eri.corr_eri->get();
  auto mf = thc.MF();
  auto& mpi = thc.mpi();

  long nkpts = mf->nkpts();
  long Np = thc.Np();
  long nt = G_tskij.shape(0);
  long nt_half = (nt % 2 == 0) ? nt / 2 : nt / 2 + 1;

  // Create lr_rpa_pi solver and compute ΔP
  solvers::lr_rpa_pi lr(q_pert);

  auto dDeltaPi = lr.evaluate_lr_Pi(nda::array_view<ComplexType, 5>(G_tskij), nda::array_view<ComplexType, 5>(DeltaG_tskij), thc);
  mpi->comm.barrier();

  // Copy distributed result to regular array
  nda::array<ComplexType, 4> DeltaPi_tqPQ(nt_half, nkpts, Np, Np);
  DeltaPi_tqPQ() = 0.0;
  {
    auto loc = dDeltaPi.local();
    auto [t_origin, q_origin, P_origin, Q_origin] = dDeltaPi.origin();
    auto [nt_loc, nq_loc, NP_loc, NQ_loc] = dDeltaPi.local_shape();
    for (long it = 0; it < nt_loc; ++it)
      for (long iq = 0; iq < nq_loc; ++iq)
        for (long iP = 0; iP < NP_loc; ++iP)
          for (long iQ = 0; iQ < NQ_loc; ++iQ)
            DeltaPi_tqPQ(it + t_origin, iq + q_origin, iP + P_origin, iQ + Q_origin)
                = loc(it, iq, iP, iQ);
  }
  mpi->comm.all_reduce_in_place_n(DeltaPi_tqPQ.data(), DeltaPi_tqPQ.size(), std::plus<>{});
  mpi->comm.barrier();

  return DeltaPi_tqPQ;
}


/**
 * Evaluate standard RPA polarization P[G] (FD helper).
 *
 * Reads IAFT from checkpoint, constructs scr_coulomb_t, calls eval_Pi_qdep,
 * copies distributed result to regular array.
 */
template<typename eri_t>
nda::array<ComplexType, 4> gw_evaluate_Pi_calc(
    eri_t &eri, ptree const& pt,
    nda::array<ComplexType, 5> const& G_tskij) {

  auto& thc = eri.corr_eri->get();
  auto mf = thc.MF();
  auto& mpi = thc.mpi();

  std::string err = std::string("gw_evaluate_Pi_calc - Incorrect input - ");
  auto prefix = io::get_value<std::string>(pt, "prefix", err + "prefix");

  std::string input_file = prefix + ".mbpt.h5";
  utils::check(std::filesystem::exists(input_file),
               "gw_evaluate_Pi_calc: Input checkpoint {} does not exist!", input_file);

  // Read IAFT from checkpoint
  imag_axes_ft::IAFT ft(imag_axes_ft::read_iaft(input_file, false));

  long nkpts = mf->nkpts();
  long Np = thc.Np();

  // Require no symmetry (same constraint as lr_rpa_pi)
  utils::check(mf->nqpts() == mf->nqpts_ibz() and mf->nqpts() == nkpts,
               "gw_evaluate_Pi_calc: No symmetry required. nqpts={}, nqpts_ibz={}, nkpts={}",
               mf->nqpts(), mf->nqpts_ibz(), nkpts);

  long nt = ft.nt_f();
  utils::check(G_tskij.shape(0) == nt,
               "gw_evaluate_Pi_calc: G tau dimension {} != IAFT nt_f {}",
               G_tskij.shape(0), nt);
  long nt_half = (nt % 2 == 0) ? nt / 2 : nt / 2 + 1;

  // Use standard scr_coulomb_t to compute P
  solvers::scr_coulomb_t scr_eri(&ft, "rpa");

  auto dPi_tqPQ = scr_eri.eval_Pi_qdep(G_tskij, thc);
  mpi->comm.barrier();

  // Copy distributed result to regular array
  nda::array<ComplexType, 4> Pi_tqPQ(nt_half, nkpts, Np, Np);
  Pi_tqPQ() = 0.0;
  {
    auto loc = dPi_tqPQ.local();
    auto [t_origin, q_origin, P_origin, Q_origin] = dPi_tqPQ.origin();
    auto [nt_loc, nq_loc, NP_loc, NQ_loc] = dPi_tqPQ.local_shape();
    for (long it = 0; it < nt_loc; ++it)
      for (long iq = 0; iq < nq_loc; ++iq)
        for (long iP = 0; iP < NP_loc; ++iP)
          for (long iQ = 0; iQ < NQ_loc; ++iQ)
            Pi_tqPQ(it + t_origin, iq + q_origin, iP + P_origin, iQ + Q_origin)
                = loc(it, iq, iP, iQ);
  }
  mpi->comm.all_reduce_in_place_n(Pi_tqPQ.data(), Pi_tqPQ.size(), std::plus<>{});
  mpi->comm.barrier();

  return Pi_tqPQ;
}


/**
 * Linear response screened interaction ΔW = (Z+W_c) · ΔΠ · (Z+W_c).
 *
 * Reads W_c(τ) from thc_screened_interaction.h5 and IAFT from checkpoint,
 * then calls lr_scr_coulomb_t::lr_dyson_W_from_Pi_tau to compute ΔW_c(τ).
 */
template<typename eri_t>
nda::array<ComplexType, 4> lr_gw_W_calc(
    eri_t &eri, ptree const& pt,
    nda::array<double, 1> const& q_pert,  // Currently unused; reserved for future q→0 head corrections
    nda::array<ComplexType, 4> const& DeltaPi_tqPQ) {

  using local_Array_4D_t = nda::array<ComplexType, 4>;
  using math::nda::make_distributed_array;

  auto& thc = eri.corr_eri->get();
  auto mf = thc.MF();
  auto& mpi = thc.mpi();

  std::string err = std::string("lr_gw_W_calc - Incorrect input - ");
  auto prefix = io::get_value<std::string>(pt, "prefix", err + "prefix");
  auto input_grp = io::get_value_with_default<std::string>(pt, "input_type", "scf");
  auto input_iter = io::get_value_with_default<long>(pt, "input_iter", -1);

  std::string input_file = prefix + ".mbpt.h5";
  utils::check(std::filesystem::exists(input_file),
               "lr_gw_W_calc: Input checkpoint {} does not exist!", input_file);

  // Read IAFT from checkpoint
  imag_axes_ft::IAFT ft(imag_axes_ft::read_iaft(input_file, false));

  long nkpts = mf->nkpts();
  long NP = thc.Np();
  long nt = ft.nt_f();
  long nt_half = (nt % 2 == 0) ? nt / 2 : nt / 2 + 1;

  // Require no symmetry (same as gw_evaluate_W_from_Pi_calc)
  utils::check(mf->nqpts() == mf->nqpts_ibz() and mf->nqpts() == nkpts,
               "lr_gw_W_calc: No symmetry required. nqpts={}, nqpts_ibz={}, nkpts={}",
               mf->nqpts(), mf->nqpts_ibz(), nkpts);

  utils::check(DeltaPi_tqPQ.shape(0) == nt_half && DeltaPi_tqPQ.shape(1) == nkpts &&
               DeltaPi_tqPQ.shape(2) == NP && DeltaPi_tqPQ.shape(3) == NP,
               "lr_gw_W_calc: DeltaPi_tqPQ shape mismatch: expected ({},{},{},{}), got ({},{},{},{})",
               nt_half, nkpts, NP, NP,
               DeltaPi_tqPQ.shape(0), DeltaPi_tqPQ.shape(1),
               DeltaPi_tqPQ.shape(2), DeltaPi_tqPQ.shape(3));

  // Load W_c(τ) from thc_screened_interaction.h5 — stored as (q, t, P, Q)
  // eps_inv_head: not yet used here, will be needed for div_corr (TODO)
  auto [dW_qtPQ, eps_inv_head] = load_W_and_eps_inv_head(*mpi, thc, input_file, input_grp, input_iter, nt);

  // Transpose W_c from (q, t, P, Q) → (t, q, P, Q) for FT
  auto dW_tqPQ = utils::transpose_axes_01(dW_qtPQ, mpi->comm);
  dW_qtPQ.reset();

  // Convert ΔΠ regular array → distributed array (τ-dist)
  auto [pi_pgrid, pi_bsize] = utils::lr_W_q_local_dist(mpi->comm.size(), nt_half, NP);
  auto dDeltaPi_tqPQ = math::nda::make_distributed_array<local_Array_4D_t>(
      mpi->comm, pi_pgrid, {nt_half, nkpts, NP, NP}, pi_bsize);
  {
    nda::array<ComplexType, 4> pi_src(DeltaPi_tqPQ);
    mpi->comm.broadcast_n(pi_src.data(), pi_src.size(), 0);
    auto pi_loc = dDeltaPi_tqPQ.local();
    auto [po0, po1, po2, po3] = dDeltaPi_tqPQ.origin();
    auto [pn0, pn1, pn2, pn3] = dDeltaPi_tqPQ.local_shape();
    for (long i0 = 0; i0 < pn0; ++i0)
      for (long i1 = 0; i1 < pn1; ++i1)
        for (long i2 = 0; i2 < pn2; ++i2)
          for (long i3 = 0; i3 < pn3; ++i3)
            pi_loc(i0, i1, i2, i3) = pi_src(i0+po0, i1+po1, i2+po2, i3+po3);
    mpi->comm.barrier();
  }

  // LR Dyson: ΔΠ(τ) → ΔW_c(τ) via ΔW_c(iω) = W_full · ΔΠ · W_full
  solvers::lr_scr_coulomb_t lr_scr(&ft);
  auto dW_full_wqPQ = lr_scr.compute_W_full_omega(dW_tqPQ, thc);
  lr_scr.lr_dyson_W_from_Pi_tau(dDeltaPi_tqPQ, dW_full_wqPQ, thc);
  // dDeltaPi_tqPQ now contains ΔW_c(τ)
  mpi->comm.barrier();

  // Copy distributed result to regular array
  nda::array<ComplexType, 4> DeltaW_tqPQ(nt_half, nkpts, NP, NP);
  DeltaW_tqPQ() = 0.0;
  {
    auto loc = dDeltaPi_tqPQ.local();
    auto [t_origin, q_origin, P_origin, Q_origin] = dDeltaPi_tqPQ.origin();
    auto [nt_loc, nq_loc, NP_loc, NQ_loc] = dDeltaPi_tqPQ.local_shape();
    for (long it = 0; it < nt_loc; ++it)
      for (long iq = 0; iq < nq_loc; ++iq)
        for (long iP = 0; iP < NP_loc; ++iP)
          for (long iQ = 0; iQ < NQ_loc; ++iQ)
            DeltaW_tqPQ(it + t_origin, iq + q_origin, iP + P_origin, iQ + Q_origin)
                = loc(it, iq, iP, iQ);
  }
  mpi->comm.all_reduce_in_place_n(DeltaW_tqPQ.data(), DeltaW_tqPQ.size(), std::plus<>{});
  mpi->comm.barrier();

  return DeltaW_tqPQ;
}


/**
 * Evaluate screened interaction W_c from polarization Π (FD helper).
 *
 * Reads IAFT from checkpoint, converts input to distributed array,
 * calls scr_coulomb_t::dyson_W_from_Pi_tau<false>, copies result to regular array.
 */
template<typename eri_t>
nda::array<ComplexType, 4> gw_evaluate_W_from_Pi_calc(
    eri_t &eri, ptree const& pt,
    nda::array<ComplexType, 4> const& Pi_tqPQ) {

  using local_Array_4D_t = nda::array<ComplexType, 4>;
  using math::nda::make_distributed_array;

  auto& thc = eri.corr_eri->get();
  auto mf = thc.MF();
  auto& mpi = thc.mpi();

  std::string err = std::string("gw_evaluate_W_from_Pi_calc - Incorrect input - ");
  auto prefix = io::get_value<std::string>(pt, "prefix", err + "prefix");

  std::string input_file = prefix + ".mbpt.h5";
  utils::check(std::filesystem::exists(input_file),
               "gw_evaluate_W_from_Pi_calc: Input checkpoint {} does not exist!", input_file);

  // Read IAFT from checkpoint
  imag_axes_ft::IAFT ft(imag_axes_ft::read_iaft(input_file, false));

  long nkpts = mf->nkpts();
  long NP = thc.Np();
  long nt = ft.nt_f();
  long nt_half = (nt % 2 == 0) ? nt / 2 : nt / 2 + 1;

  utils::check(Pi_tqPQ.shape(0) == nt_half && Pi_tqPQ.shape(1) == nkpts &&
               Pi_tqPQ.shape(2) == NP && Pi_tqPQ.shape(3) == NP,
               "gw_evaluate_W_from_Pi_calc: Pi_tqPQ shape mismatch: expected ({},{},{},{}), got ({},{},{},{})",
               nt_half, nkpts, NP, NP,
               Pi_tqPQ.shape(0), Pi_tqPQ.shape(1),
               Pi_tqPQ.shape(2), Pi_tqPQ.shape(3));

  // Require no symmetry
  utils::check(mf->nqpts() == mf->nqpts_ibz() and mf->nqpts() == nkpts,
               "gw_evaluate_W_from_Pi_calc: No symmetry required. nqpts={}, nqpts_ibz={}, nkpts={}",
               mf->nqpts(), mf->nqpts_ibz(), nkpts);

  // Convert input to distributed array
  auto dPi_tqPQ = distribute_array_4D(mpi->comm, {nt_half, nkpts, NP, NP}, Pi_tqPQ);

  // Use scr_coulomb_t to solve W Dyson equation: Π → W_c(τ)
  solvers::scr_coulomb_t scr(&ft, "rpa");
  auto dW_tqPQ = scr.dyson_W_from_Pi_tau<false>(dPi_tqPQ, thc, true);
  mpi->comm.barrier();

  // Copy distributed result to regular array
  nda::array<ComplexType, 4> W_tqPQ(nt_half, nkpts, NP, NP);
  W_tqPQ() = 0.0;
  {
    auto loc = dW_tqPQ.local();
    auto [t_origin, q_origin, P_origin, Q_origin] = dW_tqPQ.origin();
    auto [nt_loc, nq_loc, NP_loc, NQ_loc] = dW_tqPQ.local_shape();
    for (long it = 0; it < nt_loc; ++it)
      for (long iq = 0; iq < nq_loc; ++iq)
        for (long iP = 0; iP < NP_loc; ++iP)
          for (long iQ = 0; iQ < NQ_loc; ++iQ)
            W_tqPQ(it + t_origin, iq + q_origin, iP + P_origin, iQ + Q_origin)
                = loc(it, iq, iP, iQ);
  }
  mpi->comm.all_reduce_in_place_n(W_tqPQ.data(), W_tqPQ.size(), std::plus<>{});
  mpi->comm.barrier();

  return W_tqPQ;
}


/**
 * Linear response GW self-energy term 2: -G ⊙ ΔW.
 *
 * Computes ΔΣ = -G ⊙ ΔW from a pre-computed DeltaW.
 * Uses lr_gw::evaluate_sigma_DeltaW for the G⊙ΔW convolution.
 * TODO: Divergence correction not yet implemented (see lr_gw.hpp class docstring).
 */
template<typename eri_t>
nda::array<ComplexType, 5> lr_gw_sigma_DeltaW_calc(
    eri_t &eri, ptree const& pt,
    nda::array<double, 1> const& q_pert,
    nda::array<ComplexType, 5> const& G_tskij,
    nda::array<ComplexType, 4> const& DeltaW_qtPQ) {

  auto& thc = eri.corr_eri->get();
  auto mf = thc.MF();
  auto& mpi = thc.mpi();

  std::string err = std::string("lr_gw_sigma_DeltaW_calc - Incorrect input - ");
  auto prefix = io::get_value<std::string>(pt, "prefix", err + "prefix");

  std::string input_file = prefix + ".mbpt.h5";
  utils::check(std::filesystem::exists(input_file),
               "lr_gw_sigma_DeltaW_calc: Input checkpoint {} does not exist!", input_file);

  // Read IAFT from checkpoint
  imag_axes_ft::IAFT ft(imag_axes_ft::read_iaft(input_file, false));

  long ns = mf->nspin();
  long nkpts_ibz = mf->nkpts_ibz();
  long nbnd = mf->nbnd();
  long nt = ft.nt_f();

  // Distribute DeltaW_qtPQ across MPI ranks (τ-dist with axes swapped for (q,t))
  long NP_dw = DeltaW_qtPQ.shape()[2];
  long nt_half_dw = DeltaW_qtPQ.shape()[1];
  auto [tq_pgrid_dw, tq_bsize_dw] = utils::lr_W_q_local_dist(mpi->comm.size(), nt_half_dw, NP_dw);
  std::array<long,4> qt_pgrid_dw = {tq_pgrid_dw[1], tq_pgrid_dw[0], tq_pgrid_dw[2], tq_pgrid_dw[3]};
  std::array<long,4> qt_bsize_dw = {tq_bsize_dw[1], tq_bsize_dw[0], tq_bsize_dw[2], tq_bsize_dw[3]};
  using local_Array_4D_t = nda::array<ComplexType, 4>;
  auto dDeltaW_qtPQ = math::nda::make_distributed_array<local_Array_4D_t>(
      mpi->comm, qt_pgrid_dw,
      {DeltaW_qtPQ.shape()[0], DeltaW_qtPQ.shape()[1],
       DeltaW_qtPQ.shape()[2], DeltaW_qtPQ.shape()[3]}, qt_bsize_dw);
  {
    nda::array<ComplexType, 4> dw_src(DeltaW_qtPQ);
    mpi->comm.broadcast_n(dw_src.data(), dw_src.size(), 0);
    auto dw_loc = dDeltaW_qtPQ.local();
    auto [dwo0, dwo1, dwo2, dwo3] = dDeltaW_qtPQ.origin();
    auto [dwn0, dwn1, dwn2, dwn3] = dDeltaW_qtPQ.local_shape();
    for (long i0 = 0; i0 < dwn0; ++i0)
      for (long i1 = 0; i1 < dwn1; ++i1)
        for (long i2 = 0; i2 < dwn2; ++i2)
          for (long i3 = 0; i3 < dwn3; ++i3)
            dw_loc(i0, i1, i2, i3) = dw_src(i0+dwo0, i1+dwo1, i2+dwo2, i3+dwo3);
    mpi->comm.barrier();
  }

  // Compute ΔΣ = -G ⊙ ΔW using lr_gw::evaluate_sigma_DeltaW
  app_log(1, "\n  lr_gw_sigma_DeltaW_calc: Computing -G ⊙ ΔW...");
  solvers::lr_gw lr(&ft, q_pert);

  auto sDeltaSigma_tskij = math::shm::make_shared_array<Array_view_5D_t>(
      *mpi, {nt, ns, nkpts_ibz, nbnd, nbnd});

  lr.evaluate_sigma_DeltaW(sDeltaSigma_tskij, nda::array_view<ComplexType, 5>(G_tskij), dDeltaW_qtPQ, thc);
  mpi->comm.barrier();

  // Copy result
  nda::array<ComplexType, 5> DeltaSigma_tskij(nt, ns, nkpts_ibz, nbnd, nbnd);
  if (mpi->node_comm.root()) {
    DeltaSigma_tskij = sDeltaSigma_tskij.local();
  }
  mpi->comm.barrier();
  return DeltaSigma_tskij;
}


/**
 * Evaluate GW self-energy with provided W and G (FD helper).
 *
 * Computes Σ = -G ⊙ W_c [+ div_corr] using provided G and W_c arrays.
 * Used for finite-difference testing of full LR-GW.
 */
template<typename eri_t>
nda::array<ComplexType, 5> gw_evaluate_sigma_with_W_calc(
    eri_t &eri, ptree const& pt,
    nda::array<ComplexType, 5> const& G_tskij,
    nda::array<ComplexType, 4> const& W_c_qtPQ,
    nda::array<ComplexType, 1> const& eps_inv_head,
    bool div_corr) {

  using local_Array_4D_t = nda::array<ComplexType, 4>;
  using math::nda::make_distributed_array;

  auto& thc = eri.corr_eri->get();
  auto mf = thc.MF();
  auto& mpi = thc.mpi();

  std::string err = std::string("gw_evaluate_sigma_with_W_calc - Incorrect input - ");
  auto prefix = io::get_value<std::string>(pt, "prefix", err + "prefix");

  std::string input_file = prefix + ".mbpt.h5";
  utils::check(std::filesystem::exists(input_file),
               "gw_evaluate_sigma_with_W_calc: Input checkpoint {} does not exist!", input_file);

  // Read IAFT from checkpoint
  imag_axes_ft::IAFT ft(imag_axes_ft::read_iaft(input_file, false));

  long ns = mf->nspin();
  long nkpts_ibz = mf->nkpts_ibz();
  long nbnd = mf->nbnd();
  long nt = ft.nt_f();

  // Convert W_c to distributed array — keep q-axis (axis 0) undivided
  auto dW_qtPQ = distribute_array_4D(mpi->comm,
      {W_c_qtPQ.shape(0), W_c_qtPQ.shape(1), W_c_qtPQ.shape(2), W_c_qtPQ.shape(3)},
      W_c_qtPQ, {true, false, false, false});

  // Read overlap matrix from checkpoint for divergence correction
  auto sS_skij = math::shm::make_shared_array<Array_view_4D_t>(
      *mpi, {ns, nkpts_ibz, nbnd, nbnd});
  chkpt::read_ovlp(mpi->node_comm, prefix, sS_skij);

  // Use standard gw_t code path
  solvers::gw_t gw(&ft);

  auto sSigma_tskij = math::shm::make_shared_array<Array_view_5D_t>(
      *mpi, {nt, ns, nkpts_ibz, nbnd, nbnd});

  gw.eval_Sigma_all(G_tskij, dW_qtPQ, sSigma_tskij, thc, "R");
  if (div_corr) {
    gw.Sigma_div_correction(sSigma_tskij, G_tskij, sS_skij.local(), thc, eps_inv_head);
  }
  mpi->comm.barrier();

  // Copy result
  nda::array<ComplexType, 5> Sigma_tskij(nt, ns, nkpts_ibz, nbnd, nbnd);
  if (mpi->node_comm.root()) {
    Sigma_tskij = sSigma_tskij.local();
  }
  mpi->comm.barrier();
  return Sigma_tskij;
}


// instantiations
using mpi3::communicator;

template std::tuple<nda::array<ComplexType, 4>, nda::array<ComplexType, 5> >
downfold_coulomb_impl(thc_reader_t &, MBState&& mb_state, ptree const& pt,
                   std::optional<std::map<std::string, nda::array<ComplexType, 5> > > local_polarizabilities);

template std::tuple<nda::array<ComplexType, 4>, nda::array<ComplexType, 5> >
downfold_coulomb_with_projector_from_h5(
  thc_reader_t &, ptree const&, std::optional<std::map<std::string, nda::array<ComplexType, 5> > >);

template std::tuple<nda::array<ComplexType, 4>, nda::array<ComplexType, 5> >
downfold_coulomb(thc_reader_t &, ptree const&,
              nda::array<ComplexType, 5> const&,
              nda::array<long, 3> const&,
              nda::array<RealType, 2> const&,
              std::optional<std::map<std::string, nda::array<ComplexType, 5> > >);

template void downfolding_2e(
     thc_reader_t&, ptree const&, std::optional<std::map<std::string, nda::array<ComplexType, 5> > >);

template void hf_downfold(thc_reader_t&, ptree const&);
template void gw_downfold(thc_reader_t&, ptree&);

#define MBPT_INST(HF, HARTREE, EXCHANGE, CORR) \
template void mbpt(std::string, \
     mb_eri_t<HF, HARTREE, EXCHANGE, CORR>&,    \
     ptree const&);                             \
template void mbpt(std::string, \
     mb_eri_t<HF, HARTREE, EXCHANGE, CORR>&, \
     ptree const&,                             \
     nda::array<ComplexType, 5> const&,  \
     nda::array<long, 3> const&,   \
     nda::array<RealType, 2> const&,           \
     std::optional<std::map<std::string, nda::array<ComplexType, 5> > >);

// All combinations of thc/chol for 4 eri slots
  MBPT_INST(thc_reader_t, thc_reader_t, thc_reader_t, thc_reader_t)
  MBPT_INST(thc_reader_t, thc_reader_t, thc_reader_t, chol_reader_t)
  MBPT_INST(thc_reader_t, thc_reader_t, chol_reader_t, thc_reader_t)
  MBPT_INST(thc_reader_t, thc_reader_t, chol_reader_t, chol_reader_t)
  MBPT_INST(thc_reader_t, chol_reader_t, thc_reader_t, thc_reader_t)
  MBPT_INST(thc_reader_t, chol_reader_t, thc_reader_t, chol_reader_t)
  MBPT_INST(thc_reader_t, chol_reader_t, chol_reader_t, thc_reader_t)
  MBPT_INST(thc_reader_t, chol_reader_t, chol_reader_t, chol_reader_t)
  MBPT_INST(chol_reader_t, thc_reader_t, thc_reader_t, thc_reader_t)
  MBPT_INST(chol_reader_t, thc_reader_t, thc_reader_t, chol_reader_t)
  MBPT_INST(chol_reader_t, thc_reader_t, chol_reader_t, thc_reader_t)
  MBPT_INST(chol_reader_t, thc_reader_t, chol_reader_t, chol_reader_t)
  MBPT_INST(chol_reader_t, chol_reader_t, thc_reader_t, thc_reader_t)
  MBPT_INST(chol_reader_t, chol_reader_t, thc_reader_t, chol_reader_t)
  MBPT_INST(chol_reader_t, chol_reader_t, chol_reader_t, thc_reader_t)
  MBPT_INST(chol_reader_t, chol_reader_t, chol_reader_t, chol_reader_t)

#undef MBPT_INST

// run_lr_calc instantiations (THC only — lr_hf and lr_gw require THC).
// Cholesky LR Dyson (one-shot, no HF/GW) was never used externally,
// so Cholesky ERI combinations are intentionally not instantiated here.
// If needed in the future, add Cholesky instantiations and guard the
// lr_hf/lr_gw code paths behind THC-only checks.
template std::tuple<int, double> run_lr_calc(
    mb_eri_t<thc_reader_t, thc_reader_t, thc_reader_t, thc_reader_t>&,
    ptree const&,
    nda::array<double, 1> const&,
    nda::array<ComplexType, 4> const&,
    bool, bool, lr_gw_update_mode, int, double, bool, const lr_iter_params&,
    const nda::array<ComplexType, 4>*, const nda::array<ComplexType, 4>*);

// lr_gw_sigma_DeltaG_calc instantiations (THC only — lr_gw requires THC)
template nda::array<ComplexType, 5> lr_gw_sigma_DeltaG_calc(
    mb_eri_t<thc_reader_t, thc_reader_t, thc_reader_t, thc_reader_t>&,
    ptree const&,
    nda::array<double, 1> const&,
    nda::array<ComplexType, 5> const&);

// gw_evaluate_sigma_calc instantiations (THC only)
template nda::array<ComplexType, 5> gw_evaluate_sigma_calc(
    mb_eri_t<thc_reader_t, thc_reader_t, thc_reader_t, thc_reader_t>&,
    ptree const&,
    nda::array<ComplexType, 5> const&,
    bool);

// lr_gw_Pi_calc instantiations (THC only — lr_gw requires THC)
template nda::array<ComplexType, 4> lr_gw_Pi_calc(
    mb_eri_t<thc_reader_t, thc_reader_t, thc_reader_t, thc_reader_t>&,
    nda::array<double, 1> const&,
    nda::array<ComplexType, 5> const&,
    nda::array<ComplexType, 5> const&);

// gw_evaluate_Pi_calc instantiations (THC only)
template nda::array<ComplexType, 4> gw_evaluate_Pi_calc(
    mb_eri_t<thc_reader_t, thc_reader_t, thc_reader_t, thc_reader_t>&,
    ptree const&,
    nda::array<ComplexType, 5> const&);

// lr_gw_W_calc instantiations (THC only)
template nda::array<ComplexType, 4> lr_gw_W_calc(
    mb_eri_t<thc_reader_t, thc_reader_t, thc_reader_t, thc_reader_t>&,
    ptree const&,
    nda::array<double, 1> const&,
    nda::array<ComplexType, 4> const&);

// gw_evaluate_W_from_Pi_calc instantiations (THC only)
template nda::array<ComplexType, 4> gw_evaluate_W_from_Pi_calc(
    mb_eri_t<thc_reader_t, thc_reader_t, thc_reader_t, thc_reader_t>&,
    ptree const&,
    nda::array<ComplexType, 4> const&);

// lr_gw_sigma_DeltaW_calc instantiations (THC only)
template nda::array<ComplexType, 5> lr_gw_sigma_DeltaW_calc(
    mb_eri_t<thc_reader_t, thc_reader_t, thc_reader_t, thc_reader_t>&,
    ptree const&,
    nda::array<double, 1> const&,
    nda::array<ComplexType, 5> const&,
    nda::array<ComplexType, 4> const&);

// gw_evaluate_sigma_with_W_calc instantiations (THC only)
template nda::array<ComplexType, 5> gw_evaluate_sigma_with_W_calc(
    mb_eri_t<thc_reader_t, thc_reader_t, thc_reader_t, thc_reader_t>&,
    ptree const&,
    nda::array<ComplexType, 5> const&,
    nda::array<ComplexType, 4> const&,
    nda::array<ComplexType, 1> const&,
    bool);

}
