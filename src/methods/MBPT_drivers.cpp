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



#include <string>

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
#include "methods/SCF/lr_state.hpp"
#include "methods/SCF/lr_driver.hpp"
#include "methods/tools/chkpt_utils.h"
#include "numerics/shared_array/shared_array_io.hpp"
#include "utilities/lr_utils.hpp"
#include "utilities/proc_meminfo.hpp"
#include "utilities/Timer.hpp"
#include "methods/SCF/dca_dyson.h"
#include "methods/SCF/simple_dyson.h"
#include "methods/embedding/embed_t.h"
#include "methods/embedding/embed_eri_t.h"
#include "numerics/imag_axes_ft/IAFT.hpp"
#include "numerics/iter_scf/iter_scf_utils.hpp"

#include "SCF/scf_driver.hpp"
#include "MBPT_drivers.h"

namespace mpi3 = boost::mpi3;
namespace methods
{

inline std::string resolve_mbpt_output_stem(ptree const& pt) {
  auto output_opt = pt.get_optional<std::string>("output");
  if (output_opt and !output_opt->empty()) return output_opt.value();

  std::string err = std::string("Incorrect input - ");
  auto outdir = io::get_value_with_default<std::string>(pt, "outdir", "./");
  auto prefix = io::get_value<std::string>(pt,"prefix",err+"prefix");
  if (prefix.empty()) {
    utils::check(false, "Incorrect input - prefix cannot be empty string.");
  }
  return outdir + "/" + prefix;
}

/**
 * Factory function for creating simple_dyson with conditional H0 source.
 *
 * @param h0_source     - [INPUT] Source of H0: "compute" or "checkpoint"
 * @param mf_ptr        - [INPUT] Mean-field object pointer
 * @param ft_ptr        - [INPUT] IAFT pointer
 * @param chkpt_prefix  - [INPUT] Checkpoint file prefix (used if h0_source="checkpoint")
 * @param mu_tol        - [INPUT] Chemical potential tolerance (default 1e-9)
 * @param mu_update_alg - [INPUT] Chemical potential update algorithm
 * @return simple_dyson object
 */
inline simple_dyson make_dyson_with_h0_source(
    std::string_view h0_source,
    mf::MF* mf_ptr,
    imag_axes_ft::IAFT* ft_ptr,
    const std::string& chkpt_prefix,
    double mu_tol = 1e-9,
    std::string mu_update_alg = "bisection") {
  if (h0_source == "checkpoint") {
    app_log(1, "  H0 source                = checkpoint ({}.mbpt.h5)", chkpt_prefix);
    return simple_dyson(mf_ptr, ft_ptr, chkpt_prefix, mu_tol, mu_update_alg);
  } else {
    app_log(1, "  H0 source                = computed from orbitals");
    return simple_dyson(mf_ptr, ft_ptr, mu_tol, mu_update_alg);
  }
}

/**
 * Compute a 4D processor grid for nproc ranks and a given global shape.
 * Axes 0 and 1 get pool-style splits; the remainder goes to axes 2 and 3.
 * Axes flagged in skip_axes stay undivided.
 *
 * @return processor grid {n0, n1, n2, n3} with n0*n1*n2*n3 == nproc
 */
inline std::array<long, 4> compute_proc_grid_4D(long nproc, std::array<long, 4> shape,
                                                 std::array<bool, 4> skip_axes = {false, false, false, false}) {
  long np = nproc;
  std::array<long, 4> grid = {1, 1, 1, 1};
  for (int ax = 0; ax < 2; ++ax) {
    if (!skip_axes[ax]) {
      grid[ax] = utils::find_proc_grid_max_npools(np, shape[ax], 0.2);
      np /= grid[ax];
    }
  }
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
    
    imag_axes_ft::IAFT ft(pt, false, mf::wmax_from_mf(*mf));
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
 * Many-body perturbation calculations from a given mean-field and ERI objects with arguments in property tree.
 * Optional arguments (with default values):
 *  - beta: "1000" Inverse temperature (a.u.)
 *  - wmax: Optional. Frequency cutoff for the IAFT grids (a.u.).
 *          If not provided, wmax is estimated from mean_field. 
 *  - iaft_prec: "high" Precision of IAFT grids. {choices: "high", "medium", "low"}
 *  - div_treatment: "gygi" Divergent treatment for Coulomb kernel. {choices: "ignore_g0", "gygi"}
 *  - hf_div_treatment: "gygi" Divergent treatment for Coulomb kernel in HF. {choices: "ignore_g0", "gygi"}
 *  - niter: "1" Number of iterations in the self-consistent loop.
 *  - conv_thr: "1e-9" Convergence threshold for the self-consistent loop.
 *  - const_mu: "false" Fix the chemical potential during the self-consistent loop.
 *  - output: Optional legacy output flag. If present, this is used directly.
 *  - outdir: "./" Output directory used when output is not provided.
 *  - prefix: "bdft.mbpt" Prefix used when output is not provided.
 *  - restart: "false" Restart from a previous bdft.scf calculation.
 *  - t_prescreen_thresh: "0.0" Threshold for prescreening in time (GF2 only for now)
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
  auto mu_tol = io::get_value_with_default<double>(pt,"mu_tolerance", 1e-9);
  auto output = resolve_mbpt_output_stem(pt);
  auto mu_update_alg = io::get_value_with_default<std::string>(pt, "mu_update_alg", "midpoint");
  auto compute_exchange = io::get_value_with_default<bool>(pt,"compute_exchange",true);
  auto h0_source = io::get_value_with_default<std::string>(pt, "h0_source", "compute");
  io::tolower(h0_source);
  utils::check(h0_source == "compute" || h0_source == "checkpoint",
      "mbpt: unknown h0_source=\"{}\". Valid options: \"compute\", \"checkpoint\"", h0_source);

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

  imag_axes_ft::IAFT ft(
    !restart ? imag_axes_ft::IAFT(pt, false, mf::wmax_from_mf(*mf))
             : imag_axes_ft::read_iaft(output+".mbpt.h5", false)
  );

  std::unique_ptr<iter_scf::iter_scf_t> iter_solver;

  using namespace solvers;
  hf_t hf(hf_div_treatment);
  if(solver_type == "rpa") {

    simple_dyson dyson = make_dyson_with_h0_source(h0_source, mf.get(), &ft, output, mu_tol, mu_update_alg);
    gw_t gw(&ft, div_treatment, output);
    MBState mb_state(mpi, ft, output);
    rpa_loop(mb_state, dyson, eri, ft, mb_solver_t(&hf,&gw));

  } else if(solver_type == "hf") {

    simple_dyson dyson = make_dyson_with_h0_source(h0_source, mf.get(), &ft, output, mu_tol, mu_update_alg);
    if (io::get_value_with_default<bool>(pt,"iter_alg.enable", true)) {
      iter_solver = std::make_unique<iter_scf::iter_scf_t>(iter_scf::make_iter_scf(pt));
    } else {
      iter_solver = nullptr;
    }
    MBState mb_state(mpi, ft, output);
    scf_loop(mb_state, dyson, eri, ft, mb_solver_t(&hf),
             iter_solver.get(), niter, restart, conv_thr, const_mu,
             greens_func_source, greens_func_iteration, 
             /*eval_thermodynamics=*/io::get_value_with_default<bool>(pt, "eval_thermodynamics", false),
             /*compute_exchange=*/compute_exchange);

  } else if(solver_type == "gw") {

    auto screen_type = io::get_value_with_default<std::string>(pt,"screen_type", "rpa");

    simple_dyson dyson = make_dyson_with_h0_source(h0_source, mf.get(), &ft, output, mu_tol, mu_update_alg);
    if (io::get_value_with_default<bool>(pt,"iter_alg.enable", true)) {
      iter_solver = std::make_unique<iter_scf::iter_scf_t>(iter_scf::make_iter_scf(pt));
    } else {
      iter_solver = nullptr;
    }
    solvers::scr_coulomb_t scr_eri(&ft, screen_type, div_treatment);
    solvers::gw_t gw(&ft, div_treatment, output);
    if (screen_type.substr(0,8)=="gw_edmft") {

      auto wannier_file = io::get_value<std::string>(pt,"wannier_file",err+"wannier_file");
      auto trans_home_cell = io::get_value_with_default<bool>(pt,"translate_home_cell",false);

      MBState mb_state(ft, output, mf, wannier_file, trans_home_cell);
      auto dump_w_to_h5 = io::get_value_with_default<bool>(pt,"dump_w_to_h5", false);
      scf_loop(mb_state, dyson, eri, ft, mb_solver_t(&hf, &gw, &scr_eri),
               iter_solver.get(), niter, restart, conv_thr, const_mu,
               greens_func_source, greens_func_iteration, 
               /*eval_thermodynamics=*/io::get_value_with_default<bool>(pt, "eval_thermodynamics", false),
               /*compute_exchange=*/compute_exchange, /*keep_w=*/dump_w_to_h5);

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
      auto dump_w_to_h5 = io::get_value_with_default<bool>(pt,"dump_w_to_h5", false);
      scf_loop(mb_state, dyson, eri, ft, mb_solver_t(&hf, &gw, &scr_eri),
               iter_solver.get(), niter, restart, conv_thr, const_mu,
               greens_func_source, greens_func_iteration, 
               /*eval_thermodynamics=*/io::get_value_with_default<bool>(pt, "eval_thermodynamics", false),
               /*compute_exchange=*/compute_exchange, /*keep_w=*/dump_w_to_h5);

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

    auto eval_thermodynamics = io::get_value_with_default<bool>(pt, "eval_thermodynamics", false);
    if (eval_thermodynamics) {
      utils::check(false, "thermodynamic-property (grand potential) evaluation is not yet supported for the GF2 solver.");
    }

    simple_dyson dyson = make_dyson_with_h0_source(h0_source, mf.get(), &ft, output, mu_tol, mu_update_alg);
    if (io::get_value_with_default<bool>(pt,"iter_alg.enable", true)) {
      iter_solver = std::make_unique<iter_scf::iter_scf_t>(iter_scf::make_iter_scf(pt));
    } else {
      iter_solver = nullptr;
    }
    solvers::gf2_t gf2(mf.get(), &ft, div_treatment,
                       gf2_direct_type, gf2_exchange_alg, gf2_exchange_type, output,
                       gf2_save_C, gf2_sosex_save_memory);
    gf2.t_thresh() = t_prescreen_thresh;

    MBState mb_state(mpi, ft, output);

    if (gf2_direct_type == "gf2") {
      scf_loop(mb_state, dyson, eri, ft, mb_solver_t(&hf, &gf2),
               iter_solver.get(), niter, restart, conv_thr, const_mu,
               greens_func_source, greens_func_iteration,
               /*eval_thermodynamics=*/eval_thermodynamics, /*compute_exchange=*/compute_exchange);
    } else {
      solvers::scr_coulomb_t scr_eri(&ft, "rpa", div_treatment);
      scf_loop(mb_state, dyson, eri, ft, mb_solver_t(&hf, &gf2, &scr_eri),
               iter_solver.get(), niter, restart, conv_thr, const_mu,
               greens_func_source, greens_func_iteration,
               /*eval_thermodynamics=*/eval_thermodynamics, /*compute_exchange=*/compute_exchange);
    }

  } else if(solver_type == "gw_dca") {

    utils::check(false, "mbpt: gw_dca is not implemented!");
    /*ptree dca_pt = io::find_child(pt, "gw_mean_field");
    std::string mf_type = (mf.mf_type()==mf::qe_source)?
        "qe" : (mf.mf_type()==mf::pyscf_source)? "pyscf" : "bdft";
    mf::MF dca_mf(mf::make_MF(mpi, dca_pt, mf_type));
    dca_dyson dyson(mpi, &mf, &ft, dca_mf);
    solvers::gw_t gw(&ft, div_treatment, output);
    scf_loop(dyson, eri, ft, mb_solver_t(&hf,&gw), nullptr,
             output, niter, restart, conv_thr, const_mu);*/

  } else if (solver_type == "qphf") {

    if (io::get_value_with_default<bool>(pt,"iter_alg.enable", true)) {
      iter_solver = std::make_unique<iter_scf::iter_scf_t>(iter_scf::make_iter_scf(pt));
    } else {
      iter_solver = nullptr;
    }
    MBState mb_state(mpi, ft, output);
    qp_params_t qp_params;
    qp_params.mu_tolerance = mu_tol;
    qp_params.mu_update_alg = mu_update_alg;
    qp_scf_loop(mb_state, eri, ft, qp_params, mb_solver_t(&hf), iter_solver.get(),
                niter, restart, conv_thr, /*compute_exchange=*/compute_exchange);

  } else if (solver_type == "evgw") {

    auto keep_scr_coulomb_fixed = io::get_value_with_default<bool>(pt,"keep_scr_coulomb_fixed", false);
    auto qp_type = io::get_value_with_default<std::string>(pt,"qp_type","sc");
    auto ac_alg  = io::get_value_with_default<std::string>(pt,"ac_alg","pade");
    auto eta     = io::get_value_with_default<double>(pt,"eta", M_PI/ft.beta());
    auto Nfit    = io::get_value_with_default<int>(pt,"Nfit",18);
    io::tolower(ac_alg);
    io::tolower(qp_type);
    qp_params_t qp_params(qp_type, ac_alg, Nfit, eta, conv_thr, "evscf", keep_scr_coulomb_fixed,
                          "fermi", mu_tol, mu_update_alg);
    if (io::get_value_with_default<bool>(pt,"iter_alg.enable", true)) {
      iter_solver = std::make_unique<iter_scf::iter_scf_t>(iter_scf::make_iter_scf(pt, 0.7, true));
    } else {
      iter_solver = nullptr;
    }
    solvers::scr_coulomb_t scr_eri(&ft, "rpa", div_treatment);
    solvers::gw_t gw(&ft, div_treatment, output);
    MBState mb_state(mpi, ft, output);
    qp_scf_loop(mb_state, eri, ft, qp_params, mb_solver_t(&hf,&gw,&scr_eri), iter_solver.get(),
                niter, restart, conv_thr, /*compute_exchange=*/compute_exchange);

  } else if (solver_type == "qpgw") {

    auto ac_alg  = io::get_value_with_default<std::string>(pt,"ac_alg","pade");
    auto eta     = io::get_value_with_default<double>(pt,"eta", M_PI/ft.beta());
    auto Nfit    = io::get_value_with_default<int>(pt,"Nfit",18);
    auto off_diag_mode = io::get_value_with_default<std::string>(pt,"off_diag_mode","fermi");
    io::tolower(ac_alg);
    io::tolower(off_diag_mode);
    utils::check(off_diag_mode=="fermi" or off_diag_mode=="qp_energy",
                 "unknown off_diag_mode: {}. Valid options are \"fermi\" and \"qp_energy\"");
    qp_params_t qp_params("sc", ac_alg, Nfit, eta, 1e-8, "qpscf", false, off_diag_mode,
                          mu_tol, mu_update_alg);
    if (io::get_value_with_default<bool>(pt,"iter_alg.enable", true)) {
      iter_solver = std::make_unique<iter_scf::iter_scf_t>(iter_scf::make_iter_scf(pt));
    } else {
      iter_solver = nullptr;
    }
    solvers::scr_coulomb_t scr_eri(&ft, "rpa", div_treatment);
    solvers::gw_t gw(&ft, div_treatment, output);
    MBState mb_state(mpi, ft, output);
    qp_scf_loop(mb_state, eri, ft, qp_params, mb_solver_t(&hf,&gw,&scr_eri), iter_solver.get(),
                niter, restart, conv_thr, /*compute_exchange=*/compute_exchange);

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
  auto mu_tol = io::get_value_with_default<double>(pt,"mu_tolerance", 1e-9);
  auto output = resolve_mbpt_output_stem(pt);
  auto mu_update_alg = io::get_value_with_default<std::string>(pt, "mu_update_alg", "midpoint");
  auto compute_exchange = io::get_value_with_default<bool>(pt,"compute_exchange",true);
  auto h0_source = io::get_value_with_default<std::string>(pt, "h0_source", "compute");
  io::tolower(h0_source);
  utils::check(h0_source == "compute" || h0_source == "checkpoint",
      "mbpt: unknown h0_source=\"{}\". Valid options: \"compute\", \"checkpoint\"", h0_source);

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

  imag_axes_ft::IAFT ft(
    !restart ? imag_axes_ft::IAFT(pt, false, mf::wmax_from_mf(*mf))
             : imag_axes_ft::read_iaft(output+".mbpt.h5", false)
  );

  std::unique_ptr<iter_scf::iter_scf_t> iter_solver;

  using namespace solvers;
  hf_t hf(hf_div_treatment);
  if (solver_type == "gw") {

    auto screen_type = io::get_value_with_default<std::string>(pt,"screen_type", "rpa");

    simple_dyson dyson = make_dyson_with_h0_source(h0_source, mf.get(), &ft, output, mu_tol, mu_update_alg);
    if (io::get_value_with_default<bool>(pt,"iter_alg.enable", true)) {
      iter_solver = std::make_unique<iter_scf::iter_scf_t>(iter_scf::make_iter_scf(pt));
    } else {
      iter_solver = nullptr;
    }
    solvers::scr_coulomb_t scr_eri(&ft, screen_type, div_treatment);
    solvers::gw_t gw(&ft, div_treatment, output);
    MBState mb_state(ft, output, mf, projector_ksIai, band_window, kpts_crys, trans_home_cell, false);
    if (local_polarizabilities) {
      mb_state.set_local_polarizabilities(std::move(local_polarizabilities.value()));
      local_polarizabilities.reset();
    }

    scf_loop(mb_state, dyson, eri, ft, mb_solver_t(&hf, &gw, &scr_eri),
             iter_solver.get(), niter, restart, conv_thr, const_mu,
             greens_func_source, greens_func_iteration, 
             /*eval_thermodynamics=*/io::get_value_with_default<bool>(pt, "eval_thermodynamics", false),
             /*compute_exchange=*/compute_exchange);

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
    qp_params_t qp_params("sc", ac_alg,
                io::get_value_with_default<int>(pt,"Nfit",30),
                io::get_value_with_default<double>(pt,"eta", M_PI/ft.beta()),
                1e-8, "qpscf", false, off_diag_mode);
    embed.downfolding(mb_state, pt, &qp_params);
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
  embed_eri_t embed_eri(*mf, div_treatment, bare_div_treatment, "default");
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

  embed_eri_t embed_eri(*mf, div_treatment, bare_div_treatment, "default");

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
  imag_axes_ft::IAFT ft(pt, true, mf::wmax_from_mf(*mf));
  hamilt::pseudopot psp(*mf);
  write_mf_data(*mf, ft, psp, output);
  MBState mb_state(ft, output, mf, wannier_file, trans_home_cell, false);

  // Two-body Hamiltonian
  embed_eri_t embed_eri(*mf, "ignore_g0", hf_div_treatment, "model_static");
  embed_eri.downfolding_crpa(eri, mb_state, pt, "bare", factorization_type,
                             io::get_value_with_default<double>(pt, "thresh", 1e-6));

  // One-body Hamiltonian
  embed_t embed(*mf, wannier_file, trans_home_cell);
  embed.hf_downfolding(outdir, prefix, eri, ft,
                       io::get_value_with_default<bool>(pt, "force_real", true),
                       hf_div_treatment);

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
  embed_eri_t embed_eri(*mf, div_treatment, hf_div_treatment, "model_static");
  embed_eri.downfolding_crpa(eri, mb_state, pt, "crpa", factorization_type,
                             io::get_value_with_default<double>(pt, "thresh", 1e-6));

  // one body hamiltonian
  auto ac_alg  = io::get_value_with_default<std::string>(pt,"ac_alg","pade");
  auto off_diag_mode = io::get_value_with_default<std::string>(pt,"off_diag_mode","qp_energy");
  io::tolower(ac_alg);
  io::tolower(off_diag_mode);
  utils::check(off_diag_mode=="fermi" or off_diag_mode=="qp_energy",
               "unknown off_diag_mode: {}. Valid options are \"fermi\" and \"qp_energy\"");
  qp_params_t qp_params(
      "sc", ac_alg,
      io::get_value_with_default<int>(pt,"Nfit",30),
      io::get_value_with_default<double>(pt,"eta", M_PI/ft.beta()),
      1e-8, "qpscf", false, off_diag_mode);
  embed_t embed(*mf, wannier_file, trans_home_cell);
  pt.put("update_dc", true);
  pt.put("dc_type", "gw");
  embed.downfolding(mb_state, pt, &qp_params, "model_static");
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
                            io::get_value_with_default<double>(pt,"mu_tolerance", 1e-9),
                            io::get_value_with_default<std::string>(pt, "mu_update_alg", "midpoint"));

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
                            io::get_value_with_default<double>(pt,"mu_tolerance", 1e-9),
                            io::get_value_with_default<std::string>(pt, "mu_update_alg", "midpoint"));

  embed_t embed(*mf);
  embed.dmft_embed(mb_state, dyson, iter_solver.get(),
                   io::get_value_with_default<bool>(pt,"qp_approx_mbpt",false),
                   io::get_value_with_default<bool>(pt,"corr_only",false));
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

/**
 * Unified linear response calculation.
 *
 * Runs the LR SCF loop with configurable Hartree, Exchange, and GW self-energy components:
 *   ΔH0 → ΔG → ΔDm → [ΔF] → [ΔΣ] → ΔG → ... (iterate until convergence)
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
                                     std::optional<nda::array<ComplexType, 4>> const& DeltaH0_skij_root,
                                     bool include_hartree,
                                     bool include_exchange,
                                     lr_gw_update_mode gw_mode,
                                     int max_iter,
                                     double tol,
                                     bool fix_density,
                                     const lr_iter_params& iter_params,
                                     std::optional<nda::array<ComplexType, 4>> const& DeltaX_left_root,
                                     std::optional<nda::array<ComplexType, 4>> const& DeltaX_right_root,
                                     std::optional<nda::array<ComplexType, 3>> const& DeltaV_qPQ_root) {
  auto mf = eri.corr_eri->get().MF();
  auto& mpi = eri.corr_eri->get().mpi();

  std::string err = std::string("run_lr_calc - Incorrect input - ");
  auto prefix = io::get_value<std::string>(pt, "prefix", err + "prefix");
  auto output = io::get_value_with_default<std::string>(pt, "output", prefix);
  auto input_grp = io::get_value_with_default<std::string>(pt, "input_type", "scf");
  auto input_iter = io::get_value_with_default<long>(pt, "input_iter", -1);
  auto h0_source = io::get_value_with_default<std::string>(pt, "h0_source", "compute");
  auto div_corr = io::get_value_with_default<bool>(pt, "div_corr", true);

  std::string input_file = prefix + ".mbpt.h5";
  utils::check(std::filesystem::exists(input_file),
               "run_lr_calc: Input checkpoint {} does not exist!", input_file);

  utils::memlog("run_lr_calc: entry");

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

  utils::TimerManager lr_init_timer;
  for (auto& v : {"LR_INIT", "LR_INIT_DYSON", "LR_INIT_READ_SCF",
                  "LR_INIT_UPDATE_G", "LR_INIT_LOAD_W"}) {
    lr_init_timer.add(v);
  }
  lr_init_timer.start("LR_INIT");

  // Read IAFT from checkpoint
  imag_axes_ft::IAFT ft(imag_axes_ft::read_iaft(input_file, false));

  // Create simple_dyson
  utils::check(h0_source == "compute" || h0_source == "checkpoint",
               "run_lr_calc: h0_source must be 'compute' or 'checkpoint', got '{}'", h0_source);
  lr_init_timer.start("LR_INIT_DYSON");
  simple_dyson dyson = make_dyson_with_h0_source(h0_source, mf.get(), &ft, prefix);
  lr_init_timer.stop("LR_INIT_DYSON");

  utils::memlog("run_lr_calc: after simple_dyson ctor");

  // Allocate shared arrays for the unperturbed state
  auto sF_skij = math::shm::make_shared_array<Array_view_4D_t>(
      *mpi, {mf->nspin(), mf->nkpts_ibz(), mf->nbnd(), mf->nbnd()});
  auto sDm_skij = math::shm::make_shared_array<Array_view_4D_t>(
      *mpi, {mf->nspin(), mf->nkpts_ibz(), mf->nbnd(), mf->nbnd()});
  auto sG_tskij = math::shm::make_shared_array<Array_view_5D_t>(
      *mpi, {ft.nt_f(), mf->nspin(), mf->nkpts_ibz(), mf->nbnd(), mf->nbnd()});
  auto sSigma_tskij = math::shm::make_shared_array<Array_view_5D_t>(
      *mpi, {ft.nt_f(), mf->nspin(), mf->nkpts_ibz(), mf->nbnd(), mf->nbnd()});

  utils::memlog("run_lr_calc: after sF/sDm/sG/sSigma alloc");

  // Read F and Sigma from checkpoint
  double mu = 0.0;
  lr_init_timer.start("LR_INIT_READ_SCF");
  chkpt::read_scf(mpi->node_comm, sF_skij, sSigma_tskij, mu,
                  prefix, input_grp, input_iter);
  mpi->comm.barrier();
  lr_init_timer.stop("LR_INIT_READ_SCF");

  // Compute G from F and Sigma via Dyson equation
  app_log(2, "Computing unperturbed Green's function from checkpoint...");
  app_log(2, "  h0_source = {}", h0_source);
  app_log(2, "  mu = {:.8f}", mu);
  lr_init_timer.start("LR_INIT_UPDATE_G");
  update_G(dyson, *mf, ft, sDm_skij, sG_tskij, sF_skij, sSigma_tskij, mu, true);
  mpi->comm.barrier();
  lr_init_timer.stop("LR_INIT_UPDATE_G");

  // Validate DeltaH0 shape on root before broadcasting
  if (mpi->comm.root()) {
    utils::check(DeltaH0_skij_root.has_value(),
                 "run_lr_calc: DeltaH0_skij must be provided on the MPI global root");
    auto const& dh = *DeltaH0_skij_root;
    utils::check(dh.shape(0) == mf->nspin() &&
                 dh.shape(1) == mf->nkpts_ibz() &&
                 dh.shape(2) == mf->nbnd() &&
                 dh.shape(3) == mf->nbnd(),
                 "run_lr_calc: DeltaH0_skij shape mismatch: expected ({},{},{},{}), got ({},{},{},{})",
                 mf->nspin(), mf->nkpts_ibz(), mf->nbnd(), mf->nbnd(),
                 dh.shape(0), dh.shape(1), dh.shape(2), dh.shape(3));
  }

  // LR state: perturbation + response quantities
  lr_state_t lr_state;
  lr_state.q_vec.emplace(q_vec);
  lr_state.sDeltaH0_skij.emplace(
      math::shm::make_shared_from_root_input<ComplexType, 4>(*mpi, DeltaH0_skij_root));
  lr_state.sDeltaG_tskij.emplace(math::shm::make_shared_array<Array_view_5D_t>(
      *mpi, {ft.nt_f(), mf->nspin(), mf->nkpts_ibz(), mf->nbnd(), mf->nbnd()}));
  lr_state.sDeltaDm_skij.emplace(math::shm::make_shared_array<Array_view_4D_t>(
      *mpi, {mf->nspin(), mf->nkpts_ibz(), mf->nbnd(), mf->nbnd()}));
  lr_state.sDeltaF_skij.emplace(math::shm::make_shared_array<Array_view_4D_t>(
      *mpi, {mf->nspin(), mf->nkpts_ibz(), mf->nbnd(), mf->nbnd()}));
  utils::memlog("run_lr_calc: after sDeltaG/sDeltaDm/sDeltaF/sDeltaH0 alloc");
  // Only allocate ΔΣ array when GW is active
  bool include_gw_sigma = (gw_mode != lr_gw_update_mode::none);
  if (include_gw_sigma) {
    lr_state.sDeltaSigma_tskij.emplace(math::shm::make_shared_array<Array_view_5D_t>(
        *mpi, {ft.nt_f(), mf->nspin(), mf->nkpts_ibz(), mf->nbnd(), mf->nbnd()}));
  }

  // GW inputs (W_c + eps_inv_head) are loaded here once the LR-GW steps land.
  utils::check(gw_mode == lr_gw_update_mode::none,
               "run_lr_calc: gw_mode != none not yet ported.");

  // Bcast presence flag for optional inputs so non-root ranks know whether
  // to participate in make_shared_from_root_input.
  using sArray_4D_t = math::shm::shared_array<Array_view_4D_t>;
  std::optional<sArray_4D_t> opt_sDeltaX_left, opt_sDeltaX_right;
  bool has_deltax_root = false;
  if (mpi->comm.root()) {
    has_deltax_root = DeltaX_left_root.has_value() && DeltaX_right_root.has_value();
  }
  int has_deltax_int = has_deltax_root ? 1 : 0;
  mpi->comm.broadcast_n(&has_deltax_int, 1, 0);
  if (has_deltax_int) {
    opt_sDeltaX_left.emplace(
        math::shm::make_shared_from_root_input<ComplexType, 4>(*mpi, DeltaX_left_root));
    opt_sDeltaX_right.emplace(
        math::shm::make_shared_from_root_input<ComplexType, 4>(*mpi, DeltaX_right_root));
  }

  using Array_view_3D_t = nda::array_view<ComplexType, 3>;
  using sArray_3D_t = math::shm::shared_array<Array_view_3D_t>;
  std::optional<sArray_3D_t> opt_sDeltaV_qPQ;
  bool has_dv_root = false;
  if (mpi->comm.root()) {
    has_dv_root = DeltaV_qPQ_root.has_value();
  }
  int has_dv_int = has_dv_root ? 1 : 0;
  mpi->comm.broadcast_n(&has_dv_int, 1, 0);
  if (has_dv_int) {
    opt_sDeltaV_qPQ.emplace(
        math::shm::make_shared_from_root_input<ComplexType, 3>(*mpi, DeltaV_qPQ_root));
  }

  lr_init_timer.stop("LR_INIT");
  app_log(2, "\n  LR_INIT timers");
  app_log(2, "  --------------");
  app_log(2, "    LR Init:                    {0:8.3f} sec  {1:4d} calls",
          lr_init_timer.elapsed("LR_INIT"), lr_init_timer.number_of_calls("LR_INIT"));
  app_log(2, "      - simple_dyson ctor:      {0:8.3f} sec  {1:4d} calls",
          lr_init_timer.elapsed("LR_INIT_DYSON"), lr_init_timer.number_of_calls("LR_INIT_DYSON"));
  app_log(2, "      - read_scf (F + Σ):       {0:8.3f} sec  {1:4d} calls",
          lr_init_timer.elapsed("LR_INIT_READ_SCF"), lr_init_timer.number_of_calls("LR_INIT_READ_SCF"));
  app_log(2, "      - update_G (G unpert.):   {0:8.3f} sec  {1:4d} calls",
          lr_init_timer.elapsed("LR_INIT_UPDATE_G"), lr_init_timer.number_of_calls("LR_INIT_UPDATE_G"));
  app_log(2, "      - load W + eps_inv_head:  {0:8.3f} sec  {1:4d} calls\n",
          lr_init_timer.elapsed("LR_INIT_LOAD_W"), lr_init_timer.number_of_calls("LR_INIT_LOAD_W"));

  utils::memlog("run_lr_calc: before driver.run_lr");

  // Create lr_driver and run unified SCF loop
  lr_driver driver(dyson, q_vec);
  lr_state.kpq_map.emplace(driver.kpq_map());
  auto& thc = eri.corr_eri->get();
  auto* pDeltaSigma = lr_state.sDeltaSigma_tskij ? &(*lr_state.sDeltaSigma_tskij) : nullptr;
  auto* pDeltaX_left = opt_sDeltaX_left ? &(*opt_sDeltaX_left) : nullptr;
  auto* pDeltaX_right = opt_sDeltaX_right ? &(*opt_sDeltaX_right) : nullptr;
  // Bind a view to the DeltaV shared window (lifetime tied to opt_sDeltaV_qPQ).
  std::optional<Array_view_3D_t> dv_view;
  if (opt_sDeltaV_qPQ) dv_view.emplace(opt_sDeltaV_qPQ->local());
  const Array_view_3D_t* pDeltaV_qPQ = dv_view ? &(*dv_view) : nullptr;
  // Pass unperturbed Dm for the DeltaX/DeltaV corrections
  const nda::array<ComplexType, 4>* Dm_ab_ptr = nullptr;
  nda::array<ComplexType, 4> Dm_ab_local;
  if (pDeltaX_left || pDeltaV_qPQ) {
    Dm_ab_local = sDm_skij.local();
    Dm_ab_ptr = &Dm_ab_local;
  }
  auto [niter, Delta_mu] = driver.run_lr(
      lr_state.sDeltaG_tskij.value(), lr_state.sDeltaDm_skij.value(),
      lr_state.sDeltaF_skij.value(), pDeltaSigma,
      sG_tskij, lr_state.sDeltaH0_skij.value(), thc,
      include_hartree, include_exchange, gw_mode,
      static_cast<memory::darray_t<nda::array<ComplexType, 4>, mpi3::communicator>*>(nullptr),
      nullptr,
      max_iter, tol, fix_density, iter_params,
      pDeltaX_left, pDeltaX_right, Dm_ab_ptr, div_corr,
      pDeltaV_qPQ);
  lr_state.Delta_mu.emplace(Delta_mu);
  mpi->comm.barrier();

  utils::memlog("run_lr_calc: after driver.run_lr");

  // Write results
  chkpt::dump_lr(mpi->comm, output + ".mbpt.h5", q_vec,
                 lr_state.sDeltaG_tskij.value(), lr_state.sDeltaDm_skij.value(),
                 lr_state.sDeltaF_skij.value(), pDeltaSigma,
                 Delta_mu, niter,
                 include_hartree, include_exchange, include_gw_sigma);

  utils::memlog("run_lr_calc: exit (before RAII)");
  return std::make_tuple(niter, Delta_mu);
}


// run_lr_calc instantiations (THC only — lr_hf and lr_gw require THC).
// Cholesky LR Dyson (one-shot, no HF/GW) was never used externally,
// so Cholesky ERI combinations are intentionally not instantiated here.
template std::tuple<int, double> run_lr_calc(
    mb_eri_t<thc_reader_t, thc_reader_t, thc_reader_t, thc_reader_t>&,
    ptree const&,
    nda::array<double, 1> const&,
    std::optional<nda::array<ComplexType, 4>> const&,
    bool, bool, lr_gw_update_mode, int, double, bool, const lr_iter_params&,
    std::optional<nda::array<ComplexType, 4>> const&,
    std::optional<nda::array<ComplexType, 4>> const&,
    std::optional<nda::array<ComplexType, 3>> const&);


}
