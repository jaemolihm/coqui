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
#include "methods/SCF/lr_hessian.hpp"
#include "methods/SCF/lr_precompute.hpp"
#include "methods/HF/lr_thc_comm.hpp"
#include "methods/GW/lr_gw.hpp"
#include "methods/GW/g0_div_utils.hpp"
#include "methods/scr_coulomb/lr_rpa_pi.hpp"
#include "methods/scr_coulomb/lr_scr_coulomb_t.hpp"
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
  // Both sources are valid for an augmented basis: "compute" rebuilds H0 from the
  // orbitals, while "checkpoint" reads the stored H0_skij (a genuine computed H0,
  // not the placeholder eigenvalues), so the two agree.
  if (h0_source == "checkpoint") {
    app_log(1, "  H0 source                = checkpoint ({}.mbpt.h5)", chkpt_prefix);
    return simple_dyson(mf_ptr, ft_ptr, chkpt_prefix, mu_tol, mu_update_alg);
  } else {
    app_log(1, "  H0 source                = computed from orbitals");
    return simple_dyson(mf_ptr, ft_ptr, mu_tol, mu_update_alg);
  }
}

/**
 * Write the converged W to <output dir>/thc_screened_interaction.h5.
 *
 * W is held in (t,q,P,Q); the on-disk dataset keeps its (q,t,P,Q) ordering, so
 * the transposed copy is built here and released as soon as it is written.
 */
inline void dump_W_to_h5(MBState& mb_state, const std::string& output) {
  auto W_qtPQ = utils::transpose_axes_01(mb_state.dW_tqPQ.value(), mb_state.mpi->comm);
  auto w_path = (std::filesystem::path(output).parent_path() / "thc_screened_interaction.h5").string();
  if (mb_state.mpi->comm.root()) {
    h5::file file(w_path, 'w');
    h5::group grp(file);
    math::nda::h5_write(grp, "W_qtPQ", W_qtPQ);
  } else {
    h5::group grp;
    math::nda::h5_write(grp, "W_qtPQ", W_qtPQ);
  }
  W_qtPQ.reset();
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
  // Opt-in: additionally write the exchange-only Fock K_skij (F = J + K) to the
  // checkpoint for every iteration F is written. Default (false) leaves output unchanged.
  auto dump_exchange = io::get_value_with_default<bool>(pt,"dump_exchange",false);
  // Opt-in slim checkpoint: skip frequency-dependent Sigma_tskij and G_tskij on
  // non-final SCF iterations (and skip zero Sigma). Default (false) writes the
  // full old dataset layout.
  auto chkpt_slim = io::get_value_with_default<bool>(pt,"chkpt_slim",false);
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
             /*compute_exchange=*/compute_exchange, /*keep_w=*/false, /*chkpt_slim=*/chkpt_slim, /*dump_exchange=*/dump_exchange);

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
               /*compute_exchange=*/compute_exchange, /*keep_w=*/dump_w_to_h5,
               /*chkpt_slim=*/chkpt_slim, /*dump_exchange=*/dump_exchange);

      if (dump_w_to_h5) dump_W_to_h5(mb_state, output);

    } else {

      MBState mb_state(mpi, ft, output);
      auto dump_w_to_h5 = io::get_value_with_default<bool>(pt,"dump_w_to_h5", false);
      scf_loop(mb_state, dyson, eri, ft, mb_solver_t(&hf, &gw, &scr_eri),
               iter_solver.get(), niter, restart, conv_thr, const_mu,
               greens_func_source, greens_func_iteration, 
               /*eval_thermodynamics=*/io::get_value_with_default<bool>(pt, "eval_thermodynamics", false),
               /*compute_exchange=*/compute_exchange, /*keep_w=*/dump_w_to_h5,
               /*chkpt_slim=*/chkpt_slim, /*dump_exchange=*/dump_exchange);

      if (dump_w_to_h5) dump_W_to_h5(mb_state, output);
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
               /*eval_thermodynamics=*/eval_thermodynamics, /*compute_exchange=*/compute_exchange,
               /*keep_w=*/false, /*chkpt_slim=*/chkpt_slim, /*dump_exchange=*/dump_exchange);
    } else {
      solvers::scr_coulomb_t scr_eri(&ft, "rpa", div_treatment);
      scf_loop(mb_state, dyson, eri, ft, mb_solver_t(&hf, &gf2, &scr_eri),
               iter_solver.get(), niter, restart, conv_thr, const_mu,
               greens_func_source, greens_func_iteration,
               /*eval_thermodynamics=*/eval_thermodynamics, /*compute_exchange=*/compute_exchange,
               /*keep_w=*/false, /*chkpt_slim=*/chkpt_slim, /*dump_exchange=*/dump_exchange);
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
                niter, restart, conv_thr, /*compute_exchange=*/compute_exchange, h0_source);

    // Always persist the qp params + div_treatment so a later LR-qpGW run can
    // statify ΔΣ with exactly the continuation this qpGW run used.
    chkpt::dump_qp_params(mpi->comm, output + ".mbpt.h5",
                          off_diag_mode, eta, ac_alg, Nfit, div_treatment);

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
  // Opt-in: additionally write the exchange-only Fock K_skij (F = J + K) to the
  // checkpoint for every iteration F is written. Default (false) leaves output unchanged.
  auto dump_exchange = io::get_value_with_default<bool>(pt,"dump_exchange",false);
  // Opt-in slim checkpoint: skip frequency-dependent Sigma_tskij and G_tskij on
  // non-final SCF iterations (and skip zero Sigma). Default (false) writes the
  // full old dataset layout.
  auto chkpt_slim = io::get_value_with_default<bool>(pt,"chkpt_slim",false);
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
             /*compute_exchange=*/compute_exchange, /*keep_w=*/false, /*chkpt_slim=*/chkpt_slim, /*dump_exchange=*/dump_exchange);

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
 * Read the div_treatment the scGW run used (stashed in the checkpoint scf
 * group by scr_coulomb_t::dump_eps_inv_head); checkpoints that predate this
 * dataset must be regenerated. Root reads, then broadcasts via a fixed-size
 * buffer. LR/GW evaluators must reuse this value so divergence heads stay
 * consistent with how W and Sigma were computed.
 */
template<typename mpi_context_t>
std::string read_div_treatment(mpi_context_t& mpi,
                               const std::string& input_file,
                               const std::string& input_grp)
{
  char div_buf[64] = {0};
  if (mpi.comm.root()) {
    h5::file file(input_file, 'r');
    auto root_grp = h5::group(file);
    auto scf_grp = root_grp.open_group(input_grp);
    utils::check(scf_grp.has_dataset("div_treatment"),
                 "read_div_treatment: '{}/div_treatment' missing in {}. "
                 "This checkpoint predates div_treatment dump; please regenerate "
                 "the scGW checkpoint with the current code.",
                 input_grp, input_file);
    std::string div_str;
    h5::h5_read(scf_grp, "div_treatment", div_str);
    utils::check(div_str.size() < sizeof(div_buf),
                 "read_div_treatment: div_treatment string too long: {}", div_str);
    std::copy(div_str.begin(), div_str.end(), div_buf);
  }
  mpi.comm.broadcast_n(div_buf, sizeof(div_buf), 0);
  return std::string(div_buf);
}

/**
 * Read the hf_div_treatment the ground-state run used (stashed on the checkpoint
 * scf group by chkpt::dump_hf_div_treatment), so LR applies the exchange
 * Madelung term exactly when the unperturbed Fock did.
 *
 * Unlike read_div_treatment this does NOT error when the dataset is absent:
 * checkpoints predating the stash fall back to "gygi", which is both hf_t's
 * default and what LR applied unconditionally before, so they keep their
 * previous behaviour instead of needing to be regenerated.
 */
template<typename mpi_context_t>
std::string read_hf_div_treatment(mpi_context_t& mpi,
                                  const std::string& input_file,
                                  const std::string& input_grp)
{
  char div_buf[64] = {0};
  int found = 0;
  if (mpi.comm.root()) {
    h5::file file(input_file, 'r');
    auto root_grp = h5::group(file);
    if (root_grp.has_subgroup(input_grp)) {
      auto scf_grp = root_grp.open_group(input_grp);
      if (scf_grp.has_dataset("hf_div_treatment")) {
        std::string div_str;
        h5::h5_read(scf_grp, "hf_div_treatment", div_str);
        utils::check(div_str.size() < sizeof(div_buf),
                     "read_hf_div_treatment: hf_div_treatment string too long: {}", div_str);
        std::copy(div_str.begin(), div_str.end(), div_buf);
        found = 1;
      }
    }
  }
  mpi.comm.broadcast_n(&found, 1, 0);
  if (!found) {
    app_log(2, "  '{}/hf_div_treatment' not found in {}; assuming \"gygi\" for the LR "
               "exchange divergence. Regenerate the checkpoint if the ground state "
               "used hf_div_treatment = \"ignore_g0\".", input_grp, input_file);
    return "gygi";
  }
  mpi.comm.broadcast_n(div_buf, sizeof(div_buf), 0);
  return std::string(div_buf);
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
 * @param screened_interaction_file - Explicit path to the W_qtPQ HDF5 file.
 *                      If empty, falls back to thc_screened_interaction.h5 in
 *                      the input checkpoint's directory.
 * @return (dW_tqPQ, eps_inv_head, div_treatment) tuple. W is returned as
 *         (t,q,P,Q): the dataset on disk is (q,t), but eps_inv_head_t needs (t,q)
 *         so that copy is made here regardless — handing it back costs nothing,
 *         while returning (q,t) would make every LR caller transpose it again.
 */
template<typename mpi_context_t, typename THC_t>
auto load_W_and_eps_inv_head(
    mpi_context_t& mpi,
    THC_t& thc,
    const std::string& input_file,
    const std::string& input_grp,
    long input_iter,
    long nt,
    const std::string& screened_interaction_file = "")
{
  auto mf = thc.MF();
  long nkpts = mf->nkpts();
  long NP = thc.Np();
  long nt_half = (nt % 2 == 0) ? nt / 2 : nt / 2 + 1;

  // Read W from the explicit screened_interaction_file if given; otherwise fall
  // back to thc_screened_interaction.h5 in the input checkpoint's directory.
  auto w_file = screened_interaction_file.empty()
      ? (std::filesystem::path(input_file).parent_path() / "thc_screened_interaction.h5").string()
      : screened_interaction_file;
  app_log(2, "Reading W from {}...", w_file);

  // The dataset on disk is (q,t,P,Q), so the distributed read must target that
  // axis order: h5_read maps the array's index space onto
  // the dataset's and cannot permute axes.
  // τ-dist for (q,t,P,Q): swap axes 0,1 of the (t,q) τ-dist grid
  auto [tq_pgrid, tq_bsize] = utils::lr_W_q_local_dist(mpi.comm.size(), nt_half, NP);
  std::array<long,4> qt_pgrid = {tq_pgrid[1], tq_pgrid[0], tq_pgrid[2], tq_pgrid[3]};
  std::array<long,4> qt_bsize = {tq_bsize[1], tq_bsize[0], tq_bsize[2], tq_bsize[3]};

  using local_Array_4D_t = nda::array<ComplexType, 4>;
  auto dW_qtPQ = math::nda::make_distributed_array<local_Array_4D_t>(
      mpi.comm, qt_pgrid, {nkpts, nt_half, NP, NP}, qt_bsize);

  // Each rank reads only its local (q,t,P,Q) slab via the distributed h5_read.
  // A "root reads full + broadcast" pattern OOMs at scale: full W for nkpts=512,
  // NP=204 is ~30 GB per rank (×2 with broadcast staging), well over per-node RAM
  // when many ranks share a node.
  {
    h5::file file(w_file, 'r');
    h5::group grp(file);
    // Verify the W-on-disk matches the THC currently loaded. Silent shape
    // mismatch between stale W (from scGW build-time) and current THC (from
    // current ISDF) produces a garbage W that explodes downstream by ~|W|²
    // amplification in the ΔW = W·ΔΠ·W Dyson step.
    auto ds_info = h5::array_interface::get_dataset_info(grp, "W_qtPQ");
    long ds_np_a = ds_info.lengths[2];
    long ds_np_b = ds_info.lengths[3];
    utils::check(ds_np_a == NP && ds_np_b == NP,
                 "load_W_and_eps_inv_head: W_qtPQ in {} has shape (...,{},{}) "
                 "but current THC has Np={}. The W file was generated with a "
                 "different THC decomposition; regenerate it (or re-run GW) "
                 "with the current THC before calling run_lr.",
                 w_file, ds_np_a, ds_np_b, NP);
    math::nda::h5_read(grp, "W_qtPQ", dW_qtPQ);
  }
  mpi.comm.barrier();

  auto div_treatment = read_div_treatment(mpi, input_file, input_grp);

  // Recompute eps_inv_head from the loaded W_c
  // We don't read eps_inv_head from input_file to make sure it is consistent with W_c
  imag_axes_ft::IAFT ft(imag_axes_ft::read_iaft(input_file, false));
  auto dW_tqPQ = utils::transpose_axes_01(dW_qtPQ, mpi.comm);
  dW_qtPQ.reset();
  auto [eps_inv_head_q, eps_inv_head] =
      solvers::div_utils::eps_inv_head_t(dW_tqPQ, thc, *mf, &ft, div_treatment);
  mpi.comm.barrier();

  return std::make_tuple(std::move(dW_tqPQ), std::move(eps_inv_head),
                         std::move(div_treatment));
}

/**
 * Helper: read W from the screened-interaction file and hand it to the LR path
 * on the Matsubara axis, so lr_driver has exactly one W layout to support.
 *
 * The τ→ω transform lives here rather than in load_W_and_eps_inv_head because
 * that function's other callers (the standalone scGW Σ paths) want W(τ). Only
 * this from-file path pays the transform; recompute_W_and_eps_inv_head, which
 * solves the W Dyson in ω anyway, hands ω back for free.
 *
 * @param mpi         - MPI context
 * @param thc         - THC ERI
 * @param ft          - IAFT (from checkpoint)
 * @param input_file  - Path to checkpoint HDF5 file
 * @param input_grp   - SCF group name in checkpoint
 * @param input_iter  - Iteration to read (-1 = final_iter)
 * @param screened_interaction_file - Explicit path to W_qtPQ, "" = auto-derive
 * @return (dW_wqPQ, eps_inv_head, div_treatment) — W on the Matsubara axis on
 *         the q-dist distribution (utils::lr_W_tau_local_dist),
 *         head on the τ axis
 */
template<typename mpi_context_t, typename THC_t>
auto lr_load_W_omega(
    mpi_context_t& mpi,
    THC_t& thc,
    imag_axes_ft::IAFT const& ft,
    const std::string& input_file,
    const std::string& input_grp,
    long input_iter,
    const std::string& screened_interaction_file = "")
{
  auto [dW_tqPQ, eps_inv_head, div_treatment] = load_W_and_eps_inv_head(
      mpi, thc, input_file, input_grp, input_iter, ft.nt_f(),
      screened_interaction_file);

  long nw_half = (ft.nw_b() % 2 == 0) ? ft.nw_b() / 2 : ft.nw_b() / 2 + 1;
  auto [w_pgrid, w_bsize] = utils::lr_W_tau_local_dist(
      mpi.comm.size(), nw_half, thc.MF()->nkpts(), thc.Np());

  solvers::scr_coulomb_fourier_t setup_ft(&ft);
  auto dW_wqPQ = setup_ft.tau_to_w(dW_tqPQ, w_pgrid, w_bsize, /*reset_input=*/true,
                                   __app_verbosity__ >= 3);
  mpi.comm.barrier();

  // The same q→0 head on the ω axis. A consumer wanting one Matsubara component
  // (HSEX reads ν = 0) must not recover it by transforming the τ array back:
  // for div_treatment "*_metal", extrapolate_eps_inv_q0 hand-sets
  // eps_inv_q0_w(0) = -1, and a hand-poked point is not generally
  // IR-representable, so the round trip smears exactly that value.
  auto MF = thc.MF();
  auto [eps_inv_head_wq, eps_inv_head_w] =
      solvers::div_utils::eps_inv_head_w(dW_wqPQ, thc, *MF, div_treatment);

  return std::make_tuple(std::move(dW_wqPQ), std::move(eps_inv_head),
                         std::move(eps_inv_head_w), std::move(div_treatment));
}

/**
 * Helper: recompute W and eps_inv_head from the (already-built) unperturbed
 * Green's function instead of reading W from a screened-interaction file.
 *
 * This reuses the exact scGW pipeline (scr_coulomb_t::eval_Pi_qdep +
 * dyson_W_from_Pi_tau + div_utils::eps_inv_head_t), so the recomputed W_c is,
 * by construction, consistent with the currently-loaded THC. RPA screening
 * only (standard GW); cRPA / gw_edmft build W differently and are rejected via
 * the no-symmetry requirement below + the RPA screen_type.
 *
 * W comes back on the **frequency** axis as (w,q,P,Q), on the grid the LR path
 * wants (utils::lr_W_tau_local_dist) — the one layout lr_driver accepts.
 * The W Dyson is solved in ω anyway, so returning ω saves the ω→τ transform
 * back onto Π's τ grid *and* the redistribute onto the LR tiling that would
 * follow it — the LR driver does one ω→τ straight onto its own grid instead.
 *
 * eps_inv_head is built with eps_inv_head_w rather than eps_inv_head_t and then
 * transformed to τ. eval_eps_inv_q is linear and pointwise in the imaginary-axis
 * index, and eps_inv_head_t is exactly "eval → tau_to_w_PHsym → extrapolate →
 * w_to_tau_PHsym", so this is the same quantity with two fewer transforms.
 *
 * @param mpi           - MPI context
 * @param thc           - THC ERI
 * @param ft            - IAFT (from checkpoint)
 * @param sG_tskij      - Unperturbed G(τ) shared array (nt, ns, nk, nb, nb)
 * @param div_treatment - Divergence treatment scGW used (from checkpoint)
 * @return (dW_wqPQ, eps_inv_head) — W on the Matsubara axis, head on the τ axis
 */
template<typename mpi_context_t, typename THC_t, typename sArray_5D_t>
auto recompute_W_and_eps_inv_head(
    mpi_context_t& mpi,
    THC_t& thc,
    imag_axes_ft::IAFT const& ft,
    sArray_5D_t& sG_tskij,
    const std::string& div_treatment)
{
  auto mf = thc.MF();
  long nkpts = mf->nkpts();

  // RPA polarization Π and W Dyson require the full q-grid (no symmetry).
  utils::check(mf->nqpts() == mf->nqpts_ibz() and mf->nqpts() == nkpts,
               "recompute_W_and_eps_inv_head: No symmetry required. "
               "nqpts={}, nqpts_ibz={}, nkpts={}",
               mf->nqpts(), mf->nqpts_ibz(), nkpts);

  // Standard GW: W is RPA-screened. cRPA / edmft are not supported here.
  solvers::scr_coulomb_t scr(&ft, "rpa", div_treatment);

  long nw_half = (ft.nw_b() % 2 == 0) ? ft.nw_b() / 2 : ft.nw_b() / 2 + 1;
  auto [w_pgrid, w_bsize] = utils::lr_W_tau_local_dist(
      mpi.comm.size(), nw_half, nkpts, thc.Np());

  auto dPi_tqPQ = scr.eval_Pi_qdep(sG_tskij.local(), thc);
  mpi.comm.barrier();
  auto dW_wqPQ = scr.dyson_W_from_Pi_tau<true>(dPi_tqPQ, thc, true, w_pgrid, w_bsize);
  mpi.comm.barrier();

  // eps_inv_head_w returns the head at every (iω, q) plus the q→0 extrapolated
  // head at every iω; only the latter is wanted here.
  auto [eps_inv_head_wq, eps_inv_head_q0_w] =
      solvers::div_utils::eps_inv_head_w(dW_wqPQ, thc, *mf, div_treatment);

  // The head is consumed on the τ axis (lr_gw::apply_div_correction_*), so make
  // the same final ω→τ step eps_inv_head_t ends with.
  long nt_half = (ft.nt_b() % 2 == 0) ? ft.nt_b() / 2 : ft.nt_b() / 2 + 1;
  nda::array<ComplexType, 1> eps_inv_head(nt_half);
  auto head_w_2D = nda::reshape(eps_inv_head_q0_w, std::array<long, 2>{nw_half, 1});
  auto head_t_2D = nda::reshape(eps_inv_head, std::array<long, 2>{nt_half, 1});
  ft.w_to_tau_PHsym(head_w_2D, head_t_2D);

  return std::make_tuple(std::move(dW_wqPQ), std::move(eps_inv_head),
                         std::move(eps_inv_head_q0_w));
}

/**
 * Unified linear response calculation.
 *
 * Runs the LR SCF loop with configurable Hartree, Exchange, and GW self-energy components:
 *   ΔH0 → ΔG → ΔDm → [ΔF] → [ΔΣ] → ΔG → ... (iterate until convergence)
 *
 * Several perturbations at the same q are solved in one call: the unperturbed
 * state, the screened interaction, the G^R/G(iω) caches and the solvers are set
 * up once and reused, and each perturbation is written to its own
 * "linear_response/mode{m}" group. This is the whole reason the ΔH0 argument
 * carries a leading mode axis. q_vec is fixed for the call — the driver and all
 * its solvers latch it at construction.
 *
 * @param eri              - [INPUT] ERI handler (must be THC)
 * @param pt               - [INPUT] Parameters as property tree
 * @param q_vec            - [INPUT] Perturbation wavevector in crystal coords (3,)
 * @param DeltaH0_mskij    - [INPUT] Perturbations (nmodes, ns, nk, nb, nb), root only
 * @param include_hartree  - [INPUT] Include ΔJ in SCF loop; overridden by the
 *                             "method" key when that is given
 * @param include_exchange - [INPUT] Include ΔK in SCF loop; likewise
 * @param gw_mode          - [INPUT] GW self-energy mode (none/fixed_W/full); likewise
 * @param max_iter         - [INPUT] Maximum SCF iterations (1 = one-shot)
 * @param tol              - [INPUT] Convergence tolerance for ||ΔDm_new - ΔDm_old||
 * @param fix_density      - [INPUT] If true, compute Δμ to enforce ΔN=0
 * @param iter_params      - [INPUT] Iteration algorithm parameters (damping/DIIS)
 *
 * Kernel-selection keys read from `pt`:
 *   method                 method-ladder alias ("none"/"Hartree"/"HF"/"GW0"/"GW")
 *                          that expands to include_hartree/include_exchange/gw_mode.
 *                          Names the TOTAL kernel, for two-step runs as well.
 *   lr_two_step            enable the split-kernel schedule (default false)
 *   two_step_inner_method  method whose kernel is resummed self-consistently; the
 *                          perturbative kernel is the total minus this one
 *   two_step_order         truncation order n of the K_pert expansion (default 1)
 *
 * @return Per-perturbation (number of iterations, final Δμ), each of length nmodes
 */
template<typename eri_t>
std::tuple<nda::array<long, 1>, nda::array<double, 1>>
                        run_lr_calc(eri_t &eri, ptree const& pt,
                                     nda::array<double, 1> const& q_vec,
                                     std::optional<nda::array<ComplexType, 5>> const& DeltaH0_mskij_root,
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
  // NOTE: output must be an existing .mbpt.h5 checkpoint (dump_lr appends)
  auto output = io::get_value_with_default<std::string>(pt, "output", prefix);
  auto input_grp = io::get_value_with_default<std::string>(pt, "input_type", "scf");
  auto input_iter = io::get_value_with_default<long>(pt, "input_iter", -1);
  auto h0_source = io::get_value_with_default<std::string>(pt, "h0_source", "compute");
  // Explicit screened-interaction (W) file. Empty => auto-derive from the input
  // checkpoint's directory (thc_screened_interaction.h5).
  auto screened_interaction_file =
      io::get_value_with_default<std::string>(pt, "screened_interaction_file", "");
  // When true, recompute W from the checkpoint Green's function (RPA) instead
  // of reading it from disk.
  auto recompute_W = io::get_value_with_default<bool>(pt, "recompute_W", false);
  // Unperturbed reference: "checkpoint" (interacting G rebuilt from the
  // checkpoint F+Σ, default) or "mf_dft" (DFT/KS mean-field G0 built directly
  // from the mean-field eigenvalues/orbitals, for one-shot G0W0@DFT LR). In the
  // "mf_dft" case W0 is the RPA screened interaction from G0 (recompute_W is
  // forced on) and the checkpoint is used only for the IAFT grid + metadata.
  auto unperturbed = io::get_value_with_default<std::string>(pt, "unperturbed", "checkpoint");
  utils::check(unperturbed == "checkpoint" || unperturbed == "mf_dft",
               "run_lr_calc: unperturbed must be 'checkpoint' or 'mf_dft', got '{}'", unperturbed);
  // One-shot G0W0: store the two ΔΣ terms (dG0·W0, G0·dW0) separately.
  auto split_sigma_terms = io::get_value_with_default<bool>(pt, "split_sigma_terms", false);

  // Kernel-component ladder: none ⊂ Hartree ⊂ HF ⊂ GW0 ⊂ GW.
  //
  // "method" is an alias that expands to the include_hartree / include_exchange
  // / gw_mode triple; when given it defines them, so a caller never has to keep
  // the triple and the method name in sync.
  //
  // "lr_two_step" splits the kernel into a part resummed by the SCF loop
  // (two_step_inner_method) and a remainder applied `two_step_order` times
  //   K_pert = kernel(method) \ kernel(two_step_inner_method).
  // "method" is the TOTAL kernel here too; the remainder ({Σ2} for
  // GW-on-top-of-GW0, say) has no name on the ladder and can only be expressed
  // as a difference.
  auto method = io::get_value_with_default<std::string>(pt, "method", "");
  auto lr_two_step = io::get_value_with_default<bool>(pt, "lr_two_step", false);
  auto two_step_inner_method =
      io::get_value_with_default<std::string>(pt, "two_step_inner_method", "");
  auto two_step_order = io::get_value_with_default<long>(pt, "two_step_order", 1);

  // Acceleration of the OUTER (perturbative-source) iteration. The outer
  // sequence is a Picard iteration on an affine map, so the same accelerator
  // the inner SCF uses applies to it — as a second, fully independent instance.
  // Defaults reproduce the plain Neumann series exactly.
  auto two_step_outer_alg =
      io::get_value_with_default<std::string>(pt, "two_step_outer_alg", "damping");
  auto two_step_outer_mixing =
      io::get_value_with_default<double>(pt, "two_step_outer_mixing", 1.0);
  auto two_step_outer_subsp =
      io::get_value_with_default<long>(pt, "two_step_outer_subsp", 3);
  auto two_step_outer_warmup =
      io::get_value_with_default<long>(pt, "two_step_outer_warmup", 0);
  auto two_step_outer_tol =
      io::get_value_with_default<double>(pt, "two_step_outer_tol", 0.0);
  // History vectors required before the outer DIIS extrapolates. 2 puts the
  // first extrapolation at the second outer step, which on a short sequence
  // saves a whole expensive kernel evaluation.
  auto two_step_outer_min_subsp =
      io::get_value_with_default<long>(pt, "two_step_outer_min_subsp", 2);

  auto gw_mode_of = [](lr_kernel_spec const& s) {
    return s.sigma_G_dW ? lr_gw_update_mode::full
         : s.sigma_dG_W ? lr_gw_update_mode::fixed_W
                        : lr_gw_update_mode::none;
  };
  auto spec_of_flags = [](bool h, bool x, lr_gw_update_mode m) {
    return lr_kernel_spec{h, x, m != lr_gw_update_mode::none,
                          m == lr_gw_update_mode::full};
  };
  // The spelling the Python layer accepts, so a checkpoint field can be fed
  // straight back into a run.
  auto gw_mode_str = [](lr_gw_update_mode m) -> std::string {
    return m == lr_gw_update_mode::full    ? "full"
         : m == lr_gw_update_mode::fixed_W ? "fixed_W"
                                           : "none";
  };

  // "HSEX" expands to the HF mask plus lr_params::exchange_static_W rather than
  // to a kernel_spec_from_method name; see that flag for why it cannot be a
  // mask component. It may name the TOTAL kernel (a standalone run) or the
  // self-consistent half of a split one, but not both — the counter-term that
  // makes a split sum back to bare exchange is defined against a
  // self-consistent screened exchange.
  bool exchange_static_W = (method == "HSEX") || (two_step_inner_method == "HSEX");
  utils::check(!lr_two_step || method != "HSEX",
               "run_lr_calc: with lr_two_step, HSEX must be the "
               "two_step_inner_method (the kernel that is resummed), not the "
               "total 'method'.");
  // The spelling is kept for the checkpoint, since the mask cannot carry it.
  const std::string two_step_inner_method_in = two_step_inner_method;
  if (method == "HSEX") method = "HF";
  if (two_step_inner_method == "HSEX") two_step_inner_method = "HF";
  if (!method.empty()) {
    auto s = kernel_spec_from_method(method);
    include_hartree = s.hartree;
    include_exchange = s.exchange;
    gw_mode = gw_mode_of(s);
  }
  const lr_kernel_spec total_kernel =
      spec_of_flags(include_hartree, include_exchange, gw_mode);

  // LR-DFT: add the semilocal xc kernel to the direct channel, i.e. use
  // (V + Vxc)(q) in ΔJ. Deliberately an explicit opt-in rather than keyed off
  // the presence of a Vxc dataset in the THC: otherwise an --mbpt Hartree run
  // against a Vxc-carrying THC file would silently become LR-DFT.
  auto include_xc = io::get_value_with_default<bool>(pt, "include_xc", false);

  // Aux-basis (THC) Fock matrices F_PQ / ΔF_PQ, consumed only by the DeltaX/IBC
  // ΔΔF curvature post-processors in Python. Opt-in, because they are
  // (nspin, nkpts_ibz, Np, Np): at production Np they dwarf every other LR
  // output (~13 GB per array at Np = 3.7k) in the checkpoint, and capturing
  // ΔF_PQ costs one extra lr_hf::evaluate on the converged ΔDm. Runs that do
  // not do IBC should never pay for them.
  auto output_aux_fock = io::get_value_with_default<bool>(pt, "output_aux_fock", false);
  // Variationally-stationary (quadratic-error) free-energy hessian, on top of the
  // plain contraction. Opt-in: one extra Dyson solve per perturbation plus two
  // striped ω stores, and it adds the top-level linear_response/hessian* datasets
  // to the checkpoint. Off by default, so the default output is unchanged.
  auto hessian = io::get_value_with_default<bool>(pt, "lr_hessian", false);
  // Checked before the include_xc validation below, so that a split-kernel run
  // is rejected for being a split-kernel run rather than for whichever
  // include_xc precondition the bed happens to violate first.
  utils::check(!(lr_two_step && include_xc),
               "run_lr_calc: lr_two_step is incompatible with include_xc.");

  // LR output volume. Defaults reproduce the previous checkpoint byte for byte.
  //   save_DeltaG — write DeltaG_tskij at all. It is the biggest LR dataset and
  //     nothing reads it back, so a phonon sweep can drop it.
  //   nbnd_save   — keep only the leading nbnd_save x nbnd_save band block of the
  //     imaginary-time arrays. The result is a protected-band block, not a
  //     full-basis object; each trimmed dataset says so via an "nbnd_save"
  //     attribute. Absent = no trim.
  auto save_DeltaG = io::get_value_with_default<bool>(pt, "save_DeltaG", true);
  std::optional<long> nbnd_save;
  if (io::check_exists<long>(pt, "nbnd_save")) {
    nbnd_save = io::get_value<long>(pt, "nbnd_save");
  }
  if (include_xc) {
    utils::check(include_hartree,
                 "run_lr_calc: include_xc = true requires include_hartree = true.");
    utils::check(!include_exchange,
                 "run_lr_calc: include_xc = true is incompatible with "
                 "include_exchange = true. The semilocal xc kernel contracts with "
                 "the diagonal density response only; LR-DFT is include_hartree = "
                 "true, include_exchange = false.");
    utils::check(gw_mode == lr_gw_update_mode::none,
                 "run_lr_calc: include_xc = true is incompatible with a GW self-energy "
                 "(gw_mode != none): f_xc and ΔΣ_GW both carry the correlation "
                 "response, so the two together double-count it. include_xc works only "
                 "in the Hartree mode.");
    utils::check(eri.corr_eri->get().has_Vxc(),
                 "run_lr_calc: include_xc = true but the THC integrals carry no "
                 "xc-kernel matrix. Rebuild the THC with 'Vxc_file' set (deleting "
                 "any stale THC checkpoint first), or set include_xc = false.");
  }

  // LR-qpGW: statify the dynamic ΔΣ(iω) into a static ΔV_QPGW each iteration
  // (frozen QP orbitals from the qpGW checkpoint). Uses W0=RPA[G_QP] (recompute_W
  // forced on).
  auto qp_static_sigma = io::get_value_with_default<bool>(pt, "qp_static_sigma", false);
  if (qp_static_sigma) {
    utils::check(unperturbed == "checkpoint",
                 "run_lr_calc: qp_static_sigma requires unperturbed='checkpoint' (a qpGW checkpoint).");
    utils::check(gw_mode != lr_gw_update_mode::none,
                 "run_lr_calc: qp_static_sigma requires a GW self-energy (gw_mode != none).");
    utils::check(!split_sigma_terms,
                 "run_lr_calc: qp_static_sigma is incompatible with split_sigma_terms.");
    recompute_W = true;  // W0 = RPA[G_QP]
  }

  // Scope of the stationary hessian estimator: the plain single- or split-kernel
  // bare-vertex path. Both exclusions are places where its algebra stops holding
  // rather than places it merely has not been tested.
  if (hessian) {
    // IBC / δX / δV: the left vertex handed to the hessian contraction is then
    // not the ΔH0 that drives the equation, so the functional is no longer the
    // second derivative of one functional and would need the adjoint solve.
    int has_pert_arrays = 0;
    if (mpi->comm.root())
      has_pert_arrays = (DeltaX_left_root.has_value() || DeltaX_right_root.has_value() ||
                         DeltaV_qPQ_root.has_value()) ? 1 : 0;
    mpi->comm.broadcast_n(&has_pert_arrays, 1, 0);
    utils::check(!has_pert_arrays,
                 "run_lr_calc: lr_hessian is incompatible with the DeltaX "
                 "(IBC) and DeltaV_qPQ perturbations. With g_left != ΔH0 the "
                 "stationary functional is not a second derivative of a single "
                 "functional and the correction formula does not apply.");
    // qpGW static map: the mixed/tracked quantity is the static ΔV_QPGW rather
    // than ΔΣ(iω), and the effective kernel carries the lr_qp_approx
    // statification, so the raw-ΔΣ algebra does not carry over.
    utils::check(!qp_static_sigma,
                 "run_lr_calc: lr_hessian is incompatible with "
                 "qp_static_sigma. In qp mode the accelerator tracks the static "
                 "ΔV_QPGW instead of ΔΣ(iω) and the kernel includes the qp "
                 "statification, so the functional's raw-ΔΣ algebra does not hold.");
  }

  // Split-kernel schedule: build and validate the two component masks.
  lr_kernel_spec sc_kernel = total_kernel;
  lr_kernel_spec pert_kernel;
  int pert_order = 0;
  if (lr_two_step) {
    utils::check(!two_step_inner_method.empty(),
                 "run_lr_calc: lr_two_step = true requires 'two_step_inner_method'.");
    sc_kernel = kernel_spec_from_method(two_step_inner_method);
    utils::check(!(sc_kernel == total_kernel),
                 "run_lr_calc: two_step_inner_method = '{}' names the whole "
                 "active kernel ({}), so lr_two_step is a no-op. Drop "
                 "lr_two_step, or name a smaller self-consistent kernel.",
                 two_step_inner_method, total_kernel.to_string());
    pert_kernel = kernel_diff(total_kernel, sc_kernel);
    utils::check(two_step_order >= 0,
                 "run_lr_calc: two_step_order must be >= 0, got {}.", two_step_order);
    utils::check(!split_sigma_terms,
                 "run_lr_calc: lr_two_step is incompatible with split_sigma_terms.");
    utils::check(!qp_static_sigma,
                 "run_lr_calc: lr_two_step is incompatible with qp_static_sigma.");
    // The IBC / δV perturbations enter every kernel evaluator separately, so a
    // two-pass schedule would double-count them; reject rather than half-apply.
    int has_pert_arrays = 0;
    if (mpi->comm.root())
      has_pert_arrays = (DeltaX_left_root.has_value() || DeltaX_right_root.has_value() ||
                         DeltaV_qPQ_root.has_value()) ? 1 : 0;
    mpi->comm.broadcast_n(&has_pert_arrays, 1, 0);
    utils::check(!has_pert_arrays,
                 "run_lr_calc: lr_two_step is incompatible with the DeltaX (IBC) "
                 "and DeltaV_qPQ perturbations.");
    pert_order = static_cast<int>(two_step_order);
    app_log(2, "LR split kernel: K_sc = {} ('{}'), K_pert = {} (order {})",
            sc_kernel.to_string(), two_step_inner_method_in,
            pert_kernel.to_string(), pert_order);
  }

  // Outer-loop acceleration: validate, and reject a non-default outer key on a
  // run that has no outer loop rather than silently ignoring it.
  const bool outer_is_default =
      two_step_outer_alg == "damping" && two_step_outer_mixing >= 1.0 &&
      two_step_outer_subsp == 3 && two_step_outer_warmup == 0 &&
      two_step_outer_tol == 0.0 && two_step_outer_min_subsp == 2;
  utils::check(lr_two_step || outer_is_default,
               "run_lr_calc: the two_step_outer_* keys require lr_two_step = true; "
               "there is no outer (perturbative-source) loop to accelerate otherwise.");
  // Rejected here rather than in lr_setup so the caller is told before the basis
  // and the ERI are opened. Must stay in step with
  // lr_outer_accel_params::active() (lr_driver.hpp), which is not built yet.
  const bool outer_requested =
      (two_step_outer_alg == "DIIS" || two_step_outer_tol > 0.0);
  utils::check(!outer_requested || two_step_order > 0,
               "run_lr_calc: the two_step_outer_* keys require two_step_order >= 1; "
               "there is no outer sequence at order 0.");
  utils::check(two_step_outer_subsp >= 2,
               "run_lr_calc: two_step_outer_subsp must be >= 2, got {}.",
               two_step_outer_subsp);
  utils::check(two_step_outer_min_subsp >= 2 &&
               two_step_outer_min_subsp <= two_step_outer_subsp,
               "run_lr_calc: two_step_outer_min_subsp must be in [2, "
               "two_step_outer_subsp = {}], got {}.",
               two_step_outer_subsp, two_step_outer_min_subsp);
  utils::check(two_step_outer_warmup >= 0,
               "run_lr_calc: two_step_outer_warmup must be >= 0, got {}.",
               two_step_outer_warmup);
  utils::check(two_step_outer_tol >= 0.0,
               "run_lr_calc: two_step_outer_tol must be >= 0, got {}.",
               two_step_outer_tol);
  // The outer accelerator does not damp: `mixing` only affects its warmup
  // steps, so a damping-only request would silently do nothing.
  utils::check(two_step_outer_alg == "DIIS" || two_step_outer_mixing >= 1.0,
               "run_lr_calc: two_step_outer_mixing = {} < 1 requires "
               "two_step_outer_alg = 'DIIS'; the outer loop has no damping-only "
               "mode.", two_step_outer_mixing);

  lr_outer_accel_params outer_params;
  outer_params.iter.alg = two_step_outer_alg;
  outer_params.iter.mixing = two_step_outer_mixing;
  outer_params.iter.max_subsp_size = static_cast<size_t>(two_step_outer_subsp);
  outer_params.iter.diis_warmup = static_cast<size_t>(two_step_outer_warmup);
  outer_params.tol = two_step_outer_tol;
  outer_params.min_subsp = static_cast<size_t>(two_step_outer_min_subsp);
  const bool outer_active = lr_two_step && outer_params.active();

  std::string input_file = prefix + ".mbpt.h5";
  utils::check(std::filesystem::exists(input_file),
               "run_lr_calc: Input checkpoint {} does not exist!", input_file);

  utils::memlog("run_lr_calc: entry");

  // Validate checkpoint contains required data. The "mf_dft" reference rebuilds
  // G0 from the mean field and reads only IAFT + metadata from the checkpoint,
  // so it does not require the SCF F/Σ groups.
  if (unperturbed == "checkpoint") {
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
                  "LR_INIT_UPDATE_G", "LR_INIT_LOAD_W", "LR_DUMP"}) {
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

  double mu = 0.0;
  if (unperturbed == "mf_dft") {
    // DFT/KS mean-field reference G0 for one-shot G0W0@DFT LR. Build the MO
    // coefficients / energies from the mean field, set μ to conserve N, and
    // build G0(τ) directly (Σ=0). sF_skij and sSigma_tskij are left unused in
    // this branch. See qp_scf_common::write_mf_data for the same recipe.
    app_log(2, "Building DFT/KS mean-field reference G0 (G0W0@DFT)...");
    lr_init_timer.start("LR_INIT_UPDATE_G");
    auto [sMO_skia, sE_ska] = get_mf_MOs(*mpi, *mf, *dyson.PSP());
    mu = update_mu(0.0, *mf, sE_ska, ft.beta());
    app_log(2, "  DFT reference mu = {:.8f}", mu);
    update_G(sG_tskij, sMO_skia, sE_ska, mu, ft);
    update_Dm(sDm_skij, sMO_skia, sE_ska, mu, ft.beta());
    mpi->comm.barrier();
    lr_init_timer.stop("LR_INIT_UPDATE_G");
  } else {
    // Read F and Sigma from checkpoint
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
  }

  // LR-qpGW: read the frozen QP eigenbasis (C, ε) directly from the qpGW
  // checkpoint. The qpGW run stores its eigenbasis (MO_skia/E_ska), so we read
  // it rather than re-diagonalizing H0+F — re-diagonalization can introduce an
  // arbitrary gauge in the degenerate/near-degenerate subspaces.
  std::optional<math::shm::shared_array<Array_view_4D_t>> opt_sMO_qp;
  std::optional<math::shm::shared_array<nda::array_view<ComplexType, 3>>> opt_sE_qp;
  qp_params_t qp_params;
  if (qp_static_sigma) {
    opt_sMO_qp.emplace(math::shm::make_shared_array<Array_view_4D_t>(
        *mpi, {mf->nspin(), mf->nkpts_ibz(), mf->nbnd(), mf->nbnd()}));
    opt_sE_qp.emplace(math::shm::make_shared_array<nda::array_view<ComplexType, 3>>(
        *mpi, {mf->nspin(), mf->nkpts_ibz(), mf->nbnd()}));
    chkpt::read_qp_MOs(mpi->node_comm, opt_sMO_qp.value(), opt_sE_qp.value(),
                       prefix, input_grp, input_iter);
    mpi->comm.barrier();

    // AC parameters: always read from the persisted qpGW values (dump_qp_params).
    std::string ac_alg, off_diag_mode;
    double eta = 0.0;
    int Nfit = 0;
    bool from_chkpt = chkpt::read_qp_params(mpi->comm, input_file,
                                            off_diag_mode, eta, ac_alg, Nfit);
    utils::check(from_chkpt,
                 "run_lr_calc: qp_static_sigma requires 'scf/qp_params' in {}. "
                 "Re-run qpGW so the AC parameters are persisted.", input_file);
    utils::check(off_diag_mode == "fermi" || off_diag_mode == "qp_energy",
                 "run_lr_calc: unknown off_diag_mode '{}'", off_diag_mode);
    qp_params = qp_params_t{"sc", ac_alg, Nfit, eta, 1e-8, "qpscf", false, off_diag_mode};
    app_log(2, "LR-qpGW static map: off_diag_mode={}, ac_alg={}, Nfit={}, eta={:.6e} (from checkpoint)",
            off_diag_mode, ac_alg, Nfit, eta);
  }

  // Validate DeltaH0 shape on root, then broadcast the mode count.
  long nmodes = 0;
  if (mpi->comm.root()) {
    utils::check(DeltaH0_mskij_root.has_value(),
                 "run_lr_calc: DeltaH0_mskij must be provided on the MPI global root");
    auto const& dh = *DeltaH0_mskij_root;
    utils::check(dh.shape(1) == mf->nspin() &&
                 dh.shape(2) == mf->nkpts_ibz() &&
                 dh.shape(3) == mf->nbnd() &&
                 dh.shape(4) == mf->nbnd(),
                 "run_lr_calc: DeltaH0_mskij shape mismatch: expected (nmodes,{},{},{},{}), "
                 "got ({},{},{},{},{})",
                 mf->nspin(), mf->nkpts_ibz(), mf->nbnd(), mf->nbnd(),
                 dh.shape(0), dh.shape(1), dh.shape(2), dh.shape(3), dh.shape(4));
    utils::check(dh.shape(0) > 0, "run_lr_calc: DeltaH0_mskij has no perturbations.");
    nmodes = dh.shape(0);
  }
  mpi->comm.broadcast_n(&nmodes, 1, 0);

  // The δX/δV IBC path is per-perturbation in a way this loop does not yet
  // handle: build_lr_ibc mixes mode-independent V_HF/F_PQ with the ΔF_ibc/ΔΣ_ibc
  // built from *this* δX, and lr_driver moves F_PQ_skij out of the IBC object on
  // the first solve, so perturbations 2..n would silently get no F_PQ. Batching
  // it needs DeltaX/DeltaV to grow a mode axis of their own.
  if (nmodes > 1) {
    utils::check(!DeltaX_left_root.has_value() && !DeltaX_right_root.has_value() &&
                 !DeltaV_qPQ_root.has_value(),
                 "run_lr_calc: batching {} perturbations in one call is not supported "
                 "together with DeltaX / DeltaV (the IBC correction is itself "
                 "per-perturbation). Call run_lr once per mode.", nmodes);
  }

  // LR state: perturbation + response quantities. sDeltaH0_skij is one rank-4
  // shm window, refilled from the caller's rank-5 array per perturbation — no
  // node memory scales with nmodes.
  lr_state_t lr_state;
  lr_state.q_vec.emplace(q_vec);
  {
    std::optional<nda::array<ComplexType, 4>> h0_first;
    if (mpi->comm.root())
      h0_first.emplace((*DeltaH0_mskij_root)(0, nda::ellipsis{}));
    lr_state.sDeltaH0_skij.emplace(
        math::shm::make_shared_from_root_input<ComplexType, 4>(*mpi, h0_first));
  }
  lr_state.sDeltaG_tskij.emplace(math::shm::make_shared_array<Array_view_5D_t>(
      *mpi, {ft.nt_f(), mf->nspin(), mf->nkpts_ibz(), mf->nbnd(), mf->nbnd()}));
  lr_state.sDeltaDm_skij.emplace(math::shm::make_shared_array<Array_view_4D_t>(
      *mpi, {mf->nspin(), mf->nkpts_ibz(), mf->nbnd(), mf->nbnd()}));
  lr_state.sDeltaF_skij.emplace(math::shm::make_shared_array<Array_view_4D_t>(
      *mpi, {mf->nspin(), mf->nkpts_ibz(), mf->nbnd(), mf->nbnd()}));
  utils::memlog("run_lr_calc: after sDeltaG/sDeltaDm/sDeltaF/sDeltaH0 alloc");
  // The kernel actually applied. With two_step_order = 0 the perturbative half
  // is never evaluated, so the run is a plain two_step_inner_method run: allocating
  // and persisting datasets for the total method would advertise components
  // (a ΔΣ, an exchange ΔF) that stayed identically zero, and would pay for the
  // W load they need. Everything downstream — allocation, W load, dump_lr — keys
  // off this, and it matches what run_lr derives internally.
  // The exception is order 0 WITH the hessian: its refresh evaluates K_pert once
  // on the returned ΔG, so the run does need the total kernel's ΔΣ store and W
  // load. This is lr_params::has_pert_kernel(), recomputed here because `p` is
  // not filled in until below.
  const bool has_pert_kernel = !pert_kernel.empty() && (pert_order > 0 || hessian);
  const lr_kernel_spec run_kernel = has_pert_kernel ? total_kernel : sc_kernel;
  // What the DATASETS are, which at order 0 is not what was allocated: the
  // schedule applies K_pert never, so ΔF/ΔΣ come from the pure-K_sc solve, and
  // the refresh that does apply it runs after the dump. Describing them with
  // run_kernel would stamp include_exchange on a Hartree-only ΔF, and
  // include_gw_sigma on an identically zero ΔΣ.
  const lr_kernel_spec dumped_kernel = (pert_order > 0) ? run_kernel : sc_kernel;
  const bool dump_hartree = dumped_kernel.hartree;
  const bool dump_exchange = dumped_kernel.exchange;
  const bool dump_gw_sigma = dumped_kernel.has_sigma();
  // Only allocate ΔΣ array when GW is active
  bool include_gw_sigma = run_kernel.has_sigma();
  if (include_gw_sigma) {
    lr_state.sDeltaSigma_tskij.emplace(math::shm::make_shared_array<Array_view_5D_t>(
        *mpi, {ft.nt_f(), mf->nspin(), mf->nkpts_ibz(), mf->nbnd(), mf->nbnd()}));
  }
  // Split-term output (one-shot G0W0): array for the G0·dW0 piece; the total
  // ΔΣ (= dG0·W_c0 + G0·dW0) stays in sDeltaSigma_tskij.
  if (split_sigma_terms)
    utils::check(include_gw_sigma && gw_mode == lr_gw_update_mode::full,
                 "run_lr_calc: split_sigma_terms requires gw_mode='full'.");
  std::optional<math::shm::shared_array<Array_view_5D_t>> opt_sDeltaSigma2;
  if (include_gw_sigma && split_sigma_terms) {
    opt_sDeltaSigma2.emplace(math::shm::make_shared_array<Array_view_5D_t>(
        *mpi, {ft.nt_f(), mf->nspin(), mf->nkpts_ibz(), mf->nbnd(), mf->nbnd()}));
  }

  // Load W and eps_inv_head if GW self-energy is requested. W is either read
  // from disk (explicit screened_interaction_file, else the auto-derived path)
  // or recomputed on the fly from the checkpoint Green's function (RPA). Both
  // hand W to lr_driver as W_c(iω) on the q-dist distribution
  // (utils::lr_W_tau_local_dist).
  using dW_type = std::tuple_element_t<0, decltype(lr_load_W_omega(
      *mpi, eri.corr_eri->get(), ft, input_file, input_grp, input_iter))>;
  std::optional<dW_type> opt_dW;
  std::optional<nda::array<ComplexType, 1>> opt_eps_inv;
  /// The same head on the ω axis; HSEX reads its ν = 0 component.
  std::optional<nda::array<ComplexType, 1>> opt_eps_inv_w;
  std::string div_treatment = "gygi";
  // An LR run must reproduce the divergence treatment of the ground state it
  // linearizes: the q→0 head of Σ_GW is the exchange Madelung term screened by
  // ε⁻¹, so the correlation and exchange corrections only cancel (exactly, in a
  // metal) when both are built the way the unperturbed run built them. Hence
  // neither is a free parameter here — both come from the checkpoint. The one
  // exception is unperturbed='mf_dft', where W0 is built fresh from the DFT G0
  // and the mean-field checkpoint carries no ground-state value to inherit.
  utils::check(unperturbed == "mf_dft" or !io::check_child_exists(pt, "div_treatment"),
               "run_lr_calc: 'div_treatment' is not accepted for unperturbed='{}'. "
               "The LR run always reuses the ground-state value read from "
               "'{}/div_treatment' in {}, so that ΔΣ matches how Σ was built. "
               "Remove the argument (it is only meaningful for unperturbed='mf_dft').",
               unperturbed, input_grp, input_file);
  // Exchange side: needed whenever ΔF carries exchange, independently of W.
  // Absent on pre-stash checkpoints, in which case this falls back to "gygi".
  std::string hf_div_treatment = read_hf_div_treatment(*mpi, input_file, input_grp);
  // HSEX needs W too, for the single ν=0 slice its exchange kernel adds to V.
  // The ω array is released inside lr_setup once that slice is taken.
  if (include_gw_sigma || exchange_static_W) {
    lr_init_timer.start("LR_INIT_LOAD_W");
    bool div_from_chkpt = false;
    if (unperturbed == "mf_dft") {
      // G0W0@DFT: W0 is the RPA screened interaction built from the DFT G0.
      // The checkpoint holds no scGW W (or div_treatment), so recompute and
      // take div_treatment from params (default matches the C++ default).
      utils::check(screened_interaction_file.empty(),
                   "run_lr_calc: unperturbed='mf_dft' builds W0 from G0 (RPA); "
                   "screened_interaction_file must not be set.");
      recompute_W = true;
      div_treatment = io::get_value_with_default<std::string>(pt, "div_treatment", "gygi");
    } else if (qp_static_sigma) {
      // The qpGW checkpoint always stores scf/div_treatment (dump_qp_params), so
      // read it directly; W0 = RPA[G_QP].
      div_treatment = read_div_treatment(*mpi, input_file, input_grp);
      div_from_chkpt = true;
    } else {
      div_treatment = read_div_treatment(*mpi, input_file, input_grp);
      div_from_chkpt = true;
    }
    if (recompute_W) {
      app_log(2, "Recomputing W from checkpoint Green's function (RPA)...");
      auto [dW, eps_inv, eps_inv_w] = recompute_W_and_eps_inv_head(
          *mpi, eri.corr_eri->get(), ft, sG_tskij, div_treatment);
      opt_dW.emplace(std::move(dW));
      opt_eps_inv.emplace(std::move(eps_inv));
      opt_eps_inv_w.emplace(std::move(eps_inv_w));
    } else {
      auto [dW, eps_inv, eps_inv_w, div_str] = lr_load_W_omega(
          *mpi, eri.corr_eri->get(), ft, input_file, input_grp, input_iter,
          screened_interaction_file);
      opt_dW.emplace(std::move(dW));
      opt_eps_inv.emplace(std::move(eps_inv));
      opt_eps_inv_w.emplace(std::move(eps_inv_w));
      // W carries its own stash; a disagreement means the W file and the
      // checkpoint came from different runs, and Σ would mix two treatments.
      utils::check(!div_from_chkpt or div_str == div_treatment,
                   "run_lr_calc: div_treatment mismatch. The screened interaction "
                   "reports '{}' while '{}/div_treatment' in {} reports '{}'. The two "
                   "must come from the same ground-state run.",
                   div_str, input_grp, input_file, div_treatment);
      div_treatment = div_str;
    }
    lr_init_timer.stop("LR_INIT_LOAD_W");
  }

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

  utils::memlog("run_lr_calc: before driver.lr_setup");

  // Create lr_driver and run unified SCF loop
  lr_driver driver(dyson, q_vec);
  lr_state.kpq_map.emplace(driver.kpq_map());
  auto& thc = eri.corr_eri->get();
  auto* pDeltaSigma = lr_state.sDeltaSigma_tskij ? &(*lr_state.sDeltaSigma_tskij) : nullptr;
  auto* pDeltaSigma2 = opt_sDeltaSigma2 ? &(*opt_sDeltaSigma2) : nullptr;
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

  // LR-qpGW static-map inputs + output ΔV_QPGW array.
  std::optional<lr_qp_static_params> opt_qp_static;
  std::optional<math::shm::shared_array<Array_view_4D_t>> opt_sDeltaVcorr;
  if (qp_static_sigma) {
    lr_qp_static_params qps;
    qps.qp_params = qp_params;
    qps.sMO_skia = &opt_sMO_qp.value();
    qps.sE_ska = &opt_sE_qp.value();
    qps.mu = mu;
    opt_qp_static.emplace(qps);
    opt_sDeltaVcorr.emplace(math::shm::make_shared_array<Array_view_4D_t>(
        *mpi, {mf->nspin(), mf->nkpts_ibz(), mf->nbnd(), mf->nbnd()}));
  }
  const lr_qp_static_params* pQpStatic = opt_qp_static ? &(*opt_qp_static) : nullptr;
  auto* pDeltaVcorr = opt_sDeltaVcorr ? &(*opt_sDeltaVcorr) : nullptr;

  // Under output_aux_fock, catches the precomputed IBC aux→primary correction
  //   DeltaF_ibc = δX†·F_PQ·X + X†·F_PQ·δX
  // when DeltaX is in play, plus F_PQ (unperturbed V_HF in aux basis) and ΔF_PQ
  // (LR Fock in aux basis at convergence). The Python phonon post-processors build
  // the ΔΔF_ibc T1/T3 curvature terms from these; there is no C++ path for them.
  // All three ride the same flag so the default checkpoint keeps exactly the
  // dataset set it had before. Each is left at size 0 when its branch does not
  // run, and only the populated ones are persisted.
  nda::array<ComplexType, 4> DeltaF_ibc_skij;
  nda::array<ComplexType, 4> F_PQ_skij;
  nda::array<ComplexType, 4> DeltaF_PQ_skij;

  lr_params p;
  p.include_hartree  = include_hartree;
  p.include_exchange = include_exchange;
  p.gw_mode          = gw_mode;
  p.include_xc       = include_xc;
  p.exchange_static_W  = exchange_static_W;
  p.max_iter         = max_iter;
  p.tol              = tol;
  p.fix_density      = fix_density;
  // dump_lr below is the only reader of lr_state.sDeltaG_tskij, and it writes it
  // only under this flag, so it also decides whether ΔG(τ) is replicated at all.
  p.save_DeltaG      = save_DeltaG;
  p.iter_params      = iter_params;
  p.eps_inv_head     = opt_eps_inv ? &(*opt_eps_inv) : nullptr;
  p.eps_inv_head_w   = opt_eps_inv_w ? &(*opt_eps_inv_w) : nullptr;
  p.div_treatment    = div_treatment;
  p.hf_div_treatment = hf_div_treatment;
  p.sDeltaX_left     = pDeltaX_left;
  p.sDeltaX_right    = pDeltaX_right;
  p.Dm_ab            = Dm_ab_ptr;
  p.DeltaV_qPQ       = pDeltaV_qPQ;
  p.qp_static        = pQpStatic;
  p.split_sigma_terms = (pDeltaSigma2 != nullptr);
  p.keep_F_PQ        = output_aux_fock;
  p.hessian_nmodes = hessian ? nmodes : 0;
  // Split-kernel schedule. It selects which solvers lr_setup builds and how many
  // ΔΣ-sized buffers it allocates, so it has to be in `p` rather than per-solve.
  p.sc_kernel        = sc_kernel;
  p.pert_kernel      = pert_kernel;
  p.pert_order       = pert_order;
  if (lr_two_step) p.outer_accel = outer_params;

  driver.lr_setup(sG_tskij, thc, opt_dW ? &(*opt_dW) : nullptr, p);

  nda::array<long, 1> niter_m(nmodes);
  nda::array<double, 1> Delta_mu_m(nmodes);

  // Stationary hessian estimator. Built before the mode loop: it owns the striped
  // per-mode stores, so its lifetime spans both passes. The k-weight and the spin
  // factor follow lr_dyson::compute_lr_Nelec, which is what makes the C++ matrices
  // directly comparable with the Python reference implementation.
  std::optional<lr_hessian_t> opt_hessian;
  if (hessian) {
    nda::array<double, 1> k_weight(mf->k_weight());
    const double spin_factor = (mf->nspin() == 1 && mf->npol() == 1) ? 2.0 : 1.0;
    opt_hessian.emplace(mpi, ft, k_weight, spin_factor, nmodes, include_gw_sigma,
                     mf->nspin(), mf->nkpts_ibz(), mf->nbnd());
  }

  for (long m = 0; m < nmodes; ++m) {
    if (nmodes > 1)
      app_log(1, "\n===== perturbation mode {} of {} =====", m + 1, nmodes);

    // Refill the shm ΔH0 window from this perturbation's slice: root writes,
    // then broadcast_to_nodes publishes it (its leading node_sync makes root's
    // write visible before the internode broadcast, and its trailing one makes
    // the result visible to every rank on every node). Mode 0 is already in the
    // window from the allocation above. Costs one 0.1 GB broadcast, no memory.
    if (m > 0) {
      auto& sDeltaH0 = lr_state.sDeltaH0_skij.value();
      if (mpi->comm.root())
        sDeltaH0.local() = (*DeltaH0_mskij_root)(m, nda::ellipsis{});
      sDeltaH0.broadcast_to_nodes(0);
      mpi->comm.barrier();
    }

    // K_pert evaluations actually made for THIS perturbation. With an outer
    // tolerance it is not two_step_order, and it is the cost figure a
    // downstream reader needs.
    int n_pert_applied = 0;

    auto [niter, Delta_mu] = driver.lr_solve_one(
        lr_state.sDeltaG_tskij.value(), lr_state.sDeltaDm_skij.value(),
        lr_state.sDeltaF_skij.value(), pDeltaSigma,
        sG_tskij, lr_state.sDeltaH0_skij.value(), thc, p,
        pDeltaSigma2, pDeltaVcorr,
        output_aux_fock ? &DeltaF_ibc_skij : nullptr,
        output_aux_fock ? &F_PQ_skij : nullptr,
        output_aux_fock ? &DeltaF_PQ_skij : nullptr,
        &n_pert_applied);
    niter_m(m) = niter;
    Delta_mu_m(m) = Delta_mu;
    lr_state.Delta_mu.emplace(Delta_mu);
    mpi->comm.barrier();

    utils::memlog(fmt::format("run_lr_calc: after lr_solve_one (mode {}/{})", m + 1, nmodes));

    // Write results. A single-perturbation run keeps writing into
    // "linear_response/" so its checkpoint layout is unchanged.
    lr_init_timer.start("LR_DUMP");
    // ε⁻¹(iν=0) for an HSEX run, absent otherwise. dump_lr keys the whole HSEX
    // provenance block off whether this is engaged.
    std::optional<double> hsex_head;
    if (exchange_static_W) hsex_head = driver.hsex_head_factor();
    chkpt::dump_lr(mpi->comm, output + ".mbpt.h5", q_vec,
                   lr_state.sDeltaG_tskij.value(), lr_state.sDeltaDm_skij.value(),
                   lr_state.sDeltaF_skij.value(),
                   dump_gw_sigma ? pDeltaSigma : nullptr,
                   Delta_mu, niter,
                   dump_hartree, dump_exchange, dump_gw_sigma,
                   pDeltaSigma2, pDeltaVcorr,
                   nmodes > 1 ? std::optional<long>(m + 1) : std::nullopt,
                   save_DeltaG, nbnd_save,
                   gw_mode_str(gw_mode_of(dumped_kernel)),
                   lr_two_step, two_step_inner_method_in,
                   static_cast<int>(two_step_order),
                   outer_active, two_step_outer_alg, two_step_outer_tol,
                   n_pert_applied, has_pert_kernel && hessian, hsex_head);
    mpi->comm.barrier();
    lr_init_timer.stop("LR_DUMP");

    // Store this mode's operands for the stationary hessian estimator, after the
    // dump so what lands on disk stays the mixed iterate exactly as before. A copy
    // out and nothing else: the raw (pre-mixing) kernel output lives in the
    // accelerator's per-solve history and ΔDm/ΔG in shm windows, so both have to be
    // taken here, before the next perturbation destroys them. Every contraction
    // happens after this loop.
    if (opt_hessian) {
      driver.get_full_kernel_result(
          lr_state.sDeltaF_skij.value(), pDeltaSigma,
          lr_state.sDeltaDm_skij.value(), lr_state.sDeltaG_tskij.value(),
          sG_tskij, thc, p);
      opt_hessian->store_mode(m, lr_state.sDeltaDm_skij.value(),
                           lr_state.sDeltaH0_skij.value(),
                           lr_state.sDeltaF_skij.value(), pDeltaSigma,
                           include_gw_sigma ? &lr_state.sDeltaG_tskij.value() : nullptr);
      mpi->comm.barrier();
    }

    // Persist the IBC aux→primary correction and the aux-basis Fock matrices
    // alongside the LR results. Hellmann-Feynman-style δX gradient consumers
    // read DeltaF_ibc as
    //   dE/dx = spin · Σ_{s,k} w_k · Tr(DeltaF_ibc[s,k] · Dm[s,k]),
    // and the phonon drivers contract F_PQ / ΔF_PQ against δX for the ΔΔF_ibc
    // curvature terms. All three are per-perturbation, so they go into the same
    // group as the rest of this perturbation's output. (DeltaF_ibc and F_PQ
    // require IBC, which a batched call rejects, so in practice only ΔF_PQ ever
    // repeats.)
    if (mpi->comm.root() && (DeltaF_ibc_skij.size() > 0 ||
                             F_PQ_skij.size() > 0 ||
                             DeltaF_PQ_skij.size() > 0)) {
      h5::file file(output + ".mbpt.h5", 'a');
      h5::group grp(file);
      auto lr_grp = grp.has_subgroup("linear_response") ?
                    grp.open_group("linear_response") :
                    grp.create_group("linear_response");
      std::string where = "linear_response/";
      if (nmodes > 1) {
        std::string mg = "mode" + std::to_string(m + 1);
        lr_grp = lr_grp.has_subgroup(mg) ? lr_grp.open_group(mg) : lr_grp.create_group(mg);
        where += mg + "/";
      }
      if (DeltaF_ibc_skij.size() > 0) {
        nda::h5_write(lr_grp, "DeltaF_ibc_skij", DeltaF_ibc_skij, false);
        app_log(2, "  - DeltaF_ibc_skij written to \"{}\"", where);
      }
      if (F_PQ_skij.size() > 0) {
        nda::h5_write(lr_grp, "F_PQ_skij", F_PQ_skij, false);
        app_log(2, "  - F_PQ_skij written to \"{}\"", where);
      }
      if (DeltaF_PQ_skij.size() > 0) {
        nda::h5_write(lr_grp, "DeltaF_PQ_skij", DeltaF_PQ_skij, false);
        app_log(2, "  - DeltaF_PQ_skij written to \"{}\"", where);
      }
    }
    mpi->comm.barrier();
  }

  // Every mode is stored now, so the estimator can be evaluated. One call does all
  // of it: per mode it rebuilds the stored raw ΔF/ΔΣ, runs the extra Dyson on
  // ΔV = ΔH0 + ΔF + ΔΣ and contracts the result against every stored mode, then
  // reduces and combines. It works in the mode loop's own shm arrays — free scratch
  // after the last dump — so no array is allocated here and ΔDm' is never
  // persisted.
  if (opt_hessian) {
    auto c1 = opt_hessian->evaluate(
        driver.dyson(), lr_state.sDeltaH0_skij.value(),
        lr_state.sDeltaDm_skij.value(), lr_state.sDeltaF_skij.value(),
        pDeltaSigma,
        include_gw_sigma ? &lr_state.sDeltaG_tskij.value() : nullptr,
        DeltaH0_mskij_root, fix_density);
    // The K_pert refresh is the one clock lr_driver owns; the rest of the table is
    // lr_hessian's own.
    opt_hessian->print_timers(driver.hessian_refresh_sec(),
                              driver.hessian_refresh_calls());

    // Top-level linear_response/ group, NOT the per-mode subgroups: these are
    // mode-PAIR matrices. dump_lr rebinds its group to mode{m} for a batched run,
    // which is why the write does not go through it.
    if (mpi->comm.root()) {
      // D7: the block spans the perturbations solved in THIS call, so its axes are
      // (npert, npert) and never padded to the full mode count.
      //
      // The index written here is the 0-based perturbation index WITHIN the call
      // — the only thing the C++ knows, since run_lr is handed a bare ΔH0 stack
      // and no mode numbering. It is deliberately NOT called "hessian_modes":
      // the phonon drivers write a dataset of that name holding 1-based phonon
      // mode numbers, and a consumer that confused the two would be off by one
      // and on the wrong subset.
      nda::array<long, 1> hessian_call_index(nmodes);
      for (long m = 0; m < nmodes; ++m) hessian_call_index(m) = m;

      h5::file file(output + ".mbpt.h5", 'a');
      h5::group grp(file);
      auto lr_grp = grp.has_subgroup("linear_response") ?
                    grp.open_group("linear_response") :
                    grp.create_group("linear_response");
      nda::h5_write(lr_grp, "hessian", c1.hessian_plain, false);
      nda::h5_write(lr_grp, "hessian_sym", c1.hessian_sym, false);
      nda::h5_write(lr_grp, "hessian_M", c1.M, false);
      nda::h5_write(lr_grp, "hessian_M_prime", c1.M_prime, false);
      nda::h5_write(lr_grp, "hessian_static_prime", c1.static_prime, false);
      nda::h5_write(lr_grp, "hessian_call_index", hessian_call_index, false);
      nda::h5_write(lr_grp, "Delta_mu_improved", c1.Delta_mu_improved, false);
      h5::h5_write(lr_grp, "hessian_herm_dev", c1.herm_plain);
      h5::h5_write(lr_grp, "hessian_sym_herm_dev", c1.herm_sym);
      h5::h5_write(lr_grp, "hessian_M_herm_dev", c1.herm_M);
      app_log(2, "  - hessian / hessian_sym ({0}x{0}) written to \"linear_response/\"",
              nmodes);
    }
    mpi->comm.barrier();
  }

  // Reported here, not with the LR_INIT block, because it accumulates over the mode loop.
  app_log(2, "\n  LR checkpoint write: {0:8.3f} sec  {1:4d} calls\n",
          lr_init_timer.elapsed("LR_DUMP"), lr_init_timer.number_of_calls("LR_DUMP"));

  utils::memlog("run_lr_calc: exit (before RAII)");
  return std::make_tuple(std::move(niter_m), std::move(Delta_mu_m));
}


// run_lr_calc instantiations (THC only — lr_hf and lr_gw require THC).
// Cholesky LR Dyson (one-shot, no HF/GW) was never used externally,
// so Cholesky ERI combinations are intentionally not instantiated here.
template std::tuple<nda::array<long, 1>, nda::array<double, 1>> run_lr_calc(
    mb_eri_t<thc_reader_t, thc_reader_t, thc_reader_t, thc_reader_t>&,
    ptree const&,
    nda::array<double, 1> const&,
    std::optional<nda::array<ComplexType, 5>> const&,
    bool, bool, lr_gw_update_mode, int, double, bool, const lr_iter_params&,
    std::optional<nda::array<ComplexType, 4>> const&,
    std::optional<nda::array<ComplexType, 4>> const&,
    std::optional<nda::array<ComplexType, 3>> const&);



/**
 * Linear response GW self-energy with fixed W (term 1).
 *
 * Computes ΔΣ = -ΔG ⊙ W_c + div_corr (R-space).
 * Reads W from thc_screened_interaction.h5 and eps_inv_head from checkpoint.
 */
template<typename eri_t>
nda::array<ComplexType, 5> lr_gw_sigma_DeltaG_calc(
    eri_t &eri, ptree const& pt,
    nda::array<double, 1> const& q_pert,
    std::optional<nda::array<ComplexType, 5>> const& DeltaG_tskij_root) {

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

  auto sDeltaG_tskij = math::shm::make_shared_from_root_input<ComplexType, 5>(
      *mpi, DeltaG_tskij_root);

  // Load W and eps_inv_head using helper, in the (t,q) layout lr_precompute_W_tRPQ wants
  auto [dW_tqPQ, eps_inv_head, div_treatment] = load_W_and_eps_inv_head(
      *mpi, thc, input_file, input_grp, input_iter, nt);

  // Divergence correction flag (default true)
  auto div_corr = io::get_value_with_default<bool>(pt, "div_corr", true);

  // Read overlap matrix from checkpoint for divergence correction
  auto sS_skij = math::shm::make_shared_array<Array_view_4D_t>(
      *mpi, {ns, nkpts_ibz, nbnd, nbnd});
  if (div_corr) {
    chkpt::read_ovlp(mpi->node_comm, prefix, sS_skij);
  }

  // Pre-transform W to R-space: transpose (q,t)→(t,R), with q→R FT
  auto dW_tRPQ = lr_precompute_W_tRPQ(dW_tqPQ, thc);

  // Create lr_gw solver and compute ΔΣ
  solvers::lr_gw lr(&ft, q_pert, div_treatment);

  auto sDeltaSigma_tskij = math::shm::make_shared_array<Array_view_5D_t>(
      *mpi, {nt, ns, nkpts_ibz, nbnd, nbnd});

  // Divergence correction term 1 (all q): ΔΣ^div += -madelung * eps_inv_head * S(k+q) · ΔG · S(k)
  // The evaluator applies it when handed the overlap and the head; withholding
  // them is how div_corr = false gets the bare convolution for FD tests.
  auto S_loc = sS_skij.local();
  lr.evaluate_sigma_DeltaG(sDeltaSigma_tskij, sDeltaG_tskij.local(), dW_tRPQ, thc,
                           nullptr,
                           div_corr ? &S_loc : nullptr,
                           div_corr ? &eps_inv_head : nullptr);
  mpi->comm.barrier();

  // Rank-0-only return; non-root ranks return an empty array (→ None in Python).
  nda::array<ComplexType, 5> out;
  if (mpi->comm.root()) {
    out = sDeltaSigma_tskij.local();
  }
  mpi->comm.barrier();
  return out;
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
    std::optional<nda::array<ComplexType, 5>> const& G_tskij_root,
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

  auto sG_tskij = math::shm::make_shared_from_root_input<ComplexType, 5>(
      *mpi, G_tskij_root);

  // Load W and eps_inv_head using helper
  auto [dW_tqPQ, eps_inv_head, div_treatment] = load_W_and_eps_inv_head(*mpi, thc, input_file, input_grp, input_iter, nt);

  // Read overlap matrix from checkpoint for divergence correction
  auto sS_skij = math::shm::make_shared_array<Array_view_4D_t>(
      *mpi, {ns, nkpts_ibz, nbnd, nbnd});
  chkpt::read_ovlp(mpi->node_comm, prefix, sS_skij);

  // Use standard gw_t code path
  solvers::gw_t gw(&ft, div_treatment);

  auto sSigma_tskij = math::shm::make_shared_array<Array_view_5D_t>(
      *mpi, {nt, ns, nkpts_ibz, nbnd, nbnd});

  gw.eval_Sigma_all(sG_tskij.local(), dW_tqPQ, sSigma_tskij, thc, "R");
  if (div_corr) {
    gw.Sigma_div_correction(sSigma_tskij, sG_tskij.local(), sS_skij.local(), thc, eps_inv_head);
  }
  mpi->comm.barrier();

  // Rank-0-only return.
  nda::array<ComplexType, 5> out;
  if (mpi->comm.root()) {
    out = sSigma_tskij.local();
  }
  mpi->comm.barrier();
  return out;
}


/**
 * Band-basis perturbation from a local potential given in the THC aux basis.
 *
 * Runs the aux→primary map of the diagonal Hartree channel
 * (lr_thc_comm::aux_to_primary_diagonal) on a caller-supplied u_P instead of on
 * ΔJ_P, so an external scalar field projected onto the interpolating vectors
 * (u_P = ∫ dr ζ_P(r) u(r)) becomes a perturbation run_lr can consume.
 *
 * Per mode, and summed over the diagonal polarization blocks p, the result is
 *
 *   ΔH0_ij(k) = Σ_(p,P) X_p(k+q)†_iP · u_P · X_p(k)_Pj
 *             = Σ_(p,P) conj(X_p(k+q)_Pi) · u_P · X_p(k)_Pj
 *
 * i.e. a row-scaled X†X with no (P,Q) gemm, which is exactly what the diagonal
 * aux→primary routine already does; hence the reuse rather than an open-coded
 * pair of gemms here.
 *
 * Each (s, k_ibz) block is written by exactly one rank, as in
 * lr_hf::thc_lr_hartree_only, and reduced once per mode.
 */
template<typename eri_t>
nda::array<ComplexType, 5> lr_DeltaH0_from_thc_aux_calc(
    eri_t &eri,
    nda::array<double, 1> const& q_pert,
    std::optional<nda::array<ComplexType, 2>> const& u_mP_root) {

  auto& thc = eri.corr_eri->get();
  auto mf = thc.MF();
  auto& mpi = thc.mpi();

  long ns        = mf->nspin();
  long npol      = mf->npol();
  long nkpts     = mf->nkpts();
  long nkpts_ibz = mf->nkpts_ibz();
  long nbnd      = mf->nbnd();
  long NP        = thc.Np();

  // An arbitrary u_P at finite q is not symmetric under the little group of q, so the
  // (ns, nkpts_ibz, ...) perturbation this returns is only the full-BZ object when the
  // IBZ is the full BZ. run_lr_calc's shape check accepts nkpts_ibz either way, so the
  // requirement is enforced here, at the only entry point that manufactures such a
  // perturbation.
  utils::check(nkpts_ibz == nkpts,
               "lr_DeltaH0_from_thc_aux: needs a mean field without symmetry "
               "(nkpts_ibz = {} != nkpts = {}). An aux-basis potential at finite q "
               "breaks the crystal symmetry, so a symmetry-reduced k-set cannot carry "
               "the resulting perturbation.", nkpts_ibz, nkpts);

  auto su_mP = math::shm::make_shared_from_root_input<ComplexType, 2>(*mpi, u_mP_root);
  long nmodes = su_mP.shape()[0];
  utils::check(su_mP.shape()[1] == NP,
               "lr_DeltaH0_from_thc_aux: u_mP has {} aux components, expected Np = {}.",
               su_mP.shape()[1], NP);

  nda::array<int, 1> kpq_map(nkpts);
  utils::calculate_kpq_map(mf->kpts_crystal(), q_pert, kpq_map);

  using Array_view_4D_t = nda::array_view<ComplexType, 4>;
  auto sDeltaH0_skij = math::shm::make_shared_array<Array_view_4D_t>(
      *mpi, std::array<long, 4>{ns, nkpts_ibz, nbnd, nbnd});

  nda::array<ComplexType, 5> DeltaH0_mskij;
  if (mpi->comm.root())
    DeltaH0_mskij = nda::array<ComplexType, 5>({nmodes, ns, nkpts_ibz, nbnd, nbnd});

  auto kp_map = mf->ks_to_k(0);
  long nsk_ibz = ns * nkpts_ibz;
  nda::range sk_rng(std::min<long>(mpi->comm.rank(), nsk_ibz), nsk_ibz, mpi->comm.size());

  for (long m = 0; m < nmodes; ++m) {
    sDeltaH0_skij.set_zero();
    auto u_P = su_mP.local()(m, nda::range::all);
    auto DeltaH0_loc = sDeltaH0_skij.local();
    for (auto ip : nda::range(npol))
      solvers::lr_thc_comm::aux_to_primary_diagonal(ip, ip, ComplexType(1.0), u_P,
                                                    DeltaH0_loc, sk_rng, thc,
                                                    kp_map, kpq_map);
    sDeltaH0_skij.win().fence();
    sDeltaH0_skij.all_reduce_parallel();
    if (mpi->comm.root()) DeltaH0_mskij(m, nda::ellipsis{}) = sDeltaH0_skij.local();
  }

  return DeltaH0_mskij;
}


/**
 * Linear response polarization ΔP = -ΔG·G - G·ΔG (R-space).
 *
 * Creates lr_rpa_pi solver, calls evaluate_lr_Pi,
 * copies distributed result to regular array.
 *
 * When DeltaX_left (δ^q X) and DeltaX_right (δ^{-q} X) are both provided,
 * an lr_ibc_DeltaX struct is constructed on the fly and passed to
 * evaluate_lr_Pi so the primary→aux IBC correction is applied.
 */
template<typename eri_t>
nda::array<ComplexType, 4> lr_gw_Pi_calc(
    eri_t &eri,
    nda::array<double, 1> const& q_pert,
    std::optional<nda::array<ComplexType, 5>> const& G_tskij_root,
    std::optional<nda::array<ComplexType, 5>> const& DeltaG_tskij_root,
    std::optional<nda::array<ComplexType, 4>> const& DeltaX_left_root,
    std::optional<nda::array<ComplexType, 4>> const& DeltaX_right_root) {

  auto& thc = eri.corr_eri->get();
  auto mf = thc.MF();
  auto& mpi = thc.mpi();

  long nkpts = mf->nkpts();
  long Np = thc.Np();

  // Distribute G and DeltaG into node-local shared-memory windows.
  using Array_view_4D_t = nda::array_view<ComplexType, 4>;
  auto sG_tskij = math::shm::make_shared_from_root_input<ComplexType, 5>(
      *mpi, G_tskij_root);
  auto sDeltaG_tskij = math::shm::make_shared_from_root_input<ComplexType, 5>(
      *mpi, DeltaG_tskij_root);

  long nt = sG_tskij.shape()[0];
  long nt_half = (nt % 2 == 0) ? nt / 2 : nt / 2 + 1;

  // Create lr_rpa_pi solver and compute ΔP
  solvers::lr_rpa_pi lr(q_pert);

  // Decide whether IBC correction is requested (rank-0 input → bcast presence flag).
  bool has_deltax_root = false;
  if (mpi->comm.root()) {
    has_deltax_root = DeltaX_left_root.has_value() && DeltaX_right_root.has_value();
  }
  int has_deltax_int = has_deltax_root ? 1 : 0;
  mpi->comm.broadcast_n(&has_deltax_int, 1, 0);

  // Build IBC struct on demand. The IBC path for Pi only uses the
  // primary→aux correction, which needs DeltaX / DeltaX_minusq, q_vec,
  // kpq_map, and sG_tskij. The aux→primary pieces (DeltaF_ibc_skij,
  // sDeltaSigma_ibc_tskij) are only consumed by HF/GW self-energy paths
  // and remain default-initialised here.
  std::optional<math::shm::shared_array<Array_view_4D_t>> opt_sDeltaX_left, opt_sDeltaX_right;
  std::optional<lr_ibc_DeltaX> opt_ibc;
  if (has_deltax_int) {
    opt_sDeltaX_left.emplace(
        math::shm::make_shared_from_root_input<ComplexType, 4>(*mpi, DeltaX_left_root));
    opt_sDeltaX_right.emplace(
        math::shm::make_shared_from_root_input<ComplexType, 4>(*mpi, DeltaX_right_root));

    // Compute kpq_map (full BZ) — mirror of lr_rpa_pi::_init_kpq_map.
    nda::array<int, 1> kpq_map(nkpts);
    utils::calculate_kpq_map(mf->kpts_crystal(), q_pert, kpq_map);

    opt_ibc.emplace(lr_ibc_DeltaX{
        opt_sDeltaX_left->local(), opt_sDeltaX_right->local(),
        q_pert, std::move(kpq_map),
        nullptr, &sG_tskij,
        nda::array<ComplexType, 4>{}, std::nullopt
    });
  }

  // Precompute G^R(τ) and G^R(β−τ) used in evaluate_lr_Pi.
  auto [dG_tsRPQ, dG_mtau_tsRPQ] = lr_precompute_G_R_pair(sG_tskij.local(), thc);

  auto dDeltaPi = lr.evaluate_lr_Pi(sG_tskij.local(),
                                     sDeltaG_tskij.local(),
                                     thc,
                                     dG_tsRPQ, dG_mtau_tsRPQ,
                                     opt_ibc ? &(*opt_ibc) : nullptr);
  mpi->comm.barrier();

  // Gather to node-local shared window, return on rank 0 only.
  auto sDeltaPi_tqPQ = math::shm::make_shared_array<Array_view_4D_t>(
      *mpi, std::array<long, 4>{nt_half, nkpts, Np, Np});
  math::nda::gather_to_shm(dDeltaPi, sDeltaPi_tqPQ);

  nda::array<ComplexType, 4> out;
  if (mpi->comm.root()) {
    out = sDeltaPi_tqPQ.local();
  }
  mpi->comm.barrier();
  return out;
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
    std::optional<nda::array<ComplexType, 5>> const& G_tskij_root) {

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

  auto sG_tskij = math::shm::make_shared_from_root_input<ComplexType, 5>(
      *mpi, G_tskij_root);

  long nt = ft.nt_f();
  utils::check(sG_tskij.shape()[0] == nt,
               "gw_evaluate_Pi_calc: G tau dimension {} != IAFT nt_f {}",
               sG_tskij.shape()[0], nt);
  long nt_half = (nt % 2 == 0) ? nt / 2 : nt / 2 + 1;

  // Use standard scr_coulomb_t to compute P
  solvers::scr_coulomb_t scr_eri(&ft, "rpa");

  auto dPi_tqPQ = scr_eri.eval_Pi_qdep(sG_tskij.local(), thc);
  mpi->comm.barrier();

  // Gather to node-local shared window, return on rank 0 only.
  using Array_view_4D_t = nda::array_view<ComplexType, 4>;
  auto sPi_tqPQ = math::shm::make_shared_array<Array_view_4D_t>(
      *mpi, std::array<long, 4>{nt_half, nkpts, Np, Np});
  math::nda::gather_to_shm(dPi_tqPQ, sPi_tqPQ);

  nda::array<ComplexType, 4> out;
  if (mpi->comm.root()) {
    out = sPi_tqPQ.local();
  }
  mpi->comm.barrier();
  return out;
}


/**
 * Linear response screened interaction ΔW = (Z+W_c) · ΔΠ · (Z+W_c).
 *
 * Reads W_c(τ) from thc_screened_interaction.h5 and IAFT from checkpoint,
 * then calls lr_scr_coulomb_t::solve_lr_dyson_W to compute ΔW_c(τ).
 */
template<typename eri_t>
nda::array<ComplexType, 4> lr_gw_W_calc(
    eri_t &eri, ptree const& pt,
    nda::array<double, 1> const& q_pert,
    std::optional<nda::array<ComplexType, 4>> const& DeltaPi_tqPQ_root) {

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

  using Array_view_4D_t = nda::array_view<ComplexType, 4>;
  auto sDeltaPi_tqPQ_in = math::shm::make_shared_from_root_input<ComplexType, 4>(
      *mpi, std::array<long, 4>{nt_half, nkpts, NP, NP},
      DeltaPi_tqPQ_root);

  // Load W_c from thc_screened_interaction.h5 and take it in ω, the axis the LR
  // W Dyson works on.
  // eps_inv_head: not yet used here, will be needed for div_corr (TODO)
  auto [dW_wqPQ, eps_inv_head, eps_inv_head_w, div_treatment] = lr_load_W_omega(
      *mpi, thc, ft, input_file, input_grp, input_iter);
  (void)div_treatment;

  // Convert ΔΠ shared-memory window → distributed array (τ-dist)
  auto [pi_pgrid, pi_bsize] = utils::lr_W_q_local_dist(mpi->comm.size(), nt_half, NP);
  auto dDeltaPi_tqPQ = math::nda::make_distributed_array<local_Array_4D_t>(
      mpi->comm, pi_pgrid, {nt_half, nkpts, NP, NP}, pi_bsize);
  {
    auto pi_src = sDeltaPi_tqPQ_in.local();
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

  // LR Dyson: ΔΠ(τ) → ΔW_c(τ) via ΔW_c(iω) = W_full(q+p) · ΔΠ · W_full(q)
  solvers::lr_scr_coulomb_t lr_scr(&ft, q_pert);
  // W_c has no correlation-only consumer here, so += Z right away, in place.
  lr_scr.lr_Wc_to_Wfull(dW_wqPQ, thc);
  lr_scr.solve_lr_dyson_W(dDeltaPi_tqPQ, dW_wqPQ, thc);
  // dDeltaPi_tqPQ now contains ΔW_c(τ)
  mpi->comm.barrier();

  // Gather to node-local shared window, return on rank 0 only.
  auto sDeltaW_tqPQ = math::shm::make_shared_array<Array_view_4D_t>(
      *mpi, std::array<long, 4>{nt_half, nkpts, NP, NP});
  math::nda::gather_to_shm(dDeltaPi_tqPQ, sDeltaW_tqPQ);

  nda::array<ComplexType, 4> out;
  if (mpi->comm.root()) {
    out = sDeltaW_tqPQ.local();
  }
  mpi->comm.barrier();
  return out;
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
    std::optional<nda::array<ComplexType, 4>> const& Pi_tqPQ_root) {

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

  // Require no symmetry
  utils::check(mf->nqpts() == mf->nqpts_ibz() and mf->nqpts() == nkpts,
               "gw_evaluate_W_from_Pi_calc: No symmetry required. nqpts={}, nqpts_ibz={}, nkpts={}",
               mf->nqpts(), mf->nqpts_ibz(), nkpts);

  using Array_view_4D_t = nda::array_view<ComplexType, 4>;
  auto sPi_tqPQ_in = math::shm::make_shared_from_root_input<ComplexType, 4>(
      *mpi, std::array<long, 4>{nt_half, nkpts, NP, NP}, Pi_tqPQ_root);

  // Scatter shared-memory input into a distributed array (each rank owns
  // a slab). The shared window is already synchronised across nodes by
  // make_shared_from_root_input, so no broadcast is needed.
  auto pgrid = compute_proc_grid_4D(mpi->comm.size(), {nt_half, nkpts, NP, NP},
                                     {false, false, false, false});
  auto dPi_tqPQ = make_distributed_array<local_Array_4D_t>(
      mpi->comm, pgrid, {nt_half, nkpts, NP, NP});
  {
    auto pi_src = sPi_tqPQ_in.local();
    auto pi_loc = dPi_tqPQ.local();
    auto [o0, o1, o2, o3] = dPi_tqPQ.origin();
    auto [n0, n1, n2, n3] = dPi_tqPQ.local_shape();
    for (long i0 = 0; i0 < n0; ++i0)
      for (long i1 = 0; i1 < n1; ++i1)
        for (long i2 = 0; i2 < n2; ++i2)
          for (long i3 = 0; i3 < n3; ++i3)
            pi_loc(i0, i1, i2, i3) = pi_src(i0 + o0, i1 + o1, i2 + o2, i3 + o3);
    mpi->comm.barrier();
  }

  // Use scr_coulomb_t to solve W Dyson equation: Π → W_c(τ)
  solvers::scr_coulomb_t scr(&ft, "rpa");
  auto dW_tqPQ = scr.dyson_W_from_Pi_tau<false>(dPi_tqPQ, thc, true);
  mpi->comm.barrier();

  // Gather to node-local shared window, return on rank 0 only.
  auto sW_tqPQ = math::shm::make_shared_array<Array_view_4D_t>(
      *mpi, std::array<long, 4>{nt_half, nkpts, NP, NP});
  math::nda::gather_to_shm(dW_tqPQ, sW_tqPQ);

  nda::array<ComplexType, 4> out;
  if (mpi->comm.root()) {
    out = sW_tqPQ.local();
  }
  mpi->comm.barrier();
  return out;
}


/**
 * Linear response GW self-energy term 2: -G ⊙ ΔW + div_corr.
 *
 * Computes ΔΣ = -G ⊙ ΔW from a pre-computed DeltaW.
 * Uses lr_gw::evaluate_sigma_DeltaW for the G⊙ΔW convolution.
 * At q_pert=0, also applies term 2 divergence correction using Δeps_inv_head from ΔW.
 */
template<typename eri_t>
nda::array<ComplexType, 5> lr_gw_sigma_DeltaW_calc(
    eri_t &eri, ptree const& pt,
    nda::array<double, 1> const& q_pert,
    std::optional<nda::array<ComplexType, 5>> const& G_tskij_root,
    std::optional<nda::array<ComplexType, 4>> const& DeltaW_qtPQ_root) {

  auto& thc = eri.corr_eri->get();
  auto mf = thc.MF();
  auto& mpi = thc.mpi();

  std::string err = std::string("lr_gw_sigma_DeltaW_calc - Incorrect input - ");
  auto prefix = io::get_value<std::string>(pt, "prefix", err + "prefix");

  std::string input_file = prefix + ".mbpt.h5";
  utils::check(std::filesystem::exists(input_file),
               "lr_gw_sigma_DeltaW_calc: Input checkpoint {} does not exist!", input_file);

  // Same div_treatment as the scGW run, so the Δε⁻¹ head stays consistent
  // with how W and Σ were computed
  auto input_grp = io::get_value_with_default<std::string>(pt, "input_type", "scf");
  auto div_treatment = read_div_treatment(*mpi, input_file, input_grp);

  // Read IAFT from checkpoint
  imag_axes_ft::IAFT ft(imag_axes_ft::read_iaft(input_file, false));

  long ns = mf->nspin();
  long nkpts = mf->nkpts();
  long nkpts_ibz = mf->nkpts_ibz();
  long nbnd = mf->nbnd();
  long NP = thc.Np();
  long nt = ft.nt_f();
  long nt_half_dw = (nt % 2 == 0) ? nt / 2 : nt / 2 + 1;

  auto sG_tskij = math::shm::make_shared_from_root_input<ComplexType, 5>(
      *mpi, G_tskij_root);

  auto sDeltaW_qtPQ_in = math::shm::make_shared_from_root_input<ComplexType, 4>(
      *mpi, std::array<long, 4>{nkpts, nt_half_dw, NP, NP},
      DeltaW_qtPQ_root);

  // Scatter into a distributed array in the τ-dist (t,q,P,Q) layout the Σ
  // evaluator consumes, transposing the (q,t) input on the fly.
  auto [tq_pgrid_dw, tq_bsize_dw] = utils::lr_W_q_local_dist(mpi->comm.size(), nt_half_dw, NP);
  using local_Array_4D_t = nda::array<ComplexType, 4>;
  auto dDeltaW_tqPQ = math::nda::make_distributed_array<local_Array_4D_t>(
      mpi->comm, tq_pgrid_dw,
      {nt_half_dw, nkpts, NP, NP}, tq_bsize_dw);
  {
    auto dw_src = sDeltaW_qtPQ_in.local();   // (q, t, P, Q), shared, full
    auto dw_loc = dDeltaW_tqPQ.local();      // (t, q, P, Q), this rank's block
    auto [t_org, q_org, P_org, Q_org] = dDeltaW_tqPQ.origin();
    auto [nt_loc, nq_loc, nP_loc, nQ_loc] = dDeltaW_tqPQ.local_shape();
    for (long it = 0; it < nt_loc; ++it)
      for (long iq = 0; iq < nq_loc; ++iq)
        for (long iP = 0; iP < nP_loc; ++iP)
          for (long iQ = 0; iQ < nQ_loc; ++iQ)
            dw_loc(it, iq, iP, iQ) = dw_src(iq+q_org, it+t_org, iP+P_org, iQ+Q_org);
    mpi->comm.barrier();
  }

  // Divergence correction flag (default true)
  auto div_corr = io::get_value_with_default<bool>(pt, "div_corr", true);

  // Read overlap matrix from checkpoint for divergence correction
  auto sS_skij = math::shm::make_shared_array<Array_view_4D_t>(
      *mpi, {ns, nkpts_ibz, nbnd, nbnd});
  if (div_corr) {
    chkpt::read_ovlp(mpi->node_comm, prefix, sS_skij);
  }

  // Compute ΔΣ = -G ⊙ ΔW using lr_gw::evaluate_sigma_DeltaW
  app_log(1, "\n  lr_gw_sigma_DeltaW_calc: Computing -G ⊙ ΔW...");
  solvers::lr_gw lr(&ft, q_pert, div_treatment);

  auto sDeltaSigma_tskij = math::shm::make_shared_array<Array_view_5D_t>(
      *mpi, {nt, ns, nkpts_ibz, nbnd, nbnd});

  // Precompute G^R(τ) and G^R(β−τ) used in evaluate_sigma_DeltaW.
  auto [dG_tsRPQ, dG_mtau_tsRPQ] = lr_precompute_G_R_pair(sG_tskij.local(), thc);

  // Divergence correction term 2 (q_pert=0 only): Δeps_inv_head from ΔW, applied to G.
  // The evaluator applies it when handed the overlap and the head (and skips it
  // off Γ itself); withholding them is how div_corr = false gets the bare
  // convolution for FD tests.
  auto S_loc = sS_skij.local();
  nda::array<ComplexType, 1> delta_eps_inv_head;
  if (div_corr && utils::is_q_gamma(q_pert)) {
    auto [delta_eps_inv_q, delta_head] =
        solvers::div_utils::eps_inv_head_t(dDeltaW_tqPQ, thc, *mf, &ft, div_treatment);
    delta_eps_inv_head = std::move(delta_head);
  }
  bool apply_div = div_corr && utils::is_q_gamma(q_pert);
  lr.evaluate_sigma_DeltaW(sDeltaSigma_tskij, sG_tskij.local(), dDeltaW_tqPQ, thc,
                           dG_tsRPQ, dG_mtau_tsRPQ,
                           apply_div ? &S_loc : nullptr,
                           apply_div ? &delta_eps_inv_head : nullptr);
  mpi->comm.barrier();

  // Rank-0-only return.
  nda::array<ComplexType, 5> out;
  if (mpi->comm.root()) {
    out = sDeltaSigma_tskij.local();
  }
  mpi->comm.barrier();
  return out;
}


/**
 * Compute eps_inv_head from screened interaction W_c(t,q,P,Q).
 *
 * Extracts the G=0,G'=0 component of (ε⁻¹ - 1) from W_c in the THC product
 * basis, and extrapolates to q→0. This matches the convention used by the
 * SCF loop and Sigma_div_correction (which stores/uses ε⁻¹-1, not ε⁻¹).
 *
 * @param W_c_tqPQ - [INPUT] W_c in (nt_half, nkpts, NP, NP) layout
 * @return eps_inv_head at q=0, shape (nt_half,)
 */
template<typename eri_t>
nda::array<ComplexType, 1> compute_eps_inv_head_calc(
    eri_t &eri, ptree const& pt,
    std::optional<nda::array<ComplexType, 4>> const& W_c_tqPQ_root) {

  auto& thc = eri.corr_eri->get();
  auto mf = thc.MF();
  auto& mpi = thc.mpi();

  std::string err = std::string("compute_eps_inv_head_calc - Incorrect input - ");
  auto prefix = io::get_value<std::string>(pt, "prefix", err + "prefix");

  std::string input_file = prefix + ".mbpt.h5";
  utils::check(std::filesystem::exists(input_file),
               "compute_eps_inv_head_calc: Input checkpoint {} does not exist!", input_file);

  // Same div_treatment as the scGW run (consistent head extrapolation)
  auto input_grp = io::get_value_with_default<std::string>(pt, "input_type", "scf");
  auto div_treatment = read_div_treatment(*mpi, input_file, input_grp);

  // Read IAFT from checkpoint
  imag_axes_ft::IAFT ft(imag_axes_ft::read_iaft(input_file, false));

  long nkpts_ibz = mf->nkpts_ibz();
  long NP = thc.Np();
  long nt = ft.nt_f();
  long nt_half_w = (nt % 2 == 0) ? nt / 2 : nt / 2 + 1;

  auto sW_c_tqPQ_in = math::shm::make_shared_from_root_input<ComplexType, 4>(
      *mpi, std::array<long, 4>{nt_half_w, nkpts_ibz, NP, NP}, W_c_tqPQ_root);

  // Scatter into distributed (t,q,P,Q) array.
  using local_Array_4D_t = nda::array<ComplexType, 4>;
  auto pgrid = compute_proc_grid_4D(mpi->comm.size(),
      {nt_half_w, nkpts_ibz, NP, NP}, {false, false, false, false});
  auto dW_tqPQ = math::nda::make_distributed_array<local_Array_4D_t>(
      mpi->comm, pgrid, {nt_half_w, nkpts_ibz, NP, NP});
  {
    auto wc_src = sW_c_tqPQ_in.local();
    auto wc_loc = dW_tqPQ.local();
    auto [o0, o1, o2, o3] = dW_tqPQ.origin();
    auto [n0, n1, n2, n3] = dW_tqPQ.local_shape();
    for (long i0 = 0; i0 < n0; ++i0)
      for (long i1 = 0; i1 < n1; ++i1)
        for (long i2 = 0; i2 < n2; ++i2)
          for (long i3 = 0; i3 < n3; ++i3)
            wc_loc(i0, i1, i2, i3) = wc_src(i0 + o0, i1 + o1, i2 + o2, i3 + o3);
    mpi->comm.barrier();
  }

  auto [eps_inv_head_q, eps_inv_head] =
      solvers::div_utils::eps_inv_head_t(dW_tqPQ, thc, *mf, &ft, div_treatment);
  mpi->comm.barrier();

  return eps_inv_head;
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
    std::optional<nda::array<ComplexType, 5>> const& G_tskij_root,
    std::optional<nda::array<ComplexType, 4>> const& W_c_qtPQ_root,
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

  // Same div_treatment as the scGW run (consistent divergence handling)
  auto input_grp = io::get_value_with_default<std::string>(pt, "input_type", "scf");
  auto div_treatment = read_div_treatment(*mpi, input_file, input_grp);

  // Read IAFT from checkpoint
  imag_axes_ft::IAFT ft(imag_axes_ft::read_iaft(input_file, false));

  long ns = mf->nspin();
  long nkpts = mf->nkpts();
  long nkpts_ibz = mf->nkpts_ibz();
  long nbnd = mf->nbnd();
  long NP = thc.Np();
  long nt = ft.nt_f();
  long nt_half_w = (nt % 2 == 0) ? nt / 2 : nt / 2 + 1;

  auto sG_tskij = math::shm::make_shared_from_root_input<ComplexType, 5>(
      *mpi, G_tskij_root);

  using local_Array_4D_t = nda::array<ComplexType, 4>;
  auto sW_c_qtPQ_in = math::shm::make_shared_from_root_input<ComplexType, 4>(
      *mpi, std::array<long, 4>{nkpts, nt_half_w, NP, NP}, W_c_qtPQ_root);

  // Scatter into a distributed (t,q,P,Q) array (q-axis undivided), transposing
  // out of the (q,t,P,Q) input as it goes -- eval_Sigma_all takes (t,q,P,Q).
  auto pgrid_w = compute_proc_grid_4D(mpi->comm.size(),
      {nt_half_w, nkpts, NP, NP}, {false, true, false, false});
  auto dW_tqPQ = math::nda::make_distributed_array<local_Array_4D_t>(
      mpi->comm, pgrid_w, {nt_half_w, nkpts, NP, NP});
  {
    auto wc_src = sW_c_qtPQ_in.local();   // (q, t, P, Q), shared, full
    auto wc_loc = dW_tqPQ.local();
    auto [o0, o1, o2, o3] = dW_tqPQ.origin();
    auto [n0, n1, n2, n3] = dW_tqPQ.local_shape();
    for (long i0 = 0; i0 < n0; ++i0)
      for (long i1 = 0; i1 < n1; ++i1)
        for (long i2 = 0; i2 < n2; ++i2)
          for (long i3 = 0; i3 < n3; ++i3)
            wc_loc(i0, i1, i2, i3) = wc_src(i1 + o1, i0 + o0, i2 + o2, i3 + o3);
    mpi->comm.barrier();
  }

  // Read overlap matrix from checkpoint for divergence correction
  auto sS_skij = math::shm::make_shared_array<Array_view_4D_t>(
      *mpi, {ns, nkpts_ibz, nbnd, nbnd});
  chkpt::read_ovlp(mpi->node_comm, prefix, sS_skij);

  // Use standard gw_t code path
  solvers::gw_t gw(&ft, div_treatment);

  auto sSigma_tskij = math::shm::make_shared_array<Array_view_5D_t>(
      *mpi, {nt, ns, nkpts_ibz, nbnd, nbnd});

  gw.eval_Sigma_all(sG_tskij.local(), dW_tqPQ, sSigma_tskij, thc, "R");
  if (div_corr) {
    gw.Sigma_div_correction(sSigma_tskij, sG_tskij.local(), sS_skij.local(), thc, eps_inv_head);
  }
  mpi->comm.barrier();

  // Rank-0-only return.
  nda::array<ComplexType, 5> out;
  if (mpi->comm.root()) {
    out = sSigma_tskij.local();
  }
  mpi->comm.barrier();
  return out;
}


// lr_gw_sigma_DeltaG_calc instantiations (THC only — lr_gw requires THC)
template nda::array<ComplexType, 5> lr_gw_sigma_DeltaG_calc(
    mb_eri_t<thc_reader_t, thc_reader_t, thc_reader_t, thc_reader_t>&,
    ptree const&,
    nda::array<double, 1> const&,
    std::optional<nda::array<ComplexType, 5>> const&);

// gw_evaluate_sigma_calc instantiations (THC only)
template nda::array<ComplexType, 5> gw_evaluate_sigma_calc(
    mb_eri_t<thc_reader_t, thc_reader_t, thc_reader_t, thc_reader_t>&,
    ptree const&,
    std::optional<nda::array<ComplexType, 5>> const&,
    bool);

// lr_DeltaH0_from_thc_aux_calc instantiations (THC only — the aux basis is THC's)
template nda::array<ComplexType, 5> lr_DeltaH0_from_thc_aux_calc(
    mb_eri_t<thc_reader_t, thc_reader_t, thc_reader_t, thc_reader_t>&,
    nda::array<double, 1> const&,
    std::optional<nda::array<ComplexType, 2>> const&);

// lr_gw_Pi_calc instantiations (THC only — lr_gw requires THC)
template nda::array<ComplexType, 4> lr_gw_Pi_calc(
    mb_eri_t<thc_reader_t, thc_reader_t, thc_reader_t, thc_reader_t>&,
    nda::array<double, 1> const&,
    std::optional<nda::array<ComplexType, 5>> const&,
    std::optional<nda::array<ComplexType, 5>> const&,
    std::optional<nda::array<ComplexType, 4>> const&,
    std::optional<nda::array<ComplexType, 4>> const&);

// gw_evaluate_Pi_calc instantiations (THC only)
template nda::array<ComplexType, 4> gw_evaluate_Pi_calc(
    mb_eri_t<thc_reader_t, thc_reader_t, thc_reader_t, thc_reader_t>&,
    ptree const&,
    std::optional<nda::array<ComplexType, 5>> const&);

// lr_gw_W_calc instantiations (THC only)
template nda::array<ComplexType, 4> lr_gw_W_calc(
    mb_eri_t<thc_reader_t, thc_reader_t, thc_reader_t, thc_reader_t>&,
    ptree const&,
    nda::array<double, 1> const&,
    std::optional<nda::array<ComplexType, 4>> const&);

// gw_evaluate_W_from_Pi_calc instantiations (THC only)
template nda::array<ComplexType, 4> gw_evaluate_W_from_Pi_calc(
    mb_eri_t<thc_reader_t, thc_reader_t, thc_reader_t, thc_reader_t>&,
    ptree const&,
    std::optional<nda::array<ComplexType, 4>> const&);

// lr_gw_sigma_DeltaW_calc instantiations (THC only)
template nda::array<ComplexType, 5> lr_gw_sigma_DeltaW_calc(
    mb_eri_t<thc_reader_t, thc_reader_t, thc_reader_t, thc_reader_t>&,
    ptree const&,
    nda::array<double, 1> const&,
    std::optional<nda::array<ComplexType, 5>> const&,
    std::optional<nda::array<ComplexType, 4>> const&);

// compute_eps_inv_head_calc instantiations (THC only)
template nda::array<ComplexType, 1> compute_eps_inv_head_calc(
    mb_eri_t<thc_reader_t, thc_reader_t, thc_reader_t, thc_reader_t>&,
    ptree const&,
    std::optional<nda::array<ComplexType, 4>> const&);

// gw_evaluate_sigma_with_W_calc instantiations (THC only)
template nda::array<ComplexType, 5> gw_evaluate_sigma_with_W_calc(
    mb_eri_t<thc_reader_t, thc_reader_t, thc_reader_t, thc_reader_t>&,
    ptree const&,
    std::optional<nda::array<ComplexType, 5>> const&,
    std::optional<nda::array<ComplexType, 4>> const&,
    nda::array<ComplexType, 1> const&,
    bool);



}
