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


#include <cmath>

#include "methods/SCF/lr_driver.hpp"
#include "methods/SCF/lr_precompute.hpp"
#include "methods/ERI/thc_reader_t.hpp"
#include "numerics/distributed_array/nda.hpp"
#include "utilities/check.hpp"
#include "utilities/lr_utils.hpp"
#include "utilities/proc_meminfo.hpp"
#include "nda/nda.hpp"

using thc_reader_t = methods::thc_reader_t;

namespace methods {

lr_driver::lr_driver(simple_dyson& dyson, nda::array<double, 1> const& q_vec)
    : _dyson(dyson),
      _mpi(dyson.mpi()),
      _MF(dyson.MF()),
      _lr_dyson(dyson, q_vec),
      _nts(dyson.FT()->nt_f()),
      _ns(_MF->nspin()),
      _nkpts(_MF->nkpts()),
      _nkpts_ibz(_MF->nkpts_ibz()),
      _nbnd(_MF->nbnd()),
      _Timer() {

  app_log(1, "\n"
             "╔═╗╔═╗╔═╗ ╦ ╦╦  ┬  ┬─┐  ┌─┐┌─┐┌─┐\n"
             "║  ║ ║║═╬╗║ ║║  │  ├┬┘──└─┐│  ├┤ \n"
             "╚═╝╚═╝╚═╝╚╚═╝╩  ┴─┘┴└─  └─┘└─┘└  \n");
  app_log(1, "  Linear Response SCF Driver");
  app_log(1, "  q-vector: ({:.6f}, {:.6f}, {:.6f})",
          q_vec(0), q_vec(1), q_vec(2));
  app_log(1, "  q is Gamma point: {}\n", _lr_dyson.is_q_gamma() ? "yes" : "no");

  for (auto& v : {"LR_SCF", "LR_DRIVER_SETUP",
                  "LR_DRIVER_SETUP_W_FULL", "LR_DRIVER_SETUP_W_TRPQ", "LR_DRIVER_SETUP_G_OMEGA", "LR_DRIVER_SETUP_G_R",
                  "LR_DRIVER_SETUP_DN_DMU", "LR_DRIVER_SETUP_ALLOC", "LR_DRIVER_SETUP_IBC", "LR_DRIVER_SETUP_MISC",
                  "LR_DYSON", "LR_HF", "LR_GW_SIGMA", "LR_GW_DW_TRANSPOSE",
                   "LR_GW_PI", "LR_GW_W", "LR_GW_SIGMA_TERM2",
                   "LR_ITER_ALG"}) {
    _Timer.add(v);
  }
  _mpi->comm.barrier();
}


template<THC_ERI THC_t, typename dW_t>
std::tuple<int, double> lr_driver::run_lr(
    sArray_t<Array_view_5D_t>& sDeltaG_tskij,
    sArray_t<Array_view_4D_t>& sDeltaDm_skij,
    sArray_t<Array_view_4D_t>& sDeltaF_skij,
    sArray_t<Array_view_5D_t>* sDeltaSigma_tskij,
    const sArray_t<Array_view_5D_t>& sG_tskij,
    const sArray_t<Array_view_4D_t>& sDeltaH0_skij,
    THC_t& thc,
    bool include_hartree, bool include_exchange, lr_gw_update_mode gw_mode,
    dW_t* dW_qtPQ, const nda::array<ComplexType, 1>* eps_inv_head,
    int max_iter, double tol, bool fix_density,
    const lr_iter_params& iter_params,
    const sArray_t<Array_view_4D_t>* sDeltaX_left,
    const sArray_t<Array_view_4D_t>* sDeltaX_right,
    const nda::array<ComplexType, 4>* Dm_ab,
    bool div_corr,
    const nda::array_view<ComplexType, 3>* DeltaV_qPQ) {
  (void) thc; (void) dW_qtPQ; (void) eps_inv_head; (void) Dm_ab; (void) div_corr;

  _Timer.start("LR_SCF");

  bool need_hf = include_hartree || include_exchange;
  bool include_gw_sigma = (gw_mode != lr_gw_update_mode::none);

  // Not yet ported: filled in by the LR-Hartree / LR-HF / LR-GW / IBC steps.
  utils::check(!need_hf, "lr_driver::run_lr: LR-HF (Hartree/exchange) not yet ported.");
  utils::check(!include_gw_sigma, "lr_driver::run_lr: LR-GW self-energy not yet ported.");
  utils::check(!sDeltaX_left && !sDeltaX_right,
               "lr_driver::run_lr: DeltaX IBC correction not yet ported.");
  utils::check(!DeltaV_qPQ, "lr_driver::run_lr: DeltaV correction not yet ported.");

  utils::check(iter_params.alg == "damping" || iter_params.alg == "DIIS",
               "lr_driver::run_lr: unknown iter_alg '{}'. Must be 'damping' or 'DIIS'.",
               iter_params.alg);

  bool use_diis = (iter_params.alg == "DIIS");
  double mixing = iter_params.mixing;

  app_log(1, "Starting Linear Response SCF loop:");
  app_log(1, "  max_iter = {}", max_iter);
  app_log(1, "  tol = {:.2e}", tol);
  app_log(1, "  fix_density = {}", fix_density ? "true" : "false");
  app_log(1, "  include_hartree = {}", include_hartree ? "true" : "false");
  app_log(1, "  include_exchange = {}", include_exchange ? "true" : "false");
  app_log(1, "  gw_mode = none");
  app_log(1, "  iter_alg = {}", iter_params.alg);
  app_log(1, "  mixing = {:.2f}", mixing);
  if (use_diis) {
    app_log(1, "  max_subsp_size = {}", iter_params.max_subsp_size);
    app_log(1, "  diis_warmup = {}", iter_params.diis_warmup);
  }

  _Timer.start("LR_DRIVER_SETUP");

  // Precompute G(iω) in shared memory and pass to lr_dyson (avoids redundant FT per iteration)
  utils::memlog("lr_driver::run_lr: before sG_wskij precompute");
  _Timer.start("LR_DRIVER_SETUP_G_OMEGA");
  auto sG_wskij = lr_precompute_G_omega(*_mpi, sG_tskij, *_dyson.FT());
  _lr_dyson.set_cached_G_omega(&sG_wskij);
  _Timer.stop("LR_DRIVER_SETUP_G_OMEGA");
  utils::memlog("lr_driver::run_lr: after sG_wskij precompute");

  // Precompute dN/dμ if needed for fix_density mode at q=0
  if (fix_density && _lr_dyson.is_q_gamma()) {
    _Timer.start("LR_DRIVER_SETUP_DN_DMU");
    _lr_dyson.compute_dN_dmu();
    _Timer.stop("LR_DRIVER_SETUP_DN_DMU");
  }

  _Timer.start("LR_DRIVER_SETUP_MISC");
  if (use_diis) {
    _lr_diis = std::make_unique<lr_diis>(
        iter_params.max_subsp_size, iter_params.diis_warmup, mixing);
  }
  _Timer.stop("LR_DRIVER_SETUP_MISC");

  _Timer.start("LR_DRIVER_SETUP_ALLOC");
  // Previous density matrix (for convergence check)
  auto sDeltaDm_prev_skij = math::shm::make_shared_array<Array_view_4D_t>(
      *_mpi, {_ns, _nkpts_ibz, _nbnd, _nbnd});
  _Timer.stop("LR_DRIVER_SETUP_ALLOC");

  _Timer.start("LR_DRIVER_SETUP_MISC");
  // ΔF stays zero on the Dyson-only path
  if (_mpi->node_comm.root()) {
    sDeltaF_skij.local() = ComplexType(0.0);
    if (sDeltaSigma_tskij) sDeltaSigma_tskij->local() = ComplexType(0.0);
  }
  _mpi->comm.barrier();
  _Timer.stop("LR_DRIVER_SETUP_MISC");
  _Timer.stop("LR_DRIVER_SETUP");
  utils::memlog("lr_driver::run_lr: end of LR_DRIVER_SETUP");
  print_setup_timers();

  double Delta_mu = 0.0;
  int iter = 0;
  bool converged = false;

  app_log(1, "\n  iter    ||ΔDm||         ||ΔDm-ΔDm_prev||  Δμ");
  app_log(1, "  " + std::string(60, '-'));

  for (iter = 1; iter <= max_iter; ++iter) {

    _Timer.start("LR_SAVE");
    if (_mpi->node_comm.root() && iter > 1) {
      sDeltaDm_prev_skij.local() = sDeltaDm_skij.local();
    }
    _mpi->comm.barrier();
    _Timer.stop("LR_SAVE");

    // ΔG = G_{k+q} · [ΔH0 + ΔF + ΔΣ - Δμ·S] · G_k
    _Timer.start("LR_DYSON");
    Delta_mu = _lr_dyson.solve_lr_dyson(
        sDeltaG_tskij, sDeltaDm_skij, sDeltaH0_skij,
        sDeltaF_skij, sDeltaSigma_tskij,
        fix_density, Delta_mu);
    _Timer.stop("LR_DYSON");
    _mpi->comm.barrier();

    // Convergence norms. lr_distributed_norm stripes the elements over
    // node_comm ranks; the shared array is node-replicated, so the trailing
    // broadcast from global rank 0 preserves exact global agreement.
    _Timer.start("LR_CONVERGENCE");
    auto norms_Dm = utils::lr_distributed_norm(
        _mpi->node_comm, sDeltaDm_skij.local(), sDeltaDm_prev_skij.local(), iter > 1);
    double norm_DeltaDm = norms_Dm.first;
    double norm_DeltaDm_diff = norms_Dm.second;
    _mpi->comm.broadcast_n(&norm_DeltaDm, 1, 0);
    _mpi->comm.broadcast_n(&norm_DeltaDm_diff, 1, 0);

    if (iter == 1) {
      app_log(1, "  {:4d}    {:.6e}     {:13s}     {:.3e}",
              iter, norm_DeltaDm, "---", Delta_mu);
    } else {
      app_log(1, "  {:4d}    {:.6e}     {:.6e}      {:.3e}",
              iter, norm_DeltaDm, norm_DeltaDm_diff, Delta_mu);
    }
    _Timer.stop("LR_CONVERGENCE");

    // Without ΔF/ΔΣ feedback the Dyson solve is exact after one pass
    if (iter > 1 && norm_DeltaDm_diff < tol) {
      converged = true;
      break;
    }
    if (max_iter == 1) break;

    _mpi->comm.barrier();
  }

  _Timer.stop("LR_SCF");

  if (converged) {
    app_log(1, "\n  LR SCF converged in {} iterations!", iter);
  } else if (max_iter > 1) {
    app_log(1, "\n  [WARNING] LR SCF did NOT converge after {} iterations.", max_iter);
  }
  app_log(1, "  Final Δμ = {:.6e}", Delta_mu);

  print_timers();

  return std::make_tuple(iter, Delta_mu);
}


void lr_driver::print_setup_timers() {
  app_log(2, "\n  LR_DRIVER_SETUP timers");
  app_log(2, "  -----------------------");
  app_log(2, "    LR Driver Setup:            {0:8.3f} sec  {1:4d} calls", _Timer.elapsed("LR_DRIVER_SETUP"), _Timer.number_of_calls("LR_DRIVER_SETUP"));
  app_log(2, "      - W_full(iω):             {0:8.3f} sec  {1:4d} calls", _Timer.elapsed("LR_DRIVER_SETUP_W_FULL"), _Timer.number_of_calls("LR_DRIVER_SETUP_W_FULL"));
  app_log(2, "      - W_tRPQ:                 {0:8.3f} sec  {1:4d} calls", _Timer.elapsed("LR_DRIVER_SETUP_W_TRPQ"), _Timer.number_of_calls("LR_DRIVER_SETUP_W_TRPQ"));
  app_log(2, "      - G(iω) precompute:       {0:8.3f} sec  {1:4d} calls", _Timer.elapsed("LR_DRIVER_SETUP_G_OMEGA"), _Timer.number_of_calls("LR_DRIVER_SETUP_G_OMEGA"));
  app_log(2, "      - G^R pair precompute:    {0:8.3f} sec  {1:4d} calls", _Timer.elapsed("LR_DRIVER_SETUP_G_R"), _Timer.number_of_calls("LR_DRIVER_SETUP_G_R"));
  app_log(2, "      - dN/dμ precompute:       {0:8.3f} sec  {1:4d} calls", _Timer.elapsed("LR_DRIVER_SETUP_DN_DMU"), _Timer.number_of_calls("LR_DRIVER_SETUP_DN_DMU"));
  app_log(2, "      - Alloc:                  {0:8.3f} sec  {1:4d} calls", _Timer.elapsed("LR_DRIVER_SETUP_ALLOC"), _Timer.number_of_calls("LR_DRIVER_SETUP_ALLOC"));
  app_log(2, "      - Build IBC (DeltaX):     {0:8.3f} sec  {1:4d} calls", _Timer.elapsed("LR_DRIVER_SETUP_IBC"), _Timer.number_of_calls("LR_DRIVER_SETUP_IBC"));
  app_log(2, "      - Misc:                   {0:8.3f} sec  {1:4d} calls\n", _Timer.elapsed("LR_DRIVER_SETUP_MISC"), _Timer.number_of_calls("LR_DRIVER_SETUP_MISC"));
}


void lr_driver::print_timers() {
  const std::string sub_indent = "        ";
  app_log(2, "\n  LR_DRIVER timers");
  app_log(2, "  -----------------");
  app_log(2, "    Total LR SCF:               {0:8.3f} sec  {1:4d} calls", _Timer.elapsed("LR_SCF"), _Timer.number_of_calls("LR_SCF"));
  app_log(2, "      - LR Driver Setup:        {0:8.3f} sec  {1:4d} calls", _Timer.elapsed("LR_DRIVER_SETUP"), _Timer.number_of_calls("LR_DRIVER_SETUP"));
  app_log(2, "      - LR Dyson (total):       {0:8.3f} sec  {1:4d} calls", _Timer.elapsed("LR_DYSON"), _Timer.number_of_calls("LR_DYSON"));
  _lr_dyson.print_subclocks(2, sub_indent);
  app_log(2, "      - LR Save (prev arrays):  {0:8.3f} sec  {1:4d} calls", _Timer.elapsed("LR_SAVE"), _Timer.number_of_calls("LR_SAVE"));
  app_log(2, "      - LR Convergence (norms): {0:8.3f} sec  {1:4d} calls\n", _Timer.elapsed("LR_CONVERGENCE"), _Timer.number_of_calls("LR_CONVERGENCE"));
}


// Template instantiations
// dW type: distributed_array<nda::array<ComplexType, 4>, mpi3::communicator>
using dW_concrete_t = memory::darray_t<nda::array<ComplexType, 4>, mpi3::communicator>;

template std::tuple<int, double> lr_driver::run_lr(
    sArray_t<Array_view_5D_t>&,
    sArray_t<Array_view_4D_t>&,
    sArray_t<Array_view_4D_t>&,
    sArray_t<Array_view_5D_t>*,
    const sArray_t<Array_view_5D_t>&,
    const sArray_t<Array_view_4D_t>&,
    thc_reader_t&,
    bool, bool, lr_gw_update_mode,
    dW_concrete_t*, const nda::array<ComplexType, 1>*,
    int, double, bool, const lr_iter_params&,
    const sArray_t<Array_view_4D_t>*, const sArray_t<Array_view_4D_t>*,
    const nda::array<ComplexType, 4>*, bool,
    const nda::array_view<ComplexType, 3>*);

} // namespace methods
