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


#include <cmath>

#include "methods/SCF/lr_driver.hpp"
#include "methods/SCF/lr_precompute.hpp"
#include "methods/ERI/thc_reader_t.hpp"
#include "numerics/distributed_array/nda.hpp"
#include "utilities/check.hpp"
#include "nda/nda.hpp"

using thc_reader_t = methods::thc_reader_t;

namespace methods {

lr_driver::lr_driver(simple_dyson& dyson, nda::array<double, 1> const& q_vec)
    : _dyson(dyson),
      _mpi(dyson.mpi()),
      _MF(dyson.MF()),
      _lr_dyson(dyson, q_vec),
      _lr_hf(nullptr),
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

  for (auto& v : {"LR_SCF", "LR_SETUP", "LR_DYSON", "LR_HF", "LR_GW_SIGMA",
                   "LR_GW_PI", "LR_GW_W", "LR_GW_SIGMA_TERM2",
                   "LR_ITER_ALG"}) {
    _Timer.add(v);
  }
  _mpi->comm.barrier();
}


// Distribution flow through the LR-GW pipeline
// Three distribution patterns:
//   τ-dist (q-local):  pgrid = {tpools, 1, np_P, np_Q}  — q undivided
//   τ-local:           pgrid = {1, nqpools, np_P, np_Q}  — tau/omega undivided (FT buffer)
//   PQ-local:          pgrid = {nwpools, nqpools, 1, 1}  — PQ undivided (matrix multiply)
//
//   W_c loaded:          (q,t,P,Q), τ-dist
//   compute_W_full_omega: τ-dist → FT(τ→ω) → PQ-local, + Z(q) → W_full(iω) [cached]
//
//   evaluate_lr_Pi:      → (t,q,P,Q), τ-dist
//   lr_dyson_W_from_Pi_tau (in-place):
//     tau_to_w:          τ-dist → PQ-local (via q-distributed FT buffer)
//     lr_dyson_W_in_place: PQ-local (direct BLAS gemm, no SLATE)
//     w_to_tau:          PQ-local → τ-dist (via q-distributed FT buffer)
//     output:            τ-dist (overwrites input)
//   transpose (t,q)→(q,t): τ-dist
//   evaluate_sigma_*:    τ-dist in (q,t) order

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
    const lr_iter_params& iter_params) {

  _Timer.start("LR_SCF");

  bool need_hf = include_hartree || include_exchange;
  bool include_gw_sigma = (gw_mode != lr_gw_update_mode::none);
  bool gw_full = (gw_mode == lr_gw_update_mode::full);

  utils::check(iter_params.alg == "damping" || iter_params.alg == "DIIS",
               "lr_driver::run_lr: unknown iter_alg '{}'. Must be 'damping' or 'DIIS'.",
               iter_params.alg);
  if (include_gw_sigma) {
    utils::check(sDeltaSigma_tskij != nullptr,
                 "lr_driver::run_lr: gw_mode != none but sDeltaSigma_tskij is null.");
    // eps_inv_head: not yet used, will be needed for div_corr (TODO)
    utils::check(dW_qtPQ != nullptr && eps_inv_head != nullptr,
                 "lr_driver::run_lr: gw_mode != none but dW_qtPQ or eps_inv_head is null.");
  }

  const char* gw_mode_str = "none";
  switch (gw_mode) {
    case lr_gw_update_mode::none:    gw_mode_str = "none"; break;
    case lr_gw_update_mode::fixed_W: gw_mode_str = "fixed_W"; break;
    case lr_gw_update_mode::full:    gw_mode_str = "full"; break;
  }

  bool use_diis = (iter_params.alg == "DIIS");
  double mixing = iter_params.mixing;

  app_log(1, "Starting Linear Response SCF loop:");
  app_log(1, "  max_iter = {}", max_iter);
  app_log(1, "  tol = {:.2e}", tol);
  app_log(1, "  fix_density = {}", fix_density ? "true" : "false");
  app_log(1, "  include_hartree = {}", include_hartree ? "true" : "false");
  app_log(1, "  include_exchange = {}", include_exchange ? "true" : "false");
  app_log(1, "  gw_mode = {}", gw_mode_str);
  app_log(1, "  iter_alg = {}", iter_params.alg);
  app_log(1, "  mixing = {:.2f}", mixing);
  if (use_diis) {
    app_log(1, "  max_subsp_size = {}", iter_params.max_subsp_size);
    app_log(1, "  diis_warmup = {}", iter_params.diis_warmup);
  }

  // Initialize lr_hf solver if needed
  if (need_hf && !_lr_hf) {
    _lr_hf = std::make_unique<solvers::lr_hf>(_mpi, _MF, _lr_dyson.q_vec());
  }

  // Create lr_gw solver if needed (local to this call, no need to store as member)
  std::unique_ptr<solvers::lr_gw> lr_gw_solver;
  if (include_gw_sigma) {
    lr_gw_solver = std::make_unique<solvers::lr_gw>(_dyson.FT(), _lr_dyson.q_vec());
  }

  // Create lr_rpa_pi and lr_scr_coulomb_t solvers for full mode
  std::unique_ptr<solvers::lr_rpa_pi> lr_pi_solver;
  std::unique_ptr<solvers::lr_scr_coulomb_t> lr_scr_solver;
  if (gw_full) {
    lr_pi_solver = std::make_unique<solvers::lr_rpa_pi>(_lr_dyson.q_vec());
    lr_scr_solver = std::make_unique<solvers::lr_scr_coulomb_t>(_dyson.FT());
  }
  // Precompute W_full(iω) = W_c(iω) + V (cached across iterations).
  _Timer.start("LR_SETUP");
  using local_Array_4D_t = nda::array<ComplexType, 4>;
  std::optional<memory::darray_t<local_Array_4D_t, mpi3::communicator>> opt_dW_full_wqPQ;
  if (gw_full) {
    // Transpose dW_qtPQ from (q,t,P,Q) → (t,q,P,Q), then compute W_full(iω)
    auto dW_tqPQ = utils::transpose_axes_01(*dW_qtPQ, _mpi->comm);
    opt_dW_full_wqPQ.emplace(
        lr_scr_solver->compute_W_full_omega(dW_tqPQ, thc));
    // dW_tqPQ is consumed (reset) by compute_W_full_omega
  }

  // Precompute W in R-space: transpose (q,t)→(t,R), with q→R FT.
  // Result: dW_tRPQ with (t,R,P,Q) layout, pgrid (tpools,1,np_P,np_Q).
  std::optional<dW_t> opt_dW_tRPQ;
  if (include_gw_sigma) {
    opt_dW_tRPQ.emplace(lr_precompute_W_tRPQ(*dW_qtPQ, thc));
    dW_qtPQ->reset();  // free (q,t,P,Q) memory; no longer needed
  }

  // Precompute G(iω) in shared memory and pass to lr_dyson (avoids redundant FT per iteration)
  auto sG_wskij = lr_precompute_G_omega(*_mpi, sG_tskij, *_dyson.FT());
  _lr_dyson.set_cached_G_omega(&sG_wskij);

  // Precompute dN/dμ if needed for fix_density mode at q=0
  if (fix_density && _lr_dyson.is_q_gamma()) {
    _lr_dyson.compute_dN_dmu();
  }

  // Initialize DIIS if requested
  if (use_diis) {
    _lr_diis = std::make_unique<lr_diis>(
        iter_params.max_subsp_size, iter_params.diis_warmup, mixing);
  }

  // Get overlap matrix from dyson
  auto& sS_skij = _dyson.sS_skij();

  // Allocate array for previous density matrix (for convergence check)
  auto sDeltaDm_prev_skij = math::shm::make_shared_array<Array_view_4D_t>(
      *_mpi, {_ns, _nkpts_ibz, _nbnd, _nbnd});

  // Allocate arrays for previous Fock matrix (for damping/DIIS)
  auto sDeltaF_prev_skij = math::shm::make_shared_array<Array_view_4D_t>(
      *_mpi, {_ns, _nkpts_ibz, _nbnd, _nbnd});
  // Previous ΔΣ only needed when GW is active (saves a full 5D allocation)
  auto sDeltaSigma_prev_tskij = include_gw_sigma
      ? math::shm::make_shared_array<Array_view_5D_t>(*_mpi, {_nts, _ns, _nkpts_ibz, _nbnd, _nbnd})
      : math::shm::make_shared_array<Array_view_5D_t>(*_mpi, {1, 1, 1, 1, 1});

  // Term 2 buffer for full mode: ΔΣ_term2 (shared array for accumulation)
  auto sDeltaSigma_term2_tskij = gw_full
      ? math::shm::make_shared_array<Array_view_5D_t>(*_mpi, {_nts, _ns, _nkpts_ibz, _nbnd, _nbnd})
      : math::shm::make_shared_array<Array_view_5D_t>(*_mpi, {1, 1, 1, 1, 1});

  // Initialize ΔF = 0 (and ΔΣ = 0 if GW active)
  if (_mpi->node_comm.root()) {
    sDeltaF_skij.local() = ComplexType(0.0);
    if (sDeltaSigma_tskij) sDeltaSigma_tskij->local() = ComplexType(0.0);
  }
  _mpi->comm.barrier();
  _Timer.stop("LR_SETUP");

  double Delta_mu = 0.0;
  int iter = 0;
  bool converged = false;

  // SCF iteration header
  if (include_gw_sigma) {
    app_log(1, "\n  iter    ||ΔDm||         ||ΔDm-ΔDm_prev||  ||ΔF||          ||ΔF-ΔF_prev||   ||ΔΣ||          ||ΔΣ-ΔΣ_prev||   Δμ");
    app_log(1, "  " + std::string(120, '-'));
  } else {
    app_log(1, "\n  iter    ||ΔDm||         ||ΔDm-ΔDm_prev||  ||ΔF||          ||ΔF-ΔF_prev||   Δμ");
    app_log(1, "  " + std::string(92, '-'));
  }

  for (iter = 1; iter <= max_iter; ++iter) {

    // Save previous density matrix and Fock matrix
    if (_mpi->node_comm.root() && iter > 1) {
      sDeltaDm_prev_skij.local() = sDeltaDm_skij.local();
      sDeltaF_prev_skij.local() = sDeltaF_skij.local();
      if (include_gw_sigma) {
        sDeltaSigma_prev_tskij.local() = sDeltaSigma_tskij->local();
      }
    }
    _mpi->comm.barrier();

    // Step 1: Solve LR Dyson equation
    // ΔG = G_{k+q} · [ΔH0 + ΔF + ΔΣ - Δμ·S] · G_k
    _Timer.start("LR_DYSON");
    Delta_mu = _lr_dyson.solve_lr_dyson(
        sDeltaG_tskij, sDeltaDm_skij, sDeltaH0_skij,
        sDeltaF_skij, sDeltaSigma_tskij,
        fix_density, Delta_mu);
    _Timer.stop("LR_DYSON");
    _mpi->comm.barrier();

    // Compute norms for logging
    double norm_DeltaDm = 0.0;
    double norm_DeltaDm_diff = 0.0;
    if (_mpi->node_comm.root()) {
      for (int is = 0; is < _ns; ++is) {
        for (int ik = 0; ik < _nkpts_ibz; ++ik) {
          auto DeltaDm_k = sDeltaDm_skij.local()(is, ik, nda::range::all, nda::range::all);
          norm_DeltaDm += std::pow(nda::frobenius_norm(DeltaDm_k), 2);

          if (iter > 1) {
            auto DeltaDm_prev_k = sDeltaDm_prev_skij.local()(is, ik, nda::range::all, nda::range::all);
            nda::matrix<ComplexType> diff = DeltaDm_k - DeltaDm_prev_k;
            norm_DeltaDm_diff += std::pow(nda::frobenius_norm(diff), 2);
          }
        }
      }
      norm_DeltaDm = std::sqrt(norm_DeltaDm);
      norm_DeltaDm_diff = std::sqrt(norm_DeltaDm_diff);
    }
    _mpi->comm.broadcast_n(&norm_DeltaDm, 1, 0);
    _mpi->comm.broadcast_n(&norm_DeltaDm_diff, 1, 0);

    // Step 2: Compute LR Fock matrix (if requested)
    if (need_hf) {
      _Timer.start("LR_HF");
      _lr_hf->evaluate(sDeltaF_skij, sDeltaDm_skij, thc, sS_skij.local(),
                       include_hartree, include_exchange);
      _Timer.stop("LR_HF");
      _mpi->comm.barrier();
    }

    // Step 3: Compute LR GW self-energy term 1: ΔΣ = -ΔG ⊙ W_c
    if (include_gw_sigma) {
      _Timer.start("LR_GW_SIGMA");
      lr_gw_solver->evaluate_sigma_DeltaG(
          *sDeltaSigma_tskij, sDeltaG_tskij.local(), *opt_dW_tRPQ, thc);
      _Timer.stop("LR_GW_SIGMA");
      _mpi->comm.barrier();
    }

    // Step 3b-3g: Full mode — compute ΔP, ΔW, and ΔΣ term 2 = -G ⊙ ΔW
    if (gw_full) {
      // Step 3b: ΔP = -ΔG·G - G·ΔG
      _Timer.start("LR_GW_PI");
      auto dDeltaPi_tqPQ = lr_pi_solver->evaluate_lr_Pi(
          sG_tskij.local(), sDeltaG_tskij.local(), thc);
      _mpi->comm.barrier();
      _Timer.stop("LR_GW_PI");

      // Step 3c-3d: ΔW_c(τ) via lr_dyson_W_from_Pi_tau (in-place, uses cached W_full)
      _Timer.start("LR_GW_W");
      lr_scr_solver->lr_dyson_W_from_Pi_tau(dDeltaPi_tqPQ, *opt_dW_full_wqPQ, thc);
      // dDeltaPi_tqPQ now contains ΔW_c(τ) in q-local distribution
      auto& dDeltaW_tqPQ = dDeltaPi_tqPQ;  // alias for clarity
      _mpi->comm.barrier();
      _Timer.stop("LR_GW_W");

      // Step 3e: Transpose ΔW from (t,q,P,Q) → (q,t,P,Q) for lr_gw
      _Timer.start("LR_GW_SIGMA_TERM2");
      auto dDeltaW_qtPQ = utils::transpose_axes_01(dDeltaW_tqPQ, _mpi->comm);
      dDeltaW_tqPQ.reset();

      // Step 3f: ΔΣ_term2 = -G ⊙ ΔW
      lr_gw_solver->evaluate_sigma_DeltaW(
          sDeltaSigma_term2_tskij, sG_tskij.local(), dDeltaW_qtPQ, thc);
      _mpi->comm.barrier();

      // Step 3g: ΔΣ += ΔΣ_term2 (accumulate on shared memory, root only)
      if (_mpi->node_comm.root()) {
        sDeltaSigma_tskij->local() += sDeltaSigma_term2_tskij.local();
      }
      _mpi->comm.barrier();
      _Timer.stop("LR_GW_SIGMA_TERM2");
    }

    // Step 4: Apply iteration algorithm (DIIS or damping) on combined (ΔF, ΔΣ)
    _Timer.start("LR_ITER_ALG");
    if (iter > 1 && (need_hf || include_gw_sigma)) {
      if (_mpi->node_comm.root()) {
        if (use_diis) {
          nda::array<ComplexType, 4> DeltaF_new{sDeltaF_skij.local()};
          nda::array<ComplexType, 4> DeltaF_prev{sDeltaF_prev_skij.local()};
          nda::array<ComplexType, 5> DeltaSigma_new;
          nda::array<ComplexType, 5> DeltaSigma_prev;
          if (include_gw_sigma) {
            DeltaSigma_new = nda::array<ComplexType, 5>{sDeltaSigma_tskij->local()};
            DeltaSigma_prev = nda::array<ComplexType, 5>{sDeltaSigma_prev_tskij.local()};
          }
          auto [F_mixed, Sigma_mixed] = _lr_diis->next_step_combined(
              DeltaF_new, DeltaF_prev, DeltaSigma_new, DeltaSigma_prev, iter);
          sDeltaF_skij.local() = F_mixed;
          if (include_gw_sigma && Sigma_mixed.size() > 0) {
            sDeltaSigma_tskij->local() = Sigma_mixed;
          }
        } else if (mixing < 1.0) {
          sDeltaF_skij.local() = mixing * sDeltaF_skij.local()
                                 + (1.0 - mixing) * sDeltaF_prev_skij.local();
          if (include_gw_sigma) {
            sDeltaSigma_tskij->local() = mixing * sDeltaSigma_tskij->local()
                                         + (1.0 - mixing) * sDeltaSigma_prev_tskij.local();
          }
        }
      }
      _mpi->comm.barrier();
    }
    _Timer.stop("LR_ITER_ALG");

    // Compute norms of ΔF and ΔF-ΔF_prev for logging
    double norm_DeltaF = 0.0;
    double norm_DeltaF_diff = 0.0;
    if (_mpi->node_comm.root()) {
      for (int is = 0; is < _ns; ++is) {
        for (int ik = 0; ik < _nkpts_ibz; ++ik) {
          auto DeltaF_k = sDeltaF_skij.local()(is, ik, nda::range::all, nda::range::all);
          norm_DeltaF += std::pow(nda::frobenius_norm(DeltaF_k), 2);

          if (iter > 1) {
            auto DeltaF_prev_k = sDeltaF_prev_skij.local()(is, ik, nda::range::all, nda::range::all);
            nda::matrix<ComplexType> diff_F = DeltaF_k - DeltaF_prev_k;
            norm_DeltaF_diff += std::pow(nda::frobenius_norm(diff_F), 2);
          }
        }
      }
      norm_DeltaF = std::sqrt(norm_DeltaF);
      norm_DeltaF_diff = std::sqrt(norm_DeltaF_diff);
    }
    _mpi->comm.broadcast_n(&norm_DeltaF, 1, 0);
    _mpi->comm.broadcast_n(&norm_DeltaF_diff, 1, 0);

    // Compute norms of ΔΣ and ΔΣ-ΔΣ_prev for logging
    double norm_DeltaSigma = 0.0;
    double norm_DeltaSigma_diff = 0.0;
    if (include_gw_sigma && _mpi->node_comm.root()) {
      for (int it = 0; it < _nts; ++it) {
        for (int is = 0; is < _ns; ++is) {
          for (int ik = 0; ik < _nkpts_ibz; ++ik) {
            auto DeltaSigma_k = sDeltaSigma_tskij->local()(it, is, ik, nda::range::all, nda::range::all);
            norm_DeltaSigma += std::pow(nda::frobenius_norm(DeltaSigma_k), 2);

            if (iter > 1) {
              auto DeltaSigma_prev_k = sDeltaSigma_prev_tskij.local()(it, is, ik, nda::range::all, nda::range::all);
              nda::matrix<ComplexType> diff_S = DeltaSigma_k - DeltaSigma_prev_k;
              norm_DeltaSigma_diff += std::pow(nda::frobenius_norm(diff_S), 2);
            }
          }
        }
      }
      norm_DeltaSigma = std::sqrt(norm_DeltaSigma);
      norm_DeltaSigma_diff = std::sqrt(norm_DeltaSigma_diff);
    }
    if (include_gw_sigma) {
      _mpi->comm.broadcast_n(&norm_DeltaSigma, 1, 0);
      _mpi->comm.broadcast_n(&norm_DeltaSigma_diff, 1, 0);
    }

    // Log iteration
    if (iter == 1) {
      if (include_gw_sigma) {
        app_log(1, "  {:4d}    {:.6e}     {:13s}   {:.6e}   {:13s}   {:.6e}   {:13s}   {:.3e}",
                iter, norm_DeltaDm, "---", norm_DeltaF, "---", norm_DeltaSigma, "---", Delta_mu);
      } else {
        app_log(1, "  {:4d}    {:.6e}     {:13s}   {:.6e}   {:13s}   {:.3e}",
                iter, norm_DeltaDm, "---", norm_DeltaF, "---", Delta_mu);
      }
    } else {
      if (include_gw_sigma) {
        app_log(1, "  {:4d}    {:.6e}     {:.6e}      {:.6e}   {:.6e}    {:.6e}   {:.6e}    {:.3e}",
                iter, norm_DeltaDm, norm_DeltaDm_diff, norm_DeltaF, norm_DeltaF_diff,
                norm_DeltaSigma, norm_DeltaSigma_diff, Delta_mu);
      } else {
        app_log(1, "  {:4d}    {:.6e}     {:.6e}      {:.6e}   {:.6e}    {:.3e}",
                iter, norm_DeltaDm, norm_DeltaDm_diff, norm_DeltaF, norm_DeltaF_diff, Delta_mu);
      }
    }

    // Step 5: Check convergence (all active quantities must converge)
    if (iter > 1) {
      bool dm_converged = norm_DeltaDm_diff < tol;
      bool f_converged = !need_hf || norm_DeltaF_diff < tol;
      bool sigma_converged = !include_gw_sigma || norm_DeltaSigma_diff < tol;
      if (dm_converged && f_converged && sigma_converged) {
        converged = true;
        break;
      }
    }

    _mpi->comm.barrier();
  }

  _Timer.stop("LR_SCF");

  // Report results
  if (converged) {
    app_log(1, "\n  LR SCF converged in {} iterations!", iter);
  } else if (max_iter > 1) {
    app_log(1, "\n  [WARNING] LR SCF did NOT converge after {} iterations.", max_iter);
  }
  app_log(1, "  Final Δμ = {:.6e}", Delta_mu);

  print_timers();

  return std::make_tuple(iter, Delta_mu);
}


void lr_driver::print_timers() {
  app_log(2, "\n  LR_DRIVER timers");
  app_log(2, "  -----------------");
  app_log(2, "    Total LR SCF:               {0:8.3f} sec  {1:4d} calls", _Timer.elapsed("LR_SCF"), _Timer.number_of_calls("LR_SCF"));
  app_log(2, "      - LR Setup:               {0:8.3f} sec  {1:4d} calls", _Timer.elapsed("LR_SETUP"), _Timer.number_of_calls("LR_SETUP"));
  app_log(2, "      - LR Dyson (total):       {0:8.3f} sec  {1:4d} calls", _Timer.elapsed("LR_DYSON"), _Timer.number_of_calls("LR_DYSON"));
  app_log(2, "      - LR HF (total):          {0:8.3f} sec  {1:4d} calls", _Timer.elapsed("LR_HF"), _Timer.number_of_calls("LR_HF"));
  app_log(2, "      - LR GW Sigma (total):    {0:8.3f} sec  {1:4d} calls", _Timer.elapsed("LR_GW_SIGMA"), _Timer.number_of_calls("LR_GW_SIGMA"));
  app_log(2, "      - LR GW Pi (total):       {0:8.3f} sec  {1:4d} calls", _Timer.elapsed("LR_GW_PI"), _Timer.number_of_calls("LR_GW_PI"));
  app_log(2, "      - LR GW W (total):        {0:8.3f} sec  {1:4d} calls", _Timer.elapsed("LR_GW_W"), _Timer.number_of_calls("LR_GW_W"));
  app_log(2, "      - LR GW Sig T2 (total):   {0:8.3f} sec  {1:4d} calls", _Timer.elapsed("LR_GW_SIGMA_TERM2"), _Timer.number_of_calls("LR_GW_SIGMA_TERM2"));
  app_log(2, "      - LR Iter Alg (total):    {0:8.3f} sec  {1:4d} calls\n", _Timer.elapsed("LR_ITER_ALG"), _Timer.number_of_calls("LR_ITER_ALG"));
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
    int, double, bool, const lr_iter_params&);

} // namespace methods
