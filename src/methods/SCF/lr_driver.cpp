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

  for (auto& v : {"LR_SCF", "LR_DYSON", "LR_HF", "LR_GW_SIGMA"}) {
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
    bool include_hartree, bool include_exchange, bool include_gw_sigma,
    dW_t* dW_qtPQ, const nda::array<ComplexType, 1>* eps_inv_head,
    int max_iter, double tol, bool fix_density,
    const lr_iter_params& iter_params) {

  _Timer.start("LR_SCF");

  bool need_hf = include_hartree || include_exchange;

  utils::check(iter_params.alg == "damping" || iter_params.alg == "DIIS",
               "lr_driver::run_lr: unknown iter_alg '{}'. Must be 'damping' or 'DIIS'.",
               iter_params.alg);
  if (include_gw_sigma) {
    utils::check(sDeltaSigma_tskij != nullptr,
                 "lr_driver::run_lr: include_gw_sigma=true but sDeltaSigma_tskij is null.");
    utils::check(dW_qtPQ != nullptr && eps_inv_head != nullptr,
                 "lr_driver::run_lr: include_gw_sigma=true but dW_qtPQ or eps_inv_head is null.");
  }

  bool use_diis = (iter_params.alg == "DIIS");
  double mixing = iter_params.mixing;

  app_log(1, "Starting Linear Response SCF loop:");
  app_log(1, "  max_iter = {}", max_iter);
  app_log(1, "  tol = {:.2e}", tol);
  app_log(1, "  fix_density = {}", fix_density ? "true" : "false");
  app_log(1, "  include_hartree = {}", include_hartree ? "true" : "false");
  app_log(1, "  include_exchange = {}", include_exchange ? "true" : "false");
  app_log(1, "  include_gw_sigma = {}", include_gw_sigma ? "true" : "false");
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

  // Initialize ΔF = 0 (and ΔΣ = 0 if GW active)
  if (_mpi->node_comm.root()) {
    sDeltaF_skij.local() = ComplexType(0.0);
    if (sDeltaSigma_tskij) sDeltaSigma_tskij->local() = ComplexType(0.0);
  }
  _mpi->comm.barrier();

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
        sDeltaG_tskij, sDeltaDm_skij, sG_tskij, sDeltaH0_skij,
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

    // Step 3: Compute LR GW self-energy (if requested)
    if (include_gw_sigma) {
      _Timer.start("LR_GW_SIGMA");
      // TODO: evaluate_sigma_fixed_W could accept a shared_array view directly
      // to avoid this O(nt*ns*nk*nb^2) copy every iteration.
      nda::array<ComplexType, 5> DeltaG_local;
      if (_mpi->node_comm.root()) {
        DeltaG_local = nda::array<ComplexType, 5>{sDeltaG_tskij.local()};
      }
      _mpi->comm.barrier();

      lr_gw_solver->evaluate_sigma_fixed_W(
          *sDeltaSigma_tskij, DeltaG_local, *dW_qtPQ, thc, *eps_inv_head);
      _Timer.stop("LR_GW_SIGMA");
      _mpi->comm.barrier();
    }

    // Step 4: Apply iteration algorithm (DIIS or damping) on combined (ΔF, ΔΣ)
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
  app_log(2, "    Total LR SCF:               {0:.3f} sec", _Timer.elapsed("LR_SCF"));
  app_log(2, "      - LR Dyson (total):       {0:.3f} sec", _Timer.elapsed("LR_DYSON"));
  app_log(2, "      - LR HF (total):          {0:.3f} sec", _Timer.elapsed("LR_HF"));
  app_log(2, "      - LR GW Sigma (total):    {0:.3f} sec\n", _Timer.elapsed("LR_GW_SIGMA"));
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
    bool, bool, bool,
    dW_concrete_t*, const nda::array<ComplexType, 1>*,
    int, double, bool, const lr_iter_params&);

} // namespace methods
