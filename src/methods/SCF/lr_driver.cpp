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
             "╔═╗╔═╗╔═╗ ╦ ╦╦  ┬  ┬─┐   ┬ ┬┌─┐  ┌─┐┌─┐┌─┐\n"
             "║  ║ ║║═╬╗║ ║║  │  ├┬┘───├─┤├┤   └─┐│  ├┤ \n"
             "╚═╝╚═╝╚═╝╚╚═╝╩  ┴─┘┴└─   ┴ ┴└    └─┘└─┘└  \n");
  app_log(1, "  Linear Response Hartree-Fock SCF Driver");
  app_log(1, "  q-vector: ({:.6f}, {:.6f}, {:.6f})",
          q_vec(0), q_vec(1), q_vec(2));
  app_log(1, "  q is Gamma point: {}\n", _lr_dyson.is_q_gamma() ? "yes" : "no");

  for (auto& v : {"LR_HF_SCF", "LR_DYSON", "LR_HF"}) {
    _Timer.add(v);
  }
  _mpi->comm.barrier();
}


template<THC_ERI THC_t>
std::tuple<int, double> lr_driver::run_lr_hf(
    sArray_t<Array_view_5D_t>& sDeltaG_tskij,
    sArray_t<Array_view_4D_t>& sDeltaDm_skij,
    sArray_t<Array_view_4D_t>& sDeltaF_skij,
    const sArray_t<Array_view_5D_t>& sG_tskij,
    const sArray_t<Array_view_4D_t>& sDeltaH0_skij,
    THC_t& thc,
    int max_iter,
    double tol,
    bool fix_density) {

  _Timer.start("LR_HF_SCF");

  app_log(1, "Starting LR-HF SCF loop:");
  app_log(1, "  max_iter = {}", max_iter);
  app_log(1, "  tol = {:.2e}", tol);
  app_log(1, "  fix_density = {}", fix_density ? "true" : "false");

  // Initialize lr_hf solver if not already done
  if (!_lr_hf) {
    _lr_hf = std::make_unique<solvers::lr_hf>(_mpi, _MF, _lr_dyson.q_vec());
  }

  // Get overlap matrix from dyson
  auto& sS_skij = _dyson.sS_skij();

  // Allocate arrays for LR self-energy (zero for HF)
  auto sDeltaSigma_tskij = math::shm::make_shared_array<Array_view_5D_t>(
      *_mpi, {_nts, _ns, _nkpts_ibz, _nbnd, _nbnd});
  if (_mpi->node_comm.root()) {
    sDeltaSigma_tskij.local() = ComplexType(0.0);
  }
  _mpi->comm.barrier();

  // Allocate array for previous density matrix (for convergence check)
  auto sDeltaDm_prev_skij = math::shm::make_shared_array<Array_view_4D_t>(
      *_mpi, {_ns, _nkpts_ibz, _nbnd, _nbnd});

  // Initialize ΔF = 0
  if (_mpi->node_comm.root()) {
    sDeltaF_skij.local() = ComplexType(0.0);
  }
  _mpi->comm.barrier();

  double Delta_mu = 0.0;
  int iter = 0;
  bool converged = false;

  // SCF iteration header
  app_log(1, "\n  iter    ||ΔDm||         ||ΔDm-ΔDm_prev||  ||ΔF||          Δμ");
  app_log(1, "  " + std::string(75, '-'));

  for (iter = 1; iter <= max_iter; ++iter) {

    // Save previous density matrix for convergence check
    if (_mpi->node_comm.root() && iter > 1) {
      sDeltaDm_prev_skij.local() = sDeltaDm_skij.local();
    }
    _mpi->comm.barrier();

    // Step 1: Solve LR Dyson equation
    // ΔG = G_{k+q} · [ΔH0 + ΔF - Δμ·S] · G_k
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

    // Step 2: Compute LR Fock matrix
    // ΔF = ΔJ + ΔK from ΔDm
    _Timer.start("LR_HF");
    _lr_hf->evaluate(sDeltaF_skij, sDeltaDm_skij, thc, sS_skij.local(),
                     true,   // compute_hartree
                     true);  // compute_exchange
    _Timer.stop("LR_HF");
    _mpi->comm.barrier();

    // Compute norm of ΔF for logging
    double norm_DeltaF = 0.0;
    if (_mpi->node_comm.root()) {
      for (int is = 0; is < _ns; ++is) {
        for (int ik = 0; ik < _nkpts_ibz; ++ik) {
          auto DeltaF_k = sDeltaF_skij.local()(is, ik, nda::range::all, nda::range::all);
          norm_DeltaF += std::pow(nda::frobenius_norm(DeltaF_k), 2);
        }
      }
      norm_DeltaF = std::sqrt(norm_DeltaF);
    }
    _mpi->comm.broadcast_n(&norm_DeltaF, 1, 0);

    // Log iteration
    if (iter == 1) {
      app_log(1, "  {:4d}    {:.6e}     {:13s}   {:.6e}   {:.3e}",
              iter, norm_DeltaDm, "---", norm_DeltaF, Delta_mu);
    } else {
      app_log(1, "  {:4d}    {:.6e}     {:.6e}      {:.6e}   {:.3e}",
              iter, norm_DeltaDm, norm_DeltaDm_diff, norm_DeltaF, Delta_mu);
    }

    // Step 3: Check convergence
    if (iter > 1 && norm_DeltaDm_diff < tol) {
      converged = true;
      break;
    }

    _mpi->comm.barrier();
  }

  _Timer.stop("LR_HF_SCF");

  // Report results
  if (converged) {
    app_log(1, "\n  LR-HF SCF converged in {} iterations!", iter);
  } else {
    app_log(1, "\n  [WARNING] LR-HF SCF did NOT converge after {} iterations.", max_iter);
  }
  app_log(1, "  Final Δμ = {:.6e}", Delta_mu);

  print_timers();

  return std::make_tuple(iter, Delta_mu);
}


void lr_driver::print_timers() {
  app_log(2, "\n  LR_DRIVER timers");
  app_log(2, "  -----------------");
  app_log(2, "    Total LR-HF SCF:            {0:.3f} sec", _Timer.elapsed("LR_HF_SCF"));
  app_log(2, "      - LR Dyson (total):       {0:.3f} sec", _Timer.elapsed("LR_DYSON"));
  app_log(2, "      - LR HF (total):          {0:.3f} sec\n", _Timer.elapsed("LR_HF"));
}


// Template instantiations
template std::tuple<int, double> lr_driver::run_lr_hf(
    sArray_t<Array_view_5D_t>&,
    sArray_t<Array_view_4D_t>&,
    sArray_t<Array_view_4D_t>&,
    const sArray_t<Array_view_5D_t>&,
    const sArray_t<Array_view_4D_t>&,
    thc_reader_t&,
    int, double, bool);

} // namespace methods
