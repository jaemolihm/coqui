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

#ifndef COQUI_DIIS_TIMERS_HPP
#define COQUI_DIIS_TIMERS_HPP

// Cumulative timers for the Dyson-SCF mixing internals (commutator residual and
// DIIS). They run on every rank; only the root's values are printed, so plain
// inline (single-definition) Watch objects suffice with no MPI coordination.
// Instrumentation only: wrapping start/stop pairs, no change to numerics.

#include "utilities/Timer.hpp"
#include "IO/app_loggers.h"

namespace iter_scf::diis_timers {

  // k-striped distributed commutator sub-timers
  inline utils::Watch com_dist_total    {"commutator_dist_total"};
  inline utils::Watch com_dist_scan     {"commutator_dist_sigma_scan"};
  inline utils::Watch com_dist_ftG      {"commutator_dist_tau_to_w_G"};
  inline utils::Watch com_dist_ftSigma  {"commutator_dist_tau_to_w_Sigma"};
  inline utils::Watch com_dist_gemm     {"commutator_dist_iwsk_gemm"};
  inline utils::Watch com_dist_wtau     {"commutator_dist_w_to_tau"};
  inline utils::Watch com_dist_reduce   {"commutator_dist_reduce"};

  // Element-sliced mixing sub-timers. Per-rank; the global root's values
  // are the ones printed.
  inline utils::Watch mix_total      {"mix_total"};
  inline utils::Watch mix_hist_store {"mix_hist_store_slices"};
  inline utils::Watch mix_B_dots     {"mix_B_row_dots"};
  inline utils::Watch mix_B_comm     {"mix_B_row_comm"};
  inline utils::Watch mix_coefs      {"mix_coefs_solve"};
  inline utils::Watch mix_lincomb    {"mix_lincomb"};
  inline utils::Watch mix_conv       {"mix_conv_maxdiff"};
  inline utils::Watch mix_damp_mix   {"mix_damp_mix"};
  inline utils::Watch mix_prev_store {"mix_prev_state_store"};

  inline void log() {
    auto row = [](utils::Watch& w) {
      app_log(2, "  {:<28}: {:12.4f} s   (calls {})", w.name, w.elapsed(), w.number_of_calls());
    };
    app_log(2, "\n[DIIS breakdown] cumulative over all Dyson-SCF mixing steps:");
    row(com_dist_total);
    row(com_dist_scan);
    row(com_dist_ftG);
    row(com_dist_ftSigma);
    row(com_dist_gemm);
    row(com_dist_wtau);
    row(com_dist_reduce);
    row(mix_total);
    row(mix_hist_store);
    row(mix_B_dots);
    row(mix_B_comm);
    row(mix_coefs);
    row(mix_lincomb);
    row(mix_conv);
    row(mix_damp_mix);
    row(mix_prev_store);
    app_log(2, "");
  }

  // RAII guard: logs the cumulative breakdown when a mixing step returns.
  struct ScopeLog {
    ~ScopeLog() { log(); }
  };

} // namespace iter_scf::diis_timers

#endif // COQUI_DIIS_TIMERS_HPP
