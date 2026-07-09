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

// D2 diagnostic: cumulative timers for the Dyson-DIIS internals.
// Only the root rank runs this code path, so plain inline (single-definition)
// Watch objects suffice with no MPI coordination. Instrumentation only:
// wrapping start/stop pairs, no change to numerics.

#include "utilities/Timer.hpp"
#include "IO/app_loggers.h"

namespace iter_scf::diis_timers {

  // commutator_t sub-timers
  inline utils::Watch com_total   {"commutator_total"};
  inline utils::Watch com_alloc   {"commutator_alloc"};    // G_w/Sigma_w/C_w/C_t alloc+zero
  inline utils::Watch com_ftG     {"commutator_tau_to_w_G"};
  inline utils::Watch com_ftSigma {"commutator_tau_to_w_Sigma"};
  inline utils::Watch com_gemm    {"commutator_iwsk_gemm"};
  inline utils::Watch com_wtau    {"commutator_w_to_tau"};

  // A10 k-striped distributed commutator sub-timers (measured on root, which
  // processes its own (s,k) subset; the D2 split is mirrored per component).
  inline utils::Watch com_dist_total    {"commutator_dist_total"};
  inline utils::Watch com_dist_scan     {"commutator_dist_sigma_scan"};
  inline utils::Watch com_dist_ftG      {"commutator_dist_tau_to_w_G"};
  inline utils::Watch com_dist_ftSigma  {"commutator_dist_tau_to_w_Sigma"};
  inline utils::Watch com_dist_gemm     {"commutator_dist_iwsk_gemm"};
  inline utils::Watch com_dist_wtau     {"commutator_dist_w_to_tau"};
  inline utils::Watch com_dist_reduce   {"commutator_dist_reduce"};

  // FockSigma whole-vector value copies (copy-ctor + value-ctor + copy-assign);
  // covers opt_state put/get and VSpace get_vec copies (aggregate)
  inline utils::Watch fs_copies   {"focksigma_copies"};

  // A11: residual-chain whole-vector (nda-array-level) copies that bypass the
  // FockSigma copy ctor/assign and are therefore NOT in fs_copies.
  inline utils::Watch res_inject_copy {"residual_inject_copy"};  // upload_residual C_t copy
  inline utils::Watch gmu_inject_copy {"g_mu_inject_copy"};      // upload_g_mu G copy
  inline utils::Watch res_set_copy    {"residual_set_into_res"}; // set_fock_sigma into res
  inline utils::Watch res_fz          {"residual_Fz_zero"};      // Fz alloc + zero
  inline utils::Watch solve_convscan  {"solve_conv_maxdiff_scan"}; // max|ΔF|,max|ΔΣ|
  inline utils::Watch solve_copyback  {"solve_extrap_copyback"};   // F=,Sigma= result

  // VSpace operations (inclusive: lincomb/get overlap fs_copies etc. nested)
  inline utils::Watch vsp_add     {"vspace_add_to_vspace"};
  inline utils::Watch vsp_get     {"vspace_get_vec"};
  inline utils::Watch vsp_overlap {"vspace_overlap"};
  inline utils::Watch vsp_lincomb {"vspace_make_linear_comb"};

  // damp_t::solve checkpoint read (warmup iterations)
  inline utils::Watch damp_read   {"damp_checkpoint_read"};
  inline utils::Watch damp_mix    {"damp_mix_arithmetic"};   // A11: warmup mixing passes

  // diis_t::get_mu checkpoint read
  inline utils::Watch get_mu      {"diis_get_mu"};

  // A12 SPMD (element-sliced) FockSigma DIIS sub-timers. Per-rank; the global
  // root's values are the ones printed. Rough mapping to the serial path:
  // hist_store ~ fs_copies + vsp_add + inject, B_dots ~ vsp_overlap,
  // lincomb ~ vsp_lincomb, damp_mix ~ damp_mix, damp_stage ~ damp_read,
  // conv ~ solve_convscan, writeback ~ solve_copyback.
  inline utils::Watch spmd_total      {"spmd_total"};
  inline utils::Watch spmd_hist_store {"spmd_hist_store_slices"};
  inline utils::Watch spmd_B_dots     {"spmd_B_row_dots"};
  inline utils::Watch spmd_B_comm     {"spmd_B_row_comm"};
  inline utils::Watch spmd_coefs      {"spmd_coefs_solve"};
  inline utils::Watch spmd_lincomb    {"spmd_lincomb"};
  inline utils::Watch spmd_conv       {"spmd_conv_maxdiff"};
  inline utils::Watch spmd_writeback  {"spmd_writeback_slices"};
  inline utils::Watch spmd_damp_mix   {"spmd_damp_mix"};
  inline utils::Watch spmd_damp_stage {"spmd_damp_stage_slice"};
  inline utils::Watch spmd_prev_store {"spmd_prev_state_store"};

  inline void log() {
    auto row = [](utils::Watch& w) {
      app_log(2, "  {:<28}: {:12.4f} s   (calls {})", w.name, w.elapsed(), w.number_of_calls());
    };
    app_log(2, "\n[D2 DIIS breakdown] cumulative over all Dyson-DIIS solves:");
    row(com_total);
    row(com_alloc);
    row(com_ftG);
    row(com_ftSigma);
    row(com_gemm);
    row(com_wtau);
    row(com_dist_total);
    row(com_dist_scan);
    row(com_dist_ftG);
    row(com_dist_ftSigma);
    row(com_dist_gemm);
    row(com_dist_wtau);
    row(com_dist_reduce);
    row(fs_copies);
    row(res_inject_copy);
    row(gmu_inject_copy);
    row(res_set_copy);
    row(res_fz);
    row(solve_convscan);
    row(solve_copyback);
    row(vsp_add);
    row(vsp_get);
    row(vsp_overlap);
    row(vsp_lincomb);
    row(damp_read);
    row(damp_mix);
    row(get_mu);
    row(spmd_total);
    row(spmd_hist_store);
    row(spmd_B_dots);
    row(spmd_B_comm);
    row(spmd_coefs);
    row(spmd_lincomb);
    row(spmd_conv);
    row(spmd_writeback);
    row(spmd_damp_mix);
    row(spmd_damp_stage);
    row(spmd_prev_store);
    app_log(2, "");
  }

  // RAII guard: logs the cumulative breakdown when a Dyson-DIIS solve returns.
  struct ScopeLog {
    ~ScopeLog() { log(); }
  };

} // namespace iter_scf::diis_timers

#endif // COQUI_DIIS_TIMERS_HPP
