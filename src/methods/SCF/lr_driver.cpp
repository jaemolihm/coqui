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
#include "methods/HF/thc_solver_comm.hpp"
#include "methods/scr_coulomb/scr_coulomb_fourier_t.h"
#include "numerics/distributed_array/nda.hpp"
#include "utilities/check.hpp"
#include "utilities/lr_utils.hpp"
#include "utilities/proc_meminfo.hpp"
#include "methods/GW/g0_div_utils.hpp"
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

  for (auto& v : {"LR_SCF", "LR_DRIVER_SETUP",
                  "LR_DRIVER_SETUP_W_FULL", "LR_DRIVER_SETUP_W_TRPQ", "LR_DRIVER_SETUP_G_OMEGA", "LR_DRIVER_SETUP_G_R",
                  "LR_DRIVER_SETUP_DN_DMU", "LR_DRIVER_SETUP_ALLOC", "LR_DRIVER_SETUP_IBC", "LR_DRIVER_SETUP_MISC",
                  "LR_DYSON", "LR_HF", "LR_GW_SIGMA", "LR_GW_DW_TRANSPOSE",
                   "LR_GW_PI", "LR_GW_W", "LR_GW_SIGMA_TERM2",
                   "LR_ITER_ALG", "LR_SAVE", "LR_CONVERGENCE"}) {
    _Timer.add(v);
  }
  _mpi->comm.barrier();
}


// Distribution flow through the LR-GW pipeline
// Three distribution patterns:
//   τ-dist (q-local):  pgrid = {tpools, 1, np_P, np_Q}  — q undivided
//   τ-local:           pgrid = {1, nqpools, np_P, np_Q}  — tau/omega undivided (FT buffer)
//   ω-side:            pgrid = {nwpools, nqpools, np_P, np_Q}  — distributed over (w, q, P, Q)
//
//   W_c loaded:          (q,t,P,Q), τ-dist
//   compute_W_full_omega: τ-dist → FT(τ→ω) → ω-side, + Z(q) → W_full(iω) [cached]
//
//   evaluate_lr_Pi:      → (t,q,P,Q), τ-dist
//   solve_lr_dyson_W (in-place):
//     tau_to_w:          τ-dist → ω-side (via q-distributed FT buffer)
//     lr_dyson_W_in_place: ω-side; SLATE GEMM batched over (iw, iq).
//                          For Q≠Γ, gathers W_full(kpq_map(iq)) via Alltoallv on q_pool_comm.
//     w_to_tau:          ω-side → τ-dist (via q-distributed FT buffer)
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
    const lr_iter_params& iter_params,
    const sArray_t<Array_view_4D_t>* sDeltaX_left,
    const sArray_t<Array_view_4D_t>* sDeltaX_right,
    const nda::array<ComplexType, 4>* Dm_ab,
    bool div_corr,
    std::string div_treatment,
    const nda::array_view<ComplexType, 3>* DeltaV_qPQ,
    sArray_t<Array_view_5D_t>* sDeltaSigma_term2_tskij) {

  _Timer.start("LR_SCF");

  bool need_hf = include_hartree || include_exchange;
  bool include_gw_sigma = (gw_mode != lr_gw_update_mode::none);
  bool gw_full = (gw_mode == lr_gw_update_mode::full);
  // One-shot G0W0: store the two ΔΣ terms separately instead of fusing them.
  // ΔΣ term1 (-ΔG⊙W_c) → sDeltaSigma_tskij, ΔΣ term2 (-G⊙ΔW) → sDeltaSigma_term2.
  bool split_sigma_terms = (sDeltaSigma_term2_tskij != nullptr);

  utils::check(iter_params.alg == "damping" || iter_params.alg == "DIIS",
               "lr_driver::run_lr: unknown iter_alg '{}'. Must be 'damping' or 'DIIS'.",
               iter_params.alg);
  if (include_gw_sigma) {
    utils::check(sDeltaSigma_tskij != nullptr,
                 "lr_driver::run_lr: gw_mode != none but sDeltaSigma_tskij is null.");
    utils::check(dW_qtPQ != nullptr && eps_inv_head != nullptr,
                 "lr_driver::run_lr: gw_mode != none but dW_qtPQ or eps_inv_head is null.");
  }
  if (split_sigma_terms) {
    utils::check(gw_full,
                 "lr_driver::run_lr: split ΔΣ terms require gw_mode='full'.");
    utils::check(max_iter == 1,
                 "lr_driver::run_lr: split ΔΣ terms are only meaningful for a "
                 "one-shot solve (max_iter=1), got max_iter={}.", max_iter);
    utils::check(sDeltaX_left == nullptr && sDeltaX_right == nullptr,
                 "lr_driver::run_lr: split ΔΣ terms do not support the DeltaX IBC "
                 "correction (term 2 has no IBC path).");
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

  // Estimate the persistent large-array memory footprint for this path, then
  // summarize the MPI distribution patterns the large arrays use.
  print_memory_estimate(thc.Np(), include_gw_sigma, gw_full);
  print_distribution_summary(thc.Np(), include_gw_sigma, gw_full);

  // Force dW_qtPQ onto the canonical LR q-local distribution. The fused ΔΣ loop
  // pairs the P/Q tile of W_c (from dW_qtPQ → dW_tRPQ) with that of ΔW and the
  // G^R cache, both built via lr_W_q_local_dist. When dW_qtPQ arrives on a
  // different tiling — e.g. mb_state.dW_qtPQ from full-scGW inherits the scGW
  // polarizability's block_size={1,1,1,1} — the contiguous PQ split disagrees
  // whenever Np % np_P != 0 (bsize=1 vs bsize=Np/np_P round differently), and
  // the fused pairing aborts. Redistributing here makes all operands share one
  // tiling by construction. (The from-file G0W0 path already builds dW_qtPQ on
  // this distribution, so the redistribute is a no-op there.)
  if (include_gw_sigma) {
    long nt_f = sG_tskij.shape()[0];
    long nt_half = (nt_f % 2 == 0) ? nt_f / 2 : nt_f / 2 + 1;
    auto [tq_pgrid, tq_bsize] =
        utils::lr_W_q_local_dist(_mpi->comm.size(), nt_half, thc.Np());
    std::array<long, 4> qt_pgrid = {tq_pgrid[1], tq_pgrid[0], tq_pgrid[2], tq_pgrid[3]};
    std::array<long, 4> qt_bsize = {tq_bsize[1], tq_bsize[0], tq_bsize[2], tq_bsize[3]};
    if (dW_qtPQ->grid() != qt_pgrid || dW_qtPQ->block_size() != qt_bsize) {
      app_log(2, "lr_driver::run_lr: redistributing dW_qtPQ onto canonical "
                 "LR q-local tiling (pgrid ({},{},{},{}), bsize ({},{},{},{}))",
              qt_pgrid[0], qt_pgrid[1], qt_pgrid[2], qt_pgrid[3],
              qt_bsize[0], qt_bsize[1], qt_bsize[2], qt_bsize[3]);
      math::nda::redistribute_in_place(*dW_qtPQ, qt_pgrid, qt_bsize);
    }
  }

  // Initialize lr_hf solver if needed
  if (need_hf && !_lr_hf) {
    _lr_hf = std::make_unique<solvers::lr_hf>(_mpi, _MF, _lr_dyson.q_vec());
  }

  // Create lr_gw solver if needed (local to this call, no need to store as member)
  std::unique_ptr<solvers::lr_gw> lr_gw_solver;
  if (include_gw_sigma) {
    lr_gw_solver = std::make_unique<solvers::lr_gw>(_dyson.FT(), _lr_dyson.q_vec(), div_treatment);
  }
  // Split-term mode (one-shot G0W0) needs a second lr_gw for term 2: each solver
  // caches a workspace keyed on its (term1,term2) usage. Built once here rather
  // than in the (single-iteration) loop.
  std::unique_ptr<solvers::lr_gw> lr_gw_solver2;
  if (split_sigma_terms) {
    lr_gw_solver2 = std::make_unique<solvers::lr_gw>(_dyson.FT(), _lr_dyson.q_vec(), div_treatment);
  }

  // Create lr_rpa_pi and lr_scr_coulomb_t solvers for full mode
  std::unique_ptr<solvers::lr_rpa_pi> lr_pi_solver;
  std::unique_ptr<solvers::lr_scr_coulomb_t> lr_scr_solver;
  if (gw_full) {
    lr_pi_solver = std::make_unique<solvers::lr_rpa_pi>(_lr_dyson.q_vec());
    lr_scr_solver = std::make_unique<solvers::lr_scr_coulomb_t>(_dyson.FT(), _lr_dyson.q_vec());
  }
  // Precompute W_full(iω) = W_c(iω) + V (cached across iterations).
  _Timer.start("LR_DRIVER_SETUP");
  using local_Array_4D_t = nda::array<ComplexType, 4>;
  std::optional<memory::darray_t<local_Array_4D_t, mpi3::communicator>> opt_dW_full_wqPQ;
  if (gw_full) {
    _Timer.start("LR_DRIVER_SETUP_W_FULL");
    // Transpose dW_qtPQ from (q,t,P,Q) → (t,q,P,Q), then compute W_full(iω)
    auto dW_tqPQ = utils::transpose_axes_01(*dW_qtPQ, _mpi->comm);
    opt_dW_full_wqPQ.emplace(
        lr_scr_solver->compute_W_full_omega(dW_tqPQ, thc));
    // dW_tqPQ is consumed (reset) by compute_W_full_omega
    _Timer.stop("LR_DRIVER_SETUP_W_FULL");
  }

  // Precompute W in R-space: transpose (q,t)→(t,R), with q→R FT.
  // Result: dW_tRPQ with (t,R,P,Q) layout, pgrid (tpools,1,np_P,np_Q).
  std::optional<dW_t> opt_dW_tRPQ;
  if (include_gw_sigma) {
    _Timer.start("LR_DRIVER_SETUP_W_TRPQ");
    opt_dW_tRPQ.emplace(lr_precompute_W_tRPQ(*dW_qtPQ, thc));
    dW_qtPQ->reset();  // free (q,t,P,Q) memory; no longer needed
    _Timer.stop("LR_DRIVER_SETUP_W_TRPQ");
  }

  // Precompute the unperturbed G^R(τ)/G^R(β−τ) pair in aux basis (constant
  // across SCF iterations; consumed by evaluate_lr_Pi and Σ term 2).
  using dArr_5D_t = memory::darray_t<nda::array<ComplexType, 5>, mpi3::communicator>;
  std::optional<dArr_5D_t> opt_dG_tsRPQ, opt_dG_mtau_tsRPQ;
  if (gw_full) {
    _Timer.start("LR_DRIVER_SETUP_G_R");
    utils::memlog("lr_driver::run_lr: before G^R pair precompute");
    auto [dG_tsRPQ, dG_mtau_tsRPQ] = lr_precompute_G_R_pair(sG_tskij.local(), thc);
    opt_dG_tsRPQ.emplace(std::move(dG_tsRPQ));
    opt_dG_mtau_tsRPQ.emplace(std::move(dG_mtau_tsRPQ));
    utils::memlog("lr_driver::run_lr: after G^R pair precompute");
    _Timer.stop("LR_DRIVER_SETUP_G_R");
  }

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
  // Initialize DIIS if requested
  if (use_diis) {
    _lr_diis = std::make_unique<lr_diis>(
        iter_params.max_subsp_size, iter_params.diis_warmup, mixing);
  }

  // Get overlap matrix from dyson
  auto& sS_skij = _dyson.sS_skij();
  _Timer.stop("LR_DRIVER_SETUP_MISC");

  _Timer.start("LR_DRIVER_SETUP_ALLOC");
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
  _Timer.stop("LR_DRIVER_SETUP_ALLOC");

  _Timer.start("LR_DRIVER_SETUP_MISC");
  // Initialize ΔF = 0 (and ΔΣ = 0 if GW active); set_zero ends with fence + node_sync
  sDeltaF_skij.set_zero();
  if (sDeltaSigma_tskij) sDeltaSigma_tskij->set_zero();
  _mpi->comm.barrier();
  _Timer.stop("LR_DRIVER_SETUP_MISC");

  // DeltaX IBC correction setup
  bool has_deltax = sDeltaX_left && sDeltaX_right;
  std::optional<lr_ibc_DeltaX> opt_ibc;
  if (has_deltax) {
    _Timer.start("LR_DRIVER_SETUP_IBC");
    app_log(2, "  DeltaX IBC correction: building lr_ibc_DeltaX...");

    utils::memlog("lr_driver::run_lr: before build_lr_ibc");
    opt_ibc.emplace(build_lr_ibc(
        *_mpi, _MF, thc,
        sDeltaX_left->local(), sDeltaX_right->local(),
        _lr_dyson.q_vec(), _lr_dyson.kpq_map(),
        Dm_ab, &sG_tskij,
        opt_dW_tRPQ ? &(*opt_dW_tRPQ) : nullptr,
        include_hartree, include_exchange, include_gw_sigma));

    app_log(2, "  DeltaX IBC correction: setup complete.");
    utils::memlog("lr_driver::run_lr: after build_lr_ibc");
    _Timer.stop("LR_DRIVER_SETUP_IBC");
  }
  _Timer.stop("LR_DRIVER_SETUP");
  utils::memlog("lr_driver::run_lr: end of LR_DRIVER_SETUP");
  print_setup_timers();

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
    _Timer.start("LR_SAVE");
    if (_mpi->node_comm.root() && iter > 1) {
      sDeltaDm_prev_skij.local() = sDeltaDm_skij.local();
      sDeltaF_prev_skij.local() = sDeltaF_skij.local();
      if (include_gw_sigma) {
        sDeltaSigma_prev_tskij.local() = sDeltaSigma_tskij->local();
      }
    }
    _mpi->comm.barrier();
    _Timer.stop("LR_SAVE");

    // Step 1: Solve LR Dyson equation
    // ΔG = G_{k+q} · [ΔH0 + ΔF + ΔΣ - Δμ·S] · G_k
    _Timer.start("LR_DYSON");
    Delta_mu = _lr_dyson.solve_lr_dyson(
        sDeltaG_tskij, sDeltaDm_skij, sDeltaH0_skij,
        sDeltaF_skij, sDeltaSigma_tskij,
        fix_density, Delta_mu);
    _Timer.stop("LR_DYSON");
    _mpi->comm.barrier();

    // Compute norms for logging. lr_distributed_norm stripes the (s,k) blocks
    // over node_comm ranks and reduces within the node; the shared array is
    // node-replicated, so the trailing broadcast from global rank 0 preserves
    // exact global agreement.
    _Timer.start("LR_CONVERGENCE");
    auto norms_Dm = utils::lr_distributed_norm(
        _mpi->node_comm, sDeltaDm_skij.local(), sDeltaDm_prev_skij.local(), iter > 1);
    double norm_DeltaDm = norms_Dm.first;
    double norm_DeltaDm_diff = norms_Dm.second;
    _mpi->comm.broadcast_n(&norm_DeltaDm, 1, 0);
    _mpi->comm.broadcast_n(&norm_DeltaDm_diff, 1, 0);
    _Timer.stop("LR_CONVERGENCE");

    // Step 2: Compute LR Fock matrix (if requested)
    if (need_hf) {
      _Timer.start("LR_HF");
      const lr_ibc_DeltaX* ibc_ptr = opt_ibc ? &(*opt_ibc) : nullptr;
      _lr_hf->evaluate(sDeltaF_skij, sDeltaDm_skij, thc, sS_skij.local(),
                       include_hartree, include_exchange, ibc_ptr,
                       DeltaV_qPQ, Dm_ab);
      _Timer.stop("LR_HF");
      _mpi->comm.barrier();
    }

    // Step 3: Compute LR GW self-energy
    if (include_gw_sigma && !gw_full) {
      // DeltaG-only mode: ΔΣ = -ΔG ⊙ W_c
      _Timer.start("LR_GW_SIGMA");
      {
        const lr_ibc_DeltaX* ibc_ptr = opt_ibc ? &(*opt_ibc) : nullptr;
        lr_gw_solver->evaluate_sigma_DeltaG(
            *sDeltaSigma_tskij, sDeltaG_tskij.local(), *opt_dW_tRPQ, thc,
            ibc_ptr);
      }
      // Divergence correction term 1 (all q): ΔΣ^div += -madelung * eps_inv_head * S(k+q) · ΔG · S(k)
      if (div_corr) {
        lr_gw_solver->apply_div_correction_DeltaG(
            *sDeltaSigma_tskij, sDeltaG_tskij.local(), sS_skij.local(), thc, *eps_inv_head);
      }
      _Timer.stop("LR_GW_SIGMA");
      _mpi->comm.barrier();
    }

    if (gw_full) {
      // Step 3b: ΔP = -ΔG·G - G·ΔG
      _Timer.start("LR_GW_PI");
      const lr_ibc_DeltaX* ibc_ptr = opt_ibc ? &(*opt_ibc) : nullptr;
      auto dDeltaPi_tqPQ = lr_pi_solver->evaluate_lr_Pi(
          sG_tskij.local(), sDeltaG_tskij.local(), thc,
          *opt_dG_tsRPQ, *opt_dG_mtau_tsRPQ, ibc_ptr);
      _mpi->comm.barrier();
      _Timer.stop("LR_GW_PI");

      // Step 3c-3d: ΔW_c(τ) via solve_lr_dyson_W (in-place, uses cached W_full)
      _Timer.start("LR_GW_W");
      lr_scr_solver->solve_lr_dyson_W(dDeltaPi_tqPQ, *opt_dW_full_wqPQ, thc);
      // dDeltaPi_tqPQ now contains ΔW_c(τ) in q-local distribution
      auto& dDeltaW_tqPQ = dDeltaPi_tqPQ;  // alias for clarity

      // Extract Δeps_inv_head from ΔW for divergence correction term 2 (q_pert=0 only)
      nda::array<ComplexType, 1> delta_eps_inv_head;
      if (div_corr && is_q_gamma()) {
        auto [delta_eps_inv_q, delta_head] =
            solvers::div_utils::eps_inv_head_t(
                dDeltaW_tqPQ, thc, *thc.MF(), _dyson.FT(), div_treatment);
        delta_eps_inv_head = std::move(delta_head);
      }

      _mpi->comm.barrier();
      _Timer.stop("LR_GW_W");

      // Step 3e-3f: ΔΣ = -ΔG ⊙ W_c - G ⊙ ΔW
      _Timer.start("LR_GW_SIGMA");
      _Timer.start("LR_GW_DW_TRANSPOSE");
      auto dDeltaW_qtPQ = utils::transpose_axes_01(dDeltaW_tqPQ, _mpi->comm);
      dDeltaW_tqPQ.reset();
      _Timer.stop("LR_GW_DW_TRANSPOSE");
      if (split_sigma_terms) {
        // One-shot G0W0: compute the two terms separately, then store
        //   sDeltaSigma_tskij       = term1 + term2  (total ΔΣ, same as fused)
        //   sDeltaSigma_term2_tskij = term2 (G0·dW0)  [written as DeltaSigma_GdW]
        // term 1 (-ΔG⊙W_c + div) and term 2 (-G⊙ΔW + div) use separate solver
        // instances (lr_gw_solver / lr_gw_solver2, built once above) — the
        // workspace is cached per (term1,term2) combination.
        lr_gw_solver->evaluate_sigma_DeltaG(
            *sDeltaSigma_tskij, sDeltaG_tskij.local(), *opt_dW_tRPQ, thc, ibc_ptr);
        lr_gw_solver2->evaluate_sigma_DeltaW(
            *sDeltaSigma_term2_tskij, sG_tskij.local(), dDeltaW_qtPQ, thc,
            *opt_dG_tsRPQ, *opt_dG_mtau_tsRPQ);
        if (div_corr) {
          lr_gw_solver->apply_div_correction_DeltaG(
              *sDeltaSigma_tskij, sDeltaG_tskij.local(), sS_skij.local(), thc, *eps_inv_head);
          if (is_q_gamma()) {
            lr_gw_solver2->apply_div_correction_G(
                *sDeltaSigma_term2_tskij, sG_tskij.local(), sS_skij.local(), thc, delta_eps_inv_head);
          }
        }
        // Accumulate term2 into sDeltaSigma_tskij so it holds the total ΔΣ.
        // Both arrays are node-replicated shared memory (each solver all_reduced
        // its result), so add once per node on the node root.
        sDeltaSigma_tskij->win().fence();
        sDeltaSigma_term2_tskij->win().fence();
        if (_mpi->node_comm.root())
          sDeltaSigma_tskij->local() += sDeltaSigma_term2_tskij->local();
        sDeltaSigma_tskij->win().fence();
        _mpi->comm.barrier();
      } else {
        // Fused ΔΣ = -ΔG ⊙ W_c - G ⊙ ΔW (single R-space pass)
        lr_gw_solver->evaluate_sigma(
            *sDeltaSigma_tskij, sDeltaG_tskij.local(), *opt_dW_tRPQ,
            sG_tskij.local(), dDeltaW_qtPQ, thc,
            *opt_dG_tsRPQ, *opt_dG_mtau_tsRPQ, ibc_ptr);
        // Divergence correction term 1 (all q): eps_inv_head from W, applied to ΔG
        if (div_corr) {
          lr_gw_solver->apply_div_correction_DeltaG(
              *sDeltaSigma_tskij, sDeltaG_tskij.local(), sS_skij.local(), thc, *eps_inv_head);
          // Divergence correction term 2 (q_pert=0 only): Δeps_inv_head from ΔW, applied to G
          if (is_q_gamma()) {
            lr_gw_solver->apply_div_correction_G(
                *sDeltaSigma_tskij, sG_tskij.local(), sS_skij.local(), thc, delta_eps_inv_head);
          }
        }
      }
      _mpi->comm.barrier();
      _Timer.stop("LR_GW_SIGMA");
    }

    // Step 4: Apply iteration algorithm (DIIS or damping) on combined (ΔF, ΔΣ)
    _Timer.start("LR_ITER_ALG");
    if (iter > 1 && (need_hf || include_gw_sigma)) {
      if (use_diis) {
        // Distributed DIIS: every node rank participates, each operating on its
        // element-slice of the shared ΔF/ΔΣ and writing the mixed result back in
        // place. Pass .local() views directly (in/out).
        if (include_gw_sigma) {
          _lr_diis->next_step_combined(
              _mpi->node_comm,
              sDeltaF_skij.local(), sDeltaF_prev_skij.local(),
              sDeltaSigma_tskij->local(), sDeltaSigma_prev_tskij.local(), iter);
        } else {
          nda::array<ComplexType, 5> empty_sigma;
          _lr_diis->next_step_combined(
              _mpi->node_comm,
              sDeltaF_skij.local(), sDeltaF_prev_skij.local(),
              empty_sigma, empty_sigma, iter);
        }
      } else if (mixing < 1.0 && _mpi->node_comm.root()) {
        sDeltaF_skij.local() = mixing * sDeltaF_skij.local()
                               + (1.0 - mixing) * sDeltaF_prev_skij.local();
        if (include_gw_sigma) {
          sDeltaSigma_tskij->local() = mixing * sDeltaSigma_tskij->local()
                                       + (1.0 - mixing) * sDeltaSigma_prev_tskij.local();
        }
      }
      // Distributed DIIS writes each node_comm rank's slice of the shared
      // ΔF/ΔΣ buffer in place with no trailing collective. Fence + barrier
      // make every slice visible to node root before it reads the whole
      // buffer below (barrier alone is insufficient under the MPI-3
      // separate shared-memory model).
      sDeltaF_skij.win().fence();
      if (sDeltaSigma_tskij) sDeltaSigma_tskij->win().fence();
      _mpi->node_comm.barrier();
      // The iter-alg step is computed independently on each node's shared replica,
      // so the per-node results can drift apart by floating-point noise (different
      // all_reduce instances within each node_comm). Broadcast rank 0's result to
      // all nodes so every node's replica is bit-identical before the next iter.
      if (_mpi->node_comm.root()) {
        _mpi->internode_comm.broadcast_n(sDeltaF_skij.local().data(),
                                         sDeltaF_skij.local().size(), 0);
        if (include_gw_sigma) {
          _mpi->internode_comm.broadcast_n(sDeltaSigma_tskij->local().data(),
                                           sDeltaSigma_tskij->local().size(), 0);
        }
      }
      // Make node root's overwrite visible to its node peers
      sDeltaF_skij.win().fence();
      if (sDeltaSigma_tskij) sDeltaSigma_tskij->win().fence();
      _mpi->comm.barrier();
    }
    _Timer.stop("LR_ITER_ALG");

    // Compute norms of ΔF and ΔF-ΔF_prev for logging
    _Timer.start("LR_CONVERGENCE");
    auto norms_F = utils::lr_distributed_norm(
        _mpi->node_comm, sDeltaF_skij.local(), sDeltaF_prev_skij.local(), iter > 1);
    double norm_DeltaF = norms_F.first;
    double norm_DeltaF_diff = norms_F.second;
    _mpi->comm.broadcast_n(&norm_DeltaF, 1, 0);
    _mpi->comm.broadcast_n(&norm_DeltaF_diff, 1, 0);

    // Compute norms of ΔΣ and ΔΣ-ΔΣ_prev for logging
    double norm_DeltaSigma = 0.0;
    double norm_DeltaSigma_diff = 0.0;
    if (include_gw_sigma) {
      auto norms_Sigma = utils::lr_distributed_norm(
          _mpi->node_comm, sDeltaSigma_tskij->local(), sDeltaSigma_prev_tskij.local(), iter > 1);
      norm_DeltaSigma = norms_Sigma.first;
      norm_DeltaSigma_diff = norms_Sigma.second;
      _mpi->comm.broadcast_n(&norm_DeltaSigma, 1, 0);
      _mpi->comm.broadcast_n(&norm_DeltaSigma_diff, 1, 0);
    }

    // Log iteration
    if (iter == 1) {
      if (include_gw_sigma) {
        app_log(1, "  {:4d}    {:.6e}     {:13s}     {:.6e}   {:13s}   {:.6e}   {:13s}   {:.3e}",
                iter, norm_DeltaDm, "---", norm_DeltaF, "---", norm_DeltaSigma, "---", Delta_mu);
      } else {
        app_log(1, "  {:4d}    {:.6e}     {:13s}     {:.6e}   {:13s}   {:.3e}",
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

    _Timer.stop("LR_CONVERGENCE");

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

  // Free the G^R cache (not needed by the post-loop HF/IBC epilogue)
  opt_dG_tsRPQ.reset();
  opt_dG_mtau_tsRPQ.reset();

  // Report results
  if (converged) {
    app_log(1, "\n  LR SCF converged in {} iterations!", iter);
  } else if (max_iter > 1) {
    app_log(1, "\n  [WARNING] LR SCF did NOT converge after {} iterations.", max_iter);
  }
  app_log(1, "  Final Δμ = {:.6e}", Delta_mu);

  // Final hierarchical timer report (printed once, at verbosity >= 2).
  // Per-step solver prints inside the loop are gated to verbosity >= 3.
  print_timers(lr_pi_solver.get(), lr_scr_solver.get(), lr_gw_solver.get());

  return std::make_tuple(iter, Delta_mu);
}


void lr_driver::print_memory_estimate(long NP, bool include_gw_sigma, bool gw_full) {
  // Dimensions of the large arrays.
  const long nt   = _nts;                          // # imaginary-time points (full grid)
  const long nw   = _dyson.FT()->nw_f();           // # fermionic Matsubara frequencies (G(iω))
  const long nwb  = _dyson.FT()->nw_b();           // # bosonic Matsubara frequencies (W(iω))
  const long nwbh = (nwb % 2 == 0) ? nwb / 2 : nwb / 2 + 1;      // half bosonic ω-grid (W_full)
  const long nth  = (_nts % 2 == 0) ? _nts / 2 : _nts / 2 + 1;   // half τ-grid (W/Π/G^R)
  const long ns   = _ns;
  const long nki  = _nkpts_ibz;                    // IBZ k-points (shared band-basis arrays)
  const long nq   = _nkpts;                        // full-BZ q/R points (distributed aux arrays)
  const long nb   = _nbnd;
  const int  n_nodes = std::max(1, _mpi->internode_comm.size());

  const double bytes_per = static_cast<double>(sizeof(ComplexType));
  const double to_GB = 1.0 / (1024.0 * 1024.0 * 1024.0);

  // {name, shape string, # elements, is_distributed}
  struct entry_t { std::string name; std::string shape; double nelem; bool dist; };
  std::vector<entry_t> arrays;

  auto band5 = [&](long n0) { return double(n0) * ns * nki * nb * nb; };  // (n0,ns,nk_ibz,nb,nb)
  auto aux4  = [&](long n0) { return double(n0) * nq * NP * NP; };        // (n0,nq,NP,NP)
  auto aux5  = [&](long n0) { return double(n0) * ns * nq * NP * NP; };   // (n0,ns,nq,NP,NP)

  auto shp5b = [&](long n0) { return fmt::format("({},{},{},{},{})", n0, ns, nki, nb, nb); };
  auto shp4a = [&](long n0) { return fmt::format("({},{},{},{})", n0, nq, NP, NP); };
  auto shp5a = [&](long n0) { return fmt::format("({},{},{},{},{})", n0, ns, nq, NP, NP); };

  // --- Shared (replicated per node), band basis ~ nk·nt·nb² ---
  // sG_tskij is caller-owned but resident throughout run_lr, so count it here.
  arrays.push_back({"sG_tskij",       shp5b(nt), band5(nt), false});
  arrays.push_back({"sDeltaG_tskij",  shp5b(nt), band5(nt), false});
  arrays.push_back({"sG_wskij",       shp5b(nw), band5(nw), false});
  if (include_gw_sigma) {
    arrays.push_back({"sDeltaSigma_tskij",      shp5b(nt), band5(nt), false});
    arrays.push_back({"sDeltaSigma_prev_tskij", shp5b(nt), band5(nt), false});
  }

  // --- Distributed (over global comm), aux basis ~ nk·nt·NP² ---
  if (include_gw_sigma) {
    arrays.push_back({"dW_tRPQ",       shp4a(nth), aux4(nth), true});
  }
  if (gw_full) {
    arrays.push_back({"dW_full_wqPQ",  shp4a(nwbh), aux4(nwbh), true});
    arrays.push_back({"dG_tsRPQ",      shp5a(nth),  aux5(nth),  true});
    arrays.push_back({"dG_mtau_tsRPQ", shp5a(nth),  aux5(nth),  true});
    // Solver-owned FT staging buffers: allocated once in compute_W_full_omega,
    // resident for the whole loop (persistent by lifetime, though internal).
    arrays.push_back({"_ft_buffer_t",  shp4a(nth),  aux4(nth),  true});
    arrays.push_back({"_ft_buffer_w",  shp4a(nwbh), aux4(nwbh), true});
    if (!is_q_gamma())
      arrays.push_back({"_dW_full_qpQ (W(q+Q))", shp4a(nwbh), aux4(nwbh), true});
  }

  auto gb = [&](double nelem) { return nelem * bytes_per * to_GB; };

  // Persistent totals.
  double shared_GB = 0.0, dist_GB = 0.0;
  for (auto const& a : arrays) {
    if (a.dist) dist_GB += gb(a.nelem); else shared_GB += gb(a.nelem);
  }
  double dist_per_node_GB = dist_GB / n_nodes;
  double total_per_node_GB = shared_GB + dist_per_node_GB;

  // --- Per-iteration transients: scratch arrays (~ nk·nt·n²) allocated and
  //     freed within one SCF iteration, on top of the persistent set.
  //     Two mutually-exclusive phases:
  //       dyson : ΔG(iω)/ΔΣ(iω) inside lr_dyson (distributed band basis)
  //       gwsig : ΔΠ/ΔW(τ) + its (t,q)→(q,t) transpose scratch (gw_full only)
  //     lr_dyson runs before the Π/W/Σ steps and frees its scratch first, so the
  //     two never coexist — the peak adds only the larger: max(dyson, gwsig).
  enum phase_t { DYSON, GWSIG };
  struct tentry_t { std::string name; std::string shape; double nelem; phase_t ph; };
  std::vector<tentry_t> trans;

  // lr_dyson ω-side band scratch: ΔG(iω) + (ΔΣ(iω) when GW active). Distributed.
  trans.push_back({include_gw_sigma ? "ΔG(iω)+ΔΣ(iω) (lr_dyson)" : "ΔG(iω) (lr_dyson)",
                   fmt::format("{}x({},{},{},{},{})", include_gw_sigma ? 2 : 1, nw, ns, nki, nb, nb),
                   (include_gw_sigma ? 2.0 : 1.0) * band5(nw), DYSON});
  if (gw_full) {
    trans.push_back({"ΔΠ/ΔW(τ)",          shp4a(nth), aux4(nth), GWSIG});
    trans.push_back({"ΔW transpose copy", shp4a(nth), aux4(nth), GWSIG});
  }

  double t_dyson = 0.0, t_gwsig = 0.0;
  for (auto const& t : trans)
    (t.ph == DYSON ? t_dyson : t_gwsig) += t.nelem;
  bool dyson_dominates = t_dyson >= t_gwsig;
  double peak_trans_nelem = std::max(t_dyson, t_gwsig);
  // All transients here are distributed (band ΔG(iω) included).
  double peak_dist_GB = dist_GB + gb(peak_trans_nelem);
  double peak_per_node_GB = shared_GB + peak_dist_GB / n_nodes;

  // Level-2 persistent breakdown (printed before the level-1 totals).
  app_log(2, "\n  LR memory estimate (persistent arrays ~ nk·nt·n², n ∈ {{nbnd={}, NP={}}})", nb, NP);
  app_log(2, "  {}", std::string(72, '-'));
  app_log(2, "    {:<26s}{:<26s}{:>8s}   {}", "quantity", "shape", "GB", "location");
  for (auto const& a : arrays)
    app_log(2, "    {:<26s}{:<26s}{:>8.3f}   {}",
            a.name, a.shape, gb(a.nelem), a.dist ? "distributed" : "shared");
  app_log(2, "  {}", std::string(72, '-'));
  app_log(2, "    shared total:        {:8.3f} GB/node  (replicated on each of {} node(s))",
          shared_GB, n_nodes);
  app_log(2, "    distributed total:   {:8.3f} GB        (÷ {} node(s) = {:.3f} GB/node)",
          dist_GB, n_nodes, dist_per_node_GB);

  // Level-2 per-iteration transient breakdown.
  app_log(2, "\n  LR memory per-iteration transients (allocated/freed within an iteration):");
  app_log(2, "  {}", std::string(72, '-'));
  app_log(2, "    {:<26s}{:<26s}{:>8s}   {}", "quantity", "shape", "GB", "phase");
  for (auto const& t : trans)
    app_log(2, "    {:<26s}{:<26s}{:>8.3f}   {}", t.name, t.shape, gb(t.nelem),
            (t.ph == DYSON) ? "Dyson" : "ΔW/Σ");
  app_log(2, "  {}", std::string(72, '-'));
  app_log(2, "    peak transient:      {:8.3f} GB        = max(Dyson {:.3f}, ΔW/Σ {:.3f}) [{} dominates]",
          gb(peak_trans_nelem), gb(t_dyson), gb(t_gwsig),
          dyson_dominates ? "Dyson" : "ΔW/Σ");
  app_log(2, "    (distributed; ÷ {} node(s) = {:.3f} GB/node added)",
          n_nodes, gb(peak_trans_nelem) / n_nodes);

  app_log(1, "  Estimated LR memory (persistent): {:.3f} GB/node", total_per_node_GB);
  app_log(1, "  Estimated LR memory (peak):       {:.3f} GB/node", peak_per_node_GB);
  app_log(2, "");
}


void lr_driver::print_distribution_summary(long NP, bool include_gw_sigma, bool gw_full) {
  const long nproc = _mpi->comm.size();
  const long nw   = _dyson.FT()->nw_f();
  const long nwb  = _dyson.FT()->nw_b();
  const long nwbh = (nwb % 2 == 0) ? nwb / 2 : nwb / 2 + 1;
  const long nth  = (_nts % 2 == 0) ? _nts / 2 : _nts / 2 + 1;
  const long nq   = _nkpts;
  const long nki  = _nkpts_ibz;

  // Aux τ-dist (q-local) grid — the same helper the allocators use.
  auto [tau_pg, tau_bs] = utils::lr_W_q_local_dist(nproc, nth, NP);
  (void)tau_bs;

  auto pg4 = [](const std::array<long,4>& g, const char* ax) {
    return fmt::format("{}=({},{},{},{})", ax, g[0], g[1], g[2], g[3]); };
  auto pg5 = [](const std::array<long,5>& g, const char* ax) {
    return fmt::format("{}=({},{},{},{},{})", ax, g[0], g[1], g[2], g[3], g[4]); };

  app_log(2, "\n  LR distribution patterns (nproc = {}):", nproc);
  app_log(2, "  {}", std::string(72, '-'));
  app_log(2, "    {:<22s}{:<30s}{}", "pattern", "pgrid", "arrays");

  // Aux τ-dist (q-local) — present whenever a W self-energy is active.
  if (include_gw_sigma) {
    const char* arrs = gw_full ? "dW_tRPQ, dG_tsRPQ, dG_mtau, ΔΠ/ΔW" : "dW_tRPQ";
    app_log(2, "    {:<22s}{:<30s}{}", "aux τ-dist (q-local)",
            pg4(tau_pg, "(t,q,P,Q)"), arrs);
  }
  // Aux FT-buffer + ω-side — only the full-GW W Dyson pipeline (mirrors the
  // distribution choice in lr_scr_coulomb_t::compute_W_full_omega).
  if (gw_full) {
    auto [ftb_pg, ftb_bs] =
        solvers::scr_coulomb_fourier_t::ft_buffer_dist(nproc, {nth, nq, NP, NP});
    (void)ftb_bs;
    std::array<long,4> w_pg, w_bs;
    if (ftb_pg[2] == 1 && ftb_pg[3] == 1) {
      w_pg = ftb_pg; w_bs = ftb_bs;
    } else {
      std::tie(w_pg, w_bs) = utils::lr_W_proc_grid(nproc, nq, nwbh, NP);
    }
    (void)w_bs;
    app_log(2, "    {:<22s}{:<30s}{}", "aux FT-buffer",
            pg4(ftb_pg, "(·,q,P,Q)"), "_ft_buffer_t, _ft_buffer_w");
    app_log(2, "    {:<22s}{:<30s}{}", "aux ω-side",
            pg4(w_pg, "(w,q,P,Q)"),
            is_q_gamma() ? "dW_full_wqPQ" : "dW_full_wqPQ, _dW_full_qpQ");
  }

  // Band-basis Dyson grids — mirror the inline proc-grid math in
  // lr_dyson::solve_lr_dyson (ω-side and the τ redistribute target).
  std::array<long,5> dyw_pg;
  {
    long np = nproc;
    long nwpools = utils::find_proc_grid_max_npools(np, nw, 0.4);
    np /= nwpools;
    long nkpools = utils::find_proc_grid_max_npools(np, nki, 0.4);
    np /= nkpools;
    long np_i = utils::find_proc_grid_min_diff(np, 1, 1);
    long np_j = np / np_i;
    dyw_pg = {nwpools, 1, nkpools, np_i, np_j};
  }
  std::array<long,5> dyt_pg;
  {
    long np = nproc;
    long nkpools = utils::find_proc_grid_max_npools(np, nki, 0.2);
    np /= nkpools;
    long np_i = utils::find_proc_grid_min_diff(np, 1, 1);
    long np_j = np / np_i;
    dyt_pg = {1, 1, nkpools, np_i, np_j};
  }
  app_log(2, "    {:<22s}{:<30s}{}", "band Dyson(ω)",
          pg5(dyw_pg, "(w,s,k,i,j)"),
          include_gw_sigma ? "ΔG(iω), ΔΣ(iω)" : "ΔG(iω)");
  app_log(2, "    {:<22s}{:<30s}{}", "band Dyson(τ)",
          pg5(dyt_pg, "(·,·,k,i,j)"), "ΔG(τ) (+ redistribute tmp)");
  app_log(2, "  {}", std::string(72, '-'));
  app_log(2, "    (aux & band arrays are distributed over comm; "
             "shared arrays are node-replicated)\n");
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


void lr_driver::print_timers(solvers::lr_rpa_pi* pi_solver,
                             solvers::lr_scr_coulomb_t* scr_solver,
                             solvers::lr_gw* gw_solver) {
  // Driver totals in execution order, each followed by the corresponding
  // solver's subclocks (deeper indent). Solver pointers may be null when the
  // step was not active; subclocks are then skipped.
  const std::string sub_indent = "        ";
  app_log(2, "\n  LR_DRIVER timers");
  app_log(2, "  -----------------");
  app_log(2, "    Total LR SCF:               {0:8.3f} sec  {1:4d} calls", _Timer.elapsed("LR_SCF"), _Timer.number_of_calls("LR_SCF"));
  app_log(2, "      - LR Driver Setup:        {0:8.3f} sec  {1:4d} calls", _Timer.elapsed("LR_DRIVER_SETUP"), _Timer.number_of_calls("LR_DRIVER_SETUP"));
  app_log(2, "      - LR Dyson (total):       {0:8.3f} sec  {1:4d} calls", _Timer.elapsed("LR_DYSON"), _Timer.number_of_calls("LR_DYSON"));
  _lr_dyson.print_subclocks(2, sub_indent);
  app_log(2, "      - LR HF (total):          {0:8.3f} sec  {1:4d} calls", _Timer.elapsed("LR_HF"), _Timer.number_of_calls("LR_HF"));
  if (_lr_hf) _lr_hf->print_subclocks(2, sub_indent);
  app_log(2, "      - LR GW Pi (total):       {0:8.3f} sec  {1:4d} calls", _Timer.elapsed("LR_GW_PI"), _Timer.number_of_calls("LR_GW_PI"));
  if (pi_solver) pi_solver->print_subclocks(2, sub_indent);
  app_log(2, "      - LR GW W (total):        {0:8.3f} sec  {1:4d} calls", _Timer.elapsed("LR_GW_W"), _Timer.number_of_calls("LR_GW_W"));
  if (scr_solver) scr_solver->print_subclocks(2, sub_indent);
  app_log(2, "      - LR GW Sigma (total):    {0:8.3f} sec  {1:4d} calls", _Timer.elapsed("LR_GW_SIGMA"), _Timer.number_of_calls("LR_GW_SIGMA"));
  app_log(2, "          - ΔW transpose (t,q)->(q,t):  {0:8.3f} sec  {1:4d} calls", _Timer.elapsed("LR_GW_DW_TRANSPOSE"), _Timer.number_of_calls("LR_GW_DW_TRANSPOSE"));
  if (gw_solver) gw_solver->print_subclocks(2, sub_indent);
  app_log(2, "      - LR GW Sig T2 (total):   {0:8.3f} sec  {1:4d} calls", _Timer.elapsed("LR_GW_SIGMA_TERM2"), _Timer.number_of_calls("LR_GW_SIGMA_TERM2"));
  app_log(2, "      - LR Iter Alg (total):    {0:8.3f} sec  {1:4d} calls", _Timer.elapsed("LR_ITER_ALG"), _Timer.number_of_calls("LR_ITER_ALG"));
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
    const nda::array<ComplexType, 4>*, bool, std::string,
    const nda::array_view<ComplexType, 3>*,
    sArray_t<Array_view_5D_t>*);

} // namespace methods
