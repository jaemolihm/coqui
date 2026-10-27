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


#include <functional>
#include <limits>

#include "methods/SCF/lr_driver.hpp"
#include "methods/embedding/local_projection.hpp"
#include "methods/SCF/lr_precompute.hpp"
#include "methods/SCF/scf_common.hpp"
#include "methods/ERI/thc_reader_t.hpp"
#include "methods/scr_coulomb/scr_coulomb_fourier_t.h"
#include "numerics/distributed_array/nda.hpp"
#include "utilities/check.hpp"
#include "utilities/lr_utils.hpp"
#include "utilities/proc_meminfo.hpp"
#include "methods/GW/g0_div_utils.hpp"
#include "nda/nda.hpp"

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
                  "LR_DRIVER_SETUP_W_FULL", "LR_DRIVER_SETUP_W_TRPQ", "LR_DRIVER_SETUP_WC0",
                  "LR_DRIVER_SETUP_G_OMEGA", "LR_DRIVER_SETUP_G_R",
                  "LR_DRIVER_SETUP_DELTAMU", "LR_DRIVER_SETUP_ALLOC", "LR_DRIVER_SETUP_IBC", "LR_RESET",
                  "LR_DYSON", "LR_HF", "LR_GW_SIGMA",
                   "LR_GW_PI", "LR_GW_W", "LR_QPGW_STATIC",
                   "LR_ITER_ALG", "LR_PRECOND", "LR_SAVE", "LR_CONVERGENCE"}) {
    _Timer.add(v);
  }
  _mpi->comm.barrier();
}


std::string lr_kernel_spec::to_string() const {
  std::string s;
  auto add = [&](const char* c) { if (!s.empty()) s += ", "; s += c; };
  if (hartree)    add("H");
  if (exchange)   add("X");
  if (sigma_dG_W) add("dGW");
  if (sigma_G_dW) add("GdW");
  return s.empty() ? std::string("-") : s;
}


lr_kernel_spec kernel_spec_from_method(std::string const& name) {
  if (name == "none")    return lr_kernel_spec{};
  if (name == "Hartree") return lr_kernel_spec{true, false, false, false};
  if (name == "HF")      return lr_kernel_spec{true, true,  false, false};
  if (name == "GW0")     return lr_kernel_spec{true, true,  true,  false};
  if (name == "GW")      return lr_kernel_spec{true, true,  true,  true};
  utils::check(false,
               "kernel_spec_from_method: unknown LR method '{}'. Must be one of "
               "'none', 'Hartree', 'HF', 'GW0', 'GW'.", name);
  return lr_kernel_spec{};
}


// Distribution flow through the LR-GW pipeline
// Two distribution patterns:
//   τ-dist (q-local):  pgrid = {tpools, 1, np_P, np_Q}  — q undivided
//   q-dist (τ-local):  pgrid = {1, nqpools, np_P, np_Q}  — tau/omega undivided
//
// Everything on the ω axis lives on q-dist as well
// (utils::lr_W_tau_local_dist returns it), which is what lets both
// Fourier transforms write straight into their output with no staging buffer.
//
//   W_c in:              (w,q,P,Q), q-dist (utils::lr_W_tau_local_dist)
//   lr_setup_W:
//     w_to_tau:          q-dist → (t,q,P,Q) straight onto the τ-dist tiling
//     lr_Wc_to_Wfull:    + Z(q) in place on the ω copy → W_full(iω) [cached]
//     lr_precompute_W_tRPQ: q→R in place on the τ copy → (t,R,P,Q), τ-dist [cached]
//
//   evaluate_lr_Pi:      → (t,q,P,Q), τ-dist
//   solve_lr_dyson_W (in-place):
//     tau_to_w:          τ-dist → q-dist
//     lr_dyson_W_in_place: q-dist; SLATE GEMM batched over (iw, iq) on the
//                          (P, Q) subgrid.
//                          For Q≠Γ, gathers W_full(kpq_map(iq)) via Alltoallv on q_pool_comm.
//     w_to_tau:          q-dist → τ-dist
//     output:            τ-dist (overwrites input)
//   evaluate_sigma_*:    τ-dist in (t,q) order, i.e. ΔW is consumed as produced

/**
 * Build the two cached W operands the ΔΣ/ΔW pipeline reads every iteration, from
 * the single W_c(iω) the caller hands in:
 *
 *   dW_full_wqPQ = W_c(iω) + Z(q), the ΔW Dyson operand (gw_full only)
 *   dW_tRPQ      = W_c(t,R,P,Q),   the ΔΣ = −ΔG⊙W_c operand
 *
 * Ordering is load-bearing: dW_tRPQ carries the *correlation-only* W_c, so Z is
 * added to the ω array only once the τ copy has been taken.
 *
 * Out-params rather than a return value because the two operands are owned by
 * the caller's scope, and each is absent on the paths that do not need it.
 */
void lr_driver::lr_setup_W(dArr_4D_t* dW_wqPQ_in, thc_reader_t& thc, bool gw_full,
                           solvers::lr_scr_coulomb_t* lr_scr,
                           std::optional<dArr_4D_t>& opt_dW_full_wqPQ,
                           std::optional<dArr_4D_t>& opt_dW_tRPQ) {
  // The canonical LR q-local (t,q,P,Q) tiling every τ-side operand shares, and
  // the target of the ω→τ transform below. The fused ΔΣ loop pairs the P/Q tile
  // of W_c (from dW_tqPQ → dW_tRPQ) with that of ΔW and the G^R cache, both
  // built via lr_W_q_local_dist, so a W_c on a different tiling makes the fused
  // pairing abort: the contiguous PQ split disagrees whenever Np % np_P != 0
  // (bsize=1 vs bsize=Np/np_P round differently).
  long nt_half = (_nts % 2 == 0) ? _nts / 2 : _nts / 2 + 1;
  auto [tq_pgrid, tq_bsize] =
      utils::lr_W_q_local_dist(_mpi->comm.size(), nt_half, thc.Np());

  _Timer.start("LR_DRIVER_SETUP_W_FULL");
  // One ω→τ, landing directly on the LR τ tiling. The ω array survives it only
  // when the ΔW Dyson needs it as W_full; otherwise the FT releases it before
  // allocating the output, which a caller-side reset cannot do.
  solvers::scr_coulomb_fourier_t setup_ft(_dyson.FT());
  auto dW_tqPQ = setup_ft.w_to_tau(*dW_wqPQ_in, tq_pgrid, tq_bsize,
                                   /*reset_input=*/!gw_full,
                                   __app_verbosity__ >= 3);
  if (gw_full) {
    lr_scr->lr_Wc_to_Wfull(*dW_wqPQ_in, thc);
    opt_dW_full_wqPQ.emplace(std::move(*dW_wqPQ_in));
  }
  _Timer.stop("LR_DRIVER_SETUP_W_FULL");

  _Timer.start("LR_DRIVER_SETUP_W_TRPQ");
  opt_dW_tRPQ.emplace(lr_precompute_W_tRPQ(dW_tqPQ, thc));
  _Timer.stop("LR_DRIVER_SETUP_W_TRPQ");
}

void lr_driver::lr_setup(
    const sArray_t<Array_view_5D_t>& sG_tskij,
    thc_reader_t& thc,
    dArr_4D_t* dW_wqPQ_in,
    const lr_params& p) {

  const lr_kernel_spec& K = p.kernel;
  const bool qp = (p.qp_static != nullptr);

  // The double counting of a GW+EDMFT ground state needs the plain unsplit ΔΣ (run_lr_calc has already rejected
  // a kernel without Σ).
  if (p.embedding && p.embedding->dc) {
    utils::check(!qp && !p.hessian() && !p.split_sigma_terms,
                 "lr_driver::lr_setup: the double counting of a GW+EDMFT ground state is incompatible with "
                 "qp_static mode, the hessian and split_sigma_terms.");
  }

  // GCR's basis stays valid only while the map is one fixed affine operator. qp
  // mode statifies ΔΣ through a Padé continuation, which is not additive in ΔΣ,
  // so neither w = A z nor the recycled pairs would hold.
  utils::check(p.iter_params.alg != "GCR" || !qp,
               "lr_driver::lr_setup: iter_alg 'GCR' is incompatible with qp_static "
               "mode: the Padé statification of ΔΣ is not additive, so the map "
               "is not affine.");

  // apply_kernel evaluates any Σ on the operator's _lr_gw.
  utils::check(!p.precond_kernel.has_sigma(),
               "lr_driver::lr_setup: the GCR preconditioner kernel ({}) must be "
               "Σ-free.", p.precond_kernel.to_string());

  // Every output downstream, ΔG(τ) included, is produced by the Dyson pass after
  // the loop, so zero iterations return the start of the iteration: that is the
  // preconditioner solution (initial_guess_precond), not the zero iterate.
  const int min_iter = p.initial_guess_precond ? 0 : 1;
  utils::check(p.max_iter >= min_iter,
               "lr_driver::lr_setup: max_iter must be >= {}, got {}.", min_iter, p.max_iter);
  if (p.split_sigma_terms) {
    utils::check(p.max_iter == 1,
                 "lr_driver::lr_setup: split ΔΣ terms are only meaningful for a "
                 "one-shot solve (max_iter=1), got max_iter={}.", p.max_iter);
    // The one step would scale the total ΔΣ by the mixing but not the stored
    // G·ΔW term.
    utils::check(p.iter_params.mixing >= 1.0,
                 "lr_driver::lr_setup: split ΔΣ terms need mixing >= 1 (undamped), got {}.",
                 p.iter_params.mixing);
    utils::check(!p.has_deltax(),
                 "lr_driver::lr_setup: split ΔΣ terms do not support the DeltaX IBC "
                 "correction (term 2 has no IBC path).");
  }
  if (p.exchange_static_W) {
    // The IBC ΔX correction is built against the bare V_HF in the aux basis
    // (build_lr_ibc's exchange branch); pairing it with a screened exchange
    // kernel would mix two different interactions in one ΔF.
    utils::check(!p.has_deltax(),
                 "lr_driver::lr_setup: exchange_static_W is incompatible with the "
                 "DeltaX (IBC) correction, which is built from the bare-exchange "
                 "V_HF.");
    utils::check(p.DeltaV_qPQ == nullptr,
                 "lr_driver::lr_setup: exchange_static_W is incompatible with the "
                 "DeltaV_qPQ perturbation, whose exchange term contracts the bare "
                 "δV against the unperturbed Dm.");
  }
  utils::check(!_setup_done,
               "lr_driver::lr_setup: called twice on the same driver. One driver "
               "serves one q-vector and one unperturbed state; construct a new "
               "one to change either.");

  app_log(1, "Starting Linear Response SCF loop:");
  app_log(1, "  max_iter = {}", p.max_iter);
  app_log(1, "  tol = {:.2e}", p.tol);
  app_log(1, "  fix_density = {}", p.fix_density ? "true" : "false");
  app_log(1, "  kernel = {}", K.to_string());
  app_log(1, "  include_xc = {}", p.include_xc ? "true" : "false");
  if (p.exchange_static_W)
    app_log(1, "  exchange kernel = V + W_c(iν=0)  [static screened exchange]");
  if (!p.precond_kernel.empty())
    app_log(1, "  preconditioner kernel = {}{}  (<= {} sweeps to {:.0e} relative)",
            p.precond_kernel.to_string(),
            p.precond_hsex ? " with X = V + W_c(iν=0)" : "",
            p.precond_max_iter, p.precond_tol);
  if (p.initial_guess_precond)
    app_log(1, "  initial guess = preconditioner solution of ΔH0");
  if (p.fixed_order)
    app_log(1, "  fixed order: exactly max_iter iterations, no convergence test");
  app_log(1, "  qp_static_sigma = {}", qp ? "true" : "false");
  app_log(1, "  iter_alg = {}", p.iter_params.alg);
  app_log(1, "  mixing = {:.2f}", p.iter_params.mixing);
  if (p.iter_params.alg == "DIIS") {
    app_log(1, "  max_subsp_size = {}", p.iter_params.max_subsp_size);
    app_log(1, "  diis_warmup = {}", p.iter_params.diis_warmup);
  }
  // ΔΣ-sized striped previous iterates: the tracked ΔΣ.
  const long n_sigma_prev = (K.has_sigma() && !qp) ? 1 : 0;

  // The accelerator (DIIS or GCR), built once: lr_solve_one reset()s it rather
  // than rebuilding it, so the (job-wide, striped) history is allocated exactly
  // once no matter how many perturbations follow. Built here, ahead of the memory
  // report below, which reads its shape off the object. Damping and undamped
  // Picard keep no history: lr_solve_one mixes them elementwise. Neither
  // accelerator is built for an empty kernel, whose solve runs no iteration.
  if (!K.empty()) {
    if (p.iter_params.alg == "GCR") {
      // Capacity 0 = default: a ΔF-only pair is tiny, so keep many; with a Σ
      // every pair costs a ΔΣ, as one DIIS depth unit does.
      size_t cap = p.iter_params.max_subsp_size;
      if (cap == 0) cap = (K.has_sigma() && !qp) ? 10 : 50;
      _lr_gcr = std::make_unique<lr_gcr>(cap);
    } else if (p.iter_params.alg == "DIIS") {
      _lr_diis = std::make_unique<lr_diis>(p.iter_params);
    }
  }
  // The preconditioner's inner GCR iterates ΔF alone, so its pairs are tiny.
  if (!p.precond_kernel.empty()) _pc_gcr = std::make_unique<lr_gcr>(50);

  // Accelerator history, for the memory report.
  lr_diis_hist_t inner_hist;
  if (_lr_gcr) {
    // 2M basis vectors plus x, r and the per-step g, y (and M⁻¹r when
    // preconditioned).
    inner_hist.depth = static_cast<long>(_lr_gcr->capacity());
    inner_hist.n_vec = 2 * inner_hist.depth + 4 + (p.precond_kernel.empty() ? 0 : 1);
    inner_hist.n_F = 1;
    inner_hist.n_Sigma = (K.has_sigma() && !qp) ? 1 : 0;
  } else if (_lr_diis) {
    // The ring holds max_subsp_size + 1 slots: push_slot() grows to that before
    // the first eviction. Each slot stores a trial AND a residual.
    inner_hist.depth = static_cast<long>(_lr_diis->max_subsp_size());
    inner_hist.n_vec = 2 * (inner_hist.depth + 1);
    inner_hist.n_F = qp ? 2 : 1;   // ΔF (+ the static ΔV_QPGW in qp mode)
    inner_hist.n_Sigma = (K.has_sigma() && !qp) ? 1 : 0;
  }

  // Estimate the persistent large-array memory footprint for this path, then
  // summarize the MPI distribution patterns the large arrays use.
  print_memory_estimate(thc.Np(), K.has_sigma(), K.sigma_G_dW, n_sigma_prev,
                        inner_hist, !p.precond_kernel.empty(),
                        p.fix_density && _lr_dyson.is_q_gamma(),
                        p.exchange_static_W || p.precond_hsex,
                        p.fixed_order ? 0 : p.hessian_nmodes);
  print_distribution_summary(thc.Np(), K.has_sigma(), K.sigma_G_dW);

  _Timer.start("LR_DRIVER_SETUP");

  // Solvers. Each latches the perturbation q at construction and caches a
  // workspace, so they are built once here and reused by every lr_solve_one.
  if (K.has_F()) {
    _lr_hf = std::make_unique<solvers::lr_hf>(_mpi, _MF, _lr_dyson.q_vec(), p.hf_div_treatment);
  }
  // The preconditioner gets its own lr_hf: lr_hf caches U(R) keyed on the
  // exchange kernel it was first built with, and an HSEX preconditioner contracts
  // V + W_c(0) where the operator contracts V.
  if (p.precond_kernel.has_F()) {
    _lr_hf_pc = std::make_unique<solvers::lr_hf>(_mpi, _MF, _lr_dyson.q_vec(), p.hf_div_treatment);
  }
  if (K.has_sigma()) {
    _lr_gw = std::make_unique<solvers::lr_gw>(_lr_dyson.q_vec(), p.div_treatment);
  }
  // Split-term mode (one-shot G0W0) needs a second lr_gw for term 2: each solver
  // caches a workspace keyed on its (term1,term2) usage.
  if (p.split_sigma_terms) {
    _lr_gw2 = std::make_unique<solvers::lr_gw>(_lr_dyson.q_vec(), p.div_treatment);
  }
  if (K.sigma_G_dW) {
    _lr_pi = std::make_unique<solvers::lr_rpa_pi>(_lr_dyson.q_vec());
    _lr_scr = std::make_unique<solvers::lr_scr_coulomb_t>(_dyson.FT(), _lr_dyson.q_vec());
  }

  // HSEX kernel, taken BEFORE lr_setup_W: that call consumes dW_wqPQ_in (it is
  // either moved into W_full or released by the ω→τ transform), and it also adds
  // Z(q) to the ω array in place, which would double-count the bare V here.
  if (p.exchange_static_W || p.precond_hsex) {
    _Timer.start("LR_DRIVER_SETUP_WC0");
    _opt_dWc0_qPQ.emplace(lr_precompute_Wc_static(*dW_wqPQ_in));
    // eps_inv_head_w holds ε⁻¹ - 1, q→0 extrapolated, on the PH-symmetric ω
    // half-axis whose index 0 is ν = 0.
    _hsex_head_factor = 1.0 + (*p.eps_inv_head_w)(0).real();
    app_log(1, "  HSEX ε⁻¹(iν=0) = {:.6f}", _hsex_head_factor);
    _Timer.stop("LR_DRIVER_SETUP_WC0");
  }

  if (K.has_sigma()) {
    lr_setup_W(dW_wqPQ_in, thc, K.sigma_G_dW, _lr_scr.get(),
               _opt_dW_full_wqPQ, _opt_dW_tRPQ);
  } else if (dW_wqPQ_in != nullptr) {
    // Nothing below reads the ω-axis W; whatever wanted it above has taken its
    // slice (today that is only HSEX's ν=0 one). Keyed on the array still being
    // there rather than on which option asked for it, so a future borrower needs
    // no change here. The caller keeps it alive to the end of the call, so
    // release it rather than carry the whole ω W through the SCF loop.
    dW_wqPQ_in->reset();
  }

  // Precompute the unperturbed G^R(τ)/G^R(β−τ) pair in aux basis (constant
  // across SCF iterations and across perturbations; consumed by evaluate_lr_Pi
  // and Σ term 2).
  if (K.sigma_G_dW) {
    _Timer.start("LR_DRIVER_SETUP_G_R");
    utils::memlog("lr_driver::lr_setup: before G^R pair precompute");
    auto [dG_tsRPQ, dG_mtau_tsRPQ] = lr_precompute_G_R_pair(sG_tskij.local(), thc);
    _opt_dG_tsRPQ.emplace(std::move(dG_tsRPQ));
    _opt_dG_mtau_tsRPQ.emplace(std::move(dG_mtau_tsRPQ));
    utils::memlog("lr_driver::lr_setup: after G^R pair precompute");
    _Timer.stop("LR_DRIVER_SETUP_G_R");
  }

  // Precompute G(iω) in shared memory and pass to lr_dyson (avoids redundant FT
  // per iteration). Held as a member: lr_dyson keys its dN/dμ cache on this
  // array's address, so a per-call local would invalidate it every solve.
  utils::memlog("lr_driver::lr_setup: before sG_wskij precompute");
  _Timer.start("LR_DRIVER_SETUP_G_OMEGA");
  _sG_wskij.emplace(lr_precompute_G_omega(*_mpi, sG_tskij, *_dyson.FT()));
  _lr_dyson.set_cached_G_omega(&(*_sG_wskij));
  _Timer.stop("LR_DRIVER_SETUP_G_OMEGA");
  utils::memlog("lr_driver::lr_setup: after sG_wskij precompute");

  // Precompute the Δμ response if needed for fix_density mode at q=0
  if (p.fix_density && _lr_dyson.is_q_gamma()) {
    _Timer.start("LR_DRIVER_SETUP_DELTAMU");
    _lr_dyson.build_dmu_response();
    _Timer.stop("LR_DRIVER_SETUP_DELTAMU");
  }

  _Timer.start("LR_DRIVER_SETUP_ALLOC");
  // Element partition of the node-replicated band arrays over the *global*
  // comm: the DIIS history, the "previous iterate" copies and their norms are
  // all elementwise, so each rank handles one slice and the whole job stores
  // each of those quantities once instead of once per node.
  _pmap = utils::make_part_map(*_mpi);

  // Previous ΔDm / ΔF / ΔΣ / ΔV_QPGW (for damping/DIIS and the difference norms)
  // are rank-private slices of the flattened arrays, in the `_pmap` partition. ΔΣ
  // is tracked only when the kernel carries a Σ and we are not in qp mode (qp mode
  // tracks the static ΔV_QPGW in its place). An inactive quantity gets
  // n_flat = 0.
  const long nF = _ns * _nkpts_ibz * _nbnd * _nbnd;
  _DeltaDm.alloc(_pmap, nF);
  _DeltaF.alloc(_pmap, nF);
  _DeltaSigma.alloc(_pmap, (K.has_sigma() && !qp) ? _nts * nF : 0);
  _DeltaVcorr.alloc(_pmap, qp ? nF : 0);

  if (!K.has_sigma()) {
    // ΔDm-only Dyson pass (Σ-free kernel)
    _sDeltaF_dyson_in.emplace(math::shm::make_shared_array<Array_view_4D_t>(
        *_mpi, {_ns, _nkpts_ibz, _nbnd, _nbnd}));
  }
  if (!p.precond_kernel.empty()) {
    // Preconditioner: the gathered residual, the ΔDm of its ΔΣ part, and the
    // inner iterate with its ΔDm.
    auto make_F = [&]() { return math::shm::make_shared_array<Array_view_4D_t>(
        *_mpi, {_ns, _nkpts_ibz, _nbnd, _nbnd}); };
    _sDeltaF_pc_r.emplace(make_F());
    _sDeltaF_pc_y.emplace(make_F());
    _sDeltaDm_pc.emplace(make_F());
    _sDeltaDm_pc_y.emplace(make_F());
    _sDeltaSigma_r.emplace(math::shm::make_shared_array<Array_view_5D_t>(
        *_mpi, {_nts, _ns, _nkpts_ibz, _nbnd, _nbnd}));
  }
  _Timer.stop("LR_DRIVER_SETUP_ALLOC");

  // DeltaX IBC correction setup
  if (p.has_deltax()) {
    _Timer.start("LR_DRIVER_SETUP_IBC");
    app_log(2, "  DeltaX IBC correction: building lr_ibc_DeltaX...");

    utils::memlog("lr_driver::lr_setup: before build_lr_ibc");
    _opt_ibc.emplace(build_lr_ibc(
        *_mpi, _MF, thc,
        p.sDeltaX_left->local(), p.sDeltaX_right->local(),
        _lr_dyson.kpq_map(),
        p.Dm_ab, &sG_tskij,
        _opt_dW_tRPQ ? &(*_opt_dW_tRPQ) : nullptr,
        K.hartree, K.exchange, K.has_sigma(),
        p.keep_F_PQ));

    app_log(2, "  DeltaX IBC correction: setup complete.");
    utils::memlog("lr_driver::lr_setup: after build_lr_ibc");
    _Timer.stop("LR_DRIVER_SETUP_IBC");
  }
  _Timer.stop("LR_DRIVER_SETUP");
  utils::memlog("lr_driver::lr_setup: end of LR_DRIVER_SETUP");
  print_setup_timers();

  _setup_done = true;
}


namespace {

// τ -> iω -> τ round trip of the double counting. The ground state does the same: weiss.solve_gw_dc (weiss.py) stores
// Σ_dc as iaft.tau_to_w(eval_gw_dc_t(...)) and Π_dc as iaft.tau_to_w_phsym(eval_pi_rpa(...)); the embedded Σ returns to τ
// with w_to_tau in embed_t::add_Sigma_dyn_correction (embed_t.cpp), and Π_imp and Π_dc with w_to_tau_PHsym in
// scr_coulomb_t::eval_Pi_qdep, reached from scr_coulomb_t::update_w in the GW+EDMFT SCF and from
// recompute_W_and_eps_inv_head (MBPT_drivers.cpp) in run_lr's W rebuild.
// The round trip is the identity on an IR grid but not on a DLR grid (white noise: 3e-2 fermion, 2e-3 boson on the
// Hubbard test grid), so the derivative of the ground-state map includes it.
void dc_fermi_round_trip(const imag_axes_ft::IAFT& ft, nda::array<ComplexType, 5>& dSigma_tsIab) {
  nda::array<ComplexType, 5> dSigma_w(ft.nw_f(), dSigma_tsIab.extent(1), 1, dSigma_tsIab.extent(3),
                                      dSigma_tsIab.extent(4));
  ft.tau_to_w(dSigma_tsIab, dSigma_w, imag_axes_ft::fermion);
  ft.w_to_tau(dSigma_w, dSigma_tsIab, imag_axes_ft::fermion);
}

void dc_bose_round_trip(const imag_axes_ft::IAFT& ft, nda::array<ComplexType, 5>& dPi_tabcd) {
  const long nw_half = (ft.nw_b() % 2 == 0) ? ft.nw_b() / 2 : ft.nw_b() / 2 + 1;
  nda::array<ComplexType, 5> dPi_w(nw_half, dPi_tabcd.extent(1), dPi_tabcd.extent(2), dPi_tabcd.extent(3),
                                   dPi_tabcd.extent(4));
  ft.tau_to_w_PHsym(dPi_tabcd, dPi_w);
  ft.w_to_tau_PHsym(dPi_w, dPi_tabcd);
}

} // anonymous

nda::array<ComplexType, 5> lr_driver::dc_sigma_response(
    const embedding_info& emb, const nda::array<ComplexType, 5>& dGloc_tsIab,
    const nda::array<ComplexType, 5>* dWloc_pb) {
  utils::check(emb.dc.has_value(), "lr_driver::dc_sigma_response: the embedding has no double counting.");
  auto dSigma_dc_tsIab = lr_dc_sigma(emb.dc->Gloc_tsIab, dGloc_tsIab, emb.dc->Wloc_pb, dWloc_pb, emb.dc->cells);
  dc_fermi_round_trip(*_dyson.FT(), dSigma_dc_tsIab);
  return dSigma_dc_tsIab;
}

void lr_driver::apply_kernel_gw(
    sArray_t<Array_view_5D_t>& sSigma_tskij_out,
    const sArray_t<Array_view_5D_t>& sDeltaG_tskij,
    const sArray_t<Array_view_5D_t>& sG_tskij,
    thc_reader_t& thc,
    const lr_params& p,
    sArray_t<Array_view_5D_t>* sDeltaSigma_term2_tskij) {

  solvers::lr_gw& gw_solver = *_lr_gw;
  const lr_ibc_DeltaX* ibc_ptr = _opt_ibc ? &(*_opt_ibc) : nullptr;
  auto& sS_skij = _dyson.sS_skij();
  const auto* dc = (p.embedding && p.embedding->dc) ? &(*p.embedding->dc) : nullptr;
  nda::array<long, 1> kpq_map;  // the k+Q (= q+Q) map; local_projection takes it 64-bit
  if (p.embedding) kpq_map = nda::array<long, 1>(_lr_dyson.kpq_map());
  if (!p.kernel.sigma_G_dW) {
    // Term 1 only: ΔΣ = -ΔG ⊙ W_c
    _Timer.start("LR_GW_SIGMA");
    // The term-1 divergence correction is applied inside the evaluator.
    auto S_loc = sS_skij.local();
    gw_solver.evaluate_sigma_DeltaG(
        sSigma_tskij_out, sDeltaG_tskij.local(), *_opt_dW_tRPQ, thc, ibc_ptr,
        &S_loc, p.eps_inv_head);
    // up_Σ(δΣ_imp - δΣ_dc[δG^loc, 0]): W is frozen, so δW^loc = 0
    if (dc) {
      auto dGloc_tsIab = local_projection::sum_C_O_Cdag(
          sDeltaG_tskij.local(), p.embedding->C_skIai, p.embedding->W_rng, nullptr, nullptr, &kpq_map, _mpi->comm);
      auto dSigma_dc_tsIab = dc_sigma_response(*p.embedding, dGloc_tsIab, nullptr);
      // ΔΣ_out += up(δΣ_imp - δΣ_dc)
      nda::array<ComplexType, 5> dSigma_imp_tsIab = p.embedding->Sigma_imp_tsIab - dSigma_dc_tsIab;
      local_projection::upfold_accumulate(sSigma_tskij_out, dSigma_imp_tsIab, p.embedding->C_skIai, p.embedding->W_rng,
                                          &kpq_map, ComplexType(1.0));
    }
    _Timer.stop("LR_GW_SIGMA");
    _mpi->comm.barrier();
    return;
  }

  // Step 3b: ΔP = -ΔG·G - G·ΔG
  _Timer.start("LR_GW_PI");
  auto dDeltaPi_tqPQ = _lr_pi->evaluate_lr_Pi(
      sG_tskij.local(), sDeltaG_tskij.local(), thc,
      *_opt_dG_tsRPQ, *_opt_dG_mtau_tsRPQ, ibc_ptr);
  // ΔΠ += upfold(δΠ_imp - δΠ_dc[δG^loc]), on ΔΠ's own blocks, before the ΔW solve
  nda::array<ComplexType, 5> dGloc_tsIab;
  if (dc) {
    dGloc_tsIab = local_projection::sum_C_O_Cdag(
        sDeltaG_tskij.local(), p.embedding->C_skIai, p.embedding->W_rng, nullptr, nullptr, &kpq_map, _mpi->comm);
    auto dPi_dc_tabcd = lr_dc_pi(dc->Gloc_tsIab, dGloc_tsIab, dc->cells);
    dc_bose_round_trip(*_dyson.FT(), dPi_dc_tabcd);
    nda::array<ComplexType, 5> dPi_imp_tabcd = p.embedding->Pi_imp_tabcd - dPi_dc_tabcd;
    local_projection::upfold_pi_block(dPi_imp_tabcd, p.embedding->B_qIPab, dDeltaPi_tqPQ, &kpq_map, true);
  }
  _mpi->comm.barrier();
  _Timer.stop("LR_GW_PI");

  // Step 3c-3d: ΔW_c(τ) via solve_lr_dyson_W (in-place, uses cached W_full)
  _Timer.start("LR_GW_W");
  _lr_scr->solve_lr_dyson_W(dDeltaPi_tqPQ, *_opt_dW_full_wqPQ, thc);
  // dDeltaPi_tqPQ now contains ΔW_c(τ) in q-local distribution
  auto& dDeltaW_tqPQ = dDeltaPi_tqPQ;  // alias for clarity

  // δW^loc = (1/N_q) Σ_q B^{q+Q†} ΔW_c B^q on the blocks of the distributed ΔW_c, kept for the output
  // (overwritten at every call)
  if (p.embedding) {
    _DeltaWloc_tabcd = local_projection::downfold_W_block(
        dDeltaW_tqPQ.local(), dDeltaW_tqPQ.origin(), dDeltaW_tqPQ.global_shape()[0],
        p.embedding->B_qIPab, kpq_map, false, _mpi->comm);
  }

  // Extract Δeps_inv_head from ΔW for divergence correction term 2 (q_pert=0 only)
  nda::array<ComplexType, 1> delta_eps_inv_head;
  if (p.div_treatment != "ignore_g0" && is_q_gamma()) {
    auto [delta_eps_inv_q, delta_head] =
        solvers::div_utils::eps_inv_head_t(
            dDeltaW_tqPQ, thc, *thc.MF(), _dyson.FT(), p.div_treatment);
    delta_eps_inv_head = std::move(delta_head);
  }

  _mpi->comm.barrier();
  _Timer.stop("LR_GW_W");

  // Step 3e-3f: ΔΣ = -ΔG ⊙ W_c - G ⊙ ΔW.
  // ΔW stays in (t,q,P,Q): the Σ evaluator consumes one τ slice at a time,
  // which is contiguous in this layout and matches term 1's dW_tRPQ.
  _Timer.start("LR_GW_SIGMA");
  if (p.split_sigma_terms) {
    // One-shot G0W0: compute the two terms separately, then store
    //   sDeltaSigma_tskij       = term1 + term2  (total ΔΣ, same as fused)
    //   sDeltaSigma_term2_tskij = term2 (G0·dW0)  [written as DeltaSigma_GdW]
    // term 1 (-ΔG⊙W_c + div) and term 2 (-G⊙ΔW + div) use separate solver
    // instances (gw_solver / _lr_gw2, built once by lr_setup) — the workspace
    // is cached per (term1,term2) combination.
    auto S_loc = sS_skij.local();
    gw_solver.evaluate_sigma_DeltaG(
        sSigma_tskij_out, sDeltaG_tskij.local(), *_opt_dW_tRPQ, thc, ibc_ptr,
        &S_loc, p.eps_inv_head);
    _lr_gw2->evaluate_sigma_DeltaW(
        *sDeltaSigma_term2_tskij, sG_tskij.local(), dDeltaW_tqPQ, thc,
        *_opt_dG_tsRPQ, *_opt_dG_mtau_tsRPQ, &S_loc, &delta_eps_inv_head);
    // Accumulate term2 into sSigma_tskij_out so it holds the total ΔΣ.
    // Both arrays are node-replicated shared memory (each solver all_reduced
    // its result), so add once per node on the node root.
    sSigma_tskij_out.win().fence();
    sDeltaSigma_term2_tskij->win().fence();
    if (_mpi->node_comm.root())
      sSigma_tskij_out.local() += sDeltaSigma_term2_tskij->local();
    sSigma_tskij_out.win().fence();
    _mpi->comm.barrier();
  } else {
    // Both terms, fused: ΔΣ = -ΔG ⊙ W_c - G ⊙ ΔW in a single R-space pass.
    auto S_loc = sS_skij.local();
    gw_solver.evaluate_sigma(
        sSigma_tskij_out, sDeltaG_tskij.local(), *_opt_dW_tRPQ,
        sG_tskij.local(), dDeltaW_tqPQ, thc,
        *_opt_dG_tsRPQ, *_opt_dG_mtau_tsRPQ, ibc_ptr,
        &S_loc, p.eps_inv_head, &delta_eps_inv_head);
  }
  // up_Σ(δΣ_imp - δΣ_dc[δG^loc, δW^loc])
  if (dc) {
    auto dWloc_pb = chemistry_to_product_basis(*_DeltaWloc_tabcd);
    auto dSigma_dc_tsIab = dc_sigma_response(*p.embedding, dGloc_tsIab, &dWloc_pb);
    // ΔΣ_out += up(δΣ_imp - δΣ_dc)
    nda::array<ComplexType, 5> dSigma_imp_tsIab = p.embedding->Sigma_imp_tsIab - dSigma_dc_tsIab;
    local_projection::upfold_accumulate(sSigma_tskij_out, dSigma_imp_tsIab, p.embedding->C_skIai, p.embedding->W_rng,
                                        &kpq_map, ComplexType(1.0));
  }
  _mpi->comm.barrier();
  _Timer.stop("LR_GW_SIGMA");
}


std::optional<solvers::lr_hf::hsex_kernel_t> lr_driver::hsex_kernel() {
  using hsex_kernel_t = solvers::lr_hf::hsex_kernel_t;
  if (!_opt_dWc0_qPQ) return std::nullopt;
  return hsex_kernel_t{&(*_opt_dWc0_qPQ), _hsex_head_factor};
}


void lr_driver::apply_kernel(
    bool precond,
    sArray_t<Array_view_4D_t>& sDeltaF_out,
    sArray_t<Array_view_5D_t>* pDeltaSigma_out,
    const sArray_t<Array_view_4D_t>& sDeltaDm_skij,
    const sArray_t<Array_view_5D_t>& sDeltaG_tskij,
    const sArray_t<Array_view_5D_t>& sG_tskij,
    thc_reader_t& thc,
    const lr_params& p,
    sArray_t<Array_view_5D_t>* sDeltaSigma_term2_tskij,
    nda::array<ComplexType, 4>* DeltaF_PQ_out) {

  // The operator alone takes the static inputs and the IBC correction.
  const lr_kernel_spec& terms = precond ? p.precond_kernel : p.kernel;
  if (terms.has_F()) {
    auto hsex = (precond ? p.precond_hsex : p.exchange_static_W) ? hsex_kernel() : std::nullopt;
    _Timer.start("LR_HF");
    (precond ? *_lr_hf_pc : *_lr_hf).evaluate(
        sDeltaF_out, sDeltaDm_skij, thc, _dyson.sS_skij().local(),
        terms.hartree, terms.exchange,
        !precond && _opt_ibc ? &(*_opt_ibc) : nullptr,
        precond ? nullptr : p.DeltaV_qPQ,
        precond ? nullptr : p.Dm_ab,
        DeltaF_PQ_out,
        !precond && p.include_xc,
        hsex ? &(*hsex) : nullptr);
    _Timer.stop("LR_HF");
    _mpi->comm.barrier();
  }

  // K_pc is Σ-free.
  if (terms.has_sigma() && pDeltaSigma_out) {
    apply_kernel_gw(*pDeltaSigma_out, sDeltaG_tskij, sG_tskij, thc, p,
                    sDeltaSigma_term2_tskij);
  }
}


void lr_driver::precond_apply(nda::array<ComplexType, 1> const& r,
                              nda::array<ComplexType, 1>& z,
                              const sArray_t<Array_view_5D_t>& sG_tskij,
                              thc_reader_t& thc, const lr_params& p) {
  auto& r_F = *_sDeltaF_pc_r;
  auto& r_S = *_sDeltaSigma_r;
  const long mF = _DeltaF.i1 - _DeltaF.i0;
  const long mS = _DeltaSigma.i1 - _DeltaSigma.i0;
  auto rF = nda::range(0, mF);
  auto rS = nda::range(mF, mF + mS);

  // Gather r = (r_F, r_Σ) into shared memory.
  r_F.win().fence();
  r_S.win().fence();
  _DeltaF.slice(r_F.local()) = r(rF);
  _DeltaSigma.slice(r_S.local()) = r(rS);
  utils::complete_shared_slices(*_mpi, _pmap, r_F, r_S);

  precond_solve(r_F, &r_S, sG_tskij, thc, p);

  // z = (r_F + y_F, r_Σ)
  z(rF) = r(rF) + _DeltaF.slice(_sDeltaF_pc_y->local());
  z(rS) = r(rS);
}


void lr_driver::precond_solve(const sArray_t<Array_view_4D_t>& r_F,
                              const sArray_t<Array_view_5D_t>* r_S,
                              const sArray_t<Array_view_5D_t>& sG_tskij,
                              thc_reader_t& thc, const lr_params& p) {
  auto& y_F = *_sDeltaF_pc_y;
  const long mF = _DeltaF.i1 - _DeltaF.i0;

  // y_F = 0 serves as both the zero ΔH0 and ΔF of the ΔDm pass below, and as the
  // start of the inner solve.
  y_F.set_zero();
  if (r_S) {
    // ΔDm of r_Σ alone, at Δμ = 0: the sweeps' ΔDm-only passes apply the
    // fix_density shift to the sum.
    _Timer.start("LR_DYSON");
    _lr_dyson.solve_lr_dyson(*_sDeltaDm_pc, y_F, y_F, r_S, /*fix_density=*/false,
                             static_cast<const sArray_t<Array_view_4D_t>*>(nullptr));
    _Timer.stop("LR_DYSON");
  }

  // y_F = K_pc[Pχ(r_F + y_F, r_Σ)], from y_F = 0, by the ΔF-only _pc_gcr.
  _pc_gcr->reset_rhs();
  nda::array<ComplexType, 1> y_prev(mF);
  nda::array<ComplexType, 5> empty_sigma;
  nda::array<ComplexType, 1> empty_prev;
  double tol_in = 0.0;
  double rn = 0.0;
  _pc_sweeps = 0;
  for (int sweep = 1; sweep <= p.precond_max_iter; ++sweep) {
    y_prev = _DeltaF.slice(y_F.local());
    _Timer.start("LR_DYSON");
    _lr_dyson.solve_lr_dyson_dm_only(*_sDeltaDm_pc_y, r_F, y_F,
                                     r_S ? &(*_sDeltaDm_pc) : nullptr, p.fix_density);
    _Timer.stop("LR_DYSON");
    // K_pc is Σ-free and reads no ΔG; sG_tskij only fills the slot.
    apply_kernel(/*precond=*/true, y_F, nullptr, *_sDeltaDm_pc_y, sG_tskij, sG_tskij,
                 thc, p);
    // ‖r_in,0‖ = ‖K_pc[Pχ(r_F, r_Σ)]‖, the inner residual at y_F = 0.
    if (sweep == 1)
      tol_in = p.precond_tol * utils::striped_norm(_mpi->comm, _DeltaF.slice(y_F.local()),
                                                   y_prev, false).first;
    // The last sweep takes any residual as converged, so y_F is left at the
    // inner iterate rather than at its next probe.
    const double tol_s = (sweep == p.precond_max_iter)
                             ? std::numeric_limits<double>::infinity() : tol_in;
    bool conv = false;
    rn = _pc_gcr->step(_mpi->comm, _pmap, y_F.local(), y_prev, empty_sigma, empty_prev,
                       tol_s, conv);
    utils::complete_shared_slices(*_mpi, _pmap, y_F);
    _pc_sweeps = sweep;
    if (conv) break;
  }
  if ((p.fixed_order || p.initial_guess_precond) && _pc_sweeps == p.precond_max_iter &&
      rn > tol_in)
    app_log(1, "    [WARNING] preconditioner: inner sweep cap precond_max_iter = {} hit "
               "with ||r_in|| / tol_in = {:.3e}; the inner error enters the result.",
            p.precond_max_iter, rn / tol_in);
}


std::tuple<int, double> lr_driver::lr_solve_one(
    sArray_t<Array_view_5D_t>& sDeltaG_tskij,
    sArray_t<Array_view_4D_t>& sDeltaDm_skij,
    sArray_t<Array_view_4D_t>& sDeltaF_skij,
    sArray_t<Array_view_5D_t>* sDeltaSigma_tskij,
    const sArray_t<Array_view_5D_t>& sG_tskij,
    const sArray_t<Array_view_4D_t>& sDeltaH0_skij,
    thc_reader_t& thc,
    const lr_params& p,
    sArray_t<Array_view_5D_t>* sDeltaSigma_term2_tskij,
    sArray_t<Array_view_4D_t>* sDeltaVcorr_skij,
    nda::array<ComplexType, 4>* DeltaF_ibc_out,
    nda::array<ComplexType, 4>* F_PQ_out,
    nda::array<ComplexType, 4>* DeltaF_PQ_out) {

  utils::check(_setup_done, "lr_driver::lr_solve_one: call lr_setup first.");
  _DeltaWloc_tabcd.reset();
  const lr_kernel_spec& K = p.kernel;
  const bool qp = (p.qp_static != nullptr);
  auto& opt_ibc            = _opt_ibc;
  utils::check(!qp || sDeltaVcorr_skij, "lr_driver::lr_solve_one: qp mode needs a ΔV_QPGW array.");
  utils::check(!K.has_sigma() || sDeltaSigma_tskij,
               "lr_driver::lr_solve_one: a kernel with a Σ needs a ΔΣ array.");

  // The ΔΣ the kernel writes and the Dyson RHS reads; qp mode replaces it in the
  // RHS by ΔV_QPGW.
  sArray_t<Array_view_5D_t>* pDeltaSigma = K.has_sigma() ? sDeltaSigma_tskij : nullptr;
  const bool track_sigma = K.has_sigma() && !qp;

  // Reset every quantity carried across SCF iterations, so this solve cannot see
  // the previous perturbation's state. set_zero ends with fence + node_sync.
  // The previous iterates need no reset: ΔF/ΔΣ/ΔV_QPGW are saved at every
  // iteration and ΔDm from iteration 2, each before it is read.
  _Timer.start("LR_RESET");
  sDeltaF_skij.set_zero();
  if (sDeltaSigma_tskij) sDeltaSigma_tskij->set_zero();
  if (qp) sDeltaVcorr_skij->set_zero();
  if (_lr_diis) _lr_diis->reset();
  if (_lr_gcr) _lr_gcr->reset();
  if (_pc_gcr) _pc_gcr->reset();
  // No clock is reset here: the SCF timers and the solvers' sub-clocks both
  // accumulate over every perturbation of the call, so the table printed after
  // the last one is the cost of the whole batch. Each intermediate table is
  // therefore a running total, and one perturbation's cost is the difference
  // between consecutive tables.
  _mpi->comm.barrier();
  _Timer.stop("LR_RESET");

  // dst <- src on the node-replicated shared window.
  auto copy_shm = [&](auto& dst, auto& src) {
    dst.win().fence();
    if (_mpi->node_comm.root()) dst.local() = src.local();
    dst.win().fence();
    _mpi->node_comm.barrier();
  };

  _Timer.start("LR_SCF");

  // Start at the preconditioner solution of ΔH0: ΔF = y_F, ΔΣ = 0 (zeroed by the
  // reset above), so that V_0 = ΔH0 + ΔF = M⁻¹ ΔH0.
  if (p.initial_guess_precond) {
    _Timer.start("LR_PRECOND");
    precond_solve(sDeltaH0_skij, nullptr, sG_tskij, thc, p);
    _Timer.stop("LR_PRECOND");
    copy_shm(sDeltaF_skij, *_sDeltaF_pc_y);
    app_log(1, "  initial guess: preconditioner solution of ΔH0 ({} sweeps)", _pc_sweeps);
  }

  double Delta_mu = 0.0;
  int iter = 0;
  // An empty kernel has no feedback: the loop is skipped and the Dyson pass after
  // it is the answer.
  const int max_iter = K.empty() ? 0 : p.max_iter;
  bool converged = K.empty();

  const bool use_gcr = (_lr_gcr != nullptr);
  // The preconditioner map z = M⁻¹ r (GCR direction, damping step), preconditioned runs only.
  std::function<void(nda::array<ComplexType, 1> const&, nda::array<ComplexType, 1>&)> pc_fn;
  if (!p.precond_kernel.empty())
    pc_fn = [&](auto const& r, auto& z) {
      _Timer.start("LR_PRECOND");
      precond_apply(r, z, sG_tskij, thc, p);
      _Timer.stop("LR_PRECOND");
    };

  // Residual and preconditioned step of the damping iteration, sized on first use.
  nda::array<ComplexType, 1> pc_r, pc_z;

  // Fill sDeltaG_tskij with the ΔG(τ) of the current ΔDm, at most once per Dyson solve.
  bool DeltaG_in_shm = false;
  auto update_DeltaG = [&]() {
    if (DeltaG_in_shm) return;
    if (!K.has_sigma()) {
      // The last pass was ΔDm-only and formed no ΔG(τ): rerun the full pass on its input.
      _Timer.start("LR_DYSON");
      Delta_mu = _lr_dyson.solve_lr_dyson(
          sDeltaDm_skij, sDeltaH0_skij, *_sDeltaF_dyson_in, pDeltaSigma,
          p.fix_density, static_cast<const sArray_t<Array_view_4D_t>*>(nullptr));
      _Timer.stop("LR_DYSON");
      _mpi->comm.barrier();
    }
    _lr_dyson.materialize_DeltaG_tau(sDeltaG_tskij);
    DeltaG_in_shm = true;
  };

  // One Dyson pass on the current ΔF/ΔΣ:
  // ΔG = G_{k+q} · [ΔH0 + ΔF + (ΔΣ | ΔV_QPGW) - Δμ·S] · G_k
  // In qp mode the dynamic ΔΣ is dropped from the RHS (skipping its τ→ω) and
  // the static ΔV_QPGW enters as a frequency-independent one-body term.
  // A Σ-free kernel reads ΔDm alone, so it takes the ΔDm-only pass.
  auto dyson_pass = [&]() {
    if (K.has_sigma()) {
      // Kernel with a Σ: full pass, ΔG(τ) materialized
      Delta_mu = _lr_dyson.solve_lr_dyson(
          sDeltaDm_skij, sDeltaH0_skij,
          sDeltaF_skij, track_sigma ? sDeltaSigma_tskij : nullptr,
          p.fix_density, qp ? sDeltaVcorr_skij : nullptr);
      // The kernel evaluates a ΔΣ from ΔG(τ), so replicate it into shared memory now.
      _lr_dyson.materialize_DeltaG_tau(sDeltaG_tskij);
      DeltaG_in_shm = true;
    } else {
      // Σ-free kernel: ΔDm-only pass. The ΔF it is fed is kept for update_DeltaG.
      copy_shm(*_sDeltaF_dyson_in, sDeltaF_skij);
      Delta_mu = _lr_dyson.solve_lr_dyson_dm_only(
          sDeltaDm_skij, sDeltaH0_skij, sDeltaF_skij, nullptr, p.fix_density);
      DeltaG_in_shm = false;
    }
  };

  const bool log_sigma_col = K.has_sigma();

  // SCF iteration header
  if (log_sigma_col) {
    app_log(1, "\n  iter    ||ΔDm||         ||ΔDm-ΔDm_prev||  ||ΔF||          ||ΔF-ΔF_prev||   ||ΔΣ||          ||ΔΣ-ΔΣ_prev||   Δμ");
    app_log(1, "  " + std::string(120, '-'));
  } else {
    app_log(1, "\n  iter    ||ΔDm||         ||ΔDm-ΔDm_prev||  ||ΔF||          ||ΔF-ΔF_prev||   Δμ");
    app_log(1, "  " + std::string(92, '-'));
  }

  for (iter = 1; iter <= max_iter; ++iter) {

    // Save the previous iterates, each as this rank's partition slice only.
    _Timer.start("LR_SAVE");
    // The iteration-algorithm step reads the input ΔF/ΔΣ (ΔV_QPGW) of every
    // iteration, the reset zeros at the first. ΔDm only enters a difference norm.
    _DeltaF.prev = _DeltaF.slice(sDeltaF_skij.local());
    if (track_sigma)
      _DeltaSigma.prev = _DeltaSigma.slice(sDeltaSigma_tskij->local());
    if (qp)
      _DeltaVcorr.prev = _DeltaVcorr.slice(sDeltaVcorr_skij->local());
    if (iter > 1)
      _DeltaDm.prev = _DeltaDm.slice(sDeltaDm_skij.local());
    _mpi->comm.barrier();
    _Timer.stop("LR_SAVE");

    // Step 1: Solve LR Dyson equation
    _Timer.start("LR_DYSON");
    dyson_pass();
    _Timer.stop("LR_DYSON");
    _mpi->comm.barrier();

    // Compute norms for logging, reduced over the global comm from the slices.
    _Timer.start("LR_CONVERGENCE");
    auto norms_Dm = utils::striped_norm(
        _mpi->comm, _DeltaDm.slice(sDeltaDm_skij.local()), _DeltaDm.prev, iter > 1);
    double norm_DeltaDm = norms_Dm.first;
    double norm_DeltaDm_diff = norms_Dm.second;
    _Timer.stop("LR_CONVERGENCE");

    // Steps 2-3: apply the kernel — the ΔF branch, then the ΔΠ -> ΔW -> ΔΣ pipeline.
    apply_kernel(/*precond=*/false, sDeltaF_skij, pDeltaSigma,
                 sDeltaDm_skij, sDeltaG_tskij, sG_tskij, thc, p,
                 sDeltaSigma_term2_tskij);

    // Step 3g (qp mode): statify the dynamic ΔΣ(iω) into the static ΔV_QPGW(k)
    // using the frozen QP orbitals/energies. ΔV_QPGW is the tracked/mixed static
    // quantity and enters the Dyson RHS at the next iteration.
    if (qp) {
      _Timer.start("LR_QPGW_STATIC");
      auto sVcorr = lr_qp_approx(
          *sDeltaSigma_tskij, *p.qp_static->sMO_skia, *p.qp_static->sE_ska,
          p.qp_static->mu, _lr_dyson.kpq_map(), is_q_gamma(),
          *_dyson.FT(), p.qp_static->qp_params);
      copy_shm(*sDeltaVcorr_skij, sVcorr);
      _mpi->comm.barrier();
      _Timer.stop("LR_QPGW_STATIC");
    }

    // Step 4: Apply iteration algorithm (DIIS, damping or GCR) on the combined
    // (ΔF, ΔΣ).
    _Timer.start("LR_ITER_ALG");
    double gcr_rnorm = -1.0;
    bool gcr_converged = false;
    _pc_sweeps = 0;
    // Striped: every rank of the global comm participates, each operating on its
    // `_pmap` element-slice of the shared ΔF/ΔΣ and writing the mixed result
    // back in place. Pass .local() views directly (in/out); the "prev" arguments
    // are already this rank's slice.
    //
    // The static second quantity mixed alongside ΔF is the dynamic ΔΣ in the
    // standard path, or the static ΔV_QPGW in qp mode. Both accelerators take
    // the same two-slot arguments.
    //
    // Damping; mixing >= 1 (undamped Picard) is a no-op.
    const double mixing = p.iter_params.mixing;
    auto damp = [&](auto A, auto const& A_prev) {
      if (mixing >= 1.0 || A.size() == 0) return;
      const long n = A.size();
      auto [a0, a1] = _pmap.my_slice(n);
      utils::check(A_prev.size() == a1 - a0,
                   "lr_driver: damping history slice ({}) != partition slice ({}).",
                   A_prev.size(), a1 - a0);
      auto A_loc = nda::reshape(A, std::array<long, 1>{n})(nda::range(a0, a1));
      A_loc = mixing * A_loc + (1.0 - mixing) * A_prev;
    };
    auto step = [&](auto F, auto const& F_prev, auto S, auto const& S_prev) {
      if (use_gcr) {
        // GCR
        gcr_rnorm = _lr_gcr->step(_mpi->comm, _pmap, F, F_prev, S, S_prev,
                                   p.tol, gcr_converged, pc_fn ? &pc_fn : nullptr);
      } else if (_lr_diis) {
        // DIIS
        _lr_diis->next_step_combined(_mpi->comm, _pmap, F, F_prev, S, S_prev, iter);
      } else if (pc_fn) {
        // Preconditioned damping, y = y_prev + mixing · M⁻¹ r with r = g(y_prev) - y_prev,
        // on this rank's slices of (ΔF, ΔΣ), laid out as in lr_gcr::step.
        const long nF = F.size();
        const long nS = S.size();
        auto [f0, f1] = _pmap.my_slice(nF);
        auto [s0, s1] = _pmap.my_slice(nS);
        utils::check(static_cast<long>(F_prev.size()) == f1 - f0 &&
                     static_cast<long>(S_prev.size()) == s1 - s0,
                     "lr_driver: preconditioned damping history slices ({}, {}) != "
                     "partition slices ({}, {}).", F_prev.size(), S_prev.size(),
                     f1 - f0, s1 - s0);
        const long mF = f1 - f0;
        auto rF = nda::range(0, mF);
        auto rS = nda::range(mF, mF + (s1 - s0));
        auto F_loc = nda::reshape(F, std::array<long, 1>{nF})(nda::range(f0, f1));
        auto S_loc = nda::reshape(S, std::array<long, 1>{nS})(nda::range(s0, s1));
        if (pc_r.size() != mF + (s1 - s0)) {
          pc_r = nda::array<ComplexType, 1>(mF + (s1 - s0));
          pc_z = nda::array<ComplexType, 1>(mF + (s1 - s0));
        }
        pc_r(rF) = F_loc - F_prev;
        pc_r(rS) = S_loc - S_prev;
        pc_fn(pc_r, pc_z);
        F_loc = F_prev + mixing * pc_z(rF);
        S_loc = S_prev + mixing * pc_z(rS);
      } else {
        // Damping
        damp(F, F_prev);
        damp(S, S_prev);
      }
    };
    if (qp) {
      // qp mode: (ΔF, ΔV_QPGW)
      step(sDeltaF_skij.local(), _DeltaF.prev,
           sDeltaVcorr_skij->local(), _DeltaVcorr.prev);
    } else if (track_sigma) {
      // Kernel with a Σ: (ΔF, ΔΣ)
      step(sDeltaF_skij.local(), _DeltaF.prev,
           sDeltaSigma_tskij->local(), _DeltaSigma.prev);
    } else {
      // Σ-free kernel: ΔF alone
      nda::array<ComplexType, 5> empty_sigma;
      nda::array<ComplexType, 1> empty_prev;
      step(sDeltaF_skij.local(), _DeltaF.prev, empty_sigma, empty_prev);
    }
    // The mixing above writes each rank's slice of the shared ΔF/ΔΣ buffer in
    // place with no trailing collective; each node then holds a valid copy of
    // its own contiguous element run only. With the striping global, each
    // element is mixed exactly once in the whole job.
    if (qp)
      utils::complete_shared_slices(*_mpi, _pmap, sDeltaF_skij, *sDeltaVcorr_skij);
    else if (track_sigma)
      utils::complete_shared_slices(*_mpi, _pmap, sDeltaF_skij, *sDeltaSigma_tskij);
    else
      utils::complete_shared_slices(*_mpi, _pmap, sDeltaF_skij);
    _Timer.stop("LR_ITER_ALG");

    // Compute norms of ΔF and ΔF-ΔF_prev for logging. The previous iterates are
    // stored striped, so the norms are reduced over the global comm from the
    // same slices.
    _Timer.start("LR_CONVERGENCE");
    auto norms_F = utils::striped_norm(
        _mpi->comm, _DeltaF.slice(sDeltaF_skij.local()), _DeltaF.prev, iter > 1);
    double norm_DeltaF = norms_F.first;
    double norm_DeltaF_diff = norms_F.second;

    // Compute norms of the tracked static second quantity (dynamic ΔΣ in the
    // standard path; static ΔV_QPGW in qp mode) for logging/convergence.
    double norm_DeltaSigma = 0.0;
    double norm_DeltaSigma_diff = 0.0;
    if (qp) {
      auto norms_V = utils::striped_norm(
          _mpi->comm, _DeltaVcorr.slice(sDeltaVcorr_skij->local()), _DeltaVcorr.prev,
          iter > 1);
      norm_DeltaSigma = norms_V.first;
      norm_DeltaSigma_diff = norms_V.second;
    } else if (track_sigma) {
      auto norms_Sigma = utils::striped_norm(
          _mpi->comm, _DeltaSigma.slice(sDeltaSigma_tskij->local()), _DeltaSigma.prev,
          iter > 1);
      norm_DeltaSigma = norms_Sigma.first;
      norm_DeltaSigma_diff = norms_Sigma.second;
    }
    _Timer.stop("LR_CONVERGENCE");

    // DIIS/damping: all successive-iterate differences below tol. GCR: the honest
    // residual ‖r‖ of the affine problem; the difference norms above remain
    // diagnostics.
    bool dm_converged = norm_DeltaDm_diff < p.tol;
    bool f_converged = !K.has_F() || norm_DeltaF_diff < p.tol;
    bool sigma_converged = !K.has_sigma() || norm_DeltaSigma_diff < p.tol;
    bool conv_std = (iter > 1) && dm_converged && f_converged && sigma_converged;
    bool iter_converged = !p.fixed_order && (use_gcr ? gcr_converged : conv_std);

    // Log iteration
    _Timer.start("LR_CONVERGENCE");
    std::string iter_lbl = fmt::format("{}", iter);
    if (iter == 1) {
      if (log_sigma_col) {
        app_log(1, "  {:>4s}    {:.6e}     {:13s}     {:.6e}   {:13s}   {:.6e}   {:13s}   {:.3e}",
                iter_lbl, norm_DeltaDm, "---", norm_DeltaF, "---", norm_DeltaSigma, "---", Delta_mu);
      } else {
        app_log(1, "  {:>4s}    {:.6e}     {:13s}     {:.6e}   {:13s}   {:.3e}",
                iter_lbl, norm_DeltaDm, "---", norm_DeltaF, "---", Delta_mu);
      }
    } else {
      if (log_sigma_col) {
        app_log(1, "  {:>4s}    {:.6e}     {:.6e}      {:.6e}   {:.6e}    {:.6e}   {:.6e}    {:.3e}",
                iter_lbl, norm_DeltaDm, norm_DeltaDm_diff, norm_DeltaF, norm_DeltaF_diff,
                norm_DeltaSigma, norm_DeltaSigma_diff, Delta_mu);
      } else {
        app_log(1, "  {:>4s}    {:.6e}     {:.6e}      {:.6e}   {:.6e}    {:.3e}",
                iter_lbl, norm_DeltaDm, norm_DeltaDm_diff, norm_DeltaF, norm_DeltaF_diff, Delta_mu);
      }
    }
    if (!use_gcr && pc_fn && !_lr_diis)
      app_log(1, "          [precond damping]  precond sweeps {}", _pc_sweeps);
    if (use_gcr) {
      const std::string pc_lbl =
          pc_fn ? fmt::format("  precond sweeps {}", _pc_sweeps) : std::string();
      app_log(1, "          [GCR] ||r|| = {:.6e}  (basis {}){}{}", gcr_rnorm,
              _lr_gcr->basis_size(), _lr_gcr->kicked() ? "  kick" : "", pc_lbl);
      if (_lr_gcr->stagnated())
        app_log(1, "    [WARNING] LR GCR: ||r|| = {:.3e} has not dropped "
                   "below {:.0f}% of its value {} steps ago (tol = {:.3e}), which "
                   "the Krylov kick normally prevents; if it does not converge, "
                   "retry with DIIS or a larger max_subsp_size.",
                gcr_rnorm, 100.0 * lr_gcr::stag_ratio, lr_gcr::stag_steps, p.tol);
    }
    _Timer.stop("LR_CONVERGENCE");

    // Step 5: Check convergence.
    if (iter_converged) {
      converged = true;
      break;
    }

    _mpi->comm.barrier();
  }

  // A fixed-order run has no loop exit but the cap; it reports the order.
  if (p.fixed_order) iter = max_iter;

  // Every exit, converged or at max_iter: the step of the last iteration has
  // replaced ΔF (and ΔΣ / ΔV_QPGW) by its next iterate (a DIIS or damping mix, or
  // GCR's x), while ΔDm is the response to that iteration's input. This Dyson
  // pass forms ΔDm at the returned iterate (with a Σ it also materializes ΔG(τ);
  // otherwise update_DeltaG forms it on demand). For an empty kernel it is the
  // only pass.
  _Timer.start("LR_DYSON");
  dyson_pass();
  _Timer.stop("LR_DYSON");
  _mpi->comm.barrier();

  _Timer.stop("LR_SCF");

  // After the solve ΔG(τ) is read by the checkpoint. sDeltaG_tskij outlives the
  // solve and is reused by the next perturbation, so skipping this would hand the
  // checkpoint the previous perturbation's ΔG next to this one's ΔDm/ΔF. With a Σ
  // every pass has already materialized it.
  // The local projection of ΔG(τ) is a consumer as well.
  if (p.save_DeltaG || p.embedding) update_DeltaG();

  // Hessian: the estimator needs the returned ΔF/ΔΣ to be K of the returned
  // ΔDm/ΔG(τ), but ΔF/ΔΣ are the accelerator's iterate and ΔDm/ΔG(τ) its Dyson
  // response. One kernel evaluation at the returned ΔDm/ΔG(τ) makes the pair exact.
  // After update_DeltaG, whose full pass may have replaced ΔDm.
  // A fixed-order run takes the plain hessian only, which reads ΔH0 and ΔDm and so
  // keeps ΔF/ΔΣ as the iterate the final Dyson pass was fed.
  if (p.hessian() && !p.fixed_order) {
    apply_kernel(/*precond=*/false, sDeltaF_skij, pDeltaSigma,
                 sDeltaDm_skij, sDeltaG_tskij, sG_tskij, thc, p,
                 sDeltaSigma_term2_tskij);
  }

  // Report results
  if (p.fixed_order) {
    app_log(1, "\n  LR SCF: fixed order n = {} (no convergence test).", max_iter);
  } else if (converged) {
    app_log(1, "\n  LR SCF converged in {} iterations!", iter);
  } else if (p.max_iter > 1) {
    app_log(1, "\n  [WARNING] LR SCF did NOT converge after {} iterations.", p.max_iter);
  }
  app_log(1, "  Final Δμ = {:.6e}", Delta_mu);

  // Expose the precomputed IBC aux→primary correction
  //   δX†·F_PQ·X + X†·F_PQ·δX   (= ∂_τ Σ^(δX) in band basis)
  // so callers can persist it for downstream gradient evaluations.
  if (DeltaF_ibc_out && opt_ibc && opt_ibc->DeltaF_ibc_skij.size() > 0) {
    *DeltaF_ibc_out = opt_ibc->DeltaF_ibc_skij;
  }

  // Expose F_PQ (unperturbed) and ΔF_PQ (LR Fock at convergence) in aux basis
  // for the Python phonon post-processors, which build the ΔΔF_ibc T1/T3 terms
  // that have no C++ path.
  //
  // The move empties the IBC object, so a second lr_solve_one would find
  // F_PQ_skij.size() == 0 and quietly write nothing. That is one of the reasons
  // callers must reject IBC together with more than one perturbation; lifting
  // that restriction means copying here instead.
  if (F_PQ_out && opt_ibc && opt_ibc->F_PQ_skij.size() > 0) {
    *F_PQ_out = std::move(opt_ibc->F_PQ_skij);
  }
  if (DeltaF_PQ_out && K.has_F()) {
    // One extra ΔF evaluation on the converged ΔDm just to capture ΔF_PQ. It writes
    // into a scratch ΔF rather than sDeltaF_skij: the converged sDeltaF_skij is the
    // mixed (DIIS/damped) iterate the caller persists, and re-evaluating from ΔDm
    // would replace it with a different matrix, so an output-only flag would change
    // the DeltaF_skij dataset.
    auto sDeltaF_scratch = math::shm::make_shared_array<Array_view_4D_t>(
        *_mpi, {_ns, _nkpts_ibz, _nbnd, _nbnd});
    apply_kernel(/*precond=*/false, sDeltaF_scratch, nullptr, sDeltaDm_skij,
                 sDeltaG_tskij, sG_tskij, thc, p, nullptr, DeltaF_PQ_out);
  }

  // Hierarchical timer report for this perturbation (verbosity >= 2). Per-step
  // solver prints inside the loop are gated to verbosity >= 3.
  print_timers(_lr_pi.get(), _lr_scr.get(), _lr_gw.get(), _lr_gw2.get());

  return std::make_tuple(iter, Delta_mu);
}


void lr_driver::print_memory_estimate(long NP, bool any_Sigma, bool needs_dW,
                                      long n_sigma_prev,
                                      lr_diis_hist_t inner_hist,
                                      bool precond,
                                      bool need_Delta_mu,
                                      bool exchange_static_W,
                                      long hessian_nmodes) {
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

  auto gb = [&](double nelem) { return nelem * bytes_per * to_GB; };

  // Pad to a display width counting UTF-8 code points, not bytes, so rows with
  // Δ/ω/Σ in the name line up with the ASCII ones.
  auto pad = [](std::string const& s, size_t w) {
    size_t n = 0;
    for (unsigned char c : s) if ((c & 0xC0) != 0x80) ++n;
    return s + std::string(n < w ? w - n : 0, ' ');
  };

  // Lifetime of an array: resident for the whole run, or scratch allocated and
  // freed within one iteration in either the Dyson or the ΔW/Σ phase.
  enum life_t { PERSIST, T_DYSON, T_GWSIG };
  auto life_str = [](life_t l) {
    return (l == PERSIST) ? "persistent"
         : (l == T_DYSON) ? "transient (Dyson)" : "transient (ΔW/Σ)";
  };

  // {name, shape string, # elements, is_distributed, lifetime}
  struct entry_t { std::string name; std::string shape; double nelem; bool dist; life_t life; };
  std::vector<entry_t> arrays;

  auto band5 = [&](long n0) { return double(n0) * ns * nki * nb * nb; };  // (n0,ns,nk_ibz,nb,nb)
  auto aux4  = [&](long n0) { return double(n0) * nq * NP * NP; };        // (n0,nq,NP,NP)
  auto aux5  = [&](long n0) { return double(n0) * ns * nq * NP * NP; };   // (n0,ns,nq,NP,NP)

  auto shp5b = [&](long n0) { return fmt::format("({},{},{},{},{})", n0, ns, nki, nb, nb); };
  auto shp4a = [&](long n0) { return fmt::format("({},{},{},{})", n0, nq, NP, NP); };
  auto shp5a = [&](long n0) { return fmt::format("({},{},{},{},{})", n0, ns, nq, NP, NP); };

  // --- Persistent, shared (replicated per node), band basis ~ nk·nt·nb² ---
  // sG_tskij is caller-owned but resident throughout the solve, so count it here.
  arrays.push_back({"sG_tskij",       shp5b(nt), band5(nt), false, PERSIST});
  arrays.push_back({"sDeltaG_tskij",  shp5b(nt), band5(nt), false, PERSIST});
  arrays.push_back({"sG_wskij",       shp5b(nw), band5(nw), false, PERSIST});
  if (any_Sigma) {
    arrays.push_back({"sDeltaSigma_tskij", shp5b(nt), band5(nt), false, PERSIST});
  }
  // The preconditioner's gathered residual ΔΣ part.
  if (precond) {
    arrays.push_back({"r_Σ (preconditioner)", shp5b(nt), band5(nt), false, PERSIST});
  }

  // --- Persistent, striped over the global comm (each rank keeps one element
  //     slice of a node-replicated band array) ---
  // Previous iterates for damping/DIIS. ΔF_prev /
  // ΔV_QPGW_prev are ~nb²·nk and negligible next to these; only the ΔΣ-sized
  // ones are worth a row.
  for (long i = 0; i < n_sigma_prev; ++i) {
    arrays.push_back({"ΔΣ_prev (striped)", shp5b(nt), band5(nt), true, PERSIST});
  }

  // Accelerator history (lr_diis ring or lr_gcr basis), n_vec vectors counted by
  // lr_setup. It is striped over the global comm, so the whole job stores it once.
  auto push_hist = [&](lr_diis_hist_t const& h, const char* who) {
    if (h.n_vec <= 0 || (h.n_F == 0 && h.n_Sigma == 0)) return;
    const double nelem = h.n_vec * (h.n_F * band5(1) + h.n_Sigma * band5(nt));
    arrays.push_back({std::string(who),
                      fmt::format("{}x[{}xΔF + {}xΔΣ]", h.n_vec, h.n_F, h.n_Sigma),
                      nelem, true, PERSIST});
  };
  push_hist(inner_hist, _lr_gcr ? "GCR basis" : "DIIS history");

  // Stationary hessian estimator. The ω stores — ΔΣ_λ(iω) and ΔG_p(iω) for every
  // perturbation of the batch — are the largest thing the feature allocates and the
  // only part that scales with the batch. Its extra Dyson solve reuses the mode
  // loop's arrays, so it adds nothing else here.
  if (hessian_nmodes > 0) {
    if (any_Sigma) {
      arrays.push_back({"hessian ω stores (ΔΣ+ΔG)",
                        fmt::format("2x{}x{}", hessian_nmodes, shp5b(nw)),
                        2.0 * double(hessian_nmodes) * band5(nw), true, PERSIST});
    }
  }

  // --- Persistent, distributed, band basis ---
  // ΔG(τ) as lr_dyson hands it over: distributed, and alive from the end of the
  // solve until it is replicated into shared memory. It overlaps neither
  // transient peak in practice — the next solve drops it before allocating
  // anything, and the Dyson pass replicates it (materialize_DeltaG_tau) ahead of
  // any ΔΠ/ΔW — so counting it as persistent is a deliberate over-estimate.
  arrays.push_back({"ΔG(τ) pending (lr_dyson)", shp5b(nt), band5(nt), true, PERSIST});

  // dG/dμ(τ) = ∂ΔG(τ)/∂Δμ and the matching ΔDm response. ΔG is affine in Δμ, so
  // fix_density at q=Γ adds Δμ·dG/dμ instead of a second Dyson pass; both are
  // built once from the reference G(iω) and resident for the run. It is exactly one
  // more ΔG(τ) — the largest single distributed array here — so it is listed even
  // though only this one path allocates it.
  if (need_Delta_mu) {
    arrays.push_back({"dG_dmu(τ) = ∂ΔG/∂Δμ (lr_dyson)", shp5b(nt), band5(nt), true, PERSIST});
    arrays.push_back({"∂Dm/∂μ (lr_dyson)",
                      fmt::format("({},{},{},{})", ns, nki, nb, nb),
                      band5(1), false, PERSIST});
  }

  // --- Persistent, distributed (over global comm), aux basis ~ nk·nt·NP² ---
  if (any_Sigma) {
    arrays.push_back({"dW_tRPQ",       shp4a(nth), aux4(nth), true, PERSIST});
  }
  if (exchange_static_W) {
    arrays.push_back({"dWc0_qPQ (HSEX)", fmt::format("({},{},{})", nq, NP, NP),
                      aux4(1), true, PERSIST});
  }
  if (needs_dW) {
    arrays.push_back({"dW_full_wqPQ",  shp4a(nwbh), aux4(nwbh), true, PERSIST});
    arrays.push_back({"dG_tsRPQ",      shp5a(nth),  aux5(nth),  true, PERSIST});
    arrays.push_back({"dG_mtau_tsRPQ", shp5a(nth),  aux5(nth),  true, PERSIST});
    if (!is_q_gamma())
      arrays.push_back({"_dW_full_qpQ (W(q+Q))", shp4a(nwbh), aux4(nwbh), true, PERSIST});
  }

  // --- Per-iteration transients: scratch arrays (~ nk·nt·n²) allocated and
  //     freed within one SCF iteration, on top of the persistent set.
  //     Two mutually-exclusive phases:
  //       Dyson : ΔG(iω)/ΔΣ(iω) inside lr_dyson (distributed band basis)
  //       ΔW/Σ  : ΔΠ/ΔW(τ) + the FT staging buffers (needs_dW only)
  //     lr_dyson runs before the Π/W/Σ steps and frees its scratch first, so the
  //     two never coexist — the peak adds only the larger of the two phases.
  arrays.push_back({"ΔG(iω) (lr_dyson)", shp5b(nw), band5(nw), true, T_DYSON});
  if (any_Sigma)
    arrays.push_back({"ΔΣ(iω) (lr_dyson)", shp5b(nw), band5(nw), true, T_DYSON});
  if (needs_dW) {
    arrays.push_back({"ΔΠ/ΔW(τ)",       shp4a(nth), aux4(nth), true, T_GWSIG});
    // FT staging buffers, allocated and released inside each tau_to_w/w_to_tau.
    arrays.push_back({"FT buffer (τ)",   shp4a(nth),  aux4(nth),  true, T_GWSIG});
    arrays.push_back({"ΔΠ/ΔW(iω)",       shp4a(nwbh), aux4(nwbh), true, T_GWSIG});
  }

  // Shared / distributed totals, per lifetime.
  double shared_GB = 0.0, dist_GB = 0.0;          // persistent
  double dy_sh = 0.0, dy_di = 0.0;                // transient, Dyson phase
  double gw_sh = 0.0, gw_di = 0.0;                // transient, ΔW/Σ phase
  for (auto const& a : arrays) {
    double g = gb(a.nelem);
    switch (a.life) {
      case PERSIST: (a.dist ? dist_GB : shared_GB) += g; break;
      case T_DYSON: (a.dist ? dy_di   : dy_sh)     += g; break;
      case T_GWSIG: (a.dist ? gw_di   : gw_sh)     += g; break;
    }
  }
  double dist_per_node_GB = dist_GB / n_nodes;
  double total_per_node_GB = shared_GB + dist_per_node_GB;

  double dy_per_node = dy_sh + dy_di / n_nodes;
  double gw_per_node = gw_sh + gw_di / n_nodes;
  bool dyson_dominates = dy_per_node >= gw_per_node;
  double pk_sh        = dyson_dominates ? dy_sh : gw_sh;
  double pk_di        = dyson_dominates ? dy_di : gw_di;
  double pk_per_node  = dyson_dominates ? dy_per_node : gw_per_node;
  double peak_per_node_GB = total_per_node_GB + pk_per_node;

  // Level-2 breakdown (printed before the level-1 totals).
  app_log(2, "\n  LR memory estimate (arrays ~ nk·nt·n², n ∈ {{nbnd={}, NP={}}})", nb, NP);
  app_log(2, "  {}", std::string(94, '-'));
  app_log(2, "    {}{}{:>8s}   {}{}", pad("quantity", 26), pad("shape", 26), "GB",
          pad("location", 14), "lifetime");
  for (auto const& a : arrays)
    app_log(2, "    {}{}{:>8.3f}   {}{}", pad(a.name, 26), pad(a.shape, 26), gb(a.nelem),
            pad(a.dist ? "distributed" : "shared", 14), life_str(a.life));
  app_log(2, "  {}", std::string(94, '-'));
  app_log(2, "    persistent shared:      {:9.3f} GB/node  (replicated on each of {} node(s))",
          shared_GB, n_nodes);
  app_log(2, "    persistent distributed: {:9.3f} GB/node  (x {} node(s) = {:.3f} GB)",
          dist_per_node_GB, n_nodes, dist_GB);
  app_log(2, "    peak transient:         {:9.3f} GB/node  (x {} node(s) = {:.3f} GB) [{} dominates]",
          pk_di / n_nodes, n_nodes, pk_di, dyson_dominates ? "Dyson" : "ΔW/Σ");
  if (pk_sh > 0.0)
    app_log(2, "                            {:9.3f} GB/node  (replicated on each of {} node(s))",
            pk_sh, n_nodes);

  app_log(1, "  Estimated LR memory (persistent): {:.3f} GB/node", total_per_node_GB);
  app_log(1, "  Estimated LR memory (peak):       {:.3f} GB/node", peak_per_node_GB);
  app_log(2, "");
}



void lr_driver::print_distribution_summary(long NP, bool any_Sigma, bool needs_dW) {
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
  if (any_Sigma) {
    const char* arrs = needs_dW ? "dW_tRPQ, dG_tsRPQ, dG_mtau, ΔΠ/ΔW" : "dW_tRPQ";
    app_log(2, "    {:<22s}{:<30s}{}", "aux τ-dist (q-local)",
            pg4(tau_pg, "(t,q,P,Q)"), arrs);
  }
  // Aux q-dist (ω-side) — only the full-GW W Dyson pipeline. One row: the FT
  // staging buffers and W(iω) share a distribution, which is what lets both
  // transforms fuse. The (P, Q) block is reported because it is the SLATE tile the
  // ω-side Dyson runs on.
  if (needs_dW) {
    auto [w_pg, w_bs] = utils::lr_W_tau_local_dist(nproc, nwbh, nq, NP);
    const char* arrs = is_q_gamma() ? "FT staging buffers, dW_full_wqPQ"
                                    : "FT staging buffers, dW_full_wqPQ, _dW_full_qpQ";
    app_log(2, "    {:<22s}{:<30s}{}", "aux q-dist (ω-side)",
            pg4(w_pg, "(w,q,P,Q)"),
            fmt::format("{}; (P,Q) block {}x{}", arrs, w_bs[2], w_bs[3]));
  }

  // Band-basis Dyson grids, from the helpers solve_lr_dyson_impl allocates with.
  auto [dyw_pg, dyw_bs] = lr_dyson_omega_pgrid(nproc, nw, nki, _nbnd);
  (void)dyw_bs;
  auto dyt_pg = lr_dyson_tau_pgrid(nproc, nki);
  app_log(2, "    {:<22s}{:<30s}{}", "band Dyson(ω)",
          pg5(dyw_pg, "(w,s,k,i,j)"),
          any_Sigma ? "ΔG(iω), ΔΣ(iω)" : "ΔG(iω)");
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
  app_log(2, "      - W_c(iν=0) (HSEX):       {0:8.3f} sec  {1:4d} calls", _Timer.elapsed("LR_DRIVER_SETUP_WC0"), _Timer.number_of_calls("LR_DRIVER_SETUP_WC0"));
  app_log(2, "      - G(iω) precompute:       {0:8.3f} sec  {1:4d} calls", _Timer.elapsed("LR_DRIVER_SETUP_G_OMEGA"), _Timer.number_of_calls("LR_DRIVER_SETUP_G_OMEGA"));
  app_log(2, "      - G^R pair precompute:    {0:8.3f} sec  {1:4d} calls", _Timer.elapsed("LR_DRIVER_SETUP_G_R"), _Timer.number_of_calls("LR_DRIVER_SETUP_G_R"));
  app_log(2, "      - Δμ response precompute: {0:8.3f} sec  {1:4d} calls", _Timer.elapsed("LR_DRIVER_SETUP_DELTAMU"), _Timer.number_of_calls("LR_DRIVER_SETUP_DELTAMU"));
  app_log(2, "      - Alloc:                  {0:8.3f} sec  {1:4d} calls", _Timer.elapsed("LR_DRIVER_SETUP_ALLOC"), _Timer.number_of_calls("LR_DRIVER_SETUP_ALLOC"));
  app_log(2, "      - Build IBC (DeltaX):     {0:8.3f} sec  {1:4d} calls\n", _Timer.elapsed("LR_DRIVER_SETUP_IBC"), _Timer.number_of_calls("LR_DRIVER_SETUP_IBC"));
}



void lr_driver::print_timers(solvers::lr_rpa_pi* pi_solver,
                             solvers::lr_scr_coulomb_t* scr_solver,
                             solvers::lr_gw* gw_solver,
                             solvers::lr_gw* gw_solver_term2) {
  // Driver totals in execution order, each followed by the corresponding
  // solver's subclocks (deeper indent). Solver pointers may be null when the
  // step was not active; subclocks are then skipped.
  //
  // The report has the same shape for every kernel: every line sums over the
  // instances a step uses (the preconditioner's lr_hf, split-output's second
  // lr_gw). Lines a given run never exercises print as 0.000 sec / 0 calls rather
  // than disappearing, so two runs' reports stay directly comparable line by line.
  const std::string sub_indent = "        ";
  app_log(2, "\n  LR_DRIVER timers");
  app_log(2, "  -----------------");
  // The setup is a sibling of the SCF loop, not a part of it, paid once for all
  // perturbations; every other clock accumulates over the perturbations solved
  // so far.
  app_log(2, "    LR Driver Setup (once):     {0:8.3f} sec  {1:4d} calls", _Timer.elapsed("LR_DRIVER_SETUP"), _Timer.number_of_calls("LR_DRIVER_SETUP"));
  app_log(2, "    LR reset (per solve):       {0:8.3f} sec  {1:4d} calls", _Timer.elapsed("LR_RESET"), _Timer.number_of_calls("LR_RESET"));
  app_log(2, "    Total LR SCF:               {0:8.3f} sec  {1:4d} calls", _Timer.elapsed("LR_SCF"), _Timer.number_of_calls("LR_SCF"));
  app_log(2, "      - LR Dyson (total):       {0:8.3f} sec  {1:4d} calls", _Timer.elapsed("LR_DYSON"), _Timer.number_of_calls("LR_DYSON"));
  _lr_dyson.print_subclocks(2, sub_indent);
  app_log(2, "      - LR HF (total):          {0:8.3f} sec  {1:4d} calls", _Timer.elapsed("LR_HF"), _Timer.number_of_calls("LR_HF"));
  if (_lr_hf or _lr_hf_pc)
    solvers::lr_hf::print_subclocks_all(2, sub_indent, {_lr_hf.get(), _lr_hf_pc.get()});
  app_log(2, "      - LR GW Pi (total):       {0:8.3f} sec  {1:4d} calls", _Timer.elapsed("LR_GW_PI"), _Timer.number_of_calls("LR_GW_PI"));
  if (pi_solver) pi_solver->print_subclocks(2, sub_indent);
  app_log(2, "      - LR GW W (total):        {0:8.3f} sec  {1:4d} calls", _Timer.elapsed("LR_GW_W"), _Timer.number_of_calls("LR_GW_W"));
  if (scr_solver) scr_solver->print_subclocks(2, sub_indent);
  app_log(2, "      - LR GW Sigma (total):    {0:8.3f} sec  {1:4d} calls", _Timer.elapsed("LR_GW_SIGMA"), _Timer.number_of_calls("LR_GW_SIGMA"));
  // Summed over both Σ evaluators.
  if (gw_solver or gw_solver_term2)
    solvers::lr_gw::print_subclocks_all(2, sub_indent, {gw_solver, gw_solver_term2});
  app_log(2, "      - LR qpGW static (total): {0:8.3f} sec  {1:4d} calls", _Timer.elapsed("LR_QPGW_STATIC"), _Timer.number_of_calls("LR_QPGW_STATIC"));
  // The GCR preconditioner runs inside the iteration algorithm's step; its Dyson
  // passes and K_pc evaluations are already in the Dyson and HF lines above.
  app_log(2, "      - LR Iter Alg (total):    {0:8.3f} sec  {1:4d} calls",
          _Timer.elapsed("LR_ITER_ALG") - _Timer.elapsed("LR_PRECOND"),
          _Timer.number_of_calls("LR_ITER_ALG"));
  app_log(2, "      - LR Precond (total):     {0:8.3f} sec  {1:4d} calls  "
             "(its Dyson/HF counted above)",
          _Timer.elapsed("LR_PRECOND"), _Timer.number_of_calls("LR_PRECOND"));
  app_log(2, "      - LR Save (prev arrays):  {0:8.3f} sec  {1:4d} calls", _Timer.elapsed("LR_SAVE"), _Timer.number_of_calls("LR_SAVE"));
  app_log(2, "      - LR Convergence (norms): {0:8.3f} sec  {1:4d} calls", _Timer.elapsed("LR_CONVERGENCE"), _Timer.number_of_calls("LR_CONVERGENCE"));
  app_log(2, "");
  // The hessian clocks are reported by lr_hessian_t::print_timers.
}


} // namespace methods
