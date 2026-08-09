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
#include "methods/SCF/scf_common.hpp"
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
                  "LR_DYSON", "LR_HF", "LR_GW_SIGMA",
                   "LR_GW_PI", "LR_GW_W", "LR_GW_SIGMA_TERM2", "LR_QPGW_STATIC",
                   "LR_ITER_ALG", "LR_SAVE", "LR_CONVERGENCE", "LR_TOTALS",
                   // Perturbative channel of a split-kernel run. Separate clocks
                   // because both channels call the same evaluators and the cost
                   // argument for the split is exactly the sc/pert breakdown.
                   "LR_HF_PERT", "LR_GW_SIGMA_PERT",
                   "LR_GW_PI_PERT", "LR_GW_W_PERT",
                   // Mixing of the outer (perturbative-source) iteration.
                   "LR_OUTER_ITER_ALG"}) {
    _Timer.add(v);
  }
  _mpi->comm.barrier();
}


std::string lr_kernel_spec::to_string() const {
  std::string s;
  auto add = [&](const char* c) { if (!s.empty()) s += ", "; s += c; };
  if (hartree)    add("H");
  if (exchange)   add("X");
  if (sigma_dG_W) add("Σ1");
  if (sigma_G_dW) add("Σ2");
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


lr_kernel_spec kernel_diff(lr_kernel_spec const& total, lr_kernel_spec const& sc) {
  utils::check(total.contains(sc),
               "kernel_diff: the self-consistent kernel ({}) is not a subset of "
               "the total kernel ({}).", sc.to_string(), total.to_string());
  return lr_kernel_spec{total.hartree    && !sc.hartree,
                        total.exchange   && !sc.exchange,
                        total.sigma_dG_W && !sc.sigma_dG_W,
                        total.sigma_G_dW && !sc.sigma_G_dW};
}


// Distribution flow through the LR-GW pipeline
// Three distribution patterns:
//   τ-dist (q-local):  pgrid = {tpools, 1, np_P, np_Q}  — q undivided
//   τ-local:           pgrid = {1, nqpools, np_P, np_Q}  — tau/omega undivided (FT buffer)
//   ω-side:            pgrid = {nwpools, nqpools, np_P, np_Q}  — distributed over (w, q, P, Q)
//
//   W_c in:              (t,q,P,Q), τ-dist — one copy, consumed by both setup steps
//   compute_W_full_omega: τ-dist → FT(τ→ω) → ω-side, + Z(q) → W_full(iω) [cached]
//   lr_precompute_W_tRPQ: q→R in place → (t,R,P,Q), τ-dist [cached]
//
//   evaluate_lr_Pi:      → (t,q,P,Q), τ-dist
//   solve_lr_dyson_W (in-place):
//     tau_to_w:          τ-dist → ω-side (via q-distributed FT buffer)
//     lr_dyson_W_in_place: ω-side; SLATE GEMM batched over (iw, iq).
//                          For Q≠Γ, gathers W_full(kpq_map(iq)) via Alltoallv on q_pool_comm.
//     w_to_tau:          ω-side → τ-dist (via q-distributed FT buffer)
//     output:            τ-dist (overwrites input)
//   evaluate_sigma_*:    τ-dist in (t,q) order, i.e. ΔW is consumed as produced

template<THC_ERI THC_t, typename dW_t>
std::tuple<int, double> lr_driver::run_lr(
    sArray_t<Array_view_5D_t>& sDeltaG_tskij,
    sArray_t<Array_view_4D_t>& sDeltaDm_skij,
    sArray_t<Array_view_4D_t>& sDeltaF_skij,
    sArray_t<Array_view_5D_t>* sDeltaSigma_tskij,
    const sArray_t<Array_view_5D_t>& sG_tskij,
    const sArray_t<Array_view_4D_t>& sDeltaH0_skij,
    THC_t& thc,
    lr_kernel_spec sc_kernel, lr_kernel_spec pert_kernel, int pert_order,
    dW_t* dW_tqPQ_in, const nda::array<ComplexType, 1>* eps_inv_head,
    int max_iter, double tol, bool fix_density,
    const lr_iter_params& iter_params,
    const sArray_t<Array_view_4D_t>* sDeltaX_left,
    const sArray_t<Array_view_4D_t>* sDeltaX_right,
    const nda::array<ComplexType, 4>* Dm_ab,
    bool div_corr,
    std::string div_treatment,
    const nda::array_view<ComplexType, 3>* DeltaV_qPQ,
    sArray_t<Array_view_5D_t>* sDeltaSigma_term2_tskij,
    const lr_qp_static_params* qp_static,
    sArray_t<Array_view_4D_t>* sDeltaVcorr_out_skij,
    nda::array<ComplexType, 4>* DeltaF_ibc_out,
    nda::array<ComplexType, 4>* F_PQ_out,
    nda::array<ComplexType, 4>* DeltaF_PQ_out,
    bool include_xc,
    const lr_outer_accel_params* outer_accel,
    int* n_pert_applied_out) {

  _Timer.start("LR_SCF");

  // One-shot G0W0: store the two ΔΣ terms separately instead of fusing them.
  // ΔΣ term1 (-ΔG⊙W_c) → sDeltaSigma_tskij, ΔΣ term2 (-G⊙ΔW) → sDeltaSigma_term2.
  bool split_sigma_terms = (sDeltaSigma_term2_tskij != nullptr);

  // Split-kernel (two-step) schedule: K = K_sc + K_pert, with K_sc resummed to
  // all orders by the inner SCF and K_pert applied `pert_order` times, each on a
  // converged inner solve. An empty K_pert or pert_order = 0 leaves `pert` empty
  // and the loop below is the single-kernel path it has always been.
  utils::check(pert_order >= 0,
               "lr_driver::run_lr: pert_order must be >= 0, got {}.", pert_order);
  utils::check(pert_order == 0 || !pert_kernel.empty(),
               "lr_driver::run_lr: pert_order = {} >= 1 requires a non-empty "
               "perturbative kernel.", pert_order);
  const bool do_pert = (pert_order >= 1) && !pert_kernel.empty();
  const lr_kernel_spec pert = do_pert ? pert_kernel : lr_kernel_spec{};
  if (do_pert) {
    utils::check(!sc_kernel.overlaps(pert),
                 "lr_driver::run_lr: the self-consistent ({}) and perturbative "
                 "({}) kernels share components. pert_kernel must be the "
                 "DIFFERENCE kernel(total) \\ kernel(sc), not the total kernel — "
                 "a shared component would be applied twice, once resummed and "
                 "once as a frozen source. Build it with kernel_diff().",
                 sc_kernel.to_string(), pert.to_string());
    utils::check(sDeltaX_left == nullptr && sDeltaX_right == nullptr,
                 "lr_driver::run_lr: the split-kernel schedule does not support "
                 "the DeltaX IBC correction (each kernel evaluator adds its own "
                 "IBC term, so a two-pass schedule would double-count it).");
    utils::check(DeltaV_qPQ == nullptr,
                 "lr_driver::run_lr: the split-kernel schedule does not support "
                 "the DeltaV_qPQ perturbation.");
    utils::check(!include_xc,
                 "lr_driver::run_lr: the split-kernel schedule is incompatible "
                 "with include_xc.");
    utils::check(!split_sigma_terms,
                 "lr_driver::run_lr: the split-kernel schedule is incompatible "
                 "with split ΔΣ terms.");
    utils::check(qp_static == nullptr,
                 "lr_driver::run_lr: the split-kernel schedule is incompatible "
                 "with qp_static mode.");
  }

  // Outer-loop acceleration. The defaults (alg "damping", tol = 0) switch every
  // flag below off, so nothing is allocated and the outer loop is the plain
  // Neumann series.
  const std::string outer_alg = outer_accel ? outer_accel->iter.alg : "damping";
  const double outer_tol      = outer_accel ? outer_accel->tol : 0.0;
  const bool outer_diis_on = (outer_alg == "DIIS");
  // "the outer loop is no longer an order-pert_order truncation": drives the
  // logging, the ΔDm stage buffer and the checkpoint provenance fields.
  const bool outer_track = outer_accel && outer_accel->active();
  if (outer_accel) {
    utils::check(outer_alg == "damping" || outer_alg == "DIIS",
                 "lr_driver::run_lr: unknown outer iter_alg '{}'. Must be "
                 "'damping' or 'DIIS'.", outer_alg);
    utils::check(outer_tol >= 0.0,
                 "lr_driver::run_lr: the outer tolerance must be >= 0, got {}.",
                 outer_tol);
    utils::check(!outer_track || do_pert,
                 "lr_driver::run_lr: outer-loop acceleration requires a "
                 "split-kernel run (pert_order >= 1 with a non-empty K_pert). "
                 "There is no outer sequence to accelerate otherwise.");
  }

  // Per-channel component flags. Every derived quantity below is the union of
  // the two channels; the per-iteration dispatch uses the per-channel flags.
  const bool sc_hf      = sc_kernel.hartree || sc_kernel.exchange || include_xc;
  const bool pert_hf    = pert.hartree || pert.exchange;
  const bool sc_sigma   = sc_kernel.has_sigma();
  const bool pert_sigma = pert.has_sigma();

  bool need_hf = sc_hf || pert_hf;
  bool include_gw_sigma = sc_sigma || pert_sigma;
  bool gw_full = sc_kernel.sigma_G_dW || pert.sigma_G_dW;
  bool need_W_tRPQ = sc_kernel.sigma_dG_W || pert.sigma_dG_W;
  // LR-qpGW static-map mode: statify the dynamic ΔΣ(iω) into a static ΔV_QPGW(k)
  // each iteration (frozen orbitals), feed it into the Dyson RHS in place of the
  // dynamic ΔΣ, and track it for DIIS/damping/convergence.
  bool qp_mode = (qp_static != nullptr);
  // Which quantity is tracked/mixed in the Dyson RHS alongside ΔF:
  //   has_Vcorr — static ΔV_QPGW (qp mode) is the mixed/tracked quantity
  //   has_Sigma — dynamic ΔΣ(iω) (standard GW path) is the mixed/tracked quantity
  // include_gw_sigma still gates construction/compute of the GW pipeline (qpGW
  // computes ΔΣ before statifying it into ΔV_QPGW).
  bool has_Vcorr = qp_mode;
  // The ΔΣ fed to the Dyson equation is the total (sc + pert) one; the ΔΣ that
  // is mixed and tracked for convergence is the sc-channel one only.
  bool has_Sigma = include_gw_sigma && !qp_mode;
  bool has_Sigma_sc = sc_sigma && !qp_mode;

  // A quantity carried by only ONE channel needs no per-channel buffer: that
  // channel writes the caller's total array directly and no sum is ever formed.
  // Only a quantity both channels contribute to is "split", and its total is
  // rebuilt each iteration. On the method ladder that is ΔF for pGW_Hartree and
  // ΔΣ for pGW_GW0; pGW_HF / pGW_none split neither.
  const bool split_F     = sc_hf && pert_hf;
  const bool split_Sigma = sc_sigma && pert_sigma;

  utils::check(iter_params.alg == "damping" || iter_params.alg == "DIIS",
               "lr_driver::run_lr: unknown iter_alg '{}'. Must be 'damping' or 'DIIS'.",
               iter_params.alg);
  if (include_gw_sigma) {
    utils::check(sDeltaSigma_tskij != nullptr,
                 "lr_driver::run_lr: a GW self-energy is active but sDeltaSigma_tskij is null.");
    utils::check(dW_tqPQ_in != nullptr && eps_inv_head != nullptr,
                 "lr_driver::run_lr: a GW self-energy is active but dW_tqPQ or eps_inv_head is null.");
  }
  if (split_sigma_terms) {
    utils::check(gw_full,
                 "lr_driver::run_lr: split ΔΣ terms require the Σ2 (-G⊙ΔW) component.");
    utils::check(max_iter == 1,
                 "lr_driver::run_lr: split ΔΣ terms are only meaningful for a "
                 "one-shot solve (max_iter=1), got max_iter={}.", max_iter);
    utils::check(sDeltaX_left == nullptr && sDeltaX_right == nullptr,
                 "lr_driver::run_lr: split ΔΣ terms do not support the DeltaX IBC "
                 "correction (term 2 has no IBC path).");
  }
  if (qp_mode) {
    utils::check(include_gw_sigma,
                 "lr_driver::run_lr: qp_static mode requires a GW self-energy.");
    utils::check(!split_sigma_terms,
                 "lr_driver::run_lr: qp_static mode is incompatible with split ΔΣ terms.");
    utils::check(qp_static->sMO_skia != nullptr && qp_static->sE_ska != nullptr,
                 "lr_driver::run_lr: qp_static mode requires sMO_skia and sE_ska.");
  }

  bool use_diis = (iter_params.alg == "DIIS");
  double mixing = iter_params.mixing;

  app_log(1, "Starting Linear Response SCF loop:");
  app_log(1, "  max_iter = {}", max_iter);
  app_log(1, "  tol = {:.2e}", tol);
  app_log(1, "  fix_density = {}", fix_density ? "true" : "false");
  app_log(1, "  K_sc  (self-consistent) = {}", sc_kernel.to_string());
  if (do_pert) {
    app_log(1, "  K_pert (perturbative)   = {}", pert.to_string());
    app_log(1, "  pert_order = {}  ({} stage(s); max_iter counts total inner iterations)",
            pert_order, pert_order + 1);
  }
  app_log(1, "  include_xc = {}", include_xc ? "true" : "false");
  app_log(1, "  qp_static_sigma = {}", qp_mode ? "true" : "false");
  app_log(1, "  iter_alg = {}", iter_params.alg);
  app_log(1, "  mixing = {:.2f}", mixing);
  if (use_diis) {
    app_log(1, "  max_subsp_size = {}", iter_params.max_subsp_size);
    app_log(1, "  diis_warmup = {}", iter_params.diis_warmup);
  }
  if (outer_track) {
    app_log(1, "  outer (K_pert source) acceleration:");
    app_log(1, "    outer_alg = {}", outer_alg);
    if (outer_diis_on)
      app_log(1, "    outer_subsp = {}, outer_warmup = {}, outer_min_subsp = {} "
                 "(first extrapolation at outer step {})",
              outer_accel->iter.max_subsp_size, outer_accel->iter.diis_warmup,
              outer_accel->min_subsp,
              std::max(outer_accel->iter.diis_warmup + 2, outer_accel->min_subsp));
    app_log(1, "    outer_tol = {:.2e}", outer_tol);
    if (outer_diis_on) {
      app_log(1, "    [NOTE] with outer acceleration the source is a combination "
                 "of previous sources, so the result is NOT an order-{} "
                 "truncation of K_pert: it is an accelerated iterate toward the "
                 "FULL K_sc + K_pert fixed point, and pert_order is an iteration "
                 "cap.", pert_order);
    }
    if (outer_tol > 0.0 && tol > 0.1 * outer_tol) {
      app_log(1, "    [WARNING] the inner tol ({:.2e}) is not at least a decade "
                 "below outer_tol ({:.2e}); the outer residual then measures "
                 "inner-solve noise rather than the outer error.", tol, outer_tol);
    }
  }

  // ΔΣ-sized shared arrays on top of the total ΔΣ the base estimate already
  // lists.
  std::vector<std::string> extra_sigma;
  if (split_Sigma) {
    extra_sigma.push_back("sDeltaSigma (sc channel)");
    extra_sigma.push_back("sDeltaSigma (pert channel)");
  }
  if (outer_diis_on && pert_sigma)
    extra_sigma.push_back("sDeltaSigma_pert_prev (outer)");

  // DIIS histories, for the memory report. Each subspace entry stores a trial
  // AND a residual vector of every quantity the accelerator mixes.
  lr_diis_hist_t inner_hist, outer_hist;
  if (use_diis && (sc_hf || sc_sigma || has_Vcorr)) {
    inner_hist.depth = static_cast<long>(iter_params.max_subsp_size);
    inner_hist.n_F = has_Vcorr ? 2 : 1;   // ΔF (+ the static ΔV_QPGW in qp mode)
    inner_hist.n_Sigma = has_Sigma_sc ? 1 : 0;
  }
  if (outer_diis_on) {
    outer_hist.depth = static_cast<long>(outer_accel->iter.max_subsp_size);
    outer_hist.n_F     = pert_hf ? 1 : 0;
    outer_hist.n_Sigma = pert_sigma ? 1 : 0;
  }

  // Estimate the persistent large-array memory footprint for this path, then
  // summarize the MPI distribution patterns the large arrays use.
  print_memory_estimate(thc.Np(), include_gw_sigma, gw_full, extra_sigma,
                        inner_hist, outer_hist);
  print_distribution_summary(thc.Np(), include_gw_sigma, gw_full);

  // Force dW_tqPQ onto the canonical LR q-local distribution. The fused ΔΣ loop
  // pairs the P/Q tile of W_c (from dW_tqPQ → dW_tRPQ) with that of ΔW and the
  // G^R cache, both built via lr_W_q_local_dist. When dW_tqPQ arrives on a
  // different tiling — e.g. a W_c recomputed through the scGW pipeline inherits
  // the polarizability's block_size={1,1,1,1} — the contiguous PQ split disagrees
  // whenever Np % np_P != 0 (bsize=1 vs bsize=Np/np_P round differently), and
  // the fused pairing aborts. Redistributing here makes all operands share one
  // tiling by construction. (The from-file G0W0 path already builds dW_tqPQ on
  // this distribution, so the redistribute is a no-op there.)
  if (include_gw_sigma) {
    long nt_f = sG_tskij.shape()[0];
    long nt_half = (nt_f % 2 == 0) ? nt_f / 2 : nt_f / 2 + 1;
    auto [tq_pgrid, tq_bsize] =
        utils::lr_W_q_local_dist(_mpi->comm.size(), nt_half, thc.Np());
    if (dW_tqPQ_in->grid() != tq_pgrid || dW_tqPQ_in->block_size() != tq_bsize) {
      app_log(2, "lr_driver::run_lr: redistributing dW_tqPQ onto canonical "
                 "LR q-local tiling (pgrid ({},{},{},{}), bsize ({},{},{},{}))",
              tq_pgrid[0], tq_pgrid[1], tq_pgrid[2], tq_pgrid[3],
              tq_bsize[0], tq_bsize[1], tq_bsize[2], tq_bsize[3]);
      math::nda::redistribute_in_place(*dW_tqPQ_in, tq_pgrid, tq_bsize);
    }
  }

  // Initialize lr_hf solver if needed
  if (need_hf && !_lr_hf) {
    _lr_hf = std::make_unique<solvers::lr_hf>(_mpi, _MF, _lr_dyson.q_vec());
  }

  // Create lr_gw solver if needed (local to this call, no need to store as member)
  std::unique_ptr<solvers::lr_gw> lr_gw_solver;
  if (sc_sigma) {
    lr_gw_solver = std::make_unique<solvers::lr_gw>(_dyson.FT(), _lr_dyson.q_vec(), div_treatment);
  }
  // The perturbative channel gets its own lr_gw: the cached workspace is keyed
  // on the (term1, term2) combination it was first used with, and the two
  // channels generally run different combinations (e.g. Σ1 in K_sc, Σ2 in K_pert).
  std::unique_ptr<solvers::lr_gw> lr_gw_solver_pert;
  if (pert_sigma) {
    lr_gw_solver_pert = std::make_unique<solvers::lr_gw>(_dyson.FT(), _lr_dyson.q_vec(), div_treatment);
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
  // Both W_c consumers below read the same (t,q,P,Q) array the caller handed in:
  // compute_W_full_omega FTs it τ→ω, then lr_precompute_W_tRPQ turns it
  // into (t,R,P,Q). So the FT must not release it — hence reset_input=false — and
  // the R-space step needs no copy of its own.
  if (gw_full) {
    _Timer.start("LR_DRIVER_SETUP_W_FULL");
    opt_dW_full_wqPQ.emplace(
        lr_scr_solver->compute_W_full_omega(*dW_tqPQ_in, thc, /*reset_input=*/false));
    _Timer.stop("LR_DRIVER_SETUP_W_FULL");
  }

  // Precompute W in R-space: q→R FT in place on the (t,q) input.
  // Result: dW_tRPQ with (t,R,P,Q) layout, pgrid (tpools,1,np_P,np_Q).
  std::optional<dW_t> opt_dW_tRPQ;
  if (need_W_tRPQ) {
    _Timer.start("LR_DRIVER_SETUP_W_TRPQ");
    opt_dW_tRPQ.emplace(lr_precompute_W_tRPQ(*dW_tqPQ_in, thc));
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
  // Element partition of the node-replicated band arrays over the *global*
  // comm: the DIIS history, the "previous iterate" copies and their norms are
  // all elementwise, so each rank handles one slice and the whole job stores
  // each of those quantities once instead of once per node.
  auto pmap = utils::make_part_map(*_mpi);

  // Allocate array for previous density matrix (for convergence check).
  // Kept whole (0.1 GB) — it only feeds a norm, and the ΔDm norm stays on the
  // node_comm path.
  auto sDeltaDm_prev_skij = math::shm::make_shared_array<Array_view_4D_t>(
      *_mpi, {_ns, _nkpts_ibz, _nbnd, _nbnd});

  // Previous ΔF / ΔΣ / ΔV_QPGW (for damping/DIIS and the difference norms) are
  // rank-private slices of the flattened arrays, in the `pmap` partition. Only
  // the sc channel is mixed, so the tracked ΔΣ is that channel's.
  // Previous ΔΣ only needed when the sc channel carries a Σ and we are not in qp
  // mode (in qp mode the dynamic ΔΣ is not tracked; the static ΔV_QPGW is).
  const long nF_flat = _ns * _nkpts_ibz * _nbnd * _nbnd;
  const long nS_flat = has_Sigma_sc ? _nts * nF_flat : 0;
  const long nV_flat = has_Vcorr ? nF_flat : 0;
  auto [iF0, iF1] = pmap.my_slice(nF_flat);
  auto [iS0, iS1] = pmap.my_slice(nS_flat);
  auto [iV0, iV1] = pmap.my_slice(nV_flat);
  nda::array<ComplexType, 1> DeltaF_prev(iF1 - iF0);
  nda::array<ComplexType, 1> DeltaSigma_prev(iS1 - iS0);
  nda::array<ComplexType, 1> DeltaVcorr_prev(iV1 - iV0);

  // Static ΔV_QPGW tracked in qp mode.
  auto sDeltaVcorr_skij = has_Vcorr
      ? math::shm::make_shared_array<Array_view_4D_t>(*_mpi, {_ns, _nkpts_ibz, _nbnd, _nbnd})
      : math::shm::make_shared_array<Array_view_4D_t>(*_mpi, {1, 1, 1, 1});

  // Split-kernel channel buffers. sDeltaF_skij / sDeltaSigma_tskij always hold
  // the TOTAL (sc + pert) quantities that the Dyson RHS and the checkpoint dump
  // consume. A per-channel buffer exists only for a *split* quantity; otherwise
  // the sole contributing channel writes the caller's array directly, so the
  // single-kernel path — and every composition that splits only one of the two —
  // forms no sum at all.
  std::optional<sArray_t<Array_view_4D_t>> opt_sDeltaF_sc, opt_sDeltaF_pert;
  std::optional<sArray_t<Array_view_5D_t>> opt_sDeltaSigma_sc, opt_sDeltaSigma_pert;
  if (split_F) {
    opt_sDeltaF_sc.emplace(math::shm::make_shared_array<Array_view_4D_t>(
        *_mpi, {_ns, _nkpts_ibz, _nbnd, _nbnd}));
    opt_sDeltaF_pert.emplace(math::shm::make_shared_array<Array_view_4D_t>(
        *_mpi, {_ns, _nkpts_ibz, _nbnd, _nbnd}));
  }
  if (split_Sigma) {
    opt_sDeltaSigma_sc.emplace(math::shm::make_shared_array<Array_view_5D_t>(
        *_mpi, {_nts, _ns, _nkpts_ibz, _nbnd, _nbnd}));
    opt_sDeltaSigma_pert.emplace(math::shm::make_shared_array<Array_view_5D_t>(
        *_mpi, {_nts, _ns, _nkpts_ibz, _nbnd, _nbnd}));
  }
  // Per-channel write targets. When a quantity is not split, both channels
  // resolve to the caller's array (at most one of them is ever active).
  auto& sDeltaF_sc_skij   = split_F ? *opt_sDeltaF_sc   : sDeltaF_skij;
  auto& sDeltaF_pert_skij = split_F ? *opt_sDeltaF_pert : sDeltaF_skij;
  sArray_t<Array_view_5D_t>* pDeltaSigma_sc =
      !sc_sigma ? nullptr : (split_Sigma ? &(*opt_sDeltaSigma_sc) : sDeltaSigma_tskij);
  sArray_t<Array_view_5D_t>* pDeltaSigma_pert =
      !pert_sigma ? nullptr : (split_Sigma ? &(*opt_sDeltaSigma_pert) : sDeltaSigma_tskij);

  // Outer-loop buffers. The previous-source pair exists only for the
  // accelerator (it is what the extrapolation and its residual are measured
  // against); the ΔDm stage buffer only for the tolerance test. A
  // tolerance-only run therefore never allocates a ΔΣ-sized buffer.
  std::optional<sArray_t<Array_view_4D_t>> opt_sDeltaF_pert_prev;
  std::optional<sArray_t<Array_view_5D_t>> opt_sDeltaSigma_pert_prev;
  std::optional<sArray_t<Array_view_4D_t>> opt_sDeltaDm_stage_prev;
  if (outer_diis_on) {
    if (pert_hf)
      opt_sDeltaF_pert_prev.emplace(math::shm::make_shared_array<Array_view_4D_t>(
          *_mpi, {_ns, _nkpts_ibz, _nbnd, _nbnd}));
    if (pert_sigma)
      opt_sDeltaSigma_pert_prev.emplace(math::shm::make_shared_array<Array_view_5D_t>(
          *_mpi, {_nts, _ns, _nkpts_ibz, _nbnd, _nbnd}));
  }
  if (outer_tol > 0.0)
    opt_sDeltaDm_stage_prev.emplace(math::shm::make_shared_array<Array_view_4D_t>(
        *_mpi, {_ns, _nkpts_ibz, _nbnd, _nbnd}));
  _Timer.stop("LR_DRIVER_SETUP_ALLOC");

  // Flattened views of the current arrays, sliced to this rank's partition —
  // the save step, the mixing and the norms all work on these.
  auto flat_slice = [](auto&& A, long n, long i0, long i1) {
    return nda::reshape(A, std::array<long, 1>{n})(nda::range(i0, i1));
  };

  _Timer.start("LR_DRIVER_SETUP_MISC");
  // Initialize ΔF = 0 (and ΔΣ = 0 if GW active); set_zero ends with fence + node_sync
  sDeltaF_skij.set_zero();
  if (sDeltaSigma_tskij) sDeltaSigma_tskij->set_zero();
  if (has_Vcorr) sDeltaVcorr_skij.set_zero();
  if (opt_sDeltaF_sc)       opt_sDeltaF_sc->set_zero();
  if (opt_sDeltaF_pert)     opt_sDeltaF_pert->set_zero();
  if (opt_sDeltaSigma_sc)   opt_sDeltaSigma_sc->set_zero();
  if (opt_sDeltaSigma_pert) opt_sDeltaSigma_pert->set_zero();
  // The outer sequence starts from S_0 = 0, which is a genuine iterate: the
  // first inner solve is exactly the one with no source. So the first outer
  // residual S_1 - 0 needs no special casing.
  if (opt_sDeltaF_pert_prev)     opt_sDeltaF_pert_prev->set_zero();
  if (opt_sDeltaSigma_pert_prev) opt_sDeltaSigma_pert_prev->set_zero();
  if (opt_sDeltaDm_stage_prev)   opt_sDeltaDm_stage_prev->set_zero();
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
        sc_kernel.hartree, sc_kernel.exchange, sc_sigma,
        F_PQ_out != nullptr));

    app_log(2, "  DeltaX IBC correction: setup complete.");
    utils::memlog("lr_driver::run_lr: after build_lr_ibc");
    _Timer.stop("LR_DRIVER_SETUP_IBC");
  }
  _Timer.stop("LR_DRIVER_SETUP");
  utils::memlog("lr_driver::run_lr: end of LR_DRIVER_SETUP");
  print_setup_timers();

  // Timer keys of one Σ channel. The two channels run the same evaluators, so
  // each gets its own clocks — the whole point of the split is a cost claim,
  // and it can only be read off if sc and pert are timed apart.
  struct sigma_clocks { const char *pi, *w, *sigma; };
  constexpr sigma_clocks sc_clocks{"LR_GW_PI", "LR_GW_W", "LR_GW_SIGMA"};
  constexpr sigma_clocks pert_clocks{"LR_GW_PI_PERT", "LR_GW_W_PERT",
                                     "LR_GW_SIGMA_PERT"};

  // Evaluate the Σ components of `mask` into `sSigma_out`, overwriting it. The
  // divergence corrections follow their own component into whichever channel
  // owns it: apply_div_correction_DeltaG belongs to Σ1, apply_div_correction_G
  // to Σ2.
  auto eval_sigma_channel = [&](lr_kernel_spec const& mask,
                                solvers::lr_gw& gw_solver,
                                sArray_t<Array_view_5D_t>& sSigma_out,
                                const lr_ibc_DeltaX* ibc_ptr,
                                sigma_clocks const& clk) {
    utils::check(mask.has_sigma(),
                 "lr_driver::run_lr: eval_sigma_channel called with a Σ-free "
                 "kernel mask ({}).", mask.to_string());
    if (!mask.sigma_G_dW) {
      // Term 1 only: ΔΣ = -ΔG ⊙ W_c
      _Timer.start(clk.sigma);
      gw_solver.evaluate_sigma_DeltaG(
          sSigma_out, sDeltaG_tskij.local(), *opt_dW_tRPQ, thc, ibc_ptr);
      // Divergence correction term 1 (all q): ΔΣ^div += -madelung * eps_inv_head * S(k+q) · ΔG · S(k)
      if (div_corr) {
        gw_solver.apply_div_correction_DeltaG(
            sSigma_out, sDeltaG_tskij.local(), sS_skij.local(), thc, *eps_inv_head);
      }
      _Timer.stop(clk.sigma);
      _mpi->comm.barrier();
      return;
    }

    // Step 3b: ΔP = -ΔG·G - G·ΔG
    _Timer.start(clk.pi);
    auto dDeltaPi_tqPQ = lr_pi_solver->evaluate_lr_Pi(
        sG_tskij.local(), sDeltaG_tskij.local(), thc,
        *opt_dG_tsRPQ, *opt_dG_mtau_tsRPQ, ibc_ptr);
    _mpi->comm.barrier();
    _Timer.stop(clk.pi);

    // Step 3c-3d: ΔW_c(τ) via solve_lr_dyson_W (in-place, uses cached W_full)
    _Timer.start(clk.w);
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
    _Timer.stop(clk.w);

    // Step 3e-3f: ΔΣ = -ΔG ⊙ W_c - G ⊙ ΔW.
    // ΔW stays in (t,q,P,Q): the Σ evaluator consumes one τ slice at a time,
    // which is contiguous in this layout and matches term 1's dW_tRPQ.
    _Timer.start(clk.sigma);
    if (split_sigma_terms) {
      // One-shot G0W0: compute the two terms separately, then store
      //   sDeltaSigma_tskij       = term1 + term2  (total ΔΣ, same as fused)
      //   sDeltaSigma_term2_tskij = term2 (G0·dW0)  [written as DeltaSigma_GdW]
      // term 1 (-ΔG⊙W_c + div) and term 2 (-G⊙ΔW + div) use separate solver
      // instances (gw_solver / lr_gw_solver2, built once above) — the
      // workspace is cached per (term1,term2) combination.
      gw_solver.evaluate_sigma_DeltaG(
          sSigma_out, sDeltaG_tskij.local(), *opt_dW_tRPQ, thc, ibc_ptr);
      lr_gw_solver2->evaluate_sigma_DeltaW(
          *sDeltaSigma_term2_tskij, sG_tskij.local(), dDeltaW_tqPQ, thc,
          *opt_dG_tsRPQ, *opt_dG_mtau_tsRPQ);
      if (div_corr) {
        gw_solver.apply_div_correction_DeltaG(
            sSigma_out, sDeltaG_tskij.local(), sS_skij.local(), thc, *eps_inv_head);
        if (is_q_gamma()) {
          lr_gw_solver2->apply_div_correction_G(
              *sDeltaSigma_term2_tskij, sG_tskij.local(), sS_skij.local(), thc, delta_eps_inv_head);
        }
      }
      // Accumulate term2 into sSigma_out so it holds the total ΔΣ.
      // Both arrays are node-replicated shared memory (each solver all_reduced
      // its result), so add once per node on the node root.
      sSigma_out.win().fence();
      sDeltaSigma_term2_tskij->win().fence();
      if (_mpi->node_comm.root())
        sSigma_out.local() += sDeltaSigma_term2_tskij->local();
      sSigma_out.win().fence();
      _mpi->comm.barrier();
    } else if (mask.sigma_dG_W) {
      // Fused ΔΣ = -ΔG ⊙ W_c - G ⊙ ΔW (single R-space pass)
      gw_solver.evaluate_sigma(
          sSigma_out, sDeltaG_tskij.local(), *opt_dW_tRPQ,
          sG_tskij.local(), dDeltaW_tqPQ, thc,
          *opt_dG_tsRPQ, *opt_dG_mtau_tsRPQ, ibc_ptr);
      // Divergence correction term 1 (all q): eps_inv_head from W, applied to ΔG
      if (div_corr) {
        gw_solver.apply_div_correction_DeltaG(
            sSigma_out, sDeltaG_tskij.local(), sS_skij.local(), thc, *eps_inv_head);
        // Divergence correction term 2 (q_pert=0 only): Δeps_inv_head from ΔW, applied to G
        if (is_q_gamma()) {
          gw_solver.apply_div_correction_G(
              sSigma_out, sG_tskij.local(), sS_skij.local(), thc, delta_eps_inv_head);
        }
      }
    } else {
      // Term 2 only: ΔΣ = -G ⊙ ΔW. Reached only from the perturbative channel
      // of a split-kernel run whose K_sc already resums Σ1.
      gw_solver.evaluate_sigma_DeltaW(
          sSigma_out, sG_tskij.local(), dDeltaW_tqPQ, thc,
          *opt_dG_tsRPQ, *opt_dG_mtau_tsRPQ);
      if (div_corr && is_q_gamma()) {
        gw_solver.apply_div_correction_G(
            sSigma_out, sG_tskij.local(), sS_skij.local(), thc, delta_eps_inv_head);
      }
    }
    _mpi->comm.barrier();
    _Timer.stop(clk.sigma);
  };

  // Rebuild the total of a split quantity, total = sc + pert, striped over
  // node_comm: every node rank owns a contiguous element slice of the shared
  // window and writes only that slice. The result is bit-identical to a serial
  // sum — this is an element-wise map with no reduction — but it runs at 1/nrank
  // of the cost, which matters because a split ΔΣ is rebuilt on EVERY inner
  // iteration (a ΔΣ array is nk·nt·nb², i.e. GBs at production sizes).
  // Callers fence the total and both operand windows before calling: sources are
  // node-replicated shared memory and every rank reads slices written by others,
  // so barriers alone are insufficient under the MPI-3 separate shared-memory
  // model. The operands are taken by const reference (shared_array::win() is
  // non-const), which is why the operand fences live at the call site.
  auto refresh_total = [&](auto& total, auto const& sc_part, auto const& pert_part) {
    auto tot_v  = total.local();
    auto sc_v   = sc_part.local();
    auto pert_v = pert_part.local();
    const long n = tot_v.size();
    const long nr = _mpi->node_comm.size();
    const long r = _mpi->node_comm.rank();
    const long chunk = (n + nr - 1) / nr;
    const long i0 = std::min(r * chunk, n);
    const long i1 = std::min(i0 + chunk, n);
    if (i1 > i0) {
      auto rng = nda::range(i0, i1);
      auto t_s = nda::reshape(tot_v,  std::array<long, 1>{n})(rng);
      auto a_s = nda::reshape(sc_v,   std::array<long, 1>{n})(rng);
      auto b_s = nda::reshape(pert_v, std::array<long, 1>{n})(rng);
      t_s = a_s + b_s;
    }
    total.win().fence();
    _mpi->node_comm.barrier();
  };

  // Refresh whichever totals are actually split. A quantity carried by one
  // channel needs nothing: that channel already wrote the caller's array.
  auto refresh_totals = [&]() {
    if (!split_F && !split_Sigma) return;
    _Timer.start("LR_TOTALS");
    // split_F / split_Sigma guarantee the corresponding optionals are engaged and
    // are exactly the per-channel buffers, so these are the same objects as
    // sDeltaF_sc_skij / pDeltaSigma_sc and friends.
    if (split_F) {
      sDeltaF_skij.win().fence();
      opt_sDeltaF_sc->win().fence();
      opt_sDeltaF_pert->win().fence();
      refresh_total(sDeltaF_skij, *opt_sDeltaF_sc, *opt_sDeltaF_pert);
    }
    if (split_Sigma) {
      sDeltaSigma_tskij->win().fence();
      opt_sDeltaSigma_sc->win().fence();
      opt_sDeltaSigma_pert->win().fence();
      refresh_total(*sDeltaSigma_tskij, *opt_sDeltaSigma_sc, *opt_sDeltaSigma_pert);
    }
    _mpi->comm.barrier();
    _Timer.stop("LR_TOTALS");
  };

  // Outer accelerator. Deliberately function-local and deliberately NOT
  // _lr_diis: the inner object is rebuilt at every stage boundary and keyed on
  // stage_iter, while this one lives across the whole run and is keyed on the
  // outer step index. The two share no history, no subspace and no warmup.
  std::unique_ptr<lr_diis> outer_diis;
  if (outer_diis_on) {
    outer_diis = std::make_unique<lr_diis>(
        outer_accel->iter.max_subsp_size, outer_accel->iter.diis_warmup,
        outer_accel->iter.mixing, outer_accel->min_subsp);
  }

  // Make an in-place outer mixing visible everywhere, exactly as the inner
  // epilogue does: each rank wrote only its own `pmap` slice, so each node holds
  // one contiguous element run and an allgatherv among the node roots completes
  // every replica.
  auto outer_sync = [&](auto*... arrs) {
    auto fence = [](auto* p) { if (p) p->win().fence(); };
    (fence(arrs), ...);
    _mpi->node_comm.barrier();
    if (_mpi->node_comm.root()) {
      auto complete = [&](auto* p) {
        if (p) utils::lr_complete_node_slices(_mpi->internode_comm, pmap,
                                              p->local().data(), p->local().size());
      };
      (complete(arrs), ...);
    }
    (fence(arrs), ...);
    _mpi->comm.barrier();
  };

  // dst <- src on the node-replicated shared window.
  auto outer_save = [&](auto& dst, auto& src) {
    dst.win().fence();
    if (_mpi->node_comm.root()) dst.local() = src.local();
    dst.win().fence();
    _mpi->node_comm.barrier();
  };

  // ‖A - A_prev‖ over the node, broadcast so every rank agrees.
  auto outer_diff_norm = [&](auto& A, auto& A_prev) {
    auto nrm = utils::lr_distributed_norm(
        _mpi->node_comm, A.local(), A_prev.local(), true);
    double d = nrm.second;
    _mpi->comm.broadcast_n(&d, 1, 0);
    return d;
  };

  double Delta_mu = 0.0;
  int iter = 0;
  bool converged = false;
  // Split-kernel stage state. `stage_iter` is the iteration index within the
  // current stage: DIIS keys its warmup off it and it gates the prev-array save,
  // so a stage boundary looks like a fresh start even though ΔF_sc/ΔΣ_sc are
  // deliberately carried over (warm start) into the next stage.
  int n_applied = 0;
  int stage = 1;
  int stage_iter = 0;
  // Tolerance-driven outer termination: set once the stage-to-stage change of
  // ΔDm falls below outer_tol, which stops the schedule short of pert_order.
  bool outer_converged = false;

  const bool log_sigma_col = has_Sigma_sc || has_Vcorr;

  // SCF iteration header
  if (do_pert) {
    app_log(1, "\n  (split kernel: the iter column carries the stage index as "
               "[s<n>]; K_pert is re-evaluated at each stage boundary)");
    if (outer_diis_on)
      app_log(1, "  (outer acceleration on: each stage boundary extrapolates the "
                 "perturbative source and prints the outer residual; the stage "
                 "count is an iteration count, not a truncation order)");
  }
  if (log_sigma_col) {
    app_log(1, "\n  iter    ||ΔDm||         ||ΔDm-ΔDm_prev||  ||ΔF||          ||ΔF-ΔF_prev||   ||ΔΣ||          ||ΔΣ-ΔΣ_prev||   Δμ");
    app_log(1, "  " + std::string(120, '-'));
  } else {
    app_log(1, "\n  iter    ||ΔDm||         ||ΔDm-ΔDm_prev||  ||ΔF||          ||ΔF-ΔF_prev||   Δμ");
    app_log(1, "  " + std::string(92, '-'));
  }

  for (iter = 1; iter <= max_iter; ++iter) {

    ++stage_iter;
    const bool first_of_stage = (stage_iter == 1);
    bool pert_refreshed_this_iter = false;

    // Save previous density matrix and Fock matrix. ΔDm is node-replicated, so
    // node root copies it whole; the mixed quantities are saved as this rank's
    // partition slice only, in parallel across the node.
    _Timer.start("LR_SAVE");
    if (stage_iter > 1) {
      if (_mpi->node_comm.root())
        sDeltaDm_prev_skij.local() = sDeltaDm_skij.local();
      DeltaF_prev = flat_slice(sDeltaF_sc_skij.local(), nF_flat, iF0, iF1);
      if (has_Vcorr) {
        DeltaVcorr_prev = flat_slice(sDeltaVcorr_skij.local(), nV_flat, iV0, iV1);
      } else if (has_Sigma_sc) {
        DeltaSigma_prev = flat_slice(pDeltaSigma_sc->local(), nS_flat, iS0, iS1);
      }
    }
    _mpi->comm.barrier();
    _Timer.stop("LR_SAVE");

    // Step 1: Solve LR Dyson equation
    // ΔG = G_{k+q} · [ΔH0 + ΔF + (ΔΣ | ΔV_QPGW) - Δμ·S] · G_k
    // In qp mode the dynamic ΔΣ is dropped from the RHS (skipping its τ→ω) and
    // the static ΔV_QPGW enters as a frequency-independent one-body term.
    _Timer.start("LR_DYSON");
    sArray_t<Array_view_5D_t>* dyson_sigma = has_Sigma ? sDeltaSigma_tskij : nullptr;
    const sArray_t<Array_view_4D_t>* dyson_vcorr = has_Vcorr ? &sDeltaVcorr_skij : nullptr;
    Delta_mu = _lr_dyson.solve_lr_dyson(
        sDeltaG_tskij, sDeltaDm_skij, sDeltaH0_skij,
        sDeltaF_skij, dyson_sigma,
        fix_density, Delta_mu, dyson_vcorr);
    _Timer.stop("LR_DYSON");
    _mpi->comm.barrier();

    // Compute norms for logging. lr_distributed_norm stripes the (s,k) blocks
    // over node_comm ranks and reduces within the node; the shared array is
    // node-replicated, so the trailing broadcast from global rank 0 preserves
    // exact global agreement.
    _Timer.start("LR_CONVERGENCE");
    auto norms_Dm = utils::lr_distributed_norm(
        _mpi->node_comm, sDeltaDm_skij.local(), sDeltaDm_prev_skij.local(), stage_iter > 1);
    double norm_DeltaDm = norms_Dm.first;
    double norm_DeltaDm_diff = norms_Dm.second;
    _mpi->comm.broadcast_n(&norm_DeltaDm, 1, 0);
    _mpi->comm.broadcast_n(&norm_DeltaDm_diff, 1, 0);
    _Timer.stop("LR_CONVERGENCE");

    // Step 2: Compute the K_sc LR Fock matrix (if requested)
    if (sc_hf) {
      _Timer.start("LR_HF");
      const lr_ibc_DeltaX* ibc_ptr = opt_ibc ? &(*opt_ibc) : nullptr;
      _lr_hf->evaluate(sDeltaF_sc_skij, sDeltaDm_skij, thc, sS_skij.local(),
                       sc_kernel.hartree, sc_kernel.exchange, ibc_ptr,
                       DeltaV_qPQ, Dm_ab, nullptr, include_xc);
      _Timer.stop("LR_HF");
      _mpi->comm.barrier();
    }

    // Step 3: Compute the K_sc LR GW self-energy
    if (sc_sigma) {
      const lr_ibc_DeltaX* ibc_ptr = opt_ibc ? &(*opt_ibc) : nullptr;
      eval_sigma_channel(sc_kernel, *lr_gw_solver, *pDeltaSigma_sc, ibc_ptr, sc_clocks);
    }

    // Step 3g (qp mode): statify the dynamic ΔΣ(iω) into the static ΔV_QPGW(k)
    // using the frozen QP orbitals/energies. ΔV_QPGW is the tracked/mixed static
    // quantity and enters the Dyson RHS at the next iteration.
    if (qp_mode) {
      _Timer.start("LR_QPGW_STATIC");
      auto sVcorr = lr_qp_approx(
          *sDeltaSigma_tskij, *qp_static->sMO_skia, *qp_static->sE_ska,
          qp_static->mu, _lr_dyson.kpq_map(), is_q_gamma(),
          *_dyson.FT(), qp_static->qp_params);
      sDeltaVcorr_skij.win().fence();
      if (_mpi->node_comm.root())
        sDeltaVcorr_skij.local() = sVcorr.local();
      sDeltaVcorr_skij.win().fence();
      _mpi->comm.barrier();
      _Timer.stop("LR_QPGW_STATIC");
    }

    // Step 4: Apply iteration algorithm (DIIS or damping) on the combined
    // sc-channel (ΔF, ΔΣ). The perturbative source is frozen input, not an SCF
    // variable, so it never takes part in the mixing.
    _Timer.start("LR_ITER_ALG");
    if (stage_iter > 1 && (sc_hf || sc_sigma || has_Vcorr)) {
      // The static second quantity mixed alongside ΔF is the dynamic ΔΣ in the
      // standard path, or the static ΔV_QPGW in qp mode.
      if (use_diis) {
        // Striped DIIS: every rank of the global comm participates, each
        // operating on its `pmap` element-slice of the shared ΔF/ΔΣ and writing
        // the mixed result back in place. Pass .local() views directly (in/out);
        // the "prev" arguments are already this rank's slice.
        if (has_Vcorr) {
          _lr_diis->next_step_combined(
              _mpi->comm, pmap,
              sDeltaF_sc_skij.local(), DeltaF_prev,
              sDeltaVcorr_skij.local(), DeltaVcorr_prev, stage_iter);
        } else if (has_Sigma_sc) {
          _lr_diis->next_step_combined(
              _mpi->comm, pmap,
              sDeltaF_sc_skij.local(), DeltaF_prev,
              pDeltaSigma_sc->local(), DeltaSigma_prev, stage_iter);
        } else {
          nda::array<ComplexType, 5> empty_sigma;
          nda::array<ComplexType, 1> empty_prev;
          _lr_diis->next_step_combined(
              _mpi->comm, pmap,
              sDeltaF_sc_skij.local(), DeltaF_prev,
              empty_sigma, empty_prev, stage_iter);
        }
      } else if (mixing < 1.0) {
        // Damping is elementwise too, so stripe it over the same partition —
        // one completion path then covers both algorithms.
        auto F_loc = flat_slice(sDeltaF_sc_skij.local(), nF_flat, iF0, iF1);
        F_loc = mixing * F_loc + (1.0 - mixing) * DeltaF_prev;
        if (has_Vcorr) {
          auto V_loc = flat_slice(sDeltaVcorr_skij.local(), nV_flat, iV0, iV1);
          V_loc = mixing * V_loc + (1.0 - mixing) * DeltaVcorr_prev;
        } else if (has_Sigma_sc) {
          auto S_loc = flat_slice(pDeltaSigma_sc->local(), nS_flat, iS0, iS1);
          S_loc = mixing * S_loc + (1.0 - mixing) * DeltaSigma_prev;
        }
      }
      // The mixing above writes each rank's slice of the shared ΔF/ΔΣ buffer in
      // place with no trailing collective. Fence + barrier make every slice
      // visible to node root before it gathers below (barrier alone is
      // insufficient under the MPI-3 separate shared-memory model).
      sDeltaF_sc_skij.win().fence();
      if (has_Vcorr) sDeltaVcorr_skij.win().fence();
      else if (has_Sigma_sc) pDeltaSigma_sc->win().fence();
      _mpi->node_comm.barrier();
      // Each node now holds a valid copy of its own contiguous element run
      // only; one allgatherv among the node roots completes every replica. With
      // the striping global, each element is mixed exactly once in the whole
      // job, so the per-node drift the old node-0 broadcast papered over cannot
      // arise by construction.
      if (_mpi->node_comm.root()) {
        utils::complete_node_slices(_mpi->internode_comm, pmap,
                                       sDeltaF_sc_skij.local().data(), nF_flat);
        if (has_Vcorr) {
          utils::complete_node_slices(_mpi->internode_comm, pmap,
                                         sDeltaVcorr_skij.local().data(), nV_flat);
        } else if (has_Sigma_sc) {
          utils::complete_node_slices(_mpi->internode_comm, pmap,
                                         pDeltaSigma_sc->local().data(), nS_flat);
        }
      }
      // Make node root's overwrite visible to its node peers
      sDeltaF_sc_skij.win().fence();
      if (has_Vcorr) sDeltaVcorr_skij.win().fence();
      else if (has_Sigma_sc) pDeltaSigma_sc->win().fence();
      _mpi->comm.barrier();
    }
    _Timer.stop("LR_ITER_ALG");

    // Compute norms of ΔF and ΔF-ΔF_prev for logging. The previous iterates are
    // stored striped, so the norms are reduced over the global comm from the
    // same slices.
    _Timer.start("LR_CONVERGENCE");
    auto norms_F = utils::striped_norm(
        _mpi->comm, flat_slice(sDeltaF_sc_skij.local(), nF_flat, iF0, iF1),
        DeltaF_prev, stage_iter > 1);
    double norm_DeltaF = norms_F.first;
    double norm_DeltaF_diff = norms_F.second;
    _mpi->comm.broadcast_n(&norm_DeltaF, 1, 0);
    _mpi->comm.broadcast_n(&norm_DeltaF_diff, 1, 0);

    // Compute norms of the tracked static second quantity (dynamic ΔΣ in the
    // standard path; static ΔV_QPGW in qp mode) for logging/convergence.
    double norm_DeltaSigma = 0.0;
    double norm_DeltaSigma_diff = 0.0;
    if (has_Vcorr) {
      auto norms_V = utils::striped_norm(
          _mpi->comm, flat_slice(sDeltaVcorr_skij.local(), nV_flat, iV0, iV1),
          DeltaVcorr_prev, stage_iter > 1);
      norm_DeltaSigma = norms_V.first;
      norm_DeltaSigma_diff = norms_V.second;
      _mpi->comm.broadcast_n(&norm_DeltaSigma, 1, 0);
      _mpi->comm.broadcast_n(&norm_DeltaSigma_diff, 1, 0);
    } else if (has_Sigma_sc) {
      auto norms_Sigma = utils::striped_norm(
          _mpi->comm, flat_slice(pDeltaSigma_sc->local(), nS_flat, iS0, iS1),
          DeltaSigma_prev, stage_iter > 1);
      norm_DeltaSigma = norms_Sigma.first;
      norm_DeltaSigma_diff = norms_Sigma.second;
      _mpi->comm.broadcast_n(&norm_DeltaSigma, 1, 0);
      _mpi->comm.broadcast_n(&norm_DeltaSigma_diff, 1, 0);
    }
    _Timer.stop("LR_CONVERGENCE");

    // Has the inner (K_sc only, frozen perturbative source) solve converged?
    // An empty K_sc makes it exact in a single Dyson application, so every
    // stage is then one iteration long.
    bool dm_converged = norm_DeltaDm_diff < tol;
    bool f_converged = !sc_hf || norm_DeltaF_diff < tol;
    bool sigma_converged = !(has_Sigma_sc || has_Vcorr) || norm_DeltaSigma_diff < tol;
    bool inner_conv_std = (stage_iter > 1) && dm_converged && f_converged && sigma_converged;
    bool inner_converged = (do_pert && sc_kernel.empty()) ? true : inner_conv_std;

    // Log iteration. This closes the iteration that produced the norms above,
    // so it comes before the stage-boundary block: K_pert logs of its own
    // (head extrapolation, outer residual) then follow their iteration's row
    // instead of splitting it from the previous one.
    _Timer.start("LR_CONVERGENCE");
    std::string iter_lbl = do_pert ? fmt::format("{}[s{}]", iter, stage)
                                   : fmt::format("{}", iter);
    if (first_of_stage) {
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
    _Timer.stop("LR_CONVERGENCE");

    // Stage boundary: one K_pert evaluation on the converged ΔG of this stage,
    // overwriting (not accumulating) the perturbative source — ΔG already
    // carries every lower order.
    if (do_pert && inner_converged && n_applied < pert_order && !outer_converged) {
      const int outer_step = n_applied + 1;  // 1-based outer iteration index
      double outer_dm_diff = -1.0;
      double outer_res = -1.0;

      // Outer convergence is tested on the stage-to-stage change of ΔDm, and
      // BEFORE the next K_pert evaluation: the source residual would need the
      // very evaluation it proves unnecessary, and one such evaluation is the
      // entire unit of cost here. ΔDm is tiny, so this test is free.
      if (outer_tol > 0.0 && n_applied >= 1) {
        outer_dm_diff = outer_diff_norm(sDeltaDm_skij, *opt_sDeltaDm_stage_prev);
        if (outer_dm_diff < outer_tol) outer_converged = true;
      }

      if (!outer_converged) {
        if (outer_tol > 0.0) outer_save(*opt_sDeltaDm_stage_prev, sDeltaDm_skij);

        if (pert_hf) {
          _Timer.start("LR_HF_PERT");
          _lr_hf->evaluate(sDeltaF_pert_skij, sDeltaDm_skij, thc, sS_skij.local(),
                           pert.hartree, pert.exchange, nullptr,
                           nullptr, nullptr, nullptr, false);
          _Timer.stop("LR_HF_PERT");
          _mpi->comm.barrier();
        }
        if (pert_sigma) {
          eval_sigma_channel(pert, *lr_gw_solver_pert, *pDeltaSigma_pert, nullptr,
                             pert_clocks);
        }

        // Extrapolate the perturbative source. Only a channel that actually
        // carries the quantity may be mixed — otherwise the handle aliases the
        // SELF-CONSISTENT array (see the per-channel write targets above) and
        // mixing it would extrapolate the sc channel with the outer sequence.
        //
        // Extrapolating the source rather than the solution loses nothing:
        // compute_coefs normalizes Σ c_i = 1 and the inner solve A is affine,
        // so Σ c_i ΔG_i = A(ΔH0 + Σ c_i S_{i-1}) — the combination of the
        // stage solutions IS the exact inner solution of the combined source.
        // The two choices are the same iteration in different inner products.
        if (outer_diis_on) {
          double r2 = 0.0;
          if (pert_hf) {
            double d = outer_diff_norm(sDeltaF_pert_skij, *opt_sDeltaF_pert_prev);
            r2 += d * d;
          }
          if (pert_sigma) {
            double d = outer_diff_norm(*pDeltaSigma_pert, *opt_sDeltaSigma_pert_prev);
            r2 += d * d;
          }
          outer_res = std::sqrt(r2);

          _Timer.start("LR_OUTER_ITER_ALG");
          // The outer accelerator stripes over the same `pmap` partition as the
          // inner one: each rank mixes its own element slice of the source and
          // outer_sync completes the node replicas afterwards. The previous
          // source stays whole (outer_diff_norm reads it as a node array), so
          // it is sliced here rather than stored striped.
          const long nSp_flat = _nts * nF_flat;
          auto [oF0, oF1] = pmap.my_slice(nF_flat);
          auto [oS0, oS1] = pmap.my_slice(nSp_flat);
          nda::array<ComplexType, 1> empty_prev;
          if (pert_hf && pert_sigma) {
            outer_diis->next_step_combined(
                _mpi->comm, pmap,
                sDeltaF_pert_skij.local(),
                flat_slice(opt_sDeltaF_pert_prev->local(), nF_flat, oF0, oF1),
                pDeltaSigma_pert->local(),
                flat_slice(opt_sDeltaSigma_pert_prev->local(), nSp_flat, oS0, oS1),
                outer_step);
          } else if (pert_sigma) {
            // Σ-only source: next_step_combined is generic in both slots, so
            // the 5D ΔΣ rides the first one with an empty second slot.
            nda::array<ComplexType, 5> empty_slot;
            outer_diis->next_step_combined(
                _mpi->comm, pmap,
                pDeltaSigma_pert->local(),
                flat_slice(opt_sDeltaSigma_pert_prev->local(), nSp_flat, oS0, oS1),
                empty_slot, empty_prev, outer_step);
          } else {
            nda::array<ComplexType, 4> empty_slot;
            outer_diis->next_step_combined(
                _mpi->comm, pmap,
                sDeltaF_pert_skij.local(),
                flat_slice(opt_sDeltaF_pert_prev->local(), nF_flat, oF0, oF1),
                empty_slot, empty_prev, outer_step);
          }
          outer_sync(pert_hf ? &sDeltaF_pert_skij : nullptr,
                     pert_sigma ? pDeltaSigma_pert : nullptr);
          // The mixed source is what the next stage uses, hence what the next
          // outer residual is measured against.
          if (pert_hf)    outer_save(*opt_sDeltaF_pert_prev, sDeltaF_pert_skij);
          if (pert_sigma) outer_save(*opt_sDeltaSigma_pert_prev, *pDeltaSigma_pert);
          _Timer.stop("LR_OUTER_ITER_ALG");
        }

        ++n_applied;
        pert_refreshed_this_iter = true;
        if (outer_track) {
          // The source residual exists only when the accelerator ran; the ΔDm
          // change only from the second stage of a tolerance-driven run.
          app_log(1, "    [outer {}/{}]{}{}", n_applied, pert_order,
                  outer_res >= 0.0
                      ? fmt::format("  ||S_new - S_used|| = {:.6e}", outer_res)
                      : std::string(),
                  outer_dm_diff >= 0.0
                      ? fmt::format("  ||ΔDm - ΔDm_stage_prev|| = {:.6e}", outer_dm_diff)
                      : std::string());
        }
        // The next stage solves a different fixed point: the DIIS history from
        // this one is invalid (lr_diis has no reset) and its warmup keys off the
        // stage-local iteration index. ΔF_sc/ΔΣ_sc are deliberately kept as the
        // warm start for the next stage.
        if (use_diis) {
          _lr_diis = std::make_unique<lr_diis>(
              iter_params.max_subsp_size, iter_params.diis_warmup, mixing);
        }
      } else {
        app_log(1, "    [outer] converged after {} K_pert evaluation(s): "
                   "||ΔDm - ΔDm_stage_prev|| = {:.6e} < {:.2e}",
                n_applied, outer_dm_diff, outer_tol);
      }
    }

    // Refresh the totals the next Dyson solve (and the checkpoint) consume.
    refresh_totals();

    // Step 5: Check convergence (all active quantities must converge, and the
    // perturbative expansion must have been applied to the requested order).
    // An iteration that just refreshed the source is never the converged one:
    // its ΔG has not yet seen the new source.
    const bool outer_done = outer_converged || (n_applied == pert_order);
    if (inner_converged && outer_done && !pert_refreshed_this_iter) {
      converged = true;
      break;
    }

    // Open the next stage: the stage-local index restarts so DIIS warmup and
    // the prev-array bookkeeping treat it as a fresh solve.
    if (pert_refreshed_this_iter) {
      stage_iter = 0;
      ++stage;
    }

    _mpi->comm.barrier();
  }

  _Timer.stop("LR_SCF");

  // Free the G^R cache (not needed by the post-loop HF/IBC epilogue)
  opt_dG_tsRPQ.reset();
  opt_dG_mtau_tsRPQ.reset();

  // Copy the converged static ΔV_QPGW into the caller's output array (qp mode).
  if (qp_mode && sDeltaVcorr_out_skij != nullptr) {
    sDeltaVcorr_out_skij->win().fence();
    if (_mpi->node_comm.root())
      sDeltaVcorr_out_skij->local() = sDeltaVcorr_skij.local();
    sDeltaVcorr_out_skij->win().fence();
    _mpi->comm.barrier();
  }

  // Report results. On a split-kernel run the applied order is part of the
  // verdict: exhausting max_iter mid-schedule silently returns an order-
  // n_applied result for an order-pert_order request.
  if (converged) {
    app_log(1, "\n  LR SCF converged in {} iterations!", iter);
  } else if (max_iter > 1) {
    if (do_pert)
      app_log(1, "\n  [WARNING] LR SCF did NOT converge after {} iterations "
                 "(K_pert applied {} of {} times).", max_iter, n_applied, pert_order);
    else
      app_log(1, "\n  [WARNING] LR SCF did NOT converge after {} iterations.", max_iter);
  }
  if (do_pert && outer_track) {
    if (outer_converged)
      app_log(1, "  Outer loop converged: K_pert applied {} of at most {} times.",
              n_applied, pert_order);
    else if (outer_tol > 0.0)
      app_log(1, "  [WARNING] the outer loop hit its cap: K_pert applied {} of {} "
                 "times without reaching outer_tol = {:.2e}. The result is a "
                 "non-converged accelerated iterate, NOT an order-{} truncation.",
              n_applied, pert_order, outer_tol, pert_order);
    else
      app_log(1, "  Outer loop ran its full schedule: K_pert applied {} of {} times.",
              n_applied, pert_order);
  }
  if (n_pert_applied_out) *n_pert_applied_out = n_applied;
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
  if (F_PQ_out && opt_ibc && opt_ibc->F_PQ_skij.size() > 0) {
    *F_PQ_out = std::move(opt_ibc->F_PQ_skij);
  }
  if (DeltaF_PQ_out && need_hf) {
    // One extra lr_hf::evaluate on the converged ΔDm just to capture ΔF_PQ. It writes
    // into a scratch ΔF rather than sDeltaF_skij: the converged sDeltaF_skij is the
    // mixed (DIIS/damped) iterate the caller persists, and re-evaluating from ΔDm
    // would replace it with a different matrix, so an output-only flag would change
    // the DeltaF_skij dataset.
    auto sDeltaF_scratch = math::shm::make_shared_array<Array_view_4D_t>(
        *_mpi, {_ns, _nkpts_ibz, _nbnd, _nbnd});
    const lr_ibc_DeltaX* ibc_ptr = opt_ibc ? &(*opt_ibc) : nullptr;
    // Total (K_sc ∪ K_pert) Hartree/exchange content.
    _lr_hf->evaluate(sDeltaF_scratch, sDeltaDm_skij, thc, sS_skij.local(),
                     sc_kernel.hartree || pert.hartree, sc_kernel.exchange || pert.exchange, ibc_ptr,
                     DeltaV_qPQ, Dm_ab,
                     DeltaF_PQ_out, include_xc);
  }

  // Final hierarchical timer report (printed once, at verbosity >= 2).
  // Per-step solver prints inside the loop are gated to verbosity >= 3.
  print_timers(lr_pi_solver.get(), lr_scr_solver.get(),
               lr_gw_solver.get(), lr_gw_solver_pert.get(), lr_gw_solver2.get());

  return std::make_tuple(iter, Delta_mu);
}


void lr_driver::print_memory_estimate(long NP, bool include_gw_sigma, bool gw_full,
                                      std::vector<std::string> const& extra_sigma,
                                      lr_diis_hist_t inner_hist,
                                      lr_diis_hist_t outer_hist) {
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
  // sG_tskij is caller-owned but resident throughout run_lr, so count it here.
  arrays.push_back({"sG_tskij",       shp5b(nt), band5(nt), false, PERSIST});
  arrays.push_back({"sDeltaG_tskij",  shp5b(nt), band5(nt), false, PERSIST});
  arrays.push_back({"sG_wskij",       shp5b(nw), band5(nw), false, PERSIST});
  if (include_gw_sigma) {
    arrays.push_back({"sDeltaSigma_tskij",      shp5b(nt), band5(nt), false, PERSIST});
    // Split-kernel per-channel ΔΣ buffers and the outer accelerator's previous
    // source, on top of the total ΔΣ.
    for (auto const& name : extra_sigma)
      arrays.push_back({name, shp5b(nt), band5(nt), false, PERSIST});
  }

  // --- Persistent, striped over the global comm (each rank keeps one element
  //     slice of a node-replicated band array) ---
  // Previous iterates for damping/DIIS. ΔF_prev / ΔV_QPGW_prev are ~nb²·nk and
  // negligible next to these; only the ΔΣ history is worth a row.
  if (include_gw_sigma) {
    arrays.push_back({"ΔΣ_prev (striped)", shp5b(nt), band5(nt), true, PERSIST});
  }

  // DIIS histories. Each subspace entry keeps a trial AND a residual vector,
  // and the residual is not reconstructible from the trials once extrapolation
  // is active, so a depth-d history is 2d arrays. The history is striped over
  // the global comm, so the whole job stores it once.
  auto push_hist = [&](lr_diis_hist_t const& h, const char* who) {
    if (h.depth <= 0 || (h.n_F == 0 && h.n_Sigma == 0)) return;
    const double nelem = 2.0 * h.depth * (h.n_F * band5(1) + h.n_Sigma * band5(nt));
    arrays.push_back({fmt::format("{} DIIS history", who),
                      fmt::format("2x{}x[{}xΔF + {}xΔΣ]", h.depth, h.n_F, h.n_Sigma),
                      nelem, true, PERSIST});
  };
  push_hist(inner_hist, "inner");
  push_hist(outer_hist, "outer");

  // --- Persistent, distributed (over global comm), aux basis ~ nk·nt·NP² ---
  if (include_gw_sigma) {
    arrays.push_back({"dW_tRPQ",       shp4a(nth), aux4(nth), true, PERSIST});
  }
  if (gw_full) {
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
  //       ΔW/Σ  : ΔΠ/ΔW(τ) + the FT staging buffers (gw_full only)
  //     lr_dyson runs before the Π/W/Σ steps and frees its scratch first, so the
  //     two never coexist — the peak adds only the larger of the two phases.
  arrays.push_back({"ΔG(iω) (lr_dyson)", shp5b(nw), band5(nw), true, T_DYSON});
  if (include_gw_sigma)
    arrays.push_back({"ΔΣ(iω) (lr_dyson)", shp5b(nw), band5(nw), true, T_DYSON});
  if (gw_full) {
    arrays.push_back({"ΔΠ/ΔW(τ)",       shp4a(nth), aux4(nth), true, T_GWSIG});
    // FT staging buffers, allocated and released inside each tau_to_w/w_to_tau.
    arrays.push_back({"FT buffer (τ)",   shp4a(nth),  aux4(nth),  true, T_GWSIG});
    arrays.push_back({"FT buffer (ω)",   shp4a(nwbh), aux4(nwbh), true, T_GWSIG});
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
            pg4(ftb_pg, "(·,q,P,Q)"), "FT staging buffers (τ, ω)");
    app_log(2, "    {:<22s}{:<30s}{}", "aux ω-side",
            pg4(w_pg, "(w,q,P,Q)"),
            is_q_gamma() ? "dW_full_wqPQ" : "dW_full_wqPQ, _dW_full_qpQ");
  }

  // Band-basis Dyson grids — the ω-side comes from the same helper
  // solve_lr_dyson_impl allocates with; the τ redistribute target mirrors the
  // inline proc-grid math there.
  auto [dyw_pg, dyw_bs] = lr_dyson_omega_pgrid(nproc, nw, nki, _nbnd);
  (void)dyw_bs;
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
                             solvers::lr_gw* gw_solver,
                             solvers::lr_gw* gw_solver_pert,
                             solvers::lr_gw* gw_solver_term2) {
  // Driver totals in execution order, each followed by the corresponding
  // solver's subclocks (deeper indent). Solver pointers may be null when the
  // step was not active; subclocks are then skipped.
  //
  // The report has the same shape for every kernel. Splitting the kernel splits
  // an evaluator across several instances and clock keys, but that is an
  // implementation detail of how ΔF/ΔΣ is assembled, not a different
  // measurement: every line below sums over all of a step's channels. Lines a
  // given run never exercises print as 0.000 sec / 0 calls rather than
  // disappearing, so two runs' reports stay directly comparable line by line.
  const std::string sub_indent = "        ";
  auto sec = [&](const char* a, const char* b) {
    return _Timer.elapsed(a) + _Timer.elapsed(b);
  };
  auto cnt = [&](const char* a, const char* b) {
    return _Timer.number_of_calls(a) + _Timer.number_of_calls(b);
  };
  app_log(2, "\n  LR_DRIVER timers");
  app_log(2, "  -----------------");
  app_log(2, "    Total LR SCF:               {0:8.3f} sec  {1:4d} calls", _Timer.elapsed("LR_SCF"), _Timer.number_of_calls("LR_SCF"));
  app_log(2, "      - LR Driver Setup:        {0:8.3f} sec  {1:4d} calls", _Timer.elapsed("LR_DRIVER_SETUP"), _Timer.number_of_calls("LR_DRIVER_SETUP"));
  app_log(2, "      - LR Dyson (total):       {0:8.3f} sec  {1:4d} calls", _Timer.elapsed("LR_DYSON"), _Timer.number_of_calls("LR_DYSON"));
  _lr_dyson.print_subclocks(2, sub_indent);
  app_log(2, "      - LR HF (total):          {0:8.3f} sec  {1:4d} calls", sec("LR_HF", "LR_HF_PERT"), cnt("LR_HF", "LR_HF_PERT"));
  if (_lr_hf) _lr_hf->print_subclocks(2, sub_indent);
  app_log(2, "      - LR GW Pi (total):       {0:8.3f} sec  {1:4d} calls", sec("LR_GW_PI", "LR_GW_PI_PERT"), cnt("LR_GW_PI", "LR_GW_PI_PERT"));
  if (pi_solver) pi_solver->print_subclocks(2, sub_indent);
  app_log(2, "      - LR GW W (total):        {0:8.3f} sec  {1:4d} calls", sec("LR_GW_W", "LR_GW_W_PERT"), cnt("LR_GW_W", "LR_GW_W_PERT"));
  if (scr_solver) scr_solver->print_subclocks(2, sub_indent);
  app_log(2, "      - LR GW Sigma (total):    {0:8.3f} sec  {1:4d} calls", sec("LR_GW_SIGMA", "LR_GW_SIGMA_PERT"), cnt("LR_GW_SIGMA", "LR_GW_SIGMA_PERT"));
  if (gw_solver) gw_solver->print_subclocks(2, sub_indent, {gw_solver_pert, gw_solver_term2});
  app_log(2, "      - LR GW Sig T2 (total):   {0:8.3f} sec  {1:4d} calls", _Timer.elapsed("LR_GW_SIGMA_TERM2"), _Timer.number_of_calls("LR_GW_SIGMA_TERM2"));
  app_log(2, "      - LR qpGW static (total): {0:8.3f} sec  {1:4d} calls", _Timer.elapsed("LR_QPGW_STATIC"), _Timer.number_of_calls("LR_QPGW_STATIC"));
  app_log(2, "      - LR Iter Alg (total):    {0:8.3f} sec  {1:4d} calls", sec("LR_ITER_ALG", "LR_OUTER_ITER_ALG"), cnt("LR_ITER_ALG", "LR_OUTER_ITER_ALG"));
  app_log(2, "      - LR Totals (ΔF/ΔΣ):      {0:8.3f} sec  {1:4d} calls", _Timer.elapsed("LR_TOTALS"), _Timer.number_of_calls("LR_TOTALS"));
  app_log(2, "      - LR Save (prev arrays):  {0:8.3f} sec  {1:4d} calls", _Timer.elapsed("LR_SAVE"), _Timer.number_of_calls("LR_SAVE"));
  app_log(2, "      - LR Convergence (norms): {0:8.3f} sec  {1:4d} calls", _Timer.elapsed("LR_CONVERGENCE"), _Timer.number_of_calls("LR_CONVERGENCE"));
  app_log(2, "");
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
    lr_kernel_spec, lr_kernel_spec, int,
    dW_concrete_t*, const nda::array<ComplexType, 1>*,
    int, double, bool, const lr_iter_params&,
    const sArray_t<Array_view_4D_t>*, const sArray_t<Array_view_4D_t>*,
    const nda::array<ComplexType, 4>*, bool, std::string,
    const nda::array_view<ComplexType, 3>*,
    sArray_t<Array_view_5D_t>*,
    const lr_qp_static_params*,
    sArray_t<Array_view_4D_t>*,
    nda::array<ComplexType, 4>*,
    nda::array<ComplexType, 4>*,
    nda::array<ComplexType, 4>*,
    bool,
    const lr_outer_accel_params*,
    int*);

} // namespace methods
