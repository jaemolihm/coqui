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
                  "LR_DRIVER_SETUP_W_FULL", "LR_DRIVER_SETUP_W_TRPQ", "LR_DRIVER_SETUP_WC0",
                  "LR_DRIVER_SETUP_G_OMEGA", "LR_DRIVER_SETUP_G_R",
                  "LR_DRIVER_SETUP_DELTAMU", "LR_DRIVER_SETUP_ALLOC", "LR_DRIVER_SETUP_IBC", "LR_DRIVER_SETUP_MISC",
                  "LR_DYSON", "LR_HF", "LR_GW_SIGMA",
                   "LR_GW_PI", "LR_GW_W", "LR_QPGW_STATIC",
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


lr_kernel_spec kernel_diff(lr_kernel_spec const& total, lr_kernel_spec const& sc) {
  utils::check(total.contains(sc),
               "kernel_diff: the self-consistent kernel ({}) is not a subset of "
               "the total kernel ({}).", sc.to_string(), total.to_string());
  return lr_kernel_spec{total.hartree    && !sc.hartree,
                        total.exchange   && !sc.exchange,
                        total.sigma_dG_W && !sc.sigma_dG_W,
                        total.sigma_G_dW && !sc.sigma_G_dW};
}


namespace {

/**
 * The kernel split a run executes: the components the inner SCF loop resums
 * (K_sc), those applied perturbatively (K_pert), and the unions on which every
 * solver, buffer and W-operand decision is taken.
 *
 * Derived from `p` alone and recomputed identically by lr_setup and
 * lr_solve_one, so what one allocates is exactly what the other reads.
 */
struct lr_kernel_split {
  lr_kernel_spec sc{};
  lr_kernel_spec pert{};
  bool do_pert = false;            ///< a perturbative pass actually runs

  bool sc_hf = false,    pert_hf = false;
  bool sc_sigma = false, pert_sigma = false;
  bool need_hf = false;            ///< ΔF is evaluated by some channel
  bool include_gw_sigma = false;   ///< ΔΣ is evaluated by some channel
  bool gw_full = false;            ///< Σ2 = -G⊙ΔW anywhere, hence the ΔW Dyson

  /// A quantity BOTH channels contribute to. It is the only case needing
  /// per-channel buffers and a total rebuilt every inner iteration; a quantity
  /// carried by one channel is written straight into the caller's array.
  bool split_F = false, split_Sigma = false;

  bool qp_mode = false;
  bool has_Vcorr = false;     ///< the static ΔV_QPGW is the mixed/tracked quantity
  bool has_Sigma = false;     ///< the dynamic ΔΣ enters the Dyson RHS
  bool has_Sigma_sc = false;  ///< ... and the sc channel's copy is the tracked one

  /// The perturbative channel owes the static counter-term +ΔDm ⊙ W_c(0) on
  /// top of its mask, because K_sc is HSEX. See lr_params::exchange_static_W.
  bool pert_sex_counterterm = false;
};

lr_kernel_split make_kernel_split(lr_params const& p) {
  lr_kernel_split k;
  // The active-term flags always name the TOTAL kernel; on the single-kernel
  // path sc_kernel merely repeats it and may be left empty. A caller that asked
  // for a split is authoritative about K_sc, the empty mask included — there it
  // means "K_sc = none", not "fill it in from the flags".
  const lr_kernel_spec total{p.include_hartree, p.include_exchange,
                             p.include_gw_sigma(), p.gw_full()};
  const bool split_requested = !p.pert_kernel.empty();
  k.do_pert = p.two_step();
  k.sc   = (split_requested || !p.sc_kernel.empty()) ? p.sc_kernel : total;
  k.pert = k.do_pert ? p.pert_kernel : lr_kernel_spec{};

  // A split run whose K_sc is HSEX owes the remainder the static counter-term
  // +ΔDm ⊙ W_c(0), which is an exchange-shaped evaluation the pert mask does
  // not carry. It exists exactly when the screened kernel sits in K_sc and the
  // total kernel it is being subtracted from is the bare-exchange one.
  k.pert_sex_counterterm = k.do_pert && p.exchange_static_W && k.sc.exchange;

  k.sc_hf      = k.sc.hartree || k.sc.exchange || p.include_xc;
  k.pert_hf    = k.pert.hartree || k.pert.exchange || k.pert_sex_counterterm;
  k.sc_sigma   = k.sc.has_sigma();
  k.pert_sigma = k.pert.has_sigma();

  k.need_hf          = k.sc_hf || k.pert_hf;
  k.include_gw_sigma = k.sc_sigma || k.pert_sigma;
  k.gw_full          = k.sc.sigma_G_dW || k.pert.sigma_G_dW;

  k.split_F     = k.sc_hf && k.pert_hf;
  k.split_Sigma = k.sc_sigma && k.pert_sigma;

  k.qp_mode   = p.qp_mode();
  k.has_Vcorr = k.qp_mode;
  // The ΔΣ fed to the Dyson equation is the total (sc + pert) one; the ΔΣ that
  // is mixed and tracked for convergence is the sc-channel one only.
  k.has_Sigma    = k.include_gw_sigma && !k.qp_mode;
  k.has_Sigma_sc = k.sc_sigma && !k.qp_mode;
  return k;
}

} // namespace


// Distribution flow through the LR-GW pipeline
// Two distribution patterns:
//   τ-dist (q-local):  pgrid = {tpools, 1, np_P, np_Q}  — q undivided
//   q-dist (τ-local):  pgrid = {1, nqpools, np_P, np_Q}  — tau/omega undivided
//
// Everything on the ω axis lives on q-dist as well
// (solvers::lr_scr_coulomb_t::W_omega_dist returns it), which is what lets both
// Fourier transforms write straight into their output with no staging buffer.
//
//   W_c in:              (w,q,P,Q), q-dist (solvers::lr_scr_coulomb_t::W_omega_dist)
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
template<THC_ERI THC_t, typename dW_t>
void lr_driver::lr_setup_W(dW_t* dW_wqPQ_in, THC_t& thc, bool gw_full,
                           solvers::lr_scr_coulomb_t* lr_scr,
                           std::optional<dW_t>& opt_dW_full_wqPQ,
                           std::optional<dW_t>& opt_dW_tRPQ) {
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

template<THC_ERI THC_t, typename dW_t>
void lr_driver::lr_setup(
    const sArray_t<Array_view_5D_t>& sG_tskij,
    THC_t& thc,
    dW_t* dW_wqPQ_in,
    const lr_params& p) {

  const lr_kernel_split k = make_kernel_split(p);

  utils::check(p.iter_params.alg == "damping" || p.iter_params.alg == "DIIS",
               "lr_driver::lr_setup: unknown iter_alg '{}'. Must be 'damping' or 'DIIS'.",
               p.iter_params.alg);

  // Split-kernel (two-step) schedule: K = K_sc + K_pert, with K_sc resummed to
  // all orders by the inner SCF and K_pert applied `pert_order` times, each on a
  // converged inner solve. An empty K_pert or pert_order = 0 leaves K_pert empty
  // and the loop is the single-kernel path it has always been.
  utils::check(p.pert_order >= 0,
               "lr_driver::lr_setup: pert_order must be >= 0, got {}.", p.pert_order);
  utils::check(p.pert_order == 0 || !p.pert_kernel.empty(),
               "lr_driver::lr_setup: pert_order = {} >= 1 requires a non-empty "
               "perturbative kernel.", p.pert_order);
  if (k.do_pert) {
    utils::check(!k.sc.overlaps(k.pert),
                 "lr_driver::lr_setup: the self-consistent ({}) and perturbative "
                 "({}) kernels share components. pert_kernel must be the "
                 "DIFFERENCE kernel(total) \\ kernel(sc), not the total kernel — "
                 "a shared component would be applied twice, once resummed and "
                 "once as a frozen source. Build it with kernel_diff().",
                 k.sc.to_string(), k.pert.to_string());
    utils::check(!p.has_deltax(),
                 "lr_driver::lr_setup: the split-kernel schedule does not support "
                 "the DeltaX IBC correction (each kernel evaluator adds its own "
                 "IBC term, so a two-pass schedule would double-count it).");
    utils::check(p.DeltaV_qPQ == nullptr,
                 "lr_driver::lr_setup: the split-kernel schedule does not support "
                 "the DeltaV_qPQ perturbation.");
    utils::check(!p.include_xc,
                 "lr_driver::lr_setup: the split-kernel schedule is incompatible "
                 "with include_xc.");
    utils::check(!p.split_sigma_terms,
                 "lr_driver::lr_setup: the split-kernel schedule is incompatible "
                 "with split ΔΣ terms.");
    utils::check(!k.qp_mode,
                 "lr_driver::lr_setup: the split-kernel schedule is incompatible "
                 "with qp_static mode.");
  }

  // Outer-loop acceleration. The defaults (alg "damping", tol = 0) switch every
  // flag below off, so nothing is allocated and the outer loop is the plain
  // Neumann series.
  const std::string& outer_alg = p.outer_accel.iter.alg;
  const double outer_tol       = p.outer_accel.tol;
  const bool outer_diis_on = (outer_alg == "DIIS");
  // "the outer loop is no longer an order-pert_order truncation": drives the
  // logging, the ΔDm stage buffer and the checkpoint provenance fields.
  const bool outer_track = p.outer_accel.active();
  utils::check(outer_alg == "damping" || outer_alg == "DIIS",
               "lr_driver::lr_setup: unknown outer iter_alg '{}'. Must be "
               "'damping' or 'DIIS'.", outer_alg);
  utils::check(outer_tol >= 0.0,
               "lr_driver::lr_setup: the outer tolerance must be >= 0, got {}.",
               outer_tol);
  utils::check(!outer_track || k.do_pert,
               "lr_driver::lr_setup: outer-loop acceleration requires a "
               "split-kernel run (pert_order >= 1 with a non-empty K_pert). "
               "There is no outer sequence to accelerate otherwise.");

  if (k.include_gw_sigma) {
    utils::check(dW_wqPQ_in != nullptr && p.eps_inv_head != nullptr,
                 "lr_driver::lr_setup: a GW self-energy is active but dW or "
                 "eps_inv_head is null.");
  }
  if (p.include_xc) {
    utils::check(p.include_hartree,
                 "lr_driver::lr_setup: include_xc = true requires include_hartree = true.");
    utils::check(!p.include_exchange,
                 "lr_driver::lr_setup: include_xc = true is incompatible with "
                 "include_exchange = true. The semilocal xc kernel contracts with the "
                 "diagonal density response only; LR-DFT is include_hartree = true, "
                 "include_exchange = false.");
    utils::check(!p.include_gw_sigma(),
                 "lr_driver::lr_setup: include_xc = true is incompatible with a GW "
                 "self-energy (gw_mode != none): f_xc and ΔΣ_GW both carry the "
                 "correlation response, so the two together double-count it. "
                 "include_xc works only in the Hartree mode.");
  }
  // The SCF loop must run at least once: every output downstream, ΔG(τ)
  // included, is produced by the solve inside it.
  utils::check(p.max_iter >= 1,
               "lr_driver::lr_setup: max_iter must be >= 1, got {}.", p.max_iter);
  if (p.split_sigma_terms) {
    utils::check(k.gw_full,
                 "lr_driver::lr_setup: split ΔΣ terms require the Σ2 (-G⊙ΔW) component.");
    utils::check(p.max_iter == 1,
                 "lr_driver::lr_setup: split ΔΣ terms are only meaningful for a "
                 "one-shot solve (max_iter=1), got max_iter={}.", p.max_iter);
    utils::check(!p.has_deltax(),
                 "lr_driver::lr_setup: split ΔΣ terms do not support the DeltaX IBC "
                 "correction (term 2 has no IBC path).");
  }
  if (p.exchange_static_W) {
    utils::check(p.include_exchange,
                 "lr_driver::lr_setup: exchange_static_W = true requires "
                 "include_exchange = true. HSEX substitutes V -> V + W_c(iν=0) in "
                 "the exchange contraction; there is nothing to substitute without "
                 "an exchange channel.");
    utils::check(dW_wqPQ_in != nullptr,
                 "lr_driver::lr_setup: exchange_static_W = true needs W_c(iω); "
                 "dW is null.");
    // Per-channel, not global: a split run deliberately puts Σ in K_pert.
    utils::check(!k.sc.has_sigma(),
                 "lr_driver::lr_setup: exchange_static_W puts W_c(iν=0) in the "
                 "exchange kernel, and the self-consistent channel also carries "
                 "Σ ({}), whose W_c(τ) contains the same ν=0 plane. That "
                 "double-counts the static screening. Use HSEX as a standalone "
                 "kernel, or as two_step_inner_method with Σ in K_pert.",
                 k.sc.to_string());
    if (k.do_pert) {
      // The counter-term is defined against the whole exchange operator, so it
      // only closes if the bare-exchange slot is wholly inside K_sc.
      utils::check(k.sc.exchange && !k.pert.exchange,
                   "lr_driver::lr_setup: with exchange_static_W the exchange "
                   "channel must be wholly self-consistent (K_sc = {}, K_pert = "
                   "{}). The static counter-term that makes K_sc + K_pert sum "
                   "back to bare exchange is defined against the whole exchange "
                   "operator, not a share of it.",
                   k.sc.to_string(), k.pert.to_string());
    }
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
  if (k.qp_mode) {
    utils::check(k.include_gw_sigma,
                 "lr_driver::lr_setup: qp_static mode requires a GW self-energy.");
    utils::check(!p.split_sigma_terms,
                 "lr_driver::lr_setup: qp_static mode is incompatible with split ΔΣ terms.");
    utils::check(p.qp_static->sMO_skia != nullptr && p.qp_static->sE_ska != nullptr,
                 "lr_driver::lr_setup: qp_static mode requires sMO_skia and sE_ska.");
  }
  utils::check(!_setup_done,
               "lr_driver::lr_setup: called twice on the same driver. One driver "
               "serves one q-vector and one unperturbed state; construct a new "
               "one to change either.");

  const char* gw_mode_str = "none";
  switch (p.gw_mode) {
    case lr_gw_update_mode::none:    gw_mode_str = "none"; break;
    case lr_gw_update_mode::fixed_W: gw_mode_str = "fixed_W"; break;
    case lr_gw_update_mode::full:    gw_mode_str = "full"; break;
  }

  app_log(1, "Starting Linear Response SCF loop:");
  app_log(1, "  max_iter = {}", p.max_iter);
  app_log(1, "  tol = {:.2e}", p.tol);
  app_log(1, "  fix_density = {}", p.fix_density ? "true" : "false");
  app_log(1, "  include_hartree = {}", p.include_hartree ? "true" : "false");
  app_log(1, "  include_exchange = {}", p.include_exchange ? "true" : "false");
  app_log(1, "  gw_mode = {}", gw_mode_str);
  app_log(1, "  K_sc  (self-consistent) = {}", k.sc.to_string());
  if (k.do_pert) {
    app_log(1, "  K_pert (perturbative)   = {}", k.pert.to_string());
    app_log(1, "  pert_order = {}  ({} stage(s); max_iter counts total inner iterations)",
            p.pert_order, p.pert_order + 1);
  }
  app_log(1, "  include_xc = {}", p.include_xc ? "true" : "false");
  if (p.exchange_static_W) {
    app_log(1, "  exchange kernel = V + W_c(iν=0)  [static screened exchange]");
    if (k.pert_sex_counterterm)
      app_log(1, "    K_pert additionally carries the static counter-term "
                 "+ΔDm ⊙ W_c(iν=0), so K_sc + K_pert restores bare exchange");
  }
  app_log(1, "  qp_static_sigma = {}", k.qp_mode ? "true" : "false");
  app_log(1, "  iter_alg = {}", p.iter_params.alg);
  app_log(1, "  mixing = {:.2f}", p.mixing());
  if (p.use_diis()) {
    app_log(1, "  max_subsp_size = {}", p.iter_params.max_subsp_size);
    app_log(1, "  diis_warmup = {}", p.iter_params.diis_warmup);
  }
  if (outer_track) {
    app_log(1, "  outer (K_pert source) acceleration:");
    app_log(1, "    outer_alg = {}", outer_alg);
    if (outer_diis_on)
      app_log(1, "    outer_subsp = {}, outer_warmup = {}, outer_min_subsp = {} "
                 "(first extrapolation at outer step {})",
              p.outer_accel.iter.max_subsp_size, p.outer_accel.iter.diis_warmup,
              p.outer_accel.min_subsp,
              std::max(p.outer_accel.iter.diis_warmup + 2, p.outer_accel.min_subsp));
    app_log(1, "    outer_tol = {:.2e}", outer_tol);
    if (outer_diis_on) {
      app_log(1, "    [NOTE] with outer acceleration the source is a combination "
                 "of previous sources, so the result is NOT an order-{} "
                 "truncation of K_pert: it is an accelerated iterate toward the "
                 "FULL K_sc + K_pert fixed point, and pert_order is an iteration "
                 "cap.", p.pert_order);
    }
    if (outer_tol > 0.0 && p.tol > 0.1 * outer_tol) {
      app_log(1, "    [WARNING] the inner tol ({:.2e}) is not at least a decade "
                 "below outer_tol ({:.2e}); the outer residual then measures "
                 "inner-solve noise rather than the outer error.", p.tol, outer_tol);
    }
  }

  // ΔΣ-sized shared arrays on top of the total ΔΣ the base estimate already
  // lists.
  std::vector<std::string> extra_sigma;
  if (k.split_Sigma) {
    extra_sigma.push_back("sDeltaSigma (sc channel)");
    extra_sigma.push_back("sDeltaSigma (pert channel)");
  }

  // ΔΣ-sized striped previous iterates: the inner loop's tracked ΔΣ, plus the
  // outer accelerator's previous source when it mixes one.
  const long n_sigma_prev = (k.has_Sigma_sc ? 1 : 0)
                          + ((outer_diis_on && k.pert_sigma) ? 1 : 0);

  // DIIS histories, for the memory report. Each subspace entry stores a trial
  // AND a residual vector of every quantity the accelerator mixes.
  lr_diis_hist_t inner_hist, outer_hist;
  if (p.use_diis() && (k.sc_hf || k.sc_sigma || k.has_Vcorr)) {
    inner_hist.depth = static_cast<long>(p.iter_params.max_subsp_size);
    inner_hist.n_F = k.has_Vcorr ? 2 : 1;   // ΔF (+ the static ΔV_QPGW in qp mode)
    inner_hist.n_Sigma = k.has_Sigma_sc ? 1 : 0;
  }
  if (outer_diis_on) {
    outer_hist.depth = static_cast<long>(p.outer_accel.iter.max_subsp_size);
    outer_hist.n_F     = k.pert_hf ? 1 : 0;
    outer_hist.n_Sigma = k.pert_sigma ? 1 : 0;
  }

  // Estimate the persistent large-array memory footprint for this path, then
  // summarize the MPI distribution patterns the large arrays use.
  print_memory_estimate(thc.Np(), k.include_gw_sigma, k.gw_full, extra_sigma,
                        n_sigma_prev, inner_hist, outer_hist,
                        p.fix_density && _lr_dyson.is_q_gamma(), p.exchange_static_W);
  print_distribution_summary(thc.Np(), k.include_gw_sigma, k.gw_full);

  _Timer.start("LR_DRIVER_SETUP");

  // Solvers. Each latches the perturbation q at construction and caches a
  // workspace, so they are built once here and reused by every lr_solve_one.
  if (k.need_hf && !_lr_hf) {
    _lr_hf = std::make_unique<solvers::lr_hf>(_mpi, _MF, _lr_dyson.q_vec(), p.hf_div_treatment);
  }
  // The perturbative channel gets its own lr_hf for the same reason lr_gw does:
  // lr_hf caches U(R) keyed on the exchange kernel it was first built with, and an
  // HSEX split run contracts V + W_c(0) in K_sc but -W_c(0) in K_pert.
  if (k.pert_hf && !_lr_hf_pert) {
    _lr_hf_pert = std::make_unique<solvers::lr_hf>(_mpi, _MF, _lr_dyson.q_vec(), p.hf_div_treatment);
  }
  if (k.sc_sigma) {
    _lr_gw = std::make_unique<solvers::lr_gw>(_dyson.FT(), _lr_dyson.q_vec(), p.div_treatment);
  }
  // The perturbative channel gets its own lr_gw: the cached workspace is keyed
  // on the (term1, term2) combination it was first used with, and the two
  // channels generally run different combinations (e.g. Σ1 in K_sc, Σ2 in K_pert).
  if (k.pert_sigma) {
    _lr_gw_pert = std::make_unique<solvers::lr_gw>(_dyson.FT(), _lr_dyson.q_vec(), p.div_treatment);
  }
  // Split-term mode (one-shot G0W0) needs a second lr_gw for term 2: each solver
  // caches a workspace keyed on its (term1,term2) usage.
  if (p.split_sigma_terms) {
    _lr_gw2 = std::make_unique<solvers::lr_gw>(_dyson.FT(), _lr_dyson.q_vec(), p.div_treatment);
  }
  if (k.gw_full) {
    _lr_pi = std::make_unique<solvers::lr_rpa_pi>(_lr_dyson.q_vec());
    _lr_scr = std::make_unique<solvers::lr_scr_coulomb_t>(_dyson.FT(), _lr_dyson.q_vec());
  }

  // HSEX kernel, taken BEFORE lr_setup_W: that call consumes dW_wqPQ_in (it is
  // either moved into W_full or released by the ω→τ transform), and it also adds
  // Z(q) to the ω array in place, which would double-count the bare V here.
  if (p.exchange_static_W) {
    _Timer.start("LR_DRIVER_SETUP_WC0");
    _opt_dWc0_qPQ.emplace(lr_precompute_Wc_static(*dW_wqPQ_in));
    // eps_inv_head_w holds ε⁻¹ - 1, q→0 extrapolated, on the PH-symmetric ω
    // half-axis whose index 0 is ν = 0.
    utils::check(p.eps_inv_head_w != nullptr,
                 "lr_driver::lr_setup: exchange_static_W needs eps_inv_head_w; "
                 "it is null.");
    _hsex_head_factor = 1.0 + (*p.eps_inv_head_w)(0).real();
    app_log(1, "  HSEX ε⁻¹(iν=0) = {:.6f}", _hsex_head_factor);
    _Timer.stop("LR_DRIVER_SETUP_WC0");
  }

  if (k.include_gw_sigma) {
    lr_setup_W(dW_wqPQ_in, thc, k.gw_full, _lr_scr.get(),
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
  if (k.gw_full) {
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

  _Timer.start("LR_DRIVER_SETUP_MISC");
  // Initialize DIIS if requested. Built once: lr_solve_one reset()s the subspace
  // rather than rebuilding it, so the (job-wide, striped) history is allocated
  // exactly once no matter how many perturbations follow.
  if (p.use_diis()) {
    _lr_diis = std::make_unique<lr_diis>(
        p.iter_params.max_subsp_size, p.iter_params.diis_warmup, p.mixing());
  }
  // The outer accelerator is a separate object with its own subspace, history
  // and warmup: it is keyed on the outer step index while the inner one restarts
  // at every stage boundary, and the two must share nothing.
  if (outer_diis_on) {
    _outer_diis = std::make_unique<lr_diis>(
        p.outer_accel.iter.max_subsp_size, p.outer_accel.iter.diis_warmup,
        p.outer_accel.iter.mixing, p.outer_accel.min_subsp);
  }
  _Timer.stop("LR_DRIVER_SETUP_MISC");

  _Timer.start("LR_DRIVER_SETUP_ALLOC");
  // Element partition of the node-replicated band arrays over the *global*
  // comm: the DIIS history, the "previous iterate" copies and their norms are
  // all elementwise, so each rank handles one slice and the whole job stores
  // each of those quantities once instead of once per node.
  _pmap = utils::make_part_map(*_mpi);

  // Allocate array for previous density matrix (for convergence check).
  // Kept whole (0.1 GB) — it only feeds a norm, and the ΔDm norm stays on the
  // node_comm path.
  _sDeltaDm_prev_skij.emplace(math::shm::make_shared_array<Array_view_4D_t>(
      *_mpi, {_ns, _nkpts_ibz, _nbnd, _nbnd}));

  // Previous ΔF / ΔΣ / ΔV_QPGW (for damping/DIIS and the difference norms) are
  // rank-private slices of the flattened arrays, in the `_pmap` partition. Only
  // the sc channel is mixed, so the tracked ΔΣ is that channel's; it is tracked
  // at all only when the sc channel carries a Σ and we are not in qp mode (qp
  // mode tracks the static ΔV_QPGW in its place). An inactive quantity gets
  // n_flat = 0.
  const long nF = _ns * _nkpts_ibz * _nbnd * _nbnd;
  _DeltaF.alloc(_pmap, nF);
  _DeltaSigma.alloc(_pmap, k.has_Sigma_sc ? _nts * nF : 0);
  _DeltaVcorr.alloc(_pmap, k.has_Vcorr ? nF : 0);

  // Static ΔV_QPGW tracked in qp mode.
  _sDeltaVcorr_skij.emplace(k.has_Vcorr
      ? math::shm::make_shared_array<Array_view_4D_t>(*_mpi, {_ns, _nkpts_ibz, _nbnd, _nbnd})
      : math::shm::make_shared_array<Array_view_4D_t>(*_mpi, {1, 1, 1, 1}));

  // Split-kernel channel buffers. sDeltaF_skij / sDeltaSigma_tskij always hold
  // the TOTAL (sc + pert) quantities that the Dyson RHS and the checkpoint dump
  // consume. A per-channel buffer exists only for a *split* quantity; otherwise
  // the sole contributing channel writes the caller's array directly, so the
  // single-kernel path — and every composition that splits only one of the two —
  // forms no sum at all.
  if (k.split_F) {
    _sDeltaF_sc.emplace(math::shm::make_shared_array<Array_view_4D_t>(
        *_mpi, {_ns, _nkpts_ibz, _nbnd, _nbnd}));
    _sDeltaF_pert.emplace(math::shm::make_shared_array<Array_view_4D_t>(
        *_mpi, {_ns, _nkpts_ibz, _nbnd, _nbnd}));
  }
  if (k.split_Sigma) {
    _sDeltaSigma_sc.emplace(math::shm::make_shared_array<Array_view_5D_t>(
        *_mpi, {_nts, _ns, _nkpts_ibz, _nbnd, _nbnd}));
    _sDeltaSigma_pert.emplace(math::shm::make_shared_array<Array_view_5D_t>(
        *_mpi, {_nts, _ns, _nkpts_ibz, _nbnd, _nbnd}));
  }

  // Outer-loop buffers. The previous-source pair exists only for the
  // accelerator (it is what the extrapolation and its residual are measured
  // against) and is striped exactly like the inner loop's; the ΔDm stage buffer
  // only for the tolerance test, and it stays whole because its norm is the
  // termination criterion and is taken on the node_comm path. A tolerance-only
  // run therefore never allocates a ΔΣ-sized buffer.
  if (outer_diis_on) {
    if (k.pert_hf)    _DeltaF_pert.alloc(_pmap, nF);
    if (k.pert_sigma) _DeltaSigma_pert.alloc(_pmap, _nts * nF);
  }
  if (outer_tol > 0.0)
    _sDeltaDm_stage_prev.emplace(math::shm::make_shared_array<Array_view_4D_t>(
        *_mpi, {_ns, _nkpts_ibz, _nbnd, _nbnd}));
  _Timer.stop("LR_DRIVER_SETUP_ALLOC");

  // DeltaX IBC correction setup
  if (p.has_deltax()) {
    _Timer.start("LR_DRIVER_SETUP_IBC");
    app_log(2, "  DeltaX IBC correction: building lr_ibc_DeltaX...");

    utils::memlog("lr_driver::lr_setup: before build_lr_ibc");
    _opt_ibc.emplace(build_lr_ibc(
        *_mpi, _MF, thc,
        p.sDeltaX_left->local(), p.sDeltaX_right->local(),
        _lr_dyson.q_vec(), _lr_dyson.kpq_map(),
        p.Dm_ab, &sG_tskij,
        _opt_dW_tRPQ ? &(*_opt_dW_tRPQ) : nullptr,
        k.sc.hartree, k.sc.exchange, k.sc_sigma,
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


template<THC_ERI THC_t>
std::tuple<int, double> lr_driver::lr_solve_one(
    sArray_t<Array_view_5D_t>& sDeltaG_tskij,
    sArray_t<Array_view_4D_t>& sDeltaDm_skij,
    sArray_t<Array_view_4D_t>& sDeltaF_skij,
    sArray_t<Array_view_5D_t>* sDeltaSigma_tskij,
    const sArray_t<Array_view_5D_t>& sG_tskij,
    const sArray_t<Array_view_4D_t>& sDeltaH0_skij,
    THC_t& thc,
    const lr_params& p,
    sArray_t<Array_view_5D_t>* sDeltaSigma_term2_tskij,
    sArray_t<Array_view_4D_t>* sDeltaVcorr_out_skij,
    nda::array<ComplexType, 4>* DeltaF_ibc_out,
    nda::array<ComplexType, 4>* F_PQ_out,
    nda::array<ComplexType, 4>* DeltaF_PQ_out,
    int* n_pert_applied_out) {

  utils::check(_setup_done, "lr_driver::lr_solve_one: call lr_setup first.");
  // Read once: the loop below refers to these on nearly every line, and the
  // predicates are cheap but not free.
  const lr_kernel_split k = make_kernel_split(p);
  const bool include_gw_sigma = k.include_gw_sigma;
  const bool has_Vcorr   = k.has_Vcorr;
  const bool has_Sigma   = k.has_Sigma;
  const bool has_Sigma_sc = k.has_Sigma_sc;
  const bool do_pert     = k.do_pert;
  const double mixing    = p.mixing();
  const std::string& outer_alg = p.outer_accel.iter.alg;
  const double outer_tol   = p.outer_accel.tol;
  const bool outer_diis_on = (outer_alg == "DIIS");
  const bool outer_track   = p.outer_accel.active();
  auto& sDeltaDm_prev_skij = *_sDeltaDm_prev_skij;
  auto& sDeltaVcorr_skij   = *_sDeltaVcorr_skij;
  auto& opt_ibc            = _opt_ibc;
  auto& sS_skij            = _dyson.sS_skij();
  // The kernels lr_hf contracts under HSEX, empty otherwise: V + W_c(iν=0) in
  // the self-consistent channel, -W_c(iν=0) in a split run's remainder. Their
  // heads ε⁻¹(0) and 1 - ε⁻¹(0) sum back to bare exchange's 1.
  using hsex_kernel_t = solvers::lr_hf::hsex_kernel_t;
  std::optional<hsex_kernel_t> hsex, hsex_ct;
  if (_opt_dWc0_qPQ) {
    hsex = hsex_kernel_t{&(*_opt_dWc0_qPQ), _hsex_head_factor,
                         hsex_kernel_t::kernel_e::V_plus_Wc0};
    if (k.pert_sex_counterterm)
      hsex_ct = hsex_kernel_t{&(*_opt_dWc0_qPQ), 1.0 - _hsex_head_factor,
                              hsex_kernel_t::kernel_e::minus_Wc0};
  }

  if (include_gw_sigma) {
    utils::check(sDeltaSigma_tskij != nullptr,
                 "lr_driver::lr_solve_one: a GW self-energy is active but "
                 "sDeltaSigma_tskij is null.");
  }
  utils::check((sDeltaSigma_term2_tskij != nullptr) == p.split_sigma_terms,
               "lr_driver::lr_solve_one: sDeltaSigma_term2_tskij presence must match "
               "the p.split_sigma_terms lr_setup was given.");

  // Per-channel write targets. When a quantity is not split, both channels
  // resolve to the caller's array (at most one of them is ever active).
  auto& sDeltaF_sc_skij   = k.split_F ? *_sDeltaF_sc   : sDeltaF_skij;
  auto& sDeltaF_pert_skij = k.split_F ? *_sDeltaF_pert : sDeltaF_skij;
  sArray_t<Array_view_5D_t>* pDeltaSigma_sc =
      !k.sc_sigma ? nullptr : (k.split_Sigma ? &(*_sDeltaSigma_sc) : sDeltaSigma_tskij);
  sArray_t<Array_view_5D_t>* pDeltaSigma_pert =
      !k.pert_sigma ? nullptr : (k.split_Sigma ? &(*_sDeltaSigma_pert) : sDeltaSigma_tskij);

  // Reset every quantity carried across SCF iterations, so this solve cannot see
  // the previous perturbation's state. set_zero ends with fence + node_sync.
  _Timer.start("LR_DRIVER_SETUP_MISC");
  sDeltaF_skij.set_zero();
  if (sDeltaSigma_tskij) sDeltaSigma_tskij->set_zero();
  // ΔΣ term 2 is zeroed by lr_gw::evaluate_sigma_DeltaW before it accumulates,
  // but do not rely on that from here.
  if (sDeltaSigma_term2_tskij) sDeltaSigma_term2_tskij->set_zero();
  if (has_Vcorr) sDeltaVcorr_skij.set_zero();
  if (_sDeltaF_sc)       _sDeltaF_sc->set_zero();
  if (_sDeltaF_pert)     _sDeltaF_pert->set_zero();
  if (_sDeltaSigma_sc)   _sDeltaSigma_sc->set_zero();
  if (_sDeltaSigma_pert) _sDeltaSigma_pert->set_zero();
  // The outer sequence starts from S_0 = 0, which is a genuine iterate: the
  // first inner solve is exactly the one with no source. So the first outer
  // residual S_1 - 0 needs no special casing — the histories start zeroed.
  if (_sDeltaDm_stage_prev) _sDeltaDm_stage_prev->set_zero();
  sDeltaDm_prev_skij.set_zero();
  // Not strictly required — every read of these is guarded by stage_iter > 1 —
  // but it makes a solve depend on nothing but its own ΔH0.
  _DeltaF.zero(); _DeltaSigma.zero(); _DeltaVcorr.zero();
  _DeltaF_pert.zero(); _DeltaSigma_pert.zero();
  if (p.use_diis()) _lr_diis->reset();
  if (outer_diis_on) _outer_diis->reset();
  // No clock is reset here: the SCF timers and the solvers' sub-clocks both
  // accumulate over every perturbation of the call, so the table printed after
  // the last one is the cost of the whole batch. Each intermediate table is
  // therefore a running total, and one perturbation's cost is the difference
  // between consecutive tables.
  _mpi->comm.barrier();
  _Timer.stop("LR_DRIVER_SETUP_MISC");

  // Timer keys of one Σ channel. The two channels run the same evaluators, so
  // each gets its own clocks — the whole point of the split is a cost claim,
  // and it can only be read off if sc and pert are timed apart.
  struct sigma_clocks { const char *pi, *w, *sigma; };
  constexpr sigma_clocks sc_clocks{"LR_GW_PI", "LR_GW_W", "LR_GW_SIGMA"};
  constexpr sigma_clocks pert_clocks{"LR_GW_PI_PERT", "LR_GW_W_PERT",
                                     "LR_GW_SIGMA_PERT"};

  // Evaluate the Σ components of `mask` into `sSigma_out`, overwriting it. Each
  // divergence correction is applied by the evaluator that owns the term it
  // corrects, so passing the overlap and the heads is all this has to do.
  auto eval_sigma_channel = [&](lr_kernel_spec const& mask,
                                solvers::lr_gw& gw_solver,
                                sArray_t<Array_view_5D_t>& sSigma_out,
                                const lr_ibc_DeltaX* ibc_ptr,
                                sigma_clocks const& clk) {
    utils::check(mask.has_sigma(),
                 "lr_driver::lr_solve_one: eval_sigma_channel called with a Σ-free "
                 "kernel mask ({}).", mask.to_string());
    if (!mask.sigma_G_dW) {
      // Term 1 only: ΔΣ = -ΔG ⊙ W_c
      _Timer.start(clk.sigma);
      // The term-1 divergence correction is applied inside the evaluator.
      auto S_loc = sS_skij.local();
      gw_solver.evaluate_sigma_DeltaG(
          sSigma_out, sDeltaG_tskij.local(), *_opt_dW_tRPQ, thc, ibc_ptr,
          &S_loc, p.eps_inv_head);
      _Timer.stop(clk.sigma);
      _mpi->comm.barrier();
      return;
    }

    // Step 3b: ΔP = -ΔG·G - G·ΔG
    _Timer.start(clk.pi);
    auto dDeltaPi_tqPQ = _lr_pi->evaluate_lr_Pi(
        sG_tskij.local(), sDeltaG_tskij.local(), thc,
        *_opt_dG_tsRPQ, *_opt_dG_mtau_tsRPQ, ibc_ptr);
    _mpi->comm.barrier();
    _Timer.stop(clk.pi);

    // Step 3c-3d: ΔW_c(τ) via solve_lr_dyson_W (in-place, uses cached W_full)
    _Timer.start(clk.w);
    _lr_scr->solve_lr_dyson_W(dDeltaPi_tqPQ, *_opt_dW_full_wqPQ, thc);
    // dDeltaPi_tqPQ now contains ΔW_c(τ) in q-local distribution
    auto& dDeltaW_tqPQ = dDeltaPi_tqPQ;  // alias for clarity

    // Extract Δeps_inv_head from ΔW for divergence correction term 2 (q_pert=0 only)
    nda::array<ComplexType, 1> delta_eps_inv_head;
    if (p.div_treatment != "ignore_g0" && is_q_gamma()) {
      auto [delta_eps_inv_q, delta_head] =
          solvers::div_utils::eps_inv_head_t(
              dDeltaW_tqPQ, thc, *thc.MF(), _dyson.FT(), p.div_treatment);
      delta_eps_inv_head = std::move(delta_head);
    }

    _mpi->comm.barrier();
    _Timer.stop(clk.w);

    // Step 3e-3f: ΔΣ = -ΔG ⊙ W_c - G ⊙ ΔW.
    // ΔW stays in (t,q,P,Q): the Σ evaluator consumes one τ slice at a time,
    // which is contiguous in this layout and matches term 1's dW_tRPQ.
    _Timer.start(clk.sigma);
    if (p.split_sigma_terms) {
      // One-shot G0W0: compute the two terms separately, then store
      //   sDeltaSigma_tskij       = term1 + term2  (total ΔΣ, same as fused)
      //   sDeltaSigma_term2_tskij = term2 (G0·dW0)  [written as DeltaSigma_GdW]
      // term 1 (-ΔG⊙W_c + div) and term 2 (-G⊙ΔW + div) use separate solver
      // instances (gw_solver / _lr_gw2, built once by lr_setup) — the workspace
      // is cached per (term1,term2) combination.
      auto S_loc = sS_skij.local();
      gw_solver.evaluate_sigma_DeltaG(
          sSigma_out, sDeltaG_tskij.local(), *_opt_dW_tRPQ, thc, ibc_ptr,
          &S_loc, p.eps_inv_head);
      _lr_gw2->evaluate_sigma_DeltaW(
          *sDeltaSigma_term2_tskij, sG_tskij.local(), dDeltaW_tqPQ, thc,
          *_opt_dG_tsRPQ, *_opt_dG_mtau_tsRPQ, &S_loc, &delta_eps_inv_head);
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
      auto S_loc = sS_skij.local();
      gw_solver.evaluate_sigma(
          sSigma_out, sDeltaG_tskij.local(), *_opt_dW_tRPQ,
          sG_tskij.local(), dDeltaW_tqPQ, thc,
          *_opt_dG_tsRPQ, *_opt_dG_mtau_tsRPQ, ibc_ptr,
          &S_loc, p.eps_inv_head, &delta_eps_inv_head);
    } else {
      // Term 2 only: ΔΣ = -G ⊙ ΔW. Reached only from the perturbative channel
      // of a split-kernel run whose K_sc already resums Σ1.
      auto S_loc = sS_skij.local();
      gw_solver.evaluate_sigma_DeltaW(
          sSigma_out, sG_tskij.local(), dDeltaW_tqPQ, thc,
          *_opt_dG_tsRPQ, *_opt_dG_mtau_tsRPQ, &S_loc, &delta_eps_inv_head);
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
    if (!k.split_F && !k.split_Sigma) return;
    _Timer.start("LR_TOTALS");
    // split_F / split_Sigma guarantee the corresponding optionals are engaged and
    // are exactly the per-channel buffers, so these are the same objects as
    // sDeltaF_sc_skij / pDeltaSigma_sc and friends.
    if (k.split_F) {
      sDeltaF_skij.win().fence();
      _sDeltaF_sc->win().fence();
      _sDeltaF_pert->win().fence();
      refresh_total(sDeltaF_skij, *_sDeltaF_sc, *_sDeltaF_pert);
    }
    if (k.split_Sigma) {
      sDeltaSigma_tskij->win().fence();
      _sDeltaSigma_sc->win().fence();
      _sDeltaSigma_pert->win().fence();
      refresh_total(*sDeltaSigma_tskij, *_sDeltaSigma_sc, *_sDeltaSigma_pert);
    }
    _mpi->comm.barrier();
    _Timer.stop("LR_TOTALS");
  };

  // Make an in-place outer mixing visible everywhere, exactly as the inner
  // epilogue does: each rank wrote only its own `_pmap` slice, so each node holds
  // one contiguous element run and an allgatherv among the node roots completes
  // every replica.
  auto outer_sync = [&](auto*... arrs) {
    auto fence = [](auto* q) { if (q) q->win().fence(); };
    (fence(arrs), ...);
    _mpi->node_comm.barrier();
    if (_mpi->node_comm.root()) {
      auto complete = [&](auto* q) {
        if (q) utils::complete_node_slices(_mpi->internode_comm, _pmap,
                                              q->local().data(), q->local().size());
      };
      (complete(arrs), ...);
    }
    (fence(arrs), ...);
    _mpi->comm.barrier();
  };

  // dst <- src on the node-replicated shared window. Only the ΔDm stage buffer
  // still needs this: the mixed sources are stored striped.
  auto outer_save = [&](auto& dst, auto& src) {
    dst.win().fence();
    if (_mpi->node_comm.root()) dst.local() = src.local();
    dst.win().fence();
    _mpi->node_comm.barrier();
  };

  // ‖A - A_prev‖ over the node, broadcast so every rank agrees. Serves the ΔDm
  // stage buffer, whose norm is the outer termination criterion.
  auto outer_diff_norm = [&](auto& A, auto& A_prev) {
    auto nrm = utils::lr_distributed_norm(
        _mpi->node_comm, A.local(), A_prev.local(), true);
    double d = nrm.second;
    _mpi->comm.broadcast_n(&d, 1, 0);
    return d;
  };

  _Timer.start("LR_SCF");

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

  for (iter = 1; iter <= p.max_iter; ++iter) {

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
      _DeltaF.prev = _DeltaF.slice(sDeltaF_sc_skij.local());
      if (has_Vcorr) {
        _DeltaVcorr.prev = _DeltaVcorr.slice(sDeltaVcorr_skij.local());
      } else if (has_Sigma_sc) {
        _DeltaSigma.prev = _DeltaSigma.slice(pDeltaSigma_sc->local());
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
        sDeltaDm_skij, sDeltaH0_skij,
        sDeltaF_skij, dyson_sigma,
        p.fix_density, dyson_vcorr);

    // The solve leaves ΔG(τ) distributed; replicating it is the single most
    // expensive step of the Dyson phase, so it happens only where something
    // reads it. k.include_gw_sigma is loop-invariant, so which iterations
    // replicate is fixed before the loop starts, not discovered inside it —
    // the Σ evaluators can assume sDeltaG_tskij is current, and the tail below
    // knows from the same flag whether the converged ΔG(τ) still needs one.
    if (k.include_gw_sigma) _lr_dyson.materialize_DeltaG_tau(sDeltaG_tskij);
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
    if (k.sc_hf) {
      _Timer.start("LR_HF");
      const lr_ibc_DeltaX* ibc_ptr = opt_ibc ? &(*opt_ibc) : nullptr;
      _lr_hf->evaluate(sDeltaF_sc_skij, sDeltaDm_skij, thc, sS_skij.local(),
                       k.sc.hartree, k.sc.exchange, ibc_ptr,
                       p.DeltaV_qPQ, p.Dm_ab, nullptr, p.include_xc,
                       hsex ? &(*hsex) : nullptr);
      _Timer.stop("LR_HF");
      _mpi->comm.barrier();
    }

    // Step 3: Compute the K_sc LR GW self-energy
    if (k.sc_sigma) {
      const lr_ibc_DeltaX* ibc_ptr = opt_ibc ? &(*opt_ibc) : nullptr;
      eval_sigma_channel(k.sc, *_lr_gw, *pDeltaSigma_sc, ibc_ptr, sc_clocks);
    }

    // Step 3g (qp mode): statify the dynamic ΔΣ(iω) into the static ΔV_QPGW(k)
    // using the frozen QP orbitals/energies. ΔV_QPGW is the tracked/mixed static
    // quantity and enters the Dyson RHS at the next iteration.
    if (k.qp_mode) {
      _Timer.start("LR_QPGW_STATIC");
      auto sVcorr = lr_qp_approx(
          *sDeltaSigma_tskij, *p.qp_static->sMO_skia, *p.qp_static->sE_ska,
          p.qp_static->mu, _lr_dyson.kpq_map(), is_q_gamma(),
          *_dyson.FT(), p.qp_static->qp_params);
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
    if (stage_iter > 1 && (k.sc_hf || k.sc_sigma || has_Vcorr)) {
      // The static second quantity mixed alongside ΔF is the dynamic ΔΣ in the
      // standard path, or the static ΔV_QPGW in qp mode.
      if (p.use_diis()) {
        // Striped DIIS: every rank of the global comm participates, each
        // operating on its `_pmap` element-slice of the shared ΔF/ΔΣ and writing
        // the mixed result back in place. Pass .local() views directly (in/out);
        // the "prev" arguments are already this rank's slice.
        if (has_Vcorr) {
          _lr_diis->next_step_combined(
              _mpi->comm, _pmap,
              sDeltaF_sc_skij.local(), _DeltaF.prev,
              sDeltaVcorr_skij.local(), _DeltaVcorr.prev, stage_iter);
        } else if (has_Sigma_sc) {
          _lr_diis->next_step_combined(
              _mpi->comm, _pmap,
              sDeltaF_sc_skij.local(), _DeltaF.prev,
              pDeltaSigma_sc->local(), _DeltaSigma.prev, stage_iter);
        } else {
          nda::array<ComplexType, 5> empty_sigma;
          nda::array<ComplexType, 1> empty_prev;
          _lr_diis->next_step_combined(
              _mpi->comm, _pmap,
              sDeltaF_sc_skij.local(), _DeltaF.prev,
              empty_sigma, empty_prev, stage_iter);
        }
      } else if (mixing < 1.0) {
        // Damping is elementwise too, so stripe it over the same partition —
        // one completion path then covers both algorithms.
        auto F_loc = _DeltaF.slice(sDeltaF_sc_skij.local());
        F_loc = mixing * F_loc + (1.0 - mixing) * _DeltaF.prev;
        if (has_Vcorr) {
          auto V_loc = _DeltaVcorr.slice(sDeltaVcorr_skij.local());
          V_loc = mixing * V_loc + (1.0 - mixing) * _DeltaVcorr.prev;
        } else if (has_Sigma_sc) {
          auto S_loc = _DeltaSigma.slice(pDeltaSigma_sc->local());
          S_loc = mixing * S_loc + (1.0 - mixing) * _DeltaSigma.prev;
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
      // the striping global, each element is mixed exactly once in the whole job.
      if (_mpi->node_comm.root()) {
        utils::complete_node_slices(_mpi->internode_comm, _pmap,
                                       sDeltaF_sc_skij.local().data(), _DeltaF.n_flat);
        if (has_Vcorr) {
          utils::complete_node_slices(_mpi->internode_comm, _pmap,
                                         sDeltaVcorr_skij.local().data(), _DeltaVcorr.n_flat);
        } else if (has_Sigma_sc) {
          utils::complete_node_slices(_mpi->internode_comm, _pmap,
                                         pDeltaSigma_sc->local().data(), _DeltaSigma.n_flat);
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
        _mpi->comm, _DeltaF.slice(sDeltaF_sc_skij.local()), _DeltaF.prev, stage_iter > 1);
    double norm_DeltaF = norms_F.first;
    double norm_DeltaF_diff = norms_F.second;

    // Compute norms of the tracked static second quantity (dynamic ΔΣ in the
    // standard path; static ΔV_QPGW in qp mode) for logging/convergence.
    double norm_DeltaSigma = 0.0;
    double norm_DeltaSigma_diff = 0.0;
    if (has_Vcorr) {
      auto norms_V = utils::striped_norm(
          _mpi->comm, _DeltaVcorr.slice(sDeltaVcorr_skij.local()), _DeltaVcorr.prev,
          stage_iter > 1);
      norm_DeltaSigma = norms_V.first;
      norm_DeltaSigma_diff = norms_V.second;
    } else if (has_Sigma_sc) {
      auto norms_Sigma = utils::striped_norm(
          _mpi->comm, _DeltaSigma.slice(pDeltaSigma_sc->local()), _DeltaSigma.prev,
          stage_iter > 1);
      norm_DeltaSigma = norms_Sigma.first;
      norm_DeltaSigma_diff = norms_Sigma.second;
    }
    _Timer.stop("LR_CONVERGENCE");

    // Has the inner (K_sc only, frozen perturbative source) solve converged?
    // An empty K_sc makes it exact in a single Dyson application, so every
    // stage is then one iteration long.
    bool dm_converged = norm_DeltaDm_diff < p.tol;
    bool f_converged = !k.sc_hf || norm_DeltaF_diff < p.tol;
    bool sigma_converged = !(has_Sigma_sc || has_Vcorr) || norm_DeltaSigma_diff < p.tol;
    bool inner_conv_std = (stage_iter > 1) && dm_converged && f_converged && sigma_converged;
    bool inner_converged = (do_pert && k.sc.empty()) ? true : inner_conv_std;

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
    if (do_pert && inner_converged && n_applied < p.pert_order && !outer_converged) {
      const int outer_step = n_applied + 1;  // 1-based outer iteration index
      double outer_dm_diff = -1.0;
      double outer_res = -1.0;

      // Outer convergence is tested on the stage-to-stage change of ΔDm, and
      // BEFORE the next K_pert evaluation: the source residual would need the
      // very evaluation it proves unnecessary, and one such evaluation is the
      // entire unit of cost here. ΔDm is tiny, so this test is free.
      if (outer_tol > 0.0 && n_applied >= 1) {
        outer_dm_diff = outer_diff_norm(sDeltaDm_skij, *_sDeltaDm_stage_prev);
        if (outer_dm_diff < outer_tol) outer_converged = true;
      }

      if (!outer_converged) {
        if (outer_tol > 0.0) outer_save(*_sDeltaDm_stage_prev, sDeltaDm_skij);

        if (k.pert_hf) {
          _Timer.start("LR_HF_PERT");
          // The counter-term IS an exchange contraction, just with the kernel
          // -W_c(0), so it turns the exchange branch on even when the component
          // mask leaves exchange wholly in K_sc.
          _lr_hf_pert->evaluate(sDeltaF_pert_skij, sDeltaDm_skij, thc, sS_skij.local(),
                                k.pert.hartree, k.pert.exchange || hsex_ct.has_value(),
                                nullptr, nullptr, nullptr, nullptr, false,
                                hsex_ct ? &(*hsex_ct) : nullptr);
          _Timer.stop("LR_HF_PERT");
          _mpi->comm.barrier();
        }
        if (k.pert_sigma) {
          eval_sigma_channel(k.pert, *_lr_gw_pert, *pDeltaSigma_pert, nullptr,
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
          if (k.pert_hf) {
            double d = utils::striped_norm(
                _mpi->comm, _DeltaF_pert.slice(sDeltaF_pert_skij.local()),
                _DeltaF_pert.prev, true).second;
            r2 += d * d;
          }
          if (k.pert_sigma) {
            double d = utils::striped_norm(
                _mpi->comm, _DeltaSigma_pert.slice(pDeltaSigma_pert->local()),
                _DeltaSigma_pert.prev, true).second;
            r2 += d * d;
          }
          outer_res = std::sqrt(r2);

          _Timer.start("LR_OUTER_ITER_ALG");
          // The outer accelerator stripes over the same `_pmap` partition as the
          // inner one: each rank mixes its own element slice of the source and
          // outer_sync completes the node replicas afterwards.
          nda::array<ComplexType, 1> empty_prev;
          if (k.pert_hf && k.pert_sigma) {
            _outer_diis->next_step_combined(
                _mpi->comm, _pmap,
                sDeltaF_pert_skij.local(), _DeltaF_pert.prev,
                pDeltaSigma_pert->local(), _DeltaSigma_pert.prev,
                outer_step);
          } else if (k.pert_sigma) {
            // Σ-only source: next_step_combined is generic in both slots, so
            // the 5D ΔΣ rides the first one with an empty second slot.
            nda::array<ComplexType, 5> empty_slot;
            _outer_diis->next_step_combined(
                _mpi->comm, _pmap,
                pDeltaSigma_pert->local(), _DeltaSigma_pert.prev,
                empty_slot, empty_prev, outer_step);
          } else {
            nda::array<ComplexType, 4> empty_slot;
            _outer_diis->next_step_combined(
                _mpi->comm, _pmap,
                sDeltaF_pert_skij.local(), _DeltaF_pert.prev,
                empty_slot, empty_prev, outer_step);
          }
          outer_sync(k.pert_hf ? &sDeltaF_pert_skij : nullptr,
                     k.pert_sigma ? pDeltaSigma_pert : nullptr);
          // The mixed source is what the next stage uses, hence what the next
          // outer residual is measured against.
          if (k.pert_hf)
            _DeltaF_pert.prev = _DeltaF_pert.slice(sDeltaF_pert_skij.local());
          if (k.pert_sigma)
            _DeltaSigma_pert.prev = _DeltaSigma_pert.slice(pDeltaSigma_pert->local());
          _Timer.stop("LR_OUTER_ITER_ALG");
        }

        ++n_applied;
        pert_refreshed_this_iter = true;
        if (outer_track) {
          // The source residual exists only when the accelerator ran; the ΔDm
          // change only from the second stage of a tolerance-driven run.
          app_log(1, "    [outer {}/{}]{}{}", n_applied, p.pert_order,
                  outer_res >= 0.0
                      ? fmt::format("  ||S_new - S_used|| = {:.6e}", outer_res)
                      : std::string(),
                  outer_dm_diff >= 0.0
                      ? fmt::format("  ||ΔDm - ΔDm_stage_prev|| = {:.6e}", outer_dm_diff)
                      : std::string());
        }
        // The next stage solves a different fixed point: the DIIS history from
        // this one is invalid and its warmup keys off the stage-local iteration
        // index. ΔF_sc/ΔΣ_sc are deliberately kept as the warm start for the
        // next stage.
        if (p.use_diis()) _lr_diis->reset();
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
    const bool outer_done = outer_converged || (n_applied == p.pert_order);
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

  // A Σ-free run never replicated ΔG(τ) inside the loop, so if the caller is
  // going to read the converged one it has to be replicated now: sDeltaG_tskij
  // outlives the solve and is reused by the next perturbation, so leaving it
  // would hand the caller the previous perturbation's ΔG next to this one's
  // ΔDm/ΔF. With Σ the last iteration already replicated it and nothing is
  // pending; with save_DeltaG off nobody reads it and the gather is skipped
  // outright — the only case where ΔG(τ) is never replicated at all.
  if (!k.include_gw_sigma and p.save_DeltaG)
    _lr_dyson.materialize_DeltaG_tau(sDeltaG_tskij);

  // Copy the converged static ΔV_QPGW into the caller's output array (qp mode).
  if (k.qp_mode && sDeltaVcorr_out_skij != nullptr) {
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
  } else if (p.max_iter > 1) {
    if (do_pert)
      app_log(1, "\n  [WARNING] LR SCF did NOT converge after {} iterations "
                 "(K_pert applied {} of {} times).", p.max_iter, n_applied, p.pert_order);
    else
      app_log(1, "\n  [WARNING] LR SCF did NOT converge after {} iterations.", p.max_iter);
  }
  if (do_pert && outer_track) {
    if (outer_converged)
      app_log(1, "  Outer loop converged: K_pert applied {} of at most {} times.",
              n_applied, p.pert_order);
    else if (outer_tol > 0.0)
      app_log(1, "  [WARNING] the outer loop hit its cap: K_pert applied {} of {} "
                 "times without reaching outer_tol = {:.2e}. The result is a "
                 "non-converged accelerated iterate, NOT an order-{} truncation.",
              n_applied, p.pert_order, outer_tol, p.pert_order);
    else
      app_log(1, "  Outer loop ran its full schedule: K_pert applied {} of {} times.",
              n_applied, p.pert_order);
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
  //
  // The move empties the IBC object, so a second lr_solve_one would find
  // F_PQ_skij.size() == 0 and quietly write nothing. That is one of the reasons
  // callers must reject IBC together with more than one perturbation; lifting
  // that restriction means copying here instead.
  if (F_PQ_out && opt_ibc && opt_ibc->F_PQ_skij.size() > 0) {
    *F_PQ_out = std::move(opt_ibc->F_PQ_skij);
  }
  if (DeltaF_PQ_out && k.need_hf) {
    // The harvest is ONE lr_hf call on the union mask, so it cannot represent a
    // kernel the two channels only add up to. HSEX + split is exactly that: the
    // sc channel contracted V + W_c(0) and the pert channel -W_c(0).
    utils::check(!k.pert_sex_counterterm,
                 "lr_driver::lr_solve_one: the aux-basis ΔF_PQ output cannot be "
                 "produced for a split run whose K_sc is HSEX — the total "
                 "exchange kernel is the sum of two channels, and this is a "
                 "single evaluation. Set output_aux_fock = false.");
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
                     k.sc.hartree || k.pert.hartree,
                     k.sc.exchange || k.pert.exchange, ibc_ptr,
                     p.DeltaV_qPQ, p.Dm_ab,
                     DeltaF_PQ_out, p.include_xc, hsex ? &(*hsex) : nullptr);
  }

  // Hierarchical timer report for this perturbation (verbosity >= 2). Per-step
  // solver prints inside the loop are gated to verbosity >= 3.
  print_timers(_lr_pi.get(), _lr_scr.get(), _lr_gw.get(),
               _lr_gw_pert.get(), _lr_gw2.get());

  return std::make_tuple(iter, Delta_mu);
}


void lr_driver::print_memory_estimate(long NP, bool include_gw_sigma, bool gw_full,
                                      std::vector<std::string> const& extra_sigma,
                                      long n_sigma_prev,
                                      lr_diis_hist_t inner_hist,
                                      lr_diis_hist_t outer_hist,
                                      bool need_Delta_mu,
                                      bool exchange_static_W) {
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
  if (include_gw_sigma) {
    arrays.push_back({"sDeltaSigma_tskij",      shp5b(nt), band5(nt), false, PERSIST});
    // Split-kernel per-channel ΔΣ buffers and the outer accelerator's previous
    // source, on top of the total ΔΣ.
    for (auto const& name : extra_sigma)
      arrays.push_back({name, shp5b(nt), band5(nt), false, PERSIST});
  }

  // --- Persistent, striped over the global comm (each rank keeps one element
  //     slice of a node-replicated band array) ---
  // Previous iterates for damping/DIIS, inner and outer alike. ΔF_prev /
  // ΔV_QPGW_prev are ~nb²·nk and negligible next to these; only the ΔΣ-sized
  // ones are worth a row.
  for (long i = 0; i < n_sigma_prev; ++i) {
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

  // --- Persistent, distributed, band basis ---
  // ΔG(τ) as lr_dyson hands it over: distributed, and alive from the end of the
  // solve until the first consumer replicates it. It overlaps neither transient
  // peak in practice — the next solve drops it before allocating anything, and a
  // consumer replicates it at the top of eval_sigma_channel, ahead of any ΔΠ/ΔW
  // — so counting it as persistent is a deliberate over-estimate.
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
  if (include_gw_sigma) {
    arrays.push_back({"dW_tRPQ",       shp4a(nth), aux4(nth), true, PERSIST});
  }
  if (exchange_static_W) {
    arrays.push_back({"dWc0_qPQ (HSEX)", fmt::format("({},{},{})", nq, NP, NP),
                      aux4(1), true, PERSIST});
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
  // Aux q-dist (ω-side) — only the full-GW W Dyson pipeline. One row: the FT
  // staging buffers and W(iω) share a distribution, which is what lets both
  // transforms fuse. The (P, Q) block is reported because it is the SLATE tile the
  // ω-side Dyson runs on.
  if (gw_full) {
    auto [w_pg, w_bs] =
        solvers::lr_scr_coulomb_t::W_omega_dist(nproc, nq, nwbh, NP);
    const char* arrs = is_q_gamma() ? "FT staging buffers, dW_full_wqPQ"
                                    : "FT staging buffers, dW_full_wqPQ, _dW_full_qpQ";
    app_log(2, "    {:<22s}{:<30s}{}", "aux q-dist (ω-side)",
            pg4(w_pg, "(w,q,P,Q)"),
            fmt::format("{}; (P,Q) block {}x{}", arrs, w_bs[2], w_bs[3]));
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
  app_log(2, "      - W_c(iν=0) (HSEX):       {0:8.3f} sec  {1:4d} calls", _Timer.elapsed("LR_DRIVER_SETUP_WC0"), _Timer.number_of_calls("LR_DRIVER_SETUP_WC0"));
  app_log(2, "      - G(iω) precompute:       {0:8.3f} sec  {1:4d} calls", _Timer.elapsed("LR_DRIVER_SETUP_G_OMEGA"), _Timer.number_of_calls("LR_DRIVER_SETUP_G_OMEGA"));
  app_log(2, "      - G^R pair precompute:    {0:8.3f} sec  {1:4d} calls", _Timer.elapsed("LR_DRIVER_SETUP_G_R"), _Timer.number_of_calls("LR_DRIVER_SETUP_G_R"));
  app_log(2, "      - Δμ response precompute: {0:8.3f} sec  {1:4d} calls", _Timer.elapsed("LR_DRIVER_SETUP_DELTAMU"), _Timer.number_of_calls("LR_DRIVER_SETUP_DELTAMU"));
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
  // The setup is a sibling of the SCF loop, not a part of it: its clock is
  // never reset, so it reads the same once-paid cost in every perturbation's
  // report, while "Total LR SCF" and its subclocks below cover this
  // perturbation only.
  app_log(2, "    LR Driver Setup (once):     {0:8.3f} sec  {1:4d} calls", _Timer.elapsed("LR_DRIVER_SETUP"), _Timer.number_of_calls("LR_DRIVER_SETUP"));
  app_log(2, "    Total LR SCF:               {0:8.3f} sec  {1:4d} calls", _Timer.elapsed("LR_SCF"), _Timer.number_of_calls("LR_SCF"));
  app_log(2, "      - LR Dyson (total):       {0:8.3f} sec  {1:4d} calls", _Timer.elapsed("LR_DYSON"), _Timer.number_of_calls("LR_DYSON"));
  _lr_dyson.print_subclocks(2, sub_indent);
  app_log(2, "      - LR HF (total):          {0:8.3f} sec  {1:4d} calls", sec("LR_HF", "LR_HF_PERT"), cnt("LR_HF", "LR_HF_PERT"));
  if (_lr_hf or _lr_hf_pert)
    solvers::lr_hf::print_subclocks_all(2, sub_indent, {_lr_hf.get(), _lr_hf_pert.get()});
  app_log(2, "      - LR GW Pi (total):       {0:8.3f} sec  {1:4d} calls", sec("LR_GW_PI", "LR_GW_PI_PERT"), cnt("LR_GW_PI", "LR_GW_PI_PERT"));
  if (pi_solver) pi_solver->print_subclocks(2, sub_indent);
  app_log(2, "      - LR GW W (total):        {0:8.3f} sec  {1:4d} calls", sec("LR_GW_W", "LR_GW_W_PERT"), cnt("LR_GW_W", "LR_GW_W_PERT"));
  if (scr_solver) scr_solver->print_subclocks(2, sub_indent);
  app_log(2, "      - LR GW Sigma (total):    {0:8.3f} sec  {1:4d} calls", sec("LR_GW_SIGMA", "LR_GW_SIGMA_PERT"), cnt("LR_GW_SIGMA", "LR_GW_SIGMA_PERT"));
  // Summed over all three Σ evaluators, and gated on any of them existing rather
  // than on the self-consistent one: a kernel whose Σ sits entirely in the
  // perturbative channel (pGW0 over a Hartree K_sc) never builds gw_solver.
  if (gw_solver or gw_solver_pert or gw_solver_term2)
    solvers::lr_gw::print_subclocks_all(2, sub_indent,
                                        {gw_solver, gw_solver_pert, gw_solver_term2});
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

template void lr_driver::lr_setup(
    const sArray_t<Array_view_5D_t>&,
    thc_reader_t&,
    dW_concrete_t*,
    const lr_params&);

template std::tuple<int, double> lr_driver::lr_solve_one(
    sArray_t<Array_view_5D_t>&,
    sArray_t<Array_view_4D_t>&,
    sArray_t<Array_view_4D_t>&,
    sArray_t<Array_view_5D_t>*,
    const sArray_t<Array_view_5D_t>&,
    const sArray_t<Array_view_4D_t>&,
    thc_reader_t&,
    const lr_params&,
    sArray_t<Array_view_5D_t>*,
    sArray_t<Array_view_4D_t>*,
    nda::array<ComplexType, 4>*,
    nda::array<ComplexType, 4>*,
    nda::array<ComplexType, 4>*,
    int*);


} // namespace methods
