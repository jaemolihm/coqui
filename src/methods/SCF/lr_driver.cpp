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
//   W_c in:              (w,q,P,Q), ω-side (solvers::lr_scr_coulomb_t::W_omega_dist)
//   lr_setup_W:
//     w_to_tau:          ω-side → (t,q,P,Q) straight onto the τ-dist tiling
//     lr_Wc_to_Wfull:    + Z(q) in place on the ω copy → W_full(iω) [cached]
//     lr_precompute_W_tRPQ: q→R in place on the τ copy → (t,R,P,Q), τ-dist [cached]
//
//   evaluate_lr_Pi:      → (t,q,P,Q), τ-dist
//   solve_lr_dyson_W (in-place):
//     tau_to_w:          τ-dist → ω-side (via q-distributed FT buffer)
//     lr_dyson_W_in_place: ω-side; SLATE GEMM batched over (iw, iq).
//                          For Q≠Γ, gathers W_full(kpq_map(iq)) via Alltoallv on q_pool_comm.
//     w_to_tau:          ω-side → τ-dist (via q-distributed FT buffer)
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
std::tuple<int, double> lr_driver::run_lr(
    sArray_t<Array_view_5D_t>& sDeltaG_tskij,
    sArray_t<Array_view_4D_t>& sDeltaDm_skij,
    sArray_t<Array_view_4D_t>& sDeltaF_skij,
    sArray_t<Array_view_5D_t>* sDeltaSigma_tskij,
    const sArray_t<Array_view_5D_t>& sG_tskij,
    const sArray_t<Array_view_4D_t>& sDeltaH0_skij,
    THC_t& thc,
    bool include_hartree, bool include_exchange, lr_gw_update_mode gw_mode,
    dW_t* dW_wqPQ_in,
    const nda::array<ComplexType, 1>* eps_inv_head,
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
    bool include_xc) {

  _Timer.start("LR_SCF");

  bool need_hf = include_hartree || include_exchange;
  bool include_gw_sigma = (gw_mode != lr_gw_update_mode::none);
  bool gw_full = (gw_mode == lr_gw_update_mode::full);
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
  bool has_Sigma = include_gw_sigma && !qp_mode;
  // One-shot G0W0: store the two ΔΣ terms separately instead of fusing them.
  // ΔΣ term1 (-ΔG⊙W_c) → sDeltaSigma_tskij, ΔΣ term2 (-G⊙ΔW) → sDeltaSigma_term2.
  bool split_sigma_terms = (sDeltaSigma_term2_tskij != nullptr);

  utils::check(iter_params.alg == "damping" || iter_params.alg == "DIIS",
               "lr_driver::run_lr: unknown iter_alg '{}'. Must be 'damping' or 'DIIS'.",
               iter_params.alg);
  if (include_gw_sigma) {
    utils::check(sDeltaSigma_tskij != nullptr,
                 "lr_driver::run_lr: gw_mode != none but sDeltaSigma_tskij is null.");
    utils::check(dW_wqPQ_in != nullptr && eps_inv_head != nullptr,
                 "lr_driver::run_lr: gw_mode != none but dW or eps_inv_head is null.");
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
  if (qp_mode) {
    utils::check(include_gw_sigma,
                 "lr_driver::run_lr: qp_static mode requires gw_mode != none.");
    utils::check(!split_sigma_terms,
                 "lr_driver::run_lr: qp_static mode is incompatible with split ΔΣ terms.");
    utils::check(qp_static->sMO_skia != nullptr && qp_static->sE_ska != nullptr,
                 "lr_driver::run_lr: qp_static mode requires sMO_skia and sE_ska.");
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
  app_log(1, "  include_xc = {}", include_xc ? "true" : "false");
  app_log(1, "  gw_mode = {}", gw_mode_str);
  app_log(1, "  qp_static_sigma = {}", qp_mode ? "true" : "false");
  app_log(1, "  iter_alg = {}", iter_params.alg);
  app_log(1, "  mixing = {:.2f}", mixing);
  if (use_diis) {
    app_log(1, "  max_subsp_size = {}", iter_params.max_subsp_size);
    app_log(1, "  diis_warmup = {}", iter_params.diis_warmup);
  }

  // Estimate the persistent large-array memory footprint for this path, then
  // summarize the MPI distribution patterns the large arrays use.
  print_memory_estimate(thc.Np(), include_gw_sigma, gw_full,
                        use_diis, iter_params.max_subsp_size);
  print_distribution_summary(thc.Np(), include_gw_sigma, gw_full);

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
  _Timer.start("LR_DRIVER_SETUP");
  std::optional<dW_t> opt_dW_full_wqPQ;   // W_full(iω), ω-side distribution
  std::optional<dW_t> opt_dW_tRPQ;        // W_c(t,R,P,Q), τ-dist (q-local)
  if (include_gw_sigma) {
    lr_setup_W(dW_wqPQ_in, thc, gw_full, lr_scr_solver.get(),
               opt_dW_full_wqPQ, opt_dW_tRPQ);
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
  // rank-private slices of the flattened arrays, in the `pmap` partition.
  // Previous ΔΣ only needed when GW is active and not in qp mode (in qp mode the
  // dynamic ΔΣ is not tracked; the static ΔV_QPGW is tracked instead).
  const long nF_flat = _ns * _nkpts_ibz * _nbnd * _nbnd;
  const long nS_flat = has_Sigma ? _nts * nF_flat : 0;
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
        include_hartree, include_exchange, include_gw_sigma,
        F_PQ_out != nullptr));

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

    // Save previous density matrix and Fock matrix. ΔDm is node-replicated, so
    // node root copies it whole; the mixed quantities are saved as this rank's
    // partition slice only, in parallel across the node.
    _Timer.start("LR_SAVE");
    if (iter > 1) {
      if (_mpi->node_comm.root())
        sDeltaDm_prev_skij.local() = sDeltaDm_skij.local();
      DeltaF_prev = flat_slice(sDeltaF_skij.local(), nF_flat, iF0, iF1);
      if (has_Vcorr) {
        DeltaVcorr_prev = flat_slice(sDeltaVcorr_skij.local(), nV_flat, iV0, iV1);
      } else if (has_Sigma) {
        DeltaSigma_prev = flat_slice(sDeltaSigma_tskij->local(), nS_flat, iS0, iS1);
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
                       DeltaV_qPQ, Dm_ab, nullptr, include_xc);
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

      // Step 3e-3f: ΔΣ = -ΔG ⊙ W_c - G ⊙ ΔW.
      // ΔW stays in (t,q,P,Q): the Σ evaluator consumes one τ slice at a time,
      // which is contiguous in this layout and matches term 1's dW_tRPQ.
      _Timer.start("LR_GW_SIGMA");
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
            *sDeltaSigma_term2_tskij, sG_tskij.local(), dDeltaW_tqPQ, thc,
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
            sG_tskij.local(), dDeltaW_tqPQ, thc,
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

    // Step 4: Apply iteration algorithm (DIIS or damping) on combined (ΔF, ΔΣ)
    _Timer.start("LR_ITER_ALG");
    if (iter > 1 && (need_hf || include_gw_sigma)) {
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
              sDeltaF_skij.local(), DeltaF_prev,
              sDeltaVcorr_skij.local(), DeltaVcorr_prev, iter);
        } else if (has_Sigma) {
          _lr_diis->next_step_combined(
              _mpi->comm, pmap,
              sDeltaF_skij.local(), DeltaF_prev,
              sDeltaSigma_tskij->local(), DeltaSigma_prev, iter);
        } else {
          nda::array<ComplexType, 5> empty_sigma;
          nda::array<ComplexType, 1> empty_prev;
          _lr_diis->next_step_combined(
              _mpi->comm, pmap,
              sDeltaF_skij.local(), DeltaF_prev,
              empty_sigma, empty_prev, iter);
        }
      } else if (mixing < 1.0) {
        // Damping is elementwise too, so stripe it over the same partition —
        // one completion path then covers both algorithms.
        auto F_loc = flat_slice(sDeltaF_skij.local(), nF_flat, iF0, iF1);
        F_loc = mixing * F_loc + (1.0 - mixing) * DeltaF_prev;
        if (has_Vcorr) {
          auto V_loc = flat_slice(sDeltaVcorr_skij.local(), nV_flat, iV0, iV1);
          V_loc = mixing * V_loc + (1.0 - mixing) * DeltaVcorr_prev;
        } else if (has_Sigma) {
          auto S_loc = flat_slice(sDeltaSigma_tskij->local(), nS_flat, iS0, iS1);
          S_loc = mixing * S_loc + (1.0 - mixing) * DeltaSigma_prev;
        }
      }
      // The mixing above writes each rank's slice of the shared ΔF/ΔΣ buffer in
      // place with no trailing collective. Fence + barrier make every slice
      // visible to node root before it gathers below (barrier alone is
      // insufficient under the MPI-3 separate shared-memory model).
      sDeltaF_skij.win().fence();
      if (has_Vcorr) sDeltaVcorr_skij.win().fence();
      else if (has_Sigma) sDeltaSigma_tskij->win().fence();
      _mpi->node_comm.barrier();
      // Each node now holds a valid copy of its own contiguous element run
      // only; one allgatherv among the node roots completes every replica. With
      // the striping global, each element is mixed exactly once in the whole
      // job, so the per-node drift the old node-0 broadcast papered over cannot
      // arise by construction.
      if (_mpi->node_comm.root()) {
        utils::complete_node_slices(_mpi->internode_comm, pmap,
                                       sDeltaF_skij.local().data(), nF_flat);
        if (has_Vcorr) {
          utils::complete_node_slices(_mpi->internode_comm, pmap,
                                         sDeltaVcorr_skij.local().data(), nV_flat);
        } else if (has_Sigma) {
          utils::complete_node_slices(_mpi->internode_comm, pmap,
                                         sDeltaSigma_tskij->local().data(), nS_flat);
        }
      }
      // Make node root's overwrite visible to its node peers
      sDeltaF_skij.win().fence();
      if (has_Vcorr) sDeltaVcorr_skij.win().fence();
      else if (has_Sigma) sDeltaSigma_tskij->win().fence();
      _mpi->comm.barrier();
    }
    _Timer.stop("LR_ITER_ALG");

    // Compute norms of ΔF and ΔF-ΔF_prev for logging. The previous iterates are
    // stored striped, so the norms are reduced over the global comm from the
    // same slices.
    _Timer.start("LR_CONVERGENCE");
    auto norms_F = utils::striped_norm(
        _mpi->comm, flat_slice(sDeltaF_skij.local(), nF_flat, iF0, iF1),
        DeltaF_prev, iter > 1);
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
          DeltaVcorr_prev, iter > 1);
      norm_DeltaSigma = norms_V.first;
      norm_DeltaSigma_diff = norms_V.second;
      _mpi->comm.broadcast_n(&norm_DeltaSigma, 1, 0);
      _mpi->comm.broadcast_n(&norm_DeltaSigma_diff, 1, 0);
    } else if (has_Sigma) {
      auto norms_Sigma = utils::striped_norm(
          _mpi->comm, flat_slice(sDeltaSigma_tskij->local(), nS_flat, iS0, iS1),
          DeltaSigma_prev, iter > 1);
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
      bool sigma_converged = !(has_Sigma || has_Vcorr) || norm_DeltaSigma_diff < tol;
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

  // Copy the converged static ΔV_QPGW into the caller's output array (qp mode).
  if (qp_mode && sDeltaVcorr_out_skij != nullptr) {
    sDeltaVcorr_out_skij->win().fence();
    if (_mpi->node_comm.root())
      sDeltaVcorr_out_skij->local() = sDeltaVcorr_skij.local();
    sDeltaVcorr_out_skij->win().fence();
    _mpi->comm.barrier();
  }

  // Report results
  if (converged) {
    app_log(1, "\n  LR SCF converged in {} iterations!", iter);
  } else if (max_iter > 1) {
    app_log(1, "\n  [WARNING] LR SCF did NOT converge after {} iterations.", max_iter);
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
    _lr_hf->evaluate(sDeltaF_scratch, sDeltaDm_skij, thc, sS_skij.local(),
                     include_hartree, include_exchange, ibc_ptr,
                     DeltaV_qPQ, Dm_ab,
                     DeltaF_PQ_out, include_xc);
  }

  // Final hierarchical timer report (printed once, at verbosity >= 2).
  // Per-step solver prints inside the loop are gated to verbosity >= 3.
  print_timers(lr_pi_solver.get(), lr_scr_solver.get(), lr_gw_solver.get());

  return std::make_tuple(iter, Delta_mu);
}


void lr_driver::print_memory_estimate(long NP, bool include_gw_sigma, bool gw_full,
                                      bool use_diis, size_t max_subsp_size) {
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
  }

  // --- Persistent, striped over the global comm (each rank keeps one element
  //     slice of a node-replicated band array) ---
  // Previous iterates for damping/DIIS. ΔF_prev / ΔV_QPGW_prev are ~nb²·nk and
  // negligible next to these; only the ΔΣ history is worth a row.
  if (include_gw_sigma) {
    arrays.push_back({"ΔΣ_prev (striped)", shp5b(nt), band5(nt), true, PERSIST});
  }
  if (use_diis) {
    // DIIS keeps `max_subsp_size` trial + residual vector pairs for each mixed
    // quantity, each striped over the global comm.
    const double per_vec = band5(1) * (include_gw_sigma ? (1.0 + nt) : 1.0);
    arrays.push_back({fmt::format("DIIS history (subsp {})", max_subsp_size),
                      fmt::format("2 x {} x |ΔF|+|ΔΣ|", max_subsp_size),
                      2.0 * double(max_subsp_size) * per_vec, true, PERSIST});
  }

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
  // distribution choice in solvers::lr_scr_coulomb_t::W_omega_dist).
  if (gw_full) {
    auto [ftb_pg, ftb_bs] =
        solvers::scr_coulomb_fourier_t::ft_buffer_dist(nproc, {nth, nq, NP, NP});
    (void)ftb_bs;
    auto [w_pg, w_bs] =
        solvers::lr_scr_coulomb_t::W_omega_dist(nproc, nq, nwbh, NP);
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
  if (gw_solver) gw_solver->print_subclocks(2, sub_indent);
  app_log(2, "      - LR GW Sig T2 (total):   {0:8.3f} sec  {1:4d} calls", _Timer.elapsed("LR_GW_SIGMA_TERM2"), _Timer.number_of_calls("LR_GW_SIGMA_TERM2"));
  app_log(2, "      - LR qpGW static (total): {0:8.3f} sec  {1:4d} calls", _Timer.elapsed("LR_QPGW_STATIC"), _Timer.number_of_calls("LR_QPGW_STATIC"));
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
    sArray_t<Array_view_5D_t>*,
    const lr_qp_static_params*,
    sArray_t<Array_view_4D_t>*,
    nda::array<ComplexType, 4>*,
    nda::array<ComplexType, 4>*,
    nda::array<ComplexType, 4>*,
    bool);

} // namespace methods
