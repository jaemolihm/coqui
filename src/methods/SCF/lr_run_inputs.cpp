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

#include "methods/SCF/lr_run_inputs.hpp"
#include "methods/ERI/thc_reader_t.hpp"
#include "methods/scr_coulomb/scr_coulomb_t.h"

// The inputs run_lr_calc and run_lr_efield_calc share, declared in lr_run_inputs.hpp: the unperturbed state, the
// screened interaction with its process-wide cache, and the lr_params fields taken from them.

namespace methods {

namespace {

/// W_c(iω) and the ε⁻¹ heads kept by a reuse_W call. No member calls MPI in its destructor, so destruction after
/// MPI_Finalize is safe.
struct lr_W_cache {
  lr_W_t W_c_wqPQ;
  nda::array<ComplexType, 1> eps_inv_head, eps_inv_head_w;
};

/// The W kept by the last reuse_W call that built one (one instance per process); release_lr_W_cache() drops it.
std::optional<lr_W_cache> W_cache;

} // namespace

void release_lr_W_cache() { W_cache.reset(); }

lr_iter_params read_lr_iter_params(ptree const& pt, bool qp_static_sigma) {
  lr_iter_params iter_params;
  // 1. alg: GCR by default, DIIS for the qpGW map, whose Padé statification is not affine
  //    (lr_driver::lr_setup rejects GCR there)
  iter_params.alg = io::get_value_with_default<std::string>(
      pt, "iter_alg.alg", qp_static_sigma ? std::string("DIIS") : std::string("GCR"));
  utils::check(iter_params.alg == "damping" || iter_params.alg == "DIIS" ||
               iter_params.alg == "GCR",
               "read_lr_iter_params: unknown iter_alg '{}'. Must be 'damping', 'DIIS' or 'GCR'.",
               iter_params.alg);
  // 2. mixing, subspace size and DIIS warmup
  iter_params.mixing = io::get_value_with_default<double>(pt, "iter_alg.mixing", 1.0);
  // GCR: 0 lets lr_driver::lr_setup pick the capacity from the kernel.
  iter_params.max_subsp_size = static_cast<size_t>(io::get_value_with_default<long>(
      pt, "iter_alg.max_subsp_size", iter_params.alg == "GCR" ? 0 : 10));
  iter_params.diis_warmup = static_cast<size_t>(
      io::get_value_with_default<long>(pt, "iter_alg.diis_warmup", 0));
  return iter_params;
}

lr_unperturbed_state load_lr_unperturbed_state(utils::mpi_context_t<mpi3::communicator>& mpi, mf::MF& mf, imag_axes_ft::IAFT& ft,
                                               std::string const& h0_source, std::string const& prefix,
                                               std::string const& input_grp, long input_iter,
                                               std::string const& unperturbed, utils::TimerManager& init_timer) {
  // 1. Dyson solver with the requested H0 source
  utils::check(h0_source == "compute" || h0_source == "checkpoint",
               "load_lr_unperturbed_state: h0_source must be 'compute' or 'checkpoint', got '{}'", h0_source);
  init_timer.start("LR_INIT_DYSON");
  simple_dyson dyson = make_dyson_with_h0_source(h0_source, &mf, &ft, prefix);
  init_timer.stop("LR_INIT_DYSON");

  utils::memlog("load_lr_unperturbed_state: after simple_dyson ctor");

  // 2. Shared arrays of the unperturbed state
  auto sF_skij = math::shm::make_shared_array<Array_view_4D_t>(
      mpi, {mf.nspin(), mf.nkpts_ibz(), mf.nbnd(), mf.nbnd()});
  auto sDm_skij = math::shm::make_shared_array<Array_view_4D_t>(
      mpi, {mf.nspin(), mf.nkpts_ibz(), mf.nbnd(), mf.nbnd()});
  std::optional<math::shm::shared_array<Array_view_5D_t>> opt_sG_tskij;
  opt_sG_tskij.emplace(math::shm::make_shared_array<Array_view_5D_t>(
      mpi, {ft.nt_f(), mf.nspin(), mf.nkpts_ibz(), mf.nbnd(), mf.nbnd()}));
  auto& sG_tskij = *opt_sG_tskij;
  auto sSigma_tskij = math::shm::make_shared_array<Array_view_5D_t>(
      mpi, {ft.nt_f(), mf.nspin(), mf.nkpts_ibz(), mf.nbnd(), mf.nbnd()});

  utils::memlog("load_lr_unperturbed_state: after sF/sDm/sG/sSigma alloc");

  auto sG_wskij = math::shm::make_shared_array<Array_view_5D_t>(
      mpi, {ft.nw_f(), mf.nspin(), mf.nkpts_ibz(), mf.nbnd(), mf.nbnd()});
  utils::memlog("load_lr_unperturbed_state: after sG_wskij alloc");

  // 3. G(τ), Dm, G(iω)
  double mu = 0.0;
  if (unperturbed == "mf_dft") {
    // DFT/KS mean-field reference G0 for one-shot G0W0@DFT LR. Build the MO
    // coefficients / energies from the mean field, set μ to conserve N, and
    // build G0(τ) directly (Σ=0). sF_skij and sSigma_tskij are left unused in
    // this branch. See qp_scf_common::write_mf_data for the same recipe.
    app_log(2, "Building DFT/KS mean-field reference G0 (G0W0@DFT)...");
    init_timer.start("LR_INIT_UPDATE_G");
    auto [sMO_skia, sE_ska] = get_mf_MOs(mpi, mf);
    mu = update_mu(0.0, mf, sE_ska, ft.beta());
    app_log(2, "  DFT reference mu = {:.8f}", mu);
    update_G(sG_tskij, sMO_skia, sE_ska, mu, ft);
    update_Dm(sDm_skij, sMO_skia, sE_ska, mu, ft.beta());
    update_G_omega(sG_wskij, sMO_skia, sE_ska, mu, ft);
    mpi.comm.barrier();
    init_timer.stop("LR_INIT_UPDATE_G");
  } else {
    // Read F and Sigma from checkpoint
    init_timer.start("LR_INIT_READ_SCF");
    chkpt::read_scf(mpi.node_comm, sF_skij, sSigma_tskij, mu,
                    prefix, input_grp, input_iter);
    mpi.comm.barrier();
    init_timer.stop("LR_INIT_READ_SCF");

    // Compute G from F and Sigma via Dyson equation
    app_log(2, "Computing unperturbed Green's function from checkpoint...");
    app_log(2, "  h0_source = {}", h0_source);
    app_log(2, "  mu = {:.8f}", mu);
    init_timer.start("LR_INIT_UPDATE_G");
    update_G(dyson, mf, ft, sDm_skij, sG_tskij, sF_skij, sSigma_tskij, mu, true, &sG_wskij);
    mpi.comm.barrier();
    init_timer.stop("LR_INIT_UPDATE_G");
  }

  return lr_unperturbed_state{std::move(dyson), std::move(sF_skij), std::move(sDm_skij), std::move(opt_sG_tskij),
                              std::move(sSigma_tskij), std::move(sG_wskij), mu};
}

lr_W_inputs load_lr_W(utils::mpi_context_t<mpi3::communicator>& mpi, thc_reader_t& thc, imag_axes_ft::IAFT const& ft,
                      std::string const& input_file, std::string const& input_grp, ptree const& pt,
                      bool needs_W, bool recompute_W, bool reuse_W,
                      std::string const& screened_interaction_file, std::string const& unperturbed,
                      sArray_t<Array_view_5D_t>& sG_tskij, MBState* mb_local, utils::TimerManager& init_timer) {
  auto mf = thc.MF();
  lr_W_inputs W;
  // An LR run must reproduce the divergence treatment of the ground state it
  // linearizes: the q→0 head of Σ_GW is the exchange Madelung term screened by
  // ε⁻¹, so the correlation and exchange corrections only cancel (exactly, in a
  // metal) when both are built the way the unperturbed run built them. Hence
  // neither is a free parameter here — both come from the checkpoint. The one
  // exception is unperturbed='mf_dft', where W0 is built fresh from the DFT G0
  // and the mean-field checkpoint carries no ground-state value to inherit.
  utils::check(unperturbed == "mf_dft" or !io::check_child_exists(pt, "div_treatment"),
               "load_lr_W: 'div_treatment' is not accepted for unperturbed='{}'. "
               "The LR run always reuses the ground-state value read from "
               "'{}/div_treatment' in {}, so that ΔΣ matches how Σ was built. "
               "Remove the argument (it is only meaningful for unperturbed='mf_dft').",
               unperturbed, input_grp, input_file);
  // Exchange side: needed whenever ΔF carries exchange, independently of W.
  // Absent on pre-stash checkpoints, in which case this falls back to "gygi".
  W.hf_div_treatment = read_hf_div_treatment(mpi, input_file, input_grp);
  // HSEX needs W too, for the single ν=0 slice its exchange kernel adds to V.
  // The ω array is released inside lr_driver::lr_setup once that slice is taken.
  if (needs_W) {
    init_timer.start("LR_INIT_LOAD_W");
    if (unperturbed == "mf_dft") {
      // G0W0@DFT: W0 is the RPA screened interaction built from the DFT G0.
      // The checkpoint holds no scGW W (or div_treatment), so recompute and
      // take div_treatment from params (default matches the C++ default).
      utils::check(screened_interaction_file.empty(),
                   "load_lr_W: unperturbed='mf_dft' builds W0 from G0 (RPA); "
                   "screened_interaction_file must not be set.");
      recompute_W = true;
      W.div_treatment = io::get_value_with_default<std::string>(pt, "div_treatment", "gygi");
    } else {
      W.div_treatment = read_div_treatment(mpi, input_file, input_grp);
    }
    // reuse_W: the caller guarantees that the kept W belongs to this call's ground state and THC; only the
    // THC size is checked. W is reused only if every rank keeps one (a release on some ranks rebuilds on all).
    bool reuse_W_done = false;
    if (reuse_W) {
      int kept = W_cache.has_value();
      mpi.comm.all_reduce_in_place_n(&kept, 1, std::plus<>{});
      reuse_W_done = (kept == mpi.comm.size());  // reuse is done only if the cache is present in all ranks
      if (reuse_W_done) {
        auto const& Wshape = W_cache->W_c_wqPQ.global_shape();
        long nw_half = (ft.nw_b() + 1) / 2;
        utils::check(Wshape[0] == nw_half && Wshape[1] == mf->nkpts() && Wshape[2] == thc.Np() && Wshape[3] == thc.Np(),
                     "load_lr_W: reuse_W keeps W_c(iw) of shape ({}, {}, {}, {}), but this call needs "
                     "({}, {}, {}, {}). reuse_W requires the ground state and THC the kept W was built with; "
                     "call coqui.mbpt.release_lr_W_cache() before switching.",
                     Wshape[0], Wshape[1], Wshape[2], Wshape[3], nw_half, mf->nkpts(), thc.Np(), thc.Np());
        app_log(1, "reuse_W: W_c(iw) and eps^-1 head reused");
        W.opt_dW.emplace(W_cache->W_c_wqPQ);
        W.opt_eps_inv.emplace(W_cache->eps_inv_head);
        W.opt_eps_inv_w.emplace(W_cache->eps_inv_head_w);
      } else {
        app_log(1, "reuse_W: building W (nothing kept)");
        W_cache.reset();
      }
    }
    if (!reuse_W_done) {
      // W is not reused from memory, either recompute or read from file
      if (recompute_W) {
        // Recompute W using Green's functions
        app_log(2, "Recomputing W from checkpoint Green's function ({})...",
                mb_local != nullptr ? "GW+EDMFT: RPA + upfolded local polarizabilities" : "RPA");
        auto [dW, eps_inv, eps_inv_w] = recompute_W_and_eps_inv_head(
            mpi, thc, ft, sG_tskij, W.div_treatment, mb_local);
        W.opt_dW.emplace(std::move(dW));
        W.opt_eps_inv.emplace(std::move(eps_inv));
        W.opt_eps_inv_w.emplace(std::move(eps_inv_w));
      } else {
        // Read W from disk
        auto [dW, eps_inv, eps_inv_w, div_str] = lr_load_W_omega(
            mpi, thc, ft, input_file, input_grp, screened_interaction_file);
        W.opt_dW.emplace(std::move(dW));
        W.opt_eps_inv.emplace(std::move(eps_inv));
        W.opt_eps_inv_w.emplace(std::move(eps_inv_w));
        // W carries its own stash; a disagreement means the W file and the
        // checkpoint came from different runs, and Σ would mix two treatments.
        // (mf_dft always recomputes W, so this branch reads the checkpoint value.)
        utils::check(div_str == W.div_treatment,
                     "load_lr_W: div_treatment mismatch. The screened interaction "
                     "reports '{}' while '{}/div_treatment' in {} reports '{}'. The two "
                     "must come from the same ground-state run.",
                     div_str, input_grp, input_file, W.div_treatment);
        W.div_treatment = div_str;
      }
    }
    if (reuse_W && !reuse_W_done) {
      W_cache.emplace(lr_W_cache{*W.opt_dW, *W.opt_eps_inv, *W.opt_eps_inv_w});
    }
    init_timer.stop("LR_INIT_LOAD_W");
  }
  return W;
}

void fill_lr_kernel_params(lr_params& p, lr_kernel_spec const& kernel, lr_W_inputs const& W,
                           lr_iter_params const& iter_params, long max_iter, double tol, bool fix_density,
                           bool save_DeltaG) {
  p.kernel           = kernel;
  p.max_iter         = max_iter;
  p.tol              = tol;
  p.fix_density      = fix_density;
  // dump_lr writes sDeltaG_tskij only under save_DeltaG.
  p.save_DeltaG      = save_DeltaG;
  p.iter_params      = iter_params;
  p.eps_inv_head     = W.opt_eps_inv ? &(*W.opt_eps_inv) : nullptr;
  p.eps_inv_head_w   = W.opt_eps_inv_w ? &(*W.opt_eps_inv_w) : nullptr;
  p.div_treatment    = W.div_treatment;
  p.hf_div_treatment = W.hf_div_treatment;
}

} // namespace methods
