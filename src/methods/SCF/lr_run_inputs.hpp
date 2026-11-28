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

#ifndef COQUI_LR_RUN_INPUTS_HPP
#define COQUI_LR_RUN_INPUTS_HPP

#include <optional>
#include <string>
#include <filesystem>
#include <tuple>

#include "configuration.hpp"
#include "mpi3/communicator.hpp"

#include "IO/app_loggers.h"
#include "IO/ptree/ptree_utilities.hpp"
#include "mean_field/MF.hpp"
#include "mean_field/mf_utils.hpp"
#include "methods/ERI/thc_reader_t.hpp"
#include "methods/mb_state/mb_state.hpp"
#include "methods/SCF/lr_driver.hpp"
#include "methods/SCF/scf_common.hpp"
#include "methods/SCF/simple_dyson.h"
#include "methods/GW/g0_div_utils.hpp"
#include "methods/scr_coulomb/lr_scr_coulomb_t.hpp"
#include "methods/scr_coulomb/scr_coulomb_t.h"
#include "methods/tools/chkpt_utils.h"
#include "numerics/shared_array/shared_array_io.hpp"
#include "numerics/imag_axes_ft/IAFT.hpp"
#include "utilities/lr_utils.hpp"
#include "utilities/Timer.hpp"
#include "utilities/proc_meminfo.hpp"

/**
 * Setup shared by the LR drivers run_lr_calc and run_lr_efield_calc: the unperturbed state, the screened interaction
 * W_c(iω) (read, recomputed or reused from the process-wide W cache), the iter_alg keys and the kernel/W fields of
 * lr_params, together with the checkpoint and W readers they call.
 */
namespace methods {

/**
 * Factory function for creating simple_dyson with conditional H0 source.
 *
 * @param h0_source     - [INPUT] Source of H0: "compute" or "checkpoint"
 * @param mf_ptr        - [INPUT] Mean-field object pointer
 * @param ft_ptr        - [INPUT] IAFT pointer
 * @param chkpt_prefix  - [INPUT] Checkpoint file prefix (used if h0_source="checkpoint")
 * @param mu_tol        - [INPUT] Chemical potential tolerance (default 1e-9)
 * @param mu_update_alg - [INPUT] Chemical potential update algorithm
 * @return simple_dyson object
 */
inline simple_dyson make_dyson_with_h0_source(
    std::string_view h0_source,
    mf::MF* mf_ptr,
    imag_axes_ft::IAFT* ft_ptr,
    const std::string& chkpt_prefix,
    double mu_tol = 1e-9,
    std::string mu_update_alg = "bisection") {
  // Both sources are valid for an augmented basis: "compute" rebuilds H0 from the
  // orbitals, while "checkpoint" reads the stored H0_skij (a genuine computed H0,
  // not the placeholder eigenvalues), so the two agree.
  if (h0_source == "checkpoint") {
    app_log(1, "  H0 source                = checkpoint ({}.mbpt.h5)", chkpt_prefix);
    return simple_dyson(mf_ptr, ft_ptr, chkpt_prefix, mu_tol, mu_update_alg);
  } else {
    app_log(1, "  H0 source                = computed from orbitals");
    return simple_dyson(mf_ptr, ft_ptr, mu_tol, mu_update_alg);
  }
}


/**
 * Read the string dataset `name` of group `input_grp` in `input_file` on the
 * root and broadcast it. std::nullopt when the group or the dataset is absent.
 */
template<typename mpi_context_t>
std::optional<std::string> read_scf_string(mpi_context_t& mpi,
                                           const std::string& input_file,
                                           const std::string& input_grp,
                                           const std::string& name)
{
  char buf[64] = {0};
  int found = 0;
  if (mpi.comm.root()) {
    h5::file file(input_file, 'r');
    auto root_grp = h5::group(file);
    if (root_grp.has_subgroup(input_grp)) {
      auto scf_grp = root_grp.open_group(input_grp);
      if (scf_grp.has_dataset(name)) {
        std::string str;
        h5::h5_read(scf_grp, name, str);
        utils::check(str.size() < sizeof(buf), "read_scf_string: {} string too long: {}",
                     name, str);
        std::copy(str.begin(), str.end(), buf);
        found = 1;
      }
    }
  }
  mpi.comm.broadcast_n(&found, 1, 0);
  if (!found) return std::nullopt;
  mpi.comm.broadcast_n(buf, sizeof(buf), 0);
  return std::string(buf);
}

/**
 * Read the div_treatment the scGW run used (stashed in the checkpoint scf
 * group by scr_coulomb_t::dump_eps_inv_head); checkpoints that predate this
 * dataset must be regenerated. LR/GW evaluators must reuse this value so
 * divergence heads stay consistent with how W and Sigma were computed.
 * A group without the stash (e.g. the embed group) takes the scf group's.
 */
template<typename mpi_context_t>
std::string read_div_treatment(mpi_context_t& mpi,
                               const std::string& input_file,
                               const std::string& input_grp)
{
  auto div = read_scf_string(mpi, input_file, input_grp, "div_treatment");
  if (!div and input_grp != "scf") {
    div = read_scf_string(mpi, input_file, "scf", "div_treatment");
    if (div) app_log(2, "  '{}/div_treatment' not found in {}; using \"{}\" from the scf group.",
                     input_grp, input_file, *div);
  }
  utils::check(div.has_value(),
               "read_div_treatment: '{}/div_treatment' (and 'scf/div_treatment') missing in {}. "
               "This checkpoint predates div_treatment dump; please regenerate "
               "the scGW checkpoint with the current code.",
               input_grp, input_file);
  return *div;
}

/**
 * Read the hf_div_treatment the ground-state run used (stashed on the checkpoint
 * scf group by chkpt::dump_hf_div_treatment), so LR applies the exchange
 * Madelung term exactly when the unperturbed Fock did.
 *
 * Unlike read_div_treatment this does NOT error when the dataset is absent:
 * checkpoints predating the stash fall back to "gygi", which is both hf_t's
 * default and what LR applied unconditionally before, so they keep their
 * previous behaviour instead of needing to be regenerated. A group without
 * the stash (e.g. the embed group) takes the scf group's.
 */
template<typename mpi_context_t>
std::string read_hf_div_treatment(mpi_context_t& mpi,
                                  const std::string& input_file,
                                  const std::string& input_grp)
{
  auto div = read_scf_string(mpi, input_file, input_grp, "hf_div_treatment");
  if (!div and input_grp != "scf") {
    div = read_scf_string(mpi, input_file, "scf", "hf_div_treatment");
    if (div) app_log(2, "  '{}/hf_div_treatment' not found in {}; using \"{}\" from the scf group.",
                     input_grp, input_file, *div);
  }
  if (!div) {
    app_log(2, "  '{}/hf_div_treatment' (and 'scf/hf_div_treatment') not found in {}; assuming \"gygi\" for the LR "
               "exchange divergence. Regenerate the checkpoint if the ground state "
               "used hf_div_treatment = \"ignore_g0\".", input_grp, input_file);
    return "gygi";
  }
  return *div;
}

/**
 * Helper: load W and eps_inv_head from checkpoint/thc_screened_interaction.h5
 *
 * Called by lr_gw_sigma_DeltaG_calc and gw_evaluate_sigma_calc (MBPT_drivers.cpp) and, through lr_load_W_omega,
 * by run_lr_calc.
 *
 * @param mpi         - MPI context
 * @param thc         - THC ERI (provides Np())
 * @param input_file  - Path to checkpoint HDF5 file (for eps_inv_head)
 * @param input_grp   - SCF group name in checkpoint (e.g., "scf")
 * @param nt          - Number of tau points (from IAFT)
 * @param screened_interaction_file - Explicit path to the W_qtPQ HDF5 file.
 *                      If empty, falls back to thc_screened_interaction.h5 in
 *                      the input checkpoint's directory.
 * @return (dW_tqPQ, eps_inv_head, div_treatment) tuple. W is returned as
 *         (t,q,P,Q): the dataset on disk is (q,t), but eps_inv_head_t needs (t,q)
 *         so that copy is made here regardless — handing it back costs nothing,
 *         while returning (q,t) would make every LR caller transpose it again.
 */
template<typename mpi_context_t, typename THC_t>
auto load_W_and_eps_inv_head(
    mpi_context_t& mpi,
    THC_t& thc,
    const std::string& input_file,
    const std::string& input_grp,
    long nt,
    const std::string& screened_interaction_file = "")
{
  auto mf = thc.MF();
  long nkpts = mf->nkpts();
  long NP = thc.Np();
  long nt_half = (nt % 2 == 0) ? nt / 2 : nt / 2 + 1;

  // Read W from the explicit screened_interaction_file if given; otherwise fall
  // back to thc_screened_interaction.h5 in the input checkpoint's directory.
  auto w_file = screened_interaction_file.empty()
      ? (std::filesystem::path(input_file).parent_path() / "thc_screened_interaction.h5").string()
      : screened_interaction_file;
  app_log(2, "Reading W from {}...", w_file);

  // The dataset on disk is (q,t,P,Q), so the distributed read must target that
  // axis order: h5_read maps the array's index space onto
  // the dataset's and cannot permute axes.
  // τ-dist for (q,t,P,Q): swap axes 0,1 of the (t,q) τ-dist grid
  auto [tq_pgrid, tq_bsize] = utils::lr_W_q_local_dist(mpi.comm.size(), nt_half, NP);
  std::array<long,4> qt_pgrid = {tq_pgrid[1], tq_pgrid[0], tq_pgrid[2], tq_pgrid[3]};
  std::array<long,4> qt_bsize = {tq_bsize[1], tq_bsize[0], tq_bsize[2], tq_bsize[3]};

  using local_Array_4D_t = nda::array<ComplexType, 4>;
  auto dW_qtPQ = math::nda::make_distributed_array<local_Array_4D_t>(
      mpi.comm, qt_pgrid, {nkpts, nt_half, NP, NP}, qt_bsize);

  // Each rank reads only its local (q,t,P,Q) slab via the distributed h5_read.
  // A "root reads full + broadcast" pattern OOMs at scale: full W for nkpts=512,
  // NP=204 is ~30 GB per rank (×2 with broadcast staging), well over per-node RAM
  // when many ranks share a node.
  {
    h5::file file(w_file, 'r');
    h5::group grp(file);
    // Verify the W-on-disk matches the THC currently loaded. Silent shape
    // mismatch between stale W (from scGW build-time) and current THC (from
    // current ISDF) produces a garbage W that explodes downstream by ~|W|²
    // amplification in the ΔW = W·ΔΠ·W Dyson step.
    auto ds_info = h5::array_interface::get_dataset_info(grp, "W_qtPQ");
    long ds_np_a = ds_info.lengths[2];
    long ds_np_b = ds_info.lengths[3];
    utils::check(ds_np_a == NP && ds_np_b == NP,
                 "load_W_and_eps_inv_head: W_qtPQ in {} has shape (...,{},{}) "
                 "but current THC has Np={}. The W file was generated with a "
                 "different THC decomposition; regenerate it (or re-run GW) "
                 "with the current THC before calling run_lr.",
                 w_file, ds_np_a, ds_np_b, NP);
    math::nda::h5_read(grp, "W_qtPQ", dW_qtPQ);
  }
  mpi.comm.barrier();

  auto div_treatment = read_div_treatment(mpi, input_file, input_grp);

  // Recompute eps_inv_head from the loaded W_c
  // We don't read eps_inv_head from input_file to make sure it is consistent with W_c
  imag_axes_ft::IAFT ft(imag_axes_ft::read_iaft(input_file, false));
  auto dW_tqPQ = utils::transpose_axes_01(dW_qtPQ, mpi.comm);
  dW_qtPQ.reset();
  auto [eps_inv_head_q, eps_inv_head] =
      solvers::div_utils::eps_inv_head_t(dW_tqPQ, thc, *mf, &ft, div_treatment);
  mpi.comm.barrier();

  return std::make_tuple(std::move(dW_tqPQ), std::move(eps_inv_head),
                         std::move(div_treatment));
}

/**
 * Helper: read W from the screened-interaction file and hand it to the LR path
 * on the Matsubara axis, so lr_driver has exactly one W layout to support.
 *
 * The τ→ω transform lives here rather than in load_W_and_eps_inv_head because
 * that function's other callers (the standalone scGW Σ paths) want W(τ). Only
 * this from-file path pays the transform; recompute_W_and_eps_inv_head, which
 * solves the W Dyson in ω anyway, hands ω back for free.
 *
 * @param mpi         - MPI context
 * @param thc         - THC ERI
 * @param ft          - IAFT (from checkpoint)
 * @param input_file  - Path to checkpoint HDF5 file
 * @param input_grp   - SCF group name in checkpoint
 * @param screened_interaction_file - Explicit path to W_qtPQ, "" = auto-derive
 * @return (dW_wqPQ, eps_inv_head, div_treatment) — W on the Matsubara axis on
 *         the q-dist distribution (utils::lr_W_tau_local_dist),
 *         head on the τ axis
 */
template<typename mpi_context_t, typename THC_t>
auto lr_load_W_omega(
    mpi_context_t& mpi,
    THC_t& thc,
    imag_axes_ft::IAFT const& ft,
    const std::string& input_file,
    const std::string& input_grp,
    const std::string& screened_interaction_file = "")
{
  auto [dW_tqPQ, eps_inv_head, div_treatment] = load_W_and_eps_inv_head(
      mpi, thc, input_file, input_grp, ft.nt_f(), screened_interaction_file);

  long nw_half = (ft.nw_b() % 2 == 0) ? ft.nw_b() / 2 : ft.nw_b() / 2 + 1;
  auto [w_pgrid, w_bsize] = utils::lr_W_tau_local_dist(
      mpi.comm.size(), nw_half, thc.MF()->nkpts(), thc.Np());

  solvers::scr_coulomb_fourier_t setup_ft(&ft);
  auto dW_wqPQ = setup_ft.tau_to_w(dW_tqPQ, w_pgrid, w_bsize, /*reset_input=*/true,
                                   __app_verbosity__ >= 3);
  mpi.comm.barrier();

  // The same q→0 head on the ω axis. A consumer wanting one Matsubara component
  // (HSEX reads ν = 0) must not recover it by transforming the τ array back:
  // for div_treatment "*_metal", extrapolate_eps_inv_q0 hand-sets
  // eps_inv_q0_w(0) = -1, and a hand-poked point is not generally
  // IR-representable, so the round trip smears exactly that value.
  auto MF = thc.MF();
  auto [eps_inv_head_wq, eps_inv_head_w] =
      solvers::div_utils::eps_inv_head_w(dW_wqPQ, thc, *MF, div_treatment);

  return std::make_tuple(std::move(dW_wqPQ), std::move(eps_inv_head),
                         std::move(eps_inv_head_w), std::move(div_treatment));
}

/**
 * Helper: recompute W and eps_inv_head from the (already-built) unperturbed
 * Green's function instead of reading W from a screened-interaction file.
 *
 * This reuses the exact scGW pipeline (scr_coulomb_t::eval_Pi_qdep +
 * dyson_W_from_Pi_tau + div_utils::eps_inv_head_t), so the recomputed W_c is,
 * by construction, consistent with the currently-loaded THC. RPA screening
 * (standard GW), or GW+EDMFT screening when mb_local is given: its projector
 * and local polarizabilities Π_imp, Π_dc enter through the gw_edmft branch of
 * scr_coulomb_t::eval_Pi_qdep, as in run_gw. cRPA is not supported.
 *
 * W comes back on the **frequency** axis as (w,q,P,Q), on the grid the LR path
 * wants (utils::lr_W_tau_local_dist) — the one layout lr_driver accepts.
 * The W Dyson is solved in ω anyway, so returning ω saves the ω→τ transform
 * back onto Π's τ grid *and* the redistribute onto the LR tiling that would
 * follow it — the LR driver does one ω→τ straight onto its own grid instead.
 *
 * eps_inv_head is built with eps_inv_head_w rather than eps_inv_head_t and then
 * transformed to τ. eval_eps_inv_q is linear and pointwise in the imaginary-axis
 * index, and eps_inv_head_t is exactly "eval → tau_to_w_PHsym → extrapolate →
 * w_to_tau_PHsym", so this is the same quantity with two fewer transforms.
 *
 * @param mpi           - MPI context
 * @param thc           - THC ERI
 * @param ft            - IAFT (from checkpoint)
 * @param sG_tskij      - Unperturbed G(τ) shared array (nt, ns, nk, nb, nb)
 * @param div_treatment - Divergence treatment scGW used (from checkpoint)
 * @return (dW_wqPQ, eps_inv_head) — W on the Matsubara axis, head on the τ axis
 */
template<typename mpi_context_t, typename THC_t, typename sArray_5D_t>
auto recompute_W_and_eps_inv_head(
    mpi_context_t& mpi,
    THC_t& thc,
    imag_axes_ft::IAFT const& ft,
    sArray_5D_t& sG_tskij,
    const std::string& div_treatment,
    MBState* mb_local = nullptr)
{
  auto mf = thc.MF();
  long nkpts = mf->nkpts();

  // RPA polarization Π and W Dyson require the full q-grid (no symmetry).
  utils::check(mf->nqpts() == mf->nqpts_ibz() and mf->nqpts() == nkpts,
               "recompute_W_and_eps_inv_head: No symmetry required. "
               "nqpts={}, nqpts_ibz={}, nkpts={}",
               mf->nqpts(), mf->nqpts_ibz(), nkpts);

  // W is RPA-screened, or GW+EDMFT-screened with the local polarizabilities of mb_local.
  solvers::scr_coulomb_t scr(&ft, (mb_local == nullptr) ? "rpa" : "gw_edmft", div_treatment);

  long nw_half = (ft.nw_b() % 2 == 0) ? ft.nw_b() / 2 : ft.nw_b() / 2 + 1;
  auto [w_pgrid, w_bsize] = utils::lr_W_tau_local_dist(
      mpi.comm.size(), nw_half, nkpts, thc.Np());

  auto dPi_tqPQ = [&]() {
    if (mb_local == nullptr) {
      // RPA
      return scr.eval_Pi_qdep(sG_tskij.local(), thc);
    } else {
      // GW+EDMFT: Π_RPA[G] + upfold(Π_imp - Π_dc). G is lent to mb_local for the call and moved back; this
      // is safe only because no view of sG_tskij exists yet (the Dyson solver and driver are built later).
      utils::check(mb_local->sPi_imp_wabcd.has_value() and mb_local->sPi_dc_wabcd.has_value(),
                   "recompute_W_and_eps_inv_head: GW+EDMFT screening requires the local polarizabilities.");
      mb_local->sG_tskij.emplace(std::move(sG_tskij));
      auto dPi = scr.eval_Pi_qdep(*mb_local, thc);
      sG_tskij = std::move(mb_local->sG_tskij.value());
      mb_local->sG_tskij.reset();
      return dPi;
    }
  }();
  mpi.comm.barrier();
  auto dW_wqPQ = scr.dyson_W_from_Pi_tau<true>(dPi_tqPQ, thc, true, w_pgrid, w_bsize);
  mpi.comm.barrier();

  // eps_inv_head_w returns the head at every (iω, q) plus the q→0 extrapolated
  // head at every iω; only the latter is wanted here.
  auto [eps_inv_head_wq, eps_inv_head_q0_w] =
      solvers::div_utils::eps_inv_head_w(dW_wqPQ, thc, *mf, div_treatment);

  // The head is consumed on the τ axis (lr_gw::apply_div_correction_*), so make
  // the same final ω→τ step eps_inv_head_t ends with.
  long nt_half = (ft.nt_b() % 2 == 0) ? ft.nt_b() / 2 : ft.nt_b() / 2 + 1;
  nda::array<ComplexType, 1> eps_inv_head(nt_half);
  auto head_w_2D = nda::reshape(eps_inv_head_q0_w, std::array<long, 2>{nw_half, 1});
  auto head_t_2D = nda::reshape(eps_inv_head, std::array<long, 2>{nt_half, 1});
  ft.w_to_tau_PHsym(head_w_2D, head_t_2D);

  return std::make_tuple(std::move(dW_wqPQ), std::move(eps_inv_head),
                         std::move(eps_inv_head_q0_w));
}

using lr_W_t = memory::darray_t<memory::array<HOST_MEMORY, ComplexType, 4>, mpi3::communicator>;

/// Drop the W_c(iω) and ε⁻¹ head kept by a reuse_W call of load_lr_W (run_lr_calc or run_lr_efield_calc). No-op if none
/// is kept. Called by Python `release_lr_W_cache` (declared in MBPT_drivers.h as well).
void release_lr_W_cache();

/**
 * Reads the `iter_alg.*` keys of an LR run into lr_iter_params.
 *
 * Called by run_lr_calc and run_lr_efield_calc, once per run; shared so that every LR driver reads the same keys with
 * the same defaults.
 */
lr_iter_params read_lr_iter_params(ptree const& pt, bool qp_static_sigma);

/// The unperturbed state an LR run linearizes about; built by load_lr_unperturbed_state.
struct lr_unperturbed_state {
  simple_dyson dyson;
  math::shm::shared_array<Array_view_4D_t> sF_skij, sDm_skij;
  /// G(τ), read by the W recompute and the GW+EDMFT Gloc before lr_driver::lr_setup and released there
  std::optional<math::shm::shared_array<Array_view_5D_t>> opt_sG_tskij;
  math::shm::shared_array<Array_view_5D_t> sSigma_tskij;
  /// Exact unperturbed G(iω) that lr_dyson linearizes about
  math::shm::shared_array<Array_view_5D_t> sG_wskij;
  double mu = 0.0;
};

/**
 * Builds the unperturbed Dyson solver, F, Σ(τ), μ, G(τ), Dm and G(iω) of an LR run.
 *
 * Called by run_lr_calc and run_lr_efield_calc after their input checks. `unperturbed` = "checkpoint" reads F, Σ, μ from the checkpoint
 * and solves the Dyson equation; "mf_dft" builds the DFT/KS mean-field G0 (F, Σ stay unused). `ft` stays
 * caller-owned: the Dyson solver keeps a pointer to it. The timers LR_INIT_DYSON, LR_INIT_READ_SCF and
 * LR_INIT_UPDATE_G of `init_timer` are the run's init table.
 */
lr_unperturbed_state load_lr_unperturbed_state(utils::mpi_context_t<mpi3::communicator>& mpi, mf::MF& mf,
                                               imag_axes_ft::IAFT& ft, std::string const& h0_source,
                                               std::string const& prefix, std::string const& input_grp,
                                               long input_iter, std::string const& unperturbed,
                                               utils::TimerManager& init_timer);

/// W_c(iω), the ε⁻¹ heads and the divergence treatments an LR run hands to lr_driver; built by load_lr_W.
struct lr_W_inputs {
  std::optional<lr_W_t> opt_dW;
  std::optional<nda::array<ComplexType, 1>> opt_eps_inv;
  /// The same head on the ω axis; HSEX reads its ν = 0 component.
  std::optional<nda::array<ComplexType, 1>> opt_eps_inv_w;
  std::string div_treatment = "gygi";
  std::string hf_div_treatment;
};

/**
 * Loads (or recomputes, or reuses) the screened interaction W_c(iω) of an LR run, with the ε⁻¹ heads, and reads
 * the divergence treatments of the ground state.
 *
 * Called by run_lr_calc and run_lr_efield_calc, once per run, after the unperturbed state is built. W is either read
 * from disk (explicit screened_interaction_file, else the path derived from the checkpoint), recomputed on the fly from the G(τ)
 * `sG_tskij` (RPA, or GW+EDMFT when `mb_local` is non-null), or taken from the process-wide W cache (`reuse_W`, released by release_lr_W_cache).
 * Both hand W to lr_driver as W_c(iω) on the q-dist distribution (utils::lr_W_tau_local_dist). `needs_W` is
 * false when the kernel carries neither a Σ nor the HSEX exchange; then only the divergence treatment of the
 * exchange side is read. `init_timer` carries LR_INIT_LOAD_W.
 */
lr_W_inputs load_lr_W(utils::mpi_context_t<mpi3::communicator>& mpi, thc_reader_t& thc, imag_axes_ft::IAFT const& ft,
                      std::string const& input_file, std::string const& input_grp, ptree const& pt,
                      bool needs_W, bool recompute_W, bool reuse_W,
                      std::string const& screened_interaction_file, std::string const& unperturbed,
                      sArray_t<Array_view_5D_t>& sG_tskij, MBState* mb_local, utils::TimerManager& init_timer);

/**
 * Sets the kernel, solver and W fields of `p` that every LR driver takes from its inputs.
 *
 * Called by run_lr_calc and run_lr_efield_calc right before lr_driver::lr_setup. The pointers into `W` stay valid only while `W` lives, which is as
 * long as the lr_driver that `p` is given to.
 */
void fill_lr_kernel_params(lr_params& p, lr_kernel_spec const& kernel, lr_W_inputs const& W,
                           lr_iter_params const& iter_params, long max_iter, double tol, bool fix_density,
                           bool save_DeltaG);

} // namespace methods

#endif // COQUI_LR_RUN_INPUTS_HPP
