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

#include "methods/SCF/lr_dyson.hpp"
#include "nda/nda.hpp"
#include "nda/blas.hpp"

namespace methods {

lr_dyson::lr_dyson(simple_dyson& dyson, nda::array<double, 1> const& q_vec)
    : _dyson(dyson),
      _context(dyson.mpi()),
      _nts(dyson.FT()->nt_f()),
      _nw(dyson.FT()->nw_f()),
      _ns(dyson.MF()->nspin()),
      _nkpts(dyson.MF()->nkpts()),
      _nkpts_ibz(dyson.MF()->nkpts_ibz()),
      _nbnd(dyson.MF()->nbnd()),
      _q_vec(q_vec),
      _kpq_map(_nkpts),
      _Timer() {

  // Compute k+q mapping
  auto kpts_crys = dyson.MF()->kpts_crystal();
  utils::calculate_kpq_map(kpts_crys, _q_vec, _kpq_map);

  // Check if q is approximately gamma
  _is_q_gamma = utils::is_q_gamma(_q_vec);

  app_log(2, "LR Dyson equation initialized:");
  app_log(2, "  - Perturbation wavevector q = ({:.6f}, {:.6f}, {:.6f})",
          _q_vec(0), _q_vec(1), _q_vec(2));
  app_log(2, "  - q is Gamma point: {}", _is_q_gamma ? "yes" : "no");

  for (auto& v : {"LR_DYSON", "LR_DYSON_TAU_TO_W", "LR_DYSON_LOOP", "LR_DYSON_GATHER",
                  "LR_DYSON_ALLOC", "LR_DYSON_REDIST", "LR_DYSON_G_W_TO_T",
                  "LR_DYSON_DM", "LR_DYSON_NELEC", "LR_DYSON_DELTAMU", "LR_DYSON_MISC",
                  "GATHER_SHM_ZERO", "GATHER_SHM_ASSIGN", "GATHER_SHM_SKEW",
                  "GATHER_SHM_REDUCE", "GATHER_SHM_BARRIER"}) {
    _Timer.add(v);
  }
  _context->comm.barrier();
}


template<typename DeltaDm_t, typename DeltaH0_t,
         typename DeltaF_t, typename DeltaSigma_t>
double lr_dyson::solve_lr_dyson(
    DeltaDm_t& sDeltaDm_skij,
    const DeltaH0_t& sDeltaH0_skij,
    const DeltaF_t& sDeltaF_skij,
    const DeltaSigma_t* sDeltaSigma_tskij,
    bool fix_density,
    const DeltaF_t* sDeltaVcorr_skij) {

  utils::check(_cached_G_wskij != nullptr,
               "solve_lr_dyson: cached G(iω) not set. Call set_cached_G_omega() first.");

  _Timer.start("LR_DYSON");

  // A ΔG(τ) nobody gathered before this solve is dead by definition. Dropped
  // before anything is allocated, so the in-solve memory peak is unchanged.
  _dDeltaG_tau_buffer.reset();

  // Processor grid of the ω-side arrays. A pure function of the sizes, so
  // solve_lr_dyson_impl derives the same one and the two agree by construction.
  auto [w_pgrid, w_bsize] =
      lr_dyson_omega_pgrid(_context->comm.size(), _nw, _nkpts_ibz, _nbnd);

  // ΔΣ(τ) → ΔΣ(iω) is done here rather than inside solve_lr_dyson_impl, keeping
  // the transform and its clock out of the Dyson pass proper.
  // A local rather than a member — ΔΣ changes every SCF iteration.
  std::optional<dArray_5D_t> opt_dDeltaSigma_wskij;
  if (sDeltaSigma_tskij) {
    _Timer.start("LR_DYSON_TAU_TO_W");
    // ΔΣ leakage diagnostics gated at verbosity >= 3.
    opt_dDeltaSigma_wskij.emplace(
        distributed_tau_to_w(_context->comm, *sDeltaSigma_tskij, *_dyson.FT(), w_pgrid, w_bsize,
                             __app_verbosity__ >= 3));
    _Timer.stop("LR_DYSON_TAU_TO_W");
  }
  const dArray_5D_t* dDeltaSigma_wskij =
      opt_dDeltaSigma_wskij ? &(*opt_dDeltaSigma_wskij) : nullptr;

  // At q≠0 the Δμ·S term vanishes and Δμ is meaningless.
  if (!_is_q_gamma and fix_density)
    app_log(3, "solve_lr_dyson: fix_density ignored for q≠0 (Δμ term vanishes)");

  // The perturbation is solved at fixed μ, so the Dyson pass always runs at
  // Δμ = 0; only fix_density at q=0 shifts off it, in three steps:
  // 1. the single Dyson pass below, giving ΔG(0) and ΔN(0)
  // 2. Δμ = -ΔN(0) / (dN/dμ), with dN/dμ from the cached Δμ response
  // 3. ΔG(Δμ) = ΔG(0) + Δμ·dG/dμ and ΔDm(Δμ) = ΔDm(0) + Δμ·dDm/dμ
  //
  // Δμ reaches the RHS only through the −Δμ·S term, so ΔG is affine in it and
  // dG/dμ is a function of the reference G(iω) and S alone — the same statement
  // the closed form in step 2 rests on. build_dmu_response() builds that response
  // once, before the SCF loop, which makes step 3 a local axpy (exact in exact
  // arithmetic) rather than a second Dyson pass.
  //
  // There is therefore one Dyson pass in every mode, always at Δμ = 0; only
  // steps 2-3 are conditional.
  double Delta_mu = 0.0;
  double dN_dmu = 0.0;
  if (fix_density and _is_q_gamma) {
    utils::check(_dN_dmu_cached and _dG_dmu_tskij and _sdDm_dmu_skij,
                 "solve_lr_dyson: fix_density=true but the Δμ response is not cached. "
                 "Call build_dmu_response() before the SCF loop.");

    app_log(3, "solve_lr_dyson: fix_density mode (computing Δμ to enforce ΔN=0)");
    dN_dmu = _cached_dN_dmu;
  }

  // Step 1 in fix_density mode, the whole solve otherwise.
  solve_lr_dyson_impl(sDeltaDm_skij, sDeltaH0_skij, sDeltaF_skij,
                      dDeltaSigma_wskij, /*Delta_mu=*/0.0,
                      sDeltaVcorr_skij);

  if (fix_density and _is_q_gamma) {
    _Timer.start("LR_DYSON_NELEC");
    const double DeltaN_0 = compute_lr_Nelec(sDeltaDm_skij);
    _Timer.stop("LR_DYSON_NELEC");

    if (std::abs(dN_dmu) < 1e-15) {
      // Δμ stays 0, which leaves ΔDm untouched, so ΔN(Δμ=0) is already the final
      // ΔN — and it is exactly the density error the warning is about.
      app_log(1, "[WARNING] solve_lr_dyson: dN/dμ ≈ 0, cannot compute Δμ. Using Δμ=0.");
      app_log(3, "  Final ΔN = {:.6e} (density not restored)", DeltaN_0);
    } else {
      // Closed-form solution: Δμ = -ΔN(0) / (dN/dμ)
      // (The LR Dyson equation is linear in Δμ, so this is exact)
      Delta_mu = -DeltaN_0 / dN_dmu;
      app_log(3, "  ΔN(Δμ=0) = {:.6e}, dN/dμ = {:.6e}", DeltaN_0, dN_dmu);
      app_log(3, "  Computed Δμ = {:.6e}", Delta_mu);

      // Step 3, as a local axpy on the Δμ=0 solution:
      //   ΔG(τ; Δμ) = ΔG(τ; 0) + Δμ·(dG/dμ)(τ)
      //   ΔDm(Δμ)   = ΔDm(0)   + Δμ·(dDm/dμ)
      _Timer.start("LR_DYSON_DELTAMU");
      apply_dmu_shift(sDeltaDm_skij, Delta_mu);
      _Timer.stop("LR_DYSON_DELTAMU");

      // Verify ΔN ≈ 0
      _Timer.start("LR_DYSON_NELEC");
      const double DeltaN_final = compute_lr_Nelec(sDeltaDm_skij);
      _Timer.stop("LR_DYSON_NELEC");
      app_log(3, "  Final ΔN = {:.6e} (should be ~0)", DeltaN_final);
    }
  }

  _Timer.stop("LR_DYSON");
  print_timers(3);  // per-step diagnostics only at verbosity >= 3

  return Delta_mu;
}


template<typename DeltaDm_t, typename DeltaH0_t, typename DeltaF_t>
void lr_dyson::solve_lr_dyson_impl(
    DeltaDm_t& sDeltaDm_skij,
    const DeltaH0_t& sDeltaH0_skij,
    const DeltaF_t& sDeltaF_skij,
    const dArray_5D_t* dDeltaSigma_wskij,
    double Delta_mu,
    const DeltaF_t* sDeltaVcorr_skij) {

  using math::nda::make_distributed_array;
  using Array_5D_t = nda::array<ComplexType, 5>;
  using Array_4D_t = nda::array<ComplexType, 4>;

  // Δμ·S term is only meaningful at q=0; assert no accidental nonzero value at q≠0
  utils::check(_is_q_gamma || std::abs(Delta_mu) < 1e-15,
               "solve_lr_dyson_impl: Delta_mu = {:.6e} but q≠0. "
               "Delta_mu must be zero for q≠0 perturbations.", Delta_mu);

  app_log(3, "Solving LR Dyson equation:");
  if (dDeltaSigma_wskij) {
    app_log(3, "  ΔG(k,iω) = G(k+q,iω) · [ΔH0(k) + ΔF(k) + ΔΣ(k,iω) - Δμ·S(k+q,k)] · G(k,iω)");
  } else {
    app_log(3, "  ΔG(k,iω) = G(k+q,iω) · [ΔH0(k) + ΔF(k) - Δμ·S(k+q,k)] · G(k,iω)");
  }
  app_log(3, "  Δμ = {:.6e}", Delta_mu);

  // Cached G(iω) in shared memory — all (iω, k) points accessible to all ranks
  auto G_w = _cached_G_wskij->local();

  // Processor grid for ΔG (and ΔΣ if present)
  auto [w_pgrid, w_bsize] =
      lr_dyson_omega_pgrid(_context->comm.size(), _nw, _nkpts_ibz, _nbnd);
  utils::check(w_pgrid[0] * w_pgrid[2] * w_pgrid[3] * w_pgrid[4] == _context->comm.size(),
               "lr_dyson: pgrid mismatches!");

  _Timer.start("LR_DYSON_ALLOC");
  auto dDeltaG_wskij = make_distributed_array<Array_5D_t>(_context->comm, w_pgrid,
                                                          {_nw, _ns, _nkpts_ibz, _nbnd, _nbnd}, w_bsize);
  _Timer.stop("LR_DYSON_ALLOC");

  // Loop bounds from the output distributed array. lr_dyson_omega_pgrid keeps
  // the band axes undivided whenever the ω axis has room, but in the fallback
  // each rank owns only the (i_rng, j_rng) block of ΔG.
  auto [nw_loc, ns_loc, nk_loc, ni_loc, nj_loc] = dDeltaG_wskij.local_shape();
  auto [w_org, s_org, k_org, i_org, j_org] = dDeltaG_wskij.origin();
  auto i_rng = dDeltaG_wskij.local_range(3);
  auto j_rng = dDeltaG_wskij.local_range(4);

  // ΔG = G_{k+q}·X·G_k couples every band, so X must be the full nbnd × nbnd
  // matrix. ΔH0, ΔF, S and ΔV_QPGW live in shared memory and are always whole,
  // but ΔΣ(iω) is distributed on w_pgrid: in the band-split fallback no rank
  // holds all of it and the product cannot be formed.
  utils::check(dDeltaSigma_wskij == nullptr or w_pgrid[3]*w_pgrid[4] == 1,
               "solve_lr_dyson_impl: ΔΣ is distributed over the band axes "
               "(pgrid (w,s,k,i,j) = ({},{},{},{},{})), which the ΔG = G·X·G "
               "product does not support. This happens only when nw = {} is too "
               "small to absorb the ranks the (ω, k) pools left over; run LR-GW "
               "on fewer ranks, or with a finer imaginary-frequency grid.",
               w_pgrid[0], w_pgrid[1], w_pgrid[2], w_pgrid[3], w_pgrid[4], _nw);

  auto DeltaG_w_loc = dDeltaG_wskij.local();
  auto DeltaH0_loc = sDeltaH0_skij.local();
  auto DeltaF_loc = sDeltaF_skij.local();
  auto S_loc = _dyson.sS_skij().local();
  // Optional static ΔV_QPGW (LR-qpGW): frequency-independent one-body term.
  bool has_Vcorr = (sDeltaVcorr_skij != nullptr);

  // Temporary matrices
  nda::matrix<ComplexType> X_ij(_nbnd, _nbnd);
  nda::matrix<ComplexType> tmp(_nbnd, _nbnd);
  nda::matrix<ComplexType> DeltaG_ij(_nbnd, _nbnd);

  // LR Dyson loop — G is read from shared memory with global indices
  _Timer.start("LR_DYSON_LOOP");
  for (long n = 0; n < nw_loc; ++n) {
    for (long s = 0; s < ns_loc; ++s) {
      for (long k = 0; k < nk_loc; ++k) {
        long iw = n + w_org;
        long ik = k + k_org;
        long is = s + s_org;
        long ikq = _kpq_map(ik);

        auto G_k = G_w(iw, is, ik, nda::range::all, nda::range::all);
        auto G_kq = G_w(iw, is, ikq, nda::range::all, nda::range::all);
        auto DeltaH0_k = DeltaH0_loc(is, ik, nda::range::all, nda::range::all);
        auto DeltaF_k = DeltaF_loc(is, ik, nda::range::all, nda::range::all);

        // Build X = ΔH0 + ΔF [+ ΔΣ] [+ ΔV_QPGW] - Δμ·S
        auto S_k = S_loc(is, ik, nda::range::all, nda::range::all);
        X_ij = DeltaH0_k + DeltaF_k - Delta_mu * S_k;
        if (dDeltaSigma_wskij) {
          auto DeltaSigma_w_loc = dDeltaSigma_wskij->local();
          auto DeltaSigma_k = DeltaSigma_w_loc(n, s, k, nda::range::all, nda::range::all);
          X_ij += DeltaSigma_k;
        }
        if (has_Vcorr) {
          X_ij += sDeltaVcorr_skij->local()(is, ik, nda::range::all, nda::range::all);
        }

        // ΔG = G_{k+q} @ X @ G_k
        nda::blas::gemm(ComplexType(1.0), X_ij, G_k, ComplexType(0.0), tmp);
        nda::blas::gemm(ComplexType(1.0), G_kq, tmp, ComplexType(0.0), DeltaG_ij);

        DeltaG_w_loc(n, s, k, nda::ellipsis{}) = DeltaG_ij(i_rng, j_rng);
      }
    }
  }
  _Timer.stop("LR_DYSON_LOOP");

  _Timer.start("LR_DYSON_MISC");
  _context->comm.barrier();
  _Timer.stop("LR_DYSON_MISC");

  // Convert ΔG(w) -> ΔG(tau)
  {
    int np = _context->comm.size();
    long nkpools = utils::find_proc_grid_max_npools(np, _nkpts_ibz, 0.2);
    np /= nkpools;
    long np_i = utils::find_proc_grid_min_diff(np, 1, 1);
    long np_j = np / np_i;

    _Timer.start("LR_DYSON_ALLOC");
    auto dDeltaG_wskij_tmp = make_distributed_array<Array_5D_t>(_context->comm,
        {1, 1, nkpools, np_i, np_j}, {_nw, _ns, _nkpts_ibz, _nbnd, _nbnd});
    _Timer.stop("LR_DYSON_ALLOC");

    _Timer.start("LR_DYSON_REDIST");
    math::nda::redistribute(dDeltaG_wskij, dDeltaG_wskij_tmp);
    _Timer.stop("LR_DYSON_REDIST");

    _Timer.start("LR_DYSON_MISC");
    dDeltaG_wskij.reset();
    _Timer.stop("LR_DYSON_MISC");

    _Timer.start("LR_DYSON_ALLOC");
    auto dDeltaG_tskij = make_distributed_array<Array_5D_t>(_context->comm,
        {1, 1, nkpools, np_i, np_j}, {_nts, _ns, _nkpts_ibz, _nbnd, _nbnd});
    _Timer.stop("LR_DYSON_ALLOC");

    auto DeltaGt_loc = dDeltaG_tskij.local();
    auto DeltaGw_loc = dDeltaG_wskij_tmp.local();
    _Timer.start("LR_DYSON_G_W_TO_T");
    _dyson.FT()->w_to_tau(DeltaGw_loc, DeltaGt_loc, imag_axes_ft::fermion);
    _Timer.stop("LR_DYSON_G_W_TO_T");

    _Timer.start("LR_DYSON_MISC");
    dDeltaG_wskij_tmp.reset();
    _Timer.stop("LR_DYSON_MISC");

    // ΔDm = -ΔG(τ=β⁻), formed on the distributed τ array before it is replicated.
    // tau_to_beta contracts the leading τ axis only and the pgrid above leaves τ
    // and spin whole, so each rank evaluates exactly its own (k, i, j) block; the
    // 4-D pgrid below reproduces the same rank->block map, so the two line up.
    _Timer.start("LR_DYSON_DM");
    {
      auto dDeltaDm_skij = make_distributed_array<Array_4D_t>(_context->comm,
          {1, nkpools, np_i, np_j}, {_ns, _nkpts_ibz, _nbnd, _nbnd});

      // The rank->block map has to agree on (s, k, i, j) or tau_to_beta writes this
      // rank's τ contraction into someone else's block: sizes still match, the
      // gather still reduces disjoint blocks, and ΔDm comes out silently wrong.
      // Enforced rather than assumed, since it rests on make_distributed_array
      // treating a leading pgrid axis of 1 as a no-op.
      {
        auto g_org = dDeltaG_tskij.origin();
        auto g_lsh = dDeltaG_tskij.local_shape();
        auto d_org = dDeltaDm_skij.origin();
        auto d_lsh = dDeltaDm_skij.local_shape();
        for (int r = 0; r < 4; ++r)
          utils::check(g_org[r + 1] == d_org[r] and g_lsh[r + 1] == d_lsh[r],
                       "solve_lr_dyson_impl: ΔG(τ) and ΔDm disagree on axis {} — "
                       "origin {} vs {}, local shape {} vs {}. The 5-D grid "
                       "(1,1,{},{},{}) and the 4-D grid (1,{},{},{}) must map every "
                       "rank onto the same (s, k, i, j) block.",
                       r, g_org[r + 1], d_org[r], g_lsh[r + 1], d_lsh[r],
                       nkpools, np_i, np_j, nkpools, np_i, np_j);
      }

      _dyson.FT()->tau_to_beta(dDeltaG_tskij.local(), dDeltaDm_skij.local());
      dDeltaDm_skij.local() *= -1.0;

      math::nda::gather_to_shm(dDeltaDm_skij, sDeltaDm_skij);
    }
    _Timer.stop("LR_DYSON_DM");

    // Handed over distributed; materialize_DeltaG_tau() replicates it into the
    // caller's shared array if and when something reads it.
    _dDeltaG_tau_buffer.emplace(std::move(dDeltaG_tskij));

    _gather_bytes = sizeof(ComplexType) * _nts * _ns * _nkpts_ibz * _nbnd * _nbnd;
  }

  _Timer.start("LR_DYSON_MISC");
  _context->comm.barrier();
  _Timer.stop("LR_DYSON_MISC");
}


template<typename DeltaDm_t>
void lr_dyson::apply_dmu_shift(DeltaDm_t& sDeltaDm_skij, double Delta_mu) {
  // dG/dμ(τ) came out of solve_lr_dyson_impl, so its grid is derived from the same
  // (comm.size(), nkpts_ibz, nbnd) as the ΔG(τ) sitting in _dDeltaG_tau_buffer
  // and the two blocks coincide. Checked rather than assumed: a mismatch would
  // add one rank's block to another's, which is silent and wrong everywhere.
  utils::check(bool(_dDeltaG_tau_buffer),
               "apply_dmu_shift: no ΔG(τ) retained — the Δμ=0 pass must run first.");
  utils::check(_dDeltaG_tau_buffer->origin() == _dG_dmu_tskij->origin() and
               _dDeltaG_tau_buffer->local_shape() == _dG_dmu_tskij->local_shape(),
               "apply_dmu_shift: ΔG(τ) and dG/dμ(τ) are distributed differently.");

  _dDeltaG_tau_buffer->local() += ComplexType(Delta_mu) * _dG_dmu_tskij->local();

  // ΔDm and dDm/dμ are both node-replicated, so the add runs once per node.
  sDeltaDm_skij.win().fence();
  if (_context->node_comm.root())
    sDeltaDm_skij.local() += ComplexType(Delta_mu) * _sdDm_dmu_skij->local();
  sDeltaDm_skij.win().fence();
}


void lr_dyson::materialize_DeltaG_tau(sArray_t<Array_view_5D_t>& sDeltaG_tskij) {
  // Unconditional, not idempotent: a solve leaves one ΔG(τ) in the buffer and
  // the caller gathers it at most once. Checked rather than silently skipped,
  // so a caller that loses track fails here instead of reading a stale ΔG(τ).
  utils::check(bool(_dDeltaG_tau_buffer),
               "lr_dyson::materialize_DeltaG_tau: no ΔG(τ) to replicate. Either "
               "solve_lr_dyson() has not run since the last call, or ΔG(τ) was "
               "already replicated.");

  _Timer.start("LR_DYSON_GATHER");
  math::nda::gather_to_shm(*_dDeltaG_tau_buffer, sDeltaG_tskij, &_Timer);
  _Timer.stop("LR_DYSON_GATHER");

  _dDeltaG_tau_buffer.reset();
}


template<typename DeltaDm_t>
double lr_dyson::compute_lr_Nelec(const DeltaDm_t& sDeltaDm_skij) {
  // ΔN = Tr[S · ΔDm] = Σ_k w_k Tr[S(k) · ΔDm(k)]
  utils::check(_is_q_gamma,
               "compute_lr_Nelec: ΔN is not well-defined for q≠0. "
               "This function should only be called for q=0 perturbations.");

  auto k_weight = _dyson.MF()->k_weight();
  auto S_loc = _dyson.sS_skij().local();
  auto DeltaDm_loc = sDeltaDm_skij.local();

  // Spin factor: for unpolarized calculations (ns=1, npol=1), count each electron twice
  double spin_factor = (_ns == 1 && _dyson.MF()->npol() == 1) ? 2.0 : 1.0;

  ComplexType DeltaN(0.0);

  // (s, k) is distributed over the whole communicator — ΔDm and S are both shared
  // per node, so any rank can evaluate any item.
  int rank = _context->comm.rank();
  int size = _context->comm.size();
  nda::matrix<ComplexType> buffer(_nbnd, _nbnd);
  for (int i = rank; i < _ns * _nkpts_ibz; i += size) {
    int is = i / _nkpts_ibz;
    int ik = i % _nkpts_ibz;
    auto S_k = S_loc(is, ik, nda::range::all, nda::range::all);
    auto DeltaDm_k = DeltaDm_loc(is, ik, nda::range::all, nda::range::all);

    // Tr[S · ΔDm] via matrix multiply and trace
    nda::blas::gemm(ComplexType(1.0), S_k, DeltaDm_k, ComplexType(0.0), buffer);
    DeltaN += k_weight(ik) * nda::trace(buffer);
  }

  DeltaN = _context->comm.all_reduce_value(DeltaN);
  // after the reduce, so it multiplies the total exactly once
  DeltaN *= spin_factor;

  if (std::abs(DeltaN.imag()) > 1e-10) {
    app_log(1, "[WARNING] compute_lr_Nelec: Im(ΔN) = {:.2e}", DeltaN.imag());
  }

  app_log(3, "compute_lr_Nelec: ΔN = {:.6e}", DeltaN.real());
  return DeltaN.real();
}


void lr_dyson::build_dmu_response() {
  // Builds the whole Δμ-response cache: dG/dμ(τ), dDm/dμ and dN/dμ. They all
  // come from one Dyson pass, so they are produced together and never separately.
  //
  // ΔG(k,iω) = G_{k+q}·[ΔH0 + ΔF + ΔΣ + ΔV_QPGW − Δμ·S]·G_k is affine in Δμ:
  //   ΔG(Δμ) = ΔG(0) + Δμ·dG/dμ,   dG/dμ(k,iω) = −G_{k+q}(iω)·S(k)·G_k(iω),
  // and w_to_tau / tau_to_beta are linear, so ΔDm = −ΔG(β⁻) obeys the same
  // relation with dDm/dμ = −(dG/dμ)(β⁻). fix_density therefore needs one Dyson
  // pass and an axpy, not two passes.
  //
  // dG/dμ is built by the Dyson kernel itself, with a zero one-body perturbation,
  // no ΔΣ/ΔV_QPGW and Δμ = 1 ⇒ X = −S. Then ΔG(0) = 0 and the pass returns
  // exactly ΔG(Δμ=1) = dG/dμ and ΔDm(Δμ=1) = dDm/dμ, with no post-hoc sign to
  // get wrong, and laid out the way the real solve lays out its ΔG(τ) by
  // construction.
  //
  // Idempotent, so lr_driver can call it as the fix_density setup hook without
  // tracking whether the current G(iω) has already been through it.
  if (_dN_dmu_cached) return;

  utils::check(_is_q_gamma,
               "build_dmu_response: the Δμ response is not meaningful for q≠0. "
               "This should only be called for q=0 perturbations.");
  utils::check(_cached_G_wskij != nullptr,
               "build_dmu_response: cached G(iω) not set. "
               "Call set_cached_G_omega() first.");

  auto sZero_skij = math::shm::make_shared_array<Array_view_4D_t>(
      *_context, {_ns, _nkpts_ibz, _nbnd, _nbnd});
  auto sdDm_dmu_skij = math::shm::make_shared_array<Array_view_4D_t>(
      *_context, {_ns, _nkpts_ibz, _nbnd, _nbnd});

  app_log(3, "build_dmu_response: building dG/dμ(τ) = ∂ΔG(τ)/∂Δμ");
  solve_lr_dyson_impl(sdDm_dmu_skij, sZero_skij, sZero_skij,
                      /*opt_dDeltaSigma_wskij=*/nullptr, /*Delta_mu=*/1.0,
                      static_cast<const sArray_t<Array_view_4D_t>*>(nullptr));

  utils::check(bool(_dDeltaG_tau_buffer),
               "build_dmu_response: the Dyson pass retained no ΔG(τ).");
  _dG_dmu_tskij.emplace(std::move(*_dDeltaG_tau_buffer));
  _dDeltaG_tau_buffer.reset();
  _sdDm_dmu_skij.emplace(std::move(sdDm_dmu_skij));

  // N = Tr[S·Dm] summed over k with the spin factor, so differentiating at fixed
  // everything-else gives dN/dμ = spin·Σ_k w_k Tr[S(k)·(dDm/dμ)(k)] — exactly
  // compute_lr_Nelec of the response matrix. Taking it from the response rather
  // than from a separate (G·S·G)(β⁻) loop keeps one definition of dN/dμ, and it
  // is the one the Δμ = -ΔN(0)/(dN/dμ) closed form must be consistent with for
  // the axpy to land on ΔN = 0.
  _cached_dN_dmu = compute_lr_Nelec(*_sdDm_dmu_skij);
  _dN_dmu_cached = true;
  app_log(2, "build_dmu_response: dN/dμ = {:.12e} (cached)", _cached_dN_dmu);

  // This ran one full Dyson pass through the shared sub-clocks. It happens once,
  // before any solve, and lr_driver already bills it to LR_DRIVER_SETUP_DELTAMU,
  // so clear them here and leave the LR Dyson report counting SCF-loop work only.
  _Timer.reset();
}


// Template instantiations
template double lr_dyson::solve_lr_dyson(
    sArray_t<Array_view_4D_t>&,
    const sArray_t<Array_view_4D_t>&,
    const sArray_t<Array_view_4D_t>&,
    const sArray_t<Array_view_5D_t>*,
    bool,
    const sArray_t<Array_view_4D_t>*);

template void lr_dyson::solve_lr_dyson_impl(
    sArray_t<Array_view_4D_t>&,
    const sArray_t<Array_view_4D_t>&,
    const sArray_t<Array_view_4D_t>&,
    const lr_dyson::dArray_5D_t*,
    double,
    const sArray_t<Array_view_4D_t>*);

template double lr_dyson::compute_lr_Nelec(
    const sArray_t<Array_view_4D_t>&);

} // namespace methods
