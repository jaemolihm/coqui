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
                  "LR_DYSON_DM", "LR_DYSON_NELEC", "LR_DYSON_MISC"}) {
    _Timer.add(v);
  }
  _context->comm.barrier();
}


template<typename DeltaG_t, typename DeltaDm_t, typename DeltaH0_t,
         typename DeltaF_t, typename DeltaSigma_t>
double lr_dyson::solve_lr_dyson(
    DeltaG_t& sDeltaG_tskij,
    DeltaDm_t& sDeltaDm_skij,
    const DeltaH0_t& sDeltaH0_skij,
    const DeltaF_t& sDeltaF_skij,
    const DeltaSigma_t* sDeltaSigma_tskij,
    bool fix_density,
    double Delta_mu,
    const DeltaF_t* sDeltaVcorr_skij) {

  utils::check(_cached_G_wskij != nullptr,
               "solve_lr_dyson: cached G(iω) not set. Call set_cached_G_omega() first.");

  _Timer.start("LR_DYSON");

  // For fix_density mode at q=0, we need to compute Δμ
  // The LR Dyson equation is linear in Δμ, so we solve twice:
  // 1. ΔG(0) with Δμ=0 to get ΔN(0)
  // 2. Use cached dN/dμ to compute Δμ = -ΔN(0) / (dN/dμ)
  // 3. ΔG_final with the computed Δμ
  if (fix_density && _is_q_gamma) {
    utils::check(_dN_dmu_cached,
                 "solve_lr_dyson: fix_density=true but dN/dμ not cached. "
                 "Call compute_dN_dmu() before the SCF loop.");

    app_log(3, "solve_lr_dyson: fix_density mode (computing Δμ to enforce ΔN=0)");

    // First pass: solve with Δμ=0
    solve_lr_dyson_impl(sDeltaG_tskij, sDeltaH0_skij,
                        sDeltaF_skij, sDeltaSigma_tskij, 0.0, sDeltaVcorr_skij);
    _Timer.start("LR_DYSON_DM");
    compute_lr_dm(sDeltaDm_skij, sDeltaG_tskij);
    _Timer.stop("LR_DYSON_DM");

    // Compute ΔN at Δμ=0
    _Timer.start("LR_DYSON_NELEC");
    double DeltaN_0 = compute_lr_Nelec(sDeltaDm_skij);
    _Timer.stop("LR_DYSON_NELEC");

    // Use cached dN/dμ
    double dN_dmu = _cached_dN_dmu;

    if (std::abs(dN_dmu) < 1e-15) {
      app_log(1, "[WARNING] solve_lr_dyson: dN/dμ ≈ 0, cannot compute Δμ. Using Δμ=0.");
      Delta_mu = 0.0;
    } else {
      // Closed-form solution: Δμ = -ΔN(0) / (dN/dμ)
      // (The LR Dyson equation is linear in Δμ, so this is exact)
      Delta_mu = -DeltaN_0 / dN_dmu;
      app_log(3, "  ΔN(Δμ=0) = {:.6e}, dN/dμ = {:.6e}", DeltaN_0, dN_dmu);
      app_log(3, "  Computed Δμ = {:.6e}", Delta_mu);

      // Second pass: solve with computed Δμ
      solve_lr_dyson_impl(sDeltaG_tskij, sDeltaH0_skij,
                          sDeltaF_skij, sDeltaSigma_tskij, Delta_mu, sDeltaVcorr_skij);
      _Timer.start("LR_DYSON_DM");
      compute_lr_dm(sDeltaDm_skij, sDeltaG_tskij);
      _Timer.stop("LR_DYSON_DM");

      // Verify ΔN ≈ 0
      _Timer.start("LR_DYSON_NELEC");
      double DeltaN_final = compute_lr_Nelec(sDeltaDm_skij);
      _Timer.stop("LR_DYSON_NELEC");
      app_log(3, "  Final ΔN = {:.6e} (should be ~0)", DeltaN_final);
    }
  } else {
    // Standard mode: use provided Delta_mu
    if (fix_density && !_is_q_gamma) {
      app_log(3, "solve_lr_dyson: fix_density ignored for q≠0 (Δμ term vanishes)");
    }
    // For q≠0, Δμ is meaningless — force to zero to prevent silent pollution
    if (!_is_q_gamma) Delta_mu = 0.0;
    solve_lr_dyson_impl(sDeltaG_tskij, sDeltaH0_skij,
                        sDeltaF_skij, sDeltaSigma_tskij, Delta_mu, sDeltaVcorr_skij);
    _Timer.start("LR_DYSON_DM");
    compute_lr_dm(sDeltaDm_skij, sDeltaG_tskij);
    _Timer.stop("LR_DYSON_DM");
  }

  _Timer.stop("LR_DYSON");
  print_timers(3);  // per-step diagnostics only at verbosity >= 3

  return Delta_mu;
}


template<typename DeltaG_t, typename DeltaH0_t, typename DeltaF_t, typename DeltaSigma_t>
void lr_dyson::solve_lr_dyson_impl(
    DeltaG_t& sDeltaG_tskij,
    const DeltaH0_t& sDeltaH0_skij,
    const DeltaF_t& sDeltaF_skij,
    const DeltaSigma_t* sDeltaSigma_tskij,
    double Delta_mu,
    const DeltaF_t* sDeltaVcorr_skij) {

  using math::nda::make_distributed_array;
  using Array_5D_t = nda::array<ComplexType, 5>;

  // Δμ·S term is only meaningful at q=0; assert no accidental nonzero value at q≠0
  utils::check(_is_q_gamma || std::abs(Delta_mu) < 1e-15,
               "solve_lr_dyson_impl: Delta_mu = {:.6e} but q≠0. "
               "Delta_mu must be zero for q≠0 perturbations.", Delta_mu);

  app_log(3, "Solving LR Dyson equation:");
  if (sDeltaSigma_tskij) {
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

  // Transform ΔΣ from tau to frequency (only when GW is active)
  _Timer.start("LR_DYSON_TAU_TO_W");
  using dArray_5D_t = memory::darray_t<nda::array<ComplexType, 5>, mpi3::communicator>;
  std::optional<dArray_5D_t> opt_dDeltaSigma_wskij;
  if (sDeltaSigma_tskij) {
    // ΔΣ leakage diagnostics gated at verbosity >= 3. __app_output_level__ is
    // set only on the root, so need to broadcast chk_leak.
    int chk_leak = (__app_output_level__ >= 3) ? 1 : 0;
    _context->comm.broadcast_n(&chk_leak, 1, 0);
    opt_dDeltaSigma_wskij.emplace(
        distributed_tau_to_w(_context->comm, *sDeltaSigma_tskij, *_dyson.FT(), w_pgrid, w_bsize,
                             chk_leak != 0));
  }
  _Timer.stop("LR_DYSON_TAU_TO_W");

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
  utils::check(sDeltaSigma_tskij == nullptr or w_pgrid[3]*w_pgrid[4] == 1,
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
        if (sDeltaSigma_tskij) {
          auto DeltaSigma_w_loc = opt_dDeltaSigma_wskij->local();
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
  opt_dDeltaSigma_wskij.reset();
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

    _Timer.start("LR_DYSON_GATHER");
    math::nda::gather_to_shm(dDeltaG_tskij, sDeltaG_tskij);
    _Timer.stop("LR_DYSON_GATHER");
  }

  _Timer.start("LR_DYSON_MISC");
  _context->comm.barrier();
  _Timer.stop("LR_DYSON_MISC");
}


template<typename DeltaDm_t, typename DeltaG_t>
void lr_dyson::compute_lr_dm(DeltaDm_t& sDeltaDm_skij, const DeltaG_t& sDeltaG_tskij) {
  // ΔDm(k) = -ΔG(k, τ=β⁻)
  // Distribute (s, k) work over _context->comm; each rank writes its slab into
  // the shared array, then all_reduce across nodes.
  decltype(nda::range::all) all;
  sDeltaDm_skij.set_zero();  // ends with fence + node_sync

  int rank = _context->comm.rank();
  int size = _context->comm.size();
  nda::array<ComplexType, 3> DeltaG_buf(_nts, _nbnd, _nbnd);
  nda::array<ComplexType, 2> DeltaDm_buf(_nbnd, _nbnd);
  auto DeltaG_loc = sDeltaG_tskij.local();
  auto DeltaDm_loc = sDeltaDm_skij.local();
  for (int i = rank; i < _ns * _nkpts_ibz; i += size) {
    int is = i / _nkpts_ibz;
    int ik = i % _nkpts_ibz;
    DeltaG_buf = DeltaG_loc(all, is, ik, all, all);
    _dyson.FT()->tau_to_beta(DeltaG_buf, DeltaDm_buf);
    DeltaDm_loc(is, ik, all, all) = -DeltaDm_buf;
  }
  sDeltaDm_skij.win().fence();
  sDeltaDm_skij.all_reduce();
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

  if (_context->node_comm.root()) {
    nda::matrix<ComplexType> buffer(_nbnd, _nbnd);
    for (int is = 0; is < _ns; ++is) {
      for (int ik = 0; ik < _nkpts_ibz; ++ik) {
        auto S_k = S_loc(is, ik, nda::range::all, nda::range::all);
        auto DeltaDm_k = DeltaDm_loc(is, ik, nda::range::all, nda::range::all);

        // Tr[S · ΔDm] via matrix multiply and trace
        nda::blas::gemm(ComplexType(1.0), S_k, DeltaDm_k, ComplexType(0.0), buffer);
        DeltaN += k_weight(ik) * nda::trace(buffer);
      }
    }
    DeltaN *= spin_factor;
  }

  // Broadcast result
  _context->comm.broadcast_n(&DeltaN, 1, 0);

  if (std::abs(DeltaN.imag()) > 1e-10) {
    app_log(1, "[WARNING] compute_lr_Nelec: Im(ΔN) = {:.2e}", DeltaN.imag());
  }

  app_log(3, "compute_lr_Nelec: ΔN = {:.6e}", DeltaN.real());
  return DeltaN.real();
}


double lr_dyson::compute_dN_dmu() {
  // Return cached value if already computed
  if (_dN_dmu_cached) {
    app_log(2, "compute_dN_dmu: returning cached dN/dμ = {:.6e}", _cached_dN_dmu);
    return _cached_dN_dmu;
  }

  // dN/dμ = Tr[S · (G·S·G)(τ=β⁻)]
  // This is the response of particle number to chemical potential shift.
  //
  // NOTE: dN/dμ depends ONLY on the unperturbed Green's function G, NOT on any
  // LR quantities (ΔH0, ΔF, ΔΣ). It can be precomputed once and reused for all
  // perturbations at the same reference point.
  //
  // The LR Dyson equation in FREQUENCY space is:
  //   ΔG(k,iω) = G(k,iω) · [ΔH0 - Δμ·S] · G(k,iω)
  //
  // So the Δμ response is computed in frequency space:
  //   δΔG(k,iω)/δΔμ = -G(k,iω) · S · G(k,iω)
  //
  // To get the density matrix response:
  //   (G·S·G)(τ) = IFFT[G(iω) · S · G(iω)]
  //   δΔDm/δΔμ = (G·S·G)(β⁻)  (positive contribution)
  //
  // IMPORTANT: Must compute G·S·G in frequency space, then transform to tau!
  // Computing G(τ)·S·G(τ) directly is WRONG (not equivalent).
  //
  // Uses cached G(iω) in shared memory.

  utils::check(_is_q_gamma,
               "compute_dN_dmu: dN/dμ is not meaningful for q≠0. "
               "This function should only be called for q=0 perturbations.");
  utils::check(_cached_G_wskij != nullptr,
               "compute_dN_dmu: cached G(iω) not set. Call set_cached_G_omega() first.");

  auto k_weight = _dyson.MF()->k_weight();
  auto S_loc = _dyson.sS_skij().local();
  double spin_factor = (_ns == 1 && _dyson.MF()->npol() == 1) ? 2.0 : 1.0;

  // Distribute (s, k) work over _context->comm. Output is a scalar, so each
  // rank computes its partial sum and we all_reduce the scalar at the end.
  decltype(nda::range::all) all;
  auto G_w = _cached_G_wskij->local();  // (nw, ns, nk, nb, nb), shared memory

  ComplexType dN_dmu(0.0);

  int rank = _context->comm.rank();
  int size = _context->comm.size();
  nda::array<ComplexType, 3> GSG_w_buf(_nw, _nbnd, _nbnd);
  nda::array<ComplexType, 3> GSG_t_buf(_nts, _nbnd, _nbnd);
  nda::matrix<ComplexType> GSG_beta_buf(_nbnd, _nbnd);
  nda::matrix<ComplexType> tmp(_nbnd, _nbnd);
  nda::matrix<ComplexType> buffer(_nbnd, _nbnd);

  for (int i = rank; i < _ns * _nkpts_ibz; i += size) {
    int is = i / _nkpts_ibz;
    int ik = i % _nkpts_ibz;
    auto S_k = S_loc(is, ik, all, all);

    // GSG(iω) = G(iω) · S · G(iω) for all iω at this (is, ik)
    for (int n = 0; n < _nw; ++n) {
      auto G_w_k = G_w(n, is, ik, all, all);
      auto GSG_w_k = GSG_w_buf(n, all, all);
      nda::blas::gemm(ComplexType(1.0), S_k, G_w_k, ComplexType(0.0), tmp);
      nda::blas::gemm(ComplexType(1.0), G_w_k, tmp, ComplexType(0.0), GSG_w_k);
    }

    // FT: GSG(iω) → GSG(τ) → GSG(β⁻)
    _dyson.FT()->w_to_tau(GSG_w_buf, GSG_t_buf, imag_axes_ft::fermion);
    _dyson.FT()->tau_to_beta(GSG_t_buf, GSG_beta_buf);

    // Tr[S · GSG(β⁻)]
    nda::blas::gemm(ComplexType(1.0), S_k, GSG_beta_buf, ComplexType(0.0), buffer);
    dN_dmu += k_weight(ik) * nda::trace(buffer);
  }
  dN_dmu *= spin_factor;

  // Sum partial contributions across ranks
  dN_dmu = _context->comm.all_reduce_value(dN_dmu);

  _cached_dN_dmu = dN_dmu.real();
  _dN_dmu_cached = true;
  app_log(2, "compute_dN_dmu: dN/dμ = {:.6e} (cached)", _cached_dN_dmu);
  return _cached_dN_dmu;
}


template<typename DeltaDm_t>
double lr_dyson::compute_Delta_mu(const DeltaDm_t& sDeltaDm_skij) {
  utils::check(_is_q_gamma,
               "compute_Delta_mu: Δμ adjustment is not meaningful for q≠0. "
               "This function should only be called for q=0 perturbations.");

  // Compute ΔN at Δμ=0
  double DeltaN_0 = compute_lr_Nelec(sDeltaDm_skij);

  // Compute dN/dμ (uses cached G(iω))
  double dN_dmu = compute_dN_dmu();

  if (std::abs(dN_dmu) < 1e-15) {
    app_log(1, "[WARNING] compute_Delta_mu: dN/dμ ≈ 0, cannot compute Δμ. Returning 0.");
    return 0.0;
  }

  // Closed-form solution: Δμ = -ΔN_0 / (dN/dμ)
  double Delta_mu = -DeltaN_0 / dN_dmu;

  app_log(2, "compute_Delta_mu: ΔN_0 = {:.6e}, dN/dμ = {:.6e}, Δμ = {:.6e}",
          DeltaN_0, dN_dmu, Delta_mu);

  return Delta_mu;
}


// Template instantiations
template double lr_dyson::solve_lr_dyson(
    sArray_t<Array_view_5D_t>&,
    sArray_t<Array_view_4D_t>&,
    const sArray_t<Array_view_4D_t>&,
    const sArray_t<Array_view_4D_t>&,
    const sArray_t<Array_view_5D_t>*,
    bool,
    double,
    const sArray_t<Array_view_4D_t>*);

template void lr_dyson::solve_lr_dyson_impl(
    sArray_t<Array_view_5D_t>&,
    const sArray_t<Array_view_4D_t>&,
    const sArray_t<Array_view_4D_t>&,
    const sArray_t<Array_view_5D_t>*,
    double,
    const sArray_t<Array_view_4D_t>*);

template void lr_dyson::compute_lr_dm(
    sArray_t<Array_view_4D_t>&,
    const sArray_t<Array_view_5D_t>&);

template double lr_dyson::compute_lr_Nelec(
    const sArray_t<Array_view_4D_t>&);

template double lr_dyson::compute_Delta_mu(
    const sArray_t<Array_view_4D_t>&);

} // namespace methods
