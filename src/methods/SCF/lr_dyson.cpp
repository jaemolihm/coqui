/**
 * ==========================================================================
 * CoQuí: Correlated Quantum ínterface
 *
 * Copyright (c) 2022-2025 Simons Foundation & The CoQuí developer team
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

  for (auto& v : {"LR_DYSON", "G_TAU_TO_W", "LR_DYSON_LOOP", "LR_DYSON_GATHER"}) {
    _Timer.add(v);
  }
  _context->comm.barrier();
}


template<typename DeltaG_t, typename DeltaDm_t, typename G_t, typename DeltaH0_t,
         typename DeltaF_t, typename DeltaSigma_t>
double lr_dyson::solve_lr_dyson(
    DeltaG_t& sDeltaG_tskij,
    DeltaDm_t& sDeltaDm_skij,
    const G_t& sG_tskij,
    const DeltaH0_t& sDeltaH0_skij,
    const DeltaF_t& sDeltaF_skij,
    const DeltaSigma_t& sDeltaSigma_tskij,
    bool fix_density,
    double Delta_mu) {

  _Timer.start("LR_DYSON");

  // For fix_density mode at q=0, we need to compute Δμ
  // The LR Dyson equation is linear in Δμ, so we solve twice:
  // 1. ΔG(0) with Δμ=0 to get ΔN(0)
  // 2. Use dN/dμ to compute Δμ = -ΔN(0) / (dN/dμ)
  // 3. ΔG_final with the computed Δμ
  if (fix_density && _is_q_gamma) {
    app_log(2, "solve_lr_dyson: fix_density mode (computing Δμ to enforce ΔN=0)");

    // First pass: solve with Δμ=0
    solve_lr_dyson_impl(sDeltaG_tskij, sG_tskij, sDeltaH0_skij,
                        sDeltaF_skij, sDeltaSigma_tskij, 0.0);
    compute_lr_dm(sDeltaDm_skij, sDeltaG_tskij);

    // Compute ΔN at Δμ=0
    double DeltaN_0 = compute_lr_Nelec(sDeltaDm_skij);

    // Compute dN/dμ
    double dN_dmu = compute_dN_dmu(sG_tskij);

    if (std::abs(dN_dmu) < 1e-15) {
      app_log(1, "[WARNING] solve_lr_dyson: dN/dμ ≈ 0, cannot compute Δμ. Using Δμ=0.");
      Delta_mu = 0.0;
    } else {
      // Closed-form solution: Δμ = -ΔN(0) / (dN/dμ)
      // (The LR Dyson equation is linear in Δμ, so this is exact)
      Delta_mu = -DeltaN_0 / dN_dmu;
      app_log(2, "  ΔN(Δμ=0) = {:.6e}, dN/dμ = {:.6e}", DeltaN_0, dN_dmu);
      app_log(2, "  Computed Δμ = {:.6e}", Delta_mu);

      // Second pass: solve with computed Δμ
      solve_lr_dyson_impl(sDeltaG_tskij, sG_tskij, sDeltaH0_skij,
                          sDeltaF_skij, sDeltaSigma_tskij, Delta_mu);
      compute_lr_dm(sDeltaDm_skij, sDeltaG_tskij);

      // Verify ΔN ≈ 0
      double DeltaN_final = compute_lr_Nelec(sDeltaDm_skij);
      app_log(2, "  Final ΔN = {:.6e} (should be ~0)", DeltaN_final);
    }
  } else {
    // Standard mode: use provided Delta_mu
    if (fix_density && !_is_q_gamma) {
      app_log(2, "solve_lr_dyson: fix_density ignored for q≠0 (Δμ term vanishes)");
    }
    // For q≠0, Δμ is meaningless — force to zero to prevent silent pollution
    if (!_is_q_gamma) Delta_mu = 0.0;
    solve_lr_dyson_impl(sDeltaG_tskij, sG_tskij, sDeltaH0_skij,
                        sDeltaF_skij, sDeltaSigma_tskij, Delta_mu);
    compute_lr_dm(sDeltaDm_skij, sDeltaG_tskij);
  }

  _Timer.stop("LR_DYSON");
  print_timers();

  return Delta_mu;
}


template<typename DeltaG_t, typename G_t, typename DeltaH0_t, typename DeltaF_t, typename DeltaSigma_t>
void lr_dyson::solve_lr_dyson_impl(
    DeltaG_t& sDeltaG_tskij,
    const G_t& sG_tskij,
    const DeltaH0_t& sDeltaH0_skij,
    const DeltaF_t& sDeltaF_skij,
    const DeltaSigma_t& sDeltaSigma_tskij,
    double Delta_mu) {

  using math::nda::make_distributed_array;
  using Array_5D_t = nda::array<ComplexType, 5>;

  // Δμ·S term is only meaningful at q=0; assert no accidental nonzero value at q≠0
  utils::check(_is_q_gamma || std::abs(Delta_mu) < 1e-15,
               "solve_lr_dyson_impl: Delta_mu = {:.6e} but q≠0. "
               "Delta_mu must be zero for q≠0 perturbations.", Delta_mu);

  app_log(2, "Solving LR Dyson equation:");
  app_log(2, "  ΔG(k,iω) = G(k+q,iω) · [ΔH0(k) + ΔF(k) + ΔΣ(k,iω) - Δμ·S(k+q,k)] · G(k,iω)");
  app_log(2, "  Δμ = {:.6e}", Delta_mu);

  // Processor grid
  std::array<long, 5> w_pgrid;
  std::array<long, 5> w_bsize;
  {
    int np = _context->comm.size();
    int nwpools = utils::find_proc_grid_max_npools(np, _nw, 0.4);
    np /= nwpools;
    int nkpools = utils::find_proc_grid_max_npools(np, _nkpts_ibz, 0.4);
    np /= nkpools;
    int np_i = utils::find_proc_grid_min_diff(np, 1, 1);
    int np_j = np / np_i;

    w_pgrid = {nwpools, 1, nkpools, np_i, np_j};
    long ibsize = std::min({1024L, (long)_nbnd / np_i, (long)_nbnd / np_j});
    if (ibsize < 1) ibsize = 1;
    w_bsize = {1, 1, 1, ibsize, ibsize};

    utils::check(nwpools * nkpools * np_i * np_j == _context->comm.size(),
                 "lr_dyson: pgrid mismatches!");
  }

  // Transform G and ΔΣ from tau to frequency
  _Timer.start("G_TAU_TO_W");
  auto dG_wskij = distributed_tau_to_w(_context->comm, sG_tskij, *_dyson.FT(), w_pgrid, w_bsize);
  auto dDeltaSigma_wskij = distributed_tau_to_w(_context->comm, sDeltaSigma_tskij,
                                                 *_dyson.FT(), w_pgrid, w_bsize);
  _Timer.stop("G_TAU_TO_W");

  auto dDeltaG_wskij = make_distributed_array<Array_5D_t>(_context->comm, w_pgrid,
                                                          {_nw, _ns, _nkpts_ibz, _nbnd, _nbnd}, w_bsize);

  auto [nw_loc, ns_loc, nk_loc, ni_loc, nj_loc] = dG_wskij.local_shape();
  auto [w_org, s_org, k_org, i_org, j_org] = dG_wskij.origin();

  auto G_w_loc = dG_wskij.local();
  auto DeltaSigma_w_loc = dDeltaSigma_wskij.local();
  auto DeltaG_w_loc = dDeltaG_wskij.local();
  auto DeltaH0_loc = sDeltaH0_skij.local();
  auto DeltaF_loc = sDeltaF_skij.local();
  auto S_loc = _dyson.sS_skij().local();

  // Temporary matrices
  nda::matrix<ComplexType> X_ij(_nbnd, _nbnd);
  nda::matrix<ComplexType> tmp(_nbnd, _nbnd);
  nda::matrix<ComplexType> DeltaG_ij(_nbnd, _nbnd);

  // LR Dyson loop
  _Timer.start("LR_DYSON_LOOP");
  for (long n = 0; n < nw_loc; ++n) {
    for (long s = 0; s < ns_loc; ++s) {
      for (long k = 0; k < nk_loc; ++k) {
        long ik = k + k_org;
        long is = s + s_org;
        long ikq = _kpq_map(ik);

        auto G_k = G_w_loc(n, s, k, nda::range::all, nda::range::all);
        auto DeltaH0_k = DeltaH0_loc(is, ik, nda::range::all, nda::range::all);
        auto DeltaF_k = DeltaF_loc(is, ik, nda::range::all, nda::range::all);
        auto DeltaSigma_k = DeltaSigma_w_loc(n, s, k, nda::range::all, nda::range::all);

        // Build X = ΔH0 + ΔF + ΔΣ - Δμ·S (Δμ is nonzero only for q=0)
        auto S_k = S_loc(is, ik, nda::range::all, nda::range::all);
        X_ij = DeltaH0_k + DeltaF_k + DeltaSigma_k - Delta_mu * S_k;

        if (ikq >= k_org && ikq < k_org + nk_loc) {
          auto G_kq = G_w_loc(n, s, ikq - k_org, nda::range::all, nda::range::all);

          // ΔG = G_{k+q} @ X @ G_k
          nda::blas::gemm(ComplexType(1.0), X_ij, G_k, ComplexType(0.0), tmp);
          nda::blas::gemm(ComplexType(1.0), G_kq, tmp, ComplexType(0.0), DeltaG_ij);

          DeltaG_w_loc(n, s, k, nda::range::all, nda::range::all) = DeltaG_ij;
        } else {
          utils::check(false,
                       "lr_dyson: k+q ({}) not in local range. "
                       "Inter-processor communication not yet implemented.",
                       ikq);
        }
      }
    }
  }
  _Timer.stop("LR_DYSON_LOOP");

  dG_wskij.reset();
  dDeltaSigma_wskij.reset();
  _context->comm.barrier();

  // Convert ΔG(w) -> ΔG(tau)
  {
    int np = _context->comm.size();
    long nkpools = utils::find_proc_grid_max_npools(np, _nkpts_ibz, 0.2);
    np /= nkpools;
    long np_i = utils::find_proc_grid_min_diff(np, 1, 1);
    long np_j = np / np_i;

    auto dDeltaG_wskij_tmp = make_distributed_array<Array_5D_t>(_context->comm,
        {1, 1, nkpools, np_i, np_j}, {_nw, _ns, _nkpts_ibz, _nbnd, _nbnd});
    math::nda::redistribute(dDeltaG_wskij, dDeltaG_wskij_tmp);
    dDeltaG_wskij.reset();

    auto dDeltaG_tskij = make_distributed_array<Array_5D_t>(_context->comm,
        {1, 1, nkpools, np_i, np_j}, {_nts, _ns, _nkpts_ibz, _nbnd, _nbnd});
    auto DeltaGt_loc = dDeltaG_tskij.local();
    auto DeltaGw_loc = dDeltaG_wskij_tmp.local();
    _dyson.FT()->w_to_tau(DeltaGw_loc, DeltaGt_loc, imag_axes_ft::fermi);
    dDeltaG_wskij_tmp.reset();

    _Timer.start("LR_DYSON_GATHER");
    math::nda::gather_to_shm(dDeltaG_tskij, sDeltaG_tskij);
    _Timer.stop("LR_DYSON_GATHER");
  }

  _context->comm.barrier();
}


template<typename DeltaDm_t, typename DeltaG_t>
void lr_dyson::compute_lr_dm(DeltaDm_t& sDeltaDm_skij, const DeltaG_t& sDeltaG_tskij) {
  // ΔDm(k) = -ΔG(k, τ=β⁻)
  if (_context->node_comm.root()) {
    auto DeltaDm = sDeltaDm_skij.local();
    _dyson.FT()->tau_to_beta(sDeltaG_tskij.local(), DeltaDm);
    DeltaDm *= -1;
  }
  _context->comm.barrier();
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

  app_log(2, "compute_lr_Nelec: ΔN = {:.6e}", DeltaN.real());
  return DeltaN.real();
}


template<typename G_t>
double lr_dyson::compute_dN_dmu(const G_t& sG_tskij) {
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

  utils::check(_is_q_gamma,
               "compute_dN_dmu: dN/dμ is not meaningful for q≠0. "
               "This function should only be called for q=0 perturbations.");

  using math::nda::make_distributed_array;
  using Array_5D_t = nda::array<ComplexType, 5>;

  auto k_weight = _dyson.MF()->k_weight();
  auto S_loc = _dyson.sS_skij().local();

  double spin_factor = (_ns == 1 && _dyson.MF()->npol() == 1) ? 2.0 : 1.0;

  // Set up processor grid for frequency-space computation
  std::array<long, 5> w_pgrid;
  std::array<long, 5> w_bsize;
  {
    int np = _context->comm.size();
    int nwpools = utils::find_proc_grid_max_npools(np, _nw, 0.4);
    np /= nwpools;
    int nkpools = utils::find_proc_grid_max_npools(np, _nkpts_ibz, 0.4);
    np /= nkpools;
    int np_i = utils::find_proc_grid_min_diff(np, 1, 1);
    int np_j = np / np_i;

    w_pgrid = {nwpools, 1, nkpools, np_i, np_j};
    long ibsize = std::min({1024L, (long)_nbnd / np_i, (long)_nbnd / np_j});
    if (ibsize < 1) ibsize = 1;
    w_bsize = {1, 1, 1, ibsize, ibsize};
  }

  // Transform G(τ) -> G(iω)
  auto dG_wskij = distributed_tau_to_w(_context->comm, sG_tskij, *_dyson.FT(), w_pgrid, w_bsize);

  // Compute G(iω) · S · G(iω) in frequency space
  auto dGSG_wskij = make_distributed_array<Array_5D_t>(_context->comm, w_pgrid,
                                                       {_nw, _ns, _nkpts_ibz, _nbnd, _nbnd}, w_bsize);

  auto [nw_loc, ns_loc, nk_loc, ni_loc, nj_loc] = dG_wskij.local_shape();
  auto [w_org, s_org, k_org, i_org, j_org] = dG_wskij.origin();

  auto G_w_loc = dG_wskij.local();
  auto GSG_w_loc = dGSG_wskij.local();

  nda::matrix<ComplexType> tmp(_nbnd, _nbnd);
  nda::matrix<ComplexType> GSG_w_k(_nbnd, _nbnd);

  for (long n = 0; n < nw_loc; ++n) {
    for (long s = 0; s < ns_loc; ++s) {
      for (long k = 0; k < nk_loc; ++k) {
        long ik = k + k_org;
        long is = s + s_org;

        auto G_w_k = G_w_loc(n, s, k, nda::range::all, nda::range::all);
        auto S_k = S_loc(is, ik, nda::range::all, nda::range::all);

        // GSG(iω) = G(iω) · S · G(iω)
        nda::blas::gemm(ComplexType(1.0), S_k, G_w_k, ComplexType(0.0), tmp);
        nda::blas::gemm(ComplexType(1.0), G_w_k, tmp, ComplexType(0.0), GSG_w_k);

        GSG_w_loc(n, s, k, nda::range::all, nda::range::all) = GSG_w_k;
      }
    }
  }

  dG_wskij.reset();
  _context->comm.barrier();

  // Transform GSG(iω) -> GSG(τ)
  // Redistribute for tau transform
  {
    int np = _context->comm.size();
    long nkpools = utils::find_proc_grid_max_npools(np, _nkpts_ibz, 0.2);
    np /= nkpools;
    long np_i = utils::find_proc_grid_min_diff(np, 1, 1);
    long np_j = np / np_i;

    auto dGSG_wskij_tmp = make_distributed_array<Array_5D_t>(_context->comm,
        {1, 1, nkpools, np_i, np_j}, {_nw, _ns, _nkpts_ibz, _nbnd, _nbnd});
    math::nda::redistribute(dGSG_wskij, dGSG_wskij_tmp);
    dGSG_wskij.reset();

    auto dGSG_tskij = make_distributed_array<Array_5D_t>(_context->comm,
        {1, 1, nkpools, np_i, np_j}, {_nts, _ns, _nkpts_ibz, _nbnd, _nbnd});
    auto GSGt_loc = dGSG_tskij.local();
    auto GSGw_loc = dGSG_wskij_tmp.local();
    _dyson.FT()->w_to_tau(GSGw_loc, GSGt_loc, imag_axes_ft::fermi);
    dGSG_wskij_tmp.reset();

    // Gather to shared memory for final computation
    auto sGSG_tskij = math::shm::make_shared_array<Array_view_5D_t>(
        _context->comm, _context->internode_comm, _context->node_comm,
        {_nts, _ns, _nkpts_ibz, _nbnd, _nbnd});
    math::nda::gather_to_shm(dGSG_tskij, sGSG_tskij);

    // Interpolate GSG to τ=β⁻ and compute trace
    ComplexType dN_dmu(0.0);

    if (_context->node_comm.root()) {
      nda::array<ComplexType, 4> GSG_beta_skij(_ns, _nkpts_ibz, _nbnd, _nbnd);
      _dyson.FT()->tau_to_beta(sGSG_tskij.local(), GSG_beta_skij);

      nda::matrix<ComplexType> buffer(_nbnd, _nbnd);
      for (int is = 0; is < _ns; ++is) {
        for (int ik = 0; ik < _nkpts_ibz; ++ik) {
          auto S_k = S_loc(is, ik, nda::range::all, nda::range::all);
          auto GSG_beta_k = GSG_beta_skij(is, ik, nda::range::all, nda::range::all);

          // Tr[S · GSG(β⁻)]
          nda::blas::gemm(ComplexType(1.0), S_k, GSG_beta_k, ComplexType(0.0), buffer);
          dN_dmu += k_weight(ik) * nda::trace(buffer);
        }
      }
      dN_dmu *= spin_factor;
    }

    // Broadcast result
    _context->comm.broadcast_n(&dN_dmu, 1, 0);

    app_log(2, "compute_dN_dmu: dN/dμ = {:.6e}", dN_dmu.real());
    return dN_dmu.real();
  }
}


template<typename DeltaDm_t, typename G_t>
double lr_dyson::compute_Delta_mu(const DeltaDm_t& sDeltaDm_skij,
                                   const G_t& sG_tskij) {
  utils::check(_is_q_gamma,
               "compute_Delta_mu: Δμ adjustment is not meaningful for q≠0. "
               "This function should only be called for q=0 perturbations.");

  // Compute ΔN at Δμ=0
  double DeltaN_0 = compute_lr_Nelec(sDeltaDm_skij);

  // Compute dN/dμ
  double dN_dmu = compute_dN_dmu(sG_tskij);

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
    const sArray_t<Array_view_5D_t>&,
    const sArray_t<Array_view_4D_t>&,
    const sArray_t<Array_view_4D_t>&,
    const sArray_t<Array_view_5D_t>&,
    bool,
    double);

template void lr_dyson::solve_lr_dyson_impl(
    sArray_t<Array_view_5D_t>&,
    const sArray_t<Array_view_5D_t>&,
    const sArray_t<Array_view_4D_t>&,
    const sArray_t<Array_view_4D_t>&,
    const sArray_t<Array_view_5D_t>&,
    double);

template void lr_dyson::compute_lr_dm(
    sArray_t<Array_view_4D_t>&,
    const sArray_t<Array_view_5D_t>&);

template double lr_dyson::compute_lr_Nelec(
    const sArray_t<Array_view_4D_t>&);

template double lr_dyson::compute_dN_dmu(
    const sArray_t<Array_view_5D_t>&);

template double lr_dyson::compute_Delta_mu(
    const sArray_t<Array_view_4D_t>&,
    const sArray_t<Array_view_5D_t>&);

} // namespace methods
