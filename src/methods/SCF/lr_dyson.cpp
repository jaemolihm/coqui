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


template<typename DeltaG_t, typename G_t, typename DeltaH0_t>
void lr_dyson::solve_lr_dyson_fixed_sigma(
    DeltaG_t& sDeltaG_tskij,
    const G_t& sG_tskij,
    const DeltaH0_t& sDeltaH0_skij) {

  // Wrapper: create zero ΔF and ΔΣ arrays, then call solve_lr_dyson with Δμ=0
  app_log(2, "solve_lr_dyson_fixed_sigma: delegating to solve_lr_dyson with zero ΔF, ΔΣ, Δμ");

  // Create zero arrays for ΔF (4D) and ΔΣ (5D)
  auto sDeltaF_skij = math::shm::make_shared_array<Array_view_4D_t>(
      _context->comm, _context->internode_comm, _context->node_comm,
      {_ns, _nkpts_ibz, _nbnd, _nbnd});
  sDeltaF_skij.local() = ComplexType(0.0);

  auto sDeltaSigma_tskij = math::shm::make_shared_array<Array_view_5D_t>(
      _context->comm, _context->internode_comm, _context->node_comm,
      {_nts, _ns, _nkpts_ibz, _nbnd, _nbnd});
  sDeltaSigma_tskij.local() = ComplexType(0.0);

  _context->comm.barrier();

  // Call the full solver with Δμ=0
  solve_lr_dyson(sDeltaG_tskij, sG_tskij, sDeltaH0_skij,
                 sDeltaF_skij, sDeltaSigma_tskij, 0.0);
}


template<typename DeltaG_t, typename G_t, typename DeltaH0_t, typename DeltaF_t, typename DeltaSigma_t>
void lr_dyson::solve_lr_dyson(
    DeltaG_t& sDeltaG_tskij,
    const G_t& sG_tskij,
    const DeltaH0_t& sDeltaH0_skij,
    const DeltaF_t& sDeltaF_skij,
    const DeltaSigma_t& sDeltaSigma_tskij,
    double Delta_mu) {

  _Timer.start("LR_DYSON");
  using math::nda::make_distributed_array;
  using Array_5D_t = nda::array<ComplexType, 5>;

  app_log(2, "Solving LR Dyson equation (full):");
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
  _Timer.stop("LR_DYSON");
  print_timers();
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


// Template instantiations
template void lr_dyson::solve_lr_dyson_fixed_sigma(
    sArray_t<Array_view_5D_t>&,
    const sArray_t<Array_view_5D_t>&,
    const sArray_t<Array_view_4D_t>&);

template void lr_dyson::solve_lr_dyson(
    sArray_t<Array_view_5D_t>&,
    const sArray_t<Array_view_5D_t>&,
    const sArray_t<Array_view_4D_t>&,
    const sArray_t<Array_view_4D_t>&,
    const sArray_t<Array_view_5D_t>&,
    double);

template void lr_dyson::compute_lr_dm(
    sArray_t<Array_view_4D_t>&,
    const sArray_t<Array_view_5D_t>&);

} // namespace methods
