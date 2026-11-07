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


#include <algorithm>
#include <cmath>
#include <cstdlib>

#include "methods/SCF/simple_dyson.h"
#include "nda/h5.hpp"
#include "nda/linalg.hpp"
#include "numerics/shared_array/nda.hpp"
#include "nda/nda.hpp"
#include "numerics/gemm_guard.hpp"

namespace methods {

namespace {
  // Detect an exactly-zero (static) self-energy, e.g. HF. One scan per node root
  // (Sigma lives in node-shared memory), early-exiting at the first nonzero element
  // (dynamic Sigma, e.g. GW); min-reduced so every rank agrees.
  template<typename Context_t, typename Sigma_view>
  bool sigma_is_zero(Context_t &context, const Sigma_view &Sigma) {
    int is_zero = 1;
    if (context.node_comm.root())
      is_zero = std::none_of(Sigma.data(), Sigma.data() + Sigma.size(),
                             [](const ComplexType &x) { return x != ComplexType(0); }) ? 1 : 0;
    context.comm.all_reduce_in_place_n(&is_zero, 1, mpi3::min<>{});
    return is_zero == 1;
  }
}

  template<typename G_t, typename F_t, typename Sigma_t>
  void simple_dyson::solve_dyson_tau(G_t &_G_shm, const F_t &_sF_skij, const Sigma_t &_Sigma_shm,
                                     double mu, int nsweep) {
    _Timer.start("DYSON");
    using math::shm::make_shared_array;
    decltype(nda::range::all) all;
    const long nt = _nts, nb = _nbnd;
    const double beta = _FT->beta();
    nda::array<double, 1> tau(nt);
    {
      auto tr = _FT->tau_mesh();
      for (long t = 0; t < nt; ++t) tau(t) = (tr(t) + 1.0) * beta / 2.0;
    }
    auto S = _sS_skij.local(); auto H0 = _sH0_skij.local(); auto F = _sF_skij.local();
    auto Sig = _Sigma_shm.local(); auto Gold = _G_shm.local();
    // tasks = (s, k, column block of G); the columns stay in their block through every sweep,
    // so the n sweeps need no communication
    long nsk = long(_ns) * _nkpts_ibz;
    long nblk = std::max(1L, std::min(long(nb), (long(_context->comm.size()) + nsk - 1) / nsk));
    long bj = (nb + nblk - 1) / nblk;
    auto sGnew = make_shared_array<Array_view_5D_t>(*_context, {nt, long(_ns), long(_nkpts_ibz), nb, nb});
    sGnew.set_zero();
    double dmax = 0.0, smax = 0.0;
    static const bool use_ref = [] { const char* e = std::getenv("COQUI_DYSON_TAU_REF"); return !(e and std::string(e) == "0"); }();
    long n_lowest = -1;
    {
      auto wnm = _FT->wn_mesh();
      for (long i = 0; i < wnm.size(); ++i) if (wnm(i) == 1) n_lowest = i;
      utils::check(n_lowest >= 0, "solve_dyson_tau: lowest positive fermionic Matsubara node (n = 1) not found");
    }
    for (long task = _context->comm.rank(); task < nsk * nblk; task += _context->comm.size()) {
      long sk = task / nblk, blk = task % nblk;
      long is = sk / _nkpts_ibz, ik = sk % _nkpts_ibz;
      long j0 = blk * bj, j1 = std::min(nb, j0 + bj);
      if (j0 >= j1) continue;
      auto J = nda::range(j0, j1);
      nda::matrix<ComplexType> Sk = S(is, ik, all, all);
      for (long i = 0; i < nb; ++i) Sk(i, i) -= 1.0;
      smax = std::max(smax, nda::max_element(nda::abs(Sk)));
      // Static reference Sigma_ref = Hermitian part of Sigma at the lowest positive Matsubara node: the update is
      // G = G0' + G0' * (Sigma - Sigma_ref delta) * G with G0' = [(iw+mu) - H0 - F - Sigma_ref]^-1, which has the
      // same fixed point and a much smaller contraction factor (CdO: rho 0.93 -> 0.16 at the lowest frequency).
      nda::array<ComplexType, 3> SigK = Sig(all, is, ik, all, all);
      nda::matrix<ComplexType> Sref(nb, nb);
      Sref() = 0.0;
      if (use_ref) {
        nda::matrix<ComplexType> Sw(nb, nb);
        _FT->tau_to_w(SigK, Sw, imag_axes_ft::fermion, n_lowest);
        Sref = 0.5 * (Sw + nda::dagger(Sw));
      }
      // G0'(tau) = U g(tau, e - mu) U^H with H0 + F + Sigma_ref = U e U^H (orthonormal basis)
      nda::matrix<ComplexType> Hk = H0(is, ik, all, all) + F(is, ik, all, all) + Sref;
      auto [ev, U] = nda::linalg::eigenelements(Hk);
      nda::array<ComplexType, 3> G0(nt, nb, nb);
      nda::matrix<ComplexType> Ug(nb, nb);
      for (long t = 0; t < nt; ++t) {
        for (long a = 0; a < nb; ++a) {
          double e = ev(a) - mu, g;
          if (e >= 0) g = -std::exp(-tau(t) * e) / (1.0 + std::exp(-beta * e));
          else        g = -std::exp((beta - tau(t)) * e) / (1.0 + std::exp(beta * e));
          for (long b = 0; b < nb; ++b) Ug(b, a) = U(b, a) * g;
        }
        G0(t, all, all) = Ug * nda::dagger(U);
      }
      auto cS = _FT->dlr_it_vals2coefs(SigK);
      auto cG0 = _FT->dlr_it_vals2coefs(G0);
      nda::array<ComplexType, 3> cG0R(nt, nb, nb);                    // coefficients of G0'(tau) Sigma_ref
      for (long l = 0; l < nt; ++l) cG0R(l, all, all) = nda::matrix<ComplexType>(cG0(l, all, all)) * Sref;
      nda::array<ComplexType, 3> Gp = Gold(all, is, ik, all, J);
      nda::array<ComplexType, 3> Gn(nt, nb, j1 - j0);
      for (int sw = 0; sw < nsweep; ++sw) {
        auto cGp = _FT->dlr_it_vals2coefs(Gp);
        auto X = _FT->dlr_it_convolve(cS, cGp);          // (Sigma * G)(tau)
        auto cX = _FT->dlr_it_vals2coefs(X);
        Gn = G0(all, all, J) + _FT->dlr_it_convolve(cG0, cX);
        if (use_ref) Gn -= _FT->dlr_it_convolve(cG0R, cGp);
        if (sw == nsweep - 1) dmax = std::max(dmax, nda::max_element(nda::abs(Gn - Gp)));
        Gp = Gn;
      }
      sGnew.local()(all, is, ik, all, J) = Gn;
    }
    sGnew.all_reduce();
    _context->comm.all_reduce_in_place_n(&dmax, 1, mpi3::max<>{});
    _context->comm.all_reduce_in_place_n(&smax, 1, mpi3::max<>{});
    utils::check(smax < 1e-8, "solve_dyson_tau: needs an orthonormal basis (max |S - 1| = {})", smax);
    if (_context->node_comm.root()) _G_shm.local() = sGnew.local();
    _context->comm.barrier();
    app_log(1, "DYSONTAU sweeps= {} ref= {} last_sweep_change= {:.3e}", nsweep, use_ref ? 1 : 0, dmax);
    _Timer.stop("DYSON");
  }

  template<typename G_t, typename F_t, typename Sigma_t>
  void simple_dyson::solve_dyson(G_t &_G_shm,
                                 const F_t &_sF_skij, const Sigma_t &_Sigma_shm, double mu) {
    {
      static const int nsw = [] { const char* e = std::getenv("COQUI_DYSON_TAU_SWEEPS"); return e ? std::atoi(e) : 0; }();
      if (nsw > 0 and _tau_ready and not sigma_is_zero(*_context, _Sigma_shm.local())) {
        solve_dyson_tau(_G_shm, _sF_skij, _Sigma_shm, mu, nsw);
        return;
      }
      _tau_ready = true;
    }
    _Timer.start("DYSON");
    using math::nda::make_distributed_array;
    using Array_5D_t = nda::array<ComplexType, 5>;

    // processor grid for Dyson equation
    std::array<long, 5> w_pgrid;
    std::array<long, 5> w_bsize;
    {
      std::tie(w_pgrid, w_bsize) =
          dyson_omega_proc_grid(_context->comm.size(), _nw, _nkpts_ibz, _nbnd);

      utils::check(w_pgrid[0]*w_pgrid[2]*w_pgrid[3]*w_pgrid[4] == _context->comm.size(),
                   "solve_dyson: pgrid mismatches!");

      app_log(2, "Dyson equation for Green's function:");
      app_log(2, "  - processor grid for G/Self-energy: (w, k, i, j) = ({}, {}, {}, {})",
              w_pgrid[0], w_pgrid[2], w_pgrid[3], w_pgrid[4]);
      app_log(2, "  - block size: (w, k, i, j) = ({}, {}, {}, {})",
              w_bsize[0], w_bsize[2], w_bsize[3], w_bsize[4]);
    }

    _Timer.start("SIGMA_TAU_TO_W");
    // Static case (Sigma == 0, e.g. HF): the imag-axis FT of an exactly-zero Sigma is
    // exactly zero, so a zero-initialized array replaces the tau->w transform.
    auto dSigma_wskij = sigma_is_zero(*_context, _Sigma_shm.local())
        ? make_distributed_array<Array_5D_t>(_context->comm, w_pgrid,
                                             {_nw, _ns, _nkpts_ibz, _nbnd, _nbnd}, w_bsize)
        : distributed_tau_to_w(_context->comm, _Sigma_shm, *_FT, w_pgrid, w_bsize);
    _Timer.stop("SIGMA_TAU_TO_W");
    auto dG_wskij = make_distributed_array<Array_5D_t>(_context->comm, w_pgrid,
                                                       {_nw, _ns, _nkpts_ibz, _nbnd, _nbnd}, w_bsize);
    auto [nw_loc, ns_loc, nk_loc, ni_loc, nj_loc] = dSigma_wskij.local_shape();
    auto [w_org, s_org, k_org, i_org, j_org] = dSigma_wskij.origin();
    auto i_rng = dSigma_wskij.local_range(3);
    auto j_rng = dSigma_wskij.local_range(4);

    // Setup wk_intra_comm
    int color = w_org*_nkpts_ibz + k_org;
    int key = _context->comm.rank();
    mpi3::communicator wk_intra_comm = _context->comm.split(color, key);
    utils::check(wk_intra_comm.size() == w_pgrid[3]*w_pgrid[4], "wk_intra_comm.size() != pgrid[3]*pgrid[4]");
    auto dX = make_distributed_array<nda::array<ComplexType, 2>>(wk_intra_comm, {w_pgrid[3],w_pgrid[4]},
        {_nbnd,_nbnd}, {w_bsize[3],w_bsize[4]});

    auto S  = _sS_skij.local();
    auto H0 = _sH0_skij.local();
    auto F  = _sF_skij.local();
    auto Sigma_w_loc = dSigma_wskij.local();
    auto G_w_loc = dG_wskij.local();
    auto X_loc = dX.local();

    // Dyson on w-axis
    _Timer.start("DYSON_LOOP");
    for (long nsk = 0; nsk < nw_loc*ns_loc*nk_loc; ++nsk) {
      long n = nsk / (ns_loc*nk_loc); // nsk = n*ns_loc*nk_loc + s*nk_loc + k
      long s = (nsk / nk_loc) % ns_loc;
      long k = nsk % nk_loc;

      long wn = _FT->wn_mesh()(n+w_org);
      ComplexType omega_mu = _FT->omega(wn) + mu;
      X_loc = omega_mu * S(s+s_org,k+k_org,i_rng,j_rng) - H0(s+s_org,k+k_org,i_rng,j_rng)
              - F(s+s_org,k+k_org,i_rng,j_rng) - Sigma_w_loc(n,s,k,nda::ellipsis{});
      math::nda::slate_ops::inverse(dX);
      G_w_loc(n,s,k,nda::ellipsis{}) = X_loc;
    }
    _Timer.stop("DYSON_LOOP");
    dSigma_wskij.reset();
    _context->comm.barrier();

    // G(w) -> G(tau)
    {
      auto t_pgrid = dyson_tau_proc_grid(_context->comm.size(), _nkpts_ibz);

      auto dG_wskij_tmp = make_distributed_array<Array_5D_t>(_context->comm, t_pgrid,
                                                             {_nw, _ns, _nkpts_ibz, _nbnd, _nbnd});
      _Timer.start("REDISTRIBUTE");
      math::nda::redistribute(dG_wskij, dG_wskij_tmp);
      _Timer.stop("REDISTRIBUTE");
      dG_wskij.reset(); 

      auto dG_tskij = make_distributed_array<Array_5D_t>(_context->comm, t_pgrid,
                                                         {_nts, _ns, _nkpts_ibz, _nbnd, _nbnd});
      auto Gt_loc = dG_tskij.local();
      auto Gw_loc = dG_wskij_tmp.local();
      _FT->w_to_tau(Gw_loc, Gt_loc, imag_axes_ft::fermion);
      dG_wskij_tmp.reset();

      _FT->check_leakage(dG_tskij, imag_axes_ft::fermion, "Green's function");

      // Gather to shared memory
      _Timer.start("DYSON_GATHER");
      math::nda::gather_to_shm(dG_tskij, _G_shm);
      _Timer.stop("DYSON_GATHER");
    }
    _context->comm.barrier();
    _Timer.stop("DYSON");
    print_timers();
  }

  template<typename Dm_t, typename G_t, typename F_t, typename Sigma_t>
  void simple_dyson::solve_dyson(Dm_t &_sDm_skij, G_t &_G_shm, const F_t &_sF_skij,
                                 const Sigma_t &_Sigma_shm, double mu) {
    solve_dyson(_G_shm, _sF_skij, _Sigma_shm, mu);
    _Timer.start("DM_TAU_TO_BETA");
    if (_context->node_comm.root()) {
      auto Dm = _sDm_skij.local();
      _FT->tau_to_beta(_G_shm.local(), Dm);
      Dm *= -1;
    }
    _context->comm.barrier();
    _Timer.stop("DM_TAU_TO_BETA");
  }


  template<typename X_t, typename Xt_t>
  void simple_dyson::compute_eigenspectra(
    const X_t&_sF_skij, 
    const Xt_t &_Sigma_shm, 
    nda::array<ComplexType, 4> &spectra){

    utils::check(spectra.shape() == std::array<long, 4>{_nw, _ns, _nkpts_ibz, _nbnd},
                 "simple_dyson::compute_eigenspectra: Incorrect dimension for spectra.");
    using math::shm::make_shared_array;
    
    spectra() = 0.0;
    nda::matrix<ComplexType> Sigmaw_ij(_nbnd, _nbnd);
    nda::matrix<ComplexType> FpSigma(_nbnd, _nbnd);
    auto& SFS = Sigmaw_ij;
    auto sS_inv = make_shared_array<Array_view_4D_t>(*_context, {_ns, _nkpts_ibz, _nbnd, _nbnd});
    auto Sigma_tskij = _Sigma_shm.local();
    auto S  = _sS_skij.local();
    auto H0 = _sH0_skij.local();
    auto F  = _sF_skij.local();
    auto S_inv = sS_inv.local();

    int node_rank = _context->node_comm.rank();
    int node_size = _context->node_comm.size();
    sS_inv.win().fence();
    for (size_t sk = node_rank; sk < _ns*_nkpts_ibz; sk+=node_size) {
      size_t is = sk / _nkpts_ibz;
      size_t ik = sk % _nkpts_ibz;
      nda::matrix_const_view<ComplexType> S_ij = S(is, ik, nda::ellipsis{});
      S_inv(is, ik, nda::ellipsis{}) = nda::inverse(S_ij);
    }
    sS_inv.win().fence();

    auto nw_half = _nw/2 + _nw%2;

    // Static case (Sigma == 0, e.g. HF): M(iw) below is frequency-independent, so a
    // single eigensolve per (s,k) yields the spectra at every frequency. The FT of an
    // exactly-zero Sigma is exactly zero, so this is bit-identical to the general path.
    if (sigma_is_zero(*_context, Sigma_tskij)) {
      for (size_t sk = _context->comm.rank(); sk < _ns*_nkpts_ibz; sk += _context->comm.size()) {
        auto is = sk / _nkpts_ibz;
        auto ik = sk % _nkpts_ibz;

        FpSigma = H0(is, ik, nda::ellipsis{}) + F(is, ik, nda::ellipsis{});
        math::detail::gemm_guarded(ComplexType(1.0), S_inv(is, ik, nda::ellipsis{}), FpSigma, ComplexType(0.0), SFS);
        auto eigvals = nda::linalg::geigenvalues(SFS);

        for (size_t n_neg = 0; n_neg < nw_half; ++n_neg) {
          auto n_pos = _nw - n_neg - 1;
          spectra(n_pos, is, ik, nda::range::all) = eigvals;
          if (n_neg != n_pos)
            spectra(n_neg, is, ik, nda::range::all) = nda::conj(eigvals);
        }
      }
      _context->comm.all_reduce_in_place_n(spectra.data(), spectra.size(), std::plus<>{});
      return;
    }

    // Define M(iw) = S^{-1} * [H0 + F + Sigma(iw)], we have M(-iw) = S^{-1} * M(iw)^{\dagger} * S and thus
    // eigvals(M(-iw)) = conj(eigvals(M(iw))). Thus we only need to compute eigvals for half of the frequencies.
    // This relation holds for general non-orthogonal basis with general oeverlap matrix S.
    for (size_t nsk = _context->comm.rank(); nsk < nw_half*_ns*_nkpts_ibz; nsk+=_context->comm.size()) {

      // nsk = n_neg*ns*nkpts_ibz + is*nkpts_ibz + k
      auto n_neg = nsk / (_ns*_nkpts_ibz); 
      auto n_pos = _nw - n_neg - 1;
      auto is = (nsk / _nkpts_ibz) % _ns;
      auto ik = nsk % _nkpts_ibz;

      auto Sigma_tij = nda::make_regular(Sigma_tskij(nda::range::all, is, ik, nda::ellipsis{}));
      _FT->tau_to_w(Sigma_tij, Sigmaw_ij, imag_axes_ft::fermion, n_pos);

      FpSigma = H0(is, ik, nda::ellipsis{}) + F(is, ik, nda::ellipsis{}) + Sigmaw_ij;

      math::detail::gemm_guarded(ComplexType(1.0), S_inv(is, ik, nda::ellipsis{}), FpSigma, ComplexType(0.0), SFS);

      // Matsubara quantities are not Hermitian!
      spectra(n_pos, is, ik, nda::range::all) = nda::linalg::geigenvalues(SFS);

      // exploit eigvals(M(-iw)) = conj(eigvals(M(iw)))
      if (n_neg != n_pos) {
        spectra(n_neg, is, ik, nda::range::all) = nda::conj(spectra(n_pos, is, ik, nda::range::all));
      }
    }
    _context->comm.all_reduce_in_place_n(spectra.data(), spectra.size(), std::plus<>{});
  }




  /** Instantiation of public template **/
  template void simple_dyson::solve_dyson(sArray_t<Array_view_5D_t>&,
      const sArray_t<Array_view_4D_t>&, const sArray_t<Array_view_5D_t> &, double);
  template void simple_dyson::solve_dyson(sArray_t<Array_view_4D_t>&, sArray_t<Array_view_5D_t>&,
      const sArray_t<Array_view_4D_t>&, const sArray_t<Array_view_5D_t> &, double);

  template void simple_dyson::compute_eigenspectra(const sArray_t<Array_view_4D_t>&,
      const sArray_t<Array_view_5D_t> &, nda::array<ComplexType, 4> &);

} // methods
