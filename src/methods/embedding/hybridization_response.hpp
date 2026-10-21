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


#ifndef COQUI_HYBRIDIZATION_RESPONSE_HPP
#define COQUI_HYBRIDIZATION_RESPONSE_HPP

#include <array>
#include <cmath>
#include <map>
#include <vector>

#include "configuration.hpp"
#include "nda/nda.hpp"
#include "nda/blas.hpp"
#include "nda/linalg.hpp"
#include "numerics/imag_axes_ft/IAFT.hpp"
#include "numerics/shared_array/nda.hpp"
#include "utilities/check.hpp"
#include "methods/SCF/dyson_omega.hpp"
#include "methods/embedding/dc_response.hpp"
#include "methods/embedding/local_projection.hpp"

/**
 * Local Green's function of a ground state and the linear response of the Weiss-field hybridization, both
 * evaluated pointwise on the Matsubara axis.
 *
 * Not shared with downfold_loc / lr_dyson: those return G(τ) / δG(τ), and B·T (B = tau_to_w, T = w_to_tau;
 * B·T ≠ I on DLR), applied after a product of two pointwise factors in iω, is what these functions avoid.
 */
namespace methods {

  /**
   * Orbital symmetrization X(iω)_{s I ab} <- [X_{ab} + X_{ba}]/2 of a (nw, ns, nImps, n, n) array, the projection of
   * the local response (see project_lr_local_response).
   */
  inline void symmetrize_orbitals(nda::array<ComplexType, 5>& X) {
    long nw = X.shape(0), ns = X.shape(1), nImps = X.shape(2), n = X.shape(3);
    for (long iw = 0; iw < nw; ++iw)
      for (long is = 0; is < ns; ++is)
        for (long imp = 0; imp < nImps; ++imp)
          for (long a = 0; a < n; ++a)
            for (long b = a + 1; b < n; ++b) {
              ComplexType avg = 0.5 * (X(iw, is, imp, a, b) + X(iw, is, imp, b, a));
              X(iw, is, imp, a, b) = avg;
              X(iw, is, imp, b, a) = avg;
            }
  }

  /**
   * The iω form of the force_real projection of downfold_loc<true> (hermitize in τ, drop the imaginary part) on a
   * (nw, ns, nImps, n, n) array of the fermionic mesh: P(X)(iω_n) = [S(iω_n) + conj S(iω_{-n})]/2, S = (X + Xᵀ)/2.
   * With nw = 1 it is the static P(X) = (X + Xᵀ + X* + X†)/4 = Re (X + Xᵀ)/2. Frequency flip n -> nw-1-n pairs iω_n with -iω_n.
   */
  inline void force_real_omega(nda::array<ComplexType, 5>& X) {
    symmetrize_orbitals(X);
    long nw = X.shape(0);
    for (long iw = 0; iw < nw / 2 + nw % 2; ++iw) {
      long jw = nw - 1 - iw;
      nda::array<ComplexType, 4> xi(X(iw, nda::ellipsis{})), xj(X(jw, nda::ellipsis{}));
      X(iw, nda::ellipsis{}) = 0.5 * (xi + nda::conj(xj));
      X(jw, nda::ellipsis{}) = 0.5 * (xj + nda::conj(xi));
    }
  }

  // Adds the contribution of the k point ik of the spin is, with the block G_w = G_k(iω)[nw, nb, nb], to the
  // not yet normalized local G_loc(iω) (nw, ns, nImps, n, n). trev: G_k = G_{-k}'s time-reversal partner,
  // G_{-k}(iω_n) = conj G_k(iω_{-n}).
  inline void add_local_G_k(nda::array<ComplexType, 5>& G_loc, const nda::MemoryArrayOfRank<3> auto& G_w, long is,
                            long ik, bool trev, const nda::array<ComplexType, 5>& C_skIai,
                            const std::vector<nda::range>& W_rng) {
    long nw = G_loc.shape(0);
    long nImps = C_skIai.shape(2);
    long n = C_skIai.shape(3);
    long nW = C_skIai.shape(4);
    nda::array<ComplexType, 2> O_ij_W(nW, nW);
    nda::array<ComplexType, 2> buffer_aj(n, nW);
    for (long imp = 0; imp < nImps; ++imp)
      for (long iw = 0; iw < nw; ++iw) {
        if (trev) O_ij_W = nda::conj(G_w(nw - 1 - iw, W_rng[imp], W_rng[imp]));
        else O_ij_W = G_w(iw, W_rng[imp], W_rng[imp]);
        local_projection::add_C_O_Cdag(C_skIai, is, ik, ik, imp, O_ij_W, buffer_aj, G_loc(iw, is, imp, nda::ellipsis{}));
      }
  }

  // Sum over the ranks, 1/N_k and the optional force_real projection of the accumulated G_loc.
  template<typename comm_t>
  void finish_local_G(comm_t comm, nda::array<ComplexType, 5>& G_loc, long nk, bool force_real) {
    comm.all_reduce_in_place_n(G_loc.data(), G_loc.size(), std::plus<>{});
    G_loc() /= double(nk);
    if (force_real) force_real_omega(G_loc);
  }

  inline void check_antisymmetric_fermionic_mesh(const imag_axes_ft::IAFT& ft) {
    long nw = ft.nw_f();
    for (long iw = 0; iw < nw; ++iw)
      utils::check(ft.wn_mesh_f()(nw - 1 - iw) == -ft.wn_mesh_f()(iw),
                   "local_G_omega: the fermionic mesh is not antisymmetric at index {}.", iw);
  }

  /**
   * G_loc^pw(iω)_{s I ab} = (1/N_k) Σ_k C^k_{s I} G_{s,k}(iω)[W,W] C^{k†}_{s I}, (nw_f, ns, nImps, n, n), with
   * G_k(iω) = [(iω + μ) S - H0 - F - (B Σ)(iω)]⁻¹ the pointwise Dyson G of the stored ground state
   * (B = tau_to_w; T·G_k(iω) is the stored G_k(τ), T = w_to_tau).
   *
   * H0, S, F, Σ(τ) are IBZ arrays; kp_to_ibz / kp_trev (both nullptr: identity, no time reversal) unfold them to the
   * k mesh of C. Time reversal: G_{-k}(τ) = conj G_k(τ), hence G_{-k}(iω_n) = conj G_k(iω_{-n}).
   * force_real: force_real_omega (the iω form of downfold_loc<true>). Collective on comm; the (s, k_ibz) blocks are
   * dealt round-robin and no G_k is kept.
   */
  template<math::shm::SharedArray sArr4_t, math::shm::SharedArray sArr5_t, typename comm_t>
  nda::array<ComplexType, 5> local_G_omega(comm_t comm, const imag_axes_ft::IAFT& ft, double mu,
                                           const sArr4_t& sH0_skij, const sArr4_t& sS_skij, const sArr4_t& sF_skij,
                                           const sArr5_t& sSigma_tskij, const nda::array<ComplexType, 5>& C_skIai,
                                           const std::vector<nda::range>& W_rng,
                                           const nda::array<int, 1>* kp_to_ibz, const nda::array<bool, 1>* kp_trev,
                                           bool force_real) {
    auto [nt, ns, nk_ibz, nb, nb2] = sSigma_tskij.shape();
    long nw = ft.nw_f();
    long nk = C_skIai.shape(1);
    long nImps = C_skIai.shape(2);
    long n = C_skIai.shape(3);
    utils::check(ns == C_skIai.shape(0) and long(W_rng.size()) == nImps,
                 "local_G_omega: ns mismatch ({} vs C {}) or W_rng size {} != nImps {}.", ns, C_skIai.shape(0),
                 W_rng.size(), nImps);
    utils::check(kp_to_ibz != nullptr or nk_ibz == nk,
                 "local_G_omega: without kp_to_ibz the arrays must be on the full mesh (nk = {}, C nk = {}).", nk_ibz, nk);
    utils::check(kp_to_ibz == nullptr or kp_to_ibz->shape(0) == nk,
                 "local_G_omega: kp_to_ibz size != nk = {}.", nk);
    check_antisymmetric_fermionic_mesh(ft);

    nda::array<ComplexType, 5> G_loc(nw, ns, nImps, n, n);
    G_loc() = 0.0;
    auto H0 = sH0_skij.local();
    auto S = sS_skij.local();
    auto F = sF_skij.local();
    auto Sigma = sSigma_tskij.local();
    nda::array<ComplexType, 3> G_w(nw, nb, nb);
    for (long i = comm.rank(); i < ns * nk_ibz; i += comm.size()) {
      long is = i / nk_ibz;
      long ik_ibz = i % nk_ibz;
      dyson_G_omega_block(ft, mu, H0(is, ik_ibz, nda::ellipsis{}), S(is, ik_ibz, nda::ellipsis{}),
                          F(is, ik_ibz, nda::ellipsis{}),
                          Sigma(nda::range::all, is, ik_ibz, nda::range::all, nda::range::all), G_w);
      for (long ik = 0; ik < nk; ++ik) {
        if ((kp_to_ibz == nullptr ? ik : (*kp_to_ibz)(ik)) != ik_ibz) continue;
        add_local_G_k(G_loc, G_w, is, ik, kp_trev != nullptr and (*kp_trev)(ik), C_skIai, W_rng);
      }
    }
    finish_local_G(comm, G_loc, nk, force_real);
    return G_loc;
  }

  /**
   * G_loc^pw(iω) of local_G_omega from the node-shared G_k(iω) of the full BZ, sG_wskij (nw_f, ns, nk, nb, nb)
   * (lr_driver::G_omega()), instead of a second Dyson solve; the same sum in the same order, so it is bit-identical to
   * local_G_omega on the full mesh for the same communicator. Collective on comm.
   */
  template<math::shm::SharedArray sArr5_t, typename comm_t>
  nda::array<ComplexType, 5> local_G_omega_from_G(comm_t comm, const imag_axes_ft::IAFT& ft, const sArr5_t& sG_wskij,
                                                  const nda::array<ComplexType, 5>& C_skIai,
                                                  const std::vector<nda::range>& W_rng, bool force_real) {
    auto [nw_G, ns, nk, nb, nb2] = sG_wskij.shape();
    long nw = ft.nw_f();
    long nImps = C_skIai.shape(2);
    long n = C_skIai.shape(3);
    utils::check(nw_G == nw and ns == C_skIai.shape(0) and nk == C_skIai.shape(1) and long(W_rng.size()) == nImps,
                 "local_G_omega_from_G: shape mismatch (G nw = {}, ns = {}, nk = {}; C ns = {}, nk = {}; W_rng size {} vs "
                 "nImps {}); the full BZ is required.", nw_G, ns, nk, C_skIai.shape(0), C_skIai.shape(1), W_rng.size(), nImps);
    check_antisymmetric_fermionic_mesh(ft);

    nda::array<ComplexType, 5> G_loc(nw, ns, nImps, n, n);
    G_loc() = 0.0;
    auto G = sG_wskij.local();
    for (long i = comm.rank(); i < ns * nk; i += comm.size()) {
      long is = i / nk;
      long ik = i % nk;
      add_local_G_k(G_loc, G(nda::range::all, is, ik, nda::range::all, nda::range::all), is, ik, false, C_skIai, W_rng);
    }
    finish_local_G(comm, G_loc, nk, force_real);
    return G_loc;
  }

  /**
   * δG_loc^pw(iω)_{s ab} = (1/N_k) Σ_k C^{k+Q} G_{k+Q}(iω)[W,:] δX_k(iω) G_k(iω)[:,W] C^{k†}, (nw_f, ns, 1, n, n), with
   * the converged source of lr_dyson,
   *   δX_k(iω) = ΔH0_k + ΔF_k - Δμ S_k + (B ΔΣ_k)(iω),
   * and sG_wskij = lr_driver::G_omega() (full BZ; kpq_map(k) is the index of k+Q). ΔΣ is the (nt_f, ns, nk, nb, nb)
   * array of lr_dyson (nullptr: none). No transform acts on the product. Collective on comm.
   */
  template<math::shm::SharedArray sArr4_t, math::shm::SharedArray sArr5_t, typename comm_t>
  nda::array<ComplexType, 5> lr_local_DeltaG_omega(comm_t comm, const imag_axes_ft::IAFT& ft, const sArr5_t& sG_wskij,
                                                   const sArr4_t& sDeltaH0_skij, const sArr4_t& sDeltaF_skij,
                                                   const sArr5_t* pDeltaSigma_tskij, double Delta_mu,
                                                   const sArr4_t& sS_skij, const nda::array<ComplexType, 5>& C_skIai,
                                                   const std::vector<nda::range>& W_rng,
                                                   const nda::array<long, 1>& kpq_map) {
    auto [nw_G, ns, nk, nb, nb2] = sG_wskij.shape();
    long nw = ft.nw_f();
    long n = C_skIai.shape(3);
    long nW = C_skIai.shape(4);
    utils::check(nw_G == nw and C_skIai.shape(1) == nk and C_skIai.shape(0) == ns and C_skIai.shape(2) == 1 and
                     W_rng.size() == 1 and kpq_map.shape(0) == nk,
                 "lr_local_DeltaG_omega: shape mismatch (G nw = {}, nk = {}, ns = {}; C {}x{}x{}; kpq_map {}); a single "
                 "impurity on the full BZ is required.", nw_G, nk, ns, C_skIai.shape(0), C_skIai.shape(1),
                 C_skIai.shape(2), kpq_map.shape(0));
    long nt = ft.nt_f();

    nda::array<ComplexType, 5> dG_loc(nw, ns, 1, n, n);
    dG_loc() = 0.0;
    auto G = sG_wskij.local();
    auto dH0 = sDeltaH0_skij.local();
    auto dF = sDeltaF_skij.local();
    auto S = sS_skij.local();
    nda::array<ComplexType, 3> dSigma_t(nt, nb, nb), dSigma_w(nw, nb, nb);
    nda::array<ComplexType, 2> X(nb, nb), XR(nb, nW), O_ij_W(nW, nW);
    nda::array<ComplexType, 2> buffer_aj(n, nW);
    for (long i = comm.rank(); i < ns * nk; i += comm.size()) {
      long is = i / nk;
      long ik = i % nk;
      long ikq = kpq_map(ik);
      if (pDeltaSigma_tskij) {
        dSigma_t = pDeltaSigma_tskij->local()(nda::range::all, is, ik, nda::range::all, nda::range::all);
        ft.tau_to_w(dSigma_t, dSigma_w, imag_axes_ft::fermion);
      }
      for (long iw = 0; iw < nw; ++iw) {
        X = dH0(is, ik, nda::ellipsis{}) + dF(is, ik, nda::ellipsis{}) - Delta_mu * S(is, ik, nda::ellipsis{});
        if (pDeltaSigma_tskij) X += dSigma_w(iw, nda::ellipsis{});
        auto G_k = G(iw, is, ik, nda::range::all, W_rng[0]);
        auto G_kq = G(iw, is, ikq, W_rng[0], nda::range::all);
        nda::blas::gemm(X, G_k, XR);
        nda::blas::gemm(G_kq, XR, O_ij_W);
        local_projection::add_C_O_Cdag(C_skIai, is, ikq, ik, 0, O_ij_W, buffer_aj, dG_loc(iw, is, 0, nda::ellipsis{}));
      }
    }
    comm.all_reduce_in_place_n(dG_loc.data(), dG_loc.size(), std::plus<>{});
    dG_loc() /= double(nk);
    return dG_loc;
  }

  /**
   * Projection of the local response onto what the force_real ground-state map applies to G_loc and H^loc:
   *   Q = 0 (q_zero):  δG_loc^pw <- P(δG_loc^pw) = force_real_omega, i.e. [S(iω_n) + conj S(iω_{-n})]/2, S = (δG + δGᵀ)/2;
   *                    δH_loc <- P(δH_loc) = Re (δH_loc + δH_locᵀ)/2. P is real-linear: ΔH0 at Q = 0 must be Hermitian.
   *   Q != 0:          X <- Sym(X) = (X + Xᵀ)/2 at the same Q and iω_n, for δG_loc^pw and δH_loc.
   * For a GS with [G_k(iω_n)]† = G_k(iω_{-n}) and V = X e^{iQR} + X† e^{-iQR} (δG^{-Q}: the response to X†),
   * [δG^Q(iω_n)]† = δG^{-Q}(iω_{-n}) gives conj δG^{-Q}(iω_{-n}) = δG^Q(iω_n)ᵀ, so the Q component of P applied to
   * the real-space response is Sym(δG^Q).
   * dGloc_wsIab (nw_f, ns, 1, n, n), dHloc_sIab (ns, 1, n, n).
   * Returns the relative change {|proj dG - dG|/|dG|, |proj dH - dH|/|dH|} (0 for a zero response).
   */
  inline std::array<double, 2> project_lr_local_response(nda::array<ComplexType, 5>& dGloc_wsIab,
                                                         nda::array<ComplexType, 4>& dHloc_sIab, bool q_zero) {
    auto rel_change = [](auto const& proj, auto const& orig) {
      double d = 0.0, x = 0.0;
      for (auto const& v : orig) x += std::norm(v);
      nda::array<ComplexType, 5> diff = proj - orig;
      for (auto const& v : diff) d += std::norm(v);
      return x > 0.0 ? std::sqrt(d / x) : 0.0;
    };
    std::array<double, 2> change{0.0, 0.0};
    nda::array<ComplexType, 5> orig = dGloc_wsIab;
    if (q_zero) force_real_omega(dGloc_wsIab);
    else symmetrize_orbitals(dGloc_wsIab);
    change[0] = rel_change(dGloc_wsIab, orig);
    nda::array<ComplexType, 5> H(1, dHloc_sIab.shape(0), dHloc_sIab.shape(1), dHloc_sIab.shape(2), dHloc_sIab.shape(3));
    H(0, nda::ellipsis{}) = dHloc_sIab;
    orig = nda::array<ComplexType, 5>(H);
    if (q_zero) force_real_omega(H);
    else symmetrize_orbitals(H);
    change[1] = rel_change(H, orig);
    dHloc_sIab = H(0, nda::ellipsis{});
    return change;
  }

  /**
   * δΔ(iω) of the Weiss field, per 1e alpha block α of cells.orb_block, layout (nw_f, ns, 1, n, n), zero off-block:
   *
   *   δΔ[α,α] = Ginv_α δG_loc^pw[α,α] Ginv_α - dHloc[α,α] - dSigma[α,α],   Ginv_α = [G_loc^pw[α,α]]⁻¹
   *
   * Gloc_wsIab = G_loc^pw (local_G_omega / local_G_omega_from_G), dGloc_wsIab = δG_loc^pw (lr_local_DeltaG_omega), dHloc_sIab the static
   * δH^loc, dSigma_wsIab = δΣ_imp(iω) as given to run_lr (zero if none). Not collective: called on one rank.
   *
   * dGloc_wsIab and dHloc_sIab are taken as given: pass them through project_lr_local_response to differentiate the
   * force_real ground-state map.
   * The block loop runs over the orbitals of each dc_cells.orb_block run; it is the ground-state map only when these
   * runs coincide with the gf_struct blocks of embedding['1e'] (not checked).
   */
  inline nda::array<ComplexType, 5> lr_hybridization_w(
      const nda::array<ComplexType, 5>& Gloc_wsIab,
      const nda::array<ComplexType, 5>& dGloc_wsIab,
      const nda::array<ComplexType, 4>& dHloc_sIab,
      const nda::array<ComplexType, 5>& dSigma_wsIab,
      const dc_cells_t& cells) {
    long nw = Gloc_wsIab.shape(0);
    long ns = Gloc_wsIab.shape(1);
    long n = Gloc_wsIab.shape(3);
    cells.check(n);
    utils::check(Gloc_wsIab.shape() == std::array<long, 5>{nw, ns, 1, n, n} and dGloc_wsIab.shape() == Gloc_wsIab.shape()
                     and dSigma_wsIab.shape() == Gloc_wsIab.shape(),
                 "lr_hybridization_w: Gloc, dGloc and dSigma must have shape (nw_f, ns, 1, n, n) = ({}, {}, 1, {}, {}).",
                 nw, ns, n, n);
    utils::check(dHloc_sIab.shape() == std::array<long, 4>{ns, 1, n, n},
                 "lr_hybridization_w: dHloc must have shape (ns, 1, n, n) = ({}, 1, {}, {}).", ns, n, n);

    // The orbitals of each 1e alpha block, in increasing order.
    std::map<long, std::vector<long>> blocks;
    for (long o = 0; o < n; ++o)
      if (cells.orb_block(o) >= 0) blocks[cells.orb_block(o)].push_back(o);

    nda::array<ComplexType, 5> dDelta(nw, ns, 1, n, n);
    dDelta() = 0.0;
    for (auto const& [alpha, idx] : blocks) {
      long m = idx.size();
      nda::matrix<ComplexType> Gb(m, m), Xb(m, m);
      for (long w = 0; w < nw; ++w)
        for (long s = 0; s < ns; ++s) {
          for (long a = 0; a < m; ++a)
            for (long b = 0; b < m; ++b) {
              Gb(a, b) = Gloc_wsIab(w, s, 0, idx[a], idx[b]);
              Xb(a, b) = dGloc_wsIab(w, s, 0, idx[a], idx[b]);
            }
          nda::matrix<ComplexType> Ginv = nda::inverse(Gb);
          nda::matrix<ComplexType> Rb = Ginv * Xb * Ginv;
          for (long a = 0; a < m; ++a)
            for (long b = 0; b < m; ++b)
              dDelta(w, s, 0, idx[a], idx[b]) =
                  Rb(a, b) - dHloc_sIab(s, 0, idx[a], idx[b]) - dSigma_wsIab(w, s, 0, idx[a], idx[b]);
        }
    }
    return dDelta;
  }

} // namespace methods

#endif // COQUI_HYBRIDIZATION_RESPONSE_HPP
