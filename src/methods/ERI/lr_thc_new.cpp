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

#include "lr_thc_new.hpp"
#include "lr_psi_io.hpp"

#include <cmath>
#include <string>
#include <utility>
#include <vector>

#include "configuration.hpp"
#include "IO/app_loggers.h"
#include "utilities/check.hpp"
#include "utilities/kpoint_utils.hpp"
#include "utilities/lr_utils.hpp"
#include "numerics/fft/nda.hpp"
#include "nda/blas.hpp"
#include "nda/lapack/getrf.hpp"
#include "nda/lapack/getrs.hpp"

namespace methods {

// ----------------------------------------------------------------------------
// Step 1: read ψ and (+Q, -Q) δψ on the full FFT grid for all (is, ik).
//
// Outputs:
//   psi_skmr(is, ik, m, r)           = ψ^{k_ik}_m(r) · e^{i k·r}
//   Deltapsi_pQ_skmr(is, ik, m, r)   = δ^{+Q} ψ^{k_ik}_m(r) · e^{i(k+Q)·r}
//   Deltapsi_mQ_skmr(is, ik, m, r)   = δ^{-Q} ψ^{k_ik}_m(r) · e^{i(k-Q)·r}
//
// The +Q file at filename index ik stores δ^{+Q} ψ^{k_ik} directly. The -Q
// (adj) file at filename index ik stores δ^{-Q} ψ^{k_ik + Q}; to fill the
// slot for ψ at k_ik, we read the adj file at filename index kmQ_map(ik),
// whose data is δ^{-Q} ψ^{k_{kmQ_map(ik)} + Q} = δ^{-Q} ψ^{k_ik} (since
// kmQ_map maps k_ik - Q → ik).
// ----------------------------------------------------------------------------
void lr_thc_new::read_psi_and_dpsi(
    std::string const& Deltapsi_prefix,
    std::string const& Deltapsi_adj_prefix,
    nda::stack_array<double, 3> const& q_pert_cart,
    nda::stack_array<long, 3> const& fft_mesh,
    nda::array<int, 1> const& kmQ_map,
    nda::array<ComplexType, 4>& psi_skmr,
    nda::array<ComplexType, 4>& Deltapsi_pQ_skmr,
    nda::array<ComplexType, 4>& Deltapsi_mQ_skmr)
{
  Timer.start("read_psi_and_dpsi");
  auto const& mf = *_reader.MF();
  long const nspin = mf.nspin();
  long const nkpts = mf.nkpts();
  long const nbnd  = mf.nbnd();
  long const N1 = fft_mesh(0), N2 = fft_mesh(1), N3 = fft_mesh(2);
  long const nnr = N1 * N2 * N3;

  auto lattv_mf = mf.lattv();
  auto kpts_mf  = mf.kpts(); // (nk, 3) Cartesian (2π included)

  // FFT workspace
  nda::array<ComplexType, 4> fft_buf(1, N1, N2, N3);
  auto fft_buf_1d = nda::reshape(fft_buf, std::array<long, 1>{nnr});
  math::nda::fft<true> F(fft_buf);

  for (long is = 0; is < nspin; ++is) {
    for (long ik = 0; ik < nkpts; ++ik) {
      nda::stack_array<double, 3> k_cart, kpQ_cart, kmQ_cart;
      for (int a = 0; a < 3; ++a) {
        k_cart  (a) = kpts_mf(ik, a);
        kpQ_cart(a) = k_cart(a) + q_pert_cart(a);
        kmQ_cart(a) = k_cart(a) - q_pert_cart(a);
      }

      // --- ψ: full file at (is, ik). Bloch phase e^{ik·r}.
      Timer.start("io_wfc");
      auto wfc = detail::read_wfc_k(mf, is, ik, fft_mesh);
      Timer.stop("io_wfc");
      utils::check(wfc.nbnd >= nbnd,
                   "lr_thc_new::read_psi_and_dpsi: wfc file at (is={},ik={}) "
                   "has {} bands, expected >= {}.", is, ik, wfc.nbnd, nbnd);
      {
        nda::array<ComplexType, 1> phase_full(nnr);
        Timer.start("phase_factor");
        utils::rspace_phase_factor(lattv_mf, k_cart,
            nda::reshape(phase_full, std::array<long,3>{N1, N2, N3}));
        Timer.stop("phase_factor");
        for (long m = 0; m < nbnd; ++m) {
          fft_buf_1d() = ComplexType(0.0);
          detail::decode_band_to_fft_grid(wfc.evc_raw, m, wfc.k2g, fft_buf_1d);
          F.backward(fft_buf);
          for (long ir = 0; ir < nnr; ++ir)
            psi_skmr(is, ik, m, ir) = fft_buf_1d(ir) * phase_full(ir);
        }
      }

      // --- +Q δψ: file at filename ik stores δ^{+Q} ψ^{k_ik}. Bloch phase e^{i(k+Q)·r}.
      Timer.start("io_dpsi");
      auto dpsi_pQ = detail::read_Deltapsi_k(Deltapsi_prefix, ik, fft_mesh);
      Timer.stop("io_dpsi");
      utils::check(dpsi_pQ.nbnd >= nbnd,
                   "lr_thc_new::read_psi_and_dpsi: +Q dpsi file at ik={} "
                   "has {} bands, expected >= {}.", ik, dpsi_pQ.nbnd, nbnd);
      {
        nda::array<ComplexType, 1> phase_full(nnr);
        Timer.start("phase_factor");
        utils::rspace_phase_factor(lattv_mf, kpQ_cart,
            nda::reshape(phase_full, std::array<long,3>{N1, N2, N3}));
        Timer.stop("phase_factor");
        for (long m = 0; m < nbnd; ++m) {
          fft_buf_1d() = ComplexType(0.0);
          detail::decode_band_to_fft_grid(dpsi_pQ.evc_raw, m, dpsi_pQ.k2g, fft_buf_1d);
          F.backward(fft_buf);
          for (long ir = 0; ir < nnr; ++ir)
            Deltapsi_pQ_skmr(is, ik, m, ir) = fft_buf_1d(ir) * phase_full(ir);
        }
      }

      // --- -Q δψ: read adj file at filename kmQ_map(ik); data is δ^{-Q} ψ^{k_ik}.
      // Bloch phase e^{i(k-Q)·r}.
      long const ik_adj = kmQ_map(ik);
      Timer.start("io_dpsi_adj");
      auto dpsi_mQ = detail::read_Deltapsi_k(Deltapsi_adj_prefix, ik_adj, fft_mesh);
      Timer.stop("io_dpsi_adj");
      utils::check(dpsi_mQ.nbnd >= nbnd,
                   "lr_thc_new::read_psi_and_dpsi: -Q dpsi_adj file at ik={} "
                   "has {} bands, expected >= {}.", ik_adj, dpsi_mQ.nbnd, nbnd);
      {
        nda::array<ComplexType, 1> phase_full(nnr);
        Timer.start("phase_factor");
        utils::rspace_phase_factor(lattv_mf, kmQ_cart,
            nda::reshape(phase_full, std::array<long,3>{N1, N2, N3}));
        Timer.stop("phase_factor");
        for (long m = 0; m < nbnd; ++m) {
          fft_buf_1d() = ComplexType(0.0);
          detail::decode_band_to_fft_grid(dpsi_mQ.evc_raw, m, dpsi_mQ.k2g, fft_buf_1d);
          F.backward(fft_buf);
          for (long ir = 0; ir < nnr; ++ir)
            Deltapsi_mQ_skmr(is, ik, m, ir) = fft_buf_1d(ir) * phase_full(ir);
        }
      }
    }
  }
  Timer.stop("read_psi_and_dpsi");
}

// ----------------------------------------------------------------------------
// Step 2: compute T, ΔT_pQ, ΔT_mQ per-k caches.
//   T^k(μ, r)         = Σ_m conj(ψ^k_m(r_P(μ))) · ψ^k_m(r)
//   ΔT_pQ^k(μ, r)     = Σ_m [ conj(δ^{-Q}ψ^k_m(r_P(μ))) · ψ^k_m(r)        (bra-side)
//                            + conj(ψ^k_m(r_P(μ))) · δ^{+Q}ψ^k_m(r) ]     (ket-side)
//   ΔT_mQ^k(μ, r)     = Σ_m [ conj(δ^{+Q}ψ^k_m(r_P(μ))) · ψ^k_m(r)        (bra-side)
//                            + conj(ψ^k_m(r_P(μ))) · δ^{-Q}ψ^k_m(r) ]     (ket-side)
// (theory note: definitions of T and δT around eqs. 60–63, with the +Q/-Q
// variants needed for the q+Q ≠ q channel.)
// ----------------------------------------------------------------------------
void lr_thc_new::compute_T_and_DeltaT(
    nda::array<ComplexType, 4> const& psi_skmr,
    nda::array<ComplexType, 4> const& Deltapsi_pQ_skmr,
    nda::array<ComplexType, 4> const& Deltapsi_mQ_skmr,
    nda::array<long, 1> const& r_P,
    nda::array<ComplexType, 4>& T_skur,
    nda::array<ComplexType, 4>& DeltaT_pQ_skur,
    nda::array<ComplexType, 4>& DeltaT_mQ_skur)
{
  Timer.start("compute_T_and_DeltaT");
  long const nspin = psi_skmr.shape(0);
  long const nkpts = psi_skmr.shape(1);
  long const nbnd  = psi_skmr.shape(2);
  long const nnr   = psi_skmr.shape(3);
  long const Np    = r_P.shape(0);

  T_skur()         = ComplexType(0.0);
  DeltaT_pQ_skur() = ComplexType(0.0);
  DeltaT_mQ_skur() = ComplexType(0.0);

  // Per-k workspaces: pivot slices (nbnd, Np) of ψ and δ^{±Q}ψ.
  nda::array<ComplexType, 2> psi_pivot     (nbnd, Np);
  nda::array<ComplexType, 2> dpsi_pQ_pivot (nbnd, Np);
  nda::array<ComplexType, 2> dpsi_mQ_pivot (nbnd, Np);

  for (long is = 0; is < nspin; ++is) {
    for (long ik = 0; ik < nkpts; ++ik) {
      // Pull (nbnd, nnr) slices and pivot (nbnd, Np) sub-arrays for this (is, ik).
      auto psi_full     = psi_skmr        (is, ik, nda::range::all, nda::range::all);
      auto dpsi_pQ_full = Deltapsi_pQ_skmr(is, ik, nda::range::all, nda::range::all);
      auto dpsi_mQ_full = Deltapsi_mQ_skmr(is, ik, nda::range::all, nda::range::all);
      for (long m = 0; m < nbnd; ++m)
        for (long u = 0; u < Np; ++u) {
          psi_pivot    (m, u) = psi_full    (m, r_P(u));
          dpsi_pQ_pivot(m, u) = dpsi_pQ_full(m, r_P(u));
          dpsi_mQ_pivot(m, u) = dpsi_mQ_full(m, r_P(u));
        }

      auto T_sk    = T_skur        (is, ik, nda::range::all, nda::range::all);
      auto dTpQ_sk = DeltaT_pQ_skur(is, ik, nda::range::all, nda::range::all);
      auto dTmQ_sk = DeltaT_mQ_skur(is, ik, nda::range::all, nda::range::all);

      // T^k(u, r)        = Σ_m conj(psi_pivot(m, u)) · psi_full(m, r)
      //                  = (psi_pivot^H · psi_full)(u, r)
      nda::blas::gemm(ComplexType(1.0), nda::dagger(psi_pivot), psi_full,
                      ComplexType(0.0), T_sk);

      // ΔT_pQ: bra (conj(δ^{-Q}ψ_pivot) · ψ_full) + ket (conj(ψ_pivot) · δ^{+Q}ψ_full).
      nda::blas::gemm(ComplexType(1.0), nda::dagger(dpsi_mQ_pivot), psi_full,
                      ComplexType(0.0), dTpQ_sk);
      nda::blas::gemm(ComplexType(1.0), nda::dagger(psi_pivot), dpsi_pQ_full,
                      ComplexType(1.0), dTpQ_sk);

      // ΔT_mQ: bra (conj(δ^{+Q}ψ_pivot) · ψ_full) + ket (conj(ψ_pivot) · δ^{-Q}ψ_full).
      nda::blas::gemm(ComplexType(1.0), nda::dagger(dpsi_pQ_pivot), psi_full,
                      ComplexType(0.0), dTmQ_sk);
      nda::blas::gemm(ComplexType(1.0), nda::dagger(psi_pivot), dpsi_mQ_full,
                      ComplexType(1.0), dTmQ_sk);
    }
  }
  Timer.stop("compute_T_and_DeltaT");
}

// ----------------------------------------------------------------------------
// Step 3: Z, C for one internal q (k-q convention).
//   Z^q_{μr} = Σ_k conj(T^{k-q}_{μr}) · T^k_{μr}
//   C^q_{μν} = Z^q_{μ, r_P(ν)}
// kmq_map is the index of (k - q) in the BZ k-grid; obtained by calling
// utils::calculate_kpq_map with -q (which returns the index of k+(-q) = k-q).
// (theory note: eqs. defining Z^q and C^q at fixed q.)
// ----------------------------------------------------------------------------
void lr_thc_new::build_Z_C_for_q(
    nda::array<ComplexType, 4> const& T_skur,
    nda::array<double, 2> const& kpts_crys,
    nda::array<long, 1> const& r_P,
    nda::array<double, 1> const& q_vec_cryst,
    nda::array<ComplexType, 2>& Z_q,
    nda::array<ComplexType, 2>& C_q,
    nda::array<int, 1>& kmq_map)
{
  Timer.start("build_Z_C_for_q");
  long const nspin = T_skur.shape(0);
  long const nkpts = T_skur.shape(1);
  long const Np    = T_skur.shape(2);
  long const nnr   = T_skur.shape(3);

  // k - q index = calculate_kpq_map with q' = -q
  nda::array<double, 1> q_neg(3);
  for (int a = 0; a < 3; ++a) q_neg(a) = -q_vec_cryst(a);
  kmq_map.resize(nkpts);
  utils::calculate_kpq_map(kpts_crys, q_neg, kmq_map);

  Z_q.resize(Np, nnr);
  Z_q() = ComplexType(0.0);

  // Hadamard reduction over k: Z_q(u, r) += conj(T_kmq(u, r)) * T_k(u, r).
  // The T_skur slabs are contiguous (Np, nnr) sub-arrays; loop on raw pointers
  // so the compiler auto-vectorizes the inner complex muladd.
  Timer.start("accumulate_Z");
  {
    long const N = Np * nnr;
    ComplexType* z_ptr = Z_q.data();
    for (long is = 0; is < nspin; ++is) {
      for (long ik = 0; ik < nkpts; ++ik) {
        long const ikmq = kmq_map(ik);
        ComplexType const* tk_ptr   = &T_skur(is, ik,   0, 0);
        ComplexType const* tkmq_ptr = &T_skur(is, ikmq, 0, 0);
        for (long n = 0; n < N; ++n)
          z_ptr[n] += std::conj(tkmq_ptr[n]) * tk_ptr[n];
      }
    }
  }
  Timer.stop("accumulate_Z");

  C_q.resize(Np, Np);
  for (long u = 0; u < Np; ++u)
    for (long v = 0; v < Np; ++v)
      C_q(u, v) = Z_q(u, r_P(v));
  Timer.stop("build_Z_C_for_q");
}

// ----------------------------------------------------------------------------
// Step 4: ΔZ, ΔC for one internal q. kmq_map is precomputed by step 3.
//   δZ^q_{μr} = Σ_k [ conj(ΔT_mQ^{k-q}_{μr}) · T^k_{μr}
//                   + conj(T^{k-q}_{μr}) · ΔT_pQ^k_{μr} ]   (Bloch q+Q)
//   δC^q_{μν} = δZ^q_{μ, r_P(ν)}
// (theory note eqs. for δZ^q with explicit ±Q channels.)
// ----------------------------------------------------------------------------
void lr_thc_new::build_DeltaZ_DeltaC_for_q(
    nda::array<ComplexType, 4> const& T_skur,
    nda::array<ComplexType, 4> const& DeltaT_pQ_skur,
    nda::array<ComplexType, 4> const& DeltaT_mQ_skur,
    nda::array<int, 1> const& kmq_map,
    nda::array<long, 1> const& r_P,
    nda::array<ComplexType, 2>& DeltaZ_q,
    nda::array<ComplexType, 2>& DeltaC_q)
{
  Timer.start("build_DeltaZ_DeltaC_for_q");
  long const nspin = T_skur.shape(0);
  long const nkpts = T_skur.shape(1);
  long const Np    = T_skur.shape(2);
  long const nnr   = T_skur.shape(3);

  DeltaZ_q.resize(Np, nnr);
  DeltaZ_q() = ComplexType(0.0);

  // Hadamard reduction over k via raw pointers for auto-vectorization.
  Timer.start("accumulate_dZ");
  {
    long const N = Np * nnr;
    ComplexType* dz_ptr = DeltaZ_q.data();
    for (long is = 0; is < nspin; ++is) {
      for (long ik = 0; ik < nkpts; ++ik) {
        long const ikmq = kmq_map(ik);
        ComplexType const* tk_ptr      = &T_skur        (is, ik,   0, 0);
        ComplexType const* tkmq_ptr    = &T_skur        (is, ikmq, 0, 0);
        ComplexType const* dTpQ_k_ptr  = &DeltaT_pQ_skur(is, ik,   0, 0);
        ComplexType const* dTmQ_kmq_ptr= &DeltaT_mQ_skur(is, ikmq, 0, 0);
        for (long n = 0; n < N; ++n)
          dz_ptr[n] += std::conj(dTmQ_kmq_ptr[n]) * tk_ptr[n]
                     + std::conj(tkmq_ptr[n])    * dTpQ_k_ptr[n];
      }
    }
  }
  Timer.stop("accumulate_dZ");

  DeltaC_q.resize(Np, Np);
  for (long u = 0; u < Np; ++u)
    for (long v = 0; v < Np; ++v)
      DeltaC_q(u, v) = DeltaZ_q(u, r_P(v));
  Timer.stop("build_DeltaZ_DeltaC_for_q");
}

// ----------------------------------------------------------------------------
// Step 5: Solve C_q · Ξ^q = Z^q in real space (one LU), then FFT Ξ^q to G.
//
// Outputs both the standard FFT image
//   Xi_q_uG(μ, G)     = FFT_forward[e^{-iq·r} · Ξ^q(μ, r)](G)
// and the conjugate FFT image
//   Xi_bar_q_uG(μ, G) = FFT_forward[e^{+iq·r} · conj(Ξ^q(μ, r))](G)
// so that Ξ^q(μ, -G) = [Xi_bar_q_uG(μ, G)]^* without G-reflection lookups.
//
// On exit, C_f_lu / ipiv contain the LU factorization of C_q to be reused
// by step 6 (solve_dxi_for_q).
// ----------------------------------------------------------------------------
void lr_thc_new::solve_xi_for_q(
    nda::array<ComplexType, 2>& C_q,
    nda::array<ComplexType, 2> const& Z_q,
    nda::stack_array<double, 3> const& q_cart,
    nda::array<ComplexType, 2>& Xi_q_ur,
    nda::array<ComplexType, 2>& Xi_q_uG,
    nda::array<ComplexType, 2>& Xi_bar_q_uG,
    nda::matrix<ComplexType, nda::F_layout>& C_f_lu,
    nda::array<int, 1>& ipiv)
{
  Timer.start("solve_xi_for_q");
  long const Np  = C_q.shape(0);
  long const nnr = Z_q.shape(1);

  auto const& mf = *_reader.MF();
  auto const& rho_g = _reader.rho_g();
  long const nG_red = rho_g.size();
  auto mesh = rho_g.mesh();
  long const N1 = mesh(0), N2 = mesh(1), N3 = mesh(2);

  auto lattv_mf = mf.lattv();

  // --- Real-space LU solve. Copy C_q into Fortran layout, factor once. ---
  C_f_lu.resize(Np, Np);
  for (long u = 0; u < Np; ++u)
    for (long v = 0; v < Np; ++v) C_f_lu(u, v) = C_q(u, v);
  ipiv.resize(Np);

  Timer.start("lu_factor");
  int info_f = nda::lapack::getrf(C_f_lu, ipiv);
  Timer.stop("lu_factor");
  utils::check(info_f == 0, "lr_thc_new::solve_xi_for_q: getrf failed, info = {}", info_f);

  // RHS in Fortran layout: (Np, nnr) — overwritten with Ξ^q(μ, r).
  nda::matrix<ComplexType, nda::F_layout> Xi_f(Np, nnr);
  for (long u = 0; u < Np; ++u)
    for (long ir = 0; ir < nnr; ++ir) Xi_f(u, ir) = Z_q(u, ir);

  Timer.start("lu_solve");
  int info_s = nda::lapack::getrs(C_f_lu, Xi_f, ipiv);
  Timer.stop("lu_solve");
  utils::check(info_s == 0, "lr_thc_new::solve_xi_for_q: getrs failed, info = {}", info_s);

  // Copy Ξ^q(μ, r) into the output real-space buffer.
  Xi_q_ur.resize(Np, nnr);
  for (long u = 0; u < Np; ++u)
    for (long ir = 0; ir < nnr; ++ir) Xi_q_ur(u, ir) = Xi_f(u, ir);

  // --- FFT paths ---
  nda::array<ComplexType, 4> fft_buf(1, N1, N2, N3);
  auto fft_buf_1d = nda::reshape(fft_buf, std::array<long, 1>{nnr});
  math::nda::fft<true> F(fft_buf);

  // Phase factors e^{-iq·r} and e^{+iq·r} on full grid.
  nda::array<ComplexType, 1> phase_mq(nnr);
  nda::array<ComplexType, 1> phase_pq(nnr);
  {
    nda::stack_array<double, 3> q_neg;
    for (int a = 0; a < 3; ++a) q_neg(a) = -q_cart(a);
    Timer.start("phase_factor");
    utils::rspace_phase_factor(lattv_mf, q_neg,
        nda::reshape(phase_mq, std::array<long,3>{N1, N2, N3}));
    utils::rspace_phase_factor(lattv_mf, q_cart,
        nda::reshape(phase_pq, std::array<long,3>{N1, N2, N3}));
    Timer.stop("phase_factor");
  }

  Xi_q_uG.resize(Np, nG_red);
  Xi_bar_q_uG.resize(Np, nG_red);

  // Standard FFT path: f̃(G) = FFT[e^{-iq·r} · f(r)](G)
  Timer.start("fft_forward_std");
  for (long u = 0; u < Np; ++u) {
    for (long ir = 0; ir < nnr; ++ir)
      fft_buf_1d(ir) = phase_mq(ir) * Xi_q_ur(u, ir);
    F.forward(fft_buf);
    for (long g = 0; g < nG_red; ++g)
      Xi_q_uG(u, g) = fft_buf_1d(rho_g.gv_to_fft(g));
  }
  Timer.stop("fft_forward_std");

  // Conjugate FFT path: f̄(G) = FFT[e^{+iq·r} · conj(f(r))](G)
  // (allows recovery of f̃(-G) = [f̄(G)]^* without G-reflection lookup.)
  Timer.start("fft_forward_conj");
  for (long u = 0; u < Np; ++u) {
    for (long ir = 0; ir < nnr; ++ir)
      fft_buf_1d(ir) = phase_pq(ir) * std::conj(Xi_q_ur(u, ir));
    F.forward(fft_buf);
    for (long g = 0; g < nG_red; ++g)
      Xi_bar_q_uG(u, g) = fft_buf_1d(rho_g.gv_to_fft(g));
  }
  Timer.stop("fft_forward_conj");

  Timer.stop("solve_xi_for_q");
}

// ----------------------------------------------------------------------------
// Step 6: Solve C_q · δΞ^q = δZ^q - δC^q · Ξ^{q+Q}, then FFT δΞ^q to G.
//
// δΞ^q has Bloch momentum q + Q, so the FFT phase is e^{∓i(q+Q)·r}.
// Reuses the LU factorization from step 5 (C_f_lu, ipiv).
// ----------------------------------------------------------------------------
void lr_thc_new::solve_dxi_for_q(
    nda::matrix<ComplexType, nda::F_layout> const& C_f_lu,
    nda::array<int, 1> const& ipiv,
    nda::array<ComplexType, 2> const& DeltaZ_q,
    nda::array<ComplexType, 2> const& DeltaC_q,
    nda::array<ComplexType, 2> const& Xi_qpQ_ur,
    nda::stack_array<double, 3> const& q_pQ_cart,
    nda::array<ComplexType, 2>& DeltaXi_q_ur,
    nda::array<ComplexType, 2>& DeltaXi_q_uG,
    nda::array<ComplexType, 2>& DeltaXi_bar_q_uG)
{
  Timer.start("solve_dxi_for_q");
  long const Np  = C_f_lu.shape(0);
  long const nnr = DeltaZ_q.shape(1);

  auto const& mf = *_reader.MF();
  auto const& rho_g = _reader.rho_g();
  long const nG_red = rho_g.size();
  auto mesh = rho_g.mesh();
  long const N1 = mesh(0), N2 = mesh(1), N3 = mesh(2);
  auto lattv_mf = mf.lattv();

  // --- RHS: rhs = δZ_q - δC_q · Ξ^{q+Q} (real space). ---
  nda::matrix<ComplexType, nda::F_layout> rhs_f(Np, nnr);
  for (long u = 0; u < Np; ++u)
    for (long ir = 0; ir < nnr; ++ir) rhs_f(u, ir) = DeltaZ_q(u, ir);
  // rhs_f -= DeltaC_q · Xi^{q+Q}
  nda::blas::gemm(ComplexType(-1.0), DeltaC_q, Xi_qpQ_ur,
                  ComplexType(1.0), rhs_f);

  // --- LU solve (reuse factor from step 5). ---
  Timer.start("lu_solve");
  int info_s = nda::lapack::getrs(C_f_lu, rhs_f, ipiv);
  Timer.stop("lu_solve");
  utils::check(info_s == 0, "lr_thc_new::solve_dxi_for_q: getrs failed, info = {}", info_s);

  // Copy δΞ^q(μ, r) into output real-space buffer.
  DeltaXi_q_ur.resize(Np, nnr);
  for (long u = 0; u < Np; ++u)
    for (long ir = 0; ir < nnr; ++ir) DeltaXi_q_ur(u, ir) = rhs_f(u, ir);

  // --- FFT paths with Bloch (q+Q) phases. ---
  nda::array<ComplexType, 4> fft_buf(1, N1, N2, N3);
  auto fft_buf_1d = nda::reshape(fft_buf, std::array<long, 1>{nnr});
  math::nda::fft<true> F(fft_buf);

  nda::array<ComplexType, 1> phase_mqpQ(nnr);
  nda::array<ComplexType, 1> phase_pqpQ(nnr);
  {
    nda::stack_array<double, 3> qpQ_neg;
    for (int a = 0; a < 3; ++a) qpQ_neg(a) = -q_pQ_cart(a);
    Timer.start("phase_factor");
    utils::rspace_phase_factor(lattv_mf, qpQ_neg,
        nda::reshape(phase_mqpQ, std::array<long,3>{N1, N2, N3}));
    utils::rspace_phase_factor(lattv_mf, q_pQ_cart,
        nda::reshape(phase_pqpQ, std::array<long,3>{N1, N2, N3}));
    Timer.stop("phase_factor");
  }

  DeltaXi_q_uG.resize(Np, nG_red);
  DeltaXi_bar_q_uG.resize(Np, nG_red);

  Timer.start("fft_forward_std");
  for (long u = 0; u < Np; ++u) {
    for (long ir = 0; ir < nnr; ++ir)
      fft_buf_1d(ir) = phase_mqpQ(ir) * DeltaXi_q_ur(u, ir);
    F.forward(fft_buf);
    for (long g = 0; g < nG_red; ++g)
      DeltaXi_q_uG(u, g) = fft_buf_1d(rho_g.gv_to_fft(g));
  }
  Timer.stop("fft_forward_std");

  Timer.start("fft_forward_conj");
  for (long u = 0; u < Np; ++u) {
    for (long ir = 0; ir < nnr; ++ir)
      fft_buf_1d(ir) = phase_pqpQ(ir) * std::conj(DeltaXi_q_ur(u, ir));
    F.forward(fft_buf);
    for (long g = 0; g < nG_red; ++g)
      DeltaXi_bar_q_uG(u, g) = fft_buf_1d(rho_g.gv_to_fft(g));
  }
  Timer.stop("fft_forward_conj");

  Timer.stop("solve_dxi_for_q");
}

// ----------------------------------------------------------------------------
// Step 7: drive steps 3–6 over all q' on the BZ k-grid.
//
// Pass A builds Ξ at every q' and caches the real-space Ξ_ur, the LU factor,
// ipiv, and the k-q' index map. Pass B builds δΞ at every q' by consulting
// the cached Ξ at q' + Q.
// ----------------------------------------------------------------------------
void lr_thc_new::solve_all_q(
    nda::array<ComplexType, 4> const& T_skur,
    nda::array<ComplexType, 4> const& DeltaT_pQ_skur,
    nda::array<ComplexType, 4> const& DeltaT_mQ_skur,
    nda::array<double, 2> const& kpts_crys,
    nda::array<long, 1> const& r_P,
    nda::array<double, 1> const& q_pert_cryst,
    nda::array<ComplexType, 3>& Xi_uG_all,
    nda::array<ComplexType, 3>& Xi_bar_uG_all,
    nda::array<ComplexType, 3>& DeltaXi_uG_all,
    nda::array<ComplexType, 3>& DeltaXi_bar_uG_all)
{
  Timer.start("solve_all_q");
  long const nq  = kpts_crys.shape(0);
  long const Np  = T_skur.shape(2);
  long const nnr = T_skur.shape(3);

  auto const& mf = *_reader.MF();
  auto const& rho_g = _reader.rho_g();
  long const nG_red = rho_g.size();
  auto recv_mf = mf.recv();

  Xi_uG_all.resize        (nq, Np, nG_red);
  Xi_bar_uG_all.resize    (nq, Np, nG_red);
  DeltaXi_uG_all.resize   (nq, Np, nG_red);
  DeltaXi_bar_uG_all.resize(nq, Np, nG_red);

  // --- Pass A: build Ξ at every q'. Cache real-space Ξ, LU(C), ipiv, kmq. ---
  std::vector<nda::array<ComplexType, 2>> Xi_ur_cache(nq);
  std::vector<nda::matrix<ComplexType, nda::F_layout>> C_lu_cache(nq);
  std::vector<nda::array<int, 1>> ipiv_cache(nq);
  std::vector<nda::array<int, 1>> kmq_cache(nq);

  for (long iq = 0; iq < nq; ++iq) {
    nda::array<double, 1> q_vec_cryst(3);
    for (int a = 0; a < 3; ++a) q_vec_cryst(a) = kpts_crys(iq, a);

    nda::stack_array<double, 3> q_cart;
    for (int a = 0; a < 3; ++a)
      q_cart(a) = q_vec_cryst(0) * recv_mf(0, a)
                + q_vec_cryst(1) * recv_mf(1, a)
                + q_vec_cryst(2) * recv_mf(2, a);

    nda::array<ComplexType, 2> Z_q;
    nda::array<ComplexType, 2> C_q;
    nda::array<int, 1>         kmq_map;
    build_Z_C_for_q(T_skur, kpts_crys, r_P, q_vec_cryst,
                    Z_q, C_q, kmq_map);

    nda::array<ComplexType, 2> Xi_q_ur, Xi_q_uG, Xi_bar_q_uG;
    nda::matrix<ComplexType, nda::F_layout> C_f_lu;
    nda::array<int, 1> ipiv;
    solve_xi_for_q(C_q, Z_q, q_cart,
                   Xi_q_ur, Xi_q_uG, Xi_bar_q_uG,
                   C_f_lu, ipiv);

    Xi_uG_all    (iq, nda::ellipsis{}) = Xi_q_uG;
    Xi_bar_uG_all(iq, nda::ellipsis{}) = Xi_bar_q_uG;

    Xi_ur_cache[iq] = std::move(Xi_q_ur);
    C_lu_cache [iq] = std::move(C_f_lu);
    ipiv_cache [iq] = std::move(ipiv);
    kmq_cache  [iq] = std::move(kmq_map);
  }

  // --- Pass B: build δΞ at every q' using cached Ξ at q' + Q. ---
  nda::array<int, 1> qpQ_map(nq);
  utils::calculate_kpq_map(kpts_crys, q_pert_cryst, qpQ_map);

  for (long iq = 0; iq < nq; ++iq) {
    nda::array<double, 1> q_vec_cryst(3);
    for (int a = 0; a < 3; ++a) q_vec_cryst(a) = kpts_crys(iq, a);

    nda::stack_array<double, 3> q_pQ_cart;
    for (int a = 0; a < 3; ++a)
      q_pQ_cart(a) = (q_vec_cryst(0) + q_pert_cryst(0)) * recv_mf(0, a)
                   + (q_vec_cryst(1) + q_pert_cryst(1)) * recv_mf(1, a)
                   + (q_vec_cryst(2) + q_pert_cryst(2)) * recv_mf(2, a);

    nda::array<ComplexType, 2> DeltaZ_q, DeltaC_q;
    build_DeltaZ_DeltaC_for_q(T_skur, DeltaT_pQ_skur, DeltaT_mQ_skur,
                              kmq_cache[iq], r_P, DeltaZ_q, DeltaC_q);

    long const iq_qpQ = qpQ_map(iq);
    nda::array<ComplexType, 2> DeltaXi_q_ur, DeltaXi_q_uG, DeltaXi_bar_q_uG;
    solve_dxi_for_q(C_lu_cache[iq], ipiv_cache[iq],
                    DeltaZ_q, DeltaC_q, Xi_ur_cache[iq_qpQ], q_pQ_cart,
                    DeltaXi_q_ur, DeltaXi_q_uG, DeltaXi_bar_q_uG);

    DeltaXi_uG_all    (iq, nda::ellipsis{}) = DeltaXi_q_uG;
    DeltaXi_bar_uG_all(iq, nda::ellipsis{}) = DeltaXi_bar_q_uG;
  }

  Timer.stop("solve_all_q");
}

// ----------------------------------------------------------------------------
// Step 8: assemble δV^q for one internal q.
//
// theory note (last eq.) in G-space with conjugate-FFT identity:
//   term1_{μν} = (1/Vol) · Σ_G [DeltaXi_bar^{-q-Q}_μ(G)]^* · v(|q+G|)   · Ξ^q_ν(G)
//   term2_{μν} = (1/Vol) · Σ_G [Xi_bar^{-q-Q}_μ(G)]^*      · v(|q+Q+G|) · DeltaXi^q_ν(G)
// μ on the left (at r), ν on the right (at r'); same index order on both lines.
// ----------------------------------------------------------------------------
void lr_thc_new::compute_delta_V_for_q(
    long iq,
    long iq_mqmQ,
    nda::array<ComplexType, 3> const& Xi_uG_all,
    nda::array<ComplexType, 3> const& Xi_bar_uG_all,
    nda::array<ComplexType, 3> const& DeltaXi_uG_all,
    nda::array<ComplexType, 3> const& DeltaXi_bar_uG_all,
    nda::stack_array<double, 3> const& q_cart,
    nda::stack_array<double, 3> const& q_pQ_cart,
    nda::array<ComplexType, 2>& dV_q)
{
  Timer.start("compute_delta_V_for_q");
  long const Np     = Xi_uG_all.shape(1);
  long const nG_red = Xi_uG_all.shape(2);

  auto const& mf = *_reader.MF();
  auto const& rho_g = _reader.rho_g();
  auto lattv_mf = mf.lattv();

  // Slices at iq and at iq_mqmQ (index of -q - Q in the BZ k-grid).
  auto Xi_q             = Xi_uG_all        (iq,      nda::range::all, nda::range::all);
  auto DeltaXi_q        = DeltaXi_uG_all   (iq,      nda::range::all, nda::range::all);
  auto Xi_bar_mqmQ      = Xi_bar_uG_all    (iq_mqmQ, nda::range::all, nda::range::all);
  auto DeltaXi_bar_mqmQ = DeltaXi_bar_uG_all(iq_mqmQ, nda::range::all, nda::range::all);

  // --- Coulomb factors v(|q+G|) and v(|q+Q+G|). ---
  // vG().evaluate(V, lattv, gv, kp, kq) computes V(G) = v(|G + (kp - kq)|).
  // For v(|q + G|):     pass kp = q_cart, kq = 0 (so dk = +q, giving v(|G + q|)).
  // For v(|q + Q + G|): pass kp = q+Q,    kq = 0 (so dk = +(q+Q), giving v(|G + q + Q|)).
  // (Note: passing v_zero, q_cart would give v(|G - q|), which is NOT equal to
  //  v(|q + G|) in general — only related by G → -G relabel, but the new code
  //  uses Ξ at fixed G-index without that relabel; see lr_thc_theory.txt.)
  memory::unified_array<ComplexType, 1> v_q_G  (rho_g.size());
  memory::unified_array<ComplexType, 1> v_qpQ_G(rho_g.size());
  nda::array<RealType, 1> v_zero = {0.0, 0.0, 0.0};
  {
    nda::array<RealType, 1> q_cart_a(3);
    nda::array<RealType, 1> qpQ_cart_a(3);
    for (int a = 0; a < 3; ++a) {
      q_cart_a  (a) = q_cart(a);
      qpQ_cart_a(a) = q_pQ_cart(a);
    }
    Timer.start("v_evaluate");
    _reader.vG().evaluate(v_q_G,   lattv_mf, rho_g.g_vectors(), q_cart_a,   v_zero);
    _reader.vG().evaluate(v_qpQ_G, lattv_mf, rho_g.g_vectors(), qpQ_cart_a, v_zero);
    Timer.stop("v_evaluate");
  }

  // --- term1: (1/Vol) Σ_G [DeltaXi_bar_mqmQ(μ, G)]^* · v_q_G(G) · Xi_q(ν, G) ---
  nda::array<ComplexType, 2> tmp1(Np, nG_red);
  for (long u = 0; u < Np; ++u)
    for (long g = 0; g < nG_red; ++g)
      tmp1(u, g) = std::conj(DeltaXi_bar_mqmQ(u, g)) * v_q_G(g);

  nda::array<ComplexType, 2> term1(Np, Np);
  term1() = ComplexType(0.0);
  Timer.start("gemm_term1");
  // term1(μ, ν) = Σ_G tmp1(μ, G) · Xi_q(ν, G) = tmp1 · Xi_q^T
  nda::blas::gemm(ComplexType(1.0), tmp1, nda::transpose(Xi_q),
                  ComplexType(0.0), term1);
  Timer.stop("gemm_term1");

  // --- term2: (1/Vol) Σ_G [Xi_bar_mqmQ(μ, G)]^* · v_qpQ_G(G) · DeltaXi_q(ν, G) ---
  nda::array<ComplexType, 2> tmp2(Np, nG_red);
  for (long u = 0; u < Np; ++u)
    for (long g = 0; g < nG_red; ++g)
      tmp2(u, g) = std::conj(Xi_bar_mqmQ(u, g)) * v_qpQ_G(g);

  nda::array<ComplexType, 2> term2(Np, Np);
  term2() = ComplexType(0.0);
  Timer.start("gemm_term2");
  nda::blas::gemm(ComplexType(1.0), tmp2, nda::transpose(DeltaXi_q),
                  ComplexType(0.0), term2);
  Timer.stop("gemm_term2");

  // --- Sum + (1/Vol) scaling (match existing build_dV_from_xi convention). ---
  ComplexType const scale = ComplexType(1.0) / mf.volume();
  dV_q.resize(Np, Np);
  dV_q = scale * (term1 + term2);

  Timer.stop("compute_delta_V_for_q");
}

// ----------------------------------------------------------------------------
// Print timers as a hierarchy reflecting the call structure.
//   compute_delta_V_new
//     read_psi_and_dpsi
//       io_wfc / io_dpsi / io_dpsi_adj / phase_factor
//     compute_T_and_DeltaT
//     solve_all_q
//       build_Z_C_for_q             → accumulate_Z
//       build_DeltaZ_DeltaC_for_q   → accumulate_dZ
//       solve_xi_for_q              → lu_factor / lu_solve / fft_forward_*
//       solve_dxi_for_q             → lu_solve / fft_forward_*
//     compute_delta_V_for_q
//       v_evaluate / gemm_term1 / gemm_term2
// ----------------------------------------------------------------------------
void lr_thc_new::print_timers()
{
  // (label, indent_level)
  std::vector<std::pair<char const*, int>> entries = {
    {"compute_delta_V_new",          0},
    {  "read_psi_and_dpsi",          1},
    {    "io_wfc",                   2},
    {    "io_dpsi",                  2},
    {    "io_dpsi_adj",              2},
    {    "phase_factor",             2},
    {  "compute_T_and_DeltaT",       1},
    {  "solve_all_q",                1},
    {    "build_Z_C_for_q",          2},
    {      "accumulate_Z",           3},
    {    "build_DeltaZ_DeltaC_for_q",2},
    {      "accumulate_dZ",          3},
    {    "solve_xi_for_q",           2},
    {      "lu_factor",              3},
    {      "lu_solve",                3},
    {      "fft_forward_std",        3},
    {      "fft_forward_conj",       3},
    {    "solve_dxi_for_q",          2},
    {  "compute_delta_V_for_q",      1},
    {    "v_evaluate",               2},
    {    "gemm_term1",               2},
    {    "gemm_term2",               2},
  };
  app_log(2, "\n  LR_THC_NEW timers");
  app_log(2, "  -----------------");
  for (auto& [name, depth] : entries) {
    std::string indent(2 + 2 * depth, ' ');
    std::string label = indent + name;
    app_log(2, "  {0:<36s}: {1:8.3f} sec  {2:4d} calls",
            label, Timer.elapsed(name), Timer.number_of_calls(name));
  }
  app_log(2, "");
}

// ----------------------------------------------------------------------------
// Step 9: driver. Wires steps 1–8.
// ----------------------------------------------------------------------------
nda::array<ComplexType, 3> lr_thc_new::compute_delta_V_new(
    std::string const& Deltapsi_prefix,
    std::string const& Deltapsi_adj_prefix,
    nda::array<double, 1> const& q_pert_cryst)
{
  Timer.start("compute_delta_V_new");

  utils::check(q_pert_cryst.shape(0) == 3,
               "lr_thc_new::compute_delta_V_new: q_pert_cryst must have size 3.");

  auto const& mf = *_reader.MF();

  // --- Restrictions (single mode, single Q, no symmetry). ---
  utils::check(mf.npol() == 1,
               "lr_thc_new::compute_delta_V_new: non-collinear (npol > 1) not supported.");
  utils::check(mf.nkpts() == mf.nkpts_ibz(),
               "lr_thc_new::compute_delta_V_new: requires nkpts == nkpts_ibz "
               "(no k-symmetry).");
  utils::check(mf.orb_on_fft_grid(),
               "lr_thc_new::compute_delta_V_new: requires mf->orb_on_fft_grid() == true.");
  // The returned δV is indexed by iq in mf.kpts_crystal() order. We assume the
  // BZ q-grid equals the k-grid (typical CoQuí construction). Verify:
  utils::check(mf.nqpts() == mf.nkpts(),
               "lr_thc_new::compute_delta_V_new: requires nqpts == nkpts "
               "(BZ q-grid must equal the k-grid). got nqpts={}, nkpts={}.",
               mf.nqpts(), mf.nkpts());
  utils::check(mf.nqpts_ibz() == mf.nkpts_ibz(),
               "lr_thc_new::compute_delta_V_new: requires nqpts_ibz == nkpts_ibz "
               "(no q-symmetry). got nqpts_ibz={}, nkpts_ibz={}.",
               mf.nqpts_ibz(), mf.nkpts_ibz());

  long const nspin = mf.nspin();
  long const nkpts = mf.nkpts();
  long const nbnd  = mf.nbnd();

  auto const& rho_g = _reader.rho_g();
  auto mesh = rho_g.mesh();
  nda::stack_array<long, 3> fft_mesh;
  for (int i = 0; i < 3; ++i) fft_mesh(i) = mesh(i);
  long const nnr = rho_g.nnr();

  auto const& r_P = _reader.ri();
  long const Np = r_P.shape(0);

  auto recv_mf = mf.recv();
  auto kpts_crys_mf = mf.kpts_crystal();
  // Local copy as plain nda::array<double, 2> for utility helpers.
  nda::array<double, 2> kpts_crys(nkpts, 3);
  for (long ik = 0; ik < nkpts; ++ik)
    for (int a = 0; a < 3; ++a) kpts_crys(ik, a) = kpts_crys_mf(ik, a);

  // Q in Cartesian (2π included)
  nda::stack_array<double, 3> Q_cart;
  for (int a = 0; a < 3; ++a)
    Q_cart(a) = q_pert_cryst(0) * recv_mf(0, a)
              + q_pert_cryst(1) * recv_mf(1, a)
              + q_pert_cryst(2) * recv_mf(2, a);

  // Local copy of r_P pivots as plain nda::array (utility consumers expect this).
  nda::array<long, 1> r_P_local(Np);
  for (long u = 0; u < Np; ++u) r_P_local(u) = r_P(u);

  // kmQ_map: index of (k - Q) for each k. Needed for adj-file shuffle.
  nda::array<double, 1> Q_neg_cryst(3);
  for (int a = 0; a < 3; ++a) Q_neg_cryst(a) = -q_pert_cryst(a);
  nda::array<int, 1> kmQ_map(nkpts);
  utils::calculate_kpq_map(kpts_crys, Q_neg_cryst, kmQ_map);

  // --- Memory estimate for the per-k caches (peak live = step 1 ψ/δψ + step 2 T's). ---
  {
    double const GB = 1024.0 * 1024.0 * 1024.0;
    double mem_psi  = 3.0 * double(nspin) * double(nkpts) * double(nbnd) * double(nnr) * 16.0 / GB;
    double mem_T    = 3.0 * double(nspin) * double(nkpts) * double(Np)   * double(nnr) * 16.0 / GB;
    double mem_peak = mem_psi + mem_T;  // ψ/δψ and T caches coexist during step 2.
    app_log(2, "  lr_thc_new estimated peak host-memory for caches:");
    app_log(2, "    psi/dpsi (3 x nspin x nkpts x nbnd x nnr): {0:8.3f} GB", mem_psi);
    app_log(2, "    T/dT     (3 x nspin x nkpts x Np   x nnr): {0:8.3f} GB", mem_T);
    app_log(2, "    peak (sum during step 2):                  {0:8.3f} GB", mem_peak);
  }

  // --- Step 1: read ψ and δψ caches. ---
  nda::array<ComplexType, 4> psi_skmr        (nspin, nkpts, nbnd, nnr);
  nda::array<ComplexType, 4> Deltapsi_pQ_skmr(nspin, nkpts, nbnd, nnr);
  nda::array<ComplexType, 4> Deltapsi_mQ_skmr(nspin, nkpts, nbnd, nnr);
  read_psi_and_dpsi(Deltapsi_prefix, Deltapsi_adj_prefix, Q_cart, fft_mesh,
                   kmQ_map, psi_skmr, Deltapsi_pQ_skmr, Deltapsi_mQ_skmr);

  // --- Step 2: T, ΔT_pQ, ΔT_mQ caches. ---
  nda::array<ComplexType, 4> T_skur        (nspin, nkpts, Np, nnr);
  nda::array<ComplexType, 4> DeltaT_pQ_skur(nspin, nkpts, Np, nnr);
  nda::array<ComplexType, 4> DeltaT_mQ_skur(nspin, nkpts, Np, nnr);
  compute_T_and_DeltaT(psi_skmr, Deltapsi_pQ_skmr, Deltapsi_mQ_skmr,
                       r_P_local,
                       T_skur, DeltaT_pQ_skur, DeltaT_mQ_skur);

  // ψ / δψ are no longer needed after T-cache is built.
  psi_skmr         = nda::array<ComplexType, 4>{};
  Deltapsi_pQ_skmr = nda::array<ComplexType, 4>{};
  Deltapsi_mQ_skmr = nda::array<ComplexType, 4>{};

  // --- Steps 3–6 (driven by step 7): solve Ξ and δΞ at all q's. ---
  nda::array<ComplexType, 3> Xi_uG_all, Xi_bar_uG_all;
  nda::array<ComplexType, 3> DeltaXi_uG_all, DeltaXi_bar_uG_all;
  solve_all_q(T_skur, DeltaT_pQ_skur, DeltaT_mQ_skur,
              kpts_crys, r_P_local, q_pert_cryst,
              Xi_uG_all, Xi_bar_uG_all, DeltaXi_uG_all, DeltaXi_bar_uG_all);

  // T-caches no longer needed.
  T_skur         = nda::array<ComplexType, 4>{};
  DeltaT_pQ_skur = nda::array<ComplexType, 4>{};
  DeltaT_mQ_skur = nda::array<ComplexType, 4>{};

  // --- mkmQ_map: index of (-k - Q) on the BZ k-grid, per iq.
  // The Coulomb assembly needs the Ξ slice at internal momentum -q - Q.
  // Equivalent to "find ip such that kpts(ip) + kpts(iq) + Q ≡ 0 (mod G)".
  // Same wrap/threshold convention as utils::calculate_kpq_map.
  long const nq = nkpts;
  double const tol2 = 1e-6 * 1e-6;
  nda::array<int, 1> mkmQ_map(nq);
  for (long iq = 0; iq < nq; ++iq) {
    int found = -1;
    for (long ip = 0; ip < nq; ++ip) {
      double d0 = std::abs(kpts_crys(ip, 0) + kpts_crys(iq, 0) + q_pert_cryst(0));
      double d1 = std::abs(kpts_crys(ip, 1) + kpts_crys(iq, 1) + q_pert_cryst(1));
      double d2 = std::abs(kpts_crys(ip, 2) + kpts_crys(iq, 2) + q_pert_cryst(2));
      d0 -= std::floor(d0); d1 -= std::floor(d1); d2 -= std::floor(d2);
      d0 -= std::round(d0); d1 -= std::round(d1); d2 -= std::round(d2);
      if (d0*d0 + d1*d1 + d2*d2 < tol2) { found = (int)ip; break; }
    }
    utils::check(found >= 0,
                 "lr_thc_new::compute_delta_V_new: cannot find -k - Q on the "
                 "BZ k-grid for iq={}.", iq);
    mkmQ_map(iq) = found;
  }

  // --- Step 8: assemble δV^q for every q. ---
  nda::array<ComplexType, 3> out(nq, Np, Np);
  out() = ComplexType(0.0);
  for (long iq = 0; iq < nq; ++iq) {
    nda::stack_array<double, 3> q_cart, q_pQ_cart;
    for (int a = 0; a < 3; ++a) {
      q_cart  (a) = kpts_crys(iq, 0) * recv_mf(0, a)
                  + kpts_crys(iq, 1) * recv_mf(1, a)
                  + kpts_crys(iq, 2) * recv_mf(2, a);
      q_pQ_cart(a) = q_cart(a) + Q_cart(a);
    }
    nda::array<ComplexType, 2> dV_q;
    compute_delta_V_for_q(iq, mkmQ_map(iq),
                          Xi_uG_all, Xi_bar_uG_all,
                          DeltaXi_uG_all, DeltaXi_bar_uG_all,
                          q_cart, q_pQ_cart, dV_q);
    out(iq, nda::ellipsis{}) = dV_q;
  }

  Timer.stop("compute_delta_V_new");
  print_timers();
  return out;
}

} // namespace methods
