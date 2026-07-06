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

#include "lr_thc.hpp"
#include "lr_psi_io.hpp"

#include <cmath>
#include <cstdlib>
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
#include "nda/linalg/eigenelements.hpp"  // C_q hermitian eig path (env-var gated)
#include "itertools/itertools.hpp"       // chunk_range for q-distribution

// Bloch phase conventions in this file
// -------------------------------------
// Real-space arrays carry the Bloch phase: f^{k}(r) = e^{i k·r} f^{k}_{periodic}(r).
// Exceptions:
// - QE wavefunctions and gradients are stored as periodic parts; the Bloch phase
//   is applied at read time. For the adjoint gradient, QE computes
//   Δ^{-Q}ψ^{k+Q}, so the slot for Δ^{-Q}ψ^{kpts(ik)} is filled by reading the
//   file at filename index ik_adj = kmQ_map(ik) (the BZ-folded representative
//   of k - Q) with phase e^{i·kpts(ik_adj)·r} — NOT e^{i(k-Q)·r}, which differs
//   by e^{-i·G_BZ·r} when k-Q wraps.
// - Ξ̃ and δΞ̃ in G-space use the BZ-folded Bloch phase: δΞ̃^q_uG is the FT
//   of e^{-i q'·r} δΞ^q(r) with q' = (q+Q) mod G.

namespace methods {

// ----------------------------------------------------------------------------
// Step 1: read ψ and (+Q, -Q) δψ on the full FFT grid for all (is, ik).
//
// Outputs:
//   psi_skmr(is, ik, m, r)         = ψ^{k_ik}_m(r) · e^{i k·r}
//   Deltapsi_pQ_skmr(is, ik, m, r) = δ^{+Q} ψ^{k_ik}_m(r) · e^{i(k+Q)·r}
//   Deltapsi_mQ_skmr(is, ik, m, r) = δ^{-Q} ψ^{k_ik}_m(r) · e^{i·kpts(ik_adj)·r}
//                                    where kpts(ik_adj) = k-Q+G. See the
//                                    top-of-file note on the adj-file phase.
// ----------------------------------------------------------------------------
void lr_thc::read_psi_and_dpsi(
    std::string const& Deltapsi_prefix,
    std::string const& Deltapsi_adj_prefix,
    nda::stack_array<long, 3> const& fft_mesh,
    nda::array_view<ComplexType, 4> psi_skmr,
    nda::array_view<ComplexType, 4> Deltapsi_pQ_skmr,
    nda::array_view<ComplexType, 4> Deltapsi_mQ_skmr)
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

  // Distribute (is, ik) round-robin across node_comm; each rank populates its
  // assigned slabs of the per-node shared cache. Caller is responsible for
  // having zero-initialized the buffers (shared_array ctor does this) and for
  // the post-loop node_sync().
  long const node_size = _mpi->node_comm.size();
  long const node_rank = _mpi->node_comm.rank();
  utils::check(node_size > 0,
               "lr_thc::read_psi_and_dpsi: node_comm.size() must be > 0.");

  for (long is = 0; is < nspin; ++is) {
    for (long ik = 0; ik < nkpts; ++ik) {
      if (((is * nkpts + ik) % node_size) != node_rank) continue;
      long const ik_kmQ = _kqpoint_maps.kmQ_map(ik);

      // --- ψ: Bloch phase e^{i k·r}.
      Timer.start("io_wfc");
      auto wfc = detail::read_wfc_k(mf, is, ik, fft_mesh);
      Timer.stop("io_wfc");
      utils::check(wfc.nbnd >= nbnd,
                   "lr_thc::read_psi_and_dpsi: wfc at (is={},ik={}) has {} bands, "
                   "expected >= {}.", is, ik, wfc.nbnd, nbnd);
      {
        nda::array<ComplexType, 1> phase_full(nnr);
        Timer.start("phase_factor");
        utils::rspace_phase_factor(lattv_mf, kpts_mf(ik, nda::range::all),
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

      // --- +Q δψ: Bloch phase e^{i(k+Q)·r}.
      nda::stack_array<double, 3> kpQ_cart;
      for (int a = 0; a < 3; ++a) kpQ_cart(a) = kpts_mf(ik, a) + _kqpoint_maps.Q_cart(a);
      Timer.start("io_dpsi");
      auto dpsi_pQ = detail::read_Deltapsi_k(Deltapsi_prefix, ik, fft_mesh);
      Timer.stop("io_dpsi");
      utils::check(dpsi_pQ.nbnd >= nbnd,
                   "lr_thc::read_psi_and_dpsi: +Q dpsi at ik={} has {} bands, "
                   "expected >= {}.", ik, dpsi_pQ.nbnd, nbnd);
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

      // --- -Q δψ: read adj file at ik_kmQ; Bloch phase e^{i·kpts(ik_kmQ)·r}
      // ik-th file of QE contains δ^{-Q}ψ^{k+Q}, while we want δ^{-Q}ψ^{k} at the ik slot.
      // So we read from kmQ_map(ik)-th QE file.
      Timer.start("io_dpsi_adj");
      auto dpsi_mQ = detail::read_Deltapsi_k(Deltapsi_adj_prefix, ik_kmQ, fft_mesh);
      Timer.stop("io_dpsi_adj");
      utils::check(dpsi_mQ.nbnd >= nbnd,
                   "lr_thc::read_psi_and_dpsi: -Q dpsi_adj at ik={} has {} bands, "
                   "expected >= {}.", ik_kmQ, dpsi_mQ.nbnd, nbnd);
      {
        nda::array<ComplexType, 1> phase_full(nnr);
        Timer.start("phase_factor");
        utils::rspace_phase_factor(lattv_mf, kpts_mf(ik_kmQ, nda::range::all),
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
// Step 2: T, ΔT_pQ, ΔT_mQ per-k caches.
//
//   T^k(μ, r)     = Σ_m conj(ψ^k_m(r_P(μ))) · ψ^k_m(r)
//   ΔT_pQ^k(μ, r) = Σ_m [ conj(δ^{-Q}ψ^{k+Q}_m(r_P(μ))) · ψ^{k+Q}_m(r)    (bra)
//                       +  conj(ψ^k_m(r_P(μ)))         · δ^{+Q}ψ^k_m(r) ] (ket)
//   ΔT_mQ^k(μ, r) = Σ_m [ conj(δ^{+Q}ψ^{k-Q}_m(r_P(μ))) · ψ^{k-Q}_m(r)    (bra)
//                       +  conj(ψ^k_m(r_P(μ)))         · δ^{-Q}ψ^k_m(r) ] (ket)
//
// Bra side reads ψ at the BZ-folded k±Q index so that each ΔT carries a single
// r-Bloch momentum (k±Q). Theory note: docs/lr_thc_theory.txt.
// ----------------------------------------------------------------------------
void lr_thc::compute_T_and_DeltaT(
    nda::array_view<ComplexType, 4> const psi_skmr,
    nda::array_view<ComplexType, 4> const Deltapsi_pQ_skmr,
    nda::array_view<ComplexType, 4> const Deltapsi_mQ_skmr,
    nda::array<long, 1> const& r_P,
    nda::array_view<ComplexType, 4> T_skur,
    nda::array_view<ComplexType, 4> DeltaT_pQ_skur,
    nda::array_view<ComplexType, 4> DeltaT_mQ_skur)
{
  Timer.start("compute_T_and_DeltaT");
  long const nspin = psi_skmr.shape(0);
  long const nkpts = psi_skmr.shape(1);
  long const nbnd  = psi_skmr.shape(2);
  long const Np    = r_P.shape(0);

  // T / ΔT outputs live in node-shared memory and are already zeroed by
  // shared_array's constructor; each rank populates only its owned (is, ik)
  // slabs below.

  nda::array<ComplexType, 2> _psi_k_pivot(nbnd, Np),
                             _Deltapsi_mQ_kpQ_pivot(nbnd, Np),
                             _Deltapsi_pQ_kmQ_pivot(nbnd, Np);

  // Distribute (is, ik) round-robin across node_comm using the same partition
  // as step 1. The step-1 → step-2 node_sync() in compute_delta_V ensures
  // every (is, ik) slot in psi_*_skmr is fully populated before any rank reads
  // it here at ik, ik_kpQ, or ik_kmQ.
  long const node_size = _mpi->node_comm.size();
  long const node_rank = _mpi->node_comm.rank();
  utils::check(node_size > 0,
               "lr_thc::compute_T_and_DeltaT: node_comm.size() must be > 0.");

  for (long is = 0; is < nspin; ++is) {
    for (long ik = 0; ik < nkpts; ++ik) {
      if (((is * nkpts + ik) % node_size) != node_rank) continue;
      long const ik_kpQ = _kqpoint_maps.kpQ_map(ik);
      long const ik_kmQ = _kqpoint_maps.kmQ_map(ik);

      // Build pivots: pivot(m, u) = full(m, r_P(u)).
      for (long m = 0; m < nbnd; ++m)
        for (long u = 0; u < Np; ++u) {
          _psi_k_pivot            (m, u) = psi_skmr        (is, ik,     m, r_P(u));
          _Deltapsi_mQ_kpQ_pivot(m, u) = Deltapsi_mQ_skmr(is, ik_kpQ, m, r_P(u));  // bra of ΔT_pQ
          _Deltapsi_pQ_kmQ_pivot(m, u) = Deltapsi_pQ_skmr(is, ik_kmQ, m, r_P(u));  // bra of ΔT_mQ
        }

      auto _psi_k         = psi_skmr        (is, ik,     nda::range::all, nda::range::all);
      auto _psi_kpQ       = psi_skmr        (is, ik_kpQ, nda::range::all, nda::range::all);
      auto _psi_kmQ       = psi_skmr        (is, ik_kmQ, nda::range::all, nda::range::all);
      auto _Deltapsi_pQ_k = Deltapsi_pQ_skmr(is, ik,     nda::range::all, nda::range::all);
      auto _Deltapsi_mQ_k = Deltapsi_mQ_skmr(is, ik,     nda::range::all, nda::range::all);
      auto _T_k           = T_skur          (is, ik,     nda::range::all, nda::range::all);
      auto _DeltaT_pQ_k   = DeltaT_pQ_skur  (is, ik,     nda::range::all, nda::range::all);
      auto _DeltaT_mQ_k   = DeltaT_mQ_skur  (is, ik,     nda::range::all, nda::range::all);

      // T^k = ψ_pivot^H · ψ
      nda::blas::gemm(ComplexType(1.0), nda::dagger(_psi_k_pivot), _psi_k,
                      ComplexType(0.0), _T_k);

      // ΔT_pQ = (δ^{-Q}ψ_pivot at k+Q)^H · ψ^{k+Q}  +  ψ_pivot^H · δ^{+Q}ψ^k
      nda::blas::gemm(ComplexType(1.0), nda::dagger(_Deltapsi_mQ_kpQ_pivot), _psi_kpQ,
                      ComplexType(0.0), _DeltaT_pQ_k);
      nda::blas::gemm(ComplexType(1.0), nda::dagger(_psi_k_pivot), _Deltapsi_pQ_k,
                      ComplexType(1.0), _DeltaT_pQ_k);

      // ΔT_mQ = (δ^{+Q}ψ_pivot at k-Q)^H · ψ^{k-Q}  +  ψ_pivot^H · δ^{-Q}ψ^k
      nda::blas::gemm(ComplexType(1.0), nda::dagger(_Deltapsi_pQ_kmQ_pivot), _psi_kmQ,
                      ComplexType(0.0), _DeltaT_mQ_k);
      nda::blas::gemm(ComplexType(1.0), nda::dagger(_psi_k_pivot), _Deltapsi_mQ_k,
                      ComplexType(1.0), _DeltaT_mQ_k);
    }
  }

  Timer.stop("compute_T_and_DeltaT");
}

// ----------------------------------------------------------------------------
// Step 3: Z, C for one internal q (k-q convention).
//   Z^q_{μr} = Σ_k conj(T^{k-q}_{μr}) · T^k_{μr}
//   C^q_{μν} = Z^q_{μ, r_P(ν)}
// kmq_map[ik] = index of (k - q) on the BZ k-grid (obtained via
// calculate_kpq_map with -q).
// ----------------------------------------------------------------------------
void lr_thc::build_Z_C_for_q(
    nda::array_view<ComplexType, 4> const T_skur,
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

  auto conj_prod = nda::map([](ComplexType a, ComplexType b) { return std::conj(a) * b; });
  Timer.start("accumulate_Z");
  for (long is = 0; is < nspin; ++is)
    for (long ik = 0; ik < nkpts; ++ik) {
      long const ikmq = kmq_map(ik);
      Z_q += conj_prod(T_skur(is, ikmq, nda::ellipsis{}), T_skur(is, ik, nda::ellipsis{}));
    }
  Timer.stop("accumulate_Z");

  C_q.resize(Np, Np);
  for (long u = 0; u < Np; ++u)
    for (long v = 0; v < Np; ++v)
      C_q(u, v) = Z_q(u, r_P(v));
  Timer.stop("build_Z_C_for_q");
}

// ----------------------------------------------------------------------------
// Step 4: ΔZ, ΔC for one internal q (kmq_map precomputed by step 3).
//   δZ^q_{μr} = Σ_k [ conj(ΔT_mQ^{k-q}_{μr}) · T^k_{μr}
//                   + conj(T^{k-q}_{μr}) · ΔT_pQ^k_{μr} ]   (Bloch q+Q)
//   δC^q_{μν} = δZ^q_{μ, r_P(ν)}
// ----------------------------------------------------------------------------
void lr_thc::build_DeltaZ_DeltaC_for_q(
    nda::array_view<ComplexType, 4> const T_skur,
    nda::array_view<ComplexType, 4> const DeltaT_pQ_skur,
    nda::array_view<ComplexType, 4> const DeltaT_mQ_skur,
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

  auto conj_prod = nda::map([](ComplexType a, ComplexType b) { return std::conj(a) * b; });
  Timer.start("accumulate_dZ");
  for (long is = 0; is < nspin; ++is)
    for (long ik = 0; ik < nkpts; ++ik) {
      long const ikmq = kmq_map(ik);
      DeltaZ_q += conj_prod(DeltaT_mQ_skur(is, ikmq, nda::ellipsis{}),
                            T_skur        (is, ik,   nda::ellipsis{}))
                + conj_prod(T_skur        (is, ikmq, nda::ellipsis{}),
                            DeltaT_pQ_skur(is, ik,   nda::ellipsis{}));
    }
  Timer.stop("accumulate_dZ");

  DeltaC_q.resize(Np, Np);
  for (long u = 0; u < Np; ++u)
    for (long v = 0; v < Np; ++v)
      DeltaC_q(u, v) = DeltaZ_q(u, r_P(v));
  Timer.stop("build_DeltaZ_DeltaC_for_q");
}

// ----------------------------------------------------------------------------
// Solve C · X = B in-place on B (F-layout, Np rows). Dispatches on `factor`:
//   LU    : getrs with the cached LU + ipiv.
//   Eigen : B ← U · diag(1/λ) · U^H · B.
// ----------------------------------------------------------------------------
void lr_thc::apply_C_inverse(
    C_factor const& factor,
    nda::matrix<ComplexType, nda::F_layout>& B,
    char const* timer_solve_tag)
{
  if (!factor.use_eig) {
    // LU factorization path
    Timer.start(timer_solve_tag);
    int info = nda::lapack::getrs(factor.C_f_lu, B, factor.ipiv);
    Timer.stop(timer_solve_tag);
    utils::check(info == 0,
                 "lr_thc::apply_C_inverse: getrs failed, info = {}", info);
    return;
  }

  // Hermitian eigendecomposition path: B ← U · diag(inv_evals) · U^H · B.
  // Note: C_q is hermitian positive semidefinite by definition.
  long const Np = factor.U.shape(0);
  long const m  = B.shape(1);
  Timer.start(timer_solve_tag);
  nda::matrix<ComplexType, nda::F_layout> W(Np, m);
  nda::blas::gemm(ComplexType(1.0), nda::dagger(factor.U), B,
                  ComplexType(0.0), W);
  for (long i = 0; i < Np; ++i) {
    double const inv_l = factor.inv_evals(i);
    for (long j = 0; j < m; ++j) W(i, j) *= inv_l;
  }
  nda::blas::gemm(ComplexType(1.0), factor.U, W,
                  ComplexType(0.0), B);
  Timer.stop(timer_solve_tag);
}

// ----------------------------------------------------------------------------
// Step 5: solve C_q · Ξ^q = Z^q in real space, then FFT to G-space.
//   Xi_q_ur(μ, r) = real-space Ξ^q (also returned; consumed by step 6).
//   Xi_q_uG(μ, G) = FFT[e^{-iq·r} · Ξ^q(μ, r)](G).
// `factor` holds the C_q factorization for reuse in step 6.
// ----------------------------------------------------------------------------
void lr_thc::solve_xi_for_q(
    nda::array<ComplexType, 2>& C_q,
    nda::array<ComplexType, 2> const& Z_q,
    nda::stack_array<double, 3> const& q_cart,
    nda::array<ComplexType, 2>& Xi_q_ur,
    nda::array<ComplexType, 2>& Xi_q_uG,
    C_factor& factor)
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

  // Check that C_q is hermitian up to round-off, then symmetrize C ← (C + C^H)/2.
  {
    double sym_err = 0.0, norm = 0.0;
    for (long u = 0; u < Np; ++u)
      for (long v = 0; v < Np; ++v) {
        ComplexType const d = C_q(u, v) - std::conj(C_q(v, u));
        sym_err += std::norm(d);
        norm    += std::norm(C_q(u, v));
      }
    double const rel = std::sqrt(sym_err / std::max(norm, 1e-300));
    utils::check(rel < 1e-8,
                 "lr_thc::solve_xi_for_q: C_q is not hermitian, "
                 "||C - C^H|| / ||C|| = {:.3e} (expected ≤ 1e-8).", rel);
  }
  for (long u = 0; u < Np; ++u) {
    for (long v = u + 1; v < Np; ++v) {
      ComplexType const avg = 0.5 * (C_q(u, v) + std::conj(C_q(v, u)));
      C_q(u, v) = avg;
      C_q(v, u) = std::conj(avg);
    }
    C_q(u, u) = ComplexType(C_q(u, u).real(), 0.0);
  }

  // For solving C_q · Ξ^q = Z^q, we have two paths:
  // - LU factorization: faster factorization, less stable.
  // - Hermitian eigendecomposition: slower factorization, more stable.
  // LU by default; COQUI_LR_THC_USE_HERMITIAN_EIG=1 switches to heev.
  bool use_eig = false;
  if (char const* env_val = std::getenv("COQUI_LR_THC_USE_HERMITIAN_EIG")) {
    std::string val(env_val);
    use_eig = !(val.empty() || val == "0" || val == "false" || val == "FALSE");
  }
  factor.use_eig = use_eig;

  if (!use_eig) {
    factor.C_f_lu.resize(Np, Np);
    for (long u = 0; u < Np; ++u)
      for (long v = 0; v < Np; ++v) factor.C_f_lu(u, v) = C_q(u, v);
    factor.ipiv.resize(Np);
    Timer.start("xi_lu_factor");
    int info_f = nda::lapack::getrf(factor.C_f_lu, factor.ipiv);
    Timer.stop("xi_lu_factor");
    utils::check(info_f == 0,
                 "lr_thc::solve_xi_for_q: getrf failed, info = {}", info_f);
  } else {
    Timer.start("xi_eig_factor");
    nda::matrix<ComplexType, nda::F_layout> C_f(Np, Np);
    for (long u = 0; u < Np; ++u)
      for (long v = 0; v < Np; ++v) C_f(u, v) = C_q(u, v);
    auto eig_pair = nda::linalg::eigenelements(C_f);
    auto& evals   = eig_pair.first;
    factor.U      = std::move(eig_pair.second);
    // C is hermitian PSD by construction; fail loudly on λ ≤ 0 rather than
    // flooring. Store 1/λ so the solve path is a plain scale.
    factor.inv_evals.resize(Np);
    for (long i = 0; i < Np; ++i) {
      utils::check(evals(i) > 0.0,
                   "lr_thc::solve_xi_for_q: non-positive eigenvalue "
                   "lam[{}]={:+15.8e}.", i, evals(i));
      factor.inv_evals(i) = 1.0 / evals(i);
    }
    Timer.stop("xi_eig_factor");
    app_log(2,
            "    solve_xi: heev ({}x{}) lam_min={:10.3e} lam_max={:10.3e}",
            Np, Np, evals(0), evals(Np - 1));
  }

  // RHS in F-layout (Np, nnr); overwritten with Ξ^q(μ, r) by the solve.
  nda::matrix<ComplexType, nda::F_layout> Xi_f(Np, nnr);
  Timer.start("xi_copy_rhs");
  for (long u = 0; u < Np; ++u)
    for (long ir = 0; ir < nnr; ++ir) Xi_f(u, ir) = Z_q(u, ir);
  Timer.stop("xi_copy_rhs");

  apply_C_inverse(factor, Xi_f, use_eig ? "xi_eig_solve" : "xi_lu_solve");

  Xi_q_ur.resize(Np, nnr);
  Timer.start("xi_copy_out");
  for (long u = 0; u < Np; ++u)
    for (long ir = 0; ir < nnr; ++ir) Xi_q_ur(u, ir) = Xi_f(u, ir);
  Timer.stop("xi_copy_out");

  // FFT: f̃(G) = FFT[e^{-iq·r} · f(r)](G).
  nda::array<ComplexType, 4> fft_buf(1, N1, N2, N3);
  auto fft_buf_1d = nda::reshape(fft_buf, std::array<long, 1>{nnr});
  math::nda::fft<true> F(fft_buf);

  nda::array<ComplexType, 1> phase_mq(nnr);
  {
    nda::stack_array<double, 3> q_neg;
    for (int a = 0; a < 3; ++a) q_neg(a) = -q_cart(a);
    Timer.start("xi_phase_factor");
    utils::rspace_phase_factor(lattv_mf, q_neg,
        nda::reshape(phase_mq, std::array<long,3>{N1, N2, N3}));
    Timer.stop("xi_phase_factor");
  }

  Xi_q_uG.resize(Np, nG_red);

  Timer.start("xi_fft_forward_std");
  for (long u = 0; u < Np; ++u) {
    for (long ir = 0; ir < nnr; ++ir)
      fft_buf_1d(ir) = phase_mq(ir) * Xi_q_ur(u, ir);
    F.forward(fft_buf);
    for (long g = 0; g < nG_red; ++g)
      Xi_q_uG(u, g) = fft_buf_1d(rho_g.gv_to_fft(g));
  }
  Timer.stop("xi_fft_forward_std");

  Timer.stop("solve_xi_for_q");
}

// ----------------------------------------------------------------------------
// Step 6: build δΞ̃^q entirely in G-space.
//   1) δZ̃(μ, G) = FFT[e^{-i(q+Q)·r} · δZ(μ, r)](G), truncated to rho_g
//   2) rhs̃(G)   = δZ̃(G) − δC · Ξ_uG[iq_qpQ](G)        (G-space gemm)
//   3) C_q · δΞ̃^q = rhs̃                                (G-space solve)
//
// Equivalent to the (gemm-in-r, then forward-FFT) form by FFT linearity and
// the fact that δC acts only on the μ-index. Working in G-space avoids the
// (Np × nnr) rhs_ur and Xi_qpQ_ur buffers and saves one (Np × FFT) per q.
//
// q_pQ_cart is kpts(iq_qpQ) (BZ-folded), NOT literal q+Q — see step 7. The
// gemm against Ξ_uG[iq_qpQ] is therefore consistent: both factors live in the
// same BZ-folded G basis.
// ----------------------------------------------------------------------------
void lr_thc::solve_dxi_for_q(
    C_factor const& factor,
    nda::array<ComplexType, 2> const& DeltaZ_q,
    nda::array<ComplexType, 2> const& DeltaC_q,
    nda::array_view<ComplexType, 2> const Xi_qpQ_uG,
    nda::stack_array<double, 3> const& q_pQ_cart,
    nda::array<ComplexType, 2>& DeltaXi_q_uG)
{
  Timer.start("solve_dxi_for_q");
  long const Np  = DeltaZ_q.shape(0);
  long const nnr = DeltaZ_q.shape(1);

  auto const& mf = *_reader.MF();
  auto const& rho_g = _reader.rho_g();
  long const nG_red = rho_g.size();
  auto mesh = rho_g.mesh();
  long const N1 = mesh(0), N2 = mesh(1), N3 = mesh(2);
  auto lattv_mf = mf.lattv();

  // e^{-i(q+Q)·r}. q_pQ_cart is q' = (q + Q) folded into the kpts BZ
  // (caller passes kpts(_kqpoint_maps.kpQ_map(iq))), so this differs from
  // the literal q+Q phase by an integer G when (q + Q) wraps.
  nda::array<ComplexType, 1> phase_mqpQ(nnr);
  {
    nda::stack_array<double, 3> qpQ_neg;
    for (int a = 0; a < 3; ++a) qpQ_neg(a) = -q_pQ_cart(a);
    Timer.start("dxi_phase_factor");
    utils::rspace_phase_factor(lattv_mf, qpQ_neg,
        nda::reshape(phase_mqpQ, std::array<long,3>{N1, N2, N3}));
    Timer.stop("dxi_phase_factor");
  }

  nda::array<ComplexType, 4> fft_buf(1, N1, N2, N3);
  auto fft_buf_1d = nda::reshape(fft_buf, std::array<long, 1>{nnr});
  math::nda::fft<true> F(fft_buf);

  // δZ̃(μ, G) = FFT[e^{-i(q+Q)·r} · δZ(μ, r)](G), truncated to rho_g.
  nda::matrix<ComplexType, nda::F_layout> rhs_f(Np, nG_red);
  Timer.start("dxi_fft_forward_std");
  for (long u = 0; u < Np; ++u) {
    for (long ir = 0; ir < nnr; ++ir)
      fft_buf_1d(ir) = phase_mqpQ(ir) * DeltaZ_q(u, ir);
    F.forward(fft_buf);
    for (long g = 0; g < nG_red; ++g)
      rhs_f(u, g) = fft_buf_1d(rho_g.gv_to_fft(g));
  }
  Timer.stop("dxi_fft_forward_std");

  // rhs̃(G) = δZ̃(G) − δC · Ξ_qpQ_uG(G). Pure (Np × Np) · (Np × nG_red).
  Timer.start("dxi_rhs_gemm");
  nda::blas::gemm(ComplexType(-1.0), DeltaC_q, Xi_qpQ_uG,
                  ComplexType(1.0), rhs_f);
  Timer.stop("dxi_rhs_gemm");

  apply_C_inverse(factor, rhs_f,
                  factor.use_eig ? "dxi_eig_solve" : "dxi_lu_solve");

  DeltaXi_q_uG.resize(Np, nG_red);
  Timer.start("dxi_copy_out");
  for (long u = 0; u < Np; ++u)
    for (long g = 0; g < nG_red; ++g) DeltaXi_q_uG(u, g) = rhs_f(u, g);
  Timer.stop("dxi_copy_out");

  Timer.stop("solve_dxi_for_q");
}

// ----------------------------------------------------------------------------
// Step 7: drive steps 3–6 over all q' on the BZ k-grid.
//   Pass A: build Ξ^q' at every q'; cache real-space Ξ_ur, C-factor, kmq.
//   Pass B: build δΞ̃^q' at every q' using the cached Ξ_ur at slot iq_pQ_map(q').
// ----------------------------------------------------------------------------
void lr_thc::solve_xi_dxi_all_q(
    nda::array_view<ComplexType, 4> const T_skur,
    nda::array_view<ComplexType, 4> const DeltaT_pQ_skur,
    nda::array_view<ComplexType, 4> const DeltaT_mQ_skur,
    nda::array<double, 2> const& kpts_crys,
    nda::array<long, 1> const& r_P,
    nda::array<ComplexType, 3>& Xi_uG_all,
    nda::array<ComplexType, 3>& DeltaXi_uG_all)
{
  Timer.start("solve_xi_dxi_all_q");
  long const nq  = kpts_crys.shape(0);
  long const Np  = T_skur.shape(2);

  auto const& mf = *_reader.MF();
  auto const& rho_g = _reader.rho_g();
  long const nG_red = rho_g.size();
  auto recv_mf = mf.recv();

  Xi_uG_all.resize     (nq, Np, nG_red);
  DeltaXi_uG_all.resize(nq, Np, nG_red);
  Xi_uG_all()      = ComplexType(0.0);
  DeltaXi_uG_all() = ComplexType(0.0);

  // q-distribution across the full comm. Same partition is used for pass A
  // and pass B so the rank that built factor_cache[iq_loc] in pass A also
  // uses it in pass B (no cross-rank handoff of the LU/heev factor).
  auto [q_origin, q_end] = itertools::chunk_range(
      0, nq, _mpi->comm.size(), _mpi->comm.rank());
  long const nq_loc = q_end - q_origin;

  // Pass A: build Ξ^{iq}_uG for the rank-local q's. Ξ_ur_cache is dropped —
  // pass B works directly off Ξ_uG_all in G-space inside solve_dxi_for_q.
  std::vector<C_factor>           factor_cache(nq_loc);
  std::vector<nda::array<int, 1>> kmq_cache   (nq_loc);

  for (long iq_loc = 0; iq_loc < nq_loc; ++iq_loc) {
    long const iq = q_origin + iq_loc;
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

    nda::array<ComplexType, 2> Xi_q_ur, Xi_q_uG;
    C_factor factor;
    solve_xi_for_q(C_q, Z_q, q_cart,
                   Xi_q_ur, Xi_q_uG,
                   factor);

    Xi_uG_all(iq, nda::ellipsis{}) = Xi_q_uG;

    factor_cache[iq_loc] = std::move(factor);
    kmq_cache   [iq_loc] = std::move(kmq_map);
  }

  // All-reduce Xi_uG_all so pass B sees every iq_qpQ slab. Each rank wrote
  // only its owned q_origin..q_end slabs; the rest are zero.
  _mpi->comm.all_reduce_in_place_n(Xi_uG_all.data(), Xi_uG_all.size(),
                                   std::plus<>{});

  // Pass B. Same q-partition. The δΞ FFT phase uses kpts(iq_qpQ) (BZ-folded),
  // not the literal kpts(iq) + Q — required for termwise consistency with
  // Step 8.
  for (long iq_loc = 0; iq_loc < nq_loc; ++iq_loc) {
    long const iq = q_origin + iq_loc;
    long const iq_qpQ = _kqpoint_maps.kpQ_map(iq);
    nda::stack_array<double, 3> q_pQ_cart;
    for (int a = 0; a < 3; ++a)
      q_pQ_cart(a) = kpts_crys(iq_qpQ, 0) * recv_mf(0, a)
                   + kpts_crys(iq_qpQ, 1) * recv_mf(1, a)
                   + kpts_crys(iq_qpQ, 2) * recv_mf(2, a);

    nda::array<ComplexType, 2> DeltaZ_q, DeltaC_q;
    build_DeltaZ_DeltaC_for_q(T_skur, DeltaT_pQ_skur, DeltaT_mQ_skur,
                              kmq_cache[iq_loc], r_P, DeltaZ_q, DeltaC_q);

    nda::array<ComplexType, 2> DeltaXi_q_uG;
    solve_dxi_for_q(factor_cache[iq_loc],
                    DeltaZ_q, DeltaC_q,
                    Xi_uG_all(iq_qpQ, nda::ellipsis{}),
                    q_pQ_cart,
                    DeltaXi_q_uG);

    DeltaXi_uG_all(iq, nda::ellipsis{}) = DeltaXi_q_uG;
  }

  // All-reduce δΞ so step 8 sees every iq, iq_mqmQ slab.
  _mpi->comm.all_reduce_in_place_n(DeltaXi_uG_all.data(),
                                   DeltaXi_uG_all.size(), std::plus<>{});

  Timer.stop("solve_xi_dxi_all_q");
}

// ----------------------------------------------------------------------------
// Step 8: assemble δV^q for one internal q.
//
//   term1[μ,ν] = (1/V) Σ_G v(|kpts(iq_mq) + G|)  · δΞ_uG[iq_mqmQ, μ, G] · conj(Ξ_uG[iq_mq, ν, G])
//   term2[μ,ν] = (1/V) Σ_G v(|kpts(iq_qpQ) + G|) · conj(Ξ_uG[iq_qpQ, μ, G]) · δΞ_uG[iq, ν, G]
//
// BZ k-grid indices: iq (current q), iq_mq (−q), iq_qpQ (q+Q), iq_mqmQ (−q−Q).
// Bloch labels use kpts(iq_mq) / kpts(iq_qpQ), not literal ±q / q+Q — these
// match the BZ-folded phases used when the slabs were built. Ξ^{-q} =
// conj(Ξ^q) lets us read only +G data.
// ----------------------------------------------------------------------------
void lr_thc::compute_delta_V_for_q(
    long iq,
    long iq_mq,
    long iq_qpQ,
    long iq_mqmQ,
    nda::array<ComplexType, 3> const& Xi_uG_all,
    nda::array<ComplexType, 3> const& DeltaXi_uG_all,
    nda::stack_array<double, 3> const& q_mq_cart,
    nda::stack_array<double, 3> const& q_qpQ_cart,
    nda::array<ComplexType, 2>& dV_q)
{
  Timer.start("compute_delta_V_for_q");
  long const Np     = Xi_uG_all.shape(1);
  long const nG_red = Xi_uG_all.shape(2);

  auto const& mf = *_reader.MF();
  auto const& rho_g = _reader.rho_g();
  auto lattv_mf = mf.lattv();

  auto Xi_mq        = Xi_uG_all     (iq_mq,   nda::range::all, nda::range::all);
  auto Xi_qpQ       = Xi_uG_all     (iq_qpQ,  nda::range::all, nda::range::all);
  auto DeltaXi_q    = DeltaXi_uG_all(iq,      nda::range::all, nda::range::all);
  auto DeltaXi_mqmQ = DeltaXi_uG_all(iq_mqmQ, nda::range::all, nda::range::all);

  // Coulomb factors. vG().evaluate(V, lattv, gv, kp, kq) gives v(|G + (kp-kq)|),
  // so passing kp = kpts(iq_*), kq = 0 yields v(|kpts(iq_*) + G|).
  memory::unified_array<ComplexType, 1> v_kmq_G (rho_g.size());
  memory::unified_array<ComplexType, 1> v_kqpQ_G(rho_g.size());
  nda::array<RealType, 1> v_zero = {0.0, 0.0, 0.0};
  {
    nda::array<RealType, 1> q_mq_cart_a (3);
    nda::array<RealType, 1> q_qpQ_cart_a(3);
    for (int a = 0; a < 3; ++a) {
      q_mq_cart_a (a) = q_mq_cart(a);
      q_qpQ_cart_a(a) = q_qpQ_cart(a);
    }
    Timer.start("v_evaluate");
    _reader.vG().evaluate(v_kmq_G,  lattv_mf, rho_g.g_vectors(), q_mq_cart_a,  v_zero);
    _reader.vG().evaluate(v_kqpQ_G, lattv_mf, rho_g.g_vectors(), q_qpQ_cart_a, v_zero);
    Timer.stop("v_evaluate");
  }

  // term1 = (δΞ_mqmQ ⊙ v_kmq_G) · Xi_mq^H
  nda::array<ComplexType, 2> tmp1(Np, nG_red);
  for (long u = 0; u < Np; ++u)
    for (long g = 0; g < nG_red; ++g)
      tmp1(u, g) = DeltaXi_mqmQ(u, g) * v_kmq_G(g);

  nda::array<ComplexType, 2> term1(Np, Np);
  term1() = ComplexType(0.0);
  Timer.start("gemm_term1");
  nda::blas::gemm(ComplexType(1.0), tmp1, nda::dagger(Xi_mq),
                  ComplexType(0.0), term1);
  Timer.stop("gemm_term1");

  // term2 = (conj(Xi_qpQ) ⊙ v_kqpQ_G) · δΞ_q^T
  nda::array<ComplexType, 2> tmp2(Np, nG_red);
  for (long u = 0; u < Np; ++u)
    for (long g = 0; g < nG_red; ++g)
      tmp2(u, g) = std::conj(Xi_qpQ(u, g)) * v_kqpQ_G(g);

  nda::array<ComplexType, 2> term2(Np, Np);
  term2() = ComplexType(0.0);
  Timer.start("gemm_term2");
  nda::blas::gemm(ComplexType(1.0), tmp2, nda::transpose(DeltaXi_q),
                  ComplexType(0.0), term2);
  Timer.stop("gemm_term2");

  // Sum and (1/Vol) scaling (matches build_dV_from_xi).
  ComplexType const scale = ComplexType(1.0) / mf.volume();
  dV_q.resize(Np, Np);
  dV_q = scale * (term1 + term2);

  Timer.stop("compute_delta_V_for_q");
}

// ----------------------------------------------------------------------------
// Print timers as a hierarchy reflecting the call structure.
// ----------------------------------------------------------------------------
void lr_thc::print_timers()
{
  // (label, indent_level)
  std::vector<std::pair<char const*, int>> entries = {
    {"compute_delta_V",          0},
    {  "read_psi_and_dpsi",          1},
    {    "io_wfc",                   2},
    {    "io_dpsi",                  2},
    {    "io_dpsi_adj",              2},
    {    "phase_factor",             2},
    {  "compute_T_and_DeltaT",       1},
    {  "solve_xi_dxi_all_q",         1},
    {    "build_Z_C_for_q",          2},
    {      "accumulate_Z",           3},
    {    "build_DeltaZ_DeltaC_for_q",2},
    {      "accumulate_dZ",          3},
    {    "solve_xi_for_q",           2},
    {      "xi_lu_factor",           3},
    {      "xi_lu_solve",            3},
    {      "xi_eig_factor",          3},
    {      "xi_eig_solve",           3},
    {      "xi_phase_factor",        3},
    {      "xi_fft_forward_std",     3},
    {      "xi_copy_rhs",            3},
    {      "xi_copy_out",            3},
    {    "solve_dxi_for_q",          2},
    {      "dxi_lu_solve",           3},
    {      "dxi_eig_solve",          3},
    {      "dxi_phase_factor",       3},
    {      "dxi_fft_forward_std",    3},
    {      "dxi_rhs_gemm",           3},
    {      "dxi_copy_out",           3},
    {  "compute_delta_V_for_q",      1},
    {    "v_evaluate",               2},
    {    "gemm_term1",               2},
    {    "gemm_term2",               2},
  };
  app_log(2, "\n  LR_THC timers");
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
// Populate _kqpoint_maps: kpQ_map / kmQ_map via utils::calculate_kpq_map (with
// ±Q); mk_map, mkmQ_map via a small scan with the same wrap convention as
// calculate_kpq_map.
// ----------------------------------------------------------------------------
void lr_thc::build_momentum_maps(nda::array<double, 2> const& kpts_crys,
                                     nda::stack_array<double, 3> const& Q_cryst,
                                     nda::stack_array<double, 3> const& Q_cart)
{
  long const nk = kpts_crys.shape(0);
  _kqpoint_maps.Q_cryst = Q_cryst;
  _kqpoint_maps.Q_cart  = Q_cart;

  _kqpoint_maps.kpQ_map.resize(nk);
  utils::calculate_kpq_map(kpts_crys, Q_cryst, _kqpoint_maps.kpQ_map);

  nda::stack_array<double, 3> Q_neg_cryst;
  for (int a = 0; a < 3; ++a) Q_neg_cryst(a) = -Q_cryst(a);
  _kqpoint_maps.kmQ_map.resize(nk);
  utils::calculate_kpq_map(kpts_crys, Q_neg_cryst, _kqpoint_maps.kmQ_map);

  // mk_map(p) and mkmQ_map(p): same wrap+tolerance convention as calculate_kpq_map.
  auto find_match = [&](long ip, double off0, double off1, double off2) {
    double d0 = std::abs(kpts_crys(ip, 0) + off0);
    double d1 = std::abs(kpts_crys(ip, 1) + off1);
    double d2 = std::abs(kpts_crys(ip, 2) + off2);
    d0 -= std::floor(d0); d1 -= std::floor(d1); d2 -= std::floor(d2);
    d0 -= std::round(d0); d1 -= std::round(d1); d2 -= std::round(d2);
    return d0*d0 + d1*d1 + d2*d2;
  };
  double const tol2 = 1e-12;  // (1e-6)^2

  _kqpoint_maps.mk_map.resize(nk);
  _kqpoint_maps.mkmQ_map.resize(nk);
  for (long ip = 0; ip < nk; ++ip) {
    int found_mk = -1, found_mkmQ = -1;
    for (long iq = 0; iq < nk; ++iq) {
      if (found_mk   < 0 && find_match(iq, kpts_crys(ip, 0),
                                           kpts_crys(ip, 1),
                                           kpts_crys(ip, 2)) < tol2)
        found_mk = (int)iq;
      if (found_mkmQ < 0 && find_match(iq, kpts_crys(ip, 0) + Q_cryst(0),
                                           kpts_crys(ip, 1) + Q_cryst(1),
                                           kpts_crys(ip, 2) + Q_cryst(2)) < tol2)
        found_mkmQ = (int)iq;
      if (found_mk >= 0 && found_mkmQ >= 0) break;
    }
    utils::check(found_mk   >= 0, "lr_thc::build_momentum_maps: -k not found on grid for ip={}.", ip);
    utils::check(found_mkmQ >= 0, "lr_thc::build_momentum_maps: -k-Q not found on grid for ip={}.", ip);
    _kqpoint_maps.mk_map  (ip) = found_mk;
    _kqpoint_maps.mkmQ_map(ip) = found_mkmQ;
  }
}

// ----------------------------------------------------------------------------
// Driver: wires steps 1–8.
// ----------------------------------------------------------------------------
nda::array<ComplexType, 3> lr_thc::compute_delta_V(
    std::string const& Deltapsi_prefix,
    std::string const& Deltapsi_adj_prefix,
    nda::array<double, 1> const& q_pert_cryst)
{
  Timer.start("compute_delta_V");

  utils::check(q_pert_cryst.shape(0) == 3,
               "lr_thc::compute_delta_V: q_pert_cryst must have size 3.");

  auto const& mf = *_reader.MF();

  utils::check(mf.npol() == 1,
               "lr_thc::compute_delta_V: non-collinear (npol > 1) not supported.");
  utils::check(mf.nkpts() == mf.nkpts_ibz(),
               "lr_thc::compute_delta_V: requires nkpts == nkpts_ibz "
               "(no k-symmetry).");
  utils::check(mf.orb_on_fft_grid(),
               "lr_thc::compute_delta_V: requires mf->orb_on_fft_grid() == true.");
  // The returned δV is indexed by iq in mf.kpts_crystal() order; require the
  // BZ q-grid to equal the k-grid.
  utils::check(mf.nqpts() == mf.nkpts(),
               "lr_thc::compute_delta_V: requires nqpts == nkpts "
               "(BZ q-grid must equal the k-grid). got nqpts={}, nkpts={}.",
               mf.nqpts(), mf.nkpts());
  utils::check(mf.nqpts_ibz() == mf.nkpts_ibz(),
               "lr_thc::compute_delta_V: requires nqpts_ibz == nkpts_ibz "
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
  nda::array<double, 2> kpts_crys(nkpts, 3);
  for (long ik = 0; ik < nkpts; ++ik)
    for (int a = 0; a < 3; ++a) kpts_crys(ik, a) = kpts_crys_mf(ik, a);

  // Q in crystal and Cartesian (2π included).
  nda::stack_array<double, 3> Q_cryst, Q_cart;
  for (int a = 0; a < 3; ++a) {
    Q_cryst(a) = q_pert_cryst(a);
    Q_cart(a)  = q_pert_cryst(0) * recv_mf(0, a)
               + q_pert_cryst(1) * recv_mf(1, a)
               + q_pert_cryst(2) * recv_mf(2, a);
  }

  nda::array<long, 1> r_P_local(Np);
  for (long u = 0; u < Np; ++u) r_P_local(u) = r_P(u);

  // Build BZ-index maps consumed by every step.
  build_momentum_maps(kpts_crys, Q_cryst, Q_cart);

  // Memory estimate. Caches live in node-shared memory: one copy per NODE,
  // not per rank. Peak per node = step 1 ψ/δψ + step 2 T caches (both live at
  // step 2).
  {
    double const GB = 1024.0 * 1024.0 * 1024.0;
    double mem_psi  = 3.0 * double(nspin) * double(nkpts) * double(nbnd) * double(nnr) * 16.0 / GB;
    double mem_T    = 3.0 * double(nspin) * double(nkpts) * double(Np)   * double(nnr) * 16.0 / GB;
    double mem_peak = mem_psi + mem_T;  // ψ/δψ and T caches coexist during step 2.
    app_log(2, "  lr_thc estimated peak host-memory for caches (per node):");
    app_log(2, "    psi/dpsi (3 x nspin x nkpts x nbnd x nnr): {0:8.3f} GB", mem_psi);
    app_log(2, "    T/dT     (3 x nspin x nkpts x Np   x nnr): {0:8.3f} GB", mem_T);
    app_log(2, "    peak (sum during step 2):                  {0:8.3f} GB", mem_peak);
  }

  // --- Step 1: read ψ and δψ into node-shared caches. ---
  // shared_array's constructor zero-fills, so the round-robin (is, ik)
  // distribution inside read_psi_and_dpsi safely populates only owned slabs.
  using shm_arr4_t = math::shm::shared_array<nda::array_view<ComplexType, 4>>;
  shm_arr4_t s_psi_skmr        (*_mpi, {nspin, nkpts, nbnd, nnr});
  shm_arr4_t s_Deltapsi_pQ_skmr(*_mpi, {nspin, nkpts, nbnd, nnr});
  shm_arr4_t s_Deltapsi_mQ_skmr(*_mpi, {nspin, nkpts, nbnd, nnr});
  read_psi_and_dpsi(Deltapsi_prefix, Deltapsi_adj_prefix, fft_mesh,
                    s_psi_skmr.local(), s_Deltapsi_pQ_skmr.local(),
                    s_Deltapsi_mQ_skmr.local());
  // All node ranks must see populated ψ/δψ before step 2 reads cross-k.
  s_psi_skmr.node_sync();
  s_Deltapsi_pQ_skmr.node_sync();
  s_Deltapsi_mQ_skmr.node_sync();

  // --- Step 2: T, ΔT_pQ, ΔT_mQ caches in node-shared memory. ---
  shm_arr4_t s_T_skur        (*_mpi, {nspin, nkpts, Np, nnr});
  shm_arr4_t s_DeltaT_pQ_skur(*_mpi, {nspin, nkpts, Np, nnr});
  shm_arr4_t s_DeltaT_mQ_skur(*_mpi, {nspin, nkpts, Np, nnr});
  compute_T_and_DeltaT(s_psi_skmr.local(), s_Deltapsi_pQ_skmr.local(),
                       s_Deltapsi_mQ_skmr.local(),
                       r_P_local,
                       s_T_skur.local(), s_DeltaT_pQ_skur.local(),
                       s_DeltaT_mQ_skur.local());
  s_T_skur.node_sync();
  s_DeltaT_pQ_skur.node_sync();
  s_DeltaT_mQ_skur.node_sync();

  // ψ / δψ are no longer needed after T-cache is built. Release the shared
  // windows by reassigning to size-1 placeholders.
  s_psi_skmr         = shm_arr4_t(*_mpi, {1, 1, 1, 1});
  s_Deltapsi_pQ_skmr = shm_arr4_t(*_mpi, {1, 1, 1, 1});
  s_Deltapsi_mQ_skmr = shm_arr4_t(*_mpi, {1, 1, 1, 1});

  long const nq = nkpts;

  // --- Steps 3–6 (driven by step 7): solve Ξ and δΞ at all q's. ---
  // q's are distributed across _mpi->comm inside solve_xi_dxi_all_q; on return
  // every rank holds the full Xi_uG_all / DeltaXi_uG_all via an internal
  // all_reduce.
  nda::array<ComplexType, 3> Xi_uG_all;
  nda::array<ComplexType, 3> DeltaXi_uG_all;
  solve_xi_dxi_all_q(s_T_skur.local(), s_DeltaT_pQ_skur.local(),
                     s_DeltaT_mQ_skur.local(),
                     kpts_crys, r_P_local,
                     Xi_uG_all, DeltaXi_uG_all);

  // T-caches no longer needed.
  s_T_skur         = shm_arr4_t(*_mpi, {1, 1, 1, 1});
  s_DeltaT_pQ_skur = shm_arr4_t(*_mpi, {1, 1, 1, 1});
  s_DeltaT_mQ_skur = shm_arr4_t(*_mpi, {1, 1, 1, 1});

  // Step 8: δV^q for every q. Pass BZ-folded slot k-points (not literal ±q /
  // q+Q) so the Coulomb factors match the slab Bloch phases termwise.
  // q-distributed across _mpi->comm; rank-local slots are filled, the rest
  // stay zero, then all_reduce gives every rank the same full output.
  nda::array<ComplexType, 3> out(nq, Np, Np);
  out() = ComplexType(0.0);
  {
    auto [q_origin, q_end] = itertools::chunk_range(
        0, nq, _mpi->comm.size(), _mpi->comm.rank());
    long const nq_loc = q_end - q_origin;
    for (long iq_loc = 0; iq_loc < nq_loc; ++iq_loc) {
      long const iq      = q_origin + iq_loc;
      long const iq_mq   = _kqpoint_maps.mk_map(iq);
      long const iq_qpQ  = _kqpoint_maps.kpQ_map(iq);
      long const iq_mqmQ = _kqpoint_maps.mkmQ_map(iq);
      nda::stack_array<double, 3> q_mq_cart, q_qpQ_cart;
      for (int a = 0; a < 3; ++a) {
        q_mq_cart (a) = kpts_crys(iq_mq,  0) * recv_mf(0, a)
                      + kpts_crys(iq_mq,  1) * recv_mf(1, a)
                      + kpts_crys(iq_mq,  2) * recv_mf(2, a);
        q_qpQ_cart(a) = kpts_crys(iq_qpQ, 0) * recv_mf(0, a)
                      + kpts_crys(iq_qpQ, 1) * recv_mf(1, a)
                      + kpts_crys(iq_qpQ, 2) * recv_mf(2, a);
      }
      nda::array<ComplexType, 2> dV_q;
      compute_delta_V_for_q(iq, iq_mq, iq_qpQ, iq_mqmQ,
                            Xi_uG_all, DeltaXi_uG_all,
                            q_mq_cart, q_qpQ_cart, dV_q);
      out(iq, nda::ellipsis{}) = dV_q;
    }
  }
  _mpi->comm.all_reduce_in_place_n(out.data(), out.size(), std::plus<>{});

  Timer.stop("compute_delta_V");
  print_timers();
  return out;
}

} // namespace methods
