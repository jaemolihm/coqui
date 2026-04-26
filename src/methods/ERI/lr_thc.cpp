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

#include "lr_thc.hpp"
#include "lr_psi_io.hpp"

#include <cmath>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
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
#include "nda/h5.hpp"
#include "h5/h5.hpp"
#include "itertools/itertools.hpp"

namespace methods {

// ============================================================================
// Stage A: build q-independent per-k caches T, δT, and δr·∇₂T once.
//
//   T_k(μ, r)        = Σ_a conj(ψ_{a,k}(r_μ)) · ψ_{a,k}(r)
//   δT_k(μ, r)       = Σ_a [ conj(δψ_{a,k}(r_μ) + (δr_μ·∇)ψ_{a,k}(r_μ)) · ψ_{a,k}(r)
//                          + conj(ψ_{a,k}(r_μ)) · δψ_{a,k}(r) ]
//   δT_grad2_k(μ, ν) = Σ_a conj(ψ_{a,k}(r_μ)) · (δr_ν·∇)ψ_{a,k}(r_ν)
//
// All three are built from the same (ψ, δψ, (δr·∇)ψ) slabs decoded per (is, ik).
// δT fuses the δψ and δr_μ-gradient contributions; δT_grad2 is the ν-pivot-shift
// used to form δC_q in stage B directly (replacing the old FFT-based pivot-shift).
//
// Nothing here references the internal q of V^q / Ξ^q; all stage-A work is
// reused across the batched per-q loop in xi_Deltaxi_one_q.
//
// T / δT are held in full on every rank (after a single Allreduce). Memory cost:
//   T, δT        : nspin · nkpts · Np · nnr · 16 B
//   δT_grad2     : nspin · nkpts · Np · Np  · 16 B
// ============================================================================
lr_thc::kcache_t lr_thc::build_kcache(
    std::string const& Deltapsi_prefix,
    nda::array<double, 2> const& delta_r_P,
    nda::array<double, 1> const& q_phonon_cryst)
{
  Timer.start("build_kcache");
  utils::check(q_phonon_cryst.shape(0) == 3,
               "lr_thc::build_kcache: q_phonon_cryst must have size 3.");
  using nda::range;

  // --- Access reader internals (lr_thc holds a thc_reader_t&) ---
  auto const& mf = *_reader.MF();
  auto* mpi_ctx = _reader.mpi().get();
  auto const& rho_g = _reader.rho_g();
  auto& mpi_comm = mpi_ctx->comm;
  int rank = mpi_comm.rank();
  int nranks = mpi_comm.size();

  // --- MVP restrictions (MPI > 1 supported via k-parallel chunking + Allreduce) ---
  utils::check(mf.npol() == 1,
               "lr_thc::build_kcache: non-collinear (npol > 1) not yet supported.");
  utils::check(mf.nkpts() == mf.nkpts_ibz(),
               "lr_thc::build_kcache: k-point symmetry (nkpts != nkpts_ibz) not yet "
               "supported; rerun with symmetry disabled (nkpts == nkpts_ibz).");
  utils::check(mf.orb_on_fft_grid(),
               "lr_thc::build_kcache: requires mf->orb_on_fft_grid()==true.");
  utils::check(mf.nspin_in_basis() * mf.npol_in_basis() == mf.nspin(),
               "lr_thc::build_kcache: nspin_in_basis*npol_in_basis ({}) != nspin ({}).",
               mf.nspin_in_basis() * mf.npol_in_basis(), mf.nspin());

  long nbnd = mf.nbnd();
  long nkpts = mf.nkpts();

  auto const& r_P = _reader.ri();
  long Np = r_P.shape(0);
  long nG_red = rho_g.size();

  utils::check(delta_r_P.shape(0) == Np && delta_r_P.shape(1) == 3,
               "lr_thc::build_kcache: delta_r_P must have shape ({}, 3), got ({}, {}).",
               Np, delta_r_P.shape(0), delta_r_P.shape(1));

  long nspin = mf.nspin();
  auto lattv_mf = mf.lattv();
  auto recv_mf  = mf.recv();
  auto kpts_mf  = mf.kpts();  // (nk, 3) Cartesian with 2π

  auto mesh = rho_g.mesh();
  long N1 = mesh(0), N2 = mesh(1), N3 = mesh(2);
  long nnr = rho_g.nnr();

  nda::stack_array<long, 3> fft_mesh;
  for (int i = 0; i < 3; ++i) fft_mesh(i) = mesh(i);

  // q_phonon in Cartesian (Bloch momentum shift of δψ; q-independent)
  nda::stack_array<double, 3> q_phonon_cart;
  for (int a = 0; a < 3; ++a)
    q_phonon_cart(a) = q_phonon_cryst(0) * recv_mf(0, a)
                     + q_phonon_cryst(1) * recv_mf(1, a)
                     + q_phonon_cryst(2) * recv_mf(2, a);

  // --- Allocate T, δT, and δr·∇₂T caches. Held in full on every rank after Allreduce. ---
  nda::array<ComplexType, 4> T_sk_ur(nspin, nkpts, Np, nnr);
  nda::array<ComplexType, 4> DeltaT_sk_ur(nspin, nkpts, Np, nnr);
  nda::array<ComplexType, 4> DeltaT_grad2_sk_PP(nspin, nkpts, Np, Np);
  T_sk_ur()             = ComplexType(0.0);
  DeltaT_sk_ur()        = ComplexType(0.0);
  DeltaT_grad2_sk_PP()  = ComplexType(0.0);

  // Per-band ψ and δψ slices on the full FFT grid (nbnd, nnr) with Bloch phases applied.
  nda::array<ComplexType, 2> psi_ar(nbnd, nnr);
  nda::array<ComplexType, 2> Deltapsi_ar(nbnd, nnr);
  // (δr_μ · ∇)ψ at pivot points (Np, nbnd), evaluated with Bloch phase.
  nda::array<ComplexType, 2> Gradpsi_Pa(Np, nbnd);

  // FFT workspaces (reused across bands and k-points).
  nda::array<ComplexType, 4> fft_buf(1, N1, N2, N3);
  auto fft_buf_1d = nda::reshape(fft_buf, std::array<long, 1>{nnr});
  math::nda::fft<true> F(fft_buf);
  nda::array<ComplexType, 4> grad_buf(1, N1, N2, N3);
  auto grad_buf_flat = nda::reshape(grad_buf, std::array<long, 2>{1, nnr});
  nda::array<ComplexType, 1> c_on_grid(nnr);

  // Helper: for a given (is, ik), fill the three per-(is,ik) band/pivot arrays from
  // the wfc + δψ files. Both files must contain at least `nbnd` bands (checked).
  //   psi_ar_out(a, r)      = ψ_a^k(r) on full FFT grid (Bloch phase applied)
  //   Deltapsi_ar_out(a, r) = δψ_a^{k+Q}(r) on full FFT grid (Bloch phase applied)
  //   Gradpsi_out(u, a)     = (δr_μ · ∇)ψ_a^k(r_μ) at pivot μ=u (Bloch phase applied)
  // k_cart is the Bloch momentum of ψ in Cartesian (with 2π). k_cryst is its
  // crystal-coord counterpart (used by rspace_phase_factor's sparse-index form).
  // kplusQ_cart = k + Q_phonon in Cartesian — Bloch momentum of δψ.
  // δψ at pivots is read on demand from Deltapsi_ar(a, r_P(u)).
  auto fill_psi_slices = [&](long is, long ik,
                             nda::stack_array<double, 3> const& k_cart,
                             nda::stack_array<double, 3> const& k_cryst,
                             nda::stack_array<double, 3> const& kplusQ_cart,
                             nda::array<ComplexType, 2>& psi_ar_out,
                             nda::array<ComplexType, 2>& Deltapsi_ar_out,
                             nda::array<ComplexType, 2>& Gradpsi_out)
  {
    auto wfc      = detail::read_wfc_k(mf, is, ik, fft_mesh);
    auto Deltapsi = detail::read_Deltapsi_k(Deltapsi_prefix, ik, fft_mesh);
    utils::check(wfc.nbnd      >= nbnd, "lr_thc: wfc file at (is={}, ik={}) has {} bands, expected >= {}.",
                 is, ik, wfc.nbnd, nbnd);
    utils::check(Deltapsi.nbnd >= nbnd, "lr_thc: dpsi file at ik={} has {} bands, expected >= {}.",
                 ik, Deltapsi.nbnd, nbnd);

    nda::array<ComplexType, 1> phase_k_full(nnr);
    nda::array<ComplexType, 1> phase_Deltapsi_full(nnr);
    nda::array<ComplexType, 1> phase_k_P(Np);
    utils::rspace_phase_factor(lattv_mf, k_cart,
        nda::reshape(phase_k_full, std::array<long,3>{N1,N2,N3}));
    utils::rspace_phase_factor(lattv_mf, kplusQ_cart,
        nda::reshape(phase_Deltapsi_full, std::array<long,3>{N1,N2,N3}));
    utils::rspace_phase_factor(k_cryst, fft_mesh, r_P, phase_k_P);

    Gradpsi_out() = ComplexType(0.0);

    for (long a = 0; a < nbnd; ++a) {
      // ψ full-grid slice + (δr_μ · ∇)ψ at pivots.
      fft_buf_1d() = ComplexType(0.0);
      detail::decode_band_to_fft_grid(wfc.evc_raw, a, wfc.k2g, fft_buf_1d);
      F.backward(fft_buf);
      for (long ir = 0; ir < nnr; ++ir)
        psi_ar_out(a, ir) = fft_buf_1d(ir) * phase_k_full(ir);

      c_on_grid() = ComplexType(0.0);
      detail::decode_band_to_fft_grid(wfc.evc_raw, a, wfc.k2g, c_on_grid);
      for (int alpha = 0; alpha < 3; ++alpha) {
        grad_buf_flat(0, nda::range::all) = c_on_grid;
        detail::multiply_by_ikpG(grad_buf, k_cart, recv_mf, alpha);
        F.backward(grad_buf);
        for (long u = 0; u < Np; ++u)
          Gradpsi_out(u, a) += grad_buf_flat(0, r_P(u)) * phase_k_P(u) * delta_r_P(u, alpha);
      }

      // δψ full-grid slice.
      fft_buf_1d() = ComplexType(0.0);
      detail::decode_band_to_fft_grid(Deltapsi.evc_raw, a, Deltapsi.k2g, fft_buf_1d);
      F.backward(fft_buf);
      for (long ir = 0; ir < nnr; ++ir)
        Deltapsi_ar_out(a, ir) = fft_buf_1d(ir) * phase_Deltapsi_full(ir);
    }
  };

  // --- k-chunked loop: each rank fills its (is, ik) slab of T_sk_ur / δT_sk_ur. ---
  long const nsk = nspin * nkpts;
  auto [sk_lo, sk_hi] = itertools::chunk_range(0L, nsk, nranks, rank);

  auto kpts_crys_mf = mf.kpts_crystal();
  for (long sk = sk_lo; sk < sk_hi; ++sk) {
    long const is = sk / nkpts;
    long const ik = sk % nkpts;

    nda::stack_array<double, 3> k_cart, kplusQ_cart, k_cryst;
    for (int a = 0; a < 3; ++a) {
      k_cart(a)      = kpts_mf(ik, a);
      kplusQ_cart(a) = k_cart(a) + q_phonon_cart(a);
      k_cryst(a)     = kpts_crys_mf(ik, a);
    }

    fill_psi_slices(is, ik, k_cart, k_cryst, kplusQ_cart, psi_ar, Deltapsi_ar, Gradpsi_Pa);

    // Build T, δT, and δr·∇₂T in a single (a, u) pass:
    //   T(μ, r)        = Σ_a conj(ψ(r_μ)) · ψ(r)
    //   δT(μ, r)       = Σ_a [ conj(δψ(r_μ) + (δr_μ·∇)ψ(r_μ)) · ψ(r)
    //                        + conj(ψ(r_μ)) · δψ(r) ]
    //   δT_grad2(μ, ν) = Σ_a conj(ψ(r_μ)) · (δr_ν·∇)ψ(r_ν)
    auto T_slice  = T_sk_ur(is, ik, nda::range::all, nda::range::all);
    auto dT_slice = DeltaT_sk_ur(is, ik, nda::range::all, nda::range::all);
    auto C2_slice = DeltaT_grad2_sk_PP(is, ik, nda::range::all, nda::range::all);
    for (long a = 0; a < nbnd; ++a) {
      for (long u = 0; u < Np; ++u) {
        ComplexType psi_conj    = std::conj(psi_ar(a, r_P(u)));
        ComplexType Deltax_conj = std::conj(Deltapsi_ar(a, r_P(u)) + Gradpsi_Pa(u, a));
        for (long ir = 0; ir < nnr; ++ir) {
          T_slice (u, ir) += psi_conj    * psi_ar(a, ir);
          dT_slice(u, ir) += Deltax_conj * psi_ar(a, ir)
                           + psi_conj    * Deltapsi_ar(a, ir);
        }
        for (long v = 0; v < Np; ++v)
          C2_slice(u, v) += psi_conj * Gradpsi_Pa(v, a);
      }
    }
  }

  // Replicate full caches on every rank so stage B needs no per-q communication.
  if (nranks > 1) {
    mpi_comm.all_reduce_in_place_n(T_sk_ur.data(),             T_sk_ur.size(),             std::plus<>{});
    mpi_comm.all_reduce_in_place_n(DeltaT_sk_ur.data(),        DeltaT_sk_ur.size(),        std::plus<>{});
    mpi_comm.all_reduce_in_place_n(DeltaT_grad2_sk_PP.data(),  DeltaT_grad2_sk_PP.size(),  std::plus<>{});
  }

  // --- Pack the cache struct with stage-B metadata ---
  kcache_t cache;
  cache.T_sk_ur             = std::move(T_sk_ur);
  cache.DeltaT_sk_ur        = std::move(DeltaT_sk_ur);
  cache.DeltaT_grad2_sk_PP  = std::move(DeltaT_grad2_sk_PP);
  cache.N1 = N1; cache.N2 = N2; cache.N3 = N3;
  cache.nnr = nnr;
  cache.Np = Np;
  cache.nspin = nspin;
  cache.nkpts = nkpts;
  cache.nG_red = nG_red;
  {
    auto gv_to_fft = rho_g.gv_to_fft();
    cache.gv_to_fft.resize(nG_red);
    for (long g = 0; g < nG_red; ++g) cache.gv_to_fft(g) = gv_to_fft(g);
  }
  {
    auto kpts_crys_all = mf.kpts_crystal();
    cache.kpts_crys.resize(nkpts, 3);
    for (long ik = 0; ik < nkpts; ++ik)
      for (int a = 0; a < 3; ++a) cache.kpts_crys(ik, a) = kpts_crys_all(ik, a);
  }
  cache.recv.resize(3, 3);
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) cache.recv(i, j) = recv_mf(i, j);
  cache.lattv.resize(3, 3);
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) cache.lattv(i, j) = lattv_mf(i, j);
  Timer.stop("build_kcache");
  return cache;
}

// ============================================================================
// Stage B+LU: given the q-independent cache, compute Ξ^q and δΞ^q on the
// reduced G-grid for a single internal q_vec_cryst.
//
// No MPI communication here — the cache is already replicated, and all ranks
// redo this work in lockstep. That simplifies the batched loop (no per-q
// Allreduce) at the cost of redundant arithmetic; the FFT-r→G and LU are cheap
// compared to stage A.
// ============================================================================
lr_thc::xi_Deltaxi_on_Gred lr_thc::xi_Deltaxi_one_q(
    kcache_t const& cache,
    nda::array<double, 1> const& q_vec_cryst)
{
  Timer.start("xi_Deltaxi_one_q");
  utils::check(q_vec_cryst.shape(0) == 3,
               "lr_thc::xi_Deltaxi_one_q: q_vec_cryst must have size 3.");
  using nda::range;

  long const Np     = cache.Np;
  long const nnr    = cache.nnr;
  long const nG_red = cache.nG_red;
  long const N1 = cache.N1, N2 = cache.N2, N3 = cache.N3;
  long const nspin = cache.nspin;
  long const nkpts = cache.nkpts;

  auto const& r_P = _reader.ri();

  // --- Locate iq_target in the BZ k-grid ---
  long iq_target = -1;
  for (long iq = 0; iq < nkpts; ++iq) {
    double d0 = cache.kpts_crys(iq, 0) - q_vec_cryst(0);
    double d1 = cache.kpts_crys(iq, 1) - q_vec_cryst(1);
    double d2 = cache.kpts_crys(iq, 2) - q_vec_cryst(2);
    d0 -= std::round(d0); d1 -= std::round(d1); d2 -= std::round(d2);
    if (d0*d0 + d1*d1 + d2*d2 < 1e-10) { iq_target = iq; break; }
  }
  utils::check(iq_target >= 0,
               "lr_thc::xi_Deltaxi_one_q: q_vec_cryst = ({}, {}, {}) not found in BZ k-grid.",
               q_vec_cryst(0), q_vec_cryst(1), q_vec_cryst(2));

  // q in Cartesian
  nda::stack_array<double, 3> q_cart;
  for (int a = 0; a < 3; ++a)
    q_cart(a) = q_vec_cryst(0) * cache.recv(0, a)
              + q_vec_cryst(1) * cache.recv(1, a)
              + q_vec_cryst(2) * cache.recv(2, a);

  nda::stack_array<long, 3> fft_mesh;
  fft_mesh(0) = N1; fft_mesh(1) = N2; fft_mesh(2) = N3;

  // k+q map
  nda::array<int, 1> kpq_map(nkpts);
  utils::calculate_kpq_map(cache.kpts_crys, q_vec_cryst, kpq_map);

  // --- Accumulate Z_r, δZ_r from the cached T / δT over all (is, ik). ---
  nda::array<ComplexType, 2> Z_r(Np, nnr);
  nda::array<ComplexType, 2> DeltaZ_r(Np, nnr);
  Z_r()      = ComplexType(0.0);
  DeltaZ_r() = ComplexType(0.0);

  for (long is = 0; is < nspin; ++is) {
    for (long ik = 0; ik < nkpts; ++ik) {
      int const ikq = kpq_map(ik);
      auto T_k   = cache.T_sk_ur(is, ik,  nda::range::all, nda::range::all);
      auto dT_k  = cache.DeltaT_sk_ur(is, ik,  nda::range::all, nda::range::all);
      auto T_kq  = cache.T_sk_ur(is, ikq, nda::range::all, nda::range::all);
      auto dT_kq = cache.DeltaT_sk_ur(is, ikq, nda::range::all, nda::range::all);

      for (long u = 0; u < Np; ++u)
        for (long ir = 0; ir < nnr; ++ir) {
          Z_r(u, ir)      += std::conj(T_kq(u, ir)) * T_k(u, ir);
          DeltaZ_r(u, ir) += std::conj(dT_kq(u, ir)) * T_k(u, ir)
                           + std::conj(T_kq(u, ir)) * dT_k(u, ir);
        }
    }
  }

  // --- C_q(μ, ν) = Z_r(μ, r_P(ν)), δC_q analogous. Done BEFORE the e^{iq·r} phase. ---
  nda::array<ComplexType, 2> C_q (Np, Np);
  nda::array<ComplexType, 2> DeltaC_q(Np, Np);
  for (long u = 0; u < Np; ++u)
    for (long v = 0; v < Np; ++v) {
      C_q (u, v) = Z_r (u, r_P(v));
      DeltaC_q(u, v) = DeltaZ_r(u, r_P(v));
    }

  // ν-pivot-shift contribution to δC_q from the cached δr_ν·∇₂T:
  //   δC_q(μ,ν) += Σ_k [conj(δr_ν·∇₂T^{k+q})(μ,ν) · T^k(r_μ, r_ν)
  //                    + conj(T^{k+q})(r_μ, r_ν) · (δr_ν·∇₂T^k)(μ,ν)]
  // Equivalent to δr_ν · ∇_r Z^q_{μ,r}|_{r=r_ν} computed directly from ∇ψ at
  // pivot points, replacing the earlier FFT-based reconstruction of ∂_α Z_r.
  {
    nda::array<ComplexType, 2> T_k_rP  (Np, Np);
    nda::array<ComplexType, 2> T_kq_rP (Np, Np);
    for (long is = 0; is < nspin; ++is) {
      for (long ik = 0; ik < nkpts; ++ik) {
        int const ikq = kpq_map(ik);
        auto T_k   = cache.T_sk_ur(is, ik,  nda::range::all, nda::range::all);
        auto T_kq  = cache.T_sk_ur(is, ikq, nda::range::all, nda::range::all);
        auto dT2_k  = cache.DeltaT_grad2_sk_PP(is, ik,  nda::range::all, nda::range::all);
        auto dT2_kq = cache.DeltaT_grad2_sk_PP(is, ikq, nda::range::all, nda::range::all);
        for (long u = 0; u < Np; ++u)
          for (long v = 0; v < Np; ++v) {
            T_k_rP (u, v) = T_k (u, r_P(v));
            T_kq_rP(u, v) = T_kq(u, r_P(v));
          }
        for (long u = 0; u < Np; ++u)
          for (long v = 0; v < Np; ++v)
            DeltaC_q(u, v) += std::conj(dT2_kq(u, v)) * T_k_rP(u, v)
                            + std::conj(T_kq_rP(u, v)) * dT2_k(u, v);
      }
    }
  }

  // --- Apply exp(iq·r) phase on full grid, then FFT r→G and truncate ---
  {
    nda::array<ComplexType, 1> phase_q_full(nnr);
    nda::stack_array<double, 3, 3> lattv_stk;
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j) lattv_stk(i, j) = cache.lattv(i, j);
    utils::rspace_phase_factor(lattv_stk, q_cart,
        nda::reshape(phase_q_full, std::array<long,3>{N1,N2,N3}));
    for (long ir = 0; ir < nnr; ++ir) {
      ComplexType ph = phase_q_full(ir);
      for (long u = 0; u < Np; ++u) {
        Z_r(u, ir)      *= ph;
        DeltaZ_r(u, ir) *= ph;
      }
    }
  }

  nda::array<ComplexType, 4> fft_buf(1, N1, N2, N3);
  auto fft_buf_1d = nda::reshape(fft_buf, std::array<long, 1>{nnr});
  math::nda::fft<true> F(fft_buf);

  nda::array<ComplexType, 2> Z_G(Np, nG_red);
  nda::array<ComplexType, 2> DeltaZ_G(Np, nG_red);
  for (long u = 0; u < Np; ++u) {
    for (long ir = 0; ir < nnr; ++ir) fft_buf_1d(ir) = Z_r(u, ir);
    F.forward(fft_buf);
    for (long g = 0; g < nG_red; ++g) Z_G(u, g) = fft_buf_1d(cache.gv_to_fft(g));

    for (long ir = 0; ir < nnr; ++ir) fft_buf_1d(ir) = DeltaZ_r(u, ir);
    F.forward(fft_buf);
    for (long g = 0; g < nG_red; ++g) DeltaZ_G(u, g) = fft_buf_1d(cache.gv_to_fft(g));
  }

  // --- Solve C_q · Ξ = Z_q and C_q · δΞ = (δZ − δC·Ξ) with one LU factorization. ---
  nda::matrix<ComplexType, nda::F_layout> C_f(Np, Np);
  nda::matrix<ComplexType, nda::F_layout> Z_f(Np, nG_red);
  for (long u = 0; u < Np; ++u)
    for (long v = 0; v < Np; ++v) C_f(u, v) = C_q(u, v);
  for (long u = 0; u < Np; ++u)
    for (long g = 0; g < nG_red; ++g) Z_f(u, g) = Z_G(u, g);

  nda::array<int, 1> ipiv(Np);
  int info_f = nda::lapack::getrf(C_f, ipiv);
  utils::check(info_f == 0, "lr_thc::xi_Deltaxi_one_q: getrf failed, info = {}", info_f);

  int info_s1 = nda::lapack::getrs(C_f, Z_f, ipiv);
  utils::check(info_s1 == 0, "lr_thc::xi_Deltaxi_one_q: getrs(Ξ) failed, info = {}", info_s1);

  nda::array<ComplexType, 2> Xi_P_G(Np, nG_red);
  for (long u = 0; u < Np; ++u)
    for (long g = 0; g < nG_red; ++g) Xi_P_G(u, g) = Z_f(u, g);

  // Build RHS for δΞ: rhs = δZ − δC · Ξ  (copy into Fortran layout directly).
  nda::matrix<ComplexType, nda::F_layout> rhs_f(Np, nG_red);
  for (long u = 0; u < Np; ++u)
    for (long g = 0; g < nG_red; ++g) rhs_f(u, g) = DeltaZ_G(u, g);
  nda::blas::gemm(ComplexType(-1.0), DeltaC_q, Xi_P_G, ComplexType(1.0), rhs_f);

  int info_s2 = nda::lapack::getrs(C_f, rhs_f, ipiv);
  utils::check(info_s2 == 0, "lr_thc::xi_Deltaxi_one_q: getrs(δΞ) failed, info = {}", info_s2);

  xi_Deltaxi_on_Gred out;
  out.Xi.resize(Np, nG_red);
  out.DeltaXi.resize(Np, nG_red);
  for (long u = 0; u < Np; ++u)
    for (long g = 0; g < nG_red; ++g) {
      out.Xi(u, g)  = Xi_P_G(u, g);
      out.DeltaXi(u, g) = rhs_f(u, g);
    }
  out.gv_to_fft.resize(nG_red);
  for (long g = 0; g < nG_red; ++g) out.gv_to_fft(g) = cache.gv_to_fft(g);
  out.N1 = N1; out.N2 = N2; out.N3 = N3;
  out.iq_target = iq_target;
  Timer.stop("xi_Deltaxi_one_q");
  return out;
}

std::tuple<nda::array<ComplexType, 2>, nda::array<ComplexType, 2>>
lr_thc::compute_delta_Xi(
    std::string const& Deltapsi_prefix,
    nda::array<double, 1> const& q_vec_cryst,
    nda::array<double, 2> const& delta_r_P,
    nda::array<double, 1> const& q_phonon_cryst)
{
  auto cache = build_kcache(Deltapsi_prefix, delta_r_P, q_phonon_cryst);
  auto g_out = xi_Deltaxi_one_q(cache, q_vec_cryst);
  long const Np     = g_out.Xi.shape(0);
  long const nG_red = g_out.Xi.shape(1);
  long const nnr    = g_out.N1 * g_out.N2 * g_out.N3;

  nda::array<ComplexType, 2> Xi_full(Np, nnr);
  nda::array<ComplexType, 2> DeltaXi_full(Np, nnr);
  Xi_full()      = ComplexType(0.0);
  DeltaXi_full() = ComplexType(0.0);
  for (long g = 0; g < nG_red; ++g) {
    long n = g_out.gv_to_fft(g);
    for (long u = 0; u < Np; ++u) {
      Xi_full     (u, n) = g_out.Xi     (u, g);
      DeltaXi_full(u, n) = g_out.DeltaXi(u, g);
    }
  }
  print_timers();
  return {std::move(Xi_full), std::move(DeltaXi_full)};
}

// Stage C for a single q: Ξ^q, δΞ^q on G_red → δV^q via √v(G+q) scaling + gemm.
// δV = (δZ · Z† + Z · δZ†) / (Vol · N_k) with Z = Ξ · √v(G+q). V itself is not
// computed — it is already available from the ordinary THC decomposition carried
// by _reader. Shared by compute_delta_V and compute_delta_V_batched.
// Mutates g_out.Xi / g_out.DeltaXi (scales both by √v(G+q)).
void lr_thc::build_dV_from_xi(
    xi_Deltaxi_on_Gred& g_out,
    nda::array<ComplexType, 2>& DeltaV_out)
{
  Timer.start("build_dV_from_xi");
  using nda::range;
  range::all_t all;

  auto const& mf = *_reader.MF();
  auto const& rho_g = _reader.rho_g();

  long const Np     = g_out.Xi.shape(0);
  long const nG_red = g_out.Xi.shape(1);

  auto Q = mf.Qpts()(g_out.iq_target, all);
  nda::array<RealType, 1> v_zero = {0.0, 0.0, 0.0};
  memory::unified_array<ComplexType, 1> sqrtVg(rho_g.size());
  _reader.vG().evaluate(sqrtVg, mf.lattv(), rho_g.g_vectors(), v_zero, Q);
  for (auto& v : sqrtVg) v = std::sqrt(v);

  for (long u = 0; u < Np; ++u)
    for (long g = 0; g < nG_red; ++g) {
      g_out.Xi(u, g)      *= sqrtVg(g);
      g_out.DeltaXi(u, g) *= sqrtVg(g);
    }

  nda::matrix<ComplexType> DeltaV_q(Np, Np);
  DeltaV_q() = ComplexType(0.0);

  nda::blas::gemm(ComplexType(1.0),
                  g_out.DeltaXi, nda::dagger(g_out.Xi),
                  ComplexType(0.0), DeltaV_q);
  nda::blas::gemm(ComplexType(1.0),
                  g_out.Xi, nda::dagger(g_out.DeltaXi),
                  ComplexType(1.0), DeltaV_q);

  // δV is returned in the same convention as V stored in thc_reader_t::_dZ,
  // which is Z·Z†/Vol: the `_dZ *= _nkpts` in thc_reader_t::build() cancels
  // the 1/Nk that thc::intvec_impl applies to C_quv. Hence no 1/Nk here.
  ComplexType const scale = ComplexType(1.0) / mf.volume();
  DeltaV_out.resize(Np, Np);
  for (long u = 0; u < Np; ++u)
    for (long v = 0; v < Np; ++v)
      DeltaV_out(u, v) = scale * DeltaV_q(u, v);
  Timer.stop("build_dV_from_xi");
}

void lr_thc::print_timers()
{
  app_log(2, "\n  LR_THC timers");
  app_log(2, "  -------------");
  app_log(2, "    build_kcache:     {0:8.3f} sec  {1:4d} calls",
          Timer.elapsed("build_kcache"),     Timer.number_of_calls("build_kcache"));
  app_log(2, "    xi_Deltaxi_one_q: {0:8.3f} sec  {1:4d} calls",
          Timer.elapsed("xi_Deltaxi_one_q"), Timer.number_of_calls("xi_Deltaxi_one_q"));
  app_log(2, "    build_dV_from_xi: {0:8.3f} sec  {1:4d} calls",
          Timer.elapsed("build_dV_from_xi"), Timer.number_of_calls("build_dV_from_xi"));
  app_log(2, "");
}

nda::array<ComplexType, 2> lr_thc::compute_delta_V(
    std::string const& Deltapsi_prefix,
    nda::array<double, 1> const& q_vec_cryst,
    nda::array<double, 2> const& delta_r_P,
    nda::array<double, 1> const& q_phonon_cryst)
{
  auto cache = build_kcache(Deltapsi_prefix, delta_r_P, q_phonon_cryst);
  auto g_out = xi_Deltaxi_one_q(cache, q_vec_cryst);

  nda::array<ComplexType, 2> DeltaV_q;
  build_dV_from_xi(g_out, DeltaV_q);
  print_timers();
  return DeltaV_q;
}

nda::array<ComplexType, 3> lr_thc::compute_delta_V_batched(
    std::string const& Deltapsi_prefix,
    nda::array<double, 2> const& q_vec_cryst_list,
    nda::array<double, 2> const& delta_r_P,
    nda::array<double, 1> const& q_phonon_cryst)
{
  utils::check(q_vec_cryst_list.shape(1) == 3,
               "lr_thc::compute_delta_V_batched: q_vec_cryst_list must have shape (nq, 3); got ({}, {}).",
               q_vec_cryst_list.shape(0), q_vec_cryst_list.shape(1));
  long const nq = q_vec_cryst_list.shape(0);

  // Stage A: build the q-independent per-k cache ONCE (expensive I/O + FFT).
  auto cache = build_kcache(Deltapsi_prefix, delta_r_P, q_phonon_cryst);
  long const Np = cache.Np;

  nda::array<ComplexType, 3> out(nq, Np, Np);
  out() = ComplexType(0.0);

  // Stage B + C: loop over q. Ξ^q / δΞ^q are materialized per q; no inter-q fusion.
  for (long iq = 0; iq < nq; ++iq) {
    nda::array<double, 1> q_vec_cryst(3);
    for (int a = 0; a < 3; ++a) q_vec_cryst(a) = q_vec_cryst_list(iq, a);

    auto g_out = xi_Deltaxi_one_q(cache, q_vec_cryst);

    nda::array<ComplexType, 2> DeltaV_q;
    build_dV_from_xi(g_out, DeltaV_q);

    for (long u = 0; u < Np; ++u)
      for (long v = 0; v < Np; ++v)
        out(iq, u, v) = DeltaV_q(u, v);
  }

  print_timers();
  return out;
}

} // namespace methods
