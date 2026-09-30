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

#ifndef NUMERICS_FFT_FFT_KR_HPP
#define NUMERICS_FFT_FFT_KR_HPP

#include <array>
#include <cmath>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include "configuration.hpp"
#include "nda/nda.hpp"
#include "utilities/check.hpp"

#if defined(ENABLE_FFTW)
#include "numerics/fft/fft_define.hpp"
#include "numerics/fft/fftw.h"
#include "numerics/fft/nda.hpp"
#endif

namespace math {
namespace fft {

/**
 * Blocked-FFT replacement for the dense k<->R Fourier-transform gemms used by
 * the LR solvers (gemm(f_Rk, ·) / gemm(f_kR, ·) in lr_rpa_pi / lr_gw).
 *
 * Design note (why the existing gemm / plain batched-FFT paths are not reused):
 * the LR arrays are (nk, ncols) row-major with k as the SLOW axis. A batched
 * FFT directly on that layout (stride = ncols) is slower than the gemm
 * (cache-hostile). The measured win (2.4–4.1x over gemm at production shapes)
 * requires transposing blocks
 * of columns into a k-contiguous scratch, FFT-ing there, and scattering back —
 * plus a permutation of the k axis into lexicographic Monkhorst-Pack mesh
 * order, which no existing wrapper provides.
 *
 * Conventions (matching utils::k_to_R_coefficients / R_to_k_coefficients with
 * unit R-weights and the lexicographic R enumeration used throughout the LR
 * code, e.g. phase_ipR in lr_rpa_pi):
 *   k_to_R : out(R, c) = (1/nk) Σ_k e^{-i k·R} in(k, c)   [= gemm(f_Rk,  in)]
 *   k_to_mR: out(R, c) = (1/nk) Σ_k e^{+i k·R} in(k, c)   [= gemm(conj(f_Rk), in)]
 *   R_to_k : out(k, c) =        Σ_R e^{+i k·R} in(R, c)   [= gemm(f_kR,  in)]
 *
 * Works only for a Gamma-centered, uniform, full Monkhorst-Pack mesh.
 * Otherwise the constructor errors.
 */
class fft_kR_t {
public:
  /**
   * @param kpts_cart - [INPUT] k-points in CARTESIAN coordinates (nk, 3)
   * @param lattv     - [INPUT] lattice vectors as rows (3, 3)
   * @param kp_grid   - [INPUT] MP mesh dimensions (3)
   * @param block     - [INPUT] columns per transpose block (cache tile)
   */
  fft_kR_t(::nda::ArrayOfRank<2> auto const& kpts_cart,
           ::nda::ArrayOfRank<2> auto const& lattv,
           ::nda::ArrayOfRank<1> auto const& kp_grid,
           long block = 128)
    : _block(block) {
#if defined(ENABLE_FFTW)
    _n[0] = kp_grid(0); _n[1] = kp_grid(1); _n[2] = kp_grid(2);
    _nk = _n[0] * _n[1] * _n[2];
    // Preconditions are the caller's responsibility (the LR path forbids
    // symmetry and guards the Γ-only case): assert loudly rather than silently
    // reverting to the gemm path on a grid the FFT cannot represent. A caller
    // that wants the fallback asks mesh_ok() first.
    std::string why;
    auto slots = mesh_slots(kpts_cart, lattv, kp_grid, &why);
    utils::check(slots.has_value(), "fft_kR_t: {}", why);
    _mesh_of_k = std::move(*slots);
    _scratch.resize(_block, _nk);
#else
    (void)kpts_cart; (void)lattv; (void)kp_grid;
    utils::check(false, "fft_kR_t: requires an FFTW build");
#endif
  }

  /**
   * Whether the constructor accepts this grid: a Γ-centered, uniform, full
   * Monkhorst-Pack mesh with more than one point, in an FFTW build. Never
   * aborts, so a caller can fall back to the gemm path instead.
   */
  static bool mesh_ok(::nda::ArrayOfRank<2> auto const& kpts_cart,
                      ::nda::ArrayOfRank<2> auto const& lattv,
                      ::nda::ArrayOfRank<1> auto const& kp_grid) {
#if defined(ENABLE_FFTW)
    return mesh_slots(kpts_cart, lattv, kp_grid, nullptr).has_value();
#else
    (void)kpts_cart; (void)lattv; (void)kp_grid;
    return false;
#endif
  }

  fft_kR_t(const fft_kR_t&) = delete;
  fft_kR_t& operator=(const fft_kR_t&) = delete;

  ~fft_kR_t() {
#if defined(ENABLE_FFTW)
    for (auto& [w, p] : _plans) impl::host::destroy_plan(p);
#endif
  }

  /// out(R, c) = (1/nk) Σ_k e^{-i k·R} in(k, c)
  void k_to_R(::nda::MemoryArrayOfRank<2> auto const& in,
              ::nda::MemoryArrayOfRank<2> auto&& out) {
    apply(in, out, /*forward=*/true, 1.0 / double(_nk),
          /*perm_in=*/true, /*perm_out=*/false);
  }

  /// out(R, c) = (1/nk) Σ_k e^{+i k·R} in(k, c)
  void k_to_mR(::nda::MemoryArrayOfRank<2> auto const& in,
               ::nda::MemoryArrayOfRank<2> auto&& out) {
    apply(in, out, /*forward=*/false, 1.0 / double(_nk),
          /*perm_in=*/true, /*perm_out=*/false);
  }

  /// out(k, c) = Σ_R e^{+i k·R} in(R, c)
  void R_to_k(::nda::MemoryArrayOfRank<2> auto const& in,
              ::nda::MemoryArrayOfRank<2> auto&& out) {
    apply(in, out, /*forward=*/false, 1.0,
          /*perm_in=*/false, /*perm_out=*/true);
  }

private:
  /**
   * Lexicographic MP-mesh slot of every input k-point, or nullopt (with the
   * reason in *why) when the grid is not a bijection onto a Γ-centered mesh:
   * wrong count, a single point, off-mesh (shifted) or duplicate
   * (symmetry-reduced) points.
   */
  static std::optional<std::vector<long>> mesh_slots(
      ::nda::ArrayOfRank<2> auto const& kpts_cart,
      ::nda::ArrayOfRank<2> auto const& lattv,
      ::nda::ArrayOfRank<1> auto const& kp_grid, std::string* why) {
    auto fail = [&](std::string msg) -> std::optional<std::vector<long>> {
      if (why) *why = std::move(msg);
      return std::nullopt;
    };
    const std::array<long, 3> n = {long(kp_grid(0)), long(kp_grid(1)), long(kp_grid(2))};
    const long nk = n[0] * n[1] * n[2];
    const long nk_in = kpts_cart.shape(0);
    if (nk_in != nk)
      return fail(fmt::format("k-point count {} does not match the MP-mesh product {}",
                              nk_in, nk));
    if (nk <= 1)
      return fail("single-k (Γ-only) grid; the caller must handle this as a "
                  "do-nothing copy");

    constexpr double tpi = 2.0 * M_PI;
    constexpr double tol = 1e-8;
    std::vector<long> slot(nk);
    std::vector<bool> seen(nk, false);
    // Map each input k-point's crystal coords to its lexicographic MP-mesh slot.
    for (long q = 0; q < nk; ++q) {
      std::array<long, 3> m;
      for (int i = 0; i < 3; ++i) {
        // crystal coordinate: k·a_i / 2π
        double kcry = (kpts_cart(q, 0) * lattv(i, 0) +
                       kpts_cart(q, 1) * lattv(i, 1) +
                       kpts_cart(q, 2) * lattv(i, 2)) / tpi;
        double md = kcry * n[i];
        long mi = std::llround(md);
        if (std::abs(md - double(mi)) > tol)
          return fail(fmt::format("shifted / non-Monkhorst-Pack k-grid not supported "
                                  "(off-mesh crystal coord {})", md));
        mi %= n[i];
        if (mi < 0) mi += n[i];
        m[i] = mi;
      }
      long p = (m[0] * n[1] + m[1]) * n[2] + m[2];
      if (seen[p])
        return fail("non-bijective k-grid (symmetry-reduced / IBZ?) not supported");
      seen[p] = true;
      slot[q] = p;
    }
    return slot;
  }

  void apply(::nda::MemoryArrayOfRank<2> auto const& in,
             ::nda::MemoryArrayOfRank<2> auto&& out,
             bool forward, double scale, bool perm_in, bool perm_out) {
#if defined(ENABLE_FFTW)
    utils::check(in.shape(0) == _nk && out.shape(0) == _nk &&
                 in.shape(1) == out.shape(1),
                 "fft_kR_t::apply: shape mismatch in=({},{}) out=({},{}) nk={}",
                 in.shape(0), in.shape(1), out.shape(0), out.shape(1), _nk);
    utils::check(in.indexmap().is_contiguous() && out.indexmap().is_contiguous(),
                 "fft_kR_t::apply: arrays must be contiguous");
    const long nc = in.shape(1);
    ComplexType* s = _scratch.data();  // raw buffer for the in-place FFT

    // Process columns in blocks of width <= _block: transpose a block into
    // k-contiguous scratch, FFT in place, then transpose (scatter) back.
    for (long c0 = 0; c0 < nc; c0 += _block) {
      const long b = std::min(_block, nc - c0);
      auto& plan = plan_for_width(b);
      // gather: _scratch(j, p) = in(r, c0 + j), p = mesh index of row r
      for (long r = 0; r < _nk; ++r) {
        const long p = perm_in ? _mesh_of_k[r] : r;
        for (long j = 0; j < b; ++j) _scratch(j, p) = in(r, c0 + j);
      }
      if (forward) impl::host::fwdfft(plan, s, s);
      else         impl::host::invfft(plan, s, s);
      // scatter (with scale): out(r, c0 + j) = scale * _scratch(j, p)
      for (long r = 0; r < _nk; ++r) {
        const long p = perm_out ? _mesh_of_k[r] : r;
        for (long j = 0; j < b; ++j) out(r, c0 + j) = scale * _scratch(j, p);
      }
    }
#else
    (void)in; (void)out; (void)forward; (void)scale; (void)perm_in; (void)perm_out;
    utils::check(false, "fft_kR_t::apply: built without FFTW");
#endif
  }

#if defined(ENABLE_FFTW)
  // Batched-FFT plan for `width` (<= _block) columns, built on first use and
  // cached. fft_kR_t is reused across LR iterations, so plans are built once.
  fftplan_t& plan_for_width(long width) {
    utils::check(width <= _block, "fft_kR_t: plan width {} exceeds block {}", width, _block);

    // find if plan of desired width already exists
    auto it = _plans.find(width);
    if (it != _plans.end()) return it->second;

    // plan of desired width does not exist, generate one
    // scratch space for FFT, reuse _scratch for all widths
    // (FFT_MEASURE may scribble on it during planning; no live data here)
    auto s4 = ::nda::reshape(_scratch(::nda::range(width), ::nda::range::all),
                             std::array<long, 4>{width, _n[0], _n[1], _n[2]});
    auto plan = create_plan_many(s4, FFT_MEASURE);
    auto [pos, inserted] = _plans.emplace(width, plan);
    return pos->second;
  }
#endif

  // Cache tile for the hand-rolled transpose+FFT+transpose.
  // Columns are processed in groups of _block. Block size 128–256 was empirically optimal.
  long _block = 128;
  std::array<long, 3> _n = {0, 0, 0};     // MP mesh dimensions (n0, n1, n2)
  long _nk = 0;                           // total k-points = n0*n1*n2
  std::vector<long> _mesh_of_k;           // _mesh_of_k[k] = lexicographic mesh slot of input k-point k
  ::nda::array<ComplexType, 2> _scratch;  // (_block, _nk) k-contiguous transpose buffer for the FFT
#if defined(ENABLE_FFTW)
  // FFT plans keyed by column width; see plan_for_width.
  std::map<long, fftplan_t> _plans;
#endif
};

} // fft
} // math

#endif // NUMERICS_FFT_FFT_KR_HPP
