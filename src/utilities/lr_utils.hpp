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


#ifndef UTILITIES_LR_UTILS_HPP
#define UTILITIES_LR_UTILS_HPP

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <string_view>
#include <utility>
#include <vector>

#include "IO/app_loggers.h"
#include "utilities/check.hpp"
#include "utilities/proc_grid_partition.hpp"
#include "utilities/element_partition.hpp"
#include "numerics/distributed_array/nda.hpp"
#include "mpi3/communicator.hpp"
#include "nda/nda.hpp"
#include "nda/blas.hpp"

namespace utils {

/**
 * @brief Compute k+q mapping for linear response calculations
 *
 * Given a k-point grid and a perturbation wavevector q, compute the mapping
 * kpq_map[ik] = ik' where k[ik] + q = k[ik'] (mod G).
 *
 * @param kpts_crys   - [INPUT] k-points in crystal coordinates (nkpts, 3)
 * @param q_vec       - [INPUT] perturbation wavevector in crystal coordinates (3,)
 * @param kpq_map     - [OUTPUT] k → k+q index mapping (nkpts,)
 * @param threshold   - [INPUT] tolerance for k-point matching
 */
inline void calculate_kpq_map(nda::ArrayOfRank<2> auto const& kpts_crys,
                              nda::ArrayOfRank<1> auto const& q_vec,
                              nda::ArrayOfRank<1> auto&& kpq_map,
                              double threshold = 1e-6) {
  long nkpts = kpts_crys.shape(0);
  utils::check(kpts_crys.shape(1) == 3, "calculate_kpq_map: kpts_crys.shape(1) != 3");
  utils::check(q_vec.shape(0) == 3, "calculate_kpq_map: q_vec.shape(0) != 3");
  utils::check(kpq_map.shape(0) == nkpts, "calculate_kpq_map: kpq_map size mismatch");

  kpq_map() = -1;

  for (long ik = 0; ik < nkpts; ++ik) {
    // k + q in crystal coordinates
    double kpq0 = kpts_crys(ik, 0) + q_vec(0);
    double kpq1 = kpts_crys(ik, 1) + q_vec(1);
    double kpq2 = kpts_crys(ik, 2) + q_vec(2);

    // Find k' such that k' = k + q (mod G)
    for (long ikp = 0; ikp < nkpts; ++ikp) {
      double d0 = std::abs(kpts_crys(ikp, 0) - kpq0);
      double d1 = std::abs(kpts_crys(ikp, 1) - kpq1);
      double d2 = std::abs(kpts_crys(ikp, 2) - kpq2);

      // Apply periodic boundary conditions
      d0 -= std::floor(d0);
      d1 -= std::floor(d1);
      d2 -= std::floor(d2);
      d0 -= std::round(d0);
      d1 -= std::round(d1);
      d2 -= std::round(d2);

      if (d0 * d0 + d1 * d1 + d2 * d2 < threshold * threshold) {
        kpq_map(ik) = ikp;
        break;
      }
    }

    utils::check(kpq_map(ik) >= 0,
                 "calculate_kpq_map: Could not find k+q for ik={}, k=({}, {}, {}), q=({}, {}, {})",
                 ik, kpts_crys(ik, 0), kpts_crys(ik, 1), kpts_crys(ik, 2),
                 q_vec(0), q_vec(1), q_vec(2));
  }
}

/**
 * @brief Check if q is commensurate with the k-point grid
 *
 * @param kpts_crys   - [INPUT] k-points in crystal coordinates (nkpts, 3)
 * @param q_vec       - [INPUT] perturbation wavevector in crystal coordinates (3,)
 * @param threshold   - [INPUT] tolerance for k-point matching
 * @return true if q is commensurate, false otherwise
 */
inline bool is_q_commensurate(nda::ArrayOfRank<2> auto const& kpts_crys,
                              nda::ArrayOfRank<1> auto const& q_vec,
                              double threshold = 1e-6) {
  long nkpts = kpts_crys.shape(0);

  for (long ik = 0; ik < nkpts; ++ik) {
    double kpq0 = kpts_crys(ik, 0) + q_vec(0);
    double kpq1 = kpts_crys(ik, 1) + q_vec(1);
    double kpq2 = kpts_crys(ik, 2) + q_vec(2);

    bool found = false;
    for (long ikp = 0; ikp < nkpts; ++ikp) {
      double d0 = std::abs(kpts_crys(ikp, 0) - kpq0);
      double d1 = std::abs(kpts_crys(ikp, 1) - kpq1);
      double d2 = std::abs(kpts_crys(ikp, 2) - kpq2);

      d0 -= std::floor(d0);
      d1 -= std::floor(d1);
      d2 -= std::floor(d2);
      d0 -= std::round(d0);
      d1 -= std::round(d1);
      d2 -= std::round(d2);

      if (d0 * d0 + d1 * d1 + d2 * d2 < threshold * threshold) {
        found = true;
        break;
      }
    }
    if (!found) return false;
  }
  return true;
}

/**
 * @brief Check if q is approximately zero (Gamma point)
 *
 * @param q_vec       - [INPUT] perturbation wavevector in crystal coordinates (3,)
 * @param threshold   - [INPUT] tolerance
 * @return true if q is approximately zero, false otherwise
 */
inline bool is_q_gamma(nda::ArrayOfRank<1> auto const& q_vec, double threshold = 1e-6) {
  double d0 = std::abs(q_vec(0));
  double d1 = std::abs(q_vec(1));
  double d2 = std::abs(q_vec(2));

  // Apply periodic boundary conditions
  d0 -= std::floor(d0);
  d1 -= std::floor(d1);
  d2 -= std::floor(d2);
  d0 -= std::round(d0);
  d1 -= std::round(d1);
  d2 -= std::round(d2);

  return d0 * d0 + d1 * d1 + d2 * d2 < threshold * threshold;
}

/**
 * @brief Find IBZ index of a given q-vector
 *
 * Searches k-points (k-grid == q-grid) for the one matching q_vec (mod G),
 * then maps to IBZ via qp_to_ibz.
 *
 * @param kpts_crys   - [INPUT] k-points in crystal coordinates (nkpts, 3)
 * @param q_vec       - [INPUT] perturbation wavevector in crystal coordinates (3,)
 * @param qp_to_ibz   - [INPUT] full BZ q-point → IBZ q-point mapping (nkpts,)
 * @param threshold   - [INPUT] tolerance for k-point matching
 * @return IBZ index of q_vec
 */
inline long find_q_ibz_index(nda::ArrayOfRank<2> auto const& kpts_crys,
                             nda::ArrayOfRank<1> auto const& q_vec,
                             nda::ArrayOfRank<1> auto const& qp_to_ibz,
                             double threshold = 1e-6) {
  long nkpts = kpts_crys.shape(0);
  utils::check(kpts_crys.shape(1) == 3, "find_q_ibz_index: kpts_crys.shape(1) != 3");
  utils::check(q_vec.shape(0) == 3, "find_q_ibz_index: q_vec.shape(0) != 3");
  utils::check(qp_to_ibz.shape(0) == nkpts, "find_q_ibz_index: qp_to_ibz size mismatch");

  for (long ik = 0; ik < nkpts; ++ik) {
    double d0 = std::abs(kpts_crys(ik, 0) - q_vec(0));
    double d1 = std::abs(kpts_crys(ik, 1) - q_vec(1));
    double d2 = std::abs(kpts_crys(ik, 2) - q_vec(2));

    // Apply periodic boundary conditions
    d0 -= std::floor(d0);
    d1 -= std::floor(d1);
    d2 -= std::floor(d2);
    d0 -= std::round(d0);
    d1 -= std::round(d1);
    d2 -= std::round(d2);

    if (d0 * d0 + d1 * d1 + d2 * d2 < threshold * threshold) {
      return qp_to_ibz(ik);
    }
  }

  utils::check(false, "find_q_ibz_index: Could not find q=({}, {}, {}) in k-point grid",
               q_vec(0), q_vec(1), q_vec(2));
  return -1; // unreachable
}

/**
 * @brief Find full-BZ index of a given q-vector
 *
 * Searches a full-BZ q-grid for the entry matching q_vec (mod G).
 * Mirrors find_q_ibz_index but returns the raw full-BZ index (no IBZ folding).
 *
 * @param qpts_crys   - [INPUT] q-points in crystal coordinates (nkpts, 3)
 * @param q_vec       - [INPUT] target q-vector in crystal coordinates (3,)
 * @param threshold   - [INPUT] tolerance for q-point matching
 * @return full-BZ index of q_vec
 */
inline long find_q_full_index(nda::ArrayOfRank<2> auto const& qpts_crys,
                              nda::ArrayOfRank<1> auto const& q_vec,
                              double threshold = 1e-6) {
  long nkpts = qpts_crys.shape(0);
  utils::check(qpts_crys.shape(1) == 3, "find_q_full_index: qpts_crys.shape(1) != 3");
  utils::check(q_vec.shape(0) == 3, "find_q_full_index: q_vec.shape(0) != 3");

  for (long ik = 0; ik < nkpts; ++ik) {
    double d0 = std::abs(qpts_crys(ik, 0) - q_vec(0));
    double d1 = std::abs(qpts_crys(ik, 1) - q_vec(1));
    double d2 = std::abs(qpts_crys(ik, 2) - q_vec(2));

    // Apply periodic boundary conditions
    d0 -= std::floor(d0);
    d1 -= std::floor(d1);
    d2 -= std::floor(d2);
    d0 -= std::round(d0);
    d1 -= std::round(d1);
    d2 -= std::round(d2);

    if (d0 * d0 + d1 * d1 + d2 * d2 < threshold * threshold) {
      return ik;
    }
  }

  utils::check(false, "find_q_full_index: Could not find q=({}, {}, {}) in q-point grid",
               q_vec(0), q_vec(1), q_vec(2));
  return -1; // unreachable
}

/// q-local distribution: pgrid = {tpools, 1, np_P, np_Q}
/// q-axis is local (undivided). Distributes over τ and PQ.
inline auto lr_W_q_local_dist(long nproc, long nt, long NP)
    -> std::tuple<std::array<long,4>, std::array<long,4>>
{
  long tpools = find_proc_grid_max_npools(nproc, nt, 0.2);
  long np_PQ = nproc / tpools;
  long np_P = find_proc_grid_min_diff(np_PQ, 1, 1);
  long np_Q = np_PQ / np_P;

  std::array<long, 4> pgrid = {tpools, 1, np_P, np_Q};
  // Per-dimension block sizes: each rank gets 1 SLATE tile per PQ dimension.
  // This ensures the block distribution is recognized as 2D cyclic by SLATE.
  long P_bs = std::max(NP / std::max(np_P, 1L), 1L);
  long Q_bs = std::max(NP / std::max(np_Q, 1L), 1L);
  std::array<long, 4> bsize = {1, 1, P_bs, Q_bs};

  return {pgrid, bsize};
}

/**
 * The q-dist distribution for W(iω) on the LR path: pgrid = {1, nqpools, np_P,
 * np_Q}, i.e. the ω axis local, distributed over q and then (P, Q). Square,
 * 1024-capped (P, Q) block.
 *
 * This is *the* distribution every LR W(iω) is carried on, and that is an
 * invariant rather than any one producer's preference: it is what makes both
 * Fourier transforms fuse — tau_to_w writes straight into W(iω) and w_to_tau
 * reads straight out of it, one global redistribute each instead of two — at the
 * price of the ω-side W Dyson running as a SLATE SUMMA on the (P, Q) subgrid
 * instead of a rank-local gemm. Square is what makes that legal: the C-order
 * branch of slate_ops::multiply issues slate::multiply(a, Bs, As, b, Cs), which
 * requires Bs.nt() == As.mt().
 *
 * Deliberately a DUPLICATE of solvers::scr_coulomb_fourier_t::ft_buffer_dist
 * (user, 2026-08-23), rather than a call to it: the LR and ground-state
 * distribution helpers stay separate for now. The consequence is a hard
 * constraint — this function MUST stay value-identical to ft_buffer_dist. The FT
 * engine decides internally whether to fuse by comparing the requested layout
 * against its own ft_buffer_dist, so the moment the two drift, fusion silently
 * stops firing and the LR-GW run pays two extra global redistributes per
 * iteration (~24% of it) with nothing in the output to say so. Two things enforce
 * the equality: the guard at the top of lr_scr_coulomb_t::solve_lr_dyson_W, which
 * compares against ft_buffer_dist at runtime, and the equality sweep in
 * test_slate.cpp ("ft_buffer_dist" test case) at build time.
 *
 * Planned direction: fold this and ft_buffer_dist into one distribution_utils
 * that owns every distribution pattern, at which point the duplication goes away.
 *
 * Returns a std::pair, mirroring ft_buffer_dist, so the two can be compared
 * directly. nw_half does not enter the layout (the ω axis is undivided); it is in
 * the signature to name the array's shape. Argument order follows the gshape
 * order {nw_half, nq, NP, NP}.
 */
inline auto lr_W_omega_dist([[maybe_unused]] long nproc,
                            [[maybe_unused]] long nw_half, long nq, long NP)
    -> std::pair<std::array<long,4>, std::array<long,4>>
{
  std::array<long, 4> b_pgrid = {1, 1, 1, 1};
  std::array<long, 4> b_bsize = {1, 1, 1, 1};
  b_pgrid[1] = find_proc_grid_max_npools(nproc, nq, 0.2);
  long np_PQ = nproc / b_pgrid[1];
  if (NP * NP >= np_PQ) {
    b_pgrid[2] = find_proc_grid_min_diff(np_PQ, NP, NP);
    b_pgrid[3] = np_PQ / b_pgrid[2];
  } else {
    check(np_PQ == 1,
          "lr_W_omega_dist: PQ too small for proc count (NP*NQ < np_PQ)");
  }
  b_bsize[2] = std::min({1024L, NP / std::max(b_pgrid[2], 1L),
                                NP / std::max(b_pgrid[3], 1L)});
  b_bsize[2] = std::max(b_bsize[2], 1L);
  b_bsize[3] = b_bsize[2];
  return {b_pgrid, b_bsize};
}

/// Aux-basis kernel distribution: pgrid = {1, np_P, np_Q} over (q, P, Q), q
/// undivided. The tiling lr_hf's exchange path fetches the Coulomb kernel on;
/// any kernel meant to be added to it (the static screened W of HSEX) has to be
/// built on the same one.
inline auto lr_aux_kernel_pgrid(long nproc) -> std::array<long, 3>
{
  long np_P = find_proc_grid_min_diff(nproc, 1, 1);
  return {1, np_P, nproc / np_P};
}

/// Validate that a distributed 4D array follows the lr_W_q_local_dist pattern:
/// pgrid[1] == 1 (q undivided).
template<typename darray_t>
void check_W_q_local_dist(const darray_t& d, const std::string& caller) {
  auto pgrid = d.grid();
  check(pgrid[1] == 1,
        "{}: expected q-local dist (pgrid[1]==1), got pgrid=({},{},{},{})",
        caller, pgrid[0], pgrid[1], pgrid[2], pgrid[3]);
}

/// Debug switch: force the gemm k<->R path (instead of the blocked FFT) in the
/// LR solvers when the env var COQUI_LR_DEBUG_GEMM_FT is set to a non-zero value.
inline bool lr_debug_gemm_ft() {
  static const bool flag = [] {
    const char* env = std::getenv("COQUI_LR_DEBUG_GEMM_FT");
    return env != nullptr && std::string_view(env) != "0";
  }();
  return flag;
}

/**
 * Transpose first two axes of a distributed 4D array: (A, B, P, Q) → (B, A, P, Q).
 *
 * Requires one of the first two axes to be undivided (pgrid=1), which allows
 * a purely local reorder with no MPI communication.
 * All LR call sites satisfy this precondition (τ-dist has pgrid[1]==1).
 */
template<typename Array_4D_t, typename communicator_t>
auto transpose_axes_01(
    memory::darray_t<Array_4D_t, communicator_t>& d_in,
    communicator_t& comm) {
  auto [g0, g1, g2, g3] = d_in.grid();
  auto [s0, s1, s2, s3] = d_in.global_shape();
  auto [b0, b1, b2, b3] = d_in.block_size();

  check(g0 == 1 || g1 == 1,
      "transpose_axes_01: one of the first two axes must be undivided "
      "(pgrid[0]={}, pgrid[1]={})", g0, g1);

  auto d_out = math::nda::make_distributed_array<Array_4D_t>(
      comm, {g1, g0, g2, g3}, {s1, s0, s2, s3}, {b1, b0, b2, b3});

  auto in_loc = d_in.local();
  auto out_loc = d_out.local();
  long n0 = d_in.local_shape()[0];
  long n1 = d_in.local_shape()[1];
  check(d_out.local_shape()[0] == n1 && d_out.local_shape()[1] == n0,
      "transpose_axes_01: output local shape ({},{}) != expected ({},{})",
      d_out.local_shape()[0], d_out.local_shape()[1], n1, n0);
  for (long i01 = 0; i01 < n0 * n1; ++i01) {
    long i0 = i01 / n1;
    long i1 = i01 % n1;
    out_loc(i1, i0, nda::ellipsis{}) = in_loc(i0, i1, nda::ellipsis{});
  }
  comm.barrier();
  return d_out;
}

/**
 * @brief Distributed Frobenius norm and successive-iterate difference norm.
 *
 * Computes ||A||_F and ||A - A_prev||_F with the work distributed over the
 * ranks of `comm`. Since the Frobenius norm is independent of how the elements
 * are grouped, the array is viewed as 1D and each rank takes a contiguous slice
 * of the element range; the two partial sums are then all-reduced so every rank
 * returns identical totals.
 *
 * `A` and `A_prev` must be contiguous (e.g. a shared-array `.local()` view) and
 * share the same shape. The difference norm reads `A - A_prev` lazily (no
 * temporary is allocated). When `compute_diff` is false the second component is
 * returned as 0.
 *
 * Typical use: pass `mpi.node_comm` with node-replicated shared arrays; every
 * rank on the node then holds the full norm. (A trailing global broadcast is
 * still appropriate if exact cross-node bit-agreement is required.)
 *
 * @param comm         - [INPUT] communicator the block work is striped over
 * @param A            - [INPUT] current array/view, rank >= 2
 * @param A_prev       - [INPUT] previous array/view (same shape as A)
 * @param compute_diff - [INPUT] whether to also compute ||A - A_prev||_F
 * @return {||A||_F, ||A - A_prev||_F}
 */
template<typename Comm, typename ArrA, typename ArrB>
std::pair<double, double> lr_distributed_norm(Comm& comm,
                                              ArrA const& A,
                                              ArrB const& A_prev,
                                              bool compute_diff) {
  // The Frobenius norm is partition-invariant, so view the (contiguous) array
  // as 1D and give each rank a contiguous slice of the element range.
  const long n = A.size();
  if (compute_diff) {
    check(static_cast<long>(A_prev.size()) == n,
          "lr_distributed_norm: A and A_prev sizes differ ({} vs {})", A_prev.size(), n);
  }
  const long nr = comm.size();
  const long r = comm.rank();
  const long chunk = (n + nr - 1) / nr;
  const long i0 = std::min(r * chunk, n);
  const long i1 = std::min(i0 + chunk, n);

  double n2 = 0.0, d2 = 0.0;
  if (i1 > i0) {
    auto a_sub = nda::reshape(A, std::array<long, 1>{n})(nda::range(i0, i1));
    n2 = std::real(nda::blas::dotc(a_sub, a_sub));  // ||slice||^2 via BLAS
    if (compute_diff) {
      auto b_sub = nda::reshape(A_prev, std::array<long, 1>{A_prev.size()})(nda::range(i0, i1));
      // ||a - b||^2 as a no-alloc lazy reduction (BLAS can't take the lazy a-b).
      d2 = nda::sum(nda::map([](auto x, auto y) {
        double d = std::abs(x - y);
        return d * d;
      })(a_sub, b_sub));
    }
  }

  double acc[2] = {n2, d2};
  comm.all_reduce_in_place_n(acc, 2, std::plus<>{});
  return {std::sqrt(acc[0]), std::sqrt(acc[1])};
}

} // namespace utils

#endif // UTILITIES_LR_UTILS_HPP
