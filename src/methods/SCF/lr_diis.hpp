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

#ifndef COQUI_LR_DIIS_HPP
#define COQUI_LR_DIIS_HPP

#include <algorithm>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
#include <numeric>
#include <cmath>

#include "nda/nda.hpp"
#include "nda/blas.hpp"
#include "nda/linalg/eigenelements.hpp"
#include "IO/app_loggers.h"

namespace methods {

/**
 * @brief Parameters for LR SCF iteration algorithm
 *
 * Mirrors the ground-state iter_alg pattern (damping vs DIIS).
 */
struct lr_iter_params {
  std::string alg = "damping";
  double mixing = 1.0;
  size_t max_subsp_size = 5;
  size_t diis_warmup = 3;
};


/**
 * @brief Standalone DIIS accelerator for linear-response SCF
 *
 * Why not reuse diis_alg<Vector> / VSpace<Vector>?
 *   The existing DIIS infrastructure (numerics/iter_scf/diis/) has a hard
 *   dependency on HDF5 file I/O: VSpace stores and retrieves every vector
 *   via write_to_file/read_from_file, and overlap(i,j) reads two vectors
 *   from disk. This design is appropriate for ground-state SCF where
 *   checkpointing is essential, but for LR the arrays are tiny (e.g. 512
 *   complex numbers for Si) and the SCF loop is tight, so disk I/O would
 *   be pure overhead. The Vector interface also requires read_from_file/
 *   write_to_file methods, and the ground-state uses a composite FockSigma
 *   vector type with commutator residuals — quite different from LR where
 *   we have a single 4D ΔF with a simple difference residual.
 *
 *   This class reuses the same B-matrix solve algorithm (eigendecomposition
 *   pseudoinverse from diis_alg.hpp::compute_coefs_c1) with in-memory storage
 *   instead of HDF5-backed VSpace.
 *
 *   SPMD / distributed: next_step_combined is called on every rank of the
 *   supplied communicator. Each rank stores and operates on only its own
 *   contiguous element-slice of every trial/residual vector (std::vector<Vec1D>),
 *   so the DIIS history is partitioned across the node rather than replicated.
 *   The B-matrix overlaps are formed from per-rank partial dot products combined
 *   with a single all_reduce, so every rank computes identical coefficients.
 *
 * Uses difference residual: res = ΔF_new - ΔF_prev
 */
class lr_diis {
public:
  using Array_4D = nda::array<ComplexType, 4>;
  using Array_5D = nda::array<ComplexType, 5>;
  using Vec1D    = nda::array<ComplexType, 1>;

  lr_diis(size_t max_subsp_size, size_t warmup_iter, double mixing)
      : _max_subsp_size(max_subsp_size),
        _warmup_iter(warmup_iter),
        _mixing(mixing),
        _B(0, 0) {}

  /**
   * @brief One combined DIIS step on (ΔF, ΔΣ), distributed over `comm`.
   *
   * SPMD: called on every rank of `comm`. Each rank owns a contiguous element
   * slice of the flattened ΔF/ΔΣ and stores only that slice of every trial /
   * residual vector, so the DIIS history is partitioned across the node (no
   * per-rank full-array replication). The B-matrix overlaps are formed from
   * per-rank partial dot products combined with a single all_reduce, so every
   * rank holds the identical B and computes identical coefficients. The mixed
   * result is written in place into each rank's slice of DeltaF / DeltaSigma.
   *
   * Combined residual: B(i,j) = <res_F_i, res_F_j> + <res_Sigma_i, res_Sigma_j>.
   *
   * @param comm             - [INPUT] communicator the array work is striped over
   * @param DeltaF           - [IN/OUT] ΔF (new on entry, mixed on exit); pass .local()
   * @param DeltaF_prev      - [INPUT]  ΔF from previous iteration
   * @param DeltaSigma       - [IN/OUT] ΔΣ (new on entry, mixed on exit); empty if no GW
   * @param DeltaSigma_prev  - [INPUT]  ΔΣ from previous iteration (empty if no GW)
   * @param iter             - [INPUT]  current iteration number (1-based)
   */
  template<typename Comm, typename FView, typename FPrev, typename SView, typename SPrev>
  void next_step_combined(Comm& comm,
                          FView DeltaF, FPrev const& DeltaF_prev,
                          SView DeltaSigma, SPrev const& DeltaSigma_prev,
                          int iter) {
    const bool has_sigma = (DeltaSigma.size() > 0);
    const long nr = comm.size();
    const long r  = comm.rank();

    // This rank's contiguous slice of the flattened ΔF.
    const long nF = DeltaF.size();
    auto [fF0, fF1] = rank_slice(nF, nr, r);
    auto F_loc      = nda::reshape(DeltaF,      std::array<long, 1>{nF})(nda::range(fF0, fF1));
    auto F_prev_loc = nda::reshape(DeltaF_prev, std::array<long, 1>{nF})(nda::range(fF0, fF1));

    long sS0 = 0, sS1 = 0;
    const long nS = has_sigma ? static_cast<long>(DeltaSigma.size()) : 0;
    if (has_sigma) std::tie(sS0, sS1) = rank_slice(nS, nr, r);

    // Damping write (warmup / ill-conditioned fallback): out_slice <- mixing*new +
    // (1-mixing)*prev. mixing >= 1 leaves the arrays unchanged (out == new).
    auto write_damping = [&]() {
      if (_mixing < 1.0) {
        F_loc = _mixing * F_loc + (1.0 - _mixing) * F_prev_loc;
        if (has_sigma) {
          auto S_loc      = nda::reshape(DeltaSigma,      std::array<long, 1>{nS})(nda::range(sS0, sS1));
          auto S_prev_loc = nda::reshape(DeltaSigma_prev, std::array<long, 1>{nS})(nda::range(sS0, sS1));
          S_loc = _mixing * S_loc + (1.0 - _mixing) * S_prev_loc;
        }
      }
    };

    // Residual + trial-vector slices (this rank's portion).
    Vec1D xF{F_loc};
    Vec1D resF{F_loc - F_prev_loc};
    Vec1D xS, resS;
    if (has_sigma) {
      auto S_loc      = nda::reshape(DeltaSigma,      std::array<long, 1>{nS})(nda::range(sS0, sS1));
      auto S_prev_loc = nda::reshape(DeltaSigma_prev, std::array<long, 1>{nS})(nda::range(sS0, sS1));
      xS   = Vec1D{S_loc};
      resS = Vec1D{S_loc - S_prev_loc};
    }

    // Decide warmup BEFORE storing (need >= 2 prior vectors to extrapolate).
    const bool warmup = (iter <= static_cast<int>(_warmup_iter) + 1 || _xF.size() < 2);

    // Store this rank's slices, then update B (one all_reduce over comm).
    _xF.push_back(std::move(xF));
    _resF.push_back(std::move(resF));
    if (has_sigma) {
      _xS.push_back(std::move(xS));
      _resS.push_back(std::move(resS));
    }

    update_B(comm, has_sigma);

    if (_xF.size() > _max_subsp_size) purge_oldest();

    if (warmup) {
      app_log(3, "    DIIS: warmup iter {} -> damping (mixing={:.2f}, subspace={})",
              iter, _mixing, _xF.size());
      write_damping();
      return;
    }

    auto C = compute_coefs();

    double c_norm = 0.0;
    for (long i = 0; i < C.size(); ++i) c_norm += std::abs(C(i));

    if (c_norm < 1e-14) {
      app_log(2, "    DIIS: extrapolation failed (ill-conditioned B) -> fallback to damping");
      write_damping();
      return;
    }

    // Extrapolate: out_slice <- Σ_i C(i) * trial_i  (in place).
    make_linear_comb_into(F_loc, _xF, C);
    if (has_sigma) {
      auto S_loc = nda::reshape(DeltaSigma, std::array<long, 1>{nS})(nda::range(sS0, sS1));
      make_linear_comb_into(S_loc, _xS, C);
    }

    app_log(3, "    DIIS: extrapolation with subspace size {}", _xF.size());
  }

private:
  size_t _max_subsp_size;
  size_t _warmup_iter;
  double _mixing;
  // History stores only THIS rank's contiguous element-slice of each flattened
  // trial / residual vector (distributed across comm), not the full arrays.
  std::vector<Vec1D> _xF, _resF;   // ΔF trial / residual slices
  std::vector<Vec1D> _xS, _resS;   // ΔΣ trial / residual slices
  nda::matrix<ComplexType> _B;

  /// Contiguous [i0, i1) slice of [0, n) owned by rank r of nr.
  static std::pair<long, long> rank_slice(long n, long nr, long r) {
    const long chunk = (n + nr - 1) / nr;
    const long i0 = std::min(r * chunk, n);
    const long i1 = std::min(i0 + chunk, n);
    return {i0, i1};
  }

  /// Local (this rank's slice) contribution to the combined overlap
  /// <res_F_i, res_F_j> + <res_Sigma_i, res_Sigma_j>.
  ComplexType local_overlap(size_t i, size_t j, bool has_sigma) const {
    ComplexType result = 0.0;
    if (_resF[i].size() > 0) result += nda::blas::dotc(_resF[i], _resF[j]);
    if (has_sigma && !_resS.empty() && _resS[i].size() > 0) {
      result += nda::blas::dotc(_resS[i], _resS[j]);
    }
    return result;
  }

  /**
   * @brief Rebuild last row/column of B using partial overlaps + one all_reduce.
   *
   * Each rank contributes the dot products over its slice; the new row is
   * all_reduced so every rank ends with the identical full B.
   */
  template<typename Comm>
  void update_B(Comm& comm, bool has_sigma) {
    const size_t n = _xF.size();
    nda::matrix<ComplexType> Bnew(n, n);
    Bnew() = 0;

    const size_t n_old = n - 1;
    for (size_t i = 0; i < n_old; ++i)
      for (size_t j = 0; j < n_old; ++j)
        Bnew(i, j) = _B(i, j);

    // New row/col: per-rank partial overlaps reduced in a single all_reduce.
    nda::array<ComplexType, 1> col(n);
    col() = 0;
    for (size_t i = 0; i < n; ++i) col(i) = local_overlap(i, n - 1, has_sigma);
    comm.all_reduce_in_place_n(col.data(), static_cast<long>(n), std::plus<>{});

    for (size_t i = 0; i < n_old; ++i) {
      Bnew(i, n - 1) = col(i);
      Bnew(n - 1, i) = std::conj(col(i));
    }
    Bnew(n - 1, n - 1) = col(n - 1);

    _B = Bnew;
  }

  /**
   * @brief Remove oldest vector from subspace and shrink B-matrix
   */
  void purge_oldest() {
    size_t n = _B.shape()[0];
    nda::matrix<ComplexType> Bnew(n - 1, n - 1);
    for (size_t i = 0; i < n - 1; ++i) {
      for (size_t j = 0; j < n - 1; ++j) {
        Bnew(i, j) = _B(i + 1, j + 1);
      }
    }
    _B = Bnew;

    _xF.erase(_xF.begin());
    _resF.erase(_resF.begin());
    if (!_xS.empty()) {
      _xS.erase(_xS.begin());
      _resS.erase(_resS.begin());
    }
  }

  /**
   * @brief Compute DIIS coefficients via eigendecomposition pseudoinverse
   *
   * Algorithm from diis_alg.hpp::compute_coefs_c1():
   * 1. Eigendecompose B (real part)
   * 2. Pseudoinverse with threshold
   * 3. Solve B^{-1} * 1 and normalize to sum=1
   */
  nda::array<double, 1> compute_coefs() const {
    auto B_real = nda::make_regular(nda::real(_B));

    nda::array<double, 1> bb(B_real.shape()[1]);
    bb() = 1.0;

    auto [eig, evecs] = nda::linalg::eigenelements(B_real);
    auto evecs_tr = nda::make_regular(nda::transpose(evecs));

    nda::matrix<double> Binv(B_real.shape()[0], B_real.shape()[1]);
    nda::matrix<double> eig_inv(B_real.shape()[0], B_real.shape()[1]);
    nda::matrix<double> I_tmp(B_real.shape()[0], B_real.shape()[1]);
    Binv() = 0;
    eig_inv() = 0;

    double eig_max = nda::max_element(eig);
    double eig_min = nda::min_element(eig);
    double cond = (std::abs(eig_min) > 1e-30) ? eig_max / eig_min : 1e30;

    const double eig_thresh = 1E-12;

    app_log(3, "    DIIS: B condition number = {:.2e}", cond);

    for (long i = 0; i < eig.size(); ++i) {
      if (eig(i) * cond > eig_thresh) {
        eig_inv(i, i) = 1.0 / eig(i);
      }
    }

    nda::blas::gemm(evecs, eig_inv, I_tmp);
    nda::blas::gemm(I_tmp, evecs_tr, Binv);

    nda::array<double, 1> x(B_real.shape()[1]);
    nda::blas::gemv(1.0, Binv, bb, 0.0, x);

    double sum = std::accumulate(x.begin(), x.end(), 0.0);
    if (std::abs(sum) < 1e-14) {
      // Ill-conditioned: return zeros to signal failure
      nda::array<double, 1> zeros(x.size());
      zeros() = 0.0;
      return zeros;
    }

    nda::array<double, 1> C(x.size());
    for (long i = 0; i < x.size(); ++i) {
      C(i) = x(i) / sum;
    }
    return C;
  }

  /**
   * @brief Write Σ_i C(i) * hist[i] into `out` (this rank's slice).
   *
   * `out` is a 1D slice view of the output array; `hist` are the matching
   * trial-vector slices.
   */
  template<typename OutView>
  void make_linear_comb_into(OutView&& out, const std::vector<Vec1D>& hist,
                             const nda::array<double, 1>& C) const {
    out() = ComplexType{0};
    for (size_t i = 0; i < hist.size(); ++i) {
      out += C(i) * hist[i];
    }
  }
};

} // namespace methods

#endif // COQUI_LR_DIIS_HPP
