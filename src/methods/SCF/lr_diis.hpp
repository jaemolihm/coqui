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

#ifndef COQUI_LR_DIIS_HPP
#define COQUI_LR_DIIS_HPP

#include <string>
#include <tuple>
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
 *   pseudoinverse from diis_alg.hpp::compute_coefs_c1) with in-memory
 *   std::vector<Array_4D> storage instead of HDF5-backed VSpace.
 *
 * Uses difference residual: res = ΔF_new - ΔF_prev
 */
class lr_diis {
public:
  using Array_4D = nda::array<ComplexType, 4>;
  using Array_5D = nda::array<ComplexType, 5>;

  lr_diis(size_t max_subsp_size, size_t warmup_iter, double mixing)
      : _max_subsp_size(max_subsp_size),
        _warmup_iter(warmup_iter),
        _mixing(mixing),
        _B(0, 0) {}

  /**
   * @brief Perform one DIIS step (ΔF only, backward-compatible)
   */
  Array_4D next_step(const Array_4D& DeltaF_new,
                     const Array_4D& DeltaF_prev, int iter) {
    Array_5D empty;
    auto [F, _] = next_step_combined(DeltaF_new, DeltaF_prev, empty, empty, iter);
    return F;
  }

  /**
   * @brief Perform one combined DIIS step on (ΔF, ΔΣ)
   *
   * Combined residual: B(i,j) = <res_F_i, res_F_j> + <res_Sigma_i, res_Sigma_j>
   * When DeltaSigma arrays are empty (size 0), reduces to ΔF-only DIIS.
   *
   * @param DeltaF_new       - [INPUT] Newly computed ΔF
   * @param DeltaF_prev      - [INPUT] ΔF from previous iteration
   * @param DeltaSigma_new   - [INPUT] Newly computed ΔΣ (empty if no GW)
   * @param DeltaSigma_prev  - [INPUT] ΔΣ from previous iteration (empty if no GW)
   * @param iter             - [INPUT] Current iteration number (1-based)
   * @return (mixed ΔF, mixed ΔΣ)
   */
  std::tuple<Array_4D, Array_5D> next_step_combined(
      const Array_4D& DeltaF_new, const Array_4D& DeltaF_prev,
      const Array_5D& DeltaSigma_new, const Array_5D& DeltaSigma_prev,
      int iter) {

    bool has_sigma = (DeltaSigma_new.size() > 0);

    // Compute residuals
    Array_4D res_F = DeltaF_new - DeltaF_prev;
    Array_5D res_Sigma;
    if (has_sigma) {
      res_Sigma = Array_5D{DeltaSigma_new - DeltaSigma_prev};
    }

    // During warmup or with insufficient subspace: use damping
    if (iter <= static_cast<int>(_warmup_iter) + 1 || _x_vectors.size() < 2) {
      store_vectors(DeltaF_new, res_F, DeltaSigma_new, res_Sigma, has_sigma);

      app_log(2, "    DIIS: warmup iter {} -> damping (mixing={:.2f}, subspace={})",
              iter, _mixing, _x_vectors.size());

      if (_mixing < 1.0) {
        Array_4D F_mixed{_mixing * DeltaF_new + (1.0 - _mixing) * DeltaF_prev};
        Array_5D Sigma_mixed;
        if (has_sigma) {
          Sigma_mixed = Array_5D{_mixing * DeltaSigma_new + (1.0 - _mixing) * DeltaSigma_prev};
        }
        return {std::move(F_mixed), std::move(Sigma_mixed)};
      }
      Array_5D Sigma_out;
      if (has_sigma) Sigma_out = Array_5D{DeltaSigma_new};
      return {Array_4D{DeltaF_new}, std::move(Sigma_out)};
    }

    // DIIS extrapolation
    store_vectors(DeltaF_new, res_F, DeltaSigma_new, res_Sigma, has_sigma);

    auto C = compute_coefs();

    double c_norm = 0.0;
    for (long i = 0; i < C.size(); ++i) {
      c_norm += std::abs(C(i));
    }

    if (c_norm < 1e-14) {
      app_log(2, "    DIIS: extrapolation failed (ill-conditioned B) -> fallback to damping");
      if (_mixing < 1.0) {
        Array_4D F_mixed{_mixing * DeltaF_new + (1.0 - _mixing) * DeltaF_prev};
        Array_5D Sigma_mixed;
        if (has_sigma) {
          Sigma_mixed = Array_5D{_mixing * DeltaSigma_new + (1.0 - _mixing) * DeltaSigma_prev};
        }
        return {std::move(F_mixed), std::move(Sigma_mixed)};
      }
      Array_5D Sigma_out;
      if (has_sigma) Sigma_out = Array_5D{DeltaSigma_new};
      return {Array_4D{DeltaF_new}, std::move(Sigma_out)};
    }

    Array_4D result_F = make_linear_comb_4d(C);
    Array_5D result_Sigma;
    if (has_sigma) {
      result_Sigma = make_linear_comb_5d(C);
    }

    app_log(2, "    DIIS: extrapolation with subspace size {}", _x_vectors.size());

    return {std::move(result_F), std::move(result_Sigma)};
  }

private:
  size_t _max_subsp_size;
  size_t _warmup_iter;
  double _mixing;
  std::vector<Array_4D> _x_vectors;
  std::vector<Array_4D> _res_vectors;
  std::vector<Array_5D> _x_vectors_5d;
  std::vector<Array_5D> _res_vectors_5d;
  nda::matrix<ComplexType> _B;

  /**
   * @brief Frobenius inner product of two arrays (works for any rank via reshape)
   */
  template<int N>
  ComplexType dot_prod(const nda::array<ComplexType, N>& a,
                       const nda::array<ComplexType, N>& b) const {
    auto a_flat = nda::reshape(a, a.size());
    auto b_flat = nda::reshape(b, b.size());
    ComplexType result = 0.0;
    for (long i = 0; i < a_flat.size(); ++i) {
      result += std::conj(a_flat(i)) * b_flat(i);
    }
    return result;
  }

  /**
   * @brief Combined overlap: <res_F_a, res_F_b> + <res_Sigma_a, res_Sigma_b>
   */
  ComplexType combined_overlap(size_t i, size_t j) const {
    ComplexType result = dot_prod(_res_vectors[i], _res_vectors[j]);
    if (!_res_vectors_5d.empty() && _res_vectors_5d[i].size() > 0) {
      result += dot_prod(_res_vectors_5d[i], _res_vectors_5d[j]);
    }
    return result;
  }

  /**
   * @brief Store trial vectors and residuals, update B-matrix
   */
  void store_vectors(const Array_4D& x_4d, const Array_4D& res_4d,
                     const Array_5D& x_5d, const Array_5D& res_5d,
                     bool has_sigma) {
    _x_vectors.push_back(x_4d);
    _res_vectors.push_back(res_4d);
    if (has_sigma) {
      _x_vectors_5d.push_back(x_5d);
      _res_vectors_5d.push_back(res_5d);
    } else if (!_x_vectors_5d.empty()) {
      // Keep vectors aligned: store empty
      _x_vectors_5d.push_back(Array_5D{});
      _res_vectors_5d.push_back(Array_5D{});
    }

    update_B();

    if (_x_vectors.size() > _max_subsp_size) {
      purge_oldest();
    }
  }

  /**
   * @brief Rebuild last row/column of B-matrix using combined overlap
   */
  void update_B() {
    size_t n = _x_vectors.size();
    nda::matrix<ComplexType> Bnew(n, n);
    Bnew() = 0;

    // Copy existing entries
    size_t n_old = n - 1;
    for (size_t i = 0; i < n_old; ++i) {
      for (size_t j = 0; j < n_old; ++j) {
        Bnew(i, j) = _B(i, j);
      }
    }

    // Compute new overlaps using combined overlap
    for (size_t i = 0; i < n_old; ++i) {
      Bnew(i, n - 1) = combined_overlap(i, n - 1);
      Bnew(n - 1, i) = std::conj(Bnew(i, n - 1));
    }
    Bnew(n - 1, n - 1) = combined_overlap(n - 1, n - 1);

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

    _x_vectors.erase(_x_vectors.begin());
    _res_vectors.erase(_res_vectors.begin());
    if (!_x_vectors_5d.empty()) {
      _x_vectors_5d.erase(_x_vectors_5d.begin());
      _res_vectors_5d.erase(_res_vectors_5d.begin());
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

    app_log(2, "    DIIS: B condition number = {:.2e}", cond);

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
   * @brief Build linear combination of 4D trial vectors
   */
  Array_4D make_linear_comb_4d(const nda::array<double, 1>& C) const {
    Array_4D result(_x_vectors[0].shape());
    result() = 0.0;
    for (size_t i = 0; i < _x_vectors.size(); ++i) {
      result += C(i) * _x_vectors[i];
    }
    return result;
  }

  /**
   * @brief Build linear combination of 5D trial vectors
   */
  Array_5D make_linear_comb_5d(const nda::array<double, 1>& C) const {
    Array_5D result(_x_vectors_5d[0].shape());
    result() = 0.0;
    for (size_t i = 0; i < _x_vectors_5d.size(); ++i) {
      result += C(i) * _x_vectors_5d[i];
    }
    return result;
  }
};

} // namespace methods

#endif // COQUI_LR_DIIS_HPP
