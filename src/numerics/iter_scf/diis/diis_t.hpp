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


#ifndef COQUI_DIIS_T_HPP
#define COQUI_DIIS_T_HPP

#include <algorithm>
#include <cctype>
#include <string>

#include "configuration.hpp"
#include "utilities/check.hpp"
#include "IO/app_loggers.h"

#include "numerics/iter_scf/iter_scf_type_e.hpp"

namespace iter_scf {
  /**
   * Parameters of DIIS: the damping factor of the warmup and fallback steps, the number
   * of stored (trial, residual) pairs, the number of damping-only steps, and the residual
   * definition ("commutator" or "vector_diff"). The history lives in iter_scf::mixing_state.
   */
  struct diis_t {
    static constexpr iter_alg_e iter_alg = DIIS;
    static constexpr iter_alg_e get_iter_alg() { return iter_alg; }

    diis_t() = default;
    diis_t(double mixing_, size_t max_subsp_size_, size_t warmup_iter_,
           std::string residual_type_ = "commutator"):
        mixing(mixing_),
        max_subsp_size(max_subsp_size_),
        warmup_iter(warmup_iter_),
        residual_type(normalize_residual_type(std::move(residual_type_))) {}

    void metadata_log() const {
      app_log(2, "\nIterative algorithm for SCF");
      app_log(2, "-----------------------------");
      // Dyson SCF: frequency-dependent commutator residual; qp SCF: residual_type
      app_log(2, "  * algorithm: DIIS");
      app_log(2, "    Dyson SCF: frequency-dependent commutator DIIS\n"
                 "               P. Pokhilko, C.-N. Yeh, D. Zgid. J. Chem. Phys., 2022, 156, 094101\n"
                 "               https://doi.org/10.1063/5.0082586");
      app_log(2, "  * DIIS parameters: ");
      app_log(2, "    mixing                   = {}", mixing);
      app_log(2, "    max subspace size        = {}", max_subsp_size);
      app_log(2, "    warmup iteration         = {}", warmup_iter);
      app_log(2, "    residual type (qp SCF)   = {}\n", residual_type);
    }

    double mixing = 0.2;
    size_t max_subsp_size = 5;
    size_t warmup_iter = 5;
    std::string residual_type = "commutator";

  private:
    static std::string normalize_residual_type(std::string residual_type) {
      std::transform(residual_type.begin(), residual_type.end(), residual_type.begin(),
                     [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
      utils::check(residual_type == "commutator" or residual_type == "vector_diff",
                   "diis_t::normalize_residual_type: unknown residual_type = {}. Valid options are \"commutator\" and \"vector_diff\"",
                   residual_type);
      return residual_type;
    }
  };
} // iter_scf

#endif //COQUI_DIIS_T_HPP
