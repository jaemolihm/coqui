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


#ifndef COQUI_DAMP_T_HPP
#define COQUI_DAMP_T_HPP

#include "configuration.hpp"
#include "IO/app_loggers.h"

#include "numerics/iter_scf/iter_scf_type_e.hpp"

namespace iter_scf {
  /**
   * Parameters of damping: x <- mixing * x + (1 - mixing) * x_previous.
   */
  struct damp_t {
    static constexpr iter_alg_e iter_alg = damping;
    static constexpr iter_alg_e get_iter_alg() { return iter_alg; }

    damp_t() = default;
    damp_t(double mixing_): mixing(mixing_) {}

    void metadata_log() const {
      app_log(2, "\nIterative algorithm for SCF");
      app_log(2, "-----------------------------");
      app_log(2, "  * algorithm: simple mixing");
      app_log(2, "  * mixing parameter = {}\n", mixing);
    }

    double mixing = 0.7;
  };
} // iter_scf

#endif //COQUI_DAMP_T_HPP
