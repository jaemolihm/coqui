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


#ifndef COQUI_ITER_SCF_T_HPP
#define COQUI_ITER_SCF_T_HPP

#include <string>
#include <variant>

#include "configuration.hpp"
#include "utilities/check.hpp"

#include "numerics/iter_scf/damp/damp_t.hpp"
#include "numerics/iter_scf/diis/diis_t.hpp"
#include "numerics/iter_scf/in_memory_mixing.hpp"

#include "numerics/iter_scf/iter_scf_type_e.hpp"

namespace iter_scf {
  /**
   * Iterative self-consistent algorithm, damping or DIIS: its parameters and the in-memory
   * mixing state (the previous accepted state and, for DIIS, the history).
   */
  class iter_scf_t {
  public:
    iter_scf_t(std::string alg) {
      if (string_to_alg_enum(alg) == iter_alg_e::damping) {
        _params = damp_t();
      } else if (string_to_alg_enum(alg) == iter_alg_e::DIIS) {
        _params = diis_t();
      } else {
        APP_ABORT(" iter_scf_t: Invalid type of iterative algorithm");
      }
    }

    explicit iter_scf_t(damp_t damp): _params(std::move(damp)) {}
    iter_scf_t& operator=(damp_t damp) { _params = std::move(damp); _state = {}; return *this; }

    explicit iter_scf_t(diis_t diis): _params(std::move(diis)) {}
    iter_scf_t& operator=(diis_t diis) { _params = std::move(diis); _state = {}; return *this; }

    // Mixing factor of the active algorithm.
    double mixing() const {
      return std::visit( [&](auto&& v) { return v.mixing; }, _params);
    }

    // The DIIS parameters (nullptr when the active algorithm is not DIIS).
    diis_t const* diis() const { return std::get_if<diis_t>(&_params); }

    // The element-sliced in-memory mixing state of the active algorithm.
    mixing_state& state() { return _state; }

    void metadata_log() const {
      std::visit( [&](auto&& v) { v.metadata_log(); }, _params);
    }

    // return type of iterative solver
    iter_alg_e iter_alg() const
    { return std::visit( [&](auto&& v) { return v.get_iter_alg(); }, _params); }

  private:
    std::variant<damp_t,diis_t> _params;
    mixing_state _state;
  };

} // iter_scf

#endif //COQUI_ITER_SCF_T_HPP
