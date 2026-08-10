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


#include "methods/scr_coulomb/lr_scr_coulomb_t.hpp"

namespace methods {
namespace solvers {

  lr_scr_coulomb_t::lr_scr_coulomb_t(const imag_axes_ft::IAFT* ft,
                                     nda::array<double, 1> q_pert)
    : _ft(ft), _Timer(), _scr_fourier(ft), _q_pert(std::move(q_pert)) {

    utils::check(_q_pert.size() == 3,
                 "lr_scr_coulomb_t: q_pert must have size 3, got {}",
                 _q_pert.size());
    _is_q_gamma = utils::is_q_gamma(_q_pert);

    for (auto& v: {"SOLVE_LR_DYSON_W", "EVALUATE_LR_W",
                    "LR_W_FT_TAU_TO_W", "LR_W_FT_W_TO_TAU",
                    "LR_W_COMPUTE_W_FULL", "LR_W_COMM_QPQ",
                    "LR_W_QPOOL_REDISTRIBUTE"}) {
      _Timer.add(v);
    }
  }

  // Note: lr_dyson_W_in_place and solve_lr_dyson_W are templates with
  // THC_ERI auto& (concept-constrained), so they cannot be explicitly instantiated
  // here. They are implicitly instantiated where called (in MBPT_drivers.cpp via
  // lr_scr_coulomb_t.icc include).

} // solvers
} // methods
