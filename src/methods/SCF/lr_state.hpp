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


#ifndef COQUI_LR_STATE_HPP
#define COQUI_LR_STATE_HPP

#include <optional>

#include "configuration.hpp"
#include "nda/nda.hpp"
#include "numerics/shared_array/nda.hpp"

namespace methods {

/**
 * @brief State container for a linear-response calculation.
 *
 * Groups the perturbation info and all LR response quantities for one
 * perturbation (q_vec, ΔH0). LR drivers take it alongside MBState: the
 * unperturbed many-body state stays in MBState, the response lives here.
 *
 * Note: "Delta" prefix (not "d") since "d" denotes distributed arrays.
 */
struct lr_state_t {
  template<nda::Array Array_base_t>
  using sArray_t = math::shm::shared_array<Array_base_t>;

  // Perturbation info
  std::optional<nda::array<double, 1>> q_vec;   // (3,) perturbation wavevector in crystal coords
  std::optional<nda::array<int, 1>> kpq_map;    // (nkpts,) k → k+q index mapping

  // LR quantities (indexed by k for the (k+q, k) block)
  std::optional<sArray_t<nda::array_view<ComplexType, 4>>> sDeltaH0_skij;     // input perturbation (ns, nk, nb, nb)
  std::optional<sArray_t<nda::array_view<ComplexType, 5>>> sDeltaG_tskij;     // LR Green's function (nt, ns, nk, nb, nb)
  std::optional<sArray_t<nda::array_view<ComplexType, 4>>> sDeltaDm_skij;     // LR density matrix (ns, nk, nb, nb)
  std::optional<sArray_t<nda::array_view<ComplexType, 5>>> sDeltaSigma_tskij; // LR self-energy (nt, ns, nk, nb, nb)
  std::optional<sArray_t<nda::array_view<ComplexType, 4>>> sDeltaF_skij;      // LR Fock matrix (ns, nk, nb, nb)
  std::optional<double> Delta_mu;               // chemical potential shift (q=0 only)
};

} // methods

#endif // COQUI_LR_STATE_HPP
