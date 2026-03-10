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


#ifndef COQUI_LR_PRECOMPUTE_HPP
#define COQUI_LR_PRECOMPUTE_HPP

#include "IO/app_loggers.h"
#include "numerics/imag_axes_ft/IAFT.hpp"
#include "numerics/shared_array/nda.hpp"
#include "utilities/mpi_context.h"

namespace methods {

/**
 * @brief Precompute G(iω) in shared memory for LR Dyson equation.
 *
 * Transforms sG_tskij from tau to frequency space and stores the result in
 * a shared memory array. Node root performs the FT; all ranks on the same
 * node share the result via shared memory.
 *
 * @param mpi        - [INPUT] MPI context (comm, node_comm, internode_comm)
 * @param sG_tskij   - [INPUT] Unperturbed Green's function in tau (nt, ns, nk, nb, nb)
 * @param FT         - [INPUT] IAFT object for Fourier transform
 * @return Shared memory G(iω) array (nw, ns, nk, nb, nb)
 */
template<math::shm::SharedArray sArr_t>
auto lr_precompute_G_omega(utils::mpi_context_t<mpi3::communicator>& mpi,
                           const sArr_t& sG_tskij,
                           const imag_axes_ft::IAFT& FT) {
  auto [nts, ns, nkpts_ibz, nbnd, nbnd2] = sG_tskij.shape();
  auto nw = FT.nw_f();

  app_log(2, "lr_precompute_G_omega: FT G(tau) -> G(iw) [shared memory]");
  app_log(2, "  shape: ({}, {}, {}, {}, {}) -> ({}, {}, {}, {}, {})",
          nts, ns, nkpts_ibz, nbnd, nbnd, nw, ns, nkpts_ibz, nbnd, nbnd);

  using Array_view_5D_t = nda::array_view<ComplexType, 5>;
  auto sG_wskij = math::shm::make_shared_array<Array_view_5D_t>(
      mpi, {nw, ns, nkpts_ibz, nbnd, nbnd});

  if (mpi.node_comm.root()) {
    FT.tau_to_w(sG_tskij.local(), sG_wskij.local(), imag_axes_ft::fermi);
  }
  sG_wskij.node_sync();
  mpi.comm.barrier();

  return sG_wskij;
}

} // namespace methods

#endif // COQUI_LR_PRECOMPUTE_HPP
