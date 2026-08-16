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


#ifndef COQUI_SCR_COULOMB_FOURIER_T_H
#define COQUI_SCR_COULOMB_FOURIER_T_H

#include <array>
#include <optional>
#include <utility>

#include "configuration.hpp"
#include "nda/nda.hpp"
#include "numerics/distributed_array/nda.hpp"

#include "utilities/mpi_context.h"
#include "utilities/Timer.hpp"
#include "utilities/proc_grid_partition.hpp"

#include "numerics/imag_axes_ft/IAFT.hpp"
#include "numerics/imag_axes_ft/iaft_utils.hpp"

namespace methods {
namespace solvers {

  /**
   * @brief Distributed bosonic τ↔ω Fourier transform for THC product-basis
   *        quantities (axes (τ/ω, q, P, Q)).
   *
   * This is the narrow FT utility carved out of scr_coulomb_t. It transforms any
   * particle-hole-symmetric bosonic quantity of the screened-Coulomb pipeline —
   * both the polarizability Π and the screened interaction W (and, on the LR path,
   * ΔΠ / ΔW). The transform redistributes the leading τ/ω axis to be rank-local,
   * applies the IAFT PH-symmetric kernel locally, and redistributes back.
   */
  class scr_coulomb_fourier_t {
  public:
    explicit scr_coulomb_fourier_t(const imag_axes_ft::IAFT* ft) : _ft(ft) {
      for (auto& v : {"IMAG_FT_TtoW", "IMAG_FT_WtoT", "FT_REDISTRIBUTE"})
        _Timer.add(v);
    }

    scr_coulomb_fourier_t(scr_coulomb_fourier_t const&) = default;
    scr_coulomb_fourier_t(scr_coulomb_fourier_t &&) = default;
    scr_coulomb_fourier_t& operator=(const scr_coulomb_fourier_t &) = default;
    scr_coulomb_fourier_t& operator=(scr_coulomb_fourier_t &&) = default;

    ~scr_coulomb_fourier_t() {}

    /**
     * Specialized FT function for a distributed array along the (τ, ω) axes.
     *
     * The staging buffers are allocated and released within the call. Each holds
     * a full aux grid — the whole (τ/ω, q, P, Q) array in the ft_buffer_dist
     * layout, so ~1/nproc of it per rank — because the local IAFT kernel consumes
     * the entire τ/ω axis at once. A buffer is skipped when the caller's
     * distribution already matches ft_buffer_dist.
     *
     * check_leakage runs the IAFT leakage diagnostic on the τ-side array. It is a
     * caller-supplied flag rather than something decided here, matching
     * scf_common's distributed_tau_to_w: it costs two gemms plus two collectives
     * per transform, so a hot path turns it off, and it is collective — every rank
     * must pass the same value. Callers gate it on __app_verbosity__, which is
     * rank-uniform.
     */
    template<nda::MemoryArrayOfRank<4> local_Array_t, typename communicator_t>
    auto tau_to_w(memory::darray_t<local_Array_t, communicator_t> &dPi_tqPQ_pos,
                  std::array<long, 4> w_pgrid_out,
                  std::array<long, 4> w_bsize_out = {},
                  bool reset_input = false,
                  bool check_leakage = true)
    -> memory::darray_t<local_Array_t, mpi3::communicator>;

    template<nda::MemoryArrayOfRank<4> local_Array_t, typename communicator_t>
    auto w_to_tau(memory::darray_t<local_Array_t, communicator_t> &dW_wqPQ_pos,
                  std::array<long, 4> t_pgrid_out,
                  std::array<long, 4> t_bsize_out = {},
                  bool reset_input = false,
                  bool check_leakage = true)
    -> memory::darray_t<local_Array_t, mpi3::communicator>;

    /**
     * Internal FT-buffer distribution used by tau_to_w / w_to_tau:
     * axis 0 (τ/ω) local, q and (P, Q) distributed. Exposed so LR callers can
     * request the FT output directly in this distribution (the FT then skips
     * the matching redistribute) and build W_full(iω) to match.
     * Only gshape[1..3] (q, P, Q) are read; τ/ω extent is irrelevant.
     */
    static std::pair<std::array<long, 4>, std::array<long, 4>>
    ft_buffer_dist(long np, std::array<long, 4> gshape) {
      std::array<long, 4> b_pgrid = {1, 1, 1, 1};
      std::array<long, 4> b_bsize = {1, 1, 1, 1};
      long nq = gshape[1];
      b_pgrid[1] = utils::find_proc_grid_max_npools(np, nq, 0.2);
      long np_PQ = np / b_pgrid[1];
      if (gshape[2] * gshape[3] >= np_PQ) {
        b_pgrid[2] = utils::find_proc_grid_min_diff(np_PQ, gshape[2], gshape[3]);
        b_pgrid[3] = np_PQ / b_pgrid[2];
      } else {
        utils::check(np_PQ == 1,
            "scr_coulomb_fourier_t::ft_buffer_dist: PQ too small for proc count (NP*NQ < np_PQ)");
      }
      b_bsize[2] = std::max(gshape[2] / std::max(b_pgrid[2], 1L), 1L);
      b_bsize[3] = std::max(gshape[3] / std::max(b_pgrid[3], 1L), 1L);
      return {b_pgrid, b_bsize};
    }

    utils::TimerManager& timer() { return _Timer; }

  private:
    const imag_axes_ft::IAFT* _ft = nullptr;
    utils::TimerManager _Timer;

  }; // scr_coulomb_fourier_t

} // solvers
} // methods

#endif //COQUI_SCR_COULOMB_FOURIER_T_H
