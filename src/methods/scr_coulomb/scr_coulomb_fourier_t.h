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
#include "utilities/tile_partition.hpp"

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
                  std::array<long, 4> w_tcount_out = {},
                  bool reset_input = false,
                  bool check_leakage = true)
    -> memory::darray_t<local_Array_t, mpi3::communicator>;

    template<nda::MemoryArrayOfRank<4> local_Array_t, typename communicator_t>
    auto w_to_tau(memory::darray_t<local_Array_t, communicator_t> &dW_wqPQ_pos,
                  std::array<long, 4> t_pgrid_out,
                  std::array<long, 4> t_tcount_out = {},
                  bool reset_input = false,
                  bool check_leakage = true)
    -> memory::darray_t<local_Array_t, mpi3::communicator>;

    /**
     * The q-dist distribution: axis 0 (τ/ω) local, q and (P, Q) distributed.
     * Used internally by tau_to_w / w_to_tau for their staging buffers, and
     * exposed so callers can request the FT output directly in it (the FT then
     * skips the matching redistribute) and build W_full(iω) to match.
     * Only gshape[1..3] (q, P, Q) are read; τ/ω extent is irrelevant.
     *
     * On the LR path this is *the* distribution every W(iω) is carried on, which
     * is an invariant rather than any one producer's preference: it is what makes
     * both transforms fuse — tau_to_w writes straight into W(iω) and w_to_tau
     * reads straight out of it, one global redistribute each instead of two — at
     * the price of the ω-side W Dyson running as a SLATE SUMMA on the (P, Q)
     * subgrid instead of a rank-local gemm. solve_lr_dyson_W asserts it, since
     * losing it costs ~24% of the LR-GW run and is otherwise silent.
     *
     * The returned (P, Q) tile count is square, and that is load-bearing rather
     * than cosmetic: the fused branches of tau_to_w / w_to_tau compare tile count
     * as well as processor grid, and the W Dyson runs SLATE on whatever this
     * returns (see below).
     */
    static std::pair<std::array<long, 4>, std::array<long, 4>>
    ft_buffer_dist(long np, std::array<long, 4> gshape) {
      std::array<long, 4> b_pgrid = {1, 1, 1, 1};
      std::array<long, 4> b_tcount = {0, 0, 0, 0};
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
      // Square tile count, the same formula as scr_coulomb_t::W_omega_proc_grid.
      // Square is required because the W Dyson runs slate_ops::multiply on this
      // distribution, and its C-order branch issues slate::multiply(a,Bs,As,b,Cs),
      // which needs Bs.nt() == As.mt() — with NP == NQ that is
      // tcount[2] == tcount[3]. The (τ/ω, q) axes keep one element per tile (0).
      b_tcount[2] = utils::balanced_tile_count(gshape[2],
                        std::max(b_pgrid[2], b_pgrid[3]), 1024);
      b_tcount[3] = b_tcount[2];
      return {b_pgrid, b_tcount};
    }

    utils::TimerManager& timer() { return _Timer; }

  private:
    const imag_axes_ft::IAFT* _ft = nullptr;
    utils::TimerManager _Timer;

  }; // scr_coulomb_fourier_t

} // solvers
} // methods

#endif //COQUI_SCR_COULOMB_FOURIER_T_H
