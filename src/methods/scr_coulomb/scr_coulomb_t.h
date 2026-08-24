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


#ifndef COQUI_SCR_COULOMB_T_H
#define COQUI_SCR_COULOMB_T_H

#include "configuration.hpp"
#include "nda/nda.hpp"
#include "nda/h5.hpp"
#include "numerics/distributed_array/nda.hpp"
#include "numerics/shared_array/nda.hpp"

#include "utilities/mpi_context.h"
#include "utilities/Timer.hpp"
#include "utilities/proc_grid_partition.hpp"
#include "utilities/tile_partition.hpp"
#include "IO/app_loggers.h"

#include "mean_field/MF.hpp"
#include "methods/embedding/projector_boson_t.h"
#include "methods/scr_coulomb/scr_coulomb_fourier_t.h"
#include "numerics/fft/fft_kR.hpp"
#include "numerics/imag_axes_ft/iaft_utils.hpp"
#include "methods/mb_state/mb_state.hpp"
#include "methods/ERI/detail/concepts.hpp"

namespace methods {
namespace solvers {
  // TODO
  //    1. timer
  /**
   * @brief scr_coulomb_t class
   *
   * This class handles the computation of screened Coulomb interactions for RPA and beyond theories.
   *
   * The RPA-beyond corrections are controlled by the `screen_type` based on
   * whether certain keywords appear in the `screen_type` string.
   *
   * Meaningful keywords in `screen_type` include:
   *   - "crpa"  - Subtracting q-dependent RPA polarization function within an active subspace
   *   - "edmft" - Adding EDMFT correction to the RPA polarization function
   *
   * For examples:
   *   1) screen_type = "ABC" will return RPA polarization function
   *   2) screen_type = "crpa" will return cRPA polarization function
   *   3) screen_type = "edmft" or "gw_edmft" or "edmft_ABC" will all return EDMFT-corrected RPA polarization function
   *   4) screen_type = "crpa_edmft" will return cRPA polarization function with EDMFT correction
   *
   * Usage:
   *   scr_coulomb_t scr_eri(..., screen_type, ...);
   *
   *   // compute the screened interaction
   *   scr_eri.compute(...);
   *
   *   // compute polarzability
   *   auto Pi = scr_eri.eval_Pi_qdep(...);
   */
  class scr_coulomb_t {
  public:
    using mpi_context_t = utils::mpi_context_t<mpi3::communicator>;
    template<nda::MemoryArray Array_base_t>
    using sArray_t = math::shm::shared_array<Array_base_t>;
    template<nda::MemoryArray Array_base_t>
    using dsArray_t = math::shm::distributed_shared_array<Array_base_t>;
    template<int N>
    using shape_t = std::array<long,N>;

  public:
    scr_coulomb_t(
        const imag_axes_ft::IAFT *ft,
        std::string screen_type,
      std::string div = "gygi");

    scr_coulomb_t(scr_coulomb_t const&) = default;
    scr_coulomb_t(scr_coulomb_t &&) = default;
    scr_coulomb_t& operator=(const scr_coulomb_t &) = default;
    scr_coulomb_t& operator=(scr_coulomb_t &&) = default;

    ~scr_coulomb_t() {}

    /**
     * Compute the dynamic screened interaction and store it in the MBState.
     * @param mb_state  - [INPUT/OUTPUT] MBState object to store the results
     * @param thc       - [INPUT] THC bare Coulomb interaction
     */
    void update_w(MBState &mb_state, THC_ERI auto &thc, long h5_iter=-1);
    void update_w(MBState &mb_state, Cholesky_ERI auto &chol, long h5_iter=-1);

    /**
     * Evaluate the screened interaction from a imaginary-time polarizability
     * @tparam w_out - true: output W(iw); false: output W(t)
     * @param dPi_tqPQ_pos - [INPUT] Polarization function with particle-hole symmetry
     * @param thc          - [INPUT] THC ERI
     * @param reset_input  - [INPUT] free the memory of dPi_tqPQ_pos
     * @param w_pgrid      - [OPTIONAL] processor grid for W(iw)
     * @param w_tcount     - [OPTIONAL] tile count for W(iw)
     * @return - Screened interaction W in the THC product basis
     */
    template<bool w_out, nda::MemoryArrayOfRank<4> local_Array_t, typename communicator_t>
    auto dyson_W_from_Pi_tau(memory::darray_t<local_Array_t, communicator_t> &dPi_tqPQ_pos,
                             THC_ERI auto &thc, bool reset_input,
                             std::array<long, 4> w_pgrid = {0, 0, 0, 0},
                             std::array<long, 4> w_tcount = {0, 0, 0, 0})
    -> memory::darray_t<local_Array_t, mpi3::communicator>;

    static auto W_omega_proc_grid(long nproc, long nqpts_ibz, long nw_b, long Np)
    -> std::tuple<std::array<long, 4>, std::array<long, 4>> {
      // setup pgrid for Pi_wPQ: maximize nqpools: a) nqpts = i * b; b) np = i * bb
      // not sure if this is the optimal configuration
      long np = nproc;
      long nw_half = (nw_b%2==0)? nw_b/2 : nw_b/2 + 1;

      long nqpools = utils::find_proc_grid_max_npools(np, nqpts_ibz, 0.2);
      np /= nqpools;
      long nwpools = utils::find_proc_grid_max_npools(np, nw_half, 0.2);
      np /= nwpools;
      long np_P = utils::find_proc_grid_min_diff(np, 1, 1);
      long np_Q = np / np_P;
      utils::check(nqpools > 0 and nqpools <= nqpts_ibz,
                   "scr_coulomb_t:: nqpools <= 0 or nqpools > nqpts_ibz. nqpools = {}", nqpools);
      utils::check(nqpools > 0 and nwpools <= nw_half,
                   "scr_coulomb_t:: nwpools <= 0 or nwpools > nw_half. nwpools = {}", nwpools);
      utils::check(nproc % nqpools == 0, "gw_t:: gcomm.size() % nqpools != 0");
      utils::check(nproc % (nqpools * nwpools) == 0, "gw_t:: gcomm.size() & (nqpools*nwpools) != 0");


      std::array<long, 4> w_pgrid = {nwpools, nqpools, np_P, np_Q}; // (w, q, P, Q)

      // Square (P,Q) tile count: the same count on both axes, so the two axes get
      // identical tile boundaries and the SUMMA / getri preconditions hold. The
      // 1024 cap keeps a tile from growing without bound; below it there is
      // exactly one tile per rank on the larger of the two PQ axes.
      std::array<long, 4> w_tcount;
      w_tcount.fill(0);   // 0 = one element per tile, i.e. the (w,q) axes stay untiled
      w_tcount[2] = utils::balanced_tile_count(Np, std::max(w_pgrid[2], w_pgrid[3]), 1024);
      w_tcount[3] = w_tcount[2];

      return std::make_tuple(w_pgrid, w_tcount);
    }

    /**
     * Processor grid and block size for the τ-side auxiliary arrays: Π(t,q,P,Q) as
     * allocated in eval_Pi_rpa_Rspace, and by inheritance W(τ) (same grid) and the
     * (q,t) transpose W(q,t,P,Q) (axes 0↔1 swapped). The whole τ-side family, plus
     * the t-pool sub-communicator size np_P*np_Q, comes from here, so the SCF
     * memory/distribution report predicts exactly what the allocator builds.
     * @return {pgrid, bsize} over the (t, q, P, Q) axes.
     */
    static auto Pi_tau_proc_grid(long nproc, long nt_half, long nqpts_ibz, long ns, long nkpts)
    -> std::tuple<std::array<long, 4>, std::array<long, 4>> {
      // CNY: Memory for intermediate objects scales with ntpools.
      // Therefore, we restrict ntpools*ns*nkpts*Np*Np*2 < nt_half*nqpts_ibz*Np*Np
      long ntpools_max = std::max(long(nt_half * nqpts_ibz / (ns * nkpts * 2)), 1L);
      while (nproc % ntpools_max != 0) ntpools_max -= 1;
      long ntpools = utils::find_proc_grid_max_npools(nproc, nt_half, 0.2);
      if (ntpools > ntpools_max) ntpools = ntpools_max;
      long np_PQ = nproc / ntpools;
      long np_P = utils::find_proc_grid_min_diff(np_PQ, 1, 1);
      long np_Q = np_PQ / np_P;

      return std::make_tuple(std::array<long, 4>{ntpools, 1, np_P, np_Q},
                             std::array<long, 4>{0, 0, 0, 0});
    }

    /**
     * Evaluate the screened interaction W in place from a Matsubara polarizability
     * @param dPi_wqPQ - [INPUT/OUTPUT] polarizability / screened interaction: (nw, nqpts_ibz, Np, Np)
     * @param thc      - [INTPUT] THC ERI object
     */
    template<nda::MemoryArray Array_4D_t, typename communicator_t>
    void dyson_W_in_place(memory::darray_t<Array_4D_t, communicator_t> &dPi_wqPQ,
                          THC_ERI auto &thc);

    /**
     * Evaluate q-dependent polarization function
     *
     * @param G_tskij         - [INPUT] Green's function in primary basis: (nts, ns, nkpts_ibz, nbnd, nbnd)
     * @param thc             - [INPUT] THC-ERI instance
     * @param screen_type     - [INPUT] method for polarizability
     * @param proj            - [INPUT] bosonic projector (optional)
     * @param coqui_h5_prefix - [INPUT] prefix for CoQui h5 where pi_imp/pi_dc are stored (optional)
     * @param pi_imp          - [INPUT] impurity polarizability (optional)
     * @param pi_dc           - [INPUT] double-counting polarizability (optional)
     * @return - Polarization function in k space: (nts, nqpts_ibz, Np, Np)
     */
    auto eval_Pi_qdep(const nda::MemoryArrayOfRank<5> auto &G_tskij, THC_ERI auto &thc,
                      const projector_boson_t* proj=nullptr,
                      const nda::array_view<ComplexType, 5> *pi_imp=nullptr,
                      const nda::array_view<ComplexType, 5> *pi_dc=nullptr)
    -> memory::darray_t<memory::array<HOST_MEMORY, ComplexType, 4>, mpi3::communicator>;

    auto eval_Pi_qdep(MBState &mb_state, THC_ERI auto &thc)
    -> memory::darray_t<memory::array<HOST_MEMORY, ComplexType, 4>, mpi3::communicator>;

    // TODO function to return Pi(w) directly

    /**
     * compute the Green's function on the THC interpolating points on a specific time point
     * @param it       - [INPUT] time index
     * @param G_tskij - [INPUT] Green's function in a primary basis on shared memory
     * @param dG_skPQ  - [OUTPUT] Green's function on THC interpolating points at it in a distributed array
     * @param thc      - [INPUT] THC-ERI instance
     * @param minus_t  - [INPUT] false: compute G at tau=(0,beta/2); true: compute G at tau=(0,-beta/2)
     */

    /**
     * Specialized version of FT function for distributed array along (tau, w)-axes
     */
    template<nda::MemoryArrayOfRank<4> local_Array_t, typename communicator_t>
    auto tau_to_w(memory::darray_t<local_Array_t, communicator_t> &dPi_tqPQ_pos,
                  std::array<long, 4> w_pgrid_out,
                  std::array<long, 4> w_bsize_out = {},
                  bool reset_input = false)
    -> memory::darray_t<local_Array_t, mpi3::communicator>;

    template<nda::MemoryArrayOfRank<4> local_Array_t, typename communicator_t>
    auto w_to_tau(memory::darray_t<local_Array_t, communicator_t> &dW_wqPQ_pos,
                  std::array<long, 4> t_pgrid_out,
                  std::array<long, 4> t_bsize_out = {},
                  bool reset_input = false)
    -> memory::darray_t<local_Array_t, mpi3::communicator>;

    template<typename comm_t>
    void dump_eps_inv_head(const nda::ArrayOfRank<2> auto &eps_inv_head_tq,
                           const nda::ArrayOfRank<1> auto &eps_inv_head_t,
                           std::string coqui_h5_prefix, long iter,
                           comm_t &comm, mf::MF &mf);

  private:
    /**
     * Evaluate polarization function by computing the convolution on the R space
     * @param G_tskij - [INPUT] Green's function in a primary basis on shared memory:
     *                   (nts, ns , nkpts_ibz, nbnd, nbnd)
     * @param thc      - [INPUT] THC-ERI instance
     * @return - Polarization function in k space: (nts, nqpts_ibz, Np, Np)
     */
    auto eval_Pi_rpa_Rspace(const nda::MemoryArrayOfRank<5> auto &G_tskij, THC_ERI auto &thc)
    -> memory::darray_t<memory::array<HOST_MEMORY, ComplexType, 4>, mpi3::communicator>;

    /**
     * Evaluate polarization function by computing the convolution on the k space
     * @param G_tskij - [INPUT] Green's function in a primary basis on shared memory:
     *                   (nts, ns , nkpts_ibz, nbnd, nbnd)
     * @param thc      - [INPUT] THC-ERI instance
     * @return - Polarization function in k space: (nts, nqpts_ibz, Np, Np)
     */
    auto eval_Pi_rpa_kspace(const nda::MemoryArrayOfRank<5> auto &G_tskij, THC_ERI auto &thc)
    -> memory::darray_t<memory::array<HOST_MEMORY, ComplexType, 4>, mpi3::communicator>;

    // details of eval_Pi_all_kspace
    template<nda::MemoryArray Array_5D_t, nda::MemoryArray Array_4D_t, typename communicator_t>
    void eval_Pi_rpa_k_impl(const memory::darray_t<Array_5D_t, communicator_t> &dGp_sktPQ_c,
                            const memory::darray_t<Array_5D_t, communicator_t> &dGn_sktPQ,
                            memory::darray_t<Array_4D_t, communicator_t> &dPi_qtPQ,
                            THC_ERI auto &thc);

    auto eval_Pi_rpa_active(const nda::ArrayOfRank<5> auto &G_tskij, THC_ERI auto &thc,
                            const projector_t &proj, int scheme)
    -> memory::darray_t<memory::array<HOST_MEMORY, ComplexType, 4>, mpi3::communicator>;

    auto upfold_pi_local(
        const nda::MemoryArrayOfRank<5> auto &Pi_loc_tabcd,
        THC_ERI auto &thc,
        const projector_boson_t &proj,
        std::array<long, 4> pgrid, std::array<long, 4> bsize)
    -> memory::darray_t<memory::array<HOST_MEMORY, ComplexType, 4>, mpi3::communicator>;

  private:
    const imag_axes_ft::IAFT* _ft = nullptr;

    std::string _screen_type = "";

    std::string _div_treatment;
    utils::TimerManager _Timer;

    // Distributed bosonic τ↔ω FT engine. The public tau_to_w / w_to_tau
    // delegate to this; it owns the FT timers and the leak-check gate.
    scr_coulomb_fourier_t _scr_fourier;

    // Blocked-FFT engines for the spatial k<->R transform in eval_Pi_rpa_Rspace.
    // Built lazily; _fft_k transforms the kpts-indexed G, _fft_q the Qpts-indexed Pi.
    // Used when COQUI_DEBUG_GEMM_FT is unset and the q-mesh is symmetry-unreduced.
    std::optional<math::fft::fft_kR_t> _fft_k, _fft_q;
  public:
    utils::TimerManager& timer() { return _Timer; }

    /**
     * Print the RPA polarizability (Π) and screened-interaction (W) timers.
     * These are recorded by eval_Pi_qdep / dyson_W_in_place and the bosonic
     * FT engine but were previously never surfaced, leaving the dominant scGW
     * cost invisible. Called at the end of update_w.
     */
    void print_timers(int level = 2);

    std::string div_treatment() const { return _div_treatment; }
    std::string& screen_type() { return _screen_type; };
    std::string screen_type() const { return _screen_type; };

  }; // scr_coulomb_t
} // solvers
}// methods

#endif //COQUI_SCR_COULOMB_T_H
