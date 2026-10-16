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


#ifndef COQUI_LR_THC_COMM_HPP
#define COQUI_LR_THC_COMM_HPP

#include <limits>

#include "mpi3/communicator.hpp"
#include "nda/nda.hpp"
#include "utilities/proc_grid_partition.hpp"
#include "utilities/Timer.hpp"
#include "numerics/nda_functions.hpp"
#include "numerics/distributed_array/nda.hpp"
#include "numerics/shared_array/nda.hpp"
#include "numerics/shared_array/detail/concepts.hpp"

#include "methods/ERI/detail/concepts.hpp"
#include "methods/SCF/lr_ibc.hpp"

namespace methods {
  namespace solvers {
    /**
     * LR-specific basis transformations between primary and THC auxiliary bases.
     *
     * Mirrors thc_solver_comm but uses different k-points for left/right X matrices:
     *   - Left X: X(k+q) via kpq_map
     *   - Right X: X(k)
     *
     * Cannot reuse thc_solver_comm because its _primary_to_aux_impl and
     * _aux_to_primary_impl assume left and right X matrices share the same k-point,
     * which is incompatible with q≠0 linear response.
     *
     * Time-reversal handling:
     * Unlike rpa_pi.icc, which uses a two-step approach (compute on nkpts_no_trev
     * k-points, then unfold via G_PQ(k) = conj(G_PQ(kp_trev_pair(k)))), this class
     * loops over all nkpts k-points and handles trev inline via transpose on the
     * primary-basis input: O_ab(k) = O_ab(ibz_k)^T for Hermitian quantities.
     *
     * The rpa_pi.icc conjugation shortcut (kp_trev_pair) relies on X(k) = conj(X(k'))
     * for time-reversal pairs, making O_PQ(k) = conj(O_PQ(k')). This breaks for q≠0
     * because the left X is at k+q, and k+q = -k'+q ≠ -(k'+q), so X(k+q) ≠ conj(X(k'+q))
     * in general. Each trev k-point must be computed independently with its own X(k+q).
     */
    struct lr_thc_comm {
      template<nda::MemoryArray local_Array_t>
      using dArray_t = math::nda::distributed_array<local_Array_t, mpi3::communicator>;
      template<nda::MemoryArray Array_base_t>
      using sArray_t = math::shm::shared_array<Array_base_t>;
      template<int N>
      using shape_t = std::array<long, N>;

      /**
       * LR primary→aux wrapper (rank-4 distributed array).
       * Matches thc_solver_comm::primary_to_aux, its rank-4 distributed-array overload.
       *
       * @param ip         - [INPUT] left polarization index
       * @param iq         - [INPUT] right polarization index
       * @param O_Iab      - [INPUT] tensor in primary basis (ns, nkpts_ibz, nbnd, nbnd)
       * @param O_IPQ      - [OUTPUT] tensor in auxiliary basis (ns, nkpts, NP, NP)
       * @param thc        - [INPUT] THC-ERI handler
       * @param kp_to_ibz  - [INPUT] full BZ k → IBZ k mapping (nkpts,)
       * @param kp_trev    - [INPUT] time-reversal flag per full BZ k-point (nkpts,)
       * @param kpq_map    - [INPUT] full BZ k → full BZ k+q mapping (nkpts,)
       * @param ibc        - [INPUT] optional IBC correction data (DeltaX arrays + unperturbed quantities)
       * @param O_unpert   - [INPUT] optional unperturbed quantity in primary basis (ns, nk_ibz, nb, nb)
       *
       * When ibc and O_unpert are both non-null, adds correction terms (either one null
       * skips them):
       *   O_PQ += δ^q X(k) · A(k) · X(k)† + X(k+q) · A(k+q) · [δ^{-q} X(k+q)]†
       */
      template<nda::MemoryArray Array_primary_t, nda::MemoryArray Array_aux_t, typename communicator_t,
               typename O_unpert_4D_t = nda::array<ComplexType, 4>>
      static void primary_to_aux(int ip, int iq,
                                 const Array_primary_t &O_Iab,
                                 memory::darray_t<Array_aux_t, communicator_t> &O_IPQ,
                                 THC_ERI auto& thc,
                                 nda::ArrayOfRank<1> auto const& kp_to_ibz,
                                 nda::ArrayOfRank<1> auto const& kp_trev,
                                 nda::ArrayOfRank<1> auto const& kpq_map,
                                 const lr_ibc_DeltaX* ibc = nullptr,
                                 const O_unpert_4D_t* O_unpert = nullptr) {
        _primary_to_aux_impl<false>(ip, iq, O_Iab, O_IPQ, thc, kp_to_ibz, kp_trev, kpq_map,
                                    ibc, O_unpert);
      }

      /**
       * LR aux→primary (rank-4 distributed → shared window):
       *   O_Iab[offset + ·] += scl · X(k+q)† · O_PQ(k) · X(k)
       * for every (s,k), into the ns·nkpts·nbnd² contiguous window elements starting at
       * flat `offset` (0 for a (s,k,a,b) window, a τ slab of a (t,s,k,a,b) one).
       *
       * Each rank contracts its (P,Q) tile into a full (s,k,a,b) partial; the partials
       * are summed over `node_comm` with shm::reduce_scatter_add, each of its ranks
       * adding its own disjoint element stripe into the window. Does NOT reduce over
       * nodes and does not synchronize the window: the stripes are disjoint, so a group
       * of these calls races with nothing.
       *
       * `node_comm` is the node split of O_IPQ's communicator: O_Iab.node_comm() when
       * O_IPQ lives on the window's communicator (lr_hf), the τ pool's node split when it
       * lives on a pool (lr_gw, one τ slab per pool).
       *
       * Caller's side of the contract:
       *   - zero O_Iab before the first call of a group (this accumulates with +=,
       *     and a stale window would be folded into the reduction);
       *   - every rank of O_IPQ's communicator calls, with the same arguments;
       *   - all_reduce_parallel() before anything reads O_Iab across ranks: it
       *     publishes the stripes and sums the per-node partials.
       * set_zero() and all_reduce_parallel() each open and close with node_sync(),
       * so those two calls alone publish the writes made in between.
       *
       * @param ip        - [INPUT] left polarization index
       * @param iq        - [INPUT] right polarization index
       * @param scl       - [INPUT] scaling factor for accumulation
       * @param O_IPQ     - [INPUT] tensor in auxiliary basis (ns, nkpts_ibz, NP, NP),
       *                    (s,k) undivided
       * @param O_Iab     - [OUTPUT] window whose last two axes are (nbnd, nbnd), accumulated
       * @param offset    - [INPUT] flat window offset of the (s,k,a,b) block
       * @param node_comm - [INPUT] node split of O_IPQ's communicator
       * @param thc       - [INPUT] THC-ERI handler
       * @param kp_map    - [INPUT] IBZ k → full BZ k mapping, i.e. ks_to_k(0) (nkpts_ibz,)
       * @param kpq_map   - [INPUT] full BZ k → full BZ k+q mapping (nkpts,)
       * @param Timer     - [INPUT] optional sub-clock manager: SIGMA_A2P_ALLOC for the
       *                    scratch, SIGMA_A2P_GEMM for the local contraction and
       *                    SIGMA_A2P_REDUCE for the reduce-scatter and stripe add.
       * @param O_buf     - [INPUT/OUTPUT] optional caller-owned (ns·nkpts, nbnd, nbnd)
       *                    scratch, resized on shape mismatch; allocated per call if null.
       */
      template<nda::MemoryArray AF_t, nda::MemoryArray Array_aux_t, typename communicator_t>
      static void aux_to_primary(int ip, int iq,
                                 ComplexType scl,
                                 const memory::darray_t<Array_aux_t, communicator_t>& O_IPQ,
                                 sArray_t<AF_t>& O_Iab,
                                 long offset,
                                 mpi3::shared_communicator& node_comm,
                                 THC_ERI auto& thc,
                                 nda::ArrayOfRank<1> auto const& kp_map,
                                 nda::ArrayOfRank<1> auto const& kpq_map,
                                 utils::TimerManager* Timer = nullptr,
                                 nda::array<ComplexType, 3>* O_buf = nullptr) {
        static_assert(nda::get_rank<Array_aux_t> == 4,
                      "lr_thc_comm::aux_to_primary: aux rank must be 4");
        constexpr int R = nda::get_rank<AF_t>;
        static_assert(R >= 4, "lr_thc_comm::aux_to_primary: primary rank must be >= 4");

        auto pgrid = O_IPQ.grid();
        long ns = O_IPQ.global_shape()[0];
        long nkpts = O_IPQ.global_shape()[1];
        auto const& org = O_IPQ.origin();
        auto wshape = O_Iab.shape();
        long nbnd = wshape[R - 1];
        // (s,k) undivided: every rank holds a partial of the whole (s,k,a,b) block.
        utils::check(pgrid[0] == 1 and pgrid[1] == 1,
                     "lr_thc_comm::aux_to_primary: expected (s,k) undivided, "
                     "got pgrid ({},{})", pgrid[0], pgrid[1]);
        utils::check(ns == wshape[R - 4] and nkpts == wshape[R - 3],
                     "lr_thc_comm::aux_to_primary: (s,k) extents ({},{}) differ from the "
                     "window's ({},{})", ns, nkpts, wshape[R - 4], wshape[R - 3]);
        // node_comm must partition O_IPQ's communicator, or some node's reduce-scatter
        // misses ranks (hang) or partials: its roots' sizes add up to that size.
        auto& gcomm = *O_IPQ.communicator();
        long covered = gcomm.all_reduce_value(long(node_comm.root() ? node_comm.size() : 0),
                                              std::plus<>{});
        utils::check(covered == long(gcomm.size()),
                     "lr_thc_comm::aux_to_primary: node_comm does not partition the aux "
                     "array's communicator ({} of {} ranks)", covered, gcomm.size());

        nda::array<ComplexType, 3> O_buf_own;
        if (Timer) Timer->start("SIGMA_A2P_ALLOC");
        if (O_buf == nullptr) O_buf = &O_buf_own;
        if (O_buf->shape() != shape_t<3>{ns * nkpts, nbnd, nbnd})
          O_buf->resize(ns * nkpts, nbnd, nbnd);
        if (Timer) Timer->stop("SIGMA_A2P_ALLOC");

        _aux_to_primary_gemm(ip, iq, O_IPQ.local(), nbnd, thc, kp_map, kpq_map,
                             org[1], org[2], org[3], Timer, *O_buf);

        if (Timer) Timer->start("SIGMA_A2P_REDUCE");
        math::shm::reduce_scatter_add(O_Iab, node_comm, O_buf->data(), offset,
                                      long(O_buf->size()), scl);
        if (Timer) Timer->stop("SIGMA_A2P_REDUCE");
      }

      /**
       * LR primary→aux contraction for only the diagonal aux elements:
       *   n_P += Σ_(s,k) diag_P[ X(k+q) · O_ab(k) · X(k)† ]
       * over the flattened (s,k) indices in sk_rng, with k running over the full BZ.
       *
       * Deliberately not a mode of _primary_to_aux_impl: that routine is built around
       * a distributed (P,Q) tile and the reduction over it, which is exactly what an
       * operator diagonal in (P,Q) does not need. Here the outer product X·O·X† is
       * never formed — only its diagonal, through a row dot — so the NP×NQ gemm, the
       * (P,Q) distribution and its reduction all disappear. Reusing the dense impl
       * would keep every one of them.
       *
       * No MPI inside: the caller owns the (s,k) split and the reduction of n_P.
       *
       * @param ip        - [INPUT] left polarization index
       * @param iq        - [INPUT] right polarization index
       * @param O_skab    - [INPUT] tensor in primary basis (ns, nkpts_ibz, nbnd, nbnd)
       * @param sk_rng    - [INPUT] flattened (s,k) indices, i = s*nkpts + k
       * @param P_rng     - [INPUT] aux-basis rows to accumulate
       * @param n_P       - [INPUT/OUTPUT] length-|P_rng| accumulator
       * @param thc       - [INPUT] THC-ERI handler
       * @param kp_to_ibz - [INPUT] full BZ k → IBZ k mapping (nkpts,)
       * @param kp_trev   - [INPUT] time-reversal flag per full BZ k-point (nkpts,)
       * @param kpq_map   - [INPUT] full BZ k → full BZ k+q mapping (nkpts,)
       */
      template<nda::MemoryArray Array_primary_t>
      static void primary_to_aux_diagonal(int ip, int iq,
                                          const Array_primary_t& O_skab,
                                          nda::range sk_rng,
                                          nda::range P_rng,
                                          nda::array<ComplexType, 1>& n_P,
                                          THC_ERI auto& thc,
                                          nda::ArrayOfRank<1> auto const& kp_to_ibz,
                                          nda::ArrayOfRank<1> auto const& kp_trev,
                                          nda::ArrayOfRank<1> auto const& kpq_map) {
        static_assert(nda::get_rank<Array_primary_t> == 4,
                      "lr_thc_comm::primary_to_aux_diagonal: rank must be 4");
        decltype(nda::range::all) all;

        long nkpts = kpq_map.extent(0);
        long nbnd  = O_skab.extent(2);
        long NP_loc = P_rng.size();
        utils::check(n_P.extent(0) == NP_loc,
                     "lr_thc_comm::primary_to_aux_diagonal: n_P size {} != |P_rng| {}",
                     n_P.extent(0), NP_loc);
        utils::check(P_rng.last() <= thc.Np(),
                     "lr_thc_comm::primary_to_aux_diagonal: P_rng.last() > thc.Np()");

        nda::array<ComplexType, 2> Ask_Pb(NP_loc, nbnd);
        for (auto i : sk_rng) {
          long s = i / nkpts;
          long k = i % nkpts;

          // Left X at k+q, right X at k — the LR convention of _primary_to_aux_impl.
          auto Xsk_Pa_l = thc.X(s, ip, kpq_map(k));
          auto Xsk_Pa_r = thc.X(s, iq, k);

          if (kp_trev(k)) {
            nda::blas::gemm(Xsk_Pa_l(P_rng, all), nda::transpose(O_skab(s, kp_to_ibz(k), all, all)), Ask_Pb);
          } else {
            nda::blas::gemm(Xsk_Pa_l(P_rng, all), O_skab(s, kp_to_ibz(k), all, all), Ask_Pb);
          }

          // diag_P[A · X(k)†] = Σ_b A_Pb · conj(X(k)_Pb)
          for (long iP = 0; iP < NP_loc; ++iP) {
            auto A_b = Ask_Pb(iP, all);
            auto X_b = Xsk_Pa_r(iP + P_rng.first(), all);
            ComplexType acc(0.0, 0.0);
            for (long b = 0; b < nbnd; ++b) acc += A_b(b) * std::conj(X_b(b));
            n_P(iP) += acc;
          }
        }
      }

      /**
       * LR aux→primary contraction for an aux operator with only diagonal elements,
       * independent of (s,k):
       *   O_ab(k) += scl · Σ_P conj(X(k+q)_Pa) · J_P · X(k)_Pb
       * for the flattened (s,k_ibz) indices in sk_rng.
       *
       * Deliberately not a mode of aux_to_primary, for the mirror-image reason
       * of primary_to_aux_diagonal: that routine contracts a distributed (P,Q) tile
       * and reduces over the tile grid, whereas a diagonal operator is replicated, so
       * each (s,k) block can be produced whole by a single rank with no reduction at
       * all. J_P carries the full P range on every rank.
       *
       * No MPI inside. The destination is accumulated in place, so the caller owns
       * both the (s,k) split (each block must be written by exactly one rank) and any
       * synchronisation of the shared window it lives in.
       *
       * @param ip      - [INPUT] left polarization index
       * @param iq      - [INPUT] right polarization index
       * @param scl     - [INPUT] scaling factor for accumulation
       * @param J_P     - [INPUT] the (P,P) diagonal, length thc.Np(), replicated
       * @param O_skab  - [OUTPUT] tensor in primary basis (ns, nkpts_ibz, nbnd, nbnd)
       * @param sk_rng  - [INPUT] flattened (s,k_ibz) indices, i = s*nkpts_ibz + k_ibz
       * @param thc     - [INPUT] THC-ERI handler
       * @param kp_map  - [INPUT] IBZ k → full BZ k mapping, i.e. ks_to_k(0) (nkpts_ibz,)
       * @param kpq_map - [INPUT] full BZ k → full BZ k+q mapping (nkpts,)
       */
      template<nda::MemoryArray Array_primary_t>
      static void aux_to_primary_diagonal(int ip, int iq,
                                      ComplexType scl,
                                      nda::ArrayOfRank<1> auto const& J_P,
                                      Array_primary_t& O_skab,
                                      nda::range sk_rng,
                                      THC_ERI auto& thc,
                                      nda::ArrayOfRank<1> auto const& kp_map,
                                      nda::ArrayOfRank<1> auto const& kpq_map) {
        static_assert(nda::get_rank<Array_primary_t> == 4,
                      "lr_thc_comm::aux_to_primary_diagonal: rank must be 4");
        decltype(nda::range::all) all;

        long nkpts_ibz = O_skab.extent(1);
        long nbnd      = O_skab.extent(2);
        long NP        = J_P.extent(0);
        utils::check(NP == thc.Np(),
                     "lr_thc_comm::aux_to_primary_diagonal: J_P size {} != thc.Np() {}", NP, thc.Np());
        nda::range P_rng(0, NP);

        nda::array<ComplexType, 2> Bsk_Pb(NP, nbnd);
        for (auto i : sk_rng) {
          long s  = i / nkpts_ibz;
          long ik = i % nkpts_ibz;
          long k  = kp_map(ik);

          auto Xsk_Pa_l = thc.X(s, ip, kpq_map(k));
          auto Xsk_Pa_r = thc.X(s, iq, k);

          for (long P = 0; P < NP; ++P) Bsk_Pb(P, all) = J_P(P) * Xsk_Pa_r(P, all);
          nda::blas::gemm(scl, nda::dagger(Xsk_Pa_l(P_rng, all)), Bsk_Pb,
                          ComplexType(1.0), O_skab(s, ik, all, all));
        }
      }

      /**
       * LR primary→aux transposed: computes [O_PQ]^T directly.
       *
       * Uses the identity: [O_PQ]^T = conj( X_R @ O^H @ X_L^H )
       * where X_L = X(k+q) and X_R = X(k) in the standard transform.
       * Caller passes the original O (not O^H). Returns [O_PQ]^T directly.
       */
      template<nda::MemoryArray Array_primary_t, nda::MemoryArray Array_aux_t, typename communicator_t,
               typename O_unpert_4D_t = nda::array<ComplexType, 4>>
      static void primary_to_aux_transposed(int ip, int iq,
                                 const Array_primary_t &O_Iab,
                                 memory::darray_t<Array_aux_t, communicator_t> &O_IPQ,
                                 THC_ERI auto& thc,
                                 nda::ArrayOfRank<1> auto const& kp_to_ibz,
                                 nda::ArrayOfRank<1> auto const& kp_trev,
                                 nda::ArrayOfRank<1> auto const& kpq_map,
                                 const lr_ibc_DeltaX* ibc = nullptr,
                                 const O_unpert_4D_t* O_unpert = nullptr) {
        _primary_to_aux_impl<true>(ip, iq, O_Iab, O_IPQ, thc, kp_to_ibz, kp_trev, kpq_map,
                                   ibc, O_unpert);
      }

    private:
      /**
       * LR primary→aux impl. Matches thc_solver_comm::_primary_to_aux_impl
       * except the X lookup:
       *   - Left X uses X(kpq_map(k)) instead of X(k)
       *   - Right X uses X(k) (unchanged)
       * Serial over the caller's local PQ tile (thc_solver_comm's P-batching
       * is dropped: every LR caller runs one rank per tile).
       *
       * `transposed` computes [O_PQ]^T = conj( X(k) @ O^H @ X(k+q)^H ) instead:
       * swapped X, O^H formed per (s,k) slice, and the output conjugated.
       *
       * The DeltaX correction (ibc and O_unpert both non-null) adds
       *   δ^q X(k) · A(k) · X(k)†  +  X(k+q) · A(k+q) · [δ^{-q} X(k+q)]†,
       * or, transposed and before the conjugation,
       *   X(k) · A(k)^H · [δ^q X(k)]†  +  δ^{-q} X(k+q) · A(k+q)^H · X(k+q)†.
       */
      template<bool transposed, typename Array_primary_t, typename Array_aux_t,
               typename communicator_t, typename O_unpert_4D_t>
      static void _primary_to_aux_impl(int ip, int iq,
                                       const Array_primary_t& O_skab,
                                       memory::darray_t<Array_aux_t, communicator_t>& O_IPQ,
                                       THC_ERI auto& thc,
                                       nda::ArrayOfRank<1> auto const& kp_map,
                                       nda::ArrayOfRank<1> auto const& kp_trev,
                                       nda::ArrayOfRank<1> auto const& kpq_map,
                                       const lr_ibc_DeltaX* ibc,
                                       const O_unpert_4D_t* O_unpert) {
        static_assert(nda::get_rank<Array_primary_t> == 4,
                      "lr_thc_comm::primary_to_aux: rank must be 4");
        constexpr const char* fn = transposed ? "primary_to_aux_transposed" : "primary_to_aux";
        utils::check(O_IPQ.global_shape()[1] == O_IPQ.local_shape()[1],
                     "lr_thc_comm::{}: Does not support mpi distributed along k-axis.", fn);

        decltype(nda::range::all) all;
        auto O_skPQ = O_IPQ.local();
        long P_offset  = O_IPQ.origin()[2];
        long Q_offset  = O_IPQ.origin()[3];
        long ns        = O_skab.shape(0);
        long nkpts_ibz = O_skab.shape(1);
        long nbnd      = O_skab.shape(2);
        long nkpts     = O_skPQ.shape(1);
        long NP_loc    = O_skPQ.shape(2);
        long NQ_loc    = O_skPQ.shape(3);
        utils::check(O_skPQ.shape(0) == ns, "lr_thc_comm::{}: spin extent mismatched", fn);
        utils::check(NP_loc+P_offset <= thc.Np(), "lr_thc_comm::{}: NP_loc+P_offset > thc.Np()", fn);
        utils::check(NQ_loc+Q_offset <= thc.Np(), "lr_thc_comm::{}: NQ_loc+Q_offset > thc.Np()", fn);
        bool has_deltax = ibc && O_unpert;
        if (has_deltax) {
          utils::check(nkpts_ibz == nkpts,
                       "lr_thc_comm::{}: DeltaX correction requires full BZ grid (nkpts_ibz={} != nkpts={})",
                       fn, nkpts_ibz, nkpts);
        }

        nda::array<ComplexType, 2> Ask_Pb(NP_loc, nbnd);
        // O^H slice (transposed), then the unperturbed slice of the DeltaX terms
        nda::matrix<ComplexType> O_buf(nbnd, nbnd);

        nda::range X_P_rng(P_offset, P_offset + NP_loc);
        nda::range O_Q_rng(Q_offset, Q_offset + NQ_loc);
        for (long s = 0; s < ns; ++s) {
          for (long k = 0; k < nkpts; ++k) {
            // Standard: left X at k+q, right X at k. Transposed: swapped.
            auto Xsk_Pa_l = thc.X(s, ip, transposed ? long(k) : long(kpq_map(k)));
            auto Xsk_Pa_r = thc.X(s, iq, transposed ? long(kpq_map(k)) : long(k));
            auto O_PQ = O_skPQ(s, k, all, all);

            auto left_times = [&](auto const& O_ab) {
              if (kp_trev(k)) {
                nda::blas::gemm(Xsk_Pa_l(X_P_rng, all), nda::transpose(O_ab), Ask_Pb);
              } else {
                nda::blas::gemm(Xsk_Pa_l(X_P_rng, all), O_ab, Ask_Pb);
              }
            };
            if constexpr (transposed) {
              O_buf = nda::dagger(O_skab(s, kp_map(k), all, all));
              left_times(O_buf);
            } else {
              left_times(O_skab(s, kp_map(k), all, all));
            }

            // Osk_PQ = Ask_Pb * conj(Xsk_Qb)
            nda::blas::gemm(Ask_Pb, nda::dagger(Xsk_Pa_r(O_Q_rng, all)), O_PQ);

            if (has_deltax) {
              auto DX_q  = ibc->DeltaX(s, k, all, all);         // δ^q X(k), stored at index k
              // DeltaX_minusq[ik] stores δ^{-q} X(k_ik + q). We want δ^{-q} X(k+q) where
              // k is the outer loop index, so read at index k — NOT kpq_map(k).
              auto DX_mq = ibc->DeltaX_minusq(s, k, all, all);  // δ^{-q} X(k+q)

              // O_buf = A(kk), or A(kk)^H when transposed
              auto load_unpert = [&](long kk) {
                if constexpr (transposed) O_buf = nda::dagger((*O_unpert)(s, kk, all, all));
                else                      O_buf = (*O_unpert)(s, kk, all, all);
              };
              // O_PQ += L_P · O_buf · R_Q†
              auto add_term = [&](auto const& L, auto const& R) {
                nda::blas::gemm(L(X_P_rng, all), O_buf, Ask_Pb);
                nda::blas::gemm(ComplexType(1.0), Ask_Pb, nda::dagger(R(O_Q_rng, all)),
                                ComplexType(1.0), O_PQ);
              };

              load_unpert(k);
              if constexpr (transposed) add_term(Xsk_Pa_l, DX_q);
              else                      add_term(DX_q, Xsk_Pa_r);
              load_unpert(kpq_map(k));
              if constexpr (transposed) add_term(DX_mq, Xsk_Pa_r);
              else                      add_term(Xsk_Pa_l, DX_mq);
            }
          }
        }

        // Conjugate output (applies to both standard and DeltaX terms)
        if constexpr (transposed) O_skPQ = nda::conj(O_skPQ);
      }

      /**
       * Local part of the LR aux→primary contraction: fills O_buf(i,·,·) with this
       * rank's (P,Q)-tile contribution to X(k+q)† · O_PQ(k) · X(k) for every flattened
       * leading index i, with no communication. Matches the gemm loop of
       * thc_solver_comm::_aux_to_primary_impl, except the X lookup:
       *   - Left X uses X(kpq_map(kp_map(k))) instead of X(kp_map(k))
       *   - Right X uses X(kp_map(k)) (unchanged)
       */
      template<nda::Array Array_aux_t>
      static void _aux_to_primary_gemm(int ip, int iq,
                                       const Array_aux_t &O_tskPQ,
                                       size_t nbnd,
                                       THC_ERI auto& thc,
                                       nda::ArrayOfRank<1> auto const& kp_map,
                                       nda::ArrayOfRank<1> auto const& kpq_map,
                                       long k_offset, long P_offset, long Q_offset,
                                       utils::TimerManager* Timer,
                                       nda::array<ComplexType, 3>& O_buf) {
        static_assert(nda::get_rank<Array_aux_t> >= 4,
                      "lr_thc_comm::_aux_to_primary_gemm: Rank < 4");

        auto tic = [&](const char* c) { if(Timer) Timer->start(c); };
        auto toc = [&](const char* c) { if(Timer) Timer->stop(c); };

        decltype(nda::range::all) all;

        constexpr int N = nda::get_rank<Array_aux_t>;

        size_t ns_loc = O_tskPQ.shape(N-4);
        size_t nk_loc = O_tskPQ.shape(N-3);
        size_t NP_loc = O_tskPQ.shape(N-2);
        size_t NQ_loc = O_tskPQ.shape(N-1);
        nda::range P_rng(P_offset, P_offset+NP_loc);
        nda::range Q_rng(Q_offset, Q_offset+NQ_loc);
        utils::check(NP_loc+P_offset <= thc.Np(), "lr_thc_comm::_aux_to_primary_gemm: NP_loc+P_offset > thc.Np()");
        utils::check(NQ_loc+Q_offset <= thc.Np(), "lr_thc_comm::_aux_to_primary_gemm: NQ_loc+Q_offset > thc.Np()");

        size_t dim0 = std::accumulate(O_tskPQ.shape().begin(), O_tskPQ.shape().end()-2, (size_t)1, std::multiplies<>{});

        // Flattened over the leading axes, which is only a relabelling of the same
        // memory when it is contiguous.
        utils::check(O_tskPQ.indexmap().is_contiguous(),
                     "lr_thc_comm::_aux_to_primary_gemm: the aux array must be contiguous.");
        auto O_iPQ_3D = nda::reshape(O_tskPQ, shape_t<3>{dim0, NP_loc, NQ_loc});

        // X† · O_PQ · X can be associated either way. Both cost nbnd·NP_loc·NQ_loc
        // for the first gemm plus nbnd² times the extent contracted last, so
        // contract the larger of the two aux extents first.
        const bool q_first = (NP_loc <= NQ_loc);
        nda::array<ComplexType, 2> Ask_buf(q_first ? NP_loc : nbnd,
                                           q_first ? nbnd : NQ_loc);

        // Caller-owned buffer holding local (t,s,k) slices of O_iab. It needs no
        // zeroing: the loop below closes every (i,·,·) block with a 3-argument gemm,
        // i.e. β = 0, so the whole buffer is assigned before it is read.
        auto buf_shape = shape_t<3>{(long)dim0, (long)nbnd, (long)nbnd};
        if (O_buf.shape() != buf_shape) O_buf.resize(buf_shape);
        nda::array_view<ComplexType, 3> O_buf_iab(O_buf);

        tic("SIGMA_A2P_GEMM");
        for (size_t i = 0; i < dim0; ++i) {
          // i = (it * ns_loc + is) * nk_loc + ik
          size_t s = (i / nk_loc) % ns_loc;
          size_t k = i % nk_loc + k_offset;

          // === LR change: left X uses kp_map(k)+q, right X uses kp_map(k) ===
          auto Xsk_Pa_l = thc.X(s, ip, kpq_map(kp_map(k)));
          auto Xsk_Pa_r = thc.X(s, iq, kp_map(k));

          auto Oab_i = O_buf_iab(i, all, all);
          if (q_first) {
            // Ask_Pb = Osk_PQ * Xsk_Qb;  Osk_ab = conj(Xsk_Pa) * Ask_Pb
            nda::blas::gemm(O_iPQ_3D(i, all, all), Xsk_Pa_r(Q_rng, all), Ask_buf);
            nda::blas::gemm(nda::dagger(Xsk_Pa_l(P_rng, all)), Ask_buf, Oab_i);
          } else {
            // Ask_aQ = conj(Xsk_Pa) * Osk_PQ;  Osk_ab = Ask_aQ * Xsk_Qb
            nda::blas::gemm(nda::dagger(Xsk_Pa_l(P_rng, all)), O_iPQ_3D(i, all, all), Ask_buf);
            nda::blas::gemm(Ask_buf, Xsk_Pa_r(Q_rng, all), Oab_i);
          }
        } // i
        toc("SIGMA_A2P_GEMM");
      }

    }; // lr_thc_comm
  } // solvers
} // methods

#endif // COQUI_LR_THC_COMM_HPP
