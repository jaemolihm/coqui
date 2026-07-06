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

#include "mpi3/communicator.hpp"
#include "nda/nda.hpp"
#include "utilities/proc_grid_partition.hpp"
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
       * LR primary→aux wrapper (rank-4 and rank-5 distributed array overload).
       * Matches thc_solver_comm::primary_to_aux (lines 118-153) using if constexpr
       * to branch on rank.
       *
       * @param ip         - [INPUT] left polarization index
       * @param iq         - [INPUT] right polarization index
       * @param O_Iab      - [INPUT] tensor in primary basis
       *                     rank-5: (nt_half, ns, nkpts_ibz, nbnd, nbnd)
       *                     rank-4: (ns, nkpts_ibz, nbnd, nbnd)
       * @param O_IPQ      - [OUTPUT] tensor in auxiliary basis
       *                     rank-5: (nt_half, ns, nkpts, NP, NP)
       *                     rank-4: (ns, nkpts, NP, NP)
       * @param thc        - [INPUT] THC-ERI handler
       * @param kp_to_ibz  - [INPUT] full BZ k → IBZ k mapping (nkpts,)
       * @param kp_trev    - [INPUT] time-reversal flag per full BZ k-point (nkpts,)
       * @param kpq_map    - [INPUT] full BZ k → full BZ k+q mapping (nkpts,)
       * @param ibc        - [INPUT] optional IBC correction data (DeltaX arrays + unperturbed quantities)
       * @param O_unpert   - [INPUT] optional unperturbed quantity in primary basis (same shape as O_Iab)
       *                     Required when ibc is non-null. For rank-4: (ns, nk_ibz, nb, nb).
       *                     For rank-5: pass as rank-4 with flattened (nt_loc*ns, nk, nb, nb).
       * @param O_unpert_5d - [INPUT] optional unperturbed rank-5 quantity (nt, ns, nk_ibz, nb, nb)
       *                      Alternative to O_unpert for rank-5 transforms — auto-sliced to local τ range.
       *
       * When ibc is provided, adds correction terms:
       *   O_PQ += δ^q X(k) · A(k) · X(k)† + X(k+q) · A(k+q) · [δ^{-q} X(k+q)]†
       */
      template<nda::MemoryArray Array_primary_t, nda::MemoryArray Array_aux_t, typename communicator_t,
               typename O_unpert_4D_t = nda::array<ComplexType, 4>,
               typename O_unpert_5D_t = nda::array<ComplexType, 5>>
      static void primary_to_aux(int ip, int iq,
                                 const Array_primary_t &O_Iab,
                                 memory::darray_t<Array_aux_t, communicator_t> &O_IPQ,
                                 THC_ERI auto& thc,
                                 nda::ArrayOfRank<1> auto const& kp_to_ibz,
                                 nda::ArrayOfRank<1> auto const& kp_trev,
                                 nda::ArrayOfRank<1> auto const& kpq_map,
                                 const lr_ibc_DeltaX* ibc = nullptr,
                                 const O_unpert_4D_t* O_unpert = nullptr,
                                 const O_unpert_5D_t* O_unpert_5d = nullptr) {
        constexpr int rank = nda::get_rank<Array_primary_t>;
        static_assert(rank == 4 || rank == 5,
                      "lr_thc_comm::primary_to_aux: rank must be 4 or 5");

        // Extract DeltaX pointers from ibc (null when no correction)
        const nda::array_view<ComplexType, 4>* DeltaX_ptr = ibc ? &ibc->DeltaX : nullptr;
        const nda::array_view<ComplexType, 4>* DeltaX_mq_ptr = ibc ? &ibc->DeltaX_minusq : nullptr;

        if constexpr (rank == 5) {
          auto t_loc_rng = O_IPQ.local_range(0);
          auto P_offset = O_IPQ.origin()[3];
          auto Q_offset = O_IPQ.origin()[4];

          auto O_Iab_loc = O_Iab(t_loc_rng, nda::ellipsis{});
          auto O_IPQ_loc = O_IPQ.local();
          utils::check(O_IPQ.global_shape()[2] == O_IPQ.local_shape()[2],
                       "lr_thc_comm::primary_to_aux(rank-5): Does not support mpi distributed along k-axis.");

          if (O_unpert_5d) {
            // Slice to local tau range (rank-5 view), then copy into a rank-5 array
            // and reshape to rank-4 (dim_i = nt_loc * ns) for _impl.
            nda::array<ComplexType, 5> O_5d_loc{(*O_unpert_5d)(t_loc_rng, nda::ellipsis{})};
            long nt_loc_ = O_5d_loc.shape(0), ns_ = O_5d_loc.shape(1);
            long nk_ = O_5d_loc.shape(2), nb_ = O_5d_loc.shape(3);
            auto O_unpert_loc = nda::array<ComplexType, 4>{
                nda::reshape(O_5d_loc, shape_t<4>{nt_loc_ * ns_, nk_, nb_, nb_})};
            _primary_to_aux_impl(ip, iq, O_Iab_loc, O_IPQ_loc, thc,
                                 kp_to_ibz, kp_trev, kpq_map,
                                 1, 0, 1, P_offset, Q_offset,
                                 DeltaX_ptr, DeltaX_mq_ptr, &O_unpert_loc);
          } else {
            _primary_to_aux_impl(ip, iq, O_Iab_loc, O_IPQ_loc, thc,
                                 kp_to_ibz, kp_trev, kpq_map,
                                 1, 0, 1, P_offset, Q_offset);
          }
        } else if constexpr (rank == 4) {
          auto P_offset = O_IPQ.origin()[2];
          auto Q_offset = O_IPQ.origin()[3];

          auto O_IPQ_loc = O_IPQ.local();
          utils::check(O_IPQ.global_shape()[1] == O_IPQ.local_shape()[1],
                       "lr_thc_comm::primary_to_aux: Does not support mpi distributed along k-axis.");

          _primary_to_aux_impl(ip, iq, O_Iab, O_IPQ_loc, thc,
                               kp_to_ibz, kp_trev, kpq_map,
                               1, 0, 1, P_offset, Q_offset,
                               DeltaX_ptr, DeltaX_mq_ptr, O_unpert);
        }
      }

      /**
       * LR aux→primary wrapper (rank-4 and rank-5 distributed→shared overload).
       * Matches thc_solver_comm::aux_to_primary (lines 260-325) using if constexpr
       * to branch on rank.
       *
       * @param ip       - [INPUT] left polarization index
       * @param iq       - [INPUT] right polarization index
       * @param scl      - [INPUT] scaling factor for accumulation
       * @param O_IPQ    - [INPUT] tensor in auxiliary basis
       *                   rank-5: (nt_half, ns, nkpts_ibz, NP, NP)
       *                   rank-4: (ns, nkpts_ibz, NP, NP)
       * @param O_Iab    - [OUTPUT] tensor in primary basis, accumulated
       *                   rank-5: (nt_half, ns, nkpts_ibz, nbnd, nbnd)
       *                   rank-4: (ns, nkpts_ibz, nbnd, nbnd)
       * @param thc      - [INPUT] THC-ERI handler
       * @param kp_map   - [INPUT] IBZ k → full BZ k mapping, i.e. ks_to_k(0) (nkpts_ibz,)
       * @param kpq_map  - [INPUT] full BZ k → full BZ k+q mapping (nkpts,)
       */
      template<nda::MemoryArray AF_t, nda::MemoryArray Array_aux_t, typename communicator_t>
      static void aux_to_primary(int ip, int iq,
                                 ComplexType scl,
                                 const memory::darray_t<Array_aux_t, communicator_t>& O_IPQ,
                                 sArray_t<AF_t>& O_Iab,
                                 THC_ERI auto& thc,
                                 nda::ArrayOfRank<1> auto const& kp_map,
                                 nda::ArrayOfRank<1> auto const& kpq_map) {
        constexpr int rank = nda::get_rank<Array_aux_t>;
        static_assert(nda::get_rank<AF_t> == rank,
                      "lr_thc_comm::aux_to_primary: rank mismatch between darray and sarray");
        static_assert(rank == 4 || rank == 5,
                      "lr_thc_comm::aux_to_primary: rank must be 4 or 5");
        using value_type = typename std::decay_t<AF_t>::value_type;

        // to compensate for reduction
        O_Iab.win().fence();
        O_Iab.communicator()->barrier();
        if(O_Iab.node_comm()->root())
          O_Iab.local() /= value_type(O_Iab.internode_comm()->size());
        O_Iab.node_comm()->barrier();

        if constexpr (rank == 5) {
          auto pgrid = O_IPQ.grid();
          auto t_rng = O_IPQ.local_range(0);
          auto s_rng = O_IPQ.local_range(1);
          auto k_rng = O_IPQ.local_range(2);

          auto [t_org, s_org, k_org, P_org, Q_org] = O_IPQ.origin();
          auto [nt, ns, nkpts, NP, NQ] = O_IPQ.global_shape();
          utils::check(nkpts == O_IPQ.local_shape()[2],
                       "lr_thc_comm::aux_to_primary(rank-5): Does not support mpi distributed along k-axis.");

          auto O_Iab_loc = O_Iab.local()(t_rng, s_rng, k_rng, nda::ellipsis{});
          auto O_IPQ_loc = O_IPQ.local();

          // Setup wq_intra_comm
          communicator_t *gcomm = O_IPQ.communicator();
          int color = t_org * ns * nkpts + s_org * nkpts + k_org;
          int key = gcomm->rank();
          communicator_t dim0_intra_comm = gcomm->split(color, key);
          utils::check(dim0_intra_comm.size() == pgrid[3] * pgrid[4],
                       "dim0_intra_comm.size() != pgrid[3]*pgrid[4]");

          _aux_to_primary_impl(ip, iq, dim0_intra_comm, scl,
                               O_IPQ_loc, O_Iab_loc, thc,
                               kp_map, kpq_map, k_org, P_org, Q_org);
        } else if constexpr (rank == 4) {
          auto pgrid = O_IPQ.grid();
          auto s_rng = O_IPQ.local_range(0);
          auto k_rng = O_IPQ.local_range(1);

          auto [s_org, k_org, P_org, Q_org] = O_IPQ.origin();
          auto [ns, nkpts, NP, NQ] = O_IPQ.global_shape();
          utils::check(nkpts == O_IPQ.local_shape()[1],
                       "lr_thc_comm::aux_to_primary: Does not support mpi distributed along k-axis.");

          auto O_Iab_loc = O_Iab.local()(s_rng, k_rng, nda::ellipsis{});
          auto O_IPQ_loc = O_IPQ.local();

          // Setup q_intra_comm
          communicator_t *gcomm = O_IPQ.communicator();
          int color = s_org * nkpts + k_org;
          int key = gcomm->rank();
          communicator_t dim0_intra_comm = gcomm->split(color, key);
          utils::check(dim0_intra_comm.size() == pgrid[2] * pgrid[3],
                       "dim0_intra_comm.size() != pgrid[2]*pgrid[3]");

          _aux_to_primary_impl(ip, iq, dim0_intra_comm, scl,
                               O_IPQ_loc, O_Iab_loc, thc,
                               kp_map, kpq_map, k_org, P_org, Q_org);
        }

        // reduce
        O_Iab.win().fence();
        O_Iab.all_reduce();
      }

      /**
       * LR primary→aux transposed: computes [O_PQ]^T directly.
       *
       * Uses the identity: [O_PQ]^T = conj( X_R @ O^H @ X_L^H )
       * where X_L = X(k+q) and X_R = X(k) in the standard transform.
       *
       * Internally computes O^H = dagger(O), swaps left/right X matrices,
       * and conjugates the output. Cannot reuse _primary_to_aux_impl because
       * both the X assignment and the input/output transformations differ.
       *
       * Only rank-4 is supported; rank-5 support can be added if needed.
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
        constexpr int rank = nda::get_rank<Array_primary_t>;
        static_assert(rank == 4,
                      "lr_thc_comm::primary_to_aux_transposed: only rank-4 supported");

        // Extract DeltaX pointers from ibc (null when no correction)
        const nda::array_view<ComplexType, 4>* DeltaX_ptr = ibc ? &ibc->DeltaX : nullptr;
        const nda::array_view<ComplexType, 4>* DeltaX_mq_ptr = ibc ? &ibc->DeltaX_minusq : nullptr;

        auto P_offset = O_IPQ.origin()[2];
        auto Q_offset = O_IPQ.origin()[3];

        auto O_IPQ_loc = O_IPQ.local();
        utils::check(O_IPQ.global_shape()[1] == O_IPQ.local_shape()[1],
                     "lr_thc_comm::primary_to_aux_transposed: Does not support mpi distributed along k-axis.");

        _primary_to_aux_transposed_impl(ip, iq, O_Iab, O_IPQ_loc, thc,
                             kp_to_ibz, kp_trev, kpq_map,
                             1, 0, 1, P_offset, Q_offset,
                             DeltaX_ptr, DeltaX_mq_ptr, O_unpert);
      }

    private:
      /**
       * LR primary→aux impl. Matches thc_solver_comm::_primary_to_aux_impl
       * (lines 383-450) line-by-line, except the X lookup:
       *   - Left X uses X(kpq_map(k)) instead of X(k)
       *   - Right X uses X(k) (unchanged)
       */
      template<nda::Array Array_primary_t, nda::Array Array_aux_t, typename O_unpert_t = nda::array<ComplexType, 4>>
      static void _primary_to_aux_impl(int ip, int iq,
                                       const Array_primary_t& O_tskab,
                                       Array_aux_t& O_tskPQ,
                                       THC_ERI auto& thc,
                                       nda::ArrayOfRank<1> auto const& kp_map,
                                       nda::ArrayOfRank<1> auto const& kp_trev,
                                       nda::ArrayOfRank<1> auto const& kpq_map,
                                       long nbatch,
                                       int rank, int comm_size,
                                       long P_offset = 0, long Q_offset = 0,
                                       const nda::array_view<ComplexType, 4>* DeltaX_left = nullptr,
                                       const nda::array_view<ComplexType, 4>* DeltaX_right = nullptr,
                                       const O_unpert_t* O_unpert = nullptr) {
        static_assert(nda::get_rank<Array_primary_t> == nda::get_rank<Array_aux_t>,
                      "lr_thc_comm::_primary_to_aux_impl: Rank mismatch");
        static_assert(nda::get_rank<Array_primary_t> >= 4,
                      "lr_thc_comm::_primary_to_aux_impl: Rank < 4");

        bool has_deltax = DeltaX_left && DeltaX_right && O_unpert;

        decltype(nda::range::all) all;
        constexpr int N = nda::get_rank<Array_primary_t>;
        size_t ns         = O_tskab.shape(N-4);
        size_t nkpts_ibz  = O_tskab.shape(N-3);
        size_t nbnd       = O_tskab.shape(N-2);
        size_t nkpts      = O_tskPQ.shape(N-3);
        size_t NP_loc     = O_tskPQ.shape(N-2);
        size_t NQ_loc     = O_tskPQ.shape(N-1);
        utils::check(NP_loc+P_offset <= thc.Np(), "lr_thc_comm::_primary_to_aux_impl: NP_loc+P_offset > thc.Np()");
        utils::check(NQ_loc+Q_offset <= thc.Np(), "lr_thc_comm::_primary_to_aux_impl: NQ_loc+Q_offset > thc.Np()");
        if (has_deltax) {
          utils::check(nkpts_ibz == nkpts,
                       "lr_thc_comm::_primary_to_aux_impl: DeltaX correction requires full BZ grid (nkpts_ibz={} != nkpts={})",
                       nkpts_ibz, nkpts);
        }

        // dim_i = (t, s) or (s)
        size_t dim_i = std::accumulate(O_tskPQ.shape().begin(), O_tskPQ.shape().end()-3, (size_t)1, std::multiplies<>{});
        utils::check(dim_i == std::accumulate(O_tskab.shape().begin(), O_tskab.shape().end()-3, (size_t)1, std::multiplies<>{}),
                     "lr_thc_comm::_primary_to_aux_impl: dim_i mismatched");

        auto O_ikPQ_4D = nda::reshape(O_tskPQ, shape_t<4>{dim_i, nkpts, NP_loc, NQ_loc});
        auto O_ikab_4D = nda::reshape(O_tskab, shape_t<4>{dim_i, nkpts_ibz, nbnd, nbnd});

        if (nbatch < 0) nbatch = utils::find_min_col(comm_size, dim_i*nkpts);
        utils::check(comm_size % nbatch == 0, "lr_thc_comm::_primary_to_aux_impl: comm_size % nbatch != 0");
        auto [origin, end] = itertools::chunk_range(0, NP_loc, nbatch, rank % nbatch);
        int batch_size = end - origin;
        int n_large_batch = NP_loc - nbatch * (NP_loc / nbatch);
        int offset = (rank % nbatch < n_large_batch)? 0 : 0 + n_large_batch;

        nda::array<ComplexType, 2> Ask_Pb(batch_size, nbnd);

        // DeltaX correction buffers (allocated only when needed)
        nda::matrix<ComplexType> U_ab_buf;
        if (has_deltax) {
          U_ab_buf.resize(nbnd, nbnd);
        }

        for (size_t ikP = rank; ikP < dim_i*nkpts*nbatch; ikP += comm_size) {
          // ikP = (i * nkpts + k) * nbatch + PP
          size_t i = ikP / (nkpts*nbatch);
          size_t s = i % ns; // i = it * ns + is
          size_t k = (ikP / nbatch) % nkpts;
          size_t PP = ikP % nbatch;
          nda::range X_P_rng(PP*batch_size + offset + P_offset, (PP+1)*batch_size + offset + P_offset);
          nda::range O_P_rng(PP*batch_size + offset, (PP+1)*batch_size + offset);
          nda::range O_Q_rng(Q_offset, Q_offset + NQ_loc);

          // === LR change: left X uses k+q, right X uses k ===
          auto Xsk_Pa_l = thc.X(s, ip, kpq_map(k));
          auto Xsk_Pa_r = thc.X(s, iq, k);

          if(kp_trev(k)) {
            nda::blas::gemm(Xsk_Pa_l(X_P_rng, all), nda::transpose(O_ikab_4D(i, kp_map(k), all, all)), Ask_Pb);
          } else {
            nda::blas::gemm(Xsk_Pa_l(X_P_rng, all), O_ikab_4D(i, kp_map(k), all, all), Ask_Pb);
          }

          // Osk_PQ = Ask_Pb * conj(Xsk_Qb)
          nda::blas::gemm(Ask_Pb, nda::dagger(Xsk_Pa_r(O_Q_rng, all)), O_ikPQ_4D(i, k, O_P_rng, all));

          // DeltaX correction:
          //   δ^q X(k) · A(k) · X(k)†  +  X(k+q) · A(k+q) · [δ^{-q} X(k+q)]†
          if (has_deltax) {
            auto DX_L = (*DeltaX_left)(s, k, all, all);  // δ^q X(k), stored at index k
            // DeltaX_right[ik] stores δ^{-q} X(k_ik + q). We want δ^{-q} X(k+q) where
            // k is the outer loop index, so read at index k — NOT kpq_map(k).
            auto DX_R = (*DeltaX_right)(s, k, all, all);  // δ^{-q} X(k+q)

            // Term 2: δ^q X(k) · A(k) · X(k)†
            U_ab_buf = (*O_unpert)(i, k, all, all);
            nda::blas::gemm(DX_L(X_P_rng, all), U_ab_buf, Ask_Pb);
            nda::blas::gemm(ComplexType(1.0), Ask_Pb, nda::dagger(Xsk_Pa_r(O_Q_rng, all)),
                            ComplexType(1.0), O_ikPQ_4D(i, k, O_P_rng, all));

            // Term 3: X(k+q) · A(k+q) · [δ^{-q} X(k+q)]†
            U_ab_buf = (*O_unpert)(i, kpq_map(k), all, all);
            nda::blas::gemm(Xsk_Pa_l(X_P_rng, all), U_ab_buf, Ask_Pb);
            nda::blas::gemm(ComplexType(1.0), Ask_Pb, nda::dagger(DX_R(O_Q_rng, all)),
                            ComplexType(1.0), O_ikPQ_4D(i, k, O_P_rng, all));
          }
        }
      }

      /**
       * LR primary→aux transposed impl.
       * Computes [O_PQ]^T = conj( X_R @ O^H @ X_L^H ) by:
       *   1. Using swapped X: left=X(k), right=X(k+q), with O^H computed on-the-fly
       *   2. Conjugating the output
       */
      template<nda::Array Array_primary_t, nda::Array Array_aux_t, typename O_unpert_t = nda::array<ComplexType, 4>>
      static void _primary_to_aux_transposed_impl(int ip, int iq,
                                       const Array_primary_t& O_tskab,
                                       Array_aux_t& O_tskPQ,
                                       THC_ERI auto& thc,
                                       nda::ArrayOfRank<1> auto const& kp_map,
                                       nda::ArrayOfRank<1> auto const& kp_trev,
                                       nda::ArrayOfRank<1> auto const& kpq_map,
                                       long nbatch,
                                       int rank, int comm_size,
                                       long P_offset = 0, long Q_offset = 0,
                                       const nda::array_view<ComplexType, 4>* DeltaX_left = nullptr,
                                       const nda::array_view<ComplexType, 4>* DeltaX_right = nullptr,
                                       const O_unpert_t* O_unpert = nullptr) {
        static_assert(nda::get_rank<Array_primary_t> == nda::get_rank<Array_aux_t>,
                      "lr_thc_comm::_primary_to_aux_transposed_impl: Rank mismatch");
        static_assert(nda::get_rank<Array_primary_t> >= 4,
                      "lr_thc_comm::_primary_to_aux_transposed_impl: Rank < 4");

        // Transposed: [O_PQ]^T = conj( X(k) @ O^H @ X(k+q)^H )
        // DeltaX correction (before final conjugation):
        //   pre-conj term 2: X(k) · A(k)^H · [δ^q X(k)]†
        //   pre-conj term 3: δ^{-q} X(k+q) · A(k+q)^H · X(k+q)†

        bool has_deltax = DeltaX_left && DeltaX_right && O_unpert;

        decltype(nda::range::all) all;
        constexpr int N = nda::get_rank<Array_primary_t>;
        size_t ns         = O_tskab.shape(N-4);
        size_t nkpts_ibz  = O_tskab.shape(N-3);
        size_t nbnd       = O_tskab.shape(N-2);
        size_t nkpts      = O_tskPQ.shape(N-3);
        size_t NP_loc     = O_tskPQ.shape(N-2);
        size_t NQ_loc     = O_tskPQ.shape(N-1);
        utils::check(NP_loc+P_offset <= thc.Np(), "lr_thc_comm::_primary_to_aux_transposed_impl: NP_loc+P_offset > thc.Np()");
        utils::check(NQ_loc+Q_offset <= thc.Np(), "lr_thc_comm::_primary_to_aux_transposed_impl: NQ_loc+Q_offset > thc.Np()");
        if (has_deltax) {
          utils::check(nkpts_ibz == nkpts,
                       "lr_thc_comm::_primary_to_aux_transposed_impl: DeltaX correction requires full BZ grid (nkpts_ibz={} != nkpts={})",
                       nkpts_ibz, nkpts);
        }

        // dim_i = (t, s) or (s)
        size_t dim_i = std::accumulate(O_tskPQ.shape().begin(), O_tskPQ.shape().end()-3, (size_t)1, std::multiplies<>{});
        utils::check(dim_i == std::accumulate(O_tskab.shape().begin(), O_tskab.shape().end()-3, (size_t)1, std::multiplies<>{}),
                     "lr_thc_comm::_primary_to_aux_transposed_impl: dim_i mismatched");

        auto O_ikPQ_4D = nda::reshape(O_tskPQ, shape_t<4>{dim_i, nkpts, NP_loc, NQ_loc});
        auto O_ikab_4D = nda::reshape(O_tskab, shape_t<4>{dim_i, nkpts_ibz, nbnd, nbnd});

        if (nbatch < 0) nbatch = utils::find_min_col(comm_size, dim_i*nkpts);
        utils::check(comm_size % nbatch == 0, "lr_thc_comm::_primary_to_aux_transposed_impl: comm_size % nbatch != 0");
        auto [origin, end] = itertools::chunk_range(0, NP_loc, nbatch, rank % nbatch);
        int batch_size = end - origin;
        int n_large_batch = NP_loc - nbatch * (NP_loc / nbatch);
        int offset = (rank % nbatch < n_large_batch)? 0 : n_large_batch;

        nda::array<ComplexType, 2> Ask_Pb(batch_size, nbnd);
        nda::matrix<ComplexType> O_H_slice(nbnd, nbnd);

        // DeltaX: reshape unperturbed quantity once
        using reshape_4D_t = decltype(nda::reshape(O_tskab, shape_t<4>{dim_i, nkpts_ibz, nbnd, nbnd}));
        std::optional<reshape_4D_t> U_ikab_4D_opt;
        nda::matrix<ComplexType> U_H_buf;
        if (has_deltax) {
          U_ikab_4D_opt.emplace(nda::reshape(*O_unpert, shape_t<4>{dim_i, nkpts_ibz, nbnd, nbnd}));
          U_H_buf.resize(nbnd, nbnd);
        }

        // Transform with swapped X: left=X(k), right=X(k+q), input=O^H
        for (size_t ikP = rank; ikP < dim_i*nkpts*nbatch; ikP += comm_size) {
          size_t i = ikP / (nkpts*nbatch);
          size_t s = i % ns;
          size_t k = (ikP / nbatch) % nkpts;
          size_t PP = ikP % nbatch;
          nda::range X_P_rng(PP*batch_size + offset + P_offset, (PP+1)*batch_size + offset + P_offset);
          nda::range O_P_rng(PP*batch_size + offset, (PP+1)*batch_size + offset);
          nda::range O_Q_rng(Q_offset, Q_offset + NQ_loc);

          // === Swapped X: left=X(k), right=X(k+q) ===
          auto Xsk_Pa_l = thc.X(s, ip, k);
          auto Xsk_Pa_r = thc.X(s, iq, kpq_map(k));

          // Compute O^H = dagger(O) on-the-fly for this (i, k) slice
          O_H_slice = nda::dagger(O_ikab_4D(i, kp_map(k), all, all));

          if(kp_trev(k)) {
            nda::blas::gemm(Xsk_Pa_l(X_P_rng, all), nda::transpose(O_H_slice), Ask_Pb);
          } else {
            nda::blas::gemm(Xsk_Pa_l(X_P_rng, all), O_H_slice, Ask_Pb);
          }

          nda::blas::gemm(Ask_Pb, nda::dagger(Xsk_Pa_r(O_Q_rng, all)), O_ikPQ_4D(i, k, O_P_rng, all));

          // DeltaX correction (before final conjugation)
          // Transpose of eq terms:
          //   pre-conj term 2: X(k) · A(k)^H · [δ^q X(k)]†
          //   pre-conj term 3: δ^{-q} X(k+q) · A(k+q)^H · X(k+q)†
          if (has_deltax) {
            auto& U_ikab_4D = *U_ikab_4D_opt;
            auto DX_q_at_k   = (*DeltaX_left)(s, k, all, all);  // δ^q X(k), stored at index k
            // DeltaX_right[ik] stores δ^{-q} X(k_ik + q). We want δ^{-q} X(k+q) for outer
            // loop index k, so read at index k — NOT kpq_map(k).
            auto DX_mq_at_kpq = (*DeltaX_right)(s, k, all, all);  // δ^{-q} X(k+q)

            // Pre-conj term 2: X(k) · A(k)^H · [δ^q X(k)]†
            U_H_buf = nda::dagger(U_ikab_4D(i, k, all, all));
            nda::blas::gemm(Xsk_Pa_l(X_P_rng, all), U_H_buf, Ask_Pb);
            nda::blas::gemm(ComplexType(1.0), Ask_Pb, nda::dagger(DX_q_at_k(O_Q_rng, all)),
                            ComplexType(1.0), O_ikPQ_4D(i, k, O_P_rng, all));

            // Pre-conj term 3: δ^{-q} X(k+q) · A(k+q)^H · X(k+q)†
            U_H_buf = nda::dagger(U_ikab_4D(i, kpq_map(k), all, all));
            nda::blas::gemm(DX_mq_at_kpq(X_P_rng, all), U_H_buf, Ask_Pb);
            nda::blas::gemm(ComplexType(1.0), Ask_Pb, nda::dagger(Xsk_Pa_r(O_Q_rng, all)),
                            ComplexType(1.0), O_ikPQ_4D(i, k, O_P_rng, all));
          }
        }

        // Conjugate output (applies to both standard and DeltaX terms)
        O_ikPQ_4D = nda::conj(O_ikPQ_4D);
      }

      /**
       * LR aux→primary impl. Matches thc_solver_comm::_aux_to_primary_impl
       * (communicator version, lines 536-590) line-by-line, except the X lookup:
       *   - Left X uses X(kpq_map(kp_map(k))) instead of X(kp_map(k))
       *   - Right X uses X(kp_map(k)) (unchanged)
       */
      template<nda::Array Array_aux_t, nda::Array Array_primary_t, typename communicator_t>
      static void _aux_to_primary_impl(int ip, int iq,
                                       communicator_t &dim0_comm,
                                       ComplexType scl,
                                       const Array_aux_t &O_tskPQ,
                                       Array_primary_t &O_tskab,
                                       THC_ERI auto& thc,
                                       nda::ArrayOfRank<1> auto const& kp_map,
                                       nda::ArrayOfRank<1> auto const& kpq_map,
                                       long k_offset, long P_offset, long Q_offset) {
        static_assert(nda::get_rank<Array_primary_t> == nda::get_rank<Array_aux_t>,
                      "lr_thc_comm::_aux_to_primary_impl: Rank mismatch");
        static_assert(nda::get_rank<Array_primary_t> >= 4,
                      "lr_thc_comm::_aux_to_primary_impl: Rank < 4");

        decltype(nda::range::all) all;

        constexpr int N = nda::get_rank<Array_primary_t>;

        size_t nbnd = O_tskab.shape(N-2);
        size_t ns_loc = O_tskPQ.shape(N-4);
        size_t nk_loc = O_tskPQ.shape(N-3);
        size_t NP_loc = O_tskPQ.shape(N-2);
        size_t NQ_loc = O_tskPQ.shape(N-1);
        nda::range P_rng(P_offset, P_offset+NP_loc);
        nda::range Q_rng(Q_offset, Q_offset+NQ_loc);
        utils::check(NP_loc+P_offset <= thc.Np(), "lr_thc_comm::_aux_to_primary_impl: NP_loc+P_offset > thc.Np()");
        utils::check(NQ_loc+Q_offset <= thc.Np(), "lr_thc_comm::_aux_to_primary_impl: NQ_loc+Q_offset > thc.Np()");

        size_t dim0 = std::accumulate(O_tskPQ.shape().begin(), O_tskPQ.shape().end()-2, (size_t)1, std::multiplies<>{});

        auto O_iPQ_3D = nda::reshape(O_tskPQ, shape_t<3>{dim0, NP_loc, NQ_loc});
        auto O_iab_3D = nda::reshape(O_tskab, shape_t<3>{dim0, nbnd, nbnd});

        nda::array<ComplexType, 2> Ask_aQ(nbnd, NQ_loc);

        // buffer array to hold local (t,s,k) slices of O_iab for reduction
        nda::array<ComplexType, 3> O_buf_iab(dim0, nbnd, nbnd);

        for (size_t i = 0; i < dim0; ++i) {
          // i = (it * ns_loc + is) * nk_loc + ik
          size_t s = (i / nk_loc) % ns_loc;
          size_t k = i % nk_loc + k_offset;

          // === LR change: left X uses kp_map(k)+q, right X uses kp_map(k) ===
          auto Xsk_Pa_l = thc.X(s, ip, kpq_map(kp_map(k)));
          auto Xsk_Pa_r = thc.X(s, iq, kp_map(k));
          nda::blas::gemm(nda::dagger(Xsk_Pa_l(P_rng, all)), O_iPQ_3D(i, all, all), Ask_aQ);

          // Osk_ab = Ask_aQ * Xsk_Qb
          auto Oab_i = O_buf_iab(i, all, all);
          nda::blas::gemm(Ask_aQ, Xsk_Pa_r(Q_rng, all), Oab_i);
        } // i

        // Accumulate all (t,s,k) slices locally and reduce once.
        dim0_comm.reduce_in_place_n(O_buf_iab.data(), O_buf_iab.size(), std::plus<>{}, 0);
        if (dim0_comm.root()) {
          O_iab_3D += scl*O_buf_iab;
        }
      }

    }; // lr_thc_comm
  } // solvers
} // methods

#endif // COQUI_LR_THC_COMM_HPP
