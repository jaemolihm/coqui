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
     */
    struct lr_thc_comm {
      template<nda::MemoryArray local_Array_t>
      using dArray_t = math::nda::distributed_array<local_Array_t, mpi3::communicator>;
      template<nda::MemoryArray Array_base_t>
      using sArray_t = math::shm::shared_array<Array_base_t>;
      template<int N>
      using shape_t = std::array<long, N>;

      /**
       * LR primary→aux wrapper. Matches thc_solver_comm::primary_to_aux (rank-4
       * distributed array overload, lines 118-153) line-by-line.
       *
       * IBZ → full BZ: O_skab is indexed by IBZ k-points, O_skPQ by full BZ k-points.
       *
       * @param ip         - [INPUT] left polarization index
       * @param iq         - [INPUT] right polarization index
       * @param O_skab     - [INPUT] tensor in primary basis (ns, nkpts_ibz, nbnd, nbnd)
       * @param O_skPQ     - [OUTPUT] tensor in auxiliary basis (ns, nkpts, NP, NP)
       * @param thc        - [INPUT] THC-ERI handler
       * @param kp_to_ibz  - [INPUT] full BZ k → IBZ k mapping (nkpts,)
       * @param kp_trev    - [INPUT] time-reversal flag per full BZ k-point (nkpts,)
       * @param kpq_map    - [INPUT] full BZ k → full BZ k+q mapping (nkpts,)
       */
      template<nda::MemoryArray AF_t>
      static void primary_to_aux(int ip, int iq,
                                 const nda::MemoryArrayOfRank<4> auto& O_skab,
                                 dArray_t<nda::array<ComplexType, 4>>& O_skPQ,
                                 THC_ERI auto& thc,
                                 nda::ArrayOfRank<1> auto const& kp_to_ibz,
                                 nda::ArrayOfRank<1> auto const& kp_trev,
                                 nda::ArrayOfRank<1> auto const& kpq_map) {
        // Matches thc_solver_comm::primary_to_aux rank-4 case (lines 139-149)
        auto P_offset = O_skPQ.origin()[2];
        auto Q_offset = O_skPQ.origin()[3];

        auto O_skPQ_loc = O_skPQ.local();
        utils::check(O_skPQ.global_shape()[1] == O_skPQ.local_shape()[1],
                     "lr_thc_comm::primary_to_aux: Does not support mpi distributed along k-axis.");

        _primary_to_aux_impl(ip, iq, O_skab, O_skPQ_loc, thc,
                             kp_to_ibz, kp_trev, kpq_map,
                             1, 0, 1, P_offset, Q_offset);
      }

      /**
       * LR aux→primary wrapper. Matches thc_solver_comm::aux_to_primary (rank-4
       * distributed→shared overload, lines 260-325) line-by-line.
       *
       * IBZ → IBZ: both O_skPQ and O_skab are indexed by IBZ k-points.
       * The full BZ k-index for X lookup is obtained via kp_map (= ks_to_k(0)).
       *
       * @param ip       - [INPUT] left polarization index
       * @param iq       - [INPUT] right polarization index
       * @param scl      - [INPUT] scaling factor for accumulation
       * @param O_skPQ   - [INPUT] tensor in auxiliary basis (ns, nkpts_ibz, NP, NP)
       * @param O_skab   - [OUTPUT] tensor in primary basis (ns, nkpts_ibz, nbnd, nbnd), accumulated
       * @param thc      - [INPUT] THC-ERI handler
       * @param kp_map   - [INPUT] IBZ k → full BZ k mapping, i.e. ks_to_k(0) (nkpts_ibz,)
       * @param kpq_map  - [INPUT] full BZ k → full BZ k+q mapping (nkpts,)
       */
      template<nda::MemoryArray AF_t, nda::MemoryArray Array_aux_t, typename communicator_t>
      static void aux_to_primary(int ip, int iq,
                                 ComplexType scl,
                                 const memory::darray_t<Array_aux_t, communicator_t>& O_skPQ,
                                 sArray_t<AF_t>& O_skab,
                                 THC_ERI auto& thc,
                                 nda::ArrayOfRank<1> auto const& kp_map,
                                 nda::ArrayOfRank<1> auto const& kpq_map) {
        using value_type = typename std::decay_t<AF_t>::value_type;

        // Matches thc_solver_comm::aux_to_primary rank-4 case (lines 267-324)

        // to compensate for reduction
        O_skab.win().fence();
        O_skab.communicator()->barrier();
        if(O_skab.node_comm()->root())
          O_skab.local() /= value_type(O_skab.internode_comm()->size());
        O_skab.node_comm()->barrier();

        auto pgrid = O_skPQ.grid();
        auto s_rng = O_skPQ.local_range(0);
        auto k_rng = O_skPQ.local_range(1);

        auto [s_org, k_org, P_org, Q_org] = O_skPQ.origin();
        auto [ns, nkpts, NP, NQ] = O_skPQ.global_shape();
        utils::check(nkpts == O_skPQ.local_shape()[1],
                     "lr_thc_comm::aux_to_primary: Does not support mpi distributed along k-axis.");

        auto O_skab_loc = O_skab.local()(s_rng, k_rng, nda::ellipsis{});
        auto O_skPQ_loc = O_skPQ.local();

        // Setup q_intra_comm
        communicator_t *gcomm = O_skPQ.communicator();
        int color = s_org * nkpts + k_org;
        int key = gcomm->rank();
        communicator_t dim0_intra_comm = gcomm->split(color, key);
        utils::check(dim0_intra_comm.size() == pgrid[2] * pgrid[3],
                     "dim0_intra_comm.size() != pgrid[2]*pgrid[3]");

        _aux_to_primary_impl(ip, iq, dim0_intra_comm, scl,
                             O_skPQ_loc, O_skab_loc, thc,
                             kp_map, kpq_map, k_org, P_org, Q_org);

        // reduce
        O_skab.win().fence();
        O_skab.all_reduce();
      }

    private:
      /**
       * LR primary→aux impl. Matches thc_solver_comm::_primary_to_aux_impl
       * (lines 383-450) line-by-line, except the X lookup:
       *   - Left X uses X(kpq_map(k)) instead of X(k)
       *   - Right X uses X(k) (unchanged)
       */
      template<nda::Array Array_primary_t, nda::Array Array_aux_t>
      static void _primary_to_aux_impl(int ip, int iq,
                                       const Array_primary_t& O_tskab,
                                       Array_aux_t& O_tskPQ,
                                       THC_ERI auto& thc,
                                       nda::ArrayOfRank<1> auto const& kp_map,
                                       nda::ArrayOfRank<1> auto const& kp_trev,
                                       nda::ArrayOfRank<1> auto const& kpq_map,
                                       long nbatch,
                                       int rank, int comm_size,
                                       long P_offset = 0, long Q_offset = 0) {
        static_assert(nda::get_rank<Array_primary_t> == nda::get_rank<Array_aux_t>,
                      "lr_thc_comm::_primary_to_aux_impl: Rank mismatch");
        static_assert(nda::get_rank<Array_primary_t> >= 4,
                      "lr_thc_comm::_primary_to_aux_impl: Rank < 4");

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
        }
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
        nda::array<ComplexType, 2> Oab_buffer(nbnd, nbnd);

        for (size_t i = 0; i < dim0; ++i) {
          // i = (it * ns_loc + is) * nk_loc + ik
          size_t s = (i / nk_loc) % ns_loc;
          size_t k = i % nk_loc + k_offset;

          // === LR change: left X uses kp_map(k)+q, right X uses kp_map(k) ===
          auto Xsk_Pa_l = thc.X(s, ip, kpq_map(kp_map(k)));
          auto Xsk_Pa_r = thc.X(s, iq, kp_map(k));
          nda::blas::gemm(nda::dagger(Xsk_Pa_l(P_rng, all)), O_iPQ_3D(i, all, all), Ask_aQ);

          // Osk_ab = Ask_aQ * Xsk_Qb
          nda::blas::gemm(Ask_aQ, Xsk_Pa_r(Q_rng, all), Oab_buffer);
          dim0_comm.reduce_in_place_n(Oab_buffer.data(), Oab_buffer.size(), std::plus<>{}, 0);
          if (dim0_comm.root()) {
            O_iab_3D(i, all, all) += scl*Oab_buffer;
          }
        } // i
      }

    }; // lr_thc_comm
  } // solvers
} // methods

#endif // COQUI_LR_THC_COMM_HPP
