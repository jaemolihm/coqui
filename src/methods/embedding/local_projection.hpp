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


#ifndef COQUI_LOCAL_PROJECTION_HPP
#define COQUI_LOCAL_PROJECTION_HPP

#include <vector>

#include "configuration.hpp"
#include "nda/nda.hpp"
#include "nda/blas.hpp"
#include "numerics/nda_functions.hpp"
#include "numerics/gemm_guard.hpp"
#include "numerics/shared_array/nda.hpp"
#include "numerics/distributed_array/nda.hpp"
#include "utilities/check.hpp"

// Projections between the band basis (k, i, j) / THC auxiliary basis (q, P, Q) and the local
// impurity basis, shared by projector_t, embed_eri_t and scr_coulomb_t.
// Header-only, so any library can include it without adding a link edge.
//
// A response at momentum transfer Q is the (k+Q, k) block of the band-basis operator;
// kpq_map(k) is the index of k+Q. kpq_map = nullptr means Q = 0.
namespace methods::local_projection {

  // O_ab += C^{ik_l}_{ai} O_ij C^{ik_r*}_{bj} for spin is and impurity imp
  inline void add_C_O_Cdag(const nda::array<ComplexType, 5> &C_skIai, long is, long ik_l, long ik_r, long imp,
                           const nda::array<ComplexType, 2> &O_ij_W, nda::array<ComplexType, 2> &buffer_aj,
                           nda::MemoryArrayOfRank<2> auto &&O_ab) {
    nda::blas::gemm(C_skIai(is, ik_l, imp, nda::ellipsis{}), O_ij_W, buffer_aj);

    nda::blas::gemm(ComplexType(1.0), buffer_aj, nda::dagger(C_skIai(is, ik_r, imp, nda::ellipsis{})),
                    ComplexType(1.0), O_ab);
  }

  // O_tsIab = (1/N_k) sum_k C^{kpq_map(k)}_{ai} O^k_tsij C^{k*}_{bj} over the full MF k mesh (N_k = C_skIai.shape(1)).
  // O_tskij lives on the IBZ; kp_to_ibz / kp_trev (both nullptr: identity, no time reversal) unfold it to
  // the full mesh. Collective on comm.
  template<typename comm_t>
  auto sum_C_O_Cdag(const nda::MemoryArrayOfRank<5> auto &O_tskij,
                    const nda::array<ComplexType, 5> &C_skIai, const std::vector<nda::range> &W_rng,
                    const nda::array<int, 1> *kp_to_ibz, const nda::array<bool, 1> *kp_trev,
                    const nda::array<long, 1> *kpq_map, comm_t comm)
  -> nda::array<ComplexType, 5> {

    utils::check(O_tskij.shape()[1] == C_skIai.shape(0), "sum_C_O_Cdag: ns mismatches.{}, {}",
                 O_tskij.shape()[1], C_skIai.shape(0));
    utils::check(kp_to_ibz != nullptr or O_tskij.shape()[2] == C_skIai.shape(1),
                 "sum_C_O_Cdag: without kp_to_ibz, O must be on the full mesh (O nk = {}, C nk = {}).",
                 O_tskij.shape()[2], C_skIai.shape(1));
    utils::check(kp_to_ibz == nullptr or kp_to_ibz->shape(0) == C_skIai.shape(1),
                 "sum_C_O_Cdag: kp_to_ibz size {} != nkpts {}.", (kp_to_ibz == nullptr)? 0 : kp_to_ibz->shape(0), C_skIai.shape(1));

    auto [nts, ns, nkpts_ibz, nbnd, nbnd_b] = O_tskij.shape();
    auto nImps = C_skIai.shape(2);
    auto nImpOrbs = C_skIai.shape(3);
    auto nOrbs_W = C_skIai.shape(4);
    auto nkpts = C_skIai.shape(1);

    nda::array<ComplexType, 5> O_tsIab(nts, ns, nImps, nImpOrbs, nImpOrbs);
    O_tsIab() = 0.0;
    nda::array<ComplexType, 2> buffer_aj(nImpOrbs, nOrbs_W);
    nda::array<ComplexType, 2> O_ij_W(nOrbs_W, nOrbs_W);

    int rank = comm.rank();
    int size = comm.size();
    for (long tsk = rank; tsk < nts*ns*nkpts; tsk += size) {
      long it = tsk / (ns*nkpts); // tsk = it * ns*nkpts + is * nkpts + ik
      long is = (tsk / nkpts) % ns;
      long ik = tsk % nkpts;
      long ik_ibz = (kp_to_ibz == nullptr)? ik : (*kp_to_ibz)(ik);
      for (long imp = 0; imp < nImps; ++imp) {
        if (kp_trev != nullptr and (*kp_trev)(ik))
          O_ij_W = nda::conj(O_tskij(it, is, ik_ibz, W_rng[imp], W_rng[imp]));
        else
          O_ij_W = O_tskij(it, is, ik_ibz, W_rng[imp], W_rng[imp]);

        long ik_l = (kpq_map == nullptr)? ik : (*kpq_map)(ik);
        add_C_O_Cdag(C_skIai, is, ik_l, ik, imp, O_ij_W, buffer_aj, O_tsIab(it, is, imp, nda::ellipsis{}));
      }
    }
    comm.all_reduce_in_place_n(O_tsIab.data(), O_tsIab.size(), std::plus<>{});
    O_tsIab() /= nkpts;
    return O_tsIab;
  }

  // O_tskij[k, W_rng, W_rng] += alpha * C^{kpq_map(k)\dagger} Oloc_tsIab C^k over the k axis of O_tskij (the IBZ
  // without a k+Q map, where C is indexed by the IBZ index; the full BZ with one).
  // Adjoint of sum_C_O_Cdag up to 1/N_k. Collective on O_tskij.node_comm(); the ranks of a node deal the
  // (t, s, k) blocks round-robin and write them directly into the shared window.
  template<nda::ArrayOfRank<5> Array_base_t, nda::ArrayOfRank<5> Oloc_t>
  void upfold_accumulate(math::shm::shared_array<Array_base_t> &O_tskij, const Oloc_t &Oloc_tsIab,
                         const nda::array<ComplexType, 5> &C_skIai, const std::vector<nda::range> &W_rng,
                         const nda::array<long, 1> *kpq_map, ComplexType alpha) {
    auto [nts, ns, nkpts, nbnd, nbnd_b] = O_tskij.shape();
    long nImps = C_skIai.shape(2);
    long nImpOrbs = C_skIai.shape(3);
    long nOrbs_W = C_skIai.shape(4);
    utils::check(nkpts <= C_skIai.shape(1) and (kpq_map == nullptr or (nkpts == C_skIai.shape(1) and kpq_map->shape(0) == nkpts)),
                 "upfold_accumulate: a k+Q map requires the full BZ (nkpts = {}, C nk = {}, kpq_map: {}).",
                 nkpts, C_skIai.shape(1), (kpq_map == nullptr)? nkpts : kpq_map->shape(0));
    utils::check(nts == Oloc_tsIab.shape(0) and ns == Oloc_tsIab.shape(1) and ns == C_skIai.shape(0) and
                 Oloc_tsIab.shape(2) == nImps and Oloc_tsIab.shape(3) == nImpOrbs and Oloc_tsIab.shape(4) == nImpOrbs,
                 "upfold_accumulate: shape mismatch (O: {}, {}, {}; Oloc: {}, {}, {}, {}, {}).",
                 nts, ns, nkpts, Oloc_tsIab.shape(0), Oloc_tsIab.shape(1), Oloc_tsIab.shape(2), Oloc_tsIab.shape(3),
                 Oloc_tsIab.shape(4));
    utils::check(long(W_rng.size()) == nImps, "upfold_accumulate: W_rng has {} ranges for {} impurities.",
                 W_rng.size(), nImps);
    for (long imp = 0; imp < nImps; ++imp)
      utils::check(W_rng[imp].size() == nOrbs_W, "upfold_accumulate: W_rng[{}] has size {} != nOrbs_W = {}.",
                   imp, W_rng[imp].size(), nOrbs_W);

    nda::array<ComplexType, 2> buffer_ib(nOrbs_W, nImpOrbs);
    nda::array<ComplexType, 2> block(nOrbs_W, nOrbs_W);
    auto O_loc = O_tskij.local();
    int rank = O_tskij.node_comm()->rank();
    int size = O_tskij.node_comm()->size();

    O_tskij.win().fence();
    for (long imp = 0; imp < nImps; ++imp) {
      for (long tsk = rank; tsk < nts*ns*nkpts; tsk += size) {
        long it = tsk / (ns*nkpts); // tsk = it * ns*nkpts + is * nkpts + ik
        long is = (tsk / nkpts) % ns;
        long ik = tsk % nkpts;
        long ik_l = (kpq_map == nullptr)? ik : (*kpq_map)(ik);

        nda::blas::gemm(ComplexType(1.0),
                        nda::dagger(C_skIai(is, ik_l, imp, nda::ellipsis{})),
                        Oloc_tsIab(it, is, imp, nda::ellipsis{}),
                        ComplexType(0.0),
                        buffer_ib);
        nda::blas::gemm(buffer_ib, C_skIai(is, ik, imp, nda::ellipsis{}), block);
        O_loc(it, is, ik, W_rng[imp], W_rng[imp]) += alpha * block;
      }
    }
    O_tskij.win().fence();
  }

  // W_cd_ab += conj(B^{iq_l}_{P,dc}) W_PQ B^{iq_r}_{Q,ab}, with P = P_origin + [0, NP_loc) and Q in Q_rng
  template<nda::ArrayOfRank<5> B_t>
  void add_Bdag_W_B(const nda::array<ComplexType, 2> &W_PQ, const B_t &B_qIPab, long iq_l, long iq_r,
                    long P_origin, nda::range Q_rng,
                    nda::array<ComplexType, 2> &T_P_ab, nda::array<ComplexType, 3> &B_cdP_conj,
                    nda::MemoryArrayOfRank<2> auto &&W_cd_ab) {
    long nImpOrbs = B_cdP_conj.shape(0);
    long NP_loc = B_cdP_conj.shape(2);
    long NQ_loc = Q_rng.size();
    auto B_Q_ab = nda::reshape(B_qIPab(iq_r, 0, Q_rng, nda::ellipsis{}), std::array<long, 2>{NQ_loc, nImpOrbs*nImpOrbs});
    math::detail::gemm_guarded(W_PQ, B_Q_ab, T_P_ab);

    for (long P = 0; P < NP_loc; ++P) {
      auto B_dc = B_qIPab(iq_l, 0, P_origin+P, nda::ellipsis{});
      B_cdP_conj(nda::range::all, nda::range::all, P) = nda::conj(nda::transpose(B_dc));
    }
    auto B_cd_P_conj = nda::reshape(B_cdP_conj, std::array<long, 2>{nImpOrbs*nImpOrbs, NP_loc});
    math::detail::gemm_guarded(ComplexType(1.0), B_cd_P_conj, T_P_ab, ComplexType(1.0), W_cd_ab);
  }

  // W_xcdab = (1/N_q) sum_q conj(B^{kpq(q)}_{P,dc}) W_xqPQ B^q_{Q,ab} over the full q mesh, where W_block is the
  // (q+Q, q) block of a response with leading axis x, stored at `origin` = (x, q, P, Q) of an (nx, nq, NP, NP) array.
  // replicated: every rank holds the same block, dealt out round-robin over comm; otherwise the blocks of the
  // ranks are disjoint. An empty block still enters the all_reduce. Collective on comm.
  template<nda::ArrayOfRank<5> B_t, typename comm_t>
  auto downfold_W_block(nda::array_const_view<ComplexType, 4> W_block, std::array<long, 4> origin, long nx,
                        const B_t &B_qIPab, const nda::array<long, 1> &kpq_map,
                        bool replicated, comm_t &comm)
  -> nda::array<ComplexType, 5> {
    auto [nqpts, nImps, NP, nImpOrbs, nImpOrbs2] = B_qIPab.shape();
    utils::check(nImps == 1, "downfold_W_block: only a single impurity is supported (nImps = {}).", nImps);
    utils::check(kpq_map.shape(0) == nqpts,
                 "downfold_W_block: requires the full q mesh (B nq = {}, kpq_map: {}).", nqpts, kpq_map.shape(0));
    auto [nx_loc, nq_loc, NP_loc, NQ_loc] = W_block.shape();
    auto [x_origin, q_origin, P_origin, Q_origin] = origin;
    utils::check(W_block.size() == 0 or
                 (x_origin + nx_loc <= nx and q_origin + nq_loc <= nqpts and
                  P_origin + NP_loc <= NP and Q_origin + NQ_loc <= NP),
                 "downfold_W_block: the block ({}, {}, {}, {}) at ({}, {}, {}, {}) exceeds (nx, nq, NP, NP) = ({}, {}, {}, {}).",
                 nx_loc, nq_loc, NP_loc, NQ_loc, x_origin, q_origin, P_origin, Q_origin, nx, nqpts, NP, NP);

    nda::array<ComplexType, 5> W_xcdab(nx, nImpOrbs, nImpOrbs, nImpOrbs, nImpOrbs);
    W_xcdab() = ComplexType(0.0);

    // W_xcdab = conj(B^{q+Q}_{P,dc}) * [ W_xqPQ ] * B^q_{Q,ab}
    nda::array<ComplexType, 2> T_P_ab(NP_loc, nImpOrbs*nImpOrbs);
    nda::array<ComplexType, 3> B_cdP_conj(nImpOrbs, nImpOrbs, NP_loc);
    nda::array<ComplexType, 2> W_PQ(NP_loc, NQ_loc);
    nda::range Q_rng(Q_origin, Q_origin + NQ_loc);

    for (long xq = 0; xq < (W_block.size() == 0 ? 0 : nx_loc*nq_loc); ++xq) {
      long ix = x_origin + xq / nq_loc; // xq = ix_loc * nq_loc + iq_loc
      long iq = q_origin + xq % nq_loc;
      if (replicated and (ix*nqpts + iq) % comm.size() != comm.rank()) continue;
      auto W_cd_ab = nda::reshape(W_xcdab(ix, nda::ellipsis{}), std::array<long, 2>{nImpOrbs*nImpOrbs, nImpOrbs*nImpOrbs});
      W_PQ = W_block(xq / nq_loc, xq % nq_loc, nda::ellipsis{});
      add_Bdag_W_B(W_PQ, B_qIPab, kpq_map(iq), iq, P_origin, Q_rng, T_P_ab, B_cdP_conj, W_cd_ab);
    }
    comm.all_reduce_in_place_n(W_xcdab.data(), W_xcdab.size(), std::plus<>{});
    W_xcdab() /= nqpts;
    return W_xcdab;
  }

  // dPi_tqPQ[t,q,P,Q] (+)= B[kpq(q),0,P,ab] * Pi_loc[t,ab,cd] * conj(B[q,0,Q,cd]) on the local block of the
  // distributed array, with kpq(q) = q+Q, or q when kpq_map = nullptr. accumulate = false overwrites the block.
  // B_qIPab holds the same q points as the global q axis of dPi_tqPQ (all of them when kpq_map != nullptr).
  // The caller synchronizes the ranks before the call.
  template<typename dArray_t>
  void upfold_pi_block(const nda::MemoryArrayOfRank<5> auto &Pi_loc_tabcd,
                       const nda::ArrayOfRank<5> auto &B_qIPab,
                       dArray_t &dPi_tqPQ, const nda::array<long, 1> *kpq_map, bool accumulate) {
    decltype(nda::range::all) all;

    auto nqpts = B_qIPab.shape()[0];
    auto NP = B_qIPab.shape()[2];
    auto nImpOrbs = B_qIPab.shape()[3];
    auto [nts_half, nqpts_ibz, NP_Pi, NQ_Pi] = dPi_tqPQ.global_shape();
    utils::check(NP_Pi == NP and NQ_Pi == NP and Pi_loc_tabcd.shape(0) == nts_half,
                 "upfold_pi_block: shape mismatch (Pi: nt = {}, NP = {}, NQ = {}; B: NP = {}; Pi_loc: nt = {}).",
                 nts_half, NP_Pi, NQ_Pi, NP, Pi_loc_tabcd.shape(0));
    utils::check(kpq_map == nullptr or (nqpts_ibz == nqpts and kpq_map->shape(0) == nqpts),
                 "upfold_pi_block: a q+Q map requires the full q mesh (nqpts_ibz = {}, nqpts = {}).",
                 nqpts_ibz, nqpts);

    auto [nt_loc, nq_loc, NP_loc, NQ_loc] = dPi_tqPQ.local_shape();
    auto P_rng = dPi_tqPQ.local_range(2);
    auto Q_rng = dPi_tqPQ.local_range(3);
    auto [t_origin, q_origin, P_origin, Q_origin] = dPi_tqPQ.origin();

    // Pi[t,q,P,Q] = B[kpq(q),0,P,ab] * X[t,q,ab,Q]
    auto Pi_imp = nda::reshape(Pi_loc_tabcd, std::array<long, 3>{nts_half, nImpOrbs*nImpOrbs, nImpOrbs*nImpOrbs});
    auto Pi_loc = dPi_tqPQ.local();
    auto B_loc = nda::reshape(B_qIPab, std::array<long, 3>{nqpts, NP, nImpOrbs*nImpOrbs});
    nda::matrix<ComplexType> B_cd_Q_conj(nImpOrbs*nImpOrbs, NQ_loc);
    nda::matrix<ComplexType> X(nImpOrbs*nImpOrbs, NQ_loc);
    for (size_t itq = 0; itq < nt_loc*nq_loc; ++itq) {
      size_t it = itq / nq_loc;
      size_t iq = itq % nq_loc;

      // X[t,q,ab,Q] = Pi_imp[t,ab,cd] * conj(B[q,0,Q,cd])
      B_cd_Q_conj = nda::transpose(nda::conj(B_loc(iq+q_origin,Q_rng,all)));
      math::detail::gemm_guarded(ComplexType(1.0),Pi_imp(it+t_origin,nda::ellipsis{}), B_cd_Q_conj, ComplexType(0.0), X);

      // Pi[t,q,P,Q] (+)= B[kpq(q),0,P,ab] * X[t,q,ab,Q]
      long iq_l = (kpq_map == nullptr)? long(iq+q_origin) : (*kpq_map)(iq+q_origin);
      math::detail::gemm_guarded(ComplexType(1.0), B_loc(iq_l,P_rng,all), X,
                                 accumulate ? ComplexType(1.0) : ComplexType(0.0), Pi_loc(it,iq,nda::ellipsis{}));
    }
  }

} // methods::local_projection

#endif // COQUI_LOCAL_PROJECTION_HPP
