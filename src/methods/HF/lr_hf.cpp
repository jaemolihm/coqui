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


#include "methods/HF/lr_hf.hpp"
#include "methods/HF/lr_thc_comm.hpp"
#include "methods/ERI/thc_reader_t.hpp"

#include "nda/nda.hpp"
#include "nda/blas.hpp"

using thc_reader_t = methods::thc_reader_t;

namespace methods {
namespace solvers {

lr_hf::lr_hf(std::shared_ptr<mpi_context_t> mpi,
             const mf::MF* MF,
             nda::array<double, 1> const& q_vec)
    : _mpi(mpi),
      _MF(MF),
      _ns(MF->nspin()),
      _nkpts(MF->nkpts()),
      _nkpts_ibz(MF->nkpts_ibz()),
      _nbnd(MF->nbnd()),
      _npol(MF->npol()),
      _q_vec(q_vec),
      _kpq_map(_nkpts),
      _Timer() {

  // Compute k+q mapping
  auto kpts_crys = MF->kpts_crystal();
  utils::calculate_kpq_map(kpts_crys, _q_vec, _kpq_map);

  // Check if q is approximately gamma
  _is_q_gamma = utils::is_q_gamma(_q_vec);

  // Find IBZ index of perturbation q (for V(q) lookup)
  _q_ibz_idx = utils::find_q_ibz_index(kpts_crys, _q_vec, MF->qp_to_ibz());

  // Precompute IBZ k → IBZ k+q mapping (for Madelung correction)
  _kpq_ibz_map.resize(_nkpts_ibz);
  _kpq_ibz_trev.resize(_nkpts_ibz);
  auto ks_to_k = MF->ks_to_k(0);     // IBZ → full BZ (identity symmetry)
  auto kp_to_ibz = MF->kp_to_ibz();   // full BZ → IBZ
  auto kp_trev = MF->kp_trev();       // time-reversal flag per full BZ k
  for (int ik = 0; ik < _nkpts_ibz; ++ik) {
    int ik_full = ks_to_k(ik);
    int ikq_full = _kpq_map(ik_full);
    _kpq_ibz_map(ik) = kp_to_ibz(ikq_full);
    _kpq_ibz_trev(ik) = kp_trev(ikq_full);
  }

  app_log(2, "LR-HF initialized:");
  app_log(2, "  - Perturbation wavevector q = ({:.6f}, {:.6f}, {:.6f})",
          _q_vec(0), _q_vec(1), _q_vec(2));
  app_log(2, "  - q is Gamma point: {}", _is_q_gamma ? "yes" : "no");
  app_log(2, "  - q IBZ index: {}", _q_ibz_idx);

  for (auto& v : {"LR_HF", "ALLOC", "PRIM_TO_AUX", "COULOMB", "EXCHANGE", "AUX_TO_PRIM"}) {
    _Timer.add(v);
  }
  _mpi->comm.barrier();
}


template<nda::MemoryArray AF_t>
void lr_hf::evaluate(sArray_t<AF_t>& sDeltaF_skij,
                     const sArray_t<AF_t>& sDeltaDm_skij,
                     THC_ERI auto& thc,
                     const nda::MemoryArrayOfRank<4> auto& S_skij,
                     bool compute_hartree,
                     bool compute_exchange,
                     const lr_ibc_DeltaX* ibc) {
  _Timer.start("LR_HF");
  app_log(2, "Evaluating LR Fock matrix (ΔF from ΔDm):");
  app_log(2, "  - Compute Hartree (ΔJ): {}", compute_hartree ? "yes" : "no");
  app_log(2, "  - Compute Exchange (ΔK): {}", compute_exchange ? "yes" : "no");
  app_log(2, "  - DeltaX correction: {}", ibc ? "yes" : "no");

  thc_lr_hf(sDeltaDm_skij, sDeltaF_skij, thc, compute_hartree, compute_exchange, ibc);

  // Add LR finite-size correction for exchange
  if (compute_exchange) {
    LR_HF_K_correction(sDeltaF_skij, sDeltaDm_skij.local(), S_skij, _MF->madelung());
  }

  _Timer.stop("LR_HF");
  print_timers();
}


template<nda::MemoryArray AF_t>
void lr_hf::thc_lr_hf(const sArray_t<AF_t>& sDeltaDm_skij,
                      sArray_t<AF_t>& sDeltaF_skij,
                      THC_ERI auto& thc,
                      bool compute_hartree,
                      bool compute_exchange,
                      const lr_ibc_DeltaX* ibc) {
  // LR version of hf_t::thc_hf_Xqindep (thc_hf.icc).
  // LR differences: uses lr_thc_comm (left X at k+q, right X at k),
  // V(q) at _q_ibz_idx instead of q=0, and _MF->Qpts() for U FT.
  using local_Array_4D_t = memory::array<HOST_MEMORY, ComplexType, 4>;
  using math::nda::make_distributed_array;

  long NP = thc.Np();
  long ns = sDeltaDm_skij.shape()[0];
  long npol = _MF->npol();
  long nkpts = _MF->nkpts();
  long nkpts_ibz = _MF->nkpts_ibz();
  utils::check(sDeltaDm_skij.shape()[1] == nkpts_ibz, "Shape mismatch: sDeltaDm_skij");
  utils::check(sDeltaF_skij.shape()[1] == nkpts_ibz, "Shape mismatch: sDeltaF_skij");

  sDeltaF_skij.set_zero();
  if (not compute_hartree and not compute_exchange) return;

  // Extract unperturbed Dm for primary→aux DeltaX correction
  const nda::array<ComplexType, 4>* Dm_unpert = ibc ? ibc->Dm_ab : nullptr;

  // Determine processor grid
  int np = _mpi->comm.size();
  int np_P = utils::find_proc_grid_min_diff(np, 1, 1);
  int np_Q = np / np_P;
  nda::array<long, 1> R_grid = _MF->kp_grid();

  _Timer.start("ALLOC");
  math::shm::shared_array<nda::array_view<ComplexType, 2>> sf_Rk(*_mpi, {nkpts, nkpts});
  nda::matrix<ComplexType> buffer;
  _Timer.stop("ALLOC");

  app_log(2, "  LR-HF J/K evaluation:");
  app_log(2, "    - processor grid for ΔDm:  (s, k, P, Q) = ({}, {}, {}, {})\n", 1, 1, np_P, np_Q);

  _Timer.start("ALLOC");
  // ΔDm_skPQ is needed at all k-points in 1st-BZ
  auto dDeltaDm_skPQ = make_distributed_array<local_Array_4D_t>(_mpi->comm, {1, 1, np_P, np_Q},
                                                                 {ns, nkpts, NP, NP});
  auto dDeltaF_skPQ = make_distributed_array<local_Array_4D_t>(_mpi->comm, {1, 1, np_P, np_Q},
                                                                {ns, nkpts_ibz, NP, NP});
  auto NP_loc = dDeltaDm_skPQ.local_shape()[2];
  auto NQ_loc = dDeltaDm_skPQ.local_shape()[3];
  auto P_origin = dDeltaDm_skPQ.origin()[2];
  auto Q_origin = dDeltaDm_skPQ.origin()[3];
  _Timer.stop("ALLOC");

  // Get Coulomb kernel U(q)_PQ; later FT'd in-place to U(R)_PQ for exchange
  auto dU_qPQ = thc.dZ({1, np_P, np_Q});
  auto dU_qPQ_loc = dU_qPQ.local();  // U(q)_PQ before FT, U(R)_PQ after

  // Keep a copy of V(q) for Hartree (V(q=0) when q is Gamma, V(q) otherwise)
  nda::array<ComplexType, 2> Uq_PQ(NP_loc, NQ_loc);
  Uq_PQ() = dU_qPQ_loc(_q_ibz_idx, nda::ellipsis{});

  // Accumulate diagonal indices of ΔDm for the Hartree term
  nda::array<ComplexType, 1> DeltaDm_QQ(NP, ComplexType(0.0));
  std::vector<std::pair<long, long>> diag_idx;
  for (long iP = 0; iP < NP_loc; ++iP) {
    long P = iP + P_origin;
    for (long iQ = 0; iQ < NQ_loc; ++iQ) {
      long Q = iQ + Q_origin;
      if (P == Q) diag_idx.push_back({iP, iQ});
    }
  }

  if (compute_hartree and not compute_exchange) {
    // === HARTREE-ONLY PATH ===

    for (auto ip : nda::range(npol)) {

      _Timer.start("PRIM_TO_AUX");
      lr_thc_comm::primary_to_aux<AF_t>(ip, ip, sDeltaDm_skij.local(), dDeltaDm_skPQ, thc,
                                              _MF->kp_to_ibz(), _MF->kp_trev(), _kpq_map,
                                              ibc, Dm_unpert);
      _Timer.stop("PRIM_TO_AUX");

      _Timer.start("COULOMB");
      // FT ΔDm k→R
      if (nkpts != 1) {
        auto f_Rk = sf_Rk.local();
        utils::k_to_R_coefficients(_mpi->comm, nda::range(nkpts), _MF->kpts(), _MF->lattv(), R_grid, sf_Rk);
        auto DeltaDm_3D = nda::reshape(dDeltaDm_skPQ.local(), shape_t<3>{ns, nkpts, NP_loc * NQ_loc});
        if (buffer.shape() != shape_t<2>{nkpts, NP_loc * NQ_loc})
          buffer.resize(shape_t<2>{nkpts, NP_loc * NQ_loc});
        for (int s = 0; s < ns; ++s) {
          nda::blas::gemm(f_Rk, DeltaDm_3D(s, nda::ellipsis{}), buffer);
          DeltaDm_3D(s, nda::ellipsis{}) = buffer;
        }
      }
      // After FT: dDeltaDm_skPQ is now dDeltaDm_sRPQ
      auto& dDeltaDm_sRPQ = dDeltaDm_skPQ;

      // Accumulate DeltaDm_QQ diagonal
      double factor = (ns == 1 and npol == 1) ? 2.0 : 1.0;
      auto DeltaDm_sRPQ_loc = dDeltaDm_sRPQ.local();
      for (long is = 0; is < ns; ++is) {
        for (auto idx : diag_idx)
          DeltaDm_QQ(idx.first + P_origin) += factor * DeltaDm_sRPQ_loc(is, 0, idx.first, idx.second);
      }
      _Timer.stop("COULOMB");

    } // ip

    _Timer.start("COULOMB");
    // MAM: communication not optimized
    dDeltaDm_skPQ.communicator()->all_reduce_in_place_n(DeltaDm_QQ.data(), DeltaDm_QQ.size(), std::plus<>{});
    nda::array<ComplexType, 1> DeltaJ_PP(NP, ComplexType(0.0));
    nda::blas::gemv(Uq_PQ, DeltaDm_QQ(dU_qPQ.local_range(2)), DeltaJ_PP(dU_qPQ.local_range(1)));
    dDeltaDm_skPQ.communicator()->all_reduce_in_place_n(DeltaJ_PP.data(), DeltaJ_PP.size(), std::plus<>{});

    auto DeltaF_skPQ = dDeltaF_skPQ.local();
    DeltaF_skPQ() = ComplexType(0.0);
    for (long is = 0; is < ns; ++is) {
      for (long ik = 0; ik < nkpts_ibz; ++ik) {
        for (auto idx : diag_idx) {
          DeltaF_skPQ(is, ik, idx.first, idx.second) += DeltaJ_PP(idx.first + P_origin);
        }
      }
    }
    _Timer.stop("COULOMB");

    _Timer.start("AUX_TO_PRIM");
    for (auto ip : nda::range(npol))
      lr_thc_comm::aux_to_primary<AF_t>(ip, ip, ComplexType(1.0),
                                              dDeltaF_skPQ, sDeltaF_skij, thc,
                                              _MF->ks_to_k(0), _kpq_map);
    _Timer.stop("AUX_TO_PRIM");

  } else {
    // === EXCHANGE (+ optional HARTREE) PATH ===

    utils::check(compute_exchange, "lr_hf::thc_lr_hf: entered exchange path but compute_exchange=false");

    // FT U(q) to real space U(R) for exchange — overwrites dU_qPQ_loc in-place
    if (nkpts != 1) {
      buffer.resize(shape_t<2>{nkpts, NP_loc * NQ_loc});
      auto f_Rk = sf_Rk.local();
      utils::k_to_R_coefficients(_mpi->comm, nda::range(nkpts), _MF->Qpts(), _MF->lattv(), R_grid, sf_Rk);
      auto U_2D = nda::reshape(dU_qPQ_loc, shape_t<2>{nkpts, NP_loc * NQ_loc});
      nda::blas::gemm(f_Rk, U_2D, buffer);
      U_2D = buffer;
    }
    // After FT: dU_qPQ_loc is now U(R)_PQ
    auto& U_RPQ_loc = dU_qPQ_loc;

    // aux_to_primary needs to be called for each {ip,iq} block.
    // You can still use the fact that off diagonal blocks are hermitian with respect
    // to each other, so skip lower diagonal, multiply off diagonal by 2.0,
    // symmetrize at the end
    for (auto ip : nda::range(npol)) {
      for (auto iq : nda::range(ip, npol)) {
        dDeltaF_skPQ.local() = ComplexType(0.0);

        _Timer.start("PRIM_TO_AUX");
        lr_thc_comm::primary_to_aux<AF_t>(ip, iq, sDeltaDm_skij.local(), dDeltaDm_skPQ, thc,
                                                _MF->kp_to_ibz(), _MF->kp_trev(), _kpq_map,
                                                ibc, Dm_unpert);
        _Timer.stop("PRIM_TO_AUX");

        // FT ΔDm from k-space to R-space
        _Timer.start("EXCHANGE");
        if (nkpts != 1) {
          auto f_Rk = sf_Rk.local();
          utils::k_to_R_coefficients(_mpi->comm, nda::range(nkpts), _MF->kpts(), _MF->lattv(), R_grid, sf_Rk);
          auto DeltaDm_3D = nda::reshape(dDeltaDm_skPQ.local(), shape_t<3>{ns, nkpts, NP_loc * NQ_loc});
          if (buffer.shape() != shape_t<2>{nkpts, NP_loc * NQ_loc})
            buffer.resize(shape_t<2>{nkpts, NP_loc * NQ_loc});
          for (int s = 0; s < ns; ++s) {
            nda::blas::gemm(f_Rk, DeltaDm_3D(s, nda::ellipsis{}), buffer);
            DeltaDm_3D(s, nda::ellipsis{}) = buffer;
          }
        }
        _Timer.stop("EXCHANGE");
        // After FT: dDeltaDm_skPQ is now dDeltaDm_sRPQ
        auto& dDeltaDm_sRPQ = dDeltaDm_skPQ;

        // Hartree contribution (diagonal blocks only)
        if (compute_hartree and ip == iq) {
          _Timer.start("COULOMB");
          double factor = (ns == 1 and npol == 1) ? 2.0 : 1.0;
          auto DeltaDm_sRPQ_loc = dDeltaDm_sRPQ.local();
          for (long is = 0; is < ns; ++is) {
            for (auto idx : diag_idx)
              DeltaDm_QQ(idx.first + P_origin) += factor * DeltaDm_sRPQ_loc(is, 0, idx.first, idx.second);
          }

          if (npol == 1) {
            // if npol==1, add Coulomb contribution to F_skPQ here
            dDeltaDm_sRPQ.communicator()->all_reduce_in_place_n(DeltaDm_QQ.data(), DeltaDm_QQ.size(), std::plus<>{});
            nda::array<ComplexType, 1> DeltaJ_PP(NP, ComplexType(0.0));
            nda::blas::gemv(Uq_PQ, DeltaDm_QQ(dU_qPQ.local_range(2)), DeltaJ_PP(dU_qPQ.local_range(1)));
            dDeltaDm_sRPQ.communicator()->all_reduce_in_place_n(DeltaJ_PP.data(), DeltaJ_PP.size(), std::plus<>{});

            auto DeltaF_skPQ = dDeltaF_skPQ.local();
            for (long is = 0; is < ns; ++is) {
              for (long ik = 0; ik < nkpts_ibz; ++ik) {
                for (auto idx : diag_idx) {
                  DeltaF_skPQ(is, ik, idx.first, idx.second) += DeltaJ_PP(idx.first + P_origin);
                }
              }
            }
          }
          _Timer.stop("COULOMB");
        }

        // Exchange contribution: ΔK(R) = -ΔDm(R) ⊙ U(R) (Hadamard product in real space)
        _Timer.start("EXCHANGE");
        auto had_prod2 = nda::map([](ComplexType x, ComplexType y) { return -1.0 * (x * y); });
        for (long s = 0; s < ns; ++s) {
          auto DeltaK_RPQ = dDeltaDm_sRPQ.local()(s, nda::ellipsis{});
          DeltaK_RPQ = had_prod2(DeltaK_RPQ, U_RPQ_loc);
        }
        // dDeltaDm_sRPQ now holds ΔK(R)_sRPQ
        auto& dDeltaK_sRPQ = dDeltaDm_sRPQ;

        if (nkpts != 1) {
          // FT ΔK from R-space back to k-space; accumulates onto dDeltaF_skPQ (may contain ΔJ)
          auto f_kR = sf_Rk.local();
          utils::R_to_k_coefficients(_mpi->comm, nda::range(nkpts), _MF->kpts(), _MF->lattv(), R_grid, sf_Rk);
          auto DeltaK_R_3D = nda::reshape(dDeltaK_sRPQ.local(), shape_t<3>{ns, nkpts, NP_loc * NQ_loc});
          auto DeltaF_k_3D = nda::reshape(dDeltaF_skPQ.local(), shape_t<3>{ns, nkpts_ibz, NP_loc * NQ_loc});
          for (int s = 0; s < ns; ++s) {
            nda::blas::gemm(ComplexType(1.0), f_kR(nda::range(nkpts_ibz), nda::range::all),
                            DeltaK_R_3D(s, nda::ellipsis{}), ComplexType(1.0), DeltaF_k_3D(s, nda::ellipsis{}));
          }
        } else {
          // Gamma-point only: R=0 only, so ΔK(R=0) = ΔK(k)
          auto DeltaF_loc = dDeltaF_skPQ.local();
          auto DeltaK_loc = dDeltaK_sRPQ.local();
          DeltaF_loc += DeltaK_loc;
        }
        _Timer.stop("EXCHANGE");

        _Timer.start("AUX_TO_PRIM");
        // Factor 2 for ip /= iq because we only compute (1, 0), not (0, 1).
        // After this, we will symmetrize ΔF_ij and ΔF_ji to enforce Hermiticity.
        lr_thc_comm::aux_to_primary<AF_t>(ip, iq, (ip == iq ? ComplexType(1.0) : ComplexType(2.0)),
                                                dDeltaF_skPQ, sDeltaF_skij, thc,
                                                _MF->ks_to_k(0), _kpq_map);
        _Timer.stop("AUX_TO_PRIM");
      } // iq
    } // ip

    if (npol > 1) {
      // add Coulomb contribution (summed over polarizations) and symmetrize

      if (compute_hartree) {
        // add Coulomb contribution, Dm_QQ has local contribution of diagonal of Dm summed
        // over spin and polarization
        _Timer.start("COULOMB");
        dDeltaDm_skPQ.communicator()->all_reduce_in_place_n(DeltaDm_QQ.data(), DeltaDm_QQ.size(), std::plus<>{});
        nda::array<ComplexType, 1> DeltaJ_PP(NP, ComplexType(0.0));
        nda::blas::gemv(Uq_PQ, DeltaDm_QQ(dU_qPQ.local_range(2)), DeltaJ_PP(dU_qPQ.local_range(1)));
        dDeltaDm_skPQ.communicator()->all_reduce_in_place_n(DeltaJ_PP.data(), DeltaJ_PP.size(), std::plus<>{});

        dDeltaF_skPQ.local() = ComplexType(0.0);
        auto DeltaF_skPQ = dDeltaF_skPQ.local();
        for (long is = 0; is < ns; ++is) {
          for (long ik = 0; ik < nkpts_ibz; ++ik) {
            for (auto idx : diag_idx) {
              DeltaF_skPQ(is, ik, idx.first, idx.second) += DeltaJ_PP(idx.first + P_origin);
            }
          }
        }
        _Timer.stop("COULOMB");

        _Timer.start("AUX_TO_PRIM");
        for (auto ip : nda::range(npol))
          lr_thc_comm::aux_to_primary<AF_t>(ip, ip, ComplexType(1.0),
                                                  dDeltaF_skPQ, sDeltaF_skij, thc,
                                                  _MF->ks_to_k(0), _kpq_map);
        _Timer.stop("AUX_TO_PRIM");
      }

      // Symmetrize: the (ip, iq) loop only processes upper-triangle blocks (iq >= ip)
      // with a factor of 2 for off-diagonal blocks. Enforce Hermiticity:
      // ΔF_ij = (ΔF_ij + ΔF_ji*) / 2, ΔF_ji = ΔF_ij*.
      auto node_comm = sDeltaF_skij.node_comm();
      node_comm->barrier();

      auto DeltaF = sDeltaF_skij.local();
      for (auto isk : nda::range(DeltaF.extent(0) * DeltaF.extent(1))) {
        if (isk % node_comm->size() != node_comm->rank()) continue;
        auto DeltaF_ij = DeltaF(isk / DeltaF.extent(1), isk % DeltaF.extent(1), nda::ellipsis{});
        for (auto i : nda::range(DeltaF.extent(2))) {
          for (auto j : nda::range(i + 1, DeltaF.extent(3))) {
            DeltaF_ij(i, j) += std::conj(DeltaF_ij(j, i));
            DeltaF_ij(i, j) *= ComplexType(0.5);
            DeltaF_ij(j, i) = std::conj(DeltaF_ij(i, j));
          }
        }
      }
      node_comm->barrier();
    } // npol > 1

  }

  // Add precomputed IBC correction for aux→primary (DeltaX terms on unperturbed F_PQ).
  // Added once here — covers all (ip,iq) pairs since V_HF_PQ includes the full sum.
  if (ibc && ibc->DeltaF_ibc_skij.size() > 0) {
    if (_mpi->node_comm.root()) {
      sDeltaF_skij.local() += ibc->DeltaF_ibc_skij;
    }
    sDeltaF_skij.win().fence();
  }

  dU_qPQ.reset();
  dDeltaDm_skPQ.reset();
  _mpi->comm.barrier();
}


/**
 * LR finite-size correction for K based on "PRB 80, 085114(2009)"
 *
 * Computes: ΔDelta_ij(k) = -madelung * S(k+q)_ia * ΔDm_ab(k) * S(k)_bj
 *
 * For q=0: S(k+q) = S(k), recovering the original formula.
 */
template<nda::MemoryArray AF_t>
void lr_hf::LR_HF_K_correction(sArray_t<AF_t>& sDeltaF_skij,
                                const nda::MemoryArrayOfRank<4> auto& DeltaDm_skij,
                                const nda::MemoryArrayOfRank<4> auto& S_skij,
                                double madelung) {
  app_log(2, "  LR finite-size correction (Madelung = {:.6f})", madelung);

  decltype(nda::range::all) all;
  long ns = DeltaDm_skij.extent(0);
  long nkpts_ibz = DeltaDm_skij.extent(1);
  long nbnd = DeltaDm_skij.extent(2);

  auto sDelta_skij = math::shm::make_shared_array<AF_t>(
      *sDeltaF_skij.communicator(), *sDeltaF_skij.internode_comm(), *sDeltaF_skij.node_comm(),
      {ns, nkpts_ibz, nbnd, nbnd});

  int rank = sDelta_skij.communicator()->rank();
  int size = sDelta_skij.communicator()->size();
  nda::matrix<ComplexType> buffer(nbnd, nbnd);
  sDelta_skij.win().fence();
  for (int i = rank; i < ns * nkpts_ibz; i += size) {
    int is = i / nkpts_ibz;
    int ik = i % nkpts_ibz;
    // ΔDelta_ij(k) = -madelung * S(k+q)_ia * ΔDm_ab(k) * S(k)_bj
    auto Delta_sk = sDelta_skij.local()(is, ik, all, all);
    auto DeltaDm_sk = DeltaDm_skij(is, ik, all, all);
    auto S_k = S_skij(is, ik, all, all);               // S at k (right)

    // S at k+q (left): may need complex conjugation if reached via time-reversal
    int ikq_ibz = _kpq_ibz_map(ik);
    auto S_kq_raw = S_skij(is, ikq_ibz, all, all);

    if (_kpq_ibz_trev(ik)) {
      nda::matrix<ComplexType> S_kq = nda::conj(S_kq_raw);
      nda::blas::gemm(ComplexType(-1.0 * madelung), S_kq, DeltaDm_sk, ComplexType(0.0), buffer);
    } else {
      nda::blas::gemm(ComplexType(-1.0 * madelung), S_kq_raw, DeltaDm_sk, ComplexType(0.0), buffer);
    }
    nda::blas::gemm(ComplexType(1.0), buffer, S_k, ComplexType(0.0), Delta_sk);
  }
  sDelta_skij.win().fence();
  sDelta_skij.all_reduce();

  if (sDeltaF_skij.node_comm()->root())
    sDeltaF_skij.local() += sDelta_skij.local();
  sDeltaF_skij.communicator()->barrier();
}


// Template instantiations
using Arr4D = nda::array<ComplexType, 4>;
using Arrv4D = nda::array_view<ComplexType, 4>;

template void lr_hf::evaluate(sArray_t<Arrv4D>&,
                               const sArray_t<Arrv4D>&,
                               thc_reader_t&,
                               Arr4D const&,
                               bool, bool,
                               const lr_ibc_DeltaX*);

template void lr_hf::evaluate(sArray_t<Arrv4D>&,
                               const sArray_t<Arrv4D>&,
                               thc_reader_t&,
                               Arrv4D const&,
                               bool, bool,
                               const lr_ibc_DeltaX*);

template void lr_hf::LR_HF_K_correction(sArray_t<Arrv4D>&,
                                         Arrv4D const&,
                                         Arr4D const&,
                                         double);

template void lr_hf::LR_HF_K_correction(sArray_t<Arrv4D>&,
                                         Arrv4D const&,
                                         Arrv4D const&,
                                         double);

} // namespace solvers
} // namespace methods
