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
#include "methods/HF/thc_solver_comm.hpp"
#include "methods/ERI/thc_reader_t.hpp"

#include "nda/nda.hpp"
#include "nda/blas.hpp"

using thc_reader_t = methods::thc_reader_t;

namespace methods {
namespace solvers {

lr_hf::lr_hf(std::shared_ptr<mpi_context_t> mpi,
             std::shared_ptr<mf::MF> MF,
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

  app_log(2, "LR-HF initialized:");
  app_log(2, "  - Perturbation wavevector q = ({:.6f}, {:.6f}, {:.6f})",
          _q_vec(0), _q_vec(1), _q_vec(2));
  app_log(2, "  - q is Gamma point: {}", _is_q_gamma ? "yes" : "no");

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
                     bool compute_exchange) {
  _Timer.start("LR_HF");
  app_log(2, "Evaluating LR Fock matrix (ΔF from ΔDm):");
  app_log(2, "  - Compute Hartree (ΔJ): {}", compute_hartree ? "yes" : "no");
  app_log(2, "  - Compute Exchange (ΔK): {}", compute_exchange ? "yes" : "no");

  thc_lr_hf(sDeltaDm_skij, sDeltaF_skij, thc, compute_hartree, compute_exchange);

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
                      bool compute_exchange) {
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

  // Get Coulomb kernel (q=0 for Hartree)
  auto dU_qPQ = thc.dZ({1, np_P, np_Q});
  auto dU_qPQ_loc = dU_qPQ.local();

  // Keep a copy of U(q=0) for Hartree
  nda::array<ComplexType, 2> Uq0_PQ(NP_loc, NQ_loc);
  Uq0_PQ() = dU_qPQ_loc(0, nda::ellipsis{});

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
    // Hartree-only computation
    for (auto ip : nda::range(npol)) {
      _Timer.start("PRIM_TO_AUX");
      // ΔDm_skij -> ΔDm_skPQ
      thc_solver_comm::primary_to_aux(ip, ip, sDeltaDm_skij.local(), dDeltaDm_skPQ,
                                      thc, _MF->kp_to_ibz(), _MF->kp_trev());
      _Timer.stop("PRIM_TO_AUX");

      _Timer.start("COULOMB");
      if (nkpts != 1) {
        // Fourier transform from "k" space to "R" space in-place
        auto f_Rk = sf_Rk.local();
        utils::k_to_R_coefficients(_mpi->comm, nda::range(nkpts), _MF->kpts(), _MF->lattv(), R_grid, sf_Rk);
        auto DmR_3D = nda::reshape(dDeltaDm_skPQ.local(), shape_t<3>{ns, nkpts, NP_loc * NQ_loc});
        if (buffer.shape() != shape_t<2>{nkpts, NP_loc * NQ_loc})
          buffer.resize(shape_t<2>{nkpts, NP_loc * NQ_loc});
        for (int s = 0; s < ns; ++s) {
          nda::blas::gemm(f_Rk, DmR_3D(s, nda::ellipsis{}), buffer);
          DmR_3D(s, nda::ellipsis{}) = buffer;
        }
      }

      double factor = (ns == 1 and npol == 1) ? 2.0 : 1.0;
      auto Dm_sRPQ = dDeltaDm_skPQ.local();
      // Extract diagonal of density matrix
      for (long is = 0; is < ns; ++is) {
        for (auto idx : diag_idx)
          DeltaDm_QQ(idx.first + P_origin) += factor * Dm_sRPQ(is, 0, idx.first, idx.second);
      }
      _Timer.stop("COULOMB");
    } // ip

    _Timer.start("COULOMB");
    // MPI reduction for DeltaDm_QQ
    dDeltaDm_skPQ.communicator()->all_reduce_in_place_n(DeltaDm_QQ.data(), DeltaDm_QQ.size(), std::plus<>{});
    // ΔJ_P = U_PQ * ΔDm_QQ
    nda::array<ComplexType, 1> DeltaJ_PP(NP, ComplexType(0.0));
    nda::blas::gemv(Uq0_PQ, DeltaDm_QQ(dU_qPQ.local_range(2)), DeltaJ_PP(dU_qPQ.local_range(1)));
    dDeltaDm_skPQ.communicator()->all_reduce_in_place_n(DeltaJ_PP.data(), DeltaJ_PP.size(), std::plus<>{});

    auto F_skPQ = dDeltaF_skPQ.local();
    F_skPQ() = ComplexType(0.0);
    for (long is = 0; is < ns; ++is) {
      for (long ik = 0; ik < nkpts_ibz; ++ik) {
        for (auto idx : diag_idx) {
          F_skPQ(is, ik, idx.first, idx.second) += DeltaJ_PP(idx.first + P_origin);
        }
      }
    }
    _Timer.stop("COULOMB");
    _Timer.start("AUX_TO_PRIM");
    for (auto ip : nda::range(npol))
      thc_solver_comm::aux_to_primary(ip, ip, ComplexType(1.0), dDeltaF_skPQ, sDeltaF_skij, thc, _MF->ks_to_k(0));
    _Timer.stop("AUX_TO_PRIM");

  } else {
    // Hartree + Exchange computation
    utils::check(compute_exchange, "lr_hf::thc_lr_hf: Expected compute_exchange=true");

    // FT U(q) to real space U(R) for exchange
    if (nkpts != 1) {
      buffer.resize(shape_t<2>{nkpts, NP_loc * NQ_loc});
      auto f_Rk = sf_Rk.local();
      utils::k_to_R_coefficients(_mpi->comm, nda::range(nkpts), _MF->Qpts(), _MF->lattv(), R_grid, sf_Rk);
      auto U_2D = nda::reshape(dU_qPQ_loc, shape_t<2>{nkpts, NP_loc * NQ_loc});
      nda::blas::gemm(f_Rk, U_2D, buffer);
      U_2D = buffer;
    }

    // Process each polarization block
    for (auto ip : nda::range(npol)) {
      for (auto iq : nda::range(ip, npol)) {
        dDeltaF_skPQ.local() = ComplexType(0.0);

        // Transform ΔDm to auxiliary basis
        _Timer.start("PRIM_TO_AUX");
        thc_solver_comm::primary_to_aux(ip, iq, sDeltaDm_skij.local(), dDeltaDm_skPQ,
                                        thc, _MF->kp_to_ibz(), _MF->kp_trev());
        _Timer.stop("PRIM_TO_AUX");

        // FT ΔDm from k-space to R-space
        _Timer.start("EXCHANGE");
        if (nkpts != 1) {
          auto f_Rk = sf_Rk.local();
          utils::k_to_R_coefficients(_mpi->comm, nda::range(nkpts), _MF->kpts(), _MF->lattv(), R_grid, sf_Rk);
          auto DmR_3D = nda::reshape(dDeltaDm_skPQ.local(), shape_t<3>{ns, nkpts, NP_loc * NQ_loc});
          if (buffer.shape() != shape_t<2>{nkpts, NP_loc * NQ_loc})
            buffer.resize(shape_t<2>{nkpts, NP_loc * NQ_loc});
          for (int s = 0; s < ns; ++s) {
            nda::blas::gemm(f_Rk, DmR_3D(s, nda::ellipsis{}), buffer);
            DmR_3D(s, nda::ellipsis{}) = buffer;
          }
        }
        _Timer.stop("EXCHANGE");

        // Hartree contribution (diagonal blocks only)
        if (compute_hartree and ip == iq) {
          _Timer.start("COULOMB");
          double factor = (ns == 1 and npol == 1) ? 2.0 : 1.0;
          auto Dm_sRPQ = dDeltaDm_skPQ.local();
          for (long is = 0; is < ns; ++is) {
            for (auto idx : diag_idx)
              DeltaDm_QQ(idx.first + P_origin) += factor * Dm_sRPQ(is, 0, idx.first, idx.second);
          }

          if (npol == 1) {
            // Add Coulomb contribution to ΔF_skPQ here
            dDeltaDm_skPQ.communicator()->all_reduce_in_place_n(DeltaDm_QQ.data(), DeltaDm_QQ.size(), std::plus<>{});
            nda::array<ComplexType, 1> DeltaJ_PP(NP, ComplexType(0.0));
            nda::blas::gemv(Uq0_PQ, DeltaDm_QQ(dU_qPQ.local_range(2)), DeltaJ_PP(dU_qPQ.local_range(1)));
            dDeltaDm_skPQ.communicator()->all_reduce_in_place_n(DeltaJ_PP.data(), DeltaJ_PP.size(), std::plus<>{});

            auto F_skPQ = dDeltaF_skPQ.local();
            for (long is = 0; is < ns; ++is) {
              for (long ik = 0; ik < nkpts_ibz; ++ik) {
                for (auto idx : diag_idx) {
                  F_skPQ(is, ik, idx.first, idx.second) += DeltaJ_PP(idx.first + P_origin);
                }
              }
            }
          }
          _Timer.stop("COULOMB");
        }

        // Exchange contribution: ΔK = -ΔDm * U (Hadamard product)
        _Timer.start("EXCHANGE");
        auto had_prod2 = nda::map([](ComplexType x, ComplexType y) { return -1.0 * (x * y); });
        for (long s = 0; s < ns; ++s) {
          auto Dm_RPQ = dDeltaDm_skPQ.local()(s, nda::ellipsis{});
          Dm_RPQ = had_prod2(Dm_RPQ, dU_qPQ_loc);
        }

        if (nkpts != 1) {
          // FT from R-space back to k-space
          auto f_kR = sf_Rk.local();
          utils::R_to_k_coefficients(_mpi->comm, nda::range(nkpts), _MF->kpts(), _MF->lattv(), R_grid, sf_Rk);
          auto Dm_3D = nda::reshape(dDeltaDm_skPQ.local(), shape_t<3>{ns, nkpts, NP_loc * NQ_loc});
          auto FR_3D = nda::reshape(dDeltaF_skPQ.local(), shape_t<3>{ns, nkpts_ibz, NP_loc * NQ_loc});
          for (int s = 0; s < ns; ++s) {
            nda::blas::gemm(ComplexType(1.0), f_kR(nda::range(nkpts_ibz), nda::range::all),
                            Dm_3D(s, nda::ellipsis{}), ComplexType(1.0), FR_3D(s, nda::ellipsis{}));
          }
        } else {
          // Gamma-point only
          auto F_loc = dDeltaF_skPQ.local();
          auto buff_loc = dDeltaDm_skPQ.local();
          F_loc += buff_loc;
        }
        _Timer.stop("EXCHANGE");

        _Timer.start("AUX_TO_PRIM");
        thc_solver_comm::aux_to_primary(ip, iq, (ip == iq ? ComplexType(1.0) : ComplexType(2.0)),
                                        dDeltaF_skPQ, sDeltaF_skij, thc, _MF->ks_to_k(0));
        _Timer.stop("AUX_TO_PRIM");
      } // iq
    } // ip

    if (npol > 1) {
      // Add Coulomb contribution (summed over polarizations) and symmetrize
      if (compute_hartree) {
        _Timer.start("COULOMB");
        dDeltaDm_skPQ.communicator()->all_reduce_in_place_n(DeltaDm_QQ.data(), DeltaDm_QQ.size(), std::plus<>{});
        nda::array<ComplexType, 1> DeltaJ_PP(NP, ComplexType(0.0));
        nda::blas::gemv(Uq0_PQ, DeltaDm_QQ(dU_qPQ.local_range(2)), DeltaJ_PP(dU_qPQ.local_range(1)));
        dDeltaDm_skPQ.communicator()->all_reduce_in_place_n(DeltaJ_PP.data(), DeltaJ_PP.size(), std::plus<>{});

        auto F_skPQ = dDeltaF_skPQ.local();
        F_skPQ() = ComplexType(0.0);
        for (long is = 0; is < ns; ++is) {
          for (long ik = 0; ik < nkpts_ibz; ++ik) {
            for (auto idx : diag_idx) {
              F_skPQ(is, ik, idx.first, idx.second) += DeltaJ_PP(idx.first + P_origin);
            }
          }
        }
        _Timer.stop("COULOMB");
        _Timer.start("AUX_TO_PRIM");
        for (auto ip : nda::range(npol))
          thc_solver_comm::aux_to_primary(ip, ip, ComplexType(1.0), dDeltaF_skPQ, sDeltaF_skij, thc, _MF->ks_to_k(0));
        _Timer.stop("AUX_TO_PRIM");
      }

      // Symmetrization for npol > 1
      auto node_comm = sDeltaF_skij.node_comm();
      node_comm->barrier();

      auto F = sDeltaF_skij.local();
      for (auto isk : nda::range(F.extent(0) * F.extent(1))) {
        if (isk % node_comm->size() != node_comm->rank()) continue;
        auto Fij = F(isk / F.extent(1), isk % F.extent(1), nda::ellipsis{});
        for (auto i : nda::range(F.extent(2))) {
          for (auto j : nda::range(i + 1, F.extent(3))) {
            Fij(i, j) += std::conj(Fij(j, i));
            Fij(i, j) *= ComplexType(0.5);
            Fij(j, i) = std::conj(Fij(i, j));
          }
        }
      }
      node_comm->barrier();
    } // npol > 1
  }

  dU_qPQ.reset();
  dDeltaDm_skPQ.reset();
  _mpi->comm.barrier();
}


/**
 * LR finite-size correction for K based on "PRB 80, 085114(2009)"
 *
 * Computes: ΔDelta_ij = -madelung * S_ia * ΔDm_ab * S_bj
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
    // ΔDelta_ij = (-1.0) * madelung * S_ia * ΔDm_ab * S_bj
    auto Delta_sk = sDelta_skij.local()(is, ik, all, all);
    auto DeltaDm_sk = DeltaDm_skij(is, ik, all, all);
    auto S_sk = S_skij(is, ik, all, all);
    nda::blas::gemm(ComplexType(-1.0 * madelung), S_sk, DeltaDm_sk, ComplexType(0.0), buffer);
    nda::blas::gemm(ComplexType(1.0), buffer, S_sk, ComplexType(0.0), Delta_sk);
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
                               bool, bool);

template void lr_hf::LR_HF_K_correction(sArray_t<Arrv4D>&,
                                         Arrv4D const&,
                                         Arr4D const&,
                                         double);

} // namespace solvers
} // namespace methods
