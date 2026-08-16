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


#include "methods/HF/lr_hf.hpp"
#include "methods/HF/lr_thc_comm.hpp"
#include "methods/HF/thc_solver_comm.hpp"
#include "methods/ERI/thc_reader_t.hpp"

#include "nda/nda.hpp"
#include "nda/blas.hpp"

using thc_reader_t = methods::thc_reader_t;

namespace methods {
namespace solvers {

lr_hf::lr_hf(std::shared_ptr<mpi_context_t> mpi,
             const mf::MF* MF,
             nda::array<double, 1> const& q_vec,
             std::string hf_div_treatment)
    : _mpi(mpi),
      _MF(MF),
      _ns(MF->nspin()),
      _nkpts(MF->nkpts()),
      _nkpts_ibz(MF->nkpts_ibz()),
      _nbnd(MF->nbnd()),
      _npol(MF->npol()),
      _q_vec(q_vec),
      _kpq_map(_nkpts),
      _hf_div_treatment(std::move(hf_div_treatment)),
      _Timer() {

  // hf_t coerces because it takes user input; here the value is read back from the
  // checkpoint, already normalized by hf_t. Anything else means the stash was not
  // written by hf_t, and silently defaulting would make ΔF and F disagree on the
  // divergence correction.
  utils::check(_hf_div_treatment == "ignore_g0" or _hf_div_treatment == "gygi",
               "lr_hf: hf_div_treatment must be \"ignore_g0\" or \"gygi\", got \"{}\".",
               _hf_div_treatment);

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

  for (auto& v : {"LR_HF", "ALLOC", "PRIM_TO_AUX", "COULOMB", "EXCHANGE", "AUX_TO_PRIM",
                  "Z_FETCH", "UQ_TO_UR", "MADELUNG", "MISC"}) {
    _Timer.add(v);
  }
  _mpi->comm.barrier();
}


void lr_hf::build_Uq_PQ(THC_ERI auto& thc, int np_P, int np_Q,
                        nda::range P_rng, nda::range Q_rng, bool compute_xc,
                        nda::array<ComplexType, 2>& Uq_PQ) {
  if (not _Vq_cached) {
    // One (P,Q) tile at the perturbation q, instead of the whole (nq, NP, NP)
    // array redistributed on every call.
    _Timer.start("Z_FETCH");
    _Vq_PQ = thc.Z(_q_ibz_idx, P_rng, Q_rng, 0, 1, _mpi->comm);
    _Timer.stop("Z_FETCH");
    _Vq_cached = true;
  }
  utils::check(_Vq_PQ.extent(0) == P_rng.size() and _Vq_PQ.extent(1) == Q_rng.size(),
               "lr_hf: cached V(q) block is ({},{}), expected ({},{}).",
               _Vq_PQ.extent(0), _Vq_PQ.extent(1), P_rng.size(), Q_rng.size());

  if (Uq_PQ.shape() != _Vq_PQ.shape()) Uq_PQ.resize(_Vq_PQ.shape());
  Uq_PQ() = _Vq_PQ();

  // LR-DFT: the direct channel uses (V + Vxc)(q). Vxc goes into this private copy,
  // read only by the ΔJ gemv — the exchange path works off U(R)_PQ, so the xc
  // kernel structurally cannot reach ΔK. It carries no spin factor of its own: the
  // (ns==1) factor 2 multiplies the density, and QE's dmuxc at nspin_mag=1 is
  // dV_xc/dρ_total, the same structure as v, so it inherits that factor exactly as
  // the Coulomb term does.
  if (compute_xc) {
    if (not _Vxc_q_cached) {
      utils::check(thc.has_Vxc(),
                   "lr_hf: include_xc = true but the THC integrals carry no xc-kernel "
                   "matrix. Rebuild the THC with 'Vxc_file' set (and delete any stale "
                   "THC checkpoint), or set include_xc = false.");
      _Timer.start("Z_FETCH");
      auto dVxc_qPQ = thc.dVxc({1, np_P, np_Q});
      _Timer.stop("Z_FETCH");
      utils::check(dVxc_qPQ.local_range(1).first() == P_rng.first() and
                   dVxc_qPQ.local_range(1).size() == P_rng.size() and
                   dVxc_qPQ.local_range(2).first() == Q_rng.first() and
                   dVxc_qPQ.local_range(2).size() == Q_rng.size(),
                   "lr_hf: Vxc and Coulomb distributions differ.");
      _Vxc_q_PQ = dVxc_qPQ.local()(_q_ibz_idx, nda::ellipsis{});
      _Vxc_q_cached = true;
    }
    Uq_PQ() += _Vxc_q_PQ();
    app_log(2, "  LR-DFT: direct channel uses V(q) + Vxc(q) at q index {}.",
            _q_ibz_idx);
  }
}


template<nda::MemoryArray AF_t>
void lr_hf::evaluate(sArray_t<AF_t>& sDeltaF_skij,
                     const sArray_t<AF_t>& sDeltaDm_skij,
                     THC_ERI auto& thc,
                     const nda::MemoryArrayOfRank<4> auto& S_skij,
                     bool compute_hartree,
                     bool compute_exchange,
                     const lr_ibc_DeltaX* ibc,
                     const nda::array_view<ComplexType, 3>* DeltaV_qPQ,
                     const nda::array<ComplexType, 4>* Dm_skij_unpert,
                     nda::array<ComplexType, 4>* DeltaF_PQ_out,
                     bool compute_xc) {
  _Timer.start("LR_HF");
  app_log(3, "Evaluating LR Fock matrix (ΔF from ΔDm):");
  app_log(3, "  - Compute Hartree (ΔJ): {}", compute_hartree ? "yes" : "no");
  app_log(3, "  - Compute Exchange (ΔK): {}", compute_exchange ? "yes" : "no");
  app_log(3, "  - Semilocal xc kernel in ΔJ: {}", compute_xc ? "yes" : "no");
  app_log(3, "  - DeltaX correction: {}", ibc ? "yes" : "no");
  app_log(3, "  - DeltaV correction: {}", DeltaV_qPQ ? "yes" : "no");

  // If DeltaV is provided, the unperturbed Dm is required to contract against it.
  // Prefer an explicit Dm_skij_unpert; fall back to ibc->Dm_ab if available.
  const nda::array<ComplexType, 4>* Dm_unpert_for_dV = Dm_skij_unpert;
  if (!Dm_unpert_for_dV && ibc) Dm_unpert_for_dV = ibc->Dm_ab;
  utils::check(!DeltaV_qPQ || Dm_unpert_for_dV,
               "lr_hf::evaluate: DeltaV_qPQ provided but no unperturbed Dm "
               "(pass Dm_skij_unpert or an lr_ibc_DeltaX with Dm_ab set).");

  // The semilocal xc kernel only exists for the diagonal (direct) density
  // response, so it may only be requested for a Hartree-type LR. Refusing the
  // combination here rather than downstream keeps v + f_xc + Fock from being
  // requestable at all.
  utils::check(!compute_xc || compute_hartree,
               "lr_hf::evaluate: compute_xc requires compute_hartree.");
  utils::check(!compute_xc || !compute_exchange,
               "lr_hf::evaluate: compute_xc is incompatible with compute_exchange. "
               "The semilocal xc kernel contracts with the diagonal density "
               "response only; v + f_xc + Fock is not a defined theory here. "
               "LR-DFT is include_hartree = true, include_exchange = false.");

  thc_lr_hf(sDeltaDm_skij, sDeltaF_skij, thc, compute_hartree, compute_exchange,
            ibc, DeltaV_qPQ, Dm_unpert_for_dV, DeltaF_PQ_out, compute_xc);

  // Add LR finite-size correction for exchange
  if (compute_exchange) {
    _Timer.start("MADELUNG");
    LR_HF_K_correction(sDeltaF_skij, sDeltaDm_skij.local(), S_skij, _MF->madelung());
    _Timer.stop("MADELUNG");
  }

  _Timer.stop("LR_HF");
  print_timers(3);  // per-step diagnostics only at verbosity >= 3
}


template<nda::MemoryArray AF_t>
void lr_hf::thc_lr_hf(const sArray_t<AF_t>& sDeltaDm_skij,
                      sArray_t<AF_t>& sDeltaF_skij,
                      THC_ERI auto& thc,
                      bool compute_hartree,
                      bool compute_exchange,
                      const lr_ibc_DeltaX* ibc,
                      const nda::array_view<ComplexType, 3>* DeltaV_qPQ,
                      const nda::array<ComplexType, 4>* Dm_skij_unpert,
                      nda::array<ComplexType, 4>* DeltaF_PQ_out,
                      bool compute_xc) {
  // LR version of hf_t::thc_hf_Xqindep (thc_hf.icc).
  // LR differences: uses lr_thc_comm (left X at k+q, right X at k),
  // V(q) at _q_ibz_idx instead of q=0, and _MF->Qpts() for U FT.
  //
  // δV correction (when DeltaV_qPQ is provided):
  //   - Hartree (δJ): δV at Coulomb q=Γ contracts with Dm_unpert at R=0.
  //   - Fock (δK): R-space Hadamard δV(R) ⊙ Dm_unpert(R).
  using local_Array_4D_t = memory::array<HOST_MEMORY, ComplexType, 4>;
  using math::nda::make_distributed_array;

  long NP = thc.Np();
  long ns = sDeltaDm_skij.shape()[0];
  long npol = _MF->npol();
  long nkpts = _MF->nkpts();
  long nkpts_ibz = _MF->nkpts_ibz();
  utils::check(sDeltaDm_skij.shape()[1] == nkpts_ibz, "Shape mismatch: sDeltaDm_skij");
  utils::check(sDeltaF_skij.shape()[1] == nkpts_ibz, "Shape mismatch: sDeltaF_skij");

  _Timer.start("MISC");
  sDeltaF_skij.set_zero();
  _Timer.stop("MISC");
  if (not compute_hartree and not compute_exchange) return;

  // Hartree-only fast path. Dispatched here rather than at the Hartree branch below
  // so that none of the dense path's (P,Q)-distributed arrays or FT buffers are
  // allocated at all. IBC and δV fall through to the dense branch.
  if (compute_hartree and not compute_exchange and
      ibc == nullptr and DeltaV_qPQ == nullptr) {
    thc_lr_hartree_only(sDeltaDm_skij, sDeltaF_skij, thc, DeltaF_PQ_out, compute_xc);
    return;
  }

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

  app_log(3, "  LR-HF J/K evaluation:");
  app_log(3, "    - processor grid for ΔDm:  (s, k, P, Q) = ({}, {}, {}, {})\n", 1, 1, np_P, np_Q);

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

  // Direct-channel kernel V(q) (+ Vxc(q)) on this rank's tile, cached across calls
  nda::range P_rng(P_origin, P_origin + NP_loc);
  nda::range Q_rng(Q_origin, Q_origin + NQ_loc);
  nda::array<ComplexType, 2> Uq_PQ;
  build_Uq_PQ(thc, np_P, np_Q, P_rng, Q_rng, compute_xc, Uq_PQ);

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

  // --- DeltaV correction setup ---
  // When DeltaV_qPQ is supplied, also distribute it (same pgrid as dU_qPQ), FT
  // q→R, and precompute the unperturbed Dm in aux+R space. These feed the
  // new δV·Dm terms in both the Hartree (δJ) and Fock (δK) contractions.
  //
  // δJ_μν += 2 δ_μν Σ_{μ'} δV^{q=Γ}_{μμ'} · Dm^{R=0}_{μ'μ'}
  // δK^k_μν += -Σ_R e^{ikR} δV^R_{μν} · Dm^R_{μν}
  //
  // npol > 1 with DeltaV_qPQ is not yet supported.
  utils::check(!DeltaV_qPQ || npol == 1,
               "lr_hf: DeltaV_qPQ correction not yet implemented for npol > 1.");
  utils::check(!DeltaV_qPQ || Dm_skij_unpert,
               "lr_hf::thc_lr_hf: DeltaV_qPQ provided but Dm_skij_unpert is null.");
  if (DeltaV_qPQ) {
    utils::check(DeltaV_qPQ->shape(0) == nkpts && DeltaV_qPQ->shape(1) == NP &&
                 DeltaV_qPQ->shape(2) == NP,
                 "lr_hf: DeltaV_qPQ shape mismatch: expected ({},{},{}), got ({},{},{})",
                 nkpts, NP, NP,
                 DeltaV_qPQ->shape(0), DeltaV_qPQ->shape(1), DeltaV_qPQ->shape(2));
  }

  // Distributed δV in R-space and unperturbed Dm in aux+R space (only allocated if needed)
  std::optional<memory::darray_t<local_Array_4D_t, mpi3::communicator>> opt_dDm_sRPQ_unpert;
  // DeltaV is replicated on all ranks; we only need the local (P, Q) slab to
  // match dU_qPQ's distribution. Fill a local buffer of shape (nkpts, NP_loc, NQ_loc).
  nda::array<ComplexType, 3> DeltaV_RPQ_loc;  // (nkpts, NP_loc, NQ_loc) after q→R FT
  nda::array<ComplexType, 2> DeltaV_qGamma_PQ;  // (NP_loc, NQ_loc) δV^{q=Γ}_PQ

  // Unperturbed Dm_QQ diagonal at R=0 (global NP)
  nda::array<ComplexType, 1> Dm_QQ_unpert(NP, ComplexType(0.0));

  if (DeltaV_qPQ) {
    _Timer.start("ALLOC");
    // (1) Distribute DeltaV locally: copy the (nkpts, P, Q) slab this rank owns.
    DeltaV_RPQ_loc.resize(nkpts, NP_loc, NQ_loc);
    for (long iq = 0; iq < nkpts; ++iq) {
      for (long iP = 0; iP < NP_loc; ++iP) {
        for (long iQ = 0; iQ < NQ_loc; ++iQ) {
          DeltaV_RPQ_loc(iq, iP, iQ) = (*DeltaV_qPQ)(iq, iP + P_origin, iQ + Q_origin);
        }
      }
    }

    // (1b) Load δV^{q=Γ}_PQ for the δV correction to Hartree term
    nda::array<double, 1> q_gamma_vec({0.0, 0.0, 0.0});
    long q_gamma_full_idx = utils::find_q_full_index(_MF->Qpts(), q_gamma_vec);
    DeltaV_qGamma_PQ.resize(NP_loc, NQ_loc);
    DeltaV_qGamma_PQ() = DeltaV_RPQ_loc(q_gamma_full_idx, nda::ellipsis{});

    // (2) FT DeltaV q→R using the same convention as dU_qPQ in the exchange
    //     path (MF->Qpts() grid).  DeltaV_qPQ is full-BZ q-indexed.
    if (nkpts != 1) {
      auto f_Rk = sf_Rk.local();
      utils::k_to_R_coefficients(_mpi->comm, nda::range(nkpts), _MF->Qpts(),
                                 _MF->lattv(), R_grid, sf_Rk);
      if (buffer.shape() != shape_t<2>{nkpts, NP_loc * NQ_loc})
        buffer.resize(shape_t<2>{nkpts, NP_loc * NQ_loc});
      auto V_2D = nda::reshape(DeltaV_RPQ_loc, shape_t<2>{nkpts, NP_loc * NQ_loc});
      nda::blas::gemm(f_Rk, V_2D, buffer);
      V_2D = buffer;
    }
    // DeltaV_RPQ_loc now holds δV^R_PQ (used by the Exchange path).

    // (3) Forward-transform unperturbed Dm to aux (q=0), then FT k→R.
    //     Reuses thc_solver_comm::primary_to_aux (no DeltaX correction — we
    //     want the plain equilibrium Dm_PQ).
    opt_dDm_sRPQ_unpert.emplace(make_distributed_array<local_Array_4D_t>(
        _mpi->comm, {1, 1, np_P, np_Q}, {ns, nkpts, NP, NP}));
    auto& dDm_sRPQ_unpert = *opt_dDm_sRPQ_unpert;

    solvers::thc_solver_comm::primary_to_aux(0, 0, *Dm_skij_unpert, dDm_sRPQ_unpert,
                                             thc, _MF->kp_to_ibz(), _MF->kp_trev());
    if (nkpts != 1) {
      auto f_Rk = sf_Rk.local();
      utils::k_to_R_coefficients(_mpi->comm, nda::range(nkpts), _MF->kpts(),
                                 _MF->lattv(), R_grid, sf_Rk);
      if (buffer.shape() != shape_t<2>{nkpts, NP_loc * NQ_loc})
        buffer.resize(shape_t<2>{nkpts, NP_loc * NQ_loc});
      auto Dm_3D = nda::reshape(dDm_sRPQ_unpert.local(), shape_t<3>{ns, nkpts, NP_loc * NQ_loc});
      for (int s = 0; s < ns; ++s) {
        nda::blas::gemm(f_Rk, Dm_3D(s, nda::ellipsis{}), buffer);
        Dm_3D(s, nda::ellipsis{}) = buffer;
      }
    }

    // (4) Diagonal of Dm_unpert at R=0, with the same spin factor as DeltaDm_QQ
    double factor = (ns == 1 and npol == 1) ? 2.0 : 1.0;
    auto Dm_unpert_sRPQ_loc = dDm_sRPQ_unpert.local();
    for (long is = 0; is < ns; ++is) {
      for (auto idx : diag_idx)
        Dm_QQ_unpert(idx.first + P_origin) +=
            factor * Dm_unpert_sRPQ_loc(is, 0, idx.first, idx.second);
    }
    _mpi->comm.all_reduce_in_place_n(Dm_QQ_unpert.data(), Dm_QQ_unpert.size(), std::plus<>{});
    _Timer.stop("ALLOC");
  }
  // --- end DeltaV correction setup ---

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
        utils::k_to_R_coefficients(_mpi->comm, nda::range(nkpts), _MF->kpts(),
                                   _MF->lattv(), R_grid, sf_Rk);
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
    nda::blas::gemv(Uq_PQ, DeltaDm_QQ(Q_rng), DeltaJ_PP(P_rng));
    // DeltaV correction: DeltaJ_PP += δV^{q=Γ}_PQ · Dm_QQ
    if (DeltaV_qPQ) {
      nda::blas::gemv(ComplexType(1.0), DeltaV_qGamma_PQ,
                      Dm_QQ_unpert(Q_rng),
                      ComplexType(1.0), DeltaJ_PP(P_rng));
    }
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
    // One reduction over the polarization blocks, which accumulated node-locally.
    sDeltaF_skij.all_reduce_parallel();
    _Timer.stop("AUX_TO_PRIM");

  } else {
    // === EXCHANGE (+ optional HARTREE) PATH ===

    utils::check(compute_exchange, "lr_hf::thc_lr_hf: entered exchange path but compute_exchange=false");

    // U(R)_PQ for exchange: fetch U(q) and FT it q→R once, then keep it. Neither
    // the kernel nor the transform depends on ΔDm, so both are pure setup.
    if (not _U_RPQ_cached) {
      _Timer.start("Z_FETCH");
      auto dU_qPQ = thc.dZ({1, np_P, np_Q});
      _Timer.stop("Z_FETCH");
      auto dU_qPQ_loc = dU_qPQ.local();
      utils::check(dU_qPQ.local_range(1).first() == P_rng.first() and
                   dU_qPQ.local_range(1).size() == P_rng.size() and
                   dU_qPQ.local_range(2).first() == Q_rng.first() and
                   dU_qPQ.local_range(2).size() == Q_rng.size(),
                   "lr_hf: the Coulomb (P,Q) tiling differs from ΔDm's.");
      utils::check(dU_qPQ_loc.extent(0) == nkpts,
                   "lr_hf: the exchange q→R transform needs one Coulomb q per k-point "
                   "(got {} q, {} k).", dU_qPQ_loc.extent(0), nkpts);

      _Timer.start("UQ_TO_UR");
      if (nkpts != 1) {
        buffer.resize(shape_t<2>{nkpts, NP_loc * NQ_loc});
        auto f_Rk = sf_Rk.local();
        utils::k_to_R_coefficients(_mpi->comm, nda::range(nkpts), _MF->Qpts(),
                                   _MF->lattv(), R_grid, sf_Rk);
        auto U_2D = nda::reshape(dU_qPQ_loc, shape_t<2>{nkpts, NP_loc * NQ_loc});
        nda::blas::gemm(f_Rk, U_2D, buffer);
        U_2D = buffer;
      }
      _U_RPQ = dU_qPQ_loc;
      _U_RPQ_cached = true;
      dU_qPQ.reset();
      _Timer.stop("UQ_TO_UR");
    }
    auto U_RPQ_loc = _U_RPQ();

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
          utils::k_to_R_coefficients(_mpi->comm, nda::range(nkpts), _MF->kpts(),
                                     _MF->lattv(), R_grid, sf_Rk);
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
            nda::blas::gemv(Uq_PQ, DeltaDm_QQ(Q_rng), DeltaJ_PP(P_rng));
            // DeltaV correction: DeltaJ_PP += δV^{q=Γ}_PQ · Dm_QQ_unpert
            if (DeltaV_qPQ) {
              nda::blas::gemv(ComplexType(1.0), DeltaV_qGamma_PQ,
                              Dm_QQ_unpert(Q_rng),
                              ComplexType(1.0), DeltaJ_PP(P_rng));
            }
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
        // DeltaV correction: ΔK(R) += -Dm_unpert(R) ⊙ δV(R)
        // Only contributes on diagonal polarization blocks (ip == iq).
        if (DeltaV_qPQ && ip == iq) {
          auto Dm_unpert_sRPQ_loc = opt_dDm_sRPQ_unpert->local();
          for (long s = 0; s < ns; ++s) {
            auto DeltaK_RPQ = dDeltaDm_sRPQ.local()(s, nda::ellipsis{});
            auto Dm_unpert_RPQ = Dm_unpert_sRPQ_loc(s, nda::ellipsis{});
            DeltaK_RPQ = DeltaK_RPQ + had_prod2(Dm_unpert_RPQ, DeltaV_RPQ_loc);
          }
        }
        // dDeltaDm_sRPQ now holds ΔK(R)_sRPQ
        auto& dDeltaK_sRPQ = dDeltaDm_sRPQ;

        if (nkpts != 1) {
          // FT ΔK from R-space back to k-space; accumulates onto dDeltaF_skPQ (may contain ΔJ)
          auto f_kR = sf_Rk.local();
          utils::R_to_k_coefficients(_mpi->comm, nda::range(nkpts), _MF->kpts(),
                                     _MF->lattv(), R_grid, sf_Rk);
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
      // add Coulomb contribution (summed over polarizations)
      if (compute_hartree) {
        // add Coulomb contribution, Dm_QQ has local contribution of diagonal of Dm summed
        // over spin and polarization
        _Timer.start("COULOMB");
        dDeltaDm_skPQ.communicator()->all_reduce_in_place_n(DeltaDm_QQ.data(), DeltaDm_QQ.size(), std::plus<>{});
        nda::array<ComplexType, 1> DeltaJ_PP(NP, ComplexType(0.0));
        nda::blas::gemv(Uq_PQ, DeltaDm_QQ(Q_rng), DeltaJ_PP(P_rng));
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
    }

    // Every (ip, iq) block above, and the npol > 1 Hartree block, accumulated into
    // sDeltaF_skij node-locally. Reduce once here, before the symmetrization reads
    // the transposed element and before the caller sees ΔF.
    _Timer.start("AUX_TO_PRIM");
    sDeltaF_skij.all_reduce_parallel();
    _Timer.stop("AUX_TO_PRIM");

    if (npol > 1) {
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

  _Timer.start("MISC");
  // Root-only ΔF_PQ for the Python phonon post-processors, gathered rather than
  // replicated: the array is (ns, nkpts_ibz, NP, NP), ~14 GB per rank at NP ~ 3.6k.
  // Captured before the band-basis IBC correction is added to sDeltaF_skij, so this
  // is the pure aux-basis LR Fock (it likewise excludes the Madelung K correction,
  // which lr_hf::evaluate applies in band basis after this routine returns).
  if (DeltaF_PQ_out) {
    // dDeltaF_skPQ is re-zeroed per polarization block, so for npol > 1 it holds only
    // the last block at this point, not the full ΔF.
    utils::check(npol == 1,
                 "lr_hf: the aux-basis ΔF_PQ output is implemented for npol = 1 only "
                 "(npol = {}). Set output_aux_fock = false.", npol);
    if (_mpi->comm.rank() == 0)
      *DeltaF_PQ_out = nda::array<ComplexType, 4>(dDeltaF_skPQ.global_shape());
    math::nda::gather(0, dDeltaF_skPQ, DeltaF_PQ_out);
  }

  // Add precomputed IBC correction for aux→primary (DeltaX terms on unperturbed F_PQ).
  // Added once here — covers all (ip,iq) pairs since V_HF_PQ includes the full sum.
  if (ibc && ibc->DeltaF_ibc_skij.size() > 0) {
    if (_mpi->node_comm.root()) {
      sDeltaF_skij.local() += ibc->DeltaF_ibc_skij;
    }
    sDeltaF_skij.win().fence();
  }

  dDeltaDm_skPQ.reset();
  _mpi->comm.barrier();
  _Timer.stop("MISC");
}


template<nda::MemoryArray AF_t>
void lr_hf::thc_lr_hartree_only(const sArray_t<AF_t>& sDeltaDm_skij,
                                sArray_t<AF_t>& sDeltaF_skij,
                                THC_ERI auto& thc,
                                nda::array<ComplexType, 4>* DeltaF_PQ_out,
                                bool compute_xc) {
  long NP = thc.Np();
  long ns = sDeltaDm_skij.shape()[0];
  long npol = _npol;
  long nkpts = _MF->nkpts();
  long nkpts_ibz = _MF->nkpts_ibz();

  int np = _mpi->comm.size();
  int np_P = utils::find_proc_grid_min_diff(np, 1, 1);
  int np_Q = np / np_P;

  app_log(3, "  LR-HF J evaluation (diagonal Hartree kernel):");
  app_log(3, "    - processor grid for V(q): (P, Q) = ({}, {})\n", np_P, np_Q);

  // Direct-channel kernel on the same (P,Q) tiling the dense path uses. Only the
  // one q block the Hartree channel contracts is ever fetched. The tile comes from
  // make_distributed_array rather than a hand-rolled block rule, so it cannot drift
  // from the distribution the dense path builds.
  using local_Array_2D_t = memory::array<HOST_MEMORY, ComplexType, 2>;
  auto dPQ = math::nda::make_distributed_array<local_Array_2D_t>(
      _mpi->comm, {np_P, np_Q}, {NP, NP});
  nda::range P_rng(dPQ.origin()[0], dPQ.origin()[0] + dPQ.local_shape()[0]);
  nda::range Q_rng(dPQ.origin()[1], dPQ.origin()[1] + dPQ.local_shape()[1]);
  nda::array<ComplexType, 2> Uq_PQ;
  build_Uq_PQ(thc, np_P, np_Q, P_rng, Q_rng, compute_xc, Uq_PQ);

  // (1) aux-basis density diagonal, averaged over the full BZ and summed over the
  //     diagonal polarization blocks — the only ones the direct channel sees.
  //     n_P = (factor/nk) Σ_(s,k,p) diag_P[ X_p(k+q) ΔDm(k) X_p(k)† ]
  _Timer.start("PRIM_TO_AUX");
  nda::array<ComplexType, 1> DeltaDm_QQ(NP, ComplexType(0.0));
  // Strided (s,k) split: with more ranks than blocks the surplus ranks get an empty
  // range, hence the clamp of the first index.
  long nsk = ns * nkpts;
  nda::range sk_rng(std::min<long>(_mpi->comm.rank(), nsk), nsk, np);
  for (auto ip : nda::range(npol))
    lr_thc_comm::primary_to_aux_diagonal(ip, ip, sDeltaDm_skij.local(), sk_rng,
                                         nda::range(0, NP), DeltaDm_QQ, thc,
                                         _MF->kp_to_ibz(), _MF->kp_trev(), _kpq_map);
  _Timer.stop("PRIM_TO_AUX");

  _Timer.start("COULOMB");
  _mpi->comm.all_reduce_in_place_n(DeltaDm_QQ.data(), DeltaDm_QQ.size(), std::plus<>{});
  double factor = (ns == 1 and npol == 1) ? 2.0 : 1.0;
  DeltaDm_QQ *= ComplexType(factor / double(nkpts));

  // (2) ΔJ_P = Σ_Q [V(q) + Vxc(q)]_PQ n_Q
  nda::array<ComplexType, 1> DeltaJ_PP(NP, ComplexType(0.0));
  nda::blas::gemv(Uq_PQ, DeltaDm_QQ(Q_rng), DeltaJ_PP(P_rng));
  _mpi->comm.all_reduce_in_place_n(DeltaJ_PP.data(), DeltaJ_PP.size(), std::plus<>{});
  _Timer.stop("COULOMB");

  // (3) ΔF_ij(k) = Σ_(p,P) conj(X_p(k+q)_Pi) ΔJ_P X_p(k)_Pj, one (s,k) block per rank.
  //     Each block is written by exactly one rank — the polarization blocks accumulate
  //     on the same one — so the closing reduction sums a value with zeros and
  //     all_reduce_parallel is bit-identical to all_reduce.
  _Timer.start("AUX_TO_PRIM");
  auto& F_comm = *sDeltaF_skij.communicator();
  sDeltaF_skij.win().fence();
  auto DeltaF_loc = sDeltaF_skij.local();
  long nsk_ibz = ns * nkpts_ibz;
  nda::range sk_ibz_rng(std::min<long>(F_comm.rank(), nsk_ibz), nsk_ibz, F_comm.size());
  for (auto ip : nda::range(npol))
    lr_thc_comm::aux_to_primary_diagonal(ip, ip, ComplexType(1.0), DeltaJ_PP, DeltaF_loc,
                                         sk_ibz_rng, thc, _MF->ks_to_k(0), _kpq_map);
  sDeltaF_skij.win().fence();
  sDeltaF_skij.all_reduce_parallel();
  _Timer.stop("AUX_TO_PRIM");

  // Root-only ΔF_PQ for the Python phonon post-processors. Built directly from the
  // diagonal instead of gathering a distributed (ns, nk_ibz, NP, NP) array. Restricted
  // to npol = 1 so that output_aux_fock has one contract on both branches, even though
  // ΔJ_P is polarization-independent and would be complete here for any npol.
  _Timer.start("MISC");
  if (DeltaF_PQ_out) {
    utils::check(npol == 1,
                 "lr_hf: the aux-basis ΔF_PQ output is implemented for npol = 1 only "
                 "(npol = {}). Set output_aux_fock = false.", npol);
  }
  if (DeltaF_PQ_out and _mpi->comm.rank() == 0) {
    *DeltaF_PQ_out = nda::array<ComplexType, 4>({ns, nkpts_ibz, NP, NP});
    (*DeltaF_PQ_out)() = ComplexType(0.0);
    for (long is = 0; is < ns; ++is)
      for (long ik = 0; ik < nkpts_ibz; ++ik)
        for (long P = 0; P < NP; ++P)
          (*DeltaF_PQ_out)(is, ik, P, P) = DeltaJ_PP(P);
  }
  _mpi->comm.barrier();
  _Timer.stop("MISC");
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
  if (_hf_div_treatment == "ignore_g0") {
    app_log(3, "  No finite-size correction to the non-local HF exchange potential.");
    return;
  }
  app_log(3, "  LR finite-size correction (Madelung = {:.6f})", madelung);

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
  // The strided loop assigns each (is,ik) block with a beta = 0 gemm on exactly one
  // rank, and the array is zero-initialized, so every element is contributed by one
  // node: splitting the reduction across the node's ranks is bit-identical here.
  sDelta_skij.all_reduce_parallel();

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
                               const lr_ibc_DeltaX*,
                               const nda::array_view<ComplexType, 3>*,
                               const nda::array<ComplexType, 4>*,
                               nda::array<ComplexType, 4>*,
                               bool);

template void lr_hf::evaluate(sArray_t<Arrv4D>&,
                               const sArray_t<Arrv4D>&,
                               thc_reader_t&,
                               Arrv4D const&,
                               bool, bool,
                               const lr_ibc_DeltaX*,
                               const nda::array_view<ComplexType, 3>*,
                               const nda::array<ComplexType, 4>*,
                               nda::array<ComplexType, 4>*,
                               bool);

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
