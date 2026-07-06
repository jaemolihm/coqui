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

#ifndef METHODS_ERI_LR_THC_HPP
#define METHODS_ERI_LR_THC_HPP

// Linear response of the THC auxiliary basis V^q for non-zero phonon
// wavevector Q (theory-note convention, see docs/lr_thc_theory.txt).
//
// This module is a clean from-scratch implementation written alongside the
// existing methods::lr_thc. It cannot be reused because:
//
//   1. The existing lr_thc uses internal-q convention k+q (ψ^{k+q*} ψ^k);
//      the theory note uses the k−q convention (ψ^{k−q*} ψ^k). Mapping
//      between the two is straightforward in math but leaks throughout
//      the caching/index code.
//   2. The existing lr_thc::build_kcache fuses +Q and −Q δψ contributions
//      into a single δT cache. For Q ≠ 0 these are independent quantities
//      (δ^{+Q}ψ and δ^{−Q}ψ are not related by conjugation), and they must
//      be carried separately.
//   3. The existing lr_thc::build_dV_from_xi assembles δV with a single
//      Coulomb factor v(|q+G|). For Q ≠ 0 the two pieces of δV carry
//      different Coulomb momenta — v(|q+G|) vs v(|q+Q+G|) — and the two
//      Ξ operands live at different internal momenta (q and q+Q).
//   4. The existing module is designed around SHM-cache + MPI-distributed
//      stage A; retrofitting two independent δψ caches and a coupled
//      stage-A/B pipeline would obscure the new physics. We choose
//      serial / no-GPU / no-OpenMP / no-MPI here to make the new path
//      trivially auditable.
//
// Restrictions:
//   - single phonon Q (q_pert_cryst), single mode (caller composes modes)
//   - pivot response Δr_P ignored
//   - MPI parallelization: Phase 1 distributes the (is, ik) loops in steps 1
//     and 2 across node_comm; the ψ/δψ/T caches live in node-shared memory.
//     Steps 7 and 8 are still replicated (Phase 2 will q-distribute them).
//   - no OpenMP, no GPU
//   - npol == 1, nkpts == nkpts_ibz (no k-symmetry)

#include <memory>
#include <string>
#include "configuration.hpp"
#include "utilities/Timer.hpp"
#include "utilities/mpi_context.h"
#include "nda/nda.hpp"
#include "mpi3/communicator.hpp"
#include "numerics/shared_array/nda.hpp"

#include "thc_reader_t.hpp"

namespace methods {

class lr_thc {
public:
  // Holds a reference to a thc_reader_t. The reader provides pivot indices
  // ri(), the truncated G-grid rho_g(), the Coulomb evaluator vG(), and the
  // mean-field object. The reader's MPI context is captured for shared-memory
  // caches and work distribution.
  explicit lr_thc(thc_reader_t& reader)
      : _reader(reader), _mpi(reader.mpi()) {
    for (auto& name : {
        // top-level (one per helper)
        "read_psi_and_dpsi", "compute_T_and_DeltaT",
        "build_Z_C_for_q",  "build_DeltaZ_DeltaC_for_q",
        "solve_xi_for_q",   "solve_dxi_for_q",
        "solve_xi_dxi_all_q",      "compute_delta_V_for_q",
        "compute_delta_V",
        // sub-clocks (only around bulk operations, not hot inner loops)
        "io_wfc", "io_dpsi", "io_dpsi_adj", "phase_factor",
        "accumulate_Z", "accumulate_dZ",
        "xi_lu_factor", "xi_lu_solve", "dxi_lu_solve",
        "xi_eig_factor", "xi_eig_solve", "dxi_eig_solve",
        "xi_phase_factor", "dxi_phase_factor",
        "xi_fft_forward_std", "dxi_fft_forward_std",
        "xi_copy_rhs", "xi_copy_out",
        "dxi_rhs_gemm", "dxi_copy_out",
        "v_evaluate", "gemm_term1", "gemm_term2"})
      Timer.add(name);
  }

  ~lr_thc() = default;
  lr_thc(lr_thc const&) = delete;
  lr_thc& operator=(lr_thc const&) = delete;

  // Compute δV^q for every internal q on the BZ k-grid, single phonon Q,
  // single mode. Returns shape (nq, Np, Np) complex.
  //
  //   Deltapsi_prefix     : +Q δψ files. File {prefix}_ik{ik+1}.hdf5 stores
  //                         δ^{+Q} ψ^{k_ik}.
  //   Deltapsi_adj_prefix : −Q δψ (adjoint) files. File {prefix}_ik{ik+1}.hdf5
  //                         stores δ^{−Q} ψ^{k_ik + Q}.
  //   q_pert_cryst        : phonon Q in crystal coordinates.
  nda::array<ComplexType, 3> compute_delta_V(
      std::string const& Deltapsi_prefix,
      std::string const& Deltapsi_adj_prefix,
      nda::array<double, 1> const& q_pert_cryst);

private:
  // C_q factorization. LU by default; hermitian eig path gated on env var
  // COQUI_LR_THC_USE_HERMITIAN_EIG (eig solve = U · diag(inv_evals) · U^H · B).
  // Only the fields for the active path are populated.
  struct C_factor {
    bool use_eig = false;
    // LU path
    nda::matrix<ComplexType, nda::F_layout> C_f_lu;
    nda::array<int, 1> ipiv;
    // Eigen path (U columns are eigenvectors; inv_evals = 1/λ, λ > 0 checked
    // at factorization)
    nda::matrix<ComplexType, nda::F_layout> U;
    nda::array<double, 1> inv_evals;
  };

  // Phonon Q and BZ-index maps on the k-grid (= q-grid). Populated once by
  // compute_delta_V and consumed by every step; centralizing avoids
  // passing slightly different map subsets to each callee.
  //   kpQ_map(p):  ip s.t. kpts(ip) ≡  kpts(p) + Q   (mod G)
  //   kmQ_map(p):  ip s.t. kpts(ip) ≡  kpts(p) - Q   (mod G)
  //   mk_map(p):   ip s.t. kpts(ip) ≡ -kpts(p)       (mod G)
  //   mkmQ_map(p): ip s.t. kpts(ip) ≡ -kpts(p) - Q   (mod G)
  struct momentum_maps_t {
    nda::stack_array<double, 3> Q_cryst;
    nda::stack_array<double, 3> Q_cart;
    nda::array<int, 1> kpQ_map, kmQ_map, mk_map, mkmQ_map;
  };

  // Build the BZ-index maps for the given perturbation Q on the k-grid.
  void build_momentum_maps(nda::array<double, 2> const& kpts_crys,
                           nda::stack_array<double, 3> const& Q_cryst,
                           nda::stack_array<double, 3> const& Q_cart);

  // --- Helpers (see lr_thc.cpp for full descriptions) ---

  // Step 1: read ψ and (+Q, −Q) δψ on the full FFT grid for all (is, ik).
  // Bloch phases: e^{ik·r} on ψ, e^{i(k+Q)·r} on +Q δψ,
  // e^{i·kpts(ik_kmQ)·r} on −Q δψ (ik_kmQ = _kqpoint_maps.kmQ_map(ik)).
  void read_psi_and_dpsi(
      std::string const& Deltapsi_prefix,
      std::string const& Deltapsi_adj_prefix,
      nda::stack_array<long, 3> const& fft_mesh,
      nda::array_view<ComplexType, 4> psi_skmr,
      nda::array_view<ComplexType, 4> Deltapsi_pQ_skmr,
      nda::array_view<ComplexType, 4> Deltapsi_mQ_skmr);

  // Step 2: (T, ΔT_pQ, ΔT_mQ) per-k caches. Bra-side ΔT terms are read at
  // the BZ-folded k±Q index (via _kqpoint_maps) so each ΔT carries a single r-Bloch
  // momentum. See the formula in compute_T_and_DeltaT.
  void compute_T_and_DeltaT(
      nda::array_view<ComplexType, 4> const psi_skmr,
      nda::array_view<ComplexType, 4> const Deltapsi_pQ_skmr,
      nda::array_view<ComplexType, 4> const Deltapsi_mQ_skmr,
      nda::array<long, 1> const& r_P,
      nda::array_view<ComplexType, 4> T_skur,
      nda::array_view<ComplexType, 4> DeltaT_pQ_skur,
      nda::array_view<ComplexType, 4> DeltaT_mQ_skur);

  // Step 3: assemble (Z, C) for one internal q, and return the k−q map.
  // Z^q_{μr}    = Σ_k conj(T^{k−q}_{μr}) · T^k_{μr}    (Bloch q)
  // C^q_{μν}    = Z^q_{μ, r_P(ν)}
  void build_Z_C_for_q(
      nda::array_view<ComplexType, 4> const T_skur,
      nda::array<double, 2> const& kpts_crys,
      nda::array<long, 1> const& r_P,
      nda::array<double, 1> const& q_vec_cryst,
      nda::array<ComplexType, 2>& Z_q,
      nda::array<ComplexType, 2>& C_q,
      nda::array<int, 1>& kmq_map);

  // Step 4: assemble (ΔZ, ΔC) for one internal q. Same k−q map as step 3.
  // δZ^q_{μr} = Σ_k [conj(ΔT_mQ^{k−q}_{μr}) · T^k_{μr}
  //                  + conj(T^{k−q}_{μr}) · ΔT_pQ^k_{μr}]    (Bloch q+Q)
  void build_DeltaZ_DeltaC_for_q(
      nda::array_view<ComplexType, 4> const T_skur,
      nda::array_view<ComplexType, 4> const DeltaT_pQ_skur,
      nda::array_view<ComplexType, 4> const DeltaT_mQ_skur,
      nda::array<int, 1> const& kmq_map,
      nda::array<long, 1> const& r_P,
      nda::array<ComplexType, 2>& DeltaZ_q,
      nda::array<ComplexType, 2>& DeltaC_q);

  // Step 5: solve C_q · Ξ^q = Z^q in real space, then FFT to G with
  // phase e^{-iq·r}. Returns real-space Ξ^q as well (consumed by step 6).
  // C_q is symmetrized in-place. Negative-q slots are recovered via
  // Ξ^{-q}(r) = [Ξ^q(r)]^*, so only the +G image is needed.
  void solve_xi_for_q(
      nda::array<ComplexType, 2>& C_q,        // mutated: symmetrized in-place
      nda::array<ComplexType, 2> const& Z_q,
      nda::stack_array<double, 3> const& q_cart,
      nda::array<ComplexType, 2>& Xi_q_ur,
      nda::array<ComplexType, 2>& Xi_q_uG,
      C_factor& factor);

  // Step 6: build δΞ̃^q entirely in G-space.
  //   δZ̃(μ, G) = FFT[e^{-i(q+Q)·r} · δZ(μ, r)](G)        (one fwd FFT per μ)
  //   rhs̃(G)   = δZ̃(G) − δC · Ξ_uG[iq_qpQ](G)            (G-space gemm)
  //   C_q · δΞ̃^q = rhs̃                                   (G-space solve)
  // FFT linearity + δC acting only on μ make this equivalent to the original
  // (gemm-in-r, then forward-FFT) form, but avoids reconstructing Ξ_qpQ_ur
  // and never allocates an (Np × nnr) rhs buffer. Reuses the step-5 factor.
  void solve_dxi_for_q(
      C_factor const& factor,
      nda::array<ComplexType, 2> const& DeltaZ_q,
      nda::array<ComplexType, 2> const& DeltaC_q,
      nda::array_view<ComplexType, 2> const Xi_qpQ_uG,
      nda::stack_array<double, 3> const& q_pQ_cart,
      nda::array<ComplexType, 2>& DeltaXi_q_uG);

  // Solve C · X = B in-place on B (F-layout, Np rows) using `factor`.
  void apply_C_inverse(
      C_factor const& factor,
      nda::matrix<ComplexType, nda::F_layout>& B,
      char const* timer_solve_tag);

  // Step 7: drive steps 3–6 over all q' on the BZ k-grid.
  //   Pass A: build Ξ^q' at every q'; cache real-space Ξ_ur, C-factor, kmq.
  //   Pass B: build δΞ̃^q' using the cached Ξ_ur at slot _kqpoint_maps.kpQ_map(q').
  void solve_xi_dxi_all_q(
      nda::array_view<ComplexType, 4> const T_skur,
      nda::array_view<ComplexType, 4> const DeltaT_pQ_skur,
      nda::array_view<ComplexType, 4> const DeltaT_mQ_skur,
      nda::array<double, 2> const& kpts_crys,
      nda::array<long, 1> const& r_P,
      nda::array<ComplexType, 3>& Xi_uG_all,
      nda::array<ComplexType, 3>& DeltaXi_uG_all);

  // Step 8: δV^q for one internal q.
  //   term1[μ,ν] = (1/V) Σ_G v(|kpts(iq_mq)+G|)  · δΞ_uG[iq_mqmQ, μ, G] · conj(Ξ_uG[iq_mq, ν, G])
  //   term2[μ,ν] = (1/V) Σ_G v(|kpts(iq_qpQ)+G|) · conj(Ξ_uG[iq_qpQ, μ, G]) · δΞ_uG[iq, ν, G]
  // BZ-folded slot k-points (kpts(iq_mq), kpts(iq_qpQ)) — NOT literal ±q / q+Q.
  void compute_delta_V_for_q(
      long iq,
      long iq_mq,
      long iq_qpQ,
      long iq_mqmQ,
      nda::array<ComplexType, 3> const& Xi_uG_all,
      nda::array<ComplexType, 3> const& DeltaXi_uG_all,
      nda::stack_array<double, 3> const& q_mq_cart,
      nda::stack_array<double, 3> const& q_qpQ_cart,
      nda::array<ComplexType, 2>& dV_q);

  void print_timers();

  thc_reader_t& _reader;
  // MPI context captured from the reader. Shared (node-replicated) caches and
  // (in Phase 2) q-distribution use comm / node_comm / internode_comm here.
  std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> _mpi;
  utils::TimerManager Timer;
  momentum_maps_t _kqpoint_maps;     // populated by compute_delta_V
};

} // namespace methods

#endif // METHODS_ERI_LR_THC_HPP
