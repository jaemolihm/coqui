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

#ifndef METHODS_ERI_LR_THC_NEW_HPP
#define METHODS_ERI_LR_THC_NEW_HPP

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
//   - serial: no MPI distribution, no OpenMP, no GPU
//   - npol == 1, nkpts == nkpts_ibz (no k-symmetry)

#include <string>
#include "configuration.hpp"
#include "utilities/Timer.hpp"
#include "nda/nda.hpp"

#include "thc_reader_t.hpp"

namespace methods {

class lr_thc_new {
public:
  // Holds a reference to a thc_reader_t. The reader provides pivot indices
  // ri(), the truncated G-grid rho_g(), the Coulomb evaluator vG(), and the
  // mean-field object.
  explicit lr_thc_new(thc_reader_t& reader) : _reader(reader) {
    for (auto& name : {
        // top-level (one per helper)
        "read_psi_and_dpsi", "compute_T_and_DeltaT",
        "build_Z_C_for_q",  "build_DeltaZ_DeltaC_for_q",
        "solve_xi_for_q",   "solve_dxi_for_q",
        "solve_all_q",      "compute_delta_V_for_q",
        "compute_delta_V_new",
        // sub-clocks (only around bulk operations, not hot inner loops)
        "io_wfc", "io_dpsi", "io_dpsi_adj", "phase_factor",
        "accumulate_Z", "accumulate_dZ",
        "lu_factor", "lu_solve",
        "fft_forward_std", "fft_forward_conj",
        "v_evaluate", "gemm_term1", "gemm_term2"})
      Timer.add(name);
  }

  ~lr_thc_new() = default;
  lr_thc_new(lr_thc_new const&) = delete;
  lr_thc_new& operator=(lr_thc_new const&) = delete;

  // Compute δV^q for every internal q on the BZ k-grid, single phonon Q,
  // single mode. Returns shape (nq, Np, Np) complex.
  //
  //   Deltapsi_prefix     : +Q δψ files. File {prefix}_ik{ik+1}.hdf5 stores
  //                         δ^{+Q} ψ^{k_ik}.
  //   Deltapsi_adj_prefix : −Q δψ (adjoint) files. File {prefix}_ik{ik+1}.hdf5
  //                         stores δ^{−Q} ψ^{k_ik + Q}.
  //   q_pert_cryst        : phonon Q in crystal coordinates.
  nda::array<ComplexType, 3> compute_delta_V_new(
      std::string const& Deltapsi_prefix,
      std::string const& Deltapsi_adj_prefix,
      nda::array<double, 1> const& q_pert_cryst);

private:
  // --- Helpers (see lr_thc_new.cpp for full descriptions) ---

  // Step 1: read ψ and (+Q, −Q) δψ on the full FFT grid for all (is, ik).
  // Bloch phases applied: e^{ik·r} on ψ; e^{i(k+Q)·r} on +Q δψ; e^{i(k−Q)·r}
  // on −Q δψ (after the adj-file index shuffle).
  void read_psi_and_dpsi(
      std::string const& Deltapsi_prefix,
      std::string const& Deltapsi_adj_prefix,
      nda::stack_array<double, 3> const& q_pert_cart,
      nda::stack_array<long, 3> const& fft_mesh,
      nda::array<int, 1> const& kmQ_map,
      nda::array<ComplexType, 4>& psi_skmr,
      nda::array<ComplexType, 4>& Deltapsi_pQ_skmr,
      nda::array<ComplexType, 4>& Deltapsi_mQ_skmr);

  // Step 2: compute the three (T, ΔT_pQ, ΔT_mQ) per-k caches from the
  // ψ / δψ slabs. Same inner kernel; only the right operand changes.
  void compute_T_and_DeltaT(
      nda::array<ComplexType, 4> const& psi_skmr,
      nda::array<ComplexType, 4> const& Deltapsi_pQ_skmr,
      nda::array<ComplexType, 4> const& Deltapsi_mQ_skmr,
      nda::array<long, 1> const& r_P,
      nda::array<ComplexType, 4>& T_skur,
      nda::array<ComplexType, 4>& DeltaT_pQ_skur,
      nda::array<ComplexType, 4>& DeltaT_mQ_skur);

  // Step 3: assemble (Z, C) for one internal q, and return the k−q map.
  // Z^q_{μr}    = Σ_k conj(T^{k−q}_{μr}) · T^k_{μr}    (Bloch q)
  // C^q_{μν}    = Z^q_{μ, r_P(ν)}
  void build_Z_C_for_q(
      nda::array<ComplexType, 4> const& T_skur,
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
      nda::array<ComplexType, 4> const& T_skur,
      nda::array<ComplexType, 4> const& DeltaT_pQ_skur,
      nda::array<ComplexType, 4> const& DeltaT_mQ_skur,
      nda::array<int, 1> const& kmq_map,
      nda::array<long, 1> const& r_P,
      nda::array<ComplexType, 2>& DeltaZ_q,
      nda::array<ComplexType, 2>& DeltaC_q);

  // Step 5: solve C_q · Ξ^q = Z^q, then FFT Ξ^q from real space to G-space
  // (standard and conjugate paths). Returns ipiv + C_f for reuse in step 6.
  void solve_xi_for_q(
      nda::array<ComplexType, 2>& C_q,        // mutated: contains LU after call
      nda::array<ComplexType, 2> const& Z_q,
      nda::stack_array<double, 3> const& q_cart,
      nda::array<ComplexType, 2>& Xi_q_ur,
      nda::array<ComplexType, 2>& Xi_q_uG,
      nda::array<ComplexType, 2>& Xi_bar_q_uG,
      nda::matrix<ComplexType, nda::F_layout>& C_f_lu,
      nda::array<int, 1>& ipiv);

  // Step 6: solve C_q · δΞ^q = δZ^q − δC^q · Ξ^{q+Q}, then FFT δΞ^q with
  // phase e^{±i(q+Q)·r}. Reuses the LU factor from step 5.
  void solve_dxi_for_q(
      nda::matrix<ComplexType, nda::F_layout> const& C_f_lu,
      nda::array<int, 1> const& ipiv,
      nda::array<ComplexType, 2> const& DeltaZ_q,
      nda::array<ComplexType, 2> const& DeltaC_q,
      nda::array<ComplexType, 2> const& Xi_qpQ_ur,
      nda::stack_array<double, 3> const& q_pQ_cart,
      nda::array<ComplexType, 2>& DeltaXi_q_ur,
      nda::array<ComplexType, 2>& DeltaXi_q_uG,
      nda::array<ComplexType, 2>& DeltaXi_bar_q_uG);

  // Step 7: drive steps 3–6 over all q' on the BZ k-grid.
  // Pass A: build Ξ at every q' (also caches kmq_map and the real-space Ξ).
  // Pass B: build δΞ at every q' using cached Ξ at q'+Q.
  void solve_all_q(
      nda::array<ComplexType, 4> const& T_skur,
      nda::array<ComplexType, 4> const& DeltaT_pQ_skur,
      nda::array<ComplexType, 4> const& DeltaT_mQ_skur,
      nda::array<double, 2> const& kpts_crys,
      nda::array<long, 1> const& r_P,
      nda::array<double, 1> const& q_pert_cryst,
      nda::array<ComplexType, 3>& Xi_uG_all,
      nda::array<ComplexType, 3>& Xi_bar_uG_all,
      nda::array<ComplexType, 3>& DeltaXi_uG_all,
      nda::array<ComplexType, 3>& DeltaXi_bar_uG_all);

  // Step 8: assemble δV^q for one internal q via the two-term gemm formula.
  void compute_delta_V_for_q(
      long iq,
      long iq_mqmQ,
      nda::array<ComplexType, 3> const& Xi_uG_all,
      nda::array<ComplexType, 3> const& Xi_bar_uG_all,
      nda::array<ComplexType, 3> const& DeltaXi_uG_all,
      nda::array<ComplexType, 3> const& DeltaXi_bar_uG_all,
      nda::stack_array<double, 3> const& q_cart,
      nda::stack_array<double, 3> const& q_pQ_cart,
      nda::array<ComplexType, 2>& dV_q);

  void print_timers();

  thc_reader_t& _reader;
  utils::TimerManager Timer;
};

} // namespace methods

#endif // METHODS_ERI_LR_THC_NEW_HPP
