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


#ifndef COQUI_LR_DYSON_HPP
#define COQUI_LR_DYSON_HPP

#include "utilities/Timer.hpp"
#include "IO/app_loggers.h"

#include "utilities/mpi_context.h"
#include "mean_field/MF.hpp"
#include "numerics/imag_axes_ft/IAFT.hpp"
#include "methods/SCF/simple_dyson.h"
#include "utilities/lr_utils.hpp"

namespace methods {

/**
 * @class lr_dyson
 * @brief Handler for solving the linear response Dyson equation
 *
 * This class is responsible for solving the linearized Dyson equation:
 *
 *   ΔG(k,iω) = G(k+q,iω) · [ΔH0(k) + ΔF(k) + ΔΣ(k,iω) - Δμ·S(k+q,k)] · G(k,iω)
 *
 * where:
 *   - ΔG(k) represents the (k+q, k) block of the perturbed Green's function
 *   - G(k+q) and G(k) are the unperturbed Green's functions
 *   - ΔH0(k) is the perturbation of the non-interacting Hamiltonian
 *   - ΔF(k), ΔΣ(k) are the linearized Fock and self-energy contributions
 *   - Δμ is the chemical potential shift (for particle conservation at q=0)
 *
 * Note: The ordering follows bra-ket convention: ΔG(k) = ⟨ψ_{k+q}|ΔG|ψ_k⟩
 */
class lr_dyson {
public:
  using mpi_context_t = utils::mpi_context_t<mpi3::communicator>;
  template<nda::Array Array_base_t>
  using sArray_t = math::shm::shared_array<Array_base_t>;
  using Array_view_4D_t = nda::array_view<ComplexType, 4>;
  using Array_view_5D_t = nda::array_view<ComplexType, 5>;

  /**
   * @brief Constructor from existing simple_dyson
   *
   * @param dyson     - [INPUT] Simple Dyson solver (provides MF, FT, H0, S)
   * @param q_vec     - [INPUT] Perturbation wavevector in crystal coordinates (3,)
   */
  lr_dyson(simple_dyson& dyson, nda::array<double, 1> const& q_vec);

  lr_dyson(lr_dyson const&) = delete;
  lr_dyson(lr_dyson &&) = default;
  lr_dyson & operator=(const lr_dyson &) = delete;
  lr_dyson & operator=(lr_dyson &&) = delete;

  ~lr_dyson() {}

  /**
   * @brief Solve LR Dyson equation with fixed self-energy (Phase 1)
   *
   * Computes: ΔG(k,iω) = G(k+q,iω) · ΔH0(k) · G(k,iω)
   *
   * This is the simplest form where Σ is fixed (no ΔΣ term).
   *
   * @param sDeltaG_tskij   - [OUTPUT] LR Green's function (nt, ns, nk, nb, nb)
   * @param sG_tskij        - [INPUT] Unperturbed Green's function (nt, ns, nk, nb, nb)
   * @param sDeltaH0_skij   - [INPUT] Perturbation (ns, nk, nb, nb)
   */
  template<typename DeltaG_t, typename G_t, typename DeltaH0_t>
  void solve_lr_dyson_fixed_sigma(
      DeltaG_t& sDeltaG_tskij,
      const G_t& sG_tskij,
      const DeltaH0_t& sDeltaH0_skij);
  // Note: This is a convenience wrapper around solve_lr_dyson with ΔF=0, ΔΣ=0, Δμ=0.
  // Provides a simpler interface for Phase 1 testing and when self-energy is fixed.

  /**
   * @brief Solve full LR Dyson equation
   *
   * Computes: ΔG(k,iω) = G(k+q,iω) · [ΔH0(k) + ΔF(k) + ΔΣ(k,iω) - Δμ·S(k+q,k)] · G(k,iω)
   *
   * @param sDeltaG_tskij     - [OUTPUT] LR Green's function (nt, ns, nk, nb, nb)
   * @param sG_tskij          - [INPUT] Unperturbed Green's function (nt, ns, nk, nb, nb)
   * @param sDeltaH0_skij     - [INPUT] Perturbation (ns, nk, nb, nb)
   * @param sDeltaF_skij      - [INPUT] LR Fock matrix (ns, nk, nb, nb)
   * @param sDeltaSigma_tskij - [INPUT] LR self-energy (nt, ns, nk, nb, nb)
   * @param Delta_mu          - [INPUT] Chemical potential shift
   */
  template<typename DeltaG_t, typename G_t, typename DeltaH0_t, typename DeltaF_t, typename DeltaSigma_t>
  void solve_lr_dyson(
      DeltaG_t& sDeltaG_tskij,
      const G_t& sG_tskij,
      const DeltaH0_t& sDeltaH0_skij,
      const DeltaF_t& sDeltaF_skij,
      const DeltaSigma_t& sDeltaSigma_tskij,
      double Delta_mu);

  /**
   * @brief Compute LR density matrix from LR Green's function
   *
   * Computes: ΔDm(k) = -ΔG(k, τ=β⁻)
   *
   * @param sDeltaDm_skij   - [OUTPUT] LR density matrix (ns, nk, nb, nb)
   * @param sDeltaG_tskij   - [INPUT] LR Green's function (nt, ns, nk, nb, nb)
   */
  template<typename DeltaDm_t, typename DeltaG_t>
  void compute_lr_dm(DeltaDm_t& sDeltaDm_skij, const DeltaG_t& sDeltaG_tskij);

  inline void print_timers() {
    app_log(2, "\n  LR_DYSON timers");
    app_log(2, "  ----------------");
    app_log(2, "    LR Dyson eqn:                   {0:.3f} sec", _Timer.elapsed("LR_DYSON"));
    app_log(2, "      - G(t)->G(w):                 {0:.3f} sec", _Timer.elapsed("G_TAU_TO_W"));
    app_log(2, "      - LR Dyson loop:              {0:.3f} sec", _Timer.elapsed("LR_DYSON_LOOP"));
    app_log(2, "      - Gather:                     {0:.3f} sec\n", _Timer.elapsed("LR_DYSON_GATHER"));
  }

  // Accessors
  const nda::array<int, 1>& kpq_map() const { return _kpq_map; }
  const nda::array<double, 1>& q_vec() const { return _q_vec; }
  bool is_q_gamma() const { return _is_q_gamma; }

private:
  simple_dyson& _dyson;
  std::shared_ptr<mpi_context_t> _context;

  int _nts;
  int _nw;
  int _ns;
  int _nkpts;
  int _nkpts_ibz;
  int _nbnd;

  nda::array<double, 1> _q_vec;     // Perturbation wavevector (3,)
  nda::array<int, 1> _kpq_map;      // k → k+q index mapping (nkpts,)
  bool _is_q_gamma;                 // True if q is approximately zero

  utils::TimerManager _Timer;
};

} // namespace methods

#endif // COQUI_LR_DYSON_HPP
