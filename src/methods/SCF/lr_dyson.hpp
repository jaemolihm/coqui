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


#ifndef COQUI_LR_DYSON_HPP
#define COQUI_LR_DYSON_HPP

#include "utilities/Timer.hpp"
#include "IO/app_loggers.h"

#include "utilities/mpi_context.h"
#include "mean_field/MF.hpp"
#include "numerics/imag_axes_ft/IAFT.hpp"
#include "methods/SCF/simple_dyson.h"
#include "utilities/lr_utils.hpp"
#include "utilities/proc_grid_partition.hpp"

namespace methods {

/**
 * @brief Processor grid and block size for the ω-side band-basis LR Dyson arrays
 *        (ΔG(iω), and ΔΣ(iω) when GW is active).
 *
 * ΔG = G_{k+q}·X·G_k is evaluated as two local gemms on complete nbnd x nbnd
 * matrices, so a rank cannot work from a partial band block: X is assembled from
 * ΔH0, ΔF, S (+ ΔV_QPGW, ΔΣ) and every one of those has to be whole. The band
 * axes are therefore left undivided whenever possible — ranks that the (ω, k)
 * pools cannot absorb go onto the ω axis, which only needs nw >= nwpools (the
 * split may be uneven; chunk_range handles that).
 *
 * Splitting the band axes is the last resort, taken only when the ω axis has no
 * room. It costs every rank the full nbnd^3 product per (ω, s, k) of which it
 * keeps one block, and it leaves ΔΣ unusable, so LR-GW rejects it.
 *
 * Kept in one place because lr_driver's distribution report has to agree with
 * what solve_lr_dyson_impl actually allocates.
 *
 * @return {pgrid, bsize} over the (w, s, k, i, j) axes.
 */
inline std::pair<std::array<long, 5>, std::array<long, 5>>
lr_dyson_omega_pgrid(long nproc, long nw, long nkpts_ibz, long nbnd) {
  long np = nproc;
  long nwpools = utils::find_proc_grid_max_npools(np, nw, 0.4);
  np /= nwpools;
  long nkpools = utils::find_proc_grid_max_npools(np, nkpts_ibz, 0.4);
  np /= nkpools;

  long np_i = 1, np_j = 1;
  if (np > 1) {
    if (nwpools * np <= nw) {
      nwpools *= np;
    } else {
      np_i = utils::find_proc_grid_min_diff(np, 1, 1);
      np_j = np / np_i;
    }
  }

  long ibsize = std::min({1024L, nbnd / np_i, nbnd / np_j});
  if (ibsize < 1) ibsize = 1;
  return {{nwpools, 1, nkpools, np_i, np_j}, {1, 1, 1, ibsize, ibsize}};
}

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
   * @brief Set cached G(iω) for the LR Dyson equation
   *
   * Must be called before solve_lr_dyson. The caller retains ownership;
   * the pointed-to array must outlive all solve_lr_dyson calls.
   *
   * @param G_wskij - [INPUT] Pointer to precomputed shared G(iω) (nw, ns, nk, nb, nb)
   */
  void set_cached_G_omega(const sArray_t<Array_view_5D_t>* G_wskij) {
    // Invalidate dN/dμ cache when G pointer changes (dN/dμ depends on G)
    if (_cached_G_wskij != G_wskij) {
      _dN_dmu_cached = false;
      _cached_dN_dmu = 0.0;
    }
    _cached_G_wskij = G_wskij;
  }

  /**
   * @brief Solve full LR Dyson equation
   *
   * Computes: ΔG(k,iω) = G(k+q,iω) · [ΔH0(k) + ΔF(k) + ΔΣ(k,iω) - Δμ·S(k+q,k)] · G(k,iω)
   *
   * Requires: set_cached_G_omega() must have been called first.
   * For fix_density mode at q=0: compute_dN_dmu() must have been called first.
   *
   * Two modes are available:
   * - fix_density=false (default): Use the provided Delta_mu value
   * - fix_density=true: Automatically compute Δμ to enforce ΔN=0 (particle conservation)
   *
   * For q≠0, fix_density is ignored (Δμ contribution vanishes).
   *
   * @param sDeltaG_tskij     - [OUTPUT] LR Green's function (nt, ns, nk, nb, nb)
   * @param sDeltaDm_skij     - [OUTPUT] LR density matrix (ns, nk, nb, nb)
   * @param sDeltaH0_skij     - [INPUT] Perturbation (ns, nk, nb, nb)
   * @param sDeltaF_skij      - [INPUT] LR Fock matrix (ns, nk, nb, nb)
   * @param sDeltaSigma_tskij - [INPUT] LR self-energy (nt, ns, nk, nb, nb), nullptr if not used
   * @param fix_density       - [INPUT] If true, compute Δμ to enforce ΔN=0
   * @param Delta_mu          - [INPUT] Chemical potential shift (used only if fix_density=false)
   * @param sDeltaVcorr_skij  - [INPUT] static ΔV_QPGW (ns, nk, nb, nb), nullptr if not used.
   *   Added to the frequency-independent one-body term of the RHS (LR-qpGW mode).
   * @return The Δμ value used (computed if fix_density=true, otherwise the input value)
   */
  template<typename DeltaG_t, typename DeltaDm_t, typename DeltaH0_t,
           typename DeltaF_t, typename DeltaSigma_t>
  double solve_lr_dyson(
      DeltaG_t& sDeltaG_tskij,
      DeltaDm_t& sDeltaDm_skij,
      const DeltaH0_t& sDeltaH0_skij,
      const DeltaF_t& sDeltaF_skij,
      const DeltaSigma_t* sDeltaSigma_tskij = nullptr,
      bool fix_density = false,
      double Delta_mu = 0.0,
      const DeltaF_t* sDeltaVcorr_skij = nullptr);

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

  /**
   * @brief Compute particle number change from LR density matrix (q=0 only)
   *
   * Computes: ΔN = Tr[S · ΔDm] = Σ_k w_k Tr[S(k) · ΔDm(k)]
   *
   * For q=0, ΔN should be zero at convergence when Δμ is adjusted properly.
   *
   * @throws std::runtime_error if q≠0 (ΔN is not well-defined for q≠0)
   * @param sDeltaDm_skij   - [INPUT] LR density matrix (ns, nk, nb, nb)
   * @return ΔN, the change in particle number
   */
  template<typename DeltaDm_t>
  double compute_lr_Nelec(const DeltaDm_t& sDeltaDm_skij);

  /**
   * @brief Compute dN/dμ from cached G(iω) (q=0 only)
   *
   * Computes: dN/dμ = Tr[S · (G·S·G)(τ=β⁻)] = Σ_k w_k Tr[S(k) · G(k)·S(k)·G(k)(τ=β⁻)]
   *
   * This is used to compute Δμ via: Δμ = -ΔN_0 / (dN/dμ)
   *
   * NOTE: This quantity depends only on the unperturbed Green's function G, NOT on
   * any LR quantities (ΔH0, ΔF, ΔΣ). The result is cached internally. Subsequent
   * calls return the cached value without recomputation.
   *
   * Requires: set_cached_G_omega() must have been called first.
   *
   * @throws std::runtime_error if q≠0
   * @return dN/dμ, the density of states at the Fermi level
   */
  double compute_dN_dmu();

  /**
   * @brief Compute Δμ to enforce particle conservation ΔN = 0 (q=0 only)
   *
   * For q=0 perturbations, the particle number changes unless we add a
   * chemical potential shift Δμ. This method computes Δμ such that:
   *
   *   ΔN(Δμ) = Tr[S · ΔDm(Δμ)] = 0
   *
   * The LR Dyson equation is linear in Δμ, so the closed-form solution is:
   *   Δμ = -ΔN_0 / (dN/dμ)
   *
   * Note: sDeltaDm_skij should be computed at Δμ=0 (i.e., ΔN_0).
   *
   * @throws std::runtime_error if q≠0
   * @param sDeltaDm_skij   - [INPUT] LR density matrix at Δμ=0
   * @return Δμ value that enforces ΔN=0
   */
  template<typename DeltaDm_t>
  double compute_Delta_mu(const DeltaDm_t& sDeltaDm_skij);

  /// Print the LR Dyson timer block (header + total + subclocks) at log level `level`.
  inline void print_timers(int level = 2) {
    app_log(level, "\n  LR_DYSON timers");
    app_log(level, "  ----------------");
    app_log(level, "    LR Dyson eqn:                   {0:8.3f} sec  {1:4d} calls", _Timer.elapsed("LR_DYSON"), _Timer.number_of_calls("LR_DYSON"));
    print_subclocks(level, "    ");
    app_log(level, "");
  }

  /// Print only the component clocks, each line prefixed by `indent`.
  /// Embedded (with deeper indent) in lr_driver's final hierarchical report.
  inline void print_subclocks(int level, const std::string& indent) {
    app_log(level, "{0}  - Alloc (dist arrays):        {1:8.3f} sec  {2:4d} calls", indent, _Timer.elapsed("LR_DYSON_ALLOC"), _Timer.number_of_calls("LR_DYSON_ALLOC"));
    app_log(level, "{0}  - ΔΣ(τ)->ΔΣ(iω):              {1:8.3f} sec  {2:4d} calls", indent, _Timer.elapsed("LR_DYSON_TAU_TO_W"), _Timer.number_of_calls("LR_DYSON_TAU_TO_W"));
    app_log(level, "{0}  - LR Dyson loop:              {1:8.3f} sec  {2:4d} calls", indent, _Timer.elapsed("LR_DYSON_LOOP"), _Timer.number_of_calls("LR_DYSON_LOOP"));
    app_log(level, "{0}  - Redistribute ΔG(iω):        {1:8.3f} sec  {2:4d} calls", indent, _Timer.elapsed("LR_DYSON_REDIST"), _Timer.number_of_calls("LR_DYSON_REDIST"));
    app_log(level, "{0}  - ΔG(iω)->ΔG(τ):              {1:8.3f} sec  {2:4d} calls", indent, _Timer.elapsed("LR_DYSON_G_W_TO_T"), _Timer.number_of_calls("LR_DYSON_G_W_TO_T"));
    app_log(level, "{0}  - Gather:                     {1:8.3f} sec  {2:4d} calls", indent, _Timer.elapsed("LR_DYSON_GATHER"), _Timer.number_of_calls("LR_DYSON_GATHER"));
    app_log(level, "{0}  - Compute ΔDm:                {1:8.3f} sec  {2:4d} calls", indent, _Timer.elapsed("LR_DYSON_DM"), _Timer.number_of_calls("LR_DYSON_DM"));
    app_log(level, "{0}  - Compute ΔN:                 {1:8.3f} sec  {2:4d} calls", indent, _Timer.elapsed("LR_DYSON_NELEC"), _Timer.number_of_calls("LR_DYSON_NELEC"));
    app_log(level, "{0}  - Misc (barrier/reset):       {1:8.3f} sec  {2:4d} calls", indent, _Timer.elapsed("LR_DYSON_MISC"), _Timer.number_of_calls("LR_DYSON_MISC"));
  }

  // Accessors
  const nda::array<int, 1>& kpq_map() const { return _kpq_map; }
  const nda::array<double, 1>& q_vec() const { return _q_vec; }
  bool is_q_gamma() const { return _is_q_gamma; }

private:
  // Internal implementation of the LR Dyson solve (single pass with given Δμ)
  // Uses _cached_G_wskij for G(iω); caller must ensure cache is populated.
  template<typename DeltaG_t, typename DeltaH0_t,
           typename DeltaF_t, typename DeltaSigma_t>
  void solve_lr_dyson_impl(
      DeltaG_t& sDeltaG_tskij,
      const DeltaH0_t& sDeltaH0_skij,
      const DeltaF_t& sDeltaF_skij,
      const DeltaSigma_t* sDeltaSigma_tskij,
      double Delta_mu,
      const DeltaF_t* sDeltaVcorr_skij = nullptr);

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

  // Cached G(iω) in shared memory — set via set_cached_G_omega().
  // Non-owning pointer; caller (lr_driver) owns the array.
  const sArray_t<Array_view_5D_t>* _cached_G_wskij = nullptr;
  // Cached dN/dμ — populated by first call to compute_dN_dmu()
  double _cached_dN_dmu = 0.0;
  bool _dN_dmu_cached = false;

  utils::TimerManager _Timer;
};

} // namespace methods

#endif // COQUI_LR_DYSON_HPP
