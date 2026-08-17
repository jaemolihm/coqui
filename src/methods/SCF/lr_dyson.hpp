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

#include <optional>

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
 * Splitting the band axes is the last resort, taken only when no (ω, k) pool pair
 * covers all the ranks. It costs every rank the full nbnd^3 product per (ω, s, k)
 * of which it keeps one block, and it leaves ΔΣ unusable, so LR-GW rejects it.
 *
 * Greedy ω-then-k pool filling can leave a remainder that neither axis can absorb
 * and spill it onto the bands. Only in that case do we search the factorisations
 * nwpools*nkpools == nproc with nwpools <= nw and nkpools <= nkpts_ibz and take the
 * best.
 *
 * Cost model for the search: a rank owns ceil(nw/nwpools) x ceil(nkpts_ibz/nkpools)
 * of the (ω, k) plane and the Dyson loop is one nbnd^3 pair of gemms per element, so
 * that product is the max per-rank work. The ΔΣ tau->omega transform scales with the
 * same product and reads its input from shared memory, so it does not favour either
 * axis; the metric alone therefore leaves ties (nw=40, nkpts_ibz=64, nproc=960 admits
 * both (15, 64) and (40, 24) at max work 3).
 *
 * Ties go to more k pools. The ΔG(iω) redistribute that follows the Dyson loop targets
 * a grid whose k pools are maximised first, so a source grid that splits k the same way
 * keeps that all-to-all inside one k column; the ω-heavy alternative fans every rank
 * out across all of them for no gain elsewhere.
 *
 * If no factorisation exists either (nproc has a factor exceeding both axes) the
 * band split stands and LR-GW rejects it downstream with a diagnostic.
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

  if (np_i * np_j > 1) {
    auto ceil_div = [](long a, long b) { return (a + b - 1) / b; };
    long best_w = 0, best_k = 0, best_work = 0;
    for (long a = 1; a <= std::min(nproc, nw); ++a) {
      if (nproc % a != 0) continue;
      long b = nproc / a;
      if (b > nkpts_ibz) continue;
      long work = ceil_div(nw, a) * ceil_div(nkpts_ibz, b);
      // strict <, ascending a: ties keep the smallest nwpools, i.e. the most k pools
      if (best_w == 0 or work < best_work) {
        best_work = work;
        best_w = a;
        best_k = b;
      }
    }
    if (best_w > 0) {
      nwpools = best_w;
      nkpools = best_k;
      np_i = np_j = 1;
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
  using dArray_5D_t = memory::darray_t<nda::array<ComplexType, 5>, mpi3::communicator>;

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
   * ΔG(τ) itself is left distributed: the solve retains it and the caller must
   * call materialize_DeltaG_tau() exactly once to replicate it into shared
   * memory. ΔDm, which most callers actually want, is produced here as usual.
   *
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
  template<typename DeltaDm_t, typename DeltaH0_t,
           typename DeltaF_t, typename DeltaSigma_t>
  double solve_lr_dyson(
      DeltaDm_t& sDeltaDm_skij,
      const DeltaH0_t& sDeltaH0_skij,
      const DeltaF_t& sDeltaF_skij,
      const DeltaSigma_t* sDeltaSigma_tskij = nullptr,
      bool fix_density = false,
      double Delta_mu = 0.0,
      const DeltaF_t* sDeltaVcorr_skij = nullptr);

  /**
   * @brief Replicate the ΔG(τ) retained by the last solve into shared memory.
   *
   * The gather is the single most expensive step of the LR Dyson solve — a
   * node-replicating internode all_reduce of the whole nt·ns·nk·nb² array — and
   * a Σ-free SCF iteration never reads ΔG(τ), so the solve hands over the
   * distributed array and the caller decides whether to pay for it.
   *
   * The caller owns that decision outright: this throws if nothing is pending
   * rather than skipping quietly, so "has ΔG(τ) been replicated?" is answered by
   * the call sites and never by state inside lr_dyson. A ΔG(τ) nobody claims is
   * dropped by the next solve.
   *
   * Collective: every rank must call it, or none. In lr_driver that follows from
   * k.include_gw_sigma, which is loop-invariant and identical on every rank.
   */
  void materialize_DeltaG_tau(sArray_t<Array_view_5D_t>& sDeltaG_tskij);

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

  /// Zero this solver's sub-clocks. lr_driver calls it per perturbation so a
  /// batched run's timing report covers one perturbation, not the running total.
  inline void reset_timers() { _Timer.reset(); }

  /// Print only the component clocks, each line prefixed by `indent`.
  /// Embedded (with deeper indent) in lr_driver's final hierarchical report.
  ///
  /// "Gather ΔG(τ)" is the one line that is not a component of the LR Dyson
  /// total above it: the replication is demand-driven and runs wherever the
  /// first consumer sits (inside the Σ step, or after the SCF loop for the
  /// checkpoint), not inside the solve.
  inline void print_subclocks(int level, const std::string& indent) {
    app_log(level, "{0}  - Alloc (dist arrays):        {1:8.3f} sec  {2:4d} calls", indent, _Timer.elapsed("LR_DYSON_ALLOC"), _Timer.number_of_calls("LR_DYSON_ALLOC"));
    app_log(level, "{0}  - ΔΣ(τ)->ΔΣ(iω):              {1:8.3f} sec  {2:4d} calls", indent, _Timer.elapsed("LR_DYSON_TAU_TO_W"), _Timer.number_of_calls("LR_DYSON_TAU_TO_W"));
    app_log(level, "{0}  - LR Dyson loop:              {1:8.3f} sec  {2:4d} calls", indent, _Timer.elapsed("LR_DYSON_LOOP"), _Timer.number_of_calls("LR_DYSON_LOOP"));
    app_log(level, "{0}  - Redistribute ΔG(iω):        {1:8.3f} sec  {2:4d} calls", indent, _Timer.elapsed("LR_DYSON_REDIST"), _Timer.number_of_calls("LR_DYSON_REDIST"));
    app_log(level, "{0}  - ΔG(iω)->ΔG(τ):              {1:8.3f} sec  {2:4d} calls", indent, _Timer.elapsed("LR_DYSON_G_W_TO_T"), _Timer.number_of_calls("LR_DYSON_G_W_TO_T"));
    app_log(level, "{0}  - Gather ΔG(τ):               {1:8.3f} sec  {2:4d} calls", indent, _Timer.elapsed("LR_DYSON_GATHER"), _Timer.number_of_calls("LR_DYSON_GATHER"));
    app_log(level, "{0}      - set_zero:               {1:8.3f} sec  {2:4d} calls", indent, _Timer.elapsed("GATHER_SHM_ZERO"), _Timer.number_of_calls("GATHER_SHM_ZERO"));
    app_log(level, "{0}      - local assign:           {1:8.3f} sec  {2:4d} calls", indent, _Timer.elapsed("GATHER_SHM_ASSIGN"), _Timer.number_of_calls("GATHER_SHM_ASSIGN"));
    app_log(level, "{0}      - pre-reduce barrier (skew):          {1:8.3f} sec  {2:4d} calls", indent, _Timer.elapsed("GATHER_SHM_SKEW"), _Timer.number_of_calls("GATHER_SHM_SKEW"));
    app_log(level, "{0}      - internode all_reduce:   {1:8.3f} sec  {2:4d} calls", indent, _Timer.elapsed("GATHER_SHM_REDUCE"), _Timer.number_of_calls("GATHER_SHM_REDUCE"));
    app_log(level, "{0}      - trailing barrier:       {1:8.3f} sec  {2:4d} calls", indent, _Timer.elapsed("GATHER_SHM_BARRIER"), _Timer.number_of_calls("GATHER_SHM_BARRIER"));
    print_gather_bandwidth(level, indent);
    app_log(level, "{0}  - Compute ΔDm (incl. gather): {1:8.3f} sec  {2:4d} calls", indent, _Timer.elapsed("LR_DYSON_DM"), _Timer.number_of_calls("LR_DYSON_DM"));
    app_log(level, "{0}  - Compute ΔN:                 {1:8.3f} sec  {2:4d} calls", indent, _Timer.elapsed("LR_DYSON_NELEC"), _Timer.number_of_calls("LR_DYSON_NELEC"));
    app_log(level, "{0}  - Misc (barrier/reset):       {1:8.3f} sec  {2:4d} calls", indent, _Timer.elapsed("LR_DYSON_MISC"), _Timer.number_of_calls("LR_DYSON_MISC"));
  }

  /// Rate implied by the internode all_reduce of ΔG(τ). An allreduce of a V-byte
  /// buffer over n nodes puts ~2V(n-1)/n bytes per node on the wire, not V, so that
  /// is what the rate below is computed from and what is comparable to the fabric's
  /// per-node bandwidth: if it already sits near line rate there is no headroom left
  /// in the reduction itself, and the remaining Gather cost is elsewhere.
  ///
  /// The second rate adds the trailing barrier. The reduction is split into
  /// per-chunk allreduces issued in parallel by the ranks of a node, and they do
  /// not finish together, so that barrier is part of the transfer's critical path
  /// and the (reduce + barrier) rate is the honest end-to-end number; the reduce-
  /// only rate overstates the fabric.
  inline void print_gather_bandwidth(int level, const std::string& indent) {
    int ncalls = _Timer.number_of_calls("GATHER_SHM_REDUCE");
    if (ncalls == 0 or _gather_bytes == 0) return;
    double t = _Timer.elapsed("GATHER_SHM_REDUCE") / double(ncalls);
    double t_e2e = t + _Timer.elapsed("GATHER_SHM_BARRIER") / double(ncalls);
    double V_MB = double(_gather_bytes) / (1024.0 * 1024.0);
    long nnodes = _context->internode_comm.size();
    if (nnodes < 2 or t <= 0.0) {
      app_log(level, "{0}      [V = {1:.1f} MB/node, {2:.4f} sec/reduce, {3} node(s)]",
              indent, V_MB, t, nnodes);
      return;
    }
    double wire = 2.0 * double(_gather_bytes) * double(nnodes - 1) / double(nnodes);
    app_log(level, "{0}      [V = {1:.1f} MB/node, {2:.4f} sec/reduce, {3} nodes "
                   "-> {4:.2f} GB/s/node on the wire]",
            indent, V_MB, t, nnodes, wire / t / 1.0e9);
    if (t_e2e > 0.0)
      app_log(level, "{0}      [reduce + trailing barrier: {1:.4f} sec "
                     "-> {2:.2f} GB/s/node end to end]",
              indent, t_e2e, wire / t_e2e / 1.0e9);
  }

  // Accessors
  const nda::array<int, 1>& kpq_map() const { return _kpq_map; }
  const nda::array<double, 1>& q_vec() const { return _q_vec; }
  bool is_q_gamma() const { return _is_q_gamma; }

private:
  // Internal implementation of the LR Dyson solve (single pass with given Δμ).
  // Uses _cached_G_wskij for G(iω); caller must ensure cache is populated.
  // ΔDm = -ΔG(τ=β⁻) is produced here rather than in a separate pass: it is a
  // contraction over τ alone, and the distributed τ array this builds leaves the
  // τ and spin axes undivided, so every rank can form its own (k, i, j) block
  // locally instead of re-reading the gathered replica.
  //
  // ΔΣ arrives already on the ω grid: it does not depend on Δμ, so the caller
  // transforms it once and both fix_density passes read the same darray.
  //
  // last_pass marks the pass whose ΔG(τ) survives the solve; an earlier pass's
  // ΔG is overwritten before anything can read it, so only the last is retained.
  template<typename DeltaDm_t, typename DeltaH0_t, typename DeltaF_t>
  void solve_lr_dyson_impl(
      DeltaDm_t& sDeltaDm_skij,
      const DeltaH0_t& sDeltaH0_skij,
      const DeltaF_t& sDeltaF_skij,
      const dArray_5D_t* dDeltaSigma_wskij,
      double Delta_mu,
      bool last_pass,
      const DeltaF_t* sDeltaVcorr_skij);

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

  // ΔG(τ) of the last solve, still distributed, awaiting a consumer.
  // Empty once materialize_DeltaG_tau() has replicated it, and reset at the top
  // of every solve so a solve never inherits the previous one's array.
  std::optional<dArray_5D_t> _dDeltaG_tau_pending;

  // Bytes of ΔG(τ) replicated per node by the gather; used to report the achieved rate.
  size_t _gather_bytes = 0;

  utils::TimerManager _Timer;
};

} // namespace methods

#endif // COQUI_LR_DYSON_HPP
