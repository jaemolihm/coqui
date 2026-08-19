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


#ifndef COQUI_LR_HF_HPP
#define COQUI_LR_HF_HPP

#include <initializer_list>
#include <optional>

#include "utilities/Timer.hpp"
#include "IO/app_loggers.h"

#include "utilities/mpi_context.h"
#include "mean_field/MF.hpp"
#include "numerics/shared_array/nda.hpp"
#include "numerics/distributed_array/nda.hpp"
#include "methods/ERI/detail/concepts.hpp"
#include "methods/SCF/lr_ibc.hpp"
#include "utilities/lr_utils.hpp"

namespace methods {
namespace solvers {

/**
 * @class lr_hf
 * @brief Handler for computing Linear Response Fock matrix from LR density matrix
 *
 * This class computes the LR Fock matrix (ΔF) from the LR density matrix (ΔDm):
 *
 *   ΔF(k) = ΔJ(k) + ΔK(k)
 *
 * where:
 *   - ΔJ: LR Hartree (Coulomb) contribution
 *   - ΔK: LR Exchange contribution
 *
 * For q=0 perturbations, ΔF represents the (k, k) block.
 * For q≠0 perturbations, ΔF represents the (k+q, k) block.
 *
 * The LR Hartree term is:
 *   ΔJ_ij(k) = Σ_k' V(q=0) · ΔDm(k')_ii · δ_{ij}
 *
 * The LR Exchange term is:
 *   ΔK_ij(k) = -Σ_k' V(k-k'+q) · ΔDm_ij(k')
 */
class lr_hf {
public:
  using mpi_context_t = utils::mpi_context_t<mpi3::communicator>;
  template<nda::Array Array_base_t>
  using sArray_t = math::shm::shared_array<Array_base_t>;
  template<nda::MemoryArray local_Array_t>
  using dArray_t = math::nda::distributed_array<local_Array_t, mpi3::communicator>;
  template<int N>
  using shape_t = std::array<long, N>;
  using Array_view_4D_t = nda::array_view<ComplexType, 4>;

  /**
   * @brief Constructor
   *
   * @param mpi        - [INPUT] MPI context
   * @param MF         - [INPUT] Mean-field object (non-owning pointer, caller manages lifetime)
   * @param q_vec      - [INPUT] Perturbation wavevector in crystal coordinates (3,)
   * @param hf_div_treatment - [INPUT] Divergence treatment of the ground-state HF
   *                   exchange ("gygi" or "ignore_g0"). Gates the Madelung K
   *                   correction exactly as hf_t does, so ΔF matches how the
   *                   unperturbed Fock was built; the caller reads it from the
   *                   checkpoint.
   */
  lr_hf(std::shared_ptr<mpi_context_t> mpi,
        const mf::MF* MF,
        nda::array<double, 1> const& q_vec,
        std::string hf_div_treatment = "gygi");

  lr_hf(lr_hf const&) = delete;
  lr_hf(lr_hf &&) = default;
  lr_hf & operator=(const lr_hf &) = delete;
  lr_hf & operator=(lr_hf &&) = delete;

  ~lr_hf() = default;

  using dArray_4D_t = dArray_t<nda::array<ComplexType, 4>>;
  using dArray_3D_t = dArray_t<nda::array<ComplexType, 3>>;

  /**
   * Static screened-exchange (HSEX) kernel: the exchange channel contracts
   * W(iν=0) = V + W_c(iν=0) in place of the bare V.
   *
   * The q→0 head rides along: the aux-basis arrays carry no G=0 component, so
   * the head enters as a separate Madelung term, and the one W(iν=0) needs is
   * -madelung·ε⁻¹(iν=0) — the same decomposition lr_gw uses, where bare
   * exchange contributes -madelung·1 and Σ_c contributes -madelung·(ε⁻¹-1).
   * `head_factor` carries that ε⁻¹(iν=0).
   */
  struct hsex_kernel_t {
    /// Which kernel the exchange contraction installs. Named after the kernel
    /// itself, since that is all that differs:
    ///   V_plus_Wc0 = V + W_c(iν=0), the HSEX kernel;
    ///   minus_Wc0  =   − W_c(iν=0), the counter-term a split run whose K_sc is
    ///                  HSEX owes its remainder, so that the two channels sum
    ///                  back to bare exchange.
    enum class kernel_e { V_plus_Wc0, minus_Wc0 };

    /// W_c(q, iν=0), (nq, NP, NP) on utils::lr_aux_kernel_pgrid(comm.size()).
    const dArray_3D_t* Wc0_qPQ;
    /// Scales the exchange Madelung term: ε⁻¹(iν=0) for V_plus_Wc0,
    /// 1 − ε⁻¹(iν=0) for minus_Wc0, so the two sum to bare exchange's 1.
    double head_factor;
    kernel_e kernel;
  };

  /**
   * @brief Compute LR Fock matrix from LR density matrix using THC-ERI
   *
   * Computes: ΔF(k) = ΔJ(k) + ΔK(k)
   *
   * @param sDeltaF_skij     - [OUTPUT] LR Fock matrix (ns, nk_ibz, nb, nb)
   * @param sDeltaDm_skij    - [INPUT] LR density matrix (ns, nk_ibz, nb, nb)
   * @param thc              - [INPUT] THC-ERI handler
   * @param S_skij           - [INPUT] Overlap matrix (ns, nk_ibz, nb, nb)
   * @param compute_hartree  - [INPUT] Whether to compute Hartree term (default: true)
   * @param compute_exchange - [INPUT] Whether to compute Exchange term (default: true)
   * @param DeltaF_PQ_out    - [OUTPUT] If non-null, a replicated copy of ΔF in
   *                           aux (PQ) basis, captured before the band-basis IBC
   *                           correction is added. Consumed by the Python phonon
   *                           post-processors (ΔΔF_ibc T1 term).
   * @param compute_xc       - [INPUT] Add the semilocal xc kernel to the direct
   *                           channel, i.e. use (V + Vxc)(q) in ΔJ. Requires a
   *                           THC carrying Vxc and compute_exchange = false; see
   *                           thc_lr_hf for why it cannot reach ΔK.
   * @param hsex             - [INPUT] Static screened exchange. Non-null replaces
   *                           the exchange kernel V(q) by V(q) + W_c(q, iν=0),
   *                           i.e. ΔK = -ΔDm ⊙ W(iν=0). The direct (ΔJ) channel
   *                           is untouched.
   */
  template<nda::MemoryArray AF_t>
  void evaluate(sArray_t<AF_t>& sDeltaF_skij,
                const sArray_t<AF_t>& sDeltaDm_skij,
                THC_ERI auto& thc,
                const nda::MemoryArrayOfRank<4> auto& S_skij,
                bool compute_hartree = true,
                bool compute_exchange = true,
                const lr_ibc_DeltaX* ibc = nullptr,
                const nda::array_view<ComplexType, 3>* DeltaV_qPQ = nullptr,
                const nda::array<ComplexType, 4>* Dm_skij_unpert = nullptr,
                nda::array<ComplexType, 4>* DeltaF_PQ_out = nullptr,
                bool compute_xc = false,
                const hsex_kernel_t* hsex = nullptr);

  /// Print the LR HF timer block (header + total + subclocks) at log level `level`.
  inline void print_timers(int level = 2) {
    app_log(level, "\n  LR_HF timers");
    app_log(level, "  ------------");
    app_log(level, "    LR Fock eval:                   {0:8.3f} sec  {1:4d} calls", _Timer.elapsed("LR_HF"), _Timer.number_of_calls("LR_HF"));
    print_subclocks(level, "    ");
    app_log(level, "");
  }

  /// Zero this solver's sub-clocks. lr_driver calls it per perturbation so a
  /// batched run's timing report covers one perturbation, not the running total.
  inline void reset_timers() { _Timer.reset(); }

  /// Print only the component clocks, each line prefixed by `indent`.
  /// Embedded (with deeper indent) in lr_driver's final hierarchical report.
  inline void print_subclocks(int level, const std::string& indent) {
    print_subclocks_all(level, indent, {this});
  }

  /// Print one ΔF component table summed over every instance in `solvers`
  /// (nulls ignored), each line prefixed by `indent`.
  ///
  /// A split kernel splits ΔF across two instances — the self-consistent channel
  /// and the perturbative one — so summing them keeps one table whatever the
  /// kernel was split into. Static rather than a member, because a run whose ΔF
  /// lives only in the perturbative channel leaves the self-consistent instance
  /// null and the table must still print.
  static void print_subclocks_all(int level, const std::string& indent,
                                  std::initializer_list<lr_hf*> solvers) {
    auto sec = [&](const char* key) {
      double v = 0.0;
      for (auto* o : solvers) if (o) v += o->_Timer.elapsed(key);
      return v;
    };
    auto cnt = [&](const char* key) {
      int v = 0;
      for (auto* o : solvers) if (o) v += o->_Timer.number_of_calls(key);
      return v;
    };
    app_log(level, "{0}  - Misc (init/cleanup):        {1:8.3f} sec  {2:4d} calls", indent, sec("MISC"), cnt("MISC"));
    app_log(level, "{0}  - Alloc:                      {1:8.3f} sec  {2:4d} calls", indent, sec("ALLOC"), cnt("ALLOC"));
    app_log(level, "{0}  - Z fetch (U(q)):             {1:8.3f} sec  {2:4d} calls", indent, sec("Z_FETCH"), cnt("Z_FETCH"));
    app_log(level, "{0}  - Primary->Aux:               {1:8.3f} sec  {2:4d} calls", indent, sec("PRIM_TO_AUX"), cnt("PRIM_TO_AUX"));
    app_log(level, "{0}  - U(q)->U(R) FT:              {1:8.3f} sec  {2:4d} calls", indent, sec("UQ_TO_UR"), cnt("UQ_TO_UR"));
    app_log(level, "{0}  - Coulomb (J):                {1:8.3f} sec  {2:4d} calls", indent, sec("COULOMB"), cnt("COULOMB"));
    app_log(level, "{0}  - Exchange (K):               {1:8.3f} sec  {2:4d} calls", indent, sec("EXCHANGE"), cnt("EXCHANGE"));
    app_log(level, "{0}  - Aux->Primary:               {1:8.3f} sec  {2:4d} calls", indent, sec("AUX_TO_PRIM"), cnt("AUX_TO_PRIM"));
    app_log(level, "{0}  - Final reduce (ΔF):          {1:8.3f} sec  {2:4d} calls", indent, sec("FINAL_REDUCE"), cnt("FINAL_REDUCE"));
    app_log(level, "{0}  - Madelung correction:        {1:8.3f} sec  {2:4d} calls", indent, sec("MADELUNG"), cnt("MADELUNG"));
  }

  // Accessors
  const nda::array<int, 1>& kpq_map() const { return _kpq_map; }
  const nda::array<double, 1>& q_vec() const { return _q_vec; }
  bool is_q_gamma() const { return _is_q_gamma; }

private:
  std::shared_ptr<mpi_context_t> _mpi;
  const mf::MF* _MF;

  int _ns;
  int _nkpts;
  int _nkpts_ibz;
  int _nbnd;
  int _npol;

  nda::array<double, 1> _q_vec;     // Perturbation wavevector (3,)
  nda::array<int, 1> _kpq_map;      // k → k+q index mapping in full BZ (nkpts,)
  bool _is_q_gamma;                 // True if q is approximately zero

  long _q_ibz_idx;                  // IBZ index of perturbation q-vector (for V(q) lookup)
  nda::array<int, 1> _kpq_ibz_map;  // IBZ k → IBZ k+q mapping (nkpts_ibz,)
  nda::array<bool, 1> _kpq_ibz_trev; // Whether k+q reaches its IBZ rep via time-reversal

  std::string _hf_div_treatment;    // "gygi" or "ignore_g0"; gates the Madelung K correction

  utils::TimerManager _Timer;

  // --- Data cached across evaluate() calls ---------------------------------
  // lr_hf is constructed per (q, THC handler) and lives for the whole LR SCF, so
  // the Coulomb kernel is the same on every call — including its HSEX form, since
  // one instance is only ever handed one `hsex` kernel. Fetching and transforming it
  // once is the trade hf_t::_dZ_cache already makes in the ground state, and the
  // caches die with the solver at the end of the run.
  //
  // Filled on first use rather than at construction: lr_hf never holds thc (it is
  // a template parameter of evaluate), and which of the three a run touches
  // depends on the per-call channel flags — V(q) only in the direct channel, U(R)
  // only in the exchange channel, Vxc only when compute_xc.

  bool _Vq_cached = false;
  nda::array<ComplexType, 2> _Vq_PQ;      // V(q)_PQ tile at _q_ibz_idx
  bool _Vxc_q_cached = false;
  nda::array<ComplexType, 2> _Vxc_q_PQ;   // Vxc(q)_PQ tile at _q_ibz_idx
  bool _U_RPQ_cached = false;
  nda::array<ComplexType, 3> _U_RPQ;      // U(R)_PQ, exchange channel
  /// The HSEX kernel _U_RPQ was built for (nullopt = bare V), checked on reuse.
  std::optional<hsex_kernel_t::kernel_e> _U_RPQ_hsex;

  /// Fill `Uq_PQ` with the direct-channel kernel V(q) (+ Vxc(q) when compute_xc)
  /// on the (P_rng, Q_rng) tile, fetching and caching the blocks on first use.
  void build_Uq_PQ(THC_ERI auto& thc, int np_P, int np_Q,
                   nda::range P_rng, nda::range Q_rng, bool compute_xc,
                   nda::array<ComplexType, 2>& Uq_PQ);

  /**
   * @brief Internal THC-based LR-HF implementation
   */
  template<nda::MemoryArray AF_t>
  void thc_lr_hf(const sArray_t<AF_t>& sDeltaDm_skij,
                 sArray_t<AF_t>& sDeltaF_skij,
                 THC_ERI auto& thc,
                 bool compute_hartree,
                 bool compute_exchange,
                 const lr_ibc_DeltaX* ibc = nullptr,
                 const nda::array_view<ComplexType, 3>* DeltaV_qPQ = nullptr,
                 const nda::array<ComplexType, 4>* Dm_skij_unpert = nullptr,
                 nda::array<ComplexType, 4>* DeltaF_PQ_out = nullptr,
                 bool compute_xc = false,
                 const hsex_kernel_t* hsex = nullptr);

  /**
   * @brief Hartree-only LR Fock, exploiting that ΔF is diagonal in the THC
   *        auxiliary basis and independent of (s,k).
   *
   * The dense path builds the full ΔDm_PQ(k), Fourier transforms it k→R and
   * transforms a full ΔF_PQ(k) back, but the Hartree channel reads only the P == Q
   * diagonal at R = 0 and writes back an operator that is δ_PQ times a k-independent
   * vector. This routine evaluates that contraction directly:
   *
   *   n_P   = (factor/nk) Σ_(s,k,p) diag_P[ X_p(k+q) ΔDm(k) X_p(k)† ]
   *   ΔJ_P  = Σ_Q [V(q) + Vxc(q)]_PQ n_Q
   *   ΔF_ij(k) = Σ_(p,P) conj(X_p(k+q)_Pi) ΔJ_P X_p(k)_Pj
   *
   * p runs over the npol diagonal polarization blocks, the only ones the direct
   * channel sees; ΔJ_P is common to all of them.
   *
   * The k-average in the first line is what the dense path's k→R transform does at
   * R = 0: k_to_R_coefficients gives f_Rk(0,k) = 1/nk exactly.
   *
   * Numerically equivalent to, but not bit-identical with, the dense path: the P sum
   * is one gemm over the full P range instead of a tiled MPI reduction, and the k sum
   * applies the same weights in a different order.
   *
   * Selected by thc_lr_hf for the Hartree-only case with no IBC and no δV; every other
   * configuration takes the dense branch, which stays the reference implementation. Extending it to IBC (three more row dots, one per DeltaX term) or
   * to δV (one more cached diagonal) is mechanical, and worth doing only if such a
   * configuration ever shows up in a profile.
   */
  template<nda::MemoryArray AF_t>
  void thc_lr_hartree_only(const sArray_t<AF_t>& sDeltaDm_skij,
                           sArray_t<AF_t>& sDeltaF_skij,
                           THC_ERI auto& thc,
                           nda::array<ComplexType, 4>* DeltaF_PQ_out,
                           bool compute_xc);

  /**
   * @brief LR finite-size correction for K based on "PRB 80, 085114(2009)"
   *
   * Computes: ΔDelta_ij = -madelung * S(k+q)_ia * ΔDm_ab * S(k)_bj
   *
   * @param sDeltaF_skij - [OUTPUT] LR Fock matrix (accumulated)
   * @param DeltaDm_skij - [INPUT] LR density matrix
   * @param S_skij       - [INPUT] Overlap matrix
   * @param madelung     - [INPUT] Madelung constant
   */
  template<nda::MemoryArray AF_t>
  void LR_HF_K_correction(sArray_t<AF_t>& sDeltaF_skij,
                          const nda::MemoryArrayOfRank<4> auto& DeltaDm_skij,
                          const nda::MemoryArrayOfRank<4> auto& S_skij,
                          double madelung);

};

} // namespace solvers
} // namespace methods

#endif // COQUI_LR_HF_HPP
