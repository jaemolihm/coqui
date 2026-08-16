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
   */
  using dArray_4D_t = dArray_t<nda::array<ComplexType, 4>>;

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
                bool compute_xc = false);

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
    app_log(level, "{0}  - Misc (init/cleanup):        {1:8.3f} sec  {2:4d} calls", indent, _Timer.elapsed("MISC"), _Timer.number_of_calls("MISC"));
    app_log(level, "{0}  - Alloc:                      {1:8.3f} sec  {2:4d} calls", indent, _Timer.elapsed("ALLOC"), _Timer.number_of_calls("ALLOC"));
    app_log(level, "{0}  - Z fetch (U(q)):             {1:8.3f} sec  {2:4d} calls", indent, _Timer.elapsed("Z_FETCH"), _Timer.number_of_calls("Z_FETCH"));
    app_log(level, "{0}  - Primary->Aux:               {1:8.3f} sec  {2:4d} calls", indent, _Timer.elapsed("PRIM_TO_AUX"), _Timer.number_of_calls("PRIM_TO_AUX"));
    app_log(level, "{0}  - U(q)->U(R) FT:              {1:8.3f} sec  {2:4d} calls", indent, _Timer.elapsed("UQ_TO_UR"), _Timer.number_of_calls("UQ_TO_UR"));
    app_log(level, "{0}  - Coulomb (J):                {1:8.3f} sec  {2:4d} calls", indent, _Timer.elapsed("COULOMB"), _Timer.number_of_calls("COULOMB"));
    app_log(level, "{0}  - Exchange (K):               {1:8.3f} sec  {2:4d} calls", indent, _Timer.elapsed("EXCHANGE"), _Timer.number_of_calls("EXCHANGE"));
    app_log(level, "{0}  - Aux->Primary:               {1:8.3f} sec  {2:4d} calls", indent, _Timer.elapsed("AUX_TO_PRIM"), _Timer.number_of_calls("AUX_TO_PRIM"));
    app_log(level, "{0}  - Madelung correction:        {1:8.3f} sec  {2:4d} calls", indent, _Timer.elapsed("MADELUNG"), _Timer.number_of_calls("MADELUNG"));
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
                 bool compute_xc = false);

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
