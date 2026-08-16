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


#ifndef COQUI_CHKPT_UTILS_H
#define COQUI_CHKPT_UTILS_H

#include <optional>

#include "configuration.hpp"
#include "mpi3/communicator.hpp"

#include "numerics/shared_array/nda.hpp"
#include "mean_field/MF.hpp"
#include "numerics/imag_axes_ft/IAFT.hpp"

namespace methods {
  namespace chkpt {
  template<nda::Array Array_base_t>
  using sArray_t = math::shm::shared_array<Array_base_t>;
  using Array_view_2D_t = nda::array_view<ComplexType, 2>;
  using Array_view_3D_t = nda::array_view<ComplexType, 3>;
  using Array_view_4D_t = nda::array_view<ComplexType, 4>;
  using Array_view_5D_t = nda::array_view<ComplexType, 5>;

  /**
   * Write metadata to a SCF checkpoint file.
   * @param comm     - [INPUT] MPI communicator
   * @param mf       - [INPUT] Mean-field instance
   * @param ft       - [INPUT] Imaginary Fourier transform driver
   * @param sH0_skij - [INPUT] Non-interacting Hamiltonian (nspins, nkpts, nbnds, nbnds)
   * @param sS_skij  - [INPUT] Overlap matrices for the primary basis (nspins, nkpts, nbnds, nbnds)
   * @param output   - [INPUT] Prefix for the checkpoint file: output.mbpt.h5.
   */
  template<typename communicator_t>
  void write_metadata(communicator_t &comm, const mf::MF &mf, const imag_axes_ft::IAFT &ft,
                       const sArray_t<Array_view_4D_t> &sH0_skij, const sArray_t<Array_view_4D_t> &sS_skij,
                       std::string output);

  template <typename communicator_t, typename X_t, typename Xt_t>
  void dump_scf(communicator_t &comm, long iter,
                const X_t &Dm, const Xt_t &G, const X_t &F, const Xt_t &Sigma,
                double mu, std::string output = "bdft",
                std::string input_grp="scf", long input_iter=-1,
                bool slim=false, const X_t *K=nullptr);

  template<typename communicator_t, typename X_4D_t, typename X_3D_t>
  void dump_scf(communicator_t &comm, long iter,
                const X_4D_t &Dm_skij, const X_4D_t &Heff_skij,
                const X_4D_t &MO_skia, const X_3D_t &E_ska,
                double mu, std::string output = "bdft");

  template<typename X_t, typename Xt_t>
  long read_scf(mpi3::shared_communicator node_comm,
                X_t &F, Xt_t &Sigma, double &mu,
                std::string output, std::string h5_grp="scf", long iter=-1);

  template<typename shared_array_t>
  void read_H0(mpi3::shared_communicator node_comm, std::string output, shared_array_t &H0);

  template<typename shared_array_t>
  void read_ovlp(mpi3::shared_communicator node_comm, std::string output, shared_array_t &S);

  template<typename shared_array_t>
  void read_dm(mpi3::shared_communicator node_comm, std::string output, long iter, shared_array_t &Dm);

  template<typename X_4D_t>
  long read_qpscf(mpi3::shared_communicator node_comm,
                  X_4D_t &Heff_skij, double &mu, std::string output);

  /**
   * Read the frozen QP eigenbasis (MO coefficients + energies) stored by a qpGW
   * run in "scf/iter{iter}" (datasets MO_skia / E_ska; see the dump_scf qp
   * overload). Fills the shared arrays sMO_skia (ns,nk,nb,nb) and sE_ska
   * (ns,nk,nb). iter=-1 selects final_iter. Errors if the datasets are absent.
   */
  template<typename X_4D_t, typename X_3D_t>
  void read_qp_MOs(mpi3::shared_communicator node_comm,
                   X_4D_t &sMO_skia, X_3D_t &sE_ska,
                   std::string output, std::string h5_grp="scf", long iter=-1);

  template<typename X_4D_t, typename X_3D_t>
  void write_qpgw_results(std::string filename, long gw_iter,
                          const X_3D_t &sE_ska,
                          const X_4D_t &sMO_skia,
                          const X_4D_t &sVcorr_skij,
                          double mu);

  template<typename X_4D_t>
  void read_qp_hamilt_components(X_4D_t &Vhf_skij,
                                 X_4D_t &Vcorr_skij,
                                 double &mu,
                                 std::string filename,
                                 long gw_iter);

  /**
   * Write the SCF run status into the "scf" group of output.mbpt.h5.
   *
   * Called once, after the SCF loop has exited, so a checkpoint carrying no
   * "scf_status" dataset comes from a run that was killed mid-flight (or from a
   * run that predates the marker). Readers must use has_dataset and treat the
   * absence as unknown; see read_scf_status.
   *
   * @param comm   - [INPUT] MPI communicator; only the root writes
   * @param output - [INPUT] Prefix for the checkpoint file: output.mbpt.h5
   * @param status - [INPUT] "converged" or "max_iter"
   */
  void write_scf_status(mpi3::communicator &comm, std::string output, std::string status);

  /// SCF run status stored by write_scf_status; "unknown" when absent.
  std::string read_scf_status(std::string filename);

  auto read_input_iterations(std::string filename) -> std::tuple<long, long, long, long>;

  bool is_qp_selfenergy(std::string filename);

  bool read_sigma_local(nda::array<ComplexType, 5> &Sigma_imp_wsIab,
                        nda::array<ComplexType, 4> &Vhf_imp_sIab,
                        std::string filename, long weiss_f_iter);

  bool read_sigma_local(nda::array<ComplexType, 5> &Sigma_imp_wsIab,
                        nda::array<ComplexType, 4> &Vcorr_dc_sIab,
                        nda::array<ComplexType, 4> &Vhf_imp_sIab,
                        nda::array<ComplexType, 4> &Vhf_dc_sIab,
                        std::string filename, long weiss_f_iter);

  bool read_sigma_local(nda::array<ComplexType, 5> &Sigma_imp_wsIab,
                        nda::array<ComplexType, 5> &Sigma_dc_wsIab,
                        nda::array<ComplexType, 4> &Vhf_imp_sIab,
                        nda::array<ComplexType, 4> &Vhf_dc_sIab,
                        std::string filename, long weiss_f_iter);

  template<typename shared_array_t>
  bool read_pi_local(shared_array_t &sPi_imp, shared_array_t &sPi_dc,
                     std::string filename, long weiss_b_iter=-1);

  /**
   * Read linear response perturbation ΔH0 from HDF5 file.
   *
   * Expected HDF5 structure:
   *   /linear_response/
   *     q_vec              # (3,) perturbation wavevector in crystal coords
   *     DeltaH0_skij       # (ns, nk, nb, nb) complex perturbation matrix
   *
   * @param comm           - [INPUT] MPI communicator (shared or regular)
   * @param filename       - [INPUT] HDF5 file path
   * @param q_vec          - [OUTPUT] perturbation wavevector
   * @param sDeltaH0_skij  - [OUTPUT] perturbation matrix
   * @return true if read successful, false otherwise
   */
  template<typename communicator_t, typename shared_array_t>
  bool read_DeltaH0(communicator_t& comm,
                    std::string filename,
                    nda::array<double, 1>& q_vec,
                    shared_array_t& sDeltaH0_skij);

  /**
   * Write linear response perturbation ΔH0 to HDF5 file.
   *
   * @param comm           - [INPUT] MPI communicator
   * @param filename       - [INPUT] HDF5 file path
   * @param q_vec          - [INPUT] perturbation wavevector
   * @param sDeltaH0_skij  - [INPUT] perturbation matrix
   */
  template<typename communicator_t, typename shared_array_t>
  void write_DeltaH0(communicator_t& comm,
                     std::string filename,
                     nda::array<double, 1> const& q_vec,
                     shared_array_t const& sDeltaH0_skij);

  /**
   * Write unified LR results to HDF5 checkpoint file.
   *
   * Always writes ΔG and ΔDm. Writes ΔF only if include_hartree || include_exchange.
   * Writes ΔΣ only if include_gw_sigma.
   *
   * @param comm              - [INPUT] MPI communicator
   * @param filename          - [INPUT] HDF5 file path (e.g., "coqui.mbpt.h5")
   * @param q_vec             - [INPUT] perturbation wavevector
   * @param sDeltaG_tskij     - [INPUT] LR Green's function
   * @param sDeltaDm_skij     - [INPUT] LR density matrix
   * @param sDeltaF_skij      - [INPUT] LR Fock matrix
   * @param sDeltaSigma_tskij - [INPUT] LR self-energy (nullptr if not used)
   * @param Delta_mu          - [INPUT] chemical potential shift
   * @param niter             - [INPUT] number of SCF iterations
   * @param include_hartree   - [INPUT] whether Hartree was included
   * @param include_exchange  - [INPUT] whether Exchange was included
   * @param include_gw_sigma  - [INPUT] whether GW self-energy was included
   * @param sDeltaSigma2_tskij - [INPUT] optional G0·dW0 term for split-term
   *   one-shot G0W0 output. When non-null, sDeltaSigma_tskij holds the total ΔΣ
   *   and this holds the G0·dW0 piece, written as "DeltaSigma_GdW_tskij".
   *   Nullptr for the standard fused output.
   * @param imode          - [INPUT] perturbation index. Unset (default) writes
   *   "linear_response/" exactly as before; set writes
   *   "linear_response/mode{imode}/", for a run covering several perturbations.
   * @param save_DeltaG    - [INPUT] write DeltaG_tskij (default true). ΔG is the
   *   single largest LR dataset, so a batched run that only needs ΔDm/ΔF can drop
   *   it. The one reader, python `coqui.mbpt.read_lr_results`, then returns None
   *   for it.
   * @param nbnd_save      - [INPUT] keep only the leading nbnd_save x nbnd_save
   *   band block of the imaginary-time arrays (DeltaG_tskij, DeltaSigma_tskij,
   *   DeltaSigma_GdW_tskij); unset (default) writes them whole. Must be in
   *   [0, nbnd]. Each trimmed dataset carries an "nbnd_save" HDF5 attribute —
   *   the only thing on disk that distinguishes a protected-band block from a
   *   full-basis array, so any reader must check for it. The one-time-slice
   *   arrays (DeltaDm_skij, DeltaF_skij, DeltaVcorr_skij) are never trimmed.
   * @param gw_mode        - [INPUT] GW self-energy mode of the kernel that was
   *   applied ("none"/"fixed_W"/"full"), written whenever non-empty. It refines
   *   include_gw_sigma, which cannot tell fixed_W from full.
   * @param lr_two_step        - [INPUT] whether the split-kernel schedule was
   *   used. When true the stored ΔF/ΔΣ are the sums of the two channels, with
   *   the perturbative part evaluated at the previous stage's ΔG — NOT a
   *   self-consistent response to the ΔG that was written — so the schedule is
   *   persisted alongside them for downstream readers.
   */
  template<typename communicator_t, typename G_t, typename Dm_t, typename F_t, typename Sigma_t>
  void dump_lr(communicator_t& comm,
               std::string filename,
               nda::array<double, 1> const& q_vec,
               G_t const& sDeltaG_tskij,
               Dm_t const& sDeltaDm_skij,
               F_t const& sDeltaF_skij,
               Sigma_t const* sDeltaSigma_tskij,
               double Delta_mu,
               int niter,
               bool include_hartree,
               bool include_exchange,
               bool include_gw_sigma,
               Sigma_t const* sDeltaSigma2_tskij = nullptr,
               F_t const* sDeltaVcorr_skij = nullptr,
               std::optional<long> imode = std::nullopt,
               bool save_DeltaG = true,
               std::optional<long> nbnd_save = std::nullopt,
               std::string const& gw_mode = "",
               bool lr_two_step = false,
               std::string const& two_step_inner_method = "",
               int two_step_order = 0,
               bool two_step_outer_accel = false,
               std::string const& two_step_outer_alg = "",
               double two_step_outer_tol = 0.0,
               int two_step_stages_applied = 0);

  /**
   * Write the qpGW analytic-continuation parameters into the SCF checkpoint
   * ("scf/qp_params"), so a later LR-qpGW run can statify ΔΣ with exactly the
   * parameters the unperturbed qpGW run used. Also stashes div_treatment on the
   * top-level "scf" group so the divergence head is reconstructed consistently.
   */
  template<typename communicator_t>
  void dump_qp_params(communicator_t& comm, std::string filename,
                      std::string const& off_diag_mode, double eta,
                      std::string const& ac_alg, int Nfit,
                      std::string const& div_treatment);

  /**
   * Read the qpGW AC parameters written by dump_qp_params. Returns true and
   * fills the outputs if "scf/qp_params" is present; returns false (leaving
   * the outputs untouched) otherwise.
   */
  template<typename communicator_t>
  bool read_qp_params(communicator_t& comm, std::string filename,
                      std::string& off_diag_mode, double& eta,
                      std::string& ac_alg, int& Nfit);

  }; // chkpt

} // methods

#endif //COQUI_CHKPT_UTILS_H
