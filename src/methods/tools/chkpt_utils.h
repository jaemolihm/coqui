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


#ifndef COQUI_CHKPT_UTILS_H
#define COQUI_CHKPT_UTILS_H

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
                double mu, std::string output = "bdft");

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
   * Write LR Dyson results to HDF5 checkpoint file.
   *
   * @param comm              - [INPUT] MPI communicator
   * @param filename          - [INPUT] HDF5 file path (e.g., "coqui.mbpt.h5")
   * @param q_vec             - [INPUT] perturbation wavevector
   * @param sDeltaG_tskij     - [INPUT] LR Green's function
   * @param sDeltaDm_skij     - [INPUT] LR density matrix
   */
  template<typename communicator_t, typename G_t, typename Dm_t>
  void dump_lr_dyson(communicator_t& comm,
                     std::string filename,
                     nda::array<double, 1> const& q_vec,
                     G_t const& sDeltaG_tskij,
                     Dm_t const& sDeltaDm_skij);

  }; // chkpt

} // methods

#endif //COQUI_CHKPT_UTILS_H
