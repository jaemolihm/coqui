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


#include <c2py/c2py.hpp>
#include "IO/app_loggers.h"
#include "methods/MBPT_drivers.h"
#include "utilities/lr_utils.hpp"

#include "python/interaction/eri_module.hpp"
#include "python/interaction/eri_module.wrap.hxx"

namespace coqui_py {

  /**
   * @brief Unified linear response calculation
   *
   * Reads the unperturbed state from the checkpoint, runs the LR SCF loop
   *   ΔH0 → ΔG → ΔDm → [ΔF] → [ΔΣ] → ΔG → ... until convergence,
   * and writes results to the "linear_response" group of the output checkpoint.
   *
   * @param lr_params         - [INPUT] JSON string with params (prefix, output, input_type, input_iter, h0_source, div_corr)
   * @param h_int             - [INPUT] THC ERI handler
   * @param q_vec             - [INPUT] Perturbation wavevector in crystal coords (3,)
   * @param DeltaH0_skij      - [INPUT] Perturbation matrix (ns, nk, nb, nb);
   *                            required on the MPI global root, None elsewhere.
   * @param include_hartree   - [INPUT] Include ΔJ in SCF loop
   * @param include_exchange  - [INPUT] Include ΔK in SCF loop
   * @param gw_mode           - [INPUT] GW mode: "none", "fixed_W", or "full"
   * @param max_iter          - [INPUT] Maximum SCF iterations (1 = one-shot)
   * @param tol               - [INPUT] Convergence tolerance
   * @param fix_density       - [INPUT] If true, compute Δμ to enforce ΔN=0
   * @param iter_alg          - [INPUT] Iteration algorithm: "damping" or "DIIS"
   * @param mixing            - [INPUT] Damping/mixing parameter
   * @param max_subsp_size    - [INPUT] DIIS subspace size
   * @param diis_warmup       - [INPUT] DIIS warmup iterations
   * @param DeltaX_left       - [INPUT] Optional δ^q X collocation perturbation (root only)
   * @param DeltaX_right      - [INPUT] Optional δ^{-q} X collocation perturbation (root only)
   * @param DeltaV_qPQ        - [INPUT] Optional THC Coulomb perturbation δV (root only)
   * @return                  - [OUTPUT] Tuple of (niter, Delta_mu)
   */
  std::tuple<int, double> run_lr(
      const std::string &lr_params,
      ThcCoulomb &h_int,
      nda::array<double, 1> const& q_vec,
      std::optional<nda::array<ComplexType, 4>> DeltaH0_skij,
      bool include_hartree,
      bool include_exchange,
      std::string gw_mode_str,
      int max_iter,
      double tol,
      bool fix_density,
      std::string iter_alg,
      double mixing,
      int max_subsp_size,
      int diis_warmup,
      std::optional<nda::array<ComplexType, 4>> DeltaX_left,
      std::optional<nda::array<ComplexType, 4>> DeltaX_right,
      std::optional<nda::array<ComplexType, 3>> DeltaV_qPQ) {
    auto parser = InputParser(lr_params);
    methods::mb_eri_t mb_eri(h_int.get_eri());
    methods::lr_iter_params iter_params;
    iter_params.alg = iter_alg;
    iter_params.mixing = mixing;
    iter_params.max_subsp_size = static_cast<size_t>(max_subsp_size);
    iter_params.diis_warmup = static_cast<size_t>(diis_warmup);

    methods::lr_gw_update_mode gw_mode;
    if (gw_mode_str == "none") {
      gw_mode = methods::lr_gw_update_mode::none;
    } else if (gw_mode_str == "fixed_W") {
      gw_mode = methods::lr_gw_update_mode::fixed_W;
    } else if (gw_mode_str == "full") {
      gw_mode = methods::lr_gw_update_mode::full;
    } else {
      throw std::invalid_argument("run_lr: gw_mode must be 'none', 'fixed_W', or 'full', got '" + gw_mode_str + "'");
    }

    return methods::run_lr_calc(mb_eri, parser.get_root(), q_vec, DeltaH0_skij,
                                include_hartree, include_exchange, gw_mode,
                                max_iter, tol, fix_density, iter_params,
                                DeltaX_left, DeltaX_right, DeltaV_qPQ);
  }


  /**
   * @brief Compute k+q mapping for linear response calculations
   *
   * @param kpts_crys   - [INPUT] k-points in crystal coordinates (nkpts, 3)
   * @param q_vec       - [INPUT] perturbation wavevector in crystal coordinates (3,)
   * @return            - [OUTPUT] k → k+q index mapping (nkpts,)
   */
  nda::array<long, 1> calculate_kpq_map(
      nda::array<double, 2> const& kpts_crys,
      nda::array<double, 1> const& q_vec) {

    long nkpts = kpts_crys.shape(0);
    nda::array<long, 1> kpq_map(nkpts);
    utils::calculate_kpq_map(kpts_crys, q_vec, kpq_map);
    return kpq_map;
  }


  template<typename eri_handler>
  void mbpt(const std::string &solver_type, const std::string &mbpt_params, eri_handler &h_int,
            const nda::array<ComplexType, 5> &C_ksIai,
            const nda::array<long, 3> &band_window,
            const nda::array<RealType, 2> &kpts_crys,
            std::optional<std::map<std::string, nda::array<ComplexType, 5> > > local_polarizabilities) {
    auto parser = InputParser(mbpt_params);
    methods::mb_eri_t mb_eri(h_int.get_eri());
    methods::mbpt(solver_type, mb_eri, parser.get_root(),
                  C_ksIai, band_window, kpts_crys,
                  std::move(local_polarizabilities));
  }
  template void mbpt(const std::string&, const std::string&, ThcCoulomb&,
                     const nda::array<ComplexType, 5>&,
                     const nda::array<long, 3>&,
                     const nda::array<RealType, 2>&,
                     std::optional<std::map<std::string, nda::array<ComplexType, 5> > >);
  template void mbpt(const std::string&, const std::string&, CholCoulomb&,
                     const nda::array<ComplexType, 5>&,
                     const nda::array<long, 3>&,
                     const nda::array<RealType, 2>&,
                     std::optional<std::map<std::string, nda::array<ComplexType, 5> > >);


  // Pure MBPT interface without C_ksIai, band_window, kpts_crys
  template<typename eri_handler>
  void mbpt(const std::string &solver_type, const std::string &mbpt_params, eri_handler &h_int) {
    auto parser = InputParser(mbpt_params);
    methods::mb_eri_t mb_eri(h_int.get_eri());
    methods::mbpt(solver_type, mb_eri, parser.get_root());
  }

  template void mbpt(const std::string&, const std::string&, ThcCoulomb&);
  template void mbpt(const std::string&, const std::string&, CholCoulomb&);

  template<typename hf_eri_handler, typename eri_handler>
  void mbpt(const std::string &solver_type, const std::string &mbpt_params,
            eri_handler &h_int, hf_eri_handler &h_int_hf) {
    utils::check(h_int.get_mpi() == h_int_hf.get_mpi(),
                 "mbpt: h_int and h_int_hf must be on the same MPI communicator.");
    utils::check(h_int.get_mf() == h_int_hf.get_mf(),
                 "mbpt: h_int and h_int_hf must be on the same mean-field object.");
    auto parser = InputParser(mbpt_params);
    methods::mb_eri_t mb_eri(h_int_hf.get_eri(), h_int.get_eri());
    methods::mbpt(solver_type, mb_eri, parser.get_root());
  }

#define MBPT_INST(CORR, HF) \
template void mbpt(const std::string&, const std::string&, CORR&, HF&);

// All combinations of thc/chol for 2 eri slots
MBPT_INST(ThcCoulomb, ThcCoulomb)
MBPT_INST(ThcCoulomb, CholCoulomb)
MBPT_INST(CholCoulomb, ThcCoulomb)
MBPT_INST(CholCoulomb, CholCoulomb)

#undef MBPT_INST


  template<typename hartree_eri_handler, typename exchange_eri_handler, typename eri_handler>
  void mbpt(const std::string &solver_type, const std::string &mbpt_params,
            eri_handler &h_int, hartree_eri_handler &h_int_hartree, exchange_eri_handler &h_int_exchange) {
    utils::check(h_int.get_mpi() == h_int_hartree.get_mpi() and h_int.get_mpi() == h_int_exchange.get_mpi(),
                 "mbpt: h_int, h_int_hartree and h_int_exchange must be on the same MPI communicator.");
    utils::check(h_int.get_mf() == h_int_hartree.get_mf() and h_int.get_mf() == h_int_exchange.get_mf(),
                 "mbpt: h_int, h_int_hartree and h_int_exchange must be on the same mean-field object.");

    auto parser = InputParser(mbpt_params);
    methods::mb_eri_t mb_eri(h_int_hartree.get_eri(), h_int_exchange.get_eri(), h_int.get_eri());
    methods::mbpt(solver_type, mb_eri, parser.get_root());
  }

#define MBPT_INST(CORR, HARTREE, EXCHANGE) \
template void mbpt(const std::string&, const std::string&, CORR&, HARTREE&, EXCHANGE&);

// All combinations of thc/chol for 3 eri slots
MBPT_INST(ThcCoulomb, ThcCoulomb, ThcCoulomb)
MBPT_INST(ThcCoulomb, ThcCoulomb, CholCoulomb)
MBPT_INST(ThcCoulomb, CholCoulomb, ThcCoulomb)
MBPT_INST(ThcCoulomb, CholCoulomb, CholCoulomb)
MBPT_INST(CholCoulomb, ThcCoulomb, ThcCoulomb)
MBPT_INST(CholCoulomb, ThcCoulomb, CholCoulomb)
MBPT_INST(CholCoulomb, CholCoulomb, ThcCoulomb)
MBPT_INST(CholCoulomb, CholCoulomb, CholCoulomb)

#undef MBPT_INST

} // coqui_py

#include "mbpt_module.wrap.cxx"
