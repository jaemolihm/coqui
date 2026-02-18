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


#include <c2py/c2py.hpp>
#include "IO/app_loggers.h"
#include "methods/MBPT_drivers.h"
#include "methods/HF/lr_hf.hpp"
#include "methods/HF/hf_t.h"
#include "utilities/lr_utils.hpp"
#include "numerics/shared_array/nda.hpp"

#include "python/interaction/eri_module.hpp"
#include "python/interaction/eri_module.wrap.hxx"

namespace coqui_py {

  /**
   * @brief Unified linear response calculation
   *
   * This calls the C++ run_lr_calc function which:
   * 1. Reads unperturbed G from checkpoint
   * 2. Runs the LR SCF loop with configurable components:
   *    ΔH0 → ΔG → ΔDm → [ΔF] → [ΔΣ] → ΔG → ... until convergence
   * 3. Writes results to checkpoint
   *
   * @param lr_params         - [INPUT] JSON string with params (prefix, output)
   * @param h_int             - [INPUT] THC ERI handler
   * @param q_vec             - [INPUT] Perturbation wavevector (3,)
   * @param DeltaH0_skij      - [INPUT] Perturbation matrix (ns, nk, nb, nb)
   * @param include_hartree   - [INPUT] Include ΔJ in SCF loop
   * @param include_exchange  - [INPUT] Include ΔK in SCF loop
   * @param include_gw_sigma  - [INPUT] Include ΔΣ_GW in SCF loop
   * @param max_iter          - [INPUT] Maximum SCF iterations (1 = one-shot)
   * @param tol               - [INPUT] Convergence tolerance
   * @param fix_density       - [INPUT] If true, compute Δμ to enforce ΔN=0
   * @param iter_alg          - [INPUT] Iteration algorithm: "damping" or "DIIS"
   * @param mixing            - [INPUT] Damping/mixing parameter
   * @param max_subsp_size    - [INPUT] DIIS subspace size
   * @param diis_warmup       - [INPUT] DIIS warmup iterations
   * @return                  - [OUTPUT] Tuple of (niter, Delta_mu)
   */
  std::tuple<int, double> run_lr(
      const std::string &lr_params,
      ThcCoulomb &h_int,
      nda::array<double, 1> const& q_vec,
      nda::array<ComplexType, 4> const& DeltaH0_skij,
      bool include_hartree,
      bool include_exchange,
      bool include_gw_sigma,
      int max_iter,
      double tol,
      bool fix_density,
      std::string iter_alg,
      double mixing,
      int max_subsp_size,
      int diis_warmup) {
    auto parser = InputParser(lr_params);
    methods::mb_eri_t mb_eri(h_int.get_eri());
    methods::lr_iter_params iter_params;
    iter_params.alg = iter_alg;
    iter_params.mixing = mixing;
    iter_params.max_subsp_size = static_cast<size_t>(max_subsp_size);
    iter_params.diis_warmup = static_cast<size_t>(diis_warmup);
    return methods::run_lr_calc(mb_eri, parser.get_root(), q_vec, DeltaH0_skij,
                                include_hartree, include_exchange, include_gw_sigma,
                                max_iter, tol, fix_density, iter_params);
  }


  /**
   * @brief Compute LR Fock matrix from LR density matrix
   *
   * This function computes ΔF = ΔJ + ΔK from the LR density matrix ΔDm.
   * Used for testing the LR-HF implementation (Step 2.1 of Phase 2).
   *
   * @param h_int           - [INPUT] THC ERI handler
   * @param q_vec           - [INPUT] Perturbation wavevector (3,)
   * @param DeltaDm_skij    - [INPUT] LR density matrix (ns, nk, nb, nb)
   * @param S_skij          - [INPUT] Overlap matrix (ns, nk, nb, nb)
   * @param compute_hartree - [INPUT] Whether to compute Hartree term
   * @param compute_exchange - [INPUT] Whether to compute Exchange term
   * @return                - [OUTPUT] LR Fock matrix (ns, nk, nb, nb)
   */
  nda::array<ComplexType, 4> lr_hf(
      ThcCoulomb& h_int,
      nda::array<double, 1> const& q_vec,
      nda::array<ComplexType, 4> const& DeltaDm_skij,
      nda::array<ComplexType, 4> const& S_skij,
      bool compute_hartree,
      bool compute_exchange) {

    using Array_view_4D_t = nda::array_view<ComplexType, 4>;
    using sArray_t = math::shm::shared_array<Array_view_4D_t>;

    auto& thc = h_int.get_eri();
    auto mf = thc.MF();
    auto mpi = thc.mpi();

    long ns = DeltaDm_skij.shape(0);
    long nkpts_ibz = DeltaDm_skij.shape(1);
    long nbnd = DeltaDm_skij.shape(2);

    // Create shared arrays
    auto sDeltaDm_skij = math::shm::make_shared_array<Array_view_4D_t>(
        *mpi, {ns, nkpts_ibz, nbnd, nbnd});
    auto sDeltaF_skij = math::shm::make_shared_array<Array_view_4D_t>(
        *mpi, {ns, nkpts_ibz, nbnd, nbnd});

    // Copy input to shared array
    if (mpi->node_comm.root()) {
      sDeltaDm_skij.local() = DeltaDm_skij;
    }
    mpi->comm.barrier();

    // Create lr_hf solver and compute ΔF
    methods::solvers::lr_hf lr_hf_solver(mpi, mf.get(), q_vec);
    lr_hf_solver.evaluate(sDeltaF_skij, sDeltaDm_skij, thc, S_skij,
                          compute_hartree, compute_exchange);

    // Copy result to output array
    nda::array<ComplexType, 4> DeltaF_skij(ns, nkpts_ibz, nbnd, nbnd);
    if (mpi->node_comm.root()) {
      DeltaF_skij = sDeltaF_skij.local();
    }
    mpi->comm.barrier();

    return DeltaF_skij;
  }


  /**
   * @brief Compute HF self-energy (Fock matrix) from density matrix
   *
   * This function computes F = J + K from the density matrix Dm.
   * Temporarily exposed for linear-response debugging.
   *
   * @param h_int           - [INPUT] THC or Cholesky ERI handler
   * @param Dm_skij         - [INPUT] Density matrix (ns, nk, nb, nb)
   * @param S_skij          - [INPUT] Overlap matrix (ns, nk, nb, nb)
   * @param compute_hartree - [INPUT] Whether to compute Hartree term
   * @param compute_exchange - [INPUT] Whether to compute Exchange term
   * @return                - [OUTPUT] Fock matrix (ns, nk, nb, nb)
   */
  template<typename eri_handler>
  nda::array<ComplexType, 4> hf_evaluate(
      eri_handler& h_int,
      nda::array<ComplexType, 4> const& Dm_skij,
      nda::array<ComplexType, 4> const& S_skij,
      bool compute_hartree,
      bool compute_exchange) {

    using Array_view_4D_t = nda::array_view<ComplexType, 4>;
    using sArray_t = math::shm::shared_array<Array_view_4D_t>;

    auto& eri = h_int.get_eri();
    auto mpi = eri.mpi();

    long ns = Dm_skij.shape(0);
    long nkpts_ibz = Dm_skij.shape(1);
    long nbnd = Dm_skij.shape(2);

    // Create shared arrays for output
    auto sF_skij = math::shm::make_shared_array<Array_view_4D_t>(
        *mpi, {ns, nkpts_ibz, nbnd, nbnd});

    // Initialize output to zero
    if (mpi->node_comm.root()) {
      sF_skij.local() = ComplexType(0.0, 0.0);
    }
    mpi->comm.barrier();

    // Create hf_t solver and compute F
    methods::solvers::hf_t hf_solver;
    hf_solver.evaluate(sF_skij, Dm_skij, eri, S_skij,
                       compute_hartree, compute_exchange);

    // Copy result to output array
    nda::array<ComplexType, 4> F_skij(ns, nkpts_ibz, nbnd, nbnd);
    if (mpi->node_comm.root()) {
      F_skij = sF_skij.local();
    }
    mpi->comm.barrier();

    return F_skij;
  }

  template nda::array<ComplexType, 4> hf_evaluate(
      ThcCoulomb&,
      nda::array<ComplexType, 4> const&,
      nda::array<ComplexType, 4> const&,
      bool, bool);
  template nda::array<ComplexType, 4> hf_evaluate(
      CholCoulomb&,
      nda::array<ComplexType, 4> const&,
      nda::array<ComplexType, 4> const&,
      bool, bool);


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


  /**
   * @brief Compute LR GW self-energy ΔΣ = -ΔG ⊙ W_c + div_corr (fixed W, R-space)
   *
   * @param lr_params     - [INPUT] JSON string with params (prefix)
   * @param h_int         - [INPUT] THC ERI handler
   * @param q_pert        - [INPUT] LR perturbation wavevector (3,)
   * @param DeltaG_tskij  - [INPUT] LR Green's function (nt, ns, nk, nb, nb)
   * @return              - [OUTPUT] ΔΣ (nt, ns, nk, nb, nb)
   */
  nda::array<ComplexType, 5> lr_gw_sigma(
      const std::string &lr_params,
      ThcCoulomb &h_int,
      nda::array<double, 1> const& q_pert,
      nda::array<ComplexType, 5> const& DeltaG_tskij) {
    auto parser = InputParser(lr_params);
    methods::mb_eri_t mb_eri(h_int.get_eri());
    return methods::lr_gw_sigma_calc(mb_eri, parser.get_root(), q_pert, DeltaG_tskij);
  }


  /**
   * @brief Evaluate GW self-energy Σ = -G ⊙ W_c + div_corr using W from file
   *
   * @param lr_params  - [INPUT] JSON string with params (prefix)
   * @param h_int      - [INPUT] THC ERI handler
   * @param G_tskij    - [INPUT] Green's function (nt, ns, nk, nb, nb)
   * @return           - [OUTPUT] Σ (nt, ns, nk, nb, nb)
   */
  nda::array<ComplexType, 5> gw_evaluate_sigma(
      const std::string &lr_params,
      ThcCoulomb &h_int,
      nda::array<ComplexType, 5> const& G_tskij) {
    auto parser = InputParser(lr_params);
    methods::mb_eri_t mb_eri(h_int.get_eri());
    return methods::gw_evaluate_sigma_calc(mb_eri, parser.get_root(), G_tskij);
  }


  /**
   * @brief Compute LR polarization ΔP = -ΔG·G - G·ΔG (R-space)
   *
   * @param h_int          - [INPUT] THC ERI handler
   * @param q_pert         - [INPUT] LR perturbation wavevector (3,)
   * @param G_tskij        - [INPUT] Unperturbed Green's function (nt, ns, nk, nb, nb)
   * @param DeltaG_tskij   - [INPUT] LR Green's function (nt, ns, nk, nb, nb)
   * @return               - [OUTPUT] ΔP (nt_half, nkpts, NP, NP)
   */
  nda::array<ComplexType, 4> lr_gw_Pi(
      ThcCoulomb &h_int,
      nda::array<double, 1> const& q_pert,
      nda::array<ComplexType, 5> const& G_tskij,
      nda::array<ComplexType, 5> const& DeltaG_tskij) {
    methods::mb_eri_t mb_eri(h_int.get_eri());
    return methods::lr_gw_Pi_calc(mb_eri, q_pert, G_tskij, DeltaG_tskij);
  }


  /**
   * @brief Evaluate standard RPA polarization P[G] (FD helper)
   *
   * @param lr_params  - [INPUT] JSON string with params (prefix)
   * @param h_int      - [INPUT] THC ERI handler
   * @param G_tskij    - [INPUT] Green's function (nt, ns, nk, nb, nb)
   * @return           - [OUTPUT] P (nt_half, nkpts, NP, NP)
   */
  nda::array<ComplexType, 4> gw_evaluate_Pi(
      const std::string &lr_params,
      ThcCoulomb &h_int,
      nda::array<ComplexType, 5> const& G_tskij) {
    auto parser = InputParser(lr_params);
    methods::mb_eri_t mb_eri(h_int.get_eri());
    return methods::gw_evaluate_Pi_calc(mb_eri, parser.get_root(), G_tskij);
  }


  /**
   * @brief Compute LR screened interaction ΔW = (Z+W_c) · ΔΠ · (Z+W_c)
   *
   * @param lr_params      - [INPUT] JSON string with params (prefix)
   * @param h_int          - [INPUT] THC ERI handler
   * @param q_pert         - [INPUT] LR perturbation wavevector (3,)
   * @param DeltaPi_tqPQ   - [INPUT] LR polarization (nt_half, nkpts, NP, NP)
   * @return               - [OUTPUT] ΔW_c (nt_half, nkpts, NP, NP)
   */
  nda::array<ComplexType, 4> lr_gw_W(
      const std::string &lr_params,
      ThcCoulomb &h_int,
      nda::array<double, 1> const& q_pert,
      nda::array<ComplexType, 4> const& DeltaPi_tqPQ) {
    auto parser = InputParser(lr_params);
    methods::mb_eri_t mb_eri(h_int.get_eri());
    return methods::lr_gw_W_calc(mb_eri, parser.get_root(), q_pert, DeltaPi_tqPQ);
  }


  /**
   * @brief Evaluate W_c from Π via W Dyson equation (FD helper)
   *
   * @param lr_params  - [INPUT] JSON string with params (prefix)
   * @param h_int      - [INPUT] THC ERI handler
   * @param Pi_tqPQ    - [INPUT] Polarization (nt_half, nkpts, NP, NP)
   * @return           - [OUTPUT] W_c (nt_half, nkpts, NP, NP)
   */
  nda::array<ComplexType, 4> gw_evaluate_W_from_Pi(
      const std::string &lr_params,
      ThcCoulomb &h_int,
      nda::array<ComplexType, 4> const& Pi_tqPQ) {
    auto parser = InputParser(lr_params);
    methods::mb_eri_t mb_eri(h_int.get_eri());
    return methods::gw_evaluate_W_from_Pi_calc(mb_eri, parser.get_root(), Pi_tqPQ);
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
