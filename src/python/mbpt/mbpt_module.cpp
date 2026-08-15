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
#include "methods/HF/lr_hf.hpp"
#include "methods/HF/hf_t.h"
#include "methods/SCF/scf_common.hpp"
#include "numerics/imag_axes_ft/iaft_utils.hpp"
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
   * @param DeltaH0_mskij     - [INPUT] Perturbations (nmodes, ns, nk, nb, nb);
   *                            required on the MPI global root, None elsewhere.
   *                            All share the one q_vec; each is written to its
   *                            own "linear_response/mode{m}" group when nmodes > 1.
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
   * @return                  - [OUTPUT] Tuple of per-mode (niter, Delta_mu) arrays
   */
  std::tuple<nda::array<long, 1>, nda::array<double, 1>> run_lr(
      const std::string &lr_params,
      ThcCoulomb &h_int,
      nda::array<double, 1> const& q_vec,
      std::optional<nda::array<ComplexType, 5>> DeltaH0_mskij,
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

    return methods::run_lr_calc(mb_eri, parser.get_root(), q_vec, DeltaH0_mskij,
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


  /**
   * @brief Compute LR Fock matrix from LR density matrix
   *
   * Computes ΔF = ΔJ + ΔK from the LR density matrix ΔDm.
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

    auto& thc = h_int.get_eri();
    auto mf = thc.MF();
    auto mpi = thc.mpi();

    long ns = DeltaDm_skij.shape(0);
    long nkpts_ibz = DeltaDm_skij.shape(1);
    long nbnd = DeltaDm_skij.shape(2);

    auto sDeltaDm_skij = math::shm::make_shared_array<Array_view_4D_t>(
        *mpi, {ns, nkpts_ibz, nbnd, nbnd});
    auto sDeltaF_skij = math::shm::make_shared_array<Array_view_4D_t>(
        *mpi, {ns, nkpts_ibz, nbnd, nbnd});

    if (mpi->node_comm.root()) {
      sDeltaDm_skij.local() = DeltaDm_skij;
    }
    mpi->comm.barrier();

    methods::solvers::lr_hf lr_hf_solver(mpi, mf.get(), q_vec);
    lr_hf_solver.evaluate(sDeltaF_skij, sDeltaDm_skij, thc, S_skij,
                          compute_hartree, compute_exchange);

    // Shared memory is visible to all ranks on a node
    nda::array<ComplexType, 4> DeltaF_skij(ns, nkpts_ibz, nbnd, nbnd);
    DeltaF_skij = sDeltaF_skij.local();
    mpi->comm.barrier();

    return DeltaF_skij;
  }


  /**
   * @brief Statify a dynamic LR self-energy ΔΣ into a static ΔV_QPGW (test API)
   *
   * Python entry point for methods::lr_qp_approx (the q-aware LR-qpGW static
   * map). Wraps the numpy inputs into node-shared arrays, reads the IAFT from
   * the checkpoint, builds a qp_params_t from the AC parameters, and returns the
   * resulting static ΔV_QPGW(k) in the primary basis.
   *
   * @param h_int           - [INPUT] THC ERI handler (source of MPI + MF)
   * @param prefix          - [INPUT] checkpoint prefix; IAFT read from prefix.mbpt.h5
   * @param DeltaSigma_tskij - [INPUT] dynamic ΔΣ_k(τ), (nt, ns, nk, nb, nb)
   * @param MO_skia         - [INPUT] frozen QP MO coefficients C, (ns, nk, nb, nb)
   * @param E_ska           - [INPUT] frozen QP energies ε, (ns, nk, nb)
   * @param mu              - [INPUT] frozen chemical potential
   * @param kpq_map         - [INPUT] k → k+q index map, (nk,)
   * @param q_is_gamma      - [INPUT] whether q ≈ 0 (enables Hermitization)
   * @param off_diag_mode   - [INPUT] "qp_energy" or "fermi"
   * @param ac_alg          - [INPUT] analytic-continuation algorithm (e.g. "pade")
   * @param Nfit            - [INPUT] # of AC fit parameters
   * @param eta             - [INPUT] AC broadening
   * @return                - [OUTPUT] static ΔV_QPGW(k), (ns, nk, nb, nb)
   */
  nda::array<ComplexType, 4> lr_qp_approx(
      ThcCoulomb& h_int,
      const std::string& prefix,
      nda::array<ComplexType, 5> const& DeltaSigma_tskij,
      nda::array<ComplexType, 4> const& MO_skia,
      nda::array<ComplexType, 3> const& E_ska,
      double mu,
      nda::array<long, 1> const& kpq_map,
      bool q_is_gamma,
      std::string off_diag_mode,
      std::string ac_alg,
      int Nfit,
      double eta) {

    using Array_view_5D_t = nda::array_view<ComplexType, 5>;
    using Array_view_4D_t = nda::array_view<ComplexType, 4>;
    using Array_view_3D_t = nda::array_view<ComplexType, 3>;

    auto& thc = h_int.get_eri();
    auto mpi = thc.mpi();

    long nt = DeltaSigma_tskij.shape(0);
    long ns = DeltaSigma_tskij.shape(1);
    long nk = DeltaSigma_tskij.shape(2);
    long nb = DeltaSigma_tskij.shape(3);

    auto sDeltaSigma_tskij = math::shm::make_shared_array<Array_view_5D_t>(
        *mpi, {nt, ns, nk, nb, nb});
    auto sMO_skia = math::shm::make_shared_array<Array_view_4D_t>(
        *mpi, {ns, nk, nb, nb});
    auto sE_ska = math::shm::make_shared_array<Array_view_3D_t>(
        *mpi, {ns, nk, nb});

    sDeltaSigma_tskij.win().fence();
    sMO_skia.win().fence();
    sE_ska.win().fence();
    if (mpi->node_comm.root()) {
      sDeltaSigma_tskij.local() = DeltaSigma_tskij;
      sMO_skia.local() = MO_skia;
      sE_ska.local() = E_ska;
    }
    sDeltaSigma_tskij.win().fence();
    sMO_skia.win().fence();
    sE_ska.win().fence();
    mpi->comm.barrier();

    // lr_qp_approx takes a 32-bit kpq_map; the exposed calculate_kpq_map returns
    // int64, so narrow it here.
    nda::array<int, 1> kpq_int(kpq_map.shape(0));
    for (long i = 0; i < kpq_map.shape(0); ++i)
      kpq_int(i) = static_cast<int>(kpq_map(i));

    imag_axes_ft::IAFT ft(imag_axes_ft::read_iaft(prefix + ".mbpt.h5", false));
    methods::qp_params_t qp_params{"sc", ac_alg, Nfit, eta, 1e-8, "qpscf", false, off_diag_mode};

    auto sDeltaVcorr_skij = methods::lr_qp_approx(
        sDeltaSigma_tskij, sMO_skia, sE_ska, mu, kpq_int, q_is_gamma, ft, qp_params);

    nda::array<ComplexType, 4> DeltaVcorr_skij(ns, nk, nb, nb);
    DeltaVcorr_skij = sDeltaVcorr_skij.local();
    mpi->comm.barrier();

    return DeltaVcorr_skij;
  }


  /**
   * @brief Compute HF self-energy (Fock matrix) from a density matrix
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

    auto& eri = h_int.get_eri();
    auto mpi = eri.mpi();

    long ns = Dm_skij.shape(0);
    long nkpts_ibz = Dm_skij.shape(1);
    long nbnd = Dm_skij.shape(2);

    auto sF_skij = math::shm::make_shared_array<Array_view_4D_t>(
        *mpi, {ns, nkpts_ibz, nbnd, nbnd});

    if (mpi->node_comm.root()) {
      sF_skij.local() = ComplexType(0.0, 0.0);
    }
    mpi->comm.barrier();

    methods::solvers::hf_t hf_solver;
    hf_solver.evaluate(sF_skij, Dm_skij, eri, S_skij,
                       compute_hartree, compute_exchange);

    // Shared memory is visible to all ranks on a node
    nda::array<ComplexType, 4> F_skij(ns, nkpts_ibz, nbnd, nbnd);
    F_skij = sF_skij.local();
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
   * @brief Compute LR GW self-energy term 1: ΔΣ = -ΔG ⊙ W_c + div_corr (fixed W, R-space)
   *
   * @param lr_params     - [INPUT] JSON string with params (prefix)
   * @param h_int         - [INPUT] THC ERI handler
   * @param q_pert        - [INPUT] LR perturbation wavevector (3,)
   * @param DeltaG_tskij  - [INPUT] LR Green's function (nt, ns, nk, nb, nb)
   * @return              - [OUTPUT] ΔΣ (nt, ns, nk, nb, nb)
   */
  nda::array<ComplexType, 5> lr_gw_sigma_DeltaG(
      const std::string &lr_params,
      ThcCoulomb &h_int,
      nda::array<double, 1> const& q_pert,
      std::optional<nda::array<ComplexType, 5>> DeltaG_tskij) {
    auto parser = InputParser(lr_params);
    methods::mb_eri_t mb_eri(h_int.get_eri());
    return methods::lr_gw_sigma_DeltaG_calc(mb_eri, parser.get_root(), q_pert, DeltaG_tskij);
  }


  /**
   * @brief Evaluate GW self-energy Σ = -G ⊙ W_c [+ div_corr] using W from file
   *
   * @param lr_params  - [INPUT] JSON string with params (prefix)
   * @param h_int      - [INPUT] THC ERI handler
   * @param G_tskij    - [INPUT] Green's function (nt, ns, nk, nb, nb)
   * @param div_corr   - [INPUT] Whether to apply divergence correction
   * @return           - [OUTPUT] Σ (nt, ns, nk, nb, nb)
   */
  nda::array<ComplexType, 5> gw_evaluate_sigma(
      const std::string &lr_params,
      ThcCoulomb &h_int,
      std::optional<nda::array<ComplexType, 5>> G_tskij,
      bool div_corr) {
    auto parser = InputParser(lr_params);
    methods::mb_eri_t mb_eri(h_int.get_eri());
    return methods::gw_evaluate_sigma_calc(mb_eri, parser.get_root(), G_tskij, div_corr);
  }


  /**
   * @brief Compute LR polarization ΔP = -ΔG·G - G·ΔG (R-space)
   *
   * @param h_int          - [INPUT] THC ERI handler
   * @param q_pert         - [INPUT] LR perturbation wavevector (3,)
   * @param G_tskij        - [INPUT] Unperturbed Green's function (nt, ns, nk, nb, nb)
   * @param DeltaG_tskij   - [INPUT] LR Green's function (nt, ns, nk, nb, nb)
   * @param DeltaX_left    - [INPUT, optional] δ^q X(k) (ns, nkpts, NP, nb)
   * @param DeltaX_right   - [INPUT, optional] δ^{-q} X(k+q) at storage k
   *                         When both are provided, primary→aux IBC is applied.
   * @return               - [OUTPUT] ΔP (nt_half, nkpts, NP, NP)
   */
  nda::array<ComplexType, 4> lr_gw_Pi(
      ThcCoulomb &h_int,
      nda::array<double, 1> const& q_pert,
      std::optional<nda::array<ComplexType, 5>> G_tskij,
      std::optional<nda::array<ComplexType, 5>> DeltaG_tskij,
      std::optional<nda::array<ComplexType, 4>> DeltaX_left,
      std::optional<nda::array<ComplexType, 4>> DeltaX_right) {
    methods::mb_eri_t mb_eri(h_int.get_eri());
    return methods::lr_gw_Pi_calc(mb_eri, q_pert, G_tskij, DeltaG_tskij,
                                    DeltaX_left, DeltaX_right);
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
      std::optional<nda::array<ComplexType, 5>> G_tskij) {
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
      std::optional<nda::array<ComplexType, 4>> DeltaPi_tqPQ) {
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
      std::optional<nda::array<ComplexType, 4>> Pi_tqPQ) {
    auto parser = InputParser(lr_params);
    methods::mb_eri_t mb_eri(h_int.get_eri());
    return methods::gw_evaluate_W_from_Pi_calc(mb_eri, parser.get_root(), Pi_tqPQ);
  }


  /**
   * @brief Compute LR GW self-energy term 2: -G ⊙ ΔW (no div correction)
   *
   * Computes ΔΣ = -G ⊙ ΔW from a pre-computed DeltaW.
   *
   * @param lr_params     - [INPUT] JSON string with params (prefix)
   * @param h_int         - [INPUT] THC ERI handler
   * @param q_pert        - [INPUT] LR perturbation wavevector (3,)
   * @param G_tskij       - [INPUT] Unperturbed Green's function (nt, ns, nk, nb, nb)
   * @param DeltaW_qtPQ   - [INPUT] LR screened interaction (nkpts, nt_half, NP, NP)
   * @return              - [OUTPUT] ΔΣ (nt, ns, nk, nb, nb)
   */
  nda::array<ComplexType, 5> lr_gw_sigma_DeltaW(
      const std::string &lr_params,
      ThcCoulomb &h_int,
      nda::array<double, 1> const& q_pert,
      std::optional<nda::array<ComplexType, 5>> G_tskij,
      std::optional<nda::array<ComplexType, 4>> DeltaW_qtPQ) {
    auto parser = InputParser(lr_params);
    methods::mb_eri_t mb_eri(h_int.get_eri());
    return methods::lr_gw_sigma_DeltaW_calc(mb_eri, parser.get_root(), q_pert, G_tskij, DeltaW_qtPQ);
  }


  /**
   * @brief Compute eps_inv_head from W_c in THC product basis
   *
   * @param lr_params      - [INPUT] JSON string with params (prefix for IAFT)
   * @param h_int          - [INPUT] THC ERI handler
   * @param W_c_tqPQ       - [INPUT] Correlation screened interaction W_c (nt_half, nkpts, NP, NP)
   * @return               - [OUTPUT] eps_inv_head (nt_half,)
   */
  nda::array<ComplexType, 1> compute_eps_inv_head(
      const std::string &lr_params,
      ThcCoulomb &h_int,
      std::optional<nda::array<ComplexType, 4>> W_c_tqPQ) {
    auto parser = InputParser(lr_params);
    methods::mb_eri_t mb_eri(h_int.get_eri());
    return methods::compute_eps_inv_head_calc(mb_eri, parser.get_root(), W_c_tqPQ);
  }

  /**
   * @brief Evaluate GW self-energy with provided W and G (FD helper)
   *
   * @param lr_params      - [INPUT] JSON string with params (prefix)
   * @param h_int          - [INPUT] THC ERI handler
   * @param G_tskij        - [INPUT] Green's function (nt, ns, nk, nb, nb)
   * @param W_c_qtPQ       - [INPUT] Screened interaction (nkpts, nt_half, NP, NP)
   * @param eps_inv_head   - [INPUT] Inverse dielectric head (nt_half,)
   * @param div_corr       - [INPUT] Whether to apply divergence correction
   * @return               - [OUTPUT] Σ (nt, ns, nk, nb, nb)
   */
  nda::array<ComplexType, 5> gw_evaluate_sigma_with_W(
      const std::string &lr_params,
      ThcCoulomb &h_int,
      std::optional<nda::array<ComplexType, 5>> G_tskij,
      std::optional<nda::array<ComplexType, 4>> W_c_qtPQ,
      nda::array<ComplexType, 1> const& eps_inv_head,
      bool div_corr) {
    auto parser = InputParser(lr_params);
    methods::mb_eri_t mb_eri(h_int.get_eri());
    return methods::gw_evaluate_sigma_with_W_calc(mb_eri, parser.get_root(), G_tskij, W_c_qtPQ, eps_inv_head, div_corr);
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
