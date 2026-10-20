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


#ifndef METHODS_MBPT_DRIVERS_H
#define METHODS_MBPT_DRIVERS_H

#include <map>
#include <optional>
#include <string>

#include "configuration.hpp"
#include "mpi3/environment.hpp"
#include "mpi3/communicator.hpp"
#include "cxxopts.hpp"

#include "IO/AppAbort.hpp"
#include "IO/app_loggers.h"
#include "IO/ptree/ptree_utilities.hpp"

#include "mean_field/MF.hpp"
#include "methods/mb_state/mb_state.hpp"
#include "methods/SCF/lr_driver.hpp"

namespace mpi3 = boost::mpi3;
namespace methods
{

/**
 * @brief Many-body perturbation calculations from a given mean-field and ERI objects with arguments in property tree.
 */
template<typename eri_t>
void mbpt(std::string solver_type, eri_t &eri, ptree const& pt);

template<typename eri_t>
void mbpt(std::string solver_type, eri_t &eri, ptree const& pt,
          nda::array<ComplexType, 5> const& projector_ksIai,
          nda::array<long, 3> const& band_window,
          nda::array<RealType, 2> const& kpts_crys,
          std::optional<std::map<std::string, nda::array<ComplexType, 5> > > local_polarizabilities=std::nullopt);


/**
 * @brief Downfolding of the local Green's function
 */
auto downfold_gloc_impl(std::shared_ptr<mf::MF> mf,
                        MBState&& mb_state, ptree const& pt)
-> nda::array<ComplexType, 5>;

auto downfold_gloc_with_projector_from_h5(std::shared_ptr<mf::MF> mf, ptree const& pt)
-> nda::array<ComplexType, 5>;

auto downfold_gloc(std::shared_ptr<mf::MF> mf, ptree const& pt,
                   nda::array<ComplexType, 5> const& projector_ksIai,
                   nda::array<long, 3> const& band_window,
                   nda::array<RealType, 2> const& kpts_crys)
-> nda::array<ComplexType, 5>;


/**
 * @brief LR of the local Green's function at momentum transfer Q
 *
 * ΔG_loc = (1/N_k) Σ_k C^{k+Q} ΔG^k C^{k†}, with ΔG^k the (k+Q, k) block. Full BZ only, no hermitization.
 * The projector algebra is tested at Q != 0 (synthetic adjoint and NumPy references); the physical Q != 0
 * end-to-end response is not yet validated.
 *
 * @param pt                 - [INPUT] translate_home_cell (default false)
 * @param DeltaG_tskij_root  - [INPUT] ΔG (nt, ns, nk, nb, nb) on the MPI root; ignored elsewhere
 * @param q_vec              - [INPUT] Q in crystal coordinates (3,)
 * @return ΔG_loc (nt, ns, nImps, nImpOrbs, nImpOrbs) on every rank
 */
auto lr_downfold_gloc(std::shared_ptr<mf::MF> mf, ptree const& pt,
                      nda::array<ComplexType, 5> const& projector_ksIai,
                      nda::array<long, 3> const& band_window,
                      nda::array<RealType, 2> const& kpts_crys,
                      std::optional<nda::array<ComplexType, 5>> const& DeltaG_tskij_root,
                      nda::array<double, 1> const& q_vec)
-> nda::array<ComplexType, 5>;

/**
 * @brief LR of the local screened interaction at momentum transfer Q
 *
 * ΔW_loc_{x,cdab} = (1/N_q) Σ_q conj(B^{q+Q}_{P,dc}) ΔW^q_{x,PQ} B^q_{Q,ab}, with ΔW^q the (q+Q, q) block.
 * Full q mesh only, single impurity. The q->0 head term is not included: the result is exact when the head of
 * ΔW vanishes (models), otherwise it omits the head that the ground-state W_loc carries with "gygi".
 * The projector algebra is tested at Q != 0 (synthetic adjoint and NumPy references); the physical Q != 0
 * end-to-end response is not yet validated.
 *
 * @param pt                 - [INPUT] translate_home_cell
 * @param DeltaW_xqPQ_root   - [INPUT] ΔW (nx, nq, NP, NP), any leading axis, on the MPI root
 * @param q_vec              - [INPUT] Q in crystal coordinates (3,)
 * @return ΔW_loc (nx, c, d, a, b) on every rank
 */
template<typename eri_t>
nda::array<ComplexType, 5> lr_downfold_coulomb(
    eri_t &eri, ptree const& pt,
    nda::array<ComplexType, 5> const& projector_ksIai,
    nda::array<long, 3> const& band_window,
    nda::array<RealType, 2> const& kpts_crys,
    std::optional<nda::array<ComplexType, 4>> const& DeltaW_xqPQ_root,
    nda::array<double, 1> const& q_vec);


/**
 * @brief Downfolding of one-electron quantities, Green's function, self-energy, and effective potential
 * This is currently used in the GW+EDMFT calculations in the toml input mode.
 * Note: This functions is doing too many things at once, it will be refactored in the future.
 */
void downfolding_1e(std::shared_ptr<mf::MF> mf, ptree const& pt);


/**
 * @brief Downfolds the (screened) Coulomb interaction matrix elements to a localized basis (Python API entry point).
 *
 * This function computes the downfolded (projected) Coulomb matrix elements for a given electronic structure
 * and projector, using the provided ERI (electron repulsion integral) object and parameters.
 * The output can be either purely local or include non-local (q-dependent) screened interactions,
 * depending on the "q_dependent_output" parameter in the input property tree.
 *
 * The projector is specified via an external Wannier function HDF5 file, as indicated in the "params" property tree.
 *
 * @tparam eri_t Type of the ERI object.
 * @param eri The ERI object (e.g., THC or Cholesky representation).
 * @param pt Property tree containing all user parameters and options.
 * @param local_polarizabilities (optional) Map of polarizability corrections in the local downfolded basis for EDMFT screening.
 * @return Tuple of (Vloc_abcd, Wloc_wabcd) where Vloc_abcd is the bare local Coulomb tensor and Wloc_wabcd is the screened tensor.
 *
 * @note The function supports both local and non-local (q-dependent) output, controlled by user options.
 * @note The result is projected to the discretized (localized) basis defined by the projector.
 */
template<typename eri_t>
std::tuple<nda::array<ComplexType, 4>, nda::array<ComplexType, 5> >
downfold_coulomb_with_projector_from_h5(
    eri_t &eri, ptree const& pt,
    std::optional<std::map<std::string, nda::array<ComplexType, 5> > > local_polarizabilities=std::nullopt);

/**
 * @brief Downfolds the (screened) Coulomb interaction matrix elements to a localized basis (direct projector specification overload).
 *
 * This overload allows the user to specify the projector directly via the projector_ksIai matrix, band_window, and kpts_crys arrays,
 * rather than through an external Wannier function HDF5 file. All other behavior is identical to the main overload.
 *
 * @tparam eri_t Type of the ERI object.
 * @param eri The ERI object (e.g., THC or Cholesky representation).
 * @param pt Property tree containing all user parameters and options.
 * @param projector_ksIai Projector matrix from KS to impurity orbitals.
 * @param band_window Band window array.
 * @param kpts_crys k-point array in crystal coordinates.
 * @param local_polarizabilities (optional) Map of polarizability corrections in the local downfolded basis for EDMFT screening.
 * @return Tuple of (Vloc_abcd, Wloc_wabcd) where Vloc_abcd is the bare local Coulomb tensor and Wloc_wabcd is the screened tensor.
 *
 * @note The function supports both local and non-local (q-dependent) output, controlled by user options.
 * @note The result is projected to the discretized (localized) basis defined by the projector.
 */
template<typename eri_t>
std::tuple<nda::array<ComplexType, 4>, nda::array<ComplexType, 5> >
downfold_coulomb(eri_t &eri, ptree const& pt, nda::array<ComplexType, 5> const& projector_ksIai, 
                 nda::array<long, 3> const& band_window, nda::array<RealType, 2> const& kpts_crys,
                 std::optional<std::map<std::string, nda::array<ComplexType, 5> > > local_polarizabilities=std::nullopt);

/**
 * @brief Implementation of downfolding the (screened) Coulomb interaction matrix elements to a localized basis.
 *
 * This function performs the actual computation of the downfolded (projected) Coulomb matrix elements for a given electronic structure
 * and projector, using the provided ERI (electron repulsion integral) object and parameters. It is typically called internally by the Python API or other wrappers.
 *
 * @tparam eri_t Type of the ERI object.
 * @param eri The ERI object (e.g., THC or Cholesky representation).
 * @param mb_state Many-body state object containing all relevant data for the calculation.
 * @param pt Property tree containing all user parameters and options.
 * @param local_polarizabilities (optional) Map of polarizability corrections in the local downfolded basis for EDMFT screening.
 * @return Tuple of (Vloc_abcd, Wloc_wabcd) where Vloc_abcd is the bare local Coulomb tensor and Wloc_wabcd is the screened tensor.
 *
 * @note The function supports both local and non-local (q-dependent) output, controlled by user options.
 * @note The result is projected to the discretized (localized) basis defined by the projector.
 */
template<typename eri_t>
std::tuple<nda::array<ComplexType, 4>, nda::array<ComplexType, 5> >
downfold_coulomb_impl(eri_t &eri, MBState&& mb_state, ptree const& pt,
                      std::optional<std::map<std::string, nda::array<ComplexType, 5> > > local_polarizabilities);


/**
 * @brief Downfolding of various two-electron quantities, bare Coulomb, screened Coulomb and partially screened Coulomb interactions.
 * This is currently used in the GW+EDMFT calculations in the toml input mode.
 * Note: This functions is doing too many things at once, it will be refactored in the future.
 */
template<typename eri_t>
void downfolding_2e(eri_t &eri, ptree const& pt,
                    std::optional<std::map<std::string, nda::array<ComplexType, 5> > > local_polarizabilities=std::nullopt);

template<typename eri_t>
void hf_downfold(eri_t &eri, ptree const& pt);

template<typename eri_t>
void gw_downfold(eri_t &eri, ptree &pt);

/**
 * @brief Embedding (i.e. upfolding) of local DMFT self-energy corrections to a MBPT solution stored in the checkpoint file.
 */
void dmft_embed_with_projector_from_h5(std::shared_ptr<mf::MF> mf, ptree const& pt,
                std::optional<std::map<std::string, nda::array<ComplexType, 4> > > local_hf_potentials=std::nullopt,
                std::optional<std::map<std::string, nda::array<ComplexType, 5> > > local_selfenergies=std::nullopt);

void dmft_embed(std::shared_ptr<mf::MF> mf, ptree const& pt,
                nda::array<ComplexType, 5> const& projector_ksIai,
                nda::array<long, 3> const& band_window,
                nda::array<RealType, 2> const& kpts_crys,
                std::optional<std::map<std::string, nda::array<ComplexType, 4> > > local_hf_potentials,
                std::optional<std::map<std::string, nda::array<ComplexType, 5> > > local_selfenergies);

/**
 * Local projection of an LR run: the projector of the ground-state interface (run_gw / dmft_embed).
 * Given, it switches on the local outputs DeltaGloc_tsIab and, with gw_mode = "full", DeltaWloc_tabcd.
 *
 * local_polarizabilities declares a GW+EDMFT ground state: the ground-state Π_imp, Π_dc ({"imp", "dc"},
 * (nw_half, a, b, c, d), product basis) enter the W of the run (recompute_W), and the linearized double
 * counting of the local terms is part of the kernel. It requires dc_cells and writes the ground-state local
 * factors Gloc_tsIab, Wloc_tabcd and Vloc_abcd to "linear_response/".
 */
struct lr_local_inputs {
  nda::array<ComplexType, 5> projector_ksIai;
  nda::array<long, 3> band_window;
  nda::array<RealType, 2> kpts_crys;
  std::optional<std::map<std::string, nda::array<ComplexType, 5>>> local_polarizabilities;
  std::optional<dc_cells_t> dc_cells;
  /// Response of the impurity of a GW+EDMFT ground state, in the layouts of the ground state: "Sigma"
  /// δΣ_imp(iω) (nw_f, ns, 1, n, n) and "Pi" δΠ_imp(iν) (nw_half, n, n, n, n), product basis. A missing key
  /// is zero. Requires local_polarizabilities and a single perturbation; "Pi" requires gw_mode = full.
  std::optional<std::map<std::string, nda::array<ComplexType, 5>>> impurity_response;
};

/**
 * @brief Unified linear response calculation
 *
 * Runs the LR SCF loop with configurable Hartree, Exchange, and GW self-energy:
 *   ΔH0 → ΔG → ΔDm → [ΔF] → [ΔΣ] → ΔG → ... (iterate until convergence)
 *
 * One call covers any number of perturbations at the same q: the setup (W,
 * G^R/G(iω) caches, solvers) is paid once and each perturbation is written to
 * its own "linear_response/mode{m}" group.
 *
 * @param eri              - [INPUT] ERI handler (must be THC)
 * @param pt               - [INPUT] Parameters as property tree
 * @param q_vec            - [INPUT] Perturbation wavevector in crystal coords (3,)
 * @param DeltaH0_mskij_root - [INPUT] Perturbations (nmodes, ns, nk, nb, nb) on the
 *                            MPI global root; std::nullopt on every other rank.
 * @param local            - [INPUT] Optional projector (nullptr: none); full BZ, single impurity.
 *                            Writes DeltaGloc_tsIab (nt, ns, 1, n, n) and, with gw_mode "full",
 *                            DeltaWloc_tabcd (nt_half, n, n, n, n) per mode to "linear_response/".
 *                            With local_polarizabilities (GW+EDMFT ground state) it also writes
 *                            Gloc_tsIab, Wloc_tabcd (chemistry order) and Vloc_abcd, and the kernel
 *                            (gw_mode fixed_W or full) includes the linearized double counting; DeltaSigma_dc_tsIab and
 *                            DeltaSigma_imp_tsIab (nt, ns, 1, n, n) are then written per mode too.
 * Every other option is read from `pt`; the key list is on the definition in MBPT_drivers.cpp.
 * @return Per-perturbation (number of iterations, final Δμ), each of length nmodes
 */
template<typename eri_t>
std::tuple<nda::array<long, 1>, nda::array<double, 1>>
                        run_lr_calc(eri_t &eri, ptree const& pt,
                                     nda::array<double, 1> const& q_vec,
                                     std::optional<nda::array<ComplexType, 5>> const& DeltaH0_mskij_root,
                                     std::optional<nda::array<ComplexType, 4>> const& DeltaX_left_root = std::nullopt,
                                     std::optional<nda::array<ComplexType, 4>> const& DeltaX_right_root = std::nullopt,
                                     std::optional<nda::array<ComplexType, 3>> const& DeltaV_qPQ_root = std::nullopt,
                                     const lr_local_inputs* local = nullptr);

/**
 * @brief Linear response GW self-energy with fixed W (term 1)
 *
 * Computes ΔΣ = -ΔG ⊙ W_c + div_corr (R-space Hadamard product).
 * Reads W and eps_inv_head from checkpoint.
 *
 * @param eri               - [INPUT] THC ERI handler
 * @param pt                - [INPUT] Parameters as property tree (prefix, output)
 * @param q_pert            - [INPUT] LR perturbation wavevector in crystal coords (3,)
 * @param DeltaG_tskij_root - [INPUT] LR Green's function (nt, ns, nk, nb, nb) on the
 *                            MPI global root; std::nullopt elsewhere.
 * @return ΔΣ(τ,s,k,i,j) on rank 0 (nt, ns, nk, nb, nb); empty array on non-root.
 */
template<typename eri_t>
nda::array<ComplexType, 5> lr_gw_sigma_DeltaG_calc(
    eri_t &eri, ptree const& pt,
    nda::array<double, 1> const& q_pert,
    std::optional<nda::array<ComplexType, 5>> const& DeltaG_tskij_root);

/**
 * @brief Evaluate GW self-energy Σ = -G ⊙ W_c [+ div_corr] using W from file
 *
 * Same R-space computation as lr_gw but for a full (non-LR) Green's function.
 * Used for finite-difference testing: Σ[G+εΔG] computed by passing G+εΔG as input.
 *
 * @param eri      - [INPUT] THC ERI handler
 * @param pt       - [INPUT] Parameters as property tree (prefix, output)
 * @param G_tskij  - [INPUT] Green's function (nt, ns, nk, nb, nb)
 * @param div_corr - [INPUT] Whether to apply divergence correction (default true)
 * @return Σ(τ,s,k,i,j) as (nt, ns, nk, nb, nb)
 */
template<typename eri_t>
nda::array<ComplexType, 5> gw_evaluate_sigma_calc(
    eri_t &eri, ptree const& pt,
    std::optional<nda::array<ComplexType, 5>> const& G_tskij_root,
    bool div_corr = true);

/**
 * @brief Linear response GW self-energy term 2: -G ⊙ ΔW + div_corr
 *
 * Computes ΔΣ = -G ⊙ ΔW from a pre-computed DeltaW.
 * At q_pert=0, also applies term 2 divergence correction using Δeps_inv_head from ΔW.
 *
 * @param eri          - [INPUT] THC ERI handler
 * @param pt           - [INPUT] Parameters as property tree (prefix)
 * @param q_pert       - [INPUT] LR perturbation wavevector in crystal coords (3,)
 * @param G_tskij      - [INPUT] Unperturbed Green's function (nt, ns, nk, nb, nb)
 * @param DeltaW_qtPQ  - [INPUT] LR screened interaction (nkpts, nt_half, NP, NP)
 * @return ΔΣ(τ,s,k,i,j) as (nt, ns, nk, nb, nb)
 */
template<typename eri_t>
nda::array<ComplexType, 5> lr_gw_sigma_DeltaW_calc(
    eri_t &eri, ptree const& pt,
    nda::array<double, 1> const& q_pert,
    std::optional<nda::array<ComplexType, 5>> const& G_tskij_root,
    std::optional<nda::array<ComplexType, 4>> const& DeltaW_qtPQ_root);

/**
 * @brief Evaluate GW self-energy with provided W and G (FD helper)
 *
 * Computes Σ = -G ⊙ W_c [+ div_corr] using provided G and W_c arrays
 * (not from file). Used for finite-difference testing of full LR-GW.
 *
 * @param eri           - [INPUT] THC ERI handler
 * @param pt            - [INPUT] Parameters as property tree (prefix)
 * @param G_tskij       - [INPUT] Green's function (nt, ns, nk, nb, nb)
 * @param W_c_qtPQ      - [INPUT] Screened interaction (nkpts, nt_half, NP, NP)
 * @param eps_inv_head  - [INPUT] Inverse dielectric head (nt_half,)
 * @param div_corr      - [INPUT] Whether to apply divergence correction
 * @return Σ(τ,s,k,i,j) as (nt, ns, nk, nb, nb)
 */
template<typename eri_t>
nda::array<ComplexType, 5> gw_evaluate_sigma_with_W_calc(
    eri_t &eri, ptree const& pt,
    std::optional<nda::array<ComplexType, 5>> const& G_tskij_root,
    std::optional<nda::array<ComplexType, 4>> const& W_c_qtPQ_root,
    nda::array<ComplexType, 1> const& eps_inv_head,
    bool div_corr);

/**
 * @brief Compute eps_inv_head from W_c in THC product basis
 *
 * Extracts (ε⁻¹-1) head from W_c and extrapolates to q→0.
 * Matches the convention used by Sigma_div_correction.
 *
 * @param W_c_tqPQ  - [INPUT] Screened interaction W_c (nt_half, nkpts, NP, NP)
 * @return eps_inv_head at q=0, shape (nt_half,)
 */
template<typename eri_t>
nda::array<ComplexType, 1> compute_eps_inv_head_calc(
    eri_t &eri, ptree const& pt,
    std::optional<nda::array<ComplexType, 4>> const& W_c_tqPQ_root);

/**
 * @brief Band-basis perturbation from a local potential given in the THC aux basis
 *
 * Maps a local potential u(r), supplied through its projections onto the THC
 * interpolating vectors,
 *
 *   u_P = ∫ dr ζ_P(r) u(r)
 *
 * onto the band-basis perturbation the LR solver takes:
 *
 *   ΔH0_ij(k) = Σ_(p,P) conj(X_p(k+q)_Pi) u_P X_p(k)_Pj
 *
 * No overlap matrix enters even though the ζ are not orthonormal: the ISDF ansatz
 * expands the pair density with its own values at the interpolating points,
 * conj(X(k+q)_Pi) X(k)_Pj, so ζ appears exactly once per matrix element. A caller
 * holding expansion coefficients c_Q with u(r) = Σ_Q c_Q ζ_Q(r) must convert them,
 * u_P = Σ_Q S_PQ c_Q with S_PQ = ∫ dr ζ_P(r) ζ_Q(r); the metric belongs to that
 * conversion, not here.
 *
 * The map itself is the same aux→primary one the Hartree channel applies to ΔJ_P
 * (lr_thc_comm::aux_to_primary_diagonal). A stack of potentials at the one q is
 * mapped in a single call and returned as the rank-5 perturbation stack run_lr
 * accepts.
 *
 * @param eri     - [INPUT] THC ERI handler
 * @param q_pert  - [INPUT] Perturbation wavevector in crystal coords (3,)
 * @param u_mP    - [INPUT] Aux-basis potential projections u_P (nmodes, NP); root only
 * @return ΔH0 (nmodes, ns, nkpts_ibz, nbnd, nbnd) on the root, empty elsewhere
 *
 * Requires nkpts_ibz == nkpts: an arbitrary u_P at finite q breaks the crystal
 * symmetry, so a symmetry-reduced k-set cannot carry the resulting perturbation.
 */
template<typename eri_t>
nda::array<ComplexType, 5> lr_DeltaH0_from_thc_aux_calc(
    eri_t &eri,
    nda::array<double, 1> const& q_pert,
    std::optional<nda::array<ComplexType, 2>> const& u_mP_root);

/**
 * @brief Linear response polarization ΔP = -ΔG·G - G·ΔG (R-space)
 *
 * Computes LR polarization via product-rule differentiation of P = -G·G.
 * Uses lr_thc_comm for ΔG factors (asymmetric X(k+q)/X(k)),
 * thc_solver_comm for G factors (symmetric).
 *
 * @param eri           - [INPUT] THC ERI handler
 * @param q_pert        - [INPUT] LR perturbation wavevector in crystal coords (3,)
 * @param G_tskij       - [INPUT] Unperturbed Green's function (nt, ns, nk, nb, nb)
 * @param DeltaG_tskij  - [INPUT] LR Green's function (nt, ns, nk, nb, nb)
 * @param DeltaX_left   - [INPUT, optional] δ^q X(k), (ns, nkpts, NP, nb). Full BZ.
 * @param DeltaX_right  - [INPUT, optional] δ^{-q} X(k+q) at storage k, same shape.
 *                        When both DeltaX arrays are provided, the primary→aux
 *                        IBC correction is applied inside evaluate_lr_Pi.
 * @return ΔP(τ,q,P,Q) as (nt_half, nkpts, NP, NP)
 */
template<typename eri_t>
nda::array<ComplexType, 4> lr_gw_Pi_calc(
    eri_t &eri,
    nda::array<double, 1> const& q_pert,
    std::optional<nda::array<ComplexType, 5>> const& G_tskij_root,
    std::optional<nda::array<ComplexType, 5>> const& DeltaG_tskij_root,
    std::optional<nda::array<ComplexType, 4>> const& DeltaX_left_root = std::nullopt,
    std::optional<nda::array<ComplexType, 4>> const& DeltaX_right_root = std::nullopt);

/**
 * @brief Evaluate the polarization P[G] (FD helper)
 *
 * Computes the RPA polarization from a given G. Used for finite-difference testing: P[G+εΔG] computed by
 * passing G+εΔG. With a projector and local_polarizabilities {"imp", "dc"} (iν, product basis, as run_lr), the
 * GW+EDMFT polarization P_RPA[G] + upfold(Π_imp − Π_dc) of the GW+EDMFT ground state (screen type "gw_edmft").
 *
 * @param eri      - [INPUT] THC ERI handler
 * @param pt       - [INPUT] Parameters as property tree (prefix)
 * @param G_tskij  - [INPUT] Green's function (nt, ns, nk, nb, nb)
 * @param local    - [INPUT] Optional projector and local_polarizabilities (dc_cells, impurity_response unused)
 * @return P(τ,q,P,Q) as (nt_half, nkpts, NP, NP)
 */
template<typename eri_t>
nda::array<ComplexType, 4> gw_evaluate_Pi_calc(
    eri_t &eri, ptree const& pt,
    std::optional<nda::array<ComplexType, 5>> const& G_tskij_root,
    lr_local_inputs const* local = nullptr);

/**
 * @brief Linear response screened interaction ΔW = (Z+W_c) · ΔΠ · (Z+W_c)
 *
 * Reads W_c(τ) from thc_screened_interaction.h5 and IAFT from checkpoint,
 * then calls lr_scr_coulomb_t::solve_lr_dyson_W to compute ΔW_c(τ).
 *
 * @param eri           - [INPUT] THC ERI handler
 * @param pt            - [INPUT] Parameters as property tree (prefix)
 * @param q_pert        - [INPUT] LR perturbation wavevector in crystal coords (3,)
 * @param DeltaPi_tqPQ  - [INPUT] LR polarization (nt_half, nkpts, NP, NP)
 * @return ΔW_c(τ,q,P,Q) as (nt_half, nkpts, NP, NP)
 */
template<typename eri_t>
nda::array<ComplexType, 4> lr_gw_W_calc(
    eri_t &eri, ptree const& pt,
    nda::array<double, 1> const& q_pert,
    std::optional<nda::array<ComplexType, 4>> const& DeltaPi_tqPQ_root);

/**
 * @brief Evaluate screened interaction W_c from polarization Π (FD helper)
 *
 * Exposes scr_coulomb_t::dyson_W_from_Pi_tau for finite-difference testing.
 * Reads IAFT from checkpoint, applies W Dyson equation Π→W_c.
 *
 * @param eri      - [INPUT] THC ERI handler
 * @param pt       - [INPUT] Parameters as property tree (prefix)
 * @param Pi_tqPQ  - [INPUT] Polarization (nt_half, nkpts, NP, NP)
 * @return W_c(τ,q,P,Q) as (nt_half, nkpts, NP, NP)
 */
template<typename eri_t>
nda::array<ComplexType, 4> gw_evaluate_W_from_Pi_calc(
    eri_t &eri, ptree const& pt,
    std::optional<nda::array<ComplexType, 4>> const& Pi_tqPQ_root);


}


#endif
