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
#include "methods/SCF/lr_diis.hpp"
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
 * @brief Unified linear response calculation
 *
 * Runs the LR SCF loop with configurable Hartree, Exchange, and GW self-energy:
 *   ΔH0 → ΔG → ΔDm → [ΔF] → [ΔΣ] → ΔG → ... (iterate until convergence)
 *
 * @param eri              - [INPUT] ERI handler (must be THC)
 * @param pt               - [INPUT] Parameters as property tree
 * @param q_vec            - [INPUT] Perturbation wavevector in crystal coords (3,)
 * @param DeltaH0_skij_root - [INPUT] Perturbation matrix (ns, nk, nb, nb) on the
 *                            MPI global root; std::nullopt on every other rank.
 * @param include_hartree  - [INPUT] Include ΔJ in SCF loop
 * @param include_exchange - [INPUT] Include ΔK in SCF loop
 * @param gw_mode          - [INPUT] GW self-energy mode (none/fixed_W/full)
 * @param max_iter         - [INPUT] Maximum SCF iterations (1 = one-shot)
 * @param tol              - [INPUT] Convergence tolerance for ||ΔDm_new - ΔDm_old||
 * @param fix_density      - [INPUT] If true, compute Δμ to enforce ΔN=0
 * @param iter_params      - [INPUT] Iteration algorithm parameters (damping/DIIS)
 * @return Tuple of (number of iterations, final Δμ)
 */
template<typename eri_t>
std::tuple<int, double> run_lr_calc(eri_t &eri, ptree const& pt,
                                     nda::array<double, 1> const& q_vec,
                                     std::optional<nda::array<ComplexType, 4>> const& DeltaH0_skij_root,
                                     bool include_hartree,
                                     bool include_exchange,
                                     lr_gw_update_mode gw_mode,
                                     int max_iter,
                                     double tol,
                                     bool fix_density,
                                     const lr_iter_params& iter_params,
                                     std::optional<nda::array<ComplexType, 4>> const& DeltaX_left_root = std::nullopt,
                                     std::optional<nda::array<ComplexType, 4>> const& DeltaX_right_root = std::nullopt,
                                     std::optional<nda::array<ComplexType, 3>> const& DeltaV_qPQ_root = std::nullopt);

}


#endif
