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


#ifndef METHODS_MBPT_DRIVERS_H
#define METHODS_MBPT_DRIVERS_H

#include <string>
#include <tuple>

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
          nda::array<ComplexType, 5> const& C_ksIai,
          nda::array<long, 3> const& band_window,
          nda::array<RealType, 2> const& kpts_crys,
          std::optional<std::map<std::string, nda::array<ComplexType, 5> > > local_polarizabilities=std::nullopt);


/**
 * @brief Downfolding of the local Green's function
 */
auto downfold_gloc_impl(std::shared_ptr<mf::MF> mf,
                        MBState&& mb_state, ptree const& pt)
-> nda::array<ComplexType, 5>;

auto downfold_gloc(std::shared_ptr<mf::MF> mf, ptree const& pt)
-> nda::array<ComplexType, 5>;

auto downfold_gloc(std::shared_ptr<mf::MF> mf, ptree const& pt,
                   nda::array<ComplexType, 5> const& C_ksIai,
                   nda::array<long, 3> const& band_window,
                   nda::array<RealType, 2> const& kpts_crys)
-> nda::array<ComplexType, 5>;


/**
 * @brief Downfolding of one-electron quantities, Green's function, self-energy, and effective potential
 * This is currently used in the GW+EDMFT calculations in the toml input mode.
 * Note: This functions is doing too many things at once, it will be refactored in the future.
 */
void downfolding_1e(std::shared_ptr<mf::MF> mf, ptree const& pt,
                    std::optional<std::map<std::string, nda::array<ComplexType, 5> > > local_selfenergies=std::nullopt,
                    std::optional<std::map<std::string, nda::array<ComplexType, 4> > > local_hf_potentials=std::nullopt);

void downfolding_1e(std::shared_ptr<mf::MF> mf, ptree const& pt,
                    nda::array<ComplexType, 5> const& C_ksIai,
                    nda::array<long, 3> const& band_window,
                    nda::array<RealType, 2> const& kpts_crys,
                    std::optional<std::map<std::string, nda::array<ComplexType, 5> > > local_selfenergies=std::nullopt,
                    std::optional<std::map<std::string, nda::array<ComplexType, 4> > > local_hf_potentials=std::nullopt);

/**
 * @brief Downfolding of the local Coulomb interactions
 */
template<typename eri_t>
std::tuple<nda::array<ComplexType, 4>, nda::array<ComplexType, 5> >
downfold_wloc_impl(eri_t &eri, MBState&& mb_state, ptree const& pt,
                   std::optional<std::map<std::string, nda::array<ComplexType, 5> > > local_polarizabilities);

template<typename eri_t>
std::tuple<nda::array<ComplexType, 4>, nda::array<ComplexType, 5> >
downfold_wloc(eri_t &eri, ptree const& pt,
              std::optional<std::map<std::string, nda::array<ComplexType, 5> > > local_polarizabilities=std::nullopt);

template<typename eri_t>
std::tuple<nda::array<ComplexType, 4>, nda::array<ComplexType, 5> >
downfold_wloc(eri_t &eri, ptree const& pt,
              nda::array<ComplexType, 5> const& C_ksIai,
              nda::array<long, 3> const& band_window,
              nda::array<RealType, 2> const& kpts_crys,
              std::optional<std::map<std::string, nda::array<ComplexType, 5> > > local_polarizabilities=std::nullopt);


/**
 * @brief Downfolding of various two-electron quantities, bare Coulomb, screened Coulomb and partially screened Coulomb interactions.
 * This is currently used in the GW+EDMFT calculations in the toml input mode.
 * Note: This functions is doing too many things at once, it will be refactored in the future.
 */
template<bool return_vw=false, typename eri_t>
std::conditional_t<return_vw, std::tuple<nda::array<ComplexType, 4>, nda::array<ComplexType, 5>>, void>
downfolding_2e(eri_t &eri, ptree const& pt,
               std::optional<std::map<std::string, nda::array<ComplexType, 5> > > local_polarizabilities=std::nullopt);

template<bool return_vw=false, typename eri_t>
std::conditional_t<return_vw, std::tuple<nda::array<ComplexType, 4>, nda::array<ComplexType, 5>>, void>
downfolding_2e(eri_t &eri, ptree const& pt,
               nda::array<ComplexType, 5> const& C_ksIai,
               nda::array<long, 3> const& band_window,
               nda::array<RealType, 2> const& kpts_crys,
               std::optional<std::map<std::string, nda::array<ComplexType, 5> > > local_polarizabilities=std::nullopt);



template<typename eri_t>
void hf_downfold(eri_t &eri, ptree const& pt);

template<typename eri_t>
void gw_downfold(eri_t &eri, ptree const& pt);

/**
 * @brief Embedding (i.e. upfolding) of local DMFT self-energy corrections to a MBPT solution stored in the checkpoint file.
 */
void dmft_embed(std::shared_ptr<mf::MF> mf, ptree const& pt,
                nda::array<ComplexType, 5> const& C_ksIai,
                nda::array<long, 3> const& band_window,
                nda::array<RealType, 2> const& kpts_crys,
                std::optional<std::map<std::string, nda::array<ComplexType, 4> > > local_hf_potentials=std::nullopt,
                std::optional<std::map<std::string, nda::array<ComplexType, 5> > > local_selfenergies=std::nullopt);

void dmft_embed(std::shared_ptr<mf::MF> mf, ptree const& pt);

/**
 * @brief Linear response Dyson equation calculation
 *
 * Computes the linear response Green's function ΔG from an external perturbation ΔH0.
 * Requires a previous HF/GW calculation to provide the unperturbed Green's function.
 *
 * The LR Dyson equation solved is:
 *   ΔG(k,iω) = G(k+q,iω) · ΔH0(k) · G(k,iω)
 *
 * Required parameters in pt:
 *  - prefix: Input checkpoint prefix (reads {prefix}.mbpt.h5)
 *  - output: Output checkpoint prefix (default: same as prefix)
 *
 * @param eri          - [INPUT] ERI handler (provides MPI, MF)
 * @param pt           - [INPUT] Parameters as property tree
 * @param q_vec        - [INPUT] Perturbation wavevector in crystal coords (3,)
 * @param DeltaH0_skij - [INPUT] Perturbation matrix (ns, nk, nb, nb)
 * @param fix_density  - [INPUT] If true, compute Δμ to enforce ΔN=0 (default: false)
 * @return The Δμ value used (computed if fix_density=true, otherwise 0.0)
 */
template<typename eri_t>
double lr_dyson_calc(eri_t &eri, ptree const& pt,
                     nda::array<double, 1> const& q_vec,
                     nda::array<ComplexType, 4> const& DeltaH0_skij,
                     bool fix_density = false);

/**
 * @brief Linear response Hartree-Fock SCF calculation
 *
 * Runs the full LR-HF SCF loop:
 *   ΔH0 → ΔG → ΔDm → ΔF → ΔG → ... (iterate until convergence)
 *
 * Required parameters in pt:
 *  - prefix: Input checkpoint prefix (reads {prefix}.mbpt.h5)
 *  - output: Output checkpoint prefix (default: same as prefix)
 *
 * Optional parameters in pt:
 *  - input_type: HDF5 group to read checkpoint from (default: "scf")
 *                Specifies which calculation's data to use as starting point.
 *                Options: "scf" (standard SCF), "embed" (embedding), etc.
 *  - input_iter: Iteration number to read (default: -1 = use final_iter)
 *
 * @param eri          - [INPUT] ERI handler (must be THC)
 * @param pt           - [INPUT] Parameters as property tree
 * @param q_vec        - [INPUT] Perturbation wavevector in crystal coords (3,)
 * @param DeltaH0_skij - [INPUT] Perturbation matrix (ns, nk, nb, nb)
 * @param max_iter     - [INPUT] Maximum SCF iterations (default: 50)
 * @param tol          - [INPUT] Convergence tolerance for ||ΔDm_new - ΔDm_old|| (default: 1e-8)
 * @param fix_density  - [INPUT] If true, compute Δμ to enforce ΔN=0 (default: true)
 * @return Tuple of (number of iterations, final Δμ)
 */
template<typename eri_t>
std::tuple<int, double> lr_hf_scf_calc(eri_t &eri, ptree const& pt,
                                        nda::array<double, 1> const& q_vec,
                                        nda::array<ComplexType, 4> const& DeltaH0_skij,
                                        int max_iter = 50,
                                        double tol = 1e-8,
                                        bool fix_density = true,
                                        const lr_iter_params& iter_params = {});

}
#endif
