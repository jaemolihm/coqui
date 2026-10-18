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
#include "methods/embedding/dc_response.hpp"

#include "python/mean_field/mf_module.hpp"
#include "python/mean_field/mf_module.wrap.hxx"
#include "python/interaction/eri_module.hpp"
#include "python/interaction/eri_module.wrap.hxx"

namespace coqui_py {


  template<typename eri_handler_t>
  void downfold_2e(eri_handler_t &eri, const std::string &df_params,
                   std::optional<std::map<std::string, nda::array<ComplexType, 5> > > local_polarizabilities) {
    auto parser = InputParser(df_params);
    methods::downfolding_2e(eri.get_eri(), parser.get_root(), std::move(local_polarizabilities));
  }




  template<typename eri_handler_t>
  std::tuple<nda::array<ComplexType, 4>, nda::array<ComplexType, 5>>
  downfold_coulomb_with_projector_from_h5(eri_handler_t &eri, const std::string &df_params,
                std::optional<std::map<std::string, nda::array<ComplexType, 5> > > local_polarizabilities) {
    auto parser = InputParser(df_params);
    return methods::downfold_coulomb_with_projector_from_h5(
      eri.get_eri(), parser.get_root(), std::move(local_polarizabilities));
  }

  template<typename eri_handler_t>
  std::tuple<nda::array<ComplexType, 4>, nda::array<ComplexType, 5>>
  downfold_coulomb(eri_handler_t &eri, const std::string &df_params,
                const nda::array<ComplexType, 5> &projector_ksIai,
                const nda::array<long, 3> &band_window,
                const nda::array<RealType, 2> &kpts_crys,
                std::optional<std::map<std::string, nda::array<ComplexType, 5> > > local_polarizabilities) {
    auto parser = InputParser(df_params);
    return methods::downfold_coulomb(
      eri.get_eri(), parser.get_root(),
      projector_ksIai, band_window, kpts_crys, 
      std::move(local_polarizabilities));
  }




  auto downfold_gloc_with_projector_from_h5(const Mf &mf, const std::string &df_params)
  -> nda::array<ComplexType, 5> {
    auto parser = InputParser(df_params);
    return methods::downfold_gloc_with_projector_from_h5(mf.get_mf(), parser.get_root());
  }

  auto downfold_gloc(const Mf &mf, const std::string &df_params,
                     const nda::array<ComplexType, 5> &projector_ksIai,
                     const nda::array<long, 3> &band_window,
                     const nda::array<RealType, 2> &kpts_crys)
  -> nda::array<ComplexType, 5> {
    auto parser = InputParser(df_params);
    return methods::downfold_gloc(mf.get_mf(), parser.get_root(), projector_ksIai, band_window, kpts_crys);
  }

  auto lr_downfold_gloc(const Mf &mf, const std::string &df_params,
                        const nda::array<ComplexType, 5> &projector_ksIai,
                        const nda::array<long, 3> &band_window,
                        const nda::array<RealType, 2> &kpts_crys,
                        std::optional<nda::array<ComplexType, 5>> DeltaG_tskij,
                        const nda::array<double, 1> &q_vec)
  -> nda::array<ComplexType, 5> {
    auto parser = InputParser(df_params);
    return methods::lr_downfold_gloc(mf.get_mf(), parser.get_root(), projector_ksIai, band_window, kpts_crys,
                                     DeltaG_tskij, q_vec);
  }

  template<typename eri_handler_t>
  nda::array<ComplexType, 5>
  lr_downfold_coulomb(eri_handler_t &eri, const std::string &df_params,
                      const nda::array<ComplexType, 5> &projector_ksIai,
                      const nda::array<long, 3> &band_window,
                      const nda::array<RealType, 2> &kpts_crys,
                      std::optional<nda::array<ComplexType, 4>> DeltaW_xqPQ,
                      const nda::array<double, 1> &q_vec) {
    auto parser = InputParser(df_params);
    return methods::lr_downfold_coulomb(eri.get_eri(), parser.get_root(), projector_ksIai, band_window, kpts_crys,
                                        DeltaW_xqPQ, q_vec);
  }

  auto lr_upfold_sigma(const Mf &mf, const std::string &params,
                       const nda::array<ComplexType, 5> &projector_ksIai,
                       const nda::array<long, 3> &band_window,
                       const nda::array<RealType, 2> &kpts_crys,
                       const nda::array<ComplexType, 5> &Sigma_loc_wsIab,
                       const nda::array<double, 1> &q_vec)
  -> nda::array<ComplexType, 5> {
    auto parser = InputParser(params);
    return methods::lr_upfold_sigma(mf.get_mf(), parser.get_root(), projector_ksIai, band_window, kpts_crys,
                                    Sigma_loc_wsIab, q_vec);
  }

  template<typename eri_handler_t>
  nda::array<ComplexType, 4>
  lr_upfold_pi(eri_handler_t &eri, const std::string &params,
               const nda::array<ComplexType, 5> &projector_ksIai,
               const nda::array<long, 3> &band_window,
               const nda::array<RealType, 2> &kpts_crys,
               const nda::array<ComplexType, 5> &Pi_loc_wabcd,
               const nda::array<double, 1> &q_vec) {
    auto parser = InputParser(params);
    return methods::lr_upfold_pi(eri.get_eri(), parser.get_root(), projector_ksIai, band_window, kpts_crys,
                                 Pi_loc_wabcd, q_vec);
  }

  void downfold_1e(const Mf &mf, const std::string &df_params) {
    auto parser = InputParser(df_params);
    methods::downfolding_1e(mf.get_mf(), parser.get_root());
  }

  void dmft_embed_with_projector_from_h5(const Mf &mf, const std::string &embed_params,
                  std::optional<std::map<std::string, nda::array<ComplexType, 4> > > local_hf_potentials,
                  std::optional<std::map<std::string, nda::array<ComplexType, 5> > > local_selfenergies) {
    auto parser = InputParser(embed_params);
    methods::dmft_embed_with_projector_from_h5(mf.get_mf(), parser.get_root(), 
                        local_hf_potentials, local_selfenergies);
  }

  void dmft_embed(const Mf &mf, const std::string &embed_params,
                  const nda::array<ComplexType, 5> &projector_ksIai,
                  const nda::array<long, 3> &band_window,
                  const nda::array<RealType, 2> &kpts_crys,
                  std::optional<std::map<std::string, nda::array<ComplexType, 4> > > local_hf_potentials,
                  std::optional<std::map<std::string, nda::array<ComplexType, 5> > > local_selfenergies) {
    auto parser = InputParser(embed_params);
    methods::dmft_embed(mf.get_mf(), parser.get_root(),
                        projector_ksIai, band_window, kpts_crys,
                        local_hf_potentials, local_selfenergies);
  }


  // ----- per-cell double-counting response and basis helpers (methods/embedding/dc_response.hpp) -----

  template<int R> using carr_t = nda::array<ComplexType, R>;

  template<int R>
  auto chemistry_to_product_basis(const carr_t<R> &A) -> carr_t<R> {
    return methods::chemistry_to_product_basis(A);
  }

  template<int R>
  auto density_density_to_product_basis(const carr_t<R> &A) -> carr_t<R+2> {
    return methods::density_density_to_product_basis(A);
  }

  auto lr_eval_gw_dc_cells(const carr_t<4> &G_tsab, const carr_t<4> &dG_tsab,
                           const carr_t<5> &W_tabcd, const carr_t<5> &dW_tabcd,
                           const nda::array<long, 1> &cell_offset, const nda::array<long, 1> &cell_size,
                           const nda::array<long, 1> &orb_block) -> carr_t<4> {
    return methods::lr_eval_gw_dc_cells(G_tsab, dG_tsab, W_tabcd, dW_tabcd,
                                        methods::dc_cells_t{cell_offset, cell_size, orb_block});
  }

  auto lr_eval_pi_rpa_cells(const carr_t<4> &G_tsab, const carr_t<4> &dG_tsab,
                            const nda::array<long, 1> &cell_offset, const nda::array<long, 1> &cell_size,
                            const nda::array<long, 1> &orb_block) -> carr_t<3> {
    return methods::lr_eval_pi_rpa_cells(G_tsab, dG_tsab, methods::dc_cells_t{cell_offset, cell_size, orb_block});
  }

  auto lr_dc_response(const carr_t<5> &G_tsIab, const carr_t<5> &dG_tsIab,
                      const carr_t<5> &W_tabcd, const carr_t<5> &dW_tabcd,
                      const nda::array<long, 1> &cell_offset, const nda::array<long, 1> &cell_size,
                      const nda::array<long, 1> &orb_block) -> std::tuple<carr_t<5>, carr_t<5>> {
    return methods::lr_dc_response(G_tsIab, dG_tsIab, W_tabcd, dW_tabcd,
                                   methods::dc_cells_t{cell_offset, cell_size, orb_block});
  }


  // public template instantiation
  template std::tuple<nda::array<ComplexType, 4>, nda::array<ComplexType, 5>>
  downfold_coulomb_with_projector_from_h5(
    ThcCoulomb&, const std::string &, std::optional<std::map<std::string, nda::array<ComplexType, 5> > >);

  template std::tuple<nda::array<ComplexType, 4>, nda::array<ComplexType, 5>>
  downfold_coulomb(ThcCoulomb&, const std::string &,
                const nda::array<ComplexType, 5> &,
                const nda::array<long, 3> &,
                const nda::array<RealType, 2> &,
                std::optional<std::map<std::string, nda::array<ComplexType, 5> > >);

  template void downfold_2e(ThcCoulomb&, const std::string&, std::optional<std::map<std::string, nda::array<ComplexType, 5> > >);

  template nda::array<ComplexType, 5>
  lr_downfold_coulomb(ThcCoulomb&, const std::string &,
                      const nda::array<ComplexType, 5> &,
                      const nda::array<long, 3> &,
                      const nda::array<RealType, 2> &,
                      std::optional<nda::array<ComplexType, 4>>,
                      const nda::array<double, 1> &);

  template nda::array<ComplexType, 4>
  lr_upfold_pi(ThcCoulomb&, const std::string &,
               const nda::array<ComplexType, 5> &,
               const nda::array<long, 3> &,
               const nda::array<RealType, 2> &,
               const nda::array<ComplexType, 5> &,
               const nda::array<double, 1> &);

} // coqui_py

#include "embed_module.wrap.cxx"
