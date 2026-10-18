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

#ifndef COQUI_DC_RESPONSE_HPP
#define COQUI_DC_RESPONSE_HPP

#include "configuration.hpp"
#include "nda/nda.hpp"

/**
 * Linear response of the GW+EDMFT double counting, evaluated per cell of the correlated space, and the basis
 * helpers that connect its arrays to the ground-state layout.
 *
 * Python twins (src/python/dmft/weiss.py): chemistry_to_product_basis, density_density_to_product_basis. The
 * ground-state maps weiss.eval_gw_dc_t and weiss.eval_pi_rpa are the reference of the per-cell responses. The
 * Python functions stay the ground-state code path; tests/test_dc_response.py asserts that both agree.
 *
 * modest (TRIQS) cannot be linked into CoQuí (it builds against a different nda), so the embedding enters only
 * through index arrays read in Python (src/python/dmft/impurity_block_map.py).
 */
namespace methods {

  template<int R>
  using carray_t = nda::array<ComplexType, R>;

  // ---------------------------------------------------------------------------------------------
  // Basis helpers
  // ---------------------------------------------------------------------------------------------

  // Twin of weiss.chemistry_to_product_basis: A_pb(..., a, b, c, d) = A_chem(..., b, a, c, d).
  carray_t<5> chemistry_to_product_basis(const carray_t<5> &A_wabcd);

  // Twin of weiss.density_density_to_product_basis: A_abcd(..., a, a, b, b) = A_ab(..., a, b), else zero.
  carray_t<5> density_density_to_product_basis(const carray_t<3> &A_wab);

  // ---------------------------------------------------------------------------------------------
  // Double counting evaluated per cell (per alpha block of the spinless, impurity-merged embedding)
  // ---------------------------------------------------------------------------------------------

  /**
   * Cells of the correlated space on which the double counting is evaluated independently.
   *  cell_offset/cell_size : the cells (alpha blocks of embedding.merge_embed_block_by_imp.slice_sigma)
   *  orb_block(o)          : the alpha block of the 1e embedding that contains orbital o, -1 if none.
   * G is kept only where both orbitals are in the same 1e block, and so is the output Sigma, as the
   * ground state evaluates the double counting on extracted blocks (embedding.extract/arr_to_blk_arr).
   */
  struct dc_cells_t {
    nda::array<long, 1> cell_offset;
    nda::array<long, 1> cell_size;
    nda::array<long, 1> orb_block;

    void check(long dim_C) const;
  };

  // dSigma_dc(t,s) = Sigma[dG, W] + Sigma[G, dW] per cell, Sigma of weiss.eval_gw_dc_t; the half tau grid assumes
  // the tau-reflection evenness of the response, which is exact for a time-reversal-even perturbation at any Q and
  // any number of orbitals (a physical electron-phonon dV in a time-reversal-symmetric crystal) and fails at O(1)
  // for a generic complex dH0. G_tsab, dG_tsab (nt, ns, nC, nC), W_tabcd, dW_tabcd (nt_half, nC, nC, nC, nC).
  carray_t<4> lr_eval_gw_dc_cells(const carray_t<4> &G_tsab, const carray_t<4> &dG_tsab,
                                  const carray_t<5> &W_tabcd, const carray_t<5> &dW_tabcd,
                                  const dc_cells_t &cells);

  // Density-only dPi_dc(t) = P(dG, G) + P(G, dG) per cell, P of weiss.eval_pi_rpa(density_only=True); both factors
  // carry the same momentum transfer. (nt_half, nC, nC), zero outside the cells. Same half tau grid assumption.
  carray_t<3> lr_eval_pi_rpa_cells(const carray_t<4> &G_tsab, const carray_t<4> &dG_tsab, const dc_cells_t &cells);

} // methods

#endif
