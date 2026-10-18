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

#include <vector>

#include "configuration.hpp"
#include "nda/nda.hpp"

/**
 * Double-counting maps of the GW+EDMFT local problem, their linear responses, and the index-array
 * description of the impurity blocks of a modest embedding.
 *
 * Python twins (src/python/dmft/weiss.py): eval_gw_dc_t, eval_pi_rpa_pair, eval_pi_rpa,
 * chemistry_to_product_basis, density_density_to_product_basis, embed_impurities (spin average),
 * scf_driver.run_gw_edmft (nspin = 1 duplication). The Python functions stay the ground-state code
 * path; tests/test_dc_response.py asserts that both agree.
 *
 * modest (TRIQS) cannot be linked into CoQuí (it builds against a different nda), so the embedding
 * enters only through index arrays read in Python (src/python/dmft/impurity_block_map.py).
 */
namespace methods {

  template<int R>
  using carray_t = nda::array<ComplexType, R>;

  // ---------------------------------------------------------------------------------------------
  // Ground-state maps on the half tau grid (mt = nts-1-t). W is in the product basis
  // W_tabcd = W_{ca,db} order of weiss.chemistry_to_product_basis; spin axis of G has extent 1 or 2.
  // ---------------------------------------------------------------------------------------------

  // Twin of weiss.eval_gw_dc_t: Sigma_ab(t) = -sum_cd G_cd(t) W(t,c,a,d,b), mt mirror.
  carray_t<4> eval_gw_dc_t(const carray_t<4> &G_tsab, const carray_t<5> &W_tabcd);

  // dSigma = Sigma[dG, W] + Sigma[G, dW]. The half tau grid assumes the tau-reflection evenness of the response,
  // which is exact for a time-reversal-even perturbation at any Q and any number of orbitals (a physical
  // electron-phonon dV in a time-reversal-symmetric crystal) and fails at O(1) for a generic complex dH0.
  carray_t<4> lr_eval_gw_dc_t(const carray_t<4> &G_tsab, const carray_t<4> &dG_tsab,
                              const carray_t<5> &W_tabcd, const carray_t<5> &dW_tabcd);

  // Twin of weiss.eval_pi_rpa_pair (density_only=False):
  // Pi[t]_{abcd} = sf * sum_s G1_bd(t) G2_ca(mt), sf = -2 (nspin = 1) or -1.
  carray_t<5> eval_pi_rpa_pair(const carray_t<4> &G1_tsab, const carray_t<4> &G2_tsab);

  // Twin of weiss.eval_pi_rpa_pair (density_only=True): Pi[t]_{ab} = sf * sum_s G1_ab(t) G2_ba(mt).
  carray_t<3> eval_pi_rpa_density_pair(const carray_t<4> &G1_tsab, const carray_t<4> &G2_tsab);

  // dPi = P(dG, G) + P(G, dG): both factors carry the same momentum transfer (the product is site-diagonal).
  // Same half tau grid assumption as lr_eval_gw_dc_t.
  carray_t<5> lr_eval_pi_rpa(const carray_t<4> &G_tsab, const carray_t<4> &dG_tsab);
  carray_t<3> lr_eval_pi_rpa_density(const carray_t<4> &G_tsab, const carray_t<4> &dG_tsab);

  // ---------------------------------------------------------------------------------------------
  // Basis and spin helpers
  // ---------------------------------------------------------------------------------------------

  // Twin of weiss.chemistry_to_product_basis: A_pb(..., a, b, c, d) = A_chem(..., b, a, c, d).
  carray_t<4> chemistry_to_product_basis(const carray_t<4> &A_abcd);
  carray_t<5> chemistry_to_product_basis(const carray_t<5> &A_wabcd);

  // Twin of weiss.density_density_to_product_basis: A_abcd(..., a, a, b, b) = A_ab(..., a, b), else zero.
  carray_t<4> density_density_to_product_basis(const carray_t<2> &A_ab);
  carray_t<5> density_density_to_product_basis(const carray_t<3> &A_wab);

  // Twin of the nspin = 1 duplication in scf_driver.run_gw_edmft: an axis of extent 1 is copied to extent 2.
  template<int R>
  carray_t<R> duplicate_spin(const carray_t<R> &A, int axis);

  // Twin of the spin average in weiss.embed_impurities: (A_up + A_dn) * 0.5, the averaged axis kept with extent 1.
  template<int R>
  carray_t<R> spin_average(const carray_t<R> &A, int axis);

  // ---------------------------------------------------------------------------------------------
  // Impurity blocks of a modest embedding, from index arrays
  // ---------------------------------------------------------------------------------------------

  /**
   * Index arrays of a triqs_modest embedding, read once in Python.
   *  alpha_offset/alpha_size : the diagonal blocks alpha of the correlated space (sigma_embed_decomp)
   *  psi(alpha, sigma, :)    : (imp, gamma, tau); imp = -1 for a block not mapped to an impurity
   *  n_gamma(imp)            : number of gamma blocks of the impurity
   *  imp_block_dim(imp)(b)   : dimension of the impurity block b = gamma + n_gamma(imp) * tau
   */
  struct imp_block_map_t {
    nda::array<long, 1> alpha_offset;
    nda::array<long, 1> alpha_size;
    nda::array<long, 3> psi;
    nda::array<long, 1> n_gamma;
    std::vector<std::vector<long>> imp_block_dim;

    long n_alpha() const { return alpha_size.extent(0); }
    long n_sigma() const { return psi.extent(1); }
    long n_impurities() const { return n_gamma.extent(0); }
    long dim_C() const;
    void check() const;
  };

  // Twin of embedding.extract of triqs_modest: X[sigma] are arrays of the correlated space (rank 2/4 static,
  // 3/5 with a leading frequency or time axis). Block (imp, gamma + n_gamma*tau) is the alpha-diagonal block of
  // the first (alpha, sigma) in row-major order that psi maps to (imp, gamma, tau).
  template<int R>
  std::vector<std::vector<carray_t<R>>>
  extract_impurity_blocks(const imp_block_map_t &map, const std::vector<carray_t<R>> &X_sigma);

  // Twin of embedding.embed of triqs_modest: every (alpha, sigma) takes the impurity block psi maps it to
  // (equivalent sites share a block); blocks with imp = -1 stay zero. Returns one array per sigma.
  template<int R>
  std::vector<carray_t<R>>
  embed_impurity_blocks(const imp_block_map_t &map, const std::vector<std::vector<carray_t<R>>> &blocks);

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

  // Sigma_dc(t,s) of eval_gw_dc_t per cell; G_tsab (nt, ns, nC, nC), W_tabcd (nt_half, nC, nC, nC, nC).
  carray_t<4> eval_gw_dc_cells(const carray_t<4> &G_tsab, const carray_t<5> &W_tabcd, const dc_cells_t &cells);
  carray_t<4> lr_eval_gw_dc_cells(const carray_t<4> &G_tsab, const carray_t<4> &dG_tsab,
                                  const carray_t<5> &W_tabcd, const carray_t<5> &dW_tabcd,
                                  const dc_cells_t &cells);

  // Density-only Pi_dc(t) of eval_pi_rpa_density_pair(G, G) per cell, (nt_half, nC, nC), zero outside the cells.
  carray_t<3> eval_pi_rpa_cells(const carray_t<4> &G_tsab, const dc_cells_t &cells);
  carray_t<3> lr_eval_pi_rpa_cells(const carray_t<4> &G_tsab, const carray_t<4> &dG_tsab, const dc_cells_t &cells);

} // methods

#endif
