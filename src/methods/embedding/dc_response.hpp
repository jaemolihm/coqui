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

#include <string_view>

#include "configuration.hpp"
#include "nda/nda.hpp"
#include "utilities/check.hpp"
#include "methods/embedding/dc_kernels.hpp"

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

  namespace dc_detail {

    inline long half_grid(long nts) { return (nts%2==0)? nts/2 : nts/2 + 1; }

    template<typename Array_t>
    void check_G_shape(const Array_t &G, std::string_view name) {
      utils::check(G.extent(2) == G.extent(3), "{}: G must be square in the orbital indices.", name);
      utils::check(G.extent(1) == 1 or G.extent(1) == 2, "{}: spin axis must have extent 1 or 2, got {}.",
                   name, G.extent(1));
    }

    inline void check_W_shape(const carray_t<4> &G, const carray_t<5> &W, std::string_view name) {
      long n = G.extent(2);
      utils::check(W.extent(1) == n and W.extent(2) == n and W.extent(3) == n and W.extent(4) == n,
                   "{}: W must be (nt_half, n, n, n, n) with n = {}.", name, n);
      utils::check(W.extent(0) == half_grid(G.extent(0)),
                   "{}: W has {} half-grid points, expected {} for nts = {}.",
                   name, W.extent(0), half_grid(G.extent(0)), G.extent(0));
    }

    // The dense (nt, ns, n, n) copy of a cell block of G, zero where the orbitals are in different 1e blocks.
    inline carray_t<4> masked_cell_G(const carray_t<4> &G_tsab, const dc_cells_t &cells, long k) {
      long off = cells.cell_offset(k), n = cells.cell_size(k);
      carray_t<4> Gk(G_tsab.extent(0), G_tsab.extent(1), n, n);
      Gk() = ComplexType(0.0);
      for (long a = 0; a < n; ++a) {
        for (long b = 0; b < n; ++b) {
          long ba = cells.orb_block(off + a);
          if (ba < 0 or ba != cells.orb_block(off + b)) continue;
          Gk(nda::range::all, nda::range::all, a, b) = G_tsab(nda::range::all, nda::range::all, off + a, off + b);
        }
      }
      return Gk;
    }

    // Writes the same-1e-block entries of Sigma_k into Sigma_tsab (the cell blocks do not overlap).
    inline void scatter_masked_sigma(const carray_t<4> &Sigma_k, const dc_cells_t &cells, long k, carray_t<4> &Sigma_tsab) {
      long off = cells.cell_offset(k), n = cells.cell_size(k);
      for (long a = 0; a < n; ++a) {
        for (long b = 0; b < n; ++b) {
          long ba = cells.orb_block(off + a);
          if (ba < 0 or ba != cells.orb_block(off + b)) continue;
          Sigma_tsab(nda::range::all, nda::range::all, off + a, off + b) = Sigma_k(nda::range::all, nda::range::all, a, b);
        }
      }
    }

    inline void check_cells_inputs(const carray_t<4> &G, const carray_t<5> &W, const dc_cells_t &cells, std::string_view name) {
      check_G_shape(G, name);
      check_W_shape(G, W, name);
      cells.check(G.extent(2));
    }

  } // dc_detail

  // -------------------------------------------------------------------------------------------
  // Basis helpers
  // -------------------------------------------------------------------------------------------

  // Twin of weiss.chemistry_to_product_basis: A_pb(..., a, b, c, d) = A_chem(..., b, a, c, d).
  inline carray_t<5> chemistry_to_product_basis(const carray_t<5> &A_wabcd) {
    long n = A_wabcd.extent(1);
    utils::check(A_wabcd.extent(2) == n and A_wabcd.extent(3) == n and A_wabcd.extent(4) == n,
                 "chemistry_to_product_basis: the last four axes must have equal length.");
    carray_t<5> A_pb(A_wabcd.shape());
    for (long w = 0; w < A_wabcd.extent(0); ++w)
      for (long a = 0; a < n; ++a)
        for (long b = 0; b < n; ++b)
          for (long c = 0; c < n; ++c)
            for (long d = 0; d < n; ++d)
              A_pb(w, a, b, c, d) = A_wabcd(w, b, a, c, d);
    return A_pb;
  }

  // Twin of weiss.density_density_to_product_basis: A_abcd(..., a, a, b, b) = A_ab(..., a, b), else zero.
  inline carray_t<5> density_density_to_product_basis(const carray_t<3> &A_wab) {
    long n = A_wab.extent(1);
    utils::check(A_wab.extent(2) == n, "density_density_to_product_basis: last two axes must be equal.");
    carray_t<5> A_wabcd(A_wab.extent(0), n, n, n, n);
    A_wabcd() = ComplexType(0.0);
    for (long w = 0; w < A_wab.extent(0); ++w)
      for (long a = 0; a < n; ++a)
        for (long b = 0; b < n; ++b)
          A_wabcd(w, a, a, b, b) = A_wab(w, a, b);
    return A_wabcd;
  }

  // -------------------------------------------------------------------------------------------
  // Double counting evaluated per cell (per alpha block of the spinless, impurity-merged embedding)
  // -------------------------------------------------------------------------------------------

  inline void dc_cells_t::check(long dim_C) const {
    utils::check(cell_offset.extent(0) == cell_size.extent(0), "dc_cells_t: cell_offset and cell_size differ in length.");
    utils::check(orb_block.extent(0) == dim_C, "dc_cells_t: orb_block has length {}, expected {}.",
                 orb_block.extent(0), dim_C);
    long end = 0;
    for (long k = 0; k < cell_size.extent(0); ++k) {
      utils::check(cell_offset(k) >= end and cell_size(k) >= 0 and cell_offset(k) + cell_size(k) <= dim_C,
                   "dc_cells_t: cell {} is out of range or overlaps the previous one.", k);
      end = cell_offset(k) + cell_size(k);
    }
  }

  // dSigma_dc(t,s) = Sigma[dG, W] + Sigma[G, dW] per cell, Sigma of weiss.eval_gw_dc_t; the half tau grid assumes
  // the tau-reflection evenness of the response, which is exact for a time-reversal-even perturbation at any Q and
  // any number of orbitals (a physical electron-phonon dV in a time-reversal-symmetric crystal) and fails at O(1)
  // for a generic complex dH0. G_tsab, dG_tsab (nt, ns, nC, nC), W_tabcd, dW_tabcd (nt_half, nC, nC, nC, nC).
  inline carray_t<4> lr_eval_gw_dc_cells(const carray_t<4> &G_tsab, const carray_t<4> &dG_tsab,
                                  const carray_t<5> &W_tabcd, const carray_t<5> &dW_tabcd,
                                  const dc_cells_t &cells) {
    dc_detail::check_cells_inputs(G_tsab, W_tabcd, cells, "lr_eval_gw_dc_cells");
    utils::check(dG_tsab.shape() == G_tsab.shape(), "lr_eval_gw_dc_cells: dG and G shapes differ.");
    utils::check(dW_tabcd.shape() == W_tabcd.shape(), "lr_eval_gw_dc_cells: dW and W shapes differ.");
    carray_t<4> dSigma_tsab(G_tsab.shape());
    dSigma_tsab() = ComplexType(0.0);
    for (long k = 0; k < cells.cell_size.extent(0); ++k) {
      long off = cells.cell_offset(k), n = cells.cell_size(k);
      auto r = nda::range(off, off + n);
      auto Gk = dc_detail::masked_cell_G(G_tsab, cells, k);
      auto dGk = dc_detail::masked_cell_G(dG_tsab, cells, k);
      carray_t<5> Wk = W_tabcd(nda::range::all, r, r, r, r);
      carray_t<5> dWk = dW_tabcd(nda::range::all, r, r, r, r);
      carray_t<4> Sk(Gk.shape());
      Sk() = ComplexType(0.0);
      gw_sigma_dc_kernel(dGk, Wk, Sk);
      gw_sigma_dc_kernel(Gk, dWk, Sk);
      dc_detail::scatter_masked_sigma(Sk, cells, k, dSigma_tsab);
    }
    return dSigma_tsab;
  }

  // Density-only dPi_dc(t) = P(dG, G) + P(G, dG) per cell, P of weiss.eval_pi_rpa(density_only=True); both factors
  // carry the same momentum transfer. (nt_half, nC, nC), zero outside the cells. Same half tau grid assumption.
  inline carray_t<3> lr_eval_pi_rpa_cells(const carray_t<4> &G_tsab, const carray_t<4> &dG_tsab, const dc_cells_t &cells) {
    dc_detail::check_G_shape(G_tsab, "lr_eval_pi_rpa_cells");
    utils::check(dG_tsab.shape() == G_tsab.shape(), "lr_eval_pi_rpa_cells: dG and G shapes differ.");
    cells.check(G_tsab.extent(2));
    long nC = G_tsab.extent(2);
    carray_t<3> dPi_tab(dc_detail::half_grid(G_tsab.extent(0)), nC, nC);
    dPi_tab() = ComplexType(0.0);
    for (long k = 0; k < cells.cell_size.extent(0); ++k) {
      long off = cells.cell_offset(k), n = cells.cell_size(k);
      auto r = nda::range(off, off + n);
      auto Gk = dc_detail::masked_cell_G(G_tsab, cells, k);
      auto dGk = dc_detail::masked_cell_G(dG_tsab, cells, k);
      carray_t<3> Pk(dPi_tab.extent(0), n, n);
      Pk() = ComplexType(0.0);
      pi_density_pair_kernel(dGk, Gk, Pk);
      pi_density_pair_kernel(Gk, dGk, Pk);
      dPi_tab(nda::range::all, r, r) = Pk;
    }
    return dPi_tab;
  }

} // methods

#endif
