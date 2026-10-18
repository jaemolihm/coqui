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

#include "utilities/check.hpp"
#include "methods/embedding/dc_response.hpp"
#include "methods/embedding/dc_utilities.hpp"

namespace methods {

  namespace {

    long half_grid(long nts) { return (nts%2==0)? nts/2 : nts/2 + 1; }

    template<typename Array_t>
    void check_G_shape(const Array_t &G, std::string_view name) {
      utils::check(G.extent(2) == G.extent(3), "{}: G must be square in the orbital indices.", name);
      utils::check(G.extent(1) == 1 or G.extent(1) == 2, "{}: spin axis must have extent 1 or 2, got {}.",
                   name, G.extent(1));
    }

    void check_W_shape(const carray_t<4> &G, const carray_t<5> &W, std::string_view name) {
      long n = G.extent(2);
      utils::check(W.extent(1) == n and W.extent(2) == n and W.extent(3) == n and W.extent(4) == n,
                   "{}: W must be (nt_half, n, n, n, n) with n = {}.", name, n);
      utils::check(W.extent(0) == half_grid(G.extent(0)),
                   "{}: W has {} half-grid points, expected {} for nts = {}.",
                   name, W.extent(0), half_grid(G.extent(0)), G.extent(0));
    }

  } // anonymous

  // -------------------------------------------------------------------------------------------
  // Basis
  // -------------------------------------------------------------------------------------------

  carray_t<5> chemistry_to_product_basis(const carray_t<5> &A_wabcd) {
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

  carray_t<5> density_density_to_product_basis(const carray_t<3> &A_wab) {
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
  // Double counting per cell
  // -------------------------------------------------------------------------------------------

  void dc_cells_t::check(long dim_C) const {
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

  namespace {

    // The dense (nt, ns, n, n) copy of a cell block of G, zero where the orbitals are in different 1e blocks.
    carray_t<4> masked_cell_G(const carray_t<4> &G_tsab, const dc_cells_t &cells, long k) {
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
    void scatter_masked_sigma(const carray_t<4> &Sigma_k, const dc_cells_t &cells, long k, carray_t<4> &Sigma_tsab) {
      long off = cells.cell_offset(k), n = cells.cell_size(k);
      for (long a = 0; a < n; ++a) {
        for (long b = 0; b < n; ++b) {
          long ba = cells.orb_block(off + a);
          if (ba < 0 or ba != cells.orb_block(off + b)) continue;
          Sigma_tsab(nda::range::all, nda::range::all, off + a, off + b) = Sigma_k(nda::range::all, nda::range::all, a, b);
        }
      }
    }

    void check_cells_inputs(const carray_t<4> &G, const carray_t<5> &W, const dc_cells_t &cells, std::string_view name) {
      check_G_shape(G, name);
      check_W_shape(G, W, name);
      cells.check(G.extent(2));
    }

  } // anonymous

  carray_t<4> lr_eval_gw_dc_cells(const carray_t<4> &G_tsab, const carray_t<4> &dG_tsab,
                                  const carray_t<5> &W_tabcd, const carray_t<5> &dW_tabcd,
                                  const dc_cells_t &cells) {
    check_cells_inputs(G_tsab, W_tabcd, cells, "lr_eval_gw_dc_cells");
    utils::check(dG_tsab.shape() == G_tsab.shape(), "lr_eval_gw_dc_cells: dG and G shapes differ.");
    utils::check(dW_tabcd.shape() == W_tabcd.shape(), "lr_eval_gw_dc_cells: dW and W shapes differ.");
    carray_t<4> dSigma_tsab(G_tsab.shape());
    dSigma_tsab() = ComplexType(0.0);
    for (long k = 0; k < cells.cell_size.extent(0); ++k) {
      long off = cells.cell_offset(k), n = cells.cell_size(k);
      auto r = nda::range(off, off + n);
      auto Gk = masked_cell_G(G_tsab, cells, k);
      auto dGk = masked_cell_G(dG_tsab, cells, k);
      carray_t<5> Wk = W_tabcd(nda::range::all, r, r, r, r);
      carray_t<5> dWk = dW_tabcd(nda::range::all, r, r, r, r);
      carray_t<4> Sk(Gk.shape());
      Sk() = ComplexType(0.0);
      gw_sigma_dc_kernel(dGk, Wk, Sk);
      gw_sigma_dc_kernel(Gk, dWk, Sk);
      scatter_masked_sigma(Sk, cells, k, dSigma_tsab);
    }
    return dSigma_tsab;
  }

  carray_t<3> lr_eval_pi_rpa_cells(const carray_t<4> &G_tsab, const carray_t<4> &dG_tsab, const dc_cells_t &cells) {
    check_G_shape(G_tsab, "lr_eval_pi_rpa_cells");
    utils::check(dG_tsab.shape() == G_tsab.shape(), "lr_eval_pi_rpa_cells: dG and G shapes differ.");
    cells.check(G_tsab.extent(2));
    long nC = G_tsab.extent(2);
    carray_t<3> dPi_tab(half_grid(G_tsab.extent(0)), nC, nC);
    dPi_tab() = ComplexType(0.0);
    for (long k = 0; k < cells.cell_size.extent(0); ++k) {
      long off = cells.cell_offset(k), n = cells.cell_size(k);
      auto r = nda::range(off, off + n);
      auto Gk = masked_cell_G(G_tsab, cells, k);
      auto dGk = masked_cell_G(dG_tsab, cells, k);
      carray_t<3> Pk(dPi_tab.extent(0), n, n);
      Pk() = ComplexType(0.0);
      pi_density_pair_kernel(dGk, Gk, Pk);
      pi_density_pair_kernel(Gk, dGk, Pk);
      dPi_tab(nda::range::all, r, r) = Pk;
    }
    return dPi_tab;
  }

} // methods
