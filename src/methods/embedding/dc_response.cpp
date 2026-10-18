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

#include <numeric>

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
  // Ground-state maps
  // -------------------------------------------------------------------------------------------

  carray_t<4> eval_gw_dc_t(const carray_t<4> &G_tsab, const carray_t<5> &W_tabcd) {
    check_G_shape(G_tsab, "eval_gw_dc_t");
    check_W_shape(G_tsab, W_tabcd, "eval_gw_dc_t");
    carray_t<4> Sigma_tsab(G_tsab.shape());
    Sigma_tsab() = ComplexType(0.0);
    gw_sigma_dc_kernel(G_tsab, W_tabcd, Sigma_tsab);
    return Sigma_tsab;
  }

  carray_t<4> lr_eval_gw_dc_t(const carray_t<4> &G_tsab, const carray_t<4> &dG_tsab,
                              const carray_t<5> &W_tabcd, const carray_t<5> &dW_tabcd) {
    check_G_shape(G_tsab, "lr_eval_gw_dc_t");
    utils::check(dG_tsab.shape() == G_tsab.shape(), "lr_eval_gw_dc_t: dG and G shapes differ.");
    utils::check(dW_tabcd.shape() == W_tabcd.shape(), "lr_eval_gw_dc_t: dW and W shapes differ.");
    check_W_shape(G_tsab, W_tabcd, "lr_eval_gw_dc_t");
    carray_t<4> dSigma_tsab(G_tsab.shape());
    dSigma_tsab() = ComplexType(0.0);
    gw_sigma_dc_kernel(dG_tsab, W_tabcd, dSigma_tsab);
    gw_sigma_dc_kernel(G_tsab, dW_tabcd, dSigma_tsab);
    return dSigma_tsab;
  }

  carray_t<5> eval_pi_rpa_pair(const carray_t<4> &G1_tsab, const carray_t<4> &G2_tsab) {
    utils::check(G1_tsab.shape() == G2_tsab.shape(), "eval_pi_rpa_pair: G1 and G2 shapes differ.");
    check_G_shape(G1_tsab, "eval_pi_rpa_pair");
    long n = G1_tsab.extent(2);
    carray_t<5> Pi_tabcd(half_grid(G1_tsab.extent(0)), n, n, n, n);
    Pi_tabcd() = ComplexType(0.0);
    pi_pair_kernel(G1_tsab, G2_tsab, Pi_tabcd);
    return Pi_tabcd;
  }

  carray_t<3> eval_pi_rpa_density_pair(const carray_t<4> &G1_tsab, const carray_t<4> &G2_tsab) {
    utils::check(G1_tsab.shape() == G2_tsab.shape(), "eval_pi_rpa_density_pair: G1 and G2 shapes differ.");
    check_G_shape(G1_tsab, "eval_pi_rpa_density_pair");
    long n = G1_tsab.extent(2);
    carray_t<3> Pi_tab(half_grid(G1_tsab.extent(0)), n, n);
    Pi_tab() = ComplexType(0.0);
    pi_density_pair_kernel(G1_tsab, G2_tsab, Pi_tab);
    return Pi_tab;
  }

  carray_t<5> lr_eval_pi_rpa(const carray_t<4> &G_tsab, const carray_t<4> &dG_tsab) {
    utils::check(dG_tsab.shape() == G_tsab.shape(), "lr_eval_pi_rpa: dG and G shapes differ.");
    check_G_shape(G_tsab, "lr_eval_pi_rpa");
    long n = G_tsab.extent(2);
    carray_t<5> dPi_tabcd(half_grid(G_tsab.extent(0)), n, n, n, n);
    dPi_tabcd() = ComplexType(0.0);
    pi_pair_kernel(dG_tsab, G_tsab, dPi_tabcd);
    pi_pair_kernel(G_tsab, dG_tsab, dPi_tabcd);
    return dPi_tabcd;
  }

  carray_t<3> lr_eval_pi_rpa_density(const carray_t<4> &G_tsab, const carray_t<4> &dG_tsab) {
    utils::check(dG_tsab.shape() == G_tsab.shape(), "lr_eval_pi_rpa_density: dG and G shapes differ.");
    check_G_shape(G_tsab, "lr_eval_pi_rpa_density");
    long n = G_tsab.extent(2);
    carray_t<3> dPi_tab(half_grid(G_tsab.extent(0)), n, n);
    dPi_tab() = ComplexType(0.0);
    pi_density_pair_kernel(dG_tsab, G_tsab, dPi_tab);
    pi_density_pair_kernel(G_tsab, dG_tsab, dPi_tab);
    return dPi_tab;
  }

  // -------------------------------------------------------------------------------------------
  // Basis and spin helpers
  // -------------------------------------------------------------------------------------------

  carray_t<4> chemistry_to_product_basis(const carray_t<4> &A_abcd) {
    long n = A_abcd.extent(0);
    utils::check(A_abcd.extent(1) == n and A_abcd.extent(2) == n and A_abcd.extent(3) == n,
                 "chemistry_to_product_basis: the last four axes must have equal length.");
    carray_t<4> A_pb(A_abcd.shape());
    for (long a = 0; a < n; ++a)
      for (long b = 0; b < n; ++b)
        for (long c = 0; c < n; ++c)
          for (long d = 0; d < n; ++d)
            A_pb(a, b, c, d) = A_abcd(b, a, c, d);
    return A_pb;
  }

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

  carray_t<4> density_density_to_product_basis(const carray_t<2> &A_ab) {
    long n = A_ab.extent(0);
    utils::check(A_ab.extent(1) == n, "density_density_to_product_basis: last two axes must be equal.");
    carray_t<4> A_abcd(n, n, n, n);
    A_abcd() = ComplexType(0.0);
    for (long a = 0; a < n; ++a)
      for (long b = 0; b < n; ++b)
        A_abcd(a, a, b, b) = A_ab(a, b);
    return A_abcd;
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

  namespace {
    // A (C-layout, contiguous) viewed as (outer, extent(axis), inner)
    template<int R>
    std::array<long, 3> split_at_axis(const carray_t<R> &A, int axis) {
      utils::check(axis >= 0 and axis < R, "axis {} out of range for rank {}.", axis, R);
      long outer = 1, inner = 1;
      for (int i = 0; i < axis; ++i) outer *= A.extent(i);
      for (int i = axis+1; i < R; ++i) inner *= A.extent(i);
      return {outer, A.extent(axis), inner};
    }
  }

  template<int R>
  carray_t<R> duplicate_spin(const carray_t<R> &A, int axis) {
    auto [outer, n_ax, inner] = split_at_axis<R>(A, axis);
    utils::check(n_ax == 1, "duplicate_spin: axis {} must have extent 1, got {}.", axis, n_ax);
    auto shape = A.shape();
    shape[axis] = 2;
    carray_t<R> B(shape);
    for (long o = 0; o < outer; ++o)
      for (long s = 0; s < 2; ++s)
        for (long i = 0; i < inner; ++i)
          B.data()[(o*2 + s)*inner + i] = A.data()[o*inner + i];
    return B;
  }

  template<int R>
  carray_t<R> spin_average(const carray_t<R> &A, int axis) {
    auto [outer, n_ax, inner] = split_at_axis<R>(A, axis);
    utils::check(n_ax == 2, "spin_average: axis {} must have extent 2, got {}.", axis, n_ax);
    auto shape = A.shape();
    shape[axis] = 1;
    carray_t<R> B(shape);
    for (long o = 0; o < outer; ++o)
      for (long i = 0; i < inner; ++i)
        B.data()[o*inner + i] = (A.data()[(o*2)*inner + i] + A.data()[(o*2 + 1)*inner + i]) * 0.5;
    return B;
  }

  template carray_t<4> duplicate_spin<4>(const carray_t<4> &, int);
  template carray_t<5> duplicate_spin<5>(const carray_t<5> &, int);
  template carray_t<4> spin_average<4>(const carray_t<4> &, int);
  template carray_t<5> spin_average<5>(const carray_t<5> &, int);

  // -------------------------------------------------------------------------------------------
  // Impurity blocks of a modest embedding
  // -------------------------------------------------------------------------------------------

  long imp_block_map_t::dim_C() const {
    return std::accumulate(alpha_size.begin(), alpha_size.end(), 0L);
  }

  void imp_block_map_t::check() const {
    utils::check(alpha_offset.extent(0) == n_alpha(), "imp_block_map_t: alpha_offset and alpha_size differ in length.");
    utils::check(psi.extent(0) == n_alpha() and psi.extent(2) == 3,
                 "imp_block_map_t: psi must have shape (n_alpha, n_sigma, 3).");
    utils::check(long(imp_block_dim.size()) == n_impurities(),
                 "imp_block_map_t: imp_block_dim has {} impurities, n_gamma has {}.",
                 imp_block_dim.size(), n_impurities());
    long off = 0;
    for (long a = 0; a < n_alpha(); ++a) {
      utils::check(alpha_offset(a) == off, "imp_block_map_t: alpha blocks must be contiguous and in order.");
      off += alpha_size(a);
      for (long s = 0; s < n_sigma(); ++s) {
        long imp = psi(a, s, 0), gamma = psi(a, s, 1), tau = psi(a, s, 2);
        if (imp < 0) continue;
        utils::check(imp < n_impurities(), "imp_block_map_t: psi({},{}) has impurity {} out of range.", a, s, imp);
        long b = gamma + n_gamma(imp) * tau;
        utils::check(gamma >= 0 and gamma < n_gamma(imp) and tau >= 0 and b < long(imp_block_dim[imp].size()),
                     "imp_block_map_t: psi({},{}) = ({},{},{}) is not a block of the impurity.", a, s, imp, gamma, tau);
        utils::check(imp_block_dim[imp][b] == alpha_size(a),
                     "imp_block_map_t: alpha {} has size {} but block ({},{},{}) has dimension {}.",
                     a, alpha_size(a), imp, gamma, tau, imp_block_dim[imp][b]);
      }
    }
  }

  namespace {

    template<int R>
    auto diag_block(auto &&X, long off, long n) {
      auto r = nda::range(off, off + n);
      auto all = nda::range::all;
      if constexpr (R == 2) return X(r, r);
      else if constexpr (R == 3) return X(all, r, r);
      else if constexpr (R == 4) return X(r, r, r, r);
      else return X(all, r, r, r, r);
    }

    template<int R>
    constexpr bool has_frequency = (R == 3 or R == 5);

    template<int R>
    carray_t<R> zero_block(long dim, long n_w) {
      carray_t<R> A;
      if constexpr (R == 2) A = carray_t<R>(dim, dim);
      else if constexpr (R == 3) A = carray_t<R>(n_w, dim, dim);
      else if constexpr (R == 4) A = carray_t<R>(dim, dim, dim, dim);
      else A = carray_t<R>(n_w, dim, dim, dim, dim);
      A() = ComplexType(0.0);
      return A;
    }

  } // anonymous

  template<int R>
  std::vector<std::vector<carray_t<R>>>
  extract_impurity_blocks(const imp_block_map_t &map, const std::vector<carray_t<R>> &X_sigma) {
    map.check();
    utils::check(long(X_sigma.size()) == map.n_sigma(),
                 "extract_impurity_blocks: got {} spin channels, the embedding has n_sigma = {}.",
                 X_sigma.size(), map.n_sigma());
    long dim_C = map.dim_C();
    long n_w = 0;
    if constexpr (has_frequency<R>) n_w = X_sigma[0].extent(0);
    for (auto &X : X_sigma) {
      utils::check(X.extent(has_frequency<R> ? 1 : 0) == dim_C,
                   "extract_impurity_blocks: dimension mismatch, expected C-space dimension {}.", dim_C);
      if constexpr (has_frequency<R>)
        utils::check(X.extent(0) == n_w, "extract_impurity_blocks: spin channels have different frequency extents.");
    }

    std::vector<std::vector<carray_t<R>>> blocks(map.n_impurities());
    for (long imp = 0; imp < map.n_impurities(); ++imp) {
      long nb = long(map.imp_block_dim[imp].size());
      for (long b = 0; b < nb; ++b) blocks[imp].push_back(zero_block<R>(map.imp_block_dim[imp][b], n_w));
    }

    // The first (alpha, sigma) mapped to a block, in row-major order, provides it.
    std::vector<std::vector<bool>> filled(map.n_impurities());
    for (long imp = 0; imp < map.n_impurities(); ++imp) filled[imp].assign(blocks[imp].size(), false);
    for (long a = 0; a < map.n_alpha(); ++a) {
      for (long s = 0; s < map.n_sigma(); ++s) {
        long imp = map.psi(a, s, 0);
        if (imp < 0) continue;
        long b = map.psi(a, s, 1) + map.n_gamma(imp) * map.psi(a, s, 2);
        if (filled[imp][b]) continue;
        filled[imp][b] = true;
        blocks[imp][b] = diag_block<R>(X_sigma[s], map.alpha_offset(a), map.alpha_size(a));
      }
    }
    for (long imp = 0; imp < map.n_impurities(); ++imp)
      for (size_t b = 0; b < blocks[imp].size(); ++b)
        utils::check(filled[imp][b], "extract_impurity_blocks: impurity {} block {} is not mapped from any alpha.", imp, b);
    return blocks;
  }

  template<int R>
  std::vector<carray_t<R>>
  embed_impurity_blocks(const imp_block_map_t &map, const std::vector<std::vector<carray_t<R>>> &blocks) {
    map.check();
    utils::check(long(blocks.size()) == map.n_impurities(),
                 "embed_impurity_blocks: got {} impurities, the embedding has {}.", blocks.size(), map.n_impurities());
    // Frequency extent of the first block, if any
    long n_w = 0;
    bool n_w_set = false;
    for (long imp = 0; imp < map.n_impurities(); ++imp) {
      utils::check(blocks[imp].size() == map.imp_block_dim[imp].size(),
                   "embed_impurity_blocks: impurity {} has {} blocks, the embedding expects {}.",
                   imp, blocks[imp].size(), map.imp_block_dim[imp].size());
      for (size_t b = 0; b < blocks[imp].size(); ++b) {
        long d = blocks[imp][b].extent(has_frequency<R> ? 1 : 0);
        utils::check(d == map.imp_block_dim[imp][b],
                     "embed_impurity_blocks: impurity {} block {} has dimension {}, the embedding expects {}.",
                     imp, b, d, map.imp_block_dim[imp][b]);
        if constexpr (has_frequency<R>) {
          if (not n_w_set) { n_w = blocks[imp][b].extent(0); n_w_set = true; }
          utils::check(blocks[imp][b].extent(0) == n_w,
                       "embed_impurity_blocks: impurity {} block {} has {} frequency points, expected {}.",
                       imp, b, blocks[imp][b].extent(0), n_w);
        }
      }
    }

    long dim_C = map.dim_C();
    std::vector<carray_t<R>> X_sigma;
    for (long s = 0; s < map.n_sigma(); ++s) X_sigma.push_back(zero_block<R>(dim_C, n_w));
    for (long a = 0; a < map.n_alpha(); ++a) {
      for (long s = 0; s < map.n_sigma(); ++s) {
        long imp = map.psi(a, s, 0);
        if (imp < 0) continue;
        long b = map.psi(a, s, 1) + map.n_gamma(imp) * map.psi(a, s, 2);
        diag_block<R>(X_sigma[s], map.alpha_offset(a), map.alpha_size(a)) = blocks[imp][b];
      }
    }
    return X_sigma;
  }

#define COQUI_INSTANTIATE_BLOCKS(R)                                                                           \
  template std::vector<std::vector<carray_t<R>>>                                                              \
  extract_impurity_blocks<R>(const imp_block_map_t &, const std::vector<carray_t<R>> &);                      \
  template std::vector<carray_t<R>>                                                                           \
  embed_impurity_blocks<R>(const imp_block_map_t &, const std::vector<std::vector<carray_t<R>>> &);

  COQUI_INSTANTIATE_BLOCKS(2)
  COQUI_INSTANTIATE_BLOCKS(3)
  COQUI_INSTANTIATE_BLOCKS(4)
  COQUI_INSTANTIATE_BLOCKS(5)
#undef COQUI_INSTANTIATE_BLOCKS

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

  carray_t<4> eval_gw_dc_cells(const carray_t<4> &G_tsab, const carray_t<5> &W_tabcd, const dc_cells_t &cells) {
    check_cells_inputs(G_tsab, W_tabcd, cells, "eval_gw_dc_cells");
    carray_t<4> Sigma_tsab(G_tsab.shape());
    Sigma_tsab() = ComplexType(0.0);
    for (long k = 0; k < cells.cell_size.extent(0); ++k) {
      long off = cells.cell_offset(k), n = cells.cell_size(k);
      auto r = nda::range(off, off + n);
      auto Gk = masked_cell_G(G_tsab, cells, k);
      carray_t<5> Wk = W_tabcd(nda::range::all, r, r, r, r);
      carray_t<4> Sk(Gk.shape());
      Sk() = ComplexType(0.0);
      gw_sigma_dc_kernel(Gk, Wk, Sk);
      scatter_masked_sigma(Sk, cells, k, Sigma_tsab);
    }
    return Sigma_tsab;
  }

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

  carray_t<3> eval_pi_rpa_cells(const carray_t<4> &G_tsab, const dc_cells_t &cells) {
    check_G_shape(G_tsab, "eval_pi_rpa_cells");
    cells.check(G_tsab.extent(2));
    long nC = G_tsab.extent(2);
    carray_t<3> Pi_tab(half_grid(G_tsab.extent(0)), nC, nC);
    Pi_tab() = ComplexType(0.0);
    for (long k = 0; k < cells.cell_size.extent(0); ++k) {
      long off = cells.cell_offset(k), n = cells.cell_size(k);
      auto r = nda::range(off, off + n);
      auto Gk = masked_cell_G(G_tsab, cells, k);
      carray_t<3> Pk(Pi_tab.extent(0), n, n);
      Pk() = ComplexType(0.0);
      pi_density_pair_kernel(Gk, Gk, Pk);
      Pi_tab(nda::range::all, r, r) = Pk;
    }
    return Pi_tab;
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
