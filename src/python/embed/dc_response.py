"""
==========================================================================
CoQuí: Correlated Quantum ínterface

Copyright (c) 2022-2026 Simons Foundation & The CoQuí developer team

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==========================================================================
"""

# Double-counting maps of the GW+EDMFT local problem, their linear responses, and the impurity blocks of a modest
# embedding, evaluated in C++ (methods/embedding/dc_response.hpp). The impurity-block index arrays come from
# coqui.dmft.ImpurityBlockMap.

import numpy as np

import coqui._lib.embed_module as embed_cxx


def _c(x):
    return np.ascontiguousarray(x, dtype=complex)


def _i(x):
    return np.ascontiguousarray(x, dtype=np.int64)


def eval_gw_dc_t(G_tsab, W_tabcd):
    """Sigma_ab(t) = -sum_cd G_cd(t) W(t,c,a,d,b) on the half tau grid with the mt mirror (C++ twin of
    coqui.dmft.weiss.eval_gw_dc_t). G_tsab (nt, ns, n, n), W_tabcd (nt_half, n, n, n, n) in the product basis."""
    return embed_cxx.eval_gw_dc_t(_c(G_tsab), _c(W_tabcd))


def lr_eval_gw_dc_t(G_tsab, dG_tsab, W_tabcd, dW_tabcd):
    """dSigma = Sigma[dG, W] + Sigma[G, dW]."""
    return embed_cxx.lr_eval_gw_dc_t(_c(G_tsab), _c(dG_tsab), _c(W_tabcd), _c(dW_tabcd))


def eval_pi_rpa_pair(G1_tsab, G2_tsab, *, density_only=False):
    """Pi[t]_abcd = sf * G1_bd(t) G2_ca(mt), or Pi[t]_ab = sf * G1_ab(t) G2_ba(mt) if density_only, spin summed,
    sf = -2 (ns = 1) or -1 (C++ twin of coqui.dmft.weiss.eval_pi_rpa_pair)."""
    f = embed_cxx.eval_pi_rpa_density_pair if density_only else embed_cxx.eval_pi_rpa_pair
    return f(_c(G1_tsab), _c(G2_tsab))


def lr_eval_pi_rpa(G_tsab, dG_tsab, *, density_only=True):
    """dPi = P(dG, G) + P(G, dG)."""
    f = embed_cxx.lr_eval_pi_rpa_density if density_only else embed_cxx.lr_eval_pi_rpa
    return f(_c(G_tsab), _c(dG_tsab))


def chemistry_to_product_basis(A):
    """A_pb[..., a, b, c, d] = A[..., b, a, c, d] for rank 4 or 5 (C++ twin of coqui.dmft.weiss)."""
    return embed_cxx.chemistry_to_product_basis(_c(A))


def density_density_to_product_basis(A):
    """A_abcd[..., a, a, b, b] = A_ab[..., a, b] for rank 2 or 3 (C++ twin of coqui.dmft.weiss)."""
    return embed_cxx.density_density_to_product_basis(_c(A))


def duplicate_spin(A, axis):
    """Copy an axis of extent 1 to extent 2 (the nspin = 1 duplication of coqui.dmft.run_gw_edmft)."""
    return embed_cxx.duplicate_spin(_c(A), int(axis))


def spin_average(A, axis):
    """(A_up + A_dn) / 2 along `axis` of extent 2, kept with extent 1 (spin average of embed_impurities)."""
    return embed_cxx.spin_average(_c(A), int(axis))


def extract_impurity_blocks(X_sigma, *, alpha_offset, alpha_size, psi, n_gamma, imp_block_dim):
    """C++ twin of triqs_modest embedding.extract. X_sigma: list over sigma of arrays of rank 2-5 (rank 3 and 5
    with a leading frequency/time axis). Returns [imp][gamma + n_gamma[imp] * tau]. Index arrays:
    coqui.dmft.ImpurityBlockMap.block_args."""
    return embed_cxx.extract_impurity_blocks([_c(x) for x in X_sigma], _i(alpha_offset), _i(alpha_size), _i(psi),
                                             _i(n_gamma), [list(map(int, d)) for d in imp_block_dim])


def embed_impurity_blocks(blocks, *, alpha_offset, alpha_size, psi, n_gamma, imp_block_dim):
    """C++ twin of triqs_modest embedding.embed. blocks: [imp][block] arrays. Returns a list over sigma."""
    return embed_cxx.embed_impurity_blocks([[_c(b) for b in imp] for imp in blocks], _i(alpha_offset),
                                           _i(alpha_size), _i(psi), _i(n_gamma),
                                           [list(map(int, d)) for d in imp_block_dim])


def eval_gw_dc_cells(G_tsab, W_tabcd, *, cell_offset, cell_size, orb_block):
    """eval_gw_dc_t on every cell of the correlated space (all equivalent sites, no representative):
    G_tsab (nt, ns, nC, nC), W_tabcd (nt_half, nC, nC, nC, nC). Index arrays: ImpurityBlockMap.cell_args."""
    return embed_cxx.eval_gw_dc_cells(_c(G_tsab), _c(W_tabcd), _i(cell_offset), _i(cell_size), _i(orb_block))


def lr_eval_gw_dc_cells(G_tsab, dG_tsab, W_tabcd, dW_tabcd, *, cell_offset, cell_size, orb_block):
    """dSigma_dc per cell: Sigma[dG, W] + Sigma[G, dW]."""
    return embed_cxx.lr_eval_gw_dc_cells(_c(G_tsab), _c(dG_tsab), _c(W_tabcd), _c(dW_tabcd),
                                         _i(cell_offset), _i(cell_size), _i(orb_block))


def eval_pi_rpa_cells(G_tsab, *, cell_offset, cell_size, orb_block):
    """Density-only Pi_dc(t) of G on every cell, (nt_half, nC, nC)."""
    return embed_cxx.eval_pi_rpa_cells(_c(G_tsab), _i(cell_offset), _i(cell_size), _i(orb_block))


def lr_eval_pi_rpa_cells(G_tsab, dG_tsab, *, cell_offset, cell_size, orb_block):
    """dPi_dc per cell: P(dG, G) + P(G, dG), density-only."""
    return embed_cxx.lr_eval_pi_rpa_cells(_c(G_tsab), _c(dG_tsab), _i(cell_offset), _i(cell_size), _i(orb_block))
