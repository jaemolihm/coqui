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

# Linear response of the GW+EDMFT double counting per cell of the correlated space, evaluated in C++
# (methods/embedding/dc_response.hpp). The cell index arrays come from
# coqui.dmft.ImpurityBlockMap.cell_args.

import numpy as np

import coqui._lib.embed_module as embed_cxx


def _c(x):
    return np.ascontiguousarray(x, dtype=complex)


def _i(x):
    return np.ascontiguousarray(x, dtype=np.int64)


def lr_eval_gw_dc_cells(G_tsab, dG_tsab, W_tabcd, dW_tabcd, *, cell_offset, cell_size, orb_block):
    """dSigma_dc = Sigma[dG, W] + Sigma[G, dW] on every cell of the correlated space (all equivalent sites, no
    representative), Sigma of coqui.dmft.weiss.eval_gw_dc_t. G_tsab, dG_tsab (nt, ns, nC, nC), W_tabcd, dW_tabcd
    (nt_half, nC, nC, nC, nC) in the product basis. Index arrays: ImpurityBlockMap.cell_args."""
    return embed_cxx.lr_eval_gw_dc_cells(_c(G_tsab), _c(dG_tsab), _c(W_tabcd), _c(dW_tabcd),
                                         _i(cell_offset), _i(cell_size), _i(orb_block))


def lr_eval_pi_rpa_cells(G_tsab, dG_tsab, *, cell_offset, cell_size, orb_block):
    """Density-only dPi_dc = P(dG, G) + P(G, dG) on every cell, P of coqui.dmft.weiss.eval_pi_rpa(density_only=True);
    (nt_half, nC, nC)."""
    return embed_cxx.lr_eval_pi_rpa_cells(_c(G_tsab), _c(dG_tsab), _i(cell_offset), _i(cell_size), _i(orb_block))
