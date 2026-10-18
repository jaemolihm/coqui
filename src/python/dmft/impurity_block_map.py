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

import re

import numpy as np


def _alpha_sizes(embedding):
    """
    Sizes of the diagonal blocks alpha of the correlated space (sigma_embed_decomp of the modest embedding).
    modest exposes no accessor, so the size is the dimension of the impurity block psi maps alpha to; a block that
    is mapped to no impurity is read from the printed decomposition.
    """
    psi = embedding.psi_map
    n_alpha, n_sigma = psi.shape
    sizes = np.full(n_alpha, -1, dtype=np.int64)
    for alpha in range(n_alpha):
        for sigma in range(n_sigma):
            blk = psi[alpha, sigma]
            if blk.imp_idx < 0:
                continue
            size = embedding.imp_decomposition(blk.imp_idx)[blk.gamma]
            if sizes[alpha] >= 0 and sizes[alpha] != size:
                raise ValueError(f"alpha block {alpha} is mapped to impurity blocks of different dimensions.")
            sizes[alpha] = size

    if np.any(sizes < 0):
        match = re.search(r"dim_α:([\d\s]+)\n", embedding.description(True))
        decomp = [int(x) for x in match.group(1).split()] if match else []
        if len(decomp) != n_alpha:
            raise ValueError("Cannot read the sizes of the alpha blocks that are mapped to no impurity.")
        for alpha in np.nonzero(sizes < 0)[0]:
            sizes[alpha] = decomp[alpha]
    return sizes


class ImpurityBlockMap:
    """
    Index arrays of the alpha blocks of a triqs_modest embedding, read once from the embedding object.
    The per-cell double counting (coqui.embed) is driven by these arrays, so that modest is not needed in C++.

    Parameters
    ----------
    embedding_1e : modest.embedding.Embedding
        Embedding of the one-body quantities (G, Sigma): alpha blocks of the correlated space, spin channel sigma,
        and psi(alpha, sigma) = (imp, gamma, tau).
    embedding_2e : modest.embedding.Embedding, optional
        Spinless embedding of the two-body quantities (W, Pi), `embedding.merge_embed_block_by_imp.slice_sigma`
        as in run_gw_edmft. Its alpha blocks are the cells on which the double counting is evaluated; each must
        contain whole alpha blocks of embedding_1e.

    Attributes (1e)
    ----------------
    alpha_offset, alpha_size : (n_alpha,) int64

    Attributes (2e)
    ----------------
    cell_offset, cell_size : (n_cell,) int64
    orb_block : (dim_C,) int64, the alpha block of embedding_1e of each orbital (-1: none)
    """

    def __init__(self, embedding_1e, embedding_2e=None):
        psi = embedding_1e.psi_map
        self.alpha_size = _alpha_sizes(embedding_1e)
        self.alpha_offset = np.concatenate(([0], np.cumsum(self.alpha_size)[:-1])).astype(np.int64)
        self.dim_C = int(self.alpha_size.sum())

        self.orb_block = np.full(self.dim_C, -1, dtype=np.int64)
        for alpha in range(len(self.alpha_size)):
            if any(blk.imp_idx >= 0 for blk in psi[alpha]):
                off, size = self.alpha_offset[alpha], self.alpha_size[alpha]
                self.orb_block[off:off+size] = alpha

        self.cell_offset = self.cell_size = None
        if embedding_2e is not None:
            self._set_cells(embedding_2e)

    def _set_cells(self, embedding_2e):
        sizes = _alpha_sizes(embedding_2e)
        if int(sizes.sum()) != self.dim_C:
            raise ValueError(f"The two embeddings have different correlated-space dimensions "
                             f"({self.dim_C} and {int(sizes.sum())}).")
        offsets = np.concatenate(([0], np.cumsum(sizes)[:-1])).astype(np.int64)

        # Each alpha block of the 1e embedding must sit inside one cell.
        for alpha, (off, size) in enumerate(zip(self.alpha_offset, self.alpha_size)):
            cell = np.searchsorted(offsets, off, side="right") - 1
            if off + size > offsets[cell] + sizes[cell]:
                raise ValueError(f"alpha block {alpha} of the 1e embedding straddles two cells of the 2e embedding.")
        self.cell_offset, self.cell_size = offsets, sizes.astype(np.int64)

    @property
    def cell_args(self):
        """Keyword arguments of the coqui.embed per-cell double-counting response functions."""
        if self.cell_offset is None:
            raise ValueError("ImpurityBlockMap was built without embedding_2e: no cells.")
        return {"cell_offset": self.cell_offset, "cell_size": self.cell_size, "orb_block": self.orb_block}
