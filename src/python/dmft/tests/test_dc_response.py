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

import numpy as np
import pytest

pytest.importorskip("triqs", reason="triqs is required for dmft tests")
modest_embedding = pytest.importorskip("triqs_modest.embedding", reason="triqs_modest is required")
Embedding, ImpBlockT = modest_embedding.Embedding, modest_embedding.ImpBlockT

import coqui.embed as cq_embed
import coqui._lib.embed_module as embed_cxx
from coqui.dmft import weiss
from coqui.dmft.impurity_block_map import ImpurityBlockMap
from coqui.dmft.triqs_utils import blk_arr_to_arr, arr_to_blk_arr


def rand(rng, *shape):
    return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)


def rel(a, b):
    return np.linalg.norm(a - b) / np.linalg.norm(b)


# ---------------------------------------------------------------------------------------------------------
# C++ twins of the basis helpers
# ---------------------------------------------------------------------------------------------------------

def test_basis_helper_twins():
    # C++ chemistry_to_product_basis and density_density_to_product_basis vs weiss.* on random arrays;
    # bit-identical (pure index copies)
    rng = np.random.default_rng(1)
    n = 3
    A5 = rand(rng, 5, n, n, n, n)
    assert np.array_equal(embed_cxx.chemistry_to_product_basis(A5), weiss.chemistry_to_product_basis(A5))
    B3 = rand(rng, 5, n, n)
    assert np.array_equal(embed_cxx.density_density_to_product_basis(B3), weiss.density_density_to_product_basis(B3))


def imp_block_dims(emb):
    return [[int(dim) for _, dim in blocks] for blocks in emb.imp_block_structure]


def test_embed_impurities_pi_twin():
    # C++ density_density_to_product_basis vs the Pi output of weiss.embed_impurities on a modest embedding;
    # bit-identical (pure index copy)
    emb1 = make_embedding(**SPEC_EQUIV)
    emb2 = emb1.merge_embed_block_by_imp.slice_sigma
    rng = np.random.default_rng(8)
    nw, nw_b = 5, 3

    results = []
    for dims, dims2 in zip(imp_block_dims(emb1), imp_block_dims(emb2)):
        results.append({
            "Sigma_iw_data": [rand(rng, nw, d, d) for d in dims],
            "Sigma_infty": [rand(rng, d, d) for d in dims],
            "Pi_iw_data": [rand(rng, nw_b, d, d) for d in dims2],
            "Sigma_iw_dc_data": [rand(rng, nw, d, d) for d in dims],
            "Sigma_infty_dc": [rand(rng, d, d) for d in dims],
            "Pi_iw_dc_data": [rand(rng, nw_b, d, d) for d in dims2],
        })
    _, _, ref_pi = weiss.embed_impurities(emb1, emb2, results)

    for kind, key in (("imp", ""), ("dc", "_dc")):
        pi = emb2.embed([r["Pi_iw" + key + "_data"] for r in results])[0]
        assert np.array_equal(embed_cxx.density_density_to_product_basis(pi), ref_pi[kind])


# ---------------------------------------------------------------------------------------------------------
# Per-cell LR double counting against the ground-state Python maps
# ---------------------------------------------------------------------------------------------------------

def single_cell(n):
    return dict(cell_offset=np.array([0]), cell_size=np.array([n]), orb_block=np.zeros(n, dtype=np.int64))


SHAPES = [(nts, ns, n) for nts in (7, 8) for ns in (1, 2) for n in (1, 3)]


@pytest.mark.parametrize("nts, ns, n", SHAPES)
def test_single_cell_lr_vs_gs(nts, ns, n):
    # C++ lr_eval_gw_dc_cells / lr_eval_pi_rpa_cells (one cell) vs the central difference of weiss.eval_gw_dc_t / eval_pi_rpa
    # and vs the bilinear expansion S[dG,W] + S[G,dW] (eval_pi_rpa_pair for Pi); both maps are bilinear, so the
    # difference is exact and the tolerance 1e-13 is round-off.
    rng = np.random.default_rng(100 * nts + 3 * ns + n)
    nt_half = nts // 2 + nts % 2
    G, dG = rand(rng, nts, ns, n, n), rand(rng, nts, ns, n, n)
    W, dW = rand(rng, nt_half, n, n, n, n), rand(rng, nt_half, n, n, n, n)
    cell = single_cell(n)

    lr = cq_embed.lr_eval_gw_dc_cells(G, dG, W, dW, **cell)
    fd = 0.5 * (weiss.eval_gw_dc_t(G + dG, W + dW) - weiss.eval_gw_dc_t(G - dG, W - dW))
    assert rel(lr, fd) < 1e-13
    assert rel(lr, weiss.eval_gw_dc_t(dG, W) + weiss.eval_gw_dc_t(G, dW)) < 1e-13

    # density-only polarizability
    lr = cq_embed.lr_eval_pi_rpa_cells(G, dG, **cell)
    fd = 0.5 * (weiss.eval_pi_rpa(G + dG, density_only=True) - weiss.eval_pi_rpa(G - dG, density_only=True))
    assert rel(lr, fd) < 1e-13
    ref = (weiss.eval_pi_rpa_pair(dG, G, density_only=True) + weiss.eval_pi_rpa_pair(G, dG, density_only=True))
    assert rel(lr, ref) < 1e-13


def make_embedding(dec, imps, spec, swap=(), n_sigma=2):
    """spec[alpha] = (imp, gamma): every sigma channel of alpha maps to tau = sigma, or imp = -1 (unmapped)."""
    psi = np.empty((len(dec), n_sigma), dtype=object)
    for alpha, (imp, gamma) in enumerate(spec):
        for sigma in range(n_sigma):
            psi[alpha, sigma] = ImpBlockT(imp, gamma, sigma)
    emb = Embedding(dec, imps, psi, ["up", "dn"][:n_sigma])
    for alpha in swap:
        emb = emb.swap_sigma(alpha)
    return emb


# multi-impurity, equivalent sites: imp 0 = (gamma 0 [2], gamma 1 [1]) on two sites, imp 1 = [2] on one site
SPEC_EQUIV = dict(dec=[2, 1, 2, 1, 2], imps=[[2, 1], [2]], spec=[(0, 0), (0, 1), (0, 0), (0, 1), (1, 0)])
# a block that no impurity is mapped to
SPEC_SPECTATOR = dict(dec=[2, 3, 1, 2], imps=[[2], [1]], spec=[(0, 0), (-1, 0), (1, 0), (0, 0)])

def gs_dc_per_imp(emb1, emb2, G_C, W_C):
    """Ground-state path of run_gw_edmft: extract the first site of each impurity, evaluate eval_gw_dc_t and
    eval_pi_rpa on it, embed the result on every equivalent site. A single spin channel is duplicated as for
    nspin = 1."""
    ns = G_C.shape[1]
    G_sigma = [G_C[:, s] for s in range(ns)] if ns == 2 else [G_C[:, 0], G_C[:, 0].copy()]
    G_blocks = emb1.extract(G_sigma)
    W_blocks = emb2.extract([W_C])
    sigma_blocks, pi_blocks = [], []
    for imp in range(emb1.n_impurities):
        gf_struct = emb1.imp_block_structure[imp]
        G_arr = blk_arr_to_arr(G_blocks[imp], gf_struct)
        sigma = weiss.eval_gw_dc_t(G_arr, W_blocks[imp][0])
        sigma_blocks.append(arr_to_blk_arr(sigma, gf_struct))
        pi_blocks.append([weiss.eval_pi_rpa(G_arr, density_only=True)])
    sigma_C = np.stack(emb1.embed(sigma_blocks), axis=1)[:, :ns]
    pi_C = emb2.embed(pi_blocks)[0]
    return sigma_C, pi_C


@pytest.mark.parametrize("spec, swap", [(SPEC_EQUIV, ()), (SPEC_EQUIV, (2, 3)), (SPEC_SPECTATOR, ())],
                         ids=["equiv", "equiv_swapped", "spectator"])
@pytest.mark.parametrize("ns", [1, 2])
@pytest.mark.parametrize("nts", [7, 8])
def test_cells_lr_equals_gs_when_sites_equivalent(nts, ns, spec, swap):
    # C++ per-cell LR (cell args from ImpurityBlockMap) vs the central difference of the Python ground-state path
    # extract -> eval_gw_dc_t / eval_pi_rpa -> embed, for equivalent sites; bilinear maps, so 1e-12 is round-off.
    emb1 = make_embedding(**spec, swap=swap)
    # The spinless embedding does not depend on the swap; modest cannot merge a swapped embedding.
    emb2 = make_embedding(**spec).merge_embed_block_by_imp.slice_sigma
    bm = ImpurityBlockMap(emb1, emb2)
    rng = np.random.default_rng(10 * nts + ns)
    nt_half = nts // 2 + nts % 2
    dims1, dims2 = imp_block_dims(emb1), imp_block_dims(emb2)

    # Impurity data embedded on all sites: equivalent sites carry equal blocks. A single spin channel is spin
    # symmetric, so it is embedded without the swap.
    emb1_G = emb1 if ns == 2 else make_embedding(**spec)

    def embed_G():
        C = np.stack(emb1_G.embed([[rand(rng, nts, d, d) for d in dims] for dims in dims1]), axis=1)
        return C[:, :1] if ns == 1 else C
    G_C, dG_C = embed_G(), embed_G()
    W_C = emb2.embed([[rand(rng, nt_half, d, d, d, d) for d in dims] for dims in dims2])[0]
    dW_C = emb2.embed([[rand(rng, nt_half, d, d, d, d) for d in dims] for dims in dims2])[0]

    lr_sigma = cq_embed.lr_eval_gw_dc_cells(G_C, dG_C, W_C, dW_C, **bm.cell_args)
    lr_pi = cq_embed.lr_eval_pi_rpa_cells(G_C, dG_C, **bm.cell_args)

    sp, pp = gs_dc_per_imp(emb1, emb2, G_C + dG_C, W_C + dW_C)
    sm, pm = gs_dc_per_imp(emb1, emb2, G_C - dG_C, W_C - dW_C)
    assert rel(lr_sigma, 0.5 * (sp - sm)) < 1e-12
    assert rel(lr_pi, 0.5 * (pp - pm)) < 1e-12


def test_cells_lr_handles_broken_equivalence():
    # C++ per-cell LR vs weiss.eval_gw_dc_t / eval_pi_rpa_pair applied cell by cell with G and the output masked to
    # the 1e blocks, for different data on equivalent sites; tolerance 1e-13 is round-off.
    emb1 = make_embedding(**SPEC_EQUIV)
    emb2 = emb1.merge_embed_block_by_imp.slice_sigma
    bm = ImpurityBlockMap(emb1, emb2)
    rng = np.random.default_rng(3)
    nts, ns = 8, 2
    nt_half = nts // 2
    nC = bm.dim_C
    G, dG = rand(rng, nts, ns, nC, nC), rand(rng, nts, ns, nC, nC)
    W, dW = rand(rng, nt_half, nC, nC, nC, nC), rand(rng, nt_half, nC, nC, nC, nC)

    sigma = cq_embed.lr_eval_gw_dc_cells(G, dG, W, dW, **bm.cell_args)
    pi = cq_embed.lr_eval_pi_rpa_cells(G, dG, **bm.cell_args)

    sigma_ref = np.zeros_like(sigma)
    pi_ref = np.zeros_like(pi)
    for off, size in zip(bm.cell_offset, bm.cell_size):
        sl = slice(off, off + size)
        blk = bm.orb_block[sl]
        mask = (blk[:, None] == blk[None, :]) & (blk[:, None] >= 0)
        Gk, dGk = G[:, :, sl, sl] * mask, dG[:, :, sl, sl] * mask
        Wk, dWk = W[:, sl, sl, sl, sl], dW[:, sl, sl, sl, sl]
        sigma_ref[:, :, sl, sl] = (weiss.eval_gw_dc_t(dGk, Wk) + weiss.eval_gw_dc_t(Gk, dWk)) * mask
        pi_ref[:, sl, sl] = (weiss.eval_pi_rpa_pair(dGk, Gk, density_only=True)
                             + weiss.eval_pi_rpa_pair(Gk, dGk, density_only=True))
    assert rel(sigma, sigma_ref) < 1e-13
    assert rel(pi, pi_ref) < 1e-13

    # the two equivalent sites of impurity 0 differ
    assert np.linalg.norm(sigma[:, :, :3, :3] - sigma[:, :, 3:6, 3:6]) > 1e-3 * np.linalg.norm(sigma)


@pytest.mark.parametrize("spec", [SPEC_EQUIV, SPEC_SPECTATOR], ids=["equiv", "spectator"])
@pytest.mark.parametrize("ns", [1, 2])
@pytest.mark.parametrize("nts", [7, 8])
def test_lr_dc_response_vs_gs_chain(nts, ns, spec):
    # C++ lr_dc_response (projector layout (nt, ns, 1, n, n), W in chemistry order) vs the central difference of the
    # ground-state chain extract -> weiss.eval_gw_dc_t / eval_pi_rpa -> embed, with Pi through
    # density_density_to_product_basis; bilinear maps, so 1e-12 is round-off.
    emb1 = make_embedding(**spec)
    emb2 = make_embedding(**spec).merge_embed_block_by_imp.slice_sigma
    bm = ImpurityBlockMap(emb1, emb2)
    rng = np.random.default_rng(7 * nts + ns)
    nt_half = nts // 2 + nts % 2
    dims1, dims2 = imp_block_dims(emb1), imp_block_dims(emb2)

    def embed_G():
        C = np.stack(emb1.embed([[rand(rng, nts, d, d) for d in dims] for dims in dims1]), axis=1)
        return C[:, :1] if ns == 1 else C
    G_C, dG_C = embed_G(), embed_G()
    W_chem = emb2.embed([[rand(rng, nt_half, d, d, d, d) for d in dims] for dims in dims2])[0]
    dW_chem = emb2.embed([[rand(rng, nt_half, d, d, d, d) for d in dims] for dims in dims2])[0]

    dS, dPi = cq_embed.lr_dc_response(G_C[:, :, None], dG_C[:, :, None], W_chem, dW_chem, **bm.cell_args)
    pb = weiss.chemistry_to_product_basis
    sp, pp = gs_dc_per_imp(emb1, emb2, G_C + dG_C, pb(W_chem + dW_chem))
    sm, pm = gs_dc_per_imp(emb1, emb2, G_C - dG_C, pb(W_chem - dW_chem))
    assert dS.shape == G_C[:, :, None].shape
    assert rel(dS[:, :, 0], 0.5 * (sp - sm)) < 1e-12
    assert rel(dPi, weiss.density_density_to_product_basis(0.5 * (pp - pm))) < 1e-12

    # dW = None (W fixed) vs the same central difference with W unchanged; the Pi part does not depend on dW
    dS0, dPi0 = cq_embed.lr_dc_response(G_C[:, :, None], dG_C[:, :, None], W_chem, None, **bm.cell_args)
    sp, _ = gs_dc_per_imp(emb1, emb2, G_C + dG_C, pb(W_chem))
    sm, _ = gs_dc_per_imp(emb1, emb2, G_C - dG_C, pb(W_chem))
    assert rel(dS0[:, :, 0], 0.5 * (sp - sm)) < 1e-12
    assert rel(dPi0, dPi) < 1e-15
