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

import itertools

import numpy as np
import pytest

pytest.importorskip("triqs", reason="triqs is required for dmft tests")
modest_embedding = pytest.importorskip("triqs_modest.embedding", reason="triqs_modest is required")
Embedding, ImpBlockT = modest_embedding.Embedding, modest_embedding.ImpBlockT

import coqui.embed as cq_embed
from coqui.dmft import weiss
from coqui.dmft.impurity_block_map import ImpurityBlockMap
from coqui.dmft.triqs_utils import blk_arr_to_arr, arr_to_blk_arr


def rand(rng, *shape):
    return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)


def rel(a, b):
    return np.linalg.norm(a - b) / np.linalg.norm(b)


# ---------------------------------------------------------------------------------------------------------
# C++ twins of the ground-state Python functions
# ---------------------------------------------------------------------------------------------------------

SHAPES = [(nts, ns, n) for nts in (7, 8) for ns in (1, 2) for n in (1, 3)]


@pytest.mark.parametrize("nts, ns, n", SHAPES)
def test_gw_dc_and_pi_twins(nts, ns, n):
    rng = np.random.default_rng(10 * nts + 3 * ns + n)
    nt_half = nts // 2 + nts % 2
    G, G2 = rand(rng, nts, ns, n, n), rand(rng, nts, ns, n, n)
    W = rand(rng, nt_half, n, n, n, n)

    assert rel(cq_embed.eval_gw_dc_t(G, W), weiss.eval_gw_dc_t(G, W)) < 1e-13
    for density_only in (False, True):
        ref = weiss.eval_pi_rpa_pair(G, G2, density_only=density_only)
        assert rel(cq_embed.eval_pi_rpa_pair(G, G2, density_only=density_only), ref) < 1e-13
        assert rel(cq_embed.eval_pi_rpa_pair(G, G, density_only=density_only),
                   weiss.eval_pi_rpa(G, density_only=density_only)) < 1e-13

    # the density-only polarizability is the Pi_{aabb} element of the full one
    full = cq_embed.eval_pi_rpa_pair(G, G2)
    dens = cq_embed.eval_pi_rpa_pair(G, G2, density_only=True)
    for a, b in itertools.product(range(n), repeat=2):
        assert np.linalg.norm(full[:, a, a, b, b] - dens[:, a, b]) < 1e-14 * max(1.0, np.linalg.norm(dens))


def test_basis_and_spin_twins():
    rng = np.random.default_rng(1)
    n = 3
    A4, A5 = rand(rng, n, n, n, n), rand(rng, 5, n, n, n, n)
    assert np.array_equal(cq_embed.chemistry_to_product_basis(A4), weiss.chemistry_to_product_basis(A4))
    assert np.array_equal(cq_embed.chemistry_to_product_basis(A5), weiss.chemistry_to_product_basis(A5))
    B2, B3 = rand(rng, n, n), rand(rng, 5, n, n)
    assert np.array_equal(cq_embed.density_density_to_product_basis(B2), weiss.density_density_to_product_basis(B2))
    assert np.array_equal(cq_embed.density_density_to_product_basis(B3), weiss.density_density_to_product_basis(B3))

    # nspin = 1 duplication of run_gw_edmft
    G = rand(rng, 6, 1, n, n)
    dup = cq_embed.duplicate_spin(G, 1)
    ref = np.stack([G[:, 0], G[:, 0].copy()], axis=1)
    assert np.array_equal(dup, ref)

    # spin average of embed_impurities
    S = rand(rng, 4, 2, 1, n, n)
    assert np.array_equal(cq_embed.spin_average(S, 1), np.sum(S, axis=1)[:, None] * 0.5)
    V = rand(rng, 2, 1, n, n)
    assert np.array_equal(cq_embed.spin_average(V, 0), np.sum(V, axis=0)[None, :] * 0.5)


# ---------------------------------------------------------------------------------------------------------
# LR kernels are the derivative of the ground-state maps
# ---------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("nts, ns, n", SHAPES)
def test_lr_is_derivative_of_gs(nts, ns, n):
    """The maps are bilinear, so S(G+dG, W+dW) - S(G-dG, W-dW) = 2 (S[dG,W] + S[G,dW]) up to rounding."""
    rng = np.random.default_rng(100 * nts + 3 * ns + n)
    nt_half = nts // 2 + nts % 2
    G, dG = rand(rng, nts, ns, n, n), rand(rng, nts, ns, n, n)
    W, dW = rand(rng, nt_half, n, n, n, n), rand(rng, nt_half, n, n, n, n)

    lr = cq_embed.lr_eval_gw_dc_t(G, dG, W, dW)
    fd = 0.5 * (cq_embed.eval_gw_dc_t(G + dG, W + dW) - cq_embed.eval_gw_dc_t(G - dG, W - dW))
    assert rel(lr, fd) < 1e-13

    for density_only in (False, True):
        lr = cq_embed.lr_eval_pi_rpa(G, dG, density_only=density_only)
        fd = 0.5 * (cq_embed.eval_pi_rpa_pair(G + dG, G + dG, density_only=density_only)
                    - cq_embed.eval_pi_rpa_pair(G - dG, G - dG, density_only=density_only))
        assert rel(lr, fd) < 1e-13


# ---------------------------------------------------------------------------------------------------------
# Impurity block map: C++ extract/embed against modest
# ---------------------------------------------------------------------------------------------------------

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

EMBEDDINGS = {
    "equiv": lambda: make_embedding(**SPEC_EQUIV),
    "equiv_swapped": lambda: make_embedding(**SPEC_EQUIV, swap=(2,)),
    "spectator": lambda: make_embedding(**SPEC_SPECTATOR),
    "equiv_1e_sigma1": lambda: make_embedding(**SPEC_EQUIV, n_sigma=1),
    "equiv_2e": lambda: make_embedding(**SPEC_EQUIV).merge_embed_block_by_imp.slice_sigma,
}


def zeros_like_shape(rank, nw, dim):
    return {2: (dim, dim), 3: (nw, dim, dim), 4: (dim,) * 4, 5: (nw, dim, dim, dim, dim)}[rank]


@pytest.mark.parametrize("name", list(EMBEDDINGS))
@pytest.mark.parametrize("rank", [2, 3, 4, 5])
def test_extract_embed_vs_modest(name, rank):
    emb = EMBEDDINGS[name]()
    bm = ImpurityBlockMap(emb)
    rng = np.random.default_rng(5)
    nw = 3

    X = [rand(rng, *zeros_like_shape(rank, nw, bm.dim_C)) for _ in range(emb.n_sigma)]
    ref = emb.extract(X)
    out = cq_embed.extract_impurity_blocks(X, **bm.block_args)
    assert len(out) == len(ref)
    for imp_out, imp_ref in zip(out, ref):
        assert len(imp_out) == len(imp_ref)
        for b_out, b_ref in zip(imp_out, imp_ref):
            assert np.array_equal(b_out, b_ref)

    blocks = [[rand(rng, *zeros_like_shape(rank, nw, dim)) for dim in dims] for dims in bm.imp_block_dim]
    ref = emb.embed(blocks)
    out = cq_embed.embed_impurity_blocks(blocks, **bm.block_args)
    assert len(out) == len(ref)
    for x_out, x_ref in zip(out, ref):
        assert np.array_equal(x_out, x_ref)


# ---------------------------------------------------------------------------------------------------------
# Embedded impurity results: C++ twin of embed_impurities
# ---------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("spin_average", [False, True])
def test_embed_impurities_twin(spin_average):
    emb1 = make_embedding(**SPEC_EQUIV)
    emb2 = emb1.merge_embed_block_by_imp.slice_sigma
    bm1, bm2 = ImpurityBlockMap(emb1), ImpurityBlockMap(emb2)
    rng = np.random.default_rng(8)
    nw, nw_b = 5, 3

    results = []
    for imp in range(emb1.n_impurities):
        dims = bm1.imp_block_dim[imp]
        dims2 = bm2.imp_block_dim[imp]
        results.append({
            "Sigma_iw_data": [rand(rng, nw, d, d) for d in dims],
            "Sigma_infty": [rand(rng, d, d) for d in dims],
            "Pi_iw_data": [rand(rng, nw_b, d, d) for d in dims2],
            "Sigma_iw_dc_data": [rand(rng, nw, d, d) for d in dims],
            "Sigma_infty_dc": [rand(rng, d, d) for d in dims],
            "Pi_iw_dc_data": [rand(rng, nw_b, d, d) for d in dims2],
        })
    ref_sigma, ref_hf, ref_pi = weiss.embed_impurities(emb1, emb2, results, spin_average)

    def embed_1e(key, rank):
        out = cq_embed.embed_impurity_blocks([r[key] for r in results], **bm1.block_args)
        if rank == 3:
            arr = np.stack(out, axis=1)[:, :, None]
            return cq_embed.spin_average(arr, 1) if spin_average else arr
        arr = np.stack(out, axis=0)[:, None]
        return cq_embed.spin_average(arr, 0) if spin_average else arr

    def embed_pi(key):
        out = cq_embed.embed_impurity_blocks([r[key] for r in results], **bm2.block_args)
        return cq_embed.density_density_to_product_basis(out[0])

    for kind, key in (("imp", ""), ("dc", "_dc")):
        assert np.array_equal(embed_1e("Sigma_iw" + key + "_data", 3), ref_sigma[kind])
        assert np.array_equal(embed_1e("Sigma_infty" + key, 2), ref_hf[kind])
        assert np.array_equal(embed_pi("Pi_iw" + key + "_data"), ref_pi[kind])


# ---------------------------------------------------------------------------------------------------------
# Per-cell double counting
# ---------------------------------------------------------------------------------------------------------

def gs_dc_per_imp(emb1, emb2, G_C, W_C):
    """Ground-state path of run_gw_edmft: extract the first site of each impurity, evaluate eval_gw_dc_t and
    eval_pi_rpa on it, embed the result on every equivalent site."""
    G_blocks = emb1.extract([G_C[:, s] for s in range(G_C.shape[1])])
    W_blocks = emb2.extract([W_C])
    sigma_blocks, pi_blocks = [], []
    for imp in range(emb1.n_impurities):
        gf_struct = emb1.imp_block_structure[imp]
        G_arr = blk_arr_to_arr(G_blocks[imp], gf_struct)
        sigma = weiss.eval_gw_dc_t(G_arr, W_blocks[imp][0])
        sigma_blocks.append(arr_to_blk_arr(sigma, gf_struct))
        pi_blocks.append([weiss.eval_pi_rpa(G_arr, density_only=True)])
    sigma_C = np.stack(emb1.embed(sigma_blocks), axis=1)
    pi_C = emb2.embed(pi_blocks)[0]
    return sigma_C, pi_C


@pytest.mark.parametrize("nts", [7, 8])
def test_cells_equal_gs_when_sites_equivalent(nts):
    emb1 = make_embedding(**SPEC_EQUIV)
    emb2 = emb1.merge_embed_block_by_imp.slice_sigma
    bm = ImpurityBlockMap(emb1, emb2)
    rng = np.random.default_rng(nts)
    nt_half = nts // 2 + nts % 2

    # Impurity data embedded on all sites: equivalent sites carry equal blocks
    G_imp = [[rand(rng, nts, d, d) for d in dims] for dims in bm.imp_block_dim]
    G_C = np.stack(emb1.embed(G_imp), axis=1)
    dG_imp = [[rand(rng, nts, d, d) for d in dims] for dims in bm.imp_block_dim]
    dG_C = np.stack(emb1.embed(dG_imp), axis=1)
    bm2 = ImpurityBlockMap(emb2)
    W_C = emb2.embed([[rand(rng, nt_half, d, d, d, d) for d in dims] for dims in bm2.imp_block_dim])[0]
    dW_C = emb2.embed([[rand(rng, nt_half, d, d, d, d) for d in dims] for dims in bm2.imp_block_dim])[0]

    sigma_ref, pi_ref = gs_dc_per_imp(emb1, emb2, G_C, W_C)
    sigma = cq_embed.eval_gw_dc_cells(G_C, W_C, **bm.cell_args)
    pi = cq_embed.eval_pi_rpa_cells(G_C, **bm.cell_args)
    assert rel(sigma, sigma_ref) < 1e-13
    assert rel(pi, pi_ref) < 1e-13

    # one spin channel with sf = -2 equals the two identical channels with sf = -1
    sigma1 = cq_embed.eval_gw_dc_cells(G_C[:, :1], W_C, **bm.cell_args)
    pi1 = cq_embed.eval_pi_rpa_cells(G_C[:, :1], **bm.cell_args)
    G_dup = np.repeat(G_C[:, :1], 2, axis=1)
    assert rel(cq_embed.eval_gw_dc_cells(G_dup, W_C, **bm.cell_args)[:, :1], sigma1) < 1e-14
    assert rel(cq_embed.eval_pi_rpa_cells(G_dup, **bm.cell_args), pi1) < 1e-14

    # LR: derivative of the GS map, and the first order of the GS map at the perturbed point
    lr_sigma = cq_embed.lr_eval_gw_dc_cells(G_C, dG_C, W_C, dW_C, **bm.cell_args)
    fd_sigma = 0.5 * (cq_embed.eval_gw_dc_cells(G_C + dG_C, W_C + dW_C, **bm.cell_args)
                      - cq_embed.eval_gw_dc_cells(G_C - dG_C, W_C - dW_C, **bm.cell_args))
    assert rel(lr_sigma, fd_sigma) < 1e-13
    lr_pi = cq_embed.lr_eval_pi_rpa_cells(G_C, dG_C, **bm.cell_args)
    fd_pi = 0.5 * (cq_embed.eval_pi_rpa_cells(G_C + dG_C, **bm.cell_args)
                   - cq_embed.eval_pi_rpa_cells(G_C - dG_C, **bm.cell_args))
    assert rel(lr_pi, fd_pi) < 1e-13

    # LR per cell equals the GS path on the perturbation when the perturbation keeps the sites equivalent
    # (S[dG,W] + S[G,dW] = (S(G+dG, W+dW) - S(G-dG, W-dW))/2 for the GS path)
    sp, pp = gs_dc_per_imp(emb1, emb2, G_C + dG_C, W_C + dW_C)
    sm, pm = gs_dc_per_imp(emb1, emb2, G_C - dG_C, W_C - dW_C)
    assert rel(lr_sigma, 0.5 * (sp - sm)) < 1e-12
    assert rel(lr_pi, 0.5 * (pp - pm)) < 1e-12


def test_cells_handle_broken_equivalence():
    """With different blocks on equivalent sites, every cell is evaluated on its own data."""
    emb1 = make_embedding(**SPEC_EQUIV)
    emb2 = emb1.merge_embed_block_by_imp.slice_sigma
    bm = ImpurityBlockMap(emb1, emb2)
    rng = np.random.default_rng(3)
    nts, ns = 8, 2
    nt_half = nts // 2
    nC = bm.dim_C
    G, W = rand(rng, nts, ns, nC, nC), rand(rng, nt_half, nC, nC, nC, nC)

    sigma = cq_embed.eval_gw_dc_cells(G, W, **bm.cell_args)
    pi = cq_embed.eval_pi_rpa_cells(G, **bm.cell_args)

    # Reference: Python DC on each cell with G masked to the 1e blocks, output masked to the 1e blocks
    sigma_ref = np.zeros_like(sigma)
    pi_ref = np.zeros_like(pi)
    for off, size in zip(bm.cell_offset, bm.cell_size):
        sl = slice(off, off + size)
        blk = bm.orb_block[sl]
        mask = (blk[:, None] == blk[None, :]) & (blk[:, None] >= 0)
        Gk = G[:, :, sl, sl] * mask
        sigma_ref[:, :, sl, sl] = weiss.eval_gw_dc_t(Gk, W[:, sl, sl, sl, sl]) * mask
        pi_ref[:, sl, sl] = weiss.eval_pi_rpa(Gk, density_only=True)
    assert rel(sigma, sigma_ref) < 1e-13
    assert rel(pi, pi_ref) < 1e-13

    # the two equivalent sites of impurity 0 differ now
    assert np.linalg.norm(sigma[:, :, :3, :3] - sigma[:, :, 3:6, 3:6]) > 1e-3 * np.linalg.norm(sigma)
