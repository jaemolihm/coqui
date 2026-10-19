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

pytest.importorskip("triqs_ctseg", reason="coqui.dmft.ctseg needs triqs_ctseg")

from triqs.gfs import BlockGf, Gf, MeshDLRImFreq, MeshImFreq
from coqui.dmft.ctseg.ctseg_utils import fill_dlr_imfreq_gf, fit_dlr_imfreq_gf

BETA, WMAX, EPS = 50.0, 3.0, 1e-10
POLES = np.array([-1.2, -0.3, 0.4, 1.5])
WEIGHTS = np.array([0.1, 0.3, 0.2, 0.05])


def _pole_sum(iw):
    """Sigma(iw) = sum_k w_k / (iw - e_k): Hermitian, vanishes at infinity, spectrum inside [-WMAX, WMAX]."""
    return np.sum(WEIGHTS / (iw[:, None] - POLES), axis=1)


def _boson_sum(iv):
    """Pi(i nu) = sum_k w_k [1/(i nu - e_k) - 1/(i nu + e_k)] for e_k > 0: real, even, vanishes at infinity."""
    e = np.abs(POLES)
    return np.sum(WEIGHTS * (1 / (iv[:, None] - e) - 1 / (iv[:, None] + e)), axis=1)


def _dense_mesh(statistic="Fermion"):
    nodes = MeshDLRImFreq(beta=BETA, statistic=statistic, w_max=WMAX, eps=EPS, symmetrize=True)
    n_max = max(abs(w.index) for w in nodes)
    return MeshImFreq(beta=BETA, statistic=statistic, n_iw=int(1.4 * n_max))


def _iw(mesh):
    return np.array([complex(w) for w in mesh])


def _block(values, mesh, name="up"):
    g = Gf(mesh=mesh, target_shape=(1, 1))
    g.data[:, 0, 0] = values
    return BlockGf(name_list=[name], block_list=[g])


def _two_blocks(mesh):
    iw = _iw(mesh)
    gs = []
    for scale in (1, 2):
        g = Gf(mesh=mesh, target_shape=(1, 1))
        g.data[:, 0, 0] = scale * _pole_sum(iw)
        gs.append(g)
    return BlockGf(name_list=["up", "dn"], block_list=gs)


def test_fit_reproduces_pole_sum():
    # fit_dlr_imfreq_gf of a noise-free fermionic pole sum (two-block BlockGf) vs the exact function at the DLR nodes,
    # which must be those of fill_dlr_imfreq_gf. Tolerance: the tau-grid Fourier transform (n_tau = 10 x mesh points),
    # not the DLR eps
    mesh = _dense_mesh()
    blocks = _two_blocks(mesh)
    fit = fit_dlr_imfreq_gf(blocks, WMAX, EPS)
    nodes = fill_dlr_imfreq_gf(blocks, WMAX, EPS)
    for bl, scale in (("up", 1), ("dn", 2)):
        assert fit[bl].mesh == nodes[bl].mesh
        exact = scale * _pole_sum(_iw(fit[bl].mesh))
        assert np.linalg.norm(fit[bl].data[:, 0, 0] - exact) / np.linalg.norm(exact) < 1e-6


def test_fit_bosonic_gf():
    # fit_dlr_imfreq_gf of a noise-free bosonic Gf (not a BlockGf), Pi-like: sum_k w_k [1/(i nu - e_k) - 1/(i nu + e_k)],
    # real, even, vanishing at infinity, vs the exact function at the bosonic DLR nodes. Tolerance as above
    mesh = _dense_mesh("Boson")
    iv = _iw(mesh)
    g = Gf(mesh=mesh, target_shape=(1, 1))
    g.data[:, 0, 0] = _boson_sum(iv)
    fit = fit_dlr_imfreq_gf(g, WMAX, EPS)
    assert fit.mesh == fill_dlr_imfreq_gf(g, WMAX, EPS).mesh
    exact = _boson_sum(_iw(fit.mesh))
    assert np.linalg.norm(fit.data[:, 0, 0] - exact) / np.linalg.norm(exact) < 1e-6


def test_fit_reduces_noise():
    # fit_dlr_imfreq_gf vs fill_dlr_imfreq_gf (copy at the DLR nodes) of a pole sum with Hermitian
    # white noise below |omega_n| = WMAX (above: noise-free, as after a tail fit), both against the exact pole sum: the
    # least-squares fit must be closer. The margin is small (ratio 0.89 +- 0.04 over 5 seeds): at low |omega_n| the DLR
    # nodes sit on nearly every Matsubara frequency, so there is little to average over
    mesh = _dense_mesh()
    iw = _iw(mesh)
    rng = np.random.default_rng(3)
    noise = 1e-4 * (rng.standard_normal(len(iw)) + 1j * rng.standard_normal(len(iw)))
    noise = 0.5 * (noise + np.conj(noise[::-1]))
    noise[np.abs(iw.imag) > WMAX] = 0.0
    g = _block(_pole_sum(iw) + noise, mesh)

    fit_nodes = fit_dlr_imfreq_gf(g, WMAX, EPS)["up"]
    smp_nodes = fill_dlr_imfreq_gf(g, WMAX, EPS)["up"]
    exact = _pole_sum(_iw(smp_nodes.mesh))
    err_fit = np.linalg.norm(fit_nodes.data[:, 0, 0] - exact)
    err_smp = np.linalg.norm(smp_nodes.data[:, 0, 0] - exact)
    assert err_fit < err_smp
