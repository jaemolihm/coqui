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


def _dense_mesh():
    nodes = MeshDLRImFreq(beta=BETA, statistic="Fermion", w_max=WMAX, eps=EPS, symmetrize=True)
    n_max = max(abs(w.index) for w in nodes)
    return MeshImFreq(beta=BETA, statistic="Fermion", n_iw=int(1.4 * n_max))


def _gf(values, mesh):
    g = Gf(mesh=mesh, target_shape=(1, 1))
    g.data[:, 0, 0] = values
    return g


def _at_nodes(g):
    iw = np.array([complex(w) for w in g.mesh])
    return g.data[:, 0, 0], iw


def test_fit_reproduces_pole_sum():
    # fit_dlr_imfreq_gf of a noise-free pole sum vs the pole sum at the DLR nodes, for both blocks of a BlockGf;
    # tolerance: the tau-grid Fourier transform (n_tau = 10 n_iw), not the DLR eps
    mesh = _dense_mesh()
    iw = np.array([complex(w) for w in mesh])
    blocks = BlockGf(name_list=["up", "dn"], block_list=[_gf(_pole_sum(iw), mesh), _gf(2 * _pole_sum(iw), mesh)])
    fit = fit_dlr_imfreq_gf(blocks, WMAX, EPS)
    for scale, bl in ((1, "up"), (2, "dn")):
        vals, iw_nodes = _at_nodes(fit[bl])
        exact = scale * _pole_sum(iw_nodes)
        assert np.linalg.norm(vals - exact) / np.linalg.norm(exact) < 1e-6
    # same nodes as fill_dlr_imfreq_gf, so the two can be combined node by node
    sampled = fill_dlr_imfreq_gf(blocks, WMAX, EPS)
    assert [w.index for w in fit.mesh] == [w.index for w in sampled.mesh]


def test_fit_reduces_noise():
    # fit_dlr_imfreq_gf vs fill_dlr_imfreq_gf (sampling at the nodes) of a pole sum with Hermitian white noise below
    # |omega_n| = WMAX (above: noise-free, as after a tail fit): the least-squares fit must be closer to the exact
    # pole sum at the nodes than the sampled values. The margin is small (ratio 0.89 +- 0.04 over 5 seeds): at low
    # |omega_n| the DLR nodes sit on nearly every Matsubara frequency, so there is little to average over
    mesh = _dense_mesh()
    iw = np.array([complex(w) for w in mesh])
    rng = np.random.default_rng(3)
    noise = 1e-4 * (rng.standard_normal(len(iw)) + 1j * rng.standard_normal(len(iw)))
    noise = 0.5 * (noise + np.conj(noise[::-1]))
    noise[np.abs(iw.imag) > WMAX] = 0.0
    g = _gf(_pole_sum(iw) + noise, mesh)

    vals_fit, iw_nodes = _at_nodes(fit_dlr_imfreq_gf(g, WMAX, EPS))
    vals_smp, _ = _at_nodes(fill_dlr_imfreq_gf(g, WMAX, EPS))
    exact = _pole_sum(iw_nodes)
    err_fit = np.linalg.norm(vals_fit - exact)
    err_smp = np.linalg.norm(vals_smp - exact)
    assert err_fit < err_smp
