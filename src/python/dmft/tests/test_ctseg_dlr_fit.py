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
C0 = 0.7    # constant part, like Sigma_infty


def _pole_sum(iw):
    """Sigma(iw) = sum_k w_k / (iw - e_k): Hermitian, vanishes at infinity, spectrum inside [-WMAX, WMAX]."""
    return np.sum(WEIGHTS / (iw[:, None] - POLES), axis=1)


def _dense_mesh():
    nodes = MeshDLRImFreq(beta=BETA, statistic="Fermion", w_max=WMAX, eps=EPS, symmetrize=True)
    n_max = max(abs(w.index) for w in nodes)
    return MeshImFreq(beta=BETA, statistic="Fermion", n_iw=int(1.4 * n_max))


def _iw(mesh):
    return np.array([complex(w) for w in mesh])


def _block(values, mesh, name="up"):
    g = Gf(mesh=mesh, target_shape=(1, 1))
    g.data[:, 0, 0] = values
    return BlockGf(name_list=[name], block_list=[g])


@pytest.mark.parametrize("known", [True, False])
def test_fit_reproduces_pole_sum(known):
    # fit_dlr_imfreq_gf of a noise-free pole sum + constant C0 vs the exact function on the whole dense mesh, for
    # both blocks of a BlockGf; C0 given as the known zeroth moment, or fitted from the tail. Tolerance: the tau-grid
    # Fourier transform (n_tau = 10 x mesh points), not the DLR eps
    mesh = _dense_mesh()
    iw = _iw(mesh)
    exact = {"up": _pole_sum(iw) + C0, "dn": 2 * _pole_sum(iw) + C0}
    gs = []
    for bl in ("up", "dn"):
        g = Gf(mesh=mesh, target_shape=(1, 1))
        g.data[:, 0, 0] = exact[bl]
        gs.append(g)
    blocks = BlockGf(name_list=["up", "dn"], block_list=gs)
    moments = {bl: np.array([[[C0]]], dtype=complex) for bl in ("up", "dn")} if known else None
    fit = fit_dlr_imfreq_gf(blocks, WMAX, EPS, known_moments=moments)
    assert fit.mesh == mesh
    for bl in ("up", "dn"):
        assert np.linalg.norm(fit[bl].data[:, 0, 0] - exact[bl]) / np.linalg.norm(exact[bl]) < 1e-6


def test_fit_reduces_noise():
    # DLR-node values of fit_dlr_imfreq_gf vs fill_dlr_imfreq_gf (sampling at the nodes) of a pole sum with Hermitian
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

    fit_nodes = fill_dlr_imfreq_gf(fit_dlr_imfreq_gf(g, WMAX, EPS), WMAX, EPS)["up"]
    smp_nodes = fill_dlr_imfreq_gf(g, WMAX, EPS)["up"]
    exact = _pole_sum(_iw(smp_nodes.mesh))
    err_fit = np.linalg.norm(fit_nodes.data[:, 0, 0] - exact)
    err_smp = np.linalg.norm(smp_nodes.data[:, 0, 0] - exact)
    assert err_fit < err_smp
