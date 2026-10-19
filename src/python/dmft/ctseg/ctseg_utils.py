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

from triqs.gfs import (MeshImFreq, MeshImTime, MeshDLRImFreq, Gf, BlockGf,
                      fit_gf_dlr, fit_hermitian_tail, make_gf_from_fourier)
from triqs.gfs.tools import make_zero_tail


def fill_dlr_imfreq_gf(g_iw, wmax, eps):

    assert isinstance(g_iw.mesh, MeshImFreq), (
        "fill_dlr_imfreq_gf: input Green's function should live on MeshImFreq."
    )

    mesh_dlr_iw = MeshDLRImFreq(
        beta=g_iw.mesh.beta, statistic=g_iw.mesh.statistic,
        w_max=wmax, eps=eps, symmetrize=True
    )

    mesh_dlr_idx = np.array([iw.index for iw in mesh_dlr_iw])
    max_dlr_idx = max(abs(mesh_dlr_idx[0]), abs(mesh_dlr_idx[-1]))
    assert max_dlr_idx <= g_iw.mesh.n_iw, (
        f"fill_dlr_imfreq_gf: g_iw.mesh.n_iw = {g_iw.mesh.n_iw} < maximum DLRImFreq index ({max_dlr_idx})."
    )

    if isinstance(g_iw, BlockGf):
        gf_struct = [(bl, gf.target_shape[0]) for (bl, gf) in g_iw]
        g_dlr_iw = BlockGf(mesh=mesh_dlr_iw, gf_struct=gf_struct)
        for w in g_dlr_iw.mesh:
            for block, gf in g_dlr_iw:
                gf[w] = g_iw[block](w)
        return g_dlr_iw

    elif isinstance(g_iw, Gf):
        g_dlr_iw = Gf(mesh=mesh_dlr_iw, target_shape=g_iw.target_shape)
        for w in g_dlr_iw.mesh:
            g_dlr_iw[w] = g_iw(w)
        return g_dlr_iw

    else:
        raise NotImplemented


def fit_dlr_imfreq_gf(g_iw, wmax, eps):
    """
    Same as fill_dlr_imfreq_gf (g_iw on MeshImFreq -> values at the DLR Matsubara nodes of (wmax, eps)), but by a
    least-squares DLR fit to the whole function instead of copying g_iw at the nodes, which passes the Monte Carlo
    noise of each node value straight into the DLR expansion.

    g_iw must vanish at infinite frequency (e.g. G, Sigma - Sigma_infty) and have a smooth tail (tail-fitted): it is
    Fourier transformed to imaginary time with its high-frequency moments (fitted, zeroth moment fixed to 0), fitted
    with fit_gf_dlr, and the fit is evaluated at the nodes.
    """
    assert isinstance(g_iw.mesh, MeshImFreq), (
        "fit_dlr_imfreq_gf: input Green's function should live on MeshImFreq."
    )

    mesh_dlr_iw = MeshDLRImFreq(
        beta=g_iw.mesh.beta, statistic=g_iw.mesh.statistic,
        w_max=wmax, eps=eps, symmetrize=True
    )
    tau_mesh = MeshImTime(beta=g_iw.mesh.beta, statistic=g_iw.mesh.statistic, n_tau=10 * len(g_iw.mesh) + 1)

    def _fit_gf(gf):
        tail, _ = fit_hermitian_tail(gf, make_zero_tail(gf, n_moments=1))
        gf_dlr = fit_gf_dlr(make_gf_from_fourier(gf, tau_mesh, tail), w_max=wmax, eps=eps)
        g_dlr_iw = Gf(mesh=mesh_dlr_iw, target_shape=gf.target_shape)
        for w in mesh_dlr_iw:
            g_dlr_iw[w] = gf_dlr(w)
        return g_dlr_iw

    if isinstance(g_iw, BlockGf):
        gf_struct = [(bl, gf.target_shape[0]) for (bl, gf) in g_iw]
        g_dlr_iw = BlockGf(mesh=mesh_dlr_iw, gf_struct=gf_struct)
        for block, gf in g_dlr_iw:
            gf << _fit_gf(g_iw[block])
        return g_dlr_iw

    elif isinstance(g_iw, Gf):
        return _fit_gf(g_iw)

    else:
        raise NotImplemented
