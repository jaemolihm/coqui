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
pytest.importorskip("triqs_modest", reason="coqui.dmft.ctseg needs triqs_modest")

from triqs.gfs import Block2Gf, BlockGf, Gf, MeshDLRImFreq
from triqs.operators import n

import coqui.dmft.ctseg.ctseg_solver as ctseg_solver
from coqui.dmft.ctseg.ctseg_solver import solve_dynamic_dlr_mesh

BETA, WMAX, EPS = 10.0, 5.0, 1e-6
U, MU, V2, G2, W0 = 2.0, 1.0, 0.3, 0.5, 1.5
BLOCKS = ["up_0", "dn_0"]


def _problem():
    """Hubbard atom (two 1x1 blocks) with a two-pole hybridization and a screening D0(i nu) on DLR meshes."""
    mf = MeshDLRImFreq(beta=BETA, statistic="Fermion", w_max=WMAX, eps=EPS, symmetrize=True)
    mb = MeshDLRImFreq(beta=BETA, statistic="Boson", w_max=WMAX, eps=EPS, symmetrize=True)
    Delta = BlockGf(mesh=mf, gf_struct=[(bl, 1) for bl in BLOCKS])
    for _, g in Delta:
        for w in mf:
            g[w] = V2 * 0.5 * (1 / (complex(w) - 1.0) + 1 / (complex(w) + 1.0)) * np.eye(1)
    D0 = Block2Gf(name_list1=BLOCKS, name_list2=BLOCKS,
                  block_list=[[Gf(mesh=mb, target_shape=[1, 1]) for _ in BLOCKS] for _ in BLOCKS])
    for a in BLOCKS:
        for b in BLOCKS:
            for nu in mb:
                D0[a, b][nu] = -G2 * 2 * W0 / (complex(nu).imag ** 2 + W0 ** 2) * np.eye(1)
    h_int = U * n("up_0", 0) * n("dn_0", 0)
    h_loc0 = -MU * (n("up_0", 0) + n("dn_0", 0))
    return Delta, h_loc0, D0, h_int


class _SolveSpy:
    """Solver wrapper recording the keyword arguments of solve."""
    calls = []

    def __init__(self, *args, **kwargs):
        object.__setattr__(self, "_s", ctseg_solver_cls(*args, **kwargs))

    def __getattr__(self, key):
        return getattr(self._s, key)

    def __setattr__(self, key, value):
        setattr(self._s, key, value)

    def solve(self, **kwargs):
        _SolveSpy.calls.append(kwargs)
        self._s.solve(**kwargs)


ctseg_solver_cls = ctseg_solver.Solver


def test_solve_dynamic_dlr_mesh_defaults(monkeypatch):
    # solve_dynamic_dlr_mesh with its defaults (sIE) and a relative tail-fit window (fit_window_fraction=0.3) on a
    # 1x1-block Hubbard atom with D0:
    # the solver is asked for I(tau); G, Sigma - Sigma_inf, Pi, W, Chi live on the input DLR meshes; Sigma_inf equals
    # the analytic static self-energy sum_c [U_ac + D0_ac(i nu = 0)] n_c with the measured densities.
    # Tolerance 1e-5 on Sigma_inf: both sides use the same measured densities, so the only difference is the
    # quadrature of the D0(tau) -> D0(i nu = 0) transform on the n_tau grid (measured ~7e-7 at n_tau = 5001)
    _SolveSpy.calls.clear()
    monkeypatch.setattr(ctseg_solver, "Solver", _SolveSpy)
    Delta, h_loc0, D0, h_int = _problem()
    res = solve_dynamic_dlr_mesh(Delta, h_loc0, D0, h_int, n_cycles=2000, n_warmup_cycles=200, length_cycle=50,
                                 n_tau=5001, perform_tail_fit=True, fit_window_fraction=0.3)

    assert _SolveSpy.calls[-1]["measure_I_tau"] is True

    for key in ("G_iw", "Sigma_iw"):
        assert res[key].mesh == Delta.mesh
        assert all(np.all(np.isfinite(g.data)) for _, g in res[key])
    for key in ("Pi_iw", "W_iw", "Chi_iw"):
        assert res[key].mesh == D0["up_0", "up_0"].mesh
        assert np.all(np.isfinite(res[key].data))

    iw0 = next(k for k, nu in enumerate(D0["up_0", "up_0"].mesh) if nu.index == 0)
    D0_static = D0["up_0", "up_0"].data[iw0, 0, 0].real
    dens = res["orbital_occupations"]
    n_up, n_dn = dens["up_0"][0], dens["dn_0"][0]
    sigma_inf = {"up_0": (U + D0_static) * n_dn + D0_static * n_up,
                 "dn_0": (U + D0_static) * n_up + D0_static * n_dn}
    for bl, sig in zip(BLOCKS, res["Sigma_infty"]):
        assert abs(sig[0, 0] - sigma_inf[bl]) < 1e-5
