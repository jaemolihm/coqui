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

pytest.importorskip("coqui._lib.iaft_module")

from coqui.utils.imag_axes_ft import IAFT
from coqui.impurity_solver.bath_fit import extract_t_and_delta


def _rand_sym(rng, n):
    A = rng.standard_normal((n, n))
    return 0.5 * (A + A.T)


def test_extract_t_and_delta_per_spin_and_impurity():
    # extract_t_and_delta vs the exact (t, Delta) of g^{-1}(iw) = iw - t_sI - V_sI V_sI^dagger / (iw - eps_sI),
    # with t, V, eps different for every (spin, impurity). t is real: estimate_zero_moment keeps Re only.
    # Tolerance: the two-point omega^2 extrapolation of Re[t + Delta] leaves an O(1/w^4) error in t
    # (2.4e-6 relative here, 3.1e-6 in Delta).
    rng = np.random.default_rng(11)
    ns, nImp, nb = 2, 2, 2
    iaft = IAFT(beta=10.0, wmax=10.0, prec="medium", basis="dlr", verbose=False)
    wn = iaft.wn_mesh('f') * np.pi / iaft.beta
    eye = np.eye(nb)

    t = np.array([[_rand_sym(rng, nb) for _ in range(nImp)] for _ in range(ns)])
    V = rng.standard_normal((ns, nImp, nb, 1))
    eps = rng.standard_normal((ns, nImp))
    delta = np.array([[[V[s, I] @ V[s, I].T / (1j * w - eps[s, I]) for I in range(nImp)]
                       for s in range(ns)] for w in wn])
    g_weiss = np.array([[[np.linalg.inv(1j * w * eye - t[s, I] - delta[n, s, I]) for I in range(nImp)]
                         for s in range(ns)] for n, w in enumerate(wn)])

    t_est, delta_est = extract_t_and_delta(g_weiss, iaft)

    assert np.linalg.norm(t_est - t) / np.linalg.norm(t) < 1e-5
    assert np.linalg.norm(delta_est - delta) / np.linalg.norm(delta) < 1e-5
