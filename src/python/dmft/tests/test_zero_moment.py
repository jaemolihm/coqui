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

pytest.importorskip("h5", reason="coqui.dmft.weiss imports h5")

from coqui.dmft.weiss import estimate_zero_moment


def _random_complex(rng, *shape):
    return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)


def test_estimate_zero_moment():
    """A(±iω) = t ± B/(iω) + C/(iω)² gives t exactly: the even part cancels B and the
    two-point ω² extrapolation removes C. Non-Hermitian t is kept as is."""
    rng = np.random.default_rng(7)
    t, B, C = (_random_complex(rng, 2, 3, 3) for _ in range(3))
    w = np.array([100.0, 150.0, 200.0])
    iw = 1j * w[:, None, None, None]

    def A(iw, B):
        return t + B / iw + C / iw**2

    t_est = estimate_zero_moment(A(iw, B), A(-iw, B), w)
    assert t_est.shape == t.shape
    assert np.linalg.norm(t_est - t) / np.linalg.norm(t) < 1e-13

    B_other = 10.0 * _random_complex(rng, 2, 3, 3)
    t_est_other = estimate_zero_moment(A(iw, B_other), A(-iw, B_other), w)
    assert np.linalg.norm(t_est_other - t) / np.linalg.norm(t) < 1e-13

    t_est_herm = 0.5 * (t_est + np.swapaxes(t_est, -1, -2).conj())
    assert np.linalg.norm(t_est - t_est_herm) / np.linalg.norm(t) > 0.1
