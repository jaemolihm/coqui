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
pytest.importorskip("triqs_modest.embedding", reason="triqs_modest is required")

from coqui.dmft import lr_scf_driver as drv


def rand(rng, *shape):
    return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)


def test_anderson_affine_map_converges_in_dim_steps():
    # Anderson acceleration of x = A x + b (A random, spectral radius 1.5: plain iteration diverges) against the
    # exact solution (1 - A)^-1 b. An affine map is solved exactly once the history spans its dimension, so n + 1
    # iterations with history n reach round-off (1e-10 allows for the conditioning of the least-squares problem).
    rng = np.random.default_rng(0)
    n = 6
    A = rng.standard_normal((n, n))
    A *= 1.5 / np.max(np.abs(np.linalg.eigvals(A)))
    b = rng.standard_normal(n)
    exact = np.linalg.solve(np.eye(n) - A, b)
    and_ = drv._Anderson(history=n, mixing=1.0)
    x = np.zeros(n)
    for _ in range(n + 2):
        x = and_.step(x, A @ x + b)
    assert np.linalg.norm(x - exact) < 1e-10 * np.linalg.norm(exact)

    plain = drv._Anderson(history=0, mixing=1.0)
    x = np.zeros(n)
    for _ in range(50):
        x = plain.step(x, A @ x + b)
    assert np.linalg.norm(x - exact) > 1e3 * np.linalg.norm(exact)


def test_anderson_history_zero_is_linear_mixing():
    # history 0 against x + beta (F(x) - x) written out, bit for bit (it goes through dmft_state._mix_into).
    rng = np.random.default_rng(1)
    x, fx = rng.standard_normal(10), rng.standard_normal(10)
    for beta in (1.0, 0.3):
        got = drv._Anderson(history=0, mixing=beta).step(x, fx)
        assert np.allclose(got, beta * fx + (1 - beta) * x, rtol=0, atol=1e-15)


def test_pack_unpack_roundtrip():
    # _unpack(_pack(y)) against y: the packing is a bijection onto real vectors, so the match is exact.
    rng = np.random.default_rng(2)
    y = [{"Vhf_imp": rand(rng, 2, 2, 2), "Sigma_imp_w": rand(rng, 5, 2, 2, 2), "Pi_imp_w": rand(rng, 3, 2, 2)}
         for _ in range(2)]
    back = drv._unpack(drv._pack(y), y)
    for a, b in zip(y, back):
        for k in a:
            assert np.array_equal(a[k], b[k])


def test_ground_state_projections():
    # The projections of G^loc (hermitize per tau, real part) and W^loc (real part, 8-fold symmetry) are idempotent,
    # and an array already in their range (Hermitian real G, 8-fold symmetric real W) is a fixed point (round-off).
    rng = np.random.default_rng(3)
    n = 3
    G = rand(rng, 4, 2, n, n)
    PG = drv._project_gs_G(G)
    assert np.array_equal(drv._project_gs_G(PG), PG)
    assert np.allclose(PG, np.conj(np.swapaxes(PG, -1, -2)), rtol=0, atol=1e-16)
    assert np.abs(PG.imag).max() == 0.0

    W = rand(rng, 4, n, n, n, n)
    PW = drv._project_gs_W(W)
    assert np.abs(drv._project_gs_W(PW) - PW).max() < 1e-15
    for perm in drv._PERMUT_8FOLD:
        assert np.abs(PW.transpose((0,) + tuple(1 + p for p in perm)) - PW).max() < 1e-15
    assert np.abs(PW.imag).max() == 0.0


def test_lr_params_validation():
    # Defaults, an unknown key and the keys the driver fixes (gw_mode, recompute_W) against the documented rules.
    kernel, driver = drv._split_lr_params(None)
    assert kernel["iter_alg"] == {"alg": "GCR"} and driver["anderson_history"] == 5
    kernel, driver = drv._split_lr_params({"tol": 1e-12, "max_outer": 3})
    assert kernel["tol"] == 1e-12 and driver["max_outer"] == 3
    for bad in ({"gw_mode": "none"}, {"recompute_W": False}, {"bogus": 1}, {"max_outer": 0}):
        with pytest.raises(ValueError):
            drv._split_lr_params(bad)


def test_ground_state_flag_checks():
    # Each unsupported ground-state setting against its rejection; the supported settings pass.
    good = {"wloc": {"screen_type": "gw_edmft"}, "dmft_embed": {"corr_only": True},
            "impurity": {"solver": [{"n_cycles": 1}]}}
    drv._check_ground_state_flags(good)
    cases = [
        {"wloc": {"screen_type": "rpa"}},
        {"dmft_embed": {"corr_only": False}},
        {"impurity": {"solver": [{"causal_projection": {"nbath_per_orbital_wloc": 3}}]}},
        {"impurity": {"solver": [{"set_sigma_infty_to_dc": True}]}},
    ]
    for override in cases:
        with pytest.raises(ValueError):
            drv._check_ground_state_flags(dict(good, **override))
