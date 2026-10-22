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
import shutil
import types

import h5py
import numpy as np
import pytest

pytest.importorskip("triqs", reason="triqs is required for dmft tests")
pytest.importorskip("triqs_modest.embedding", reason="triqs_modest is required")

from coqui.dmft import lr_scf_driver as drv


def rand(rng, *shape):
    return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)


def anderson_window(history, mixing, xs, fxs):
    """Reference of the Anderson step on the last history + 1 pairs written out as vectors,
    x_new = x + beta r - (dX + beta dR) gamma, gamma = argmin |dR gamma - r|."""
    xs, fxs = xs[-(history + 1):], fxs[-(history + 1):]
    rs = [f - x for x, f in zip(xs, fxs)]
    x_new = xs[-1] + mixing * rs[-1]
    if len(rs) > 1:
        dR = np.stack([b - a for a, b in zip(rs[:-1], rs[1:])], axis=1)
        dX = np.stack([b - a for a, b in zip(xs[:-1], xs[1:])], axis=1)
        gamma = np.linalg.lstsq(dR, rs[-1], rcond=None)[0]
        x_new = x_new - (dX + mixing * dR) @ gamma
    return x_new


def apply_coefficients(a, b, xs, fxs):
    return sum(am * x + bm * f for am, bm, x, f in zip(a, b, xs[-len(a):], fxs[-len(a):]))


def test_anderson_coefficients_reproduce_the_step_vector():
    # The coefficients (a, b) over the window pairs against the Anderson step written out as vectors, along an
    # iteration of a random linear map (spectral radius 1.5) for several histories and mixings: they are the same
    # linear combination, so the match is at round-off.
    rng = np.random.default_rng(0)
    n = 8
    A = rng.standard_normal((n, n))
    A *= 1.5 / np.max(np.abs(np.linalg.eigvals(A)))
    b0 = rng.standard_normal(n)
    for history in (0, 1, 3):
        for mixing in (1.0, 0.4):
            acc = drv._Anderson(history, mixing)
            x, xs, fxs = np.zeros(n), [], []
            for _ in range(8):
                fx = A @ x + b0
                xs.append(x), fxs.append(fx)
                a, b = acc.coefficients(x, fx)
                assert len(a) == len(b) == min(len(xs), history + 1)
                x_new = apply_coefficients(a, b, xs, fxs)
                assert np.linalg.norm(x_new - anderson_window(history, mixing, xs, fxs)) < 1e-12 * max(1, np.linalg.norm(x_new))
                x = x_new


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
    for history, steps, converges in ((n, n + 2, True), (0, 50, False)):
        acc = drv._Anderson(history=history, mixing=1.0)
        x, xs, fxs = np.zeros(n), [], []
        for _ in range(steps):
            fx = A @ x + b
            xs.append(x), fxs.append(fx)
            x = apply_coefficients(*acc.coefficients(x, fx), xs, fxs)
        err = np.linalg.norm(x - exact) / np.linalg.norm(exact)
        assert err < 1e-10 if converges else err > 1e3


def test_anderson_history_zero_is_linear_mixing():
    # history 0 gives the coefficients (1 - beta, beta), the linear mixing x_new = beta F(x) + (1 - beta) x.
    rng = np.random.default_rng(1)
    for beta in (1.0, 0.3):
        x, fx = rng.standard_normal(10), rng.standard_normal(10)
        a, b = drv._Anderson(history=0, mixing=beta).coefficients(x, fx)
        assert np.array_equal(a, [1 - beta]) and np.array_equal(b, [beta])


def random_w(rng, n_imp=2, nw=6, nb=3, n=2):
    return [{"dg_weiss_w": rand(rng, nw, 2, n, n), "du_weiss_w": rand(rng, nb, n, n, n, n),
             "dh0": rand(rng, 2, n, n), "ddelta_w": rand(rng, nw, 2, n, n)} for _ in range(n_imp)]


def test_mix_keeps_components_linearly_related():
    # w components are linear images of one another: ddelta = -dh0 - d(g^-1) with dg = -g_weiss d(g^-1) g_weiss. A mixture
    # with the same coefficients on all four components keeps both relations (to round-off) for random a, b.
    rng = np.random.default_rng(3)
    nw, n = 6, 2
    g = rand(rng, nw, 2, n, n) + 3 * np.eye(n)

    def make():
        dginv, dh0 = rand(rng, nw, 2, n, n), rand(rng, 2, n, n)
        return {"dg_weiss_w": -g @ dginv @ g, "du_weiss_w": rand(rng, 3, n, n, n, n), "dh0": dh0,
                "ddelta_w": -dh0[None] - dginv}
    history = [([make() for _ in range(2)], [make() for _ in range(2)]) for _ in range(3)]
    a, b = rng.standard_normal(3), rng.standard_normal(3)
    for w in drv._mix(history, a, b):
        dginv = -np.linalg.inv(g) @ w["dg_weiss_w"] @ np.linalg.inv(g)
        assert np.linalg.norm(w["ddelta_w"] + w["dh0"][None] + dginv) < 1e-12 * np.linalg.norm(w["ddelta_w"])
    # a (1 - beta, beta) mixture of the last pair is the linear mixing of every component
    w_in, w_out = history[-1]
    for w, wi, wo in zip(drv._mix(history[-1:], [0.7], [0.3]), w_in, w_out):
        for k in w:
            assert np.allclose(w[k], 0.7 * wi[k] + 0.3 * wo[k], rtol=0, atol=1e-15)


def test_outer_state_resume_and_meta(tmp_path):
    # The stage storage on one rank: groups are complete only with `done`; a group that lost `done` (a kill before
    # the last write) is incomplete and is replaced by put; a restart with the same meta keeps the file, a different
    # meta raises and leaves the file alone, restart=False truncates.
    import h5py
    path = str(tmp_path / "out.mbpt.h5")
    meta = {"q_vec": np.array([0.5, 0.0, 0.0]), "dH0_sha1": "abc", "embed_iter": 3, "dmft_iter": 2, "fix_density": False}
    st = drv._OuterState(path)
    st.start(meta, True)
    rng = np.random.default_rng(4)
    w = random_w(rng, 1)[0]
    loc = {"DeltaGloc_tsIab": rand(rng, 4, 1, 1, 2, 2)}
    st.put("iter0/impurity_0/w_in", w)
    st.put("iter0/local", loc, {"DeltaN": 0.25 + 0.5j, "run_lr_iterations": 7})
    st.put("iter0/impurity_0/callback", {"n_imp": 0.5 + 0.25j, "w_A_rel": 0.1})
    extras, _ = st.get("iter0/impurity_0/callback")
    assert extras["n_imp"] == 0.5 + 0.25j and extras["w_A_rel"] == 0.1
    assert st.has("iter0/impurity_0/w_in") and st.has("iter0/local") and not st.has("iter0")
    back, _ = st.get("iter0/impurity_0/w_in")
    assert all(np.array_equal(back[k], w[k]) for k in w)
    data, attrs = st.get("iter0/local")
    assert np.array_equal(data["DeltaGloc_tsIab"], loc["DeltaGloc_tsIab"]) and attrs["DeltaN"] == 0.25 + 0.5j
    with h5py.File(path, "a") as f:
        del f["lr_gw_edmft/iter0/local"].attrs["done"]
    assert not st.has("iter0/local") and st.has("iter0/impurity_0/w_in")
    st.put("iter0/local", loc, {"DeltaN": 1.0, "run_lr_iterations": 3})
    st.close_group("iter0", {"residual": 0.1})
    assert st.has("iter0") and st.get("iter0")[1]["residual"] == 0.1

    # resume: same meta keeps the stages
    drv._OuterState(path).start(dict(meta), True)
    assert st.has("iter0")
    # mismatching meta raises and keeps them
    for key, val in (("q_vec", np.array([0.25, 0.0, 0.0])), ("dH0_sha1", "xyz"), ("dmft_iter", 1), ("fix_density", True)):
        with pytest.raises(ValueError, match=key):
            drv._OuterState(path).start(dict(meta, **{key: val}), True)
    assert st.has("iter0")
    # a stale final solve is invalidated
    with h5py.File(path, "a") as f:
        f.create_group("linear_response")
    st.set_final(True, [0.1], 5)
    assert st.final_valid(1, False) and not st.final_valid(2, False) and not st.final_valid(1, True)
    with h5py.File(path, "a") as f:
        f["linear_response"].create_dataset("DeltaG_tskij", data=np.zeros(1))
    assert st.final_valid(1, True)
    st.invalidate_final()
    assert not st.final_valid(1, False)
    with h5py.File(path, "r") as f:
        assert "linear_response" not in f
    # restart=False truncates
    drv._OuterState(path).start(meta, False)
    with h5py.File(path, "r") as f:
        assert list(f["lr_gw_edmft"].keys()) == ["meta"]


class MeshFake:
    def __init__(self, wn):
        self.wn = np.asarray(wn)

    def wn_mesh(self, stats):
        return self.wn


def test_antihermitian_norms():
    # A Hermitian w and y (static: a = a^dagger; fermionic: a(-iw) = a(iw)^dagger on the mesh pairs wn <-> -wn of
    # wn_mesh('f') (CoQuí notation, odd n); bosonic: a = a^dagger in the matrix indices) give 0; an anti-Hermitian
    # static part gives its relative weight; an asymmetric mesh skips the diagnostic.
    rng = np.random.default_rng(5)
    iaft = MeshFake([-3, -1, 1, 3, 5])
    sym = MeshFake([-5, -3, -1, 1, 3, 5])
    nw, n = 6, 2

    def herm_static(*lead):
        a = rand(rng, *lead, n, n)
        return a + np.conj(np.swapaxes(a, -1, -2))

    def herm_f():
        a = rand(rng, nw, 2, n, n)
        return a + np.conj(np.swapaxes(a[::-1], -1, -2))

    du = rand(rng, 3, n * n, n * n)
    du = (du + np.conj(np.swapaxes(du, -1, -2))).reshape(3, n, n, n, n)
    w = [{"dg_weiss_w": herm_f(), "ddelta_w": herm_f(), "dh0": herm_static(2), "du_weiss_w": du}]
    pi = rand(rng, 3, n, n)
    y = [{"Vhf_imp": herm_static(2), "Sigma_imp_w": herm_f(), "Pi_imp_w": pi + np.conj(np.swapaxes(pi, -1, -2))}]
    nw_, ny_ = drv._antihermitian_norms(w, y, sym)
    assert nw_ < 1e-15 and ny_ < 1e-15
    # y reduced to a static anti-Hermitian part
    y2 = [{"Vhf_imp": 1j * np.eye(n)[None] * np.ones((2, 1, 1)), "Sigma_imp_w": np.zeros((nw, 2, n, n), complex),
           "Pi_imp_w": np.zeros((3, n, n), complex)}]
    assert abs(drv._antihermitian_norms(w, y2, sym)[1] - 1.0) < 1e-15
    assert drv._antihermitian_norms(w, y, iaft) is None


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


def test_lr_params_validation():
    # Defaults, an unknown key and the keys the driver fixes (gw_mode, recompute_W) or no longer accepts (tol_imp)
    # against the documented rules.
    kernel, driver = drv._split_lr_params(None)
    assert kernel["iter_alg"] == {"alg": "GCR"} and kernel["include_hartree"] and kernel["include_exchange"]
    assert driver["tol_outer"] == 1e-3 and driver["max_outer"] == 10 and driver["anderson_history"] == 0
    assert driver["mixing"] == 1.0 and driver["restart"] is True
    assert driver["iteration"] is None and driver["fix_density"] is None
    kernel, driver = drv._split_lr_params({"tol": 1e-12, "max_outer": 3})
    assert kernel["tol"] == 1e-12 and driver["max_outer"] == 3
    for bad in ({"gw_mode": "none"}, {"recompute_W": False}, {"tol_imp": 1e-6}, {"bogus": 1}, {"max_outer": 0},
                {"mixing": 0.0}, {"mixing": -0.5}, {"anderson_history": -1}):
        with pytest.raises(ValueError):
            drv._split_lr_params(bad)


def test_check_response_keeps_numeric_extras(tmp_path):
    gs = {"Vhf_imp": np.zeros((2, 2, 2)), "Sigma_imp_w": np.zeros((4, 2, 2, 2)), "Pi_imp_w": np.zeros((3, 2, 2))}
    ret = dict(gs, fd_error=np.ones(3), flag=np.array([True, False]), err=np.complex128(1 + 2j),
               note="fd step 1e-3", nothing=None, info={"a": 1}, ragged=[[1], [1, 2]])
    y, extras = drv._check_response(ret, {"ground_state": gs})
    assert set(y) == set(gs) and set(extras) == {"fd_error", "flag", "err"}
    # the kept extras round-trip through the stage file
    st = drv._OuterState(str(tmp_path / "s.h5"))
    st.start({}, False)
    st.put("x/callback", extras)
    stored = st.get("x/callback")[0]
    assert set(stored) == set(extras)
    for k, v in extras.items():
        np.testing.assert_array_equal(stored[k], v)


# ---------------------------------------------------------------------------------------------------------
# Restart replay of the outer loop against an affine fake: x -> A x + b on the local responses
# ---------------------------------------------------------------------------------------------------------

class Killed(Exception):
    pass


class AffineFake:
    """
    Affine stand-ins for the pieces of the outer loop that are not the loop itself, for two impurities:
        y_i = C_i w_i + d_i                          (impurity_response)
        tot = tot0 + L y                             (run_lr increments; DeltaN is the 13th entry)
        w_out,i = R_i tot + S_i y_i                  (Weiss LR)
    The map w_in -> w_out,i is affine with the spectral radius ``rho``. ``tick`` counts every call of the fake;
    the call with the index ``kill_at`` raises Killed.
    """
    W_SHAPES = {"dg_weiss_w": (2, 2, 1, 1), "du_weiss_w": (2, 1, 1, 1, 1), "dh0": (2, 1, 1), "ddelta_w": (2, 2, 1, 1)}
    Y_SHAPES = {"Vhf_imp": (2, 1, 1), "Sigma_imp_w": (2, 2, 1, 1), "Pi_imp_w": (2, 1, 1)}
    LOC_KEYS = ("DeltaGloc_tsIab", "DeltaHloc_sIab", "DeltaWloc_tabcd", "DeltaSigma_dc_tsIab")

    def __init__(self, output, rho=0.3, nimp=2, seed=11):
        rng = np.random.default_rng(seed)
        self.output, self.nimp, self.kill_at, self.calls = output, nimp, None, []
        nw = sum(int(np.prod(v)) for v in self.W_SHAPES.values())
        ny = sum(int(np.prod(v)) for v in self.Y_SHAPES.values())
        nl = 3 * len(self.LOC_KEYS) + 1
        self.C = [rand(rng, ny, nw) for _ in range(nimp)]
        self.d = [rand(rng, ny) for _ in range(nimp)]
        self.L = rand(rng, nl, nimp * ny)
        self.R = [rand(rng, nw, nl) for _ in range(nimp)]
        self.S = [rand(rng, nw, ny) for _ in range(nimp)]
        self.tot0 = rand(rng, nl)
        self.gs = {k: rand(rng, 2, 2, 1, 1) for k in ("Gloc_tsIab", "Hloc_sIab", "Wloc_tabcd", "Vloc_abcd")}
        Cb = np.zeros((nimp * ny, nimp * nw), complex)
        Sb = np.zeros((nimp * nw, nimp * ny), complex)
        for i in range(nimp):
            Cb[i * ny:(i + 1) * ny, i * nw:(i + 1) * nw] = self.C[i]
            Sb[i * nw:(i + 1) * nw, i * ny:(i + 1) * ny] = self.S[i]
        M = (np.vstack(self.R) @ self.L + Sb) @ Cb
        scale = rho / np.max(np.abs(np.linalg.eigvals(M)))
        self.R, self.S = [r * scale for r in self.R], [s * scale for s in self.S]
        zero = {k: np.zeros(v, complex) for k, v in self.Y_SHAPES.items()}
        self.imps = [{"ground_state": dict(zero)} for _ in range(nimp)]

    def tick(self, name):
        if self.kill_at is not None and len(self.calls) == self.kill_at:
            raise Killed(name)
        self.calls.append(name)

    @staticmethod
    def pack(d, shapes):
        return np.concatenate([np.asarray(d[k]).ravel() for k in shapes])

    @staticmethod
    def unpack(v, shapes):
        out, p = {}, 0
        for k, shp in shapes.items():
            m = int(np.prod(shp))
            out[k] = v[p:p + m].reshape(shp).copy()
            p += m
        return out

    def loc(self, v):
        d = {k: v[3 * j:3 * j + 3].copy() for j, k in enumerate(self.LOC_KEYS)}
        return dict(d, DeltaN=complex(v[-1]))

    def pack_loc(self, tot):
        return np.concatenate([tot[k] for k in self.LOC_KEYS] + [[tot["DeltaN"]]])

    # replacements of the module functions
    def impurity_response(self, i, dg, du, dh0, ddelta, *, ground_state, outer_iteration):
        self.tick("callback")
        w = self.pack(dict(dg_weiss_w=dg, du_weiss_w=du, dh0=dh0, ddelta_w=ddelta), self.W_SHAPES)
        y = self.unpack(self.C[i] @ w + self.d[i], self.Y_SHAPES)
        return dict(y, n_imp=np.array([outer_iteration, i], dtype=float))

    def embed_response(self, embedding, embedding_2e, y, y_prev, nspin):
        return [{k: yi[k] - (y_prev[i][k] if y_prev is not None else 0.0) for k in yi} for i, yi in enumerate(y)]

    def weiss_responses(self, imps, tot, y, embedding, embedding_2e, iaft):
        self.tick("weiss")
        t = self.pack_loc(tot)
        return [self.unpack(self.R[i] @ t + self.S[i] @ self.pack(y[i], self.Y_SHAPES), self.W_SHAPES)
                for i in range(self.nimp)]

    def solve(self, dH0, y_embedded, out_prefix, save_DeltaG, ground_state=False):
        self.tick("solve_final" if out_prefix == self.output else "solve_ground" if ground_state else "solve")
        if y_embedded is None:
            return 5, dict(self.loc(self.tot0), **self.gs)
        ys = np.concatenate([self.pack(dy, self.Y_SHAPES) for dy in y_embedded])
        inc = self.loc(self.L @ ys)
        if out_prefix != self.output:
            return 7, inc
        full = self.loc(self.tot0 + self.L @ ys)
        with h5py.File(out_prefix + ".mbpt.h5", "a") as f:
            g = f.create_group("linear_response")
            g.create_dataset("DeltaGloc_tsIab", data=full["DeltaGloc_tsIab"])
            g.create_dataset("DeltaWloc_tabcd", data=full["DeltaWloc_tabcd"])
            g.create_dataset("DeltaG_tskij", data=full["DeltaHloc_sIab"])
        return 11, {"DeltaGloc_tsIab": full["DeltaGloc_tsIab"], "DeltaWloc_tabcd": full["DeltaWloc_tabcd"]}


def run_outer(monkeypatch, tmp_path, name, fake=None, kill_at=None, restart=True, **lr_params):
    """_outer_loop on the affine fake in tmp_path/name; returns (result, fake)."""
    out = str(tmp_path / name)
    fake = fake or AffineFake(out)
    fake.output, fake.kill_at, fake.calls = out, kill_at, []
    monkeypatch.setattr(drv, "_ground_state_impurities", lambda *a: fake.imps)
    monkeypatch.setattr(drv, "_log_distance_from_fixed_point", lambda *a: None)
    monkeypatch.setattr(drv, "_embed_response", fake.embed_response)
    monkeypatch.setattr(drv, "_weiss_responses", fake.weiss_responses)
    monkeypatch.setattr(drv, "lr_dc_response", lambda *a, **k: (None, np.zeros(2, complex)))
    kernel, driver = drv._split_lr_params(dict({"tol_outer": 1e-4, "max_outer": 10, "restart": restart}, **lr_params))
    st = drv._OuterState(out + ".mbpt.h5")
    st.start({"q_vec": np.zeros(3), "dH0_sha1": "x"}, restart)
    res = drv._outer_loop(
        st, fake.solve, fake.impurity_response, state=types.SimpleNamespace(solver_results=[0] * fake.nimp),
        embedding=None, embedding_2e=None, iaft=MeshFake([1, 3]), nspin=1, proj_info={"dc_cells": {}},
        solver_chkpt="", DeltaH0_skij=np.ones((1, 1, 1, 1)), kernel_kw=kernel, drv=driver, q_vec=np.zeros(3),
        fix_density=False, embed_iter=1, dmft_iter=0, output=out,
        work_prefix=out + "_work")
    return res, fake


def dump_file(path):
    """All datasets and attributes of an h5 file as {path: value}."""
    items = {}

    def visit(name, obj):
        for k, v in obj.attrs.items():
            items[f"{name}@{k}"] = np.asarray(v)
        if isinstance(obj, h5py.Dataset):
            items[name] = obj[()]
    with h5py.File(path, "r") as f:
        visit("", f)
        f.visititems(visit)
    return items


def assert_same_file(a, b):
    da, db = dump_file(a), dump_file(b)
    assert da.keys() == db.keys()
    for k in da:
        assert np.array_equal(da[k], db[k]), k


def assert_same_result(r, ref):
    assert r["residuals"] == ref["residuals"] and r["converged"] == ref["converged"]
    assert r["n_outer"] == ref["n_outer"] and r["niter"] == ref["niter"] and r["DeltaN"] == ref["DeltaN"]
    for k in ("dGloc_tsIab", "dWloc_tabcd"):
        assert np.array_equal(r[k], ref[k])
    for yi, yr in zip(r["impurity_response"], ref["impurity_response"]):
        assert all(np.array_equal(yi[k], yr[k]) for k in yi)


@pytest.mark.parametrize("history, mixing", [(0, 1.0), (2, 0.8)])
def test_outer_loop_restart_replay(monkeypatch, tmp_path, history, mixing):
    # The outer loop on an affine fake (two impurities, spectral radius 0.3 of w_in -> w_out). A run killed at any
    # call of the fake (the ground-state solve, each impurity callback, increment solve, Weiss map and the final
    # solve: every stage boundary a, b, c, d and the points inside them) and resumed with restart=True reproduces
    # the uninterrupted run bit for bit: the output file (w_in, w_out, y, local responses, residuals, final) and the
    # result. Then a looser tol_outer (final reused when n is unchanged, recomputed when n changes) and a larger
    # max_outer.
    kw = dict(anderson_history=history, mixing=mixing)
    ref, fake = run_outer(monkeypatch, tmp_path, f"ref{history}", **kw)
    ref_file = str(tmp_path / f"ref{history}_copy.mbpt.h5")
    shutil.copy(str(tmp_path / f"ref{history}.mbpt.h5"), ref_file)
    ref_calls = list(fake.calls)
    n_calls = len(ref_calls)
    n = ref["n_outer"]
    assert ref["converged"] and n >= 3 and fake.calls.count("callback") == 2 * n

    for m in range(n_calls):
        name = f"kill{history}_{m}"
        with pytest.raises(Killed):
            run_outer(monkeypatch, tmp_path, name, kill_at=m, **kw)
        res, fk = run_outer(monkeypatch, tmp_path, name, fake=fake, **kw)
        assert_same_result(res, ref)
        assert_same_file(str(tmp_path / f"{name}.mbpt.h5"), ref_file)
        # m == 1: the initial local responses live only in init/local, so a kill before init closes redoes the
        # ground-state solve
        assert fk.calls == (ref_calls if m == 1 else ref_calls[m:]) and res["resumed_at"] <= n

    # a complete file resumed with a tol_outer between the last two residuals: same n, nothing is recomputed
    r = ref["residuals"]
    res, fk = run_outer(monkeypatch, tmp_path, f"ref{history}", fake=fake, tol_outer=np.sqrt(r[-1] * r[-2]), **kw)
    assert fk.calls == [] and res["resumed_at"] == n
    assert_same_result(res, ref)

    # a looser tol_outer stops earlier: the loop is not run, the final solve is redone for the earlier y; it
    # equals a fresh run with that tol_outer
    tol = np.sqrt(r[1] * r[2])
    res, fk = run_outer(monkeypatch, tmp_path, f"ref{history}", fake=fake, tol_outer=tol, **kw)
    assert fk.calls == ["solve_final"] and res["n_outer"] == 3 and res["resumed_at"] == 3
    fresh, _ = run_outer(monkeypatch, tmp_path, f"fresh{history}", tol_outer=tol, **kw)
    assert_same_result(res, fresh)

    # a larger max_outer continues a run that stopped at max_outer
    run_outer(monkeypatch, tmp_path, f"cap{history}", max_outer=2, **kw)
    res, fk = run_outer(monkeypatch, tmp_path, f"cap{history}", fake=fake, **kw)
    assert res["resumed_at"] == 2 and fk.calls.count("callback") == 2 * (n - 2)
    assert_same_result(res, ref)
    assert_same_file(str(tmp_path / f"cap{history}.mbpt.h5"), ref_file)
