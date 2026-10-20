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

from array import array
import os

import numpy as np
import pytest

pytest.importorskip("triqs", reason="triqs is required for dmft tests")

import coqui
from coqui.utils.tests.test_coqui_env import mpi
from coqui.dmft.bath_fit import causal_projection_boson
from coqui.dmft import weiss, lr_weiss
from coqui.mean_field.tests.test_qe import construct_qe_mf
from coqui.utils.imag_axes_ft import IAFT


@pytest.fixture(scope="module", params=["ir", "dlr"])
def downfold_inputs(mpi, request):
    iaft_basis = request.param
    if iaft_basis == "dlr":
        pytest.importorskip("coqui._lib.iaft_module")
    prefix = f"svo_{iaft_basis}"
    chkpt_h5 = f"./{prefix}.mbpt.h5"
    wan_h5 = coqui.TEST_INPUT_DIR + "qe/svo_kp222_nbnd40/mlwf/svo.mlwf.h5"

    mf = construct_qe_mf(mpi, "qe_svo222_sym")
    eri_params = {
        "storage": "incore",
        "thresh": 1e-4,
        "ecut": mf.ecutwfc() * 1.5,
        "init": True,
    }
    thc = coqui.make_thc_coulomb(mf, eri_params)

    gw_params = {
        "restart": False,
        "output": prefix,
        "niter": 1,
        "beta": 100,
        "iaft": {"prec": "high", "basis": iaft_basis},
        "iter_alg": {"alg": "damping", "mixing": 0.7},
    }
    coqui.run_gw(gw_params, h_int=thc)
    mpi.barrier()

    ir_kernel = IAFT.from_coqui_chkpt(chkpt_h5, verbose=False)

    gloc_params = {
        "outdir": "./",
        "prefix": prefix,
        "greens_func_source": "scf",
        "greens_func_iteration": 1,
        "wannier_file": wan_h5,
        "force_real": True,
    }
    hloc, gloc_t = coqui.downfold_local_gf(mf, gloc_params)

    wloc_params = {
        "outdir": "./",
        "prefix": prefix,
        "screen_type": "gw_edmft_density",
        "greens_func_source": "scf",
        "greens_func_iteration": 1,
        "wannier_file": wan_h5,
        "output_in_tau": True,
    }
    _, wloc_t = coqui.downfold_coulomb(thc, wloc_params)

    yield {
        "basis": iaft_basis,
        "iaft": ir_kernel,
        "hloc": hloc,
        "gloc_t": gloc_t,
        "wloc_t": wloc_t,
    }

    if mpi.root() and os.path.isfile(chkpt_h5):
        os.remove(chkpt_h5)
    mpi.barrier()


# Measured (SVO, IR and DLR): max |Hloc - H0_SAB_REF| = 7.3e-7 (the error of the fit; tolerance twice that),
# leakage of Delta 4e-11 (IR) and 9e-13 (DLR); a constant 1e-3 in Hloc reads 1.3e-2 (IR) and 2.9e-4 (DLR).
TOL_HLOC_VS_FIT = 1.5e-6
TOL_LEAKAGE = 1e-9
TOL_LEAKAGE_CONTROL = 1e-4


def _delta_tail(iaft, hloc, g_iw):
    """Delta(iw) = iw - Hloc - G_loc^{-1}(iw) of a local Green's function with static term Hloc."""
    iw = 1j * iaft.wn_mesh('f') * np.pi / iaft.beta
    nbnd = hloc.shape[-1]
    return np.array([[iw[n] * np.eye(nbnd) - hloc[s] - np.linalg.inv(g_iw[n, s]) for s in range(g_iw.shape[1])]
                     for n in range(g_iw.shape[0])])


def _tail_ratio(iaft, delta_iw):
    """|Delta_even(top)| / |Delta_even(previous)| at the two highest positive frequencies, and the 1/w^2 value
    (w_prev / w_top)^2, where Delta_even(w) = (Delta(iw) + Delta(-iw)) / 2."""
    w = iaft.wn_mesh('f') * np.pi / iaft.beta
    nw = len(w)
    even = lambda n: 0.5 * (delta_iw[n] + delta_iw[nw - 1 - n])
    return np.linalg.norm(even(nw - 1)) / np.linalg.norm(even(nw - 2)), (w[-2] / w[-1]) ** 2


# Analytic Hloc vs reference values of a high-frequency fit of G_loc^{-1}, the Delta tail of the analytic
# split (leakage and the 1/w^2 decay of the even part), and the M0 = P S P^dagger = 1 precondition of the split.
def test_hloc_split_svo(downfold_inputs):
    assert downfold_inputs["iaft"].basis == downfold_inputs["basis"]

    iaft = downfold_inputs["iaft"]
    hloc = downfold_inputs["hloc"]
    gloc_t = downfold_inputs["gloc_t"]
    g_iw = iaft.tau_to_w(gloc_t, stats="f")

    # (a) Hloc vs the fit reference values; the tolerance is the fit's error, measured
    H0_SAB_REF = np.array([[
        [ 1.98632457e-01, -3.39822832e-05,  3.27477287e-05],
        [-3.39822832e-05,  1.98823108e-01, -9.76175950e-05],
        [ 3.27477287e-05, -9.76175950e-05,  1.98808918e-01]
    ]])
    assert hloc.shape == g_iw.shape[1:] == H0_SAB_REF.shape
    assert np.allclose(hloc.imag, 0.0, atol=1e-12)
    assert np.max(np.abs(hloc.real - H0_SAB_REF)) < TOL_HLOC_VS_FIT

    # M0 precondition: -(G(0+) + G(beta-)) = 1
    g_edges = iaft.tau_interpolate(gloc_t, np.array([0.0, iaft.beta]), stats="f")
    m0 = -(g_edges[0] + g_edges[1])
    assert np.linalg.norm(m0 - np.eye(m0.shape[-1])[None]) < 1e-8

    # (b) Delta tail
    delta_iw = _delta_tail(iaft, hloc, g_iw)
    assert np.all(np.isfinite(delta_iw))
    leakage = iaft.check_leakage(delta_iw, 'f', 'delta', w_input=True)
    assert leakage < TOL_LEAKAGE
    ratio, ratio_ref = _tail_ratio(iaft, delta_iw)
    assert abs(ratio / ratio_ref - 1) < 0.1

    # negative control: a constant 1e-3 in Hloc is read as leakage and breaks the 1/w^2 decay
    delta_bad = _delta_tail(iaft, hloc + 1e-3 * np.eye(hloc.shape[-1]), g_iw)
    assert iaft.check_leakage(delta_bad, 'f', 'delta', w_input=True) >= TOL_LEAKAGE_CONTROL
    ratio_bad, _ = _tail_ratio(iaft, delta_bad)
    assert abs(ratio_bad / ratio_ref - 1) > 0.1

    # Causality check
    wzero_idx = delta_iw.shape[0] // 2
    delta_imag_diag = np.diagonal(delta_iw[wzero_idx:,0].imag, axis1=1, axis2=2)
    assert np.all(delta_imag_diag < 1e-12)


def test_causal_projection_boson_with_real_downfold_coulomb(downfold_inputs):
    pytest.importorskip("adapol")

    assert downfold_inputs["iaft"].basis == downfold_inputs["basis"]

    ir_kernel = downfold_inputs["iaft"]
    wloc_t = downfold_inputs["wloc_t"]
    wloc_iw = ir_kernel.tau_to_w_phsym(wloc_t, stats="b")
    nbnd = wloc_iw.shape[-1]
    wloc_dd_iw = wloc_iw.reshape(-1, nbnd * nbnd, nbnd * nbnd)

    wloc_fit_iw = causal_projection_boson(
        wloc_dd_iw,
        ir_kernel,
        2,
        ph_symmetry=True,
        n_exclude_low_freq=0,
        target_name="Wloc downfolded",
    )

    assert wloc_fit_iw.shape == wloc_dd_iw.shape
    assert np.allclose(wloc_fit_iw.imag, 0.0, atol=1e-12)


def _rand_c(rng, *shape):
    return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)


def _rand_herm(rng, n, scale=1.0):
    a = _rand_c(rng, n, n)
    return 0.5 * scale * (a + a.conj().T)


def _fd_richardson(f, h):
    """O(h^4) derivative at 0: Richardson extrapolation of central differences with steps h and h/2."""
    def d(e):
        fp, fm = f(e), f(-e)
        return [(p - m) / (2 * e) for p, m in zip(fp, fm)]
    d1, d2 = d(h), d(h / 2)
    return [(4 * b - a) / 3 for a, b in zip(d1, d2)]


def _rel(a, b):
    return np.linalg.norm(a - b) / np.linalg.norm(b)


@pytest.fixture(scope="module")
def weiss_lr_inputs():
    """Random three-orbital, two-spin inputs of the Weiss maps (ground state and generic perturbation), Q = 0."""
    pytest.importorskip("coqui._lib.iaft_module")
    rng = np.random.default_rng(2026)
    nbnd, nspin, nbnd2 = 3, 2, 9
    iaft = IAFT(beta=10.0, wmax=10.0, prec="medium", basis="dlr", verbose=False)
    wn_f = iaft.wn_mesh('f') * np.pi / iaft.beta
    nu_b = iaft.wn_mesh('b', positive_only=True) * np.pi / iaft.beta
    eye = np.eye(nbnd)

    # Fermions: G = (iw - H)^-1, Hermitian in iw; Sigma = B (iw - E)^-1 B^dagger; perturbations generic.
    H = np.array([_rand_herm(rng, nbnd) for _ in range(nspin)])
    G_w = np.array([[np.linalg.inv(1j * w * eye - H[s]) for s in range(nspin)] for w in wn_f])
    dH = _rand_c(rng, nspin, nbnd, nbnd)
    dHloc = _rand_c(rng, nspin, nbnd, nbnd)
    dG_w = np.einsum('wsab,sbc,wscd->wsad', G_w, dH, G_w)
    E = _rand_herm(rng, nbnd)
    B, B2, C2 = (0.5 * _rand_c(rng, nspin, nbnd, nbnd) for _ in range(3))
    R_w = np.array([np.linalg.inv(1j * w * eye - E) for w in wn_f])
    Sigma_w = np.einsum('sab,wbc,sdc->wsad', B, R_w, B.conj())
    dSigma_w = np.einsum('sab,wbc,scd->wsad', B2, R_w, C2)
    Vhf = np.array([_rand_herm(rng, nbnd, 0.3) for _ in range(nspin)])
    dVhf = 0.3 * _rand_c(rng, nspin, nbnd, nbnd)

    # Bosons (product basis, positive half mesh): Lorentzian-shaped W and Pi, Hermitian V; perturbations generic.
    lor = 2.0 / (nu_b**2 + 4.0)
    V = (_rand_herm(rng, nbnd2) + 3 * np.eye(nbnd2)).reshape((nbnd,) * 4)
    W_w = np.einsum('w,ab->wab', lor, _rand_herm(rng, nbnd2)).reshape((-1,) + (nbnd,) * 4)
    Pi_w = np.einsum('w,ab->wab', -lor, _rand_herm(rng, nbnd2, 0.05)).reshape(W_w.shape)
    dW_w = np.einsum('w,ab->wab', lor, _rand_c(rng, nbnd2, nbnd2)).reshape(W_w.shape)
    dPi_w = np.einsum('w,ab->wab', lor, 0.05 * _rand_c(rng, nbnd2, nbnd2)).reshape(W_w.shape)

    return dict(iaft=iaft, G_w=G_w, dG_w=dG_w, Hloc=H, dHloc=dHloc, Vhf=Vhf, dVhf=dVhf, Sigma_w=Sigma_w, dSigma_w=dSigma_w,
                V=V, W_w=W_w, dW_w=dW_w, Pi_w=Pi_w, dPi_w=dPi_w,
                G_t=iaft.w_to_tau(G_w, 'f'), dG_t=iaft.w_to_tau(dG_w, 'f'),
                W_t=iaft.w_to_tau_phsym(W_w, 'b'), dW_t=iaft.w_to_tau_phsym(dW_w, 'b'))


def test_lr_weiss_fermion_boson_multiorbital(weiss_lr_inputs):
    x, h = weiss_lr_inputs, 1e-4
    dg, _ = lr_weiss.lr_weiss_fermion_w(x["G_w"], x["dG_w"], x["Vhf"], x["dVhf"], x["Sigma_w"], x["dSigma_w"])
    dg_fd, = _fd_richardson(lambda e: [weiss.compute_weiss_fermion_w(
        x["G_w"] + e * x["dG_w"], x["Vhf"] + e * x["dVhf"], x["Sigma_w"] + e * x["dSigma_w"])], h)
    assert _rel(dg, dg_fd) < 1e-9

    du = lr_weiss.lr_weiss_boson_w(x["V"], x["W_w"], x["dW_w"], x["Pi_w"], x["dPi_w"])
    du_fd, = _fd_richardson(lambda e: [weiss.compute_weiss_boson_w(
        x["V"], x["W_w"] + e * x["dW_w"], x["Pi_w"] + e * x["dPi_w"])], h)
    assert _rel(du, du_fd) < 1e-9


# lr_compute_weiss_fields_w (all four outputs, dHloc generic) vs the Richardson central difference of
# compute_weiss_fields_w.
def test_lr_compute_weiss_fields_multiorbital(weiss_lr_inputs):
    x, h = weiss_lr_inputs, 1e-4
    def gs(e):
        return list(weiss.compute_weiss_fields_w(
            iaft=x["iaft"],
            local_gf={"Gloc_t": x["G_t"] + e * x["dG_t"], "Hloc": x["Hloc"] + e * x["dHloc"],
                      "Wloc_t": x["W_t"] + e * x["dW_t"], "Vloc": x["V"]},
            impurity_selfenergies={"Vhf_imp": x["Vhf"] + e * x["dVhf"],
                                   "Sigma_imp_w": x["Sigma_w"] + e * x["dSigma_w"],
                                   "Pi_imp_w": x["Pi_w"] + e * x["dPi_w"]}))
    out = lr_weiss.lr_compute_weiss_fields_w(
        iaft=x["iaft"],
        local_gf={"Gloc_t": x["G_t"], "Wloc_t": x["W_t"], "Vloc": x["V"]},
        d_local_gf={"Gloc_t": x["dG_t"], "Hloc": x["dHloc"], "Wloc_t": x["dW_t"]},
        impurity_selfenergies={"Vhf_imp": x["Vhf"], "Sigma_imp_w": x["Sigma_w"], "Pi_imp_w": x["Pi_w"]},
        d_impurity_selfenergies={"Vhf_imp": x["dVhf"], "Sigma_imp_w": x["dSigma_w"], "Pi_imp_w": x["dPi_w"]})
    # outputs: (dg_weiss, du_weiss, dh0, ddelta) vs the derivative of (g_weiss, u_weiss, h0, delta)
    for name, d, d_fd in zip(("dg_weiss", "du_weiss", "dh0", "ddelta"), out, _fd_richardson(gs, h)):
        assert _rel(d, d_fd) < 1e-9, name
