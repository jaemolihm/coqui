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

from mpi4py import MPI
import os
import shutil

import h5py
import numpy as np
import pytest

import coqui
from coqui.mean_field import write_model_h5
from coqui.interaction import write_thc_h5
from coqui.utils.tests.test_coqui_env import mpi


def square_lattice(N, t, tp):
    i = np.arange(N) / N
    kc = np.stack(np.meshgrid(i, i, indexing="ij"), -1).reshape(-1, 2)
    kpts = np.zeros((N * N, 3))
    kpts[:, :2] = kc
    kx, ky = 2 * np.pi * kc[:, 0], 2 * np.pi * kc[:, 1]
    ek = -2 * t * (np.cos(kx) + np.cos(ky)) - 4 * tp * np.cos(kx) * np.cos(ky)
    return kpts, ek


def read_complex(ds):
    a = ds[...]
    return a[..., 0] + 1j * a[..., 1] if "__complex__" in ds.attrs or a.shape[-1] == 2 else a


@pytest.mark.parametrize("with_klist, model_div, madelung",
                         [(True, None, 0.0), (False, None, 0.0), (True, "gygi", 0.05)])
def test_hubbard_writers_hf(mpi, with_klist, model_div, madelung):
    N, t, tp, U, nelec = 4, 0.1, -0.03, 0.4, 0.85
    nk = N * N
    kpts, ek = square_lattice(N, t, tp)
    H0 = ek.reshape(1, nk, 1, 1).astype(complex)
    tag = f"{int(with_klist)}_{model_div}"
    prefix, thc_file = f"hubbard_test_{tag}", f"hubbard_test_thc_{tag}.h5"

    write_model_h5(f"./{prefix}.h5", H0, nelec,
                   kpts_crys=kpts if with_klist else None,
                   kp_grid=None if with_klist else (N, N, 1),
                   latt=np.diag([1.0, 1.0, 20.0]), madelung=madelung, div_treatment=model_div, mpi=mpi)
    mf = coqui.make_mf(mpi, {"prefix": prefix, "outdir": "./"}, "model")
    assert mf.nkpts() == nk and mf.nkpts_ibz() == nk and mf.nbnd() == 1
    assert np.abs(np.asarray(mf.kpts_crystal()) - kpts).max() < 1e-12
    assert abs(np.sum(mf.k_weights()) - 1.0) < 1e-14

    write_thc_h5(thc_file, np.ones((1, nk, 1, 1)), np.full((nk, 1, 1), U), mpi=mpi)
    if mpi.root():
        with h5py.File(thc_file, "r") as f:
            assert f["Np"][()] == 1
            assert f["interpolating_points"].shape == (1,)
            assert np.abs(read_complex(f["coulomb_matrix"]) - U).max() == 0.0
    mpi.barrier()
    thc = coqui.make_thc_coulomb(mf, {"save": thc_file, "storage": "incore"})

    hf_params = {
        "restart": False, "prefix": prefix, "outdir": "./", "niter": 1, "beta": 50.0,
        "div_treatment": "gygi", "hf_div_treatment": "gygi",
        "iaft": {"prec": "medium", "wmax": 3.0},
    }
    coqui.run_hf(hf_params, h_int=thc)
    mpi.barrier()

    # Hubbard HF: F - H0 = 2 U n_sigma - U n_sigma = U n_sigma for every k (paramagnetic, ns=1). The input
    # treatment "gygi" is overridden by the model file: "ignore_g0" by default, and with the file's "gygi"
    # the exchange gets the finite-size correction -madelung * n_sigma(k).
    if mpi.root():
        with h5py.File(f"{prefix}.mbpt.h5", "r") as f:
            it = f["scf/final_iter"][()]
            F = read_complex(f[f"scf/iter{it}/F_skij"])
            Dm = read_complex(f[f"scf/iter{it}/Dm_skij"])
            Dm_in = read_complex(f[f"scf/iter{it - 1}/Dm_skij"])  # density F was built from
        n_sigma = Dm[0, :, 0, 0].real.mean()
        assert np.abs(F[0, :, 0, 0] - (U * n_sigma - madelung * Dm_in[0, :, 0, 0])).max() < 1e-12
        assert abs(2 * n_sigma - nelec) < 1e-8
    mpi.barrier()

    if mpi.root():
        for fn in (f"{prefix}.h5", thc_file, f"{prefix}.mbpt.h5"):
            if os.path.exists(fn):
                os.remove(fn)
    mpi.barrier()


# Converged scGW of the extended Hubbard model (N = 4, t = 0.1, U = 4t, V = 0.5t, beta t = 5, n = 1) from an
# independent NumPy scGW code on the same DLR grid (charge-channel GW, W = V(q) / (1 - V(q) Pi(q)), same-spin
# bubbles included). k in crystal coordinates; Sigma_c(k, i w_0) is the dynamic self-energy at w_0 = pi/beta.
SCGW_REF = {
    "mu": 0.4000000000000865,
    "sigma_hf": {(0.0, 0.0): 0.36460269435736026, (0.5, 0.0): 0.40000000000004476,
                 (0.25, 0.25): 0.40000000000004476, (0.5, 0.5): 0.4353973056427293},
    "sigma_c_w0": {(0.0, 0.0): 0.011914758218137186 - 0.016580027000820105j,
                   (0.5, 0.0): 0.0 - 0.01874539067888059j,
                   (0.25, 0.25): 0.0 - 0.01874539067888059j,
                   (0.5, 0.5): -0.011914758218106678 - 0.01658002700082039j},
}


def test_extended_hubbard_scgw(mpi):
    from coqui.utils.imag_axes_ft import IAFT
    N, t, U, V, beta = 4, 0.1, 0.4, 0.05, 50.0
    nk = N * N
    kpts, ek = square_lattice(N, t, 0.0)
    qx, qy = 2 * np.pi * kpts[:, 0], 2 * np.pi * kpts[:, 1]
    vq = U + 2 * V * (np.cos(qx) + np.cos(qy))
    prefix, thc_file = "ext_hubbard_scgw", "ext_hubbard_scgw_thc.h5"
    write_model_h5(f"./{prefix}.h5", ek.reshape(1, nk, 1, 1).astype(complex), 1.0, kpts_crys=kpts,
                   kp_grid=(N, N, 1), latt=np.diag([1.0, 1.0, 50.0]), mpi=mpi)
    write_thc_h5(thc_file, np.ones((1, nk, 1, 1)), vq.reshape(nk, 1, 1).astype(complex), mpi=mpi)
    mf = coqui.make_mf(mpi, {"prefix": prefix, "outdir": "./"}, "model")
    thc = coqui.make_thc_coulomb(mf, {"save": thc_file, "storage": "incore"})

    common = {"outdir": "./", "prefix": prefix, "beta": beta, "conv_thr": 1e-11,
              "iaft": {"prec": "high", "wmax": 3.0, "basis": "dlr"}}
    coqui.run_hf(dict(common, restart=False, niter=20, iter_alg={"alg": "damping", "mixing": 0.7}), h_int=thc)
    coqui.run_gw(dict(common, restart=True, niter=100, iter_alg={"alg": "damping", "mixing": 0.5}), h_int=thc)
    mpi.barrier()

    if mpi.root():
        chk = f"{prefix}.mbpt.h5"
        with h5py.File(chk, "r") as f:
            it = f["scf/final_iter"][()]
            F = read_complex(f[f"scf/iter{it}/F_skij"])[0, :, 0, 0]
            S_t = read_complex(f[f"scf/iter{it}/Sigma_tskij"])[:, 0, :, 0, 0]
            mu = f[f"scf/iter{it}/mu"][()]
        assert it < 100, "scGW did not converge"
        ft = IAFT.from_coqui_chkpt(chk, verbose=False)
        S_w = ft.tau_to_w(S_t, "f")
        i0 = int(np.where(np.asarray(ft.wn_mesh("f")) == 1)[0][0])
        assert abs(mu - SCGW_REF["mu"]) < 1e-8
        for k, ref in SCGW_REF["sigma_hf"].items():
            ik = int(np.where(np.abs(kpts[:, :2] - k).max(axis=1) < 1e-12)[0][0])
            assert abs(F[ik] - ref) < 1e-8, (k, F[ik], ref)
        for k, ref in SCGW_REF["sigma_c_w0"].items():
            ik = int(np.where(np.abs(kpts[:, :2] - k).max(axis=1) < 1e-12)[0][0])
            assert abs(S_w[i0, ik] - ref) < 1e-8, (k, S_w[i0, ik], ref)
    mpi.barrier()

    if mpi.root():
        for fn in (f"{prefix}.h5", thc_file, f"{prefix}.mbpt.h5"):
            if os.path.exists(fn):
                os.remove(fn)
    mpi.barrier()


def test_hubbard_scgw_fixed_mu(mpi):
    """scGW restarts at fixed mu = mu0 +- dmu (input "mu"): every iteration keeps mu, and particle-hole symmetry
    of the half-filled t' = 0 Hubbard model gives n(mu0 + dmu) - 1 = 1 - n(mu0 - dmu)."""
    N, t, U, beta, dmu = 4, 0.1, 0.4, 50.0, 0.01
    nk = N * N
    kpts, ek = square_lattice(N, t, 0.0)
    prefix, thc_file = "hubbard_fixed_mu", "hubbard_fixed_mu_thc.h5"
    write_model_h5(f"./{prefix}.h5", ek.reshape(1, nk, 1, 1).astype(complex), 1.0, kpts_crys=kpts,
                   kp_grid=(N, N, 1), latt=np.diag([1.0, 1.0, 50.0]), mpi=mpi)
    write_thc_h5(thc_file, np.ones((1, nk, 1, 1)), np.full((nk, 1, 1), U, dtype=complex), mpi=mpi)
    mf = coqui.make_mf(mpi, {"prefix": prefix, "outdir": "./"}, "model")
    thc = coqui.make_thc_coulomb(mf, {"save": thc_file, "storage": "incore"})

    common = {"outdir": "./", "prefix": prefix, "beta": beta, "conv_thr": 1e-11,
              "iaft": {"prec": "high", "wmax": 3.0, "basis": "dlr"}}
    coqui.run_hf(dict(common, restart=False, niter=20, iter_alg={"alg": "damping", "mixing": 0.7}), h_int=thc)
    coqui.run_gw(dict(common, restart=True, niter=100, iter_alg={"alg": "damping", "mixing": 0.5}), h_int=thc)
    mpi.barrier()
    with h5py.File(f"{prefix}.mbpt.h5", "r") as f:
        it0 = f["scf/final_iter"][()]
        mu0 = f[f"scf/iter{it0}/mu"][()]
    assert abs(mu0 - U / 2) < 1e-8

    def density(chk):
        with h5py.File(chk, "r") as f:
            it = f["scf/final_iter"][()]
            mus = [f[f"scf/iter{i}/mu"][()] for i in range(it0 + 1, it + 1)]
            dm = read_complex(f[f"scf/iter{it}/Dm_skij"])[0, :, 0, 0].real
            w = f["system/k_weight"][()]
        return it, mus, 2 * np.dot(w, dm)

    n = {}
    for sign in (+1, -1):
        pre = f"{prefix}_mu{sign:+d}"
        if mpi.root():
            shutil.copy(f"{prefix}.mbpt.h5", f"{pre}.mbpt.h5")
        mpi.barrier()
        coqui.run_gw(dict(common, prefix=pre, restart=True, niter=100, mu=mu0 + sign * dmu,
                          iter_alg={"alg": "damping", "mixing": 0.5}), h_int=thc)
        mpi.barrier()
        it, mus, n[sign] = density(f"{pre}.mbpt.h5")
        assert it0 < it < it0 + 100, "fixed-mu scGW did not converge"
        assert np.allclose(mus, mu0 + sign * dmu, rtol=0, atol=1e-14)
    assert n[+1] - 1 > 1e-3
    assert abs((n[+1] - 1) - (1 - n[-1])) < 1e-8, (n[+1], n[-1])
    mpi.barrier()

    if mpi.root():
        for fn in (f"{prefix}.h5", thc_file, f"{prefix}.mbpt.h5", f"{prefix}_mu+1.mbpt.h5", f"{prefix}_mu-1.mbpt.h5"):
            if os.path.exists(fn):
                os.remove(fn)
    mpi.barrier()


def test_gw_edmft_params_forward_mu(tmp_path):
    """gw_edmft "mu" reaches the GW and dmft_embed groups, and nowhere when absent."""
    pytest.importorskip("triqs")
    from coqui.dmft.io import convert_gw_edmft_params
    (tmp_path / "coqui.mbpt.h5").touch()
    base = {"niter": 1, "outdir": str(tmp_path), "prefix": "coqui",
            "edmft": {"impurity": {"n_cycles": 100, "n_warmup_cycles": 10, "length_cycle": 5}}}
    p = convert_gw_edmft_params(dict(base, mu=0.123))
    assert p["gw"]["mu"] == 0.123
    assert p["dmft_embed"]["mu"] == 0.123
    assert "mu" not in p["wloc"] and "mu" not in p["gloc"] and "mu" not in p["impurity"]
    p = convert_gw_edmft_params(base)
    assert "mu" not in p["gw"] and "mu" not in p["dmft_embed"]


def test_impurity_mu_update_validation():
    pytest.importorskip("triqs")
    from coqui.dmft.io import _normalize_solver_params_list
    solver = {"n_cycles": 100, "n_warmup_cycles": 10, "length_cycle": 5}
    for update in ("search", "predict"):
        _normalize_solver_params_list([dict(solver, chemical_potential={"update": update})], 1)
    with pytest.raises(ValueError):
        _normalize_solver_params_list([dict(solver, chemical_potential={"update": "serach"})], 1)


# LR on the t' != 0, n = 0.85 Hubbard model (particle-hole symmetry broken, so the GW kernel carries a
# non-zero DeltaW): per (q, kernel), DeltaN = 2 sum_k w_k DeltaDm_k (q = 0 only) and ||DeltaDm||, generated
# at np = 1. The LR result does not depend on the MPI distribution, so every rank count reproduces them;
# np = 2, 3, 4, 8 and 32 put the one-orbital (Np = 1, nbnd = 1) arrays on the fallback grids.
# W is recomputed from the checkpoint G, except for "GW0_file", which reads the W the ground state dumps
# (dump_w_to_h5) and must reproduce "GW0".
LR_METHODS = ("Hartree", "HF", "HSEX", "GW0", "GW0_file", "GW")
LR_Q = {"q0": (0.0, 0.0, 0.0), "q1": (0.25, 0.0, 0.0)}
# The references are np = 1 solves converged to tol = 1e-13 (the test's 1e-10 lands within 1e-11 of them). They
# depend on the DLR grid: they are those of cppdlr 7a0f60c (cppdlr f6bd6ab differs by ~1e-9 at q0 and up to 2e-6
# relative at q1).
LR_REF = {
    ("q0", "Hartree"): (1.070176354223493, 3.872862523032440),
    ("q0", "HF"): (1.361608625076144, 4.927527126051464),
    ("q0", "HSEX"): (1.174732178269491, 4.264864905394354),
    ("q0", "GW0"): (1.324067851334970, 4.791980130300074),
    ("q0", "GW"): (1.351445166030098, 4.947668504661141),
    ("q1", "Hartree"): (None, 5.676292692311771),
    ("q1", "HF"): (None, 5.920503262449584),
    ("q1", "HSEX"): (None, 5.913910231009422),
    ("q1", "GW0"): (None, 5.922730387520320),
    ("q1", "GW"): (None, 6.009907824914611),
}
LR_W_FILE_TOL = 1e-12


@pytest.fixture(scope="module")
def hubbard_lr_gs(mpi):
    """HF + scGW ground state of the N = 4, t' = -0.3t, n = 0.85 Hubbard model."""
    N, t, tp, U, nelec, beta = 4, 0.1, -0.03, 0.4, 0.85, 50.0
    nk = N * N
    kpts, ek = square_lattice(N, t, tp)
    prefix, thc_file = "hubbard_lr", "hubbard_lr_thc.h5"
    write_model_h5(f"./{prefix}.h5", ek.reshape(1, nk, 1, 1).astype(complex), nelec, kpts_crys=kpts,
                   kp_grid=(N, N, 1), latt=np.diag([1.0, 1.0, 50.0]), mpi=mpi)
    write_thc_h5(thc_file, np.ones((1, nk, 1, 1)), np.full((nk, 1, 1), U, dtype=complex), mpi=mpi)
    mf = coqui.make_mf(mpi, {"prefix": prefix, "outdir": "./"}, "model")
    thc = coqui.make_thc_coulomb(mf, {"save": thc_file, "storage": "incore"})
    common = {"outdir": "./", "prefix": prefix, "beta": beta, "conv_thr": 1e-11,
              "iaft": {"prec": "high", "wmax": 3.0, "basis": "dlr"}}
    coqui.run_hf(dict(common, restart=False, niter=40, iter_alg={"alg": "damping", "mixing": 0.7}), h_int=thc)
    coqui.run_gw(dict(common, restart=True, niter=100, iter_alg={"alg": "damping", "mixing": 0.5}), h_int=thc)
    mpi.barrier()
    with h5py.File(f"{prefix}.mbpt.h5", "r") as f:
        assert f["scf/final_iter"][()] < 140, "scGW did not converge"
    # W of the converged G: one more scGW iteration with dump_w_to_h5 on a copy of the checkpoint in wdir
    # (the W of iteration n + 1 is built from the G of iteration n, the one run_lr linearizes)
    wdir = f"./{prefix}_wdump"
    if mpi.root():
        os.makedirs(wdir, exist_ok=True)
        shutil.copy(f"{prefix}.mbpt.h5", f"{wdir}/{prefix}.mbpt.h5")
    mpi.barrier()
    coqui.run_gw(dict(common, outdir=wdir, restart=True, niter=1, dump_w_to_h5=True,
                      iter_alg={"alg": "damping", "mixing": 0.5}), h_int=thc)
    mpi.barrier()
    yield {"prefix": prefix, "thc": thc, "kpts": kpts, "common": common,
           "w_file": f"{wdir}/thc_screened_interaction.h5"}

    mpi.barrier()
    if mpi.root():
        for fn in (f"{prefix}.h5", thc_file, f"{prefix}.mbpt.h5"):
            if os.path.exists(fn):
                os.remove(fn)
        shutil.rmtree(wdir, ignore_errors=True)
    mpi.barrier()


def run_lr_chi(mpi, gs, method, q, dH0, tag):
    """run_lr on the ground state of hubbard_lr_gs; returns (DeltaN, DeltaDm_skij)."""
    out = f"{gs['prefix']}_lr_{tag}_{method}"
    if mpi.root():
        h5py.File(f"{out}.mbpt.h5", "w").close()
    mpi.barrier()
    if method == "GW0_file":
        w_kw = {"method": "GW0", "screened_interaction_file": gs["w_file"]}
    else:
        w_kw = {"method": method, "recompute_W": method in ("HSEX", "GW0", "GW")}
    coqui.mbpt.run_lr({"prefix": f"./{gs['prefix']}", "output": out, "h0_source": "checkpoint"},
                      gs["thc"], np.asarray(q), dH0 if mpi.root() else None,
                      fix_density=False, max_iter=100, tol=1e-10, iter_alg={"alg": "GCR"}, **w_kw)
    mpi.barrier()
    with h5py.File(f"{out}.mbpt.h5", "r") as f:
        ddm = read_complex(f["linear_response/DeltaDm_skij"])
    with h5py.File(f"{gs['prefix']}.mbpt.h5", "r") as f:
        w = f["system/k_weight"][()]
    mpi.barrier()
    if mpi.root():
        os.remove(f"{out}.mbpt.h5")
    mpi.barrier()
    return 2.0 * np.einsum("k,kii->", w, ddm[0]), ddm


@pytest.mark.parametrize("qtag", list(LR_Q))
def test_hubbard_lr_ladder(mpi, hubbard_lr_gs, qtag):
    """Hartree, HF, HSEX, GW0 and GW LR at q = 0 (DeltaH0 = -1, the charge susceptibility) and at finite q
    (DeltaH0_k = cos(2 pi k_x)) against the np = 1 references; GW0 with W read from file against GW0."""
    gs = hubbard_lr_gs
    kpts = gs["kpts"]
    nk = kpts.shape[0]
    if qtag == "q0":
        dH0 = -np.ones((1, nk, 1, 1), dtype=complex)
    else:
        dH0 = np.cos(2 * np.pi * kpts[:, 0]).reshape(1, nk, 1, 1).astype(complex)
    res = {}
    for method in LR_METHODS:
        dn, ddm = run_lr_chi(mpi, gs, method, LR_Q[qtag], dH0, qtag)
        res[method] = (dn, ddm)
        if mpi.root():
            print(f"LR_LADDER {qtag} {method}: DeltaN = {dn.real:.15e} {dn.imag:+.3e}j, "
                  f"||DeltaDm|| = {np.linalg.norm(ddm):.15e}", flush=True)
    # DeltaW != 0: the GW kernel differs from GW0 by -G DeltaW
    assert np.linalg.norm(res["GW"][1] - res["GW0"][1]) > 1e-4 * np.linalg.norm(res["GW0"][1])
    e_file = np.linalg.norm(res["GW0_file"][1] - res["GW0"][1]) / np.linalg.norm(res["GW0"][1])
    if mpi.root():
        print(f"LR_LADDER {qtag}: ||DeltaDm(GW0_file) - DeltaDm(GW0)|| / ||DeltaDm(GW0)|| = {e_file:.3e}", flush=True)
    assert e_file < LR_W_FILE_TOL
    for method in LR_METHODS:
        if method == "GW0_file":
            continue
        dn, ddm = res[method]
        ref_dn, ref_norm = LR_REF[(qtag, method)]
        if qtag == "q0":
            assert abs(dn - ref_dn) < 1e-9, (method, dn, ref_dn)
        assert abs(np.linalg.norm(ddm) - ref_norm) < 1e-9, (method, np.linalg.norm(ddm), ref_norm)


def test_hubbard_lr_gw_fd(mpi, hubbard_lr_gs):
    """LR GW at q = 0 is the exact derivative of scGW: chi = dN/dmu against fixed-mu scGW runs at mu0 +- h.
    The central difference is off by ~ -53 h^2 here, so it is Richardson-extrapolated from h and 2h."""
    gs = hubbard_lr_gs
    nk = gs["kpts"].shape[0]
    prefix, dmu = gs["prefix"], 5e-4
    chi, _ = run_lr_chi(mpi, gs, "GW", LR_Q["q0"], -np.ones((1, nk, 1, 1), dtype=complex), "fd")
    with h5py.File(f"{prefix}.mbpt.h5", "r") as f:
        it0 = f["scf/final_iter"][()]
        mu0 = f[f"scf/iter{it0}/mu"][()]
    chi_cd = {}
    for h in (dmu, 2 * dmu):
        n = {}
        for sign in (+1, -1):
            pre = f"{prefix}_fd_mu{sign:+d}"
            if mpi.root():
                shutil.copy(f"{prefix}.mbpt.h5", f"{pre}.mbpt.h5")
            mpi.barrier()
            coqui.run_gw(dict(gs["common"], prefix=pre, restart=True, niter=200, mu=mu0 + sign * h,
                              conv_thr=1e-13, iter_alg={"alg": "damping", "mixing": 0.5}), h_int=gs["thc"])
            mpi.barrier()
            with h5py.File(f"{pre}.mbpt.h5", "r") as f:
                it = f["scf/final_iter"][()]
                dm = read_complex(f[f"scf/iter{it}/Dm_skij"])[0, :, 0, 0].real
                w = f["system/k_weight"][()]
            assert it0 < it < it0 + 200, "fixed-mu scGW did not converge"
            n[sign] = 2 * np.dot(w, dm)
        chi_cd[h] = (n[+1] - n[-1]) / (2 * h)
    chi_fd = (4 * chi_cd[dmu] - chi_cd[2 * dmu]) / 3
    if mpi.root():
        print(f"LR_GW_FD: chi_LR = {chi.real:.12f}, chi_FD = {chi_fd:.12f} (central differences "
              f"{chi_cd[dmu]:.12f}, {chi_cd[2 * dmu]:.12f}), diff = {chi.real - chi_fd:.3e}", flush=True)
    assert abs(chi.imag) < 1e-10
    assert abs(chi.real - chi_fd) < 1e-8 * abs(chi_fd), (chi, chi_fd)
    mpi.barrier()
    if mpi.root():
        for sign in (+1, -1):
            fn = f"{prefix}_fd_mu{sign:+d}.mbpt.h5"
            if os.path.exists(fn):
                os.remove(fn)
    mpi.barrier()


def test_write_model_h5_validation():
    H0 = np.zeros((1, 4, 1, 1), complex)
    with pytest.raises(ValueError):
        write_model_h5("never_written.h5", H0, 1.0)  # no k-points
    with pytest.raises(ValueError):
        write_model_h5("never_written.h5", H0, 1.0, kpts_crys=np.full((4, 3), 0.5))  # not Gamma first
    with pytest.raises(ValueError):
        write_model_h5("never_written.h5", H0, 1.0, kp_grid=(3, 1, 1))
    with pytest.raises(ValueError):
        write_model_h5("never_written.h5", H0, 1.0, kp_grid=(4, 1, 1), overlap=2 * np.ones_like(H0))
    with pytest.raises(ValueError):
        write_model_h5("never_written.h5", H0, 1.0, kp_grid=(4, 1, 1), kpoint_weights=np.full(4, 0.3))
    with pytest.raises(ValueError):
        write_thc_h5("never_written.h5", np.ones((1, 4, 1, 1)), np.ones((4, 2, 2)))  # Np mismatch
    with pytest.raises(ValueError):
        write_thc_h5("never_written.h5", np.ones((1, 4, 2, 1)),
                     np.array([[[1, 1j], [1j, 1]]] * 4))  # not Hermitian
    assert not os.path.exists("never_written.h5")
