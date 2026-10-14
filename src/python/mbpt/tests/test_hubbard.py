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
