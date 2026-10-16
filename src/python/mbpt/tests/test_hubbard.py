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


def test_model_spectrum_degenerate_shell(mpi):
    # nk = 2, nb = 3, ns = 1: each state holds 2 * 1/2 = 1 electron. The level e0 is threefold degenerate
    # (twice at k = 0, in a rotated basis, once at k = 1) and holds 1.5 of the 2.5 electrons, so each of
    # its members is half filled and efermi = e0.
    e0 = 0.3
    rot = np.linalg.qr(np.array([[1.0, 0.4j, 0.2], [0.1, 1.0, -0.3j], [0.5, 0.2, 1.0]]))[0]
    H0 = np.zeros((1, 2, 3, 3), dtype=complex)
    H0[0, 0] = rot @ np.diag([-1.0, e0, e0]) @ rot.conj().T
    H0[0, 1] = np.diag([e0, 1.0, 2.0])
    prefix = "model_degenerate_test"
    write_model_h5(f"./{prefix}.h5", H0, 2.5, kpts_crys=[[0, 0, 0], [0.5, 0, 0]], mpi=mpi)
    if mpi.root():
        with h5py.File(f"{prefix}.h5", "r") as f:
            eigval, occ = f["System/eigval"][()], f["System/occ"][()]
            efermi = f["System"].attrs["fermi_energy"]
        os.remove(f"{prefix}.h5")
        shell = np.abs(eigval - e0) < 1e-10
        assert shell.sum() == 3
        assert np.abs(occ[shell] - 0.5).max() < 1e-14
        assert np.all(occ[eigval < e0 - 1e-10] == 1.0) and np.all(occ[eigval > e0 + 1e-10] == 0.0)
        assert abs(efermi - e0) < 1e-12
    mpi.barrier()

    with pytest.raises(ValueError, match="Hermitian"):
        H0[0, 1, 0, 1] = 0.1
        write_model_h5(f"./{prefix}.h5", H0, 2.5, kpts_crys=[[0, 0, 0], [0.5, 0, 0]], mpi=mpi)


@pytest.mark.parametrize("hf_div, madelung", [("gygi", 0.0), ("gygi", 0.05), ("ignore_g0", 0.05)])
def test_hubbard_writers_hf(mpi, hf_div, madelung):
    N, t, tp, U, nelec = 4, 0.1, -0.03, 0.4, 0.85
    nk = N * N
    kpts, ek = square_lattice(N, t, tp)
    H0 = ek.reshape(1, nk, 1, 1).astype(complex)
    tag = f"{hf_div}_{madelung}"
    prefix, thc_file = f"hubbard_test_{tag}", f"hubbard_test_thc_{tag}.h5"

    write_model_h5(f"./{prefix}.h5", H0, nelec,
                   kpts_crys=kpts,
                   latt=np.diag([1.0, 1.0, 20.0]), madelung=madelung, mpi=mpi)
    mf = coqui.make_mf(mpi, {"prefix": prefix, "outdir": "./"}, "model")
    assert mf.nkpts() == nk and mf.nkpts_ibz() == nk and mf.nbnd() == 1
    assert np.abs(np.asarray(mf.kpts_crystal()) - kpts).max() < 1e-12
    assert abs(np.sum(mf.k_weights()) - 1.0) < 1e-14

    # mean-field spectrum written with the model: eigval = eps_k, T = 0 filling of nelec electrons, and
    # efermi between the highest occupied and the lowest not fully occupied level
    if mpi.root():
        with h5py.File(f"{prefix}.h5", "r") as f:
            eigval, occ = f["System/eigval"][()], f["System/occ"][()]
            efermi = f["System"].attrs["fermi_energy"]
        assert eigval.shape == occ.shape == (1, nk, 1)
        assert np.abs(eigval[0, :, 0] - ek).max() < 1e-14
        assert abs(np.sum(occ[0, :, 0] * mf.k_weights() * 2) - nelec) < 1e-14
        assert np.all((occ >= 0) & (occ <= 1))
        assert ek[occ[0, :, 0] > 0].max() <= efermi <= ek[occ[0, :, 0] < 1].min()
    mpi.barrier()

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
        "div_treatment": "gygi", "hf_div_treatment": hf_div,
        "iaft": {"prec": "medium", "wmax": 3.0},
    }
    coqui.run_hf(hf_params, h_int=thc)
    mpi.barrier()

    # Hubbard HF: F - H0 = 2 U n_sigma - U n_sigma = U n_sigma for every k (paramagnetic, ns=1). With
    # hf_div_treatment="gygi" the exchange gets the finite-size correction -madelung * n_sigma(k), which
    # vanishes for the default madelung = 0; "ignore_g0" drops it.
    if mpi.root():
        with h5py.File(f"{prefix}.mbpt.h5", "r") as f:
            it = f["scf/final_iter"][()]
            F = read_complex(f[f"scf/iter{it}/F_skij"])
            Dm = read_complex(f[f"scf/iter{it}/Dm_skij"])
            Dm_in = read_complex(f[f"scf/iter{it - 1}/Dm_skij"])  # density F was built from
        n_sigma = Dm[0, :, 0, 0].real.mean()
        v_m = madelung if hf_div == "gygi" else 0.0
        assert np.abs(F[0, :, 0, 0] - (U * n_sigma - v_m * Dm_in[0, :, 0, 0])).max() < 1e-12
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
                   latt=np.diag([1.0, 1.0, 50.0]), mpi=mpi)
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
            div = f["scf/div_treatment"][()]
            heads = [read_complex(f[f"scf/{g}/eps_inv_head_w"]) for g in f["scf"]
                     if g.startswith("iter") and "eps_inv_head_w" in f[f"scf/{g}"]]
        assert it < 100, "scGW did not converge"
        # the input default "gygi" is honoured; with the zero q->0 heads of the model THC file and
        # madelung = 0 its extrapolated eps^-1 head is exactly 0 (finite, so every madelung * head term is 0)
        assert (div.decode() if isinstance(div, bytes) else div) == "gygi"
        assert len(heads) > 0
        assert all(np.all(np.isfinite(h)) and np.all(h == 0.0) for h in heads)
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


def test_write_model_h5_validation():
    H0 = np.zeros((1, 4, 1, 1), complex)
    kpts = np.zeros((4, 3))
    kpts[:, 0] = np.arange(4) / 4
    with pytest.raises(ValueError):
        write_model_h5("never_written.h5", H0, 1.0, kpts_crys=np.full((4, 3), 0.5))  # not Gamma first
    with pytest.raises(ValueError):
        write_model_h5("never_written.h5", H0, 1.0, kpts_crys=kpts[:3])  # nk mismatch
    with pytest.raises(ValueError):
        write_model_h5("never_written.h5", H0, 1.0, kpts_crys=kpts, fock=np.zeros((1, 3, 1, 1)))
    with pytest.raises(ValueError):
        write_model_h5("never_written.h5", H0, 1.0, kpts_crys=kpts, kpoint_weights=np.full(4, 0.3))
    with pytest.raises(ValueError):
        write_thc_h5("never_written.h5", np.ones((1, 4, 1, 1)), np.ones((4, 2, 2)))  # Np mismatch
    with pytest.raises(ValueError):
        write_thc_h5("never_written.h5", np.ones((1, 4, 2, 1)),
                     np.array([[[1, 1j], [1j, 1]]] * 4))  # not Hermitian
    assert not os.path.exists("never_written.h5")
