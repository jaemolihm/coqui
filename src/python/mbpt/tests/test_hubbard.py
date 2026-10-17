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
import subprocess
import sys
import textwrap

import h5py
import numpy as np
import pytest

import coqui
from coqui.mbpt import run_lr
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


def test_hubbard_scgw_fixed_mu(mpi):
    """scGW restarts at fixed mu = mu0 +- dmu (input "mu"): every iteration keeps mu, and particle-hole symmetry
    of the half-filled t' = 0 Hubbard model gives n(mu0 + dmu) - 1 = 1 - n(mu0 - dmu)."""
    N, t, U, beta, dmu = 4, 0.1, 0.4, 50.0, 0.01
    nk = N * N
    kpts, ek = square_lattice(N, t, 0.0)
    prefix, thc_file = "hubbard_fixed_mu", "hubbard_fixed_mu_thc.h5"
    write_model_h5(f"./{prefix}.h5", ek.reshape(1, nk, 1, 1).astype(complex), 1.0, kpts_crys=kpts,
                   latt=np.diag([1.0, 1.0, 50.0]), mpi=mpi)
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

    # the checkpoint records the fixed mu, the one of the searched-mu run does not
    if mpi.root():
        with h5py.File(f"{prefix}_mu+1.mbpt.h5", "r") as f:
            assert abs(f["scf/fixed_mu"][()] - (mu0 + dmu)) < 1e-14
        with h5py.File(f"{prefix}.mbpt.h5", "r") as f:
            assert "fixed_mu" not in f["scf"]
    mpi.barrier()

    # run_lr: fix_density = True on a fixed-mu ground state aborts, fix_density = False runs
    q_vec = [0.0, 0.0, 0.0]
    dH0 = np.zeros((1, nk, 1, 1), dtype=complex)
    dH0[0, :, 0, 0] = 1.0
    lr_params = {"outdir": "./", "prefix": f"{prefix}_mu+1", "max_iter": 5, "tol": 1e-8}
    run_lr(dict(lr_params), thc, q_vec, dH0, fix_density=False)
    mpi.barrier()
    msg = None
    if mpi.root():
        runner = textwrap.dedent(f"""
            from mpi4py import MPI  # noqa: F401 - MPI init before coqui
            import numpy as np, coqui
            from coqui.mbpt import run_lr
            mpi = coqui.MpiHandler()
            mf = coqui.make_mf(mpi, {{"prefix": "{prefix}", "outdir": "./"}}, "model")
            thc = coqui.make_thc_coulomb(mf, {{"save": "{thc_file}", "storage": "incore"}})
            dH0 = np.ones(({1}, {nk}, 1, 1), dtype=complex)
            run_lr({{"outdir": "./", "prefix": "{prefix}_mu+1", "max_iter": 5, "tol": 1e-8}},
                         thc, [0.0, 0.0, 0.0], dH0, fix_density=True)
            print("GUARD_NOT_TRIGGERED")
        """)
        # utils::check aborts via MPI_Abort instead of raising, so the call runs in its own process
        env = {k: v for k, v in os.environ.items() if not k.startswith(("OMPI_", "PMIX_", "PMI_"))}
        proc = subprocess.run(["mpirun", "-np", "1", sys.executable, "-c", runner], env=env,
                              capture_output=True, text=True, timeout=600)
        out = proc.stdout + proc.stderr
        if proc.returncode == 0 or "GUARD_NOT_TRIGGERED" in out:
            msg = f"fix_density = True on a fixed-mu checkpoint was accepted:\n{out[-2000:]}"
        elif "fix_density = true is not meaningful" not in out:
            msg = f"rejected, but not by the fixed-mu guard:\n{out[-2000:]}"
    msg = MPI.COMM_WORLD.bcast(msg, root=0)
    assert msg is None, msg
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


# LR on the t' != 0, n = 0.85 Hubbard model (particle-hole symmetry broken, so the GW kernel carries a
# non-zero DeltaW): per (q, kernel), DeltaN = 2 sum_k w_k DeltaDm_k (q = 0 only) and ||DeltaDm||, generated
# at np = 1. The LR result does not depend on the MPI distribution, so every rank count reproduces them;
# one-orbital (Np = 1, nbnd = 1) arrays have empty local blocks on most ranks at np > 1.
# These multi-rank tests are not run by ctest; run them by hand under mpirun at np 1, 4, 8 and 16 (np 16 is the
# count that reached the empty-block path of lr_gw's cached aux_to_primary buffer).
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
                   latt=np.diag([1.0, 1.0, 50.0]), mpi=mpi)
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
