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
import pytest
import numpy as np

import coqui
from coqui.utils.tests.test_coqui_env import mpi
from coqui.mean_field.tests.test_qe import construct_qe_mf


def test_gw_thc(mpi):
    mf = construct_qe_mf(mpi, "qe_lih222_sym")
    eri_params = {
        "storage": "incore",
        "thresh": 1e-4,
        "ecut": mf.ecutrho(),
        "chol_block_size": 1,
        "init": True
    }
    thc = coqui.make_thc_coulomb(mf, eri_params)

    gw_params = {
        "restart": False, 
        "output": "gw", 
        "niter": 1,
        "beta": 100, 
        "iaft": {
            "prec": "medium", 
            "basis": "dlr"
        },
        "iter_alg": {"alg": "damping", "mixing": 0.7}
    }
    coqui.run_gw(gw_params, h_int=thc)
    mpi.barrier()


def test_gw_mix_thc_chol(mpi):
    mf = construct_qe_mf(mpi, "qe_lih222")
    thc_params = {
        "storage": "incore",
        "thresh": 1e-4,
        "ecut": mf.ecutrho(),
        "chol_block_size": 1,
        "init": True
    }
    thc = coqui.make_thc_coulomb(mf, thc_params)

    gw_params = {
        "restart": False, 
        "output": "gw", 
        "niter": 1,
        "beta": 100, 
        "iaft": {
            "prec": "medium", 
            "basis": "dlr"
        },
        "iter_alg": {
            "alg": "damping", 
            "mixing": 0.7
        }
    }
    coqui.run_gw(gw_params, h_int=thc)
    mpi.barrier()

    # mix thc/cholesky eri for J/K terms
    chol_params = {
        "tol": 1e-3,
        "storage": "outcore",
        "path": "./",
        "chol_block_size": 32
    }
    chol = coqui.make_chol_coulomb(mf, chol_params)
    coqui.run_gw(gw_params, h_int=thc, h_int_hf=chol)
    mpi.barrier()

    if mpi.root():
        os.remove("./chol_info.h5")
        for iq in range(mf.nqpts_ibz()):
            os.remove(f"./Vq{iq}.h5")
    mpi.barrier()


def test_g0w0_thc(mpi):
    mf = construct_qe_mf(mpi, "qe_lih222_sym")
    eri_params = {
        "storage": "incore",
        "thresh": 1e-4,
        "ecut": mf.ecutrho(),
        "chol_block_size": 1,
        "init": True
    }
    thc = coqui.make_thc_coulomb(mf, eri_params)

    gw_params = {
        "restart": False, 
        "output": "gw", 
        "niter": 1,
        "beta": 100, 
        "iaft": {
            "prec": "medium",
            "basis": "dlr"
        },
        "qp_type": "sc", 
        "eta": 1e-6, 
    }
    coqui.run_evgw(gw_params, h_int=thc)
    mpi.barrier()



def _scgw_slim(mpi, thc, output, slim, iter_alg, niter, restart=False):
    gw_params = {
        "restart": restart,
        "output": output,
        "niter": niter,
        "beta": 100,
        "iaft": {"prec": "medium", "basis": "dlr"},
        "conv_thr": 1e-14,
        "chkpt_slim": slim,
        "iter_alg": iter_alg,
    }
    coqui.run_gw(gw_params, h_int=thc)
    mpi.barrier()


def _final_state(output):
    import h5py
    with h5py.File(f"{output}.mbpt.h5", "r") as f:
        it = f["scf/final_iter"][()]
        g = f[f"scf/iter{it}"]
        return it, g["F_skij"][()], g["Sigma_tskij"][()], g["G_tskij"][()]


def _remove_outputs(mpi, outputs):
    if mpi.root():
        for name in outputs:
            os.remove(f"{name}.mbpt.h5")
    mpi.barrier()


@pytest.mark.parametrize("iter_alg", [
    {"alg": "damping", "mixing": 0.7},
    {"alg": "diis", "mixing": 0.7, "diis_warmup": 3},
], ids=["damping", "diis"])
def test_gw_chkpt_slim_mixing(mpi, iter_alg):
    # scGW, 3 iterations, chkpt_slim = true vs false: F, Sigma and G of the final
    # iteration agree to round-off. Iterations 2 and 3 damp (DIIS: warmup), so a
    # mixing with the omitted Sigma read as zero fails this (measured: F rel. error
    # 3.7e-3 against the 1e-12 bound). 3 iterations with diis_warmup = 3 cover only
    # the warmup damping.
    mf = construct_qe_mf(mpi, "qe_lih222_sym")
    eri_params = {"storage": "incore", "thresh": 1e-4, "ecut": mf.ecutrho(),
                  "chol_block_size": 1, "init": True}
    thc = coqui.make_thc_coulomb(mf, eri_params)

    _scgw_slim(mpi, thc, "gw_full", False, iter_alg, 3)
    _scgw_slim(mpi, thc, "gw_slim", True, iter_alg, 3)

    if mpi.root():
        it_r, F_r, S_r, G_r = _final_state("gw_full")
        it_s, F_s, S_s, G_s = _final_state("gw_slim")
        assert it_r == it_s == 3
        assert np.linalg.norm(F_s - F_r) / np.linalg.norm(F_r) < 1e-12
        assert np.linalg.norm(S_s - S_r) / np.linalg.norm(S_r) < 1e-12
        assert np.linalg.norm(G_s - G_r) / np.linalg.norm(G_r) < 1e-12
    mpi.barrier()
    _remove_outputs(mpi, ["gw_full", "gw_slim"])


# Restart floor measured on this system: 2.3e-14 (F), 1.8e-13 (Sigma), 2.1e-12 (G).
RESTART_TOL = 1e-10


def test_gw_chkpt_slim_damping_restart(mpi):
    # scGW with damping, 2 iterations + a restart for 1 more:
    # (a) chkpt_slim = true vs false: final F, Sigma and G agree to round-off (both
    #     arms take the same restart path);
    # (b) vs 3 iterations in one run: the restarted mixing stages the previous
    #     state from the checkpoint, so a wrong staged state changes iteration 3
    #     at O(|Sigma|). The tolerance is the restart floor: the restart's Dyson
    #     solve searches mu again from the stored mu (mu_tolerance 1e-9).
    mf = construct_qe_mf(mpi, "qe_lih222_sym")
    eri_params = {"storage": "incore", "thresh": 1e-4, "ecut": mf.ecutrho(),
                  "chol_block_size": 1, "init": True}
    thc = coqui.make_thc_coulomb(mf, eri_params)
    iter_alg = {"alg": "damping", "mixing": 0.7}

    _scgw_slim(mpi, thc, "gw_once", False, iter_alg, 3)
    _scgw_slim(mpi, thc, "gw_full", False, iter_alg, 2)
    _scgw_slim(mpi, thc, "gw_full", False, iter_alg, 1, restart=True)
    _scgw_slim(mpi, thc, "gw_slim", True, iter_alg, 2)
    _scgw_slim(mpi, thc, "gw_slim", True, iter_alg, 1, restart=True)

    if mpi.root():
        it_r, F_r, S_r, G_r = _final_state("gw_full")
        it_s, F_s, S_s, G_s = _final_state("gw_slim")
        assert it_r == it_s == 3
        assert np.linalg.norm(F_s - F_r) / np.linalg.norm(F_r) < 1e-12
        assert np.linalg.norm(S_s - S_r) / np.linalg.norm(S_r) < 1e-12
        assert np.linalg.norm(G_s - G_r) / np.linalg.norm(G_r) < 1e-12
        it_o, F_o, S_o, G_o = _final_state("gw_once")
        assert it_o == 3
        errs = [np.linalg.norm(X_s - X_o) / np.linalg.norm(X_o)
                for X_s, X_o in ((F_s, F_o), (S_s, S_o), (G_s, G_o))]
        print("restart vs one run, rel. error F / Sigma / G:", errs)
        assert max(errs) < RESTART_TOL
    mpi.barrier()
    _remove_outputs(mpi, ["gw_once", "gw_full", "gw_slim"])


_STORAGE_KEY_RUNNER = """
import sys
from mpi4py import MPI
import coqui
from coqui.mean_field.tests.test_qe import construct_qe_mf
mpi = coqui.MpiHandler()
mf = construct_qe_mf(mpi, "qe_lih222_sym")
thc = coqui.make_thc_coulomb(mf, {"storage": "incore", "thresh": 1e-4, "ecut": mf.ecutrho(),
                                  "chol_block_size": 1, "init": True})
coqui.run_gw({"restart": False, "output": "gw_storage_key", "niter": 2, "beta": 100,
              "iaft": {"prec": "medium", "basis": "dlr"},
              "iter_alg": {"alg": sys.argv[1], sys.argv[2]: sys.argv[3]}},
             h_int=thc)
print("KEY_ACCEPTED", flush=True)
"""


@pytest.mark.parametrize("alg", ["diis", "damping"])
@pytest.mark.parametrize("key", ["storage", "diis_storage"])
def test_iter_alg_storage_key_rejected(mpi, alg, key):
    # iter_alg (DIIS or damping) with the removed key "storage" or "diis_storage" (DIIS history is always in
    # memory) vs the input parsing: the run aborts with a message naming the key that was written.
    # utils::check aborts MPI, so the run is its own mpirun -np 1 job, spawned by rank 0.
    import subprocess
    import sys
    msg = None
    if mpi.root():
        env = {k: v for k, v in os.environ.items()
               if not k.startswith(("OMPI_", "PMIX_", "PMI_"))}
        proc = subprocess.run(["mpirun", "--bind-to", "none", "-np", "1",
                               sys.executable, "-c", _STORAGE_KEY_RUNNER, alg, key, "memory"],
                              env=env, capture_output=True, text=True, timeout=900)
        out = proc.stdout + proc.stderr
        expected = f'the key "{key}" is not recognized'
        if proc.returncode == 0 or "KEY_ACCEPTED" in out:
            msg = f"the key {key} was accepted:\n{out[-2000:]}"
        elif expected not in out:
            msg = f"run failed, but not by the key check:\n{out[-2000:]}"
        if os.path.exists("gw_storage_key.mbpt.h5"):
            os.remove("gw_storage_key.mbpt.h5")
    msg = MPI.COMM_WORLD.bcast(msg, root=0)
    assert msg is None, msg


def test_evgw_diis_ignores_stale_files(mpi):
    # evGW (residual_type "vector_diff"), DIIS with 2 warmup steps, 6 iterations in a clean directory vs in a
    # directory that holds a stale diis_heff_vectors.h5 (8 vectors of the right shape, nonzero): the final
    # Heff is identical, and the planted file is not touched. The DIIS history is in memory.
    import hashlib
    import shutil
    import h5py
    mf = construct_qe_mf(mpi, "qe_lih222_sym")
    thc = coqui.make_thc_coulomb(mf, {"storage": "incore", "thresh": 1e-4, "ecut": mf.ecutrho(),
                                      "chol_block_size": 1, "init": True})
    params = {"restart": False, "output": "evgw", "niter": 6, "beta": 100,
              "iaft": {"prec": "medium", "basis": "dlr"}, "conv_thr": 1e-14,
              "iter_alg": {"alg": "diis", "mixing": 0.7, "diis_warmup": 2}}
    start = os.getcwd()
    heff = {}
    stale_hash = {}
    for tag in ("clean", "stale"):
        if mpi.root():
            os.makedirs(tag, exist_ok=True)
        mpi.barrier()
        os.chdir(os.path.join(start, tag))
        if tag == "stale" and mpi.root():
            with h5py.File("../clean/evgw.mbpt.h5", "r") as f:
                shape = f["scf/iter0/Heff_skij"].shape
            with h5py.File("diis_heff_vectors.h5", "w") as f:
                for i in range(8):
                    f.create_group(f"vec{i}").create_dataset("Heff_skij", data=np.full(shape, 0.5 + 0.1j * i))
            with open("diis_heff_vectors.h5", "rb") as f:
                stale_hash[tag] = hashlib.md5(f.read()).hexdigest()
        mpi.barrier()
        coqui.run_evgw(params, h_int=thc)
        mpi.barrier()
        if mpi.root():
            with h5py.File("evgw.mbpt.h5", "r") as f:
                heff[tag] = f[f"scf/iter{f['scf/final_iter'][()]}/Heff_skij"][()]
        os.chdir(start)
    msg = None
    if mpi.root():
        if not np.array_equal(heff["clean"], heff["stale"]):
            msg = "final Heff differs between the clean and the stale-file run"
        with open(os.path.join(start, "stale", "diis_heff_vectors.h5"), "rb") as f:
            if hashlib.md5(f.read()).hexdigest() != stale_hash["stale"]:
                msg = "the planted diis_heff_vectors.h5 was modified"
        if os.path.exists(os.path.join(start, "clean", "diis_heff_vectors.h5")):
            msg = "a diis_heff_vectors.h5 was written"
    msg = MPI.COMM_WORLD.bcast(msg, root=0)
    if mpi.root():
        for tag in ("clean", "stale"):
            shutil.rmtree(os.path.join(start, tag))
    mpi.barrier()
    assert msg is None, msg
