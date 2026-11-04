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
        for name in ("diis_vectors.h5", "diis_residuals.h5"):
            if os.path.exists(name):
                os.remove(name)
    mpi.barrier()


@pytest.mark.parametrize("iter_alg", [
    {"alg": "damping", "mixing": 0.7},
    {"alg": "diis", "mixing": 0.7, "diis_warmup": 3},
], ids=["damping", "diis_slim_default"])
def test_gw_chkpt_slim_mixing(mpi, iter_alg):
    # scGW, 3 iterations, chkpt_slim = true vs false: F, Sigma and G of the final
    # iteration agree to round-off. Iterations 2 and 3 damp (DIIS: warmup), so a
    # mixing with the omitted Sigma read as zero fails this (measured: F rel. error
    # 3.7e-3 against the 1e-12 bound). For DIIS the slim run switches the default
    # diis_storage to "memory", so the reference is memory DIIS; 3 iterations with
    # diis_warmup = 3 cover only the warmup damping (the memory-DIIS extrapolation
    # never reads the checkpoint).
    mf = construct_qe_mf(mpi, "qe_lih222_sym")
    eri_params = {"storage": "incore", "thresh": 1e-4, "ecut": mf.ecutrho(),
                  "chol_block_size": 1, "init": True}
    thc = coqui.make_thc_coulomb(mf, eri_params)

    ref_alg = dict(iter_alg, diis_storage="memory") if iter_alg["alg"] == "diis" else iter_alg
    _scgw_slim(mpi, thc, "gw_full", False, ref_alg, 3)
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


_SLIM_GUARD_RUNNER = """
import sys
from mpi4py import MPI
import coqui
from coqui.mean_field.tests.test_qe import construct_qe_mf
mpi = coqui.MpiHandler()
mf = construct_qe_mf(mpi, "qe_lih222_sym")
thc = coqui.make_thc_coulomb(mf, {"storage": "incore", "thresh": 1e-4, "ecut": mf.ecutrho(),
                                  "chol_block_size": 1, "init": True})
coqui.run_gw({"restart": False, "output": "gw_slim_guard", "niter": 2, "beta": 100,
              "iaft": {"prec": "medium", "basis": "dlr"}, "chkpt_slim": True,
              "iter_alg": {"damping": {"alg": "damping", "mixing": 0.7},
                           "disk_diis": {"alg": "diis", "diis_storage": "disk"},
                           "storage_key": {"alg": "diis", "storage": "memory"}}[sys.argv[1]]},
             h_int=thc)
print("GUARD_NOT_TRIGGERED", flush=True)
"""


@pytest.mark.parametrize("case, expected", [
    ("damping", "omitted from the checkpoint by chkpt_slim"),
    ("disk_diis", "does not work with chkpt_slim = true"),
    ("storage_key", 'set by "diis_storage"'),
])
def test_gw_chkpt_slim_guard(mpi, case, expected):
    # Slim scGW runs that would mix from the omitted Sigma_tskij vs the chkpt_slim
    # guards: "damping" is the checkpoint-reading damping (forced by
    # COQUI_DEBUG_SERIAL_DIIS), which aborts at the read of iteration 1;
    # "disk_diis" is an explicit diis_storage = "disk", rejected at input parsing;
    # "storage_key" is the removed key "storage", rejected at input parsing with a
    # message naming "diis_storage".
    # utils::check aborts MPI, so the run is its own mpirun -np 1 job, spawned by
    # rank 0.
    import subprocess
    import sys
    msg = None
    if mpi.root():
        env = {k: v for k, v in os.environ.items()
               if not k.startswith(("OMPI_", "PMIX_", "PMI_"))}
        env["COQUI_DEBUG_SERIAL_DIIS"] = "1"
        proc = subprocess.run(["mpirun", "--bind-to", "none", "-np", "1",
                               sys.executable, "-c", _SLIM_GUARD_RUNNER, case],
                              env=env, capture_output=True, text=True, timeout=900)
        out = proc.stdout + proc.stderr
        if proc.returncode == 0 or "GUARD_NOT_TRIGGERED" in out:
            msg = f"slim checkpoint read was accepted:\n{out[-2000:]}"
        elif expected not in out:
            msg = f"run failed, but not by the chkpt_slim guard:\n{out[-2000:]}"
        if os.path.exists("gw_slim_guard.mbpt.h5"):
            os.remove("gw_slim_guard.mbpt.h5")
    msg = MPI.COMM_WORLD.bcast(msg, root=0)
    assert msg is None, msg
