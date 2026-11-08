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
import pytest
import numpy as np
import h5py

import coqui
from coqui.utils.tests.test_coqui_env import mpi
from coqui.mean_field.tests.test_qe import construct_qe_mf


def _remove_files(mpi, names):
    if mpi.root():
        for name in names:
            if os.path.exists(name):
                os.remove(name)
    mpi.barrier()


def _scgw(mpi, thc, output, dyson_no_inversion):
    coqui.run_gw({"restart": False, "output": output, "niter": 40, "beta": 100,
                  "iaft": {"prec": "high", "basis": "dlr"}, "conv_thr": 1e-6,
                  "iter_alg": {"alg": "diis", "mixing": 0.7, "diis_warmup": 3},
                  "dyson_no_inversion": dyson_no_inversion}, h_int=thc)
    mpi.barrier()


def _read_complex(f, name):
    raw = f[name][()]
    return raw[..., 0] + 1j * raw[..., 1]


def _lr_dyson_only(mpi, thc, prefix, output, dH0, **kwargs):
    """Non-interacting response (max_iter = 0: the Dyson pass of dH0 on the stored state) at q = Gamma; returns
    (Delta_mu, ΔDm_skij) of the output checkpoint."""
    coqui.mbpt.run_lr({"prefix": prefix, "output": output}, thc, [0.0, 0.0, 0.0], dH0,
                      include_hartree=True, include_exchange=True, gw_mode="none", max_iter=0, **kwargs)
    mpi.barrier()
    with h5py.File(f"{output}.mbpt.h5", "r") as f:
        return f["linear_response/Delta_mu"][()], _read_complex(f, "linear_response/DeltaDm_skij")


def test_lr_dyson_tau_checkpoint(mpi, capfd):
    # run_gw with dyson_no_inversion = false vs true on LiH (THC, DLR, beta = 100): only the Picard-sweep run carries
    # scf/dyson_no_inversion. run_lr names the Dyson kernel it takes from the mark (the imaginary-time convolution on the
    # stored G(tau), Matsubara nodes without the mark), and its non-interacting response ΔDm of a random Hermitian ΔH0
    # on the sweep checkpoint matches the reference -T_beta w_to_tau[G(iw) ΔH0 G(iw)] of the stored G(tau) to 1e-9
    # (7.8e-11 measured; the two kernels differ at the DLR-grid floor, 6.3e-7 with prec = medium, hence prec = high). It
    # checks the plumbing: the stored G is read, the kernel is applied, the result is written. With fix_density the shifted
    # ΔDm has ΔN = 0 to the DLR floor (1.0e-8 against ‖ΔDm‖ = 13.8 measured: Δμ zeroes the node ΔN, not the τ trace). The
    # real backend (THC, C++ run_gw and lr_dyson) runs in every step; nothing is stubbed.
    mf = construct_qe_mf(mpi, "qe_lih222")
    thc = coqui.make_thc_coulomb(mf, {"storage": "incore", "thresh": 1e-4, "ecut": mf.ecutrho(),
                                      "chol_block_size": 1, "init": True, "save": "lrtau_thc.h5"})
    _scgw(mpi, thc, "lrtau_node", False)
    _scgw(mpi, thc, "lrtau_picard", True)

    ns, nk, nb = mf.nspin(), mf.nkpts_ibz(), mf.nbnd()
    if mpi.root():
        for name, marked in (("lrtau_node", False), ("lrtau_picard", True)):
            with h5py.File(f"{name}.mbpt.h5", "r") as f:
                assert ("dyson_no_inversion" in f["scf"]) == marked, name
        # run_lr appends its result to an existing checkpoint: one copy of the ground state per run
        for name in ("lrtau_picard_lr", "lrtau_picard_fd", "lrtau_node_lr"):
            shutil.copy("lrtau_node.mbpt.h5" if "node" in name else "lrtau_picard.mbpt.h5", f"{name}.mbpt.h5")
    mpi.barrier()

    rng = np.random.default_rng(7)
    dH0 = rng.standard_normal((1, nk, nb, nb)) + 1j * rng.standard_normal((1, nk, nb, nb))
    dH0 = 0.5 * (dH0 + dH0.conj().transpose(0, 1, 3, 2))

    coqui.set_verbosity(mpi, output_level=2)
    capfd.readouterr()
    _, dDm_tau = _lr_dyson_only(mpi, thc, "lrtau_picard", "lrtau_picard_lr", dH0)
    out_tau = capfd.readouterr().out
    _lr_dyson_only(mpi, thc, "lrtau_node", "lrtau_node_lr", dH0)
    out_node = capfd.readouterr().out
    dmu, dDm_fd = _lr_dyson_only(mpi, thc, "lrtau_picard", "lrtau_picard_fd", dH0, fix_density=True)
    coqui.set_verbosity(mpi, output_level=0)

    if mpi.root():
        assert "LR Dyson kernel: imaginary-time convolution (scf/dyson_no_inversion)" in out_tau
        assert "LR Dyson kernel: Matsubara nodes" in out_node

        # Reference: the Matsubara-node product at the stored G(tau)
        from coqui.utils.imag_axes_ft import IAFT
        iaft = IAFT.from_coqui_chkpt("lrtau_picard.mbpt.h5", verbose=False)
        with h5py.File("lrtau_picard.mbpt.h5", "r") as f:
            G_tskij = _read_complex(f, f"scf/iter{f['scf/final_iter'][()]}/G_tskij")
        G_w = iaft.tau_to_w(G_tskij, "f")
        dG_w = G_w @ dH0[None] @ G_w
        dG_t = iaft.w_to_tau(dG_w, "f")
        dDm_ref = -np.einsum("t,t...->...", np.asarray(iaft._impl.T_beta_t_ff).reshape(-1), dG_t)
        err = np.linalg.norm(dDm_tau - dDm_ref) / np.linalg.norm(dDm_ref)
        print("rel. difference of ΔDm, tau kernel vs node product at the stored G:", err)
        assert np.linalg.norm(dDm_ref) > 1e-6
        assert err < 1e-9

        # ΔN = spin Σ_k w_k Tr[ΔDm] of the density-fixed response
        dN = 2.0 * np.trace(dDm_fd, axis1=-2, axis2=-1).sum().real / nk
        print("fix_density: Delta_mu =", dmu, "; ΔN =", dN, "; ‖ΔDm‖ =", np.linalg.norm(dDm_fd))
        assert abs(dmu) > 1e-6
        assert abs(dN) < 1e-8 * np.linalg.norm(dDm_fd)
    mpi.barrier()

    # Out of process (utils::check aborts via MPI_Abort): dyson_no_inversion in the run_lr input, and an input_iter other
    # than the final iteration, are rejected on a marked checkpoint
    msg = None
    if mpi.root():
        env = {k: v for k, v in os.environ.items() if not k.startswith(("OMPI_", "PMIX_", "PMI_"))}
        for extra, expected in (('"dyson_no_inversion": True', "run_lr takes the Dyson kernel from the checkpoint"),
                                ('"input_iter": 1', "input_iter must be the final iteration")):
            runner = textwrap.dedent(f"""
                from mpi4py import MPI  # noqa: F401 - MPI init before coqui
                import numpy as np, coqui
                from coqui.mean_field.tests.test_qe import construct_qe_mf
                mpi = coqui.MpiHandler()
                mf = construct_qe_mf(mpi, "qe_lih222")
                thc = coqui.make_thc_coulomb(mf, {{"storage": "incore", "thresh": 1e-4, "ecut": mf.ecutrho(),
                                                  "chol_block_size": 1, "init": True, "save": "lrtau_thc.h5"}})
                dH0 = np.zeros(({1}, {nk}, {nb}, {nb}), dtype=complex)
                coqui.mbpt.run_lr({{"prefix": "lrtau_picard", "output": "lrtau_picard_lr", {extra}}},
                                  thc, [0.0, 0.0, 0.0], dH0, gw_mode="none")
                print("GUARD_NOT_TRIGGERED")
            """)
            proc = subprocess.run(["mpirun", "-np", "1", sys.executable, "-c", runner], env=env,
                                  capture_output=True, text=True, timeout=600)
            out = proc.stdout + proc.stderr
            if proc.returncode == 0 or "GUARD_NOT_TRIGGERED" in out:
                msg = f"{extra} was accepted:\n{out[-2000:]}"
            elif expected not in out:
                msg = f"{extra} was rejected, but not by the expected guard ({expected}):\n{out[-2000:]}"
            if msg:
                break
    msg = MPI.COMM_WORLD.bcast(msg, root=0)
    assert msg is None, msg
    mpi.barrier()

    _remove_files(mpi, [f"{n}.mbpt.h5" for n in ("lrtau_node", "lrtau_picard", "lrtau_picard_lr", "lrtau_picard_fd",
                                                 "lrtau_node_lr")] + ["lrtau_thc.h5"])
