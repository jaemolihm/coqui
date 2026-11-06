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

Tests of mean_field.unfold_mf and post_proc.unfold_bz(..., mf_unfold=) on qe_lih223_sym.

The input guards of chkpt::unfold_scf end in MPI_Abort (utils::check), so they are run out of process:
this file is also the runner,  mpirun -np 1 python test_unfold_bz.py <checkpoint prefix>  (exit status 0 = accepted).
"""

from mpi4py import MPI
import os
import shutil
import subprocess
import sys

import h5py
import numpy as np
import pytest

import coqui
from coqui.utils.tests.test_coqui_env import mpi
from coqui.mean_field.tests.test_qe import construct_qe_mf
from coqui._lib.utils_module import utest_filename

UNFOLD_PREFIX = "unfold_mf_out"


def _unfold(mpi_ctx, mf, in_prefix, out_prefix):
    mf_unfold = coqui.make_mf(mpi_ctx, {"prefix": UNFOLD_PREFIX, "outdir": "./"}, "bdft")
    coqui.post_proc.unfold_bz(mf, {"prefix": in_prefix, "output": out_prefix}, mf_unfold=mf_unfold)


def _run_unfold_in_process(in_prefix):
    mpi_ctx = coqui.MpiHandler()
    mf = construct_qe_mf(mpi_ctx, "qe_lih223_sym")
    _unfold(mpi_ctx, mf, in_prefix, in_prefix + "_runner_out")


def _run_unfold_mf_in_process(scratch):
    # unfold_mf on the fixture copy in the directory scratch (see _scratch_fixture)
    mpi_ctx = coqui.MpiHandler()
    mf = coqui.make_mf(mpi_ctx, {"prefix": "pwscf", "outdir": scratch, "filetype": "h5"}, "qe")
    coqui.mean_field.unfold_mf(mf, UNFOLD_PREFIX, "./")
    if mpi_ctx.root():
        print("UNFOLD_OK")


def _scratch_fixture(name, edit):
    """Copy of the qe_lih223_sym h5 fixture (the wavefunctions are linked) in the directory name, with edit(h5py.File)
    applied to the copy."""
    outdir, prefix = utest_filename("qe_lih223_sym")
    os.makedirs(name, exist_ok=True)
    shutil.copy(os.path.join(outdir, prefix + ".coqui.h5"), os.path.join(name, "pwscf.coqui.h5"))
    link = os.path.join(name, "pwscf.save")
    if not os.path.exists(link):
        os.symlink(os.path.abspath(os.path.join(outdir, prefix + ".save")), link)
    with h5py.File(os.path.join(name, "pwscf.coqui.h5"), "r+") as f:
        edit(f)
    return os.path.abspath(name)


def _final_iter_group(f):
    return f["scf/iter" + str(int(f["scf/final_iter"][()]))]


def _runner(in_prefix, mode="unfold"):
    """Returns the (exit status, output) of unfold_bz on the checkpoint in_prefix (mode "unfold"), or of unfold_mf on the
    fixture copy in the directory in_prefix (mode "unfold_mf"), in a separate process."""
    # Strip the launcher's MPI environment so that the child is a fresh job.
    env = {k: v for k, v in os.environ.items() if not k.startswith(("OMPI_", "PMIX_", "PMI_"))}
    res = subprocess.run(["mpirun", "--bind-to", "none", "-np", "1", sys.executable, os.path.abspath(__file__),
                          mode, in_prefix], capture_output=True, text=True, env=env, timeout=600)
    return res.returncode, res.stdout + res.stderr


def test_unfold_mf_and_unfold_bz(mpi):
    # Smoke test of the c2py overloads (mean_field.unfold_mf, post_proc.unfold_bz with mf_unfold) on a GW checkpoint of
    # qe_lih223_sym, and of the input guards of chkpt::unfold_scf against a copy of that checkpoint with a swapped
    # k list (guard on mf_ibz vs the checkpoint) and with a G, Sigma and tau mesh of one point fewer than the grid rebuilt from beta, wmax, prec (read_iaft validates the stored tau mesh against the grid rebuilt from the metadata, which is the
    # guard that keeps G_tskij, sized from the stored tau mesh, consistent with the IAFT written to the output).
    mf = construct_qe_mf(mpi, "qe_lih223_sym")
    thc = coqui.make_thc_coulomb(mf, {"storage": "incore", "thresh": 1e-4, "ecut": mf.ecutrho(),
                                      "chol_block_size": 1, "init": True})
    coqui.run_gw({"restart": False, "output": "unfold_gw", "niter": 1, "beta": 10,
                  "iaft": {"prec": "medium", "basis": "dlr"},
                  "iter_alg": {"alg": "damping", "mixing": 0.7}}, h_int=thc)
    mpi.barrier()

    coqui.mean_field.unfold_mf(mf, UNFOLD_PREFIX, "./")
    mpi.barrier()
    assert os.path.exists(UNFOLD_PREFIX + ".h5")
    _unfold(mpi, mf, "unfold_gw", "unfold_gw_full")
    mpi.barrier()

    files = ["unfold_gw.mbpt.h5", "unfold_gw_full.mbpt.h5"]
    if mpi.root():
        assert os.path.exists("unfold_gw_full.mbpt.h5")
        with h5py.File("unfold_gw.mbpt.h5", "r") as fi, h5py.File("unfold_gw_full.mbpt.h5", "r") as fo:
            assert fo["scf/final_iter"][()] == fi["scf/final_iter"][()]
            assert fo["system/number_of_kpoints_ibz"][()] == fi["system/number_of_kpoints"][()]

        # copies for the guard cases
        for name in ("kswap", "truncG", "control"):
            shutil.copy("unfold_gw.mbpt.h5", f"unfold_{name}.mbpt.h5")
            files.append(f"unfold_{name}.mbpt.h5")
        with h5py.File("unfold_kswap.mbpt.h5", "a") as f:
            k = f["system/kpoints_crys"][...]
            assert abs(k[0] - k[1]).max() > 1e-6
            k[[0, 1]] = k[[1, 0]]
            f["system/kpoints_crys"][...] = k
        with h5py.File("unfold_truncG.mbpt.h5", "a") as f:
            # G_tskij, Sigma_tskij and tau_mesh/fermion all have nt - 1 points, as if written on a different DLR grid
            g = _final_iter_group(f)
            for ds in (g["G_tskij"], g["Sigma_tskij"], f["imaginary_fourier_transform/tau_mesh/fermion"]):
                data, attrs, path = ds[...][:-1], dict(ds.attrs), ds.name
                del f[path]
                new = f.create_dataset(path, data=data)
                for key, val in attrs.items():
                    new.attrs[key] = val

        # control: an unmodified copy is accepted, so the rejections below come from the modification
        rc, out = _runner("unfold_control")
        assert rc == 0, out
        files += ["unfold_control_runner_out.mbpt.h5"]
        rc, out = _runner("unfold_kswap")
        assert rc != 0 and "differs from mf_ibz" in out, out
        rc, out = _runner("unfold_truncG")
        assert rc != 0 and "does not match the grid rebuilt" in out, out
    mpi.barrier()

    if mpi.root():
        for fn in files + [UNFOLD_PREFIX + ".h5"]:
            if os.path.exists(fn):
                os.remove(fn)
        for fn in ("unfold_kswap_runner_out.mbpt.h5", "unfold_truncG_runner_out.mbpt.h5"):
            if os.path.exists(fn):
                os.remove(fn)
    mpi.barrier()


def test_unfold_mf_guards(mpi):
    # Out-of-process (utils::check calls MPI_Abort). unfold_mf on an unmodified copy of qe_lih223_sym is accepted
    # (control), and rejected on a copy of it with the Li projectors 1 and 2 swapped at every k (the angular momentum
    # of the projectors cannot be assigned: projector 1 is an s radial function, projector 2 the first p function), and
    # on a copy with the sign of the first p function (projector 2) flipped at every k: that function is still
    # F(|q|) Y_1m(q) up to a constant sign, so a check of each m on its own accepts it, while the p shell is no longer
    # parallel to (Y_10, Y_11, Y_1-1) at any q.
    def swap_projectors(f):
        g = f["Hamiltonian/ncpp"]
        for ik in range(int(g.attrs["number_of_kpoints"])):
            d = g["projector_k%d" % ik]
            tmp = d[1, ...].copy()
            d[1, ...] = d[2, ...]
            d[2, ...] = tmp

    def flip_projector(f):
        g = f["Hamiltonian/ncpp"]
        for ik in range(int(g.attrs["number_of_kpoints"])):
            g["projector_k%d" % ik][2, ...] = -g["projector_k%d" % ik][2, ...]

    if mpi.root():
        rc, out = _runner(_scratch_fixture("unfold_guard_ok", lambda f: None), mode="unfold_mf")
        assert rc == 0 and "UNFOLD_OK" in out, out
        rc, out = _runner(_scratch_fixture("unfold_guard_l", swap_projectors), mode="unfold_mf")
        assert rc != 0 and "cannot assign an angular momentum" in out, out
        rc, out = _runner(_scratch_fixture("unfold_guard_flip", flip_projector), mode="unfold_mf")
        assert rc != 0 and "cannot assign an angular momentum" in out, out
        for d in ("unfold_guard_ok", "unfold_guard_l", "unfold_guard_flip"):
            shutil.rmtree(d, ignore_errors=True)
        if os.path.exists(UNFOLD_PREFIX + ".h5"):
            os.remove(UNFOLD_PREFIX + ".h5")
    mpi.barrier()


@pytest.mark.parametrize("system", ["qe_lih223_sym", "qe_si333_sym"])
def test_hf_restart_on_unfold(mpi, system):
    # End-to-end gauge gate of the unfolded mean field (orbitals, THC collocation, H0 from the unfolded projectors, unfolded
    # checkpoint): one HF iteration restarted on the unfolded mean field from the unfolded checkpoint against the unfolded result of
    # the same iteration restarted on the IBZ system (both fixtures have time-reversal k; the conjugation of the unfolded
    # projectors at time-reversal k changes F only for qe_si333_sym, whose projectors are complex there). F_skij is compared in
    # L2 norm; the two runs differ by the THC fit of each system (separate fits, fit not run-to-run reproducible),
    # hence a tolerance of the THC fit (measured 2.5e-4 for lih223, 7.6e-4 for si333), not round-off. Without the
    # conjugation of the unfolded projectors at time-reversal k, si333 gives 4.4e-2.
    thc_params = {"storage": "incore", "thresh": 1e-4, "chol_block_size": 1, "init": True}
    hf = {"outdir": "./", "niter": 2, "beta": 10, "conv_thr": 1e-14, "iaft": {"prec": "medium", "basis": "dlr"},
          "iter_alg": {"alg": "damping", "mixing": 0.7}}
    mf = construct_qe_mf(mpi, system)
    thc = coqui.make_thc_coulomb(mf, dict(thc_params, ecut=mf.ecutrho()))
    coqui.run_hf(dict(hf, restart=False, prefix="uh_ibz"), h_int=thc)
    mpi.barrier()
    if mpi.root():
        shutil.copy("uh_ibz.mbpt.h5", "uh_start.mbpt.h5")
    mpi.barrier()
    coqui.mean_field.unfold_mf(mf, UNFOLD_PREFIX, "./")
    mpi.barrier()
    mf_unfold = coqui.make_mf(mpi, {"prefix": UNFOLD_PREFIX, "outdir": "./"}, "bdft")
    # starting checkpoint on the unfolded mean field
    coqui.post_proc.unfold_bz(mf, {"prefix": "uh_start", "output": "uh_unfold"}, mf_unfold=mf_unfold)
    # one more iteration, on the IBZ system and on the unfolded mean field
    coqui.run_hf(dict(hf, restart=True, prefix="uh_ibz", niter=1), h_int=thc)
    thc_unfold = coqui.make_thc_coulomb(mf_unfold, dict(thc_params, ecut=mf_unfold.ecutrho()))
    coqui.run_hf(dict(hf, restart=True, prefix="uh_unfold", niter=1), h_int=thc_unfold)
    mpi.barrier()
    coqui.post_proc.unfold_bz(mf, {"prefix": "uh_ibz", "output": "uh_ibz_full"}, mf_unfold=mf_unfold)
    mpi.barrier()

    err = None
    if mpi.root():
        with h5py.File("uh_ibz_full.mbpt.h5", "r") as fa, h5py.File("uh_unfold.mbpt.h5", "r") as fb:
            assert fa["scf/final_iter"][()] == fb["scf/final_iter"][()] == 3
            Fa = _final_iter_group(fa)["F_skij"][...]
            Fb = _final_iter_group(fb)["F_skij"][...]
            Fa = Fa[..., 0] + 1j * Fa[..., 1] if Fa.dtype != complex and Fa.shape[-1] == 2 else Fa
            Fb = Fb[..., 0] + 1j * Fb[..., 1] if Fb.dtype != complex and Fb.shape[-1] == 2 else Fb
            err = np.linalg.norm(Fa - Fb) / np.linalg.norm(Fb)
            print(f"HF restart on the unfolded mean field vs unfolded IBZ restart: |dF|/|F| = {err:.3e}")
    err = MPI.COMM_WORLD.bcast(err, root=0)
    assert err < 2e-3
    mpi.barrier()
    if mpi.root():
        for fn in ("uh_ibz", "uh_start", "uh_unfold", "uh_ibz_full"):
            if os.path.exists(fn + ".mbpt.h5"):
                os.remove(fn + ".mbpt.h5")
        if os.path.exists(UNFOLD_PREFIX + ".h5"):
            os.remove(UNFOLD_PREFIX + ".h5")
    mpi.barrier()


if __name__ == "__main__":
    if sys.argv[1] == "unfold_mf":
        _run_unfold_mf_in_process(sys.argv[2])
    else:
        _run_unfold_in_process(sys.argv[2])
