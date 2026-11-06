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

Tests of post_proc.unfold_checkpoint on qe_lih223_sym and qe_si333_sym.

The input guards of mf::unfold_mf and chkpt::unfold_scf end in MPI_Abort (utils::check), so they are run out of
process: this file is also the runner,  mpirun -np 1 python test_unfold_bz.py <checkpoint prefix> [mean-field dir]
(exit status 0 = accepted).
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


def _remove_outputs(prefixes):
    """Removes the unfolded mean field (.h5) and the checkpoint (.mbpt.h5) of each prefix."""
    for p in prefixes:
        for fn in (p + ".h5", p + ".mbpt.h5"):
            if os.path.exists(fn):
                os.remove(fn)


def _run_unfold_in_process(in_prefix, mf_dir=None):
    # unfold_checkpoint of the checkpoint in_prefix, on qe_lih223_sym or on the fixture copy in mf_dir
    mpi_ctx = coqui.MpiHandler()
    if mf_dir is None:
        mf = construct_qe_mf(mpi_ctx, "qe_lih223_sym")
    else:
        mf = coqui.make_mf(mpi_ctx, {"prefix": "pwscf", "outdir": mf_dir, "filetype": "h5"}, "qe")
    coqui.post_proc.unfold_checkpoint(mf, {"prefix": in_prefix, "output": in_prefix + "_runner_out"})


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


def _runner(in_prefix, mf_dir=None):
    """Returns the (exit status, output) of unfold_checkpoint on the checkpoint in_prefix, with the mean field of
    qe_lih223_sym or of the fixture copy in mf_dir, in a separate process."""
    # Strip the launcher's MPI environment so that the child is a fresh job.
    env = {k: v for k, v in os.environ.items() if not k.startswith(("OMPI_", "PMIX_", "PMI_"))}
    args = [in_prefix] + ([mf_dir] if mf_dir is not None else [])
    res = subprocess.run(["mpirun", "--bind-to", "none", "-np", "1", sys.executable, os.path.abspath(__file__)] + args,
                         capture_output=True, text=True, env=env, timeout=600)
    return res.returncode, res.stdout + res.stderr


def test_unfold_checkpoint(mpi):
    # Smoke test of post_proc.unfold_checkpoint on a GW checkpoint of qe_lih223_sym (both files written, nkpts_ibz of
    # the output = nkpts of the input), and of its input guards, out of process: an unmodified copy of the checkpoint
    # is accepted (control); rejected are a copy with a swapped k list (guard on mf_ibz vs the checkpoint), a copy with a
    # G, Sigma and tau mesh of one point fewer than the grid rebuilt from beta, wmax, prec (read_iaft validates the
    # stored tau mesh against the rebuilt grid, which keeps G_tskij, sized from the stored tau mesh, consistent with the
    # IAFT written to the output), a missing input checkpoint (rejected before the unfolded mean field is written), and a mean field without the group Hamiltonian/ncpp/fullbz (message asks for
    # add_pp_fullbz; its control is the unmodified fixture copy).
    mf = construct_qe_mf(mpi, "qe_lih223_sym")
    thc = coqui.make_thc_coulomb(mf, {"storage": "incore", "thresh": 1e-4, "ecut": mf.ecutrho(),
                                      "chol_block_size": 1, "init": True})
    coqui.run_gw({"restart": False, "output": "unfold_gw", "niter": 1, "beta": 10,
                  "iaft": {"prec": "medium", "basis": "dlr"},
                  "iter_alg": {"alg": "damping", "mixing": 0.7}}, h_int=thc)
    mpi.barrier()

    # unfolded mean field and full-BZ checkpoint
    coqui.post_proc.unfold_checkpoint(mf, {"prefix": "unfold_gw", "output": "unfold_gw_full"})
    mpi.barrier()

    names = ("kswap", "truncG", "control")
    if mpi.root():
        assert os.path.exists("unfold_gw_full.h5") and os.path.exists("unfold_gw_full.mbpt.h5")
        with h5py.File("unfold_gw.mbpt.h5", "r") as fi, h5py.File("unfold_gw_full.mbpt.h5", "r") as fo:
            assert fo["scf/final_iter"][()] == fi["scf/final_iter"][()]
            assert fo["system/number_of_kpoints_ibz"][()] == fi["system/number_of_kpoints"][()]

        # copies of the checkpoint for the guard cases
        for name in names:
            shutil.copy("unfold_gw.mbpt.h5", f"unfold_{name}.mbpt.h5")
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

        # guards of unfold_scf: the control is accepted, so the rejections below come from the modification
        rc, out = _runner("unfold_control")
        assert rc == 0, out
        rc, out = _runner("unfold_kswap")
        assert rc != 0 and "different k list or order" in out, out
        rc, out = _runner("unfold_truncG")
        assert rc != 0 and "does not match the grid rebuilt" in out, out
        # input checks of unfold_checkpoint, before anything is written: missing input checkpoint
        rc, out = _runner("unfold_missing")
        assert rc != 0 and "input checkpoint" in out and "not found" in out, out
        assert not os.path.exists("unfold_missing_runner_out.h5")

        # guard of unfold_mf on the full-BZ projectors, with the unmodified fixture copy as its control
        def drop_fullbz(f):
            del f["Hamiltonian/ncpp/fullbz"]
        rc, out = _runner("unfold_control", _scratch_fixture("unfold_guard_ok", lambda f: None))
        assert rc == 0, out
        rc, out = _runner("unfold_control", _scratch_fixture("unfold_guard_nofullbz", drop_fullbz))
        assert rc != 0 and "has no full-BZ projectors" in out and "add_pp_fullbz" in out, out

        for d in ("unfold_guard_ok", "unfold_guard_nofullbz"):
            shutil.rmtree(d, ignore_errors=True)
        _remove_outputs(["unfold_gw", "unfold_gw_full"] + [f"unfold_{n}" for n in names]
                        + [f"unfold_{n}_runner_out" for n in names])
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
    # unfolded mean field and starting checkpoint on it
    coqui.post_proc.unfold_checkpoint(mf, {"prefix": "uh_start", "output": "uh_unfold"})
    mpi.barrier()
    mf_unfold = coqui.make_mf(mpi, {"prefix": "uh_unfold", "outdir": "./"}, "bdft")
    # one more iteration, on the IBZ system and on the unfolded mean field
    coqui.run_hf(dict(hf, restart=True, prefix="uh_ibz", niter=1), h_int=thc)
    thc_unfold = coqui.make_thc_coulomb(mf_unfold, dict(thc_params, ecut=mf_unfold.ecutrho()))
    coqui.run_hf(dict(hf, restart=True, prefix="uh_unfold", niter=1), h_int=thc_unfold)
    mpi.barrier()
    coqui.post_proc.unfold_checkpoint(mf, {"prefix": "uh_ibz", "output": "uh_ibz_full"})
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
        _remove_outputs(["uh_ibz", "uh_start", "uh_unfold", "uh_ibz_full"])
    mpi.barrier()


if __name__ == "__main__":
    _run_unfold_in_process(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
