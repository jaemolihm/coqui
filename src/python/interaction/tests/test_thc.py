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

import coqui
from coqui.utils.tests.test_coqui_env import mpi
from coqui.mean_field.tests.test_qe import construct_qe_mf


def test_thc_eri(mpi):
    mf = construct_qe_mf(mpi, "qe_lih222_sym")
    eri_params = {
        "storage": "incore",
        "nIpts": mf.nbnd() * 10,
        "thresh": 1e-10,
        "ecut": mf.ecutrho(),
        "chol_block_size": 1,
        "init": True
    }
    eri = coqui.make_thc_coulomb(mf, eri_params)
    assert isinstance(eri, coqui._lib.eri_module.ThcCoulomb)
    assert mf.mpi() == eri.mpi()


def test_thc_restart(mpi):
    mf = construct_qe_mf(mpi, "qe_lih222_sym")
    eri_params = {
        "save": "thc.eri.h5",
        "storage": "incore",
        "nIpts": mf.nbnd() * 10,
        "thresh": 1e-10,
        "ecut": mf.ecutrho(),
        "chol_block_size": 1,
        "init": True
    }
    eri = coqui.make_thc_coulomb(mf, eri_params)
    assert isinstance(eri, coqui._lib.eri_module.ThcCoulomb)
    eri_restart = coqui.make_thc_coulomb(mf, eri_params)
    assert isinstance(eri_restart, coqui._lib.eri_module.ThcCoulomb)

    assert eri.mpi() == eri_restart.mpi()
    assert eri.mf() == eri_restart.mf()
    mpi.barrier()

    if mpi.root():
        os.remove("thc.eri.h5")
    mpi.barrier()


def test_thc_given_pivots(mpi):
    import numpy as np
    import h5py

    mf = construct_qe_mf(mpi, "qe_lih222_sym")
    ref_file = "thc_pivot_ref.eri.h5"
    new_file = "thc_pivot_given.eri.h5"
    eri_params = {
        "storage": "incore",
        "nIpts": mf.nbnd() * 10,
        "thresh": 1e-10,
        "ecut": mf.ecutrho(),
        "chol_block_size": 1,
        "init": True
    }
    if mpi.root():
        for f in (ref_file, new_file):
            if os.path.exists(f):
                os.remove(f)
    mpi.barrier()

    eri_ref = coqui.make_thc_coulomb(mf, {**eri_params, "save": ref_file})
    # rebuild reusing the pivots of the reference: same collocation matrices
    # and Coulomb matrix must come out
    eri_given = coqui.make_thc_coulomb(
        mf, {**eri_params, "save": new_file, "pivot_file": ref_file})
    assert eri_given.Np() == eri_ref.Np()
    mpi.barrier()

    if mpi.root():
        with h5py.File(ref_file, "r") as fr, h5py.File(new_file, "r") as fn:
            rp_ref = fr["interpolating_points"][()]
            rp_new = fn["interpolating_points"][()]
            X_ref = fr["collocation_matrix"][()]
            X_new = fn["collocation_matrix"][()]
            V_ref = fr["coulomb_matrix"][()]
            V_new = fn["coulomb_matrix"][()]
        assert np.array_equal(rp_ref, rp_new)
        assert np.linalg.norm(X_new - X_ref) / np.linalg.norm(X_ref) < 1e-10
        assert np.linalg.norm(V_new - V_ref) / np.linalg.norm(V_ref) < 1e-8
        os.remove(ref_file)
        os.remove(new_file)
    mpi.barrier()


def test_thc_pivots_only(mpi):
    import numpy as np
    import h5py

    mf = construct_qe_mf(mpi, "qe_lih222_sym")
    pivot_file = "thc_pivots_only.h5"
    thc_file = "thc_from_pivots_only.eri.h5"
    if mpi.root():
        for f in (pivot_file, thc_file):
            if os.path.exists(f):
                os.remove(f)
    mpi.barrier()

    # pivot search only (no interpolating vectors / Coulomb matrix)
    coqui.make_thc_pivots(mf, {
        "nIpts": mf.nbnd() * 10,
        "thresh": 1e-10,
        "ecut": mf.ecutrho(),
        "chol_block_size": 1,
        "save": pivot_file,
    })
    # full THC build on those pivots
    eri = coqui.make_thc_coulomb(mf, {
        "storage": "incore",
        "thresh": 1e-10,
        "ecut": mf.ecutrho(),
        "chol_block_size": 1,
        "init": True,
        "save": thc_file,
        "pivot_file": pivot_file,
    })
    mpi.barrier()

    if mpi.root():
        with h5py.File(pivot_file, "r") as fp, h5py.File(thc_file, "r") as ft:
            rp_only = fp["interpolating_points"][()]
            rp_thc = ft["interpolating_points"][()]
            assert fp["Np"][()] == rp_only.size
        assert eri.Np() == rp_only.size
        assert np.array_equal(rp_only, rp_thc)
        os.remove(pivot_file)
        os.remove(thc_file)
    mpi.barrier()


def test_ls_thc_eri(mpi):
    mf = construct_qe_mf(mpi, "qe_lih222")
    chol_params = {
        "tol": 1e-3,
        "storage": "outcore",
        "path": "./",
        "chol_block_size": 32
    }
    chol = coqui.make_chol_coulomb(mf, chol_params)
    assert isinstance(chol, coqui._lib.eri_module.CholCoulomb)

    thc_params = {
        "storage": "incore",
        "nIpts": mf.nbnd() * 10,
        "thresh": 1e-10,
        "ecut": mf.ecutrho(),
        "cd_dir": "./",
        "chol_block_size": 1,
        "init": True
    }
    ls_thc = coqui.make_thc_coulomb(mf, thc_params)
    assert isinstance(ls_thc, coqui._lib.eri_module.ThcCoulomb)
    assert ls_thc.mpi() == chol.mpi()
    assert ls_thc.mf() == chol.mf()
    mpi.barrier()

    if mpi.root():
        os.remove("./chol_info.h5")
        for iq in range(mf.nqpts_ibz()):
            os.remove(f"./Vq{iq}.h5")
    mpi.barrier()

