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

Linear Response Driver for CoQuí

This module provides Python wrappers for linear response calculations.
The core LR Dyson solver is implemented in C++ and exposed via Python bindings.

Note: Pure Python/NumPy reference implementations for testing are located in
lr_debug/si_work/test_lr_phase1.py, not here (per project guidelines that
src/python/ should contain only wrappers/bindings, not numerical implementations).
"""

import numpy as np
import h5py
from typing import Tuple

# Import C++ implementation
from coqui._lib.mbpt_module import calculate_kpq_map as calculate_kpq_map_cpp

# Re-export run_lr from mbpt_driver (the main entry point)
from .mbpt_driver import run_lr


def calculate_kpq_map(kpts_crys: np.ndarray, q_vec: np.ndarray,
                       threshold: float = 1e-6) -> np.ndarray:
    """
    Compute k+q mapping for linear response calculations.

    Given a k-point grid and a perturbation wavevector q, compute the mapping
    kpq_map[ik] = ik' where k[ik] + q = k[ik'] (mod G).

    Parameters
    ----------
    kpts_crys : np.ndarray
        k-points in crystal coordinates, shape (nkpts, 3)
    q_vec : np.ndarray
        Perturbation wavevector in crystal coordinates, shape (3,)
    threshold : float, optional
        Tolerance for k-point matching, default 1e-6
        Note: Currently not passed to C++ (uses default 1e-6)

    Returns
    -------
    np.ndarray
        k → k+q index mapping, shape (nkpts,)
    """
    kpts_crys = np.asarray(kpts_crys, dtype=np.float64)
    q_vec = np.asarray(q_vec, dtype=np.float64)
    return calculate_kpq_map_cpp(kpts_crys, q_vec)


def is_q_commensurate(kpts_crys: np.ndarray, q_vec: np.ndarray,
                       threshold: float = 1e-6) -> bool:
    """
    Check if q is commensurate with the k-point grid.

    Parameters
    ----------
    kpts_crys : np.ndarray
        k-points in crystal coordinates, shape (nkpts, 3)
    q_vec : np.ndarray
        Perturbation wavevector in crystal coordinates, shape (3,)
    threshold : float, optional
        Tolerance for k-point matching

    Returns
    -------
    bool
        True if q is commensurate, False otherwise
    """
    kpts_crys = np.asarray(kpts_crys, dtype=np.float64)
    q_vec = np.asarray(q_vec, dtype=np.float64)
    # k + q must land on the grid (mod G) for every k. Pure-numpy check: the
    # C++ counterpart aborts on failure, which cannot be caught from python.
    for kpq in kpts_crys + q_vec[None, :]:
        d = kpts_crys - kpq[None, :]
        d = d - np.floor(d)
        d = d - np.round(d)
        if not np.any(np.sum(d**2, axis=1) < threshold**2):
            return False
    return True


def is_q_gamma(q_vec: np.ndarray, threshold: float = 1e-6) -> bool:
    """
    Check if q is approximately zero (Gamma point).

    Parameters
    ----------
    q_vec : np.ndarray
        Perturbation wavevector in crystal coordinates, shape (3,)
    threshold : float, optional
        Tolerance

    Returns
    -------
    bool
        True if q is approximately zero
    """
    d = np.abs(q_vec)
    d = d - np.floor(d)
    d = d - np.round(d)
    return bool(np.sum(d**2) < threshold**2)


def read_DeltaH0(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read linear response perturbation ΔH0 from HDF5 file.

    Expected HDF5 structure:
        /linear_response/
            q_vec              # (3,) perturbation wavevector in crystal coords
            DeltaH0_skij       # (ns, nk, nb, nb) complex perturbation matrix

    Parameters
    ----------
    filename : str
        HDF5 file path

    Returns
    -------
    tuple
        (q_vec, DeltaH0_skij)

    Under MPI, prefer calling on rank 0 only: concurrent h5py opens of the
    same file can contend on the file lock.
    """
    with h5py.File(filename, 'r') as f:
        lr_grp = f['linear_response']
        q_vec = lr_grp['q_vec'][:]
        DeltaH0_skij = lr_grp['DeltaH0_skij'][:]
    return q_vec, DeltaH0_skij


def write_DeltaH0(filename: str, q_vec: np.ndarray,
                   DeltaH0_skij: np.ndarray) -> None:
    """
    Write linear response perturbation ΔH0 to HDF5 file.

    Parameters
    ----------
    filename : str
        HDF5 file path
    q_vec : np.ndarray
        Perturbation wavevector in crystal coords, shape (3,)
    DeltaH0_skij : np.ndarray
        Perturbation matrix, shape (ns, nk, nb, nb)

    MPI-safe: only rank 0 writes (concurrent h5py writers contend on the
    file lock); all ranks synchronize before returning.
    """
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    if comm.Get_rank() == 0:
        with h5py.File(filename, 'a') as f:
            if 'linear_response' not in f:
                lr_grp = f.create_group('linear_response')
            else:
                lr_grp = f['linear_response']

            if 'q_vec' in lr_grp:
                del lr_grp['q_vec']
            if 'DeltaH0_skij' in lr_grp:
                del lr_grp['DeltaH0_skij']

            lr_grp.create_dataset('q_vec', data=q_vec)
            lr_grp.create_dataset('DeltaH0_skij', data=DeltaH0_skij)
    comm.Barrier()


def read_lr_results(filename: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Read LR Dyson results from HDF5 checkpoint file.

    Parameters
    ----------
    filename : str
        HDF5 file path

    Returns
    -------
    tuple
        (q_vec, DeltaG_tskij, DeltaDm_skij)

    Under MPI, prefer calling on rank 0 only: concurrent h5py opens of the
    same file can contend on the file lock.
    """
    with h5py.File(filename, 'r') as f:
        lr_grp = f['linear_response']
        q_vec = lr_grp['q_vec'][:]
        DeltaG_tskij = lr_grp['DeltaG_tskij'][:]
        DeltaDm_skij = lr_grp['DeltaDm_skij'][:]
    return q_vec, DeltaG_tskij, DeltaDm_skij


