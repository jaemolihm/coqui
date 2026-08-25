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

import warnings

import numpy as np
import h5py
from typing import Optional, Tuple

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


def lr_DeltaH0_from_thc_aux(h_int, q_vec: np.ndarray,
                        u_mP: Optional[np.ndarray]) -> np.ndarray:
    """
    Band-basis LR perturbation from a local potential in the THC auxiliary basis.

    Thin wrapper over the C++ ``lr_DeltaH0_from_thc_aux``. Maps a local potential
    ``u(r)``, supplied through its projections onto the THC interpolating vectors
    ``u_P = int dr zeta_P(r) u(r)``, onto

        DeltaH0_ij(k) = sum_(p,P) conj(X_p(k+q)_Pi) u_P X_p(k)_Pj,

    the same aux -> primary transform the Hartree channel applies to DeltaJ_P, so
    the result is directly consumable by :func:`run_lr` as its rank-5 stack of
    perturbations at the one q.

    No overlap matrix enters even though the zeta are not orthonormal: the ISDF
    ansatz expands the pair density with its own values at the interpolating
    points, so zeta appears exactly once per matrix element. A caller holding
    expansion coefficients ``c_Q`` with ``u(r) = sum_Q c_Q zeta_Q(r)`` must
    convert them, ``u_P = sum_Q S_PQ c_Q`` with ``S_PQ = int dr zeta_P zeta_Q``.

    The call is collective (every rank must invoke it) and the result is returned
    only on the MPI root; every other rank gets an empty array.

    Parameters
    ----------
    h_int : ThcCoulomb
        THC ERI handler. Supplies the interpolating vectors X and Np.
    q_vec : array_like, shape (3,)
        Perturbation wavevector in crystal (fractional) coordinates.
    u_mP : np.ndarray or None
        Aux-basis potential projections u_P, shape (nmodes, Np) or (Np,) for a
        single one.
        Required on the MPI root; ignored on every other rank.

    Returns
    -------
    np.ndarray
        On the root: complex, shape (nmodes, nspin, nkpts_ibz, nbnd, nbnd), or
        (nspin, nkpts_ibz, nbnd, nbnd) when ``u_mP`` was passed as (Np,).
        On all other ranks: an empty array.
    """
    from mpi4py import MPI
    from coqui._lib.mbpt_module import lr_DeltaH0_from_thc_aux as lr_DeltaH0_from_thc_aux_cpp

    q_vec = np.ascontiguousarray(q_vec, dtype=np.float64)
    # C++ always carries the mode axis (the c2py converter fixes the rank), so
    # promote a single potential here and unwrap the length-1 result on the way out.
    # Only the root holds data, so `squeeze` is a root-local quantity; what has to be
    # collective is the *verdict*, because the C++ call below is collective and a
    # root-only raise would leave the other ranks inside it forever.
    root = MPI.COMM_WORLD.Get_rank() == 0
    squeeze = False
    err = None
    if root:
        try:
            if u_mP is None:
                err = "lr_DeltaH0_from_thc_aux: u_mP must be provided on the MPI root rank"
            else:
                u_mP = np.ascontiguousarray(u_mP, dtype=np.complex128)
                if u_mP.ndim not in (1, 2):
                    err = ("lr_DeltaH0_from_thc_aux: u_mP must be (Np,) or (nmodes, Np), "
                           f"got shape {u_mP.shape}")
                else:
                    squeeze = (u_mP.ndim == 1)
                    if squeeze:
                        u_mP = u_mP[None]
        except Exception as exc:                                      # noqa: BLE001
            err = f"lr_DeltaH0_from_thc_aux: {type(exc).__name__}: {exc}"
    else:
        u_mP = None
    err = MPI.COMM_WORLD.bcast(err, root=0)
    if err is not None:
        raise ValueError(err)

    DeltaH0 = lr_DeltaH0_from_thc_aux_cpp(h_int, q_vec, u_mP)
    return DeltaH0[0] if squeeze else DeltaH0


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


def _is_packed_complex(dset) -> bool:
    """
    True when a dataset holds a complex array as float64 with a trailing
    (real, imag) axis.

    CoQuí tags every such dataset with a ``__complex__`` attribute, which is
    authoritative: a genuinely real array whose last axis happens to have length 2
    is not reinterpreted. The storage layout is only consulted for files written
    without the attribute.
    """
    flag = dset.attrs.get('__complex__')
    if flag is not None:
        if isinstance(flag, bytes):
            flag = flag.decode()
        return int(flag) != 0
    return len(dset.shape) > 0 and dset.shape[-1] == 2 and dset.dtype.kind == 'f'


def _as_complex(dset) -> Optional[np.ndarray]:
    """Assemble a CoQuí on-disk array (trailing real/imag axis) into complex128."""
    if dset is None:
        return None
    raw = dset[:]
    if _is_packed_complex(dset):
        return raw[..., 0] + 1j * raw[..., 1]
    return raw


def read_lr_results(filename: str,
                    mode: Optional[int] = None,
                    aux_fock: bool = False
                    ) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray,
                               Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Read LR Dyson results from HDF5 checkpoint file.

    Parameters
    ----------
    filename : str
        HDF5 file path
    mode : int, optional
        Perturbation index for a batched run, which writes each perturbation to
        ``linear_response/mode{m}/``. Required when the file holds several; leave
        as None for a single-perturbation file, which writes straight to
        ``linear_response/``.
    aux_fock : bool, optional
        Also read the aux-basis (THC) Fock matrices ``DeltaF_PQ_skij`` and
        ``F_PQ_skij`` from the same group (default False). They are written only
        when the run set the ``output_aux_fock`` params key — it is a params key,
        not a run_lr keyword argument.

    Returns
    -------
    tuple
        (q_vec, DeltaG_tskij, DeltaDm_skij, DeltaF_PQ_skij, F_PQ_skij), always of
        this length. DeltaG_tskij is None when the run was made with
        save_DeltaG=False. The two aux-basis arrays are None unless aux_fock=True,
        and F_PQ_skij is None unless the run also did IBC.

    Raises
    ------
    KeyError
        If the file is batched and `mode` is None, listing the modes present, or
        if aux_fock=True and the group carries no DeltaF_PQ_skij.

    Notes
    -----
    A dataset written with nbnd_save carries an ``nbnd_save`` attribute and is
    the leading protected-band block, not the full-basis array. This function
    warns when it returns one, since the band axes are then shorter than the
    calculation's nbnd.

    The ``_PQ_skij`` arrays are (nspin, nkpts_ibz, Np, Np): the last two axes are
    THC auxiliary indices, not bands, despite the ``_skij`` suffix.

    CoQuí stores complex arrays as float64 with a trailing length-2 (real, imag)
    axis; every array returned here is assembled back into complex128.

    Under MPI, prefer calling on rank 0 only: concurrent h5py opens of the
    same file can contend on the file lock.
    """
    with h5py.File(filename, 'r') as f:
        lr_grp = f['linear_response']
        if mode is not None:
            lr_grp = lr_grp[f'mode{mode}']
        elif 'q_vec' not in lr_grp:
            modes = sorted(k for k in lr_grp.keys() if k.startswith('mode'))
            raise KeyError(
                f"{filename} holds a batched linear response ({', '.join(modes)}); "
                f"pass mode=<n> to pick one.")

        q_vec = lr_grp['q_vec'][:]
        DeltaDm_skij = _as_complex(lr_grp['DeltaDm_skij'])
        # Absent when the run set save_DeltaG=False.
        DeltaG_tskij = _as_complex(lr_grp.get('DeltaG_tskij'))

        for name, dset in (('DeltaG_tskij', lr_grp.get('DeltaG_tskij')),
                           ('DeltaDm_skij', lr_grp['DeltaDm_skij'])):
            if dset is not None and 'nbnd_save' in dset.attrs:
                warnings.warn(
                    f"{name} is a protected-band block of "
                    f"nbnd_save={int(dset.attrs['nbnd_save'])} bands, not the "
                    f"full-basis array.", stacklevel=2)

        DeltaF_PQ_skij = None
        F_PQ_skij = None
        if aux_fock:
            if 'DeltaF_PQ_skij' not in lr_grp:
                raise KeyError(
                    f"{filename} carries no DeltaF_PQ_skij in the requested group. "
                    "It is written only when the run passed "
                    "params[\"output_aux_fock\"] = True with a Fock-carrying kernel, "
                    "and it additionally requires npol = 1 and is rejected for a "
                    "split-kernel HSEX run.")
            DeltaF_PQ_skij = _as_complex(lr_grp['DeltaF_PQ_skij'])
            # Written only by the IBC path.
            F_PQ_skij = _as_complex(lr_grp.get('F_PQ_skij'))

    return q_vec, DeltaG_tskij, DeltaDm_skij, DeltaF_PQ_skij, F_PQ_skij


def read_lr_hessian(filename: str) -> dict:
    """
    Read the stationary free-energy-hessian datasets from an LR checkpoint.

    These are mode-PAIR matrices, so they live in the TOP-LEVEL
    ``linear_response/`` group rather than in the per-perturbation ``mode{m}/``
    subgroups, and they are written only by a run made with
    ``run_lr(..., hessian=True)``.

    Parameters
    ----------
    filename : str
        HDF5 file path

    Returns
    -------
    dict
        ``hessian`` (the plain estimator), ``hessian_sym`` (the stationary one),
        the two kernel pairings ``hessian_M`` = <K dX_l, dX_p> and
        ``hessian_M_prime`` = <K dX_l, dX'_p> together with
        ``hessian_static_prime`` = Tr(dH0_l, dDm'_p), so that
        ``hessian_sym = hessian_static_prime + hessian_M_prime - hessian_M``, the
        ``hessian_call_index`` list, ``Delta_mu_improved``, and the scalar
        ``hessian_herm_dev`` / ``hessian_sym_herm_dev`` / ``hessian_M_herm_dev``
        residuals.

    Raises
    ------
    KeyError
        If the file carries no hessian block, i.e. the run did not set
        ``hessian=True``.

    Notes
    -----
    Every matrix is ``(npert, npert)`` over the perturbations the run actually
    solved — never padded to a full mode count. A partial batch is a sub-block
    for inspection, not a dynamical matrix.

    ``hessian_call_index`` is the 0-based perturbation index WITHIN the run_lr
    call, which is all the C++ knows: it is handed a bare DeltaH0 stack and no
    mode numbering. It is not a phonon mode number, and deliberately not named
    like the ``hessian_modes`` dataset the phonon drivers write (1-based phonon
    modes) — mapping one to the other is the caller's job.

    Under MPI, prefer calling on rank 0 only: concurrent h5py opens of the same
    file can contend on the file lock.
    """
    complex_keys = ("hessian", "hessian_sym", "hessian_M", "hessian_M_prime",
                    "hessian_static_prime")
    real_keys = ("hessian_call_index", "Delta_mu_improved")
    scalar_keys = ("hessian_herm_dev", "hessian_sym_herm_dev", "hessian_M_herm_dev")

    with h5py.File(filename, 'r') as f:
        if 'linear_response' not in f:
            raise KeyError(f"{filename} holds no 'linear_response' group.")
        lr_grp = f['linear_response']
        if 'hessian_sym' not in lr_grp:
            raise KeyError(
                f"{filename} holds no hessian block; rerun with "
                f"run_lr(..., hessian=True).")

        out = {}
        for key in complex_keys:
            a = lr_grp[key][()]
            # CoQui writes complex as a trailing (..., 2) real/imag axis.
            out[key] = a[..., 0] + 1j * a[..., 1]
        for key in real_keys:
            out[key] = lr_grp[key][()]
        for key in scalar_keys:
            out[key] = float(lr_grp[key][()])
    return out
