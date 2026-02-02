"""
==========================================================================
CoQuí: Correlated Quantum ínterface

Copyright (c) 2022-2025 Simons Foundation & The CoQuí developer team

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

import json

from coqui._lib.mbpt_module import mbpt as mbpt_cxx


def _run_mbpt(solver_type, params, h_int,
             h_int_hf = None, h_int_hartree = None, h_int_exchange = None,
             *, projector_info = None, local_polarizabilities = None):
    args = [solver_type, json.dumps(params), h_int]

    if projector_info is not None:
        ## GW+EDMFT interface with optional local polarizabilities
        if local_polarizabilities is not None:
            required_keys = {"imp", "dc"}
            missing = required_keys - local_polarizabilities.keys()
            if missing:
                raise ValueError(f"Missing keys: {missing}")

        proj_mat = projector_info.get("proj_mat")
        band_window = projector_info.get("band_window")
        kpts_w90 = projector_info.get("kpts_w90")
        mbpt_cxx(*args, proj_mat, band_window, kpts_w90, local_polarizabilities)
    else:
        # Pure MBPT interface without projector info
        if h_int_hf is not None:
            args.append(h_int_hf)
        elif h_int_hartree is not None and h_int_exchange is not None:
            args.extend([h_int_hartree, h_int_exchange])
        elif h_int_hf is None and (h_int_hartree is not None or h_int_exchange is not None):
            raise ValueError("Invalid mbpt input: hartree_eri and exchange_eri must be both provided, or neither.")
        mbpt_cxx(*args)


def run_hf(params, h_int, h_int_exchange=None):
    args = ["hf", json.dumps(params), h_int]
    if h_int_exchange is not None:
        args.append(h_int_exchange)
    mbpt_cxx(*args)


def run_gw(params, h_int,
           h_int_hf = None, h_int_hartree = None, h_int_exchange = None,
           *, projector_info = None, local_polarizabilities = None):
    _run_mbpt("gw", params, h_int,
              h_int_hf = h_int_hf, h_int_hartree = h_int_hartree, h_int_exchange = h_int_exchange,
              projector_info = projector_info, local_polarizabilities = local_polarizabilities)


def run_qpg0w0(params, h_int,
               h_int_hf = None, h_int_hartree = None, h_int_exchange = None):
    _run_mbpt("evgw0", params, h_int,
              h_int_hf = h_int_hf, h_int_hartree = h_int_hartree, h_int_exchange = h_int_exchange,
              projector_info = None, local_polarizabilities = None)


def run_lr_dyson(params, h_int, q_vec, DeltaH0_skij, fix_density=False):
    """
    Run linear response Dyson equation calculation.

    Computes ΔG in response to a perturbation ΔH0 at wavevector q.
    Requires a previous HF/GW calculation to provide the unperturbed G.

    Parameters
    ----------
    params : dict
        Parameters including:
        - prefix: Input checkpoint prefix (reads {prefix}.mbpt.h5)
        - output: Output checkpoint prefix (default: same as prefix)
    h_int : ThcCoulomb or CholCoulomb
        ERI handler from the original calculation
    q_vec : array-like
        Perturbation wavevector in crystal coordinates, shape (3,)
    DeltaH0_skij : np.ndarray
        Perturbation matrix, shape (ns, nk, nb, nb)
    fix_density : bool, optional
        If True, compute Δμ to enforce particle conservation ΔN=0 (default False).
        Only meaningful for q=0 perturbations.

    Returns
    -------
    float
        The Δμ value used (computed if fix_density=True, otherwise 0.0).

    Notes
    -----
    Results (ΔG, ΔDm) are written to {output}.mbpt.h5 under
    the "linear_response" group.
    """
    import numpy as np
    from coqui._lib.mbpt_module import lr_dyson as lr_dyson_cxx

    q_vec = np.asarray(q_vec, dtype=np.float64)
    DeltaH0_skij = np.asarray(DeltaH0_skij, dtype=np.complex128)

    return lr_dyson_cxx(json.dumps(params), h_int, q_vec, DeltaH0_skij, bool(fix_density))


def run_lr_hf(h_int, q_vec, DeltaDm_skij, S_skij=None, compute_hartree=True, compute_exchange=True):
    """
    Compute linear response Fock matrix from LR density matrix.

    Computes ΔF = ΔJ + ΔK from the LR density matrix ΔDm using THC-ERI.
    This function is used for Step 2.1 of LR-HF Phase 2 implementation.

    Parameters
    ----------
    h_int : ThcCoulomb
        THC ERI handler (currently only THC is supported)
    q_vec : array-like
        Perturbation wavevector in crystal coordinates, shape (3,)
    DeltaDm_skij : np.ndarray
        LR density matrix, shape (ns, nk, nb, nb)
    S_skij : np.ndarray, optional
        Overlap matrix, shape (ns, nk, nb, nb). If None, identity is assumed.
    compute_hartree : bool, optional
        Whether to compute the Hartree (Coulomb) term, default True
    compute_exchange : bool, optional
        Whether to compute the Exchange term, default True

    Returns
    -------
    np.ndarray
        LR Fock matrix ΔF, shape (ns, nk, nb, nb)

    Notes
    -----
    The LR Fock matrix is computed as:
        ΔF(k) = ΔJ(k) + ΔK(k)

    where:
        - ΔJ is the LR Hartree term (diagonal in THC auxiliary basis)
        - ΔK is the LR Exchange term

    If S_skij is not provided, an identity overlap matrix is assumed (orthonormal basis).
    """
    import numpy as np
    from coqui._lib.mbpt_module import lr_hf_cpp

    q_vec = np.asarray(q_vec, dtype=np.float64)
    DeltaDm_skij = np.asarray(DeltaDm_skij, dtype=np.complex128)

    # If S not provided, assume identity matrix (orthonormal basis)
    if S_skij is None:
        ns, nk, nb, _ = DeltaDm_skij.shape
        S_skij = np.zeros((ns, nk, nb, nb), dtype=np.complex128)
        for s in range(ns):
            for k in range(nk):
                S_skij[s, k] = np.eye(nb, dtype=np.complex128)
    else:
        S_skij = np.asarray(S_skij, dtype=np.complex128)

    return lr_hf_cpp(h_int, q_vec, DeltaDm_skij, S_skij, compute_hartree, compute_exchange)


# JML: Temporarily exposed for linear-response debugging. TODO: Remove from wrapper
def hf_evaluate(h_int, Dm_skij, S_skij, compute_hartree=True, compute_exchange=True):
    """
    Compute HF self-energy (Fock matrix) from density matrix.

    Computes F = J + K from the density matrix Dm.

    Parameters
    ----------
    h_int : ThcCoulomb or CholCoulomb
        ERI handler (THC or Cholesky)
    Dm_skij : np.ndarray
        Density matrix, shape (ns, nk, nb, nb)
    S_skij : np.ndarray
        Overlap matrix, shape (ns, nk, nb, nb)
    compute_hartree : bool, optional
        Whether to compute the Hartree (Coulomb) term, default True
    compute_exchange : bool, optional
        Whether to compute the Exchange term, default True

    Returns
    -------
    np.ndarray
        Fock matrix F, shape (ns, nk, nb, nb)
    """
    import numpy as np
    from coqui._lib.mbpt_module import hf_evaluate_cpp

    Dm_skij = np.asarray(Dm_skij, dtype=np.complex128)
    S_skij = np.asarray(S_skij, dtype=np.complex128)

    return hf_evaluate_cpp(h_int, Dm_skij, S_skij, compute_hartree, compute_exchange)
