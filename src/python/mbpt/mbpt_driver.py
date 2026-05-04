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
    """
    Run Hartree-Fock calculation.

    Parameters
    ----------
    params : dict
        Parameters including:
        - beta : float
            Inverse temperature in a.u. (default: 1000)
        - niter : int
            Maximum number of SCF iterations (default: 1)
        - conv_thr : float
            Convergence threshold (default: 1e-8)
        - output : str
            Output checkpoint prefix (default: "bdft.mbpt")
        - h0_source : str, optional
            Source of H0 matrix. Options:
            - "compute" (default): Calculate H0 from plane-wave orbitals
            - "checkpoint": Read H0 from {output}.mbpt.h5 checkpoint
        - compute_exchange : bool, optional
            Whether to compute exchange (K) term (default: True).
            Set to False for Hartree-only calculations.
    h_int : ThcCoulomb or CholCoulomb
        ERI handler
    h_int_exchange : optional
        Separate ERI for exchange (if different from h_int)
    """
    args = ["hf", json.dumps(params), h_int]
    if h_int_exchange is not None:
        args.append(h_int_exchange)
    mbpt_cxx(*args)


def run_gw(params, h_int,
           h_int_hf = None, h_int_hartree = None, h_int_exchange = None,
           *, projector_info = None, local_polarizabilities = None):
    """
    Run GW calculation.

    Parameters
    ----------
    params : dict
        Parameters including:
        - beta : float
            Inverse temperature in a.u. (default: 1000)
        - niter : int
            Maximum number of SCF iterations (default: 1)
        - conv_thr : float
            Convergence threshold (default: 1e-8)
        - output : str
            Output checkpoint prefix (default: "bdft.mbpt")
        - h0_source : str, optional
            Source of H0 matrix. Options:
            - "compute" (default): Calculate H0 from plane-wave orbitals
            - "checkpoint": Read H0 from {output}.mbpt.h5 checkpoint
    h_int : ThcCoulomb or CholCoulomb
        ERI handler
    h_int_hf : optional
        Separate ERI for HF (if different from h_int)
    h_int_hartree : optional
        Separate ERI for Hartree (if different from h_int)
    h_int_exchange : optional
        Separate ERI for exchange (if different from h_int)
    projector_info : dict, optional
        Projector information for GW+EDMFT calculations
    local_polarizabilities : dict, optional
        Local polarizabilities for GW+EDMFT calculations
    """
    _run_mbpt("gw", params, h_int,
              h_int_hf = h_int_hf, h_int_hartree = h_int_hartree, h_int_exchange = h_int_exchange,
              projector_info = projector_info, local_polarizabilities = local_polarizabilities)


def run_qpg0w0(params, h_int,
               h_int_hf = None, h_int_hartree = None, h_int_exchange = None):
    _run_mbpt("evgw0", params, h_int,
              h_int_hf = h_int_hf, h_int_hartree = h_int_hartree, h_int_exchange = h_int_exchange,
              projector_info = None, local_polarizabilities = None)


def run_lr(params, h_int, q_vec, DeltaH0_skij,
           include_hartree=True, include_exchange=True,
           gw_mode="none",
           max_iter=50, tol=1e-8, fix_density=False, iter_alg=None,
           include_gw_sigma=None,
           DeltaX_left=None, DeltaX_right=None,
           DeltaV_qPQ=None,
           div_corr=True):
    """
    Run unified linear response calculation.

    Runs the LR SCF loop with configurable Hartree, Exchange, and GW self-energy:
        ΔH0 → ΔG → ΔDm → [ΔF] → [ΔΣ] → ΔG → ... (iterate until convergence)

    Subsumes both run_lr_dyson (one-shot, all flags false) and
    run_lr_hf_scf (HF, no GW).

    Parameters
    ----------
    params : dict
        Parameters including:
        - prefix: Input checkpoint prefix (reads {prefix}.mbpt.h5)
        - output: Output checkpoint prefix (default: same as prefix)
        - input_type: HDF5 group to read checkpoint from (default: "scf")
        - input_iter: Iteration number to read (default: -1 = use final_iter)
    h_int : ThcCoulomb
        THC ERI handler
    q_vec : array-like
        Perturbation wavevector in crystal coordinates, shape (3,)
    DeltaH0_skij : np.ndarray or None
        External perturbation matrix, shape (ns, nk, nb, nb).
        Required on the MPI global root; ignored on non-root ranks.
    include_hartree : bool, optional
        Include Hartree (Coulomb) term in SCF loop (default True)
    include_exchange : bool, optional
        Include Exchange term in SCF loop (default True)
    gw_mode : str, optional
        GW self-energy update mode (default "none"):
        - "none": No GW self-energy
        - "fixed_W": ΔΣ = -ΔG⊙W (W frozen)
        - "full": ΔΣ = -ΔG⊙W - G⊙ΔW (full LR-scGW)
    max_iter : int, optional
        Maximum number of SCF iterations (default 50). Use 1 for one-shot.
    tol : float, optional
        Convergence tolerance for ||ΔDm_new - ΔDm_old|| (default 1e-8)
    fix_density : bool, optional
        If True, compute Δμ to enforce particle conservation ΔN=0 (default False).
        Only meaningful for q=0 perturbations.
    iter_alg : dict or None, optional
        Iteration algorithm configuration. If None, uses damping with mixing=1.0.
        Keys:
        - alg : str - "damping" (default) or "DIIS"
        - mixing : float - Damping/mixing parameter (default 1.0)
        - max_subsp_size : int - DIIS subspace size (default 5)
        - diis_warmup : int - Warmup iterations before DIIS (default 3)
    include_gw_sigma : bool or None, optional
        Deprecated. Use gw_mode instead. If provided, maps True -> "fixed_W",
        False -> "none". Overrides gw_mode if both are specified.
    DeltaX_left : np.ndarray or None, optional
        Perturbation of collocation matrix δ^q X, shape (ns, nkpts, Np, nb).
        Full BZ indexed. Required together with DeltaX_right. Required on the
        MPI global root; ignored on non-root ranks.
    DeltaX_right : np.ndarray or None, optional
        Perturbation of collocation matrix δ^{-q} X, shape (ns, nkpts, Np, nb).
        Full BZ indexed. Required together with DeltaX_left. Required on the
        MPI global root; ignored on non-root ranks.
    DeltaV_qPQ : np.ndarray or None, optional
        Perturbation of THC auxiliary-basis Coulomb matrix δV^q, shape
        (nkpts, Np, Np), full-BZ q-grid. When provided, adds the δV
        correction terms to ΔJ (δV^{q=q_pert}·Dm) and ΔK (δV^R ⊙ Dm^R).

    Returns
    -------
    tuple[int, float]
        (niter, Delta_mu) - number of iterations and final chemical potential shift

    Notes
    -----
    Results are written to {output}.mbpt.h5 under the "linear_response" group.
    """
    import numpy as np
    from coqui._lib.mbpt_module import run_lr_cpp

    q_vec = np.asarray(q_vec, dtype=np.float64)
    # Force non-root ranks to pass None: tolerates legacy callers that
    # still hand a replicated array on every rank.
    from mpi4py import MPI
    if MPI.COMM_WORLD.Get_rank() == 0:
        if DeltaH0_skij is None:
            raise ValueError("run_lr: DeltaH0_skij must be provided on the MPI root rank")
        DeltaH0_skij = np.asarray(DeltaH0_skij, dtype=np.complex128)
    else:
        DeltaH0_skij = None

    # Handle deprecated include_gw_sigma parameter
    if include_gw_sigma is not None:
        import warnings
        warnings.warn(
            "include_gw_sigma is deprecated, use gw_mode='fixed_W' or gw_mode='full' instead",
            DeprecationWarning, stacklevel=2)
        gw_mode = "fixed_W" if include_gw_sigma else "none"

    if gw_mode not in ("none", "fixed_W", "full"):
        raise ValueError(f"Unknown gw_mode '{gw_mode}'. Must be 'none', 'fixed_W', or 'full'.")

    # Parse iter_alg dict with defaults
    if iter_alg is None:
        iter_alg = {}
    alg = str(iter_alg.get("alg", "damping"))
    if alg not in ("damping", "DIIS"):
        raise ValueError(f"Unknown iter_alg '{alg}'. Must be 'damping' or 'DIIS'.")
    mixing = float(iter_alg.get("mixing", 1.0))
    max_subsp_size = int(iter_alg.get("max_subsp_size", 5))
    diis_warmup = int(iter_alg.get("diis_warmup", 3))

    if MPI.COMM_WORLD.Get_rank() == 0:
        if (DeltaX_left is None) != (DeltaX_right is None):
            raise ValueError("DeltaX_left and DeltaX_right must both be provided or both be None.")
        dx_left = np.asarray(DeltaX_left, dtype=np.complex128) if DeltaX_left is not None else None
        dx_right = np.asarray(DeltaX_right, dtype=np.complex128) if DeltaX_right is not None else None
        dv_qPQ = np.asarray(DeltaV_qPQ, dtype=np.complex128) if DeltaV_qPQ is not None else None
    else:
        dx_left = None
        dx_right = None
        dv_qPQ = None

    # Pass div_corr flag through params dict (read by C++ run_lr_calc)
    params_with_div = dict(params)
    params_with_div["div_corr"] = bool(div_corr)

    return run_lr_cpp(json.dumps(params_with_div), h_int, q_vec, DeltaH0_skij,
                      bool(include_hartree), bool(include_exchange), str(gw_mode),
                      int(max_iter), float(tol), bool(fix_density),
                      alg, mixing, max_subsp_size, diis_warmup,
                      dx_left, dx_right, dv_qPQ)


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


def run_lr_gw_sigma_DeltaG(params, h_int, q_pert, DeltaG_tskij, div_corr=True):
    """
    Compute LR GW self-energy term 1: ΔΣ = -ΔG ⊙ W_c + div_corr (fixed W, R-space).

    Reads W from thc_screened_interaction.h5 and eps_inv_head from checkpoint.

    Parameters
    ----------
    params : dict
        Parameters including:
        - prefix: Input checkpoint prefix (reads {prefix}.mbpt.h5)
    h_int : ThcCoulomb
        THC ERI handler
    q_pert : array-like
        LR perturbation wavevector in crystal coordinates, shape (3,)
    DeltaG_tskij : np.ndarray or None
        LR Green's function, shape (nt, ns, nk, nb, nb). Required on the MPI
        global root; ignored on non-root ranks.
    div_corr : bool, optional
        Apply q→0 divergence correction (default True).

    Returns
    -------
    np.ndarray on rank 0, shape (nt, ns, nk, nb, nb)
    None on non-root ranks
    """
    import numpy as np
    from mpi4py import MPI
    from coqui._lib.mbpt_module import lr_gw_sigma_DeltaG_cpp

    q_pert = np.asarray(q_pert, dtype=np.float64)
    if MPI.COMM_WORLD.Get_rank() == 0:
        if DeltaG_tskij is None:
            raise ValueError("run_lr_gw_sigma_DeltaG: DeltaG_tskij must be provided on the MPI root rank")
        DeltaG_tskij = np.asarray(DeltaG_tskij, dtype=np.complex128)
    else:
        DeltaG_tskij = None

    params_with_div = dict(params)
    params_with_div["div_corr"] = bool(div_corr)

    arr = lr_gw_sigma_DeltaG_cpp(json.dumps(params_with_div), h_int, q_pert, DeltaG_tskij)
    return arr if arr.size else None


def run_lr_gw_Pi(h_int, q_pert, G_tskij, DeltaG_tskij,
                  DeltaX_left=None, DeltaX_right=None):
    """
    Compute LR polarization ΔP = -ΔG·G - G·ΔG (R-space Hadamard product).

    Uses lr_thc_comm for ΔG factors (asymmetric X(k+q)/X(k)),
    thc_solver_comm for G factors (symmetric).

    Parameters
    ----------
    h_int : ThcCoulomb
        THC ERI handler
    q_pert : array-like
        LR perturbation wavevector in crystal coordinates, shape (3,)
    G_tskij : np.ndarray
        Unperturbed Green's function, shape (nt, ns, nk, nb, nb)
    DeltaG_tskij : np.ndarray
        LR Green's function, shape (nt, ns, nk, nb, nb)
    DeltaX_left : np.ndarray or None, optional
        Perturbation of collocation matrix δ^q X, shape (ns, nkpts, Np, nb).
        Full BZ indexed. Must be provided together with DeltaX_right. Required
        on the MPI global root; ignored on non-root ranks.
        When both DeltaX arrays are given, the primary→aux IBC correction is
        applied inside ``lr_rpa_pi::evaluate_lr_Pi``. Without IBC the primary→
        aux transform of ΔG is incomplete (see run_4 in 43_lr_vs_fd_THC for
        the rationale), so the FD comparison typically needs IBC.
    DeltaX_right : np.ndarray or None, optional
        Perturbation of collocation matrix δ^{-q} X at storage k+q, shape
        (ns, nkpts, Np, nb). Full BZ indexed. Must be provided together with
        DeltaX_left. Required on the MPI global root; ignored on non-root ranks.

    Returns
    -------
    np.ndarray
        LR polarization ΔP, shape (nt_half, nkpts, NP, NP)
    """
    import numpy as np
    from mpi4py import MPI
    from coqui._lib.mbpt_module import lr_gw_Pi_cpp

    q_pert = np.asarray(q_pert, dtype=np.float64)
    if MPI.COMM_WORLD.Get_rank() == 0:
        if G_tskij is None:
            raise ValueError("run_lr_gw_Pi: G_tskij must be provided on the MPI root rank")
        if DeltaG_tskij is None:
            raise ValueError("run_lr_gw_Pi: DeltaG_tskij must be provided on the MPI root rank")
        G_tskij = np.asarray(G_tskij, dtype=np.complex128)
        DeltaG_tskij = np.asarray(DeltaG_tskij, dtype=np.complex128)
        if (DeltaX_left is None) != (DeltaX_right is None):
            raise ValueError("DeltaX_left and DeltaX_right must both be provided or both be None.")
        if DeltaX_left is not None:
            DeltaX_left = np.asarray(DeltaX_left, dtype=np.complex128)
        if DeltaX_right is not None:
            DeltaX_right = np.asarray(DeltaX_right, dtype=np.complex128)
    else:
        G_tskij = None
        DeltaG_tskij = None
        DeltaX_left = None
        DeltaX_right = None

    arr = lr_gw_Pi_cpp(h_int, q_pert, G_tskij, DeltaG_tskij,
                       DeltaX_left=DeltaX_left, DeltaX_right=DeltaX_right)
    return arr if arr.size else None


def gw_evaluate_Pi(params, h_int, G_tskij):
    """
    Evaluate standard RPA polarization P[G].

    Computes the RPA polarization from a given Green's function.
    Used for finite-difference testing of LR polarization.

    Parameters
    ----------
    params : dict
        Parameters including:
        - prefix: Input checkpoint prefix (reads {prefix}.mbpt.h5)
    h_int : ThcCoulomb
        THC ERI handler
    G_tskij : np.ndarray or None
        Green's function, shape (nt, ns, nk, nb, nb). Required on the MPI
        global root; ignored on non-root ranks.

    Returns
    -------
    np.ndarray on rank 0, shape (nt_half, nkpts, NP, NP)
    None on non-root ranks
    """
    import numpy as np
    from mpi4py import MPI
    from coqui._lib.mbpt_module import gw_evaluate_Pi_cpp

    if MPI.COMM_WORLD.Get_rank() == 0:
        if G_tskij is None:
            raise ValueError("gw_evaluate_Pi: G_tskij must be provided on the MPI root rank")
        G_tskij = np.asarray(G_tskij, dtype=np.complex128)
    else:
        G_tskij = None

    arr = gw_evaluate_Pi_cpp(json.dumps(params), h_int, G_tskij)
    return arr if arr.size else None


def gw_evaluate_sigma(params, h_int, G_tskij, div_corr=True):
    """
    Evaluate GW self-energy Σ = -G ⊙ W_c [+ div_corr] using W from file.

    Same R-space computation as lr_gw_sigma_DeltaG but for a full Green's function.
    Used for finite-difference testing: Σ[G+εΔG] computed by passing G+εΔG.

    Parameters
    ----------
    params : dict
        Parameters including:
        - prefix: Input checkpoint prefix (reads {prefix}.mbpt.h5)
    h_int : ThcCoulomb
        THC ERI handler
    G_tskij : np.ndarray or None
        Green's function, shape (nt, ns, nk, nb, nb). Required on the MPI
        global root; ignored on non-root ranks.
    div_corr : bool, optional
        Whether to apply divergence correction (default True).
        Set to False for FD testing of LR code (which has no div_corr).

    Returns
    -------
    np.ndarray on rank 0, shape (nt, ns, nk, nb, nb)
    None on non-root ranks
    """
    import numpy as np
    from mpi4py import MPI
    from coqui._lib.mbpt_module import gw_evaluate_sigma_cpp

    if MPI.COMM_WORLD.Get_rank() == 0:
        if G_tskij is None:
            raise ValueError("gw_evaluate_sigma: G_tskij must be provided on the MPI root rank")
        G_tskij = np.asarray(G_tskij, dtype=np.complex128)
    else:
        G_tskij = None

    arr = gw_evaluate_sigma_cpp(json.dumps(params), h_int, G_tskij, bool(div_corr))
    return arr if arr.size else None


def run_lr_gw_W(params, h_int, q_pert, DeltaPi_tqPQ):
    """
    Compute LR screened interaction ΔW = (Z+W_c) · ΔΠ · (Z+W_c).

    Reads W_c from thc_screened_interaction.h5 and IAFT from checkpoint,
    then calls lr_scr_coulomb_t::solve_lr_dyson_W to compute ΔW_c(τ).

    Parameters
    ----------
    params : dict
        Parameters including:
        - prefix: Input checkpoint prefix (reads {prefix}.mbpt.h5)
    h_int : ThcCoulomb
        THC ERI handler
    q_pert : array-like
        LR perturbation wavevector in crystal coordinates, shape (3,)
    DeltaPi_tqPQ : np.ndarray or None
        LR polarization, shape (nt_half, nkpts, NP, NP). Required on the MPI
        global root; ignored on non-root ranks.

    Returns
    -------
    np.ndarray on rank 0, shape (nt_half, nkpts, NP, NP)
    None on non-root ranks
    """
    import numpy as np
    from mpi4py import MPI
    from coqui._lib.mbpt_module import lr_gw_W_cpp

    q_pert = np.asarray(q_pert, dtype=np.float64)
    if MPI.COMM_WORLD.Get_rank() == 0:
        if DeltaPi_tqPQ is None:
            raise ValueError("run_lr_gw_W: DeltaPi_tqPQ must be provided on the MPI root rank")
        DeltaPi_tqPQ = np.asarray(DeltaPi_tqPQ, dtype=np.complex128)
    else:
        DeltaPi_tqPQ = None

    arr = lr_gw_W_cpp(json.dumps(params), h_int, q_pert, DeltaPi_tqPQ)
    return arr if arr.size else None


def gw_evaluate_W_from_Pi(params, h_int, Pi_tqPQ):
    """
    Evaluate screened interaction W_c from polarization Π via Dyson equation.

    Reads IAFT from checkpoint, applies W Dyson equation Π → W_c(τ).
    Used for finite-difference testing of LR screened interaction.

    Parameters
    ----------
    params : dict
        Parameters including:
        - prefix: Input checkpoint prefix (reads {prefix}.mbpt.h5)
    h_int : ThcCoulomb
        THC ERI handler
    Pi_tqPQ : np.ndarray or None
        Polarization, shape (nt_half, nkpts, NP, NP). Required on the MPI
        global root; ignored on non-root ranks.

    Returns
    -------
    np.ndarray on rank 0, shape (nt_half, nkpts, NP, NP)
    None on non-root ranks
    """
    import numpy as np
    from mpi4py import MPI
    from coqui._lib.mbpt_module import gw_evaluate_W_from_Pi_cpp

    if MPI.COMM_WORLD.Get_rank() == 0:
        if Pi_tqPQ is None:
            raise ValueError("gw_evaluate_W_from_Pi: Pi_tqPQ must be provided on the MPI root rank")
        Pi_tqPQ = np.asarray(Pi_tqPQ, dtype=np.complex128)
    else:
        Pi_tqPQ = None

    arr = gw_evaluate_W_from_Pi_cpp(json.dumps(params), h_int, Pi_tqPQ)
    return arr if arr.size else None


def run_lr_gw_sigma_DeltaW(params, h_int, q_pert, G_tskij, DeltaW_qtPQ, div_corr=True):
    """
    Compute LR GW self-energy term 2: ΔΣ = -G ⊙ ΔW + div_corr.

    Computes ΔΣ = -G ⊙ ΔW from a pre-computed DeltaW.
    At q_pert=0, also applies term 2 divergence correction using Δeps_inv_head from ΔW.

    Parameters
    ----------
    params : dict
        Parameters including:
        - prefix: Input checkpoint prefix (reads {prefix}.mbpt.h5)
    h_int : ThcCoulomb
        THC ERI handler
    q_pert : array-like
        LR perturbation wavevector in crystal coordinates, shape (3,)
    G_tskij : np.ndarray or None
        Unperturbed Green's function, shape (nt, ns, nk, nb, nb). Required on
        the MPI global root; ignored on non-root ranks.
    DeltaW_qtPQ : np.ndarray
        LR screened interaction, shape (nk, nt_half, NP, NP)
    div_corr : bool, optional
        Apply q→0 divergence correction (default True).

    Returns
    -------
    np.ndarray on rank 0, shape (nt, ns, nk, nb, nb)
    None on non-root ranks
    """
    import numpy as np
    from mpi4py import MPI
    from coqui._lib.mbpt_module import lr_gw_sigma_DeltaW_cpp

    q_pert = np.asarray(q_pert, dtype=np.float64)
    if MPI.COMM_WORLD.Get_rank() == 0:
        if G_tskij is None:
            raise ValueError("run_lr_gw_sigma_DeltaW: G_tskij must be provided on the MPI root rank")
        if DeltaW_qtPQ is None:
            raise ValueError("run_lr_gw_sigma_DeltaW: DeltaW_qtPQ must be provided on the MPI root rank")
        G_tskij = np.asarray(G_tskij, dtype=np.complex128)
        DeltaW_qtPQ = np.asarray(DeltaW_qtPQ, dtype=np.complex128)
    else:
        G_tskij = None
        DeltaW_qtPQ = None

    params_with_div = dict(params)
    params_with_div["div_corr"] = bool(div_corr)

    arr = lr_gw_sigma_DeltaW_cpp(json.dumps(params_with_div), h_int, q_pert, G_tskij, DeltaW_qtPQ)
    return arr if arr.size else None


def compute_eps_inv_head(params, h_int, W_c_tqPQ):
    """
    Compute eps_inv_head from correlation screened interaction W_c in THC product basis.

    Extracts the G=0,G'=0 component of (ε⁻¹ - 1) and extrapolates to q→0
    using Gygi extrapolation. This matches the convention used by
    Sigma_div_correction (which uses ε⁻¹-1, not ε⁻¹).

    Parameters
    ----------
    params : dict
        Parameters including:
        - prefix: Input checkpoint prefix (reads {prefix}.mbpt.h5 for IAFT)
    h_int : ThcCoulomb
        THC ERI handler
    W_c_tqPQ : np.ndarray or None
        Correlation screened interaction W_c = W - V, shape (nt_half, nkpts, NP, NP).
        NOT the full W. Required on the MPI global root; ignored on non-root ranks.

    Returns
    -------
    np.ndarray
        (ε⁻¹ - 1) head at q=0, shape (nt_half,) — replicated on all ranks
    """
    import numpy as np
    from mpi4py import MPI
    from coqui._lib.mbpt_module import compute_eps_inv_head_cpp

    if MPI.COMM_WORLD.Get_rank() == 0:
        if W_c_tqPQ is None:
            raise ValueError("compute_eps_inv_head: W_c_tqPQ must be provided on the MPI root rank")
        W_c_tqPQ = np.asarray(W_c_tqPQ, dtype=np.complex128)
    else:
        W_c_tqPQ = None

    return compute_eps_inv_head_cpp(json.dumps(params), h_int, W_c_tqPQ)


def gw_evaluate_sigma_with_W(params, h_int, G_tskij, W_c_qtPQ, eps_inv_head, div_corr=True):
    """
    Evaluate GW self-energy Σ = -G ⊙ W_c [+ div_corr] with provided G and W_c.

    Uses the provided G and W_c arrays (not from file). Used for finite-difference
    testing of the full LR-GW self-energy.

    Parameters
    ----------
    params : dict
        Parameters including:
        - prefix: Input checkpoint prefix (reads {prefix}.mbpt.h5 for IAFT)
    h_int : ThcCoulomb
        THC ERI handler
    G_tskij : np.ndarray or None
        Green's function, shape (nt, ns, nk, nb, nb). Required on the MPI
        global root; ignored on non-root ranks.
    W_c_qtPQ : np.ndarray
        Screened interaction, shape (nkpts, nt_half, NP, NP)
    eps_inv_head : np.ndarray
        Inverse dielectric head, shape (nt_half,)
    div_corr : bool, optional
        Whether to apply divergence correction (default True)

    Returns
    -------
    np.ndarray on rank 0, shape (nt, ns, nk, nb, nb)
    None on non-root ranks
    """
    import numpy as np
    from mpi4py import MPI
    from coqui._lib.mbpt_module import gw_evaluate_sigma_with_W_cpp

    if MPI.COMM_WORLD.Get_rank() == 0:
        if G_tskij is None:
            raise ValueError("gw_evaluate_sigma_with_W: G_tskij must be provided on the MPI root rank")
        if W_c_qtPQ is None:
            raise ValueError("gw_evaluate_sigma_with_W: W_c_qtPQ must be provided on the MPI root rank")
        G_tskij = np.asarray(G_tskij, dtype=np.complex128)
        W_c_qtPQ = np.asarray(W_c_qtPQ, dtype=np.complex128)
    else:
        G_tskij = None
        W_c_qtPQ = None
    eps_inv_head = np.asarray(eps_inv_head, dtype=np.complex128)

    arr = gw_evaluate_sigma_with_W_cpp(json.dumps(params), h_int, G_tskij, W_c_qtPQ, eps_inv_head, bool(div_corr))
    return arr if arr.size else None
