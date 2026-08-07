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
    Run a self-consistent Hartree-Fock (HF) calculation. 

    Results are written to an HDF5 checkpoint file at ``outdir/prefix.mbpt.h5``, 
    which can be used for restarting or post-processing. 

    Parameters
    ----------
    params : dict
        Calculation options. Supported keys:

        - ``outdir`` *(str, optional, default ``"./"``)*  — output directory for the
          HDF5 checkpoint file.
        - ``prefix`` *(str, required)* — prefix for the checkpoint filename 
          (written to ``outdir/prefix.mbpt.h5``).
        - ``beta`` *(float, optional, default ``1000.0``, units: 1/Hartree)* —
          inverse temperature. Large values approximate the zero-temperature limit.
        - ``niter`` *(int, optional, default ``1``)* — maximum number of SCF
          iterations.
        - ``conv_thr`` *(float, optional, default ``1e-8``)* — convergence threshold
          on the self-consistency between iterations.
        - ``restart`` *(bool, optional, default ``False``)* — if ``True``, resumes
          from an existing checkpoint file.
        - ``const_mu`` *(bool, optional, default ``False``)* — if ``True``, fixes the
          chemical potential throughout the SCF loop.
        - ``mu_tolerance`` *(float, optional, default ``1e-9``)* — tolerance for the
          chemical potential search.
        - ``div_treatment`` *(str, optional, default ``"gygi"``)* — treatment of the
          Coulomb kernel divergence at q→0. Common choices:

          - ``"gygi"`` *(recommended)* — polynomial extrapolation of ε⁻¹ to q=0
            along the reciprocal lattice directions. The default maximum extrapolation 
            order is 10. Append ``"_order_N"`` to set the polynomial fit order manually 
            (e.g. ``"gygi_order_4"``).
          - ``"gygi_smallest_q"`` — approximates q=0 using the smallest finite
            |q| point; this was the behavior of ``"gygi"`` in earlier versions.
          - ``"ignore_g0"`` — identical to ``"gygi_smallest_q"``; used
            automatically when only one q-point is available.

          Suffix modifiers (combinable, e.g. ``"gygi_2d"`` or ``"gygi_metal"``):

          - ``"_2d"`` — restricts the extrapolation to the xy-plane; use for 2D
            materials.
          - ``"_metal"`` — enforces ε⁻¹(q=0, ω=0) → 0; use for metallic systems.

        - ``hf_div_treatment`` *(str, optional, default ``"gygi"``)* — divergence
          treatment for the HF Coulomb kernel specifically. Accepts the same values
          and suffix modifiers as ``div_treatment``.
        - ``greens_func_source`` *(str, optional, default ``"scf"``)* — source of the
          initial Green's function. ``"mf"`` uses the mean-field Green's function;
          ``"scf"`` reads from the checkpoint.
        - ``greens_func_iteration`` *(int, optional, default ``-1``)* — checkpoint
          iteration to read the Green's function from. ``-1`` selects the latest.
        - ``h0_source`` *(str, optional, default ``"compute"``)* — source of the H0
          matrix. ``"compute"`` calculates H0 from the plane-wave orbitals;
          ``"checkpoint"`` reads it from ``outdir/prefix.mbpt.h5``.
        - ``compute_exchange`` *(bool, optional, default ``True``)* — whether to
          compute the exchange (K) term in the Fock matrix. Set to ``False`` for
          Hartree-only calculations.
        - ``dump_exchange`` *(bool, optional, default ``False``)* — if ``True``,
          additionally writes the exchange-only Fock matrix ``K_skij`` to the
          checkpoint (``scf/iter{N}/K_skij``) alongside the full Fock
          ``F_skij`` (= J + K), for every iteration ``F_skij`` is written. The
          stored ``F_skij`` is unchanged. Useful downstream for the bare-exchange
          static self-energy correction ``K - Vxc``.
        - ``iaft`` *(dict, optional)* — imaginary-axis frequency-grid settings:

          - ``wmax`` *(float, optional, units: Hartree)* — maximum frequency in Hartree. If omitted,
            estimated automatically from the mean-field bandwidth.
          - ``prec`` *(str, optional, default ``"medium"``)* — target numerical
            precision: ``"high"`` (~1e-15), ``"medium"`` (~1e-10), ``"low"`` (~1e-6).
          - ``basis`` *(str, optional, default ``"dlr"``)* — frequency basis.
            ``"dlr"`` (Discrete Lehmann Representation) or ``"ir"`` (intermediate
            representation).
          - ``eps`` *(float, optional)* — explicit accuracy target; alternative to
            ``prec``.

        - ``iter_alg`` *(dict, optional)* — SCF mixing algorithm settings:

          - ``alg`` *(str, optional, default ``"diis"``)* — mixing algorithm.
            ``"damping"`` (linear mixing) or ``"diis"`` (Pulay DIIS).
          - ``mixing`` *(float, optional, default ``0.7``)* — mixing fraction in
            ``[0, 1]`` for damping; smaller values give more conservative updates.
          - ``max_subsp_size`` *(int, optional, default ``5``)* — maximum DIIS
            subspace size (ignored for ``"damping"``).
          - ``residual_type`` *(str, optional, default ``"commutator"``)* — residual
            definition for DIIS. ``"commutator"`` or ``"vector_diff"``.

    h_int : ThcCoulomb or CholCoulomb
        Coulomb interaction object used for both the Hartree and exchange terms.
        Obtained from ``make_thc_coulomb`` or ``make_chol_coulomb``.
    h_int_exchange : ThcCoulomb or CholCoulomb, optional
        If provided, used for the exchange term only; ``h_int`` is then used for
        the Hartree term only. Useful when Hartree and exchange are computed with
        different interaction objects 

    Returns
    -------
    None
        Results are written to ``outdir/prefix.mbpt.h5``.

    Examples
    --------
    ::

        from coqui.mbpt import run_hf

        run_hf(
            {"beta": 300, "niter": 10, "prefix": "svo.hf",
             "iaft": {"prec": "medium"},
             "iter_alg": {"alg": "damping", "mixing": 0.7}},
            h_int=thc,
        )
    """
    args = ["hf", json.dumps(params), h_int]
    if h_int_exchange is not None:
        args.append(h_int_exchange)
    mbpt_cxx(*args)


def run_gw(params, h_int,
           h_int_hf = None, h_int_hartree = None, h_int_exchange = None,
           *, projector_info = None, local_polarizabilities = None):
    """
    Run a full-frequency self-consistent GW calculation.

    Computes the GW self-energy using the screened Coulomb interaction W and
    writes results to ``outdir/prefix.mbpt.h5``. Set ``niter=1`` for G0W0.

    Parameters
    ----------
    params : dict
        Accepts all keys documented in ``run_hf``, plus:

        - ``screen_type`` *(str, optional, default ``"rpa"``)* — approximation used
          to build the screened interaction W. Common choices:

          - ``"rpa"`` — random-phase approximation; standard choice for GW.
          - ``"crpa"`` — constrained RPA; excludes selected (e.g. correlated) bands
            from the polarizability, typically used when constructing low-energy
            downfolded models. This requires ``projector_info`` to be provided for 
            defining the correlated subspace. 
          - ``"gw_edmft"`` — GW+EDMFT; requires ``projector_info`` and
            ``local_polarizabilities`` to be provided so that the local EDMFT
            polarizability is added on top of the RPA contribution within the 
            correlated subspace.
        - ``dump_w_to_h5`` *(bool, optional, default ``False``)* — if ``True``,
          saves the screened interaction W to the checkpoint file.

    h_int : ThcCoulomb or CholCoulomb
        Primary Coulomb interaction object, used for both the polarizability and
        the GW self-energy when no split ERI arguments are provided.
    h_int_hf : ThcCoulomb or CholCoulomb, optional
        If provided, used for the HF (Hartree + exchange) part of the self-energy
        instead of ``h_int``. Mutually exclusive with ``h_int_hartree`` /
        ``h_int_exchange``.
    h_int_hartree : ThcCoulomb or CholCoulomb, optional
        Coulomb object for the Hartree channel. Must be paired with
        ``h_int_exchange``; cannot be combined with ``h_int_hf``.
    h_int_exchange : ThcCoulomb or CholCoulomb, optional
        Coulomb object for the exchange channel. Must be paired with
        ``h_int_hartree``; cannot be combined with ``h_int_hf``.
    projector_info : dict, optional
        Wannier projector data for defining corrleated subspace. Required keys:
        ``"proj_mat"``, ``"band_window"``, ``"kpts_w90"``. 
    local_polarizabilities : dict, optional
        Local polarizabilities for local EDMFT corrections. 
        Required keys: ``"imp"`` (impurity) and ``"dc"`` (double-counting). 
        Only used together with ``projector_info``.

    Returns
    -------
    None
        Results are written to ``outdir/prefix.mbpt.h5``.

    Examples
    --------
    ::

        from coqui.mbpt import run_gw

        run_gw(
            {"beta": 200, "niter": 1, "prefix": "svo.gw",
             "iaft": {"prec": "medium"},
             "screen_type": "rpa",
             "div_treatment": "gygi"},
            h_int=thc,
        )
    """
    _run_mbpt("gw", params, h_int,
              h_int_hf = h_int_hf, h_int_hartree = h_int_hartree, h_int_exchange = h_int_exchange,
              projector_info = projector_info, local_polarizabilities = local_polarizabilities)


def run_evgw(params, h_int,
               h_int_hf = None, h_int_hartree = None, h_int_exchange = None):
    """
    Run an eigenvalue-only self-consistent GW (evGW or evGW0) calculation.

    Updates quasiparticle energies self-consistently while keeping the
    wavefunctions fixed at the DFT level. 

    Parameters
    ----------
    params : dict
        Accepts all keys documented in ``run_hf``, plus:

        - ``keep_scr_coulomb_fixed`` *(bool, optional, default ``False``)* — if
          ``True``, the screened Coulomb interaction W is held fixed at the 
          first iteration (evGW0); if ``False``, W is updated each iteration (evGW).
        - ``ac_alg`` *(str, optional, default ``"pade"``)* — analytic continuation
          algorithm used to extract quasiparticle energies from the Matsubara
          self-energy. Currently only ``"pade"`` is implemented. 
        - ``eta`` *(float, optional, default ``π/beta``, units: Hartree)* —
          broadening for the analytic continuation.
        - ``Nfit`` *(int, optional, default ``18``)* — number of Matsubara
          frequencies used for the Padé fit. Set ``-1`` to use all Matsubara frequencies 
          on a DLR mesh. 

    h_int : ThcCoulomb or CholCoulomb
        Primary Coulomb interaction object.
    h_int_hf : ThcCoulomb or CholCoulomb, optional
        If provided, used for the HF part instead of ``h_int``. Mutually
        exclusive with ``h_int_hartree`` / ``h_int_exchange``.
    h_int_hartree : ThcCoulomb or CholCoulomb, optional
        Coulomb object for the Hartree channel. Must be paired with
        ``h_int_exchange``.
    h_int_exchange : ThcCoulomb or CholCoulomb, optional
        Coulomb object for the exchange channel. Must be paired with
        ``h_int_hartree``.

    Returns
    -------
    None
        Results are written to ``outdir/prefix.mbpt.h5``.

    Examples
    --------
    ::

        from coqui.mbpt import run_evgw

        run_evgw(
            {"beta": 200, "niter": 5, "prefix": "svo.evgw",
             "iaft": {"prec": "medium"},
             "keep_scr_coulomb_fixed": False},
            h_int=thc,
        )
    """
    _run_mbpt("evgw", params, h_int,
              h_int_hf = h_int_hf, h_int_hartree = h_int_hartree, h_int_exchange = h_int_exchange,
              projector_info = None, local_polarizabilities = None)


def run_qpgw(params, h_int,
               h_int_hf = None, h_int_hartree = None, h_int_exchange = None):
    """
    Run a quasiparticle self-consistent GW (qpGW) calculation.

    Updates both quasiparticle energies and wavefunctions self-consistently by
    constructing a Hermitian, energy-independent effective Hamiltonian from the
    GW self-energy at each iteration.

    Parameters
    ----------
    params : dict
        Accepts all keys documented in ``run_hf``, plus:

        - ``ac_alg`` *(str, optional, default ``"pade"``)* — analytic continuation
          algorithm used to extract quasiparticle energies from the Matsubara
          self-energy.
        - ``eta`` *(float, optional, default ``π/beta``, units: Hartree)* —
          broadening for the analytic continuation.
        - ``Nfit`` *(int, optional, default ``18``)* — number of Matsubara
          frequencies used for the Padé fit. Set ``-1`` to use all Matsubara 
          frequencies on a DLR mesh. 
        - ``off_diag_mode`` *(str, optional, default ``"fermi"``)* — frequency at
          which off-diagonal self-energy matrix elements are evaluated when
          constructing the qpGW Hamiltonian. ``"fermi"`` evaluates at the Fermi
          level; ``"qp_energy"`` evaluates at the quasiparticle energy of each
          state.

    h_int : ThcCoulomb or CholCoulomb
        Primary Coulomb interaction object.
    h_int_hf : ThcCoulomb or CholCoulomb, optional
        If provided, used for the HF part instead of ``h_int``. Mutually
        exclusive with ``h_int_hartree`` / ``h_int_exchange``.
    h_int_hartree : ThcCoulomb or CholCoulomb, optional
        Coulomb object for the Hartree channel. Must be paired with
        ``h_int_exchange``.
    h_int_exchange : ThcCoulomb or CholCoulomb, optional
        Coulomb object for the exchange channel. Must be paired with
        ``h_int_hartree``.

    Returns
    -------
    None
        Results are written to ``outdir/prefix.mbpt.h5``.

    Examples
    --------
    ::

        from coqui.mbpt import run_qpgw

        run_qpgw(
            {"beta": 200, "niter": 5, "prefix": "svo.qpgw",
             "iaft": {"prec": "medium"},
             "off_diag_mode": "fermi"},
            h_int=thc,
        )
    """
    _run_mbpt("qpgw", params, h_int,
              h_int_hf = h_int_hf, h_int_hartree = h_int_hartree, h_int_exchange = h_int_exchange,
              projector_info = None, local_polarizabilities = None)

def run_lr(params, h_int, q_vec, DeltaH0_skij,
           include_hartree=True, include_exchange=True,
           gw_mode="none",
           max_iter=50, tol=1e-8, fix_density=False, iter_alg=None,
           include_gw_sigma=None,
           DeltaX_left=None, DeltaX_right=None,
           DeltaV_qPQ=None,
           div_corr=True,
           screened_interaction_file=None,
           recompute_W=False,
           unperturbed="checkpoint",
           split_sigma_terms=False,
           div_treatment=None):
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
        - output: Output checkpoint prefix (default: same as prefix).
          Must refer to an existing .mbpt.h5 checkpoint; results are appended.
        - input_type: HDF5 group to read checkpoint from (default: "scf")
        - input_iter: Iteration number to read (default: -1 = use final_iter)
        - include_xc: LR-DFT. Add the semilocal xc kernel to the direct
          (Hartree) channel, i.e. use (V + Vxc)(q) in ΔJ (default False).
          Works only in the Hartree mode: requires ``include_hartree=True``,
          ``include_exchange=False``, ``gw_mode="none"``, and a ``h_int`` built
          with the THC ``Vxc_file`` option; each is a hard error otherwise.
          (f_xc and ΔΣ_GW both carry the correlation response, so combining them
          would double-count it.) It is an explicit flag rather than being keyed
          off the presence of Vxc in the THC file, so a Hartree run against a
          Vxc-carrying THC stays a Hartree run.
        - output_aux_fock: Also write DeltaF_ibc_skij and the aux-basis (THC)
          Fock matrices F_PQ_skij / DeltaF_PQ_skij to "linear_response/"
          (default False). Only the DeltaX/IBC ddF curvature post-processors read
          them. The two _PQ arrays are (nspin, nkpts_ibz, Np, Np) — ~13 GB at
          Np = 3.7k — and capturing DeltaF_PQ costs one extra Fock build on the
          converged DeltaDm, so a run that does not do IBC should leave this off.
          npol > 1 is not supported for the _PQ output.
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
    screened_interaction_file : str or None, optional
        Explicit path to the screened-interaction HDF5 file (W_qtPQ). Used only
        when gw_mode != "none" and recompute_W is False. If None (default), W is
        read from thc_screened_interaction.h5 in the input checkpoint's
        directory. Ignored when recompute_W is True.
    recompute_W : bool, optional
        If True, recompute W on the fly from the checkpoint Green's function
        (RPA: Π = eval_Pi_qdep(G), W_c = dyson_W_from_Pi_tau(Π)) instead of
        reading it from disk. Guarantees W is consistent with the currently-
        loaded THC. Standard GW (RPA) only; requires the no-symmetry (full-q)
        ground state. Default False.
    unperturbed : str, optional
        Unperturbed reference (default "checkpoint"):
        - "checkpoint": interacting G rebuilt from the checkpoint F+Σ (scGW LR)
        - "mf_dft": DFT/KS mean-field G0 built directly from the mean-field
          eigenvalues/orbitals (one-shot G0W0@DFT). W0 is then the RPA screened
          interaction from G0 (recompute_W is forced on) and the checkpoint is
          used only for the IAFT grid + metadata.
    split_sigma_terms : bool, optional
        If True, store the two one-shot ΔΣ terms separately (default False):
        DeltaSigma_tskij = term 1 (dG0·W_c0), DeltaSigma_term2_tskij = term 2
        (G0·dW0). Requires gw_mode="full" and max_iter=1. See run_lr_g0w0.
    div_treatment : str or None, optional
        Divergence treatment for the ε⁻¹ head. For unperturbed="mf_dft" the
        checkpoint holds none, so this selects it (default "gygi" in C++).
        Ignored for unperturbed="checkpoint" (read from the checkpoint).

    Returns
    -------
    tuple[int, float]
        (niter, Delta_mu) - number of iterations and final chemical potential shift

    Notes
    -----
    Results are written to {output}.mbpt.h5 under the "linear_response" group.
    """
    import numpy as np
    from coqui._lib.mbpt_module import run_lr as run_lr_cpp

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

    if recompute_W and screened_interaction_file is not None:
        raise ValueError(
            "run_lr: screened_interaction_file and recompute_W are mutually "
            "exclusive. Pass a W file path to read W from disk, or recompute_W=True "
            "to recompute it from the checkpoint Green's function.")

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
    # Optional explicit path to the screened-interaction (W) HDF5 file. When
    # omitted, C++ auto-derives it from the input checkpoint's directory.
    if screened_interaction_file is not None:
        params_with_div["screened_interaction_file"] = str(screened_interaction_file)
    # Recompute W from the checkpoint Green's function instead of reading it.
    params_with_div["recompute_W"] = bool(recompute_W)
    # Unperturbed reference and split-term output (one-shot G0W0@DFT).
    params_with_div["unperturbed"] = str(unperturbed)
    params_with_div["split_sigma_terms"] = bool(split_sigma_terms)
    if div_treatment is not None:
        params_with_div["div_treatment"] = str(div_treatment)

    return run_lr_cpp(json.dumps(params_with_div), h_int, q_vec, DeltaH0_skij,
                      bool(include_hartree), bool(include_exchange), str(gw_mode),
                      int(max_iter), float(tol), bool(fix_density),
                      alg, mixing, max_subsp_size, diis_warmup,
                      dx_left, dx_right, dv_qPQ)

def run_lr_g0w0(params, h_int, q_vec, DeltaH0_skij, div_corr=True, div_treatment=None):
    """
    One-shot G0W0@DFT electron-phonon linear response.

    Computes the one-shot response of the G0W0 self-energy to a perturbation,
    on top of a DFT/KS mean-field reference:

        G0  = DFT/KS Green's function (built from the mean-field eigenvalues/orbitals)
        W0  = RPA screened interaction from G0
        dG0 = G0 · ΔH0 · G0
        dΣ  = dG0·W0 + G0·dW0,   dW0 = (Z+W_c0)·dΠ0·(Z+W_c0),  dΠ0 = -dG0·G0 - G0·dG0

    This is a thin wrapper over run_lr that pins the one-shot G0W0 configuration:
    unperturbed="mf_dft", gw_mode="full", max_iter=1, and split-term output. The
    Hartree response is excluded (include_hartree=False) because the perturbation
    ΔH0 is expected to be the DFT-screened potential ΔH_KS (= g_scr), which already
    contains the DFT Hartree/xc screening; the exchange response is kept
    (include_exchange=True) so that, together with ΔΣ term 1, it reconstitutes the
    full-W term dG0·W0. The DFT exchange-correlation should be subtracted downstream.

    Output in {output}.mbpt.h5 / "linear_response":
        DeltaSigma_tskij     = total correlation ΔΣ = dG0·W_c0 + G0·dW0
        DeltaSigma_GdW_tskij = the G0·dW0 piece, broken out separately
        DeltaF_skij          = exchange ΔK (= -dG0·v, the v part of dG0·W0)
    so dG0·W_c0 = DeltaSigma_tskij - DeltaSigma_GdW_tskij, and the full
    dG0·W0 = DeltaF_skij + (DeltaSigma_tskij - DeltaSigma_GdW_tskij).

    Parameters
    ----------
    params : dict
        Parameters including prefix/output/input_type/input_iter (see run_lr).
        The checkpoint is used only for the IAFT grid + metadata; G0 is rebuilt
        from the mean field.
    h_int : ThcCoulomb
        THC ERI handler. Requires the no-symmetry (full-q) ground state.
    q_vec : array-like
        Perturbation wavevector in crystal coordinates, shape (3,).
    DeltaH0_skij : np.ndarray or None
        DFT-screened perturbation ΔH_KS (= g_scr), shape (ns, nk, nb, nb).
        Required on the MPI global root; ignored on non-root ranks.
    div_corr : bool, optional
        Apply the Madelung/head divergence corrections to ΔΣ (default True).
        Kept as an opt-out for experimentation; production G0W0 wants True.
    div_treatment : str or None, optional
        q→0 divergence scheme for the ε⁻¹ head (default "gygi" in C++). Selects
        how W0's head is built (and, when div_corr, how the LR head correction
        is applied); relevant even when div_corr=False.

    Returns
    -------
    tuple[int, float]
        (niter, Delta_mu). One-shot performs a single LR iteration.
    """
    return run_lr(params, h_int, q_vec, DeltaH0_skij,
                  include_hartree=False, include_exchange=True,
                  gw_mode="full", max_iter=1,
                  unperturbed="mf_dft", split_sigma_terms=True,
                  div_corr=div_corr, div_treatment=div_treatment)

def run_lr_qpgw(params, h_int, q_vec, DeltaH0_skij,
                gw_mode="full", max_iter=50, tol=1e-8,
                fix_density=False, iter_alg=None):
    """
    Quasiparticle-GW (qpGW) electron-phonon linear response.

    Linear response of the static qpGW effective Hamiltonian
    H_eff = H0 + F[Dm] + V_QPGW to a one-body perturbation ΔH0, on top of a
    converged qpGW checkpoint. Each LR iteration:

        ΔG_QP = G_QP · [ΔH0 + ΔF + ΔV_QPGW - Δμ·S] · G_QP     (frozen G_QP)
        ΔF     = lr_hf(ΔDm)                                   (Hartree + exchange)
        ΔΣ(iω) = -ΔG_QP⊙W0 - G_QP⊙ΔW                          (gw_mode="full")
        ΔV_QPGW = lr_qp_approx(ΔΣ; frozen C, ε, μ, kpq_map)   (q-aware static map)

    The frozen QP orbitals/energies (C, ε, μ) are reconstructed from the
    checkpoint's Heff = H0 + F; W0 is the RPA screened interaction from G_QP
    (recompute_W is forced on). The static ΔV_QPGW is the DIIS/damping-mixed and
    convergence-tracked quantity; the dynamic ΔΣ is not in the Dyson RHS.

    Adopted approximations (both quantified by the FD test, see docs):
      - Frozen-orbital linearization of the static map (drop ΔC, Δε inside
        lr_qp_approx; the full QP-state response is still kept exactly in G_QP
        via the two-sided resolvent).
      - The Padé analytic continuation is nonlinear in its input, so feeding ΔΣ
        through it gives Padé[ΔΣ], not the consistent d/dλ(Padé[Σ]).

    Parameters
    ----------
    params : dict
        prefix/output/input_type/input_iter (see run_lr). Must point at a
        converged qpGW checkpoint.
    h_int : ThcCoulomb
        THC ERI handler. Requires the no-symmetry (full-q) ground state.
    q_vec : array-like
        Perturbation wavevector in crystal coordinates, shape (3,).
    DeltaH0_skij : np.ndarray or None
        Perturbation, shape (ns, nk, nb, nb). Required on the MPI global root.
    gw_mode : str, optional
        Screening response of ΔΣ: "full" (ΔW responds via lr_rpa_pi, default) or
        "fixed_W" (W frozen). Match the unperturbed qpGW run.
    max_iter : int, optional
        Max LR SCF iterations (default 50 = self-consistent). Use 1 for the
        one-shot/linearized variant.
    tol : float, optional
        Convergence tolerance on ||ΔV_QPGW_new - ΔV_QPGW_old|| (default 1e-8).
    fix_density : bool, optional
        Enforce ΔN=0 via Δμ (q=0 only). Default False.
    iter_alg : dict or None, optional
        Iteration algorithm (damping/DIIS); see run_lr.

    Notes on parameters
    -------------------
    The AC parameters (ac_alg, Nfit, eta, off_diag_mode) and div_treatment are
    NOT accepted here: the LR run always reuses exactly what the ground-state
    qpGW run used, read from the checkpoint ("scf/qp_params" and
    "scf/div_treatment", written by run_qpgw). Re-run qpGW if the checkpoint
    predates this and does not carry them.

    Returns
    -------
    tuple[int, float]
        (niter, Delta_mu).

    Notes
    -----
    Output in {output}.mbpt.h5 / "linear_response": DeltaVcorr_skij (static
    ΔV_QPGW) alongside ΔG/ΔDm/ΔF and the (informational) dynamic ΔΣ.
    """
    p = dict(params)
    p["qp_static_sigma"] = True
    if gw_mode not in ("fixed_W", "full"):
        raise ValueError(f"run_lr_qpgw: gw_mode must be 'fixed_W' or 'full', got '{gw_mode}'.")
    return run_lr(p, h_int, q_vec, DeltaH0_skij,
                  include_hartree=True, include_exchange=True,
                  gw_mode=gw_mode, max_iter=max_iter, tol=tol,
                  fix_density=fix_density, iter_alg=iter_alg,
                  unperturbed="checkpoint")

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
    from coqui._lib.mbpt_module import lr_hf as lr_hf_cpp

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


def lr_qp_approx(h_int, prefix, DeltaSigma_tskij, MO_skia, E_ska, mu,
                 kpq_map, q_is_gamma, off_diag_mode="qp_energy",
                 ac_alg="pade", Nfit=18, eta=1e-3):
    """
    Statify a dynamic LR self-energy ΔΣ into a static ΔV_QPGW (LR-qpGW map).

    Thin wrapper over the C++ methods::lr_qp_approx (see the C++ docstring). The
    IAFT grid is read from the qpGW checkpoint {prefix}.mbpt.h5; the frozen QP
    eigenbasis (MO_skia, E_ska), μ, k→k+q map, and AC parameters are provided by
    the caller. Intended for testing the static map from Python.

    Parameters
    ----------
    h_int : ThcCoulomb
        THC ERI handler (source of MPI + mean field).
    prefix : str
        Checkpoint prefix; the IAFT is read from {prefix}.mbpt.h5.
    DeltaSigma_tskij : np.ndarray
        Dynamic ΔΣ_k(τ), shape (nt, ns, nk, nb, nb), complex.
    MO_skia : np.ndarray
        Frozen QP MO coefficients C, shape (ns, nk, nb, nb), complex.
    E_ska : np.ndarray
        Frozen QP energies ε, shape (ns, nk, nb), complex.
    mu : float
        Frozen chemical potential.
    kpq_map : np.ndarray
        k → k+q index map, shape (nk,), int. Build via coqui.mbpt.calculate_kpq_map.
    q_is_gamma : bool
        Whether q ≈ 0 (enables the q=0 Hermitization branch).
    off_diag_mode : str, optional
        "qp_energy" (default) or "fermi". Match the qpGW run.
    ac_alg : str, optional
        Analytic-continuation algorithm (default "pade").
    Nfit : int, optional
        Number of AC fit parameters (default 18).
    eta : float, optional
        AC broadening (default 1e-3). Pass the value the qpGW run used (π/β).

    Returns
    -------
    np.ndarray
        Static ΔV_QPGW(k) in the primary basis, shape (ns, nk, nb, nb).
    """
    import numpy as np
    from coqui._lib.mbpt_module import lr_qp_approx as lr_qp_approx_cpp

    DeltaSigma_tskij = np.asarray(DeltaSigma_tskij, dtype=np.complex128)
    MO_skia = np.asarray(MO_skia, dtype=np.complex128)
    E_ska = np.asarray(E_ska, dtype=np.complex128)
    kpq_map = np.asarray(kpq_map, dtype=np.int64)

    return lr_qp_approx_cpp(h_int, str(prefix), DeltaSigma_tskij, MO_skia, E_ska,
                            float(mu), kpq_map, bool(q_is_gamma),
                            str(off_diag_mode), str(ac_alg), int(Nfit), float(eta))


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
    from coqui._lib.mbpt_module import hf_evaluate as hf_evaluate_cpp

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
    from coqui._lib.mbpt_module import lr_gw_sigma_DeltaG as lr_gw_sigma_DeltaG_cpp

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
    from coqui._lib.mbpt_module import lr_gw_Pi as lr_gw_Pi_cpp

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
    from coqui._lib.mbpt_module import gw_evaluate_Pi as gw_evaluate_Pi_cpp

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
    from coqui._lib.mbpt_module import gw_evaluate_sigma as gw_evaluate_sigma_cpp

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
    from coqui._lib.mbpt_module import lr_gw_W as lr_gw_W_cpp

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
    from coqui._lib.mbpt_module import gw_evaluate_W_from_Pi as gw_evaluate_W_from_Pi_cpp

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
    from coqui._lib.mbpt_module import lr_gw_sigma_DeltaW as lr_gw_sigma_DeltaW_cpp

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
    from coqui._lib.mbpt_module import compute_eps_inv_head as compute_eps_inv_head_cpp

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
    from coqui._lib.mbpt_module import gw_evaluate_sigma_with_W as gw_evaluate_sigma_with_W_cpp

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
