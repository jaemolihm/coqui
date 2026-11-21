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

Electronic dielectric tensor from linear response at Q = 0.

Thin wrapper of methods::run_lr_efield_calc: eight Q = 0 LR solves on one self-consistent ground state give
chi2 = chi_bar^(2) and eps_inf = delta - 4 pi Re chi2 (see the C++ docstring for the four phases and the equations).
"""

import json

import h5py
import numpy as np
from mpi4py import MPI

_METHODS = ("none", "Hartree", "HF", "GW0", "GW")


def run_lr_efield(params, h_int, method="GW0", max_iter=50, tol=1e-8, iter_alg=None,
                  reuse_W=False, recompute_W=False,
                  screened_interaction_file=None):
    r"""
    Electronic dielectric tensor eps_inf of an insulator from Q = 0 LR solves.

        eps_inf_ab = delta_ab - 4 pi Re chi2_ab,   chi2 = <G^E_a, V^E_b> + (1/4) Tr[v_a G' v_b G']
                                                          + (1/4) Tr[Sigma'' G v_a G v_b G],

    symmetrized in (a, b), with the field source V^E_a = -(i/2)(v_a G B - B G v_a), B = S - Sigma', G' = -G B G.
    v_a is the velocity matrix of the mean field (pw2coqui ``add_velocity``) dressed by the kernel without its Hartree
    rung. The Hartree rung of ``method`` enters through B and G^E (the macroscopically unscreened response).

    Parameters
    ----------
    params : dict
        ``prefix`` of the ground-state checkpoint and optionally ``output`` (default prefix), ``input_type``,
        ``input_iter``, ``h0_source``, as run_lr.
    h_int : ThcCoulomb
        THC Coulomb integrals of a full-BZ, npol = 1 mean field that carries ``Orbitals/v0_skxij``.
    method : {"none", "Hartree", "HF", "GW0", "GW"}
        Kernel ladder rung. The ground state must have the self-energy of the rung: Sigma = 0 for
        none/Hartree/HF, the GW Sigma for GW0 and GW (checked in C++). "GW" is the full kernel
        K = K^H + K^X + K^W with dW = W dPi W: the G' and G^E_a solves use K, the v_a and Sigma'' solves
        use K^X at fixed W (K^W, and the Hartree rung, are left out of them). "GW0" omits K^W from all solves.
    max_iter, tol, iter_alg
        LR SCF iterations, tolerance and accelerator of every solve, as run_lr. The solves of GW0 and GW are
        preconditioned by HSEX (run_lr's ``precond_method``, default tolerance and sweeps) unless the accelerator
        is DIIS, which cannot take a preconditioner; the preconditioner changes the rate, not the answer.
    reuse_W, recompute_W, screened_interaction_file
        As run_lr. With ``reuse_W`` the W of the call stays cached for the next call
        (``release_lr_W_cache()`` drops it); with ``recompute_W`` and no ``reuse_W`` the W built by this call is
        released at its end; with neither, a W kept by an earlier call is neither read nor changed.

    Returns
    -------
    dict
        The datasets of ``linear_response_efield/`` of ``<output>.mbpt.h5``, the same on every rank: ``eps_inf``,
        ``chi2`` (3, 3); its terms ``chi2_GE_sym``, ``chi2_GE_plain``, ``chi2_vGpvGp``, ``chi2_Sigma2``; the
        diagnostics ``herm_plain``, ``herm_sym``, ``herm_M``, ``eps_inf_imag_max``, ``g_thermal`` (max |tr G(beta/2)| of the ground
        state, not the thermal occupation), ``kappa_bar``,
        ``dkappa_dmu``, ``DeltaDm_Gp_rel``, ``B_minus_S_rel``, ``trace_GE_rel`` (3,), ``niter`` (8,); and
        ``method``, ``tol``, ``precond_method`` ("HSEX" or "none").

    Notes
    -----
    A velocity matrix written before ``unfold_wfc`` rotated the orbitals of a nosym run belongs to other orbitals
    than the ones the mean field holds now; the file cannot tell, so regenerate v0 after any such rewrite.
    """
    if method not in _METHODS:
        raise ValueError(f"run_lr_efield: method must be one of {_METHODS}, got {method!r}.")
    if "prefix" not in params:
        raise ValueError("run_lr_efield: params must contain 'prefix'.")
    if screened_interaction_file is not None and recompute_W:
        raise ValueError("run_lr_efield: screened_interaction_file and recompute_W are mutually exclusive.")

    efield_params = dict(params)
    efield_params["method"] = method
    efield_params["max_iter"] = int(max_iter)
    efield_params["tol"] = float(tol)
    efield_params["iter_alg"] = dict(iter_alg or {})
    efield_params["reuse_W"] = bool(reuse_W)
    efield_params["recompute_W"] = bool(recompute_W)
    if screened_interaction_file is not None:
        efield_params["screened_interaction_file"] = str(screened_interaction_file)

    from coqui._lib.mbpt_module import run_lr_efield as run_lr_efield_cpp
    run_lr_efield_cpp(json.dumps(efield_params), h_int)

    # Rank 0 reads the group back; a failure there becomes a verdict for every rank.
    comm = MPI.COMM_WORLD
    payload = None
    if comm.Get_rank() == 0:
        try:
            out_file = str(params.get("output", params["prefix"])) + ".mbpt.h5"
            with h5py.File(out_file, "r") as f:
                grp = f["linear_response_efield"]
                # complex datasets (flagged by the __complex__ attribute) are stored as (..., 2) reals
                result = {}
                for k in grp.keys():
                    a = grp[k][()]
                    result[k] = a[..., 0] + 1j * a[..., 1] if "__complex__" in grp[k].attrs else a
                result.update({k: (v.decode() if isinstance(v, bytes) else v) for k, v in grp.attrs.items()})
            payload = ("ok", result)
        except Exception as exc:                                      # noqa: BLE001
            payload = ("error", f"{type(exc).__name__}: {exc}")
    payload = comm.bcast(payload, root=0)
    if payload[0] != "ok":
        raise RuntimeError(f"run_lr_efield: reading back linear_response_efield/ failed: {payload[1]}")
    return payload[1]
