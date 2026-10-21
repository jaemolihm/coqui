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
import os

import h5py
import numpy as np
from h5 import HDFArchive
from mpi4py import MPI

import triqs_modest as modest

import coqui
import coqui.mbpt
from coqui import app_log
from coqui.utils.h5_utils import write_complex_dataset
from coqui.mbpt.lr_driver import _as_complex

from .dmft_state import DMFTState, _mix_into
from .impurity_block_map import ImpurityBlockMap
from .io import convert_gw_edmft_params, read_impurity_chkpt
from .lr_weiss import lr_compute_weiss_fields_w
from .triqs_utils import blk_arr_to_arr, arr_to_blk_arr
from .weiss import chemistry_to_product_basis, compute_weiss_fields_w, embed_impurities, get_proj_info

__all__ = ["run_lr_gw_edmft"]

_KERNEL_KEYS = {"include_hartree": True, "include_exchange": True, "max_iter": 200, "tol": 1e-10,
                "iter_alg": {"alg": "GCR"}, "precond_method": "none", "save_DeltaG": True}
_DRIVER_KEYS = {"tol_imp": 1e-8, "max_outer": 30, "anderson_history": 5, "mixing": 1.0, "output": None}

def run_lr_gw_edmft(h_int, embedding, *, proj_info=None, params: dict, q_vec, DeltaH0_skij,
                    impurity_response, lr_params=None):
    """
    Linear response of a GW+EDMFT ground state: the outer (impurity) loop of the self-consistent LR GW+EDMFT.

        run_lr (lattice + double counting, frozen impurity response y) -> delta G^loc, delta W^loc
        -> Weiss-field responses -> ``impurity_response`` -> y' ; Anderson on y until y' = y.

    ``run_lr`` solves lattice and double counting together (the ground state is given by ``projector_info`` and
    ``local_polarizabilities``), so the loop iterates the impurity response y = (delta Sigma_imp, delta Vhf_imp,
    delta Pi_imp) alone, passed to ``run_lr`` as ``lr_impurity_response`` (dynamic Sigma and Pi; Vhf enters only the Weiss fields,
    corr_only). Every call after the first runs with
    ``DeltaH0_skij = 0`` and the increment of y: the local responses are accumulated, and one final full solve with
    the converged y gives the standard LR output.

    The ground state is read, not passed. ``params`` is the ``run_gw_edmft`` dictionary: it fixes the checkpoints
    (``outdir``, ``prefix``, ``edmft.chkpt_h5``), the projector (``wannier_file``, unless ``proj_info`` is given),
    ``screen_type`` and ``mu``, and is checked against the linearization (below). The linearization point is
        - G (and F, Sigma, mu): ``embed/iter{J+1}`` of the coqui checkpoint, J = ``dmft/final_iter`` of the impurity
          checkpoint (the embedding of the impurity results of dmft iteration J);
        - Sigma_imp, Vhf_imp, Pi_imp: the (damped) results of dmft iteration J, which were embedded;
        - G^loc, H^loc, W^loc, V^loc: computed by the first ``run_lr`` call from its own G, F, mu and W
          (``Gloc_tsIab``, ``Hloc_sIab``, ``Wloc_tabcd``, ``Vloc_abcd`` of ``linear_response``), the factors of its
          double counting; H^loc = P (H0 + F) P^dagger - mu is the static local one-body term of G^loc. The Weiss
          fields follow from them (with the pointwise G^loc(iw) = P <G_k(iw)> P^dagger, ``Gloc_wsIab``) and the
          impurity results.
    The distance of G^loc from the stored dmft input of iteration J (one embedding earlier) is logged.

    Parameters
    ----------
    h_int : ThcCoulomb
    embedding : triqs_modest.embedding.Embedding
        As for ``run_gw_edmft``. One impurity site per cell (single projector impurity). The responses are
        extracted with the representative (first) site of each impurity block, which is exact while the
        perturbation keeps the sites of a block equivalent.
    proj_info : dict, optional
        Projector metadata; ``params["wannier_file"]`` if None.
    params : dict
        The ``run_gw_edmft`` parameter dictionary. Checked: ``corr_only = True``, ``screen_type`` one of
        ``gw_edmft`` / ``gw_edmft_density`` (``rpa`` is not supported), and, per impurity solver set, no
        ``causal_projection`` and ``set_sigma_infty_to_dc = False`` (the response of the Weiss maps is defined
        for these only). A fixed lattice ``mu`` gives ``fix_density = False``; otherwise the density is
        conserved at Q = 0 (``fix_density = True``) and not at Q != 0, the only distinction made on Q.
    q_vec : array-like (3,)
        Momentum transfer, crystal coordinates.
    DeltaH0_skij : np.ndarray (ns, nk, nb, nb)
        Perturbation, required on the MPI root.
    impurity_response : callable
        ``impurity_response(imp_index, dg_weiss_w, du_weiss_w, dh0, ddelta_w, *, ground_state) -> dict``,
        called collectively on all ranks, once per impurity and outer iteration, with identical arguments on all
        ranks. The Weiss-field responses are in the layouts of the ground-state impurity inputs:
            dg_weiss_w  (nw_f, 2, n, n)         fermionic Weiss field response, spin blocks up/down
            du_weiss_w  (nw_b_half, n, n, n, n) bosonic Weiss field response, product basis
            dh0         (2, n, n)               response of h0 = H^loc - Vhf_imp
            ddelta_w    (nw_f, 2, n, n)         response of the hybridization (run_lr's DeltaDelta_wsIab)
        The responses are complex at Q != 0 (the perturbation is Hermitian and time-reversal even, so
        dX^Q_ab = dX^Q_ba is symmetric and complex; real at Q = 0). A solver that is real-only solves Re dX and
        Im dX separately, each a real symmetric perturbation, and recombines by linearity.
        The perturbation must be time-reversal even and Hermitian: run_lr projects dDelta (P at Q = 0, Sym at Q != 0)
        but returns dHloc unprojected, and dg_weiss is the derivative of the ground-state map only if the projection
        leaves dHloc unchanged.
        ``ground_state`` is a dict of the unperturbed impurity quantities in the same layouts (``Gloc_t`` with the
        shape (nt_f, 2, n, n), ``Gloc_w`` (nw_f, 2, n, n), ``Wloc_t`` and ``Vloc`` in the product basis, ``g_weiss_w``, ``u_weiss_w``,
        ``Hloc``, ``h0``, ``delta_w``, ``Vhf_imp``, ``Sigma_imp_w``, ``Pi_imp_w``, ``gf_struct``, ``iaft``). It returns
            {"Vhf_imp": (2, n, n), "Sigma_imp_w": (nw_f, 2, n, n), "Pi_imp_w": (nw_b_half, n, n)}
        the response of the static and dynamic impurity self-energy and of the density-density polarizability,
        at fixed impurity chemical potential (the ground state fixes mu_imp; a finite-difference impurity must hold
        it fixed).
    lr_params : dict, optional
        Kernel keys passed to ``run_lr`` (defaults): ``include_hartree`` (True), ``include_exchange`` (True),
        ``max_iter`` (200), ``tol`` (1e-10), ``iter_alg`` ({"alg": "GCR"}), ``precond_method`` ("none"),
        ``save_DeltaG`` (True; the final solve only). ``gw_mode`` is "full" and ``recompute_W`` True.
        Driver keys: ``tol_imp`` (1e-8; relative change of y per outer iteration), ``max_outer`` (30),
        ``anderson_history`` (5; 0 is plain iteration, mixed linearly with ``mixing``), ``mixing`` (1.0),
        ``output`` (prefix of the output checkpoint, default ``{outdir}/{prefix}_lr_gw_edmft``).

    Returns
    -------
    dict
        ``converged``, ``n_outer``, ``residuals`` (relative change of y per outer iteration), ``niter`` (kernel
        iterations of every run_lr call), ``dGloc_tsIab`` / ``dWloc_tabcd`` (local responses of the final solve),
        ``impurity_response`` (the converged response of every impurity), ``output`` (path).
        The file ``{output}.mbpt.h5`` holds the standard LR output of the final solve under ``linear_response``
        and, under ``lr_gw_edmft/iter{n}``, the local responses, the Weiss-field responses and the impurity
        responses of every outer iteration.
    """
    comm = MPI.COMM_WORLD
    root = comm.Get_rank() == 0
    q_vec = np.asarray(q_vec, dtype=np.float64)
    is_gamma = bool(np.all(np.abs(q_vec - np.round(q_vec)) < 1e-10))
    kernel_kw, drv = _split_lr_params(lr_params)

    if not isinstance(embedding, modest.embedding.Embedding):
        raise TypeError(f"Expected a modest.embedding object for 'embedding', got {type(embedding)}")
    params = convert_gw_edmft_params(params)
    _check_ground_state_flags(params)
    if proj_info is None:
        obe = modest.make_one_body_elements_gw(params["wannier_file"])
        proj_info = get_proj_info(obe.P)

    mf = h_int.mf()
    wloc_params = params["wloc"]
    embed_params, impurity_params = params["dmft_embed"], params["impurity"]
    outdir, prefix = embed_params["outdir"], embed_params["prefix"]
    coqui_chkpt = f"{outdir}/{prefix}.mbpt.h5"
    solver_chkpt = impurity_params.get("chkpt_h5", coqui_chkpt)
    imp_iaft = impurity_params.get("iaft", {})
    fix_density = embed_params.get("mu") is None and is_gamma
    output = drv["output"] or f"{outdir}/{prefix}_lr_gw_edmft"

    # ---- ground state: impurity results and the embedding iteration they were embedded into
    embedding_2e = embedding.merge_embed_block_by_imp.slice_sigma
    state = DMFTState.make_dmft_state(
        coqui_chkpt, embedding, embedding_2e, wmax_imp=imp_iaft.get("wmax"), eps_imp=imp_iaft.get("eps"),
        spin_average=mf.nspin() == 1, screen_type=wloc_params["screen_type"], verbal=root)
    state.load(solver_chkpt)
    if state.local_pi_w is None:
        raise FileNotFoundError(f"run_lr_gw_edmft: no impurity results in {solver_chkpt}.")
    iaft = state.iaft
    embed_iter = state.iteration   # = J + 1
    with HDFArchive(coqui_chkpt, "r") as ar:
        embed_keys = list(ar["embed"].keys())
        embed_final = ar["embed/final_iter"]
    if f"iter{embed_iter}" not in embed_keys:
        raise ValueError(f"run_lr_gw_edmft: {coqui_chkpt} has no embed/iter{embed_iter}, the embedding of the "
                         f"impurity results of dmft iteration {embed_iter - 1}.")
    if embed_final != embed_iter:
        app_log(1, f"[WARNING] embed/final_iter = {embed_final} but the impurity results are those of dmft "
                   f"iteration {embed_iter - 1}: linearizing at embed/iter{embed_iter}.")
    comm.Barrier()

    proj_info = dict(proj_info, dc_cells=ImpurityBlockMap(embedding, embedding_2e).cell_args)

    # ---- kernel
    out_file = output + ".mbpt.h5"
    work_prefix = output + "_work"
    if root:
        h5py.File(out_file, "w").close()
    comm.Barrier()
    chkpt_prefix = f"{outdir}/{prefix}"
    run_kw = dict(kernel_kw, gw_mode="full", recompute_W=True, fix_density=fix_density,
                  projector_info=proj_info, local_polarizabilities=state.local_pi_w)

    def solve(dH0, y_embedded, out_prefix, save_DeltaG, ground_state=False):
        """One run_lr call (collective); returns (niter, local responses of the call, plus the ground-state local
        factors if ground_state)."""
        response = (None if y_embedded is None else
                    {"Sigma": y_embedded["sigma_w"], "Pi": y_embedded["pi_w"]})
        if root:
            if out_prefix != output:
                h5py.File(out_prefix + ".mbpt.h5", "w").close()
        comm.Barrier()
        niter, _ = coqui.mbpt.run_lr(
            {"prefix": chkpt_prefix, "output": out_prefix, "input_type": "embed", "input_iter": embed_iter},
            h_int, q_vec, dH0 if root else None, lr_impurity_response=response,
            **dict(run_kw, save_DeltaG=save_DeltaG))
        loc = None
        if root:
            with h5py.File(out_prefix + ".mbpt.h5", "r") as f:
                g = f["linear_response"]
                keys = ("DeltaGloc_tsIab", "DeltaHloc_sIab", "DeltaWloc_tabcd", "DeltaDelta_wsIab") + (
                    ("Gloc_tsIab", "Gloc_wsIab", "Hloc_sIab", "Wloc_tabcd", "Vloc_abcd") if ground_state else ())
                loc = {k: _as_complex(g[k]) for k in keys}
        return int(niter), comm.bcast(loc, root=0)

    zero_H0 = np.zeros_like(np.asarray(DeltaH0_skij, dtype=np.complex128)) if root else None
    anderson = _Anderson(drv["anderson_history"], drv["mixing"])

    # ---- first solve without an impurity response: also the ground-state local factors
    niters, residuals = [], []
    niter, tot = solve(DeltaH0_skij, None, work_prefix, False, ground_state=True)
    niters.append(niter)
    imps = _ground_state_impurities(state, embedding, embedding_2e, tot["Gloc_tsIab"][:, :, 0],
                                    tot["Gloc_wsIab"][:, :, 0], tot["Hloc_sIab"][:, 0], tot["Vloc_abcd"],
                                    tot["Wloc_tabcd"])
    _log_distance_from_fixed_point(state, solver_chkpt, imps)
    y = [_zero_response(imp) for imp in imps]

    # ---- outer loop
    app_log(1, f"LR GW+EDMFT outer loop: tol_imp = {drv['tol_imp']}, max_outer = {drv['max_outer']}, "
               f"Anderson history = {drv['anderson_history']}; Q = {q_vec.tolist()}, fix_density = {fix_density}")
    converged = False
    for n in range(drv["max_outer"]):
        loc_resp = _local_responses(tot, embedding, embedding_2e)
        y_new, weiss = [], []
        for i, imp in enumerate(imps):
            dg_w, du_w, dh0, ddelta_w = _weiss_response(imp, loc_resp[i], y[i], iaft)
            ret = impurity_response(i, dg_w, du_w, dh0, ddelta_w, ground_state=imp["ground_state"])
            y_new.append(_check_response(ret, imp))
            weiss.append((dg_w, du_w))
        x, x_new = _pack(y), _pack(y_new)
        res = float(np.linalg.norm(x_new - x) / max(np.linalg.norm(x_new), np.finfo(float).tiny))
        residuals.append(res)
        app_log(1, f"  outer {n}: |y' - y| / |y'| = {res:.3e}, run_lr iterations = {niters[-1]}")
        _write_outer(out_file, root, n, tot, loc_resp, weiss, y_new, res, niters[-1])
        if res < drv["tol_imp"]:
            converged = True
            y = y_new
            break
        y_next = _unpack(anderson.step(x, x_new), y)
        niter, inc = solve(zero_H0, _embed_response(embedding, embedding_2e, y_next, y, mf.nspin()),
                           work_prefix, False)
        niters.append(niter)
        tot = dict(tot, **{k: tot[k] + inc[k] for k in ("DeltaGloc_tsIab", "DeltaHloc_sIab", "DeltaWloc_tabcd",
                                                         "DeltaDelta_wsIab")})
        y = y_next
    if not converged:
        app_log(1, f"[WARNING] run_lr_gw_edmft: the impurity response is not converged after {drv['max_outer']} "
                   f"outer iterations (|y' - y| / |y'| = {residuals[-1]:.3e}).")

    # ---- final full solve with the converged response
    niter, final = solve(DeltaH0_skij, _embed_response(embedding, embedding_2e, y, None, mf.nspin()),
                         output, kernel_kw["save_DeltaG"])
    niters.append(niter)
    if root:
        os.remove(work_prefix + ".mbpt.h5")
        with h5py.File(out_file, "a") as f:
            g = f.require_group("lr_gw_edmft")
            g.attrs.update(converged=converged, n_outer=len(residuals))
            g.create_dataset("residuals", data=np.asarray(residuals))
    comm.Barrier()
    app_log(1, f"LR GW+EDMFT: {len(residuals)} outer iterations, converged = {converged}; final run_lr iterations "
               f"= {niter}; output {out_file}")
    return {"converged": converged, "n_outer": len(residuals), "residuals": residuals, "niter": niters,
            "dGloc_tsIab": final["DeltaGloc_tsIab"], "dWloc_tabcd": final["DeltaWloc_tabcd"],
            "impurity_response": y, "output": out_file}


def _split_lr_params(lr_params):
    """(kernel keys with defaults, driver keys with defaults) of the lr_params dictionary."""
    lr_params = dict(lr_params or {})
    kernel = {k: lr_params.pop(k, v) for k, v in _KERNEL_KEYS.items()}
    drv = {k: lr_params.pop(k, v) for k, v in _DRIVER_KEYS.items()}
    if lr_params:
        raise ValueError(f"run_lr_gw_edmft: unknown or driver-fixed lr_params keys {sorted(lr_params)}; "
                         f"valid keys are {sorted(_KERNEL_KEYS) + sorted(_DRIVER_KEYS)}.")
    if drv["max_outer"] < 1 or drv["anderson_history"] < 0:
        raise ValueError("run_lr_gw_edmft: max_outer >= 1 and anderson_history >= 0 are required.")
    return kernel, drv


def _check_ground_state_flags(params):
    """The linearization is defined for the ground-state settings documented in run_lr_gw_edmft."""
    screen_type = params["wloc"]["screen_type"]
    if screen_type not in ("gw_edmft", "gw_edmft_density"):
        raise ValueError(f"run_lr_gw_edmft: screen_type = {screen_type!r} is not supported "
                         f"(gw_edmft, gw_edmft_density).")
    if not params["dmft_embed"].get("corr_only", True):
        raise ValueError("run_lr_gw_edmft: corr_only = False is not supported: the lattice must get the dynamic "
                         "Sigma_imp - Sigma_dc only.")
    for k, solver in enumerate(params["impurity"].get("solver", [])):
        if solver.get("causal_projection") is not None:
            raise ValueError(f"run_lr_gw_edmft: causal_projection (impurity {k}) is a nonlinear bath fit and has no "
                             f"linear response.")
        if solver.get("set_sigma_infty_to_dc", False):
            raise ValueError(f"run_lr_gw_edmft: set_sigma_infty_to_dc = True (impurity {k}) is not supported: the "
                             f"Weiss field must use the impurity's own Sigma_infty.")


# ---------------------------------------------------------------------------------------------------------
# Ground-state impurity factors
# ---------------------------------------------------------------------------------------------------------

def _spin_list(G_tsab):
    """(nt, ns, n, n) -> list of 2 spin arrays, a single spin channel duplicated as the ground state does."""
    ns = G_tsab.shape[1]
    G = [G_tsab[:, s] for s in range(ns)]
    return G if ns == 2 else [G[0], G[0].copy()]


def _static_spin_list(H_sab):
    """(ns, n, n) -> list of 2 spin arrays, a single spin channel duplicated as _spin_list does."""
    H = list(H_sab)
    return H if len(H) == 2 else [H[0], H[0].copy()]


def _ground_state_impurities(state, embedding, embedding_2e, Gloc, Gloc_w, Hloc, Vloc_chem, Wloc_chem):
    """Per impurity: the ground-state local quantities, impurity results and Weiss fields in the dense layouts of the
    impurity callback (spin blocks up/down on axis 1; V, W in the product basis)."""
    iaft = state.iaft
    G_C = embedding.extract(_spin_list(Gloc))
    Gw_C = embedding.extract(_spin_list(Gloc_w))
    H_C = embedding.extract(_static_spin_list(Hloc))
    V_C = embedding_2e.extract([Vloc_chem])
    W_C = embedding_2e.extract([Wloc_chem])
    imps = []
    for i, res in enumerate(state.solver_results):
        gf_struct = res["gf_struct"]
        if len(res["Pi_iw_data"]) != 1:
            raise NotImplementedError("run_lr_gw_edmft: an impurity polarizability with several blocks.")
        gs = {"Gloc_t": blk_arr_to_arr(G_C[i], gf_struct),
              "Gloc_w": blk_arr_to_arr(Gw_C[i], gf_struct),
              "Hloc": blk_arr_to_arr(H_C[i], gf_struct),
              "Wloc_t": chemistry_to_product_basis(W_C[i][0]),
              "Vloc": chemistry_to_product_basis(V_C[i][0]),
              "Vhf_imp": blk_arr_to_arr(res["Sigma_infty"], gf_struct),
              "Sigma_imp_w": blk_arr_to_arr(res["Sigma_iw_data"], gf_struct),
              "Pi_imp_w": res["Pi_iw_data"][0],
              "gf_struct": gf_struct, "iaft": iaft}
        gs["g_weiss_w"], gs["u_weiss_w"], gs["h0"], gs["delta_w"] = compute_weiss_fields_w(
            iaft=iaft, local_gf={k: gs[k] for k in ("Gloc_w", "Hloc", "Wloc_t", "Vloc")},
            impurity_selfenergies={k: gs[k] for k in ("Vhf_imp", "Sigma_imp_w", "Pi_imp_w")}, density_only=True)
        imps.append({"ground_state": gs, "gf_struct": gf_struct})
    return imps


def _log_distance_from_fixed_point(state, solver_chkpt, imps):
    """Relative distance of G^loc from the (undamped) dmft input of the last iteration, one embedding earlier."""
    J = state.iteration - 1
    has = None
    if MPI.COMM_WORLD.Get_rank() == 0:
        with h5py.File(solver_chkpt, "r") as f:
            has = f"dmft/iter{J}/impurity_0/inputs/Gloc_t" in f
    if not MPI.COMM_WORLD.bcast(has, root=0):
        app_log(1, f"Distance from the fixed point: no dmft/iter{J} inputs in {solver_chkpt}.")
        return
    inputs = read_impurity_chkpt(solver_chkpt, J, read="inputs")
    for i, imp in enumerate(imps):
        gs = imp["ground_state"]
        stored = blk_arr_to_arr(inputs[i]["Gloc_t"], imp["gf_struct"])
        app_log(1, f"Distance from the fixed point, impurity {i}: |Gloc(embed {J + 1}) - Gloc(dmft {J} input)| / "
                   f"|Gloc| = {np.linalg.norm(gs['Gloc_t'] - stored) / np.linalg.norm(stored):.3e}")


# ---------------------------------------------------------------------------------------------------------
# Local responses, Weiss maps, impurity responses
# ---------------------------------------------------------------------------------------------------------

def _local_responses(tot, embedding, embedding_2e):
    """Per impurity {"Delta_w": (nw_f, 2, n, n), "Hloc": (2, n, n), "Wloc_t": product basis} of the accumulated
    local responses; Delta_w is run_lr's DeltaDelta_wsIab."""
    dDelta = tot["DeltaDelta_wsIab"][:, :, 0]  # (nw, ns, nC, nC)
    dH = tot["DeltaHloc_sIab"][:, 0]         # (ns, nC, nC)
    dW = tot["DeltaWloc_tabcd"]              # (nt_half, nC^4), chemistry order
    Delta_C = embedding.extract(_spin_list(dDelta))
    H_C = embedding.extract(_static_spin_list(dH))
    W_C = embedding_2e.extract([dW])
    out = []
    for i in range(len(Delta_C)):
        gf_struct = embedding.imp_block_structure[i]
        out.append({"Delta_w": blk_arr_to_arr(Delta_C[i], gf_struct), "Hloc": blk_arr_to_arr(H_C[i], gf_struct),
                    "Wloc_t": chemistry_to_product_basis(W_C[i][0])})
    return out


def _weiss_response(imp, dloc, y, iaft):
    """(dg_weiss_w, du_weiss_w, dh0, ddelta_w) from the local responses and the impurity response y of this
    iteration."""
    gs = imp["ground_state"]
    return lr_compute_weiss_fields_w(
        iaft=iaft,
        local_gf={k: gs[k] for k in ("Gloc_w", "Wloc_t", "Vloc")}, d_local_gf=dloc,
        impurity_selfenergies={k: gs[k] for k in ("Vhf_imp", "Sigma_imp_w", "Pi_imp_w")},
        d_impurity_selfenergies=y)


def _zero_response(imp):
    gs = imp["ground_state"]
    return {k: np.zeros_like(gs[k]) for k in ("Vhf_imp", "Sigma_imp_w", "Pi_imp_w")}


def _check_response(ret, imp):
    gs = imp["ground_state"]
    missing = {"Vhf_imp", "Sigma_imp_w", "Pi_imp_w"} - set(ret)
    if missing:
        raise ValueError(f"impurity_response: missing keys {sorted(missing)} in the returned dictionary.")
    out = {k: np.asarray(ret[k], dtype=np.complex128) for k in ("Vhf_imp", "Sigma_imp_w", "Pi_imp_w")}
    for k, v in out.items():
        if v.shape != gs[k].shape:
            raise ValueError(f"impurity_response: {k} has the shape {v.shape}, expected {gs[k].shape} "
                             f"(the layout of the ground-state impurity results).")
    return out


def _embed_response(embedding, embedding_2e, y, y_prev, nspin):
    """The embedded local arrays of lr_impurity_response, for y (or y - y_prev): Sigma_imp (nw, ns, 1, nC, nC), spin
    averaged for a spin-unpolarized mean field, and Pi_imp (nw_half, nC^4) in the product basis."""
    results = []
    for i, yi in enumerate(y):
        d = {k: yi[k] - (y_prev[i][k] if y_prev is not None else 0.0) for k in yi}
        gf_struct = embedding.imp_block_structure[i]
        blk = {"Sigma_iw_data": arr_to_blk_arr(d["Sigma_imp_w"], gf_struct),
               "Sigma_infty": arr_to_blk_arr(d["Vhf_imp"], gf_struct),
               "Pi_iw_data": [d["Pi_imp_w"]]}
        results.append(dict(blk, Sigma_iw_dc_data=blk["Sigma_iw_data"], Sigma_infty_dc=blk["Sigma_infty"],
                            Pi_iw_dc_data=blk["Pi_iw_data"]))
    # Sigma_infty, Sigma_iw_dc_data, Sigma_infty_dc and Pi_iw_dc_data only fill embed_impurities' signature; its
    # hf and dc outputs are discarded.
    sigma_w, _, pi_w = embed_impurities(embedding, embedding_2e, results, spin_average=nspin == 1)
    return {"sigma_w": sigma_w["imp"], "pi_w": pi_w["imp"]}


# ---------------------------------------------------------------------------------------------------------
# The local vector y and its acceleration
# ---------------------------------------------------------------------------------------------------------

def _pack(y):
    """The impurity responses as one real vector (real and imaginary parts, so that real-linear maps are covered)."""
    v = np.concatenate([y_i[k].ravel() for y_i in y for k in ("Vhf_imp", "Sigma_imp_w", "Pi_imp_w")])
    return np.concatenate([v.real, v.imag])


def _unpack(x, template):
    half = x.size // 2
    v = x[:half] + 1j * x[half:]
    out, pos = [], 0
    for t in template:
        d = {}
        for k in ("Vhf_imp", "Sigma_imp_w", "Pi_imp_w"):
            d[k] = v[pos:pos + t[k].size].reshape(t[k].shape)
            pos += t[k].size
        out.append(d)
    return out


class _Anderson:
    """
    Anderson acceleration of x = F(x) on a real vector, x_new = x + beta r - (dX + beta dR) gamma with r = F(x) - x and
    gamma the least-squares solution of dR gamma = r over the last `history` differences. history = 0 is the plain
    iteration with linear mixing beta (`_mix_into`).
    """
    def __init__(self, history, mixing):
        self.history, self.mixing = history, mixing
        self._x, self._r = [], []

    def step(self, x, fx):
        if self.history == 0:
            x_new = fx.copy()
            _mix_into(x_new, x, self.mixing)
            return x_new
        r = fx - x
        self._x = (self._x + [x.copy()])[-(self.history + 1):]
        self._r = (self._r + [r.copy()])[-(self.history + 1):]
        x_new = x + self.mixing * r
        if len(self._r) > 1:
            dR = np.stack([b - a for a, b in zip(self._r[:-1], self._r[1:])], axis=1)
            dX = np.stack([b - a for a, b in zip(self._x[:-1], self._x[1:])], axis=1)
            gamma = np.linalg.lstsq(dR, r, rcond=None)[0]
            x_new = x_new - (dX + self.mixing * dR) @ gamma
        return x_new


def _write_outer(out_file, root, n, tot, loc_resp, weiss, y_new, res, niter):
    """Per-outer-iteration output on the root: the accumulated local responses before the Weiss maps (as run_lr
    wrote them), the Weiss-field responses and the impurity responses they produced."""
    if root:
        with h5py.File(out_file, "a") as f:
            g = f.require_group("lr_gw_edmft").create_group(f"iter{n}")
            g.attrs.update(residual=res, run_lr_iterations=niter)
            write_complex_dataset(g, "DeltaGloc_tsIab", tot["DeltaGloc_tsIab"])
            write_complex_dataset(g, "DeltaDelta_wsIab", tot["DeltaDelta_wsIab"])
            write_complex_dataset(g, "DeltaWloc_tabcd", tot["DeltaWloc_tabcd"])
            for i, ((dg, du), y) in enumerate(zip(weiss, y_new)):
                gi = g.create_group(f"impurity_{i}")
                write_complex_dataset(gi, "dg_weiss_w", dg)
                write_complex_dataset(gi, "du_weiss_w", du)
                for k, v in y.items():
                    write_complex_dataset(gi, k, v)
    MPI.COMM_WORLD.Barrier()
