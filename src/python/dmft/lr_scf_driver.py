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
import hashlib
import json
import os

import h5py
import numpy as np
from mpi4py import MPI

import triqs_modest as modest

import coqui
import coqui.mbpt
from coqui import app_log
from coqui.embed import lr_dc_response
from coqui.utils.h5_utils import write_complex_dataset

from .dmft_state import DMFTState
from .impurity_block_map import ImpurityBlockMap
from .io import convert_gw_edmft_params, read_impurity_chkpt, update_impurity_results_from_chkpt
from .lr_weiss import lr_compute_weiss_fields_w
from .triqs_utils import blk_arr_to_arr, arr_to_blk_arr
from .weiss import chemistry_to_product_basis, compute_weiss_fields_w, embed_impurities, get_proj_info

__all__ = ["run_lr_gw_edmft"]

_KERNEL_KEYS = {"include_hartree": True, "include_exchange": True, "max_iter": 200, "tol": 1e-10,
                "iter_alg": {"alg": "GCR"}, "precond_method": "none", "save_DeltaG": True}
_DRIVER_KEYS = {"tol_outer": 1e-3, "max_outer": 10, "mixing": 1.0, "anderson_history": 0, "output": None,
                "restart": True, "iteration": None, "fix_density": None}
_W_KEYS = ("dg_weiss_w", "du_weiss_w", "dh0", "ddelta_w")
_Y_KEYS = ("Vhf_imp", "Sigma_imp_w", "Pi_imp_w")
_LOCAL_KEYS = ("DeltaGloc_tsIab", "DeltaHloc_sIab", "DeltaWloc_tabcd", "DeltaSigma_dc_tsIab")
_GS_KEYS = ("Gloc_tsIab", "Hloc_sIab", "Wloc_tabcd", "Vloc_abcd")


def run_lr_gw_edmft(h_int, embedding, *, proj_info=None, params: dict, q_vec, DeltaH0_skij,
                    impurity_response, lr_params=None):
    """
    Linear response of a GW+EDMFT ground state: the outer (impurity) loop of the self-consistent LR GW+EDMFT.

    ``run_lr`` solves the lattice and the double counting together at a given impurity response y = (delta Vhf_imp,
    delta Sigma_imp, delta Pi_imp); the loop iterates the Weiss-field responses w = (dg_weiss, du_weiss, dh0, ddelta)
    that the impurity solver receives. Per outer iteration n:

        1. y_n = ``impurity_response``(w_n);
        2. one ``run_lr`` call with DeltaH0 = 0 and the source y_n - y_{n-1}: the increment of the local responses
           (delta G^loc, delta H^loc, delta W^loc, delta Sigma_dc, delta N), accumulated;
        3. the Weiss LR of the accumulated local responses and y_n gives w_out,n;
        4. w_{n+1} = w_n + mixing (w_out,n - w_n), or Anderson mixing on w (``anderson_history`` > 0); the mixing
           coefficients apply to all four components of w.

    The start w_0 is the Weiss LR of the first ``run_lr`` call (DeltaH0, y = 0). After convergence one final
    ``run_lr`` call with DeltaH0 and the last y gives the standard LR output.

    The ground state is read, not passed. ``params`` is the ``run_gw_edmft`` dictionary: it fixes the checkpoints
    (``outdir``, ``prefix``, ``edmft.chkpt_h5``), the projector (``wannier_file``, unless ``proj_info`` is given),
    ``screen_type`` and ``mu``, and is checked against the linearization (below). The linearization point is
        - G (and F, Sigma, mu): ``embed/iter{K}`` of the coqui checkpoint, K = ``lr_params["iteration"]`` or
          ``embed/final_iter``;
        - Sigma_imp, Vhf_imp, Pi_imp, mu_imp: the (damped) results of dmft iteration J = ``embed/iter{K}/dmft_iter``,
          the iteration whose results ``coqui.dmft_embed(..., dmft_iter=J)`` embedded into K. The driver raises if
          the dataset is absent (a checkpoint of an older run is tagged by the project that owns it);
        - G^loc, H^loc, W^loc, V^loc: computed by the first ``run_lr`` call from its own G, F, mu and W
          (``Gloc_tsIab``, ``Hloc_sIab``, ``Wloc_tabcd``, ``Vloc_abcd`` of ``linear_response``), the factors of its
          double counting; H^loc = P (H0 + F) P^dagger - mu is the static local one-body term of G^loc. The Weiss
          fields follow from them and the impurity results.
    The distance of G^loc from the stored dmft input (iteration J + 1 if present, else J) is logged.

    The only assumption on the perturbation is that DeltaH0 is time-reversal even. The responses are not
    symmetrized: w and y are complex at Q != 0 and w(-iw) is not w(iw)^dagger in general. The anti-Hermitian norms
    of w_out and y are logged and stored (``antiherm_w_out``, ``antiherm_y``) as a diagnostic. The symmetry
    handling of a complex w (a real-only solver, an inversion-symmetric system) is the callback's job.

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
        for these only).
    q_vec : array-like (3,)
        Momentum transfer, crystal coordinates.
    DeltaH0_skij : np.ndarray (ns, nk, nb, nb)
        Perturbation, required on the MPI root.
    impurity_response : callable
        ``impurity_response(imp_index, dg_weiss_w, du_weiss_w, dh0, ddelta_w, *, ground_state, outer_iteration)
        -> dict``, called collectively on all ranks, once per impurity and outer iteration, with identical
        arguments on all ranks (the driver is agnostic of the process topology of the call). The Weiss-field
        responses are in the layouts of the ground-state impurity inputs:
            dg_weiss_w  (nw_f, 2, n, n)         fermionic Weiss field response, spin blocks up/down
            du_weiss_w  (nw_b_half, n, n, n, n) bosonic Weiss field response, product basis
            dh0         (2, n, n)               response of h0 = H^loc - Vhf_imp
            ddelta_w    (nw_f, 2, n, n)         response of the hybridization
        The responses are complex at every Q (the frequency-dependent fields satisfy dX(iw) != conj(dX(-iw)) in
        general). A real-only solver splits w into its Hermitian part ((dDelta(iw) + dDelta(-iw)^dagger) / 2,
        (dh0 + dh0^dagger) / 2, ...) and the anti-Hermitian part times -i, each a real-type perturbation, and
        recombines y = y(w_H) + i y(w_A'). The callback is called on all ranks and must return the same response
        on all of them; the driver uses the root's copy.
        ``outer_iteration`` is n, so that a callback can name its files by n and skip work that is already done
        after a restart. ``ground_state`` is a dict of the unperturbed impurity quantities in the same layouts
        (``Gloc_t`` with the shape (nt_f, 2, n, n), ``Wloc_t`` and ``Vloc`` in the product basis, ``g_weiss_w``,
        ``u_weiss_w``, ``Hloc``, ``h0``, ``delta_w``, ``Vhf_imp``, ``Sigma_imp_w``, ``Pi_imp_w``, ``mu_imp``,
        ``gf_struct``, ``iaft``). It returns
            {"Vhf_imp": (2, n, n), "Sigma_imp_w": (nw_f, 2, n, n), "Pi_imp_w": (nw_b_half, n, n)}
        the response of the static and dynamic impurity self-energy and of the density-density polarizability,
        at fixed impurity chemical potential (the ground state fixes mu_imp; a finite-difference impurity must hold
        it fixed). Further keys with numeric array values are stored under ``callback/`` and not used; other values
        are dropped.
    lr_params : dict, optional
        Kernel keys passed to ``run_lr`` (defaults): ``include_hartree`` (True), ``include_exchange`` (True),
        ``max_iter`` (200), ``tol`` (1e-10), ``iter_alg`` ({"alg": "GCR"}), ``precond_method`` ("none"),
        ``save_DeltaG`` (True; the final solve only). ``gw_mode`` is "full" and ``recompute_W`` True.
        Driver keys:
            ``tol_outer`` (1e-3): stop when |w_out - w_in| / |w_out| < tol_outer, the norm over the real and
                imaginary parts of (dh0, ddelta_w, du_weiss_w) of all impurities;
            ``max_outer`` (10): the maximum number of outer iterations (a warning is logged if the residual grows
                three times in a row);
            ``mixing`` (1.0): the linear mixing of w, w_{n+1} = w_n + mixing (w_out,n - w_n);
            ``anderson_history`` (0): Anderson mixing on w over this many differences; 0 is linear mixing;
            ``output``: prefix of the output checkpoint, default ``{outdir}/{prefix}_lr_gw_edmft``;
            ``restart`` (True): resume from ``{output}.mbpt.h5`` (below); False starts anew;
            ``iteration``: the embedding iteration K, default ``embed/final_iter``;
            ``fix_density``: the density constraint of ``run_lr``. Default: conserved at Q = 0 when ``mu`` is not
                fixed in ``params`` (``fix_density = True``), not at Q != 0 or at a fixed lattice ``mu``.

    Restart. The root writes every stage to ``{output}.mbpt.h5`` before the next one starts; a group is complete only
    when its attribute ``done`` is set, written last. A call on an existing file with ``restart = True`` checks
    ``lr_gw_edmft/meta`` (Q, DeltaH0, K, J, ``fix_density``, and the settings that change the solution:
    ``include_hartree``, ``include_exchange``, ``tol``, ``max_iter``, ``screen_type``, ``mu``, the impurity IAFT; a
    mismatch raises and nothing is overwritten), drops the incomplete groups and continues at the first missing
    stage: the callback is called again only for the impurities and iterations without stored response, a
    ``run_lr`` call that was killed is redone, and the mixing history is rebuilt from the stored w. ``tol_outer``,
    ``max_outer`` and the mixing may change. A final solve that is stale because the loop runs further is deleted.
    Only ``meta`` is checked: a restart trusts that the ground-state checkpoints, the projector and ``h_int`` are
    unchanged since the stored stages were written.

    Returns
    -------
    dict
        ``converged``, ``n_outer``, ``residuals`` (|w_out - w_in| / |w_out| per outer iteration), ``niter``
        (kernel iterations of the first call, the increments and the final solve), ``DeltaN`` (change of the
        electron number per outer iteration, (2/ns) sum_k w_k Tr DeltaDm of the accumulated response),
        ``dGloc_tsIab`` / ``dWloc_tabcd`` (local responses of the final solve), ``impurity_response`` (the last
        response y of every impurity), ``embed_iter`` (K), ``dmft_iter`` (J), ``resumed_at`` (the number of outer
        iterations found complete in the file), ``output`` (path).
        The file ``{output}.mbpt.h5`` holds the standard LR output of the final solve under ``linear_response``
        and the loop under ``lr_gw_edmft``: attributes ``converged``, ``n_outer``, ``done_final``, dataset
        ``residuals``, groups ``meta``, ``ground_state`` (``Gloc_tsIab``, ``Hloc_sIab``, ``Wloc_tabcd``,
        ``Vloc_abcd``), ``init`` (the local responses at y = 0 and w_0 as ``impurity_{i}/w_out``) and
        ``iter{n}``: attributes ``residual``, ``run_lr_iterations``, ``DeltaN``, ``antiherm_w_out``,
        ``antiherm_y``; ``local`` (accumulated ``DeltaGloc_tsIab``, ``DeltaHloc_sIab``, ``DeltaWloc_tabcd``,
        ``DeltaSigma_dc_tsIab``, ``DeltaPi_dc_tabcd``, consistent with y_n); ``impurity_{i}/{w_in, w_out, y,
        callback}``. Groups ``iter{n}`` with n >= ``n_outer`` can remain from an earlier call with a larger
        ``max_outer`` or a tighter ``tol_outer``; they are not part of this solution.
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
    fix_density = drv["fix_density"] if drv["fix_density"] is not None else (
        embed_params.get("mu") is None and is_gamma)
    fix_density = bool(fix_density)
    output = drv["output"] or f"{outdir}/{prefix}_lr_gw_edmft"

    # ---- ground state: the embedding iteration K and the impurity results J that it embeds
    embed_iter, dmft_iter = _read_pairing(coqui_chkpt, solver_chkpt, drv["iteration"])
    embedding_2e = embedding.merge_embed_block_by_imp.slice_sigma
    state = DMFTState.make_dmft_state(
        coqui_chkpt, embedding, embedding_2e, wmax_imp=imp_iaft.get("wmax"), eps_imp=imp_iaft.get("eps"),
        spin_average=mf.nspin() == 1, screen_type=wloc_params["screen_type"], verbal=root)
    update_impurity_results_from_chkpt(state.solver_results, solver_chkpt, iteration=dmft_iter)
    state.check_loaded_results()
    state.embed_impurity_results()
    iaft = state.iaft

    def read_kweight():
        with h5py.File(coqui_chkpt, "r") as f:
            return _read_array(f["system/k_weight"])
    kweight = _on_root(read_kweight)
    comm.Barrier()

    proj_info = dict(proj_info, dc_cells=ImpurityBlockMap(embedding, embedding_2e).cell_args)

    # ---- output file: stages, meta
    out_file = output + ".mbpt.h5"
    work_prefix = output + "_work"
    st = _OuterState(out_file)
    sha1 = hashlib.sha1(np.ascontiguousarray(DeltaH0_skij, dtype=np.complex128).tobytes()).hexdigest() if root else None
    hashed = ("include_hartree", "include_exchange", "tol", "max_iter")
    settings = json.dumps([{k: kernel_kw[k] for k in hashed},
                           wloc_params["screen_type"], embed_params.get("mu"), imp_iaft], sort_keys=True, default=repr)
    meta = {"q_vec": q_vec, "dH0_sha1": comm.bcast(sha1, root=0), "embed_iter": embed_iter, "dmft_iter": dmft_iter,
            "fix_density": fix_density, "settings_sha1": hashlib.sha1(settings.encode()).hexdigest()}
    st.start(meta, drv["restart"])
    comm.Barrier()

    chkpt_prefix = f"{outdir}/{prefix}"
    run_kw = dict(kernel_kw, gw_mode="full", recompute_W=True, fix_density=fix_density,
                  projector_info=proj_info, local_polarizabilities=state.local_pi_w)

    def solve(dH0, y_embedded, out_prefix, save_DeltaG, ground_state=False):
        """One run_lr call (collective); (niter, {local responses, DeltaN, [ground-state local factors]})."""
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

        def read():
            with h5py.File(out_prefix + ".mbpt.h5", "r") as f:
                g = f["linear_response"]
                loc = {k: _read_array(g[k]) for k in _LOCAL_KEYS + (_GS_KEYS if ground_state else ())}
                dm = _read_array(g["DeltaDm_skij"])
            if kweight.shape[0] != dm.shape[1]:
                raise ValueError(f"run_lr_gw_edmft: system/k_weight has {kweight.shape[0]} points, DeltaDm_skij "
                                 f"{dm.shape[1]}.")
            loc["DeltaN"] = (2.0 / dm.shape[0]) * np.einsum("k,skii->", kweight, dm)
            return loc
        return int(niter), _on_root(read)

    return _outer_loop(
        st, solve, impurity_response, state=state, embedding=embedding, embedding_2e=embedding_2e, iaft=iaft,
        nspin=mf.nspin(), proj_info=proj_info, solver_chkpt=solver_chkpt, DeltaH0_skij=DeltaH0_skij,
        kernel_kw=kernel_kw, drv=drv, q_vec=q_vec, fix_density=fix_density, embed_iter=embed_iter,
        dmft_iter=dmft_iter, output=output, work_prefix=work_prefix)


def _outer_loop(st, solve, impurity_response, *, state, embedding, embedding_2e, iaft, nspin, proj_info,
                solver_chkpt, DeltaH0_skij, kernel_kw, drv, q_vec, fix_density, embed_iter, dmft_iter, output,
                work_prefix):
    """The ground-state local factors, the stage-wise outer loop with its restart replay, and the final solve of
    ``run_lr_gw_edmft``. ``solve(dH0, y_embedded, out_prefix, save_DeltaG, ground_state=False)`` is the collective
    ``run_lr`` call."""
    comm = MPI.COMM_WORLD
    root = comm.Get_rank() == 0
    tol_outer, max_outer = drv["tol_outer"], drv["max_outer"]
    zero_H0 = np.zeros_like(np.asarray(DeltaH0_skij, dtype=np.complex128)) if root else None
    anderson = _Anderson(drv["anderson_history"], drv["mixing"])
    nimp = len(state.solver_results)

    def local_with_dc(tot, gs_loc):
        """The accumulated local responses, with delta Pi_dc of the accumulated delta G^loc."""
        _, dPi_dc = lr_dc_response(gs_loc["Gloc_tsIab"], tot["DeltaGloc_tsIab"], gs_loc["Wloc_tabcd"], None,
                                   **proj_info["dc_cells"])
        return dict({k: tot[k] for k in _LOCAL_KEYS}, DeltaPi_dc_tabcd=dPi_dc)

    # ---- ground state and the response at y = 0
    niters = []
    if st.has("ground_state") and st.has("init"):
        gs_loc = st.get("ground_state")[0]
        data, attrs = st.get("init/local")
        tot0 = dict({k: data[k] for k in _LOCAL_KEYS}, DeltaN=attrs["DeltaN"])
        niters.append(int(attrs["run_lr_iterations"]))
        imps = _ground_state_impurities(state, embedding, embedding_2e, gs_loc["Gloc_tsIab"][:, :, 0],
                                        gs_loc["Hloc_sIab"][:, 0], gs_loc["Vloc_abcd"], gs_loc["Wloc_tabcd"])
        w0 = [st.get(f"init/impurity_{i}/w_out")[0] for i in range(nimp)]
    else:
        niter, tot0 = solve(DeltaH0_skij, None, work_prefix, False, ground_state=True)
        niters.append(niter)
        gs_loc = {k: tot0[k] for k in _GS_KEYS}
        st.put("ground_state", gs_loc)
        imps = _ground_state_impurities(state, embedding, embedding_2e, gs_loc["Gloc_tsIab"][:, :, 0],
                                        gs_loc["Hloc_sIab"][:, 0], gs_loc["Vloc_abcd"], gs_loc["Wloc_tabcd"])
        y_zero = [_zero_response(imp) for imp in imps]
        w0 = _weiss_responses(imps, tot0, y_zero, embedding, embedding_2e, iaft)
        st.put("init/local", local_with_dc(tot0, gs_loc), {"DeltaN": tot0["DeltaN"], "run_lr_iterations": niter})
        for i in range(nimp):
            st.put(f"init/impurity_{i}/w_out", w0[i])
        st.close_group("init")
    _log_distance_from_fixed_point(solver_chkpt, imps, embed_iter, dmft_iter)

    # ---- iterations completed by an earlier call; the first one below tol_outer ends the loop, later ones are stale
    stored = 0
    while st.has(f"iter{stored}"):
        stored += 1
    residuals, converged = [], False
    for n in range(min(stored, max_outer)):
        residuals.append(float(st.get(f"iter{n}")[1]["residual"]))
        if residuals[-1] < tol_outer:
            converged = True
            break
    n = len(residuals)
    resumed_at = n
    final_ok = st.final_valid(n, kernel_kw["save_DeltaG"])
    if not final_ok:
        st.invalidate_final()
    deltaN = [st.get(f"iter{m}")[1]["DeltaN"] for m in range(n)]
    niters += [int(st.get(f"iter{m}")[1]["run_lr_iterations"]) for m in range(n)]

    # state at the end of the last complete iteration: accumulated local responses, y, the mixing history, w_{n}
    hist = []
    if n == 0:
        tot, y_prev, w_next = tot0, [_zero_response(imp) for imp in imps], w0
    else:
        tot = _read_local(st, f"iter{n - 1}/local")
        y_prev = [st.get(f"iter{n - 1}/impurity_{i}/y")[0] for i in range(nimp)]
        for m in range(max(0, n - drv["anderson_history"] - 1), n):
            w_in = [st.get(f"iter{m}/impurity_{i}/w_in")[0] for i in range(nimp)]
            w_out = [st.get(f"iter{m}/impurity_{i}/w_out")[0] for i in range(nimp)]
            hist.append((w_in, w_out))
            a, b = anderson.coefficients(_w_pack(w_in), _w_pack(w_out))
        w_next = _mix(hist, a, b)

    # ---- outer loop
    app_log(1, f"LR GW+EDMFT outer loop: tol_outer = {tol_outer}, max_outer = {max_outer}, mixing = {drv['mixing']}, "
               f"Anderson history = {drv['anderson_history']}; Q = {q_vec.tolist()}, fix_density = {fix_density}; "
               f"embed iteration {embed_iter}, dmft iteration {dmft_iter}; {resumed_at} complete outer iterations "
               f"found in {st.path}")
    if not converged and n < max_outer:
        st.invalidate_final()
        final_ok = False
    while not converged and n < max_outer:
        it = f"iter{n}"
        # stage a: w_n
        if all(st.has(f"{it}/impurity_{i}/w_in") for i in range(nimp)):
            w_in = [st.get(f"{it}/impurity_{i}/w_in")[0] for i in range(nimp)]
        else:
            w_in = w_next
            for i in range(nimp):
                st.put(f"{it}/impurity_{i}/w_in", w_in[i])
        # stage b: y_n
        y = []
        for i in range(nimp):
            if st.has(f"{it}/impurity_{i}/y"):
                y.append(st.get(f"{it}/impurity_{i}/y")[0])
                continue
            w = w_in[i]
            ret = impurity_response(i, w["dg_weiss_w"], w["du_weiss_w"], w["dh0"], w["ddelta_w"],
                                    ground_state=imps[i]["ground_state"], outer_iteration=n)
            y_i, extras = _on_root(lambda: _check_response(ret, imps[i]))
            st.put(f"{it}/impurity_{i}/callback", extras)
            st.put(f"{it}/impurity_{i}/y", y_i)
            y.append(y_i)
        # stage c: the accumulated local responses for y_n
        if st.has(f"{it}/local"):
            tot = _read_local(st, f"{it}/local")
            niter = int(st.get(f"{it}/local")[1]["run_lr_iterations"])
        else:
            if max(np.linalg.norm(yi[k] - pi[k]) for yi, pi in zip(y, y_prev) for k in _Y_KEYS) == 0.0:
                # No change of y: the increment is zero
                niter = 0
            else:
                niter, inc = solve(zero_H0, _embed_response(embedding, embedding_2e, y, y_prev, nspin),
                                   work_prefix, False)
                tot = dict(tot, **{k: tot[k] + inc[k] for k in _LOCAL_KEYS + ("DeltaN",)})
            st.put(f"{it}/local", local_with_dc(tot, gs_loc), {"DeltaN": tot["DeltaN"], "run_lr_iterations": niter})
        # stage d: Weiss LR -> w_out,n, residual, mixing
        w_out = _weiss_responses(imps, tot, y, embedding, embedding_2e, iaft)
        x_in, x_out = _w_pack(w_in), _w_pack(w_out)
        res = float(np.linalg.norm(x_out - x_in) / max(np.linalg.norm(x_out), np.finfo(float).tiny))
        anti = _antihermitian_norms(w_out, y, iaft)
        for i in range(nimp):
            st.put(f"{it}/impurity_{i}/w_out", w_out[i])
        st.close_group(it, {"residual": res, "run_lr_iterations": niter, "DeltaN": tot["DeltaN"],
                            **({} if anti is None else {"antiherm_w_out": anti[0], "antiherm_y": anti[1]})})
        residuals.append(res)
        deltaN.append(tot["DeltaN"])
        niters.append(niter)
        app_log(1, f"  outer {n}: |w_out - w_in| / |w_out| = {res:.3e}, DeltaN = {tot['DeltaN'].real:+.6e}, "
                   f"run_lr iterations = {niter}" +
                   ("" if anti is None else f", anti-Hermitian part of w_out {anti[0]:.2e}, of y {anti[1]:.2e}"))
        if len(residuals) > 3 and all(residuals[-k] > residuals[-k - 1] for k in (1, 2, 3)):
            app_log(1, "[WARNING] run_lr_gw_edmft: the outer residual grew three times in a row.")
        hist = (hist + [(w_in, w_out)])[-(drv["anderson_history"] + 1):]
        a, b = anderson.coefficients(x_in, x_out)
        w_next = _mix(hist, a, b)
        y_prev = y
        n += 1
        converged = res < tol_outer
    if not converged:
        app_log(1, f"[WARNING] run_lr_gw_edmft: the Weiss-field response is not converged after {max_outer} "
                   f"outer iterations (|w_out - w_in| / |w_out| = {residuals[-1]:.3e}).")

    # ---- final full solve with the last response
    if final_ok:
        final = st.read_final()
        niters.append(final["niter"])
        st.set_final(converged, residuals, final["niter"])
    else:
        niter, final = solve(DeltaH0_skij, _embed_response(embedding, embedding_2e, y_prev, None, nspin),
                             output, kernel_kw["save_DeltaG"])
        niters.append(niter)
        st.set_final(converged, residuals, niter)
        final = {"dGloc_tsIab": final["DeltaGloc_tsIab"], "dWloc_tabcd": final["DeltaWloc_tabcd"]}
    if root and os.path.exists(work_prefix + ".mbpt.h5"):
        os.remove(work_prefix + ".mbpt.h5")
    comm.Barrier()
    app_log(1, f"LR GW+EDMFT: {n} outer iterations, converged = {converged}; final run_lr iterations "
               f"= {niters[-1]}; output {st.path}")
    return {"converged": converged, "n_outer": n, "residuals": residuals, "niter": niters, "DeltaN": deltaN,
            "dGloc_tsIab": final["dGloc_tsIab"], "dWloc_tabcd": final["dWloc_tabcd"], "impurity_response": y_prev,
            "embed_iter": embed_iter, "dmft_iter": dmft_iter, "resumed_at": resumed_at, "output": st.path}


def _split_lr_params(lr_params):
    """(kernel keys with defaults, driver keys with defaults) of the lr_params dictionary."""
    lr_params = dict(lr_params or {})
    kernel = {k: lr_params.pop(k, v) for k, v in _KERNEL_KEYS.items()}
    drv = {k: lr_params.pop(k, v) for k, v in _DRIVER_KEYS.items()}
    if lr_params:
        raise ValueError(f"run_lr_gw_edmft: unknown or driver-fixed lr_params keys {sorted(lr_params)}; "
                         f"valid keys are {sorted(_KERNEL_KEYS) + sorted(_DRIVER_KEYS)}.")
    if drv["max_outer"] < 1 or drv["anderson_history"] < 0 or drv["mixing"] <= 0:
        raise ValueError("run_lr_gw_edmft: max_outer >= 1, anderson_history >= 0 and mixing > 0 are required.")
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


def _ground_state_impurities(state, embedding, embedding_2e, Gloc, Hloc, Vloc_chem, Wloc_chem):
    """Per impurity: the ground-state local quantities, impurity results and Weiss fields in the dense layouts of the
    impurity callback (spin blocks up/down on axis 1; V, W in the product basis)."""
    iaft = state.iaft
    G_C = embedding.extract(_spin_list(Gloc))
    H_C = embedding.extract(_static_spin_list(Hloc))
    V_C = embedding_2e.extract([Vloc_chem])
    W_C = embedding_2e.extract([Wloc_chem])
    imps = []
    for i, res in enumerate(state.solver_results):
        gf_struct = res["gf_struct"]
        if len(res["Pi_iw_data"]) != 1:
            raise NotImplementedError("run_lr_gw_edmft: an impurity polarizability with several blocks.")
        gs = {"Gloc_t": blk_arr_to_arr(G_C[i], gf_struct),
              "Hloc": blk_arr_to_arr(H_C[i], gf_struct),
              "Wloc_t": chemistry_to_product_basis(W_C[i][0]),
              "Vloc": chemistry_to_product_basis(V_C[i][0]),
              "Vhf_imp": blk_arr_to_arr(res["Sigma_infty"], gf_struct),
              "mu_imp": float(np.real(res["mu_imp"])),
              "Sigma_imp_w": blk_arr_to_arr(res["Sigma_iw_data"], gf_struct),
              "Pi_imp_w": res["Pi_iw_data"][0],
              "gf_struct": gf_struct, "iaft": iaft}
        gs["g_weiss_w"], gs["u_weiss_w"], gs["h0"], gs["delta_w"] = compute_weiss_fields_w(
            iaft=iaft, local_gf={k: gs[k] for k in ("Gloc_t", "Hloc", "Wloc_t", "Vloc")},
            impurity_selfenergies={k: gs[k] for k in ("Vhf_imp", "Sigma_imp_w", "Pi_imp_w")}, density_only=True)
        imps.append({"ground_state": gs, "gf_struct": gf_struct})
    return imps



def _log_distance_from_fixed_point(solver_chkpt, imps, embed_iter, dmft_iter):
    """Relative distance of G^loc of the embedding from the stored dmft input of iteration J + 1 (the input that
    follows the embedded results) if present, else of iteration J."""
    def find():
        with h5py.File(solver_chkpt, "r") as f:
            return [j for j in (dmft_iter + 1, dmft_iter) if f"dmft/iter{j}/impurity_0/inputs/Gloc_t" in f]
    found = _on_root(find)
    if not found:
        app_log(1, f"Distance from the fixed point: no dmft/iter{dmft_iter + 1} or dmft/iter{dmft_iter} inputs in "
                   f"{solver_chkpt}.")
        return
    inputs = read_impurity_chkpt(solver_chkpt, found[0], read="inputs")
    for i, imp in enumerate(imps):
        gs = imp["ground_state"]
        stored = blk_arr_to_arr(inputs[i]["Gloc_t"], imp["gf_struct"])
        app_log(1, f"Distance from the fixed point, impurity {i}: |Gloc(embed {embed_iter}) - Gloc(dmft {found[0]} "
                   f"input)| / |Gloc| = {np.linalg.norm(gs['Gloc_t'] - stored) / np.linalg.norm(stored):.3e}")


# ---------------------------------------------------------------------------------------------------------
# Local responses, Weiss maps, impurity responses
# ---------------------------------------------------------------------------------------------------------

def _local_responses(tot, embedding, embedding_2e):
    """Per impurity {"Gloc_t": (nt, 2, n, n), "Hloc": (2, n, n), "Wloc_t": product basis} of the accumulated local
    responses."""
    dG = tot["DeltaGloc_tsIab"][:, :, 0]     # (nt, ns, nC, nC)
    dH = tot["DeltaHloc_sIab"][:, 0]         # (ns, nC, nC)
    dW = tot["DeltaWloc_tabcd"]              # (nt_half, nC^4), chemistry order
    G_C = embedding.extract(_spin_list(dG))
    H_C = embedding.extract(_static_spin_list(dH))
    W_C = embedding_2e.extract([dW])
    out = []
    for i in range(len(G_C)):
        gf_struct = embedding.imp_block_structure[i]
        out.append({"Gloc_t": blk_arr_to_arr(G_C[i], gf_struct), "Hloc": blk_arr_to_arr(H_C[i], gf_struct),
                    "Wloc_t": chemistry_to_product_basis(W_C[i][0])})
    return out


def _weiss_response(imp, dloc, y, iaft):
    """(dg_weiss_w, du_weiss_w, dh0, ddelta_w) from the local responses and the impurity response y of this
    iteration."""
    gs = imp["ground_state"]
    return lr_compute_weiss_fields_w(
        iaft=iaft,
        local_gf={k: gs[k] for k in ("Gloc_t", "Wloc_t", "Vloc")}, d_local_gf=dloc,
        impurity_selfenergies={k: gs[k] for k in ("Vhf_imp", "Sigma_imp_w", "Pi_imp_w")},
        d_impurity_selfenergies=y)


def _zero_response(imp):
    gs = imp["ground_state"]
    return {k: np.zeros_like(gs[k]) for k in ("Vhf_imp", "Sigma_imp_w", "Pi_imp_w")}


def _check_response(ret, imp):
    """(the response y as complex arrays of the ground-state shapes, the extra keys of the callback's dictionary)."""
    gs = imp["ground_state"]
    missing = set(_Y_KEYS) - set(ret)
    if missing:
        raise ValueError(f"impurity_response: missing keys {sorted(missing)} in the returned dictionary.")
    out = {k: np.asarray(ret[k], dtype=np.complex128) for k in _Y_KEYS}
    for k, v in out.items():
        if v.shape != gs[k].shape:
            raise ValueError(f"impurity_response: {k} has the shape {v.shape}, expected {gs[k].shape} "
                             f"(the layout of the ground-state impurity results).")
    extras = {}
    for k, v in ret.items():
        if k in _Y_KEYS:
            continue
        try:
            a = np.asarray(v)
        except ValueError:
            a = None
        if a is not None and a.dtype.kind in "biufc":
            extras[k] = a
        else:
            app_log(1, f"[WARNING] impurity_response: extra key {k} ({type(v).__name__}) is not a numeric array; "
                       f"not stored.")
    return out, extras


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
# The Weiss-field response w, its mixing, and the diagnostic
# ---------------------------------------------------------------------------------------------------------

def _weiss_responses(imps, tot, y, embedding, embedding_2e, iaft):
    """Per impurity, the Weiss LR {"dg_weiss_w", "du_weiss_w", "dh0", "ddelta_w"} of the accumulated local responses
    tot and the impurity responses y."""
    loc_resp = _local_responses(tot, embedding, embedding_2e)
    return [dict(zip(_W_KEYS, _weiss_response(imp, loc_resp[i], y[i], iaft))) for i, imp in enumerate(imps)]


def _w_pack(w):
    """The metric vector of the outer residual and of the Anderson mixing: (dh0, ddelta_w, du_weiss_w) of all
    impurities as one real vector (real and imaginary parts, so that real-linear maps are covered)."""
    v = np.concatenate([wi[k].ravel() for wi in w for k in ("dh0", "ddelta_w", "du_weiss_w")])
    return np.concatenate([v.real, v.imag])


def _mix(history, a, b):
    """sum_m a_m w_in,m + b_m w_out,m of history = [(w_in, w_out), ...], every w a list over impurities of dicts with
    the keys _W_KEYS. The same coefficients on all four components keep them linear images of each other."""
    return [{k: sum(am * h[0][i][k] + bm * h[1][i][k] for am, bm, h in zip(a, b, history)) for k in _W_KEYS}
            for i in range(len(history[0][0]))]


class _Anderson:
    """
    Anderson acceleration of x = F(x) on a real vector over the last `history` differences, as mixing coefficients.
    With r = F(x) - x, x_new = x + beta r - (dX + beta dR) gamma, gamma the least-squares solution of dR gamma = r
    (dX, dR: differences of consecutive window entries). Written over the window pairs (x_m, F(x_m)) this is
    x_new = sum_m a_m x_m + b_m F(x_m). history = 0 is linear mixing, a = [1 - beta], b = [beta].
    """
    def __init__(self, history, mixing):
        self.history, self.mixing = history, mixing
        self._x, self._r = [], []

    def coefficients(self, x, fx):
        """Append (x, F(x)) to the window of the last history + 1 pairs; returns the coefficients (a, b) over it."""
        n = self.history + 1
        self._x = (self._x + [x.copy()])[-n:]
        self._r = (self._r + [fx - x])[-n:]
        beta, m = self.mixing, len(self._x) - 1
        a, b = np.zeros(m + 1), np.zeros(m + 1)
        a[m], b[m] = 1 - beta, beta
        if m > 0:
            dR = np.stack([r1 - r0 for r0, r1 in zip(self._r[:-1], self._r[1:])], axis=1)
            gamma = np.linalg.lstsq(dR, self._r[-1], rcond=None)[0]
            # dX + beta dR = (1 - beta) dX + beta dF with dF = dR + dX the difference of the F(x)
            for j in range(1, m + 1):
                a[j] -= (1 - beta) * gamma[j - 1]
                a[j - 1] += (1 - beta) * gamma[j - 1]
                b[j] -= beta * gamma[j - 1]
                b[j - 1] += beta * gamma[j - 1]
        return a, b


def _as_matrices(a, kind):
    """a as matrices on its last two axes; a bosonic rank-5 (nw, n, n, n, n) response is (nw, n^2, n^2)."""
    if kind == "b" and a.ndim == 5:
        return a.reshape(a.shape[0], a.shape[1] ** 2, a.shape[1] ** 2)
    return a


def _antihermitian_norms(w, y, iaft):
    """
    (|anti-Hermitian part| / |.| of w, same of y), over the components of all impurities, a log-only diagnostic.
    Fermionic dX(iw): (dX(iw) - dX(-iw)^dagger) / 2 on the pairs wn <-> -wn of the symmetric mesh wn_mesh('f')
    (CoQuí notation, odd n), dX(-iw) = dX[::-1]; bosonic (iv >= 0) and static dX: (dX - dX^dagger) / 2. None if the
    mesh is not symmetric.
    """
    wn = np.asarray(iaft.wn_mesh("f"))
    if not np.array_equal(wn, -wn[::-1]):
        return None

    def rel(items):
        num = den = 0.0
        for a, kind in items:
            a = _as_matrices(np.asarray(a), kind)
            partner = np.conj(np.swapaxes(a[::-1] if kind == "f" else a, -1, -2))
            num += np.linalg.norm(a - partner) ** 2 / 4
            den += np.linalg.norm(a) ** 2
        return float(np.sqrt(num / den)) if den > 0 else 0.0

    kinds_w = {"dg_weiss_w": "f", "ddelta_w": "f", "dh0": "static", "du_weiss_w": "b"}
    kinds_y = {"Vhf_imp": "static", "Sigma_imp_w": "f", "Pi_imp_w": "b"}
    return (rel([(wi[k], kinds_w[k]) for wi in w for k in _W_KEYS]),
            rel([(yi[k], kinds_y[k]) for yi in y for k in _Y_KEYS]))


# ---------------------------------------------------------------------------------------------------------
# Output file: the stages of the outer loop
# ---------------------------------------------------------------------------------------------------------

def _on_root(fn):
    """fn() on the root, its result on all ranks; an exception of fn is raised on all ranks (a root-only failure
    would leave the others in their next collective call)."""
    out = None
    if MPI.COMM_WORLD.Get_rank() == 0:
        try:
            out = (True, fn())
        except Exception as e:
            out = (False, (type(e) if isinstance(e, (ValueError, FileNotFoundError)) else RuntimeError, str(e)))
    ok, val = MPI.COMM_WORLD.bcast(out, root=0)
    if not ok:
        raise val[0](val[1])
    return val


def _read_array(ds):
    a = ds[()]
    return a[..., 0] + 1j * a[..., 1] if "__complex__" in ds.attrs else a


def _write_array(g, name, a):
    a = np.asarray(a)
    if np.iscomplexobj(a) and a.ndim == 0:
        g.create_dataset(name, data=np.array([a.real, a.imag])).attrs["__complex__"] = np.int8(1)
    elif np.iscomplexobj(a):
        write_complex_dataset(g, name, a)
    else:
        g.create_dataset(name, data=a)


def _read_pairing(coqui_chkpt, solver_chkpt, iteration):
    """(K, J): the embedding iteration K (``iteration`` or embed/final_iter) and the dmft iteration J =
    embed/iter{K}/dmft_iter whose impurity results K embeds."""
    def read():
        with h5py.File(coqui_chkpt, "r") as f:
            if "embed" not in f:
                raise ValueError(f"run_lr_gw_edmft: {coqui_chkpt} has no embed group, i.e. no GW+EDMFT ground state.")
            K = int(f["embed/final_iter"][()]) if iteration is None else int(iteration)
            if f"embed/iter{K}" not in f:
                raise ValueError(f"run_lr_gw_edmft: {coqui_chkpt} has no embed/iter{K}.")
            if "dmft_iter" not in f[f"embed/iter{K}"]:
                raise ValueError(f"run_lr_gw_edmft: {coqui_chkpt}: embed/iter{K}/dmft_iter is absent. It links the "
                                 f"embedding to the dmft iteration whose impurity results it embeds and is written "
                                 f"by coqui.dmft_embed(..., dmft_iter=).")
            J = int(f[f"embed/iter{K}/dmft_iter"][()])
        if not os.path.isfile(solver_chkpt):
            raise FileNotFoundError(f"run_lr_gw_edmft: no impurity results in {solver_chkpt}.")
        with h5py.File(solver_chkpt, "r") as f:
            if f"dmft/iter{J}/impurity_0/results" not in f:
                raise ValueError(f"run_lr_gw_edmft: embed/iter{K} embeds dmft iteration {J}, which has no results "
                                 f"in {solver_chkpt}.")
        return K, J
    return _on_root(read)


class _OuterState:
    """
    The stages of the outer loop in the group ``lr_gw_edmft`` of the output file, written by the root. A group counts
    as complete only when its attribute ``done`` is set, which is written last. Names are relative to ``lr_gw_edmft``.
    """
    G = "lr_gw_edmft"

    def __init__(self, path):
        self.path = path

    def start(self, meta, restart):
        """Open the file for a new run (truncate it, write ``meta``) or, with restart and a file that has ``meta``,
        check ``meta`` against the stored one; a mismatch raises and leaves the file alone."""
        def work():
            if restart and os.path.isfile(self.path):
                with h5py.File(self.path, "r") as f:
                    if f"{self.G}/meta" in f and f[f"{self.G}/meta"].attrs.get("done", False):
                        stored = dict(f[f"{self.G}/meta"].attrs)
                        return [k for k, v in meta.items() if k not in stored or not np.array_equal(stored[k], v)]
            with h5py.File(self.path, "w") as f:
                g = f.create_group(f"{self.G}/meta")
                g.attrs.update(meta)
                g.attrs["done"] = True
            return []
        bad = _on_root(work)
        if bad:
            raise ValueError(f"run_lr_gw_edmft: {self.path} was written for a different {bad}; use another output "
                             f"or restart=False.")

    def has(self, name):
        def work():
            with h5py.File(self.path, "r") as f:
                g = f.get(f"{self.G}/{name}")
                return g is not None and bool(g.attrs.get("done", False))
        return _on_root(work)

    def get(self, name):
        """(datasets, attributes) of a group."""
        def work():
            with h5py.File(self.path, "r") as f:
                g = f[f"{self.G}/{name}"]
                return {k: _read_array(v) for k, v in g.items() if isinstance(v, h5py.Dataset)}, dict(g.attrs)
        return _on_root(work)

    def put(self, name, data=None, attrs=None):
        """Replace the group by a complete one."""
        def work():
            with h5py.File(self.path, "a") as f:
                path = f"{self.G}/{name}"
                if path in f:
                    del f[path]
                g = f.require_group(path)
                for k, v in (data or {}).items():
                    _write_array(g, k, v)
                g.attrs.update(attrs or {})
                g.attrs["done"] = True
        _on_root(work)

    def close_group(self, name, attrs=None):
        """Mark a group (whose subgroups were put) complete."""
        def work():
            with h5py.File(self.path, "a") as f:
                g = f.require_group(f"{self.G}/{name}")
                g.attrs.update(attrs or {})
                g.attrs["done"] = True
        _on_root(work)

    def final_valid(self, n_outer, need_dG):
        """The final solve of the stored loop (n_outer iterations) is complete, with DeltaG_tskij if need_dG; it
        depends on y_{n_outer - 1} only."""
        def work():
            with h5py.File(self.path, "r") as f:
                a = f[self.G].attrs
                return bool(a.get("done_final", False) and "linear_response" in f and a.get("n_outer") == n_outer
                            and (not need_dG or "DeltaG_tskij" in f["linear_response"]))
        return _on_root(work)

    def invalidate_final(self):
        def work():
            with h5py.File(self.path, "a") as f:
                if "linear_response" in f:
                    del f["linear_response"]
                f[self.G].attrs["done_final"] = False
        _on_root(work)

    def read_final(self):
        def work():
            with h5py.File(self.path, "r") as f:
                g = f["linear_response"]
                return {"dGloc_tsIab": _read_array(g["DeltaGloc_tsIab"]), "dWloc_tabcd": _read_array(g["DeltaWloc_tabcd"]),
                        "niter": int(f[self.G].attrs["final_run_lr_iterations"])}
        return _on_root(work)

    def set_final(self, converged, residuals, niter):
        def work():
            with h5py.File(self.path, "a") as f:
                g = f[self.G]
                if "residuals" in g:
                    del g["residuals"]
                g.create_dataset("residuals", data=np.asarray(residuals))
                g.attrs.update(converged=converged, n_outer=len(residuals), final_run_lr_iterations=niter,
                               done_final=True)
        _on_root(work)


def _read_local(st, name):
    """The accumulated local responses of a ``local`` group."""
    data, attrs = st.get(name)
    return dict({k: data[k] for k in _LOCAL_KEYS}, DeltaN=attrs["DeltaN"])
