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

Charge susceptibility and dielectric matrix from linear response.

Drives the LR stack with an external *scalar* plane-wave field instead of an
electron-phonon perturbation: one LR mode per plane wave e^{i(q+G').r}, and the
free-energy hessian of that mode set is chi(q) up to the cell volume. The kernel
ladder decides which approximation chi is (none = chi0, Hartree = RPA, HF =
TDHF, HSEX = BSE, GW = dynamical BSE).

Numerics live in C++: this module builds the perturbations
(mean_field.compute_pw_matrix_elements), calls run_lr(..., hessian="symmetric") and
divides by the volume.

Two susceptibilities come out of a Hartree-rung kernel. ``chi`` is the full reducible
one, chi = chi_bar + chi_bar v chi, with v(q+G) = 4*pi/|q+G|^2 in every channel.
``chi_bar`` is the one whose Hartree rung omits the G = 0 term of v(q+G), the
macroscopically unscreened response, finite at q = 0 and well conditioned at small q
(the full-chi solve divides out eps_M). ``exclude_hartree_head=True`` solves for
chi_bar directly; chi is then rebuilt from it on the G block. The macroscopic
dielectric constant is eps_M(q) = 1 - v(q) chi_bar_00(q).
"""

import warnings

import numpy as np

from .mbpt_driver import run_lr
from .lr_driver import read_lr_hessian, resolve_kernel

# Passed to run_lr on every call. run_lr's own default is unaccelerated Picard,
# whose convergence rate is the spectral radius of the LR kernel, so the
# accelerator is never left to that default here.
DEFAULT_ITER_ALG = {"alg": "DIIS", "mixing": 0.7}

# |q+G| below this counts as the divergent q+G = 0 column of v(q+G).
_QPG_ZERO = 1e-8


def _full_and_bar_chi(chi_solved, i0, v0, solved_is_bar):
    r"""
    (chi, chi_bar) from the matrix a Hartree-rung LR solve returned.

    Called by run_chi for ``chi`` and ``chi_plain``. The solve returns chi_bar when
    ``solved_is_bar`` (``exclude_hartree_head``) or at q = 0 (``v0`` = inf, where
    the THC has no G = 0 term), else the full chi. The two differ by the head channel
    G = 0 (row ``i0``) of v(q) alone:

        chi = chi_bar + chi_bar_{G0} v chi_bar_{0G'} / (1 - v chi_bar_{00}),
        chi_bar = chi - chi_{G0} v chi_{0G'} / (1 + v chi_{00}).

    At q = 0 chi has no finite head, so the solved matrix is returned unchanged as
    both and run_chi masks the head row and column of chi.
    """
    if not np.isfinite(v0):  # q = 0: the solve is chi_bar, no finite full chi
        return chi_solved, chi_solved
    # Rank-1 head channel in either direction
    col, row = chi_solved[:, i0], chi_solved[i0, :]
    if solved_is_bar:  # Solved chi_bar -> rebuild chi
        return chi_solved + v0 * np.outer(col, row) / (1.0 - v0 * chi_solved[i0, i0]), chi_solved
    # Solved chi -> derive chi_bar
    return chi_solved, chi_solved - v0 * np.outer(col, row) / (1.0 + v0 * chi_solved[i0, i0])


def run_chi(params, h_int, mf, q_vec, G_mill,
            method="Hartree",
            unperturbed="checkpoint",
            max_iter=100,
            tol=1e-10,
            iter_alg=None,
            fix_density=False,
            include_hartree=True,
            include_exchange=True,
            gw_mode="none",
            div_treatment=None,
            hessian="symmetric",
            exclude_hartree_head=False,
            **kwargs):
    r"""
    Charge susceptibility chi_{G G'}(q) and inverse dielectric matrix at nu = 0.

    One LR perturbation per plane-wave column: the probe of column G' is the bare
    field e^{i(q+G').r} in the band basis. All nG columns are solved in a single
    run_lr call with ``hessian="symmetric"``, and the mode-pair matrix that call returns
    *is* chi:

        Omega * chi_{G G'}(q) = Tr(DeltaH0_G, DeltaDm_{G'})
                              = spin sum_{s,k} w_k sum_{ij}
                                    conj(M^G_{ij}(k)) DeltaDm^{G'}_{ij}(k),

    which is lr_hessian_t's own contraction (lr_hessian.hpp section 4) with
    ``DeltaH0_G = M^G``. Static (nu = 0) only.

    Parameters
    ----------
    params : dict
        Params dict handed to run_lr ("prefix", "output", "include_xc", ...).
    h_int : ThcCoulomb
        THC ERI handler.
    mf : Mf
        Mean field providing the orbitals for the plane-wave matrix elements.
        Must be the one the checkpoint was made with, with npol = 1 and a
        full-BZ k-grid.
    q_vec : array_like, shape (3,)
        Perturbation wavevector in crystal (fractional) coordinates.
    G_mill : array_like of int, shape (nG, 3)
        Miller indices of the plane waves. The returned matrices are ordered as
        this list. Validated by compute_pw_matrix_elements.
    method : str or None, optional
        Kernel ladder alias (default "Hartree" = RPA); see resolve_kernel. None
        falls back to include_hartree / include_exchange / gw_mode, whose
        defaults are Hartree *and* exchange -- so ``method=None`` alone gives
        TDHF, not RPA.
    unperturbed : str, optional
        Unperturbed reference, "checkpoint" (default) or "mf_dft".
    max_iter, tol : int, float, optional
        LR SCF iteration cap and tolerance (default 100, 1e-10). A column that
        reaches the cap is a hard error, not a result. With ``tol = 0`` every
        column runs exactly ``max_iter`` iterations by construction and the
        check is skipped (an order-n chi, see ``hessian``).
    iter_alg : dict or None, optional
        LR accelerator, as in run_lr. None (default) means DEFAULT_ITER_ALG.
    fix_density : bool, optional
        Present only so that a True can be rejected (see Raises).
    include_hartree, include_exchange, gw_mode : optional
        Kernel flags, used only when ``method is None``.
    div_treatment : str or None, optional
        Divergence treatment. Accepted only with ``unperturbed="mf_dft"``.
    hessian : {"symmetric", "plain"}, optional
        "symmetric" (default): ``chi = hessian_sym/Omega`` from the stationary
        estimator. "plain": ``chi = hessian/Omega`` from the plain contraction
        only (no stationary block is computed); with it an order-n chi is damping
        with mixing 1, a ``precond_method``, ``max_iter = n``, ``tol = 0``.
        ``chi_plain`` is returned either way. "none", bools and other values raise.
    exclude_hartree_head : bool, optional
        Default False. Solve for chi_bar (see run_lr) and rebuild ``chi`` from it (see
        Returns). Requires ``G = 0`` in ``G_mill`` and ``q_vec`` in ``mf.qpts_ibz()``.
    **kwargs
        Forwarded to run_lr unchanged (e.g. recompute_W, screened_interaction_file,
        precond_method, precond_tol, precond_max_iter, reuse_W); see run_lr.

    Returns
    -------
    dict
        ``q_vec`` (3,); ``G_mill`` (nG, 3) as given; ``qpG_cart`` (nG, 3) q + G
        in Cartesian a.u.; ``v_qpG`` (nG,) = 4*pi/|q+G|^2, no 1/Omega, inf at
        q+G = 0; ``chi`` (nG, nG) complex, the full reducible response from the
        stationary estimator, row = response G, column = probe G'; ``chi_plain``
        (nG, nG) from the plain one; ``chi_bar`` (nG, nG) and ``eps_M`` (scalar,
        complex) with the Hartree rung on, else None, defined by

            chi_bar_{GG'} = chi_{GG'} - chi_{G0} v(q) chi_{0G'} / (1 + v(q) chi_{00}),
            eps_M = 1 - v(q) chi_bar_{00}  (= 1/eps_inv[0,0] at q != 0),

        with G = 0 the zero row of ``G_mill`` and v(q) = 4*pi/|q|^2. With
        ``exclude_hartree_head`` chi_bar is the solved matrix and
        chi = chi_bar + chi_bar_{G0} v(q) chi_bar_{0G'} / (1 - v(q) chi_bar_{00}), exact
        on the G block. ``chi_bar`` and ``eps_M`` are None without G = 0 in ``G_mill``
        (and with ``exclude_hartree_head`` that raises). At q + G = 0 (q = 0, Hartree
        rung on) the solved matrix is chi_bar itself, so ``chi_bar`` is that matrix, the
        q + G = 0 row and column of ``chi`` and ``chi_plain`` are NaN (the full chi has
        no finite head there), and ``eps_M`` is NaN. ``eps_inv`` (nG, nG);
        ``niter`` (nG,) and ``delta_mu`` (nG,) per column; ``metadata`` with the
        resolved kernel, the solver settings and the C++ hermiticity residuals.

    Raises
    ------
    ValueError
        - ``mf.nkpts_ibz() != mf.nkpts()`` or ``mf.npol() != 1``. The k axis of
          the hessian contraction is the IBZ, so a symmetry-reduced mean field
          would sum the wrong k-set. Both are C++ ``utils::check``\ s, i.e. an
          MPI_Abort a q/G sweep driver cannot catch, so they are re-checked here.
        - ``exclude_hartree_head=True`` without the Hartree rung, without G = 0 in
          ``G_mill``, or with a ``q_vec`` that is not a point of ``mf.qpts_ibz()``.
        - A nonzero reciprocal-lattice ``q_vec`` (equivalent to Gamma) with the
          Hartree rung and G = 0 in ``G_mill``: the THC solve is chi_bar there, but
          chi, chi_bar and eps_M would be built from a finite v(q). Pass q_vec = 0.
        - ``fix_density=True``. At q = 0 it enforces DeltaN = 0 and so projects
          out chi_00(q->0) = dn/dmu, the quantity being measured; at q != 0 it is
          a silent no-op, so a q-conditional guard would let a sweep pass
          everywhere except the one point where it matters.
        - ``unperturbed`` other than "checkpoint" / "mf_dft", or a
          ``div_treatment`` on any route but "mf_dft" (both C++ ``utils::check``).
    RuntimeError
        A column that hit `max_iter`. An unconverged column is not a chi.

    Notes
    -----
    **Conventions.** compute_pw_matrix_elements returns
    ``M[G,s,k,i,j] = <psi_{k+q,i}| e^{i(q+G).r} |psi_{k,j}>`` with the orbitals
    normalized to ``(1/nnr) sum_r |u|^2 = 1``, so ``M[G']`` is directly the
    band-basis matrix of a unit-amplitude external field: no scaling before it is
    handed to run_lr. Its ``conj``-on-i, ``(i = k+q, j = k)`` slot order is the
    hessian contraction's own, and the hessian's ``A^{-q} = [A^{q}]^dagger``
    premise is exact for a plane wave, so nothing about the pairing is assumed
    here. The induced density is then in electrons per cell, and CoQuí's Coulomb
    kernel carries no ``1/Omega`` either (potentials/coulomb.hpp:50), hence

        chi = hessian / Omega,   eps^-1 = 1 + v(q+G) chi,   v = 4*pi/|q+G|^2.

    **chi_bar and the q = 0 solve.** At q = 0 the THC Coulomb has no G = 0 term, so a
    Hartree-rung solve returns chi_bar there, with or without ``exclude_hartree_head``;
    the full chi has no finite G = 0 row and column, hence the NaN above. Away from
    q = 0 the full-v solve returns chi and chi_bar is derived from it. At q = 0
    ``eps_inv = 1 + v chi`` is evaluated from the solved chi_bar, so its G' = 0 column
    (rows G != 0) is finite and not the NaN-masked ``chi[:, 0]``; its G = 0 row is NaN.

    **The q = 0 head is not trustworthy on a correlated checkpoint.** A probe
    with G' != 0 has zero cell average and so cannot change N in a gapped system
    at fixed mu, i.e. ``chi_{0,G'}(q=0) = 0`` exactly. On lr_tests/Si that sum
    rule holds to 1.4e-6 relative on ``checkpoint_hf`` but fails at 8.4e-3 on the
    scGW ``checkpoint_gw``, so the anomaly is in the reference G rather than in
    this contraction. Until it is explained, treat the whole first row *and*
    column of chi at q = 0 on a correlated reference as contaminated at the
    percent level -- they come back as finite numbers in ``eps_inv`` and block
    any q -> 0 eps_inf claim.

    **Cost.** The root holds the whole M, ``nG*nspin*nkpts*nb^2*16`` bytes, and a
    full chi matrix cannot avoid it: column G' is the probe and row G the
    projector. The hessian adds three striped static stores of that size and one
    extra Dyson solve per column; with ``gw_mode != "none"`` it adds two
    ``nG``-deep omega stores, which is what binds there. Splitting the G list
    across several run_lr calls is not a lever: each call rebuilds lr_setup (W is
    kept between calls only with ``reuse_W``), and a call's hessian block spans
    only its own columns, so it would return diagonal sub-blocks rather than a
    chi matrix.

    """
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # ---- validation. Every input here is replicated, so all ranks raise alike,
    # and all of it precedes the first collective.
    if mf.nkpts_ibz() != mf.nkpts():
        raise ValueError(
            f"run_chi: needs a full-BZ mean field, got nkpts_ibz = "
            f"{mf.nkpts_ibz()} != nkpts = {mf.nkpts()}.")
    if mf.npol() != 1:
        raise ValueError(f"run_chi: requires npol = 1, got {mf.npol()}.")
    if fix_density:
        raise ValueError(
            "run_chi: fix_density must be False. At q = 0 it enforces "
            "DeltaN = 0 and thereby projects out chi_00(q->0) = dn/dmu, the "
            "quantity being measured; at q != 0 it is a silent no-op.")
    if not isinstance(hessian, str):
        raise TypeError(
            f"run_chi: hessian must be 'symmetric' or 'plain', got {hessian!r}.")
    if hessian not in ("symmetric", "plain"):
        raise ValueError(
            f"run_chi: hessian must be 'symmetric' or 'plain', got {hessian!r}.")
    if unperturbed not in ("checkpoint", "mf_dft"):
        raise ValueError(
            f"run_chi: unperturbed must be 'checkpoint' or 'mf_dft', got "
            f"{unperturbed!r}.")
    if div_treatment is not None and unperturbed != "mf_dft":
        raise ValueError(
            f"run_chi: div_treatment is accepted only for unperturbed='mf_dft', "
            f"not {unperturbed!r}. On that route the LR run reuses the "
            "ground-state treatment read from the checkpoint, so that Delta_Sigma "
            "matches how Sigma was built.")

    q_vec = np.ascontiguousarray(q_vec, dtype=np.float64).reshape(-1)
    if q_vec.size != 3:
        raise ValueError(f"run_chi: q_vec must have 3 entries, got {q_vec.size}")
    G_mill = np.ascontiguousarray(G_mill, dtype=np.int64)
    if G_mill.ndim == 1:
        G_mill = G_mill.reshape(1, -1)
    nG = G_mill.shape[0]
    if G_mill.ndim == 2 and len({tuple(g) for g in G_mill.tolist()}) != nG:
        raise ValueError("run_chi: G_mill has repeated entries")

    hartree, exchange, gw = resolve_kernel(
        method, include_hartree, include_exchange, gw_mode, caller="run_chi")
    iter_alg = dict(DEFAULT_ITER_ALG if iter_alg is None else iter_alg)

    # Row of G = 0, whose q+G plane wave is the head of v(q+G)
    G_zero_rows = np.flatnonzero(~G_mill.any(axis=1))
    i0 = int(G_zero_rows[0]) if G_zero_rows.size else None
    if exclude_hartree_head and not hartree:
        raise ValueError(
            "run_chi: exclude_hartree_head=True requires the Hartree rung (method "
            f"{method!r} has none).")
    if exclude_hartree_head and i0 is None:
        raise ValueError(
            "run_chi: exclude_hartree_head=True needs G = 0 in G_mill: chi is rebuilt "
            "from chi_bar through the G = 0 channel.")
    # The THC has no G = 0 term of v at any q equivalent to Gamma (it works modulo G),
    # so its solve is chi_bar there, but |q+G| of this call's rows sees a finite v(q).
    q_is_gamma_image = bool(np.allclose(q_vec, np.round(q_vec), atol=1e-8))
    if hartree and i0 is not None and q_is_gamma_image and np.abs(q_vec).max() > 1e-8:
        raise ValueError(
            f"run_chi: q_vec = {q_vec.tolist()} is a nonzero reciprocal lattice vector, "
            "equivalent to Gamma: the THC solve is chi_bar there while the rows of "
            "this call see a finite v(q), so chi, chi_bar and eps_M would be wrong. "
            "Pass q_vec = 0 and the plane waves through G_mill.")

    from coqui.mean_field import compute_pw_matrix_elements

    # M is root-only; the remaining shape checks are that call's.
    M, qpG_cart = compute_pw_matrix_elements(mf, q_vec, G_mill)

    if exclude_hartree_head:
        # The THC removes the G = 0 term at its own q image; that must be q_vec itself,
        # since chi, chi_bar and eps_M below use the G = 0 row of q_vec.
        q_cart = qpG_cart[i0]
        dq = np.abs(np.asarray(mf.qpts_ibz()) - q_cart[None, :]).max(axis=1)
        if dq.min() > 1e-8:
            raise ValueError(
                f"run_chi: exclude_hartree_head=True needs q_vec to be a q-point of the "
                f"THC (one of mf.qpts_ibz()); q_vec = {q_vec.tolist()} (crystal) -> "
                f"{q_cart.tolist()} a.u. is none of them (nearest differs by "
                f"{dq.min():.2e}). Pass its THC image q_vec, since the removed term is "
                "the image's plane wave.")

    # ---- one solve for the whole column set: the setup and W are shared, and
    # the hessian block is (nG, nG) over exactly the columns of this call.
    # save_DeltaG=False because nothing here reads DeltaG(tau), and for a
    # Sigma-free kernel the hessian does not either.
    niter, delta_mu = run_lr(
        params, h_int=h_int, q_vec=q_vec,
        DeltaH0_skij=(M if rank == 0 else None),
        method=method,
        include_hartree=hartree, include_exchange=exchange, gw_mode=gw,
        exclude_hartree_head=exclude_hartree_head, unperturbed=unperturbed,
        max_iter=max_iter, tol=tol, fix_density=False, iter_alg=iter_alg,
        save_DeltaG=False, hessian=hessian, div_treatment=div_treatment,
        **kwargs)
    niter = np.atleast_1d(niter).astype(np.int64)
    delta_mu = np.atleast_1d(delta_mu).astype(np.float64)

    # A stalled column reports max_iter + 1 (lr_solve_one hands back the loop
    # variable), so max_iter itself is a legitimate last-iteration convergence.
    # At tol = 0 nothing converges: every column runs max_iter iterations by design.
    stalled = np.flatnonzero(niter > max_iter) if tol != 0 else np.empty(0, dtype=np.int64)
    if stalled.size:
        raise RuntimeError(
            f"run_chi: {stalled.size} of {nG} columns hit max_iter = {max_iter} "
            f"without converging to tol = {tol:g}: G = "
            f"{G_mill[stalled].tolist()}. An unconverged column is not a "
            "susceptibility.")

    # Only the root touches the HDF5; a failure there becomes a verdict for
    # every rank rather than being left in the broadcast.
    payload = None
    if rank == 0:
        try:
            out_file = str(params.get("output", params["prefix"])) + ".mbpt.h5"
            hess = read_lr_hessian(out_file)
            if not np.array_equal(hess["hessian_call_index"], np.arange(nG)):
                raise ValueError(
                    "run_chi: the hessian block does not span the requested "
                    f"columns: call_index = {hess['hessian_call_index'].tolist()}")
            payload = ("ok", hess)
        except Exception as exc:                                      # noqa: BLE001
            payload = ("error", f"{type(exc).__name__}: {exc}")
    payload = comm.bcast(payload, root=0)
    if payload[0] != "ok":
        raise RuntimeError(f"run_chi: reading back the hessian failed: {payload[1]}")
    hess = payload[1]

    volume = mf.volume()
    chi_plain_solved = hess["hessian"] / volume
    # hessian="plain" has only the plain block.
    chi_solved = (hess["hessian_sym"] / volume if hessian == "symmetric"
                  else chi_plain_solved)

    # v is reported as inf exactly where the eps_inv row is NaN, so a caller can
    # select the usable rows on either array and get the same answer.
    qpG_abs = np.linalg.norm(qpG_cart, axis=1)
    head = qpG_abs < _QPG_ZERO
    with np.errstate(divide="ignore"):
        v_qpG = np.where(head, np.inf, 4.0 * np.pi / np.where(head, 1.0, qpG_abs)**2)

    # Full chi and chi_bar from the solved matrix (chi_bar when exclude_hartree_head
    # or q = 0, chi otherwise); eps_M from chi_bar.
    v0 = v_qpG[i0] if i0 is not None else None   # v(q); inf at q = 0
    if hartree and i0 is not None:  # Hartree on with G = 0: chi and chi_bar from the solve
        chi, chi_bar = _full_and_bar_chi(chi_solved, i0, v0, exclude_hartree_head)
        chi_plain, _ = _full_and_bar_chi(chi_plain_solved, i0, v0, exclude_hartree_head)
        eps_M = (complex(np.nan, np.nan) if head[i0]
                 else complex(1.0 - v0 * chi_bar[i0, i0]))
    else:  # Hartree off or no G = 0: chi = solved, no chi_bar
        chi, chi_plain, chi_bar, eps_M = chi_solved, chi_plain_solved, None, None

    eps_inv = (np.eye(nG, dtype=np.complex128)
               + np.where(head, 0.0, v_qpG)[:, None] * chi)
    if hartree and head.any():
        # The solve is chi_bar here: the full chi has no finite q+G = 0 row and column.
        chi = chi.copy()
        chi_plain = chi_plain.copy()
        for c in (chi, chi_plain):
            c[head, :] = np.nan
            c[:, head] = np.nan
    if head.any():
        eps_inv[head, :] = np.nan
        if rank == 0:
            warnings.warn(
                f"run_chi: |q+G| = 0 for G = {G_mill[head].tolist()}, where "
                "v(q+G) diverges; those rows of eps_inv are NaN. chi itself is "
                "finite there.", stacklevel=2)

    return {
        "q_vec": q_vec,
        "G_mill": G_mill,
        "qpG_cart": qpG_cart,
        "v_qpG": v_qpG,
        "chi": chi,
        "chi_plain": chi_plain,
        "chi_bar": chi_bar,
        "eps_M": eps_M,
        "eps_inv": eps_inv,
        "niter": niter,
        "delta_mu": delta_mu,
        "metadata": {
            "method": method,
            "include_hartree": hartree,
            "include_exchange": exchange,
            "gw_mode": gw,
            "include_xc": bool(params.get("include_xc", False)),
            "exclude_hartree_head": bool(exclude_hartree_head),
            "unperturbed": unperturbed,
            "max_iter": max_iter,
            "tol": tol,
            "iter_alg": iter_alg,
            "hessian": hessian,
            "volume": volume,
            # C++ diagnostics: the antihermitian part of chi (chi_{GG'} =
            # conj(chi_{G'G}) is exact) and of the kernel pairing.
            "herm_dev": hess.get("hessian_sym_herm_dev"),
            "herm_dev_plain": hess["hessian_herm_dev"],
            "herm_dev_M": hess.get("hessian_M_herm_dev"),
        },
    }
