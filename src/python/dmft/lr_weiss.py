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

import numpy as np

from coqui import app_log

from .weiss import compute_weiss_fermion_w, density_density_to_product_basis


def lr_weiss_boson_w(V_abcd, W_wabcd, dW_wabcd, Pi_wabcd, dPi_wabcd):
    """
    Linear response of compute_weiss_boson_w, u = U - V with U = W_f (1 + Pi W_f)^{-1} and
    W_f = W + V (product basis, (n^2, n^2) per iv). The bare V does not respond, so dW_f = dW and du = dU:
        du = (1 + W_f Pi)^{-1} dW (1 + Pi W_f)^{-1} - U dPi U.
    Both ground-state factors use the full W_f and U = u + V.
    Pi, dPi are either in the product basis (nw, n, n, n, n) or density-density (nw, n, n), which is placed
    at [a,a,b,b] by density_density_to_product_basis. The ground-state pinv is replaced by solve/inv, which is
    the same where 1 + Pi W_f is nonsingular (compute_weiss_boson_w warns when its condition number exceeds 100).

    Returns du in the shape of W_wabcd.
    """
    if Pi_wabcd.ndim == 3:
        Pi_wabcd = density_density_to_product_basis(Pi_wabcd)
    if dPi_wabcd.ndim == 3:
        dPi_wabcd = density_density_to_product_basis(dPi_wabcd)

    nbnd = V_abcd.shape[0]
    nbnd2 = nbnd*nbnd
    Wfull_pb = (W_wabcd + V_abcd[np.newaxis, ...]).reshape(-1, nbnd2, nbnd2)
    Pi_pb = Pi_wabcd.reshape(-1, nbnd2, nbnd2)
    dW_pb = dW_wabcd.reshape(-1, nbnd2, nbnd2)
    dPi_pb = dPi_wabcd.reshape(-1, nbnd2, nbnd2)

    dU_pb = np.zeros(Wfull_pb.shape, dtype=complex)
    for n, W in enumerate(Wfull_pb):
        left = np.linalg.solve(np.eye(nbnd2) + W @ Pi_pb[n], dW_pb[n])
        right = np.linalg.inv(np.eye(nbnd2) + Pi_pb[n] @ W)
        U = W @ right
        dU_pb[n] = left @ right - U @ dPi_pb[n] @ U

    return dU_pb.reshape(W_wabcd.shape)


def lr_compute_weiss_fields_w(*, iaft, local_gf, d_local_gf, impurity_selfenergies, d_impurity_selfenergies):
    """
    Linear response of compute_weiss_fields_w: (dg_weiss_w, du_weiss_w, dh0_sab, ddelta_wsab) for responses of the
    local lattice quantities, d_local_gf = {"Hloc", "Delta_w", "Wloc_t"} (dVloc = 0), and of the impurity
    self-energies, d_impurity_selfenergies = {"Vhf_imp", "Pi_imp_w"}, in the layouts of the ground-state dicts
    (the response of Sigma_imp enters through Delta_w; a "Sigma_imp_w" key in d_impurity_selfenergies is ignored).
    The ground state local_gf = {"Gloc_w", "Wloc_t", "Vloc"} is that of compute_weiss_fields_w, with the pointwise
    G^loc(iw). The ground-state Hloc is not needed: the map is linear in dHloc.
    Wloc_t, Vloc and d Wloc_t are in the product basis, as the driver's solver inputs: apply
    chemistry_to_product_basis to the chemistry-order output of downfold_coulomb / lr_downfold_coulomb first.

    The ground state splits g_weiss^{-1}(iw) = iw - h0 - Delta(iw) with h0 = Hloc - Vhf_imp, so
        dh0 = dHloc - dVhf_imp,    dg_weiss = g_weiss (dh0 + dDelta) g_weiss.
    dDelta is run_lr's DeltaDelta_wsIab (extracted to the impurity blocks), passed as d_local_gf["Delta_w"]
    (nw_f, 2, n, n):
        dDelta = G^{-1} dG^loc G^{-1} - dHloc - dSigma_imp,
    with G^loc and dG^loc the pointwise-in-iw local Green's function and its response (see run_lr), so that dDelta is
    the derivative of the ground-state map. For a spin-unpolarized lattice it holds the spin-mean dSigma_imp that the
    lattice was given and is the same for both spins.
    There is no Hermitian projection of dh0 and no pairing of dDelta(iw) with dDelta(-iw): at Q != 0 the response is
    complex and dDelta(-iw) is not dDelta(iw)^dagger.
    dDelta subtracts the projected P/Sym(dHloc) of run_lr, while dh0 uses d_local_gf["Hloc"] as given, so dg_weiss is
    the derivative of the ground-state map only for a perturbation with proj(dHloc) = dHloc: a time-reversal-even
    Hermitian one (as in lr_scf_driver), for which P and Sym are the identity on dHloc.

    Defined only for causal_projection = None: fit_local_results_boson and fit_u_weiss are nonlinear bath fits,
    and for set_sigma_infty_to_dc = False: Vhf_imp is the impurity's own Sigma_infty, as in the ground state.
    Covers screen_type = "gw_edmft" only; the Pi_imp response of screen_type = "rpa" is
    coqui.embed.lr_eval_pi_rpa_cells with a single cell.
    """
    missing = {"Gloc_w", "Wloc_t", "Vloc"} - local_gf.keys()
    if missing:
        raise ValueError(f"Missing keys in local_gf: {missing}")
    missing = {"Hloc", "Delta_w", "Wloc_t"} - d_local_gf.keys()
    if missing:
        raise ValueError(f"Missing keys in d_local_gf: {missing}")
    missing = {"Vhf_imp", "Sigma_imp_w", "Pi_imp_w"} - impurity_selfenergies.keys()
    if missing:
        raise ValueError(f"Missing keys in impurity_selfenergies: {missing}")
    missing = {"Vhf_imp", "Pi_imp_w"} - d_impurity_selfenergies.keys()
    if missing:
        raise ValueError(f"Missing keys in d_impurity_selfenergies: {missing}")

    app_log(2, "Evaluate the linear response of the bosonic Weiss field.")
    du_weiss_w = lr_weiss_boson_w(
        local_gf["Vloc"],
        iaft.tau_to_w_phsym(local_gf["Wloc_t"], stats='b'),
        iaft.tau_to_w_phsym(d_local_gf["Wloc_t"], stats='b'),
        impurity_selfenergies["Pi_imp_w"],
        d_impurity_selfenergies["Pi_imp_w"]
    )
    iaft.check_leakage_phsym(du_weiss_w, 'b', 'du_weiss', w_input=True)
    app_log(2, "")

    app_log(2, "Evaluate the linear response of the fermionic Weiss field.")
    g_weiss_w = compute_weiss_fermion_w(local_gf["Gloc_w"], impurity_selfenergies["Vhf_imp"],
                                        impurity_selfenergies["Sigma_imp_w"])
    dh0_sab = d_local_gf["Hloc"] - d_impurity_selfenergies["Vhf_imp"]
    ddelta_wsab = d_local_gf["Delta_w"]
    dg_weiss_w = np.einsum('wsab,wsbc,wscd->wsad', g_weiss_w, dh0_sab[None] + ddelta_wsab, g_weiss_w)
    iaft.check_leakage(dg_weiss_w, 'f', 'dg_weiss', w_input=True)
    app_log(2, "")
    iaft.check_leakage(ddelta_wsab, 'f', 'ddelta', w_input=True)
    app_log(2, "")

    return dg_weiss_w, du_weiss_w, dh0_sab, ddelta_wsab
