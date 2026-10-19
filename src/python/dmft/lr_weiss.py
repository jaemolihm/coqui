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

from .weiss import compute_weiss_fermion_w, density_density_to_product_basis, estimate_zero_moment


def lr_weiss_fermion_w(g_wsab, dg_wsab, vhf_sab, dvhf_sab, sigma_wsab, dsigma_wsab):
    """
    Linear response of compute_weiss_fermion_w, g_weiss = [G^{-1} + vhf + Sigma]^{-1}, per (iw, s):
        d(g_weiss^{-1}) = -G^{-1} dG G^{-1} + dvhf + dSigma,   dg_weiss = -g_weiss d(g_weiss^{-1}) g_weiss.
    vhf (Sigma_infty of the impurity) and Sigma (its dynamic part) enter as in the ground state.
    All factors are local and pointwise in iw, so a response at momentum transfer Q uses the same
    ground-state factors and no -Q partner.

    Returns (dg_weiss_wsab, dg_weiss_inv_wsab).
    """
    g_weiss_wsab = compute_weiss_fermion_w(g_wsab, vhf_sab, sigma_wsab)

    nw, nspin = g_wsab.shape[:2]
    dg_weiss_wsab = np.zeros(g_wsab.shape, dtype=complex)
    dg_weiss_inv_wsab = np.zeros(g_wsab.shape, dtype=complex)
    for ws in range(nw*nspin):
        w = ws // nspin
        s = ws % nspin
        g_inv = np.linalg.inv(g_wsab[w, s])
        dX_inv = -g_inv @ dg_wsab[w, s] @ g_inv + dvhf_sab[s] + dsigma_wsab[w, s]
        dg_weiss_inv_wsab[w, s] = dX_inv
        dg_weiss_wsab[w, s] = -g_weiss_wsab[w, s] @ dX_inv @ g_weiss_wsab[w, s]

    return dg_weiss_wsab, dg_weiss_inv_wsab


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


def lr_extract_h0_and_delta(g_weiss_wsab, dg_weiss_wsab, iaft, high_freq_multiplier=10):
    """
    Linear response of extract_h0_and_delta (make_hermitian=False) for a perturbation of any momentum Q. The ground
    state builds
        g_weiss^{-1}(iw) = iw - h0 - Delta(iw),
    so the response is
        d(g_weiss^{-1})(iw) = -dh0 - dDelta(iw),   d(g_weiss^{-1}) = -g_weiss^{-1} dg_weiss g_weiss^{-1}.
      dh0 = -estimate_zero_moment of d(g_weiss^{-1})(+-iw) at the same interpolated frequencies +-iw as the
            ground state (g_weiss and dg_weiss are interpolated, then inverted there),
      dDelta(iw) = -dh0 - d(g_weiss^{-1})(iw) on the mesh.
    No Hermitian projection of dh0 and no pairing of dDelta(iw) with dDelta(-iw): at Q != 0 the response is complex
    and dDelta(-iw) is not dDelta(iw)^dagger. estimate_zero_moment is real-linear, so this is the exact linearization
    of the ground-state map without projections.

    Returns (dh0_sab, ddelta_wsab).
    """
    nspin = g_weiss_wsab.shape[1]

    iwn_interp = iaft.wn_mesh('f', phys_notation=True)[-3:] * high_freq_multiplier
    dg_weiss_inv_interp = []
    for iwn_idx in (iwn_interp, -iwn_interp - 1):   # +iw and -iw
        g_weiss_interp = iaft.w_interpolate(g_weiss_wsab, iwn_idx, 'f', phys_notation=True)
        dg_weiss_interp = iaft.w_interpolate(dg_weiss_wsab, iwn_idx, 'f', phys_notation=True)
        dg_weiss_inv = np.zeros(g_weiss_interp.shape, dtype=complex)
        for n in range(g_weiss_interp.shape[0]):
            for s in range(nspin):
                g_inv = np.linalg.inv(g_weiss_interp[n, s])
                dg_weiss_inv[n, s] = -g_inv @ dg_weiss_interp[n, s] @ g_inv
        dg_weiss_inv_interp.append(dg_weiss_inv)
    iwn_interp = (2*iwn_interp.astype(float) + 1) * np.pi / iaft.beta
    dh0_sab = -estimate_zero_moment(dg_weiss_inv_interp[0], dg_weiss_inv_interp[1], iwn_interp)

    ddelta_wsab = np.zeros(g_weiss_wsab.shape, dtype=complex)
    for n in range(g_weiss_wsab.shape[0]):
        for s in range(nspin):
            g_inv = np.linalg.inv(g_weiss_wsab[n, s])
            dg_weiss_inv = -g_inv @ dg_weiss_wsab[n, s] @ g_inv
            ddelta_wsab[n, s] = -dh0_sab[s] - dg_weiss_inv

    return dh0_sab, ddelta_wsab


def lr_compute_weiss_fields_w(*, iaft, local_gf, d_local_gf, impurity_selfenergies, d_impurity_selfenergies):
    """
    Linear response of compute_weiss_fields_w: (dg_weiss_w, du_weiss_w) for responses of the local lattice
    quantities, d_local_gf = {"Gloc_t", "Wloc_t"} (dVloc = 0), and of the impurity self-energies,
    d_impurity_selfenergies = {"Vhf_imp", "Sigma_imp_w", "Pi_imp_w"}, in the layouts of the ground-state dicts.
    Wloc_t, Vloc and d Wloc_t are in the product basis, as the driver's solver inputs: apply
    chemistry_to_product_basis to the chemistry-order output of downfold_coulomb / lr_downfold_coulomb first.

    Defined only for causal_projection = None: fit_local_results_boson and fit_u_weiss are nonlinear bath fits,
    and for set_sigma_infty_to_dc = False: Vhf_imp is the impurity's own Sigma_infty, as in the ground state.
    Covers screen_type = "gw_edmft" only; the Pi_imp response of screen_type = "rpa" is
    coqui.embed.lr_eval_pi_rpa_cells with a single cell.
    """
    missing = {"Gloc_t", "Wloc_t", "Vloc"} - local_gf.keys()
    if missing:
        raise ValueError(f"Missing keys in local_gf: {missing}")
    missing = {"Gloc_t", "Wloc_t"} - d_local_gf.keys()
    if missing:
        raise ValueError(f"Missing keys in d_local_gf: {missing}")
    for name, d in (("impurity_selfenergies", impurity_selfenergies),
                    ("d_impurity_selfenergies", d_impurity_selfenergies)):
        missing = {"Vhf_imp", "Sigma_imp_w", "Pi_imp_w"} - d.keys()
        if missing:
            raise ValueError(f"Missing keys in {name}: {missing}")

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
    dg_weiss_w, _ = lr_weiss_fermion_w(
        iaft.tau_to_w(local_gf["Gloc_t"], stats='f'),
        iaft.tau_to_w(d_local_gf["Gloc_t"], stats='f'),
        impurity_selfenergies["Vhf_imp"], d_impurity_selfenergies["Vhf_imp"],
        impurity_selfenergies["Sigma_imp_w"], d_impurity_selfenergies["Sigma_imp_w"]
    )
    iaft.check_leakage(dg_weiss_w, 'f', 'dg_weiss', w_input=True)
    app_log(2, "")

    return dg_weiss_w, du_weiss_w
