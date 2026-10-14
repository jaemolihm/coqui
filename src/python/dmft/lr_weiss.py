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

from .weiss import eval_gw_dc_t, eval_pi_rpa_pair


def lr_sigma_dc_t(G_tsab, dG_tsab, W_tabcd, dW_tabcd):
    """
    Linear response of eval_gw_dc_t (bilinear in G and W): dSigma_dc(tau) = Sigma[dG, W] + Sigma[G, dW].
    W, dW are in the product basis on the half tau grid, as for eval_gw_dc_t: apply chemistry_to_product_basis to
    the chemistry-order output of downfold_coulomb / lr_downfold_coulomb first, as the GW+EDMFT driver does.
    The half tau grid (tau <= beta/2) assumes the tau-reflection evenness of the response, which is exact when
    the perturbation is time-reversal-even (a physical electron-phonon dV in a time-reversal-symmetric crystal,
    at any Q and any number of orbitals); for a generic perturbation, e.g. a random complex Hermitian dH0, it
    fails at O(1) even at Q = 0.
    """
    return eval_gw_dc_t(dG_tsab, W_tabcd) + eval_gw_dc_t(G_tsab, dW_tabcd)


def lr_pi_dc_t(G_tsab, dG_tsab, *, density_only=True):
    """
    Linear response of eval_pi_rpa: dPi_dc(tau) = P(dG, G) + P(G, dG), P = eval_pi_rpa_pair. Both factors carry
    the same momentum transfer (the product is site-diagonal), so no transpose or -Q partner appears.
    G_tsab, dG_tsab and the result are on the half tau grid (tau <= beta/2), which assumes the tau-reflection
    evenness of the response. That evenness is exact when the perturbation is time-reversal-even (a physical
    electron-phonon dV in a time-reversal-symmetric crystal, at any Q and any number of orbitals); for a generic
    perturbation, e.g. a random complex Hermitian dH0, it fails at O(1) even at Q = 0.
    """
    return (eval_pi_rpa_pair(dG_tsab, G_tsab, density_only=density_only)
            + eval_pi_rpa_pair(G_tsab, dG_tsab, density_only=density_only))
