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

from .thc import make_thc_coulomb, run_isdf
from .cholesky import make_chol_coulomb

# import PySCF converter module only if PySCF is available
try:
    import pyscf
except ImportError:
    pass
else:
    from . import pyscf_interface

from coqui._lib.eri_module import (
    compute_delta_X as _compute_delta_X_cpp,
    compute_delta_X_adj as _compute_delta_X_adj_cpp,
)


def compute_delta_X(*args, **kwargs):
    """Linear response of the THC collocation matrix X(k, P, m) = ψ_{mk}(r_P).

    ΔX(k, P, m) = δψ_{mk}(r_P) + ∇ψ_{mk}(r_P) · Δr_P

    See ``coqui._lib.eri_module.compute_delta_X`` for parameter details.

    Returns
    -------
    np.ndarray of shape (nspin, nk_ibz, nP, nbnd) on rank 0; ``None`` on non-root ranks.
    """
    arr = _compute_delta_X_cpp(*args, **kwargs)
    return arr if arr.size else None


def compute_delta_X_adj(*args, **kwargs):
    """Adjoint (-q) companion of compute_delta_X.

    See ``coqui._lib.eri_module.compute_delta_X_adj`` for parameter details.

    Returns
    -------
    np.ndarray of shape (nspin, nk_ibz, nP, nbnd) on rank 0; ``None`` on non-root ranks.
    """
    arr = _compute_delta_X_adj_cpp(*args, **kwargs)
    return arr if arr.size else None


__all__ = ["make_thc_coulomb", "make_chol_coulomb", "run_isdf", "compute_delta_X", "compute_delta_X_adj"]
