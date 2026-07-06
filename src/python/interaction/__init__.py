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


def compute_delta_X(mf, Deltapsi_prefix, r_P, delta_r_P, q_vec_cryst, fft_grid):
    """Linear response of the THC collocation matrix X(k, P, m) = ψ_{mk}(r_P).

    ΔX(k, P, m) = δψ_{mk}(r_P) + ∇ψ_{mk}(r_P) · Δr_P

    Parameters
    ----------
    mf : Mf
        Mean-field handler.
    Deltapsi_prefix : str
        Prefix of the QE δψ HDF5 files ({prefix}_ik{ik+1}.hdf5).
    r_P : np.ndarray of long, shape (nP,)
        THC interpolation-point indices on the FFT grid.
    delta_r_P : np.ndarray of float, shape (nP, 3)
        Displacements of the interpolation points (crystal coordinates).
    q_vec_cryst : np.ndarray of float, shape (3,)
        Perturbation wavevector in crystal coordinates.
    fft_grid : np.ndarray of long, shape (3,)
        FFT grid dimensions.

    Returns
    -------
    np.ndarray of shape (nspin, nk_ibz, nP, nbnd) on rank 0; ``None`` on non-root ranks.
    """
    arr = _compute_delta_X_cpp(mf, Deltapsi_prefix, r_P, delta_r_P, q_vec_cryst, fft_grid)
    return arr if arr.size else None


def compute_delta_X_adj(mf, Deltapsi_adj_prefix, r_P, delta_r_P, q_vec_cryst, fft_grid):
    """Adjoint (-q) companion of compute_delta_X.

    At index ik the returned array stores δ^{-q} X evaluated at k_{ik}+q (the
    DeltaX_right consumer convention). Parameters as in ``compute_delta_X``,
    with ``Deltapsi_adj_prefix`` naming the adjoint Sternheimer δψ files.

    Returns
    -------
    np.ndarray of shape (nspin, nk_ibz, nP, nbnd) on rank 0; ``None`` on non-root ranks.
    """
    arr = _compute_delta_X_adj_cpp(mf, Deltapsi_adj_prefix, r_P, delta_r_P, q_vec_cryst, fft_grid)
    return arr if arr.size else None


__all__ = ["make_thc_coulomb", "make_chol_coulomb", "run_isdf", "compute_delta_X", "compute_delta_X_adj"]
