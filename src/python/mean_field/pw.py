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


def compute_pw_matrix_elements(mf, q_cryst, G_mill):
    """Plane-wave matrix elements in the mean-field band basis.

    Thin wrapper over the C++ ``Mf.compute_pw_matrix_elements``. Computes

        M[G, s, k, i, j] = <phi_{k+q,i}| e^{i(q+G).r} |phi_{k,j}>
                         = (1/nnr) sum_r e^{i(G+G0).r}
                             conj(u_{s,k',i}(r)) u_{s,k,j}(r),

    where k' is the stored k-point with k + q = k' + G0 (G0 a reciprocal lattice
    vector) and u are the cell-periodic orbitals on the mean-field FFT grid. The
    conjugate sits on the k+q band index ``i``; ``j`` is the k band.

    Normalization: exactly the formula above and nothing else. The orbitals are
    normalized to ``(1/nnr) sum_r |u|^2 = 1`` in the unit cell, so M is the
    dimensionless Bloch overlap; in particular ``M[G=0, s, k]`` is the identity
    matrix at q = 0. No Coulomb weight ``sqrt(v(q+G))``, no ``1/sqrt(Omega*nkpts)``
    and no ``1/nkpts`` are applied -- every volume, cell-count and Coulomb factor
    is the caller's to supply.

    The k-loop is split across ranks and gathered, so M is returned only on the
    MPI root (rank 0); every other rank gets an empty array. This call is still
    collective (all ranks must invoke it). The q+G table comes back on every rank.

    Parameters
    ----------
    mf : Mf
        Mean-field system providing the orbitals. Requires npol = 1 and a
        full-BZ k-grid (``nkpts == nkpts_ibz``): a general q breaks the crystal
        symmetry, so a symmetry-reduced mean field would describe a different
        perturbation.
    q_cryst : array_like, shape (3,)
        Wavevector in crystal (fractional reciprocal-lattice) coordinates.
    G_mill : array_like of int, shape (nG, 3) or (3,)
        Miller indices (integer crystal coordinates) of the plane waves, so that
        ``G = sum_d G_mill[p, d] * recv[d]``. A single triple is accepted and
        promoted. Each component must satisfy ``|G_mill[p, d]| <= mesh[d] // 2``,
        the Nyquist limit of the orbital FFT grid; beyond it ``e^{iG.r}`` aliases
        to a different G on the mesh.

    Returns
    -------
    (M, qpG_cart) : tuple of ndarray
        ``M`` complex: (nG, nspin, nkpts, nbnd, nbnd) on the root, empty elsewhere.
        ``qpG_cart`` (nG, 3) float: q + G in Cartesian coordinates, ready for
        ``v(q+G) = 4*pi/|q+G|**2``. CoQuí's Coulomb kernel carries no ``1/Omega``,
        so a caller assembling ``v * chi`` also needs ``mf.volume()``.

    Raises
    ------
    ValueError
        On an empty or misshaped ``G_mill``, or a component past the FFT Nyquist
        limit.

    Notes
    -----
    Memory grows fast in ``nG``: the C++ side holds three ``nbnd x nnr`` complex
    real-space buffers plus an ``nG x nnr`` plane-wave table per rank, and the root
    holds the whole ``(nG, nspin, nkpts, nbnd, nbnd)`` result. Sweeping many G is
    cheaper as several calls of a few G columns each.

    The checks above are raised here, before the collective C++ call is entered, so
    a q/G sweep driver gets a catchable exception. The C++ side re-checks the same
    conditions as a backstop, but there it is a ``utils::check`` and therefore an
    ``MPI_Abort``: it cannot deadlock (every rank sees the same replicated inputs
    and the checks precede the only collectives) but it does kill the interpreter,
    which is why the catchable copy lives here.
    """
    q_cryst = np.ascontiguousarray(q_cryst, dtype=np.float64)
    G_mill = np.ascontiguousarray(G_mill, dtype=np.int64)
    if G_mill.ndim == 1:
        G_mill = G_mill.reshape(1, -1)
    if G_mill.ndim != 2 or G_mill.shape[1] != 3:
        raise ValueError(
            "compute_pw_matrix_elements: G_mill must be (nG, 3) Miller triples, "
            f"got shape {G_mill.shape}")
    if G_mill.shape[0] == 0:
        raise ValueError("compute_pw_matrix_elements: G_mill is empty")

    mesh = np.asarray(mf.fft_grid(), dtype=np.int64)
    over = np.abs(G_mill) > (mesh // 2)[None, :]
    if over.any():
        p = int(np.flatnonzero(over.any(axis=1))[0])
        raise ValueError(
            f"compute_pw_matrix_elements: G_mill[{p}] = {G_mill[p].tolist()} "
            f"exceeds the FFT grid Nyquist limit {(mesh // 2).tolist()}")

    return mf.compute_pw_matrix_elements(q_cryst, G_mill)
