"""
==========================================================================
CoQuí: Correlated Quantum ínterface

Copyright (c) 2022-2025 Simons Foundation & The CoQuí developer team

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


def compute_bare_eph_vertex(mf, q_cryst):
    """Bare electron-phonon vertex g = g_loc + g_nl in the mean-field band basis.

    Thin wrapper over the C++ ``Mf.compute_bare_eph_vertex``. Assembles the full bare vertex
    at phonon wavevector ``q_cryst`` entirely from CoQuI's own quantities (no
    external elph or UPF file): the bare phonon perturbation are computed from the
    local ionic potential and nonlocal pseudopotential terms from pseudopotential
    data stored in the mean-field h5. Nonlinear core correction is not included
    since we compute the bare vertex.

        g[s, mode, k, m, n] = <phi_{m,k+q}| dV_mode |phi_{n,k}>,
        mode = 3*kappa + d   (atom kappa, Cartesian direction d = x,y,z),
        nmodes = 3 * natom.

    The vertex is large, so it is returned only on the MPI root (rank 0); every
    other rank gets an empty array. This call is still collective (all ranks must
    invoke it).

    Parameters
    ----------
    mf : Mf
        Mean-field system providing the orbitals and pseudopotential. Requires the
        h5 backend (which must carry the "vloc_radial" group written by pw2coqui),
        npol=1, and a full-BZ k-grid (``nkpts == nkpts_ibz``).
    q_cryst : array_like, shape (3,)
        Phonon wavevector in crystal (fractional reciprocal-lattice) coordinates.

    Returns
    -------
    ndarray, complex, Hartree. On the root: shape
    (nspin, nmodes, nkpts, nbnd, nbnd). On all other ranks: an empty array.
    """
    q_cryst = np.ascontiguousarray(q_cryst, dtype=np.float64)
    return mf.compute_bare_eph_vertex(q_cryst)


def compute_bare_eph_vertex_d2(mf):
    """Bare second-order electron-phonon vertex g2 = <phi|d^2 V_bare|phi> at q=0.

    Thin wrapper over the C++ ``Mf.compute_bare_eph_vertex_d2``. This is the
    second derivative of the bare Hamiltonian matrix element w.r.t. two atomic
    displacements, evaluated at q=0 (the "g2_bare"/"d2H0_bare" quantity QE stores
    at q=Gamma), assembled entirely from CoQuI's own quantities (no external elph
    or UPF file): the local part from ``d^2 V_loc`` rebuilt from the per-species
    radial vloc in the mean-field h5, the nonlocal part from the pseudopotential
    projector overlaps. Nonlinear core correction is not included.

        g2[s, kappa, i, j, k, m, n]
            = <phi_{m,k}| d^2 V_bare/dtau_i dtau_j |phi_{n,k}>,
        with mode1 = 3*kappa + i, mode2 = 3*kappa + j (i, j Cartesian x,y,z).

    V'' is diagonal in the atom index, so only same-atom blocks are nonzero.
    The result exploits this: it is stored compactly as
    (nspin, nat, 3, 3, nkpts, nbnd, nbnd) — dims (atom, cart_i, cart_j) — rather
    than the dense (nspin, nmodes, nmodes, ...), which is nat times larger and
    mostly zero. It is returned only on the MPI root (rank 0); every other rank
    gets an empty array. This call is still collective (all ranks must invoke it).

    Parameters
    ----------
    mf : Mf
        Mean-field system providing the orbitals and pseudopotential. Requires the
        h5 backend (which must carry the "vloc_radial" group written by pw2coqui),
        npol=1, nspin=1, and a full-BZ k-grid (``nkpts == nkpts_ibz``).

    Returns
    -------
    ndarray, complex, Hartree. On the root: shape
    (nspin, nat, 3, 3, nkpts, nbnd, nbnd). On all other ranks: an empty array.
    """
    return mf.compute_bare_eph_vertex_d2()
