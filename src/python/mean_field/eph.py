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

    The result is replicated on every MPI rank (the underlying C++ steps are
    collective).

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
    ndarray, complex, shape (nspin, nmodes, nkpts, nbnd, nbnd), Hartree.
    """
    q_cryst = np.ascontiguousarray(q_cryst, dtype=np.float64)
    return mf.compute_bare_eph_vertex(q_cryst)
