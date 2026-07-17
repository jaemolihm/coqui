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

import json

from coqui._lib.eri_module import ThcCoulomb
from coqui._lib.eri_module import run_isdf as isdf_cxx
from coqui._lib.eri_module import make_thc_pivots as make_thc_pivots_cxx


def make_thc_coulomb(mf, params):
    """
    Compute or read THC-decomposed Coulomb integrals and return a ``ThcCoulomb`` handler.

    The ``ThcCoulomb`` object returned contains a THC representation of the 
    two-electron Coulomb integrals that are either computed during the function call 
    or read from a pre-existing HDF5. (``save`` key in ``params``). 

    The resulting object can be passed to electronic structure methods such as ``run_hf`` 
    and ``downfold_coulomb``. 

    Parameters
    ----------
    mf : Mf
        Mean-field object for the target system, obtained from ``make_mf``.
    params : dict
        THC construction options. Supported keys:

        - ``nIpts`` *(int, optional, default ``0``)* — number of THC interpolation
          points. Primary stopping criterion when set to a positive value. If ``0``,
          ``thresh`` controls termination instead.
        - ``thresh`` *(float, optional, default ``1e-5``)* — convergence threshold 
          for the THC auxiliary basis construction. 
          Defaults to ``1e-5`` when ``nIpts=0`` or to ``1e-13`` when ``nIpts > 0``. 
          When both ``nIpts`` and ``thresh`` are given explicitly, the algorithm stops 
          as soon as either criterion is satisfied.
        - ``ecut`` *(float, optional, default ``1.4 * mf.ecutwfc()``)* - kinetic-energy
          cutoff for evaluating Coulomb matrix elements. For backends without a
          wavefunction grid (e.g. PySCF, model), the default falls back to
          ``0.4 * mf.ecutrho()``.
        - ``storage`` *(str, optional, default ``"incore"``)* — how integrals are
          stored after construction. ``"incore"`` keeps them in memory;
          ``"outcore"`` reads them from the HDF5 file on demand.
        - ``save`` *(str, optional, default ``""``)* — path to an HDF5 file for
          saving (or loading) the THC integrals. An empty string disables file I/O. 
          If the file exists, the THC integrals are automatically loaded. 
        - ``cd_dir`` *(str, optional, default ``""``)* — directory containing
          pre-computed Cholesky-decomposed Coulomb integrals. When provided, a
          least-squares THC fit is performed instead of ISDF.
        - ``pivot_file`` *(str, optional, default ``""``)* — HDF5 file holding
          precomputed interpolating points (``interpolating_points`` dataset,
          e.g. the ``save`` file of a previous THC run). When provided, the
          ISDF pivot search is skipped and those pivot points are reused; only
          the interpolating vectors and the Coulomb matrix are recomputed for
          the current mean field. The pivot indices refer to the FFT grid, so
          both calculations must use the same ``ecut`` (checked against the
          file's ``fft_grid``). ``nIpts`` is ignored when set.
        - ``chol_block_size`` *(int, optional, default ``8``)* — block size for
          the internal Cholesky step.
        - ``band_weights`` *(bool, optional, default ``True``)* — for an
          augmented mean field (``augment_mf`` / ``augment_mf_dpsi``), weight
          each band by its stored augmentation singular value (1 for the
          original bands) in the pivot search and the interpolating-vector
          fit. The collocation matrices and the resulting ERIs remain
          unweighted. Set to ``False`` to treat all bands equally.
        - ``init`` *(bool, optional, default ``True``)* — if ``True``, runs the
          full THC computation immediately at construction. Set to ``False`` to
          defer until ``.init()`` is called explicitly.

    Returns
    -------
    ThcCoulomb
        A THC Coulomb interaction object that can be passed to MBPT functions
        such as ``run_hf`` and ``run_gw``.

    Examples
    --------
    ::

        from coqui.interaction import make_thc_coulomb

        thc = make_thc_coulomb(mf, {"save": "svo_isdf.h5", "storage": "incore"})
    """
    return ThcCoulomb(mf, json.dumps(params))


def make_thc_pivots(mf, params):
    """
    Compute only the ISDF interpolating (pivot) points and save them to an
    HDF5 file, without evaluating the interpolating vectors or the Coulomb
    matrix. The resulting file can be passed as ``pivot_file`` to
    ``make_thc_coulomb`` to build THC integrals on those fixed pivot points
    (possibly for a different mean field on the same FFT grid, e.g. a
    basis-augmented one).

    Parameters
    ----------
    mf : Mf
        Mean-field object for the target system, obtained from ``make_mf``.
    params : dict
        Pivot computation options. Supported keys:

        - ``thresh`` *(float)* / ``nIpts`` *(int)* — stopping criteria of the
          pivoted Cholesky selection, as in ``make_thc_coulomb``. At least one
          must be set.
        - ``ecut`` *(float, optional, default ``1.4 * mf.ecutwfc()``)* —
          plane-wave cutoff defining the FFT grid the pivot indices refer to.
          Use the same value in the subsequent ``make_thc_coulomb`` call.
        - ``save`` *(str, optional, default ``"thc_pivots.h5"``)* — output HDF5
          file (``Np``, ``interpolating_points``, ``ecut``, ``fft_grid``).
        - ``X_orbital_range`` / ``Y_orbital_range`` *(optional)* — orbital
          ranges of the pair densities used in the pivot search.
        - ``band_weights`` *(bool, optional, default ``True``)* — as in
          ``make_thc_coulomb``: weight the pivot search by the augmentation
          singular values of an augmented mean field.
        - ``chol_block_size``, ``matrix_block_size``, ... — as in
          ``make_thc_coulomb``.

    Examples
    --------
    ::

        from coqui.interaction import make_thc_pivots, make_thc_coulomb

        make_thc_pivots(mf_orig, {"thresh": 1e-4, "save": "pivots.h5"})
        thc = make_thc_coulomb(mf_aug, {"thresh": 1e-4, "pivot_file": "pivots.h5"})
    """
    make_thc_pivots_cxx(mf, json.dumps(params))


def run_isdf(mf, params):
    """
    Run the Interpolative Separable Density Fitting (ISDF) decomposition for 
    pair densities and save the result to an HDF5 file. 

    Parameters
    ----------
    mf : Mf
        Mean-field object for the target system, obtained from ``make_mf``.
    params : dict
        ISDF computation options. Supported keys:

        - ``thresh`` *(float, optional, default ``1e-10``)* — threshold for
          constructing the ISDF auxiliary basis. Either ``thresh > 0`` or
          ``nIpts > 0`` must be set.
        - ``nIpts`` *(int, optional, default ``0``)* — fixed number of
          interpolating points. When positive, overrides threshold-based selection;
          ``thresh`` then defaults to ``1e-13``.
        - ``ecut`` *(float, optional, default matches ``mf`` value)*
          — plane-wave kinetic-energy cutoff for evaluating Coulomb matrix elements.
        - ``save`` *(str, optional, default ``"isdf.h5"``)* — HDF5 file where the
          ISDF result (interpolating points and pair-density values) is written.
        - ``write_zeta_on_fft_mesh`` *(bool, optional, default ``False``)* — if
          ``True``, saves the ISDF basis functions on the full FFT mesh.
        - ``check_accuracy`` *(bool, optional, default ``False``)* — if ``True``,
          perform accuracy diagnostics for the decomposition.
        - ``chol_block_size`` *(int, optional, default ``8``)* — block size for
          the pivoted-Cholesky algorithm. Larger values are faster but use more
          memory; typical range: 1–12.
        - ``matrix_block_size`` *(int, optional, default ``1024``)* — block size
          for distributed array operations.
        - ``memory_frac`` *(float, optional, default ``0.75``)* — fraction of
          available node memory to budget for intermediate arrays.

    Examples
    --------
    ::

        from coqui.interaction import run_isdf

        run_isdf(mf, {"thresh": 1e-4, "save": "svo_isdf.h5"})
    """
    isdf_cxx(mf, json.dumps(params))

