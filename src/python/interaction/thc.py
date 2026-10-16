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
        - ``source`` *(str, optional, default ``"auto"``)* — where the THC
          integrals come from. ``"auto"`` reads from ``save`` if that file
          exists and otherwise builds them (the historical behavior).
          ``"read"`` always reads from ``save`` and raises if the file is
          missing; ``nIpts``/``thresh`` are not needed in this mode (``Np`` is
          taken from the file). ``"compute"`` always rebuilds, ignoring any
          pre-existing ``save`` file.
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
        - ``band_weights`` *(bool, optional, default ``True``)* — master switch
          for per-band fit weighting; see ``nbnd_protected`` below for what is
          weighted. ``False`` treats all bands equally and turns off **both**
          sources of weight, leaving ``nbnd_protected`` only as the selector of
          the unprotected band range for ``exclude_unprotected_pairs``.
        - ``nbnd_protected`` *(int, optional, default ``-1``)* — number of
          protected bands ``N_P``, and the reference description of the
          protected-band scheme. Two weightings enter the pivot search and the
          interpolating-vector fit, both gated on ``band_weights=True``: an
          augmented mean field (``augment_mf`` / ``augment_mf_dpsi``) weights each
          band by its stored augmentation singular value (1 for the original
          bands), and a positive ``N_P`` keeps bands ``b < N_P`` at weight 1 while
          suppressing the tail ``b >= N_P`` by
          ``(E(s,k,N_P-1) - mu)/(E(s,k,b) - mu)``, with ``mu`` the mean field's
          Fermi energy and ``E(s,k,N_P-1)`` the last protected band. The two
          combine rather than exclude each other: on an augmented mean field the
          energy weights stop at ``nbnd_orig`` and the augmentation block keeps
          its singular values. The collocation matrices and the resulting ERIs
          remain unweighted.
          Preconditions: a mean field carrying a Fermi energy
          (``System/fermi_energy``) — a checkpoint written without one is rejected
          rather than treated as ``mu = 0`` — and an ``N_P`` leaving at least one
          unoccupied band protected, since the reference band is ``N_P-1`` and
          ``E(N_P-1) - mu`` must be positive.
          ``nbnd_protected == nbnd`` is accepted as an explicit no-op: the
          unprotected tail is empty, so no weights are built and
          ``exclude_unprotected_pairs`` is forced off, giving a run identical to
          plain THC while the preconditions are still checked (a dry run).
          Passing ``exclude_unprotected_pairs=True`` explicitly in that case is an
          error rather than a silent no-op.
        - ``nbnd_orig`` *(int, optional, default ``-1``)* — number of original
          (non-augmentation) bands of an augmented mean field, i.e. where the
          augmentation block starts in the ``[originals | augmentation]`` basis.
          Required together with ``nbnd_protected`` on an augmented mean field and
          rejected otherwise: the block boundary cannot be inferred from the mean
          field. It is the split point of the two weightings above — energy
          weights cover ``[nbnd_protected, nbnd_orig)``, augmentation singular
          values ``[nbnd_orig, nbnd)``.
        - ``exclude_unprotected_pairs`` *(bool, optional, default ``True`` when
          ``nbnd_protected > 0``)* — research diagnostic. When ``True``, drop
          the unprotected-unprotected pair densities
          (``M_keep = M_full - M_aug``) from BOTH the pivot-point-selection
          metric AND the interpolating-vector / Coulomb-matrix (zeta) fit; the
          returned collocation and the pair densities in the ERIs are unchanged,
          so it only changes which pairs the fit prioritizes. The unprotected set
          is the band tail ``[nbnd_protected, nbnd)`` (so ``nbnd_protected`` is
          required, and on an augmented mean field the augmentation states count
          as unprotected). The zeta-fit part is implemented on the host/CPU
          backend only (a GPU run with ``True`` errors).
        - ``init`` *(bool, optional, default ``True``)* — if ``True``, runs the
          full THC computation immediately at construction. Set to ``False`` to
          defer until ``.init()`` is called explicitly.
        - ``Vxc_file`` *(str, optional, default ``""``)* — path to an HDF5 file
          holding semilocal xc-kernel coefficient fields (QE ``elph.x`` run with
          ``write_xc_kernel = .true.``). When non-empty, the xc-kernel matrix
          ``Vxc(q)_PQ`` is built alongside the Coulomb matrix, stored in the
          ``save`` file as the ``Vxc`` dataset, and reachable through
          ``thc.Vxc(iq)``. It is *not* added to the Coulomb matrix: ``Vxc`` is
          only defined in the direct (Hartree) channel, where it contracts with
          the diagonal density response. Requires the ISDF algorithm (no
          ``cd_dir``), a ``nspin = 1`` / ``npol = 1`` mean field to match the
          spin-unresolved kernel fields, and a kernel FFT grid that contains the
          THC density grid. Empty (default) leaves the Coulomb path untouched.
          The path is recorded in the ``save`` file as ``Vxc_source``, and reusing
          a file whose ``Vxc`` came from a different dump — or which has no
          ``Vxc`` at all — is rejected rather than silently accepted under the
          default ``source="auto"``; delete the file or pass ``source="compute"``.
        - ``Vxc_block_size`` *(int, optional, default ``64``)* — number of pivots
          per real-space block while building ``Vxc``. Memory scales as
          ``8 * Vxc_block_size * nnr`` complex numbers — several GB per rank on a
          dense mesh — and the FFT count as ``nIpts**2 / Vxc_block_size``, so
          raise it only if memory allows. The per-rank footprint is reported at
          verbosity 2.

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

    A model Hamiltonian (``mf_type="model"``) has no orbitals, so its THC integrals cannot be
    built: write the file with ``write_thc_h5`` and read it (``nIpts``/``thresh`` are not needed)::

        thc = make_thc_coulomb(mf, {"save": "hubbard_thc.h5", "storage": "incore"})
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
        - ``nbnd_protected`` / ``nbnd_orig`` / ``exclude_unprotected_pairs`` — as
          in ``make_thc_coulomb`` (see ``nbnd_protected`` there for the full
          description). Only the pivot-selection metric is built here, so they
          affect the point selection alone; the interpolating-vector (zeta) part
          of ``exclude_unprotected_pairs`` applies in ``make_thc_coulomb``.
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


def write_thc_h5(filename, X, Z, mpi=None):
    """
    Write a THC interaction file read by ``make_thc_coulomb(mf, {"save": filename, "storage": "incore"})``.

    This is pure I/O for model Hamiltonians: it checks shapes and writes the arrays given.

    A one-orbital density-density interaction ``V(q)`` is exactly rank-1 THC:
    ``Np = 1``, ``X = 1``, ``Z[q] = V(q)``
    (Hubbard: ``Z[q] = U``; extended Hubbard: ``U + 2V(cos 2pi q1 + cos 2pi q2)``, q crystal).

    File layout (file root):

    ============================== ========= ===================== =============================
    entry                          dtype     shape                 content
    ============================== ========= ===================== =============================
    Np                             int32     scalar                ``Np``
    X_orbital_range                int32     [2]                   ``[0, nb]``
    Y_orbital_range                int32     [2]                   ``[0, nb]``
    collocation_matrix             complex   [ns*npol, nk, Np, nb] ``X``
    coulomb_matrix                 complex   [nq, Np, Np]          ``Z``
    interpolating_points           int64     [Np]                  dummy zeros (see below)
    interpolating_vectors_G0       complex   [nq, Np]              dummy zeros (see below)
    dual_interpolating_vectors_G0  complex   [nq, Np]              dummy zeros (see below)
    ============================== ========= ===================== =============================

    ``Z[q]`` is the per-cell ``V(q)``, **not** scaled by the number of k-points or by the cell
    volume. q runs over the full k-list of the model file (no q symmetry, ``nq = nk``), in the same
    order, so ``Z[iq] = V(q = kpts_crys[iq])``; the first entry is q = Gamma.

    The last three datasets are a workaround: they belong to ISDF (interpolating-point grid
    indices and the q->0 heads) and mean nothing for a model, but ``thc_reader_t::read()`` reads
    ``interpolating_points``, ``interpolating_vectors_G0`` and ``dual_interpolating_vectors_G0``
    unconditionally and checks the length of ``interpolating_points`` against ``Np``. Their values
    are never used with ``div_treatment="ignore_g0"``, which a model run forces.

    Parameters
    ----------
    filename : str
        Output HDF5 file.
    X : array, complex, shape ``(ns*npol, nk, Np, nb)``
        Collocation matrix.
    Z : array, complex, shape ``(nq, Np, Np)``
        Coulomb matrix per unit cell; Hermitian in ``(P, Q)`` for every q.
    mpi : MpiHandler, optional
        If given, only the root rank writes, followed by a barrier.
    """
    import numpy as np
    import h5py

    X = np.ascontiguousarray(X, dtype=np.complex128)
    Z = np.ascontiguousarray(Z, dtype=np.complex128)
    if X.ndim != 4:
        raise ValueError(f"write_thc_h5: X must be [ns*npol, nk, Np, nb], got {X.shape}")
    nb, Np = X.shape[3], X.shape[2]
    if Z.ndim != 3 or Z.shape[1:] != (Np, Np):
        raise ValueError(f"write_thc_h5: Z must be [nq, {Np}, {Np}], got {Z.shape}")
    if not np.allclose(Z, np.conj(Z.transpose(0, 2, 1))):
        raise ValueError("write_thc_h5: Z[q] must be Hermitian")
    nq = Z.shape[0]

    def write_complex(f, name, a):
        ds = f.create_dataset(name, data=a.view(np.float64).reshape(a.shape + (2,)))
        ds.attrs["__complex__"] = np.int8(1)

    if mpi is None or mpi.root():
        with h5py.File(filename, "w") as f:
            f.create_dataset("Np", data=np.int32(Np))
            f.create_dataset("X_orbital_range", data=np.array([0, nb], dtype=np.int32))
            f.create_dataset("Y_orbital_range", data=np.array([0, nb], dtype=np.int32))
            write_complex(f, "collocation_matrix", X)
            write_complex(f, "coulomb_matrix", Z)
            # Dummies, required by thc_reader_t::read(); unused for a model.
            f.create_dataset("interpolating_points", data=np.zeros(Np, dtype=np.int64))
            write_complex(f, "interpolating_vectors_G0", np.zeros((nq, Np), dtype=np.complex128))
            write_complex(f, "dual_interpolating_vectors_G0", np.zeros((nq, Np), dtype=np.complex128))
    if mpi is not None:
        mpi.barrier()


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

