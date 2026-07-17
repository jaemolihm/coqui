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

from coqui._lib.mf_module import Mf


def make_mf(mpi, params, mf_type):
    """
    Read a pre-computed mean-field (DFT) solution and construct a CoQuí Mf object.

    CoQuí does not run DFT itself. Instead, it reads the output of an external
    DFT code — selected by ``mf_type`` — and wraps it in a read-only ``Mf`` object
    that subsequent CoQuí steps (interaction construction, MBPT) operate on.

    Supported DFT codes (``mf_type`` values):

    - ``"qe"``    — Quantum ESPRESSO, via the pw2coqui post-processing step
    - ``"pyscf"`` — PySCF
    - ``"bdft"``  — CoQuí's internal mean-field format
    - ``"model"`` — model Hamiltonian (no external DFT code required)

    Parameters
    ----------
    mpi : MpiHandler
        MPI communicator handle obtained from ``coqui.utils.MpiHandler()``.
    mf_type : str
        Selects which DFT code's output to read. One of ``"qe"``, ``"pyscf"``,
        ``"bdft"``, or ``"model"``. 
    params : dict
        File location and read options for the chosen DFT code.
        Keys common to all ``mf_type`` values:

        - ``prefix`` *(str, required)* — file name prefix of the DFT output
        - ``outdir`` *(str, optional, default ``"./"``)*  — directory containing
          the DFT output files.

        Additional keys for each ``mf_type``:

        **"qe"** — reads Quantum ESPRESSO output converted by pw2coqui:

        - ``nbnd`` *(int, optional, default ``-1``)* — number of bands to read;
          ``-1`` reads all bands present in the QE output. 
        - ``ecut`` *(float, optional, default ``0.0``, units: Hartree)* — plane-wave
          kinetic-energy cutoff for the charge-density grid. ``0`` or negative keeps
          the value written by QE; a positive value requests a new FFT grid at that
          cutoff.
        - ``filetype`` *(str, optional, default ``"h5"``)* — input file format.
          ``"h5"`` for pw2coqui HDF5 output (recommended); ``"xml"`` for the legacy
          pw2bgw XML format.

        **"bdft"** — reads a CoQuí BDFT HDF5 file:

        - ``nbnd`` *(int, optional, default ``-1``)* — number of bands to read;
          ``-1`` reads all bands.
        - ``ecut`` *(float, optional, default ``0.0``, units: Hartree)* — same
          semantics as for ``"qe"``.

        **"model"** — reads a model Hamiltonian HDF5 file:

        - ``nbnd`` *(int, optional, default ``-1``)* — number of bands to read;
          ``-1`` reads all bands.

        **"pyscf"** — reads a PySCF HDF5 file; only ``prefix`` and ``outdir`` apply.

    Returns
    -------
    Mf
        A read-only mean-field object that can be passed to interaction and MBPT
        functions such as ``make_thc_coulomb`` and ``run_hf``.

    Examples
    --------
    Quantum ESPRESSO::

        from coqui.utils import MpiHandler
        from coqui.mean_field import make_mf

        mpi = MpiHandler()
        mf = make_mf(mpi,
                     {"prefix": "svo", "outdir": "qe_output/", "nbnd": 40},
                     "qe")

    PySCF::

        mf = make_mf(mpi, {"prefix": "h2o", "outdir": "pyscf_output/"}, "pyscf")
    """
    return Mf(mpi, json.dumps(params), mf_type)


def augment_mf(mf, prefix, outdir="./", augment_type="momentum",
               nbnd_aug=-1, epstol=1e-4, dtau_step=0.1):
    """Create an augmented mean-field system from an existing one.

    Keeps the original ``nbnd`` bands and appends orthonormalized augmentation
    states generated from the first ``nbnd_aug`` bands. For ``augment_type =
    "momentum"`` the raw states are the momentum-operator images p_alpha psi_b
    (alpha = x, y, z; units 1/bohr). The raw states are scaled by ``dtau_step``
    (bohr) — making them dimensionless displacement responses — then
    orthogonalized against the originals and among themselves, truncated with
    the dimensionless singular-value cutoff ``epstol``, and padded to a uniform
    band count across k-points. The singular values s (capped at 1) are stored
    as per-band weights and applied to the THC pivot search and interpolating
    vector fit; the SCF basis and THC collocation matrices stay orthonormal
    and unweighted.

    The result is written to ``{outdir}/{prefix}.h5`` as a bdft mean-field and
    returned as a new :class:`Mf`. Because the augmented basis is not an
    eigenbasis, many-body runs on it must use ``h0_source="compute"``.

    Parameters
    ----------
    mf : Mf
        Base mean-field system to augment.
    prefix : str
        Prefix of the new mean-field file.
    outdir : str
        Directory for the new mean-field file.
    augment_type : str
        Augmentation transform. Currently only ``"momentum"``.
    nbnd_aug : int
        Number of bands to transform (``<= nbnd``; ``-1`` means all). ``0``
        adds no states: the original orbitals are rewritten in the augmented
        bdft format, giving a no-augmentation baseline for the same pipeline.
    epstol : float
        Dimensionless singular-value cutoff: a state is kept when the residual
        amplitude s of its ``dtau_step``-scaled raw state satisfies
        ``s >= epstol``.
    dtau_step : float
        Displacement step in bohr multiplying the raw dpsi/dtau states
        (default 0.1).
    """
    return mf.augment_basis(prefix, outdir, augment_type, int(nbnd_aug),
                            float(epstol), float(dtau_step))


def augment_mf_dpsi(mf, prefix, outdir="./", *, deltapsi_dir, elph_dir,
                    iq_list=(1,), nmodes=None, modes=None, nbnd_aug=-1,
                    nbnd_mf=None, smearing_deltapsi=0.02, epstol=1e-4,
                    dtau_step=0.1):
    """Create a δψ-augmented mean-field system from an existing one.

    The base ``mf`` carries ``N = mf.nbnd()`` nscf/h5 bands; ``nbnd_mf`` (=M)
    selects how many are kept as the originals of the augmented system (``None``
    keeps all N). Appends orthonormalized DFPT response wavefunctions (δψ) read
    from ``{deltapsi_dir}/deltapsi_iq{iq}_mode{m}_ik{k}.hdf5``. For each phonon
    ``iq`` in ``iq_list``, each of ``nmodes`` modes, and each source k-point, the
    first ``nbnd_aug`` (=R) response bands δψ(n,k) are used. Since δψ(n,k) carries
    crystal momentum k+q, it augments the wavefunction at **k+q (mod G)**.

    QE orthogonalizes δψ against all N nscf bands, so the contribution of the
    bands m ∈ [M, N) above the kept originals is added back using the screened
    electron-phonon vertex ``g_scr`` and eigenvalues (and q) read from
    ``{elph_dir}/elph_bare.iq{iq}.h5`` (converted Ry→Ha):

        δψ(n,k) += Σ_{m=M}^{N-1} ψ(m,k+q) g_scr(mode,m,n)
                                 · reg(e(n,k) - e(m,k+q)),

    making the state the response orthogonal to only the M kept bands. ``reg`` is
    a sharp, continuous 1/x cutoff (``1/x`` for ``|x|>σ``, ``x/σ²`` otherwise,
    with σ = ``smearing_deltapsi``); with it the augmentation is **independent of
    N** — a single large dataset
    (big nbnd, nbnd_dpsi) reproduces any smaller (M, R) calculation. The
    R·nmodes·len(iq_list) raw states per k (units 1/bohr) are scaled by
    ``dtau_step`` (bohr), orthogonalized against the M originals (although they
    should already be orthogonal), truncated with the dimensionless
    singular-value cutoff ``epstol``, and padded to a uniform band count. The
    singular values s (capped at 1) are stored as per-band weights and applied
    to the THC pivot search and interpolating vector fit; the SCF basis and THC
    collocation matrices stay orthonormal and unweighted.

    The result is written to ``{outdir}/{prefix}.h5`` as a bdft mean-field and
    returned as a new :class:`Mf`. Because the augmented basis is not an
    eigenbasis, many-body runs on it must use ``h0_source="compute"``. Requires
    ``npol == 1`` and a full-BZ k-grid (``nkpts == nkpts_ibz``).

    Parameters
    ----------
    mf : Mf
        Base mean-field system to augment (carries all N nscf/h5 bands).
    prefix : str
        Prefix of the new mean-field file.
    outdir : str
        Directory for the new mean-field file.
    deltapsi_dir : str
        Directory holding the ``deltapsi_iq{iq}_mode{m}_ik{k}.hdf5`` files.
    elph_dir : str
        Directory holding the ``elph_bare.iq{iq}.h5`` files (source of
        ``g_scr``, eigenvalues, and the phonon q-vector).
    iq_list : sequence of int
        Phonon q indices to include (default ``(1,)``).
    nmodes : int or None
        Number of modes per q. ``None`` uses 3*natom.
    modes : sequence of int or None
        1-based mode indices contributing δψ raw states (subset of
        ``1..nmodes``). ``None`` uses all modes.
    nbnd_aug : int
        Number of δψ bands used per mode (R; ``-1`` = all bands in the file).
    nbnd_mf : int or None
        Number of original bands kept (M). ``None`` (or > N) keeps all N bands,
        giving an empty buffer [M, N) — use this for a dedicated bundle where
        N already equals the desired band count. Set M < N to emulate a smaller
        calculation from a larger dataset.
    smearing_deltapsi : float
        Buffer denominator smearing σ in Hartree (default 0.02).
    epstol : float
        Dimensionless singular-value cutoff: a state is kept when the residual
        amplitude s of its ``dtau_step``-scaled raw state satisfies
        ``s >= epstol``.
    dtau_step : float
        Displacement step in bohr multiplying the raw δψ states (default 0.1).
    """
    return mf.augment_basis_deltapsi(prefix, outdir, deltapsi_dir, elph_dir,
                                 [int(i) for i in iq_list],
                                 -1 if nmodes is None else int(nmodes),
                                 [] if modes is None else [int(m) for m in modes],
                                 int(nbnd_aug),
                                 -1 if nbnd_mf is None else int(nbnd_mf),
                                 float(smearing_deltapsi), float(epstol),
                                 float(dtau_step))
