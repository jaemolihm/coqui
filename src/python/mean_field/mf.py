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
               nbnd_aug=-1, epstol=1e-4, dirs=None, dtau_step=0.1):
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
    eigenbasis, its Kohn-Sham Hamiltonian is a full matrix rather than
    ``diag(eigval)``; the whole matrix is stored (``Orbitals/H_KS_skij``, IBZ
    only) and used as the one-body seed of any run on this basis, with
    ``Orbitals/eigval`` keeping its diagonal. Many-body runs on it must use
    ``h0_source="compute"``. Building that matrix needs the parent's DFT
    ``V_Hxc``, so the parent must export a multiplicative ``V_xc``: a QE-xml
    parent (re-export it through the h5 route) and a meta-GGA functional are
    refused outright rather than augmented without it.

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
    dirs : sequence of int or None
        For ``augment_type="momentum"``, the Cartesian directions (subset of
        ``{0, 1, 2}`` = x, y, z) whose momentum images p_alpha psi are appended.
        ``None`` (or empty) uses all three directions — the standard momentum
        augmentation. A single direction, e.g. ``dirs=[0]``, augments with p_x
        only, for phonon-mode-resolved studies where mode (kappa, d) is
        augmented with p_d alone.
    dtau_step : float
        Displacement step in bohr multiplying the raw dpsi/dtau states
        (default 0.1).
    """
    dirs_list = [] if dirs is None else [int(d) for d in dirs]
    return mf.augment_basis(prefix, outdir, augment_type, int(nbnd_aug),
                            float(epstol), dirs_list, float(dtau_step))


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
    eigenbasis, its Kohn-Sham Hamiltonian is a full matrix rather than
    ``diag(eigval)``; the whole matrix is stored (``Orbitals/H_KS_skij``, IBZ
    only) and used as the one-body seed of any run on this basis, with
    ``Orbitals/eigval`` keeping its diagonal. Many-body runs on it must use
    ``h0_source="compute"``. Building that matrix needs the parent's DFT
    ``V_Hxc``, so the parent must export a multiplicative ``V_xc``: a QE-xml
    parent (re-export it through the h5 route) and a meta-GGA functional are
    refused outright rather than augmented without it. Requires
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


def _write_complex(grp, name, a):
    """Write a complex array in the TRIQS/nda layout: float64 with a trailing (re, im) dimension
    and a ``__complex__`` attribute."""
    import numpy as np
    a = np.ascontiguousarray(a, dtype=np.complex128)
    ds = grp.create_dataset(name, data=a.view(np.float64).reshape(a.shape + (2,)))
    ds.attrs["__complex__"] = np.int8(1)


def write_model_h5(filename, H0, nelec, latt=None, kpts_crys=None, kp_grid=None,
                   kpoint_weights=None, fock=None, overlap=None, density_matrix=None,
                   madelung=0.0, nuclear_energy=0.0, fermi_energy=None, div_treatment=None, mpi=None):
    """
    Write a model mean-field HDF5 file, read by ``make_mf(mpi, {"prefix": p, "outdir": d}, "model")``
    from ``d/p.h5``.

    This is pure I/O: it checks shapes and writes datasets and attributes. ``H0(k)``, the k-point
    list and every other array come from the caller. Entries that are not given are not written;
    the C++ reader then uses its defaults.

    File layout (group ``/System``; "attr" = HDF5 attribute of ``/System``):

    ============================== ============== ============== ============================
    entry                          kind           shape          absent ->
    ============================== ============== ============== ============================
    number_of_spins                attr int                      required (``H0.shape[0]``)
    number_of_polarizations        attr int                      required (always 1)
    number_of_bands                attr int                      required (``H0.shape[2]``)
    number_of_elec                 attr float                    required (``nelec``)
    madelung_constant              attr float                    required (``madelung``)
    nuclear_energy                 attr float                    required (``nuclear_energy``)
    fermi_energy                   attr float                    computed from ``H0``
    div_treatment                  attr str                      ``"ignore_g0"``
    lattice_vectors                float64        [3,3]          required (identity if None)
    kpoints_crystal                float64        [nk,3]         generated from ``kp_grid``
    kp_grid                        int64          [3]            derived from ``kpoints_crystal``
    kpoint_weights                 float64        [nk]           ``1/nk``
    H0                             complex        [ns,nk,nb,nb]  required
    fock_matrix                    complex        [ns,nk,nb,nb]  ``H0``
    overlap                        complex        [ns,nk,nb,nb]  identity (only identity allowed)
    density_matrix                 complex        [ns,nk,nb,nb]  0
    ============================== ============== ============== ============================

    Conventions: Hartree, Bohr; ``k`` in crystal coordinates on a full regular Gamma-containing
    grid without symmetry, **Gamma first**. The k index of every array follows the k-list. The
    default list (only ``kp_grid`` given) is ``k = (i/N1, j/N2, l/N3)`` with index
    ``(i*N2 + j)*N3 + l``. Complex arrays use the TRIQS/nda layout (float64 with a trailing
    (re, im) dimension and a ``__complex__`` attribute). The BZ maps are built by the reader.

    Parameters
    ----------
    filename : str
        Output file, e.g. ``"./hubbard.h5"`` for prefix ``"hubbard"`` and outdir ``"./"``.
    H0 : array, complex, shape ``(ns, nk, nb, nb)``
        One-body Hamiltonian ``H0(k)`` in an orthonormal basis. ``ns = 1`` is paramagnetic.
    nelec : float
        Electrons per cell, both spins (a half-filled single band has ``nelec = 1``).
    latt : array, shape ``(3, 3)``, optional
        Lattice vectors (rows), Bohr. Default: identity. Only ``k.R`` enters a model.
    kpts_crys : array, shape ``(nk, 3)``, optional
        k-points in crystal coordinates; ``kpts_crys[0]`` must be Gamma.
    kp_grid : sequence of 3 int, optional
        Monkhorst-Pack mesh, e.g. ``(N, N, 1)``. At least one of ``kpts_crys``/``kp_grid``.
    kpoint_weights : array, shape ``(nk,)``, optional
        Must sum to 1.
    fock, overlap, density_matrix : array, complex, shape of ``H0``, optional
        ``fock`` is the full mean-field Hamiltonian including ``H0``.
    madelung, nuclear_energy : float
    fermi_energy : float, optional
        Written only if given; the reader keeps it.
    div_treatment : str, optional
        q->0 treatment of the Coulomb kernel, imposed on every MBPT driver over its input. Only the HF
        exchange of HF-type solvers supports a value other than ``"ignore_g0"`` (e.g. ``"gygi"`` with
        ``madelung``); GW, RPA and the downfolding drivers abort otherwise, since a model interaction file
        has no THC q->0 heads.
    mpi : MpiHandler, optional
        If given, only the root rank writes, followed by a barrier.
    """
    import numpy as np
    import h5py

    H0 = np.asarray(H0)
    if H0.ndim != 4 or H0.shape[2] != H0.shape[3] or H0.shape[0] not in (1, 2):
        raise ValueError(f"write_model_h5: H0 must be [ns,nk,nb,nb] with ns in (1,2), got {H0.shape}")
    ns, nk, nb, _ = H0.shape
    if kpts_crys is None and kp_grid is None:
        raise ValueError("write_model_h5: give kpts_crys and/or kp_grid")
    if kpts_crys is not None:
        kpts_crys = np.asarray(kpts_crys, dtype=np.float64)
        if kpts_crys.shape != (nk, 3):
            raise ValueError(f"write_model_h5: kpts_crys must be ({nk}, 3), got {kpts_crys.shape}")
        if np.abs(kpts_crys[0]).max() > 1e-12:
            raise ValueError("write_model_h5: the first k-point must be Gamma")
    if kp_grid is not None:
        kp_grid = np.asarray(kp_grid, dtype=np.int64)
        if kp_grid.shape != (3,) or np.prod(kp_grid) != nk:
            raise ValueError(f"write_model_h5: kp_grid must have 3 entries with product {nk}, got {kp_grid}")
    latt = np.eye(3) if latt is None else np.asarray(latt, dtype=np.float64)
    if latt.shape != (3, 3):
        raise ValueError(f"write_model_h5: latt must be (3, 3), got {latt.shape}")
    if kpoint_weights is not None:
        kpoint_weights = np.asarray(kpoint_weights, dtype=np.float64)
        if kpoint_weights.shape != (nk,):
            raise ValueError(f"write_model_h5: kpoint_weights must be ({nk},), got {kpoint_weights.shape}")
        if abs(kpoint_weights.sum() - 1.0) > 1e-6:
            raise ValueError(f"write_model_h5: kpoint_weights must sum to 1, got {kpoint_weights.sum()}")
    mats = {"fock_matrix": fock, "overlap": overlap, "density_matrix": density_matrix}
    for name, a in mats.items():
        if a is not None and np.shape(a) != H0.shape:
            raise ValueError(f"write_model_h5: {name} must have the shape of H0 {H0.shape}")
    if overlap is not None and np.abs(np.asarray(overlap) - np.eye(nb)).max() > 1e-10:
        raise ValueError("write_model_h5: overlap must be the identity (the C++ reader aborts otherwise)")

    if mpi is None or mpi.root():
        with h5py.File(filename, "w") as f:
            g = f.create_group("System")
            g.attrs["number_of_spins"] = np.int32(ns)
            g.attrs["number_of_polarizations"] = np.int32(1)
            g.attrs["number_of_bands"] = np.int32(nb)
            g.attrs["number_of_elec"] = np.float64(nelec)
            g.attrs["madelung_constant"] = np.float64(madelung)
            g.attrs["nuclear_energy"] = np.float64(nuclear_energy)
            if fermi_energy is not None:
                g.attrs["fermi_energy"] = np.float64(fermi_energy)
            if div_treatment is not None:
                g.attrs["div_treatment"] = np.bytes_(div_treatment)
            g.create_dataset("lattice_vectors", data=latt)
            if kpts_crys is not None:
                g.create_dataset("kpoints_crystal", data=kpts_crys)
            if kp_grid is not None:
                g.create_dataset("kp_grid", data=kp_grid)
            if kpoint_weights is not None:
                g.create_dataset("kpoint_weights", data=kpoint_weights)
            _write_complex(g, "H0", H0)
            for name, a in mats.items():
                if a is not None:
                    _write_complex(g, name, a)
    if mpi is not None:
        mpi.barrier()
