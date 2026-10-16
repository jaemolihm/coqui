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


def _model_spectrum(H0, nelec, kpoint_weights):
    """
    T = 0 mean-field spectrum of ``H0`` in an orthonormal basis: ``(eigval, occ, efermi)``.

    Each state holds ``kpoint_weights[k] * spin_factor`` electrons when full (``spin_factor = 2``
    for ``ns = 1``, else 1) and ``nelec`` electrons are filled from the bottom. Levels within
    ``1e-10 * max(1, |e|)`` of the lowest level of a shell form that shell and share one fractional
    occupation. ``efermi`` is the midpoint between the highest occupied shell and the lowest not
    fully occupied one (the shell energy if one is partially occupied).
    """
    import numpy as np

    ns, nk, nb, _ = H0.shape
    eigval = np.linalg.eigvalsh(H0)
    cap = np.broadcast_to(kpoint_weights[None, :, None] * (2.0 if ns == 1 else 1.0), eigval.shape).ravel()
    order = np.argsort(eigval, axis=None, kind="stable")
    e = eigval.ravel()[order]
    occ = np.zeros(e.size)
    ne, e_homo, e_lumo = float(nelec), e[0], None
    i = 0
    while i < e.size and (e_lumo is None or ne > 1e-12):
        j = i + np.searchsorted(e[i:] - e[i], 1e-10 * max(1.0, abs(e[i])), side="right")
        c = cap[order[i:j]].sum()
        f = min(1.0, ne / c) if ne > 1e-12 else 0.0
        occ[order[i:j]] = f
        ne -= f * c
        if f > 0.0:
            e_homo = e[i]
        if f < 1.0 and e_lumo is None:
            e_lumo = e[i]
        i = j
    efermi = e_homo if e_lumo is None else 0.5 * (e_homo + e_lumo)
    return eigval, occ.reshape(eigval.shape), efermi


def write_model_h5(filename, H0, nelec, kpts_crys, latt=None,
                   kpoint_weights=None, fock=None, density_matrix=None,
                   madelung=0.0, nuclear_energy=0.0, fermi_energy=None, mpi=None):
    """
    Write a model mean-field HDF5 file, read by ``make_mf(mpi, {"prefix": p, "outdir": d}, "model")``
    from ``d/p.h5``.

    It checks shapes and writes datasets and attributes. ``H0(k)``, the k-point list and every
    other array come from the caller; the only quantities computed here are the T = 0 mean-field
    spectrum ``eigval``/``occ`` and ``fermi_energy`` of ``H0``, which the C++ reader takes from the
    file.

    The file format (every dataset and attribute, required / optional / rejected) is documented
    once, at the reader: ``model_system::initialize_from_h5`` in
    ``src/mean_field/model_hamiltonian/model_system.hpp``. This function writes, in group ``/System``:

    - attributes ``number_of_spins`` (``H0.shape[0]``), ``number_of_polarizations`` (1),
      ``number_of_bands`` (``H0.shape[2]``), ``number_of_elec``, ``madelung_constant``,
      ``nuclear_energy`` and ``fermi_energy`` (computed from ``H0`` unless given);
    - ``lattice_vectors``, ``kpoints_crystal``, ``H0``, ``fock_matrix`` (``H0`` unless given),
      ``eigval`` and ``occ`` (T = 0 filling of ``H0``);
    - ``kpoint_weights`` and ``density_matrix`` only when given; never ``overlap`` (the basis is
      orthonormal) or ``/System/BZ`` (the reader builds the BZ from the k list).

    k-points are in crystal coordinates on a full regular grid, **Gamma first**; the k index of every
    array follows ``kpts_crys``.

    Parameters
    ----------
    filename : str
        Output file, e.g. ``"./hubbard.h5"`` for prefix ``"hubbard"`` and outdir ``"./"``.
    H0 : array, complex, shape ``(ns, nk, nb, nb)``
        One-body Hamiltonian ``H0(k)`` in an orthonormal basis. ``ns = 1`` is paramagnetic.
    nelec : float
        Electrons per cell, both spins (a half-filled single band has ``nelec = 1``).
    kpts_crys : array, shape ``(nk, 3)``
        k-points in crystal coordinates; ``kpts_crys[0]`` must be Gamma.
    latt : array, shape ``(3, 3)``, optional
        Lattice vectors (rows), Bohr. Default: identity. Only ``k.R`` enters a model.
    kpoint_weights : array, shape ``(nk,)``, optional
        Must sum to 1.
    fock, density_matrix : array, complex, shape of ``H0``, optional
        ``fock`` is the full mean-field Hamiltonian including ``H0``; default ``H0``.
        ``density_matrix`` is not written when not given.
    madelung : float
        Weight of the q->0 head of the kernel. A model's V(q) is finite and Z[q=0] already holds it,
        so leave 0. Every q->0 correction is proportional to it, so ``div_treatment`` inputs then have
        no effect. A nonzero value adds ``-madelung * S D S`` to the HF exchange
        (``hf_div_treatment="gygi"``) while the GW head stays unscreened: the THC file has no q->0 heads.
    nuclear_energy : float
    fermi_energy : float, optional
        Overrides the Fermi energy computed from ``H0``.
    mpi : MpiHandler, optional
        If given, only the root rank writes, followed by a barrier.
    """
    import numpy as np
    import h5py
    from coqui.utils.h5_utils import write_complex_dataset

    H0 = np.asarray(H0)
    if H0.ndim != 4 or H0.shape[2] != H0.shape[3] or H0.shape[0] not in (1, 2):
        raise ValueError(f"write_model_h5: H0 must be [ns,nk,nb,nb] with ns in (1,2), got {H0.shape}")
    # eigvalsh reads one triangle only, so a non-Hermitian H0 would get a spectrum that is not the C++ H0's
    if np.abs(H0 - H0.conj().swapaxes(-1, -2)).max() > 1e-10 * np.abs(H0).max():
        raise ValueError("write_model_h5: H0(k) must be Hermitian")
    ns, nk, nb, _ = H0.shape
    kpts_crys = np.asarray(kpts_crys, dtype=np.float64)
    if kpts_crys.shape != (nk, 3):
        raise ValueError(f"write_model_h5: kpts_crys must be ({nk}, 3), got {kpts_crys.shape}")
    if np.abs(kpts_crys[0]).max() > 1e-12:
        raise ValueError("write_model_h5: the first k-point must be Gamma")
    latt = np.eye(3) if latt is None else np.asarray(latt, dtype=np.float64)
    if latt.shape != (3, 3):
        raise ValueError(f"write_model_h5: latt must be (3, 3), got {latt.shape}")
    if kpoint_weights is not None:
        kpoint_weights = np.asarray(kpoint_weights, dtype=np.float64)
        if kpoint_weights.shape != (nk,):
            raise ValueError(f"write_model_h5: kpoint_weights must be ({nk},), got {kpoint_weights.shape}")
        if abs(kpoint_weights.sum() - 1.0) > 1e-6:
            raise ValueError(f"write_model_h5: kpoint_weights must sum to 1, got {kpoint_weights.sum()}")
    fock = H0 if fock is None else fock
    mats = {"fock_matrix": fock, "density_matrix": density_matrix}
    for name, a in mats.items():
        if a is not None and np.shape(a) != H0.shape:
            raise ValueError(f"write_model_h5: {name} must have the shape of H0 {H0.shape}")

    if mpi is None or mpi.root():
        weights = np.full(nk, 1.0 / nk) if kpoint_weights is None else kpoint_weights
        eigval, occ, efermi = _model_spectrum(H0, nelec, weights)
        if fermi_energy is not None:
            efermi = fermi_energy
        with h5py.File(filename, "w") as f:
            g = f.create_group("System")
            g.attrs["number_of_spins"] = np.int32(ns)
            g.attrs["number_of_polarizations"] = np.int32(1)
            g.attrs["number_of_bands"] = np.int32(nb)
            g.attrs["number_of_elec"] = np.float64(nelec)
            g.attrs["madelung_constant"] = np.float64(madelung)
            g.attrs["nuclear_energy"] = np.float64(nuclear_energy)
            g.attrs["fermi_energy"] = np.float64(efermi)
            g.create_dataset("lattice_vectors", data=latt)
            g.create_dataset("kpoints_crystal", data=kpts_crys)
            if kpoint_weights is not None:
                g.create_dataset("kpoint_weights", data=kpoint_weights)
            write_complex_dataset(g, "H0", H0)
            g.create_dataset("eigval", data=eigval)
            g.create_dataset("occ", data=occ)
            for name, a in mats.items():
                if a is not None:
                    write_complex_dataset(g, name, a)
    if mpi is not None:
        mpi.barrier()
