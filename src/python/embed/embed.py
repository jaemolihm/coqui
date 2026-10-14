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
import numpy as np

from h5 import HDFArchive
import coqui._lib.embed_module as embed_cxx


def combine_impurities(C_ksIai, imp_dims):
    nkpts, nspins, nImps, nImpOrbs, Norbs = C_ksIai.shape
    C_ksIai_new = np.zeros((nkpts, nspins, 1, np.sum(imp_dims), Norbs), dtype=C_ksIai.dtype)
    offset = 0
    for I, dim in enumerate(imp_dims):
        C_ksIai_new[:, :, 0, offset:offset+dim] = C_ksIai[:, :, I, :dim]
        offset += dim

    return C_ksIai_new


def read_proj_info(wannier_h5):
  with HDFArchive(wannier_h5, 'r') as ar:
    C_ksIai = ar['dft_input/proj_mat']
    band_window = ar['dft_misc_input/band_window']
    kpts_w90 = ar['dft_input/kpts']

    nImps = band_window.shape[0]
    imp_dims = []
    for I in range(nImps):
        imp_dims.append(ar[f'dft_input/shells/{I}/dim'])

    if nImps > 1:
        # combine multiple impurities into one
        C_ksIai = combine_impurities(C_ksIai, imp_dims)
        band_window = band_window[:1]

  return {'proj_mat': C_ksIai, 'band_window': band_window, 'kpts_w90': kpts_w90}



def downfold_local_gf(mf, params, *, projector_info=None):
    """
    Downfolds the local Green's function (GF) from the Kohn-Sham (KS) basis to the Wannier basis.

    Args:
        mf: Mean-field object.
        params (dict): Parameters for the downfolding process.
        projector_info (dict, optional): Projector information containing:
            - "proj_mat": The projector matrix.
            - "band_window": The band window information.
            - "kpts_w90": The k-points in Wannier90 format.

    Returns:
        np.ndarray: The downfolded local Green's function.
    """
    if projector_info is None:
        return embed_cxx.downfold_gloc_with_projector_from_h5(mf, json.dumps(params))[:,:,0]
    else:
        proj_mat = projector_info.get("proj_mat")
        band_window = projector_info.get("band_window")
        kpts_w90 = projector_info.get("kpts_w90")
        # ignore the number of impurities for now
        return embed_cxx.downfold_gloc(
            mf, json.dumps(params), proj_mat, band_window, kpts_w90
        )[:,:,0]




def downfold_coulomb(h_int, params, *, projector_info=None, local_polarizabilities=None):
    """
    Downfolds the local Coulomb interaction matrix.

    This function computes the downfolded local Coulomb interaction matrix
    using the provided Coulomb hamiltonian, parameters, and optional projector
    information or local polarizabilities.

    Args:
        h_int (ThcCoulomb): An instance of the ThcCoulomb class, which represents
            the THC (Tensor HyperContraction) Coulomb interaction object. This
            class provides methods to access properties such as the number of
            k-points, spin states, and bands, as well as MPI and mean-field data.
        params (dict): A dictionary of parameters required for the downfolding process.
        projector_info (dict, optional): A dictionary containing projector-related
            information. Expected keys include:
            - "proj_mat": The projector matrix.
            - "band_window": The band window information.
            - "kpts_w90": The k-points in Wannier90 format.
        local_polarizabilities (dict, optional): A dictionary containing local
            polarizability information. Expected keys include:
            - "imp": Impurity polarizabilities.
            - "dc": Double-counting corrections.

    Returns:
        np.ndarray: The downfolded local Coulomb interaction matrix.

    Raises:
        ValueError: If required keys are missing in the `local_polarizabilities`
        dictionary.
    """
    if local_polarizabilities is not None:
        required_keys = {"imp", "dc"}
        missing = required_keys - local_polarizabilities.keys()
        if missing:
            raise ValueError(f"Missing keys: {missing}")

    if projector_info is None:
        return embed_cxx.downfold_coulomb_with_projector_from_h5(
            h_int, json.dumps(params), local_polarizabilities=local_polarizabilities
        )
    else:
        proj_mat = projector_info.get("proj_mat")
        band_window = projector_info.get("band_window")
        kpts_w90 = projector_info.get("kpts_w90")
        return embed_cxx.downfold_coulomb(
            h_int, json.dumps(params), proj_mat, band_window, kpts_w90,
            local_polarizabilities=local_polarizabilities
        )



def _root_input(arr, dtype):
    from mpi4py import MPI
    return np.asarray(arr, dtype=dtype) if MPI.COMM_WORLD.Get_rank() == 0 else None


def lr_downfold_local_gf(mf, params, dG_tskij, q_vec, *, projector_info):
    """
    Linear response of the local Green's function at momentum transfer Q:
    dGloc = (1/N_k) sum_k C^{k+Q} dG^k C^{k,dagger}, with dG^k the (k+Q, k) block.
    The projector algebra is tested at Q != 0 (synthetic adjoint and NumPy references); the physical Q != 0
    end-to-end response is not yet validated.

    Args:
        mf: Mean-field object (full BZ, no symmetry).
        params (dict): optional "translate_home_cell".
        dG_tskij (np.ndarray): (nt, ns, nk, nbnd, nbnd); required on the MPI root, ignored elsewhere.
        q_vec: Q in crystal coordinates (3,).
        projector_info (dict): "proj_mat", "band_window", "kpts_w90" as for downfold_local_gf.

    Returns:
        np.ndarray: (nt, ns, nImpOrbs, nImpOrbs) on every rank. Not hermitized.
    """
    return embed_cxx.lr_downfold_gloc(
        mf, json.dumps(params), projector_info["proj_mat"], projector_info["band_window"],
        projector_info["kpts_w90"], _root_input(dG_tskij, np.complex128), np.asarray(q_vec, dtype=np.float64)
    )[:,:,0]


def lr_downfold_coulomb(h_int, params, dW_xqPQ, q_vec, *, projector_info):
    """
    Linear response of the local screened interaction at momentum transfer Q:
    dWloc_{x,cdab} = (1/N_q) sum_q conj(B^{q+Q}_{P,dc}) dW^q_{x,PQ} B^q_{Q,ab}, with dW^q the (q+Q, q) block.
    Full q mesh, single impurity. The q->0 head term is not included: the result is exact when the head of dW
    vanishes (models), otherwise it omits the head that the ground-state Wloc carries with "gygi".
    The projector algebra is tested at Q != 0 (synthetic adjoint and NumPy references); the physical Q != 0
    end-to-end response is not yet validated.

    Args:
        h_int (ThcCoulomb): THC Coulomb handler (full q mesh).
        params (dict): optional "translate_home_cell".
        dW_xqPQ (np.ndarray): (nx, nq, NP, NP) with any leading axis (tau or iw); required on the MPI root.
        q_vec: Q in crystal coordinates (3,).
        projector_info (dict): "proj_mat", "band_window", "kpts_w90" as for downfold_coulomb.

    Returns:
        np.ndarray: (nx, c, d, a, b) in chemistry order, as the screened part returned by downfold_coulomb.
        The DC and Weiss kernels of coqui.dmft.weiss take the product basis (chemistry_to_product_basis).
    """
    return embed_cxx.lr_downfold_coulomb(
        h_int, json.dumps(params), projector_info["proj_mat"], projector_info["band_window"],
        projector_info["kpts_w90"], _root_input(dW_xqPQ, np.complex128), np.asarray(q_vec, dtype=np.float64)
    )


def downfold_1e(mf, params):
    embed_cxx.downfold_1e(mf, json.dumps(params))


def downfold_2e(h_int, params, *, local_polarizabilities=None):
    if local_polarizabilities is not None:
        required_keys = {"imp", "dc"}
        missing = required_keys - local_polarizabilities.keys()
        if missing:
            raise ValueError(f"Missing keys: {missing}")

    embed_cxx.downfold_2e(h_int, json.dumps(params),
                          local_polarizabilities=local_polarizabilities)




def dmft_embed(mf, params, *, local_hf_potentials=None, local_sigma_dynamic=None,
               projector_info=None):

    if local_sigma_dynamic is not None and local_hf_potentials is not None:
        required_keys = {"imp", "dc"}
        missing = required_keys - local_sigma_dynamic.keys()
        if missing:
            raise ValueError(f"Missing keys in local_sigma_dynamic: {missing}")

        # Append additional axis for the number of impurities
        for key in required_keys:
            if len(local_hf_potentials[key].shape) == 3:
                local_hf_potentials[key] = np.expand_dims(local_hf_potentials[key], axis=1)
            if len(local_sigma_dynamic[key].shape) == 4:
                local_sigma_dynamic[key] = np.expand_dims(local_sigma_dynamic[key], axis=2)
    else:
        local_hf_potentials = None
        local_sigma_dynamic = None

    if projector_info is None:
        embed_cxx.dmft_embed_with_projector_from_h5(
            mf, json.dumps(params),
            local_hf_potentials, local_sigma_dynamic
        )
    else:
        proj_mat = projector_info.get("proj_mat")
        band_window = projector_info.get("band_window")
        kpts_w90 = projector_info.get("kpts_w90")
        embed_cxx.dmft_embed(
            mf, json.dumps(params), proj_mat, band_window, kpts_w90,
            local_hf_potentials, local_sigma_dynamic,
        )
