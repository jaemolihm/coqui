#!/usr/bin/env python3
"""Retrofit the augmented Kohn-Sham matrix into an existing augmented basis h5.

Adds ``Orbitals/H_KS_skij`` -- the full augmented-basis DFT Kohn-Sham matrix
``H_KS(i,j) = <phi_i|H0|phi_j> + <phi_i|V_H + V_xc|phi_j>`` over the IBZ -- to a
``<prefix>_aug.h5`` that was written before that dataset existed. The matrix is
computed from the file's *own* orbitals and its *own* stored DFT potentials
(``Hamiltonian/ncpp/scf_local_potential`` minus ``pp_local_component``). No parent
DFT mean field is opened, nothing is re-augmented, and no orbital is regenerated --
so the ISDF/THC fit sitting on top of the basis stays valid.

Why this exists
---------------
``hamilt::set_fock`` seeds the one-body Hamiltonian from the stored Kohn-Sham matrix
when the basis carries one, and otherwise falls back to ``diag(eigval)``. An
augmented basis is *not* an eigenbasis, so the fallback silently discards its
off-diagonal couplings (several percent of the norm) and every study on such a
basis runs on a diagonal G0. Bases built before 2026-08-17 have no
``H_KS_skij`` and take that fallback. Rebuilding them from the parent DFT would
regenerate the orbitals and force a THC refit, which is not wanted; this retrofits
the dataset in place instead.

WARNING
-------
Adding the dataset **changes what every study reading that basis computes on its
next run** -- ground-state SCF, qpGW, the linear-response "mf_dft" G0, RPA and
downfolding all take their iter-0 one-body seed through ``set_fock``. Results
obtained before and after are not comparable. Hence ``--dry-run`` is the default:
without ``--write`` nothing is ever modified.

Validation
----------
CoQuí reports, per basis and whether or not it writes:
  * the hermitization residual ``||H_herm - H||/||H||``
  * ``||Re diag(H_KS) - eigval||/||eigval||`` and its max -- a **hard gate**: the
    augmenter seeded ``Orbitals/eigval`` with exactly this diagonal, so a mismatch
    means the recomputed matrix is not the object the file was built with
  * ``||Im diag(H_KS)||``, ``||H_KS||`` and the off-diagonal weight
    ``||H - diag(H)||/||H||`` (what the fallback was dropping)
  * with ``nbnd_protected`` known, the protected-to-augmentation block
    ``||H[0:np, np:]||/||H||``, which vanishes analytically and is therefore a free
    basis-ordering and conjugation check (~1e-8 on a real dpsi basis)

``nbnd_protected`` is read from each basis folder's ``aug_meta.json``; the gate is
skipped with a warning if the file or the key is missing.

A failed gate is an ``MPI_Abort``, so it takes the whole job down and the remaining
folders on the command line are not processed. Bases written before the Kohn-Sham
eigval seed (CoQuí ``a9f074b``, 2026-07-18) carry the kinetic Rayleigh diagonal in
``Orbitals/eigval`` and will legitimately trip the gate -- those cannot be retrofitted
by this path.

Usage
-----
    source ~/.bashrc && load_coqui
    srun -n <N> python add_aug_hks.py <basis-dir> [more dirs ...]           # dry run
    srun -n <N> python add_aug_hks.py <basis-dir> [more dirs ...] --write
    srun -n <N> python add_aug_hks.py <basis-dir> --write --overwrite

Each <basis-dir> holds exactly one ``*_aug.h5`` plus (ideally) ``aug_meta.json``.
This script does not submit anything -- run it inside your own allocation.
"""
import argparse
import glob
import json
import os
import sys

from mpi4py import MPI  # must precede coqui / h5py imports
import h5py


def find_aug_h5(path):
    """The {prefix}_aug.h5 in `path` (or `path` itself if it is the file)."""
    if os.path.isfile(path):
        return path
    hits = sorted(glob.glob(os.path.join(path, "*_aug.h5")))
    if len(hits) != 1:
        raise FileNotFoundError(
            f"{path}: expected exactly one *_aug.h5, found {len(hits)}")
    return hits[0]


def read_nbnd_protected(aug_dir, log):
    """nbnd_protected from aug_dir/aug_meta.json, or -1 (with a warning)."""
    meta_p = os.path.join(aug_dir, "aug_meta.json")
    if not os.path.exists(meta_p):
        log(f"  [WARNING] {meta_p} missing -- the protected-block gate is skipped.")
        return -1
    with open(meta_p) as f:
        meta = json.load(f)
    if "nbnd_protected" not in meta:
        log(f"  [WARNING] {meta_p} has no 'nbnd_protected' -- the protected-block "
            f"gate is skipped.")
        return -1
    return int(meta["nbnd_protected"])


def main():
    parser = argparse.ArgumentParser(
        description="Add Orbitals/H_KS_skij to existing augmented basis h5 files.")
    parser.add_argument("dirs", nargs="+",
                        help="Basis folders, each holding one <prefix>_aug.h5 "
                             "(a path to the h5 itself also works).")
    parser.add_argument("--write", action="store_true",
                        help="Actually add the dataset. Without it this is a dry "
                             "run: the matrix is computed and validated, and "
                             "nothing is modified.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Explicitly request the default (no write).")
    parser.add_argument("--overwrite", action="store_true",
                        help="Replace an existing Orbitals/H_KS_skij instead of "
                             "skipping the basis.")
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    def log(m=""):
        if rank == 0:
            print(m, flush=True)

    write = args.write and not args.dry_run
    if args.write and args.dry_run:
        log("[add_aug_hks] --dry-run overrides --write; nothing will be modified.")

    import coqui

    mpi = coqui.MpiHandler()
    coqui.set_verbosity(mpi, output_level=2)

    log(f"[add_aug_hks] mode = {'WRITE' if write else 'dry run'}"
        f"{', overwrite' if args.overwrite else ''}, ranks = {comm.Get_size()}")
    if write:
        log("[add_aug_hks] WARNING: adding Orbitals/H_KS_skij changes what every "
            "study reading these bases computes on its next run.")

    n_done, n_skipped = 0, 0
    for d in args.dirs:
        aug_h5 = os.path.abspath(find_aug_h5(d))
        aug_dir = os.path.dirname(aug_h5)
        prefix = os.path.basename(aug_h5)[:-len(".h5")]

        present = None
        if rank == 0:
            with h5py.File(aug_h5, "r") as f:
                present = "H_KS_skij" in f["Orbitals"]
        present = comm.bcast(present, root=0)

        log()
        log(f"[add_aug_hks] {aug_h5}")
        log(f"  H_KS_skij already present = {present}")
        if present and not args.overwrite:
            log("  -> skipped (pass --overwrite to recompute and replace it).")
            n_skipped += 1
            continue

        nbnd_protected = read_nbnd_protected(aug_dir, log)
        log(f"  nbnd_protected = {nbnd_protected}")

        aug = coqui.make_mf(mpi, {"prefix": prefix, "outdir": aug_dir}, "bdft")
        coqui.post_proc.add_augmented_h_ks(aug, {
            "write": write,
            "overwrite": args.overwrite,
            "nbnd_protected": nbnd_protected,
        })
        n_done += 1

    log()
    log(f"[add_aug_hks] {n_done} basis/bases processed, {n_skipped} skipped"
        f"{'' if write else ' (dry run: no file modified)'}.")


if __name__ == "__main__":
    sys.exit(main())
