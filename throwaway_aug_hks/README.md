# THROWAWAY — do not merge, do not review, delete this branch

Pushed 2026-08-17 as a **short-term backup only**. Expected to be stale within weeks.
Nothing here is meant to land: no design review, no test hardening, no API stability.
If you are reading this months later, delete the branch.

## What it was for

A one-time retrofit that added `Orbitals/H_KS_skij` — the full augmented-basis DFT
Kohn-Sham matrix over the IBZ — to augmented basis h5 files written *before* that
dataset existed, computing it from each file's **own** orbitals and **own** stored DFT
potentials (`Hamiltonian/ncpp/scf_local_potential` minus `pp_local_component`). No
parent DFT is reopened and nothing is re-augmented, so the ISDF/THC fit sitting on top
of the basis stays valid.

Needed because `hamilt::set_fock` seeds the one-body Hamiltonian from the stored matrix
when the basis has one and silently falls back to `diag(eigval)` otherwise. An augmented
basis is not an eigenbasis, so that fallback discarded **11–45% of the Frobenius norm of
H_KS** on the real bases — not the "few percent" that had been assumed. Every study on
such a basis was running on a diagonal G0.

## What it did, and the result

Ran against the 21 augmented bases the live studies read:
`~/ceph/eph_diagrammatics/12_diamond/{41_phonon.aug,43_phonon.aug.nk8}` and
`~/ceph/eph_diagrammatics/13_BaBiO3/{41_phonon.aug,43_phonon.aug.nk8}`. All 21 written
and independently verified. The 60 bases under `12_diamond/{32_phonon.fullaug,
33_phonon.paug,34_phonon.pmode}` and the 4 `13_BaBiO3/dev_41_phonon.aug.different_*`
were deliberately **not** retrofitted.

## The one idea worth keeping if this is ever redone

Since `a9f074b` (2026-07-18) the augmenter seeds `Orbitals/eigval` with exactly
`Re[diag(H_KS)]`. So recomputing H_KS and comparing its diagonal against the stored
`eigval` is a **free external correctness check** — it catches transposition,
conjugation and band-ordering errors that no self-consistency check can. It came out at
2.6e-16 … 3.3e-15 across all 21 bases. The protected↔augmentation block is a second free
gate (it vanishes analytically), but tolerate ~1e-4 there rather than machine epsilon:
it measures ~1e-8 on some fixtures for understood reasons.

Also load-bearing and easy to miss: every rank holds a lazily-opened read handle on the
basis h5, so they must all close it before the root reopens it for append, or HDF5
locking rejects the write — silently, in some configurations.

## Contents

- `add_aug_hks.py` — the driver (a copy; the original lived outside the repo in
  `~/eph_diagrammatics/dev/`). Dry run is the default; `--write` commits.
- The rest of the diff: `add_augmented_h_ks` in `src/hamiltonian/one_body_hamiltonian.hpp`,
  a `post_processing` branch in `src/methods/pproc/pproc_drivers.hpp`, its TOML name in
  `src/main/main.cpp`, the Python binding under `src/python/post_proc/`, and a round-trip
  test in `src/hamiltonian/tests/test_hamilt.cpp`.

## Not included

The worktree's uncommitted `CMakeLists.txt` c2py-ordering fix is deliberately excluded —
it is a local build fix that must stay uncommitted.
