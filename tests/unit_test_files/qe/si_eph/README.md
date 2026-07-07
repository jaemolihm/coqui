# si_eph — fixture for the bare electron-phonon vertex test

Data consumed by the Catch2 test `TEST_CASE("eph_bare_vertex", "[eph]")` in
`src/hamiltonian/tests/test_eph.cpp`. The test reassembles the bare e-ph vertex
`g = g_loc + g_nl` from CoQuí quantities only (the C++ side of
`coqui.compute_bare_eph_vertex`) and compares it to the `g_bare` that Quantum
ESPRESSO stores in its elph file, with **no fitting**.

If the fixture files below are absent the test **skips cleanly** (so the suite
stays green until the reference data is committed).

## System
Silicon, 2×2×2 k-mesh (nk=8), nbnd=20, 3·nat = 6 phonon modes, 2 q-points
(Γ and a zone-boundary q). Generated with the patched QE (`pw2coqui` +
`elph_coqui`); prefix `Si`.

## Files

| file | produced by | contents |
|------|-------------|----------|
| `Si.coqui.h5` | patched `pw2coqui` | QE mean field. Carries the `Hamiltonian/ncpp/vloc_radial` group (`r`, `rab`, `vloc` per species) + `z_valence`, which `pseudopot::build_dvloc_ion` reads. Loaded with `mf::h5_input_type`. |
| `Si.save/` | QE `pw.x` | QE save dir the h5 MF reader needs: `wfc{1..8}.hdf5`, `data-file-schema.xml`, `charge-density.hdf5` (+ the `.upf`). |
| `elph_bare.iq{1,2}.h5` | patched QE `elph_coqui` | reference per q-point, **stripped to just the two datasets the test reads** (`g_bare`, `xq_cryst`) to keep the fixture small. |

### Datasets read from each `elph_bare.iq{N}.h5`
- `xq_cryst` — `(3)` phonon wavevector, crystal (fractional) coordinates.
- `g_bare`   — `(nk, nmode, nb, 2*nb)` interleaved-real (`re, im, re, im, …`),
  **Rydberg**, band pair column-major (QE convention).

The test decodes/orients the reference exactly as the validated
`validate_gbare_allq.py` script does:
`ref(mode,k,m,n) = 0.5 * ( g_bare[k,mode,n,2m] + i·g_bare[k,mode,n,2m+1] )`
(0.5 = Ry→Ha; the `(m,n)` swap undoes the Fortran column-major storage), and
checks `||g − ref|| / ||ref|| < 1e-6` over the first-`min(nb, nb_ref)` band block.

Observed agreement (rank-independent, np=1/2/4): rel err `6.0e-11` at Γ,
`1.4e-10` at the zone-boundary q — machine precision.

## Regenerating / extending
The elph files were stripped from the full QE output with:
```python
import h5py
with h5py.File(src, "r") as f, h5py.File(dst, "w") as g:
    for k in ("g_bare", "xq_cryst"):
        f.copy(k, g)
```
Source data: `/mnt/ceph/users/jlihm/eph_diagrammatics/Si.nk4/testdata`.
If the prefix or q-point numbering changes, adjust `prefix` / the `elph_bare.iqN`
enumeration at the top of `test_eph.cpp`.
