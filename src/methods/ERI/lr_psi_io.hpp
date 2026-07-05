/**
 * ==========================================================================
 * CoQuí: Correlated Quantum ínterface
 *
 * Copyright (c) 2022-2026 Simons Foundation & The CoQuí developer team
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ==========================================================================
 */

#ifndef COQUI_LR_PSI_IO_HPP
#define COQUI_LR_PSI_IO_HPP

// Shared utilities for reading perturbed wavefunctions (δψ) from
// per-k HDF5 files and applying the i(k+G) gradient in plane-wave space.
//
// For Bloch phases e^{ik·r} on the FFT grid or at sparse FFT-grid points,
// call utils::rspace_phase_factor (in utilities/kpoint_utils.hpp) directly.

#include <string>
#include "configuration.hpp"
#include "utilities/check.hpp"
#include "utilities/kpoint_utils.hpp"
#include "mean_field/MF.hpp"
#include "h5/h5.hpp"
#include "nda/h5.hpp"
#include "nda/nda.hpp"

namespace methods {
namespace detail {

// Multiply PW coefficients on an FFT grid by i(k+G)_α in place.
// c_grid: (1, N1, N2, N3) complex, PW coefficients indexed by the FFT grid.
// k_cart: k-vector in Cartesian (with 2π).
// recv  : (3,3) reciprocal lattice vectors, recv(i,:) = i-th reciprocal vector (with 2π).
// alpha : Cartesian direction (0, 1, or 2).
inline void multiply_by_ikpG(
    nda::array_view<ComplexType, 4> c_grid,
    nda::stack_array<double, 3> const& k_cart,
    nda::stack_array<double, 3, 3> const& recv,
    int alpha)
{
  long N1 = c_grid.shape(1), N2 = c_grid.shape(2), N3 = c_grid.shape(3);
  for (long i = 0; i < N1; ++i) {
    long n1 = (i <= N1 / 2) ? i : i - N1;
    for (long j = 0; j < N2; ++j) {
      long n2 = (j <= N2 / 2) ? j : j - N2;
      for (long k = 0; k < N3; ++k) {
        long n3 = (k <= N3 / 2) ? k : k - N3;
        double kpG_alpha = k_cart(alpha)
                         + double(n1) * recv(0, alpha)
                         + double(n2) * recv(1, alpha)
                         + double(n3) * recv(2, alpha);
        c_grid(0, i, j, k) *= ComplexType{0.0, kpG_alpha};
      }
    }
  }
}

// Container for per-k δψ data read from {prefix}_ik{ik+1}.hdf5.
// evc_raw is raw (real, imag) interleaved doubles as written by QE; decode with decode_band_to_fft_grid.
struct Deltapsi_k_data {
  nda::array<int, 2> miller;      // (npw, 3)
  nda::array<int, 1> k2g;         // (npw,) FFT-grid linear indices
  nda::array<double, 2> evc_raw;  // (nbnd_file, 2 * npw)
  long npw{0};
  long nbnd{0};
};

// Read {prefix}_ik{ik+1}.hdf5 and return miller, k2g, and raw evc.
// fft_mesh is required because Miller → FFT-grid linearisation depends on the target grid.
inline Deltapsi_k_data read_Deltapsi_k(
    std::string const& Deltapsi_prefix,
    long ik,
    nda::stack_array<long, 3> const& fft_mesh)
{
  Deltapsi_k_data out;
  std::string path = Deltapsi_prefix + "_ik" + std::to_string(ik + 1) + ".hdf5";
  h5::file f(path, 'r');

  nda::h5_read(f, "MillerIndices", out.miller);
  out.npw = out.miller.shape(0);

  out.k2g = nda::array<int, 1>(out.npw);
  utils::generate_k2g(out.miller, out.k2g, fft_mesh);

  nda::h5_read(f, "evc", out.evc_raw);
  out.nbnd = out.evc_raw.shape(0);
  return out;
}

// Container for per-k ψ data read from the QE wfc HDF5 file.
// Same shape as Deltapsi_k_data; filename selected from (is, nspin).
struct wfc_data_t {
  nda::array<int, 2> miller;       // (npw, 3)
  nda::array<int, 1> k2g;          // (npw,) FFT-grid linear indices
  nda::array<double, 2> evc_raw;   // (nbnd_file, 2 * npw)
  long npw{0};
  long nbnd{0};
};

// Read the QE wfc HDF5 file for (is, ik) into a wfc_data_t. Mirrors read_Deltapsi_k.
inline wfc_data_t read_wfc_k(
    mf::MF const& mf,
    long is,
    long ik,
    nda::stack_array<long, 3> const& fft_mesh)
{
  wfc_data_t out;
  long nspin = mf.nspin();
  std::string wfc_path;
  if (nspin == 1)
    wfc_path = mf.outdir() + "/" + mf.prefix() + ".save/wfc" + std::to_string(ik + 1) + ".hdf5";
  else if (is == 0)
    wfc_path = mf.outdir() + "/" + mf.prefix() + ".save/wfcup" + std::to_string(ik + 1) + ".hdf5";
  else
    wfc_path = mf.outdir() + "/" + mf.prefix() + ".save/wfcdw" + std::to_string(ik + 1) + ".hdf5";

  h5::file f(wfc_path, 'r');
  nda::h5_read(f, "MillerIndices", out.miller);
  out.npw = out.miller.shape(0);
  out.k2g = nda::array<int, 1>(out.npw);
  utils::generate_k2g(out.miller, out.k2g, fft_mesh);
  nda::h5_read(f, "evc", out.evc_raw);
  out.nbnd = out.evc_raw.shape(0);
  return out;
}

// Decode a single band from interleaved (real, imag) to a complex FFT grid.
// c_on_grid must be pre-zeroed; only PW components listed in k2g are written.
// (nnr = size of c_on_grid; k2g values must satisfy 0 <= k2g(ig) < nnr.)
inline void decode_band_to_fft_grid(
    nda::array<double, 2> const& evc_raw,
    long m,
    nda::array<int, 1> const& k2g,
    nda::array_view<ComplexType, 1> c_on_grid)
{
  long npw = k2g.shape(0);
  for (long ig = 0; ig < npw; ++ig) {
    double re = evc_raw(m, 2 * ig);
    double im = evc_raw(m, 2 * ig + 1);
    c_on_grid(k2g(ig)) = ComplexType{re, im};
  }
}

} // namespace detail
} // namespace methods

#endif // COQUI_LR_PSI_IO_HPP
