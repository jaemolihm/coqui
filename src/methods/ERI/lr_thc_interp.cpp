/**
 * ==========================================================================
 * CoQuí: Correlated Quantum ínterface
 *
 * Copyright (c) 2022-2025 Simons Foundation & The CoQuí developer team
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

#include "lr_thc_interp.hpp"

#include "configuration.hpp"
#include "utilities/check.hpp"
#include "utilities/kpoint_utils.hpp"
#include "numerics/fft/nda.hpp"
#include "h5/h5.hpp"
#include "nda/h5.hpp"
#include "nda/nda.hpp"

namespace methods {

namespace {

// Compute Bloch phase e^{ik.r} at interpolating points r_P.
// k_cart: (3,) k-vector in Cartesian (with 2pi).
// r_P: (nP,) linearized FFT grid indices.
// fft_mesh: FFT grid dimensions.
// lattv: (3,3) lattice vectors, rows = lattice vectors.
// Returns: (nP,) complex phase factors.
nda::array<ComplexType, 1> compute_phase_at_points(
    nda::stack_array<double, 3> const& k_cart,
    nda::array<long, 1> const& r_P,
    nda::stack_array<long, 3> const& fft_mesh,
    nda::stack_array<double, 3, 3> const& lattv)
{
  long nP = r_P.shape(0);
  long N1 = fft_mesh(0), N2 = fft_mesh(1), N3 = fft_mesh(2);
  nda::array<ComplexType, 1> phase(nP);

  for (long u = 0; u < nP; ++u) {
    long idx = r_P(u);
    long k_idx = idx % N3;
    long rem = idx / N3;
    long j_idx = rem % N2;
    long i_idx = rem / N2;

    // Fractional coordinates
    double fi = double(i_idx) / double(N1);
    double fj = double(j_idx) / double(N2);
    double fk = double(k_idx) / double(N3);

    // Cartesian position: r = fi*a1 + fj*a2 + fk*a3
    double rx = fi * lattv(0, 0) + fj * lattv(1, 0) + fk * lattv(2, 0);
    double ry = fi * lattv(0, 1) + fj * lattv(1, 1) + fk * lattv(2, 1);
    double rz = fi * lattv(0, 2) + fj * lattv(1, 2) + fk * lattv(2, 2);

    double kr = k_cart(0) * rx + k_cart(1) * ry + k_cart(2) * rz;
    phase(u) = std::exp(ComplexType{0.0, kr});
  }
  return phase;
}

// Multiply PW coefficients on FFT grid by i*(k+G)_alpha for gradient computation.
// c_grid: (1, N1, N2, N3) complex, PW coefficients on FFT grid. Modified in place.
// k_cart: k-vector in Cartesian.
// recv: (3,3) reciprocal lattice vectors, recv(i,:) = i-th reciprocal vector (Cartesian, with 2pi).
// alpha: Cartesian direction (0=x, 1=y, 2=z).
void multiply_by_ikpG(
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
        // G_cart[alpha] = n1*recv(0,alpha) + n2*recv(1,alpha) + n3*recv(2,alpha)
        double kpG_alpha = k_cart(alpha)
                         + double(n1) * recv(0, alpha)
                         + double(n2) * recv(1, alpha)
                         + double(n3) * recv(2, alpha);
        c_grid(0, i, j, k) *= ComplexType{0.0, kpG_alpha};
      }
    }
  }
}

} // anonymous namespace

nda::array<ComplexType, 4> compute_delta_X(
    std::shared_ptr<mf::MF> mf,
    std::string const& dpsi_prefix,
    nda::array<long, 1> const& r_P,
    nda::array<double, 2> const& delta_r_P,
    nda::array<double, 1> const& q_vec_cryst,
    nda::array<long, 1> const& fft_grid)
{
  utils::check(mf->npol() == 1, "compute_delta_X: Non-collinear not yet implemented");
  utils::check(fft_grid.shape(0) == 3,
               "compute_delta_X: fft_grid must have size 3, got {}", fft_grid.shape(0));

  long nspin = mf->nspin();
  long nk_ibz = mf->nkpts_ibz();
  long nbnd = mf->nbnd();
  long nP = r_P.shape(0);
  auto lattv = mf->lattv();
  auto recv = mf->recv();
  auto kpts = mf->kpts();  // (nk, 3) Cartesian with 2pi

  nda::stack_array<long, 3> fft_mesh;
  for (int i = 0; i < 3; ++i) fft_mesh(i) = fft_grid(i);
  long nnr = fft_mesh(0) * fft_mesh(1) * fft_mesh(2);

  utils::check(delta_r_P.shape(0) == nP && delta_r_P.shape(1) == 3,
               "compute_delta_X: delta_r_P must have shape (nP, 3), got ({}, {})",
               delta_r_P.shape(0), delta_r_P.shape(1));
  utils::check(q_vec_cryst.shape(0) == 3,
               "compute_delta_X: q_vec_cryst must have size 3");

  // q in Cartesian: q_cart = q_cryst @ recv
  nda::stack_array<double, 3> q_cart;
  for (int a = 0; a < 3; ++a)
    q_cart(a) = q_vec_cryst(0) * recv(0, a) + q_vec_cryst(1) * recv(1, a) + q_vec_cryst(2) * recv(2, a);

  nda::array<ComplexType, 4> result(nspin, nk_ibz, nP, nbnd);
  result() = ComplexType(0.0);

  // FFT workspace: (1, N1, N2, N3)
  nda::array<ComplexType, 2> fft_buf(1, nnr);
  auto fft_buf_4d = nda::reshape(fft_buf, std::array<long, 4>{1, fft_mesh(0), fft_mesh(1), fft_mesh(2)});

  // Workspace for gradient direction
  nda::array<ComplexType, 2> grad_buf(1, nnr);
  auto grad_buf_4d = nda::reshape(grad_buf, std::array<long, 4>{1, fft_mesh(0), fft_mesh(1), fft_mesh(2)});

  // Create FFT plan (reused for all transforms)
  math::nda::fft<true> F(fft_buf_4d);

  for (long is = 0; is < nspin; ++is) {
    for (long ik = 0; ik < nk_ibz; ++ik) {
      // k-vector in Cartesian
      nda::stack_array<double, 3> k_cart;
      for (int a = 0; a < 3; ++a) k_cart(a) = kpts(ik, a);

      // (k+q) in Cartesian for deltapsi Bloch phase
      nda::stack_array<double, 3> kpq_cart;
      for (int a = 0; a < 3; ++a) kpq_cart(a) = k_cart(a) + q_cart(a);

      // Bloch phases at interpolating points
      auto phase_k = compute_phase_at_points(k_cart, r_P, fft_mesh, lattv);
      auto phase_kpq = compute_phase_at_points(kpq_cart, r_P, fft_mesh, lattv);

      // --- Read unperturbed wfc for gradient computation ---
      // Read wfc directly to ensure k2g uses the correct FFT mesh (not mf's native grid)
      std::string wfc_path;
      if (nspin == 1)
        wfc_path = mf->outdir() + "/" + mf->prefix() + ".save/wfc" + std::to_string(ik + 1) + ".hdf5";
      else if (is == 0)
        wfc_path = mf->outdir() + "/" + mf->prefix() + ".save/wfcup" + std::to_string(ik + 1) + ".hdf5";
      else
        wfc_path = mf->outdir() + "/" + mf->prefix() + ".save/wfcdn" + std::to_string(ik + 1) + ".hdf5";

      h5::file wfc_file(wfc_path, 'r');

      nda::array<int, 2> wfc_miller;
      nda::h5_read(wfc_file, "MillerIndices", wfc_miller);
      long wfc_npw = wfc_miller.shape(0);

      nda::array<int, 1> wfc_k2g(wfc_npw);
      utils::generate_k2g(wfc_miller, wfc_k2g, fft_mesh);

      nda::array<double, 2> wfc_evc_raw;
      nda::h5_read(wfc_file, "evc", wfc_evc_raw);
      long nbnd_wfc = wfc_evc_raw.shape(0);

      // --- Term 2: gradient of psi . delta_r_P ---
      for (long m = 0; m < std::min(nbnd, nbnd_wfc); ++m) {
        // Place PW coefficients on FFT grid
        nda::array<ComplexType, 1> c_on_grid(nnr);
        c_on_grid() = ComplexType(0.0);
        for (long ig = 0; ig < wfc_npw; ++ig) {
          double re = wfc_evc_raw(m, 2 * ig);
          double im = wfc_evc_raw(m, 2 * ig + 1);
          c_on_grid(wfc_k2g(ig)) = ComplexType{re, im};
        }

        for (int alpha = 0; alpha < 3; ++alpha) {
          // Copy and multiply by i*(k+G)_alpha
          grad_buf(0, nda::range::all) = c_on_grid;
          multiply_by_ikpG(grad_buf_4d, k_cart, recv, alpha);

          // IFFT
          F.backward(grad_buf_4d);

          // Sample at r_P, multiply by phase, accumulate with delta_r_P
          for (long u = 0; u < nP; ++u)
            result(is, ik, u, m) += grad_buf(0, r_P(u)) * phase_k(u) * delta_r_P(u, alpha);
        }
      }

      // --- Term 1: Deltapsi at r_P ---
      // Per-mode per-k files: {dpsi_prefix}_ik{ik+1}.hdf5
      // Each file has evc (nbnd_file, 2*npw) and MillerIndices (npw, 3)
      std::string dpsi_path = dpsi_prefix + "_ik" + std::to_string(ik + 1) + ".hdf5";

      h5::file dpsi_file(dpsi_path, 'r');

      // Read MillerIndices: (npw, 3)
      nda::array<int, 2> miller;
      nda::h5_read(dpsi_file, "MillerIndices", miller);
      long npw = miller.shape(0);

      // k2g mapping for deltapsi miller indices
      nda::array<int, 1> k2g(npw);
      utils::generate_k2g(miller, k2g, fft_mesh);

      // Read evc: (nbnd_file, 2*npw) as raw doubles
      nda::array<double, 2> evc_raw;
      nda::h5_read(dpsi_file, "evc", evc_raw);
      long nbnd_file = evc_raw.shape(0);

      for (long m = 0; m < std::min(nbnd, nbnd_file); ++m) {
        // Decode interleaved real/imag for this band
        fft_buf(0, nda::range::all) = ComplexType(0.0);
        for (long ig = 0; ig < npw; ++ig) {
          double re = evc_raw(m, 2 * ig);
          double im = evc_raw(m, 2 * ig + 1);
          fft_buf(0, k2g(ig)) = ComplexType{re, im};
        }

        // IFFT
        F.backward(fft_buf_4d);

        // Sample at r_P, multiply by (k+q) Bloch phase
        for (long u = 0; u < nP; ++u)
          result(is, ik, u, m) += fft_buf(0, r_P(u)) * phase_kpq(u);
      }
    }
  }

  return result;
}

} // namespace methods
