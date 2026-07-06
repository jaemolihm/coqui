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

#include "lr_thc_interp.hpp"

#include "configuration.hpp"
#include "utilities/check.hpp"
#include "utilities/kpoint_utils.hpp"
#include "utilities/lr_utils.hpp"
#include "numerics/fft/nda.hpp"
#include "numerics/shared_array/nda.hpp"
#include "h5/h5.hpp"
#include "nda/h5.hpp"
#include "nda/nda.hpp"
#include "itertools/itertools.hpp"
#include "lr_psi_io.hpp"

namespace methods {

nda::array<ComplexType, 4> compute_delta_X(
    mf::MF const& mf,
    std::string const& Deltapsi_prefix,
    nda::array<long, 1> const& r_P,
    nda::array<double, 2> const& delta_r_P,
    nda::array<double, 1> const& q_vec_cryst,
    nda::array<long, 1> const& fft_grid)
{
  utils::check(mf.npol() == 1, "compute_delta_X: Non-collinear not yet implemented");
  utils::check(fft_grid.shape(0) == 3,
               "compute_delta_X: fft_grid must have size 3, got {}", fft_grid.shape(0));
  utils::check(mf.nkpts_ibz() == mf.nkpts(),
               "compute_delta_X: requires full-BZ grid (nkpts_ibz={} != nkpts={}); "
               "matches the assumption of the IBC consumer in lr_thc_comm.",
               mf.nkpts_ibz(), mf.nkpts());

  long nspin = mf.nspin();
  long nk_ibz = mf.nkpts_ibz();
  long nbnd = mf.nbnd();
  long nP = r_P.shape(0);
  auto recv = mf.recv();
  auto kpts_crys = mf.kpts_crystal();

  nda::stack_array<long, 3> fft_mesh;
  for (int i = 0; i < 3; ++i) fft_mesh(i) = fft_grid(i);
  long nnr = fft_mesh(0) * fft_mesh(1) * fft_mesh(2);

  utils::check(delta_r_P.shape(0) == nP && delta_r_P.shape(1) == 3,
               "compute_delta_X: delta_r_P must have shape (nP, 3), got ({}, {})",
               delta_r_P.shape(0), delta_r_P.shape(1));
  utils::check(q_vec_cryst.shape(0) == 3,
               "compute_delta_X: q_vec_cryst must have size 3");

  // One shared-memory window per node; ranks write disjoint (is, ik) slices.
  auto mpi_ctx = mf.mpi();
  auto& comm = mpi_ctx->comm;
  using local_Array_4D_t = nda::array<ComplexType, 4>;
  auto sDeltaX_skPm = math::shm::make_shared_array<local_Array_4D_t>(
      *mpi_ctx, std::array<long, 4>{nspin, nk_ibz, nP, nbnd});
  auto DeltaX_skPm = sDeltaX_skPm.local();

  nda::array<ComplexType, 2> fft_buf(1, nnr);
  auto fft_buf_4d = nda::reshape(fft_buf, std::array<long, 4>{1, fft_mesh(0), fft_mesh(1), fft_mesh(2)});

  nda::array<ComplexType, 2> grad_buf(1, nnr);
  auto grad_buf_4d = nda::reshape(grad_buf, std::array<long, 4>{1, fft_mesh(0), fft_mesh(1), fft_mesh(2)});

  math::nda::fft<true> F(fft_buf_4d);

  long const nsk = nspin * nk_ibz;
  auto [sk_lo, sk_hi] = itertools::chunk_range(0L, nsk, comm.size(), comm.rank());

  for (long sk = sk_lo; sk < sk_hi; ++sk) {
    long const is = sk / nk_ibz;
    long const ik = sk % nk_ibz;

    nda::stack_array<double, 3> k_cart, k_cryst, kpq_cryst;
    for (int a = 0; a < 3; ++a) {
      k_cryst(a)   = kpts_crys(ik, a);
      kpq_cryst(a) = k_cryst(a) + q_vec_cryst(a);
    }
    for (int a = 0; a < 3; ++a) {
      k_cart(a)    = k_cryst(0) * recv(0, a) + k_cryst(1) * recv(1, a) + k_cryst(2) * recv(2, a);
    }

    nda::array<ComplexType, 1> phase_k(nP);
    nda::array<ComplexType, 1> phase_kpq(nP);
    utils::rspace_phase_factor(k_cryst,   fft_mesh, r_P, phase_k);
    utils::rspace_phase_factor(kpq_cryst, fft_mesh, r_P, phase_kpq);

    // Term 2: gradient of unperturbed ψ · delta_r_P
    auto wfc = detail::read_wfc_k(mf, is, ik, fft_mesh);
    for (long m = 0; m < std::min(nbnd, wfc.nbnd); ++m) {
      nda::array<ComplexType, 1> c_on_grid(nnr);
      c_on_grid() = ComplexType(0.0);
      detail::decode_band_to_fft_grid(wfc.evc_raw, m, wfc.k2g, c_on_grid);

      for (int alpha = 0; alpha < 3; ++alpha) {
        grad_buf(0, nda::range::all) = c_on_grid;
        detail::multiply_by_ikpG(grad_buf_4d, k_cart, recv, alpha);
        F.backward(grad_buf_4d);
        for (long u = 0; u < nP; ++u)
          DeltaX_skPm(is, ik, u, m) += grad_buf(0, r_P(u)) * phase_k(u) * delta_r_P(u, alpha);
      }
    }

    // Term 1: Δψ at r_P
    auto Deltapsi = detail::read_Deltapsi_k(Deltapsi_prefix, ik, fft_mesh);
    for (long m = 0; m < std::min(nbnd, Deltapsi.nbnd); ++m) {
      fft_buf(0, nda::range::all) = ComplexType(0.0);
      auto fft_slice_1d = fft_buf(0, nda::range::all);
      detail::decode_band_to_fft_grid(Deltapsi.evc_raw, m, Deltapsi.k2g, fft_slice_1d);
      F.backward(fft_buf_4d);
      for (long u = 0; u < nP; ++u)
        DeltaX_skPm(is, ik, u, m) += fft_buf(0, r_P(u)) * phase_kpq(u);
    }
  }

  sDeltaX_skPm.all_reduce();

  // c2py boundary: rank 0 returns the data, other ranks return empty (→ None in Python).
  nda::array<ComplexType, 4> out;
  if (comm.root()) {
    out = DeltaX_skPm;
  }
  return out;
}


nda::array<ComplexType, 4> compute_delta_X_adj(
    mf::MF const& mf,
    std::string const& Deltapsi_adj_prefix,
    nda::array<long, 1> const& r_P,
    nda::array<double, 2> const& delta_r_P,
    nda::array<double, 1> const& q_vec_cryst,
    nda::array<long, 1> const& fft_grid)
{
  utils::check(mf.npol() == 1, "compute_delta_X_adj: Non-collinear not yet implemented");
  utils::check(fft_grid.shape(0) == 3,
               "compute_delta_X_adj: fft_grid must have size 3, got {}", fft_grid.shape(0));
  utils::check(mf.nkpts_ibz() == mf.nkpts(),
               "compute_delta_X_adj: requires full-BZ grid (nkpts_ibz={} != nkpts={}); "
               "matches the assumption of the IBC consumer in lr_thc_comm.",
               mf.nkpts_ibz(), mf.nkpts());

  long nspin = mf.nspin();
  long nk_ibz = mf.nkpts_ibz();
  long nbnd = mf.nbnd();
  long nP = r_P.shape(0);
  auto recv = mf.recv();
  auto kpts_crys = mf.kpts_crystal();

  nda::stack_array<long, 3> fft_mesh;
  for (int i = 0; i < 3; ++i) fft_mesh(i) = fft_grid(i);
  long nnr = fft_mesh(0) * fft_mesh(1) * fft_mesh(2);

  utils::check(delta_r_P.shape(0) == nP && delta_r_P.shape(1) == 3,
               "compute_delta_X_adj: delta_r_P must have shape (nP, 3), got ({}, {})",
               delta_r_P.shape(0), delta_r_P.shape(1));
  utils::check(q_vec_cryst.shape(0) == 3,
               "compute_delta_X_adj: q_vec_cryst must have size 3");

  nda::array<int, 1> kpq_map(nk_ibz);
  utils::calculate_kpq_map(kpts_crys, q_vec_cryst, kpq_map);

  auto mpi_ctx = mf.mpi();
  auto& comm = mpi_ctx->comm;
  using local_Array_4D_t = nda::array<ComplexType, 4>;
  auto sDeltaX_skPm = math::shm::make_shared_array<local_Array_4D_t>(
      *mpi_ctx, std::array<long, 4>{nspin, nk_ibz, nP, nbnd});
  auto DeltaX_skPm = sDeltaX_skPm.local();

  nda::array<ComplexType, 2> fft_buf(1, nnr);
  auto fft_buf_4d = nda::reshape(fft_buf, std::array<long, 4>{1, fft_mesh(0), fft_mesh(1), fft_mesh(2)});

  nda::array<ComplexType, 2> grad_buf(1, nnr);
  auto grad_buf_4d = nda::reshape(grad_buf, std::array<long, 4>{1, fft_mesh(0), fft_mesh(1), fft_mesh(2)});

  math::nda::fft<true> F(fft_buf_4d);

  long const nsk = nspin * nk_ibz;
  auto [sk_lo, sk_hi] = itertools::chunk_range(0L, nsk, comm.size(), comm.rank());

  for (long sk = sk_lo; sk < sk_hi; ++sk) {
    long const is = sk / nk_ibz;
    long const ik = sk % nk_ibz;
    int const ik_pq = kpq_map(ik);

    nda::stack_array<double, 3> k_cart_pq, k_cryst, kpq_cryst;
    for (int a = 0; a < 3; ++a) {
      k_cryst(a)   = kpts_crys(ik, a);
      kpq_cryst(a) = k_cryst(a) + q_vec_cryst(a);
    }
    for (int a = 0; a < 3; ++a) {
      k_cart_pq(a) = kpq_cryst(0) * recv(0, a) + kpq_cryst(1) * recv(1, a) + kpq_cryst(2) * recv(2, a);
    }

    // phase_k   → term 1 ([δ^q]† ψ_{k+q} has k Bloch character)
    // phase_kpq → term 2 (∇ψ_{k+q} has k+q character)
    nda::array<ComplexType, 1> phase_k(nP);
    nda::array<ComplexType, 1> phase_kpq(nP);
    utils::rspace_phase_factor(k_cryst,   fft_mesh, r_P, phase_k);
    utils::rspace_phase_factor(kpq_cryst, fft_mesh, r_P, phase_kpq);

    // Term 2: gradient of unperturbed ψ_{m, k+q} · delta_r_P
    auto wfc = detail::read_wfc_k(mf, is, ik_pq, fft_mesh);
    for (long m = 0; m < std::min(nbnd, wfc.nbnd); ++m) {
      nda::array<ComplexType, 1> c_on_grid(nnr);
      c_on_grid() = ComplexType(0.0);
      detail::decode_band_to_fft_grid(wfc.evc_raw, m, wfc.k2g, c_on_grid);

      for (int alpha = 0; alpha < 3; ++alpha) {
        grad_buf(0, nda::range::all) = c_on_grid;
        detail::multiply_by_ikpG(grad_buf_4d, k_cart_pq, recv, alpha);
        F.backward(grad_buf_4d);
        for (long u = 0; u < nP; ++u)
          DeltaX_skPm(is, ik, u, m) += grad_buf(0, r_P(u)) * phase_kpq(u) * delta_r_P(u, alpha);
      }
    }

    // Term 1: [δ^q]† ψ_{m, k+q} at r_P (adjoint Sternheimer)
    auto Deltapsi = detail::read_Deltapsi_k(Deltapsi_adj_prefix, ik, fft_mesh);
    for (long m = 0; m < std::min(nbnd, Deltapsi.nbnd); ++m) {
      fft_buf(0, nda::range::all) = ComplexType(0.0);
      auto fft_slice_1d = fft_buf(0, nda::range::all);
      detail::decode_band_to_fft_grid(Deltapsi.evc_raw, m, Deltapsi.k2g, fft_slice_1d);
      F.backward(fft_buf_4d);
      for (long u = 0; u < nP; ++u)
        DeltaX_skPm(is, ik, u, m) += fft_buf(0, r_P(u)) * phase_k(u);
    }
  }


  sDeltaX_skPm.all_reduce();

  nda::array<ComplexType, 4> out;
  if (comm.root()) {
    out = DeltaX_skPm;
  }
  return out;
}

} // namespace methods
