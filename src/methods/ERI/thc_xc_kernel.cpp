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

#include "methods/ERI/thc_xc_kernel.hpp"

#include <algorithm>
#include <cmath>
#include <exception>
#include <filesystem>
#include <functional>
#include <string>

#include "IO/app_loggers.h"
#include "utilities/check.hpp"
#include "itertools/itertools.hpp"
#include "nda/blas.hpp"
#include "nda/h5.hpp"
#include "h5/h5.hpp"
#include "numerics/fft/nda.hpp"

namespace methods {

namespace detail {

// Ry -> Ha. Every xc-kernel coefficient QE dumps is one second derivative of
// E_xc, so each of A, B, C carries exactly one factor of this.
//
// TODO: the unit conversion belongs on the QE side (the elph.x writer, as
// pw2coqui already does for the quantities it exports), so that everything
// CoQuí reads is in Hartree and this file makes no assumption about the
// producer's unit system.
constexpr double Ry2Ha = 0.5;

template<typename T>
T read_xc_attribute(h5::file const& f, std::string const& name,
                    std::string const& fname) {
  T x{};
  try {
    h5::h5_read_attribute(f, name, x);
  } catch (std::exception const&) {
    APP_ABORT("read_xc_kernel_fields: attribute '{}' not found in {}. The file "
              "must be written by elph.x with write_xc_kernel = .true.",
              name, fname);
  }
  return x;
}

// Read one rank-1 REAL(DP) dense-grid dataset and relayout Fortran (x fastest)
// -> C (z fastest). QE writes index i1 + i2*M1 + i3*M1*M2; CoQuí's FFT grids
// use (i1*M2 + i2)*M3 + i3.
void read_xc_field(h5::group const& grp, std::string const& name,
                   nda::stack_array<long, 3> const& mesh,
                   nda::array<double, 1>& out) {
  long const M1 = mesh(0), M2 = mesh(1), M3 = mesh(2);
  long const nnr = M1 * M2 * M3;
  nda::array<double, 1> raw;
  utils::check(grp.has_dataset(name),
               "read_xc_kernel_fields: dataset '{}' not found.", name);
  nda::h5_read(grp, name, raw);
  utils::check(raw.shape(0) == nnr,
               "read_xc_kernel_fields: dataset '{}' has length {}, expected {} "
               "= nr1x_dense*nr2x_dense*nr3x_dense.", name, raw.shape(0), nnr);
  out.resize(nnr);
  for (long i1 = 0; i1 < M1; ++i1)
    for (long i2 = 0; i2 < M2; ++i2)
      for (long i3 = 0; i3 < M3; ++i3)
        out((i1 * M2 + i2) * M3 + i3) = raw(i1 + i2 * M1 + i3 * M1 * M2);
}

} // namespace detail

xc_kernel_fields_t read_xc_kernel_fields(std::string const& fname,
                                        mpi3::communicator& comm) {
  xc_kernel_fields_t xc;

  // Checked on every rank rather than inside the root-only block below: the h5
  // open happens on the root alone, so a throw there leaves every other rank
  // blocked in the broadcasts that follow and the run hangs instead of failing.
  utils::check(std::filesystem::exists(fname),
               "read_xc_kernel_fields: file '{}' does not exist. It must be "
               "written by elph.x with write_xc_kernel = .true.", fname);

  int is_gradient = 0;
  nda::array<long, 1> mesh_h(3);

  // Size the kernel fields from the grid metadata: on the root inside the read
  // below, on every other rank once the metadata has been broadcast.
  auto size_fields = [&]() {
    for (int i = 0; i < 3; ++i) xc.mesh(i) = mesh_h(i);
    xc.is_gradient = (is_gradient != 0);
    long const nnr = xc.nnr();
    xc.A = nda::array<double, 1>(nnr);
    if (xc.is_gradient) {
      xc.B = nda::array<double, 2>(3, nnr);
      xc.C = nda::array<double, 3>(3, 3, nnr);
    }
  };

  // The h5 work below is root-only, so a C++ exception escaping it would leave every
  // other rank blocked in the broadcasts that follow. Convert it to a global abort.
  if (comm.root()) try {
    h5::file file(fname, 'r');
    h5::group grp(file);

    long const ngrid_dense = detail::read_xc_attribute<int>(file, "ngrid_dense", fname);
    mesh_h(0) = detail::read_xc_attribute<int>(file, "nr1x_dense", fname);
    mesh_h(1) = detail::read_xc_attribute<int>(file, "nr2x_dense", fname);
    mesh_h(2) = detail::read_xc_attribute<int>(file, "nr3x_dense", fname);
    is_gradient = detail::read_xc_attribute<int>(file, "xc_is_gradient", fname);
    utils::check(mesh_h(0) * mesh_h(1) * mesh_h(2) == ngrid_dense,
                 "read_xc_kernel_fields: ngrid_dense = {} != {}*{}*{} in {}.",
                 ngrid_dense, mesh_h(0), mesh_h(1), mesh_h(2), fname);

    // The kernel lives on QE's dense (dfftp) grid, the wavefunctions on the
    // smooth (dffts) one. Only the doublegrid == false case (the two grids
    // identical) has been checked end to end; refuse the other rather than
    // producing a silently mismatched Vxc.
    long const nr1x = detail::read_xc_attribute<int>(file, "nr1x", fname);
    long const nr2x = detail::read_xc_attribute<int>(file, "nr2x", fname);
    long const nr3x = detail::read_xc_attribute<int>(file, "nr3x", fname);
    utils::check(nr1x == mesh_h(0) and nr2x == mesh_h(1) and nr3x == mesh_h(2),
                 "read_xc_kernel_fields: {} was written with different dense "
                 "({},{},{}) and smooth ({},{},{}) FFT grids (QE doublegrid). "
                 "This configuration is untested; rerun with ecutrho = 4*ecutwfc.",
                 fname, mesh_h(0), mesh_h(1), mesh_h(2), nr1x, nr2x, nr3x);

    size_fields();
    long const nnr = xc.nnr();

    nda::array<double, 1> dmuxc;
    detail::read_xc_field(grp, "dmuxc", xc.mesh, dmuxc);
    xc.A() = dmuxc;

    if (xc.is_gradient) {
      nda::array<double, 1> dvxc_rr, dvxc_sr, dvxc_ss, dvxc_s;
      detail::read_xc_field(grp, "dvxc_rr", xc.mesh, dvxc_rr);
      detail::read_xc_field(grp, "dvxc_sr", xc.mesh, dvxc_sr);
      detail::read_xc_field(grp, "dvxc_ss", xc.mesh, dvxc_ss);
      detail::read_xc_field(grp, "dvxc_s",  xc.mesh, dvxc_s);
      nda::array<double, 2> grho(3, nnr);
      for (int a = 0; a < 3; ++a) {
        nda::array<double, 1> g_a;
        detail::read_xc_field(grp, "grho_" + std::to_string(a + 1), xc.mesh, g_a);
        grho(a, nda::range::all) = g_a;
      }

      // A = dmuxc + dvxc_rr, B_a = dvxc_sr*grho_a,
      // C_ab = dvxc_s delta_ab + dvxc_ss grho_a grho_b  (dgradcorr.f90, nspin=1)
      for (long ir = 0; ir < nnr; ++ir) {
        xc.A(ir) += dvxc_rr(ir);
        for (int a = 0; a < 3; ++a) {
          xc.B(a, ir) = dvxc_sr(ir) * grho(a, ir);
          for (int b = 0; b < 3; ++b)
            xc.C(a, b, ir) = (a == b ? dvxc_s(ir) : 0.0)
                           + dvxc_ss(ir) * grho(a, ir) * grho(b, ir);
        }
      }
    }

    xc.A() *= detail::Ry2Ha;
    if (xc.is_gradient) {
      xc.B() *= detail::Ry2Ha;
      xc.C() *= detail::Ry2Ha;
    }
  } catch (std::exception const& e) {
    APP_ABORT("read_xc_kernel_fields: failed to read '{}': {}", fname, e.what());
  }

  comm.broadcast_n(mesh_h.data(), 3, 0);
  comm.broadcast_n(&is_gradient, 1, 0);
  if (!comm.root()) size_fields();

  comm.broadcast_n(xc.A.data(), xc.A.size(), 0);
  if (xc.is_gradient) {
    comm.broadcast_n(xc.B.data(), xc.B.size(), 0);
    comm.broadcast_n(xc.C.data(), xc.C.size(), 0);
  }

  app_log(2, "  xc kernel fields read from {}", fname);
  app_log(2, "    - dense FFT mesh   = ({}, {}, {})", xc.mesh(0), xc.mesh(1), xc.mesh(2));
  app_log(2, "    - gradient-corrected = {}", xc.is_gradient);

  return xc;
}

namespace detail {

// (G - q) in Cartesian coordinates and the position of each truncated-grid G on
// the kernel's (finer or equal) FFT mesh. Errors out if any G of rho_g cannot be
// represented on that mesh.
struct g_map_t {
  nda::array<long, 1> idx;      // (nG) linear index on the kernel mesh, C order
  nda::array<double, 2> kG;     // (3, nG) G - q, Cartesian
};

g_map_t build_g_map(grids::truncated_g_grid const& rho_g,
                    nda::stack_array<long, 3> const& mesh,
                    nda::stack_array<double, 3> const& q_cart) {
  long const nG = rho_g.size();
  auto src = rho_g.mesh();
  long const n[3] = {src(0), src(1), src(2)};
  long const m[3] = {mesh(0), mesh(1), mesh(2)};

  g_map_t map;
  map.idx.resize(nG);
  map.kG.resize(3, nG);
  auto gv = rho_g.g_vectors();

  for (long g = 0; g < nG; ++g) {
    long N = rho_g.gv_to_fft(g);
    long src_i[3];
    src_i[2] = N % n[2];
    src_i[1] = (N / n[2]) % n[1];
    src_i[0] = N / (n[2] * n[1]);
    long dst = 0;
    for (int a = 0; a < 3; ++a) {
      long mi = (src_i[a] > n[a] / 2) ? src_i[a] - n[a] : src_i[a];
      utils::check(mi <= m[a] / 2 and mi > m[a] / 2 - m[a],
                   "thc_xc_kernel: G-vector with Miller index {} along axis {} does "
                   "not fit the xc-kernel FFT mesh ({},{},{}). The kernel grid must "
                   "contain the THC density grid; lower the THC ecut or dump the "
                   "kernel on a denser grid.", mi, a, m[0], m[1], m[2]);
      long di = (mi < 0) ? mi + m[a] : mi;
      dst = dst * m[a] + di;
    }
    map.idx(g) = dst;
    for (int a = 0; a < 3; ++a) map.kG(a, g) = gv(g, a) - q_cart(a);
  }
  return map;
}

} // namespace detail

nda::array<ComplexType, 2> compute_Vxc_for_q(
    xc_kernel_fields_t const& xc,
    grids::truncated_g_grid const& rho_g,
    nda::array<ComplexType, 2> const& zeta_uG,
    nda::stack_array<double, 3> const& q_cart,
    double volume,
    long block_size,
    mpi3::communicator& comm) {

  long const Np = zeta_uG.shape(0);
  long const nG = rho_g.size();
  utils::check(zeta_uG.shape(1) == nG,
               "compute_Vxc_for_q: zeta_uG has {} G-vectors, expected {}.",
               zeta_uG.shape(1), nG);
  utils::check(block_size > 0, "compute_Vxc_for_q: block_size must be > 0.");

  long const M1 = xc.mesh(0), M2 = xc.mesh(1), M3 = xc.mesh(2);
  long const Nd = M1 * M2 * M3;
  // 1 component (zeta) for LDA, 4 (zeta and its three covariant derivatives)
  // for a gradient-corrected kernel.
  long const nc = xc.is_gradient ? 4 : 1;
  long const B  = std::min(block_size, Np);

  auto gmap = detail::build_g_map(rho_g, xc.mesh, q_cart);

  // Batched inverse FFT over (component, pivot). Row (a*B + i) holds component a
  // of the i-th pivot of the current block.
  nda::array<ComplexType, 4> Lbuf(nc * B, M1, M2, M3);
  nda::array<ComplexType, 4> Rbuf(nc * B, M1, M2, M3);
  math::nda::fft<true> F_L(Lbuf, math::fft::FFT_ESTIMATE | math::fft::FFT_PRESERVE_INPUT);
  math::nda::fft<true> F_R(Rbuf, math::fft::FFT_ESTIMATE | math::fft::FFT_PRESERVE_INPUT);

  // zeta_u(r) and (D zeta_u)(r) on the kernel mesh, from the truncated G-sphere.
  // Zero-padding the sphere into a finer mesh is exact: the interpolating
  // vectors are band-limited to it. D -> i(G - q) (see thc_xc_kernel.hpp).
  auto build_block = [&](nda::range rng, nda::array<ComplexType, 4>& buf,
                         math::nda::fft<true>& F) {
    buf() = ComplexType(0.0);
    auto flat = nda::array_view<ComplexType, 2>({nc * B, Nd}, buf.data());
    for (auto [i, u] : itertools::enumerate(rng)) {
      for (long g = 0; g < nG; ++g) {
        ComplexType const z = zeta_uG(u, g);
        long const ir = gmap.idx(g);
        flat(i, ir) = z;
        for (long a = 1; a < nc; ++a)
          flat(a * B + i, ir) = ComplexType(0.0, gmap.kG(a - 1, g)) * z;
      }
    }
    F.backward(buf);
  };

  // R <- M(r) R, with M the real symmetric 4x4 kernel matrix
  //   M = [[A, B_b], [B_a, C_ab]].
  auto apply_kernel = [&](long nr) {
    auto flat = nda::array_view<ComplexType, 2>({nc * B, Nd}, Rbuf.data());
    for (long i = 0; i < nr; ++i) {
      if (nc == 1) {
        for (long ir = 0; ir < Nd; ++ir) flat(i, ir) *= xc.A(ir);
        continue;
      }
      for (long ir = 0; ir < Nd; ++ir) {
        ComplexType const z0 = flat(i, ir);
        ComplexType const z1 = flat(B + i, ir);
        ComplexType const z2 = flat(2 * B + i, ir);
        ComplexType const z3 = flat(3 * B + i, ir);
        flat(i, ir) = xc.A(ir) * z0
                    + xc.B(0, ir) * z1 + xc.B(1, ir) * z2 + xc.B(2, ir) * z3;
        for (int a = 0; a < 3; ++a)
          flat((a + 1) * B + i, ir) = xc.B(a, ir) * z0
                                    + xc.C(a, 0, ir) * z1
                                    + xc.C(a, 1, ir) * z2
                                    + xc.C(a, 2, ir) * z3;
      }
    }
  };

  nda::array<ComplexType, 2> Vxc(Np, Np);
  Vxc() = ComplexType(0.0);

  long const nblk = (Np + B - 1) / B;
  // Hermiticity halves the work: only blocks (ib, jb >= ib) are built and the
  // lower triangle is filled by conjugate transposition below. Row blocks are
  // spread round-robin over comm and all-reduced at the end.
  //
  // The hermiticity residual is measured against the norms of the four
  // per-component partial products, not against the norm of their sum: kernels
  // whose terms cancel (a uniform B gives dV_xc = -(div B) drho = 0) would
  // otherwise be judged by a 0/0 ratio, while a sign error between the two
  // cross terms shows up as an anti-hermitian result of the same size as the
  // terms that were supposed to cancel.
  double herm_err = 0.0, term_nrm = 0.0, blk_nrm = 0.0;
  nda::array<ComplexType, 2> tmp(B, B), part(B, B);

  // Row block ib builds nblk-ib column blocks, so a plain ib % size round robin
  // loads rank 0 with ~nblk blocks and the last rank with ~1. The rows are
  // folded into pairs {ib, nblk-1-ib} of nearly equal total cost, and it is
  // those ceil(nblk/2) units that are handed round-robin to the ranks — so the
  // work parallelizes over ceil(nblk/2) ranks, not nblk.
  long const nunits = (nblk + 1) / 2;
  auto owner = [nblk, np = long(comm.size())](long ib) {
    long const unit = std::min(ib, nblk - 1 - ib);
    return unit % np;
  };
  if (nunits < long(comm.size()))
    app_log(3, "    compute_Vxc_for_q: {} row blocks fold into {} work units over {} "
               "ranks; {} ranks idle. Lower Vxc_block_size to use them.",
            nblk, nunits, comm.size(), long(comm.size()) - nunits);

  for (long ib = 0; ib < nblk; ++ib) {
    if (owner(ib) != comm.rank()) continue;
    nda::range rrng(ib * B, std::min((ib + 1) * B, Np));
    long const nr = rrng.size();
    build_block(rrng, Lbuf, F_L);

    for (long jb = ib; jb < nblk; ++jb) {
      nda::range crng(jb * B, std::min((jb + 1) * B, Np));
      long const ncol = crng.size();
      build_block(crng, Rbuf, F_R);
      apply_kernel(ncol);

      auto blk = tmp(nda::range(nr), nda::range(ncol));
      auto prt = part(nda::range(nr), nda::range(ncol));
      blk() = ComplexType(0.0);
      for (long a = 0; a < nc; ++a) {
        auto La = nda::array_view<ComplexType, 2>({nr, Nd}, Lbuf.data() + a * B * Nd);
        auto Ra = nda::array_view<ComplexType, 2>({ncol, Nd}, Rbuf.data() + a * B * Nd);
        nda::blas::gemm(ComplexType(1.0), La, nda::dagger(Ra), ComplexType(0.0), prt);
        blk() += prt;
        if (jb == ib)
          for (long i = 0; i < nr; ++i)
            for (long j = 0; j < ncol; ++j) term_nrm += std::norm(prt(i, j));
      }
      for (long i = 0; i < nr; ++i)
        for (long j = 0; j < ncol; ++j)
          Vxc(rrng.first() + i, crng.first() + j) = blk(i, j);

      // Diagonal blocks are computed in full, so they carry a free hermiticity
      // check on the four-term integrand (A, B, C are real).
      if (jb == ib) {
        for (long i = 0; i < nr; ++i)
          for (long j = 0; j < nr; ++j) {
            ComplexType const d = blk(i, j) - std::conj(blk(j, i));
            herm_err += std::norm(d);
            blk_nrm  += std::norm(blk(i, j));
          }
      }
    }
  }

  comm.all_reduce_in_place_n(Vxc.data(), Vxc.size(), std::plus<>{});
  comm.all_reduce_in_place_n(&herm_err, 1, std::plus<>{});
  comm.all_reduce_in_place_n(&term_nrm, 1, std::plus<>{});
  comm.all_reduce_in_place_n(&blk_nrm,  1, std::plus<>{});

  // Diagnostics on the diagonal blocks, all pre-scaling and over the same
  // entries so the ratios are apples-to-apples. ||Vxc||/||terms|| is O(1) for a
  // generic kernel and at roundoff for one that cancels analytically (e.g. a
  // uniform B, where dV_xc = -(div B) drho = 0).
  double const herm_rel = std::sqrt(herm_err / std::max(term_nrm, 1e-300));
  app_log(3, "    compute_Vxc_for_q: ||Vxc||/||terms|| = {:.3e}, "
             "||Vxc - Vxc^H||/||terms|| = {:.3e}",
          std::sqrt(blk_nrm / std::max(term_nrm, 1e-300)), herm_rel);
  utils::check(herm_rel < 1e-8,
               "compute_Vxc_for_q: Vxc is not hermitian on the diagonal blocks, "
               "||Vxc - Vxc^H|| / ||per-component terms|| = {:.3e} (expected <= 1e-8).",
               herm_rel);

  // Integral measure: dr -> volume/Nd, and the interpolating vectors carry a
  // 1/volume each relative to the stored zeta_u(G) (same normalization that
  // makes the Coulomb matrix (1/volume) sum_G zeta_u v conj(zeta_v)).
  Vxc() *= ComplexType(1.0 / (volume * double(Nd)));

  // Fill the lower triangle.
  for (long u = 0; u < Np; ++u)
    for (long v = u + 1; v < Np; ++v)
      Vxc(v, u) = std::conj(Vxc(u, v));

  // At q = Gamma the covariant derivative reduces to i*G and zeta^Gamma is real
  // (Z^Gamma_{ur} = sum_k |T^k_{ur}|^2), so Vxc(Gamma) must be real symmetric.
  // Measured against the per-component term norms for the same reason as the
  // hermiticity residual above.
  double const q2 = q_cart(0) * q_cart(0) + q_cart(1) * q_cart(1) + q_cart(2) * q_cart(2);
  if (q2 < 1e-12) {
    // Restricted to the diagonal blocks, the only entries term_nrm accumulates
    // (see the jb == ib guard above); summing the full matrix against a
    // diagonal-block denominator would inflate the ratio by nblk.
    double im2 = 0.0;
    for (long ib = 0; ib < nblk; ++ib) {
      long const u0 = ib * B, u1 = std::min((ib + 1) * B, Np);
      for (long u = u0; u < u1; ++u)
        for (long v = u0; v < u1; ++v) {
          double const x = Vxc(u, v).imag();
          im2 += x * x;
        }
    }
    // term_nrm is pre-scaling, Vxc post-scaling; put them on the same footing.
    double const scale = std::sqrt(term_nrm) / (volume * double(Nd));
    double const im_rel = std::sqrt(im2) / std::max(scale, 1e-300);
    app_log(3, "    compute_Vxc_for_q: ||Im Vxc(Gamma)|| / ||terms|| = {:.3e}", im_rel);
    utils::check(im_rel < 1e-8,
                 "compute_Vxc_for_q: Vxc at q = Gamma is not real, "
                 "||Im Vxc|| / ||per-component terms|| = {:.3e}. The iq terms of "
                 "D = grad - iq did not drop out.", im_rel);
  }

  return Vxc;
}

} // namespace methods
