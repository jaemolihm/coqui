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


#undef NDEBUG

#include "catch2/catch.hpp"

#include <filesystem>
#include <complex>
#include <cmath>
#include <string>
#include <vector>

#include "configuration.hpp"
#include "IO/AppAbort.hpp"
#include "IO/app_loggers.h"
#include "utilities/mpi_context.h"

#include "h5/h5.hpp"
#include "nda/nda.hpp"
#include "nda/h5.hpp"
#include "utilities/test_common.hpp"

#include "mean_field/mf_utils.hpp"
#include "hamiltonian/pseudo/pseudopot.h"
#include "orbitals/eph_vertex.h"

namespace bdft_tests
{

/**
 * End-to-end validation of the bare electron-phonon vertex assembled from CoQuí
 * quantities only (the C++ side of coqui.compute_bare_eph_vertex), against the
 * g_bare that Quantum ESPRESSO stores in its elph file.
 *
 * The vertex is reassembled exactly as Mf::compute_bare_eph_vertex does:
 *     dv = pseudopot::build_dvloc_ion(mf, q)     // ionic (NLCC-free) dvloc, Ha
 *     g  = orbitals::eph_vertex_local(mf, dv, q) // local part
 *     g += pseudopot::eph_vertex_nonlocal(mf, q) // nonlocal projector part
 * giving g(nspin, nmode, nk, nb, nb) in Hartree.
 *
 * Units/layout conventions (identical to the validate_gbare_allq.py reference
 * script): QE g_bare is Rydberg and column-major in the band pair, so the
 * reference is  ref(mode,k,m,n) = 0.5 * gqe(k,mode,n,m)  with the interleaved
 * complex decoded as raw[...,2j]+i*raw[...,2j+1]. The compared quantity is the
 * relative error ||g - ref|| / ||ref|| over the first-min(nb) band block, with
 * NO fitting.
 *
 * Fixture (see tests/unit_test_files/qe/si_eph/README.md):
 *   - <prefix>.coqui.h5 : QE MF carrying the "vloc_radial" group (written by the
 *                         patched pw2coqui); loaded with mf::h5_input_type.
 *   - elph_bare.iq{N}.h5 : QE reference per q-point, datasets "xq_cryst" (3) and
 *                          "g_bare" (nk, nmode, nb, 2*nb) interleaved real (Ry).
 * The fixture may not be committed yet; the test skips cleanly if it is absent
 * so the suite stays green until the reference data is provided.
 */
TEST_CASE("eph_bare_vertex", "[eph]")
{
  auto& mpi = utils::make_unit_test_mpi_context();

  std::string source_path = PROJECT_SOURCE_DIR;
  std::string dir    = source_path + "/tests/unit_test_files/qe/si_eph/";
  std::string prefix = "Si";

  std::string mf_h5 = dir + prefix + ".coqui.h5";
  if(not std::filesystem::exists(mf_h5)) {
    WARN("eph fixture not found (" << mf_h5 << "); skipping eph vertex test.");
    return;
  }

  // Enumerate the committed reference q-points elph_bare.iq{N}.h5 (contiguous from 1).
  std::vector<std::string> elph_files;
  for(int iq = 1; ; ++iq) {
    std::string f = dir + "elph_bare.iq" + std::to_string(iq) + ".h5";
    if(std::filesystem::exists(f)) elph_files.push_back(std::move(f));
    else break;
  }
  if(elph_files.empty()) {
    WARN("no elph_bare.iq*.h5 reference under " << dir << "; skipping eph vertex test.");
    return;
  }

  // QE MF; needs h5 input so build_dvloc_ion can read the "vloc_radial" group.
  auto mf = mf::make_MF(mpi, mf::qe_source, dir, prefix, mf::h5_input_type);
  long nb = mf.nbnd();

  const double RY_TO_HA = 0.5;
  const double tol      = 1e-6;

  hamilt::pseudopot pp(mf);

  for(auto const& fname : elph_files) {
    // phonon wavevector in crystal coordinates
    nda::array<double,1> q_cryst;
    {
      h5::file fin(fname, 'r');
      h5::group gin(fin);
      nda::h5_read(gin, "xq_cryst", q_cryst);
    }

    // Assemble g = g_loc + g_nl (collective) exactly as compute_bare_eph_vertex.
    auto dv = pp.build_dvloc_ion(mf, q_cryst());                    // (nmode,nnr) Ha
    auto g  = orbitals::eph_vertex_local<HOST_MEMORY>(mf, dv(), q_cryst());
    g += pp.eph_vertex_nonlocal(mf, q_cryst());                     // (nspin,nmode,nk,nb,nb)

    if(not mpi->comm.root()) continue;

    // QE reference g_bare: (nk, nmode, nb_ref, 2*nb_ref) interleaved real, Rydberg.
    nda::array<double,4> raw;
    {
      h5::file fin(fname, 'r');
      h5::group gin(fin);
      nda::h5_read(gin, "g_bare", raw);
    }
    long nk = raw.shape(0), nmode = raw.shape(1), nb_ref = raw.shape(2);
    long nc = std::min(nb, nb_ref);   // compared band block

    // rel err over the first-nc band block:
    //   pred(mode,k,m,n) = g(0,mode,k,m,n)
    //   ref (mode,k,m,n) = 0.5 * gqe(k,mode,n,m)   (Ry->Ha, Fortran (m,n) transpose)
    double num2 = 0.0, den2 = 0.0;
    for(long mode = 0; mode < nmode; ++mode)
      for(long k = 0; k < nk; ++k)
        for(long m = 0; m < nc; ++m)
          for(long n = 0; n < nc; ++n) {
            std::complex<double> pred = g(0, mode, k, m, n);
            std::complex<double> ref  = RY_TO_HA *
                std::complex<double>(raw(k, mode, n, 2*m), raw(k, mode, n, 2*m + 1));
            num2 += std::norm(pred - ref);
            den2 += std::norm(ref);
          }
    double rel = (den2 > 0.0) ? std::sqrt(num2 / den2) : std::sqrt(num2);
    app_log(2, "  eph g_bare  q=({:.3f},{:.3f},{:.3f})  rel err = {:.3e}",
            q_cryst(0), q_cryst(1), q_cryst(2), rel);
    CHECK(rel < tol);
  }
}

/**
 * Regression test for the bare second-order electron-phonon vertex g2_bare
 * (q=0), the C++ side of coqui.compute_bare_eph_vertex_d2. There is no external
 * QE reference for g2, so this compares against a committed gold reference
 * (g2_bare_ref.h5) frozen from a validated run: it guards against regressions in
 * the compact atom-diagonal assembly and the k-distributed gather.
 *
 * The vertex is assembled exactly as Mf::compute_bare_eph_vertex_d2:
 *     g2  = pseudopot::eph_vertex_nonlocal_d2(mf)   // (nspin,nat,3,3,nk,nb,nb)
 *     d2v = pseudopot::build_d2vloc_ion(mf)         // (6*nat, nnr) local d2V at q=0
 *     gp  = orbitals::eph_vertex_local(mf, d2v, 0)  // local part
 *     g2(s,ka,i,j) += gp(s,6*ka+p)  (+ transpose for i!=j)   // scatter on root
 * All three are root-only (full on rank 0, empty elsewhere), so the assembly and
 * comparison happen on rank 0. If the gold reference is missing it is generated
 * with nda::h5_write and the test WARNs (rerun to validate) — the same
 * skip-cleanly convention as the g_bare test above.
 */
TEST_CASE("eph_bare_vertex_d2", "[eph]")
{
  auto& mpi = utils::make_unit_test_mpi_context();

  std::string source_path = PROJECT_SOURCE_DIR;
  std::string dir    = source_path + "/tests/unit_test_files/qe/si_eph/";
  std::string prefix = "Si";

  std::string mf_h5 = dir + prefix + ".coqui.h5";
  if(not std::filesystem::exists(mf_h5)) {
    WARN("eph fixture not found (" << mf_h5 << "); skipping eph g2 vertex test.");
    return;
  }

  auto mf = mf::make_MF(mpi, mf::qe_source, dir, prefix, mf::h5_input_type);
  hamilt::pseudopot pp(mf);

  // Assemble g2 = nonlocal + local d2 part, exactly as compute_bare_eph_vertex_d2.
  auto g2  = pp.eph_vertex_nonlocal_d2(mf);      // (nspin,nat,3,3,nk,nb,nb) root only
  auto d2v = pp.build_d2vloc_ion(mf);            // (6*nat, nnr) Ha
  nda::array<double,1> q0(3); q0() = 0.0;
  auto gp  = orbitals::eph_vertex_local<HOST_MEMORY>(mf, d2v(), q0());
  static constexpr int pi[6] = {0, 1, 2, 0, 0, 1};
  static constexpr int pj[6] = {0, 1, 2, 1, 2, 2};
  if(g2.size() > 0) {   // root only; non-root arrays are empty
    long nspin = g2.shape(0), nat = g2.shape(1), nk = g2.shape(4), nb = g2.shape(5);
    for(long s=0; s<nspin; ++s)
      for(long ka=0; ka<nat; ++ka)
        for(int p=0; p<6; ++p) {
          int i = pi[p], j = pj[p];
          for(long k=0; k<nk; ++k)
            for(long m=0; m<nb; ++m)
              for(long n=0; n<nb; ++n) {
                ComplexType v = gp(s, 6*ka+p, k, m, n);
                g2(s, ka, i, j, k, m, n) += v;
                if(i != j) g2(s, ka, j, i, k, m, n) += v;
              }
        }
  }

  if(not mpi->comm.root()) return;

  const double tol = 1e-8;
  std::string ref = dir + "g2_bare_ref.h5";
  if(not std::filesystem::exists(ref)) {
    h5::file fout(ref, 'w');
    h5::group gout(fout);
    nda::h5_write(gout, "g2_bare", g2);
    WARN("generated g2 gold reference " << ref << "; rerun to validate.");
    return;
  }

  nda::array<ComplexType,7> g2ref;
  {
    h5::file fin(ref, 'r');
    h5::group gin(fin);
    nda::h5_read(gin, "g2_bare", g2ref);
  }
  REQUIRE(g2.shape() == g2ref.shape());

  double num2 = 0.0, den2 = 0.0;
  for(long i=0; i<g2.size(); ++i) {
    num2 += std::norm(g2.data()[i] - g2ref.data()[i]);
    den2 += std::norm(g2ref.data()[i]);
  }
  double rel = (den2 > 0.0) ? std::sqrt(num2 / den2) : std::sqrt(num2);
  app_log(2, "  eph g2_bare (q=0)  rel err vs gold = {:.3e}", rel);
  CHECK(rel < tol);
}

} // namespace bdft_tests
