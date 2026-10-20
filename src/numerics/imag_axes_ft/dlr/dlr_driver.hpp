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

#ifndef COQUI_DLR_GENERATOR_HPP
#define COQUI_DLR_GENERATOR_HPP

#ifdef ENABLE_DLR

#include <string>
#include <map>
#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>

#include "nda/nda.hpp"
#include "nda/lapack.hpp"
#include "nda/lapack/gqr.hpp"

#include "numerics/nda_functions.hpp"
// FIXME hacky solution to updates nda namespace until cppdlr's nda dependency is updated. 
#include "numerics/imag_axes_ft/dlr/nda_linalg_compat.hpp"
// FIXME avoid including full cppdlr header until cppdlr's nda dependency is updated. 
//#include "cppdlr/cppdlr.hpp"
#include "cppdlr/dlr_build.hpp"
#include "cppdlr/dlr_imtime.hpp"
#include "cppdlr/dlr_imfreq.hpp"

#include "configuration.hpp"
#include "utilities/check.hpp"
#include "IO/app_loggers.h"
#include "../iaft_enum_e.hpp"

namespace imag_axes_ft {
  namespace dlr {

    struct DLR {
    public:
      DLR() = default;

      DLR(double beta_, double wmax_, std::string prec_ = "high", bool print_meta_log = false):
          beta(beta_), wmax(wmax_), prec(prec_) {
        eps = eps_from_prec(prec);
        init(print_meta_log);
      }

      DLR(double beta_, double wmax_, double eps_, bool print_meta_log = false):
          beta(beta_), wmax(wmax_), prec("custom"), eps(eps_) {
        utils::check(eps > 0.0,
          "imag_axes_ft::dlr::DLR: eps = {} must be positive.", eps);
        init(print_meta_log);
      }

      static double eps_from_prec(std::string const &prec) {
        if (prec == "high") {
          return 1e-13;
        } else if (prec == "medium") {
          return 1e-10;
        } else if (prec == "low") {
          return 1e-06;
        }

        utils::check(false,
          "imag_axes_ft::dlr::DLR: prec = {} is not acceptable. Acceptable list = \"high\", \"medium\", \"low\"",
          prec);
        return 0.0;
      }

      DLR(const DLR &other) = default;
      DLR(DLR &&other) = default;

      DLR &operator=(const DLR &other) = default;
      DLR &operator=(DLR &&other) = default;

      ~DLR() = default;

      void metadata_log(int level = 1) const {
        app_log(level, "  Mesh details on the imaginary axis");
        app_log(level, "  ----------------------------------");
        app_log(level, "  Discrete Lehmann Representation");
        app_log(level, "  Beta                   = {} a.u.", beta);
        app_log(level, "  Frequency cutoff       = {} a.u.", wmax);
        app_log(level, "  Lambda                 = {}", lambda);
        if (prec == "custom") {
          app_log(level, "  Precision              = {}", eps);
        } else {
          app_log(level, "  Precision              = {} (i.e. eps = {})", prec, eps);
        }
        app_log(level, "  nt_f, nt_b, nw_f, nw_b = {}, {}, {}, {}\n", nt_f, nt_b, nw_f, nw_b);
      }


    // T(x, :) interpolates the values on the DLR tau nodes to tau = beta (x + 1) / 2, x in [-1, 1].
    // Each row is solved separately (see solve_cf2it_row) and averaged with the row of the mirror point,
    //   T(x, t) = (r(x, t) + r(-x, nt - 1 - t)) / 2,   r(x, :) = cf2eval(x) cf2it^{-1},
    // so that T(-x, t) = T(x, nt - 1 - t) bitwise, as for the exact matrix (cppdlr SYM nodes: tau_{nt-1-t} = beta - tau_t).
    auto construct_tau_interpolate_matrix(
      const nda::MemoryArrayOfRank<1> auto &tau_mesh_out,
      bool ph_sym=false) const
    -> nda::array<ComplexType, 2> {
      auto nt_out = tau_mesh_out.size();
      auto nt_in = _it_ops.rank();
      auto eye_nt_in = nda::eye<ComplexType>(nt_in);
      auto zlu = nda::matrix<ComplexType>(_it_ops.get_it2cf_zlu());
      auto piv = nda::array<int, 1>(_it_ops.get_it2cf_piv());

      auto solve_tau_row = [&](double x) {
        auto cf2eval = _it_ops.coefs2eval(eye_nt_in, cppdlr::abs2rel((x + 1.0) * 0.5)); // [-1, 1] -> [0, 1]
        return solve_cf2it_row(zlu, piv, nda::vector<ComplexType>(cf2eval));
      };

      auto T_full = nda::array<ComplexType, 2>(nt_out, nt_in);
      for (int it_out = 0; it_out < nt_out; ++it_out) {
        double x = tau_mesh_out(it_out);
        auto r_x = solve_tau_row(x);
        auto r_mx = solve_tau_row(-x);
        for (int it = 0; it < nt_in; ++it) T_full(it_out, it) = 0.5 * (r_x(it) + r_mx(nt_in - 1 - it));
      }

      if (not ph_sym) return T_full;

      // only keep the first half of the tau points for particle-hole symmetry.
      auto nt_half = nt_in / 2 + nt_in % 2;
      auto T_tt = nda::array<ComplexType, 2>(nt_out, nt_half);
      for (int it_out = 0; it_out < nt_out; ++it_out) {
        for (int it_in = 0; it_in < nt_half; ++it_in) {
          int imt = nt_in - it_in - 1;
          T_tt(it_out, it_in) = (it_in == imt)? T_full(it_out, it_in) : T_full(it_out, it_in) + T_full(it_out, imt);
        }
      }
      return T_tt;
    }

    auto construct_w_interpolate_matrix(
       const nda::MemoryArrayOfRank<1> auto &wn_mesh_out,
       imag_axes_ft::stats_e stats,
       bool ph_sym=false) const
    -> nda::array<ComplexType, 2> {

      auto const &if_ops = (stats == imag_axes_ft::fermion) ? _if_ops_f : _if_ops_b;
      auto nw_in    = static_cast<int>(if_ops.get_ifnodes().size());
      auto nw_out   = static_cast<int>(wn_mesh_out.size());

      // rhs = B_out * Ttw, with B_out = beta * cf2if(wn_out) * cf2it^{-1}: tau_to_w at the output points,
      // composed with the paired w_to_tau. B_out is built by build_tau_to_w_rows, as Twt, so that
      // B_out(wn_mesh) = Twt.
      auto const &Ttw = (stats == imag_axes_ft::fermion) ? Ttw_ff : Ttw_bb;
      std::vector<int> n_dlr(nw_out);
      for (int iw_out = 0; iw_out < nw_out; ++iw_out) {
        long n_ir = wn_mesh_out(iw_out);
        if (stats == imag_axes_ft::fermion) {
          utils::check((n_ir % 2) != 0,
                       "dlr_driver.hpp::DLR::construct_w_interpolate_matrix: "
                       "fermionic wn_mesh_out must be odd in IR notation, got {}.", n_ir);
        } else {
          utils::check((n_ir % 2) == 0,
                       "dlr_driver.hpp::DLR::construct_w_interpolate_matrix: "
                       "bosonic wn_mesh_out must be even in IR notation, got {}.", n_ir);
        }
        // IR notation -> cppdlr notation: fermion n_ir = 2*n_dlr + 1; boson n_ir = 2*n_dlr
        n_dlr[iw_out] = static_cast<int>((stats == imag_axes_ft::fermion) ? (n_ir - 1) / 2 : n_ir / 2);
      }
      auto B_out = build_tau_to_w_rows(if_ops, n_dlr, stats);
      auto rhs = nda::matrix<ComplexType>(B_out * Ttw);

      if (not ph_sym) 
        return rhs;

      utils::check(stats == imag_axes_ft::boson, 
        "dlr_driver.hpp::DLR::construct_w_interpolate_matrix: "
        "ph_sym=true is only supported for bosons.");
      // Only keep the positive-frequency half of the input (including w=0),
      // matching wn_mesh[nw/2:] ordering in the Python ph-sym path.
      auto nw_half = nw_in / 2 + nw_in % 2;
      auto iw_zero = nw_in / 2;
      nda::array<ComplexType, 2> T_ww(nw_out, nw_half);
      for (int iw_pos = 0; iw_pos < nw_half; ++iw_pos) {
        int ipw = iw_zero + iw_pos;
        int imw = iw_zero - iw_pos;
        if (ipw == imw) {
          T_ww(nda::range::all, iw_pos) = rhs(nda::range::all, ipw);
        } else {
          for (int iw_out = 0; iw_out < nw_out; ++iw_out) {
            T_ww(iw_out, iw_pos) = rhs(iw_out, ipw) + rhs(iw_out, imw);
          }
        }
      }
      return T_ww;
    }

    auto construct_tau_to_zero_and_beta_matrices() 
    -> std::pair<nda::array<ComplexType, 1>, nda::array<ComplexType, 1>> {
      nda::array<double, 1> tau_mesh_0beta = {-1.0, 1.0}; // tau = 0, beta
      auto T_tt = construct_tau_interpolate_matrix(tau_mesh_0beta);
      return {T_tt(0, nda::range::all), T_tt(1, nda::range::all)};
    }

    private:
      // x^T cf2it^{-1} for one row x (cf2it^T y = x, nrhs = 1). getrs rounds differently for different nrhs, so a
      // row solved together with other rows depends on them; a single-row solve depends only on x.
      static nda::vector<ComplexType> solve_cf2it_row(nda::matrix<ComplexType> const &zlu, nda::array<int, 1> const &piv,
                                                      nda::vector<ComplexType> const &x) {
        auto y = nda::matrix<ComplexType, nda::F_layout>(x.size(), 1);
        y(nda::range::all, 0) = x;
        auto info = nda::lapack::getrs(nda::transpose(zlu), y, piv);
        utils::check(info == 0, "dlr_driver.hpp::DLR: Linear solve failed for cf2it^T y = x (getrs info = {}).", info);
        return nda::vector<ComplexType>(y(nda::range::all, 0));
      }

      // Enforce, per row (tau_axis = 1: M(n, t)) or per column (tau_axis = 0: M(t, n)),
      //   M(t) = s conj M(nt - 1 - t),   s = +1 boson, -1 fermion,
      // by M(t) <- (M(t) + s conj M(nt - 1 - t)) / 2 and M(nt-1-t) <- s conj M(t) (bitwise s conj of the same sum).
      // The exact tau -> iw matrix has it: cppdlr SYM gives poles omega_j = -omega_{r-1-j} and nodes
      // tau_{nt-1-t} = beta - tau_t, with kernels K(beta - tau, omega) = K(tau, -omega); the LU solve with cf2it
      // (cond ~1e13) does not keep it (~3e-4 relative). An entry with t = nt - 1 - t becomes Re (boson) or i Im (fermion).
      static void project_tau_reflection(nda::matrix<ComplexType> &M, int tau_axis, imag_axes_ft::stats_e stats) {
        double s = (stats == imag_axes_ft::fermion) ? -1.0 : 1.0;
        long nt = M.extent(tau_axis);
        long nother = M.extent(1 - tau_axis);
        auto at = [&](long t, long k) -> ComplexType & { return tau_axis == 1 ? M(k, t) : M(t, k); };
        for (long k = 0; k < nother; ++k) {
          for (long t = 0; t <= (nt - 1) / 2; ++t) {
            long mt = nt - 1 - t;
            ComplexType a = 0.5 * (at(t, k) + s * std::conj(at(mt, k)));
            at(t, k) = a;
            at(mt, k) = s * std::conj(a);
          }
        }
      }

      // Rows B(n, :) = beta * cf2if(n, :) * cf2it^{-1} of the tau -> iw matrix at the cppdlr frequency indices n_dlr,
      // solved one row at a time (solve_cf2it_row) and reflection-projected (project_tau_reflection(., 1)), so that
      // a row depends only on its own n and B(-n, :) = conj B(n, :) bitwise.
      nda::matrix<ComplexType> build_tau_to_w_rows(cppdlr::imfreq_ops const &if_ops, std::vector<int> const &n_dlr,
                                                   imag_axes_ft::stats_e stats) const {
        auto r = _it_ops.rank();
        auto eye_r = nda::eye<ComplexType>(r);
        auto zlu = nda::matrix<ComplexType>(_it_ops.get_it2cf_zlu());
        auto piv = nda::array<int, 1>(_it_ops.get_it2cf_piv());
        auto B = nda::matrix<ComplexType>(n_dlr.size(), r);
        for (long i = 0; i < long(n_dlr.size()); ++i) {
          auto cf2if = nda::vector<ComplexType>(if_ops.coefs2eval(eye_r, n_dlr[i]));
          B(i, nda::range::all) = beta * solve_cf2it_row(zlu, piv, cf2if);
        }
        project_tau_reflection(B, 1, stats);
        return B;
      }

      // Ttw = pinv(Twt), with the exact pairing Ttw(:, -wn) = conj(Ttw(:, wn)) and the exact reflection
      // Ttw(nt - 1 - t, :) = s conj Ttw(t, :) (project_tau_reflection(., 0)).
      // cppdlr::build_if2it is not used: it inverts the complex cf2if (cond ~1e14), whose rounding
      // breaks the wn <-> -wn conjugation pairing and leaves Ttw an inaccurate inverse of Twt.
      // Twt = beta * cf2if * cf2it^{-1} inverts only the real cf2it, so Twt(-wn, :) = conj(Twt(wn, :))
      // holds bitwise and pinv(Twt) = cf2it * cf2if^+ / beta is the same operator in exact arithmetic.
      // The reflection of the projected Twt carries over to its pinv up to rounding of the SVD solve,
      // which the projection makes exact.
      static nda::matrix<ComplexType> paired_left_inverse(nda::matrix<ComplexType> const &B,
                                                          nda::array<long, 1> const &wn_mesh,
                                                          imag_axes_ft::stats_e stats) {
        long nw = B.extent(0);
        long nt = B.extent(1);
        auto Bf = nda::matrix<ComplexType, nda::F_layout>(B);
        auto rhs = nda::matrix<ComplexType, nda::F_layout>(nw, nw);
        rhs() = 0.0;
        for (long i = 0; i < nw; ++i) rhs(i, i) = 1.0;
        auto s = nda::vector<double>(nt);
        int rank = 0;
        nda::lapack::gelss(Bf, rhs, s, -1.0, rank);
        utils::check(rank == nt, "imag_axes_ft::dlr::DLR: Twt is rank deficient, rank = {}, nt = {}.", rank, nt);
        auto T = nda::matrix<ComplexType>(rhs(nda::range(nt), nda::range::all));

        // mirror(i): index of -wn_mesh(i)
        std::map<long, long> index_of;
        for (long i = 0; i < nw; ++i) index_of[wn_mesh(i)] = i;
        std::vector<long> mirror(nw);
        for (long i = 0; i < nw; ++i) {
          auto it = index_of.find(-wn_mesh(i));
          utils::check(it != index_of.end(), "imag_axes_ft::dlr::DLR: mirror frequency of wn = {} is not in the mesh.", wn_mesh(i));
          mirror[i] = it->second;
        }

        // Pair the columns: T(:, -wn) = conj(T(:, wn)), and T(:, wn = 0) real, so that w_to_tau maps
        // X(-iwn) = conj(X(iwn)) to an exactly real X(tau). The gelss result already satisfies this up to
        // rounding, ~eps * cond(Twt) relative (the complex SVD rounds rows wn and -wn independently):
        // 1e-14 at Lambda = 600, 1e-10 at Lambda = 5e5. The averaging only makes it exact. A deviation
        // above 1e-8 means a wrong Twt or mesh, not rounding.
        double T_max = nda::max_element(nda::abs(T));
        double pair_err = 0.0;
        for (long i = 0; i < nw; ++i)
          for (long t = 0; t < nt; ++t)
            pair_err = std::max(pair_err, std::abs(T(t, i) - std::conj(T(t, mirror[i]))));
        utils::check(pair_err <= 1e-8 * T_max,
                     "imag_axes_ft::dlr::DLR: pinv(Twt) violates T(:, -wn) = conj(T(:, wn)) by {} (relative), > 1e-8.",
                     pair_err / T_max);
        for (long i = 0; i < nw; ++i) {
          long m = mirror[i];
          if (m < i) continue;
          if (m == i) {
            // wn = 0
            for (long t = 0; t < nt; ++t) T(t, i) = std::real(T(t, i));
          } else {
            for (long t = 0; t < nt; ++t) {
              ComplexType a = 0.5 * (T(t, i) + std::conj(T(t, m)));
              T(t, i) = a;
              T(t, m) = std::conj(a);
            }
          }
        }

        // The projection acts per column and conj distributes over its average, so the pairing is kept.
        // A reflection defect above 1e-8 means a wrong Twt or mesh, not rounding.
        double refl_err = 0.0;
        double sgn = (stats == imag_axes_ft::fermion) ? -1.0 : 1.0;
        for (long i = 0; i < nw; ++i)
          for (long t = 0; t < nt; ++t)
            refl_err = std::max(refl_err, std::abs(T(t, i) - sgn * std::conj(T(nt - 1 - t, i))));
        utils::check(refl_err <= 1e-8 * T_max,
                     "imag_axes_ft::dlr::DLR: pinv(Twt) violates T(nt-1-t, :) = s conj(T(t, :)) by {} (relative), > 1e-8.",
                     refl_err / T_max);
        project_tau_reflection(T, 0, stats);
        return T;
      }


      // Tlt (r x r): DLR tau-node values g -> coefficients a_l = int_0^beta u_l(tau) g(tau) dtau of the modes u_l
      // that diagonalize the Lehmann kernel inside the DLR span (the DLR counterpart of the IR Tct, same ordering by
      // singular value). With X = W^{1/2} E_f = Q R (E_f: node values -> values on a fine tau grid, W: fine tau
      // quadrature weights) and Kt = W^{1/2} K(tau_f, omega_f) V^{1/2} (V: fine omega quadrature weights),
      // Q^T Kt = U S Vt gives Tlt = U^T R (u_l = Q U_l).
      // The DLR Tct (= it2cf) gives pole weights, which are not ordered by singular value, so their tail does not
      // measure truncation; the IR Tct is used as is.
      // E_f goes through an LU solve with cf2it (cond ~1e13); as a value -> value map it is accurate (error ~ eps
      // against an analytic model G; dlr_leakage_modes checks the orthonormality of the modes built from it).
      // Do not replace it with B_f * Tct (explicit inverse) without re-measuring.
      void build_Tlt() {
        auto r = _it_ops.rank();
        cppdlr::fineparams fine(lambda);

        // fine tau grid: relative format, sqrt of quadrature weights
        auto [t_rel, sqw_t] = cppdlr::build_it_fine(fine);
        long nt_fine = t_rel.size();
        nda::array<double, 1> tau_fine(nt_fine); // [-1, 1] notation of construct_tau_interpolate_matrix
        for (long i = 0; i < nt_fine; ++i) tau_fine(i) = 2.0 * cppdlr::rel2abs(t_rel(i)) - 1.0;

        // fine omega grid: Gauss-Legendre rule on the panels of cppdlr::build_rf_fine, mirrored about 0
        auto [x_gl, w_gl] = cppdlr::gaussquad(fine.p);
        long nom_fine = 2 * fine.npom * fine.p;
        nda::vector<double> om_fine(nom_fine);
        nda::vector<double> sqw_om(nom_fine);
        double a = 0.0;
        for (int ip = 0; ip < fine.npom; ++ip) {
          double b = fine.lambda / std::pow(2.0, fine.npom - ip - 1);
          for (int k = 0; k < fine.p; ++k) {
            double om = a + 0.5 * (b - a) * (x_gl(k) + 1.0);
            double sw = std::sqrt(0.5 * (b - a) * w_gl(k));
            long ipos = fine.npom * fine.p + ip * fine.p + k;
            long ineg = fine.npom * fine.p - 1 - (ip * fine.p + k);
            om_fine(ipos) = om;
            om_fine(ineg) = -om;
            sqw_om(ipos) = sw;
            sqw_om(ineg) = sw;
          }
          a = b;
        }

        // X = W^{1/2} E_f = Q R
        auto E_f = construct_tau_interpolate_matrix(tau_fine);
        auto X = nda::matrix<double, nda::F_layout>(nt_fine, r);
        for (long i = 0; i < nt_fine; ++i)
          for (long j = 0; j < r; ++j) X(i, j) = sqw_t(i) * std::real(E_f(i, j));
        nda::vector<double> tau_qr(r);
        nda::lapack::geqrf(X, tau_qr);
        auto R = nda::matrix<double, nda::F_layout>(r, r);
        R() = 0.0;
        double rmax = 0.0, rmin = std::numeric_limits<double>::max();
        for (long i = 0; i < r; ++i) {
          for (long j = i; j < r; ++j) R(i, j) = X(i, j);
          rmax = std::max(rmax, std::abs(X(i, i)));
          rmin = std::min(rmin, std::abs(X(i, i)));
        }
        utils::check(rmin > 1e-10 * rmax,
                     "imag_axes_ft::dlr::DLR: interpolated DLR basis is rank deficient on the fine tau grid "
                     "(lambda = {}, eps = {}, rank = {}, min|R_ii|/max|R_ii| = {}).", lambda, eps, r, rmin / rmax);
        nda::lapack::gqr(X, tau_qr); // X <- Q

        // M = Q^T Kt = U S Vt
        auto Kt = cppdlr::build_k_it(nda::vector_const_view<double>(t_rel), nda::vector_const_view<double>(sqw_t),
                                     nda::vector_const_view<double>(om_fine));
        for (long j = 0; j < nom_fine; ++j) Kt(nda::range::all, j) *= sqw_om(j);
        auto Kt_f = nda::matrix<double, nda::F_layout>(Kt);
        auto M = nda::matrix<double, nda::F_layout>(nda::transpose(X) * Kt_f);
        nda::vector<double> sv(r);
        auto U = nda::matrix<double, nda::F_layout>(r, r);
        auto Vt = nda::matrix<double, nda::F_layout>(nom_fine, nom_fine);
        nda::lapack::gesvd(M, sv, U, Vt);

        // Tlt = U^T R, in the IR normalization int_0^beta u_l u_m dtau = delta_lm
        auto T_real = nda::matrix<double>(nda::transpose(U) * R);
        T_real *= std::sqrt(beta);
        Tlt = nda::matrix<ComplexType>(T_real);
      }

      void init(bool print_meta_log) {
        lambda = beta * wmax;
        utils::check(lambda > 0.0, "Invalid value of lambda, {}, in imag_axes_ft::dlr::DLR.", lambda);
        utils::check(beta > 0.0, "Invalid value of beta, {}, in imag_axes_ft::dlr::DLR.", beta);
        utils::check(wmax > 0.0, "Invalid value of wmax, {}, in imag_axes_ft::dlr::DLR.", wmax);
        utils::check(eps > 0.0, "Invalid value of eps, {}, in imag_axes_ft::dlr::DLR.", eps);

        // Construct DLR basis by obtaining DLR real frequencies 
        auto dlr_rf = cppdlr::build_dlr_rf(lambda, eps, cppdlr::SYM);
        auto dlr_rank = dlr_rf.size();

        // get DLR imaginary time object
        _it_ops = cppdlr::imtime_ops(lambda, dlr_rf, cppdlr::SYM);
        utils::check(dlr_rank == _it_ops.rank(), 
        "imag_axes_ft::dlr::DLR: size of DLR real frequency basis, {}, does not match the rank of DLR imaginary time object, {}", 
        dlr_rank, _it_ops.rank());
        // dlr imagianry time interpolation nodes. 
        // TODO keep only one set of nt and tau_mesh for both fermions and bosons
        nt_f = _it_ops.rank();
        nt_b = _it_ops.rank();
        tau_mesh_f = nda::array<double, 1>(nt_f);
        tau_mesh_b = nda::array<double, 1>(nt_b);
        // The native DLR tau nodes live in [0, 1], rather than [0, beta] nor [-1, 1]
        // Convert to the IR basis notation which lives in [-1, 1] instead...
        for (int i = 0; i < nt_f; ++i) {
          tau_mesh_f(i) = 2.0 * cppdlr::rel2abs(_it_ops.get_itnodes(i)) - 1.0;
        }
        tau_mesh_b() = tau_mesh_f;

        // get DLR imaginary frequency objects
        _if_ops_f = cppdlr::imfreq_ops(lambda, dlr_rf, cppdlr::Fermion, cppdlr::SYM);
        _if_ops_b = cppdlr::imfreq_ops(lambda, dlr_rf, cppdlr::Boson, cppdlr::SYM);

        nw_f = _if_ops_f.get_ifnodes().size();
        nw_b = _if_ops_b.get_ifnodes().size();
        wn_mesh_f = nda::array<long, 1>(nw_f);
        wn_mesh_b = nda::array<long, 1>(nw_b);
        // Convert to IR basis notation
        for (int i = 0; i < nw_f; ++i) wn_mesh_f(i) = long(2.0 * _if_ops_f.get_ifnodes(i) + 1);
        for (int i = 0; i < nw_b; ++i) wn_mesh_b(i) = long(2.0 * _if_ops_b.get_ifnodes(i));

        // The reflections enforced below need mirror-symmetric meshes: wn(i) = -wn(nw-1-i), tau(i) = -tau(nt-1-i)
        for (int i = 0; i < nw_f; ++i)
          utils::check(wn_mesh_f(i) == -wn_mesh_f(nw_f - 1 - i), "imag_axes_ft::dlr::DLR: fermionic wn mesh is not mirror symmetric.");
        for (int i = 0; i < nw_b; ++i)
          utils::check(wn_mesh_b(i) == -wn_mesh_b(nw_b - 1 - i), "imag_axes_ft::dlr::DLR: bosonic wn mesh is not mirror symmetric.");
        for (int i = 0; i < nt_f; ++i)
          utils::check(std::abs(tau_mesh_f(i) + tau_mesh_f(nt_f - 1 - i)) <= 1e-12,
                       "imag_axes_ft::dlr::DLR: tau mesh is not mirror symmetric.");

        // imaginary time to Matsubara frequency
        std::vector<int> n_dlr_f(nw_f), n_dlr_b(nw_b);
        for (int i = 0; i < nw_f; ++i) n_dlr_f[i] = _if_ops_f.get_ifnodes(i);
        for (int i = 0; i < nw_b; ++i) n_dlr_b[i] = _if_ops_b.get_ifnodes(i);
        Twt_ff = build_tau_to_w_rows(_if_ops_f, n_dlr_f, imag_axes_ft::fermion);
        Twt_bb = build_tau_to_w_rows(_if_ops_b, n_dlr_b, imag_axes_ft::boson);
        // Matsubara frequency to imaginary time
        Ttw_ff = paired_left_inverse(Twt_ff, wn_mesh_f, imag_axes_ft::fermion);
        Ttw_bb = paired_left_inverse(Twt_bb, wn_mesh_b, imag_axes_ft::boson);

        // fermion <-> boson tau interpolation (same DLR tau nodes)
        Ttt_bf = nda::eye<ComplexType>(nt_b);
        Ttt_fb = nda::eye<ComplexType>(nt_f); 

        // tau -> coefficients
        auto it2cf  = _it_ops.vals2coefs(nda::eye<ComplexType>(nt_f));
        Tct_ff = it2cf;
        Tct_bb = it2cf;
        
        std::tie(T_zero_t_ff, T_beta_t_ff) = construct_tau_to_zero_and_beta_matrices();
        build_Tlt();
        if (print_meta_log) metadata_log();
      }

    public:
      double beta = 0.0;
      double wmax = 0.0;
      std::string prec = "high";
      double eps = 0.0;
      double lambda = 0.0;

      cppdlr::imtime_ops _it_ops;
      cppdlr::imfreq_ops _if_ops_f;
      cppdlr::imfreq_ops _if_ops_b;

      int nt_f = 0;
      int nt_b = 0;
      int nw_f = 0;
      int nw_b = 0;

      // Matsubara frequency mesh
      nda::array<long, 1> wn_mesh_f;
      nda::array<long, 1> wn_mesh_b;
      // tau mesh: relative format (same convention as IR backend)
      nda::array<double, 1> tau_mesh_f;
      // TODO tau_mesh_b as an array view of tau_mesh_f since they are the same. 
      nda::array<double, 1> tau_mesh_b;

      // transformation from w_f to t_f
      nda::matrix<ComplexType> Ttw_ff;
      // transformation from t_f to w_f
      nda::matrix<ComplexType> Twt_ff;
      // transformation from w_b to t_b
      nda::matrix<ComplexType> Ttw_bb;
      // transformation from t_b to w_b
      nda::matrix<ComplexType> Twt_bb;
      // interpolation from t_f to t_b
      nda::matrix<ComplexType> Ttt_bf;
      // interpolation from t_b to t_f
      nda::matrix<ComplexType> Ttt_fb;
      // interpolation matrix to t_f = beta^{-}
      nda::array<ComplexType, 1> T_beta_t_ff;
      // interpolation matrix to t_f = 0^{+}
      nda::array<ComplexType, 1> T_zero_t_ff;
      // tau to DLR coefficients
      nda::matrix<ComplexType> Tct_ff;
      nda::matrix<ComplexType> Tct_bb;
      // tau-node values to kernel-SVD mode coefficients (leakage diagnostic), same for fermions and bosons
      nda::matrix<ComplexType> Tlt;
    };
  } // dlr
} // imag_axes_ft

#endif // ENABLE_DLR
#endif // COQUI_DLR_GENERATOR_HPP
