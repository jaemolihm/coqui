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


#include <algorithm>
#include <cstdlib>
#include <limits>
#include <optional>

#include "scf_common.hpp"
#include "hamiltonian/one_body_hamiltonian.hpp"
#include "mean_field/MF.hpp"
#include "mean_field/mf_utils.hpp"
#include "utilities/mpi_context.h"
#include "utilities/element_partition.hpp"
#include "utilities/h5_flat_slice.hpp"
#include "methods/tools/chkpt_utils.h"
#include "simple_dyson.h"
#include "numerics/iter_scf/commutator_residual.hpp"

namespace methods {

double compute_Nelec(double mu, const nda::array<ComplexType, 4> &spectra,
                     const mf::MF &mf, const imag_axes_ft::IAFT &FT) {
  auto [_nw, _ns, nkpts, _nbnd] = spectra.shape();
  nda::array<ComplexType, 2> Xw(_nw, _ns);
  nda::array<ComplexType, 2> Xt(FT.nt_f(), _ns);
  nda::array<ComplexType, 1> nelecs(_ns);
  auto k_weight = mf.k_weight();
  double scl = (_ns == 1 and mf.npol() == 1 ? -2.0 : -1.0); 

  for (size_t n = 0; n < _nw; ++n) {
    long wn = FT.wn_mesh()(n);
    ComplexType omega_mu = FT.omega(wn) + mu;
    for (size_t is = 0; is < _ns; ++is) {
      for (size_t ik = 0; ik < nkpts; ++ik) {
        for (size_t ib = 0; ib < _nbnd; ++ib) {
          Xw(n, is) += k_weight(ik) / (omega_mu - spectra(n, is, ik, ib));
        }
      }
    }
  }

  FT.w_to_tau(Xw, Xt, imag_axes_ft::fermion);
  FT.tau_to_beta(Xt, nelecs);

  ComplexType nelec = scl*std::accumulate(nelecs.begin(),nelecs.end(),ComplexType(0.0));
  if (std::abs(nelec.imag()) / mf.nelec() >= 1e3*FT.eps()) {
    app_log(1, "[WARNING] nelec.imag()/nelec_target = {}", nelec.imag() / mf.nelec());
  }

  return nelec.real();
}

template<typename X_t, nda::ArrayOfRank<1> Array1D>
auto eval_hf_energy(const X_t &sDm_skij, const X_t &sF_skij, const X_t &sH0_skij,
                    Array1D &k_weight, bool F_has_H0)
  -> std::tuple<double, double> {
  auto [ns, nkpts, nbnd, nbnd2] = sDm_skij.shape();
  // HF energy = Tr[Dm*H0] + 0.5*Tr[Dm*F] + e_nuc
  nda::matrix<ComplexType> buffer(nbnd, nbnd);
  ComplexType e_1e(0.0, 0.0);
  ComplexType e_hf(0.0, 0.0);
  for (size_t is = 0; is < ns; ++is) {
    for (size_t ik = 0; ik < nkpts; ++ik) {
      nda::matrix_const_view<ComplexType> Dm_ij =
          sDm_skij.local()(is, ik, nda::ellipsis{});
      nda::matrix_const_view<ComplexType> F_ij =
          sF_skij.local()(is, ik, nda::ellipsis{});
      nda::matrix_const_view<ComplexType> H0_ij =
          sH0_skij.local()(is, ik, nda::ellipsis{});

      buffer = Dm_ij * H0_ij;
      auto diag_H0 = nda::diagonal(buffer);
      e_1e += k_weight(ik) * nda::sum(diag_H0);

      buffer = (F_has_H0)? 0.5 * Dm_ij * (F_ij - H0_ij) : 0.5 * Dm_ij * F_ij;
      auto diag_F = nda::diagonal(buffer);
      e_hf += k_weight(ik) * nda::sum(diag_F);
    }
  }
  // MAM: need to know npol here, scale only when npol==1 and ns==1
  RealType spin_factor = (ns == 2) ? 1.0 : 2.0;
  e_1e *= spin_factor;
  e_hf *= spin_factor;
  // TODO CNY: _MF->e_nuc() is missing
  if (e_1e.imag() / e_1e.real() >= 1e-8) {
    app_log(1, "[WARNING] e_1e.imag()/e_1e.real() = {}, e_1e.imag() = {}, e_1e.real() = {}",
            e_1e.imag()/e_1e.real(), e_1e.imag(), e_1e.real());
  }
  if (e_hf.imag() / e_hf.real() >= 1e-8) {
    app_log(1, "[WARNING] e_hf.imag()/e_hf.real() = {}, e_hf.imag() = {}, e_hf.real() = {}",
            e_hf.imag()/e_hf.real(), e_hf.imag(), e_hf.real());
  }
  return std::make_tuple(e_1e.real(), e_hf.real());
}

template<typename comm_t, typename X_t, nda::ArrayOfRank<1> Array1D>
double eval_corr_energy(comm_t& comm, const imag_axes_ft::IAFT &FT,
                        const X_t & G_shm, const X_t & Sigma_shm,
                        Array1D &k_weight) {
  decltype(nda::range::all) all;
  int nw = FT.nw_f();
  auto [nts, ns, nkpts, nbnd, nbnd2] = G_shm.shape();
  nda::array<ComplexType, 2> SigmaG_ws(nw, ns);
  nda::array<ComplexType, 4> Sigma_tski(nts, ns, nkpts, nbnd);
  nda::array<ComplexType, 4> G_tski(nts, ns, nkpts, nbnd);
  nda::array<ComplexType, 4> Sigma_wski(nw, ns, nkpts, nbnd);
  nda::array<ComplexType, 4> G_wski(nw, ns, nkpts, nbnd);
  auto SigmaG_ws_1D =
      nda::reshape(SigmaG_ws, std::array<long, 1>{nw * ns});
  auto Sigma_w_3D = nda::reshape(
      Sigma_wski, std::array<long, 3>{nw * ns, nkpts, nbnd});
  auto G_w_3D = nda::reshape(
      G_wski, std::array<long, 3>{nw * ns, nkpts, nbnd});

  int size = comm.size();
  int rank = comm.rank();
  comm.barrier();
  for (size_t i = rank; i < nbnd; i += size) {
    Sigma_tski = Sigma_shm.local()(all, all, all, i, all);
    G_tski = G_shm.local()(all, all, all, all, i);
    FT.tau_to_w(Sigma_tski, Sigma_wski, imag_axes_ft::fermion);
    FT.tau_to_w(G_tski, G_wski, imag_axes_ft::fermion);
    for (size_t ws = 0; ws < nw * ns; ++ws) {
      for (size_t ik = 0; ik < nkpts; ++ik ) {
        SigmaG_ws_1D(ws) += k_weight(ik) * nda::blas::dot(Sigma_w_3D(ws, ik, all), G_w_3D(ws, ik, all));
      }
    }
  }
  comm.all_reduce_in_place_n(SigmaG_ws.data(), SigmaG_ws.size(),
                             std::plus<>{});

  nda::array<ComplexType, 2> SigmaG_ts(nts, ns);
  nda::array<ComplexType, 1> SigmaG_beta_s(ns);
  FT.w_to_tau(SigmaG_ws, SigmaG_ts, imag_axes_ft::fermion);
  FT.tau_to_beta(SigmaG_ts, SigmaG_beta_s);

  // MAM: need to know npol here, scale only when npol==1 and ns==1
  RealType spin_factor = (ns == 2) ? 1.0 : 2.0;
  ComplexType e_corr = (-0.5 * spin_factor) * nda::sum(SigmaG_beta_s);
  if (e_corr.imag() / e_corr.real() >= 1e2*FT.eps()) {
    app_log(1, "[WARNING] e_corr.imag()/e_corr.real() = {}, e_corr.imag() = {}, e_corr.real() = {}",
            e_corr.imag()/e_corr.real(), e_corr.imag(), e_corr.real());
  }
  return e_corr.real();
}

template<typename comm_t, typename dyson_type, typename X_t, typename Xt_t>
auto eval_thermodynamic_properties(comm_t& comm, dyson_type &dyson, 
                                   const X_t &sF_skij, const Xt_t &sSigma_tskij, 
                                   const std::vector<double> &elec_energies, double Phi_dynamical, 
                                   double mu, bool F_has_H0) 
  -> thermodynamics_t {
  decltype(nda::range::all) all;

  auto MF = dyson.MF();
  auto FT = dyson.FT();

  auto npol = MF->npol();
  auto k_weight = MF->k_weight();

  auto beta = FT->beta();
  auto nt = FT->nt_f();
  auto nw = FT->nw_f();
  auto [ns, nkpts_ibz, nbnd, nbnd2] = sF_skij.shape();

  RealType spin_factor = (npol == 1 and ns == 1) ? 2.0 : 1.0;

  ComplexType Phi = elec_energies[1] + Phi_dynamical;
  ComplexType tr_Sigma_G = 2 * (elec_energies[1] + elec_energies[2]);

  int size = comm.size();
  int rank = comm.rank();

  // Evaluate Tr ln(-G0) in the (F, S) eigenbasis.
  ComplexType tr_ln_G0(0.0, 0.0);
  if (rank < ns * nkpts_ibz) {

    nda::matrix<ComplexType> F(nbnd, nbnd);
    nda::matrix<ComplexType> S_ij(nbnd, nbnd);
    for (size_t sk = rank; sk < ns * nkpts_ibz; sk += size) {
      size_t is = sk / nkpts_ibz;
      size_t ik = sk % nkpts_ibz;
      if (F_has_H0) {
        F = sF_skij.local()(is, ik, all, all);
      } else {
        F = sF_skij.local()(is, ik, all, all) + dyson.sH0_skij().local()(is, ik, all, all);
      }
      S_ij = dyson.sS_skij().local()(is, ik, all, all);
      auto eigenvalues = nda::linalg::eigenvalues(F, S_ij);
      ComplexType buffer(0.0, 0.0);
      for (size_t ibnd = 0; ibnd < nbnd; ++ibnd) {
        if (eigenvalues(ibnd) - mu > 0) {
          buffer += std::log(1.0 + std::exp(-beta * (eigenvalues(ibnd) - mu)));
        } else {
          buffer += std::log(1.0 + std::exp(beta * (eigenvalues(ibnd) - mu)));
          buffer -= (eigenvalues(ibnd) - mu) * beta;
        }
      }
      tr_ln_G0 += buffer * k_weight(ik);
    }
  }
  comm.all_reduce_in_place_n(&tr_ln_G0, 1, std::plus<>{});
  tr_ln_G0 *= spin_factor / beta;

  auto tr_ln_1_minus_G0_Sigma_w = nda::array<ComplexType, 2>::zeros({nw, 1});
  size_t n_wsk = nw * ns * nkpts_ibz;
  // Diagnostic for the |det(1 - G0 Sigma)| evaluation. 
  double min_U_diag_abs = std::numeric_limits<double>::max();
  double max_U_diag_abs = 0.0;
  
  if (rank < n_wsk) {
    auto I = nda::eye<ComplexType>(nbnd);
    nda::array<ComplexType, 3> Sigma_t_ij(nt, nbnd, nbnd);
    // Buffer allocations
    //   buffer_a: Sigma_w_ij -> one_minus_G0_Sigma_w
    //   buffer_b: F
    //   buffer_c: G0_inv    -> gemm/LU buffer
    nda::matrix<ComplexType> buffer_a(nbnd, nbnd);
    nda::matrix<ComplexType> buffer_b(nbnd, nbnd);
    nda::matrix<ComplexType> buffer_c(nbnd, nbnd);
    nda::matrix<ComplexType, nda::F_layout> G0_Sigma_w_F_layout(nbnd, nbnd);
    nda::array<int, 1> ipiv(nbnd);
    auto& Sigma_w_ij            = buffer_a;
    auto& one_minus_G0_Sigma_w = buffer_a;
    auto& F                    = buffer_b;
    auto& G0_inv               = buffer_c;

    for (size_t wsk = rank; wsk < n_wsk; wsk += size) {
      size_t n  = wsk / (ns * nkpts_ibz);
      size_t sk = wsk % (ns * nkpts_ibz);
      size_t is = sk / nkpts_ibz;
      size_t ik = sk % nkpts_ibz;

      auto wn = FT->wn_mesh()(n);
      ComplexType omega_mu = FT->omega(wn) + mu;

      // tau_to_w reshapes (hence requires a contiguous input), so copy the
      // (nt, nbnd, nbnd) slice for this (is, ik) into the contiguous scratch
      // buffer before transforming frequency n.
      Sigma_t_ij = sSigma_tskij.local()(all, is, ik, all, all);
      FT->tau_to_w(Sigma_t_ij, Sigma_w_ij, imag_axes_ft::fermion, n);

      if (F_has_H0) {
        F = sF_skij.local()(is, ik, all, all);
      } else {
        F = sF_skij.local()(is, ik, all, all) + dyson.sH0_skij().local()(is, ik, all, all);
      }
      // calculate G_0 \Sigma by solving G_0^{-1} X = \Sigma
      G0_inv = omega_mu * dyson.sS_skij().local()(is, ik, all, all) - F;
      // nda tensor branch requires F_layout. 
      G0_Sigma_w_F_layout = Sigma_w_ij;
      nda::lapack::getrf(G0_inv, ipiv);
      nda::lapack::getrs(G0_inv, G0_Sigma_w_F_layout, ipiv);
      one_minus_G0_Sigma_w = I - G0_Sigma_w_F_layout;
      // JHL: Is (1-G_0\Sigma) hermitian?
      nda::blas::gemm(1.0, nda::conj(nda::transpose(one_minus_G0_Sigma_w)), one_minus_G0_Sigma_w, 0.0, buffer_c);
      nda::lapack::getrf(buffer_c, ipiv);
      for (size_t ibnd = 0; ibnd < nbnd; ++ibnd) {
        // Due to the pivoting in getrf, diagonals of U are not necessarily real, the determinant 
        // is therefore the product of the absolute values of diagonal elements of U. 
        double U_diag_abs = std::abs(buffer_c(ibnd, ibnd));
        min_U_diag_abs = std::min(min_U_diag_abs, U_diag_abs);
        max_U_diag_abs = std::max(max_U_diag_abs, U_diag_abs);
        tr_ln_1_minus_G0_Sigma_w(n, 0) += std::log(U_diag_abs) * 0.5 * k_weight(ik);
      }
    }
  }
  comm.all_reduce_in_place_n(tr_ln_1_minus_G0_Sigma_w.data(),
                             tr_ln_1_minus_G0_Sigma_w.size(), std::plus<>{});

  // Checking min/max |U(i,i)| as a proxy to see whether 1-G0S is singular or not. 
  comm.all_reduce_in_place_n(&min_U_diag_abs, 1, mpi3::min<>{});
  comm.all_reduce_in_place_n(&max_U_diag_abs, 1, mpi3::max<>{});
  constexpr double rcond_thresh = 1e-12;
  if (min_U_diag_abs < rcond_thresh * max_U_diag_abs) {
    app_log(1, "[WARNING] eval_thermodynamic_properties: (1-G0*Sigma) is (near-)singular "
               "(min/max |U(i,i)| of (1-G0*Sigma)^dag (1-G0*Sigma) = {} < {}); "
               "Tr ln(1-G0*Sigma) is unreliable.",
            min_U_diag_abs / max_U_diag_abs, rcond_thresh);
  }

  auto tr_ln_1_minus_G0_Sigma_t = nda::array<ComplexType, 2>::zeros({nt, 1});
  FT->w_to_tau(tr_ln_1_minus_G0_Sigma_w, tr_ln_1_minus_G0_Sigma_t, imag_axes_ft::fermion);

  auto tr_ln_1_minus_G0_Sigma_beta = nda::array<ComplexType, 1>::zeros({1});
  FT->tau_to_beta(tr_ln_1_minus_G0_Sigma_t, tr_ln_1_minus_G0_Sigma_beta);
  tr_ln_1_minus_G0_Sigma_beta(0) *= -1 * spin_factor;

  ComplexType grand_potential = Phi - tr_Sigma_G - tr_ln_G0 - tr_ln_1_minus_G0_Sigma_beta(0);

  app_log(1, "\n");
  app_log(1, "Grand potential contributions");
  app_log(1, "--------------------");
  app_log(1, "  Luttinger-Ward:                  {:>20.12f} a.u.", Phi.real());
  app_log(1, "  tr G*Sigma:                      {:>20.12f} a.u.", -tr_Sigma_G.real());
  app_log(1, "  tr ln(-G0):                      {:>20.12f} a.u.", -tr_ln_G0.real());
  app_log(1, "  tr ln(1-G0*Sigma):               {:>20.12f} a.u.", -tr_ln_1_minus_G0_Sigma_beta(0).real());
  app_log(1, "  total grand potential:           {:>20.12f} a.u.", grand_potential.real());
  app_log(1, "\n");

  // Sanity check for the imaginary-time Fourier-transform accuracy: the input
  // to w_to_tau is real, so a non-negligible imaginary part in the result is FT
  // noise. The threshold is therefore tied to the FT accuracy.
  if (std::abs(tr_ln_1_minus_G0_Sigma_beta(0).imag()) >= 1e2 * FT->eps()) {
    app_log(1, "[WARNING] Abs (Tr ln(1-G0*Sigma).imag()) = {},\n", tr_ln_1_minus_G0_Sigma_beta(0).imag());
    app_log(1, "          (Tr ln(1-G0*Sigma)).imag() = {},\n", tr_ln_1_minus_G0_Sigma_beta(0).imag());
  }

  // evaluate n_electron on-the-fly here. 
  nda::array<ComplexType, 4> spectra(FT->nw_f(), MF->nspin(), MF->nkpts_ibz(), MF->nbnd());
  dyson.compute_eigenspectra(sF_skij, sSigma_tskij, spectra);
  double n_electron = compute_Nelec(mu, spectra, *MF, *FT);
  double helmholtz_free_energy = grand_potential.real() + mu * n_electron;
  double entropy = (elec_energies[3] - helmholtz_free_energy) * beta;

  app_log(1, "\n");
  app_log(1, "Electron thermodynamic properties");
  app_log(1, "--------------------");
  app_log(1, "  energy:                          {:>20.12f} a.u.", elec_energies[3]);
  app_log(1, "  grand potential:                 {:>20.12f} a.u.", grand_potential.real());
  app_log(1, "  Helmholtz free energy:           {:>20.12f} a.u.", helmholtz_free_energy);
  app_log(1, "  beta:                            {:>20.12f} a.u.", beta);
  app_log(1, "  entropy:                         {:>20.12f} a.u.", entropy);
  app_log(1, "  chemical potential:              {:>20.12f} a.u.", mu);
  app_log(1, "  number of electrons:             {:>20.12f}", n_electron);
  app_log(1, "\n");

  return thermodynamics_t{grand_potential.real(), helmholtz_free_energy, 
                          entropy, n_electron};
}

template<typename dyson_type, typename X_t, typename Xt_t>
void update_G(dyson_type &dyson, const mf::MF &mf, const imag_axes_ft::IAFT &FT, X_t & Dm, Xt_t &G,
              const X_t & F, const Xt_t &Sigma, double &mu, bool const_mu) {
  app_log(2, "* Solving Green's function:");
  if(!const_mu) {
    dyson.Timer().start("UPDATE_MU");
    mu = update_mu(mu, dyson, mf, FT, F, Sigma);
    dyson.Timer().stop("UPDATE_MU");
  }
  dyson.solve_dyson(Dm, G, F, Sigma, mu);
}

namespace detail {
// For an augmented basis mf.eigval() is only diag(H_KS); the eigenvalues of the
// Dyson pencil are the spectrum the run actually samples. Report the two against
// each other, since the IAFT grid is sized from eigval (see log_spectrum_vs_wmax).
inline void log_augmented_spectrum(const mf::MF &mf,
                                   nda::array<ComplexType, 4> const& spectra) {
  if (not mf.is_augmented()) return;
  double ef = mf.efermi(), max_abs_E = 0.0;
  for (long i = 0; i < spectra.size(); ++i)
    max_abs_E = std::max(max_abs_E, std::abs(spectra.data()[i].real() - ef));
  mf::log_spectrum_vs_wmax(mf, max_abs_E, "Dyson eigenspectrum");
}
} // detail

template<typename dyson_type, typename X_t, typename Xt_t>
double update_mu_bisection(double old_mu, dyson_type& dyson, const mf::MF &mf,
                           const imag_axes_ft::IAFT &FT,
                           const X_t&F, const Xt_t&Sigma) {
  double nel_target = mf.nelec();
  double delta = 0.2;
  nda::array<ComplexType, 4> FpSigma_spectra(FT.nw_f(), mf.nspin(), mf.nkpts_ibz(), mf.nbnd());
  dyson.Timer().start("EIGENSPECTRA");
  dyson.compute_eigenspectra(F, Sigma, FpSigma_spectra);
  dyson.Timer().stop("EIGENSPECTRA");
  detail::log_augmented_spectrum(mf, FpSigma_spectra);
  auto eval_f = [&](double mu) {
    return compute_Nelec(mu, FpSigma_spectra, mf, FT) - nel_target;
  };

  double nel_old = compute_Nelec(old_mu, FpSigma_spectra, mf, FT);
  app_log(2, "Initial chemical potential (mu) = {}, nelec = {}", old_mu, nel_old);

  auto [mu, f_mu] = detail::update_mu_bisection_impl(old_mu, dyson.mu_tol(), delta, eval_f);
  double nel = f_mu + nel_target;
  app_log(1, "Chemical potential found (mu) = {} a.u.", mu);
  app_log(1, "Number of electrons per unit cell = {}", nel);
  return mu;
}

template<typename dyson_type, typename X_t, typename Xt_t>
double update_mu_midpoint(double old_mu, dyson_type& dyson, const mf::MF &mf,
                          const imag_axes_ft::IAFT &FT, const X_t&F,
                          const Xt_t&Sigma) {
  double nel_target = mf.nelec();
  double tol = dyson.mu_tol();
  double delta = 0.2;

  nda::array<ComplexType, 4> FpSigma_spectra(
      FT.nw_f(), mf.nspin(), mf.nkpts_ibz(), mf.nbnd());
  dyson.Timer().start("EIGENSPECTRA");
  dyson.compute_eigenspectra(F, Sigma, FpSigma_spectra);
  dyson.Timer().stop("EIGENSPECTRA");
  detail::log_augmented_spectrum(mf, FpSigma_spectra);

  auto eval_f = [&](double mu) {
    return compute_Nelec(mu, FpSigma_spectra, mf, FT) - nel_target;
  };

  double f_old = eval_f(old_mu);
  app_log(2, "Initial chemical potential (mu) = {}, nelec - target = {}",
          old_mu, f_old);

  auto [mu, f_mu, mu_left, mu_right] =
      detail::update_mu_midpoint_impl(old_mu, tol, delta, eval_f);
  double nel = f_mu + nel_target;
  app_log(1, "Chemical potential bounds found (mu_left, mu_right) = ({}, {}) a.u.",
          mu_left, mu_right);
  app_log(1, "Chemical potential found (mu) = {} a.u.", mu);
  app_log(1, "Number of electrons per unit cell = {}", nel);
  return mu;
}

template<typename dyson_type, typename X_t, typename Xt_t>
double update_mu(double old_mu, dyson_type& dyson, const mf::MF &mf,
                 const imag_axes_ft::IAFT &FT,
                 const X_t&F, const Xt_t&Sigma) {
  utils::check(dyson.mu_update_alg() == "bisection" or dyson.mu_update_alg() == "midpoint",
               "scf_common.cpp::update_mu: unknown mu update algorithm {}.", dyson.mu_update_alg());
  if (dyson.fixed_mu()) {
    double mu = *dyson.fixed_mu();
    nda::array<ComplexType, 4> FpSigma_spectra(FT.nw_f(), mf.nspin(), mf.nkpts_ibz(), mf.nbnd());
    dyson.compute_eigenspectra(F, Sigma, FpSigma_spectra);
    app_log(1, "Chemical potential fixed (mu) = {} a.u.", mu);
    app_log(1, "Number of electrons per unit cell = {}", compute_Nelec(mu, FpSigma_spectra, mf, FT));
    return mu;
  }
  if (dyson.mu_update_alg() == "bisection") {
    return update_mu_bisection(old_mu, dyson, mf, FT, F, Sigma);
  } else {
    return update_mu_midpoint(old_mu, dyson, mf, FT, F, Sigma);
  }
}

// Abort when `dataset` is missing from `iter_grp` and the group records Sigma_not_stored
// = true (dump_scf). A missing dataset without that flag (HF checkpoints) passes: readers
// that call this take it as an exactly-zero quantity.
static void check_checkpoint_dataset_present(h5::group &iter_grp, std::string const& dataset, long iter) {
  if (iter_grp.has_dataset(dataset) or not iter_grp.has_dataset("Sigma_not_stored")) return;
  bool not_stored = false;
  h5::h5_read(iter_grp, "Sigma_not_stored", not_stored);
  utils::check(not not_stored,
               "iter_scf: {} of iteration {} is not stored in the checkpoint (Sigma_not_stored = true) "
               "and cannot be read as the previous state for mixing.", dataset, iter);
}

// Fill m.prev with this rank's slices of the previous accepted (F, Sigma) from
// {datasets[0]}/iter{iteration-1}, for the first in-memory mixing without an in-memory copy
// (the first one after a restart, and every dmft_embed call). Every rank reads exactly the
// element slice it owns as an h5 hyperslab, so no full copy of F or Sigma is materialized.
// A missing Sigma means an exactly zero Sigma (HF checkpoints written before the
// Sigma_not_stored flag), unless chkpt_slim omitted it. Without Sigma (m.layout.nS() = 0,
// sSigma_tskij null) datasets[2] is not read.
// Every rank opening the checkpoint read-only is affordable because this runs at
// most once per run: after the first solve the previous state is in memory.
template<typename X_t, typename Xt_t>
void stage_prev_state(iter_scf::mixing_state& m, std::string h5_prefix,
                      std::array<std::string,3> const& datasets, long iteration,
                      X_t const& sF_skij, Xt_t const* sSigma_tskij) {
  auto [f0, f1] = m.layout.F_range();
  auto [s0, s1] = m.layout.S_range();
  m.prev.F = nda::array<ComplexType, 1>(f1 - f0);
  m.prev.S = nda::array<ComplexType, 1>(s1 - s0);
  m.prev.F() = ComplexType(0.0);
  m.prev.S() = ComplexType(0.0);
  std::string filename = h5_prefix + ".mbpt.h5";
  h5::file file(filename, 'r');
  h5::group grp(file);
  std::string grp_name = datasets[0]+"/iter"+std::to_string(iteration-1);
  utils::check(grp.has_subgroup(grp_name),
               "stage_prev_state: {} does not exist in {}.", grp_name, filename);
  auto it_grp = grp.open_group(grp_name);
  if (it_grp.has_dataset(datasets[1]))
    utils::h5_read_flat_range<4>(it_grp, datasets[1], sF_skij.shape(), f0, f1, m.prev.F.data());
  if (sSigma_tskij) {
    check_checkpoint_dataset_present(it_grp, datasets[2], iteration-1);
    if (it_grp.has_dataset(datasets[2]))
      utils::h5_read_flat_range<5>(it_grp, datasets[2], sSigma_tskij->shape(), s0, s1, m.prev.S.data());
  }
  m.has_prev = true;
}

template<typename comm_t, typename X_t, typename Xt_t>
auto in_memory_mix_step(utils::mpi_context_t<comm_t> &context, iter_scf::mixing_state& m,
                        std::string const& h5_prefix, std::array<std::string,3> const& datasets,
                        long iteration, X_t &sF_skij, Xt_t *sSigma_tskij,
                        iter_scf::hist_vec const* residual)
  -> std::tuple<double, double> {
  if (not m.has_prev) stage_prev_state(m, h5_prefix, datasets, iteration, sF_skij, sSigma_tskij);
  nda::array<ComplexType, 5> no_Sigma;
  sF_skij.win().fence();
  if (sSigma_tskij) sSigma_tskij->win().fence();
  std::array<double, 2> conv = sSigma_tskij
      ? m.step(context.comm, sF_skij.local(), sSigma_tskij->local(), residual)  // F and Sigma
      : m.step(context.comm, sF_skij.local(), no_Sigma, residual);               // F only
  // max is exactly order-independent
  context.comm.all_reduce_in_place_n(conv.data(), 2, mpi3::max<>{});
  if (sSigma_tskij) utils::complete_shared_slices(context, m.layout.pmap(), sF_skij, *sSigma_tskij);
  else utils::complete_shared_slices(context, m.layout.pmap(), sF_skij);
  return std::make_tuple(conv[0], conv[1]);
}

template<typename comm_t, typename X_t, typename Xt_t>
void init_mixing_state(utils::mpi_context_t<comm_t> &context, iter_scf::iter_scf_t& iter_solver,
                       iter_scf::mixing_state& m, X_t &sF_skij, Xt_t *sSigma_tskij) {
  // DIIS extrapolates from two stored pairs; the first step stores none (needs_residual() is
  // false at steps() == 0)
  std::optional<iter_scf::diis_mixer::config> cfg;
  if (auto* dp = iter_solver.diis())
    cfg = iter_scf::diis_mixer::config{dp->mixing, dp->max_subsp_size, dp->warmup_iter, 2, 2};
  long nS = sSigma_tskij ? sSigma_tskij->local().size() : 0;
  m.init(context, sF_skij.local().size(), nS, iter_solver.mixing(), cfg);
  app_log(2, "In-memory mixing: element-sliced state active ({} parts over {} node(s), "
             "slice sizes F {} / Sigma {})",
          m.layout.pmap().nparts, context.internode_comm.size(), m.layout.mF(), m.layout.mS());
}

// The k-striped distributed commutator residual C_t of the Dyson-SCF DIIS, computed by all
// ranks. It is accumulated into a node-shared window: the (s,k) partition is disjoint, so
// each rank writes distinct blocks (no intra-node reduce) and only the internode sum + a
// fence are needed to complete it.
template<typename MPI_Context_t, typename X_t, typename Xt_t>
auto distributed_commutator(MPI_Context_t &context, const imag_axes_ft::IAFT *FT,
                            const Xt_t& sG_tskij, X_t &sF_skij, Xt_t &sSigma_tskij, double mu,
                            const X_t& sS_skij, const X_t& sH0_skij) {
  auto [nt, ns, nk, nao, nao2] = sSigma_tskij.shape();
  auto sC_t = math::shm::make_shared_array<Array_view_5D_t>(
      context.comm, context.internode_comm, context.node_comm, {nt, ns, nk, nao, nao2});
  sC_t.win().fence();
  iter_scf::dyson_commutator(
      context.comm, sC_t.local(), FT,
      sG_tskij.local(), sF_skij.local(), sSigma_tskij.local(), mu,
      sS_skij.local(), sH0_skij.local());
  iter_scf::diis_timers::com_dist_reduce.start();
  sC_t.win().fence();
  sC_t.all_reduce(); // combine per-node windows across nodes (no-op on 1 node)
  iter_scf::diis_timers::com_dist_reduce.stop();
  return sC_t;
}

template<typename comm_t, typename X_t, typename Xt_t>
auto solve_iterative(utils::mpi_context_t<comm_t> &context, iter_scf::iter_scf_t& iter_solver,
                     long iteration, std::string h5_prefix,
                     X_t &sF_skij, Xt_t &sSigma_tskij, const imag_axes_ft::IAFT *FT,
                     std::array<std::string,3> datasets,
                     const Xt_t* sG_tskij, double mu,
                     const X_t* sS_skij, const X_t* sH0_skij)
  -> std::tuple<double, double> {
  double conv_F = 0;
  double conv_Sigma = 0;
  auto& m = iter_solver.state();
  if (iteration == 1) {
    // Just check changes w.r.t. mf
    if (context.node_comm.root()) {
      auto F_mf = nda::make_regular(sF_skij.local());
      h5::file file(h5_prefix+".mbpt.h5", 'r');
      h5::group grp(file);
      if (grp.has_subgroup("scf/iter0")) {
        auto mf_grp = grp.open_group("scf/iter0");
        if (mf_grp.has_dataset("F_skij")) {
          nda::h5_read(mf_grp, "F_skij", F_mf);
        } else if (mf_grp.has_dataset("Heff_skij")) {
          // checkpoint from a qp scf
          nda::h5_read(mf_grp, "Heff_skij", F_mf);
          nda::array<ComplexType, 4> H0(F_mf.shape());
          auto sys_grp = grp.open_group("system");
          nda::h5_read(sys_grp, "H0_skij", H0);
          F_mf -= H0;
        }
      }
      F_mf -= sF_skij.local();
      auto Fmax_iter = max_element(F_mf.data(), F_mf.data()+F_mf.size(),
                                   [](auto a, auto b) { return std::abs(a) < std::abs(b); });
      conv_F =  std::abs((*Fmax_iter));
    }
    context.node_comm.broadcast_n(&conv_F, 1, 0);
    auto Sigma_max_iter = max_element(sSigma_tskij.local().data(), sSigma_tskij.local().data()+sSigma_tskij.local().size(),
                                      [](auto a, auto b) { return std::abs(a) < std::abs(b); });
    conv_Sigma =  std::abs((*Sigma_max_iter));
    // Every rank captures its slices of the iteration-1 state. No mixing is applied
    // at iteration 1, so this state is also the accepted previous state of the
    // first mixing.
    init_mixing_state(context, iter_solver, m, sF_skij, &sSigma_tskij);
    m.layout.store(m.prev, sF_skij.local(), sSigma_tskij.local());
    m.has_prev = true;
    context.comm.barrier();
  } else {
    iter_solver.metadata_log();
    // Restart: no in-memory previous state; in_memory_mix_step stages it from the checkpoint.
    if (not m.initialized())
      init_mixing_state(context, iter_solver, m, sF_skij, &sSigma_tskij);
    // Commutator residual of the trial, for the DIIS steps that store a pair
    std::optional<iter_scf::hist_vec> residual;
    if (m.needs_residual()) {
      utils::check(sG_tskij and sS_skij and sH0_skij,
                   "solve_iterative: the DIIS commutator residual needs G, S and H0.");
      auto sC_t = distributed_commutator(context, FT, *sG_tskij, sF_skij, sSigma_tskij, mu,
                                         *sS_skij, *sH0_skij);
      residual.emplace();
      residual->S = m.layout.S(sC_t.local()); // only the Sigma block of the commutator is non-zero
    }
    std::optional<iter_scf::diis_timers::ScopeLog> breakdown_log;
    if (iter_solver.iter_alg() == iter_scf::DIIS) breakdown_log.emplace();
    std::tie(conv_F, conv_Sigma) = in_memory_mix_step(
        context, m, h5_prefix, datasets, iteration, sF_skij, &sSigma_tskij,
        residual ? &*residual : nullptr);
    context.comm.barrier();
  }
  return std::make_tuple(conv_F, conv_Sigma);
}

template<typename dyson_type>
void write_mf_data(mf::MF &mf,
                   const imag_axes_ft::IAFT &ft, dyson_type &dyson,
                   std::string output) {
  auto mpi = mf.mpi();
  sArray_t<Array_view_4D_t> sF_skij(math::shm::make_shared_array<Array_view_4D_t>(
      *mpi, {mf.nspin(), mf.nkpts_ibz(), mf.nbnd(), mf.nbnd()}));
  sArray_t<Array_view_4D_t> sDm_skij(math::shm::make_shared_array<Array_view_4D_t>(
      *mpi, {mf.nspin(), mf.nkpts_ibz(), mf.nbnd(), mf.nbnd()}));
  sArray_t<Array_view_5D_t> G_shm(math::shm::make_shared_array<Array_view_5D_t>(
      *mpi, {ft.nt_f(), mf.nspin(), mf.nkpts_ibz(), mf.nbnd(), mf.nbnd()}));
  sArray_t<Array_view_5D_t> Sigma_shm(math::shm::make_shared_array<Array_view_5D_t>(
      *mpi, {ft.nt_f(), mf.nspin(), mf.nkpts_ibz(), mf.nbnd(), mf.nbnd()}));
  hamilt::set_fock(mf, dyson.PSP(), sF_skij, true, &dyson.sH0_skij());
  double mu = 0.0;

  // init Green's function. By default, we update mu as well.
  update_G(dyson, mf, ft, sDm_skij, G_shm, sF_skij, Sigma_shm, mu, false);

  chkpt::write_metadata(mpi->comm, mf, ft, dyson.sH0_skij(), dyson.sS_skij(), output);
  chkpt::dump_scf(mpi->comm, 0, sDm_skij, G_shm, sF_skij, Sigma_shm, mu, output);
}

template<typename MPI_Context_t>
auto read_greens_function(MPI_Context_t &context, mf::MF *mf,
                          std::string filename, long scf_iter, std::string scf_grp)
-> sArray_t<Array_view_5D_t> {
  using math::shm::make_shared_array;

  h5::file file(filename, 'r');
  h5::group grp(file);

  nda::array<double, 1> tau_mesh;
  auto iaft_grp = h5::group(file).open_group("imaginary_fourier_transform");
  auto tau_grp = iaft_grp.open_group("tau_mesh");
  nda::h5_read(tau_grp, "fermion", tau_mesh);
  int nts = tau_mesh.shape(0);
  int ns = mf->nspin();
  int nkpts_ibz = mf->nkpts_ibz();
  int nbnd = mf->nbnd();

  auto sG_tskij = make_shared_array<Array_view_5D_t>(context.comm, context.internode_comm,
                                                     context.node_comm, {nts, ns, nkpts_ibz, nbnd, nbnd});

  auto iter_grp = h5::group(file).open_group(scf_grp+"/iter"+std::to_string(scf_iter));
  if (iter_grp.has_dataset("G_tskij")) {
    // it's a Dyson type calculation -> read Green's function
    sG_tskij.win().fence();
    if (context.node_comm.root()) {
      auto Gloc = sG_tskij.local();
      nda::h5_read(iter_grp, "G_tskij", Gloc);
    }
    sG_tskij.win().fence();
  } else if (iter_grp.has_dataset("F_skij")) {
    // Dyson checkpoint with G omitted: chkpt_slim writes G only on the final
    // iteration, so a run killed mid-loop leaves intermediate groups G-less.
    utils::check(false,
        "read_greens_function: {}/iter{} is a Dyson checkpoint without G_tskij "
        "(chkpt_slim intermediate iteration). Request the final complete iteration "
        "or rerun with chkpt_slim disabled.", scf_grp, scf_iter);
  } else {
    // it's a qp type calculation -> construct the Green's function on-the-fly
    auto ft = imag_axes_ft::read_iaft(filename, false);
    auto sMO_skia = make_shared_array<Array_view_4D_t>(
        context.comm, context.internode_comm, context.node_comm, {ns, nkpts_ibz, nbnd, nbnd});
    auto sE_ska = make_shared_array<Array_view_3D_t>(
        context.comm, context.internode_comm, context.node_comm, {ns, nkpts_ibz, nbnd});
    double mu;

    sMO_skia.win().fence();
    if (context.node_comm.root()) {
      auto MO_loc = sMO_skia.local();
      auto E_loc = sE_ska.local();
      nda::h5_read(iter_grp, "MO_skia", MO_loc);
      nda::h5_read(iter_grp, "E_ska", E_loc);
    }
    sMO_skia.win().fence();
    h5::h5_read(iter_grp, "mu", mu);

    update_G(sG_tskij, sMO_skia, sE_ska, mu, ft);
  }
  context.comm.barrier();
  return sG_tskij;
}


template auto eval_hf_energy(const sArray_t<Array_view_4D_t>&, const sArray_t<Array_view_4D_t>&, const sArray_t<Array_view_4D_t>&,
                             nda::array_contiguous_const_view<double, 1>&, bool)
    -> std::tuple<double, double>;

template double eval_corr_energy(mpi3::communicator& comm, const imag_axes_ft::IAFT &,
                                 const sArray_t<Array_view_5D_t> &, const sArray_t<Array_view_5D_t> &,
                                 nda::array_contiguous_const_view<double, 1>&);

template auto eval_thermodynamic_properties(mpi3::communicator&, simple_dyson&, const sArray_t<Array_view_4D_t>&, 
                                            const sArray_t<Array_view_5D_t>&, const std::vector<double>&, 
                                            double, double, bool)
    -> thermodynamics_t;

template void update_G(simple_dyson &, const mf::MF &, const imag_axes_ft::IAFT &,
                       sArray_t<Array_view_4D_t> & Dm, sArray_t<Array_view_5D_t> &G,
                       const sArray_t<Array_view_4D_t> & F, const sArray_t<Array_view_5D_t> &Sigma, double&,
                       bool);

template double update_mu(double, simple_dyson&, const mf::MF &, const imag_axes_ft::IAFT &,
                          const sArray_t<Array_view_4D_t>&, const sArray_t<Array_view_5D_t>&);

template auto solve_iterative(utils::mpi_context_t<mpi3::communicator>&, iter_scf::iter_scf_t&, long, std::string,
                              sArray_t<Array_view_4D_t>&, sArray_t<Array_view_5D_t>&, const imag_axes_ft::IAFT*,
                              std::array<std::string,3>,
                              const sArray_t<Array_view_5D_t>*, double,
                              const sArray_t<Array_view_4D_t>*, const sArray_t<Array_view_4D_t>*)
         -> std::tuple<double, double>;

template void init_mixing_state(utils::mpi_context_t<mpi3::communicator>&, iter_scf::iter_scf_t&,
                                iter_scf::mixing_state&, sArray_t<Array_view_4D_t>&,
                                sArray_t<Array_view_5D_t>*);

template auto in_memory_mix_step(utils::mpi_context_t<mpi3::communicator>&, iter_scf::mixing_state&,
                                 std::string const&, std::array<std::string,3> const&, long,
                                 sArray_t<Array_view_4D_t>&, sArray_t<Array_view_5D_t>*,
                                 iter_scf::hist_vec const*)
         -> std::tuple<double, double>;

template void write_mf_data(mf::MF&, const imag_axes_ft::IAFT&, simple_dyson&,
                            std::string);
template auto read_greens_function(utils::mpi_context_t<>&, mf::MF*, std::string, long, std::string)
    -> sArray_t<Array_view_5D_t>;
} // methods
