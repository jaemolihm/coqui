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


#ifndef COQUI_SIMPLE_DYSON_H
#define COQUI_SIMPLE_DYSON_H

#include <optional>

#include "utilities/Timer.hpp"
#include "IO/app_loggers.h"

#include "utilities/mpi_context.h"
#include "mean_field/MF.hpp"
#include "mean_field/distributed_orbital_readers.hpp"
#include "hamiltonian/one_body_hamiltonian.hpp"
#include "hamiltonian/pseudo/pseudopot.h"
#include "numerics/imag_axes_ft/IAFT.hpp"
#include "methods/SCF/scf_common.hpp"
#include "methods/tools/chkpt_utils.h"

namespace methods {

/*
 * MAM: Consider creating an object to represent overlap matrices. If the basis is orthonormal,
 * nothing is stored, otherwise the overlap matrix is stored in shared memory. 
 * Subsequent routines can poll the object to know if the basis is orthonormal or not,
 * and act accordingly
 */

// CNY: This class is currently responsible for storing "H0" and "ovlp" matrices.
//      Is this the right place?

/**
 * @class simple_dyson
 * @brief Handler for solving the Dyson equation
 *
 * This class is responsible for solving the Dyson equation for a specified physical system.
 * The system, including nbnd, nkpt, H0 etc, is defined upon construction through a mean-field
 * object. The Dyson equation for computing the Green's function 'G_tau' can be solved by calling:
 *
 *     simple_dyson.solve_dyson(G_tau, F, Sigma_tau, mu);
 *
 * where `F` and `Sigma_tau` represent the static and dynamic components of the self-energy,
 * respectively, and `mu` is the chemical potential.
 */
class simple_dyson {
public:
  simple_dyson(mf::MF* MF, imag_axes_ft::IAFT* FT,
               double mu_tol = 1e-9,
               std::string mu_update_alg = "bisection",
               bool dyson_no_inversion = false):
      _MF(MF), _context(MF->mpi()), _FT(FT), _PSP(hamilt::make_pseudopot(*_MF)),
      _mu_tol(mu_tol), _mu_update_alg(mu_update_alg), _dyson_no_inversion(dyson_no_inversion),
      _nts(_FT->nt_f()), _nw(_FT->nw_f()),
      _ns(_MF->nspin()), _nkpts(_MF->nkpts()), _nkpts_ibz(_MF->nkpts_ibz()), 
      _nbnd(_MF->nbnd()),
      _sH0_skij(math::shm::make_shared_array<Array_view_4D_t>(*_context, {_ns, _nkpts_ibz, _nbnd, _nbnd})),
      _sS_skij(math::shm::make_shared_array<Array_view_4D_t>(*_context, {_ns, _nkpts_ibz, _nbnd, _nbnd})),
      _Timer() {

    hamilt::set_H0(*_MF, _PSP.get(), _sH0_skij);
    hamilt::set_ovlp(*_MF, _sS_skij);
    if (_context->node_comm.root()) {
      hermitize_in_tau(_sH0_skij.local());
      hermitize_in_tau(_sS_skij.local());
    }

    for( auto& v: {"DYSON", "SIGMA_TAU_TO_W", "DYSON_LOOP", "REDISTRIBUTE", "DYSON_GATHER",
                   "UPDATE_MU", "EIGENSPECTRA", "DM_TAU_TO_BETA"} ) {
      _Timer.add(v);
    }
    _context->comm.barrier();
    if (_dyson_no_inversion) check_dyson_no_inversion_requirements();
  }

  simple_dyson(mf::MF* MF, imag_axes_ft::IAFT* FT,
               std::string H0_S_chkpt,
               double mu_tol = 1e-9,
               std::string mu_update_alg = "bisection",
               bool dyson_no_inversion = false):
      _MF(MF), _context(_MF->mpi()),
      _FT(FT), _mu_tol(mu_tol), _mu_update_alg(mu_update_alg), _dyson_no_inversion(dyson_no_inversion),
      _nts(_FT->nt_f()), _nw(_FT->nw_f()),
      _ns(_MF->nspin()), _nkpts(_MF->nkpts()), _nkpts_ibz(_MF->nkpts_ibz()),
      _nbnd(_MF->nbnd()),
      _sH0_skij(math::shm::make_shared_array<Array_view_4D_t>(*_context, {_ns, _nkpts_ibz, _nbnd, _nbnd})),
      _sS_skij(math::shm::make_shared_array<Array_view_4D_t>(*_context, {_ns, _nkpts_ibz, _nbnd, _nbnd})),
      _Timer() {

    chkpt::read_H0(_context->node_comm, H0_S_chkpt, _sH0_skij);
    chkpt::read_ovlp(_context->node_comm, H0_S_chkpt, _sS_skij);
    _context->comm.barrier();

    for( auto& v: {"DYSON", "SIGMA_TAU_TO_W", "DYSON_LOOP", "REDISTRIBUTE", "DYSON_GATHER",
                   "UPDATE_MU", "EIGENSPECTRA", "DM_TAU_TO_BETA"} ) {
      _Timer.add(v);
    }
    _context->comm.barrier();
    if (_dyson_no_inversion) check_dyson_no_inversion_requirements();
  }

  simple_dyson(simple_dyson const&) = default;
  simple_dyson(simple_dyson &&) = default;
  simple_dyson & operator=(const simple_dyson &) = default;
  simple_dyson & operator=(simple_dyson &&) = default;

  ~simple_dyson(){}

  /**
   * Solve Dyson's equation and update _G_tskij_view for current _mu, _F, and _Sigma.
   * With dyson_no_inversion, every call with a nonzero Sigma except the very first call of this object updates
   * G by an imaginary-time Picard sweep instead (solve_dyson_tau), starting from the G in _G_shm. In a fresh run
   * only the Sigma = 0 initialization is a node solve and every GW iteration, the first included, is a sweep from
   * the initial G; the first call of a restarted run is a node solve.
   */
  template<typename G_t, typename F_t, typename Sigma_t>
  void solve_dyson(G_t&_G_shm, const F_t&_sF_skij, const Sigma_t &_Sigma_shm, double mu);
  template<typename Dm_t, typename G_t, typename F_t, typename Sigma_t>
  void solve_dyson(Dm_t&_sDm_skij, G_t&_G_shm, const F_t&_sF_skij, const Sigma_t &_Sigma_shm, double mu);

  /**
   * max|Sweep(G) - G| for one imaginary-time Picard sweep (solve_dyson_tau) of the given (F, Sigma, mu) from G, which is
   * left unchanged. Used by run_lr to log how far a stored ground-state G is from the sweep's fixed point
   * (for a sweep of contraction factor ρ < 1, |G - G*| <= change / (1 - ρ)). Requires dyson_no_inversion; collective.
   */
  template<typename G_t, typename F_t, typename Sigma_t>
  double picard_sweep_change(G_t&_G_shm, const F_t&_sF_skij, const Sigma_t &_Sigma_shm, double mu);

  /**
   * Compute the eigenvalues of _F + _Sigma(iwn)
   * @param spectra - [OUTPUT] eigenvalues, (nw, ns, nkpts_ibz, nbnd)
   */
  template<typename X_t, typename Xt_t>
  void compute_eigenspectra(const X_t&_sF_skij, const Xt_t &_Sigma_shm, nda::array<ComplexType, 4> &spectra);

  inline void print_timers() {
    app_log(2, "\n  DYSON timers");
    app_log(2, "  ------------");
    app_log(2, "    Update mu:                      {0:.3f} sec", _Timer.elapsed("UPDATE_MU"));
    app_log(2, "      - Eigenspectra:               {0:.3f} sec", _Timer.elapsed("EIGENSPECTRA"));
    app_log(2, "    Dyson eqn:                      {0:.3f} sec", _Timer.elapsed("DYSON"));
    app_log(2, "      - Sigma(t)->Sigma(w):         {0:.3f} sec", _Timer.elapsed("SIGMA_TAU_TO_W"));
    app_log(2, "      - Dyson loop:                 {0:.3f} sec", _Timer.elapsed("DYSON_LOOP"));
    app_log(2, "      - Redistribute                {0:.3f} sec", _Timer.elapsed("REDISTRIBUTE"));
    app_log(2, "      - Gather:                     {0:.3f} sec", _Timer.elapsed("DYSON_GATHER"));
    app_log(2, "    Dm tau->beta:                   {0:.3f} sec\n", _Timer.elapsed("DM_TAU_TO_BETA"));
  }

private:
  /**
   * Update G in _G_shm by one Picard sweep of the imaginary-time Dyson equation, without Matsubara nodes.
   *
   * Called by solve_dyson after its first (Matsubara-node) solve when dyson_no_inversion is set and Sigma is nonzero.
   * The node solve inverts (iw + mu) S - H0 - F - Sigma(iw) on the DLR Matsubara nodes and refits G(tau); that
   * solve-and-refit amplifies the worst-conditioned DLR tau directions of the SCF response, which makes some gapped
   * systems stall or diverge in scGW. This path never touches the Matsubara nodes: it uses exact DLR convolutions on
   * the tau nodes,
   *     G <- G0' + G0' * Sigma * G - (G0' Sigma_ref) * G,    (f * g)(tau) = int_0^beta f(tau - tau') g(tau') dtau',
   * where G0' = [(iw + mu) - H0 - F - Sigma_ref]^-1 and Sigma_ref is the Hermitian part of Sigma at the lowest
   * positive Matsubara node. The fixed point is the Dyson equation G = G0 + G0 * Sigma * G for G0 from H0 + F;
   * Sigma_ref only shrinks the contraction factor.
   *
   * G in _G_shm is the previous call's result and is carried between SCF iterations: a restart begins with a node
   * solve, and a run that stops before convergence leaves a G that is one sweep from the previous one, not the
   * Dyson solution of the stored (F, Sigma). Limits: DLR basis only, S = 1 (checked at construction), tested on
   * gapped systems.
   *
   * @param store_G - [INPUT] false: _G_shm is left untouched (only the change is computed)
   * @return max|G_new - G_prev| over all elements and ranks
   */
  template<typename G_t, typename F_t, typename Sigma_t>
  double solve_dyson_tau(G_t&_G_shm, const F_t&_sF_skij, const Sigma_t &_Sigma_shm, double mu, bool store_G = true);

  /// Abort unless the IAFT basis is DLR and the overlap matrix is the identity (requirements of solve_dyson_tau).
  void check_dyson_no_inversion_requirements();

  mf::MF* _MF = nullptr;
  std::shared_ptr<utils::mpi_context_t<mpi3::communicator>> _context;
  imag_axes_ft::IAFT* _FT = nullptr;
  std::shared_ptr<hamilt::pseudopot> _PSP;
  double _mu_tol = 1e-9;
  std::string _mu_update_alg = "bisection";
  // true: every solve_dyson call with a nonzero Sigma except the first call of this object is a Picard sweep
  // (solve_dyson_tau)
  bool _dyson_no_inversion = false;
  // if set, update_mu returns this value instead of searching for the target number of electrons
  std::optional<double> _fixed_mu;

  int _nts;
  int _nw;
  int _ns;
  int _nkpts;
  int _nkpts_ibz;
  int _nbnd;

  sArray_t<Array_view_4D_t> _sH0_skij;
  sArray_t<Array_view_4D_t> _sS_skij;

  utils::TimerManager _Timer;
  bool _last_solve_tau = false;  // set by every solve_dyson call: true iff it ran solve_dyson_tau
  bool _G_has_previous_solution = false;  // true once solve_dyson has stored a G in _G_shm (the start of the next Picard sweep)

public:
  const mf::MF* MF() const { return _MF; }
  auto& mpi() const { return _context; }
  imag_axes_ft::IAFT* FT() { return _FT; }
  hamilt::pseudopot* PSP() { return _PSP.get(); }
  const hamilt::pseudopot* PSP() const { return _PSP.get(); }
  double mu_tol() const { return _mu_tol; }
  std::string mu_update_alg() const { return _mu_update_alg; }
  bool dyson_no_inversion() const { return _dyson_no_inversion; }
  /// True iff the last solve_dyson call was an imaginary-time Picard sweep (a ground state whose stored G is the
  /// fixed point of the sweep, not the Matsubara-node solution); false before any call.
  bool last_solve_was_tau() const { return _last_solve_tau; }
  std::optional<double>& fixed_mu() { return _fixed_mu; }
  const std::optional<double>& fixed_mu() const { return _fixed_mu; }
  auto H0() const& { return _sH0_skij.local(); }
  const auto &sS_skij() const {return _sS_skij;}
  const auto &sH0_skij() const {return _sH0_skij;}
  auto &Timer() { return _Timer; }
};

}


#endif // COQUI_SIMPLE_DYSON_H
