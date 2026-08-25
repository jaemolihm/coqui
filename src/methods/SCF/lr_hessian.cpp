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
#include <cmath>
#include <utility>

#include "methods/SCF/lr_hessian.hpp"
#include "nda/blas.hpp"
#include "utilities/check.hpp"
#include "IO/app_loggers.h"

namespace methods {

namespace {

/// ‖A − A†‖_F / ‖A‖_F, 0 for an all-zero matrix.
double herm_dev(nda::array<ComplexType, 2> const& A) {
  const long n = A.shape(0);
  double d2 = 0.0, a2 = 0.0;
  for (long i = 0; i < n; ++i)
    for (long j = 0; j < n; ++j) {
      a2 += std::norm(A(i, j));
      d2 += std::norm(A(i, j) - std::conj(A(j, i)));
    }
  return (a2 > 0.0) ? std::sqrt(d2 / a2) : 0.0;
}

} // namespace


lr_hessian_t::lr_hessian_t(
    std::shared_ptr<mpi_context_t> mpi,
    imag_axes_ft::IAFT const& FT,
    nda::array<double, 1> const& k_weight,
    double spin_factor, long nmodes, bool has_sigma,
    long ns, long nk_ibz, long nbnd)
    : _mpi(std::move(mpi)),
      _FT(&FT),
      _spin_factor(spin_factor),
      _nmodes(nmodes),
      _has_sigma(has_sigma),
      _nt(FT.nt_f()),
      _nw(FT.nw_f()),
      _nk(nk_ibz),
      _nbnd(nbnd),
      _nF(ns * nk_ibz * nbnd * nbnd),
      _Timer() {

  for (auto& v : {"LR_HESS_STORE", "LR_HESS_REBUILD", "LR_HESS_DYSON",
                  "LR_HESS_CONTRACT"})
    _Timer.add(v);

  utils::check(nmodes > 0, "lr_hessian_t: nmodes must be > 0, got {}.", nmodes);
  utils::check(k_weight.size() == nk_ibz,
               "lr_hessian_t: k_weight has {} entries, expected nk_ibz = {}.",
               k_weight.size(), nk_ibz);
  // No square-grid requirement, and nt_f != nw_f in practice (95 vs 96 here).
  //
  // What the estimator needs is that the ΔΣ it CONTRACTED (`_Sw`, in ω) and the
  // ΔΣ the extra Dyson SOLVES with are the same operator. rebuild_raw_kernel
  // hands the solve w_to_tau(_Sw), and the solve transforms that back to ω
  // itself, so the composition in play is
  //
  //   tau_to_w ∘ w_to_tau = I  on the ω coefficients,
  //
  // which holds whenever the τ sampling determines those coefficients — a
  // property of the IR basis, not of the two grids having equal size.

  _pmap = utils::make_part_map(*_mpi);
  std::tie(_i0, _i1) = _pmap.my_slice(_nF);
  _nloc = _i1 - _i0;

  // iω_n → −iω_n on the fermionic grid. The pairing is the τ-domain conjugate
  // transpose, FT[A(τ)†](iω_n) = A(−iω_n)†, so the grid has to be closed under
  // negation for it to be representable at all.
  {
    nda::array<long, 1> wn(_FT->wn_mesh_f());
    utils::check(wn.size() == _nw,
                 "lr_hessian_t: wn_mesh_f has {} entries, expected nw_f = {}.",
                 wn.size(), _nw);
    _refl = nda::array<long, 1>(_nw);
    _refl() = -1;
    for (long n = 0; n < _nw; ++n)
      for (long m = 0; m < _nw; ++m)
        if (wn(m) == -wn(n)) { _refl(n) = m; break; }
    for (long n = 0; n < _nw; ++n)
      utils::check(_refl(n) >= 0,
                   "lr_hessian_t: the fermionic Matsubara grid is not closed "
                   "under negation — no partner for index {} (wn = {}). The hessian "
                   "estimator's inner product is not representable on it.",
                   n, wn(n));
  }

  // w_k of each of this rank's elements. The flattened (s,k,i,j) index e gives
  // k = (e / nbnd²) % nk.
  _wloc = nda::array<ComplexType, 1>(_nloc);
  for (long l = 0; l < _nloc; ++l) {
    const long e = _i0 + l;
    const long ik = (e / (_nbnd * _nbnd)) % _nk;
    _wloc(l) = ComplexType(k_weight(ik));
  }

  _Sw.resize(_has_sigma ? _nmodes : 0);
  _Gw.resize(_has_sigma ? _nmodes : 0);
  _dH0.resize(_nmodes);
  _dF.resize(_nmodes);
  _dDm.resize(_nmodes);

  for (auto* a : {&_plain_stat, &_static_prime_stat, &_M_stat, &_Mp_stat,
                  &_M_dyn, &_Mp_dyn}) {
    *a = nda::array<ComplexType, 2>(_nmodes, _nmodes);
    (*a)() = ComplexType(0.0);
  }

  _stored.assign(_nmodes, false);

  // The ω stores are the largest allocation the feature makes; they are sized and
  // reported by lr_driver::print_memory_estimate along with every other large LR
  // array, so only the configuration is announced here.
  app_log(1, "\n  Stationary free-energy-hessian estimator: ON");
  app_log(1, "    perturbations = {}, dynamic ΔΣ = {}", _nmodes, _has_sigma ? "yes" : "no");
  app_log(1, "    plus one extra LR Dyson solve per perturbation");
  app_log(2, "    this rank owns {} of {} elements", _nloc, _nF);
}


void lr_hessian_t::pack_to_omega(sArray_t<Array_view_5D_t> const& src,
                                 nda::array<ComplexType, 2>& dst_w,
                                 bool weighted) const {
  auto loc = src.local();
  // nda::reshape only EXPECTS the sizes to match, which compiles out under
  // NDEBUG and would silently reinterpret memory in a release build. An
  // nbnd_save-trimmed array reaching here is exactly that case.
  utils::check(loc.size() == _nt * _nF,
               "lr_hessian_t: rank-5 array has {} elements, expected "
               "nt*ns*nk*nb*nb = {}. A band-trimmed array cannot be used here.",
               loc.size(), _nt * _nF);
  auto flat = nda::reshape(loc, std::array<long, 1>{_nt * _nF});
  nda::array<ComplexType, 2> buf_t(_nt, _nloc);
  dst_w = nda::array<ComplexType, 2>(_nw, _nloc);
  if (_nloc == 0) return;
  for (long t = 0; t < _nt; ++t)
    buf_t(t, nda::range::all) = flat(nda::range(t * _nF + _i0, t * _nF + _i1));
  _FT->tau_to_w(buf_t, dst_w, imag_axes_ft::fermion);
  if (weighted)
    for (long n = 0; n < _nw; ++n)
      for (long l = 0; l < _nloc; ++l) dst_w(n, l) *= _wloc(l);
}


nda::array<ComplexType, 1> lr_hessian_t::pack_static(
    sArray_t<Array_view_4D_t> const& src, bool weighted) const {
  auto loc = src.local();
  utils::check(loc.size() == _nF,
               "lr_hessian_t: rank-4 array has {} elements, expected "
               "ns*nk*nb*nb = {}.", loc.size(), _nF);
  auto flat = nda::reshape(loc, std::array<long, 1>{_nF});
  nda::array<ComplexType, 1> out(_nloc);
  if (_nloc > 0) out = flat(nda::range(_i0, _i1));
  if (weighted)
    for (long l = 0; l < _nloc; ++l) out(l) *= _wloc(l);
  return out;
}


void lr_hessian_t::store_mode(long p,
                                       sArray_t<Array_view_4D_t> const& sDeltaDm,
                                       sArray_t<Array_view_4D_t> const& sDeltaH0,
                                       sArray_t<Array_view_4D_t> const& sDeltaF,
                                       sArray_t<Array_view_5D_t> const* sDeltaSigma,
                                       sArray_t<Array_view_5D_t> const* sDeltaG) {
  utils::check(p >= 0 && p < _nmodes,
               "lr_hessian_t::store_mode: mode {} outside [0, {}).", p, _nmodes);
  utils::check(!_stored[p],
               "lr_hessian_t::store_mode: mode {} stored twice.", p);
  utils::check(_has_sigma == (sDeltaSigma != nullptr),
               "lr_hessian_t::store_mode: ΔΣ presence must match the "
               "has_sigma the object was built with.");
  utils::check(!_has_sigma || sDeltaG != nullptr,
               "lr_hessian_t::store_mode: a Σ-carrying kernel needs ΔG.");

  //
  // w_k belongs to the RIGHT operand of each trace, folded in once here so a pair
  // contraction is a single unweighted dot over the whole local slab. Which side a
  // quantity lands on is fixed by the equation: ΔH0, ΔF, ΔΣ are always left, ΔDm
  // and ΔG always right.
  _Timer.start("LR_HESS_STORE");
  _dH0[p] = pack_static(sDeltaH0, /*weighted=*/false);   // left  of plain, static'
  _dF[p]  = pack_static(sDeltaF,  /*weighted=*/false);   // left  of M, M'
  _dDm[p] = pack_static(sDeltaDm, /*weighted=*/true);    // right of plain, M
  if (_has_sigma) {
    pack_to_omega(*sDeltaSigma, _Sw[p], /*weighted=*/false);  // left  of M, M'
    pack_to_omega(*sDeltaG,     _Gw[p], /*weighted=*/true);   // right of M
  }
  _stored[p] = true;
  _Timer.stop("LR_HESS_STORE");
}


void lr_hessian_t::rebuild_raw_kernel(
    long p,
    sArray_t<Array_view_4D_t>& sDeltaF_out,
    sArray_t<Array_view_5D_t>* sDeltaSigma_out) {
  utils::check(p >= 0 && p < _nmodes && _stored[p],
               "lr_hessian_t::rebuild_raw_kernel: mode {} was never stored.", p);
  utils::check(_has_sigma == (sDeltaSigma_out != nullptr),
               "lr_hessian_t::rebuild_raw_kernel: ΔΣ destination presence "
               "must match has_sigma.");

  _Timer.start("LR_HESS_REBUILD");
  // Every rank writes its own disjoint element slice of the node-replicated
  // window, then one allgatherv among the node roots completes every replica —
  // the same fence / barrier / complete_node_slices sequence the SCF loop's
  // mixing epilogue uses. The slices cover the array exactly once job-wide, so
  // nothing has to be zeroed first.
  {
    auto loc = sDeltaF_out.local();
    utils::check(loc.size() == _nF,
                 "lr_hessian_t::rebuild_raw_kernel: ΔF destination has {} "
                 "elements, expected {}.", loc.size(), _nF);
    auto flat = nda::reshape(loc, std::array<long, 1>{_nF});
    if (_nloc > 0) flat(nda::range(_i0, _i1)) = _dF[p];
    sDeltaF_out.win().fence();
    _mpi->node_comm.barrier();
    if (_mpi->node_comm.root())
      utils::complete_node_slices(_mpi->internode_comm, _pmap, loc.data(), _nF);
    sDeltaF_out.win().fence();
  }

  if (_has_sigma) {
    // ΔΣ(τ) comes back from this rank's own ω slab: w_to_tau contracts the
    // leading axis only and the partition never splits it, so the transform is
    // rank-local. The completion runs once per τ point, because a rank's element
    // slice is contiguous within a τ block rather than across the whole array.
    nda::array<ComplexType, 2> buf_t(_nt, _nloc);
    auto loc = sDeltaSigma_out->local();
    utils::check(loc.size() == _nt * _nF,
                 "lr_hessian_t::rebuild_raw_kernel: ΔΣ destination has {} "
                 "elements, expected {}.", loc.size(), _nt * _nF);
    auto flat = nda::reshape(loc, std::array<long, 1>{_nt * _nF});
    if (_nloc > 0) {
      _FT->w_to_tau(_Sw[p], buf_t, imag_axes_ft::fermion);
      for (long t = 0; t < _nt; ++t)
        flat(nda::range(t * _nF + _i0, t * _nF + _i1)) = buf_t(t, nda::range::all);
    }
    sDeltaSigma_out->win().fence();
    _mpi->node_comm.barrier();
    if (_mpi->node_comm.root())
      for (long t = 0; t < _nt; ++t)
        utils::complete_node_slices(_mpi->internode_comm, _pmap,
                                    loc.data() + t * _nF, _nF);
    sDeltaSigma_out->win().fence();
  }
  _mpi->comm.barrier();
  _Timer.stop("LR_HESS_REBUILD");
}


ComplexType lr_hessian_t::trace_static_local(nda::array<ComplexType, 1> const& a,
                                              nda::array<ComplexType, 1> const& b) const {
  if (_nloc == 0) return ComplexType(0.0);
  return nda::blas::dotc(a, b);
}


ComplexType lr_hessian_t::matsubara_sum(nda::array<ComplexType, 2>& c) const {
  // c is a per-rank partial over the element slice, so it is reduced before the FT.
  _mpi->comm.all_reduce_in_place_n(c.data(), c.size(), std::plus<>{});
  // (1/β) Σ_n c(iω_n) = −c(τ = β⁻); the second form is the one evaluated.
  nda::array<ComplexType, 2> c_tau(_nt, 1);
  nda::array<ComplexType, 1> c_beta(1);
  _FT->w_to_tau(c, c_tau, imag_axes_ft::fermion);
  _FT->tau_to_beta(c_tau, c_beta);
  return -c_beta(0);
}


ComplexType lr_hessian_t::trace_matsubara(nda::array<ComplexType, 2> const& Aw,
                                          nda::array<ComplexType, 2> const& Bw) const {
  nda::array<ComplexType, 2> c(_nw, 1);
  c() = ComplexType(0.0);
  // A rank owning no elements still enters matsubara_sum: the reduction is
  // collective and every rank must reach it.
  if (_nloc > 0)
    for (long n = 0; n < _nw; ++n)
      c(n, 0) = nda::blas::dotc(Aw(_refl(n), nda::range::all), Bw(n, nda::range::all));
  return matsubara_sum(c);
}


// The two terms of the functional that carry the improved solution:
//
//   static'[λ,p] += Tr  (ΔH0_λ, ΔDm'_p)
//   M'     [λ,p] += Tr  (ΔF_λ,  ΔDm'_p) + Tr_ω(ΔΣ_λ, ΔG'_p)
//
// accumulated for every λ against this one p, so ΔDm'_p / ΔG'_p never have to
// persist. That is also why every mode must already be stored.
lr_hessian_result_t lr_hessian_t::evaluate(
    lr_dyson& dyson,
    sArray_t<Array_view_4D_t>& sDeltaH0,
    sArray_t<Array_view_4D_t>& sDeltaDm,
    sArray_t<Array_view_4D_t>& sDeltaF,
    sArray_t<Array_view_5D_t>* sDeltaSigma,
    sArray_t<Array_view_5D_t>* sDeltaG,
    std::optional<nda::array<ComplexType, 5>> const& DeltaH0_mskij_root,
    bool fix_density) {

  // The accumulators are reduced and rescaled in place, so a second call would
  // return a different (wrong) answer rather than the same one.
  utils::check(!_evaluated, "lr_hessian_t::evaluate: called twice.");
  _evaluated = true;
  utils::check(!_has_sigma || (sDeltaSigma != nullptr && sDeltaG != nullptr),
               "lr_hessian_t::evaluate: a Σ-carrying kernel needs the ΔΣ and ΔG "
               "scratch arrays.");
  for (long p = 0; p < _nmodes; ++p)
    utils::check(_stored[p],
                 "lr_hessian_t::evaluate: mode {} was never stored. Every mode must "
                 "be stored before evaluation begins, because λ below runs over all "
                 "of them.", p);

  lr_hessian_result_t r;
  r.Delta_mu_improved = nda::array<double, 1>(_nmodes);
  r.Delta_mu_improved() = 0.0;

  // The equation, term by term. Every trace of it is taken in the loop below:
  //
  //   plain  [λ,p] = Tr  (ΔH0_λ, ΔDm_p )
  //   M      [λ,p] = Tr  (ΔF_λ,  ΔDm_p ) + Tr_ω(ΔΣ_λ, ΔG_p )
  //   static'[λ,p] = Tr  (ΔH0_λ, ΔDm'_p)
  //   M'     [λ,p] = Tr  (ΔF_λ,  ΔDm'_p) + Tr_ω(ΔΣ_λ, ΔG'_p)
  //
  //   H_plain = spin · plain
  //   H_sym   = spin · (static' + M' − M)
  //
  // Both triangles are built and no symmetry is assumed anywhere; the Hermiticity
  // numbers at the end are measurements of the finished matrices.
  //
  // Why the traces sit inside the p loop rather than in one double loop after it:
  // the PRIMED right operands ΔDm'_p / ΔG'_p exist only between this mode's Dyson
  // solve and the next mode's, and are never stored — keeping them for every mode
  // would be a third ω store beside _Sw and _Gw, +50% on the largest allocation
  // this object makes. The unprimed operands are in the stores and could be
  // contracted at any later point; they ride along here so the equation is written
  // once.
  //
  // trace_matsubara is COLLECTIVE, so every rank must reach both loops with the
  // same bounds. _nmodes and _has_sigma are identical on every rank, so they do.
  for (long p = 0; p < _nmodes; ++p) {

    // Mode p's ΔH0 into the shm window: root writes its slice, broadcast_to_nodes
    // publishes it. The stack is engaged on the global root only, which is where
    // the caller validated its shape.
    if (_mpi->comm.root()) {
      utils::check(DeltaH0_mskij_root.has_value(),
                   "lr_hessian_t::evaluate: the ΔH0 stack must be provided on the "
                   "MPI global root.");
      sDeltaH0.local() = (*DeltaH0_mskij_root)(p, nda::ellipsis{});
    }
    sDeltaH0.broadcast_to_nodes(0);
    _mpi->comm.barrier();

    // Mode p's raw (pre-mixing) ΔF/ΔΣ out of the stores and back into shm, which is
    // the form the Dyson solver takes its right-hand side in.
    rebuild_raw_kernel(p, sDeltaF, sDeltaSigma);

    // ΔX'_p = D[ΔH0_p + ΔF_p + ΔΣ_p], the one extra solve the stationary form
    // costs. ΔG(τ) is replicated only when the Matsubara term needs it — that
    // replication is the most expensive single step in LR.
    _Timer.start("LR_HESS_DYSON");
    r.Delta_mu_improved(p) =
        dyson.solve_lr_dyson(sDeltaDm, sDeltaH0, sDeltaF, sDeltaSigma, fix_density);
    if (_has_sigma) dyson.materialize_DeltaG_tau(*sDeltaG);
    _mpi->comm.barrier();
    _Timer.stop("LR_HESS_DYSON");

    // Column p of all four terms, against every stored λ.
    _Timer.start("LR_HESS_CONTRACT");
    auto dDm_prime = pack_static(sDeltaDm, /*weighted=*/true);
    nda::array<ComplexType, 2> Gw_prime;
    if (_has_sigma) pack_to_omega(*sDeltaG, Gw_prime, /*weighted=*/true);

    for (long l = 0; l < _nmodes; ++l) {
      _plain_stat(l, p)        += trace_static_local(_dH0[l], _dDm[p]);
      _M_stat(l, p)            += trace_static_local(_dF[l],  _dDm[p]);
      _static_prime_stat(l, p) += trace_static_local(_dH0[l], dDm_prime);
      _Mp_stat(l, p)           += trace_static_local(_dF[l],  dDm_prime);
      if (_has_sigma) {
        _M_dyn(l, p)  += trace_matsubara(_Sw[l], _Gw[p]);
        _Mp_dyn(l, p) += trace_matsubara(_Sw[l], Gw_prime);
      }
    }
    _Timer.stop("LR_HESS_CONTRACT");
    _mpi->comm.barrier();
  }

  // ---- finish: reduce, scale, combine, measure -----------------------------
  _Timer.start("LR_HESS_CONTRACT");
  const ComplexType self_pair = _has_sigma ? self_pairing() : ComplexType(0.0);
  _Timer.stop("LR_HESS_CONTRACT");

  // Only the static halves are rank-local partials. The dynamic ones came back
  // complete from trace_matsubara, which reduced them itself, so they are left
  // alone here — reducing them again would multiply them by the rank count.
  const long nm2 = _nmodes * _nmodes;
  for (auto* a : {&_plain_stat, &_static_prime_stat, &_M_stat, &_Mp_stat})
    _mpi->comm.all_reduce_in_place_n(a->data(), nm2, std::plus<>{});

  r.hessian_plain = nda::array<ComplexType, 2>(_nmodes, _nmodes);
  r.M             = nda::array<ComplexType, 2>(_nmodes, _nmodes);
  r.M_prime       = nda::array<ComplexType, 2>(_nmodes, _nmodes);
  r.static_prime  = nda::array<ComplexType, 2>(_nmodes, _nmodes);
  r.hessian_sym   = nda::array<ComplexType, 2>(_nmodes, _nmodes);

  const ComplexType sf(_spin_factor);
  for (long l = 0; l < _nmodes; ++l)
    for (long p = 0; p < _nmodes; ++p) {
      r.hessian_plain(l, p) = sf * _plain_stat(l, p);
      r.static_prime(l, p)  = sf * _static_prime_stat(l, p);
      r.M(l, p)             = sf * (_M_stat(l, p)  + _M_dyn(l, p));
      r.M_prime(l, p)       = sf * (_Mp_stat(l, p) + _Mp_dyn(l, p));
      r.hessian_sym(l, p)   = r.static_prime(l, p) + r.M_prime(l, p) - r.M(l, p);
    }

  r.herm_plain = herm_dev(r.hessian_plain);
  r.herm_sym   = herm_dev(r.hessian_sym);
  r.herm_M     = herm_dev(r.M);

  double dev = 0.0, nrm = 0.0;
  for (long l = 0; l < _nmodes; ++l)
    for (long p = 0; p < _nmodes; ++p) {
      dev += std::norm(r.hessian_sym(l, p) - r.hessian_plain(l, p));
      nrm += std::norm(r.hessian_plain(l, p));
    }
  const double rel_sym_plain = (nrm > 0.0) ? std::sqrt(dev / nrm) : 0.0;

  app_log(1, "\n  Stationary free-energy-hessian diagnostics");
  app_log(1, "  -----------------------------------------");
  app_log(1, "    ||H_sym - H_sym^dag|| / ||H_sym||  = {:.3e}   [H1: mixed instead "
             "of raw ΔF/ΔΣ]", r.herm_sym);
  app_log(1, "    ||M - M^dag|| / ||M||              = {:.3e}   [H1, direct]",
          r.herm_M);
  // M is Hermitian exactly insofar as K is self-adjoint under this pairing, and
  // the same self-adjointness is what makes the estimator second order. So this
  // residual BOUNDS the accuracy hessian_sym can reach, and a non-tiny value has to
  // appear in the log next to the number it bounds rather than be read as noise.
  if (r.herm_M > 1e-6)
    app_log(1, "    [WARNING] ||M - M^dag|| is not at roundoff. M is Hermitian only "
               "as far as the kernel is self-adjoint under this pairing, and the "
               "estimator is second order only as far as the same holds — so "
               "~{:.1e} bounds the relative accuracy hessian_sym can reach here. Check "
               "the imaginary-time grid and, for a GW kernel, the -G(.)dW channel.",
            r.herm_M);
  // Away from convergence this is the size of the correction, and large is the
  // point. It is a DETECTOR only at convergence, where the correction cancels
  // identically: a nonzero value there means the extra Dyson applied a different
  // operator than the original solve (a frozen Δμ, say), which Hermiticity cannot
  // see.
  app_log(1, "    ||H_sym - H_plain|| / ||H_plain||  = {:.3e}   [the correction; "
             "-> 0 at convergence, where it is the D5 detector]", rel_sym_plain);
  app_log(1, "    ||H_plain - H_plain^dag||/||H_plain|| = {:.3e}", r.herm_plain);
  if (_has_sigma) {
    app_log(1, "    <A,A> on the stored ΔΣ_0           = {:.6e} + {:.3e}i "
               "(must be real and positive)", self_pair.real(), self_pair.imag());
    const bool ok = self_pair.real() > 0.0 &&
                    std::abs(self_pair.imag()) <= 1e-6 * std::abs(self_pair.real());
    if (!ok)
      app_log(1, "    [WARNING] <A,A> is not real and positive: the Matsubara pairing "
                 "or the ω reflection is wrong.");
  }
  app_log(1, "");

  return r;
}


ComplexType lr_hessian_t::self_pairing() const {
  // ⟨A,A⟩ on a genuinely dynamic operand. Relabelling n → refl(n) conjugates it,
  // so it is real by construction; the sign is what a wrong pairing breaks. A
  // static operand cannot serve here — a constant has no fermionic FT.
  nda::array<ComplexType, 2> sp(_nw, 1);
  sp() = ComplexType(0.0);
  if (_nloc > 0) {
    nda::array<ComplexType, 1> bw(_nloc);
    for (long n = 0; n < _nw; ++n) {
      for (long l = 0; l < _nloc; ++l) bw(l) = _Sw[0](n, l) * _wloc(l);
      sp(n, 0) = nda::blas::dotc(_Sw[0](_refl(n), nda::range::all), bw);
    }
  }
  return matsubara_sum(sp);
}


void lr_hessian_t::print_timers(double pert_refresh_sec, int pert_refresh_calls) {
  app_log(2, "\n  LR hessian timers");
  app_log(2, "  -----------------");
  app_log(2, "    Store (pack + τ→ω):        {0:8.3f} sec  {1:4d} calls",
          _Timer.elapsed("LR_HESS_STORE"), _Timer.number_of_calls("LR_HESS_STORE"));
  app_log(2, "    K_pert refresh (per mode): {0:8.3f} sec  {1:4d} calls",
          pert_refresh_sec, pert_refresh_calls);
  app_log(2, "    Rebuild raw ΔF/ΔΣ in shm:  {0:8.3f} sec  {1:4d} calls",
          _Timer.elapsed("LR_HESS_REBUILD"), _Timer.number_of_calls("LR_HESS_REBUILD"));
  app_log(2, "    Extra Dyson (per mode):    {0:8.3f} sec  {1:4d} calls",
          _Timer.elapsed("LR_HESS_DYSON"), _Timer.number_of_calls("LR_HESS_DYSON"));
  app_log(2, "    Pair contractions:         {0:8.3f} sec  {1:4d} calls\n",
          _Timer.elapsed("LR_HESS_CONTRACT"), _Timer.number_of_calls("LR_HESS_CONTRACT"));
}

} // namespace methods
