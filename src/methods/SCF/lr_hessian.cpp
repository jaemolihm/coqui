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
                  "LR_HESS_CONTRACT", "LR_HESS_FT"})
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

  if (_has_sigma) {
    for (auto* a : {&_M_mats, &_Mp_mats}) {
      *a = nda::array<ComplexType, 3>(_nw, _nmodes, _nmodes);
      (*a)() = ComplexType(0.0);
    }
  }

  for (auto* a : {&_plain_stat, &_static_prime_stat, &_M_stat, &_Mp_stat}) {
    *a = nda::array<ComplexType, 2>(_nmodes, _nmodes);
    (*a)() = ComplexType(0.0);
  }

  _stored.assign(_nmodes, false);
  _improved.assign(_nmodes, false);

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
  _dH0[p]  = pack_static(sDeltaH0, /*weighted=*/false);   // left  of plain, static2
  _dF[p]   = pack_static(sDeltaF,  /*weighted=*/false);   // left  of N, M2
  _dDm[p] = pack_static(sDeltaDm, /*weighted=*/true);    // right of plain, M
  if (_has_sigma) {
    pack_to_omega(*sDeltaSigma, _Sw[p], /*weighted=*/false);  // left  of N, M2
    pack_to_omega(*sDeltaG,     _Gw[p], /*weighted=*/true);   // right of N
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


void lr_hessian_t::trace_matsubara_local(nda::array<ComplexType, 2> const& Aw,
                                       nda::array<ComplexType, 2> const& Bw,
                                       nda::array_view<ComplexType, 1> acc) const {
  if (_nloc == 0) return;
  for (long n = 0; n < _nw; ++n)
    acc(n) += nda::blas::dotc(Aw(_refl(n), nda::range::all), Bw(n, nda::range::all));
}


// The two terms of the functional that carry the improved solution:
//
//   static'[λ,p] += Tr  (ΔH0_λ, ΔDm'_p)
//   M'     [λ,p] += Tr  (ΔF_λ,  ΔDm'_p) + Tr_ω(ΔΣ_λ, ΔG'_p)
//
// accumulated for every λ against this one p, so ΔDm'_p / ΔG'_p never have to
// persist. That is also why every mode must already be stored.
void lr_hessian_t::accumulate_improved(long p,
                                       sArray_t<Array_view_4D_t> const& sDeltaDm,
                                       sArray_t<Array_view_5D_t> const* sDeltaG) {
  utils::check(p >= 0 && p < _nmodes && _stored[p],
               "lr_hessian_t::accumulate_improved: mode {} was never stored.", p);
  utils::check(!_improved[p],
               "lr_hessian_t::accumulate_improved: mode {} contracted twice.", p);
  utils::check(!_has_sigma || sDeltaG != nullptr,
               "lr_hessian_t::accumulate_improved: a Σ-carrying kernel needs ΔG'.");
  for (long l = 0; l < _nmodes; ++l)
    utils::check(_stored[l],
                 "lr_hessian_t::accumulate_improved: every mode must be stored in "
                 "pass 1 before any pair is contracted; mode {} is missing.", l);

  _Timer.start("LR_HESS_CONTRACT");
  auto dDm2 = pack_static(sDeltaDm, /*weighted=*/true);
  nda::array<ComplexType, 2> G2w;
  if (_has_sigma) pack_to_omega(*sDeltaG, G2w, /*weighted=*/true);

  for (long l = 0; l < _nmodes; ++l) {
    _static_prime_stat(l, p) += trace_static_local(_dH0[l], dDm2);
    _Mp_stat(l, p)      += trace_static_local(_dF[l], dDm2);
    if (_has_sigma)
      trace_matsubara_local(_Sw[l], G2w, _Mp_mats(nda::range::all, l, p));
  }
  _improved[p] = true;
  _Timer.stop("LR_HESS_CONTRACT");
}


nda::array<double, 1> lr_hessian_t::solve_improved(
    lr_dyson& dyson,
    sArray_t<Array_view_4D_t>& sDeltaH0,
    sArray_t<Array_view_4D_t>& sDeltaDm,
    sArray_t<Array_view_4D_t>& sDeltaF,
    sArray_t<Array_view_5D_t>* sDeltaSigma,
    sArray_t<Array_view_5D_t>* sDeltaG,
    std::optional<nda::array<ComplexType, 5>> const& DeltaH0_mskij_root,
    bool fix_density) {

  utils::check(!_has_sigma || (sDeltaSigma != nullptr && sDeltaG != nullptr),
               "lr_hessian_t::solve_improved: a Σ-carrying kernel needs the ΔΣ and "
               "ΔG scratch arrays.");
  for (long p = 0; p < _nmodes; ++p)
    utils::check(_stored[p],
                 "lr_hessian_t::solve_improved: mode {} was never stored; pass 1 "
                 "must cover every mode before pass 2 begins.", p);

  nda::array<double, 1> Delta_mu(_nmodes);
  Delta_mu() = 0.0;

  for (long p = 0; p < _nmodes; ++p) {
    // Refill the ΔH0 window from this mode's slice: root writes, then
    // broadcast_to_nodes publishes it. The stack is engaged on the global root
    // only, which is where the caller validated it.
    if (_mpi->comm.root()) {
      utils::check(DeltaH0_mskij_root.has_value(),
                   "lr_hessian_t::solve_improved: the ΔH0 stack must be provided on "
                   "the MPI global root.");
      sDeltaH0.local() = (*DeltaH0_mskij_root)(p, nda::ellipsis{});
    }
    sDeltaH0.broadcast_to_nodes(0);
    _mpi->comm.barrier();

    rebuild_raw_kernel(p, sDeltaF, sDeltaSigma);

    // ΔX'_p = D[ΔH0_p + ΔF_p + ΔΣ_p], the one extra solve the stationary form
    // costs. ΔG(τ) is replicated only when the Matsubara term needs it — it is the
    // most expensive step in LR.
    _Timer.start("LR_HESS_DYSON");
    Delta_mu(p) = dyson.solve_lr_dyson(sDeltaDm, sDeltaH0, sDeltaF, sDeltaSigma,
                                       fix_density);
    if (_has_sigma) dyson.materialize_DeltaG_tau(*sDeltaG);
    _mpi->comm.barrier();
    _Timer.stop("LR_HESS_DYSON");

    accumulate_improved(p, sDeltaDm, _has_sigma ? sDeltaG : nullptr);
    _mpi->comm.barrier();
  }
  return Delta_mu;
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
  _mpi->comm.all_reduce_in_place_n(sp.data(), sp.size(), std::plus<>{});
  nda::array<ComplexType, 2> sp_t(_nt, 1);
  nda::array<ComplexType, 1> sp_b(1);
  _FT->w_to_tau(sp, sp_t, imag_axes_ft::fermion);
  _FT->tau_to_beta(sp_t, sp_b);
  return -sp_b(0);
}


lr_hessian_result_t lr_hessian_t::assemble() {
  // The accumulators are reduced and rescaled in place, so a second call would
  // return a different (wrong) answer rather than the same one.
  utils::check(!_assembled, "lr_hessian_t::assemble: called twice.");
  _assembled = true;
  for (long p = 0; p < _nmodes; ++p)
    utils::check(_stored[p] && _improved[p],
                 "lr_hessian_t::assemble: mode {} is missing its pass-1 store "
                 "or its pass-2 improved solution.", p);

  // The equation this function evaluates, term by term:
  //
  //   plain  [λ,p] = Tr  (ΔH0_λ, ΔDm_p )                        <- here
  //   M      [λ,p] = Tr  (ΔF_λ,  ΔDm_p ) + Tr_ω(ΔΣ_λ, ΔG_p )    <- here
  //   static'[λ,p] = Tr  (ΔH0_λ, ΔDm'_p)                        <- accumulate_improved
  //   M'     [λ,p] = Tr  (ΔF_λ,  ΔDm'_p) + Tr_ω(ΔΣ_λ, ΔG'_p)    <- accumulate_improved
  //
  //   H_plain = spin * plain
  //   H_sym   = spin * (static' + M' - M)
  //
  // The two terms that need only pass-1 stores are accumulated here, over ALL mode
  // pairs and both triangles. No symmetry is used to build any entry — the
  // Hermiticity numbers below are measurements of the finished matrices.
  _Timer.start("LR_HESS_CONTRACT");
  for (long l = 0; l < _nmodes; ++l)
    for (long p = 0; p < _nmodes; ++p) {
      _plain_stat(l, p) += trace_static_local(_dH0[l], _dDm[p]);
      _M_stat(l, p)     += trace_static_local(_dF[l], _dDm[p]);
      if (_has_sigma)
        trace_matsubara_local(_Sw[l], _Gw[p], _M_mats(nda::range::all, l, p));
    }

  const ComplexType self_pair = _has_sigma ? self_pairing() : ComplexType(0.0);
  _Timer.stop("LR_HESS_CONTRACT");

  // Every term's static half is striped over comm, so each needs one reduction.
  const long nm2 = _nmodes * _nmodes;
  for (auto* a : {&_plain_stat, &_static_prime_stat, &_M_stat, &_Mp_stat})
    _mpi->comm.all_reduce_in_place_n(a->data(), nm2, std::plus<>{});

  // Matsubara tail: (1/β) Σ_n c(iω_n) = −c(τ=β⁻). Replicated on every rank, so
  // assemble() returns the same matrices everywhere.
  nda::array<ComplexType, 2> M_mats(_nmodes, _nmodes), Mp_mats(_nmodes, _nmodes);
  M_mats()  = ComplexType(0.0);
  Mp_mats() = ComplexType(0.0);
  if (_has_sigma) {
    _Timer.start("LR_HESS_FT");
    nda::array<ComplexType, 2> tail(_nt, nm2);
    // Tr_ω(A,B) = (1/β) Σ_n c(iω_n) = −c(τ=β⁻): reduce the per-ω accumulator, then
    // take it to τ = β⁻ and flip the sign.
    auto matsubara_tail = [&](nda::array<ComplexType, 3>& acc,
                              nda::array<ComplexType, 2>& out) {
      _mpi->comm.all_reduce_in_place_n(acc.data(), acc.size(), std::plus<>{});
      auto acc_2d = nda::reshape(acc, std::array<long, 2>{_nw, nm2});
      _FT->w_to_tau(acc_2d, tail, imag_axes_ft::fermion);
      auto out_1d = nda::reshape(out, std::array<long, 1>{nm2});
      _FT->tau_to_beta(tail, out_1d);
      for (long i = 0; i < nm2; ++i) out_1d(i) *= ComplexType(-1.0);
    };
    matsubara_tail(_M_mats, M_mats);
    matsubara_tail(_Mp_mats, Mp_mats);
    _Timer.stop("LR_HESS_FT");
  }

  lr_hessian_result_t r;
  r.hessian_plain = nda::array<ComplexType, 2>(_nmodes, _nmodes);
  r.M        = nda::array<ComplexType, 2>(_nmodes, _nmodes);
  r.M_prime       = nda::array<ComplexType, 2>(_nmodes, _nmodes);
  r.static_prime  = nda::array<ComplexType, 2>(_nmodes, _nmodes);
  r.hessian_sym   = nda::array<ComplexType, 2>(_nmodes, _nmodes);

  const ComplexType sf(_spin_factor);
  for (long l = 0; l < _nmodes; ++l)
    for (long p = 0; p < _nmodes; ++p) {
      r.hessian_plain(l, p) = sf * _plain_stat(l, p);
      r.static_prime(l, p)       = sf * _static_prime_stat(l, p);
      r.M(l, p)             = sf * (_M_stat(l, p)  + M_mats(l, p));
      r.M_prime(l, p)            = sf * (_Mp_stat(l, p) + Mp_mats(l, p));
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
  app_log(1, "    ||N - N^dag|| / ||N||              = {:.3e}   [H1, direct]",
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
  // operator than pass 1 (a frozen Δμ, say), which Hermiticity cannot see.
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
  app_log(2, "    Pair contractions:         {0:8.3f} sec  {1:4d} calls",
          _Timer.elapsed("LR_HESS_CONTRACT"), _Timer.number_of_calls("LR_HESS_CONTRACT"));
  app_log(2, "    Matsubara tail (FT):       {0:8.3f} sec  {1:4d} calls\n",
          _Timer.elapsed("LR_HESS_FT"), _Timer.number_of_calls("LR_HESS_FT"));
}

} // namespace methods
