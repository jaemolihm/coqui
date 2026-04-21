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

#ifndef METHODS_ERI_LR_THC_HPP
#define METHODS_ERI_LR_THC_HPP

// Linear response of the THC auxiliary basis.
//
// Given the unperturbed THC decomposition
//   C^q · Xi^q = Z^q,
//   Z^q_{μr} = Σ_k P^{k+q}(r_μ, r) · P^k(r, r_μ),
//   C^q_{μν} = Z^q_{μ, r_ν},
// this class computes the first-order response δΞ^q from
//   C^q · δΞ^q = δZ^q − δC^q · Ξ^q,
// with
//   δP^k(r,r') = Σ_n [ δψ^k_n(r) ψ^{k*}_n(r') + ψ^k_n(r) δψ^{k*}_n(r') ].
//
// The linear response of the interpolating points r_μ (Δr_μ) is supplied by the
// caller as delta_r_P. Passing all-zeros recovers the fixed-r_μ case. δψ is read
// from per-k HDF5 files {Deltapsi_prefix}_ik{ik+1}.hdf5, same format as compute_delta_X.
//
// Implementation strategy: per (is, ik) we decode ψ_a, δψ_a on the full FFT mesh
// and (δr_μ·∇)ψ_a at the pivot points, then build three per-k caches T, δT, and
// δr·∇₂T. These k-sums are chunked across ranks with itertools::chunk_range and
// Allreduced so every rank carries the full caches. Stage B (Z_r, δZ_r, C_q, δC_q
// accumulation and the dense LU solve) then runs deterministically on every rank.
// All public-facing returns are replicated — no final Broadcast is needed.

#include <string>
#include <tuple>
#include "configuration.hpp"
#include "utilities/Timer.hpp"
#include "nda/nda.hpp"

#include "thc_reader_t.hpp"

namespace methods {

class lr_thc {
public:
  // Holds a reference to a thc_reader_t. The reader provides the saved
  // interpolating-point indices ri() so compute_delta_Xi reuses the *exact*
  // pivots the reader was built/read with (MPI-independent). rho_g and vG are
  // accessed via reader.rho_g() and reader.vG() (persistent copies captured
  // from the thc builder at init time).
  explicit lr_thc(thc_reader_t& reader) : _reader(reader) {
    for (auto& v : {"build_kcache", "xi_Deltaxi_one_q", "build_dV_from_xi"})
      Timer.add(v);
  }

  ~lr_thc() = default;
  lr_thc(lr_thc const&) = delete;
  lr_thc& operator=(lr_thc const&) = delete;

  // Compute Ξ^q and δΞ^q for a single q-vector.
  //
  //  Deltapsi_prefix  : per-k files are {Deltapsi_prefix}_ik{ik+1}.hdf5 (QE-format MillerIndices + evc).
  //  q_vec_cryst      : (3,) internal q-vector of V^q / Ξ^q in crystal coordinates.
  //                     Must lie on the BZ k-grid.
  //  delta_r_P        : (nIpts, 3) linear response of interpolating points r_μ in
  //                     Cartesian Bohr. Pass zeros for the fixed-r_μ case.
  //  q_phonon_cryst   : (3,) phonon wavevector Q of δψ in crystal coordinates.
  //                     Defaults to Γ = (0,0,0). The stored δψ has Bloch momentum
  //                     k + Q_phonon, independent of the internal q.
  //
  // Returns a tuple (Ξ^q, δΞ^q), each a (nIpts, nnr) complex array replicated on
  // every rank. Ξ^q is computed self-consistently inside this routine from the
  // same C_q and Z_q that enter the LR equation; returning it alongside δΞ^q
  // lets callers verify the THC identity
  //   ψ^{k+q*}_a(r) ψ^k_b(r) ≈ Σ_μ conj(Xa[k+q,μ,a]) Xa[k,μ,b] Ξ^q_μ(r)
  // with the SAME Ξ that δΞ is the linear response of.
  std::tuple<nda::array<ComplexType, 2>, nda::array<ComplexType, 2>>
  compute_delta_Xi(
      std::string const& Deltapsi_prefix,
      nda::array<double, 1> const& q_vec_cryst,
      nda::array<double, 2> const& delta_r_P,
      nda::array<double, 1> const& q_phonon_cryst = nda::array<double, 1>{0.0, 0.0, 0.0});

  // Compute δV^q_{μν} in the THC auxiliary basis.
  //
  //   δV^q_{μν} = ∫∫ dr dr' [ δξ^{-q}_μ(r) v(r,r') ξ^q_ν(r')
  //                         + ξ^{-q}_μ(r) v(r,r') δξ^q_ν(r') ]
  //
  // In CoQuí's THC G-space form (see thc::intvec_impl in thc.icc):
  //   Z^q_{μG} = ξ^q_μ(G+q) · sqrt(v(G+q))
  //   δV^q     = (δZ · Z^† + Z · δZ^†) / (Vol · N_k)
  //
  // V^q itself is NOT returned — it is already available from the ordinary THC
  // decomposition carried by the reader.
  //
  // Arguments mirror compute_delta_Xi. Returns a (Np, Np) complex array,
  // replicated on every rank.
  nda::array<ComplexType, 2> compute_delta_V(
      std::string const& Deltapsi_prefix,
      nda::array<double, 1> const& q_vec_cryst,
      nda::array<double, 2> const& delta_r_P,
      nda::array<double, 1> const& q_phonon_cryst = nda::array<double, 1>{0.0, 0.0, 0.0});

  // Batched version of compute_delta_V: computes δV^q for all internal q's in
  // q_vec_cryst_list in a single call, amortizing the q-independent wfc/δψ I/O
  // and FFT work (stage A) across all q's.
  //
  //   q_vec_cryst_list : (nq, 3) list of internal q-vectors in crystal coords.
  //   delta_r_P        : (Np, 3) linear response of interpolating points r_μ
  //                      in Cartesian Bohr.
  //   q_phonon_cryst   : (3,) phonon wavevector Q of δψ in crystal coords
  //                      (shared across all internal q's).
  //
  // Returns (nq, Np, Np) complex. Replicated on every rank.
  nda::array<ComplexType, 3> compute_delta_V_batched(
      std::string const& Deltapsi_prefix,
      nda::array<double, 2> const& q_vec_cryst_list,
      nda::array<double, 2> const& delta_r_P,
      nda::array<double, 1> const& q_phonon_cryst = nda::array<double, 1>{0.0, 0.0, 0.0});

private:
  // Shared internals. Builds Ξ^q(G) and δΞ^q(G) on the reduced G-grid used by
  // rho_g, plus auxiliary info needed to either scatter to the full FFT mesh
  // (for compute_delta_Xi) or apply v(G+q) and contract (for compute_delta_V).
  struct xi_Deltaxi_on_Gred {
    nda::array<ComplexType, 2> Xi;         // (Np, nG_red)
    nda::array<ComplexType, 2> DeltaXi;    // (Np, nG_red)
    nda::array<long, 1>        gv_to_fft;  // (nG_red,) indices into flat FFT grid
    long                       N1 = 0, N2 = 0, N3 = 0;  // full FFT mesh
    long                       iq_target = -1;          // index in BZ k-grid (== Qpts)
  };

  // q-independent per-k cache of T(k,μ,r) and δT(k,μ,r), plus metadata needed
  // for stage B (the per-q Z/δZ accumulation → LU solve).
  //
  // Large: nspin · nkpts · Np · nnr complex per array. Held in full on every
  // rank (after stage A's Allreduce) so the stage B loop over q can execute
  // without additional comms per q.
  struct kcache_t {
    nda::array<ComplexType, 4> T_sk_ur;      // (nspin, nkpts, Np, nnr)
    nda::array<ComplexType, 4> DeltaT_sk_ur; // (nspin, nkpts, Np, nnr)
    // δr_ν · ∇_2 T^k(r_μ, r_ν) at pivot pairs (ν-pivot shift of T evaluated at
    // r_ν). Directly replaces the old FFT-based pivot-shift correction in
    // xi_Deltaxi_one_q: δC_q gets Σ_k [conj(·)_{k+q}·T_k + conj(T)_{k+q}·(·)_k]
    // summed from this cache instead of ∂_α Z_r reconstructed via FFT.
    nda::array<ComplexType, 4> DeltaT_grad2_sk_PP; // (nspin, nkpts, Np, Np)
    // Stage-B-shared scalars and arrays.
    long N1 = 0, N2 = 0, N3 = 0;             // FFT mesh
    long nnr = 0;
    long Np = 0;
    long nspin = 0;
    long nkpts = 0;
    long nG_red = 0;
    nda::array<long, 1> gv_to_fft;           // (nG_red,)
    nda::array<double, 2> kpts_crys;         // (nkpts, 3)
    nda::array<double, 2> recv;              // (3, 3) reciprocal vectors
    nda::array<double, 2> lattv;             // (3, 3) real-space vectors
  };
  kcache_t build_kcache(
      std::string const& Deltapsi_prefix,
      nda::array<double, 2> const& delta_r_P,
      nda::array<double, 1> const& q_phonon_cryst);

  // Stage B+LU: given the q-independent cache, compute Ξ^q and δΞ^q on the
  // reduced G-grid for a single internal q.
  xi_Deltaxi_on_Gred xi_Deltaxi_one_q(
      kcache_t const& cache,
      nda::array<double, 1> const& q_vec_cryst);

  // Stage C (shared by scalar compute_delta_V and compute_delta_V_batched):
  // turn Ξ^q, δΞ^q on the reduced G-grid into δV^q via √v(G+q) scaling + gemm.
  // Mutates g_out (scales Xi/DeltaXi by √v(G+q)); caller should discard g_out
  // afterwards.
  void build_dV_from_xi(
      xi_Deltaxi_on_Gred& g_out,
      nda::array<ComplexType, 2>& DeltaV_out);

  // Emit per-stage elapsed times to app_log(4). Called at the end of each
  // top-level public API (compute_delta_Xi / _V / _V_batched).
  void print_timers();

  thc_reader_t& _reader;
  utils::TimerManager Timer;
};

} // namespace methods

#endif // METHODS_ERI_LR_THC_HPP
