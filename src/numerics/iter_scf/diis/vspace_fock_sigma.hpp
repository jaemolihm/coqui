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


#ifndef COQUI_VSPACE_FOCK_SIGMA_HPP
#define COQUI_VSPACE_FOCK_SIGMA_HPP

#include <algorithm>

#include "numerics/iter_scf/diis/vspace.h"
#include "numerics/iter_scf/diis/diis_timers.hpp"
namespace iter_scf {

// Implementation of vector algebra for Fock matrix and self-energy union
// The class must be initialized before usage

class FockSigma {
private:
    using Array_4D = nda::array<ComplexType, 4>;
    using Array_5D = nda::array<ComplexType, 5>;

    Array_4D _Fock;
    Array_5D _Sigma;
    double _mu;
    bool inited_F = false;
    bool inited_S = false;
    bool inited_mu = false;
public:
    FockSigma() {}
    FockSigma(const FockSigma & rhs) {
        diis_timers::fs_copies.start();
        _Fock = rhs._Fock;
        _Sigma = rhs._Sigma;
        _mu = rhs._mu;
        diis_timers::fs_copies.stop();
        inited_F = true;
        inited_S = true;
        inited_mu = true;
    }

    FockSigma(const Array_4D& Fock_, const Array_5D& Sigma_, const double mu_) {
        diis_timers::fs_copies.start();
        _Fock = Fock_;
        _Sigma = Sigma_;
        _mu = mu_;
        diis_timers::fs_copies.stop();
        inited_F = true;
        inited_S = true;
        inited_mu = true;
    }

    FockSigma& operator =(const FockSigma& rhs) {
      diis_timers::fs_copies.start();
      _Fock = rhs._Fock;
      _Sigma = rhs._Sigma;
      _mu = rhs._mu;
      diis_timers::fs_copies.stop();
        inited_F = true;
        inited_S = true;
        inited_mu = true;
      return *this;
    }

    // Move ctor/assign steal the nda buffers (no whole-vector copy), enabling
    // std::move into VSpace/opt_state on the last use of a temporary.
    FockSigma(FockSigma&&) = default;
    FockSigma& operator =(FockSigma&&) = default;

    ComplexType dot_prod(const FockSigma& rhs) const {
      utils::check(inited_F, "FockSigma: Fock matrix is not initialized");
      utils::check(inited_S, "FockSigma: Sigma is not initialized");
      size_t Fdim = std::reduce(_Fock.shape().begin(), _Fock.shape().end(), 1, std::multiplies<size_t>());
      size_t Sdim = std::reduce(_Sigma.shape().begin(), _Sigma.shape().end(), 1, std::multiplies<size_t>());
/*
      auto vec_F= nda::reshape(_Fock, std::array<long, 1>{Fdim});
      auto vec_S= nda::reshape(_Sigma, std::array<long, 1>{Sdim});
*/
      auto matvec_F= nda::reshape(_Fock, std::array<long, 2>{Fdim, 1});
      auto matvec_S= nda::reshape(_Sigma, std::array<long, 2>{Sdim, 1});

      const auto& rFock = rhs.get_fock();
      const auto& rSigma = rhs.get_sigma();
      size_t rFdim = std::reduce(rFock.shape().begin(), rFock.shape().end(), 1, std::multiplies<size_t>());
      size_t rSdim = std::reduce(rSigma.shape().begin(), rSigma.shape().end(), 1, std::multiplies<size_t>());
      auto matvec_rF= nda::reshape(rFock, std::array<long, 2>{rFdim, 1});
      auto matvec_rS= nda::reshape(rSigma, std::array<long, 2>{rSdim, 1});
      // The conjugated copies are materialized on purpose: routing the conjugation
      // into the zgemm as the 'C' op selects a different MKL kernel whose rounding
      // differs, breaking digit-identity of the DIIS extrapolation.
      nda::array<ComplexType, 2> res1(1,1);
      nda::array<ComplexType, 2> res2(1,1);
      nda::blas::gemm(nda::make_regular(nda::conj(nda::transpose(matvec_F))), matvec_rF, res1);
      nda::blas::gemm(nda::make_regular(nda::conj(nda::transpose(matvec_S))), matvec_rS, res2);
      return res1(0,0) + res2(0,0);
    }

    // H7: a materialized conjugate "handle" for this vector — the (1,N) conjugated
    // rows of the flattened Fock and Sigma, i.e. exactly the LHS buffers dot_prod
    // builds internally. Caching it lets update_overlaps compute an entire B-row
    // (<x_i|u> for all subspace vectors x_i) with one conj(u) materialization reused
    // across every i, instead of one 1.8 GB conj copy per pair.
    struct ConjFlat {
        nda::array<ComplexType, 2> Frow; // (1, Fdim) = conj(flatten(Fock))^T
        nda::array<ComplexType, 2> Srow; // (1, Sdim) = conj(flatten(Sigma))^T
    };

    ConjFlat make_conj_flat() const {
      utils::check(inited_F, "FockSigma: Fock matrix is not initialized");
      utils::check(inited_S, "FockSigma: Sigma is not initialized");
      size_t Fdim = std::reduce(_Fock.shape().begin(), _Fock.shape().end(), 1, std::multiplies<size_t>());
      size_t Sdim = std::reduce(_Sigma.shape().begin(), _Sigma.shape().end(), 1, std::multiplies<size_t>());
      auto matvec_F = nda::reshape(_Fock, std::array<long, 2>{Fdim, 1});
      auto matvec_S = nda::reshape(_Sigma, std::array<long, 2>{Sdim, 1});
      ConjFlat h;
      h.Frow = nda::make_regular(nda::conj(nda::transpose(matvec_F)));
      h.Srow = nda::make_regular(nda::conj(nda::transpose(matvec_S)));
      return h;
    }

    // H7: computes <u|x> where *this == x and u_conj == u.make_conj_flat().
    // The gemm signature/shapes/kernel are IDENTICAL to dot_prod's (plain zgemm,
    // no 'C'/'T' op) — only the pre-materialized conjugation moves from the LHS
    // buffer to the operand carried by u_conj — so <u|x> is bitwise conj(<x|u>),
    // and conj(<u|x_i>) reproduces the per-pair overlap <x_i|u> to the last digit.
    ComplexType dot_prod_conj_lhs(const ConjFlat& u_conj) const {
      utils::check(inited_F, "FockSigma: Fock matrix is not initialized");
      utils::check(inited_S, "FockSigma: Sigma is not initialized");
      size_t Fdim = std::reduce(_Fock.shape().begin(), _Fock.shape().end(), 1, std::multiplies<size_t>());
      size_t Sdim = std::reduce(_Sigma.shape().begin(), _Sigma.shape().end(), 1, std::multiplies<size_t>());
      auto matvec_F = nda::reshape(_Fock, std::array<long, 2>{Fdim, 1});
      auto matvec_S = nda::reshape(_Sigma, std::array<long, 2>{Sdim, 1});
      nda::array<ComplexType, 2> res1(1,1);
      nda::array<ComplexType, 2> res2(1,1);
      nda::blas::gemm(u_conj.Frow, matvec_F, res1);
      nda::blas::gemm(u_conj.Srow, matvec_S, res2);
      return res1(0,0) + res2(0,0);
    }

    const Array_4D& get_fock() const {
        utils::check(inited_F, "FockSigma: Fock matrix is not initialized");
        return _Fock;
    }
    const Array_5D& get_sigma() const {
        utils::check(inited_S, "FockSigma: Sigma is not initialized");
        return _Sigma;
    }
    double get_mu() const {
        utils::check(inited_mu, "FockSigma: mu is not initialized");
        return _mu;
    }

    void set_mu(double mu) { _mu = mu; inited_mu = true;}
    void set_fock(Array_4D& F_) {
        _Fock = F_;
        inited_F = true;
    }
    void set_sigma(Array_5D& S_) {
        _Sigma = S_;
        inited_S = true;
    }
    // Move overload: steals the buffer of a temporary/consumed Sigma (no copy).
    void set_sigma(Array_5D&& S_) {
        _Sigma = std::move(S_);
        inited_S = true;
    }
    void set_fock_sigma(Array_4D& F_, Array_5D& S_) {
        set_fock(F_);
        set_sigma(S_);
    }
    void set_fock_sigma(Array_4D& F_, Array_5D&& S_) {
        set_fock(F_);
        set_sigma(std::move(S_));
    }

    void set_zero() {
        _Fock() = 0;
        _Sigma() = 0;
        _mu = 0;
        inited_F = true;
        inited_S = true;
        inited_mu = true;
    }

    FockSigma operator*=(std::complex<double> c)  {
        utils::check(inited_F, "FockSigma: Fock matrix is not initialized");
        utils::check(inited_S, "FockSigma: Sigma is not initialized");
        _Fock *= c;
        _Sigma *= c;
        return *this;
    }

    FockSigma operator+=(FockSigma & vec)  {
        utils::check(inited_F, "FockSigma: Fock matrix is not initialized");
        utils::check(inited_S, "FockSigma: Sigma is not initialized");
        _Fock += vec.get_fock();
        _Sigma += vec.get_sigma();
        return *this;
    }

    FockSigma operator+=(FockSigma && vec)  {
        utils::check(inited_F, "FockSigma: Fock matrix is not initialized");
        utils::check(inited_S, "FockSigma: Sigma is not initialized");
        _Fock += vec.get_fock();
        _Sigma += vec.get_sigma();
        return *this;
    }

    void add(FockSigma&& a, ComplexType c) {
        utils::check(inited_F, "FockSigma: Fock matrix is not initialized");
        utils::check(inited_S, "FockSigma: Sigma is not initialized");
        _Fock += c * a.get_fock();
        _Sigma += c * a.get_sigma();
    }

    void add(const FockSigma& a, ComplexType c) {
        utils::check(inited_F, "FockSigma: Fock matrix is not initialized");
        utils::check(inited_S, "FockSigma: Sigma is not initialized");
        _Fock += c * a.get_fock();
        _Sigma += c * a.get_sigma();
    }

    void read_from_file(std::string filename, const size_t vec_number) {
        h5::file file(filename, 'r');
        auto vec_grp = h5::group(file).open_group("vec" + std::to_string(vec_number));
        
        nda::h5_read(vec_grp, "Sigma_tskij", _Sigma);
        nda::h5_read(vec_grp, "F_skij", _Fock);
        h5::h5_read(vec_grp, "mu", _mu);
        inited_F = true;
        inited_S = true;
        inited_mu = true;
    }
    void write_to_file(std::string filename, const size_t vec_number) const {
        utils::check(inited_F, "FockSigma: Fock matrix is not initialized");
        utils::check(inited_S, "FockSigma: Sigma is not initialized");
        h5::file file(filename, 'a');
        if(!h5::group(file).has_subgroup("vec" + std::to_string(vec_number))) {
            //app_log(2, "write_to_file: creating {} in file {}", "vec" + std::to_string(vec_number), filename);
            auto vec_grp = h5::group(file).create_group("vec" + std::to_string(vec_number));
            nda::h5_write(vec_grp, "Sigma_tskij", _Sigma, false);
            nda::h5_write(vec_grp, "F_skij", _Fock, false);
            h5::h5_write(vec_grp, "mu", _mu);
        } else {
            //app_log(2, "write_to_file: opening existing {} in file {}", "vec" + std::to_string(vec_number), filename);
            auto vec_grp = h5::group(file).open_group("vec" + std::to_string(vec_number));
            nda::h5_write(vec_grp, "Sigma_tskij", _Sigma, false);
            nda::h5_write(vec_grp, "F_skij", _Fock, false);
            h5::h5_write(vec_grp, "mu", _mu);
        }
    }            
    
};


/** 
 * Evaluation of the commutator in the tau space between G and G_0^{-1} - Sigma
 *
 * @param C_t     - [OUTPUT] Commutator in tau space
 * @param FT      - [INPUT] Imaginary frequency FT axes
 * @param G_t     - [INPUT] Green's function in tau space
 * @param FS_t    - [INPUT] Fock and Sigma in tau space
 * @param mu      - [INPUT] Chemical potential
 * @param S       - [INPUT] Overlap matrix
 * @param H0      - [INPUT] Non-interacting Hamiltonian
 **/
template<typename Array_G, typename Array_ov>
void commutator_t(Array_G& C_t, const imag_axes_ft::IAFT *FT,
                  const Array_G& G_t, const FockSigma& FS_t, double mu,
                  const Array_ov& S, const Array_ov& H0) {
    diis_timers::com_total.start();
    decltype(nda::range::all) all;
    size_t nt = G_t.shape()[0];
    size_t ns = G_t.shape()[1];
    size_t nk = G_t.shape()[2];
    size_t nao = G_t.shape()[3];
    size_t nw = FT->nw_f();
    diis_timers::com_alloc.start();
    nda::array<ComplexType, 5> G_w(nw,ns,nk,nao,nao);
    nda::array<ComplexType, 5> Sigma_w(nw,ns,nk,nao,nao);
    diis_timers::com_alloc.stop();
    // G_w is filled
    diis_timers::com_ftG.start();
    FT->tau_to_w(G_t, G_w, imag_axes_ft::fermion);
    diis_timers::com_ftG.stop();
    const auto& Sigma_t = FS_t.get_sigma();
    const auto& Fock = FS_t.get_fock();
    // HF: Sigma(tau) is exactly zero, so its FT is exactly zero. Detect this and
    // fill Sigma_w with exact zeros instead of transforming (bit-identical). For
    // scGW the scan hits a nonzero element and the transform is taken as usual.
    diis_timers::com_ftSigma.start();
    bool sigma_is_zero = std::none_of(Sigma_t.data(), Sigma_t.data() + Sigma_t.size(),
                                      [](ComplexType z) { return z != ComplexType{0}; });
    if (sigma_is_zero) Sigma_w() = 0;
    else FT->tau_to_w(Sigma_t, Sigma_w, imag_axes_ft::fermion);
    diis_timers::com_ftSigma.stop();

    nda::array<ComplexType, 4> Dm(ns,nk,nao,nao);
    FT->tau_to_beta(G_t, Dm);

    diis_timers::com_alloc.start();
    // C_w is fully overwritten in the (iw,s,k) loop below; C_t is fully
    // overwritten by w_to_tau (a single beta=0 gemm), so neither needs zeroing.
    nda::array<ComplexType, 5> C_w(nw, ns, nk, nao, nao);
    C_t = nda::array<ComplexType, 5>(nt,ns,nk,nao,nao); // appropriately sized output
    diis_timers::com_alloc.stop();

    nda::array<ComplexType, 2> I1(nao, nao);
    nda::array<ComplexType, 2> I2(nao, nao);

    diis_timers::com_gemm.start();
    for(size_t iw = 0; iw < nw; iw++)
    for(size_t s = 0; s < ns; s++)
    for(size_t k = 0; k < nk; k++) {
        long wn = FT->wn_mesh()(iw);
        ComplexType omega_mu = FT->omega(wn) + mu;
        auto S_sk = S(s,k,all,all);
        auto F_sk = Fock(s,k,all,all);
        auto H0_sk = H0(s,k,all,all);
        auto G_wsk = G_w(iw,s,k,all,all);
        auto Sigma_wsk = Sigma_w(iw,s,k,all,all);

        nda::array<ComplexType, 2> G0inv_Sigma_wsk = nda::make_regular(omega_mu * S_sk - H0_sk - F_sk - Sigma_wsk);
        nda::array_view<ComplexType, 2> C_wsk = C_w(iw,s,k,all,all);
        // gemm uses beta=0, fully overwriting I1/I2.
        nda::blas::gemm(G_wsk, G0inv_Sigma_wsk, I1);
        nda::blas::gemm(G0inv_Sigma_wsk, G_wsk, I2);
        C_wsk = nda::make_regular(I1 - I2);
    }
    diis_timers::com_gemm.stop();

    diis_timers::com_wtau.start();
    FT->w_to_tau(C_w, C_t, imag_axes_ft::fermion);
    diis_timers::com_wtau.stop();
    diis_timers::com_total.stop();
}


/**
 * A10: k-striped SPMD version of commutator_t.
 *
 * Every rank reads G, F, Sigma, S, H0 directly from the node-shared arrays
 * (byte-identical on all ranks) and processes only its own round-robin subset
 * of the (s,k) index end-to-end: per owned (s,k) it tau_to_w's the G (and, unless
 * Sigma is exactly zero, the Sigma) slice, runs the (iw) gemm loop, w_to_tau's the
 * resulting C slice, and writes it into the caller-provided output view C_t_out.
 *
 * C_t_out is a node-shared window (pre-zeroed by its shared_array ctor): the (s,k)
 * partition is disjoint, so every rank writes distinct blocks and no reduce over
 * the node is needed — the caller only fences the window and (for multi-node) sums
 * the per-node windows across the internode communicator. Each (t,s,k) block is
 * produced by exactly one rank, so this reproduces the serial commutator bit-for-bit
 * provided the per-slice FT gemm shapes match the serial full-array FT to the last
 * digit (validated by the energy gate).
 *
 * @param comm    - [INPUT] communicator the (s,k) work is striped over
 * @param C_t_out - [OUTPUT] commutator in tau space; this rank fills its owned (s,k)
 *                  blocks (a node-shared window; the fence/internode sum is the
 *                  caller's responsibility)
 * @param FT      - [INPUT] imaginary-axis FT axes
 * @param G_t     - [INPUT] Green's function in tau space (rank 5)
 * @param Fock    - [INPUT] Fock matrix (rank 4)
 * @param Sigma_t - [INPUT] self-energy in tau space (rank 5)
 * @param mu      - [INPUT] chemical potential
 * @param S       - [INPUT] overlap matrix (rank 4)
 * @param H0      - [INPUT] non-interacting Hamiltonian (rank 4)
 **/
template<typename comm_t, typename Arr_out, typename Arr_G, typename Arr_F,
         typename Arr_Sig, typename Arr_S, typename Arr_H0>
void commutator_t_distributed(comm_t& comm, Arr_out&& C_t_out,
                              const imag_axes_ft::IAFT *FT,
                              const Arr_G& G_t, const Arr_F& Fock,
                              const Arr_Sig& Sigma_t, double mu,
                              const Arr_S& S, const Arr_H0& H0) {
    diis_timers::com_dist_total.start();
    decltype(nda::range::all) all;
    long nk  = G_t.shape()[2];
    long nao = G_t.shape()[3];
    long nt  = G_t.shape()[0];
    long ns  = G_t.shape()[1];
    long nw  = FT->nw_f();
    long nsk = ns * nk;
    int rank    = comm.rank();
    int nranks  = comm.size();

    // Global Sigma==0 detection. Each rank scans only its owned slices; the
    // reduce reproduces the serial whole-array none_of decision (the FT of an
    // exactly-zero slice is exactly zero, matching the serial skip).
    diis_timers::com_dist_scan.start();
    int local_nonzero = 0;
    for (long sk = rank; sk < nsk; sk += nranks) {
        long s = sk / nk, k = sk % nk;
        auto Sig_sk = Sigma_t(all, s, k, all, all);
        if (std::any_of(Sig_sk.begin(), Sig_sk.end(),
                        [](ComplexType z) { return z != ComplexType{0}; })) {
            local_nonzero = 1;
            break;
        }
    }
    int global_nonzero = local_nonzero;
    comm.all_reduce_in_place_n(&global_nonzero, 1, std::plus<>{});
    bool sigma_is_zero = (global_nonzero == 0);
    diis_timers::com_dist_scan.stop();

    // Per-(s,k) contiguous scratch (the trailing (t/w, i, j) blocks feed the FT).
    nda::array<ComplexType, 3> G_t_sk(nt, nao, nao);
    nda::array<ComplexType, 3> G_w_sk(nw, nao, nao);
    nda::array<ComplexType, 3> Sigma_t_sk(nt, nao, nao);
    nda::array<ComplexType, 3> Sigma_w_sk(nw, nao, nao);
    nda::array<ComplexType, 3> C_w_sk(nw, nao, nao);
    nda::array<ComplexType, 3> C_t_sk(nt, nao, nao);
    nda::array<ComplexType, 2> I1(nao, nao);
    nda::array<ComplexType, 2> I2(nao, nao);

    for (long sk = rank; sk < nsk; sk += nranks) {
        long s = sk / nk, k = sk % nk;

        G_t_sk = G_t(all, s, k, all, all);
        diis_timers::com_dist_ftG.start();
        FT->tau_to_w(G_t_sk, G_w_sk, imag_axes_ft::fermion);
        diis_timers::com_dist_ftG.stop();

        diis_timers::com_dist_ftSigma.start();
        if (sigma_is_zero) {
            Sigma_w_sk() = 0;
        } else {
            Sigma_t_sk = Sigma_t(all, s, k, all, all);
            FT->tau_to_w(Sigma_t_sk, Sigma_w_sk, imag_axes_ft::fermion);
        }
        diis_timers::com_dist_ftSigma.stop();

        auto S_sk  = S(s, k, all, all);
        auto F_sk  = Fock(s, k, all, all);
        auto H0_sk = H0(s, k, all, all);

        diis_timers::com_dist_gemm.start();
        for (long iw = 0; iw < nw; iw++) {
            long wn = FT->wn_mesh()(iw);
            ComplexType omega_mu = FT->omega(wn) + mu;
            auto G_wsk = G_w_sk(iw, all, all);
            nda::array<ComplexType, 2> G0inv_Sigma_wsk =
                nda::make_regular(omega_mu * S_sk - H0_sk - F_sk - Sigma_w_sk(iw, all, all));
            // gemm uses beta=0, fully overwriting I1/I2.
            nda::blas::gemm(G_wsk, G0inv_Sigma_wsk, I1);
            nda::blas::gemm(G0inv_Sigma_wsk, G_wsk, I2);
            C_w_sk(iw, all, all) = nda::make_regular(I1 - I2);
        }
        diis_timers::com_dist_gemm.stop();

        diis_timers::com_dist_wtau.start();
        FT->w_to_tau(C_w_sk, C_t_sk, imag_axes_ft::fermion);
        diis_timers::com_dist_wtau.stop();

        C_t_out(all, s, k, all, all) = C_t_sk;
    }
    diis_timers::com_dist_total.stop();
}


}
#endif
