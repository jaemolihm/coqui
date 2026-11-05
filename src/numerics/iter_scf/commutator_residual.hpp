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



#ifndef COQUI_COMMUTATOR_RESIDUAL_HPP
#define COQUI_COMMUTATOR_RESIDUAL_HPP

#include <algorithm>

#include "configuration.hpp"
#include "nda/nda.hpp"
#include "nda/blas.hpp"
#include "numerics/imag_axes_ft/IAFT.hpp"
#include "numerics/nda_functions.hpp"  // nda::blas 3-argument gemm
#include "numerics/iter_scf/diis/diis_timers.hpp"

// Commutator residuals of the SCF mixing (DIIS), computed on node-shared arrays.
//
// dyson_commutator is the (s,k)-striped form of commutator_t (diis/vspace_fock_sigma.hpp) for
// the frequency-dependent commutator DIIS of the Dyson SCF: P. Pokhilko, C.-N. Yeh, D. Zgid,
// J. Chem. Phys. 156, 094101 (2022), https://doi.org/10.1063/5.0082586. heff_commutator is
// the (s,k)-striped H D S - S D H of qp_com_diis_residual::get_diis_residual (qp SCF).
namespace iter_scf {

/**
 * Commutator in tau space between G and G_0^{-1} - Sigma, C = G (G_0^{-1} - Sigma) - (G_0^{-1} - Sigma) G
 * with G_0^{-1} = (iw + mu) S - H0 - F, striped over (s,k).
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
 * produced by exactly one rank.
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
void dyson_commutator(comm_t& comm, Arr_out&& C_t_out,
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
    // reduce gives the whole-array none_of decision (the FT of an exactly-zero
    // slice is exactly zero).
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



/**
 * Commutator of the effective Hamiltonian with the density matrix, C = H D S - S D H,
 * for every (s,k), striped over the ranks of `comm`.
 *
 * C_out is a window shared by the ranks of `comm`; each rank writes distinct (s,k) blocks, so
 * the caller only fences the window.
 *
 * @param comm  - [INPUT] communicator the (s,k) work is striped over (the ranks sharing C_out)
 * @param C_out - [OUTPUT] commutator, shape (ns,nk,nao,nao)
 * @param H     - [INPUT] effective Hamiltonian (rank 4)
 * @param Dm    - [INPUT] density matrix (rank 4)
 * @param S     - [INPUT] overlap matrix (rank 4)
 **/
template<typename comm_t, typename Arr_out, typename Arr_H, typename Arr_D, typename Arr_S>
void heff_commutator(comm_t& comm, Arr_out&& C_out, const Arr_H& H, const Arr_D& Dm, const Arr_S& S) {
    decltype(nda::range::all) all;
    long ns  = H.shape()[0];
    long nk  = H.shape()[1];
    long nao = H.shape()[2];
    nda::array<ComplexType, 2> I1(nao, nao), I2(nao, nao), I3(nao, nao), I4(nao, nao);
    for (long sk = comm.rank(); sk < ns * nk; sk += comm.size()) {
        long s = sk / nk, k = sk % nk;
        auto H_sk = H(s, k, all, all);
        auto D_sk = Dm(s, k, all, all);
        auto S_sk = S(s, k, all, all);
        // gemm uses beta=0, fully overwriting I1..I4.
        nda::blas::gemm(H_sk, D_sk, I1);
        nda::blas::gemm(I1, S_sk, I2);
        nda::blas::gemm(S_sk, D_sk, I3);
        nda::blas::gemm(I3, H_sk, I4);
        C_out(s, k, all, all) = nda::make_regular(I2 - I4);
    }
}

} // iter_scf

#endif // COQUI_COMMUTATOR_RESIDUAL_HPP
