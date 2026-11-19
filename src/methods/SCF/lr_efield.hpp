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

#ifndef COQUI_LR_EFIELD_HPP
#define COQUI_LR_EFIELD_HPP

#include <array>

#include "nda/nda.hpp"
#include "configuration.hpp"
#include "utilities/mpi_context.h"
#include "numerics/imag_axes_ft/IAFT.hpp"
#include "numerics/shared_array/nda.hpp"
#include "methods/SCF/scf_common.hpp"

/**
 * Band-basis building blocks of run_lr_efield_calc (MBPT_drivers.cpp), the electronic dielectric tensor from Q = 0
 * LR solves (note efield_lr.tex, eq:chi2work):
 *
 *   chi2_ab = <G^E_a, V^E_b> + 1/4 Tr[v_a G' v_b G'] + 1/4 Tr[Sigma'' G v_a G v_b G],
 *   V^E_a   = -(i/2) (v_a G B - B G v_a),   B = S - Sigma',   G' = -G B G,
 *
 * with every operator a (nb, nb) matrix per (iω, s, k) of a node-shared (nw_f, ns, nk, nb, nb) array.
 *
 * Why lr_hessian_t's pairing machinery is not reused here: terms 2 and 3 are products of four or five matrices at one
 * (iω, k), not pairings of two stored operands, and the K^W pairing of the driver would need `trace_matsubara` and
 * `pack_to_omega`, which are private stripe helpers of the hessian stores. The pairing of two node-shared ω arrays
 * is therefore written out once here (matsubara_pairing).
 */
namespace methods {

using Array_view_6D_t = nda::array_view<ComplexType, 6>;

/**
 * X(iω) = ΔH0 + ΔF + τ->ω ΔΣ(τ): the total one-body term the Dyson pass of an LR solve sees, on the iω grid.
 *
 * Called by run_lr_efield_calc after each solve whose ΔF and ΔΣ(τ) are those of the final Dyson pass: B of phase A1
 * (ΔH0 = S), v_α of phase X (ΔH0 = v0_α) and, without ΔH0 and without the source, Σ'' of phase X. The two
 * node-shared ΔF, ΔΣ(τ) are what lr_solve_one leaves in its output arrays; nothing else turns them into iω arrays.
 *
 * @param sDeltaH0_skij    - [INPUT] ΔH0 (ns, nk, nb, nb); nullptr: zero
 * @param sDeltaF_skij     - [INPUT] ΔF (ns, nk, nb, nb)
 * @param sDeltaSigma_tskij- [INPUT] ΔΣ(τ) (nt_f, ns, nk, nb, nb); nullptr for a Σ-free kernel
 * @param sX_wskij         - [OUTPUT] X(iω) (nw_f, ns, nk, nb, nb), overwritten
 *
 * Collective on the node communicator; every node fills its own copy.
 */
void lr_total_one_body_w(utils::mpi_context_t<mpi3::communicator>& mpi, imag_axes_ft::IAFT const& FT,
                         sArray_t<Array_view_4D_t> const* sDeltaH0_skij,
                         sArray_t<Array_view_4D_t> const& sDeltaF_skij,
                         sArray_t<Array_view_5D_t> const* sDeltaSigma_tskij,
                         sArray_t<Array_view_5D_t>& sX_wskij);

/**
 * The effective field sources V^E_α(iω), α = x, y, z:  V^E_α = -(i/2)(v_α G B - B G v_α).
 *
 * Called by run_lr_efield_calc once, after phases A1 and X, as the dynamic source ΔH(iω) of the three G^E_α
 * solves of phase A2. Built from G, B and v of the same grid, so V^E(-iω) = V^E(iω)† holds by construction, which
 * a dynamic source of run_lr needs.
 *
 * @param sG_wskij     - [INPUT] unperturbed G(iω)
 * @param sB_wskij     - [INPUT] B(iω) = S - Σ'
 * @param sv_wskij     - [INPUT] v_α(iω), α = 0..2 (the total one-body term of the v solves)
 * @param sVE_xwskij   - [OUTPUT] V^E (3, nw_f, ns, nk, nb, nb), overwritten
 *
 * Collective on the node communicator; every node fills its own copy.
 */
void build_field_sources(utils::mpi_context_t<mpi3::communicator>& mpi,
                         sArray_t<Array_view_5D_t> const& sG_wskij, sArray_t<Array_view_5D_t> const& sB_wskij,
                         std::array<sArray_t<Array_view_5D_t> const*, 3> sv_wskij,
                         sArray_t<Array_view_6D_t>& sVE_xwskij);

/**
 * The source of the Σ'' solve, 2 B G B (iω).
 *
 * Called by run_lr_efield_calc in phase X. Σ'' = ∂²_μ Σ is the kernel response K[G''] to G'' = G (2 B G B + Σ'') G,
 * so the dynamic source of that solve is 2 B G B, while Σ'' itself is the ΔF + ΔΣ the kernel returns.
 *
 * @param sDeltaH_wskij - [OUTPUT] 2 B G B (nw_f, ns, nk, nb, nb), overwritten
 *
 * Collective on the node communicator.
 */
void build_mu2_source(utils::mpi_context_t<mpi3::communicator>& mpi,
                      sArray_t<Array_view_5D_t> const& sG_wskij, sArray_t<Array_view_5D_t> const& sB_wskij,
                      sArray_t<Array_view_5D_t>& sDeltaH_wskij);

/// Terms 2 and 3 of chi2, (3, 3) complex, unsymmetrized, with the 1/4 and 1/Ω applied.
struct efield_trace_terms {
  nda::array<ComplexType, 2> vGpvGp;       ///< 1/4 Tr[v_a G' v_b G'] / Ω
  nda::array<ComplexType, 2> Sigma2_GvGvG; ///< 1/4 Tr[Σ'' G v_a G v_b G] / Ω
};

/**
 * Terms 2 and 3 of chi2: the two Matsubara traces that are products of four or five band matrices at one (iω, k).
 *
 *   term 2_ab = 1/(4Ω) (1/β) Σ_n  spin Σ_k w_k tr[v_a G' v_b G'](iω_n, k),   G' = -G B G
 *   term 3_ab = 1/(4Ω) (1/β) Σ_n  spin Σ_k w_k tr[Σ'' G v_a G v_b G](iω_n, k)
 *
 * Called by run_lr_efield_calc after phase X, while B, v and Σ'' are alive. The flattened (n, s, k) is striped over
 * the global communicator, the (nw_f, 9) traces are reduced and each column goes through
 * imag_axes_ft::fermion_matsubara_sum (the same sum the hessian's Tr_ω closes with).
 *
 * @param sv_wskij    - [INPUT] v_α(iω), α = 0..2
 * @param sSigma2_wskij - [INPUT] Σ''(iω)
 * @param k_weight    - [INPUT] w_k over the IBZ (= the full BZ here), summing to 1
 * @param spin_factor - [INPUT] 2 for ns = 1, npol = 1, else 1
 * @param volume      - [INPUT] cell volume Ω
 *
 * Collective on the global communicator; every rank returns the same terms.
 */
efield_trace_terms contract_efield_terms(utils::mpi_context_t<mpi3::communicator>& mpi,
                                         imag_axes_ft::IAFT const& FT,
                                         sArray_t<Array_view_5D_t> const& sG_wskij,
                                         sArray_t<Array_view_5D_t> const& sB_wskij,
                                         std::array<sArray_t<Array_view_5D_t> const*, 3> sv_wskij,
                                         sArray_t<Array_view_5D_t> const& sSigma2_wskij,
                                         nda::array<double, 1> const& k_weight, double spin_factor,
                                         double volume);

/**
 * (1/Ω) (1/β) Σ_n spin Σ_k w_k tr[A(iω_n, k) B(iω_n, k)]: the Matsubara pairing of two node-shared iω arrays.
 *
 * Called by run_lr_efield_calc for the K^W diagnostic, <G^E_a, K^W[G^E_b]>, on the iω forms of the stored G^E(τ) and
 * of K^W[G^E](τ). tr[A B] = Σ_ij conj(A_ij(-iω)) B_ij for A(-iω) = A(iω)†, i.e. the pairing the hessian uses.
 *
 * Collective on the global communicator; every rank returns the same value.
 */
ComplexType matsubara_pairing(utils::mpi_context_t<mpi3::communicator>& mpi, imag_axes_ft::IAFT const& FT,
                              sArray_t<Array_view_5D_t> const& sA_wskij, sArray_t<Array_view_5D_t> const& sB_wskij,
                              nda::array<double, 1> const& k_weight, double spin_factor, double volume);

/**
 * max_{s,k} |tr G(τ*)|, the thermal occupation of the ground state: tr G(τ) at the node nearest the middle of the
 * interval, e^{-βΔ/2} for a gapped system at midgap μ.
 *
 * Called by run_lr_efield_calc on the unperturbed G(τ) for the g_thermal diagnostic. The IR grid cannot interpolate
 * to β/2, so the node of smallest |x| in the reduced coordinate x = 2τ/β - 1 is used; its τ is not exactly β/2.
 */
double thermal_occupation(imag_axes_ft::IAFT const& FT, sArray_t<Array_view_5D_t> const& sG_tskij);

} // namespace methods

#endif // COQUI_LR_EFIELD_HPP
