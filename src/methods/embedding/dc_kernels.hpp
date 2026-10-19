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


#ifndef COQUI_DC_KERNELS_HPP
#define COQUI_DC_KERNELS_HPP

#include "nda/nda.hpp"

namespace methods {

// Bilinear kernels of the GW double counting on the half tau grid (mt = nts-1-t). They accumulate into the
// output, so a linear response S[dG,W] + S[G,dW] is two calls on one output. G arrays are (t, s, a, b) views.
// Python twins: weiss.eval_gw_dc_t, weiss.eval_pi_rpa_pair (src/python/dmft/weiss.py).

// Sigma_ab(t) -= sum_cd G_cd(t) W(t,c,a,d,b) for t on the half grid of W, mirrored to mt.
// Twin of weiss.eval_gw_dc_t.
template<typename G_t, typename W_t, typename Sigma_t>
void gw_sigma_dc_kernel(const G_t &G_tsab, const W_t &W_tabcd, Sigma_t &&Sigma_tsab) {
  long nts = G_tsab.extent(0);
  long ns = G_tsab.extent(1);
  long n = G_tsab.extent(2);
  long nt_half = W_tabcd.extent(0);
  for (long it=0; it<nt_half; ++it) {
    long imt = nts-it-1;
    for (long is=0; is<ns; ++is) {
      for (long a=0; a<n; ++a) {
        for (long b=0; b<n; ++b) {
          for (long c=0; c<n; ++c) {
            for (long d=0; d<n; ++d) {
              Sigma_tsab(it,is,a,b) -= G_tsab(it,is,c,d) * W_tabcd(it,c,a,d,b);
              if (it != imt)
                Sigma_tsab(imt,is,a,b) -= G_tsab(imt,is,c,d) * W_tabcd(it,c,a,d,b);
            }
          }
        }
      }
    }
  }
}

// Pi_abcd(t) += sf * sum_s G1_bd(t) G2_ca(mt), sf = -2 (ns = 1) or -1. Twin of weiss.eval_pi_rpa_pair.
template<typename G1_t, typename G2_t, typename Pi_t>
void pi_pair_kernel(const G1_t &G1_tsab, const G2_t &G2_tsab, Pi_t &&Pi_tabcd) {
  long nts = G1_tsab.extent(0);
  long ns = G1_tsab.extent(1);
  long n = G1_tsab.extent(2);
  long nt_half = Pi_tabcd.extent(0);
  double spin_factor = (ns==1)? -2.0 : -1.0;
  for (long it=0; it<nt_half; ++it) {
    long itt = nts-it-1;
    for (long a=0; a<n; ++a) {
      for (long b=0; b<n; ++b) {
        for (long c=0; c<n; ++c) {
          for (long d=0; d<n; ++d) {
            for (long s=0; s<ns; ++s) {
              Pi_tabcd(it,a,b,c,d) += spin_factor * G1_tsab(it,s,b,d) * G2_tsab(itt,s,c,a);
            }
          }
        }
      }
    }
  }
}

// Density-only variant: Pi_ab(t) += sf * sum_s G1_ab(t) G2_ba(mt), the element Pi_{aabb} of the density-density
// polarizability. Twin of weiss.eval_pi_rpa_pair(density_only=True).
template<typename G1_t, typename G2_t, typename Pi_t>
void pi_density_pair_kernel(const G1_t &G1_tsab, const G2_t &G2_tsab, Pi_t &&Pi_tab) {
  long nts = G1_tsab.extent(0);
  long ns = G1_tsab.extent(1);
  long n = G1_tsab.extent(2);
  long nt_half = Pi_tab.extent(0);
  double spin_factor = (ns==1)? -2.0 : -1.0;
  for (long it=0; it<nt_half; ++it) {
    long itt = nts-it-1;
    for (long a=0; a<n; ++a) {
      for (long b=0; b<n; ++b) {
        for (long s=0; s<ns; ++s) {
          Pi_tab(it,a,b) += spin_factor * G1_tsab(it,s,a,b) * G2_tsab(itt,s,b,a);
        }
      }
    }
  }
}

} // namespace methods

#endif // COQUI_DC_KERNELS_HPP
