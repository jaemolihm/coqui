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

#ifndef COQUI_MODEL_TEST_UTILS_HPP
#define COQUI_MODEL_TEST_UTILS_HPP

#include <cmath>
#include <optional>
#include <string>
#include <vector>

#include "configuration.hpp"
#include "nda/nda.hpp"
#include "nda/h5.hpp"
#include <h5/h5.hpp>

namespace model_tests {

  /*
   * One-body Hamiltonian of an N x N x 1 square lattice, nb orbitals per cell:
   *   H_ab(k) = delta_ab (eps_k + 0.3 a) + (1-delta_ab) 0.1 exp(i kx (a-b)),
   *   eps_k = -2t (cos kx + cos ky) - 4t' cos kx cos ky,
   * on the i-major Gamma-first list k = (i/N, j/N, 0), index i*N + j.
   */
  inline auto square_lattice_H0(long N, double t, double tp, long nb = 1) {
    long nk = N*N;
    nda::array<double,2> kpts_crys(nk,3);
    nda::array<ComplexType,4> H0(1,nk,nb,nb);
    H0() = ComplexType(0.0);
    for(long i=0; i<N; ++i)
      for(long j=0; j<N; ++j) {
        long ik = i*N+j;
        kpts_crys(ik,0) = double(i)/double(N);
        kpts_crys(ik,1) = double(j)/double(N);
        kpts_crys(ik,2) = 0.0;
        double kx = 2.0*M_PI*kpts_crys(ik,0), ky = 2.0*M_PI*kpts_crys(ik,1);
        double eps = -2.0*t*(std::cos(kx)+std::cos(ky)) - 4.0*tp*std::cos(kx)*std::cos(ky);
        for(long a=0; a<nb; ++a)
          for(long b=0; b<nb; ++b)
            H0(0,ik,a,b) = (a==b) ? ComplexType(eps + 0.3*a)
                                  : 0.1*std::exp(ComplexType(0.0, kx*double(a-b)));
      }
    return std::make_tuple(kpts_crys, H0);
  }

  /*
   * Arbitrary mean-field spectrum [1, nk, nb] for read-back checks:
   *   eigval(0,k,a) = 0.01 k + a - 0.5,  occ(0,k,a) = ((k + a) % 3) / 2.
   */
  inline auto model_spectrum(long nk, long nb) {
    nda::array<double,3> eigval(1,nk,nb), occ(1,nk,nb);
    for(long ik=0; ik<nk; ++ik)
      for(long a=0; a<nb; ++a) {
        eigval(0,ik,a) = 0.01*double(ik) + double(a) - 0.5;
        occ(0,ik,a) = double((ik+a) % 3) / 2.0;
      }
    return std::make_tuple(eigval, occ);
  }

  /*
   * Writes a model mean-field file with only the required entries of the model format:
   * /System attrs (number_of_spins, number_of_polarizations, number_of_bands, number_of_elec,
   * madelung_constant, nuclear_energy), lattice_vectors (identity), kpoints_crystal, H0 and
   * fock_matrix (= H0). Optionally fermi_energy, and eigval/occ from model_spectrum
   * (with_spectrum). Call on one rank only.
   */
  inline void write_minimal_model_h5(std::string fname, long N, double t, double tp, double nelec,
                                     long nb = 1,
                                     std::optional<double> fermi_energy = std::nullopt,
                                     bool with_spectrum = false) {
    auto [kpts_crys, H0] = square_lattice_H0(N, t, tp, nb);
    h5::file file(fname, 'w');
    h5::group grp(file);
    h5::group sgrp = grp.create_group("System");
    h5::h5_write_attribute(sgrp, "number_of_spins", int(1));
    h5::h5_write_attribute(sgrp, "number_of_polarizations", int(1));
    h5::h5_write_attribute(sgrp, "number_of_bands", int(nb));
    h5::h5_write_attribute(sgrp, "number_of_elec", nelec);
    h5::h5_write_attribute(sgrp, "madelung_constant", 0.0);
    h5::h5_write_attribute(sgrp, "nuclear_energy", 0.0);
    if(fermi_energy.has_value())
      h5::h5_write_attribute(sgrp, "fermi_energy", fermi_energy.value());
    nda::array<double,2> latt = {{1.0,0.0,0.0},{0.0,1.0,0.0},{0.0,0.0,1.0}};
    nda::h5_write(sgrp, "lattice_vectors", latt, false);
    nda::h5_write(sgrp, "kpoints_crystal", kpts_crys, false);
    nda::h5_write(sgrp, "H0", H0, false);
    nda::h5_write(sgrp, "fock_matrix", H0, false);
    if(with_spectrum) {
      auto [eigval, occ] = model_spectrum(N*N, nb);
      nda::h5_write(sgrp, "eigval", eigval, false);
      nda::h5_write(sgrp, "occ", occ, false);
    }
  }

} // model_tests

#endif
