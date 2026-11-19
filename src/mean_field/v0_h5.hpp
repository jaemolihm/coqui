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

#ifndef MEANFIELD_V0_H5_HPP
#define MEANFIELD_V0_H5_HPP

#include <string>
#include <tuple>

#include <nda/nda.hpp>
#include <nda/h5.hpp>
#include <h5/h5.hpp>

#include "configuration.hpp"
#include "utilities/check.hpp"

namespace mf
{

/**
 * Read the Cartesian velocity matrix Orbitals/v0_skxij, v0[s,k,x,m,n] = <m|-i[r_x,H0]|n> in
 * Hartree a.u. over the IBZ, from an h5 file into `v0` (nspin, nkpts_ibz, 3, nb, nb).
 * Called by qe_readonly::read_v0 and bdft_readonly::read_v0, which share this schema.
 * Only the leading nb x nb bands are fetched, as an h5 hyperslab, so a basis with fewer bands
 * never materializes the full stored matrix. Returns nbnd_stored, so the caller can report
 * a truncating read.
 */
inline long read_v0_dataset(std::string const& filename, nda::array_view<ComplexType,5> v0)
{
  long nb = v0.extent(3);
  utils::check(v0.extent(4) == nb and v0.extent(2) == 3,
               "read_v0: v0 has shape ({},{},{},{},{}); expected (ns,nk,3,nb,nb).",
               v0.extent(0), v0.extent(1), v0.extent(2), v0.extent(3), v0.extent(4));
  h5::file file(filename, 'r');
  h5::group grp(file);
  auto ogrp = grp.open_group("Orbitals");
  utils::check(ogrp.has_dataset("v0_skxij"), "read_v0: {} has no Orbitals/v0_skxij.", filename);
  // complex datasets carry a trailing length-2 axis, hence rank 6 on disk
  auto info = h5::array_interface::get_dataset_info(ogrp, "v0_skxij");
  utils::check(info.rank() == 6,
               "read_v0: v0_skxij in {} has rank {}; expected a complex rank-5 dataset.",
               filename, info.rank());
  long nbnd_stored = long(info.lengths[3]);
  utils::check(long(info.lengths[0]) == v0.extent(0) and long(info.lengths[1]) == v0.extent(1) and
               long(info.lengths[2]) == 3 and nbnd_stored == long(info.lengths[4]) and nbnd_stored >= nb,
               "read_v0: v0_skxij in {} has shape ({},{},{},{},{}), incompatible with the requested "
               "({},{},3,{},{}).", filename, long(info.lengths[0]), long(info.lengths[1]),
               long(info.lengths[2]), long(info.lengths[3]), long(info.lengths[4]),
               v0.extent(0), v0.extent(1), nb, nb);
  nda::h5_read(ogrp, "v0_skxij", v0,
               std::tuple{nda::range::all, nda::range::all, nda::range::all, nda::range(nb), nda::range(nb)});
  return nbnd_stored;
}

} // namespace mf

#endif
