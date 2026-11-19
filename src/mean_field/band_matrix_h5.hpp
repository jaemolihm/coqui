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

#ifndef MEANFIELD_BAND_MATRIX_H5_HPP
#define MEANFIELD_BAND_MATRIX_H5_HPP

#include <string>
#include <tuple>
#include <utility>

#include <nda/nda.hpp>
#include <nda/h5.hpp>
#include <h5/h5.hpp>

#include "configuration.hpp"
#include "utilities/check.hpp"

namespace mf
{

/**
 * Read the leading nb x nb band block of the complex dataset Orbitals/<name> of an h5 mean-field file into `A`,
 * whose last two axes are the band axes (nb = A.extent(R-2) = A.extent(R-1)) and whose leading R-2 axes
 * (spin, k, [Cartesian direction]) must match the stored extents. Complex datasets carry a trailing length-2
 * axis on disk, so the dataset has rank R+1.
 * Called by bdft_readonly::read_hks_matrix (name = "H_KS_skij", R = 4) and by qe_readonly::read_v0 and
 * bdft_readonly::read_v0 (name = "v0_skxij", R = 5), which share this schema. Only the leading nb x nb bands
 * are fetched, as an h5 hyperslab, so a basis with fewer bands never materializes the full stored matrix.
 * Returns nbnd_stored, so the caller can report a truncating read.
 */
template <int R>
long read_band_matrix_dataset(std::string const& filename, std::string const& name,
                              nda::array_view<ComplexType,R> A)
{
  static_assert(R >= 3, "read_band_matrix_dataset: A needs at least one leading axis and two band axes.");
  long nb = A.extent(R-2);
  utils::check(A.extent(R-1) == nb,
               "read_band_matrix_dataset: {} target is not square in the band axes: ({}, {}).",
               name, nb, A.extent(R-1));
  h5::file file(filename, 'r');
  h5::group grp(file);
  auto ogrp = grp.open_group("Orbitals");
  utils::check(ogrp.has_dataset(name), "read_band_matrix_dataset: {} has no Orbitals/{}.", filename, name);
  // complex datasets carry a trailing length-2 axis, hence rank R+1 on disk
  auto info = h5::array_interface::get_dataset_info(ogrp, name);
  utils::check(info.rank() == R+1,
               "read_band_matrix_dataset: {} in {} has rank {}; expected a complex rank-{} dataset.",
               name, filename, info.rank(), R);
  long nbnd_stored = long(info.lengths[R-2]);
  bool lead_ok = true;
  for (int i = 0; i < R-2; ++i) lead_ok = lead_ok and (long(info.lengths[i]) == A.extent(i));
  utils::check(lead_ok and nbnd_stored == long(info.lengths[R-1]) and nbnd_stored >= nb,
               "read_band_matrix_dataset: {} in {} has a stored band block of {}x{} and leading extents "
               "incompatible with the requested array ({} bands).",
               name, filename, nbnd_stored, long(info.lengths[R-1]), nb);
  // leading axes whole, band axes cut to the leading nb x nb
  auto slice = [&]<std::size_t... I>(std::index_sequence<I...>) {
    return std::tuple{(void(I), nda::range::all)..., nda::range(nb), nda::range(nb)};
  }(std::make_index_sequence<R-2>{});
  nda::h5_read(ogrp, name, A, slice);
  return nbnd_stored;
}

} // namespace mf

#endif
