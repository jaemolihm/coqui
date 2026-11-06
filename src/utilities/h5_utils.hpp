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


#ifndef UTILITIES_H5_UTILS_HPP
#define UTILITIES_H5_UTILS_HPP

#include <string>

#include <filesystem>
#include <hdf5.h>
#include "h5/h5.hpp"
#include "utilities/check.hpp"

// Helpers for operations the h5 library does not wrap: h5::group::has_subgroup accepts a path, but a missing
// intermediate group makes H5Lexists fail (has_key reads the error as true, and HDF5 prints its error stack);
// there is no object copy.
namespace utils {

/**
 * True if the file exists and holds a group at path (e.g. "Hamiltonian/ncpp/fullbz"), at any depth below the root;
 * false if the file does not exist. Throws if the file exists but is not HDF5.
 * Each component is checked in turn, since H5Lexists requires every intermediate link to exist.
 */
inline bool h5_group_exists(std::string const& filename, std::string const& path) {
  if (not std::filesystem::is_regular_file(filename)) return false;
  h5::file file(filename, 'r');
  h5::group grp(file);
  std::size_t start = 0;
  while (start < path.size()) {
    std::size_t end = path.find('/', start);
    if (end == std::string::npos) end = path.size();
    std::string name = path.substr(start, end - start);
    if (not name.empty()) {
      if (not grp.has_subgroup(name)) return false;
      grp = grp.open_group(name);
    }
    start = end + 1;
  }
  return true;
}

/**
 * Copy the object (dataset or group) name of src into dst under dst_name (default: the same name), with its
 * attributes. dst_name must not exist in dst.
 */
inline void h5_copy(h5::group const& src, std::string const& name, h5::group& dst, std::string dst_name = "") {
  if (dst_name.empty()) dst_name = name;
  herr_t status = H5Ocopy(src, name.c_str(), dst, dst_name.c_str(), H5P_DEFAULT, H5P_DEFAULT);
  utils::check(status >= 0, "h5_copy: failed to copy {} to {}.", name, dst_name);
}

} // utils

#endif // UTILITIES_H5_UTILS_HPP
