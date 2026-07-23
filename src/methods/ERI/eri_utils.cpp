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


#include "eri_utils.hpp"

namespace methods
{

auto make_thc(std::shared_ptr<mf::MF> mf, ptree const& pt) -> thc_reader_t {
  // generic optional
  auto storage = io::get_value_with_default<std::string>(pt,"storage","incore");
  io::tolower(storage);
  auto init = io::get_value_with_default<bool>(pt, "init", true);
  auto source = io::get_value_with_default<std::string>(pt,"source","auto");
  io::tolower(source);

  // check options
  std::string err = std::string("make_thc - Incorrect input - ");
  utils::check(storage == "incore" or storage == "outcore", err+"storage: {}", storage);
  utils::check(source == "auto" or source == "read" or source == "compute",
               err+"source: {} (expected 'auto', 'read', or 'compute')", source);

  auto save = io::get_value_with_default<std::string>(pt,"save",(storage == "incore"?"":"./thc.eri.h5"));
  bool file_exists = (save != "" and std::filesystem::exists(save));

  // decide whether to build (recompute) the ERIs or read them from file:
  //   "auto"    - read if the save file exists, otherwise build (default)
  //   "read"    - always read; error if the save file is missing
  //   "compute" - always build (ignore any pre-existing save file)
  bool build_eri;
  if (source == "read") {
    utils::check(save != "", "Error in make_thc: source='read' requires a 'save' file path.");
    utils::check(file_exists, "Error in make_thc: source='read' but save file ({}) does not exist.", save);
    build_eri = false;
  } else if (source == "compute") {
    build_eri = true;
  } else { // auto
    build_eri = not file_exists;
  }

  // nIpts/thresh are stopping criteria for the build; only required when building
  if (build_eri) {
    auto nIpts = io::get_value_with_default<int>(pt,"nIpts",0);
    auto thresh = io::get_value_with_default<double>(pt,"thresh",0.0);
    utils::check( nIpts>0 or thresh>0.0, "Error: Must set nIpts and/or thresh");
  }

  // only mf with orbitals can be built, otherwise must be read from file
  utils::check(mf->has_orbital_set() or not build_eri, "Error in make_thc: MF types that have no orbital sets (e.g type=model), can not be built, they must be read from file. save file ({}) must be provided or could not be opened.",save);

  thc_reader_t eri = (build_eri?
                      thc_reader_t(std::move(mf), pt, false, false, init) :
                      thc_reader_t(std::move(mf), storage, save, false, init));
  return eri;
};

void make_isdf(std::shared_ptr<mf::MF> mf, ptree const& pt) {
  std::string err = "make_isdf - missing required input: ";
  // required
  auto nIpts = io::get_value_with_default<int>(pt,"nIpts",0);
  auto thresh = io::get_value_with_default<double>(pt,"thresh",1e-10);
  utils::check( nIpts>0 or thresh>0.0, "{} Must set nIpts and/or thresh", err);

  bool isdf_only = true;
  thc_reader_t isdf(std::move(mf), pt, false, isdf_only);
};

void make_thc_pivots(std::shared_ptr<mf::MF> mf, ptree const& pt) {
  std::string err = "make_thc_pivots - missing required input: ";
  auto nIpts = io::get_value_with_default<int>(pt,"nIpts",0);
  auto thresh = io::get_value_with_default<double>(pt,"thresh",0.0);
  utils::check( nIpts>0 or thresh>0.0, "{} Must set nIpts and/or thresh", err);
  auto save = io::get_value_with_default<std::string>(pt,"save","thc_pivots.h5");
  utils::check(mf->has_orbital_set(),
               "Error in make_thc_pivots: MF types that have no orbital sets (e.g type=model) are not supported.");

  long nbnd = mf->nbnd();
  auto x_range = io::get_value_with_default<nda::range>(pt,"X_orbital_range",nda::range(nbnd));
  auto y_range = io::get_value_with_default<nda::range>(pt,"Y_orbital_range",x_range);
  utils::check(x_range.first() >= 0 and x_range.last() <= nbnd,
               "make_thc_pivots: X orbitals out of range: ({},{}), nbnd:{}",
               x_range.first(),x_range.last(),nbnd);
  utils::check(y_range.first() >= 0 and y_range.last() <= nbnd,
               "make_thc_pivots: Y orbitals out of range: ({},{}), nbnd:{}",
               y_range.first(),y_range.last(),nbnd);

  auto mpi = mf->mpi();
  thc builder(mf.get(), *mpi, pt);
  auto [ri, dXa, dXb] = builder.interpolating_points<HOST_MEMORY>(0, nIpts, x_range, y_range);

  app_log(1, "make_thc_pivots: found {} interpolating points, saving to: {}", ri.size(), save);
  if (mpi->comm.root()) {
    h5::file file(save, 'w');
    h5::group grp(file);
    h5::h5_write(grp, "Np", (int)ri.size());
    nda::h5_write(grp, "interpolating_points", ri, false);
    h5::h5_write(grp, "ecut", builder.get_ecut());
    nda::h5_write(grp, "fft_grid", builder.get_fft_mesh(), false);
  }
  mpi->comm.barrier();
  builder.print_timers();
};


auto make_cholesky(std::shared_ptr<mf::MF> mf, ptree const& pt) -> chol_reader_t
{
  // create and return cholesky reader
  auto storage = io::get_value_with_default<std::string>(pt,"storage","outcore");
  auto path = io::get_value_with_default<std::string>(pt,"path","./");
  auto output = io::get_value_with_default<std::string>(pt,"output","chol_info.h5");
  auto read_type = io::get_value_with_default<std::string>(pt,"read_type","all");
  auto write_type = io::get_value_with_default<std::string>(pt,"write_type","multi");
  auto redo = io::get_value_with_default<bool>(pt,"overwrite",false);
  io::tolower(storage);
  io::tolower(read_type);
  io::tolower(write_type);
  utils::check(storage=="outcore", "make_cholesky - the incore version is not implemented yet!");
  utils::check(read_type == "all" or read_type == "single", "make_cholesky: Invalid value read_type:{}",read_type);
  utils::check(write_type=="multi" or write_type=="single", "make_cholesky: Invalid value write_type:{}",write_type);
  auto rtype = (read_type=="all" ? each_q : single_kpair);
  auto wtype = (write_type=="multi" ? multi_file : single_file);

  auto nq = mf->nqpts_ibz();
  bool read_chol = (chol_reader_t::check_init(path,output,nq,wtype) and not redo);

  // only mf with orbitals can be built, otherwise must be read from file
  utils::check(mf->has_orbital_set() or read_chol, "Error in make_chol: MF types that have no orbital sets (e.g type=model), can not be built, they must be read from file. output file ({}) must be provided or could not be opened.",path+"/"+output);

  return ( read_chol ?
           chol_reader_t(std::move(mf), path, output, rtype, wtype) :
           chol_reader_t(std::move(mf), pt) );
};


}