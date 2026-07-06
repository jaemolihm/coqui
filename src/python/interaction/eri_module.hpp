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


#ifndef ERI_MODULE_HPP
#define ERI_MODULE_HPP

#include "python/utils/mpi_handler.hpp"
#include "python/utils/mpi_handler.wrap.hxx"
#include "python/mean_field/mf_module.hpp"
#include "python/mean_field/mf_module.wrap.hxx"

#include "IO/ptree/InputParser.hpp"
#include "methods/ERI/eri_utils.hpp"
#include "methods/ERI/lr_thc_interp.hpp"

namespace coqui_py {

  void run_isdf(const Mf &mf, const std::string &thc_params) {
    auto parser = InputParser(thc_params);
    methods::make_isdf(mf.get_mf(), parser.get_root());
  }

  C2PY_IGNORE
  inline decltype(auto) make_thc(const Mf &mf, const std::string &thc_params) {
    auto parser = InputParser(thc_params);
    return methods::make_thc(mf.get_mf(), parser.get_root());
  }

  class ThcCoulomb {
  public:
    ThcCoulomb(const Mf &mf, const std::string &thc_params):
    _thc(make_thc(mf, thc_params)) {}

    ~ThcCoulomb() = default;
    ThcCoulomb(ThcCoulomb const&) = default;
    ThcCoulomb(ThcCoulomb &&) = default;
    ThcCoulomb& operator=(ThcCoulomb const&) = default;
    ThcCoulomb& operator=(ThcCoulomb &&) = default;

    void init() { _thc.init(!_thc.thc_builder_is_null()); }
    auto initialized() const { return _thc.initialized(); }

    auto Np() const { return _thc.Np(); }
    auto nkpts() const { return _thc.nkpts(); }
    auto nkpts_ibz() const { return _thc.nkpts_ibz(); }
    auto nqpts() const { return _thc.nqpts(); }
    auto nqpts_ibz() const { return _thc.nqpts_ibz(); }
    auto nspin() const { return _thc.ns(); }
    auto nspin_in_basis() const { return _thc.ns_in_basis(); }
    auto nbnd() const { return _thc.nbnd(); }

    // create a new MpiHandler from _thc's mpi
    auto mpi() const { return MpiHandler(_thc.mpi()); }
    // create a new Mf from _thc's MF
    auto mf() const { return Mf(_thc.MF()); }

    C2PY_IGNORE
    auto& get_eri() { return _thc; }
    C2PY_IGNORE
    auto get_mpi() const { return _thc.mpi(); }
    C2PY_IGNORE
    auto get_mf() const { return _thc.MF(); }

  private:
    methods::thc_reader_t _thc;

  }; // ThcCoulomb

  C2PY_IGNORE
  inline decltype(auto) make_cholesky(const Mf &mf, const std::string &chol_params) {
    auto parser = InputParser(chol_params);
    return methods::make_cholesky(mf.get_mf(), parser.get_root());
  }

  class CholCoulomb {
  public:
    CholCoulomb(const Mf &mf, const std::string &chol_params):
    _cholesky(make_cholesky(mf, chol_params)) {}

    ~CholCoulomb() = default;
    CholCoulomb(CholCoulomb const&) = default;
    CholCoulomb(CholCoulomb &&) = default;
    CholCoulomb& operator=(CholCoulomb const&) = default;
    CholCoulomb& operator=(CholCoulomb &&) = default;

    //void init() { _thc.init(!_thc.thc_builder_is_null()); }
    //auto initialized() { return _thc.initialized(); }

    // create a new MpiHandler from _thc's mpi
    auto mpi() const { return MpiHandler(_cholesky.mpi()); }
    // create a new Mf from _thc's MF
    auto mf() const { return Mf(_cholesky.MF()); }

    C2PY_IGNORE
    auto& get_eri() { return _cholesky; }
    C2PY_IGNORE
    auto get_mpi() const { return _cholesky.mpi(); }
    C2PY_IGNORE
    auto get_mf() const { return _cholesky.MF(); }

  private:
    methods::chol_reader_t _cholesky;

  }; // CholCoulomb

  // Linear response of the THC collocation matrix X(k, P, m) = ψ_{mk}(r_P).
  // See methods::compute_delta_X in src/methods/ERI/lr_thc_interp.hpp for the
  // full mathematical setup (term 1 = δψ(r_P), term 2 = ∇ψ(r_P)·Δr_P).
  nda::array<ComplexType, 4> compute_delta_X(
      const Mf &mf,
      const std::string &Deltapsi_prefix,
      nda::array<long, 1> const& r_P,
      nda::array<double, 2> const& delta_r_P,
      nda::array<double, 1> const& q_vec_cryst,
      nda::array<long, 1> const& fft_grid)
  {
    return methods::compute_delta_X(
        *mf.get_mf(), Deltapsi_prefix, r_P, delta_r_P, q_vec_cryst, fft_grid);
  }

  // Adjoint (−q) companion of compute_delta_X. At index ik the returned array
  // stores δ^{-q} X evaluated at k_{ik}+q — the consumer convention of
  // DeltaX_right in lr_thc_comm. Term 1 reads [δ^q]†ψ_{k+q} from
  // {Deltapsi_adj_prefix}_ik{ik+1}.hdf5 (adjoint Sternheimer output).
  nda::array<ComplexType, 4> compute_delta_X_adj(
      const Mf &mf,
      const std::string &Deltapsi_adj_prefix,
      nda::array<long, 1> const& r_P,
      nda::array<double, 2> const& delta_r_P,
      nda::array<double, 1> const& q_vec_cryst,
      nda::array<long, 1> const& fft_grid)
  {
    return methods::compute_delta_X_adj(
        *mf.get_mf(), Deltapsi_adj_prefix, r_P, delta_r_P, q_vec_cryst, fft_grid);
  }


} // coqui_py

#endif
