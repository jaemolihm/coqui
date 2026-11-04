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


#ifndef COQUI_DAMP_T_HPP
#define COQUI_DAMP_T_HPP

#include "configuration.hpp"
#include "utilities/check.hpp"

#include "h5/h5.hpp"
#include "nda/nda.hpp"
#include "nda/h5.hpp"

#include "numerics/iter_scf/iter_scf_type_e.hpp"
#include "numerics/iter_scf/diis/diis_timers.hpp"
#include "numerics/iter_scf/in_memory_mixing.hpp"

namespace iter_scf {
  /**
   * Abort when `dataset` of `iter_grp` was omitted by chkpt_slim (dump_scf writes
   * Sigma_not_stored = true then). Readers that take a missing dataset as an
   * exactly-zero quantity call this first.
   */
  inline void check_not_omitted_by_slim(h5::group &iter_grp, std::string const& dataset, long iter) {
    if (iter_grp.has_dataset(dataset) or not iter_grp.has_dataset("Sigma_not_stored")) return;
    bool not_stored = false;
    h5::h5_read(iter_grp, "Sigma_not_stored", not_stored);
    utils::check(not not_stored,
                 "iter_scf: {} of iteration {} was omitted from the checkpoint by chkpt_slim "
                 "and cannot be read as the previous state for mixing.", dataset, iter);
  }

  /**
   * Damping for an arbitrary target
   */
  struct damp_t {
  public:

    static constexpr iter_alg_e iter_alg = damping;
    static constexpr iter_alg_e get_iter_alg() { return iter_alg; }

    damp_t() = default;
    damp_t(double mixing_): mixing(mixing_) {}

    damp_t(const damp_t& other) = default;
    damp_t(damp_t&& other) = default;
    damp_t& operator=(const damp_t& other) = default;
    damp_t& operator=(damp_t&& other) = default;

    ~damp_t(){}

    template<nda::MemoryArray Array_H_t>
    double solve(Array_H_t &&H, std::string dataset, h5::group &grp, long iter) {
      utils::check(grp.has_subgroup("iter" + std::to_string(iter-1)), "damp: h5 group /scf/iter{} does not exist.", iter-1);
      auto iter_grp = grp.open_group("iter" + std::to_string(iter-1));

      // H_previous is fully overwritten by the h5_read below (or left zero when the
      // dataset is absent), so it is only shaped here, not copied from H (complex nda
      // arrays are calloc-backed). A missing dataset means the previous-iteration
      // quantity was exactly zero (e.g. Sigma_tskij on an HF checkpoint).
      std::decay_t<decltype(nda::make_regular(H))> H_previous(H.shape());
      check_not_omitted_by_slim(iter_grp, dataset, iter-1);
      diis_timers::damp_read.start();
      if (iter_grp.has_dataset(dataset))
        nda::h5_read(iter_grp, dataset, H_previous);
      else
        H_previous() = 0.0;
      diis_timers::damp_read.stop();

      diis_timers::damp_mix.start();
      H = mixing*H + (1.0-mixing)*H_previous;

      H_previous -= H;
      auto max_iter = max_element(H_previous.data(), H_previous.data()+H_previous.size(),
                              [](auto a, auto b) { return std::abs(a) < std::abs(b); });
      diis_timers::damp_mix.stop();
      return std::abs((*max_iter));
    }

    // This version is compatible with DIIS
    template<nda::MemoryArray Array_4D_t, nda::MemoryArray Array_5D_t>
    std::array<double, 2> solve(Array_4D_t &&F, std::string dataset_F, Array_5D_t &&Sigma, std::string dataset_Sigma,
                 h5::group &grp, long iter)  {
        return std::array<double, 2>{
            solve(F, dataset_F, grp, iter),
            solve(Sigma, dataset_Sigma, grp, iter)
        };
    }

    void metadata_log() const {
      app_log(2, "\nIterative algorithm for SCF");
      app_log(2, "-----------------------------");
      app_log(2, "  * algorithm: simple mixing");
      app_log(2, "  * mixing parameter = {}\n", mixing);
    }

    // Dummy function to comply with the interface
    template<class... Args>
    void initialize([[maybe_unused]] Args&&... args) {}

  public:
    double mixing = 0.7;
    bool initialized = true;
    // Element-sliced in-memory damping, driven directly by solve_iterative
    // (scf_common.cpp) for the Dyson SCF (the caller that supplies the in-memory
    // G/S/H0); solve() above keeps the checkpoint-reading path for the others.
    in_memory_mixing in_memory;

  };
} // iter_scf

#endif //COQUI_DAMP_T_HPP
