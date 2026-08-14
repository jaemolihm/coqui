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


#ifndef COQUI_SCF_MEM_REPORT_HPP
#define COQUI_SCF_MEM_REPORT_HPP

#include <algorithm>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#include "utilities/mpi_context.h"
#include "numerics/imag_axes_ft/IAFT.hpp"
#include "numerics/iter_scf/iter_scf_t.hpp"
#include "methods/ERI/detail/concepts.hpp"
#include "methods/SCF/mb_solver_t.h"

namespace methods {

/**
 * Scalar description of one scf_loop invocation: everything the up-front memory
 * and distribution reports need, with no MPI handles and no ERI/solver types, so
 * the printers themselves are plain non-template functions.
 */
struct scf_mem_params {
  // ---- dimensions ----
  long nt_f = 0;         // fermionic imaginary-time points
  long nw_f = 0;         // fermionic Matsubara frequencies
  long nw_b = 0;         // bosonic Matsubara frequencies
  long nth  = 0;         // half tau-grid,  ceil(nt_f/2)
  long nwh  = 0;         // half bosonic omega-grid, ceil(nw_b/2)
  long ns   = 0;         // nspin
  long nsp_basis = 0;    // nspin_in_basis * npol_in_basis (THC collocation axis 0)
  long nk   = 0;         // full-BZ k-points
  long nki  = 0;         // IBZ k-points
  long nq   = 0;         // full-BZ q-points
  long nqi  = 0;         // IBZ q-points
  long nk_trev = 0;      // time-reversal k pairs
  long nb   = 0;         // nbnd
  long NP   = 0;         // THC auxiliary-basis rank

  // ---- active code paths ----
  bool have_hf   = false;   // static Fock solver present
  bool have_corr = false;   // dynamic self-energy solver present (gw/gf2)
  bool have_scr  = false;   // screened-interaction solver present (Pi + W)
  bool is_gw     = false;
  bool is_gf2    = false;
  bool thc_eri   = false;   // correlated-solver ERI is THC (else Cholesky)
  bool eri_incore = false;  // THC Coulomb kernel resident (dZ_incore)
  bool separate_Y = false;  // X_orbital_range != Y_orbital_range -> a second collocation array
  bool keep_w    = false;
  bool dump_exchange = false;
  bool eval_thermodynamics = false;
  int  n_thc_readers = 0;   // distinct THC reader objects among the mb_eri slots

  // ---- options ----
  std::string screen_type;
  std::string sigma_alg;       // "R" or "k"
  std::string div_treatment;
  std::string iter_alg;        // "damping" / "DIIS" ("" when no iterative solver)
  std::string diis_storage;    // "disk" / "memory" ("" unless DIIS)
  long max_subsp = 0;          // DIIS max subspace size
};

/**
 * Fill scf_mem_params at the scf_loop call site. The only template in the report,
 * so no instantiation block is needed for scf_loop's 32 permutations.
 *
 * mb_eri is taken whole (not just the corr slot) so the distinct-THC-reader count
 * can be established: each reader owns its own multi-GB X/Z arrays.
 */
template<typename mf_ptr_t, typename eri_t, typename corr_solver_t>
scf_mem_params make_scf_mem_params(mf_ptr_t mf, const imag_axes_ft::IAFT& FT,
                                  eri_t& mb_eri,
                                  const solvers::mb_solver_t<corr_solver_t>& mb_solver,
                                  iter_scf::iter_scf_t* iter_solver,
                                  bool keep_w, bool dump_exchange,
                                  bool eval_thermodynamics) {
  scf_mem_params p;

  p.nt_f = FT.nt_f();
  p.nw_f = FT.nw_f();
  p.nw_b = FT.nw_b();
  // Half-grid extents follow the allocators: eval_Pi_rpa_Rspace takes the tau extent
  // from G_tskij, i.e. nt_f, and the bosonic FT halves nw_b.
  p.nth  = (p.nt_f % 2 == 0) ? p.nt_f / 2 : p.nt_f / 2 + 1;
  p.nwh  = (p.nw_b % 2 == 0) ? p.nw_b / 2 : p.nw_b / 2 + 1;

  p.ns      = mf->nspin();
  p.nsp_basis = long(mf->nspin_in_basis()) * long(mf->npol_in_basis());
  p.nk      = mf->nkpts();
  p.nki     = mf->nkpts_ibz();
  p.nq      = mf->nqpts();
  p.nqi     = mf->nqpts_ibz();
  p.nk_trev = mf->nkpts_trev_pairs();
  p.nb      = mf->nbnd();

  p.have_hf   = (mb_solver.hf != nullptr);
  p.have_corr = (mb_solver.corr != nullptr);
  p.have_scr  = (mb_solver.scr_eri != nullptr);
  p.is_gw     = std::is_same_v<corr_solver_t, solvers::gw_t> and p.have_corr;
  p.is_gf2    = std::is_same_v<corr_solver_t, solvers::gf2_t> and p.have_corr;
  if (p.have_scr) {
    p.screen_type   = mb_solver.scr_eri->screen_type();
    p.div_treatment = mb_solver.scr_eri->div_treatment();
  }

  // ERI reader properties. Only the correlated-solver reader's shapes are probed;
  // the others are assumed to share them (footnoted in the printed table).
  using corr_eri_t = std::decay_t<decltype(mb_eri.corr_eri->get())>;
  p.thc_eri = THC_ERI<corr_eri_t>;
  if constexpr (THC_ERI<corr_eri_t>) {
    auto& eri = mb_eri.corr_eri->get();
    p.NP = eri.Np();
    if constexpr (requires { eri.dZ_incore(); }) p.eri_incore = eri.dZ_incore();
    if constexpr (requires { eri.X_orbital_range() != eri.Y_orbital_range(); }) {
      p.separate_Y = (eri.X_orbital_range() != eri.Y_orbital_range());
    }
  }

  // Distinct THC readers among the populated mb_eri slots: each holds its own
  // collocation matrices and (incore) Coulomb kernel.
  std::vector<const void*> thc_readers;
  auto count_reader = [&](auto& slot) {
    if (!slot) return;
    using slot_eri_t = std::decay_t<decltype(slot->get())>;
    if constexpr (THC_ERI<slot_eri_t>) {
      const void* addr = static_cast<const void*>(std::addressof(slot->get()));
      if (std::find(thc_readers.begin(), thc_readers.end(), addr) == thc_readers.end())
        thc_readers.push_back(addr);
    }
  };
  count_reader(mb_eri.hf_eri);
  count_reader(mb_eri.hartree_eri);
  count_reader(mb_eri.exchange_eri);
  count_reader(mb_eri.corr_eri);
  p.n_thc_readers = int(thc_readers.size());

  // Self-energy algorithm. Mirrors gw_t::thc_gw_Xqindep (thc_gw.icc:31-32), which
  // picks "R" only on a symmetry-unreduced q mesh, and gf2_t's direct path, which
  // hardcodes "R" (thc_gf2.icc:153).
  p.sigma_alg = p.is_gf2 ? "R" : ((p.nq == p.nqi) ? "R" : "k");

  if (iter_solver != nullptr) {
    p.iter_alg = iter_scf::alg_enum_to_string(iter_solver->iter_alg());
    if (auto* diis = iter_solver->get_diis(); diis != nullptr) {
      p.diis_storage = diis->storage;
      p.max_subsp    = long(diis->max_subsp_size);
    }
  }

  p.keep_w = keep_w;
  p.dump_exchange = dump_exchange;
  p.eval_thermodynamics = eval_thermodynamics;

  return p;
}

/**
 * Predicted footprint of the large arrays of one scf_loop run: a verbosity-2
 * table (shape / GB / GB per node / location / lifetime) plus the persistent and
 * peak GB-per-node one-liners at verbosity 1. Printed before any allocation.
 */
/// Returns the predicted peak, in GB/node, so the caller can compare it with
/// what the run actually used (see the [MEM] report at the end of scf_loop).
double print_scf_memory_estimate(const utils::mpi_context_t<mpi3::communicator>& mpi,
                               const scf_mem_params& p);

/**
 * Predicted processor grids and block sizes of those arrays (verbosity 2), each
 * taken from the same helper the allocator itself calls.
 */
void print_scf_distribution_summary(const utils::mpi_context_t<mpi3::communicator>& mpi,
                                    const scf_mem_params& p);

} // methods

#endif // COQUI_SCF_MEM_REPORT_HPP
