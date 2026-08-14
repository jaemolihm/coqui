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


#ifndef COQUI_HF_T_H
#define COQUI_HF_T_H

#include <optional>

#include "mpi3/communicator.hpp"
#include "nda/nda.hpp"
#include "numerics/distributed_array/nda.hpp"
#include "numerics/shared_array/nda.hpp"

#include "IO/app_loggers.h"
#include "utilities/Timer.hpp"

#include "mean_field/MF.hpp"
#include "methods/ERI/detail/concepts.hpp"

namespace methods {
  namespace solvers {

    namespace mpi3 = boost::mpi3;
    /**
     * The Hartree-Fock solver to compute the Fock matrix.
     * One-body Hamiltonian is provided from MF while density matrix
     * and electron repulsion integrals are provided at runtime.
     *
     * Usage:
     *   hf_t myhf(comm, &myMF);
     *   myhf.evaluate(F_thc, Dm, thc_eri); // THC-HF
     *   myhf.evaluate(F_chol, Dm, cholesky_eri); // Cholesky-HF
     *
     */
    class hf_t {
    public:
      template<nda::MemoryArray local_Array_t>
      using dArray_t = math::nda::distributed_array<local_Array_t,mpi3::communicator>;
      template<nda::Array Array_base_t>
      using sArray_t = math::shm::shared_array<Array_base_t>;
      template<int N>
      using shape_t = std::array<long,N>;

    public:
      hf_t(std::string div = "gygi");

      ~hf_t() = default;

      /* THC-HF */
      /**
       * F_skij from thc-type ERIs
       * @param Dm_skij - [INPUT] Density matrix in the primary basis
       * @param thc     - [INTPUT] thc-type ERIs
       * @return Fock matrix in the primary basis
       */
      template<nda::MemoryArray AF_t>
      void evaluate(sArray_t<AF_t> &sF_skij,
                    const nda::MemoryArrayOfRank<4> auto &Dm_skij, THC_ERI auto &&thc,
                    const nda::MemoryArrayOfRank<4> auto &S_skij,
                    bool hartree=true, bool exchange=true);

      /* Cholesky-HF */
      /**
       * Fock matrix from Cholesky-type ERIs
       * @param Dm_skij - [INPUT] density matrix in the primary basis
       * @param chol    - [INPUT] Cholesky-type eri
       * @return Fock matrix in the primary basis
       */
      template<nda::MemoryArray AF_t>
      void evaluate(sArray_t<AF_t> &sF_skij,
                    const nda::MemoryArrayOfRank<4> auto &Dm_skij, Cholesky_ERI auto &&chol,
                    const nda::MemoryArrayOfRank<4> auto &S_skij,
                    bool hartree=true, bool exchange=true);

      /**
       * Coulomb matrix J from Cholesky-type ERIs
       * @param Dm_skij - [INPUT] density matrix in the primary basis
       * @param chol    - [INPUT] Cholesky-type eri
       * @return J matrix in the primary basis
       */
      template<nda::MemoryArray AF_t>
      void add_J(sArray_t<AF_t> &sF_skij, const nda::MemoryArrayOfRank<4> auto &Dm_skij,
                 Cholesky_ERI auto &&chol);

      /**
       * Exchange matrix K from Cholesky-type ERIs
       * @param Dm_skij - density matrix in the primary basis
       * @param chol    - Cholesky-type eri
       * @param comm    - communicator (optional)
       * @return Exchange matrix K in the primary basis
       */
      template<nda::MemoryArray AF_t>
      auto add_K(sArray_t<AF_t> &sF_skij, const nda::MemoryArrayOfRank<4> auto &Dm_skij,
                 Cholesky_ERI auto &&chol, const nda::MemoryArrayOfRank<4> auto &S_skij);

      /**
       * Finite-size correction for K based on "PRB 80, 085114(2009)"
       * @param Dm_skij - [INPUT] density matrix in the primary basis
       * @return finite-correction for K in the primary basis
       */
      template<nda::MemoryArray AF_t>
      void HF_K_correction(sArray_t<AF_t> &sF_skij, const nda::MemoryArrayOfRank<4> auto &Dm_skij, 
                           const nda::MemoryArrayOfRank<4> auto &S_skij, double madelung);

      std::string& div_treatment() { return _div_treatment; }
      void print_chol_hf_timers(); 
      void print_thc_hf_timers(); 

    private:
      std::string _div_treatment;

      utils::TimerManager _Timer;

      // Cache of the redistributed bare Coulomb matrix returned by thc.dZ({1,np_P,np_Q}).
      // The bare Coulomb interaction is iteration-invariant across an SCF loop, and the
      // redistribute is arithmetic-free data movement, so a cached copy is byte-identical
      // to a fresh redistribute (bit-exact). hf_t and its THC reader both persist across
      // SCF iterations (mb_solver.hf), so this replaces a per-Fock-build MPI redistribute
      // of the full Coulomb matrix (~nqpts_ibz*Np*Np complex, distributed) with a local
      // copy. Only used for incore readers. A fresh copy is returned each call because the
      // exchange path overwrites the array in place (U(q)->U(R)). Keyed on the reader's
      // monotonic instance id (an address would be recycled by the allocator, so a
      // destroyed reader replaced at the same address would hit a stale cache), basis
      // dimensions, and target processor grid.
      using dCoulomb_t = math::nda::distributed_array<nda::array<ComplexType,3>, mpi3::communicator>;
      std::optional<dCoulomb_t> _dZ_cache;
      long _dZ_cache_src = -1;
      std::array<long,3> _dZ_cache_pgrid{0,0,0};
      long _dZ_cache_nq = -1;
      long _dZ_cache_np = -1;

      /**
       * Single k-point (molecular) Coulomb matrix J w/o SOC from Cholesky-type ERIs.
       * Uses real arithmetic since integrals are real for molecules and Gamma-only cases.
       */
      template<nda::MemoryArray AF_t>
      void add_J_mol(sArray_t<AF_t> &sF_skij, const nda::MemoryArrayOfRank<4> auto &Dm_skij,
                     Cholesky_ERI auto &&chol);

      /**
       * Single k-point (molecular) exchange matrix K w/o SOC from Cholesky-type ERIs.
       * Uses real arithmetic since integrals are real for molecules and Gamma-only cases.
       */
      template<nda::MemoryArray AF_t>
      auto add_K_mol(sArray_t<AF_t> &sF_skij, const nda::MemoryArrayOfRank<4> auto &Dm_skij,
                     Cholesky_ERI auto &&chol, const nda::MemoryArrayOfRank<4> auto &S_skij);

      /**
       * THC-HF implementation for q-independent interpolating points
       * @param Dm_skij
       * @param F_skij
       * @param thc
       */
      template<nda::MemoryArray AF_t>
      void thc_hf_Xqindep(const nda::MemoryArrayOfRank<4> auto &Dm_skij,
                          sArray_t<AF_t> &sF_skij, THC_ERI auto &thc,
                          const nda::MemoryArrayOfRank<4> auto &S_skij,
                          bool compute_hartree=true, bool compute_exchange=true);

      /**
       * THC-HF implementation for q-independent interpolating points with symmetry
       * @param Dm_skij
       * @param F_skij
       * @param thc
       */
      template<nda::MemoryArray AF_t>
      void thc_hf_Xqindep_wsymm(const nda::MemoryArrayOfRank<4> auto &Dm_skij,
                                sArray_t<AF_t> &sF_skij, THC_ERI auto &thc,
                                const nda::MemoryArrayOfRank<4> auto &S_skij,
                                bool compute_hartree=true, bool compute_exchange=true);

    };
  }
} // methods

#endif //COQUI_HF_T_H
