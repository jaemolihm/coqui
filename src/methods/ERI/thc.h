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


#ifndef METHODS_ERI_THC_THC_H
#define METHODS_ERI_THC_THC_H

#include <tuple>
#include <iomanip>
#include <optional>

#include "configuration.hpp"
#include "IO/ptree/ptree_utilities.hpp"
#include "utilities/check.hpp"
#include "utilities/Timer.hpp"
#include "utilities/proc_grid_partition.hpp"
#include "grids/g_grids.hpp"
#include "potentials/potentials.hpp"

#include "mpi3/communicator.hpp"
#if defined(ENABLE_NCCL)
#include "mpi3/nccl/communicator.hpp"
#endif
#include "utilities/mpi_context.h"
#include "itertools/itertools.hpp"
#include "nda/nda.hpp"
#include "numerics/fft/nda.hpp"
#include "numerics/distributed_array/nda.hpp"
#include "numerics/distributed_array/h5.hpp"
#include "numerics/shared_array/nda.hpp"

#include "mean_field/MF.hpp"

namespace methods
{

/*
 * MAM: Fix problem that requires nqpool to exactly partition Qpts, this is unneccesary 
 */ 

namespace mpi3 = boost::mpi3;

class thc
{
  template<int N>
  using shape_t = std::array<long,N>;
  template<MEMORY_SPACE MEM, int N>
  using _darray_t_ = memory::darray_t<memory::array<MEM,ComplexType,N>,mpi3::communicator>;
  using mpi_context_t = utils::mpi_context_t<mpi3::communicator>;

  public:
  /*
   * Creates a thc object with arguments in property tree.
   *  Important options:
   *  - ecut: "1.4 * ecutwfc" (falls back to "0.4 * ecutrho" when no wfc grid is available),
   *          Plane wave cutoff used for the evaluation of coulomb matrix elements.
   *  - thresh: "1e-5", Threshold in cholesky decomposition.
   *  Performance related options:
   *  - matrix_block_size: 1024, Block size used in distributed arrays.
   *  - chol_block_size: "8", Block size in cholesky decomposition.
   *  - r_blk: "1", Number of iterations used to process real space grid in real space algorithm.  
   *  - distr_tol: "0.2". Controls the processor grid. Larger values lead to more processors in k/Q grid axis.
   *  - memory_frac: "0.75". fraction of available memory in a node used to estimate memory requirements/utilization. 
   */
  thc(mf::MF *mf_,
      mpi_context_t& mpi_,
      ptree const& pt,
      bool print_metadata_ = true);
 
  ~thc();

  thc(thc const&) = default;
  thc(thc &&) = default;
  thc& operator=(thc const&) = default;
  thc& operator=(thc &&) = default;

  void print_metadata();

  auto get_ecut() const { return ecut; }
  auto const& get_fft_mesh() const { return rho_g.mesh(); }
  // Semilocal xc-kernel matrix Vxc(q,u,v), built during evaluate() when
  // Vxc_file is set. Same shape/distribution as the Coulomb matrix, but a
  // separate object: it is only valid in the direct (Hartree) channel, so it
  // must never be folded into the Coulomb matrix returned by evaluate().
  // has_Vxc() is "the matrix exists"; vxc_requested() is "the input asked for it".
  // The two differ on any path that never reaches intvec_impl, so callers guarding a
  // get_Vxc() must use has_Vxc() and callers rejecting an unsupported algorithm must
  // use vxc_requested().
  bool has_Vxc() const { return Vxc_quv.has_value(); }
  bool vxc_requested() const { return vxc_file != ""; }
  std::string const& get_vxc_file() const { return vxc_file; }
  auto const& get_Vxc() const {
    utils::check(Vxc_quv.has_value(),
                 "thc::get_Vxc: the xc-kernel matrix has not been built. "
                 "Vxc_file requires the full ISDF path (evaluate()).");
    return Vxc_quv.value();
  }
  // Accessors used by thc_reader_t to persist rho_g and vG beyond the builder's lifetime.
  auto const& get_rho_g() const { return rho_g; }
  auto const& get_vG()   const { return vG; }

  /**
   * Compute collocation matrices on a given set of interpolating (pivot) points:
   *     Xa(s,k,a,u) = phi^{k}_a(r_u) * exp(i k.r_u),        a in a_range
   *     Xb(s,k,b,u) = phi^{k-q}_b(r_u) * exp(i (k-q).r_u),  b in b_range
   * The returned arrays have the same content, format and distribution as the
   * collocation matrices returned by interpolating_points(), so they can be
   * passed directly to evaluate(). Xb is std::nullopt when a_range == b_range
   * at q = Gamma (same convention as interpolating_points).
   * Only q = Gamma (iq with |Q|=0) is currently supported.
   *
   * @param IPts     - [INPUT] pivot point indices on the rho_g FFT grid, (Np)
   * @param iq       - [INPUT] index of the q-point (must be at Gamma)
   * @param a_range  - [INPUT] orbital range for phi^{k}_a
   * @param b_range  - [INPUT] orbital range for phi^{k-q}_b
   */
  template<MEMORY_SPACE MEM = HOST_MEMORY>
  auto interpolating_basis(memory::array<MEM,long,1> const& IPts, int iq=0,
              nda::range a_range = nda::range(-1,-1),
              nda::range b_range = nda::range(-1,-1))
       -> std::tuple<_darray_t_<MEM,4>, std::optional<_darray_t_<MEM,4>>>;

  /**
   * Compute interpolating points for phi^{k,*}_a(r)phi^{k-q}_b(r) at a given q-point.
   * @param iq       - [INPUT] Index of the q-point
   * @param max      - [INPUT] Maximum number of interpolating points.
   *                           If max=-1, there will be no hard limit, and the number of
   *                           interpolating points is computed until the error is smaller
   *                           than this->thresh.
   * @param a_range  - [INPUT] Orbital range for phi^{k,*}_a
   * @param b_range  - [INPUT] Orbital range for phi^{k-q}_b
   * @return A tuple containing:
   *         - index of interpolating points, (Np)
   *         - distributed array for phi^{k}_a on interpolating points: (ns, nkpts, nbnd_a, Np)
   *         - distributed array for phi^{k-q}_b on interpolating points: (ns, nkpts, nbnd_b, Np)
   */
  template<MEMORY_SPACE MEM = HOST_MEMORY>
  auto interpolating_points(int iq = 0, int max = -1,
              nda::range a_range = nda::range(-1,-1),
              nda::range b_range = nda::range(-1,-1))
       -> std::tuple<memory::array<MEM,long,1>,
                     _darray_t_<MEM,4>, 
                     std::optional<_darray_t_<MEM,4>>
                    >; 

  template<MEMORY_SPACE MEM = HOST_MEMORY>
  auto interpolating_points(nda::MemoryArrayOfRank<4> auto const& C_skai, 
              int iq = 0, int max = -1)
       -> std::tuple<memory::array<MEM,long,1>,
                     _darray_t_<MEM,4>,
                     std::optional<_darray_t_<MEM,4>>
                    >;

  /**
   * THC-OV:
   * Calculates interpolating vectors with density fitting in the overlap metric,
   * using precalculated interpolating points, and calculate Coulomb matrices.
   * All q-points are calculated simultaneously.
   *
   *      phi^{k*}_a(r)phi^{k-q}_b(r)
   *          = \sum_{\mu} phi^{k*}_a(r_mu)phi^{k-q}_b(r_mu) \zeta^{q}_mu(r)
   *
   * @param ri  - [INPUT] interpolating points
   * @param Xa  - [INPUT] orbital "a" on interpolating points: phi^{k*}_a(r_mu)
   * @param Xb  - [INPUT] orbital "b" on interpolating points: phi^{k-q*}_b(r_mu)
   * @param return_Sinv_Ivec - [INPUT] return inverse of overlap matrix for \zeta^{q}_mu(r)
   * @param a_range - [INPUT] range of orbital "a"
   * @param b_range - [INPUT] range of orbital "b"
   * @param pgrid3D - [INPUT] processor grid
   * @return A tuple containing:
   *         - V_{\mu,\nu}: Distributed array with Coulomb matrix in the basis of interpolating vectors.
   *                        dims: (nqpts_ibz, nIpts, nIpts), distributed along nIpts rows only.
   *         -
   */
  template<MEMORY_SPACE MEM = HOST_MEMORY, typename Tensor_t = _darray_t_<MEM,4>>
  auto evaluate(memory::array<MEM,long,1> const& ri, 
              Tensor_t const& Xa,
              std::optional<Tensor_t> const& Xb,
              bool return_Sinv_Ivec = false, 
              nda::range a_range = nda::range(-1,-1),
              nda::range b_range = nda::range(-1,-1),
              std::array<long, 3> pgrid3D = {0,0,0})
        -> std::tuple<_darray_t_<MEM,3>,
                      memory::array<MEM, ComplexType, 2>, memory::array<MEM, ComplexType, 2>, 
                      std::optional<_darray_t_<MEM,3>>
                     >;

  /**
   * THC-OV:
   * Calculates interpolating vectors with density fitting in the overlap metric,
   * using precalculated interpolating vectors. A rotation matrix can be provided, which will
   * be applied to the left hand side factor in the pair density product. The interpolating
   * points and collation matrices must be consistent with the results of interpolating_vectors
   * using an identical rotation matrix, otherwise results are incorrect.
   *
   * Returns Vuv, where:
   *  - Vuv: Distributed array with coulomb matrix in the basis of interpolating vectors.
   *    -> dims: [nIpts, nIpts], distributed along rows only.
   */
  template<MEMORY_SPACE MEM = HOST_MEMORY, typename Tensor_t = _darray_t_<MEM,4>>
  auto evaluate(memory::array<MEM,long,1> const& ri,
              nda::MemoryArrayOfRank<4> auto const& C_skai,
              Tensor_t const& Xa,
              Tensor_t const& Xb,
              bool return_Sinv_Ivec = false,
              std::array<long, 3> pgrid3D = {0,0,0})
        -> std::tuple<_darray_t_<MEM,3>,
                      memory::array<MEM, ComplexType, 2>, memory::array<MEM, ComplexType, 2>,
                      std::optional<_darray_t_<MEM,3>>
                     >;

  /**
   * THC-LS:
   *  Interpolating vectors are obtained by solving the overdetermined linear system, using
   *  the provided density fitting basis and precalculated interpolating points.
   *
   *  Given B(ab,n), solves (omitting spin and k-point index for simplicity):
   *
   *    B(ab,n) = sum_u conj(Pa(a,ru)) * Pb(b,ru) * I(u,n),
   *
   *  where ru are provided interpolating points.
   *
   * Returns a tuple with: Vuv, where:
   *  - Vuv: Distributed array (... with coulomb matrix) in the basis of interpolating vectors.
   *    -> dims: [nIpts, nIpts], distributed along rows only.
   */
  template<MEMORY_SPACE MEM = HOST_MEMORY>
  auto evaluate(int iq, memory::array<MEM,long,1> const& ri, 
              memory::darray_t<memory::array<MEM,ComplexType,5>,mpi3::communicator> const& B,
              nda::range a_range = nda::range(-1,-1),
              nda::range b_range = nda::range(-1,-1))
	-> memory::darray_t<memory::array<MEM,ComplexType,2>,mpi3::communicator>;

 /**
  * THC-DF:
  * Interpolating vectors are obtained by solving the overdetermined linear system, using
  * the provided density fitting basis and precalculated interpolating points.
  *
  *  Given Psi(is,ik,a,G), solves (omitting spin and kpoint index for simplicity):
  *
  *    B(q,ab,n) = sum_u conj(Pa(k,a,ru)) * Pb(k-q,b,ru) * I(q,u,n),
  *
  *  where ru are provided interpolating points, B(q,ab,n) are 3-center DF integrals
  *  calculated from Psi.
  *
  * Returns Vuv(q,u,v), where:
  *  - Vuv: Distributed array with the coulomb matrix, Vuv = sum_n I(q,u,n)*conj(I(q,v,n))
  *    -> dims: [nqpts_ibz,nIpts, nIpts], distributed along rows only.
  */
 /*
  template<MEMORY_SPACE MEM = HOST_MEMORY>
  auto evaluate(memory::array<MEM,long,1> const& ri,
              memory::darray_t<memory::array<MEM,ComplexType,3>,mpi3::communicator>& Psi,
              nda::range a_range = nda::range(-1,-1),
              nda::range b_range = nda::range(-1,-1),
              std::array<long, 3> pgrid3D = {0,0,0})
        -> memory::darray_t<memory::array<MEM,ComplexType,3>,mpi3::communicator>;
*/

  template<MEMORY_SPACE MEM = HOST_MEMORY, typename Tensor_t = _darray_t_<MEM,4>>
  void evaluate(h5::group& gh5, std::string format,
              memory::array<MEM,long,1> const& ri, 
              Tensor_t const& Xa,
              std::optional<Tensor_t> const& Xb,
              nda::range a_range = nda::range(-1,-1),
              nda::range b_range = nda::range(-1,-1),
              std::array<long, 3> pgrid3D = {0,0,0});

  template<MEMORY_SPACE MEM = HOST_MEMORY, typename Tensor_t = _darray_t_<MEM,4>>
  void evaluate(h5::group& gh5, std::string format,
              memory::array<MEM,long,1> const& ri,
              nda::MemoryArrayOfRank<4> auto const& C_skai,
              Tensor_t const& Xa,
              Tensor_t const& Xb,
              std::array<long, 3> pgrid3D = {0,0,0});

  template<MEMORY_SPACE MEM = HOST_MEMORY, typename Tensor_t = _darray_t_<MEM,4>>
  auto evaluate_isdf_only(memory::array<MEM,long,1> const& ri,
                          Tensor_t const& Xa,
                          std::optional<Tensor_t> const& Xb,
                          nda::range a_range = nda::range(-1,-1),
                          nda::range b_range = nda::range(-1,-1),
                          std::array<long, 3> pgrid3D = {0,0,0})
  -> _darray_t_<MEM,3>;

  /**
   * Saves the interpolating points and coulomb matrix to h5 group
   * @param gh5          - [INPUT]
   * @param format       - [INPUT]
   * @param ri           - [INPUT]
   * @param V            - [INPUT]
   * @param Z_head_qu    - [INPUT]
   * @param Zbar_head_qu - [INPUT]
   */
  template<MEMORY_SPACE MEM = HOST_MEMORY>
  void save(h5::group& gh5, std::string format, memory::array<MEM,long,1> const& ri,
        memory::darray_t<memory::array<MEM,ComplexType,3>,mpi3::communicator> const& V,
        memory::array<MEM,ComplexType,2> const& Z_head_qu,
        memory::array<MEM,ComplexType,2> const& Zbar_head_qu);

  /**
   * Save the interpolating points and interpolating vectors to h5 group.
   * @param gh5      - [INPUT]
   * @param format   - [INPUT]
   * @param ri       - [INPUT]
   * @param zeta_qur - [INPUT]
   */
  template<MEMORY_SPACE MEM = HOST_MEMORY>
  void save(h5::group& gh5, std::string format, memory::array<MEM,long,1> const& ri,
            memory::darray_t<memory::array<MEM,ComplexType,3>,mpi3::communicator> const& zeta_qur,
            bool write_zeta_on_fft_mesh=false);

  // writes metadata to h5 file, includes all information in addition to actual
  // thc vectors. File should be self-contained upon read
  void write_meta_data(h5::group& gh5, std::string format="default");
  void print_timers();
  void reset_timers() { Timer.reset(); }

  private:

  // mpi context with global, node, internode and gpu communicators
  mpi_context_t *mpi;

  // pointer to MF object
  mf::MF *mf = nullptr;

  utils::TimerManager Timer;

  // pw cutoff for density grid  
  double ecut = 0.0;

  // truncated density grid
  grids::truncated_g_grid rho_g; 

  // maps from the wfc truncated grid to full fft grid of rho_g  
  math::shm::shared_array<nda::array_view<long,1>> swfc_to_rho;

  // object that evaluates potential, v[G,Q]
  pots::potential_t vG;  

  long default_block_size;
  long default_cholesky_block_size;
  double thresh=1e-5;
  int nnr_blk = 1;
  double distr_tol = 0.2;
  double memory_frac = 0.75;
  bool use_least_squares = false;

  // Per-band fit weights, (nspin_in_basis, nkpts, nbnd), all > 0. Each orbital
  // leg of the pivot-search metric and of the zeta-fit normal equations is
  // scaled by its band weight (so each Gram factor carries w^2 per band); the
  // returned collocation matrices and all downstream ERIs stay unweighted.
  // Empty (has_band_weights = false) for trivial weights, or when neither the
  // augmentation singular values nor nbnd_protected supply any.
  nda::array<double,3> band_weight;
  bool has_band_weights = false;

  // Number of protected bands N_P. When > 0, thc builds its own band_weight
  // table: 1 for b < N_P and (E(s,k,N_P-1) - mu)/(E(s,k,b) - mu) for b >= N_P,
  // i.e. the band tail is energy-suppressed in the pivot search and the zeta
  // fit. -1 disables. On an augmented mean field the energy weights stop at the
  // nbnd_orig input and the augmentation block keeps its singular values.
  long nbnd_protected = -1;
  // Research diagnostic: when true, drop the unprotected-unprotected pair
  // densities (M_keep = M_full - M_aug) from the pivot-point-selection metric
  // and the host zeta-fit. The unprotected set is the band tail
  // [unprot_band_start, nbnd), independent of any band weights. Default false
  // leaves both paths byte-identical.
  bool exclude_unprotected_pairs = false;
  long unprot_band_start = -1;

  // Semilocal xc-kernel matrix. An empty path (the default) disables the whole
  // path and leaves the Coulomb-only behavior byte-identical. When set, the
  // interpolating vectors are contracted against the kernel fields in that file
  // while they are still available inside intvec_impl, and the result is picked
  // up by thc_reader_t through get_Vxc(). See thc_xc_kernel.hpp.
  // std::optional: a default-constructed distributed_array carries a null
  // communicator, and copying/moving one aborts in check_dimensions().
  std::string vxc_file = "";
  long vxc_block_size = 64;
  std::optional<_darray_t_<HOST_MEMORY,3>> Vxc_quv;

  //fft plans
  int howmany_fft = -1;

  /**
   * Solves normal equation given a set of interpolating points and three-index tensor B
   * (needs to be careful with stability issues)
   *
   *
   * @tparam MEM
   * @tparam return_coul_matrix
   * @param iq
   * @param IPoints
   * @param a_range
   * @param b_range
   * @param B
   * @return
   */
  template<MEMORY_SPACE MEM = HOST_MEMORY, bool return_coul_matrix>
  auto intvec_impl(int iq, nda::MemoryArrayOfRank<1> auto const& IPoints,
        nda::range a_range, nda::range b_range,
        memory::darray_t<memory::array<MEM,ComplexType,5>,mpi3::communicator> const& B);

  /**
   * Solves normal equations for interpolating vectors with density fitting in the overlap metric,
   * using precalculated interpolating points, and calculate the Coulomb matrices.
   * All q-points are calculated simultaneously.
   *
   *      \phi^{k*}_a(r)\phi^{k-q}_b(r)
   *          = \sum_{\mu} \phi^{k*}_a(r_{\mu})\phi^{k-q}_b(r_{\mu}) \zeta^{q}_{\mu}(r)
   *
   * @param ri  - [INPUT] interpolating points
   * @param Xa  - [INPUT] orbital "a" on interpolating points: phi^{k*}_a(r_mu)
   * @param Xb  - [INPUT] orbital "b" on interpolating points: phi^{k-q*}_b(r_mu)
   * @param return_Sinv_Ivec - [INPUT] return inverse of overlap matrix for \zeta^{q}_mu(r)
   * @param a_range - [INPUT] range of orbital "a"
   * @param b_range - [INPUT] range of orbital "b"
   * @param pgrid3D - [INPUT] processor grid
   * @return A tuple containing:
   *         if return_coul_matrix == False:
   *           - Z_quG: Distributed array with interpolating vectors. Depending
   *             dims: (nqpts_ibz, nIpts, nG if (mf->orb_on_fft_grid()) else nr)
   *         else:
   *           - V_{\mu,\nu}: Distributed array with Coulomb matrix in the basis of interpolating vectors.
   *             dims: (nqpts_ibz, nIpts, nIpts), distributed along nIpts rows only.
   *           - \zeta^{q}_{\mu}(G=0):
   *           - \tilde{\zeta}^{q}_{\mu}(G=0):
   */
  template<MEMORY_SPACE MEM = HOST_MEMORY, bool return_coul_matrix, typename Tensor_t, typename Tensor2_t>
  auto intvec_impl(nda::MemoryArrayOfRank<1> auto const& IPoints, 
        Tensor_t const& Xa,
        Tensor2_t const* Xb,
        bool return_Sinv_Ivec, nda::range a_range, nda::range b_range, 
        std::array<long, 3> pgrid3D={0,0,0});
  /**
   * Calculate the following quantity for orbitals stored on a non-uniform real-space grid:
   *     T_{uv} = \sum_{i} \sum_{k} \phi^{k*}_{i}(r_u)\phi^{k}_{i}(r_v)
   * This quantity is needed when solving the normal equations for interpolating vectors.
   * @param add_phase - [INPUT]
   * @param ispin     - [INPUT]
   * @param k         - [INPUT]
   * @param orb_range - [INPUT]
   * @param IPts      - [INPUT]
   * @param Tuv       - [OUTPUT]
   */
  template<MEMORY_SPACE MEM = HOST_MEMORY>
  void get_Tuv_nonuniform_rgrid(bool add_phase, int ispin, int k,
        nda::range orb_range, memory::array<MEM,long,1> const& IPts,
        memory::darray_t<memory::array<MEM,ComplexType,2>,mpi3::communicator>& Tuv);

  /**
   * Calculate the following quantity for orbitals stored on a FFT grid:
   *     T{uv} = \sum_{i} \sum_{k} \phi^{k*}_{i}(r_u)\phi^{k}_{i}(r_v)
   * This quantity is needed when solving the normal equations for interpolating vectors.
   * @param add_phase  - [INPUT]
   * @param ispin      - [INPUT]
   * @param k          - [INPUT]
   * @param a_range    - [INPUT]
   * @param ru         - [INPUT]
   * @param Tuv        - [OUTPUT]
   */
  template<MEMORY_SPACE MEM = HOST_MEMORY>
  void get_Tuv_fft_grid(bool add_phase, int ispin, int k, 
        nda::range a_range, memory::array<MEM,long,1> const& ru,
        memory::darray_t<memory::array<MEM,ComplexType,2>,mpi3::communicator>& Tuv);

  /**
   * Calculate the following quantity for orbitals stored on a FFT grid:
   *     T^{k}_{ur} = \sum_{i} \phi^{k*}_{i}(r_u)\phi^{k}_{i}(r)
   * This quantity is needed when solving the normal equations for interpolating vectors.
   * @param add_phase - [INPUT]
   * @param ispin     - [INPUT]
   * @param a_range   - [INPUT]
   * @param r_range   - [INPUT]
   * @param ru        - [INPUT]
   * @param Tkur      - [OUTPUT]
   */
  template<MEMORY_SPACE MEM = HOST_MEMORY>
  void get_Tkur_fft_grid(bool add_phase, int ispin, nda::range a_range, nda::range r_range,
        memory::array<MEM,long,1> const& ru,
        memory::darray_t<memory::array<MEM,ComplexType,3>,mpi3::communicator>& Tkur);

  /**
   * Calculate the following quantity for orbitals stored on a non-uniform grid:
   *     T^{k}_{ur} = \sum_{i} \phi^{k*}_{i}(r_u)\phi^{k}_{i}(r)
   * This quantity is needed when solving the normal equations for interpolating vectors.
   * @param add_phase - [INPUT]
   * @param ispin     - [INPUT]
   * @param a_range   - [INPUT]
   * @param r_range   - [INPUT]
   * @param ru        - [INPUT]
   * @param Tkur      - [OUTPUT]
   */
  template<MEMORY_SPACE MEM = HOST_MEMORY>
  void get_Tkur_nonuniform_rgrid(bool add_phase, int ispin, nda::range a_range, nda::range r_range,
         memory::array<MEM,long,1> const& ru,
         memory::darray_t<memory::array<MEM,ComplexType,3>,mpi3::communicator>& Tkur);
  /**
   *
   * @param ispin
   * @param kp_to_ibz
   * @param kp_order
   * @param Xa
   * @param psi
   * @param dT_g
   * @param dT_u
   */
  template<MEMORY_SPACE MEM = HOST_MEMORY, typename dArray_t, typename dArray2_t>
  void get_Tkug(int ispin, int ipol, nda::array<int,1> const& kp_to_ibz,
                nda::array<int,1> const& kp_order,
                nda::ArrayOfRank<3> auto const& Xa, _darray_t_<MEM,5> const& psi,
                dArray_t& dT_g, dArray2_t& dT_u);

  template<MEMORY_SPACE MEM = HOST_MEMORY, typename dArray_t>
  void get_Tkug(int ispin, int ipol, nda::array<int,1> const& kp_to_ibz,
                nda::array<int,1> const& kp_order,
                nda::ArrayOfRank<3> auto const& Xa, _darray_t_<MEM,5> const& psi,
                dArray_t& dT_u);

  // wska (optional): per-band fit weights (spin, full-BZ k, band on Xa's band
  // axis) multiplied into the copied X (weighting one leg of the fit).
  template<MEMORY_SPACE MEM = HOST_MEMORY, typename Tensor_t>
  auto Xskau_to_sXbkua(int ispin, nda::ArrayOfRank<1> auto const& iu_for_sXb,
                       Tensor_t const& Xa, nda::array<int,1>& kp_order,
                       nda::array<double,3> const* wska = nullptr);

  //template<MEMORY_SPACE MEM = HOST_MEMORY, typename Tensor_t>
  //auto Xskau_to_sXbkua(int ispin, nda::ArrayOfRank<2> auto const& iu_for_sXb,
  //                     Tensor_t const& Xa, nda::array<int,1>& kp_order);

  // Scale the bands of a freshly-read (ns, nkpts_ibz, nb, npol, g) orbital set
  // by the per-band fit weights; orb_range maps the band axis to absolute band
  // indices. No-op when has_band_weights is false.
  template<MEMORY_SPACE MEM>
  void apply_band_weights_psi(memory::darray_t<memory::array<MEM,ComplexType,5>,mpi3::communicator>& dPsi,
                              nda::range orb_range);

  // Weight table for the X (pivot) leg: (nspin, full-BZ k, band-in-orb_range).
  auto make_band_weight_table(nda::range orb_range) const
  {
    utils::check(orb_range.first() >= 0 and orb_range.last() <= band_weight.extent(2),
                 "make_band_weight_table: orbital range [{},{}) exceeds the band-weight table ({} bands).",
                 orb_range.first(), orb_range.last(), band_weight.extent(2));
    nda::array<double,3> w(band_weight.extent(0), band_weight.extent(1), orb_range.size());
    for( long is=0; is<w.extent(0); ++is )
      for( long k=0; k<w.extent(1); ++k )
        for( long ia=0; ia<w.extent(2); ++ia )
          w(is,k,ia) = band_weight(is,k,orb_range.first()+ia);
    return w;
  }

  // Effective (weight-squared) band count of orb_range for the metric
  // normalization: with band weights, near-null states contribute ~nothing to
  // the weighted metric but would still inflate the plain 1/(na*nb) factor, so
  // the same absolute `thresh` would terminate the pivoted Cholesky earlier as
  // more of them are kept. Normalizing by sum_a w^2 (averaged over spin and the
  // full BZ) keeps the metric - and hence nIpts at fixed thresh - independent of
  // how many near-null states are retained, and identical on the full-BZ and
  // symmetry-reduced pivot paths. Reduces to orb_range.size() for unit weights.
  double effective_band_count(nda::range orb_range) const
  {
    if(not has_band_weights) return double(orb_range.size());
    long ns = band_weight.extent(0), nk = band_weight.extent(1);
    double w2 = 0.0;
    for( long is=0; is<ns; ++is )
      for( long k=0; k<nk; ++k )
        for( long a=0; a<orb_range.size(); ++a ) {
          double w = band_weight(is,k,orb_range.first()+a);
          w2 += w*w;
        }
    double n_eff = w2 / double(ns*nk);
    app_log(2,"  thc: effective weighted band count for metric normalization: "
              "n_eff = {:.3f} (n = {})", n_eff, orb_range.size());
    return n_eff;
  }

  /**
   * Build Vxc_quv from the interpolating vectors, in the same shape and
   * distribution as the Coulomb matrix. Called from intvec_impl while Z_quG
   * still holds the unweighted interpolating vectors (the Coulomb step scales
   * them by sqrt(v(G)) in place afterwards).
   *
   * @param Z_quG      - [INPUT] interpolating vectors zeta^q_u(G), distributed
   * @param pgrid3D    - [INPUT] processor grid of the Coulomb matrix
   * @param block_size - [INPUT] block sizes of the Coulomb matrix
   */
  template<typename DArr_t>
  void build_Vxc_quv(DArr_t const& Z_quG,
                     std::array<long, 3> pgrid3D,
                     std::array<long, 3> block_size);

  /**
   * Compute
   *     - Z^{q}_u(G) or Z^{q}_u(r):
   *          \sum_{ab} \sum_{k} \phi^{k}_a(r_u}) \phi^{k-q*}_b(r_u) \phi^{k*}_a(G or r) \phi^{k-q}_b(G or r)
   *     - C^{q}_{uv}: \sum_{ab} \sum_{k} \phi^{k}_a(r_u}) \phi^{k-q*}_b(r_u) \phi^{k*}_a(r_v) \phi^{k-q}_b(r_v)
   * These are intermediate quantities used in the normal equation for interpolating vectors.
   * @param IPts    - [INPUT] Interpolating points
   * @param Xa      - [INPUT] orbital "a" on interpolating points: phi^{k*}_a(r_mu)
   * @param Xb      - [INPUT] orbital "b" on interpolating points: phi^{k-q*}_b(r_mu)
   * @param a_range - [INPUT] range of orbital "a"
   * @param b_range - [INPUT] range of orbital "b"
   * @param pgrid   - [INPUT] processor grid for ZquG and Cquv
   * @return A tuple containing:
   *         - Z^{q}_u(r) or Z^{q}_u(G): Distributed array with interpolating vectors.
   *           dims: (nqpts_ibz, nIpts, nG if (mf->orb_on_fft_grid()) else nr)
   *         - Zquv: distributed array with dimensions: (nqpts_ibz, nIpts, n_Ipts)
   */
  template<MEMORY_SPACE MEM, typename Tensor_t, typename Tensor2_t>
  auto get_ZquG_Cquv(nda::MemoryArrayOfRank<1> auto const& IPts,
                     Tensor_t const& Xa,
                     Tensor2_t const* Xb,
                     nda::range a_range, nda::range b_range,
                     std::array<long, 3> pgrid); 

  /**
   * Compute the following quantities on a real-space grid:
   *     - Z^{q}_u(r): \sum_{ab} \sum_{k} \phi^{k}_a(r_u}) \phi^{k-q*}_b(r_u) \phi^{k*}_a(r) \phi^{k-q}_b(r)
   *     - C^{q}_{uv}: \sum_{ab} \sum_{k} \phi^{k}_a(r_u}) \phi^{k-q*}_b(r_u) \phi^{k*}_a(r_v) \phi^{k-q}_b(r_v)
   * Z^{q}_{u}(r) is Fourier transformed to the plane-waves basis if mf->orb_on_fft_grid() = true.
   * @param IPts    - [INPUT] Interpolating points
   * @param a_range - [INPUT] range of orbital "a"
   * @param b_range - [INPUT] range of orbital "b"
   * @param pgrid   - [INPUT] processor grid for ZquG and Cquv
   * @param block_size - [INPUT]
   * @return A tuple containing:
   *         - Z^{q}_u(r) or Z^{q}_u(G): Distributed array with interpolating vectors.
   *           dims: (nqpts_ibz, nIpts, nG if (mf->orb_on_fft_grid()) else nr)
   *         - Zquv: distributed array with dimensions: (nqpts_ibz, nIpts, n_Ipts)
   */
  template<MEMORY_SPACE MEM>
  auto get_ZquG_Cquv_rspace(nda::MemoryArrayOfRank<1> auto const& IPts,
                     nda::range a_range, nda::range b_range,
                     std::array<long, 3> pgrid,   
                     std::array<long, 3> block_size);

  template<typename Tensor_t, typename Tensor2_t>
  auto get_ZquG_Cquv_fft_shared_memory(nda::MemoryArrayOfRank<1> auto const& IPts,
                     Tensor_t const& Xa,
                     Tensor2_t const* Xb,
                     nda::range a_range, nda::range b_range,
                     std::array<long, 3> pgrid,   
                     std::array<long, 3> block_size);

  /**
   * Compute the following quantities on the plane-wave basis using FFT:
   *     - Z^{q}_u(G): \sum_{ab} \sum_{k} \phi^{k}_a(r_u}) \phi^{k-q*}_b(r_u) \phi^{k*}_a(G) \phi^{k-q}_b(G)
   *     - C^{q}_{uv}: \sum_{ab} \sum_{k} \phi^{k}_a(r_u}) \phi^{k-q*}_b(r_u) \phi^{k*}_a(r_v) \phi^{k-q}_b(r_v)
   * @param IPts    - [INPUT] Interpolating points
   * @param a_range - [INPUT] range of orbital "a"
   * @param b_range - [INPUT] range of orbital "b"
   * @param pgrid   - [INPUT] processor grid for ZquG and Cquv
   * @param block_size - [INPUT]
   * @return A tuple containing:
   *         - Z^{q}_u(G): Distributed array with interpolating vectors.
   *           dims: (nqpts_ibz, nIpts, nG)
   *         - Zquv: distributed array with dimensions: (nqpts_ibz, nIpts, n_Ipts)
   */
  template<MEMORY_SPACE MEM = HOST_MEMORY, typename Tensor_t, typename Tensor2_t>
  auto get_ZquG_Cquv_fft(nda::MemoryArrayOfRank<1> auto const& IPts,
                     Tensor_t const& Xa,
                     Tensor2_t const* Xb,
                     nda::range a_range, nda::range b_range,
                     std::array<long, 3> pgrid,
                     std::array<long, 3> block_size);

  template<MEMORY_SPACE MEM = HOST_MEMORY, bool Ipts_only, bool return_Ruv, typename Tensor_t>
  auto chol_metric_impl(int iq, int nmax, nda::range a_range, nda::range b_range, int block_size, 
                        Tensor_t const* C_skai);

  /**
   * [symmetry-adapted version]
   * ISDF for phi^{k,*}_a(r)phi^{k-q}_b(r) at a given q-point using the Cholesky decomposition method
   * from Matthews D. A., J. Chem. Theory Comput. 2020, 16, 1382–1385.
   * @tparam Ipts_only   - interpolating points only
   * @tparam return_Ruv  -
   * @param iq           - [INPUT]  Index of the q-point
   * @param nmax         - Maximum number of interpolating points.
   *                       If max=-1, there will be no hard limit, and the number of
   *                       interpolating points is computed until the error is smaller
   *                       than this->thresh.
   * @param a_range  - [INPUT] Orbital range for phi^{k,*}_a
   * @param b_range  - [INPUT] Orbital range for phi^{k-q}_b
   * @param block_size - [INPUT] block size for the iterative pivoted Cholesky algorithm
   * @return A tuple containing:
   *         if Ipts_only is True:
   *           - index of interpolating points: (Np)
   *           - distributed array for phi^{k}_a on interpolating points: (ns, nkpts, nbnd_a, Np)
   *           - distributed array for phi^{k-q}_b on interpolating points: (ns, nkpts, nbnd_b, Np)
   *         else:
   *           if return_Ruv is True:
   *             - index of interpolating points: (Np)
   *             -
   *             - distributed array for phi^{k}_a on interpolating points: (ns, nkpts, nbnd_a, Np)
   *             - distributed array for phi^{k-q}_b on interpolating points: (ns, nkpts, nbnd_b, Np)
   *           else:
   *             - index of interpolating points: (Np)
   *             -
   *             - distributed array for phi^{k}_a on interpolating points: (ns, nkpts, nbnd_a, Np)
   *             - distributed array for phi^{k-q}_b on interpolating points: (ns, nkpts, nbnd_b, Np)
   */
  template<MEMORY_SPACE MEM = HOST_MEMORY, bool Ipts_only, bool return_Ruv>
  auto chol_metric_impl_ibz(int iq, int nmax, nda::range a_range, nda::range b_range, int block_size);
  //auto chol_metric_impl(int iq, int nmax, nda::range a_range, nda::range b_range, int block_size);


  template<MEMORY_SPACE MEM = HOST_MEMORY>
  auto load_basis_subset_nonuniform_rgrid(nda::MemoryArrayOfRank<1> auto const& IPts,
          int iq, nda::range kp_rg, nda::range a_rg, nda::range b_rg);

  template<MEMORY_SPACE MEM = HOST_MEMORY>
  auto load_basis_subset_fft_grid(nda::MemoryArrayOfRank<1> auto const& IPts,
          int iq, nda::range kp_rg, nda::range a_rg, nda::range b_rg);

  template<typename Arr>
  auto chol(Arr& A, nda::array<int,1>& piv, double cut);

  void set_range(nda::range& a_range); 

  void set_k_range(nda::range& k_range); 

  long get_nblocks_nnr() { return nnr_blk; }
};

} // methods

#endif
