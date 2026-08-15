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



#include <algorithm>
#include <cmath>
#include <tuple>
#include <iomanip>

#include "configuration.hpp"
#include "IO/ptree/ptree_utilities.hpp"
#include "utilities/check.hpp"
#include "utilities/Timer.hpp"
#include "utilities/freemem.h"
#include "utilities/proc_grid_partition.hpp"
#include "utilities/mpi_context.h"
#include "arch/arch.h"
#include "grids/g_grids.hpp"
#include "hamiltonian/potentials.hpp"

#include "itertools/itertools.hpp"
#include "nda/nda.hpp"
#include "numerics/fft/nda.hpp"
#include "numerics/distributed_array/nda.hpp"
#include "numerics/distributed_array/h5.hpp"

#include "mean_field/MF.hpp"
#include "methods/ERI/thc_xc_kernel.hpp"

#include "methods/ERI/thc.h"

namespace methods
{

namespace detail 
{

// encapsulates details of construction of rho_g 
auto make_grid(utils::Communicator auto&& comm, double ecut, mf::MF& mf)
{
  // until you propagate the change everywhere
  if(not mf.has_wfc_grid() or (ecut <= 0) or (std::abs(ecut-mf.ecutrho()) < 1e-3) ) 
    return grids::truncated_g_grid( mf.ecutrho(), mf.fft_grid_dim(), mf.recv() ); 
  auto mesh = grids::find_fft_mesh(comm,ecut,mf.recv(),mf.symm_list()); 
  auto wfc_mesh = mf.wfc_truncated_grid()->mesh();
  if( mesh[0] < wfc_mesh[0] or
      mesh[1] < wfc_mesh[1] or
      mesh[2] < wfc_mesh[2] ) {
    return grids::truncated_g_grid( ecut, wfc_mesh, mf.recv() );
  } else {
    return grids::truncated_g_grid( ecut, mesh, mf.recv() );
  }
}

// The interpolating vectors are truncated to rho_g before the ISDF fit, so at most
// rho_g.size() of them can be linearly independent. Beyond that the THC overlap
// matrix S = Z*dagger(Z) is exactly singular; its LU inverse in thc::evaluate() still
// returns info==0 on roundoff-sized pivots, so the integrals come out silently wrong
// rather than failing. Refuse to build in that regime.
void check_npts_vs_pw(long nIpts, long npw, double ecut)
{
  utils::check( nIpts <= npw,
                "thc::interpolating_points: number of interpolating points ({}) exceeds the "
                "number of plane waves in the auxiliary basis ({}) at ecut = {} a.u.\n"
                "  The THC integrals would be rank deficient. Increase 'ecut', or raise "
                "'thresh' / lower 'nIpts' to keep the point count at or below {}.",
                nIpts, npw, ecut, npw);
}

auto make_wfc_to_rho(utils::mpi_context_t<mpi3::communicator>& mpi,
                   grids::truncated_g_grid const& wfc_g,
                   grids::truncated_g_grid const& rho_g)
{
  using arr_t = math::shm::shared_array<nda::array_view<long,1>>;
  long ngm = wfc_g.size();
  arr_t swfc_to_rho(mpi,std::array<long,1>{ngm});
  if(mpi.comm.root()) {
    grids::map_truncated_grids(true,wfc_g,rho_g,swfc_to_rho.local());
    mpi.internode_comm.broadcast_n(swfc_to_rho.local().data(),ngm,0); 
  } else if(mpi.node_comm.root()) { 
      mpi.internode_comm.broadcast_n(swfc_to_rho.local().data(),ngm,0); 
  }
  mpi.node_comm.barrier();
  return swfc_to_rho;
}

}

/*
 * Creates a thc object with arguments in property tree.
 *  Important options:
 *  - ecut: "1.4 * ecutwfc" (falls back to "0.4 * ecutrho" when no wfc grid is available),
 *          Plane wave cutoff used for the evaluation of coulomb matrix elements.
 *  - thresh: "1e-5", Threshold in cholesky decomposition.
 *  - band_weights: "true". For an augmented mean field, weight the bands by the
 *          stored augmentation singular values in the pivot search and zeta fit.
 *  - nbnd_protected: "-1". Number of protected bands N_P. When > 0, the bands
 *          b >= N_P are energy suppressed in the pivot search and zeta fit by
 *          the weight (E(s,k,N_P-1) - mu)/(E(s,k,b) - mu), with mu the Fermi
 *          energy of the mean field. Requires a mean field carrying a Fermi
 *          energy. On an augmented mean field the energy weights stop at
 *          nbnd_orig (see below) and the augmentation block keeps its singular
 *          values. N_P == nbnd is accepted as an explicit no-op: the
 *          unprotected tail is empty, so no weights are built and
 *          exclude_unprotected_pairs is forced off, reproducing plain THC while
 *          still checking the preconditions.
 *  - nbnd_orig: "-1". Number of original (non-augmentation) bands of an
 *          augmented mean field, i.e. the start of the augmentation block in
 *          the [originals | augmentation] basis. Required with nbnd_protected
 *          on an augmented mean field, meaningless otherwise: thc cannot infer
 *          the block boundary from the mean field. Energy weights then cover
 *          [nbnd_protected, nbnd_orig) and the stored augmentation singular
 *          values cover [nbnd_orig, nbnd).
 *  - exclude_unprotected_pairs: "nbnd_protected > 0". Research diagnostic. When
 *          true, drop the unprotected-unprotected pair densities
 *          (M_keep = M_full - M_aug) from the pivot-point-selection metric and,
 *          in make_thc_coulomb, the host zeta-fit. The unprotected set is the
 *          band tail [nbnd_protected, nbnd), so nbnd_protected is required. On
 *          an augmented mean field that tail runs past nbnd_orig, i.e. the
 *          augmentation states count as unprotected.
 *  - Vxc_file: "". Path to an HDF5 file holding semilocal xc-kernel coefficient
 *          fields (QE elph.x with write_xc_kernel = .true.). When non-empty, the
 *          xc-kernel matrix Vxc(q,u,v) is built alongside the Coulomb matrix and
 *          exposed by get_Vxc(). Empty (default) leaves the Coulomb path
 *          untouched. Only valid for the ISDF algorithm.
 *  Performance related options:
 *  - matrix_block_size: 1024, Block size used in distributed arrays.
 *  - chol_block_size: "8", Block size in cholesky decomposition.
 *  - r_blk: "1", Number of iterations used to process real space grid in real space algorithm.
 *  - Vxc_block_size: "64". Number of pivots per real-space block when building
 *          Vxc. Memory scales as 8 * Vxc_block_size * nnr; the FFT count scales
 *          as nIpts^2 / Vxc_block_size, so raise it if memory allows.
 *  - distr_tol: "0.2". Controls the processor grid. Larger values lead to more processors in k/Q grid axis.
 *  - memory_frac: "0.75". fraction of available memory in a node used to estimate memory requirements/utilization. 
 */
thc::thc(mf::MF *mf_,
         utils::mpi_context_t<mpi3::communicator>& mpi_,
         ptree const& pt,
         bool print_metadata_
        ) :
  mpi(std::addressof(mpi_)),
  mf(mf_),
  Timer(),
  ecut( io::get_value_with_default<double>(pt,"ecut",
          mf->has_wfc_grid() ? 1.4*mf->wfc_truncated_grid()->ecut() : 0.4*mf->ecutrho()) ),
  rho_g( detail::make_grid(mpi->comm,ecut,*mf) ),
  swfc_to_rho(detail::make_wfc_to_rho(*mpi,(mf->has_wfc_grid()?*(mf->wfc_truncated_grid()):rho_g),rho_g)),
  vG( io::check_child_exists(pt,"potential") ? io::find_child(pt,"potential") : ptree{}),
  default_block_size( io::get_value_with_default<int>(pt,"matrix_block_size",1024) ), 
  default_cholesky_block_size( io::get_value_with_default<int>(pt,"chol_block_size",8) ),
  thresh( io::get_value_with_default<double>(pt,"thresh",1e-5) ),
  nnr_blk( io::get_value_with_default<int>(pt,"r_blk",1) ),
  distr_tol( io::get_value_with_default<double>(pt,"distr_tol",0.2) ),
  memory_frac( io::get_value_with_default<double>(pt,"memory_frac",0.75) ),
  use_least_squares( io::get_value_with_default<bool>(pt,"use_least_squares",false) ),
  nbnd_protected( io::get_value_with_default<long>(pt,"nbnd_protected",-1) ),
  // default on whenever protected bands are requested; either can be overridden
  exclude_unprotected_pairs( io::get_value_with_default<bool>(pt,"exclude_unprotected_pairs",
                                                              nbnd_protected > 0) ),
  vxc_file( io::get_value_with_default<std::string>(pt,"Vxc_file","") ),
  vxc_block_size( io::get_value_with_default<long>(pt,"Vxc_block_size",64) ),
  howmany_fft(-1)
{
  utils::check(mf != nullptr, "thc::Null pointer.");
  utils::check(mf->has_orbital_set(), "Error in thc: Invalid mf type. ");
  utils::check(default_block_size>0, "Error in thc: Invalid matrix_block_size:{}",default_block_size);
  utils::check(default_cholesky_block_size>0, "Error in thc: Invalid chol_block_size:{}",default_cholesky_block_size);
  utils::check(vxc_block_size>0, "Error in thc: Invalid Vxc_block_size:{}",vxc_block_size);

  memory_frac = std::min( 0.90, std::max( 0.25, memory_frac ) );

  // Per-band fit weights for the pivot search and the zeta fit. band_weights is the
  // master switch for both sources, which are combinable band range by band range:
  // the energy suppression of the band tail [nbnd_protected, nbnd_energy_end) and,
  // on an augmented mean field, the stored augmentation singular values on the
  // augmentation block [nbnd_orig, nbnd). Both feed the same band_weight table.
  bool protected_weights = false;
  bool augmented = mf->is_augmented();
  bool band_weights = io::get_value_with_default<bool>(pt,"band_weights",true);
  bool aug_weights = augmented and band_weights;
  long nbnd_orig = io::get_value_with_default<long>(pt,"nbnd_orig",-1);
  long nbnd_energy_end = mf->nbnd();
  if( nbnd_orig > 0 ) {
    utils::check(augmented,
                 "thc: nbnd_orig = {} is only meaningful for an augmented mean field.", nbnd_orig);
    utils::check(nbnd_protected > 0,
                 "thc: nbnd_orig = {} requires nbnd_protected > 0.", nbnd_orig);
    utils::check(nbnd_orig <= mf->nbnd(),
                 "thc: invalid nbnd_orig = {}. Requires nbnd_orig <= nbnd = {}.",
                 nbnd_orig, mf->nbnd());
  }
  if( nbnd_protected > 0 ) {
    long nb = mf->nbnd();
    utils::check(nbnd_protected <= nb,
                 "thc: invalid nbnd_protected = {}. Requires 0 < nbnd_protected <= nbnd = {}.",
                 nbnd_protected, nb);
    utils::check(mf->has_efermi(),
                 "thc: nbnd_protected requires a mean field carrying a Fermi energy; regenerate "
                 "the h5 with a pw2coqui that writes System/fermi_energy.");
    if( augmented ) {
      // The augmented basis is [originals | augmentation]; the energy weight is
      // defined on the originals only, and their count is not recoverable from
      // the mean field, so the caller must supply it.
      utils::check(nbnd_orig > 0,
                   "thc: nbnd_protected = {} on an augmented mean field requires nbnd_orig, the "
                   "number of original (non-augmentation) bands: the energy weights cover "
                   "[nbnd_protected, nbnd_orig) and the augmentation singular values cover "
                   "[nbnd_orig, nbnd = {}).", nbnd_protected, nb);
      utils::check(nbnd_protected <= nbnd_orig,
                   "thc: invalid nbnd_protected = {}. Requires nbnd_protected <= nbnd_orig = {}.",
                   nbnd_protected, nbnd_orig);
      nbnd_energy_end = nbnd_orig;
    }
  }
  if( nbnd_protected > 0 and nbnd_protected == mf->nbnd() ) {
    // Every band protected: the tail [N_P, nbnd) is empty, so all weights would
    // be one and there are no unprotected-unprotected pairs. Accepted as an
    // explicit no-op (identical to omitting nbnd_protected), but the weight
    // table and the pair exclusion are both skipped rather than built trivially:
    // the exclusion's metric subtraction would otherwise run over a zero-length
    // band range. The has_efermi precondition above still applies, so this stays
    // a faithful dry run of a real protected-band setup.
    utils::check(not (exclude_unprotected_pairs and
                      io::check_child_exists(pt,"exclude_unprotected_pairs")),
                 "thc: exclude_unprotected_pairs = true is meaningless with nbnd_protected = "
                 "nbnd = {}, since the unprotected band range [{}, {}) is empty.",
                 mf->nbnd(), mf->nbnd(), mf->nbnd());
    exclude_unprotected_pairs = false;
    app_log(1,"  nbnd_protected = nbnd = {}: all bands protected, so every weight is one and no "
              "pairs are dropped. This run is identical to plain THC.", mf->nbnd());
  } else if( nbnd_protected > 0 and not band_weights ) {
    // band_weights is the master switch: it turns off the energy suppression of the
    // protected-band tail as well as the augmentation singular values. nbnd_protected
    // still selects the unprotected tail for exclude_unprotected_pairs.
    app_log(1,"  band_weights = false: no fit weights are built, so nbnd_protected = {} only "
              "selects the unprotected band range.", nbnd_protected);
  } else if( nbnd_protected > 0 or aug_weights ) {
    long nb = mf->nbnd();
    long ns = mf->nspin(), nk = mf->nkpts();
    // Base table: the augmentation singular values (1 on the original bands),
    // or all ones when there is no augmentation to weight.
    if( aug_weights ) {
      band_weight = mf->augmented_band_weights();
    } else {
      band_weight = nda::array<double,3>(ns,nk,nb);
      band_weight() = 1.0;
    }
    utils::check(band_weight.extent(0)==ns and band_weight.extent(1)==nk and
                 band_weight.extent(2)==nb,
                 "thc: band weight table shape ({},{},{}) does not match (nspin,nkpts,nbnd) = "
                 "({},{},{}).", band_weight.extent(0), band_weight.extent(1),
                 band_weight.extent(2), ns, nk, nb);
    // w(s,k,b) = (E(s,k,N_P-1) - mu)/(E(s,k,b) - mu) on [N_P, nbnd_energy_end),
    // which is the full tail of an ordinary mean field and the original-band
    // tail of an augmented one. E(s,k,N_P-1) is the last protected band.
    if( nbnd_protected > 0 and nbnd_energy_end > nbnd_protected ) {
      double mu = mf->efermi();
      auto eig = mf->eigval();
      for( long is=0; is<ns; ++is )
        for( long k=0; k<nk; ++k ) {
          double eP = eig(is,k,nbnd_protected-1) - mu;
          utils::check(eP > 0.0,
                       "thc: nbnd_protected = {}: reference band E(s={},k={},b={}) - mu = {} is not "
                       "positive. The reference band is nbnd_protected-1, so choose nbnd_protected "
                       "strictly greater than the number of occupied bands (at least one "
                       "unoccupied band must be protected).",
                       nbnd_protected, is, k, nbnd_protected-1, eP);
          for( long b=nbnd_protected; b<nbnd_energy_end; ++b ) {
            double de = eig(is,k,b) - mu;
            utils::check(de > 0.0,
                         "thc: nbnd_protected = {}: E(s={},k={},b={}) - mu = {} is not positive. "
                         "Choose nbnd_protected strictly greater than the number of occupied bands.",
                         nbnd_protected, is, k, b, de);
            band_weight(is,k,b) = eP/de;
          }
        }
      protected_weights = true;
    }
  }
  if( band_weight.size() > 0 ) {
    auto const* w0 = band_weight.data();
    auto const* w1 = w0 + band_weight.size();
    has_band_weights = std::any_of(w0, w1, [](double w) { return std::abs(w-1.0) > 1e-14; });
    if(has_band_weights) {
      utils::check(mf->npol_in_basis() == 1 and mf->nspin_in_basis() == mf->nspin(),
                   "thc: band weights require npol==1 and nspin_in_basis==nspin.");
      double wmin = *std::min_element(w0, w1);
      utils::check(wmin > 0.0, "thc: non-positive band weight: {}.", wmin);
      if(protected_weights) {
        double wmax = *std::max_element(w0, w1);
        app_log(1,"  Protected-band weights applied to THC pivot search and fit: "
                  "nbnd_protected = {} (reference band index {}), mu = {} a.u., "
                  "energy-weighted band range [{}, {}), weights in [{:.3e}, {:.3e}].",
                nbnd_protected, nbnd_protected-1, mf->efermi(),
                nbnd_protected, nbnd_energy_end, wmin, wmax);
        if(aug_weights and nbnd_orig < mf->nbnd()) {
          // Report the range actually present on the augmentation block: an h5 written
          // before augmented_band_weights existed loads as all ones, which would make a
          // claim of "singular values" false.
          auto aug_blk = band_weight(nda::range::all, nda::range::all,
                                     nda::range(nbnd_orig, mf->nbnd()));
          double amin = nda::min_element(aug_blk), amax = nda::max_element(aug_blk);
          app_log(1,"  Augmentation band weights on [{}, {}): [{:.3e}, {:.3e}]{}",
                  nbnd_orig, mf->nbnd(), amin, amax,
                  (std::abs(amin-1.0) <= 1e-14 and std::abs(amax-1.0) <= 1e-14) ?
                    " (all ones: the mean field carries no augmentation singular values)" : "");
        }
      } else if(aug_weights) {
        app_log(1,"  Augmented-basis band weights applied to THC pivot search and fit "
                  "(min = {:.3e}). Disable with band_weights = false.", wmin);
      }
    } else {
      band_weight = nda::array<double,3>{};
    }
  }

  // Research diagnostic (exclude_unprotected_pairs): drop the unprotected-
  // unprotected pair densities. Defaults on whenever nbnd_protected > 0; the
  // unprotected set is the band tail [nbnd_protected, nbnd). Off by default
  // otherwise, in which case nothing below runs and the pivot/fit paths stay
  // byte-identical.
  if(exclude_unprotected_pairs) {
    long nb = mf->nbnd();
    utils::check(nbnd_protected > 0 and nbnd_protected < nb,
                 "thc: exclude_unprotected_pairs is on but the number of protected bands is "
                 "unknown (nbnd_protected = {}, need 0 < nbnd_protected < nbnd = {}).",
                 nbnd_protected, nb);
    unprot_band_start = nbnd_protected;
    bool by_default = not io::check_child_exists(pt,"exclude_unprotected_pairs");
    app_log(1,"  THC exclude_unprotected_pairs active ({}): dropping unprotected-unprotected "
              "pair densities; unprotected band range = [{}, {}).",
            by_default ? "default for nbnd_protected > 0" : "explicit", unprot_band_start, nb);
  }

  if (print_metadata_) print_metadata();

  for( auto& v: {"TOTAL","IO_SAVE","IO_ORBS","ALLOC","ip_COMM","COMM","FFT","FFTPLAN","DistOrbs","IpIter",
                 "IntPts","IntVecs","VCoul","LSSolve","ip_SERIAL","SERIAL","TUR","ZUR","EXTRA",
                 "ip_setup_comm","ip_chol","ip_update_res",
                 "GEMM", "shmX", "VXC",
                 // leaf sub-clocks inside the ZUR loop of get_ZquG_Cquv_fft_shared_memory
                 "ZUR_kR","ZUR_had","ZUR_RQ","ZUR_Cquv","ZUR_pack","ZUR_copy",
                 // leaf sub-clocks splitting the ALLOC total
                 "A_outzero","A_redist","A_darray","A_shmwin"} )
    Timer.add(v);
}

thc::~thc()
{
}

void thc::print_metadata()
{
  // MAM: need to print mf identifier, otherwise we don't know which mf this output corresponds to
  app_log(1,"  ERI::thc Computation Details");
  app_log(1,"  ----------------------------");
  app_log(1,"  Energy cutoff                = {} a.u. | FFT mesh = ({},{},{}), Number of PWs = {}",
          ecut, rho_g.mesh(0),rho_g.mesh(1),rho_g.mesh(2), rho_g.size());
  app_log(2,"  Default Slate block size     = {}",default_block_size);
  app_log(2,"  Default cholesky block size  = {}",default_cholesky_block_size);
  app_log(1,"  Threshold                    = {}",thresh);
  app_log(2,"  Distribution tolerance       = {}",distr_tol);
  app_log(2,"  Fraction of memory used for estimation = {}",memory_frac);
  utils::memory_report(2);
  app_log(1,"");
}

template<MEMORY_SPACE MEM>
auto thc::interpolating_points(int iq, int max, nda::range a_range, nda::range b_range)
      -> std::tuple<memory::array<MEM,long,1>,
                     _darray_t_<MEM,4>,
                     std::optional<_darray_t_<MEM,4>>
                    >
{
  Timer.start("TOTAL");
  app_log(2,"*******************************");
  app_log(2," ERI::thc::interpolating_points ");
  app_log(2,"*******************************");  
  app_log(2,"  -memory space: {}",memory_space_to_string(MEM));
  utils::memory_report(2, "thc::interpolating_points");
  set_range(a_range);
  set_range(b_range);
  utils::check( max <= rho_g.nnr(),
                "thc::interpolating_points: nmax > nnr - {}, {}",max,mf->nnr());

  // empty optionals 
  using Arr4D = memory::array<MEM,ComplexType,4>;
  Arr4D* C_skai = nullptr;

  auto Q  = mf->Qpts()(iq,nda::range::all);
  bool gamma = (Q(0)*Q(0)+Q(1)*Q(1)+Q(2)*Q(2) < 1e-8);
  if((a_range==b_range) and gamma and (mf->nkpts()!=mf->nkpts_ibz()) ) {
      auto return_v = chol_metric_impl_ibz<MEM,true,true>(iq,max,a_range,b_range,default_cholesky_block_size);
      detail::check_npts_vs_pw(std::get<0>(return_v).size(), rho_g.size(), ecut);
      Timer.stop("TOTAL");
      return return_v;
  } else {
      auto return_v = chol_metric_impl<MEM,true,true>(iq,max,a_range,b_range,default_cholesky_block_size, C_skai);
      detail::check_npts_vs_pw(std::get<0>(return_v).size(), rho_g.size(), ecut);
      Timer.stop("TOTAL");
      return return_v;
  }
}

template<MEMORY_SPACE MEM>
auto thc::interpolating_points(nda::MemoryArrayOfRank<4> auto const& C_skai, int iq, int max) 
      -> std::tuple<memory::array<MEM,long,1>,
                     _darray_t_<MEM,4>,
                     std::optional<_darray_t_<MEM,4>>
                    >
{
  Timer.start("TOTAL");
  app_log(2,"*******************************");
  app_log(2," ERI::thc::interpolating_points (rotated orbitals)");
  app_log(2,"*******************************");
  app_log(2,"  -memory space: {}",memory_space_to_string(MEM));
  utils::memory_report(2, "thc::interpolating_points");
  utils::check( mf->nkpts() == mf->nkpts_ibz(), 
                "thc::interpolating_points: Not yet implemented with symmetries when C_skai is provided."); 
  utils::check( max <= rho_g.nnr(),
                "thc::interpolating_points: nmax > nnr - {}, {}",max,mf->nnr());
  utils::check( C_skai.extent(0) == mf->nspin() and
                C_skai.extent(1) == mf->nkpts() and
                C_skai.extent(3) == mf->nbnd(), 
                "thc::interpolating_points: Shape mismatch of C_skai: - ns: {}, nk: {}, nb:{}",
                mf->nspin(), mf->nkpts(), mf->nbnd());

  nda::range a_range(C_skai.extent(2));
  nda::range b_range(mf->nbnd());
  auto return_v = chol_metric_impl<MEM,true,true>(iq,max,a_range,b_range,default_cholesky_block_size,std::addressof(C_skai));
  detail::check_npts_vs_pw(std::get<0>(return_v).size(), rho_g.size(), ecut);
  Timer.stop("TOTAL");
  return return_v;
}

template<MEMORY_SPACE MEM>
auto thc::evaluate(int iq, memory::array<MEM,long,1> const& ri,
                   memory::darray_t<memory::array<MEM,ComplexType,5>,mpi3::communicator> const& B,
                   nda::range a_range, nda::range b_range)
	-> memory::darray_t<memory::array<MEM,ComplexType,2>,mpi3::communicator>
{
  Timer.start("TOTAL");
  app_log(2,"*******************************");
  app_log(2," ERI::thc::evaluate (LS-THC)");
  app_log(2,"*******************************");
  app_log(2,"  -memory space: {}",memory_space_to_string(MEM));
  utils::memory_report(2, "thc::evaluate");
  set_range(a_range);
  set_range(b_range);
  // calculate interpolating points and V matrix
  auto return_v = intvec_impl<MEM,true>(iq,ri,a_range,b_range,B);
  Timer.stop("TOTAL");
  return return_v; 
}

template<MEMORY_SPACE MEM, typename Tensor_t>
auto thc::evaluate(memory::array<MEM,long,1> const& ri, 
                   Tensor_t const& Xa,
                   std::optional<Tensor_t> const& Xb,
                   bool return_Sinv_Ivec, 
                   nda::range a_range, nda::range b_range,
                   std::array<long, 3> pgrid3D)
        -> std::tuple<_darray_t_<MEM,3>, memory::array<MEM, ComplexType, 2>, 
                      memory::array<MEM, ComplexType, 2>, std::optional<_darray_t_<MEM,3>> >
{
  Timer.start("TOTAL");
  app_log(2,"*******************************");
  app_log(2," ERI::thc::evaluate (ISDF)");
  app_log(2,"*******************************");
  app_log(2,"  -memory space: {}",memory_space_to_string(MEM));
  utils::memory_report(2, "thc::evaluate");
  set_range(a_range);
  set_range(b_range);
  if(Xb.has_value()) {
    auto [return_v, Z_head_qu, Zbar_head_qu, Sinv_IVec] = intvec_impl<MEM,true>(ri,Xa,std::addressof(*Xb),return_Sinv_Ivec,a_range,b_range,pgrid3D);
    Timer.stop("TOTAL");
    return std::make_tuple(std::move(return_v), std::move(Z_head_qu), std::move(Zbar_head_qu), std::move(Sinv_IVec));
  } else { 
    std::decay_t<Tensor_t>* nullXb = nullptr;
    auto [return_v, Z_head_qu, Zbar_head_qu, Sinv_IVec] = intvec_impl<MEM,true>(ri,Xa,nullXb,return_Sinv_Ivec,a_range,b_range,pgrid3D);
    Timer.stop("TOTAL");
    return std::make_tuple(std::move(return_v), std::move(Z_head_qu), std::move(Zbar_head_qu), std::move(Sinv_IVec));
  } 
}

template<MEMORY_SPACE MEM, typename Tensor_t>
auto thc::evaluate(memory::array<MEM,long,1> const& ri,
                   nda::MemoryArrayOfRank<4> auto const& C_skai,
                   Tensor_t const& Xa,
                   Tensor_t const& Xb,
                   bool return_Sinv_Ivec,
                   std::array<long, 3> pgrid3D)
        -> std::tuple<_darray_t_<MEM,3>, memory::array<MEM, ComplexType, 2>,
                      memory::array<MEM, ComplexType, 2>, std::optional<_darray_t_<MEM,3>> >
{
  decltype(nda::range::all) all;
  Timer.start("TOTAL");
  app_log(2,"*******************************");
  app_log(2," ERI::thc::evaluate (ISDF)");
  app_log(2,"*******************************");
  app_log(2,"  -memory space: {}",memory_space_to_string(MEM));
  utils::memory_report(2, "thc::evaluate");
  long nspins = mf->nspin_in_basis();
  long nkpts = mf->nkpts();
  long nbnd = mf->nbnd();
  long nchol = Xa.global_shape()[3];
  utils::check( not has_band_weights,
                "Error in thc::evaluate: band weights not supported with rotated "
                "orbitals (C_skai). Set band_weights = false." );
  utils::check( not exclude_unprotected_pairs,
                "Error in thc::evaluate: exclude_unprotected_pairs not supported with rotated "
                "orbitals (C_skai). Set exclude_unprotected_pairs = false." );
  utils::check( C_skai.shape() == std::array<long,4>{nspins,nkpts,Xa.global_shape()[2],nbnd},
                "Error in thc::evaluate: Shape mismatch of C_skai." );
  utils::check( Xa.global_shape() == std::array<long,4>{nspins,nkpts,Xa.global_shape()[2],nchol}, 
                "Error in thc::evaluate: Shape mismatch of Xb." );
  utils::check( Xb.global_shape() == std::array<long,4>{nspins,nkpts,nbnd,nchol}, 
                "Error in thc::evaluate: Shape mismatch of Xb." );
  nda::range a_range(nbnd);
  // CtX = transpose(C)*Xa
  auto CtX = math::nda::make_distributed_array<memory::array<MEM, ComplexType, 4>>(mpi->comm,
                          Xa.grid(), {nspins,nkpts,nbnd,nchol}, {1,1,1,1}); 
  {
    mpi3::communicator k_intra_comm = mpi->comm.split(Xa.origin()[0]*nkpts+Xa.origin()[1],mpi->comm.rank());
    memory::array<MEM, ComplexType, 2> T(nbnd,nchol);
    // simple for now
    auto C_loc = C_skai(all,all,Xa.local_range(2),all);
    auto Xa_loc = Xa.local();
    auto T_loc = T(all,Xa.local_range(3));
    auto Xi_loc = CtX.local();
    for( auto [is,s] : itertools::enumerate(Xa.local_range(0)) ) {
      for( auto [ik,k] : itertools::enumerate(Xa.local_range(1)) ) {
        T() = ComplexType(0.0);
        nda::blas::gemm(ComplexType(1.0),nda::transpose(C_loc(s,k,all,all)),Xa_loc(is,ik,all,all),
                        ComplexType(0.0),T_loc);
        k_intra_comm.all_reduce_in_place_n(T.data(),T.size(),std::plus<>{});
        Xi_loc(is,ik,all,all) = T(CtX.local_range(2),CtX.local_range(3)); 
      }
    }
  }
  mpi->comm.barrier();
  auto [return_v, Z_head_qu, Zbar_head_qu, Sinv_IVec] = intvec_impl<MEM,true>(ri,CtX,std::addressof(Xb),return_Sinv_Ivec,a_range,a_range,pgrid3D);
  Timer.stop("TOTAL");
  return std::make_tuple(std::move(return_v), std::move(Z_head_qu), std::move(Zbar_head_qu), std::move(Sinv_IVec));
}

template<MEMORY_SPACE MEM, typename Tensor_t>
void thc::evaluate(h5::group& gh5, std::string format,
                  memory::array<MEM,long,1> const& ri, 
                  Tensor_t const& Xa,
                  std::optional<Tensor_t> const& Xb,
                  nda::range a_range, nda::range b_range,
                  std::array<long, 3> pgrid3D)
{
  Timer.start("TOTAL");
  set_range(a_range);
  set_range(b_range);
  auto [V, Z_head_qu, Zbar_head_qu, inv_intvec] = evaluate<MEM>(ri,Xa,Xb,false,a_range,b_range,pgrid3D);
  Timer.stop("TOTAL");
  save(gh5,format,ri,V, Z_head_qu, Zbar_head_qu);
}

template<MEMORY_SPACE MEM, typename Tensor_t>
void thc::evaluate(h5::group& gh5, std::string format,
                  memory::array<MEM,long,1> const& ri,
                  nda::MemoryArrayOfRank<4> auto const& C_skai,
                  Tensor_t const& Xa,
                  Tensor_t const& Xb,
                  std::array<long, 3> pgrid3D)
{
  Timer.start("TOTAL");
  auto [V, Z_head_qu, Zbar_head_qu, inv_intvec] = evaluate<MEM>(ri,C_skai,Xa,Xb,false,pgrid3D);
  Timer.stop("TOTAL");
  save(gh5,format,ri,V, Z_head_qu, Zbar_head_qu);
}

template<MEMORY_SPACE MEM, typename Tensor_t>
auto thc::evaluate_isdf_only(memory::array<MEM,long,1> const& ri,
                             Tensor_t const& Xa,
                             std::optional<Tensor_t> const& Xb,
                             nda::range a_range,
                             nda::range b_range,
                             std::array<long, 3> pgrid3D)
-> _darray_t_<MEM,3> {
  Timer.start("TOTAL");
  set_range(a_range);
  set_range(b_range);

  if(Xb.has_value()) {
    auto Z_qur = intvec_impl<MEM,false>(ri,Xa,std::addressof(*Xb),false,a_range,b_range,pgrid3D);
    Timer.stop("TOTAL");
    return Z_qur;
  } else {
    std::decay_t<Tensor_t>* nullXb = nullptr;
    auto Z_qur = intvec_impl<MEM,false>(ri,Xa,nullXb,false,a_range,b_range,pgrid3D);
    Timer.stop("TOTAL");
    return Z_qur;
  }
};

template<MEMORY_SPACE MEM>
void thc::save(h5::group& gh5, std::string format, memory::array<MEM,long,1> const& ri, 
	memory::darray_t<memory::array<MEM,ComplexType,3>,mpi3::communicator> const& V,
        memory::array<MEM,ComplexType,2> const& Z_head_qu, 
        memory::array<MEM,ComplexType,2> const& Zbar_head_qu)
{
  Timer.start("TOTAL");
  utils::memory_report(3, "thc::save");
  Timer.start("IO_SAVE");
  if(format == "default" or format == "bdft") {
    if(mpi->comm.root()) {
      auto ri_h = nda::to_host(ri);
      nda::h5_write(gh5, "interpolating_points", ri_h, false);
      nda::h5_write(gh5, "interpolating_vectors_G0", Z_head_qu, false);
      nda::h5_write(gh5, "dual_interpolating_vectors_G0", Zbar_head_qu, false);
    }
    // V [ q, u, v ]
    math::nda::h5_write(gh5, "coulomb_matrix", V);
    // Vxc [ q, u, v ]. Separate dataset, never merged into coulomb_matrix: it is
    // only valid in the direct (Hartree) channel. Vxc_source records which kernel
    // dump produced it, so a reader can reject a file built from a different one.
    if(has_Vxc()) {
      math::nda::h5_write(gh5, "Vxc", get_Vxc());
      h5::h5_write(gh5, "Vxc_source", vxc_file);
    }
  } else
    APP_ABORT("Error: Unknown format type: {}",format);
  Timer.stop("IO_SAVE");
  Timer.stop("TOTAL");
}

template<MEMORY_SPACE MEM>
void thc::save(h5::group& gh5, std::string format, memory::array<MEM,long,1> const& ri,
               memory::darray_t<memory::array<MEM,ComplexType,3>,mpi3::communicator> const& zeta_qur,
               bool write_zeta_on_fft_mesh)
{
  Timer.start("TOTAL");
  utils::memory_report(3, "thc::save");
  Timer.start("IO_SAVE");
  if(format == "default" or format == "bdft") {

    memory::array<MEM, ComplexType, 3> zeta_qur_local;
    if(mpi->comm.root()) {
      auto ri_h = nda::to_host(ri);
      nda::h5_write(gh5, "interpolating_points", ri_h, false);

      zeta_qur_local = memory::array<MEM, ComplexType, 3>(zeta_qur.global_shape());
      math::nda::gather(0, zeta_qur, &zeta_qur_local);

      if (mf->has_wfc_grid()) {

        nda::h5_write(gh5, "fft_mesh", rho_g.mesh(), false);

        if (write_zeta_on_fft_mesh) {
          memory::array<MEM, ComplexType, 3> zeta_qur_fft(
              zeta_qur_local.shape(0), zeta_qur_local.shape(1), rho_g.nnr());
          zeta_qur_fft() = 0.0;
          for (auto [i, n]: itertools::enumerate(rho_g.gv_to_fft())) {
            zeta_qur_fft(nda::range::all, nda::range::all, n) =
                zeta_qur_local(nda::range::all, nda::range::all, i);
          }
          nda::h5_write(gh5, "interpolating_vectors", zeta_qur_fft, false);
        } else {
          nda::h5_write(gh5, "interpolating_vectors", zeta_qur_local, false);
          nda::h5_write(gh5, "g_vectors", rho_g.g_vectors(), false);
          nda::h5_write(gh5, "gv_to_fft", rho_g.gv_to_fft(), false);
        }

      } else {
          nda::h5_write(gh5, "interpolating_vectors", zeta_qur_local, false);
      }

    } else {
      gather(0, zeta_qur, &zeta_qur_local);
    }
  } else
    APP_ABORT("Error: Unknown format type: {}",format);
  Timer.stop("IO_SAVE");
  Timer.stop("TOTAL");
}

void thc::print_timers()
{
  app_log(2,"\n");
  app_log(2,"  THC timers for the Cholesky algorithm");
  app_log(2,"  -------------------------------------");
  app_log(2,"  Total:                   {}",Timer.elapsed("TOTAL"));
  app_log(2,"    IO (save):             {}",Timer.elapsed("IO_SAVE"));
  app_log(2,"    IO (orbs):             {}",Timer.elapsed("IO_ORBS"));
  app_log(2,"    allocations:           {}",Timer.elapsed("ALLOC"));
  app_log(2,"      - out-array zero:    {}",Timer.elapsed("A_outzero"));
  app_log(2,"      - redistribute:      {}",Timer.elapsed("A_redist"));
  app_log(2,"      - darray alloc:      {}",Timer.elapsed("A_darray"));
  app_log(2,"      - shm windows:       {}",Timer.elapsed("A_shmwin"));
  app_log(2,"    communications:        {}",Timer.elapsed("COMM")+Timer.elapsed("ip_COMM"));
  app_log(2,"    fft:                   {}",Timer.elapsed("FFT"));
  app_log(2,"      - fft (planning):    {}",Timer.elapsed("FFTPLAN"));
  app_log(2,"    int. points:           {}",Timer.elapsed("IntPts"));
  app_log(2,"      -orbs IO+FFT:        {}",Timer.elapsed("DistOrbs"));
  app_log(2,"      -serial:             {}",Timer.elapsed("SERIAL")+Timer.elapsed("ip_SERIAL"));
  app_log(2,"      -iters:              {}",Timer.elapsed("IpIter"));
  app_log(2,"        -setup_comm:       {}",Timer.elapsed("ip_setup_comm"));
  app_log(2,"        -comm:             {}",Timer.elapsed("ip_COMM"));
  app_log(2,"        -chol:             {}",Timer.elapsed("ip_chol"));
  app_log(2,"        -residual:         {}",Timer.elapsed("ip_update_res"));
  app_log(2,"    int. vectors:          {}",Timer.elapsed("IntVecs"));
  app_log(2,"      - gemm:              {}",Timer.elapsed("GEMM"));
  app_log(2,"      - shmX:              {}",Timer.elapsed("shmX"));
  app_log(2,"      - Tur:               {}",Timer.elapsed("TUR"));
  app_log(2,"      - Zur:               {}",Timer.elapsed("ZUR"));
  app_log(2,"        -> k->R gemm:      {}",Timer.elapsed("ZUR_kR"));
  app_log(2,"        -> hadamard:       {}",Timer.elapsed("ZUR_had"));
  app_log(2,"        -> R->q gemm:      {}",Timer.elapsed("ZUR_RQ"));
  app_log(2,"        -> C(q,u,v) gather:{}",Timer.elapsed("ZUR_Cquv"));
  app_log(2,"        -> r->G pack:      {}",Timer.elapsed("ZUR_pack"));
  app_log(2,"        -> fft copy-back:  {}",Timer.elapsed("ZUR_copy"));
  app_log(2,"      - extra:             {}",Timer.elapsed("EXTRA"));
  app_log(2,"      - ls solve:          {}",Timer.elapsed("LSSolve"));
  app_log(2,"    coulomb matrix:        {}",Timer.elapsed("VCoul"));
  utils::memory_report(2);
  app_log(2,"\n");
}

void thc::set_range(nda::range& a_range) 
{
  if(a_range.first() < 0 and a_range.last() < 0) a_range = nda::range(mf->nbnd());
  if(a_range.first() < 0) a_range = nda::range(0,a_range.last());
  if(a_range.last() < 0) a_range = nda::range(a_range.first(),mf->nbnd());

  utils::check( a_range.last() > a_range.first() and a_range.last() <= mf->nbnd(),
                "thc::evaluate: Inconsistent a_range: ({},{})",
                a_range.first(),a_range.last());
}

void thc::set_k_range(nda::range& k_range)
{
  if(k_range.first() < 0 and k_range.last() < 0) k_range = nda::range(mf->nkpts());
  if(k_range.first() < 0) k_range = nda::range(0,k_range.last());
  if(k_range.last() < 0) k_range = nda::range(k_range.first(),mf->nkpts());

  utils::check( k_range.last() > k_range.first() and k_range.last() <= mf->nkpts(),
                "thc::evaluate: Inconsistent k_range: ({},{})",
                k_range.first(),k_range.last());
}

void thc::write_meta_data(h5::group& gh5, std::string format)
{
  Timer.start("TOTAL");
  Timer.start("IO_SAVE");
#ifndef HAVE_PHDF5
  if(not mpi->comm.root()) return;
#endif
  if(format == "default" or format == "bdft") {
    std::string fmt = "bdft";
    h5::h5_write(gh5, "format", fmt);
    h5::h5_write(gh5, "maximum_number_of_orbitals", mf->nbnd());
    h5::h5_write(gh5, "maximum_number_of_auxiliary_orbitals", 0);
    h5::h5_write(gh5, "number_of_kpoints", mf->nkpts());
    h5::h5_write(gh5, "number_of_kpoints_ibz", mf->nkpts_ibz());
    h5::h5_write(gh5, "number_of_qpoints", mf->nqpts());
    h5::h5_write(gh5, "number_of_qpoints_ibz", mf->nqpts_ibz());
    h5::h5_write(gh5, "number_of_spins", mf->nspin());
    h5::h5_write(gh5, "number_of_spins_in_basis", mf->nspin_in_basis());
    h5::h5_write(gh5, "volume", mf->volume());
    nda::h5_write(gh5, "kpoints", mf->kpts(), false);
    nda::h5_write(gh5, "qpoints", mf->Qpts(), false);
    nda::h5_write(gh5, "qk_to_k2", mf->qk_to_k2(), false);
    nda::h5_write(gh5, "qminus", mf->qminus(), false);
    nda::h5_write(gh5, "kp_to_ibz", mf->kp_to_ibz(), false);
    nda::h5_write(gh5, "qp_to_ibz", mf->qp_to_ibz(), false);
    //nda::h5_write(gh5, "kp_symm", mf->kp_symm(), false);
    //nda::h5_write(gh5, "qp_symm", mf->qp_symm(), false);
  } else
    APP_ABORT("Error: Unknown format type: {}",format);
  Timer.stop("IO_SAVE");
  Timer.stop("TOTAL");
}

} // methods

// definition of more complicated templates
#include "methods/ERI/thc.icc"


// instantiation of "public" templates
namespace methods 
{
using nda::range;
using memory::array;
using memory::array_view;
using memory::darray_t;
using memory::host_array;
using memory::host_array_view;
using mpi3::communicator;

// interpolating_points
#define __ipts__(M) \
template std::tuple<array<M,long,1>,  \
    darray_t<array<M,ComplexType,4>,communicator>,   \
    std::optional<darray_t<array<M,ComplexType,4>,communicator>>>  \
thc::interpolating_points<M>(int,int,range,range);  \
template std::tuple<array<M,long,1>,  \
    darray_t<array<M,ComplexType,4>,communicator>,   \
    std::optional<darray_t<array<M,ComplexType,4>,communicator>>>  \
thc::interpolating_points<M>(array<M,ComplexType,4> const&,int,int);  \
template std::tuple<array<M,long,1>,  \
    darray_t<array<M,ComplexType,4>,communicator>,   \
    std::optional<darray_t<array<M,ComplexType,4>,communicator>>>  \
thc::interpolating_points<M>(array_view<M,ComplexType,4> const&,int,int); \
template std::tuple<array<M,long,1>,  \
    darray_t<array<M,ComplexType,4>,communicator>,   \
    std::optional<darray_t<array<M,ComplexType,4>,communicator>>>  \
thc::interpolating_points<M>(array_view<M,ComplexType,4,nda::C_layout> const&,int,int);


// interpolating_basis
#define __ibasis__(M) \
template std::tuple<darray_t<array<M,ComplexType,4>,communicator>,  \
    std::optional<darray_t<array<M,ComplexType,4>,communicator>>>  \
thc::interpolating_basis<M>(array<M,long,1> const&,int,range,range);

// evaluate
#define __eval_ls__(M)  \
template darray_t<array<M,ComplexType,2>,communicator> \
thc::evaluate(int,array<M,long,1> const&,   \
    darray_t<memory::array<M,ComplexType,5>,communicator> const&,range,range);  

#define __eval__(M)  \
template std::tuple<darray_t<array<M,ComplexType,3>,communicator>,  \
    memory::array<M, ComplexType, 2>, memory::array<M, ComplexType, 2>,   \
    std::optional<darray_t<array<M,ComplexType,3>,communicator>> >  \
thc::evaluate<M>(array<M,long,1> const&,  \
    darray_t<array<M,ComplexType,4>,communicator> const&,  \
    std::optional<darray_t<array<M,ComplexType,4>,communicator>> const&,  \
    bool,range,range,std::array<long,3>);  \
template std::tuple<darray_t<array<M,ComplexType,3>,communicator>, \
    memory::array<M, ComplexType, 2>, memory::array<M, ComplexType, 2>, \
    std::optional<darray_t<array<M,ComplexType,3>,communicator>> > \
thc::evaluate<M>(array<M,long,1> const&,  \
    array<M,ComplexType,4> const&,  \
    darray_t<array<M,ComplexType,4>,communicator> const&,  \
    darray_t<array<M,ComplexType,4>,communicator> const&,  \
    bool,std::array<long,3>);  \
template std::tuple<darray_t<array<M,ComplexType,3>,communicator>,  \
    memory::array<M, ComplexType, 2>, memory::array<M, ComplexType, 2>,  \
    std::optional<darray_t<array<M,ComplexType,3>,communicator>> >  \
thc::evaluate<M>(array<M,long,1> const&,  \
    array_view<M,ComplexType,4> const&,  \
    darray_t<array<M,ComplexType,4>,communicator> const&,  \
    darray_t<array<M,ComplexType,4>,communicator> const&,  \
    bool,std::array<long,3>);  \
template std::tuple<darray_t<array<M,ComplexType,3>,communicator>,  \
    memory::array<M, ComplexType, 2>, memory::array<M, ComplexType, 2>,   \
    std::optional<darray_t<array<M,ComplexType,3>,communicator>> >  \
thc::evaluate<M>(array<M,long,1> const&,  \
    array_view<M,ComplexType,4,nda::C_layout> const&,  \
    darray_t<array<M,ComplexType,4>,communicator> const&,  \
    darray_t<array<M,ComplexType,4>,communicator> const&,  \
    bool,std::array<long,3>);  \
template void   \
thc::evaluate<M>(h5::group&,std::string,array<M,long,1> const&,  \
    darray_t<array<M,ComplexType,4>,communicator> const&,  \
    std::optional<darray_t<array<M,ComplexType,4>,communicator>> const&,  \
    range,range,std::array<long, 3>);  \
template void  \
thc::evaluate<M>(h5::group&,std::string,array<M,long,1> const&,  \
    array<M,ComplexType,4> const&,  \
    darray_t<array<M,ComplexType,4>,communicator> const&,  \
    darray_t<array<M,ComplexType,4>,communicator> const&,std::array<long, 3>);  \
template void  \
thc::evaluate<M>(h5::group&,std::string,array<M,long,1> const&,  \
    array_view<M,ComplexType,4> const&,  \
    darray_t<array<M,ComplexType,4>,communicator> const&,  \
    darray_t<array<M,ComplexType,4>,communicator> const&,std::array<long, 3>);  \
template void  \
thc::evaluate<M>(h5::group&,std::string,array<M,long,1> const&,  \
    array_view<M,ComplexType,4,nda::C_layout> const&,  \
    darray_t<array<M,ComplexType,4>,communicator> const&,  \
    darray_t<array<M,ComplexType,4>,communicator> const&,std::array<long, 3>);  \

__ipts__(HOST_MEMORY)
__ibasis__(HOST_MEMORY)
__eval_ls__(HOST_MEMORY)
__eval__(HOST_MEMORY)

template darray_t<host_array<ComplexType,3>,communicator>
thc::evaluate_isdf_only<HOST_MEMORY>(memory::host_array<long,1> const&,
    darray_t<host_array<ComplexType,4>,communicator> const&,
    std::optional<darray_t<host_array<ComplexType,4>,communicator>> const&,
    range,range,std::array<long,3>);

// save
template void thc::save<HOST_MEMORY>(h5::group&,std::string,memory::host_array<long,1> const&,
    darray_t<memory::host_array<ComplexType,3>,communicator> const&,
    memory::host_array<ComplexType, 2> const&, memory::host_array<ComplexType, 2> const&);

template void thc::save<HOST_MEMORY>(h5::group&,std::string,memory::host_array<long,1> const&,
    darray_t<memory::host_array<ComplexType,3>,communicator> const&, bool);

#if defined(ENABLE_DEVICE)

__ipts__(DEVICE_MEMORY)
__ipts__(UNIFIED_MEMORY)

__ibasis__(DEVICE_MEMORY)
__ibasis__(UNIFIED_MEMORY)

__eval__(DEVICE_MEMORY)
__eval__(UNIFIED_MEMORY)

template void thc::save<DEVICE_MEMORY>(h5::group&,std::string,memory::device_array<long,1> const&, 
    memory::darray_t<memory::device_array<ComplexType,3>,communicator> const&,
    memory::device_array<ComplexType,2> const&, memory::device_array<ComplexType, 2> const&);

template void thc::save<UNIFIED_MEMORY>(h5::group&,std::string,memory::unified_array<long,1> const&, 
    memory::darray_t<memory::unified_array<ComplexType,3>,communicator> const&,
    memory::unified_array<ComplexType,2 > const&, memory::unified_array<ComplexType, 2> const&);

#endif

} // methods

