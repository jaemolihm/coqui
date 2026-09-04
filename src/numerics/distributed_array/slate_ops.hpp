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


#ifndef NUMERICS_DISTRIBUTED_ARRAY_SLATE_OPS_HPP
#define NUMERICS_DISTRIBUTED_ARRAY_SLATE_OPS_HPP

/*
 * Utilities for use of SLATE with math::nda::distributed_matrix
 */ 

#include <functional>
#include <string>
#include <string_view>

#include "utilities/check.hpp"
#include "utilities/tile_partition.hpp"
#include "nda/nda.hpp"
#include "nda/tensor.hpp"
#include "numerics/distributed_array/ops.hpp"
#include "numerics/distributed_array/detail/ops_aux.hpp"
#include "numerics/distributed_array/detail/slate_aux.hpp"

namespace math::nda::slate_ops
{

/***************************************************************************/
/*  				Lapack	  				   */
/***************************************************************************/

/*
template<SlateMatrix A_t, SlateMatrix B_t, SlateMatrix C_t>
void svd(A_t&& A, B_t&& U, std::vector<double> S, C_t&& VH)
{
  using dA_t = typename std::decay_t<A_t>;
  using dB_t = typename std::decay_t<B_t>;
  using dC_t = typename std::decay_t<C_t>;
#if defined(ENABLE_SLATE)
  static_assert(std::is_same_v<typename dA_t::value_type, typename dB_t::value_type>,
                               "Value mismatch");
  static_assert(std::is_same_v<typename dA_t::value_type, typename dC_t::value_type>,
                               "Value mismatch");
  auto As = detail::to_slate_view<dA_t::is_stride_order_C()>(A);
  auto Us = detail::to_slate_view<dB_t::is_stride_order_C()>(U);
  auto VHs = detail::to_slate_view<dC_t::is_stride_order_C()>(VH);

  // need to conjugate matrix in c order
  //slate::svd(As,Us,S,VHs);
  utils::check(false, "svd not yet available in slate.");
#else
  utils::check(false, "svd: requires SLATE, compile with ENABLE_SLATE.");
#endif
}

template<SlateMatrix A_t, SlateMatrix B_t>
void eig(A_t&& A, std::vector<double> L, B_t&& X)
{
  using dA_t = typename std::decay_t<A_t>;
  using dB_t = typename std::decay_t<B_t>;
#if defined(ENABLE_SLATE)
  static_assert(std::is_same_v<typename dA_t::value_type, typename dB_t::value_type>,
                               "Value mismatch");
  auto As = detail::to_slate_view<dA_t::is_stride_order_C()>(A);
  auto Xs = detail::to_slate_view<dB_t::is_stride_order_C()>(X);

  // need to conjugate matrix in c order
  //slate::eig(As,L,Xs);
  utils::check(false, "eig not yet available in slate.");
#else
  utils::check(false, "eig: requires SLATE, compile with ENABLE_SLATE.");
#endif
}
*/

template<bool hermitian = false>
long lu_solve(DistributedMatrix auto&& A, DistributedMatrix auto&& B)
{
  using dA_t = typename std::decay_t<decltype(A)>;
  using dB_t = typename std::decay_t<decltype(B)>;
  using value_type = typename dA_t::value_type;
  long info=0;
  static_assert(std::is_same_v<typename dA_t::value_type, typename dB_t::value_type>,
                               "Value mismatch");
  utils::check(A.global_shape()[0] == A.global_shape()[1], "Shape mismatch in lu_solve.");
  utils::check( *(A.communicator()) == 
		*(B.communicator()),"Communicator mismatch");

  static_assert(::nda::mem::have_compatible_addr_space<typename dA_t::Array_t,
							 typename dB_t::Array_t
							>, "Memory location mismatch.");

// MAM: check if this works on device and enable!
//  if constexpr (::nda::mem::on_host<typename dA_t::Array_t,typename dB_t::Array_t>) {
  if(A.communicator()->size()==1) {

    auto Al = A.local();
    auto Bl = B.local();
    ::nda::basic_array<int, 1, ::nda::C_layout, 'A', 
        ::nda::heap<::nda::mem::get_addr_space<typename dA_t::Array_t>>> ipiv(Al.extent(0));
    info = ::nda::lapack::getrf(Al,ipiv);
    if( info != 0 ) {
      app_warning(" serial lu_solve: getrf info != 0 , info:{}",info); 
      return info;
    }

    if constexpr(dB_t::is_stride_order_Fortran()) {
      info = ::nda::lapack::getrs(Al,Bl,ipiv);
    } else {
      if constexpr ( ::nda::mem::get_addr_space<typename dA_t::Array_t> == ::nda::mem::Host ) {
        ::nda::basic_array<value_type, 2, ::nda::F_layout, 'A', 
            ::nda::heap<::nda::mem::get_addr_space<typename dB_t::Array_t>>> Bf(Bl); 
        info = ::nda::lapack::getrs(Al,Bf,ipiv);
        Bl = Bf;
      } else {
        ::nda::basic_array<value_type, 2, ::nda::F_layout, 'A', 
            ::nda::heap<::nda::mem::get_addr_space<typename dB_t::Array_t>>> Bf(Bl.shape()); 
        // since it is not clear if tensor backend will always accept mixed layouts, 
        // I'm creating a view to the transposed array
        using layout_t = typename ::nda::F_stride_layout::template mapping<2>;
        layout_t idx_{std::array<long,2>{Bl.extent(1),Bl.extent(0)},
            std::array<long,2>{Bl.strides()[1],Bl.strides()[0]}};
        ::nda::basic_array_view<value_type, 2, ::nda::F_stride_layout, 'A', 
            ::nda::default_accessor, 
            ::nda::borrowed<::nda::mem::get_addr_space<typename dB_t::Array_t>>> 
            Bl_f(idx_,Bl.data()); 
        ::nda::tensor::add(value_type(1.0),Bl_f,"ij",value_type(0.0),Bf,"ji");
        info = ::nda::lapack::getrs(Al,Bf,ipiv);
        ::nda::tensor::add(value_type(1.0),Bf,"ij",value_type(0.0),Bl_f,"ji");
      }
    }
    if( info != 0 )
      app_warning(" serial lu_solve: getri info != 0 , info:{}",info); 
    return info;
  }

  constexpr bool _dev_ = ::nda::mem::have_device_compatible_addr_space<
							 typename dA_t::Array_t,
                             typename dB_t::Array_t>;
#if defined(ENABLE_SLATE)

  auto slate_lu = [&](auto &a, auto &b) {
   if constexpr (_dev_) {
      return slate::lu_solve(a,b, {
        // Set execution target to GPU Devices
        { slate::Option::Target, slate::Target::Devices },
        { slate::Option::Lookahead, 1 }
                                    });
   }  else {
      return slate::lu_solve(a,b
#if defined(USE_SLATE_HOSTBATCH)
        ,{ { slate::Option::Target, slate::Target::HostBatch} }
#endif
        );
   } 
  };

//  static_assert(not ::nda::mem::on_device<typename dA_t::Array_t,
///					  typename dB_t::Array_t
//					 >, "lu_solve not working with device arrays!");
  if constexpr ((not hermitian) or dA_t::is_stride_order_Fortran()) {
    static_assert(dA_t::is_stride_order_Fortran(),"Stride order mismatch/hermitian mismatch.");
    auto As = detail::to_slate_view<dA_t::is_stride_order_C()>(A);
    if constexpr(dB_t::is_stride_order_C()) {
      auto Bs = detail::to_slate_view<dB_t::is_stride_order_C()>(transpose(B));
      info = slate_lu(As,Bs); 
    } else {
      auto Bs = detail::to_slate_view<dB_t::is_stride_order_C()>(B);
      info = slate_lu(As,Bs); 
    }
  } else { 
    ::nda::tensor::scale(value_type(1.0),A.local(),::nda::tensor::op::CONJ);
    auto As = detail::to_slate_view<dA_t::is_stride_order_C()>(A);
    if constexpr(dB_t::is_stride_order_C()) {
      auto Bs = detail::to_slate_view<dB_t::is_stride_order_C()>(transpose(B));
      info = slate_lu(As,Bs);
    } else {
      auto Bs = detail::to_slate_view<dB_t::is_stride_order_C()>(B);
      info = slate_lu(As,Bs);
    }
  }

#else
  utils::check(false, "lu_solve: requires SLATE, compile with ENABLE_SLATE.");
#endif
  return info;
}

template<bool hermitian = false>
long least_squares_solve(DistributedMatrix auto&& A, DistributedMatrix auto&& B)
{
  using dA_t = typename std::decay_t<decltype(A)>;
  using dB_t = typename std::decay_t<decltype(B)>;
  using value_type = typename dA_t::value_type;
  long info=0;
  static_assert(std::is_same_v<typename dA_t::value_type, typename dB_t::value_type>,
                               "Value mismatch");
  utils::check(A.global_shape()[0] == A.global_shape()[1], "Shape mismatch in lu_solve.");
  utils::check( *(A.communicator()) == 
		*(B.communicator()),"Communicator mismatch");

  static_assert(::nda::mem::have_compatible_addr_space<typename dA_t::Array_t,
							 typename dB_t::Array_t
							>, "Memory location mismatch.");

  constexpr bool _dev_ = ::nda::mem::have_device_compatible_addr_space<
                          typename dA_t::Array_t,typename dB_t::Array_t>;

  // no gels in nda cuda backend yet!
  if constexpr (not _dev_) {

    if(A.communicator()->size()==1) {

      int rank;
      auto Al = A.local();
      auto Bl = B.local();
      long dmin = std::min(Al.extent(0),Al.extent(1));
      ::nda::basic_array<double, 1, ::nda::F_layout, 'A', 
          ::nda::heap<::nda::mem::get_addr_space<typename dA_t::Array_t>>> S(dmin);

      if constexpr (dA_t::is_stride_order_Fortran()) {

        if constexpr(dB_t::is_stride_order_C()) {
          ::nda::basic_array<value_type, 2, ::nda::F_layout, 'A', 
              ::nda::heap<::nda::mem::get_addr_space<typename dB_t::Array_t>>> B_(Bl.extent(0),Bl.extent(1));
          B_() = Bl();
          long info_ = ::nda::lapack::gelss(Al,B_,S,-1.0,rank);
          Bl() = B_();
          return info_;
        } else {
          return ::nda::lapack::gelss(Al,Bl,S,-1.0,rank);
        }

      } else {

        if constexpr (hermitian) {

          ::nda::tensor::scale(value_type(1.0),Al,::nda::tensor::op::CONJ);
          if constexpr(dB_t::is_stride_order_C()) {
            ::nda::basic_array<value_type, 2, ::nda::F_layout, 'A',
                ::nda::heap<::nda::mem::get_addr_space<typename dB_t::Array_t>>> B_(Bl.extent(0),Bl.extent(1));
            B_() = Bl();
            long info_ = ::nda::lapack::gelss(::nda::transpose(Al),B_,S,-1.0,rank);
            Bl() = B_();
            return info_;
          } else {
            return ::nda::lapack::gelss(::nda::transpose(Al),Bl,S,-1.0,rank);
          }

        } else {

          ::nda::basic_array<value_type, 2, ::nda::F_layout, 'A',
              ::nda::heap<::nda::mem::get_addr_space<typename dA_t::Array_t>>> A_(Al.extent(0),Al.extent(1));
          A_() = Al();
          if constexpr(dB_t::is_stride_order_C()) {
            ::nda::basic_array<value_type, 2, ::nda::F_layout, 'A',
                ::nda::heap<::nda::mem::get_addr_space<typename dB_t::Array_t>>> B_(Bl.extent(0),Bl.extent(1));
            B_() = Bl();
            long info_ = ::nda::lapack::gelss(A_,B_,S,-1.0,rank);
            Bl() = B_();
            return info_;
          } else {
            return ::nda::lapack::gelss(A_,Bl,S,-1.0,rank);
          }

        }

      }

    } // (A.communicator()->size()==1)

  } // constexpr (not _dev_)

#if defined(ENABLE_SLATE)

  // slate's QR needs every tile at or below the diagonal to be at least as tall as
  // the diagonal one: geqrf -> tpqrt accumulates a panel's k x k triangular factor in
  // a tile i >= j of tile column j and hands lapack that tile's height as `lda` while
  // asking for k = the panel width. utils::tile_offset orders the ragged partition
  // with its larger tiles LAST, so it holds for every (N,t) by construction as long
  // as the two axes carry the SAME count. A is square here (checked above), but its
  // two counts need not agree: tiled {N,1} it has one-element row tiles against one
  // N-element column tile, and tpqrt gets lda = 1 against k = N. Hence the check.
  {
    // slate is handed A^T/A^H for a C-order array, so its row axis is CoQui's axis 1.
    constexpr int r_ax = dA_t::is_stride_order_C() ? 1 : 0, c_ax = 1 - r_ax;
    const long m = A.global_shape()[r_ax], tm = A.tile_count()[r_ax];
    const long n = A.global_shape()[c_ax], tn = A.tile_count()[c_ax];
    // Panel j needs mb(i) >= nb(j) for every tile row i >= j that can hold its
    // triangular factor. Row extents are non-decreasing, so the smallest such row
    // tile is mb(j) itself and one comparison per panel settles it. Note this is
    // NOT "shortest row tile >= widest column tile": with the two counts equal the
    // partitions are identical, so mb(i) = nb(i) >= nb(j) holds for i >= j even
    // though the axis mixes a and a+1 element tiles. Panels run to min(tm,tn).
    for (long j = 0; j < std::min(tm,tn); ++j) {
      const long mb = utils::tile_extent(m,tm,j), nb = utils::tile_extent(n,tn,j);
      utils::check(mb >= nb,
          "least_squares_solve: A's row tile {} is {} elements against a {}-element "
          "column tile, so slate's tpqrt gets lda < the panel width. The two axes "
          "carry {} and {} tiles; give the row axis at least as coarse a partition.",
          j, mb, nb, tm, tn);
    }
    utils::check(B.tile_count()[0] == tm,
        "least_squares_solve: B's row tile count ({}) must match A's ({}).",
        B.tile_count()[0], tm);
  }

  auto slate_ls = [&](auto &a, auto &b) {
   if constexpr (_dev_) {
      slate::least_squares_solve(a,b, {
        // Set execution target to GPU Devices
        { slate::Option::Target, slate::Target::Devices },
        { slate::Option::Lookahead, 1 }
                                    });
   }  else {
      slate::least_squares_solve(a,b
#if defined(USE_SLATE_HOSTBATCH)
        ,{ { slate::Option::Target, slate::Target::HostBatch} }
#endif
        );
   } 
  };

  if constexpr ((not hermitian) or dA_t::is_stride_order_Fortran()) {
    static_assert(dA_t::is_stride_order_Fortran(),"Stride order mismatch/hermitian mismatch.");
    auto As = detail::to_slate_view<dA_t::is_stride_order_C()>(A);
    if constexpr(dB_t::is_stride_order_C()) {
      auto Bs = detail::to_slate_view<dB_t::is_stride_order_C()>(transpose(B));
      slate_ls(As,Bs); 
    } else {
      auto Bs = detail::to_slate_view<dB_t::is_stride_order_C()>(B);
      slate_ls(As,Bs); 
    }
  } else { 
    ::nda::tensor::scale(value_type(1.0),A.local(),::nda::tensor::op::CONJ);
    auto As = detail::to_slate_view<dA_t::is_stride_order_C()>(A);
    if constexpr(dB_t::is_stride_order_C()) {
      auto Bs = detail::to_slate_view<dB_t::is_stride_order_C()>(transpose(B));
      slate_ls(As,Bs);
    } else {
      auto Bs = detail::to_slate_view<dB_t::is_stride_order_C()>(B);
      slate_ls(As,Bs);
    }
  }

#else
  utils::check(false, "lu_solve: requires SLATE, compile with ENABLE_SLATE.");
#endif
  return info;
}

void inverse(DistributedMatrix auto&& A)
{
  using dA_t = typename std::decay_t<decltype(A)>;
  using local_Array_t = typename dA_t::Array_t;
  using value_type = typename dA_t::value_type;
  static_assert(local_Array_t::layout_t::is_stride_order_C() or
                local_Array_t::layout_t::is_stride_order_Fortran(),
                "Layout mismatch" );

  // getri asserts A.mt() == A.nt() (slate/src/getri.cc), which needs the same
  // extent AND the same tile count on both axes -- the tile boundaries are a
  // function of (extent, tile count) alone. Nothing else checks it.
  utils::check(A.global_shape()[0] == A.global_shape()[1],
      "inverse: matrix is not square: ({}, {}).",
      A.global_shape()[0], A.global_shape()[1]);
  utils::check(A.tile_count()[0] == A.tile_count()[1],
      "inverse: row and column tile counts differ: ({}, {}).",
      A.tile_count()[0], A.tile_count()[1]);

  if (A.communicator()->size() == 1) {
    auto Aloc = A.local();
    ::nda::basic_array<int, 1, ::nda::C_layout, 'A',
                       ::nda::heap<::nda::mem::get_addr_space<local_Array_t>>> ipiv(Aloc.extent(0));
    long info = ::nda::lapack::getrf(Aloc, ipiv);
    utils::check(info == 0, "inverse: getrf info: {}.", info);
    info = ::nda::lapack::getri(Aloc, ipiv);
    utils::check(info == 0, "inverse: getri info: {}.", info);
    return;
  }

  //if ( __bypass__slate__lapack__ ) {
  if ( false ) {
    ::nda::basic_array<value_type, 2, ::nda::C_layout, 'A',
          ::nda::heap<::nda::mem::get_addr_space<local_Array_t>>> A_s; 
    if(A.communicator()->root()) {
      A_s = ::nda::array<ComplexType,2>(A.global_shape()[0],A.global_shape()[1]);
      gather(0,A,std::addressof(A_s));
      ::nda::basic_array<int, 1, ::nda::C_layout, 'A',
          ::nda::heap<::nda::mem::get_addr_space<local_Array_t>>> ipiv(A_s.extent(0));
      long info = ::nda::lapack::getrf(A_s, ipiv);
      utils::check(info == 0, "inverse: getrf info: {}.", info);
      info = ::nda::lapack::getri(A_s, ipiv);
      utils::check(info == 0, "inverse: getri info: {}.", info);
      scatter(0,std::addressof(A_s),A); 
    } else {
      gather(0,A,std::addressof(A_s));
      scatter(0,std::addressof(A_s),A); 
    }
    return;
  }

#if defined(ENABLE_SLATE)

  auto As = detail::to_slate_view<dA_t::is_stride_order_C()>(A);
  if constexpr (::nda::mem::on_host<local_Array_t>) {
    slate::Pivots pivots;
    long info = slate::getrf ( As , pivots
#if defined(USE_SLATE_HOSTBATCH)
	,{ { slate::Option::Target, slate::Target::HostBatch} }
#endif
	);
    // A singular A is reported here: slate::getri returns void, so the
    // factorization's info is the only status the inversion produces.
    utils::check(info == 0, "inverse: getrf info: {}.", info);
    slate::getri ( As , pivots
#if defined(USE_SLATE_HOSTBATCH)
	,{ { slate::Option::Target, slate::Target::HostBatch} }
#endif
	);
  } else {
    slate::Pivots pivots ;
    long info = slate::getrf ( As , pivots, 
        // Set execution target to GPU Devices
        {{ slate::Option::Target, slate::Target::Devices },
        { slate::Option::Lookahead, 1 }});
    utils::check(info == 0, "inverse: getrf info: {}.", info);
    info = slate::getri ( As , pivots,
        // Set execution target to GPU Devices
        {{ slate::Option::Target, slate::Target::Devices },
        { slate::Option::Lookahead, 1 }} );
    utils::check(info == 0, "inverse: getri info: {}.", info);
  }
#else
  utils::check(false, "inverse: requires SLATE, compile with ENABLE_SLATE.");
#endif
}

#if defined(ENABLE_SLATE)
namespace detail
{

/*
 * Parity of the row interchange getrf applied to global row `i_glob` of an axis of
 * extent `nrow` cut into `t_row` tiles: -1 if the row moved, +1 if it did not.
 *
 * slate::Pivots is indexed by the GLOBAL panel (tile row) index and is replicated
 * on every rank, while each Pivot is relative to its own panel -- slate documents
 * Pivot::tileIndex() as "tile index in the panel submatrix". So "row ia of panel
 * ib did not move" is (tileIndex, elementOffset) == (0, ia), the same test
 * slate::internal::permuteRows applies.
 *
 * That convention appears in no installed slate header -- it is read off
 * slate::internal::permuteRows -- so it is an assumption about slate's internals
 * rather than about its API. Verified against slate ded15290.
 *
 * It cannot be pinned by comparing this parity against LAPACK's ipiv: slate is
 * free to choose a different set of pivots than reference LAPACK, and does (at
 * one tile per element, N = 12, they disagree on 3 rows and the parities differ).
 * Both factorizations are valid, so diag(U) differs in sign between them and the
 * parity differs to compensate; only the product, the determinant, is invariant.
 * The guard is therefore TEST_CASE("determinant") in test_slate.cpp, which sweeps
 * grids and blockings against serial LAPACK's determinant -- it caught this
 * function's original miscount, and would catch a convention change the same way.
 */
inline int pivot_parity(slate::Pivots const& pivots, long i_glob, long nrow, long t_row)
{
  long ib = utils::tile_of(nrow,t_row,i_glob);            // global panel index
  long ia = i_glob - utils::tile_offset(nrow,t_row,ib);   // row offset within the panel
  utils::check(ib < long(pivots.size()) and ia < long(pivots[ib].size()),
      "pivot_parity: pivot index out of range: ({},{}) for {} panels.",
      ib, ia, pivots.size());
  return (pivots[ib][ia].tileIndex() != 0 or pivots[ib][ia].elementOffset() != ia)? -1 : 1;
}

} // namespace detail
#endif

/*
 * det(A) of a square, square-tiled distributed matrix, from slate::getrf: the
 * product of the diagonal of U times the parity of the row permutation.
 *
 * The diagonal is derived from the array's own origin and local shape, so no
 * caller has to know which local entries lie on it.
 */
auto determinant(DistributedMatrix auto&&A) {
  using dA_t = typename std::decay_t<decltype(A)>;
  using local_Array_t = typename dA_t::Array_t;
  using value_type = typename dA_t::value_type;
  value_type det_A = value_type(1.0);

  // getrf needs mt == nt (slate/src/getri.cc), and the pivot bookkeeping below
  // needs the row and column tile boundaries to coincide.  Neither slate nor
  // is_slate_compatible checks it.
  utils::check(A.global_shape()[0] == A.global_shape()[1],
      "determinant: matrix is not square: ({}, {}).",
      A.global_shape()[0], A.global_shape()[1]);
  utils::check(A.tile_count()[0] == A.tile_count()[1],
      "determinant: row and column tile counts differ: ({}, {}).",
      A.tile_count()[0], A.tile_count()[1]);

  if constexpr (::nda::mem::on_host<local_Array_t>) {
    if (A.communicator()->size() == 1) {
      auto A_loc = A.local();
      ::nda::matrix_view <value_type> Am(A_loc);
      det_A = ::nda::determinant_in_place(Am);
      return det_A;
    }
  }

#if defined(ENABLE_SLATE)
  if constexpr (::nda::mem::on_host<local_Array_t>) {
    auto As = detail::to_slate_view<dA_t::is_stride_order_C()>(A);
    slate::Pivots pivots ;
    slate::getrf ( As , pivots );

    auto A_loc = A.local();
    // Walk the local part of the diagonal.  Because the row and column partitions
    // agree (checked above), every global row is on exactly one rank's diagonal,
    // so the product all-reduce below accumulates one factor of -1 per row
    // interchange over the whole matrix -- each panel's parity counted once.
    const long nrow = A.global_shape()[0];
    const long t_row = A.tile_count()[0];
    auto [i_origin, j_origin] = A.origin();
    auto [ni_loc, nj_loc] = A.local_shape();
    long ndiag = 0;
    for (long ii = 0; ii < ni_loc; ++ii) {
      long jj = ii + i_origin - j_origin;
      if (jj < 0 or jj >= nj_loc) continue;
      det_A *= A_loc(ii,jj);
      if (detail::pivot_parity(pivots,ii+i_origin,nrow,t_row) < 0) det_A *= -1;
      ++ndiag;
    }
    // The local diagonals must tile the global one, or a row's parity is counted
    // twice or not at all.
    A.communicator()->all_reduce_in_place_n(&ndiag, 1, std::plus<>{});
    utils::check(ndiag == A.global_shape()[0],
        "determinant: the local diagonals cover {} of {} rows.",
        ndiag, A.global_shape()[0]);
    A.communicator()->all_reduce_in_place_n(&det_A, 1, std::multiplies<>{});
  } else {
    utils::check(false, "determinant: requires GPU supports.");
  }
#else
  utils::check(false, "determinant: requires SLATE, compile with ENABLE_SLATE.");
#endif
  return det_A;
}

/*
void cholesky(DistributedMatrix auto&& A, char UPLO = "L")
{
  using dA_t = typename std::decay_t<decltype(A)>;
  using local_Array_t = typename dA_t::Array_t;
  using value_type = typename dA_t::value_type;
  static_assert(local_Array_t::layout_t::is_stride_order_C() or
                local_Array_t::layout_t::is_stride_order_Fortran(),
                "Layout mismatch" );
  using layout_t = std::conditional_t<
                    local_Array_t::layout_t::is_stride_order_C(),
                    ::nda::C_layout,
                    ::nda::F_layout>; 

  if (A.communicator()->size() == 1) {
    auto Aloc = A.local();
    ::nda::basic_array<int, 1, ::nda::C_layout, 'A',
                       ::nda::heap<::nda::mem::get_addr_space<local_Array_t>>> ipiv(Aloc.extent(0));
    long info = ::nda::lapack::potrf(Aloc);
    utils::check(info == 0, "cholesky: potrf info: {}.", info);
    return;
  }

#if defined(ENABLE_SLATE)

  auto As = detail::to_slate_view<dA_t::is_stride_order_C()>(A);

  // redistribute into HermitianMatrix, call slate
  if constexpr (::nda::mem::on_host<local_Array_t>) {
    long info = slate::potrf ( As ,
//#if defined(USE_SLATE_HOSTBATCH)
	,{ { slate::Option::Target, slate::Target::HostBatch} }
#endif
	);
    utils::check(info == 0, "cholesky: potrf info: {}.", info);
  } else {
    long info = slate::potrf ( As, 
        // Set execution target to GPU Devices
        {{ slate::Option::Target, slate::Target::Devices },
        { slate::Option::Lookahead, 1 }});
    utils::check(info == 0, "cholesky: potrf info: {}.", info);
  }
#else
  utils::check(false, "cholesky: requires SLATE, compile with ENABLE_SLATE.");
#endif
}
*/

namespace detail
{

/// One gemm operand as slate sees it: global extents and tile counts, after any
/// transpose/conjugate op and after the row/column swap of a C-order array.
struct gemm_operand { long m, n, mt, nt; };

/**
 * The conformability `X*Y = Z` needs and `slate::gemm` never checks: the three shared
 * axes must agree in EXTENT and in TILE COUNT. Returns an empty string when the three
 * operands conform, otherwise the reason, naming them `xname`, `yname`, `zname`.
 *
 * slate's `src/gemm.cc` contains no asserts at all, and a wrong tile count is silently
 * wrong: `gemmC.cc` loops `for(k = 1; k < A.nt(); ++k)` and indexes
 * `B.sub(k, k, 0, B.nt()-1)`, so `Y.mt > X.nt` drops the tail of the contraction and
 * returns a plausible wrong number, while `Y.mt < X.nt` indexes a tile that does not
 * exist. Equal counts with mismatched BOUNDARIES is caught, but only down in
 * `Tile_blas.hh` (`A.nb() == B.mb()`), thrown from inside slate by whichever ranks
 * happen to own the offending tile pair, in the middle of a collective.
 *
 * Kept a predicate rather than inlined checks so it can be tested directly:
 * utils::check aborts the process, so the failing configurations have no other way
 * into the test suite. See TEST_CASE("gemm_tile_conformability").
 */
inline std::string gemm_tile_mismatch(gemm_operand X, gemm_operand Y, gemm_operand Z,
                                      std::string_view xname = "A",
                                      std::string_view yname = "B",
                                      std::string_view zname = "C")
{
  auto show = [](std::string_view name, gemm_operand const& O) {
    return std::string(name) + " is " + std::to_string(O.m) + "x" + std::to_string(O.n)
         + " (" + std::to_string(O.mt) + "x" + std::to_string(O.nt) + " tiles)";
  };
  if (X.n != Y.m or X.nt != Y.mt)
    return "contracted axis mismatch: " + show(xname,X) + ", " + show(yname,Y) + ".";
  if (X.m != Z.m or X.mt != Z.mt)
    return "row axis mismatch: " + show(xname,X) + ", " + show(zname,Z) + ".";
  if (Y.n != Z.n or Y.nt != Z.nt)
    return "column axis mismatch: " + show(yname,Y) + ", " + show(zname,Z) + ".";
  return {};
}

} // namespace detail

/***************************************************************************/
/*  				Blas	  				   */
/***************************************************************************/

namespace detail
{

template<typename T, typename A_t, typename B_t, DistributedMatrix C_t>
auto multiply_impl(T a, A_t&& A, B_t&& B, T b, C_t&& C)
{
  using dA_t = std::decay_t<A_t>;
  using dB_t = std::decay_t<B_t>;
  using dC_t = std::decay_t<C_t>;
  constexpr int Arank = ::nda::get_rank<typename dA_t::Array_t>;
  constexpr int Brank = ::nda::get_rank<typename dB_t::Array_t>;
  static_assert(Arank==2 and Brank==2,"Rank mismatch");
  static_assert(std::is_same_v<std::decay_t<typename dA_t::value_type>, std::decay_t<typename dB_t::value_type>>,
			       "Value mismatch");
  static_assert(std::is_same_v<std::decay_t<typename dA_t::value_type>, std::decay_t<typename dC_t::value_type>>,
			       "Value mismatch");
  static_assert(::nda::mem::have_compatible_addr_space<typename dA_t::Array_t,
                                                         typename dB_t::Array_t,
                                                         typename dC_t::Array_t
                                                        >, "Memory location mismatch.");
  constexpr bool _dev_ = ::nda::mem::have_device_compatible_addr_space<
                                                         typename dA_t::Array_t,
                                                         typename dB_t::Array_t,
                                                         typename dC_t::Array_t
                                                        >;
  utils::check( *((math::detail::arg(A).communicator())) == 
		*((math::detail::arg(B).communicator())),"Communicator mismatch");
  utils::check( *((math::detail::arg(A).communicator())) == 
		*(C.communicator()),"Communicator mismatch");

  if(C.communicator()->size()==1) {
    ::nda::blas::gemm(a, math::detail::local_with_tags(A), math::detail::local_with_tags(B), b, C.local());
    return std::forward<C_t>(C);
  }

#if defined(ENABLE_SLATE)
  // C is fixed
  auto As = detail::to_slate_view<dA_t::is_stride_order_C()>(A);
  auto Bs = detail::to_slate_view<dB_t::is_stride_order_C()>(B);
  auto Cs = detail::to_slate_view<dC_t::is_stride_order_C()>(C);

  // slate::gemm asserts none of its own conformability -- see gemm_tile_mismatch.
  // mt()/nt()/m()/n() already account for the transpose/conjugate-transpose op, so
  // these are the operands slate will receive. But to_slate_view hands slate the
  // TRANSPOSE of a C-order array, so for that layout every field is swapped relative
  // to what the caller declared: reporting them as they come would print C's shape
  // reversed and call a column axis "row". Swap them back, uniformly -- all three
  // operands share a stride order (static_assert below) -- and the message names the
  // axis and the shape the caller wrote.
  //
  // One call covers both layouts because the three conditions are frame-invariant:
  // transposing all of X, Y, Z maps the set {X.n==Y.m, X.m==Z.m, Y.n==Z.n} onto
  // itself, permuting the first with nothing and the last two with each other.
  constexpr bool row_col_swapped = dA_t::is_stride_order_C();
  auto as_operand = [](auto const& S) {
    if constexpr (row_col_swapped)
      return detail::gemm_operand{long(S.n()), long(S.m()), long(S.nt()), long(S.mt())};
    else
      return detail::gemm_operand{long(S.m()), long(S.n()), long(S.mt()), long(S.nt())};
  };
  {
    auto why = detail::gemm_tile_mismatch(as_operand(As),as_operand(Bs),as_operand(Cs),
                                          "A","B","C");
    utils::check(why.empty(), "multiply: {}", why);
  }

  if constexpr (dA_t::is_stride_order_Fortran()) {
    static_assert(dB_t::is_stride_order_Fortran(),"Stride order mismatch.");
    static_assert(dC_t::is_stride_order_Fortran(),"Stride order mismatch.");
    if constexpr (_dev_) {
      slate::multiply(a,As,Bs,b,Cs, {
	// Set execution target to GPU Devices
        { slate::Option::Target, slate::Target::Devices }, 
	{ slate::Option::Lookahead, 1 }
				    });
    } else {
      slate::multiply(a,As,Bs,b,Cs
#if defined(USE_SLATE_HOSTBATCH)
	,{ { slate::Option::Target, slate::Target::HostBatch} }
#endif	
      );
    }
  } else {
    static_assert(dB_t::is_stride_order_C(),"Stride order mismatch.");
    static_assert(dC_t::is_stride_order_C(),"Stride order mismatch.");
    if constexpr (_dev_) {
      slate::multiply(a,Bs,As,b,Cs, {
          // Set execution target to GPU Devices
          { slate::Option::Target, slate::Target::Devices },
	  { slate::Option::Lookahead, 1 }
				    });
    } else {
      slate::multiply(a,Bs,As,b,Cs
#if defined(USE_SLATE_HOSTBATCH)
	  ,{ { slate::Option::Target, slate::Target::HostBatch} }
#endif	
	);
    }
  }

#else
  utils::check(false, "requires SLATE, compile with ENABLE_SLATE.");
#endif
  return std::forward<C_t>(C);
}

}

template<typename T, typename A_t, typename B_t, DistributedArray C_t>
auto multiply(T a_v, A_t&& A, B_t&& B, T b_v, C_t&& C)
{
  decltype(::nda::range::all) all;
  using dA_t = std::decay_t<A_t>;
  using dB_t = std::decay_t<B_t>;
  using dC_t = std::decay_t<C_t>;
  constexpr int Arank = ::nda::get_rank<typename dA_t::Array_t>;
  constexpr int Brank = ::nda::get_rank<typename dB_t::Array_t>;
  constexpr int Crank = ::nda::get_rank<typename dC_t::Array_t>;
  static_assert(Arank==Brank and Brank==Crank,"Rank mismatch");

  if constexpr (Arank==2) {
    return detail::multiply_impl(a_v,std::forward<A_t>(A),std::forward<B_t>(B),
				 b_v,std::forward<C_t>(C));
  } else {

    auto& comm = *C.communicator();
    auto&& dA = math::detail::arg(A);    
    auto&& dB = math::detail::arg(B);    

    constexpr bool Atr = math::detail::is_transpose<dA_t>;
    constexpr bool Btr = math::detail::is_transpose<dB_t>;
    constexpr bool Acg = math::detail::is_conjugate_transpose<dA_t>;
    constexpr bool Bcg = math::detail::is_conjugate_transpose<dB_t>;

    // consistency checks
    utils::check(*dA.communicator()==*dB.communicator(),"Communicator mismatch"); 
    utils::check(*dA.communicator()==*C.communicator(),"Communicator mismatch"); 

    if constexpr (dA_t::is_stride_order_C()) {
      static_assert(dB_t::is_stride_order_C() and dC_t::is_stride_order_C(), "Stride mismatch");

      long color=0, px=1;
      for(int r=0; r<Arank-2; ++r) { 
        utils::check(dA.global_shape()[r]==dB.global_shape()[r] and
                     dA.global_shape()[r]==C.global_shape()[r],"Global shape mismatch"); 
        utils::check(dA.local_shape()[r]==dB.local_shape()[r] and
                     dA.local_shape()[r]==C.local_shape()[r],"Local shape mismatch"); 
        utils::check(dA.origin()[r]==dB.origin()[r] and
                     dA.origin()[r]==C.origin()[r],"Origin mismatch"); 
        // these two should ensure consistency across tasks if created with make_distributed
        // otherwise it will need communication to check
        utils::check(dA.grid()[r]==dB.grid()[r] and
                     dA.grid()[r]==C.grid()[r],"Grid mismatch"); 
        utils::check(dA.tile_count()[r]==dB.tile_count()[r] and
                     dA.tile_count()[r]==C.tile_count()[r],"Grid mismatch"); 
        color += px*C.origin()[r]; 
        px *= C.global_shape()[r];
      }

      auto get_arr = [](auto const& a) 
	{ return std::array<long,2>{*(a.rbegin()+1),*(a.rbegin())}; };

      auto new_comm = comm.split(color,comm.rank());
      // doing by hand for now
      if constexpr (Arank==3)  {
        for( auto [ia,a] : itertools::enumerate(dA.local_range(0)) ) {
          auto A2d = dA.local()(ia,all,all);
          auto B2d = dB.local()(ia,all,all);
          auto C2d = C.local()(ia,all,all);
          auto A_ = distributed_array_view<decltype(A2d),decltype(new_comm)>(
                std::addressof(new_comm),get_arr(dA.grid()),get_arr(dA.global_shape()),
                get_arr(dA.origin()),get_arr(dA.tile_count()),A2d);
          auto B_ = distributed_array_view<decltype(B2d),decltype(new_comm)>(
                std::addressof(new_comm),get_arr(dB.grid()),get_arr(dB.global_shape()),
                get_arr(dB.origin()),get_arr(dB.tile_count()),B2d);
          auto C_ = distributed_array_view<decltype(C2d),decltype(new_comm)>(
                std::addressof(new_comm),get_arr(C.grid()),get_arr(C.global_shape()),
                get_arr(C.origin()),get_arr(C.tile_count()),C2d);
          detail::multiply_impl(a_v,math::detail::add_tags<Atr,Acg>(A_),math::detail::add_tags<Btr,Bcg>(B_),
                             b_v,C_);
        }
      } else if constexpr (Arank==4) {
        for( auto [ia,a] : itertools::enumerate(dA.local_range(0)) ) 
          for( auto [ib,b] : itertools::enumerate(dA.local_range(1)) ) {
            auto A2d = dA.local()(ia,ib,all,all);
            auto B2d = dB.local()(ia,ib,all,all);
            auto C2d = C.local()(ia,ib,all,all);
            auto A_ = distributed_array_view<decltype(A2d),decltype(new_comm)>(
                  std::addressof(new_comm),get_arr(dA.grid()),get_arr(dA.global_shape()),
                  get_arr(dA.origin()),get_arr(dA.tile_count()),A2d);
            auto B_ = distributed_array_view<decltype(B2d),decltype(new_comm)>(
                  std::addressof(new_comm),get_arr(dB.grid()),get_arr(dB.global_shape()),
                  get_arr(dB.origin()),get_arr(dB.tile_count()),B2d);
            auto C_ = distributed_array_view<decltype(C2d),decltype(new_comm)>(
                  std::addressof(new_comm),get_arr(C.grid()),get_arr(C.global_shape()),
                  get_arr(C.origin()),get_arr(C.tile_count()),C2d);
            detail::multiply_impl(a_v,math::detail::add_tags<Atr,Acg>(A_),math::detail::add_tags<Btr,Bcg>(B_),
                                 b_v,C_);
          }
      } else if constexpr (Arank==5) {
        for( auto [ia,a] : itertools::enumerate(dA.local_range(0)) )
          for( auto [ib,b] : itertools::enumerate(dA.local_range(1)) ) 
            for( auto [ic,c] : itertools::enumerate(dA.local_range(2)) ) {
              auto A2d = dA.local()(ia,ib,ic,all,all);
              auto B2d = dB.local()(ia,ib,ic,all,all);
              auto C2d = C.local()(ia,ib,ic,all,all);
              auto A_ = distributed_array_view<decltype(A2d),decltype(new_comm)>(
                    std::addressof(new_comm),get_arr(dA.grid()),get_arr(dA.global_shape()),
                    get_arr(dA.origin()),get_arr(dA.tile_count()),A2d);
              auto B_ = distributed_array_view<decltype(B2d),decltype(new_comm)>(
                    std::addressof(new_comm),get_arr(dB.grid()),get_arr(dB.global_shape()),
                    get_arr(dB.origin()),get_arr(dB.tile_count()),B2d);
              auto C_ = distributed_array_view<decltype(C2d),decltype(new_comm)>(
                    std::addressof(new_comm),get_arr(C.grid()),get_arr(C.global_shape()),
                    get_arr(C.origin()),get_arr(C.tile_count()),C2d);
              detail::multiply_impl(a_v, math::detail::add_tags<Atr,Acg>(A_),
                                 math::detail::add_tags<Btr,Bcg>(B_),
                                 b_v,C_);
            }
      } else {
        static_assert(Arank==2,"Finish implementation!");
      }

    } else {
      static_assert(dA_t::is_stride_order_Fortran() and
		    dB_t::is_stride_order_Fortran() and 
		    dC_t::is_stride_order_Fortran(), "Stride mismatch");

      long color=0, px=1;
      for(int r=Arank-1; r>=2; --r) {
        utils::check(dA.global_shape()[r]==dB.global_shape()[r] and
                     dA.global_shape()[r]==C.global_shape()[r],"Global shape mismatch");
        utils::check(dA.local_shape()[r]==dB.local_shape()[r] and
                     dA.local_shape()[r]==C.local_shape()[r],"Local shape mismatch");
        utils::check(dA.origin()[r]==dB.origin()[r] and
                     dA.origin()[r]==C.origin()[r],"Origin mismatch");         
        // these two should ensure consistency across tasks if created with make_distributed
        // otherwise it will need communication to check
        utils::check(dA.grid()[r]==dB.grid()[r] and
                     dA.grid()[r]==C.grid()[r],"Grid mismatch");               
        utils::check(dA.tile_count()[r]==dB.tile_count()[r] and
                     dA.tile_count()[r]==C.tile_count()[r],"Grid mismatch");
        color += px*C.origin()[r]; 
        px *= C.global_shape()[r];
      }

      auto get_arr = [](auto const& a)
        { return std::array<long,2>{*a.begin(),*(a.begin()+1)}; };

      auto new_comm = comm.split(color,comm.rank());
      // doing by hand for now
      if constexpr (Arank==3)  {
        for( auto [ia,a] : itertools::enumerate(dA.local_range(2)) ) {
          auto A2d = dA.local()(all,all,ia);
          auto B2d = dB.local()(all,all,ia);
          auto C2d = C.local()(all,all,ia);
          auto A_ = distributed_array_view<decltype(A2d),decltype(new_comm)>(
                std::addressof(new_comm),get_arr(dA.grid()),get_arr(dA.global_shape()),
                get_arr(dA.origin()),get_arr(dA.tile_count()),A2d);
          auto B_ = distributed_array_view<decltype(B2d),decltype(new_comm)>(
                std::addressof(new_comm),get_arr(dB.grid()),get_arr(dB.global_shape()),
                get_arr(dB.origin()),get_arr(dB.tile_count()),B2d);
          auto C_ = distributed_array_view<decltype(C2d),decltype(new_comm)>(
                std::addressof(new_comm),get_arr(C.grid()),get_arr(C.global_shape()),
                get_arr(C.origin()),get_arr(C.tile_count()),C2d);
          detail::multiply_impl(a_v,math::detail::add_tags<Atr,Acg>(A_),math::detail::add_tags<Btr,Bcg>(B_),
	  			                b_v,C_);
        }
      } else if constexpr (Arank==4) {
        for( auto [ia,a] : itertools::enumerate(dA.local_range(2)) )
          for( auto [ib,b] : itertools::enumerate(dA.local_range(3)) ) {
            auto A2d = dA.local()(all,all,ia,ib);
            auto B2d = dB.local()(all,all,ia,ib);
            auto C2d = C.local()(all,all,ia,ib);
            auto A_ = distributed_array_view<decltype(A2d),decltype(new_comm)>(
                  std::addressof(new_comm),get_arr(dA.grid()),get_arr(dA.global_shape()),
                  get_arr(dA.origin()),get_arr(dA.tile_count()),A2d);
            auto B_ = distributed_array_view<decltype(B2d),decltype(new_comm)>(
                  std::addressof(new_comm),get_arr(dB.grid()),get_arr(dB.global_shape()),
                  get_arr(dB.origin()),get_arr(dB.tile_count()),B2d);
            auto C_ = distributed_array_view<decltype(C2d),decltype(new_comm)>(
                  std::addressof(new_comm),get_arr(C.grid()),get_arr(C.global_shape()),
                  get_arr(C.origin()),get_arr(C.tile_count()),C2d);
            detail::multiply_impl(a_v,math::detail::add_tags<Atr,Acg>(A_),math::detail::add_tags<Btr,Bcg>(B_),
                                  b_v,C_);
          }
      } else if constexpr (Arank==5) {
        for( auto [ia,a] : itertools::enumerate(dA.local_range(2)) )
          for( auto [ib,b] : itertools::enumerate(dA.local_range(3)) )
            for( auto [ic,c] : itertools::enumerate(dA.local_range(4)) ) {
              auto A2d = dA.local()(all,all,ia,ib,ic);
              auto B2d = dB.local()(all,all,ia,ib,ic);
              auto C2d = C.local()(all,all,ia,ib,ic);
              auto A_ = distributed_array_view<decltype(A2d),decltype(new_comm)>(
                    std::addressof(new_comm),get_arr(dA.grid()),get_arr(dA.global_shape()),
                    get_arr(dA.origin()),get_arr(dA.tile_count()),A2d);
              auto B_ = distributed_array_view<decltype(B2d),decltype(new_comm)>(
                    std::addressof(new_comm),get_arr(dB.grid()),get_arr(dB.global_shape()),
                    get_arr(dB.origin()),get_arr(dB.tile_count()),B2d);
              auto C_ = distributed_array_view<decltype(C2d),decltype(new_comm)>(
                    std::addressof(new_comm),get_arr(C.grid()),get_arr(C.global_shape()),
                    get_arr(C.origin()),get_arr(C.tile_count()),C2d);
              detail::multiply_impl(a_v,math::detail::add_tags<Atr,Acg>(A_),
                                    math::detail::add_tags<Btr,Bcg>(B_),
                                    b_v,C_);
            }
      } else {
        static_assert(Arank==2,"Finish implementation!");
      }

    }

  }

  return std::forward<C_t>(C);
}

template<typename A_t, typename B_t, typename C_t>
auto multiply(A_t&& A, B_t&& B, C_t&& C)
{
  using T = typename std::decay_t<A_t>::value_type;
  return multiply(T{1.0},std::forward<A_t>(A),std::forward<B_t>(B),
		  T{0.0},std::forward<C_t>(C));
}

} // math::nda

#endif
