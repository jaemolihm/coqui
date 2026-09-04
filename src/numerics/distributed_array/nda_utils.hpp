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


#ifndef NUMERICS_DISTRIBUTED_ARRAY_NDA_UTILS_HPP
#define NUMERICS_DISTRIBUTED_ARRAY_NDA_UTILS_HPP

#include <algorithm>
#include <cstdlib>
#include <numeric>
#include <utility>
#include <tuple>
#include <string_view>
#include "configuration.hpp"
#include "utilities/check.hpp"
#include "utilities/tile_partition.hpp"
#include "utilities/Timer.hpp"
#include "nda/nda.hpp"
#include "nda/tensor.hpp"
#include "nda/device.hpp"
#include "numerics/fft/nda.hpp"
#include "numerics/distributed_array/nda_matrix.hpp"
#include "numerics/distributed_array/slate_ops.hpp"
#include "numerics/shared_array/detail/concepts.hpp"

#include "mpi3/communicator.hpp"
#include "mpi3/request.hpp"

namespace math::nda 
{

/*
 * Factories for distributed_array
 */ 
/*
 * `tcount[n]` is the number of TILES axis n is cut into, not a tile size. The
 * partition is the balanced one of utilities/tile_partition.hpp -- read the worked
 * example at the top of that header for what a tile count means and which rank ends
 * up with which elements. Two of its properties matter here:
 *
 *  - tile boundaries depend on (shape[n], tcount[n]) alone, never on grid[n], so two
 *    axes of equal extent and equal tile count are tiled identically whatever grids
 *    they live on;
 *  - per-rank loads differ by at most one element per tile, and no rank or tile is
 *    empty, as long as grid[n] <= tcount[n] <= shape[n].
 *
 * `tcount[n] == 0` means "one element per tile", i.e. tcount[n] = shape[n]: the plain
 * balanced element distribution, and what an omitted tcount argument gives.
 *
 * Callers who want to bound the tile SIZE rather than fix a count pass
 * utils::max_tile_sizes and let the overload below derive the counts; two axes that
 * must share a partition instead pass explicit counts from
 * utils::balanced_tile_count(N, max(grid over the paired axes), max_tile_size).
 *
 * Out-of-range tile counts are rejected, not repaired: a silently clamped count
 * gives two axes different tile boundaries, which slate's gemm does not check.
 */

namespace detail
{

// (origin, local extent) per axis, walking the proc grid in the order the memory
// layout wants: column major over the grid for Fortran layout, row major otherwise.
template<bool is_stride_order_Fortran, int rank>
auto local_block(long ip,
                 std::array<long,rank> const& shape,
                 std::array<long,rank> const& grid,
                 std::array<long,rank> const& tcount)
{
  std::array<long,rank> origin, lshape;
  auto set = [&](int n) {
    auto [f,l] = utils::local_range_of_rank(shape[n],tcount[n],grid[n],ip%grid[n]);
    origin[n] = f;
    lshape[n] = l-f;
    ip /= grid[n];
  };
  if constexpr (is_stride_order_Fortran) {
    for(int n=0; n<rank; ++n) set(n);
  } else {
    for(int n=rank-1; n>=0; --n) set(n);
  }
  return std::make_pair(origin,lshape);
}

}

template<::nda::Array Array_base_t, typename communicator_t>
auto make_distributed_array(communicator_t& comm, 
		std::array<long,::nda::get_rank<std::decay_t<Array_base_t>>> grid,
		std::array<long,::nda::get_rank<std::decay_t<Array_base_t>>> shape,
		std::array<long,::nda::get_rank<std::decay_t<Array_base_t>>> tcount = {})
{
  using local_Array_t = typename std::decay_t<Array_base_t>::regular_type;
  static_assert( local_Array_t::layout_t::is_stride_order_Fortran() or
		 local_Array_t::layout_t::is_stride_order_C(), "Ordering mismatch.");
  static constexpr int rank = ::nda::get_rank<local_Array_t>;
  using Array_t = distributed_array<local_Array_t,communicator_t>;
  long np = std::accumulate(grid.cbegin(), grid.cend(), 1, std::multiplies<>{});
  utils::check( comm.size() == np, 
      "make_distributed_array: Number of processors does not match grid: size:{} grid:{}",comm.size(),np);
  for(int n=0; n<rank; ++n) 
    utils::check( shape[n] >= grid[n], 
      "make_distributed_array: Too many processors i:{}, shape:{}, grid:{}",n,shape[n],grid[n]); 
  utils::resolve_tile_counts<rank>(tcount,shape,grid,"make_distributed_array");

  auto [origin,lshape] = detail::local_block<
        local_Array_t::layout_t::is_stride_order_Fortran(),rank>(long(comm.rank()),shape,grid,tcount);
  return Array_t{ std::addressof(comm), grid, shape, lshape, origin, tcount}; 
}

/*
 * Same, from per-axis maximum tile SIZES instead of tile counts: the caller says how
 * big a tile may get and the factory -- which already has the shape and the grid --
 * derives the balanced counts. Use it whenever no two axes of the array have to
 * share a partition; when they do, pass explicit counts from
 * utils::balanced_tile_counts with the paired `p` on both axes.
 */
template<::nda::Array Array_base_t, typename communicator_t>
auto make_distributed_array(communicator_t& comm,
		std::array<long,::nda::get_rank<std::decay_t<Array_base_t>>> grid,
		std::array<long,::nda::get_rank<std::decay_t<Array_base_t>>> shape,
		utils::max_tile_sizes<::nda::get_rank<std::decay_t<Array_base_t>>> max_sizes)
{
  static constexpr int rank = ::nda::get_rank<std::decay_t<Array_base_t>>;
  for(int n=0; n<rank; ++n)
    utils::check( shape[n] >= grid[n],
      "make_distributed_array: Too many processors i:{}, shape:{}, grid:{}",n,shape[n],grid[n]);
  return make_distributed_array<Array_base_t>(comm,grid,shape,
      utils::balanced_tile_counts(shape,grid,max_sizes.max_size));
}

template<::nda::Array Array_base_t, typename communicator_t>
auto make_distributed_array_view(communicator_t& comm,
                std::array<long,::nda::get_rank<std::decay_t<Array_base_t>>> grid,
                std::array<long,::nda::get_rank<std::decay_t<Array_base_t>>> shape,
                std::array<long,::nda::get_rank<std::decay_t<Array_base_t>>> tcount,
                Array_base_t&& A_)
{ 
  using local_Array_t = typename std::decay_t<Array_base_t>::regular_type;
  static_assert( local_Array_t::layout_t::is_stride_order_Fortran() or
                 local_Array_t::layout_t::is_stride_order_C(), "Ordering mismatch.");
  static constexpr int rank = ::nda::get_rank<local_Array_t>;
  using Array_t = distributed_array_view<local_Array_t,communicator_t>;
  long np = std::accumulate(grid.cbegin(), grid.cend(), 1, std::multiplies<>{});
  utils::check( comm.size() == np,
      "make_distributed_array_view: Number of processors does not match grid: size:{} grid:{}",comm.size(),np);
  for(int n=0; n<rank; ++n)
    utils::check( shape[n] >= grid[n],
      "make_distributed_array_view: Too many processors i:{}, shape:{}, grid:{}",n,shape[n],grid[n]);
  utils::resolve_tile_counts<rank>(tcount,shape,grid,"make_distributed_array_view");

  auto [origin,lshape] = detail::local_block<
        local_Array_t::layout_t::is_stride_order_Fortran(),rank>(long(comm.rank()),shape,grid,tcount);
  for(int n=0; n<rank; ++n)
    utils::check(A_.shape(n) == lshape[n],
      "make_distributed_array_view: Size mismatch - dim:{}, local:{}, expected:{}",n,A_.shape(n),lshape[n]);
  return Array_t{ std::addressof(comm), grid, shape, origin, tcount, A_ };
}

namespace detail 
{

/*
// find elegant way to do this
template<long unsigned int N, ::nda::MemoryArray Arr_t>
auto get_sub_matrix(Arr_t && A , std::array<::nda::range,N> const& r)
{
  using Array_t = std::decay_t<Arr_t>;
  static_assert(::nda::get_rank<Array_t> == N,"Rank mismatch.");
  static_assert(N <= 7, "Extend manual unroll");
  if constexpr (N==1) {
    return A(r[0]);
  } else if constexpr (N==2) {
    return A(r[0],r[1]);
  } else if constexpr (N==3) {
    return A(r[0],r[1],r[2]);
  } else if constexpr (N==4) {
    return A(r[0],r[1],r[2],r[3]);
  } else if constexpr (N==5) {
    return A(r[0],r[1],r[2],r[3],r[4]);
  } else if constexpr (N==6) {
    return A(r[0],r[1],r[2],r[3],r[4],r[5]);
  } else if constexpr (N==7) {
    return A(r[0],r[1],r[2],r[3],r[4],r[5],r[6]);
  }
}
*/

template<long unsigned int N, ::nda::MemoryArray Arr_t>
auto get_sub_matrix(Arr_t && A , std::vector<::nda::range> const& r)
{
  using Array_t = std::decay_t<Arr_t>;
  static_assert(::nda::get_rank<Array_t> == N,"Rank mismatch.");
  static_assert(N <= 7, "Extend manual unroll");
  utils::check(r.size() == N, "get_sub_matrix: Size mismatch.");
  if constexpr (N==1) {
    return A(r[0]);
  } else if constexpr (N==2) {
    return A(r[0],r[1]);
  } else if constexpr (N==3) {
    return A(r[0],r[1],r[2]);
  } else if constexpr (N==4) {
    return A(r[0],r[1],r[2],r[3]);
  } else if constexpr (N==5) {
    return A(r[0],r[1],r[2],r[3],r[4]);
  } else if constexpr (N==6) {
    return A(r[0],r[1],r[2],r[3],r[4],r[5]);
  } else if constexpr (N==7) {
    return A(r[0],r[1],r[2],r[3],r[4],r[5],r[6]);
  }
}

// whether two ranges overlap
inline bool do_ranges_overlap(std::vector<::nda::range> const& R1, std::vector<::nda::range> const& R2)
{
  utils::check(R1.size() == R2.size(), "Size mismatch");
  int N = R1.size();
  bool disjoint = false;
  for(size_t i = 0; i < N; i++) {
      bool R2_out = R2[i].last() <= R1[i].first();
      bool R1_out = R1[i].last() <= R2[i].first();
      disjoint = disjoint || (R2_out || R1_out);  
  }
  return !disjoint;
}

// compute intersection of ranges
inline auto range_overlap(std::vector<::nda::range> const& R1, std::vector<::nda::range> const& R2)
{
  utils::check(R1.size() == R2.size(), "Size mismatch");
  utils::check(do_ranges_overlap(R1, R2), "Ranges are disjoint, cannot get non-empty overlap");
  int N = R1.size();
  std::vector<::nda::range> U(N,::nda::range(0));
  for(size_t i = 0; i < N; i++) {
    size_t first = std::max(R1[i].first(), R2[i].first());
    size_t last = std::min(R1[i].last(), R2[i].last());
    U[i] = ::nda::range(first, last);
  }
  return U;
}

} //detail

// Redistribute distributed array A to distributed array B
// Arr1/2_t can be views or value types of DistributedArray of SlateMatrix
template<DistributedArray Arr1_t, DistributedArray Arr2_t> // , int alg_type>
void redistribute_slow(Arr1_t& A, Arr2_t& B)
{
  using value_t = typename std::decay_t<Arr2_t>::Array_t::value_type;
  static_assert(get_rank<Arr1_t> == get_rank<Arr2_t>, "Rank mismatch.");
  utils::check(A.global_shape() == B.global_shape(), "Size mismatch.");
  utils::check(*A.communicator() == *B.communicator(), "Communicator mismatch.");
  // using A's communicator, since they should be compatible
  auto comm = (A.communicator());
  long mpi_size = comm->size();

  if(mpi_size==1) {
    B.local() = A.local();
    return;
  }

  ::nda::array<value_t,get_rank<Arr2_t>> Z(B.global_shape());
  Z()=0;
  if constexpr (get_rank<Arr2_t> == 1)
    Z(A.local_range(0)) = A.local();
  else if constexpr (get_rank<Arr2_t> == 2)
    Z(A.local_range(0),A.local_range(1)) = A.local();
  else if constexpr (get_rank<Arr2_t> == 3)
    Z(A.local_range(0),A.local_range(1),A.local_range(2)) = A.local();
  else if constexpr (get_rank<Arr2_t> == 4)
    Z(A.local_range(0),A.local_range(1),A.local_range(2),A.local_range(3)) = A.local();
  else if constexpr (get_rank<Arr2_t> == 5)
    Z(A.local_range(0),A.local_range(1),A.local_range(2),A.local_range(3),A.local_range(4)) = A.local();
  comm->all_reduce_in_place_n(Z.data(),Z.size(),std::plus<>{});
  if constexpr (get_rank<Arr2_t> == 1)
    B.local() = Z(B.local_range(0));
  else if constexpr (get_rank<Arr2_t> == 2)
    B.local() = Z(B.local_range(0),B.local_range(1));
  else if constexpr (get_rank<Arr2_t> == 3)
    B.local() = Z(B.local_range(0),B.local_range(1),B.local_range(2));
  else if constexpr (get_rank<Arr2_t> == 4)
    B.local() = Z(B.local_range(0),B.local_range(1),B.local_range(2),B.local_range(3));
  else if constexpr (get_rank<Arr2_t> == 5)
    B.local() = Z(B.local_range(0),B.local_range(1),B.local_range(2),B.local_range(3),B.local_range(4));
}

// Arr1/2_t can be views or value types of DistributedArray of SlateMatrix
/*
 * Matrix addition for distributed matrices with (possibly...) different distribution patterns.
 *
 *  B = a * A + b * B
 */
template<DistributedArray Arr1_t, DistributedArray Arr2_t> // , int alg_type>
void redistribute_standard(Arr1_t& A, Arr2_t& B, get_value_t<Arr1_t> a = 1, get_value_t<Arr2_t> b = 0)
{
  using local_Arr1_t = typename std::decay_t<Arr1_t>::Array_t::regular_type;
  using local_Arr2_t = typename std::decay_t<Arr2_t>::Array_t::regular_type;
  static_assert(get_rank<Arr1_t> == get_rank<Arr2_t>, "Rank mismatch.");
  utils::check(A.global_shape() == B.global_shape(), "Size mismatch.");
  utils::check(*A.communicator() == *B.communicator(), "Communicator mismatch.");
  constexpr long rank = get_rank<Arr1_t>; 
  const std::string indx = ::nda::tensor::default_index<uint8_t(rank)>();
  auto b_one = get_value_t<Arr2_t>{1};
  // using A's communicator, since they should be compatible
  auto comm = (A.communicator());
  long mpi_size = comm->size();
  long mpi_rank = comm->rank();

  auto Aloc = A.local();
  auto Bloc = B.local(); 

  if( b == get_value_t<Arr2_t>(0) ) {
    if(Bloc.size()>0) ::nda::tensor::set(get_value_t<Arr2_t>(0), Bloc);
  } else if(b != get_value_t<Arr2_t>(1)) {
    if(Bloc.size()>0) ::nda::tensor::scale(b, Bloc);
  } 

  // nothing else to do
  if( a == get_value_t<Arr1_t>(0) ) return;

  // serial case...
  if(mpi_size==1) {
    if constexpr( ::nda::mem::have_device_compatible_addr_space<local_Arr1_t,local_Arr2_t> ) {
      ::nda::tensor::add(a, Aloc, b_one, Bloc);
    } else {  
      static_assert( ::nda::mem::have_host_compatible_addr_space<local_Arr1_t,local_Arr2_t>, "oh oh." );
      Bloc += a * Aloc;
    }
    return;
  }

  ::nda::array<long,3> blocks{mpi_size,4,rank};
  ::nda::array<long,2> local_blocks{4,rank};
  std::vector<::nda::range> subblock(rank,::nda::range(0)); 
  std::copy_n(A.origin().data(),rank,local_blocks.data());
  std::copy_n(A.local_shape().data(),rank,local_blocks.data()+rank);
  std::copy_n(B.origin().data(),rank,local_blocks.data()+2*rank);
  std::copy_n(B.local_shape().data(),rank,local_blocks.data()+3*rank);
  comm->all_gather_n(local_blocks.data(),4*rank,blocks.data(),4*rank);

//   if constexpr (alg_type == 0) // iSend/iRecv
  std::vector<local_Arr2_t> send, recv;
  send.reserve(mpi_size);
  recv.reserve(mpi_size);
  std::vector<decltype(comm->isend_n(Aloc.data(),1,0))> stat_send;
  std::vector<std::pair<int,decltype(comm->isend_n(Aloc.data(),1,0))>> stat_recv;

  // compare local range in A with all ranges in B. iSend when needed
  for( auto p : itertools::range(mpi_size) ) { 
    bool ovlp = true;
    for( auto r : itertools::range(rank) ) {
      long i0 = local_blocks(0,r);
      long i1 = i0+local_blocks(1,r);
      long j0 = blocks(p,2,r);
      long j1 = j0+blocks(p,3,r);
      
      if( j1 > i0 and j0 < i1 ) {
        subblock[r] = ::nda::range(std::max(0l,j0-i0), std::min(j1-i0,local_blocks(1,r)));
      } else {
        ovlp = false;
        break;
      }
    }
    if(ovlp) {
      if( p == mpi_rank ) {
        auto A_ = detail::get_sub_matrix<rank>(Aloc,subblock);  
        // get subblock from B's perspective
        ovlp = true;
        for( auto r : itertools::range(rank) ) {
          long i0 = local_blocks(2,r);
          long i1 = i0+local_blocks(3,r);
          long j0 = blocks(p,0,r);
          long j1 = j0+blocks(p,1,r);
      
          if( j1 > i0 and j0 < i1 ) {
            subblock[r] = ::nda::range(std::max(0l,j0-i0), std::min(j1-i0,local_blocks(3,r)));
          } else { 
            ovlp = false;
            break;
          }
        }
        utils::check(ovlp,"Logic error in redistribute. FIX!");

        auto Bloc_p = detail::get_sub_matrix<rank>(Bloc,subblock);
        if constexpr( ::nda::mem::have_device_compatible_addr_space<local_Arr1_t,local_Arr2_t> ) {
          ::nda::tensor::add(a, A_, b_one, Bloc_p);
        } else {
          static_assert( ::nda::mem::have_host_compatible_addr_space<local_Arr1_t,local_Arr2_t>, "oh oh." );
          Bloc_p += a * A_;
        }

      } else {
        send.emplace_back( detail::get_sub_matrix<rank>(Aloc,subblock) );
        stat_send.emplace_back(comm->isend_n(send.back().data(),send.back().size(),p));
      }
    }
  }

  // compare local range in B with all ranges in A. iRecv when needed
  for( auto p : itertools::range(mpi_size) ) {
    if(p == mpi_rank) continue;
    bool ovlp = true;
    for( auto r : itertools::range(rank) ) {
      long i0 = local_blocks(2,r);
      long i1 = i0+local_blocks(3,r);
      long j0 = blocks(p,0,r);
      long j1 = j0+blocks(p,1,r);

      if( j1 > i0 and j0 < i1 ) {
        subblock[r] = ::nda::range(std::max(0l,j0-i0), std::min(j1-i0,local_blocks(3,r)));
      } else {
        ovlp = false;
        break;
      }
    }
    if(ovlp) {
      if( p != mpi_rank ) { // local copy already performed above
        recv.emplace_back( detail::get_sub_matrix<rank>(Bloc,subblock).shape() );
        stat_recv.emplace_back(std::make_pair(p,comm->ireceive_n(recv.back().data(),recv.back().size(),p)));
      }
    }
  }     

  // wait for messages and copy arrays
  // MAM: look for a way to process messages as they arrive, rather than just wait in order
  for( int i=0; i<stat_recv.size(); ++i ) { 
    auto& p = stat_recv[i].first;
    auto& st = stat_recv[i].second;
    st.wait();
    
    bool ovlp = true;
    for( auto r : itertools::range(rank) ) {
      long i0 = local_blocks(2,r);
      long i1 = i0+local_blocks(3,r);
      long j0 = blocks(p,0,r);
      long j1 = j0+blocks(p,1,r);

      if( j1 > i0 and j0 < i1 ) {
        subblock[r] = ::nda::range(std::max(0l,j0-i0), std::min(j1-i0,local_blocks(3,r)));
      } else {
        ovlp = false;
        break;
      }
    }
    utils::check(ovlp,"Logic error in redistribute. FIX!");
    auto Bloc_p = detail::get_sub_matrix<rank>(Bloc,subblock);
    if constexpr( ::nda::mem::have_device_compatible_addr_space<local_Arr1_t,local_Arr2_t> ) {
      ::nda::tensor::add(a, recv[i], b_one, Bloc_p);
    } else {
      static_assert( ::nda::mem::have_host_compatible_addr_space<local_Arr1_t,local_Arr2_t>, "oh oh." );
      Bloc_p += a * recv[i];
    }
  }

  // wait for sends
  for( auto& v: stat_send ) v.wait();

//   } else if constexpr() {
//      // Use 1-sided communicator
//   } else {
//     stattic_assert(alg_type > 1,"Invalid alg_type.");
//   } 
}

template<DistributedArray Arr1_t, DistributedArray Arr2_t> // , int alg_type>
void redistribute_no_order(Arr1_t& A, Arr2_t& B, get_value_t<Arr1_t> a = 1, get_value_t<Arr2_t> b = 0)
{
  using local_Arr1_t = typename std::decay_t<Arr1_t>::Array_t::regular_type;
  using local_Arr2_t = typename std::decay_t<Arr2_t>::Array_t::regular_type;
  static_assert(get_rank<Arr1_t> == get_rank<Arr2_t>, "Rank mismatch.");
  utils::check(A.global_shape() == B.global_shape(), "Size mismatch.");
  utils::check(*A.communicator() == *B.communicator(), "Communicator mismatch.");
  constexpr long rank = get_rank<Arr1_t>; 
  const std::string indx = ::nda::tensor::default_index<uint8_t(rank)>();
  auto b_one = get_value_t<Arr2_t>{1};
  // using A's communicator, since they should be compatible
  auto comm = (A.communicator());
  long mpi_size = comm->size();
  long mpi_rank = comm->rank();

  auto Aloc = A.local();
  auto Bloc = B.local(); 

  if( b == get_value_t<Arr2_t>(0) ) {
    if(Bloc.size()>0) ::nda::tensor::set(get_value_t<Arr2_t>(0), Bloc);
  } else if(b != get_value_t<Arr2_t>(1)) {
    if(Bloc.size()>0) ::nda::tensor::scale(b, Bloc);
  } 

  // nothing else to do
  if( a == get_value_t<Arr1_t>(0) ) return;

  // serial case...
  if(mpi_size==1) {
    if constexpr( ::nda::mem::have_device_compatible_addr_space<local_Arr1_t,local_Arr2_t> ) {
      ::nda::tensor::add(a, Aloc, b_one, Bloc);
    } else {  
      static_assert( ::nda::mem::have_host_compatible_addr_space<local_Arr1_t,local_Arr2_t>, "oh oh." );
      Bloc += a * Aloc;
    }
    return;
  }

  ::nda::array<long,3> blocks{mpi_size,4,rank};
  ::nda::array<long,2> local_blocks{4,rank};
  std::vector<std::vector<::nda::range>> subblocks_from_A(mpi_size); 
  std::vector<std::vector<::nda::range>> subblocks_from_B(mpi_size); 
  for(int i=0; i<mpi_size; ++i) {
    subblocks_from_A[i] = std::vector<::nda::range>(rank,::nda::range(0));
    subblocks_from_B[i] = std::vector<::nda::range>(rank,::nda::range(0));
  }
  std::vector<bool> ovlps_from_A(mpi_size);
  std::vector<bool> ovlps_from_B(mpi_size);

  std::copy_n(A.origin().data(),rank,local_blocks.data());
  std::copy_n(A.local_shape().data(),rank,local_blocks.data()+rank);
  std::copy_n(B.origin().data(),rank,local_blocks.data()+2*rank);
  std::copy_n(B.local_shape().data(),rank,local_blocks.data()+3*rank);
  comm->all_gather_n(local_blocks.data(),4*rank,blocks.data(),4*rank);

//   if constexpr (alg_type == 0) // iSend/iRecv
  std::vector<local_Arr2_t> send, recv;
  send.reserve(mpi_size);
  recv.reserve(mpi_size);
  std::vector<decltype(comm->isend_n(Aloc.data(),1,0))> stat_send;
//  std::vector<std::pair<int,decltype(comm->isend_n(Aloc.data(),1,0))>> stat_recv;
  std::vector<boost::mpi3::request> stat_recv2;
  std::vector<size_t> stat_recv2_p;

  // prepare subblock ranges from A
  for( auto p : itertools::range(mpi_size) ) { 
    ovlps_from_A[p] = true;
    for( auto r : itertools::range(rank) ) {
      long i0 = local_blocks(0,r);
      long i1 = i0+local_blocks(1,r);
      long j0 = blocks(p,2,r);
      long j1 = j0+blocks(p,3,r);
      
      if( j1 > i0 and j0 < i1 ) {
        subblocks_from_A[p][r] = ::nda::range(std::max(0l,j0-i0), std::min(j1-i0,local_blocks(1,r)));
      } else {
        ovlps_from_A[p] = false;
        break;
      }
    }
  }

  // prepare subblock ranges from B
  for( auto p : itertools::range(mpi_size) ) {
    ovlps_from_B[p] = true;
    for( auto r : itertools::range(rank) ) {
      long i0 = local_blocks(2,r);
      long i1 = i0+local_blocks(3,r);
      long j0 = blocks(p,0,r);
      long j1 = j0+blocks(p,1,r);

      if( j1 > i0 and j0 < i1 ) {
        subblocks_from_B[p][r] = ::nda::range(std::max(0l,j0-i0), std::min(j1-i0,local_blocks(3,r)));
      } else {
        ovlps_from_B[p] = false;
        break;
      }
    }
  }




  // compare local range in A with all ranges in B. iSend when needed
  for( auto p : itertools::range(mpi_size) ) { 
    if(ovlps_from_A[p]) {
      if( p == mpi_rank ) {
        auto A_ = detail::get_sub_matrix<rank>(Aloc,subblocks_from_A[p]);  
        // get subblock from B's perspective
        utils::check(ovlps_from_B[p],"1Logic error in redistribute. FIX!");

        auto Bloc_p = detail::get_sub_matrix<rank>(Bloc,subblocks_from_B[p]);
        if constexpr( ::nda::mem::have_device_compatible_addr_space<local_Arr1_t,local_Arr2_t> ) {
          ::nda::tensor::add(a, A_, b_one, Bloc_p);
        } else {
          static_assert( ::nda::mem::have_host_compatible_addr_space<local_Arr1_t,local_Arr2_t>, "oh oh." );
          Bloc_p += a * A_;
        }

      } else {
        send.emplace_back( detail::get_sub_matrix<rank>(Aloc,subblocks_from_A[p]) );
        stat_send.emplace_back(comm->isend_n(send.back().data(),send.back().size(),p,0));
      }
    }
  }

  // compare local range in B with all ranges in A. iRecv when needed
  for( auto p : itertools::range(mpi_size) ) {
    if(p == mpi_rank) continue;
    if(ovlps_from_B[p]) {
      if( p != mpi_rank ) { // local copy already performed above
        stat_recv2_p.emplace_back(p);
        recv.emplace_back( detail::get_sub_matrix<rank>(Bloc,subblocks_from_B[p]).shape() );
        stat_recv2.emplace_back(comm->ireceive_n(recv.back().data(),recv.back().size(),p));
      }
    }
  }     

  size_t recv_counter = stat_recv2.size(); // count unprocessed recv reqs
  std::vector<bool> recv_processed(stat_recv2.size(), false);
  while(recv_counter > 0) {
    bool any_valid = false;
    for(auto ireq :  itertools::range(stat_recv2.size())) 
        any_valid = any_valid || stat_recv2[ireq].valid();
    if(any_valid) { // not all are deallocated
      auto it_done = boost::mpi3::wait_any(stat_recv2.begin(), stat_recv2.end());

      // wait_any changes stat_recv2 buffer and sets the async deallocated requests to MPI_REQUEST_NULL
      // that's why need to get the actual index through distance and retrieve p from another vector
      long i = std::distance(stat_recv2.begin(), it_done);
      utils::check(i>=0,"Logic error in redistribute. FIX!");
      utils::check(i < stat_recv2.size(),"Logic error in redistribute. FIX!");

      auto p = stat_recv2_p[i];
      utils::check(p < mpi_size,"Logic error in redistribute. FIX!");
      utils::check(ovlps_from_B[p],"Logic error in redistribute. FIX!"); 
      auto Bloc_p = detail::get_sub_matrix<rank>(Bloc,subblocks_from_B[p]);
    
      if constexpr( ::nda::mem::have_device_compatible_addr_space<local_Arr1_t,local_Arr2_t> ) {
        ::nda::tensor::add(a, recv[i], b_one, Bloc_p);
      } else {
        static_assert( ::nda::mem::have_host_compatible_addr_space<local_Arr1_t,local_Arr2_t>, "oh oh." );
        Bloc_p += a * recv[i];
      }
      recv_processed[i] = true; 
      recv_counter--;
    }
    else { // all recv requests are deallocated, process all the remaining ones
        for(auto i : itertools::range(recv_processed.size())) {
            if(recv_processed[i]) continue; // already processed
            auto p = stat_recv2_p[i];
            
            utils::check(p < mpi_size,"Logic error in redistribute. FIX!");
            utils::check(ovlps_from_B[p],"Logic error in redistribute. FIX!"); 
            auto Bloc_p = detail::get_sub_matrix<rank>(Bloc,subblocks_from_B[p]);
            
            if constexpr( ::nda::mem::have_device_compatible_addr_space<local_Arr1_t,local_Arr2_t> ) {
              ::nda::tensor::add(a, recv[i], b_one, Bloc_p);
            } else {
              static_assert( ::nda::mem::have_host_compatible_addr_space<local_Arr1_t,local_Arr2_t>, "oh oh." );
              Bloc_p += a * recv[i];
            }
            recv_processed[i] = true; 
        }
        break; // no need to wait anymore, all reqs have been processed
    }
  } // while

  // wait for sends
  for( auto& v: stat_send ) v.wait();

}

/// Default per-rank staging-buffer budget for redistribute_alltoallv. A
/// transfer whose largest local block fits below this is a single
/// all_to_all_v_n, i.e. exactly what the routine did before chunking existed.
/// COQUI_REDISTRIBUTE_CHUNK_MB overrides it, so the chunked-vs-unchunked timing
/// arm needs no rebuild (a very large value disables chunking).
inline size_t redistribute_chunk_cap_bytes() {
  static const size_t cap = [] {
    const char* env = std::getenv("COQUI_REDISTRIBUTE_CHUNK_MB");
    const double mb = (env != nullptr) ? std::atof(env) : 0.0;
    return (mb > 0.0) ? size_t(mb * 1024.0 * 1024.0) : size_t(256) * 1024 * 1024;
  }();
  return cap;
}

/// Round schedule of the chunked exchange in redistribute_alltoallv. Round c
/// covers the rank offsets [c*grp, (c+1)*grp): a rank sends to (me + o) % np and
/// receives from (me - o) % np.
///
/// MPI_Alltoallv requires sendcount_i[j] == recvcount_j[i] in every call, so the
/// round in which i sends to j must be the round in which j receives from i.
/// Grouping the rounds by offset is what makes that hold; grouping by peer index
/// does not -- it posts "everyone -> group" sends against "group -> everyone"
/// receives, which agree only inside the group. That is illegal even where an
/// MPI survives it: small messages are buffered and matched out of order by a
/// later round, so the result can come out bit-correct on one node and still
/// deadlock over a fabric at scale.
///
/// Split out of the loop so that invariant can be asserted directly, rather than
/// inferred from whether a given transport tolerates violating it.
struct redistribute_schedule {
  long np;      // communicator size
  long nchunk;  // number of rounds
  long grp;     // offsets per round

  redistribute_schedule(long np_, long nchunk_)
    : np(np_), nchunk(nchunk_), grp((np_ + nchunk_ - 1) / nchunk_) {}

  long send_peer(long me, long o) const { return (me + o) % np; }
  long recv_peer(long me, long o) const { return (me - o + np) % np; }
  long round_of(long o)           const { return o / grp; }
};

/**
 * @param chunk_cap_bytes - staging-buffer budget per rank; 0 selects
 *        redistribute_chunk_cap_bytes. Only the blocking of the exchange
 *        depends on it, never the result.
 */
template<DistributedArray Arr1_t, DistributedArray Arr2_t>
void redistribute_alltoallv(Arr1_t& A, Arr2_t& B, get_value_t<Arr1_t> a = 1, get_value_t<Arr2_t> b = 0,
                            size_t chunk_cap_bytes = 0) {
  using value_t = typename std::decay_t<Arr2_t>::Array_t::value_type;
  using local_Arr1_t = typename std::decay_t<Arr1_t>::Array_t::regular_type;
  using local_Arr2_t = typename std::decay_t<Arr2_t>::Array_t::regular_type;
  static_assert(get_rank<Arr1_t> == get_rank<Arr2_t>, "Rank mismatch.");
  utils::check(A.global_shape() == B.global_shape(), "Size mismatch.");
  utils::check(*A.communicator() == *B.communicator(), "Communicator mismatch.");


  utils::check(a == get_value_t<Arr2_t>(1) && b == get_value_t<Arr2_t>(0),"Non-default redistribute_alltoallv is not ready yet");

  constexpr long rank = get_rank<Arr1_t>; 
  const std::string indx = ::nda::tensor::default_index<uint8_t(rank)>();
  auto b_one = get_value_t<Arr2_t>{1};
  // using A's communicator, since they should be compatible
  auto comm = (A.communicator());
  long mpi_size = comm->size();

  auto Aloc = A.local();
  auto Bloc = B.local();

  // nothing else to do
  if( a == get_value_t<Arr1_t>(0) ) return;

  // serial case...
  if(mpi_size==1) {
    // add-into semantics below need B pre-scaled by b (b==0 -> zero first)
    if( b == get_value_t<Arr2_t>(0) ) {
      if(Bloc.size()>0) ::nda::tensor::set(get_value_t<Arr2_t>(0), Bloc);
    } else if(b != get_value_t<Arr2_t>(1)) {
      if(Bloc.size()>0) ::nda::tensor::scale(b, Bloc);
    }
    if constexpr( ::nda::mem::have_device_compatible_addr_space<local_Arr1_t,local_Arr2_t> ) {
      ::nda::tensor::add(a, Aloc, b_one, Bloc);
    } else {
      static_assert( ::nda::mem::have_host_compatible_addr_space<local_Arr1_t,local_Arr2_t>, "oh oh." );
      Bloc += a * Aloc;
    }
    return;
  }
  // parallel path: no pre-zero needed - the sz_buf_B == Bloc.size() full-coverage
  // assertion below guarantees the unpack loop overwrites every element of Bloc.

  ::nda::array<long,3> blocks{mpi_size,4,rank};
  ::nda::array<long,2> local_blocks{4,rank};
  std::vector<std::vector<::nda::range>> subblocks_from_A(mpi_size);
  std::vector<std::vector<::nda::range>> subblocks_from_B(mpi_size);
  for(int i=0; i<mpi_size; ++i) {
    subblocks_from_A[i] = std::vector<::nda::range>(rank,::nda::range(0));
    subblocks_from_B[i] = std::vector<::nda::range>(rank,::nda::range(0));
  }
  std::vector<bool> ovlps_from_A(mpi_size);
  std::vector<bool> ovlps_from_B(mpi_size);

  std::copy_n(A.origin().data(),rank,local_blocks.data());
  std::copy_n(A.local_shape().data(),rank,local_blocks.data()+rank);
  std::copy_n(B.origin().data(),rank,local_blocks.data()+2*rank);
  std::copy_n(B.local_shape().data(),rank,local_blocks.data()+3*rank);
  comm->all_gather_n(local_blocks.data(),4*rank,blocks.data(),4*rank);

//   if constexpr (alg_type == 0) // iSend/iRecv
  std::vector<local_Arr2_t> send, recv;
  send.reserve(mpi_size);
  recv.reserve(mpi_size);
  std::vector<decltype(comm->isend_n(Aloc.data(),1,0))> stat_send;
//  std::vector<std::pair<int,decltype(comm->isend_n(Aloc.data(),1,0))>> stat_recv;
  std::vector<boost::mpi3::request> stat_recv2;
  std::vector<size_t> stat_recv2_p;

  // prepare subblock ranges from A
  for( auto p : itertools::range(mpi_size) ) { 
    ovlps_from_A[p] = true;
    for( auto r : itertools::range(rank) ) {
      long i0 = local_blocks(0,r);
      long i1 = i0+local_blocks(1,r);
      long j0 = blocks(p,2,r);
      long j1 = j0+blocks(p,3,r);
      
      if( j1 > i0 and j0 < i1 ) {
        subblocks_from_A[p][r] = ::nda::range(std::max(0l,j0-i0), std::min(j1-i0,local_blocks(1,r)));
      } else {
        ovlps_from_A[p] = false;
        break;
      }
    }
  }

  // prepare subblock ranges from B
  for( auto p : itertools::range(mpi_size) ) {
    ovlps_from_B[p] = true;
    for( auto r : itertools::range(rank) ) {
      long i0 = local_blocks(2,r);
      long i1 = i0+local_blocks(3,r);
      long j0 = blocks(p,0,r);
      long j1 = j0+blocks(p,1,r);

      if( j1 > i0 and j0 < i1 ) {
        subblocks_from_B[p][r] = ::nda::range(std::max(0l,j0-i0), std::min(j1-i0,local_blocks(3,r)));
      } else {
        ovlps_from_B[p] = false;
        break;
      }
    }
  }

  std::vector<int> A_counts(mpi_size);
  std::vector<int> B_counts(mpi_size);
  for( auto p : itertools::range(mpi_size) ) {
    A_counts[p] = (ovlps_from_A[p]) ? detail::get_sub_matrix<rank>(Aloc,subblocks_from_A[p]).size() : 0;
    B_counts[p] = (ovlps_from_B[p]) ? detail::get_sub_matrix<rank>(Bloc,subblocks_from_B[p]).size() : 0;
  }

  size_t sz_buf_A = std::accumulate(A_counts.begin(), A_counts.end(), size_t(0), std::plus<>());
  size_t sz_buf_B = std::accumulate(B_counts.begin(), B_counts.end(), size_t(0), std::plus<>());

  utils::check(sz_buf_A == size_t(Aloc.size()), "A Size mismatch.");
  utils::check(sz_buf_B == size_t(Bloc.size()), "B Size mismatch.");

  // Offset-round chunking: rather than staging the whole local block of both
  // endpoints, exchange with a subset of the peers at a time, so the pack/unpack
  // buffers hold ~1/nchunk of it. Which elements go where, and in what order they
  // are packed, does not depend on nchunk.
  //
  // The rounds group *rank offsets*, not peer indices: in round c this rank sends
  // to (my_rank + o) % mpi_size and receives from (my_rank - o) % mpi_size, for
  // the offsets o of that round. MPI_Alltoallv requires sendcount_i[j] ==
  // recvcount_j[i], and offset grouping is what makes that hold -- rank i sends
  // to j = (i+o) % mpi_size in the very round in which j receives from
  // (j-o) % mpi_size == i. Grouping by peer index instead posts
  // "everyone -> group" sends against "group -> everyone" receives, which agree
  // only inside the group; every other pair is an unmatched send, which an MPI
  // may buffer and match out of order in a later round, or may block on
  // indefinitely, depending on message size and transport. It also spreads the
  // traffic: at each offset every rank has a distinct partner, instead of the
  // whole communicator pushing into the same group of ranks.
  //
  // nchunk comes from the *globally* largest local block, which the all_gather
  // above already put on every rank, so the round count is agreed without extra
  // communication.
  // This also bounds the displacements, which are int: they now span one round
  // instead of the whole block. (The per-peer counts are unchanged and still
  // int; one peer's sub-block exceeding 2^31 elements would overflow as before,
  // but such a chunk is already past the byte budget.)
  long nchunk = 1;
  {
    const size_t cap = (chunk_cap_bytes > 0) ? chunk_cap_bytes : redistribute_chunk_cap_bytes();
    const size_t cap_elem = std::max(size_t(1), cap / sizeof(value_t));
    size_t gmax = 0;
    for( auto p : itertools::range(mpi_size) ) {
      size_t szA = 1, szB = 1;
      for( auto r : itertools::range(rank) ) {
        szA *= size_t(blocks(p,1,r));
        szB *= size_t(blocks(p,3,r));
      }
      gmax = std::max(gmax, std::max(szA, szB));
    }
    nchunk = std::clamp(long((gmax + cap_elem - 1) / cap_elem), 1l, mpi_size);
  }
  const redistribute_schedule sched(mpi_size, nchunk);
  const long grp_size = sched.grp;  // offsets per round
  const long my_rank = comm->rank();
  auto send_peer = [&](long o) { return sched.send_peer(my_rank, o); };
  auto recv_peer = [&](long o) { return sched.recv_peer(my_rank, o); };

  size_t max_A = 0, max_B = 0;
  for( long c = 0; c < nchunk; ++c ) {
    const long o0 = std::min(c*grp_size, mpi_size), o1 = std::min(o0+grp_size, mpi_size);
    size_t sa = 0, sb = 0;
    for( long o = o0; o < o1; ++o ) {
      sa += size_t(A_counts[send_peer(o)]);
      sb += size_t(B_counts[recv_peer(o)]);
    }
    max_A = std::max(max_A, sa);
    max_B = std::max(max_B, sb);
  }

  std::vector<value_t> buffer_A(max_A);
  std::vector<value_t> buffer_B(max_B);
  std::vector<int> A_cnt(mpi_size, 0), A_disp(mpi_size, 0);
  std::vector<int> B_cnt(mpi_size, 0), B_disp(mpi_size, 0);

  size_t count_sz_check = 0;
  for( long c = 0; c < nchunk; ++c ) {
    const long o0 = std::min(c*grp_size, mpi_size), o1 = std::min(o0+grp_size, mpi_size);
    if(o1 <= o0) continue;
    std::fill(A_cnt.begin(), A_cnt.end(), 0);
    std::fill(B_cnt.begin(), B_cnt.end(), 0);
    std::fill(A_disp.begin(), A_disp.end(), 0);
    std::fill(B_disp.begin(), B_disp.end(), 0);
    int offA = 0, offB = 0;
    for( long o = o0; o < o1; ++o ) {
      const long ps = send_peer(o), pr = recv_peer(o);
      A_cnt[ps] = A_counts[ps];  A_disp[ps] = offA;  offA += A_counts[ps];
      B_cnt[pr] = B_counts[pr];  B_disp[pr] = offB;  offB += B_counts[pr];
    }

    // copy intersections of A with this round's peers into the send buffer
    for( long o = o0; o < o1; ++o ) {
      const long p = send_peer(o);
      if(ovlps_from_A[p]) {
        auto A_ = detail::get_sub_matrix<rank>(Aloc,subblocks_from_A[p]);
        if constexpr( ::nda::mem::have_host_compatible_addr_space<local_Arr1_t,local_Arr2_t> ) {
          // pack strided sub-block directly into the contiguous send buffer
          ::nda::array_view<value_t,rank>(A_.shape(), buffer_A.data()+A_disp[p]) = A_;
        } else {
          auto A_reg = make_regular(A_);
          std::copy_n(A_reg.data(),A_reg.size(),buffer_A.data()+A_disp[p]);
        }
        count_sz_check += A_.size();
      }
    }

    comm->all_to_all_v_n(buffer_A.data(), A_cnt.data(), A_disp.data(),
                         buffer_B.data(), B_cnt.data(), B_disp.data());

    for( long o = o0; o < o1; ++o ) {
      const long p = recv_peer(o);
      if(ovlps_from_B[p]) {
        auto B_ = detail::get_sub_matrix<rank>(Bloc,subblocks_from_B[p]);
        if constexpr( ::nda::mem::have_host_compatible_addr_space<local_Arr1_t,local_Arr2_t> ) {
          // Direct overwrite of the strided sub-block from the contiguous recv
          // buffer. This relies on the a==1 && b==0 assertion at the top of the
          // function. If that restriction is ever lifted, this must become
          //   B_ += a * (buffer view)
          // and the b-scaling prologue (currently only in the serial branch) must
          // return to the parallel path.
          B_ = ::nda::array_view<const value_t,rank>(B_.shape(), buffer_B.data()+B_disp[p]);
        } else {
          auto B_reg = make_regular(B_);
          std::copy_n(buffer_B.data()+B_disp[p],B_reg.size(),B_reg.data());
          B_ = B_reg;
        }
      }
    }
  }
  utils::check(count_sz_check == size_t(Aloc.size()), "A Size mismatch.");
}

template<DistributedArray Arr1_t, DistributedArray Arr2_t, int Alg = 3> 
void redistribute(Arr1_t& A, Arr2_t& B, get_value_t<Arr1_t> a = 1, get_value_t<Arr2_t> b = 0) {

  switch(Alg) {
    case 1:
      redistribute_standard(A, B, a, b);
      break;
    case 2:
      redistribute_no_order(A, B, a, b);
      break;
    case 3:
      redistribute_alltoallv(A, B, a, b);
      break;
  }
}


template<DistributedArray Arr1_t> // , int alg_type>
void redistribute_in_place(Arr1_t& A, std::array<long,get_rank<Arr1_t>> grid,  
        std::array<long,get_rank<Arr1_t>> tcount, get_value_t<Arr1_t> a = 1)
{
  using local_Array_t = typename std::decay_t<Arr1_t>::Array_t::regular_type;
  auto B{make_distributed_array<local_Array_t>(*(A.communicator()),grid,A.global_shape(),tcount)};
  redistribute(A,B,a,get_value_t<Arr1_t>{0});
  A = std::move(B);
}

template<DistributedArrayOfRank<2> Arr1_t, DistributedArrayOfRank<2> Arr2_t>
void distributed_column_select(::nda::array<long,1> const& rn, Arr1_t const& A, Arr2_t& B)
{
  utils::check(A.global_shape()[0] == B.global_shape()[0], "Size mismatch.");
  utils::check(B.global_shape()[1] == rn.shape(0), "Size mismatch.");
  utils::check(*A.communicator() == *B.communicator(), "Communicator mismatch.");
  decltype(::nda::range::all) all;
 
  auto comm = A.communicator();
  auto Aloc = A.local();
  auto Bloc = B.local();
  long nrows = A.global_shape()[0];
  long nproc = comm->size();
  auto a_range = A.local_range(1);
  auto b_range = B.local_range(1);

  if(nproc==1) {
    for( auto [i,r] : itertools::enumerate(rn) ) 
      Bloc(all,i) = Aloc(all,r);
    return;
  }

  // assuming packed range, e.g. ending of range or proc 'n' is the beginning of range of proc 'n+1'
  ::nda::array<long,2> bounds(2,nproc);
  bounds()=0;
  bounds(0,comm->rank()) = A.local_shape()[1];
  bounds(1,comm->rank()) = B.local_shape()[1];
  comm->all_reduce_in_place_n(bounds.data(),2*nproc,std::plus<>{});

  // setup bounds
  {
    long cnt(0);
    for( auto i : itertools::range(nproc) ) { 
      cnt+=bounds(0,i);
      bounds(0,i) = cnt;
    }
    utils::check(cnt==A.global_shape()[1], "Distribution mismatch.");
    cnt=0;
    for( auto i : itertools::range(nproc) ) { 
      cnt+=bounds(1,i);
      bounds(1,i) = cnt;
    }
    utils::check(cnt==B.global_shape()[1], "Distribution mismatch.");
  }

  /*
   * (0,i): number of terms/vectors to send to proc i 
   * (1,i): send_disp for all_to_all
   * (2,i): number of terms/vectors to receive from proc i
   * (3,i): recv displ for all_to_all
   */   
  ::nda::array<int,2> cnts(4,nproc);
  cnts() = 0; 

  auto a_bounds = bounds(0,all);
  auto b_bounds = bounds(1,all);
  for( auto [i,r] : itertools::enumerate(rn) ) {
    // check if I have point 'r', if I do increase counter based on where is 'i' in bounds(1,:)
    if( a_range.first() <= r and r < a_range.last() ) {
      long ip = std::distance(b_bounds.begin(), 
                              std::lower_bound(b_bounds.begin(), b_bounds.end(), long(i+1))); 
      cnts(0,ip)++;
    }
  }
  // now calculate number of terms received by each processor 
  for( auto [i,r] : itertools::enumerate(rn(b_range)) ) {
    long ip = std::distance(a_bounds.begin(), 
                            std::lower_bound(a_bounds.begin(), a_bounds.end(), long(r+1)));
    cnts(2,ip)++;    
  }

  long nsend(0);
  long nrecv(0);
  for( auto i : itertools::range(nproc) ) {
    cnts(1,i) = int(nsend);
    cnts(3,i) = int(nrecv);
    nsend+=long(cnts(0,i));
    nrecv+=long(cnts(2,i));
  }
  utils::check(nrecv == b_range.size(), "Distribution mismatch: nrecv==b_range.size()");

  ::nda::array<ComplexType,2> At(nsend,nrows);
  ::nda::array<ComplexType,2> Bt(nrecv,nrows);

  // now copy into array
  cnts(0,all) = 0;
  for( auto [i,r] : itertools::enumerate(rn) ) {
    // check if I have point 'r', if I do increase counter based on where is 'i' in bounds(1,:)
    if( a_range.first() <= r and r < a_range.last() ) {
      long ip = std::distance(b_bounds.begin(),
                              std::lower_bound(b_bounds.begin(), b_bounds.end(), long(i+1)));
      At( cnts(1,ip)+cnts(0,ip) ,all) = Aloc(all,r-a_range.first()); 
      cnts(0,ip)++;
    }
  }

  // rescale counts
  cnts() *= int(nrows);
  // hard-wired to complex<double>, fix fix fix
  MPI_Alltoallv(At.data(), cnts.data(), cnts.data()+nproc, MPI_CXX_DOUBLE_COMPLEX,
                Bt.data(), cnts.data()+2*nproc, cnts.data()+3*nproc, MPI_CXX_DOUBLE_COMPLEX, 
                comm->get());
  // rescale counts back
  cnts() /= int(nrows);

  // now copy into B
  cnts(2,all) = 0;
  for( auto [i,r] : itertools::enumerate(rn(b_range)) ) {
    long ip = std::distance(a_bounds.begin(),
                            std::lower_bound(a_bounds.begin(), a_bounds.end(), long(r+1)));
    Bloc(all, i) = Bt( cnts(3,ip)+cnts(2,ip) ,all); 
    cnts(2,ip)++;
  }     
}

// Gather a distributed array G to a local array L at processor "p"
template<DistributedArray dArrG_t, ::nda::MemoryArray ArrL_t>
void gather(int p, dArrG_t const& G, ArrL_t* L) 
requires( get_rank<dArrG_t> == ::nda::get_rank<ArrL_t> )
{
  using local_Array_t = typename std::decay_t<ArrL_t>::regular_type;
  auto comm = G.communicator();
  utils::check( p>=0 and p < comm->size(), "Error: Communicator mismatch." );
  if( comm->rank() == p ) { 
    utils::check( L != nullptr, " Error: Nullptr in math::nda::gather." );
    utils::check( L->shape() == G.global_shape(), "Error: Shape mismatch.");
    //if( L->shape() != G.global_shape() )
    //  L->resize(G.global_shape());
  }
  constexpr long rank = get_rank<dArrG_t>;
  long mpi_size = comm->size();

  if(mpi_size==1) {
    *L = G.local();
    return;
  }

  if( comm->rank() == p ) {

    auto Lloc = (*L)();
    ::nda::array<long,3> blocks{mpi_size,2,rank};
    {
      ::nda::array<long,2> local_block{2,rank};  
      std::copy_n(G.origin().data(),rank,local_block.data());
      std::copy_n(G.local_shape().data(),rank,local_block.data()+rank);
      comm->gather_n(local_block.data(),2*rank,blocks.data(),2*rank,p); 
    }

    // MAM: nda::range deprecated default constr, so swithc to vector
    std::vector<::nda::range> subblock(rank,::nda::range(0));  
    // assemble subblock from blocks given an mpi rank
    auto get_subblock = [&] (int q) {
      for( auto r : itertools::range(rank) )
        subblock[r] = ::nda::range( blocks(q,0,r), blocks(q,0,r)+blocks(q,1,r) );
    };

    std::vector<local_Array_t> recv;
    std::vector<std::pair<int,decltype(comm->ireceive_n(Lloc.data(),1,p))>> stat;
    recv.reserve(mpi_size-1);	
    stat.reserve(mpi_size-1);	

    for( auto i : itertools::range(mpi_size) ) {
      get_subblock(i);
      if(i==p) {
	detail::get_sub_matrix<rank>(Lloc,subblock) = G.local();
	continue;
      } else {
        // post receives
        auto Lsub = detail::get_sub_matrix<rank>(Lloc,subblock);
        recv.emplace_back( Lsub.shape() );
        stat.emplace_back(std::make_pair(i,comm->ireceive_n(recv.back().data(),recv.back().size(),i)));
      }
    }    

    // now wait for messages
    for( int i=0; i<stat.size(); ++i ) {
      // MAM: look for a way to process messages as they arrive, rather than just wait in order
      auto& rk = stat[i].first;
      auto& st = stat[i].second;
      st.wait();

      get_subblock(rk);
      detail::get_sub_matrix<rank>(Lloc,subblock) = recv[i];
    }   

  } else {
    ::nda::array<long,2> block{2,rank};  
    std::copy_n(G.origin().data(),rank,block.data());
    std::copy_n(G.local_shape().data(),rank,block.data()+rank);
    comm->gather_n(block.data(),2*rank,block.data(),2*rank,p); 
    auto Gloc = G.local();
    if(Gloc.is_contiguous()) {
      comm->send_n(Gloc.data(),Gloc.size(),p);
    } else {
     typename  std::decay_t<dArrG_t>::Array_t G_ = Gloc;  
      comm->send_n(G_.data(),G_.size(),p);
    }
  }
  comm->barrier();
}

// Gather a range of distributed array G to a local array L at processor "p"
// The local array is assumed to be consistent with the requested range sizes
template<DistributedArray dArrG_t, ::nda::MemoryArray ArrL_t>
void gather_ranged(int p, dArrG_t const& G, ArrL_t* L, std::vector<::nda::range> Arr_rng) 
requires( get_rank<dArrG_t> == ::nda::get_rank<ArrL_t>)
{
  using local_Array_t = typename std::decay_t<ArrL_t>::regular_type;
  auto comm = G.communicator();
  utils::check( p>=0 and p < comm->size(), "Error: Communicator mismatch." );
  constexpr long rank = get_rank<dArrG_t>;
  for(size_t ir = 0; ir < rank; ir++)
    utils::check( Arr_rng[ir].step() == 1, "Error: Range step must be 1.");

  for(size_t ir = 0; ir < rank; ir++)
    utils::check( (Arr_rng[ir].first() >= 0 and Arr_rng[ir].first() < G.global_shape()[ir]) 
              and (Arr_rng[ir].last() > 0 and Arr_rng[ir].last() <= G.global_shape()[ir]) , "Error: Range shape does not fit into global shape.");

  if( comm->rank() == p ) { 
    utils::check( L != nullptr, " Error: Nullptr in math::nda::gather." );
    for(size_t ir = 0; ir < rank; ir++)
        utils::check( L->shape()[ir] == Arr_rng[ir].size(), "Error: Shape mismatch.");
    //if( L->shape() != G.global_shape() )
    //  L->resize(G.global_shape());
  }
  long mpi_size = comm->size();

  if(mpi_size==1) {
    *L = detail::get_sub_matrix<rank>(G.local(), Arr_rng);
    return;
  }

  if( comm->rank() == p ) {
    auto Lloc = (*L)();

    // origins and ends of requested range ⋂ local range at every proc
    ::nda::array<long,2> blocks_intsect_origins(mpi_size,rank); 
    ::nda::array<long,2> blocks_intsect_ends(mpi_size,rank); 
    // whether the requested range overlaps with loc range at every proc
    ::nda::array<bool,1> blocks_w_r_overlap(mpi_size); 
    {
      std::vector<::nda::range> local_ranges(rank,::nda::range(0));
      for(size_t i = 0; i < rank; i++) local_ranges[i] = G.local_range(i);

      ::nda::array<long, 1> local_origins(rank);
      ::nda::array<long, 1> local_ends(rank);
      local_origins() = 0;
      local_ends() = 0;
      for(size_t i = 0; i < rank; i++) local_origins(i) = local_ranges[i].first();
      for(size_t i = 0; i < rank; i++) local_ends(i) = local_ranges[i].last();

      // compute intersections with the requested ranges
      ::nda::array<long, 1> local_intsect_origins(rank); 
      ::nda::array<long, 1> local_intsect_ends(rank);

      bool whether_ranges_overlap = detail::do_ranges_overlap(local_ranges, Arr_rng);

      std::vector<::nda::range> range_intersect = Arr_rng;
      if(whether_ranges_overlap) range_intersect = detail::range_overlap(local_ranges, Arr_rng);

      std::copy_n(local_origins.data(),rank,local_intsect_origins.data());
      std::copy_n(local_ends.data(),rank,local_intsect_ends.data());
      
      if(whether_ranges_overlap) {
        for(size_t i = 0; i < rank; i++) local_intsect_origins(i) = range_intersect[i].first();
        for(size_t i = 0; i < rank; i++) local_intsect_ends(i) = range_intersect[i].last();
      }

      comm->gather_n(&whether_ranges_overlap,1,blocks_w_r_overlap.data(),1,p); 
      comm->gather_n(local_intsect_origins.data(),rank,blocks_intsect_origins.data(),rank,p); 
      comm->gather_n(local_intsect_ends.data(),rank,blocks_intsect_ends.data(),rank,p); 
    }

    std::array<long,rank> subblock_in_dist; // subblock_in[i].last()-subblock_in[i].first() for message size estimate
    std::vector<::nda::range> subblock_out(rank,::nda::range(0)); // to write in the output array
    std::vector<::nda::range> subblock_in(rank,::nda::range(0)); // to access local array
    // assemble subblock from blocks given an mpi rank
    auto get_subblock_out = [&] (int q) {
      for( auto r : itertools::range(rank) )
        subblock_out[r] = ::nda::range( blocks_intsect_origins(q,r)-Arr_rng[r].first(), blocks_intsect_ends(q,r)-Arr_rng[r].first() );
    };
    auto get_subblock_in = [&] (int q) {
      for( auto r : itertools::range(rank) )
        subblock_in[r] = ::nda::range(blocks_intsect_origins(q,r), blocks_intsect_ends(q,r) );
    };
    auto get_subblock_in_loc_origin = [&] () {
      for( auto r : itertools::range(rank) )
        subblock_in[r] = ::nda::range(blocks_intsect_origins(p,r)-G.origin()[r], blocks_intsect_ends(p,r)-G.origin()[r] );
    };

    std::vector<local_Array_t> recv;
    std::vector<std::pair<int,decltype(comm->ireceive_n(Lloc.data(),1,p))>> stat; 
    recv.reserve(mpi_size-1);	
    stat.reserve(mpi_size-1);	

    for( auto i : itertools::range(mpi_size) ) {
      if(blocks_w_r_overlap(i)) {
        get_subblock_in(i); 
        get_subblock_out(i); 
        if(i==p) {
          get_subblock_in_loc_origin(); // shift by G.origin()
          detail::get_sub_matrix<rank>(Lloc,subblock_out) = detail::get_sub_matrix<rank>(G.local(),subblock_in);
          continue;
        } else {
          // post receives
          for( auto r : itertools::range(rank) )
              subblock_in_dist[r] = subblock_in[r].last()-subblock_in[r].first();
          recv.emplace_back(subblock_in_dist);
          stat.emplace_back(std::make_pair(i,comm->ireceive_n(recv.back().data(),recv.back().size(),i)));
        }
      }    
    }    

    // now wait for messages
    for( int i=0; i<stat.size(); ++i ) {
      // MAM: look for a way to process messages as they arrive, rather than just wait in order
      auto& rk = stat[i].first;
      auto& st = stat[i].second;
      st.wait();

      get_subblock_out(rk);
      detail::get_sub_matrix<rank>(Lloc,subblock_out) = recv[i];
    }   
  } else {
    ::nda::array<long,1> block_intsect_origins(rank); 
    ::nda::array<long,1> block_intsect_ends(rank); 
    bool whether_ranges_overlap;
    
      std::vector<::nda::range> local_ranges(rank,::nda::range(0));
      for(size_t i = 0; i < rank; i++) local_ranges[i] = G.local_range(i);

      whether_ranges_overlap = detail::do_ranges_overlap(local_ranges, Arr_rng);

      std::vector<::nda::range> range_intersect = Arr_rng;
      if(whether_ranges_overlap) range_intersect = detail::range_overlap(local_ranges, Arr_rng);

      for(size_t i = 0; i < rank; i++) {
         block_intsect_origins(i) = whether_ranges_overlap ? range_intersect[i].first() : local_ranges[i].first();
         block_intsect_ends(i) = whether_ranges_overlap ? range_intersect[i].last() : local_ranges[i].last();
      }

    comm->gather_n(&whether_ranges_overlap,1,&whether_ranges_overlap,1,p); 
    comm->gather_n(block_intsect_origins.data(),rank,block_intsect_origins.data(),rank,p); 
    comm->gather_n(block_intsect_ends.data(),rank,block_intsect_ends.data(),rank,p); 

    if(whether_ranges_overlap) {
        std::vector<::nda::range> subblock_loc_indexing(rank,::nda::range(0));
        for( auto r : itertools::range(rank) )
            subblock_loc_indexing[r] = ::nda::range( block_intsect_origins(r)-G.origin()[r], block_intsect_ends(r)-G.origin()[r] );

        auto Gloc = detail::get_sub_matrix<rank>(G.local(), subblock_loc_indexing);

        if(Gloc.is_contiguous()) {
          comm->send_n(Gloc.data(),Gloc.size(),p);
        } else {
         typename  std::decay_t<dArrG_t>::Array_t G_ = Gloc;  
          comm->send_n(G_.data(),G_.size(),p);
        }
    }
  }
  comm->barrier();
}



// Scatter a local array from rank "p" to a distributed array G
template<::nda::MemoryArray ArrL_t, DistributedArray dArrG_t>
void scatter(int p, ArrL_t const* L, dArrG_t&& G) 
requires( get_rank<dArrG_t> == ::nda::get_rank<ArrL_t> )
{
  using local_Array_t = typename std::decay_t<ArrL_t>::regular_type;
  auto comm = G.communicator();
  utils::check( p>=0 and p < comm->size(), "Error: Communicator mismatch." );
  if( comm->rank() == p ) { 
    utils::check( L != nullptr, " Error: Nullptr in math::nda::scatter." );
    utils::check( L->shape() == G.global_shape(), "Error: Size mismatch." );
  }

  constexpr long rank = get_rank<dArrG_t>;
  long mpi_size = comm->size();

  if(mpi_size==1) {
    G.local() = *L;
    return;
  }

  if( comm->rank() == p ) {

    auto Lloc = (*L)();
    ::nda::array<long,3> blocks{mpi_size,2,rank};
    {
      ::nda::array<long,2> local_block{2,rank};  
      std::copy_n(G.origin().data(),rank,local_block.data());
      std::copy_n(G.local_shape().data(),rank,local_block.data()+rank);
      comm->gather_n(local_block.data(),2*rank,blocks.data(),2*rank,p); 
    }

    std::vector<::nda::range> subblock(rank,::nda::range(0));
    // assemble subblock from blocks given an mpi rank
    auto get_subblock = [&] (int q) {
      for( auto r : itertools::range(rank) )
        subblock[r] = ::nda::range( blocks(q,0,r), blocks(q,0,r)+blocks(q,1,r) );
    };

    std::vector<local_Array_t> send; 
    std::vector<std::pair<int,decltype(comm->isend_n(Lloc.data(),1,0))>> stat;
    stat.reserve(mpi_size-1);	
    send.reserve(mpi_size-1);	

    for( auto i : itertools::range(mpi_size) ) {
      get_subblock(i);
      if(i==p) {
	G.local() = detail::get_sub_matrix<rank>(Lloc,subblock);
	continue;
      } else {
        // post receives
        auto Lsub = detail::get_sub_matrix<rank>(Lloc,subblock);
        send.emplace_back( Lsub.shape() );
        send.back() = detail::get_sub_matrix<rank>(Lloc,subblock);
        stat.emplace_back(std::make_pair(i,comm->isend_n(send.back().data(),send.back().size(),i)));
      }
    }    

    // now wait for messages
    for( int i=0; i<stat.size(); ++i ) {
      auto& st = stat[i].second;
      st.wait();
    }   

  } else {
    ::nda::array<long,2> block{2,rank};  
    std::copy_n(G.origin().data(),rank,block.data());
    std::copy_n(G.local_shape().data(),rank,block.data()+rank);
    comm->gather_n(block.data(),2*rank,block.data(),2*rank,p); 

    auto Gloc = G.local();
    if(Gloc.is_contiguous()) {
      comm->receive_n(Gloc.data(),Gloc.size(),p);
    } else {
      typename std::decay_t<dArrG_t>::Array_t G_ = Gloc;  
      comm->receive_n(G_.data(),G_.size(),p);
      Gloc = G_;
    }
  }
  comm->barrier();
}

// generalize to any dimension
template<DistributedArray dArrG_t, ::nda::MemoryArray ArrL_t>
void gather_sub_matrix(int indx, int p, dArrG_t const& G, ArrL_t* L) 
requires( get_rank<dArrG_t> == (::nda::get_rank<ArrL_t>+1) )
{
  using local_Array_t = typename std::decay_t<ArrL_t>::regular_type;
  auto comm = G.communicator();
  utils::check( p>=0 and p < comm->size(), "Error: Communicator mismatch." );
  constexpr long rank = get_rank<dArrG_t>;
  if( comm->rank() == p ) { 
    utils::check( L != nullptr, " Error: Nullptr in math::nda::gather." );
    std::array<long,rank-1> sz = {0};
    bool resz = false;
    for( int i=0; i<rank-1; ++i ) 
    {
      sz[i] = G.global_shape()[i+1];
      if( L->shape()[i] != G.global_shape()[i+1] ) resz = true; 
    } 
    if( resz ) 
      L->resize(sz); 
  }
  long mpi_size = comm->size();

  if(mpi_size==1) {
    *L = G.local()(indx,::nda::ellipsis{});
    return;
  }

  if( comm->rank() == p ) {

    auto Lloc = (*L)();
    ::nda::array<long,3> blocks{mpi_size,2,rank};
    {
      ::nda::array<long,2> local_block{2,rank};  
      std::copy_n(G.origin().data(),rank,local_block.data());
      std::copy_n(G.local_shape().data(),rank,local_block.data()+rank);
      comm->gather_n(local_block.data(),2*rank,blocks.data(),2*rank,p); 
    }

    std::vector<::nda::range> subblock(rank-1,::nda::range(0));
    // assemble subblock from blocks given an mpi rank
    auto get_subblock = [&] (int q) {
      for(int r=1; r<rank; ++r) 
        subblock[r-1] = ::nda::range( blocks(q,0,r), blocks(q,0,r)+blocks(q,1,r) );
    };
    auto in_local_range = [&] (int q) {
      return (indx >= blocks(q,0,0) and indx < blocks(q,0,0)+blocks(q,1,0));
    };

    std::vector<local_Array_t> recv;
    std::vector<std::pair<int,decltype(comm->ireceive_n(Lloc.data(),1,p))>> stat;
    recv.reserve(mpi_size-1);	
    stat.reserve(mpi_size-1);	

    for( auto i : itertools::range(mpi_size) ) {
      if( not in_local_range(i) ) continue;
      get_subblock(i);
      if(i==p) {
	detail::get_sub_matrix<rank-1>(Lloc,subblock) = G.local()(indx-G.origin()[0],::nda::ellipsis{});
	continue;
      } else {
        // post receives
        auto Lsub = detail::get_sub_matrix<rank-1>(Lloc,subblock);
        recv.emplace_back( Lsub.shape() );
        stat.emplace_back(std::make_pair(i,comm->ireceive_n(recv.back().data(),recv.back().size(),i)));
      }
    }    

    // now wait for messages
    for( int i=0; i<stat.size(); ++i ) {
      // MAM: look for a way to process messages as they arrive, rather than just wait in order
      auto& rk = stat[i].first;
      auto& st = stat[i].second;
      st.wait();

      get_subblock(rk);
      detail::get_sub_matrix<rank-1>(Lloc,subblock) = recv[i];
    }   

  } else {
    ::nda::array<long,2> block{2,rank};  
    std::copy_n(G.origin().data(),rank,block.data());
    std::copy_n(G.local_shape().data(),rank,block.data()+rank);
    comm->gather_n(block.data(),2*rank,block.data(),2*rank,p); 

    if(indx >= block(0,0) and indx < block(0,0)+block(1,0)) {
      auto Gloc = G.local()(indx-G.origin()[0],::nda::ellipsis{});
      if(Gloc.is_contiguous()) {
        comm->send_n(Gloc.data(),Gloc.size(),p);
      } else {
        local_Array_t G_ = Gloc;  
        comm->send_n(G_.data(),G_.size(),p);
      }
    }
  }
  comm->barrier();
}


template<MEMORY_SPACE MEM, DistributedArray dArrG_t>
auto all_gather_slow(dArrG_t const& G)
{
  auto comm = G.communicator();
  using value_t = typename std::decay_t<dArrG_t>::value_type;
  constexpr int rank = get_rank<std::decay_t<dArrG_t>>;
  memory::array<MEM,value_t,rank> Z(G.global_shape());
  Z()=0;
  if constexpr (rank==1)
    Z(G.local_range(0)) = G.local();
  else if constexpr (rank == 2)
    Z(G.local_range(0),G.local_range(1)) = G.local();
  else if constexpr (rank == 3)
    Z(G.local_range(0),G.local_range(1),G.local_range(2)) = G.local();
  else if constexpr (rank == 4)
    Z(G.local_range(0),G.local_range(1),G.local_range(2),G.local_range(3)) = G.local();
  else if constexpr (rank == 5)
    Z(G.local_range(0),G.local_range(1),G.local_range(2),G.local_range(3),G.local_range(4)) = G.local();
  comm->all_reduce_in_place_n(Z.data(),Z.size(),std::plus<>{});
  return Z;
}

template<DistributedArray dArrG_t>
void scatter_slow(int p, ::nda::MemoryArray auto const& A, dArrG_t& G) 
{
  using value_t = typename std::decay_t<dArrG_t>::value_type;
  constexpr int rank = get_rank<std::decay_t<dArrG_t>>;
  static_assert(rank == ::nda::get_rank<decltype(A)>, "Rank mismatch");
  auto comm = G.communicator();
  utils::check( p>=0 and p < comm->size(), "Error: Communicator mismatch." );
  ::nda::array<value_t,rank> Z(G.global_shape());
  if( p == comm->rank() ) {
    utils::check( A.shape() == G.global_shape(), "Shape mismatch" );
    Z() = A();
  }
  comm->broadcast_n(Z.data(),Z.size(),p);
  if constexpr (rank == 1)
    G.local() = Z(G.local_range(0));
  else if constexpr (rank == 2)
    G.local() = Z(G.local_range(0),G.local_range(1));
  else if constexpr (rank == 3)
    G.local() = Z(G.local_range(0),G.local_range(1),G.local_range(2));
  else if constexpr (rank == 4)
    G.local() = Z(G.local_range(0),G.local_range(1),G.local_range(2),G.local_range(3));
  else if constexpr (rank == 5)
    G.local() = Z(G.local_range(0),G.local_range(1),G.local_range(2),G.local_range(3),G.local_range(4));

}

/**
 * Replicate a distributed array into a shared-memory array, one full copy per node.
 *
 * Optional `Timer` splits the call into its five phases under the clock names
 * GATHER_SHM_{ZERO,ASSIGN,SKEW,REDUCE,BARRIER}. The internode reduction is the only
 * phase that touches the fabric, and it is spread over the ranks of a node
 * (shared_array::all_reduce_parallel), so attributing the cost between it and the
 * node-local phases is what the breakdown is for.
 */
template<DistributedArray dArr_t, math::shm::SharedArray sArr_t>
void gather_to_shm(const dArr_t &dA, sArr_t &sA, utils::TimerManager* Timer = nullptr)
requires( get_rank<std::decay_t<dArr_t>> == get_rank<std::decay_t<sArr_t>> ) {

  static constexpr int rank = get_rank<std::decay_t<dArr_t>>;
  utils::check(dA.global_shape() == sA.shape(), "Shape mismatch.");

  auto tic = [&](const char* c) { if(Timer) Timer->start(c); };
  auto toc = [&](const char* c) { if(Timer) Timer->stop(c); };

  tic("GATHER_SHM_ZERO");
  sA.set_zero();
  toc("GATHER_SHM_ZERO");
  auto sA_loc = sA.local();

  // Gather at the root node
//  gather(0, dA, &sA_loc);

  tic("GATHER_SHM_ASSIGN");
  std::vector<::nda::range> rng_v(rank,::nda::range(0));
  for(int r=0; r<rank; ++r)
    rng_v[r] = dA.local_range(r);
  ::nda::tensor::assign(dA.local(),detail::get_sub_matrix<rank>(sA_loc,rng_v));
  toc("GATHER_SHM_ASSIGN");

  // The store above is purely local, so this barrier absorbs whatever imbalance
  // the ranks arrived with, keeping upstream skew out of the reduce clock below.
  // It is not needed for correctness: all_reduce_parallel opens and closes with
  // node_sync(), which publishes these stores and matches the collective by node
  // rank. Timed separately rather than added — charging it to the local store
  // reported skew as gather cost.
  tic("GATHER_SHM_SKEW");
  sA.communicator()->barrier();
  toc("GATHER_SHM_SKEW");

  // MAM Note: In some MPI implementations/systems, the first call to a collective can be very
  //           slow (e.g. x100 slower). Not clear why, seems to happen more in shared memory.
  //           If this problem persist, reduce on regular memory and copy to shm locally
  // All_reduce among all nodes, split across the ranks of a node.
  tic("GATHER_SHM_REDUCE");
  sA.all_reduce_parallel();
  toc("GATHER_SHM_REDUCE");

  tic("GATHER_SHM_BARRIER");
  sA.communicator()->barrier();
  toc("GATHER_SHM_BARRIER");
}

template<DistributedArray dArr_t, math::shm::SharedArray sArr_t>
void gather_to_shm_slow(const dArr_t &dA, sArr_t &sA)
requires( get_rank<std::decay_t<dArr_t>> == get_rank<std::decay_t<sArr_t>> ) {

  // Gather at the root node
  auto sA_loc = sA.local();
  gather(0, dA, &sA_loc);
  sA.communicator()->barrier();

  // Broadcast from the root nodes
  // CNY: broadcast is sometimes WAY slower than all_reduce, which is
  //      probably related to a bad choice of the underlying algorithm
  //      chosen by the MPI backend, e.g. OpenMPI.
  sA.broadcast_to_nodes(0);
}

} // math::nda

#endif
