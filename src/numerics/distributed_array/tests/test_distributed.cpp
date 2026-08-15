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


#undef NDEBUG

#include "catch2/catch.hpp"

#include "mpi3/environment.hpp"
#include "mpi3/communicator.hpp"

#include "configuration.hpp"
#include "IO/AppAbort.hpp"
#include "IO/app_loggers.h"
#include "utilities/proc_grid_partition.hpp"

#include "nda/nda.hpp"
#include "numerics/distributed_array/nda.hpp"
#include "numerics/distributed_array/h5.hpp"
#include "utilities/test_common.hpp"

namespace bdft_tests
{

using namespace math::nda;
template <int Rank> using shape_t = std::array<long, Rank>;

TEST_CASE("distributed_nda", "[math]")
{
  auto world = boost::mpi3::environment::get_world_instance();
  long nx = utils::find_proc_grid_min_diff(world.size(),100,100);
  long ny = world.size()/nx;
  long ix = world.rank()/ny;
  long iy = world.rank()%ny;
  {
    using larray = nda::array<double,3>; 
    using darray = distributed_array<larray,decltype(world)>;
   
    darray A(std::addressof(world),{1,nx,ny}, 
		   {10, 2*nx,2*ny},	// global 
		   {10, 2, 2},		// local
		   {0, 2*ix, 2*iy},     // origin
		   {1, 1, 1});	        // block size
    A.local()=1.0;

    REQUIRE( A.origin() == shape_t<3>{0, 2*ix, 2*iy} );
    REQUIRE( A.global_shape() == shape_t<3>{10, 2*nx, 2*ny} );
    REQUIRE( A.local_shape() == shape_t<3>{10, 2, 2} );
    REQUIRE( A.grid() == shape_t<3>{1, nx, ny} );
    REQUIRE( *A.communicator() == world );

    // copy/move constructors
    {
      auto B{A};
      REQUIRE( A == B );
      auto C{std::move(B)};
      REQUIRE( A == C );
    }

    // copy/move assignment
    {
      auto B{A}; 
      auto C{A}; 
      B = A;
      REQUIRE( A == B );
      C = std::move(B);
      REQUIRE( A == C );
    } 

    darray B(std::addressof(world),{1,nx,ny},
                   {20, 3*nx,3*ny},     // global 
                   {0, 3, 3},           // local
                   {0, 3*ix, 3*iy},     // origin
		   {1, 1, 1});	        // block size
    B.local() = 2.0;
    {
      auto C{B};
      C = A;
      REQUIRE( A == C );
    }
  }

  // view
  {
    using larray = nda::array<double,3>; 
    using darray = distributed_array_view<larray,decltype(world)>;

    larray L(10, 2, 2);
    L() = 0.0;
    
    darray A(std::addressof(world),{1,nx,ny}, 
                   {10, 2*nx,2*ny},     // global 
                   {0, 2*ix, 2*iy},     // origin
		   {1, 1, 1},	        // block size
		   L);
    
    REQUIRE( A.origin() == shape_t<3>{0, 2*ix, 2*iy} );
    REQUIRE( A.global_shape() == shape_t<3>{10, 2*nx, 2*ny} );
    REQUIRE( A.local_shape() == shape_t<3>{10, 2, 2} );
    REQUIRE( A.grid() == shape_t<3>{1, nx, ny} );
    REQUIRE( *A.communicator() == world );
    REQUIRE( A.local() == L );
    
    // copy/move constructors
    { 
      darray B{A}; 
      REQUIRE( A == B );
      darray C{std::move(B)};
      REQUIRE( A == C );
    }
    
    // copy/move assignment
    { 
      darray B{A};
      darray C{A};
      B = A;
      REQUIRE( A == B );
      C = std::move(B);
      REQUIRE( A == C );
    }
    
    larray L2(20, 3, 3);
    L2() = 1.0;
    darray B(std::addressof(world),{1,nx,ny}, 
                   {20, 3*nx,3*ny},     // global 
                   {0, 3*ix, 3*iy},     // origin
		   {1, 1, 1},	        // block size
		   L2);
    { 
      darray C{A};
      C.rebind(B);
      REQUIRE( C == B );
      C.rebind(A);
      REQUIRE( C == A );
    }    
  
  }
}

TEST_CASE("redistribute_nda", "[math]")
{
  auto world = boost::mpi3::environment::get_world_instance();
  long size = world.size();

  {
    using larray = nda::array<double,2>;
    auto A = make_distributed_array<larray>(world,{size,1},{10*size+7,21*size+11});
    auto B = make_distributed_array<larray>(world,{1,size},{10*size+7,21*size+11});
    auto origin = A.origin();
    auto gshape = A.global_shape();

    auto Aloc = A.local();
    for( int i=0; i<Aloc.shape(0); ++i )  
      for( int j=0; j<Aloc.shape(1); ++j )
        Aloc(i,j) = (origin[0]+i)*gshape[1] + (origin[1]+j);  

    redistribute(A,B); 
    
    auto Bloc = B.local();
    origin = B.origin();
    for( int i=0; i<Bloc.shape(0); ++i )  
      for( int j=0; j<Bloc.shape(1); ++j )
        REQUIRE( Bloc(i,j) == double((origin[0]+i)*gshape[1] + (origin[1]+j)) );  
  }
  { 
    using larray = nda::array<double,2>;
    auto A = make_distributed_array<larray>(world,{1,size},{9*size+13,3*size+7});
    auto B = make_distributed_array<larray>(world,{size,1},{9*size+13,3*size+7});
    auto origin = A.origin();
    auto gshape = A.global_shape();
    
    auto Aloc = A.local();
    for( int i=0; i<Aloc.shape(0); ++i )  
      for( int j=0; j<Aloc.shape(1); ++j )
        Aloc(i,j) = (origin[0]+i)*gshape[1] + (origin[1]+j);
    
    redistribute(A,B);
    
    auto Bloc = B.local();
    origin = B.origin();
    for( int i=0; i<Bloc.shape(0); ++i )  
      for( int j=0; j<Bloc.shape(1); ++j )
        REQUIRE( Bloc(i,j) == double((origin[0]+i)*gshape[1] + (origin[1]+j)) );
  }
  { 
    using larray = nda::array<double,2>;
    long nx = 1;
    for(int i=2; i<size/2; ++i) if( size%i == 0 ) nx=i; 
    auto A = make_distributed_array<larray>(world,{nx,size/nx},{10*size+7,21*size+11});
    auto B = make_distributed_array<larray>(world,{size/nx,nx},{10*size+7,21*size+11});
    auto origin = A.origin();
    auto gshape = A.global_shape();
    
    auto Aloc = A.local();
    for( int i=0; i<Aloc.shape(0); ++i )  
      for( int j=0; j<Aloc.shape(1); ++j )
        Aloc(i,j) = (origin[0]+i)*gshape[1] + (origin[1]+j);
    
    redistribute(A,B);
    
    auto Bloc = B.local();
    origin = B.origin();
    for( int i=0; i<Bloc.shape(0); ++i )  
      for( int j=0; j<Bloc.shape(1); ++j )
        REQUIRE( Bloc(i,j) == double((origin[0]+i)*gshape[1] + (origin[1]+j)) );
  }
  {
    using larray = nda::array<double,3>;
    auto A = make_distributed_array<larray>(world,{size,1,1},{10*size+7,21*size+11,8*size+13});
    auto B = make_distributed_array<larray>(world,{1,size,1},{10*size+7,21*size+11,8*size+13});
    auto origin = A.origin();
    auto gshape = A.global_shape();

    auto Aloc = A.local();
    for( int i=0; i<Aloc.shape(0); ++i )  
      for( int j=0; j<Aloc.shape(1); ++j )
        for( int k=0; k<Aloc.shape(2); ++k )
          Aloc(i,j,k) = (origin[0]+i)*gshape[1]*gshape[2] + (origin[1]+j)*gshape[2] + (origin[2]+k);  

    redistribute(A,B); 
    
    auto Bloc = B.local();
    origin = B.origin();
    for( int i=0; i<Bloc.shape(0); ++i )  
      for( int j=0; j<Bloc.shape(1); ++j )
        for( int k=0; k<Bloc.shape(2); ++k )
          REQUIRE( Bloc(i,j,k) == double((origin[0]+i)*gshape[1]*gshape[2] + (origin[1]+j)*gshape[2] + (origin[2]+k)));
  }
  { 
    using larray = nda::array<double,3>;
    auto A = make_distributed_array<larray>(world,{1,1,size},{10*size+7,21*size+11,8*size+13});
    auto B = make_distributed_array<larray>(world,{1,size,1},{10*size+7,21*size+11,8*size+13});
    auto origin = A.origin();
    auto gshape = A.global_shape();
    
    auto Aloc = A.local();
    for( int i=0; i<Aloc.shape(0); ++i )
      for( int j=0; j<Aloc.shape(1); ++j )
        for( int k=0; k<Aloc.shape(2); ++k )
          Aloc(i,j,k) = (origin[0]+i)*gshape[1]*gshape[2] + (origin[1]+j)*gshape[2] + (origin[2]+k);

    redistribute(A,B);

    auto Bloc = B.local();
    origin = B.origin();
    for( int i=0; i<Bloc.shape(0); ++i )
      for( int j=0; j<Bloc.shape(1); ++j )
        for( int k=0; k<Bloc.shape(2); ++k )
          REQUIRE( Bloc(i,j,k) == double((origin[0]+i)*gshape[1]*gshape[2] + (origin[1]+j)*gshape[2] + (origin[2]+k)));
  }
  { 
    using larray = nda::array<double,3>;
    auto A = make_distributed_array<larray>(world,{1,size,1},{10*size+7,21*size+11,8*size+13});
    auto B = make_distributed_array<larray>(world,{size,1,1},{10*size+7,21*size+11,8*size+13});
    auto origin = A.origin();
    auto gshape = A.global_shape();
        
    auto Aloc = A.local();
    for( int i=0; i<Aloc.shape(0); ++i )
      for( int j=0; j<Aloc.shape(1); ++j )
        for( int k=0; k<Aloc.shape(2); ++k )
          Aloc(i,j,k) = (origin[0]+i)*gshape[1]*gshape[2] + (origin[1]+j)*gshape[2] + (origin[2]+k);

    redistribute(A,B);

    auto Bloc = B.local();
    origin = B.origin();
    for( int i=0; i<Bloc.shape(0); ++i )
      for( int j=0; j<Bloc.shape(1); ++j )
        for( int k=0; k<Bloc.shape(2); ++k )
          REQUIRE( Bloc(i,j,k) == double((origin[0]+i)*gshape[1]*gshape[2] + (origin[1]+j)*gshape[2] + (origin[2]+k)));
  }
  { 
    using larray = nda::array<double,3>;
    long nx = 1;
    for(int i=2; i<size/2; ++i) if( size%i == 0 ) nx=i; 
    auto A = make_distributed_array<larray>(world,{nx,1,size/nx},{10*size+7,21*size+11,8*size+13});
    auto B = make_distributed_array<larray>(world,{size/nx,nx,1},{10*size+7,21*size+11,8*size+13});
    auto origin = A.origin();
    auto gshape = A.global_shape();

    auto Aloc = A.local();
    for( int i=0; i<Aloc.shape(0); ++i )
      for( int j=0; j<Aloc.shape(1); ++j )
        for( int k=0; k<Aloc.shape(2); ++k )
          Aloc(i,j,k) = (origin[0]+i)*gshape[1]*gshape[2] + (origin[1]+j)*gshape[2] + (origin[2]+k);

    redistribute(A,B);

    auto Bloc = B.local();
    origin = B.origin();
    for( int i=0; i<Bloc.shape(0); ++i )
      for( int j=0; j<Bloc.shape(1); ++j )
        for( int k=0; k<Bloc.shape(2); ++k )
          REQUIRE( Bloc(i,j,k) == double((origin[0]+i)*gshape[1]*gshape[2] + (origin[1]+j)*gshape[2] + (origin[2]+k)));
  }
  {
    using larray = nda::array<double,3>;
    long nx = 1;
    for(int i=2; i<size/2; ++i) if( size%i == 0 ) nx=i;
    auto A = make_distributed_array<larray>(world,{nx,1,size/nx},{10*size+7,21*size+11,8*size+13});
    auto origin = A.origin();
    auto gshape = A.global_shape();

    auto Aloc = A.local();
    for( int i=0; i<Aloc.shape(0); ++i )
      for( int j=0; j<Aloc.shape(1); ++j )
        for( int k=0; k<Aloc.shape(2); ++k )
          Aloc(i,j,k) = (origin[0]+i)*gshape[1]*gshape[2] + (origin[1]+j)*gshape[2] + (origin[2]+k);

    nda::array<double,3>* L = new nda::array<double,3>(A.global_shape());

    math::nda::gather(0,A,L);
    
    if(world.rank() == 0) {
      for( int i=0; i<L->shape(0); ++i )
        for( int j=0; j<L->shape(1); ++j )
          for( int k=0; k<L->shape(2); ++k )
            REQUIRE( (*L)(i,j,k) == double(i*gshape[1]*gshape[2] + j*gshape[2] + k)); 
    } 

    A.local() = 0.0;
    math::nda::scatter(0,L,A);

    for( int i=0; i<Aloc.shape(0); ++i )
      for( int j=0; j<Aloc.shape(1); ++j )
        for( int k=0; k<Aloc.shape(2); ++k )
          REQUIRE( Aloc(i,j,k) == double((origin[0]+i)*gshape[1]*gshape[2] + (origin[1]+j)*gshape[2] + (origin[2]+k)));

    if(world.size() > 1) {
    
      math::nda::gather(1,A,L);
      if(world.rank() == 0) {
        for( int i=0; i<L->shape(0); ++i )
          for( int j=0; j<L->shape(1); ++j )
            for( int k=0; k<L->shape(2); ++k )
              REQUIRE( (*L)(i,j,k) == double(i*gshape[1]*gshape[2] + j*gshape[2] + k));                                 
      }
      
      A.local() = 0.0;
      math::nda::scatter(1,L,A);

      for( int i=0; i<Aloc.shape(0); ++i )
        for( int j=0; j<Aloc.shape(1); ++j )
          for( int k=0; k<Aloc.shape(2); ++k )
            REQUIRE( Aloc(i,j,k) == double((origin[0]+i)*gshape[1]*gshape[2] + (origin[1]+j)*gshape[2] + (origin[2]+k)));       

    }

  }
  {
    using larray = nda::array<double,3>;
    long nx = 1;
    for(int i=2; i<size/2; ++i) if( size%i == 0 ) nx=i;
    auto A = make_distributed_array<larray>(world,{nx,1,size/nx},{10*size+7,21*size+11,8*size+13});
    auto origin = A.origin();
    auto gshape = A.global_shape();

    auto Aloc = A.local();
    for( int i=0; i<Aloc.shape(0); ++i )
      for( int j=0; j<Aloc.shape(1); ++j )
        for( int k=0; k<Aloc.shape(2); ++k )
          Aloc(i,j,k) = (origin[0]+i)*gshape[1]*gshape[2] + (origin[1]+j)*gshape[2] + (origin[2]+k);

    nda::array<double,2>* L = new nda::array<double,2>(A.global_shape()[1],A.global_shape()[2]);

    math::nda::gather_sub_matrix(3,0,A,L);

    if(world.rank() == 0) {
      for( int j=0; j<L->shape(0); ++j )
        for( int k=0; k<L->shape(1); ++k )
          REQUIRE( (*L)(j,k) == double(3*gshape[1]*gshape[2] + j*gshape[2] + k));
    }

    math::nda::gather_sub_matrix(10*size+6,0,A,L);
    
    if(world.rank() == 0) {
      for( int j=0; j<L->shape(0); ++j )
        for( int k=0; k<L->shape(1); ++k )
          REQUIRE( (*L)(j,k) == double((10*size+6)*gshape[1]*gshape[2] + j*gshape[2] + k));
    }

    if(world.size()>1) {

      math::nda::gather_sub_matrix(3,1,A,L);
    
      if(world.rank() == 1) {
        for( int j=0; j<L->shape(0); ++j )
          for( int k=0; k<L->shape(1); ++k )
            REQUIRE( (*L)(j,k) == double(3*gshape[1]*gshape[2] + j*gshape[2] + k));
      }

      math::nda::gather_sub_matrix(10*size+6,1,A,L);
   
      if(world.rank() == 1) {
        for( int j=0; j<L->shape(0); ++j )
          for( int k=0; k<L->shape(1); ++k )
            REQUIRE( (*L)(j,k) == double((10*size+6)*gshape[1]*gshape[2] + j*gshape[2] + k));
      }

    }
  }
  {
    using larray = nda::array<double,3>;
    long nx = 1;
    for(int i=2; i<size/2; ++i) if( size%i == 0 ) nx=i;
    auto A = make_distributed_array<larray>(world,{nx,1,size/nx},{10*size+7,21*size+11,8*size+13});
    auto origin = A.origin();
    auto gshape = A.global_shape();

    auto Aloc = A.local();
    for( int i=0; i<Aloc.shape(0); ++i )
      for( int j=0; j<Aloc.shape(1); ++j )
        for( int k=0; k<Aloc.shape(2); ++k )
          Aloc(i,j,k) = (origin[0]+i)*gshape[1]*gshape[2] + (origin[1]+j)*gshape[2] + (origin[2]+k);

    {
        auto I_rng = nda::range(3,10);
        auto J_rng = nda::range(10,30);
        auto K_rng = nda::range(1,20);
        
        nda::array<double,3>* L = new nda::array<double,3>(I_rng.size(),J_rng.size(),K_rng.size());
        
        math::nda::gather_ranged(0,A,L,{I_rng, J_rng, K_rng});
        
        if(world.rank() == 0) {
          for( int i=0; i<L->shape(0); ++i )
            for( int j=0; j<L->shape(1); ++j )
              for( int k=0; k<L->shape(2); ++k )
                REQUIRE( (*L)(i,j,k) == (I_rng.first()+i)*gshape[1]*gshape[2] + (J_rng.first()+j)*gshape[2] + (K_rng.first()+k));
        }
        delete L;
    }
    {
        auto I_rng = nda::range(3*size,4*size+5);
        auto J_rng = nda::range(10*size,18*size+8);
        auto K_rng = nda::range(3*size+2,7*size+13);
        
        nda::array<double,3>* L = new nda::array<double,3>(I_rng.size(),J_rng.size(),K_rng.size());
        
        math::nda::gather_ranged(0,A,L,{I_rng, J_rng, K_rng});
        
        if(world.rank() == 0) {
          for( int i=0; i<L->shape(0); ++i )
            for( int j=0; j<L->shape(1); ++j )
              for( int k=0; k<L->shape(2); ++k )
                REQUIRE( (*L)(i,j,k) == (I_rng.first()+i)*gshape[1]*gshape[2] + (J_rng.first()+j)*gshape[2] + (K_rng.first()+k));
        }
        delete L;
    }


    if(world.size()>1){
        auto I_rng = nda::range(3,10);
        auto J_rng = nda::range(10,30);
        auto K_rng = nda::range(1,20);
        
        nda::array<double,3>* L = new nda::array<double,3>(I_rng.size(),J_rng.size(),K_rng.size());
        
        math::nda::gather_ranged(1,A,L,{I_rng, J_rng, K_rng});
        
        if(world.rank() == 1) {
          for( int i=0; i<L->shape(0); ++i )
            for( int j=0; j<L->shape(1); ++j )
              for( int k=0; k<L->shape(2); ++k ) {
                REQUIRE( (*L)(i,j,k) == (I_rng.first()+i)*gshape[1]*gshape[2] + (J_rng.first()+j)*gshape[2] + (K_rng.first()+k));
              }
        }
        delete L;
    }

    if(world.size()>1){
        auto I_rng = nda::range(3*size,4*size+5);
        auto J_rng = nda::range(10*size,18*size+8);
        auto K_rng = nda::range(3*size+2,7*size+13);
        
        nda::array<double,3>* L = new nda::array<double,3>(I_rng.size(),J_rng.size(),K_rng.size());
        
        math::nda::gather_ranged(1,A,L,{I_rng, J_rng, K_rng});
        
        if(world.rank() == 1) {
          for( int i=0; i<L->shape(0); ++i )
            for( int j=0; j<L->shape(1); ++j )
              for( int k=0; k<L->shape(2); ++k )
                REQUIRE( (*L)(i,j,k) == (I_rng.first()+i)*gshape[1]*gshape[2] + (J_rng.first()+j)*gshape[2] + (K_rng.first()+k));
        }
        delete L;
    }

  }
}

/**
 * redistribute_alltoallv chunks its pack/unpack staging buffers to a per-rank
 * byte budget. Chunking changes only the blocking of the exchange -- every
 * element keeps its destination and its position -- so the result must be
 * bit-identical to the unchunked transfer for any nchunk.
 *
 * Cases: a ragged global shape (so the block boundaries do not divide the rank
 * count), a block size of 1 on the split axis, and caps from "one element per
 * chunk" (nchunk saturates at comm.size()) up to "no chunking at all".
 *
 * Needs >= 2 ranks: at one rank redistribute_alltoallv takes the serial branch
 * and never reaches the chunking at all, so a silent early return here would
 * report a pass that exercised nothing. Run with CTEST_NPROC >= 2.
 */
TEST_CASE("redistribute_chunked_equivalence", "[math]")
{
  auto world = boost::mpi3::environment::get_world_instance();
  const long size = world.size();
  if (size < 2) {
    WARN("redistribute_chunked_equivalence skipped: needs >= 2 ranks, got "
         << size << " -- the chunked exchange is NOT covered by this run");
    return;
  }

  using larray = nda::array<ComplexType, 3>;
  // Ragged in every axis, so no chunk boundary lands where a block does.
  shape_t<3> gshape = {7*size + 3, 5, 3*size + 1};

  auto fill = [&](auto& A) {
    auto Aloc = A.local();
    auto o = A.origin();
    for (long i = 0; i < Aloc.shape(0); ++i)
      for (long j = 0; j < Aloc.shape(1); ++j)
        for (long k = 0; k < Aloc.shape(2); ++k) {
          double v = double((o[0]+i)*gshape[1]*gshape[2] + (o[1]+j)*gshape[2] + (o[2]+k));
          // An irrational-ish scaling, so a wrongly placed element cannot
          // coincide with the right one.
          Aloc(i,j,k) = ComplexType(v * 0.3141592653589793, -v * 0.2718281828459045);
        }
  };

  auto A = make_distributed_array<larray>(world, {size,1,1}, gshape);
  fill(A);

  // Reference: no chunking (a cap far above any local block).
  auto B_ref = make_distributed_array<larray>(world, {1,1,size}, gshape);
  redistribute_alltoallv(A, B_ref, ComplexType(1.0), ComplexType(0.0),
                         size_t(1) << 40);

  for (size_t cap : {size_t(1), size_t(16), size_t(1024), size_t(1) << 20}) {
    auto B = make_distributed_array<larray>(world, {1,1,size}, gshape);
    redistribute_alltoallv(A, B, ComplexType(1.0), ComplexType(0.0), cap);
    auto Bloc = B.local();
    auto Rloc = B_ref.local();
    double err = 0.0;
    for (long i = 0; i < Bloc.shape(0); ++i)
      for (long j = 0; j < Bloc.shape(1); ++j)
        for (long k = 0; k < Bloc.shape(2); ++k)
          err = std::max(err, std::abs(Bloc(i,j,k) - Rloc(i,j,k)));
    err = world.all_reduce_value(err, boost::mpi3::max<>{});
    REQUIRE(err == 0.0);  // pure data movement: bit-identical, not "close"
  }

  // The reverse direction, with a block size of 1 on the split axis.
  auto C = make_distributed_array<larray>(world, {1,1,size}, gshape, {1,1,1});
  fill(C);
  auto D_ref = make_distributed_array<larray>(world, {size,1,1}, gshape, {1,1,1});
  redistribute_alltoallv(C, D_ref, ComplexType(1.0), ComplexType(0.0), size_t(1) << 40);
  auto D = make_distributed_array<larray>(world, {size,1,1}, gshape, {1,1,1});
  redistribute_alltoallv(C, D, ComplexType(1.0), ComplexType(0.0), size_t(8));
  {
    auto Dloc = D.local();
    auto Rloc = D_ref.local();
    double err = 0.0;
    for (long i = 0; i < Dloc.shape(0); ++i)
      for (long j = 0; j < Dloc.shape(1); ++j)
        for (long k = 0; k < Dloc.shape(2); ++k)
          err = std::max(err, std::abs(Dloc(i,j,k) - Rloc(i,j,k)));
    err = world.all_reduce_value(err, boost::mpi3::max<>{});
    REQUIRE(err == 0.0);
  }
}

/**
 * The chunked exchange must be a legal MPI_Alltoallv in every round: whatever
 * rank i sends to j in a round, j must post a matching receive from i in that
 * same round (MPI requires sendcount_i[j] == recvcount_j[i]).
 *
 * This is checked on the schedule itself, not by moving data, because moving
 * data does not reliably catch a violation. An unmatched send under the eager
 * protocol is simply buffered, and since the per-peer counts are the same either
 * way and the collective's internal tags repeat across calls, a later round's
 * receive matches it out of order and the result still comes out bit-identical.
 * A wrong schedule therefore passes redistribute_chunked_equivalence on one node
 * and deadlocks over a fabric at scale -- which is exactly what happened. Only
 * the invariant separates the two.
 *
 * Pure arithmetic on redistribute_schedule, so it runs at any rank count.
 */
TEST_CASE("redistribute_round_schedule_is_alltoallv_legal", "[math]")
{
  for (long np : {2L, 3L, 8L, 12L, 97L, 768L}) {
    std::vector<long> chunks{1, 2, 3, 7, 8, np/2, np};
    for (long nchunk : chunks) {
      if (nchunk < 1 or nchunk > np) continue;
      math::nda::redistribute_schedule sched(np, nchunk);

      // send_round(i,j): the round in which i sends to j.
      // recv_round(i,j): the round in which j receives from i.
      nda::array<long,2> send_round(np,np), recv_round(np,np);
      send_round() = -1;
      recv_round() = -1;
      for (long me = 0; me < np; ++me)
        for (long o = 0; o < np; ++o) {
          send_round(me, sched.send_peer(me,o)) = sched.round_of(o);
          recv_round(sched.recv_peer(me,o), me) = sched.round_of(o);
        }

      long unscheduled = 0, mismatched = 0, out_of_range = 0;
      for (long i = 0; i < np; ++i)
        for (long j = 0; j < np; ++j) {
          if (send_round(i,j) < 0 or recv_round(i,j) < 0) ++unscheduled;
          else if (send_round(i,j) != recv_round(i,j)) ++mismatched;
          if (send_round(i,j) >= nchunk) ++out_of_range;
        }
      INFO("np=" << np << " nchunk=" << nchunk);
      REQUIRE(unscheduled == 0);   // every pair is exchanged exactly once
      REQUIRE(mismatched == 0);    // and in the same round at both ends
      REQUIRE(out_of_range == 0);  // within the advertised number of rounds
    }
  }
}

} // bdft_tests
