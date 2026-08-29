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
#include "mpi3/shared_communicator.hpp"

#include "configuration.hpp"
#include "IO/AppAbort.hpp"
#include "IO/app_loggers.h"
#include "utilities/proc_grid_partition.hpp"
#include "utilities/tile_partition.hpp"

#include "nda/nda.hpp"
#include "nda/blas.hpp"
#include "nda/linalg.hpp"
#include "numerics/distributed_array/nda.hpp"
#include "numerics/distributed_array/h5.hpp"
#include "utilities/test_common.hpp"
#include "utilities/lr_utils.hpp"
#include "methods/scr_coulomb/scr_coulomb_fourier_t.h"

namespace bdft_tests
{

// Tile count for an N x N operand on a (nx, ny) grid with tiles of at most 16
// elements: the same count on both axes, which is what the old squared_blocks
// flag used to enforce after the fact.
inline long t16(long N, long nx, long ny)
{ return utils::balanced_tile_count(N, std::min(N, std::max(nx,ny)), 16); }

template <typename scalar_type>
void random_matrix( int64_t m, int64_t n, scalar_type* A, int64_t lda )
{   
    for (int64_t j = 0; j < n; ++j) {
        for (int64_t i = 0; i < m; ++i) {
            A[ i + j*lda ] = rand() / double(RAND_MAX);
        }
    }
}

template <typename matrix_type>
void random_matrix( matrix_type& A )
{
    for (int64_t j = 0; j < A.nt(); ++j) {
        for (int64_t i = 0; i < A.mt(); ++i) {
            if (A.tileIsLocal( i, j )) {
                try {
                    auto T = A( i, j );
                    random_matrix( T.mb(), T.nb(), T.data(), T.stride() );
                }
                catch (...) {
                    // ignore missing tiles
                                       }
            }
        }
    }
}

using namespace math::nda;
using namespace math::nda::slate_ops;
template <int Rank> using shape_t = std::array<long, Rank>;

TEST_CASE("dops_tags", "[math]")
{
  using local_Array_t = nda::array<double, 2>;
  auto world = boost::mpi3::environment::get_world_instance();
  auto A =  make_distributed_array<local_Array_t>(world, shape_t<2>{world.size(),1},
                        shape_t<2>{32*world.size(),32},
                        shape_t<2>{t16(32*world.size(),world.size(),1), 0});

  [[maybe_unused]] auto An = normal(A);
  [[maybe_unused]] auto An_ = N(std::move(A));
  [[maybe_unused]] auto At = transpose(A);
  [[maybe_unused]] auto At_ = T(std::move(A));
  [[maybe_unused]] auto Ah = dagger(A);
  [[maybe_unused]] auto Ah_ = H(std::move(A));
}

void test_ls(int64_t m, int64_t n, int64_t nrhs, int64_t nb, 
	  int64_t p0, int64_t q0, int64_t p1, int64_t q1) 
{
  using scalar_type = double;
  int64_t max_mn = std::max( m, n );
  slate::Matrix<scalar_type> A( m, n, nb, p0, q0, MPI_COMM_WORLD );
  slate::Matrix<scalar_type> BX( max_mn, nrhs, nb, p1, q1, MPI_COMM_WORLD );
  A.insertLocalTiles();
  BX.insertLocalTiles();
  auto B = BX;  // == BX.slice( 0, m-1, 0, nrhs-1 );
  auto X = BX.slice( 0, n-1, 0, nrhs-1 );
  random_matrix( A );
  random_matrix( B );
  
  { 
    // solve AX = B, solution in X 
    slate::least_squares_solve( A, BX );  
    slate::gels( A, BX );              // traditional API
  }
    
  random_matrix( A );
  random_matrix( B );
  { 
    auto AH = conj_transpose(A);
    slate::least_squares_solve( AH, BX );  // simplified API
    slate::gels( AH, BX );              // traditional API
  }
}

void test_gemm2(int64_t m, int64_t n, int64_t k, int64_t nb,
          int64_t mb, int64_t pb, 
          int64_t p0, int64_t q0, 
          int64_t p1, int64_t q1, 
	  int64_t p2, int64_t q2)
{ 
  // TODO: failing if m, n not divisible by nb?
  using scalar_type = double;
  slate::Matrix<scalar_type> A( k, m, nb, p0, q0, MPI_COMM_WORLD );
  slate::Matrix<scalar_type> B( k, n, mb, p1, q1, MPI_COMM_WORLD );
  slate::Matrix<scalar_type> C( m, n, pb, p2, q2, MPI_COMM_WORLD );
  A.insertLocalTiles();
  B.insertLocalTiles();
  C.insertLocalTiles();
  random_matrix( A );
  random_matrix( B );
  
  auto AH = conj_transpose ( A );
  slate::multiply(1.0, AH, B, 0.0, C );        
}

void test_gemm(int64_t m, int64_t n, int64_t k, 
	  int64_t nb0, int64_t nb1,
	  int64_t mb0, int64_t mb1,
	  int64_t pb0, int64_t pb1,
          int64_t p0, int64_t q0,
          int64_t p1, int64_t q1,
          int64_t p2, int64_t q2)
{ 
  // TODO: failing if m, n not divisible by nb?
  using scalar_type = double;
  slate::Matrix<scalar_type> A( m, k, nb0, nb1, p0, q0, MPI_COMM_WORLD );
  slate::Matrix<scalar_type> B( k, n, mb0, mb1, p1, q1, MPI_COMM_WORLD );
  slate::Matrix<scalar_type> C( m, n, pb0, pb1, p2, q2, MPI_COMM_WORLD );
  A.insertLocalTiles();
  B.insertLocalTiles();
  C.insertLocalTiles();
  random_matrix( A );
  random_matrix( B );
  
  slate::multiply(1.0, A, B, 0.0, C );
}

/*
TEST_CASE("distributed_pdgemm", "[math]")
{
  
  test_gemm(128,256,64,16,16,16,16,16,16,2,3,2,3,2,3);
  test_gemm(128,256,64,16,16,16,16,16,16,3,2,2,3,2,3);

  test_gemm2(128,256,64,16,16,16,2,3,2,3,2,3);
  test_gemm2(128,256,64,16,16,16,3,2,2,3,2,3);

}
*/

#if defined(ENABLE_DEVICE)
TEST_CASE("cuda_aware_mpi", "[math]")
{
  const long N = 128;
  auto world = boost::mpi3::environment::get_world_instance();
  {
    using local_Array_t = nda::cuarray<double, 2>;
    long nx = utils::find_proc_grid_min_diff(world.size(),N,N);
    long ny = world.size()/nx;
    auto A =  make_distributed_array<local_Array_t>(world, shape_t<2>{nx,ny},
                        shape_t<2>{N,N}, shape_t<2>{t16(N,nx,ny), t16(N,nx,ny)});
    world.broadcast_n(A.local().data(),A.local().size());
    world.barrier();
  }
}
#endif

// slate_ops::determinant: the log-det of the RPA correlation energy
// (methods/GW/thc_rpa.icc).  The reference is serial LAPACK on a replicated copy
// of the same analytic matrix, so it shares nothing with the distributed path.
//
// The matrix is a deterministic pseudo-random complex one, NOT the diagonally
// dominant sin/cos matrix of "distributed_inverse": a diagonally dominant matrix
// pivots trivially, so its permutation parity is +1 and a sign bug is invisible.
// The sweep below asserts the pivot sequence really is nontrivial before it
// trusts the sign comparison.
TEST_CASE("determinant", "[math]") {
  auto world = boost::mpi3::environment::get_world_instance();
  using local_Array_t = memory::array<HOST_MEMORY, ComplexType, 2>;

  auto run = [&](const long N) {
    // replicated on every rank: same seed, same sequence, no communication
    unsigned long seed = 0x9E3779B97F4A7C15ul + 1315423911ul*static_cast<unsigned long>(N);
    auto u01 = [&seed]() {
      seed = seed*6364136223846793005ul + 1442695040888963407ul;
      return double((seed >> 11) & ((1ul<<52)-1))/double(1ul<<52);
    };
    nda::array<ComplexType, 2> A(N, N);
    for (long i = 0; i < N; ++i)
      for (long j = 0; j < N; ++j)
        A(i, j) = ComplexType(2.0*u01() - 1.0, 2.0*u01() - 1.0);

    // reference value
    ComplexType det_ref;
    {
      nda::matrix<ComplexType> Am(A);
      det_ref = nda::determinant_in_place(Am);
    }
    REQUIRE(std::abs(det_ref) > 0.0);

    // the permutation is nontrivial, i.e. the sign is actually under test
    {
      nda::array<ComplexType, 2, nda::F_layout> Af(A);
      nda::array<int, 1> ipiv(N);
      long info = nda::lapack::getrf(Af, ipiv);
      REQUIRE(info == 0);
      long nswap = 0;
      for (long i = 0; i < N; ++i) if (ipiv(i)-1 != i) ++nswap;
      INFO("N = " << N << ": LAPACK swapped " << nswap << " rows");
      REQUIRE(nswap > 0);
    }

    auto check = [&](long np_i, long np_j, long t) {
      if (np_i*np_j != world.size()) return;
      if (N < np_i or N < np_j) return;
      if (t < np_i or t < np_j or t > N) return;

      auto dA = make_distributed_array<local_Array_t>(world, shape_t<2>{np_i, np_j},
                    shape_t<2>{N, N}, shape_t<2>{t, t});
      dA.local() = A(dA.local_range(0), dA.local_range(1));

      auto det = math::nda::slate_ops::determinant(dA);

      auto ratio = det/det_ref;
      app_log(2, "  determinant: N = {}, pgrid = ({}, {}), tiles = {}, "
                 "det/det_ref = ({:.12f}, {:.12f})",
              N, np_i, np_j, t, ratio.real(), ratio.imag());
      INFO("N = " << N << ", pgrid = (" << np_i << ", " << np_j << "), tiles = " << t
           << ", det = " << det << ", ref = " << det_ref);
      // The sign is exact -- assert it as such, with no tolerance.
      CHECK(ratio.real() > 0.0);
      // The magnitude is a product of N factors, so it carries N roundings.
      CHECK(std::abs(std::abs(det)/std::abs(det_ref) - 1.0) < 1e-10);
    };

    const long nx = utils::find_proc_grid_min_diff(world.size(), N, N);
    for (auto [np_i, np_j] : std::array<std::pair<long,long>,4>{{
             {world.size(), 1l}, {1l, world.size()}, {nx, world.size()/nx},
             {world.size()/nx, nx}}}) {
      const long p_max = std::max(np_i, np_j);
      // one tile per rank, two tiles per rank, one element per tile
      for (long t : {p_max, 2*p_max, N}) check(np_i, np_j, t);
    }
  };

  // Small enough that |det| stays well inside double range, ragged over the rank
  // counts ctest uses.
  run(8);
  run(9);
  run(12);
  run(41);
}

TEST_CASE("distributed_ops", "[math]")
{
  const long N = 128;
  auto world = boost::mpi3::environment::get_world_instance();

  {
    using local_Array_t = nda::array<double, 2>;
    long nx = utils::find_proc_grid_min_diff(world.size(),N,N);
    long ny = world.size()/nx;
    auto A =  make_distributed_array<local_Array_t>(world, shape_t<2>{nx,ny},
			shape_t<2>{N,N}, shape_t<2>{t16(N,nx,ny), t16(N,nx,ny)}); 
    auto B =  make_distributed_array<local_Array_t>(world, shape_t<2>{nx,ny},
			shape_t<2>{N,N}, shape_t<2>{t16(N,nx,ny), t16(N,nx,ny)}); 
    auto C =  make_distributed_array<local_Array_t>(world, shape_t<2>{nx,ny},
			shape_t<2>{N,N}, shape_t<2>{t16(N,nx,ny), t16(N,nx,ny)}); 
    random_matrix( A.local().shape()[0], A.local().shape()[1], 
  		   A.local().data(), A.local().indexmap().strides()[1] );
    random_matrix( B.local().shape()[0], B.local().shape()[1], 
 		   B.local().data(), B.local().indexmap().strides()[1] );

    multiply(A,B,C);
    multiply(T(A),B,C);
    multiply(A,T(B),C);
    multiply(H(A),B,C);
    multiply(A,H(B),C);
    multiply(T(A),T(B),C);
    multiply(H(A),T(B),C);
    multiply(T(A),H(B),C);
    multiply(H(A),H(B),C);

  }

  {
    using local_Array_t = nda::array<double, 2, nda::F_layout>;
    long nx = utils::find_proc_grid_min_diff(world.size(),N,N);
    long ny = world.size()/nx;
    auto A =  make_distributed_array<local_Array_t>(world, shape_t<2>{nx,ny},
                        shape_t<2>{N,N}, shape_t<2>{t16(N,nx,ny), t16(N,nx,ny)});  
    auto B =  make_distributed_array<local_Array_t>(world, shape_t<2>{nx,ny},
                        shape_t<2>{N,N}, shape_t<2>{t16(N,nx,ny), t16(N,nx,ny)});  
    for( auto& v: A.local()) v = rand() / double(RAND_MAX);
    for( auto& v: B.local()) v = rand() / double(RAND_MAX);

    lu_solve(A,B);
  }

  {
    using local_Array_t = nda::array<double, 2, nda::F_layout>;
    long nx = utils::find_proc_grid_min_diff(world.size(),N,N);
    long ny = world.size()/nx;
    auto A =  make_distributed_array<local_Array_t>(world, shape_t<2>{nx,ny},
                        shape_t<2>{N,N}, shape_t<2>{t16(N,nx,ny), t16(N,nx,ny)});
    auto B =  make_distributed_array<local_Array_t>(world, shape_t<2>{nx,ny},
                        shape_t<2>{N,N}, shape_t<2>{t16(N,nx,ny), t16(N,nx,ny)});
    for( auto& v: A.local()) v = rand() / double(RAND_MAX);
    for( auto& v: B.local()) v = rand() / double(RAND_MAX);

    least_squares_solve(A,B);
  }

  {
    using local_Array_t = nda::array<double, 3>;
    long nx = utils::find_proc_grid_min_diff(world.size(),N,N);
    long ny = world.size()/nx;
    long t = t16(N,nx,ny);
    auto A =  make_distributed_array<local_Array_t>(world, shape_t<3>{1,nx,ny},
                        shape_t<3>{4,N,N}, {0, t, t});  
    auto B =  make_distributed_array<local_Array_t>(world, shape_t<3>{1,nx,ny},
                        shape_t<3>{4,N,N}, {0, t, t});  
    auto C =  make_distributed_array<local_Array_t>(world, shape_t<3>{1,nx,ny},
                        shape_t<3>{4,N,N}, {0, t, t});  
    auto Aloc = A.local();
    auto Bloc = B.local();
    A.local() = nda::rand(Aloc.shape());
    B.local() = nda::rand(Bloc.shape());
    
    multiply(A,B,C);
    multiply(T(A),B,C);
    multiply(A,T(B),C);
    multiply(H(A),B,C);
    multiply(A,H(B),C);
    multiply(T(A),T(B),C);
    multiply(H(A),T(B),C);
    multiply(T(A),H(B),C);
    multiply(H(A),H(B),C);
  }

  { 
    using local_Array_t = nda::array<double, 3>;
    long nx = utils::find_proc_grid_min_diff(world.size(),N,N);
    long ny = world.size()/nx;
std::cout<<" nx: " <<nx <<std::endl;
    long t = t16(N,nx,ny);
    auto A =  make_distributed_array<local_Array_t>(world, shape_t<3>{nx,ny,1},
                        shape_t<3>{2*nx,N,N}, {0, t, t}); 
    auto B =  make_distributed_array<local_Array_t>(world, shape_t<3>{nx,ny,1},
                        shape_t<3>{2*nx,N,N}, {0, t, t}); 
    auto C =  make_distributed_array<local_Array_t>(world, shape_t<3>{nx,ny,1},
                        shape_t<3>{2*nx,N,N}, {0, t, t}); 
    auto Aloc = A.local();
    auto Bloc = B.local();
    A.local() = nda::rand(Aloc.shape());
    B.local() = nda::rand(Bloc.shape());
    
    multiply(A,B,C);
    multiply(T(A),B,C);
    multiply(A,T(B),C);
    multiply(H(A),B,C);
    multiply(A,H(B),C);
    multiply(T(A),T(B),C);
    multiply(H(A),T(B),C);
    multiply(T(A),H(B),C);
    multiply(H(A),H(B),C);
  }


#if defined(ENABLE_DEVICE)

  {
    using local_Array_t = nda::cuarray<double, 2>;
    long nx = utils::find_proc_grid_min_diff(world.size(),N,N);
    long ny = world.size()/nx;
    auto A =  make_distributed_array<local_Array_t>(world, shape_t<2>{nx,ny},
                        shape_t<2>{N,N}, shape_t<2>{t16(N,nx,ny), t16(N,nx,ny)});
    auto B =  make_distributed_array<local_Array_t>(world, shape_t<2>{nx,ny},
                        shape_t<2>{N,N}, shape_t<2>{t16(N,nx,ny), t16(N,nx,ny)});
    auto C =  make_distributed_array<local_Array_t>(world, shape_t<2>{nx,ny},
                        shape_t<2>{N,N}, shape_t<2>{t16(N,nx,ny), t16(N,nx,ny)});
    {
      A.local() = utils::make_random<double>(A.local_shape()[0],A.local_shape()[1]);;
      B.local() = utils::make_random<double>(B.local_shape()[0],B.local_shape()[1]);;
    }

    multiply(A,B,C);
    multiply(T(A),B,C);
    multiply(A,T(B),C);
    multiply(H(A),B,C);
    multiply(A,H(B),C);
    multiply(T(A),T(B),C);
    multiply(H(A),T(B),C);
    multiply(T(A),H(B),C);
    multiply(H(A),H(B),C);

  }

  {
    //using local_Array_t = nda::cuarray<double, 2, nda::F_layout>;
    using local_Array_t = memory::unified_array<double, 2, nda::F_layout>;
    long nx = utils::find_proc_grid_min_diff(world.size(),N,N);
    long ny = world.size()/nx;
    auto A =  make_distributed_array<local_Array_t>(world, shape_t<2>{nx,ny},
                        shape_t<2>{N,N}, shape_t<2>{t16(N,nx,ny), t16(N,nx,ny)});
    auto B =  make_distributed_array<local_Array_t>(world, shape_t<2>{nx,ny},
                        shape_t<2>{N,N}, shape_t<2>{t16(N,nx,ny), t16(N,nx,ny)});
    nda::array<double, 2, nda::F_layout> a(A.local_shape());
    for( auto& v: a) v = rand() / double(RAND_MAX);
    A.local()=a;
    for( auto& v: a) v = rand() / double(RAND_MAX);
    B.local()=a;

    lu_solve(A,B);
  }

  {
    //using local_Array_t = nda::cuarray<double, 2, nda::F_layout>;
    using local_Array_t = memory::unified_array<double, 2, nda::F_layout>;
    long nx = utils::find_proc_grid_min_diff(world.size(),N,N);
    long ny = world.size()/nx;
    auto A =  make_distributed_array<local_Array_t>(world, shape_t<2>{nx,ny},
                        shape_t<2>{N,N}, shape_t<2>{t16(N,nx,ny), t16(N,nx,ny)});
    auto B =  make_distributed_array<local_Array_t>(world, shape_t<2>{nx,ny},
                        shape_t<2>{N,N}, shape_t<2>{t16(N,nx,ny), t16(N,nx,ny)});
    nda::array<double, 2, nda::F_layout> a(A.local_shape());
    for( auto& v: a) v = rand() / double(RAND_MAX);
    A.local()=a;
    for( auto& v: a) v = rand() / double(RAND_MAX);
    B.local()=a;

    least_squares_solve(A,B);
  }

#endif

}

// slate_ops::inverse over the processor grids simple_dyson can pick: the band
// axis is split over the run's rank count. getrf/getri need A.mt() == A.nt(), so
// the two axes must carry the SAME tile count — with equal extents that gives
// identical tile boundaries, which is the whole point of storing the count. The
// count itself is balanced_tile_count(N, max(np_i, np_j), 1024), the same recipe
// the q-dist W grid uses (see the "ft_buffer_dist" case).
TEST_CASE("distributed_inverse", "[math]")
{
  auto world = boost::mpi3::environment::get_world_instance();

  auto run = [&](const long N) {
    nda::array<ComplexType, 2> A(N, N);
    for (long i = 0; i < N; ++i)
      for (long j = 0; j < N; ++j)
        A(i, j) = ComplexType(std::sin(0.3*i + 0.7*j + 1.0), std::cos(0.2*i - 0.5*j))
                  + (i == j ? ComplexType(2*N) : ComplexType(0));

    nda::array<ComplexType, 2> Ainv_ref(A);
    {
      nda::matrix_view<ComplexType> Am(Ainv_ref);
      Ainv_ref = nda::inverse(Am);
    }

    auto check = [&](long np_i, long np_j) {
      if (np_i*np_j != world.size()) return;
      if (N < np_i or N < np_j) return;
      // square tile count: identical boundaries on both axes, which is the getri
      // precondition mt == nt
      long t = utils::balanced_tile_count(N, std::max(np_i, np_j), 1024);

      auto dA = make_distributed_array<nda::array<ComplexType, 2>>(world,
                    shape_t<2>{np_i, np_j}, shape_t<2>{N, N}, shape_t<2>{t, t});
      dA.local() = A(dA.local_range(0), dA.local_range(1));

      math::nda::slate_ops::inverse(dA);

      double err = 0.0;
      auto Aloc = dA.local();
      for (auto [i, in] : itertools::enumerate(dA.local_range(0)))
        for (auto [j, jn] : itertools::enumerate(dA.local_range(1)))
          err = std::max(err, std::abs(Aloc(i, j) - Ainv_ref(in, jn)));
      // Identical on every rank after the reduction, so CHECK cannot diverge.
      err = world.all_reduce_value(err, boost::mpi3::max<>{});
      app_log(2, "  inverse: N = {}, pgrid = ({}, {}), tiles = {}, max error = {:.3e}",
              N, np_i, np_j, t, err);
      INFO("N = " << N << ", pgrid = (" << np_i << ", " << np_j << "), tiles = " << t);
      CHECK(err < 1e-10);
    };

    check(world.size(), 1);
    check(1, world.size());
    long nx = utils::find_proc_grid_min_diff(world.size(), N, N);
    check(nx, world.size()/nx);
    // the transposed grid: find_proc_grid_min_diff returns the larger factor
    // first, so this is the only arm with np_i < np_j.
    check(world.size()/nx, nx);
  };

  run(40);   // 2^3 * 5: exact on every grid np = 8 factors into, so no ragged tile
  run(41);   // prime: ragged last local block on both axes at any np > 1
  run(130);  // even, but not divisible by 4, 8 or 12
  run(283);  // prime, and large enough for several tiles per rank
  run(403);  // 13 * 31: no factor below 13
  run(511);  // 7 * 73: exact at np = 7, ragged at 8, 11 and 13
}

// One gemm, one process grid, several tile counts. Nothing else in the suite fixes
// the grid and varies only the tiling, so nothing else can see a tiling bug that a
// single tile count happens to hide (cf. docs/bug_lr_gw_fused_pq_tiling.md).
// The reference is a replicated nda::matmul, independent of the distributed path.
TEST_CASE("multiply_blocking_sweep", "[math]")
{
  auto world = boost::mpi3::environment::get_world_instance();

  auto run = [&](long N) {
    const long np_i = utils::find_proc_grid_min_diff(world.size(), N, N);
    const long np_j = world.size()/np_i;
    if (N < np_i or N < np_j) return;
    const long p_max = std::max(np_i, np_j);

    nda::array<ComplexType, 2> A(N, N), B(N, N);
    for (long i = 0; i < N; ++i)
      for (long j = 0; j < N; ++j) {
        A(i, j) = ComplexType(std::sin(0.31*i + 0.17*j), std::cos(0.11*i - 0.43*j));
        B(i, j) = ComplexType(std::cos(0.23*i - 0.29*j), std::sin(0.37*i + 0.13*j));
      }
    nda::array<ComplexType, 2> Ref = nda::matmul(A, B);
    double nrm = 0.0;
    for (auto v : Ref) nrm = std::max(nrm, std::abs(v));

    // one tile per rank, two tiles per rank, and one element per tile
    for (long b : {p_max, 2*p_max, N}) {
      if (b > N) continue;
      auto dA = make_distributed_array<nda::array<ComplexType, 2>>(world,
                    shape_t<2>{np_i, np_j}, shape_t<2>{N, N}, shape_t<2>{b, b});
      auto dB = make_distributed_array<nda::array<ComplexType, 2>>(world,
                    shape_t<2>{np_i, np_j}, shape_t<2>{N, N}, shape_t<2>{b, b});
      auto dC = make_distributed_array<nda::array<ComplexType, 2>>(world,
                    shape_t<2>{np_i, np_j}, shape_t<2>{N, N}, shape_t<2>{b, b});
      dA.local() = A(dA.local_range(0), dA.local_range(1));
      dB.local() = B(dB.local_range(0), dB.local_range(1));

      math::nda::slate_ops::multiply(dA, dB, dC);

      double err = 0.0;
      auto Cloc = dC.local();
      for (auto [i, in] : itertools::enumerate(dC.local_range(0)))
        for (auto [j, jn] : itertools::enumerate(dC.local_range(1)))
          err = std::max(err, std::abs(Cloc(i, j) - Ref(in, jn)));
      err = world.all_reduce_value(err, boost::mpi3::max<>{});
      app_log(2, "  multiply_blocking_sweep: N = {}, pgrid = ({},{}), tiles = {}, "
                 "stored = {}, max rel error = {:.3e}",
              N, np_i, np_j, b, dA.tile_count()[0], err/nrm);
      INFO("N = " << N << ", pgrid = (" << np_i << ", " << np_j << "), tiles = " << b);
      CHECK(err < 1e-12*nrm);
    }
  };

  run(40);
  run(41);
  run(130);
  run(283);
}

// multiply_blocking_sweep gives A, B and C the same square shape, grid and block
// size, so its three operands cannot disagree and the conformability check inside
// multiply_impl provably never fires there. gemm_tile_conformability drives the
// predicate directly, with hand-written structs. Neither reaches the plumbing between
// them -- which branch multiply_impl selects, in which order it passes the operands,
// and whether it reads the POST-op extents -- and a bug there fails silently, by never
// firing.
//
// So: one M x K times K x N with an independent block size per axis, fed from real
// distributed arrays, asserted numerically.
//
// Choosing those tile counts is the whole subtlety, and it is the reason this test
// exists. Equal counts on a shared axis give equal boundaries -- that is what storing
// a count buys -- so conformability is now a matter of passing the same number twice.
// What is NOT free is the range: the contracted axis K sits on grid axis 1 of A and on
// grid axis 0 of B, so its one count has to satisfy grid <= t on BOTH, i.e. be a
// multiple of max(np_i, np_j). That is the p_max rule, and this is the smallest test
// that exercises it. M is shared by A and C and N by B and C, both on the same grid
// axis within each pair, so those two take their own grid extent.
TEST_CASE("multiply_nonsquare_blocking", "[math]")
{
  auto world = boost::mpi3::environment::get_world_instance();

  const long M = 96, K = 60, N = 40;
  const long np_i = utils::find_proc_grid_min_diff(world.size(), M, N);
  const long np_j = world.size()/np_i;
  if (std::min({M,K,N}) < std::max(np_i,np_j)) return;

  nda::array<ComplexType, 2> A(M,K), B(K,N);
  for (long i = 0; i < M; ++i)
    for (long j = 0; j < K; ++j)
      A(i,j) = ComplexType(std::sin(0.31*i + 0.17*j), std::cos(0.11*i - 0.43*j));
  for (long i = 0; i < K; ++i)
    for (long j = 0; j < N; ++j)
      B(i,j) = ComplexType(std::cos(0.23*i - 0.29*j), std::sin(0.37*i + 0.13*j));
  nda::array<ComplexType, 2> Ref = nda::matmul(A, B);
  double nrm = 0.0;
  for (auto v : Ref) nrm = std::max(nrm, std::abs(v));

  // one tile per rank on each axis, then two per rank, then four
  for (long k : {1l, 2l, 4l}) {
    const long tM = k*np_i;
    const long tN = k*np_j;
    const long tK = k*std::max(np_i,np_j);        // the shared axis: p_max, not its own
    if (tM > M or tK > K or tN > N) continue;     // the shape cannot carry that many

    auto dA = make_distributed_array<nda::array<ComplexType,2>>(world,
                  shape_t<2>{np_i,np_j}, shape_t<2>{M,K}, shape_t<2>{tM,tK});
    auto dB = make_distributed_array<nda::array<ComplexType,2>>(world,
                  shape_t<2>{np_i,np_j}, shape_t<2>{K,N}, shape_t<2>{tK,tN});
    auto dC = make_distributed_array<nda::array<ComplexType,2>>(world,
                  shape_t<2>{np_i,np_j}, shape_t<2>{M,N}, shape_t<2>{tM,tN});

    // stored verbatim, so the operands share a partition on every contracted axis.
    // The factory validates grid <= t <= extent rather than clamping, so a count that
    // survived construction is the count the partition uses -- which is exactly the
    // property the tile-size representation could not offer.
    REQUIRE(dA.tile_count()[1] == dB.tile_count()[0]);
    REQUIRE(dA.tile_count()[0] == dC.tile_count()[0]);
    REQUIRE(dB.tile_count()[1] == dC.tile_count()[1]);

    dA.local() = A(dA.local_range(0), dA.local_range(1));
    dB.local() = B(dB.local_range(0), dB.local_range(1));

    math::nda::slate_ops::multiply(dA, dB, dC);

    double err = 0.0;
    auto Cloc = dC.local();
    for (auto [i, in] : itertools::enumerate(dC.local_range(0)))
      for (auto [j, jn] : itertools::enumerate(dC.local_range(1)))
        err = std::max(err, std::abs(Cloc(i,j) - Ref(in,jn)));
    err = world.all_reduce_value(err, boost::mpi3::max<>{});
    app_log(2, "  multiply_nonsquare_blocking: {}x{} * {}x{}, pgrid = ({},{}), "
               "tiles = ({},{},{}), max rel error = {:.3e}",
            M, K, K, N, np_i, np_j, tM, tK, tN, err/nrm);
    INFO("tiles = (" << tM << "," << tK << "," << tN << ")");
    CHECK(err < 1e-12*nrm);
  }
}

// The conformability predicate behind the utils::check in multiply_impl. It is a
// predicate precisely so that it can be tested: utils::check calls MPI_Abort, so a
// mismatched gemm cannot be run inside a Catch2 test. Extents and tile counts here are
// what slate::gemm receives, i.e. already through the transpose op and the C-order
// row/column swap.
TEST_CASE("gemm_tile_conformability", "[math]")
{
  using math::nda::slate_ops::detail::gemm_operand;
  using math::nda::slate_ops::detail::gemm_tile_mismatch;

  // (100x60) * (60x40) = (100x40), tile counts agreeing on each shared axis
  const gemm_operand A{100,60,4,3}, B{60,40,3,2}, C{100,40,4,2};
  CHECK(gemm_tile_mismatch(A,B,C) == "");
  // one element per tile is conformable too
  CHECK(gemm_tile_mismatch({100,60,100,60},{60,40,60,40},{100,40,100,40}) == "");

  // The silent one: extents agree everywhere, only the contracted axis is tiled
  // differently. slate::gemm runs it and returns a wrong number.
  CHECK(gemm_tile_mismatch(A,{60,40,4,2},C).find("contracted") != std::string::npos);
  CHECK(gemm_tile_mismatch({100,60,4,2},B,C).find("contracted") != std::string::npos);
  // extent mismatch on the contracted axis
  CHECK(gemm_tile_mismatch(A,{59,40,3,2},C).find("contracted") != std::string::npos);
  // the outer axes: A/C rows and B/C columns, tile count and extent
  CHECK(gemm_tile_mismatch(A,B,{100,40,5,2}).find("row") != std::string::npos);
  CHECK(gemm_tile_mismatch(A,B,{101,40,4,2}).find("row") != std::string::npos);
  CHECK(gemm_tile_mismatch(A,B,{100,40,4,3}).find("column") != std::string::npos);
  CHECK(gemm_tile_mismatch(A,B,{100,41,4,2}).find("column") != std::string::npos);

  // the message names the operands it was GIVEN, in the order it was given them.
  // multiply_impl relies on that: it always passes (A,B,C) in CoQui's own orientation,
  // undoing the row/column swap to_slate_view applies to a C-order array, so the axis
  // word and the shape in the message are the ones the caller wrote.
  CHECK(gemm_tile_mismatch(B,A,C,"B","A","C").find("B is 60x40") != std::string::npos);
}

// The factory's distribution, checked as a partition rather than through an
// operation: gather every rank's (origin, local_shape) and verify per axis that
// the local ranges tile [0,N) exactly, that no rank is empty, and that the
// per-rank loads are within one element of each other when every rank owns a
// single tile.
TEST_CASE("factory_partition", "[math]")
{
  auto world = boost::mpi3::environment::get_world_instance();

  auto run = [&](shape_t<2> grid, shape_t<2> shape, shape_t<2> tcount) {
    if (grid[0]*grid[1] != world.size()) return;
    if (shape[0] < grid[0] or shape[1] < grid[1]) return;
    if (tcount[0] < grid[0] or tcount[1] < grid[1]) return;
    if (tcount[0] > shape[0] or tcount[1] > shape[1]) return;
    auto dA = make_distributed_array<nda::array<ComplexType, 2>>(world, grid, shape, tcount);

    std::array<long, 4> mine{dA.origin()[0], dA.local_shape()[0],
                             dA.origin()[1], dA.local_shape()[1]};
    nda::array<long, 2> all_(world.size(), 4);
    world.all_gather_n(mine.data(), 4, all_.data(), 4);

    INFO("grid = (" << grid[0] << "," << grid[1] << "), shape = (" << shape[0]
         << "," << shape[1] << "), tiles = (" << tcount[0] << "," << tcount[1] << ")");
    for (int d = 0; d < 2; ++d) {
      // the distinct local ranges along axis d, sorted by origin
      std::vector<std::pair<long,long>> rng;
      for (int r = 0; r < world.size(); ++r) {
        std::pair<long,long> e{all_(r, 2*d), all_(r, 2*d+1)};
        if (std::find(rng.begin(), rng.end(), e) == rng.end()) rng.push_back(e);
      }
      std::sort(rng.begin(), rng.end());
      REQUIRE(long(rng.size()) == grid[d]);
      long prev = 0, lmin = shape[d]+1, lmax = -1;
      for (auto [o, l] : rng) {
        REQUIRE(o == prev);            // no gap, no overlap
        REQUIRE(l > 0);                // no empty rank
        prev = o + l;
        lmin = std::min(lmin, l);
        lmax = std::max(lmax, l);
      }
      REQUIRE(prev == shape[d]);       // exactly the index space
      // the balanced partition: per-rank loads within ceil(tiles/grid) tiles of
      // each other, and exactly within one element when every rank owns one tile
      REQUIRE(lmax <= ((tcount[d]+grid[d]-1)/grid[d])*((shape[d]+tcount[d]-1)/tcount[d]));
      if (tcount[d] == grid[d]) REQUIRE(lmax - lmin <= 1);
      app_log(2, "  factory_partition: axis {}, N = {}, grid = {}, tiles = {}, "
                 "stored = {}, local extents [{},{}], ideal {}",
              d, shape[d], grid[d], tcount[d], dA.tile_count()[d], lmin, lmax,
              (shape[d] + grid[d] - 1)/grid[d]);
    }
  };

  const long np = world.size();
  const long nx = utils::find_proc_grid_min_diff(np, 1687, 1687);
  for (long N : {40l, 41l, 130l, 283l, 403l, 511l, 1687l}) {
    if (N < np) continue;
    for (auto g : std::array<shape_t<2>,3>{{ {np,1}, {1,np}, {nx,np/nx} }}) {
      if (N < g[0] or N < g[1]) continue;
      const long p_max = std::max(g[0], g[1]);
      for (long t : {utils::balanced_tile_count(N, p_max, 1024), p_max, 2*p_max, N})
        run(g, {N,N}, {t,t});
    }
  }
}

// scr_coulomb_fourier_t::ft_buffer_dist — the q-dist distribution, which on the
// LR path is the one every W(iω) is carried on.
// Three properties are load-bearing on that path: the (P,Q) block is square
// (the C-order branch of multiply_impl issues slate::multiply(a,Bs,As,b,Cs),
// which needs Bs.nt() == As.mt()), the array still stores that square block
// after make_distributed_array's min(tile size, shape/grid) clamp — that clamp is
// what the fused FT branches and SLATE both read — and the two rank-4
// multiplies of lr_dyson_W_in_place are numerically right on the grid.
TEST_CASE("ft_buffer_dist", "[math]")
{
  using methods::solvers::scr_coulomb_fourier_t;
  using grid_t = std::array<long, 4>;

  auto world = boost::mpi3::environment::get_world_instance();
  auto all = nda::range::all;

  // --- pure function: square (P,Q) block, and the LR duplicate agrees ---
  // utils::lr_W_tau_local_dist deliberately duplicates ft_buffer_dist's body (the LR
  // and ground-state distribution helpers stay separate for now). It has to stay
  // value-identical, or the FT stops fusing silently — the FT engine decides
  // whether to fuse by comparing against its OWN ft_buffer_dist. This sweep is the
  // build-time tripwire for that drift; solve_lr_dyson_W's guard is the runtime one.
  // (nproc, nq, nw_half, NP). The last row is the production point of record:
  // BaBiO3 nk8, 768 ranks, nq = 512, THC nIpts = 1687.
  const std::array<grid_t, 10> sweep = {{
      {1, 8, 4, 98},   {8, 8, 18, 98},   {16, 8, 18, 98},  {24, 8, 18, 98},
      {32, 8, 18, 98}, {64, 8, 18, 98},  {24, 8, 18, 41},  {32, 8, 18, 41},
      {96, 512, 36, 1687}, {768, 512, 36, 1687}}};

  for (auto s : sweep) {
    const long nproc = s[0], nq = s[1], nwh = s[2], NP = s[3];
    auto [b_pgrid, b_tcount] =
        scr_coulomb_fourier_t::ft_buffer_dist(nproc, {nwh, nq, NP, NP});
    INFO("nproc = " << nproc << ", nq = " << nq << ", nw_half = " << nwh
         << ", NP = " << NP << ", pgrid = (" << b_pgrid[0] << "," << b_pgrid[1]
         << "," << b_pgrid[2] << "," << b_pgrid[3] << ")");
    CHECK(b_tcount[2] == b_tcount[3]);
    CHECK(b_tcount[2] >= std::max(b_pgrid[2], b_pgrid[3]));   // no empty rank
    CHECK(b_tcount[2] <= NP);
    CHECK((utils::lr_W_tau_local_dist(nproc, nwh, nq, NP) ==
           scr_coulomb_fourier_t::ft_buffer_dist(nproc, {nwh, nq, NP, NP})));
  }

  // The production point, spelled out: the P-split grid whose SUMMA was measured.
  // The 1024 max tile size is inactive at NP = 1687 over max(3,1) = 3 ranks, so the
  // tile count is 3 -- one tile per P rank, of 563/562/562 elements, instead of the
  // floor-division 562/562/563 that dumped the remainder on the last rank.
  {
    auto [pg, bs] = scr_coulomb_fourier_t::ft_buffer_dist(768, {36, 512, 1687, 1687});
    CHECK((pg == grid_t{1, 256, 3, 1}));
    CHECK((bs == grid_t{0, 0, 3, 3}));
    CHECK(utils::tile_extent(1687, 3, 0) == 563);
    CHECK(utils::tile_extent(1687, 3, 1) == 562);
    CHECK(utils::tile_extent(1687, 3, 2) == 562);
    CHECK((utils::lr_W_tau_local_dist(768, 36, 512, 1687) == std::make_pair(pg, bs)));
  }

  // --- the stored tile count, and the two Dyson multiplies on the grid ---
  const long nproc = world.size();

  auto check_grid = [&](long nq, long NP) {
    const long nwh = 2;
    auto [pg, bs] = scr_coulomb_fourier_t::ft_buffer_dist(nproc, {nwh, nq, NP, NP});
    if (nwh < pg[0] or nq < pg[1] or NP < pg[2] or NP < pg[3]) return;
    INFO("nproc = " << nproc << ", nq = " << nq << ", NP = " << NP
         << ", pgrid = (" << pg[0] << "," << pg[1] << "," << pg[2] << ","
         << pg[3] << "), tiles = (" << bs[2] << "," << bs[3] << ")");

    const shape_t<4> gshape{nwh, nq, NP, NP};
    using local_Array_t = nda::array<ComplexType, 4>;
    auto dPi  = make_distributed_array<local_Array_t>(world, pg, gshape, bs);
    auto dW1  = make_distributed_array<local_Array_t>(world, pg, gshape, bs);
    auto dW2  = make_distributed_array<local_Array_t>(world, pg, gshape, bs);
    auto dTmp = make_distributed_array<local_Array_t>(world, pg, gshape, bs);

    CHECK(dPi.tile_count()[2] == dPi.tile_count()[3]);
    CHECK((dPi.tile_count()[2] == bs[2] and dPi.tile_count()[3] == bs[3]));

    // Replicated operands; the reference below is formed from them rank-locally,
    // so it is independent of the distributed path under test.
    auto val = [NP](long o, long w, long q, long i, long j) {
      double x = 0.31*double(i) + 0.17*double(j) + 0.7*double(w)
               + 1.3*double(q) + double(o);
      return ComplexType(std::sin(x), std::cos(0.5*x)) / double(NP)
             + (i == j ? ComplexType(1.0) : ComplexType(0.0));
    };
    nda::array<ComplexType, 4> Pi(gshape), W1(gshape), W2(gshape), Ref(gshape);
    for (long w = 0; w < nwh; ++w)
      for (long q = 0; q < nq; ++q)
        for (long i = 0; i < NP; ++i)
          for (long j = 0; j < NP; ++j) {
            Pi(w, q, i, j) = val(0, w, q, i, j);
            W1(w, q, i, j) = val(1, w, q, i, j);
            W2(w, q, i, j) = val(2, w, q, i, j);
          }

    // ΔW = W(q+Q) · ΔΠ · W(q): the two multiplies of lr_dyson_W_in_place, in
    // the same order.
    for (long w = 0; w < nwh; ++w)
      for (long q = 0; q < nq; ++q) {
        nda::array<ComplexType, 2> a = W2(w, q, all, all);
        nda::array<ComplexType, 2> b = Pi(w, q, all, all);
        nda::array<ComplexType, 2> c = W1(w, q, all, all);
        Ref(w, q, all, all) = nda::matmul(nda::matmul(a, b), c);
      }

    auto copy_in = [](auto& d, nda::array<ComplexType, 4> const& g) {
      auto loc = d.local();
      for (auto [i0, n0] : itertools::enumerate(d.local_range(0)))
        for (auto [i1, n1] : itertools::enumerate(d.local_range(1)))
          for (auto [i2, n2] : itertools::enumerate(d.local_range(2)))
            for (auto [i3, n3] : itertools::enumerate(d.local_range(3)))
              loc(i0, i1, i2, i3) = g(n0, n1, n2, n3);
    };
    copy_in(dPi, Pi);
    copy_in(dW1, W1);
    copy_in(dW2, W2);

    math::nda::slate_ops::multiply(dW2, dPi, dTmp);
    math::nda::slate_ops::multiply(dTmp, dW1, dPi);

    double err = 0.0, nrm = 0.0;
    auto loc = dPi.local();
    for (auto [i0, n0] : itertools::enumerate(dPi.local_range(0)))
      for (auto [i1, n1] : itertools::enumerate(dPi.local_range(1)))
        for (auto [i2, n2] : itertools::enumerate(dPi.local_range(2)))
          for (auto [i3, n3] : itertools::enumerate(dPi.local_range(3))) {
            err = std::max(err, std::abs(loc(i0, i1, i2, i3) - Ref(n0, n1, n2, n3)));
            nrm = std::max(nrm, std::abs(Ref(n0, n1, n2, n3)));
          }
    err = world.all_reduce_value(err, boost::mpi3::max<>{});
    nrm = world.all_reduce_value(nrm, boost::mpi3::max<>{});
    app_log(2, "  ft_buffer_dist: nq = {}, NP = {}, pgrid = ({},{},{},{}), "
               "tiles = ({},{}), max rel error = {:.3e}",
            nq, NP, pg[0], pg[1], pg[2], pg[3],
            dPi.tile_count()[2], dPi.tile_count()[3], err/nrm);
    CHECK(err < 1e-12 * nrm);
  };

  // nq >= nproc: every rank owns one q, (P,Q) is local and multiply_impl takes
  // its rank-local gemm short circuit.
  check_grid(nproc, 41);
  // nq = 2: (P,Q) gets nproc/2 ranks, so this is a real SLATE SUMMA from 4 ranks
  // up (at nproc = 2 it is still the rank-local short circuit). NP = 41 does not
  // divide the P/Q rank counts, so the tiles are ragged.
  check_grid(2, 40);
  check_grid(2, 41);
  // nq = 1: the whole rank count goes into (P,Q) — the widest SUMMA available,
  // and the one arm guaranteed to be a real SUMMA at any nproc > 1.
  check_grid(1, 41);
}

/*
TEST_CASE("test_solve","[math]")
{
  auto world = boost::mpi3::environment::get_world_instance();
  nda::array<ComplexType,2> A,B;

  {
    h5::file fh5("ls_solve.h5",'r');
    nda::h5_read(fh5, "A", A);
    nda::h5_read(fh5, "B", B);
  }
  int N = A.shape(0);
  int M = B.shape(1);
  utils::check(A.shape(1)==N,"Size mismatch.");
  utils::check(B.shape(0)==N,"Size mismatch.");

  if(world.root()) {
    nda::array<ComplexType,2> A_ = A;
    nda::array<ComplexType,2> B_ = B;
    nda::array<ComplexType,1> work(4*N); 
    nda::array<double,1> rwork(4*N); 
    nda::array<int,1> ipiv(N); 
    int info;
    double anorm = nda::lapack::f77::lange('I',N,N,A_.data(),N,rwork.data());
    app_log(0," I-norm: {}",anorm);
    double rcond;
    nda::lapack::f77::getrf(N,N,A_.data(),N,ipiv.data(),info);
    app_log(0," getrf - info:{}",info);
    nda::lapack::f77::gecon('I',N,A_.data(),N,anorm,rcond,work.data(),rwork.data(),info);
    app_log(0," gecon - info:{}, cond: {}",info,rcond);

    A_=A;
    ::nda::array<double, 1> C(std::min(A_.shape()[0],A_.shape()[1]));
    int rank(0);
    info = ::nda::lapack::gelss(A_,B_,C,-1,rank);
    app_log(0," H(C)*C matrix in LS solve: dims:{}, rank:{}",A_.shape()[0],rank);
    utils::check( info==0, "Problems with gelss solve. ");
    nda::array<ComplexType,2> C_ = B;
    nda::blas::gemm(ComplexType(1.0),A,B_,ComplexType(0.0),C_);
    double err=0.0;
    app_log(0,"B(0,0): {}, ~B(0,0):{}",B(0,0),C_(0,0));
    for(int i=0; i<N; ++i) 
      for(int j=0; j<M; ++j)
        err += std::abs(B(i,j)-C_(i,j)); 
    app_log(0," zgelss error: {}",err);
  }

  std::array<long, 2> bsz = { N/world.size(), N};

  // using slate::lu_solve following thc.icc
  {
    auto dA =  make_distributed_array<nda::array<ComplexType,2>>(world, 
			shape_t<2>{world.size(),1},
                        shape_t<2>{N,N}, 
			shape_t<2>{bsz[0],bsz[0]});
    auto dB =  make_distributed_array<nda::array<ComplexType,2>>(world, 
			shape_t<2>{world.size(),1},
                        shape_t<2>{N,M}, 
			bsz);
    auto dC =  make_distributed_array<nda::array<ComplexType,2>>(world, 
			shape_t<2>{world.size(),1},
                        shape_t<2>{N,M}, 
			bsz);
    dA.local() = A( dA.local_range(0), dA.local_range(1) );
    dB.local() = B( dB.local_range(0), dB.local_range(1) );

    for( auto& v: dA.local() ) v = std::conj(v);
    auto As = math::nda::detail::to_slate_view<true>(dA);
    auto Bts = math::nda::detail::to_slate_view<true>(math::nda::transpose(dB));
    slate::lu_solve(As,Bts);

    // dA = A
    dA.local() = A( dA.local_range(0), dA.local_range(1) );

    math::nda::slate_ops::multiply(dA,dB,dC);
    double err=0.0;
    if(world.root()) app_log(0,"B(0,0): {}, ~B(0,0):{}",B(0,0),dC.local()(0,0));
    auto Bloc = dB.local();
    auto Cloc = dC.local();
    for( auto [i,in] : itertools::enumerate(dB.local_range(0)) )
      for( auto [j,jn] : itertools::enumerate(dB.local_range(1)) ) 
        err += std::abs(B(in,jn)-Cloc(i,j));
    err = world.reduce_value(err,std::plus<>{});
    app_log(0," zgelss error: {} ",err);
  } 

  // using slate::least_squares_solve(As,Bs);
  {
    auto dA =  make_distributed_array<nda::array<ComplexType,2>>(world,
                        shape_t<2>{world.size(),1},
                        shape_t<2>{N,N},
                        shape_t<2>{bsz[0],bsz[0]});
    auto dB =  make_distributed_array<nda::array<ComplexType,2>>(world,
                        shape_t<2>{world.size(),1},
                        shape_t<2>{N,M},
                        bsz);
    auto dC =  make_distributed_array<nda::array<ComplexType,2>>(world,
                        shape_t<2>{world.size(),1},
                        shape_t<2>{N,M},
                        bsz);
    dA.local() = A( dA.local_range(0), dA.local_range(1) );
    dB.local() = B( dB.local_range(0), dB.local_range(1) );

    auto As = math::nda::detail::to_slate_view<true>(dA);
    auto Bts = math::nda::detail::to_slate_view<true>(math::nda::transpose(dB));
    slate::lu_solve(As,Bts);

    // dA = A
    dA.local() = A( dA.local_range(0), dA.local_range(1) );

    math::nda::slate_ops::multiply(dA,dB,dC);
    double err=0.0,err1=0.0;
    if(world.root()) app_log(0,"B(0,0): {}, ~B(0,0):{}",B(0,0),dC.local()(0,0));
    auto Bloc = dB.local();
    auto Cloc = dC.local();
    for( auto [i,in] : itertools::enumerate(dB.local_range(0)) )
      for( auto [j,jn] : itertools::enumerate(dB.local_range(1)) ) {
        err += std::abs(Bloc(i,j)-Cloc(i,j));
        err1 += std::abs(B(in,jn)-Cloc(i,j));
      }
    err = world.reduce_value(err,std::plus<>{});
    err1 = world.reduce_value(err1,std::plus<>{});
    app_log(0," zgelss error: {} (global index: {})",err,err1);
  }

}
*/


} // bdft_tests
