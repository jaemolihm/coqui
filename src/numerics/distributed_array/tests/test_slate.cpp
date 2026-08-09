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

#include "nda/nda.hpp"
#include "nda/blas.hpp"
#include "nda/linalg.hpp"
#include "numerics/distributed_array/nda.hpp"
#include "numerics/distributed_array/h5.hpp"
#include "utilities/test_common.hpp"

namespace bdft_tests
{

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
                        shape_t<2>{32*world.size(),32}, {16, 16}, true);

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
                        shape_t<2>{N,N}, {16, 16}, true);
    world.broadcast_n(A.local().data(),A.local().size());
    world.barrier();
  }
}
#endif

TEST_CASE("determinant", "[math]") {
  const long N = 128;
  auto world = boost::mpi3::environment::get_world_instance();

  nda::matrix<double> A(N, N);
  A() = 1.1;
  auto detA_ref = ::nda::determinant_in_place(A);
  app_log(2, "detA_ref = {}", detA_ref);

  long nx = utils::find_proc_grid_min_diff(world.size(), N, N);
  long ny = world.size() / nx;
  using local_Array_t = memory::array<HOST_MEMORY, double, 2>;
  auto dA = make_distributed_array<local_Array_t>(world, shape_t<2>{nx, ny},
                                                 shape_t<2>{N, N}, {16,16}, true);

  auto i_rng = dA.local_range(0);
  auto j_rng = dA.local_range(1);
  auto A_loc = dA.local();
  A_loc = A(i_rng, j_rng);

  auto [Ni_loc, Nj_loc] = dA.local_shape();
  auto [i_origin, j_origin] = dA.origin();
  std::vector<std::pair<long,long> > diag_idx;
  for (long ii = 0; ii < Ni_loc; ++ii) {
    long i = ii + i_origin;
    for (size_t jj = 0; jj < Nj_loc; ++jj) {
      long j = jj + j_origin;
      if (i == j) diag_idx.push_back({ii, jj});
    }
  }

  app_log(2, "pgrid = ({}, {})", dA.grid()[0], dA.grid()[1]);
  app_log(2, "bsize = ({}, {})", dA.block_size()[0], dA.block_size()[1]);
  [[maybe_unused]] auto detA = math::nda::slate_ops::determinant(dA, diag_idx);
  app_log(2, "detA = {}", detA);

  utils::VALUE_EQUAL(detA, detA_ref);
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
			shape_t<2>{N,N}, {16, 16}, true); 
    auto B =  make_distributed_array<local_Array_t>(world, shape_t<2>{nx,ny},
			shape_t<2>{N,N}, {16, 16}, true); 
    auto C =  make_distributed_array<local_Array_t>(world, shape_t<2>{nx,ny},
			shape_t<2>{N,N}, {16, 16}, true); 
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
                        shape_t<2>{N,N}, {16, 16}, true);  
    auto B =  make_distributed_array<local_Array_t>(world, shape_t<2>{nx,ny},
                        shape_t<2>{N,N}, {16, 16}, true);  
    for( auto& v: A.local()) v = rand() / double(RAND_MAX);
    for( auto& v: B.local()) v = rand() / double(RAND_MAX);

    lu_solve(A,B);
  }

  {
    using local_Array_t = nda::array<double, 2, nda::F_layout>;
    long nx = utils::find_proc_grid_min_diff(world.size(),N,N);
    long ny = world.size()/nx;
    auto A =  make_distributed_array<local_Array_t>(world, shape_t<2>{nx,ny},
                        shape_t<2>{N,N}, {16, 16}, true);
    auto B =  make_distributed_array<local_Array_t>(world, shape_t<2>{nx,ny},
                        shape_t<2>{N,N}, {16, 16}, true);
    for( auto& v: A.local()) v = rand() / double(RAND_MAX);
    for( auto& v: B.local()) v = rand() / double(RAND_MAX);

    least_squares_solve(A,B);
  }

  {
    using local_Array_t = nda::array<double, 3>;
    long nx = utils::find_proc_grid_min_diff(world.size(),N,N);
    long ny = world.size()/nx;
    long bz = std::min(16l,std::min(N/nx,N/ny));
    auto A =  make_distributed_array<local_Array_t>(world, shape_t<3>{1,nx,ny},
                        shape_t<3>{4,N,N}, {1, bz, bz});  
    auto B =  make_distributed_array<local_Array_t>(world, shape_t<3>{1,nx,ny},
                        shape_t<3>{4,N,N}, {1, bz, bz});  
    auto C =  make_distributed_array<local_Array_t>(world, shape_t<3>{1,nx,ny},
                        shape_t<3>{4,N,N}, {1, bz, bz});  
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
    long bz = std::min(16l,N/ny);
    auto A =  make_distributed_array<local_Array_t>(world, shape_t<3>{nx,ny,1},
                        shape_t<3>{2*nx,N,N}, {1, bz, bz}); 
    auto B =  make_distributed_array<local_Array_t>(world, shape_t<3>{nx,ny,1},
                        shape_t<3>{2*nx,N,N}, {1, bz, bz}); 
    auto C =  make_distributed_array<local_Array_t>(world, shape_t<3>{nx,ny,1},
                        shape_t<3>{2*nx,N,N}, {1, bz, bz}); 
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
                        shape_t<2>{N,N}, {16, 16}, true);
    auto B =  make_distributed_array<local_Array_t>(world, shape_t<2>{nx,ny},
                        shape_t<2>{N,N}, {16, 16}, true);
    auto C =  make_distributed_array<local_Array_t>(world, shape_t<2>{nx,ny},
                        shape_t<2>{N,N}, {16, 16}, true);
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
                        shape_t<2>{N,N}, {16, 16}, true);
    auto B =  make_distributed_array<local_Array_t>(world, shape_t<2>{nx,ny},
                        shape_t<2>{N,N}, {16, 16}, true);
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
                        shape_t<2>{N,N}, {16, 16}, true);
    auto B =  make_distributed_array<local_Array_t>(world, shape_t<2>{nx,ny},
                        shape_t<2>{N,N}, {16, 16}, true);
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
// axis may end up split over an arbitrary rank count, with a block size as
// small as nbnd/np_i.
TEST_CASE("distributed_inverse", "[math]")
{
  auto world = boost::mpi3::environment::get_world_instance();
  const long N = 40;

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
    long bsize = std::min({1024l, N/np_i, N/np_j});
    if (bsize < 1) return;

    auto dA = make_distributed_array<nda::array<ComplexType, 2>>(world,
                  shape_t<2>{np_i, np_j}, shape_t<2>{N, N}, shape_t<2>{bsize, bsize});
    dA.local() = A(dA.local_range(0), dA.local_range(1));

    math::nda::slate_ops::inverse(dA);

    double err = 0.0;
    auto Aloc = dA.local();
    for (auto [i, in] : itertools::enumerate(dA.local_range(0)))
      for (auto [j, jn] : itertools::enumerate(dA.local_range(1)))
        err = std::max(err, std::abs(Aloc(i, j) - Ainv_ref(in, jn)));
    // Identical on every rank after the reduction, so CHECK cannot diverge.
    err = world.all_reduce_value(err, boost::mpi3::max<>{});
    app_log(2, "  inverse: pgrid = ({}, {}), bsize = {}, max error = {:.3e}",
            np_i, np_j, bsize, err);
    INFO("pgrid = (" << np_i << ", " << np_j << "), bsize = " << bsize);
    CHECK(err < 1e-10);
  };

  check(world.size(), 1);
  check(1, world.size());
  long nx = utils::find_proc_grid_min_diff(world.size(), N, N);
  check(nx, world.size()/nx);
}

/**
 * Value check of the rank-4 slate_ops::multiply, i.e. a batch of (P,Q) gemms
 * over two leading axes — the shape the LR W Dyson (ΔW = W·ΔΠ·W) runs on. Both
 * branches of multiply_impl are covered: a grid that only splits the leading
 * axes leaves one whole matrix per rank (local nda::blas::gemm), while a (P,Q)
 * grid dispatches a SLATE gemm on the subgrid.
 */
TEST_CASE("distributed_multiply_rank4", "[math]")
{
  auto world = boost::mpi3::environment::get_world_instance();
  const long nw = 12, nq = 2, N = 24;

  nda::array<ComplexType, 4> A(nw, nq, N, N), B(nw, nq, N, N), C(nw, nq, N, N);
  auto fill = [](auto&& X, double s) {
    long n = 0;
    for (auto& v : X) { v = ComplexType(std::sin(s*(n+1)), std::cos(s*(n+2))); ++n; }
  };
  fill(A, 0.031);
  fill(B, 0.017);
  for (long iw = 0; iw < nw; ++iw)
    for (long iq = 0; iq < nq; ++iq)
      nda::blas::gemm(ComplexType(1.0), A(iw, iq, nda::ellipsis{}),
                      B(iw, iq, nda::ellipsis{}), ComplexType(0.0),
                      C(iw, iq, nda::ellipsis{}));

  auto check = [&](long np_w, long np_q, long np_P, long np_Q) {
    if (np_w*np_q*np_P*np_Q != world.size()) return;
    if (np_w > nw or np_q > nq) return;
    long bs = std::min({8l, N/np_P, N/np_Q});
    if (bs < 1) return;

    auto mk = [&]() {
      return make_distributed_array<nda::array<ComplexType, 4>>(world,
                 shape_t<4>{np_w, np_q, np_P, np_Q}, shape_t<4>{nw, nq, N, N},
                 shape_t<4>{1, 1, bs, bs});
    };
    auto dA = mk(); auto dB = mk(); auto dC = mk();
    auto slice = [](auto const& G, auto const& d) {
      return G(d.local_range(0), d.local_range(1), d.local_range(2), d.local_range(3));
    };
    dA.local() = slice(A, dA);
    dB.local() = slice(B, dB);
    dC.local() = ComplexType(0.0);  // beta = 0 still reads C on some paths

    math::nda::slate_ops::multiply(dA, dB, dC);

    double err = 0.0;
    auto Cloc = dC.local();
    for (auto [i0, w] : itertools::enumerate(dC.local_range(0)))
      for (auto [i1, q] : itertools::enumerate(dC.local_range(1)))
        for (auto [i2, p] : itertools::enumerate(dC.local_range(2)))
          for (auto [i3, r] : itertools::enumerate(dC.local_range(3)))
            err = std::max(err, std::abs(Cloc(i0, i1, i2, i3) - C(w, q, p, r)));
    // Identical on every rank after the reduction, so CHECK cannot diverge.
    err = world.all_reduce_value(err, boost::mpi3::max<>{});
    app_log(2, "  multiply(rank-4): pgrid = ({}, {}, {}, {}), bsize = {}, max error = {:.3e}",
            np_w, np_q, np_P, np_Q, bs, err);
    INFO("pgrid = (" << np_w << ", " << np_q << ", " << np_P << ", " << np_Q << ")");
    CHECK(err < 1e-10);
  };

  const long np = world.size();
  check(np, 1, 1, 1);                 // leading-axis only: local gemm per tile
  check(1, 1, np, 1);                 // 1-D SLATE grid over P
  long nx = utils::find_proc_grid_min_diff(np, N, N);
  check(1, 1, nx, np/nx);             // 2-D SLATE grid
  if (np % 2 == 0) check(2, 1, np/2, 1);  // batched over w and split over P
}

/**
 * The LR W Dyson (lr_scr_coulomb_t::lr_dyson_W_in_place) does not call
 * slate_ops::multiply: to keep the intermediate down to one (P,Q) block it
 * inlines that routine's C-order tile loop around a rank-2 scratch. This checks
 * the inlined form — same split colour, same tile order — against the generic
 * routine with a full-size distributed temporary, which is what it replaced.
 *
 * The grids exercised deliberately split a batch axis *and* (P,Q) at once, the
 * production shape: with only one tile per rank a wrong colour rule still gives
 * the right answer.
 */
TEST_CASE("lr_dyson_W_inlined_tile_loop", "[math]")
{
  decltype(nda::range::all) all;
  auto world = boost::mpi3::environment::get_world_instance();
  const long nw = 4, nq = 2, N = 24;

  auto check = [&](long np_w, long np_q, long np_P, long np_Q) {
    if (np_w*np_q*np_P*np_Q != world.size()) return;
    if (np_w > nw or np_q > nq) return;
    long bs = std::min({8l, N/np_P, N/np_Q});
    if (bs < 1) return;

    auto mk = [&]() {
      return make_distributed_array<nda::array<ComplexType, 4>>(world,
                 shape_t<4>{np_w, np_q, np_P, np_Q}, shape_t<4>{nw, nq, N, N},
                 shape_t<4>{1, 1, bs, bs});
    };
    auto dW = mk(), dWq = mk(), dPi = mk();
    auto fill = [](auto&& d, double s) {
      long n = 0;
      for (auto& v : d.local()) { v = ComplexType(std::sin(s*(n+1)), std::cos(s*(n+2))); ++n; }
    };
    fill(dW, 0.031); fill(dWq, 0.017); fill(dPi, 0.023);

    // Reference: the two-call form with a full-size distributed temporary.
    auto dPi_ref = mk(), dTmp = mk();
    dPi_ref.local() = dPi.local();
    dTmp.local() = ComplexType(0.0);
    math::nda::slate_ops::multiply(dWq, dPi_ref, dTmp);
    math::nda::slate_ops::multiply(dTmp, dW, dPi_ref);

    // Under test: one split, a rank-2 scratch, two multiply_impl per tile.
    auto gshape = dPi.global_shape();
    auto origin = dPi.origin();
    auto lshape = dPi.local_shape();
    long color = origin[0] + gshape[0] * origin[1];
    auto pq_comm = world.split(color, world.rank());
    nda::array<ComplexType, 2> Tmp_buf(lshape[2], lshape[3]);
    auto Tmp_2D = Tmp_buf(all, all);
    auto pq_view = [&](auto&& A_2D, auto const& darr) {
      auto g = darr.grid(); auto gs = darr.global_shape();
      auto o = darr.origin(); auto b = darr.block_size();
      return math::nda::distributed_array_view<std::decay_t<decltype(A_2D)>, decltype(pq_comm)>(
          std::addressof(pq_comm),
          std::array<long, 2>{g[2], g[3]}, std::array<long, 2>{gs[2], gs[3]},
          std::array<long, 2>{o[2], o[3]}, std::array<long, 2>{b[2], b[3]}, A_2D);
    };
    auto dTmp_PQ = pq_view(Tmp_2D, dPi);
    auto Wq_loc = dWq.local(), W_loc = dW.local(), Pi_loc = dPi.local();
    for (long iw = 0; iw < lshape[0]; ++iw)
      for (long iq = 0; iq < lshape[1]; ++iq) {
        auto dWq_PQ = pq_view(Wq_loc(iw, iq, all, all), dWq);
        auto dW_PQ  = pq_view(W_loc(iw, iq, all, all), dW);
        auto dPi_PQ = pq_view(Pi_loc(iw, iq, all, all), dPi);
        math::nda::slate_ops::detail::multiply_impl(
            ComplexType(1.0), dWq_PQ, dPi_PQ, ComplexType(0.0), dTmp_PQ);
        math::nda::slate_ops::detail::multiply_impl(
            ComplexType(1.0), dTmp_PQ, dW_PQ, ComplexType(0.0), dPi_PQ);
      }

    double err = 0.0, ref = 0.0;
    auto a = dPi.local(), b = dPi_ref.local();
    for (long i = 0; i < a.size(); ++i) {
      err = std::max(err, std::abs(a.data()[i] - b.data()[i]));
      ref = std::max(ref, std::abs(b.data()[i]));
    }
    err = world.all_reduce_value(err, boost::mpi3::max<>{});
    ref = world.all_reduce_value(ref, boost::mpi3::max<>{});
    app_log(2, "  lr_dyson_W inlined: pgrid = ({}, {}, {}, {}), bsize = {}, max |diff| = {:.3e}"
               " (max |ref| = {:.3e})", np_w, np_q, np_P, np_Q, bs, err, ref);
    INFO("pgrid = (" << np_w << ", " << np_q << ", " << np_P << ", " << np_Q << ")");
    REQUIRE(ref > 0.0);
    // Same operands, same tile order, only the destination differs: bit-exact.
    CHECK(err == 0.0);
  };

  const long np = world.size();
  check(np, 1, 1, 1);                          // batch only: local gemm path
  check(1, 1, np, 1);                          // (P,Q) only: SLATE path
  if (np % 2 == 0) check(2, 1, np/2, 1);       // batch AND P split  <- production shape
  if (np % 4 == 0) check(2, 2, np/4, 1);       // both batch axes AND P split
  if (np % 8 == 0) check(2, 2, np/8, 2);       // both batch axes AND 2-D (P,Q)
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

/**
 * Strong-scaling benchmark for the rank-4 slate_ops::multiply used by the LR W
 * Dyson (ΔW = W·ΔΠ·W). Fixed global problem (nw tiles of Np x Np); the same
 * rank count is spent either on the leading (w) axis — every rank then owns
 * whole Np x Np matrices and multiply_impl takes the local nda::blas::gemm
 * path — or on the (P, Q) axes, which forces a real SLATE gemm on a p x q
 * subgrid. Per-rank FLOPs are identical in both, so the ratio is exactly
 * SLATE's efficiency relative to a local zgemm.
 *
 * The block size follows utils::lr_W_proc_grid, so a poor ratio may mean the
 * production block size is wrong rather than that SLATE is slow.
 *
 * Opt-in: the leading '.' in the tag is what keeps it out of the default ctest
 * run; the env gate makes it a no-op if it is selected anyway. Allocates
 * ~0.8 GB/rank and runs for seconds.
 *   COQUI_BENCH_SLATE=1 mpirun -n 4 ./test_math_distributed_nda "[.bench]"
 * Override the problem with COQUI_BENCH_NP / COQUI_BENCH_NW.
 */
TEST_CASE("slate_multiply_bench", "[.bench]")
{
  auto world = boost::mpi3::environment::get_world_instance();

  const char* on = std::getenv("COQUI_BENCH_SLATE");
  if (on == nullptr || std::string_view(on) == "0") return;

  auto env_long = [](const char* name, long dflt) {
    const char* v = std::getenv(name);
    return (v == nullptr) ? dflt : std::stol(v);
  };
  const long Np = env_long("COQUI_BENCH_NP", 1160);
  const long nw = env_long("COQUI_BENCH_NW", 12);
  const int  nrep = int(env_long("COQUI_BENCH_NREP", 2));

  // 8 flops per complex madd, one gemm over the whole (nw, Np, Np) problem
  const double gflop = 8.0 * double(nw) * double(Np) * double(Np) * double(Np) * 1e-9;

  app_log(0, "");
  app_log(0, "slate_multiply_bench: Np = {}, nw = {}, world = {}, {:.1f} GFLOP total",
          Np, nw, world.size(), gflop);
  app_log(0, "  {:>10} {:>16} {:>10} {:>10} {:>12}",
          "ranks", "pgrid(w,q,P,Q)", "bsize", "time[s]", "GF/s/rank");

  auto run = [&](long np_w, long np_P, long np_Q) {
    long np = np_w * np_P * np_Q;
    if (np > world.size() || nw % np_w != 0) return;
    long bs = std::min({1024l, Np/np_P, Np/np_Q});
    if (bs < 1) return;

    // Sub-communicator of the first np ranks; the rest idle through this case.
    auto sub = world.split(world.rank() < np ? 0 : 1, world.rank());
    double t = 0.0;
    if (world.rank() < np) {
      auto mk = [&]() {
        return make_distributed_array<nda::array<ComplexType, 4>>(
            sub, shape_t<4>{np_w, 1, np_P, np_Q}, shape_t<4>{nw, 1, Np, Np},
            shape_t<4>{1, 1, bs, bs});
      };
      auto dA = mk();
      auto dB = mk();
      auto dC = mk();
      // Deterministic, non-degenerate fill; values are irrelevant to timing but
      // must not be denormals.
      auto fill = [](auto&& d, double s) {
        auto loc = d.local();
        long n = 0;
        for (auto& v : loc) { v = ComplexType(s*std::sin(0.001*n), std::cos(0.002*n)); ++n; }
      };
      fill(dA, 1.0); fill(dB, 0.5); fill(dC, 0.0);

      math::nda::slate_ops::multiply(dA, dB, dC);  // warmup
      sub.barrier();
      t = 1e30;
      for (int r = 0; r < nrep; ++r) {
        sub.barrier();
        double t0 = MPI_Wtime();
        math::nda::slate_ops::multiply(dA, dB, dC);
        sub.barrier();
        t = std::min(t, MPI_Wtime() - t0);
      }
    }
    world.barrier();
    world.broadcast_n(&t, 1, 0);
    app_log(0, "  {:>10} {:>16} {:>10} {:>10.3f} {:>12.1f}",
            np, fmt::format("({},1,{},{})", np_w, np_P, np_Q), bs, t,
            gflop / t / double(np));
  };

  for (long np : {1l, 2l, 3l, 4l, 6l, 8l}) {
    if (np > world.size()) break;
    run(np, 1, 1);                       // reference: local gemm, w-parallel
    if (np > 1) run(1, np, 1);           // 1-D SLATE grid over P
    long nx = utils::find_proc_grid_min_diff(np, Np, Np);
    if (np > 1 && nx != np) run(1, nx, np/nx);  // 2-D SLATE grid
  }
  app_log(0, "");
}

} // bdft_tests
