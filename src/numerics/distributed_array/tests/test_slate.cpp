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
#include "methods/scr_coulomb/lr_scr_coulomb_t.hpp"

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
// axis is split over the run's rank count. SLATE only accepts the distribution
// when every non-final block along an axis has the full block size, and needs
// square tiles, so the block size is floor(N/np) on the finer axis — the same
// formula the ω-side W grid uses (see "lr_W_omega_dist" below).
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
      app_log(2, "  inverse: N = {}, pgrid = ({}, {}), bsize = {}, max error = {:.3e}",
              N, np_i, np_j, bsize, err);
      INFO("N = " << N << ", pgrid = (" << np_i << ", " << np_j << "), bsize = " << bsize);
      CHECK(err < 1e-10);
    };

    check(world.size(), 1);
    check(1, world.size());
    long nx = utils::find_proc_grid_min_diff(world.size(), N, N);
    check(nx, world.size()/nx);
  };

  run(40);
  // Not divisible by the rank counts ctest uses: ragged last local block and a
  // short trailing tile on both axes, which is the production geometry (THC
  // nIpts 1687 split over 3 P ranks).
  run(41);
}

// The distribution W(iω) is carried on in the LR W Dyson
// (lr_scr_coulomb_t::W_omega_dist, i.e. scr_coulomb_fourier_t::ft_buffer_dist).
// Three properties are load-bearing on that path: the (P,Q) block is square
// (the C-order branch of multiply_impl issues slate::multiply(a,Bs,As,b,Cs),
// which needs Bs.nt() == As.mt()), the array still stores that square block
// after make_distributed_array's min(bsize, shape/grid) clamp — that clamp is
// what the fused FT branches and SLATE both read — and the two rank-4
// multiplies of lr_dyson_W_in_place are numerically right on the grid.
TEST_CASE("lr_W_omega_dist", "[math]")
{
  using methods::solvers::scr_coulomb_fourier_t;
  using methods::solvers::lr_scr_coulomb_t;
  using grid_t = std::array<long, 4>;

  auto world = boost::mpi3::environment::get_world_instance();
  auto all = nda::range::all;

  // --- pure function: square (P,Q) block, and W_omega_dist == the FT buffer ---
  // (nproc, nq, nw_half, NP). The last row is the production point of record:
  // BaBiO3 nk8, 768 ranks, nq = 512, THC nIpts = 1687.
  const std::array<grid_t, 10> sweep = {{
      {1, 8, 4, 98},   {8, 8, 18, 98},   {16, 8, 18, 98},  {24, 8, 18, 98},
      {32, 8, 18, 98}, {64, 8, 18, 98},  {24, 8, 18, 41},  {32, 8, 18, 41},
      {96, 512, 36, 1687}, {768, 512, 36, 1687}}};

  for (auto s : sweep) {
    const long nproc = s[0], nq = s[1], nwh = s[2], NP = s[3];
    INFO("nproc = " << nproc << ", nq = " << nq << ", nw_half = " << nwh
         << ", NP = " << NP);
    auto [b_pgrid, b_bsize] =
        scr_coulomb_fourier_t::ft_buffer_dist(nproc, {nwh, nq, NP, NP});
    CHECK(b_bsize[2] == b_bsize[3]);
    auto [w_pgrid, w_bsize] = lr_scr_coulomb_t::W_omega_dist(nproc, nq, nwh, NP);
    CHECK((w_pgrid == b_pgrid));
    CHECK((w_bsize == b_bsize));
  }

  // The production point, spelled out: the P-split grid whose SUMMA was
  // measured, with the 1024-capped square tile min(1024, 1687/3, 1687/1).
  {
    auto [pg, bs] = lr_scr_coulomb_t::W_omega_dist(768, 512, 36, 1687);
    CHECK((pg == grid_t{1, 256, 3, 1}));
    CHECK((bs == grid_t{1, 1, 562, 562}));
  }

  // --- the stored block size, and the two Dyson multiplies on the grid ---
  const long nproc = world.size();

  auto check_grid = [&](long nq, long NP) {
    const long nwh = 2;
    auto [pg, bs] = lr_scr_coulomb_t::W_omega_dist(nproc, nq, nwh, NP);
    if (nwh < pg[0] or nq < pg[1] or NP < pg[2] or NP < pg[3]) return;
    INFO("nproc = " << nproc << ", nq = " << nq << ", NP = " << NP
         << ", pgrid = (" << pg[0] << "," << pg[1] << "," << pg[2] << ","
         << pg[3] << "), bsize = (" << bs[2] << "," << bs[3] << ")");

    const shape_t<4> gshape{nwh, nq, NP, NP};
    using local_Array_t = nda::array<ComplexType, 4>;
    auto dPi  = make_distributed_array<local_Array_t>(world, pg, gshape, bs);
    auto dW1  = make_distributed_array<local_Array_t>(world, pg, gshape, bs);
    auto dW2  = make_distributed_array<local_Array_t>(world, pg, gshape, bs);
    auto dTmp = make_distributed_array<local_Array_t>(world, pg, gshape, bs);

    CHECK(dPi.block_size()[2] == dPi.block_size()[3]);
    CHECK((dPi.block_size() == bs));

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
    app_log(2, "  lr_W_omega_dist: nq = {}, NP = {}, pgrid = ({},{},{},{}), "
               "bsize = ({},{}), max rel error = {:.3e}",
            nq, NP, pg[0], pg[1], pg[2], pg[3],
            dPi.block_size()[2], dPi.block_size()[3], err/nrm);
    CHECK(err < 1e-12 * nrm);
  };

  // nq >= nproc: every rank owns one q, (P,Q) is local and multiply_impl takes
  // its rank-local gemm short circuit.
  check_grid(nproc, 41);
  // nq < nproc: (P,Q) is split and the multiplies are real SLATE SUMMAs. NP = 41
  // does not divide the P/Q rank counts, so the tiles are ragged.
  check_grid(2, 40);
  check_grid(2, 41);
  // nq = 1: the whole rank count goes into (P,Q) — the widest SUMMA available.
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
