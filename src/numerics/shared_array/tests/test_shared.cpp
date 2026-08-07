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
#include "numerics/shared_array/nda.hpp"
#include "utilities/test_common.hpp"

namespace bdft_tests
{

namespace mpi3 = boost::mpi3;
using namespace math::shm;
using namespace math::nda;
template <int Rank> using shape_t = std::array<long, Rank>;

TEST_CASE("distributed_shared_nda", "[math]") {
  auto world = mpi3::environment::get_world_instance();
  auto node_comm = world.split_shared();
  // Setup internode communicator
  int node_size = node_comm.size();
  int color = world.rank()%node_size;
  int key   = world.rank()/node_size;
  auto internode_comm = world.split(color, key);

  using Array_view_base_t = nda::array_view<ComplexType, 3>;

  int n_nodes = internode_comm.size();
  int node_rank = internode_comm.rank();
  shape_t<3> grid = {n_nodes, 1, 1};
  shape_t<3> gshape = {39, 2, 2};

  auto array = make_distributed_shared_array<Array_view_base_t>(world, internode_comm, node_comm,
                                                                grid, gshape);
  app_log(2, "Global shape = ({}, {}, {})", array.global_shape()[0], array.global_shape()[1], array.global_shape()[2]);
  world.barrier();
  std::cout << "At node " << node_rank << ", local shape = (" << array.local_shape()[0] <<
  ", " << array.local_shape()[1] << ", " << array.local_shape()[2] << ")" << std::endl;
  std::cout << "At node " << node_rank << ", local origin = (" << array.origin()[0] <<
  ", " << array.origin()[1] << ", " << array.origin()[2] << ")" << std::endl;

  int rank = array.node_comm()->rank();
  int group_size = array.node_comm()->size();
  auto array_loc = array.local();
  nda::matrix<ComplexType> eye(2, 2);
  eye() = 2.0;
  int t_offset = array.origin()[0];
  for (int it = rank; it < array.local_shape()[0]; it += group_size) {
    int t = it + t_offset;
    nda::matrix_view<ComplexType> array_t = array_loc(t, nda::range::all, nda::range::all);
    array_t += 2.0;
  }
  array.node_sync();
}

/**
 * gather_to_shm replicates a distributed array into one shared-memory copy per node
 * via an internode reduction of disjoint blocks. The reduction is split across the
 * ranks of a node (shared_array::all_reduce_parallel), so the interesting behaviour
 * only appears with more than one node.
 *
 * A single test machine is one node, so carve it into synthetic "nodes" by splitting
 * the shared communicator — shared memory still works within each subset. The ragged
 * section is the important one: chunking by the *local* node size would leave the
 * chunks above the smallest node's rank count unreduced, silently losing data.
 */
TEST_CASE("gather_to_shm_synthetic_nodes", "[math]") {
  auto world = mpi3::environment::get_world_instance();
  auto phys_node = world.split_shared();
  if (phys_node.size() != world.size()) return;  // needs all ranks on one machine
  if (world.size() < 4) return;

  using Array_t = nda::array<ComplexType, 3>;
  using Array_view_t = nda::array_view<ComplexType, 3>;
  shape_t<3> gshape = {17, 3, 5};   // deliberately not divisible by the rank count

  // Reference: element (i,j,k) carries a unique value, so a dropped or double-counted
  // block cannot cancel out.
  Array_t ref(gshape);
  for (long i = 0; i < gshape[0]; ++i)
    for (long j = 0; j < gshape[1]; ++j)
      for (long k = 0; k < gshape[2]; ++k)
        ref(i, j, k) = ComplexType(1.0 + i * 100 + j * 10 + k, 0.5 - i);

  auto run = [&](int nfake, int split_at, const std::string& tag) {
    // split_at < 0 => even split into nfake groups; otherwise two ragged groups
    int color = (split_at < 0) ? (world.rank() * nfake) / world.size()
                               : ((world.rank() < split_at) ? 0 : 1);
    auto fake_node = phys_node.split(color, world.rank());
    auto fake_internode = world.split(fake_node.rank(), world.rank());

    auto dA = make_distributed_array<Array_t>(world, {world.size(), 1, 1}, gshape);
    auto i_rng = dA.local_range(0);
    if (i_rng.size() > 0)
      dA.local() = ref(i_rng, nda::range::all, nda::range::all);

    auto sA = shared_array<Array_view_t>(std::addressof(world),
                                         std::addressof(fake_internode),
                                         std::addressof(fake_node), gshape);
    gather_to_shm(dA, sA);

    double err = 0.0;
    if (fake_node.root()) {
      // frobenius_norm is rank-2 only; flatten the trailing axes of the difference.
      Array_t diff(sA.local() - ref);
      err = nda::frobenius_norm(
          nda::reshape(diff, std::array<long, 2>{gshape[0], gshape[1] * gshape[2]}));
    }
    err = world.all_reduce_value(err, boost::mpi3::max<>{});
    app_log(2, "gather_to_shm [{}]: nodes = {}, ranks/node = {}, max err = {:.3e}",
            tag, fake_internode.size(), fake_node.size(), err);
    REQUIRE(err < 1e-14);
  };

  run(2, -1, "even, 2 nodes");
  if (world.size() >= 8) run(4, -1, "even, 4 nodes");
  // Ragged: group 0 gets one more rank than group 1, so the smallest node bounds nchunks.
  run(2, world.size() / 2 + 1, "ragged");
  run(2, world.size() - 1, "ragged, 1-rank node");
}

} // bdft_tests
