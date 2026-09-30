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


#ifndef NUMERICS_SHARED_ARRAY_NDA_HPP
#define NUMERICS_SHARED_ARRAY_NDA_HPP

#include "configuration.hpp"
#include "mpi3/communicator.hpp"
#include "mpi3/shared_window.hpp"
#include "nda/nda.hpp"
#include "numerics/distributed_array/nda.hpp"
#include "utilities/mpi_context.h"

#include "utilities/check.hpp"

namespace math {
  namespace shm {

    namespace mpi3 = boost::mpi3;

    // TODO
    //   - merge this with distributed array
    /**
     * A simple wrapper for nda arrays and MPI shared memory (from boost::mpi3)
     * still in design stage...
     */
    // nda::Array or nda::MemoryArray?
    template<::nda::MemoryArray Array_base_t>
    class shared_array {
    public:
      using Array_view_t = decltype(std::declval<std::decay_t < Array_base_t>>()());
      static constexpr int rank = ::nda::get_rank<Array_view_t>;
      using value_type = typename std::decay_t<Array_view_t>::value_type;

      static constexpr bool is_stride_order_Fortran() noexcept {
        return Array_view_t::layout_t::is_stride_order_Fortran();
      }
      static constexpr bool is_stride_order_C() noexcept {
        return Array_view_t::layout_t::is_stride_order_C();
      }

    private:
      using darray_t = math::nda::detail::darray<rank, mpi3::communicator>;
      static_assert ( Array_view_t::layout_t::is_stride_order_Fortran()
        or Array_view_t::layout_t::is_stride_order_C(), "Ordering mismatch.");
    public:
      shared_array(mpi3::shared_communicator* node_comm,
                   std::array<long, rank> shape, bool zero_init = true) :
          _node_comm(node_comm),
          _size(std::accumulate(shape.cbegin(), shape.cend(), (mpi3::size_t)1, std::multiplies<>{})),
          _shape(shape),
          _win(std::make_unique<mpi3::shared_window<value_type>>(*node_comm, (node_comm->root()) ? _size : 0))
      {
        check_and_init(zero_init);
      }

      shared_array(mpi3::communicator *gcomm,
                   mpi3::communicator *internode_comm,
                   mpi3::shared_communicator *node_comm,
                   std::array<long, rank> shape, bool zero_init = true):
          _gcomm(gcomm), _internode_comm(internode_comm), _node_comm(node_comm),
          _size(std::accumulate(shape.cbegin(), shape.cend(), (mpi3::size_t)1, std::multiplies<>{})),
          _shape(shape),
          _win(std::make_unique<mpi3::shared_window<value_type>>(*node_comm, (node_comm->root()) ? _size : 0))
      {
        check_and_init(zero_init);
      }

      shared_array(utils::mpi_context_t<mpi3::communicator,mpi3::shared_communicator> &ctxt,
                   std::array<long, rank> shape, bool zero_init = true):
          _gcomm(std::addressof(ctxt.comm)),
          _internode_comm(std::addressof(ctxt.internode_comm)),
          _node_comm(std::addressof(ctxt.node_comm)),
          _size(std::accumulate(shape.cbegin(), shape.cend(), (mpi3::size_t)1, std::multiplies<>{})),
          _shape(shape),
          _win(std::make_unique<mpi3::shared_window<value_type>>(*_node_comm, (_node_comm->root()) ? _size : 0))
      {
        check_and_init(zero_init);
      }

      shared_array(const shared_array &other) :
          _gcomm(other.communicator()), 
          _internode_comm(other.internode_comm()), 
          _node_comm(other.node_comm()),
          _size(other.size()),
          _shape(other.shape()),
          _win(std::make_unique<mpi3::shared_window<value_type>>(*_node_comm, (_node_comm->root()) ? _size : 0))
      {
        check_and_init();
        node_sync();
        if (_node_comm->root()) 
          this->local() = other.local(); 
        node_sync(); 
      } 
      shared_array(shared_array &&other) = default;

      shared_array& operator=(const shared_array &other) {
        _gcomm = other.communicator();
        _internode_comm = other.internode_comm();
        _node_comm = other.node_comm();
        _size = other.size();
        _shape = other.shape();
        _reduce_nchunks = -1;
        _win = std::move(std::make_unique<mpi3::shared_window<value_type>>(*_node_comm, (_node_comm->root()) ? _size : 0));
        node_sync();
        if (_node_comm->root()) 
          this->local() = other.local(); 
        node_sync(); 
        return *this;
      }
      shared_array& operator=(shared_array &&other) = default;

      ~shared_array() = default; 

      void check_and_init(bool zero_init = true) {
        utils::check(_win->base(0) != nullptr, "shm::shared_array: win.base(0) == nullptr");
        utils::check(_win->size(0) == _size, "shm::shared_array: win.size(0) has incorrect dimension");
        if (_node_comm->size() > 1) {
          utils::check(_win->size(1) == 0, "shm::shared_array: win.size(!=0) has incorrect dimension");
        }
        // initialize array to 0.0, unless the caller opts out (buffer fully written before read)
        if (zero_init)
          set_zero();
      }

      void set_zero() {
        node_sync();
        auto[origin_i, end_i] = itertools::chunk_range(0, _size, _node_comm->size(), _node_comm->rank());
        ::nda::range i_range(origin_i, end_i);
        auto _array = Array_view_t(_shape, (value_type*) _win->base(0));
        auto array_1D = ::nda::reshape(_array, std::array<long, 1>{_size});
        array_1D(i_range) = value_type(0.0);
        node_sync();
      }

      void all_reduce() {
        node_sync();
        if (_node_comm->root()) {
          // split all_reduce() to avoid mpi count overflow
          for (size_t shift=0; shift<_size; shift+=size_t(1e9)) {
            value_type *start = (value_type*)_win->base(0) + shift;
            size_t count = (shift+size_t(1e9) < _size)? size_t(1e9) : _size-shift;
            _internode_comm->all_reduce_in_place_n(start, count, std::plus<>{});
          }
        }
        node_sync();
      }

      /**
       * Internode all_reduce with the buffer split across the ranks of a node.
       *
       * all_reduce() runs the whole reduction on the node root, so a single core moves
       * the entire array between nodes. Here each node-local rank reduces a disjoint
       * chunk over its own internode communicator: make_mpi_context splits by
       * node_comm.rank(), so every rank — not only the root — belongs to a communicator
       * that spans the nodes, and rank r cooperates with rank r of every other node.
       *
       * Equivalent to all_reduce(); only performance differs. Requires the internode
       * communicator to be comm.split(node_comm.rank(), ...), checked below.
       */
      void all_reduce_parallel() {
        utils::check(_internode_comm != nullptr and _gcomm != nullptr,
                     "shm::shared_array::all_reduce_parallel: array was built without a "
                     "global/internode communicator; use all_reduce().");
        // A communicator whose color exceeds some node's rank count does not span every
        // node, so its chunk would go unreduced. Chunk by the smallest node in the job.
        // Depends only on the communicators, so it is resolved once per array: every
        // rank takes this branch on the same call, which keeps the collective matched.
        if (_reduce_nchunks < 0) {
          _reduce_nchunks = _gcomm->all_reduce_value(long(_node_comm->size()), mpi3::min<>{});
          // The chunking is only a partition if _internode_comm was split by node rank
          // (make_mpi_context does). One built some other way can span fewer than all
          // nodes, which would silently leave its chunk unreduced. Checked in the same
          // one-time block so the steady state stays collective-free.
          long nnodes = _gcomm->all_reduce_value(long(_node_comm->root() ? 1 : 0), std::plus<>{});
          utils::check(_node_comm->rank() >= _reduce_nchunks or
                       long(_internode_comm->size()) == nnodes,
                       "shm::shared_array::all_reduce_parallel: internode_comm (size {}) does "
                       "not span all {} nodes; it must be built as "
                       "comm.split(node_comm.rank(), ...).", _internode_comm->size(), nnodes);
        }
        long nchunks = _reduce_nchunks;
        node_sync();
        if (_node_comm->rank() < nchunks) {
          auto [origin_i, end_i] =
              itertools::chunk_range(0, _size, nchunks, _node_comm->rank());
          // split into <1e9-element messages to avoid mpi count overflow
          for (size_t shift = size_t(origin_i); shift < size_t(end_i); shift += size_t(1e9)) {
            value_type *start = (value_type*)_win->base(0) + shift;
            size_t count = (shift + size_t(1e9) < size_t(end_i)) ? size_t(1e9)
                                                                 : size_t(end_i) - shift;
            _internode_comm->all_reduce_in_place_n(start, count, std::plus<>{});
          }
        }
        node_sync();
      }

      void broadcast_to_nodes(int src_node) {
        node_sync();
        if (_node_comm->root()) {
          for (size_t shift=0; shift<_size; shift+=size_t(1e9)) {
            value_type *start = (value_type *)_win->base(0) + shift;
            size_t count = (shift+size_t(1e9) < _size) ? size_t(1e9) : _size-shift;
            _internode_comm->broadcast_n(start, count, src_node);
          }
        }
        node_sync();
      }

      /// Makes this rank's writes to the window visible to the node and picks up
      /// everyone else's, so callers that wrote into it need no fence of their own.
      /// Both syncs are needed: a bare barrier orders the ranks but publishes nothing.
      void node_sync() {
        _win->sync();
        _node_comm->barrier();
        _win->sync();
      }

      mpi3::shared_communicator* node_comm() const { return _node_comm; }
      mpi3::communicator* communicator() const { return _gcomm; }
      mpi3::communicator* internode_comm() const { return _internode_comm; }
      mpi3::shared_window<value_type>& win() { return *_win; }

      auto const& shape() const { return _shape; }
      // for ::nda::get_rank() interface
      auto const& global_shape() const { return _shape; }
      auto size() const { return _size; }

      auto local() { return Array_view_t(_shape, (value_type*) _win->base(0)); }
      auto local() const { return Array_view_t(_shape, (value_type*) _win->base(0)); }

    protected:
      mpi3::communicator *_gcomm = nullptr;
      mpi3::communicator *_internode_comm = nullptr;
      mpi3::shared_communicator *_node_comm = nullptr;
      mpi3::size_t _size;
      std::array<long, rank> _shape;
      std::unique_ptr<mpi3::shared_window<value_type>> _win;
      // Chunk count used by all_reduce_parallel; < 0 until resolved. Follows the
      // communicators, so it must be invalidated whenever those are replaced.
      long _reduce_nchunks = -1;

    };

    template<::nda::MemoryArray Array_base_t>
    class distributed_shared_array : public shared_array<Array_base_t> {
    public:
      using Array_view_t = decltype(std::declval<std::decay_t < Array_base_t>>()());
      static constexpr int rank = ::nda::get_rank<Array_view_t>;
      using value_type = typename std::decay_t<Array_view_t>::value_type;

    private:
      using darray_t = math::nda::detail::darray<rank, mpi3::communicator>;

    public:
      /**
       * Constructor for distributed array among nodes
       */
      distributed_shared_array(mpi3::communicator *gcomm,
                               mpi3::communicator *internode_comm,
                               mpi3::shared_communicator *node_comm,
                               std::array<long, rank> grid,    // processor grid
                               std::array<long, rank> gshape,  // global shape
                               std::array<long, rank> lshape,  // local shape
                               std::array<long, rank> origin): // index origin of local array
          shared_array<Array_base_t>(gcomm, internode_comm, node_comm, lshape),
          _base(internode_comm, grid, gshape, origin, lshape) {
      }

      ~distributed_shared_array() = default;

      std::array<long, rank> const& global_shape() const { return _base.gextents; }
      std::array<long, rank> const& origin() const { return _base.lorigin; }
      std::array<long, rank> const& grid() const { return _base.grid; }

      auto const& local_shape() const { return this->_shape; }
      auto local_range(int dim) const {
        utils::check(dim >= 0 and dim < rank, "shared_array::range: Out or range d:{}", dim);
        return ::nda::range(_base.lorigin[dim], _base.lorigin[dim] + this->_shape[dim]);
      }

      void reset_loc_origin(int dim, long new_origin) {
        _base.lorigin[dim] = new_origin;
      }

      bool full_coverage() const { return _base.full_coverage_impl(this->_shape);}

    protected:
      darray_t _base;
    };

    /**
     * Shared memory array, one copy per node
     */
    template<::nda::MemoryArray Array_base_t>
    auto make_shared_array(mpi3::communicator &gcomm,
             mpi3::communicator &internode_comm,
             mpi3::shared_communicator &node_comm,
             std::array<long, ::nda::get_rank<std::decay_t<Array_base_t>>> shape) {
      using Array_t = shared_array<Array_base_t>;

      return Array_t(std::addressof(gcomm),
                     std::addressof(internode_comm),
                     std::addressof(node_comm),
                     shape);
    }

    /**
     * Shared memory array, one copy per node
     */
    template<::nda::MemoryArray Array_base_t>
    auto make_shared_array(utils::mpi_context_t<mpi3::communicator,mpi3::shared_communicator> &ctxt,
             std::array<long, ::nda::get_rank<std::decay_t<Array_base_t>>> shape) {
      using Array_t = shared_array<Array_base_t>;

      return Array_t(std::addressof(ctxt.comm),
                     std::addressof(ctxt.internode_comm),
                     std::addressof(ctxt.node_comm),
                     shape);
    }    

    /**
     * Shared memory array, one copy per node.
     * This should only be used for read-only array, i.e. no all_reduce between nodes.
     */
    template<::nda::MemoryArray Array_base_t>
    auto make_shared_array(mpi3::shared_communicator &node_comm,
                           std::array<long, ::nda::get_rank<std::decay_t<Array_base_t>>> shape) {
      using Array_t = shared_array<Array_base_t>;
      return Array_t(std::addressof(node_comm), shape);
    }

    /**
     * Distributed array among nodes.
     * Local arrays are stored in shared memory on each node.
     */
    template<::nda::MemoryArray Array_base_t>
    auto make_distributed_shared_array(mpi3::communicator &gcomm,
             mpi3::communicator &internode_comm,
             mpi3::shared_communicator &node_comm,
             std::array<long, ::nda::get_rank<std::decay_t<Array_base_t>>> grid,
             std::array<long, ::nda::get_rank<std::decay_t<Array_base_t>>> gshape) {
      static constexpr int rank = ::nda::get_rank<Array_base_t>;
      using Array_t = distributed_shared_array<Array_base_t>;

      std::array<long, rank> origin, lshape;
      long np = std::accumulate(grid.cbegin(), grid.cend(), 1, std::multiplies<>{});
      utils::check(internode_comm.size() == np,
                   "distributed_shared_array: Number of nodes does not match grid: nodes:{}, grid:{}",
                   internode_comm.size(), np);
      for (int n = 0; n < rank; n++) {
        utils::check(gshape[n] >= grid[n],
                     "distributed_shared_array: Too many processors i:{}, shape:{}, grid:{}",
                     n, gshape[n], grid[n]);
      }
      long ip = long(internode_comm.rank());
      // row major over proc grid for all other cases
      for(int n = rank - 1; n >= 0; n--) {
        std::tie(origin[n], lshape[n]) =
            itertools::chunk_range(0, gshape[n],grid[n],ip%grid[n]);
        lshape[n] -= origin[n];
        ip /= grid[n];
      }
      return Array_t(std::addressof(gcomm),
                     std::addressof(internode_comm),
                     std::addressof(node_comm),
                     grid, gshape, lshape, origin);
    }

    /**
     * Sums the n-element per-rank buffer `partial` over `comm`; rank r of `comm` adds
     * scl · (its stripe r of the sum) into ITS OWN node's window of sA, at flat offset
     * `offset + stripe_begin`. The write half of the pair whose other half is
     * shared_array::all_reduce_parallel().
     *
     * Collective on comm. No window sync and no internode reduction: when comm spans
     * several nodes, each node's window receives only the stripes of its own ranks, and
     * the caller's single all_reduce_parallel() publishes the writes and sums the
     * per-node partials. So it is correct for comm = node_comm and for a comm that
     * spans nodes alike. Repeated calls between one set_zero() and that reduction
     * accumulate into the same stripes.
     *
     * Stripes are element-granular, itertools::chunk_range over each chunk of `chunk`
     * elements. The chunking bounds the MPI accumulation temporaries and keeps every
     * count below 2^31. `partial` is clobbered.
     */
    template<::nda::MemoryArray Array_base_t>
    void reduce_scatter_add(mpi3::communicator& comm, shared_array<Array_base_t>& sA,
                            typename shared_array<Array_base_t>::value_type* partial,
                            long offset, long n,
                            typename shared_array<Array_base_t>::value_type scl,
                            long chunk = 1L << 22) {
      using value_type = typename shared_array<Array_base_t>::value_type;
      static_assert(std::is_same_v<value_type, std::complex<double>> or
                    std::is_same_v<value_type, double>,
                    "reduce_scatter_add: value_type must be double or std::complex<double>");
      MPI_Datatype dtype = std::is_same_v<value_type, double> ? MPI_DOUBLE
                                                              : MPI_CXX_DOUBLE_COMPLEX;
      utils::check(offset >= 0 and n >= 0 and offset + n <= long(sA.size()),
                   "reduce_scatter_add: range [{}, {}) exceeds the window size {}",
                   offset, offset + n, sA.size());
      utils::check(chunk > 0 and chunk <= long(std::numeric_limits<int>::max()),
                   "reduce_scatter_add: chunk {} outside (0, 2^31)", chunk);

      int np = comm.size();
      int r = comm.rank();
      std::vector<int> counts(np);
      value_type* win = sA.local().data();
      for (long c0 = 0; c0 < n; c0 += chunk) {
        long len = std::min(chunk, n - c0);
        for (int p = 0; p < np; ++p) {
          auto [b, e] = itertools::chunk_range(0, len, np, p);
          counts[p] = int(e - b);
        }
        // In place: the whole chunk is read from partial + c0, and this rank's stripe
        // of the sum lands at partial + c0, not at its offset within the chunk.
        MPI_Reduce_scatter(MPI_IN_PLACE, partial + c0, counts.data(), dtype, MPI_SUM,
                           comm.get());
        long i0 = itertools::chunk_range(0, len, np, r).first;
        value_type* dst = win + offset + c0 + i0;
        const value_type* src = partial + c0;
        for (long i = 0; i < counts[r]; ++i) dst[i] += scl * src[i];
      }
    }

  } // shm
} // math

#endif // NUMERICS_SHARED_ARRAY_NDA_HPP
