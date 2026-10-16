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


#ifndef UTILITIES_PROC_GRID_PARTITION_HPP
#define UTILITIES_PROC_GRID_PARTITION_HPP

#include <array>
#include <algorithm>
#include <cstdlib>
#include <functional>
#include <numeric>
#include <optional>
#include <string>
#include <utility>

#include "utilities/check.hpp"

namespace utils
{

/**
 * Robust processor grid for a distributed array of the given global `shape` over
 * `np` ranks. The returned grid never over-subscribes: g[i] <= shape[i] for
 * every axis, so it cannot trigger the "Too many processors" abort in
 * make_distributed_array. Axes are filled in index order (axis 0 first), so the
 * caller should order `shape` by parallelization priority (e.g. {nspin, nkpts,
 * nbnd, nbnd}).
 *
 * Two passes: the first assigns balanced divisor-clean pools per axis; the
 * second packs any ranks stranded by clean division onto axes that still have
 * spare extent (possibly uneven, handled by chunk_range). When `np` cannot be
 * tiled onto `shape` at all (e.g. a prime factor exceeds every extent), the
 * product of the returned grid, `n_active`, is strictly less than `np`: the
 * surplus ranks must stay idle and the array must be built on a sub-communicator
 * of size `n_active`. This keeps few-band/many-core runs alive (at reduced
 * efficiency) instead of aborting.
 *
 * @return {grid, n_active} with prod(grid) == n_active <= np.
 */
template<size_t R>
inline std::pair<std::array<long,R>, long>
find_proc_grid_capped(long np, std::array<long,R> shape)
{
  std::array<long,R> g;
  g.fill(1);
  utils::check(np >= 1, "find_proc_grid_capped: np must be >= 1.");
  for (auto e : shape) utils::check(e >= 1, "find_proc_grid_capped: shape extents must be >= 1.");

  // Pass 1: balanced pools -- largest divisor of the remaining budget that fits.
  long budget = np;
  for (size_t ax = 0; ax < R; ++ax) {
    long d = 1;
    for (long i = std::min(budget, shape[ax]); i >= 1; --i)
      if (budget % i == 0) { d = i; break; }
    g[ax] = d;
    budget /= d;
  }

  // Pass 2: raise each axis toward its extent (favoring earlier / higher-priority
  // axes) to pack ranks stranded by clean division, keeping the product <= np.
  // May pick a non-divisor value; make_distributed_array splits it unevenly via
  // chunk_range. E.g. np=96, shape={1,8,10,1}: pass 1 gives {1,8,6,1}=48, pass 2
  // raises the band axis to {1,8,10,1}=80 (16 idle) instead of leaving 48 idle.
  long prod = 1;
  for (auto v : g) prod *= v;
  for (size_t ax = 0; ax < R; ++ax) {
    long base = prod / g[ax];             // product of the other axes
    long best = g[ax];
    for (long v = shape[ax]; v > best; --v)
      if (base * v <= np) { best = v; break; }
    prod = base * best;
    g[ax] = best;
  }

  return {g, prod};
}

/**
 * Find a processor grid {n x m} where max(nkpts, gcomm.size()) = n*m and n is maximized
 * @tparam communicator
 * @param gcomm
 * @param nkpts
 * @return
 */
// find processor grid {n x m}, maximize and return (n) 
template<typename communicator>
long find_proc_grid_max_rows(communicator & gcomm, long nkpts)
{
  long np = std::max(nkpts,long(gcomm.size()));
  long nk = std::min(nkpts,long(gcomm.size()));
  for(long i=1; i<nk/2+1; ++i)
    if( nk%i == 0 and np%(nk/i) == 0 ) return nk/i;
  return 1;
}

inline long find_proc_grid_max_rows(long size, long nkpts)
{
  long np = std::max(nkpts,size);
  long nk = std::min(nkpts,size);
  for(long i=1; i<nk/2+1; ++i)
    if( nk%i == 0 and np%(nk/i) == 0 ) return nk/i;
  return 1;
}

/**
 * Find maximum number of pools along dimension i with the constraint
 * that 1/pool_size <= error
 * @param np    - number of processors
 * @param dim_i - dimension i
 * @param error - error constraint
 * @return - number of pools
 */
inline long find_proc_grid_max_npools(long np, long dim_i, double error) {
  long npools = 1;
  for (long i = std::min(np, dim_i); i > 0; --i) {
    long pool_size = dim_i / i;
    if (dim_i % i == 0 and np % i == 0) {
      npools = i;
      break;
    } else {
      if ((double) 1.0 / pool_size <= error and np % i == 0) {
        npools = i;
        break;
      }
    }
  }
  return npools;
}

/**
 * For a given number of processors (np),
 * find a processor grid {n x m} that minimizes (n*nc-m*nr) and n*m=np, return n
 * (basically try to find a grid that n ~ nr and m ~ nc)
 * @param np - number of processors
 * @param nr - number of rows
 * @param nc - number of columns
 * @return
 */
inline long find_proc_grid_min_diff(long np, long nr, long nc)
{
  // now look for minimum
  long maxd = np*std::max(nr,nc), indx=-1;
  for(long i=np; i>0; i--)
    if( np%i==0 and std::abs(i*nc - (np/i)*nr) < maxd ) {
      maxd = std::abs(i*nc - (np/i)*nr);
      indx = i;
    }
  utils::check(indx>0 and np%indx==0, "Problems finding processor grid.");
  return std::max(indx,np/indx);  // to make leading dimension smaller
}

/**
 * find the minimum m such that np = n * m where n is maximized and n <= nr_max
 * @return n
 */
inline long find_min_col(long np, long nr_max, long nc_min=1) {
  if (nr_max > np) {
    return 1;
  } else {
    long m = np / nr_max;
    while (np % m != 0 or m < nc_min) {
      m += 1;
    }
    return m;
  }
}

/**
 * Whether a processor grid can distribute an array of the given extents:
 * every grid dimension is >= 1 and does not exceed the corresponding extent.
 */
template<size_t N>
inline bool proc_grid_fits(std::array<long,N> const& grid, std::array<long,N> const& extents)
{
  for(size_t i=0; i<N; ++i)
    if(grid[i] < 1 or grid[i] > extents[i]) return false;
  return true;
}

/**
 * Processor grid {n_0, ..., n_{N-1}} with prod(n_i) = np and n_i <= extents[i].
 * Returns `pref` when it is such a grid. Otherwise, among all such grids, the one that maximizes
 * n_0, then n_1, ..., then n_{N-3}, and splits the remaining factor as evenly as possible between
 * the last two dimensions (the larger part first).
 * Returns std::nullopt when no such grid exists.
 * @param np      - number of processors
 * @param extents - global shape of the distributed dimensions
 * @param pref    - [OPTIONAL] preferred grid, returned unchanged when it fits
 */
template<size_t N>
inline std::optional<std::array<long,N>> find_proc_grid_fit(long np, std::array<long,N> const& extents,
                                                            std::optional<std::array<long,N>> pref = std::nullopt)
{
  static_assert(N >= 1, "find_proc_grid_fit: N < 1");
  utils::check(np > 0, "find_proc_grid_fit: np <= 0");
  if(pref.has_value()) {
    long prod = std::accumulate(pref->begin(), pref->end(), 1l, std::multiplies<>{});
    if(prod == np and proc_grid_fits(*pref, extents)) return pref;
  }
  // ranking key: (n_0, ..., n_{N-3}, -|n_{N-2}-n_{N-1}|, n_{N-2}), larger is better
  auto key = [](std::array<long,N> const& g) {
    std::array<long,N> k = g;
    if constexpr (N >= 2) {
      k[N-2] = -std::abs(g[N-2]-g[N-1]);
      k[N-1] = g[N-2];
    }
    return k;
  };
  std::optional<std::array<long,N>> best;
  std::array<long,N> g;
  auto search = [&](auto&& self, size_t i, long rem) -> void {
    if(i == N-1) {
      if(rem > extents[i]) return;
      g[i] = rem;
      if(not best.has_value() or key(g) > key(*best)) best = g;
      return;
    }
    for(long d=1; d<=std::min(rem,extents[i]); ++d) {
      if(rem%d != 0) continue;
      g[i] = d;
      self(self, i+1, rem/d);
    }
  };
  search(search, 0, np);
  return best;
}

/**
 * Splits np processors into np/g groups of g consecutive ranks, each group holding a complete copy
 * of an array of the given extents on the processor grid find_proc_grid_fit(g, extents, pref).
 * g is the largest divisor of np for which such a grid exists: g == np (one group, grid == pref
 * when pref fits) whenever the array can be distributed over all processors, g == 1 at worst.
 * @return {g, grid}
 */
template<size_t N>
inline std::pair<long, std::array<long,N>> find_proc_grid_fit_groups(long np, std::array<long,N> const& extents,
                                                                      std::optional<std::array<long,N>> pref = std::nullopt)
{
  for(long g=np; g>1; --g) {
    if(np%g != 0) continue;
    auto grid = find_proc_grid_fit(g, extents, (g==np ? pref : std::nullopt));
    if(grid.has_value()) return {g, *grid};
  }
  std::array<long,N> ones;
  ones.fill(1);
  return {1, ones};
}

/**
 * Largest g <= np for which a processor grid of g ranks fits the given extents, and that grid
 * (find_proc_grid_fit(g, extents), with `pref` considered only for g == np). g == np whenever the
 * array can be distributed over all processors, g == 1 at worst. Used with
 * math::nda::make_distributed_array_on_first_ranks, which leaves the remaining np-g ranks empty.
 *
 * Why not find_proc_grid_capped + detail::assemble_one_body (one_body_hamiltonian.hpp), which also
 * idles surplus ranks: those build the array on a sub-communicator of the active ranks (uneven tiles
 * allowed) and only fill a shared array through it. The first-ranks array instead keeps the full
 * communicator, so it takes part in math::nda::redistribute, gather_to_shm and collectives on
 * mpi->comm like any all-rank array, with no communicator split, and is the plain array when g == np.
 * @return {g, grid}
 */
template<size_t N>
inline std::pair<long, std::array<long,N>> find_proc_grid_fit_subset(long np, std::array<long,N> const& extents,
                                                                      std::optional<std::array<long,N>> pref = std::nullopt)
{
  for(long g=np; g>1; --g) {
    auto grid = find_proc_grid_fit(g, extents, (g==np ? pref : std::nullopt));
    if(grid.has_value()) return {g, *grid};
  }
  std::array<long,N> ones;
  ones.fill(1);
  return {1, ones};
}

/**
 * find_proc_grid_fit() that aborts with a clear message when no processor grid exists.
 * @param where - name of the caller, for the error message
 */
template<size_t N>
inline std::array<long,N> find_proc_grid_fit_or_abort(long np, std::array<long,N> const& extents,
                                                      std::optional<std::array<long,N>> pref,
                                                      std::string const& where)
{
  auto grid = find_proc_grid_fit(np, extents, pref);
  if(not grid.has_value()) {
    std::string ext;
    for(size_t i=0; i<N; ++i) ext += (i==0 ? "" : ", ") + std::to_string(extents[i]);
    utils::check(false, "{}: no processor grid of {} MPI ranks fits the distributed dimensions ({}). "
                        "Use a number of ranks that factorizes within these extents.", where, np, ext);
  }
  return *grid;
}

template<typename communicator>
inline auto setup_two_layer_mpi(communicator *comm, const size_t dim0, const size_t dim1) {
  size_t dim0_rank, dim0_comm_size, dim1_rank, dim1_comm_size;
  if (comm!=nullptr) {
    dim0_comm_size = find_proc_grid_min_diff(comm->size(), dim0, dim1);
    dim1_comm_size = comm->size() / dim0_comm_size;
    dim0_rank = comm->rank() / dim1_comm_size;
    dim1_rank = comm->rank() % dim1_comm_size;
  } else {
    dim0_rank = 0;
    dim0_comm_size = 1;
    dim1_rank = 0;
    dim1_comm_size = 1;
  }
  std::array<size_t, 4> mpi_info = {dim0_rank, dim0_comm_size, dim1_rank, dim1_comm_size};
  return mpi_info;
}

}


#endif
