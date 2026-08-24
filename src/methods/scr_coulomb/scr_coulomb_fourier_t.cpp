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


#include "scr_coulomb_fourier_t.h"
#include "utilities/proc_meminfo.hpp"

namespace methods {
namespace solvers {

  template<nda::MemoryArrayOfRank<4> local_Array_t, typename communicator_t>
  auto scr_coulomb_fourier_t::tau_to_w(
      memory::darray_t<local_Array_t, communicator_t> &dPi_tqPQ_pos,
      std::array<long, 4> w_pgrid_out, std::array<long, 4> w_tcount_out,
      bool reset_input, bool check_leakage)
  -> memory::darray_t<local_Array_t, mpi3::communicator>
  {
    using math::nda::make_distributed_array;

    _Timer.start("IMAG_FT_TtoW");
    auto comm = dPi_tqPQ_pos.communicator();
    long npts = dPi_tqPQ_pos.global_shape()[1];
    long Np = dPi_tqPQ_pos.global_shape()[3];
    long nw_half = (_ft->nw_b()%2==0)? _ft->nw_b()/2 : _ft->nw_b()/2 + 1;
    std::array<long, 4> w_gshape = {nw_half, npts, Np, Np};
    std::array<long, 4> t_gshape = dPi_tqPQ_pos.global_shape();

    // Single-rank fast path: the whole array is already local, so axis 0 (τ)
    // needs no redistribution — FT in place, skipping all staging buffers.
    if (dPi_tqPQ_pos.communicator()->size() == 1) {
      if (check_leakage) {
        _ft->check_leakage(dPi_tqPQ_pos, imag_axes_ft::boson, "polarizability", true);
      }
      auto dPi_wqPQ = make_distributed_array<local_Array_t>(
          // axis 0 changes extent (tau -> omega), so the input's tile count there
          // does not carry over; 0 = one element per tile
          *comm, {1, 1, 1, 1}, w_gshape,
          {0, dPi_tqPQ_pos.tile_count()[1], dPi_tqPQ_pos.tile_count()[2],
              dPi_tqPQ_pos.tile_count()[3]});
      auto Pi_ti_loc = dPi_tqPQ_pos.local();
      auto Pi_wi_loc = dPi_wqPQ.local();
      _ft->tau_to_w_PHsym(Pi_ti_loc, Pi_wi_loc);
      if (reset_input) dPi_tqPQ_pos.reset();
      _Timer.stop("IMAG_FT_TtoW");
      return dPi_wqPQ;
    }
    // redistribute to cover (tau, w)-axes locally -> FT locally -> redistribute back
    // Buffer distributes over q AND PQ (tau/omega axis 0 stays undivided for FT).
    auto [b_pgrid, b_tcount] = ft_buffer_dist(comm->size(), t_gshape);

    // τ-side staging buffer: the input keeps τ distributed, the local FT needs it
    // rank-local. Released as soon as the FT has consumed it.
    auto buffer_ti = make_distributed_array<local_Array_t>(*comm, b_pgrid, t_gshape, b_tcount);

    _Timer.start("FT_REDISTRIBUTE");
    math::nda::redistribute(dPi_tqPQ_pos, buffer_ti);
    _Timer.stop("FT_REDISTRIBUTE");
    if (reset_input) dPi_tqPQ_pos.reset();
    if (check_leakage) {
      _ft->check_leakage(buffer_ti, imag_axes_ft::boson, "polarizability", true);
    }

    if (w_pgrid_out == b_pgrid && w_tcount_out == b_tcount) {
      // Output distribution == buffer distribution: FT straight into the
      // output, no ω-side staging buffer and no final redistribute.
      auto dPi_wqPQ = make_distributed_array<local_Array_t>(
          *comm, w_pgrid_out, w_gshape, w_tcount_out);
      auto buf_ti_loc = buffer_ti.local();
      auto Pi_wi_loc = dPi_wqPQ.local();
      _ft->tau_to_w_PHsym(buf_ti_loc, Pi_wi_loc);
      buffer_ti.reset();
      _Timer.stop("IMAG_FT_TtoW");
      return dPi_wqPQ;
    }

    // Output distribution differs from the buffer distribution: FT into the
    // ω-side staging buffer, then redistribute into the output. The output is
    // allocated only once the τ-side buffer is gone, so the τ and ω grids never
    // coexist three at a time.
    auto buffer_wi = make_distributed_array<local_Array_t>(*comm, b_pgrid, w_gshape, b_tcount);
    {
      auto buf_ti_loc = buffer_ti.local();
      auto buf_wi_loc = buffer_wi.local();
      _ft->tau_to_w_PHsym(buf_ti_loc, buf_wi_loc);
    }
    buffer_ti.reset();

    auto dPi_wqPQ = make_distributed_array<local_Array_t>(
        *comm, w_pgrid_out, w_gshape, w_tcount_out);

    _Timer.start("FT_REDISTRIBUTE");
    math::nda::redistribute(buffer_wi, dPi_wqPQ);
    _Timer.stop("FT_REDISTRIBUTE");
    buffer_wi.reset();

    _Timer.stop("IMAG_FT_TtoW");
    return dPi_wqPQ;
  }

  template<nda::MemoryArrayOfRank<4> local_Array_t, typename communicator_t>
  auto scr_coulomb_fourier_t::w_to_tau(
      memory::darray_t<local_Array_t, communicator_t> &dW_wqPQ_pos,
      std::array<long, 4> t_pgrid_out, std::array<long, 4> t_tcount_out,
      bool reset_input, bool check_leakage)
  -> memory::darray_t<local_Array_t, mpi3::communicator>
  {
    using math::nda::make_distributed_array;

    _Timer.start("IMAG_FT_WtoT");
    auto comm = dW_wqPQ_pos.communicator();
    long npts = dW_wqPQ_pos.global_shape()[1];
    long Np = dW_wqPQ_pos.global_shape()[3];
    auto w_gshape = dW_wqPQ_pos.global_shape();
    size_t nt_half = (_ft->nt_b()%2==0)? _ft->nt_b() / 2 : _ft->nt_b() / 2 + 1;
    std::array<long, 4> t_gshape = {(long)nt_half, npts, Np, Np};

    // Single-rank fast path: the whole array is already local, so axis 0 (ω)
    // needs no redistribution — FT in place, skipping all staging buffers.
    if (dW_wqPQ_pos.communicator()->size() == 1) {
      auto dW_tqPQ = make_distributed_array<local_Array_t>(
          *comm, {1, 1, 1, 1}, t_gshape, {0, 0, 0, 0});
      auto W_wi_loc = dW_wqPQ_pos.local();
      auto W_ti_loc = dW_tqPQ.local();
      _ft->w_to_tau_PHsym(W_wi_loc, W_ti_loc);
      if (reset_input) dW_wqPQ_pos.reset();
      if (check_leakage) {
        _ft->check_leakage(dW_tqPQ, imag_axes_ft::boson, "screened interaction", true);
      }
      _Timer.stop("IMAG_FT_WtoT");
      return dW_tqPQ;
    }

    // redistribute to cover (tau, w)-axes locally -> FT locally -> redistribute back
    // Buffer distributes over q AND PQ (tau/omega axis 0 stays undivided for FT).
    auto [b_pgrid, b_tcount] = ft_buffer_dist(comm->size(), t_gshape);

    // τ-side staging buffer: the FT needs τ rank-local, while t_pgrid_out keeps τ
    // distributed, so the result cannot be written straight into dW_tqPQ. Released
    // as soon as the FT output has been copied out, and allocated only once the FT
    // input is in place, so it is not resident across the input redistribute.
    memory::darray_t<local_Array_t, mpi3::communicator> buffer_ti;

    if (dW_wqPQ_pos.grid() == b_pgrid && dW_wqPQ_pos.tile_count() == b_tcount) {
      // Input distribution == buffer distribution: FT straight from the input,
      // no ω-side staging buffer and no input redistribute.
      buffer_ti = make_distributed_array<local_Array_t>(*comm, b_pgrid, t_gshape, b_tcount);
      auto buf_wi_loc = dW_wqPQ_pos.local();
      auto buf_ti_loc = buffer_ti.local();
      _ft->w_to_tau_PHsym(buf_wi_loc, buf_ti_loc);
      if (reset_input) dW_wqPQ_pos.reset();
    } else {
      // Input distribution differs from the buffer distribution: redistribute
      // the input into the ω-side staging buffer, then FT.
      auto buffer_wi = make_distributed_array<local_Array_t>(*comm, b_pgrid, w_gshape, b_tcount);

      _Timer.start("FT_REDISTRIBUTE");
      math::nda::redistribute(dW_wqPQ_pos, buffer_wi);
      _Timer.stop("FT_REDISTRIBUTE");
      if (reset_input) dW_wqPQ_pos.reset();

      buffer_ti = make_distributed_array<local_Array_t>(*comm, b_pgrid, t_gshape, b_tcount);
      {
        auto buf_wi_loc = buffer_wi.local();
        auto buf_ti_loc = buffer_ti.local();
        _ft->w_to_tau_PHsym(buf_wi_loc, buf_ti_loc);
      }
      buffer_wi.reset();
    }

    if (check_leakage) {
      _ft->check_leakage(buffer_ti, imag_axes_ft::boson, "screened interaction", true);
    }

    auto dW_tqPQ = make_distributed_array<local_Array_t>(
        *comm, t_pgrid_out, t_gshape, t_tcount_out);

    _Timer.start("FT_REDISTRIBUTE");
    math::nda::redistribute(buffer_ti, dW_tqPQ);
    _Timer.stop("FT_REDISTRIBUTE");
    buffer_ti.reset();

    _Timer.stop("IMAG_FT_WtoT");
    return dW_tqPQ;
  }

  // template instantiations
  using Arr4D = nda::array<ComplexType, 4>;

  template memory::darray_t<Arr4D, mpi3::communicator>
  scr_coulomb_fourier_t::tau_to_w(memory::darray_t<Arr4D, mpi3::communicator> &,
                 std::array<long, 4>, std::array<long, 4>, bool, bool);

  template memory::darray_t<Arr4D, mpi3::communicator>
  scr_coulomb_fourier_t::w_to_tau(memory::darray_t<Arr4D, mpi3::communicator> &,
                 std::array<long, 4>, std::array<long, 4>, bool, bool);

}  // solvers
}  // methods
