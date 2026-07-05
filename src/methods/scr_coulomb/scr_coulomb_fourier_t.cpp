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


#include <optional>

#include "scr_coulomb_fourier_t.h"

namespace methods {
namespace solvers {

  // Acquire an FT staging buffer of (gshape, pgrid, bsize) on `comm`.
  // `buffer_provided` non-null → a caller-owned buffer reused across calls (the
  // LR loop owns one τ- and one ω-shaped buffer), validated against the required
  // shape/dist. null → a per-call buffer allocated into `buffer_own`.
  // Returns the live buffer either way.
  template<class local_Array_t, class communicator_t>
  static memory::darray_t<local_Array_t, communicator_t>&
  acquire_ft_buffer(
      memory::darray_t<local_Array_t, communicator_t>* buffer_provided,
      std::optional<memory::darray_t<local_Array_t, communicator_t>>& buffer_own,
      communicator_t* comm,
      std::array<long, 4> pgrid, std::array<long, 4> gshape, std::array<long, 4> bsize)
  {
    if (buffer_provided) {
      utils::check(buffer_provided->communicator() == comm
                   && buffer_provided->grid() == pgrid
                   && buffer_provided->global_shape() == gshape
                   && buffer_provided->block_size() == bsize,
                   "scr_coulomb_fourier_t::acquire_ft_buffer: provided buffer shape/dist mismatch");
      return *buffer_provided;
    }
    buffer_own.emplace(math::nda::make_distributed_array<local_Array_t>(*comm, pgrid, gshape, bsize));
    return *buffer_own;
  }

  template<nda::MemoryArrayOfRank<4> local_Array_t, typename communicator_t>
  auto scr_coulomb_fourier_t::tau_to_w(
      memory::darray_t<local_Array_t, communicator_t> &dPi_tqPQ_pos,
      std::array<long, 4> w_pgrid_out, std::array<long, 4> w_bsize_out,
      bool reset_input,
      memory::darray_t<local_Array_t, communicator_t>* buffer_t,
      memory::darray_t<local_Array_t, communicator_t>* buffer_w)
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
      if (_check_ft_leakage) {
        _ft->check_leakage(dPi_tqPQ_pos, imag_axes_ft::boson, "polarizability", true);
      }
      auto dPi_wqPQ = make_distributed_array<local_Array_t>(
          *comm, {1, 1, 1, 1}, w_gshape, dPi_tqPQ_pos.block_size());
      auto Pi_ti_loc = dPi_tqPQ_pos.local();
      auto Pi_wi_loc = dPi_wqPQ.local();
      _ft->tau_to_w_PHsym(Pi_ti_loc, Pi_wi_loc);
      if (reset_input) dPi_tqPQ_pos.reset();
      _Timer.stop("IMAG_FT_TtoW");
      return dPi_wqPQ;
    }
    // redistribute to cover (tau, w)-axes locally -> FT locally -> redistribute back
    // Buffer distributes over q AND PQ (tau/omega axis 0 stays undivided for FT).
    auto [b_pgrid, b_bsize] = ft_buffer_dist(comm->size(), t_gshape);

    using dArr_t = memory::darray_t<local_Array_t, communicator_t>;

    // Acquire the τ-side staging buffer (caller-owned or per-call).
    std::optional<dArr_t> buffer_ti_own;
    auto& buffer_ti = acquire_ft_buffer<local_Array_t>(
        buffer_t, buffer_ti_own, comm, b_pgrid, t_gshape, b_bsize);

    _Timer.start("FT_REDISTRIBUTE");
    math::nda::redistribute(dPi_tqPQ_pos, buffer_ti);
    _Timer.stop("FT_REDISTRIBUTE");
    if (reset_input) dPi_tqPQ_pos.reset();
    if (_check_ft_leakage) {
      _ft->check_leakage(buffer_ti, imag_axes_ft::boson, "polarizability", true);
    }

    // Output array in the requested distribution.
    auto dPi_wqPQ = make_distributed_array<local_Array_t>(
        *comm, w_pgrid_out, w_gshape, w_bsize_out);

    if (w_pgrid_out == b_pgrid && w_bsize_out == b_bsize) {
      // Output distribution == buffer distribution: FT straight into the
      // output, no ω-side staging buffer and no final redistribute.
      auto buf_ti_loc = buffer_ti.local();
      auto Pi_wi_loc = dPi_wqPQ.local();
      _ft->tau_to_w_PHsym(buf_ti_loc, Pi_wi_loc);
      if (buffer_ti_own) buffer_ti_own->reset();
    } else {
      // Output distribution differs from the buffer distribution: FT into the
      // ω-side staging buffer, then redistribute into the output.
      std::optional<dArr_t> buffer_wi_own;
      auto& buffer_wi = acquire_ft_buffer<local_Array_t>(
          buffer_w, buffer_wi_own, comm, b_pgrid, w_gshape, b_bsize);
      {
        auto buf_ti_loc = buffer_ti.local();
        auto buf_wi_loc = buffer_wi.local();
        _ft->tau_to_w_PHsym(buf_ti_loc, buf_wi_loc);
      }
      if (buffer_ti_own) buffer_ti_own->reset();

      _Timer.start("FT_REDISTRIBUTE");
      math::nda::redistribute(buffer_wi, dPi_wqPQ);
      _Timer.stop("FT_REDISTRIBUTE");
      if (buffer_wi_own) buffer_wi_own->reset();
    }

    _Timer.stop("IMAG_FT_TtoW");
    return dPi_wqPQ;
  }

  template<nda::MemoryArrayOfRank<4> local_Array_t, typename communicator_t>
  auto scr_coulomb_fourier_t::w_to_tau(
      memory::darray_t<local_Array_t, communicator_t> &dW_wqPQ_pos,
      std::array<long, 4> t_pgrid_out, std::array<long, 4> t_bsize_out,
      bool reset_input,
      memory::darray_t<local_Array_t, communicator_t>* buffer_t,
      memory::darray_t<local_Array_t, communicator_t>* buffer_w)
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
          *comm, {1, 1, 1, 1}, t_gshape, {1, 1, 1, 1});
      auto W_wi_loc = dW_wqPQ_pos.local();
      auto W_ti_loc = dW_tqPQ.local();
      _ft->w_to_tau_PHsym(W_wi_loc, W_ti_loc);
      if (reset_input) dW_wqPQ_pos.reset();
      if (_check_ft_leakage) {
        _ft->check_leakage(dW_tqPQ, imag_axes_ft::boson, "screened interaction", true);
      }
      _Timer.stop("IMAG_FT_WtoT");
      return dW_tqPQ;
    }

    // redistribute to cover (tau, w)-axes locally -> FT locally -> redistribute back
    // Buffer distributes over q AND PQ (tau/omega axis 0 stays undivided for FT).
    auto [b_pgrid, b_bsize] = ft_buffer_dist(comm->size(), t_gshape);

    using dArr_t = memory::darray_t<local_Array_t, communicator_t>;

    // Acquire the τ-side staging buffer (caller-owned or per-call).
    std::optional<dArr_t> buffer_ti_own;
    auto& buffer_ti = acquire_ft_buffer<local_Array_t>(
        buffer_t, buffer_ti_own, comm, b_pgrid, t_gshape, b_bsize);

    if (dW_wqPQ_pos.grid() == b_pgrid && dW_wqPQ_pos.block_size() == b_bsize) {
      // Input distribution == buffer distribution: FT straight from the input,
      // no ω-side staging buffer and no input redistribute.
      auto buf_wi_loc = dW_wqPQ_pos.local();
      auto buf_ti_loc = buffer_ti.local();
      _ft->w_to_tau_PHsym(buf_wi_loc, buf_ti_loc);
      if (reset_input) dW_wqPQ_pos.reset();
    } else {
      // Input distribution differs from the buffer distribution: redistribute
      // the input into the ω-side staging buffer, then FT.
      std::optional<dArr_t> buffer_wi_own;
      auto& buffer_wi = acquire_ft_buffer<local_Array_t>(
          buffer_w, buffer_wi_own, comm, b_pgrid, w_gshape, b_bsize);

      _Timer.start("FT_REDISTRIBUTE");
      math::nda::redistribute(dW_wqPQ_pos, buffer_wi);
      _Timer.stop("FT_REDISTRIBUTE");
      if (reset_input) dW_wqPQ_pos.reset();

      {
        auto buf_wi_loc = buffer_wi.local();
        auto buf_ti_loc = buffer_ti.local();
        _ft->w_to_tau_PHsym(buf_wi_loc, buf_ti_loc);
      }
      if (buffer_wi_own) buffer_wi_own->reset();
    }

    if (_check_ft_leakage) {
      _ft->check_leakage(buffer_ti, imag_axes_ft::boson, "screened interaction", true);
    }

    auto dW_tqPQ = make_distributed_array<local_Array_t>(
        *comm, t_pgrid_out, t_gshape, t_bsize_out);

    _Timer.start("FT_REDISTRIBUTE");
    math::nda::redistribute(buffer_ti, dW_tqPQ);
    _Timer.stop("FT_REDISTRIBUTE");
    if (buffer_ti_own) buffer_ti_own->reset();

    _Timer.stop("IMAG_FT_WtoT");
    return dW_tqPQ;
  }

  // template instantiations
  using Arr4D = nda::array<ComplexType, 4>;

  template memory::darray_t<Arr4D, mpi3::communicator>
  scr_coulomb_fourier_t::tau_to_w(memory::darray_t<Arr4D, mpi3::communicator> &,
                 std::array<long, 4>, std::array<long, 4>, bool,
                 memory::darray_t<Arr4D, mpi3::communicator>*,
                 memory::darray_t<Arr4D, mpi3::communicator>*);

  template memory::darray_t<Arr4D, mpi3::communicator>
  scr_coulomb_fourier_t::w_to_tau(memory::darray_t<Arr4D, mpi3::communicator> &,
                 std::array<long, 4>, std::array<long, 4>, bool,
                 memory::darray_t<Arr4D, mpi3::communicator>*,
                 memory::darray_t<Arr4D, mpi3::communicator>*);

}  // solvers
}  // methods
