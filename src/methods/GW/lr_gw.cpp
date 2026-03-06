/**
 * ==========================================================================
 * CoQuí: Correlated Quantum ínterface
 *
 * Copyright (c) 2022-2025 Simons Foundation & The CoQuí developer team
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


#include "nda/nda.hpp"
#include "numerics/distributed_array/nda.hpp"
#include "numerics/shared_array/nda.hpp"

#include "IO/app_loggers.h"
#include "utilities/mpi_context.h"
#include "utilities/Timer.hpp"
#include "utilities/lr_utils.hpp"

#include "mean_field/MF.hpp"
#include "numerics/imag_axes_ft/IAFT.hpp"
#include "methods/ERI/detail/concepts.hpp"
#include "methods/ERI/div_treatment_e.hpp"
#include "methods/GW/lr_gw.hpp"

#include "nda/blas.hpp"
#include "numerics/nda_functions.hpp"
#include "numerics/sparse/csr_blas.hpp"
#include "utilities/kpoint_utils.hpp"
#include "methods/HF/thc_solver_comm.hpp"
#include "methods/GW/g0_div_utils.hpp"

#include "methods/ERI/thc_reader_t.hpp"
#include "methods/GW/gw_t.h"
// Include thc_gw.icc for Sigma_div_correction definition (template in gw_t)
#include "methods/GW/thc_gw.icc"
#include "methods/GW/lr_gw.icc"

namespace methods {
  namespace solvers {

    lr_gw::lr_gw(const imag_axes_ft::IAFT *ft,
                 nda::array<double, 1> const& q_pert,
                 div_treatment_e div)
      : _q_pert(q_pert),
        _gw(ft, div),
        _Timer() {

      for (auto& v: {"EVALUATE_SIGMA_R", "EVALUATE_SIGMA_DELTAW",
                      "SIGMA_ALLOC", "SIGMA_TRANSPOSE",
                      "SIGMA_PRIM_TO_AUX", "SIGMA_FT_R",
                      "SIGMA_HADPROD_R", "SIGMA_AUX_TO_PRIM"}) {
        _Timer.add(v);
      }
    }

    // Template instantiations
    using Arr4D = nda::array<ComplexType, 4>;
    using Arr = nda::array<ComplexType, 5>;
    using Arrv = nda::array_view<ComplexType, 5>;

    template void lr_gw::evaluate_sigma_DeltaG(
        sArray_t<Arrv>&, const Arr&,
        memory::darray_t<Arr4D, mpi3::communicator>&,
        thc_reader_t&);
    template void lr_gw::evaluate_sigma_DeltaG(
        sArray_t<Arrv>&, const Arrv&,
        memory::darray_t<Arr4D, mpi3::communicator>&,
        thc_reader_t&);

    template void lr_gw::evaluate_sigma_DeltaW(
        sArray_t<Arrv>&, const Arr&,
        memory::darray_t<Arr4D, mpi3::communicator>&,
        thc_reader_t&);
    template void lr_gw::evaluate_sigma_DeltaW(
        sArray_t<Arrv>&, const Arrv&,
        memory::darray_t<Arr4D, mpi3::communicator>&,
        thc_reader_t&);

  } // solvers
} // methods
