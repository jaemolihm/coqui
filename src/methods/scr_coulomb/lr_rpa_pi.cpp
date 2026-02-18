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

#include "utilities/mpi_context.h"
#include "utilities/Timer.hpp"
#include "utilities/lr_utils.hpp"

#include "mean_field/MF.hpp"
#include "methods/ERI/detail/concepts.hpp"
#include "methods/scr_coulomb/lr_rpa_pi.hpp"
#include "methods/scr_coulomb/lr_rpa_pi.icc"

#include "methods/ERI/thc_reader_t.hpp"

namespace methods {
  namespace solvers {

    lr_rpa_pi::lr_rpa_pi(nda::array<double, 1> const& q_pert)
      : _q_pert(q_pert),
        _Timer() {

      for (auto& v: {"EVALUATE_LR_PI", "PI_LR_ALLOC", "PI_LR_PRIM_TO_AUX",
                      "PI_LR_FT_R", "PI_LR_HADPROD"}) {
        _Timer.add(v);
      }
    }

    // Template instantiations
    using Arr4D = nda::array<ComplexType, 4>;
    using Arr = nda::array<ComplexType, 5>;
    using Arrv = nda::array_view<ComplexType, 5>;

    template auto lr_rpa_pi::evaluate_lr_Pi<Arr4D>(
        const Arr&, const Arr&,
        thc_reader_t&) -> memory::darray_t<Arr4D, mpi3::communicator>;
    template auto lr_rpa_pi::evaluate_lr_Pi<Arr4D>(
        const Arrv&, const Arrv&,
        thc_reader_t&) -> memory::darray_t<Arr4D, mpi3::communicator>;

  } // solvers
} // methods
