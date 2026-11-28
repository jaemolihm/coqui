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

#include "methods/MBPT_drivers.h"
#include "methods/SCF/lr_setup.hpp"

namespace methods {

std::optional<lr_W_cache> W_cache;

void release_lr_W_cache() { W_cache.reset(); }

lr_iter_params read_lr_iter_params(ptree const& pt, bool qp_static_sigma) {
  lr_iter_params iter_params;
  // 1. alg: GCR by default, DIIS for the qpGW map, whose Padé statification is not affine (lr_setup rejects GCR there)
  iter_params.alg = io::get_value_with_default<std::string>(
      pt, "iter_alg.alg", qp_static_sigma ? std::string("DIIS") : std::string("GCR"));
  utils::check(iter_params.alg == "damping" || iter_params.alg == "DIIS" ||
               iter_params.alg == "GCR",
               "read_lr_iter_params: unknown iter_alg '{}'. Must be 'damping', 'DIIS' or 'GCR'.",
               iter_params.alg);
  // 2. mixing, subspace size and DIIS warmup
  iter_params.mixing = io::get_value_with_default<double>(pt, "iter_alg.mixing", 1.0);
  // GCR: 0 lets lr_setup pick the capacity from the kernel.
  iter_params.max_subsp_size = static_cast<size_t>(io::get_value_with_default<long>(
      pt, "iter_alg.max_subsp_size", iter_params.alg == "GCR" ? 0 : 10));
  iter_params.diis_warmup = static_cast<size_t>(
      io::get_value_with_default<long>(pt, "iter_alg.diis_warmup", 0));
  return iter_params;
}

void fill_lr_kernel_params(lr_params& p, lr_kernel_spec const& kernel, lr_W_inputs const& W,
                           lr_iter_params const& iter_params, long max_iter, double tol, bool fix_density,
                           bool save_DeltaG) {
  p.kernel           = kernel;
  p.max_iter         = max_iter;
  p.tol              = tol;
  p.fix_density      = fix_density;
  // dump_lr writes sDeltaG_tskij only under save_DeltaG.
  p.save_DeltaG      = save_DeltaG;
  p.iter_params      = iter_params;
  p.eps_inv_head     = W.opt_eps_inv ? &(*W.opt_eps_inv) : nullptr;
  p.eps_inv_head_w   = W.opt_eps_inv_w ? &(*W.opt_eps_inv_w) : nullptr;
  p.div_treatment    = W.div_treatment;
  p.hf_div_treatment = W.hf_div_treatment;
}

} // namespace methods
