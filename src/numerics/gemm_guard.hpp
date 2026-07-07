#ifndef COQUI_NUMERICS_GEMM_GUARD_HPP
#define COQUI_NUMERICS_GEMM_GUARD_HPP

#include <tuple>
#include <utility>

#include "nda/blas.hpp"

namespace math::detail {

  /**
   * Guarded wrapper around nda::blas::gemm.
   *
   * Distributed THC contractions can hand a processor an empty local block
   * whenever a processor-grid dimension exceeds the axis length (e.g.
   * np_P/np_Q > Np, tpools > nt_half, qpools > nqpts_ibz). The empty extent
   * lands in the result matrix C, so the multiply has nothing to write, yet the
   * BLAS call still passes ldb=0 and MKL rejects it with "Parameter 10 was
   * incorrect on entry to ZGEMM". Since the call is a genuine no-op (C has no
   * elements), we skip it: results are unchanged and the spurious error — tens
   * of thousands of lines per scGW run — disappears.
   *
   * C is always the last argument, for both gemm(a,b,c) and the explicit
   * gemm(alpha,a,b,beta,c) overload.
   */
  template<typename... Args>
  void gemm_guarded(Args&&... args) {
    auto& c = std::get<sizeof...(Args) - 1>(std::tie(args...));
    if (c.size() == 0) return;
    ::nda::blas::gemm(std::forward<Args>(args)...);
  }

}

#endif // COQUI_NUMERICS_GEMM_GUARD_HPP
