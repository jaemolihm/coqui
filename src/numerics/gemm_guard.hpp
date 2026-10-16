#ifndef COQUI_NUMERICS_GEMM_GUARD_HPP
#define COQUI_NUMERICS_GEMM_GUARD_HPP

#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>

#include "nda/blas.hpp"

namespace math::detail {

  /**
   * Shared guard of gemm_guarded/gemv_guarded for local blocks that may be empty.
   * targs = std::tie(args...) of gemm(a,b,c)/gemv(a,x,y) (nargs = 3) or
   * gemm(alpha,a,b,beta,c)/gemv(alpha,a,x,beta,y) (nargs = 5); the output is
   * always the last argument. Returns true when the BLAS call must be skipped:
   * - output empty: nothing to write.
   * - contracted extent k empty: the output is set to beta*output (beta = 0
   *   for the 3-argument form), which is what BLAS defines.
   * Host memory only: the beta*output update is an nda expression assignment,
   * which nda refuses on device arrays.
   */
  template<std::size_t nargs, typename Tuple>
  bool skip_empty_blas_call(Tuple const& targs) {
    auto& out = std::get<nargs - 1>(targs);
    if (out.size() == 0) return true;
    auto const& a = std::get<(nargs == 5 ? 1 : 0)>(targs);
    if (a.shape()[1] != 0) return false;
    using value_t = ::nda::get_value_t<std::decay_t<decltype(out)>>;
    value_t beta = value_t(0);
    if constexpr (nargs == 5) beta = value_t(std::get<3>(targs));
    if (beta == value_t(0)) out() = value_t(0);
    else out *= beta;
    return true;
  }

  /**
   * Guarded wrapper around nda::blas::gemm for local blocks that may be empty.
   *
   * Distributed contractions hand a processor an empty local block whenever a
   * processor-grid factor exceeds the axis length (e.g. np_P/np_Q > Np,
   * tpools > nt_half). BLAS then sees a leading dimension of 0, which MKL
   * rejects ("Parameter 10 was incorrect on entry to ZGEMM") without touching C.
   * See skip_empty_blas_call for what happens instead (host memory only).
   *
   * Accepts gemm(a,b,c) and gemm(alpha,a,b,beta,c); the 3-argument form calls
   * nda's gemm(1,a,b,0,c).
   */
  template<typename... Args>
  void gemm_guarded(Args&&... args) {
    constexpr std::size_t nargs = sizeof...(Args);
    static_assert(nargs == 3 or nargs == 5, "gemm_guarded: gemm(a,b,c) or gemm(alpha,a,b,beta,c).");
    auto targs = std::tie(args...);
    if (skip_empty_blas_call<nargs>(targs)) return;
    if constexpr (nargs == 5) {
      ::nda::blas::gemm(std::forward<Args>(args)...);
    } else {
      auto& c = std::get<2>(targs);
      using value_t = ::nda::get_value_t<std::decay_t<decltype(c)>>;
      ::nda::blas::gemm(value_t(1), std::get<0>(targs), std::get<1>(targs), value_t(0), c);
    }
  }

}

#endif // COQUI_NUMERICS_GEMM_GUARD_HPP
