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


#include <array>
#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

#include "configuration.hpp"
#include "IO/app_loggers.h"

#include "methods/SCF/scf_common.hpp"
#include "methods/SCF/scf_mem_report.hpp"
#include "methods/scr_coulomb/scr_coulomb_t.h"
#include "methods/scr_coulomb/scr_coulomb_fourier_t.h"
#include "numerics/distributed_array/nda_utils.hpp"

namespace methods {

namespace {

// Phase groups within one SCF iteration, in execution order. Every transient
// array carries a bitmask of the groups it is alive in, and the peak adds the
// largest group total to the persistent set. A group is exact only as long as
// the allocator free points do not move: dPi_tqPQ / dW_tqPQ / dG_tskPQ /
// dSigma_wskij / dG_wskij / dG_wskij_tmp are explicit reset() calls, while the
// Pi scratch and the whole k-space Sigma set rely on function-scope destruction
// in eval_Pi_rpa_Rspace / eval_Sigma_all_kspace.
// The W Fourier transforms are split into one group per *moment* (the two
// redistributes and the FT in between), not one per call: within a call the
// staging buffers are allocated and released around each step, so summing the
// whole call would double-count. Groups are combined with a max, which is
// exactly the max-over-moments the peak needs.
enum : uint32_t {
  G_PI      = 1u << 0,
  G_T2W_IN  = 1u << 1,
  G_T2W_FT  = 1u << 2,
  G_T2W_OUT = 1u << 3,
  G_W2T_IN  = 1u << 4,
  G_W2T_FT  = 1u << 5,
  G_W2T_OUT = 1u << 6,
  G_SIGMA   = 1u << 7,
  G_MIX     = 1u << 8,
  G_DYSON_W = 1u << 9,
  G_DYSON_T = 1u << 10,
  G_THERMO  = 1u << 11,
  PERSIST   = 0u
};
constexpr int N_GROUPS = 12;

const char* group_tag(int g) {
  switch (g) {
    case 0:  return "Π";
    case 1:  return "W τ→ω in";
    case 2:  return "W τ→ω FT";
    case 3:  return "W τ→ω out";
    case 4:  return "W ω→τ in";
    case 5:  return "W ω→τ FT";
    case 6:  return "W ω→τ out";
    case 7:  return "Σ";
    case 8:  return "mix";
    case 9:  return "Dyson ω";
    case 10: return "Dyson τ";
    default: return "thermo";
  }
}

// How many ranks share one copy of an array: a node-shared window is one copy per
// node, a globally distributed array one copy per comm, a t-pool array one copy per
// (nproc/ntpools) sub-comm, and a plain local array one copy per rank.
enum loc_t { SHARED, DIST, TPOOL, PERRANK };

const char* loc_str(loc_t l) {
  switch (l) {
    case SHARED:  return "shared";
    case DIST:    return "distributed";
    case TPOOL:   return "t-pool";
    default:      return "per-rank";
  }
}

struct entry_t {
  std::string name;
  std::string shape;
  double nelem;
  loc_t loc;
  uint32_t mask;
};

long ceil_div(long a, long b) { return (b > 0) ? (a + b - 1) / b : a; }

// Pad to a display width counting UTF-8 code points, not bytes, so rows carrying
// Π/Σ/τ/ω in a column line up with the ASCII ones.
std::string pad(std::string const& s, size_t w) {
  size_t n = 0;
  for (unsigned char c : s) if ((c & 0xC0) != 0x80) ++n;
  return s + std::string(n < w ? w - n : 0, ' ');
}

} // anonymous namespace

double print_scf_memory_estimate(const utils::mpi_context_t<mpi3::communicator>& mpi,
                               const scf_mem_params& p) {
  const long nproc   = std::max(1, mpi.comm.size());
  const long rpn     = std::max(1, mpi.node_comm.size());
  const long n_nodes = std::max(1, mpi.internode_comm.size());

  // The single t-pool sub-communicator size of the run: Pi splits comm on t_origin
  // into np_P*np_Q, and the k-space Sigma splits dW_tqPQ's comm the same way
  // (asserted equal at thc_gw.icc:269), so one divisor serves every t-pool row.
  auto [pi_pgrid, pi_bsize] =
      solvers::scr_coulomb_t::Pi_tau_proc_grid(nproc, p.nth, p.nqi, p.ns, p.nk);
  const long ntpools  = std::max(1L, pi_pgrid[0]);
  const long np_P     = std::max(1L, pi_pgrid[2]);
  const long np_Q     = std::max(1L, pi_pgrid[3]);
  const long np_tpool = std::max(1L, np_P * np_Q);

  const double bytes_per = static_cast<double>(sizeof(ComplexType));
  const double to_GB = 1.0 / (1024.0 * 1024.0 * 1024.0);

  auto gb = [&](double nelem) { return nelem * bytes_per * to_GB; };

  auto spread = [&](loc_t l) -> long {
    switch (l) {
      case SHARED:  return rpn;
      case DIST:    return nproc;
      case TPOOL:   return np_tpool;
      default:      return 1;     // PERRANK
    }
  };

  auto life_str = [](uint32_t mask) {
    if (mask == PERSIST) return std::string("persistent");
    if (mask == G_THERMO) return std::string("post-loop (thermo)");
    std::string s;
    for (int g = 0; g < N_GROUPS; ++g)
      if (mask & (1u << g)) s += (s.empty() ? "" : ", ") + std::string(group_tag(g));
    return "transient (" + s + ")";
  };

  // Element counts of the recurring shape families (see the shape strings below).
  auto band5 = [&](long n0) { return double(n0) * p.ns * p.nki * p.nb * p.nb; };
  auto band4 = [&]()        { return double(p.ns) * p.nki * p.nb * p.nb; };
  auto aux4  = [&](long n0) { return double(n0) * p.nqi * p.NP * p.NP; };
  auto aux5  = [&](long n0) { return double(n0) * p.ns * p.nqi * p.NP * p.NP; };
  auto auxk4 = [&](long nk) { return double(p.ns) * nk * p.NP * p.NP; };

  auto shp5b = [&](long n0) { return fmt::format("({},{},{},{},{})", n0, p.ns, p.nki, p.nb, p.nb); };
  auto shp4b = [&]()        { return fmt::format("({},{},{},{})", p.ns, p.nki, p.nb, p.nb); };
  auto shp4a = [&](long n0) { return fmt::format("({},{},{},{})", n0, p.nqi, p.NP, p.NP); };
  auto shp5a = [&](long n0) { return fmt::format("({},{},{},{},{})", n0, p.ns, p.nqi, p.NP, p.NP); };
  auto shp4k = [&](long nk) { return fmt::format("({},{},{},{})", p.ns, nk, p.NP, p.NP); };

  std::vector<entry_t> arrays;

  // ---------------- persistent ----------------
  arrays.push_back({"sG_tskij",         shp5b(p.nt_f), band5(p.nt_f), SHARED, PERSIST});
  arrays.push_back({"sSigma_tskij",     shp5b(p.nt_f), band5(p.nt_f), SHARED, PERSIST});
  arrays.push_back({"sF_skij",          shp4b(),       band4(),       SHARED, PERSIST});
  arrays.push_back({"sDm_skij",         shp4b(),       band4(),       SHARED, PERSIST});
  arrays.push_back({"sS_skij (MBState)",shp4b(),       band4(),       SHARED, PERSIST});
  if (p.dump_exchange)
    arrays.push_back({"sK_skij",        shp4b(),       band4(),       SHARED, PERSIST});
  arrays.push_back({"_sH0_skij (dyson)",shp4b(),       band4(),       SHARED, PERSIST});
  arrays.push_back({"_sS_skij (dyson)", shp4b(),       band4(),       SHARED, PERSIST});

  if (p.thc_eri) {
    const long nr = std::max(1, p.n_thc_readers);
    auto times = [&](std::string const& n) {
      return (nr > 1) ? fmt::format("{} (x{})", n, nr) : n;
    };
    double X_nelem = double(p.nsp_basis) * p.nk * p.NP * p.nb;
    arrays.push_back({times("_X_shm"),
                      fmt::format("({},{},{},{})", p.nsp_basis, p.nk, p.NP, p.nb),
                      nr * X_nelem, SHARED, PERSIST});
    if (p.separate_Y)
      arrays.push_back({times("_Y_shm"),
                        fmt::format("({},{},{},{})", p.nsp_basis, p.nk, p.NP, p.nb),
                        nr * X_nelem, SHARED, PERSIST});
    if (p.eri_incore) {
      double Z_nelem = double(p.nqi) * p.NP * p.NP;
      arrays.push_back({times("_dZ"), fmt::format("({},{},{})", p.nqi, p.NP, p.NP),
                        nr * Z_nelem, DIST, PERSIST});
      if (p.have_hf)
        arrays.push_back({"hf_t::_dZ_cache", fmt::format("({},{},{})", p.nqi, p.NP, p.NP),
                          Z_nelem, DIST, PERSIST});
    }
  }

  // In-memory DIIS history: max_subsp trial (F, Sigma) pairs plus max_subsp Sigma
  // residuals, each element-sliced across the global comm (utils::part_map), i.e.
  // one whole copy per run. The engine also keeps the previous accepted state as
  // one more sliced (F, Sigma) pair.
  if (p.iter_alg == "DIIS" and p.diis_storage == "memory" and p.max_subsp > 0) {
    arrays.push_back({"DIIS history (memory)",
                      fmt::format("{} x (|F| + 2|Sigma|)", p.max_subsp),
                      double(p.max_subsp) * (band4() + 2.0 * band5(p.nt_f)),
                      DIST, PERSIST});
    arrays.push_back({"DIIS prev state (striped)", "|F| + |Sigma|",
                      band4() + band5(p.nt_f), DIST, PERSIST});
  }

  // ---------------- transient: RPA polarizability ----------------
  // Only the THC path builds the auxiliary-basis grids; ft_buffer_dist and
  // W_omega_proc_grid have no meaning (and reject NP = 0) without them.
  const bool aux_path = p.thc_eri and p.have_scr;
  std::array<long, 4> ftb_pgrid{}, ftb_bsize{}, wo_pgrid{}, wo_bsize{};
  if (aux_path) {
    // redistribute_alltoallv stages the local block of both endpoints and chunks
    // those buffers to a per-rank byte budget; nchunk follows the largest local
    // block of either endpoint, exactly as nda_utils.hpp computes it.
    const std::array<long, 4> t_gshape = {p.nth, p.nqi, p.NP, p.NP};
    const std::array<long, 4> w_gshape = {p.nwh, p.nqi, p.NP, p.NP};
    std::tie(ftb_pgrid, ftb_bsize) =
        solvers::scr_coulomb_fourier_t::ft_buffer_dist(nproc, t_gshape);
    std::tie(wo_pgrid, wo_bsize) =
        solvers::scr_coulomb_t::W_omega_proc_grid(nproc, p.nqi, p.nw_b, p.NP);
    // When the W(iw) distribution is already the FT buffer distribution, both
    // Fourier calls take their fast branch: tau_to_w transforms straight into
    // W(iw) and w_to_tau straight out of it, so there is no omega-side staging
    // buffer and one redistribute each instead of two.
    const bool w_ft_fast = (wo_pgrid == ftb_pgrid and wo_bsize == ftb_bsize);
    auto max_local = [&](std::array<long, 4> const& gs, std::array<long, 4> const& pg) {
      double n = 1.0;
      for (int i = 0; i < 4; ++i) n *= double(ceil_div(gs[i], std::max(1L, pg[i])));
      return n;
    };
    auto nchunk_of = [&](double m1, double m2) {
      const double cap_elem =
          std::max(1.0, double(math::nda::redistribute_chunk_cap_bytes()) / bytes_per);
      return long(std::min(double(nproc), std::max(1.0, std::ceil(std::max(m1, m2) / cap_elem))));
    };
    const long nchunk_t = nchunk_of(max_local(t_gshape, pi_pgrid),
                                    max_local(t_gshape, ftb_pgrid));
    const long nchunk_w = nchunk_of(max_local(w_gshape, wo_pgrid),
                                    max_local(w_gshape, ftb_pgrid));

    arrays.push_back({"dPi_tqPQ",  shp4a(p.nth), aux4(p.nth), DIST, G_PI | G_T2W_IN});
    arrays.push_back({"dGp_sRPQ",  shp4k(p.nk),  auxk4(p.nk), TPOOL,   G_PI});
    arrays.push_back({"dGn_sRPQ",  shp4k(p.nk),  auxk4(p.nk), TPOOL,   G_PI});
    arrays.push_back({"dbuf_skPQ", shp4k(p.nk - p.nk_trev), auxk4(p.nk - p.nk_trev),
                      TPOOL, G_PI});
    // Rank-local scratch, so its "one copy" is the largest rank's local block.
    const long NP_loc = ceil_div(p.NP, np_P);
    const long NQ_loc = ceil_div(p.NP, np_Q);
    arrays.push_back({"X_R_PQ", fmt::format("({},{})", p.nk, NP_loc * NQ_loc),
                      double(p.nk) * NP_loc * NQ_loc, PERRANK, G_PI});
    arrays.push_back({"sf_Rk", fmt::format("({},{})", p.nk, p.nk),
                      double(p.nk) * p.nk, SHARED, G_PI});
    arrays.push_back({"sf_qR", fmt::format("({},{})", p.nqi, p.nk),
                      double(p.nqi) * p.nk, SHARED, G_PI});

    // ---------------- transient: W tau<->omega FT + W Dyson ----------------
    // Each FT is redistribute -> FT -> redistribute; a staging buffer pair lives
    // only inside a redistribute, and the τ- and ω-side arrays only overlap
    // during the FT itself, so every array carries the moments it is alive in.
    arrays.push_back({"buffer_ti (FT τ)", shp4a(p.nth), aux4(p.nth), DIST,
                      G_T2W_IN | G_T2W_FT | G_W2T_FT | G_W2T_OUT});
    if (!w_ft_fast)
      arrays.push_back({"buffer_wi (FT ω)", shp4a(p.nwh), aux4(p.nwh), DIST,
                        G_T2W_FT | G_T2W_OUT | G_W2T_IN | G_W2T_FT});
    arrays.push_back({"W(iω) (dPi_wqPQ)", shp4a(p.nwh), aux4(p.nwh), DIST,
                      w_ft_fast ? (G_T2W_FT | G_W2T_FT) : (G_T2W_OUT | G_W2T_IN)});
    // redistribute_alltoallv stages the local block of *both* endpoints in
    // contiguous buffers, chunked to a per-rank byte budget. Both endpoints of
    // the τ-side redistributes are aux τ grids and of the ω-side ones aux ω
    // grids, so each pair is 2 x that grid / nchunk.
    arrays.push_back({"redistribute bufs (τ)",
                      fmt::format("2 x {} / {}", shp4a(p.nth), nchunk_t),
                      2.0 * aux4(p.nth) / double(nchunk_t), DIST,
                      G_T2W_IN | G_W2T_OUT});
    if (!w_ft_fast)
      arrays.push_back({"redistribute bufs (ω)",
                        fmt::format("2 x {} / {}", shp4a(p.nwh), nchunk_w),
                        2.0 * aux4(p.nwh) / double(nchunk_w), DIST,
                        G_T2W_OUT | G_W2T_IN});

    // W stays in (t,q) all the way to the Sigma solver, which frees it at
    // scf_driver.cpp -- or keeps it to the end of the run under keep_w.
    uint32_t w_mask = G_W2T_OUT | G_SIGMA;
    if (p.keep_w) w_mask |= G_MIX | G_DYSON_W | G_DYSON_T;
    arrays.push_back({"dW_tqPQ", shp4a(p.nth), aux4(p.nth), DIST, w_mask});
  }

  // ---------------- transient: self-energy ----------------
  if (p.thc_eri and p.have_corr) {
    if (p.sigma_alg == "R") {
      // Exact: Sigma is written over G in place, and sG_ttskij (the tau-reversed
      // copy that feeds primary_to_aux) is alive while dG_tskPQ already exists.
      arrays.push_back({"dG_tskPQ (= dSigma)", shp5a(p.nth), aux5(p.nth), DIST, G_SIGMA});
      arrays.push_back({"sG_ttskij",    shp5b(p.nt_f), band5(p.nt_f), SHARED, G_SIGMA});
    } else {
      // Exact: none of the four is reset inside eval_Sigma_all_kspace.
      arrays.push_back({"dSigma_skPQ",   shp4k(p.nki), auxk4(p.nki), TPOOL, G_SIGMA});
      arrays.push_back({"dSigma_skPQ_2", shp4k(p.nki), auxk4(p.nki), TPOOL, G_SIGMA});
      arrays.push_back({"dG_skPQ",       shp4k(p.nk - p.nk_trev),
                        auxk4(p.nk - p.nk_trev), TPOOL, G_SIGMA});
      arrays.push_back({"sSigma_tskab",  shp5b(ntpools), band5(ntpools), SHARED, G_SIGMA});
    }
  }

  // ---------------- transient: DIIS commutator + Dyson ----------------
  if (p.iter_alg == "DIIS")
    arrays.push_back({"sC_t_dist", shp5b(p.nt_f), band5(p.nt_f), SHARED, G_MIX});

  arrays.push_back({"dSigma_wskij",  shp5b(p.nw_f), band5(p.nw_f), DIST, G_DYSON_W});
  arrays.push_back({"dG_wskij",      shp5b(p.nw_f), band5(p.nw_f), DIST, G_DYSON_W});
  arrays.push_back({"dG_wskij_tmp",  shp5b(p.nw_f), band5(p.nw_f), DIST, G_DYSON_T});
  arrays.push_back({"dG_tskij",      shp5b(p.nt_f), band5(p.nt_f), DIST, G_DYSON_T});

  // ---------------- post-loop: thermodynamics ----------------
  if (p.eval_thermodynamics) {
    arrays.push_back({"spectra", fmt::format("({},{},{},{})", p.nw_f, p.ns, p.nki, p.nb),
                      double(p.nw_f) * p.ns * p.nki * p.nb, PERRANK, G_THERMO});
    arrays.push_back({"sS_inv", shp4b(), band4(), SHARED, G_THERMO});
  }

  // ---------------- roll-up ----------------
  double persist_GB = 0.0, persist_pn = 0.0;
  std::array<double, N_GROUPS> grp_GB{}, grp_pn{};
  grp_GB.fill(0.0);
  grp_pn.fill(0.0);
  for (auto const& a : arrays) {
    const double g  = gb(a.nelem);
    const double pn = g * double(rpn) / double(spread(a.loc));
    if (a.mask == PERSIST) {
      persist_GB += g;
      persist_pn += pn;
    } else {
      for (int i = 0; i < N_GROUPS; ++i)
        if (a.mask & (1u << i)) { grp_GB[i] += g; grp_pn[i] += pn; }
    }
  }
  int peak_grp = 0;
  for (int i = 1; i < N_GROUPS; ++i)
    if (grp_pn[i] > grp_pn[peak_grp]) peak_grp = i;
  const double peak_pn = persist_pn + grp_pn[peak_grp];

  // ---------------- print ----------------
  app_log(2, "\n  SCF memory estimate (arrays ~ nk·nt·n², n ∈ {{nbnd={}, NP={}}})", p.nb, p.NP);
  app_log(2, "  {}", std::string(112, '-'));
  app_log(2, "    {}{}{:>8s}{:>11s}   {}{}", pad("quantity", 26), pad("shape", 26),
          "GB", "GB/node", pad("location", 14), "lifetime");
  for (auto const& a : arrays)
    app_log(2, "    {}{}{:>8.3f}{:>11.3f}   {}{}", pad(a.name, 26), pad(a.shape, 26),
            gb(a.nelem), gb(a.nelem) * double(rpn) / double(spread(a.loc)),
            pad(loc_str(a.loc), 14), life_str(a.mask));
  app_log(2, "  {}", std::string(112, '-'));
  app_log(2, "    persistent:             {:9.3f} GB/node  (total {:.3f} GB over {} node(s))",
          persist_pn, persist_GB, n_nodes);
  for (int i = 0; i < N_GROUPS; ++i)
    if (grp_pn[i] > 0.0)
      app_log(2, "    {}{:9.3f} GB/node  (total {:.3f} GB){}",
              pad(fmt::format("{}:", life_str(1u << i)), 24), grp_pn[i], grp_GB[i],
              (i == peak_grp) ? "  <- peak" : "");
  app_log(2, "    ranks/node = {}, nodes = {}, t-pools = {} (t-pool comm = {} ranks)",
          rpn, n_nodes, ntpools, np_tpool);
  app_log(2, "    not modeled: HF Fock-build scratch (~nk·NP·nb); _Vxc; the W Dyson scratch;"
             " disk-backed DIIS history");
  if (p.n_thc_readers > 1)
    app_log(2, "    the {} THC readers are assumed to share the correlated reader's NP/nbnd/"
               "x_range shapes", p.n_thc_readers);
  if (p.screen_type == "rpa_k")
    app_log(2, "    [not modeled: rpa_k polarizability]");
  if (!p.thc_eri)
    app_log(2, "    [not modeled: Cholesky-ERI resident arrays -- band-basis rows only]");
  if (p.is_gf2)
    app_log(2, "    [not modeled: GF2-specific arrays]");

  app_log(1, "  Estimated SCF memory (persistent): {:.3f} GB/node", persist_pn);
  app_log(1, "  Estimated SCF memory (peak):       {:.3f} GB/node  [{} dominates]",
          peak_pn, group_tag(peak_grp));
  app_log(2, "");

  // ---------------- distribution patterns ----------------
  auto pg4 = [](const std::array<long, 4>& g, const char* ax) {
    return fmt::format("{}=({},{},{},{})", ax, g[0], g[1], g[2], g[3]); };
  auto pg5 = [](const std::array<long, 5>& g, const char* ax) {
    return fmt::format("{}=({},{},{},{},{})", ax, g[0], g[1], g[2], g[3], g[4]); };
  auto bs4 = [](const std::array<long, 4>& b) {
    return fmt::format("({},{},{},{})", b[0], b[1], b[2], b[3]); };
  auto bs5 = [](const std::array<long, 5>& b) {
    return fmt::format("({},{},{},{},{})", b[0], b[1], b[2], b[3], b[4]); };

  auto row = [&](std::string const& pattern, std::string const& grid,
                 std::string const& bsize, std::string const& arrs) {
    app_log(2, "    {}{}{}{}", pad(pattern, 22), pad(grid, 30), pad(bsize, 18), arrs);
  };

  app_log(2, "\n  SCF distribution patterns (nproc = {}):", nproc);
  app_log(2, "  {}", std::string(112, '-'));
  row("pattern", "pgrid", "tiles", "arrays");

  if (aux_path) {
    row("aux τ (q-local)", pg4(pi_pgrid, "(t,q,P,Q)"), bs4(pi_bsize),
        "dPi_tqPQ, dW_tqPQ");
    row("aux G^R pool", fmt::format("(s,R,P,Q)=(1,1,{},{})", np_P, np_Q), "(1,1,1,1)",
        "dGp_sRPQ, dGn_sRPQ, dbuf_skPQ  [on t_intra_comm]");
    row("aux FT buffer", pg4(ftb_pgrid, "(·,q,P,Q)"), bs4(ftb_bsize),
        "buffer_ti, buffer_wi");
    row("aux ω (W Dyson)", pg4(wo_pgrid, "(w,q,P,Q)"), bs4(wo_bsize), "W(iω) (dPi_wqPQ)");
  }

  if (p.thc_eri and p.have_corr) {
    // The Sigma solver takes dW_tqPQ on the aux tau grid and forces qpools to 1.
    if (p.sigma_alg == "R") {
      std::array<long, 5> s_pgrid = {ntpools, 1, 1, np_P, np_Q};
      std::array<long, 5> s_bsize = {pi_bsize[0], 0, pi_bsize[1],
                                     pi_bsize[2], pi_bsize[3]};
      row("aux Σ(τ), R-space", pg5(s_pgrid, "(t,s,k,P,Q)"), bs5(s_bsize),
          "dG_tskPQ, dSigma_tskPQ");
    } else {
      std::array<long, 4> k_pgrid = {1, 1, np_P, np_Q};
      std::array<long, 4> k_bsize = {0, pi_bsize[1], pi_bsize[2], pi_bsize[3]};
      row("aux Σ, k-space", pg4(k_pgrid, "(s,k,P,Q)"), bs4(k_bsize),
          "dSigma_skPQ, dG_skPQ  [on t_intra_comm]");
      // Mirrors thc_gw.icc:277-279 (the aux->primary redistribute target), whose
      // input comm is t_intra_comm, i.e. nproc/ntpools ranks.
      long qpools_2 = utils::find_proc_grid_max_npools(np_tpool, p.nki, 0.2);
      long np_P_2 = utils::find_proc_grid_min_diff(np_tpool / qpools_2, 1, 1);
      long np_Q_2 = (np_tpool / qpools_2) / np_P_2;
      row("aux Σ→primary, k", fmt::format("(t,q,P,Q)=({},{},{},{})",
                                          ntpools, qpools_2, np_P_2, np_Q_2),
          "default", "dSigma_skPQ_2");
    }
  }

  auto [dyw_pgrid, dyw_bsize] = dyson_omega_proc_grid(nproc, p.nw_f, p.nki, int(p.nb));
  row("band Dyson(ω)", pg5(dyw_pgrid, "(w,s,k,i,j)"), bs5(dyw_bsize),
      "dSigma_wskij, dG_wskij");
  auto dyt_pgrid = dyson_tau_proc_grid(nproc, p.nki);
  row("band Dyson(τ)", pg5(dyt_pgrid, "(·,·,k,i,j)"), "default",
      "dG_wskij_tmp, dG_tskij");

  app_log(2, "  {}", std::string(112, '-'));
  app_log(2, "    (aux & band arrays are distributed over comm unless marked t_intra_comm; "
             "shared arrays are node-replicated)\n");

  return peak_pn;
}

} // methods
