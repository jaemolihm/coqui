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

/**
 * Per-process memory reporter (reads /proc/self/status and /proc/self/maps).
 *
 * Differs from utils::freemem (sysinfo) by reporting THIS process's RSS,
 * high-water mark, and bytes mapped from /dev/shm. Designed for diagnosing
 * memory growth across repeated calls (e.g. successive run_lr invocations).
 *
 * Gated only by app_log verbosity: memlog() prints at verbosity >= 3 on root.
 * The /proc reads are skipped when the check fails, so there is zero overhead
 * in default runs (verbosity = 2) and on every non-root rank.
 *
 * Use:
 *     #include "utilities/proc_meminfo.hpp"
 *     utils::memlog("after make_shared_array sG_tskij");
 */
#ifndef COQUI_UTILITIES_PROC_MEMINFO_HPP
#define COQUI_UTILITIES_PROC_MEMINFO_HPP

#include <cstring>
#include <fstream>
#include <string>

#include "IO/app_loggers.h"

namespace utils {

struct proc_mem_t {
  long vm_rss_kb = 0;     // resident set size (current)
  long vm_hwm_kb = 0;     // resident high-water mark
  long vm_data_kb = 0;    // size of data + heap
  long vm_size_kb = 0;    // total virtual size
  long shm_bytes = 0;     // bytes mapped from /dev/shm (per-process view)
  int  shm_count = 0;     // number of /dev/shm regions

  // Node-wide values from /proc/meminfo. Same for every rank on the node;
  // tells you how close the whole node is to OOM regardless of which rank
  // logs it.
  long mem_total_kb = 0;     // total RAM
  long mem_free_kb = 0;      // free RAM (excludes buffers/cache)
  long mem_available_kb = 0; // free RAM that can be made available without swapping
  long mem_shmem_kb = 0;     // total memory used by tmpfs/shm system-wide
};

inline proc_mem_t read_proc_mem() {
  proc_mem_t m;
#ifdef __linux__
  {
    // Read line-by-line. Streamed `f >> key >> val >> unit` breaks the
    // stream on non-numeric lines like "State: S (sleeping)" and silently
    // stops reading the rest of the file — so VmRSS often comes back as 0.
    std::ifstream f("/proc/self/status");
    std::string line;
    while (std::getline(f, line)) {
      auto parse_kb = [&](const char* tag) -> long {
        size_t n = std::strlen(tag);
        if (line.compare(0, n, tag) != 0) return -1;
        try { return std::stol(line.substr(n)); } catch (...) { return -1; }
      };
      long v;
      if      ((v = parse_kb("VmRSS:"))  >= 0) m.vm_rss_kb  = v;
      else if ((v = parse_kb("VmHWM:"))  >= 0) m.vm_hwm_kb  = v;
      else if ((v = parse_kb("VmData:")) >= 0) m.vm_data_kb = v;
      else if ((v = parse_kb("VmSize:")) >= 0) m.vm_size_kb = v;
    }
  }
  {
    std::ifstream f("/proc/self/maps");
    std::string line;
    while (std::getline(f, line)) {
      if (line.find("/dev/shm") == std::string::npos) continue;
      auto dash = line.find('-');
      auto sp = line.find(' ');
      if (dash == std::string::npos || sp == std::string::npos || dash >= sp) continue;
      try {
        long a = std::stol(line.substr(0, dash), nullptr, 16);
        long b = std::stol(line.substr(dash + 1, sp - dash - 1), nullptr, 16);
        m.shm_bytes += (b - a);
        m.shm_count += 1;
      } catch (...) {
        // ignore malformed lines
      }
    }
  }
  // Node-wide memory: /proc/meminfo is the kernel's global view, identical
  // for every process on the node. Cheaper than aggregating across ranks.
  {
    std::ifstream f("/proc/meminfo");
    std::string line;
    while (std::getline(f, line)) {
      auto parse_kb = [&](const char* tag) -> long {
        size_t n = std::strlen(tag);
        if (line.compare(0, n, tag) != 0) return -1;
        try { return std::stol(line.substr(n)); } catch (...) { return -1; }
      };
      long v;
      if      ((v = parse_kb("MemTotal:"))     >= 0) m.mem_total_kb = v;
      else if ((v = parse_kb("MemFree:"))      >= 0) m.mem_free_kb = v;
      else if ((v = parse_kb("MemAvailable:")) >= 0) m.mem_available_kb = v;
      else if ((v = parse_kb("Shmem:"))        >= 0) m.mem_shmem_kb = v;
    }
  }
#endif
  return m;
}

inline void memlog(std::string const& tag, int io_lvl = 3) {
  // Skip the /proc reads if the message would not be printed anyway. Mirrors
  // the gating inside app_log() so we pay zero cost when not logging.
  if (not __app_is_root__ || __app_verbosity__ <= 0 || io_lvl > __app_verbosity__) return;
  auto m = read_proc_mem();
  // Per-process line: rank-0's heap and shm view.
  app_log(io_lvl,
          "[MEM] {:<48s} VmRSS={:.3f}GB VmHWM={:.3f}GB VmData={:.3f}GB "
          "shm={:.3f}GB(n{})",
          tag,
          m.vm_rss_kb / 1048576.0,
          m.vm_hwm_kb / 1048576.0,
          m.vm_data_kb / 1048576.0,
          m.shm_bytes / double(1ll << 30),
          m.shm_count);
  // Node-wide line: how much RAM the whole node has free. MemAvailable is
  // the kernel's best estimate of how much we can still allocate without
  // swapping; MemUsed = MemTotal - MemAvailable shows how close we are to OOM.
  double total_gb = m.mem_total_kb / 1048576.0;
  double avail_gb = m.mem_available_kb / 1048576.0;
  double used_gb  = total_gb - avail_gb;
  double shmem_gb = m.mem_shmem_kb / 1048576.0;
  app_log(io_lvl,
          "[MEM-NODE] {:<43s} used={:.1f}GB / total={:.1f}GB "
          "(avail={:.1f}GB, shmem={:.1f}GB)",
          tag, used_gb, total_gb, avail_gb, shmem_gb);
  app_log_flush();
}

}  // namespace utils

#endif  // COQUI_UTILITIES_PROC_MEMINFO_HPP
