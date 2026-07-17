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


#ifndef UTILITIES_APP_LOGGERS_HPP
#define UTILITIES_APP_LOGGERS_HPP

#if defined(ENABLE_SPDLOG)
#include "spdlog/spdlog.h"
#include "spdlog/sinks/stdout_color_sinks.h"
#endif
#include "IO/fmt_extensions.hpp"
#include "IO/AppAbort.hpp"

extern int __app_debug_level__;
extern int __app_output_level__;

// currently using 2 separate loggers
// app_log: uses "std_console" with a clean format only on Global().root()
// app_warning/app_error/app_critical/app_debug: use "err_console" shared by everyone
// consider later using app_debug as a separate logger to file per mpi rank

void setup_loggers(bool root=true, int output_level=2, int debug_level=0);
void set_output_level(bool root, int output_level);
void set_debug_level(bool root, int debug_level);
void set_stacktrace(bool stk);

#if defined(ENABLE_SPDLOG)
// Return the named spdlog logger, creating it on-demand if setup_loggers() was
// never called on this rank. Logging must never abort the program: the Python
// API is silent by default (no set_verbosity → no setup_loggers), so a stray
// app_warning/app_log on such a rank would otherwise crash. Mirrors the lazy
// creation already done in APP_ABORT.
inline std::shared_ptr<spdlog::logger> app_get_logger(const char* name,
                                                      const char* pattern)
{
  auto l = spdlog::get(name);
  if(not l) {
    try {
      l = spdlog::stdout_color_mt(name);
      l->set_pattern(pattern);
    } catch(const spdlog::spdlog_ex&) {
      // created concurrently on another thread; fetch the registered instance
      l = spdlog::get(name);
    }
  }
  return l;
}
#endif

template<class... Args>
void app_log(int level, const std::string_view string_format, Args&&... args)
{
  if(__app_output_level__ > 0 and level <= __app_output_level__) {
#if defined(ENABLE_SPDLOG)
    app_get_logger("std_console","%v")->info(string_format,std::forward<Args>(args)...);
#else
    if constexpr (sizeof...(Args) > 0)
      std::cout<<std::vformat(string_format,std::make_format_args(args...)) <<"\n";
    else
      std::cout<<string_format <<"\n";
#endif
  }
}

template<class... Args>
void app_warning(const std::string_view string_format, Args&&... args)
{
#if defined(ENABLE_SPDLOG)
  app_get_logger("warn_console","%^[%l]%$ %v")->warn(string_format,std::forward<Args>(args)...);
#else
  if constexpr (sizeof...(Args) > 0)
    std::cerr<<std::vformat(string_format,std::make_format_args(args...)) <<"\n";
  else
    std::cerr<<string_format <<"\n";
#endif
}

template<class... Args>
void app_error(const std::string_view string_format, Args&&... args)
{
#if defined(ENABLE_SPDLOG)
  {
    auto l = app_get_logger("err_console","%^[%l]%$ %v");
    l->error(string_format,std::forward<Args>(args)...);
    l->flush();
  }
#else
  if constexpr (sizeof...(Args) > 0)
    std::cerr<<std::vformat(string_format,std::make_format_args(args...)) <<"\n";
  else
    std::cerr<<string_format <<"\n";
#endif
}

template<class... Args>
void app_debug(int level, const std::string_view string_format, Args&&... args)
{ 
  if(__app_debug_level__ > 0 and level <= __app_debug_level__) {
#if defined(ENABLE_SPDLOG)
    app_get_logger("err_console","%^[%l]%$ %v")->debug(string_format, std::forward<Args>(args)...);
#else
    if constexpr (sizeof...(Args) > 0)
      std::cerr<<std::vformat(string_format,std::make_format_args(args...)) <<"\n";
    else
      std::cerr<<string_format <<"\n";
#endif
  }
}

inline void app_log_flush() 
{
  if(__app_output_level__ > 0) {
#if defined(ENABLE_SPDLOG)
    app_get_logger("std_console","%v")->flush();
#else
    std::cout.flush();
#endif
  }
}

inline void app_warn_flush()
{
#if defined(ENABLE_SPDLOG)
  app_get_logger("warn_console","%^[%l]%$ %v")->flush();
#else
    std::cerr.flush();
#endif
}

inline void app_error_flush() 
{
#if defined(ENABLE_SPDLOG)
  app_get_logger("err_console","%^[%l]%$ %v")->flush();
#else
    std::cerr.flush();
#endif
}

#endif
