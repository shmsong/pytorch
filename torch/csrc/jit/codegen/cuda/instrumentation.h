
#pragma once

#include <torch/csrc/jit/codegen/cuda/utils.h>

#include <stdio.h>
#include <chrono>

namespace torch {
namespace jit {
namespace fuser {

class Trace : public NonCopyable {
 public:
  using Clock = std::chrono::steady_clock;

 public:
  static Trace* instance() {
    static Trace trace;
    return &trace;
  }

  void beginEvent(const char* name) {
    if (log_file_ != nullptr) {
      logEvent('B', name);
    }
  }

  void endEvent(const char* name) {
    if (log_file_ != nullptr) {
      logEvent('E', name);
    }
  }

 private:
  Trace();
  ~Trace();

  void logEvent(char ph, const char* name, char sep = ',');

 private:
  FILE* log_file_ = nullptr;
  Clock::time_point start_timestamp_;
};

class TraceScope : public NonCopyable {
 public:
  explicit TraceScope(const char* event_name) : event_name_(event_name) {
    Trace::instance()->beginEvent(event_name_);
  }

  ~TraceScope() {
    Trace::instance()->endEvent(event_name_);
  }

 private:
  const char* event_name_ = nullptr;
};

#define FUSER_MACRO_CONCAT2(a, b) a##b
#define FUSER_MACRO_CONCAT(a, b) FUSER_MACRO_CONCAT2(a, b)
#define FUSER_ANONYMOUS(prefix) FUSER_MACRO_CONCAT(prefix, __COUNTER__)

#if 0
#define FUSER_PERF_SCOPE(name) \
  fuser::TraceScope FUSER_ANONYMOUS(_perf_scope_)(name)
#else
#define FUSER_PERF_SCOPE(name)
#endif

} // namespace fuser
} // namespace jit
} // namespace torch
