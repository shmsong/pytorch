
#pragma once

#include <torch/csrc/jit/codegen/cuda/utils.h>

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <thread>

namespace torch {
namespace jit {
namespace fuser {

class Trace : public NonCopyable {
 public:
  using Clock = std::chrono::steady_clock;

 public:
  Trace() {
    const char* trace_filename = getenv("PYTORCH_CUDA_FUSER_TRACE");
    if (trace_filename != nullptr) {
      log_file_ = fopen(trace_filename, "w");
      assert(log_file_ != nullptr);
      fprintf(log_file_, "[\n");
      start_timestamp_ = Clock::now();
      logEvent('I', "TRACE_START");
    }
  }

  ~Trace() {
    if (log_file_ != nullptr) {
      logEvent('I', "TRACE_END", ' ');
      fprintf(log_file_, "]\n");
      fclose(log_file_);
    }
  }

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
  void logEvent(char ph, const char* name, char sep = ',') {
    const std::chrono::duration<double> d = Clock::now() - start_timestamp_;
    const double elapsed = d.count() * 1e6;
    const unsigned int pid = 0;
    const unsigned int tid = 0;
    fprintf(
        log_file_,
        "{ \"name\": \"%s\", \"ph\": \"%c\", \"pid\": \"%u\", \"tid\": \"%u\", \"ts\": \"%.0f\" }%c\n",
        name,
        ph,
        pid,
        tid,
        elapsed,
        sep);
  }

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

#define FUSER_PERF_SCOPE(name) \
  fuser::TraceScope FUSER_ANONYMOUS(_perf_scope_)(name)

} // namespace fuser
} // namespace jit
} // namespace torch
