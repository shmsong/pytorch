
#include <torch/csrc/jit/codegen/cuda/instrumentation.h>

#include <cassert>

namespace torch {
namespace jit {
namespace fuser {

Trace::Trace() {
  const char* trace_filename = getenv("PYTORCH_CUDA_FUSER_TRACE");
  if (trace_filename != nullptr) {
    log_file_ = fopen(trace_filename, "w");
    assert(log_file_ != nullptr);
    fprintf(log_file_, "[\n");
    start_timestamp_ = Clock::now();
    logEvent('I', "TRACE_START");
  }
}

Trace::~Trace() {
  if (log_file_ != nullptr) {
    logEvent('I', "TRACE_END", ' ');
    fprintf(log_file_, "]\n");
    fclose(log_file_);
  }
}

void Trace::logEvent(char ph, const char* name, char sep) {
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

} // namespace fuser
} // namespace jit
} // namespace torch
