#pragma once
#include <torch/csrc/jit/codegen/cuda/type.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

class TORCH_CUDA_API LaunchParams {
 public:
  LaunchParams(
      int64_t gdimx = -1,
      int64_t gdimy = -1,
      int64_t gdimz = -1,
      int64_t bdimx = -1,
      int64_t bdimy = -1,
      int64_t bdimz = -1)
      : gdimx_(gdimx),
        gdimy_(gdimy),
        gdimz_(gdimz),
        bdimx_(bdimx),
        bdimy_(bdimy),
        bdimz_(bdimz) {}

  int64_t smem() const {
    return smem_;
  }
  int64_t nBlocks() const {
    return gdimx_ * gdimy_ * gdimz_;
  }

  int64_t nThreads() const {
    return bdimx_ * bdimy_ * bdimz_;
  }

  int64_t bdimx() const {
    return static_cast<int64_t>(bdimx_ == -1 ? 1 : bdimx_);
  }

  int64_t gdimx() const {
    return static_cast<int64_t>(gdimx_ == -1 ? 1 : gdimx_);
  }

  int64_t bdimy() const {
    return static_cast<int64_t>(bdimy_ == -1 ? 1 : bdimy_);
  }

  int64_t gdimy() const {
    return static_cast<int64_t>(gdimy_ == -1 ? 1 : gdimy_);
  }

  int64_t bdimz() const {
    return static_cast<int64_t>(bdimz_ == -1 ? 1 : bdimz_);
  }

  int64_t gdimz() const {
    return static_cast<int64_t>(gdimz_ == -1 ? 1 : gdimz_);
  }

  void checkAndSet(
      const int64_t incoming_val,
      int64_t& class_val,
      std::string val) {
    TORCH_INTERNAL_ASSERT(
        class_val == -1 || incoming_val == 1 || class_val == 1 ||
            incoming_val == class_val,
        "Tried to set ",
        val,
        " to ",
        incoming_val,
        ", but it was already set and new value does not match.",
        " Thread dims all have to be bound to the same value.");
    TORCH_CHECK(
        incoming_val > 0,
        "Received a thread binding on ",
        val,
        " that is ",
        incoming_val,
        ". Cannot create negative threads.");
    if (class_val == -1 || class_val == 1) {
      class_val = incoming_val;
    }
  }

  // Binds dim assocaited with p_type to val
  void bind(int64_t val, ParallelType p_type);

  // Adjusted value based on get functions above for each value
  int64_t getDim(ParallelType p_type) const;

  // Returns raw value which may be -1
  const int64_t& getRawVal(ParallelType p_type) const;

  // Returns false if value associated with p_type == -1
  bool hasDim(ParallelType p_type) const;

 private:
  // Spell them out because I want signed ints to know if they were initialized
  // or not.
  // TODO: convert to c10::optional
  int64_t gdimx_ = -1;
  int64_t gdimy_ = -1;
  int64_t gdimz_ = -1;
  int64_t bdimx_ = -1;
  int64_t bdimy_ = -1;
  int64_t bdimz_ = -1;

  int64_t smem_ = 0;

  // TODO: Fill in output sizes
  std::vector<std::vector<int64_t>> output_sizes;
};
} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch