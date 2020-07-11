
#include <torch/csrc/jit/codegen/cuda/type.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

class TORCH_CUDA_API LaunchParams {
 public:
  unsigned int smem() const {
    return smem_;
  }
  unsigned int nBlocks() const {
    return gdimx_ * gdimy_ * gdimz_;
  }

  unsigned int nThreads() const {
    return bdimx_ * bdimy_ * bdimz_;
  }

  unsigned int bdimx() const {
    return static_cast<unsigned int>(bdimx_ == -1 ? 1 : bdimx_);
  }

  unsigned int gdimx() const {
    return static_cast<unsigned int>(gdimx_ == -1 ? 1 : gdimx_);
  }

  unsigned int bdimy() const {
    return static_cast<unsigned int>(bdimy_ == -1 ? 1 : bdimy_);
  }

  unsigned int gdimy() const {
    return static_cast<unsigned int>(gdimy_ == -1 ? 1 : gdimy_);
  }

  unsigned int bdimz() const {
    return static_cast<unsigned int>(bdimz_ == -1 ? 1 : bdimz_);
  }

  unsigned int gdimz() const {
    return static_cast<unsigned int>(gdimz_ == -1 ? 1 : gdimz_);
  }

  void checkAndSet(
      const int64_t incoming_val,
      int& class_val,
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

  void bind(int64_t val, ParallelType p_dim);

 private:
  // Spell them out because I want signed ints to know if they were initialized
  // or not.
  // TODO: convert to c10::optional
  int gdimx_ = -1;
  int gdimy_ = -1;
  int gdimz_ = -1;
  int bdimx_ = -1;
  int bdimy_ = -1;
  int bdimz_ = -1;

  unsigned int smem_ = 0;

  // TODO: Fill in output sizes
  std::vector<std::vector<int64_t>> output_sizes;
};
} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch