#include <torch/csrc/jit/codegen/cuda/executor_launch_params.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

void LaunchParams::bind(int64_t val, ParallelType p_dim) {
  switch (p_dim) {
    case (ParallelType::TIDx):
      checkAndSet(val, bdimx_, "blockDim.x");
      break;
    case (ParallelType::BIDx):
      checkAndSet(val, gdimx_, "gridDim.x");
      break;
    case (ParallelType::TIDy):
      checkAndSet(val, bdimy_, "blockDim.y");
      break;
    case (ParallelType::BIDy):
      checkAndSet(val, gdimy_, "gridDim.y");
      break;
    case (ParallelType::TIDz):
      checkAndSet(val, bdimz_, "blockdim.z");
      break;
    case (ParallelType::BIDz):
      checkAndSet(val, gdimz_, "gridDim.z");
      break;
    default:
      TORCH_INTERNAL_ASSERT(
          false,
          "Tried to bind invalid parallel type in launch config: ",
          p_dim);
  }
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch