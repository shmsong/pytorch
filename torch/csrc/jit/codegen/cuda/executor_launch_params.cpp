#include <torch/csrc/jit/codegen/cuda/executor_launch_params.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

void LaunchParams::bind(int64_t val, ParallelType p_type) {
  switch (p_type) {
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
          p_type);
  }
}

int64_t LaunchParams::getDim(ParallelType p_type) const {
  switch (p_type) {
    case (ParallelType::TIDx):
      return bdimx();
      break;
    case (ParallelType::BIDx):
      return gdimx();
      break;
    case (ParallelType::TIDy):
      return bdimy();
      break;
    case (ParallelType::BIDy):
      return gdimy();
      break;
    case (ParallelType::TIDz):
      return bdimz();
      break;
    case (ParallelType::BIDz):
      return gdimz();
      break;
    default:
      TORCH_INTERNAL_ASSERT(
          false,
          "Tried to get with invalid parallel type in launch config: ",
          p_type);
  }
}

bool LaunchParams::hasDim(ParallelType p_type) const {
  return getRawVal(p_type) != -1;
}

const int64_t& LaunchParams::getRawVal(ParallelType p_type) const {
  switch (p_type) {
    case (ParallelType::TIDx):
      return bdimx_;
      break;
    case (ParallelType::BIDx):
      return gdimx_;
      break;
    case (ParallelType::TIDy):
      return bdimy_;
      break;
    case (ParallelType::BIDy):
      return gdimy_;
      break;
    case (ParallelType::TIDz):
      return bdimz_;
      break;
    case (ParallelType::BIDz):
      return gdimz_;
      break;
    default:
      TORCH_INTERNAL_ASSERT(
          false,
          "Tried to get with invalid parallel type in launch config: ",
          p_type);
  }
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch