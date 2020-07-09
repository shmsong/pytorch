#include <torch/csrc/jit/codegen/cuda/kernel_arg.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

TensorArgAbstract* getTensorArg(c10::ScalarType dtype, int nDims) {
  switch (dtype) {
    case (at::kFloat):
      return getTensorArg<float>(nDims);
    case (at::kHalf):
      return getTensorArg<at::Half>(nDims);
    case (at::kBool):
      return getTensorArg<bool>(nDims);
    default:
      TORCH_CHECK(
          false,
          "Dtype: ",
          dtype,
          " not currently supported in code generated kernels.");
  }
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
