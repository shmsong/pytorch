#pragma once

#include <ATen/core/ivalue.h>
#include <torch/csrc/jit/codegen/cuda/fusion.h>

namespace torch {
namespace jit {
namespace fuser {

struct TORCH_CUDA_API Scheduler {
 public:
  static c10::optional<std::tuple<int,int,int,int>>
  reduction(Fusion *fusion, const at::ArrayRef<IValue> &inputs);
};

} // namespace fuser
} // namespace jit
} // namespace torch
