
#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir.h>

#include <vector>

namespace torch {
namespace jit {
namespace fuser {

class TORCH_CUDA_API Kernel final {
 public:
  void print() const;
};

} // namespace fuser
} // namespace jit
} // namespace torch
