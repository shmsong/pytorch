#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h>

#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>

#include <ostream>

namespace torch {
namespace jit {
namespace fuser {

class TORCH_CUDA_API GPULower {
 public:
  // Init printer on ostream
  GPULower(Fusion* _fusion) : fusion_(_fusion) {
    lower();
  }

  GPULower() = default;
  GPULower(const GPULower& lower) = default;
  GPULower& operator=(const GPULower& other) = default;

  // print generated code to ostream
  std::vector<Expr*> lowered_exprs();

  std::ostream& printKernel(
      std::ostream& _os,
      const std::string& kernel_name = "CUDAGeneratedKernel");

  std::string getKernel(const std::string& kernel_name = "CUDAGeneratedKernel");

  std::vector<Allocate*> global_allocations() {
    return global_allocations_;
  }

  std::vector<Allocate*> sync_allocations() {
    return sync_allocations_;
  }

 private:
  std::vector<Allocate*> global_allocations_;
  std::vector<Allocate*> sync_allocations_;

  void lower();

  std::vector<Expr*> lowered_exprs_;
  Fusion* fusion_ = nullptr;
};

} // namespace fuser
} // namespace jit
} // namespace torch
