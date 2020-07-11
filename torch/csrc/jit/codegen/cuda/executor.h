#pragma once
#include <torch/csrc/jit/codegen/cuda/executor_launch_params.h>
#include <torch/csrc/jit/codegen/cuda/executor_utils.h>
#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/ir_cloner.h>
#include <torch/csrc/jit/codegen/cuda/ir_printer.h>
#include <torch/csrc/jit/codegen/cuda/kernel.h>
#include <torch/csrc/jit/codegen/cuda/lower2device.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

struct TORCH_CUDA_API CompileOptions {
  int device = 0;
};

class TORCH_CUDA_API FusionExecutor {
 public:
  FusionExecutor() {}
  FusionExecutor(CompileOptions options) : options_(options) {}

  std::string KernelName() const {
    std::stringstream ss;
    ss << "kernel" << fusion_id;
    return ss.str();
  }

  static std::string Namespace() {
    return "CudaCodeGen";
  }

  std::string getKernel();

  void compileFusion(Fusion* fusion);

  LaunchParams computeLaunchParams(const at::ArrayRef<IValue>& aten_inputs);

  std::vector<at::Tensor> runFusion(
      const at::ArrayRef<IValue> inputs,
      const std::vector<at::Tensor>& outputs);

  std::vector<at::Tensor> runFusion(const at::ArrayRef<IValue> inputs) {
    TORCH_INTERNAL_ASSERT(false, "Not implemented yet.");
  }

 private:
  Fusion fusion_;

  CompileOptions options_;

  executor_utils::NvrtcFunction compiled_kernel;

  // State of the fusion that's important
  bool has_random = false;

  // Counter to be used for kernel name.
  int fusion_id = -1;
  static int fusion_id_counter;
};

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch