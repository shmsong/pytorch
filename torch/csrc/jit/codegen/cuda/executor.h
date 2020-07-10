#pragma once
#include <torch/csrc/jit/codegen/cuda/executor_launch_params.h>
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
  FusionExecutor(CompileOptions _options) : options_(_options) {}

  std::string KernelName() const {
    std::stringstream ss;
    ss << "kernel"; // << fusion_id;
    return ss.str();
  }

  static std::string Namespace() {
    return "CudaCodeGen";
  }

  std::string getKernel();

  void compileFusion(Fusion* fusion) {
    TORCH_INTERNAL_ASSERT(
        !fusion->outputs().empty(),
        "No output found for this kernel, aborting.");

    fusion_ = *fusion;
    FusionGuard fg(&fusion_);

    fusion_id = ++fusion_id_counter;
    has_random = fusion->hasRNG();

    // Copy constructor to make a full copy of the fusion.
    auto code = getKernel();
    nvrtcCompile(code);
  }

  LaunchParams computeLaunchParams(const at::ArrayRef<IValue>& aten_inputs);

  std::vector<at::Tensor> runFusion(
      const at::ArrayRef<IValue> inputs,
      const std::vector<at::Tensor>& outputs);

  std::vector<at::Tensor> runFusion(const at::ArrayRef<IValue> inputs) {
    TORCH_INTERNAL_ASSERT(false, "Not implemented yet.");
  }

 private:
  void nvrtcCompile(const std::string& code);

  // TODO: Make pointer to const, will take some const fixing in codegenerator
  // to work.
  Fusion fusion_;

  CompileOptions options_;

  // nvrtc state
  CUmodule module_;
  CUfunction function_;

  // State of the fusion that's important
  bool has_random;

  // Counter to be used for kernel name.
  int fusion_id = -1;
  static int fusion_id_counter;
};

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch