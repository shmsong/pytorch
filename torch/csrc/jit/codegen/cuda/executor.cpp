#include <torch/csrc/jit/codegen/cuda/executor_kernel_arg.h>
#include <torch/csrc/jit/codegen/cuda/expr_evaluator.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/iter_visitor.h>
#include <torch/csrc/jit/codegen/cuda/kernel.h>

#include <torch/csrc/jit/codegen/cuda/executor.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

int FusionExecutor::fusion_id_counter = 0;

std::string FusionExecutor::getKernel() {
  // generating cuda code;
  std::string code = std::string("namespace ") + FusionExecutor::Namespace() +
      " {\n" + executor_utils::kernelPreamble() +
      GPULower(&fusion_).getKernel(KernelName()) + "}\n";

  const char* debug_env = getenv("PYTORCH_CUDA_FUSER_DEBUG");
  if (debug_env && atoi(debug_env)) {
    std::cout << "\n==== codegen output for kernel: " << KernelName()
              << " ====" << std::endl
              << code << std::endl
              << "====================================" << std::endl;
  }

  return code;
}

void FusionExecutor::compileFusion(Fusion* fusion) {
  TORCH_INTERNAL_ASSERT(
      !fusion->outputs().empty(), "No output found for this kernel, aborting.");

  fusion_ = *fusion;
  FusionGuard fg(&fusion_);

  fusion_id = ++fusion_id_counter;
  has_random = fusion->hasRNG();

  compiled_kernel = executor_utils::nvrtcCompile(
      getKernel(), (Namespace() + "::" + KernelName()).c_str(), fusion_id);
}

// TODO: add options for constraints, if we want to enforce a parallelization
// strategy because we did something like split on a symbolic size.
LaunchParams FusionExecutor::computeLaunchParams(
    const at::ArrayRef<IValue>& aten_inputs) {
  TORCH_INTERNAL_ASSERT(
      fusion_.inputs().size() == aten_inputs.size(),
      "Something went wrong configuring launch. Inputs no longer match.");

  auto fusion_inputs = fusion_.inputs();
  EvaluationContext ec(&fusion_);

  // This should probably move to EvaluationContext as we may want to bind input
  // values frequently. Bind fusion input values to runtime values.
  for (size_t i = 0; i < fusion_.inputs().size(); i++) {
    if (fusion_.inputs()[i]->getValType() == ValType::TensorView) {
      TensorView* cg_tensor = fusion_.inputs()[i]->as<TensorView>();

      TORCH_INTERNAL_ASSERT(
          aten_inputs[i].isTensor(),
          "Something went wrong configuring launch. Inputs no longer match.");

      auto aten_tensor = aten_inputs[i].toTensor();

      TORCH_INTERNAL_ASSERT(
          aten_tensor.ndimension() == cg_tensor->getRootDomain().size(),
          "Something went wrong configuring launch. Inputs no longer match.");

      for (size_t dim = 0; dim < cg_tensor->getRootDomain().size(); dim++) {
        ec.bind(
            cg_tensor->getRootDomain()[dim]->extent(),
            aten_tensor.sizes()[dim]);
      }
    }
  }

  LaunchParams launch_params;

  // Grab all used values and run through them
  auto unordered_vals = DependencyCheck::getAllValsBetween(
      {fusion_inputs.begin(), fusion_inputs.end()}, fusion_.outputs());

  // Grab only the TensorViews
  std::set<TensorView*> unordered_tvs;
  for (auto val : unordered_vals) {
    if (val->getValType().value() == ValType::TensorView) {
      TensorView* tv = val->as<TensorView>();
      unordered_tvs.emplace(tv);
    }
  }

  // Infer bindings
  for (auto tv : unordered_tvs) {
    for (auto id : tv->domain()->domain()) {
      if (id->isThread()) {
        auto val = ExpressionEvaluator::evaluate(id->rawExtent(), &ec);
        TORCH_INTERNAL_ASSERT(
            val,
            "Tried to evaluate, ",
            id,
            ", within the tensor ",
            tv,
            " to set launch bounds but could not.");
        launch_params.bind(val.value(), id->parallel_method());
      }
    }
  }

  return launch_params;
}

// This function is here for testing purposes only
std::vector<at::Tensor> FusionExecutor::runFusion(
    const at::ArrayRef<IValue> inputs,
    const std::vector<at::Tensor>& outputs) {
  TORCH_INTERNAL_ASSERT(
      fusion_id > 0, "Cannot run fusion, it was not compiled.");

  FusionGuard fg(&fusion_);

  executor_utils::validateKernelArgs(
      &fusion_, inputs, outputs, options_.device);

  const auto prior_device = at::cuda::current_device();
  at::cuda::set_device(options_.device);
  auto stream = at::cuda::getCurrentCUDAStream();

  TORCH_INTERNAL_ASSERT(!outputs.empty(), "No outputs set for test kernel.");

  KernelArgumentHolder kernel_arguments;
  kernel_arguments.appendArgs(inputs);
  kernel_arguments.appendArgs(outputs);

  LaunchParams lp = computeLaunchParams(inputs);

  if (has_random) {
    const auto rand_offset =
        4 * (std::ceil(outputs[0].numel() / (4.0 * 128 * lp.gdimx())) + 1);
    kernel_arguments.appendPhilox(rand_offset);
  }

  // TODO SUPPORT GRID REDUCTIONS:
  // // When the kernel has global reductions, the kernel needs two
  // // additional temporary buffers, one for intermediate results and
  // // another for synchronization among thread blocks.
  // if (fusion_.hasGridReduction()) {
  //   auto temp_buf_type = at::kFloat;
  //   auto temp_buf_sizes = gridReductionTempBufferSizes(entry);
  //   auto options =
  //       at::TensorOptions().dtype(temp_buf_type).device(at::kCUDA, 0);
  //   at::Tensor reduction_work_buffer = at::empty(
  //       {(long)(temp_buf_sizes[0] / c10::elementSize(temp_buf_type))},
  //       options);
  //   kernel_args.push(reduction_work_buffer);
  //   at::Tensor sync_flags = at::zeros(
  //       {(long)(temp_buf_sizes[1] / c10::elementSize(temp_buf_type))},
  //       options);
  //   kernel_args.push(sync_flags);
  // }

  // launch kernel;
  AT_CUDA_DRIVER_CHECK(at::globalContext().getNVRTC().cuLaunchKernel(
      compiled_kernel.function,
      lp.gdimx(),
      lp.gdimy(),
      lp.gdimz(),
      lp.bdimx(),
      lp.bdimy(),
      lp.bdimz(),
      0, // smem
      stream,
      kernel_arguments.getBuffer(),
      nullptr));

  // // Resets device (see at::DeviceGuard notes above)
  // at::cuda::set_device(prior_device);

  return std::vector<at::Tensor>(outputs);
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
