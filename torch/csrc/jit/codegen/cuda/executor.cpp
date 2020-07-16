#include <torch/csrc/jit/codegen/cuda/executor_kernel_arg.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/iter_visitor.h>
#include <torch/csrc/jit/codegen/cuda/kernel.h>

#include <torch/csrc/jit/codegen/cuda/executor.h>

#include <c10/core/DeviceGuard.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

int FusionExecutor::fusion_id_counter = 0;

std::string FusionExecutor::getStructuredCode(std::string kernel) {
  // generating cuda code;
  std::string code = std::string("namespace ") + FusionExecutor::Namespace() +
      " {\n" + executor_utils::kernelPreamble() + kernel + "}\n";

  const char* debug_env = getenv("PYTORCH_CUDA_FUSER_DEBUG");
  if (debug_env && atoi(debug_env)) {
    std::cout << "\n==== codegen output for kernel: " << KernelName()
              << " ====" << std::endl
              << code << std::endl
              << "=====*===============================" << std::endl;
  }

  return code;
}

void FusionExecutor::compileFusion(Fusion* fusion) {
  TORCH_INTERNAL_ASSERT(
      !fusion->outputs().empty(), "No output found for this kernel, aborting.");

  for (auto out : fusion->outputs())
    TORCH_INTERNAL_ASSERT(
        out->getValType() == ValType::TensorView,
        "Output types from fusions that are not tensors are not supported at this point.");

  fusion_ = *fusion;
  FusionGuard fg(&fusion_);

  fusion_id = ++fusion_id_counter;
  has_random = fusion->hasRNG();
  lowered = GPULower(&fusion_);

  auto code = getStructuredCode(lowered.getKernel(KernelName()));

  compiled_kernel = executor_utils::nvrtcCompile(
      code, (Namespace() + "::" + KernelName()).c_str(), fusion_id);
}

namespace {
EvaluationContext bindInputs(
    const at::ArrayRef<IValue>& aten_inputs,
    Fusion* fusion) {
  TORCH_INTERNAL_ASSERT(
      fusion->inputs().size() == aten_inputs.size(),
      "Something went wrong configuring launch. Inputs no longer match.");

  auto fusion_inputs = fusion->inputs();
  EvaluationContext ec(fusion);

  // This should probably move to EvaluationContext as we may want to bind input
  // values frequently. Bind fusion input values to runtime values.
  for (size_t i = 0; i < fusion->inputs().size(); i++) {
    if (fusion->inputs()[i]->getValType() == ValType::TensorView) {
      TensorView* cg_tensor = fusion->inputs()[i]->as<TensorView>();

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
  return std::move(ec);
}

at::Tensor inferAndAlloc(
    TensorView* tv,
    EvaluationContext& ec,
    const CompileOptions& options,
    bool zero_init = false) {
  std::vector<int64_t> sizes;
  for (auto id : tv->getRootDomain()) {
    auto infered_val = ExpressionEvaluator::evaluate(id->rawExtent(), &ec);
    TORCH_INTERNAL_ASSERT(
        infered_val.has_value(),
        "Could not launch kernel as program could not infer ",
        id->rawExtent(),
        " for the buffer ",
        tv);
    sizes.push_back(infered_val.value());
  }

  auto at_type = data_type_to_aten(tv->getDataType().value());
  auto tensor_options =
      at::TensorOptions().dtype(at_type).device(options.device);

  if (zero_init) {
    c10::IntArrayRef isizes(sizes);
    return at::zeros(isizes, tensor_options);
  } else {
    c10::IntArrayRef isizes(sizes);
    return at::empty(isizes, tensor_options);
  }
}
} // namespace

LaunchParams FusionExecutor::computeLaunchParams(
    const at::ArrayRef<IValue>& aten_inputs,
    const LaunchParams& launch_constraints,
    EvaluationContext& ec) {
  LaunchParams launch_params;

  // Grab all values that are actually used in the fusion
  auto unordered_vals = DependencyCheck::getAllValsBetween(
      {fusion_.inputs().begin(), fusion_.inputs().end()}, fusion_.outputs());

  // Lets collect all IterDomains that are bound to a thread binding
  std::unordered_map<ParallelType, std::vector<IterDomain*>, TypeHash>
      parallel_iter_domains;

  for (auto val : unordered_vals) {
    if (val->getValType().value() == ValType::TensorView) {
      TensorView* tv = val->as<TensorView>();
      for (auto id : tv->domain()->domain()) {
        if (id->isThread()) {
          if (parallel_iter_domains.find(id->parallel_method()) !=
              parallel_iter_domains.end()) {
            parallel_iter_domains.at(id->parallel_method()).push_back(id);
          } else {
            parallel_iter_domains[id->parallel_method()] =
                std::vector<IterDomain*>({id});
          }
        }
      }
    }
  }

  // If any dimension was set in launch constraints we need to run through
  // IterDomains that have been parallelized, and bind those values. Or make
  // sure if they could be infered the inference matches what was set.
  if (launch_constraints.nBlocks() * launch_constraints.nThreads() != -1) {
    for (auto& entry : parallel_iter_domains) {
      auto p_type = entry.first;
      if (launch_constraints.hasDim(p_type)) {
        auto parallel_ids = entry.second;
        for (auto parallel_id : parallel_ids) {
          auto infered_val =
              ExpressionEvaluator::evaluate(parallel_id->rawExtent(), &ec);
          if (infered_val.has_value()) {
            // This value could have been infered, make sure it was set right.
            TORCH_CHECK(
                infered_val.value() == launch_constraints.getDim(p_type) ||
                    launch_constraints.getRawVal(p_type) == -1,
                "Infered that ",
                p_type,
                " should be set to ",
                infered_val.value(),
                " but launch constraints specified ",
                launch_constraints.getDim(p_type));
          } else {
            // Bind the launch constraint into our evaluation context
            ec.bind(
                parallel_id->rawExtent(),
                launch_constraints.getDim(entry.first));
            launch_params.bind(launch_constraints.getDim(p_type), p_type);
          }
        }
      }
    }
  }

  // Run through the rest of the parallel IterDomains and infer their size
  for (auto& entry : parallel_iter_domains) {
    auto p_type = entry.first;
    auto parallel_ids = entry.second;
    for (auto parallel_id : parallel_ids) {
      auto val = ExpressionEvaluator::evaluate(parallel_id->rawExtent(), &ec);
      TORCH_INTERNAL_ASSERT(
          val,
          "Tried to evaluate the extent of ",
          parallel_id,
          " to set launch bounds but could not.");
      launch_params.bind(val.value(), p_type);
    }
  }

  return launch_params;
}

std::vector<at::Tensor> FusionExecutor::allocGlobalVals(
    const at::ArrayRef<IValue>& aten_inputs,
    EvaluationContext& ec) {
  std::vector<at::Tensor> global_buffers;
  for (auto alloc : lowered.global_allocations()) {
    TORCH_INTERNAL_ASSERT(
        alloc->buffer()->getValType() == ValType::TensorView,
        "Cannot global buffers that are not tensors.");
    global_buffers.push_back(
        inferAndAlloc(alloc->buffer()->as<TensorView>(), ec, options_, false));
  }

  for (auto alloc : lowered.sync_allocations()) {
    TORCH_INTERNAL_ASSERT(
        alloc->buffer()->getValType() == ValType::TensorView,
        "Cannot global buffers that are not tensors.");
    global_buffers.push_back(
        inferAndAlloc(alloc->buffer()->as<TensorView>(), ec, options_, true));
  }

  return global_buffers;
}

std::vector<at::Tensor> FusionExecutor::runFusion(
    const at::ArrayRef<IValue> inputs,
    const std::vector<at::Tensor>& outputs,
    const LaunchParams& launch_constraints) {
  TORCH_INTERNAL_ASSERT(
      fusion_id > 0, "Cannot run fusion, it was not compiled.");

  FusionGuard fg(&fusion_);

  executor_utils::validateKernelArgs(
      &fusion_, inputs, outputs, options_.device);

  const auto prior_device = at::cuda::current_device();
  c10::DeviceGuard dg(options_.device);
  auto stream = at::cuda::getCurrentCUDAStream();

  TORCH_INTERNAL_ASSERT(!outputs.empty(), "No outputs set for test kernel.");

  EvaluationContext evaluation_context = bindInputs(inputs, &fusion_);

  LaunchParams launch_params =
      computeLaunchParams(inputs, launch_constraints, evaluation_context);

  KernelArgumentHolder kernel_arguments;
  kernel_arguments.appendArgs(inputs);
  kernel_arguments.appendArgs(outputs);
  auto buffers = allocGlobalVals(inputs, evaluation_context);
  kernel_arguments.appendArgs(buffers);

  if (has_random) {
    const auto rand_offset = 4 *
        (std::ceil(outputs[0].numel() / (4.0 * 128 * launch_params.gdimx())) +
         1);
    kernel_arguments.appendPhilox(rand_offset);
  }

  AT_CUDA_DRIVER_CHECK(at::globalContext().getNVRTC().cuLaunchKernel(
      compiled_kernel.function,
      launch_params.gdimx(),
      launch_params.gdimy(),
      launch_params.gdimz(),
      launch_params.bdimx(),
      launch_params.bdimy(),
      launch_params.bdimz(),
      0, // smem
      stream,
      kernel_arguments.getBuffer(),
      nullptr));
  AT_CUDA_CHECK(cudaStreamSynchronize(stream));

  return std::vector<at::Tensor>(outputs);
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
