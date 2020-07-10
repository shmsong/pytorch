#include <torch/csrc/jit/codegen/cuda/expr_evaluator.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/iter_visitor.h>
#include <torch/csrc/jit/codegen/cuda/kernel.h>
#include <torch/csrc/jit/codegen/cuda/kernel_arg.h>
#include <torch/csrc/jit/codegen/cuda/kernel_resource_strings.h>

#include <ATen/CUDAGeneratorImpl.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/nvrtc_stub/ATenNVRTC.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <torch/csrc/jit/resource_guard.h>

// #include <ATen/CUDAGeneratorImpl.h>
// #include <ATen/cuda/CUDAContext.h>
// #include <c10/core/ScalarType.h>
// #include <c10/util/ArrayRef.h>

#include <torch/csrc/jit/codegen/cuda/fusion_executor.h>

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

KernelArgumentHolder::~KernelArgumentHolder() {
  for (auto arg : arguments)
    delete arg;
}

// Push a tensor to the arguments
void KernelArgumentHolder::push(const at::Tensor& tensor) {
  changed = true;
  int nDims = tensor.ndimension();

  c10::ScalarType dtype = tensor.scalar_type();
  TensorArgAbstract* tensor_arg = getTensorArg(dtype, nDims);
  tensor_arg->setPointer(tensor.data_ptr());
  for (int i = 0; i < nDims; i++) {
    tensor_arg->setSize(i, tensor.sizes()[i]);
    tensor_arg->setStride(i, tensor.strides()[i]);
  }
  arguments.push_back(tensor_arg);
}

// Push a scalar or integer to the arguments
void KernelArgumentHolder::push(const IValue& val) {
  changed = true;
  TORCH_INTERNAL_ASSERT(
      val.isScalar(),
      "Tried to push an arg to run in a fused kernel, expected a scalar but got, ",
      val);
  switch (val.toScalar().type()) {
    case (c10::ScalarType::Double):
      arguments.push_back(new FloatArg((float)val.toDouble()));
      return;
    case (c10::ScalarType::Long):
      arguments.push_back(new IntArg((int)val.toInt()));
      return;
    default:
      TORCH_INTERNAL_ASSERT(
          false,
          " Tried to create argument to send to a fused kernel, but got an unexpected type.");
  }
  TORCH_INTERNAL_ASSERT(
      false,
      " Tried to create argument to send to a fused kernel, but got a non-scalar type.");
}

void KernelArgumentHolder::push(const uint64_t& val) {
  arguments.push_back(new ULongArg(val));
}

// Create buffer, flatten arguments into it, align by 8 Bytes, return pointers
// in the buffer
void** KernelArgumentHolder::getBuffer() {
  if (changed) {
    void_ptrs = std::vector<void*>(arguments.size(), nullptr);
    for (decltype(arguments.size()) i{0}; i < arguments.size(); i++)
      void_ptrs[i] = static_cast<void*>(arguments[i]->arg());
    changed = false;
  }
  return void_ptrs.data();
}

void KernelArgumentHolder::appendArgs(const c10::ArrayRef<c10::IValue>& args) {
  // Naive I/O setup, I'm ignoring all the potential transformation (i.e. I/O
  // allocated here from the subgraph could be, and very likely are, different
  // from I/O expected by the generated CUDA kernel.
  for (const auto& arg : args) {
    if (arg.isTensor()) {
      push(arg.toTensor());
    } else {
      push(arg);
    }
  }
}

void KernelArgumentHolder::appendArgs(const std::vector<at::Tensor>& tensors) {
  for (const auto& tensor : tensors) {
    push(tensor);
  }
}

void KernelArgumentHolder::appendPhilox(uint64_t rand_offset) {
  std::pair<uint64_t, uint64_t> philox_engine_inputs;
  auto gen = at::cuda::detail::getDefaultCUDAGenerator();
  {
    // See Note [Acquire lock when using random generators]
    std::lock_guard<std::mutex> lock(gen.mutex());
    philox_engine_inputs =
        at::check_generator<at::CUDAGeneratorImpl>(gen)->philox_engine_inputs(
            rand_offset);
  }
  push(philox_engine_inputs.first);
  push(philox_engine_inputs.second);
}

namespace {
std::string KernelUtils() {
  std::stringstream ss;
  ss << code_template_tensor_struct << "\n"
     << code_fp16_support << "\n"
     << code_random_number_gen << "\n"
     << code_helper_funcs << "\n"
     << code_template_block_reduction << "\n"
     << code_template_grid_reduction << "\n"
     << code_template_block_broadcast << "\n";
  return ss.str();
}

bool validateKernelArgTensor(
    const at::Tensor& arg,
    const Val* param,
    int device_index,
    std::stringstream& msg) {
  // Arg is a tensor. Param must be a tensor too.
  if (*param->getValType() != ValType::TensorView) {
    msg << "Argument is a tensor, but the parameter is not.";
    return false;
  }

  // Check the rank of the tensors.
  size_t arg_dim = arg.dim();
  // Note: This requires current Fusion to be active.
  size_t param_dim = TensorDomain::noReductions(
                         static_cast<const TensorView*>(param)->getRootDomain())
                         .size();
  // see [Note - broadcast support in integration]
  // Because of broadcasting support handled in integration, we relax the rank
  // check as necessary.
  if (arg_dim > param_dim) {
    msg << "Argument tensor's rank is " << arg_dim << ", but the parameter is "
        << param_dim;
    return false;
  }

  if (arg.device().index() != device_index) {
    msg << "Argument is on device that is not compiled for";
    return false;
  }
  // Check element type
  at::ScalarType arg_data_type = arg.scalar_type();
  DataType param_data_type = *param->getDataType();
  bool match = false;
  switch (arg_data_type) {
    case at::ScalarType::Half:
      match = param_data_type == DataType::Half;
      break;
    case at::ScalarType::Float:
      match = param_data_type == DataType::Float;
      break;
    case at::ScalarType::Bool:
      match = param_data_type == DataType::Bool;
      break;
    default:
      msg << "Argument element type, " << arg_data_type
          << ", is not supported.";
      return false;
  }
  if (!match)
    msg << "Argument element type is " << arg_data_type
        << ", but the parameter is " << param_data_type;
  return match;
}

bool validateKernelArgScalar(
    const c10::TypePtr& arg_type,
    const Val* param,
    std::stringstream& msg) {
  if (!param->isScalar()) {
    msg << "Argument is a scalar, but the parameter is not.";
    return false;
  }
  DataType param_type = *param->getDataType();
  bool match = false;
  switch (arg_type->kind()) {
    case c10::TypeKind::IntType:
      match = param_type == DataType::Int;
      break;
    case c10::TypeKind::FloatType:
      match = param_type == DataType::Float;
      break;
    case c10::TypeKind::BoolType:
      match = param_type == DataType::Bool;
      break;
    default:
      match = false;
  }
  if (!match) {
    msg << "Argument type is " << *arg_type << ", but the parameter is "
        << param_type;
  }
  return match;
}

bool validateKernelArg(
    const c10::IValue& arg,
    const Val* param,
    int device_index,
    std::stringstream& msg) {
  if (arg.type()->kind() != c10::TypeKind::TensorType) {
    return validateKernelArgScalar(arg.type(), param, msg);
  } else {
    return validateKernelArgTensor(arg.toTensor(), param, device_index, msg);
  }
}

void validateKernelArgs(
    Fusion* fusion,
    const at::ArrayRef<IValue>& inputs,
    const std::vector<at::Tensor>& outputs,
    int device) {
  // This is necessary as we were traversing the fusion graph later in the check
  FusionGuard fg(fusion);
  // Check inputs
  TORCH_INTERNAL_ASSERT(
      inputs.size() == fusion->inputs().size(),
      "Wrong number of kernel inputs.");
  for (size_t i = 0; i < inputs.size(); ++i) {
    const IValue& arg = inputs[i];
    const Val* param = fusion->inputs()[i];
    std::stringstream msg;
    TORCH_INTERNAL_ASSERT(
        validateKernelArg(arg, param, device, msg),
        "Input argument at position ",
        i,
        " is invalid; ",
        msg.str());
  }

  TORCH_INTERNAL_ASSERT(
      fusion->outputs().size() != 0,
      "Kernel should have at least one output tensor.");

  TORCH_INTERNAL_ASSERT(
      outputs.size() == fusion->outputs().size(),
      "Wrong number of kernel outputs.");
  for (size_t i = 0; i < outputs.size(); ++i) {
    const at::Tensor& arg = outputs[i];
    const Val* param = fusion->outputs()[i];
    std::stringstream msg;
    TORCH_INTERNAL_ASSERT(
        validateKernelArgTensor(arg, param, device, msg),
        "Output argument at position ",
        i,
        " is invalid; ",
        msg.str());
  }
}

}; // namespace

int FusionExecutor::fusion_id_counter = 0;

std::string FusionExecutor::getKernel() {
  // generating cuda code;
  std::string code = std::string("namespace ") + FusionExecutor::Namespace() +
      " {\n" + KernelUtils() + GPULower(&fusion_).getKernel(KernelName()) +
      "}\n";

  const char* debug_env = getenv("PYTORCH_CUDA_FUSER_DEBUG");
  if (debug_env && atoi(debug_env)) {
    std::cout << "\n==== codegen output for kernel: " << KernelName()
              << " ====" << std::endl
              << code << std::endl
              << "====================================" << std::endl;
  }

  return code;
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

  LaunchParams lp;

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
        lp.bind(val.value(), id->parallel_method());
      }
    }
  }

  return lp;
}

// This function is here for testing purposes only
std::vector<at::Tensor> FusionExecutor::runFusion(
    const at::ArrayRef<IValue> inputs,
    const std::vector<at::Tensor>& outputs) {
  TORCH_INTERNAL_ASSERT(
      fusion_id > 0, "Cannot run fusion, it was not compiled.");

  FusionGuard fg(&fusion_);

  validateKernelArgs(&fusion_, inputs, outputs, options_.device);

  const auto prior_device = at::cuda::current_device();
  at::cuda::set_device(options_.device);
  auto stream = at::cuda::getCurrentCUDAStream();

  TORCH_INTERNAL_ASSERT(!outputs.empty(), "No outputs set for test kernel.");

  KernelArgumentHolder kah;
  kah.appendArgs(inputs);
  kah.appendArgs(outputs);

  LaunchParams lp = computeLaunchParams(inputs);

  if (has_random) {
    const auto rand_offset =
        4 * (std::ceil(outputs[0].numel() / (4.0 * 128 * lp.gdimx())) + 1);
    kah.appendPhilox(rand_offset);
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
      function_,
      lp.gdimx(),
      lp.gdimy(),
      lp.gdimz(),
      lp.bdimx(),
      lp.bdimy(),
      lp.bdimz(),
      0, // smem
      stream,
      kah.getBuffer(),
      nullptr));

  // // Resets device (see at::DeviceGuard notes above)
  // at::cuda::set_device(prior_device);

  return std::vector<at::Tensor>(outputs);
}

void FusionExecutor::nvrtcCompile(std::string code) {
  // lazily construct context if non-existing yet;

  std::ofstream out("output_fe.cu");
  out << code;
  out.close();

  std::string func_name = (Namespace() + "::" + KernelName()).c_str();

  CUcontext pctx = nullptr;
  AT_CUDA_DRIVER_CHECK(at::globalContext().getNVRTC().cuCtxGetCurrent(&pctx));
  if (!pctx) {
    std::unique_lock<std::mutex> cudaFreeMutexLock(
        *(c10::cuda::CUDACachingAllocator::getFreeMutex()));
    cudaFree(nullptr);
  }

  const auto prop = at::cuda::getCurrentDeviceProperties();
  int nvrtc_major, nvrtc_minor;
  AT_CUDA_NVRTC_CHECK(
      at::globalContext().getNVRTC().nvrtcVersion(&nvrtc_major, &nvrtc_minor));

  // Short-circuits if NVRTC version too low
  TORCH_INTERNAL_ASSERT(nvrtc_major >= 6);
  // Major and minor is determined by device properties and
  // possibly "downcompiled" to a lower (compatible) compute architecture
  // based on the NVRTC version
  int major, minor;
  major = prop->major;
  minor = prop->minor;
  nvrtcProgram program;
  AT_CUDA_NVRTC_CHECK(at::globalContext().getNVRTC().nvrtcCreateProgram(
      &program, code.c_str(), nullptr, 0, nullptr, nullptr));
  ResourceGuard holdProgram([&] {
    AT_CUDA_NVRTC_CHECK(
        at::globalContext().getNVRTC().nvrtcDestroyProgram(&program));
  });

  const std::string compute = "--gpu-architecture=compute_" +
      std::to_string(major) + std::to_string(minor);
  const std::vector<const char*> args = {
      "--std=c++14", compute.c_str(), "-default-device"};

  at::globalContext().getNVRTC().nvrtcAddNameExpression(
      program, func_name.c_str());
  const auto result = at::globalContext().getNVRTC().nvrtcCompileProgram(
      program, args.size(), args.data());

  if (result != NVRTC_SUCCESS) {
    size_t logsize;
    at::globalContext().getNVRTC().nvrtcGetProgramLogSize(program, &logsize);
    std::vector<char> log(logsize);
    at::globalContext().getNVRTC().nvrtcGetProgramLog(program, log.data());

    TORCH_INTERNAL_ASSERT(
        false, code.c_str(), "\nCUDA NVRTC compile error: ", log.data());
  }
  const char* lowered_kernel_name;
  at::globalContext().getNVRTC().nvrtcGetLoweredName(
      program, func_name.c_str(), &lowered_kernel_name);

  AT_CUDA_NVRTC_CHECK(result);
  size_t ptx_size;
  AT_CUDA_NVRTC_CHECK(
      at::globalContext().getNVRTC().nvrtcGetPTXSize(program, &ptx_size));
  std::vector<char> ptx;
  ptx.resize(ptx_size);
  AT_CUDA_NVRTC_CHECK(
      at::globalContext().getNVRTC().nvrtcGetPTX(program, ptx.data()));

  // TODO: We do go through different code path, should investigate whether this
  // has an impact on generated binary.
  const char* prefix_env = getenv("PYTORCH_CUDA_FUSER_CUBIN");
  if (prefix_env) {
    // Output ptx file
    std::stringstream ptx_file_name;
    ptx_file_name << prefix_env << "_" << fusion_id << ".ptx";
    std::ofstream myPtxFile(ptx_file_name.str().c_str(), std::ios::out);
    if (myPtxFile.is_open()) {
      myPtxFile.write(ptx.data(), ptx.size());
      myPtxFile.close();
    }

    CUlinkState linkState;

    AT_CUDA_DRIVER_CHECK(at::globalContext().getNVRTC().cuLinkCreate(
        0, nullptr, nullptr, &linkState));

    AT_CUDA_DRIVER_CHECK(at::globalContext().getNVRTC().cuLinkAddData(
        linkState,
        CU_JIT_INPUT_PTX,
        ptx.data(),
        ptx_size,
        "compiling PTX",
        0,
        nullptr,
        nullptr));

    size_t cubinSize;
    void* cubin;
    AT_CUDA_DRIVER_CHECK(at::globalContext().getNVRTC().cuLinkComplete(
        linkState, &cubin, &cubinSize));

    // Output binary file
    std::stringstream cubin_file_name;
    cubin_file_name << prefix_env << "_" << fusion_id << ".cubin";

    std::ofstream myCubinFile(
        cubin_file_name.str().c_str(), std::ios::out | std::ios::binary);

    if (myCubinFile.is_open()) {
      myCubinFile.write(static_cast<const char*>(cubin), cubinSize);
      myCubinFile.close();
    }

    // load compiled cubin
    AT_CUDA_DRIVER_CHECK(
        at::globalContext().getNVRTC().cuModuleLoadData(&(module_), cubin));
  } else {
    // load ptx directly
    AT_CUDA_DRIVER_CHECK(at::globalContext().getNVRTC().cuModuleLoadData(
        &(module_), ptx.data()));
  }
  AT_CUDA_DRIVER_CHECK(at::globalContext().getNVRTC().cuModuleGetFunction(
      &(function_), module_, lowered_kernel_name));
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
