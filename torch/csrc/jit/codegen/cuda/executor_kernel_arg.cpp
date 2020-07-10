#include <ATen/CUDAGeneratorImpl.h>

// Extract size and strides
#include <torch/csrc/jit/codegen/cuda/kernel_cache.h>

#include <torch/csrc/jit/codegen/cuda/executor_kernel_arg.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

std::unique_ptr<TensorArgAbstract> getTensorArg(
    c10::ScalarType dtype,
    int nDims) {
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

// Push a tensor to the arguments
void KernelArgumentHolder::push(const at::Tensor& tensor) {
  changed = true;
  int nDims = tensor.ndimension();

  c10::ScalarType dtype = tensor.scalar_type();
  std::unique_ptr<TensorArgAbstract> tensor_arg = getTensorArg(dtype, nDims);
  tensor_arg->setPointer(tensor.data_ptr());
  for (int i = 0; i < nDims; i++) {
    tensor_arg->setSize(i, tensor.sizes()[i]);
    tensor_arg->setStride(i, tensor.strides()[i]);
  }
  arguments.push_back(std::move(tensor_arg));
}

// Push a tensor to the arguments
void KernelArgumentHolder::push(
    const at::Tensor& val,
    c10::optional<at::IntArrayRef> broadcasted_size) {
  changed = true;
  ExtractSizeStride ess(val, std::move(broadcasted_size));
  int nDims = ess.sizes.size();

  c10::ScalarType dtype = val.scalar_type();
  std::unique_ptr<TensorArgAbstract> tensor_arg = getTensorArg(dtype, nDims);
  tensor_arg->setPointer(val.data_ptr());
  for (int i = 0; i < nDims; i++) {
    tensor_arg->setSize(i, ess.sizes[i]);
    tensor_arg->setStride(i, ess.strides[i]);
  }
  arguments.push_back(std::move(tensor_arg));
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
      arguments.push_back(std::make_unique<FloatArg>((float)val.toDouble()));
      return;
    case (c10::ScalarType::Long):
      arguments.push_back(std::make_unique<IntArg>((int)val.toInt()));
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
  arguments.push_back(std::make_unique<ULongArg>(val));
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

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
