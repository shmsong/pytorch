#pragma once
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

class TORCH_CUDA_API LaunchParams {
 public:
  unsigned int smem() const {
    return smem_;
  }
  unsigned int nBlocks() const {
    return gdimx_ * gdimy_ * gdimz_;
  }

  unsigned int nThreads() const {
    return bdimx_ * bdimy_ * bdimz_;
  }

  unsigned int bdimx() const {
    return (unsigned int)bdimx_ == -1 ? 1 : bdimx_;
  }

  unsigned int gdimx() const {
    return (unsigned int)gdimx_ == -1 ? 1 : gdimx_;
  }

  unsigned int bdimy() const {
    return (unsigned int)bdimy_ == -1 ? 1 : bdimy_;
  }

  unsigned int gdimy() const {
    return (unsigned int)gdimy_ == -1 ? 1 : gdimy_;
  }

  unsigned int bdimz() const {
    return (unsigned int)bdimz_ == -1 ? 1 : bdimz_;
  }

  unsigned int gdimz() const {
    return (unsigned int)gdimz_ == -1 ? 1 : gdimz_;
  }

  void checkAndSet(
      const int64_t incoming_val,
      int& class_val,
      std::string val) {
    TORCH_INTERNAL_ASSERT(
        class_val == -1 || incoming_val == 1 || class_val == 1 ||
            incoming_val == class_val,
        "Tried to set ",
        val,
        " to ",
        incoming_val,
        ", but it was already set and new value does not match.",
        " Thread dims all have to be bound to the same value.");
    if (class_val == -1 || class_val == 1) {
      class_val = incoming_val;
    }
  }

  void bind(int64_t val, ParallelType p_dim);

 private:
  // Spell them out because I want signed ints to know if they were initialized
  // or not.
  int gdimx_ = -1;
  int gdimy_ = -1;
  int gdimz_ = -1;
  int bdimx_ = -1;
  int bdimy_ = -1;
  int bdimz_ = -1;

  unsigned int smem_ = 0;

  // TODO: Fill in output sizes
  std::vector<std::vector<unsigned int>> output_sizes;
};

struct TORCH_CUDA_API CompileOptions {
  int device = 0;
};

class ArgAbstract;

class KernelArgumentHolder {
 public:
  ~KernelArgumentHolder() = default;

  // Push a tensor to the arguments
  void push(const at::Tensor& tensor);

  // Push a scalar or integer to the arguments
  void push(const IValue& val);

  void push(const uint64_t& val);

  // Create buffer, flatten arguments into it, align by 8 Bytes, return pointers
  // in the buffer
  void** getBuffer();

  void appendArgs(const c10::ArrayRef<c10::IValue>& args);

  void appendArgs(const std::vector<at::Tensor>& tensors);

  void appendPhilox(uint64_t rand_offset);

 private:
  std::vector<std::unique_ptr<ArgAbstract>> arguments;
  std::vector<void*> void_ptrs;
  bool changed = true;
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