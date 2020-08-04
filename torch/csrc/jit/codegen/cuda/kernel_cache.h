#pragma once

#include <torch/csrc/jit/codegen/cuda/executor.h>
#include <torch/csrc/jit/codegen/cuda/fusion.h>

#include <c10/util/ArrayRef.h>
#include <torch/csrc/WindowsTorchApiMacro.h>

#include <type_traits>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

// Given a particular torchscript string produced by std::string
// Graph::toString(bool print_source_locations) const; cache a kernel that can
// run it. Assume contiguity information is included in the string.

// There are many things to figure out with this cacheing class, for now we will
// keep it very simple, and take in functionality as complexity grows.

// TODO: Figure out how we want to cache based on heuristics, should probably
// use something similar. Heuristics may also return a LaunchParams object.
// TODO: Validate it is included in the string.

class FusionExecutorCache {
 public:
  //FusionExecutorCache(Fusion* fusion, CompileOptions options);
  FusionExecutorCache(std::unique_ptr<Fusion>&& fusion, at::Device device);

  std::vector<at::Tensor> runFusionWithInputs(
      const at::ArrayRef<IValue>& inputs);

 private:
  at::Device device_;
  std::unique_ptr<Fusion> fusion_;
  std::vector<std::unique_ptr<FusionExecutor>> fusion_executor_cache_;
};

class GraphCache {
 public:
  GraphCache(std::shared_ptr<Graph> graph);

  std::vector<at::Tensor> runGraphWithInputs(
      const at::ArrayRef<IValue>& inputs);

 private:
  // TODO: place holder with naive implementation for now.
  struct InputsRequirement {
    c10::optional<at::Device> device_;
    at::DimVector input_permutation_;
    at::DimVector output_permutation_;
    // TODO: TensorTypePtr is not very easy to work with.
    // c10::nullopt to take place of non-tensor type;
    std::vector<c10::optional<at::TensorTypePtr>> vec_optional_ttp;

    InputsRequirement(const std::shared_ptr<Graph>& graph, const std::vector<size_t>& reduction_axes);
    InputsRequirement(const at::ArrayRef<IValue>& inputs, const std::vector<size_t>& reduction_axes);

    //bool operator==(const InputsRequirement& other);
    bool complyWith(const InputsRequirement& expect);

    bool requiresPermutation();
  };

  FusionExecutorCache* createFusionExecutorCache(const InputsRequirement& input_stack);

  // Computation graph;
  std::shared_ptr<Graph> graph_;
  // TODO: poor name, we should use `eliminated_axes_` instead;
  at::DimVector reduction_axes_;

  // TODO: we should really hash instead of iterative check. Optimize later...
  //       unordered_map<InputsRequirement, FusionExecutorCache>;
  std::vector<InputsRequirement> input_stacks_;
  std::vector<std::unique_ptr<FusionExecutorCache>> fec_cache_;
};

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
