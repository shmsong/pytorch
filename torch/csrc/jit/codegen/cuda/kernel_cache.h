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

// [ Note -- 2 level cache implementation ]
//
// 2 level hierarchically nested cache is to handle the code generation and
// execution of a given PyTorch IR graph that is unique in its computational
// graph (see note computational graph down).
//
// The nested cache structures are:
//     a. GraphCache
//        - holds a vector of `InputsRequirement` & `FusionExecutorCache`, where
//          each entry is constructed to handle a set of inputs with unique
//          contiguity info, stride order & broadcasting semantics, on a given
//          device;
//        - `InputsRequirement::complyWith` demonstrates the meta information
//          that remains unchanged for a given `FusionExecutorCache`
//        - At run-time (or compile-time with Profiling Executor), we extract
//          `InputsRequirement` from given inputs to the fused operation. We
//          iterate through existing entries within GraphCache (that is the
//          `input_stacks_`) looking for a suitable entry to execute the
//          computation.
//        - In the case of cache miss, we generate a new entry and put it in
//          the GraphCache instance (We push back to both `input_stacks_` and
//          `fe_cache_`, fusion executor cache.
//     b. FusionExecutorCache
//        - holds a vector of `FusionExecutor` to handle dynamic shape (varying
//          tensor sizes)
//        - currently this is only a dummy implementation;
//
// * note computational graph
// In theory, computational graph should refer to only the computational nodes
// in a subgraph and should remain agnostic to input meta info, like
// shape, strides, type e.t.c.. However, the contract right here is fuzzy.
// Different executor applies their own protocol of what is a unique
// computational graph. e.g. Legacy Executor embeds tensor type & dimensionality
// in the graph, while Profiling Executor keeps symbolic shape as well as stride
// order in the graph as well.
// Our definition of computational graph is relaxed to support Legacy Executor,
// so the `GraphCache` could handle varying memory layout of strided tensor
// (different stride order & contiguity information). We utilize the profiling
// information now by generating an entry in GraphCache with the given profiling
// record.

// TODO: FusionExecutorCache is only a place holder here. It's populated in a
// later PR.
class FusionExecutorCache {
 public:
  // create new fusion executor cache at a given device to handle kernel
  // generation of dynamic sizes;
  // fusion executor is taking the ownership of `fusion`;
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

    InputsRequirement(
        const std::shared_ptr<Graph>& graph,
        const std::vector<size_t>& reduction_axes);
    InputsRequirement(
        const at::ArrayRef<IValue>& inputs,
        const std::vector<size_t>& reduction_axes);

    // bool operator==(const InputsRequirement& other);
    bool complyWith(const InputsRequirement& expect);

    bool requiresPermutation();
  };

  FusionExecutorCache* createFusionExecutorCache(
      const InputsRequirement& input_stack);

 private:
  // Computation graph;
  std::shared_ptr<Graph> graph_;
  // TODO: poor name, we should use `eliminated_axes_` instead;
  at::DimVector reduction_axes_;

  // TODO: we should really hash instead of iterative check. Optimize later...
  //       unordered_map<InputsRequirement, FusionExecutorCache>;
  std::vector<InputsRequirement> input_stacks_;
  std::vector<std::unique_ptr<FusionExecutorCache>> fe_cache_;
};

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
