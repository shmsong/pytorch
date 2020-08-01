#pragma once

#include <torch/csrc/jit/codegen/cuda/executor.h>
#include <torch/csrc/jit/codegen/cuda/fusion.h>

#include <c10/util/ArrayRef.h>
#include <torch/csrc/WindowsTorchApiMacro.h>

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
  FusionExecutor* getExecutor() const {
    return entry;
  }

  FusionExecutorCache(Fusion* fusion, CompileOptions options);

 private:
  FusionExecutor* entry = nullptr;
};

class GraphCache {
 public:
  GraphCache(std::shared_ptr<Graph> graph);

 private:
  // Computation graph;
  std::shared_ptr<Graph> graph_;

  // TODO: we should really hash instead of iterative check. Optimize later...
  std::vector<InputStack> input_stacks_;
  std::vector<std::unique_ptr<FusionExecutorCache>> fusion_executor_caches_;
};

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
