#include <torch/csrc/jit/codegen/cuda/kernel_cache.h>
#include <torch/csrc/jit/codegen/cuda/scheduler.h>
#include <torch/csrc/jit/codegen/cuda/parser.h>
#include <torch/csrc/jit/runtime/graph_executor.h>

// TODO: This class is dead at the moment, but we need to figure out a generic
// cacheing system that will suite our needs.

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

namespace {

// Dimension collapsing only applicable to profiling executor at this moment
bool graphHasReduction(const std::shared_ptr<Graph>& graph) {
  for (const auto& n : graph->nodes()) {
    if (isReductionNode(n)) {
      return true;
    }
  }
  return false;
}

at::DimVector getPermutationPerSortedStride(const TensorTypePtr& type) {
  // `permute_seq` is the returned permutation to achieve sorted stride;
  at::DimVector permute_seq;

  auto stride_properties = type->stride_properties().sizes();

  TORCH_INTERNAL_ASSERT(
      stride_properties.has_value(),
      "unknown sizes or stride_properties, collapsing shouldn't happen");

  // TODO: reuse this;
  const int rank = static_cast<int>(stride_properties->size());

  // stores axes with stride_index;
  std::set<int> ordered_axes;

  // TODO: this does not support broadcast yet;
  for (int i = 0; i < rank; i++) {
    if ((*stride_properties)[i].has_value() &&
        (*stride_properties)[i]->stride_index_.has_value()) {
      ordered_axes.insert((*stride_properties)[i]->stride_index_.value());
    }
  }

  int unallocated_axis = 0;
  // we push from slowest to fastest
  for (int i = rank - 1; i >= 0; i--) {
    if ((*stride_properties)[i].has_value() &&
        (*stride_properties)[i]->stride_index_.has_value()) {
      permute_seq.emplace_back(
          (*stride_properties)[i]->stride_index_.value());
    } else {
      // no designated axis for this slot, so we push an axis w/o designated
      // order;
      while (ordered_axes.count(unallocated_axis) != 0) {
        ++unallocated_axis;
      }
      permute_seq.emplace_back(unallocated_axis++);
    }
  }
  return permute_seq;
}

at::DimVector reversePermutation(at::DimVector permuted) {
  int rank = static_cast<int>(permuted.size());
  at::DimVector permutation(rank, -1);
  for (int i = 0; i < rank; i++) {
    permutation[permuted[i]] = i;
  }
  return permutation;
}

} // namespace

FusionExecutorCache::FusionExecutorCache(std::unique_ptr<Fusion>&& fusion, at::Device device) : device_(device), fusion_(std::move(fusion)) {
}

// TODO: dummy cache
std::vector<at::Tensor> FusionExecutorCache::runFusionWithInputs(
    const at::ArrayRef<IValue>& inputs) {
  if (fusion_executor_cache_.empty()) {
    scheduleFusion(fusion_.get(), inputs);
    fusion_executor_cache_.emplace_back(std::make_unique<FusionExecutor>());
    CompileOptions options;
    options.device = device_;
    fusion_executor_cache_.back()->compileFusion(fusion_.get(), options);
  }
  return fusion_executor_cache_.back()->runFusion(inputs);
}

// FusionExecutorCache::FusionExecutorCache(
//     Fusion* fusion,
//     CompileOptions options) {
//   TORCH_INTERNAL_ASSERT(
//       entry == nullptr,
//       "At this time FusionExecutorCache only supports one entry.");
//   entry = new FusionExecutor();
//   entry->compileFusion(fusion, options);
// }

GraphCache::InputsRequirement::InputsRequirement(const std::shared_ptr<Graph>& graph) {
  // run over inputs to extract common types;
  TensorTypePtr acc_type = TensorType::get();
  for (const auto& input : graph->inputs()) {
    // only check tensor types;
    if (auto input_type = input->type()->cast<TensorType>()) {
      vec_optional_ttp.emplace_back(input_type);
      if (acc_type->dim().has_value()) {
        // TODO: I think merge cannot handle broadcast - Go verify it later;
        // TODO: Since we are only handling permutation here, we should just
        //       merge the stride_index_;
        acc_type = acc_type->merge(input_type);
      } else {
        acc_type = input_type;
      }
    } else {
      vec_optional_ttp.emplace_back(c10::nullopt);
    }
  }
  input_permutation_ = getPermutationPerSortedStride(acc_type);
  output_permutation_ = reversePermutation(input_permutation_);
  TORCH_CHECK(acc_type->device().has_value(),
      "requires fixed device for all inputs");
  device_ = acc_type->device();
}

GraphCache::InputsRequirement::InputsRequirement(const at::ArrayRef<IValue>& inputs) {
  // run over inputs to extract common types;
  TensorTypePtr acc_type = TensorType::get();
  for (const auto& input : inputs) {
    // only check tensor types;
    if (input.isTensor()) {
      // TensorType::create populates stride properties;
      auto input_type = TensorType::create(input.toTensor());
      vec_optional_ttp.emplace_back(input_type);
      if (acc_type->dim().has_value()) {
        // TODO: I think merge cannot handle broadcast - Go verify it later;
        // TODO: Since we are only handling permutation here, we should just
        //       merge the stride_index_;
        acc_type = acc_type->merge(input_type);
      } else {
        acc_type = input_type;
      }
    } else {
      vec_optional_ttp.emplace_back(c10::nullopt);
    }
  }
  input_permutation_ = getPermutationPerSortedStride(acc_type);
  output_permutation_ = reversePermutation(input_permutation_);
  TORCH_CHECK(acc_type->device().has_value(),
      "requires fixed device for all inputs");
  device_ = acc_type->device();
}

bool GraphCache::InputsRequirement::requiresPermutation() {
  return input_permutation_ == output_permutation_;
}

bool GraphCache::InputsRequirement::complyWith(const InputsRequirement& expect) {
  if (device_ != expect.device_ ||
      input_permutation_ != expect.input_permutation_ ||
      output_permutation_ != expect.output_permutation_ ||
      vec_optional_ttp.size() != expect.vec_optional_ttp.size()) {
    return false;
  }

  // trick here is, `this` is always well defined while `expect` could has
  // missing options;
  for (int i = 0; i < static_cast<int>(vec_optional_ttp.size()); i++) {
    // TensorType has to match, otherwise it's not compatible to our graph.
    TORCH_INTERNAL_ASSERT(vec_optional_ttp[i].has_value() == expect.vec_optional_ttp[i].has_value());
    if (expect.vec_optional_ttp[i].has_value()) {
      // We assume that dimensionality should always match.
      TORCH_INTERNAL_ASSERT(
          (*expect.vec_optional_ttp[i])->symbolic_sizes().sizes().has_value() &&
          (*expect.vec_optional_ttp[i])->stride_properties().sizes().has_value() &&
          (*expect.vec_optional_ttp[i])->dim().has_value() &&
					(*vec_optional_ttp[i])->dim().value() &&
					(*expect.vec_optional_ttp[i])->dim().value() == (*vec_optional_ttp[i])->dim().value(),
					"expect fixed rank of tensors");

      int rank = static_cast<int>((*expect.vec_optional_ttp[i])->dim().value());
      auto vec_shape_symbol_ex = (*expect.vec_optional_ttp[i])->symbolic_sizes().sizes().value();
      auto vec_optional_stride_ex = (*expect.vec_optional_ttp[i])->stride_properties().sizes().value();
      auto vec_shape_symbol = (*vec_optional_ttp[i])->symbolic_sizes().sizes().value();
      auto vec_optional_stride = (*vec_optional_ttp[i])->stride_properties().sizes().value();
      for (int i = 0; i < rank; i++) {
        // if broadcast rule differs, compliance is broken;
        if ((vec_shape_symbol_ex[i].is_static() && vec_shape_symbol_ex[i].static_size() == 1) ^
						(vec_shape_symbol[i].is_static() && vec_shape_symbol[i].static_size() == 1)) {
          return false;
        }

        // if contiguity / stride index differ, compliance is broken;
        if (vec_optional_stride_ex[i].has_value() != vec_optional_stride[i].has_value()) {
          return false;
        }
        if (vec_optional_stride_ex[i].has_value() &&
           (vec_optional_stride_ex[i]->stride_index_ != vec_optional_stride[i]->stride_index_ ||
           vec_optional_stride_ex[i]->contiguous_ != vec_optional_stride[i]->contiguous_)) {
          return false;
        }
      }
    }
  }
  return true;
}

template <
    typename T,
    std::enable_if_t<std::is_constructible<GraphCache::InputsRequirement, T>::value, int> = 0>
FusionExecutorCache* GraphCache::createFusionExecutorCache(
    const T& meta_input_stack) {
  input_stacks_.emplace_back(meta_input_stack);
  std::shared_ptr<Graph> parsing_graph;
  // permute inputs on `Graph` to sort dimensions on common stride order;
  if (input_stacks_.back().requiresPermutation()) {
    auto input_permutation = input_stacks_.back().input_permutation_;
    // lambda to permute `TensorType` axes per `input_permutation`
    auto type_permute_fn = [&input_permutation](const TensorTypePtr& type) {
        // std::vector<c10::ShapeSymbol> vec_shape_symbol =
        // type->symbolic_sizes().sizes().value();
        auto vec_shape_symbol = type->symbolic_sizes().sizes().value();
        // std::vector<c10::optional<c10::Stride>> vec_optional_stride =
        // type->stride_properties().sizes().value();
        auto vec_optional_stride = type->stride_properties().sizes().value();
      
        int rank = static_cast<int>(type->dim().value());
      
        std::vector<c10::ShapeSymbol> permuted_vec_ss;
        std::vector<c10::optional<c10::Stride>> permuted_vec_optional_stride;
        for (int i = 0; i < rank; i++) {
          permuted_vec_ss.emplace_back(vec_shape_symbol[input_permutation[i]]);
          permuted_vec_optional_stride.emplace_back(
              vec_optional_stride[input_permutation[i]]);
        }
      
        return TensorType::create(
            type->scalarType(),
            type->device(),
            permuted_vec_ss,
            permuted_vec_optional_stride,
            type->requires_grad());
    };
    // copy `graph_` as `parsing_graph`
    parsing_graph = graph_->copy();
    for (auto input : parsing_graph->inputs()) {
      if (auto input_type = input->type()->cast<TensorType>()) {
        input->setType(type_permute_fn(input_type));
      }
    }
  } else {
    parsing_graph = graph_;
  }

  TORCH_INTERNAL_ASSERT(input_stacks_.back().device_.has_value(), "device is not set for fusion executor, something went wrong in NvFuser");
  fec_cache_.emplace_back(
      std::make_unique<FusionExecutorCache>(parseJitIR(parsing_graph), input_stacks_.back().device_.value()));
  return fec_cache_.back().get();
}

GraphCache::GraphCache(std::shared_ptr<Graph> graph)
    : graph_(std::move(graph)) {
  // compile a kernel if we have enough information from graph.
  has_reduction_ = graphHasReduction(graph_);

  // only compile graph when we have profiling record;
  std::shared_ptr<Graph> parsing_graph = graph_;
  if (IsNewExecutorEnabled()) {
    createFusionExecutorCache(graph_);
  }
}

std::vector<at::Tensor> GraphCache::runGraphWithInputs(
    const at::ArrayRef<IValue>& inputs) {
  InputsRequirement input_stack(inputs);
  FusionExecutorCache* fusion_executor_cache = nullptr;

  // TODO: hash indexing;
  for (int i = 0; i < static_cast<int>(fec_cache_.size()); i++) {
    if (input_stack.complyWith(input_stacks_[i])) {
      fusion_executor_cache = fec_cache_[i].get();
      break;
    }
  }
  if (!fusion_executor_cache) {
    fusion_executor_cache = createFusionExecutorCache(input_stack);
  }

  // GraphCache need to permute inputs/outputs to accommodate dimension coalescing
  if (input_stack.requiresPermutation()) {
		std::vector<IValue> permuted_inputs;
    permuted_inputs.reserve(inputs.size());
    for (const auto& input : inputs) {
      if (input.isTensor()) {
        permuted_inputs.emplace_back(input.toTensor().permute(input_stack.input_permutation_));
      } else {
        permuted_inputs.emplace_back(input);
      }
    }
    auto outputs = fusion_executor_cache->runFusionWithInputs(permuted_inputs);
    std::vector<at::Tensor> permuted_outputs;
    permuted_outputs.reserve(outputs.size());
    for (const auto& output : outputs) {
      permuted_outputs.emplace_back(output.permute(input_stack.output_permutation_));
    }
    return permuted_outputs;
  } else {
    return fusion_executor_cache->runFusionWithInputs(inputs);
  }
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
