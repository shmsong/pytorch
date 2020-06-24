#include <torch/csrc/jit/codegen/cuda/arith.h>
#include <torch/csrc/jit/codegen/cuda/expr_evaluator.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/scheduler.h>

namespace torch {
namespace jit {
namespace fuser {

bool Scheduler::reduction(Fusion *fusion, const at::ArrayRef<IValue> &inputs) {
  // TODO: I am making a larger initial assumption that reductions are
  //       2D at this point to make the issue easier, right now.

  // Find Reduction TensorView
  TensorView* red_tv = nullptr;
  for (auto &expr : fusion->exprs(/*from_outputs_only*/true)) {
    if(expr->type() == ExprType::ReductionOp) {
      red_tv = static_cast<TensorView*>(expr->output(0));
    }
  }
  if (red_tv == nullptr) // No reduction found
    return false;

  EvaluationContext eval_context(fusion);

  // I am making some basic assumptions
  // 1.) I am only binding Tensor Dimension sizes.  I am not binding scalar values.
  // 2.) I am only binding the IterDomain.extent().  Do I need to worry about the start?
  for(size_t i = 0; i < inputs.size(); ++i) {
    if(inputs[i].type()->kind() == c10::TypeKind::TensorType) {
      TensorView* tv = static_cast<TensorView*>(fusion->inputs()[i]);
      size_t dims = tv->getRootDomain().size();
      for(size_t j = 0; j < dims; ++j) {
        IterDomain* id = tv->getRootDomain()[j];
        eval_context.bind(id->extent(), inputs[i].toTensor().size(j));
      }
    }
  }

  // Evaluate Dimensions of Reduction TensorView
  auto red_ids = red_tv->domain()->domain();
  std::vector<Int::ScalarType> red_dims(red_ids.size(), 0);
  size_t red_idx = 0;
  for(size_t i = 0; i < red_ids.size(); ++i) {
    red_dims[i] = ExpressionEvaluator::evaluate(red_ids[i]->extent(), &eval_context).value();
    if (red_ids[i]->isReduction())
      red_idx = i;
  }

  std::cout << "Reduction Dims: [";
  for(auto &dim : red_dims) {
    std::cout << dim << ",";
  }
  std::cout << "]" << std::endl;

  // Heuristic Definition
  // TODO: Need to factor in unrolling
  // TODO: Need to get rid of magic numbers.  These should be defined elsewhere.
  if (red_idx == (red_dims.size()-1)) {
    if(red_dims[red_idx] >= 128) {
      red_tv->split(red_idx, 128);
      auto red_rf_tv = red_tv->rFactor({-2});

      // TODO: How do I set blocks and threads to other tensors besides those of the reduction?

      red_rf_tv->axis(0)->parallelize(ParallelType::BIDx);
      red_tv->axis(0)->parallelize(ParallelType::BIDx);

      red_rf_tv->axis(-1)->parallelize(ParallelType::TIDx);
      red_tv->axis(-1)->parallelize(ParallelType::TIDx);
    } else {
      // Simply Set blocks/threads
      red_tv->axis(0)->parallelize(ParallelType::BIDx);
      red_tv->axis(-1)->parallelize(ParallelType::TIDx);
    }
  } else {
    red_tv->split(-1, 2);
    red_tv->axis(-2)->parallelize(ParallelType::BIDx);
    red_tv->axis(-1)->parallelize(ParallelType::TIDx);
  }

  return true;
}

} // namespace fuser
} // namespace jit
} // namespace torch
