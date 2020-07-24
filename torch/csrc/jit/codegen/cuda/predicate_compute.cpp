#include <torch/csrc/jit/codegen/cuda/predicate_compute.h>

#include <torch/csrc/jit/codegen/cuda/arith.h>
#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/transform_iter.h>

namespace torch {
namespace jit {
namespace fuser {

bool PredicateCompute::hasPredicates(const kir::TensorIndex* ti) {
  std::vector<Bool*> preds;
  for (auto ind : ti->indices())
    if (FusionGuard::getCurFusion()->origin(ind) != nullptr)
      return true;
  return false;
}

std::vector<kir::Bool*> PredicateCompute::computePredicates(
    const kir::TensorIndex* ti) {
  const TensorView* tv = ti->view();
  const std::vector<IterDomain*>& root = tv->getRootDomain();

  std::vector<kir::Bool*> preds;

  bool no_pred_needed = true;
  for (auto id : tv->domain()->domain())
    if (id->getOrigin() != nullptr)
      no_pred_needed = false;

  if (no_pred_needed)
    return preds;

  TORCH_INTERNAL_ASSERT(root.size() == ti->nDims());
  for (size_t i = 0; i < ti->nDims(); i++)

    // I believe the second part of this check is redundant, but it doesn't
    // hurt.
    if (FusionGuard::getCurFusion()->origin(ti->index(i)) != nullptr &&
        !root[i]->isBroadcast()) {
      auto pred = kir::ltExpr(ti->index(i), root[i]->extent());
      TORCH_INTERNAL_ASSERT(
          pred->getValType().value() == ValType::KirScalar &&
          pred->getDataType().value() == DataType::Bool);
      preds.push_back(pred->as<kir::Bool>());
    } else {
      preds.push_back(new kir::Bool(true));
    }

  return preds;
}

} // namespace fuser
} // namespace jit
} // namespace torch
