
#include <torch/csrc/jit/codegen/cuda/expr_evaluator.h>
#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>

#include <iostream>

namespace torch {
namespace jit {
namespace fuser {

void ExpressionEvaluator::handle(const TensorDomain* td) {
  // TODO
}

void ExpressionEvaluator::handle(const TensorView* tv) {
  // TODO
}

void ExpressionEvaluator::handle(const IterDomain* id) {
  // TODO
}

void ExpressionEvaluator::handle(const TensorIndex* ti) {
  // TODO
}

void ExpressionEvaluator::handle(const Float* f) {
  // TODO
}

void ExpressionEvaluator::handle(const Int* i) {
  // TODO
}

void ExpressionEvaluator::handle(const NamedScalar* i) {
  // TODO
}

void ExpressionEvaluator::handle(const UnaryOp* uop) {
  // TODO
}

void ExpressionEvaluator::handle(const BinaryOp* bop) {
  // TODO
}

void ExpressionEvaluator::handle(const ForLoop* fl) {
  // TODO
}

void ExpressionEvaluator::handle(const IfThenElse* ite) {
  // TODO
}

void ExpressionEvaluator::handle(const Allocate* a) {
  // TODO
}

void ExpressionEvaluator::handle(const Split* s) {
  // TODO
}

void ExpressionEvaluator::handle(const Merge* m) {
  // TODO
}

void ExpressionEvaluator::handle(const Reorder* ro) {
  // TODO
}

} // namespace fuser
} // namespace jit
} // namespace torch
