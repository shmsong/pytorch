
#include <torch/csrc/jit/codegen/cuda/expr_evaluator.h>
#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>

#include <iostream>

namespace torch {
namespace jit {
namespace fuser {

c10::optional<int> ExpressionEvaluator::evaluate(const Statement* expr) {
  ExpressionEvaluator evaluator;
  evaluator.OptInConstDispatch::handle(expr);
  return evaluator.result_;
}

/*
void ExpressionEvaluator::handle(const TensorDomain* td) {
  // NOP
}

void ExpressionEvaluator::handle(const TensorView* tv) {
  // NOP
}

void ExpressionEvaluator::handle(const IterDomain* id) {
  // NOP
}

void ExpressionEvaluator::handle(const TensorIndex* ti) {
  // NOP
}
*/

void ExpressionEvaluator::handle(const Float* f) {
  // NOP
}

void ExpressionEvaluator::handle(const Int* i) {
  result_ = i->evaluate();
}

void ExpressionEvaluator::handle(const NamedScalar* i) {
  // NOP
}

void ExpressionEvaluator::handle(const UnaryOp* uop) {
  const auto in = evaluate(uop->in());
  if (in.has_value()) {
    switch (uop->getUnaryOpType()) {
      case UnaryOpType::Neg:
        result_ = int(!*in);
        break;
      case UnaryOpType::Cast:
        result_ = *in;
        break;
      default:
        TORCH_CHECK(!"Unexpected operator type");
    }
  }
}

void ExpressionEvaluator::handle(const BinaryOp* bop) {
  TORCH_CHECK(bop->out()->isAnInt()); // not really needed
  const auto lhs = evaluate(bop->lhs());
  const auto rhs = evaluate(bop->rhs());
  if (lhs.has_value() && rhs.has_value()) {
    switch (bop->getBinaryOpType()) {
      case BinaryOpType::Add:
        result_ = *lhs + *rhs;
        break;
      case BinaryOpType::Sub:
        result_ = *lhs - *rhs;
        break;
      case BinaryOpType::Mul:
        result_ = *lhs * *rhs;
        break;
      case BinaryOpType::Div:
        result_ = *lhs / *rhs;
        break;
      case BinaryOpType::Mod:
        TORCH_CHECK(*rhs != 0);
        result_ = *lhs % *rhs;
        break;
      case BinaryOpType::LT:
        result_ = int(*lhs < *rhs);
        break;
      case BinaryOpType::CeilDiv:
        TORCH_CHECK(*rhs != 0);
        result_ = (*lhs + *rhs - 1) / *rhs;
        break;
      case BinaryOpType::And:
        result_ = int(*lhs && *rhs);
        break;
      default:
        TORCH_CHECK(!"Unexpected operator type");
    }
  }
}

/*
void ExpressionEvaluator::handle(const ForLoop* fl) {
  // NOP
}

void ExpressionEvaluator::handle(const IfThenElse* ite) {
  // NOP
}

void ExpressionEvaluator::handle(const Allocate* a) {
  // NOP
}

void ExpressionEvaluator::handle(const Split* s) {
  // NOP
}

void ExpressionEvaluator::handle(const Merge* m) {
  // NOP
}

void ExpressionEvaluator::handle(const Reorder* ro) {
  // NOP
}
*/

} // namespace fuser
} // namespace jit
} // namespace torch
