
#include <torch/csrc/jit/codegen/cuda/expr_evaluator.h>
#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>

#include <iostream>

namespace torch {
namespace jit {
namespace fuser {

c10::optional<int> ExpressionEvaluator::evaluate(const Statement* expr) {
  ExpressionEvaluator evaluator;
  auto visitor = static_cast<OptInConstDispatch*>(&evaluator); // workaround
  if (expr->isVal()) {
    Val::constDispatch(visitor, expr->as<Val>());
  } else if (expr->isExpr()) {
    Expr::constDispatch(visitor, expr->as<Expr>());
  } else {
    TORCH_CHECK(!"Unexpected expression kind");
  }
  return evaluator.result_;
}

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
  result_ = i->evaluate();
}

void ExpressionEvaluator::handle(const NamedScalar* i) {
  // TODO
}

void ExpressionEvaluator::handle(const UnaryOp* uop) {
  // TODO
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
        result_ = (*lhs + *rhs -1) / *rhs;
        break;
      case BinaryOpType::And:
        result_ = int(*lhs && *rhs);
        break;
      default:
        TORCH_CHECK(!"Unexpected operator type");
    }
  }
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
