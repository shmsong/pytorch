#include <torch/csrc/jit/codegen/cuda/expr_evaluator.h>
#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/instrumentation.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/ir_iostream.h>

#include <iostream>

namespace torch {
namespace jit {
namespace fuser {

void StatefulExpressionEvaluator_V1::safeBind(
    Val* value,
    Int::ScalarType concrete_value,
    GpuLower* lower) {
  auto already_concrete_val = getValue(value);

  if (already_concrete_val.has_value()) {
    TORCH_INTERNAL_ASSERT(
        concrete_value == already_concrete_val.value(),
        "Tried to bind ",
        value,
        " to ",
        " concrete value, but it's already set to ",
        already_concrete_val.value());
  } else {
    TORCH_INTERNAL_ASSERT(
        value->getOrigin() == nullptr,
        "Tried to bind to a value that is computed in the fusion IR. ",
        "Can only bind to symbolic values to the fusion that do not have an origin expr.");

    bindings_[value] = concrete_value;
  }

  if (lower != nullptr) {
    auto lowered_val = lower->getLowerValue(value);
    already_concrete_val = getValue(lowered_val);

    if (already_concrete_val.has_value()) {
      TORCH_INTERNAL_ASSERT(
          concrete_value == already_concrete_val.value(),
          "Tried to bind ",
          lowered_val,
          " to ",
          " concrete value, but it's already set to ",
          already_concrete_val.value());
    } else {
      TORCH_INTERNAL_ASSERT(
          lowered_val->getOrigin() == nullptr,
          "Tried to bind to a value that is computed in the fusion IR. ",
          "Can only bind to symbolic values to the fusion that do not have an origin expr.");

      bindings_[lowered_val] = concrete_value;
    }
  }
}

c10::optional<Int::ScalarType> StatefulExpressionEvaluator_V1::inferValue(
    Val* value) {
  FUSER_PERF_SCOPE("Evaluate Expression");
  return maybeHandle(value);
}

void StatefulExpressionEvaluator_V1::print() const {
  std::cout << "\nEvaluation context\n";
  std::cout << "--------------------\n";
  for (const auto& kv : bindings_) {
    std::cout << kv.first << " = " << kv.second;
    if (kv.first->isConstScalar()) {
      std::cout << " ; original value = "
                << kv.first->as<Int>()->value().value();
    }
    std::cout << " ; " << *kv.first->getValType() << "\n";
  }
  std::cout << "--------------------\n\n";
}

inline c10::optional<Int::ScalarType> StatefulExpressionEvaluator_V1::getValue(
    Val* value) {
  TORCH_INTERNAL_ASSERT(
      value->isAnInt(),
      "Expressoin Evaluation does not support values other than integers at this time.");

  auto v_type = value->getValType().value();
  bool is_named_scalar =
      v_type == ValType::NamedScalar || v_type == ValType::KirNamedScalar;

  if (!is_named_scalar && value->as<Int>()->value().has_value()) {
    return value->as<Int>()->value();
  }

  auto it = bindings_.find(value);
  if (it != bindings_.end()) {
    return c10::optional<Int::ScalarType>(it->second);
  }
  return c10::nullopt;
}

c10::optional<Int::ScalarType> StatefulExpressionEvaluator_V1::maybeHandle(
    Val* val) {
  auto maybe_concrete_value = getValue(val);
  if (!maybe_concrete_value.has_value()) {
    auto origin = val->getOrigin();
    if (origin != nullptr) {
      handle(origin);
      maybe_concrete_value = getValue(val);
    }
  }
  return maybe_concrete_value;
}

void StatefulExpressionEvaluator_V1::handle(UnaryOp* uop) {
  const auto in = maybeHandle(uop->in());
  if (in.has_value()) {
    switch (uop->getUnaryOpType()) {
      case UnaryOpType::Neg:
        bindings_[uop->out()] = -*in;
        break;
      case UnaryOpType::Cast:
        bindings_[uop->out()] = *in;
        break;
      default:
        TORCH_CHECK(!"Unexpected operator type");
    }
  }
}

void StatefulExpressionEvaluator_V1::handle(BinaryOp* bop) {
  const auto lhs = maybeHandle(bop->lhs());
  const auto rhs = maybeHandle(bop->rhs());
  if (lhs.has_value() && rhs.has_value()) {
    switch (bop->getBinaryOpType()) {
      case BinaryOpType::Add:
        bindings_[bop->out()] = *lhs + *rhs;
        break;
      case BinaryOpType::Sub:
        bindings_[bop->out()] = *lhs - *rhs;
        break;
      case BinaryOpType::Mul:
        bindings_[bop->out()] = *lhs * *rhs;
        break;
      case BinaryOpType::Div:
        TORCH_CHECK(*rhs != 0);
        bindings_[bop->out()] = *lhs / *rhs;
        break;
      case BinaryOpType::Mod:
        TORCH_CHECK(*rhs != 0);
        bindings_[bop->out()] = *lhs % *rhs;
        break;
      case BinaryOpType::CeilDiv:
        TORCH_CHECK(*rhs != 0);
        bindings_[bop->out()] = (*lhs + *rhs - 1) / *rhs;
        break;
      case BinaryOpType::And:
        bindings_[bop->out()] = Int::ScalarType(*lhs && *rhs);
        break;
      default:
        TORCH_CHECK(!"Unexpected operator type");
    }
  }
}

void StatefulExpressionEvaluator_V1::handle(kir::UnaryOp* uop) {
  const auto in = maybeHandle(uop->in());
  if (in.has_value()) {
    switch (uop->getUnaryOpType()) {
      case UnaryOpType::Neg:
        bindings_[uop->out()] = -*in;
        break;
      case UnaryOpType::Cast:
        bindings_[uop->out()] = *in;
        break;
      default:
        TORCH_CHECK(!"Unexpected operator type");
    }
  }
}

void StatefulExpressionEvaluator_V1::handle(kir::BinaryOp* bop) {
  const auto lhs = maybeHandle(bop->lhs());
  const auto rhs = maybeHandle(bop->rhs());
  if (lhs.has_value() && rhs.has_value()) {
    switch (bop->getBinaryOpType()) {
      case BinaryOpType::Add:
        bindings_[bop->out()] = *lhs + *rhs;
        break;
      case BinaryOpType::Sub:
        bindings_[bop->out()] = *lhs - *rhs;
        break;
      case BinaryOpType::Mul:
        bindings_[bop->out()] = *lhs * *rhs;
        break;
      case BinaryOpType::Div:
        TORCH_CHECK(*rhs != 0);
        bindings_[bop->out()] = *lhs / *rhs;
        break;
      case BinaryOpType::Mod:
        TORCH_CHECK(*rhs != 0);
        bindings_[bop->out()] = *lhs % *rhs;
        break;
      case BinaryOpType::CeilDiv:
        TORCH_CHECK(*rhs != 0);
        bindings_[bop->out()] = (*lhs + *rhs - 1) / *rhs;
        break;
      case BinaryOpType::And:
        bindings_[bop->out()] = Int::ScalarType(*lhs && *rhs);
        break;
      default:
        TORCH_CHECK(!"Unexpected operator type");
    }
  }
}

//==============================================================================

void StatefulExpressionEvaluator_V2::safeBind(
    Val* value,
    Int::ScalarType concrete_value,
    GpuLower* lower) {
  auto already_concrete_val = getValue(value);

  if (already_concrete_val.has_value()) {
    TORCH_INTERNAL_ASSERT(
        concrete_value == already_concrete_val.value(),
        "Tried to bind ",
        value,
        " to ",
        " concrete value, but it's already set to ",
        already_concrete_val.value());
  } else {
    TORCH_INTERNAL_ASSERT(
        value->getOrigin() == nullptr,
        "Tried to bind to a value that is computed in the fusion IR. ",
        "Can only bind to symbolic values to the fusion that do not have an origin expr.");

    bindings_[value] = concrete_value;
  }

  if (lower != nullptr) {
    auto lowered_val = lower->getLowerValue(value);
    already_concrete_val = getValue(lowered_val);

    if (already_concrete_val.has_value()) {
      TORCH_INTERNAL_ASSERT(
          concrete_value == already_concrete_val.value(),
          "Tried to bind ",
          lowered_val,
          " to ",
          " concrete value, but it's already set to ",
          already_concrete_val.value());
    } else {
      TORCH_INTERNAL_ASSERT(
          lowered_val->getOrigin() == nullptr,
          "Tried to bind to a value that is computed in the fusion IR. ",
          "Can only bind to symbolic values to the fusion that do not have an origin expr.");

      bindings_[lowered_val] = concrete_value;
    }
  }
}

c10::optional<Int::ScalarType> StatefulExpressionEvaluator_V2::inferValue(
    Val* value) {
  FUSER_PERF_SCOPE("Evaluate Expression");
  traverseFrom(fusion_, {value}, false);
  return maybeHandle(value);
}

void StatefulExpressionEvaluator_V2::print() const {
  std::cout << "\nEvaluation context\n";
  std::cout << "--------------------\n";
  for (const auto& kv : bindings_) {
    std::cout << kv.first << " = " << kv.second;
    if (kv.first->isConstScalar()) {
      std::cout << " ; original value = "
                << kv.first->as<Int>()->value().value();
    }
    std::cout << " ; " << *kv.first->getValType() << "\n";
  }
  std::cout << "--------------------\n\n";
}

inline c10::optional<Int::ScalarType> StatefulExpressionEvaluator_V2::getValue(
    Val* value) {
  TORCH_INTERNAL_ASSERT(
      value->isAnInt(),
      "Expressoin Evaluation does not support values other than integers at this time.");

  auto v_type = value->getValType().value();
  bool is_named_scalar =
      v_type == ValType::NamedScalar || v_type == ValType::KirNamedScalar;

  if (!is_named_scalar && value->as<Int>()->value().has_value()) {
    return value->as<Int>()->value();
  }

  auto it = bindings_.find(value);
  if (it != bindings_.end()) {
    return c10::optional<Int::ScalarType>(it->second);
  }
  return c10::nullopt;
}

c10::optional<Int::ScalarType> StatefulExpressionEvaluator_V2::maybeHandle(
    Val* val) {
  return getValue(val);
}

void StatefulExpressionEvaluator_V2::handle(NamedScalar* i) {
}

void StatefulExpressionEvaluator_V2::handle(Int* i) {
  if (i->value().has_value()) {
    bindings_[i] = *i->value();
  }
}

void StatefulExpressionEvaluator_V2::handle(kir::NamedScalar* i) {
}

void StatefulExpressionEvaluator_V2::handle(kir::Int* i) {
  if (i->value().has_value()) {
    bindings_[i] = *i->value();
  }
}

void StatefulExpressionEvaluator_V2::handle(UnaryOp* uop) {
  const auto in = maybeHandle(uop->in());
  if (in.has_value()) {
    switch (uop->getUnaryOpType()) {
      case UnaryOpType::Neg:
        bindings_[uop->out()] = -*in;
        break;
      case UnaryOpType::Cast:
        bindings_[uop->out()] = *in;
        break;
      default:
        TORCH_CHECK(!"Unexpected operator type");
    }
  }
}

void StatefulExpressionEvaluator_V2::handle(BinaryOp* bop) {
  const auto lhs = maybeHandle(bop->lhs());
  const auto rhs = maybeHandle(bop->rhs());
  if (lhs.has_value() && rhs.has_value()) {
    switch (bop->getBinaryOpType()) {
      case BinaryOpType::Add:
        bindings_[bop->out()] = *lhs + *rhs;
        break;
      case BinaryOpType::Sub:
        bindings_[bop->out()] = *lhs - *rhs;
        break;
      case BinaryOpType::Mul:
        bindings_[bop->out()] = *lhs * *rhs;
        break;
      case BinaryOpType::Div:
        TORCH_CHECK(*rhs != 0);
        bindings_[bop->out()] = *lhs / *rhs;
        break;
      case BinaryOpType::Mod:
        TORCH_CHECK(*rhs != 0);
        bindings_[bop->out()] = *lhs % *rhs;
        break;
      case BinaryOpType::CeilDiv:
        TORCH_CHECK(*rhs != 0);
        bindings_[bop->out()] = (*lhs + *rhs - 1) / *rhs;
        break;
      case BinaryOpType::And:
        bindings_[bop->out()] = Int::ScalarType(*lhs && *rhs);
        break;
      default:
        TORCH_CHECK(!"Unexpected operator type");
    }
  }
}

void StatefulExpressionEvaluator_V2::handle(kir::UnaryOp* uop) {
  const auto in = maybeHandle(uop->in());
  if (in.has_value()) {
    switch (uop->getUnaryOpType()) {
      case UnaryOpType::Neg:
        bindings_[uop->out()] = -*in;
        break;
      case UnaryOpType::Cast:
        bindings_[uop->out()] = *in;
        break;
      default:
        TORCH_CHECK(!"Unexpected operator type");
    }
  }
}

void StatefulExpressionEvaluator_V2::handle(kir::BinaryOp* bop) {
  const auto lhs = maybeHandle(bop->lhs());
  const auto rhs = maybeHandle(bop->rhs());
  if (lhs.has_value() && rhs.has_value()) {
    switch (bop->getBinaryOpType()) {
      case BinaryOpType::Add:
        bindings_[bop->out()] = *lhs + *rhs;
        break;
      case BinaryOpType::Sub:
        bindings_[bop->out()] = *lhs - *rhs;
        break;
      case BinaryOpType::Mul:
        bindings_[bop->out()] = *lhs * *rhs;
        break;
      case BinaryOpType::Div:
        TORCH_CHECK(*rhs != 0);
        bindings_[bop->out()] = *lhs / *rhs;
        break;
      case BinaryOpType::Mod:
        TORCH_CHECK(*rhs != 0);
        bindings_[bop->out()] = *lhs % *rhs;
        break;
      case BinaryOpType::CeilDiv:
        TORCH_CHECK(*rhs != 0);
        bindings_[bop->out()] = (*lhs + *rhs - 1) / *rhs;
        break;
      case BinaryOpType::And:
        bindings_[bop->out()] = Int::ScalarType(*lhs && *rhs);
        break;
      default:
        TORCH_CHECK(!"Unexpected operator type");
    }
  }
}

} // namespace fuser
} // namespace jit
} // namespace torch
