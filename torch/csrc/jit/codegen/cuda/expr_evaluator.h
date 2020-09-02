
#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/csrc/jit/codegen/cuda/ir_interface_nodes.h>
#include <torch/csrc/jit/codegen/cuda/iter_visitor.h>
#include <torch/csrc/jit/codegen/cuda/lower2device.h>
#include <torch/csrc/jit/codegen/cuda/dispatch.h>

#include <c10/util/Optional.h>
#include <c10/util/flat_hash_map.h>

#include <unordered_map>

namespace torch {
namespace jit {
namespace fuser {

// Encapsulates a set of value bindings on top of a Fusion IR
// (used to provide known values to ExpressionEvaluator)
//
// NOTE: currently it only supports Int values
//
class TORCH_CUDA_API EvaluationContext {
  friend class ExpressionEvaluator;

 public:
  explicit EvaluationContext(Fusion* fusion) : fusion_(fusion) {}

  // Set the concrete value for a Int*
  void bind(const Val* value, Int::ScalarType concrete_value);

  // Retrieves the concrete value, or nullopt if not set
  c10::optional<Int::ScalarType> concreteValue(const Val* value) const;

  Fusion* fusion() const {
    return fusion_;
  }

  // Debugging helper, prints all the currently set values
  void print() const;

 private:
  std::unordered_map<const Val*, Int::ScalarType> bindings_;
  Fusion* fusion_ = nullptr;
};

// Evaluates expressions in a Fusion IR, using the passed in
// context (EvaluationContext) to query for concrete_values. The
// evaluation context may override concrete values in the IR as well.
class TORCH_CUDA_API ExpressionEvaluator : private OptInConstDispatch {
 public:
  // Returns the result of the specified expression, or nullopt if
  // the result cannot be evaluated
  static c10::optional<Int::ScalarType> evaluate(
      Statement* val,
      EvaluationContext* context);

 private:
  explicit ExpressionEvaluator(EvaluationContext* context)
      : context_(context) {}

  ~ExpressionEvaluator() override = default;

  c10::optional<Int::ScalarType> value(const Statement* stmt) const;

  void handle(const NamedScalar*) override;
  void handle(const Int*) override;
  void handle(const UnaryOp*) override;
  void handle(const BinaryOp*) override;

  // TODO(kir): remove this
  void handle(const kir::NamedScalar*) override;
  void handle(const kir::Int*) override;
  void handle(const kir::UnaryOp*) override;
  void handle(const kir::BinaryOp*) override;

 private:
  EvaluationContext* context_ = nullptr;
  c10::optional<Int::ScalarType> result_;
};

//==============================================================================

class TORCH_CUDA_API StatefulExpressionEvaluator_V1 : private OptOutDispatch {
 public:
  explicit StatefulExpressionEvaluator_V1(Fusion* fusion) : fusion_(fusion) {}

  Fusion* fusion() const {
    return fusion_;
  }

  void safeBind(
      Val* value,
      Int::ScalarType concrete_value,
      GpuLower* lower = nullptr);

  // Returns value if found in mapping, otherwise returns c10::nullopt
  c10::optional<Int::ScalarType> getValue(Val* value);

  // Checks if value is already infered, returns infered value if so, otherwise
  // runs traversal on value. Warning: should not be called in traversal.
  c10::optional<Int::ScalarType> inferValue(Val* value);

  // Debugging helper, prints all the currently set values
  void print() const;

 private:
  std::unordered_map<const Val*, Int::ScalarType> bindings_;
  Fusion* fusion_ = nullptr;

  using OptOutDispatch::handle;

 private:
  #if 1
  void handle(Expr* expr) override {
    switch (expr->getExprType().value()) {
      case ExprType::UnaryOp:
        handle(expr->as<UnaryOp>());
        break;
      case ExprType::BinaryOp:
        handle(expr->as<BinaryOp>());
        break;
      case ExprType::KirUnaryOp:
        handle(expr->as<kir::UnaryOp>());
        break;
      case ExprType::KirBinaryOp:
        handle(expr->as<kir::BinaryOp>());
        break;
      default:
        TORCH_INTERNAL_ASSERT(
            false,
            "Cannot handle Expr type: ",
            expr->getExprType().value(),
            " in stateful expression evaluator.");
    }
  }
  #endif

  void handle(UnaryOp*) override;
  void handle(BinaryOp*) override;

  // TODO(kir): remove this
  void handle(kir::UnaryOp*) override;
  void handle(kir::BinaryOp*) override;

  c10::optional<Int::ScalarType> maybeHandle(Val*);
};

//==============================================================================

class TORCH_CUDA_API StatefulExpressionEvaluator_V2 : private IterVisitor {
 public:
  explicit StatefulExpressionEvaluator_V2(Fusion* fusion) : fusion_(fusion) {}

  Fusion* fusion() const {
    return fusion_;
  }

  void safeBind(
      Val* value,
      Int::ScalarType concrete_value,
      GpuLower* lower = nullptr);

  // Returns value if found in mapping, otherwise returns c10::nullopt
  c10::optional<Int::ScalarType> getValue(Val* value);

  // Checks if value is already infered, returns infered value if so, otherwise
  // runs traversal on value. Warning: should not be called in traversal.
  c10::optional<Int::ScalarType> inferValue(Val* value);

  // Debugging helper, prints all the currently set values
  void print() const;

 private:
  std::unordered_map<const Val*, Int::ScalarType> bindings_;
  Fusion* fusion_ = nullptr;

  using IterVisitor::handle;

 private:
  void handle(NamedScalar*) override;
  void handle(Int*) override;
  void handle(UnaryOp*) override;
  void handle(BinaryOp*) override;

  // TODO(kir): remove this
  void handle(kir::NamedScalar*) override;
  void handle(kir::Int*) override;
  void handle(kir::UnaryOp*) override;
  void handle(kir::BinaryOp*) override;

  c10::optional<Int::ScalarType> maybeHandle(Val*);
};

//==============================================================================

class TORCH_CUDA_API StatefulExpressionEvaluator_V3 {
 public:
  explicit StatefulExpressionEvaluator_V3(Fusion* fusion) : context_(fusion) {}

  Fusion* fusion() const {
    return context_.fusion();
  }

  void safeBind(
      Val* value,
      Int::ScalarType concrete_value,
      GpuLower* lower = nullptr);

  // Returns value if found in mapping, otherwise returns c10::nullopt
  c10::optional<Int::ScalarType> getValue(Val* value);

  // Checks if value is already infered, returns infered value if so, otherwise
  // runs traversal on value. Warning: should not be called in traversal.
  c10::optional<Int::ScalarType> inferValue(Val* value);

 private:
  EvaluationContext context_;
};

using StatefulExpressionEvaluator = StatefulExpressionEvaluator_V3;

} // namespace fuser
} // namespace jit
} // namespace torch
