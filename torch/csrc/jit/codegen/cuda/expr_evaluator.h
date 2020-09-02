
#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/csrc/jit/codegen/cuda/ir_interface_nodes.h>
#include <torch/csrc/jit/codegen/cuda/iter_visitor.h>
#include <torch/csrc/jit/codegen/cuda/lower2device.h>

#include <c10/util/Optional.h>
#include <c10/util/flat_hash_map.h>

#include <unordered_map>

namespace torch {
namespace jit {
namespace fuser {

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

  void handle(UnaryOp*) override;
  void handle(BinaryOp*) override;

  // TODO(kir): remove this
  void handle(kir::UnaryOp*) override;
  void handle(kir::BinaryOp*) override;

  c10::optional<Int::ScalarType> maybeHandle(Val*);
};

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

using StatefulExpressionEvaluator = StatefulExpressionEvaluator_V2;

} // namespace fuser
} // namespace jit
} // namespace torch
