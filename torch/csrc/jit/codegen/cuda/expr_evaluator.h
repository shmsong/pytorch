
#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/csrc/jit/codegen/cuda/dispatch.h>

#include <c10/util/Optional.h>

namespace torch {
namespace jit {
namespace fuser {

/*
 * TODO
 */

struct TORCH_CUDA_API ExpressionEvaluator : public OptInConstDispatch {
 public:
  static c10::optional<int> evaluate(const Statement* expr); // should be Expr*

 private:
  ExpressionEvaluator() = default;
  ~ExpressionEvaluator() override = default;

  void handle(const TensorDomain*) override;
  void handle(const TensorView*) override;
  void handle(const IterDomain*) override;
  void handle(const TensorIndex*) override;

  void handle(const Float*) override;
  void handle(const Int*) override;
  void handle(const NamedScalar*) override;

  void handle(const UnaryOp*) override;
  void handle(const BinaryOp*) override;

  void handle(const ForLoop*) override;
  void handle(const IfThenElse*) override;
  void handle(const Allocate*) override;

  void handle(const Split*) override;
  void handle(const Merge*) override;
  void handle(const Reorder*) override;

 private:
  c10::optional<int> result_;
};

} // namespace fuser
} // namespace jit
} // namespace torch
