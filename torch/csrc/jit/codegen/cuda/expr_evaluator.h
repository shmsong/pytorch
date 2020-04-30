
#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h>

#include <torch/csrc/jit/codegen/cuda/dispatch.h>

namespace torch {
namespace jit {
namespace fuser {

struct Fusion;

struct Statement;

struct Val;
struct Expr;

struct UnaryOp;
struct BinaryOp;

struct ForLoop;
struct IfThenElse;

struct TensorDomain;
struct TensorView;
struct IterDomain;
struct TensorIndex;

struct TensorContiguity;

struct Split;
struct Merge;
struct Reorder;

struct Float;
struct Int;
struct Add;

/*
 * TODO
 */

struct TORCH_CUDA_API ExpressionEvaluator : public OptInConstDispatch {
 public:
  ExpressionEvaluator() = default;
  ~ExpressionEvaluator() override = default;

  void handle(const Statement* s) override {
    OptInConstDispatch::handle(s);
  };

  void handle(const Val* v) override {
    OptInConstDispatch::handle(v);
  };

  void handle(const Expr* e) override {
    OptInConstDispatch::handle(e);
  };

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
};

} // namespace fuser
} // namespace jit
} // namespace torch
