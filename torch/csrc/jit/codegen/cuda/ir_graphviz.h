
#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/csrc/jit/codegen/cuda/dispatch.h>

#include <string>
#include <unordered_map>

namespace torch {
namespace jit {
namespace fuser {

// Generates a DOT (https://www.graphviz.org) graph
// representation of the Fusion IR
struct TORCH_CUDA_API IrGraphGenerator : public OptInConstDispatch {
 public:
  static void print(const Fusion* fusion);

 private:
  IrGraphGenerator() = default;
  ~IrGraphGenerator() override = default;

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

  // lookup the graph id, creating one if not found
  std::string getid(const Statement* stm);

  void printArc(
      const Statement* src,
      const Statement* dst,
      const std::string& style = "");

  void printExpr(const Expr* expr, const std::string& label);

 private:
  std::unordered_map<const Statement*, std::string> id_map_;
  int next_id_ = 1;
};

} // namespace fuser
} // namespace jit
} // namespace torch
