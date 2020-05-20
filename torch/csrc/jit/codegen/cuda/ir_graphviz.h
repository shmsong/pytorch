
#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/csrc/jit/codegen/cuda/dispatch.h>

#include <string>
#include <unordered_map>
#include <unordered_set>

namespace torch {
namespace jit {
namespace fuser {

// Generates a DOT (https://www.graphviz.org) graph
// representation of the Fusion IR
class TORCH_CUDA_API IrGraphGenerator : private OptInConstDispatch {
 public:
  static void print(const Fusion* fusion, bool verbose = false);

 private:
  IrGraphGenerator(const Fusion* fusion, bool verbose);
  ~IrGraphGenerator() override = default;

  void handle(const Statement* s) override {
    if (!processed(s)) {
      OptInConstDispatch::handle(s);
    }
  };

  void handle(const Val* v) override {
    if (!processed(v)) {
      OptInConstDispatch::handle(v);
    }
  };

  void handle(const Expr* e) override {
    if (!processed(e)) {
      OptInConstDispatch::handle(e);
    }
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

  bool processed(const Statement* s) const {
    return id_map_.find(s) != id_map_.end();
  }

  void printArc(
      const Statement* src,
      const Statement* dst,
      const std::string& style = "");

  void printExpr(const Expr* expr, const std::string& label);
  void printValue(const Val* val, const std::string& label);

 private:
  const bool verbose_;
  std::unordered_map<const Statement*, std::string> id_map_;
  std::unordered_set<const Val*> inputs_;
  std::unordered_set<const Val*> outputs_;
  int next_id_ = 1;
};

} // namespace fuser
} // namespace jit
} // namespace torch
