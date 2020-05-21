
#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/csrc/jit/codegen/cuda/dispatch.h>

#include <sstream>
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
  enum class DetailLevel {
    ComputeOnly, // Only dataflow (compute) nodes
    Basic, // Compute + schedule, with minimal details (default)
    Explicit, // Additional details (ex. symbolic names for scalar constants)
    Verbose, // Includes all values and dead definitions
  };

 public:
  // This is the public interface to IrGraphGenerator
  static void print(
      const Fusion* fusion,
      const std::string& filename,
      DetailLevel detail_level = DetailLevel::Basic);

 private:
  IrGraphGenerator(const Fusion* fusion, DetailLevel detail_level);
  ~IrGraphGenerator() override = default;

  std::string generate();

  void handle(const Statement*) override;
  void handle(const Val*) override;
  void handle(const Expr*) override;

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

  bool visited(const Statement* s) const {
    return visited_.find(s) != visited_.end();
  }

  void printArc(
      const Statement* src,
      const Statement* dst,
      const std::string& style = "");

  void printExpr(const Expr* expr, const std::string& label);
  void printValue(const Val* val, const std::string& label);

 private:
  const DetailLevel detail_level_;
  const Fusion* const fusion_;
  std::stringstream graph_def_;
  std::unordered_map<const Statement*, std::string> id_map_;
  std::unordered_set<const Statement*> visited_;
  std::unordered_set<const Val*> inputs_;
  std::unordered_set<const Val*> outputs_;
  int next_id_ = 1;
};

} // namespace fuser
} // namespace jit
} // namespace torch
