
#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/ir_graphviz.h>
#include <torch/csrc/jit/codegen/cuda/type.h>

#include <iostream>
#include <sstream>

namespace torch {
namespace jit {
namespace fuser {

std::string IrGraphGenerator::getid(const Statement* stm) {
  const auto it = id_map_.find(stm);
  if (it == id_map_.end()) {
    std::stringstream new_id;
    new_id << "stm_" << next_id_++;
    id_map_.insert({stm, new_id.str()});
    return new_id.str();
  } else {
    return it->second;
  }
}

void IrGraphGenerator::printArc(
    const Statement* src,
    const Statement* dst,
    const std::string& style) {
  if (arcs_.find({src, dst}) == arcs_.end()) {
    std::cout << "  " << getid(src) << " -> " << getid(dst) << " " << style
              << ";\n";
    arcs_.insert({src, dst});
  }
}

void IrGraphGenerator::printExpr(const Expr* expr, const std::string& label) {
  // node
  std::cout << "  " << getid(expr) << " "
            << "[label=\"" << label << "\", shape=rect, color=blue];\n";

  // generic (IRInputOutput) inputs & outputs
  // (paranoid - just to make sure we're not missing anything)
  for (const auto* val : expr->inputs()) {
    printArc(val, expr);
  }
  for (const auto* val : expr->outputs()) {
    printArc(expr, val);
  }
}

void IrGraphGenerator::printValue(const Val* val, const std::string& label) {
  std::cout << "  " << getid(val) << " [label=\"" << label
            << "\", shape=circle, color=green];\n";
}

void IrGraphGenerator::print(const Fusion* fusion) {
  IrGraphGenerator ir_to_dot;
  std::cout << "digraph fusion_ir {\n"
            << "  rankdir=LR;\n"
            << "  node [shape=circle, color=gray];\n"
            << "  edge [color=black];\n";
  for (const auto* expr : fusion->unordered_exprs()) {
    ir_to_dot.handle(expr);
  }
  for (const auto* val : fusion->vals()) {
    ir_to_dot.handle(val);
  }
  std::cout << "}\n";
}

void IrGraphGenerator::handle(const TensorDomain* td) {
  // TODO
}

void IrGraphGenerator::handle(const TensorView* tv) {
  // TODO
}

void IrGraphGenerator::handle(const IterDomain* id) {
  // TODO
}

void IrGraphGenerator::handle(const TensorIndex* ti) {
  // TODO
}

void IrGraphGenerator::handle(const Float* f) {
  std::stringstream label;
  if (f->isSymbolic()) {
    label << "f" << f->name();
  } else {
    label << std::fixed << std::setprecision(2) << *(f->value());
  }
  printValue(f, label.str());
}

void IrGraphGenerator::handle(const Int* i) {
  std::stringstream label;
  if (i->isSymbolic()) {
    label << "i" << i->name();
  } else {
    label << *(i->value());
  }
  printValue(i, label.str());
}

void IrGraphGenerator::handle(const NamedScalar* i) {
  printValue(i, i->name());
}

void IrGraphGenerator::handle(const UnaryOp* uop) {
  // node
  std::stringstream label;
  label << "UnaryOp(" << uop->getUnaryOpType() << ")";
  printExpr(uop, label.str());

  // UnaryOp inputs & outputs
  printArc(uop->in(), uop);
  printArc(uop, uop->out());
}

void IrGraphGenerator::handle(const BinaryOp* bop) {
  // node
  std::stringstream label;
  label << "BinaryOp(" << bop->getBinaryOpType() << ")";
  printExpr(bop, label.str());

  // BinaryOp inputs & outputs
  printArc(bop->lhs(), bop);
  printArc(bop->rhs(), bop);
  printArc(bop, bop->out());
}

void IrGraphGenerator::handle(const ForLoop* fl) {
  // TODO
}

void IrGraphGenerator::handle(const IfThenElse* ite) {
  // TODO
}

void IrGraphGenerator::handle(const Allocate* a) {
  // TODO
}

void IrGraphGenerator::handle(const Split* s) {
  // TODO
}

void IrGraphGenerator::handle(const Merge* m) {
  // TODO
}

void IrGraphGenerator::handle(const Reorder* ro) {
  // TODO
}

} // namespace fuser
} // namespace jit
} // namespace torch
