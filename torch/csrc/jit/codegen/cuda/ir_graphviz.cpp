
#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/ir_graphviz.h>
#include <torch/csrc/jit/codegen/cuda/type.h>

#include <iostream>
#include <sstream>

namespace torch {
namespace jit {
namespace fuser {

namespace {

struct TORCH_CUDA_API IrNodeLabel : public OptInConstDispatch {
 public:
  static std::string gen(const Statement* node) {
    IrNodeLabel generator;
    generator.OptInConstDispatch::handle(node);
    return generator.label_.str();
  }

 private:
  IrNodeLabel() = default;
  ~IrNodeLabel() override = default;

  void handle(const Float* f) override {
    if (f->isSymbolic()) {
      label_ << "f" << f->name();
    } else {
      label_ << std::fixed << std::setprecision(2) << *(f->value());
    }
  }

  void handle(const Int* i) override {
    if (i->isSymbolic()) {
      label_ << "i" << i->name();
    } else {
      label_ << *(i->value());
    }
  }

  void handle(const NamedScalar* ns) override {
    label_ << ns->name();
  }

  void handle(const IterDomain* id) override {
    if (id->isReduction()) {
      label_ << "r";
    } else {
      label_ << "i";
    }

    switch (id->parallel_method()) {
      case (ParallelType::Vectorize):
        label_ << "V";
        break;
      case (ParallelType::Unroll):
        label_ << "U";
        break;
      case (ParallelType::Serial):
        label_ << "S";
        break;
      default:
        label_ << id->parallel_method();
    }

    label_ << "(";
    if (!id->start()->isZeroInt()) {
      label_ << IrNodeLabel::gen(id->start()) << " : ";
    }
    label_ << IrNodeLabel::gen(id->extent());
    label_ << ")";
  }

 private:
  std::stringstream label_;
};

} // anonymous namespace

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
  std::cout << "  " << getid(src) << " -> " << getid(dst) << " " << style
            << ";\n";
}

void IrGraphGenerator::printExpr(const Expr* expr, const std::string& label) {
  // node
  std::cout << "  " << getid(expr) << " "
            << "[label=\"" << label << "\", shape=oval, color=blue];\n";

  // generic (IRInputOutput) inputs & outputs
  // (paranoid - just to make sure we're not missing anything)
#if 0
  if (verbose_) {
    for (const auto* val : expr->inputs()) {
      printArc(val, expr);
    }
    for (const auto* val : expr->outputs()) {
      printArc(expr, val);
    }
  }
#endif
}

void IrGraphGenerator::printValue(const Val* val, const std::string& label) {
  std::cout << "  " << getid(val) << " [label=\"" << label
            << "\", shape=rect, color=green, fontsize=10];\n";
}

void IrGraphGenerator::print(const Fusion* fusion, bool verbose) {
  IrGraphGenerator ir_to_dot(verbose);
  std::cout << "\n//-------------------------------------\n";
  std::cout << "digraph fusion_ir {\n"
            << "  node [shape=circle, color=gray];\n"
            << "  edge [color=black];\n";
  for (const auto* expr : fusion->unordered_exprs()) {
    ir_to_dot.handle(expr);
  }
  for (const auto* val : fusion->vals()) {
    ir_to_dot.handle(val);
  }
  std::cout << "}\n";
  std::cout << "\n//-------------------------------------\n";
}

void IrGraphGenerator::handle(const TensorDomain* td) {
  std::cout << "  " << getid(td)
            << " [label=\"TensorDomain\", shape=note, color=gray];\n";
  for (auto iter_domain : td->domain()) {
    printArc(iter_domain, td);
  }
}

void IrGraphGenerator::handle(const IterDomain* id) {
  if (!id->start()->isZeroInt()) {
    printArc(id->start(), id);
  }
  printArc(id->extent(), id);

  std::cout << "  " << getid(id) << " [label=\"" << IrNodeLabel::gen(id)
            << "\", shape=rect, color=gray, fontsize=10];\n";
}

void IrGraphGenerator::handle(const TensorIndex* ti) {
  OptInConstDispatch::handle(ti);
}

void IrGraphGenerator::handle(const Float* f) {
  printValue(f, IrNodeLabel::gen(f));
}

void IrGraphGenerator::handle(const Int* i) {
  printValue(i, IrNodeLabel::gen(i));
}

void IrGraphGenerator::handle(const NamedScalar* i) {
  printValue(i, IrNodeLabel::gen(i));
}

void IrGraphGenerator::handle(const TensorView* tv) {
  std::stringstream label;
  label << "{T" << tv->name() << "|";
  label << "{";
  bool first_axis = true;
  for (auto iter_domain : tv->domain()->domain()) {
    if (first_axis) {
      first_axis = false;
    } else {
      label << "|";
    }
    label << IrNodeLabel::gen(iter_domain);
  }
  label << "}}";
  std::cout << "  " << getid(tv) << " [label=\"" << label.str()
            << "\", shape=Mrecord, color=brown];\n";

  printArc(tv->domain(), tv, "[style=dashed, color=gray]");
}

void IrGraphGenerator::handle(const UnaryOp* uop) {
  // node
  std::stringstream label;
  label << uop->getUnaryOpType();
  printExpr(uop, label.str());

  // UnaryOp inputs & outputs
  printArc(uop->in(), uop);
  printArc(uop, uop->out());
}

void IrGraphGenerator::handle(const BinaryOp* bop) {
  // node
  std::stringstream label;
  label << bop->getBinaryOpType();
  printExpr(bop, label.str());

  // BinaryOp inputs & outputs
  printArc(bop->lhs(), bop);
  printArc(bop->rhs(), bop, "[color=green]");
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
