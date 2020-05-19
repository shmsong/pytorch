
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
    if (id->rawExtent() != id->extent()) {
      label_ << "\\<" << IrNodeLabel::gen(id->rawExtent()) << "\\>";
    }
    label_ << ")";
  }

  void handle(const Split* split) override {
    label_ << "Split(axis=" << split->axis()
           << ", factor=" << IrNodeLabel::gen(split->factor()) << ")";
  }

  void handle(const Merge* merge) override {
    label_ << "Merge(axis=" << merge->axis() << ")";
  }

  void handle(const Reorder* reorder) override {
    label_ << "Reorder( ";
    for (const int old_pos : reorder->pos2axis()) {
      label_ << old_pos << " ";
    }
    label_ << ")";
  }

 private:
  std::stringstream label_;
};

} // anonymous namespace

IrGraphGenerator::IrGraphGenerator(const Fusion* fusion, bool verbose)
    : verbose_(verbose) {
  // setup inputs & outputs
  // (indexes used to quickly check if a value is fusion input or output)
  for (const auto* input : fusion->inputs()) {
    TORCH_CHECK(inputs_.count(input) == 0);
    inputs_.insert(input);
  }
  for (const auto* output : fusion->outputs()) {
    TORCH_CHECK(outputs_.count(output) == 0);
    outputs_.insert(output);
  }
}

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
  handle(src);
  handle(dst);
  std::cout << "  " << getid(src) << " -> " << getid(dst) << " " << style
            << ";\n";
}

void IrGraphGenerator::printExpr(const Expr* expr, const std::string& label) {
  std::cout << "  " << getid(expr) << " "
            << "[label=\"" << label << "\", shape=oval, color=blue, "
            << "style=filled, fillcolor=azure];\n";
}

void IrGraphGenerator::printValue(const Val* val, const std::string& label) {
  std::cout << "  " << getid(val) << " [label=\"" << label
            << "\", shape=rect, color=green, fontsize=10];\n";
}

void IrGraphGenerator::print(const Fusion* fusion, bool verbose) {
  IrGraphGenerator ir_to_dot(fusion, verbose);
  std::cout << "\n//-------------------------------------\n\n";
  std::cout << "digraph fusion_ir {\n"
            << "  node [shape=circle, color=gray];\n"
            << "  edge [color=black];\n";

  // all expressions
  for (const auto* expr : fusion->unordered_exprs()) {
    ir_to_dot.handle(expr);
  }

  // all values (verbose only)
  if (verbose) {
    for (const auto* val : fusion->vals()) {
      ir_to_dot.handle(val);
    }
  }

  std::cout << "}\n";
  std::cout << "\n//-------------------------------------\n\n";
}

void IrGraphGenerator::handle(const TensorDomain* td) {
  std::cout << "  " << getid(td) << " [label=\"TensorDomain\", "
            << "shape=note, color=gray, fontsize=10];\n";
  for (auto iter_domain : td->domain()) {
    printArc(iter_domain, td, "[color=gray]");
  }
}

void IrGraphGenerator::handle(const IterDomain* id) {
  std::cout << "  " << getid(id) << " [label=\"" << IrNodeLabel::gen(id)
            << "\", shape=cds, color=gray, fontsize=10];\n";

  if (!id->start()->isZeroInt()) {
    printArc(id->start(), id, "[color=gray]");
  }
  printArc(id->rawExtent(), id, "[color=gray]");
  if (verbose_ && id->rawExtent() != id->extent()) {
    printArc(id->extent(), id, "[color=gray, style=dashed]");
  }
}

void IrGraphGenerator::handle(const TensorIndex* ti) {
  // TODO
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

  const bool is_input = inputs_.find(tv) != inputs_.end();
  const bool is_output = outputs_.find(tv) != outputs_.end();

  const char* style = is_input ? "style=filled, fillcolor=palegreen"
                               : is_output ? "style=filled, fillcolor=lightblue"
                                           : "style=filled, fillcolor=beige";

  std::cout << "  " << getid(tv) << " [label=\"" << label.str()
            << "\", shape=Mrecord, color=brown, " << style << "];\n";

  if (const auto* compute_at_view = tv->getComputeAtView()) {
    std::stringstream arc_style;
    arc_style << "[color=red, style=dashed, label=\""
              << "ComputeAt(" << tv->getComputeAtAxis() << ")\"]";
    printArc(tv, compute_at_view, arc_style.str());
  }

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
  printArc(bop->rhs(), bop, "[color=blue]");
  printArc(bop, bop->out());
}

void IrGraphGenerator::handle(const ForLoop* for_loop) {
  // TODO
  OptInConstDispatch::handle(for_loop);
}

void IrGraphGenerator::handle(const IfThenElse* if_then_else) {
  // TODO
  OptInConstDispatch::handle(if_then_else);
}

void IrGraphGenerator::handle(const Allocate* allocate) {
  // TODO
  OptInConstDispatch::handle(allocate);
}

void IrGraphGenerator::handle(const Split* split) {
  printExpr(split, IrNodeLabel::gen(split));
  printArc(split->in(), split);
  printArc(split, split->out());
}

void IrGraphGenerator::handle(const Merge* merge) {
  printExpr(merge, IrNodeLabel::gen(merge));
  printArc(merge->in(), merge);
  printArc(merge, merge->out());
}

void IrGraphGenerator::handle(const Reorder* reorder) {
  printExpr(reorder, IrNodeLabel::gen(reorder));
  printArc(reorder->in(), reorder);
  printArc(reorder, reorder->out());
}

} // namespace fuser
} // namespace jit
} // namespace torch
