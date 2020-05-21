
#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/ir_graphviz.h>
#include <torch/csrc/jit/codegen/cuda/type.h>

#include <fstream>

namespace torch {
namespace jit {
namespace fuser {

namespace {

// Private helper, generating node labels for IrGraphGenerator
class IrNodeLabel : private OptInConstDispatch {
  using DetailLevel = IrGraphGenerator::DetailLevel;

 public:
  static std::string gen(
      const Statement* node,
      DetailLevel detail_level = DetailLevel::Minimal) {
    IrNodeLabel generator(detail_level);
    generator.OptInConstDispatch::handle(node);
    return generator.label_.str();
  }

 private:
  explicit IrNodeLabel(DetailLevel detail_level)
      : detail_level_(detail_level) {}

  ~IrNodeLabel() override = default;

  void handle(const Float* f) override {
    if (f->isSymbolic()) {
      label_ << "f" << f->name();
    } else {
      if (detail_level_ > DetailLevel::Minimal) {
        label_ << "f" << f->name() << "=";
      }
      label_ << std::fixed << std::setprecision(2) << *f->value();
    }
  }

  void handle(const Int* i) override {
    if (i->isSymbolic()) {
      label_ << "i" << i->name();
    } else {
      if (detail_level_ > DetailLevel::Minimal) {
        label_ << "i" << i->name() << "=";
      }
      label_ << *i->value();
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
  const DetailLevel detail_level_;
};

} // anonymous namespace

void IrGraphGenerator::print(
    const Fusion* fusion,
    const std::string& filename,
    DetailLevel detail_level) {
  // output file
  std::ofstream dot_file(filename);
  TORCH_CHECK(dot_file.good(), "Failed to open the IR graph file");

  // generate the dot graph definition
  IrGraphGenerator ir_graph(fusion, detail_level);
  dot_file << ir_graph.generate();
}

IrGraphGenerator::IrGraphGenerator(
    const Fusion* fusion,
    DetailLevel detail_level)
    : detail_level_(detail_level), fusion_(fusion) {
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
    // First reference, generate a new id
    std::stringstream new_id;
    new_id << "stm_" << next_id_++;
    id_map_.insert({stm, new_id.str()});
    return new_id.str();
  } else {
    return it->second;
  }
}

// We automatically visit (handle) the arc's source and destination
void IrGraphGenerator::printArc(
    const Statement* src,
    const Statement* dst,
    const std::string& style) {
  handle(src);
  handle(dst);
  graph_def_ << "  " << getid(src) << " -> " << getid(dst) << " " << style
             << ";\n";
}

void IrGraphGenerator::printExpr(const Expr* expr, const std::string& label) {
  graph_def_ << "  " << getid(expr) << " "
             << "[label=\"" << label << "\", shape=oval, color=blue, "
             << "style=filled, fillcolor=azure];\n";
}

void IrGraphGenerator::printValue(const Val* val, const std::string& label) {
  graph_def_ << "  " << getid(val) << " [label=\"" << label
             << "\", shape=rect, color=green, fontsize=10];\n";
}

std::string IrGraphGenerator::generate() {
  // IrGraphGenerator instances are not reusable
  TORCH_CHECK(graph_def_.str().empty());
  TORCH_CHECK(visited_.empty());

  graph_def_ << "// detail level: ";
  switch (detail_level_) {
    case DetailLevel::Minimal:
      graph_def_ << "minimal\n";
      break;
    case DetailLevel::Explicit:
      graph_def_ << "explicit\n";
      break;
    case DetailLevel::Everything:
      graph_def_ << "everything\n";
      break;
    default:
      TORCH_CHECK(!"Unexpected detail level");
  }

  graph_def_ << "digraph fusion_ir {\n"
             << "  node [shape=circle, color=gray];\n"
             << "  edge [color=black];\n";

  // inputs
  for (const auto* input : fusion_->inputs()) {
    handle(input);
  }

  // outputs
  for (const auto* output : fusion_->outputs()) {
    handle(output);
  }

  // all expressions
  if (detail_level_ >= DetailLevel::Explicit) {
    for (const auto* expr : fusion_->unordered_exprs()) {
      handle(expr);
    }
  }

  // all values
  if (detail_level_ >= DetailLevel::Everything) {
    for (const auto* val : fusion_->vals()) {
      handle(val);
    }
  }

  graph_def_ << "}\n";

  // Make sure that all referenced nodes have been visited
  for (const auto& kv : id_map_) {
    TORCH_CHECK(visited(kv.first));
  }

  return graph_def_.str();
}

void IrGraphGenerator::handle(const Statement* s) {
  OptInConstDispatch::handle(s);
};

void IrGraphGenerator::handle(const Val* v) {
  if (!visited(v)) {
    visited_.insert(v);
    if (const auto* def = fusion_->origin(v)) {
      handle(def);
    }
    OptInConstDispatch::handle(v);
  }
};

void IrGraphGenerator::handle(const Expr* e) {
  if (!visited(e)) {
    visited_.insert(e);
    OptInConstDispatch::handle(e);
  }
};

void IrGraphGenerator::handle(const TensorDomain* td) {
  graph_def_ << "  " << getid(td) << " [label=\"TensorDomain\", "
             << "shape=note, color=gray, fontsize=10];\n";
  for (auto iter_domain : td->domain()) {
    printArc(iter_domain, td, "[color=gray]");
  }
}

void IrGraphGenerator::handle(const IterDomain* id) {
  graph_def_ << "  " << getid(id) << " [label=\"" << IrNodeLabel::gen(id)
             << "\", shape=cds, color=gray, fontsize=10];\n";

  if (!id->start()->isZeroInt()) {
    printArc(id->start(), id, "[color=gray]");
  }
  printArc(id->rawExtent(), id, "[color=gray]");
  if (detail_level_ > DetailLevel::Minimal && id->rawExtent() != id->extent()) {
    printArc(id->extent(), id, "[color=gray, style=dashed]");
  }
}

void IrGraphGenerator::handle(const TensorIndex* ti) {
  // TODO
  OptInConstDispatch::handle(ti);
}

void IrGraphGenerator::handle(const Float* f) {
  printValue(f, IrNodeLabel::gen(f, detail_level_));
}

void IrGraphGenerator::handle(const Int* i) {
  printValue(i, IrNodeLabel::gen(i, detail_level_));
}

void IrGraphGenerator::handle(const NamedScalar* i) {
  printValue(i, IrNodeLabel::gen(i, detail_level_));
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

  graph_def_ << "  " << getid(tv) << " [label=\"" << label.str()
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
