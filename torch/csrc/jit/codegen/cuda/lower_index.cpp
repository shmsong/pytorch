#include <torch/csrc/jit/codegen/cuda/index_compute.h>
#include <torch/csrc/jit/codegen/cuda/ir_iostream.h>
#include <torch/csrc/jit/codegen/cuda/lower_utils.h>

#include <torch/csrc/jit/codegen/cuda/lower_index.h>

namespace torch {
namespace jit {
namespace fuser {

void IndexLowering::pushBack(Expr* expr) {
  if (active_scope == nullptr)
    lowered_exprs.push_back(expr);
  else
    active_scope->push_back(expr);
}

void IndexLowering::handle(IfThenElse* ite) {
  Expr* prev_scope_expr = active_scope_expr;
  Scope* prev_scope = active_scope;

  auto new_ite = new IfThenElse(ite->cond(), {}, {}, prev_scope_expr);
  pushBack(new_ite);
  active_scope_expr = new_ite;
  active_scope = &new_ite->body();

  for (auto expr : ite->body().exprs()) {
    OptInDispatch::handle(expr);
  }

  active_scope = &new_ite->elseBody();

  for (auto expr : ite->elseBody().exprs()) {
    OptInDispatch::handle(expr);
  }

  active_scope = prev_scope;
  active_scope_expr = prev_scope_expr;
}

void IndexLowering::handle(ForLoop* fl) {
  Expr* prev_scope_expr = active_scope_expr;
  Scope* prev_scope = active_scope;

  auto newFl = new ForLoop(fl->index(), fl->iter_domain(), {}, prev_scope_expr);
  pushBack(newFl);

  active_scope_expr = newFl;
  active_scope = &newFl->body();

  for (auto expr : fl->body().exprs()) {
    OptInDispatch::handle(expr);
  }

  active_scope = prev_scope;
  active_scope_expr = prev_scope_expr;
}

void IndexLowering::handle(UnaryOp* uop) {
  if (!ir_utils::isTVOp(uop)) {
    pushBack(uop);
    return;
  }

  TensorIndex* out = Index::getConsumerIndex(
      ir_utils::asTV(uop->out()), scope_utils::getLoops(active_scope_expr));
  Val* in = uop->in();
  if (ir_utils::isTV(in))
    in = Index::getProducerIndex(
        ir_utils::asTV(in),
        ir_utils::asTV(uop->out()),
        scope_utils::getLoops(active_scope_expr));
  pushBack(new UnaryOp(uop->getUnaryOpType(), out, in));
}

void IndexLowering::handle(BinaryOp* bop) {
  if (!ir_utils::isTVOp(bop)) {
    pushBack(bop);
    return;
  }

  TensorIndex* out = Index::getConsumerIndex(
      ir_utils::asTV(bop->out()), scope_utils::getLoops(active_scope_expr));

  Val* lhs = bop->lhs();
  Val* rhs = bop->rhs();

  if (ir_utils::isTV(lhs))
    lhs = Index::getProducerIndex(
        ir_utils::asTV(lhs),
        ir_utils::asTV(bop->out()),
        scope_utils::getLoops(active_scope_expr));

  if (ir_utils::isTV(rhs))
    rhs = Index::getProducerIndex(
        ir_utils::asTV(rhs),
        ir_utils::asTV(bop->out()),
        scope_utils::getLoops(active_scope_expr));

  pushBack(new BinaryOp(bop->getBinaryOpType(), out, lhs, rhs));
}

void IndexLowering::handle(TernaryOp* top) {
  if (!ir_utils::isTVOp(top)) {
    pushBack(top);
    return;
  }

  TensorIndex* out = Index::getConsumerIndex(
      ir_utils::asTV(top->out()), scope_utils::getLoops(active_scope_expr));
  Val* in1 = top->in1();
  Val* in2 = top->in2();
  Val* in3 = top->in3();

  if (ir_utils::isTV(in1))
    in1 = Index::getProducerIndex(
        ir_utils::asTV(in1),
        ir_utils::asTV(top->out()),
        scope_utils::getLoops(active_scope_expr));

  if (ir_utils::isTV(in2))
    in2 = Index::getProducerIndex(
        ir_utils::asTV(in2),
        ir_utils::asTV(top->out()),
        scope_utils::getLoops(active_scope_expr));

  if (ir_utils::isTV(in3))
    in3 = Index::getProducerIndex(
        ir_utils::asTV(in3),
        ir_utils::asTV(top->out()),
        scope_utils::getLoops(active_scope_expr));

  pushBack(new TernaryOp(top->getTernaryOpType(), out, in1, in2, in3));
}

void IndexLowering::handle(ReductionOp* rop) {
  TORCH_INTERNAL_ASSERT(
      ir_utils::isTVOp(rop),
      "Cannot have a reduction operation on something other than a tensor view, but received ",
      rop);
  auto loops = scope_utils::getLoops(active_scope_expr);

  bool is_private_reduce =
      std::none_of(loops.begin(), loops.end(), [](ForLoop* fl) {
        return fl->iter_domain()->isThread() &&
            fl->iter_domain()->isReduction();
      });

  TensorIndex* out = Index::getConsumerIndex(ir_utils::asTV(rop->out()), loops);

  Val* in = rop->in();
  if (ir_utils::isTV(in))
    in = Index::getProducerIndex(
        ir_utils::asTV(in),
        ir_utils::asTV(rop->out()),
        scope_utils::getLoops(active_scope_expr));

  if (!is_private_reduce) {
    pushBack(new ReductionOp(rop->getReductionOpType(), rop->init(), out, in));
  } else {
    pushBack(new BinaryOp(rop->getReductionOpType(), out, out, in));
  }
}

void IndexLowering::handle(BroadcastOp* bop) {
  TORCH_INTERNAL_ASSERT(
      ir_utils::isTVOp(bop),
      "Cannot have a broadcast operation on something other than a tensor view, but received ",
      bop);

  TensorIndex* out = Index::getConsumerIndex(
      ir_utils::asTV(bop->out()), scope_utils::getLoops(active_scope_expr));
  Val* in = bop->in();
  if (ir_utils::isTV(in))
    in = Index::getProducerIndex(
        ir_utils::asTV(in),
        ir_utils::asTV(bop->out()),
        scope_utils::getLoops(active_scope_expr));
  pushBack(new BroadcastOp(out, in));
}

void IndexLowering::generate(const std::vector<Expr*>& exprs) {
  // Run through loop nests and further lower the expressions
  for (auto* expr : exprs)
    OptInDispatch::handle(expr);
}

} // namespace fuser
} // namespace jit
} // namespace torch
