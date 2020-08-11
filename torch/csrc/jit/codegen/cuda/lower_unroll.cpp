#include <torch/csrc/jit/codegen/cuda/arith.h>
#include <torch/csrc/jit/codegen/cuda/index_compute.h>
#include <torch/csrc/jit/codegen/cuda/ir_iostream.h>
#include <torch/csrc/jit/codegen/cuda/lower_utils.h>
#include <torch/csrc/jit/codegen/cuda/predicate_compute.h>

#include <torch/csrc/jit/codegen/cuda/lower_unroll.h>

namespace torch {
namespace jit {
namespace fuser {

kir::Bool* UnrollPass::getThreadPredicate(TensorView* tv) {
  // No thread predicate is needed predicate when tv is output of a
  // parallel broadcast expression.
  const auto origin = tv->getOrigin();
  if (origin != nullptr && origin->getExprType() == ExprType::BroadcastOp) {
    const auto out = origin->as<BroadcastOp>()->out();
    if (ir_utils::getParallelBroadcastDomains(out, thread_predicates_).any()) {
      return nullptr;
    }
  }

  return thread_predicates_.getExpr(tv);
}

// Custom dispatch for Expr, want to find out of it's a TV op.
void UnrollPass::handle(Expr* expr) {
  // If tv op, predciate it.
  if (ir_utils::isTVOp(expr)) {
    TensorView* out = ir_utils::asTV(ir_utils::asExpr(expr)->outputs()[0]);

    TORCH_INTERNAL_ASSERT(for_loops.size() != 0);

    auto pred = PredicateCompute::getInlinePredicate(
        expr, for_loops, getThreadPredicate(ir_utils::getTVOutput(expr)));

    // If we need a predicate, put expr inside an if then else
    if (!(pred->isConst()) || !(pred->isConst() && pred->value().value())) {
      kir::IfThenElse* inline_ite =
          new kir::IfThenElse(pred, {expr}, {}, for_loops.back());
      for_loops.back()->body().insert_before(expr, inline_ite);
      for_loops.back()->body().erase(expr);
    }

  } else {
    // If not tv op, dispatch it.
    OptOutDispatch::handle(expr);
  }
}

// We should factor our actual predicate generation from unrolling but insering
// IR nodes "unroll_pred" or "inline_pred", then generate those later.
void UnrollPass::handle(kir::ForLoop* fl) {
  // Setup for loop scoping
  bool is_unroll = ir_utils::isUnrolledFor(fl);
  // If we're not looking for an unroll loop, or didn't find one, process as
  // normal.
  if (!is_unroll || !look_for_unroll) {
    for_loops.push_back(fl);

    std::vector<Expr*> exprs_copy = fl->body().exprs();
    // Make copy of exprs because we replace them inplace in fl
    for (auto expr : exprs_copy) {
      handle(expr);
    }
    for_loops.pop_back();

    return;
  }

  auto unroll_pred = UnrollPredicate::get(for_loops, fl, p2c_root_map);

  kir::IfThenElse* unroll_ite =
      new kir::IfThenElse(unroll_pred, {}, {}, for_loops.back());

  // Get the loop nest for the unrolled path
  kir::ForLoop* unrolled_loop_nest = scope_utils::cloneLoopNest(fl, unroll_ite);

  unroll_ite->body().push_back(unrolled_loop_nest);

  // Loop nest for inlined path
  kir::ForLoop* inlined_loop = scope_utils::cloneLoopNest(fl, unroll_ite);

  // Add inline predicates for inlined loop nest
  look_for_unroll = false;
  handle(inlined_loop);
  look_for_unroll = true;

  unroll_ite->elseBody().push_back(inlined_loop);

  // Inner most inlined loop
  Expr* inner_most_inlined_loop =
      scope_utils::firstInnerMostScope(inlined_loop);

  loop_replacement_map.insert({fl, unroll_ite});
}

// Generate the loop nest structure and place it in lowered_exprs
void UnrollPass::computeMap() {
  FusionGuard fg(fusion_);

  // Run through loop nests and further lower the expressions
  for (auto* expr : incoming_exprs_) {
    OptOutDispatch::handle(expr);
  }
}

std::vector<Expr*> UnrollPass::runPass(
    Fusion* fusion,
    const std::vector<Expr*>& exprs,
    const std::unordered_set<Expr*>& init_exprs,
    const ThreadPredicateMap& thread_predicates) {
  FusionGuard fg(fusion);
  UnrollPass up(fusion, exprs, init_exprs, thread_predicates);
  up.computeMap();
  std::vector<Expr*> mutated_exprs;
  for (Expr* expr : exprs) {
    if (up.loop_replacement_map.find(expr) != up.loop_replacement_map.end()) {
      mutated_exprs.push_back(up.loop_replacement_map[expr]);
    } else {
      if (ir_utils::isScope(expr))
        scope_utils::replaceExprsInScope(expr, up.loop_replacement_map);
      mutated_exprs.push_back(expr);
    }
  }
  return mutated_exprs;
}

} // namespace fuser
} // namespace jit
} // namespace torch
