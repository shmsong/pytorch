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

// Custom dispatch for Expr, want to find out of it's a TV op
void UnrollPass::handle(Expr* expr) {
  OptOutDispatch::handle(expr);
}

namespace {

kir::Bool* getPredicate(
    TensorView* tv,
    std::vector<Val*> inds_,
    kir::Bool* thread_pred) {
  TORCH_INTERNAL_ASSERT(
      inds_.size() == tv->nDims() ||
      inds_.size() == tv->domain()->noReductions().size());

  // Do we need to adjust for reduction axes?
  bool reductions = inds_.size() != tv->nDims();

  std::vector<Val*> inds;
  if (reductions) {
    for (size_t ind_i = 0, tv_i = 0; tv_i < tv->nDims();) {
      if (tv->axis(tv_i++)->isReduction()) {
        inds.push_back(new Int(0));
      } else {
        TORCH_INTERNAL_ASSERT(
            ind_i < inds_.size(), "Ran out of indices to generate predicate.");
        inds.push_back(inds_[ind_i++]);
      }
    }
  } else {
    inds = inds_;
  }

  if (tv->nDims() > inds.size()) {
    for (decltype(tv->nDims()) i{0}; i < tv->nDims(); i++) {
      if (tv->axis(i)->isReduction())
        inds.insert(inds.begin() + i, new Int(0));
    }
  }

  auto all_preds = PredicateCompute::computePredicates(
      new kir::TensorIndex(tv, IndexCompute::get(tv->domain(), inds)));

  if (thread_pred != nullptr) {
    all_preds.push_back(thread_pred);
  }

  std::vector<kir::Bool*> preds;

  for (auto pred : all_preds)
    if (!(pred->isConst()) || !(pred->isConst() && pred->value().value()))
      preds.push_back(pred);

  if (preds.size() == 0)
    return new kir::Bool(true);

  Val* cond = preds[0];

  for (decltype(preds.size()) i{1}; i < preds.size(); i++) {
    cond = kir::andExpr(cond, preds[i]);
  }

  TORCH_INTERNAL_ASSERT(
      cond->getValType().value() == ValType::KirScalar &&
          cond->getDataType().value() == DataType::Bool,
      "Error computing predicate, should be returning a Bool, but returning ",
      cond->getDataType().value());

  return cond->as<kir::Bool>();
}

} // namespace

// This function is one huge mess that should be refactored.
// It handles the unrolling and predicate generation
void UnrollPass::handle(kir::ForLoop* fl) {
  // Setup for loop scoping
  for_loops.push_back(fl);
  bool prev_unroll = within_unroll;
  within_unroll = ir_utils::isUnrolledFor(fl) || within_unroll;

  for (auto expr : fl->body().exprs()) {
    OptOutDispatch::handle(expr);
  }

  TensorView* out = nullptr;
  bool has_global = false;
  for (Expr* expr : fl->body().exprs())
    if (ir_utils::isTVOp(expr)) {
      // Predicate determining op for unroll
      out = ir_utils::asTV(expr->output(0));
      has_global = has_global || out->getMemoryType() == MemoryType::Global;
      for (auto inp : expr->inputs())
        if (ir_utils::isTV(inp))
          has_global = has_global ||
              ir_utils::asTV(inp)->getMemoryType() == MemoryType::Global;
    }

  bool has_TV_op = out != nullptr;

  if (within_unroll && has_TV_op) {
    // Setup unrolled loop information:

    // Indices used to detect when we can unroll a loop safely
    // For loops outside the unroll, it's just the index, for loops inside
    // the unroll, if it's a thread it's the thread index, otherwise it's
    // the size-1
    std::vector<Val*> unroll_pred_inds;
    auto it = for_loops.begin();
    while (it != for_loops.end()) {
      if (ir_utils::isUnrolledFor(*it))
        break;
      unroll_pred_inds.push_back((*it)->index());
      it++;
    }

    TORCH_INTERNAL_ASSERT(
        it != for_loops.end(),
        "Error unrolling loops, expected an unrolled loop but wasn't found.");

    // This is the outer most loop that needs to be unrolled
    kir::ForLoop* first_unroll = *it;

    // Indicies inside the unroll
    while (it != for_loops.end()) {
      IterDomain* id = (*it)->iter_domain();
      if (id->isThread())
        unroll_pred_inds.push_back((*it)->index());
      else
        unroll_pred_inds.push_back(sub(id->extent(), new Int(1)));
      it++;
    }

    // Make predicates for the unrolling, and the epilogue
    kir::Bool* unroll_predicate =
        getPredicate(out, unroll_pred_inds, getThreadPredicate(out));
    // Make the IfThenElse controlling the unrolling
    kir::IfThenElse* unroll_ite = new kir::IfThenElse(
        unroll_predicate, {}, {}, first_unroll->parentScope());

    // Get the loop nest for the unrolled path
    kir::ForLoop* unrolled_loop =
        scope_utils::cloneLoopNest(first_unroll, unroll_ite);
    unroll_ite->body().push_back(unrolled_loop);

    // Loop nest for inlined path
    kir::ForLoop* inlined_loop =
        scope_utils::cloneLoopNest(first_unroll, unroll_ite);
    unroll_ite->elseBody().push_back(inlined_loop);

    // Inner most inlined loop
    Expr* inner_most_inlined_loop =
        scope_utils::firstInnerMostScope(inlined_loop);

    loop_replacement_map.insert({first_unroll, unroll_ite});

    for (auto expr : fl->body().exprs()) {
      if (!ir_utils::isTVOp(expr))
        continue;

      // Setup the expressions that need predicates around them.
      auto inline_predicate = getPredicate(
          out, ir_utils::indices(for_loops), getThreadPredicate(out));

      kir::IfThenElse* inline_ite = new kir::IfThenElse(
          inline_predicate, {expr}, {}, inner_most_inlined_loop);
      std::unordered_map<Expr*, Expr*> inline_replacement_map;
      inline_replacement_map.emplace(std::pair<Expr*, Expr*>(expr, inline_ite));
      scope_utils::replaceExprsInScope(
          inner_most_inlined_loop, inline_replacement_map);

    } // for expr
  } else { //  if(!within_unroll)
    // modify in place, so grab a copy of exprs first.
    const std::vector<Expr*> exprs = fl->body().exprs();

    for (auto expr : exprs) {
      if (!ir_utils::isTVOp(expr))
        continue;

      TensorView* out = ir_utils::asTV(ir_utils::asExpr(expr)->outputs()[0]);

      auto pred = getPredicate(
          out, ir_utils::indices(for_loops), getThreadPredicate(out));

      // If we need a predicate, put expr inside an if then else
      if (!(pred->isConst()) || !(pred->isConst() && pred->value().value())) {
        kir::IfThenElse* inline_ite =
            new kir::IfThenElse(pred, {expr}, {}, for_loops.back());
        for_loops.back()->body().insert_before(expr, inline_ite);
        for_loops.back()->body().erase(expr);
      }
    }
  } // else (if(!within_unroll))

  for_loops.pop_back();
  within_unroll = prev_unroll;
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
    const ThreadPredicateMap& thread_predicates) {
  FusionGuard fg(fusion);
  UnrollPass up(fusion, exprs, thread_predicates);
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
