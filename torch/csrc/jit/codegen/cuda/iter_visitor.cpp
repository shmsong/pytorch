#include <torch/csrc/jit/codegen/cuda/iter_visitor.h>
#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/ir_iostream.h>
#include <torch/csrc/jit/codegen/cuda/type.h>

namespace torch {
namespace jit {
namespace fuser {

/* ITER VISITOR */

std::vector<Statement*> IterVisitor::next(Statement* statement) {
  if (statement->isVal())
    return next(statement->as<Val>());
  else if (statement->isExpr())
    return next(statement->as<Expr>());
  else
    TORCH_INTERNAL_ASSERT(
        false, "IterVisitor could not detect type in next_dispatch.");
}

std::vector<Statement*> IterVisitor::next(Val* v) {
  FusionGuard::getCurFusion()->assertInFusion(v, "Cannot traverse val, ");
  if (FusionGuard::getCurFusion()->origin(v) != nullptr)
    return {FusionGuard::getCurFusion()->origin(v)};
  return {};
}

std::vector<Statement*> IterVisitor::next(Expr* expr) {
  FusionGuard::getCurFusion()->assertInFusion(expr, "Cannot traverse expr, ");
  std::vector<Statement*> next_stmts{expr->inputs().begin(),
                                     expr->inputs().end()};
  return next_stmts;
}

namespace {

// Remove any stmt in stmts that is in visited
void remove_visited(
    std::vector<Statement*>& stmts,
    const std::unordered_set<Statement*>& visited) {
  std::deque<std::vector<Statement*>::iterator> to_erase;
  for (auto it = stmts.begin(); it != stmts.end(); it++) {
    if (visited.find(*it) != visited.end())
      to_erase.push_back(it);
  }

  while (!to_erase.empty()) {
    stmts.erase(to_erase.back());
    to_erase.pop_back();
  }
}

} // namespace

void IterVisitor::traverseFrom(
    Fusion* fusion,
    const std::vector<Val*>& from,
    bool traverseAllPaths) {
  FusionGuard fg(fusion);
  std::unordered_set<Statement*> visited;
  stmt_stack.clear();
  stmt_stack.emplace_back(from.rbegin(), from.rend());
  // true when returning to a node after vistiting all its input
  // nodes. Nodes are only visited when this is true.
  bool all_inputs_visited = false;

  while (!stmt_stack.empty()) {
    auto& current_inputs = stmt_stack.back();
    // When current_inputs is empty, all the input nodes have been
    // visited. Return to the output node by popping the stack. Record
    // all inputs are visited.
    if (current_inputs.empty()) {
      stmt_stack.pop_back();
      all_inputs_visited = true;
      continue;
    }
    const auto& stmt = current_inputs.back();
    // Visit stmt when all_inputs_visited is true.
    if (all_inputs_visited) {
      // Mark visited
      visited.insert(stmt);
      // Handle
      handle(stmt);
      current_inputs.pop_back();
      all_inputs_visited = false;
    } else {
      // Visit input nodes.
      auto next_stmts = next(stmt);
      if (!traverseAllPaths) {
        remove_visited(next_stmts, visited);
      }
      if (next_stmts.empty()) {
        all_inputs_visited = true;
      } else {
        stmt_stack.emplace_back(next_stmts.rbegin(), next_stmts.rend());
        all_inputs_visited = false;
      }
    }
  }
}

void IterVisitor::traverse_(
    Fusion* fusion,
    bool from_outputs_only,
    bool traverse_all_paths) {
  FusionGuard fg(fusion);

  if (from_outputs_only) {
    auto term_val_outs = fusion->getTerminatingOutputs();
    if (!term_val_outs.empty())
      traverseFrom(fusion, term_val_outs, traverse_all_paths);
    return;
  }

  std::vector<Val*> leaves;
  // Search for Vals with no uses (output edges)
  for (Val* val : fusion->deterministic_vals())
    if (!fusion->used(val))
      leaves.push_back(val);

  if (!leaves.empty())
    traverseFrom(fusion, leaves, traverse_all_paths);
}

void IterVisitor::traverse(Fusion* fusion, bool from_outputs_only) {
  traverse_(fusion, from_outputs_only, false);
}

void IterVisitor::traverseAllPaths(
    Fusion* fusion,
    bool from_outputs_only) {
  traverse_(fusion, from_outputs_only, true);
}

namespace {

// Expr sort will take a fusion and return a topologically sorted list of
// expressions.
class Inputs : public IterVisitor {
 private:
  std::unordered_set<Val*> inputs;

  void handle(Val* val) override {
    if (val->getOrigin() == nullptr)
      inputs.emplace(val);
  }

 public:
  static std::unordered_set<Val*> getInputs(const std::vector<Val*>& of) {
    if (of.empty())
      return std::unordered_set<Val*>();
    Inputs inps;
    inps.traverseFrom(of[0]->fusion(), of);
    return inps.inputs;
  }
};

} // namespace

std::unordered_set<Val*> IterVisitor::getInputsTo(
    const std::vector<Val*>& vals) {
  return Inputs::getInputs(vals);
}

namespace {

class AllVals : public IterVisitor {
 private:
  std::unordered_set<Val*> vals;

  void handle(Val* val) final {
    vals.emplace(val);
  }

 public:
  static std::unordered_set<Val*> get(
      Fusion* fusion,
      const std::vector<Val*>& from) {
    AllVals av;
    av.traverseFrom(fusion, from, false);
    return av.vals;
  }
};

} // namespace

/* BACKWARDS VISITOR */

std::vector<Statement*> BackwardVisitor::next(Statement* stmt) {
  if (stmt->isVal())
    return next(stmt->as<Val>());
  else if (stmt->isExpr())
    return next(stmt->as<Expr>());
  else
    TORCH_INTERNAL_ASSERT(
        false, "BackwardVisitor could not detect type in next_dispatch.");
}

std::vector<Statement*> BackwardVisitor::next(Expr* expr) {
  return std::vector<Statement*>(
      expr->outputs().begin(), expr->outputs().end());
}

std::vector<Statement*> BackwardVisitor::next(Val* val) {
  // Going to sort based on relative topological position
  std::map<size_t, Statement*> exprs;

  for (auto expr : FusionGuard::getCurFusion()->unordered_uses(val))
    // Make sure it's an expr we can traverse
    if (traversal_exprs_.find(expr) != traversal_exprs_.end())
      exprs[traversal_exprs_[expr]] = expr;

  std::vector<Statement*> next_stmts(exprs.size());
  std::transform(
      exprs.begin(),
      exprs.end(),
      next_stmts.begin(),
      [](std::pair<size_t, Statement*> pair) { return pair.second; });

  return next_stmts;
}

void BackwardVisitor::traverseFrom(
    Fusion* fusion,
    const std::vector<Val*>& from,
    bool traverseAllPaths) {
  FusionGuard fg(fusion);

  // Reset members
  stmt_stack_.clear();
  traversal_exprs_.clear();

  if (from.empty())
    return;

  auto vals = AllVals::get(fusion, from);

  auto exprs = ExprSort::getExprs(fusion, from);

  {
    size_t pos = 0;
    for (auto expr : exprs)
      traversal_exprs_[expr] = pos++;
  }

  // All stmts we've called handle on
  std::unordered_set<Statement*> visited_stmts_;

  for (auto traversal_pair : traversal_exprs_)
    for (auto out : traversal_pair.first->outputs())
      TORCH_INTERNAL_ASSERT(
          vals.find(out) != vals.end(),
          "Invalid backward traversal found. Some output paths were not provided.");

  auto inputs = InputsOf::getInputsTo(from);
  stmt_stack_.emplace_back(inputs.begin(), inputs.end());

  // The rest is basically copy-pasted from IterVitor:
  while (!stmt_stack_.empty()) {
    auto next_stmts = next(stmt_stack_.back().back());

    // Remove statements we already visited if we're not traversing all paths
    if (!traverseAllPaths)
      remove_visited(next_stmts, visited_stmts_);

    // Traverse down until we get to a leaf
    while (!next_stmts.empty()) {
      stmt_stack_.emplace_back(next_stmts.rbegin(), next_stmts.rend());
      next_stmts = next(stmt_stack_.back().back());
      // Remove statements we already visited if we're not traversing all paths
      if (!traverseAllPaths)
        remove_visited(next_stmts, visited_stmts_);
    }

    // Traverse back up
    // Mark visited
    visited_stmts_.emplace(stmt_stack_.back().back());
    // Handle
    handle(stmt_stack_.back().back());
    // Remove
    stmt_stack_.back().pop_back();

    while (!stmt_stack_.empty() && stmt_stack_.back().empty()) {
      stmt_stack_.pop_back();
      if (!stmt_stack_.empty()) {
        // Mark visited
        visited_stmts_.emplace(stmt_stack_.back().back());
        // Handle
        handle(stmt_stack_.back().back());
        // Remove
        stmt_stack_.back().pop_back();
      }
    }
  }
}

/* DEPENDENCY CHECKING */

namespace {

// Looks for and returns all values in between dependencies and vals, including
// them.
struct Dependencies : public IterVisitor {
  std::unordered_set<Val*> dependencies_;
  std::unordered_set<Val*> vals;

  std::vector<Statement*> next(Val* v) override {
    if (dependencies_.find(v) != dependencies_.end())
      return std::vector<Statement*>();
    return IterVisitor::next(v);
  }

  void handle(Val* val) override {
    vals.emplace(val);
  }

  Dependencies(
      std::unordered_set<Val*> _dependencies,
      const std::vector<Val*>& of)
      : dependencies_(std::move(_dependencies)) {
    traverseFrom(of[0]->fusion(), of, false);
  };

 public:
  static std::unordered_set<Val*> getAllVals(
      const std::unordered_set<Val*>& dependencies,
      const std::vector<Val*>& of) {
    if (of.empty())
      return std::unordered_set<Val*>();

    Dependencies deps(dependencies, of);
    return deps.vals;
  }
};

// Looks for and returns
class DependencyChains : public IterVisitor {
 public:
  std::deque<std::deque<Val*>> dep_chains;
  bool is_dependency = false;
  std::unordered_set<Val*> dependencies_;

  void handle(Val* val) override {
    if (dependencies_.find(val) != dependencies_.end()) {
      is_dependency = true;
      std::deque<Val*> deps;
      for (auto stack : stmt_stack) {
        if (stack.back()->isVal())
          deps.push_back(stack.back()->as<Val>());
      }
      // Order as dependency -> of
      dep_chains.emplace_back(deps.rbegin(), deps.rend());
    }
  }

  DependencyChains(Val* _dependency, Val* _of, bool all_chains_ = false)
      : dependencies_({_dependency}) {
    traverseFrom(_of->fusion(), {_of}, all_chains_);
  }

  DependencyChains(Val* _dependency, bool all_chains_ = false)
      : dependencies_({_dependency}) {
    if (all_chains_)
      traverseAllPaths(_dependency->fusion(), false);
    else
      traverse(_dependency->fusion(), false);
  }

  DependencyChains(
      std::unordered_set<Val*> _dependencies,
      bool all_chains_ = false)
      : dependencies_(std::move(_dependencies)) {
    if (dependencies_.empty())
      return;

    if (all_chains_)
      traverseAllPaths((*dependencies_.begin())->fusion(), false);
    else
      traverse((*dependencies_.begin())->fusion(), false);
  }

  static std::deque<Val*> getDependencyChain(Val* dependency, Val* of) {
    DependencyChains dp(dependency, of, false);
    if (dp.dep_chains.empty())
      return std::deque<Val*>();
    return dp.dep_chains[0];
  }

  // I don't think this is actually hooked up, but leaving for now.
  static std::deque<std::deque<Val*>> getDependencyChains(
      Val* dependency,
      Val* of) {
    DependencyChains dp(dependency, of, true);
    if (dp.dep_chains.empty())
      return std::deque<std::deque<Val*>>();
    return dp.dep_chains;
  }

  static std::deque<std::deque<Val*>> getAllUseChains(Val* dependency) {
    DependencyChains dp(dependency, true);
    if (dp.dep_chains.empty())
      return std::deque<std::deque<Val*>>();
    return dp.dep_chains;
  }

  static std::deque<std::deque<Val*>> getAllUseChains(
      const std::unordered_set<Val*>& dependencies) {
    DependencyChains dp(dependencies, true);
    if (dp.dep_chains.empty())
      return std::deque<std::deque<Val*>>();
    return dp.dep_chains;
  }
};

} // namespace

bool DependencyCheck::isDependencyOf(Val* dependency, Val* of) {
  return !DependencyChains::getDependencyChain(dependency, of).empty();
}

std::deque<Val*> DependencyCheck::getSingleDependencyChain(
    Val* dependency,
    Val* of) {
  return DependencyChains::getDependencyChain(dependency, of);
}

std::deque<std::deque<Val*>> DependencyCheck::getAllDependencyChains(
    Val* dependency,
    Val* of) {
  return DependencyChains::getDependencyChains(dependency, of);
}

std::deque<std::deque<Val*>> DependencyCheck::getAllUseChains(Val* producer) {
  return DependencyChains::getAllUseChains(producer);
}

std::unordered_set<Val*> DependencyCheck::getAllValsBetween(
    const std::unordered_set<Val*>& dependencies,
    const std::vector<Val*>& of) {
  return Dependencies::getAllVals(dependencies, of);
}

} // namespace fuser
} // namespace jit
} // namespace torch
