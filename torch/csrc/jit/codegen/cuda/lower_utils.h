#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h>

#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>

#include <bitset>
#include <map>

// Provides utilities for dealing with nested ForLoop and IfThenElse scopes

namespace torch {
namespace jit {
namespace fuser {

class ThreadPredicateMap;

namespace scope_utils {

// Grab the ForLoop starting from scope working out
std::vector<kir::ForLoop*> getLoops(Expr* scope);

// Track how far our for loop scope is
unsigned int computeForDepth(Expr* scope);

// Push back an expr to scope
void pushBack(Expr* scope, Expr* expr);

// Insert expr in scope before ref
void insertBefore(Expr* scope, Expr* ref, Expr* expr);

// Return the parent of the active scope
Expr* getParent(Expr* scope);

// Open a new inner most for loop
kir::ForLoop* openFor(Expr* scope, IterDomain*);

// Close the inner most for loop
Expr* closeScope(Expr* scope);

// Clear all expressions from the scope
Expr* clearScope(Expr* scope);

// Provide a new for loop matching the one provided, sets parent_scope as
// parent_scope, but does not insert into parent scope.
kir::ForLoop* cloneLoopNest(kir::ForLoop* to_clone, Expr* parent_scope);

// Run through a scope and replace expressions inside with replacement_map
void replaceExprsInScope(
    Expr* scope,
    std::unordered_map<Expr*, Expr*> replacement_map);

Expr* firstInnerMostScope(Expr* scope);

} // namespace scope_utils

namespace ir_utils {

// Somtimes we want to temporarily view a tensorview with another tensordomain.
// This isn't a permanent transformation, but in indexing we want to index
// producers with a consumer set of indices, so we need to view the producer
// transformed like consumer while we index. This will set the tv with td for
// the life of this context guard.
class TVDomainGuard {
 public:
  TensorView* tv_;
  TensorDomain* prev_domain;

  explicit TVDomainGuard(TensorView* _tv, TensorDomain* td);

  ~TVDomainGuard();
};

// Return inputs of provided IterDomains that are IterDomains
std::vector<IterDomain*> iterDomainInputsOf(const std::vector<IterDomain*>&);

// Return inputs of provided IterDomains that are IterDomains, order as the
// second provided vector.
std::vector<IterDomain*> iterDomainInputsOfOrderedAs(
    const std::vector<IterDomain*>& of,
    const std::vector<IterDomain*>& order);

std::vector<Val*> indices(std::vector<kir::ForLoop*>);

std::vector<IterDomain*> iterDomains(std::vector<kir::ForLoop*>);

bool isTV(const Val* const);

bool isTVOp(const Expr*);

TensorView* getTVOutput(const Expr*);

bool isScalarOp(const Expr*);

void ASSERT_EXPR(Statement*);

bool isScope(const Expr*);

Expr* asExpr(Statement*);

// TODO: Remove in favor of ->as<TensorView>()
TensorView* asTV(Val*);

// TODO: Remove in favor of ->as<ForLoop>()
kir::ForLoop* asForLoop(Statement*);

// TODO: Remove in favor of ->as<TensorView>()
const TensorView* asConstTV(const Val*);

bool isUnrolledFor(const Expr*);

// Represents mapping to bool from BIDx, BIDy, BIDz, TIDx, TIDy and TIDz.
class ParallelTypeBitmap {
 public:
  static constexpr int num_p_type = 6;
  ParallelTypeBitmap() = default;
  bool get(ParallelType pt) const;
  bool set(ParallelType pt, bool);
  ParallelTypeBitmap operator&=(const ParallelTypeBitmap& other);
  ParallelTypeBitmap operator|=(const ParallelTypeBitmap& other);
  ParallelTypeBitmap operator^=(const ParallelTypeBitmap& other);
  ParallelTypeBitmap operator~() const;
  bool none() const;
  bool any() const;
  bool all() const;
  bool operator[](size_t pos) const;
  std::map<ParallelType, bool> getMap() const;

 private:
  ParallelTypeBitmap(const std::bitset<num_p_type>& bs) : bitset_(bs) {}
  std::bitset<num_p_type> bitset_;
  const static std::unordered_map<ParallelType, int> pt_to_offset_;
  const static std::unordered_map<int, ParallelType> offset_to_pt_;
};

ParallelTypeBitmap operator&(
    const ParallelTypeBitmap& lhs,
    const ParallelTypeBitmap& rhs);

ParallelTypeBitmap operator|(
    const ParallelTypeBitmap& lhs,
    const ParallelTypeBitmap& rhs);

ParallelTypeBitmap operator^(
    const ParallelTypeBitmap& lhs,
    const ParallelTypeBitmap& rhs);

// Returns a ParallelTypeBitmap representing which domain needs
// blockBroadcast.
// Even when a domain is broadcast and parallelized, it does not need
// blockBroadcast unless it is predicated.
ParallelTypeBitmap getParallelBroadcastDomains(
    const Val* bop_out,
    const ThreadPredicateMap& preds);

} // namespace ir_utils

namespace loop_utils {

// I wanted to make the tv's in these util functions constant, but that started
// a long const-ness project going into TensorView (making functions const
// there) then into lower_loops where we sort exprs.
// TODO: We should fix this when we have some time.

// If we're initializing a reduction buffer, we won't have the reduction
// loops. If we're actually performing the reduction, we will. Grab a reduction
// root dimension in tv and see if it maps to any loop, returns if it maps or
// not.
bool loopsHasReductions(
    TensorView* tv,
    const std::vector<kir::ForLoop*>& loops,
    const std::unordered_map<IterDomain*, IterDomain*>& ca_id_map =
        std::unordered_map<IterDomain*, IterDomain*>());

// Go through the iter domains in loops, and (in order) grab the ones that match
// tv->getComputeAtAxis(...), map from the IterDomain in
// tv->getComputeAtAxis(...) to its corresponding loop. If there are reduction
// axes in the loops, assume we need to match reduction axes, otherwise assume
// we ignore them. Provided map (if needed) maps iteration domains in
// tv->getComputeAtAxes(...) to those in loops[...]->iter_domain. This map is
// typically needed if tv is a producer.
std::unordered_map<IterDomain*, kir::ForLoop*> computeAtToLoopMap(
    TensorView* tv,
    const std::vector<kir::ForLoop*>& loops,
    const std::unordered_map<IterDomain*, IterDomain*>& ca_id_map =
        std::unordered_map<IterDomain*, IterDomain*>());

// Return inverse map of computeAtToLoopMap.
std::unordered_map<kir::ForLoop*, IterDomain*> loopToComputeAtMap(
    TensorView* tv,
    const std::vector<kir::ForLoop*>& loops,
    const std::unordered_map<IterDomain*, IterDomain*>& ca_id_map =
        std::unordered_map<IterDomain*, IterDomain*>());

// Given producer with producer->domain() replayed like consumer, map
// producer->getComputeAtAxis(...) to consumer->getComputeAtAxis(...) for all
// IterDomain's in producer->domain().
std::unordered_map<IterDomain*, IterDomain*> mapIdPtoC(
    TensorView* producer,
    TensorView* consumer);

// Run through loops which should have all indices needed to index into tv.
// Validate these indices, and potentially modify them for use with the tv. Take
// into consideration allocation point and memory type. tv->domain() is expected
// to match the loop structure in loops. Provided map (if needed) maps iteration
// domains in tv->getComputeAtAxes(...) to those in loops[...]->iter_domain.
// This map is typically needed if tv is a producer. If for_predicates is true
// don't filter indices based on memory type.
std::vector<Val*> getIndicesForTV(
    TensorView* tv,
    const std::vector<kir::ForLoop*>& loops,
    const std::unordered_map<IterDomain*, IterDomain*>& p2c_root_map,
    bool for_predicates = false,
    const std::unordered_map<IterDomain*, IterDomain*>& ca_id_map =
        std::unordered_map<IterDomain*, IterDomain*>());

// Similar to get indices for tv, but within unroll loop, will return
// loop->extent() - 1. Only callable on a consumer.
std::vector<Val*> getUnrollPredIndicesForTV(
    TensorView* consumer_tv,
    const std::vector<kir::ForLoop*>& loops);

// Run through loops, match which ones are used for indexing tv, return the
// extents of these loops. Take into consideration allocation point and memory
// type. tv->domain() is expected to match the loop structure in loops. Provided
// map (if needed) maps iteration domains in tv->getComputeAtAxes(...) to those
// in loops[...]->iter_domain. This map is typically needed if tv is a producer.
// tv must be set to shared or local memory.
std::vector<Val*> getRangesForTV(
    TensorView* tv,
    const std::vector<kir::ForLoop*>& loops,
    const std::unordered_map<IterDomain*, IterDomain*>& ca_id_map =
        std::unordered_map<IterDomain*, IterDomain*>());

// Figure out which loop the allocation needs to be in. Returns nullptr if
// outside the first loop in loops. Also find out which index in tv the
// first dimension that needs to be allocated is. Meaning we need to allocate
// that local axis and above.
std::pair<kir::ForLoop*, int64_t> getAllocPoint(
    TensorView* tv,
    const std::vector<kir::ForLoop*>& loops);

// Go through exprs mapping root domains from producer to consumer. Provides a
// ground truth for how root domains map through our expressions
std::unordered_map<IterDomain*, IterDomain*> p2cRootMap(
    std::vector<Expr*> exprs);

// Given a root IterationDomain and a p2c_root_map find the root IterationDomain
// furthest down in the sorted expr list it maps to
IterDomain* getTermIDInMap(
    IterDomain* root_id,
    std::unordered_map<IterDomain*, IterDomain*> p2c_root_map);

} // namespace loop_utils

} // namespace fuser
} // namespace jit
} // namespace torch
