#include <torch/csrc/jit/codegen/cuda/index_compute.h>
#include <c10/util/Exception.h>
#include <torch/csrc/jit/codegen/cuda/arith.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/ir_iostream.h>
#include <torch/csrc/jit/codegen/cuda/transform_replay.h>

namespace torch {
namespace jit {
namespace fuser {

namespace {

// A merge is contiguous if:
//   Inputs of outer are to the left in the root domain of the inputs of RHS.
//   All inputs are contiguous in the root domain:
//     - All marked as contiguous
//     - Only gaps between inputs are broadcast or reductoin dims
//   There are no split transformations performed on outer or inner
//   All transformations on outer or inner are contiguous merges
// If this criteria holds, then we can index the input root domains of this
// merge with the indexing provided to the output of the merge in the backward
// index pass

class ContigIDs : public OptInDispatch {
 private:
  using OptInDispatch::handle;

  // Mark if ids are result of contigous merges
  std::unordered_set<IterDomain*> contig_ids;
  const std::vector<IterDomain*>& root_domain_;
  const std::vector<bool>& root_contiguity_;
  std::unordered_map<IterDomain*, bool> is_contig_root;

  ContigIDs() = delete;

  ContigIDs(
      const std::vector<IterDomain*>& ids,
      const std::vector<IterDomain*>& _root_domain,
      const std::vector<bool>& _root_contiguity)
      : root_domain_(_root_domain), root_contiguity_(_root_contiguity) {
    if (ids.empty()) {
      return;
    }

    TORCH_INTERNAL_ASSERT(
        root_domain_.size() == root_contiguity_.size(),
        "Arguments don't match ",
        root_domain_.size(),
        " != ",
        root_contiguity_.size());

    for (size_t i = 0; i < root_domain_.size(); i++) {
      if (root_contiguity_[i]) {
        contig_ids.emplace(root_domain_[i]);
      }
      is_contig_root[root_domain_[i]] = root_contiguity_[i];
    }

    auto exprs = ExprSort::getExprs(ids[0]->fusion(), {ids.begin(), ids.end()});

    for (auto expr : exprs) {
      handle(expr);
    }
  }

  bool inRoot(const std::vector<IterDomain*>& ids) {
    return std::all_of(ids.begin(), ids.end(), [this](IterDomain* id) {
      return is_contig_root.find(id) != is_contig_root.end();
    });
  }

  bool isContig(IterDomain* id) {
    return contig_ids.find(id) != contig_ids.end();
  }

  // Split outputs are not conitguous, don't need to do anything.
  void handle(Split*) override {}

  void handle(Merge* merge) override {
    // If either input is non-contiguous so is output.
    auto inner = merge->inner();
    auto outer = merge->outer();
    if (!isContig(inner) || !isContig(outer)) {
      return;
    }

    // Grab inputs, make sure they're in root domain, check if they're
    // contiguous.

    auto lhs_inputs =
        ir_utils::iterDomainInputsOfOrderedAs({outer}, root_domain_);
    auto rhs_inputs =
        ir_utils::iterDomainInputsOfOrderedAs({inner}, root_domain_);

    TORCH_INTERNAL_ASSERT(
        inRoot(lhs_inputs) && inRoot(rhs_inputs),
        "Found an invalid merge operation, inputs of its arguments are not in the root domain.");

    std::deque<IterDomain*> ordered_inputs(
        lhs_inputs.begin(), lhs_inputs.end());
    ordered_inputs.insert(
        ordered_inputs.end(), rhs_inputs.begin(), rhs_inputs.end());

    // If any root input is not contig, output is not contig
    if (!(std::all_of(
            ordered_inputs.begin(),
            ordered_inputs.end(),
            [this](IterDomain* id) { return is_contig_root.at(id); }))) {
      return;
    }

    std::deque<IterDomain*> root_copy(root_domain_.begin(), root_domain_.end());

    // Forward to first matching argument
    while (!root_copy.empty() && !ordered_inputs.empty()) {
      if (root_copy.front() != ordered_inputs.front()) {
        root_copy.pop_front();
      } else {
        break;
      }
    }

    // Forward through all matching arguments
    while (!root_copy.empty() && !ordered_inputs.empty()) {
      if (root_copy.front() == ordered_inputs.front()) {
        root_copy.pop_front();
        ordered_inputs.pop_front();
        // We probably should be able to make access contiguous through
        // reduction domains, however, for now it's causing issues in predicate
        // generation. See test: ReductionSchedulerMultiDimNonFastest
        //  } else if (
        //     root_copy.front()->isReduction() ||
        //     root_copy.front()->isBroadcast()) {
        //   root_copy.pop_front();
      } else {
        break;
      }
    }

    // If we matched all inputs, the output is contiguous
    if (ordered_inputs.empty()) {
      contig_ids.emplace(merge->out());
    }
  }

 public:
  // Check through thie history of ids whose inputs map to root_domain with
  // contiguity root_contiguity. Return unordered_set of all merges that are
  // contiguous.
  static std::unordered_set<IterDomain*> find(
      const std::vector<IterDomain*>& ids,
      const std::vector<IterDomain*>& root_domain,
      const std::vector<bool>& root_contiguity) {
    ContigIDs finder(ids, root_domain, root_contiguity);
    return finder.contig_ids;
  }
};

// Take a set of ranges on a domain and backward proipagate them to figure out
// the extent of the root domain axes.
class RangeCompute : public BackwardVisitor {
 private:
  using BackwardVisitor::handle;

  void handle(Split* split) override {
    auto in_id = split->in();
    auto outer_id = split->outer();
    auto inner_id = split->inner();

    auto outer_it = range_map_.find(outer_id);
    auto inner_it = range_map_.find(inner_id);
    if (outer_it == range_map_.end() || inner_it == range_map_.end())
      return;

    auto outer_range = outer_it->second;
    auto inner_range = inner_it->second;

    Val* extent = nullptr;

    bool has_zero = outer_range->isZeroInt() || inner_range->isZeroInt();

    bool both_zero = outer_range->isZeroInt() && inner_range->isZeroInt();

    bool zero_merged_in = has_zero ||
        zero_merged_id.find(outer_id) != zero_merged_id.end() ||
        zero_merged_id.find(inner_id) != zero_merged_id.end();

    if (zero_merged_in) {
      zero_merged_id.emplace(in_id);
    }

    if (both_zero) {
      range_map_[in_id] = new Int(0);
    } else if (has_zero) {
      range_map_[in_id] = outer_range->isZeroInt() ? inner_range : outer_range;
    } else if (zero_merged_in) {
      range_map_[in_id] = mul(outer_range, inner_range);
    } else {
      range_map_[in_id] = in_id->extent();
    }
  }

  void handle(Merge* merge) override {
    auto out_id = merge->out();
    auto outer_id = merge->outer();
    auto inner_id = merge->inner();

    auto out_it = range_map_.find(out_id);
    if (out_it == range_map_.end())
      return;

    auto out_range = out_it->second;

    if (contig_ids.find(out_id) != contig_ids.end()) {
      auto input_ids =
          ir_utils::iterDomainInputsOfOrderedAs({out_id}, td_->rootDomain());

      // Shouldn't hit this, but don't want to segfault if somehow we do.
      TORCH_INTERNAL_ASSERT(!input_ids.empty());

      for (auto root_id : input_ids) {
        range_map_[root_id] = new Int(0);
      }

      range_map_[*(input_ids.end() - 1)] = out_range;
      return;
    }

    // If there was a 0 merged in here due to a split just move the extent to
    // the right
    if (zero_merged_id.find(out_id) != zero_merged_id.end()) {
      range_map_[outer_id] = new Int(0);
      range_map_[inner_id] = out_range;
    } else {
      range_map_[outer_id] = merge->outer()->extent();
      range_map_[inner_id] = merge->inner()->extent();
    }
  }

  void handle(Expr* e) override {
    switch (e->getExprType().value()) {
      case (ExprType::Split):
      case (ExprType::Merge):
        break;
      default:
        TORCH_INTERNAL_ASSERT(
            false, "Invalid expr type found in transform traversal.");
    }
    BackwardVisitor::handle(e);
  }

  RangeCompute(
      const TensorDomain* _td,
      const std::vector<Val*>& ranges,
      std::vector<bool> _root_contiguity)
      : td_(_td) {
    contig_ids =
        ContigIDs::find(td_->domain(), td_->rootDomain(), _root_contiguity);

    if (td_->nDims() == 0 || ranges.empty()) {
      ranges_.push_back(new Int(0));
      return;
    }

    // TODO: We will always provide reduction ranges, even though they may be 0

    // We may or may not have ranges associated with reductions.
    const bool exclude_reduction = td_->nDims() > ranges.size();

    TORCH_INTERNAL_ASSERT(
        td_->noReductions().size() == ranges.size() ||
            td_->nDims() == ranges.size(),
        "For IndexCompute the number of axes should match the number of dimensions in the TensorDomain.");

    {
      size_t i = 0;
      for (auto id : td_->domain()) {
        if (exclude_reduction && id->isReduction())
          continue;
        range_map_[id] = ranges[i++];
      }
    }

    const std::vector<Val*> domain_vals(
        td_->domain().begin(), td_->domain().end());

    // Run the split/merge operations backwards. This will modify the range_map_
    // so it can be used to index the root TensorDomain. Each entry in the root
    // TensorDomain should have an entry in range_map_ We might not want to run
    // these ranges at the root of the domain, but actually at the rfactor
    // root. Fortunately we can run them all the way back, but grab the ranges
    // from the map at the rfactor IterDomains.
    traverseFrom(ranges[0]->fusion(), domain_vals, false);

    // TODO: Don't exclude reduction axes
    auto root_dom =
        td_->hasRFactor() ? td_->rfactorDomain() : td_->rootDomain();
    for (auto id : root_dom) {
      if (exclude_reduction && id->isReduction()) {
        continue;
      } else if (id->getIterType() == IterType::BroadcastWithStride) {
        // TODO: Why not do this for any broadcast dim? Would they be non-zero?
        ranges_.push_back(new Int(1));
      } else {
        auto it = range_map_.find(id);
        TORCH_INTERNAL_ASSERT(
            it != range_map_.end(),
            "Error during index compute, missed computing a value.");
        ranges_.push_back(it->second);
      }
    }
  }

  // Tensor domain we're mapping back to root
  const TensorDomain* td_;
  // Map we update as we propagate backward
  std::unordered_map<IterDomain*, Val*> range_map_;
  // Starting with input ranges, returning as root ranges
  std::vector<Val*> ranges_;
  // IDs that are result of contiguous merges
  std::unordered_set<IterDomain*> contig_ids;
  // IDs that have a 0 merged back into them, we can't map these dims back to
  // the original id->extent.
  std::unordered_set<IterDomain*> zero_merged_id;

 public:
  static std::vector<Val*> get(
      const TensorDomain* _td,
      const std::vector<Val*>& _ranges,
      const std::vector<bool>& _root_contiguity) {
    RangeCompute rc(_td, _ranges, _root_contiguity);
    return rc.ranges_;
  }
};

} // namespace

void IndexCompute::handle(Split* split) {
  auto in_id = split->in();
  auto outer_id = split->outer();
  auto inner_id = split->inner();

  auto outer_it = index_map_.find(outer_id);
  auto inner_it = index_map_.find(inner_id);
  if (outer_it == index_map_.end() || inner_it == index_map_.end())
    return;

  auto outer_ind = outer_it->second;
  auto inner_ind = inner_it->second;

  bool outer_zero = outer_ind->isZeroInt();
  bool inner_zero = inner_ind->isZeroInt();

  if (outer_zero && inner_zero) {
    index_map_[in_id] = new Int(0);
  } else if (outer_zero) {
    index_map_[in_id] = inner_ind;
  } else if (inner_zero) {
    index_map_[in_id] = outer_ind;
  } else {
    index_map_[in_id] = add(mul(outer_ind, split->factor()), inner_ind);
  }
}

void IndexCompute::handle(Merge* merge) {
  auto out_id = merge->out();
  auto outer_id = merge->outer();
  auto inner_id = merge->inner();

  auto out_it = index_map_.find(out_id);
  if (out_it == index_map_.end())
    return;

  auto out_ind = out_it->second;

  if (out_ind->isZeroInt()) {
    index_map_[outer_id] = new Int(0);
    index_map_[inner_id] = new Int(0);
    return;
  }

  if (contig_ids.find(out_id) != contig_ids.end()) {
    auto input_ids =
        ir_utils::iterDomainInputsOfOrderedAs({out_id}, td_->rootDomain());

    // Shouldn't hit this, but don't want to segfault if somehow we do.
    TORCH_INTERNAL_ASSERT(!input_ids.empty());

    for (auto root_id : input_ids) {
      index_map_[root_id] = new Int(0);
    }

    index_map_[*(input_ids.end() - 1)] = out_ind;
    return;
  }

  Val* I = inner_id->extent();
  Val* outer_ind = div(out_ind, I);
  Val* inner_ind = mod(out_ind, I);

  index_map_[outer_id] = outer_ind;
  index_map_[inner_id] = inner_ind;
}

void IndexCompute::handle(Expr* e) {
  switch (e->getExprType().value()) {
    case (ExprType::Split):
    case (ExprType::Merge):
      break;
    default:
      TORCH_INTERNAL_ASSERT(
          false, "Invalid expr type found in transform traversal.");
  }
  BackwardVisitor::handle(e);
}

IndexCompute::IndexCompute(
    const TensorDomain* _td,
    const std::vector<Val*>& indices,
    std::vector<bool> root_contiguity,
    bool ignore_rfactor)
    : td_(_td) {
  contig_ids =
      ContigIDs::find(td_->domain(), td_->rootDomain(), root_contiguity);
  if (td_->nDims() == 0 || indices.empty()) {
    indices_.push_back(new Int(0));
    return;
  }

  // TODO: We will always provide reduction indices, even though they may be 0

  // We may or may not have indices associated with reductions.
  const bool exclude_reduction = td_->nDims() > indices.size();

  TORCH_INTERNAL_ASSERT(
      td_->noReductions().size() == indices.size() ||
          td_->nDims() == indices.size(),
      "For IndexCompute the number of axes should match the number of dimensions in the TensorDomain.");

  {
    size_t i = 0;
    for (auto id : td_->domain()) {
      if (exclude_reduction && id->isReduction())
        continue;
      index_map_[id] = indices[i++];
    }
  }

  const std::vector<Val*> domain_vals(
      td_->domain().begin(), td_->domain().end());

  // Run the split/merge operations backwards. This will modify the index_map_
  // so it can be used to index the root TensorDomain. Each entry in the root
  // TensorDomain should have an entry in index_map_ We might not want to run
  // these indices at the root of the domain, but actually at the rfactor root.
  // Fortunately we can run them all the way back, but grab the indices from the
  // map at the rfactor IterDomains.
  traverseFrom(indices[0]->fusion(), domain_vals, false);

  // TODO: Don't exclude reduction axes
  auto root_dom = td_->hasRFactor() && !ignore_rfactor ? td_->rfactorDomain()
                                                       : td_->rootDomain();
  for (auto id : root_dom) {
    if (exclude_reduction && id->isReduction()) {
      continue;
    } else if (id->getIterType() == IterType::BroadcastWithStride) {
      // TODO: Why not do this for any broadcast dim? Would they be non-zero?
      indices_.push_back(new Int(0));
    } else {
      auto it = index_map_.find(id);
      TORCH_INTERNAL_ASSERT(
          it != index_map_.end(),
          "Error during index compute, missed computing a value.");
      indices_.push_back(it->second);
    }
  }
}

std::vector<Val*> IndexCompute::get(
    const TensorDomain* td,
    const std::vector<Val*>& _indices,
    const std::vector<bool>& _root_contiguity,
    bool ignore_rfactor) {
  IndexCompute ic(td, _indices, _root_contiguity, ignore_rfactor);
  return ic.indices_;
}

std::vector<bool> IndexCompute::contiguityAnd(
    const std::vector<bool>& contig1,
    const std::vector<bool>& contig2) {
  TORCH_INTERNAL_ASSERT(
      contig1.size() == contig2.size(),
      "Called contiguityAnd with mismatched vectors.");

  std::vector<bool> contig_result;
  std::transform(
      contig1.begin(),
      contig1.end(),
      contig2.begin(),
      std::back_inserter(contig_result),
      std::logical_and<>());
  return contig_result;
}

// TODO: use new mapping functions
// This mapping might need to go through rfactor, unclear
std::vector<bool> IndexCompute::contiguityPasC(
    TensorDomain* producer,
    TensorDomain* consumer) {
  const std::vector<bool>& producer_contiguity = producer->contiguity();
  std::vector<bool> as_consumer_contiguity;

  auto c_root = consumer->rootDomain();
  auto p_root = producer->rootDomain();

  size_t p_ind = 0;
  size_t c_ind = 0;
  while (p_ind < p_root.size()) {
    if (p_root[p_ind]->isReduction()) {
      p_ind++;
    } else if (
        c_root[c_ind]->isBroadcast() &&
        p_root[p_ind]->getIterType() != c_root[c_ind]->getIterType()) {
      c_ind++;
      as_consumer_contiguity.push_back(false);
    } else {
      as_consumer_contiguity.push_back(producer_contiguity[p_ind]);
      c_ind++;
      p_ind++;
    }
  }

  while (c_ind < c_root.size()) {
    as_consumer_contiguity.push_back(false);
    c_ind++;
  }

  return as_consumer_contiguity;
}

kir::TensorIndex* Index::getGlobalProducerIndex(
    TensorView* producer_tv,
    TensorView* consumer_tv,
    const std::vector<kir::ForLoop*>& loops,
    const std::unordered_map<IterDomain*, IterDomain*>& p2c_root_map) {
  // producer_tv->domain() is not replayed as the loop strucutre we were
  // provided, so replay it to match consumer_tv which is.
  auto producerAsC = TransformReplay::replayPasC(
                         producer_tv->domain(), consumer_tv->domain(), -1)
                         .first;

  // Set producer_tv with the domain replayed as consumer to grab the right
  // indices. The guard will reset the domain when this scope ends.
  ir_utils::TVDomainGuard domain_guard(producer_tv, producerAsC);
  auto indices = loop_utils::getIndicesForTV(
      producer_tv,
      loops,
      p2c_root_map,
      false,
      loop_utils::mapIdPtoC(producer_tv, consumer_tv));

  std::vector<Val*> root_indices = IndexCompute::get(
      producerAsC, indices, producer_tv->domain()->contiguity());

  auto root_dom = producer_tv->domain()->hasRFactor()
      ? producer_tv->domain()->rfactorDomain()
      : producer_tv->getRootDomain();

  TORCH_INTERNAL_ASSERT(
      root_indices.size() == root_dom.size(),
      "Dimensionality error in code generator while computing indexing.");

  bool inner_most_dim_contig =
      producer_tv->getRootDomain()[producer_tv->getRootDomain().size() - 1]
              ->getIterType() == IterType::Iteration &&
      producer_tv->domain()
          ->contiguity()[producer_tv->getRootDomain().size() - 1];

  int64_t stride_i = 0;
  std::vector<Val*> strided_inds;
  for (size_t i = 0; i < root_dom.size(); i++) {
    if (root_dom[i]->isReduction() ||
        root_dom[i]->getIterType() == IterType::BroadcastWithoutStride) {
      continue;
    } else if (i == root_dom.size() - 1 && inner_most_dim_contig) {
      strided_inds.push_back(root_indices[i]);
    } else if (root_indices[i]->isZeroInt()) {
      stride_i++;
    } else {
      std::stringstream ss;
      ss << "T" << producer_tv->name() << ".stride[" << stride_i++ << "]";
      strided_inds.push_back(
          mul(root_indices[i], new NamedScalar(ss.str(), DataType::Int)));
    }
  }

  if (strided_inds.size() == 0)
    strided_inds.push_back(new Int(0));

  return new kir::TensorIndex(producer_tv, strided_inds);
}

// Producer index for either shared or local memory
kir::TensorIndex* Index::getProducerIndex_impl(
    TensorView* producer_tv,
    TensorView* consumer_tv,
    const std::vector<kir::ForLoop*>& loops,
    const std::unordered_map<IterDomain*, IterDomain*>& p2c_root_map) {
  // producer_tv->domain() is not replayed as the loop strucutre we were
  // provided, so replay it to match consumer_tv which is.
  auto producerAsC = TransformReplay::replayPasC(
                         producer_tv->domain(), consumer_tv->domain(), -1)
                         .first;

  // Set producer_tv with the domain replayed as consumer to grab the right
  // indices. The guard will reset the domain when this scope ends.
  ir_utils::TVDomainGuard domain_guard(producer_tv, producerAsC);

  auto domain_indices = loop_utils::getIndicesForTV(
      producer_tv,
      loops,
      p2c_root_map,
      false,
      loop_utils::mapIdPtoC(producer_tv, consumer_tv));

  auto domain_ranges = loop_utils::getRangesForTV(
      producer_tv, loops, loop_utils::mapIdPtoC(producer_tv, consumer_tv));

  auto root_dom = producer_tv->domain()->hasRFactor()
      ? producer_tv->domain()->rfactorDomain()
      : producer_tv->getRootDomain();

  // TODO: Remove contiguity entry from IndexCompute::get
  std::vector<Val*> root_indices = IndexCompute::get(
      producerAsC, domain_indices, producer_tv->domain()->contiguity());

  TORCH_INTERNAL_ASSERT(
      root_indices.size() == root_dom.size(),
      "Dimensionality error in code generator while computing indexing.");

  std::vector<Val*> root_ranges = RangeCompute::get(
      producerAsC, domain_ranges, producer_tv->domain()->contiguity());

  TORCH_INTERNAL_ASSERT(
      root_ranges.size() == root_dom.size(),
      "Dimensionality error in code generator while computing indexing.");

  std::vector<Val*> strided_inds;
  for (size_t i = 0; i < root_dom.size(); i++) {
    if (root_dom[i]->isReduction() ||
        root_dom[i]->getIterType() == IterType::BroadcastWithoutStride) {
      continue;
    } else if (root_indices[i]->isZeroInt() && root_ranges[i]->isZeroInt()) {
      continue;
    } else if (root_indices[i]->isZeroInt()) {
      continue;
    } else {
      Val* stride = nullptr;
      for (size_t j = i + 1; j < root_ranges.size(); j++) {
        if (!root_ranges[j]->isZeroInt() && !root_dom[j]->isBroadcast() &&
            !root_dom[j]->isReduction()) {
          if (stride == nullptr) {
            stride = root_ranges[j];
          } else {
            stride = mul(stride, root_ranges[j]);
          }
        }
      }

      if (stride != nullptr) {
        strided_inds.push_back(mul(root_indices[i], stride));
      } else {
        strided_inds.push_back(root_indices[i]);
      }
    }
  }

  if (strided_inds.size() == 0)
    strided_inds.push_back(new Int(0));

  return new kir::TensorIndex(producer_tv, strided_inds);
}

kir::TensorIndex* Index::getGlobalConsumerIndex(
    TensorView* consumer_tv,
    const std::vector<kir::ForLoop*>& loops,
    const std::unordered_map<IterDomain*, IterDomain*>& p2c_root_map) {
  auto indices = loop_utils::getIndicesForTV(consumer_tv, loops, p2c_root_map);

  std::vector<Val*> computed_inds = IndexCompute::get(
      consumer_tv->domain(), indices, consumer_tv->domain()->contiguity());

  auto root_dom = consumer_tv->domain()->hasRFactor()
      ? consumer_tv->domain()->rfactorDomain()
      : consumer_tv->getRootDomain();

  TORCH_INTERNAL_ASSERT(
      computed_inds.size() == root_dom.size(),
      "Dimensionality error in code generator while computing indexing.");

  bool inner_most_dim_contig =
      consumer_tv->getRootDomain()[consumer_tv->getRootDomain().size() - 1]
              ->getIterType() == IterType::Iteration &&
      consumer_tv->domain()
          ->contiguity()[consumer_tv->getRootDomain().size() - 1];

  int64_t stride_i = 0;
  std::vector<Val*> strided_inds;
  for (size_t i = 0; i < root_dom.size(); i++) {
    if (root_dom[i]->isReduction() ||
        root_dom[i]->getIterType() == IterType::BroadcastWithoutStride) {
      continue;
    } else if (i == root_dom.size() - 1 && inner_most_dim_contig) {
      strided_inds.push_back(computed_inds[i]);
    } else if (computed_inds[i]->isZeroInt()) {
      stride_i++;
    } else {
      std::stringstream ss;
      ss << "T" << consumer_tv->name() << ".stride[" << stride_i++ << "]";
      strided_inds.push_back(
          mul(computed_inds[i], new NamedScalar(ss.str(), DataType::Int)));
    }
  }

  if (strided_inds.size() == 0)
    strided_inds.push_back(new Int(0));

  return new kir::TensorIndex(consumer_tv, strided_inds);
}

// Consumer index for either shared or local memory
kir::TensorIndex* Index::getConsumerIndex_impl(
    TensorView* consumer_tv,
    const std::vector<kir::ForLoop*>& active_loops,
    const std::unordered_map<IterDomain*, IterDomain*>& p2c_root_map) {
  auto domain_indices =
      loop_utils::getIndicesForTV(consumer_tv, active_loops, p2c_root_map);
  auto domain_ranges = loop_utils::getRangesForTV(consumer_tv, active_loops);

  auto root_dom = consumer_tv->domain()->hasRFactor()
      ? consumer_tv->domain()->rfactorDomain()
      : consumer_tv->getRootDomain();

  std::vector<Val*> root_indices = IndexCompute::get(
      consumer_tv->domain(),
      domain_indices,
      consumer_tv->domain()->contiguity());

  TORCH_INTERNAL_ASSERT(
      root_indices.size() == root_dom.size(),
      "Dimensionality error in code generator while computing indexing.");

  std::vector<Val*> root_ranges = RangeCompute::get(
      consumer_tv->domain(),
      domain_ranges,
      consumer_tv->domain()->contiguity());

  TORCH_INTERNAL_ASSERT(
      root_ranges.size() == root_dom.size(),
      "Dimensionality error in code generator while computing indexing.");

  std::vector<Val*> strided_inds;
  for (size_t i = 0; i < root_dom.size(); i++) {
    if (root_dom[i]->isReduction() ||
        root_dom[i]->getIterType() == IterType::BroadcastWithoutStride) {
      continue;
    } else if (root_indices[i]->isZeroInt() && root_ranges[i]->isZeroInt()) {
      continue;
    } else if (root_indices[i]->isZeroInt()) {
      continue;
    } else {
      Val* stride = nullptr;
      for (size_t j = i + 1; j < root_ranges.size(); j++) {
        if (!root_ranges[j]->isZeroInt() && !root_dom[j]->isBroadcast() &&
            !root_dom[j]->isReduction()) {
          if (stride == nullptr) {
            stride = root_ranges[j];
          } else {
            stride = mul(stride, root_ranges[j]);
          }
        }
      }
      if (stride != nullptr) {
        strided_inds.push_back(mul(root_indices[i], stride));
      } else {
        strided_inds.push_back(root_indices[i]);
      }
    }
  }

  if (strided_inds.size() == 0)
    strided_inds.push_back(new Int(0));

  return new kir::TensorIndex(consumer_tv, strided_inds);
}

// Producer is the inputs of an expression
kir::TensorIndex* Index::getProducerIndex(
    TensorView* producer,
    TensorView* consumer,
    const std::vector<kir::ForLoop*>& loops,
    const std::unordered_map<IterDomain*, IterDomain*>& p2c_root_map) {
  if (producer->domain()->noReductions().size() == 0) {
    return new kir::TensorIndex(producer, {});
  }

  if (producer->getMemoryType() == MemoryType::Global)
    return getGlobalProducerIndex(producer, consumer, loops, p2c_root_map);
  return getProducerIndex_impl(producer, consumer, loops, p2c_root_map);
}

// Consumer is the output of an expression
kir::TensorIndex* Index::getConsumerIndex(
    TensorView* consumer,
    const std::vector<kir::ForLoop*>& loops,
    const std::unordered_map<IterDomain*, IterDomain*>& p2c_root_map) {
  if (consumer->domain()->noReductions().size() == 0) {
    return new kir::TensorIndex(consumer, {});
  }

  if (consumer->getMemoryType() == MemoryType::Global)
    return getGlobalConsumerIndex(consumer, loops, p2c_root_map);
  return getConsumerIndex_impl(consumer, loops, p2c_root_map);
}

} // namespace fuser
} // namespace jit
} // namespace torch
