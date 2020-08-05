#pragma once

#include <torch/csrc/jit/codegen/cuda/iter_visitor.h>

#include <unordered_map>
#include <unordered_set>
#include <vector>

/*
 * Index compute takes in a list of indices typically generated from the
 * surrounding for loop nest. The number of indicies are intended to match the
 * number of dimensions of the incomming TensorView which may have less or more
 * dimensions than its root due to split/merge operations.
 * Split/merge operations are then replayed backwards produce resulting
 * indices (based on input indices) that match the root dimension.
 *
 * For example with GLOBAL tensor:
 * TV[I, J]
 * TV[Io, Ii{4}, J] = TV.split(I, factor=4)
 * ALLOC: NONE
 * INDEX: indexCompute {i, j, k} -> {i * 4 + j, k}
 * FLATTENED_INDEX: {i * 4 + j, k} -> {i * 4 + j * J + k}
 * PREDICATE: {i * 4 + j, k} -> i * 4 + j < I
 *
 *
 * For example with SHARED tensor:
 *
 * global_TV[I, J]
 * global_TV[Io, Ii{4}, J] = global_TV.split(I, factor=4)
 * smem_TV.compute_at(global_TV, 1)
 * global_TV.parallelize(1, threadIDx.x)
 *
 * ALLOC: alloc(smem_TV, 4 x J)
 * INDEX: indexCompute(smem_TV, {threadIdx.x, k}) -> {threadIdx.x, k}
 * FLATTENED_INDEX: {threadIdx.x * 4 + j, k} -> {threadIdx.x * 4 + j * J + k}
 * PREDICATE: {threadIdx.x * 4 + j, k} -> threadIdx.x * 4 + j < I // Same as if
 * global
 *
 *
 * For example with LOCAL tensor:
 * global_TV[I, J, K]
 * global_TV[Io, Ii{4}, J] = global_TV.split(I, factor=4)
 * reg_TV.compute_at(global_TV, 1)
 * global_TV.parallelize(1, threadIDx.x)
 * global_TV{i, j, k, l} -> { i * 4 + j, k, l }
 * global_TV{ i * 4 + j, k, l } -> { i * 4 + j * J * K  +  k * K  +  l}
 *
 * ALLOC: alloc(reg_TV, J x K)
 * INDEX: {k, l} -> {k, l}
 * FLATTENED_INDEX: {k, l} -> {k * J + l}
 * PREDICATE: i * 4 + j < I && k < J && l < K ->  // Same as if global
 *
 * These indices can then be flattened later based on strides.
 */

namespace torch {
namespace jit {
namespace fuser {

class IndexCompute : public BackwardVisitor {
 private:
  using BackwardVisitor::handle;
  void handle(Split*) override;
  void handle(Merge*) override;
  void handle(Expr*) override;

  // Otherwise warning on runBackward as it hides an overloaded virtual
  // using TransformIter::runBackward;
  IndexCompute(
      const TensorDomain* _td,
      const std::vector<Val*>& indices,
      std::vector<bool> _root_contiguity,
      bool ignore_rfactor);

  // Tensor domain we're mapping back to root
  const TensorDomain* td_;
  // Map we update as we propagate backward, containing all IDs in the
  // propagation
  std::unordered_map<IterDomain*, Val*> index_map_;
  // Indices we start with, returning this after we propagate back as root
  // indices
  std::vector<Val*> indices_;
  // IDs that are a result of contiguous merges
  std::unordered_set<IterDomain*> contig_ids;

 public:
  static std::vector<Val*> get(
      const TensorDomain* _td,
      const std::vector<Val*>& _indices,
      const std::vector<bool>& _root_contiguity,
      bool ignore_rfactor = false);

  // Map producer contiguity information to consumer, if entries don't match
  // mark as false
  static std::vector<bool> contiguityPasC(
      TensorDomain* producer,
      TensorDomain* consumer);

  static std::vector<bool> contiguityAnd(
      const std::vector<bool>& contig1,
      const std::vector<bool>& contig2);
};

// Simple interface for IndexCompute
// If getComputeAtAxis and more generally TensorView const model is fixed, we
// can make the below tensorviews const.
class Index {
 private:
  // Producer indexing if it's in shared or local memory
  static kir::TensorIndex* getProducerIndex_impl(
      TensorView* producer,
      TensorView* consumer,
      const std::vector<kir::ForLoop*>& loops);

  // Consumer indexing if it's in shared or local memory
  static kir::TensorIndex* getConsumerIndex_impl(
      TensorView* consumer,
      const std::vector<kir::ForLoop*>& loops);

  // Producer if it's in global memory
  static kir::TensorIndex* getGlobalProducerIndex(
      TensorView* producer,
      TensorView* consumer,
      const std::vector<kir::ForLoop*>& loops);

  // Consumer indexing if it's in global memory
  static kir::TensorIndex* getGlobalConsumerIndex(
      TensorView* consumer,
      const std::vector<kir::ForLoop*>& loops);

 public:
  // Indexing functions
  // Consumer = Producer
  // i.e. T0 = T1... -> T0 is the consumer, T1 is the producer
  // Producer indexing dispatch
  static kir::TensorIndex* getProducerIndex(
      TensorView* producer,
      TensorView* consumer,
      const std::vector<kir::ForLoop*>& loops);

  // Consumer index dispatch
  static kir::TensorIndex* getConsumerIndex(
      TensorView* consumer,
      const std::vector<kir::ForLoop*>& loops);
};

} // namespace fuser
} // namespace jit
} // namespace torch
