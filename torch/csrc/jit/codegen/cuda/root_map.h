#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h>

#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/ir_iostream.h>
#include <torch/csrc/jit/codegen/cuda/iter_visitor.h>
#include <unordered_map>
#include <vector>

namespace torch {
namespace jit {
namespace fuser {

namespace{
 class DisjointSet;
 class ReduceDependency;
}

class TORCH_CUDA_API PositionalRootMap : public IterVisitor {
 protected:
  using Dependence = std::set<IterDomain*>;
  using DependenceMap = std::unordered_map<IterDomain*,Dependence*>;
  
  Fusion* fusion_;
  DisjointSet djSet;
  DependenceMap RDepMap;

  void handle(UnaryOp*) override;
  void handle(BinaryOp*) override;
  void handle(TernaryOp*) override;
  void handle(ReductionOp*) override;

  void handle(TensorView*) override;

  void mapRDependency(TensorView* from, TensorView* to);

 public:
  PositionalRootMap(Fusion* fusion) : fusion_(fusion), RDepMap(fusion){
   traverseAllPaths(fusion_, false);
  }
  
  bool isEqual(Iterdomain*,Iterdomain*);

  unordered_map<IterDomain*,IterDomain*> mapAtoB(
        const std::vector<IterDomain*> &, 
        const std::vector<IterDomain*> &);
}


} //namespace torch
} //namespace jit
} //namespace fuser