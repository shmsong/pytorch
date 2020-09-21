#include <torch/csrc/jit/codegen/cuda/ir_utils.h>
#include <torch/csrc/jit/codegen/cuda/root_map.h>
#include <tuple>

namespace torch {
namespace jit {
namespace fuser {

namespace{
 class DisjointSet{
  protected:
   using ConstraintType=std::unordered_set<int>;
   using ConstraintSet=std::unordered_map<int,ConstraintType>;

   std::vector<int> SetMap,Weights;
   std::unordered_map<IterDomain*,int> DomSetMap;
   ConstraintSet constraints;
   int count;

  public:
    DisjointSet():count(0);
    void join(IterDomain*,IterDomain*);
    void constrain(const std::vector<IterDomain*>&);
    bool isEqual(IterDomain*,IterDomain*);

    int fixedPoint(IterDomain* I){
     if(!DomSetMap.count(I)) createPoint(I);
     return fixedPoint(DomSetMap[I]); 
    }

  private:
    int fixedPoint(int e){
      TORCH_INTERNAL_ASSERT(SetMap.size()<e);
      while(SetMap[e]!=e){
        SetMap[e] = SetMap[SetMap[e]];
        e=SetMap[e];
      }
      return e;
    }

    void createPoint(IterDomain* I){
     DomSetMap[I] = count;
     SetMap.push_back(count++);
    }
    
    inline bool isConstrained(int a, int by){
     if(!constraints.count(by))  return false;
     return !constraints[by].count(a);
    }

    bool canJoin(IterDomain* I0, IterDomain* I1){
     if(!DomSetMap.count(I0)||!DomSetMap.count(I1)) return true;
     int i0=fixedPoint(I0), i1=fixedPoint(I1);
     return !isConstrained(i0,i1) && !isConstrained(i1,i0);
    }

 }// class DisjoinSet

 void DisjointSet::join(IterDomain* I0, IterDomain* I1){
  if(!canJoin(I0,I1)) return;
  int i0=fixedPoint(I0),i1=fixedPoint(I1);
  int new_parent, new_child;

  if(Weights[i0]<Weights[i1])
   std::tie(new_parent,new_child) = std::make_pair(i0,i1);
  else
   std::tie(new_parent,new_child) = std::make_pair(i1,i0);

  Weights[new_parent]+=Weights[new_child];
  SetMap[new_child]=new_parent;
 }

 bool DisjointSet::isEqual(IterDomain* I0, IterDomain* I1){
   if(I0==I1) return true;
   if(!DomSetMap.count(I0) || !DomSetMap.count(I1)) return false;
   return fixedPoint(DomSetMap[I0])==fixedPoint(DomSetMap[I1]);
 }

 bool DisjointSet::constrain(const std::vector<IterDomain*>& idVec){
  std::vector<int> iVec;
  std::transform(
   idVec.begin(),idVec.end(),
   std::back_inserter(iVec),
   fixedPoint);
  
  for(int a : iVec)
   for(int b : iVec)
    if(a!=b) constraints[a].insert(b);
 }
} //namespace

namespace{
 inline void joinSet( 
  DisjointSet& djSet,
  const std::vector<IterDomain*>& I0,
  const std::vector<IterDomain*>& I1){

  TORCH_INTERNAL_ASSERT(I0.size()==I1.size(), 
             "PositionalRootMap::pointWiseOpMap found an op with non-matching IO");

  for(int it=0;it<I0.size();it++)
   djSet.join(I0[it],I1[it]);
 }

 void pointWiseOpMap(DisjointSet& djSet, Expr* e){
  // all ops handled in this pass are single output only
  TensorView* TVo  = e->output(0);
  
  // constrain all the inputs before adding any mapping
  for(auto i* : ir_utils::filterByType<TensorView>(e->inputs()))
    djSet.constrain(i->getMaybeRfactorDomain())
  
  //add mapping to output
  for(auto i* : ir_utils::filterByType<TensorView>(e->inputs()))
    joinSet(djSet,
            TVo->getRootDomain(),
            //input can have reductions that we don't want this operator to map
            TensorDomain::noReductions(i->getMaybeRFactorDomain())); 
 } 
} //namespace

void PositionalRootMap::handle(UnaryOp* e){
 pointWiseOpMap(djSet,e->asExpr());
}

void PositionalRootMap::handle(BinaryOp* e){
 pointWiseOpMap(djSet,e->asExpr());
}

void PositionalRootMap::handle(TernaryOp* e){
 pointWiseOpMap(djSet,e->asExpr());
}

void PositionalRootMap::handle(ReductionOp* r){
 pointWiseOpMap(djSet,e->asExpr());
}

//broadcast Op:
//  no joinSet action needed for broadcast op since broadcast op doesn't create
//  any new tensor root map-able to the input


// both root domain and rfactor domain will be mapped in this case because
// it is not entirely clear how the optimization passes will try to map
// could save some mapping later on if any protocol can be assumed
void PositionalRootMap::mapRDependency(TensorView* from, TensorView* to){
 
 //lambda function adds all of to's axes to from's reduction axes if any
 auto mapIds = [&RDepMap](const std::vector<IterDomain*>& fromDomain,
                           const std::vector<IterDomain*>& toDomain){
  for(IterDomain* idfrom: fromDomain)
   if(idfrom->isReduction())
    for(IterDomain* idTo: toDomain)
     RDepMap[idfrom].insert(idTo);
  }; // lambda definition

  mapIds(from->getRootDomain(),to->getRootDomain());
  if(to->domain()->hasRFactor())
   mapIds(from->getRootDomain(),to->getRFactorDomain());
  
  if(from->domain()->hasRFactor()){
    mapIds(from->getRFactorDomain(),to->getRootDomain())
    if(to->domain()->hasRFactor())
     mapIds(from->getRFactorDomain(),to->getRFactorDomain());
  }
}

// largely copied from dependency check, try to get all the deps in one pass
void PositionalRootMap::handle(TensorView* tv) override {
 for (auto stack : stmt_stack) {
  auto stmt = stack.back();
  if (stmt->isVal() &&
      stmt->asVal()->getValType().value()==ValType::TensorView) {
        TensorView* deptv = stmt->asVal()->as<TensorView>;
        mapRDependency(deptv,tv);       
  }
 }
}

} //namespace torch
} //namespace jit
} //namespace fuser