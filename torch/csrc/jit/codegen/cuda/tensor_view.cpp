#include <torch/csrc/jit/codegen/cuda/arith.h>
#include <torch/csrc/jit/codegen/cuda/compute_at.h>
#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/ir_cloner.h>
#include <torch/csrc/jit/codegen/cuda/ir_iostream.h>
// #include <torch/csrc/jit/codegen/cuda/iter_visitor.h>
#include <torch/csrc/jit/codegen/cuda/ir_interface_nodes.h>

// Cleanup
// #include <torch/csrc/jit/codegen/cuda/mutator.h>
#include <torch/csrc/jit/codegen/cuda/transform_iter.h>
#include <torch/csrc/jit/codegen/cuda/transform_replay.h>

namespace torch {
namespace jit {
namespace fuser {

namespace {
DataType aten_opt_type_map(const c10::optional<at::ScalarType>& scalar_type) {
  return scalar_type.has_value() ? aten_to_data_type(scalar_type.value())
                                 : DataType::Null;
}
} // namespace

TensorView::TensorView(TensorDomain* _domain, DataType dtype)
    : Val(ValType::TensorView, dtype), domain_(_domain) {}

TensorView::TensorView(const std::shared_ptr<c10::TensorType>& tensor_type)
    : Val(ValType::TensorView,
          aten_opt_type_map(tensor_type->scalarType()),
          false) {
  std::vector<IterDomain*> sizes;
  TORCH_CHECK(
      tensor_type->dim().has_value(), "Requires static rank for Tensor");
  for (decltype(tensor_type->dim().value()) i = 0;
       i < tensor_type->dim().value();
       i++) {
    sizes.push_back(new IterDomain(new Int(0), new Int()));
  }
  domain_ = new TensorDomain(sizes);

  statement_number_ = fusion_->registerVal(this);
}

TensorView::TensorView(const TensorView* src, IrCloner* ir_cloner)
    : Val(src, ir_cloner),
      domain_(ir_cloner->clone(src->domain_)),
      compute_at_view_(ir_cloner->clone(src->compute_at_view_)),
      relative_compute_at_axis_(src->relative_compute_at_axis_),
      this_compute_at_axis_(src->this_compute_at_axis_),
      memory_type_(src->memory_type_) {}

bool TensorView::hasReduction() const {
  return domain()->hasReduction();
}

bool TensorView::hasBlockReduction() const {
  return domain()->hasBlockReduction();
}

bool TensorView::hasGridReduction() const {
  return domain()->hasGridReduction();
}

bool TensorView::hasBroadcast() const {
  return domain()->hasBroadcast();
}

const std::vector<IterDomain*>& TensorView::getRootDomain() const {
  return domain()->rootDomain();
};

std::vector<IterDomain*>::size_type TensorView::nDims() const {
  return domain()->nDims();
}

IterDomain* TensorView::axis(int pos) const {
  TORCH_INTERNAL_ASSERT(
      nDims() > 0, "Tried to access an axis in a 0-dim TensorView");
  if (pos < 0)
    pos += domain()->nDims();
  TORCH_CHECK(
      pos >= 0 && (unsigned int)pos < domain()->nDims(),
      "Tried to access position ",
      pos,
      " in domain: ",
      domain());
  return domain()->axis(pos);
}

TensorView* TensorView::unsafeClone() const {
  TensorView* new_view = new TensorView(domain_, getDataType().value());
  new_view->compute_at_view_ = compute_at_view_;
  new_view->relative_compute_at_axis_ = relative_compute_at_axis_;
  new_view->this_compute_at_axis_ = this_compute_at_axis_;
  new_view->setMemoryType(memory_type_);
  new_view->statement_number_ = statementNumber();
  // This is problematic if we want to use names as identifiers
  new_view->name_ = name();
  return new_view;
}

void TensorView::setComputeAt(TensorView* computeAtView, int axis) {
  compute_at_view_ = computeAtView;
  relative_compute_at_axis_ = axis;
  setThisComputeAtAxis();

  TORCH_INTERNAL_ASSERT(
      getThisComputeAtAxis() >= 0 &&
          (unsigned int)getThisComputeAtAxis() <= nDims(),
      "Invalid computeAt on ",
      this,
      " tried to set to local axis ",
      getThisComputeAtAxis());

  TORCH_INTERNAL_ASSERT(
      std::none_of(
          domain()->domain().begin(),
          domain()->domain().begin() + getThisComputeAtAxis(),
          [](IterDomain* id) { return id->isReduction(); }),
      "Invalid computeAt, reduction domain inside computeAt axis.");
}

void TensorView::setComputeAt(
    TensorView* computeAtView,
    int thisPos,
    int relPos) {
  compute_at_view_ = computeAtView;
  relative_compute_at_axis_ = relPos;
  this_compute_at_axis_ = thisPos;
  TORCH_INTERNAL_ASSERT(
      this_compute_at_axis_ <= nDims(), "Manually set an invalid computeAt.");
}

void TensorView::copyDomain(const TensorDomain* td) {
  std::vector<IterDomain*> idv;
  for (decltype(td->nDims()) i = 0; i < td->nDims(); i++)
    idv.push_back(td->axis(i));
  setDomain(new TensorDomain(idv));
}

// Where in compute_at_view does axis(pos) match up?
int TensorView::getComputeAtRelPos(int pos) {
  if (!hasComputeAt())
    return pos;

  if (!compute_at_view_->hasBroadcast())
    return pos;

  size_t pos_cav = 0, pos_this = 0;
  while ((int)pos_this < pos) {
    TORCH_INTERNAL_ASSERT(
        pos_cav < compute_at_view_->nDims(),
        "Error computing relative position in computeAt.");
    if (compute_at_view_->axis(pos_cav)->isBroadcast() &&
        !(axis(pos_this)->isBroadcast())) {
      pos_cav++;
    } else {
      pos_cav++;
      pos_this++;
    }
  }

  return pos_cav;
}

void TensorView::setThisComputeAtAxis() {
  if (compute_at_view_ == nullptr) {
    relative_compute_at_axis_ = 0;
    this_compute_at_axis_ = 0;
    return;
  }

  // this[is{i1}, is{i2},] -> compute at compute_at_view[bS{i0}, iS{i1}, iS{i2}]
  // axis = 2 this compute at axis = 1

  // pos in compute at view
  size_t pos_cav = 0, pos_this = 0;
  while (pos_cav < relative_compute_at_axis_ && pos_this < nDims()) {
    if (compute_at_view_->axis(pos_cav)->isBroadcast() &&
        !(axis(pos_this)->isBroadcast())) {
      pos_cav++;
    } else {
      pos_cav++;
      pos_this++;
    }
  }

  TORCH_INTERNAL_ASSERT(
      pos_cav == relative_compute_at_axis_ ||
          (pos_cav < compute_at_view_->nDims() &&
           compute_at_view_->axis(pos_cav)->isBroadcast()),
      "Error seting up relative position between this and what we view into.");

  this_compute_at_axis_ = pos_this;
}

TensorView* TensorView::computeAt(TensorView* consumer, int axis) {
  // Make sure this and consumer are not the same tensor, that's illegal
  TORCH_CHECK(!sameAs(consumer), "Cannot call computeAt(this, ...)");

  // We support negative axes, so increment it by consumer->nDims() + 1 and make
  // sure the result is within consumer->nDims() + 1. being at consumer->nDims()
  // means producer will be computed inline with consumer, hence the +1.
  if (axis < 0)
    axis += int(consumer->nDims()) + 1;
  TORCH_CHECK(
      axis >= 0 && (unsigned int)axis < consumer->nDims() + 1,
      "Compute at called on an axis outside valid range.");

  ComputeAt::run(this, consumer, (unsigned int)axis);

  return this;
}

TensorView* TensorView::split(int axis, unsigned int factor) {
  TORCH_INTERNAL_ASSERT(nDims() > 0, "Tried to do split on a 0-dim TensorView");
  if (axis < 0)
    axis += domain()->nDims();

  if (getComputeAtView() != nullptr)
    if (axis < (int)getThisComputeAtAxis())
      TORCH_CHECK(
          false,
          "Cannot split axis within compute at range. Axis = ",
          axis,
          " thisComputeAtAxis = ",
          getThisComputeAtAxis());

  domain()->split(axis, factor);
  return this;
}

// Merge "axis" and "axis+1" into 1 dimension
TensorView* TensorView::merge(int axis_o, int axis_i) {
  TORCH_INTERNAL_ASSERT(nDims() > 0, "Tried to do merge on a 0-dim TensorView");
  if (axis_o < 0)
    axis_o += domain()->nDims();

  if (axis_i < 0)
    axis_i += domain()->nDims();

  if (getComputeAtView() != nullptr)
    if (axis_o + 1 < (int)getThisComputeAtAxis() ||
        axis_i + 1 < (int)getThisComputeAtAxis())
      TORCH_CHECK(
          false,
          "Cannot merge axis within compute at range. Either axis ",
          axis_o,
          " or ",
          axis_i,
          " are within thisComputeAtAxis = ",
          getThisComputeAtAxis());

  domain()->merge(axis_o, axis_i);
  return this;
}

TensorView* TensorView::reorder(const std::unordered_map<int, int>& old2new_) {
  TORCH_INTERNAL_ASSERT(
      !(nDims() == 0 && old2new_.size() > 0),
      "Tried to reorder a 0-dim TensorView");
  domain()->reorder(old2new_);
  return this;
}

TensorView* TensorView::rFactor(const std::vector<int>& axes) {
  TORCH_INTERNAL_ASSERT(nDims() > 0, "Tried to rFactor a 0-dim TensorView");
  FusionGuard fg(fusion());
  Expr* origin_expr = fusion()->origin(this);
  TORCH_CHECK(
      origin_expr != nullptr &&
          origin_expr->getExprType() == ExprType::ReductionOp,
      "Error rfactoring ",
      this,
      " its origin is either a nullptr or not a reduction.");
  TORCH_CHECK(
      !domain()->hasRFactor(), "Cannot call rfactor on the same view twice.");

  ReductionOp* this_origin = static_cast<ReductionOp*>(origin_expr);

  // Split tensor view into 2 parts
  auto domain_pair = domain()->rFactor(axes);

  // Producer in the pair
  auto producer_domain = domain_pair.first;
  // Consumer in the pair
  auto consumer_domain = domain_pair.second;

  // This domain will be the consumer, so create the producer
  TensorView* producer = new TensorView(producer_domain, getDataType().value());

  // Set domain of consumer
  setDomain(consumer_domain);
  TensorView* consumer = this;

  // Setup dependency chain, inserting producer before this op.
  // Expr* producer_origin =
  new ReductionOp(
      this_origin->getReductionOpType(),
      this_origin->init(),
      producer,
      this_origin->in());

  // Expr* consumer_origin =
  new ReductionOp(
      this_origin->getReductionOpType(),
      this_origin->init(),
      consumer,
      producer);

  return producer;
}

TensorView* TensorView::cache_before() {
  FusionGuard fg(fusion());

  Expr* origin_expr = fusion()->origin(this);
  TORCH_CHECK(
      origin_expr != nullptr && !fusion()->hasInput(this),
      "Error adding cache_before ",
      this,
      " its origin is a nullptr and we restrict using cache_before on an input.");

  TORCH_CHECK(
      origin_expr != nullptr &&
          origin_expr->getExprType() != ExprType::ReductionOp,
      "Error adding cache_before ",
      this,
      " its origin is a reduction, instead please use cache_after.");

  // Create Producer Domain
  // Keep Broadcast Axis (Permanent)
  auto root_domain = getRootDomain();
  std::vector<IterDomain*> new_root_domain;
  for (auto root : root_domain) {
    if (root->isBroadcast()) {
      new_root_domain.push_back(new IterDomain(
          root->start(),
          root->extent(),
          root->parallel_method(),
          false,
          false,
          true));
    } else if (!root->isBroadcast() && !root->isReduction()) {
      new_root_domain.push_back(new IterDomain(
          root->start(), root->extent(), root->parallel_method()));
    }
  }

  // This domain will be the consumer, so create the producer
  TensorView* producer =
      new TensorView(new TensorDomain(new_root_domain), getDataType().value());

  // Set domain of consumer
  TensorView* consumer = this;

  // Insert producer - Cache_Before (CB) - before this TV.
  // Before: Prev TV -> [Origin Op] -> This TV
  // After:  Prev TV -> [Origin Op] -> New CB TV -> [Set Op] -> This TV

  // Get inputs for origin expression
  auto expr_inputs = origin_expr->inputs();

  // Expr* producer_origin =
  createExprConsumer(origin_expr, producer);

  // Expr* producer_uses =
  new UnaryOp(UnaryOpType::Set, consumer, producer);

  // Before: This TV -> Next TV
  // After:  New TV (CB) -> This TV -> Next TV
  if (hasComputeAt()) {
    TransformReplay::replayPasC(producer, consumer, -1);
    auto this_ca_pos = getThisComputeAtAxis();
    producer->computeAt(consumer, this_ca_pos);
  } else {
    // Before: Prev TV -> This TV
    // After:  Prev TV -> New TV (CB) -> This TV
    // Iterate over origin expression inputs for cache_before on outputs
    for (Val* v : expr_inputs) {
      if (v->getValType().value() == ValType::TensorView) {
        TensorView* origin_input = dynamic_cast<TensorView*>(v);
        if (origin_input->hasComputeAt() &&
            origin_input->getComputeAtView() == this) {
          TransformReplay::replayPasC(producer, consumer, -1);

          auto origin_ca_pos = origin_input->getThisComputeAtAxis();
          auto origin_rel_ca_pos = origin_input->getRelativeComputeAtAxis();
          origin_input->computeAt(producer, origin_ca_pos);
          producer->setComputeAt(consumer, origin_rel_ca_pos);
        }
      }
    }
  }

  return producer;
}

TensorView* TensorView::cache_after() {
  FusionGuard fg(fusion());

  // Get all the uses for this Tensorview
  TORCH_CHECK(
      !fusion()->hasOutput(this),
      "Error adding cache_after ",
      this,
      " we restrict using cache_after on an output.");

  // Create Consumer Domain
  // Keep Broadcast Axis (Permanent)
  auto root_domain = getRootDomain();
  std::vector<IterDomain*> new_root_domain;
  for (auto root : root_domain) {
    if (root->isBroadcast()) {
      new_root_domain.push_back(new IterDomain(
          root->start(),
          root->extent(),
          root->parallel_method(),
          false,
          false,
          true));
    } else if (!root->isBroadcast() && !root->isReduction()) {
      new_root_domain.push_back(new IterDomain(
          root->start(), root->extent(), root->parallel_method()));
    }
  }

  // This domain will be the producer, so create the consumer
  TensorView* consumer =
      new TensorView(new TensorDomain(new_root_domain), getDataType().value());

  // Set domain of producer - No Change
  TensorView* producer = this;

  // Insert consumer - Cache_After (CA) - after this TV.
  // Before: This TV -> [Use Op] -> Next TV
  // After:  This TV -> [Set Op] -> New CA TV -> [Use Op] -> Next TV

  // Expr* consumer_uses =
  size_t count = 0;
  for (auto expr : fusion()->unordered_uses(this)) {
    createExprProducer(expr, this, consumer);
    ++count;
  }

  if (count > 1) {
    std::cout
        << "WARNING: Cache_After with multiple consumers can create incorrect "
           "kernels depending on computeAt configuration."
        << std::endl;
  }

  // Expr* consumer_origin =
  new UnaryOp(UnaryOpType::Set, consumer, producer);

  // Before: This TV -> Next TV
  // After:  This TV -> New TV (After) -> Next TV
  if (hasComputeAt()) {
    TransformReplay::replayCasP(consumer, producer, -1);

    auto rel_ca_pos = getRelativeComputeAtAxis();
    auto this_ca_pos = getThisComputeAtAxis();
    auto this_ca_view = getComputeAtView();

    computeAt(consumer, this_ca_pos);
    consumer->setComputeAt(this_ca_view, rel_ca_pos);
  } else {
    // Check users of this TV for computeAt for cache_after on inputs
    for (auto expr : fusion()->unordered_uses(consumer)) {
      auto expr_outputs = expr->outputs();
      for (Val* v : expr_outputs) {
        if (v->getValType().value() == ValType::TensorView) {
          TensorView* output = dynamic_cast<TensorView*>(v);
          if (output->hasComputeAt()) {
            TransformReplay::replayPasC(consumer, output, -1);
            auto output_ca_pos = output->getThisComputeAtAxis();
            consumer->setComputeAt(output, output_ca_pos);
          }
        }
      }
    }
  }

  return consumer;
}

namespace {

// Create New Expr given consumer - [output of the expression]
struct CreateExprConsumer : public OptInDispatch {
 public:
  static void create(Expr* expr, TensorView* consumer) {
    CreateExprConsumer cec(consumer);
    cec.handle(expr);
  }

 private:
  explicit CreateExprConsumer(TensorView* consumer) : consumer_(consumer) {}

  void handle(Expr* expr) final {
    OptInDispatch::handle(expr);
  }

  void handle(UnaryOp* unary_expr) final {
    new UnaryOp(unary_expr->getUnaryOpType(), consumer_, unary_expr->in());
  }

  void handle(BinaryOp* binary_expr) final {
    new BinaryOp(
        binary_expr->getBinaryOpType(),
        consumer_,
        binary_expr->lhs(),
        binary_expr->rhs());
  }

  void handle(TernaryOp* ternary_expr) final {
    new TernaryOp(
        ternary_expr->getTernaryOpType(),
        consumer_,
        ternary_expr->in1(),
        ternary_expr->in2(),
        ternary_expr->in3());
  }

  void handle(ReductionOp* reduction_expr) final {
    new ReductionOp(
        reduction_expr->getReductionOpType(),
        reduction_expr->init(),
        consumer_,
        reduction_expr->in());
  }

  void handle(BroadcastOp* broadcast_expr) final {
    new BroadcastOp(consumer_, broadcast_expr->in());
  }

 private:
  TensorView* consumer_ = nullptr;
};

// Create New Expr given producer - [an input for the expression]
struct CreateExprProducer : public OptInDispatch {
 public:
  static void create(Expr* expr, TensorView* current, TensorView* producer) {
    CreateExprProducer cep(current, producer);
    cep.handle(expr);
  }

 private:
  explicit CreateExprProducer(TensorView* current, TensorView* producer)
      : current_(current), producer_(producer) {}

  void handle(Expr* expr) final {
    OptInDispatch::handle(expr);
  }

  void handle(UnaryOp* unary_expr) final {
    new UnaryOp(unary_expr->getUnaryOpType(), unary_expr->out(), producer_);
  }

  void handle(BinaryOp* binary_expr) final {
    if (binary_expr->lhs()->sameAs(current_)) {
      new BinaryOp(
          binary_expr->getBinaryOpType(),
          binary_expr->out(),
          producer_,
          binary_expr->rhs());
    } else {
      new BinaryOp(
          binary_expr->getBinaryOpType(),
          binary_expr->out(),
          binary_expr->lhs(),
          producer_);
    }
  }

  void handle(TernaryOp* ternary_expr) final {
    if (ternary_expr->in1()->sameAs(current_)) {
      new TernaryOp(
          ternary_expr->getTernaryOpType(),
          ternary_expr->out(),
          producer_,
          ternary_expr->in2(),
          ternary_expr->in3());
    } else if (ternary_expr->in2()->sameAs(current_)) {
      new TernaryOp(
          ternary_expr->getTernaryOpType(),
          ternary_expr->out(),
          ternary_expr->in1(),
          producer_,
          ternary_expr->in3());
    } else {
      new TernaryOp(
          ternary_expr->getTernaryOpType(),
          ternary_expr->out(),
          ternary_expr->in1(),
          ternary_expr->in2(),
          producer_);
    }
  }

  void handle(ReductionOp* reduction_expr) final {
    new ReductionOp(
        reduction_expr->getReductionOpType(),
        reduction_expr->init(),
        reduction_expr->out(),
        producer_);
  }

  void handle(BroadcastOp* broadcast_expr) final {
    new BroadcastOp(broadcast_expr->out(), producer_);
  }

 private:
  TensorView* current_ = nullptr;
  TensorView* producer_ = nullptr;
};

} // namespace

// In Cache Before, for the origin expr of the original tensor,
// we create a new operation where the original tensor is replaced
// with the new cache tensor. This function creates a new expr
// given the consumer, the output of the expression.
void TensorView::createExprConsumer(Expr* expr, TensorView* consumer) {
  CreateExprConsumer::create(expr, consumer);
}

// In Cache After, for all the uses of the original tensor, we create
// a new operation where the original tensor is replaced with the new
// cache tensor. This function creates a new expr given a producer,
// an input for the expression.
void TensorView::createExprProducer(
    Expr* expr,
    TensorView* current,
    TensorView* producer) {
  CreateExprProducer::create(expr, current, producer);
}

} // namespace fuser
} // namespace jit
} // namespace torch
