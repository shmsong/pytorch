#include <torch/csrc/jit/codegen/cuda/arith.h>
#include <torch/csrc/jit/codegen/cuda/expr_evaluator.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/scheduler.h>
#include <ATen/cuda/CUDAContext.h>

namespace torch {
namespace jit {
namespace fuser {

namespace {
inline int ceil_div(int a, int b) {
  return (a + b - 1) / b;
}

// Largest Power of 2 less-than n
inline int last_pow2(int n) {
  n |= (n >>  1);
  n |= (n >>  2);
  n |= (n >>  4);
  n |= (n >>  8);
  n |= (n >> 16);
  return std::max(1, n - (n >> 1));
}
} // empty namespace

c10::optional<std::tuple<int,int,int,int>>
Scheduler::reduction(Fusion *fusion, const at::ArrayRef<IValue> &inputs) {
  // TODO: I am making a larger initial assumption that reductions are
  //       2D at this point to make the issue easier, right now.

  // Find Reduction TensorView
  TensorView* red_tv = nullptr;
  for (auto &expr : fusion->exprs(/*from_outputs_only*/true)) {
    if(expr->type() == ExprType::ReductionOp) {
      red_tv = static_cast<TensorView*>(expr->output(0));
    }
  }
  if (red_tv == nullptr) // No reduction found
    return c10::nullopt;

  EvaluationContext eval_context(fusion);

  // I am making some basic assumptions
  // 1.) I am only binding Tensor Dimension sizes.  I am not binding scalar values.
  // 2.) I am only binding the IterDomain.extent().  Do I need to worry about the start?
  for(size_t i = 0; i < inputs.size(); ++i) {
    if(inputs[i].type()->kind() == c10::TypeKind::TensorType) {
      TensorView* tv = static_cast<TensorView*>(fusion->inputs()[i]);
      size_t dims = tv->getRootDomain().size();
      for(size_t j = 0; j < dims; ++j) {
        IterDomain* id = tv->getRootDomain()[j];
        eval_context.bind(id->extent(), inputs[i].toTensor().size(j));
      }
    }
  }

  // --------- START Heuristic ---------
  // 1. Initial Assumptions

  // Evaluate Dimensions of Reduction TensorView
  auto red_ids = red_tv->domain()->domain();
  std::vector<Int::ScalarType> red_dims(red_ids.size(), 0);
  int red_idx = 0;
  int red_inputs = 1;
  int red_outputs = 1;
  int red_elems   = 1;

  for(size_t i = 0; i < red_ids.size(); ++i) {
    red_dims[i] = ExpressionEvaluator::evaluate(red_ids[i]->extent(), &eval_context).value();
    if (red_ids[i]->isReduction()) {
      red_idx = i;
      red_elems *= red_dims[i];
    } else {
      red_outputs *= red_dims[i];
    }
    red_inputs *= red_dims[i];
  }
  std::cout << "Reduction Dims: [";
  for(auto &dim : red_dims) {
    std::cout << dim << ",";
  }
  std::cout << "]" << std::endl;

  int block_dim_x = 1;
  int block_dim_y = 1;
  int grid_dim_x = 1;
  int grid_dim_y = 1;

  // 2. Initial Definition of Block Dimensions

  // Is fastest dimension a reduction dimension?
  bool red_on_fastest_dim = red_idx == (red_dims.size()-1);
  if (red_on_fastest_dim) {
    block_dim_x = red_elems;
    block_dim_y = red_outputs;
  } else {
    block_dim_x = red_outputs;
    block_dim_y = red_elems;
  }

  // 3. Applying Power of 2 Blocking based on the Maximum Number of threads

  constexpr int MAX_NUM_THREADS  = 512;
  int DEVICE_WARP_SIZE = at::cuda::warp_size();

  if (block_dim_x < MAX_NUM_THREADS)
	block_dim_x = last_pow2(block_dim_x);
  else
	block_dim_x = MAX_NUM_THREADS;

  if (block_dim_y < MAX_NUM_THREADS)
	block_dim_y = last_pow2(block_dim_y);
  else
	block_dim_y = MAX_NUM_THREADS;

  int block_dim_x_prev = block_dim_x;
  block_dim_x = std::min(block_dim_x, DEVICE_WARP_SIZE);
  block_dim_y = std::min(block_dim_y, MAX_NUM_THREADS / block_dim_x);
  block_dim_x = std::min(block_dim_x_prev, MAX_NUM_THREADS / block_dim_y);

  // 4. Distributing work across a block

  int inputs_to_consume_per_iter_by_block_dim_y = 0;
  int inputs_to_consume_per_block_per_iter = 1;
  int inputs_per_thread = 0;

  int outputs_to_consume_per_block_per_iter = 1;

  if (red_on_fastest_dim) {
	// Reduction is performed across warp threads (cross-thread reduction)
	inputs_to_consume_per_block_per_iter *= block_dim_x;
	inputs_per_thread = ceil_div(red_inputs, inputs_to_consume_per_block_per_iter);
	if ((red_elems / block_dim_x) >= (block_dim_y * 16) || (red_elems / block_dim_x) >= 256)  {
      inputs_to_consume_per_iter_by_block_dim_y = block_dim_x * block_dim_y;
      inputs_to_consume_per_block_per_iter *= block_dim_y;
      inputs_per_thread = ceil_div(red_inputs, inputs_to_consume_per_block_per_iter);
	} else {
	  outputs_to_consume_per_block_per_iter *= block_dim_y;
    }
  } else {
	// Warp threads are applied across the output
    outputs_to_consume_per_block_per_iter *= block_dim_x;
    outputs_to_consume_per_block_per_iter *= block_dim_y;
  }

  // 5. Distributing work across blocks

  // Magic numbers of calculations allowed per thread.
  constexpr int MIN_REDUCTIONS_PER_THREAD = 16;
  constexpr int MAX_REDUCTIONS_PER_THREAD = 256;

  int DEVICE_MAX_THREADS_PER_MULTIPROCESSOR = at::cuda::getCurrentDeviceProperties()->maxThreadsPerMultiProcessor;
  int DEVICE_MULTIPROCESSOR_COUNT = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;

  int blocks_per_sm = DEVICE_MAX_THREADS_PER_MULTIPROCESSOR / (block_dim_x * block_dim_y);
  int target_grid_size = DEVICE_MULTIPROCESSOR_COUNT * blocks_per_sm;

  //Setting the number of blocks based on the number of outputs
  grid_dim_x = ceil_div(red_outputs, outputs_to_consume_per_block_per_iter);

  // Cross-block reductions (if necessary)
  /*if (    inputs_to_consume_per_iter_by_block_dim_y != 0
       && inputs_per_thread >= MAX_REDUCTIONS_PER_THREAD
       && grid_dim_x <= target_grid_size                  ) {

    grid_dim_y = std::max(std::min(ceil_div(target_grid_size, grid_dim_x),
                                   ceil_div(inputs_per_thread, MIN_REDUCTIONS_PER_THREAD)),
                          ceil_div(inputs_per_thread, MAX_REDUCTIONS_PER_THREAD));

    //If a cross-block reduction was generated
    if (grid_dim_y > 1) {
      inputs_to_consume_per_block_per_iter /= grid_dim_y;
      inputs_per_thread = ceil_div(inputs_per_thread, inputs_to_consume_per_block_per_iter);
    }
  }*/
  // --------- END Heuristic ---------

  // Heuristic Definition
  // TODO: Need to factor in unrolling
  // TODO: Need to get rid of magic numbers.  These should be defined elsewhere.
  if (red_on_fastest_dim) {
    // Initially I am not going to bother with cross-block reductions or
    // letting a block do multiple reductions to make this simple!
    red_tv->split(-1, block_dim_x);
    int rf_dim = 0;
    if(red_dims[0] != grid_dim_x) {
      red_tv->split(0, block_dim_y);
      rf_dim = -2;
    } else {
      rf_dim = -3;
      red_tv->split(-2, block_dim_y);
    }
    auto red_tv_rf = red_tv->rFactor({rf_dim});

    red_tv_rf->computeAt(red_tv, 1);

    red_tv->axis(0)->parallelize(ParallelType::BIDx);
    if(red_dims[0] != grid_dim_x) {
      red_tv_rf->axis(1)->parallelize(ParallelType::TIDy);
      red_tv->axis(1)->parallelize(ParallelType::TIDy);
    } else {
      red_tv_rf->axis(-2)->parallelize(ParallelType::TIDy);
      red_tv->axis(-2)->parallelize(ParallelType::TIDy);
    }

    red_tv_rf->axis(-1)->parallelize(ParallelType::TIDx);
    red_tv->axis(-1)->parallelize(ParallelType::TIDx);
  } else {
    red_tv->split(-1, block_dim_x);
    red_tv->split(-2, block_dim_y);
    red_tv->axis(-1)->parallelize(ParallelType::TIDx);
    red_tv->axis(-2)->parallelize(ParallelType::TIDy);
    red_tv->axis(-3)->parallelize(ParallelType::BIDx);
  }
  std::cout << grid_dim_x << " " << grid_dim_y << " " << block_dim_x << " " << block_dim_y << std::endl;
  return std::tuple<int,int,int,int>({grid_dim_x, grid_dim_y, block_dim_x, block_dim_y});
}

} // namespace fuser
} // namespace jit
} // namespace torch
