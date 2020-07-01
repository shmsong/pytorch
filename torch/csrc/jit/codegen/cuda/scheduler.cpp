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
    // TODO: Remove magic statement
    block_dim_x /= 4;
  }

  // 3. Applying Power of 2 Blocking based on the Maximum Number of threads

  // TODO: Magic number of 4 for vectorizing 
  int MAX_NUM_THREADS  = (red_on_fastest_dim ? 512 : 512 / 4);
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

  //int inputs_to_consume_per_iter_by_block_dim_y = 0;
  //int inputs_to_consume_per_block_per_iter = 1;
  int inputs_consumed_per_block_iter = 1;
  //int inputs_per_thread = 0;
  int red_elems_per_thread = red_elems;

  int outputs_produced_per_block_iter = 1;
  bool reduce_inputs_across_warps = false;
  //int outputs_to_consume_per_block_per_iter = 1;

  // Reduction is performed across warp threads (cross-thread reduction)
  if (red_on_fastest_dim) {
	//inputs_to_consume_per_block_per_iter *= block_dim_x;
    inputs_consumed_per_block_iter *= block_dim_x;
	//inputs_per_thread = ceil_div(red_inputs, inputs_to_consume_per_block_per_iter);
    red_elems_per_thread = ceil_div(red_elems_per_thread, inputs_consumed_per_block_iter);
    // Decision to do a cross-warp reduction per block
	//if ((red_elems / block_dim_x) >= (block_dim_y * 16) || (red_elems / block_dim_x) >= 256)  {
  // Warp threads are applied across the output
  } else {
    outputs_produced_per_block_iter *= block_dim_x;
  }

  // Decision to do a cross-warp reduction per block
  if (red_elems_per_thread >= (block_dim_y * 16) || red_elems_per_thread >= 256)  {
    //inputs_to_consume_per_iter_by_block_dim_y = block_dim_x * block_dim_y;
    inputs_consumed_per_block_iter *= block_dim_y;
    red_elems_per_thread = ceil_div(red_elems_per_thread, block_dim_y);
    reduce_inputs_across_warps = true;
    //inputs_per_thread = ceil_div(red_inputs, inputs_to_consume_per_block_per_iter);
  // Do multiple reductions per block
  } else {
    outputs_produced_per_block_iter *= block_dim_y;
  }

  // 5. Distributing work across blocks

  // Magic numbers of calculations allowed per thread.
  constexpr int MIN_VALUES_PER_THREAD = 16;
  constexpr int MAX_VALUES_PER_THREAD = 256;

  int DEVICE_MAX_THREADS_PER_MULTIPROCESSOR = at::cuda::getCurrentDeviceProperties()->maxThreadsPerMultiProcessor;
  int DEVICE_MULTIPROCESSOR_COUNT = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;

  int blocks_per_sm = DEVICE_MAX_THREADS_PER_MULTIPROCESSOR / (block_dim_x * block_dim_y);
  int target_grid_size = DEVICE_MULTIPROCESSOR_COUNT * blocks_per_sm;

  //Setting the number of blocks based on the number of outputs
  grid_dim_x = ceil_div(red_outputs / (red_on_fastest_dim ? 1 : 4), outputs_produced_per_block_iter);
  //grid_dim_x = ceil_div(red_outputs , outputs_produced_per_block_iter);

  // Cross-block reductions (if necessary)
  if (    reduce_inputs_across_warps
       && red_elems_per_thread >= MAX_VALUES_PER_THREAD
       && grid_dim_x <= target_grid_size                  ) {

    int blks_per_out_1  = ceil_div(target_grid_size, grid_dim_x);
    int blks_per_out_2  = ceil_div(red_elems_per_thread, MIN_VALUES_PER_THREAD);
    int blks_per_out_3  = ceil_div(red_elems_per_thread, MAX_VALUES_PER_THREAD);
    int blks_per_output = std::max(std::min(blks_per_out_1, blks_per_out_2), blks_per_out_3);

    std::cout << "GRIDDIM_y: " << target_grid_size << " " <<red_elems_per_thread << " " << blks_per_out_1 << " " << blks_per_out_2 << " " << blks_per_out_3 << std::endl;

    grid_dim_y = std::max(1, blks_per_output);
    //If a cross-block reduction was generated
    if (blks_per_output > 1) {
      //grid_dim_y = ceil_div(inputs_consumed_per_block_iter, blks_per_output);
      inputs_consumed_per_block_iter *= blks_per_output;
      red_elems_per_thread = ceil_div(red_elems_per_thread, inputs_consumed_per_block_iter);
    }
  }
  // --------- END Heuristic ---------

  std::cout << "GRIDS/BLOCKS: " << grid_dim_x << " " << grid_dim_y << " " << block_dim_x << " " << block_dim_y << std::endl;

  // Heuristic Definition
  // TODO: Need to factor in unrolling
  // TODO: Need to get rid of magic numbers.  These should be defined elsewhere.
  if (red_on_fastest_dim) {
    // Initially I am not going to bother with cross-block reductions or
    // letting a block do multiple reductions to make this simple!
    bool do_mult_reds_per_block = red_dims[0] != grid_dim_x;
    bool do_cross_block_reds    = grid_dim_y > 1;

    // Do multiple reductions per block
	if (do_mult_reds_per_block) {
      red_tv->split(-1, block_dim_x);
      // Split Grid dimension to get multiple reds per block
      red_tv->split(0, block_dim_y);

      auto red_tv_rf = red_tv->rFactor({-2});
      red_tv_rf->computeAt(red_tv, 1);

      red_tv->axis(0)->parallelize(ParallelType::BIDx);

      red_tv_rf->axis(-1)->parallelize(ParallelType::TIDx);
      red_tv->axis(-1)->parallelize(ParallelType::TIDx);

      red_tv_rf->axis(1)->parallelize(ParallelType::TIDy);
      red_tv->axis(1)->parallelize(ParallelType::TIDy);
	// Do a cross-warp reduction per block
    } else {
      if (do_cross_block_reds) {
      	red_tv->split(-1, block_dim_x);
        // Split up rFactor to reduce across warps
        red_tv->split(-2, block_dim_y);
        red_tv->split(-3, grid_dim_y);

        auto red_tv_rf = red_tv->rFactor({-4});
        red_tv_rf->computeAt(red_tv, 1);

        red_tv->axis(0)->parallelize(ParallelType::BIDx);

        // Cross-block reduction binding
        red_tv_rf->axis(-3)->parallelize(ParallelType::BIDy);
        red_tv->axis(-3)->parallelize(ParallelType::BIDy);

        red_tv_rf->axis(-1)->parallelize(ParallelType::TIDx);
        red_tv->axis(-1)->parallelize(ParallelType::TIDx);

        red_tv_rf->axis(-2)->parallelize(ParallelType::TIDy);
        red_tv->axis(-2)->parallelize(ParallelType::TIDy);

      } else {
      	red_tv->split(-1, block_dim_x);
        // Split up rFactor to reduce across warps
        red_tv->split(-2, block_dim_y);

        auto red_tv_rf = red_tv->rFactor({-3});
        red_tv_rf->computeAt(red_tv, 1);

        red_tv->axis(0)->parallelize(ParallelType::BIDx);

        red_tv_rf->axis(-1)->parallelize(ParallelType::TIDx);
        red_tv->axis(-1)->parallelize(ParallelType::TIDx);

        red_tv_rf->axis(-2)->parallelize(ParallelType::TIDy);
        red_tv->axis(-2)->parallelize(ParallelType::TIDy);
      }
    }
  } else {
    if (block_dim_y > 1) {
      red_tv->split(-1, block_dim_x);
	  if(grid_dim_y > 1)
		red_tv->split(0,grid_dim_y);
      red_tv->split(0, block_dim_y);
      auto red_tv_rf = red_tv->rFactor({0});
      red_tv_rf->axis(-1)->parallelize(ParallelType::TIDx);
      red_tv_rf->axis(-2)->parallelize(ParallelType::BIDx);
	  if(grid_dim_y > 1) {
      	red_tv_rf->axis(-3)->parallelize(ParallelType::BIDy);
      	red_tv_rf->axis(-4)->parallelize(ParallelType::TIDy);
      } else {
      	red_tv_rf->axis(-3)->parallelize(ParallelType::TIDy);
      }
      red_tv->axis(-1)->parallelize(ParallelType::TIDx);
      red_tv->axis(-2)->parallelize(ParallelType::BIDx);
      red_tv->axis(-3)->parallelize(ParallelType::TIDy);
	  if(grid_dim_y > 1) {
        red_tv->axis(-3)->parallelize(ParallelType::BIDy);
        red_tv->axis(-4)->parallelize(ParallelType::TIDy);
      } else {
        red_tv->axis(-3)->parallelize(ParallelType::TIDy);
      }
	} else {
      red_tv->split(-1, block_dim_x);
      red_tv->axis(-1)->parallelize(ParallelType::TIDx);
      red_tv->axis(-2)->parallelize(ParallelType::BIDx);
    }
  }
  std::cout << grid_dim_x << " " << grid_dim_y << " " << block_dim_x << " " << block_dim_y << std::endl;
  return std::tuple<int,int,int,int>({grid_dim_x, grid_dim_y, block_dim_x, block_dim_y});
}

} // namespace fuser
} // namespace jit
} // namespace torch
