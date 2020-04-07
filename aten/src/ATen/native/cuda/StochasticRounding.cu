#include <ATen/ATen.h>
#include <ATen/native/cuda/stochastic_rounding.cuh>

#define DISPATCH_FLOAT_AND_HALF(TYPE, LEVEL, NAME, ...) \
  switch(TYPE) \
  { \
    case at::ScalarType::Float: \
    { \
      using scalar_t_##LEVEL = float; \
      __VA_ARGS__; \
      break; \
    } \
    case at::ScalarType::Half: \
    { \
      using scalar_t_##LEVEL = at::Half; \
      __VA_ARGS__; \
      break; \
    } \
    default: \
      AT_ERROR(#NAME, " not implemented for '", toString(TYPE), "'");  \
  }

namespace at {
namespace native {

template <typename input_t, typename output_t>
__global__ void stochastic_rounding_kernel(
    const input_t* input,
    output_t* output,
    const int64_t numel,
    std::pair<uint64_t, uint64_t> seed_and_offset) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  curandStatePhilox4_32_10_t state;
  curand_init(seed_and_offset.first, tid, seed_and_offset.second, &state);

  for (int64_t i = tid; i < numel; i += blockDim.x * gridDim.x) {
    float inp = static_cast<float>(input[i]);
    output[i] = round_stochastically<output_t>(inp, curand_uniform(&state));
  }
}

Tensor stochastic_rounding_cuda(const Tensor& input, Tensor& output, Generator gen_) {
  TORCH_CHECK(input.numel() > 0 && input.numel() == output.numel());
  TORCH_CHECK(input.is_contiguous());
  TORCH_CHECK(output.is_contiguous());

  const int64_t numel = input.numel();
  const int block = 256;
  const int blocks_per_sm = at::cuda::getCurrentDeviceProperties()->maxThreadsPerMultiProcessor / block;
  unsigned int grid = (numel + block - 1) / block;
  grid = std::min((unsigned int)at::cuda::getCurrentDeviceProperties()->multiProcessorCount * blocks_per_sm, grid);

  auto gen = get_generator_or_default<CUDAGeneratorImpl>(gen_, cuda::detail::getDefaultCUDAGenerator());
  std::pair<uint64_t, uint64_t> rng_engine_inputs;
  {
    std::lock_guard<std::mutex> lock(gen->mutex_);
    rng_engine_inputs = gen->philox_engine_inputs((numel + block * grid - 1) / (block * grid));
  }

  DISPATCH_FLOAT_AND_HALF(
    input.scalar_type(), 0, "round_stochastically_input",
    DISPATCH_FLOAT_AND_HALF(
      output.scalar_type(), 1, "round_stochastically_output",
      stochastic_rounding_kernel<scalar_t_0, scalar_t_1><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        input.data_ptr<scalar_t_0>(),
        output.data_ptr<scalar_t_1>(),
        numel, rng_engine_inputs);
      ));

  return output;
}

} // namespace native
} // namespace at
#undef DISPATCH_FLOAT_AND_HALF
