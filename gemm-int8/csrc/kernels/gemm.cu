#include <gemm.h>
#include <cutlass/float8.h>
#include "cutlass/float8.h"


#include <stddef.h>
#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>


#include <cutlass/core_io.h>
#include <cutlass/cutlass.h>
#include <cutlass/half.h>

#include <cutlass/gemm/device/gemm.h>
#include <cutlass/numeric_types.h>
#include <cutlass/util/host_tensor.h>


template <typename TileShape, typename WarpShape, int kStages>
torch::Tensor int8_matmul(torch::Tensor input,  // INT8 - shape (M, K)
                                  torch::Tensor weight, // INT8 - shape (N, K)
                                  float alpha          // FP32
){
  auto M = input.size(0);
  auto N = weight.size(0);
  auto K = input.size(1);

  auto out = torch::empty({M, N}, torch::dtype(torch::kBFloat16).device(input.device()));

  using ElementOutput = cutlass::bfloat16_t;
  using ElementAccumulator = int32_t;
  using ElementComputeEpilogue = float;
  using ElementInputA = int8_t; // <- data type of input matrix A
  using ElementInputB = int8_t; // <- data type of weight matrix B

  // The code below describes matrix layout 
  // Column Major for Matrix A, Row Major for Matrix B and Row Major for Matrix C
  using LayoutInputA = cutlass::layout::RowMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::RowMajor;

  using Gemm = cutlass::gemm::device::Gemm<
      int8_t,
      cutlass::layout::RowMajor,
      int8_t,
      cutlass::layout::ColumnMajor,
      ElementOutput,
      cutlass::layout::RowMajor,
      ElementAccumulator,
      cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm80,
      TileShape,
      WarpShape,
      cutlass::gemm::GemmShape<16, 8, 32>,
      cutlass::epilogue::thread::LinearCombination<
          ElementOutput,
          128 / cutlass::sizeof_bits<ElementOutput>::value,
          ElementAccumulator,
          ElementComputeEpilogue>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
      kStages>;

  auto input_size = cutlass::MatrixCoord(M, K);
  auto weight_size = cutlass::MatrixCoord(K, N);
  auto output_size = cutlass::MatrixCoord(M, N);

  auto device = input.device();

  cutlass::gemm::GemmCoord problem_size(M, N, K);


  cutlass::TensorRef<ElementInputA, LayoutInputA> input_ref(
      static_cast<ElementInputA*>(input.data_ptr()),
      LayoutInputA::packed(input_size));

  cutlass::TensorRef<ElementInputB, LayoutInputB> weight_ref(
      static_cast<ElementInputB*>(weight.data_ptr()),
      LayoutInputB::packed(weight_size));

  cutlass::TensorRef<ElementOutput, LayoutOutput> out_ref(
      static_cast<ElementOutput*>(out.data_ptr()),
      LayoutOutput::packed(output_size));

  typename Gemm::Arguments arguments{
      problem_size, // <- problem size of matrix multiplication
      input_ref,    // <- reference to matrix A on device
      weight_ref,   // <- reference to matrix B on device
      out_ref,      // <- reference to matrix C on device
      out_ref,      // <- reference to matrix D on device
      {alpha, 0.0}, 1};
  
  Gemm gemm_op;

  // Using the arguments, query for extra workspace required for matrix
  // multiplication computation
  size_t workspace_size = Gemm::get_workspace_size(arguments);

  // Allocate workspace memory
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  // Check the problem size is supported or not
  cutlass::Status status = gemm_op.can_implement(arguments);
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error("cutlass cannot implement");
  }

  // Initialize CUTLASS kernel with arguments and workspace pointer
  status = gemm_op.initialize(arguments, workspace.get());
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error("cutlass cannot initialize");
  }

  status = gemm_op();
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error("cutlass cannot run");
  }

  return out;
}

torch::Tensor int8_matmul_host(torch::Tensor input,  // INT8
                                  torch::Tensor weight, // INT8
                                  float alpha          // FP32
){
  auto M = input.size(0);
  auto N = weight.size(0);
  auto K = input.size(1);

  if (M==512 && N==4096 && K==4096){
    using TileShape = typename cutlass::gemm::GemmShape<128, 128, 128>;
    using WarpShape = typename cutlass::gemm::GemmShape<64, 64, 128>;
    static const int kStages = 3;
    return int8_matmul<TileShape, WarpShape, kStages>(input, weight, alpha);
  } else if (M==512 && N==4096 && K==14336){
    using TileShape = typename cutlass::gemm::GemmShape<128, 128, 64>;
    using WarpShape = typename cutlass::gemm::GemmShape<64, 64, 64>;
    static const int kStages = 4;
    return int8_matmul<TileShape, WarpShape, kStages>(input, weight, alpha);
  } else if (K==4096 && N==4096){
    using TileShape = typename cutlass::gemm::GemmShape<256, 128, 64>;
    using WarpShape = typename cutlass::gemm::GemmShape<64, 64, 64>;
    static const int kStages = 3;
    return int8_matmul<TileShape, WarpShape, kStages>(input, weight, alpha);
  } else if (M==1024 && N==14336 && K==4096){
    using TileShape = typename cutlass::gemm::GemmShape<128, 128, 64>;
    using WarpShape = typename cutlass::gemm::GemmShape<64, 64, 64>;
    static const int kStages = 3;
    return int8_matmul<TileShape, WarpShape, kStages>(input, weight, alpha);
  } else {
    using TileShape = typename cutlass::gemm::GemmShape<256, 128, 64>;
    using WarpShape = typename cutlass::gemm::GemmShape<64, 64, 64>;
    static const int kStages = 3;
    return int8_matmul<TileShape, WarpShape, kStages>(input, weight, alpha);
  }
}


torch::Tensor int8_bmm_matmul_host(torch::Tensor input,  // INT8
                                  torch::Tensor weight, // INT8
                                  float alpha          // FP32
){
  auto BATCH = input.size(0);
  auto M = input.size(1);
  auto K = input.size(2);

  bool shared_weight = false;
	int64_t N = 0;
	if (weight.dim() == 2)
	{
		TORCH_CHECK(weight.size(1) == K, "B(shared) must be [N, K]");
		N = weight.size(0);
		shared_weight = true;
	}
	else
	{
		TORCH_CHECK(weight.dim() == 3, "B must be [N, K] or [B, N, K]");
		TORCH_CHECK(weight.size(0) == BATCH && weight.size(2) == K, "B is [B, N, K] and must match A");
		N = weight.size(1);
	}

  auto out = torch::empty({BATCH, M, N}, torch::dtype(torch::kBFloat16).device(input.device()));

  for (int b = 0; b < BATCH; ++b)
	{
		torch::Tensor Ab = input.select(0, b).contiguous();				   // [M, K]
		torch::Tensor Bb = shared_weight ? weight : weight.select(0, b).contiguous(); // [N, K]

    torch::Tensor Cb = int8_matmul_host(Ab, Bb, alpha); // [M, N]

    out.select(0, b).copy_(Cb, /*non_blocking=*/true);
  }
  
  return out;
}


__global__ void bf16_rowcol_scale_kernel(
    const cutlass::bfloat16_t* __restrict__ input,      // [M, N]
    const float* __restrict__ row_scale,                // [M]
    const float* __restrict__ col_scale,                // [N]
    cutlass::bfloat16_t* output,           // [M, N]
    int M,
    int N
) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row >= M || col >= N) return;

  int idx = row * N + col;

  float y = static_cast<float>(input[idx]);
  float rs = row_scale[row];
  float cs = col_scale[col];

  float val = y * rs * cs;

  output[idx] = cutlass::bfloat16_t(val);
}

torch::Tensor int8_matmul_and_dequantize_host(
    torch::Tensor input,   // INT8
    torch::Tensor weight,  // INT8
    float alpha,           // FP32
    torch::Tensor row_scale, // FP32
    torch::Tensor col_scale  // FP32
){
  TORCH_CHECK(input.is_cuda(), "input must be CUDA");
  TORCH_CHECK(row_scale.is_cuda() && col_scale.is_cuda(), "scales must be CUDA");

  auto out = int8_matmul_host(input, weight, alpha); // [N, M], BF16

  auto M = out.size(0);
  auto N = out.size(1);

  dim3 blockDim(16, 16);
  dim3 gridDim((N + blockDim.x - 1) / blockDim.x,
               (M + blockDim.y - 1) / blockDim.y);

  bf16_rowcol_scale_kernel<<<gridDim, blockDim, 0>>>(
      reinterpret_cast<const cutlass::bfloat16_t*>(out.data_ptr<at::BFloat16>()),
      row_scale.data_ptr<float>(),
      col_scale.data_ptr<float>(),
      reinterpret_cast<cutlass::bfloat16_t*>(out.data_ptr<at::BFloat16>()),
      M,
      N
  );
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

torch::Tensor int8_matmul_and_dequantize_batched(
    torch::Tensor input,   // INT8 - shape (B, N, K)
    torch::Tensor weight,  // INT8 - shape (B, M, K) or (M, K)
    float alpha,          // FP32 - scalar
    torch::Tensor row_scale, // FP32 - shape (B, N)
    torch::Tensor col_scale  // FP32 - shape (B, M) or (M)
){
  auto BATCH = input.size(0);
  auto N = input.size(1);
  auto K = input.size(2);
  auto M = 0;

  bool shared_weight = false;

  if (weight.dim() == 2)
  {
    TORCH_CHECK(weight.size(1) == K, "B(shared) must be [M, K]");
    shared_weight = true;
    M = weight.size(0);
  }
  else
  {
    TORCH_CHECK(weight.dim() == 3, "B must be [B, M, K]");
    TORCH_CHECK(weight.size(0) == BATCH && weight.size(2) == K, "B is [B, M, K] and must match A");
    M = weight.size(1);
  }

  auto out = torch::empty({BATCH, N, M}, torch::dtype(torch::kBFloat16).device(input.device()));


  for (int b = 0; b < BATCH; ++b)
  { 
    torch::Tensor Ab = input.select(0, b).contiguous();                   // [N, K]
    torch::Tensor Bb = shared_weight ? weight : weight.select(0, b).contiguous(); // [M, K]
    torch::Tensor row_scale_b = row_scale.select(0, b).contiguous();      // [N]
    torch::Tensor col_scale_b = shared_weight ? col_scale : col_scale.select(0, b).contiguous(); // [M]

    torch::Tensor Cb = int8_matmul_and_dequantize_host(Ab, Bb, alpha, row_scale_b, col_scale_b); // [N, M]

    out.select(0, b).copy_(Cb, /*non_blocking=*/true);
  }
  
  return out;
}
