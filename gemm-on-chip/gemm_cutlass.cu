#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "cutlass/cutlass.h"
#include "cutlass/core_io.h"
#include "cutlass/numeric_types.h"
#include "cutlass/half.h"
#include "cutlass/float8.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/util/host_tensor.h"

// ================================================================
// Core GEMM
// ================================================================

template <typename TileShape, typename WarpShape, int kStages>
torch::Tensor int8_matmul(
    torch::Tensor input,   // INT8 - shape (M, K)
    torch::Tensor weight,  // INT8 - shape (N, K)
    float alpha            // FP32
) {
  TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
  TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");

  TORCH_CHECK(input.dtype() == torch::kChar,
              "input must be torch.int8 (kChar)");
  TORCH_CHECK(weight.dtype() == torch::kChar,
              "weight must be torch.int8 (kChar)");

  TORCH_CHECK(input.dim() == 2 && weight.dim() == 2,
              "input and weight must be 2D tensors");

  auto M = input.size(0);
  auto K = input.size(1);
  auto N = weight.size(0);  // as in your code

  TORCH_CHECK(weight.size(1) == K,
              "weight shape must be (N, K) with same K as input");

  // Make sure we have contiguous memory in the expected layout
  input = input.contiguous();
  weight = weight.contiguous();

  auto options = torch::TensorOptions()
                     .dtype(torch::kBFloat16)
                     .device(input.device());
  auto out = torch::empty({M, N}, options);

  using ElementOutput = cutlass::bfloat16_t;
  using ElementAccumulator = int32_t;
  using ElementComputeEpilogue = float;
  using ElementInputA = int8_t;
  using ElementInputB = int8_t;

  // Layouts as in your code
  using LayoutInputA = cutlass::layout::RowMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::RowMajor;

  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
          ElementOutput,
          128 / cutlass::sizeof_bits<ElementOutput>::value,
          ElementAccumulator,
          ElementComputeEpilogue>;

  using Gemm = cutlass::gemm::device::Gemm<
      int8_t,
      LayoutInputA,
      int8_t,
      LayoutInputB,
      ElementOutput,
      LayoutOutput,
      ElementAccumulator,
      cutlass::arch::OpClassTensorOp,  // using Tensor Cores
      cutlass::arch::Sm80,
      TileShape,  // threadblock tile size
      WarpShape, // warp tile size
      cutlass::gemm::GemmShape<16, 8, 32>,  // instruction tile size (Tensor Core)
      EpilogueOp,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
      kStages>;

  cutlass::gemm::GemmCoord problem_size(M, N, K);

  cutlass::MatrixCoord input_size(M, K);
  cutlass::MatrixCoord weight_size(K, N);
  cutlass::MatrixCoord output_size(M, N);

  cutlass::TensorRef<ElementInputA, LayoutInputA> input_ref(
      reinterpret_cast<ElementInputA*>(input.data_ptr<int8_t>()),
      LayoutInputA::packed(input_size));

  // NOTE: you are interpreting weight (N, K) row-major as (K, N) col-major
  // via the layout; this is mathematically consistent but subtle.
  cutlass::TensorRef<ElementInputB, LayoutInputB> weight_ref(
      reinterpret_cast<ElementInputB*>(weight.data_ptr<int8_t>()),
      LayoutInputB::packed(weight_size));

  cutlass::TensorRef<ElementOutput, LayoutOutput> out_ref(
      reinterpret_cast<ElementOutput*>(out.data_ptr<torch::BFloat16>()),
      LayoutOutput::packed(output_size));

  typename Gemm::Arguments arguments{
      problem_size,
      input_ref,
      weight_ref,
      out_ref,
      out_ref,
      {alpha, 0.0f},
      1};

  Gemm gemm_op;

  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  cutlass::Status status = gemm_op.can_implement(arguments);
  TORCH_CHECK(status == cutlass::Status::kSuccess,
              "CUTLASS GEMM configuration not supported");

  status = gemm_op.initialize(arguments, workspace.get());
  TORCH_CHECK(status == cutlass::Status::kSuccess,
              "CUTLASS GEMM initialization failed");

  // Use the current PyTorch CUDA stream for better integration
  auto stream = at::cuda::getCurrentCUDAStream();
  status = gemm_op(stream.stream());
  TORCH_CHECK(status == cutlass::Status::kSuccess,
              "CUTLASS GEMM execution failed");

  return out;
}

// ================================================================
// Host-side dispatcher
// ================================================================

torch::Tensor int8_matmul_host(
    torch::Tensor input,   // INT8
    torch::Tensor weight,  // INT8
    float alpha            // FP32
) {
  auto M = input.size(0);
  auto K = input.size(1);
  auto N = weight.size(0);

  if (M == 512 && N == 4096 && K == 4096) {
    using TileShape = cutlass::gemm::GemmShape<128, 128, 128>;
    using WarpShape = cutlass::gemm::GemmShape<64, 64, 128>;
    constexpr int kStages = 3;
    return int8_matmul<TileShape, WarpShape, kStages>(input, weight, alpha);
  } else if (M == 512 && N == 4096 && K == 14336) {
    using TileShape = cutlass::gemm::GemmShape<128, 128, 64>;
    using WarpShape = cutlass::gemm::GemmShape<64, 64, 64>;
    constexpr int kStages = 4;
    return int8_matmul<TileShape, WarpShape, kStages>(input, weight, alpha);
  } else if (K == 4096 && N == 4096) {
    using TileShape = cutlass::gemm::GemmShape<256, 128, 64>;
    using WarpShape = cutlass::gemm::GemmShape<64, 64, 64>;
    constexpr int kStages = 3;
    return int8_matmul<TileShape, WarpShape, kStages>(input, weight, alpha);
  } else if (M == 1024 && N == 14336 && K == 4096) {
    using TileShape = cutlass::gemm::GemmShape<128, 128, 64>;
    using WarpShape = cutlass::gemm::GemmShape<64, 64, 64>;
    constexpr int kStages = 3;
    return int8_matmul<TileShape, WarpShape, kStages>(input, weight, alpha);
  } else {
    using TileShape = cutlass::gemm::GemmShape<256, 128, 64>;
    using WarpShape = cutlass::gemm::GemmShape<64, 64, 64>;
    constexpr int kStages = 3;
    return int8_matmul<TileShape, WarpShape, kStages>(input, weight, alpha);
  }
}

template <typename TileShape, typename WarpShape, int kStages>
torch::Tensor int8_matmul_relu(
    torch::Tensor input,    // INT8 - shape (M, K)
    torch::Tensor weight,   // INT8 - shape (N, K)
    float alpha // FP32 - scalar)
) {
  TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
  TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");

  TORCH_CHECK(input.dtype() == torch::kChar,
              "input must be torch.int8 (kChar)");
  TORCH_CHECK(weight.dtype() == torch::kChar,
              "weight must be torch.int8 (kChar)");

  TORCH_CHECK(input.dim() == 2 && weight.dim() == 2,
              "input and weight must be 2D tensors");

  auto M = input.size(0);
  auto K = input.size(1);
  auto N = weight.size(0);  // as before: weight is (N, K)

  TORCH_CHECK(weight.size(1) == K,
              "weight shape must be (N, K) with same K as input");

  // Make sure we have contiguous memory in the expected layout
  input = input.contiguous();
  weight = weight.contiguous();

  auto options = torch::TensorOptions()
                     .dtype(torch::kBFloat16)
                     .device(input.device());
  auto out = torch::empty({M, N}, options);

  using ElementOutput = cutlass::bfloat16_t;
  using ElementAccumulator = int32_t;
  using ElementComputeEpilogue = float;
  using ElementInputA = int8_t;
  using ElementInputB = int8_t;

  using LayoutInputA = cutlass::layout::RowMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::RowMajor;

  // Per-channel scaling epilogu
  using EpilogueOp = cutlass::epilogue::thread::LinearCombinationRelu<
      ElementOutput,
      128 / cutlass::sizeof_bits<ElementOutput>::value,
      ElementAccumulator,
      ElementComputeEpilogue>;

  using Gemm = cutlass::gemm::device::Gemm<
      ElementInputA,
      LayoutInputA,
      ElementInputB,
      LayoutInputB,
      ElementOutput,
      LayoutOutput,
      ElementAccumulator,
      cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm80,
      TileShape,
      WarpShape,
      cutlass::gemm::GemmShape<16, 8, 32>,
      EpilogueOp,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
      kStages>;

  cutlass::gemm::GemmCoord problem_size(M, N, K);

  cutlass::MatrixCoord input_size(M, K);
  cutlass::MatrixCoord weight_size(K, N);
  cutlass::MatrixCoord output_size(M, N);

  cutlass::TensorRef<ElementInputA, LayoutInputA> input_ref(
      reinterpret_cast<ElementInputA*>(input.data_ptr<int8_t>()),
      LayoutInputA::packed(input_size));

  // weight: (N, K) row-major → reinterpret as (K, N) col-major
  cutlass::TensorRef<ElementInputB, LayoutInputB> weight_ref(
      reinterpret_cast<ElementInputB*>(weight.data_ptr<int8_t>()),
      LayoutInputB::packed(weight_size));

  cutlass::TensorRef<ElementOutput, LayoutOutput> out_ref(
      reinterpret_cast<ElementOutput*>(out.data_ptr<torch::BFloat16>()),
      LayoutOutput::packed(output_size));

  typename Gemm::Arguments arguments{
      problem_size,
      input_ref,
      weight_ref,
      out_ref,        // C (not used because beta==0)
      out_ref,        // D
      {alpha, 0.0f},
      1               // split-K or batch count, same as your original code
  };

  Gemm gemm_op;

  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  cutlass::Status status = gemm_op.can_implement(arguments);
  TORCH_CHECK(status == cutlass::Status::kSuccess,
              "CUTLASS GEMM configuration not supported");

  status = gemm_op.initialize(arguments, workspace.get());
  TORCH_CHECK(status == cutlass::Status::kSuccess,
              "CUTLASS GEMM initialization failed");

  auto stream = at::cuda::getCurrentCUDAStream();
  status = gemm_op(stream.stream());
  TORCH_CHECK(status == cutlass::Status::kSuccess,
              "CUTLASS GEMM execution failed");

  return out;
}


torch::Tensor int8_matmul_relu_host(
    torch::Tensor input,    // INT8
    torch::Tensor weight,   // INT8
    float alpha // FP32
) {
  auto M = input.size(0);
  auto K = input.size(1);
  auto N = weight.size(0);

  if (M == 512 && N == 4096 && K == 4096) {
    using TileShape = cutlass::gemm::GemmShape<128, 128, 128>;
    using WarpShape = cutlass::gemm::GemmShape<64, 64, 128>;
    constexpr int kStages = 3;
    return int8_matmul_relu<TileShape, WarpShape, kStages>(
        input, weight, alpha);
  } else {
    using TileShape = cutlass::gemm::GemmShape<256, 128, 64>;
    using WarpShape = cutlass::gemm::GemmShape<64, 64, 64>;
    constexpr int kStages = 3;
    return int8_matmul_relu<TileShape, WarpShape, kStages>(
        input, weight, alpha);
  }
}


// ================================================================
// My custom version with LinearCombination_Dequant
// My custom de-quantization: C = (A @ B) * scale
// The value of alpha and beta just dummy to fit the CUTLASS epilogue signature
// ================================================================
template <typename TileShape, typename WarpShape, int kStages>
torch::Tensor int8_matmul_dequant(
    torch::Tensor input,   // INT8
    torch::Tensor weight,  // INT8
    torch::Tensor scale // BFloat16
) {
  TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
  TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
  TORCH_CHECK(scale.is_cuda(), "scale must be a CUDA tensor");

  TORCH_CHECK(input.dtype() == torch::kChar,
              "input must be torch.int8 (kChar)");
  TORCH_CHECK(weight.dtype() == torch::kChar,
              "weight must be torch.int8 (kChar)");

  TORCH_CHECK(input.dim() == 2 && weight.dim() == 2,
              "input and weight must be 2D tensors");

  auto M = input.size(0);
  auto K = input.size(1);
  auto N = weight.size(0);  // as before: weight is (N, K)

  TORCH_CHECK(weight.size(1) == K,
              "weight shape must be (N, K) with same K as input");

  TORCH_CHECK(scale.size(0) == M && scale.size(1) == N, "scale shape must be (M, N)");
  TORCH_CHECK(scale.dtype() == torch::kBFloat16, "scale must be torch.bfloat16");

  // Make sure we have contiguous memory
  input = input.contiguous();
  weight = weight.contiguous();
  scale = scale.contiguous();

  auto options = torch::TensorOptions()
                     .dtype(torch::kBFloat16)
                     .device(input.device());
  auto out = torch::empty({M, N}, options);

  using ElementOutput = cutlass::bfloat16_t;
  using ElementAccumulator = int32_t;
  using ElementComputeEpilogue = float; 
  using ElementInputA = int8_t;
  using ElementInputB = int8_t;

  using LayoutInputA = cutlass::layout::RowMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::RowMajor;

  using EpilogueOp = cutlass::epilogue::thread::LinearCombination_Dequant<
      ElementOutput,
      128 / cutlass::sizeof_bits<ElementOutput>::value,
      ElementAccumulator,
      ElementComputeEpilogue>;

  using Gemm = cutlass::gemm::device::Gemm<
      ElementInputA,
      LayoutInputA,
      ElementInputB,
      LayoutInputB,
      ElementOutput,
      LayoutOutput,
      ElementAccumulator,
      cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm80,
      TileShape,
      WarpShape,
      cutlass::gemm::GemmShape<16, 8, 32>,
      EpilogueOp,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
      kStages>;

  cutlass::gemm::GemmCoord problem_size(M, N, K);

  cutlass::MatrixCoord input_size(M, K);
  cutlass::MatrixCoord weight_size(K, N);
  cutlass::MatrixCoord output_size(M, N);

  cutlass::TensorRef<ElementInputA, LayoutInputA> input_ref(
      reinterpret_cast<ElementInputA*>(input.data_ptr<int8_t>()),
      LayoutInputA::packed(input_size));

  cutlass::TensorRef<ElementInputB, LayoutInputB> weight_ref(
      reinterpret_cast<ElementInputB*>(weight.data_ptr<int8_t>()),
      LayoutInputB::packed(weight_size));

  cutlass::TensorRef<ElementOutput, LayoutOutput> out_ref(
      reinterpret_cast<ElementOutput*>(out.data_ptr<torch::BFloat16>()),
      LayoutOutput::packed(output_size));

  // scale tensor ref
  cutlass::TensorRef<ElementOutput, LayoutOutput> scale_ref(
    reinterpret_cast<ElementOutput*>(scale.data_ptr<torch::BFloat16>()),
    LayoutOutput::packed(output_size));

  typename Gemm::Arguments arguments{
      problem_size,  
      input_ref,  // A
      weight_ref,  // B
      scale_ref,        // C 
      out_ref,        // D
      {1.0, 1.0},  // epilogue parameters
      1
  };

  Gemm gemm_op;

  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  cutlass::Status status = gemm_op.can_implement(arguments);
  TORCH_CHECK(status == cutlass::Status::kSuccess,
              "CUTLASS GEMM configuration not supported");

  status = gemm_op.initialize(arguments, workspace.get());
  TORCH_CHECK(status == cutlass::Status::kSuccess,
              "CUTLASS GEMM initialization failed");

  auto stream = at::cuda::getCurrentCUDAStream();
  status = gemm_op(stream.stream());
  TORCH_CHECK(status == cutlass::Status::kSuccess,
              "CUTLASS GEMM execution failed");

  return out;
}

torch::Tensor int8_matmul_dequant_host(
    torch::Tensor input,    // INT8
    torch::Tensor weight,   // INT8
    torch::Tensor scale // BFloat16
) {
  auto M = input.size(0);
  auto K = input.size(1);
  auto N = weight.size(0);

  if (M == 512 && N == 4096 && K == 4096) {
    using TileShape = cutlass::gemm::GemmShape<128, 128, 128>;
    using WarpShape = cutlass::gemm::GemmShape<64, 64, 128>;
    constexpr int kStages = 3;
    return int8_matmul_dequant<TileShape, WarpShape, kStages>(input, weight, scale);
  } else if (M == 512 && N == 4096 && K == 14336) {
    using TileShape = cutlass::gemm::GemmShape<128, 128, 64>;
    using WarpShape = cutlass::gemm::GemmShape<64, 64, 64>;
    constexpr int kStages = 4;
    return int8_matmul_dequant<TileShape, WarpShape, kStages>(input, weight, scale);
  } else if (K == 4096 && N == 4096) {
    using TileShape = cutlass::gemm::GemmShape<256, 128, 64>;
    using WarpShape = cutlass::gemm::GemmShape<64, 64, 64>;
    constexpr int kStages = 3;
    return int8_matmul_dequant<TileShape, WarpShape, kStages>(input, weight, scale);
  } else if (M == 1024 && N == 14336 && K == 4096) {
    using TileShape = cutlass::gemm::GemmShape<128, 128, 64>;
    using WarpShape = cutlass::gemm::GemmShape<64, 64, 64>;
    constexpr int kStages = 3;
    return int8_matmul_dequant<TileShape, WarpShape, kStages>(input, weight, scale);
  } else {
    using TileShape = cutlass::gemm::GemmShape<256, 128, 64>;
    using WarpShape = cutlass::gemm::GemmShape<64, 64, 64>;
    constexpr int kStages = 3;
    return int8_matmul_dequant<TileShape, WarpShape, kStages>(input, weight, scale);
  }
}


// ================================================================
// PyBind entry point
// ================================================================

torch::Tensor func_int8_matmul(
    torch::Tensor input,   // INT8
    torch::Tensor weight,  // INT8
    double alpha           // FP32 from Python, cast to float
) {
  const at::cuda::OptionalCUDAGuard device_guard(input.device());
  return int8_matmul_host(input, weight, static_cast<float>(alpha));
}

torch::Tensor func_int8_matmul_relu(
    torch::Tensor input,    // INT8
    torch::Tensor weight,   // INT8
    double alpha // FP32
) {
  const at::cuda::OptionalCUDAGuard device_guard(input.device());
  return int8_matmul_relu_host(input, weight, static_cast<float>(alpha));
}

torch::Tensor func_int8_matmul_dequant(
    torch::Tensor input,    // INT8
    torch::Tensor weight,   // INT8
    torch::Tensor scale // FP32
) {
  const at::cuda::OptionalCUDAGuard device_guard(input.device());
  return int8_matmul_dequant_host(input, weight, scale);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("func_int8_matmul",
        &func_int8_matmul,
        "INT8 GEMM with CUTLASS (bfloat16 output)");

  m.def("func_int8_matmul_relu",
        &func_int8_matmul_relu,
        "INT8 GEMM with CUTLASS and relu (bfloat16 output)");

  m.def("func_int8_matmul_dequant",
        &func_int8_matmul_dequant,
        "INT8 GEMM with CUTLASS my custom (bfloat16 output)");
}
