#include <torch/extension.h>
#include <gemm.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
#include <vector>
#include <iostream>
#include <utility> // For std::pair

torch::Tensor int8_matmul(const torch::Tensor &A,
                          const torch::Tensor &B,
                          double alpha)
{
    float alpha_f = static_cast<float>(alpha);
    torch::checkAllContiguous("int8_matmul", {{A, "A", 0},
                                              {B, "B", 1}});
    torch::checkDeviceType("int8_matmul", {A, B}, at::DeviceType::CUDA);

    torch::checkAllSameGPU("int8_matmul", {{A, "A", 0},
                                           {B, "B", 1}});
    uint32_t M = A.size(0);
    uint32_t N = B.size(0);

    return int8_matmul_host(A, B, alpha_f);
}

torch::Tensor bmm_int8_matmul(const torch::Tensor &A,
                                  const torch::Tensor &B,
                                  double alpha)
{
    float alpha_f = static_cast<float>(alpha);
    torch::checkAllContiguous("bmm_int8_matmul", {{A, "A", 0},
                                              {B, "B", 1}});
    torch::checkDeviceType("bmm_int8_matmul", {A, B}, at::DeviceType::CUDA);

    torch::checkAllSameGPU("bmm_int8_matmul", {{A, "A", 0},
                                           {B, "B", 1}});
                                        
    return int8_bmm_matmul_host(A, B, alpha_f);
}

torch::Tensor int8_matmul_and_dequantize(const torch::Tensor &A,
                                        const torch::Tensor &B,
                                        double alpha,
                                    const torch::Tensor &row_scale,
                                    const torch::Tensor &col_scale)
{
    float alpha_f = static_cast<float>(alpha);

    torch::checkAllContiguous("int8_matmul_and_dequantize", {{A, "A", 0},
                                                            {B, "B", 1}});
    torch::checkDeviceType("int8_matmul_and_dequantize", {A, B}, at::DeviceType::CUDA);

    torch::checkAllSameGPU("int8_matmul_and_dequantize", {{A, "A", 0},
                                                         {B, "B", 1}});

    return int8_matmul_and_dequantize_host(A, B, alpha_f, row_scale, col_scale);
}

torch::Tensor int8_bmm_matmul_and_dequantize(const torch::Tensor &A,
                                        const torch::Tensor &B,
                                        double alpha,
                                    const torch::Tensor &row_scale,
                                    const torch::Tensor &col_scale)
{
    float alpha_f = static_cast<float>(alpha);

    torch::checkAllContiguous("int8_bmm_matmul_and_quantize", {{A, "A", 0},
                                                            {B, "B", 1}});
    torch::checkDeviceType("int8_bmm_matmul_and_quantize", {A, B}, at::DeviceType::CUDA);

    torch::checkAllSameGPU("int8_bmm_matmul_and_quantize", {{A, "A", 0},
                                                         {B, "B", 1}});

    return int8_matmul_and_dequantize_batched(A, B, alpha_f, row_scale, col_scale);
}



//====== pybind ======

TORCH_LIBRARY(gemm_int8_CUDA, m)
{
    m.def("int8_matmul(Tensor A, Tensor B, float alpha) -> Tensor");
    m.def("bmm_int8_matmul(Tensor A, Tensor B, float alpha) -> Tensor");
    m.def("int8_matmul_and_dequantize(Tensor A, Tensor B, float alpha, Tensor row_scale, Tensor col_scale) -> Tensor");
    m.def("int8_bmm_matmul_and_dequantize(Tensor A, Tensor B, float alpha, Tensor row_scale, Tensor col_scale) -> Tensor");
}

TORCH_LIBRARY_IMPL(gemm_int8_CUDA, CUDA, m)
{
    m.impl("int8_matmul", &int8_matmul);
    m.impl("bmm_int8_matmul", &bmm_int8_matmul);
    m.impl("int8_matmul_and_dequantize", &int8_matmul_and_dequantize);
    m.impl("int8_bmm_matmul_and_dequantize", &int8_bmm_matmul_and_dequantize);
}
