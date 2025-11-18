#pragma once
#include <common.h>
#include <torch/types.h>


torch::Tensor int8_matmul_host(torch::Tensor input,  // INT8
                                  torch::Tensor weight, // INT8
                                  float alpha          // FP32
);

torch::Tensor int8_bmm_matmul_host(torch::Tensor input,  // INT8
                                      torch::Tensor weight, // INT8
                                      float alpha          // FP32
);

torch::Tensor int8_matmul_and_dequantize_host(torch::Tensor input,      // INT8
                                                torch::Tensor weight,     // INT8
                                                float alpha,              // FP32
                                                torch::Tensor row_scale,  // FP32
                                                torch::Tensor col_scale   // FP32
);

torch::Tensor int8_matmul_and_dequantize_batched(
    torch::Tensor input,   // INT8 - shape (B, N, K)
    torch::Tensor weight,  // INT8 - shape (B, M, K) or (M, K)
    float alpha,          // FP32 - scalar
    torch::Tensor row_scale, // FP32 - shape (B, N)
    torch::Tensor col_scale  // FP32 - shape (B, M) or (M)
);