"""
Description: Benchmarking int8 matrix multiplication of the gemm_int8 library with dequantization
"""

import os 
import torch
import gemm_int8    
from torch.profiler import profile, record_function, ProfilerActivity


def quantize_row_int8_symmetric(mat: torch.Tensor):
    """
    Symmetric int8 quantization per row.
    mat: (N, M) float tensor
    Returns:
      q_mat: (N, M) int8
      scales: (N,) float32
    """
    qmin, qmax = -128, 127
    
    max_vals = mat.abs().amax(dim=1, keepdim=True)  # (N, 1)
    max_vals = max_vals.clamp(min=1e-8)

    scales = (max_vals / qmax).squeeze(1)          # (N,)
    q_mat = torch.clamp(torch.round(mat / scales.unsqueeze(1)), qmin, qmax).to(torch.int8)

    return q_mat, scales.to(torch.float32)

func_quantize_row_int8_symmetric = torch.compile(quantize_row_int8_symmetric)


device = 'cuda'
dtype = torch.bfloat16
input_dims = 1024
hidden_dims = 1024 * 8
output_dims = 1024 * 8

X = torch.randn(input_dims, hidden_dims, device=device, dtype=dtype)
W = torch.randn(output_dims, hidden_dims, device=device, dtype=dtype)

X_q, X_scales = func_quantize_row_int8_symmetric(X)
W_q, W_scales = func_quantize_row_int8_symmetric(W)


with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
    with_stack=False,
) as prof:
    # Run multiple iterations to get stable profiling results
    for _ in range(10):
        with record_function("int8_matmul_dequant"):
            Y = gemm_int8.int8_matmul_and_dequantize(X_q, W_q, 1.0, X_scales, W_scales)

    
print(prof.key_averages().table(sort_by="cuda_time_total" if device == "cuda" else "cpu_time_total", row_limit=20))

prof.export_chrome_trace("trace.json")
