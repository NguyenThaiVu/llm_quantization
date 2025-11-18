"""
Description
This script measures the performance of different quantization methods for int8 GEMM and bitsandbytes
"""

import torch
import bitsandbytes as bnb
from bitsandbytes.functional import int8_linear_matmul, int8_mm_dequant

# from torchao.kernel.intmm import int_matmul, int_scaled_matmul
from torchao.quantization.utils import quant_int8_per_token_matmul

import gemm_int8    

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


def bnb_int8_and_dequantize(x_q, w_q, x_scales, w_scales, output_dtype=torch.bfloat16):
    """
    x_q: (B, N, D) int8
    w_q: (M, D) int8
    x_scales: (B, N) float32
    w_scales: (M,) float32
    Returns:
      y: (B, N, M) float32
    """
    y_int = int8_linear_matmul(x_q, w_q)  # (B, N, M) int32
    y = int8_mm_dequant(y_int, x_scales, w_scales)
    return y.to(output_dtype)


def torch_ao_int8_and_dequantize(x_q, w_q_t, x_scales, w_scales, output_dtype=torch.bfloat16):
    """
    x_q: (N, D) int8
    w_q: (M, D) int8
    x_scales: (N,) float32
    w_scales: (M,) float32
    Returns:
      y: (N, M) float32
    """
    return quant_int8_per_token_matmul(x_q, x_scales, w_q_t, w_scales, output_dtype=output_dtype)


func_bnb_int8_and_dequantize = torch.compile(bnb_int8_and_dequantize)
func_torch_ao_int8_and_dequantize = torch.compile(torch_ao_int8_and_dequantize)
    

# ========= Main Benchmarking Code =========
device = 'cuda'
dtype = torch.bfloat16  

list_input_dim = [1024, 1024 * 2, 1024 * 2, 1024 * 4, 1024 * 4, 1024 * 8]
list_hidden_dim = [1024, 1024 * 2, 1024 * 4, 1024 * 4, 1024 * 8, 1024 * 8]
list_output_dim = [1024, 1024 * 2, 1024 * 4, 1024 * 4, 1024 * 8, 1024 * 8]

for input_dim, hidden_dim, output_dim in zip(list_input_dim, list_hidden_dim, list_output_dim):

    X = torch.randn(input_dim, hidden_dim, device=device, dtype=dtype)
    W = torch.randn(output_dim, hidden_dim, device=device, dtype=dtype)
    Y = torch.matmul(X, W.t())
    print(f"Benchmarking dequantization with input shape: {X.shape}, weight shape: {W.shape}")

    X_q, X_scales = quantize_row_int8_symmetric(X)
    W_q, W_scales = quantize_row_int8_symmetric(W)
    W_q_t = W_q.t().contiguous()

    # 1. Benchmark gemm_int8 dequantization
    # warm-up
    for _ in range(5):
        y_gemm = gemm_int8.int8_matmul_and_dequantize(X_q, W_q, 1.0, X_scales, W_scales)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)    
    start_event.record()
    n_iter = 100
    for _ in range(n_iter):
        y_gemm = gemm_int8.int8_matmul_and_dequantize(X_q, W_q, 1.0, X_scales, W_scales)
    end_event.record()
    torch.cuda.synchronize()
    avg_time = start_event.elapsed_time(end_event) / n_iter
    print(f"Average time for gemm_int8 int8_bmm_matmul_and_quantize: {avg_time:.2f} ms")

    # 2. Benchmark bitsandbytes int8 matmul + dequantization
    # warm-up
    for _ in range(5):
        # y_bnb = bnb_int8_and_dequantize(X_q, W_q, X_scales, W_scales, output_dtype=dtype)
        y_bnb = func_bnb_int8_and_dequantize(X_q, W_q, X_scales, W_scales, output_dtype=dtype)
        
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)    
    start_event.record()
    n_iter = 100
    for _ in range(n_iter):
        # y_bnb = bnb_int8_and_dequantize(X_q, W_q, X_scales, W_scales, output_dtype=dtype)    
        y_bnb = func_bnb_int8_and_dequantize(X_q, W_q, X_scales, W_scales, output_dtype=dtype)
    end_event.record()
    torch.cuda.synchronize()
    avg_time = start_event.elapsed_time(end_event) / n_iter
    print(f"Average time for bitsandbytes int8_linear_matmul + dequantization: {avg_time:.2f} ms")


    # 3. Benchmark torchao int8 matmul + dequantization
    # warm-up
    for _ in range(5):
        # y_torchao = torch_ao_int8_and_dequantize(X_q, W_q_t, X_scales, W_scales, output_dtype=dtype)
        y_torchao = func_torch_ao_int8_and_dequantize(X_q, W_q_t, X_scales, W_scales, output_dtype=dtype)
        
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)    
    start_event.record()
    n_iter = 100
    for _ in range(n_iter):
        # y_torchao = torch_ao_int8_and_dequantize(X_q, W_q_t, X_scales, W_scales, output_dtype=dtype)    
        y_torchao = func_torch_ao_int8_and_dequantize(X_q, W_q_t, X_scales, W_scales, output_dtype=dtype)
    end_event.record()
    torch.cuda.synchronize()
    avg_time = start_event.elapsed_time(end_event) / n_iter
    print(f"Average time for torchao int8_matmul + dequantization: {avg_time:.2f} ms")
    print("=============================================\n")
