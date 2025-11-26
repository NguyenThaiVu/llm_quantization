"""
Description
This script measures the performance of different quantization methods for int8 GEMM and bitsandbytes
"""

import os 
import matplotlib.pyplot as plt

import torch
import bitsandbytes as bnb
from bitsandbytes.functional import int8_linear_matmul, int8_mm_dequant
from torchao.quantization.utils import quant_int8_per_token_matmul

import gemm_cutlass as gemm_ext

def quantize_row_int8_symmetric(mat: torch.Tensor, scale_dtype=torch.float32):
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

    return q_mat, scales.to(scale_dtype)


def bnb_int8_and_dequantize(x_q, w_q, x_scales, w_scales, output_dtype=torch.bfloat16):
    """
    x_q: (N, D) int8
    w_q: (M, D) int8
    x_scales: (N) float32
    w_scales: (M,) float32
    Returns:
      y: (B, N, M) float32
    """
    y_int = int8_linear_matmul(x_q, w_q)  # (B, N, M) int32
    y = int8_mm_dequant(y_int, x_scales, w_scales)
    return y


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


IS_TORCH_COMPILE = True

if IS_TORCH_COMPILE:
    func_bnb_int8_and_dequantize = torch.compile(bnb_int8_and_dequantize)
    func_torch_ao_int8_and_dequantize = torch.compile(torch_ao_int8_and_dequantize)
else:
    func_bnb_int8_and_dequantize = bnb_int8_and_dequantize
    func_torch_ao_int8_and_dequantize = torch_ao_int8_and_dequantize
    

# ========= Main Benchmarking Code =========
device = 'cuda'
dtype = torch.bfloat16  

list_input_dim = [1024, 1024 * 2, 1024 * 2, 1024 * 4, 1024 * 4, 1024 * 8, 1024 * 8]
list_hidden_dim = [1024, 1024 * 2, 1024 * 4, 1024 * 4, 1024 * 8, 1024 * 8, 1024 * 8]
list_output_dim = [1024, 1024 * 2, 1024 * 4, 1024 * 4, 1024 * 8, 1024 * 4, 1024 * 8]

list_times_gemm = []
list_times_bnb = []
list_times_torchao = []

for input_dim, hidden_dim, output_dim in zip(list_input_dim, list_hidden_dim, list_output_dim):

    X = torch.randn(input_dim, hidden_dim, device=device, dtype=dtype)
    W = torch.randn(output_dim, hidden_dim, device=device, dtype=dtype)
    Y = torch.matmul(X, W.t())
    print(f"Benchmarking dequantization with input shape: {X.shape}, weight shape: {W.shape}")

    # Quantize inputs and weights before benchmarking
    # This step is computed offline and not included in the benchmarking time
    X_q, X_scales = quantize_row_int8_symmetric(X)
    W_q, W_scales = quantize_row_int8_symmetric(W)
    scale = (X_scales[:, None] * W_scales[None, :]).to(dtype)  # shape (input_dims, output_dims)
    W_q_t = W_q.t().contiguous()

    # 1. Benchmark gemm_int8 dequantization
    # warm-up
    for _ in range(5):
        y_gemm = gemm_ext.func_int8_matmul_dequant(X_q, W_q, scale)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)    
    start_event.record()
    n_iter = 100
    for _ in range(n_iter):
        y_gemm = gemm_ext.func_int8_matmul_dequant(X_q, W_q, scale)
        if y_gemm.dtype != dtype:
            y_gemm = y_gemm.to(dtype)
    end_event.record()
    torch.cuda.synchronize()
    avg_time = start_event.elapsed_time(end_event) / n_iter
    print(f"Average time for gemm_int8 int8_bmm_matmul_and_quantize: {avg_time:.2f} ms")
    list_times_gemm.append(avg_time)

    # 2. Benchmark bitsandbytes int8 matmul + dequantization
    # warm-up
    for _ in range(5):
        y_bnb = func_bnb_int8_and_dequantize(X_q, W_q, X_scales, W_scales, output_dtype=dtype)
        
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)    
    start_event.record()
    n_iter = 100
    for _ in range(n_iter):
        y_bnb = func_bnb_int8_and_dequantize(X_q, W_q, X_scales, W_scales, output_dtype=dtype)
        if y_bnb.dtype != dtype:
            y_bnb = y_bnb.to(dtype)
    end_event.record()
    torch.cuda.synchronize()
    avg_time = start_event.elapsed_time(end_event) / n_iter
    print(f"Average time for bitsandbytes int8_linear_matmul + dequantization: {avg_time:.2f} ms")
    list_times_bnb.append(avg_time)


    # 3. Benchmark torchao int8 matmul + dequantization
    # warm-up
    for _ in range(5):
        y_torchao = func_torch_ao_int8_and_dequantize(X_q, W_q_t, X_scales, W_scales, output_dtype=dtype)
        
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)    
    start_event.record()
    n_iter = 100
    for _ in range(n_iter):
        y_torchao = func_torch_ao_int8_and_dequantize(X_q, W_q_t, X_scales, W_scales, output_dtype=dtype)
        if y_torchao.dtype != dtype:
            y_torchao = y_torchao.to(dtype)
    end_event.record()
    torch.cuda.synchronize()
    avg_time = start_event.elapsed_time(end_event) / n_iter
    print(f"Average time for torchao int8_matmul + dequantization: {avg_time:.2f} ms")
    list_times_torchao.append(avg_time)
    
    print("=============================================\n")
    
# ========== Plotting Results ==========
plt.figure(figsize=(6, 6))
x_labels = [f"{in_dim} x {hid_dim} x {out_dim}" for in_dim, hid_dim, out_dim in zip(list_input_dim, list_hidden_dim, list_output_dim)]
plt.rcParams.update({'font.size': 14})
plt.plot(x_labels, list_times_gemm, marker='o', label='gemm_int8', linewidth=4)
plt.plot(x_labels, list_times_bnb, marker='*', label='bitsandbytes', linewidth=2)
plt.plot(x_labels, list_times_torchao, marker='+', label='torchao', linewidth=2)
plt.xlabel('Matrix Dimensions (Input x Hidden x Output)')
plt.ylabel('Average Time (ms)')
plt.title('Int8 Matmul + Dequantization Performance Comparison')
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()

if IS_TORCH_COMPILE:
    plt.savefig(f'dequantization_performance_comparison_torch_compile.png')
else:
    plt.savefig(f'dequantization_performance_comparison.png')
