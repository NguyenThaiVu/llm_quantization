"""
In this file, we define benchmarks the time between
quantization and float matrix multiplication using different methods.
"""

import os 
import torch 
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


def quantize_col_int8_symmetric(mat: torch.Tensor, scale_dtype=torch.float32):
    """
    Symmetric int8 quantization per column.
    mat: (N, M) float tensor
    Returns:
      q_mat: (N, M) int8
      scales: (M,) float32
    """
    qmin, qmax = -128, 127
    
    max_vals = mat.abs().amax(dim=0, keepdim=True)  # (1, M)
    max_vals = max_vals.clamp(min=1e-8)

    scales = (max_vals / qmax).squeeze(0)          # (M,)
    q_mat = torch.clamp(torch.round(mat / scales.unsqueeze(0)), qmin, qmax).to(torch.int8)

    return q_mat, scales.to(scale_dtype)


def quantize_tensor_int8_symmetric(mat: torch.Tensor, scale_dtype=torch.float32):
    """
    Symmetric int8 quantization for the entire tensor.
    mat: (N, M) float tensor
    Returns:
      q_mat: (N, M) int8
      scale: float32
    """
    qmin, qmax = -128, 127
    
    max_val = mat.abs().amax()  # scalar
    max_val = max_val.clamp(min=1e-8)

    scale = max_val / qmax          # scalar
    q_mat = torch.clamp(torch.round(mat / scale), qmin, qmax).to(torch.int8)

    return q_mat, scale.to(scale_dtype)


func_quantize_row_int8_symmetric = torch.compile(quantize_row_int8_symmetric)
func_quantize_col_int8_symmetric = torch.compile(quantize_col_int8_symmetric)
func_quantize_tensor_int8_symmetric = torch.compile(quantize_tensor_int8_symmetric)

device = 'cuda'
dtype = torch.bfloat16
input_dims = 1024 * 4
hidden_dims = 1024 * 8
output_dims = 1024 * 6

X = torch.randn((input_dims, hidden_dims), device=device, dtype=dtype)
W = torch.randn((output_dims, hidden_dims), device=device, dtype=dtype)
W_t = W.t()


# Measure ground truth float matmul time
# Warm up
for _ in range(10):
    Y_float = torch.matmul(X, W_t)
    
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

# Benchmark float matmul
num_iters = 100
torch.cuda.synchronize()
start_event.record()
for _ in range(num_iters):
    Y_float = torch.matmul(X, W_t)
end_event.record()
torch.cuda.synchronize()
float_matmul_time_ms = start_event.elapsed_time(end_event) / num_iters
print(f"Float matmul time: {float_matmul_time_ms:.4f} ms")


# ==============================================
# 2. Quantize per-row + per-column + dequantized matmul time

# assume weight is quantized once and reused
W_q, w_scales = func_quantize_row_int8_symmetric(W, scale_dtype=dtype)

# Warm up
for _ in range(5):
    X_q, x_scales = func_quantize_row_int8_symmetric(X, scale_dtype=dtype)
    Y_deq_row = gemm_ext.func_int8_matmul_dequant(X_q, W_q, (x_scales[:, None] * w_scales[None, :]))

start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)
# Benchmark quantization + dequantized matmul
num_iters = 100
torch.cuda.synchronize()
start_event.record()
for _ in range(num_iters):
    # Quantize input on-the-fly
    X_q, x_scales = func_quantize_row_int8_symmetric(X, scale_dtype=dtype)  
    Y_deq_row = gemm_ext.func_int8_matmul_dequant(X_q, W_q, (x_scales[:, None] * w_scales[None, :]))
end_event.record()
torch.cuda.synchronize()
quant_dequant_matmul_time_ms = start_event.elapsed_time(end_event) / num_iters
print(f"Quantization + Dequantized matmul time: {quant_dequant_matmul_time_ms:.4f} ms")

quant_row_error = torch.mean((Y_float - Y_deq_row) ** 2).item()
print(f"Quantization per-row MSE: {quant_row_error}")
print("===========================\n")



# ==============================================
# 3. Quantize per-tensor + per-column + dequantized matmul time

# assume weight is quantized once and reused
W_q, w_scale = func_quantize_tensor_int8_symmetric(W, scale_dtype=dtype)

# Warm up
for _ in range(5):
    X_q, x_scales = func_quantize_row_int8_symmetric(X, scale_dtype=dtype)
    scales = x_scales[:, None].repeat(1, W_q.shape[0]) * w_scale
    Y_deq_tensor = gemm_ext.func_int8_matmul_dequant(X_q, W_q, scales)
    
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)
# Benchmark quantization + dequantized matmul
num_iters = 100
torch.cuda.synchronize()
start_event.record()
for _ in range(num_iters):  
    # Quantize input on-the-fly
    X_q, x_scales = func_quantize_row_int8_symmetric(X, scale_dtype=dtype)  
    scales = x_scales[:, None].repeat(1, W_q.shape[0]) * w_scale
    Y_deq_tensor = gemm_ext.func_int8_matmul_dequant(X_q, W_q, scales)
end_event.record()
torch.cuda.synchronize()
quant_dequant_matmul_time_ms = start_event.elapsed_time(end_event) / num_iters
print(f"Quantization + Dequantized matmul time: {quant_dequant_matmul_time_ms:.4f} ms")

quant_tensor_error = torch.mean((Y_float - Y_deq_tensor) ** 2).item()
print(f"Quantization per-tensor MSE: {quant_tensor_error}")
print("===========================\n")