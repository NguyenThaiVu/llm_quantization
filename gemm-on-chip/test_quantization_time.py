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

    scale = (max_val / qmax)          # scalar
    q_mat = torch.clamp(torch.round(mat / scale), qmin, qmax).to(torch.int8)

    return q_mat, scale.to(scale_dtype)


device = 'cuda'
dtype = torch.bfloat16
input_dims = 1024 * 4
hidden_dims = 1024 * 8
output_dims = 1024 * 6

X = torch.randn((input_dims, hidden_dims), device=device, dtype=dtype)
W = torch.randn((output_dims, hidden_dims), device=device, dtype=dtype)
Y_true = torch.matmul(X, W.t())

# ==============================================
# 1. Quantization per-tensor 
X_q_tensor, scale_x_tensor = quantize_tensor_int8_symmetric(X, scale_dtype=dtype)
W_q_tensor, scale_w_tensor = quantize_tensor_int8_symmetric(W, scale_dtype=dtype)
scale_tensor = scale_x_tensor * scale_w_tensor  # scalar

torch._dynamo.reset()  # reset any previous dynamo state
# Measure time
# Warm up
for i in range(5):
    Y_deq_tensor = gemm_ext.func_int8_matmul(X_q_tensor, W_q_tensor, scale_tensor)
    
n_runs = 100
times_ms = []

for _ in range(n_runs):
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    Y_deq_tensor = gemm_ext.func_int8_matmul(X_q_tensor, W_q_tensor, scale_tensor)
    # Y_deq_tensor = Y_q_tensor.to(dtype) * scale_tensor
    end_event.record()
    torch.cuda.synchronize()

    elapsed_time = start_event.elapsed_time(end_event)  # ms for `iters` runs
    times_ms.append(elapsed_time)  # per-iteration time

times_t = torch.tensor(times_ms)
avg_time = times_t.mean().item()
std_time = times_t.std().item()  
print(f'Per-iteration time: avg = {avg_time:.4f} ± {std_time:.4f} ms')

quant_tensor_error = torch.mean((Y_true - Y_deq_tensor) ** 2).item()
print(f"Quantization per-tensor MSE: {quant_tensor_error}")
print("===========================\n")

# ==============================================
# 2. Quantization per-row
X_q_row, scales_x_row = quantize_row_int8_symmetric(X, scale_dtype=dtype)
W_q_row, scales_w_row = quantize_row_int8_symmetric(W, scale_dtype=dtype)
scale_matrix = scales_x_row[:, None] * scales_w_row[None, :]

torch._dynamo.reset()  # reset any previous dynamo state
# Measure time
# Warm up
for i in range(5):
    Y_q = gemm_ext.func_int8_matmul(X_q_row, W_q_row, 1.0)
    Y_deq = Y_q.to(dtype) * scale_matrix
    
n_runs = 100
times_ms = []

for _ in range(n_runs):
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)  
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    Y_q = gemm_ext.func_int8_matmul(X_q_row, W_q_row, 1.0)
    Y_deq = Y_q.to(dtype) * scale_matrix
    end_event.record()
    torch.cuda.synchronize()
    
    elapsed_time = start_event.elapsed_time(end_event)  
    times_ms.append(elapsed_time)  # per-iteration time

times_t = torch.tensor(times_ms)
avg_time = times_t.mean().item()
std_time = times_t.std().item()  
print(f'Per-iteration time: avg = {avg_time:.4f} ± {std_time:.4f} ms')

quant_row_error = torch.mean((Y_true - Y_deq) ** 2).item()
print(f"Quantization per-row MSE: {quant_row_error}")
print("===========================\n")


# ==============================================
# 3. My fusion quantization per-row with dequantization in kernel
X_q_row, scales_x_row = quantize_row_int8_symmetric(X, scale_dtype=dtype)
W_q_row, scales_w_row = quantize_row_int8_symmetric(W, scale_dtype=dtype)
scale_matrix = scales_x_row[:, None] * scales_w_row[None, :]

torch._dynamo.reset()  # reset any previous dynamo state
# Measure time
# Warm up
for i in range(5):
    Y_deq_fused = gemm_ext.func_int8_matmul_dequant(X_q_row, W_q_row, scale_matrix)
    
n_runs = 100
times_ms = []
for _ in range(n_runs):
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)  
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    Y_deq_fused = gemm_ext.func_int8_matmul_dequant(X_q_row, W_q_row, scale_matrix)
    end_event.record()
    torch.cuda.synchronize()
    
    elapsed_time = start_event.elapsed_time(end_event)  
    times_ms.append(elapsed_time)  # per-iteration time
times_t = torch.tensor(times_ms)
avg_time = times_t.mean().item()
std_time = times_t.std().item()  
print(f'Per-iteration time: avg = {avg_time:.4f} ± {std_time:.4f} ms')

quant_row_fused_error = torch.mean((Y_true - Y_deq_fused) ** 2).item()
print(f"Fused quantization per-row MSE: {quant_row_fused_error}")