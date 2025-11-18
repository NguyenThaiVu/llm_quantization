import torch 
import gemm_int8

def quantized_column_int_symmetric(mat:torch.Tensor):
    """
    Symmetric quantization to int8 on a per-column basis.
    mat: input float tensor (e.g., torch.float32 or torch.bfloat16)
    """
    qmin, qmax = -128, 127
    
    max_vals, _ = torch.max(torch.abs(mat), dim=0, keepdim=True)  # shape (1, M)
    scales = (max_vals / qmax).squeeze(0)  # shape (M,)
    
    q_mat = torch.clamp(torch.round(mat / scales.unsqueeze(0)), qmin, qmax).to(torch.int8)  # shape (N, M)
    
    scales = scales.clone().detach().to(torch.float32)
    return q_mat, scales

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


def quantize_row_matrix_int8_symmetric_batched(mat: torch.Tensor):
    """
    Symmetric per-row quantization for batched 3D tensor.
    mat: [B, N, D]  (float tensor)
    
    Returns:
        q_mat:   [B, N, D] int8
        scales:  [B, N]    float32  (scale per row within each batch)
    """
    qmin, qmax = -128, 127

    # Compute max abs per row (per batch) - Result shape: [B, N, 1]
    max_vals, _ = torch.max(torch.abs(mat), dim=2, keepdim=True)

    # Compute scales per row
    scales = (max_vals / qmax).clamp(min=1e-12)  # avoid div-by-zero, shape [B, N, 1]

    # Quantize
    q_mat = torch.clamp(torch.round(mat / scales), qmin, qmax).to(torch.int8)

    # Return float scales of shape [B, N]
    scales = scales.squeeze(2).to(torch.float32)
    return q_mat, scales


# ===================== Main Benchmarking Code =====================
input_dim = 1024
hidden_dim = 1024
output_dim = 1024 * 4
device = 'cuda'
dtype = torch.bfloat16
quantization_error_tolerance = 10.0  # acceptable error tolerance

# ========= 1. Benchmark float16 matmul =========
X = torch.randn(input_dim, hidden_dim, device=device, dtype=dtype)
W = torch.randn(output_dim, hidden_dim, device=device, dtype=dtype)
W_t = W.transpose(0, 1).contiguous()
print(f"Input shape: {X.shape}, Weight shape: {W.shape}")

# warm-up
for _ in range(5):
    y_fp = torch.matmul(X, W_t)
    
# Measure float16 matmul time
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)    
n_iter = 100
start_event.record()
for _ in range(n_iter):
    y_fp = torch.matmul(X, W_t)
end_event.record()
torch.cuda.synchronize()
avg_time_fp = start_event.elapsed_time(end_event) / n_iter
print(f"Average time for float16 matmul: {avg_time_fp:.2f} ms")


# ========= 2. Benchmark quantized int8 matmul =========
# X_q, X_scales = quantize_row_int8_symmetric(X)
W_q, W_scales = quantize_row_int8_symmetric(W)

# warm-up
for _ in range(5):
    X_q, X_scales = gemm_int8.quantize_row_int8_symmetric(X)
    y_int8 = gemm_int8.int8_matmul_and_quantize(X_q, W_q, 1.0, X_scales, W_scales)
    
# Measure int8 matmul time
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)    
n_iter = 100
start_event.record()
for _ in range(n_iter):
    X_q, X_scales = gemm_int8.quantize_row_int8_symmetric(X)
    y_int8 = gemm_int8.int8_matmul_and_quantize(X_q, W_q, 1.0, X_scales, W_scales)
end_event.record()
torch.cuda.synchronize()
avg_time_int8 = start_event.elapsed_time(end_event) / n_iter
print(f"Average time for quantized int8 matmul: {avg_time_int8:.2f} ms")

# print some matrix outputs for verification
print("y_fp[0, :10]:", y_fp[0, :10])
print("y_int8[0, :10]:", y_int8[0, :10])

# Check correctness
y_int8_fp = y_int8.to(dtype)
if torch.allclose(y_fp, y_int8_fp, rtol=quantization_error_tolerance, atol=quantization_error_tolerance):
    print("Pass correctness.")
else:
    print("============= Fail correctness.=============")
    max_diff = torch.max(torch.abs(y_fp - y_int8_fp)).item()
    mse = torch.mean((y_fp - y_int8_fp) ** 2).item()
    print(f"Max absolute difference: {max_diff:.6f}")
    print(f"MSE between float16 matmul and quantized int8 matmul: {mse:.6f} \n")
print('-' * 50, '\n')


# ========= 3. Benchmark batched matmul =========
batch_size = 1
X = torch.randn(batch_size, input_dim, hidden_dim, device=device, dtype=dtype)
W = torch.randn(output_dim, hidden_dim, device=device, dtype=dtype)
W_t = W.transpose(0, 1).contiguous()
print(f"Batched Input shape: {X.shape}, Weight shape: {W.shape}")

# warm-up
for _ in range(5):
    y_fp = torch.matmul(X, W_t) 
    
# Measure float16 batched matmul time
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)    
n_iter = 100
start_event.record()
for _ in range(n_iter):
    y_fp = torch.matmul(X, W_t)
end_event.record()
torch.cuda.synchronize()
avg_time_fp = start_event.elapsed_time(end_event) / n_iter
print(f"Average time for float16 batched matmul: {avg_time_fp:.2f} ms")


# ========= 4. Benchmark quantized int8 batched matmul =========
W_q, W_scales = quantize_row_int8_symmetric(W)
# X_q, X_scales = quantize_row_matrix_int8_symmetric_batched(X)

# warm-up
for _ in range(5):
    X_q, X_scales = gemm_int8.batched_quantize_row_int8_symmetric(X)
    y_int8 = gemm_int8.int8_bmm_matmul_and_quantize(
        X_q, W_q, 1.0, X_scales, W_scales)
    
# Measure int8 batched matmul time
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)    
n_iter = 100
start_event.record()
for _ in range(n_iter):
    X_q, X_scales = gemm_int8.batched_quantize_row_int8_symmetric(X)
    y_int8 = gemm_int8.int8_bmm_matmul_and_quantize(
        X_q, W_q, 1.0, X_scales, W_scales)

end_event.record()
torch.cuda.synchronize()
avg_time_int8 = start_event.elapsed_time(end_event) / n_iter
print(f"Average time for quantized int8 batched matmul: {avg_time_int8:.2f} ms")

# print some matrix outputs for verification
print("y_fp[0, 0, :10]:", y_fp[0, 0, :10])
print("y_int8[0, 0, :10]:", y_int8[0, 0, :10])

# Check correctness
y_int8_fp = y_int8.to(dtype)
if torch.allclose(y_fp, y_int8_fp, rtol=quantization_error_tolerance, atol=quantization_error_tolerance):
    print("Pass correctness.")
else:
    print("============= Fail correctness for batched matmul.=============")
    max_diff = torch.max(torch.abs(y_fp - y_int8_fp)).item()
    mse = torch.mean((y_fp - y_int8_fp) ** 2).item()
    print(f"Max absolute difference: {max_diff:.6f}")
    print(f"MSE between float16 batched matmul and quantized int8 batched matmul: {mse:.6f} \n")