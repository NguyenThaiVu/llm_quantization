import os 
import torch
import gemm_cutlass as gemm_ext

print(f"Load gemm_cutlass extension: {gemm_ext.__file__}")
print()

def quantize_col_int8_symmetric(mat: torch.Tensor):
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

    return q_mat, scales.to(torch.float32)

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


input_dims = 2048
hidden_dims = 2048
output_dims = 4096
device = 'cuda'
dtype = torch.bfloat16

X = torch.randn((input_dims, hidden_dims), device=device, dtype=dtype)
W = torch.randn((output_dims, hidden_dims), device=device, dtype=dtype)

Y_true = torch.matmul(X, W.t())
print(f"Y_true shape: {Y_true.shape}")  # shape (input_dims, output_dims)

X_q, X_scales = quantize_row_int8_symmetric(X)
W_q, W_scales = quantize_row_int8_symmetric(W)

# Y_deq = gemm_ext.func_int8_matmul(X_q, W_q, 1.0)
# Y_deq = Y_deq.to(torch.float32)
# scale = X_scales[:, None] * W_scales[None, :]  # shape (input_dims, output_dims)
# Y_deq = Y_deq * scale  # dequantize

scale = X_scales[:, None] * W_scales[None, :]  # shape (input_dims, output_dims)
scale = scale.to(dtype)
Y_deq = gemm_ext.func_int8_matmul_custom(X_q, W_q, 1.0, scale, 1.0)

print(f"Y_deq shape: {Y_deq.shape}")  # shape (input_dims, output_dims)

if torch.allclose(Y_true.to(torch.float), Y_deq.to(torch.float), atol=5.0):
    print("Quantized matmul is TRUE.")
else:
    print("============ [ERROR] Quantized matmul is FALSE. ============")
    max_diff = torch.max(torch.abs(Y_true - Y_deq))
    print(f"Max difference: {max_diff}")
    
    mse = torch.mean((Y_true - Y_deq) ** 2).item()
    print(f"MSE: {mse}")