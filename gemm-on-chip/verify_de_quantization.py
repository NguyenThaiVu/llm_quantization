import os 
import torch
import gemm_cutlass as gemm_ext

print(f"Load gemm_cutlass extension: {gemm_ext.__file__}")
print()

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


device = 'cuda'
dtype = torch.bfloat16

# input_dims = 2048
# hidden_dims = 2048
# output_dims = 4096
quantized_threshold_error = 10.0

list_input_dims = [512, 1024, 2048, 2048, 4096, 4096, 8192]
list_hidden_dims = [512, 1024, 2048, 4096, 4096, 4096, 8192]
list_output_dims = [512, 1024, 2048, 4096, 4096, 8192, 8192]

for (input_dims, hidden_dims, output_dims) in zip(
    list_input_dims, list_hidden_dims, list_output_dims
):
    print(f"\nTesting quantized matmul with dimensions: {input_dims} x {hidden_dims} x {output_dims}")

    X = torch.randn((input_dims, hidden_dims), device=device, dtype=dtype)
    W = torch.randn((output_dims, hidden_dims), device=device, dtype=dtype)

    Y_true = torch.matmul(X, W.t())

    X_q, X_scales = quantize_row_int8_symmetric(X, scale_dtype=dtype)
    W_q, W_scales = quantize_row_int8_symmetric(W, scale_dtype=dtype)
    scale = X_scales[:, None] * W_scales[None, :]  # shape (input_dims, output_dims)

    Y_deq = gemm_ext.func_int8_matmul_dequant(X_q, W_q, scale)

    if torch.allclose(Y_true.to(torch.float), Y_deq.to(torch.float), atol=quantized_threshold_error):
        print("Quantized matmul is TRUE.")
    else:
        print("============ [ERROR] Quantized matmul is FALSE. ============")
        
        mse = torch.mean((Y_true - Y_deq) ** 2).item()
        print(f"MSE: {mse}")