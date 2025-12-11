"""
In this script, I will test my custom GEMM implementation using CUTLASS
"""

import torch
import gemm_cutlass

input_dims = 1
hidden_dims = 4096
output_dims = 8192

X = torch.randn(input_dims, hidden_dims, dtype=torch.bfloat16, device='cuda')
W = torch.randn(output_dims, hidden_dims, dtype=torch.bfloat16, device='cuda')

Y = gemm_cutlass.func_custom_matmul_tensor_core(X, W, 1.0)

X_new = X * 2.0
Y_ref = torch.matmul(X_new, W.t())

if torch.allclose(Y, Y_ref, atol=2.0):
    print("Custom CUTLASS pass!")
    print(f"Shape of output tensor: {Y.shape}")
else:
    print("========= [ERROR] Custom CUTLASS Tensor Core GEMM failed ==========.")
    max_diff = torch.max(torch.abs(Y - Y_ref)).item()
    print(f"Max difference: {max_diff}")