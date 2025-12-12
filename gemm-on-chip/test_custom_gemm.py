"""
In this script, I will test my custom GEMM implementation using CUTLASS
"""
import numpy as np
import torch
import gemm_cutlass

dtype = torch.bfloat16

input_dims = 1
# hidden_dims = 2048
# output_dims = 2048 * 6
# scale_input = 0.5

list_input_dims = [1, 1, 1, 1, 1]
list_hidden_dims = [512, 1024, 2048, 4096, 8192]
list_output_dims = [1024, 2048, 4096, 8192, 16384]

for (input_dims, hidden_dims, output_dims) in zip(
    list_input_dims, list_hidden_dims, list_output_dims
):
    scale_input = float(np.random.rand(1))
    X = torch.randn(input_dims, hidden_dims, dtype=dtype, device='cuda')
    W = torch.randn(output_dims, hidden_dims, dtype=dtype, device='cuda')

    Y = gemm_cutlass.func_custom_matmul_tensor_core(X, W, 1.0, scale_input)

    X_new = X * scale_input
    Y_ref = torch.matmul(X_new, W.t())

    if torch.allclose(Y, Y_ref, atol=2.0):
        print("Custom CUTLASS pass!")
        print(f"Shape of output tensor: {Y.shape}")
    else:
        print("========= [ERROR] Custom CUTLASS Tensor Core GEMM failed ==========.")
        max_diff = torch.max(torch.abs(Y - Y_ref)).item()
        print(f"Max difference: {max_diff}")
    print("------------------------------------------------------")
