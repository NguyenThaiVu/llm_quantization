import torch
import gemm_cutlass as gemm_ext
import numpy as np

print(f"Load gemm_cutlass extension: {gemm_ext.__file__} \n")

# M, K, N = 128, 256, 512
M, K, N = 1024, 2048, 512
d_type = torch.bfloat16

x = torch.randint(-5, 5, (M, K), dtype=torch.int8, device="cuda")
w = torch.randint(-5, 5, (N, K), dtype=torch.int8, device="cuda")
alpha = 1.0

C = []
for i in range(M * N):
    value = np.random.uniform(-1.0, 1.0)
    C.append(value)
C = torch.tensor(C, dtype=d_type, device="cuda").reshape(M, N)
beta = 1.0

# y = gemm_ext.func_int8_matmul_relu(x, w, alpha)  
y = gemm_ext.func_int8_matmul_custom(x, w, alpha, C, beta)  # bfloat16 output on GPU
print(f"Post-scaling y: {y}")

print(y.shape)  # torch.Size([512, 4096])
print(y.dtype)  # torch.bfloat16

y_true = torch.matmul(x.to(d_type), w.t().to(d_type))  # bfloat16 output on GPU
y_true = y_true * C
y_true = y_true.to(d_type)

# apply ReLU
# y_true = torch.nn.functional.relu(y_true)

print(f"Y true: {y_true}")
print(y_true.shape)
print()

assert y.dtype == y_true.dtype == d_type

if torch.allclose(y, y_true, atol=1e-2):
    print("Test passed!")
else:
    print("Test failed!")
    max_diff = torch.max(torch.abs(y - y_true))
    print(f"Max diff: {max_diff}")
    mse = torch.mean((y - y_true) ** 2).item()
    print(f"MSE: {mse}")
