import os 
import numpy as np
import torch
import gemm_cutlass

def measure_time(func, *args, n_warmup=10, n_repeat=100):
    # Warm-up
    for _ in range(n_warmup):
        func(*args)
    torch.cuda.synchronize()
    
    # Measure time
    list_times = []
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    for _ in range(n_repeat):
        start_event.record()
        func(*args)
        end_event.record()
        torch.cuda.synchronize()
        elapsed_time_ms = start_event.elapsed_time(end_event)
        list_times.append(elapsed_time_ms)
    
    avg_time = np.mean(list_times)
    std_time = np.std(list_times)
    return avg_time, std_time

dtype = torch.bfloat16
device = 'cuda'

input_dims = 1
hidden_dims = 4096
output_dims = 8192

X = torch.randn(input_dims, hidden_dims, device=device, dtype=dtype)
W = torch.randn(output_dims, hidden_dims, device=device, dtype=dtype)
W_t = W.transpose(0, 1).contiguous()

out_cutlass = gemm_cutlass.func_matmul_tensor_core(X, W, 1.0)
avg_time_tensor_core, std_time_tensor_core = measure_time(gemm_cutlass.func_matmul_tensor_core, X, W, 1.0)
print(f"Tensor Core GEMM time: {avg_time_tensor_core:.3f} ms ± {std_time_tensor_core:.3f} ms")

out_cutlass_cuda = gemm_cutlass.func_matmul_cuda_core(X, W, 1.0)
avg_time_cuda_core, std_time_cuda_core = measure_time(gemm_cutlass.func_matmul_cuda_core, X, W, 1.0)
print(f"CUDA Core GEMM time: {avg_time_cuda_core:.3f} ms ± {std_time_cuda_core:.3f} ms")

out_torch = torch.matmul(X, W_t)
avg_time_torch, std_time_torch = measure_time(torch.matmul, X, W_t)
print(f"PyTorch GEMM time: {avg_time_torch:.3f} ms ± {std_time_torch:.3f} ms")

print()
if torch.allclose(out_cutlass, out_torch, atol=1.0, rtol=1.0):
    print("Tensor Core Test passed!")
else:
    print("Tensor Core Test failed!")
    max_diff = torch.max(torch.abs(out_cutlass - out_torch))
    print(f"Max difference: {max_diff.item()}")
    
if torch.allclose(out_cutlass_cuda, out_torch, atol=1.0, rtol=1.0):
    print("CUDA Core Test passed!")
else:
    print("CUDA Core Test failed!")
    max_diff = torch.max(torch.abs(out_cutlass_cuda - out_torch))
    print(f"Max difference: {max_diff.item()}")
