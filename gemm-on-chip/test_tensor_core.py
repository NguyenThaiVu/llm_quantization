"""
In this script, we test the GEMM implementation Tensor Cores, for two different input sizes.
X of shape (1, N) and X of shape (16, N).
"""

import torch
import numpy as np
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

list_hidden_dims = [1024, 2048, 2048, 4096, 4096, 8192, 12288]
list_output_dims = [1024, 2048, 4096, 4096, 8192, 8192, 12288]

for hidden_dims, output_dims in zip(list_hidden_dims, list_output_dims):
    for input_dims in [1, 16]:
        X = torch.randn(input_dims, hidden_dims, device=device, dtype=dtype)
        W = torch.randn(output_dims, hidden_dims, device=device, dtype=dtype)
        
        out_cutlass_1 = gemm_cutlass.func_matmul_tensor_core(X, W, 1.0)
        avg_time_tensor_core, std_time_tensor_core = measure_time(gemm_cutlass.func_matmul_tensor_core, X, W, 1.0)
        print(f"Input dims: {input_dims}, Tensor Core GEMM time: {avg_time_tensor_core:.3f} ms ± {std_time_tensor_core:.3f} ms")
    print()
    
    