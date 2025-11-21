"""
Description: Benchmarking dynamic vs static quantization.
"""
import os 
import numpy as np
import matplotlib.pyplot as plt
import torch 
import gemm_int8

def apply_row_quantization(X: torch.Tensor, scales: torch.Tensor):
    """
    Apply row-wise quantization to a float tensor X using provided scales.
    X: (N, M) float tensor
    scales: (N,) float32
    Returns:
      X_q: (N, M) int8
    """
    qmin, qmax = -128, 127
    X_q = torch.clamp(torch.round(X / scales.unsqueeze(1)), qmin, qmax).to(torch.int8)
    return X_q

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


device = 'cuda'
dtype = torch.bfloat16

list_input_dim = [1024, 1024, 2048, 2048, 4096, 4096, 8192]
list_hidden_dim = [1024, 2048, 2048, 4096, 4096, 8192, 8192]

list_time_dynamic = []
list_time_static = []

for input_dims, hidden_dims in zip(list_input_dim, list_hidden_dim):
    print(f"Benchmarking GEMM with Input Dim: {input_dims}, Hidden Dim: {hidden_dims}")

    X = torch.randn(input_dims, hidden_dims, device=device, dtype=dtype)

    # 1. Dynamic quantization
    # warm-up
    torch._dynamo.reset()
    func_quantize_row_int8_symmetric = torch.compile(quantize_row_int8_symmetric)
    for _ in range(5):
        X_q_dynamic, X_scales = func_quantize_row_int8_symmetric(X)
        
    # actual run
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(100):
        X_q_dynamic, X_scales = func_quantize_row_int8_symmetric(X)
    end_event.record()
    torch.cuda.synchronize()
    elapsed_time_ms = start_event.elapsed_time(end_event)
    avg_time_ms = elapsed_time_ms / 50
    print(f"Dynamic Quantization GEMM Time: {avg_time_ms:.3f} ms")
    list_time_dynamic.append(avg_time_ms)

    # Static quantization
    # warm-up
    torch._dynamo.reset()
    func_apply_row_quantization = torch.compile(apply_row_quantization)
    for _ in range(5):
        X_q_static = func_apply_row_quantization(X, X_scales)
        
    # actual run
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(100):
        X_q_static = func_apply_row_quantization(X, X_scales)
    end_event.record()
    torch.cuda.synchronize()
    elapsed_time_ms = start_event.elapsed_time(end_event)
    avg_time_ms = elapsed_time_ms / 50
    print(f"Static Quantization GEMM Time: {avg_time_ms:.3f} ms")
    list_time_static.append(avg_time_ms)
    
    # check correctness
    if torch.allclose(X_q_dynamic.to(torch.float32), X_q_static.to(torch.float32), atol=1e-3):
        # print("Dynamic and Static quantization match!")
        pass
    else:
        print("[ERROR] ====== Mismatch in Dynamic and Static quantization results! ======")

# Visualize and save the results
plt.figure(figsize=(8, 6))
x_labels = [f"{in_dim} x {hid_dim}" for in_dim, hid_dim in zip(list_input_dim, list_hidden_dim)]
plt.rcParams.update({'font.size': 14})
plt.plot(x_labels, list_time_dynamic, marker='o', label='Dynamic Quantization')
plt.plot(x_labels, list_time_static, marker='s', label='Static Quantization')
plt.xlabel('Input Dimension')
plt.ylabel('Average Time (ms)')
plt.xticks(rotation=45)
plt.title('Dynamic vs Static Quantization Time')
plt.legend()
plt.grid(True)
plt.tight_layout()

output_dir = "benchmark_results"
os.makedirs(output_dir, exist_ok=True)
plt.savefig(os.path.join(output_dir, 'dynamic_vs_static_quantization_gemm_time.png'))
plt.show()

# Save the raw data to a text file
with open(os.path.join(output_dir, 'dynamic_vs_static_quantization_gemm_time.txt'), 'w') as f:
    f.write("Input_Dim\tDynamic_Time_ms\tStatic_Time_ms\n")
    for in_dim, dyn_time, stat_time in zip(x_labels, list_time_dynamic, list_time_static):
        f.write(f"{in_dim}\t{dyn_time:.6f}\t{stat_time:.6f}\n")

