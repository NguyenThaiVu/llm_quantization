import time
import torch
import gemm_int8

a = torch.randint(-128, 127, (1024, 1024 * 8), device='cuda', dtype=torch.int8)
b = torch.randint(-128, 127, (1024 * 8, 1024 * 8), device='cuda', dtype=torch.int8)

a_fp = a.to(torch.float16)
b_fp = b.to(torch.float16).t()

# Measure time for torch.matmul
# Warm up
for _ in range(10):
    y_torch = torch.matmul(a_fp, b_fp)

n_iter = 100
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

start_event.record()
for _ in range(n_iter):
    y_torch = torch.matmul(a_fp, b_fp)
end_event.record()
torch.cuda.synchronize()
avg_time = start_event.elapsed_time(end_event) / n_iter
print(f"Average time for torch.matmul over {n_iter} iterations: {avg_time:.2f} ms")


@torch.compile(dynamic=True)
def test_gemm_int8(x, y, alpha):
    return gemm_int8.matmul(x, y, alpha)

# Measure time 
# Warm up
for _ in range(10):
    y_int8 = test_gemm_int8(a, b, 1.0)

print("Done compiled and warmed up - starting timing...")

start_event.record()
for _ in range(n_iter):
    y_int8 = test_gemm_int8(a, b, 1.0)
end_event.record()
torch.cuda.synchronize()
avg_time_int8 = start_event.elapsed_time(end_event) / n_iter
print(f"Average time for gemm_int8.matmul over {n_iter} iterations: {avg_time_int8:.2f} ms")


