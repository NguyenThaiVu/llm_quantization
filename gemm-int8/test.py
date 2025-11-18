import torch
import bitsandbytes as bnb


def benchmark_functions(func, X, W, n_iter=100):
    try:
        torch.cuda.empty_cache()
        torch._dynamo.reset()
        # Warm-up
        for _ in range(5):
            y = func(X, W)
            
        # Measure cuda time
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)    
        start_event.record()
        for _ in range(n_iter):
            y = func(X, W)
        end_event.record()
        torch.cuda.synchronize()
        avg_time = start_event.elapsed_time(end_event) / n_iter
        return avg_time, y
    except Exception as e:
        print(f"[ERROR] during benchmarking: {func.__name__}")
        return float('inf'), None
    

device = 'cuda'
tolerance_error = 5.0

list_input_dim = [1024, 1024 * 2, 1024 * 4, 1024 * 4, 1024 * 8, 1024 * 8, 1024 * 10]
list_hidden_dim = [1024, 1024 * 2, 1024 * 4, 1024 * 2, 1024 * 8, 1024 * 4, 1024 * 10]
list_output_dim = [1024, 1024 * 2, 1024 * 4, 1024 * 4, 1024 * 8, 1024 * 4, 1024 * 10]
bnb_matmul_kernel = torch.compile(bnb.functional.int8_linear_matmul)

for batch_size in [1, 4, 8, 16]:
    for (input_dim, hidden_dim, output_dim) in zip(list_input_dim, list_hidden_dim, list_output_dim):
        X = torch.randint(-5, 5, (batch_size, input_dim, hidden_dim), device=device, dtype=torch.int8)
        W = torch.randint(-5, 5, (output_dim, hidden_dim), device=device, dtype=torch.int8)
        print(f"Benchmark input shapes X: {X.shape}, W: {W.shape}")

        # ========= 1. Benchmark float16 matmul =========
        X_fp = X.to(torch.float16)
        W_fp = W.to(torch.float16)
        W_fp = W_fp.transpose(0, 1).contiguous()

        torch_time, y_torch = benchmark_functions(torch.matmul, X_fp, W_fp)
        print(f"Average time for float16 matmul: {torch_time:.2f} ms")

        # ======== 2. Benchmark bitsandbytes int8 matmul ========
        bnb_time, y_bnb = benchmark_functions(bnb_matmul_kernel, X, W)
        print(f"Average time for BitsAndBytes int8 matmul: {bnb_time:.2f} ms")

        # check correctness of bitsandbytes int8 matmul
        y_bnb_fp = y_bnb.to(torch.float16)
        mse_bnb = torch.mean((y_torch - y_bnb_fp) ** 2).item()
        if mse_bnb > tolerance_error:
            print(f"[ERROR] BitsAndBytes int8 matmul MSE too high: {mse_bnb:.4f}")
        else:
            print(f"BitsAndBytes int8 matmul pass")
        
    print("-"*50)
    print()
        