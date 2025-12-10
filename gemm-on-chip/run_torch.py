import torch
import gemm_cutlass
from bitsandbytes.functional import int8_linear_matmul


dtype = torch.bfloat16

list_input_dim = [1024, 1024 * 2, 1024 * 2, 1024 * 4, 1024 * 4, 1024 * 8, 1024 * 8]
list_hidden_dim = [1024, 1024 * 2, 1024 * 4, 1024 * 4, 1024 * 8, 1024 * 8, 1024 * 4]
list_output_dim = [1024, 1024 * 2, 1024 * 4, 1024 * 4, 1024 * 8, 1024 * 8, 1024 * 4]

for (input_dim, hidden_dim, output_dim) in zip(
    list_input_dim, list_hidden_dim, list_output_dim
):
    print(
        f"Running GEMM with dimensions: Input Dim = {input_dim}, Hidden Dim = {hidden_dim}, Output Dim = {output_dim}"
    )
    X = torch.randint(-5, 5, (input_dim, hidden_dim), dtype=torch.int8).cuda()
    W = torch.randint(-5, 5, (output_dim, hidden_dim), dtype=torch.int8).cuda()

    X_fp = X.to(dtype)
    W_fp = W.to(dtype)
    W_fp_T = W_fp.t()

    # Measure Torch GEMM performance
    # warm up
    for _ in range(10):
        Y_torch = torch.matmul(X_fp, W_fp_T)
        
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    n_iter = 100
    for _ in range(n_iter):
        Y_torch = torch.matmul(X_fp, W_fp_T)
    end_event.record()
    torch.cuda.synchronize()
    torch_time = start_event.elapsed_time(end_event) / n_iter
    print(f"Torch GEMM time: {torch_time:.3f} ms")

    # Measure GemmInt8 performance
    # warm up
    for _ in range(10):
        Y_gemm = gemm_cutlass.func_int8_matmul(X, W, 1.0)

    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(n_iter):
        Y_gemm = gemm_cutlass.func_int8_matmul(X, W, 1.0)
    end_event.record()
    torch.cuda.synchronize()
    gemm_time = start_event.elapsed_time(end_event) / n_iter
    print(f"GemmInt8 GEMM time: {gemm_time:.3f} ms")

    # Verify correctness   
    if Y_gemm.dtype != dtype:
        Y_gemm = Y_gemm.to(dtype)
    if torch.allclose(Y_torch, Y_gemm, atol=1.0):
        print("Results match!")
    else:
        print("===== [ERROR] Results do not match! =====")
        max_diff = torch.max(torch.abs(Y_torch - Y_gemm))
        print(f"Max difference: {max_diff.item()}")
    
    
    # Measure bitsandbytes int8_linear_matmul performance
    # warm up
    for _ in range(10):
        Y_bnb = int8_linear_matmul(X, W)
        
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(n_iter):
        Y_bnb = int8_linear_matmul(X, W)
    end_event.record()
    torch.cuda.synchronize()
    bnb_time = start_event.elapsed_time(end_event) / n_iter
    print(f"bitsandbytes int8_linear_matmul GEMM time: {bnb_time:.3f} ms")
    
    # Verify correctness   
    if Y_bnb.dtype != dtype:
        Y_bnb = Y_bnb.to(dtype)
    if torch.allclose(Y_torch, Y_bnb, atol=1.0):
        print("Bitsandbytes match!")
    else:
        print("===== [ERROR] Bitsandbytes Results do not match! =====")
        max_diff = torch.max(torch.abs(Y_torch - Y_bnb))
        print(f"Max difference: {max_diff.item()}")


    print("-" * 50)
    print()


