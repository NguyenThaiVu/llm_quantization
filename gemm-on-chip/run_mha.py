import os 
import torch
import gemm_cutlass

def mha_torch(X_fp, W_q_fp, W_k_fp, W_v_fp, W_o_fp):
    Q = torch.matmul(X_fp, W_q_fp.t())
    K = torch.matmul(X_fp, W_k_fp.t())
    V = torch.matmul(X_fp, W_v_fp.t())

    attn_scores = torch.matmul(Q, K.t()) / (K.size(-1) ** 0.5)
    attn_probs = torch.softmax(attn_scores, dim=-1)

    context = torch.matmul(attn_probs, V)
    output = torch.matmul(context, W_o_fp.t())
    if output.dtype != X_fp.dtype:
        output = output.to(X_fp.dtype)
    return output

def mha_gemm_int8(X, W_q, W_k, W_v, W_o_fp, dtype):
    Q_fp = gemm_cutlass.func_int8_matmul(X, W_q, 1.0)
    K_fp = gemm_cutlass.func_int8_matmul(X, W_k, 1.0)
    V_fp = gemm_cutlass.func_int8_matmul(X, W_v, 1.0)

    attn_scores = torch.matmul(Q_fp, K_fp.t()) / (K_fp.size(-1) ** 0.5)
    attn_probs = torch.softmax(attn_scores, dim=-1)

    context = torch.matmul(attn_probs, V_fp)
    output = torch.matmul(context, W_o_fp.t())
    if output.dtype != dtype:
        output = output.to(dtype)
    return output
    
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
    W_q = torch.randint(-5, 5, (hidden_dim, hidden_dim), dtype=torch.int8).cuda() 
    W_k = torch.randint(-5, 5, (hidden_dim, hidden_dim), dtype=torch.int8).cuda() 
    W_v = torch.randint(-5, 5, (hidden_dim, hidden_dim), dtype=torch.int8).cuda() 
    W_o = torch.randint(-5, 5, (output_dim, hidden_dim), dtype=dtype).cuda()

    W_q_fp = W_q.to(dtype)
    W_k_fp = W_k.to(dtype)
    W_v_fp = W_v.to(dtype)
    W_o_fp = W_o.to(dtype)

    # Measure Torch MHA performance
    # warm up
    for _ in range(10):
        Y_torch = mha_torch(X.to(dtype), W_q_fp, W_k_fp, W_v_fp, W_o_fp)
        
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    n_iter = 100
    for _ in range(n_iter):
        Y_torch = mha_torch(X.to(dtype), W_q_fp, W_k_fp, W_v_fp, W_o_fp)
    end_event.record()
    torch.cuda.synchronize()
    torch_time = start_event.elapsed_time(end_event) / n_iter
    print(f"Torch MHA time: {torch_time:.3f} ms")

    # Measure GemmInt8 MHA performance
    # warm up
    for _ in range(10):
        Y_gemm = mha_gemm_int8(X, W_q, W_k, W_v, W_o_fp, dtype)
        
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(n_iter):
        Y_gemm = mha_gemm_int8(X, W_q, W_k, W_v, W_o_fp, dtype)
    end_event.record()
    torch.cuda.synchronize()
    gemm_time = start_event.elapsed_time(end_event) / n_iter
    print(f"GemmInt8 MHA time: {gemm_time:.3f} ms")

    # Verify correctness   
    if Y_gemm.dtype != dtype:
        Y_gemm = Y_gemm.to(dtype)
        
    if torch.allclose(Y_torch, Y_gemm, atol=1e-2):
        print("[SUCCESS]!")
    else:
        print("===== [ERROR] Results do not match! =====")
        max_diff = torch.max(torch.abs(Y_torch - Y_gemm))
        print(f"Max difference: {max_diff.item()}")
        
    print("-" * 50)
    print()

    