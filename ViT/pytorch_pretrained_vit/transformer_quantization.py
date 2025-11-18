"""
Adapted from https://github.com/lukemelas/simple-bert
"""
 
import numpy as np
import torch
from torch import nn
from torch import Tensor 
from torch.nn import functional as F
import gemm_int8

def quantized_column_matrix_int_symmetric(mat:torch.Tensor):
    """
    Symmetric quantization to int8 on a per-column basis.
    mat: input float tensor (e.g., torch.float32 or torch.bfloat16)
    """
    qmin, qmax = -128, 127
    
    max_vals, _ = torch.max(torch.abs(mat), dim=0, keepdim=True)  # shape (1, M)
    scales = (max_vals / qmax).squeeze(0)  # shape (M,)
    
    q_mat = torch.clamp(torch.round(mat / scales.unsqueeze(0)), qmin, qmax).to(torch.int8)  # shape (N, M)
    
    scales = scales.clone().detach().to(torch.float32)
    return q_mat, scales

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


def quantize_row_matrix_int8_symmetric_batched(mat: torch.Tensor):
    """
    Symmetric per-row quantization for batched 3D tensor.
    mat: [B, N, D]  (float tensor)
    
    Returns:
        q_mat:   [B, N, D] int8
        scales:  [B, N]    float32  (scale per row within each batch)
    """
    qmin, qmax = -128, 127

    # Compute max abs per row (per batch) - Result shape: [B, N, 1]
    max_vals, _ = torch.max(torch.abs(mat), dim=2, keepdim=True)

    # Compute scales per row
    scales = (max_vals / qmax).clamp(min=1e-12)  # avoid div-by-zero, shape [B, N, 1]

    # Quantize
    q_mat = torch.clamp(torch.round(mat / scales), qmin, qmax).to(torch.int8)

    # Return float scales of shape [B, N]
    scales = scales.squeeze(2).to(torch.float32)
    return q_mat, scales

module_quantize_row_matrix_int8_symmetric_batched = torch.compile(quantize_row_matrix_int8_symmetric_batched)

class CustomLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_features))
        else:
            self.bias = None
            
        # Quantization parameters
        self.register_buffer("weight_q", torch.empty_like(self.weight, dtype=torch.int8), persistent=False)
        self.register_buffer("weight_scale", torch.empty(out_features, dtype=torch.float32), persistent=False)
        self.is_quantized = False
        
    def __repr__(self):
        return f"CustomLinear(in_features={self.weight.shape[1]}, out_features={self.weight.shape[0]})"
        
    @torch.no_grad()
    def quantize_weights(self):
        weight_q, weight_scale = quantize_row_int8_symmetric(self.weight)
        self.weight_q.copy_(weight_q)
        self.weight_scale.copy_(weight_scale)
        print(f"[INFO] Done quantize linear layer weights - shape {self.weight_q.shape}")
        self.is_quantized = True

    def forward(self, x):
        if self.is_quantized == False:
            y = torch.matmul(x, self.weight.t())
            if self.bias is not None:
                y = y + self.bias
        else:
            x_q, x_scale = module_quantize_row_matrix_int8_symmetric_batched(x)
            # x_q, x_scale = gemm_int8.batched_quantize_row_int8_symmetric(x)
            
            y = gemm_int8.int8_bmm_matmul_and_quantize(
                x_q, 
                self.weight_q, 
                1.0, 
                x_scale, 
                self.weight_scale)

            if self.bias is not None:
                y = y + self.bias
        return y.to(x.dtype)


def split_last(x, shape):
    "split the last dimension to given shape"
    shape = list(shape)
    assert shape.count(-1) <= 1
    if -1 in shape:
        shape[shape.index(-1)] = int(x.size(-1) / -np.prod(shape))
    return x.view(*x.size()[:-1], *shape)


def merge_last(x, n_dims):
    "merge the last n_dims to a dimension"
    s = x.size()
    assert n_dims > 1 and n_dims < len(s)
    return x.view(*s[:-n_dims], -1)


class MultiHeadedSelfAttention(nn.Module):
    """Multi-Headed Dot Product Attention"""
    def __init__(self, dim, num_heads, dropout):
        super().__init__()
        # self.proj_q = nn.Linear(dim, dim)
        # self.proj_k = nn.Linear(dim, dim)
        # self.proj_v = nn.Linear(dim, dim)
        
        self.proj_q = CustomLinear(dim, dim) # replace with quantized linear
        self.proj_k = CustomLinear(dim, dim) # replace with quantized linear
        self.proj_v = CustomLinear(dim, dim) # replace with quantized linear
        
        self.drop = nn.Dropout(dropout)
        self.n_heads = num_heads
        self.scores = None # for visualization

    def forward(self, x, mask):
        """
        x, q(query), k(key), v(value) : (B(batch_size), S(seq_len), D(dim))
        mask : (B(batch_size) x S(seq_len))
        * split D(dim) into (H(n_heads), W(width of head)) ; D = H * W
        """
        # Overall: (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        
        # Project inputs to multi-head Q, K, V
        q = self.proj_q(x)
        k = self.proj_k(x)
        v = self.proj_v(x)
        
        # Split heads (B, S, D) -> (B, H, S, W)
        q, k, v = (split_last(x, (self.n_heads, -1)).transpose(1, 2) for x in [q, k, v])
        
        # (B, H, S, W) @ (B, H, W, S) -> (B, H, S, S) -softmax-> (B, H, S, S)
        scores = q @ k.transpose(-2, -1) / np.sqrt(k.size(-1))
        if mask is not None:
            mask = mask[:, None, None, :].float()
            scores -= 10000.0 * (1.0 - mask)
        scores = self.drop(F.softmax(scores, dim=-1))
        # (B, H, S, S) @ (B, H, S, W) -> (B, H, S, W) -trans-> (B, S, H, W)
        h = (scores @ v).transpose(1, 2).contiguous()
        # -merge-> (B, S, D)
        h = merge_last(h, 2)
        self.scores = scores
        return h


class PositionWiseFeedForward(nn.Module):
    """FeedForward Neural Networks for each position"""
    def __init__(self, dim, ff_dim):
        super().__init__()
        # self.fc1 = nn.Linear(dim, ff_dim)
        # self.fc2 = nn.Linear(ff_dim, dim)
        
        self.fc1 = CustomLinear(dim, ff_dim) # replace with quantized linear
        self.fc2 = CustomLinear(ff_dim, dim) # replace with quantized linear

    def forward(self, x):
        # (B, S, D) -> (B, S, D_ff) -> (B, S, D)
        return self.fc2(F.gelu(self.fc1(x)))


class Block(nn.Module):
    """Transformer Block"""
    def __init__(self, dim, num_heads, ff_dim, dropout):
        super().__init__()
        self.attn = MultiHeadedSelfAttention(dim, num_heads, dropout)
        self.proj = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.pwff = PositionWiseFeedForward(dim, ff_dim)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask):
        h = self.drop(self.proj(self.attn(self.norm1(x), mask)))
        x = x + h
        h = self.drop(self.pwff(self.norm2(x)))
        x = x + h
        return x


class Transformer(nn.Module):
    """Transformer with Self-Attentive Blocks"""
    def __init__(self, num_layers, dim, num_heads, ff_dim, dropout):
        super().__init__()
        self.blocks = nn.ModuleList([
            Block(dim, num_heads, ff_dim, dropout) for _ in range(num_layers)])

    def forward(self, x, mask=None):
        for block in self.blocks:
            x = block(x, mask)
        return x
