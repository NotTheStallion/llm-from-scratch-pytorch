import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.normalization import RMSNorm

import sys
from typing import Tuple
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from attention import scaled_dot_product_attention
from rope import positional_encoding, Rotary
from model_args import ModelArgs, get_dtype

class EncoderBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args

        self.self_attn = SelfAttention(args)
        self.mlp = FeedForward(args)
        self.input_layernorm = nn.RMSNorm(args.dim, eps=args.norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x: torch.Tensor, mask) -> torch.Tensor:
        # [B, L, D] --> [B, L, D]
        x = x + self.self_attn(self.input_layernorm(x), mask)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class SelfAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.n_heads, self.n_kv_heads, self.d_head = args.n_heads, args.n_kv_heads, args.d_head
        if self.n_kv_heads is None:
            self.n_kv_heads = self.n_heads
        if self.d_head is None:
            self.d_head = args.dim // args.n_heads

        self.q_proj = nn.Linear(args.dim, self.n_heads * self.d_head, bias=False)
        self.k_proj = nn.Linear(args.dim, self.n_kv_heads * self.d_head, bias=False)
        self.v_proj = nn.Linear(args.dim, self.n_kv_heads * self.d_head, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.d_head, args.dim, bias=False)
        
        self.custom_rope = Rotary(args)
        
        self.qk_rms_norm = args.qk_rms_norm
        
        if self.qk_rms_norm:
            self.q_norm = RMSNorm(self.d_head, eps=args.norm_eps)
            self.k_norm = RMSNorm(self.d_head, eps=args.norm_eps)

    def forward(self, x: torch.Tensor, mask: int) -> torch.Tensor:
        # x: [B, L, D]
        B, L, D = x.shape
        # pe = positional_encoding(L, D)
        # x += pe.unsqueeze(0)

        # [B, L, D] --> [B, L, D]
        q: torch.Tensor = self.q_proj(x)
        # [B, L, D] --> [B, L, D_kv], D_kv may smaller than D (MQA or GQA)
        k: torch.Tensor = self.k_proj(x)
        v: torch.Tensor = self.v_proj(x)

        # [B, L, D] --> [B, L, n_heads, d_head]
        q = q.view(B, L, self.n_heads, self.d_head)
        k = k.view(B, L, self.n_kv_heads, self.d_head)
        v = v.view(B, L, self.n_kv_heads, self.d_head)

        # [B, L, n_heads, d_head] --> [B, n_heads, L, d_head]
        q = q.permute(0, 2, 1, 3).contiguous()
        # [B, L_kv, n_kv_heads, d_head] --> [B, n_kv_heads, L_kv, d_head]
        k = k.permute(0, 2, 1, 3).contiguous()
        v = v.permute(0, 2, 1, 3).contiguous()
        
        # Qk rms_norm
        if self.qk_rms_norm:
            # ! The two lines below cause a 1e-5 error in test [ununderstandable]
            # q = self.q_norm(q)
            # k = self.k_norm(k)
            q = RMSNorm(self.d_head, eps=1e-6)(q)
            k = RMSNorm(self.d_head, eps=1e-6)(k)
        
        # RoPE
        q = self.custom_rope(q) # (B, n_heads, L, d_head)
        k = self.custom_rope(k) # (B, n_kv_heads, L_kv, d_head)
        
        # repeat k and v if n_heads != n_kv_heads
        k = k.repeat_interleave(self.n_heads//self.n_kv_heads, dim=1) # (B, n_heads, L_kv, d_head)
        v = v.repeat_interleave(self.n_heads//self.n_kv_heads, dim=1) # (B, n_heads, L_kv, d_head)

        # Attention softmax(QK^T / sqrt(d_head))V
        attn_scores = q @ k.transpose(2, 3)
        attn_scores = attn_scores.masked_fill(mask, -torch.inf)
        attn_weights = torch.softmax(attn_scores / self.d_head**0.5, dim=-1)

        # * temp fix
        attn_weights = attn_weights.to(v.dtype)  # Ensure attn_weights matches v dtype
        output = (attn_weights @ v).transpose(1, 2).reshape(B, L, self.n_heads * self.d_head)

        # [B, L, D] --> [B, L, D]
        return self.o_proj(output)


class FeedForward(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.llm_type = args.llm_type
        self.dim = args.dim
        self.hidden_dim = args.ffn_hidden_dim
        
        self.tensor_type = get_dtype(args.d_type)

        self.gate_proj = nn.Linear(self.dim, self.hidden_dim, bias=False, dtype=self.tensor_type)
        self.down_proj = nn.Linear(self.hidden_dim, self.dim, bias=False, dtype=self.tensor_type)
        self.up_proj = nn.Linear(self.dim, self.hidden_dim, bias=False, dtype=self.tensor_type)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.tensor_type)
        
        # [B, L, D] --> [B, L, hD]
        x1, x2 = F.silu(self.gate_proj(x)), self.up_proj(x)
        # [B, L, hD] --> [B, L, D]
        return self.down_proj(x1 * x2)



if __name__ == "__main__":
    model_args = ModelArgs(
        n_vocab=32,
        dim=64,
        n_layers=4,
        n_heads=8,
        n_kv_heads=None,
        norm_eps=1e-5,
        ffn_hidden_dim=128,
        max_batch_size=1,
        max_seq_len=256,
    )
    # rope = RoPE(model_args)
    # rope(torch.randn(1, 4, 8, 64))


    x = torch.rand(1, 60, 64)
    
    x_norm_torch = nn.RMSNorm(model_args.dim, eps=model_args.norm_eps)
    print(x_norm_torch(x).shape)