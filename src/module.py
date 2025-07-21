import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.normalization import RMSNorm

import sys
from typing import Tuple
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from attention import scaled_dot_product_attention
from rope import RoPE, positional_encoding, Rotary
from model_args import ModelArgs

class EncoderBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args

        self.self_attn = SelfAttention(args)
        self.mlp = FeedForward(args)
        self.input_layernorm = nn.RMSNorm(args.dim, eps=args.norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x: torch.Tensor, start_index: int) -> torch.Tensor:
        # [B, L, D] --> [B, L, D]
        x = x + self.self_attn(self.input_layernorm(x), start_index)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class SelfAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.n_heads, self.n_kv_heads = args.n_heads, args.n_kv_heads
        if self.n_kv_heads is None:
            self.n_kv_heads = self.n_heads
        self.d_head = args.dim // args.n_heads

        self.q_proj = nn.Linear(args.dim, self.n_heads * self.d_head, bias=True)
        self.k_proj = nn.Linear(args.dim, self.n_kv_heads * self.d_head, bias=True)
        self.v_proj = nn.Linear(args.dim, self.n_kv_heads * self.d_head, bias=True)
        self.o_proj = nn.Linear(self.n_heads * self.d_head, args.dim, bias=False)
        
        self.rope = RoPE(args)
        self.custom_rope = Rotary(args)

    def forward(self, x: torch.Tensor, start_index: int) -> torch.Tensor:
        # x: [B, L, D]
        B, L, D = x.shape
        # pe = positional_encoding(L, D)
        # x += pe.unsqueeze(0)

        # [B, L, D] --> [B, L, D]
        q: torch.Tensor = self.q_proj(x)
        # [B, L, D] --> [B, L, D_kv], D_kv may smaller than D
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
        
        # RoPE
        # q = self.rope(q, 0)
        # k = self.rope(k, 0)
        q = self.custom_rope(q)
        k = self.custom_rope(k)

        # --> [B, n_heads, L, d_head], if query seq length == 1,
        # set is_causal to False to avoid attention mask construction to save computation
        # output = scaled_dot_product_attention_gqa(q, k, v, is_causal=q.shape[-2] > 1)
        output = scaled_dot_product_attention(q, k, v, is_causal=True)
        # [B, n_heads, L, d_head] --> [B, L, n_heads, d_head] --> [B, L, D]
        output = output.permute(0, 2, 1, 3).reshape(B, L, -1)

        # [B, L, D] --> [B, L, D]
        return self.o_proj(output)


class FeedForward(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.llm_type = args.llm_type
        self.dim = args.dim
        self.hidden_dim = args.ffn_hidden_dim

        self.gate_proj = nn.Linear(self.dim, self.hidden_dim, bias=False)
        self.down_proj = nn.Linear(self.hidden_dim, self.dim, bias=False)
        self.up_proj = nn.Linear(self.dim, self.hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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