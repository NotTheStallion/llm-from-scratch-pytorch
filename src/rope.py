import torch
import torch.nn as nn

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from model_args import ModelArgs
import numpy as np

def positional_encoding(L, d, n=10000):
    pe = torch.zeros((L, d))
    for pos in range(L):
        for i in range(d // 2):
            denominator = torch.pow(torch.tensor(n, dtype=torch.float32), 2 * i / d)
            pe[pos, 2 * i] = torch.sin(pos / denominator)
            pe[pos, 2 * i + 1] = torch.cos(pos / denominator)
    return pe

class Rotary(nn.Module):

  def __init__(self, args: ModelArgs, dtype=torch.float32):

    super().__init__()
    self.base = args.rope_theta
    self.max_seq_len = args.max_seq_len
    self.dim = args.dim // args.n_heads
    self.rope_dim = self.dim
    self.cos_cached = None
    self.sin_cached = None
    
    self._build_cache(dtype=dtype)

  def _build_cache(self, dtype=torch.float32):
    seq_len = self.max_seq_len
    theta = 1. / (self.base ** (torch.arange(0, self.rope_dim, 2, dtype=dtype).float() / self.rope_dim)) # THETA = 10,000^(-2*i/d) or 1/10,000^(2i/d)
    pos = torch.arange(seq_len).float() #Position Index -> [0,1,2...seq-1]
    angles = pos[:, None] * theta[None, :]  # (max_seq_len, head_dim // 2)
    angles = torch.cat([angles, angles], dim=1)  # (max_seq_len, head_dim)

    # Precompute sine and cosine
    self.cos_cached = torch.cos(angles).unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, max_seq_len, head_dim)
    self.sin_cached = torch.sin(angles).unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, max_seq_len, head_dim)
    
    

  def _neg_half(self, x: torch.Tensor):
    d_2 = self.rope_dim // 2
    return torch.cat([-x[:, :, :, d_2:], x[:, :, :, :d_2]], dim=-1) # [x_1, x_2,...x_d] -> [-x_d/2, ... -x_d, x_1, ... x_d/2]

  def forward(self, x: torch.Tensor):
    # x: [B, n_heads, L, d_head]
    # x: [B, n_kv_heads, L_kv, d_head]
    
    if x.shape[0] > self.max_seq_len:
      raise ValueError(f"Sequence length {x.shape[0]} exceeds maximum supported length {self.max_seq_len}.")
    
    neg_half_x = self._neg_half(x)
    x_rope = (x * self.cos_cached[:, :, :x.shape[2], :]) + (neg_half_x * self.sin_cached[:, :, :x.shape[2], :])  # [x_1*cosTHETA_1 - x_d/2*sinTHETA_d/2, ....]
    return x_rope



if __name__ == "__main__":
    torch.random.manual_seed(0)

    args = ModelArgs(dim=128, n_heads=4, max_seq_len=15)

    # [B, L, H, D]
    x = torch.randn(1, 10, 4, 128 // 4)
    
    y = torch.randn(1, 15, 128)
    
    
    rot = Rotary(args)
    result = rot(x)
    print(result.shape)  # Should be [1, 10, 4, 128 // 4]
    
