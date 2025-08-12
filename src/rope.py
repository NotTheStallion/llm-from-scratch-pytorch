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

  def __init__(self, args: ModelArgs, base: int = 10_000):

    super().__init__()
    self.base = base
    self.dim = args.dim // args.n_heads
    self.rope_dim = self.dim
    # print(f"dim: {self.dim}, rope_dim: {self.rope_dim}, n_heads: {args.n_heads}")
    self.cos_cached = None
    self.sin_cached = None

  def _build_cache(self, x: torch.Tensor):
    if self.cos_cached is not None and x.shape[0] <= self.cos_cached.shape[0]:
      return

    seq_len = x.shape[0]
    theta = 1. / (self.base ** (torch.arange(0, self.rope_dim, 2).float() / self.rope_dim)).to(x.device) # THETA = 10,000^(-2*i/d) or 1/10,000^(2i/d)
    seq_idx = torch.arange(seq_len, device=x.device).float().to(x.device) #Position Index -> [0,1,2...seq-1]
    idx_theta = torch.einsum('n,d->nd', seq_idx, theta)  #Calculates m*(THETA) = [ [0, 0...], [THETA_1, THETA_2...THETA_d/2], ... [seq-1*(THETA_1), seq-1*(THETA_2)...] ]
    idx_theta2 = torch.cat([idx_theta, idx_theta], dim=1) # [THETA_1, THETA_2...THETA_d/2] -> [THETA_1, THETA_2...THETA_d]

    self.cos_cached = idx_theta2.cos()[:, None, None, :] #Cache [cosTHETA_1, cosTHETA_2...cosTHETA_d]
    self.sin_cached = idx_theta2.sin()[:, None, None, :] #cache [sinTHETA_1, sinTHETA_2...sinTHETA_d]

  def _neg_half(self, x: torch.Tensor):
    d_2 = self.rope_dim // 2
    return torch.cat([-x[:, :, :, d_2:], x[:, :, :, :d_2]], dim=-1) # [x_1, x_2,...x_d] -> [-x_d/2, ... -x_d, x_1, ... x_d/2]

  def forward(self, x: torch.Tensor):
    # x: [B, L, H, D]
    self._build_cache(x)
    neg_half_x = self._neg_half(x) # 
    x_rope = (x * self.cos_cached[:x.shape[0]]) + (neg_half_x * self.sin_cached[:x.shape[0]]) # [x_1*cosTHETA_1 - x_d/2*sinTHETA_d/2, ....]
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
    
