from typing import Literal, Optional
from pydantic import BaseModel
import torch


class ModelArgs(BaseModel):
    llm_type: Literal["llama", "phi", "qwen", "gemma"] = "qwen"
    # basic
    dim: int = -1
    n_vocab: int = -1
    n_layers: int = -1
    wte_tying: bool = False
    # attention
    d_head: Optional[int] = None  # None means d_head = dim // n_heads
    n_heads: int = -1
    n_kv_heads: Optional[int] = None  # None means n_kv_heads = n_heads
    qk_rms_norm: bool = False # whether to apply RMSNorm to Q and K
    # mlp
    ffn_hidden_dim: int = -1
    # norm
    norm_eps: float = 1e-5
    # position embedding
    rope_theta: float = 10000.0
    rope_partial_factor: Optional[float] = None
    # other
    max_batch_size: int = 1
    max_seq_len: int = 2048
    d_type : Literal["f16", "f32", "bf16"] = "f32"  # Data type for model parameters




def get_dtype(dtype: str) -> torch.dtype:
    if dtype == "f16":
        return torch.float16
    elif dtype == "f32":
        return torch.float32
    elif dtype == "bf16":
        return torch.bfloat16
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")