import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
from pathlib import Path
from typing import Callable, List
from collections import OrderedDict

sys.path.append(str(Path(__file__).parent))

from model_args import ModelArgs
from module import EncoderBlock
from utils import load_model_state_dict


class CausalLM(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args

        self.n_vocab = args.n_vocab
        self.dim = args.dim
        self.n_layers = args.n_layers
        self.n_heads = args.n_heads
        self.n_kv_heads = (
            args.n_kv_heads if args.n_kv_heads is not None else args.n_heads
        )
        self.d_head = self.dim // self.n_heads
        self.ffn_hidden_dim = args.ffn_hidden_dim
        self.norm_eps = args.norm_eps
        self.max_batch_size = args.max_batch_size
        self.max_seq_len = args.max_seq_len

        self.model = nn.ModuleDict(
            OrderedDict(
                {
                    "embed_tokens": nn.Embedding(self.n_vocab, self.dim),
                    "layers": nn.ModuleList(
                        [EncoderBlock(args) for _ in range(self.n_layers)]
                    ),
                    "norm": nn.RMSNorm(args.dim, eps=args.norm_eps),
                }
            )
        )
        self.lm_head = nn.Linear(self.dim, self.n_vocab, bias=False)

    def forward(self, tokens: torch.Tensor, mask=None) -> torch.Tensor:
        assert tokens.ndim <= 2
        while tokens.ndim < 2:
            tokens = tokens.unsqueeze(0)
        _, L = tokens.shape
        
        if not mask:
            mask = torch.zeros((tokens.shape[0], 1, tokens.shape[1], tokens.shape[1]), dtype=torch.bool)
            mask = torch.triu(torch.ones(tokens.shape[1], tokens.shape[1], device=tokens.device, dtype=torch.bool), diagonal=1)

        h = self.model["embed_tokens"](tokens)  # [B, L] --> [B, L, D]
        
        
        for layer in self.model["layers"]:
            h = layer(h, mask)  # [B, L, D] --> [B, L, D]
        h = self.model["norm"](h)  # [B, L, D] --> [B, L, D]

        logits = self.lm_head(h)  # [B, L, D] --> [B, L, n_vocab]
        return logits

    @staticmethod
    def from_pretrained(
        model_dir: Path | List[Path],
        model_args: ModelArgs,
        strict=True
    ) -> "CausalLM":
        state_dict: OrderedDict = load_model_state_dict(model_dir)
        model = CausalLM(model_args)
        model.load_model_params(state_dict, model_args)
        return model

    def load_model_params(self, state_dict: OrderedDict, model_args: ModelArgs = None):
        """
        Load model parameters from a state dictionary.
        """

        self.lm_head.weight = nn.Parameter(
            state_dict.pop("lm_head.weight")
        )
        
        self.model["embed_tokens"].weight = nn.Parameter(
            state_dict.pop("model.embed_tokens.weight")
        )
        
        self.model["norm"].weight = nn.Parameter(
            state_dict.pop("model.norm.weight")
        )
        
        for i in range(model_args.n_layers):
            layer_key = f"model.layers.{i}."
            layer_state_dict = OrderedDict(
                (k.replace(layer_key, ""), v)
                for k, v in state_dict.items()
                if k.startswith(layer_key)
            )
            if layer_state_dict:
                # input layernorm
                self.model["layers"][i].input_layernorm.weight = nn.Parameter(
                    layer_state_dict.pop("input_layernorm.weight")
                )
                
                # MLP
                self.model["layers"][i].mlp.gate_proj.weight = nn.Parameter(
                    layer_state_dict.pop("mlp.gate_proj.weight")
                )
                self.model["layers"][i].mlp.up_proj.weight = nn.Parameter(
                    layer_state_dict.pop("mlp.up_proj.weight")
                )
                self.model["layers"][i].mlp.down_proj.weight = nn.Parameter(
                    layer_state_dict.pop("mlp.down_proj.weight")
                )
                
                # post attention layernorm
                self.model["layers"][i].post_attention_layernorm.weight = nn.Parameter(
                    layer_state_dict.pop("post_attention_layernorm.weight")
                )
                
                # GQA
                self.model["layers"][i].self_attn.q_proj.weight = nn.Parameter(
                    layer_state_dict.pop("self_attn.q_proj.weight")
                )
                self.model["layers"][i].self_attn.k_proj.weight = nn.Parameter(
                    layer_state_dict.pop("self_attn.k_proj.weight")
                )
                self.model["layers"][i].self_attn.v_proj.weight = nn.Parameter(
                    layer_state_dict.pop("self_attn.v_proj.weight")
                )
                self.model["layers"][i].self_attn.o_proj.weight = nn.Parameter(
                    layer_state_dict.pop("self_attn.o_proj.weight")
                )
                
                # QK norms
                self.model["layers"][i].self_attn.q_norm.weight = nn.Parameter(
                    layer_state_dict.pop("self_attn.q_norm.weight")
                )
                self.model["layers"][i].self_attn.k_norm.weight = nn.Parameter(
                    layer_state_dict.pop("self_attn.k_norm.weight")
                )
            else:
                raise ValueError(f"Layer {i} not found in state_dict.")
        
        
        


if __name__ == "__main__":
    model_args = ModelArgs(
        n_vocab=345,
        dim=128,
        n_layers=6,
        n_heads=8,
        n_kv_heads=None,
        ffn_hidden_dim=256,
        norm_eps=1e-5,
        norm_type="rmsnorm",
        norm_with_affine=True,
        max_batch_size=1,
        max_seq_len=512,
    )

    model = CausalLM(model_args)
    with open("./temp/llama2-dummy.txt", "w") as f:
        for k, v in model.state_dict().items():
            f.write(f"{k}: {[*v.shape]}\n")
