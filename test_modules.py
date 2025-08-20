import torch
from torch import nn
from torch.nn import RMSNorm as CustomRMSNorm
from src.model_args import ModelArgs, get_dtype


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False)
        self.fc2 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False)
        self.fc3 = nn.Linear(cfg["hidden_dim"], cfg["emb_dim"], dtype=cfg["dtype"], bias=False)

    def forward(self, x):
        # * added line
        x = x.to(self.fc1.weight.dtype)  # Ensure input is in the same dtype as the weights
    
        x_fc1 = self.fc1(x)
        x_fc2 = self.fc2(x)
        x = nn.functional.silu(x_fc1) * x_fc2
        return self.fc3(x)


def compute_rope_params(head_dim, theta_base=1_000_000.0, context_length=40_960, dtype=torch.float32):
    assert head_dim % 2 == 0, "Embedding dimension must be even"

    # Compute the inverse frequencies
    inv_freq = 1.0 / (theta_base ** (torch.arange(0, head_dim, 2, dtype=dtype)[: (head_dim // 2)].float() / head_dim))

    # Generate position indices
    positions = torch.arange(context_length, dtype=dtype)

    # Compute the angles
    angles = positions[:, None] * inv_freq[None, :]  # Shape: (context_length, head_dim // 2)

    # Expand angles to match the head_dim
    angles = torch.cat([angles, angles], dim=1)  # Shape: (context_length, head_dim)

    # Precompute sine and cosine
    cos = torch.cos(angles)
    sin = torch.sin(angles)

    return cos, sin


def apply_rope(x, cos, sin):
    # x: (batch_size, num_heads, seq_len, head_dim)
    batch_size, num_heads, seq_len, head_dim = x.shape
    assert head_dim % 2 == 0, "Head dimension must be even"

    # Split x into first half and second half
    x1 = x[..., : head_dim // 2]  # First half
    x2 = x[..., head_dim // 2 :]  # Second half

    # Adjust sin and cos shapes
    cos = cos[:seq_len, :].unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, seq_len, head_dim)
    sin = sin[:seq_len, :].unsqueeze(0).unsqueeze(0)

    # Apply the rotary transformation
    rotated = torch.cat((-x2, x1), dim=-1)
    x_rotated = (x * cos) + (rotated * sin)

    # It's ok to use lower-precision after applying cos and sin rotation
    return x_rotated.to(dtype=x.dtype)


class RMSNorm(nn.Module):
    def __init__(self, emb_dim, eps=1e-6, bias=False, qwen3_compatible=True):
        super().__init__()
        self.eps = eps
        self.qwen3_compatible = qwen3_compatible
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim)) if bias else None

    def forward(self, x):
        input_dtype = x.dtype

        if self.qwen3_compatible:
            x = x.to(torch.float32)

        variance = x.pow(2).mean(dim=-1, keepdim=True)
        norm_x = x * torch.rsqrt(variance + self.eps)
        norm_x = norm_x * self.scale

        if self.shift is not None:
            norm_x = norm_x + self.shift

        return norm_x.to(input_dtype)


class GroupedQueryAttention(nn.Module):
    def __init__(
        self, d_in, num_heads, num_kv_groups, head_dim=None, qk_norm=False, dtype=None
    ):
        super().__init__()
        assert num_heads % num_kv_groups == 0, "num_heads must be divisible by num_kv_groups"

        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        self.group_size = num_heads // num_kv_groups

        if head_dim is None:
            assert d_in % num_heads == 0, "`d_in` must be divisible by `num_heads` if `head_dim` is not set"
            head_dim = d_in // num_heads

        self.head_dim = head_dim
        self.d_out = num_heads * head_dim
        self.dtype = dtype if dtype is not None else torch.float32

        self.W_query = nn.Linear(d_in, self.d_out, bias=False, dtype=dtype)
        self.W_key = nn.Linear(d_in, num_kv_groups * head_dim, bias=False, dtype=dtype)
        self.W_value = nn.Linear(d_in, num_kv_groups * head_dim, bias=False, dtype=dtype)

        self.out_proj = nn.Linear(self.d_out, d_in, bias=False, dtype=dtype)

        if qk_norm:
            self.q_norm = RMSNorm(head_dim, eps=1e-6)
            self.k_norm = RMSNorm(head_dim, eps=1e-6)
        else:
            self.q_norm = self.k_norm = None

    def forward(self, x, mask, cos, sin):
        # * added line
        x = x.to(self.dtype)  # Ensure input is in the same dtype as the
        
        b, num_tokens, d = x.shape
        # * added line
        # x = x.to(self.dtype)  # Ensure input is in the same dtype as the weights

        # Apply projections
        queries = self.W_query(x)  # (b, num_tokens, num_heads * head_dim)
        keys = self.W_key(x)       # (b, num_tokens, num_kv_groups * head_dim)
        values = self.W_value(x)   # (b, num_tokens, num_kv_groups * head_dim)

        # Reshape
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)
        values = values.view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)

        # Optional normalization
        if self.q_norm:
            queries = self.q_norm(queries)
        if self.k_norm:
            keys = self.k_norm(keys)

        # Apply RoPE
        queries = apply_rope(queries, cos, sin)
        keys = apply_rope(keys, cos, sin)

        # Expand K and V to match number of heads
        keys = keys.repeat_interleave(self.group_size, dim=1)
        values = values.repeat_interleave(self.group_size, dim=1)

        # Attention
        attn_scores = queries @ keys.transpose(2, 3)
        attn_scores = attn_scores.masked_fill(mask, -torch.inf)
        attn_weights = torch.softmax(attn_scores / self.head_dim**0.5, dim=-1)

        context = (attn_weights @ values).transpose(1, 2).reshape(b, num_tokens, self.d_out)
        return self.out_proj(context)



class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = GroupedQueryAttention(
            d_in=cfg["emb_dim"],
            num_heads=cfg["n_heads"],
            head_dim=cfg["head_dim"],
            num_kv_groups=cfg["n_kv_groups"],
            qk_norm=cfg["qk_norm"],
            dtype=cfg["dtype"]
        )
        self.ff = FeedForward(cfg)
        self.norm1 = RMSNorm(cfg["emb_dim"], eps=1e-6)
        self.norm2 = RMSNorm(cfg["emb_dim"], eps=1e-6)
        self.dtype = cfg["dtype"]

    def forward(self, x, mask, cos, sin):
        x = x.to(self.dtype)  # Ensure input is in the same dtype as the weights
        
        # Shortcut connection for attention block
        shortcut = x
        x = self.norm1(x)
        x = self.att(x, mask, cos, sin)  # Shape [batch_size, num_tokens, emb_size]
        x = x + shortcut  # Add the original input back

        # Shortcut connection for feed-forward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = x + shortcut  # Add the original input back

        return x



class Qwen3Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        # Main model parameters
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"], dtype=cfg["dtype"])

        self.trf_blocks = nn.ModuleList(  # ModuleList since Sequential can only accept one input, and we need `x, mask, cos, sin`
            [TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        self.final_norm = RMSNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False, dtype=cfg["dtype"])

        # Reusuable utilities
        if cfg["head_dim"] is None:
            head_dim = cfg["emb_dim"] // cfg["n_heads"]
        else:
            head_dim = cfg["head_dim"]
        cos, sin = compute_rope_params(
            head_dim=head_dim,
            theta_base=cfg["rope_base"],
            context_length=cfg["context_length"]
        )
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.cfg = cfg


    def forward(self, in_idx):
        # Forward pass
        tok_embeds = self.tok_emb(in_idx)
        x = tok_embeds
        
        # * added line
        x = x.to(self.cfg["dtype"])

        num_tokens = x.shape[1]
        mask = torch.triu(torch.ones(num_tokens, num_tokens, device=x.device, dtype=torch.bool), diagonal=1)
        
        for block in self.trf_blocks:
            x = block(x, mask, self.cos, self.sin)
        x = self.final_norm(x).to(self.cfg["dtype"])
        
        # * Ensure the output head is in the same dtype as the input
        # logits = self.out_head(x.to(self.cfg["dtype"]))
        logits = self.out_head(x)
        return logits



def test_rope():
    from src.rope import Rotary, ModelArgs

    model_args = ModelArgs(
        llm_type="qwen",
        n_vocab=151_936,
        dim=1024,
        n_layers=28,
        d_head=128,
        n_heads=16,
        n_kv_heads=8,
        qk_rms_norm=False,
        ffn_hidden_dim=3072,
        norm_eps=1e-6,
        rope_theta=1_000_000.0,
        max_batch_size=2,
        max_seq_len=40_960,  # 32768,
        d_type="f16"
    )

    rotary = Rotary(model_args)

    context_length = model_args.max_seq_len
    cos, sin = compute_rope_params(model_args.d_head, theta_base=model_args.rope_theta, context_length=context_length, dtype=get_dtype(model_args.d_type))

    x = torch.randn(2, 8, model_args.max_seq_len, model_args.d_head, dtype=get_dtype(model_args.d_type))  # (batch_size, num_heads, seq_len, head_dim)
    
    x_rotated = apply_rope(x, cos, sin)
    custom_rotated = rotary.forward(x)

    assert x_rotated.shape == custom_rotated.shape, f"Shape mismatch: {x_rotated.shape} != {custom_rotated.shape}"

    mean_diff = torch.mean(torch.abs(x_rotated - custom_rotated)).item()
    assert torch.allclose(x_rotated, custom_rotated, atol=1e-5), f"Values mismatch between x_rotated and custom_rotated. Mean difference: {mean_diff}"
    
    print("RoPE test ... OK")


def test_rms_norm():

    emb_dim = 64
    eps = 1e-6
    x = torch.randn(2, 10, emb_dim)

    rms_norm = RMSNorm(emb_dim, eps=eps)
    output = rms_norm(x)
    
    custom_rms_norm = CustomRMSNorm(emb_dim, eps=eps)
    output_custom = custom_rms_norm(x)

    assert output.shape == output_custom.shape, f"Shape mismatch: {output.shape} != {output_custom.shape}"
    assert torch.allclose(output, output_custom, atol=1e-5), "Output values do not match between custom RMSNorm and PyTorch RMSNorm"
    
    print("RMSNorm test ... OK")


def test_grouped_query_attention():
    from src.utils import ModelArgs

    model_args = ModelArgs(
        llm_type="qwen",
        n_vocab=151_936,
        dim=1024,
        n_layers=28,
        d_head=128,
        n_heads=16,
        n_kv_heads=8,
        qk_rms_norm=False,
        ffn_hidden_dim=3072,
        norm_eps=1e-6,
        rope_theta=1_000_000.0,
        max_batch_size=2,
        max_seq_len=40_960,  # 32768,
        d_type="f16"
    )

    b, num_tokens, d_in = 2, 10, model_args.dim
    x = torch.randn(b, num_tokens, d_in, dtype=get_dtype(model_args.d_type))

    cos, sin = compute_rope_params(model_args.d_head, theta_base=model_args.rope_theta, context_length=model_args.max_seq_len, dtype=get_dtype(model_args.d_type))
    
    mask = torch.zeros((b, 1, num_tokens, num_tokens), dtype=torch.bool)

    attention = GroupedQueryAttention(
        d_in=model_args.dim,
        num_heads=model_args.n_heads,
        num_kv_groups=model_args.n_kv_heads,
        head_dim=model_args.d_head,
        dtype=get_dtype(model_args.d_type)
    )

    
    
    from src.module import SelfAttention
    
    self_attention = SelfAttention(model_args)
    
    
    assert attention.W_query.weight.shape == self_attention.q_proj.weight.shape, "Query projection weights do not match"
    assert attention.W_key.weight.shape == self_attention.k_proj.weight.shape, "Key projection weights do not match"
    assert attention.W_value.weight.shape == self_attention.v_proj.weight.shape, "Value projection weights do not match"
    assert attention.out_proj.weight.shape == self_attention.o_proj.weight.shape, "Output projection weights do not match"
    
    # setting synch weights
    attention.W_query.weight.data = self_attention.q_proj.weight.data
    attention.W_key.weight.data = self_attention.k_proj.weight.data
    attention.W_value.weight.data = self_attention.v_proj.weight.data
    attention.out_proj.weight.data = self_attention.o_proj.weight.data
    
    output_self_attn = attention(x, mask, cos, sin)
    custom_output_self_attn = self_attention(x, mask)
    
    assert output_self_attn.shape == custom_output_self_attn.shape, f"Shape mismatch: {output_self_attn.shape} != {custom_output_self_attn.shape}"
    
    print(f"Self-attn Output mean difference: {torch.mean(torch.abs(output_self_attn - custom_output_self_attn)).item()}")
    print(f"Self-attn Output max difference: {torch.max(torch.abs(output_self_attn - custom_output_self_attn)).item()}")
    print(f"Self-attn Output min difference: {torch.min(torch.abs(output_self_attn - custom_output_self_attn)).item()}")
    print(f"Self-attn Output std difference: {torch.std(torch.abs(output_self_attn - custom_output_self_attn)).item()}")
    
    assert torch.allclose(output_self_attn, custom_output_self_attn, atol=1e-5), "Output values do not match between GroupedQueryAttention and SelfAttention"
    
    print("GroupedQueryAttention test ... OK")
    

def test_gqa_rms_norm():
    from src.utils import ModelArgs
    from src.model_args import get_dtype

    model_args = ModelArgs(
        llm_type="qwen",
        n_vocab=151_936,
        dim=1024,
        n_layers=28,
        d_head=128,
        n_heads=16,
        n_kv_heads=8,
        qk_rms_norm=True,
        ffn_hidden_dim=3072,
        norm_eps=1e-6,
        rope_theta=1_000_000.0,
        max_batch_size=2,
        max_seq_len=40_960,  # 32768,
        d_type="f16"
    )

    b, num_tokens, d_in = 2, 10, model_args.dim
    x = torch.randn(b, num_tokens, d_in, dtype=get_dtype(model_args.d_type))

    cos, sin = compute_rope_params(model_args.d_head, context_length=model_args.max_seq_len)

    mask = torch.zeros((b, 1, num_tokens, num_tokens), dtype=torch.bool)

    attention = GroupedQueryAttention(
        d_in=model_args.dim,
        num_heads=model_args.n_heads,
        num_kv_groups=model_args.n_kv_heads,
        head_dim=model_args.d_head,
        qk_norm=model_args.qk_rms_norm,
        dtype=get_dtype(model_args.d_type)
    )

    
    
    from src.module import SelfAttention
    
    self_attention = SelfAttention(model_args)
    
    assert attention.W_query.weight.shape == self_attention.q_proj.weight.shape, "Query projection weights do not match"
    assert attention.W_key.weight.shape == self_attention.k_proj.weight.shape, "Key projection weights do not match"
    assert attention.W_value.weight.shape == self_attention.v_proj.weight.shape, "Value projection weights do not match"
    assert attention.out_proj.weight.shape == self_attention.o_proj.weight.shape, "Output projection weights do not match"
    
    # setting synch weights
    attention.W_query.weight.data = self_attention.q_proj.weight.data
    attention.W_key.weight.data = self_attention.k_proj.weight.data
    attention.W_value.weight.data = self_attention.v_proj.weight.data
    attention.out_proj.weight.data = self_attention.o_proj.weight.data
    
    output_self_attn = attention(x, mask, cos, sin)
    custom_output_self_attn = self_attention(x, mask)
    
    assert output_self_attn.shape == custom_output_self_attn.shape, f"Shape mismatch: {output_self_attn.shape} != {custom_output_self_attn.shape}"
    
    print(f"Self-attn Output mean difference: {torch.mean(torch.abs(output_self_attn - custom_output_self_attn)).item()}")
    print(f"Self-attn Output max difference: {torch.max(torch.abs(output_self_attn - custom_output_self_attn)).item()}")
    print(f"Self-attn Output min difference: {torch.min(torch.abs(output_self_attn - custom_output_self_attn)).item()}")
    print(f"Self-attn Output std difference: {torch.std(torch.abs(output_self_attn - custom_output_self_attn)).item()}")
    
    assert torch.allclose(output_self_attn, custom_output_self_attn, atol=1e-5), "Output values do not match between GroupedQueryAttention and SelfAttention"
    
    print("GroupedQueryAttention RMSNorm test ... OK")


def test_mlp():
    from src.utils import ModelArgs

    model_args = ModelArgs(
        llm_type="qwen",
        n_vocab=151_936,
        dim=1024,
        n_layers=28,
        d_head=128,
        n_heads=16,
        n_kv_heads=8,
        qk_rms_norm=True,
        ffn_hidden_dim=3072,
        norm_eps=1e-6,
        rope_theta=1_000_000.0,
        max_batch_size=2,
        max_seq_len=40_960,  # 32768,
        d_type="f16"
    )
    
    cfg = {
        "emb_dim": model_args.dim,
        "hidden_dim": model_args.ffn_hidden_dim,
        "dtype": get_dtype(model_args.d_type)  # Use the utility function to get the correct dtype
    }
    
    mlp = FeedForward(cfg)
    
    from src.module import FeedForward as customMLP
    
    custom_mlp = customMLP(model_args)
    
    assert mlp.fc1.weight.shape == custom_mlp.gate_proj.weight.shape, "Gate projection weights do not match"
    assert mlp.fc2.weight.shape == custom_mlp.up_proj.weight.shape, "Up projection weights do not match"
    assert mlp.fc3.weight.shape == custom_mlp.down_proj.weight.shape, "Down projection weights do not match"
    
    # setting synch weights
    mlp.fc1.weight.data = custom_mlp.gate_proj.weight.data
    mlp.fc2.weight.data = custom_mlp.up_proj.weight.data
    mlp.fc3.weight.data = custom_mlp.down_proj.weight.data
    
    
    x = torch.randn(2, 10, model_args.dim, dtype=get_dtype(model_args.d_type))  # (batch_size, num_tokens, emb_dim)
    output_mlp = mlp(x)
    custom_output_mlp = custom_mlp(x)
    
    assert output_mlp.shape == custom_output_mlp.shape, f"Shape mismatch: {output_mlp.shape} != {custom_output_mlp.shape}"
    assert torch.allclose(output_mlp, custom_output_mlp, atol=1e-5), "Output values do not match between FeedForward and custom FeedForward"
    
    print("FeedForward test ... OK")



def test_block():
    from src.utils import ModelArgs

    model_args = ModelArgs(
        llm_type="qwen",
        n_vocab=151_936,
        dim=1024,
        n_layers=28,
        d_head=128,
        n_heads=16,
        n_kv_heads=8,
        qk_rms_norm=True,
        ffn_hidden_dim=3072,
        norm_eps=1e-6,
        rope_theta=1_000_000.0,
        max_batch_size=2,
        max_seq_len=40_960,  # 32768,
        d_type="f16"
    )
    
    cfg = {
        "emb_dim": model_args.dim,
        "n_heads": model_args.n_heads,
        "n_kv_groups": model_args.n_kv_heads,
        "head_dim": model_args.d_head,
        "qk_norm": model_args.qk_rms_norm,
        "hidden_dim": model_args.ffn_hidden_dim,
        "dtype": get_dtype(model_args.d_type),  # Use the utility function to get the correct dtype
    }

    b, num_tokens, d_in = 2, 10, model_args.dim
    x = torch.randn(b, num_tokens, d_in)  # (batch_size, num_tokens, emb_dim)

    cos, sin = compute_rope_params(model_args.d_head, context_length=model_args.max_seq_len, dtype=get_dtype(model_args.d_type))

    mask = torch.zeros((b, 1, num_tokens, num_tokens), dtype=torch.bool)

    block = TransformerBlock(cfg)
    
    from src.module import EncoderBlock
    
    custom_block = EncoderBlock(model_args)
    
    
    # synch weights
    block.att.W_query.weight.data = custom_block.self_attn.q_proj.weight.data
    block.att.W_key.weight.data = custom_block.self_attn.k_proj.weight.data
    block.att.W_value.weight.data = custom_block.self_attn.v_proj.weight.data
    block.att.out_proj.weight.data = custom_block.self_attn.o_proj.weight.data
    block.ff.fc1.weight.data = custom_block.mlp.gate_proj.weight.data
    block.ff.fc2.weight.data = custom_block.mlp.up_proj.weight.data
    block.ff.fc3.weight.data = custom_block.mlp.down_proj.weight.data
    
    
    output_block = block(x, mask, cos, sin)
    custom_block_output_block = custom_block(x, mask)
    
    assert output_block.shape == custom_block_output_block.shape, f"Shape mismatch: {output_block.shape} != {custom_block_output_block.shape}"
    assert torch.allclose(output_block, custom_block_output_block, atol=1e-5), "Output values do not match between TransformerBlock and custom EncoderBlock"    

    print("TransformerBlock test ... OK")



def test_causal_lm():
    from src.causal_model import CausalLM
    from src.utils import ModelArgs
    
    model_args = ModelArgs(
        llm_type="qwen",
        n_vocab=151_936,
        dim=1024,
        n_layers=28,
        d_head=128,
        n_heads=16,
        n_kv_heads=8,
        qk_rms_norm=True,
        ffn_hidden_dim=3072,
        norm_eps=1e-6,
        rope_theta=1_000_000.0,
        max_batch_size=2,
        max_seq_len=40_960,  # 32768,
        d_type="f16"
    )
    
    cfg = {
        "vocab_size": 151_936,           # Vocabulary size
        "context_length": 40_960,        # Context length that was used to train the model
        "emb_dim": 1024,                 # Embedding dimension"
        "n_heads": 16,                   # Number of attention heads
        "n_layers": 28,                  # Number of layers
        "hidden_dim": 3072,              # Size of the intermediate dimension in FeedForward
        "head_dim": 128,                 # Size of the heads in GQA
        "qk_norm": True,                 # Whether to normalize queries and values in GQA
        "n_kv_groups": 8,                # Key-Value groups for grouped-query attention
        "rope_base": 1_000_000.0,        # The base in RoPE's "theta"
        "dtype": get_dtype(model_args.d_type),         # Lower-precision dtype to reduce memory usage
    }
    
    gt_model = Qwen3Model(cfg)
    
    custom_model = CausalLM(model_args)
    
    # Synch weights
    gt_model.tok_emb.weight.data = custom_model.model.embed_tokens.weight.data
    for i in range(model_args.n_layers):
        gt_model.trf_blocks[i].att.W_query.weight.data = custom_model.model.layers[i].self_attn.q_proj.weight.data
        gt_model.trf_blocks[i].att.W_key.weight.data = custom_model.model.layers[i].self_attn.k_proj.weight.data
        gt_model.trf_blocks[i].att.W_value.weight.data = custom_model.model.layers[i].self_attn.v_proj.weight.data
        gt_model.trf_blocks[i].att.out_proj.weight.data = custom_model.model.layers[i].self_attn.o_proj.weight.data
        gt_model.trf_blocks[i].ff.fc1.weight.data = custom_model.model.layers[i].mlp.gate_proj.weight.data
        gt_model.trf_blocks[i].ff.fc2.weight.data = custom_model.model.layers[i].mlp.up_proj.weight.data
        gt_model.trf_blocks[i].ff.fc3.weight.data = custom_model.model.layers[i].mlp.down_proj.weight.data
    gt_model.out_head.weight.data = custom_model.lm_head.weight.data

    
    num_tokens = 10
    x = torch.randint(0, model_args.n_vocab, (model_args.max_batch_size, num_tokens))  # (batch_size, num_tokens)

    logits = gt_model(x)
    custom_logits = custom_model(x)
    
    assert logits.shape == custom_logits.shape, f"Shape mismatch: {logits.shape} != {custom_logits.shape}"
    
    print(f"Logits mean difference: {torch.mean(torch.abs(logits - custom_logits)).item()}")
    print(f"Logits max difference: {torch.max(torch.abs(logits - custom_logits)).item()}")
    print(f"Logits min difference: {torch.min(torch.abs(logits - custom_logits)).item()}")
    print(f"Logits std difference: {torch.std(torch.abs(logits - custom_logits)).item()}")  
    
    assert torch.allclose(logits, custom_logits, atol=1e-5), "Output values do not match between Qwen3Model and CausalLM"
    
    print(f"Full Model test ... OK")


if __name__ == "__main__":
    test_rope()
    test_rms_norm()
    test_grouped_query_attention()
    test_gqa_rms_norm()
    test_mlp()
    test_block()
    test_causal_lm()