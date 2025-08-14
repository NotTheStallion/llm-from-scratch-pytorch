import torch
from torch import nn
from torch.nn import RMSNorm as CustomRMSNorm


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False)
        self.fc2 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False)
        self.fc3 = nn.Linear(cfg["hidden_dim"], cfg["emb_dim"], dtype=cfg["dtype"], bias=False)

    def forward(self, x):
        x_fc1 = self.fc1(x)
        x_fc2 = self.fc2(x)
        x = nn.functional.silu(x_fc1) * x_fc2
        return self.fc3(x)


def compute_rope_params(head_dim, theta_base=10_000, context_length=4096, dtype=torch.float32):
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
        b, num_tokens, d = x.shape

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



def test_rope():
    from src.rope import Rotary, ModelArgs

    model_args = ModelArgs(
        llm_type="qwen",
        n_vocab=32000,
        dim=128,
        n_layers=4,
        n_heads=8,
        n_kv_heads=4,
        ffn_hidden_dim=256,
        max_batch_size=1,
        max_seq_len=4096,
    )

    rotary = Rotary(model_args, base=10_000)

    head_dim = model_args.dim // model_args.n_heads
    context_length = model_args.max_seq_len
    cos, sin = compute_rope_params(head_dim, context_length=context_length)
    

    x = torch.randn(2, 8, 256, head_dim)  # (batch_size, num_heads, seq_len, head_dim)
    
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
        n_vocab=32000,
        dim=128,
        n_layers=4,
        n_heads=8,
        n_kv_heads=2,
        ffn_hidden_dim=256,
        max_batch_size=1,
        max_seq_len=4096,
    )

    b, num_tokens, d_in = 2, 10, model_args.dim
    x = torch.randn(b, num_tokens, d_in)

    cos, sin = compute_rope_params(model_args.dim // model_args.n_heads, context_length=model_args.max_seq_len)

    mask = torch.zeros((b, 1, num_tokens, num_tokens), dtype=torch.bool)

    attention = GroupedQueryAttention(
        d_in=model_args.dim,
        num_heads=model_args.n_heads,
        num_kv_groups=model_args.n_kv_heads,
        head_dim=model_args.dim // model_args.n_heads
    )

    
    
    from src.module import SelfAttention
    
    self_attention = SelfAttention(model_args)
    
    # print(f"GroupedQueryAttention output shape: {output_self_attn.shape}")
    # print(f"SelfAttention output shape: {custom_output_self_attn.shape}")
    
    
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

    model_args = ModelArgs(
        llm_type="qwen",
        n_vocab=32000,
        dim=128,
        n_layers=4,
        n_heads=8,
        n_kv_heads=2,
        qk_rms_norm=True,  # @param Enable QK RMSNorm
        ffn_hidden_dim=256,
        max_batch_size=1,
        max_seq_len=4096,
    )

    b, num_tokens, d_in = 2, 10, model_args.dim
    x = torch.randn(b, num_tokens, d_in)

    cos, sin = compute_rope_params(model_args.dim // model_args.n_heads, context_length=model_args.max_seq_len)

    mask = torch.zeros((b, 1, num_tokens, num_tokens), dtype=torch.bool)

    attention = GroupedQueryAttention(
        d_in=model_args.dim,
        num_heads=model_args.n_heads,
        num_kv_groups=model_args.n_kv_heads,
        head_dim=model_args.dim // model_args.n_heads,
        qk_norm=model_args.qk_rms_norm
    )

    
    
    from src.module import SelfAttention
    
    self_attention = SelfAttention(model_args)
    
    # print(f"GroupedQueryAttention output shape: {output_self_attn.shape}")
    # print(f"SelfAttention output shape: {custom_output_self_attn.shape}")
    
    
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
        n_vocab=32000,
        dim=128,
        n_layers=4,
        n_heads=8,
        n_kv_heads=2,
        ffn_hidden_dim=256,
        max_batch_size=1,
        max_seq_len=4096,
    )
    
    cfg = {
        "emb_dim": model_args.dim,
        "hidden_dim": model_args.ffn_hidden_dim,
        "dtype": torch.float32
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
    
    
    x = torch.randn(2, 10, model_args.dim)  # (batch_size, num_tokens, emb_dim)
    output_mlp = mlp(x)
    custom_output_mlp = custom_mlp(x)
    
    assert output_mlp.shape == custom_output_mlp.shape, f"Shape mismatch: {output_mlp.shape} != {custom_output_mlp.shape}"
    assert torch.allclose(output_mlp, custom_output_mlp, atol=1e-5), "Output values do not match between FeedForward and custom FeedForward"
    
    print("FeedForward test ... OK")


if __name__ == "__main__":
    test_rope()
    test_rms_norm()
    test_grouped_query_attention()
    test_gqa_rms_norm()
    test_mlp()