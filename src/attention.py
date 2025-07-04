import math
import torch
import torch.nn.functional as F


# Implementation taken from Torch : https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0,
        is_causal=False, scale=None, enable_gqa=False) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias = attn_mask + attn_bias

    if enable_gqa:
        key = key.repeat_interleave(query.size(-3)//key.size(-3), -3)
        value = value.repeat_interleave(query.size(-3)//value.size(-3), -3)

    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value



if __name__ == "__main__":

    ## group query attention, n_heads != n_kv_heads
    k = torch.randn(1, 2, 10, 10)
    v = torch.eye(10).unsqueeze(0).unsqueeze(0).repeat(1, k.shape[1], 1, 1)

    def test2(q, k, v):
        """is_causal has no effect on result if query seq length == 1"""
        assert q.shape[-2] == 1
        o1 = scaled_dot_product_attention(q, k, v, is_causal=True)
        o2 = scaled_dot_product_attention(q, k, v, is_casual=False)
        assert torch.allclose(o1, o2)

    q = torch.randn(1, 6, 1, 10)
    test2(q, k, v)
    q = torch.randn(1, 2, 1, 10)
    test2(q, k, v)
