import math

import torch
from einops import einsum


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    input_max = torch.max(x, dim, keepdim=True).values
    input_exp = torch.exp(x - input_max)
    return input_exp / torch.sum(input_exp, dim=dim, keepdim=True)


def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor | None = None):
    d_k = Q.shape[-1]
    scores = einsum(
        Q, K, "batch_size ... seq_len_q d_k, batch_size ... seq_len_k d_k -> batch_size ... seq_len_q seq_len_k"
    )
    scores = scores / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(~mask, -math.inf)
    attention_weights = softmax(scores, -1)
    attention_weights = attention_weights.nan_to_num(0.0)
    return einsum(
        attention_weights,
        V,
        "batch_size ... seq_len_q seq_len_k, batch_size ... seq_len_k d_v -> batch_size ... seq_len_q d_v",
    )
