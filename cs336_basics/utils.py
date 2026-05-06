import math

import torch
from einops import einsum, reduce


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


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    logits_max = reduce(logits, "... vocab -> ... 1", "max")
    shifted_logits = logits - logits_max
    log_sum_exp = torch.log(reduce(torch.exp(shifted_logits), "... vocab -> ... 1", "sum"))
    target_logits = torch.gather(shifted_logits, -1, targets.unsqueeze(-1))

    loss = log_sum_exp - target_logits
    return reduce(loss, "... 1 -> ", "mean")
