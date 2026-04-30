import torch
from einops import einsum, rearrange
from torch import nn

from cs336_basics.utils import scaled_dot_product_attention


class Linear(nn.Module):
    def __init__(
        self, in_feature: int, out_feature: int, device: torch.device | None = None, dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(size=(out_feature, in_feature), device=device, dtype=dtype))
        std = (2 / (in_feature + out_feature)) ** 0.5
        torch.nn.init.trunc_normal_(self.weight, mean=0, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(self.weight, x, "out_feature in_feature, ... in_feature -> ... out_feature")


class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(size=(num_embeddings, embedding_dim), device=device, dtype=dtype))
        torch.nn.init.trunc_normal_(self.weight, mean=0, std=1, a=-3, b=3)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    def __init__(
        self, d_model: int, eps: float = 1e-5, device: torch.device | None = None, dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size=(d_model,), device=device, dtype=dtype))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # in_dtype = x.dtype
        # x = x.to(torch.float32)
        # x_squared = x**2
        # x_squared_sum = reduce(x_squared, "batch_size sequence_length d_model -> batch_size sequence_length 1", "mean")
        # rms = (x_squared_sum + self.eps) ** 0.5
        # rms_norm = x / rms * self.weight
        # return rms_norm.to(in_dtype)
        in_dtype = x.dtype
        x = x.to(torch.float32)
        x_squared_sum = torch.mean(x**2, dim=-1, keepdim=True)
        rms = torch.rsqrt(x_squared_sum + self.eps)
        rms_norm = x * rms * self.weight
        return rms_norm.to(in_dtype)


class Swiglu(nn.Module):
    def __init__(
        self, d_ff: int = 512, d_model: int = 192, device: torch.device | None = None, dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.w1 = Linear(in_feature=d_model, out_feature=d_ff, device=device, dtype=dtype)
        self.w2 = Linear(in_feature=d_ff, out_feature=d_model, device=device, dtype=dtype)
        self.w3 = Linear(in_feature=d_model, out_feature=d_ff, device=device, dtype=dtype)

    @staticmethod
    def silu(x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        SiLU_output = self.silu(self.w1(x))
        return self.w2(SiLU_output * self.w3(x))


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: torch.device | None = None):
        super().__init__()
        assert d_k % 2 == 0, "d_k must be even"
        freqs = theta ** -(torch.arange(0, d_k, 2, device=device, dtype=torch.float32) / d_k)
        positions = torch.arange(0, max_seq_len, device=device, dtype=torch.float32)
        angle = positions.unsqueeze(1) @ freqs.unsqueeze(0)
        # angles = torch.outer(positions, freqs)
        self.register_buffer(name="cosin_buffer", tensor=torch.cos(angle), persistent=False)
        self.register_buffer(name="sin_buffer", tensor=torch.sin(angle), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        cos = self.cosin_buffer[token_positions]
        sin = self.sin_buffer[token_positions]
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        r_even = x_even * cos - x_odd * sin
        r_odd = x_even * sin + x_odd * cos
        out = torch.stack((r_even, r_odd), dim=-1).flatten(-2)
        return out


class Multihead_self_attention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        RoPE: RotaryPositionalEmbedding | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = self.d_k

        self.W_Q = Linear(in_feature=d_model, out_feature=d_model, device=device, dtype=dtype)
        self.W_K = Linear(in_feature=d_model, out_feature=d_model, device=device, dtype=dtype)
        self.W_V = Linear(in_feature=d_model, out_feature=d_model, device=device, dtype=dtype)
        self.W_O = Linear(in_feature=d_model, out_feature=d_model, device=device, dtype=dtype)
        self.RoPE = RoPE

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        q_proj = self.W_Q(x)
        k_proj = self.W_K(x)
        v_proj = self.W_V(x)
        q_proj_multihead = rearrange(
            q_proj, "batch ... seq_len (head d_k) -> batch ... head seq_len d_k", head=self.num_heads, d_k=self.d_k
        )
        k_proj_multihead = rearrange(
            k_proj, "batch ... seq_len (head d_k) -> batch ... head seq_len d_k", head=self.num_heads, d_k=self.d_k
        )
        v_proj_multihead = rearrange(
            v_proj, "batch ... seq_len (head d_v) -> batch ... head seq_len d_v", head=self.num_heads, d_v=self.d_v
        )
        if self.RoPE is not None:
            q_proj_multihead = self.RoPE(x=q_proj_multihead, token_positions=token_positions)
            k_proj_multihead = self.RoPE(x=k_proj_multihead, token_positions=token_positions)
        seq_len_q = q_proj_multihead.shape[-2]
        seq_len_k = k_proj_multihead.shape[-2]
        causal_mask = torch.ones((seq_len_q, seq_len_k), device=x.device, dtype=torch.bool)
        causal_mask = torch.tril(input=causal_mask, diagonal=0)
        # another way to create the mask: use comparison broadcasts
        # a = torch.arange(seq_len_q).unsqueeze(-1)
        # b = torch.arange(seq_len_k).unsqueeze(0)
        # mask = (a >= b)
        attention_multihead = scaled_dot_product_attention(
            Q=q_proj_multihead, K=k_proj_multihead, V=v_proj_multihead, mask=causal_mask
        )
        attention = rearrange(
            attention_multihead,
            "batch ... head seq_len d_v -> batch ... seq_len (head d_v)",
            head=self.num_heads,
            d_v=self.d_v,
        )
        out = self.W_O(attention)
        return out
