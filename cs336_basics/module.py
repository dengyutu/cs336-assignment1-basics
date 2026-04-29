import torch
from einops import einsum
from torch import nn


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
