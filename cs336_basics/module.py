import math

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
        half_d_k = int(d_k / 2)
        self.register_buffer(name="cosin_buffer", tensor=torch.empty(size=(max_seq_len, half_d_k)), persistent=False)
        self.register_buffer(name="sin_buffer", tensor=torch.empty(size=(max_seq_len, half_d_k)), persistent=False)
        for i in range(max_seq_len):
            for j in range(half_d_k):
                angle = i / theta ** (j * 2 / d_k)
                self.cosin_buffer[i, j] = math.cos(angle)
                self.sin_buffer[i, j] = math.sin(angle)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[-2]
        d_k = x.shape[-1]
        half_d_k = int(d_k / 2)
        output = torch.empty_like(x)
        for i in range(seq_len):
            for j in range(half_d_k):
                cos = self.cosin_buffer[token_positions[..., i], j]
                sin = self.sin_buffer[token_positions[..., i], j]
                output[..., i, j * 2] = x[..., i, j * 2] * cos - x[..., i, j * 2 + 1] * sin
                output[..., i, j * 2 + 1] = x[..., i, j * 2] * sin + x[..., i, j * 2 + 1] * cos
        return output
