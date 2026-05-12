import math
import os
import typing
from collections.abc import Iterable

import numpy as np
import numpy.typing as npt
import torch
from einops import einsum, reduce

from cs336_basics.tokenizer import Tokenizer


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


def learning_rate_schedule(
    it: int, max_learning_rate: float, min_learning_rate: float, warmup_iters: int, cosine_cycle_iters: int
):
    if it < warmup_iters:
        return it / warmup_iters * max_learning_rate
    elif it <= cosine_cycle_iters:
        return min_learning_rate + 0.5 * (
            1 + math.cos((it - warmup_iters) / (cosine_cycle_iters - warmup_iters) * math.pi)
        ) * (max_learning_rate - min_learning_rate)
    else:
        return min_learning_rate


def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    eps = 1e-6
    # total_norm = torch.sqrt(sum(p.grad.data.norm() ** 2 for p in parameters if p.grad is not None))
    total_norm = torch.sqrt(sum(torch.sum(p.grad.data**2) for p in parameters if p.grad is not None))
    if total_norm > max_l2_norm:
        for p in parameters:
            scale = max_l2_norm / (total_norm + eps)
            if p.grad is not None:
                p.grad.data.mul_(scale)
    # eps = 1e-6
    # grad_l2_norm = torch.tensor(0, dtype=torch.float32, device=parameters[0].device)
    # for p in parameters:
    #     if p.grad is not None:
    #         grad_l2_norm += reduce((p.grad.data) ** 2, "... -> ", "sum")
    # grad_l2_norm = grad_l2_norm.sqrt()
    # if grad_l2_norm > max_l2_norm:
    #     for p in parameters:
    #         scale = max_l2_norm / (grad_l2_norm + eps)
    #         if p.grad is not None:
    #             p.grad.data.mul_(scale)


def data_loading(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    max_start = len(dataset) - context_length - 1
    start_indices = np.random.randint(0, max_start + 1, size=batch_size)
    inputs = np.stack([dataset[i : i + context_length] for i in start_indices])
    targets = np.stack([dataset[i + 1 : i + 1 + context_length] for i in start_indices])
    inputs = torch.tensor(inputs, dtype=torch.long, device=device)
    targets = torch.tensor(targets, dtype=torch.long, device=device)
    return (inputs, targets)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
) -> None:
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iteration": iteration,
    }
    torch.save(checkpoint, out)


def load_checkpoint(
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    checkpoint = torch.load(src, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["iteration"]


def top_p_filtering(token_logits: torch.Tensor, top_p: float):
    sorted_logits, sorted_indices = torch.sort(token_logits, dim=-1, descending=True)
    probs = softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(probs, dim=-1)
    mask = (cumulative_probs - probs) >= top_p
    sorted_logits = sorted_logits.masked_fill(mask, float("-inf"))
    logits = sorted_logits.scatter(dim=-1, index=sorted_indices, src=sorted_logits)
    return logits


# this function assumes the input is one string.
def decoding(
    model: torch.nn.Module,
    prompt: str,
    tokenizer: Tokenizer,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    device: str,
) -> torch.Tensor:

    input_ids = tokenizer.encode(prompt)
    # the [input_ids] is to set the input's the batch dim as 1.
    input_ids = torch.tensor([input_ids], dtype=torch.long, device=device)

    eos_token_id = tokenizer.encode("<|endoftext|>")[0]

    model.eval()
    generated = input_ids

    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(generated)
            next_token_logits = logits[:, -1, :]

            if temperature == 0.0:
                next_token = next_token_logits.argmax(dim=-1, keepdim=True)
            else:
                next_token_logits = next_token_logits / temperature

                if 0.0 < top_p < 1.0:
                    next_token_logits = top_p_filtering(next_token_logits, top_p)

                probs = softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            # Stop BEFORE appending — we don't want <|endoftext|> in the output
            if next_token.item() == eos_token_id:
                break

            # append the new token
            generated = torch.cat((generated, next_token), dim=-1)

    output_text = tokenizer.decode(generated[0].tolist())
    return output_text


# this function assumes the input is batch of tokens.
def decoding_batch(
    model: torch.nn.Module,
    prompt_ids: torch.Tensor,
    tokenizer: Tokenizer,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    device: str,
) -> torch.Tensor:

    eos_token_id = tokenizer.encode("<|endoftext|>")[0]
    batch_size = prompt_ids.shape[0]

    model.eval()
    generated = prompt_ids.clone().to(device)

    with torch.no_grad():
        finished = torch.zeros((batch_size, 1), device=device, dtype=torch.bool)
        for _ in range(max_new_tokens):
            logits = model(generated)
            next_token_logits = logits[:, -1, :]

            if temperature == 0.0:
                next_token = next_token_logits.argmax(dim=-1, keepdim=True)
            else:
                next_token_logits = next_token_logits / temperature

                if 0.0 < top_p < 1.0:
                    next_token_logits = top_p_filtering(next_token_logits, top_p)

                probs = softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            next_token = next_token.masked_fill(finished, eos_token_id)
            finished = finished | (next_token == eos_token_id)

            # append the new token
            generated = torch.cat((generated, next_token), dim=-1)

            if torch.all(finished):
                break

    return generated
