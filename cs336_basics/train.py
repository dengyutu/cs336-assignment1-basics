import argparse
import os
import time

import numpy as np
import numpy.typing as npt
import torch

from cs336_basics.module import AdamW, Transformer_lm
from cs336_basics.utils import (
    cross_entropy,
    data_loading,
    gradient_clipping,
    learning_rate_schedule,
    load_checkpoint,
    save_checkpoint,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a Transformer language model.")

    # --- Model hyperparameters ---
    parser.add_argument("--d_model", type=int, required=True, help="Model dimension")
    parser.add_argument("--num_heads", type=int, required=True, help="Number of attention heads")
    parser.add_argument("--d_ff", type=int, required=True, help="Feed-forward hidden dimension")
    parser.add_argument("--num_layers", type=int, required=True, help="Number of transformer blocks")
    parser.add_argument("--vocab_size", type=int, required=True, help="Vocabulary size")
    parser.add_argument("--context_length", type=int, required=True, help="Maximum sequence / context length")
    parser.add_argument("--theta", type=float, default=10000.0, help="RoPE base frequency")
    parser.add_argument("--rms_norm_eps", type=float, default=1e-5, help="RMSNorm epsilon")

    # --- Optimizer hyperparameters ---
    parser.add_argument("--max_lr", type=float, default=6e-4, help="Peak learning rate")
    parser.add_argument("--min_lr", type=float, default=6e-5, help="Minimum learning rate (after cosine decay)")
    parser.add_argument("--warmup_iters", type=int, default=500, help="Linear warmup iterations")
    parser.add_argument("--cosine_cycle_iters", type=int, default=10000, help="Cosine annealing cycle length")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="AdamW weight decay")
    parser.add_argument("--beta1", type=float, default=0.9, help="AdamW beta1")
    parser.add_argument("--beta2", type=float, default=0.999, help="AdamW beta2")
    parser.add_argument("--adam_eps", type=float, default=1e-8, help="AdamW epsilon")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping max L2 norm")

    # --- Training hyperparameters ---
    parser.add_argument("--batch_size", type=int, required=True, help="Batch size")
    parser.add_argument("--max_iters", type=int, required=True, help="Total training iterations")
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to train on"
    )
    parser.add_argument(
        "--dtype", type=str, default="float32", choices=["float32", "float16", "bfloat16"], help="Parameter dtype"
    )

    # --- Data paths ---
    parser.add_argument(
        "--train_data", type=str, required=True, help="Path to training data (numpy memmap file, dtype=uint16)"
    )
    parser.add_argument(
        "--val_data", type=str, required=True, help="Path to validation data (numpy memmap file, dtype=uint16)"
    )

    # --- Checkpointing ---
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory for saving checkpoints")
    parser.add_argument("--checkpoint_interval", type=int, default=1000, help="Save a checkpoint every N iterations")
    parser.add_argument("--resume_from", type=str, default=None, help="Path to a checkpoint to resume training from")

    # --- Logging ---
    parser.add_argument("--log_interval", type=int, default=50, help="Log training metrics every N iterations")
    parser.add_argument("--val_interval", type=int, default=200, help="Evaluate on validation set every N iterations")
    parser.add_argument(
        "--val_iters", type=int, default=20, help="Number of batches to average over for validation loss"
    )
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="transformer-lm", help="W&B project name")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="W&B run name (default: auto-generated)")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helper: memory-mapped dataset loading
# ---------------------------------------------------------------------------


def load_memmap_dataset(path: str, dtype=np.uint16) -> np.ndarray:
    """Load a dataset as a read-only memory-mapped numpy array.

    This avoids loading the entire file into RAM — only pages that are
    actually accessed are read from disk.
    """
    return np.memmap(path, dtype=dtype, mode="r")


# ---------------------------------------------------------------------------
# Helper: Validate config arguments
# ---------------------------------------------------------------------------


def validate_config(args):
    """Fail fast on invalid or inconsistent hyperparameters."""

    # --- File existence ---
    assert os.path.exists(args.train_data), f"Training data not found: {args.train_data}"
    assert os.path.exists(args.val_data), f"Validation data not found: {args.val_data}"

    # --- Model architecture ---
    assert args.d_model % args.num_heads == 0, (
        f"d_model ({args.d_model}) must be divisible by num_heads ({args.num_heads})"
    )
    assert args.d_model > 0, "d_model must be positive"
    assert args.num_layers > 0, "num_layers must be positive"
    assert args.d_ff > 0, "d_ff must be positive"
    assert args.vocab_size > 0, "vocab_size must be positive"
    assert args.context_length > 0, "context_length must be positive"

    # --- Data vs. model compatibility ---
    # This is commented out because it is a full scan and could take long time for big dataset
    # train_data = load_memmap_dataset(args.train_data)
    # max_token_id = int(train_data.max())
    # assert max_token_id < args.vocab_size, (
    #     f"Training data contains token id {max_token_id}, but vocab_size is only {args.vocab_size}"
    # )
    # assert len(train_data) > args.context_length + 1, (
    #     f"Training data ({len(train_data)} tokens) is too small for context_length={args.context_length}"
    # )
    # del train_data  # Release memmap handle

    # --- Optimizer / schedule ---
    assert args.max_lr > 0, "max_lr must be positive"
    assert args.min_lr >= 0, "min_lr must be non-negative"
    assert args.min_lr <= args.max_lr, f"min_lr ({args.min_lr}) must be <= max_lr ({args.max_lr})"
    assert args.warmup_iters >= 0, "warmup_iters must be non-negative"
    assert args.cosine_cycle_iters >= args.warmup_iters, (
        f"cosine_cycle_iters ({args.cosine_cycle_iters}) must be >= warmup_iters ({args.warmup_iters})"
    )
    assert args.weight_decay >= 0, "weight_decay must be non-negative"
    assert 0.0 <= args.beta1 < 1.0, f"beta1 ({args.beta1}) must be in [0, 1)"
    assert 0.0 <= args.beta2 < 1.0, f"beta2 ({args.beta2}) must be in [0, 1)"
    assert args.adam_eps > 0, "adam_eps must be positive"
    assert args.grad_clip >= 0, "grad_clip must be non-negative"

    # --- Training ---
    assert args.batch_size > 0, "batch_size must be positive"
    assert args.max_iters > 0, "max_iters must be positive"
    assert args.log_interval > 0, "log_interval must be positive"
    assert args.val_interval > 0, "val_interval must be positive"
    assert args.val_iters > 0, "val_iters must be positive"
    assert args.checkpoint_interval > 0, "checkpoint_interval must be positive"

    # --- Device availability ---
    if args.device == "cuda":
        assert torch.cuda.is_available(), "device='cuda' requested but CUDA is not available"
    if args.device == "mps":
        assert torch.backends.mps.is_available(), "device='mps' requested but MPS is not available"

    # --- Checkpoint resume ---
    if args.resume_from is not None:
        assert os.path.exists(args.resume_from), f"Resume checkpoint not found: {args.resume_from}"

    # --- Warnings (non-fatal but suspicious) ---
    if args.warmup_iters == 0:
        print("WARNING: warmup_iters=0, learning rate starts at max immediately")
    if args.cosine_cycle_iters < args.max_iters:
        print(
            f"WARNING: cosine_cycle_iters ({args.cosine_cycle_iters}) < max_iters "
            f"({args.max_iters}), LR will be at min_lr for the last "
            f"{args.max_iters - args.cosine_cycle_iters} iterations"
        )
    if args.batch_size * args.context_length > len(np.memmap(args.train_data, dtype=np.uint16, mode="r")) * 0.1:
        print("WARNING: batch covers >10% of training data per step — consider a smaller batch or more data")

    print("✓ Configuration validated successfully.\n")


# ---------------------------------------------------------------------------
# Helper: estimate loss over multiple batches
# ---------------------------------------------------------------------------


@torch.no_grad()
def estimate_loss(
    model: torch.nn.Module, dataset: npt.NDArray, batch_size: int, context_length: int, device: str, val_iters: int
) -> float:
    model.eval()
    total_loss = 0.0
    for _ in range(val_iters):
        inputs, targets = data_loading(dataset, batch_size, context_length, device)
        logits = model(inputs)
        loss = cross_entropy(logits, targets)
        total_loss += loss.item()
    model.train()
    return total_loss / val_iters


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------


def train():
    args = parse_args()

    # Print config
    print("\n" + "=" * 70)
    print(" CONFIGURATION")
    print("=" * 70)
    for k, v in sorted(vars(args).items()):
        print(f"  {k:.<35} {v}")
    print("=" * 70 + "\n")

    validate_config(args)

    # --- Weights & Biases (optional) ---
    if args.wandb:
        import wandb

        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args),
        )

    # --- Resolve dtype ---
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map[args.dtype]
    device = args.device

    # --- Load datasets via memmap ---
    print(f"Loading training data from: {args.train_data}")
    train_dataset = load_memmap_dataset(args.train_data)
    print(f"  Training tokens: {len(train_dataset):,}")

    print(f"Loading validation data from: {args.val_data}")
    val_dataset = load_memmap_dataset(args.val_data)
    print(f"  Validation tokens: {len(val_dataset):,}")

    # --- Create model ---
    print("Initializing model...")
    model = Transformer_lm(
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        num_layers=args.num_layers,
        theta=args.theta,
        device=device,
        dtype=dtype,
        eps=args.rms_norm_eps,
    )
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {num_params:,}")

    # --- Create optimizer ---
    print("Initializing AdamW optimizer...")
    optimizer = AdamW(
        model.parameters(),
        lr=args.max_lr,
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        weight_decay=args.weight_decay,
    )

    # --- Resume from checkpoint (if provided) ---
    resume_iter = 0
    if args.resume_from is not None:
        print(f"Resuming from checkpoint: {args.resume_from}")
        resume_iter = load_checkpoint(args.resume_from, model, optimizer) + 1
        print(f"  Resumed at iteration {resume_iter}")

    # --- Ensure checkpoint directory exists ---
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # --- Training loop ---
    model.train()
    print(f"\nStarting training for {args.max_iters} iterations...")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Context length: {args.context_length}")
    print(f"  Device: {device} | Dtype: {args.dtype}")
    print("-" * 70)

    t_start = time.time()

    for iteration in range(resume_iter, args.max_iters):
        # --- Update learning rate ---
        lr = learning_rate_schedule(
            it=iteration,
            max_learning_rate=args.max_lr,
            min_learning_rate=args.min_lr,
            warmup_iters=args.warmup_iters,
            cosine_cycle_iters=args.cosine_cycle_iters,
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # --- Forward pass ---
        inputs, targets = data_loading(train_dataset, args.batch_size, args.context_length, device)
        logits = model(inputs)
        loss = cross_entropy(logits, targets)

        # --- Backward pass ---
        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # --- Gradient clipping ---
        if args.grad_clip > 0.0:
            gradient_clipping(model.parameters(), args.grad_clip)

        # --- Optimizer step ---
        optimizer.step()

        # --- Console & W&B logging ---
        if iteration % args.log_interval == 0:
            elapsed = time.time() - t_start
            tokens_per_sec = (iteration - resume_iter + 1) * args.batch_size * args.context_length / max(elapsed, 1e-9)
            log_msg = f"iter {iteration:>6d} | loss {loss.item():.4f} | lr {lr:.2e} | tok/s {tokens_per_sec:.0f}"
            print(log_msg)

            if args.wandb:
                wandb.log(
                    {
                        "train/loss": loss.item(),
                        "train/lr": lr,
                        "train/tokens_per_sec": tokens_per_sec,
                        "iteration": iteration,
                    },
                    step=iteration,
                )

        # --- Validation evaluation ---
        if iteration % args.val_interval == 0 and iteration > 0:
            val_loss = estimate_loss(model, val_dataset, args.batch_size, args.context_length, device, args.val_iters)
            print(f"  >>> val_loss {val_loss:.4f}")

            if args.wandb:
                wandb.log({"val/loss": val_loss, "iteration": iteration}, step=iteration)

        # --- Checkpoint saving ---
        if iteration % args.checkpoint_interval == 0 and iteration > 0:
            ckpt_path = os.path.join(args.checkpoint_dir, f"ckpt_iter_{iteration:06d}.pt")
            save_checkpoint(model, optimizer, iteration, ckpt_path)
            print(f"  >>> Checkpoint saved to {ckpt_path}")

    # --- Final checkpoint ---
    final_path = os.path.join(args.checkpoint_dir, f"ckpt_iter_{args.max_iters:06d}_final.pt")
    save_checkpoint(model, optimizer, args.max_iters, final_path)
    print(f"\nTraining complete. Final checkpoint: {final_path}")

    if args.wandb:
        wandb.finish()


if __name__ == "__main__":
    train()
