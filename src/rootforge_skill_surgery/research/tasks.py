"""Deterministic, synthetic sequence transformations with exact targets."""
from __future__ import annotations

import torch

TASKS = ("copy", "reverse", "sort", "count_gt8")
LENGTH = 8
INPUT_VOCAB = 16
OUTPUT_VOCAB = 17
PAD = 16


def targets(task: str, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return target tokens and a Boolean loss/evaluation mask.

    count_gt8 predicts one scalar at position 0; other positions are ignored.
    This prevents the padded positions from inflating reported accuracy.
    """
    if task not in TASKS:
        raise ValueError(f"unknown task {task!r}; expected one of {TASKS}")
    if x.ndim != 2 or x.shape[1] != LENGTH:
        raise ValueError(f"input must have shape [batch, {LENGTH}]")
    if task == "copy":
        y = x.clone()
    elif task == "reverse":
        y = torch.flip(x, dims=(1,))
    elif task == "sort":
        y = torch.sort(x, dim=1).values
    else:
        y = torch.full_like(x, PAD)
        y[:, 0] = (x > 8).sum(dim=1)
    mask = torch.ones_like(y, dtype=torch.bool)
    if task == "count_gt8":
        mask[:, 1:] = False
    return y, mask


def batch(task: str, size: int, generator: torch.Generator, device: torch.device):
    if size < 1:
        raise ValueError("batch size must be positive")
    # Sample on CPU from an explicitly seeded Generator even on GPU runs.
    x = torch.randint(0, INPUT_VOCAB, (size, LENGTH), generator=generator)
    x = x.to(device)
    y, mask = targets(task, x)
    return x, y, mask


def fixed_inputs(size: int = 192, seed: int = 14791) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randint(0, INPUT_VOCAB, (size, LENGTH), generator=g)
