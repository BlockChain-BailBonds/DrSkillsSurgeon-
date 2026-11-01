"""Small, transparent MLP with explicit intervention sites."""
from __future__ import annotations

import torch
from torch import nn

from .tasks import INPUT_VOCAB, OUTPUT_VOCAB, LENGTH


class SkillMLP(nn.Module):
    """Independent model per task/seed; not an actual pretrained LLM.

    Sites hidden1 and hidden2 are post-ReLU vectors. `patch` replaces selected
    coordinates with clean (or corrupted) activation values from the *same*
    model. Full-site patching is an exact architectural smoke test.
    """

    def __init__(self, embedding: int = 12, hidden: int = 64):
        super().__init__()
        self.embed = nn.Embedding(INPUT_VOCAB, embedding)
        self.fc1 = nn.Linear(LENGTH * embedding, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.head = nn.Linear(hidden, LENGTH * OUTPUT_VOCAB)

    @staticmethod
    def _edit(h: torch.Tensor, site: str, patch: dict | None):
        if patch is None or patch["site"] != site:
            return h
        replacement = patch["values"]
        if replacement.shape != h.shape or replacement.device != h.device:
            raise ValueError("patch values must match hidden activation shape and device")
        indices = patch.get("indices")
        if indices is None:
            return replacement
        if indices.dtype != torch.long or indices.ndim != 1:
            raise ValueError("patch indices must be a 1D long tensor")
        if bool((indices < 0).any()) or bool((indices >= h.shape[1]).any()):
            raise ValueError("patch index out of range")
        edited = h.clone()
        edited[:, indices] = replacement[:, indices]
        return edited

    def forward(self, x: torch.Tensor, *, patch: dict | None = None, capture: bool = False):
        if x.ndim != 2 or x.shape[1] != LENGTH:
            raise ValueError(f"expected token array [batch, {LENGTH}]")
        h1 = torch.relu(self.fc1(self.embed(x).flatten(1)))
        h1 = self._edit(h1, "hidden1", patch)
        h2 = torch.relu(self.fc2(h1))
        h2 = self._edit(h2, "hidden2", patch)
        logits = self.head(h2).reshape(-1, LENGTH, OUTPUT_VOCAB)
        if capture:
            return logits, {"hidden1": h1.detach(), "hidden2": h2.detach()}
        return logits
